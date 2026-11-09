import ast

import pytest

from axiom_encode.harness.signed_condition_paths import signed_granting_paths
from axiom_encode.harness.source_completeness import (
    _formula_interval_intersection,
    _parse_formula_expression,
)


def read(formula, *, versions=None, limit=32):
    versions = versions or {}

    def resolve(name, start, end):
        if name in versions:
            return tuple(
                (_parse_formula_expression(value), lo, hi)
                for value, lo, hi in versions[name]
            )
        return None

    return signed_granting_paths(
        _parse_formula_expression(formula),
        start="2026-08-01",
        end=None,
        resolve=resolve,
        intersect=_formula_interval_intersection,
        limit=limit,
        depth_limit=16,
        numeric_names=frozenset({"count"}),
    )


def positives(path):
    return {fact.name for fact in path.facts if fact.positive}


def test_false_or_prefix_is_not_a_positive_granting_fact():
    paths = read("(occupies and billed and employed) or (moves and billed)")
    assert paths
    assert any(positives(p) == {"moves", "billed"} for p in paths)
    assert any(
        any(f.name == "occupies" and not f.positive for f in p.facts) for p in paths
    )
    assert all(
        {"occupies", "billed", "employed"} <= positives(p)
        or {"moves", "billed"} <= positives(p)
        for p in paths
    )


def test_false_prefix_cannot_license_an_unconditional_moved_arm():
    paths = read("(occupies and billed and employed) or moves")
    assert paths
    assert any("billed" not in positives(p) for p in paths)


def test_signed_contradictions_are_unreachable():
    assert read("a and not a") == ()
    assert read("False and a") == ()
    assert all({"b"} <= positives(p) for p in read("(a and not a) or b"))


def test_comparisons_retain_exact_operator_threshold_and_polarity():
    paths = read("count > 1 and billed")
    assert any(
        f.name == "count" and f.relation == "gt" and f.threshold == 1 and f.positive
        for f in paths[0].facts
    )
    paths = read("not (count > 1) and billed")
    assert any(
        f.name == "count" and f.relation == "gt" and not f.positive
        for f in paths[0].facts
    )


def test_derived_versions_keep_temporal_intersections():
    paths = read(
        "status and billed",
        versions={
            "status": [
                ("moves", "2026-08-01", "2026-12-31"),
                ("employed and occupies", "2027-01-01", None),
            ]
        },
    )
    assert {(p.start, p.end) for p in paths} == {
        ("2026-08-01", "2026-12-31"),
        ("2027-01-01", None),
    }


def test_unknown_syntax_cycles_and_path_caps_fail_closed():
    assert read("unknown(a)") is None
    assert read("a + b") is None
    assert read("count > other") is None
    assert (
        read("a", versions={"a": [("b", None, None)], "b": [("a", None, None)]}) is None
    )
    assert read("(a or b or c) and (d or e or f)", limit=2) is None


def test_reader_uses_ast_not_textual_operator_substitution():
    assert isinstance(_parse_formula_expression("a and b"), ast.BoolOp)
    assert read("holds(a) and not_holds(b)")[0].facts


@pytest.mark.parametrize(
    "versions",
    [
        [("a", "2026-09-01", None)],
        [("a", "2026-08-01", "2026-12-31")],
        [("a", "2026-08-01", "2026-12-31"), ("b", "2027-02-01", None)],
        [("a", "2026-08-01", "2027-02-01"), ("b", "2027-01-01", None)],
    ],
)
def test_version_start_middle_end_gaps_and_overlap_fail_closed(versions):
    assert read("status", versions={"status": versions}) is None


def test_fully_covered_constant_false_alias_has_no_granting_paths():
    assert read("status", versions={"status": [("False", "2026-08-01", None)]}) == ()
