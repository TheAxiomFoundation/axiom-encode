"""Citation cleanup cannot merge separate source-stated thresholds."""

import functools
import re

import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as vp
from tests.test_source_completeness_bare_markers import CITATION, _source_rule


def _colliding_source(*, before=10, between=0, after=0, value=50):
    citation = "See section 9. "
    first = (
        f"The payment is at most {value} dollars, calculated as income divided by 2."
    )
    second = f"The resource limit is at least {value} dollars."
    unpadded = first + " " + citation * between + second
    distance = unpadded.rindex(str(value)) - unpadded.index(str(value))
    # Each complete reference loses `section 9` during numeric recall cleanup.
    # Arrange an exact collision between the resource's cleaned coordinate and
    # the payment formula's raw coordinate, while keeping the source distinct.
    removed = len("section 9") * (before + between)
    padding = removed - distance
    assert padding >= 0
    source = (
        "a\n"
        + citation * before
        + first
        + " " * (padding + 1)
        + citation * between
        + second
        + " "
        + citation * after
        + "\nb\nThe person must be resident.\n"
    )
    return source, first, second


def _threshold_obligations(source, value):
    assert sc.bare_outline(source) is not None
    branches = sc.recognize_source_structure(source)
    formulas = sc._source_formula_branches(
        source, branches=branches, active_branches=branches, deferred_paths=set()
    )
    extract = functools.partial(
        vp.extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
    )
    obligations = sc._source_boundary_obligations(
        branches,
        narrative_formula_branches=formulas,
        extract_numeric_occurrences=extract,
        source_text=source,
    )
    limits = []
    for branch, boundary in obligations:
        if boundary.value != value:
            continue
        interval, _ = sc._source_interval_and_polarity_for_boundary(
            branch, boundary, extract_numeric_occurrences=extract
        )
        assert interval is not None
        limits.append(
            (
                None if interval.lower is None else interval.lower.value,
                None if interval.upper is None else interval.upper.value,
            )
        )
    return limits


def test_collector_keeps_distinct_thresholds_after_citation_cleanup():
    source, _, _ = _colliding_source()
    assert (source.index("50"), source.rindex("50")) == (175, 265)
    assert sorted(_threshold_obligations(source, 50), key=repr) == sorted(
        [(None, 50), (50, None)], key=repr
    )


def test_validator_legacy_profile_requires_independent_resource_boundary(tmp_path):
    source, first, second = _colliding_source()
    assert vp._numeric_profile_for_citation_path(CITATION) == "legacy"
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {"name": "income", "dtype": "Money"},
            {"name": "resources", "dtype": "Money"},
            {"name": "resident", "dtype": "Bool"},
        ],
        "rules": [
            _source_rule(
                "payment",
                "if income <= income_limit: income / divisor else: 0",
                first,
                f"{CITATION}(a)",
                dtype="Money",
            ),
            _source_rule(
                "residence_requirement",
                "resident",
                "The person must be resident.",
                f"{CITATION}(b)",
            ),
        ],
    }
    for name, number, proof in (
        ("income_limit", 50, "at most 50 dollars"),
        ("resource_minimum", 50, second),
        ("divisor", 2, "income divided by 2"),
    ):
        rule = _source_rule(name, str(number), proof, f"{CITATION}(a)", dtype="Money")
        rule["kind"] = "parameter"
        payload["rules"].append(rule)
    cases = [
        {
            "period": "2026-07-01",
            "input": {"income": income, "resources": 0, "resident": True},
            "output": {
                "payment": income / 2 if income <= 50 else 0,
                "residence_requirement": True,
            },
        }
        for income in (0, 50, 99, 100, 101)
    ]
    pipeline = vp.ValidatorPipeline(
        policy_repo_path=tmp_path / "nonexistent-rulespec",
        axiom_rules_path=tmp_path / "nonexistent-engine",
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
    )
    issues = pipeline._complete_source_unit_issues(
        yaml.safe_dump(payload),
        validation_source_texts={CITATION: source},
        proof_source_texts={CITATION: source},
        test_cases=cases,
    )
    assert issues == [
        "[complete-source-unit:tests] Companion tests do not exercise every "
        "source-stated boundary input; missing: (a)=50."
    ]


@settings(max_examples=80, deadline=None, derandomize=True)
@given(
    positions=st.lists(st.sampled_from(("before", "between", "after")), max_size=20),
    value=st.integers(min_value=10, max_value=999_999),
)
def test_equal_thresholds_at_distinct_raw_positions_never_merge(positions, value):
    between = positions.count("between")
    source, _, _ = _colliding_source(
        before=10 + positions.count("before") + between,
        between=between,
        after=positions.count("after"),
        value=value,
    )
    assert source.index(str(value)) != source.rindex(str(value))
    assert sorted(_threshold_obligations(source, value), key=repr) == sorted(
        [(None, value), (value, None)], key=repr
    )


def test_numeric_recall_tracks_both_thresholds_to_exact_raw_spans():
    source, _, _ = _colliding_source()
    source_map = sc._NumericRecallSourceMap(list(range(len(source))))
    cleaned = sc.authoritative_numeric_recall_text(source, _source_map=source_map)
    thresholds = [
        item
        for item in vp.extract_typed_numeric_inventory_occurrences_from_text(
            cleaned, profile="legacy"
        )
        if item.value == 50
    ]
    assert len(thresholds) == 2
    assert [source_map.source_span(item.start, item.end) for item in thresholds] == [
        (175, 177),
        (265, 267),
    ]


def test_rewritten_numeric_character_has_no_certified_source_span():
    source = "The limit is 50 dollars; another value is 50 dollars."
    source_map = sc._NumericRecallSourceMap(list(range(len(source))))
    cleaned = source_map.sub(r"(?<=is )50(?= dollars;)", "51", source)
    rewritten_start = cleaned.index("51")
    unchanged_start = cleaned.index("50")
    assert source_map.source_span(rewritten_start, rewritten_start + 2) is None
    assert source_map.source_span(unchanged_start, unchanged_start + 2) == (
        source.rindex("50"),
        source.rindex("50") + 2,
    )


def test_numeric_token_joined_across_citation_has_no_contiguous_source_span():
    source = "The limit is 5section 9" + "0 dollars."
    source_map = sc._NumericRecallSourceMap(list(range(len(source))))
    cleaned = source_map.sub(r"section 9", "", source)
    assert "50 dollars" in cleaned
    start = cleaned.index("50")
    assert source_map.source_span(start, start + 1) == (source.index("5"), start + 1)
    assert source_map.source_span(start + 1, start + 2) == (
        source.index("0"),
        source.index("0") + 1,
    )
    assert source_map.source_span(start, start + 2) is None


def test_uncertified_numeric_origins_keep_both_boundary_obligations(monkeypatch):
    source = (
        "a\nThe payment is at most 50 dollars, calculated as income divided by 2;\n"
        "b\nThe person must be resident.\n"
    )
    original = sc.authoritative_numeric_recall_text

    def uncertain_cleanup(text, **kwargs):
        cleaned = original(text, **kwargs)
        source_map = kwargs.get("_source_map")
        if source_map is not None:
            for match in re.finditer("50", cleaned):
                source_map.offsets[match.start() : match.end()] = [None, None]
        return cleaned

    monkeypatch.setattr(sc, "authoritative_numeric_recall_text", uncertain_cleanup)
    assert _threshold_obligations(source, 50) == [(None, 50), (None, 50)]


def test_citation_inside_formula_first_line_deduplicates_one_physical_threshold():
    source = (
        "a\nThe payment, under section 9, is at most 50 dollars, "
        "calculated as income divided by 2;\n"
        "b\nThe person must be resident.\n"
    )
    assert _threshold_obligations(source, 50) == [(None, 50)]
