"""Adversarial v33 delta/replay examples; hashes identify fixture blob states."""

from copy import deepcopy

import pytest

from axiom_encode.notary.canonical import jcs_dumps, sha256_hex
from axiom_encode.notary.coverage import Coverage, compute_coverage
from axiom_encode.notary.lineage import EligibleRecord
from axiom_encode.notary.refusal import Refusal

from .lineage_fixtures import correction, generation, policy


def entry(path, value, mode="100644"):
    return (path, mode, sha256_hex(value.encode()))


def transition(path, before, after, *, before_mode="100644", after_mode="100644"):
    return {
        "path": path,
        "before_blob_sha256": sha256_hex(before.encode())
        if before is not None
        else None,
        "before_mode": before_mode if before is not None else None,
        "after_blob_sha256": sha256_hex(after.encode()) if after is not None else None,
        "after_mode": after_mode if after is not None else None,
        "patch_note_sha256": None,
    }


def record(*transitions, body=None, label="draw"):
    body = deepcopy(body or generation())
    if "draw_set_id" in body:
        body["draw_set_id"] = label
    body["transitions"] = sorted(transitions, key=lambda t: t["path"].encode())
    raw = jcs_dumps(body)
    return EligibleRecord(sha256_hex(raw), raw)


def check(base, subject, *records):
    return compute_coverage(sorted(base), sorted(subject), records, policy())


@pytest.mark.parametrize("before,after", [(None, "a"), ("a", None), ("a", "b")])
def test_add_delete_modify(before, after):
    r = record(transition("rules/x", before, after))
    result = check(
        [] if before is None else [entry("rules/x", before)],
        [] if after is None else [entry("rules/x", after)],
        r,
    )
    assert result == Coverage((("rules/x", (r.body_sha256,)),), (), ())


def test_projection_excludes_new_lineage_and_unprotected_changes():
    r = record(transition("rules/x", None, "a"))
    result = check(
        [],
        [
            entry("rules/x", "a"),
            entry(".axiom/lineage/new.json", "body"),
            entry("README", "text"),
        ],
        r,
    )
    assert isinstance(result, Coverage)
    assert result.unprotected_changes == (".axiom/lineage/new.json", "README")


def test_missing_coverage_precedes_discontinuous_chain_globally():
    result = check(
        [],
        [entry("rules/a", "end"), entry("rules/z", "end")],
        record(transition("rules/a", "wrong", "end")),
    )
    assert result == Refusal(
        "uncovered-path", "rules/z", "no eligible transition covers this path"
    )


@pytest.mark.parametrize(
    "before,after", [("wrong", "end"), (None, "wrong"), ("wrong", "wrong")]
)
def test_endpoint_mismatch(before, after):
    assert (
        check(
            [], [entry("rules/a", "end")], record(transition("rules/a", before, after))
        ).code
        == "inconsistent-chain"
    )


def test_partial_record_and_unchanged_path_cannot_be_consumed():
    r = record(transition("rules/x", None, "a"), transition("rules/y", None, "b"))
    assert check([], [entry("rules/x", "a")], r).code == "inconsistent-chain"


def test_whole_record_across_two_paths():
    r = record(transition("rules/x", None, "a"), transition("rules/y", None, "b"))
    result = check([], [entry("rules/x", "a"), entry("rules/y", "b")], r)
    assert isinstance(result, Coverage)
    assert len(result.assignment) == 2


def test_duplicate_coverage_is_ambiguous():
    records = [
        record(transition("rules/x", None, "a"), label=label)
        for label in ("first", "second")
    ]
    assert check([], [entry("rules/x", "a")], *records).code == "ambiguous-assignment"


def test_split_versus_direct_is_ambiguous():
    records = [
        record(transition("rules/x", before, after))
        for before, after in [(None, "a"), ("a", "b"), (None, "b")]
    ]
    assert check([], [entry("rules/x", "b")], *records).code == "ambiguous-assignment"


def test_dead_end_retry_is_unused():
    good, dead = [record(transition("rules/x", None, value)) for value in ("a", "dead")]
    result = check([], [entry("rules/x", "a")], good, dead)
    assert isinstance(result, Coverage)
    assert result.unused_eligible_records == (dead.body_sha256,)


def test_cross_path_cycle():
    x = record(transition("rules/a", None, "one"), transition("rules/b", "one", "two"))
    y = record(transition("rules/a", "one", "two"), transition("rules/b", None, "one"))
    assert (
        check([], [entry("rules/a", "two"), entry("rules/b", "two")], x, y).code
        == "record-cycle"
    )


def test_file_directory_replacement_tries_other_topological_orders():
    remove = record(transition("rules/x", "file", None))
    add = record(transition("rules/x/child", None, "child"))
    result = check(
        [entry("rules/x", "file")], [entry("rules/x/child", "child")], add, remove
    )
    assert isinstance(result, Coverage)


def test_atomic_file_directory_replacement():
    swap = record(
        transition("rules/x", "file", None), transition("rules/x/child", None, "child")
    )
    assert isinstance(
        check([entry("rules/x", "file")], [entry("rules/x/child", "child")], swap),
        Coverage,
    )


def test_nonrealizable_intermediate_tree():
    # Record A creates a child and advances y; B must follow A to delete the
    # parent. Every per-path endpoint works, but A's intermediate tree cannot.
    a = record(
        transition("rules/x/child", None, "child"),
        transition("rules/y", "start", "mid"),
    )
    b = record(transition("rules/x", "file", None), transition("rules/y", "mid", "end"))
    result = check(
        [entry("rules/x", "file"), entry("rules/y", "start")],
        [entry("rules/x/child", "child"), entry("rules/y", "end")],
        a,
        b,
    )
    assert result.code == "no-valid-execution"


@pytest.mark.parametrize("predecessor", [None, "correct", "f" * 64])
def test_correction_predecessor(predecessor):
    first = record(transition("rules/x", None, "a"))
    body = correction()
    body["predecessor_record_sha256"] = (
        first.body_sha256 if predecessor == "correct" else predecessor
    )
    second = record(transition("rules/x", "a", "b"), body=body)
    result = check([], [entry("rules/x", "b")], first, second)
    assert (
        isinstance(result, Coverage)
        if predecessor != "f" * 64
        else result.code == "inconsistent-chain"
    )


def test_later_candidate_correction_uses_null_predecessor():
    r = record(transition("rules/x", "old", "new"), body=correction())
    assert isinstance(
        check([entry("rules/x", "old")], [entry("rules/x", "new")], r), Coverage
    )


@pytest.mark.parametrize(
    "site", ["base", "subject", "intermediate", "unused", "unchanged"]
)
def test_mode_wall_covers_entire_domain(site):
    base = [entry("rules/x", "a", "100755" if site == "base" else "100644")]
    subject = [entry("rules/x", "b", "100755" if site == "subject" else "100644")]
    records = [
        record(
            transition(
                "rules/x", "a", "b", before_mode=base[0][1], after_mode=subject[0][1]
            )
        )
    ]
    if site in ("intermediate", "unused"):
        records.append(record(transition("rules/x", "a", "exe", after_mode="100755")))
    if site == "unchanged":
        base.append(entry("rules/z", "same", "100755"))
        subject.append(entry("rules/z", "same", "100755"))
    assert check(base, subject, *records).code == "inadmissible-entry"


def test_record_permutation_does_not_change_assignment():
    records = [
        record(transition("rules/x", before, after))
        for before, after in [(None, "a"), ("a", "b"), (None, "dead")]
    ]
    assert check([], [entry("rules/x", "b")], *records) == check(
        [], [entry("rules/x", "b")], *reversed(records)
    )


def test_unprotected_only_delta_has_empty_assignment():
    assert check([], [entry("README", "text")]) == Coverage((), (), ("README",))


def test_uncovered_refusal_precedes_mode_wall():
    assert check([], [entry("rules/z", "end", "100755")]).code == "uncovered-path"


def test_atomicity_refusal_names_the_affected_path():
    good = record(transition("rules/a", None, "ok"))
    bad = record(
        transition("rules/z", None, "bad"), transition("rules/unchanged", None, "extra")
    )
    result = check([], [entry("rules/a", "ok"), entry("rules/z", "bad")], good, bad)
    assert result.code == "inconsistent-chain"
    assert result.path == "rules/z"


def test_unusable_retry_does_not_change_atomic_failure_path():
    good = record(transition("rules/a", None, "ok"))
    retry = record(
        transition("rules/a", None, "ok"), transition("rules/unused", None, "extra")
    )
    bad = record(
        transition("rules/z", None, "bad"), transition("rules/unchanged", None, "extra")
    )
    result = check(
        [], [entry("rules/a", "ok"), entry("rules/z", "bad")], good, retry, bad
    )
    assert result.code == "inconsistent-chain"
    assert result.path == "rules/z"
