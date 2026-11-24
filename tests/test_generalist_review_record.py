"""The generalist review record and the --require-review gate (#2146)."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import itertools
import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode import cli
from axiom_encode import generalist_review_record as record_module
from axiom_encode.cli import (
    _APPLY_VALIDATION_SNAPSHOT_ATTR,
    _GENERALIST_REVIEW_STATE_ATTR,
    APPLIED_ENCODING_MODEL_TOOL,
    _rulespec_anchor_base_for_output,
    _write_applied_encoding_manifest,
    cmd_encode,
)
from axiom_encode.generalist_review_record import (
    GENERALIST_REVIEW_FIELD,
    GENERALIST_REVIEW_SCHEMA,
    AppliedPair,
    GeneralistReviewState,
    ReviewRequirementRefused,
    apply_stamp_dump,
    dumped_review_content_sha256,
    review_content_sha256,
    review_evidence_digests,
    review_requirement_refusal,
)
from axiom_encode.harness import validator_pipeline
from axiom_encode.harness.evals import (
    GENERALIST_REVIEW_EVIDENCE_ATTR,
    EvalArtifactMetrics,
)
from tests.test_cli import (
    TEST_APPLY_SIGNING_BROKER,
    TEST_PINNED_ENCODER_IDENTITY,
)

RULE_BYTES = b"""format: rulespec/v1
module:
  summary: A credit of $1,000.
rules:
  - name: credit_amount
    kind: parameter
    versions:
      - effective_from: '2018-01-01'
        formula: 1000
"""
TEST_BYTES = b"""- name: base_case
  period: 2026
  output:
    credit_amount: 1000
"""
# What `encode --apply` does to every candidate before it validates and
# applies it: stamp module.source_verification and dump the YAML again.
_stamped = yaml.safe_load(RULE_BYTES)
_stamped["module"]["source_verification"] = {
    "corpus_citation_path": "us/statute/26/24",
    "source_sha256": "5" * 64,
}
STAMPED_RULE_BYTES = yaml.safe_dump(_stamped, sort_keys=False).encode()
# The manifest paths of the one reviewed module and its companion test.
RULE_PATH = "statutes/rule.yaml"
TEST_PATH = "statutes/rule.test.yaml"


def applied_pair(rulespec, test, *, rulespec_path=RULE_PATH):
    """The module's pair at the reviewed path, unless a test names another."""

    return record_module.applied_pair(rulespec, test, rulespec_path=rulespec_path)


def _with_paths(applied, sha256s):
    """Give bare digests a manifest path: that of their pair, or their own.

    Tests written before the record named paths list applied files by digest.
    A digest of a pair file is listed at the pair's path; any other digest is
    one more applied file at a path of its own.
    """

    known = {}
    for pair in applied:
        if isinstance(pair, AppliedPair):
            known.setdefault(pair.rulespec_sha256, pair.rulespec_path)
            if pair.test_sha256 is not None:
                known.setdefault(pair.test_sha256, pair.test_path)
    return [
        item
        if isinstance(item, tuple) or not isinstance(item, str)
        else (known.get(item) or f"statutes/other/{item}.yaml", item)
        for item in sha256s
    ]


def build_generalist_review_record(
    state, applied, applied_files=None, *, reviewed_rulespec_path=RULE_PATH
):
    applied = list(applied)
    return record_module.build_generalist_review_record(
        state,
        applied,
        None if applied_files is None else _with_paths(applied, applied_files),
        reviewed_rulespec_path=reviewed_rulespec_path,
    )


def generalist_review_record_issues(record, applied_files):
    """The reader, for a manifest list of (path, digest) pairs or of digests.

    A bare digest is listed at the path the record gives it, or at a path of
    its own when the record does not list it.
    """

    files = record.get("files") if isinstance(record, dict) else None
    known = {}
    for entry in files if isinstance(files, list) else []:
        if isinstance(entry, dict) and isinstance(entry.get("sha256"), str):
            known.setdefault(entry["sha256"], entry.get("path"))
    return record_module.generalist_review_record_issues(
        record,
        [
            item
            if isinstance(item, tuple) or not isinstance(item, str)
            else (known.get(item) or f"statutes/other/{item}.yaml", item)
            for item in applied_files
        ],
    )


REPAIRED_RULE_BYTES = RULE_BYTES.replace(b"formula: 1000", b"formula: max(0, 1000)")
PAIR = applied_pair(RULE_BYTES, TEST_BYTES)
STAMPED_PAIR = applied_pair(STAMPED_RULE_BYTES, TEST_BYTES)
APPLIED = [PAIR]
APPLIED_SHA256S = [PAIR.rulespec_sha256, PAIR.test_sha256]
OTHER_SHA = "3" * 64
PROMPT_SHA = "4" * 64
DIGEST_FIELDS = (
    "rulespec_sha256",
    "rulespec_content_sha256",
    "test_sha256",
    "test_content_sha256",
)


OTHER_PATH = "statutes/other.yaml"
UNBOUND_APPLIED = dict.fromkeys((*DIGEST_FIELDS, "rulespec_path", "test_path"))


def _digests(pair):
    return {field: getattr(pair, field) for field in DIGEST_FIELDS}


def _applied(pair):
    """Expected `applied`: the pair's digests and the paths it is applied at."""

    return {
        **_digests(pair),
        "rulespec_path": pair.rulespec_path,
        "test_path": pair.test_path,
    }


def _other_pair():
    """A repaired dependent: another module, applied at another path."""

    return applied_pair(REPAIRED_RULE_BYTES, None, rulespec_path=OTHER_PATH)


def _reviewed(pair, rulespec=None):
    """Expected `reviewed`: the pair's digests plus the dumped RuleSpec digest."""

    return {
        **_digests(pair),
        "rulespec_dumped_content_sha256": dumped_review_content_sha256(
            RULE_BYTES if rulespec is None else rulespec
        ),
    }


UNSCORED_REVIEWED = dict.fromkeys((*DIGEST_FIELDS, "rulespec_dumped_content_sha256"))


class _FilesByDigest(list):
    """Expected `files` that leave the paths open: equal to a record's list
    with the same (sha256, reviewed, complete) entries, at any paths."""

    def __eq__(self, other):
        if not isinstance(other, list):
            return NotImplemented
        return sorted(
            (entry["sha256"], entry["reviewed"], entry["complete"]) for entry in other
        ) == sorted(
            (entry["sha256"], entry["reviewed"], entry["complete"]) for entry in self
        )

    __hash__ = None


def _files(*entries):
    """Expected `files`: (sha256, reviewed, complete) triples, at any path.

    A (path, sha256, reviewed, complete) entry also fixes the path; a list of
    those is compared exactly.
    """

    if entries and all(len(entry) == 4 for entry in entries):
        return sorted(
            (
                {
                    "path": path,
                    "sha256": sha256,
                    "reviewed": reviewed,
                    "complete": complete,
                }
                for path, sha256, reviewed, complete in entries
            ),
            key=lambda entry: entry["path"],
        )
    return _FilesByDigest(
        {"sha256": sha256, "reviewed": reviewed, "complete": complete}
        for sha256, reviewed, complete in entries
    )


def _evidence(rulespec=RULE_BYTES, test=TEST_BYTES, **overrides):
    evidence = {
        **review_evidence_digests(rulespec, test),
        "rulespec_complete": True,
        "test_complete": True,
        "explicit_passed": True,
        "reviewer_backend": "codex",
        "reviewer_model": "gpt-test",
    }
    evidence.update(overrides)
    return evidence


def _state(**overrides):
    values = {
        "metrics_pass": True,
        "score": 8.5,
        "issue_count": 2,
        "prompt_sha256": PROMPT_SHA,
        "evidence": _evidence(),
    }
    values.update(overrides)
    return GeneralistReviewState(**values)


def _build(applied=None, applied_sha256s=None, **overrides):
    return build_generalist_review_record(
        _state(**overrides), APPLIED if applied is None else applied, applied_sha256s
    )


def _issues(record, applied_sha256s=None):
    """The reader's findings; by default the manifest lists the record's files."""

    if applied_sha256s is None:
        files = record.get("files") if isinstance(record, dict) else None
        applied_sha256s = (
            [
                (entry.get("path"), entry.get("sha256"))
                for entry in files
                if isinstance(entry, dict)
            ]
            if isinstance(files, list)
            else APPLIED_SHA256S
        )
    return generalist_review_record_issues(record, applied_sha256s)


# --- the content digest --------------------------------------------------


def test_source_stamp_changes_the_bytes_and_not_the_content():
    assert STAMPED_PAIR.rulespec_sha256 != PAIR.rulespec_sha256
    assert STAMPED_PAIR.rulespec_content_sha256 == PAIR.rulespec_content_sha256


@pytest.mark.parametrize(
    "changed",
    [
        REPAIRED_RULE_BYTES,
        RULE_BYTES.replace(b"credit_amount", b"credit_amounts"),
        RULE_BYTES.replace(b"'2018-01-01'", b"'2019-01-01'"),
        RULE_BYTES.replace(b"formula: 1000", b"formula: '1000'"),
        RULE_BYTES.replace(b"formula: 1000", b"formula: 1000.5"),
        RULE_BYTES.replace(b"summary: A credit", b"summary: The credit"),
        RULE_BYTES + b"extra: true\n",
        RULE_BYTES.replace(b"rules:", b"rule:"),
        # A third key in the stamped mapping is content.
        yaml.safe_dump(
            {
                **_stamped,
                "module": {
                    **_stamped["module"],
                    "source_verification": {
                        **_stamped["module"]["source_verification"],
                        "values": {"a": 1},
                    },
                },
            }
        ).encode(),
    ],
)
def test_every_other_edit_changes_the_content_digest(changed):
    assert review_content_sha256(changed, rulespec=True) != PAIR.rulespec_content_sha256


def test_formatting_and_comments_do_not_change_the_content_digest():
    reformatted = yaml.safe_dump(yaml.safe_load(RULE_BYTES), sort_keys=True).encode()
    commented = b"# reviewed\n" + RULE_BYTES

    assert reformatted != RULE_BYTES
    assert (
        review_content_sha256(reformatted, rulespec=True)
        == review_content_sha256(commented, rulespec=True)
        == PAIR.rulespec_content_sha256
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (b"a: yes\n", b"a: true\n"),
        (b"a: 010\n", b"a: 8\n"),
        (b"a: 1:00\n", b"a: 60\n"),
        (b"a: on\n", b"a: true\n"),
        (b"a: 1_000\n", b"a: 1000\n"),
        (b"a: 0x10\n", b"a: 16\n"),
        (b"a: '1'\n", b"a: 1\n"),
        (b"a: 'true'\n", b"a: true\n"),
        (b"a: ''\n", b"a:\n"),
    ],
)
def test_values_that_only_yaml_1_1_reads_as_equal_have_different_digests(left, right):
    # PyYAML loads each pair to the same value; the engine reads YAML 1.2.
    for rulespec in (True, False):
        assert review_content_sha256(left, rulespec=rulespec) != (
            review_content_sha256(right, rulespec=rulespec)
        )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (b"a:\n", b"a: null\n"),
        (b"a: ~\n", b"a: null\n"),
        (b"a: True\n", b"a: true\n"),
        (b"a: 'x'\n", b'a: "x"\n'),
        (b"a: x\n", b"a: 'x'\n"),
        (b"a: |-\n  x\n  y\n", b'a: "x\\ny"\n'),
        (b"{a: 1, b: [1, 2]}\n", b"b:\n- 1\n- 2\na: 1\n"),
    ],
)
def test_spellings_of_one_value_have_one_digest(left, right):
    assert review_content_sha256(left, rulespec=True) == (
        review_content_sha256(right, rulespec=True)
    )


def test_content_digest_survives_the_dump_that_apply_performs():
    from axiom_encode.yaml_serialization import safe_dump_yaml

    original = b"""# header comment
format: rulespec/v1
module:
  summary: "A credit: $1,000 per child"
rules:
  - name: credit
    kind: derived
    dtype: Money
    period: Year
    indexed: false
    cap:
    rate: 0.5
    versions:
      - effective_from: '2018-01-01'
        formula: |-
          if is_child: 1000
          else: 0
      - effective_from: '2026-01-01'
        formula: max(0, base - 2)
"""
    document = yaml.safe_load(original)
    document["module"]["source_verification"] = {
        "corpus_citation_path": "us/statute/26/24",
        "source_sha256": "5" * 64,
    }
    # The exact call `_stamp_generated_source_attestation_for_apply` makes.
    dumped = safe_dump_yaml(document, sort_keys=False, allow_unicode=True).encode()

    assert dumped != original
    assert review_content_sha256(dumped, rulespec=True) == (
        review_content_sha256(original, rulespec=True)
    )


def test_only_the_two_stamped_keys_are_left_out():
    def module(**verification):
        return yaml.safe_dump(
            {"module": {"source_verification": verification}, "rules": []}
        ).encode()

    base = review_content_sha256(module(note="kept"), rulespec=True)
    assert base == review_content_sha256(
        module(note="kept", corpus_citation_path="x", source_sha256="y"),
        rulespec=True,
    )
    for extra in ({"values": {"a": 1}}, {"source_sha256s": "y"}, {"note": "other"}):
        assert base != review_content_sha256(
            module(**{"note": "kept", **extra}), rulespec=True
        )


def test_stamp_keys_are_neutral_only_in_a_rulespec_module():
    stamp_only = b"module:\n  source_verification:\n    source_sha256: abc\n"

    assert review_content_sha256(stamp_only, rulespec=True) == review_content_sha256(
        b"", rulespec=True
    )
    # In a companion test, and outside module.source_verification, they count.
    assert review_content_sha256(stamp_only, rulespec=False) != review_content_sha256(
        b"", rulespec=False
    )
    assert review_content_sha256(
        b"source_sha256: abc\n", rulespec=True
    ) != review_content_sha256(b"", rulespec=True)
    assert review_content_sha256(
        b"module:\n  source_sha256: abc\n", rulespec=True
    ) != review_content_sha256(b"", rulespec=True)


def test_absent_null_and_empty_companion_tests_are_different_content(tmp_path):
    # The companion loader acts on the difference: nothing loads an absent
    # file, an empty list is zero cases, and it refuses a null document and
    # an empty mapping.
    absent = review_content_sha256(None, rulespec=False)
    groups = {
        "empty list": (b"[]\n", b"[]  # none\n", b"--- []\n", b"[ ]\n"),
        "null document": (b"", b"# nothing\n", b"~\n", b"null\n", b"---\n"),
        "empty mapping": (b"{}\n", b"{ }\n"),
        "cases": (TEST_BYTES,),
    }
    digests = {
        name: {review_content_sha256(raw, rulespec=False) for raw in spellings}
        for name, spellings in groups.items()
    }

    assert absent is not None
    assert all(len(found) == 1 and None not in found for found in digests.values())
    assert len({absent, *(next(iter(found)) for found in digests.values())}) == 5

    def load(raw):
        path = tmp_path / "x.test.yaml"
        path.write_bytes(raw)
        try:
            return cli._load_rulespec_test_cases(path)
        except ValueError:
            return "refused"

    assert {load(raw) == [] for raw in groups["empty list"]} == {True}
    assert {
        load(raw) for raw in (*groups["null document"], *groups["empty mapping"])
    } == {"refused"}
    # A module without content keeps one digest: the stamp can add `module`.
    assert (
        len(
            {
                review_content_sha256(raw, rulespec=True)
                for raw in (None, b"", b"~\n", b"{}\n", b"[]\n", b"module:\n")
            }
        )
        == 1
    )


@pytest.mark.parametrize(
    "raw",
    [b"\xff\xfe", b"a: [unclosed\n", b"a: b: c\n", b"a: *missing\n"],
)
def test_unreadable_yaml_has_no_content_digest(raw):
    assert review_content_sha256(raw, rulespec=True) is None
    # ...so a verdict on it, or an applied file like it, is never bound.
    assert _build(evidence=_evidence(rulespec=raw))["status"] == "unavailable"
    assert _build(applied=[applied_pair(raw, TEST_BYTES)])["status"] == "unavailable"
    unreadable_both = build_generalist_review_record(
        _state(evidence=_evidence(rulespec=raw)), [applied_pair(raw, TEST_BYTES)]
    )
    assert unreadable_both["status"] == "unavailable"


def test_map_keys_of_different_types_do_not_collide():
    assert review_content_sha256(b"1: a\n", rulespec=False) != review_content_sha256(
        b"'1': a\n", rulespec=False
    )
    assert review_content_sha256(b"a: [1, 2]\n", rulespec=False) != (
        review_content_sha256(b"a: [2, 1]\n", rulespec=False)
    )
    assert review_content_sha256(b"{a: 1, b: 2}\n", rulespec=False) == (
        review_content_sha256(b"{b: 2, a: 1}\n", rulespec=False)
    )


# --- the record: one test per status -------------------------------------


def test_passed_record_carries_identity_verdict_and_digests():
    record = _build()

    assert record == {
        "schema": GENERALIST_REVIEW_SCHEMA,
        "status": "passed",
        "reason": None,
        "required": False,
        "reviewer": {
            "name": "generalist-reviewer",
            "backend": "codex",
            "model": "gpt-test",
        },
        "verdict": {"passed": True, "score": 8.5, "issue_count": 2},
        "prompt_sha256": PROMPT_SHA,
        "reviewed": _reviewed(PAIR),
        "applied": {
            **_digests(PAIR),
            "rulespec_path": RULE_PATH,
            "test_path": TEST_PATH,
        },
        "files": _files(
            (RULE_PATH, PAIR.rulespec_sha256, True, True),
            (TEST_PATH, PAIR.test_sha256, True, True),
        ),
    }
    assert _issues(record) == []


def test_review_before_the_source_stamp_covers_the_stamped_file():
    # The real `encode --apply` order: review, then stamp, then apply.
    record = _build(applied=[STAMPED_PAIR])

    assert record["status"] == "passed"
    assert record["reviewed"]["rulespec_sha256"] == PAIR.rulespec_sha256
    assert record["applied"]["rulespec_sha256"] == STAMPED_PAIR.rulespec_sha256
    assert record["reviewed"]["rulespec_sha256"] != record["applied"]["rulespec_sha256"]
    assert _issues(record, [STAMPED_PAIR.rulespec_sha256, PAIR.test_sha256]) == []
    # The manifest must list the applied bytes, not the reviewed bytes.
    assert _issues(record, APPLIED_SHA256S) == [
        "generalist_review does not list exactly the applied file digests"
    ]


def test_failed_record_keeps_the_reviewer_score():
    record = _build(metrics_pass=False, score=3.0)

    assert record["status"] == "failed"
    assert record["reason"] is None
    assert record["verdict"] == {"passed": False, "score": 3.0, "issue_count": 2}
    assert record["applied"] == _applied(PAIR)
    assert _issues(record) == []


def test_reviewer_that_could_not_start_is_unavailable():
    # What `_run_reviewer` returns for "reviewer_unavailable": no pass, no score.
    record = _build(
        metrics_pass=False, score=None, evidence=None, prompt_sha256=PROMPT_SHA
    )

    assert record["status"] == "unavailable"
    assert record["reason"] == "reviewer_error"
    assert record["verdict"] == {"passed": None, "score": None, "issue_count": 0}
    assert record["prompt_sha256"] is None
    assert record["reviewed"] == UNSCORED_REVIEWED
    assert record["files"] == _files(
        (PAIR.rulespec_sha256, False, False), (PAIR.test_sha256, False, False)
    )
    assert record["applied"] == UNBOUND_APPLIED
    assert _issues(record) == []


def test_skip_reviewers_flag_is_skipped_whatever_the_metrics_say():
    # --skip-reviewers makes the eval pipeline report a placeholder pass.
    for state in (
        _state(skipped_by_flag=True),
        _state(
            skipped_by_flag=True,
            metrics_pass=True,
            score=None,
            prompt_sha256=None,
            evidence=None,
        ),
    ):
        record = build_generalist_review_record(state, APPLIED)
        assert record["status"] == "skipped"
        assert record["reason"] == "skip_reviewers_flag"
        assert record["verdict"]["passed"] is None
        assert _issues(record) == []


def test_placeholder_pass_without_a_reviewer_call_is_skipped_not_passed():
    record = _build(metrics_pass=True, score=None, prompt_sha256=None, evidence=None)

    assert record["status"] == "skipped"
    assert record["reason"] == "reviewer_not_run"
    assert _issues(record) == []


# --- nothing unreadable becomes "passed" ---------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"metrics_pass": None},
        {"metrics_pass": "yes"},
        {"metrics_pass": 1},
        {"score": float("nan")},
        {"score": float("inf")},
        {"score": -1.0},
        {"score": 10.5},
        {"score": "9"},
        {"score": True},
        {"metrics_pass": False, "score": True},
        {"metrics_pass": False, "score": "2"},
        {"score": 6.9},  # a pass below the threshold is not a readable pass
        {"prompt_sha256": None},
        {"prompt_sha256": "not-a-digest"},
        {"prompt_sha256": "A" * 64},
        {"evidence": None},
        {"evidence": "evidence"},
        {"evidence": {}},
        {"evidence": _evidence(reviewed_rulespec_sha256=None)},
        {"evidence": _evidence(reviewed_rulespec_sha256="zz")},
        {
            "evidence": _evidence(
                reviewed_rulespec_content_sha256=None,
                reviewed_rulespec_dumped_content_sha256=None,
            )
        },
        {"evidence": _evidence(reviewed_rulespec_content_sha256="zz")},
        {"evidence": _evidence(reviewed_test_sha256="zz")},
        {"evidence": _evidence(reviewed_test_content_sha256=None)},
        {"evidence": _evidence(reviewed_test_content_sha256="zz")},
    ],
)
def test_unreadable_review_result_is_unavailable(overrides):
    record = _build(**overrides)

    assert record["status"] == "unavailable"
    assert record["reason"] == "unreadable_review_result"
    assert record["verdict"]["passed"] is None
    assert review_requirement_refusal(record) is not None
    assert _issues(record) == []


def test_score_at_the_threshold_passes_and_just_below_does_not():
    assert _build(score=7.0)["status"] == "passed"
    assert _build(score=7)["status"] == "passed"
    assert _build(score=6.999)["status"] == "unavailable"
    assert _build(score=10.0)["status"] == "passed"
    assert _build(score=0.0, metrics_pass=False)["status"] == "failed"


@pytest.mark.parametrize("verdict", [True, False])
@pytest.mark.parametrize(
    "applied",
    [
        [],
        [applied_pair(REPAIRED_RULE_BYTES, TEST_BYTES)],  # the rule was repaired
        [applied_pair(RULE_BYTES, TEST_BYTES + b"- name: added\n")],  # the test was
        [applied_pair(RULE_BYTES, None)],  # the reviewed test was not applied
        [applied_pair(TEST_BYTES, RULE_BYTES)],
        [PAIR.rulespec_sha256],  # not a pair
        [None],
    ],
)
def test_verdict_on_other_content_is_unavailable(verdict, applied):
    record = build_generalist_review_record(
        _state(metrics_pass=verdict, score=9.0 if verdict else 2.0), applied
    )

    assert record["status"] == "unavailable"
    assert record["reason"] == "reviewed_content_not_applied"
    # The verdict the reviewer gave stays readable; it is not the status.
    assert record["verdict"]["passed"] is verdict
    assert record["reviewed"]["rulespec_sha256"] == PAIR.rulespec_sha256
    assert record["applied"] == UNBOUND_APPLIED
    assert _issues(record) == []


def test_verdict_never_binds_to_a_pair_without_valid_byte_digests():
    for field, value in (("rulespec_sha256", "zz"), ("test_sha256", "zz")):
        malformed = AppliedPair(**{**_digests(PAIR), field: value})
        record = _build(applied=[malformed])
        assert record["status"] == "unavailable"
        assert record["applied"] == UNBOUND_APPLIED


def test_pass_on_one_file_is_partial_when_another_file_is_applied():
    # A dependent that overlay validation repaired is applied unreviewed.
    other = _other_pair()

    for applied in ([other, STAMPED_PAIR], [STAMPED_PAIR, other]):
        record = _build(applied=applied, required=True)
        assert record["status"] == "partial"
        assert record["reason"] == "unreviewed_applied_files"
        assert record["applied"] == _applied(STAMPED_PAIR)
        assert record["files"] == _files(
            (other.rulespec_sha256, False, False),
            (STAMPED_PAIR.rulespec_sha256, True, True),
            (STAMPED_PAIR.test_sha256, True, True),
        )
        assert review_requirement_refusal(record) is not None
        assert _issues(_tampered(record, ("required",), False)) == []


def test_pass_is_partial_when_an_applied_file_is_in_no_pair():
    # For example a companion test whose RuleSpec is not applied with it.
    record = _build(applied_sha256s=[*APPLIED_SHA256S, OTHER_SHA])

    assert record["status"] == "partial"
    assert record["reason"] == "unreviewed_applied_files"
    assert {
        "path": f"statutes/other/{OTHER_SHA}.yaml",
        "sha256": OTHER_SHA,
        "reviewed": False,
        "complete": False,
    } in record["files"]


@pytest.mark.parametrize(
    ("evidence", "truncated"),
    [
        ({"rulespec_complete": False}, "rulespec_sha256"),
        ({"test_complete": False}, "test_sha256"),
        ({"rulespec_complete": None}, "rulespec_sha256"),
        ({"rulespec_complete": "yes"}, "rulespec_sha256"),
        ({"test_complete": 1}, "test_sha256"),
    ],
)
def test_pass_on_a_truncated_file_is_partial(evidence, truncated):
    record = _build(evidence=_evidence(**evidence), required=True)

    assert record["status"] == "partial"
    assert record["reason"] == "review_saw_truncated_file"
    assert [entry["sha256"] for entry in record["files"] if not entry["complete"]] == [
        getattr(PAIR, truncated)
    ]
    assert all(entry["reviewed"] for entry in record["files"])
    assert review_requirement_refusal(record) is not None


def test_unreviewed_file_takes_precedence_over_truncation():
    record = _build(
        applied=[PAIR, _other_pair()],
        evidence=_evidence(rulespec_complete=False),
    )

    assert (record["status"], record["reason"]) == (
        "partial",
        "unreviewed_applied_files",
    )


def test_failed_review_stays_failed_whatever_else_is_applied():
    record = _build(
        applied=[PAIR, _other_pair()],
        evidence=_evidence(rulespec_complete=False),
        metrics_pass=False,
        score=2.0,
    )

    assert record["status"] == "failed"
    assert _issues(record) == []


def test_pass_with_nothing_applied_is_never_passed():
    assert _build(applied=[], applied_sha256s=[])["status"] == "unavailable"
    # A pair that is in no applied file list is not applied: nothing is bound.
    record = _build(applied_sha256s=[])
    assert record["status"] == "unavailable"
    assert record["reason"] == "reviewed_content_not_applied"
    assert record["files"] == []


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "passed"),
        (("reason",), "review_saw_truncated_file"),
        (("files", 0, "reviewed"), True),
    ],
)
def test_reader_refuses_a_partial_record_dressed_up(path, value):
    partial = _build(applied=[PAIR, _other_pair()])
    assert partial["status"] == "partial"
    assert _issues(partial) == []
    index = next(i for i, entry in enumerate(partial["files"]) if not entry["reviewed"])
    path = tuple(index if part == 0 else part for part in path)
    changed = _tampered(partial, path, value)
    if path == ("status",):
        changed["reason"] = None

    assert _issues(changed)


def test_reader_refuses_a_pass_that_marks_an_unread_file_reviewed():
    other = _other_pair()
    partial = _build(applied=[PAIR, other])
    forged = copy.deepcopy(partial)
    forged["status"], forged["reason"] = "passed", None
    for entry in forged["files"]:
        entry["reviewed"] = entry["complete"] = True

    # Every file now reads as reviewed in full, but the record names one pair.
    assert _issues(forged) == [
        "generalist_review verdict is not bound to the applied file digests"
    ]


def test_reader_refuses_a_partial_or_passed_status_on_a_failing_verdict():
    for record in (_build(), _build(evidence=_evidence(test_complete=False))):
        failing = _tampered(
            _tampered(record, ("verdict", "passed"), False), ("verdict", "score"), 2.0
        )
        assert any("contradicts its verdict" in issue for issue in _issues(failing))


@pytest.mark.parametrize(
    "unscored",
    [
        {"metrics_pass": False, "score": None, "evidence": None},
        {"skipped_by_flag": True},
        {"applied": []},
    ],
)
def test_reader_refuses_a_file_marked_reviewed_or_complete_without_a_bound_verdict(
    unscored,
):
    record = build_generalist_review_record(
        _state(**{k: v for k, v in unscored.items() if k != "applied"}),
        unscored.get("applied", APPLIED),
        APPLIED_SHA256S,
    )
    assert record["status"] in {"unavailable", "skipped"}
    assert _issues(record) == []

    assert _issues(_tampered(record, ("files", 0, "reviewed"), True))
    assert _issues(_tampered(record, ("files", 0, "complete"), True)) == [
        "generalist_review marks an unreviewed file complete"
    ]


def test_reader_refuses_unsorted_or_repeated_files():
    passed = _build()
    reversed_files = _tampered(passed, ("files",), passed["files"][::-1])
    repeated = _tampered(passed, ("files",), [*passed["files"], passed["files"][0]])

    for changed in (reversed_files, repeated):
        assert _issues(changed, APPLIED_SHA256S) == [
            "generalist_review files are not sorted and unique"
        ]


def test_reader_refuses_a_passed_record_with_a_truncated_file():
    truncated = _build(evidence=_evidence(test_complete=False))
    assert truncated["status"] == "partial"

    forged = _tampered(_tampered(truncated, ("status",), "passed"), ("reason",), None)
    assert _issues(forged) == [
        "generalist_review status passed, but an applied file was not reviewed in full"
    ]


def test_review_without_a_companion_test_covers_an_absent_one_only():
    evidence = _evidence(test=None)
    assert evidence["reviewed_test_sha256"] is None

    absent = applied_pair(RULE_BYTES, None)
    record = _build(applied=[absent], evidence=evidence)
    assert record["status"] == "passed"
    assert record["applied"]["test_sha256"] is None
    assert _issues(record, [absent.rulespec_sha256]) == []

    # An applied companion, also an empty one, is not the content of an
    # absent one: the reviewer never read that file, so nothing is bound and
    # a required review refuses.
    for later in (b"[]\n", b"", b"~\n", b"{}\n"):
        empty = applied_pair(RULE_BYTES, later)
        record = _build(applied=[empty], evidence=evidence, required=True)
        assert record["status"] == "unavailable"
        assert record["reason"] == "reviewed_content_not_applied"
        assert record["applied"] == UNBOUND_APPLIED
        assert record["files"] == _files(
            (empty.rulespec_sha256, False, False), (empty.test_sha256, False, False)
        )
        assert review_requirement_refusal(record) is not None
        assert (
            _issues(
                {**record, "required": False},
                [empty.rulespec_sha256, empty.test_sha256],
            )
            == []
        )
    empty = applied_pair(RULE_BYTES, b"[]\n")
    # A reviewer that was shown the empty companion covers it.
    shown = _build(applied=[empty], evidence=_evidence(test=b"[]\n"), required=True)
    assert shown["status"] == "passed"
    assert _issues(shown, [empty.rulespec_sha256, empty.test_sha256]) == []
    # ...and not a companion test that has cases the reviewer never saw.
    assert _build(evidence=evidence)["status"] == "unavailable"


def test_reviewer_identity_keeps_only_known_backends_and_plain_model_names():
    assert _build(evidence=_evidence(reviewer_backend="other"))["reviewer"] == {
        "name": "generalist-reviewer",
        "backend": None,
        "model": None,
    }
    assert (
        _build(evidence=_evidence(reviewer_model="x y; rm"))["reviewer"]["model"]
        is None
    )
    assert _build(evidence=_evidence(reviewer_backend="claude", reviewer_model="opus"))[
        "reviewer"
    ] == {"name": "generalist-reviewer", "backend": "claude", "model": "opus"}


def test_record_never_carries_reviewer_prose():
    record = _build(issue_count=3)

    assert "issues" not in record["verdict"]
    assert all(
        not isinstance(value, str) or len(value) <= 128
        for nested in (
            record,
            record["reviewer"],
            record["reviewed"],
            record["applied"],
        )
        for value in nested.values()
    )


# --- the reader's check --------------------------------------------------


def _tampered(record, path, value):
    changed = copy.deepcopy(record)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return changed


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema",), "axiom-encode/generalist-review/v2"),
        (("status",), "approved"),
        (("status",), "failed"),
        (("status",), "unavailable"),
        (("status",), "skipped"),
        (("reason",), "reviewer_error"),
        (("reason",), "reviewed_content_not_applied"),
        (("required",), "yes"),
        (("verdict", "passed"), False),
        (("verdict", "passed"), None),
        (("verdict", "score"), 6.0),
        (("verdict", "score"), None),
        (("verdict", "score"), 11.0),
        (("verdict", "issue_count"), -1),
        (("verdict", "issue_count"), 1.5),
        (("prompt_sha256",), None),
        (("reviewed", "rulespec_sha256"), None),
        (("reviewed", "rulespec_content_sha256"), "zz"),
        (("reviewed", "rulespec_dumped_content_sha256"), "zz"),
        (("reviewed", "rulespec_dumped_content_sha256"), []),
        (("reviewed", "test_sha256"), "zz"),
        (("reviewed", "test_content_sha256"), OTHER_SHA),
        (("files",), []),
        (("files", 0, "reviewed"), False),
        (("files", 0, "complete"), False),
        (("files", 0, "complete"), "yes"),
        (("files", 0, "sha256"), OTHER_SHA),
        (("applied", "rulespec_sha256"), OTHER_SHA),
        (("applied", "rulespec_sha256"), None),
        (("applied", "rulespec_content_sha256"), OTHER_SHA),
        (("applied", "test_sha256"), OTHER_SHA),
        (("applied", "test_sha256"), None),
        (("applied", "test_content_sha256"), OTHER_SHA),
        (("reviewer", "name"), "someone"),
        (("reviewer", "backend"), "other"),
        (("reviewer", "model"), "x y"),
    ],
)
def test_reader_refuses_each_inconsistent_passed_record(path, value):
    passed = _build()
    assert _issues(passed) == []

    assert _issues(_tampered(passed, path, value))


def test_reader_refuses_a_forged_pass_on_an_unreviewed_record():
    for unscored in (
        _build(metrics_pass=False, score=None, evidence=None),
        _build(skipped_by_flag=True),
    ):
        forged = _tampered(
            _tampered(unscored, ("status",), "passed"), ("reason",), None
        )
        assert _issues(forged)
        # A verdict or a digest smuggled into an unscored record is refused too.
        assert _issues(_tampered(unscored, ("verdict", "passed"), True))
        assert _issues(
            _tampered(unscored, ("applied", "rulespec_sha256"), PAIR.rulespec_sha256)
        )
        assert _issues(
            _tampered(unscored, ("reviewed", "test_content_sha256"), OTHER_SHA)
        )


def test_reader_refuses_a_forged_pass_on_content_that_was_not_applied():
    not_applied = _build(applied=[applied_pair(REPAIRED_RULE_BYTES, TEST_BYTES)])
    assert not_applied["status"] == "unavailable"

    forged = _tampered(_tampered(not_applied, ("status",), "passed"), ("reason",), None)
    assert _issues(forged)
    # Naming the applied file does not help: its content digest differs.
    repaired = applied_pair(REPAIRED_RULE_BYTES, TEST_BYTES)
    forged["applied"] = _applied(repaired)
    assert (
        "generalist_review verdict is on content that differs from the applied content"
        in _issues(forged, [repaired.rulespec_sha256, repaired.test_sha256])
    )


def test_reader_refuses_a_pass_whose_digests_are_not_in_the_manifest():
    passed = _build()

    assert _issues(passed, [OTHER_SHA]) == [
        "generalist_review does not list exactly the applied file digests"
    ]
    assert _issues(passed, [PAIR.rulespec_sha256]) == [
        "generalist_review does not list exactly the applied file digests"
    ]
    assert _issues(passed, [PAIR.test_sha256]) == [
        "generalist_review does not list exactly the applied file digests"
    ]


@pytest.mark.parametrize(
    "field",
    ["rulespec_sha256", "test_content_sha256"],
)
def test_reader_refuses_a_not_applied_record_without_its_reviewed_digests(field):
    not_applied = _build(applied=[])
    assert not_applied["reason"] == "reviewed_content_not_applied"
    assert _issues(not_applied) == []

    assert _issues(_tampered(not_applied, ("reviewed", field), None)) == [
        "generalist_review scored verdict is malformed"
    ]


def test_reader_refuses_unavailable_that_names_an_applied_file():
    not_applied = _build(applied=[])
    assert not_applied["status"] == "unavailable"

    assert _issues(
        _tampered(not_applied, ("applied", "rulespec_sha256"), PAIR.rulespec_sha256)
    )


def test_reader_refuses_a_required_record_that_did_not_pass():
    failed = _build(metrics_pass=False, score=2.0, required=True)

    assert failed["required"] is True
    assert _issues(failed) == [
        "generalist_review was required but the recorded status is failed"
    ]
    assert _issues(_build(required=True)) == []


@pytest.mark.parametrize(
    "record",
    [
        None,
        [],
        "passed",
        {},
        {"status": "passed"},
    ],
)
def test_reader_refuses_a_malformed_record(record):
    assert _issues(record)


def test_reader_refuses_an_extra_or_missing_field():
    passed = _build()
    extra = {**passed, "note": "x"}
    missing = {key: value for key, value in passed.items() if key != "required"}
    nested = _tampered(passed, ("verdict",), {**passed["verdict"], "issues": ["x"]})
    applied_extra = _tampered(passed, ("applied",), {**passed["applied"], "path": "x"})

    assert _issues(extra)
    assert _issues(missing)
    assert _issues(nested)
    assert _issues(applied_extra)


# --- the setting ---------------------------------------------------------


def test_requirement_refuses_every_status_but_passed():
    assert review_requirement_refusal(_build()) is None
    for record in (
        _build(metrics_pass=False, score=2.0),
        _build(metrics_pass=False, score=None, evidence=None),
        _build(skipped_by_flag=True),
        _build(metrics_pass=True, score=None, prompt_sha256=None, evidence=None),
        _build(applied=[]),
        _build(applied=[applied_pair(REPAIRED_RULE_BYTES, TEST_BYTES)]),
        _build(evidence=_evidence(rulespec_complete=False)),
        _build(applied=[PAIR, _other_pair()]),
        {},
        {"status": "PASSED"},
    ):
        refusal = review_requirement_refusal(record)
        assert refusal is not None and "Nothing was applied" in refusal


def test_require_review_is_an_encode_option_and_off_by_default(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_encode", seen.append)
    for extra in ([], ["--require-review"]):
        monkeypatch.setattr(
            "sys.argv",
            [
                "axiom-encode",
                "encode",
                "26 USC 1",
                "--output",
                "/tmp/out",
                "--corpus-path",
                "/tmp/corpus",
                "--axiom-rules-engine-path",
                "/tmp/engine",
                "--policy-repo-path",
                "/tmp/rulespec-us",
                *extra,
            ],
        )
        try:
            cli.main()
        except SystemExit as exc:
            assert exc.code in (0, None)

    assert [args.require_review for args in seen] == [False, True]


# --- the reviewer call measures what it read -----------------------------


def _pipeline(tmp_path):
    pipeline = object.__new__(validator_pipeline.ValidatorPipeline)
    pipeline.policy_repo_path = tmp_path
    return pipeline


def _candidate(
    tmp_path, rule_text="format: rulespec/v1\nrules: []\n", test_text="[]\n"
):
    rule = tmp_path / "rule.yaml"
    rule.write_text(rule_text)
    if test_text is not None:
        (tmp_path / "rule.test.yaml").write_text(test_text)
    return rule


def _metrics_from(result):
    """Carry one reviewer result the way `_evaluate_artifact_in_scope` does."""

    metrics = EvalArtifactMetrics(
        compile_pass=True,
        compile_issues=[],
        ci_pass=True,
        ci_issues=[],
        embedded_source_present=True,
        grounded_numeric_count=0,
        ungrounded_numeric_count=0,
        grounding=[],
        generalist_review_pass=result.passed,
        generalist_review_score=result.score,
        generalist_review_issues=result.issues,
        generalist_review_prompt_sha256=result.prompt_sha256,
    )
    evidence = result.details.get("review_evidence")
    if isinstance(evidence, dict):
        setattr(metrics, GENERALIST_REVIEW_EVIDENCE_ATTR, dict(evidence))
    return metrics


def _record_for(result, rule, **state):
    """The record for one reviewer result against the candidate as it is now."""

    return build_generalist_review_record(
        cli._encode_generalist_review_state(
            SimpleNamespace(metrics=_metrics_from(result)),
            skipped_by_flag=state.get("skipped_by_flag", False),
            required=state.get("required", False),
        ),
        cli._review_applied_pairs([rule], manifest_root=rule.parent),
        reviewed_rulespec_path=rule.name,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scored_review_records_the_digests_of_the_bytes_it_read(tmp_path, monkeypatch):
    rule = _candidate(tmp_path, rule_text=RULE_BYTES.decode())
    test = tmp_path / "rule.test.yaml"
    monkeypatch.setattr(
        validator_pipeline,
        "run_claude_code",
        lambda *_a, **_k: ('{"score": 9.0, "passed": true, "blocking_issues": []}', 0),
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)

    assert result.passed is True
    assert result.details["review_evidence"] == {
        "reviewed_rulespec_sha256": _sha(rule),
        "reviewed_rulespec_content_sha256": PAIR.rulespec_content_sha256,
        "reviewed_test_sha256": _sha(test),
        "reviewed_test_content_sha256": review_content_sha256(
            test.read_bytes(), rulespec=False
        ),
        "reviewed_rulespec_dumped_content_sha256": PAIR.rulespec_content_sha256,
        "rulespec_complete": True,
        "test_complete": True,
        "explicit_passed": True,
        # The mocked runner started no CLI, so the identity stays unknown.
        "reviewer_backend": None,
        "reviewer_model": None,
    }
    assert _record_for(result, rule)["status"] == "passed"
    # The source stamp that apply writes afterwards keeps the verdict bound...
    rule.write_bytes(STAMPED_RULE_BYTES)
    stamped = _record_for(result, rule)
    assert stamped["status"] == "passed"
    assert stamped["applied"]["rulespec_sha256"] == _sha(rule)
    assert stamped["reviewed"]["rulespec_sha256"] == PAIR.rulespec_sha256
    # ...a repair to the rule or to the companion test does not.
    rule.write_bytes(REPAIRED_RULE_BYTES)
    assert _record_for(result, rule)["reason"] == "reviewed_content_not_applied"
    rule.write_bytes(STAMPED_RULE_BYTES)
    test.write_bytes(TEST_BYTES)
    assert _record_for(result, rule)["reason"] == "reviewed_content_not_applied"
    # A candidate that can no longer be read is never bound.
    rule.unlink()
    assert _record_for(result, rule)["reason"] == "reviewed_content_not_applied"


@pytest.mark.parametrize("changed_file", ["rulespec", "companion"])
def test_review_prompt_and_digests_use_the_same_file_snapshot(
    tmp_path, monkeypatch, changed_file
):
    rule = _candidate(
        tmp_path, rule_text=RULE_BYTES.decode(), test_text=TEST_BYTES.decode()
    )
    test = tmp_path / "rule.test.yaml"
    changed = rule if changed_file == "rulespec" else test
    original = RULE_BYTES if changed_file == "rulespec" else TEST_BYTES
    replacement = original.replace(b"1000", b"2000")
    original_read_bytes = Path.read_bytes
    prompts = []
    reads = []

    def replace_after_first_read(path):
        # The reviewer call reads each file with `read_bytes`. Swap the file
        # right after its first read: a second read would see other bytes.
        raw = original_read_bytes(path)
        if path == changed:
            reads.append(raw)
            if len(reads) == 1:
                changed.write_bytes(replacement)
        return raw

    def reviewer(prompt, **_kwargs):
        prompts.append(prompt)
        return '{"score": 9.0, "passed": true}', 0

    monkeypatch.setattr(Path, "read_bytes", replace_after_first_read)
    monkeypatch.setattr(validator_pipeline, "run_claude_code", reviewer)

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)

    monkeypatch.setattr(Path, "read_bytes", original_read_bytes)
    # One read of the file: the prompt and the digests come from one snapshot.
    assert reads == [original]
    assert changed.read_bytes() == replacement
    assert original.decode() in prompts[0]
    assert replacement.decode() not in prompts[0]
    digest_field = (
        "reviewed_rulespec_sha256"
        if changed_file == "rulespec"
        else "reviewed_test_sha256"
    )
    assert (
        result.details["review_evidence"][digest_field]
        == hashlib.sha256(original).hexdigest()
    )
    # The candidate is now the swapped file, which the reviewer never read.
    record = _record_for(result, rule, required=True)
    assert record["status"] == "unavailable"
    assert record["reason"] == "reviewed_content_not_applied"
    assert review_requirement_refusal(record) is not None


@pytest.mark.parametrize("verdict", ["false", "true", 0, 1, None, [], {}, [True]])
def test_non_boolean_reviewer_verdict_cannot_satisfy_required_review(
    tmp_path, monkeypatch, verdict
):
    rule = _candidate(tmp_path)
    monkeypatch.setattr(
        validator_pipeline,
        "run_claude_code",
        lambda *_a, **_k: (json.dumps({"score": 9.0, "passed": verdict}), 0),
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)

    assert result.passed is False
    assert "reviewer_parse_failed" in result.issues
    record = _record_for(result, rule, required=True)
    assert record["status"] == "unavailable"
    assert review_requirement_refusal(record) is not None


@pytest.mark.parametrize(
    ("blocking", "non_blocking", "expected_status"),
    [
        (["wrong and/or"], [], "failed"),
        (["wrong and/or"], ["clarify a comment"], "failed"),
        ([], ["clarify a comment"], "passed"),
    ],
)
def test_required_review_respects_blocking_findings(
    tmp_path, monkeypatch, blocking, non_blocking, expected_status
):
    rule = _candidate(tmp_path)
    monkeypatch.setattr(
        validator_pipeline,
        "run_claude_code",
        lambda *_a, **_k: (
            json.dumps(
                {
                    "score": 9.0,
                    "passed": True,
                    "blocking_issues": blocking,
                    "non_blocking_issues": non_blocking,
                }
            ),
            0,
        ),
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)
    record = _record_for(result, rule, required=True)

    assert record["status"] == expected_status
    assert result.passed is (not blocking)
    assert (review_requirement_refusal(record) is None) is (not blocking)


@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    verdict=st.one_of(
        st.none(),
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(),
        st.lists(st.booleans()),
        st.dictionaries(st.text(), st.integers()),
    )
)
def test_non_boolean_reviewer_verdict_property(tmp_path, monkeypatch, verdict):
    # Each example rewrites the same scratch candidate and replaces the mock.
    rule = _candidate(tmp_path)
    monkeypatch.setattr(
        validator_pipeline,
        "run_claude_code",
        lambda *_a, **_k: (json.dumps({"score": 9.0, "passed": verdict}), 0),
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)

    assert result.passed is False
    assert "reviewer_parse_failed" in result.issues
    assert review_requirement_refusal(_record_for(result, rule, required=True))


def test_review_without_a_companion_test_file(tmp_path, monkeypatch):
    rule = _candidate(tmp_path, rule_text=RULE_BYTES.decode(), test_text=None)
    monkeypatch.setattr(
        validator_pipeline,
        "run_claude_code",
        lambda *_a, **_k: ('{"score": 9.0, "passed": true}', 0),
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)

    assert result.details["review_evidence"]["reviewed_test_sha256"] is None
    assert _record_for(result, rule)["status"] == "passed"
    # This module is not deferred, so the reviewer was shown no companion. A
    # companion that appears later, also an empty one, is a file the reviewer
    # did not read: nothing is bound.
    for later in (b"[]\n", b"", TEST_BYTES):
        (tmp_path / "rule.test.yaml").write_bytes(later)
        assert _record_for(result, rule)["status"] == "unavailable"
        assert _record_for(result, rule)["reason"] == "reviewed_content_not_applied"


def test_applied_pairs_pair_each_rulespec_with_its_companion_test(tmp_path):
    rule = _candidate(tmp_path, rule_text=RULE_BYTES.decode())
    test = tmp_path / "rule.test.yaml"
    test.write_bytes(TEST_BYTES)
    lone = tmp_path / "lone.yaml"
    lone.write_bytes(REPAIRED_RULE_BYTES)

    pairs = cli._review_applied_pairs(
        [rule, test, lone, tmp_path / "gone.yaml"], manifest_root=tmp_path
    )
    assert pairs == [
        applied_pair(RULE_BYTES, TEST_BYTES, rulespec_path="rule.yaml"),
        applied_pair(REPAIRED_RULE_BYTES, None, rulespec_path="lone.yaml"),
    ]
    assert [(pair.rulespec_path, pair.test_path) for pair in pairs] == [
        ("rule.yaml", "rule.test.yaml"),
        ("lone.yaml", None),
    ]
    # A companion test is never a RuleSpec of its own.
    assert cli._review_applied_pairs([test], manifest_root=tmp_path) == []
    # A file outside the manifest root has no manifest path and no pair.
    assert cli._review_applied_pairs([rule], manifest_root=tmp_path / "other") == []


def test_review_of_a_truncated_candidate_says_so(tmp_path, monkeypatch):
    rule = _candidate(tmp_path, rule_text="format: rulespec/v1\n" + "# x\n" * 2000)
    monkeypatch.setattr(
        validator_pipeline,
        "run_claude_code",
        lambda *_a, **_k: ('{"score": 9.0, "passed": true}', 0),
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)

    assert result.details["review_evidence"]["rulespec_complete"] is False
    assert result.details["review_evidence"]["test_complete"] is True
    record = _record_for(result, rule)
    assert record["status"] == "partial"
    assert record["reason"] == "review_saw_truncated_file"
    assert review_requirement_refusal(record) is not None


def test_review_of_a_truncated_companion_test_says_so(tmp_path, monkeypatch):
    rule = _candidate(
        tmp_path,
        rule_text=RULE_BYTES.decode(),
        test_text="".join(f"- name: case_{i}\n" for i in range(400)),
    )
    monkeypatch.setattr(
        validator_pipeline,
        "run_claude_code",
        lambda *_a, **_k: ('{"score": 9.0, "passed": true}', 0),
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)

    assert result.details["review_evidence"]["rulespec_complete"] is True
    assert result.details["review_evidence"]["test_complete"] is False
    record = _record_for(result, rule)
    assert (record["status"], record["reason"]) == (
        "partial",
        "review_saw_truncated_file",
    )


@pytest.mark.parametrize(
    ("output", "returncode"),
    [
        ("reviewer_unavailable: Reviewer CLIs not found (missing claude and codex)", 1),
        ("this is not json", 0),
        ("", 0),
        ('{"score": 1e309, "passed": true}', 0),
        ("Timeout after 300s", 1),
        ('{"score": 9.0, "passed": true}', 2),
    ],
)
def test_reviewer_output_that_cannot_be_read_never_records_a_pass(
    tmp_path, monkeypatch, output, returncode
):
    rule = _candidate(tmp_path)
    monkeypatch.setattr(
        validator_pipeline, "run_claude_code", lambda *_a, **_k: (output, returncode)
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)
    record = _record_for(result, rule)

    assert record["status"] == "unavailable"
    assert record["reason"] == "reviewer_error"
    assert review_requirement_refusal(record) is not None


@pytest.mark.parametrize(
    "output",
    [
        '{"score":9,"passed":false,"passed":true,'
        '"blocking_issues":["bad"],"blocking_issues":[]}',
        '{"passed":true,"score":1,"score":9}',
        '{"passed":true,"score":9,"blocking_issues":["bad"],"blocking_issues":[],}',
    ],
)
def test_duplicate_verdict_keys_cannot_satisfy_required_review(
    tmp_path, monkeypatch, output
):
    rule = _candidate(tmp_path)
    monkeypatch.setattr(
        validator_pipeline, "run_claude_code", lambda *_a, **_k: (output, 0)
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)
    record = _record_for(result, rule, required=True)

    assert result.passed is False
    assert "reviewer_parse_failed" in result.issues
    assert record["required"] is True
    assert record["status"] == "unavailable"
    assert review_requirement_refusal(record) is not None


@pytest.mark.parametrize(
    "prefix,between",
    [
        ("The flow map `{` is never closed.\n", 'Wait, I misread "rate.\n'),
        (
            "",
            "The source has `source: {excerpt: \"based on the taxpayer's`\n",
        ),
        ('The flow map `{` and note "rate.\n', 'The "rate is fine.\n'),
        ('{"note": "unfinished\n', 'The quoted "rate is wrong.\n'),
    ],
)
@pytest.mark.parametrize("first", [True, False])
def test_desynchronized_verdicts_cannot_satisfy_required_review(
    tmp_path, monkeypatch, prefix, between, first
):
    rule = _candidate(tmp_path)
    verdicts = [
        json.dumps({"passed": p, "score": 9 if p else 2}) for p in [first, not first]
    ]
    output = prefix + verdicts[0] + between + verdicts[1]
    monkeypatch.setattr(
        validator_pipeline, "run_claude_code", lambda *_a, **_k: (output, 0)
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)
    record = _record_for(result, rule, required=True)

    assert result.passed is False
    assert result.score is None
    assert "reviewer_parse_failed" in result.issues
    assert record["required"] is True
    assert record["status"] == "unavailable"
    assert record["verdict"]["passed"] is None
    assert review_requirement_refusal(record) is not None


@pytest.mark.parametrize(
    "prefix",
    ['{"note":"{",', '{"note":"{  ",', '{"note":"{",":":1,'],
)
@pytest.mark.parametrize("failure_first", [True, False])
def test_string_ending_in_brace_cannot_hide_a_failure_from_required_review(
    tmp_path, monkeypatch, prefix, failure_first
):
    rule = _candidate(tmp_path)
    verdicts = [
        prefix + '"passed":false,"score":1,"blocking_issues":["wrong and/or"]}',
        '{"passed":true,"score":9}',
    ]
    if not failure_first:
        verdicts.reverse()
    output = "\n".join(verdicts)
    monkeypatch.setattr(
        validator_pipeline, "run_claude_code", lambda *_a, **_k: (output, 0)
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)
    record = _record_for(result, rule, required=True)

    assert result.passed is False
    assert result.score is None
    assert "reviewer_parse_failed" in result.issues
    assert record["required"] is True
    assert record["status"] == "unavailable"
    assert record["verdict"]["passed"] is None
    assert review_requirement_refusal(record) is not None


@pytest.mark.parametrize("prefix", ['{"note":"{",', '{"note":"{",":":1,'])
def test_one_failure_after_a_string_ending_in_brace_fails_required_review(
    tmp_path, monkeypatch, prefix
):
    rule = _candidate(tmp_path)
    output = prefix + '"passed":false,"score":1,"blocking_issues":["wrong and/or"]}'
    monkeypatch.setattr(
        validator_pipeline, "run_claude_code", lambda *_a, **_k: (output, 0)
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)
    record = _record_for(result, rule, required=True)

    # The key `note` is outside the schema of the prompt: the failure is not
    # read as a verdict, and the review stays unreadable.
    assert result.passed is False
    assert result.score is None
    assert record["status"] == "unavailable"
    assert review_requirement_refusal(record) is not None


_GENERALIST_PASS = (
    '{"score": 9, "passed": true, "blocking_issues": [], '
    '"non_blocking_issues": [], "reasoning": "ok"}'
)


@pytest.mark.parametrize(
    "output",
    [
        # A key outside the schema that the prompt asks for.
        '{"passed": true, "pass": false, "score": 9}',
        '{"passed": true, "score": 9, "failed": true}',
        '{"passed": true, "score": 9, "verdict": "fail"}',
        '{"passed": true, "score": 9, "note": "extra"}',
        # The schema of another reviewer type.
        '{"passed": true, "score": 9, "issues": []}',
        # Not one top-level object.
        '[{"passed": true, "score": 9}]',
        '{"result": {"passed": true, "score": 9}}',
        '{"passed": true, "score": 9} {"passed": true, "score": 9}',
        '[{"passed": true, "score": 9}, {"passed": false, "score": 2}]',
        "[]",
        "true",
        "9",
        '"passed"',
        '{"passed": true, "score": 9, "details": {"score": 8, "checks": 4}}',
        # Schema keys only at the top level, and a second verdict or another
        # spelling of a verdict key inside a value.
        '{"passed": true, "score": 9, "blocking_issues": [], '
        '"non_blocking_issues": [{"passed": false, "score": 2}]}',
        '{"passed": true, "score": 9, "non_blocking_issues": [{"Passed": false}]}',
        '{"passed": true, "score": 9, "reasoning": {"passed": false}}',
        # Text beside the object.
        "Here is my review.\n\n```json\n" + _GENERALIST_PASS + "\n```\nDone.",
        "I compared each branch.\n" + _GENERALIST_PASS + "\nNo further notes.",
        "[\n" + _GENERALIST_PASS + "\n]",
        "\ufeff\ufeff" + _GENERALIST_PASS,
        "My review:\n" + _GENERALIST_PASS,
        _GENERALIST_PASS + "\nDone.",
        "My review:\n```json\n" + _GENERALIST_PASS + "\n```",
        "```json\n" + _GENERALIST_PASS + "\n```\nDone.",
        "```json\n"
        + _GENERALIST_PASS
        + "\n```\n```json\n"
        + _GENERALIST_PASS
        + "\n```",
        "```yaml\n" + _GENERALIST_PASS + "\n```",
        "```json\n[" + _GENERALIST_PASS + "]\n```",
        # An object that the output does not close.
        '{"passed": true, "score": 9',
        "```json\n" + '{"passed": true, "score": 9' + "\n```",
    ],
)
@pytest.mark.parametrize("required", [True, False])
def test_output_that_is_not_one_schema_object_never_records_a_pass(
    tmp_path, monkeypatch, output, required
):
    rule = _candidate(tmp_path)
    monkeypatch.setattr(
        validator_pipeline, "run_claude_code", lambda *_a, **_k: (output, 0)
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)
    record = _record_for(result, rule, required=required)

    assert result.passed is False
    assert result.score is None
    assert "reviewer_parse_failed" in result.issues
    assert (record["status"], record["reason"]) == ("unavailable", "reviewer_error")
    assert record["verdict"]["passed"] is None
    assert review_requirement_refusal(record) is not None


@pytest.mark.parametrize(
    "output",
    [
        _GENERALIST_PASS,
        "\n  " + _GENERALIST_PASS + "\n\n",
        "```json\n" + _GENERALIST_PASS + "\n```",
        "```JSON\n" + _GENERALIST_PASS + "\n```\n",
        "```\n" + _GENERALIST_PASS + "\n```",
        "```json\n\n" + _GENERALIST_PASS + "\n\n```",
        # One leading byte order mark is an encoding artifact, not text.
        "\ufeff" + _GENERALIST_PASS,
        json.dumps(json.loads(_GENERALIST_PASS), indent=2),
        '{"score": 9, "passed": true}',
        # A trailing comma is repaired; the object is still the whole output.
        '{"score": 9, "passed": true, "blocking_issues": [],}',
    ],
)
def test_one_schema_object_alone_or_in_one_fence_records_a_pass(
    tmp_path, monkeypatch, output
):
    rule = _candidate(tmp_path)
    monkeypatch.setattr(
        validator_pipeline, "run_claude_code", lambda *_a, **_k: (output, 0)
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)
    record = _record_for(result, rule, required=True)

    assert result.passed is True
    assert result.score == 9
    assert record["status"] == "passed"
    assert review_requirement_refusal(record) is None


def test_one_schema_object_with_a_failure_records_the_failure(tmp_path, monkeypatch):
    rule = _candidate(tmp_path)
    output = (
        '{"score": 2, "passed": false, "blocking_issues": ["wrong"], '
        '"non_blocking_issues": [], "reasoning": "no"}'
    )
    monkeypatch.setattr(
        validator_pipeline, "run_claude_code", lambda *_a, **_k: (output, 0)
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)
    record = _record_for(result, rule, required=True)

    assert result.score == 2
    assert record["status"] == "failed"
    assert review_requirement_refusal(record) is not None


_REVIEWER_TYPES = [
    "generalist-reviewer",
    "rulespec-reviewer",
    "Formula Reviewer",
    "Parameter Reviewer",
    "Integration Reviewer",
]
_SCHEMA_VALUES = {
    "score": 9,
    "passed": True,
    "issues": [],
    "blocking_issues": [],
    "non_blocking_issues": [],
    "reasoning": "ok",
}


def _requested_schema_keys(monkeypatch, tmp_path, reviewer_type):
    """The keys of the JSON object that this reviewer type's prompt asks for."""

    prompts = []

    def reviewer(prompt, **_kwargs):
        prompts.append(prompt)
        return "{}", 0

    monkeypatch.setattr(validator_pipeline, "run_claude_code", reviewer)
    _pipeline(tmp_path)._run_reviewer(reviewer_type, _candidate(tmp_path))
    (prompt,) = prompts
    blocks = re.findall(r"^\{\n((?:  \"[a-z_]+\": [^\n]+\n)+)\}$", prompt, re.MULTILINE)
    assert len(blocks) == 1
    return frozenset(re.findall(r'^  "([a-z_]+)": ', blocks[0], re.MULTILINE))


@pytest.mark.parametrize("reviewer_type", _REVIEWER_TYPES)
def test_accepted_verdict_keys_are_the_keys_that_the_prompt_asks_for(
    tmp_path, monkeypatch, reviewer_type
):
    requested = _requested_schema_keys(monkeypatch, tmp_path, reviewer_type)

    assert requested == validator_pipeline._reviewer_schema_keys(reviewer_type)
    assert {"passed", "score"} <= requested


@pytest.mark.parametrize("reviewer_type", _REVIEWER_TYPES)
def test_each_reviewer_type_reads_its_own_schema_and_no_other_key(
    tmp_path, monkeypatch, reviewer_type
):
    rule = _candidate(tmp_path)
    schema = validator_pipeline._reviewer_schema_keys(reviewer_type)
    own = {key: _SCHEMA_VALUES[key] for key in sorted(schema)}

    def run(payload):
        output = json.dumps(payload)
        monkeypatch.setattr(
            validator_pipeline, "run_claude_code", lambda *_a, **_k: (output, 0)
        )
        return _pipeline(tmp_path)._run_reviewer(reviewer_type, rule)

    accepted = run(own)
    assert accepted.passed is True
    assert accepted.score == 9
    assert "reviewer_parse_failed" not in accepted.issues
    for key in sorted(set(_SCHEMA_VALUES) - schema) + ["pass", "verdict", "Passed"]:
        refused = run({**own, key: _SCHEMA_VALUES.get(key, False)})
        assert refused.passed is False, key
        assert refused.score is None, key
        assert "reviewer_parse_failed" in refused.issues, key


def test_reviewer_schema_sets_are_exact():
    assert validator_pipeline._reviewer_schema_keys("generalist-reviewer") == {
        "score",
        "passed",
        "blocking_issues",
        "non_blocking_issues",
        "reasoning",
    }
    for reviewer_type in _REVIEWER_TYPES[1:] + ["any other reviewer"]:
        assert validator_pipeline._reviewer_schema_keys(reviewer_type) == {
            "score",
            "passed",
            "issues",
            "reasoning",
        }


@pytest.mark.parametrize(
    "output",
    [
        '{"score": 9, "passed": 1}',
        '{"score": 9, "passed": "true"}',
        '{"score": 9, "passed": null}',
        '{"score": "9", "passed": true}',
        '{"score": true, "passed": true}',
        '{"score": null, "passed": true}',
    ],
)
def test_one_schema_object_with_a_mistyped_verdict_field_never_records_a_pass(
    tmp_path, monkeypatch, output
):
    rule = _candidate(tmp_path)
    monkeypatch.setattr(
        validator_pipeline, "run_claude_code", lambda *_a, **_k: (output, 0)
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)
    record = _record_for(result, rule, required=True)

    assert "reviewer_parse_failed" in result.issues
    assert (record["status"], record["reason"]) == ("unavailable", "reviewer_error")


def test_reviewer_exception_never_records_a_pass(tmp_path, monkeypatch):
    rule = _candidate(tmp_path)

    def boom(*_a, **_k):
        raise OSError("EACCES: permission denied, open '.codex/config.toml'")

    monkeypatch.setattr(validator_pipeline, "run_claude_code", boom)

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)
    record = _record_for(result, rule)

    assert record["status"] == "unavailable"


def test_failed_review_is_recorded_as_failed(tmp_path, monkeypatch):
    rule = _candidate(tmp_path)
    monkeypatch.setattr(
        validator_pipeline,
        "run_claude_code",
        lambda *_a, **_k: (
            '{"score": 3.0, "passed": false, "blocking_issues": ["wrong and/or"]}',
            0,
        ),
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)
    record = _record_for(result, rule)

    assert record["status"] == "failed"
    assert record["verdict"] == {"passed": False, "score": 3.0, "issue_count": 1}


def test_reviewer_identity_names_the_cli_that_started(tmp_path, monkeypatch):
    rule = _candidate(tmp_path)

    def codex_runner(*_a, **_k):
        validator_pipeline._note_reviewer_cli_started("codex", "gpt-test")
        return ('{"score": 9.0, "passed": true}', 0)

    monkeypatch.setattr(validator_pipeline, "run_claude_code", codex_runner)
    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)

    assert result.details["review_evidence"]["reviewer_backend"] == "codex"
    assert result.details["review_evidence"]["reviewer_model"] == "gpt-test"

    # A later review whose runner starts nothing does not inherit that name.
    monkeypatch.setattr(
        validator_pipeline,
        "run_claude_code",
        lambda *_a, **_k: ('{"score": 9.0, "passed": true}', 0),
    )
    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)
    assert result.details["review_evidence"]["reviewer_backend"] is None


@pytest.mark.parametrize(
    ("reviewer_output", "status"),
    [
        ('{"score": 9.0, "passed": true, "blocking_issues": []}', "passed"),
        ('{"score": 3.0, "passed": false, "blocking_issues": ["x"]}', "failed"),
        ("reviewer_unavailable: Reviewer CLIs not found", "unavailable"),
    ],
)
def test_artifact_evaluation_carries_the_review_evidence_to_the_record(
    tmp_path, reviewer_output, status
):
    from axiom_encode.harness.evals import evaluate_artifact
    from axiom_encode.harness.validator_pipeline import (
        ValidationResult,
        ValidatorPipeline,
    )
    from tests.test_evals import (
        _canonical_rulespec_content_root,
        _generated_rulespec_file_path,
        _write_test_corpus_provision,
    )

    rulespec_file = _generated_rulespec_file_path(tmp_path, "statutes/24/a.yaml")
    rulespec_file.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  summary: Source text says the amount is $1,000.\n"
        "rules:\n"
        "  - name: ctc_amount\n"
        "    kind: parameter\n"
        "    dtype: Money\n"
        "    unit: USD\n"
        "    versions:\n"
        "      - effective_from: '2018-01-01'\n"
        "        formula: 1000\n"
    )
    returncode = 0 if reviewer_output.startswith("{") else 1
    with (
        patch.object(
            ValidatorPipeline,
            "_run_compile_check",
            return_value=ValidationResult("compile", passed=True),
        ),
        patch.object(
            ValidatorPipeline,
            "_run_ci",
            return_value=ValidationResult("ci", passed=True),
        ),
        patch.object(
            validator_pipeline,
            "run_claude_code",
            return_value=(reviewer_output, returncode),
        ) as reviewer_cli,
    ):
        metrics = evaluate_artifact(
            local_corpus_release=_write_test_corpus_provision(
                tmp_path / "bound-release"
            ),
            rulespec_file=rulespec_file,
            policy_repo_root=_canonical_rulespec_content_root(tmp_path, "us"),
            axiom_rules_path=Path("/tmp/axiom-rules-engine"),
            source_text="Source text says the amount is $1,000.",
            reviewers_require_deterministic_pass=True,
        )

    reviewer_cli.assert_called_once()
    state = cli._encode_generalist_review_state(
        SimpleNamespace(metrics=metrics), skipped_by_flag=False, required=True
    )
    record = build_generalist_review_record(
        state,
        cli._review_applied_pairs([rulespec_file], manifest_root=rulespec_file.parent),
        reviewed_rulespec_path=rulespec_file.name,
    )
    assert record["status"] == status
    if status == "unavailable":
        assert getattr(metrics, GENERALIST_REVIEW_EVIDENCE_ATTR, None) is None
    else:
        # The reviewer read a staged copy; its digest is the candidate's.
        assert record["reviewed"]["rulespec_sha256"] == _sha(rulespec_file)
        assert record["prompt_sha256"] == metrics.generalist_review_prompt_sha256
    assert (review_requirement_refusal(record) is None) is (status == "passed")


def test_review_evidence_is_not_a_persisted_metric_field():
    from dataclasses import asdict

    metrics = _metrics_from(
        SimpleNamespace(
            passed=True,
            score=9.0,
            issues=[],
            prompt_sha256=PROMPT_SHA,
            details={"review_evidence": _evidence()},
        )
    )

    assert getattr(metrics, GENERALIST_REVIEW_EVIDENCE_ATTR) == _evidence()
    assert not any("evidence" in key for key in asdict(metrics))


# --- the signed manifest -------------------------------------------------


def _write_manifest(
    tmp_path,
    *,
    per_file,
    review_state,
    companion=True,
    admits_record=True,
    program_bytes=b"program: us-sc/snap\nperiod: 2026-01\n",
    test_bytes=b"[]\n",
    orphan_test=False,
    orphan_bytes=b"- name: orphan\n",
    twin_modules=(),
    results=None,
):
    checkout = tmp_path / "rulespec-us"
    (checkout / "us").mkdir(parents=True)
    relative_output = Path("programs/us-sc/snap/fy-2026.yaml")
    program = checkout / relative_output
    program.parent.mkdir(parents=True)
    program.write_bytes(program_bytes)
    applied = [program]
    if companion:
        test_file = program.with_name("fy-2026.test.yaml")
        test_file.write_bytes(test_bytes)
        applied.append(test_file)
    if orphan_test:
        # A companion test applied without its RuleSpec is in no pair.
        orphan = program.with_name("other.test.yaml")
        orphan.write_bytes(orphan_bytes)
        applied.append(orphan)
    for position, name in twin_modules:
        # Another module with the bytes of this encode's module and test.
        twin = program.with_name(f"{name}.yaml")
        twin.write_bytes(program_bytes)
        twin_test = program.with_name(f"{name}.test.yaml")
        twin_test.write_bytes(test_bytes)
        if position == "before":
            applied[:0] = [twin, twin_test]
        else:
            applied += [twin, twin_test]
    output_root = tmp_path / "out"
    generated = output_root / "openai-test" / relative_output
    generated.parent.mkdir(parents=True)
    generated.write_text(program.read_text())
    result = SimpleNamespace(
        output_file=str(generated),
        context_manifest_file=None,
        trace_file=None,
        generation_prompt_sha256=None,
        tool=APPLIED_ENCODING_MODEL_TOOL,
        citation=_rulespec_anchor_base_for_output(checkout, relative_output),
        runner="openai",
        backend="openai",
        model="review-record-test",
        source_attestation={},
    )
    setattr(
        result,
        _APPLY_VALIDATION_SNAPSHOT_ATTR,
        {
            "manifest_validation_execution": {
                "axiom_encode": dict(TEST_PINNED_ENCODER_IDENTITY)
            }
        },
    )
    if review_state is not None:
        setattr(result, _GENERALIST_REVIEW_STATE_ATTR, review_state(applied))
    if results is not None:
        results.append(result)
    with (
        patch(
            "axiom_encode.cli.load_rulespec_local_corpus_release",
            return_value=object(),
        ),
        patch(
            "axiom_encode.cli._resolver_owned_manifest_source_attestation",
            return_value={"rulespec_root": "rulespec-us/us-sc"},
        ),
        patch(
            "axiom_encode.cli.verify_rulespec_validation_waiver_set",
            return_value="b" * 64,
        ),
        patch(
            "axiom_encode.cli._target_uses_per_file_waiver_bindings",
            return_value=per_file,
        ),
        patch(
            "axiom_encode.cli._target_admits_generalist_review_record",
            return_value=admits_record,
        ),
        patch(
            "axiom_encode.cli._build_applied_manifest_file_waiver_bindings",
            return_value={"schema": "test", "files": []},
        ),
    ):
        manifest = _write_applied_encoding_manifest(
            result,
            output_root=output_root,
            policy_repo_path=checkout,
            corpus_path=tmp_path / "axiom-corpus",
            relative_output=relative_output,
            applied_files=applied,
            axiom_encode_git={"commit": "a" * 40},
            signing_broker=TEST_APPLY_SIGNING_BROKER,
        )
    return json.loads(manifest.read_text()), applied


def _bound_state(reviewed_rulespec=None, reviewed_test=None, **overrides):
    """A review of the applied files, or of the given bytes in their place."""

    def build(applied):
        applied_test = applied[1].read_bytes() if len(applied) > 1 else None
        evidence = _evidence(
            rulespec=(
                applied[0].read_bytes()
                if reviewed_rulespec is None
                else reviewed_rulespec
            ),
            test=applied_test if reviewed_test is None else reviewed_test,
        )
        return _state(**{"evidence": evidence, **overrides})

    return build


def _schema_issues(payload):
    return cli._applied_manifest_exact_schema_issues(payload, manifest_label="manifest")


@pytest.mark.parametrize(
    ("overrides", "status"),
    [
        ({}, "passed"),
        ({"metrics_pass": False, "score": 2.0}, "failed"),
        ({"metrics_pass": False, "score": None, "evidence": None}, "unavailable"),
        ({"skipped_by_flag": True}, "skipped"),
    ],
)
def test_manifest_records_each_review_status_under_the_signature(
    tmp_path, overrides, status
):
    payload, applied = _write_manifest(
        tmp_path, per_file=True, review_state=_bound_state(**overrides)
    )

    record = payload[GENERALIST_REVIEW_FIELD]
    assert record["status"] == status
    assert record["required"] is False
    assert (
        generalist_review_record_issues(
            record, [item["sha256"] for item in payload["applied_files"]]
        )
        == []
    )
    # The record is inside the signed bytes: changing it breaks the signature.
    assert (
        cli._applied_encoding_manifest_signature_issue(
            payload, TEST_APPLY_SIGNING_BROKER
        )
        is None
    )
    forged = copy.deepcopy(payload)
    forged[GENERALIST_REVIEW_FIELD] = build_generalist_review_record(
        _bound_state()(applied),
        cli._review_applied_pairs(applied, manifest_root=tmp_path / "rulespec-us"),
        reviewed_rulespec_path=payload["applied_files"][0]["path"],
    )
    if status != "passed":
        assert forged[GENERALIST_REVIEW_FIELD]["status"] == "passed"
        # A well-formed "passed" record, so only the signature can refuse it.
        assert (
            generalist_review_record_issues(
                forged[GENERALIST_REVIEW_FIELD],
                [(item["path"], item["sha256"]) for item in payload["applied_files"]],
            )
            == []
        )
        assert (
            cli._applied_encoding_manifest_signature_issue(
                forged, TEST_APPLY_SIGNING_BROKER
            )
            is not None
        )


def test_manifest_pass_is_partial_when_an_applied_file_was_not_reviewed(tmp_path):
    payload, applied = _write_manifest(
        tmp_path, per_file=True, orphan_test=True, review_state=_bound_state()
    )

    record = payload[GENERALIST_REVIEW_FIELD]
    assert (record["status"], record["reason"]) == (
        "partial",
        "unreviewed_applied_files",
    )
    assert {
        "path": "programs/us-sc/snap/other.test.yaml",
        "sha256": _sha(applied[2]),
        "reviewed": False,
        "complete": False,
    } in record["files"]
    assert [(entry["path"], entry["sha256"]) for entry in record["files"]] == sorted(
        (item["path"], item["sha256"]) for item in payload["applied_files"]
    )
    with pytest.raises(RuntimeError, match="unreviewed_applied_files"):
        _write_manifest(
            tmp_path / "required",
            per_file=True,
            orphan_test=True,
            review_state=_bound_state(required=True),
        )


@pytest.mark.parametrize("admits_record", [True, False])
@pytest.mark.parametrize("required", [False, True])
def test_writer_hands_the_record_of_the_signed_files_to_the_encode_outcome(
    tmp_path, admits_record, required
):
    # The outcome record is built before apply finalizes its files. The
    # writer's record is the one about the files that are applied, also for a
    # target whose manifest cannot carry it and when the requirement refuses.
    results = []
    arguments = {
        "per_file": True,
        "orphan_test": True,
        "admits_record": admits_record,
        "review_state": _bound_state(required=required),
        "results": results,
    }
    if required:
        with pytest.raises(RuntimeError, match="unreviewed_applied_files"):
            _write_manifest(tmp_path, **arguments)
        applied_sha256s = None
    else:
        payload, applied = _write_manifest(tmp_path, **arguments)
        assert (GENERALIST_REVIEW_FIELD in payload) is admits_record
        applied_sha256s = [item["sha256"] for item in payload["applied_files"]]

    record = vars(results[0])["_axiom_generalist_review_applied_record"]
    assert (record["status"], record["reason"], record["required"]) == (
        "partial",
        "unreviewed_applied_files",
        required,
    )
    assert len(record["files"]) == 3
    if applied_sha256s is not None:
        assert _issues(record, applied_sha256s) == []
    if admits_record and not required:
        assert record == payload[GENERALIST_REVIEW_FIELD]


def test_manifest_binds_a_review_made_before_the_source_stamp(tmp_path):
    # The real order: the reviewer reads the candidate, apply stamps it.
    payload, applied = _write_manifest(
        tmp_path,
        per_file=True,
        program_bytes=STAMPED_RULE_BYTES,
        test_bytes=TEST_BYTES,
        review_state=_bound_state(reviewed_rulespec=RULE_BYTES, required=True),
    )

    record = payload[GENERALIST_REVIEW_FIELD]
    assert record["status"] == "passed"
    assert record["reviewed"]["rulespec_sha256"] == PAIR.rulespec_sha256
    assert record["applied"]["rulespec_sha256"] == _sha(applied[0])
    assert record["applied"]["test_sha256"] == _sha(applied[1])
    assert record["reviewed"]["rulespec_sha256"] != record["applied"]["rulespec_sha256"]
    assert (
        generalist_review_record_issues(
            record, [item["sha256"] for item in payload["applied_files"]]
        )
        == []
    )


@pytest.mark.parametrize(
    ("program_bytes", "test_bytes"),
    [
        (REPAIRED_RULE_BYTES, TEST_BYTES),
        (STAMPED_RULE_BYTES, TEST_BYTES + b"- name: added\n"),
        (b"a: [unclosed\n", TEST_BYTES),
    ],
)
def test_manifest_verdict_on_content_that_is_not_applied_is_unavailable(
    tmp_path, program_bytes, test_bytes
):
    payload, _applied = _write_manifest(
        tmp_path,
        per_file=True,
        program_bytes=program_bytes,
        test_bytes=test_bytes,
        review_state=_bound_state(
            reviewed_rulespec=RULE_BYTES, reviewed_test=TEST_BYTES
        ),
    )

    record = payload[GENERALIST_REVIEW_FIELD]
    assert record["status"] == "unavailable"
    assert record["reason"] == "reviewed_content_not_applied"
    assert record["verdict"]["passed"] is True
    assert record["applied"] == UNBOUND_APPLIED


@pytest.mark.parametrize(
    ("per_file", "fields"),
    [
        (False, "_MODEL_APPLY_MANIFEST_FIELDS"),
        (True, "_MODEL_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS"),
    ],
)
def test_manifest_for_a_pinned_or_held_verifier_keeps_todays_exact_shape(
    tmp_path, per_file, fields
):
    payload, _applied = _write_manifest(
        tmp_path, per_file=per_file, admits_record=False, review_state=_bound_state()
    )

    assert GENERALIST_REVIEW_FIELD not in payload
    assert frozenset(payload) == getattr(cli, fields)


@pytest.mark.parametrize(
    ("per_file", "fields"),
    [
        (False, "_MODEL_APPLY_REVIEW_RECORD_MANIFEST_FIELDS"),
        (True, "_MODEL_APPLY_PER_FILE_WAIVER_REVIEW_RECORD_MANIFEST_FIELDS"),
    ],
)
def test_manifest_carries_the_record_on_either_waiver_shape(tmp_path, per_file, fields):
    # The record gate is separate from the waiver shape (#2097).
    payload, _applied = _write_manifest(
        tmp_path, per_file=per_file, admits_record=True, review_state=_bound_state()
    )

    assert frozenset(payload) == getattr(cli, fields)
    assert not [issue for issue in _schema_issues(payload) if "exact" in issue]
    assert not [
        issue for issue in _schema_issues(payload) if "generalist_review" in issue
    ]


@pytest.mark.parametrize("admits_record", [False, True])
def test_required_review_is_enforced_whether_or_not_the_record_is_written(
    tmp_path, admits_record
):
    with pytest.raises(
        RuntimeError, match="passed generalist review of every applied file is required"
    ):
        _write_manifest(
            tmp_path,
            per_file=True,
            admits_record=admits_record,
            review_state=_bound_state(required=True, metrics_pass=False, score=2.0),
        )


def test_manifest_without_review_state_is_unchanged(tmp_path):
    for per_file, fields in (
        (False, cli._MODEL_APPLY_MANIFEST_FIELDS),
        (True, cli._MODEL_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS),
    ):
        payload, _applied = _write_manifest(
            tmp_path / str(per_file), per_file=per_file, review_state=None
        )
        assert frozenset(payload) == fields


@pytest.mark.parametrize(
    "overrides",
    [
        {"metrics_pass": False, "score": 2.0},
        {"metrics_pass": False, "score": None, "evidence": None},
        {"metrics_pass": True, "score": None, "prompt_sha256": None, "evidence": None},
        {"skipped_by_flag": True},
        {"metrics_pass": None},
    ],
)
@pytest.mark.parametrize("per_file", [False, True])
def test_writer_refuses_to_sign_a_required_review_that_did_not_pass(
    tmp_path, overrides, per_file
):
    with pytest.raises(
        RuntimeError, match="passed generalist review of every applied file is required"
    ):
        _write_manifest(
            tmp_path,
            per_file=per_file,
            review_state=_bound_state(required=True, **overrides),
        )

    assert not list((tmp_path / "rulespec-us").rglob("*.json"))


def test_writer_refuses_a_required_pass_on_bytes_that_are_not_applied(tmp_path):
    with pytest.raises(RuntimeError, match="reviewed_content_not_applied"):
        _write_manifest(
            tmp_path,
            per_file=True,
            review_state=lambda _applied: _state(required=True),
        )

    assert not list((tmp_path / "rulespec-us").rglob("*.json"))


def test_writer_signs_a_required_review_that_passed(tmp_path):
    payload, _applied = _write_manifest(
        tmp_path, per_file=True, review_state=_bound_state(required=True)
    )

    assert payload[GENERALIST_REVIEW_FIELD]["status"] == "passed"
    assert payload[GENERALIST_REVIEW_FIELD]["required"] is True


def test_writer_refuses_a_malformed_review_state(tmp_path):
    with pytest.raises(RuntimeError, match="review state is malformed"):
        _write_manifest(
            tmp_path, per_file=True, review_state=lambda _applied: {"status": "passed"}
        )


# --- the manifest reader -------------------------------------------------


def test_reader_admits_the_record_and_still_admits_manifests_without_it(tmp_path):
    with_record, _ = _write_manifest(
        tmp_path / "a", per_file=True, review_state=_bound_state()
    )
    per_file_only, _ = _write_manifest(tmp_path / "b", per_file=True, review_state=None)
    set_digest, _ = _write_manifest(tmp_path / "c", per_file=False, review_state=None)

    def exact(payload):
        return [issue for issue in _schema_issues(payload) if "exact" in issue]

    def unrelated(payload):
        # The fixture's placeholder provenance has its own findings; they are
        # the same with and without the record.
        return sorted(
            issue.replace(payload["generated_at"], "")
            for issue in _schema_issues(payload)
        )

    # Manifests written before this change keep verifying exactly as before.
    assert exact(set_digest) == []
    assert exact(per_file_only) == []
    assert exact(with_record) == []
    assert unrelated(with_record) == unrelated(per_file_only)
    assert not any(
        "generalist_review" in issue for issue in _schema_issues(with_record)
    )


def test_reader_variants_combine_review_and_companion_preservation():
    variants = cli._APPLIED_MANIFEST_SCHEMA_VARIANTS
    record = {GENERALIST_REVIEW_FIELD}
    model_shapes = (
        cli._MODEL_APPLY_MANIFEST_FIELDS,
        cli._MODEL_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS,
        cli._MODEL_APPLY_MANIFEST_FIELDS | record,
        cli._MODEL_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS | record,
    )
    codex_shapes = (
        cli._MODEL_APPLY_MANIFEST_FIELDS,
        cli._MODEL_CODEX_APPLY_MANIFEST_FIELDS,
        cli._MODEL_CODEX_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS,
        cli._MODEL_CODEX_APPLY_MANIFEST_FIELDS | record,
        cli._MODEL_CODEX_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS | record,
    )
    preservation = {"repair_companion_preservation"}
    assert variants["model"] == model_shapes + tuple(
        shape | preservation for shape in model_shapes
    )
    assert variants["codex model"] == codex_shapes + tuple(
        shape | preservation for shape in codex_shapes
    )
    # No other contract admits the field.
    for contract, shapes in variants.items():
        if contract not in {"model", "codex model"}:
            assert all(GENERALIST_REVIEW_FIELD not in shape for shape in shapes)


def test_reader_refuses_the_record_with_both_waiver_fields(tmp_path):
    payload, _ = _write_manifest(tmp_path, per_file=False, review_state=_bound_state())
    payload["validation_waiver_bindings"] = {"schema": "test", "files": []}

    assert any(
        "does not match the exact model v5 schema" in issue
        for issue in _schema_issues(payload)
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "failed"),
        (("verdict", "score"), 2.0),
        (("applied", "rulespec_content_sha256"), OTHER_SHA),
        (("applied", "rulespec_sha256"), OTHER_SHA),
        (("applied", "test_content_sha256"), OTHER_SHA),
        (("required",), "no"),
    ],
)
def test_reader_refuses_a_manifest_whose_record_is_inconsistent(tmp_path, path, value):
    payload, _ = _write_manifest(tmp_path, per_file=True, review_state=_bound_state())
    baseline = _schema_issues(payload)
    payload[GENERALIST_REVIEW_FIELD] = _tampered(
        payload[GENERALIST_REVIEW_FIELD], path, value
    )

    added = [issue for issue in _schema_issues(payload) if issue not in baseline]
    assert added and all("manifest generalist_review" in issue for issue in added)


@pytest.mark.parametrize(
    ("path", "value", "check"),
    [
        (("status",), "approved", "manifest:generalist-review/shape"),
        (("verdict", "passed"), False, "manifest:generalist-review/verdict"),
        (
            ("applied", "rulespec_sha256"),
            OTHER_SHA,
            "manifest:generalist-review/binding",
        ),
    ],
)
def test_reader_findings_are_typed_issues_of_declared_checks(
    tmp_path, path, value, check
):
    from axiom_encode import checks

    payload, _ = _write_manifest(tmp_path, per_file=True, review_state=_bound_state())
    baseline = _schema_issues(payload)
    payload[GENERALIST_REVIEW_FIELD] = _tampered(
        payload[GENERALIST_REVIEW_FIELD], path, value
    )

    added = [issue for issue in _schema_issues(payload) if issue not in baseline]
    assert [issue.check for issue in added] == [check]
    assert added[0].startswith("manifest generalist_review ")
    assert check in checks.load_checks()


def test_reader_refuses_a_pass_bound_to_a_file_the_manifest_does_not_apply(tmp_path):
    payload, _ = _write_manifest(tmp_path, per_file=True, review_state=_bound_state())
    baseline = _schema_issues(payload)
    # Swap the applied digest: the record now covers bytes that are not applied.
    payload["applied_files"][0]["sha256"] = OTHER_SHA

    added = [issue for issue in _schema_issues(payload) if issue not in baseline]
    assert any(
        "does not list exactly the applied file digests" in issue for issue in added
    )


def test_unreceipted_retirement_scan_knows_the_record_shapes():
    classes = cli._UNRECEIPTED_RETIREMENT_MANIFEST_FIELD_CLASSES
    for old, new in (
        ("_MODEL_APPLY_MANIFEST_FIELDS", "_MODEL_APPLY_REVIEW_RECORD_MANIFEST_FIELDS"),
        (
            "_MODEL_CODEX_APPLY_MANIFEST_FIELDS",
            "_MODEL_CODEX_APPLY_REVIEW_RECORD_MANIFEST_FIELDS",
        ),
        (
            "_MODEL_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS",
            "_MODEL_APPLY_PER_FILE_WAIVER_REVIEW_RECORD_MANIFEST_FIELDS",
        ),
        (
            "_MODEL_CODEX_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS",
            "_MODEL_CODEX_APPLY_PER_FILE_WAIVER_REVIEW_RECORD_MANIFEST_FIELDS",
        ),
    ):
        old_classes, new_classes = (
            classes[getattr(cli, old)],
            classes[getattr(cli, new)],
        )
        assert new_classes[0] == old_classes[0]
        assert new_classes[1] == old_classes[1] | {GENERALIST_REVIEW_FIELD}


# --- encode settings (the --apply runs are in tests/test_cli.py) --------


class TestEncodeRequireReviewSettings:
    def _args(self, **values):
        return SimpleNamespace(
            backend="codex",
            apply=values.get("apply", True),
            require_review=values.get("require_review", False),
            skip_reviewers=values.get("skip_reviewers", False),
        )

    def test_flag_without_apply_is_refused_before_any_work(self):
        with (
            patch("axiom_encode.cli._local_verification_encode_roots") as later,
            pytest.raises(ValueError, match="--require-review requires --apply"),
        ):
            cmd_encode(self._args(apply=False, require_review=True))
        later.assert_not_called()

    def test_require_and_skip_together_are_refused(self):
        with (
            patch("axiom_encode.cli._local_verification_encode_roots") as later,
            pytest.raises(ValueError, match="cannot both require"),
        ):
            cmd_encode(self._args(require_review=True, skip_reviewers=True))
        later.assert_not_called()

    def test_flag_is_the_only_switch(self, monkeypatch):
        # The signing supervisor builds the child environment from empty, so
        # an environment switch would be dropped in the signed lanes.
        monkeypatch.setenv("AXIOM_ENCODE_REQUIRE_REVIEW", "1")
        assert cli._encode_requires_review(self._args()) is False
        assert cli._encode_requires_review(self._args(require_review=True)) is True
        assert not hasattr(record_module, "REQUIRE_REVIEW_ENV")

    def test_default_is_off_and_only_true_turns_it_on(self):
        assert cli._encode_requires_review(self._args()) is False
        for value in ("yes", 1, None):
            assert (
                cli._encode_requires_review(self._args(require_review=value)) is False
            )


def test_module_states_only_five_statuses():
    assert record_module.STATUSES == {
        "passed",
        "partial",
        "failed",
        "unavailable",
        "skipped",
    }


def test_blocked_review_is_a_failed_iteration_in_the_run_log():
    refusal = review_requirement_refusal(_build(metrics_pass=False, score=2.0))
    result = SimpleNamespace(success=True, error=None)

    assert cli._encode_iteration_verdict(
        result, {"status": "apply_blocked_review", "apply_error": refusal}
    ) == (False, refusal)
    # The statuses around it keep their meaning.
    assert cli._encode_iteration_verdict(result, {"status": "apply_applied"}) == (
        True,
        None,
    )


# --- binding across the dump that apply performs -------------------------


def _as_apply_rewrites(raw: bytes) -> bytes:
    """What `_stamp_generated_source_attestation_for_apply` writes for `raw`."""

    payload = yaml.safe_load(raw.decode())
    module = payload.setdefault("module", {})
    verification = module.setdefault("source_verification", {})
    verification["corpus_citation_path"] = "us/statute/26/24"
    verification["source_sha256"] = "5" * 64
    return apply_stamp_dump(payload).encode()


def _rule(value: str) -> bytes:
    return (
        "format: rulespec/v1\nrules:\n  - name: rate\n    value: " + value + "\n"
    ).encode()


MERGE_RULE = b"""format: rulespec/v1
base: &base
  dtype: Money
  unit: USD
rules:
  - name: amount
    <<: *base
    value: 5
"""


@pytest.mark.parametrize(
    "value",
    # Not a plain float that every reader reads as one value: an integer, a
    # float that only one dialect types, or a string that holds a number.
    [
        "1_000",
        "+5",
        "yes",
        "0x10",
        "010",
        "1:00",
        "1_0.50",
        "1:30.50",
        "!!float 0.10",
    ],
)
def test_scalar_respelling_is_not_an_alternate_review_binding(value):
    reviewed = _rule(value)
    applied = _as_apply_rewrites(reviewed)
    # The dump changes the text of this scalar, so the plain digests differ.
    assert review_content_sha256(applied, rulespec=True) != (
        review_content_sha256(reviewed, rulespec=True)
    )

    record = build_generalist_review_record(
        _state(evidence=_evidence(rulespec=reviewed), required=True),
        [applied_pair(applied, TEST_BYTES)],
    )

    assert record["status"] == "unavailable"
    assert record["reason"] == "reviewed_content_not_applied"
    assert record["reviewed"] == _reviewed(
        applied_pair(reviewed, TEST_BYTES), rulespec=reviewed
    )
    assert record["reviewed"]["rulespec_dumped_content_sha256"] is None
    assert review_requirement_refusal(record) is not None


@pytest.mark.parametrize("value", ["1e5", "-0o17", "+0o17", "0o17"])
def test_plain_yaml12_number_cannot_become_a_quoted_string(value):
    raw = _rule(value)
    quoted = _rule(json.dumps(value))
    assert review_content_sha256(raw, rulespec=True) != review_content_sha256(
        quoted, rulespec=True
    )
    rewritten = _as_apply_rewrites(raw)
    assert dumped_review_content_sha256(raw) is None
    record = build_generalist_review_record(
        _state(evidence=_evidence(rulespec=raw), required=True),
        [applied_pair(rewritten, TEST_BYTES)],
    )
    assert record["status"] == "unavailable"
    assert review_requirement_refusal(record) is not None


def test_explicit_string_tag_stays_equivalent_to_quoted_string():
    assert review_content_sha256(b"a: !!str 1e5\n", rulespec=True) == (
        review_content_sha256(b"a: '1e5'\n", rulespec=True)
    )


@pytest.mark.parametrize("value", ["1e5", "0o17"])
def test_explicit_tag_cannot_impersonate_an_internal_scalar_type_marker(value):
    raw = f"a: {value}\n".encode()
    tagged = f"a: !<axiom-review:implicit-yaml12-non-string> '{value}'\n".encode()
    for rulespec in (True, False):
        assert review_content_sha256(raw, rulespec=rulespec) != review_content_sha256(
            tagged, rulespec=rulespec
        )


@pytest.mark.parametrize(
    "left,right",
    [
        ("yes", "YES"),
        ("on", "ON"),
        ("off", "OFF"),
        ("no", "NO"),
        ("yes", "!!bool yes"),
        ("1:00", "!!int 1:00"),
        ("2018-01-01", "!!timestamp 2018-01-01"),
        ("!!null garbage", "null"),
    ],
)
def test_yaml11_implicit_typing_cannot_hide_changed_companion_content(left, right):
    reviewed = f"- name: case\n  input: {{status: {left}}}\n  output: {{credit_amount: 1}}\n".encode()
    applied = f"- name: case\n  input: {{status: {right}}}\n  output: {{credit_amount: 1}}\n".encode()
    original = applied_pair(RULE_BYTES, reviewed)
    changed = applied_pair(RULE_BYTES, applied)
    assert original.test_content_sha256 != changed.test_content_sha256
    record = build_generalist_review_record(
        _state(evidence=_evidence(test=reviewed), required=True), [changed]
    )
    assert record["status"] == "unavailable"
    assert review_requirement_refusal(record) is not None
    forged = build_generalist_review_record(
        _state(evidence=_evidence(test=reviewed), required=True), [original]
    )
    forged["applied"] = _digests(changed)
    forged["files"] = _files(
        (changed.rulespec_sha256, True, True), (changed.test_sha256, True, True)
    )
    assert _issues(forged)


@pytest.mark.parametrize("keys", [("1", "1.0"), ("yes", "true")])
def test_loader_key_collapse_cannot_bind_a_review_through_the_dump(keys):
    raw = RULE_BYTES + f"extra:\n  {keys[0]}: first\n  {keys[1]}: second\n".encode()
    rewritten = _as_apply_rewrites(raw)
    assert dumped_review_content_sha256(raw) is None
    pair = applied_pair(rewritten, TEST_BYTES)
    evidence = _evidence(rulespec=raw)
    # Even a claimed alternate digest from a producer cannot widen binding.
    evidence["reviewed_rulespec_dumped_content_sha256"] = pair.rulespec_content_sha256
    record = build_generalist_review_record(
        _state(evidence=evidence, required=True), [pair]
    )
    assert record["status"] == "unavailable"
    assert review_requirement_refusal(record) is not None


@pytest.mark.parametrize(
    "left,right",
    [
        (b"a: !!set {x: null}\n", b"a: {x: null}\n"),
        (b"!!set {}\n", b"{}\n"),
        (b"a: !!omap [x: 1]\n", b"a: [x: 1]\n"),
        (b"module: !!set {source_verification: {source_sha256: abc}}\n", b"{}\n"),
        (b"module: {source_verification: !!set {source_sha256: abc}}\n", b"{}\n"),
    ],
)
def test_collection_tags_are_reviewed_content(left, right):
    assert review_content_sha256(left, rulespec=True) != review_content_sha256(
        right, rulespec=True
    )


def test_reader_refuses_a_forged_alternate_dump_binding():
    record = _build()
    record["reviewed"]["rulespec_content_sha256"] = OTHER_SHA
    record["reviewed"]["rulespec_dumped_content_sha256"] = PAIR.rulespec_content_sha256
    assert any("differs" in issue for issue in _issues(record))


def test_reader_refuses_only_a_forged_dump_digest():
    record = _build(required=True)
    assert _issues(record) == []
    record["reviewed"]["rulespec_dumped_content_sha256"] = OTHER_SHA

    assert _issues(record) == [
        "generalist_review dump digest differs from reviewed content"
    ]


def test_builder_discards_only_a_forged_evidence_dump_digest():
    evidence = _evidence()
    evidence["reviewed_rulespec_dumped_content_sha256"] = OTHER_SHA
    record = _build(evidence=evidence, required=True)

    assert record["status"] == "passed"
    assert record["reviewed"]["rulespec_dumped_content_sha256"] is None
    assert _issues(record) == []
    # Every field except the diagnostic dump digest retains the good record.
    expected = _build(required=True)
    expected["reviewed"]["rulespec_dumped_content_sha256"] = None
    assert record == expected


@pytest.mark.parametrize(
    "value",
    [
        '! "10"',
        "! '10'",
        '! "true"',
        "! 'false'",
        '! ""',
        "! ''",
        "! |-\n      true",
        "! >-\n      10",
    ],
)
def test_styled_nonspecific_tag_type_change_cannot_bind(value, tmp_path):
    reviewed = _rule(value)
    rule = tmp_path / "rule.yaml"
    rule.write_bytes(reviewed)
    attestation = {
        "requested_corpus_citation_path": "us/statute/26/24",
        "resolved_corpus_citation_path": "us/statute/26/24",
        "source_sha256": "5" * 64,
    }
    with (
        patch.object(
            cli, "_generated_result_source_attestation", return_value=attestation
        ),
        patch.object(cli, "_source_attestation_structure_issues", return_value=[]),
    ):
        cli._stamp_generated_source_attestation_for_apply(
            SimpleNamespace(backend="openai"), rule
        )
    applied = rule.read_bytes()
    assert applied == _as_apply_rewrites(reviewed)
    assert review_content_sha256(reviewed, rulespec=True) is not None
    assert dumped_review_content_sha256(reviewed) is None
    assert review_content_sha256(reviewed, rulespec=True) != review_content_sha256(
        applied, rulespec=True
    )

    record = _build(
        applied=[applied_pair(applied, TEST_BYTES)],
        evidence=_evidence(rulespec=reviewed),
        required=True,
    )
    assert (record["status"], record["reason"]) == (
        "unavailable",
        "reviewed_content_not_applied",
    )
    assert review_requirement_refusal(record) is not None


@pytest.mark.parametrize("key", ['! "10"', "! 'true'"])
def test_styled_nonspecific_mapping_key_cannot_bind_after_type_change(key):
    reviewed = RULE_BYTES + f"extra: {{{key}: first}}\n".encode()
    applied = _as_apply_rewrites(reviewed)
    record = _build(
        applied=[applied_pair(applied, TEST_BYTES)],
        evidence=_evidence(rulespec=reviewed),
        required=True,
    )
    assert record["status"] == "unavailable"
    assert record["reason"] == "reviewed_content_not_applied"
    assert review_requirement_refusal(record) is not None


def test_styled_nonspecific_tag_cannot_hide_duplicate_string_keys():
    reviewed = RULE_BYTES + b"extra: {'1': first, ! '1': second}\n"
    assert review_content_sha256(reviewed, rulespec=True) is None
    record = _build(
        applied=[applied_pair(_as_apply_rewrites(reviewed), TEST_BYTES)],
        evidence=_evidence(rulespec=reviewed),
        required=True,
    )
    assert record["status"] == "unavailable"
    assert record["reason"] == "unreadable_review_result"
    assert review_requirement_refusal(record) is not None


@pytest.mark.parametrize("value", ["! 10", "! true", "10", "true", "'10'", "''"])
def test_plain_nonspecific_and_ordinary_scalar_bindings_still_pass(value):
    reviewed = _rule(value)
    applied = _as_apply_rewrites(reviewed)
    record = _build(
        applied=[applied_pair(applied, TEST_BYTES)],
        evidence=_evidence(rulespec=reviewed),
        required=True,
    )
    assert record["status"] == "passed"
    assert review_requirement_refusal(record) is None
    assert _issues(record) == []


def test_oversized_integer_score_fails_closed_in_producer_and_reader():
    huge = 10**400
    assert _build(score=huge)["status"] == "unavailable"
    record = _build()
    record["verdict"]["score"] = huge
    assert _issues(record)


@pytest.mark.parametrize("raw", [b"!!null not-null\n", b"module: !!null not-null\n"])
def test_invalid_explicit_null_is_never_an_empty_document_or_stamp(raw):
    for rulespec in (True, False):
        assert review_content_sha256(raw, rulespec=rulespec) != review_content_sha256(
            b"", rulespec=rulespec
        )


REPEATED_KEY_RULE = b"""format: rulespec/v1
rules:
  - name: amount
    value: 100
    value: 5000
"""
SCALAR_MERGE_RULE = b"""format: rulespec/v1
rules:
  - name: amount
    <<: {dtype: Money}
    value: 5
"""


@pytest.mark.parametrize(
    "reviewed",
    [REPEATED_KEY_RULE, SCALAR_MERGE_RULE, MERGE_RULE],
    ids=["repeated-key", "inline-merge-key", "aliased-merge-key"],
)
def test_ambiguous_reviewed_file_is_never_bound_by_its_dump(reviewed):
    # The reviewer read a file a reader can resolve in more than one way; the
    # dump silently picks one reading (the last `value`, the merged mapping).
    assert review_content_sha256(reviewed, rulespec=True) is None
    assert yaml.safe_load(reviewed) is not None
    applied = _as_apply_rewrites(reviewed)
    evidence = _evidence(rulespec=reviewed)

    record = build_generalist_review_record(
        _state(evidence=evidence, required=True),
        [applied_pair(applied, TEST_BYTES)],
    )

    assert record["status"] == "unavailable"
    assert record["reason"] == "unreadable_review_result"
    assert review_requirement_refusal(record) is not None
    # The reader refuses a record that binds through the dump alone.
    applied_digests = _applied(applied_pair(applied, TEST_BYTES))
    if applied_digests["rulespec_content_sha256"] is not None:
        forged = _build()
        forged["reviewed"] = {
            **forged["reviewed"],
            "rulespec_content_sha256": None,
            "rulespec_dumped_content_sha256": applied_digests[
                "rulespec_content_sha256"
            ],
        }
        forged["applied"] = applied_digests
        forged["files"] = _files(
            (RULE_PATH, applied_digests["rulespec_sha256"], True, True),
            (TEST_PATH, applied_digests["test_sha256"], True, True),
        )
        assert _issues(forged) == ["generalist_review scored verdict is malformed"]


def test_a_missing_digest_never_binds():
    # Reviewed: a file apply would not rewrite (no dumped digest). Applied: a
    # file with no content digest. None must not equal None.
    reviewed = b"- a list\n"
    assert dumped_review_content_sha256(reviewed) is None
    unparsable = AppliedPair(
        rulespec_sha256=OTHER_SHA,
        rulespec_content_sha256=None,
        test_sha256=PAIR.test_sha256,
        test_content_sha256=PAIR.test_content_sha256,
    )

    record = build_generalist_review_record(
        _state(evidence=_evidence(rulespec=reviewed)), [unparsable]
    )

    assert record["reviewed"]["rulespec_dumped_content_sha256"] is None
    assert record["status"] == "unavailable"
    assert record["reason"] == "reviewed_content_not_applied"

    # The reader: a passed record whose applied content digest is None.
    passed = _build()
    forged = copy.deepcopy(passed)
    forged["reviewed"]["rulespec_dumped_content_sha256"] = None
    forged["applied"]["rulespec_content_sha256"] = None
    assert (
        "generalist_review verdict is on content that differs from the applied content"
        in _issues(forged)
    )


@pytest.mark.parametrize(
    ("reviewed", "applied"),
    [
        ("0.10", "0.2"),
        ("0.10", "0.11"),
        ("1500.00", "1500.01"),
        ("1_000", "10000"),
        ("+5", "-5"),
        ("yes", "false"),
        # The dump never produces these spellings from these values.
        ("true", "yes"),
        ("8", "010"),
        ("60", "1:00"),
        ("0.1", "0.10"),
        ("1000", "1_000"),
    ],
)
def test_a_changed_value_is_still_not_bound(reviewed, applied):
    for applied_bytes in (_rule(applied), _as_apply_rewrites(_rule(applied))):
        if dumped_review_content_sha256(_rule(reviewed)) == review_content_sha256(
            applied_bytes, rulespec=True
        ):
            # The same value after the dump (for example 0.1 and 0.10).
            assert yaml.safe_load(_rule(reviewed)) == yaml.safe_load(_rule(applied))
            continue
        record = build_generalist_review_record(
            _state(evidence=_evidence(rulespec=_rule(reviewed))),
            [applied_pair(applied_bytes, TEST_BYTES)],
        )
        assert record["status"] == "unavailable"
        assert record["reason"] == "reviewed_content_not_applied"


def test_spelling_that_apply_did_not_produce_is_not_bound():
    # Reviewed `true`; an applied file that says `yes` did not come from the
    # dump of the reviewed file.
    for reviewed, applied in (("true", "yes"), ("8", "010"), ("60", "1:00")):
        record = build_generalist_review_record(
            _state(evidence=_evidence(rulespec=_rule(reviewed))),
            [applied_pair(_rule(applied), TEST_BYTES)],
        )
        assert record["status"] == "unavailable", (reviewed, applied)


def test_apply_dump_keeps_key_order_and_unicode():
    # The settings the apply stamp has always used.
    assert apply_stamp_dump({"b": 1, "a": "é", "c": [0.1]}) == (
        "b: 1\na: é\nc:\n- 0.1\n"
    )


def test_dumped_digest_is_none_when_apply_would_not_rewrite():
    assert dumped_review_content_sha256(b"- a list\n") is None
    assert dumped_review_content_sha256(b"a: [unclosed\n") is None
    assert dumped_review_content_sha256(b"\xff\xfe") is None
    assert dumped_review_content_sha256(RULE_BYTES) == PAIR.rulespec_content_sha256


STAMP_SHA = "5" * 64
# Fixed input text and the exact bytes the apply stamp writes for it. The
# expected bytes are literal (taken from unchanged `main`), so a change to the
# dump call fails here whatever the binding code does.
STAMP_CASES = {
    "non-canonical-number": (
        b"format: rulespec/v1\nrules:\n  - name: rate\n    value: 0.10\n"
        b"    cap: 1_000\n",
        b"format: rulespec/v1\nrules:\n- name: rate\n  value: 0.1\n  cap: 1000\n"
        b"module:\n  source_verification:\n"
        b"    corpus_citation_path: us/statute/26/24\n"
        b'    source_sha256: "' + STAMP_SHA.encode() + b'"\n',
    ),
    "comment-and-key-order": (
        b"# a comment\nrules:\n  - name: b\n    kind: derived\n  - name: a\n"
        b"format: rulespec/v1\nmodule:\n  summary: 'Caf\xc3\xa9: yes'\n",
        b"rules:\n- name: b\n  kind: derived\n- name: a\nformat: rulespec/v1\n"
        b"module:\n  summary: 'Caf\xc3\xa9: yes'\n  source_verification:\n"
        b"    corpus_citation_path: us/statute/26/24\n"
        b'    source_sha256: "' + STAMP_SHA.encode() + b'"\n',
    ),
    "existing-stamp": (
        b"format: rulespec/v1\nmodule:\n  source_verification:\n"
        b"    source_sha256: old\n    corpus_citation_path: us/old\n"
        b"    note: kept\nrules: []\n",
        b"format: rulespec/v1\nmodule:\n  source_verification:\n"
        b'    source_sha256: "' + STAMP_SHA.encode() + b'"\n'
        b"    corpus_citation_path: us/statute/26/24\n    note: kept\nrules: []\n",
    ),
}


@pytest.mark.parametrize("case", sorted(STAMP_CASES))
def test_apply_stamp_writes_these_exact_bytes(tmp_path, case):
    original, expected = STAMP_CASES[case]
    output_file = tmp_path / "rule.yaml"
    output_file.write_bytes(original)
    attestation = {
        "requested_corpus_citation_path": "us/statute/26/24",
        "resolved_corpus_citation_path": "us/statute/26/24",
        "source_sha256": STAMP_SHA,
    }
    with (
        patch(
            "axiom_encode.cli._generated_result_source_attestation",
            return_value=attestation,
        ),
        patch("axiom_encode.cli._source_attestation_structure_issues", return_value=[]),
    ):
        cli._stamp_generated_source_attestation_for_apply(
            SimpleNamespace(backend="openai"), output_file
        )

    assert output_file.read_bytes() == expected
    # The stamp stays byte-for-byte compatible; a scalar-changing stamp is
    # not evidence that the reviewer read the applied content.
    if case == "non-canonical-number":
        assert dumped_review_content_sha256(original) is None
    else:
        assert dumped_review_content_sha256(original) == review_content_sha256(
            expected, rulespec=True
        )
    record = build_generalist_review_record(
        _state(evidence=_evidence(rulespec=original)),
        [applied_pair(expected, TEST_BYTES)],
    )
    assert record["status"] == (
        "unavailable" if case == "non-canonical-number" else "passed"
    )


# --- documents the digest refuses ----------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        b"a: 1\na: 2\n",
        b"a: 2\na: 1\n",
        b"module:\n  source_verification: {note: x}\n  source_verification: {}\n",
        b"base: &b {x: 1}\ncopy: *b\n",
        b"module:\n  source_verification: &s {source_sha256: x}\nother: *s\n",
        MERGE_RULE,
        b"a:\n  <<: {x: 1}\n  y: 2\n",
        b"a: &a [1]\nb: [*a, *a]\n",
    ],
)
def test_repeated_keys_aliases_and_merge_keys_have_no_content_digest(raw):
    assert review_content_sha256(raw, rulespec=True) is None
    assert review_content_sha256(raw, rulespec=False) is None


def test_scalar_anchors_are_plain_values():
    assert review_content_sha256(b"a: &x 1\nb: 2\n", rulespec=False) == (
        review_content_sha256(b"a: 1\nb: 2\n", rulespec=False)
    )


def test_deep_alias_tree_is_refused_quickly():
    import time

    lines = ["l0: &l0 [x, x]"]
    for level in range(1, 40):
        lines.append(f"l{level}: &l{level} [*l{level - 1}, *l{level - 1}]")
    raw = ("\n".join(lines) + "\n").encode()

    started = time.monotonic()
    assert review_content_sha256(raw, rulespec=True) is None
    assert time.monotonic() - started < 2.0


# --- the reader never raises ---------------------------------------------


def _leaf_paths(value, prefix=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaf_paths(item, (*prefix, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _leaf_paths(item, (*prefix, index))
        yield prefix
    else:
        yield prefix
    if prefix and isinstance(value, dict):
        yield prefix


@pytest.mark.parametrize("junk", [{}, [], {"a": 1}, [1], 1.5, "x", None, True, 0])
def test_reader_returns_issues_and_never_raises_on_any_wrong_type(junk):
    records = [
        _build(),
        _build(applied=[]),
        _build(skipped_by_flag=True),
        _build(applied=[PAIR, _other_pair()]),
    ]
    for record in records:
        for path in set(_leaf_paths(record)):
            changed = _tampered(record, path, junk)
            findings = generalist_review_record_issues(changed, APPLIED_SHA256S)
            assert all(isinstance(finding, str) for finding in findings)
            assert all(
                finding.check.startswith("manifest:generalist-review/")
                for finding in findings
            )
    # The cases the second review found.
    passed = _build()
    assert _issues(_tampered(passed, ("reviewer", "backend"), {}))
    assert _issues(_tampered(passed, ("applied", "rulespec_sha256"), []))


def test_manifest_reader_returns_a_typed_issue_for_a_wrong_typed_record(tmp_path):
    payload, _ = _write_manifest(tmp_path, per_file=True, review_state=_bound_state())
    baseline = _schema_issues(payload)
    for path, junk in (
        (("reviewer", "backend"), {}),
        (("applied", "rulespec_sha256"), []),
        (("files",), {}),
        (("verdict",), []),
    ):
        changed = copy.deepcopy(payload)
        changed[GENERALIST_REVIEW_FIELD] = _tampered(
            payload[GENERALIST_REVIEW_FIELD], path, junk
        )
        added = [issue for issue in _schema_issues(changed) if issue not in baseline]
        assert added and all(
            issue.check.startswith("manifest:generalist-review/") for issue in added
        )


def test_writer_refusal_has_its_own_exception_type(tmp_path):
    with pytest.raises(ReviewRequirementRefused):
        _write_manifest(
            tmp_path,
            per_file=True,
            review_state=_bound_state(required=True, metrics_pass=False, score=2.0),
        )
    assert issubclass(ReviewRequirementRefused, RuntimeError)


# --- review 5984585097: an explicit pass, and only what the reviewer saw ---


def _review(tmp_path, monkeypatch, output, *, reviewer="generalist-reviewer", **kw):
    rule = _candidate(tmp_path, **kw)
    monkeypatch.setattr(
        validator_pipeline, "run_claude_code", lambda *_a, **_k: (output, 0)
    )
    return rule, _pipeline(tmp_path)._run_reviewer(reviewer, rule)


# The three inputs of the review, verbatim.
_REVIEW_5984585097_PROBES = [
    '{"score":9}',
    '{"score":9,"Passed":false}',
    '{"final":{"Passed":false},"draft":{"passed":true,"score":9}}',
]


@pytest.mark.parametrize("output", _REVIEW_5984585097_PROBES)
def test_score_without_an_explicit_pass_cannot_satisfy_required_review(
    tmp_path, monkeypatch, output
):
    rule, result = _review(tmp_path, monkeypatch, output)
    record = _record_for(result, rule, required=True)

    assert record["status"] == "unavailable"
    assert record["verdict"]["passed"] is None
    assert review_requirement_refusal(record) is not None


@pytest.mark.parametrize(
    "output",
    [
        '{"score":9}',
        '{"score":9,"blocking_issues":[]}',
        '{"score": 7.0, "non_blocking_issues": []}',
    ],
)
def test_legacy_score_fallback_stays_in_eval_scoring_and_out_of_the_record(
    tmp_path, monkeypatch, output
):
    rule, result = _review(tmp_path, monkeypatch, output)

    # Eval scoring keeps its fallback: the reviewer result still passes.
    assert result.passed is True
    assert result.details["review_evidence"]["explicit_passed"] is None
    # The record does not take it for a verdict, required or not.
    for required in (False, True):
        record = _record_for(result, rule, required=required)
        assert record["status"] == "unavailable"
        assert record["reason"] == "unreadable_review_result"
    assert review_requirement_refusal(_record_for(result, rule, required=True))


@pytest.mark.parametrize("explicit", [None, False, "true", 1, 1.0, [True], "absent"])
def test_record_needs_the_explicit_boolean_pass_in_the_evidence(explicit):
    evidence = _evidence()
    if explicit == "absent":
        del evidence["explicit_passed"]
    else:
        evidence["explicit_passed"] = explicit

    record = _build(evidence=evidence, required=True)

    assert record["status"] == "unavailable"
    assert record["reason"] == "unreadable_review_result"
    assert review_requirement_refusal(record) is not None
    # A failing verdict needs no explicit pass to stay failed.
    assert _build(evidence=evidence, metrics_pass=False)["status"] == "failed"


_CYRILLIC_A = "\u0430"
_ZERO_WIDTH_SPACE = "\u200b"
_CURLY_OPEN, _CURLY_CLOSE = "\u201c", "\u201d"
_FULLWIDTH_PASSED = "\uff50\uff41\uff53\uff53\uff45\uff44"
_BOM = "\ufeff"
_ADVERSARIAL_VERDICTS = [
    # no explicit verdict
    '{"score":9}',
    '{"score":9,"blocking_issues":[]}',
    '{"score":10,"issues":[],"reasoning":"passed"}',
    "\ufeff" + '{"score": 9}',
    # key case and spacing variants
    '{"score":9,"Passed":false}',
    '{"score":9,"PASSED":true}',
    '{"score":9,"passed ":true}',
    '{"score":9," passed":true}',
    '{"score":9,"p_a_s_s_e_d":true}',
    '{"passed":true,"score":9,"Passed":false}',
    '{"passed":true,"score":9,"pAsSeD":true}',
    # look-alike keys
    '{"score":9,"p' + _CYRILLIC_A + 'ssed":true}',
    '{"passed":true,"score":9,"p' + _CYRILLIC_A + 'ssed":false}',
    '{"passed":true,"score":9,"\uff50\uff41\uff53\uff53\uff45\uff44":false}',
    '{"passed":true,"score":9,"pa\u200bssed":false}',
    '{"passed":true,"score":9,"pa\u0336ssed":false}',
    '{"passed":true,"score":9,"passed\\u0000":false}',
    '{"passed":true,"score":9,"\\u0050assed":false}',
    '{"score":9,"\\u0070assed":false}',
    # duplicate keys
    '{"passed":false,"passed":true,"score":9}',
    '{"passed":true,"passed":true,"score":9}',
    # nested wrappers
    '{"final":{"Passed":false},"draft":{"passed":true,"score":9}}',
    '{"draft":{"passed":true,"score":9},"final":{"Passed":false}}',
    '{"result":{"passed":true,"score":9},"final":{"passed":false}}',
    '{"a":{"b":{"c":{"PASSED":false}}},"draft":{"passed":true,"score":9}}',
    '{"score":9,"passed":true,"details":{"Passed":false}}',
    '{"score":9,"passed":true,"details":[{"x":[{"passed ":false}]}]}',
    # arrays
    '[{"passed":false,"score":2},{"passed":true,"score":9}]',
    '[{"Passed":false},{"passed":true,"score":9}]',
    # non-boolean values
    '{"passed":"true","score":9}',
    '{"passed":1,"score":9}',
    '{"passed":null,"score":9}',
    '{"passed":[true],"score":9}',
    '{"passed":{"value":true},"score":9}',
    '{"passed":True,"score":9}',
    # strings that contain JSON
    '{"score":9,"reasoning":"{\\"passed\\": true}"}',
    '{"score":9,"note":"\\"passed\\": true, final"}',
    # a second object after the verdict
    '{"passed":true,"score":9}\n{"passed":false,"score":2}',
    '{"passed":true,"score":9}\n{"Passed":false}',
    '{"passed":true,"score":9}\n{"x":{"PASSED":false}}',
    # BOM and comments
    '{"passed":true,"score":9}\n{"passed"\ufeff: false}',
    '{"passed":true,"score":9}\n{ /* c */ passed: false }',
    '{"score":9, /* "passed": true */ "reasoning":"x"}',
    '{"score":9, // "passed": true\n "reasoning":"x"}',
    # other spellings of the document
    "{'passed': true, 'score': 9}",
    "passed: true\nscore: 9\n",
    # an explicit pass that the rest of the verdict contradicts
    '{"score":9,"passed":true,"blocking_issues":["bad"]}',
    '{"score":6.9,"passed":true}',
    '{"passed":true}',
    '{"passed":true,"score":1e309}',
]


def test_adversarial_verdict_list_is_large_enough():
    assert len(set(_ADVERSARIAL_VERDICTS)) >= 25


@pytest.mark.parametrize("output", _ADVERSARIAL_VERDICTS)
def test_adversarial_verdicts_never_yield_a_passed_record(
    tmp_path, monkeypatch, output
):
    rule, result = _review(tmp_path, monkeypatch, output)

    for required in (False, True):
        record = _record_for(result, rule, required=required)
        assert record["status"] != "passed"
        assert record["status"] != "partial"
    assert review_requirement_refusal(_record_for(result, rule, required=True))
    evidence = (result.details or {}).get("review_evidence", {})
    if evidence.get("explicit_passed") is True:
        # Only an explicit pass that something else overrides gets this far.
        assert result.passed is False


_VALID_PASS = {
    "score": 9,
    "passed": True,
    "blocking_issues": [],
    "non_blocking_issues": ["rename the helper"],
    "reasoning": "All branches of the cited text are present; tests passed {ok}.",
}


@pytest.mark.parametrize(
    "output",
    [
        '{"passed": true, "score": 9}',
        json.dumps(_VALID_PASS),
        json.dumps(_VALID_PASS, indent=2),
        "```json\n" + json.dumps(_VALID_PASS, indent=2) + "\n```",
        '```json\n{"score": 8.5, "passed": true, "blocking_issues": [],}\n```',
        "\ufeff" + json.dumps(_VALID_PASS),
    ],
)
def test_a_plain_explicit_pass_still_satisfies_required_review(
    tmp_path, monkeypatch, output
):
    rule, result = _review(tmp_path, monkeypatch, output)
    record = _record_for(result, rule, required=True)

    assert result.passed is True
    assert result.details["review_evidence"]["explicit_passed"] is True
    assert record["status"] == "passed"
    assert record["verdict"]["passed"] is True
    assert review_requirement_refusal(record) is None


@pytest.mark.parametrize(
    "key",
    [
        "Passed",
        "PASSED",
        " passed",
        "passed ",
        "p a s s e d",
        "passed?",
        "p" + _CYRILLIC_A + "ssed",
        "\u0440\u0430\u0455\u0455\u0435\u0501",
        "\uff50\uff41\uff53\uff53\uff45\uff44",
        "pa\u200bssed",
        "passed\x00",
        "Score",
        "SCORE ",
        "Blocking_Issues",
        "blockingIssues",
        "blocking issues",
        "BLOCKING-ISSUES",
    ],
)
def test_other_spellings_of_a_verdict_key_are_competing(key):
    assert validator_pipeline._is_competing_verdict_key(key) is True
    with pytest.raises(ValueError, match="Competing spelling"):
        validator_pipeline._extract_json_object(
            json.dumps({"passed": True, "score": 9, "meta": {key: False}})
        )


@pytest.mark.parametrize(
    "key",
    [
        "passed",
        "score",
        "blocking_issues",
        "non_blocking_issues",
        "issues",
        "reasoning",
        "pass",
        "passes",
        "failed",
        "verdict",
        "bypassed",
        "passed_checks",
        "scores",
    ],
)
def test_other_keys_are_not_spellings_of_a_verdict_key(key):
    assert validator_pipeline._is_competing_verdict_key(key) is False


def test_non_generalist_reviewers_keep_the_legacy_fallback(tmp_path, monkeypatch):
    _rule, result = _review(
        tmp_path, monkeypatch, '{"score": 9}', reviewer="formula-reviewer"
    )

    assert result.passed is True


_DEFERRED_RULE_TEXT = """format: rulespec/v1
module:
  status: deferred
  deferred_outputs:
    - output: us:statutes/x#y
      reason: The entity is not supported.
rules: []
"""
_EMPTY_COMPANION_SHA = hashlib.sha256(b"[]\n").hexdigest()


def _planned_record(rule, result, *, required=True):
    """The record of the real apply plan for one reviewed candidate."""

    relative = Path("statutes/rule.yaml")
    planned, wrote_empty = cli._planned_apply_file_bytes(
        rule, relative_output=relative, supplemental_files={}
    )
    test = planned.get(Path("statutes/rule.test.yaml"))
    record = build_generalist_review_record(
        cli._encode_generalist_review_state(
            SimpleNamespace(metrics=_metrics_from(result)),
            skipped_by_flag=False,
            required=required,
        ),
        [applied_pair(planned[relative], test)],
        [hashlib.sha256(raw).hexdigest() for raw in planned.values()],
    )
    return planned, wrote_empty, record


def test_reviewer_sees_the_empty_companion_that_apply_writes(tmp_path, monkeypatch):
    prompts = []
    rule = _candidate(tmp_path, rule_text=_DEFERRED_RULE_TEXT, test_text=None)
    monkeypatch.setattr(
        validator_pipeline,
        "run_claude_code",
        lambda prompt, **_k: (prompts.append(prompt), ('{"score":9,"passed":true}', 0))[
            1
        ],
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)
    planned, wrote_empty, record = _planned_record(rule, result)

    # The planner writes the companion; nothing wrote it before the review.
    assert wrote_empty is True
    assert not (tmp_path / "rule.test.yaml").exists()
    assert planned[Path("statutes/rule.test.yaml")] == b"[]\n"
    # The reviewer was shown exactly those bytes and the evidence names them.
    assert "\n## Companion Test File\n[]\n" in prompts[0]
    evidence = result.details["review_evidence"]
    assert evidence["reviewed_test_sha256"] == _EMPTY_COMPANION_SHA
    assert evidence["test_complete"] is True
    assert record["reviewed"]["test_sha256"] == _EMPTY_COMPANION_SHA
    assert record["applied"]["test_sha256"] == _EMPTY_COMPANION_SHA
    assert record["status"] == "passed"
    assert record["files"] == _files(
        (_sha(rule), True, True), (_EMPTY_COMPANION_SHA, True, True)
    )
    assert review_requirement_refusal(record) is None
    assert _issues(record, [entry["sha256"] for entry in record["files"]]) == []


def test_companion_that_the_reviewer_never_saw_is_not_credited(tmp_path, monkeypatch):
    # The probe of the review: the evidence of a review without a companion,
    # against a plan that writes the empty one.
    rule, result = _review(
        tmp_path,
        monkeypatch,
        '{"score":9,"passed":true}',
        rule_text=_DEFERRED_RULE_TEXT,
        test_text=None,
    )
    evidence = result.details["review_evidence"]
    evidence.update(review_evidence_digests(rule.read_bytes(), None))
    assert evidence["reviewed_test_sha256"] is None

    planned, wrote_empty, record = _planned_record(rule, result)

    assert wrote_empty is True
    assert record["status"] == "unavailable"
    assert record["reason"] == "reviewed_content_not_applied"
    assert record["files"] == _files(
        (_sha(rule), False, False), (_EMPTY_COMPANION_SHA, False, False)
    )
    assert review_requirement_refusal(record) is not None
    assert _issues(record, [entry["sha256"] for entry in record["files"]]) == [
        "generalist_review was required but the recorded status is unavailable"
    ]


def test_reader_refuses_a_pass_that_credits_an_unseen_companion():
    applied = applied_pair(RULE_BYTES, b"[]\n")
    assert _build(applied=[applied], evidence=_evidence(test=None))["status"] == (
        "unavailable"
    )
    # A record of a review that read no companion file, dressed up as a pass
    # on an applied companion.
    manifest = [applied.rulespec_sha256, applied.test_sha256]
    passed = _build(applied=[applied], evidence=_evidence(test=b"[]\n"))
    assert passed["status"] == "passed"
    assert _issues(passed, manifest) == []

    forged = copy.deepcopy(passed)
    forged["reviewed"]["test_sha256"] = None
    assert _issues(forged, manifest) == [
        "generalist_review scored verdict is malformed"
    ]
    # With the content digest of an absent companion on both sides too.
    absent = review_content_sha256(None, rulespec=False)
    forged["reviewed"]["test_content_sha256"] = absent
    forged["applied"]["test_content_sha256"] = absent
    assert _issues(forged, manifest) == [
        "generalist_review verdict is not bound to the applied file digests"
    ]
    # A reviewed companion file never has the content of an absent one.
    forged = copy.deepcopy(passed)
    forged["reviewed"]["test_content_sha256"] = absent
    assert _issues(forged, manifest)[0] == (
        "generalist_review scored verdict is malformed"
    )


@pytest.mark.parametrize(
    "rule_text",
    [
        _DEFERRED_RULE_TEXT,
        _DEFERRED_RULE_TEXT.replace("status: deferred", "status: active"),
        _DEFERRED_RULE_TEXT.replace("rules: []", "rules: [{name: x}]"),
        _DEFERRED_RULE_TEXT.replace("rules: []\n", ""),
        "format: rulespec/v1\nmodule:\n  status: deferred\nrules: []\n",
        "format: rulespec/v1\nmodule:\n  status: deferred\n"
        "  deferred_outputs: []\nrules: []\n",
        "format: rulespec/v1\nrules: []\n",
        "- a\n- b\n",
        "{unclosed",
        RULE_BYTES.decode(),
    ],
)
def test_reviewer_and_planner_agree_on_the_empty_companion(tmp_path, rule_text):
    rule = _candidate(tmp_path, rule_text=rule_text, test_text=None)

    planner = cli._needs_empty_deferred_companion_test(rule)

    assert (
        record_module.empty_deferred_companion_test(rule.read_bytes()) is not None
    ) is planner
    if planner:
        assert (
            record_module.empty_deferred_companion_test(rule.read_bytes())
            == cli._planned_apply_file_bytes(
                rule, relative_output=Path("statutes/rule.yaml"), supplemental_files={}
            )[0][Path("statutes/rule.test.yaml")]
        )


def test_only_the_generalist_reviewer_is_shown_a_generated_companion(
    tmp_path, monkeypatch
):
    prompts = []
    rule = _candidate(tmp_path, rule_text=_DEFERRED_RULE_TEXT, test_text=None)
    monkeypatch.setattr(
        validator_pipeline,
        "run_claude_code",
        lambda prompt, **_k: (prompts.append(prompt), ('{"score":9,"passed":true}', 0))[
            1
        ],
    )

    result = _pipeline(tmp_path)._run_reviewer("formula-reviewer", rule)

    assert "Companion Test File" not in prompts[0]
    assert result.details["review_evidence"]["reviewed_test_sha256"] is None


# Keys that a YAML 1.1 reader (yaml.safe_load) or a YAML 1.2 core reader
# gives one value.
_EQUAL_TYPED_KEYS = [
    ("0x10", "16"),
    ("1", "1.0"),
    ("010", "8"),
    ("0o10", "8"),
    ("1_000", "1000"),
    ("1:30", "90"),
    ("yes", "true"),
    ("true", "1"),
    (".nan", ".NaN"),
    ("~", "null"),
    ('! "16"', "16"),
    ("!!int '16'", "16"),
    ("+1", "1"),
    ("1e0", "1"),
]


@pytest.mark.parametrize(("left", "right"), _EQUAL_TYPED_KEYS)
def test_semantically_equal_typed_keys_have_no_content_digest(left, right):
    for first, second in ((left, right), (right, left)):
        raw = f"values: {{{first}: old, {second}: new}}\n".encode()
        block = f"values:\n  {first}: old\n  {second}: new\n".encode()
        for rulespec in (False, True):
            assert review_content_sha256(raw, rulespec=rulespec) is None
            assert review_content_sha256(block, rulespec=rulespec) is None


def test_typed_key_order_cannot_change_the_loaded_value_under_one_digest():
    # The probe of the review.
    forward = b"values: {0x10: old, 16: new}\n"
    reverse = b"values: {16: new, 0x10: old}\n"
    assert yaml.safe_load(forward) != yaml.safe_load(reverse)

    assert review_content_sha256(forward, rulespec=True) is None
    assert review_content_sha256(reverse, rulespec=True) is None
    for raw in (forward, reverse):
        record = build_generalist_review_record(
            _state(evidence=_evidence(rulespec=raw)), [applied_pair(raw, TEST_BYTES)]
        )
        assert record["status"] == "unavailable"


@pytest.mark.parametrize(
    "keys",
    [
        ("1", "'1'"),
        ("1", "2"),
        ("a", "b"),
        ("a", "A"),
        ("yes", "'true'"),
        ("2024-01-01", "2025-01-01"),
        ("0x10", "17"),
        ("1.5", "2.5"),
        ("null", "'null'"),
        ("true", "false"),
        ("1", "2", "3", "4", "5", "6", "7", "8"),
    ],
)
def test_distinct_typed_keys_keep_a_content_digest(keys):
    raw = ("values:\n" + "".join(f"  {key}: x\n" for key in keys)).encode()
    reverse = ("values:\n" + "".join(f"  {key}: x\n" for key in keys[::-1])).encode()

    assert review_content_sha256(raw, rulespec=True) is not None
    assert review_content_sha256(raw, rulespec=True) == review_content_sha256(
        reverse, rulespec=True
    )
    assert yaml.safe_load(raw) == yaml.safe_load(reverse)


@pytest.mark.parametrize(
    "key",
    ["!!bool maybe", "!!timestamp nope", "!!int abc", "!!float x"],
)
def test_a_key_that_safe_load_cannot_construct_has_no_content_digest(key):
    raw = f"values:\n  {key}: x\n".encode()
    with pytest.raises(Exception):  # noqa: B017, PT011 - any PyYAML failure
        yaml.safe_load(raw)

    assert review_content_sha256(raw, rulespec=True) is None
    assert review_content_sha256(raw, rulespec=False) is None


# --- second round: independent review of cd0ed61d4 ------------------------


@pytest.mark.parametrize(
    ("tagged", "plain"),
    [
        # The three probes of the review.
        ("!!value a", "a"),
        ("!!value =", "'='"),
        ("!!value 16", "'16'"),
        ("=", "'='"),
        ("!!str a", "a"),
        ("!!str 16", "'16'"),
        ("!!int 16", "16"),
        ("!!int 0x10", "16"),
        ("!!float 16", "16"),
        ("!!null ''", "~"),
        ("!!bool true", "1"),
        ("!!binary YQ==", "!!binary 'YQ=='"),
        ("!!timestamp 2024-01-01", "2024-01-01"),
        ("!foo a", "a"),
        ("&k a", "*k "),
        ("0o20", "1_6"),
        ("0o20", "0b10000"),
        ("1e3", "1_000"),
        ("0b10000", "16"),
        ("0o20", "0x10"),
    ],
)
def test_keys_that_any_reader_equates_have_no_content_digest(tagged, plain):
    for first, second in ((tagged, plain), (plain, tagged)):
        flow = f"values: {{{first}: old, {second}: new}}\n".encode()
        block = f"values:\n  {first}: old\n  {second}: new\n".encode()
        for raw in (flow, block):
            assert review_content_sha256(raw, rulespec=True) is None
            assert review_content_sha256(raw, rulespec=False) is None


@pytest.mark.parametrize(
    "key",
    [
        "!!value a",
        "!!timestamp 2024-01-01",
        "!!binary YQ==",
        "!foo a",
        "!!python/name:x a",
        "!!int 1_000",
        "!!bool yes",
        "? [1, 2]\n ",
        "? {a: 1}\n ",
    ],
)
def test_a_key_that_one_dialect_cannot_construct_has_no_content_digest(key):
    raw = f"values:\n  {key}: x\n".encode()

    assert review_content_sha256(raw, rulespec=True) is None
    assert review_content_sha256(raw, rulespec=False) is None


def test_value_tag_key_order_cannot_pass_a_review_of_other_content():
    # The probe of the review: A reviewed, B applied.
    reviewed = b"values: {!!value a: old, a: new}\n"
    applied = b"values: {a: new, !!value a: old}\n"
    assert yaml.safe_load(reviewed) != yaml.safe_load(applied)

    record = build_generalist_review_record(
        _state(evidence=_evidence(rulespec=reviewed)),
        [applied_pair(applied, TEST_BYTES)],
    )

    assert record["status"] == "unavailable"
    assert record["reason"] == "unreadable_review_result"


def test_every_scalar_key_has_an_identity():
    for text in ("a: 1\n", "1: 1\n", "'1': 1\n", "yes: 1\n", "=: 1\n", "~: 1\n"):
        root = yaml.compose(text, Loader=record_module._ReviewContentLoader)
        assert record_module._scalar_key_identities(root.value[0][0])


@pytest.mark.parametrize("text", ["", "[]\n", "[]  # comment\n", "~\n", "# only\n"])
def test_an_empty_companion_is_shown_to_the_reviewer_it_is_credited_to(
    tmp_path, monkeypatch, text
):
    prompts = []
    rule = _candidate(tmp_path, test_text=text)
    monkeypatch.setattr(
        validator_pipeline,
        "run_claude_code",
        lambda prompt, **_k: (prompts.append(prompt), ('{"score":9,"passed":true}', 0))[
            1
        ],
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)
    record = _record_for(result, rule, required=True)

    # The record credits the companion, so the prompt has its section, also
    # for a 0-byte file.
    assert f"\n## Companion Test File\n{text}\n" in prompts[0]
    assert result.details["review_evidence"]["reviewed_test_sha256"] == _sha(
        tmp_path / "rule.test.yaml"
    )
    assert record["status"] == "passed"


@pytest.mark.parametrize(
    ("later", "status"),
    [
        # Binding is by canonical content: an empty list with other
        # formatting or a comment is the same content.
        (b"[]\n", "passed"),
        (b"[]  # comment\n", "passed"),
        (b"[ ]\n", "passed"),
        # A null document, a 0-byte file and an empty mapping are other
        # content: the companion loader refuses them.
        (b"~\n", "unavailable"),
        (b"", "unavailable"),
        (b"# comment\n", "unavailable"),
        (b"{}\n", "unavailable"),
        # A companion with cases is other content.
        (TEST_BYTES, "unavailable"),
        (b"- name: one\n", "unavailable"),
        (b"[[]]\n", "unavailable"),
        (b"[null]\n", "unavailable"),
        (b"{a: 1}\n", "unavailable"),
    ],
)
def test_a_reviewed_empty_companion_binds_equivalent_content_only(
    tmp_path, monkeypatch, later, status
):
    rule, result = _review(tmp_path, monkeypatch, '{"score":9,"passed":true}')
    assert (tmp_path / "rule.test.yaml").read_bytes() == b"[]\n"

    (tmp_path / "rule.test.yaml").write_bytes(later)
    record = _record_for(result, rule, required=True)

    assert record["status"] == status
    assert (review_requirement_refusal(record) is None) is (status == "passed")
    if status == "passed":
        # The record names the bytes on both sides.
        assert record["reviewed"]["test_sha256"] == _EMPTY_COMPANION_SHA
        assert record["applied"]["test_sha256"] == hashlib.sha256(later).hexdigest()


_PASS = '{"passed":true,"score":9}'
_LENIENT_COMPETING_OUTPUTS = [
    # The four inputs of the review.
    '{"p' + _CYRILLIC_A + 'ssed":false,}\n' + _PASS,
    '{"passed ":false,}\n' + _PASS,
    '{"p_assed":false,}\n' + _PASS,
    _PASS + "\n{'p" + _CYRILLIC_A + "ssed': false}",
    # Objects that only a lenient reader accepts, before and after.
    '{"Passed":false,}\n' + _PASS,
    _PASS + '\n{"p' + _CYRILLIC_A + 'ssed":false,}',
    _PASS + '\n{"x":1,"PASSED ":false,}',
    _PASS + '\n{"a":[1,2,],"pa' + _ZERO_WIDTH_SPACE + 'ssed":false}',
    _PASS + "\n{'Passed ': false}",
    _PASS + "\n{" + _CURLY_OPEN + "p_assed" + _CURLY_CLOSE + ": false}",
    _PASS + "\n{ p" + _CYRILLIC_A + "ssed: false }",
    _PASS + "\n{ Passed : false }",
    '{"' + _FULLWIDTH_PASSED + '":false,}\n' + _PASS,
    '{"passed\\u0020":false,}\n' + _PASS,
    '{"pa\\ssed":false,}\n' + _PASS,
    '{"p-a-s-s-e-d": false\n' + _PASS,
    '"passed ": false\n' + _PASS,
    _PASS + '\n"p' + _CYRILLIC_A + 'ssed" /* c */ : false',
    _PASS + '\n{"Score":2,}',
    _PASS + '\n{"blockingIssues":["bad"],}',
    _PASS + "\n{'Blocking_Issues': ['bad']}",
    '{"SCORE": 1,}\n```json\n' + _PASS + "\n```",
    # Inside the verdict, decoded with the trailing-comma repair.
    '{"passed":true,"score":9,"Passed":false,}',
    '{"passed":true,"score":9,"meta":{"p' + _CYRILLIC_A + 'ssed":false,},}',
    '{"passed":true,"score":9,"Blocking_Issues":["bad"]}',
    '{"passed":true,"score":9,"blocking issues":["bad"]}',
    '{"passed":true,"score":9,"Score":2}',
    '{"passed":true,"score":9,"SCORE ":2,}',
    # A mistyped score beside an explicit pass.
    '{"passed":true,"score":"9"}',
    '{"passed":true,"score":true}',
    '{"passed":true,"score":[9]}',
    '{"passed":true,"score":null}',
    '{"passed":true,"score":{"value":9}}',
]


def test_lenient_competing_list_is_large_enough():
    assert len(set(_LENIENT_COMPETING_OUTPUTS)) >= 30


@pytest.mark.parametrize("output", _LENIENT_COMPETING_OUTPUTS)
def test_competing_verdict_keys_outside_strict_json_never_yield_a_passed_record(
    tmp_path, monkeypatch, output
):
    rule, result = _review(tmp_path, monkeypatch, output)

    assert result.passed is False
    for required in (False, True):
        record = _record_for(result, rule, required=required)
        assert record["status"] == "unavailable"
    assert review_requirement_refusal(_record_for(result, rule, required=True))


@pytest.mark.parametrize(
    "output",
    [
        # Shapes that stay valid: one object of the prompt schema, alone or
        # in one code fence.
        '{"passed": true, "score": 9}',
        '{"score": 9, "passed": true}',
        '{"passed": true, "score": 9.5, "blocking_issues": []}',
        '{"passed": true, "score": 7, "non_blocking_issues": [], "reasoning": "ok"}',
        json.dumps(_VALID_PASS),
        json.dumps(_VALID_PASS, indent=2),
        "```json\n" + json.dumps(_VALID_PASS, indent=2) + "\n```",
        '```json\n{"score": 8.5, "passed": true, "blocking_issues": [],}\n```',
        _BOM + json.dumps(_VALID_PASS),
        '{"passed": true, "score": 9, "reasoning": "It\'s fine: the \'cap\' rule, '
        'the \\"Passed\\": label and {braces}."}',
    ],
)
def test_valid_reviewer_shapes_still_pass(tmp_path, monkeypatch, output):
    rule, result = _review(tmp_path, monkeypatch, output)
    record = _record_for(result, rule, required=True)

    assert result.passed is True
    assert record["status"] == "passed"
    assert review_requirement_refusal(record) is None


def test_nested_other_case_of_passed_is_the_deliberate_refusal(tmp_path, monkeypatch):
    rule, result = _review(
        tmp_path,
        monkeypatch,
        '{"passed": true, "score": 9, "details": {"Passed": 4}}',
    )

    assert "reviewer_parse_failed" in result.issues
    assert _record_for(result, rule, required=True)["status"] == "unavailable"


@pytest.mark.parametrize("score", ['"9"', "true", "false", "null", "[9]", "{}"])
def test_a_mistyped_score_is_a_parse_failure_for_every_reviewer(
    tmp_path, monkeypatch, score
):
    for reviewer in ("generalist-reviewer", "formula-reviewer"):
        _rule, result = _review(
            tmp_path,
            monkeypatch,
            '{"passed": true, "score": ' + score + "}",
            reviewer=reviewer,
        )
        assert result.passed is False
        assert "reviewer_parse_failed" in result.issues


@pytest.mark.parametrize(
    "raw",
    [
        b"values:\n  ? [1, 2]\n  : x\n",
        b"values:\n  ? {a: 1}\n  : x\n",
        b"values: {[1, 2]: x}\n",
        b"values: {{a: 1}: x}\n",
        b"values:\n  ? - 1\n    - 2\n  : x\n  plain: y\n",
        b"values:\n  ? a: 1\n  : x\n  plain: y\n",
    ],
    ids=[
        "sequence-key-flow-in-block",
        "mapping-key-flow-in-block",
        "sequence-key-flow",
        "mapping-key-flow",
        "sequence-key-block",
        "mapping-key-block",
    ],
)
def test_a_key_that_is_not_a_scalar_has_no_content_digest(raw):
    # The node tree itself is readable: only the key rule refuses it.
    root = yaml.compose(raw, Loader=record_module._ReviewContentLoader)
    key_node = root.value[0][1].value[0][0]
    assert not isinstance(key_node, yaml.ScalarNode)
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(raw)

    assert review_content_sha256(raw, rulespec=True) is None
    assert review_content_sha256(raw, rulespec=False) is None
    record = build_generalist_review_record(
        _state(evidence=_evidence(rulespec=raw)), [applied_pair(raw, TEST_BYTES)]
    )
    assert record["status"] == "unavailable"
    assert record["reason"] == "unreadable_review_result"


# --- review of c3bf0d657: coverage is bound to each manifest path ---------

_PRIMARY = Path("statutes/26/32/c/2.yaml")
_PRIMARY_TEST = "statutes/26/32/c/2.test.yaml"
_DEPENDENT_TEST = "statutes/26/24/d.test.yaml"
_EMPTY_RULE = b"format: rulespec/v1\nrules: []\n"


def _repaired_dependent_companion(tmp_path) -> str:
    """The text the real obsolete-input repair writes for a dependent test."""

    repo = tmp_path / "rulespec-us" / "us"
    dependent = repo / "statutes/26/24/d.yaml"
    dependent_test = repo / _DEPENDENT_TEST
    target = repo / _PRIMARY
    dependent.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    dependent.write_bytes(_EMPTY_RULE)
    target.write_bytes(_EMPTY_RULE)
    dependent_test.write_text(
        "- name: stale_cross_module_input\n"
        "  period: 2026\n"
        "  input:\n"
        "    us:statutes/26/24/d#input.qualifying_children_count: 2\n"
        "    us:statutes/26/1401#input.self_employment_income: 0\n"
        "  output:\n"
        "    us:statutes/26/24/d#refundable_ctc: 2550\n"
    )
    validation = SimpleNamespace(
        results={
            "ci": SimpleNamespace(
                error=(
                    "Test case `stale_cross_module_input` execution failed: "
                    "dataset input "
                    "`tmp-rulespec:statutes/26/1401#input.self_employment_income` "
                    "must use an absolute legal RuleSpec reference that resolves "
                    "to an input slot, derived rule, or parameter in the compiled "
                    "program"
                )
            )
        }
    )
    changed = cli._remove_invalid_dependent_test_inputs(
        overlay_repo=repo,
        relative_output=_PRIMARY,
        validations=[(dependent, validation)],
    )
    assert changed == [dependent_test]
    return dependent_test.read_text()


def _generated_candidate(tmp_path, rule: bytes, test: bytes | None):
    """One generated candidate where `encode --apply` finds it."""

    output_root = tmp_path / "out"
    output_file = output_root / "codex-test" / _PRIMARY
    output_file.parent.mkdir(parents=True)
    output_file.write_bytes(rule)
    if test is not None:
        output_file.with_name("2.test.yaml").write_bytes(test)
    return output_root, SimpleNamespace(
        output_file=str(output_file), runner="codex-test"
    )


def _planned_plan_record(tmp_path, rule, test, supplemental, *, required=True):
    """The record `encode --apply` builds before it writes: the real planner,
    the real record builder, and a review of exactly ``rule`` and ``test``."""

    output_root, result = _generated_candidate(tmp_path, rule, test)
    pairs, files, reviewed_path = cli._planned_review_files(
        result, output_root=output_root, supplemental_files=supplemental
    )
    record = record_module.build_generalist_review_record(
        _state(evidence=_evidence(rulespec=rule, test=test), required=required),
        pairs,
        files,
        reviewed_rulespec_path=reviewed_path,
    )
    return record, files


def test_repaired_dependent_companion_with_the_reviewed_bytes_stays_unreviewed(
    tmp_path,
):
    # The probe of the review: the obsolete-input repair leaves a dependent's
    # companion test byte-identical to the reviewed primary companion.
    repaired = _repaired_dependent_companion(tmp_path)
    companion = repaired.encode()

    record, files = _planned_plan_record(
        tmp_path, _EMPTY_RULE, companion, {Path(_DEPENDENT_TEST): repaired}
    )

    assert len(files) == 3
    shared = hashlib.sha256(companion).hexdigest()
    assert record["files"] == _files(
        (_DEPENDENT_TEST, shared, False, False),
        (_PRIMARY_TEST, shared, True, True),
        (_PRIMARY.as_posix(), hashlib.sha256(_EMPTY_RULE).hexdigest(), True, True),
    )
    assert (record["status"], record["reason"]) == (
        "partial",
        "unreviewed_applied_files",
    )
    assert record["applied"]["rulespec_path"] == _PRIMARY.as_posix()
    assert record["applied"]["test_path"] == _PRIMARY_TEST
    assert review_requirement_refusal(record) is not None
    assert record_module.generalist_review_record_issues(record, files) == [
        "generalist_review was required but the recorded status is partial"
    ]
    # Without the dependent the same review covers the whole plan.
    alone, alone_files = _planned_plan_record(
        tmp_path / "alone", _EMPTY_RULE, companion, {}
    )
    assert alone["status"] == "passed"
    assert record_module.generalist_review_record_issues(alone, alone_files) == []


def test_reader_refuses_a_pass_that_credits_equal_bytes_at_another_path(tmp_path):
    repaired = _repaired_dependent_companion(tmp_path)
    honest, files = _planned_plan_record(
        tmp_path, _EMPTY_RULE, repaired.encode(), {Path(_DEPENDENT_TEST): repaired}
    )
    forged = copy.deepcopy(honest)
    forged["status"], forged["reason"] = "passed", None
    for entry in forged["files"]:
        entry["reviewed"] = entry["complete"] = True

    assert "generalist_review verdict is not bound to the applied file digests" in (
        record_module.generalist_review_record_issues(forged, files)
    )
    # A record in the shape before paths: two digests for three applied paths.
    by_digest = copy.deepcopy(forged)
    by_digest["files"] = [
        {key: value for key, value in entry.items() if key != "path"}
        for entry in forged["files"][1:]
    ]
    assert record_module.generalist_review_record_issues(by_digest, files) == [
        "generalist_review has a malformed nested object"
    ]
    # Leaving the dependent out of the record does not help either.
    two_paths = copy.deepcopy(forged)
    two_paths["files"] = [
        entry for entry in forged["files"] if entry["path"] != _DEPENDENT_TEST
    ]
    assert record_module.generalist_review_record_issues(two_paths, files) == [
        "generalist_review does not list exactly the applied file digests"
    ]


def test_signed_manifest_does_not_credit_a_companion_with_equal_bytes_elsewhere(
    tmp_path,
):
    # The writer, with a fixture signer: a companion at another path that is
    # byte-identical to the reviewed companion is one more applied file.
    payload, applied = _write_manifest(
        tmp_path,
        per_file=True,
        orphan_test=True,
        orphan_bytes=b"[]\n",
        review_state=_bound_state(),
    )

    assert _sha(applied[1]) == _sha(applied[2])
    record = payload[GENERALIST_REVIEW_FIELD]
    assert len(payload["applied_files"]) == len(record["files"]) == 3
    assert (record["status"], record["reason"]) == (
        "partial",
        "unreviewed_applied_files",
    )
    assert {entry["path"]: entry["reviewed"] for entry in record["files"]} == {
        "programs/us-sc/snap/fy-2026.yaml": True,
        "programs/us-sc/snap/fy-2026.test.yaml": True,
        "programs/us-sc/snap/other.test.yaml": False,
    }
    assert not any("generalist_review" in issue for issue in _schema_issues(payload))
    with pytest.raises(ReviewRequirementRefused, match="unreviewed_applied_files"):
        _write_manifest(
            tmp_path / "required",
            per_file=True,
            orphan_test=True,
            orphan_bytes=b"[]\n",
            review_state=_bound_state(required=True),
        )


_OTHER_MODULE = "statutes/26/24/d.yaml"


def _hostile_plans():
    """Plans that try to get review credit for a path the reviewer never read.

    Each is (name, supplemental files, expected status, reviewed paths).
    The reviewed candidate is `_EMPTY_RULE` with `TEST_BYTES` at `_PRIMARY`.
    """

    rule, test = _EMPTY_RULE.decode(), TEST_BYTES.decode()
    primary, primary_test = _PRIMARY.as_posix(), _PRIMARY_TEST
    both = [primary_test, primary]
    partial = "partial"
    return [
        ("no-supplemental", {}, "passed", both),
        ("same-test-at-2-paths", {_DEPENDENT_TEST: test}, partial, both),
        (
            "same-test-at-3-paths",
            {_DEPENDENT_TEST: test, "statutes/26/25.test.yaml": test},
            partial,
            both,
        ),
        (
            "same-test-at-4-paths",
            {
                _DEPENDENT_TEST: test,
                "statutes/26/25.test.yaml": test,
                "regulations/1/2.test.yaml": test,
            },
            partial,
            both,
        ),
        ("same-module-at-2-paths", {_OTHER_MODULE: rule}, partial, both),
        (
            "same-pair-at-2-paths",
            {_OTHER_MODULE: rule, _DEPENDENT_TEST: test},
            partial,
            both,
        ),
        (
            "same-pair-at-4-paths",
            {
                path: content
                for stem in ("statutes/26/24/d", "statutes/26/25", "regulations/1/2")
                for path, content in (
                    (stem + ".yaml", rule),
                    (stem + ".test.yaml", test),
                )
            },
            partial,
            both,
        ),
        # The same directory and the same file name in another directory.
        ("sibling-test", {"statutes/26/32/c/3.test.yaml": test}, partial, both),
        ("same-name-elsewhere", {"statutes/26/33/c/2.test.yaml": test}, partial, both),
        ("same-name-other-root", {"regulations/26/32/c/2.yaml": rule}, partial, both),
        # A path that differs from a reviewed path only by case or by Unicode
        # composition can be the same file: the list binds nothing.
        ("case-of-the-module", {"statutes/26/32/C/2.yaml": rule}, "unavailable", []),
        ("case-of-the-test", {"statutes/26/32/c/2.TEST.yaml": test}, "unavailable", []),
        (
            "case-of-a-dependent",
            {"statutes/26/24/D.test.yaml": test, _DEPENDENT_TEST: test},
            "unavailable",
            [],
        ),
        (
            "unicode-composition",
            {"statutes/26/café.yaml": rule, "statutes/26/café.yaml": rule},
            "unavailable",
            [],
        ),
        # A supplemental file at a reviewed path replaces the planned bytes.
        ("primary-path-same-bytes", {primary: rule}, "passed", both),
        ("primary-test-path-same-bytes", {primary_test: test}, "passed", both),
        (
            "primary-path-other-bytes",
            {primary: rule.replace("[]", "[{name: x}]")},
            "unavailable",
            [],
        ),
        (
            "primary-test-path-other-bytes",
            {primary_test: test.replace("1000", "2000")},
            "unavailable",
            [],
        ),
    ]


@pytest.mark.parametrize(
    ("supplemental", "status", "reviewed_paths"),
    [plan[1:] for plan in _hostile_plans()],
    ids=[plan[0] for plan in _hostile_plans()],
)
def test_review_credit_never_leaves_the_reviewed_paths(
    tmp_path, supplemental, status, reviewed_paths
):
    record, files = _planned_plan_record(
        tmp_path,
        _EMPTY_RULE,
        TEST_BYTES,
        {Path(path): content for path, content in supplemental.items()},
    )

    assert record["status"] == status
    assert [
        entry["path"] for entry in record["files"] if entry["reviewed"]
    ] == reviewed_paths
    assert sorted((entry["path"], entry["sha256"]) for entry in record["files"]) == (
        sorted(files)
    )
    if status != "passed":
        assert review_requirement_refusal(record) is not None
    else:
        assert len(files) == 2
    # The reader agrees with the builder on the same list: it refuses a list
    # with two paths that can be one file, and accepts every other record.
    issues = record_module.generalist_review_record_issues(
        {**record, "required": False}, files
    )
    paths = [path for path, _digest in files]
    if len({record_module._path_collision_key(path) for path in paths}) != len(paths):
        assert status == "unavailable"
        assert issues == [
            "generalist_review files are not sorted and unique",
            "generalist_review does not list exactly the applied file digests",
        ]
    else:
        assert issues == []


@pytest.mark.parametrize(
    "spelling",
    [
        "statutes/26/32/c/./2.yaml",
        "statutes/26/32/c//2.yaml",
        "statutes/26/32/x/../c/2.yaml",
        "./statutes/26/32/c/2.yaml",
        "/statutes/26/32/c/2.yaml",
        "statutes/26/32/c/2.yaml/",
        "statutes\\26\\32\\c\\2.yaml",
        "statutes/26/32/c/2.yaml\n",
        "statutes/26/32/c/2.yaml\x00",
        "statutes/26/32/c/2.yaml ",
        "statutes/26/32/C/2.yaml",
        "STATUTES/26/32/c/2.yaml",
        "statutes/26/32/c/2.YAML",
        "statutes/26/32/c/2.yml",
        "statutes/26/32/c/2",
        "statutes/26/32/c/2.test.yaml",
        "statutes/26/32/c/.yaml",
        "statutes/26/24/d.yaml",
        "",
        None,
        0,
        b"statutes/26/32/c/2.yaml",
        ["statutes/26/32/c/2.yaml"],
        Path("statutes/26/32/c/2.yaml"),
    ],
)
def test_another_spelling_of_the_reviewed_path_binds_nothing(spelling):
    primary = _PRIMARY.as_posix()
    pair = applied_pair(_EMPTY_RULE, TEST_BYTES, rulespec_path=primary)
    files = [(primary, pair.rulespec_sha256), (_PRIMARY_TEST, pair.test_sha256)]
    state = _state(evidence=_evidence(rulespec=_EMPTY_RULE), required=True)
    build = record_module.build_generalist_review_record
    assert build(state, [pair], files, reviewed_rulespec_path=primary)["status"] == (
        "passed"
    )

    # As the path the reviewer was asked about.
    record = build(state, [pair], files, reviewed_rulespec_path=spelling)
    assert (record["status"], record["reason"]) == (
        "unavailable",
        "reviewed_content_not_applied",
    )
    assert not any(entry["reviewed"] for entry in record["files"])
    # As the path of the pair, also when the reviewer is said to have read it.
    if not isinstance(spelling, list):
        moved = AppliedPair(**{**_digests(pair), "rulespec_path": spelling})
        for reviewed_path in (primary, spelling):
            record = build(state, [moved], files, reviewed_rulespec_path=reviewed_path)
            assert record["status"] == "unavailable"
            assert not any(entry["reviewed"] for entry in record["files"])
    # As the listed path of the module file.
    listed = [(spelling, pair.rulespec_sha256), (_PRIMARY_TEST, pair.test_sha256)]
    record = build(state, [pair], listed, reviewed_rulespec_path=primary)
    assert record["status"] == "unavailable"
    assert not any(entry["reviewed"] for entry in record["files"])
    assert review_requirement_refusal(record) is not None


@pytest.mark.parametrize(
    "files",
    [
        [],
        [(_PRIMARY.as_posix(), "0" * 64), (_PRIMARY_TEST, "1" * 64)],
        "swapped",
        "module-only",
        "test-only",
        "test-with-another-digest",
        "one-malformed-item-more",
        "one-bare-digest-more",
        "one-path-two-digests",
        "list-items",
        "bare-digests",
        "three-tuples",
    ],
)
def test_an_applied_file_list_that_does_not_hold_the_pair_binds_nothing(files):
    primary = _PRIMARY.as_posix()
    pair = applied_pair(_EMPTY_RULE, TEST_BYTES, rulespec_path=primary)
    good = [(primary, pair.rulespec_sha256), (_PRIMARY_TEST, pair.test_sha256)]
    files = (
        {
            # Each digest at the other file's path.
            "swapped": [
                (primary, pair.test_sha256),
                (_PRIMARY_TEST, pair.rulespec_sha256),
            ],
            # The companion is on disk and in the pair, but is not applied.
            "module-only": good[:1],
            "test-only": good[1:],
            "test-with-another-digest": [good[0], (_PRIMARY_TEST, "0" * 64)],
            "one-malformed-item-more": [*good, ("statutes/./26/x.yaml", "0" * 64)],
            "one-bare-digest-more": [*good, "0" * 64],
            "one-path-two-digests": [*good, (primary, "0" * 64)],
            "list-items": [list(item) for item in good],
            "bare-digests": [pair.rulespec_sha256, pair.test_sha256],
            "three-tuples": [(*item, True) for item in good],
        }.get(files, files)
        if isinstance(files, str)
        else files
    )

    record = record_module.build_generalist_review_record(
        _state(evidence=_evidence(rulespec=_EMPTY_RULE), required=True),
        [pair],
        files,
        reviewed_rulespec_path=primary,
    )

    assert (record["status"], record["reason"]) == (
        "unavailable",
        "reviewed_content_not_applied",
    )
    assert not any(entry["reviewed"] for entry in record["files"])
    assert record["applied"] == UNBOUND_APPLIED


def test_two_pairs_at_the_reviewed_path_bind_nothing():
    primary = _PRIMARY.as_posix()
    pair = applied_pair(_EMPTY_RULE, TEST_BYTES, rulespec_path=primary)
    files = [(primary, pair.rulespec_sha256), (_PRIMARY_TEST, pair.test_sha256)]
    for second in (
        pair,
        applied_pair(REPAIRED_RULE_BYTES, None, rulespec_path=primary),
    ):
        for pairs in ([pair, second], [second, pair]):
            record = record_module.build_generalist_review_record(
                _state(evidence=_evidence(rulespec=_EMPTY_RULE)),
                pairs,
                files,
                reviewed_rulespec_path=primary,
            )
            assert record["status"] == "unavailable"


def test_a_pair_without_a_path_is_never_bound():
    pair = record_module.applied_pair(RULE_BYTES, TEST_BYTES)
    assert pair.rulespec_path is None and pair.test_path is None

    for reviewed_path in (None, RULE_PATH):
        record = record_module.build_generalist_review_record(
            _state(), [pair], reviewed_rulespec_path=reviewed_path
        )
        assert record["status"] == "unavailable"
        assert record["files"] == []
    # The default of the builder is no reviewed path at all.
    assert (
        record_module.build_generalist_review_record(_state(), APPLIED)["status"]
        == "unavailable"
    )


def _two_path_record():
    """An honest partial record: one digest applied at two companion paths."""

    primary = _PRIMARY.as_posix()
    pair = applied_pair(_EMPTY_RULE, TEST_BYTES, rulespec_path=primary)
    files = [
        (primary, pair.rulespec_sha256),
        (_PRIMARY_TEST, pair.test_sha256),
        (_DEPENDENT_TEST, pair.test_sha256),
    ]
    record = record_module.build_generalist_review_record(
        _state(evidence=_evidence(rulespec=_EMPTY_RULE)),
        [pair],
        files,
        reviewed_rulespec_path=primary,
    )
    assert record["status"] == "partial"
    assert record_module.generalist_review_record_issues(record, files) == []
    return record, files


def _entry(record, path):
    return next(entry for entry in record["files"] if entry["path"] == path)


@pytest.mark.parametrize(
    "forge",
    [
        "credit-the-other-path",
        "move-the-credit",
        "rename-the-other-path-in-record",
        "name-the-other-path-as-the-pair",
        "name-the-other-test-path",
        "drop-the-test-path",
        "drop-the-module-path",
        "test-path-without-test-digest",
        "module-path-is-a-test",
        "module-path-spelling",
        "path-not-a-string",
        "path-case-duplicate",
        "paths-on-an-unbound-record",
        "paths-on-an-unscored-record",
        "test-path-without-a-companion",
        "manifest-lists-the-path-twice",
        "manifest-path-spelling",
        "manifest-digest-at-other-path",
    ],
)
def test_reader_refuses_every_forged_path_binding(forge):
    record, files = _two_path_record()
    primary = _PRIMARY.as_posix()
    forged = copy.deepcopy(record)
    manifest = list(files)
    if forge == "credit-the-other-path":
        _entry(forged, _DEPENDENT_TEST).update(reviewed=True, complete=True)
        forged["status"], forged["reason"] = "passed", None
    elif forge == "move-the-credit":
        _entry(forged, _DEPENDENT_TEST).update(reviewed=True, complete=True)
        _entry(forged, _PRIMARY_TEST).update(reviewed=False, complete=False)
    elif forge == "rename-the-other-path-in-record":
        _entry(forged, _DEPENDENT_TEST)["path"] = "statutes/26/32/c/2.test.yaml.bak"
    elif forge == "name-the-other-path-as-the-pair":
        forged["applied"]["rulespec_path"] = "statutes/26/24/d.yaml"
        forged["applied"]["test_path"] = _DEPENDENT_TEST
    elif forge == "name-the-other-test-path":
        forged["applied"]["test_path"] = _DEPENDENT_TEST
    elif forge == "drop-the-test-path":
        forged["applied"]["test_path"] = None
    elif forge == "drop-the-module-path":
        forged["applied"]["rulespec_path"] = None
        forged["applied"]["test_path"] = None
    elif forge == "test-path-without-test-digest":
        forged["applied"]["test_sha256"] = None
    elif forge == "module-path-is-a-test":
        forged["applied"]["rulespec_path"] = _PRIMARY_TEST
        forged["applied"]["test_path"] = "statutes/26/32/c/2.test.test.yaml"
    elif forge == "module-path-spelling":
        forged["applied"]["rulespec_path"] = "statutes/26/32/c/./2.yaml"
    elif forge == "path-not-a-string":
        _entry(forged, _DEPENDENT_TEST)["path"] = None
    elif forge == "path-case-duplicate":
        forged["files"].append(
            {**_entry(forged, _DEPENDENT_TEST), "path": "statutes/26/24/D.test.yaml"}
        )
        forged["files"].sort(key=lambda entry: entry["path"])
        manifest.append(("statutes/26/24/D.test.yaml", files[2][1]))
    elif forge == "paths-on-an-unbound-record":
        forged["status"], forged["reason"] = (
            "unavailable",
            "reviewed_content_not_applied",
        )
        forged["applied"] = {**UNBOUND_APPLIED, "rulespec_path": primary}
        for entry in forged["files"]:
            entry["reviewed"] = entry["complete"] = False
    elif forge == "paths-on-an-unscored-record":
        forged = record_module.build_generalist_review_record(
            _state(skipped_by_flag=True), [], files
        )
        assert record_module.generalist_review_record_issues(forged, files) == []
        forged["applied"]["rulespec_path"] = primary
    elif forge == "test-path-without-a-companion":
        # A pass on a module without a companion, that names a companion path.
        pair = applied_pair(_EMPTY_RULE, None, rulespec_path=primary)
        manifest = [(primary, pair.rulespec_sha256)]
        forged = record_module.build_generalist_review_record(
            _state(evidence=_evidence(rulespec=_EMPTY_RULE, test=None)),
            [pair],
            manifest,
            reviewed_rulespec_path=primary,
        )
        assert forged["status"] == "passed"
        assert record_module.generalist_review_record_issues(forged, manifest) == []
        forged["applied"]["test_path"] = _PRIMARY_TEST
    elif forge == "manifest-lists-the-path-twice":
        manifest.append((_DEPENDENT_TEST, "0" * 64))
    elif forge == "manifest-path-spelling":
        manifest[2] = ("statutes/26/24/./d.test.yaml", files[2][1])
    elif forge == "manifest-digest-at-other-path":
        manifest[2] = ("statutes/26/25.test.yaml", files[2][1])

    issues = record_module.generalist_review_record_issues(forged, manifest)

    assert issues
    assert all(
        issue.check.startswith("manifest:generalist-review/") for issue in issues
    )


def test_manifest_reader_gives_each_applied_path_to_the_record_check(tmp_path):
    payload, _applied_files = _write_manifest(
        tmp_path,
        per_file=True,
        orphan_test=True,
        orphan_bytes=b"[]\n",
        review_state=_bound_state(),
    )
    baseline = _schema_issues(payload)
    assert not any("generalist_review" in issue for issue in baseline)

    def added(changed):
        return [issue for issue in _schema_issues(changed) if issue not in baseline]

    # Crediting the equal bytes at the other path is refused from the
    # manifest alone, although every listed digest is a reviewed digest.
    forged = copy.deepcopy(payload)
    record = forged[GENERALIST_REVIEW_FIELD]
    record["status"], record["reason"] = "passed", None
    for entry in record["files"]:
        entry["reviewed"] = entry["complete"] = True
    assert added(forged) == [
        "manifest generalist_review verdict is not bound to the applied file digests"
    ]
    # A manifest entry at another path than the record lists is refused.
    renamed = copy.deepcopy(payload)
    renamed["applied_files"][-1]["path"] = "programs/us-sc/snap/renamed.test.yaml"
    assert (
        "manifest generalist_review does not list exactly the applied file digests"
        in added(renamed)
    )


@pytest.mark.parametrize(
    ("path", "valid", "module"),
    [
        ("statutes/26/32/c/2.yaml", True, True),
        ("a.yaml", True, True),
        ("statutes/26/32/c/2.test.yaml", True, False),
        ("statutes/26/32/c/2.json", True, False),
        ("statutes/.yaml", True, False),
        ("statutes/café.yaml", True, True),
        ("", False, False),
        ("/a.yaml", False, False),
        ("a//b.yaml", False, False),
        ("./a.yaml", False, False),
        ("a/../b.yaml", False, False),
        ("a/b.yaml/", False, False),
        ("a\\b.yaml", False, False),
        ("a\tb.yaml", False, False),
        ("a\x7fb.yaml", False, False),
        ("a b.yaml", False, False),
        ("a/" * 2100 + "b.yaml", False, False),
        (None, False, False),
        (b"a.yaml", False, False),
    ],
)
def test_manifest_path_rules(path, valid, module):
    assert record_module.is_manifest_file_path(path) is valid
    assert record_module.is_rulespec_module_path(path) is module


@pytest.mark.parametrize(
    "path",
    ["a.yaml", "statutes/26/32/c/2.yaml", "statutes/a.b.yaml", "x/.hidden.yaml"],
)
def test_companion_manifest_path_is_the_path_apply_writes(path):
    assert record_module.companion_test_manifest_path(path) == (
        cli._rulespec_test_path(Path(path)).as_posix()
    )


def test_a_plan_that_cannot_be_made_lists_no_file_and_binds_nothing(tmp_path):
    output_root, result = _generated_candidate(tmp_path, _EMPTY_RULE, TEST_BYTES)
    assert (
        cli._planned_review_files(
            result, output_root=output_root, supplemental_files=None
        )[2]
        == _PRIMARY.as_posix()
    )

    refused = [
        # Not protected RuleSpec YAML, a traversal, and content that is not text.
        {Path("notes/readme.txt"): "x"},
        {Path("statutes/../statutes/26/24/d.yaml"): "x"},
        {Path(_DEPENDENT_TEST): b"[]\n"},
    ]
    for supplemental in refused:
        assert cli._planned_review_files(
            result, output_root=output_root, supplemental_files=supplemental
        ) == ([], [], None)
    # A candidate that is gone, or outside the output root.
    assert cli._planned_review_files(
        result, output_root=tmp_path / "elsewhere", supplemental_files={}
    ) == ([], [], None)
    Path(result.output_file).unlink()
    plan = cli._planned_review_files(
        result, output_root=output_root, supplemental_files={}
    )
    assert plan == ([], [], None)
    record = record_module.build_generalist_review_record(
        _state(evidence=_evidence(rulespec=_EMPTY_RULE), required=True),
        plan[0],
        plan[1],
        reviewed_rulespec_path=plan[2],
    )
    assert record["status"] == "unavailable"
    assert review_requirement_refusal(record) is not None


# --- review of c3bf0d657: an equal digest is an equal loaded companion ----

# Scalars that `yaml.safe_load` (the loader of companion tests) reads with a
# type, and that a quoted spelling turns into a string.
_LOADER_TYPED_SCALARS = [
    "yes",
    "no",
    "on",
    "off",
    "Yes",
    "NO",
    "On",
    "OFF",
    "y",
    "n",
    "true",
    "False",
    "~",
    "null",
    "1_000",
    "0x10",
    "0b10",
    "0o17",
    "012",
    "1e3",
    "1.5e3",
    ".inf",
    "-.INF",
    ".nan",
    "1:30",
    "1:30.5",
    "2001-01-01",
    "2001-01-01T10:00:00Z",
    "2001-01-01 10:00:00",
    "12",
    "-7",
    "+3",
    "1.0",
    "1.00",
    "0.20",
    "0.2",
    ".2",
    "2.0e-1",
    "0.0",
    "-0.0",
    "0",
    "1.5e+3",
    "1500.0",
    "1_0.5",
    "10.5",
    ".Inf",
    "+.inf",
    ".NaN",
    "1.",
    ".5",
    "=",
    "<<",
    "",
    "plain text",
]


def _scalar_spellings(text):
    """Every way to write one scalar text in a flow-safe position."""

    spellings = [f"'{text}'", f'"{text}"', f"!!str '{text}'", f"! '{text}'"]
    if text:
        spellings += [text, f"! {text}", f"!!str {text}"]
    for tag in ("!!bool", "!!int", "!!float", "!!null", "!!timestamp"):
        spellings.append(f"{tag} '{text}'")
    return spellings


def _typed(value):
    """A loaded value with its types, so that 1, 1.0 and True stay apart."""

    if isinstance(value, dict):
        return (
            "dict",
            sorted((repr(_typed(key)), _typed(item)) for key, item in value.items()),
        )
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [_typed(item) for item in value])
    if isinstance(value, float) and value != value:
        return ("float", "nan")
    return (type(value).__name__, repr(value))


def _loaded(raw):
    try:
        return _typed(yaml.safe_load(raw))
    except Exception as exc:  # noqa: BLE001 - any loader failure is one outcome
        return ("error", type(exc).__name__)


_COMPANION_SHAPES = {
    "value": "- name: case\n  period: 2026\n  output:\n    x: {}\n",
    "input-value": "- name: case\n  period: 2026\n  input:\n    us:a#input.b: {}\n",
    "key": "- name: case\n  period: 2026\n  output:\n    {}: 1\n",
    "list-item": "- name: case\n  period: 2026\n  output:\n    x: [{}, 1]\n",
    "cases-mapping": "cases:\n  - name: case\n    output:\n      x: {}\n",
    "period": "- name: case\n  period: {}\n  output:\n    x: 1\n",
}


@pytest.mark.parametrize("shape", list(_COMPANION_SHAPES))
def test_files_with_one_content_digest_load_as_one_companion(shape):
    # Every spelling of every scalar, in one position of a companion test.
    # Files that share a digest must load as the same typed value: the test
    # runner then reads the same cases from both.
    by_digest: dict[str, set] = {}
    documents = 0
    for text in _LOADER_TYPED_SCALARS:
        for spelling in _scalar_spellings(text):
            raw = _COMPANION_SHAPES[shape].format(spelling).encode()
            documents += 1
            digest = review_content_sha256(raw, rulespec=False)
            if digest is not None:
                by_digest.setdefault(digest, set()).add(repr(_loaded(raw)))

    assert documents > 400
    assert len(by_digest) > 100
    assert {
        digest: values for digest, values in by_digest.items() if len(values) > 1
    } == {}


@pytest.mark.parametrize(
    ("plain", "quoted", "actual"),
    [
        # The two pairs of the review.
        ("0b10", "'0b10'", {"kind": "integer", "value": 2}),
        ("yes", "'yes'", {"kind": "bool", "value": True}),
        ("no", '"no"', {"kind": "bool", "value": False}),
        ("on", "'on'", {"kind": "bool", "value": True}),
        ("off", "'off'", {"kind": "bool", "value": False}),
        ("1_000", "'1_000'", {"kind": "integer", "value": 1000}),
        ("1:30", "'1:30'", {"kind": "integer", "value": 90}),
        ("012", "'012'", {"kind": "integer", "value": 10}),
        ("0x10", "'0x10'", {"kind": "integer", "value": 16}),
        ("~", "'~'", {"kind": "text", "value": None}),
        ("'12'", '! "12"', {"kind": "text", "value": "12"}),
        ("'0b10'", "! '0b10'", {"kind": "text", "value": "0b10"}),
    ],
)
def test_requoting_a_typed_companion_value_never_keeps_the_review(
    tmp_path, plain, quoted, actual
):
    # The probe of the review: the real companion loader, the real scalar
    # comparison and the record builder.
    files = []
    for name, text in (("reviewed", quoted), ("applied", plain)):
        path = tmp_path / f"{name}.test.yaml"
        path.write_text(f"- name: case\n  period: 2026\n  output:\n    x: {text}\n")
        files.append(path)
    reviewed, applied = (path.read_bytes() for path in files)
    loaded = [cli._load_rulespec_test_cases(path)[0]["output"]["x"] for path in files]
    matches = [cli._rulespec_scalar_matches(actual, value) for value in loaded]
    # The two spellings are two tests: they load, and compare, differently.
    assert _typed(loaded[0]) != _typed(loaded[1])
    assert matches[0] is not matches[1]

    assert review_content_sha256(reviewed, rulespec=False) != (
        review_content_sha256(applied, rulespec=False)
    )
    for seen, written in ((reviewed, applied), (applied, reviewed)):
        record = _build(
            applied=[applied_pair(RULE_BYTES, written)],
            evidence=_evidence(test=seen),
            required=True,
        )
        assert (record["status"], record["reason"]) == (
            "unavailable",
            "reviewed_content_not_applied",
        )
        assert review_requirement_refusal(record) is not None
    # The same file on both sides still binds.
    for same in (reviewed, applied):
        assert (
            _build(
                applied=[applied_pair(RULE_BYTES, same)],
                evidence=_evidence(test=same),
                required=True,
            )["status"]
            == "passed"
        )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (b"a: yes\n", b"a: 'yes'\n"),
        (b"a: no\n", b'a: "no"\n'),
        (b"a: on\n", b"a: 'on'\n"),
        (b"a: off\n", b"a: 'off'\n"),
        (b"a: 0b10\n", b"a: '0b10'\n"),
        (b"a: 1_000\n", b"a: '1_000'\n"),
        (b"a: 1:30\n", b"a: '1:30'\n"),
        (b"a: 2001-01-01\n", b"a: '2001-01-01'\n"),
        (b"a: 2018-01-01\n", b"a: '2018-01-01'\n"),
        (b"a: 2001-01-01T10:00:00Z\n", b"a: '2001-01-01T10:00:00Z'\n"),
        (b"a: ~\n", b"a: '~'\n"),
        (b"a: 012\n", b"a: '012'\n"),
        (b"a: 0x10\n", b"a: '0x10'\n"),
        (b"a: 1e3\n", b"a: '1e3'\n"),
        (b"a: .inf\n", b"a: '.inf'\n"),
        (b"a: ! 'yes'\n", b"a: 'yes'\n"),
        (b"a: ! '12'\n", b"a: '12'\n"),
        (b"a: ! |\n  yes\n", b'a: "yes\\n"\n'),
        (b"yes: 1\n", b"'yes': 1\n"),
        (b"no: 1\n", b"'no': 1\n"),
        (b"on: 1\n", b"'on': 1\n"),
        (b"0b10: 1\n", b"'0b10': 1\n"),
        (b"1_000: 1\n", b"'1_000': 1\n"),
        (b"2001-01-01: 1\n", b"'2001-01-01': 1\n"),
        (b"1:30: 1\n", b"'1:30': 1\n"),
        (b"~: 1\n", b"'~': 1\n"),
        (b"012: 1\n", b"'012': 1\n"),
        (b"0x10: 1\n", b"'0x10': 1\n"),
        (b"1e3: 1\n", b"'1e3': 1\n"),
        (b".inf: 1\n", b"'.inf': 1\n"),
    ],
)
def test_a_quoted_scalar_never_has_the_digest_of_the_typed_plain_scalar(left, right):
    # `yaml.safe_load` gives the two spellings different values, except for
    # `1e3`, which only a YAML 1.2 reader types.
    for rulespec in (True, False):
        digests = [
            review_content_sha256(raw, rulespec=rulespec) for raw in (left, right)
        ]
        assert digests[0] != digests[1] or digests == [None, None]


@pytest.mark.parametrize(
    "value",
    [
        "2001-01-01",
        "'2001-01-01'",
        "2001-01-01 10:00:00",
        "'yes'",
        "'0b10'",
        "'1_000'",
        "'1:30'",
        "'012'",
        "'~'",
        "true",
        "~",
        ".inf",
        "1.5",
        "'on'",
    ],
)
def test_scalars_that_the_apply_dump_keeps_still_bind(value):
    # The apply stamp loads and dumps the reviewed module. A scalar whose
    # loaded type and text survive that dump keeps its review.
    reviewed = _rule(value)
    record = _build(
        applied=[applied_pair(_as_apply_rewrites(reviewed), TEST_BYTES)],
        evidence=_evidence(rulespec=reviewed),
        required=True,
    )

    assert record["status"] == "passed"
    assert _issues(record) == []


@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    scalars=st.lists(
        st.tuples(
            st.sampled_from(_LOADER_TYPED_SCALARS),
            st.integers(min_value=0, max_value=11),
            st.sampled_from(_LOADER_TYPED_SCALARS),
            st.integers(min_value=0, max_value=11),
        ),
        min_size=1,
        max_size=3,
        unique_by=lambda item: item[0],
    ),
    respell=st.lists(st.integers(min_value=0, max_value=11), min_size=6, max_size=6),
)
def test_respelled_documents_with_one_digest_load_alike(scalars, respell):
    def document(offsets):
        lines = []
        for index, (key, key_spelling, value, value_spelling) in enumerate(scalars):
            keys = _scalar_spellings(key)
            values = _scalar_spellings(value)
            lines.append(
                f"{keys[(key_spelling + offsets[2 * index]) % len(keys)]}: "
                f"{values[(value_spelling + offsets[2 * index + 1]) % len(values)]}\n"
            )
        return "".join(lines).encode()

    first, second = document([0] * 6), document(respell)
    digests = [review_content_sha256(raw, rulespec=False) for raw in (first, second)]
    if digests[0] is not None and digests[0] == digests[1]:
        assert _loaded(first) == _loaded(second)


# --- review of c3bf0d657: limits that the guide states --------------------


@pytest.mark.parametrize(
    ("comment_length", "status"),
    [(199, "unavailable"), (200, "unavailable"), (201, "unavailable")],
)
def test_guide_bound_of_a_comment_between_a_quoted_key_and_its_colon(
    tmp_path, monkeypatch, comment_length, status
):
    # The raw key scan reads a competing key through a comment of at most 200
    # characters between the key and its colon. A longer comment is refused
    # too: text after the object means the output is not one object.
    output = (
        '{"score":9,"passed":true}\n"Passed" /*' + "x" * comment_length + "*/ : false"
    )
    rule, result = _review(tmp_path, monkeypatch, output)

    assert _record_for(result, rule, required=True)["status"] == status


@pytest.mark.parametrize(
    ("output", "status", "reason"),
    [
        ('{"score": 2}', "failed", None),
        ('{"score": 9}', "unavailable", "unreadable_review_result"),
        ('{"score": 9, "passed": "yes"}', "unavailable", "reviewer_error"),
        ('{"score": 2, "passed": 0}', "unavailable", "reviewer_error"),
    ],
)
def test_guide_statement_of_what_an_inferred_verdict_records(
    tmp_path, monkeypatch, output, status, reason
):
    rule, result = _review(tmp_path, monkeypatch, output)
    record = _record_for(result, rule, required=True)

    assert (record["status"], record["reason"]) == (status, reason)
    assert review_requirement_refusal(record) is not None


def test_accepted_retained_candidate_is_not_reviewed_and_required_mode_refuses(
    tmp_path, capsys
):
    # The guide states this limit. A retry that accepts the retained repair
    # candidate runs no model and no reviewer: the record is `skipped`, and
    # a required review refuses, also where a reviewer CLI could start.
    from axiom_encode.harness.evals import ValidationRetryCandidate, run_model_eval
    from axiom_encode.harness.validator_pipeline import (
        ValidationResult,
        ValidatorPipeline,
    )
    from tests.test_evals import _canonical_rulespec_content_root
    from tests.test_repair_overlay_failure_is_retryable import (
        _bind_corpus,
        _make_workspace,
    )

    output_root = tmp_path / "out"
    workspace = _make_workspace(output_root / "_eval_workspaces" / "workspace")
    rules_path = tmp_path / "rules"
    rules_path.mkdir()
    candidate = ValidationRetryCandidate(
        rulespec="format: rulespec/v1\nmodule: {summary: retained}\nrules: []\n",
        tests="[]\n",
    )
    with (
        patch(
            "axiom_encode.harness.evals.prepare_eval_workspace",
            return_value=workspace,
        ),
        patch(
            "axiom_encode.harness.evals._run_prompt_eval",
            side_effect=AssertionError("model must not run"),
        ),
        patch("axiom_encode.harness.evals._hydrate_eval_root"),
        patch.object(
            ValidatorPipeline,
            "_run_compile_check",
            return_value=ValidationResult("compile", passed=True),
        ),
        patch.object(
            ValidatorPipeline,
            "_run_ci",
            return_value=ValidationResult("ci", passed=True),
        ),
        patch.object(
            validator_pipeline,
            "run_claude_code",
            side_effect=AssertionError("reviewer must not run"),
        ) as reviewer_cli,
    ):
        result = run_model_eval(
            citations=["us-ca/regulation/mpp/63-503"],
            runner_specs=["codex:gpt-5.5"],
            output_root=output_root,
            policy_path=_canonical_rulespec_content_root(tmp_path, "us-ca"),
            runtime_axiom_rules_path=rules_path,
            corpus_release=_bind_corpus(tmp_path),
            mode="cold",
            validation_retry_candidate=candidate,
            accept_valid_retry_candidate=True,
        )[0]

    assert "retained_candidate_preflight=accepted" in capsys.readouterr().out
    assert result.success is True
    reviewer_cli.assert_not_called()
    pairs, files, reviewed_path = cli._planned_review_files(
        result, output_root=output_root, supplemental_files={}
    )
    assert len(files) == 2
    record = record_module.build_generalist_review_record(
        cli._encode_generalist_review_state(
            result, skipped_by_flag=False, required=True
        ),
        pairs,
        files,
        reviewed_rulespec_path=reviewed_path,
    )
    assert (record["status"], record["reason"]) == ("skipped", "reviewer_not_run")
    assert not any(entry["reviewed"] for entry in record["files"])
    assert review_requirement_refusal(record) is not None


_WHOLE_COMPANIONS = [
    b"",
    b"~\n",
    b"null\n",
    b"# only a comment\n",
    b"---\n",
    b"--- ~\n",
    b"[]\n",
    b"[ ]\n",
    b"--- []\n",
    b"[]  # none\n",
    b"{}\n",
    b"''\n",
    b"'[]'\n",
    b"'~'\n",
    b"!!seq []\n",
    b"!!map {}\n",
    b"!!set {}\n",
    b"!!omap []\n",
    b"!!pairs []\n",
    b"!!str\n",
    b"!!null ''\n",
    b"! []\n",
    b"-\n",
    b"- ~\n",
    b"- []\n",
    b"- {}\n",
    b"cases:\n",
    b"cases: ~\n",
    b"cases: []\n",
    b"cases: {}\n",
    b"cases: [{}]\n",
    b"cases: [~]\n",
    b"- name: a\n",
    b"- name: a\n  output: {}\n",
    b"- name: a\n  output:\n",
    b"- name: a\n  output: []\n",
    TEST_BYTES,
]


def test_whole_companion_documents_with_one_digest_load_alike(tmp_path):
    by_digest: dict[str, set] = {}
    outcomes: dict[str, set] = {}
    path = tmp_path / "x.test.yaml"
    for raw in _WHOLE_COMPANIONS:
        digest = review_content_sha256(raw, rulespec=False)
        assert digest is not None
        assert digest != review_content_sha256(None, rulespec=False)
        by_digest.setdefault(digest, set()).add(repr(_loaded(raw)))
        path.write_bytes(raw)
        try:
            outcome = repr(_typed(cli._load_rulespec_test_cases(path)))
        except Exception as exc:  # noqa: BLE001 - any refusal is one outcome
            outcome = f"refused: {type(exc).__name__}"
        outcomes.setdefault(digest, set()).add(outcome)

    assert len(by_digest) > 20
    assert {d: v for d, v in by_digest.items() if len(v) > 1} == {}
    # The companion loader gives one outcome per digest too.
    assert {d: v for d, v in outcomes.items() if len(v) > 1} == {}


def test_inconsistent_absent_companion_digests_bind_nothing():
    absent = review_content_sha256(None, rulespec=False)
    empty_list = review_content_sha256(b"[]\n", rulespec=False)
    # Evidence that names no companion file with the content of a file, and
    # a companion file with the content of an absent one.
    for evidence in (
        _evidence(test=None, reviewed_test_content_sha256=empty_list),
        _evidence(test=b"[]\n", reviewed_test_content_sha256=absent),
    ):
        for test in (None, b"[]\n"):
            record = _build(applied=[applied_pair(RULE_BYTES, test)], evidence=evidence)
            assert record["status"] == "unavailable"
            assert record["reason"] == "unreadable_review_result"
    # The same on the applied side.
    for test_sha256, content in ((None, empty_list), ("1" * 64, absent)):
        pair = AppliedPair(
            **{
                **_digests(applied_pair(RULE_BYTES, None)),
                "test_sha256": test_sha256,
                "test_content_sha256": content,
                "rulespec_path": RULE_PATH,
            }
        )
        for evidence in (_evidence(test=None), _evidence(test=b"[]\n")):
            record = _build(applied=[pair], evidence=evidence)
            assert record["status"] == "unavailable"
            assert record["reason"] == "reviewed_content_not_applied"


def test_a_pair_at_another_path_never_carries_the_verdict():
    # Only a dependent is applied, and it has exactly the reviewed content.
    other = applied_pair(RULE_BYTES, TEST_BYTES, rulespec_path=OTHER_PATH)
    files = [
        (OTHER_PATH, other.rulespec_sha256),
        (other.test_path, other.test_sha256),
    ]
    build = record_module.build_generalist_review_record

    record = build(
        _state(required=True), [other], files, reviewed_rulespec_path=RULE_PATH
    )

    assert (record["status"], record["reason"]) == (
        "unavailable",
        "reviewed_content_not_applied",
    )
    assert not any(entry["reviewed"] for entry in record["files"])
    # The same pair at the reviewed path is the reviewed module.
    assert (
        build(_state(), [other], files, reviewed_rulespec_path=OTHER_PATH)["status"]
        == "passed"
    )


# --- independent review of 9730b4fba ---------------------------------------


@pytest.mark.parametrize(
    "twins",
    [
        (("before", "fy-2025"),),
        (("after", "fy-2027"),),
        (("before", "fy-2025"), ("after", "fy-2027")),
    ],
)
def test_signed_manifest_credits_its_own_module_among_modules_with_equal_bytes(
    tmp_path, twins
):
    # Other applied modules have exactly the reviewed bytes, before and after
    # this encode's module in the applied list. The writer must name its own
    # module as the reviewed one, not the first or the last pair.
    program_bytes = b"program: us-sc/snap\nperiod: 2026-01\n"
    state = _bound_state(reviewed_rulespec=program_bytes, reviewed_test=b"[]\n")
    payload, _applied_files = _write_manifest(
        tmp_path, per_file=True, twin_modules=twins, review_state=state
    )

    record = payload[GENERALIST_REVIEW_FIELD]
    assert len(record["files"]) == 2 + 2 * len(twins)
    assert len({entry["sha256"] for entry in record["files"]}) == 2
    assert (record["status"], record["reason"]) == (
        "partial",
        "unreviewed_applied_files",
    )
    assert record["applied"]["rulespec_path"] == "programs/us-sc/snap/fy-2026.yaml"
    assert record["applied"]["test_path"] == "programs/us-sc/snap/fy-2026.test.yaml"
    assert sorted(entry["path"] for entry in record["files"] if entry["reviewed"]) == [
        "programs/us-sc/snap/fy-2026.test.yaml",
        "programs/us-sc/snap/fy-2026.yaml",
    ]
    assert not any("generalist_review" in issue for issue in _schema_issues(payload))
    with pytest.raises(ReviewRequirementRefused, match="unreviewed_applied_files"):
        _write_manifest(
            tmp_path / "required",
            per_file=True,
            twin_modules=twins,
            review_state=_bound_state(
                reviewed_rulespec=program_bytes, reviewed_test=b"[]\n", required=True
            ),
        )


_ONE_FLOAT_VALUE = [
    ("0.20", "0.2"),
    ("0.200", "0.2"),
    (".2", "0.2"),
    ("+0.2", "0.2"),
    ("2.0e-1", "0.2"),
    ("1762.50", "1762.5"),
    ("1500.00", "1500.0"),
    ("1.", "1.0"),
    ("1.0e+3", "1000.0"),
    ("0.10", "0.1"),
    ("-0.50", "-0.5"),
    ("-0.00", "-0.0"),
    (".Inf", ".inf"),
    ("+.INF", ".inf"),
    ("-.Inf", "-.inf"),
    (".NaN", ".nan"),
    (".NAN", ".nan"),
    (".INF", ".inf"),
    ("-.INF", "-.inf"),
    # Underflow: every modeled reader gives a finite zero.
    ("1.0e-999", "0.0"),
    ("-1.0e-999", "-0.0"),
]


@pytest.mark.parametrize(("spelling", "dumped"), _ONE_FLOAT_VALUE)
def test_a_plain_float_that_every_reader_reads_alike_binds_by_its_value(
    spelling, dumped
):
    # The probe of the review: the model writes `0.20`, the apply dump writes
    # `0.2`. Every modeled reader gives both the same float.
    reviewed = _rule(spelling)
    applied = _as_apply_rewrites(reviewed)
    assert f"value: {dumped}\n".encode() in applied
    assert record_module._agreed_plain_float(spelling) is not None
    assert _loaded(_rule(spelling)) == _loaded(_rule(dumped))

    for seen, written in ((reviewed, applied), (_rule(dumped), applied)):
        record = _build(
            applied=[applied_pair(written, TEST_BYTES)],
            evidence=_evidence(rulespec=seen),
            required=True,
        )
        assert record["status"] == "passed"
        assert review_requirement_refusal(record) is None
        assert _issues(record) == []
    # In a companion test and as a mapping key too.
    for template in ("- output: {{x: {}}}\n", "{}: 1\n"):
        assert review_content_sha256(
            template.format(spelling).encode(), rulespec=False
        ) == review_content_sha256(template.format(dumped).encode(), rulespec=False)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        # A string that holds the number, and a tagged scalar.
        ("0.2", "'0.2'"),
        ("0.20", "'0.20'"),
        ("'0.20'", "'0.2'"),
        ('"x * 0.20"', '"x * 0.2"'),
        ("x * 0.20", "x * 0.2"),
        ("0.20", "!!float 0.20"),
        ("0.20", "!!str 0.20"),
        # An integer is not the float of the same value.
        ("0", "0.0"),
        ("1", "1.0"),
        ("1000", "1.0e+3"),
        # The sign of zero, and values that differ.
        ("-0.0", "0.0"),
        ("0.2", "0.3"),
        ("0.2", "-0.2"),
        (".inf", "-.inf"),
        (".nan", ".inf"),
        (".nan", "0.0"),
        # One dialect reads a string or another number: the text stays exact.
        ("1.5e3", "1500.0"),
        ("1e3", "1000.0"),
        ("1_0.5", "10.5"),
        ("1:30.5", "90.5"),
        ("0.2_0", "0.2"),
        ("1_000.0", "1000.0"),
    ],
)
def test_float_value_binding_never_joins_values_that_a_reader_tells_apart(left, right):
    for template in ("a: {}\n", "- output: {{x: {}}}\n"):
        for rulespec in (True, False):
            digests = [
                review_content_sha256(template.format(text).encode(), rulespec=rulespec)
                for text in (left, right)
            ]
            assert None not in digests
            assert digests[0] != digests[1]
    record = _build(
        applied=[applied_pair(_rule(right), TEST_BYTES)],
        evidence=_evidence(rulespec=_rule(left)),
        required=True,
    )
    assert (record["status"], record["reason"]) == (
        "unavailable",
        "reviewed_content_not_applied",
    )


def test_two_spellings_of_one_float_key_are_a_repeated_key():
    for raw in (b"{0.20: a, 0.2: b}\n", b"{.5: a, 0.50: b}\n", b"{1.0: a, 1: b}\n"):
        assert review_content_sha256(raw, rulespec=False) is None


@pytest.mark.parametrize(
    ("text", "agreed"),
    [
        ("0.20", 0.2),
        (".5", 0.5),
        ("-1.50", -1.5),
        ("1.0e+3", 1000.0),
        ("1.5e3", 1500.0),  # PyYAML resolves this one as a string, not here
        ("1_0.5", None),
        ("1:30.5", None),
        ("abc", None),
        ("", None),
        ("12", None),
        ("0x10", None),
    ],
)
def test_agreed_plain_float(text, agreed):
    assert record_module._agreed_plain_float(text) == agreed


_FLOAT_TEXT = st.one_of(
    st.from_regex(r"[-+]?[0-9]{1,6}\.[0-9]{0,8}([eE][-+][0-9]{1,3})?", fullmatch=True),
    st.from_regex(r"[-+]?\.[0-9]{1,8}([eE][-+]?[0-9]{1,3})?", fullmatch=True),
    st.from_regex(r"[-+]?[0-9_]{1,6}\.[0-9_]{0,6}", fullmatch=True),
    st.from_regex(r"[-+]?[0-9]{1,4}(\.[0-9]{1,4})?[eE][0-9]{1,2}", fullmatch=True),
    st.sampled_from(
        [".inf", "-.inf", "+.Inf", ".nan", ".NaN", "-0.0", "0.0", "0.", "-0."]
    ),
    st.floats(allow_nan=False).map(repr),
    st.from_regex(r"[-+]?[0-9]\.[0-9]e\+(30[0-9]|[4-9][0-9]{2})", fullmatch=True),
    st.from_regex(r"[-+]?[0-9]\.[0-9]e-(3[0-9]{2}|[4-9][0-9]{2})", fullmatch=True),
)


@settings(max_examples=600, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(text=_FLOAT_TEXT, other=_FLOAT_TEXT)
def test_plain_float_spellings_bind_through_the_apply_dump(text, other):
    raw = _rule(text)
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError:
        return
    dumped = apply_stamp_dump(loaded).encode()
    digest, dumped_digest = (
        review_content_sha256(raw, rulespec=True),
        review_content_sha256(dumped, rulespec=True),
    )
    if record_module._agreed_plain_float(text) is not None and isinstance(
        loaded["rules"][0]["value"], float
    ):
        # Every reader agrees on the value: the dump keeps the content.
        assert digest is not None and digest == dumped_digest
        assert dumped_review_content_sha256(raw) == digest
    # Whatever the spelling: one digest means one typed value for PyYAML and
    # for a YAML 1.2 core reader.
    other_raw = _rule(other)
    if digest is not None and digest == review_content_sha256(other_raw, rulespec=True):
        assert _loaded(raw) == _loaded(other_raw)
        yaml12 = [
            record_module._yaml12_core_value(item)
            if record_module.is_yaml12_core_non_string(item)
            else item
            for item in (text, other)
        ]
        assert _typed(yaml12[0]) == _typed(yaml12[1])


def test_guide_limit_of_the_reader_other_bytes_under_the_reviewed_content_digest():
    # The guide states this limit. The reader has only the manifest: it
    # cannot recompute a content digest from a byte digest, so a record that
    # names other bytes under the reviewed content digest is well formed. The
    # manifest signature covers it; the writer builds the record from the
    # files it signs.
    forged = _build()
    other = "9" * 64
    forged["applied"]["rulespec_sha256"] = other
    next(entry for entry in forged["files"] if entry["path"] == RULE_PATH)["sha256"] = (
        other
    )

    assert _issues(forged, [(RULE_PATH, other), (TEST_PATH, PAIR.test_sha256)]) == []
    # The honest manifest list refuses it: the bytes it lists are not these.
    assert _issues(
        forged, [(RULE_PATH, PAIR.rulespec_sha256), (TEST_PATH, PAIR.test_sha256)]
    ) == ["generalist_review does not list exactly the applied file digests"]


@pytest.mark.parametrize(
    ("overflow", "infinity"),
    [
        ("1.0e+999", ".inf"),
        ("1.8e+308", ".inf"),
        ("9.9e+999", ".inf"),
        ("-1.0e+999", "-.inf"),
        ("+1.0e+999", ".inf"),
        ("1.0e+400", ".Inf"),
    ],
)
def test_an_overflowing_float_never_has_the_content_of_infinity(overflow, infinity):
    # PyYAML and YAML 1.2 core read infinity for both, but a reader that
    # accepts only finite parses reads the overflowing spelling as a string.
    assert _loaded(_rule(overflow)) == _loaded(_rule(infinity))
    assert record_module._agreed_plain_float(overflow) is None
    assert record_module._agreed_plain_float(infinity) is not None
    for template in ("a: {}\n", "- output: {{x: {}}}\n", "{}: 1\n", "- {{{}: 1}}\n"):
        for rulespec in (True, False):
            digests = [
                review_content_sha256(template.format(text).encode(), rulespec=rulespec)
                for text in (overflow, infinity)
            ]
            assert None not in digests
            assert digests[0] != digests[1]
    # Both ways, and across the apply dump, which writes `.inf`.
    dumped = _as_apply_rewrites(_rule(overflow))
    assert b"value: " + infinity.lower().encode() + b"\n" in dumped
    assert dumped_review_content_sha256(_rule(overflow)) is None
    for seen, written in (
        (_rule(overflow), _rule(infinity)),
        (_rule(infinity), _rule(overflow)),
        (_rule(overflow), dumped),
    ):
        record = _build(
            applied=[applied_pair(written, TEST_BYTES)],
            evidence=_evidence(rulespec=seen),
            required=True,
        )
        assert (record["status"], record["reason"]) == (
            "unavailable",
            "reviewed_content_not_applied",
        )
    # The same overflowing text on both sides is the same content.
    same = _build(
        applied=[applied_pair(_rule(overflow), TEST_BYTES)],
        evidence=_evidence(rulespec=_rule(overflow)),
    )
    assert same["status"] == "passed"


# --- the verdict reader: value types, strict JSON, prompt and backends ------


@pytest.mark.parametrize(
    "output",
    [
        # A contradiction one level down, under a schema key.
        '{"passed": true, "score": 9, "non_blocking_issues": [{"pass": false}]}',
        '{"passed": true, "score": 9, "non_blocking_issues": [{"failed": true}]}',
        '{"passed": true, "score": 9, "non_blocking_issues": [{"verdict": "fail"}]}',
        '{"passed": true, "score": 9, "blocking_issues": [], '
        '"non_blocking_issues": [{"blocking": true, "issue": "wrong rate"}]}',
        '{"passed": true, "score": 9, "reasoning": {"verdict": "fail", "pass": false}}',
        '{"passed": true, "score": 9, "reasoning": {"final": {"result": "REJECT"}}}',
        '{"passed": true, "score": 9, "reasoning": {"score": 1}}',
        '{"passed": true, "score": 9, "reasoning": {"issues": ["x"]}}',
        '{"passed": true, "score": 9, "reasoning": [{"reasoning": "nested"}]}',
        # Another type where the schema has a string or a list of strings.
        '{"passed": true, "score": 9, "reasoning": ["ok"]}',
        '{"passed": true, "score": 9, "reasoning": null}',
        '{"passed": true, "score": 9, "reasoning": 1}',
        '{"passed": true, "score": 9, "blocking_issues": "none"}',
        '{"passed": true, "score": 9, "blocking_issues": null}',
        '{"passed": true, "score": 9, "blocking_issues": {}}',
        '{"passed": true, "score": 9, "non_blocking_issues": [1]}',
        '{"passed": true, "score": 9, "non_blocking_issues": [null]}',
        '{"passed": true, "score": 9, "non_blocking_issues": [true]}',
        '{"passed": true, "score": 9, "non_blocking_issues": [["nit"]]}',
        '{"passed": true, "score": 9, "non_blocking_issues": ["nit", {"x": "y"}]}',
        # No score.
        '{"passed": true}',
        '{"passed": true, "blocking_issues": [], "reasoning": "ok"}',
        # Not strict JSON.
        '{"passed": true, "score": 9, "blocking_issues": [,]}',
        '{"passed": true, "score": 9, "blocking_issues": [ , ]}',
        '{"passed": true, "score": 9, "blocking_issues": ["a",,]}',
        '{"passed": true, "score": 9,, "blocking_issues": []}',
        '{,"passed": true, "score": 9}',
        '{"passed": true, "score": 9, "non_blocking_issues": [NaN]}',
        '{"passed": true, "score": NaN}',
        '{"passed": true, "score": Infinity}',
        '{"passed": true, "score": -Infinity}',
        '{"passed": true, "score": 9, "reasoning": "a\nb"}',
        '{"passed": true, "score": 9, "reasoning": "a\tb"}',
        '{"passed": true, "score": 9, "reasoning": "x\n```\nThis fails.\n```json\ny"}',
        '```json\n{"passed": true, "score": 9, "reasoning": "x\n```\nFAIL\n```json\ny"}\n```',
        '{"passed": true,\u3000"score": 9}',
        '{"passed": true,\u00a0"score": 9}',
        '{"passed": true, "score": 9}\u3000',
        '\u3000{"passed": true, "score": 9}',
        '{"passed": true, "score": 9}\u200b',
        '{"passed": true, "score": 9}\x00',
        '{"passed": true, "score": 9}\x0c',
        '\x0b{"passed": true, "score": 9}\x1c',
        '```json\n{"passed": true, "score": 9}\n```\u200b',
        '```json\n\u3000{"passed": true, "score": 9}\n```',
        '```json\n{"passed": true, "score": 9}\u00a0\n```',
        # A byte order mark that is not the first character.
        ' \ufeff{"passed": true, "score": 9}',
        '\n\ufeff{"passed": true, "score": 9}',
        '```json\n\ufeff{"passed": true, "score": 9}\n```',
        # A fence whose body is not on its own lines.
        '```json {"passed": true, "score": 9}\n```',
        '```{"passed": true, "score": 9}\n```',
        '```json\n{"passed": true, "score": 9}```',
        '```json\n{"passed": true, "score": 9} ```',
        '```json{"passed": true, "score": 9}```',
    ],
)
def test_output_outside_the_strict_typed_schema_never_records_a_pass(
    tmp_path, monkeypatch, output
):
    rule, result = _review(tmp_path, monkeypatch, output)
    record = _record_for(result, rule, required=True)

    assert result.passed is False
    assert result.score is None
    assert "reviewer_parse_failed" in result.issues
    assert (record["status"], record["reason"]) == ("unavailable", "reviewer_error")
    assert review_requirement_refusal(record) is not None


@pytest.mark.parametrize(
    "output",
    [
        '```json\r\n{"passed": true, "score": 9}\r\n```',
        '```\r\n{"passed": true, "score": 9}\r\n```\r\n',
        '{\r\n  "passed": true,\r\n  "score": 9\r\n}\r\n',
        '\t{"passed": true, "score": 9}\t',
        '{"passed": true, "score": 9, "reasoning": "a\\nb, } and ,] stay"}',
        '{"passed": true, "score": 9, "non_blocking_issues": ["a", "b",],}',
        '{"passed": true, "score": 9, "non_blocking_issues": ["a" , ] , }',
        '{"passed": true, "score": 9.0, "blocking_issues": [], "reasoning": ""}',
    ],
)
def test_strict_typed_schema_object_records_a_pass(tmp_path, monkeypatch, output):
    rule, result = _review(tmp_path, monkeypatch, output)
    record = _record_for(result, rule, required=True)

    assert result.passed is True
    assert record["status"] == "passed"


def test_a_failure_without_a_score_is_unreadable_not_failed(tmp_path, monkeypatch):
    rule, result = _review(tmp_path, monkeypatch, '{"passed": false}')
    record = _record_for(result, rule, required=True)

    assert "reviewer_parse_failed" in result.issues
    assert (record["status"], record["reason"]) == ("unavailable", "reviewer_error")


@pytest.mark.parametrize("reviewer_type", _REVIEWER_TYPES)
def test_each_reviewer_type_reads_only_the_value_types_of_its_schema(
    tmp_path, monkeypatch, reviewer_type
):
    schema = validator_pipeline._reviewer_schema(reviewer_type)
    own = {key: _SCHEMA_VALUES[key] for key in sorted(schema)}
    wrong = {
        "number": ["9", True, None, [9], {"value": 9}],
        "boolean": [1, 0, "true", None, [True]],
        "string": [None, 1, True, ["ok"], {"text": "ok"}],
        "string list": ["fatal: wrong", None, {}, [1], [None], [["x"]], [{"a": "b"}]],
    }
    assert set(schema.values()) == set(wrong)

    for key, kind in schema.items():
        for value in wrong[kind]:
            _rule_path, result = _review(
                tmp_path,
                monkeypatch,
                json.dumps({**own, key: value}),
                reviewer=reviewer_type,
            )
            assert result.passed is False, (key, value)
            assert result.score is None, (key, value)
            assert "reviewer_parse_failed" in result.issues, (key, value)
    _rule_path, result = _review(
        tmp_path, monkeypatch, json.dumps(own), reviewer=reviewer_type
    )
    assert result.passed is True
    without_score = {key: value for key, value in own.items() if key != "score"}
    _rule_path, result = _review(
        tmp_path, monkeypatch, json.dumps(without_score), reviewer=reviewer_type
    )
    assert "reviewer_parse_failed" in result.issues


def test_reviewer_schema_types_are_exact():
    assert dict(validator_pipeline._reviewer_schema("generalist-reviewer")) == {
        "score": "number",
        "passed": "boolean",
        "blocking_issues": "string list",
        "non_blocking_issues": "string list",
        "reasoning": "string",
    }
    for reviewer_type in _REVIEWER_TYPES[1:] + ["any other reviewer"]:
        assert dict(validator_pipeline._reviewer_schema(reviewer_type)) == {
            "score": "number",
            "passed": "boolean",
            "issues": "string list",
            "reasoning": "string",
        }


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1, "a": 2}',
        '{"a": {"b": 1, "b": 2}}',
        '[{"a": 1, "a": 1}]',
        "NaN",
        "[Infinity]",
        '{"a": -Infinity}',
        '"a\nb"',
        "[1] [2]",
        "[1,,]",
        "[,]",
        "",
        "```json\n```",
    ],
)
def test_whole_output_decoder_refuses_what_strict_json_refuses(text):
    with pytest.raises(ValueError):
        validator_pipeline._decode_whole_reviewer_output(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[1,]", "[1]"),
        ('{"a": 1 , }', '{"a": 1  }'),
        ('{"a": [1,\n],}', '{"a": [1\n]}'),
        ('["a,]", "b,}",]', '["a,]", "b,}"]'),
        ('["a\\",]",]', '["a\\",]"]'),
        ("[,]", "[,]"),
        ("[1,,]", "[1,,]"),
        ("{,}", "{,}"),
        ('{"a":,}', '{"a":,}'),
        ("[1,\u3000]", "[1,\u3000]"),
        ("[1, 2]", "[1, 2]"),
    ],
)
def test_only_a_comma_between_a_value_and_its_closer_is_dropped(text, expected):
    assert validator_pipeline._drop_trailing_json_commas(text) == expected


_REPLY_SHAPE = (
    "Your whole reply must be exactly one JSON object with exactly these keys: "
    "no text before or after it and no other key. `score` is a number, `passed` "
    "is true or false, every issue is one string and `reasoning` is one string. "
    "One ```json code fence around the object is allowed."
)


@pytest.mark.parametrize("reviewer_type", _REVIEWER_TYPES)
def test_every_reviewer_prompt_states_the_reply_shape_once(
    tmp_path, monkeypatch, reviewer_type
):
    prompts = []

    def reviewer(prompt, **_kwargs):
        prompts.append(prompt)
        return '{"score": 9, "passed": true}', 0

    monkeypatch.setattr(validator_pipeline, "run_claude_code", reviewer)
    _pipeline(tmp_path)._run_reviewer(reviewer_type, _candidate(tmp_path))
    (prompt,) = prompts

    assert validator_pipeline._REVIEW_REPLY_SHAPE == _REPLY_SHAPE
    assert prompt.count(_REPLY_SHAPE) == 1
    # Directly after the JSON object that the prompt asks for.
    assert '"reasoning": "<brief explanation>"\n}\n\n' + _REPLY_SHAPE in prompt


def _subprocess_result(stdout, stderr, returncode):
    script = (
        "import sys; sys.stdout.write(sys.argv[1]); "
        "sys.stderr.write(sys.argv[2]); sys.exit(int(sys.argv[3]))"
    )
    return validator_pipeline._run_subprocess_with_idle_timeout(
        [sys.executable, "-c", script, stdout, stderr, str(returncode)],
        timeout=60,
        idle_timeout=60,
        poll_interval=0.01,
    )


def test_claude_reply_is_stdout_alone_when_the_cli_exits_zero(caplog):
    result = _subprocess_result(_GENERALIST_PASS, "warning: slow start\n", 0)

    assert result.stdout == _GENERALIST_PASS
    assert result.output == _GENERALIST_PASS + "warning: slow start\n"
    with caplog.at_level("WARNING", logger=validator_pipeline.logger.name):
        reply = validator_pipeline._claude_reviewer_reply(result)
    assert reply == _GENERALIST_PASS
    assert "stderr" in caplog.text
    assert validator_pipeline._read_reviewer_verdict(reply, "generalist-reviewer")
    # Stderr alone is never a reply.
    quiet = _subprocess_result("", _GENERALIST_PASS, 0)
    assert validator_pipeline._claude_reviewer_reply(quiet) == ""
    # A failed run keeps both streams, for the error message of the caller.
    failed = _subprocess_result("out", "err", 3)
    assert validator_pipeline._claude_reviewer_reply(failed) == "outerr"


def test_claude_run_with_stderr_noise_records_a_pass_and_a_failed_run_does_not(
    tmp_path, monkeypatch
):
    rule = _candidate(tmp_path)
    runs = iter(
        [
            _subprocess_result(_GENERALIST_PASS, "warning: slow start\n", 0),
            # A stale passing verdict on stdout of a run that failed.
            _subprocess_result(_GENERALIST_PASS, "fatal\n", 3),
            _subprocess_result(_GENERALIST_PASS, "", 3),
        ]
    )
    monkeypatch.setattr(
        validator_pipeline,
        "_run_subprocess_with_idle_timeout",
        lambda *_a, **_k: next(runs),
    )
    monkeypatch.delenv("AXIOM_ENCODE_REVIEWER_CLI", raising=False)
    monkeypatch.setattr(
        validator_pipeline, "_local_reviewer_context_active", lambda: False
    )

    passed = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)
    assert passed.passed is True
    assert _record_for(passed, rule, required=True)["status"] == "passed"
    for _ in range(2):
        failed = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)
        record = _record_for(failed, rule, required=True)
        assert failed.passed is False
        assert (record["status"], record["reason"]) == (
            "unavailable",
            "reviewer_error",
        )


def _codex_stream(*messages):
    return "\n".join(
        json.dumps(
            {"type": "item.completed", "item": {"type": "agent_message", "text": text}}
        )
        for text in messages
    )


_GENERALIST_FAIL = (
    '{"score": 2, "passed": false, "blocking_issues": ["bad"], '
    '"non_blocking_issues": [], "reasoning": "no"}'
)


def _codex_record(tmp_path, monkeypatch, lines, *, stderr="", returncode=0):
    """The record of one Codex reviewer run with this stdout event stream."""

    stdout = "".join(line + "\n" for line in lines)
    monkeypatch.setenv("AXIOM_ENCODE_REVIEWER_CLI", "codex")
    monkeypatch.setattr(
        validator_pipeline, "_local_reviewer_context_active", lambda: False
    )
    monkeypatch.setattr(validator_pipeline, "resolve_codex_cli", lambda: "codex")
    monkeypatch.setattr(validator_pipeline, "supervisor_codex_home", lambda: None)
    monkeypatch.setattr(validator_pipeline, "codex_auth_secrets", lambda _home: set())
    monkeypatch.setattr(
        validator_pipeline,
        "_run_subprocess_with_idle_timeout",
        lambda *_a, **_k: validator_pipeline._SubprocessRunResult(
            output=stdout + stderr, returncode=returncode, stdout=stdout
        ),
    )
    rule = _candidate(tmp_path)
    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)
    return result, _record_for(result, rule, required=True)


def test_codex_reply_is_the_one_agent_message(tmp_path, monkeypatch):
    extract = validator_pipeline._extract_codex_text_output
    other_events = [
        json.dumps({"type": "thread.started", "thread_id": "t"}),
        json.dumps({"type": "turn.started"}),
        json.dumps(
            {"type": "item.completed", "item": {"type": "reasoning", "text": "x"}}
        ),
        _codex_stream(" " + _GENERALIST_PASS + "\n"),
        # An agent message without text is not a second message.
        _codex_stream("  \n"),
        json.dumps({"type": "turn.completed", "usage": {}}),
    ]

    assert extract(_codex_stream(_GENERALIST_PASS)) == _GENERALIST_PASS
    assert extract("\n".join(other_events) + "\n") == _GENERALIST_PASS
    result, record = _codex_record(
        tmp_path, monkeypatch, other_events, stderr="warning: slow start\n"
    )
    assert result.passed is True
    assert record["status"] == "passed"
    assert record["reviewer"]["backend"] == "codex"


@pytest.mark.parametrize("character", ["\u0085", "\u2028", "\u2029", "\x0b", "\x0c"])
def test_codex_event_line_ends_only_at_a_line_feed(tmp_path, monkeypatch, character):
    # JSON allows U+0085, U+2028 and U+2029 raw inside a string, and
    # `str.splitlines` breaks a line at each of them (and at an escaped-away
    # form feed or vertical tab in the outer event, which stay escaped here).
    verdict = json.loads(_GENERALIST_PASS)
    verdict["reasoning"] = f"line one{character}line two"
    reply = json.dumps(verdict, ensure_ascii=False)
    raw = character in "\u0085\u2028\u2029"
    assert (character in reply) is raw
    lines = [
        json.dumps({"type": "thread.started", "thread_id": "t"}),
        json.dumps({"type": "turn.started"}),
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": reply},
            },
            ensure_ascii=False,
        ),
        json.dumps({"type": "turn.completed", "usage": {}}),
    ]
    assert (character in lines[2]) is raw

    assert validator_pipeline._extract_codex_text_output("\n".join(lines)) == reply
    # A carriage return before the line feed is not part of the event.
    assert validator_pipeline._extract_codex_text_output("\r\n".join(lines)) == reply
    blank_lines = "\r\n \t\r\n".join(lines) + "\r\n\r\n"
    assert validator_pipeline._extract_codex_text_output(blank_lines) == reply
    # The character outside a string is not JSON whitespace: not an event.
    beside = "\n".join([*lines[:2], character + lines[2], lines[3]])
    assert validator_pipeline._extract_codex_text_output(beside).startswith(
        "Codex reviewer stream is unreadable (a line is not a JSON object)"
    )
    result, record = _codex_record(tmp_path, monkeypatch, lines)
    assert (result.passed, result.score) == (True, 9)
    assert record["status"] == "passed"
    # The same reply on the Claude path.
    rule, claude = _review(tmp_path, monkeypatch, reply)
    assert (claude.passed, claude.score) == (True, 9)
    assert _record_for(claude, rule, required=True)["status"] == "passed"


def test_reviewer_issues_are_the_issue_lists_of_the_verdict(tmp_path, monkeypatch):
    _rule_path, generalist = _review(
        tmp_path,
        monkeypatch,
        '{"score": 9, "passed": true, "blocking_issues": [], '
        '"non_blocking_issues": ["rename the helper", "dead code"]}',
    )
    assert generalist.passed is True
    assert generalist.issues == [
        "[non-blocking] rename the helper",
        "[non-blocking] dead code",
    ]
    _rule_path, failed = _review(
        tmp_path,
        monkeypatch,
        '{"score": 2, "passed": false, "blocking_issues": ["wrong rate"], '
        '"non_blocking_issues": ["dead code"]}',
    )
    assert failed.issues == ["wrong rate", "[non-blocking] dead code"]
    _rule_path, other = _review(
        tmp_path,
        monkeypatch,
        '{"score": 3, "passed": false, "issues": ["wrong rate", "no test"]}',
        reviewer="rulespec-reviewer",
    )
    assert (other.passed, other.score) == (False, 3)
    assert other.issues == ["wrong rate", "no test"]


def test_codex_event_split_by_a_line_feed_is_unreadable(tmp_path, monkeypatch):
    event = _codex_stream(_GENERALIST_PASS)
    middle = len(event) // 2
    result, record = _codex_record(
        tmp_path, monkeypatch, [event[:middle], event[middle:]]
    )

    assert result.passed is False
    assert (record["status"], record["reason"]) == ("unavailable", "reviewer_error")


_CODEX_EARLIER_MESSAGES = [
    # The seven inputs of the review of a68ea4b17: no brace, no quoted key.
    "passed: false\nscore: 2\nblocking_issues:\n- the rate is wrong",
    "score: 2, passed: false",
    "verdict: FAIL",
    "{ passed false }",
    "I must fail this: the rate is wrong.",
    "'passed': false",
    "dict(passed=False, score=2)",
    # More.
    _GENERALIST_FAIL,
    "```json\n" + _GENERALIST_FAIL + "\n```",
    "I will review the file now.",
    _GENERALIST_PASS,
    ".",
]


@pytest.mark.parametrize("other", _CODEX_EARLIER_MESSAGES)
@pytest.mark.parametrize("position", ["before", "after", "first of three"])
def test_codex_run_with_a_second_agent_message_is_unreadable(
    tmp_path, monkeypatch, other, position
):
    messages = {
        "before": [other, _GENERALIST_PASS],
        "after": [_GENERALIST_PASS, other],
        "first of three": [other, "Now the verdict.", _GENERALIST_PASS],
    }[position]
    lines = [_codex_stream(message) for message in messages]

    reply = validator_pipeline._extract_codex_text_output("\n".join(lines))
    assert reply.startswith("Codex reviewer stream is unreadable (")
    assert f"{len(messages)} agent messages with text, not one" in reply
    with pytest.raises(ValueError):
        validator_pipeline._read_reviewer_verdict(reply, "generalist-reviewer")
    result, record = _codex_record(tmp_path, monkeypatch, lines)
    assert result.passed is False
    assert (record["status"], record["reason"]) == ("unavailable", "reviewer_error")
    assert review_requirement_refusal(record) is not None


_CODEX_PASS_EVENT = _codex_stream(_GENERALIST_PASS)


@pytest.mark.parametrize(
    ("lines", "reason"),
    [
        # An error or a failed turn anywhere, also with exit 0.
        (
            [json.dumps({"type": "error", "message": "boom"}), _CODEX_PASS_EVENT],
            "error",
        ),
        ([_CODEX_PASS_EVENT, json.dumps({"type": "error", "message": "x"})], "error"),
        ([_CODEX_PASS_EVENT, json.dumps({"type": "error"})], "error"),
        (
            [
                _CODEX_PASS_EVENT,
                json.dumps({"type": "turn.failed", "error": {"message": "x"}}),
            ],
            "error",
        ),
        ([json.dumps({"type": "turn.failed"}), _CODEX_PASS_EVENT], "error"),
        # A line that is not a JSON object.
        (
            [_codex_stream(_GENERALIST_FAIL)[:-5], _CODEX_PASS_EVENT],
            "not a JSON object",
        ),
        ([_CODEX_PASS_EVENT, "warning: not an event"], "not a JSON object"),
        (["[]", _CODEX_PASS_EVENT], "not a JSON object"),
        ([_CODEX_PASS_EVENT, '"text"'], "not a JSON object"),
        (
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": [_GENERALIST_PASS]},
                    }
                )
            ],
            "text is not a string",
        ),
        # No agent message: a bare verdict is not a Codex event stream, and a
        # verdict in another item or another event is not the reply.
        ([_GENERALIST_PASS], "0 agent messages"),
        ([], "0 agent messages"),
        ([json.dumps({"type": "turn.completed"})], "0 agent messages"),
        (
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "reasoning", "text": _GENERALIST_PASS},
                    }
                )
            ],
            "0 agent messages",
        ),
        (
            [
                json.dumps(
                    {
                        "type": "item.started",
                        "item": {"type": "agent_message", "text": _GENERALIST_PASS},
                    }
                )
            ],
            "0 agent messages",
        ),
    ],
)
def test_codex_stream_that_is_not_one_clean_reply_is_unreadable(
    tmp_path, monkeypatch, lines, reason
):
    reply = validator_pipeline._extract_codex_text_output("\n".join(lines))
    assert reply.startswith("Codex reviewer stream is unreadable (")
    assert reason in reply.splitlines()[0]
    with pytest.raises(ValueError):
        validator_pipeline._read_reviewer_verdict(reply, "generalist-reviewer")

    result, record = _codex_record(tmp_path, monkeypatch, lines)
    assert result.passed is False
    assert (record["status"], record["reason"]) == ("unavailable", "reviewer_error")


def test_codex_stream_is_stdout_on_exit_zero_and_a_failed_run_has_no_reply(
    tmp_path, monkeypatch
):
    # A verdict event on stderr is not part of the stream.
    result, record = _codex_record(
        tmp_path,
        monkeypatch,
        [_codex_stream(_GENERALIST_FAIL)],
        stderr=_CODEX_PASS_EVENT + "\n",
    )
    assert (result.score, record["status"]) == (2, "failed")
    result, record = _codex_record(
        tmp_path, monkeypatch, [], stderr=_CODEX_PASS_EVENT + "\n"
    )
    assert record["status"] == "unavailable"
    # A clean passing stream of a run that exits non-zero.
    result, record = _codex_record(
        tmp_path, monkeypatch, [_CODEX_PASS_EVENT], returncode=3
    )
    assert result.passed is False
    assert (record["status"], record["reason"]) == ("unavailable", "reviewer_error")


# The strict reading of a reviewer output, written apart from the reader.
_FUZZ_FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n(.*)\r?\n[ \t]*```", re.S | re.I)
_FUZZ_TYPES = {
    "score": lambda v: type(v) in (int, float),
    "passed": lambda v: type(v) is bool,
    "reasoning": lambda v: type(v) is str,
    "blocking_issues": lambda v: type(v) is list and all(type(i) is str for i in v),
    "non_blocking_issues": lambda v: type(v) is list and all(type(i) is str for i in v),
}


def _fuzz_strict_load(text):
    def pairs(items):
        keys = [key for key, _value in items]
        if len(set(keys)) != len(keys):
            raise ValueError("repeated key")
        return dict(items)

    def constant(name):
        raise ValueError(name)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _fuzz_strict_verdicts(output):
    """Every verdict that strict JSON gives for ``output``: none or one.

    The body is the output without one leading byte order mark, JSON
    whitespace and one optional fence. It is read as it is, and with each
    choice of commas dropped from before a closer, when a value stands
    before the comma.
    """
    body = output.removeprefix("\ufeff").strip(" \t\n\r")
    fenced = _FUZZ_FENCE.fullmatch(body)
    if fenced:
        body = fenced.group(1).strip(" \t\n\r")
    commas = [
        match.start()
        for match in re.finditer(r",(?=[ \t\n\r]*[}\]])", body)
        if body[: match.start()].rstrip(" \t\n\r")[-1:] not in ("", "[", "{", ",", ":")
    ]
    # The fewest dropped commas that give valid JSON: a comma inside a string
    # is never one of them, so the strings stay as the reviewer wrote them.
    for count in range(min(len(commas), 6) + 1):
        decoded = []
        for dropped in itertools.combinations(commas[:6], count):
            text = "".join(c for i, c in enumerate(body) if i not in dropped)
            with contextlib.suppress(ValueError, RecursionError):
                decoded.append(_fuzz_strict_load(text))
        if decoded:
            return [
                data
                for data in decoded
                if type(data) is dict
                and "score" in data
                and all(k in _FUZZ_TYPES and _FUZZ_TYPES[k](v) for k, v in data.items())
            ]
    return []


_FUZZ_BASES = [
    _GENERALIST_PASS,
    '{"passed": true, "score": 9}',
    '{"score": 8.5, "passed": true, "blocking_issues": [], '
    '"non_blocking_issues": ["nit",], "reasoning": "fine, {really},}",}',
    json.dumps(json.loads(_GENERALIST_PASS), indent=2),
    '{"score": 2, "passed": false, "blocking_issues": ["bad"], "reasoning": "no"}',
    "```json\n" + _GENERALIST_PASS + "\n```",
]
_FUZZ_FRAGMENTS = [
    *'{}[],:"\\\n\t\r ',
    "\ufeff",
    "\u3000",
    "\u200b",
    "\x00",
    "\x0b",
    "true",
    "false",
    "null",
    "1",
    '"9"',
    "NaN",
    "Infinity",
    "1e999",
    '"passed"',
    '"score"',
    '"pass"',
    '"verdict"',
    '"Passed"',
    '"issues"',
    '"passed": false',
    ', "passed": false',
    ', "pass": false',
    ', "verdict": "fail"',
    "```",
    "```json\n",
    "\n```",
    "```json\r\n",
    "\r\n```",
    "/* c */",
    "// c\n",
    '{"passed": false}',
    '[{"passed": false}]',
    " ,",
    ",}",
    ",]",
    '"x": ',
    "Looks fine.",
    '"score": "9"',
    '"reasoning": {"passed": false}',
    '"non_blocking_issues": [{"pass": false}]',
    '"reasoning": "',
]


def _fuzz_mutate(rng, text):
    for _ in range(rng.choice([1, 1, 1, 2, 2, 3])):
        operation = rng.randrange(8)
        index = rng.randrange(len(text) + 1)
        fragment = rng.choice(_FUZZ_FRAGMENTS)
        if operation <= 2:
            text = text[:index] + fragment + text[index:]
        elif operation == 3:
            text = text[:index] + text[index + rng.choice([1, 1, 2, 8]) :]
        elif operation == 4:
            text = text[:index] + fragment + text[index + 1 :]
        elif operation == 5:
            text = rng.choice(["```json\n", "```\n", "```json\r\n", "``` json\n"]) + (
                text + rng.choice(["\n```", "```", "\r\n```\n", "\n```\nDone."])
            )
        elif operation == 6:
            text = rng.choice([" ", "\n", "\ufeff", "\ufeff ", " \ufeff"]) + text
        else:
            text = text + rng.choice([" ", "\n", ",", fragment])
    return text


def test_reader_gives_a_verdict_only_where_strict_json_gives_that_verdict():
    """Fixed-seed fuzz: the reader never reads more than strict JSON does."""

    rng = random.Random(2229)
    outputs = set(_FUZZ_BASES)
    while len(outputs) < 4000:
        outputs.add(_fuzz_mutate(rng, rng.choice(_FUZZ_BASES)))
    read = passes = 0
    for output in sorted(outputs):
        try:
            verdict = validator_pipeline._read_reviewer_verdict(
                output, "generalist-reviewer"
            )
        except ValueError:
            continue
        read += 1
        passes += verdict.get("passed") is True
        assert _fuzz_strict_verdicts(output) == [verdict], output
    for base in _FUZZ_BASES:
        assert _fuzz_strict_verdicts(base) == [
            validator_pipeline._read_reviewer_verdict(base, "generalist-reviewer")
        ]
    # The property is not empty: the fuzz reads verdicts and refuses outputs.
    assert passes >= 50
    assert len(outputs) - read >= 1000
