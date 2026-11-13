"""The generalist review record and the --require-review gate (#2146)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

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
    REQUIRE_REVIEW_ENV,
    AppliedPair,
    GeneralistReviewState,
    applied_pair,
    build_generalist_review_record,
    generalist_review_record_issues,
    require_review_setting,
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


def _digests(pair):
    return {field: getattr(pair, field) for field in DIGEST_FIELDS}


def _evidence(rulespec=RULE_BYTES, test=TEST_BYTES, **overrides):
    evidence = {
        **review_evidence_digests(rulespec, test),
        "content_complete": True,
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


def _build(applied=None, **overrides):
    return build_generalist_review_record(
        _state(**overrides), APPLIED if applied is None else applied
    )


def _issues(record, applied_sha256s=None):
    return generalist_review_record_issues(
        record, APPLIED_SHA256S if applied_sha256s is None else applied_sha256s
    )


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
        RULE_BYTES.replace(b"'2018-01-01'", b"2018-01-01"),  # a date, not a string
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


def test_absent_and_empty_companion_tests_have_one_content_digest():
    absent = review_content_sha256(None, rulespec=False)

    assert absent is not None
    for empty in (b"", b"[]\n", b"{}\n", b"# nothing\n"):
        assert review_content_sha256(empty, rulespec=False) == absent
    assert review_content_sha256(TEST_BYTES, rulespec=False) != absent


@pytest.mark.parametrize(
    "raw",
    [b"\xff\xfe", b"a: [unclosed\n", b"a: !!python/object:os.system x\n"],
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
        "reviewed": {**_digests(PAIR), "content_complete": True},
        "applied": _digests(PAIR),
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
        "generalist_review verdict is not bound to the applied file digests"
    ]


def test_failed_record_keeps_the_reviewer_score():
    record = _build(metrics_pass=False, score=3.0)

    assert record["status"] == "failed"
    assert record["reason"] is None
    assert record["verdict"] == {"passed": False, "score": 3.0, "issue_count": 2}
    assert record["applied"] == _digests(PAIR)
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
    assert record["reviewed"] == {
        **dict.fromkeys(DIGEST_FIELDS),
        "content_complete": None,
    }
    assert record["applied"] == dict.fromkeys(DIGEST_FIELDS)
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
        {"evidence": _evidence(reviewed_rulespec_content_sha256=None)},
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
    assert record["applied"] == dict.fromkeys(DIGEST_FIELDS)
    assert _issues(record) == []


def test_verdict_never_binds_to_a_pair_without_valid_byte_digests():
    for field, value in (("rulespec_sha256", "zz"), ("test_sha256", "zz")):
        malformed = AppliedPair(**{**_digests(PAIR), field: value})
        record = _build(applied=[malformed])
        assert record["status"] == "unavailable"
        assert record["applied"] == dict.fromkeys(DIGEST_FIELDS)


def test_verdict_binds_to_the_matching_file_among_several():
    other = applied_pair(REPAIRED_RULE_BYTES, None)

    record = _build(applied=[other, STAMPED_PAIR])

    assert record["status"] == "passed"
    assert record["applied"] == _digests(STAMPED_PAIR)


def test_review_without_a_companion_test_covers_an_absent_or_empty_one():
    evidence = _evidence(test=None)
    assert evidence["reviewed_test_sha256"] is None

    for applied in (applied_pair(RULE_BYTES, None), applied_pair(RULE_BYTES, b"[]\n")):
        record = _build(applied=[applied], evidence=evidence)
        assert record["status"] == "passed"
        assert record["applied"]["test_sha256"] == applied.test_sha256
        assert (
            _issues(
                record,
                [applied.rulespec_sha256]
                + ([applied.test_sha256] if applied.test_sha256 else []),
            )
            == []
        )
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
        (("reviewed", "rulespec_content_sha256"), OTHER_SHA),
        (("reviewed", "rulespec_content_sha256"), None),
        (("reviewed", "test_sha256"), "zz"),
        (("reviewed", "test_content_sha256"), OTHER_SHA),
        (("reviewed", "content_complete"), "yes"),
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
    forged["applied"] = _digests(repaired)
    assert _issues(forged, [repaired.rulespec_sha256, repaired.test_sha256]) == [
        "generalist_review verdict is on content that differs from the applied content"
    ]


def test_reader_refuses_a_pass_whose_digests_are_not_in_the_manifest():
    passed = _build()

    assert _issues(passed, [OTHER_SHA]) == [
        "generalist_review verdict is not bound to the applied file digests"
    ]
    assert _issues(passed, [PAIR.rulespec_sha256]) == [
        "generalist_review verdict is not bound to the applied file digests"
    ]
    assert _issues(passed, [PAIR.test_sha256]) == [
        "generalist_review verdict is not bound to the applied file digests"
    ]


@pytest.mark.parametrize(
    "field",
    ["rulespec_sha256", "rulespec_content_sha256", "test_content_sha256"],
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


@pytest.mark.parametrize(
    ("flag", "environ", "expected"),
    [
        (False, {}, False),
        (True, {}, True),
        (False, {REQUIRE_REVIEW_ENV: ""}, False),
        (False, {REQUIRE_REVIEW_ENV: "0"}, False),
        (False, {REQUIRE_REVIEW_ENV: "1"}, True),
        (True, {REQUIRE_REVIEW_ENV: "0"}, True),
        (True, {REQUIRE_REVIEW_ENV: "1"}, True),
        ("yes", {}, False),
        (None, {}, False),
    ],
)
def test_require_review_setting(flag, environ, expected):
    assert require_review_setting(flag, environ) is expected


@pytest.mark.parametrize("raw", ["true", "yes", "on", " 1", "2", "false"])
def test_unreadable_require_review_variable_refuses(raw):
    with pytest.raises(ValueError, match=REQUIRE_REVIEW_ENV):
        require_review_setting(False, {REQUIRE_REVIEW_ENV: raw})


def test_requirement_refuses_every_status_but_passed():
    assert review_requirement_refusal(_build()) is None
    for record in (
        _build(metrics_pass=False, score=2.0),
        _build(metrics_pass=False, score=None, evidence=None),
        _build(skipped_by_flag=True),
        _build(metrics_pass=True, score=None, prompt_sha256=None, evidence=None),
        _build(applied=[]),
        _build(applied=[applied_pair(REPAIRED_RULE_BYTES, TEST_BYTES)]),
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
        cli._review_applied_pairs([rule]),
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
        "content_complete": True,
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
    # Apply writes an empty companion test for a deferred module.
    (tmp_path / "rule.test.yaml").write_text("[]\n")
    assert _record_for(result, rule)["status"] == "passed"
    (tmp_path / "rule.test.yaml").write_bytes(TEST_BYTES)
    assert _record_for(result, rule)["status"] == "unavailable"


def test_applied_pairs_pair_each_rulespec_with_its_companion_test(tmp_path):
    rule = _candidate(tmp_path, rule_text=RULE_BYTES.decode())
    test = tmp_path / "rule.test.yaml"
    test.write_bytes(TEST_BYTES)
    lone = tmp_path / "lone.yaml"
    lone.write_bytes(REPAIRED_RULE_BYTES)

    assert cli._review_applied_pairs([rule, test, lone, tmp_path / "gone.yaml"]) == [
        PAIR,
        applied_pair(REPAIRED_RULE_BYTES, None),
    ]
    # A companion test is never a RuleSpec of its own.
    assert cli._review_applied_pairs([test]) == []


def test_review_of_a_truncated_candidate_says_so(tmp_path, monkeypatch):
    rule = _candidate(tmp_path, rule_text="format: rulespec/v1\n" + "# x\n" * 2000)
    monkeypatch.setattr(
        validator_pipeline,
        "run_claude_code",
        lambda *_a, **_k: ('{"score": 9.0, "passed": true}', 0),
    )

    result = _pipeline(tmp_path)._run_reviewer("generalist-reviewer", rule)

    assert result.details["review_evidence"]["content_complete"] is False


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
        state, cli._review_applied_pairs([rulespec_file])
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
    admits_record=None,
    program_bytes=b"program: us-sc/snap\nperiod: 2026-01\n",
    test_bytes=b"[]\n",
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
            return_value=per_file if admits_record is None else admits_record,
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
        _bound_state()(applied), cli._review_applied_pairs(applied)
    )
    if status != "passed":
        assert forged[GENERALIST_REVIEW_FIELD]["status"] == "passed"
        # A well-formed "passed" record, so only the signature can refuse it.
        assert (
            generalist_review_record_issues(
                forged[GENERALIST_REVIEW_FIELD],
                [item["sha256"] for item in payload["applied_files"]],
            )
            == []
        )
        assert (
            cli._applied_encoding_manifest_signature_issue(
                forged, TEST_APPLY_SIGNING_BROKER
            )
            is not None
        )


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
    assert record["applied"] == dict.fromkeys(DIGEST_FIELDS)


def test_manifest_for_a_pinned_verifier_target_keeps_todays_exact_shape(tmp_path):
    payload, _applied = _write_manifest(
        tmp_path, per_file=False, review_state=_bound_state()
    )

    assert GENERALIST_REVIEW_FIELD not in payload
    assert frozenset(payload) == cli._MODEL_APPLY_MANIFEST_FIELDS


def test_manifest_for_a_held_main_verifier_keeps_the_per_file_shape(tmp_path):
    payload, _applied = _write_manifest(
        tmp_path, per_file=True, admits_record=False, review_state=_bound_state()
    )

    assert GENERALIST_REVIEW_FIELD not in payload
    assert frozenset(payload) == cli._MODEL_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS


def test_manifest_never_carries_the_record_on_the_set_digest_shape(tmp_path):
    # Even if the record gate said yes, no admitted shape pairs the record
    # with the repository waiver-set digest.
    payload, _applied = _write_manifest(
        tmp_path, per_file=False, admits_record=True, review_state=_bound_state()
    )

    assert GENERALIST_REVIEW_FIELD not in payload


@pytest.mark.parametrize("admits_record", [False, True])
def test_required_review_is_enforced_whether_or_not_the_record_is_written(
    tmp_path, admits_record
):
    with pytest.raises(RuntimeError, match="passed generalist review is required"):
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
    with pytest.raises(RuntimeError, match="passed generalist review is required"):
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


def test_reader_variants_only_gained_the_record_shapes():
    variants = cli._APPLIED_MANIFEST_SCHEMA_VARIANTS
    assert variants["model"] == (
        cli._MODEL_APPLY_MANIFEST_FIELDS,
        cli._MODEL_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS,
        cli._MODEL_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS | {GENERALIST_REVIEW_FIELD},
    )
    assert variants["codex model"] == (
        cli._MODEL_APPLY_MANIFEST_FIELDS,
        cli._MODEL_CODEX_APPLY_MANIFEST_FIELDS,
        cli._MODEL_CODEX_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS,
        cli._MODEL_CODEX_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS
        | {GENERALIST_REVIEW_FIELD},
    )
    # No other contract admits the field.
    for contract, shapes in variants.items():
        if contract not in {"model", "codex model"}:
            assert all(GENERALIST_REVIEW_FIELD not in shape for shape in shapes)


def test_reader_refuses_the_record_on_the_set_digest_shape(tmp_path):
    set_digest, _ = _write_manifest(tmp_path, per_file=False, review_state=None)
    set_digest[GENERALIST_REVIEW_FIELD] = _build()

    assert any(
        "does not match the exact model v5 schema" in issue
        for issue in _schema_issues(set_digest)
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "failed"),
        (("verdict", "score"), 2.0),
        (("reviewed", "rulespec_content_sha256"), OTHER_SHA),
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


def test_reader_refuses_a_pass_bound_to_a_file_the_manifest_does_not_apply(tmp_path):
    payload, _ = _write_manifest(tmp_path, per_file=True, review_state=_bound_state())
    baseline = _schema_issues(payload)
    # Swap the applied digest: the record now covers bytes that are not applied.
    payload["applied_files"][0]["sha256"] = OTHER_SHA

    added = [issue for issue in _schema_issues(payload) if issue not in baseline]
    assert any("not bound to the applied file digests" in issue for issue in added)


def test_unreceipted_retirement_scan_knows_the_record_shapes():
    classes = cli._UNRECEIPTED_RETIREMENT_MANIFEST_FIELD_CLASSES
    for old, new in (
        (
            cli._MODEL_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS,
            cli._MODEL_APPLY_REVIEW_RECORD_MANIFEST_FIELDS,
        ),
        (
            cli._MODEL_CODEX_APPLY_PER_FILE_WAIVER_MANIFEST_FIELDS,
            cli._MODEL_CODEX_APPLY_REVIEW_RECORD_MANIFEST_FIELDS,
        ),
    ):
        assert classes[new][0] == classes[old][0]
        assert classes[new][1] == classes[old][1] | {GENERALIST_REVIEW_FIELD}


# --- encode settings (the --apply runs are in tests/test_cli.py) --------


class TestEncodeRequireReviewSettings:
    def _args(self, **values):
        return SimpleNamespace(
            backend="codex",
            apply=values.get("apply", True),
            require_review=values.get("require_review", False),
            skip_reviewers=values.get("skip_reviewers", False),
        )

    def test_flag_without_apply_is_refused_before_any_work(self, monkeypatch):
        monkeypatch.delenv(REQUIRE_REVIEW_ENV, raising=False)
        with (
            patch("axiom_encode.cli._local_verification_encode_roots") as later,
            pytest.raises(ValueError, match="--require-review requires --apply"),
        ):
            cmd_encode(self._args(apply=False, require_review=True))
        later.assert_not_called()

    @pytest.mark.parametrize("by_env", [False, True])
    def test_require_and_skip_together_are_refused(self, monkeypatch, by_env):
        if by_env:
            monkeypatch.setenv(REQUIRE_REVIEW_ENV, "1")
        else:
            monkeypatch.delenv(REQUIRE_REVIEW_ENV, raising=False)
        with (
            patch("axiom_encode.cli._local_verification_encode_roots") as later,
            pytest.raises(ValueError, match="cannot both require"),
        ):
            cmd_encode(self._args(require_review=not by_env, skip_reviewers=True))
        later.assert_not_called()

    def test_unreadable_variable_is_refused(self, monkeypatch):
        monkeypatch.setenv(REQUIRE_REVIEW_ENV, "true")
        with (
            patch("axiom_encode.cli._local_verification_encode_roots") as later,
            pytest.raises(ValueError, match=REQUIRE_REVIEW_ENV),
        ):
            cmd_encode(self._args())
        later.assert_not_called()

    def test_variable_without_apply_requires_nothing(self, monkeypatch):
        monkeypatch.setenv(REQUIRE_REVIEW_ENV, "1")
        assert cli._encode_requires_review(self._args(apply=False)) is False
        # ...and does not collide with --skip-reviewers on a run that applies
        # nothing.
        assert (
            cli._encode_requires_review(self._args(apply=False, skip_reviewers=True))
            is False
        )

    def test_default_is_off(self, monkeypatch):
        monkeypatch.delenv(REQUIRE_REVIEW_ENV, raising=False)
        assert cli._encode_requires_review(self._args()) is False
        assert cli._encode_requires_review(self._args(require_review=True)) is True


def test_module_states_only_four_statuses():
    assert record_module.STATUSES == {"passed", "failed", "unavailable", "skipped"}
