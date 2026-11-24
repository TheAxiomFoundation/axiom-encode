"""`promote-reviewed-candidate --require-review`: the switch of the signed lane.

Off by default: no reviewer runs and the promotion is as it was. On: the
generalist reviewer reads the exact candidate bytes after validation, and the
promotion signs only when the review record says `passed`.

No test calls a model: the reviewer CLI is replaced by scripted text.
"""

from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode import cli
from axiom_encode.cli import (
    APPLIED_ENCODING_OFFICIAL_REPOSITORY,
    _applied_encoding_manifest_path,
    cmd_promote_reviewed_candidate,
    main,
)
from axiom_encode.generalist_review_chunks import (
    CHUNK_SOURCE_CHARS,
    MAX_CHUNKS_PER_FILE,
)
from axiom_encode.generalist_review_record import (
    GENERALIST_REVIEW_SCHEMA,
    generalist_review_record_issues,
)
from axiom_encode.harness import validator_pipeline
from tests.test_cli import (
    _APPLY_VALIDATION_SNAPSHOT_ATTR,
    AXIOM_ENCODE_TEST_VERSION,
    TEST_APPLY_SIGNING_BROKER,
    _reviewed_candidate_binding,
    _reviewed_candidate_checkout,
    _write_reviewed_rulespec_records,
)

PASS = '{"score": 9.0, "passed": true, "blocking_issues": []}'
RULE_PATH = "us/statutes/7/2015/f.yaml"
TEST_PATH = "us/statutes/7/2015/f.test.yaml"
REVIEW_LINE = "reviewed candidate generalist review: "


class _Promotion:
    """One promotion of the synthetic reviewed candidate, reviewer scripted."""

    def __init__(self, tmp_path, *, source_body="authoritative source text\n"):
        self.repo, self.rule, self.companion, self.ref = _reviewed_candidate_checkout(
            tmp_path
        )
        self.record_root = _write_reviewed_rulespec_records(
            tmp_path / "reviewed-rulespec-refs",
            ("us", self.ref, _reviewed_candidate_binding(self.rule, self.companion)),
        )
        self.args = SimpleNamespace(
            repo=self.repo,
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            corpus_path=tmp_path / "axiom-corpus",
            rulespec_dependency_root=[],
            rulespec_path=RULE_PATH,
            citation="us/statute/7/2015/f",
            reviewed_rulespec_ref=self.ref,
            run_id="promotion-review-test",
        )
        self.args.axiom_rules_path.mkdir()
        self.args.corpus_path.mkdir()
        self.source_unit = SimpleNamespace(
            body=source_body,
            amendment_documents=(),
            source_attestation={
                "requested_corpus_citation_path": "us/statute/7/2015/f"
            },
        )
        self.manifest = self.repo / _applied_encoding_manifest_path(Path(RULE_PATH))
        self.prompts: list[str] = []
        self.signed: list[dict] = []
        self.order: list[str] = []

    def run(self, reviewer_output=PASS):
        def reviewer(prompt, **_kwargs):
            self.prompts.append(prompt)
            self.order.append("review")
            if isinstance(reviewer_output, tuple):
                return reviewer_output
            return reviewer_output, 0

        def validate(result, **_kwargs):
            self.order.append("validate")
            setattr(
                result,
                _APPLY_VALIDATION_SNAPSHOT_ATTR,
                {
                    "manifest_validation_execution": {
                        "schema": "axiom-encode/apply-validation-execution/v2"
                    }
                },
            )
            return True, [], {}

        real_sign = cli._sign_applied_encoding_manifest

        def sign(payload, broker):
            self.order.append("sign")
            self.signed.append(dict(payload))
            return real_sign(payload, broker)

        identity = {
            "repository": APPLIED_ENCODING_OFFICIAL_REPOSITORY,
            "commit": "b" * 40,
            "version": AXIOM_ENCODE_TEST_VERSION,
            "identity_source": "trusted-runtime-attestation",
        }
        patches = {
            "axiom_encode.cli._promotion_reviewed_rulespec_record_root": {
                "return_value": self.record_root
            },
            "axiom_encode.cli._recover_apply_transaction": {},
            "axiom_encode.cli.load_rulespec_local_corpus_release": {
                "return_value": object()
            },
            "axiom_encode.cli._manifest_primary_source_verifications": {
                "return_value": ([], "rulespec-us/us")
            },
            "axiom_encode.cli.resolve_corpus_source_unit": {
                "return_value": self.source_unit
            },
            "axiom_encode.cli._run_generated_encoding_overlay_validation": {
                "side_effect": validate
            },
            "axiom_encode.cli._require_clean_axiom_encode_git_provenance": {
                "return_value": {
                    "root": "/opt/axiom-verification",
                    "commit": "b" * 40,
                    "dirty_tracked": False,
                    "version": AXIOM_ENCODE_TEST_VERSION,
                    "version_commit": "b" * 40,
                    "identity_source": "trusted-runtime-attestation",
                }
            },
            "axiom_encode.cli.verify_rulespec_validation_waiver_set": {
                "return_value": "c" * 64
            },
            "axiom_encode.cli._target_uses_per_file_waiver_bindings": {
                "return_value": False
            },
            "axiom_encode.cli._current_guard_encoder_execution_identity": {
                "return_value": identity
            },
            "axiom_encode.cli._load_verified_applied_encoding_manifest_payload": {
                "return_value": ({}, "", "d" * 64, [])
            },
            "axiom_encode.cli._sign_applied_encoding_manifest": {"side_effect": sign},
            "axiom_encode.harness.validator_pipeline.run_claude_code": {
                "side_effect": reviewer
            },
        }
        with ExitStack() as stack:
            for target, kwargs in patches.items():
                stack.enter_context(patch(target, **kwargs))
            signer = stack.enter_context(
                patch("axiom_encode.cli._isolated_apply_manifest_signer")
            )
            signer.return_value.__enter__.return_value = TEST_APPLY_SIGNING_BROKER
            cmd_promote_reviewed_candidate(self.args)


def _review_record(output: str) -> dict:
    lines = [line for line in output.splitlines() if line.startswith(REVIEW_LINE)]
    assert len(lines) == 1
    return json.loads(lines[0][len(REVIEW_LINE) :])


def _stable(payload: dict) -> dict:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"generated_at", "signature", "reviewed_rulespec_ref"}
    }


def test_switch_off_runs_no_reviewer_and_signs_what_it_signed_before(tmp_path, capsys):
    off = _Promotion(tmp_path / "off")
    off.run()
    off_output = capsys.readouterr().out
    explicit = _Promotion(tmp_path / "explicit")
    explicit.args.require_review = False
    explicit.run()
    explicit_output = capsys.readouterr().out
    on = _Promotion(tmp_path / "on")
    on.args.require_review = True
    on.run()
    on_output = capsys.readouterr().out

    # Without the flag: no reviewer call, no review line, one signature.
    for promotion, output in ((off, off_output), (explicit, explicit_output)):
        assert promotion.prompts == []
        assert promotion.order == ["validate", "sign"]
        assert output == (
            "promoted reviewed candidate "
            ".axiom/encoding-manifests/us/statutes/7/2015/f.json\n"
        )
    # The flag adds nothing to a manifest: the signed payloads are equal
    # field for field (the commit of each synthetic checkout aside).
    assert _stable(off.signed[0]) == _stable(explicit.signed[0])
    assert _stable(off.signed[0]) == _stable(on.signed[0])
    assert set(off.signed[0]) == set(on.signed[0])
    assert "generalist_review" not in on.signed[0]
    assert on.order == ["validate", "review", "sign"]
    assert REVIEW_LINE in on_output


@pytest.mark.parametrize("truthy", ["yes", 1, "true", None, 0])
def test_only_the_boolean_flag_turns_the_review_on(tmp_path, capsys, truthy):
    promotion = _Promotion(tmp_path)
    promotion.args.require_review = truthy

    promotion.run()

    assert promotion.prompts == []
    assert REVIEW_LINE not in capsys.readouterr().out


def test_switch_on_reviews_the_exact_candidate_bytes_before_signing(tmp_path, capsys):
    promotion = _Promotion(tmp_path)
    promotion.args.require_review = True

    promotion.run()

    assert promotion.order == ["validate", "review", "sign"]
    assert len(promotion.prompts) == 1
    prompt = promotion.prompts[0]
    assert promotion.rule.read_text() in prompt
    assert "\n## Companion Test File\n[]\n" in prompt
    assert "authoritative source text" in prompt
    record = _review_record(capsys.readouterr().out)
    assert record["schema"] == GENERALIST_REVIEW_SCHEMA
    assert (record["status"], record["required"]) == ("passed", True)
    assert record["applied"]["rulespec_path"] == RULE_PATH
    assert [
        (entry["path"], entry["reviewed"], entry["complete"])
        for entry in record["files"]
    ] == [
        (TEST_PATH, True, True),
        (RULE_PATH, True, True),
    ]
    signed_files = [
        (entry["path"], entry["sha256"])
        for entry in promotion.signed[0]["applied_files"]
    ]
    # The record names exactly the files the manifest signs.
    assert generalist_review_record_issues(record, signed_files) == []
    assert promotion.manifest.exists()


@pytest.mark.parametrize(
    ("reviewer_output", "source_body", "status", "reason"),
    [
        (
            '{"score": 3.0, "passed": false, "blocking_issues": ["a defect"]}',
            None,
            "failed",
            None,
        ),
        (
            ("reviewer_unavailable: no reviewer CLI could start", 1),
            None,
            "unavailable",
            "reviewer_error",
        ),
        ("no verdict here", None, "unavailable", "reviewer_error"),
        ('{"score": 9.0}', None, "unavailable", "unreadable_review_result"),
        (
            '{"score": 9.0, "passed": true, "Passed": false}',
            None,
            "unavailable",
            "reviewer_error",
        ),
        (
            PASS,
            "above-the-caps",
            "partial",
            "review_saw_truncated_source",
        ),
    ],
)
def test_switch_on_refuses_before_signing_unless_the_review_passed(
    tmp_path, capsys, reviewer_output, source_body, status, reason
):
    promotion = (
        _Promotion(tmp_path)
        if source_body is None
        else _Promotion(
            tmp_path,
            source_body="word\n" * (MAX_CHUNKS_PER_FILE * CHUNK_SOURCE_CHARS),
        )
    )
    promotion.args.require_review = True

    with pytest.raises(RuntimeError) as refused:
        promotion.run(reviewer_output)

    detail = f"status {status}" + (f" ({reason})" if reason else "")
    assert str(refused.value) == (
        "reviewed candidate promotion requires a passed generalist review of "
        f"every promoted file (--require-review) and has none: {detail}. "
        "Nothing was signed."
    )
    # Validation ran, the reviewer ran, and nothing was signed or installed.
    assert promotion.order == ["validate", "review"]
    assert promotion.signed == []
    assert not promotion.manifest.exists()
    record = _review_record(capsys.readouterr().out)
    assert (record["status"], record["reason"]) == (status, reason)
    assert "a defect" not in json.dumps(record)


def test_a_candidate_larger_than_one_call_is_promoted_after_a_chunked_review(
    tmp_path, capsys
):
    source = "\n\n".join(
        f"({index}) A source paragraph of the synthetic section."
        for index in range(200)
    )
    promotion = _Promotion(tmp_path, source_body=source + "\n")
    promotion.args.require_review = True

    promotion.run()

    assert len(promotion.prompts) == 1
    assert "This call is the whole review." in promotion.prompts[0]
    assert source in promotion.prompts[0]
    record = _review_record(capsys.readouterr().out)
    assert record["status"] == "passed"
    assert record["coverage"]["source"]["size"] == len(source.encode("utf-8"))
    assert promotion.order == ["validate", "review", "sign"]


def test_the_review_reads_the_staged_copy_whose_bytes_are_the_signed_bytes(tmp_path):
    promotion = _Promotion(tmp_path)
    promotion.args.require_review = True
    seen = {}
    real = validator_pipeline.run_generalist_review

    def spy(rulespec_file, **kwargs):
        seen["rulespec"] = Path(rulespec_file).read_bytes()
        seen["companion"] = Path(rulespec_file).with_name("f.test.yaml").read_bytes()
        seen["cwd"] = kwargs["cwd"]
        seen["path"] = Path(rulespec_file).resolve()
        return real(rulespec_file, **kwargs)

    with patch.object(validator_pipeline, "run_generalist_review", spy):
        promotion.run()

    assert seen["rulespec"] == promotion.rule.read_bytes()
    assert seen["companion"] == promotion.companion.read_bytes()
    assert seen["cwd"] == promotion.repo / "us"
    # The reviewer reads the staged copy that validation checked, not the
    # checkout file: a change of the checkout after validation cannot reach it.
    assert promotion.repo.resolve() not in seen["path"].parents
    assert seen["path"].parts[-5:] == (
        "reviewed-candidate",
        "statutes",
        "7",
        "2015",
        "f.yaml",
    )


def _parsed_promotion_args(extra):
    argv = [
        "axiom-encode",
        "promote-reviewed-candidate",
        "--repo",
        "rulespec-us",
        "--rulespec-path",
        RULE_PATH,
        "--citation",
        "us/statute/7/2015/f",
        "--reviewed-rulespec-ref",
        "a" * 40,
        "--axiom-rules-engine-path",
        "engine",
        "--corpus-path",
        "corpus",
        "--run-id",
        "1",
        *extra,
    ]
    with (
        patch.object(sys, "argv", argv),
        patch("axiom_encode.cli.cmd_promote_reviewed_candidate") as command,
    ):
        main()
    return command.call_args.args[0]


def test_the_command_line_flag_is_off_unless_given():
    assert _parsed_promotion_args([]).require_review is False
    assert cli._promotion_requires_review(_parsed_promotion_args([])) is False
    assert _parsed_promotion_args(["--require-review"]).require_review is True
    assert (
        cli._promotion_requires_review(_parsed_promotion_args(["--require-review"]))
        is True
    )


def test_no_environment_variable_turns_the_review_on(tmp_path, capsys, monkeypatch):
    for name in (
        "AXIOM_ENCODE_REQUIRE_REVIEW",
        "AXIOM_ENCODE_PROMOTION_REQUIRE_REVIEW",
        "REQUIRE_REVIEW",
    ):
        monkeypatch.setenv(name, "1")
    promotion = _Promotion(tmp_path)

    promotion.run()

    assert promotion.prompts == []
    assert REVIEW_LINE not in capsys.readouterr().out


# --- the retained review record ---------------------------------------------


def _record_file_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_review_record_file_keeps_a_passed_review_with_the_run_attempt(
    tmp_path, capsys
):
    promotion = _Promotion(tmp_path / "work")
    record_file = tmp_path / "promotion-generalist-review.json"
    promotion.args.require_review = True
    promotion.args.review_record_file = record_file
    promotion.args.run_attempt = "2"

    promotion.run()

    printed = _review_record(capsys.readouterr().out)
    payload = _record_file_payload(record_file)
    assert payload == {
        "schema": "axiom-encode/promotion-generalist-review/v1",
        "tool": "axiom-encode promote-reviewed-candidate",
        "axiom_encode_version": cli.__version__,
        "run_id": "promotion-review-test",
        "run_attempt": 2,
        "citation": "us/statute/7/2015/f",
        "rulespec_path": RULE_PATH,
        "reviewed_rulespec_ref": promotion.ref,
        "review_passed": True,
        "generalist_review": printed,
    }
    # The signed manifest is unchanged by the file: no record in it.
    assert "generalist_review" not in promotion.signed[0]
    assert "generalist_review" not in json.loads(promotion.manifest.read_text())
    assert promotion.order == ["validate", "review", "sign"]


def test_review_record_file_keeps_a_refused_review_too(tmp_path, capsys):
    promotion = _Promotion(tmp_path / "work")
    record_file = tmp_path / "promotion-generalist-review.json"
    promotion.args.require_review = True
    promotion.args.review_record_file = record_file

    with pytest.raises(RuntimeError, match="Nothing was signed"):
        promotion.run(
            '{"score": 3.0, "passed": false, "blocking_issues": ["a defect"]}'
        )

    payload = _record_file_payload(record_file)
    assert payload["review_passed"] is False
    assert payload["run_attempt"] is None
    assert payload["generalist_review"]["status"] == "failed"
    assert payload["generalist_review"] == _review_record(capsys.readouterr().out)
    assert "a defect" not in record_file.read_text()
    assert promotion.signed == []


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"review_record_file": "FILE"}, "requires --require-review"),
        (
            {"require_review": True, "run_attempt": "1"},
            "--run-attempt requires --review-record-file",
        ),
        (
            {"require_review": True, "review_record_file": "FILE", "run_attempt": "0"},
            "positive decimal number",
        ),
        (
            {"require_review": True, "review_record_file": "FILE", "run_attempt": "1a"},
            "positive decimal number",
        ),
        (
            {"require_review": True, "review_record_file": "FILE", "run_attempt": "-1"},
            "positive decimal number",
        ),
        (
            {"require_review": True, "review_record_file": "relative.json"},
            "absolute path",
        ),
        (
            {"require_review": True, "review_record_file": "INSIDE"},
            "outside the RuleSpec checkout",
        ),
        (
            {"require_review": True, "review_record_file": "EXISTS"},
            "already exists",
        ),
        (
            {"require_review": True, "review_record_file": "NODIR"},
            "directory does not exist",
        ),
    ],
)
def test_contradictory_record_settings_refuse_before_validation(
    tmp_path, settings, message
):
    promotion = _Promotion(tmp_path / "work")
    existing = tmp_path / "existing.json"
    existing.write_text("{}")
    paths = {
        "FILE": tmp_path / "record.json",
        "INSIDE": promotion.repo / "record.json",
        "EXISTS": existing,
        "NODIR": tmp_path / "missing" / "record.json",
        "relative.json": Path("relative.json"),
    }
    for key, value in settings.items():
        setattr(promotion.args, key, paths.get(value, value))

    with pytest.raises(ValueError, match=message):
        promotion.run()

    assert promotion.order == []
    assert promotion.prompts == [] and promotion.signed == []
    assert existing.read_text() == "{}"
    assert not (tmp_path / "record.json").exists()


def test_the_record_file_flags_parse_and_default_to_absent():
    plain = _parsed_promotion_args([])
    assert plain.review_record_file is None and plain.run_attempt is None
    given = _parsed_promotion_args(
        ["--require-review", "--review-record-file", "/x/r.json", "--run-attempt", "3"]
    )
    assert given.review_record_file == Path("/x/r.json")
    assert given.run_attempt == "3"
