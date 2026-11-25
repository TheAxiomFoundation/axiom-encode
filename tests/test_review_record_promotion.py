"""`promote-reviewed-candidate --require-review-record` and `review-record`.

Off by default: no record is read and the promotion is as it was. On: the
promotion signs only when the file that the signing workflow wrote from the
candidate pull request body holds, for every unit, one passed review record
bound to the exact bytes it signs. No reviewer runs in the promotion.

No test calls a model: the reviewer CLI is replaced by scripted text.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode import candidate_review_record as crr
from axiom_encode import cli
from axiom_encode import reviewed_pr_admission as admission
from axiom_encode.cli import main
from tests.test_candidate_review_record import CLEAN, FAIL, PASS, make_record
from tests.test_generalist_review_promotion import RULE_PATH, _Promotion
from tests.test_multi_unit_promotion_cli import (
    NAMES,
    Run,
    batch_admission,
    candidate,
    citation,
    primary,
)

ROOT = Path(__file__).resolve().parents[1]
CITATION = "us/statute/7/2015/f"
RECORD_LINE = "reviewed candidate review record: "


def write_document(tmp_path: Path, records, head: str, name="records.json") -> Path:
    body = "the body that the records were read from"
    target = tmp_path / name
    target.write_text(
        json.dumps(
            crr.build_read_document(
                rulespec_pr=2044,
                head_sha=head,
                body=body,
                records=records,
                run_id="123",
                run_attempt=1,
            )
        )
    )
    return target


def unit_record(tmp_path: Path, promotion: _Promotion, **kwargs) -> dict:
    return make_record(
        tmp_path,
        rule=promotion.rule.read_bytes(),
        companion=promotion.companion.read_bytes(),
        **kwargs,
    )


# --- Single unit ----------------------------------------------------------------


def test_switch_off_reads_no_record_and_signs_what_it_signed_before(tmp_path, capsys):
    off = _Promotion(tmp_path / "off")
    off.run()
    off_output = capsys.readouterr().out
    explicit = _Promotion(tmp_path / "explicit")
    explicit.args.require_review_record = None
    explicit.run()
    assert off.order == explicit.order == ["validate", "sign"]
    assert RECORD_LINE not in off_output
    assert off.prompts == explicit.prompts == []


def test_a_passed_record_bound_to_the_exact_bytes_is_signed(tmp_path, capsys):
    promotion = _Promotion(tmp_path / "p")
    record = unit_record(tmp_path, promotion)
    promotion.args.require_review_record = write_document(
        tmp_path, [record], promotion.ref
    )
    promotion.run()
    assert promotion.order == ["validate", "sign"]
    # No reviewer runs in the promotion: the record is read, not made.
    assert promotion.prompts == []
    assert promotion.manifest.is_file()
    (line,) = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(RECORD_LINE)
    ]
    announced = json.loads(line[len(RECORD_LINE) :])
    assert announced["citation"] == CITATION
    assert announced["status"] == "passed"
    assert announced["reviewed"] == record["reviewed"]
    assert announced["record_sha256"] == crr.record_sha256(record)
    # The signed manifest is the manifest of a promotion without the flag.
    plain = _Promotion(tmp_path / "plain")
    plain.run()
    stable = lambda payload: {  # noqa: E731
        key: value
        for key, value in payload.items()
        if key not in {"generated_at", "signature", "reviewed_rulespec_ref"}
    }
    assert stable(promotion.signed[0]) == stable(plain.signed[0])


def _refused(tmp_path, records, *, head=None, message):
    promotion = _Promotion(tmp_path / "p")
    built = records(promotion) if callable(records) else records
    promotion.args.require_review_record = write_document(
        tmp_path, built, head or promotion.ref
    )
    with pytest.raises(RuntimeError, match=message):
        promotion.run()
    # Refused before validation: nothing validated, nothing signed, no manifest.
    assert promotion.order == []
    assert promotion.signed == []
    assert not promotion.manifest.exists()


def test_no_record_is_refused(tmp_path):
    _refused(tmp_path, [], message="no review record for us/statute/7/2015/f")


def test_a_record_that_did_not_pass_is_refused(tmp_path):
    _refused(
        tmp_path,
        lambda p: [unit_record(tmp_path, p, output=FAIL)],
        message="the review status is failed, not passed",
    )


def test_a_module_changed_after_the_review_is_refused(tmp_path):
    _refused(
        tmp_path,
        lambda p: [
            make_record(
                tmp_path,
                rule=p.rule.read_bytes() + b"# reviewed another version\n",
                companion=p.companion.read_bytes(),
            )
        ],
        message="the module changed after the review",
    )


def test_a_companion_changed_after_the_review_is_refused(tmp_path):
    _refused(
        tmp_path,
        lambda p: [
            make_record(
                tmp_path, rule=p.rule.read_bytes(), companion=b"- name: other\n"
            )
        ],
        message="the companion test changed after the review",
    )


def test_a_record_of_another_citation_is_refused(tmp_path):
    _refused(
        tmp_path,
        lambda p: [unit_record(tmp_path, p, citation="us/statute/7/2015/g")],
        message="the record is for citation us/statute/7/2015/g",
    )


def test_a_record_of_another_path_is_refused(tmp_path):
    _refused(
        tmp_path,
        lambda p: [unit_record(tmp_path, p, rulespec_path="us/statutes/7/2015/g.yaml")],
        message="the record is for module us/statutes/7/2015/g.yaml",
    )


def test_two_records_for_the_unit_are_refused(tmp_path):
    _refused(
        tmp_path,
        lambda p: [
            unit_record(tmp_path, p, output=FAIL),
            unit_record(tmp_path, p, attempt=2),
        ],
        message="2 review records name us/statute/7/2015/f",
    )


def test_records_read_at_another_head_are_refused(tmp_path):
    _refused(
        tmp_path,
        lambda p: [unit_record(tmp_path, p)],
        head="d" * 40,
        message="read from another candidate head",
    )


@pytest.mark.parametrize(
    "content",
    ["", "not json", "[]", '{"schema": "axiom-encode/candidate-review-record/v1"}'],
)
def test_a_file_that_is_not_the_read_document_is_refused(tmp_path, content):
    promotion = _Promotion(tmp_path / "p")
    target = tmp_path / "records.json"
    target.write_text(content)
    promotion.args.require_review_record = target
    with pytest.raises(ValueError, match="review record"):
        promotion.run()
    assert promotion.order == [] and not promotion.manifest.exists()


def test_a_missing_or_relative_file_is_refused(tmp_path):
    for value, message in (
        (tmp_path / "absent.json", "cannot be read"),
        (Path("records.json"), "absolute path"),
    ):
        promotion = _Promotion(tmp_path / str(len(message)))
        promotion.args.require_review_record = value
        with pytest.raises(ValueError, match=message):
            promotion.run()
        assert promotion.order == []


def test_a_candidate_changed_between_the_two_checks_is_refused(tmp_path):
    """The check before the signatures reads the bytes that are signed."""

    promotion = _Promotion(tmp_path / "p")
    promotion.args.require_review_record = write_document(
        tmp_path, [unit_record(tmp_path, promotion)], promotion.ref
    )
    real = cli._require_promotion_review_records
    calls = []

    def second_time_another_document(document, units, **kwargs):
        calls.append(kwargs["announce"])
        if kwargs["announce"]:
            document = copy.deepcopy(document)
            document["records"] = []
        return real(document, units, **kwargs)

    with patch(
        "axiom_encode.cli._require_promotion_review_records",
        side_effect=second_time_another_document,
    ):
        with pytest.raises(RuntimeError, match="no review record"):
            promotion.run()
    assert calls == [False, True]
    assert promotion.order == ["validate"] and promotion.signed == []


def test_the_command_line_flag_is_off_unless_given():
    base = [
        "axiom-encode",
        "promote-reviewed-candidate",
        "--repo",
        "r",
        "--rulespec-path",
        RULE_PATH,
        "--citation",
        CITATION,
        "--reviewed-rulespec-ref",
        "c" * 40,
        "--axiom-rules-engine-path",
        "e",
        "--corpus-path",
        "c",
    ]
    seen = []
    with patch("axiom_encode.cli.cmd_promote_reviewed_candidate", seen.append):
        with patch.object(sys, "argv", base):
            main()
        with patch.object(sys, "argv", [*base, "--require-review-record", "/x/r.json"]):
            main()
    assert seen[0].require_review_record is None
    assert seen[1].require_review_record == Path("/x/r.json")


# --- Batch ----------------------------------------------------------------------


@pytest.fixture
def batch_run(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "MULTI_UNIT_PROMOTION_ENABLED", True)
    repo, head = candidate(tmp_path)
    return Run(tmp_path, repo, head, batch_admission(repo, head)), repo, head


def batch_record(tmp_path, repo, name, **kwargs):
    rule = repo / primary(name)
    return make_record(
        tmp_path,
        citation=citation(name),
        rulespec_path=primary(name),
        rule=rule.read_bytes(),
        companion=rule.with_name(f"{name}.test.yaml").read_bytes(),
        **kwargs,
    )


def test_a_batch_with_a_record_for_every_unit_is_signed(
    batch_run, tmp_path, monkeypatch
):
    run, repo, head = batch_run
    run.args.require_review_record = write_document(
        tmp_path, [batch_record(tmp_path, repo, name) for name in NAMES], head
    )
    run(monkeypatch)
    assert run.signed == list(NAMES)


@pytest.mark.parametrize("missing", NAMES)
def test_a_batch_with_one_unit_without_a_record_signs_nothing(
    batch_run, tmp_path, monkeypatch, missing
):
    run, repo, head = batch_run
    run.args.require_review_record = write_document(
        tmp_path,
        [batch_record(tmp_path, repo, name) for name in NAMES if name != missing],
        head,
    )
    with pytest.raises(RuntimeError) as refusal:
        run(monkeypatch)
    assert f"no review record for {citation(missing)}" in str(refusal.value)
    assert run.validated == [] and run.signed == [] and run.installs == []
    run.assert_nothing_written()


@pytest.mark.parametrize("failing", NAMES)
def test_a_batch_with_one_failed_record_signs_nothing(
    batch_run, tmp_path, monkeypatch, failing
):
    run, repo, head = batch_run
    run.args.require_review_record = write_document(
        tmp_path,
        [
            batch_record(tmp_path, repo, name, output=FAIL if name == failing else PASS)
            for name in NAMES
        ],
        head,
    )
    with pytest.raises(RuntimeError) as refusal:
        run(monkeypatch)
    assert (
        f"{citation(failing)} ({primary(failing)}): the review status is failed"
        in str(refusal.value)
    )
    assert run.signed == []
    run.assert_nothing_written()


def test_a_record_of_one_unit_does_not_admit_another(batch_run, tmp_path, monkeypatch):
    run, repo, head = batch_run
    records = [batch_record(tmp_path, repo, name) for name in NAMES]
    # The record of unit a, relabelled as the record of unit c.
    forged = copy.deepcopy(records[0])
    forged["citation"] = citation("c")
    forged["rulespec_path"] = primary("c")
    run.args.require_review_record = write_document(
        tmp_path, [records[0], records[1], forged], head
    )
    with pytest.raises(RuntimeError, match="us/statute/7/2015/c"):
        run(monkeypatch)
    assert run.signed == []


# --- review-record create / render / verify --------------------------------------


def run_cli(*arguments) -> int:
    with patch.object(sys, "argv", ["axiom-encode", "review-record", *arguments]):
        with pytest.raises(SystemExit) as exit_info:
            main()
    return exit_info.value.code


def create(tmp_path, repo, out, *, output=CLEAN, extra=()):
    source = SimpleNamespace(body="authoritative source text\r\n")
    (tmp_path / "corpus").mkdir(exist_ok=True)
    with (
        patch("axiom_encode.cli.load_rulespec_local_corpus_release"),
        patch("axiom_encode.cli.resolve_corpus_source_unit", return_value=source),
        patch(
            "axiom_encode.harness.validator_pipeline.run_claude_code",
            side_effect=lambda prompt, **_kwargs: (output, 0),
        ),
    ):
        return run_cli(
            "create",
            "--repo",
            str(repo),
            "--citation",
            CITATION,
            "--corpus-path",
            str(tmp_path / "corpus"),
            "--out",
            str(out),
            "--run-id",
            "local:run-9",
            *extra,
        )


def test_create_render_and_verify_round_trip(tmp_path, capsys):
    promotion = _Promotion(tmp_path / "p")
    out = tmp_path / "record.json"
    assert create(tmp_path, promotion.repo, out, extra=("--attempt", "2")) == 0
    printed = capsys.readouterr().out
    assert "status=passed" in printed and f"record_file={out}" in printed
    record = json.loads(out.read_text())
    assert crr.record_issues(record) == []
    assert record["run"] == {"id": "local:run-9", "attempt": 2}
    assert record["reviewed"]["commit"] == promotion.ref
    assert record["encoder"]["version"] == cli.__version__
    # The checkout is not changed by the review.
    assert (
        subprocess.check_output(
            ["git", "-C", str(promotion.repo), "status", "--porcelain"]
        )
        == b""
    )

    body = tmp_path / "body.md"
    assert run_cli("render", "--record-file", str(out), "--out", str(body)) == 0
    assert crr.extract_records("Candidate.\n\n" + body.read_text()) == [record]
    arguments = ("--repo", str(promotion.repo), "--citation", CITATION)
    assert run_cli("verify", *arguments, "--body-file", str(body)) == 0
    assert run_cli("verify", *arguments, "--record-file", str(out)) == 0
    assert "status=admitted" in capsys.readouterr().out

    # The promotion admits the same record.
    promotion.args.require_review_record = write_document(
        tmp_path, [record], promotion.ref
    )
    promotion.run()
    assert promotion.order == ["validate", "sign"]

    # A commit that changes the module after the review is refused.
    promotion.manifest.unlink()
    promotion.rule.write_text(promotion.rule.read_text() + "# later edit\n")
    subprocess.run(
        ["git", "-C", str(promotion.repo), "commit", "-am", "edit"],
        check=True,
        capture_output=True,
    )
    assert run_cli("verify", *arguments, "--body-file", str(body)) == 1
    assert "the module changed after the review" in capsys.readouterr().out


def test_create_keeps_a_failed_review_and_exits_1(tmp_path, capsys):
    promotion = _Promotion(tmp_path / "p")
    out = tmp_path / "record.json"
    assert create(tmp_path, promotion.repo, out, output=FAIL) == 1
    record = json.loads(out.read_text())
    assert record["status"] == "failed"
    assert record["findings"][0]["severity"] == "blocking"
    assert "status=failed" in capsys.readouterr().out
    assert (
        run_cli(
            "verify",
            "--repo",
            str(promotion.repo),
            "--citation",
            CITATION,
            "--record-file",
            str(out),
        )
        == 1
    )


def test_create_refuses_an_uncommitted_candidate_and_an_existing_file(tmp_path, capsys):
    promotion = _Promotion(tmp_path / "p")
    out = tmp_path / "record.json"
    out.write_text("{}")
    assert create(tmp_path, promotion.repo, out) == 2
    assert "already exists" in capsys.readouterr().out
    out.unlink()
    promotion.rule.write_text(promotion.rule.read_text() + "# not committed\n")
    assert create(tmp_path, promotion.repo, out) == 2
    assert "differs from the commit at HEAD" in capsys.readouterr().out
    assert not out.exists()


# --- The workflow's reader script -------------------------------------------------


def load_script():
    spec = importlib.util.spec_from_file_location(
        "_read_promotion_review_records",
        ROOT / "scripts/read_promotion_review_records.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dispatch(head: str, *, required=True, units=None) -> dict:
    wrapper = {
        "schema": "axiom-encode/reviewed-pr-promotion/v1",
        "rulespec_pr": 2044,
        "transaction": {
            "schema": "axiom-encode/atomic-source-transaction/v5",
            "source_bundle": [],
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [],
            "require_complete_source_unit": True,
            "manifest_only_refresh": False,
            "reviewed_candidate_promotion": True,
        },
    }
    if required:
        wrapper["require_review_record"] = True
    if units is not None:
        wrapper["units"] = units
    return {
        "country": "us",
        "rulespec_ref": head,
        "citation": units[0] if units else CITATION,
        "source_bundle_json": json.dumps(wrapper),
    }


class Pull:
    def __init__(self, body, head, number=2044):
        self.payload = {"number": number, "head": {"sha": head}, "body": body}
        self.paths: list[str] = []

    def __call__(self, path):
        self.paths.append(path)
        return self.payload


def read(tmp_path, promotion, pull, **kwargs):
    return load_script().read_records(
        dispatch=kwargs.pop("dispatch_json", dispatch(promotion.ref)),
        repo=promotion.repo,
        out_dir=tmp_path / "out",
        run_id="123",
        run_attempt=3,
        fetch=pull,
    )


def test_the_reader_does_nothing_without_the_wrapper_key(tmp_path):
    promotion = _Promotion(tmp_path / "p")
    pull = Pull("", promotion.ref)
    for payload in (
        dispatch(promotion.ref, required=False),
        {"source_bundle_json": "[]"},
        {"source_bundle_json": "not json"},
    ):
        assert read(tmp_path, promotion, pull, dispatch_json=payload) == (False, "")
    assert pull.paths == [] and not (tmp_path / "out").exists()


def test_the_reader_reads_the_body_once_and_records_its_digest(tmp_path):
    promotion = _Promotion(tmp_path / "p")
    record = unit_record(tmp_path, promotion)
    body = "Candidate.\n\n" + crr.render_block(record)
    pull = Pull(body, promotion.ref)
    required, summary = read(tmp_path, promotion, pull)
    assert required is True
    assert pull.paths == ["/repos/TheAxiomFoundation/rulespec-us/pulls/2044"]
    out = tmp_path / "out"
    assert (out / "candidate-pull-request-body.md").read_text() == body
    document = crr.load_read_document(
        (out / "candidate-review-records.json").read_text()
    )
    import hashlib

    assert document["body_sha256"] == hashlib.sha256(body.encode()).hexdigest()
    assert document["head_sha"] == promotion.ref
    assert (document["signing_run_id"], document["signing_run_attempt"]) == ("123", 3)
    assert document["records"] == [record]
    assert (out / "candidate-review-records.md").read_text() == summary
    assert document["body_sha256"] in summary and "attempt 3" in summary
    # The promotion admits the document that the reader wrote.
    promotion.args.require_review_record = out / "candidate-review-records.json"
    promotion.run()
    assert promotion.order == ["validate", "sign"]


def test_the_reader_refuses_a_body_of_another_head(tmp_path):
    promotion = _Promotion(tmp_path / "p")
    body = crr.render_block(unit_record(tmp_path, promotion))
    with pytest.raises(ValueError, match="head is not the reviewed ref"):
        read(tmp_path, promotion, Pull(body, "d" * 40))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda b: "", "no review record"),
        (lambda b: "Reviewer verdict: passed 9/10", "no review record"),
        (lambda b: b.replace('"status":"passed"', '"status":"failed"'), "us/statute"),
        (lambda b: b.replace('{"citation"', '{"citation'), "not valid JSON"),
        (lambda b: b + "\n" + b, "2 review records name"),
        (lambda b: b.replace(crr.END_MARKER, ""), "no end marker"),
    ],
)
def test_the_reader_refuses_a_missing_edited_or_doubled_record(tmp_path, edit, message):
    promotion = _Promotion(tmp_path / "p")
    body = crr.render_block(unit_record(tmp_path, promotion))
    with pytest.raises(ValueError, match=message):
        read(tmp_path, promotion, Pull(edit(body), promotion.ref))
    assert not (tmp_path / "out" / "candidate-review-records.md").exists()


def test_the_reader_needs_a_record_for_every_unit_of_a_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "MULTI_UNIT_PROMOTION_ENABLED", True)
    repo, head = candidate(tmp_path)
    units = [citation(name) for name in NAMES]
    records = [batch_record(tmp_path, repo, name) for name in NAMES]
    script = load_script()

    def call(body, out):
        return script.read_records(
            dispatch=dispatch(head, units=units),
            repo=repo,
            out_dir=tmp_path / out,
            run_id="123",
            run_attempt=1,
            fetch=Pull(body, head),
        )

    required, summary = call(crr.render_blocks(records), "all")
    assert required and all(citation(name) in summary for name in NAMES)
    with pytest.raises(ValueError, match="no review record for us/statute/7/2015/b"):
        call(crr.render_blocks([records[0], records[2]]), "missing")


# --- Review round 1: the reader's other refusals, its entry point, the audit trail ----


def test_the_reader_refuses_when_the_pull_request_cannot_be_read(tmp_path):
    promotion = _Promotion(tmp_path / "p")

    def failing(path):
        raise OSError("HTTP Error 403: rate limit exceeded")

    with pytest.raises(ValueError) as refusal:
        read(tmp_path, promotion, failing)
    message = str(refusal.value)
    assert "could not be read (OSError)" in message
    # The limit and this run's own count are in the message.
    assert "60 requests an hour" in message and "35 + 4k" in message
    assert "(39 here)" in message
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"number": 7}, "could not be read"),
        ("not an object", "could not be read"),
        ({"number": 2044, "head": "x"}, "head is not the reviewed ref"),
        ({"number": 2044, "body": 5}, "body is not text"),
        ({"number": 2044, "body": ["x"]}, "body is not text"),
    ],
)
def test_the_reader_refuses_an_unexpected_response(tmp_path, payload, message):
    promotion = _Promotion(tmp_path / "p")
    pull = Pull("", promotion.ref)
    if isinstance(payload, dict):
        pull.payload = {"head": {"sha": promotion.ref}, "body": "", **payload}
    else:
        pull.payload = payload
    with pytest.raises(ValueError, match=message):
        read(tmp_path, promotion, pull)
    assert not (tmp_path / "out" / "candidate-review-records.json").exists()


def test_the_reader_treats_a_missing_body_as_empty(tmp_path):
    promotion = _Promotion(tmp_path / "p")
    with pytest.raises(ValueError, match="no review record"):
        read(tmp_path, promotion, Pull(None, promotion.ref))


def test_the_reader_refuses_a_checkout_that_is_not_at_the_reviewed_head(tmp_path):
    promotion = _Promotion(tmp_path / "p")
    body = crr.render_block(unit_record(tmp_path, promotion))
    pull = Pull(body, "d" * 40)
    with pytest.raises(ValueError, match="checkout is not at the reviewed head"):
        read(tmp_path, promotion, pull, dispatch_json=dispatch("d" * 40))
    # Refused before the read: GitHub is not asked.
    assert pull.paths == []


def run_reader_main(tmp_path, monkeypatch, promotion, pull, request=None):
    script = load_script()
    output, summary = tmp_path / "github-output", tmp_path / "github-summary"
    monkeypatch.setenv("DISPATCH_JSON", json.dumps(request or dispatch(promotion.ref)))
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(admission, "github_json", pull)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "read_promotion_review_records.py",
            "read",
            "--repo",
            str(promotion.repo),
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )
    status = script.main()
    return status, summary.read_text() if summary.exists() else ""


def test_the_reader_entry_point_writes_the_summary_of_an_admitted_run(
    tmp_path, monkeypatch
):
    promotion = _Promotion(tmp_path / "p")
    record = unit_record(tmp_path, promotion)
    body = crr.render_block(record)
    status, summary = run_reader_main(
        tmp_path, monkeypatch, promotion, Pull(body, promotion.ref)
    )
    assert status == 0
    # What an auditor needs, without the artifact: the switch, the status,
    # the two file hashes and the digest of the body that was read.
    import hashlib

    assert "Review record required: yes" in summary
    assert "- Status: **passed**" in summary
    assert record["reviewed"]["rulespec_sha256"] in summary
    assert record["reviewed"]["companion_sha256"] in summary
    assert hashlib.sha256(body.encode()).hexdigest() in summary
    assert "attempt 2" in summary


def test_the_reader_entry_point_fails_and_says_why(tmp_path, monkeypatch, capsys):
    promotion = _Promotion(tmp_path / "p")
    status, summary = run_reader_main(
        tmp_path, monkeypatch, promotion, Pull("no record here", promotion.ref)
    )
    assert status == 1
    assert "Review record required: yes" in summary
    assert "Review record refused:" in summary and "no review record" in summary
    assert "Review record refused:" in capsys.readouterr().err


def test_the_reader_entry_point_refuses_a_run_that_does_not_require_records(
    tmp_path, monkeypatch
):
    promotion = _Promotion(tmp_path / "p")
    pull = Pull("", promotion.ref)
    status, summary = run_reader_main(
        tmp_path,
        monkeypatch,
        promotion,
        pull,
        request=dispatch(promotion.ref, required=False),
    )
    assert status == 1 and pull.paths == []
    assert "does not require review records" in summary


def test_the_dispatch_record_states_the_switch():
    spec = importlib.util.spec_from_file_location(
        "_dispatch_record_for_review", ROOT / "scripts/dispatch_record.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def required(bundle):
        return module.build_record({"source_bundle_json": bundle}, {})[
            "review_record_required"
        ]

    on = dispatch("c" * 40)["source_bundle_json"]
    off = dispatch("c" * 40, required=False)["source_bundle_json"]
    assert required(on) is True
    assert required(off) is False
    assert required("[]") is False and required("") is False
    assert (
        required(json.dumps({"schema": "other", "require_review_record": True}))
        is False
    )
    # Unknown, never a silent "no": an unreadable bundle or another value.
    assert required("not json") is None
    assert (
        required(
            on.replace('"require_review_record": true', '"require_review_record": 1')
        )
        is None
    )
