"""The whole record path over single-call and chunked reviews, offline.

Live proof round: ``review-record create`` wrote no record for a passed
chunked review with a non-blocking finding, because the pipeline marks the
issues of a chunked review ``[review call N of M] [non-blocking] ...`` and
the record read the non-blocking mark only at the start of the text.

No test calls a model: the reviewer CLI is replaced by scripted text, one
text per call.
"""

from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode import candidate_review_record as crr
from axiom_encode import cli
from axiom_encode.cli import main
from axiom_encode.generalist_review_chunks import chunked_prompt_sha256
from axiom_encode.harness import validator_pipeline
from tests.test_candidate_review_record import (
    COMPANION,
    RULE,
    admit,
    make_record,
    reviewed,
    scripted,
    sha,
)
from tests.test_generalist_review_promotion import _Promotion
from tests.test_review_record_promotion import CITATION, write_document


def reply(score=9.0, passed=True, blocking=(), non_blocking=()):
    return json.dumps(
        {
            "score": score,
            "passed": passed,
            "blocking_issues": list(blocking),
            "non_blocking_issues": list(non_blocking),
            "reasoning": "synthetic",
        }
    )


def source_for(calls: int) -> str:
    """Source text that needs ``calls`` reviewer calls (1 call: one chunk)."""

    if calls == 0:
        return "authoritative source text\n"
    paragraph = "A source paragraph of the synthetic section, number {:05d}."
    count = 100 if calls == 1 else 430 * (calls - 1) + 60
    return "\n\n".join(paragraph.format(index) for index in range(count)) + "\n"


def planned_calls(tmp_path, calls: int) -> int:
    """How many calls the pipeline plans for ``source_for(calls)``."""

    seen = []

    def reviewer(prompt, **_kwargs):
        seen.append(prompt)
        return reply(), 0

    with patch(
        "axiom_encode.harness.validator_pipeline.run_claude_code", side_effect=reviewer
    ):
        cli._promotion_generalist_review(
            source_text=source_for(calls),
            cwd=tmp_path,
            applied={
                crr_path(): RULE,
                crr_path().with_name("f.test.yaml"): COMPANION,
            },
            rulespec_relative=crr_path(),
        )
    return len(seen)


def crr_path():
    from pathlib import Path

    return Path("us/statutes/7/2015/f.yaml")


@pytest.mark.parametrize("calls", [1, 2, 5])
def test_the_synthetic_sources_need_the_stated_number_of_calls(tmp_path, calls):
    assert planned_calls(tmp_path, calls) == calls


# --- The grammar of an issue text ------------------------------------------------------


def test_the_record_reads_exactly_the_marks_that_the_pipeline_writes():
    """Fails when the pipeline changes how it marks an issue."""

    non_blocking = validator_pipeline.non_blocking_issue_text("terse names")
    assert crr.parse_issue_text("a defect") == (None, "blocking", "a defect")
    assert crr.parse_issue_text(non_blocking) == (None, "non_blocking", "terse names")
    for index, total in ((1, 1), (3, 5), (12, 12)):
        call = {"index": index, "total": total}
        assert crr.parse_issue_text(
            validator_pipeline.review_call_issue_text(index, total, "a defect")
        ) == (call, "blocking", "a defect")
        assert crr.parse_issue_text(
            validator_pipeline.review_call_issue_text(index, total, non_blocking)
        ) == (call, "non_blocking", "terse names")


def test_the_live_defect_text_is_non_blocking():
    (finding,) = crr.findings_from_issues(["[review call 1 of 1] [non-blocking] x"])
    assert finding["severity"] == "non_blocking"
    assert finding["text"] == "x"
    assert finding["review_call"] == {"index": 1, "total": 1}


@pytest.mark.parametrize(
    "text",
    [
        "[Non-Blocking] x",
        "[non-blocking]x",
        " [non-blocking] x",
        "[review call 1 of 1][non-blocking] x",
        "[review call 0 of 1] [non-blocking] x",
        "[review call 2 of 1] [non-blocking] x",
        "[review call 1 of 99] [non-blocking] x",
        "[review call one of two] [non-blocking] x",
        "[call 1 of 1] [non-blocking] x",
        "[non-blocking] [review call 1 of 1] y",
        "note: [non-blocking] x",
        "reviewer_score_below_pass_threshold: score 6.0/10 is below 7.0",
        "reviewer_parse_failed",
    ],
)
def test_a_text_outside_the_grammar_is_blocking(text):
    (finding,) = crr.findings_from_issues([text])
    if text.startswith("[non-blocking] "):
        # The exact mark at the start is the grammar; the rest is its text.
        assert finding["severity"] == "non_blocking"
        assert finding["text"] == "[review call 1 of 1] y"
    else:
        assert finding["severity"] == "blocking"
    assert finding["review_call"] is None or text.startswith("[review call 1 of 1] ")


def test_the_real_chunked_path_gives_findings_that_parse(tmp_path):
    """Runs the pipeline itself, so a format change there fails here."""

    _review, result = reviewed(
        tmp_path,
        output=[
            reply(non_blocking=["first"]),
            reply(blocking=["second"], passed=False, score=4.0),
        ],
        source_text=source_for(2),
    )
    assert crr.findings_from_issues(result.issues) == [
        {
            "reviewer": "generalist-reviewer",
            "severity": "non_blocking",
            "text": "first",
            "truncated": False,
            "review_call": {"index": 1, "total": 2},
        },
        {
            "reviewer": "generalist-reviewer",
            "severity": "blocking",
            "text": "second",
            "truncated": False,
            "review_call": {"index": 2, "total": 2},
        },
    ]


# --- Records of single-call and chunked reviews ------------------------------------------


@pytest.mark.parametrize("calls", [0, 1, 2, 5])
@pytest.mark.parametrize("with_findings", [False, True])
def test_a_passed_review_is_recorded_for_every_number_of_calls(
    tmp_path, calls, with_findings
):
    total = max(calls, 1)
    outputs = [
        reply(non_blocking=[f"note of call {index}"] if with_findings else [])
        for index in range(1, total + 1)
    ]
    record = make_record(tmp_path, output=outputs, source_text=source_for(calls))
    assert record["status"] == "passed" and admit(record) == []
    assert [(c["index"], c["total"], c["status"]) for c in record["calls"]] == [
        (index, total, "passed") for index in range(1, total + 1)
    ]
    coverage = record["generalist_review"]["coverage"]
    assert (coverage["calls_planned"], coverage["calls_made"]) == (total, total)
    assert all(entry["complete"] for entry in record["generalist_review"]["files"])
    digests = tuple(call["prompt_sha256"] for call in record["calls"])
    assert record["reviewers"][0]["prompt_sha256"] in (
        {digests[0], chunked_prompt_sha256(digests)}
        if total == 1
        else {chunked_prompt_sha256(digests)}
    )
    assert len(record["findings"]) == (total if with_findings else 0)
    assert all(f["severity"] == "non_blocking" for f in record["findings"])
    if calls:
        assert [f["review_call"] for f in record["findings"]] == (
            [{"index": i, "total": total} for i in range(1, total + 1)]
            if with_findings
            else []
        )
        assert not any("review call" in f["text"] for f in record["findings"])
    block = crr.render_block(record)
    assert f"- Reviewer calls: {total} made of {total} planned" in block
    if calls and with_findings:
        assert f"(generalist-reviewer, call {total} of {total}): " in block
    assert crr.extract_records(block) == [record]


def test_a_blocking_finding_in_call_3_stops_the_review_and_is_recorded(tmp_path):
    record = make_record(
        tmp_path,
        output=[
            reply(non_blocking=["note 1"]),
            reply(),
            reply(score=3.0, passed=False, blocking=["The cap is read as a floor."]),
        ],
        source_text=source_for(5),
    )
    assert record["status"] == "failed"
    assert [(c["index"], c["total"], c["status"]) for c in record["calls"]] == [
        (1, 5, "passed"),
        (2, 5, "passed"),
        (3, 5, "failed"),
    ]
    assert record["findings"][-1] == {
        "reviewer": "generalist-reviewer",
        "severity": "blocking",
        "text": "The cap is read as a floor.",
        "truncated": False,
        "review_call": {"index": 3, "total": 5},
    }
    (issue,) = admit(record)
    assert "the review status is failed" in issue
    assert "- Reviewer calls: 3 made of 5 planned (1 passed, 2 passed, 3 failed)" in (
        crr.render_block(record)
    )


def test_a_call_that_errors_is_recorded_and_is_not_a_pass(tmp_path):
    record = make_record(
        tmp_path,
        output=[reply(non_blocking=["note 1"]), ("the reviewer crashed", 1)],
        source_text=source_for(2),
    )
    assert record["status"] == "unavailable"
    assert [(c["index"], c["status"]) for c in record["calls"]] == [
        (1, "passed"),
        (2, "error"),
    ]
    assert record["findings"][0]["review_call"] == {"index": 1, "total": 2}
    assert admit(record) != []


def test_a_review_that_did_not_show_every_byte_is_not_passed(tmp_path):
    """Above the chunk caps the reviewer sees the start only: ``partial``."""

    record = make_record(tmp_path, output=reply(), source_text="word\n" * (8 * 24_000))
    assert record["status"] == "partial"
    assert record["generalist_review"]["reason"] == "review_saw_truncated_source"
    (issue,) = admit(record)
    assert "the review status is partial" in issue


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["calls"].pop(),
        lambda r: r["calls"][-1].update(status="failed"),
        lambda r: r["calls"][0].update(prompt_sha256="0" * 64),
        lambda r: r["calls"].reverse(),
        lambda r: r["calls"][0].update(total=3),
        lambda r: r["calls"].clear(),
        lambda r: (
            r["findings"][0]["review_call"].update(index=2, total=3)
            or r.update(findings_sha256=crr.findings_sha256(r["findings"]))
        ),
    ],
)
def test_a_passed_record_whose_calls_do_not_fit_is_refused(tmp_path, change):
    record = make_record(
        tmp_path,
        output=[reply(non_blocking=["a"]), reply()],
        source_text=source_for(2),
    )
    assert admit(record) == []
    change(record)
    assert admit(record) != []


# --- review-record create always leaves a record -------------------------------------------


def run_create(tmp_path, repo, out, *, output, source_text, raise_in=None):
    (tmp_path / "corpus").mkdir(exist_ok=True)
    patches = [
        patch("axiom_encode.cli.load_rulespec_local_corpus_release"),
        patch(
            "axiom_encode.cli.resolve_corpus_source_unit",
            **(
                {"side_effect": RuntimeError("corpus release signature is invalid")}
                if raise_in == "source"
                else {"return_value": SimpleNamespace(body=source_text)}
            ),
        ),
        patch(
            "axiom_encode.harness.validator_pipeline.run_claude_code",
            side_effect=scripted(output),
        ),
        patch.object(
            sys,
            "argv",
            [
                "axiom-encode",
                "review-record",
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
                "local:proof",
            ],
        ),
    ]
    if raise_in == "reviewer":
        patches.append(
            patch(
                "axiom_encode.cli._promotion_generalist_review",
                side_effect=RuntimeError("reviewer process crashed"),
            )
        )
    for item in patches:
        item.start()
    try:
        with pytest.raises(SystemExit) as exit_info:
            main()
    finally:
        for item in reversed(patches):
            item.stop()
    return exit_info.value.code


def test_the_live_case_now_writes_a_passed_record(tmp_path, capsys):
    """One chunked call, passed, with a non-blocking finding (candidate of the
    live proof): before the fix, exit 2 and no file."""

    promotion = _Promotion(tmp_path / "p")
    out = tmp_path / "record.json"
    output = reply(score=8.5, non_blocking=["The inputs could be more precise."])
    code = run_create(
        tmp_path, promotion.repo, out, output=output, source_text=source_for(1)
    )
    printed = capsys.readouterr().out
    assert code == 0, printed
    record = json.loads(out.read_text())
    assert record["status"] == "passed"
    assert record["findings"][0]["severity"] == "non_blocking"
    assert record["findings"][0]["review_call"] == {"index": 1, "total": 1}
    raw = out.with_name("record.json.raw.txt")
    assert raw.read_text() == output
    assert record["raw_output_sha256"] == hashlib.sha256(output.encode()).hexdigest()
    assert f"raw_output_file={raw}" in printed and "status=passed" in printed
    # The signing check admits it.
    promotion.args.require_review_record = write_document(
        tmp_path, [record], promotion.ref
    )
    promotion.run()
    assert promotion.order == ["validate", "sign"]


@pytest.mark.parametrize(
    ("output", "calls", "status"),
    [
        (reply(score=3.0, passed=False, blocking=["a defect"]), 0, "failed"),
        ([reply(), reply(score=2.0, passed=False, blocking=["x"])], 2, "failed"),
        ([reply(), ("crash", 1)], 2, "unavailable"),
        ("not json at all", 0, "unavailable"),
    ],
)
def test_a_review_that_did_not_pass_leaves_a_record_and_exits_1(
    tmp_path, capsys, output, calls, status
):
    promotion = _Promotion(tmp_path / "p")
    out = tmp_path / "record.json"
    code = run_create(
        tmp_path, promotion.repo, out, output=output, source_text=source_for(calls)
    )
    assert code == 1
    record = json.loads(out.read_text())
    assert record["status"] == status and crr.record_issues(record) == []
    assert out.with_name("record.json.raw.txt").is_file()
    assert f"status={status}" in capsys.readouterr().out
    assert crr.extract_records(crr.render_block(record)) == [record]
    assert (
        crr.unit_record_issues(
            record,
            citation=CITATION,
            rulespec_path="us/statutes/7/2015/f.yaml",
            rulespec_sha256=record["reviewed"]["rulespec_sha256"],
            companion_sha256=record["reviewed"]["companion_sha256"],
            encoder_version=cli.__version__,
        )
        != []
    )


def test_a_passed_review_with_a_text_outside_the_grammar_is_an_error_record(
    tmp_path, capsys
):
    """Fail closed, and keep the evidence: an unknown mark stays blocking, a
    passed review cannot carry a blocking finding, so the record is ``error``."""

    promotion = _Promotion(tmp_path / "p")
    out = tmp_path / "record.json"
    real = cli._promotion_generalist_review

    def unknown_mark(**kwargs):
        record, result = real(**kwargs)
        result.issues = ["[advisory] an unknown mark"]
        return record, result

    with patch("axiom_encode.cli._promotion_generalist_review", unknown_mark):
        code = run_create(
            tmp_path, promotion.repo, out, output=reply(), source_text=source_for(0)
        )
    assert code == 1
    record = json.loads(out.read_text())
    assert record["status"] == "error"
    assert "a passed record carries a blocking finding" in record["error"]
    assert record["findings"][0]["severity"] == "blocking"
    assert record["generalist_review"]["status"] == "passed"
    assert out.with_name("record.json.raw.txt").is_file()
    printed = capsys.readouterr().out
    assert "status=error" in printed and "error=" in printed
    block = crr.render_block(record)
    assert "- Status: **error**" in block and "- Error: ` " in block
    assert crr.extract_records(block) == [record]
    # An error record never admits a unit, also with the right hashes.
    promotion.args.require_review_record = write_document(
        tmp_path, [record], promotion.ref
    )
    with pytest.raises(RuntimeError, match="the review status is error"):
        promotion.run()
    assert promotion.signed == []


@pytest.mark.parametrize("where", ["source", "reviewer"])
def test_a_crash_after_the_unit_is_read_leaves_an_error_record(tmp_path, capsys, where):
    promotion = _Promotion(tmp_path / "p")
    out = tmp_path / "record.json"
    code = run_create(
        tmp_path,
        promotion.repo,
        out,
        output=reply(),
        source_text=source_for(0),
        raise_in=where,
    )
    assert code == 1
    record = json.loads(out.read_text())
    assert record["status"] == "error" and crr.record_issues(record) == []
    assert record["error"].startswith("RuntimeError: ")
    assert record["reviewed"]["rulespec_sha256"] == sha(promotion.rule.read_bytes())
    assert record["calls"] == [] and record["generalist_review"] is None
    assert record["run"] == {"id": "local:proof", "attempt": 1}
    assert "status=error" in capsys.readouterr().out


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(status="passed"),
        lambda r: r.update(error=None),
        lambda r: r.update(error=""),
        lambda r: r.update(error="x" * 2001),
        lambda r: r.update(reviewers=[{"name": "generalist-reviewer"}]),
        lambda r: r.update(generalist_review="passed"),
    ],
)
def test_an_error_record_cannot_be_turned_into_another_record(tmp_path, change):
    record = crr.build_error_record(
        citation=CITATION,
        rulespec_path="us/statutes/7/2015/f.yaml",
        rulespec=RULE,
        companion=COMPANION,
        reviewed_commit=None,
        encoder_version=cli.__version__,
        encoder_commit=None,
        error="RuntimeError: crash",
        run_id=None,
        attempt=1,
    )
    assert crr.record_issues(record) == [] and admit(record) != []
    change(record)
    assert admit(record) != []


def test_a_normal_record_cannot_carry_an_error_text(tmp_path):
    record = make_record(tmp_path)
    record["error"] = "ignored?"
    assert admit(record) != []
