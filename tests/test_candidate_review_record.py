"""The candidate review record: build, render, read back, admit or refuse.

No test calls a model: the reviewer CLI is replaced by scripted text.
"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from axiom_encode import candidate_review_record as crr
from axiom_encode import cli
from axiom_encode.candidate_review_record import ReviewRecordError

PASS = (
    '{"score": 9.0, "passed": true, "blocking_issues": [], '
    '"non_blocking_issues": ["The test names are terse."]}'
)
CLEAN = '{"score": 9.5, "passed": true, "blocking_issues": []}'
FAIL = (
    '{"score": 3.0, "passed": false, "blocking_issues": '
    '["The cap of 600 is read as a floor."], "non_blocking_issues": []}'
)
CITATION = "us/statute/7/2015/f"
RULE_PATH = "us/statutes/7/2015/f.yaml"
RULE = (
    b"format: rulespec/v1\n"
    b"module:\n"
    b"  source_verification:\n"
    b"    corpus_citation_path: us/statute/7/2015/f\n"
    b"    source_sha256: " + b"a" * 64 + b"\n"
    b"rules: []\n"
)
COMPANION = b"[]\n"
NOW = datetime(2026, 11, 25, 21, 30, 0, tzinfo=timezone.utc)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def scripted(output):
    """A reviewer CLI double: one text for every call, or one text per call."""

    if isinstance(output, list):
        remaining = list(output)

        def reviewer(prompt, **_kwargs):
            reply = remaining.pop(0)
            return reply if isinstance(reply, tuple) else (reply, 0)

        return reviewer
    return lambda prompt, **_kwargs: (output, 0)


def reviewed(
    tmp_path: Path,
    *,
    output=PASS,
    rulespec_path: str = RULE_PATH,
    rule: bytes = RULE,
    companion: bytes = COMPANION,
    source_text: str = "authoritative source text\n",
):
    """The v2 record and the reviewer result of one scripted review."""

    relative = Path(rulespec_path)
    companion_relative = relative.with_name(relative.stem + ".test.yaml")
    with patch(
        "axiom_encode.harness.validator_pipeline.run_claude_code",
        side_effect=scripted(output),
    ):
        return cli._promotion_generalist_review(
            source_text=source_text,
            cwd=tmp_path,
            applied={relative: rule, companion_relative: companion},
            rulespec_relative=relative,
        )


def make_record(
    tmp_path: Path,
    *,
    output=PASS,
    citation: str = CITATION,
    rulespec_path: str = RULE_PATH,
    rule: bytes = RULE,
    companion: bytes = COMPANION,
    attempt: int = 1,
    encoder_version: str | None = None,
    max_finding_chars: int = crr.DEFAULT_MAX_FINDING_CHARS,
    source_text: str = "authoritative source text\n",
) -> dict:
    """A record as ``review-record create`` builds it, reviewer scripted."""

    review, result = reviewed(
        tmp_path,
        output=output,
        rulespec_path=rulespec_path,
        rule=rule,
        companion=companion,
        source_text=source_text,
    )
    return crr.build_candidate_review_record(
        citation=citation,
        rulespec_path=rulespec_path,
        rulespec=rule,
        companion=companion,
        reviewed_commit="c" * 40,
        encoder_version=encoder_version or cli.__version__,
        encoder_commit="e" * 40,
        generalist_review=review,
        issues=result.issues,
        calls=crr.review_calls(
            result.details,
            passed=result.passed,
            score=result.score,
            prompt_sha256=result.prompt_sha256,
        ),
        raw_output=result.raw_output,
        run_id="local:run-7",
        attempt=attempt,
        max_finding_chars=max_finding_chars,
        now=NOW,
    )


def admit(record, **overrides) -> list[str]:
    arguments = {
        "citation": CITATION,
        "rulespec_path": RULE_PATH,
        "rulespec_sha256": sha(RULE),
        "companion_sha256": sha(COMPANION),
        "encoder_version": cli.__version__,
    }
    arguments.update(overrides)
    return crr.unit_record_issues(record, **arguments)


# --- The record ----------------------------------------------------------------


def test_a_passed_review_gives_a_complete_record(tmp_path):
    record = make_record(tmp_path)
    assert crr.record_issues(record) == []
    assert set(record) == {
        "schema",
        "citation",
        "rulespec_path",
        "reviewed",
        "encoder",
        "status",
        "reviewers",
        "findings",
        "findings_sha256",
        "calls",
        "error",
        "raw_output_sha256",
        "generalist_review",
        "run",
        "created_at",
    }
    assert record["schema"] == "axiom-encode/candidate-review-record/v1"
    assert record["status"] == "passed"
    assert record["reviewed"] == {
        "rulespec_sha256": sha(RULE),
        "companion_sha256": sha(COMPANION),
        "commit": "c" * 40,
    }
    assert record["encoder"] == {"version": cli.__version__, "commit": "e" * 40}
    assert record["run"] == {"id": "local:run-7", "attempt": 1}
    assert record["created_at"] == "2026-11-25T21:30:00Z"
    (reviewer,) = record["reviewers"]
    assert reviewer["name"] == "generalist-reviewer"
    assert reviewer["status"] == "passed" and reviewer["score"] == 9.0
    assert reviewer["prompt_sha256"] == record["generalist_review"]["prompt_sha256"]
    assert record["findings"] == [
        {
            "reviewer": "generalist-reviewer",
            "severity": "non_blocking",
            "text": "The test names are terse.",
            "truncated": False,
            "review_call": None,
        }
    ]
    assert record["error"] is None
    (call,) = record["calls"]
    assert (call["index"], call["total"], call["status"]) == (1, 1, "passed")
    assert call["prompt_sha256"] == reviewer["prompt_sha256"]
    assert record["raw_output_sha256"] == sha(PASS.encode())
    assert admit(record) == []


def test_a_failed_review_is_recorded_with_its_blocking_finding(tmp_path):
    record = make_record(tmp_path, output=FAIL)
    assert record["status"] == "failed"
    assert [finding["severity"] for finding in record["findings"]] == ["blocking"]
    assert record["findings"][0]["text"] == "The cap of 600 is read as a floor."
    assert admit(record) == ["the review status is failed, not passed"]


def test_a_long_finding_is_cut_and_marked(tmp_path):
    long = json.dumps(
        {
            "score": 9.0,
            "passed": True,
            "blocking_issues": [],
            "non_blocking_issues": ["x" * 500],
        }
    )
    record = make_record(tmp_path, output=long, max_finding_chars=40)
    assert record["findings"][0]["text"] == "x" * 40
    assert record["findings"][0]["truncated"] is True
    assert "[text cut]" in crr.render_block(record)


def test_more_findings_than_a_record_holds_are_refused_not_dropped():
    with pytest.raises(ReviewRecordError, match="at most 64"):
        crr.findings_from_issues(["a finding"] * 65)


# --- Admission of a unit: every refusal ----------------------------------------


def test_a_changed_module_is_refused(tmp_path):
    issues = admit(make_record(tmp_path), rulespec_sha256=sha(RULE + b"# edit\n"))
    assert len(issues) == 1 and "module changed after the review" in issues[0]


def test_a_changed_companion_is_refused(tmp_path):
    issues = admit(make_record(tmp_path), companion_sha256=sha(b"- name: new\n"))
    assert len(issues) == 1 and "companion test changed after the review" in issues[0]


def test_a_record_of_another_citation_is_refused(tmp_path):
    issues = admit(make_record(tmp_path), citation="us/statute/7/2015/g")
    assert issues == [
        "the record is for citation us/statute/7/2015/f, not us/statute/7/2015/g"
    ]


def test_a_record_of_another_path_is_refused(tmp_path):
    issues = admit(make_record(tmp_path), rulespec_path="us/statutes/7/2015/g.yaml")
    assert issues == [
        "the record is for module us/statutes/7/2015/f.yaml, not "
        "us/statutes/7/2015/g.yaml"
    ]


def test_a_record_of_a_newer_encoder_is_refused(tmp_path):
    record = make_record(tmp_path, encoder_version="9.0.0")
    (issue,) = admit(record)
    assert "newer than the signing encoder" in issue
    assert admit(make_record(tmp_path, encoder_version="0.0.1")) == []


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(status="passed"),
        lambda r: r["reviewers"][0].update(status="passed", score=9.9),
        lambda r: r["generalist_review"].update(status="passed"),
        lambda r: r["findings"].clear(),
    ],
)
def test_a_failed_record_edited_to_look_passed_is_refused(tmp_path, change):
    record = make_record(tmp_path, output=FAIL)
    change(record)
    assert admit(record) != []


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(extra=1),
        lambda r: r.pop("findings"),
        lambda r: r.update(schema="axiom-encode/candidate-review-record/v2"),
        lambda r: r["reviewed"].update(rulespec_sha256="A" * 64),
        lambda r: r["run"].update(attempt=0),
        lambda r: r["run"].update(attempt=True),
        lambda r: r.update(created_at="yesterday"),
        lambda r: r["findings"][0].update(severity="minor"),
        lambda r: r["findings"][0].update(severity="blocking"),
        lambda r: r["findings"].append(dict(r["findings"][0])),
        lambda r: r["encoder"].update(version="latest"),
        lambda r: r["generalist_review"]["verdict"].update(score=1.0),
        lambda r: r["generalist_review"]["files"][0].update(sha256="0" * 64),
        lambda r: r.update(rulespec_path="us/statutes/7/2015/../f.yaml"),
    ],
)
def test_a_malformed_or_contradictory_record_admits_nothing(tmp_path, change):
    record = make_record(tmp_path)
    change(record)
    assert admit(record) != []


def test_a_batch_needs_a_record_for_every_unit(tmp_path):
    other_rule = RULE.replace(b"2015/f", b"2015/g")
    first = make_record(tmp_path)
    second = make_record(
        tmp_path,
        citation="us/statute/7/2015/g",
        rulespec_path="us/statutes/7/2015/g.yaml",
        rule=other_rule,
    )
    units = [
        (CITATION, RULE_PATH, sha(RULE), sha(COMPANION)),
        (
            "us/statute/7/2015/g",
            "us/statutes/7/2015/g.yaml",
            sha(other_rule),
            sha(COMPANION),
        ),
    ]
    admitted = crr.require_passed_unit_records(
        [second, first], units, encoder_version=cli.__version__
    )
    assert admitted == [first, second]
    with pytest.raises(
        ReviewRecordError, match="no review record for us/statute/7/2015/g"
    ):
        crr.require_passed_unit_records([first], units, encoder_version=cli.__version__)
    failed = make_record(
        tmp_path,
        output=FAIL,
        citation="us/statute/7/2015/g",
        rulespec_path="us/statutes/7/2015/g.yaml",
        rule=other_rule,
    )
    with pytest.raises(ReviewRecordError, match="status is failed"):
        crr.require_passed_unit_records(
            [first, failed], units, encoder_version=cli.__version__
        )


def test_two_records_for_one_unit_are_refused_also_when_equal(tmp_path):
    record = make_record(tmp_path)
    units = [(CITATION, RULE_PATH, sha(RULE), sha(COMPANION))]
    with pytest.raises(ReviewRecordError, match="2 review records name"):
        crr.require_passed_unit_records(
            [record, copy.deepcopy(record)], units, encoder_version=cli.__version__
        )
    failed = make_record(tmp_path, output=FAIL)
    with pytest.raises(ReviewRecordError, match="2 review records name"):
        crr.require_passed_unit_records(
            [failed, record], units, encoder_version=cli.__version__
        )


# --- The pull request body ------------------------------------------------------


def test_render_and_read_back_give_the_same_record(tmp_path):
    record = make_record(tmp_path)
    block = crr.render_block(record)
    body = "Candidate for 7 USC 2015(f).\n\n" + block + "\nMore text.\n"
    assert crr.extract_records(body) == [record]
    assert crr.extract_records(body.replace("\n", "\r\n")) == [record]
    assert block.startswith(crr.BEGIN_MARKER + "\n")
    assert block.endswith(crr.END_MARKER + "\n")
    assert "- Status: **passed** (score 9/10)" in block
    assert (
        "1. **non-blocking** (generalist-reviewer): ` The test names are terse. `"
        in block
    )
    assert f"`{sha(RULE)}`" in block and f"`{sha(COMPANION)}`" in block


def test_a_body_without_a_block_has_no_record():
    assert crr.extract_records("Reviewer verdict: passed, 9/10.\n") == []
    assert crr.extract_records("") == []


def test_finding_text_cannot_form_a_marker_a_fence_or_markup(tmp_path):
    hostile = (
        "ends here\n" + crr.END_MARKER + "\n```json\n{}\n```\n"
        "<script>x</script> `code` **bold** [link](http://example.invalid) "
        + crr.BEGIN_MARKER
    )
    output = json.dumps(
        {
            "score": 9.0,
            "passed": True,
            "blocking_issues": [],
            "non_blocking_issues": [hostile],
        }
    )
    record = make_record(tmp_path, output=output)
    block = crr.render_block(record)
    assert block.count(crr.BEGIN_MARKER) == 1 and block.count(crr.END_MARKER) == 1
    assert block.count("```") == 2
    readable = block.split("```json", 1)[0]
    # The finding is one code span: GitHub renders no markup, link, HTML or
    # mention inside it, and the text holds no backtick that could end it.
    (line,) = [item for item in readable.splitlines() if item.startswith("1. ")]
    span = line.split("): ", 1)[1]
    assert span.startswith("` ") and span.endswith(" `")
    assert "`" not in span[1:-1]
    assert "<!--" not in line and "-->" not in line
    assert "<" not in block.split("```json\n", 1)[1].split("\n```", 1)[0]
    (read,) = crr.extract_records("intro\n" + block)
    assert read == record
    assert read["findings"][0]["text"].endswith(crr.BEGIN_MARKER)


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (lambda b: b.replace(crr.END_MARKER, ""), "no end marker"),
        (lambda b: b.replace(crr.BEGIN_MARKER, ""), "without a start marker"),
        (
            lambda b: b.replace(
                crr.BEGIN_MARKER, crr.BEGIN_MARKER * 1 + "\n" + crr.BEGIN_MARKER
            ),
            "inside another block",
        ),
        (lambda b: b.replace('{"calls"', '{"calls'), "not valid JSON"),
        (
            lambda b: b.replace("```json\n{", "```json\n[{").replace(
                "}\n```", "}]\n```"
            ),
            "not an object",
        ),
        (lambda b: b.replace("```json\n", "```json\n\n"), "single-line fenced json"),
        (lambda b: b.replace("```json\n", "```\n"), "single-line fenced json"),
        (
            lambda b: b.replace(":v1:begin -->", ":v2:begin -->"),
            "not exactly one of the two marker lines",
        ),
        (
            lambda b: b.replace(crr.BEGIN_MARKER, "text " + crr.BEGIN_MARKER + " text"),
            "not exactly one of the two marker lines",
        ),
        (
            lambda b: b.replace('{"calls":', '{"calls":[],"calls":'),
            "repeats the key",
        ),
    ],
)
def test_a_damaged_block_refuses_the_whole_body(tmp_path, damage, message):
    body = "intro\n" + crr.render_block(make_record(tmp_path)) + "outro\n"
    damaged = damage(body)
    assert damaged != body
    with pytest.raises(ReviewRecordError, match=message):
        crr.extract_records(damaged)


def test_a_record_edited_in_the_body_is_refused(tmp_path):
    record = make_record(tmp_path, output=FAIL)
    body = crr.render_block(record)
    edited = body.replace('"status":"failed"', '"status":"passed"')
    assert edited != body
    (read,) = crr.extract_records(edited)
    assert admit(read) != []


def test_records_that_do_not_fit_a_body_are_refused_at_render(tmp_path):
    output = json.dumps(
        {
            "score": 9.0,
            "passed": True,
            "blocking_issues": [],
            "non_blocking_issues": ["y" * 3900] * 20,
        }
    )
    record = make_record(tmp_path, output=output, max_finding_chars=4000)
    with pytest.raises(ReviewRecordError, match="--max-finding-chars"):
        crr.render_blocks([record])


def test_the_read_document_round_trips_and_is_strict(tmp_path):
    record = make_record(tmp_path)
    body = crr.render_block(record)
    document = crr.build_read_document(
        rulespec_pr=2044,
        head_sha="c" * 40,
        body=body,
        records=[record],
        run_id="123",
        run_attempt=2,
        now=NOW,
    )
    assert document["body_sha256"] == sha(body.encode())
    assert document["signing_run_attempt"] == 2
    assert crr.load_read_document(json.dumps(document)) == document
    for change in (
        lambda d: d.update(extra=1),
        lambda d: d.update(schema="other"),
        lambda d: d.update(head_sha="c" * 39),
        lambda d: d.update(records={}),
    ):
        broken = copy.deepcopy(document)
        change(broken)
        with pytest.raises(ReviewRecordError):
            crr.load_read_document(json.dumps(broken))
    summary = crr.render_read_summary(document, [record])
    assert summary.startswith("## Review record\n")
    assert "#2044" in summary and sha(body.encode()) in summary
    assert "Signing run: 123, attempt 2" in summary
    assert crr.extract_records(summary) == [record]


# --- Review round 1 ---------------------------------------------------------------


def with_finding(tmp_path, text: str) -> dict:
    output = json.dumps(
        {
            "score": 9.0,
            "passed": True,
            "blocking_issues": [],
            "non_blocking_issues": [text],
        }
    )
    return make_record(tmp_path, output=output)


def test_mentions_and_links_are_inert_in_the_readable_part(tmp_path):
    record = with_finding(
        tmp_path, "ping @MaxGhenis and @org/team, see https://example.invalid/x #12"
    )
    readable = crr.render_block(record).split("```json", 1)[0]
    (line,) = [item for item in readable.splitlines() if item.startswith("1. ")]
    # The whole text is inside one code span, where GitHub makes no mention,
    # no link and no issue reference.
    assert line == (
        "1. **non-blocking** (generalist-reviewer): "
        "` ping @MaxGhenis and @org/team, see https://example.invalid/x #12 `"
    )


@pytest.mark.parametrize(
    "run_id", ["@MaxGhenis", "a/b", "https://example.invalid", "a b", "x" * 101, ""]
)
def test_a_run_id_that_could_mention_or_link_is_refused(tmp_path, run_id):
    record = make_record(tmp_path)
    record["run"]["id"] = run_id
    assert admit(record) == ["the run id or the attempt number is malformed"]
    assert "`local:run-7`" in crr.render_block(make_record(tmp_path))


def test_an_edited_finding_text_is_refused(tmp_path):
    record = make_record(tmp_path)
    assert record["findings_sha256"] == crr.findings_sha256(record["findings"])
    edited = copy.deepcopy(record)
    edited["findings"][0]["text"] = "No finding."
    (issue,) = admit(edited)
    assert "the findings differ from their digest" in issue
    # The same edit in a rendered body is refused when the body is checked.
    body = crr.render_block(record).replace("The test names are terse.", "No finding.")
    (read,) = crr.extract_records(body)
    assert admit(read) != []


def test_readable_text_that_differs_from_the_json_refuses_the_body(tmp_path):
    block = crr.render_block(make_record(tmp_path))
    for changed in (
        block.replace("- Status: **passed**", "- Status: **failed**"),
        block.replace("**non-blocking**", "**BLOCKING**"),
        block.replace("Findings (1):", "Findings (1):\n\nAn added line."),
        block.replace("` The test names are terse. `", "` Nothing found. `"),
    ):
        assert changed != block
        with pytest.raises(ReviewRecordError, match="exact rendered form"):
            crr.extract_records("intro\n" + changed)


@pytest.mark.parametrize(
    ("wrap", "message"),
    [
        (lambda b: "```\n" + b + "```\n", "inside a code fence"),
        (lambda b: "~~~~ text\n" + b + "~~~~\n", "inside a code fence"),
        (
            lambda b: "<details><summary>x</summary>\n\n" + b + "</details>\n",
            "<details>",
        ),
        (lambda b: "<DETAILS>\n" + b, "<details>"),
        (
            lambda b: b.replace(crr.BEGIN_MARKER, "\u00a0" + crr.BEGIN_MARKER),
            "marker lines",
        ),
        (lambda b: b.replace(crr.BEGIN_MARKER, " " + crr.BEGIN_MARKER), "marker lines"),
        (lambda b: b.replace(crr.END_MARKER, crr.END_MARKER + " "), "marker lines"),
        (lambda b: "> " + b.replace("\n", "\n> "), "marker lines"),
    ],
)
def test_a_block_that_is_not_at_the_top_level_of_the_body_is_refused(
    tmp_path, wrap, message
):
    block = crr.render_block(make_record(tmp_path))
    with pytest.raises(ReviewRecordError, match=message):
        crr.extract_records("intro\n\n" + wrap(block) + "\noutro\n")


def test_a_block_after_a_closed_fence_or_details_is_read(tmp_path):
    record = make_record(tmp_path)
    body = (
        "```yaml\nrules: []\n```\n\n<details>\nnotes\n</details>\n\n"
        + crr.render_block(record)
    )
    assert crr.extract_records(body) == [record]


def test_more_records_than_a_body_may_hold_are_refused(tmp_path):
    block = crr.render_block(make_record(tmp_path, output=CLEAN))
    assert len(crr.extract_records(block * crr.MAX_RECORDS_PER_BODY)) == 64
    with pytest.raises(ReviewRecordError, match="too many review records"):
        crr.extract_records(block * (crr.MAX_RECORDS_PER_BODY + 1))


@pytest.mark.parametrize(
    "status", ["Passed", "PASSED", " passed", "passed ", "pass", True]
)
def test_only_the_exact_word_passed_is_a_pass(tmp_path, status):
    record = make_record(tmp_path)
    record["status"] = status
    assert admit(record) != []
    record["reviewers"][0]["status"] = status
    record["generalist_review"]["status"] = status
    assert admit(record) != []
