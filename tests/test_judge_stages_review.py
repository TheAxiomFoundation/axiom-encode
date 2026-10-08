"""Keep the screen's documented policy claims consistent with their evidence."""

import re
from pathlib import Path

import pytest

from axiom_encode.judges import disposition, preclassifier

JUDGE_STAGES = Path(__file__).parents[1] / "docs" / "judge-stages.md"
METHODS_LOG = Path(__file__).parents[1] / "docs" / "axiom-encode-methods-log.md"


def _section(text, heading, next_heading):
    return text.split(heading, 1)[1].split(next_heading, 1)[0]


@pytest.mark.parametrize(
    "heading",
    [
        "Placeholder threshold and the evidence behind it",
        "Live check on real generations",
        "Cost and latency",
    ],
)
def test_reported_measurements_disclose_unpublished_provenance(heading):
    section = _section(JUDGE_STAGES.read_text(), f"### {heading}", "### ")
    text = " ".join(section.lower().split())
    assert "unreproduced" in text
    assert re.search(r"raw (?:scores|data|results).*?not (?:yet )?published", text)


def test_calibration_result_discloses_unpublished_provenance():
    paragraphs = JUDGE_STAGES.read_text().split("\n\n")
    paragraph = next(p for p in paragraphs if "0.546" in p)
    assert "unreproduced" in paragraph.lower()
    assert re.search(
        r"raw (?:scores|data|results).*?not (?:yet )?published",
        " ".join(paragraph.lower().split()),
    )


def test_synthetic_pilot_capability_claims_keep_their_scope():
    paragraphs = [" ".join(p.split()) for p in JUDGE_STAGES.read_text().split("\n\n")]
    capability_claims = [p for p in paragraphs if "dropped conjunct" in p]
    assert capability_claims
    for paragraph in capability_claims:
        assert "unreproduced synthetic pilot" in paragraph.lower()
    assert "the two kinds the pilot validated" not in JUDGE_STAGES.read_text()


@pytest.mark.parametrize("path", [JUDGE_STAGES, METHODS_LOG])
def test_pilot_documentation_uses_no_local_evidence_pointers(path):
    text = path.read_text()
    if path == METHODS_LOG:
        text = _section(text, "### 2026-09-17:", "\n## ")
    assert "_axiom-runs/jev-" not in text
    assert "foundation mirror" not in text.lower()


@pytest.mark.parametrize("path", [JUDGE_STAGES, METHODS_LOG])
def test_pilot_documentation_promises_published_evidence_links(path):
    text = path.read_text()
    if path == METHODS_LOG:
        text = _section(text, "### 2026-09-17:", "\n## ")
    text = " ".join(text.lower().split())
    assert re.search(
        r"figures.*?linked to published evidence.*?artifacts.*?released", text
    )


def test_methods_log_qualifies_pilot_based_capabilities():
    entry = _section(METHODS_LOG.read_text(), "### 2026-09-17:", "\n## ")
    text = " ".join(entry.lower().split())
    assert "unreproduced" in text
    assert re.search(r"raw (?:scores|data|results).*?not (?:yet )?published", text)
    assert "detects reliably" not in text


def test_lowest_threshold_claim_agrees_with_displayed_operating_points():
    text = JUDGE_STAGES.read_text()
    claim = re.search(r"(\d+\.\d+) is the lowest value", text)
    if claim is None:
        return
    qualifying_thresholds = []
    for (
        threshold,
        clean_sent,
        clean_total,
        amounts,
        amount_total,
        boundaries,
        boundary_total,
    ) in re.findall(
        r"\| (\d+\.\d+) \| (\d+) of (\d+) \| (\d+) of (\d+) \| (\d+) of (\d+) \|",
        text,
    ):
        if (
            int(clean_sent) < int(clean_total) / 2
            and amounts == amount_total
            and boundaries == boundary_total
        ):
            qualifying_thresholds.append(float(threshold))
    assert qualifying_thresholds
    assert float(claim.group(1)) == min(qualifying_thresholds)


@pytest.mark.parametrize("stage", ["preclassifier", "disposition"])
def test_successful_deterministic_events_do_not_require_a_judge_model(stage):
    if stage == "preclassifier":
        event = preclassifier.classify(
            {
                "citation": "Act 1134",
                "source_text": "Section 1 is amended by inserting after paragraph (2) the following new subsection.",
            },
            use_llm=False,
        ).event
    else:
        event = disposition.run(
            disposition.Disposition(
                "d",
                "claim",
                residual=1.0,
                records=[{"engine_value": 1, "oracle_value": 2}],
            )
        )
    assert event.model is None
    assert event.verdict.value != "error"
    assert (
        "Every successful event records the judge model" not in JUDGE_STAGES.read_text()
    )


def test_invalid_policy_documentation_distinguishes_cli_from_library():
    text = " ".join(JUDGE_STAGES.read_text().split())
    assert "exit status 2" in text
    assert "before running either judge" in text
    assert "library" in text
    assert "invalid policy" in text.lower()
