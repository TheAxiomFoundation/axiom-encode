"""Keep the screen's documented policy claims consistent with their evidence."""

import re
from pathlib import Path

import pytest

from axiom_encode.judges import disposition, preclassifier

JUDGE_STAGES = Path(__file__).parents[1] / "docs" / "judge-stages.md"


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
