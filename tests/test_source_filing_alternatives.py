"""Source alternatives must not become simultaneous factual obligations."""

import pytest

from axiom_encode.harness import source_completeness as sc

PAIR = (
    "If your spouse or common-law partner is filing a return, use the amounts entered on their return. "
    "If your spouse or common-law partner is not filing a return, use the amounts they would have entered "
    "on their return, schedules, and worksheets as if they were filing a return, "
    "and attach their information slips."
)


def test_complete_same_actor_pair_is_not_conjunctive():
    assert sc._closed_filing_return_alternatives(PAIR)
    assert sc._source_conjunctive_fact_gates(PAIR) == ()
    assert sc._closed_filing_return_alternatives(
        PAIR.replace("schedules, and", "schedules,\nand")
    )


@pytest.mark.parametrize(
    "text",
    [
        PAIR.replace(
            "If your spouse or common-law partner is not",
            "If your common-law partner is not",
        ),
        PAIR.replace("is not filing", "is filing"),
        PAIR.replace(
            "is not filing a return,", "is not filing a return and they are disabled,"
        ),
        PAIR + " If they are resident, attach a certificate.",
        PAIR.replace(
            "attach their information slips",
            "attach their information slips if they are resident",
        ),
        PAIR.replace(
            "use the amounts entered", "use the amounts that would be entered"
        ),
        '"' + PAIR + '"',
        PAIR.replace(
            "If your spouse or common-law partner is not", "If your child is not"
        ),
        "If they are resident, " + PAIR,
    ],
)
def test_changed_or_additional_conditions_are_not_exempt(text):
    assert not sc._closed_filing_return_alternatives(text)


def test_real_conjunction_remains_two_required_facts():
    assert (
        len(
            sc._source_conjunctive_fact_gates(
                "If your spouse is disabled and they are resident, enter the amount."
            )
        )
        == 2
    )
