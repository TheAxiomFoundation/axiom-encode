import pytest

from axiom_encode.harness.source_completeness import (
    _source_is_nonoperative_guidance_instruction,
)


def test_optional_copy_referral_is_nonoperative():
    assert _source_is_nonoperative_guidance_instruction(
        "If you want to keep a copy for your records, see the guide."
    )
    assert _source_is_nonoperative_guidance_instruction(
        "If you want to keep a copy\nfor your records, see the guide."
    )


@pytest.mark.parametrize(
    "text",
    [
        "If you keep a copy for your records, you are eligible for the credit.",
        "If you want to keep a copy for your records, see the guide; retain it to claim the credit.",
        "If you want to keep a copy for your records, see the guide and subtract line 60.",
        "You must keep records for 6 years to claim the credit.",
        "If you want to keep a copy for your records, see the guide. Claim 100 dollars.",
        "If you want to keep a copy for your records, see the guide unless your income exceeds 100.",
        "If interest was paid in 2025, complete this schedule.",
    ],
)
def test_operative_or_unknown_recordkeeping_stays_required(text):
    assert not _source_is_nonoperative_guidance_instruction(text)
