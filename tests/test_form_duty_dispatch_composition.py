"""Closed instruction frames keep their own antecedents after composition."""

import pytest

from axiom_encode.harness import source_completeness as sc
from tests.test_chart_completion_purpose import ANTECEDENT as CHART_ANTECEDENT
from tests.test_chart_completion_purpose import COMMAND as CHART_COMMAND
from tests.test_nonfiler_imperative_consequent import ANTECEDENT as NONFILER_ANTECEDENT
from tests.test_nonfiler_imperative_consequent import COMMAND as NONFILER_COMMAND


@pytest.mark.parametrize(
    "body,other,expected",
    [
        (
            CHART_ANTECEDENT + CHART_COMMAND,
            sc._nonfiler_return_instruction_antecedent,
            CHART_ANTECEDENT.replace("small\nbusiness", "small business"),
        ),
        (
            NONFILER_ANTECEDENT + NONFILER_COMMAND,
            sc._chart_completion_purpose_antecedent,
            NONFILER_ANTECEDENT,
        ),
    ],
)
def test_closed_instruction_frames_do_not_capture_each_other(body, other, expected):
    assert other(body) is None
    assert sc._without_worksheet_imperative_consequent(body) == expected
