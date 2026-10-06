"""Adversarial multi-document inputs to repoint workflow routing."""

import pytest

from tests.test_successor_repoint_cli import REPOINT_ENVELOPE, _run_tail


@pytest.mark.parametrize(
    "variable",
    [
        "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON",
        "EXISTING_SIGNED_IMPORTS_JSON",
    ],
)
@pytest.mark.parametrize("value", ['["x"] []', "[] []", "{} []", "null\n[]"])
def test_repoint_routing_refuses_multiple_json_values(tmp_path, variable, value):
    completed, _request, output = _run_tail(
        tmp_path, envelope=REPOINT_ENVELOPE, **{variable: value}
    )
    assert completed.returncode == 1
    assert "cannot mix with replacement" in completed.stderr
    assert "successor_repoint=true" not in output.read_text()
