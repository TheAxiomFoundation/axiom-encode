"""Lifetime reductions must not become fabricated dataset input slots."""

from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import evals, validator_pipeline


@pytest.fixture(
    params=[
        "sum_over_periods(net_value)",
        "max_over_periods(net_value)",
        "count_over_periods(net_value > 0)",
        "sum_top_n_over_periods(net_value, count)",
        "calendar_years_to_months(count)",
    ]
)
def lifetime_rules(tmp_path: Path, request):
    root = tmp_path / "rulespec-us"
    target = root / "us/statutes/example/lifetime.yaml"
    target.parent.mkdir(parents=True)
    payload = {
        "format": "rulespec/v1",
        "rules": [
            {"name": "count", "kind": "parameter", "versions": [{"formula": "2"}]},
            {
                "name": "net_value",
                "kind": "derived",
                "versions": [{"formula": "gross_value - adjustment"}],
            },
            {
                "name": "total",
                "kind": "derived",
                "versions": [{"formula": request.param}],
            },
        ],
    }
    target.write_text(yaml.safe_dump(payload))
    return root, target, request.param.split("(", 1)[0]


def test_context_describes_only_real_lifetime_inputs(lifetime_rules):
    _, target, _ = lifetime_rules
    assert evals._context_file_local_inputs(str(target)) == {
        "gross_value",
        "adjustment",
    }


def test_lifetime_reference_summary_excludes_reduction_names(lifetime_rules):
    _, target, _ = lifetime_rules
    summary = validator_pipeline._rulespec_reference_summary(target)
    assert summary.input_slots == {"gross_value", "adjustment"}
    assert summary.derived == {"net_value", "total"}
    assert summary.parameters == {"count"}


def test_companion_fixture_cannot_bind_a_lifetime_function_as_input(lifetime_rules):
    root, _, function = lifetime_rules
    options = {
        "label": "Test input",
        "policy_repo_path": root / "us",
        "allow_input_slots": True,
        "allow_relations": False,
        "allow_outputs": False,
    }
    issue = validator_pipeline._rulespec_absolute_test_reference_issue(
        f"us:statutes/example/lifetime#input.{function}", **options
    )
    assert issue is not None
    assert "does not resolve to an input slot" in issue
    assert (
        validator_pipeline._rulespec_absolute_test_reference_issue(
            "us:statutes/example/lifetime#input.gross_value", **options
        )
        is None
    )


def test_formula_analysis_preserves_lifetime_arguments_and_unknown_functions():
    assert validator_pipeline._formula_local_identifiers(
        "sum_top_n_over_periods(adjusted_value, selected_count) / "
        "calendar_years_to_months(selected_count)"
    ) == {"adjusted_value", "selected_count"}
    assert validator_pipeline._formula_local_identifiers(
        "sum_top_n_over_periods(adjusted_value, selected_count) + sum_over_periods(other_value)"
    ) == {"adjusted_value", "selected_count", "other_value"}
    assert validator_pipeline._formula_local_identifiers(
        "unrecognized_reduction(adjusted_value)"
    ) == {"unrecognized_reduction", "adjusted_value"}
