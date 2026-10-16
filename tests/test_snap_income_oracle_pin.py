from pathlib import Path

import yaml
from axiom_oracles.bridges import registry as runtime_registry

PREFIX = "us:policies/usda/snap/fy-2026-cola/income-eligibility-standards#"
OUTPUTS = {
    "snap_net_income_limit_100_percent_fpl_48_states_dc_table",
    "snap_net_income_limit_100_percent_fpl_48_states_dc",
    "snap_gross_income_limit_130_percent_fpl_48_states_dc_table",
    "snap_gross_income_limit_130_percent_fpl_48_states_dc",
    "snap_gross_income_limit_165_percent_fpl_48_states_dc_table",
    "snap_gross_income_limit_165_percent_fpl_48_states_dc",
}


def _income_mappings(path: Path) -> dict[str, dict]:
    document = yaml.safe_load(path.read_text())
    return {
        item["legal_id"].removeprefix(PREFIX): item
        for item in document["mappings"]
        if item.get("legal_id", "").startswith(PREFIX)
    }


def test_fy2026_income_limits_are_not_falsely_counted_as_policyengine_parity() -> None:
    root = Path(__file__).parents[1]
    bundled = _income_mappings(
        root / "src/axiom_encode/oracles/policyengine/mappings/us.yaml"
    )
    runtime = _income_mappings(
        Path(runtime_registry.__file__).with_name("mappings") / "us.yaml"
    )

    assert set(bundled) == OUTPUTS
    assert bundled == runtime
    for mapping in bundled.values():
        assert mapping["mapping_type"] == "not_comparable"
        assert mapping["candidate_priority"] == "P4"
        assert "policyengine_variable" not in mapping
        assert "policyengine_parameter" not in mapping
        assert "round" in mapping["rationale"]
