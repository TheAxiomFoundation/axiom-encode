"""Focused tests for structured companion-test apply contracts."""

import argparse
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.cli import (
    _parse_deferred_output_review_contract_json,
    _required_deferred_output_contract_issues,
)
from axiom_encode.harness.evals import _format_required_test_case_contracts
from axiom_encode.harness.validator_pipeline import ValidatorPipeline

CITATION = "us-la/statute/47:294"
RULESPEC_PATH = "us-la/statutes/47/294.yaml"
CREATE_RULESPEC_PATH = "us-la/policies/income_tax/new_rule.yaml"
CREATE_MODULE = "us-la:policies/income_tax/new_rule"
CREATE_OUTPUT = f"{CREATE_MODULE}#combined_qbi_niit_amount"
ACTIVITY_RELATION = f"{CREATE_MODULE}#relation.activity_of_owner"
SINGLE = (
    "us-la:statutes/47/294#input."
    "federal_return_filing_status_is_single_or_married_separate"
)
JOINT = (
    "us-la:statutes/47/294#input."
    "federal_return_filing_status_is_joint_surviving_spouse_or_head_of_household"
)
PRINCIPAL = "us-la:statutes/47/294#standard_deduction"
CPI = "us-la:statutes/47/294#input.previous_calendar_year_cpi_u_percentage_increase"
PRIOR_JOINT = (
    "us-la:statutes/47/294#input."
    "prior_year_standard_deduction_joint_surviving_spouse_or_head_of_household"
)
PRIOR_SINGLE = (
    "us-la:statutes/47/294#input."
    "prior_year_standard_deduction_single_or_married_separate"
)
CASE = {
    "name": "2025 single individual or married separate standard deduction",
    "period": {
        "period_kind": "tax_year",
        "start": "2025-01-01",
        "end": "2025-12-31",
    },
    "input": {SINGLE: True, JOINT: False},
    "required_output": {PRINCIPAL: 12500},
}


def _raw_contract(*, test_cases: list[dict[str, object]] | None = None) -> str:
    return json.dumps(
        {
            "schema": "axiom-encode/review-contract/v2",
            "citation": CITATION,
            "rulespec_path": RULESPEC_PATH,
            "required_deferred_outputs": [],
            "required_test_cases": test_cases if test_cases is not None else [CASE],
        },
        separators=(",", ":"),
    )


def _raw_v3_contract(
    *,
    operation: str | None = "replace",
    test_cases: list[dict[str, object]] | None = None,
    rulespec_path: str = RULESPEC_PATH,
) -> str:
    payload: dict[str, object] = {
        "schema": "axiom-encode/review-contract/v3",
        "citation": CITATION,
        "rulespec_path": rulespec_path,
        "required_deferred_outputs": [],
        "required_test_cases": test_cases if test_cases is not None else [],
    }
    if operation is not None:
        payload["target_operation"] = operation
    return json.dumps(payload, separators=(",", ":"))


def _write_candidate(tmp_path: Path, cases: list[dict[str, object]]) -> Path:
    rulespec = tmp_path / "294.yaml"
    rulespec.write_text("format: rulespec/v1\nmodule: {}\nrules: []\n")
    rulespec.with_name("294.test.yaml").write_text(
        json.dumps(cases),
        encoding="utf-8",
    )
    return rulespec


def _issues(
    rulespec: Path,
    raw: str | None = None,
    *,
    rulespec_path: str = RULESPEC_PATH,
) -> list[str]:
    contract = _parse_deferred_output_review_contract_json(raw or _raw_contract())
    return _required_deferred_output_contract_issues(
        rulespec,
        contract,
        citation=CITATION,
        rulespec_path=rulespec_path,
    )


def test_required_test_case_accepts_exact_input_and_required_output_subset(
    tmp_path: Path,
) -> None:
    candidate = {
        "name": CASE["name"],
        "period": CASE["period"],
        "input": CASE["input"],
        "output": {PRINCIPAL: 12500, "us-la:statutes/47/294#helper": 12500},
    }

    assert _issues(_write_candidate(tmp_path, [candidate])) == []


def test_required_test_case_rejects_pr_1253_extra_2025_inputs(tmp_path: Path) -> None:
    candidate = {
        "name": CASE["name"],
        "period": CASE["period"],
        "input": {
            **CASE["input"],
            CPI: 0,
            PRIOR_JOINT: 0,
            PRIOR_SINGLE: 0,
        },
        "output": {PRINCIPAL: 12500},
    }

    issues = _issues(_write_candidate(tmp_path, [candidate]))

    assert len(issues) == 1
    assert "input map does not exactly match" in issues[0]
    assert "unexpected:" in issues[0]


def test_required_test_case_contract_covers_all_four_294_cases(tmp_path: Path) -> None:
    contracts = []
    candidates = []
    for name, single, joint, expected in (
        (
            "2025 single individual or married separate standard deduction",
            True,
            False,
            12500,
        ),
        (
            "2025 joint surviving spouse or head of household standard deduction",
            False,
            True,
            25000,
        ),
        ("2025 no listed filing status group fails closed", False, False, 0),
        ("2025 conflicting filing status groups fail closed", True, True, 0),
    ):
        contract = {
            "name": name,
            "period": CASE["period"],
            "input": {SINGLE: single, JOINT: joint},
            "required_output": {PRINCIPAL: expected},
        }
        contracts.append(contract)
        candidates.append(
            {
                "name": name,
                "period": CASE["period"],
                "input": {SINGLE: single, JOINT: joint},
                "output": {PRINCIPAL: expected},
            }
        )

    assert (
        _issues(
            _write_candidate(tmp_path, candidates),
            _raw_contract(test_cases=contracts),
        )
        == []
    )

    for candidate in candidates:
        candidate["input"] = {
            **candidate["input"],
            CPI: 0,
            PRIOR_JOINT: 0,
            PRIOR_SINGLE: 0,
        }
    rejected = _issues(
        _write_candidate(tmp_path, candidates),
        _raw_contract(test_cases=contracts),
    )
    assert len(rejected) == 4
    assert all(
        all(key in issue for key in (CPI, PRIOR_JOINT, PRIOR_SINGLE))
        and "unexpected:" in issue
        for issue in rejected
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ({"period": {**CASE["period"], "end": "2026-12-31"}}, "period"),
        ({"output": {PRINCIPAL: 0}}, "missing or changes required output"),
    ],
)
def test_required_test_case_rejects_period_or_output_drift(
    tmp_path: Path,
    mutation: dict[str, object],
    expected: str,
) -> None:
    candidate = {
        "name": CASE["name"],
        "period": CASE["period"],
        "input": CASE["input"],
        "output": {PRINCIPAL: 12500},
        **mutation,
    }

    assert expected in "\n".join(_issues(_write_candidate(tmp_path, [candidate])))


@pytest.mark.parametrize("count", [0, 2])
def test_required_test_case_requires_exactly_one_named_case(
    tmp_path: Path,
    count: int,
) -> None:
    candidate = {
        "name": CASE["name"],
        "period": CASE["period"],
        "input": CASE["input"],
        "output": {PRINCIPAL: 12500},
    }

    issues = _issues(_write_candidate(tmp_path, [candidate] * count))

    assert len(issues) == 1
    assert f"found {count}" in issues[0]


def test_v2_review_contract_rejects_empty_contract() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="must require"):
        _parse_deferred_output_review_contract_json(_raw_contract(test_cases=[]))


def test_v3_review_contract_binds_replace_target_operation() -> None:
    parsed = _parse_deferred_output_review_contract_json(
        _raw_v3_contract(operation="replace")
    )

    assert parsed.target_operation == "replace"
    assert parsed.required_test_cases == ()


def test_v3_create_contract_requires_each_case_to_witness_created_module() -> None:
    case = {
        **CASE,
        "required_output": {
            f"{CREATE_MODULE}#amount": 100,
            PRINCIPAL: 12500,
        },
    }

    parsed = _parse_deferred_output_review_contract_json(
        _raw_v3_contract(
            operation="create",
            test_cases=[case],
            rulespec_path=CREATE_RULESPEC_PATH,
        )
    )

    assert parsed.target_operation == "create"
    assert parsed.required_test_cases[0].required_output == case["required_output"]


@pytest.mark.parametrize(
    "required_output",
    [
        {PRINCIPAL: 12500},
        {"us-la:policies/income_tax/new_rule_extra#amount": 100},
        {"us-la:policies/income_tax/new_rule": 100},
    ],
)
def test_v3_create_contract_rejects_imported_or_lookalike_only_witnesses(
    required_output: dict[str, object],
) -> None:
    case = {**CASE, "required_output": required_output}

    with pytest.raises(argparse.ArgumentTypeError, match="exact created module"):
        _parse_deferred_output_review_contract_json(
            _raw_v3_contract(
                operation="create",
                test_cases=[case],
                rulespec_path=CREATE_RULESPEC_PATH,
            )
        )


def test_v3_create_contract_rejects_empty_witness_set() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="must require a test case"):
        _parse_deferred_output_review_contract_json(
            _raw_v3_contract(
                operation="create",
                test_cases=[],
                rulespec_path=CREATE_RULESPEC_PATH,
            )
        )


def _relation_case(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "name": "one owner and one S corporation activity",
        "period": {
            "period_kind": "tax_year",
            "start": "2026-01-01",
            "end": "2026-12-31",
        },
        "input": {ACTIVITY_RELATION: rows},
        "required_output": {CREATE_OUTPUT: 100},
    }


def _activity_row(activity_id: str = "activity-1") -> dict[str, object]:
    return {
        "activity_id": activity_id,
        "owner_id": "owner-1",
        "entity_form": "s_corporation",
        "is_sstb": False,
        "material_participation": True,
        "ordinary_income": 100,
        "w2_wages": 20,
        "ubia": 50,
    }


def test_v3_create_relation_witness_parses_and_matches_real_companion_case(
    tmp_path: Path,
) -> None:
    required = _relation_case([_activity_row()])
    raw = _raw_v3_contract(
        operation="create",
        test_cases=[required],
        rulespec_path=CREATE_RULESPEC_PATH,
    )
    parsed = _parse_deferred_output_review_contract_json(raw)
    candidate = {
        "name": required["name"],
        "period": required["period"],
        "input": required["input"],
        "output": {CREATE_OUTPUT: 100},
    }

    assert parsed.required_test_cases[0].input == required["input"]
    assert (
        _issues(
            _write_candidate(tmp_path, [candidate]),
            raw,
            rulespec_path=CREATE_RULESPEC_PATH,
        )
        == []
    )


@pytest.mark.parametrize("mutation", ["bool-to-int", "extra-row-field", "container"])
def test_relation_witness_rejects_nested_type_shape_or_field_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    required = _relation_case([_activity_row()])
    candidate_input = json.loads(json.dumps(required["input"]))
    rows = candidate_input[ACTIVITY_RELATION]
    if mutation == "bool-to-int":
        rows[0]["material_participation"] = 1
    elif mutation == "extra-row-field":
        rows[0]["unreviewed_fact"] = True
    else:
        candidate_input[ACTIVITY_RELATION] = {"activity-1": rows[0]}
    candidate = {
        "name": required["name"],
        "period": required["period"],
        "input": candidate_input,
        "output": {CREATE_OUTPUT: 100},
    }
    raw = _raw_v3_contract(
        operation="create",
        test_cases=[required],
        rulespec_path=CREATE_RULESPEC_PATH,
    )

    issues = _issues(
        _write_candidate(tmp_path, [candidate]),
        raw,
        rulespec_path=CREATE_RULESPEC_PATH,
    )

    assert len(issues) == 1
    assert "input map does not exactly match" in issues[0]
    assert "changed:" in issues[0]


def test_relation_witness_preserves_list_order(tmp_path: Path) -> None:
    rows = [_activity_row("activity-1"), _activity_row("activity-2")]
    required = _relation_case(rows)
    candidate = {
        "name": required["name"],
        "period": required["period"],
        "input": {ACTIVITY_RELATION: list(reversed(rows))},
        "output": {CREATE_OUTPUT: 100},
    }
    raw = _raw_v3_contract(
        operation="create",
        test_cases=[required],
        rulespec_path=CREATE_RULESPEC_PATH,
    )

    issues = _issues(
        _write_candidate(tmp_path, [candidate]),
        raw,
        rulespec_path=CREATE_RULESPEC_PATH,
    )

    assert len(issues) == 1
    assert "input map does not exactly match" in issues[0]


def test_relation_witness_rejects_recursive_yaml_anchor_without_crashing(
    tmp_path: Path,
) -> None:
    required = _relation_case([_activity_row()])
    raw = _raw_v3_contract(
        operation="create",
        test_cases=[required],
        rulespec_path=CREATE_RULESPEC_PATH,
    )
    rulespec = tmp_path / "new_rule.yaml"
    rulespec.write_text("format: rulespec/v1\nmodule: {}\nrules: []\n")
    rulespec.with_name("new_rule.test.yaml").write_text(
        "- name: one owner and one S corporation activity\n"
        "  period:\n"
        "    period_kind: tax_year\n"
        "    start: '2026-01-01'\n"
        "    end: '2026-12-31'\n"
        "  input:\n"
        f"    '{ACTIVITY_RELATION}': &rows\n"
        "      - *rows\n"
        "  output:\n"
        f"    '{CREATE_OUTPUT}': 100\n",
        encoding="utf-8",
    )

    issues = _issues(rulespec, raw, rulespec_path=CREATE_RULESPEC_PATH)

    assert len(issues) == 1
    assert "invalid bounded JSON value" in issues[0]
    assert "cyclic" in issues[0]


def test_validator_pipeline_rejects_mutual_relation_cycle_before_recursive_walkers(
    tmp_path: Path,
) -> None:
    cases = yaml.safe_load(
        "- name: mutual relation cycle\n"
        "  period: 2026\n"
        "  input:\n"
        "    relation.activity_of_owner: &rows\n"
        "      - activity_id: activity-1\n"
        "        back: *rows\n"
        "  output:\n"
        "    amount: 0\n"
    )
    assert cases[0]["input"]["relation.activity_of_owner"] is cases[0]["input"][
        "relation.activity_of_owner"
    ][0]["back"]
    pipeline = ValidatorPipeline(
        tmp_path,
        tmp_path,
        local_corpus_release=None,
        enable_oracles=False,
    )

    issues = pipeline._run_rulespec_test_cases(
        rules_file=tmp_path / "cycle.yaml",
        compiled_path=tmp_path / "compiled.json",
        compiled_payload={},
        cases=cases,
    )

    assert issues == [
        "RuleSpec test YAML structure is invalid: cyclic alias graph."
    ]


@pytest.mark.parametrize("operation", [None, "replace ", "upgrade"])
def test_v3_review_contract_rejects_missing_or_invalid_target_operation(
    operation: str | None,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="v3 review contract"):
        _parse_deferred_output_review_contract_json(
            _raw_v3_contract(operation=operation)
        )


def test_v3_review_contract_rejects_unrecognized_extra_field() -> None:
    raw = json.loads(_raw_v3_contract())
    raw["unexpected"] = True

    with pytest.raises(argparse.ArgumentTypeError, match="v3 review contract"):
        _parse_deferred_output_review_contract_json(json.dumps(raw))


def test_v1_and_v2_review_contracts_do_not_fabricate_target_operation() -> None:
    v1 = json.dumps(
        {
            "schema": "axiom-encode/review-contract/v1",
            "citation": CITATION,
            "rulespec_path": RULESPEC_PATH,
            "required_deferred_outputs": [{"output": PRINCIPAL, "reason": "x"}],
        }
    )

    assert _parse_deferred_output_review_contract_json(v1).target_operation is None
    assert (
        _parse_deferred_output_review_contract_json(_raw_contract()).target_operation
        is None
    )


@pytest.mark.parametrize("period_kind", ["month", "benefit_week"])
def test_v2_review_contract_accepts_engine_period_kinds(period_kind: str) -> None:
    case = {
        **CASE,
        "period": {
            "period_kind": period_kind,
            "start": "2025-01-01",
            "end": "2025-01-31",
        },
    }

    parsed = _parse_deferred_output_review_contract_json(
        _raw_contract(test_cases=[case])
    )
    assert parsed.required_test_cases[0].period["period_kind"] == period_kind


def test_v2_review_contract_rejects_duplicate_nested_json_key() -> None:
    raw = _raw_contract().replace(
        f'"{SINGLE}":true',
        f'"{SINGLE}":true,"{SINGLE}":false',
    )

    with pytest.raises(argparse.ArgumentTypeError, match="duplicate JSON key"):
        _parse_deferred_output_review_contract_json(raw)


def test_v2_review_contract_rejects_oversized_integer_token() -> None:
    raw = _raw_contract().replace("12500", "9" * 5000)

    with pytest.raises(argparse.ArgumentTypeError, match="valid JSON"):
        _parse_deferred_output_review_contract_json(raw)


def test_required_test_case_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    rulespec = tmp_path / "294.yaml"
    rulespec.write_text("format: rulespec/v1\nmodule: {}\nrules: []\n")
    rulespec.with_name("294.test.yaml").write_text(
        "- name: " + str(CASE["name"]) + "\n"
        "  period:\n"
        "    period_kind: tax_year\n"
        "    start: '2025-01-01'\n"
        "    end: '2025-12-31'\n"
        "  input:\n"
        f"    {SINGLE}: true\n"
        f"    {SINGLE}: false\n"
        f"    {JOINT}: false\n"
        "  output:\n"
        f"    {PRINCIPAL}: 12500\n",
        encoding="utf-8",
    )

    issues = _issues(rulespec)
    assert len(issues) == 1
    assert "duplicate key" in issues[0]


def test_required_test_case_rejects_unsigned_top_level_tables(tmp_path: Path) -> None:
    candidate = {
        "name": CASE["name"],
        "period": CASE["period"],
        "input": CASE["input"],
        "tables": {
            "TaxUnit": [
                {
                    "id": "tax-unit-1",
                    "us-la:statutes/47/294#input.prior_year_deduction": 999,
                }
            ]
        },
        "output": {PRINCIPAL: 12500},
    }

    issues = _issues(_write_candidate(tmp_path, [candidate]))
    assert len(issues) == 1
    assert "unsigned top-level field(s): tables" in issues[0]


def test_required_test_case_prompt_preserves_exact_contract() -> None:
    rendered = _format_required_test_case_contracts([CASE])

    assert json.dumps(CASE, separators=(",", ":"), sort_keys=True) in rendered
    assert "complete `input` map" in rendered
    assert "final repaired overlay" in rendered
