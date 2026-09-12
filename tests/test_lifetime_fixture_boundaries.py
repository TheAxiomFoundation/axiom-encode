"""Typed lifetime facts stay distinct from scalar repairs and oracle scenarios."""

import copy
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.harness import evals, validator_pipeline
from axiom_encode.prompts.encoder import ENCODER_PROMPT, LIFETIME_FIXTURE_PROTOCOL

PREFIX = "us:statutes/example/lifetime"
RULES = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: total
    kind: derived
    entity: Person
    period: Year
    versions:
      - effective_from: '2001-01-01'
        formula: sum_over_periods(earnings)
"""


def period(year):
    return {
        "period_kind": "tax_year",
        "start": f"{year}-01-01",
        "end": f"{year}-12-31",
    }


@pytest.fixture
def history_case():
    periods = [period(1998), period(1999)]
    return {
        "name": "alternate_effective_date_boundary",
        "period": periods[-1],
        "output": {f"{PREFIX}#total": "1200.00"},
        "lifetime": {
            "entity": "Person",
            "periods": periods,
            "batches": [
                {
                    "row_count": 1,
                    "entity_ids": ["worker-1,234"],
                    "inputs": {
                        f"{PREFIX}#input.earnings": {
                            "kind": "decimal",
                            "values": ["0600.00"],
                        }
                    },
                }
                for _ in periods
            ],
        },
    }


def test_history_reference_collectors_keep_public_keys_not_column_metadata(
    history_case,
):
    history_case["lifetime"]["batches"][1]["inputs"][f"{PREFIX}#input.person_id"] = {
        "kind": "text",
        "values": ["opaque-id"],
    }
    snapshot = copy.deepcopy(history_case)
    assert validator_pipeline._all_test_input_names([history_case]) == {
        "earnings",
        "person_id",
    }
    assert validator_pipeline._test_case_assignment_keys(history_case) == [
        f"{PREFIX}#input.earnings",
        f"{PREFIX}#input.earnings",
        f"{PREFIX}#input.person_id",
    ]
    assert history_case == snapshot


def test_history_fact_in_one_period_cannot_satisfy_another(history_case):
    history_case["lifetime"]["batches"][0]["inputs"] = {}
    issues = validator_pipeline.find_test_input_assignment_issues(RULES, [history_case])
    assert len(issues) == 1
    assert "lifetime batch #1" in issues[0]
    assert "#input.earnings" in issues[0]


def test_complete_history_assignments_pass_without_scalar_inputs(history_case):
    assert (
        validator_pipeline.find_test_input_assignment_issues(RULES, [history_case])
        == []
    )


def test_scalar_input_assignment_behavior_is_preserved():
    scalar = {
        "name": "ordinary",
        "period": period(2001),
        "input": {f"{PREFIX}#input.earnings": 7},
        "output": {f"{PREFIX}#total": 7},
    }
    assert validator_pipeline.find_test_input_assignment_issues(RULES, [scalar]) == []
    scalar["input"] = {}
    assert validator_pipeline.find_test_input_assignment_issues(RULES, [scalar])


@pytest.mark.parametrize(
    "normalizer",
    [
        evals._normalize_single_amount_row_test_content,
        evals._normalize_test_periods_to_effective_dates,
    ],
)
@pytest.mark.parametrize("wrapper", ["list", "cases", "tests"])
def test_normalizers_preserve_history_and_still_repair_scalar_cases(
    history_case, normalizer, wrapper
):
    scalar = {
        "name": "ordinary",
        "period": 2001,
        "input": {f"{PREFIX}#input.earnings": "12"},
        "output": {f"{PREFIX}#total": "12"},
    }
    cases = [history_case, scalar]
    payload = cases if wrapper == "list" else {wrapper: cases}
    normalized = yaml.safe_load(normalizer(yaml.safe_dump(payload), RULES))
    assert normalized[0] == history_case
    assert normalized[1]["input"][f"{PREFIX}#input.earnings"] == 12
    assert normalized[1]["output"][f"{PREFIX}#total"] == 12


@pytest.mark.parametrize(
    "normalizer",
    [
        evals._normalize_single_amount_row_test_content,
        evals._normalize_test_periods_to_effective_dates,
    ],
)
def test_invalid_lifetime_is_preserved_for_explicit_validation(normalizer):
    case = {
        "name": "alternate_effective_date_boundary",
        "period": 1998,
        "lifetime": None,
        "output": {f"{PREFIX}#total": "1,234"},
    }
    assert yaml.safe_load(normalizer(yaml.safe_dump([case]), RULES)) == [case]


@pytest.mark.parametrize(
    "normalizer",
    [
        evals._normalize_single_amount_row_test_content,
        evals._normalize_test_periods_to_effective_dates,
    ],
)
@pytest.mark.parametrize("failure", ["parse", "length", "mode"])
def test_unmatched_history_normalization_preserves_original_text(
    monkeypatch, history_case, normalizer, failure
):
    original = (
        "[invalid YAML"
        if failure == "parse"
        else yaml.safe_dump(
            [history_case] if failure == "length" else [{"name": "scalar"}]
        )
    )
    replacement = [history_case]
    if failure == "length":
        replacement.append({"name": "extra"})
    monkeypatch.setattr(
        evals,
        "_normalize_comma_numeric_literals",
        lambda _content: yaml.safe_dump(replacement),
    )
    assert normalizer(original, RULES) == original


def test_tests_only_repair_appends_only_the_contracted_history(tmp_path, history_case):
    for batch in history_case["lifetime"]["batches"]:
        batch["inputs"][f"{PREFIX}#input.earnings"]["values"] = ["600.00"]
    old_history = copy.deepcopy(history_case)
    old_history["name"] = "preserved-history"
    scalar = {
        "name": "preserved-scalar",
        "period": period(2001),
        "input": {"fact": True},
        "output": {"result": 1},
    }
    original = [scalar, old_history]
    contract = {
        key: copy.deepcopy(value)
        for key, value in history_case.items()
        if key != "output"
    }
    contract["required_output"] = copy.deepcopy(history_case["output"])
    contract["description"] = "Required synthetic lifetime witness."
    proposed = [*original, {**history_case, "description": contract["description"]}]
    original_text = yaml.safe_dump(original)
    proposed_text = yaml.safe_dump(proposed)
    assert "lifetime" in evals._format_required_test_case_contracts([contract])
    candidate = evals.ValidationRetryCandidate(RULES, original_text)
    output_file = tmp_path / "history.yaml"
    assert evals._materialize_eval_artifact(
        f"=== FILE: history.test.yaml ===\n{proposed_text}",
        output_file,
        artifact_root=tmp_path,
        repair_candidate=candidate,
        required_test_case_contracts=[contract],
    )
    assert output_file.read_text() == RULES
    assert yaml.safe_load(output_file.with_suffix(".test.yaml").read_text()) == proposed

    for target in (1, 2):
        for mutate in (
            lambda case: case["lifetime"]["batches"][0]["entity_ids"].__setitem__(
                0, "other-person"
            ),
            lambda case: case["lifetime"]["batches"].reverse(),
            lambda case: case["output"].__setitem__(f"{PREFIX}#total", 1200),
        ):
            changed = copy.deepcopy(proposed)
            mutate(changed[target])
            # Distinguish the two otherwise equal batches for the order check.
            if changed == proposed:
                changed[target]["lifetime"]["periods"].reverse()
            assert not evals._preserves_companion_test_cases(
                original_text, yaml.safe_dump(changed), [contract]
            )
    malformed_contract = copy.deepcopy(contract)
    malformed_contract["lifetime"]["batches"].pop()
    assert not evals._preserves_companion_test_cases(
        original_text, proposed_text, [malformed_contract]
    )


def test_mixed_oracle_scores_only_scalar_comparisons_and_marks_history_unsupported(
    tmp_path, monkeypatch, history_case
):
    rule_file = tmp_path / "rules.yaml"
    rule_file.write_text(RULES)
    scalar = {
        "name": "scalar",
        "period": 2001,
        "input": {"fact": 7},
        "output": {"scalar_result": 7},
    }
    rule_file.with_suffix(".test.yaml").write_text(
        yaml.safe_dump([scalar, history_case])
    )
    pipeline = validator_pipeline.ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=tmp_path / "engine",
        local_corpus_release=None,
        enable_oracles=False,
    )
    mapping = SimpleNamespace(
        comparable=True,
        policyengine_variable="synthetic_scalar",
        policyengine_parameter=None,
        result_multiplier=None,
    )
    mapped_refs = []

    def resolve(_country, reference):
        mapped_refs.append(reference)
        return mapping

    monkeypatch.setattr(pipeline, "_resolve_pe_mapping", resolve)
    monkeypatch.setattr(pipeline, "_should_compare_pe_test_output", lambda *_: True)
    monkeypatch.setattr(pipeline, "_is_pe_test_mappable", lambda *_, **__: (True, None))
    monkeypatch.setattr(pipeline, "_build_pe_scenario_script", lambda *_, **__: "")
    monkeypatch.setattr(
        pipeline,
        "_run_pe_subprocess_detailed",
        lambda _: validator_pipeline.OracleSubprocessResult(
            returncode=0, stdout="RESULT:7\n"
        ),
    )
    result = pipeline._run_policyengine_bound(rule_file, "uk")
    assert result.passed is True and result.score == 1.0
    assert mapped_refs == ["scalar_result"]
    assert result.details["coverage"]["total_outputs"] == 2
    assert result.details["coverage"]["comparable"] == 1
    assert result.details["coverage"]["unsupported"] == 1
    assert len(result.issues) == 1
    assert "history-aware oracle adapter" in result.issues[0]


def test_oracle_extraction_preserves_lifetime_mode_without_scalar_projection(
    tmp_path, history_case
):
    pipeline = validator_pipeline.ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=tmp_path / "engine",
        local_corpus_release=None,
        enable_oracles=False,
    )
    tests = pipeline._extract_rulespec_tests(yaml.safe_dump([history_case]))
    assert len(tests) == 1
    assert tests[0]["lifetime_fixture"] is True
    assert tests[0]["expect"] == "1200.00"
    assert "inputs" not in tests[0]


def test_scalar_oracle_refuses_lifetime_before_any_mapping_or_execution(
    tmp_path, monkeypatch, history_case
):
    rule_file = tmp_path / "rules.yaml"
    rule_file.write_text(RULES)
    rule_file.with_suffix(".test.yaml").write_text(yaml.safe_dump([history_case]))
    pipeline = validator_pipeline.ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=tmp_path / "engine",
        local_corpus_release=None,
        enable_oracles=False,
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("lifetime history reached a scalar oracle path")

    for name in (
        "_should_compare_pe_test_output",
        "_resolve_pe_mapping",
        "_build_pe_scenario_script",
        "_run_pe_subprocess_detailed",
    ):
        monkeypatch.setattr(pipeline, name, forbidden)
    result = pipeline._run_policyengine_bound(rule_file, "us")
    assert result.passed is False
    assert result.score is None
    assert result.details["coverage"]["unsupported"] == 1
    assert result.details["coverage"]["comparable"] == 0
    assert "history-aware oracle adapter" in result.issues[0]


def test_lifetime_guidance_reaches_both_generation_prompts(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text(
        "Synthetic statute: sum the person's values over the supplied years."
    )
    workspace = evals.EvalWorkspace(
        root=tmp_path,
        source_text_file=source,
        manifest_file=tmp_path / "manifest.json",
    )
    prompt = evals._build_rulespec_eval_prompt(
        citation="us/statute/example/lifetime",
        mode="cold",
        workspace=workspace,
        context_files=[],
        target_file_name="lifetime.yaml",
        target_ref_prefix=PREFIX,
        include_tests=True,
        runner_backend="codex",
        policyengine_rule_hint=None,
    )
    assert LIFETIME_FIXTURE_PROTOCOL in prompt
    assert LIFETIME_FIXTURE_PROTOCOL in ENCODER_PROMPT
    assert "#input.<builtin_name>" in prompt
    assert "row_count" in prompt and "entity_ids" in prompt
