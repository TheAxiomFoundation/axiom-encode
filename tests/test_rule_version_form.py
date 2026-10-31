"""Reject synthetic encoder inputs that bypass version-based validation."""

from copy import deepcopy
from dataclasses import replace
from datetime import date
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode import cli
from axiom_encode.harness import evals
from axiom_encode.harness import source_completeness as completeness
from axiom_encode.harness import validator_pipeline as validation
from axiom_encode.rule_version_form import find_rule_version_form_issues
from tests.test_imported_export_evidence import world as world


def module_content(*, kind="derived", key="formula", versions=None):
    rule = {"name": "result", "kind": kind, key: "income + 100"}
    if versions is not None:
        rule["versions"] = versions
    return yaml.safe_dump({"format": "rulespec/v1", "rules": [rule]})


ISSUE = (
    "[rule-version-form] Rule `result` must put `formula` and its dates "
    "under `versions:`."
)


@pytest.mark.parametrize("kind", ["derived", "derived_relation"])
@pytest.mark.parametrize(
    "key", ["formula", "from", "to", "effective_from", "effective_to"]
)
@pytest.mark.parametrize(
    "versions", [None, [], [{"formula": True, "from": "2026-01-01"}]]
)
def test_rule_level_formula_or_dates_refuse_even_with_versions(kind, key, versions):
    assert find_rule_version_form_issues(
        module_content(kind=kind, key=key, versions=versions)
    ) == [ISSUE]


def test_version_level_aliases_and_parameter_shorthand_remain_accepted():
    content = yaml.safe_dump(
        {
            "rules": [
                {
                    "name": "result",
                    "kind": "derived",
                    "versions": [
                        {
                            "from": "2026-01-01",
                            "to": "2026-12-31",
                            "formula": "income + 100",
                        }
                    ],
                },
                {
                    "name": "threshold",
                    "kind": "parameter",
                    "formula": "100",
                    "from": "2026-01-01",
                },
            ]
        }
    )
    assert find_rule_version_form_issues(content) == []


@pytest.mark.parametrize(
    "method",
    ["_run_rulespec_compile_check", "_run_rulespec_ci", "_run_rulespec_ci_impl"],
)
def test_pipeline_and_rulespec_ci_refuse_before_compilation(tmp_path, method):
    module = tmp_path / "test.yaml"
    module.write_text(module_content())
    result = getattr(validation.ValidatorPipeline, method)(SimpleNamespace(), module)
    assert not result.passed
    assert result.issues == [ISSUE]
    assert result.error == ISSUE


def rejecting_pipeline():
    pipeline = validation.ValidatorPipeline.__new__(validation.ValidatorPipeline)
    pipeline.local_corpus_release = object.__new__(validation.LocalCorpusRelease)
    pipeline.rulespec_dependency_roots = ()
    pipeline.amendment_documents = ()
    pipeline.enable_oracles = False
    pipeline._log_event = lambda *args, **kwargs: None
    pipeline._run_compile_check = partial(
        validation.ValidatorPipeline._run_rulespec_compile_check, pipeline
    )
    pipeline._run_ci = partial(validation.ValidatorPipeline._run_rulespec_ci, pipeline)
    return pipeline


def test_pipeline_validate_refuses_flat_module(tmp_path):
    module = tmp_path / "test.yaml"
    module.write_text(module_content())
    result = rejecting_pipeline().validate(module, skip_reviewers=True)
    assert not result.all_passed
    assert not result.ci_pass
    assert result.results["ci"].issues == [ISSUE]


def test_validate_cli_reports_flat_module_failure(tmp_path):
    module = tmp_path / "test.yaml"
    module.write_text(module_content())
    passed, output = cli._validate_one(
        SimpleNamespace(skip_reviewers=True, json=True, oracle=False),
        rejecting_pipeline(),
        module,
        policyengine_runtime=None,
    )
    assert not passed
    assert not output["all_passed"]
    assert not output["ci_pass"]
    assert f"ci: {ISSUE}" in output["errors"]


def test_generated_apply_preflight_and_stamping_refuse_flat_module(tmp_path):
    module = tmp_path / "codex" / "test.yaml"
    module.parent.mkdir()
    content = module_content()
    module.write_text(content)
    result = SimpleNamespace(backend="codex", runner="codex", output_file=str(module))
    assert ISSUE in cli._generated_rulespec_yaml_nonparser_issue(
        result, module, output_root=tmp_path
    )
    with pytest.raises(RuntimeError, match="rule-version-form"):
        cli._stamp_generated_source_attestation_for_apply(result, module)
    assert module.read_text() == content


def test_generated_supplemental_repair_cannot_admit_flat_module():
    assert cli._supplemental_source_attestation_issues(
        {Path("policies/supplemental.yaml"): module_content()},
        attestation=None,
    ) == [f"policies/supplemental.yaml: {ISSUE}"]


@pytest.mark.parametrize("restored", [False, True])
def test_validation_retry_refuses_generated_and_restored_flat_rules(tmp_path, restored):
    module = tmp_path / "test.yaml"
    preserved = module_content() if restored else yaml.safe_dump({"rules": []})
    generated = yaml.safe_dump({"rules": []}) if restored else module_content()
    module.write_text(generated)
    with pytest.raises(ValueError, match="rule-version-form"):
        evals._overlay_validation_retry_candidate(
            module,
            artifact_root=tmp_path,
            candidate=evals.ValidationRetryCandidate(rulespec=preserved),
        )
    assert module.read_text() == generated


@pytest.mark.parametrize("kind", ["derived", "derived_relation"])
def test_input_assignment_selects_from_to_versions_for_test_period(kind):
    rule = {
        "name": "result",
        "kind": kind,
        "versions": [
            {"from": "2025-01-01", "to": "2025-12-31", "formula": "old_fact"},
            {"from": "2026-01-01", "to": "2026-12-31", "formula": "current_fact"},
        ],
    }
    content = yaml.safe_dump(
        {"module": {"proof_validation": {"required": True}}, "rules": [rule]}
    )
    cases = [
        {
            "name": "current",
            "period": {
                "period_kind": "month",
                "start": "2026-01-01",
                "end": "2026-01-31",
            },
            "input": {},
            "output": {"result": 1},
        }
    ]
    assert validation.find_test_input_assignment_issues(content, cases) == [
        "Test input assignment missing: `current` does not assign #input.current_fact. Every test for a proof-required RuleSpec module must set all local factual inputs, including false facts, so tests cannot pass through implicit defaults."
    ]


@pytest.mark.parametrize("kind", ["derived", "derived_relation"])
@pytest.mark.parametrize("native_date", [False, True])
def test_source_date_helpers_preserve_from_alias(kind, native_date):
    rule = {
        "kind": kind,
        "period": "Month",
        "versions": [
            {
                "from": date(2027, 2, 1) if native_date else "2027-02-01",
                "formula": "income",
            }
        ],
    }
    assert validation._single_version_effective_from(rule) == "2027-02-01"
    assert cli._generated_test_period_for_rule(rule) == {
        "period_kind": "month",
        "start": "2027-02-01",
        "end": "2027-02-28",
    }
    assert cli._rulespec_rule_effective_dates(rule) == ["2027-02-01"]
    assert cli._first_effective_from(rule) == "2027-02-01"
    assert cli._earliest_rule_effective_date(rule) == date(2027, 2, 1)


def test_monetary_formula_requires_versions_before_proof_selection():
    """Pin b6's monetary proof bypass at the admission gate."""
    payload = yaml.safe_load(module_content())
    payload["module"] = {"proof_validation": {"required": True}}
    payload["rules"][0].update(entity="Person", dtype="Money", unit="USD")
    assert find_rule_version_form_issues(yaml.safe_dump(payload)) == [ISSUE]


@pytest.mark.parametrize(
    "helper,formulas",
    [
        ("_collapse_additive_formula_versions", ["base", "base + increment"]),
        ("_collapse_selector_formula_versions", ["old_amount", "new_amount"]),
        (
            "_factor_conditional_rate_formula_versions",
            [
                "if eligible: amount * 0.1 else: 0",
                "if eligible: amount * (0.1 + supplement) else: 0",
            ],
        ),
    ],
)
def test_formula_version_repairs_preserve_from_alias(helper, formulas):
    canonical = [
        (
            index,
            {"effective_from": f"202{5 + index}-01-01", "formula": formula},
            formula,
        )
        for index, formula in enumerate(formulas)
    ]
    aliases = [
        (index, {"from": version["effective_from"], "formula": formula}, formula)
        for index, version, formula in canonical
    ]
    function = getattr(cli, helper)
    expected = function({"name": "result", "kind": "derived"}, canonical)
    assert expected is not None
    assert function({"name": "result", "kind": "derived"}, aliases) == expected


def test_embedded_scalar_repair_keeps_from_and_to_dates():
    result = cli._embedded_scalar_parameter_rule(
        {"kind": "derived"},
        name="amount",
        literal="100",
        dtype="Money",
        unit="USD",
        source_version={
            "from": "2027-01-01",
            "to": "2027-12-31",
            "formula": "income + 100",
        },
        source_atom={"kind": "formula", "path": "versions[0].formula"},
    )
    assert result["versions"] == [
        {"effective_from": "2027-01-01", "effective_to": "2027-12-31", "formula": "100"}
    ]


def test_existing_embedded_scalar_repair_compares_alias_dates():
    citation = "us/statute/example"
    target = "us:statutes/example#amount"
    source_atom = {
        "path": "versions[0].formula",
        "source": {"corpus_citation_path": citation},
    }
    parameter = {
        "name": "amount",
        "kind": "parameter",
        "dtype": "Money",
        "unit": "USD",
        "versions": [
            {
                "effective_from": "2027-01-01",
                "effective_to": "2027-12-31",
                "formula": "100",
            }
        ],
        "metadata": {"proof": {"atoms": [source_atom]}},
    }
    consumer = {
        "kind": "derived",
        "metadata": {
            "proof": {
                "atoms": [
                    source_atom,
                    {
                        "path": "versions[0].formula",
                        "import": {"target": target, "output": "amount"},
                    },
                ]
            }
        },
    }
    assert (
        cli._existing_embedded_scalar_parameter_name(
            [parameter],
            base_name="amount",
            literal="100",
            dtype="Money",
            unit="USD",
            target_version={"from": "2027-01-01", "to": "2027-12-31"},
            target_rule=consumer,
            target_version_index=0,
            expected_target=target,
            target_corpus_citation_path=citation,
        )
        == "amount"
    )


@pytest.mark.parametrize("existing", [None, {}, {"from": "2026-01-01"}])
def test_interval_bound_repair_preserves_selector_from_date(existing):
    previous = (
        None
        if existing is None
        else {"kind": "parameter", "indexed_by": "band", "versions": [existing]}
    )
    spec = validation._IntervalAlignmentSpec(
        selector_name="band",
        input_name="income",
        rows=(validation._IntervalSourceRow(lower="0", upper="100", outputs=("10",)),),
        lower_bound_parameter_name="lower_bound",
        upper_bound_parameter_name=None,
        lower_bound_parameter_rule=previous,
        upper_bound_parameter_rule=None,
        output_parameter_names=(),
        output_parameter_rules=(),
    )
    rules = [] if previous is None else [previous]
    name, changed = validation._ensure_interval_bound_parameter_rule(
        rules,
        spec=spec,
        selector_rule={"kind": "derived"},
        selector_version={"from": "2027-01-01"},
        side="lower",
    )
    assert name == "lower_bound"
    assert changed
    actual = rules[0]["versions"][0]
    assert actual.get("effective_from", actual.get("from")) == (
        "2026-01-01" if existing and "from" in existing else "2027-01-01"
    )


def date_aliases(payload, *, native_dates=False):
    """Use engine-supported aliases and keep proof anchors on actual fields."""
    result = deepcopy(payload)
    for rule in result.get("rules", []):
        for version in rule.get("versions", []):
            for canonical, alias in (
                ("effective_from", "from"),
                ("effective_to", "to"),
            ):
                if canonical in version:
                    raw = version.pop(canonical)
                    version[alias] = date.fromisoformat(raw) if native_dates else raw
        for atom in rule.get("metadata", {}).get("proof", {}).get("atoms", []):
            for canonical, alias in (
                ("effective_from", "from"),
                ("effective_to", "to"),
            ):
                atom["path"] = atom.get("path", "").replace(
                    "." + canonical, "." + alias
                )
    return result


@pytest.mark.parametrize("kind", ["derived", "derived_relation"])
@pytest.mark.parametrize("native_dates", [False, True])
def test_temporal_formula_selection_and_intervals_preserve_aliases(kind, native_dates):
    canonical = {
        "rules": [
            {
                "kind": kind,
                "versions": [
                    {
                        "effective_from": "2025-01-01",
                        "effective_to": "2025-12-31",
                        "formula": "10",
                    },
                    {
                        "effective_from": "2026-01-01",
                        "effective_to": "2026-12-31",
                        "formula": "20",
                    },
                ],
            }
        ]
    }
    aliased = date_aliases(canonical, native_dates=native_dates)["rules"][0]
    expected = canonical["rules"][0]
    assert (
        completeness._effective_formula_version_intervals(aliased)
        == completeness._effective_formula_version_intervals(expected)
        == (
            (0, "10", "2025-01-01", "2025-12-31"),
            (1, "20", "2026-01-01", "2026-12-31"),
        )
    )
    for period, formula, index in [
        ("2024-01-01", None, None),
        ("2025-06-01", "10", 0),
        ("2026-06-01", "20", 1),
        ("2027-01-01", None, None),
    ]:
        case = {"period": period}
        assert completeness._rule_formula_text_for_case(aliased, case) == formula
        assert completeness._selected_rule_formula_version_index(aliased, case) == index
    singleton = {"kind": kind, "versions": aliased["versions"][:1]}
    assert (
        completeness._rule_formula_text_for_case(singleton, {"period": "2026-01-01"})
        is None
    )
    assert (
        completeness._selected_rule_formula_version_index(
            singleton, {"period": "2026-01-01"}
        )
        is None
    )


@pytest.mark.parametrize("native_dates", [False, True])
def test_constant_and_rounding_environments_preserve_temporal_aliases(native_dates):
    canonical = {
        "rules": [
            {
                "name": "amount",
                "kind": "derived",
                "versions": [
                    {
                        "effective_from": "2025-01-01",
                        "effective_to": "2025-12-31",
                        "formula": "10",
                    }
                ],
            }
        ]
    }
    aliased = date_aliases(canonical, native_dates=native_dates)
    environment = completeness._constant_rule_environment(aliased)
    assert environment == completeness._constant_rule_environment(canonical)
    assert environment["amount"].versions == (("2025-01-01", "2025-12-31", 10),)
    for case, expected in [
        ({"input": {}}, {}),
        ({"period": "2024-06-01", "input": {}}, {}),
        ({"period": "2025-06-01", "input": {}}, {"amount": 10}),
    ]:
        rules = {"amount": aliased["rules"][0]}
        assert (
            completeness._closed_rounding_arithmetic_environment(rules, case)
            == expected
        )


@pytest.mark.parametrize("native_dates", [False, True])
def test_closed_imported_table_preserves_aliases_and_exact_evidence_shape(native_dates):
    from tests.test_source_completeness_fixed_table_import import _provider, _resolve

    canonical = _provider()
    expected = _resolve(canonical)
    assert "one_person_cost" in expected
    aliased = date_aliases(canonical, native_dates=native_dates)
    # This closed evidence route already rejects native date objects. Keep
    # that boundary while treating string aliases exactly like string dates.
    assert _resolve(aliased) == ({} if native_dates else expected)
    aliased["rules"][0]["versions"][0]["unknown"] = True
    assert _resolve(aliased) == {}


@pytest.mark.parametrize("native_dates", [False, True])
def test_annual_source_evidence_preserves_date_aliases(native_dates):
    from tests.test_source_owned_annual_years import fixture, witnesses

    rule, case = fixture()
    expected, covered, _ = witnesses(rule, case)
    assert expected and covered
    aliased = date_aliases({"rules": [rule]}, native_dates=native_dates)["rules"][0]
    actual, actual_covered, _ = witnesses(aliased, case)
    if native_dates:
        assert actual_covered == set()
    else:
        assert actual == expected
        assert actual_covered == covered


@pytest.fixture
def observation_world(tmp_path):
    return world.__wrapped__(tmp_path)


@pytest.mark.parametrize("native_dates", [False, True])
def test_entered_observation_evidence_preserves_date_aliases(
    observation_world, native_dates
):
    from tests.test_imported_export_evidence import bind

    expected = bind(observation_world)
    assert len(expected) == 1
    aliased = (
        date_aliases(observation_world[0], native_dates=native_dates),
        date_aliases(observation_world[1], native_dates=native_dates),
        *observation_world[2:],
    )
    actual = bind(aliased)
    if native_dates:
        assert actual == ()
    else:
        assert len(actual) == 1
        assert actual[0].provider_sha256 != expected[0].provider_sha256
        assert actual == (
            replace(expected[0], provider_sha256=actual[0].provider_sha256),
        )


@pytest.mark.parametrize("native_dates", [False, True])
def test_printed_operand_evidence_preserves_date_aliases_and_closed_shape(native_dates):
    from tests.test_printed_arithmetic_referral import bound_world

    _text, context, branch, rules, case = bound_world()
    execution = completeness._FormulaExecution((), "max(0,a-b)", None, False, {})
    assert context.local_operands_bind(
        branch, rule=rules["net"], principal_rules=rules, case=case, execution=execution
    )
    payload = date_aliases(
        yaml.safe_load(context.consumer_content), native_dates=native_dates
    )
    aliased_rules = {rule["name"]: rule for rule in payload["rules"]}
    aliased_context = replace(context, consumer_content=yaml.safe_dump(payload))
    assert aliased_context.local_operands_bind(
        branch,
        rule=aliased_rules["net"],
        principal_rules=aliased_rules,
        case=case,
        execution=execution,
    ) is (not native_dates)
    aliased_rules["net"]["versions"][0]["unknown"] = True
    invalid_context = replace(context, consumer_content=yaml.safe_dump(payload))
    assert not invalid_context.local_operands_bind(
        branch,
        rule=aliased_rules["net"],
        principal_rules=aliased_rules,
        case=case,
        execution=execution,
    )


@pytest.mark.parametrize("native_dates", [False, True])
def test_generated_day_repair_reads_from_date(tmp_path, native_dates):
    module = tmp_path / "example.yaml"
    case_file = tmp_path / "example.test.yaml"
    raw_date = date(2027, 2, 15) if native_dates else "2027-02-15"
    module.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        + yaml.safe_dump(
            {
                "rules": [
                    {
                        "name": "result",
                        "kind": "derived",
                        "versions": [{"from": raw_date, "formula": "income"}],
                    }
                ]
            }
        )
    )
    case_file.write_text(
        yaml.safe_dump(
            [
                {
                    "name": "example",
                    "period": "2027-02",
                    "output": {"us:policies/example#result": 1},
                }
            ]
        )
    )
    assert cli._rewrite_generated_day_period_test_shorthands(
        test_file=case_file,
        case_periods={"example": ""},
        rulespec_file=module,
        rulespec_module_path="policies/example",
    ) == ["example"]
    assert yaml.safe_load(case_file.read_text())[0]["period"] == {
        "period_kind": "custom",
        "name": "day",
        "start": "2027-02-15",
        "end": "2027-02-15",
    }


@pytest.mark.parametrize("native_dates", [False, True])
def test_temporal_clamp_claimant_contexts_preserve_aliases(monkeypatch, native_dates):
    from tests import test_source_completeness as fixtures

    cases = [
        {
            "name": "negative_before_future_consumer",
            "period": "2026",
            "input": {"completed_taxable_income": -1},
            "output": {"taxable_income_boundary": 0},
        },
        {
            "name": "positive_future_consumer",
            "period": "2027",
            "input": {"completed_taxable_income": 1},
            "output": {"taxable_income_boundary": 1, "future_taxable_income_report": 1},
        },
    ]
    expected = fixtures._al_owner_only_direct_input_clamp_analysis(
        cases, include_future_consumer=True, owner_open_ended=True
    ).issues
    assert any("below zero" in issue for issue in expected)
    original = fixtures._analyze

    def analyze_aliases(content, *args, **kwargs):
        aliased = date_aliases(yaml.safe_load(content), native_dates=native_dates)
        return original(yaml.safe_dump(aliased, sort_keys=False), *args, **kwargs)

    monkeypatch.setattr(fixtures, "_analyze", analyze_aliases)
    actual = fixtures._al_owner_only_direct_input_clamp_analysis(
        cases, include_future_consumer=True, owner_open_ended=True
    ).issues
    assert actual == expected


@pytest.mark.parametrize("alias_date", ["2025-10-01", "2026-01-01"])
def test_closed_table_evidence_refuses_duplicate_date_spellings(alias_date):
    from tests.test_source_completeness_fixed_table_import import _provider, _resolve

    provider = _provider()
    provider["rules"][0]["versions"][0]["from"] = alias_date
    assert _resolve(provider) == {}


@pytest.mark.parametrize("alias_date", ["2031-01-01", "2032-01-01"])
def test_frame_evidence_refuses_duplicate_date_spellings(observation_world, alias_date):
    from tests.test_imported_export_evidence import bind

    observation_world[1]["rules"][0]["versions"][0]["from"] = alias_date
    assert bind(observation_world) == ()


@pytest.mark.parametrize("native_dates", [False, True])
@pytest.mark.parametrize("effective", ["2026-01-01", "2027-04-01"])
def test_amendment_timing_uses_from_date(native_dates, effective):
    from axiom_encode.harness.amendment_timing import amendment_timing_issues
    from tests.test_amendment_timing import _content, _document

    canonical = yaml.safe_load(_content(effective))
    canonical["rules"][0]["kind"] = "derived"
    expected = amendment_timing_issues(yaml.safe_dump(canonical), [_document()])
    assert bool(expected) == (effective < "2027-04-01")
    aliased = date_aliases(canonical, native_dates=native_dates)
    assert amendment_timing_issues(yaml.safe_dump(aliased), [_document()]) == expected


@pytest.mark.parametrize("native_dates", [False, True])
def test_amendment_history_cutoff_reads_to_alias(native_dates):
    from axiom_encode.harness.amendment_timing import amendment_timing_issues
    from tests.test_amendment_timing import (
        AMENDMENT,
        PRINCIPAL,
        _content,
        _document,
        _proof,
    )

    canonical = yaml.safe_load(_content("2018-11-26", citation=PRINCIPAL))
    rule = canonical["rules"][0]
    rule["kind"] = "derived"
    rule["versions"][0]["effective_to"] = "2027-03-31"
    rule["metadata"]["proof"]["atoms"].extend(
        _proof(AMENDMENT, "versions[0].effective_to")["atoms"]
    )
    assert amendment_timing_issues(yaml.safe_dump(canonical), [_document()]) == []
    aliased = date_aliases(canonical, native_dates=native_dates)
    assert amendment_timing_issues(yaml.safe_dump(aliased), [_document()]) == []


def test_amendment_cutoff_alias_requires_actual_to_field():
    from axiom_encode.harness.amendment_timing import amendment_timing_issues
    from tests.test_amendment_timing import (
        AMENDMENT,
        PRINCIPAL,
        _content,
        _document,
        _proof,
    )

    canonical = yaml.safe_load(_content("2018-11-26", citation=PRINCIPAL))
    rule = canonical["rules"][0]
    rule["kind"] = "derived"
    rule["versions"][0]["effective_to"] = "2027-03-31"
    rule["metadata"]["proof"]["atoms"].extend(
        _proof(AMENDMENT, "versions[0].to")["atoms"]
    )
    issues = amendment_timing_issues(yaml.safe_dump(canonical), [_document()])
    assert len(issues) == 1
    assert "predates commencement" in issues[0]


@pytest.mark.parametrize("alias_date", ["2025-01-01", "2026-01-01"])
def test_annual_year_evidence_refuses_duplicate_date_spellings(alias_date):
    from tests.test_source_owned_annual_years import fixture, witnesses

    rule, case = fixture()
    expected, covered, _ = witnesses(rule, case)
    assert expected and covered
    rule["versions"][0]["from"] = alias_date
    _actual, actual_covered, _ = witnesses(rule, case)
    assert actual_covered == set()
