"""Exercise the actual #1707 deferral artifact through the validator."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import validator_pipeline as vp

FIXTURE = Path(__file__).parent / "fixtures/source_completeness/irs_rev_proc_2025_32"
CITATION = "us/guidance/irs/rev-proc-2025-32/page-14"


def _candidate():
    folder = FIXTURE / "precise_deferral"
    return (
        yaml.safe_load((folder / "child-tax-credit.yaml").read_text()),
        yaml.safe_load((folder / "child-tax-credit.test.yaml").read_text()),
        json.loads((FIXTURE / "page-14.json").read_text())["body"],
    )


@pytest.fixture
def evaluate(tmp_path, monkeypatch):
    payload, cases, source = _candidate()
    targets = {
        target
        for record in payload["module"]["deferred_outputs"]
        for target in record["blocked_by"]
    }
    providers = {}
    for index, target in enumerate(sorted(targets)):
        provider = tmp_path / f"provider-{index}.json"
        provider.write_text(
            json.dumps(
                {
                    "format": "rulespec/v1",
                    "rules": [{"name": target.partition("#")[2], "kind": "derived"}],
                }
            )
        )
        providers[target] = provider
    monkeypatch.setattr(
        vp,
        "_resolve_rulespec_import_file_static",
        lambda target, **kwargs: providers.get(target),
    )
    pipeline = vp.ValidatorPipeline(
        axiom_rules_path=tmp_path,
        policy_repo_path=tmp_path,
        local_corpus_release=None,
        enable_oracles=False,
        source_citation_path=CITATION,
        require_complete_source_unit=True,
    )
    monkeypatch.setattr(pipeline, "_validation_source_root", lambda _: tmp_path)

    def run(candidate=None, source_text=None, *, available=True, provider_content=None):
        if not available:
            providers.clear()
        if provider_content is not None:
            for provider in providers.values():
                provider.write_text(
                    provider_content(provider.read_text())
                    if callable(provider_content)
                    else provider_content
                )
        content = yaml.safe_dump(candidate if candidate is not None else payload)
        rules_file = tmp_path / "child-tax-credit.yaml"
        rules_file.write_text(content)
        return pipeline._complete_source_unit_issues(
            content,
            validation_source_texts={CITATION: source_text or source},
            test_cases=cases,
            rules_file=rules_file,
        )

    return run


def _paired(issues):
    return [issue for issue in issues if "require paired positive/blocking" in issue]


def test_real_precise_deferral_candidate_needs_no_paired_witnesses(evaluate):
    assert evaluate() == []


@pytest.mark.parametrize(
    "mutation",
    [
        "vague",
        "wrong_section",
        "wrong_major",
        "unknown_output",
        "missing_symbol",
        "wrong_module",
    ],
)
def test_imprecise_definition_deferrals_still_require_witnesses(evaluate, mutation):
    payload, _, _ = _candidate()
    for record in payload["module"]["deferred_outputs"]:
        if mutation == "vague":
            record["reason"] = "Already handled elsewhere."
        elif mutation == "wrong_section":
            record["reason"] = record["reason"].replace("3.06(1)", "3.05(1)")
        elif mutation == "wrong_major":
            record["reason"] = record["reason"].replace("3.06(1)", "99.06(1)")
        elif mutation == "unknown_output":
            record["blocked_by"] = [record["blocked_by"][0] + "_nonexistent"]
        elif mutation == "missing_symbol":
            record["blocked_by"] = [record["blocked_by"][0].partition("#")[0]]
        else:
            record["output"] = record["output"].replace(
                "rev-proc-2025-32", "rev-proc-2024-40"
            )
    assert _paired(evaluate(payload))


def test_deferral_requires_resolved_existing_upstream_outputs(evaluate):
    assert _paired(evaluate(available=False))


def test_precise_definition_deferrals_also_accept_canonical_branch_paths(evaluate):
    payload, _, _ = _candidate()
    for record in payload["module"]["deferred_outputs"]:
        if record["output"].partition("#")[2] in {
            "threshold_phaseout_amount_definition",
            "completed_phaseout_amount_definition",
        }:
            record["output"] = record["output"].replace(
                "/child-tax-credit#", "/page-14/06/1#"
            )
    assert evaluate(payload) == []


@pytest.mark.parametrize(
    "provider_content",
    [
        '{"format": "rulespec/v1", "rules": [{"name": "unrelated", "kind": "derived"}]}',
        '{"format": "rulespec/v1", "rules": [{"name": "eitc_phase_out_income", "kind": "input"}]}',
        "format: rulespec/v1\nrules: []\nrules: [{name: eitc_phase_out_income, kind: derived}]",
        '{"format": "rulespec/v1", "rules": [{"name": "eitc_phase_out_income", "kind": "derived"}, {"name": "eitc_phase_out_income", "kind": "derived"}]}',
        "rules: [",
    ],
)
def test_provider_file_must_export_each_exact_unambiguous_output(
    evaluate, provider_content
):
    assert _paired(evaluate(provider_content=provider_content))


@pytest.mark.parametrize("mutation", ["input", "duplicate_rule", "wrong_format"])
def test_resolved_symbol_must_be_an_unambiguous_output(evaluate, mutation):
    def corrupt(content):
        provider = json.loads(content)
        if mutation == "input":
            provider["rules"][0]["kind"] = "input"
        elif mutation == "duplicate_rule":
            provider["rules"].append(copy.deepcopy(provider["rules"][0]))
        else:
            provider["format"] = "untrusted/v1"
        return json.dumps(provider)

    assert _paired(evaluate(provider_content=corrupt))


@pytest.mark.parametrize("removed", ["threshold", "completed"])
def test_one_definition_deferral_does_not_exempt_its_sibling(evaluate, removed):
    payload, _, _ = _candidate()
    payload["module"]["deferred_outputs"] = [
        record
        for record in payload["module"]["deferred_outputs"]
        if f"#{removed}_phaseout_amount_definition" not in record["output"]
    ]
    issues = " ".join(_paired(evaluate(payload)))
    other = "completed" if removed == "threshold" else "threshold"
    assert f'"{removed} phaseout amount"' in issues
    assert f'"{other} phaseout amount"' not in issues


def test_unrelated_condition_in_the_same_source_paragraph_still_needs_witnesses(
    evaluate,
):
    _, _, source = _candidate()
    source += " If the taxpayer is disqualified, no credit is allowed."
    assert "taxpayer is disqualified" in " ".join(_paired(evaluate(source_text=source)))


def test_existing_but_unrelated_provider_cannot_defer_a_definition(evaluate):
    payload, _, _ = _candidate()
    sibling_targets = copy.deepcopy(
        payload["module"]["deferred_outputs"][0]["blocked_by"]
    )
    for record in payload["module"]["deferred_outputs"]:
        if "phaseout_amount_definition" in record["output"]:
            record["blocked_by"] = sibling_targets
    assert _paired(evaluate(payload))


def test_unrelated_arithmetic_in_the_same_paragraph_is_not_deferred(evaluate):
    _, _, source = _candidate()
    source += " The additional credit equals twice the qualifying expenses."
    issues = evaluate(source_text=source)
    assert any("complete-source-unit:formula" in issue for issue in issues)


def test_unrelated_condition_within_a_definition_sentence_is_not_deferred(evaluate):
    _, _, source = _candidate()
    source = source.replace(
        "begins to phase out.",
        "begins to phase out, but if the taxpayer is disqualified, no credit is allowed.",
    )
    assert "taxpayer is disqualified" in " ".join(_paired(evaluate(source_text=source)))


@pytest.mark.parametrize(
    "heading", ["SECTION 4. OTHER ADJUSTMENTS.", "Section 4. Other Adjustments."]
)
def test_conflicting_explicit_major_section_is_not_deferred(evaluate, heading):
    _, _, source = _candidate()
    assert _paired(evaluate(source_text=heading + " " + source))
