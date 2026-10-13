"""Exercise the actual #1707 deferral artifact through the validator."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from hypothesis import Phase, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as vp

FIXTURE = Path(__file__).parent / "fixtures/source_completeness/irs_rev_proc_2025_32"
CITATION = "us/guidance/irs/rev-proc-2025-32/page-14"
DOCUMENT = "policies/irs/rev-proc-2025-32"
THRESHOLD_EXPORTS = (
    f"us:{DOCUMENT}/page-15#earned_income_credit_phaseout_threshold_joint_amount",
    f"us:{DOCUMENT}/page-15#earned_income_credit_phaseout_threshold_other_amount",
)
BEHAVIOR_EXPORTS = (
    "us:statutes/26/32#eitc_phase_out_income",
    "us:statutes/26/32#eitc_phase_out_start",
    "us:statutes/26/32#eitc_reduction",
)
DEFINITION_ENDINGS = {
    "earned_income": "maximum amount of the earned income credit is allowed",
    "threshold_phaseout": "maximum amount of the credit begins to phase out",
    "completed_phaseout": "at or above which no credit is allowed",
}


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


@pytest.mark.parametrize("definition", DEFINITION_ENDINGS)
@pytest.mark.parametrize(
    "condition",
    [
        ", but if the taxpayer is disqualified, no credit is allowed",
        " unless the taxpayer is disqualified",
    ],
)
def test_unrelated_condition_within_a_definition_sentence_is_not_deferred(
    evaluate, definition, condition
):
    _, _, source = _candidate()
    ending = DEFINITION_ENDINGS[definition]
    assert source.count(ending + ".") == 1
    source = source.replace(
        ending + ".",
        ending + condition + ".",
        1,
    )
    assert "taxpayer is disqualified" in " ".join(_paired(evaluate(source_text=source)))


@pytest.mark.parametrize(
    "heading", ["SECTION 4. OTHER ADJUSTMENTS.", "Section 4. Other Adjustments."]
)
def test_conflicting_explicit_major_section_is_not_deferred(evaluate, heading):
    _, _, source = _candidate()
    assert _paired(evaluate(source_text=heading + " " + source))


def test_threshold_only_exports_cannot_defer_maximum_income_definitions(tmp_path):
    """Replay the review's canonical checkout with the real static resolver."""
    payload, cases, source = _candidate()
    root = tmp_path / "rulespec-us" / "us"
    page = root / DOCUMENT / "page-15.yaml"
    page.parent.mkdir(parents=True)
    page.write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "rules": [
                    {
                        "name": name,
                        "kind": "parameter",
                        "dtype": "Money",
                        "unit": "USD",
                        "versions": [
                            {"effective_from": "2026-01-01", "formula": "20000"}
                        ],
                    }
                    for name in (
                        "earned_income_credit_earned_income_amount",
                        *(target.partition("#")[2] for target in THRESHOLD_EXPORTS),
                    )
                ],
            }
        )
    )
    statute = root / "statutes" / "26" / "32.yaml"
    statute.parent.mkdir(parents=True)
    statute.write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "rules": [{"name": "eitc_phased_in", "kind": "derived"}],
            }
        )
    )
    for record in payload["module"]["deferred_outputs"]:
        if record["output"].partition("#")[2] in {
            "threshold_phaseout_amount_definition",
            "completed_phaseout_amount_definition",
        }:
            record["blocked_by"] = list(THRESHOLD_EXPORTS)
    content = yaml.safe_dump(payload)
    candidate_file = root / DOCUMENT / "child-tax-credit.yaml"
    candidate_file.write_text(content)
    pipeline = vp.ValidatorPipeline(
        axiom_rules_path=root,
        policy_repo_path=root,
        local_corpus_release=None,
        enable_oracles=False,
        source_citation_path=CITATION,
        require_complete_source_unit=True,
    )
    resolved = pipeline._complete_source_unit_deferred_outputs(content, candidate_file)
    assert set(THRESHOLD_EXPORTS) <= set(resolved)
    assert not set(BEHAVIOR_EXPORTS) & set(resolved)
    issues = " ".join(
        _paired(
            pipeline._complete_source_unit_issues(
                content,
                validation_source_texts={CITATION: source},
                test_cases=cases,
                rules_file=candidate_file,
            )
        )
    )
    assert '"threshold phaseout amount"' in issues
    assert '"completed phaseout amount"' in issues


def test_resolved_behavior_exports_cannot_be_parameters(evaluate):
    def replace_behavior_with_constant(content):
        provider = json.loads(content)
        rule = provider["rules"][0]
        if rule["name"].startswith("eitc_"):
            rule["kind"] = "parameter"
        return json.dumps(provider)

    issues = " ".join(
        _paired(evaluate(provider_content=replace_behavior_with_constant))
    )
    assert '"threshold phaseout amount"' in issues
    assert '"completed phaseout amount"' in issues


@settings(max_examples=40, deadline=None)
@given(
    definition=st.sampled_from(["threshold", "completed"]),
    targets=st.sets(
        st.sampled_from((*BEHAVIOR_EXPORTS, *THRESHOLD_EXPORTS)), min_size=1
    ),
)
def test_definition_deferral_requires_all_its_specific_exports(definition, targets):
    payload, _, source = _candidate()
    record = next(
        record
        for record in payload["module"]["deferred_outputs"]
        if record["output"].endswith(f"#{definition}_phaseout_amount_definition")
    )
    record["blocked_by"] = sorted(targets)
    clauses = sc._resolved_definition_deferral_clauses(
        record,
        payload=payload,
        corpus_citation_path=CITATION,
        source_text=source,
        branches=sc.recognize_source_structure(source, corpus_citation_path=CITATION),
        resolved_dependency_outputs=sorted(targets),
    )
    required = {
        BEHAVIOR_EXPORTS[0],
        BEHAVIOR_EXPORTS[1 if definition == "threshold" else 2],
    }
    assert bool(clauses) == (required <= targets)


@settings(
    max_examples=40,
    deadline=None,
    phases=tuple(phase for phase in Phase if phase != Phase.explain),
)
@given(
    definition=st.sampled_from(tuple(DEFINITION_ENDINGS)),
    cue=st.sampled_from(["if", "unless"]),
    conjunction=st.sampled_from([", but ", ", and ", " and "]),
    subject=st.sampled_from(["taxpayer", "claimant", "spouse"]),
    status=st.sampled_from(["disqualified", "ineligible", "excluded"]),
)
def test_definition_deferral_never_exempts_an_independent_condition(
    definition, cue, conjunction, subject, status
):
    payload, _, source = _candidate()
    ending = DEFINITION_ENDINGS[definition]
    source = source.replace(
        ending + ".",
        f"{ending}{conjunction}{cue} the {subject} is {status}, no credit is allowed.",
        1,
    )
    record = next(
        record
        for record in payload["module"]["deferred_outputs"]
        if record["output"].endswith(f"#{definition}_amount_definition")
    )
    clauses = sc._resolved_definition_deferral_clauses(
        record,
        payload=payload,
        corpus_citation_path=CITATION,
        source_text=source,
        branches=sc.recognize_source_structure(source, corpus_citation_path=CITATION),
        resolved_dependency_outputs=record["blocked_by"],
    )
    assert not clauses


@pytest.mark.parametrize("definition", ["threshold_phaseout", "completed_phaseout"])
def test_definition_deferral_requires_its_intrinsic_condition_profile(definition):
    payload, _, source = _candidate()
    term = definition.replace("_", " ") + " amount"
    prefix, sentence_start, remainder = source.partition(f'The "{term}"')
    sentence, separator, suffix = remainder.partition(".")
    assert sentence_start and separator
    sentence = sentence.replace("(or, if greater, earned income)", "(or earned income)")
    source = (
        prefix
        + sentence_start
        + sentence
        + " unless the taxpayer is disqualified."
        + suffix
    )
    record = next(
        record
        for record in payload["module"]["deferred_outputs"]
        if record["output"].endswith(f"#{definition}_amount_definition")
    )
    clauses = sc._resolved_definition_deferral_clauses(
        record,
        payload=payload,
        corpus_citation_path=CITATION,
        source_text=source,
        branches=sc.recognize_source_structure(source, corpus_citation_path=CITATION),
        resolved_dependency_outputs=record["blocked_by"],
    )
    assert not clauses
