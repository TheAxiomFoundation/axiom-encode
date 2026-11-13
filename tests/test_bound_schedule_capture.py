"""Actual historical captures; synthetic closure IDs are not authentication."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from axiom_encode.bound_schedule_structure import recognize_bound_schedule_structure
from axiom_encode.harness.bound_schedule_execution import (
    observe_schedule,
    observed_schedule_condition_witnesses,
)
from axiom_encode.harness.bound_schedule_operands import bind_schedule
from axiom_encode.harness.same_case_execution import (
    CaseExecution,
    ClosureFile,
    SameCaseEvidence,
    case_key,
)

FIXTURE = Path(__file__).parent / "fixtures/nb_schedule/observed-graph.json"
ROOT = "ca/policies/cra/t1-2025/nb-s12-seniors-home-renovation-credit.yaml"
CITATION = "ca/policy/cra/t1-2025/nb-s12-seniors-home-renovation-credit"


def data():
    return json.loads(FIXTURE.read_text())


def evidence(tmp_path, d):
    original = tmp_path / "original" / ROOT
    original.parent.mkdir(parents=True)
    original.write_text(d["content"])
    snapshot = tmp_path / "snapshot"
    target = snapshot / ROOT
    target.parent.mkdir(parents=True)
    target.write_text(d["content"])
    return SameCaseEvidence(
        (ClosureFile(ROOT, original, d["content"].encode()),),
        ROOT,
        snapshot,
        "a" * 40,
        "b" * 64,
        "c" * 40,
        json.dumps(d["artifact"]).encode(),
        tuple(
            CaseExecution(
                case_key(d["cases"][r["case_index"] - 1]),
                json.dumps(r["request"]),
                json.dumps(r["response"]),
                r["case_index"],
            )
            for r in d["records"]
        ),
    )


def observe(tmp_path, d):
    form = recognize_bound_schedule_structure(d["source"], CITATION)
    binding = bind_schedule(
        form, d["content"], {CITATION: d["source"]}, root_relative=ROOT
    )
    assert binding is not None
    return observe_schedule(binding, evidence(tmp_path, d), d["content"], d["cases"])


def test_actual_complete_schedule_captures(tmp_path):
    d = data()
    assert len(d["records"]) == 21
    result = observe(tmp_path, d)
    assert result is not None
    assert len(result.medical_claimants) == 2
    assert result.row_cases and result.pairs


def analyzer(tmp_path, d, *, actual=True, captured=None):
    from axiom_encode.harness.validator_pipeline import ValidatorPipeline

    pipeline = ValidatorPipeline(
        tmp_path / "rulespec-ca",
        tmp_path / "engine",
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
    )
    return pipeline._complete_source_unit_issues(
        d["content"],
        validation_source_texts={CITATION: d["source"]},
        proof_source_texts={CITATION: d["source"]},
        test_cases=d["cases"],
        execution_evidence=(captured if captured is not None else evidence(tmp_path, d))
        if actual
        else None,
    )


def test_full_analyzer_requires_complete_actual_observation(tmp_path):
    d = data()
    without = analyzer(tmp_path, d, actual=False)
    assert any("source-explicit-conditions" in str(i) for i in without)
    assert any("applicability conditions" in str(i) for i in without)
    assert analyzer(tmp_path, d) == []


def test_numeric_string_normalization_preserves_raw_capture_identity(tmp_path):
    d = data()
    d["cases"][0]["output"] = {k: str(v) for k, v in d["cases"][0]["output"].items()}
    assert analyzer(tmp_path, d) == []


@pytest.mark.parametrize(
    "change", ["raw_date", "case_identity", "extra_role", "altered_proof"]
)
def test_full_analyzer_refuses_changed_raw_case_or_role_contract(tmp_path, change):
    d = data()
    captured = evidence(tmp_path, d)
    if change == "raw_date":
        key = next(
            k
            for k in d["cases"][14]["input"]
            if k.endswith("medical_separation_reference_date")
        )
        d["cases"][14]["input"][key] = "2025-12-30"
    elif change == "case_identity":
        d["cases"][14]["name"] = "unobserved-case"
    else:
        p = yaml.safe_load(d["content"])
        rule = next(
            r
            for r in p["rules"]
            if r["name"] == "medical_separation_claim_permission_applies"
        )
        if change == "extra_role":
            rule["versions"][0]["formula"] += " and is_filing_paper_return"
        else:
            rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
                "A nonexistent medical clause."
            )
        d["content"] = yaml.safe_dump(p, sort_keys=False)
    issues = analyzer(tmp_path, d, captured=captured)
    assert any("source-explicit-conditions" in str(i) for i in issues)
    assert any("applicability conditions" in str(i) for i in issues)


@pytest.mark.parametrize(
    "before,after",
    [
        ("CRA Schedule NB(S12) 2025", "CRA Schedule NB(S12) 2024"),
        (
            "See the privacy notice on your return.",
            "See the privacy notice on your return.\n\nThe amount for 2025 is increased.",
        ),
        (
            "Amount paid (including all applicable taxes) ^",
            "Amount paid less assistance received ^",
        ),
        (
            "GST/HST Number (if applicable)",
            "GST/HST Number (if assistance is received)",
        ),
        ("Description ^\nAmount paid", "Recovery amount ^\nDescription ^\nAmount paid"),
        (
            "Date on sales slip or contract ^\nSupplier or contractor - Name ^",
            "Supplier or contractor - Name ^\nDate on sales slip or contract ^",
        ),
        (
            "Amount paid (including all applicable taxes) ^",
            "Amount paid (including all applicable taxes) ^\n\nSubtract assistance from this amount.",
        ),
    ],
)
def test_full_analyzer_preserves_unknown_year_or_table_obligations(
    tmp_path, before, after
):
    d = data()
    d["source"] = d["source"].replace(before, after)
    p = yaml.safe_load(d["content"])
    p["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        d["source"].encode()
    ).hexdigest()
    d["content"] = yaml.safe_dump(p, sort_keys=False)
    issues = analyzer(tmp_path, d)
    assert issues
    if "2025 is increased" in after:
        assert any("2025" in str(i) for i in issues)
    elif "less assistance" in after or "Subtract assistance" in after:
        assert any(
            "formula branch" in str(i) and "assistance" in str(i) for i in issues
        )


@pytest.mark.parametrize(
    "name",
    [
        "row3_assistance_excess",
        "row5_other_excess",
        "row7_other_excess",
        "row1_only",
        "entered_assistance_excess",
        "entered_other_excess",
    ],
)
def test_actual_partial_queries_enforce_only_reached_inclusion_domains(tmp_path, name):
    d = data()
    control = json.loads((FIXTURE.parent / "partial-query-controls.json").read_text())[
        name
    ]
    assert control["returncode"] == 0
    d["cases"].append(control["case"])
    d["records"].append(
        {
            "case_index": len(d["cases"]),
            "request": control["request"],
            "response": control["response"],
        }
    )
    result = observe(tmp_path, d)
    assert (result is not None) == (name == "row1_only")


def test_duplicate_compiled_local_unit_refused(tmp_path):
    d = data()
    d["artifact"]["program"]["units"].append(
        {"name": "CAD", "kind": "currency", "minor_units": 2}
    )
    assert observe(tmp_path, d) is None


def allocation(tmp_path, d, defect=None):
    from axiom_encode.harness.source_completeness import (
        SourceStructureBranch,
        _unmatched_evidence_obligations,
    )

    form = recognize_bound_schedule_structure(d["source"], CITATION)
    binding = bind_schedule(
        form, d["content"], {CITATION: d["source"]}, root_relative=ROOT
    )
    assert binding is not None
    medical = binding.conditions.medical
    residence = binding.conditions.residence
    spans = [
        (
            d["source"][: medical.span[0]].rstrip().rfind("\n\n") + 2,
            medical.permission_span[1],
        ),
        (residence.yes_span[0], len(d["source"].rstrip())),
    ]
    branches = [
        SourceStructureBranch((), "exception-clause", str(i), d["source"][a:b], a, b)
        for i, (a, b) in enumerate(spans)
    ]
    if defect == "duplicate":
        branches.append(replace(branches[0], label="duplicate"))
    elif defect == "prefix":
        b = branches[0]
        branches[0] = replace(
            b, start=b.start - 1, text=d["source"][b.start - 1 : b.end]
        )
    elif defect == "owner":
        branches[0] = replace(branches[0], required_principal_output="unrelated")
    elif defect == "path":
        branches[0] = replace(branches[0], path=("unrelated",))
    elif defect == "kind":
        branches[0] = replace(branches[0], kind="formula-clause")
    elif defect == "missing_pair":
        d["records"].pop(16)
    elif defect == "wrong_entity":
        d["records"][14]["request"]["dataset"]["inputs"][0]["entity"] = "Family"
    rules = {
        r["name"]: r
        for r in yaml.safe_load(d["content"])["rules"]
        if r["kind"] == "derived"
    }
    witnesses = observed_schedule_condition_witnesses(
        binding,
        evidence(tmp_path, d),
        d["content"],
        d["cases"],
        branches,
        source_text=d["source"].rstrip(),
        corpus_citation_path=CITATION,
        principal_rules=rules,
    )
    return _unmatched_evidence_obligations(witnesses)


def test_exact_schedule_conditions_have_unique_actual_witnesses(tmp_path):
    assert allocation(tmp_path, data()) == ()


@pytest.mark.parametrize(
    "defect",
    ["duplicate", "prefix", "owner", "path", "kind", "missing_pair", "wrong_entity"],
)
def test_schedule_condition_allocation_refuses_reuse_or_unbound_context(
    tmp_path, defect
):
    assert allocation(tmp_path, data(), defect)


@pytest.mark.parametrize(
    "defect",
    [
        "person_kind",
        "entry_kind",
        "bool_one",
        "bool_zero",
        "date_kind",
        "entity_id",
        "interval",
        "extra_fact",
        "missing_fact",
        "metadata",
        "extra_record",
        "missing_record",
        "extra_derived",
        "extra_parameter",
        "extra_relation",
        "unit",
        "expression",
        "version_expression",
        "extra_key",
        "outcome",
        "trace_expression",
        "trace_source",
        "trace_dependency",
        "frame",
    ],
)
def test_actual_capture_refuses_changed_identity_or_semantics(tmp_path, defect):
    d = data()
    program = d["artifact"]["program"]
    medical = d["records"][14]
    fact = medical["request"]["dataset"]["inputs"][0]
    trace = next(iter(medical["response"]["results"][0]["trace"].values()))
    if defect == "person_kind":
        fact["entity"] = "Family"
    elif defect == "entry_kind":
        d["records"][13]["request"]["dataset"]["inputs"][0]["entity"] = "Person"
    elif defect in {"bool_one", "bool_zero"}:
        record = medical if defect == "bool_one" else d["records"][16]
        record["request"]["dataset"]["inputs"][0]["value"]["value"] = (
            1 if defect == "bool_one" else 0
        )
    elif defect == "date_kind":
        next(
            f
            for f in medical["request"]["dataset"]["inputs"]
            if f["value"]["kind"] == "date"
        )["value"]["kind"] = "string"
    elif defect == "entity_id":
        fact["entity_id"] = "another-person"
    elif defect == "interval":
        fact["interval"]["end"] = "2025-12-30"
    elif defect == "extra_fact":
        medical["request"]["dataset"]["inputs"].append(deepcopy(fact))
    elif defect == "missing_fact":
        medical["request"]["dataset"]["inputs"].pop()
    elif defect == "metadata":
        medical["response"]["metadata"]["fallback_used"] = True
    elif defect == "extra_record":
        d["records"].append(deepcopy(medical))
    elif defect == "missing_record":
        d["records"].pop(14)
    elif defect in {"extra_derived", "extra_parameter"}:
        field = "derived" if defect == "extra_derived" else "parameters"
        program[field].append(dict(program[field][0], name="unowned"))
    elif defect == "extra_relation":
        program["relations"].append({"name": "unowned"})
    elif defect == "unit":
        next(u for u in program["units"] if u["name"] == "CAD")["minor_units"] = 0
    elif defect in {"expression", "version_expression"}:
        rule = next(
            r
            for r in program["derived"]
            if r["name"] == "additional_sheet_attachment_required"
        )
        target = rule if defect == "expression" else rule["versions"][0]
        target["expr"]["right"]["value"]["value"] = 1
    elif defect == "extra_key":
        program["derived"][0]["default"] = True
    elif defect == "outcome":
        next(iter(medical["response"]["results"][0]["outputs"].values()))["outcome"] = (
            "not_holds"
        )
    elif defect == "trace_expression":
        trace["executed_expression"] = "true"
    elif defect == "trace_source":
        trace["source"] = "unrelated"
    elif defect == "trace_dependency":
        trace["dependencies"] = ["unowned"]
    elif defect == "frame":
        medical["request"]["queries"][0]["period"]["end"] = "2025-12-30"
    assert observe(tmp_path, d) is None


@pytest.mark.parametrize(
    "change",
    ["nonmedical", "opposite_description", "relationship_and", "source_page_two"],
)
def test_coherent_factual_contract_cannot_gain_source_clearance(tmp_path, change):
    """Coherent capture variants exercise semantic refusal, not stale identity."""
    d = data()
    if change == "nonmedical":
        d = json.loads(
            json.dumps(d).replace(
                "principal_residence_separation_was_for_medical_reasons",
                "principal_residence_separation_was_for_non_medical_reasons",
            )
        )
    elif change == "relationship_and":
        d = json.loads(
            json.dumps(d).replace(
                "has_spouse_or_common_law_partner_at_reference_date",
                "has_spouse_and_common_law_partner_at_reference_date",
            )
        )
    elif change == "source_page_two":
        d = json.loads(
            json.dumps(d).replace("page 1 of your return", "page 2 of your return")
        )
    p = yaml.safe_load(d["content"])
    if change in {"nonmedical", "opposite_description"}:
        fact = next(i for i in p["inputs"] if "separation_was_for" in i["name"])
        fact["description"] = (
            "The stated separate principal residences at the reference date were for non-medical reasons."
        )
    elif change == "relationship_and":
        fact = next(i for i in p["inputs"] if i["name"].startswith("has_spouse_and"))
        fact["description"] = (
            "The claimant has both a spouse and a common-law partner at the reference date."
        )
    p["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        d["source"].encode()
    ).hexdigest()
    d["content"] = yaml.safe_dump(p, sort_keys=False)
    form = recognize_bound_schedule_structure(d["source"], CITATION)
    assert form is not None
    binding = bind_schedule(
        form, d["content"], {CITATION: d["source"]}, root_relative=ROOT
    )
    assert analyzer(tmp_path, d)
    assert binding is None


def test_residence_fact_tracks_source_page_two(tmp_path):
    d = data()
    d = json.loads(
        json.dumps(d)
        .replace("page 1 of your return", "page 2 of your return")
        .replace("return_page_one_residence", "return_page_two_residence")
    )
    p = yaml.safe_load(d["content"])
    fact = next(
        i
        for i in p["inputs"]
        if i["name"] == "expenses_relate_to_return_page_two_residence"
    )
    fact["description"] = fact["description"].replace("page1;", "page2;")
    p["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        d["source"].encode()
    ).hexdigest()
    d["content"] = yaml.safe_dump(p, sort_keys=False)
    assert observe(tmp_path / "direct", d) is not None
    assert analyzer(tmp_path / "full", d) == []
