"""Historical actual captures exercise complete graph accounting, not signing."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.bound_form_structure import recognize_bound_form_structure
from axiom_encode.harness.bound_form_operands import (
    bind_ordered_form,
    observed_form_frame,
)
from axiom_encode.harness.same_case_execution import (
    CaseExecution,
    ClosureFile,
    SameCaseEvidence,
    case_key,
)

FIXTURE = Path(__file__).parent / "fixtures/bound_credit_excess/observed-graph.json"
ROOT = "ca/policies/cra/t1-2025/nl479-credits.yaml"
CITATION = "ca/policy/cra/t1-2025/nl479-credits"


def data():
    return json.loads(FIXTURE.read_text())


def bind(d):
    form = recognize_bound_form_structure(d["source"], CITATION)
    return bind_ordered_form(
        form, d["content"], {CITATION: d["source"]}, root_relative=ROOT
    )


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


def test_actual_parent_complete_partition_and_thirteen_captures(tmp_path):
    d = data()
    b = bind(d)
    assert b is not None
    assert len(b.document_requirements) == 2
    assert len(b.row_owners) == 9 and len(b.entered) == 4
    assert b.credit_excess.return_field == "47900"
    assert b.credit_excess.credit_owner == dict(b.row_owners)[9]
    assert observed_form_frame(b, evidence(tmp_path, d), d["content"], d["cases"])


def _claimant_pair(case, record, claimant, suffix):
    """Synthetic identity variant of retained captures, not fresh engine evidence."""
    case, record = copy.deepcopy(case), copy.deepcopy(record)
    case["name"] += suffix
    case["entity_id"] = claimant
    for fact in record["request"]["dataset"]["inputs"]:
        fact["entity_id"] = claimant
    record["request"]["queries"][0]["entity_id"] = claimant
    result = record["response"]["results"][0]
    result["entity_id"] = claimant
    for trace in result["trace"].values():
        trace["entity_id"] = claimant
    return case, record


def _ordered_case_records(d, pairs):
    d["cases"] = [case for case, _record in pairs]
    d["records"] = []
    for index, (_case, record) in enumerate(pairs, 1):
        record["case_index"] = index
        d["records"].append(record)


@pytest.mark.parametrize("direction", [False, True, "both"])
@pytest.mark.parametrize("order", ["append", "prepend", "reverse"])
def test_other_claimant_cannot_erase_complete_document_pair(tmp_path, direction, order):
    from axiom_encode.harness.bound_form_obligations import _observed_document_pair_keys
    from axiom_encode.harness.validator_pipeline import ValidatorPipeline

    d = data()
    original_pairs = list(zip(d["cases"], d["records"], strict=True))
    indices = [8, 9] if direction == "both" else [9 if direction else 8]
    extras = [
        _claimant_pair(*original_pairs[i], "another-claimant", "_other_claimant")
        for i in indices
    ]
    pairs = original_pairs + extras
    if order == "prepend":
        pairs = extras + original_pairs
    elif order == "reverse":
        pairs.reverse()
    _ordered_case_records(d, pairs)
    b = bind(d)
    captured = evidence(tmp_path / "ordered", d)
    assert observed_form_frame(b, captured, d["content"], d["cases"])
    pipeline = ValidatorPipeline(
        tmp_path / "rulespec-ca",
        tmp_path / "engine",
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
    )
    assert (
        pipeline._complete_source_unit_issues(
            d["content"],
            validation_source_texts={CITATION: d["source"]},
            proof_source_texts={CITATION: d["source"]},
            test_cases=d["cases"],
            execution_evidence=captured,
        )
        == []
    )
    prefix = "ca:policies/cra/t1-2025/nl479-credits#"
    selected = _observed_document_pair_keys(
        b.document_requirements, captured, d["cases"], prefix, d["cases"][0]["period"]
    )
    assert selected is not None
    # Multiple valid pairs and case order must produce the same owner tokens.
    assert selected == _observed_document_pair_keys(
        b.document_requirements,
        captured,
        list(reversed(d["cases"])),
        prefix,
        d["cases"][0]["period"],
    )


def test_document_directions_from_different_claimants_do_not_form_pair(tmp_path):
    from axiom_encode.harness.validator_pipeline import ValidatorPipeline

    d = data()
    pairs = list(zip(d["cases"], d["records"], strict=True))
    pairs[9] = _claimant_pair(*pairs[9], "different-claimant", "_different")
    _ordered_case_records(d, pairs)
    b = bind(d)
    captured = evidence(tmp_path, d)
    assert not observed_form_frame(b, captured, d["content"], d["cases"])
    pipeline = ValidatorPipeline(
        tmp_path / "rulespec-ca",
        tmp_path / "engine",
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
    )
    issues = pipeline._complete_source_unit_issues(
        d["content"],
        validation_source_texts={CITATION: d["source"]},
        proof_source_texts={CITATION: d["source"]},
        test_cases=d["cases"],
        execution_evidence=captured,
    )
    assert any("applicability conditions" in issue for issue in issues)


@pytest.mark.parametrize(
    "defect",
    [
        "subtotal",
        "duplicate_credit_input",
        "reverse",
        "else",
        "extra_rule",
        "extra_input",
        "wrong_unit",
        "wrong_entity",
        "wrong_year",
        "default",
        "proof",
        "tax_alias",
        "relation",
    ],
)
def test_source_partition_refuses_unowned_or_misbound_excess(defect):
    d = data()
    p = yaml.safe_load(d["content"])
    r = p["rules"][-1]
    tax = p["inputs"][-1]
    if defect == "subtotal":
        r["versions"][0]["formula"] = r["versions"][0]["formula"].replace(
            "newfoundland_and_labrador_credits_line_47900",
            "nl479_physical_activity_tax_credit_line_7",
        )
    elif defect == "duplicate_credit_input":
        p["inputs"].append(dict(tax, name="duplicate_credit"))
        r["versions"][0]["formula"] = r["versions"][0]["formula"].replace(
            "newfoundland_and_labrador_credits_line_47900", "duplicate_credit"
        )
    elif defect == "reverse":
        r["versions"][0]["formula"] = r["versions"][0]["formula"].replace(" > ", " < ")
    elif defect == "else":
        r["versions"][0]["formula"] = r["versions"][0]["formula"].replace(
            "else: 0", "else: 1"
        )
    elif defect == "extra_rule":
        p["rules"].append(dict(r, name="another_excess"))
    elif defect == "extra_input":
        p["inputs"].append(dict(tax, name="unused_tax"))
    elif defect == "wrong_unit":
        r["unit"] = "USD"
    elif defect == "wrong_entity":
        tax["entity"] = "Family"
    elif defect == "wrong_year":
        r["versions"][0]["effective_to"] = "2026-12-31"
    elif defect == "default":
        tax["default"] = 0
    elif defect == "proof":
        r["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = "You get a refund."
    elif defect == "tax_alias":
        old = tax["name"]
        tax["name"] = "eligible_expenses"
        r["versions"][0]["formula"] = r["versions"][0]["formula"].replace(
            old, tax["name"]
        )
    else:
        p["relations"] = [{"name": "belongs_to_family"}]
    d["content"] = yaml.safe_dump(p, sort_keys=False)
    assert bind(d) is None


@pytest.mark.parametrize(
    "defect",
    [
        "missing_below",
        "missing_equal",
        "missing_above",
        "wrong_fact_entity",
        "bool_tax",
        "changed_rows",
        "changed_claimant",
        "compiled_else_bool",
        "compiled_reverse",
        "trace_reverse",
        "trace_dependency",
        "output",
        "request_missing",
        "stale_content",
    ],
)
def test_actual_excess_witness_guards(tmp_path, defect):
    d = data()
    b = bind(d)
    assert b is not None
    rec = d["records"][-3]
    rule = b.credit_excess.rule_name
    prefix = "ca:policies/cra/t1-2025/nl479-credits#"
    ref = prefix + rule
    if defect.startswith("missing_"):
        del d["records"][
            {"missing_below": -3, "missing_equal": -2, "missing_above": -1}[defect]
        ]
    elif defect == "wrong_fact_entity":
        rec["request"]["dataset"]["inputs"][-1]["entity"] = "Family"
    elif defect == "bool_tax":
        rec["request"]["dataset"]["inputs"][-1]["value"] = {
            "kind": "integer",
            "value": True,
        }
    elif defect == "changed_rows":
        rec["request"]["dataset"]["inputs"][0]["value"]["value"] = 999
    elif defect == "changed_claimant":
        rec["request"]["queries"][0]["entity_id"] = "someone-else"
    elif defect.startswith("compiled_"):
        compiled = d["artifact"]["program"]["derived"][-1]
        for expr in [compiled["expr"], compiled["versions"][0]["expr"]]:
            if defect == "compiled_else_bool":
                expr["else_expr"]["value"] = {"kind": "bool", "value": False}
            else:
                expr["condition"]["op"] = "lt"
    elif defect == "trace_reverse":
        rec["response"]["results"][0]["trace"][ref]["executed_expression"] = "0"
    elif defect == "trace_dependency":
        rec["response"]["results"][0]["trace"][ref]["dependencies"] = []
    elif defect == "output":
        rec["response"]["results"][0]["outputs"][ref]["value"]["value"] = "0"
    elif defect == "request_missing":
        rec["request"]["queries"][0]["outputs"].remove(ref)
    else:
        d["content"] += "\n# altered content\n"
    assert not observed_form_frame(b, evidence(tmp_path, d), d["content"], d["cases"])


@pytest.mark.parametrize("change", ["field", "negated", "qualified", "preamble"])
def test_changed_refund_source_cannot_rebind_with_coherent_proofs(change):
    import hashlib

    d = data()
    p = yaml.safe_load(d["content"])
    old = d["source"]
    if change == "field":
        new = old.replace("47900", "48000")
    elif change == "negated":
        new = old.replace("You can claim these", "Do not:\n\nYou can claim these")
    elif change == "qualified":
        new = old.replace(
            "for the difference.\n\n",
            "for the difference.\n\nunless another condition applies.\n\n",
        )
    else:
        new = old.replace("even if you do not", "only if you do not")
    d["source"] = new
    p["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        new.encode()
    ).hexdigest()
    if change == "field":
        # Update every citation coherently, so refusal is the field ownership gate.
        for rule in p["rules"]:
            for atom in rule["metadata"]["proof"]["atoms"]:
                atom["source"]["excerpt"] = atom["source"]["excerpt"].replace(
                    "47900", "48000"
                )
    d["content"] = yaml.safe_dump(p, sort_keys=False)
    assert bind(d) is None


@pytest.mark.parametrize("component", ["derived", "parameters", "relations"])
def test_compiled_union_cannot_add_ignored_declarations(tmp_path, component):
    d = data()
    b = bind(d)
    assert b is not None
    if component == "relations":
        d["artifact"]["program"][component].append({"name": "unowned_relation"})
    else:
        extra = json.loads(json.dumps(d["artifact"]["program"][component][0]))
        extra["name"] = "unowned_declaration"
        d["artifact"]["program"][component].append(extra)
    assert not observed_form_frame(b, evidence(tmp_path, d), d["content"], d["cases"])


def condition_branches(d, b):
    from axiom_encode.harness.source_completeness import SourceStructureBranch

    spans = [b.credit_excess.source_span]
    table = next(
        x for x in b.document_requirements if x.input_name.startswith("is_claiming_")
    )
    start, end = table.source_span
    spans.append((start - len("Physical activity tax credit\n\n"), end))
    return tuple(
        SourceStructureBranch(
            path=(),
            kind="exception-clause",
            label="source unit",
            text=d["source"][a:z],
            start=a,
            end=z,
        )
        for a, z in spans
    )


def allocate(tmp_path, d, b, branches, **overrides):
    from axiom_encode.harness.bound_form_operands import (
        observed_form_condition_witnesses,
    )
    from axiom_encode.harness.source_completeness import _unmatched_evidence_obligations

    args = dict(
        source_text=d["source"],
        corpus_citation_path=CITATION,
        principal_rules={
            r["name"]: r
            for r in yaml.safe_load(d["content"])["rules"]
            if r["kind"] == "derived"
        },
    )
    args.update(overrides)
    witnesses = observed_form_condition_witnesses(
        b, evidence(tmp_path, d), d["content"], d["cases"], branches, **args
    )
    return _unmatched_evidence_obligations(witnesses)


def test_exact_conditions_allocate_distinct_actual_evidence(tmp_path):
    d = data()
    b = bind(d)
    branches = condition_branches(d, b)
    assert allocate(tmp_path, d, b, branches) == ()


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "span",
        "kind",
        "principal",
        "source",
        "citation",
        "unrelated",
        "stale",
        "no_table_pair",
        "no_refund_boundary",
    ],
)
def test_condition_hook_preserves_unowned_or_unauthenticated_obligations(
    tmp_path, change
):
    from dataclasses import replace

    d = data()
    b = bind(d)
    branches = condition_branches(d, b)
    overrides = {}
    if change == "duplicate":
        branches += (replace(branches[0], label="independent duplicate obligation"),)
    elif change == "span":
        branches = (
            replace(
                branches[0], start=branches[0].start - 1, text=" " + branches[0].text
            ),
            branches[1],
        )
    elif change == "kind":
        branches = (replace(branches[0], kind="formula"), branches[1])
    elif change == "principal":
        branches = (
            replace(
                branches[0],
                required_principal_output=b.document_requirements[0].rule_name,
            ),
            branches[1],
        )
    elif change == "source":
        overrides["source_text"] = d["source"] + "Unowned condition."
    elif change == "citation":
        overrides["corpus_citation_path"] = CITATION + "/other"
    elif change == "unrelated":
        branches = (replace(branches[0], path=("independent",)), branches[1])
    elif change == "stale":
        d["content"] += "\n# changed\n"
    elif change == "no_table_pair":
        del d["records"][8]
    else:
        del d["records"][-2]
    assert allocate(tmp_path, d, b, branches, **overrides)


def test_full_analyzer_keeps_generic_route_and_requires_actual_evidence(tmp_path):
    from axiom_encode.harness.validator_pipeline import ValidatorPipeline

    d = data()
    pipeline = ValidatorPipeline(
        tmp_path / "rulespec-ca",
        tmp_path / "engine",
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
    )
    args = dict(
        validation_source_texts={CITATION: d["source"]},
        proof_source_texts={CITATION: d["source"]},
        test_cases=d["cases"],
    )
    without = pipeline._complete_source_unit_issues(d["content"], **args)
    assert any("applicability conditions" in x for x in without)
    with_actual = pipeline._complete_source_unit_issues(
        d["content"], execution_evidence=evidence(tmp_path, d), **args
    )
    assert with_actual == []


def test_coherent_negative_tax_is_not_certified(tmp_path):
    d = data()
    b = bind(d)
    assert b is not None
    prefix = "ca:policies/cra/t1-2025/nl479-credits#"
    fact = prefix + "input." + b.credit_excess.input_name
    ref = prefix + b.credit_excess.rule_name
    case, record = d["cases"][-3], d["records"][-3]
    case["input"][fact] = -1
    case["output"][ref] = 396.8
    actual_fact = next(
        x for x in record["request"]["dataset"]["inputs"] if x["name"] == fact
    )
    actual_fact["value"] = {"kind": "integer", "value": -1}
    result = record["response"]["results"][0]
    for observed in [result["outputs"][ref], result["trace"][ref]]:
        observed["value"] = {"kind": "decimal", "value": "396.8"}
    assert not observed_form_frame(b, evidence(tmp_path, d), d["content"], d["cases"])


def test_new_unowned_condition_is_not_cleared_by_complete_form_certificate(tmp_path):
    import hashlib

    from axiom_encode.harness.validator_pipeline import ValidatorPipeline

    d = data()
    d["source"] += "\n\nIf the claimant is nonresident, this credit does not apply."
    payload = yaml.safe_load(d["content"])
    payload["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        d["source"].encode()
    ).hexdigest()
    d["content"] = yaml.safe_dump(payload, sort_keys=False)
    b = bind(d)
    assert b is not None
    pipeline = ValidatorPipeline(
        tmp_path / "rulespec-ca",
        tmp_path / "engine",
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
    )
    issues = pipeline._complete_source_unit_issues(
        d["content"],
        validation_source_texts={CITATION: d["source"]},
        proof_source_texts={CITATION: d["source"]},
        test_cases=d["cases"],
        execution_evidence=evidence(tmp_path, d),
    )
    assert any("nonresident" in issue and "conditions" in issue for issue in issues)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("context", ["plain", "negated", "qualified"])
def test_sheet_enclosing_context_reaches_full_analyzer(tmp_path, newline, context):
    """Coherent source/capture variants, not fresh native execution."""
    import hashlib

    from axiom_encode.harness.validator_pipeline import ValidatorPipeline

    d = data()
    clause = "If you need more space, attach an additional sheet of paper."
    assert d["source"].count(clause) == 1
    replacement = {
        "plain": clause,
        "negated": "Do not:\n\n" + clause,
        "qualified": clause + "\n\nunless another condition applies.",
    }[context]
    paragraph = "\n\n" + clause + "\n\n"
    assert paragraph in d["source"]
    # Vary the attachment paragraph's raw line endings, not unrelated owners.
    d["source"] = d["source"].replace(
        paragraph, newline * 2 + replacement.replace("\n", newline) + newline * 2
    )
    payload = yaml.safe_load(d["content"])
    payload["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        d["source"].encode()
    ).hexdigest()
    d["content"] = yaml.safe_dump(payload, sort_keys=False)
    pipeline = ValidatorPipeline(
        tmp_path / "rulespec-ca",
        tmp_path / "engine",
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
    )
    issues = pipeline._complete_source_unit_issues(
        d["content"],
        validation_source_texts={CITATION: d["source"]},
        proof_source_texts={CITATION: d["source"]},
        test_cases=d["cases"],
        execution_evidence=evidence(tmp_path, d),
    )
    if context == "plain":
        assert bind(d) is not None
        assert issues == []
    else:
        assert bind(d) is None
        assert issues
