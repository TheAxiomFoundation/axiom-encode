"""Whole groups consume all alternatives, never a leaf or neighboring output."""

import copy
import functools
from dataclasses import replace

import pytest

from axiom_encode.harness import actual_repayment_group_evidence as group
from axiom_encode.harness import source_completeness as s
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_inventory_occurrences_from_text,
)
from tests.test_actual_repayment_boundary_evidence import (
    _oas_world,
    actual,  # noqa: F401
)
from tests.test_repayment_alternative_evidence import repayment  # noqa: F401


@pytest.fixture
def whole(actual):  # noqa: F811
    other = _oas_world(copy.deepcopy(actual))
    cases = (
        actual["asserted_by_rule"]["repayment"] + other["asserted_by_rule"]["repayment"]
    )
    captures = actual["evidence"].records + tuple(
        replace(r, case_index=i) for i, r in enumerate(other["evidence"].records, 3)
    )
    actual["evidence"] = replace(actual["evidence"], records=captures)
    actual["asserted_by_rule"] = {"repayment": cases, "income": cases}
    source = actual["source_text"]
    records = group.certify_repayment_alternatives(
        actual["content"],
        source_text=source,
        raw_source_text=actual["source_context"][actual["corpus_citation_path"]],
        citation=actual["corpus_citation_path"],
        source_context=actual["source_context"],
        imported_exports=actual["imported_exports"],
        parse_branch=s._parse_first_formula_branch_node,
        parse_expression=s._parse_formula_expression,
    ).records
    assert len(records) == 1
    record = records[0]
    actual["branch"] = s.SourceStructureBranch(
        (), "condition", "group", record.text, *record.span
    )
    actual.pop("boundary")
    actual["boundary_obligations"] = s._source_boundary_obligations(
        actual["boundary_branches"],
        source_text=source,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
    )
    return actual


def test_every_alternative_has_own_actual_pair(whole):
    result = group.witnesses(**whole)
    assert len(result) == 1
    assert len(next(iter(result))[-1]) == 2


@pytest.mark.parametrize("indices", [(0, 1), (2, 3), (0, 2), (1, 3), (0, 1, 2)])
def test_missing_alternative_effect_never_clears_group(whole, indices):
    cases = whole["asserted_by_rule"]["repayment"]
    whole["asserted_by_rule"] = {
        "repayment": [cases[i] for i in indices],
        "income": cases,
    }
    assert not group.witnesses(**whole)


@pytest.mark.parametrize(
    "change",
    ["partial", "overlap", "sibling", "duplicate", "missing", "stale", "environment"],
)
def test_exact_group_ownership_and_obligation_identity(whole, change):
    branch = whole["branch"]
    if change == "partial":
        whole["branch"] = replace(branch, start=branch.start + 1, text=branch.text[1:])
    elif change == "overlap":
        whole["branch"] = replace(branch, end=branch.end + 1, text=branch.text + "\n")
    elif change == "sibling":
        whole["branch"] = replace(branch, required_principal_output="income")
    elif change == "duplicate":
        whole["boundary_obligations"] = tuple(whole["boundary_obligations"]) * 2
    elif change == "missing":
        whole["boundary_obligations"] = ()
    elif change == "stale":
        whole["content"] += "\n"
    else:
        whole["formula_environment"]["threshold_a"] = 0
    assert not group.witnesses(**whole)


def test_duplicate_obligations_compete_for_one_resource(whole):
    branch = whole["branch"]
    second = replace(branch, label="duplicate")
    witnesses = group.witnesses(**whole)
    assert witnesses
    missing = s._unmatched_evidence_obligations({branch: witnesses, second: witnesses})
    assert len(missing) == 1


def test_actual_analyzer_allocates_only_complete_group(whole):
    from axiom_encode.harness.validator_pipeline import (
        extract_named_scalar_occurrences,
        numeric_value_is_grounded,
    )

    def run(evidence):
        return s.analyze_complete_source_unit(
            whole["content"],
            whole["source_context"][whole["corpus_citation_path"]],
            corpus_citation_path=whole["corpus_citation_path"],
            test_cases=whole["asserted_by_rule"]["repayment"],
            extract_numeric_occurrences=functools.partial(
                extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
            ),
            extract_named_scalars=extract_named_scalar_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
            imported_export_bindings=whole["imported_exports"],
            imported_proof_source_context=whole["source_context"],
            source_context=whole["source_context"],
            execution_evidence=evidence,
        )

    before = [
        i
        for i in run(None).issues
        if "Social benefits repayment" in i and "paired" in i
    ]
    after = [
        i
        for i in run(whole["evidence"]).issues
        if "Social benefits repayment" in i and "paired" in i
    ]
    assert before and not after


@pytest.mark.parametrize(
    "change",
    [
        "extra-change",
        "unrelated-bool",
        "inactive-worksheet",
        "missing-bool",
        "missing-income",
        "missing-trace",
        "assertion-list",
        "wrong-unit",
        "wrong-year",
        "wrong-person",
        "stale-case",
        "malformed-response",
    ],
)
def test_each_pair_remains_bound_to_its_own_execution(whole, change):
    import json

    from axiom_encode.harness import same_case_execution as e
    from tests import test_imported_export_evidence as observations

    assert group.witnesses(**whole)
    cases = whole["asserted_by_rule"]["repayment"]
    evidence = whole["evidence"]
    captures = list(evidence.records)
    root = evidence.root_relative.removesuffix(".yaml").replace("/", ":", 1)
    provider = observations.URI.rsplit("#", 1)[0]
    if change == "extra-change":
        cases[1]["input"][root + "#input.deduction"] = 1
    elif change == "unrelated-bool":
        cases[1]["input"][provider + "#input.third_fact"] = True
    elif change == "inactive-worksheet":
        cases[0]["input"][root + "#input.completed"] = 0
    elif change == "missing-bool":
        cases[0]["input"].pop(provider + "#input.third_fact")
    elif change == "missing-income":
        cases[0]["input"].pop(root + "#input.base_income")
    elif change == "missing-trace":
        response = json.loads(captures[0].response_json)
        response["results"][0]["trace"].pop(root + "#income")
        captures[0] = replace(captures[0], response_json=json.dumps(response))
    elif change == "assertion-list":
        cases[0]["output"] = [cases[0]["output"]]
    elif change == "wrong-unit":
        artifact = json.loads(evidence.artifact_bytes)
        artifact["program"]["derived"][0]["unit"] = "USD"
        evidence = replace(evidence, artifact_bytes=json.dumps(artifact).encode())
    elif change == "wrong-year":
        cases[0]["period"]["start"] = "2030-01-01"
    elif change == "wrong-person":
        cases[0]["entity_id"] = "bob"
    elif change == "stale-case":
        cases[0]["name"] = "changed"
        assert not group.witnesses(**whole)
        return
    else:
        captures[0] = replace(captures[0], response_json="[")
    whole["evidence"] = replace(
        evidence,
        records=tuple(
            replace(r, case_sha256=e.case_key(c)) for r, c in zip(captures, cases)
        ),
    )
    assert not group.witnesses(**whole)


@pytest.mark.parametrize(
    "change", ["and", "nonselected", "default", "extra-alternative"]
)
def test_source_changes_cannot_keep_old_group_coverage(whole, change):
    import yaml

    from tests import test_imported_export_evidence as observations

    payload = yaml.safe_load(whole["content"])
    owner = payload["rules"][0]
    if change == "and":
        owner["versions"][0]["formula"] = owner["versions"][0]["formula"].replace(
            " or ", " and "
        )
    elif change == "nonselected":
        owner["versions"].append(
            dict(effective_from="2032-01-01", effective_to="2032-12-31", formula="0")
        )
    elif change == "default":
        payload["inputs"][0]["default"] = 0
    else:
        owner["versions"][0]["formula"] = owner["versions"][0]["formula"].replace(
            "if ", "if True or ", 1
        )
    content = observations.dump(payload)
    whole["content"] = content
    whole["principal_rules"] = {
        r["name"]: r for r in payload["rules"] if r["kind"] == "derived"
    }
    evidence = whole["evidence"]
    root = evidence.files[0]
    root.original.write_text(content)
    (evidence.snapshot_root / root.relative).write_text(content)
    whole["evidence"] = replace(
        evidence, files=(replace(root, raw=content.encode()), *evidence.files[1:])
    )
    assert whole["evidence"].bound(content, whole["imported_exports"])
    assert not group.witnesses(**whole)


def test_partial_leaf_and_advisory_obligations_remain_unresolved(whole):
    branch = whole["branch"]
    offset = branch.text.index("- You")
    end = branch.text.index("\n", offset)
    leaf = replace(
        branch,
        text=branch.text[offset:end],
        start=branch.start + offset,
        end=branch.start + end,
    )
    source = whole["source_text"]
    start = source.index("Line 56")
    advisory = replace(branch, text=source[start:], start=start, end=len(source))
    obligations = {}
    for candidate in (branch, leaf, advisory):
        obligations[candidate] = group.witnesses(**{**whole, "branch": candidate})
    assert set(s._unmatched_evidence_obligations(obligations)) == {leaf, advisory}
