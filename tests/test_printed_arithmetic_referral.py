"""Verified whole-grid URL typography is never caller-provided arithmetic masking."""

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_completeness as s
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)
from axiom_encode.printed_source_rows import recognize_printed_rows

P = Path(__file__).parent / "fixtures/form-numeric-locators/step-3.txt"
CITATION = "ca/policy/test/step-3"


def world(text=None):
    text = P.read_text().strip() if text is None else text
    context = s._printed_operation_context(text, (), CITATION)
    branches = s._source_formula_branches(
        text,
        corpus_citation_path=CITATION,
        branches=(),
        active_branches=(),
        deferred_paths=set(),
    )
    branch = next(b for b in branches if b.text.startswith("Line 56"))
    return text, context, branch


def bound_world(formula="max(0,a-b)"):
    text, context, branch = world()
    header = "ca/policy/test/primary-document"
    left = next(line for line in text.splitlines() if line.startswith("Line 54"))
    right = (
        text[text.index("Social benefits repayment:") : text.index("\n\nLine 56")]
        + "\n\n"
    )
    own = branch.text[: branch.text.index(")") + 1]
    rules = {}
    for name, formula_text, proof in [
        ("net", formula, own),
        ("a", "x", left),
        ("b", "y", right),
    ]:
        atoms = [
            {
                "path": "versions[0].formula",
                "kind": "formula",
                "source": {"corpus_citation_path": CITATION, "excerpt": proof},
            }
        ]
        for field in ("effective_from", "effective_to"):
            atoms.append(
                {
                    "path": "versions[0]." + field,
                    "kind": "effective_period",
                    "source": {"corpus_citation_path": header, "excerpt": "T1 2025"},
                }
            )
        rules[name] = {
            "name": name,
            "kind": "derived",
            "dtype": "Money",
            "unit": "CAD",
            "entity": "Person",
            "period": "Year",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": formula_text,
                }
            ],
            "metadata": {"proof": {"atoms": atoms}},
        }
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "units": [{"name": "CAD", "kind": "currency", "minor_units": 2}],
        "inputs": [
            {
                "name": n,
                "dtype": "Money",
                "unit": "CAD",
                "entity": "Person",
                "period": "Year",
            }
            for n in ("x", "y")
        ],
        "rules": list(rules.values()),
    }
    context = replace(
        context,
        consumer_content=yaml.safe_dump(payload),
        proof_sources=((CITATION, text), (header, "T1 2025\n")),
    )
    case = {
        "name": "actual",
        "period": "2025",
        "input": {"x": 10, "y": 3},
        "output": {"a": 10, "b": 3, "net": 7},
    }
    return text, context, branch, rules, case


def match(formula, text, context, branch, kinds=None):
    _, _, _, rules, case = bound_world()
    execution = s._FormulaExecution((), formula, None, False, {})
    return s._formula_execution_matches_source_branch(
        execution,
        branch,
        interval=None,
        formula_environment={},
        execution_environment={"a": 10, "b": 3},
        extract_numeric_occurrences=extract_typed_numeric_occurrences_from_text,
        numeric_value_is_grounded=numeric_value_is_grounded,
        printed_operations=context,
        source_operation_kinds=kinds,
        source_text=text,
        corpus_citation_path=CITATION,
        source_rule=rules["net"],
        principal_rules=rules,
        case=case,
    )


def test_actual_projection_is_only_url_and_preserves_raw_source():
    text, context, branch, _, _ = bound_world()
    raw = branch.text
    projection = context.operation_projection(
        branch, source_text=text, citation_path=CITATION
    )
    assert projection is not None
    assert len(projection.operation_text) == len(raw)
    ((a, b),) = projection.numeric_spans
    assert raw[a:b] == "canada.ca/line-23600"
    assert projection.operation_text[:a] == raw[:a]
    assert projection.operation_text[b:] == raw[b:]
    assert branch.text == raw == text[branch.start : branch.end]
    assert "may have a non-capital loss" in projection.operation_text
    assert s._formula_operation_kinds(raw) == {"divide", "subtract"}
    assert s._formula_operation_kinds(projection.operation_text) == {"subtract"}
    assert not match("max(0, a - b)", text, None, branch)
    assert match("max(0, a - b)", text, context, branch)


@pytest.mark.parametrize("formula", ["a+b", "a*b", "a/b", "7", "0"])
def test_wrong_arithmetic_cannot_be_cured_by_referral_projection(formula):
    assert not match(formula, *world())


@pytest.mark.parametrize(
    "change",
    ["structure", "source", "citation", "offset", "clipped", "conflict", "inherited"],
)
def test_forged_or_mismatched_context_does_not_project(change):
    text, context, branch = world()
    if change == "structure":
        context = replace(
            context,
            structure=replace(
                context.structure,
                arithmetic_referral_spans=((branch.start, branch.end),),
            ),
        )
    elif change == "source":
        context = replace(
            context, source_text=text.replace("Net income =", "Other income =")
        )
    elif change == "citation":
        context = replace(context, citation_path="ca/policy/foreign")
    elif change == "offset":
        branch = replace(branch, start=branch.start + 1)
    elif change == "clipped":
        text = branch.text
        branch = replace(branch, start=0, end=len(text))
        context = replace(context, source_text=text)
    elif change == "conflict":
        conflict = s.SourceStructureBranch(
            (), "explicit", "conflict", "x", branch.start + 10, branch.start + 15
        )
        context = replace(context, source_branches=(conflict,))
    else:
        branch = replace(branch, computation_chapeaux=((0, 3, "Add"),))
    assert (
        context.operation_projection(branch, source_text=text, citation_path=CITATION)
        is None
    )
    assert not match("max(0,a-b)", text, context, branch)


@pytest.mark.parametrize(
    "replacement",
    [
        "(go to canada.ca/line-23600",
        "((go to canada.ca/line-23600))",
        '(go to "canada.ca/line-23600")',
        "(discussion (go to canada.ca/line-23600)",
        "(go to canada.ca/line-23600))",
        "(go to canada.ca/line-23600 plus 4)",
        "(go to canada.ca/line-23600 dollars)",
        "(go to canada..ca/line-23600)",
        "(go to canada.ca/line-23600/)",
        "(go to income/expenses)",
    ],
)
def test_malformed_or_quoted_referral_adds_no_span(replacement):
    text = P.read_text().strip().replace("(go to canada.ca/line-23600)", replacement)
    structure = recognize_printed_rows(text)
    assert structure is None or not structure.arithmetic_referral_spans


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "reversed", "unknown", "heading", "extra-amount"]
)
def test_incomplete_or_changed_operational_row_does_not_project(change):
    text = P.read_text().strip()
    if change == "missing":
        text = "\n".join(
            line for line in text.splitlines() if not line.startswith("Line 54")
        )
    elif change == "duplicate":
        row = next(line for line in text.splitlines() if line.startswith("Line 56"))
        text += "\n" + row
    elif change == "reversed":
        text = text.replace("Line 54 minus line 55", "Line 55 minus line 54")
    elif change == "unknown":
        text = text.replace("Line 54 minus line 55", "Line 98 minus line 99")
    elif change == "heading":
        text = text.replace("Step 3 - Net income", "Unknown heading")
    else:
        text = text.replace("Net income = ^", "Net income plus 4 dollars = ^")
    structure = recognize_printed_rows(text)
    assert structure is None or not structure.arithmetic_referral_spans


def test_real_neighboring_amount_and_operations_remain_obligations():
    text, context, branch = world(
        P.read_text()
        .strip()
        .replace("Other deductions ^", "Other deductions plus 23600 dollars ^")
    )
    projection = context.operation_projection(
        branch, source_text=text, citation_path=CITATION
    )
    assert projection is not None
    assert "23600 dollars" in text
    others = s._source_formula_branches(
        text,
        corpus_citation_path=CITATION,
        branches=(),
        active_branches=(),
        deferred_paths=set(),
    )
    neighbor = next(b for b in others if "23600 dollars" in b.text)
    assert (
        context.operation_projection(neighbor, source_text=text, citation_path=CITATION)
        is None
    )
    assert any(
        o.value == 23600
        for o in s._formula_branch_computation_occurrences(
            neighbor,
            interval=None,
            extract_numeric_occurrences=extract_typed_numeric_occurrences_from_text,
        )
    )
    assert s._formula_operation_kinds("income / expenses") == {"divide"}
    assert s._formula_operation_kinds("income - expenses") == {"subtract"}


def witness_kwargs(formula="max(0,a-b)"):
    text, context, branch, rules, case = bound_world(formula)
    kwargs = dict(
        corpus_citation_path=CITATION,
        principal_rules=rules,
        rule_names={"net"},
        asserted_by_rule={"net": [case]},
        extract_numeric_occurrences=extract_typed_numeric_occurrences_from_text,
        numeric_value_is_grounded=numeric_value_is_grounded,
        formula_environment={},
        dependency_cache={},
        execution_cache={},
        source_text=text,
        printed_operations=context,
    )
    return branch, kwargs


def test_existing_witness_orchestration_uses_the_verified_projection():
    branch, kwargs = witness_kwargs()
    assert s._formula_branch_test_witnesses(branch, **kwargs)
    kwargs.update(printed_operations=None, dependency_cache={}, execution_cache={})
    assert not s._formula_branch_test_witnesses(branch, **kwargs)


@pytest.mark.parametrize(
    "formula,expected",
    [
        ("max(0,b-a)", 0),
        ("max(0,a-a)", 0),
        ("max(0,a+b)", 13),
        ("a-b", 7),
        ("max(a-b,0)", 7),
        ("max(0,a/b)", 10 / 3),
        ("7", 7),
    ],
)
def test_source_owned_reversed_or_wrong_operands_with_correct_outputs_refuse(
    formula, expected
):
    branch, kwargs = witness_kwargs(formula)
    kwargs["asserted_by_rule"]["net"][0]["output"]["net"] = expected
    assert not s._formula_branch_test_witnesses(branch, **kwargs)


@pytest.mark.parametrize(
    "kinds,accepted",
    [
        (frozenset({"subtract"}), True),
        (frozenset({"subtract", "divide"}), False),
        (frozenset({"add"}), False),
    ],
)
def test_precomputed_operation_kinds_cannot_override_verified_source(kinds, accepted):
    assert match("max(0,a-b)", *bound_world()[:3], kinds=kinds) is accepted
    assert not match("max(0,a+b)", *world(), kinds=kinds)


def test_correct_owner_cannot_authenticate_reversed_reached_execution():
    text, context, branch, _, _ = bound_world()
    assert match("max(0,a-b)", text, context, branch)
    assert not match("max(0,b-a)", text, context, branch)
    assert not match("max(0,a-b)+0", text, context, branch)
    assert not match("max(0,a-b)", text, replace(context, consumer_content=""), branch)


def rebind(kwargs, payload):
    kwargs["principal_rules"] = {r["name"]: r for r in payload["rules"]}
    kwargs["printed_operations"] = replace(
        kwargs["printed_operations"], consumer_content=yaml.safe_dump(payload)
    )


@pytest.mark.parametrize(
    "change",
    [
        "short-owner-proof",
        "wrong-owner-proof",
        "short-right-proof",
        "wrong-left-proof",
        "missing-date-proof",
        "wrong-frame-family",
        "unit",
        "entity",
        "period",
        "input-owner",
        "missing-unit",
        "alias-collision",
        "malformed-import",
        "duplicate-rule",
        "bad-other-version",
        "wrong-year",
        "missing-period",
        "stale-rule",
        "source-context-change",
        "missing-header",
        "null-versions",
        "null-source",
        "null-imports",
        "wrong-primary",
        "duplicate-yaml",
    ],
)
def test_local_ordered_source_binding_refusals(change):
    branch, kwargs = witness_kwargs()
    context = kwargs["printed_operations"]
    payload = yaml.safe_load(context.consumer_content)
    rules = {r["name"]: r for r in payload["rules"]}
    if change == "short-owner-proof":
        rules["net"]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = "Line 56"
    elif change == "wrong-owner-proof":
        rules["net"]["metadata"]["proof"]["atoms"][0]["source"][
            "corpus_citation_path"
        ] = "ca/policy/foreign/step-3"
    elif change == "short-right-proof":
        rules["b"]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
            'Line 55 and line 23500: If not, enter "0" on line 23500. ^'
        )
    elif change == "wrong-left-proof":
        rules["a"]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
            "Line 53 and line 23300: Add lines 35 to 52. = ^"
        )
    elif change == "missing-date-proof":
        rules["a"]["metadata"]["proof"]["atoms"].pop()
    elif change == "wrong-frame-family":
        sources = dict(context.proof_sources)
        sources["ca/policy/foreign/header"] = "T1 2025\n"
        kwargs["printed_operations"] = replace(
            context, proof_sources=tuple(sources.items())
        )
        for rule in rules.values():
            for atom in rule["metadata"]["proof"]["atoms"][1:]:
                atom["source"]["corpus_citation_path"] = "ca/policy/foreign/header"
    elif change in ("unit", "entity", "period"):
        rules["a"][change] = {"unit": "USD", "entity": "Family", "period": "Month"}[
            change
        ]
    elif change == "input-owner":
        rules["a"]["kind"] = "input"
    elif change == "missing-unit":
        payload["units"] = []
    elif change == "alias-collision":
        payload["imports"] = ["ca:policies/foreign/module#a"]
    elif change == "malformed-import":
        payload["imports"] = ["ca:policies/test/../bad#name"]
    elif change == "duplicate-rule":
        payload["rules"].append(copy.deepcopy(rules["a"]))
    elif change == "bad-other-version":
        sources = dict(context.proof_sources)
        sources["ca/policy/test/primary-document"] = "T1 2025\nT1 2026\n"
        kwargs["printed_operations"] = replace(
            context, proof_sources=tuple(sources.items())
        )
        for rule in rules.values():
            new = copy.deepcopy(rule["versions"][0])
            new.update(effective_from="2026-01-01", effective_to="2026-12-31")
            if rule["name"] == "net":
                new["formula"] = "max(0,b-a)"
            rule["versions"].append(new)
            atoms = copy.deepcopy(rule["metadata"]["proof"]["atoms"])
            for atom in atoms:
                atom["path"] = atom["path"].replace("[0]", "[1]")
                if atom["kind"] == "effective_period":
                    atom["source"]["excerpt"] = "T1 2026"
            rule["metadata"]["proof"]["atoms"] += atoms
    elif change == "wrong-year":
        kwargs["asserted_by_rule"]["net"][0]["period"] = "2026"
    elif change == "missing-period":
        kwargs["asserted_by_rule"]["net"][0].pop("period")
    elif change == "stale-rule":
        kwargs["principal_rules"]["a"]["description"] = "changed current declaration"
        assert not s._formula_branch_test_witnesses(branch, **kwargs)
        return
    elif change == "source-context-change":
        sources = dict(context.proof_sources)
        sources[CITATION] = sources[CITATION].replace(
            "maximum $1,074.00", "maximum $1,075.00"
        )
        kwargs["printed_operations"] = replace(
            context, proof_sources=tuple(sources.items())
        )
    elif change == "missing-header":
        kwargs["printed_operations"] = replace(
            context, proof_sources=((CITATION, kwargs["source_text"]),)
        )
    elif change == "null-versions":
        rules["a"]["versions"] = None
    elif change == "null-source":
        rules["a"]["metadata"]["proof"]["atoms"][0]["source"] = None
    elif change == "null-imports":
        payload["imports"] = None
    elif change == "wrong-primary":
        payload["module"]["source_verification"]["corpus_citation_path"] = (
            "ca/policy/foreign"
        )
    elif change == "duplicate-yaml":
        kwargs["printed_operations"] = replace(
            context, consumer_content=context.consumer_content + "\nrules: []\n"
        )
        assert not s._formula_branch_test_witnesses(branch, **kwargs)
        return
    rebind(kwargs, payload)
    assert not s._formula_branch_test_witnesses(branch, **kwargs)


@pytest.mark.parametrize("year", [2025, 2026])
def test_all_versions_with_exact_year_header_proofs_accept(year):
    branch, kwargs = witness_kwargs()
    context = kwargs["printed_operations"]
    payload = yaml.safe_load(context.consumer_content)
    sources = dict(context.proof_sources)
    sources["ca/policy/test/primary-document"] = "T1 2025\nT1 2026\n"
    kwargs["printed_operations"] = replace(
        context, proof_sources=tuple(sources.items())
    )
    for rule in payload["rules"]:
        version = copy.deepcopy(rule["versions"][0])
        version.update(effective_from="2026-01-01", effective_to="2026-12-31")
        rule["versions"].append(version)
        atoms = copy.deepcopy(rule["metadata"]["proof"]["atoms"])
        for atom in atoms:
            atom["path"] = atom["path"].replace("[0]", "[1]")
            if atom["kind"] == "effective_period":
                atom["source"]["excerpt"] = "T1 2026"
        rule["metadata"]["proof"]["atoms"] += atoms
    rebind(kwargs, payload)
    kwargs["asserted_by_rule"]["net"][0]["period"] = str(year)
    assert s._formula_branch_test_witnesses(branch, **kwargs)
