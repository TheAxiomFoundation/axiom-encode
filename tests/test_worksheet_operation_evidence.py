from __future__ import annotations

import functools
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_path_evidence as paths
from axiom_encode.harness import validator_pipeline as v
from axiom_encode.harness import worksheet_operation_evidence as operations

FIXTURE = Path(__file__).parent / "fixtures/worksheet_operations"
SOURCE = (FIXTURE.parent / "source_path_shadow/source.txt").read_text()
SECONDARY = (FIXTURE / "secondary.txt").read_text()
CONTEXT = {operations._PRIMARY: SOURCE, operations._SECONDARY: SECONDARY}


def inputs():
    return yaml.safe_load((FIXTURE / "candidate.yaml").read_text()), yaml.safe_load(
        (FIXTURE / "cases.yaml").read_text()
    )


@pytest.fixture(scope="module")
def certificate():
    p, c = inputs()
    return paths.certify_worksheet_paths(
        p,
        source_text=SOURCE,
        corpus_citation_path=operations._PRIMARY,
        test_cases=c,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        extract_numeric_occurrences=functools.partial(
            v.extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
    )


def test_actual_five_outputs_have_independent_annual_operations(certificate):
    p, c = inputs()
    result = operations.certify_worksheet_operations(
        p,
        source_text=SOURCE,
        corpus_citation_path=operations._PRIMARY,
        source_context=CONTEXT,
        test_cases=c,
        paths_evidence=certificate,
    )
    assert not result.unresolved
    assert result.annual_spans == (
        (212, 216),
        (3109, 3113),
        (3676, 3680),
        (4269, 4273),
        (5337, 5341),
    )
    assert result.outputs == operations._OUTPUTS
    assert len(result.case_links) == 223
    assert result.cap_span is not None


def certify(payload, cases, context=None, source=SOURCE):
    path_certificate = paths.certify_worksheet_paths(
        payload,
        source_text=source,
        corpus_citation_path=operations._PRIMARY,
        test_cases=cases,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        extract_numeric_occurrences=functools.partial(
            v.extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
    )
    return operations.certify_worksheet_operations(
        payload,
        source_text=source,
        corpus_citation_path=operations._PRIMARY,
        source_context=CONTEXT if context is None else context,
        test_cases=cases,
        paths_evidence=path_certificate,
    )


def named(payload, name):
    return next(r for r in payload["rules"] if r["name"] == name)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_line1",
        "reverse_subtract",
        "floor_excess",
        "inverted_ratio",
        "wrong_net_dispatch",
        "missing_deduction",
        "wrong_deduction_sign",
        "wrong_net_secondary_owner",
        "wrong_quebec",
        "mb_without_amt",
        "wrong_equality_operand",
        "inclusive_equality",
        "no_cap",
        "wrong_cap",
        "partial_public",
        "superseded_helper",
        "bare_date",
        "wrong_count_dtype",
        "wrong_unit",
        "wrong_case",
        "missing_public_assertions",
        "missing_eq_pair",
        "no_cap_case",
    ],
)
def test_actual_candidate_and_case_mutations_never_commit_annual_evidence(mutation):
    p, c = inputs()
    target = None
    old = new = None
    if mutation == "wrong_line1":
        target, old, new = operations._LINE1, operations._PAID, operations._CREDIT
    elif mutation == "reverse_subtract":
        target = operations._LINE3
        r = named(p, target)
        r["versions"][0]["formula"] = operations._LINE2 + " - " + operations._LINE1
    elif mutation == "floor_excess":
        target = operations._LINE3
        r = named(p, target)
        r["versions"][0]["formula"] = "max(0, " + r["versions"][0]["formula"] + ")"
    elif mutation == "inverted_ratio":
        target, old, new = (
            operations._LINE4,
            "t2209_net_foreign_non_business_income_line_2",
            "t2209_net_income_line_2",
        )
    elif mutation == "wrong_net_dispatch":
        target, old, new = (
            "t2036_net_income_for_credit",
            "jurisdictions_tax_paid_count",
            "jurisdictions_tax_payable_count",
        )
    elif mutation in {"missing_deduction", "wrong_deduction_sign"}:
        target = "net_income_for_multiple_jurisdictions"
        old = "- deductible_security_options_24900"
        new = (
            ""
            if mutation == "missing_deduction"
            else "+ deductible_security_options_24900"
        )
    elif mutation == "wrong_net_secondary_owner":
        owner = named(p, "net_income_for_multiple_jurisdictions")
        atoms = owner["metadata"]["proof"]["atoms"]
        moved = [
            a
            for a in atoms
            if a["source"]["corpus_citation_path"] == operations._SECONDARY
        ]
        assert moved
        owner["metadata"]["proof"]["atoms"] = [a for a in atoms if a not in moved]
        named(p, operations._LINE3)["metadata"]["proof"]["atoms"].extend(moved)
    elif mutation == "wrong_quebec":
        target, old, new = "resident_of_quebec_at_year_end", '"Quebec"', '"Ontario"'
        # Fixture uses a single-quoted source literal.
        old = (
            "'Quebec'" if old not in named(p, target)["versions"][0]["formula"] else old
        )
        new = "'Ontario'" if old.startswith("'") else new
    elif mutation == "mb_without_amt":
        target, old, new = operations._LINE5, "minimum_tax_is_payable and ", ""
    elif mutation == "wrong_equality_operand":
        target, old, new = operations._LINE5, operations._CREDIT, operations._LINE2
    elif mutation == "inclusive_equality":
        target, old, new = operations._LINE5, " == ", " >= "
    elif mutation in {"no_cap", "wrong_cap"}:
        target = operations._LINE5
        old = ",\n    provincial_or_territorial_tax_otherwise_payable"
        formula = named(p, target)["versions"][0]["formula"]
        # Preserve actual indentation in the candidate rather than guessing it.
        import re

        match = re.search(
            r",\s+provincial_or_territorial_tax_otherwise_payable", formula
        )
        assert match
        old = match.group()
        new = "" if mutation == "no_cap" else ", 999999"
    elif mutation == "partial_public":
        named(p, operations._LINE1)["versions"][0]["effective_to"] = "2025-06-30"
    elif mutation == "superseded_helper":
        import copy

        r = named(p, "net_income_for_multiple_jurisdictions")
        extra = copy.deepcopy(r["versions"][0])
        extra["effective_from"] = "2025-07-01"
        r["versions"].append(extra)
    elif mutation == "bare_date":
        for a in named(p, operations._LINE1)["metadata"]["proof"]["atoms"]:
            if a["path"].endswith("effective_to"):
                a["source"]["excerpt"] = "for 2025"
    elif mutation in {"wrong_count_dtype", "wrong_unit"}:
        name = (
            "jurisdictions_tax_paid_count"
            if mutation == "wrong_count_dtype"
            else operations._PAID
        )
        item = next(i for i in p["inputs"] if i["name"] == name)
        if mutation == "wrong_count_dtype":
            item["dtype"] = "Decimal"
        else:
            item["unit"] = "USD"
    elif mutation == "wrong_case":
        c[-1]["output"][
            next(k for k in c[-1]["output"] if k.endswith("#" + operations._LINE5))
        ] += 1
    elif mutation == "missing_public_assertions":
        for case in c:
            case["output"] = {
                k: v
                for k, v in case["output"].items()
                if not k.endswith("#" + operations._LINE1)
            }
    elif mutation == "missing_eq_pair":
        c = [x for x in c if x["name"] != "federal_equality_pair_b"]
    elif mutation == "no_cap_case":
        c = [x for x in c if x["name"] != "principal_tax_cap"]
    if target is not None and old is not None:
        r = named(p, target)
        before = r["versions"][0]["formula"]
        assert old in before
        r["versions"][0]["formula"] = before.replace(old, new)
    result = certify(p, c)
    assert result.unresolved
    assert result.annual_spans == () and result.cap_span is None


@pytest.mark.parametrize("kind", ["missing", "quoted", "wrong", "wrong_type"])
def test_secondary_context_is_required_and_never_borrowed(kind):
    p, c = inputs()
    context = dict(CONTEXT)
    if kind == "missing":
        context.pop(operations._SECONDARY)
    elif kind == "wrong_type":
        context[operations._SECONDARY] = {"body": SECONDARY}
    elif kind == "quoted":
        context[operations._SECONDARY] = '"' + SECONDARY + '"'
    else:
        context[operations._SECONDARY] = "Unrelated form."
    result = certify(p, c, context)
    assert result.unresolved and not result.annual_spans


def test_stale_two_owner_certificate_cannot_validate_changed_atomic_result(certificate):
    p, c = inputs()
    named(p, operations._LINE2)["versions"][0]["formula"] = "0"
    result = operations.certify_worksheet_operations(
        p,
        source_text=SOURCE,
        corpus_citation_path=operations._PRIMARY,
        source_context=CONTEXT,
        test_cases=c,
        paths_evidence=certificate,
    )
    assert result.unresolved == ("stale candidate/case path certificate",)
    assert not result.annual_spans


def analyze(payload, cases, *, context=CONTEXT, source=SOURCE):
    from axiom_encode.harness import source_completeness as sc

    extract = functools.partial(
        v.extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
    )
    return sc.analyze_complete_source_unit(
        yaml.safe_dump(payload),
        source,
        corpus_citation_path=operations._PRIMARY,
        source_context=context,
        test_cases=cases,
        extract_numeric_occurrences=extract,
        extract_numeric_grounding_occurrences=extract,
        extract_named_scalars=v.extract_named_scalar_occurrences,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
    )


def test_actual_analyzer_discharge_is_local_and_transfer_remains():
    p, c = inputs()
    issues = analyze(p, c).issues
    assert not any("2025 has no named scalar" in x for x in issues)
    assert any("Enter the total from line 5" in x for x in issues)
    assert any(
        "If you have to pay tax to more than one jurisdiction" in x for x in issues
    )
    assert not any("The amount on line 5 should not be more" in x for x in issues)


@pytest.mark.parametrize(
    "mutation", ["missing_context", "partial_public", "extra_cap_claimant"]
)
def test_actual_analyzer_failed_certificate_never_hides_annual_or_other_owner(mutation):
    import copy

    p, c = inputs()
    context = CONTEXT
    if mutation == "missing_context":
        context = None
    elif mutation == "partial_public":
        named(p, operations._LINE1)["versions"][0]["effective_to"] = "2025-06-30"
    else:
        extra = copy.deepcopy(named(p, operations._LINE5))
        extra["name"] = "other_claimed_credit"
        extra["versions"][0]["formula"] = "0"
        p["rules"].append(extra)
    issues = analyze(p, c, context=context).issues
    assert any("2025 has no named scalar" in x for x in issues)
    assert any("The amount on line 5 should not be more" in x for x in issues)


def test_selectively_quoted_header_rejects_even_with_updated_raw_source_identity():
    import hashlib

    p, c = inputs()
    intro_start = SOURCE.index("Use this form")
    intro_end = SOURCE.index("2025", intro_start) + 4
    source = (
        SOURCE[:intro_start]
        + '"\n'
        + SOURCE[intro_start:intro_end]
        + '"'
        + SOURCE[intro_end:]
    )
    p["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        source.encode()
    ).hexdigest()
    context = {**CONTEXT, operations._PRIMARY: source}
    assert operations._annual_intro(source) is None
    result = certify(p, c, context, source)
    assert not result.annual_spans
    assert result.unresolved != ("raw source identity mismatch",)


def test_duplicate_year_as_new_monetary_operand_is_not_masked_by_value():
    import hashlib

    p, c = inputs()
    marker = "Country or countries for which you are making this claim:"
    assert marker in SOURCE
    source = SOURCE.replace(marker, marker + "\nReference monetary amount: 2025.")
    p["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        source.encode()
    ).hexdigest()
    context = {**CONTEXT, operations._PRIMARY: source}
    result = certify(p, c, context, source)
    monetary_start = source.index("2025", source.index("Reference monetary"))
    assert (monetary_start, monetary_start + 4) not in result.annual_spans
    assert any(
        "2025 has no named scalar" in x
        for x in analyze(p, c, context=context, source=source).issues
    )


@pytest.mark.parametrize(
    "tail",
    ["Only eligible residents may use this amount.\n", "Subtract another deduction.\n"],
)
def test_secondary_operation_cannot_ignore_new_normative_note_tail(tail):
    p, c = inputs()
    context = dict(CONTEXT)
    context[operations._SECONDARY] = SECONDARY.replace(
        "T2209 E (25) Page 4 of 5", tail + "T2209 E (25) Page 4 of 5"
    )
    result = certify(p, c, context)
    assert result.unresolved == ("secondary note boundary unresolved",)
    assert not result.annual_spans and result.cap_span is None


def test_normal_pipeline_forwards_resolved_context_without_primary_concatenation(
    monkeypatch, tmp_path
):
    from types import SimpleNamespace

    observed = []

    def capture(content, source, **kwargs):
        observed.append((source, kwargs["source_context"]))
        return SimpleNamespace(issues=("retained diagnostic",))

    monkeypatch.setattr(v, "analyze_complete_source_unit", capture)
    pipeline = v.ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-ca",
        axiom_rules_path=tmp_path / "engine",
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
    )
    payload, cases = inputs()
    for context in (CONTEXT, None):
        assert pipeline._complete_source_unit_issues(
            yaml.safe_dump(payload),
            validation_source_texts={operations._PRIMARY: SOURCE},
            proof_source_texts=context,
            test_cases=cases,
        ) == ["retained diagnostic"]
    assert observed == [(SOURCE, CONTEXT), (SOURCE, None)]
