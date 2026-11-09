"""A complete applicability recipient owns geography, polarity and annual scope."""

from dataclasses import replace
from pathlib import Path

import pytest

from axiom_encode.harness import source_completeness as c

SOURCE = (Path(__file__).parent / "fixtures/t2036_operand_gates/source.txt").read_text()
CITATION = "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit"
EXCLUSION = "This form does not apply to residents of Quebec."
INTRO = "Use this form to calculate the foreign non-business income tax credit for 2025 that you can deduct from the income tax"
SELECTOR = "resident_of_quebec_at_year_end"


def fixture():
    rule = {
        "name": "credit",
        "kind": "derived",
        "dtype": "Money",
        "entity": "Person",
        "versions": [
            {
                "effective_from": "2025-01-01",
                "effective_to": "2025-12-31",
                "formula": f"if {SELECTOR}: 0 else: 1600",
            }
        ],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {
                            "corpus_citation_path": CITATION,
                            "excerpt": EXCLUSION,
                        },
                    },
                    *[
                        {
                            "path": "versions[0]." + field,
                            "kind": "definition",
                            "source": {
                                "corpus_citation_path": CITATION,
                                "excerpt": INTRO,
                            },
                        }
                        for field in ("effective_from", "effective_to")
                    ],
                ]
            }
        },
    }
    cases = [
        {
            "name": "ordinary",
            "period": {
                "start": "2025-01-01",
                "end": "2025-12-31",
                "period_kind": "tax_year",
            },
            "input": {SELECTOR: False},
            "output": {"credit": 1600},
        },
        {
            "name": "excluded",
            "period": {
                "start": "2025-01-01",
                "end": "2025-12-31",
                "period_kind": "tax_year",
            },
            "input": {SELECTOR: True},
            "output": {"credit": 0},
        },
    ]
    return rule, cases


def branch(source, exclusion=EXCLUSION):
    start = source.index(exclusion)
    return c.SourceStructureBranch(
        (), "exception-clause", "root", exclusion, start, start + len(exclusion)
    )


def actual_witnesses(rule, cases):
    return c._toggled_formula_boolean_selectors(
        {"credit": rule}, asserted_by_rule={"credit": cases}, formula_environment={}
    )


def collect(rule, cases, source=SOURCE, exclusion=EXCLUSION, witnesses=None):
    return c._exception_witnesses_for_branch(
        branch(source, exclusion),
        source_text=source,
        corpus_citation_path=CITATION,
        principal_rules={"credit": rule},
        principal_rule_paths={"credit": {()}},
        asserted_by_rule={"credit": cases},
        toggled_exception_selectors=actual_witnesses(rule, cases)
        if witnesses is None
        else witnesses,
        extract_numeric_occurrences=lambda _text: (),
    )


def test_actual_callsite_matches_positive_residence_not_negated_application():
    rule, cases = fixture()
    found = collect(rule, cases)
    assert len(found) == 1
    assert next(iter(found)).active_value is True
    assert next(iter(found)).zeroes
    assert rule["versions"][0]["formula"] == f"if {SELECTOR}: 0 else: 1600"


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_place",
        "restriction",
        "spouse",
        "bare_date",
        "wrong_date_citation",
        "wrong_formula_citation",
        "wrong_version_path",
        "supersession",
        "wrong_year",
        "different_period",
        "missing_period",
        "wrong_output",
        "missing_formula_proof",
    ],
)
def test_actual_callsite_rejects_unowned_or_unproved_context(mutation):
    rule, cases = fixture()
    source = SOURCE
    exclusion = EXCLUSION
    atoms = rule["metadata"]["proof"]["atoms"]
    if mutation == "wrong_place":
        exclusion = EXCLUSION.replace("Quebec", "Ontario")
        source = source.replace(EXCLUSION, exclusion)
        atoms[0]["source"]["excerpt"] = exclusion
    elif mutation == "restriction":
        exclusion = EXCLUSION[:-1] + " unless disabled."
        source = source.replace(EXCLUSION, exclusion)
        atoms[0]["source"]["excerpt"] = exclusion
    elif mutation == "spouse":
        exclusion = EXCLUSION.replace("residents", "spouses of residents")
        source = source.replace(EXCLUSION, exclusion)
        atoms[0]["source"]["excerpt"] = exclusion
    elif mutation == "bare_date":
        for atom in atoms[1:]:
            atom["source"]["excerpt"] = "for 2025"
    elif mutation == "wrong_date_citation":
        atoms[1]["source"]["corpus_citation_path"] = "ca/policy/other"
    elif mutation == "wrong_formula_citation":
        atoms[0]["source"]["corpus_citation_path"] = "ca/policy/other"
    elif mutation == "wrong_version_path":
        atoms[1]["path"] = "versions[1].effective_from"
    elif mutation == "supersession":
        rule["versions"].append(
            {
                "effective_from": "2025-07-01",
                "effective_to": "2025-12-31",
                "formula": "123",
            }
        )
    elif mutation == "wrong_year":
        for case in cases:
            case["period"] = "2026"
    elif mutation == "different_period":
        cases[1]["period"] = "2025-01"
    elif mutation == "missing_period":
        del cases[1]["period"]
    elif mutation == "wrong_output":
        cases[1]["output"]["credit"] = 1
    elif mutation == "missing_formula_proof":
        del atoms[0]
    assert not collect(rule, cases, source, exclusion)


@pytest.mark.parametrize(
    "insertion",
    [
        "Form T9999 instructions\n",
        "Another form is used here.\n",
        "Only disabled claimants may use this form.\n",
    ],
)
def test_intervening_form_or_restriction_is_not_inherited(insertion):
    rule, cases = fixture()
    for location in [SOURCE.index("Use this form"), SOURCE.index(EXCLUSION)]:
        assert not collect(
            rule, cases, SOURCE[:location] + insertion + SOURCE[location:]
        )


@pytest.mark.parametrize(
    "prefix,suffix", [('"', '"'), ('"', ""), ("“", '"'), ("[", ")")]
)
def test_quoted_or_malformed_whole_source_rejects(prefix, suffix):
    rule, cases = fixture()
    assert not collect(rule, cases, prefix + "\n" + SOURCE + suffix)


@pytest.mark.parametrize("kind", ["numeric", "calendar", "relational", "wrong_pair"])
def test_other_witness_modes_cannot_use_boolean_residence_route(kind):
    rule, cases = fixture()
    witness = next(w for w in actual_witnesses(rule, cases) if w.active_value)
    updates = {
        "numeric": {"numeric_transition": (0.0, 1.0)},
        "calendar": {"calendar_attainment_age": 65},
        "relational": {"relational_transitions": (("x", "Eq", "y"),)},
        "wrong_pair": {"case_pair_identity": (1, 2)},
    }
    assert not collect(rule, cases, witnesses={replace(witness, **updates[kind])})


def test_numeric_values_named_residence_do_not_supply_boolean_fact():
    rule, cases = fixture()
    witnesses = actual_witnesses(rule, cases)
    cases[0]["input"][SELECTOR] = 0
    cases[1]["input"][SELECTOR] = 1
    assert not collect(rule, cases, witnesses=witnesses)


@pytest.mark.parametrize(
    "place",
    ["Quebec unless disabled", "Quebec Who Are Eligible", "Quebec Disabled Spouse"],
)
def test_invented_selector_cannot_mirror_restrictive_place(place):
    rule, cases = fixture()
    name = "resident_of_" + place.lower().replace(" ", "_") + "_at_year_end"
    exclusion = EXCLUSION.replace("Quebec", place)
    source = SOURCE.replace(EXCLUSION, exclusion)
    rule["versions"][0]["formula"] = rule["versions"][0]["formula"].replace(
        SELECTOR, name
    )
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = exclusion
    for case in cases:
        case["input"][name] = case["input"].pop(SELECTOR)
    assert not collect(rule, cases, source, exclusion)


def test_duplicate_source_clause_or_annual_header_is_ambiguous():
    rule, cases = fixture()
    assert not collect(rule, cases, SOURCE + "\n" + EXCLUSION)
    assert not collect(
        rule,
        cases,
        SOURCE + "\n" + SOURCE[SOURCE.index("Use this form") : SOURCE.index(EXCLUSION)],
    )


def test_explicit_year_end_without_owned_annual_form_context_stays_unsupported():
    rule, cases = fixture()
    exclusion = EXCLUSION[:-1] + " at the end of the year."
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = exclusion
    assert not collect(rule, cases, exclusion, exclusion)


def test_recipient_nonresidence_polarity_is_not_application_negation():
    rule, cases = fixture()
    exclusion = EXCLUSION.replace("residents", "non-residents")
    source = SOURCE.replace(EXCLUSION, exclusion)
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = exclusion
    rule["versions"][0]["formula"] = f"if {SELECTOR}: 1600 else: 0"
    cases[0]["output"]["credit"] = 0
    cases[1]["output"]["credit"] = 1600
    found = collect(rule, cases, source, exclusion)
    assert len(found) == 1
    assert next(iter(found)).active_value is False


def test_raw_source_whitespace_offsets_are_preserved():
    rule, cases = fixture()
    source = SOURCE.replace("This form does not apply", "This form\n does not apply")
    exclusion = EXCLUSION.replace(
        "This form does not apply", "This form\n does not apply"
    )
    found = collect(rule, cases, source, exclusion)
    assert len(found) == 1


def test_formula_proof_cannot_be_borrowed_from_another_selected_version():
    rule, cases = fixture()
    rule["metadata"]["proof"]["atoms"][0]["path"] = "versions[1].formula"
    assert not collect(rule, cases)


@pytest.mark.parametrize("place", sorted(c._CANADIAN_RESIDENCE_PLACE_NAMES))
def test_all_official_canadian_place_identities_with_owned_annual_context(place):
    rule, cases = fixture()
    exclusion = EXCLUSION.replace("Quebec", place)
    source = SOURCE.replace(EXCLUSION, exclusion)
    name = "resident_of_" + place.lower().replace(" ", "_") + "_at_year_end"
    rule["versions"][0]["formula"] = rule["versions"][0]["formula"].replace(
        SELECTOR, name
    )
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = exclusion
    for case in cases:
        case["input"][name] = case["input"].pop(SELECTOR)
    assert len(collect(rule, cases, source, exclusion)) == 1


@pytest.mark.parametrize(
    "place",
    [
        "Quebec Or Ontario",
        "Quebec With Children",
        "Quebec Without Children",
        "Quebec Aged Sixty",
        "Atlantis",
    ],
)
def test_matching_invented_selector_cannot_admit_compound_or_unknown_place(place):
    rule, cases = fixture()
    exclusion = EXCLUSION.replace("Quebec", place)
    source = SOURCE.replace(EXCLUSION, exclusion)
    name = "resident_of_" + place.lower().replace(" ", "_") + "_at_year_end"
    rule["versions"][0]["formula"] = rule["versions"][0]["formula"].replace(
        SELECTOR, name
    )
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = exclusion
    for case in cases:
        case["input"][name] = case["input"].pop(SELECTOR)
    assert not collect(rule, cases, source, exclusion)


def test_short_residence_selector_still_requires_owned_annual_form_context():
    rule, cases = fixture()
    name = "resident_of_quebec"
    rule["versions"][0]["formula"] = rule["versions"][0]["formula"].replace(
        SELECTOR, name
    )
    for case in cases:
        case["input"][name] = case["input"].pop(SELECTOR)
    assert not collect(rule, cases)
    assert not collect(rule, cases, EXCLUSION)


def test_non_canadian_route_does_not_call_canadian_context(monkeypatch):
    rule, cases = fixture()

    def unexpected(*args, **kwargs):
        raise AssertionError("Canadian context must not handle non-Canadian citation")

    monkeypatch.setattr(c, "_source_owned_residence_applicability_context", unexpected)
    c._exception_witnesses_for_branch(
        branch(SOURCE),
        source_text=SOURCE,
        corpus_citation_path="us/policy/example",
        principal_rules={"credit": rule},
        principal_rule_paths={"credit": {()}},
        asserted_by_rule={"credit": cases},
        toggled_exception_selectors=actual_witnesses(rule, cases),
        extract_numeric_occurrences=lambda _text: (),
    )


@pytest.mark.parametrize(
    "period",
    [
        {"period_kind": "tax_year", "start": "2025-01-01", "end": "2026-12-31"},
        {"period_kind": "tax_year", "start": "2025-12-31", "end": "2025-01-01"},
        {"period_kind": "tax_year", "start": "2025-01-01"},
        {"period_kind": "month", "start": "2025-01-01", "end": "2025-12-31"},
        {"start": "2025-01-01", "end": "2025-12-31"},
        {"period_kind": "tax_year", "start": "2025-01-01", "end": "invalid"},
        {"period_kind": "tax_year", "start": "2025-01-01", "end": "2025-06-30"},
        "2025-01-01",
        "2025-01",
        "2026",
    ],
)
def test_actual_callsite_rejects_incomplete_or_nonannual_period_identity(period):
    rule, cases = fixture()
    for case in cases:
        case["period"] = period
    assert not collect(rule, cases)


def test_canonical_annual_year_string_is_supported():
    rule, cases = fixture()
    for case in cases:
        case["period"] = "2025"
    assert len(collect(rule, cases)) == 1


@pytest.mark.parametrize(
    "name", ["quebec_condition", "quebec_eligible", "quebec_residence"]
)
def test_source_owned_residence_branch_does_not_fall_back_for_opaque_selector(name):
    rule, cases = fixture()
    rule["versions"][0]["formula"] = rule["versions"][0]["formula"].replace(
        SELECTOR, name
    )
    for case in cases:
        case["input"][name] = case["input"].pop(SELECTOR)
    assert not collect(rule, cases)


def test_unrelated_canadian_branch_retains_generic_route(monkeypatch):
    rule, cases = fixture()
    text = "This form does not apply to claimants with outstanding debt."
    source = SOURCE.replace(EXCLUSION, text)

    def unexpected(*args, **kwargs):
        raise AssertionError("Non-residence source must retain the original route")

    monkeypatch.setattr(c, "_source_owned_residence_applicability_context", unexpected)
    collect(rule, cases, source, text)
