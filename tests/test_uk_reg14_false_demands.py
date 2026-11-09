"""Complete-source demands that regulation 14's text does not make.

Signed re-encode run 41745442025 ended with refusals of regulation 14 of the
Council Tax Reduction (Scotland) Regulations 2012 that came from three misreadings
of its text. Each one meets every re-encode of the regulation, and council tax
reduction schemes across the UK repeat the same wording.

* "amount B is 2 6/7 per cent of the difference" read as a division, because of
  the slash, and as a computation over 2, 6 and 7 as well as the rate. No formula
  can execute that.
* "falls", which means "if" in German, read as a condition in "the day falls
  within a week".
* Condition words inside the title of another provision read as conditions of
  this one: "regulation 57 (residents of a dwelling to whom regulation 14(6) does
  not apply)".

Invariants pinned here:

* a digit mixed percentage yields one rate, (whole + numerator / denominator) /
  100, and its parts yield nothing;
* its slash adds no division, and a real division in the same text still counts;
* the collapse removes only occurrences nested in a mixed percentage;
* outside a German-language corpus "falls" is never a condition cue; inside one,
  and in text with no citation path, it stays one;
* a condition word leaves only from inside a UK cross-reference's title. Operative
  text, a parenthesis that opens with a limiting word, and every other corpus keep
  their cues;
* the cue filters only remove: the filtered cues are a subset of the unfiltered
  cues, and the difference is exactly the cues they target.
"""

from __future__ import annotations

import contextlib
import functools
import math
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as vp

CITATION = "uk/regulation/ssi/2012/319/14"
FIXTURES = Path(__file__).parent / "fixtures" / "source_completeness"
REGULATION_14 = (FIXTURES / "uk_ssi_2012_319_14_source.txt").read_text()
PARAGRAPH_LIMBS = (FIXTURES / "uk_ssi_2012_319_14_paragraph_limbs.yaml").read_text()
_NEVER = re.compile(r"(?!)")
GROUNDING = functools.partial(
    vp.extract_typed_numeric_occurrences_from_text, profile="legacy"
)
PROPERTY_SETTINGS = settings(
    max_examples=120,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

ENGLISH_CORPORA = (
    CITATION,
    "uk-adur/manual/council-tax-reduction-scheme-2026-2027/page-8",
    "us/statute/26/32",
    "nz/act/2007/97/MA1",
    "ca/statute/income-tax-act/118",
)
GERMAN_CORPORA = ("de/statute/estg/32a", "ch/sr/642.11/36", "at/gesetz/estg/33")


def occurrences(text: str) -> tuple[sc.NumericOccurrenceLike, ...]:
    branch = sc.SourceStructureBranch(
        (), "formula-clause", "clause", text, 0, len(text)
    )
    return sc._formula_branch_computation_occurrences(
        branch, interval=None, extract_numeric_occurrences=GROUNDING
    )


@contextlib.contextmanager
def mixed_percentage_collapse_disabled() -> Iterator[None]:
    saved = sc._DIGIT_MIXED_NUMERAL
    sc._DIGIT_MIXED_NUMERAL = _NEVER
    try:
        yield
    finally:
        sc._DIGIT_MIXED_NUMERAL = saved


def cues(text: str, citation: str = "") -> list[tuple[int, int, str]]:
    with sc._condition_cue_scope(citation):
        matches = sc._source_exception_or_applicability_matches(text)
    return [(match.start(), match.end(), match.group(0).lower()) for match in matches]


mixed_numbers = st.tuples(
    st.integers(min_value=1, max_value=99),
    st.integers(min_value=1, max_value=19),
    st.integers(min_value=2, max_value=20),
)
percent_markers = st.sampled_from([" per cent", " percent", "%"])


@PROPERTY_SETTINGS
@given(number=mixed_numbers, marker=percent_markers)
def test_digit_mixed_percentage_is_one_rate(
    number: tuple[int, int, int], marker: str
) -> None:
    whole, numerator, denominator = number
    numeral = f"{whole} {numerator}/{denominator}{marker}"
    text = f"amount B is {numeral} of the difference between A and C"
    start = text.index(numeral)
    end = start + len(numeral)
    inside = [
        item for item in occurrences(text) if item.start < end and start < item.end
    ]
    assert {(item.start, item.end) for item in inside} == {(start, end)}
    expected = (whole + numerator / denominator) / 100
    assert any(math.isclose(item.value, expected) for item in inside)


@PROPERTY_SETTINGS
@given(number=mixed_numbers, marker=percent_markers)
def test_mixed_percentage_collapse_only_removes_its_own_parts(
    number: tuple[int, int, int], marker: str
) -> None:
    whole, numerator, denominator = number
    numeral = f"{whole} {numerator}/{denominator}{marker}"
    text = f"pay 17 pounds, then {numeral} of the excess over 300"
    start = text.index(numeral)
    end = start + len(numeral)
    collapsed = occurrences(text)
    with mixed_percentage_collapse_disabled():
        everything = occurrences(text)
    assert set(collapsed) <= set(everything)
    removed = set(everything) - set(collapsed)
    # The extractor yields the parts only beside a worded marker.
    assert removed or marker == "%"
    assert all(start <= item.start and item.end <= end for item in removed)
    assert all((item.start, item.end) != (start, end) for item in removed)
    assert {item.value for item in collapsed} >= {17.0, 300.0}


@pytest.mark.parametrize(
    ("words", "digits"),
    [
        ("two and one-half", "2 1/2"),
        ("two and six-sevenths", "2 6/7"),
        ("twelve and one-half", "12 1/2"),
        ("one and three-quarters", "1 3/4"),
    ],
)
def test_digit_and_worded_mixed_percentages_agree(words: str, digits: str) -> None:
    worded = occurrences(f"amount B is {words} per cent of the excess")
    digit = occurrences(f"amount B is {digits} per cent of the excess")
    assert len(worded) == len(digit) == 1
    assert math.isclose(worded[0].value, digit[0].value)


@pytest.mark.parametrize(
    "numeral",
    ["2 ½ per cent", "12 ¼ percent", "2½ per cent"],
)
def test_vulgar_fraction_percentage_is_one_rate(numeral: str) -> None:
    text = f"amount B is {numeral} of the excess"
    start = text.index(numeral)
    end = start + len(numeral)
    assert {
        (item.start, item.end)
        for item in occurrences(text)
        if item.start < end and start < item.end
    } == {(start, end)}


@PROPERTY_SETTINGS
@given(number=mixed_numbers, marker=percent_markers)
def test_mixed_percentage_slash_is_no_division(
    number: tuple[int, int, int], marker: str
) -> None:
    whole, numerator, denominator = number
    numeral = f"{whole} {numerator}/{denominator}{marker}"
    assert "divide" not in sc._formula_operation_kinds(f"{numeral} of the difference")
    assert "divide" in sc._formula_operation_kinds(
        f"{numeral} of the difference divided by 7"
    )


@pytest.mark.parametrize(
    "text",
    [
        "income / 2",
        "$5/person",
        "1/2 of income",
        "the 2 6/7 ratio",
        "on 15 3/7 per person",
    ],
)
def test_other_slashes_stay_divisions(text: str) -> None:
    assert "divide" in sc._formula_operation_kinds(text)


def test_mixed_percentage_masking_keeps_offsets() -> None:
    text = "amount B is 2 6/7 per cent of X and 4/5"
    masked = sc._without_digit_mixed_percentage_slashes(text)
    assert len(masked) == len(text)
    assert masked == "amount B is 2 6 7 per cent of X and 4/5"


FALLS_CLAUSES = (
    "the day falls within a week in respect of which the person has no income",
    "the week in which the day falls and the applicable amount",
    "a resident who falls to be disregarded for the purposes of discount",
    "income that falls short of the amount",
    "Falls Church is the place of residence",
)


@pytest.mark.parametrize("citation", ENGLISH_CORPORA)
@pytest.mark.parametrize("clause", FALLS_CLAUSES)
def test_english_falls_is_no_condition(citation: str, clause: str) -> None:
    assert [cue for cue in cues(clause, citation) if cue[2] == "falls"] == []


@pytest.mark.parametrize("citation", GERMAN_CORPORA)
def test_german_falls_stays_a_condition(citation: str) -> None:
    text = "Der Betrag erhöht sich, falls der Antrag rechtzeitig gestellt wird."
    assert [cue[2] for cue in cues(text, citation)] == ["falls"]


def test_falls_stays_a_cue_when_no_citation_names_a_language() -> None:
    assert [cue[2] for cue in cues(FALLS_CLAUSES[0])] == ["falls"]


@PROPERTY_SETTINGS
@given(
    citation=st.sampled_from(ENGLISH_CORPORA),
    before=st.sampled_from(["if the day", "the day", "unless the week"]),
    after=st.sampled_from(["within a week", "to be disregarded", "if income is low"]),
)
def test_falls_filter_only_removes_falls(
    citation: str, before: str, after: str
) -> None:
    text = f"{before} falls {after}"
    scoped = cues(text, citation)
    unscoped = cues(text)
    assert set(scoped) <= set(unscoped)
    assert {cue[2] for cue in set(unscoped) - set(scoped)} == {"falls"}


CUES = (
    "subject to",
    "does not apply",
    "if",
    "except",
    "unless",
    "notwithstanding",
)
titles = st.builds(
    lambda head, cue, tail: f"{head} {cue} {tail}",
    st.sampled_from(
        [
            "persons not entitled to council tax reduction: persons",
            "residents of a dwelling to whom regulation 14(6)",
            "circumstances where the condition",
        ]
    ),
    st.sampled_from(CUES),
    st.sampled_from(["immigration control", "the applicant", "work-focused interview"]),
)
reference_forms = st.sampled_from(
    [
        "regulation 57 ({0})",
        "regulation 14(6) ({0})",
        "Schedule 2 ({0})",
        "paragraph 12 ({0})",
        "section 75 ({0})",
    ]
)


def title_cues(text: str, title: str, citation: str) -> list[tuple[int, int, str]]:
    start = text.index(title)
    end = start + len(title)
    return [cue for cue in cues(text, citation) if start <= cue[0] and cue[1] <= end]


@PROPERTY_SETTINGS
@given(
    title=titles,
    form=reference_forms,
    citation=st.sampled_from([CITATION, ENGLISH_CORPORA[1]]),
)
def test_condition_word_in_a_uk_reference_title_is_no_cue(
    title: str, form: str, citation: str
) -> None:
    reference = form.format(title)
    text = f"a person described in {reference} is entitled if the claim stands"
    assert title_cues(text, title, citation) == []
    assert ("if" in [cue[2] for cue in cues(text, citation)]) is True
    remaining = cues(text, citation)
    assert text[remaining[-1][0] : remaining[-1][1]] == "if"
    assert remaining[-1][0] == text.rindex(" if the claim") + 1


@PROPERTY_SETTINGS
@given(first=titles, second=titles, third=titles)
def test_condition_words_in_a_titled_list_are_no_cues(
    first: str, second: str, third: str
) -> None:
    text = (
        f"is not a person to whom regulation 15 ({first}), regulation 16 ({second}) "
        f"or 19 ({third}) applies"
    )
    assert cues(text, CITATION) == []


@PROPERTY_SETTINGS
@given(
    title=titles,
    form=reference_forms,
    citation=st.sampled_from(["us/statute/26/32", "nz/act/2007/97/MA1"]),
)
def test_other_corpora_keep_cues_inside_titles(
    title: str, form: str, citation: str
) -> None:
    text = f"a person described in {form.format(title)} is entitled"
    assert title_cues(text, title, citation) == title_cues(text, title, "")
    assert title_cues(text, title, citation)


@pytest.mark.parametrize(
    ("reference", "cue"),
    [
        ("paragraph 7 (except paragraph 112)", "except"),
        ("Schedule 2A (except paragraphs 2(1)(b) and 3(1)(b))", "except"),
        ("paragraph 26 (subject to paragraph 26(1)(b)(i))", "subject to"),
        ("regulation 5 (or, if paragraph (3) applies, regulation 6)", "if"),
        ("regulation 5 (other than paragraph (b) if it applies)", "if"),
        ("section 7 (unless paragraph 3 applies)", "unless"),
        ("section 7 (if any)", "if"),
    ],
)
def test_limiting_parenthesis_keeps_its_cue(reference: str, cue: str) -> None:
    assert cue in [item[2] for item in cues(f"under {reference}", CITATION)]


def test_title_that_opens_with_when_is_a_title() -> None:
    text = "regulation 58 (when entitlement begins and change of circumstances)"
    assert cues(text, CITATION) == []
    assert cues(text) != []


@PROPERTY_SETTINGS
@given(title=titles, form=reference_forms)
def test_title_filter_only_removes_cues_inside_titles(title: str, form: str) -> None:
    text = f"if a person in {form.format(title)} applies, unless excluded"
    scoped = cues(text, CITATION)
    unscoped = cues(text)
    assert set(scoped) <= set(unscoped)
    start = text.index(title)
    end = start + len(title)
    assert all(start <= cue[0] and cue[1] <= end for cue in set(unscoped) - set(scoped))
    assert {cue[2] for cue in scoped} == {"if", "unless"}


def test_condition_cue_scope_restores_and_nests() -> None:
    assert sc._CONDITION_CUE_CITATION_PATH.get() == ""
    with sc._condition_cue_scope(CITATION):
        with sc._condition_cue_scope(""):
            assert sc._CONDITION_CUE_CITATION_PATH.get() == CITATION
        with sc._condition_cue_scope("de/statute/estg/32a"):
            assert sc._CONDITION_CUE_CITATION_PATH.get() == "de/statute/estg/32a"
        assert sc._CONDITION_CUE_CITATION_PATH.get() == CITATION
    assert sc._CONDITION_CUE_CITATION_PATH.get() == ""


def test_exception_branches_read_the_citation_they_are_given() -> None:
    text = "A resident who falls to be disregarded is entitled."
    branches = sc.recognize_source_structure(text, corpus_citation_path=CITATION)
    assert (
        sc._source_exception_branches(
            text,
            branches=branches,
            active_branches=branches,
            deferred_paths=set(),
            corpus_citation_path=CITATION,
        )
        == ()
    )
    assert sc._source_exception_branches(
        text,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
        corpus_citation_path="de/statute/estg/32a",
    )


BASE_INPUT = {
    "relevant_person_is_liable_to_pay_council_tax_under_section_75": True,
    "relevant_person_is_liable_for_council_tax_for_dwelling_as_sole_or_main_residence": True,
    "regulation_15_applies": False,
    "regulation_16_applies": False,
    "regulation_19_applies": False,
    "relevant_person_has_made_application_in_accordance_with_part_9": True,
    "appropriate_maximum_council_tax_reduction_exists": True,
    "relevant_person_has_no_income": True,
    "relevant_person_weekly_income": 0,
    "applicable_amount": 300,
    "appropriate_maximum_council_tax_reduction": 100,
    "no_included_other_resident_is_liable_to_pay_rent_to_relevant_person": True,
    "relevant_person_is_entitled_to_alternative_maximum_reduction_because_of_resident_income": True,
    "appropriate_alternative_maximum_council_tax_reduction": 120,
    "relevant_person_entitlement_to_council_tax_reduction_has_commenced": True,
}
ROUTE_14A_ONLY = (
    "relevant_person_is_liable_for_council_tax_for_dwelling",
    "dwelling_is_in_band_e_f_g_or_h_and_is_sole_or_main_residence",
    "relevant_person_is_not_member_of_couple_or_polygamous_marriage",
    "relevant_person_is_not_responsible_for_child_or_young_person",
)
MODULE = "uk:regulations/ssi/2012/319/14"
RATE = 0.02857142857142857142857142857
EXCESS_INCOME = {
    "relevant_person_has_no_income": False,
    "relevant_person_weekly_income": 335,
    "relevant_person_is_entitled_to_alternative_maximum_reduction_because_of_resident_income": False,
    "appropriate_alternative_maximum_council_tax_reduction": 0,
}


def judgment(value: bool) -> str:
    return "holds" if value else "not_holds"


def model_outputs(facts: dict[str, object]) -> dict[str, object]:
    """What the paragraph-limb model computes, stated independently of the gate."""

    common = (
        facts["relevant_person_is_liable_to_pay_council_tax_under_section_75"]
        and facts[
            "relevant_person_is_liable_for_council_tax_for_dwelling_as_sole_or_main_residence"
        ]
        and not facts["regulation_15_applies"]
        and not facts["regulation_16_applies"]
        and not facts["regulation_19_applies"]
        and facts["relevant_person_has_made_application_in_accordance_with_part_9"]
    )
    income = facts["relevant_person_weekly_income"]
    applicable = facts["applicable_amount"]
    maximum = facts["appropriate_maximum_council_tax_reduction"]
    amount_b = RATE * max(0, income - applicable)
    paragraph_5a = facts["relevant_person_has_no_income"] or income <= applicable
    paragraph_5b = not paragraph_5a and maximum > amount_b
    income_route = facts["appropriate_maximum_council_tax_reduction_exists"] and (
        paragraph_5a or paragraph_5b
    )
    paragraph_6 = (
        facts["no_included_other_resident_is_liable_to_pay_rent_to_relevant_person"]
        and facts[
            "relevant_person_is_entitled_to_alternative_maximum_reduction_because_of_resident_income"
        ]
    )
    limb_a = maximum if paragraph_5a else 0
    limb_b = maximum - amount_b if paragraph_5b else 0
    limb_c = (
        facts["appropriate_alternative_maximum_council_tax_reduction"]
        if paragraph_6
        else 0
    )
    before = max((limb_a + limb_b) if income_route else 0, limb_c)
    # Regulation 14A's route stays off in every case, so its amount is zero.
    paragraph_3a = before >= 0
    entitled = (
        facts["relevant_person_entitlement_to_council_tax_reduction_has_commenced"]
        and common
        and paragraph_3a
        and (income_route or paragraph_6)
    )
    return {
        "regulation_14_common_conditions_satisfied": judgment(common),
        "regulation_14_income_excess_reduction_amount": amount_b,
        "regulation_14_paragraph_5_a_applies": judgment(paragraph_5a),
        "regulation_14_paragraph_5_b_applies": judgment(paragraph_5b),
        "regulation_14_income_route_satisfied": judgment(income_route),
        "regulation_14_paragraph_6_applies": judgment(paragraph_6),
        "regulation_14_paragraph_8_a_amount": limb_a,
        "regulation_14_paragraph_8_b_amount": limb_b,
        "regulation_14_paragraph_8_c_amount": limb_c,
        "regulation_14_reduction_before_regulation_14a_comparison": before,
        "regulation_14_paragraph_3a_condition_satisfied": judgment(paragraph_3a),
        "regulation_14_reduction_entitlement": judgment(entitled),
        "regulation_14_council_tax_reduction": before if entitled else 0,
    }


def case(name: str, **overrides: object) -> dict[str, object]:
    facts = {**BASE_INPUT, **overrides}
    inputs: dict[str, object] = {
        f"{MODULE}#input.{key}": value for key, value in facts.items()
    }
    inputs.update({f"{MODULE}A#input.{key}": False for key in ROUTE_14A_ONLY})
    return {
        "name": name,
        "period": {
            "period_kind": "custom",
            "name": "day",
            "start": "2026-01-15",
            "end": "2026-01-15",
        },
        "input": inputs,
        "output": {
            f"{MODULE}#{key}": value for key, value in model_outputs(facts).items()
        },
    }


SIX_CASES = (
    case("both_routes"),
    case("regulation_15_displaces", regulation_15_applies=True),
    case("regulation_16_displaces", regulation_16_applies=True),
    case("regulation_19_displaces", regulation_19_applies=True),
    case("excess_income", **EXCESS_INCOME),
    case(
        "income_at_applicable",
        **{**EXCESS_INCOME, "relevant_person_weekly_income": 300},
    ),
)
LIMB_B_PAIR = case(
    "excess_income_small_maximum",
    **{**EXCESS_INCOME, "appropriate_maximum_council_tax_reduction": 0.5},
)
LIMB_C_PAIR = case(
    "both_routes_without_alternative",
    relevant_person_is_entitled_to_alternative_maximum_reduction_because_of_resident_income=False,
)
FALSE_DEMANDS = (
    "formula clause 12",
    "subject to immigration control",
    "the day falls within",
    "falls to be disregarded",
    "regulation 14(6) does not apply",
    "the day falls and the applicable",
)


def analyze(cases: tuple[dict[str, object], ...]) -> list[str]:
    bindings = vp.collect_artifact_numeric_bindings(
        PARAGRAPH_LIMBS,
        extract_named_scalars=vp.extract_named_scalar_occurrences,
        imported_symbol_contents=(),
    )
    profile = vp._numeric_profile_for_citation_path(CITATION)
    result = sc.analyze_complete_source_unit(
        PARAGRAPH_LIMBS,
        REGULATION_14,
        corpus_citation_path=CITATION,
        test_cases=list(cases),
        extract_numeric_occurrences=functools.partial(
            vp.extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            vp.extract_typed_numeric_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=vp.extract_named_scalar_occurrences,
        numeric_value_is_grounded=vp.numeric_value_is_grounded,
        artifact_numeric_values=tuple(value for _name, value in bindings),
        artifact_numeric_bindings=bindings,
    )
    return list(result.issues)


def test_fixture_module_parses() -> None:
    payload = yaml.safe_load(PARAGRAPH_LIMBS)
    assert payload["format"] == "rulespec/v1"
    assert payload["module"]["source_verification"]["corpus_citation_path"] == CITATION


def test_a_model_of_each_limb_clears_the_gate() -> None:
    assert analyze((*SIX_CASES, LIMB_B_PAIR, LIMB_C_PAIR)) == []


@pytest.mark.parametrize(
    ("cases", "demanded"),
    [
        (SIX_CASES + (LIMB_B_PAIR,), ["if paragraph (6) applies"]),
        (
            SIX_CASES,
            ["if paragraph (6) applies", "appropriate maximum council tax reduction"],
        ),
    ],
)
def test_the_gate_still_demands_the_real_conditions(
    cases: tuple[dict[str, object], ...], demanded: list[str]
) -> None:
    issues = analyze(cases)
    assert len(issues) == 1
    assert "require paired positive/blocking cases" in issues[0]
    missing = issues[0].split("missing:", 1)[1].split("The evaluator", 1)[0]
    for condition in demanded:
        assert condition in missing
    assert missing.count("uk/regulation/ssi/2012/319/14 [source unit]") == len(demanded)
    for false_demand in FALSE_DEMANDS:
        assert false_demand not in issues[0]
