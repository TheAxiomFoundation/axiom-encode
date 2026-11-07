"""Grounding Hebrew month names, and the row labels an index selector returns.

National Insurance Law schedule A1 part D (לוח א׳1 חלק ד׳) sets a woman's
old-age pension age by her birth month. Its rows are ranges of months named in
Hebrew ("ספטמבר 1939 עד אפריל 1940") and it prints no month number, so an
encoding that compares a numeric birth month was blocked with "Ungrounded
generated numeric literal: 6". An encoding that picks the row with a cohort
selector returning 0 to 15 was blocked the same way on the labels, and the
embedded-scalar repair lifted them into ``woman_birth_cohort_scalar_limit_*``
parameters that could not be grounded either.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.cli import _try_repair_generated_embedded_scalar_literals_for_apply
from axiom_encode.harness.validator_pipeline import (
    _HEBREW_GREGORIAN_MONTH_NUMBERS,
    GROUNDING_ALLOWED_VALUES,
    ValidatorPipeline,
    _month_comparison_literal_values,
    _rulespec_index_selectors,
    extract_numbers_from_text,
    extract_numeric_occurrences_from_text,
    extract_typed_month_occurrences_from_text,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    find_ungrounded_numeric_issues,
    find_ungrounded_numeric_issues_scoped,
    numeric_value_is_grounded,
)
from axiom_encode.repo_routing import monorepo_checkout_name

PART_D_CITATION = "il/statute/national-insurance-law-1995/schedule-a1/sign-4"
# The corpus body of PART_D_CITATION in the 2026-09-29-il-taxben-contributions
# release, verbatim.
PART_D_TEXT = """חלק ד׳
(סעיפים 245(א), 342(ג), 351(ב) ו־406(א)(4)(א))
גיל הזכאות לקצבת אזרח ותיק לנשים לפי חודש לידתן
חודש הלידה | גיל הזכאות (בשנים)
עד יוני 1939 | 65
יולי ואוגוסט 1939 | 65 ו־4 חודשים
ספטמבר 1939 עד אפריל 1940 | 65 ו־8 חודשים
מאי עד דצמבר 1940 | 66
ינואר עד אוגוסט 1941 | 66 ו־4 חודשים
ספטמבר 1941 עד אפריל 1942 | 66 ו־8 חודשים
מאי 1942 עד דצמבר 1944 | 67
ינואר עד אוגוסט 1945 | 67 ו־4 חודשים
ספטמבר 1945 עד אפריל 1946 | 67 ו־8 חודשים
מאי עד דצמבר 1946 | 68
ינואר עד אוגוסט 1947 | 68 ו־4 חודשים
ספטמבר 1947 עד אפריל 1948 | 68 ו־8 חודשים
מאי עד דצמבר 1948 | 69
ינואר עד אוגוסט 1949 | 69 ו־4 חודשים
ספטמבר 1949 עד אפריל 1950 | 69 ו־8 חודשים
מאי 1950 ואילך | 70"""

# The months Part D names, and so the only month numbers it can ground.
PART_D_MONTH_NUMBERS = {1.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 12.0}

# Every one- and two-letter prefix a Hebrew word can carry, with and without a
# maqaf after the prefix.
HEBREW_PREFIXES = (
    "",
    *(letter for letter in "והבכלמש"),
    *(letter + "־" for letter in "והבכלמש"),
    *(first + second for first in "והבכלמש" for second in "והבכלמש"),
)

# Each sentence dates the month name "{m}" in a different way.
DATED_CONTEXTS = (
    "{m} 1942",
    "{m} שנת 2011",
    "ביום 4 {m}",
    "ב־4 {m} ישולם",
    "באחד {m} של שנת המס",
    "בשלושים ואחד {m}",
    "בחודש {m} שבתכוף",
    "מהחודשים {m}, יולי",
    "{m} של כל שנה",
    "ב־1 {m} כל שנה",
    "{m} ועד דצמבר 1940",
    "(5 {m} 2008)",
)
# Contexts dated by a day alone. The homographs מאי and מרץ need a year or the
# word month: a bare number before them is too often a section or a count.
DAY_ONLY_CONTEXTS = frozenset({"ביום 4 {m}", "ב־4 {m} ישולם", "בשלושים ואחד {m}"})
HOMOGRAPH_MONTHS = frozenset({"מאי", "מרץ"})
# Each sentence uses "{m}" with nothing that dates it.
UNDATED_CONTEXTS = (
    "{m}",
    "הסובל {m} ספיקת לב קשה",
    "המילה {m} נכתבה כאן",
    "65 ו־4 חודשים\n{m} ספיקה",
    "| 4 | {m} ספיקה",
    "1939\n{m} ספיקה",
    # A list or a range of names dates nothing by itself: people's names are
    # listed the same way ("יוני ומאי", Yoni and Mai).
    "{m} ויולי",
    "{m} ועד דצמבר",
    # A count of months is a length of time, not the months it names.
    "בתוך 3 חודשים {m} ויולי",
    # "כל שנה" with no של or ב needs a day ("יוני כל שנה מגיש").
    "{m} כל שנה",
)


def _month_occurrences(text: str) -> set[tuple[str, float, bool]]:
    """The occurrences a Hebrew month name produced: raw word, value, temporal."""
    return {
        (occurrence.raw, occurrence.value, occurrence.has_temporal_context)
        for occurrence in extract_typed_month_occurrences_from_text(text)
        if any(month in occurrence.raw for month in _HEBREW_GREGORIAN_MONTH_NUMBERS)
    }


def _ungrounded_values(issues: list[str]) -> set[str]:
    prefix = "Ungrounded generated numeric literal: "
    return {
        issue.removeprefix(prefix).split(" ", 1)[0]
        for issue in issues
        if issue.startswith(prefix)
    }


# --- (a) Hebrew month names ground month numbers ------------------------------


def _part_d_birth_month_module() -> str:
    """Part D encoded directly on a numeric birth month, row by row."""
    return f"""format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {PART_D_CITATION}
rules:
- name: woman_old_age_pension_age_years
  kind: derived
  entity: Person
  dtype: Integer
  period: Month
  versions:
  - effective_from: '0001-01-01'
    formula: |-
      if birth_year < 1939 or (birth_year == 1939 and birth_month <= 6): 65
      else: if birth_year == 1939 or (birth_year == 1940 and birth_month <= 4): 65
      else: if birth_year == 1940: 66
      else: if birth_year == 1941 or (birth_year == 1942 and birth_month <= 4): 66
      else: if birth_year <= 1944: 67
      else: if birth_year == 1945 or (birth_year == 1946 and birth_month <= 4): 67
      else: if birth_year == 1946: 68
      else: if birth_year == 1947 or (birth_year == 1948 and birth_month <= 4): 68
      else: if birth_year == 1948: 69
      else: if birth_year == 1949 or (birth_year == 1950 and birth_month <= 4): 69
      else: 70
- name: woman_old_age_pension_age_additional_months
  kind: derived
  entity: Person
  dtype: Integer
  period: Month
  versions:
  - effective_from: '0001-01-01'
    formula: |-
      if birth_year < 1939 or (birth_year == 1939 and birth_month <= 6): 0
      else: if birth_year == 1939 and birth_month >= 7 and birth_month <= 8: 4
      else: if birth_year == 1939 or (birth_year == 1940 and birth_month <= 4): 8
      else: if birth_year == 1940 and birth_month >= 5 and birth_month <= 12: 0
      else: if birth_year == 1941 and birth_month >= 1 and birth_month <= 8: 4
      else: if birth_year == 1941 and birth_month >= 9: 8
      else: 0
"""


def test_part_d_month_names_ground_the_birth_month_numbers_an_encoding_compares():
    assert (
        find_ungrounded_numeric_issues(
            _part_d_birth_month_module(),
            source_text=PART_D_TEXT,
            source_citation_path=PART_D_CITATION,
        )
        == []
    )


def test_a_month_part_d_does_not_name_stays_ungrounded():
    # October and November are named in no Part D row.
    content = _part_d_birth_month_module().replace(
        "birth_month >= 9: 8", "birth_month >= 10: 8"
    )

    issues = find_ungrounded_numeric_issues(
        content,
        source_text=PART_D_TEXT,
        source_citation_path=PART_D_CITATION,
    )

    assert _ungrounded_values(issues) == {"10"}


def _printed_numbers(text: str) -> set[float]:
    return {
        occurrence.value
        for occurrence in extract_typed_numeric_occurrences_from_text(text)
        if not occurrence.requires_month_context
    }


def _month_numbers(text: str) -> set[float]:
    """The month numbers the text's dated month names ground.

    They come from a stream of their own: ``extract_numbers_from_text`` and
    the other readers of the shared stream never see them.
    """
    return {
        occurrence.value
        for occurrence in extract_typed_month_occurrences_from_text(text)
    }


def test_part_d_grounds_exactly_the_months_it_names():
    grounded = _month_numbers(PART_D_TEXT)

    assert PART_D_MONTH_NUMBERS <= grounded
    assert not {10.0, 11.0} & grounded
    # The shared stream holds only the months Part D prints as numbers, the
    # 4 and 8 months of age.
    assert PART_D_MONTH_NUMBERS & extract_numbers_from_text(PART_D_TEXT) == {4.0, 8.0}


def test_month_numbers_ground_but_are_never_recall_obligations():
    # Part D prints 4 and 8 as months of age ("ו־4 חודשים"); those stay
    # obligations. The birth months it names add none.
    recall = set(extract_numeric_occurrences_from_text(PART_D_TEXT))
    inventory = {
        occurrence.value
        for occurrence in extract_typed_numeric_inventory_occurrences_from_text(
            PART_D_TEXT
        )
    }

    assert recall == inventory
    assert recall == {
        4.0,
        8.0,
        65.0,
        66.0,
        67.0,
        68.0,
        69.0,
        70.0,
        *(float(year) for year in (1939, 1940, 1941, 1942, 1944)),
        *(float(year) for year in range(1945, 1951)),
    }


def test_the_corpus_spells_march_both_ways():
    assert _HEBREW_GREGORIAN_MONTH_NUMBERS["מרס"] == 3.0
    assert _HEBREW_GREGORIAN_MONTH_NUMBERS["מרץ"] == 3.0
    assert sorted(set(_HEBREW_GREGORIAN_MONTH_NUMBERS.values())) == [
        float(month) for month in range(1, 13)
    ]


@pytest.mark.parametrize("context", DATED_CONTEXTS)
def test_every_month_name_under_every_prefix_grounds_its_number_when_dated(context):
    for month, value in _HEBREW_GREGORIAN_MONTH_NUMBERS.items():
        for prefix in HEBREW_PREFIXES:
            word = prefix + month
            text = context.format(m=word)

            occurrences = _month_occurrences(text)

            if month in HOMOGRAPH_MONTHS and context in DAY_ONLY_CONTEXTS:
                assert occurrences == set(), (text, occurrences)
            else:
                assert (word, value, True) in occurrences, (text, occurrences)


@pytest.mark.parametrize("context", UNDATED_CONTEXTS)
def test_no_month_name_under_any_prefix_grounds_a_number_undated(context):
    for month in _HEBREW_GREGORIAN_MONTH_NUMBERS:
        for prefix in HEBREW_PREFIXES:
            text = context.format(m=prefix + month)

            assert _month_occurrences(text) == set(), text


@pytest.mark.parametrize(
    "text",
    [
        # National Health Insurance Law schedule 2 item 12: "suffering from
        # severe heart failure", not May.
        "טיפול בחולה הסובל מאי ספיקת לב קשה למרות טיפול",
        # The same homograph joined by a maqaf is one word.
        "ילדים עד גיל 10 הסובלים מאי־ספיקה כלייתית",
        "במאי הסרט קיבל פרס",
        "עבד במרץ רב",
        # A number before a homograph is often a section or a count.
        "סעיף 7 מאי ספיקה",
        "בתוספת 12 במרץ רב",
        # The plural and the defective spelling of month also mean "new".
        "בתוך 2 חודשים מאי תשלום",
        "רכב חדש מאי",
        "חדשים מאי ספיקה",
        # Two homographs side by side are not a list of months.
        "שמאי ושמאי",
        "במרץ ובמרץ",
        # Singular "month" as a length of time, not "in the month of".
        "בתוך חודש מאי תשלום",
    ],
)
def test_month_homographs_ground_nothing(text):
    assert not ({3.0, 5.0} & _month_numbers(text))


def test_homograph_needs_in_the_month_of():
    assert 5.0 in _month_numbers("ובחודש מאי ישולם")
    assert 3.0 in _month_numbers("שבחודש מרץ ישולם")
    for text in ("לחודש מאי ישולם", "מחודש מאי ישולם", "בחודש\nמאי ישולם"):
        assert 5.0 not in _month_numbers(text), text


def test_defective_or_plural_month_word_dates_no_single_month():
    assert 7.0 not in _month_numbers("רכב חדש יולי")
    assert 7.0 not in _month_numbers("בתוך 3 חודשים יולי")
    assert 7.0 in _month_numbers("בחודש יולי")
    assert 5.0 in _month_numbers("בחודש מאי")
    assert 5.0 in _month_numbers("מאי 1942")


def test_a_list_of_months_is_dated_as_a_whole():
    # The year after August dates July too, and "the months" dates the list
    # after it, however long the list runs. The list alone dates nothing
    # (test_english_month_grounding.py, section b).
    assert {7.0, 8.0} <= _month_numbers("יולי ואוגוסט 1939")
    assert {1.0, 4.0, 7.0, 10.0} <= _month_numbers(
        "בכל אחד מהחודשים אפריל, יולי, אוקטובר וינואר, בעבור הרבעון"
    )
    assert {2.0, 11.0} <= _month_numbers(
        "ביום החמישה עשר של כל אחד מעשרת החדשים פברואר עד נובמבר של כל שנת מס"
    )


# --- (b) an index selector's results are row labels ---------------------------


def _cohort_selector_module(
    labels: list[int],
    *,
    selector_name: str = "woman_birth_cohort",
    selector_dtype: str = "Integer",
    extra_rule: str = "",
) -> str:
    """Part D encoded by picking its row with a selector, as the model did.

    ``labels`` are the selector's results, one per Part D row in order; the
    tables indexed by the selector are keyed by the same labels.
    """
    assert len(labels) == 16
    bounds = [
        "birth_year < 1939",
        "birth_year == 1939 and birth_month_name == 'יוני'",
        "birth_year == 1939 and birth_month_name == 'אוגוסט'",
        "birth_year <= 1940 and birth_month_name == 'אפריל'",
        "birth_year <= 1940",
        "birth_year <= 1941 and birth_month_name == 'אוגוסט'",
        "birth_year <= 1942 and birth_month_name == 'אפריל'",
        "birth_year <= 1944",
        "birth_year <= 1945 and birth_month_name == 'אוגוסט'",
        "birth_year <= 1946 and birth_month_name == 'אפריל'",
        "birth_year <= 1946",
        "birth_year <= 1947 and birth_month_name == 'אוגוסט'",
        "birth_year <= 1948 and birth_month_name == 'אפריל'",
        "birth_year <= 1948",
        "birth_year <= 1949 and birth_month_name == 'אוגוסט'",
    ]
    chain = [f"      if {bounds[0]}: {labels[0]}"]
    chain.extend(
        f"      else: if {bound}: {label}"
        for bound, label in zip(bounds[1:], labels[1:15], strict=True)
    )
    chain.append(f"      else: {labels[15]}")
    years = [65, 65, 65, 66, 66, 66, 67, 67, 67, 68, 68, 68, 69, 69, 69, 70]
    months = [0, 4, 8] * 5 + [0]
    year_rows = "\n".join(
        f"      {label}: {value}" for label, value in zip(labels, years, strict=True)
    )
    month_rows = "\n".join(
        f"      {label}: {value}" for label, value in zip(labels, months, strict=True)
    )
    chain_text = "\n".join(chain)
    return f"""format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {PART_D_CITATION}
rules:
- name: {selector_name}
  kind: derived
  entity: Person
  dtype: {selector_dtype}
  period: Month
  metadata:
    private: true
  versions:
  - effective_from: '0001-01-01'
    formula: |-
{chain_text}
- name: pension_age_completed_years
  kind: parameter
  dtype: Integer
  indexed_by: {selector_name}
  metadata:
    private: true
  versions:
  - effective_from: '0001-01-01'
    values:
{year_rows}
- name: pension_age_additional_months
  kind: parameter
  dtype: Integer
  indexed_by: {selector_name}
  metadata:
    private: true
  versions:
  - effective_from: '0001-01-01'
    values:
{month_rows}
- name: woman_old_age_pension_age_years
  kind: derived
  entity: Person
  dtype: Integer
  period: Month
  versions:
  - effective_from: '0001-01-01'
    formula: pension_age_completed_years[{selector_name}]
- name: woman_old_age_pension_age_additional_months
  kind: derived
  entity: Person
  dtype: Integer
  period: Month
  versions:
  - effective_from: '0001-01-01'
    formula: pension_age_additional_months[ {selector_name} ]
{extra_rule}"""


def _embedded_literals(tmp_path: Path, content: str) -> set[str]:
    rules_file = tmp_path.resolve() / "sign-4.yaml"
    rules_file.write_text(content)
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path.resolve(),
        axiom_rules_path=tmp_path.resolve(),
        enable_oracles=False,
        local_corpus_release=None,
    )
    literals = set()
    for issue in pipeline._check_embedded_scalar_literals(rules_file):
        assert issue.startswith("Embedded scalar literal: ")
        literals.add(issue.split(" embeds ", 1)[1].split(" ", 1)[0])
    return literals


SIXTEEN_ROWS = list(range(16))
# Relabelings of the sixteen rows, all within the 0 to 20 keys a selector may
# return: shifted, reversed, gapped and shuffled.
ROW_LABELINGS = {
    "zero_based": SIXTEEN_ROWS,
    "one_based": [label + 1 for label in SIXTEEN_ROWS],
    "reversed": SIXTEEN_ROWS[::-1],
    "gapped": [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 20],
    "shuffled": [7, 12, 0, 15, 3, 9, 1, 14, 5, 11, 2, 13, 8, 4, 10, 6],
}


def test_part_d_cohort_selector_is_an_index_selector_whatever_its_name():
    payload = yaml.safe_load(_cohort_selector_module(SIXTEEN_ROWS))

    assert _rulespec_index_selectors(payload["rules"]) == {"woman_birth_cohort"}


@pytest.mark.parametrize("labeling", sorted(ROW_LABELINGS))
def test_index_selector_labels_are_never_ungrounded_under_any_relabeling(
    labeling, tmp_path
):
    content = _cohort_selector_module(ROW_LABELINGS[labeling])

    assert (
        find_ungrounded_numeric_issues(
            content,
            source_text=PART_D_TEXT,
            source_citation_path=PART_D_CITATION,
        )
        == []
    )
    # The selector compares years the text prints; only those can be lifted.
    assert _embedded_literals(tmp_path, content) <= {
        str(year) for year in range(1939, 1950)
    }


@pytest.mark.parametrize("labeling", sorted(ROW_LABELINGS))
def test_a_selector_read_as_a_quantity_keeps_its_labels_under_the_check(
    labeling, tmp_path
):
    # Once another rule computes with the selector's value, its results are
    # numbers, and those the text does not print are reported: the intended
    # contrast with the test above.
    labels = ROW_LABELINGS[labeling]
    content = _cohort_selector_module(
        labels,
        extra_rule="""- name: cohort_rank
  kind: derived
  entity: Person
  dtype: Integer
  period: Month
  versions:
  - effective_from: '0001-01-01'
    formula: woman_birth_cohort + 1
""",
    )
    # Only printed numbers ground a quantity: a month number read from a month
    # name grounds a literal only where the module compares a month with it.
    grounded = _printed_numbers(PART_D_TEXT)
    unprinted = {
        str(label)
        for label in labels
        if float(label) not in grounded | GROUNDING_ALLOWED_VALUES
    }

    issues = find_ungrounded_numeric_issues(
        content,
        source_text=PART_D_TEXT,
        source_citation_path=PART_D_CITATION,
    )

    assert _rulespec_index_selectors(yaml.safe_load(content)["rules"]) == frozenset()
    assert _ungrounded_values(issues) == unprinted
    assert {
        label for label in map(str, labels) if label not in {"-1", "0", "1", "2", "3"}
    } <= _embedded_literals(tmp_path, content)


@pytest.mark.parametrize(
    ("variant", "kwargs"),
    [
        (
            "a comparison reads the selector",
            {
                "extra_rule": """- name: born_after_the_first_rows
  kind: derived
  entity: Person
  dtype: Judgment
  period: Month
  versions:
  - effective_from: '0001-01-01'
    formula: woman_birth_cohort >= 5
"""
            },
        ),
        (
            "a relation sums the selector",
            {
                "extra_rule": """- name: household_cohort_total
  kind: derived
  entity: Person
  dtype: Integer
  period: Month
  versions:
  - effective_from: '0001-01-01'
    formula: sum(member_of_household.woman_birth_cohort)
"""
            },
        ),
        (
            "a relation counts on the selector",
            {
                "extra_rule": """- name: household_late_cohorts
  kind: derived
  entity: Person
  dtype: Integer
  period: Month
  versions:
  - effective_from: '0001-01-01'
    formula: count_where(member_of_household, member_of_household.woman_birth_cohort >= 5)
"""
            },
        ),
        (
            "a list literal wraps the selector",
            {
                "extra_rule": """- name: doubled_cohort
  kind: derived
  entity: Person
  dtype: Integer
  period: Month
  versions:
  - effective_from: '0001-01-01'
    formula: max([woman_birth_cohort]) * 2
"""
            },
        ),
        (
            "a string hides a comparison",
            {
                "extra_rule": """- name: flagged_cohort
  kind: derived
  entity: Person
  dtype: Judgment
  period: Month
  versions:
  - effective_from: '0001-01-01'
    formula: x == '#' or woman_birth_cohort >= 5
"""
            },
        ),
        (
            "a table keyed by something else",
            {
                "extra_rule": """- name: month_bonus
  kind: parameter
  dtype: Integer
  indexed_by: birth_month
  versions:
  - effective_from: '0001-01-01'
    values:
      1: 100
      5: 200
- name: cohort_month_bonus
  kind: derived
  entity: Person
  dtype: Integer
  period: Month
  versions:
  - effective_from: '0001-01-01'
    formula: month_bonus[woman_birth_cohort]
"""
            },
        ),
        (
            "a Hebrew abbreviation in a comment hides arithmetic",
            {
                "extra_rule": "- name: cohort_amount\n  kind: derived\n  entity: Person\n  dtype: Integer\n  period: Month\n  versions:\n  - effective_from: '0001-01-01'\n    formula: |-\n      pension_age_completed_years[woman_birth_cohort]  # לוח א'1\n      + woman_birth_cohort * 1000  # חלק ד'\n"
            },
        ),
        (
            "a shekel abbreviation in a comment hides arithmetic",
            {
                "extra_rule": "- name: cohort_amount\n  kind: derived\n  entity: Person\n  dtype: Integer\n  period: Month\n  versions:\n  - effective_from: '0001-01-01'\n    formula: |-\n      pension_age_completed_years[woman_birth_cohort]  # in ש\"ח\n      + woman_birth_cohort * 1000  # ת\"ז\n"
            },
        ),
        (
            "an escaped double quote hides a comparison",
            {
                "extra_rule": '- name: cohort_flag\n  kind: derived\n  entity: Person\n  dtype: Judgment\n  period: Month\n  versions:\n  - effective_from: \'0001-01-01\'\n    formula: |-\n      x == "a\\\\"b" or woman_birth_cohort >= 5 or y == "c"\n'
            },
        ),
        (
            "an escaped single quote hides a comparison",
            {
                "extra_rule": "- name: cohort_flag\n  kind: derived\n  entity: Person\n  dtype: Judgment\n  period: Month\n  versions:\n  - effective_from: '0001-01-01'\n    formula: |-\n      x == 'a\\\\'b' or woman_birth_cohort >= 5 or y == 'c'\n"
            },
        ),
        (
            "a string spanning lines hides a comparison",
            {
                "extra_rule": '- name: cohort_flag\n  kind: derived\n  entity: Person\n  dtype: Judgment\n  period: Month\n  versions:\n  - effective_from: \'0001-01-01\'\n    formula: |-\n      x == "a\n      b" or woman_birth_cohort >= 5 or y == "c"\n'
            },
        ),
        (
            "a triple-quoted string hides arithmetic",
            {
                "extra_rule": '- name: cohort_flag\n  kind: derived\n  entity: Person\n  dtype: Judgment\n  period: Month\n  versions:\n  - effective_from: \'0001-01-01\'\n    formula: |-\n      x == """a\n      " + woman_birth_cohort * 2 + "\n      """\n'
            },
        ),
        ("the selector is not an integer", {"selector_dtype": "Count"}),
    ],
)
def test_a_rule_that_is_not_only_a_row_index_is_no_index_selector(variant, kwargs):
    content = _cohort_selector_module(SIXTEEN_ROWS, **kwargs)

    assert _rulespec_index_selectors(yaml.safe_load(content)["rules"]) == frozenset()
    assert _ungrounded_values(
        find_ungrounded_numeric_issues(
            content,
            source_text=PART_D_TEXT,
            source_citation_path=PART_D_CITATION,
        )
    ) >= {"10", "11", "13", "14", "15"}


def test_a_result_no_table_is_keyed_by_is_no_row_label():
    content = (
        _cohort_selector_module(SIXTEEN_ROWS)
        .replace("      15: 70\n", "")
        .replace("      15: 0\n", "")
    )

    assert _rulespec_index_selectors(yaml.safe_load(content)["rules"]) == frozenset()
    assert "15" in _ungrounded_values(
        find_ungrounded_numeric_issues(
            content,
            source_text=PART_D_TEXT,
            source_citation_path=PART_D_CITATION,
        )
    )


def test_a_branch_returning_an_expression_is_no_index_selector():
    content = _cohort_selector_module(SIXTEEN_ROWS).replace(
        "      else: 15", "      else: other_cohort"
    )

    assert _rulespec_index_selectors(yaml.safe_load(content)["rules"]) == frozenset()


def test_index_selector_labels_stay_grounded_when_grounding_is_scoped_to_proofs():
    content = _cohort_selector_module(SIXTEEN_ROWS)

    assert (
        find_ungrounded_numeric_issues_scoped(
            content,
            module_source_text=PART_D_TEXT,
            module_citation_path=PART_D_CITATION,
        )
        == []
    )


def _write_generated_part_d(tmp_path: Path, content: str) -> tuple[Path, Path, Path]:
    output_root = tmp_path / "out"
    rules_file = (
        output_root
        / "runner"
        / "statutes"
        / "national-insurance-law-1995"
        / "schedule-a1"
        / "sign-4.yaml"
    )
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(content)
    policy_repo = tmp_path / monorepo_checkout_name("il") / "il"
    policy_repo.mkdir(parents=True)
    return output_root, rules_file, policy_repo


# Selector lines as the embedded-scalar check reports them, one per shape: a
# label after a condition, after a compound condition, and the final else.
SELECTOR_LABEL_ISSUES = [
    ("4", "else: if birth_year <= 1940: 4"),
    ("6", "else: if birth_year <= 1942 and birth_month_name == 'אפריל': 6"),
    ("15", "else: 15"),
]


@pytest.mark.parametrize(("literal", "expression"), SELECTOR_LABEL_ISSUES)
def test_embedded_scalar_repair_never_lifts_an_index_selector_label(
    literal, expression, tmp_path
):
    content = _cohort_selector_module(SIXTEEN_ROWS)
    output_root, rules_file, policy_repo = _write_generated_part_d(tmp_path, content)

    repaired = _try_repair_generated_embedded_scalar_literals_for_apply(
        SimpleNamespace(output_file=rules_file, runner="runner"),
        output_root=output_root,
        policy_repo_path=policy_repo,
        issues=[
            "Embedded scalar literal: woman_birth_cohort line 1 embeds "
            f"{literal} in `{expression}`; extract the value to its own named "
            "numeric concept or indexed table/grid value"
        ],
    )

    assert repaired == []
    assert rules_file.read_text() == content


@pytest.mark.parametrize(("literal", "expression"), SELECTOR_LABEL_ISSUES)
def test_embedded_scalar_repair_still_lifts_a_quantity_selectors_result(
    literal, expression, tmp_path
):
    # The same literal in the same place, once the selector is also read as a
    # number, is a value like any other and the repair may name it.
    content = _cohort_selector_module(
        SIXTEEN_ROWS,
        extra_rule="""- name: cohort_rank
  kind: derived
  entity: Person
  dtype: Integer
  period: Month
  versions:
  - effective_from: '0001-01-01'
    formula: woman_birth_cohort + 1
""",
    )
    output_root, rules_file, policy_repo = _write_generated_part_d(tmp_path, content)

    repaired = _try_repair_generated_embedded_scalar_literals_for_apply(
        SimpleNamespace(output_file=rules_file, runner="runner"),
        output_root=output_root,
        policy_repo_path=policy_repo,
        issues=[
            "Embedded scalar literal: woman_birth_cohort line 1 embeds "
            f"{literal} in `{expression}`; extract the value to its own named "
            "numeric concept or indexed table/grid value"
        ],
    )

    assert repaired != []
    assert rules_file.read_text() != content


def test_a_public_selector_is_no_index_selector():
    # Another module could import a public rule and compare it, so only a
    # private rule's labels are exempt.
    content = _cohort_selector_module(SIXTEEN_ROWS).replace(
        "  metadata:\n    private: true\n", "", 1
    )
    payload = yaml.safe_load(content)
    assert "metadata" not in payload["rules"][0]

    assert _rulespec_index_selectors(payload["rules"]) == frozenset()
    assert _ungrounded_values(
        find_ungrounded_numeric_issues(
            content,
            source_text=PART_D_TEXT,
            source_citation_path=PART_D_CITATION,
        )
    ) >= {"10", "11", "13", "14", "15"}


@pytest.mark.parametrize("line_break", ["\r", "\x85", "\u2028", "\x0b"])
def test_a_line_break_other_than_newline_voids_the_exemption(line_break):
    payload = yaml.safe_load(_cohort_selector_module(SIXTEEN_ROWS))
    payload["rules"].append(
        {
            "name": "cohort_amount",
            "kind": "derived",
            "entity": "Person",
            "dtype": "Integer",
            "period": "Month",
            "versions": [
                {
                    "effective_from": "0001-01-01",
                    "formula": "pension_age_completed_years[woman_birth_cohort]"
                    f" # note{line_break}woman_birth_cohort * 2",
                }
            ],
        }
    )

    assert _rulespec_index_selectors(payload["rules"]) == frozenset()


# --- month numbers ground only month comparisons -----------------------------

DATED_DECEMBER_TEXT = "הקצבה תשולם החל מיום 1 בדצמבר 1940 בסכום של 500 שקלים לחודש"


def _month_module(formula: str, *, name: str = "monthly_benefit") -> str:
    return f"""format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {PART_D_CITATION}
rules:
- name: {name}
  kind: derived
  entity: Person
  dtype: Money
  period: Month
  versions:
  - effective_from: '0001-01-01'
    formula: |-
      {formula}
"""


def _ungrounded_for(formula: str) -> set[str]:
    return _ungrounded_values(
        find_ungrounded_numeric_issues(
            _month_module(formula),
            source_text=DATED_DECEMBER_TEXT,
            source_citation_path=PART_D_CITATION,
        )
    )


def test_a_dated_month_does_not_ground_an_unrelated_literal():
    # "1 בדצמבר 1940" names December. Before the scoping it also grounded the
    # annualising 12 here, which the source never prints.
    assert _ungrounded_for("monthly_amount * 12 + 500") == {"12"}


@pytest.mark.parametrize(
    "formula",
    (
        "if birth_month >= 12: 500 else: 0",
        "if 12 <= birth_month: 500 else: 0",
        "if person.birth_month == 12: 500 else: 0",
        "if month_of_birth != 12: 0 else: 500",
    ),
)
def test_a_dated_month_grounds_a_month_comparison(formula):
    assert _ungrounded_for(formula) == set()


@pytest.mark.parametrize(
    "formula",
    (
        # The same value also used as a quantity: no month grounding at all.
        "if birth_month >= 12: monthly_amount * 12 else: 500",
        # A count of months is a quantity, not a month.
        "if months_employed >= 12: 500 else: 0",
        "if month_count >= 12: 500 else: 0",
        "if age_in_months >= 12: 500 else: 0",
        # Arithmetic on the month operand is not a plain month comparison.
        "if birth_month + 1 >= 12: 500 else: 0",
        "if birth_month >= 12 * 1: 500 else: 0",
        "if 1 * 12 <= birth_month: 500 else: 0",
    ),
)
def test_a_dated_month_grounds_nothing_outside_a_plain_month_comparison(formula):
    assert "12" in _ungrounded_for(formula)


def test_month_context_values_require_every_use_to_be_a_month():
    assert _month_comparison_literal_values(
        _month_module("if birth_month <= 6: 1 else: 2")
    ) == frozenset({6.0})
    assert (
        _month_comparison_literal_values(
            _month_module("if birth_month <= 6: 6 else: 2")
        )
        == frozenset()
    )


def test_other_grounding_consumers_ignore_month_numbers_by_default():
    # The month number is in a stream of its own, which only the grounding
    # decisions read; the shared stream holds only what the source prints.
    month_occurrences = extract_typed_month_occurrences_from_text(DATED_DECEMBER_TEXT)
    assert [
        (occurrence.value, occurrence.requires_month_context)
        for occurrence in month_occurrences
    ] == [(12.0, True)]
    assert not any(
        occurrence.requires_month_context or occurrence.value == 12.0
        for occurrence in extract_typed_numeric_occurrences_from_text(
            DATED_DECEMBER_TEXT
        )
    )
    assert 12.0 not in extract_numbers_from_text(DATED_DECEMBER_TEXT)
    assert not numeric_value_is_grounded(12.0, month_occurrences)
    assert numeric_value_is_grounded(
        12.0, month_occurrences, month_context_values=frozenset({12.0})
    )


@pytest.mark.parametrize(
    "text", ("בחודשים מאי ויוני", "מהחודשים מאי, יוני", "ביום 4 במרץ עד אפריל")
)
def test_a_homograph_listed_with_another_month_is_dated_with_the_list(text):
    # The homograph gate covers groups made only of homographs: what dates a
    # non-homograph month in a list (the plural "months" before it, a day)
    # dates every name in the list.
    values = {value for _raw, value, _temporal in _month_occurrences(text)}
    assert len(values) == 2


@pytest.mark.parametrize(
    "text", ("מאי ומרץ", "מרץ, מאי", "בחודשים מאי ומרץ", "ביום 4 במרץ עד מאי")
)
def test_a_list_of_only_homographs_needs_a_year_or_in_the_month_of(text):
    assert _month_occurrences(text) == set()
    assert {3.0, 5.0} <= {value for _r, value, _t in _month_occurrences(text + " 1990")}


def test_rebased_month_occurrence_still_grounds_only_month_comparisons():
    from axiom_encode.harness.source_completeness import _shift_numeric_occurrence

    (month,) = extract_typed_month_occurrences_from_text(DATED_DECEMBER_TEXT)
    shifted = _shift_numeric_occurrence(month, 7)

    assert shifted.requires_month_context
    assert not numeric_value_is_grounded(12.0, [shifted])


# --- month scoping on every grounding path -----------------------------------


def _scoped_ungrounded_for(formula: str) -> set[str]:
    return _ungrounded_values(
        find_ungrounded_numeric_issues_scoped(
            _month_module(formula),
            module_source_text=DATED_DECEMBER_TEXT,
            module_citation_path=PART_D_CITATION,
        )
    )


def test_proof_scoped_path_grounds_only_month_comparisons():
    assert _scoped_ungrounded_for("if birth_month >= 12: 500 else: 0") == set()
    assert _scoped_ungrounded_for("monthly_amount * 12 + 500") == {"12"}


def test_flat_fallback_path_grounds_only_month_comparisons(monkeypatch):
    # Force evaluate_numeric_grounding_values past its per-rule decisions and
    # onto the flat return path.
    from axiom_encode.harness import validator_pipeline as vp

    monkeypatch.setattr(vp, "_rule_grounding_values_by_path", lambda *_a, **_k: [])
    assert _ungrounded_for("if birth_month >= 12: 500 else: 0") == set()
    assert _ungrounded_for("monthly_amount * 12 + 500") == {"12"}


def _month_module_with_extra(formula: str, extra_rule: str) -> str:
    return _month_module(formula) + extra_rule


@pytest.mark.parametrize(
    "extra_rule",
    (
        # A table value equal to the compared month.
        """- name: month_supplement
  kind: parameter
  dtype: Money
  indexed_by: supplement_band
  versions:
  - effective_from: '0001-01-01'
    values:
      1: 12
""",
        # A scalar formula equal to the compared month.
        """- name: annual_periods
  kind: parameter
  dtype: Integer
  versions:
  - effective_from: '0001-01-01'
    formula: 12
""",
    ),
)
def test_table_and_scalar_values_never_share_month_grounding(extra_rule):
    content = _month_module_with_extra("if birth_month >= 12: 500 else: 0", extra_rule)
    assert _month_comparison_literal_values(content) == frozenset()
    issues = find_ungrounded_numeric_issues(
        content,
        source_text=DATED_DECEMBER_TEXT,
        source_citation_path=PART_D_CITATION,
    )
    assert "12" in _ungrounded_values(issues)


@pytest.mark.parametrize(
    ("formula", "source", "literal"),
    (
        ("if benefit_month_amount > 9: 500 else: 0", "ספטמבר 1939", "9"),
        ("if payment_month_rate >= 6: 500 else: 0", "ביוני 1950", "6"),
        ("if month_value == 6: 500 else: 0", "ביוני 1950", "6"),
        ("if month_sum <= 6: 500 else: 0", "ביוני 1950", "6"),
    ),
)
def test_a_quantity_named_with_month_is_not_a_month(formula, source, literal):
    issues = find_ungrounded_numeric_issues(
        _month_module(formula),
        source_text=source,
        source_citation_path=PART_D_CITATION,
    )
    assert literal in _ungrounded_values(issues)


# --- month-operand names: an allowlist, not a denylist -------------------------

HALF_YEAR_TEXT = (
    "המבוטח זכאי לגמלה אם שילם דמי ביטוח במשך חצי שנה לפחות, החל מיום 1 ביוני 2010"
)
ONE_YEAR_TEXT = "המבוטח זכאי לגמלה אם שילם דמי ביטוח במשך שנה, החל מיום 31 בדצמבר 2010"


def _reported(formula: str, source: str) -> set[str]:
    return _ungrounded_values(
        find_ungrounded_numeric_issues(
            _month_module(formula),
            source_text=source,
            source_citation_path=PART_D_CITATION,
        )
    )


@pytest.mark.parametrize(
    ("formula", "source", "literal"),
    (
        # A half year or a year next to a commencement date: the 6 or 12 is
        # the encoder's own conversion and must stay reported.
        ("if qualifying_month_threshold >= 6: 1 else: 0", HALF_YEAR_TEXT, "6"),
        ("if waiting_month_limit >= 6: 1 else: 0", HALF_YEAR_TEXT, "6"),
        ("if month_cap >= 6: 1 else: 0", HALF_YEAR_TEXT, "6"),
        ("if month_fee > 6: 1 else: 0", HALF_YEAR_TEXT, "6"),
        ("if elapsed_month >= 6: 1 else: 0", HALF_YEAR_TEXT, "6"),
        ("if insured_month_span >= 6: 1 else: 0", HALF_YEAR_TEXT, "6"),
        ("if insurance_period_month >= 6: 1 else: 0", HALF_YEAR_TEXT, "6"),
        ("if contribution_month_floor >= 12: 1 else: 0", ONE_YEAR_TEXT, "12"),
        # An age in months is not a calendar month either.
        ("if age_month >= 6: 1 else: 0", HALF_YEAR_TEXT, "6"),
    ),
)
def test_a_month_named_quantity_or_span_reports_the_literal(formula, source, literal):
    assert literal in _reported(formula, source)


@pytest.mark.parametrize(
    ("formula", "source"),
    (
        ("if birth_month >= 6: 1 else: 0", HALF_YEAR_TEXT),
        ("if payment_month == 6: 1 else: 0", HALF_YEAR_TEXT),
        ("if month < 6: 1 else: 0", HALF_YEAR_TEXT),
        ("if person.month_of_birth <= 12: 1 else: 0", ONE_YEAR_TEXT),
    ),
)
def test_a_calendar_month_name_still_grounds_its_dated_month(formula, source):
    assert _reported(formula, source) == set()


def _selector_payload(selector_formula: str) -> dict:
    return yaml.safe_load(
        f"""format: rulespec/v1
rules:
- name: cohort
  kind: derived
  entity: Person
  dtype: Integer
  period: Month
  metadata:
    private: true
  versions:
  - effective_from: '0001-01-01'
    formula: |-
      {selector_formula}
- name: pension_age
  kind: parameter
  dtype: Integer
  indexed_by: cohort
  versions:
  - effective_from: '0001-01-01'
    values:
      3: 65
      4: 66
- name: woman_pension_age
  kind: derived
  entity: Person
  dtype: Integer
  period: Month
  versions:
  - effective_from: '0001-01-01'
    formula: pension_age[cohort]
"""
    )


def test_an_index_selector_needs_a_conditional_with_two_branches():
    two_branches = _selector_payload("if birth_year < 1940: 3 else: 4")
    one_branch = _selector_payload("3")

    assert _rulespec_index_selectors(two_branches["rules"]) == frozenset({"cohort"})
    assert _rulespec_index_selectors(one_branch["rules"]) == frozenset()


@pytest.mark.parametrize(
    ("formula", "source", "literal"),
    (
        # A quantity word anywhere before "month", not only right before it.
        ("if min_insured_month >= 6: 1 else: 0", HALF_YEAR_TEXT, "6"),
        ("if max_benefit_month <= 12: 1 else: 0", ONE_YEAR_TEXT, "12"),
        ("if required_contribution_month >= 12: 1 else: 0", ONE_YEAR_TEXT, "12"),
        ("if insured_month >= 6: 1 else: 0", HALF_YEAR_TEXT, "6"),
        # Structure: month must be the last token, and month_of_ takes one.
        ("if month_index >= 6: 1 else: 0", HALF_YEAR_TEXT, "6"),
        ("if month_of_birth_date >= 6: 1 else: 0", HALF_YEAR_TEXT, "6"),
    ),
)
def test_month_operand_allowlist_structure_reports_the_literal(
    formula, source, literal
):
    assert literal in _reported(formula, source)


EXPECTED_DISQUALIFYING_TOKENS = frozenset(
    {
        "age", "amount", "cap", "ceiling", "count", "duration", "elapsed",
        "fee", "floor", "insured", "length", "limit", "max", "maximum", "min",
        "minimum", "num", "number", "period", "qualifying", "rate", "required",
        "span", "sum", "threshold", "total", "value", "waiting",
    }
)  # fmt: skip


def _disqualifying_tokens() -> list[str]:
    from axiom_encode.harness import validator_pipeline as vp

    return sorted(vp._MONTH_OPERAND_DISQUALIFYING_TOKENS)


@pytest.mark.parametrize("token", sorted(EXPECTED_DISQUALIFYING_TOKENS))
def test_every_disqualifying_token_before_month_reports_the_literal(token):
    # Parametrized over a static copy of the list, so removing any one token
    # from the code fails that token's case.
    assert "6" in _reported(f"if {token}_month >= 6: 1 else: 0", HALF_YEAR_TEXT)


def test_disqualifying_token_list_is_the_documented_list():
    # Catches additions as well as removals, so the static copy stays current.
    assert frozenset(_disqualifying_tokens()) == EXPECTED_DISQUALIFYING_TOKENS
