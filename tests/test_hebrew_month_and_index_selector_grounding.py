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
    _rulespec_index_selectors,
    extract_numbers_from_text,
    extract_numeric_occurrences_from_text,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    find_ungrounded_numeric_issues,
    find_ungrounded_numeric_issues_scoped,
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
    "{m} ועד דצמבר",
    "(5 {m} 2008)",
)
# Each sentence uses "{m}" with nothing that dates it.
UNDATED_CONTEXTS = (
    "{m}",
    "הסובל {m} ספיקת לב קשה",
    "המילה {m} נכתבה כאן",
    "65 ו־4 חודשים\n{m} ספיקה",
    "| 4 | {m} ספיקה",
    "1939\n{m} ספיקה",
)


def _month_occurrences(text: str) -> set[tuple[str, float, bool]]:
    """The grounding occurrences a month name produced: raw word, value, temporal."""
    return {
        (occurrence.raw, occurrence.value, occurrence.has_temporal_context)
        for occurrence in extract_typed_numeric_occurrences_from_text(text)
        if not any(character.isdigit() for character in occurrence.raw)
        and any(month in occurrence.raw for month in _HEBREW_GREGORIAN_MONTH_NUMBERS)
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


def test_part_d_grounds_exactly_the_months_it_names():
    grounded = extract_numbers_from_text(PART_D_TEXT)

    assert PART_D_MONTH_NUMBERS <= grounded
    assert not {10.0, 11.0} & grounded


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
    ],
)
def test_month_homographs_ground_nothing(text):
    assert not ({3.0, 5.0} & extract_numbers_from_text(text))


def test_a_list_of_months_is_dated_as_a_whole():
    # July is dated by the year after August, and a list after "the months"
    # is dated by that word however long it runs.
    assert {7.0, 8.0} <= extract_numbers_from_text("יולי ואוגוסט 1939")
    assert {1.0, 4.0, 7.0, 10.0} <= extract_numbers_from_text(
        "בכל אחד מהחודשים אפריל, יולי, אוקטובר וינואר, בעבור הרבעון"
    )
    assert {2.0, 11.0} <= extract_numbers_from_text(
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
    grounded = extract_numbers_from_text(PART_D_TEXT)
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
