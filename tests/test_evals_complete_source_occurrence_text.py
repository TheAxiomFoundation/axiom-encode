"""The complete-mode numeric-occurrence count reads the source positionally (#1801).

In complete-source-unit mode the encode eval refuses a candidate when a source
number has no named scalar ("Source numeric value N appears K time(s) ...").
That check read the raw corpus body, so it demanded again the page numbers,
table labels and locators that complete-source numeric recall
(`analyze_complete_source_unit`) removes from its demand. It now counts
`complete_source_numeric_recall_occurrences`: the source inventory minus the
occurrences whose characters the numeric-recall reading removes in place, plus
every number numeric recall demands.

Invariant: the occurrence view contains every number numeric recall demands
(its source view), and it drops a source number only when a recall mask or
filter removes that number's own characters. The masked text is never
re-tokenized for the view, so a real value whose characters survive the masks
stays demanded even when typed recall loses it to a different tokenization of
the masked text (the open follow-up issue).

Source text in the unit tests is synthetic or a few words of a real
provision; the property test reads the committed replay fixtures.
"""

from __future__ import annotations

import functools
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest

from axiom_encode.harness import evals
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.evals import evaluate_artifact
from axiom_encode.harness.validator_pipeline import (
    ValidationResult,
    ValidatorPipeline,
    _numeric_profile_for_citation_path,
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)
from tests.test_evals import (
    _canonical_rulespec_content_root,
    _write_test_corpus_provision,
)

GUIDANCE = "us/guidance/agency/program/standards-memo/page-3"
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "us_replay_benchmark"

# A flattened one-line guidance page: a page header, a jurisdiction-set column
# label (`Household Size 48 States, D.C., ...`) and a household-size table.
TABLE_PAGE = (
    "Page 3 of 7 PROGRAM INCOME STANDARDS Net Monthly Income Standards "
    "Household Size 48 States, D.C., Guam, Virgin Islands Alaska Hawaii "
    "1 $1,330 $1,663 $1,530 2 $1,804 $2,255 $2,075 "
    "Each additional member $474 $592 $545"
)
TABLE_AMOUNTS = (1330, 1663, 1530, 1804, 2255, 2075, 474, 592, 545)


def _module(citation: str, amounts, *, sizes=(1, 2)) -> str:
    rules = []
    for index, amount in enumerate(amounts):
        rules.append(
            f"""  - name: amount_{index}
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-10-01'
        formula: {amount}
"""
        )
    for size in sizes:
        rules.append(
            f"""  - name: household_size_{size}
    kind: parameter
    dtype: Count
    versions:
      - effective_from: '2026-10-01'
        formula: {size}
"""
        )
    return (
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        f"    corpus_citation_path: {citation}\nrules:\n" + "".join(rules)
    )


def _metrics(
    tmp_path: Path,
    source_text: str,
    content: str,
    *,
    complete=True,
    citation=GUIDANCE,
    caller_source_text=None,
):
    corpus_release = _write_test_corpus_provision(
        tmp_path, citation_path=citation, body=source_text
    )
    rulespec_file = tmp_path / "policies" / "agency" / "program" / "page-3.yaml"
    rulespec_file.parent.mkdir(parents=True)
    rulespec_file.write_text(content)
    passed = ValidationResult("compile", True, issues=[])
    with (
        patch.object(ValidatorPipeline, "_run_compile_check", return_value=passed),
        patch.object(
            ValidatorPipeline,
            "_run_ci",
            return_value=ValidationResult("ci", True, issues=[]),
        ),
    ):
        return evaluate_artifact(
            rulespec_file=rulespec_file,
            policy_repo_root=_canonical_rulespec_content_root(
                tmp_path, citation.split("/", 1)[0]
            ),
            axiom_rules_path=Path("/tmp/axiom-rules-engine"),
            source_text=(
                source_text if caller_source_text is None else caller_source_text
            ),
            local_corpus_release=corpus_release,
            source_citation_path=citation,
            require_complete_source_unit=complete,
            skip_reviewers=True,
        )


def _recall_values(source_text: str, citation: str = GUIDANCE) -> list[float]:
    profile = _numeric_profile_for_citation_path(citation)
    return [
        occurrence.value
        for occurrence in sc.complete_source_numeric_recall_occurrences(
            source_text,
            corpus_citation_path=citation,
            extract_numeric_occurrences=functools.partial(
                extract_typed_numeric_inventory_occurrences_from_text,
                profile=profile,
            ),
        )
    ]


def test_page_number_and_table_label_are_not_demanded_again(tmp_path):
    metrics = _metrics(tmp_path, TABLE_PAGE, _module(GUIDANCE, TABLE_AMOUNTS))

    assert metrics.numeric_occurrence_issues == []
    # 9 amounts and the two household-size rows; not 3, 7 or 48.
    assert metrics.source_numeric_occurrence_count == 11
    assert sorted(_recall_values(TABLE_PAGE)) == sorted(
        [1.0, 2.0, *map(float, TABLE_AMOUNTS)]
    )


@pytest.mark.parametrize("dropped", [1330, 592, 545])
def test_every_table_amount_is_still_demanded(tmp_path, dropped):
    amounts = [amount for amount in TABLE_AMOUNTS if amount != dropped]
    metrics = _metrics(tmp_path, TABLE_PAGE, _module(GUIDANCE, amounts))

    assert metrics.numeric_occurrence_issues == [
        f"Source numeric value {dropped:g} appears 1 time(s), "
        "but only 0 named scalar definition(s) with that value were found."
    ]
    assert not metrics.ci_pass


def test_row_number_is_still_demanded(tmp_path):
    metrics = _metrics(
        tmp_path, TABLE_PAGE, _module(GUIDANCE, TABLE_AMOUNTS, sizes=(1,))
    )

    assert metrics.numeric_occurrence_issues == [
        "Source numeric value 2 appears 1 time(s), "
        "but only 0 named scalar definition(s) with that value were found."
    ]


@pytest.mark.parametrize(
    "source_text",
    (
        # A prose count of the same jurisdictions stays a value.
        "The 48 States and D.C. pay $1,330 each month.",
        # Without the column caption the label shape is not recognized.
        "Area 48 States, D.C., Guam, Virgin Islands 1 $1,330",
    ),
)
def test_a_48_that_numeric_recall_demands_stays_demanded(tmp_path, source_text):
    metrics = _metrics(tmp_path, source_text, _module(GUIDANCE, (1330,), sizes=()))

    assert 48.0 in _recall_values(source_text)
    assert any(
        issue.startswith("Source numeric value 48 appears 1 time(s)")
        for issue in metrics.numeric_occurrence_issues
    )


def test_default_mode_still_counts_the_module_scope(tmp_path):
    # Outside complete mode nothing changes: the default lane counts its own
    # operating-slice text, page header included.
    metrics = _metrics(
        tmp_path, TABLE_PAGE, _module(GUIDANCE, TABLE_AMOUNTS), complete=False
    )

    assert any(
        issue.startswith("Source numeric value 48 appears")
        for issue in metrics.numeric_occurrence_issues
    )


@pytest.mark.parametrize(
    "citation", (GUIDANCE, "de/statute/estg/32a"), ids=("us-legacy", "de-DE")
)
def test_eval_passes_the_recall_view_its_corpus_text_path_and_profile(
    tmp_path, citation
):
    calls = []
    original = sc.complete_source_numeric_recall_occurrences

    def recording(text, **kwargs):
        calls.append((text, kwargs))
        return original(text, **kwargs)

    with patch.object(
        evals, "complete_source_numeric_recall_occurrences", side_effect=recording
    ):
        _metrics(
            tmp_path,
            SOURCE_WITH_REFERENCE,
            _module(citation, TABLE_AMOUNTS),
            citation=citation,
            # The caller's operating-slice text differs from the corpus body;
            # complete mode counts the corpus body.
            caller_source_text="Household Size 1 $1,330",
        )

    assert len(calls) == 1
    text, kwargs = calls[0]
    # The unmasked corpus body, as numeric recall receives it.
    assert text == SOURCE_WITH_REFERENCE
    assert kwargs["corpus_citation_path"] == citation
    extractor = kwargs["extract_numeric_occurrences"]
    assert extractor.func is extract_typed_numeric_inventory_occurrences_from_text
    assert extractor.keywords == {
        "profile": _numeric_profile_for_citation_path(citation)
    }
    assert kwargs["exempt_source_spans"] == ()


SOURCE_WITH_REFERENCE = TABLE_PAGE + " (See Example 1 )"


@pytest.mark.parametrize("citation", ("us-zz/manual/agency/handbook/block-3", GUIDANCE))
def test_recall_view_reads_the_stripped_text_in_numeric_recall_scopes(citation):
    seen = []
    original = sc.authoritative_numeric_recall_text

    def recording(text, **kwargs):
        seen.append(
            (
                text,
                sc._US_MANUAL_PROSE_SCOPE.get(),
                sc._CONDITION_CUE_CITATION_PATH.get(),
            )
        )
        return original(text, **kwargs)

    with patch.object(sc, "authoritative_numeric_recall_text", side_effect=recording):
        _recall_values("\n  " + TABLE_PAGE + "\n", citation)

    assert seen == [(TABLE_PAGE, sc._is_us_manual_citation_path(citation), citation)]


def test_view_never_re_reads_the_masked_text():
    # An extractor that reads only the unmasked source: the masked recall text
    # yields nothing. The view still demands every source number that no mask
    # removed; a view built from the masked text would demand none.
    profile = _numeric_profile_for_citation_path(GUIDANCE)

    def source_only(text):
        if text != TABLE_PAGE:
            return []
        return extract_typed_numeric_inventory_occurrences_from_text(
            text, profile=profile
        )

    view = sc.complete_source_numeric_recall_occurrences(
        TABLE_PAGE,
        corpus_citation_path=GUIDANCE,
        extract_numeric_occurrences=source_only,
    )
    assert sorted(occurrence.value for occurrence in view) == sorted(
        [1.0, 2.0, *map(float, TABLE_AMOUNTS)]
    )


def test_exempt_source_spans_remove_only_their_characters():
    source_text = "  Pay $5 or round up by 1 then add 1."
    start = source_text.index("by 1") + 3
    profile = _numeric_profile_for_citation_path(GUIDANCE)
    view = sc.complete_source_numeric_recall_occurrences(
        source_text,
        corpus_citation_path=GUIDANCE,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        exempt_source_spans=((start, start + 1),),
    )
    assert sorted(occurrence.value for occurrence in view) == [1.0, 5.0]


def test_a_number_is_removed_only_when_all_its_characters_are():
    # Removing one digit of `15` (here by an exempt span) does not remove the
    # number: only an occurrence whose every character is gone leaves the view.
    # The extractor reads only the source, so no recall demand is added back.
    source_text = "Page 2 Pay 15 dollars."
    start = source_text.index("15")
    profile = _numeric_profile_for_citation_path(GUIDANCE)

    def source_only(text):
        if text != source_text:
            return []
        return extract_typed_numeric_inventory_occurrences_from_text(
            text, profile=profile
        )

    view = sc.complete_source_numeric_recall_occurrences(
        source_text,
        corpus_citation_path=GUIDANCE,
        extract_numeric_occurrences=source_only,
        exempt_source_spans=((start, start + 1),),
    )
    assert [occurrence.value for occurrence in view] == [15.0]


def test_a_number_numeric_recall_demands_is_added_when_the_source_reads_it_otherwise():
    # The source inventory reads `62 FR 61344` and the deleted public-law
    # numbers differently from the recall text; every number numeric recall
    # demands is still in the view.
    source_text = "Under Public Law 119-21 and 62 FR 61344 the rate is 25 percent."
    citation = "us/statute/7/1"
    assert {62.0, 0.25} <= set(_recall_values(source_text, citation))
    assert 119.0 not in _recall_values(source_text, citation)


def test_half_up_increment_is_still_spelled_out_in_complete_mode(tmp_path):
    citation = "ca/policy/cra/example/rounding"
    source_text = (
        "If the 3rd digit is 5 or more, increase the 2nd digit by 1\n"
        "If the 3rd digit is 5 or more, the 2nd digit shall be increased by 1."
    )
    corpus_release = _write_test_corpus_provision(
        tmp_path, citation_path=citation, body=source_text
    )
    rulespec_file = tmp_path / "policies" / "cra" / "example" / "rounding.yaml"
    rulespec_file.parent.mkdir(parents=True)
    rulespec_file.write_text(
        f"""format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {citation}
rules:
  - name: rounding_half_unit
    kind: parameter
    dtype: Decimal
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: parameter
            source:
              corpus_citation_path: {citation}
              excerpt: if the 3rd digit is 5 or more, increase the 2nd digit by 1
    versions:
      - effective_from: '2025-01-01'
        formula: 0.5
"""
    )
    passed = ValidationResult("compile", True, issues=[])
    with (
        patch.object(ValidatorPipeline, "_run_compile_check", return_value=passed),
        patch.object(
            ValidatorPipeline,
            "_run_ci",
            return_value=ValidationResult("ci", True, issues=[]),
        ),
    ):
        metrics = evaluate_artifact(
            rulespec_file=rulespec_file,
            policy_repo_root=_canonical_rulespec_content_root(tmp_path, "ca"),
            axiom_rules_path=Path("/tmp/axiom-rules-engine"),
            source_text=source_text,
            local_corpus_release=corpus_release,
            source_citation_path=citation,
            require_complete_source_unit=True,
            skip_reviewers=True,
        )

    # The two `by 1` increments are spelled out, as before this change; the
    # ordinals 3rd and 2nd are not counted.
    assert not any(
        issue.startswith("Source numeric value 1 appears")
        for issue in metrics.numeric_occurrence_issues
    )


# Early returns of `analyze_complete_source_unit` that refuse before numeric
# recall runs; their source-occurrence count is 0 by construction.
_EARLY_REFUSALS = (
    "[complete-source-unit:unanalyzable]",
    "[complete-source-unit:declaration-collision]",
    "[complete-source-unit:authoritative-source]",
)


def _replay_fixtures():
    for directory in sorted(FIXTURE_ROOT.iterdir()):
        if (directory / "candidate.yaml").is_file():
            yield directory


@pytest.mark.parametrize(
    "fixture", list(_replay_fixtures()), ids=lambda path: path.name
)
def test_view_contains_every_number_numeric_recall_demands(fixture):
    """Property over the replay fixtures: view ⊇ numeric recall's source view."""

    citation = json.loads((fixture / "fixture.json").read_text())["citation"]
    source_text = (fixture / "source.txt").read_text()
    content = (fixture / "candidate.yaml").read_text()
    profile = _numeric_profile_for_citation_path(citation)
    analysis = sc.analyze_complete_source_unit(
        content,
        source_text,
        corpus_citation_path=citation,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    view = _recall_values(source_text, citation)

    if analysis.source_numeric_occurrence_count == 0 and any(
        str(issue).startswith(_EARLY_REFUSALS) for issue in analysis.issues
    ):
        pytest.skip("numeric recall did not run: the candidate is refused earlier")
    missing = Counter(analysis.source_numeric_occurrence_values) - Counter(view)
    assert not missing


# --- A quantity after a citation keyword is not a locator --------------------
#
# Short excerpts of real US provisions (MN, CA, NJ) and two constructed ones.
# `_ENGLISH_LEGAL_CITATION` took the rate as part of the citation, so numeric
# recall did not demand it and the raw occurrence count was its only gate.
RATE_AFTER_CITATION = (
    (
        "us-mn/statute/290.06/block-58",
        "the credit equals, for expenses under section 290.9725 , 25 percent of "
        "the expenses paid.",
        0.25,
    ),
    (
        "us-ca/statute/rtc/18501",
        "the amount of the credit under paragraph (2) of Section 17054, 3 percent "
        "for the first dependent credit.",
        0.03,
    ),
    (
        "us-nj/statute/43:13-9",
        "each employer benefited by this article 3% of the amount paid.",
        0.03,
    ),
    (
        "us-zz/statute/1/1",
        "Rent is limited under section 8 and 50 percent of area median income.",
        0.5,
    ),
    (
        "us-zz/statute/1/2",
        "Rent is limited under section 8, 30 percent of monthly adjusted income.",
        0.3,
    ),
)


def _scalar_module(citation: str, value: float | None) -> str:
    rules = (
        f"""  - name: stated_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2026-01-01'
        formula: {value}
"""
        if value is not None
        else ""
    )
    return (
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        f"    corpus_citation_path: {citation}\nrules:\n" + (rules or "  []\n")
    )


def _typed_unnamed(content: str, source_text: str, citation: str) -> set[float]:
    profile = _numeric_profile_for_citation_path(citation)
    analysis = sc.analyze_complete_source_unit(
        content,
        source_text,
        corpus_citation_path=citation,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    prefix = "[complete-source-unit:numeric-recall] Authoritative corpus numeric value "
    return {
        float(str(issue)[len(prefix) :].split(" ", 1)[0])
        for issue in analysis.issues
        if str(issue).startswith(prefix)
    }


@pytest.mark.parametrize(("citation", "source_text", "rate"), RATE_AFTER_CITATION)
def test_rate_after_a_citation_is_demanded_by_both_gates(citation, source_text, rate):
    assert rate in _recall_values(source_text, citation)
    assert rate in _typed_unnamed(_scalar_module(citation, None), source_text, citation)
    assert rate not in _typed_unnamed(
        _scalar_module(citation, rate), source_text, citation
    )


@pytest.mark.parametrize(
    ("source_text", "recall_text"),
    (
        ("Apply section 290.9725 to it.", "Apply  to it."),
        ("Apply sections 3, 4 and 5 to it.", "Apply  to it."),
        # A unit word later in the prose does not make a locator a quantity.
        (
            "Appeal under section 39-2-125 within thirty days.",
            "Appeal under  within thirty days.",
        ),
        ("Apply section 3 and 25 percent to it.", "Apply  and 25 percent to it."),
        ("Apply section 3, 12 per cent to it.", "Apply , 12 per cent to it."),
        ("Apply section 3, 30 days after it.", "Apply , 30 days after it."),
        # Read whole: never `section 2` with `5 percent` left behind.
        ("Apply section 25 percent to it.", "Apply section 25 percent to it."),
        ("Apply article 3% to it.", "Apply article 3% to it."),
    ),
)
def test_citation_numbers_stay_masked_and_quantities_stay_whole(
    source_text, recall_text
):
    assert (
        sc.authoritative_numeric_recall_text(
            source_text, corpus_citation_path="us-zz/statute/1/3"
        )
        == recall_text
    )


def test_rate_after_a_citation_is_counted_by_the_occurrence_check(tmp_path):
    citation, source_text, rate = RATE_AFTER_CITATION[0]
    metrics = _metrics(
        tmp_path,
        source_text,
        _scalar_module(citation, None),
        citation=citation,
    )
    assert any(
        issue.startswith(f"Source numeric value {rate:g} appears 1 time(s)")
        for issue in metrics.numeric_occurrence_issues
    )
    assert not metrics.ci_pass


# --- The end of an applicability range stays demanded -------------------------


@pytest.mark.parametrize(
    ("source_text", "demanded"),
    (
        (
            "For taxable years beginning after 2025 and before 2030, the credit "
            "is $500.",
            [2030.0, 500.0],
        ),
        (
            "For taxable years beginning after December 31, 2025, the credit is $500.",
            [500.0],
        ),
    ),
)
def test_range_end_year_is_demanded_and_start_year_is_not(source_text, demanded):
    citation = "us/statute/26/1"
    assert _recall_values(source_text, citation) == demanded
    assert set(demanded) <= _typed_unnamed(
        _scalar_module(citation, None), source_text, citation
    )


# --- Documented fail-closed gaps -----------------------------------------------


def test_unnormalized_module_path_reads_fewer_masks_never_more():
    # With no `source_citation_path`, the eval passes the module's path as
    # written. A path that is not canonical (here: surrounding space) does not
    # select the US-only masks, so the view demands more, never less.
    canonical = _recall_values(TABLE_PAGE + " P.L. 119-21", GUIDANCE)
    unnormalized = _recall_values(TABLE_PAGE + " P.L. 119-21", f" {GUIDANCE} ")
    assert sorted(canonical) == sorted([1.0, 2.0, *map(float, TABLE_AMOUNTS)])
    remaining = list(unnormalized)
    for value in canonical:
        remaining.remove(value)


# --- Half-up increment: only a whole `1` -------------------------------------


@pytest.mark.parametrize(
    ("instruction", "spelled"),
    (
        ("If the 3rd digit is 5 or more, increase the 2nd digit by 1", True),
        ("If the 3rd digit is 5 or more, the 2nd digit shall be increased by 1.", True),
        ("If the 3rd digit is 5 or more, increase the 2nd digit by 1,000", False),
        ("If the 3rd digit is 5 or more, increase the 2nd digit by 1.5 percent", False),
    ),
)
def test_half_up_rewrite_spells_only_a_whole_one(instruction, spelled):
    from axiom_encode.harness.validator_pipeline import (
        normalize_source_backed_half_up_rounding_occurrence_text,
    )

    rewritten = normalize_source_backed_half_up_rounding_occurrence_text(instruction)
    assert ("by one" in rewritten) is spelled
    if not spelled:
        assert rewritten == instruction


# --- Regression: real values of the corpus sweep stay demanded -----------------
#
# A few words of each provision where a real rate or amount lost its gate on
# the first version of this change (typed recall's re-reading of masked text,
# or a mask eating the number). The positional view keeps each one.
SWEEP_REAL_VALUES = (
    ("us/statute/7/2028/e", "for the distribution of more than 20 percent of the", 0.2),
    (
        "us-nj/regulation/njac-10-90/10-90-3.3",
        "total income exceeds 130 percent of the",
        1.3,
    ),
    (
        "us-ma/statute/62/4/block-2",
        "taxed at the rate of 5.3 per cent for tax years",
        0.053,
    ),
    (
        "us-hi/statute/235-71.5",
        "the taxable year, plus\n(B) 4 per cent of the excess",
        0.04,
    ),
    ("us-wi/statute/71.07", "required to work at least 2,080 hours per year.", 2080.0),
    ("us/statute/12/1464/d", "not later than 30 days after the earlier of", 30.0),
    (
        "us/statute/42/4822/d/3",
        "prior to the expiration of 5 years after the report",
        5.0,
    ),
    (
        "us-mn/statute/290.06/block-58",
        "under section 290.9725 , 25 percent of the expenses",
        0.25,
    ),
    (
        "us-ca/statute/rtc/18501",
        "of Section 17054, 3 percent for the first dependent credit",
        0.03,
    ),
    (
        "us-ok/statute/68-2352",
        "described by paragraph 3 of this section 77.25% (c)",
        0.7725,
    ),
)


@pytest.mark.parametrize(("citation", "window", "value"), SWEEP_REAL_VALUES)
def test_sweep_real_value_stays_demanded(citation, window, value):
    assert value in _recall_values(window, citation)


# --- A parenthetical amount is not an outline label of a preface ---------------
#
# A page break can leave an amount in parentheses at a line start. The preface
# recognizer read `($9,500)` as an outline label before `for calendar year
# 2016`, so neither gate demanded the amount.
PARENTHETICAL_AMOUNTS = (
    (
        "us-pa/guidance/dli/uc-law/page-37",
        "the amount shall be\n($9,500) for calendar year 2016, nine thousand",
        9500.0,
    ),
    (
        "us-co/regulation/9-ccr-2503-5/3.543",
        "maximum shall be seventy nine dollars\n($79) effective October 1, 2016.",
        79.0,
    ),
)


@pytest.mark.parametrize(("citation", "window", "value"), PARENTHETICAL_AMOUNTS)
def test_parenthetical_amount_is_demanded_by_both_gates(citation, window, value):
    assert value in _recall_values(window, citation)
    assert value in _typed_unnamed(_scalar_module(citation, None), window, citation)
    assert value not in _typed_unnamed(
        _scalar_module(citation, value), window, citation
    )


@pytest.mark.parametrize(
    ("window", "demanded"),
    (
        # Real outline labels still lead a preface; its start year is exempt.
        ("(a) For taxable years beginning after 2025, the credit is $500.", [500.0]),
        ("(iv) For taxable years beginning after 2025, the credit is $500.", [500.0]),
        ("(12) For taxable years beginning after 2025, the credit is $500.", [500.0]),
        # Not a label: a decimal, a thousands separator, a currency sign. The
        # preface then does not start at the line, so its start year stays
        # demanded as well.
        (
            "(1.5) For taxable years beginning after 2025, the credit is $500.",
            [2025.0, 500.0],
        ),
        (
            "(9,500) For taxable years beginning after 2025, the credit is $500.",
            [9500.0, 2025.0, 500.0],
        ),
        (
            "($79) For taxable years beginning after 2025, the credit is $500.",
            [79.0, 2025.0, 500.0],
        ),
        # A history note before `Effective ...` still leads the preface.
        (
            "The credit applies.\n(Added by Stats. 2025, Ch. 231, . (SB 711) "
            "Effective October 1, 2025.)",
            [],
        ),
    ),
)
def test_preface_outline_label_shape(window, demanded):
    assert _recall_values(window, "us/statute/26/1") == demanded


# --- Positional bookkeeping ------------------------------------------------------


def _legacy(text):
    return extract_typed_numeric_inventory_occurrences_from_text(text, profile="legacy")


def test_lost_positions_fail_closed():
    # A step that changes the text length without the position map: the view
    # removes nothing and demands every source number, page number included.
    original = sc.authoritative_numeric_recall_text

    def untracked_suffix(text, **kwargs):
        return original(text, **kwargs) + " untracked"

    with patch.object(sc, "authoritative_numeric_recall_text", untracked_suffix):
        view = sc.complete_source_numeric_recall_occurrences(
            TABLE_PAGE,
            corpus_citation_path=GUIDANCE,
            extract_numeric_occurrences=_legacy,
        )
    assert Counter(occurrence.value for occurrence in view) >= Counter(
        occurrence.value for occurrence in _legacy(TABLE_PAGE)
    )


def test_untracked_in_place_mask_removes_the_number():
    # A mask that blanks characters in place without updating the position map
    # still removes them: positions are compared character by character.
    original = sc._without_us_manual_locators

    def blank_amount(text, **kwargs):
        return original(text, **kwargs).replace("$1,330", "      ")

    with patch.object(sc, "_without_us_manual_locators", blank_amount):
        values = _recall_values(TABLE_PAGE)
    assert 1330.0 not in values
    assert 1663.0 in values


def test_recall_demand_at_a_place_with_another_value_is_added():
    # Where the recall reading finds a different value at the same place, the
    # source number stays and the recall number is added.
    import dataclasses

    def shifted_recall(text):
        occurrences = _legacy(text)
        if text == TABLE_PAGE:
            return occurrences
        return [
            dataclasses.replace(occurrence, value=occurrence.value + 0.5)
            for occurrence in occurrences
        ]

    values = [
        occurrence.value
        for occurrence in sc.complete_source_numeric_recall_occurrences(
            TABLE_PAGE,
            corpus_citation_path=GUIDANCE,
            extract_numeric_occurrences=shifted_recall,
        )
    ]
    assert 1330.0 in values
    assert 1330.5 in values


@pytest.mark.parametrize(
    "window",
    (
        "The rule was published at 62 FR 61344. The fee is $25.",
        "As amended by Pub. L. No. 92-603, the fee is $25.",
    ),
)
def test_us_locator_numbers_leave_the_view(window):
    assert _recall_values(window, "us/regulation/7/273/4") == [25.0]


@pytest.mark.parametrize(
    ("citation", "window", "value"),
    (
        # The source reading drops the title number of a U.S.C./C.F.R.
        # citation that the recall reading demands; the same value elsewhere
        # (an age, a period, an amount) does not cover it.
        (
            "us-co/statute/39/39-37-103",
            "A person under age 18 who is licensed pursuant to 18 U.S.C. sec. 921 "
            "et seq. is exempt.",
            18.0,
        ),
        (
            "us-nc/manual/dhhs/fns/fns-260-able-bodied-adults-without-dependents/page-7",
            "Verify within 7 days as required by 7 C.F.R. § 273.24(k).",
            7.0,
        ),
        (
            "us-nc/statute/105/105-153.5",
            "Subtract 10 dollars; the term has the same meaning as in 10 U.S.C. "
            "§ 101(a)(5).",
            10.0,
        ),
    ),
)
def test_a_same_value_elsewhere_does_not_cover_a_recall_demand(citation, window, value):
    assert _recall_values(window, citation).count(value) == 2


def test_lost_positions_keep_source_and_recall_demands():
    # The fallback demands the union: every source occurrence and every number
    # numeric recall demands. Here the source reading has one 18 (the age) and
    # the recall reading demands two (the age and the U.S.C. title number).
    window = (
        "A person under age 18 who is licensed pursuant to 18 U.S.C. sec. 921 "
        "et seq. is exempt."
    )
    citation = "us-co/statute/39/39-37-103"
    original = sc.authoritative_numeric_recall_text

    def untracked_suffix(text, **kwargs):
        return original(text, **kwargs) + " untracked"

    assert [occurrence.value for occurrence in _legacy(window)].count(18.0) == 1
    with patch.object(sc, "authoritative_numeric_recall_text", untracked_suffix):
        view = sc.complete_source_numeric_recall_occurrences(
            window, corpus_citation_path=citation, extract_numeric_occurrences=_legacy
        )
    assert [occurrence.value for occurrence in view].count(18.0) == 3


def test_four_digit_parenthetical_does_not_lead_a_preface():
    # Mid-line, after a sentence end, so the line-marker mask does not take it.
    window = "The credit is allowed. (1234) for calendar year 2016, the amount is $5."
    assert _recall_values(window, "us/statute/26/1") == [1234.0, 5.0]
