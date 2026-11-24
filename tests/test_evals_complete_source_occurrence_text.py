"""The complete-mode numeric-occurrence check counts the numeric-recall view (#1801).

In complete-source-unit mode the encode eval refuses a candidate when a source
number has no named scalar ("Source numeric value N appears K time(s) ...").
That check read the raw corpus body, so it demanded again the page numbers,
table labels and locators that complete-source numeric recall
(`analyze_complete_source_unit`) removes from its demand. It now counts the
numbers of `complete_source_numeric_recall_occurrences`, the source view of
numeric recall, so both gates demand the same numbers.

Invariant: in complete mode, the numbers the occurrence check counts are the
numbers numeric recall demands from the whole source unit. A number that the
occurrence check no longer counts is therefore one numeric recall does not
demand either, and every number numeric recall demands stays counted.

Source text in the unit tests is synthetic; the parity test reads the
committed replay fixtures.
"""

from __future__ import annotations

import functools
import json
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
    assert kwargs["normalize_recall_text"] is None


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


def test_normalization_reads_the_masked_recall_text():
    seen = []
    sc.complete_source_numeric_recall_occurrences(
        TABLE_PAGE,
        corpus_citation_path=GUIDANCE,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text,
            profile=_numeric_profile_for_citation_path(GUIDANCE),
        ),
        normalize_recall_text=lambda text: seen.append(text) or text,
    )

    assert seen == [
        sc.authoritative_numeric_recall_text(TABLE_PAGE, corpus_citation_path=GUIDANCE)
    ]
    assert seen[0] != TABLE_PAGE


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
def test_recall_view_is_the_numeric_recall_source_view(fixture):
    """Property over the replay fixtures: same numbers as numeric recall."""

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
    assert view == list(analysis.source_numeric_occurrence_values)
    assert len(view) == analysis.source_numeric_occurrence_count


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
