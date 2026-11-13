"""Missing-boundary feedback names the source words behind each listed value."""

import functools
from types import SimpleNamespace

from axiom_encode import cli
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as vp

CITATION = "us-az/manual/des/faa5/na-categorical-eligibility/block-5"
# The Arizona FAA5 block-5 wording whose 200% limit retries saw only as `=2`.
SOURCE = (
    "AZTECS compares the income of an NA basic categorically eligible budgetary "
    "unit against the Net Income Standard.\n\n"
    "AZTECS determines when the gross income of an NA expanded categorically "
    "eligible (ECE) budgetary unit exceeds 200% of the current federal poverty "
    "level (FPL). When the budgetary unit's gross income exceeds 200% of the "
    "current FPL, the budgetary unit is not categorically eligible.\n"
)


def _obligations(source=SOURCE, profile="legacy"):
    branches = sc.recognize_source_structure(source)
    formulas = sc._source_formula_branches(
        source, branches=branches, active_branches=branches, deferred_paths=set()
    )
    extract = functools.partial(
        vp.extract_typed_numeric_inventory_occurrences_from_text, profile=profile
    )
    views: dict[int, str] = {}
    obligations = sc._source_boundary_obligations(
        branches,
        narrative_formula_branches=formulas,
        extract_numeric_occurrences=extract,
        source_text=source,
        recall_views=views,
    )
    return views, list(obligations)


def _detail(views, obligations, principal_rule_paths=None):
    return sc._missing_boundary_repair_detail(
        obligations,
        recall_views=views,
        principal_rule_paths=principal_rule_paths or {},
        corpus_citation_path=CITATION,
    )


def test_percentage_boundary_names_source_words_and_ratio():
    views, obligations = _obligations()
    assert [(b.label, o.value) for b, o in obligations] == [
        ("source unit formula clause 2", 2.0),
        ("source unit formula clause 3", 2.0),
    ]

    detail = _detail(views, obligations)

    assert detail.startswith(sc.MISSING_BOUNDARY_DETAIL_MARKER)
    assert (
        f"source unit formula clause 3=2 at {CITATION} [source unit]; "
        "source states `200` percent, listed as the ratio 2, in "
        "`When the budgetary unit's gross income exceeds ⟨200⟩% of the "
        "current FPL" in detail
    )
    assert "eligible (ECE) budgetary unit exceeds ⟨200⟩%" in detail
    assert "no principal derived output cites this source branch yet" in detail


def test_covering_principal_output_is_named():
    views, obligations = _obligations()
    branch = obligations[1][0]

    detail = _detail(
        views,
        obligations[1:],
        {"ece_blocked_by_gross_income_limit": {branch.path}},
    )

    assert "covering principal output(s) `ece_blocked_by_gross_income_limit`" in (
        detail
    )
    assert "no principal derived output cites" not in detail


def test_unverified_coordinate_quotes_no_source_text():
    views, obligations = _obligations()
    branch, occurrence = obligations[0]
    shifted = SimpleNamespace(
        value=occurrence.value,
        start=occurrence.start + 1,
        end=occurrence.end + 1,
        raw=occurrence.raw,
    )

    view = views[id(occurrence)]
    assert sc._boundary_source_context(shifted, view) is None
    # An occurrence without a recorded view quotes nothing either.
    assert sc._boundary_source_context(occurrence, None) is None
    views[id(shifted)] = view
    detail = _detail(views, [(branch, shifted)])
    assert "source states" not in detail
    assert f"{branch.label}=2 at {CITATION}" in detail


def test_detail_is_bounded():
    views, obligations = _obligations()
    many = obligations * 8

    detail = _detail(views, many)

    assert detail.count(" at us-az/") == sc._MISSING_BOUNDARY_DETAIL_LIMIT
    assert f"{len(many) - sc._MISSING_BOUNDARY_DETAIL_LIMIT} more boundary" in detail


def test_residual_count_ignores_commas_in_detail():
    views, obligations = _obligations()
    issue = (
        "[complete-source-unit:tests] Companion tests do not exercise every "
        "source-stated boundary input; missing: source unit formula clause 2=2, "
        "source unit formula clause 3=2." + _detail(views, obligations)
    )
    assert "," in issue.split(sc.MISSING_BOUNDARY_DETAIL_MARKER, 1)[1]

    assert cli._validation_residual_obligation_count([issue]) == 2


def test_unnormalized_percentage_gets_no_ratio_note():
    views, obligations = _obligations(
        "Die Leistung beträgt höchstens 200% des Einkommens.\n", profile="de-DE"
    )
    assert [o.value for _, o in obligations] == [200.0]

    detail = _detail(views, obligations)

    assert "listed as the ratio" not in detail
    assert (
        "source states `200` in `Die Leistung beträgt höchstens \u27e8200\u27e9%"
        in (detail)
    )


def test_window_uses_the_view_that_produced_the_offsets():
    # Full-branch cleanup removes the footnote markers and would shift the
    # dollar `50` onto the first-line offsets of the percentage `50`.
    source = (
        "For working entrants1, unemployed entrants1, student entrants1, "
        "disabled entrants1, elderly entrants1, homeless entrants1, or married "
        "entrants1, the payment is at most 50% of 50 dollars\n"
        "1 Cuban and Haitian entrants as defined in section 501(e).\n"
    )
    views, obligations = _obligations(source)
    assert [o.value for _, o in obligations] == [0.5]

    detail = _detail(views, obligations)

    assert "the payment is at most \u27e850\u27e9% of 50 dollars" in detail
    assert "50% of \u27e850\u27e9" not in detail
    assert "listed as the ratio 0.5" in detail


def test_written_percentages_get_the_ratio_note_once():
    for wording in ("200 percent", "200 per cent", "two hundred percent"):
        views, obligations = _obligations(
            f"The limit is at most {wording} of income.\n"
        )
        assert [o.value for _, o in obligations] == [2.0]

        detail = _detail(views, obligations)

        assert f"source states `{wording}`, listed as the ratio 2, in " in detail
        assert f"`{wording}` percent" not in detail
