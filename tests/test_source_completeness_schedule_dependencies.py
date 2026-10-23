"""Unparsed Schedule dependencies preserve base completeness obligations.

The withdrawn Schedule relaxation cannot authenticate operative dependency,
complete instrument identity, and container ownership from a locator alone.
These fixtures record the resulting abstention, including the lost NZ wins.
"""

import functools

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as vp

CITATION = "nz/statute/act/public/2018/0032/section/74"
AUTHENTICATED_ACT_ALIASES = ("Social Security Act 2018",)
# Social Security Act 2018, s 74(1)–(2). Paragraphs (2)(a) and (b)
# inherit the missing rate's locator from their parent computation.
SOURCE = """1
A winter energy payment is payable,—
a
if the person’s qualifying benefit is a main benefit under this Act, in 22 weekly instalments; or
b
if the person’s qualifying benefit is New Zealand superannuation or a veteran’s pension, in 11 fortnightly instalments.
2
The amount of an instalment is ascertained by dividing the appropriate rate per winter period set out in Part 8 of Schedule 4 ,—
a
for recipients of a main benefit under this Act, by 22; or
b
for recipients of New Zealand superannuation or a veteran’s pension, by 11.
"""
REASONS = (
    "nz/statute/act/public/2018/0032/section/74(2) requires the appropriate "
    "rate per winter period set out in Part 8 of Schedule 4 of the Social "
    "Security Act 2018. The missing legal dependency is that Schedule's "
    "appropriate rate per winter period; no executable RuleSpec output for "
    "that rate is supplied in the available context.",
    "nz/statute/act/public/2018/0032/section/74(2)(a) requires dividing the "
    "appropriate rate per winter period by 22. The missing legal dependency "
    "is the appropriate main-benefit rate in Part 8 of Schedule 4 of the "
    "Social Security Act 2018; no executable RuleSpec output for that rate "
    "is supplied in the available context.",
    "nz/statute/act/public/2018/0032/section/74(2)(b) requires dividing the "
    "appropriate rate per winter period by 11. The missing legal dependency "
    "is the appropriate New Zealand superannuation or veteran’s pension "
    "rate in Part 8 of Schedule 4 of the Social Security Act 2018; no executable "
    "RuleSpec output for that rate is supplied in the available context.",
)
AMENDMENT_CITATION = "nz/statute/act/public/2026/0054/section/15"
# Social Security (Accommodation Supplement) Amendment Act 2026, s 15,
# abbreviated to its opening container and replacement clause 6A.
AMENDMENT_SOURCE = (
    "In Schedule 4 , Part 7, replace clauses 4 to 6A with:\n"
    "6A | The rate of accommodation supplement payable to a community spouse "
    "or partner (as defined in Schedule 2) is the rate in clauses 1 to 6AA "
    "that would apply to them if they were single "
    "(and their other circumstances were unchanged)."
)
AMENDMENT_REASON = (
    "nz/statute/act/public/2026/0054/section/15, replacement clause 6A: "
    "computing the community spouse or partner's single-person rate for "
    "accommodation outside replacement clauses 4 to 6AA requires the "
    "accommodation categories and executable rates in Schedule 4, Part 7, "
    "clauses 1 to 3, which are not supplied."
)


@pytest.mark.parametrize("index,path", enumerate((("2",), ("2", "a"), ("2", "b"))))
def test_unparsed_rate_dependency_does_not_inherit_parent_scope(index, path):
    branches = sc.recognize_source_structure(SOURCE, corpus_citation_path=CITATION)
    scope = sc._deferred_source_scope_text(path, source_text=SOURCE, branches=branches)
    assert ("Part 8 of Schedule 4" in scope) == (len(path) == 1)
    assert not sc._reason_dependency_is_source_bound(
        REASONS[index],
        scope,
        corpus_citation_path=CITATION,
        path=path,
        authenticated_same_act_aliases=AUTHENTICATED_ACT_ALIASES,
    )


def test_unparsed_rate_deferrals_retain_all_three_nz_obligations():
    branches = sc.recognize_source_structure(SOURCE, corpus_citation_path=CITATION)
    payload = {
        "module": {
            "deferred_outputs": [
                {
                    "output": "nz:statutes/act/public/2018/0032/section/74/"
                    + "/".join(path)
                    + "#unrounded_instalment_amount",
                    "reason": reason,
                }
                for path, reason in zip((("2",), ("2", "a"), ("2", "b")), REASONS)
            ]
        }
    }
    covered, issues = sc._deferred_coverage(
        payload,
        corpus_citation_path=CITATION,
        source_text=SOURCE,
        branches=branches,
        authenticated_same_act_aliases=AUTHENTICATED_ACT_ALIASES,
    )
    assert not covered
    assert len(issues) == 3


def test_unparsed_amendment_range_retains_the_nz_obligation():
    assert not sc._reason_dependency_is_source_bound(
        AMENDMENT_REASON,
        AMENDMENT_SOURCE,
        corpus_citation_path=AMENDMENT_CITATION,
        path=(),
    )


@pytest.mark.parametrize("negation", ("no", "not", "neither", "without"))
def test_source_cited_reason_cannot_negate_the_schedule_dependency(negation):
    assert not sc._reason_dependency_is_source_bound(
        AMENDMENT_REASON.replace(
            "requires the accommodation", f"requires {negation} accommodation"
        ),
        AMENDMENT_SOURCE,
        corpus_citation_path=AMENDMENT_CITATION,
        path=(),
    )
    assert not sc._reason_dependency_is_source_bound(
        f"{CITATION}(2) requires {negation} appropriate rate per winter period "
        "in Part 8 of Schedule 4; no executable RuleSpec output for that rate "
        "is supplied in the available context.",
        SOURCE,
        corpus_citation_path=CITATION,
        path=("2",),
    )


@pytest.mark.parametrize(
    "replacement",
    (
        "Schedule 5, Part 7, clauses 1 to 3",
        "Schedule 4, Part 8, clauses 1 to 3",
        "Schedule 4, Part 7, clauses 1 to 7",
        "Schedule 4, Part 7, clause 7",
    ),
)
def test_amendment_rejects_unauthenticated_container_or_range(replacement):
    assert not sc._reason_dependency_is_source_bound(
        AMENDMENT_REASON.replace("Schedule 4, Part 7, clauses 1 to 3", replacement),
        AMENDMENT_SOURCE,
        corpus_citation_path=AMENDMENT_CITATION,
        path=(),
    )


@pytest.mark.parametrize(
    "reason",
    (
        "Part 8 of Schedule 4 is available. An unrelated input is unavailable.",
        "It is not true that Part 8 of Schedule 4 is unavailable.",
        "The discussion merely quotes Part 8 of Schedule 4 is unavailable.",
        "Part 8 of Schedule 4 is unavailable, but it is supplied.",
        "The missing legal dependency is not Part 8 of Schedule 4.",
        "The missing legal dependency is something other than Part 8 of Schedule 4.",
        "No missing dependency remains in Part 8 of Schedule 4.",
    ),
)
def test_schedule_dependency_requires_operative_missing_state(reason):
    assert not sc._reason_dependency_is_source_bound(
        reason,
        "The rate is stated in Part 8 of Schedule 4.",
        corpus_citation_path=CITATION,
    )


def test_schedule_dependency_cannot_change_an_explicit_instrument():
    assert not sc._reason_dependency_is_source_bound(
        "Part 8 of Schedule 4 of the Imaginary Benefits Act 2020 is unavailable.",
        "The rate is stated in Part 8 of Schedule 4 of the Social Security Act 2018.",
        corpus_citation_path=CITATION,
    )


def test_relative_clauses_do_not_cross_amendment_containers():
    assert not sc._reason_dependency_is_source_bound(
        "Schedule 4, Part 7, clause 1 is unavailable.",
        "In Schedule 4, Part 7, replace clause 4 with the following. "
        "In Schedule 5, Part 8, replace clause 1 with the following.",
        corpus_citation_path=AMENDMENT_CITATION,
    )


@pytest.mark.parametrize("jurisdiction", ("uk", "us", "de", "fr", "nz"))
@pytest.mark.parametrize(
    "source",
    (
        "In Schedule 4, Part 7, replace clause 4 with: "
        "The rate is the rate in clause 1 of Schedule 5.",
        "In Schedule 4, Part 7, replace clause 4 with the following. "
        "In Schedule 5, Part 8, repeal clause 1.",
        "In Schedule 4, Part 7, replace clause 4 with the following. "
        "In Schedule 5, Part 8, omit clause 1.",
    ),
)
def test_review_wrong_schedule_probes_cannot_authenticate_root_coverage(
    jurisdiction, source
):
    citation = f"{jurisdiction}/statute/2020/1/section/72"
    branches = sc.recognize_source_structure(source, corpus_citation_path=citation)
    covered, issues = sc._deferred_coverage(
        {
            "module": {
                "deferred_outputs": [
                    {
                        "output": f"{jurisdiction}:statutes/2020/1/section/72#rate",
                        "reason": "Schedule 4, Part 7, clause 1 is unavailable.",
                    }
                ]
            }
        },
        source_text=source,
        corpus_citation_path=citation,
        branches=branches,
    )
    assert not covered
    assert issues


@pytest.mark.parametrize("jurisdiction", ("uk", "us", "de", "fr", "nz"))
@pytest.mark.parametrize(
    "action", ("replace", "repeal", "omit", "insert", "substitute")
)
@pytest.mark.parametrize("action_first", (False, True))
def test_every_amendment_action_terminates_an_inherited_container(
    jurisdiction, action, action_first
):
    boundary = (
        f"{action} clause 1 of Schedule 5, Part 8."
        if action_first
        else f"In Schedule 5, Part 8, {action} clause 1."
    )
    assert not sc._reason_dependency_is_source_bound(
        "Schedule 4, Part 7, clause 1 is unavailable.",
        "In Schedule 4, Part 7, replace clause 4 with the following. " + boundary,
        corpus_citation_path=f"{jurisdiction}/statute/2020/1/section/72",
    )


@pytest.mark.parametrize(
    "qualification",
    ("Schedule 5", "Part 8 of Schedule 5", "Schedule 5, Part 8"),
)
@pytest.mark.parametrize("connector", ("of", "in", "under"))
def test_postfix_clause_qualification_abstains_without_dependency_authentication(
    qualification, connector
):
    source = f"The rate is in clauses 1 to 3 {connector} {qualification}."
    reason = f"clause 2 {connector} {qualification} is unavailable."
    assert not sc._reason_dependency_is_source_bound(
        reason, source, corpus_citation_path=AMENDMENT_CITATION
    )
    assert not sc._reason_dependency_is_source_bound(
        reason.replace("clause 2", "clause 4"),
        source,
        corpus_citation_path=AMENDMENT_CITATION,
    )


@pytest.mark.parametrize("jurisdiction", ("uk", "us", "de", "fr", "nz"))
@pytest.mark.parametrize(
    "new_container",
    (
        "Omit from Schedule 5, Part 8 clause 4. The rate is in clause 1.",
        "Omit clause 4 in Schedule 5, Part 8. The rate is in clause 1.",
        "Insert after clause 4 of Schedule 5, Part 8: The rate is in clause 1.",
        "In Schedule 5, Part 8, after clause 4, insert: The rate is in clause 1.",
        "In Schedule 5, Part 8 of the Other Benefits Act 2020, repeal clause 1.",
        "After clause 6 of Schedule 5, Part 8, insert clause 1.",
        "Before clause 6 of Schedule 5, Part 8, insert clause 1.",
        "In Schedule 5, Part 8 of the Other Benefits Instrument 2020, repeal clause 1.",
        "In Schedule 5, Part 8 (rates), repeal clause 1.",
        "In Schedule 5, Part 8, clause 2 is replaced by clause 1.",
        "In Schedule 5, Part 8, the rate is specified in clause 1.",
        "For Schedule 5, Part 8, omit clause 1.",
        "The rate in Schedule 5, Part 8 is specified in clause 1.",
        "The rate is in clause 1, Schedule 5.",
        "The rate is in clause 1 of (Schedule 5).",
        "The rate is in clause 1 (of Schedule 5).",
        "The rate is in clause 1 [of Schedule 5].",
        "The rate is in clause 1 set out in Schedule 5.",
        "The rate is in clause 1 specified in Schedule 5.",
        "The rate is in clause 1 as defined in Schedule 5.",
        "The rate is in clause 1 from Schedule 5.",
        "The rate is in clause 1\nset out in Schedule 5.",
        "The rate is in clause 1 (as defined in Schedule 5).",
        "The rate is in clauses 1 and 2 of Schedule 5.",
        "The rate is in clauses 1, 2 and 3 of Schedule 5.",
        "The rate is in clauses 1 or 2 of Schedule 5.",
        "The rate is in clauses 1 to 3 and 4 of Schedule 5.",
        "The rate is in clause 1 (Schedule 5).",
        "The rate is in clause 1 of the Fifth Schedule.",
        "The rate is in clause 1 in Schedule 5.",
        "The rate is in clause 1 under Schedule 5.",
    ),
)
def test_qualified_amendment_locations_never_extend_a_previous_container(
    jurisdiction, new_container
):
    assert not sc._reason_dependency_is_source_bound(
        "Schedule 4, Part 7, clause 1 is unavailable.",
        "In Schedule 4, Part 7, replace clause 4 with the following. " + new_container,
        corpus_citation_path=f"{jurisdiction}/statute/2020/1/section/72",
    )


@pytest.mark.parametrize("jurisdiction", ("uk", "us", "de", "fr", "nz"))
@pytest.mark.parametrize(
    "instrument",
    (
        "Imaginary Benefits Act 2020",
        "Imaginary Benefits Act",
        "Imaginary Benefits Code",
        "Imaginary Benefits Regulations 2020",
        "Imaginary Benefits Instrument 2020",
    ),
)
def test_review_invented_instrument_probe_cannot_authenticate_root_coverage(
    jurisdiction, instrument
):
    citation = f"{jurisdiction}/statute/2020/1/section/72"
    source = "The rate is stated in Part 8 of Schedule 4."
    branches = sc.recognize_source_structure(source, corpus_citation_path=citation)
    covered, issues = sc._deferred_coverage(
        {
            "module": {
                "deferred_outputs": [
                    {
                        "output": f"{jurisdiction}:statutes/2020/1/section/72#rate",
                        "reason": f"Part 8 of Schedule 4 of the {instrument} "
                        "is unavailable.",
                    }
                ]
            }
        },
        source_text=source,
        corpus_citation_path=citation,
        branches=branches,
    )
    assert not covered
    assert issues


@pytest.mark.parametrize("jurisdiction", ("uk", "us", "de", "fr", "nz"))
def test_explicit_instrument_alone_cannot_authenticate_an_operative_dependency(
    jurisdiction,
):
    citation = f"{jurisdiction}/statute/2020/1/section/72"
    instrument = "Actual Benefits Act 2020"
    reason = f"Part 8 of Schedule 4 of the {instrument} is unavailable."
    unqualified_source = "The rate is stated in Part 8 of Schedule 4."
    assert not sc._reason_dependency_is_source_bound(
        reason, unqualified_source, corpus_citation_path=citation
    )
    assert not sc._reason_dependency_is_source_bound(
        reason,
        unqualified_source,
        corpus_citation_path=citation,
        authenticated_same_act_aliases=("Other Benefits Act 2020",),
    )
    assert not sc._reason_dependency_is_source_bound(
        reason,
        unqualified_source,
        corpus_citation_path=citation,
        authenticated_same_act_aliases=(instrument,),
    )
    assert not sc._reason_dependency_is_source_bound(
        reason,
        f"The rate is stated in Part 8 of Schedule 4 of the {instrument}.",
        corpus_citation_path=citation,
    )
    assert not sc._reason_dependency_is_source_bound(
        reason,
        "The rate is stated in Part 8 of Schedule 4 of the Other Benefits Act 2020.",
        corpus_citation_path=citation,
        authenticated_same_act_aliases=(instrument,),
    )


def test_nz_short_fixture_title_is_not_authenticated_by_its_citation():
    branches = sc.recognize_source_structure(SOURCE, corpus_citation_path=CITATION)
    scope = sc._deferred_source_scope_text(
        ("2",), source_text=SOURCE, branches=branches
    )
    assert not sc._reason_dependency_is_source_bound(
        REASONS[0], scope, corpus_citation_path=CITATION, path=("2",)
    )


@pytest.mark.parametrize("jurisdiction", ("uk", "us", "de", "fr", "nz"))
def test_relative_clause_cannot_borrow_another_instruments_identical_container(
    jurisdiction,
):
    assert not sc._reason_dependency_is_source_bound(
        "Schedule 4, Part 7, clause 1 of the Actual Benefits Act 2020 is unavailable.",
        "In Schedule 4, Part 7, replace clause 4 with: "
        "The rate is in clause 1 of Schedule 4, Part 7 of the Other Benefits Act 2020.",
        corpus_citation_path=f"{jurisdiction}/statute/2020/1/section/72",
        authenticated_same_act_aliases=("Actual Benefits Act 2020",),
    )


@pytest.mark.parametrize("jurisdiction", ("uk", "us", "de", "fr", "nz"))
def test_an_unrelated_source_title_cannot_qualify_a_schedule_dependency(jurisdiction):
    assert not sc._reason_dependency_is_source_bound(
        "Part 8 of Schedule 4 of the Imaginary Benefits Act 2020 is unavailable.",
        "The rate is stated in Part 8 of Schedule 4. "
        "The Imaginary Benefits Act 2020 has no effect on that rate.",
        corpus_citation_path=f"{jurisdiction}/statute/2020/1/section/72",
    )


@pytest.mark.parametrize("jurisdiction", ("uk", "us", "de", "fr", "nz"))
@pytest.mark.parametrize("continuation", ("Amendment Act 2025", "Act 2025"))
def test_source_instrument_title_cannot_authenticate_a_truncated_prefix(
    jurisdiction, continuation
):
    assert not sc._reason_dependency_is_source_bound(
        "Part 8 of Schedule 4 of the Actual Benefits Act 2020 is unavailable.",
        "The rate is stated in Part 8 of Schedule 4 of the Actual Benefits Act 2020 "
        + continuation
        + ".",
        corpus_citation_path=f"{jurisdiction}/statute/2020/1/section/72",
    )


@pytest.mark.parametrize("jurisdiction", ("uk", "us", "de", "fr", "nz"))
@pytest.mark.parametrize(
    "qualification",
    (
        "(Other Benefits Act 2020)",
        "in the Other Benefits Act 2020",
        "under the Other Benefits Act 2020",
        ", of the Other Benefits Act 2020",
        "(under the Other Benefits Act 2020)",
    ),
)
def test_source_explicit_foreign_instrument_cannot_be_overridden_by_current_alias(
    jurisdiction, qualification
):
    assert not sc._reason_dependency_is_source_bound(
        "Part 8 of Schedule 4 of the Actual Benefits Act 2020 is unavailable.",
        f"The rate is stated in Part 8 of Schedule 4 {qualification}.",
        corpus_citation_path=f"{jurisdiction}/statute/2020/1/section/72",
        authenticated_same_act_aliases=("Actual Benefits Act 2020",),
    )


def test_child_scope_never_inherits_a_siblings_dependency():
    source = SOURCE.replace("by 22; or", "by the rate in Part 9 of Schedule 5; or")
    branches = sc.recognize_source_structure(source, corpus_citation_path=CITATION)
    scope = sc._deferred_source_scope_text(
        ("2", "b"), source_text=source, branches=branches
    )
    assert "Part 8 of Schedule 4" not in scope
    assert "Part 9 of Schedule 5" not in scope
    assert not sc._reason_dependency_is_source_bound(
        "Part 9 of Schedule 5 is unavailable.",
        scope,
        corpus_citation_path=CITATION,
    )


@pytest.mark.parametrize(
    "citation",
    (
        "nz/statute/act/public/2018/0032/schedule/4/part/8/clause/1",
        "uk/statute/2020/1/schedule/4/part/8/clause/1",
    ),
)
def test_current_schedule_clause_cannot_supply_its_own_external_dependency(citation):
    assert not sc._reason_dependency_is_source_bound(
        "Schedule 4, Part 8, clause 1 is unavailable.",
        "The rate is in Schedule 4, Part 8, clause 1.",
        corpus_citation_path=citation,
    )


@given(
    schedule=st.integers(min_value=1, max_value=99),
    part=st.integers(min_value=1, max_value=99),
    last=st.integers(min_value=2, max_value=99),
    jurisdiction=st.sampled_from(("nz", "uk", "za", "us")),
)
@settings(max_examples=35, deadline=None)
def test_unparsed_schedule_range_preserves_base_obligations(
    schedule, part, last, jurisdiction
):
    source = f"The rate is in Schedule {schedule}, Part {part}, clauses 1 to {last}."
    citation = f"{jurisdiction}/statute/2020/1/section/72"
    reason = f"Part {part} of Schedule {schedule}, clause {last} is unavailable."
    assert not sc._reason_dependency_is_source_bound(
        reason, source, corpus_citation_path=citation
    )
    for other in (
        reason.replace(f"Schedule {schedule}", f"Schedule {schedule + 1}"),
        reason.replace(f"Part {part}", f"Part {part + 1}"),
        reason.replace(f"clause {last}", f"clause {last + 1}"),
    ):
        assert not sc._reason_dependency_is_source_bound(
            other, source, corpus_citation_path=citation
        )


@pytest.mark.parametrize("jurisdiction", ("uk", "us", "de", "fr", "nz"))
@pytest.mark.parametrize(
    "statement",
    (
        "The rate in Part 8 of Schedule 4 does not apply.",
        "The rate is not stated in Part 8 of Schedule 4.",
        "Part 8 of Schedule 4 remains unaffected.",
    ),
)
def test_review_nonoperative_schedule_cannot_defer_a_root_computation(
    jurisdiction, statement
):
    _assert_root_deferral_rejected(
        jurisdiction,
        "The amount is 100. " + statement,
        "Part 8 of Schedule 4 is unavailable.",
    )


@pytest.mark.parametrize("jurisdiction", ("uk", "us", "de", "fr", "nz"))
@pytest.mark.parametrize("wrap", (" ", "\n", "\r\n", "\n\n", "\t"))
@pytest.mark.parametrize("continuation", ("Amendment Act 2025", "Act 2025"))
def test_review_title_reflow_never_authenticates_a_truncated_instrument(
    jurisdiction, wrap, continuation
):
    _assert_root_deferral_rejected(
        jurisdiction,
        "The rate is stated in Part 8 of Schedule 4 of the Actual Benefits "
        "Act 2020" + wrap + continuation + ".",
        "Part 8 of Schedule 4 of the Actual Benefits Act 2020 is unavailable.",
    )


@pytest.mark.parametrize("jurisdiction", ("uk", "us", "de", "fr", "nz"))
@pytest.mark.parametrize(
    "continuation",
    (
        "The rate is in clause 1 of Schedules 5 and 6.",
        "The rate is in clause 1 of\nSchedules 5 and 6.",
        "The rate is in clause 1 of the Fifth and Sixth Schedules.",
        "The rate is in clause 1 of the Fifth and\nSixth Schedules.",
        "The rate is in clause 1 of\nSchedule 5.",
        "The rate is in clause 1 of the Fifth Schedule.",
    ),
)
def test_review_plural_or_unparsed_mention_cannot_borrow_an_earlier_schedule(
    jurisdiction, continuation
):
    _assert_root_deferral_rejected(
        jurisdiction,
        "In Schedule 4, Part 7, replace clause 4 with the following. " + continuation,
        "Schedule 4, Part 7, clause 1 is unavailable.",
    )


def _assert_root_deferral_rejected(jurisdiction, source, reason):
    citation = f"{jurisdiction}/statute/2020/1/section/72"
    covered, issues = sc._deferred_coverage(
        {
            "module": {
                "deferred_outputs": [
                    {
                        "output": f"{jurisdiction}:statutes/2020/1/section/72#rate",
                        "reason": reason,
                    }
                ]
            }
        },
        corpus_citation_path=citation,
        source_text=source,
        branches=sc.recognize_source_structure(source, corpus_citation_path=citation),
    )
    assert not covered
    assert len(issues) == 1


@pytest.mark.parametrize("jurisdiction", ("uk", "us", "de", "fr", "nz"))
@pytest.mark.parametrize(
    "source,reason",
    (
        (
            "The amount is income multiplied by 100. "
            "The rate in Part 8 of Schedule 4 does not apply.",
            "Part 8 of Schedule 4 is unavailable.",
        ),
        (
            "The amount is income multiplied by 100. The rate is stated in "
            "Part 8 of Schedule 4 of the Actual Benefits Act 2020\n"
            "Amendment Act 2025.",
            "Part 8 of Schedule 4 of the Actual Benefits Act 2020 is unavailable.",
        ),
        (
            "In Schedule 4, Part 7, replace clause 4 with the following. "
            "The amount is income multiplied by 100 using the rate in "
            "clause 1 of Schedules 5 and 6.",
            "Schedule 4, Part 7, clause 1 is unavailable.",
        ),
    ),
)
def test_review_parameter_only_artifact_preserves_the_missing_formula(
    jurisdiction, source, reason
):
    citation = f"{jurisdiction}/statute/2020/1/section/72"
    payload = {
        "format": "rulespec/v1",
        "module": {
            "source_verification": {"corpus_citation_path": citation},
            "deferred_outputs": [
                {
                    "output": f"{jurisdiction}:statutes/2020/1/section/72#amount",
                    "reason": reason,
                }
            ],
        },
        "rules": [
            {
                "name": "multiplier",
                "kind": "parameter",
                "dtype": "Decimal",
                "period": "Year",
                "source": citation,
                "versions": [{"effective_from": "2020-01-01", "formula": 100}],
            }
        ],
    }
    analysis = sc.analyze_complete_source_unit(
        yaml.safe_dump(payload),
        source,
        corpus_citation_path=citation,
        test_cases=[
            {
                "name": "factor",
                "period": "2020",
                "input": {},
                "output": {"multiplier": 100},
            }
        ],
        extract_numeric_occurrences=functools.partial(
            vp.extract_typed_numeric_inventory_occurrences_from_text,
            profile="en-US",
        ),
        extract_named_scalars=vp.extract_named_scalar_occurrences,
        numeric_value_is_grounded=vp.numeric_value_is_grounded,
    )
    assert any(
        issue.startswith("[complete-source-unit:deferral]") for issue in analysis.issues
    )
    assert any(
        issue.startswith("[complete-source-unit:formula-output]")
        for issue in analysis.issues
    )
