"""Schedule dependencies bind to their legal containers and parent chapeaux."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc

CITATION = "nz/statute/act/public/2018/0032/section/74"
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
def test_rate_dependency_binds_parent_and_inherited_child_scopes(index, path):
    branches = sc.recognize_source_structure(SOURCE, corpus_citation_path=CITATION)
    scope = sc._deferred_source_scope_text(path, source_text=SOURCE, branches=branches)
    assert "Part 8 of Schedule 4" in scope
    assert sc._reason_dependency_is_source_bound(
        REASONS[index], scope, corpus_citation_path=CITATION, path=path
    )


def test_precise_rate_deferrals_need_no_invented_output_identifiers():
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
        payload, corpus_citation_path=CITATION, source_text=SOURCE, branches=branches
    )
    assert not issues
    assert {("2",), ("2", "a"), ("2", "b")} <= covered


def test_amendment_relative_clause_range_binds_its_schedule_and_part():
    assert "Schedule 4, Part 7, clauses 1 to 3" in [
        m.group() for m in sc._PRECISE_DEFERRAL_DEPENDENCY.finditer(AMENDMENT_REASON)
    ]
    assert sc._reason_dependency_is_source_bound(
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


def test_child_scope_never_inherits_a_siblings_dependency():
    source = SOURCE.replace("by 22; or", "by the rate in Part 9 of Schedule 5; or")
    branches = sc.recognize_source_structure(source, corpus_citation_path=CITATION)
    scope = sc._deferred_source_scope_text(
        ("2", "b"), source_text=source, branches=branches
    )
    assert "Part 8 of Schedule 4" in scope
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
def test_schedule_range_binding_preserves_container_identity_and_bounds(
    schedule, part, last, jurisdiction
):
    source = f"The rate is in Schedule {schedule}, Part {part}, clauses 1 to {last}."
    citation = f"{jurisdiction}/statute/2020/1/section/72"
    reason = f"Part {part} of Schedule {schedule}, clause {last} is unavailable."
    assert sc._reason_dependency_is_source_bound(
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
