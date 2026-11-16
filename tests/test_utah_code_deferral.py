import pytest
import yaml

from axiom_encode.harness.source_completeness import (
    _deferred_coverage,
    analyze_complete_source_unit,
    recognize_source_structure,
)
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)

SECTION = "us-ut/statute/59-10-1043"
SOURCE = (
    "(2) Except as provided in Section 59-10-1002.2, a claimant who receives "
    "military retirement pay may claim a nonrefundable tax credit."
)
OUTPUT = "us-ut:statutes/59-10-1043/2#apportioned_military_retirement_credit"
REASON = "Cannot be computed until us-ut/statute/59-10-1002.2 is encoded."


def _coverage(reason: str, source: str = SOURCE, output: str = OUTPUT):
    source_text = "(1) The first subsection.\n" + source + "\n(3) The third subsection."
    clauses = []
    covered, issues = _deferred_coverage(
        {"module": {"deferred_outputs": [{"output": output, "reason": reason}]}},
        corpus_citation_path=SECTION,
        source_text=source_text,
        branches=recognize_source_structure(source_text, corpus_citation_path=SECTION),
        deferred_clauses=clauses,
    )
    return covered, issues, clauses


@pytest.mark.parametrize(
    "citation", ("us-ut/statute/59-10-1002.2", "Utah Code § 59-10-1002.2")
)
def test_utah_deferral_matches_the_complete_linked_section(citation: str):
    covered, issues, clauses = _coverage(
        f"Cannot be computed until {citation} is encoded."
    )

    assert covered == set()
    assert issues == []
    assert [(clause.path, clause.text) for clause in clauses] == [
        (("2",), "(2) Except as provided in Section 59-10-1002.2,")
    ]


def test_utah_deferral_binds_only_to_the_child_that_states_the_exception():
    source = "(2)\n" + SOURCE.replace("(2)", "(2)(a)", 1)
    child_output = OUTPUT.replace("/2#", "/2/a#")

    covered, issues, clauses = _coverage(REASON, source, child_output)
    assert covered == set()
    assert issues == []
    assert [clause.path for clause in clauses] == [("2", "a")]
    assert _coverage(REASON, source)[2] == []


def test_utah_exception_is_not_deferred_without_a_clause_collector():
    source_text = "(1) First subsection.\n" + SOURCE + "\n(3) Third subsection."
    covered, issues = _deferred_coverage(
        {"module": {"deferred_outputs": [{"output": OUTPUT, "reason": REASON}]}},
        corpus_citation_path=SECTION,
        source_text=source_text,
        branches=recognize_source_structure(source_text, corpus_citation_path=SECTION),
    )

    assert covered == set()
    assert len(issues) == 1
    assert "does not name an exact missing dependency" in issues[0]


@pytest.mark.parametrize(
    "reason,source",
    [
        ("Cannot be computed until us-ut/statute/59-10-1002 is encoded.", SOURCE),
        ("Cannot be computed until us-ut/statute/59-10-1002.20 is encoded.", SOURCE),
        ("Cannot be computed until us-id/statute/59-10-1002.2 is encoded.", SOURCE),
        (
            "Cannot be computed until the apportionment requirements are encoded.",
            SOURCE,
        ),
        (
            "Cannot be computed until us-ut/statute/59-10-1043 is encoded.",
            SOURCE.replace("59-10-1002.2", "59-10-1043"),
        ),
        (
            "Cannot be computed until us-ut/statute/59-10-1002.2 is encoded.",
            SOURCE.replace("Except as provided in", "For background, see"),
        ),
        (
            "Cannot be computed until us-ut/statute/59-10-1002.2 is encoded. "
            "That provision is already implemented.",
            SOURCE,
        ),
    ],
)
def test_utah_deferral_rejects_missing_or_inexact_source_link(reason: str, source: str):
    covered, issues, clauses = _coverage(reason, source)

    assert covered == set()
    assert len(issues) == 1
    assert "does not name an exact missing dependency" in issues[0]
    assert clauses == []


def test_utah_deferral_preserves_independent_product_obligations():
    source = (
        "(1) Definitions.\n" + SOURCE.removesuffix(".") + " equal to the product of:\n"
        "(2)(a) the percentage listed in Subsection 59-10-104(2); and\n"
        "(2)(b) the amount of military retirement pay that is included in adjusted gross income.\n"
        "(3) A claimant may not carry forward excess credit."
    )
    candidate = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {
                "source_verification": {"corpus_citation_path": SECTION},
                "deferred_outputs": [{"output": OUTPUT, "reason": REASON}],
            },
            "rules": [],
        }
    )
    result = analyze_complete_source_unit(
        candidate,
        source,
        corpus_citation_path=SECTION,
        test_cases=[],
        extract_numeric_occurrences=extract_typed_numeric_inventory_occurrences_from_text,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )

    for branch in ("(2)", "(2)(a)", "(2)(b)"):
        assert any(f"{SECTION}{branch} [provision" in issue for issue in result.issues)
    assert not any("complete-source-unit:deferral" in issue for issue in result.issues)
