"""Bare outlines must keep branch and condition evidence separate."""

from axiom_encode.harness import source_completeness as sc


SOURCE = """In this subpart, unless the context otherwise requires,—
qualifying benefit means—
a
a main benefit under this Act;
b
a benefit specified in this section.
"""


def test_bare_paragraphs_do_not_join_context_exception_to_first_branch():
    branches = sc.recognize_source_structure(SOURCE)
    assert [branch.path for branch in branches] == [("a",), ("b",)]
    assert branches[0].text.strip() == "a main benefit under this Act;"
    conditions = sc._source_exception_branches(
        SOURCE,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )
    assert len(conditions) == 1
    assert "unless the context otherwise requires" in conditions[0].text
    assert "main benefit" not in conditions[0].text
    assert "a a main" not in sc._collapse_text(conditions[0].text)


def test_ambiguous_bare_sequence_keeps_legacy_source_obligation():
    ambiguous = SOURCE.replace("\nb\n", "\nd\n")
    branches = sc.recognize_source_structure(ambiguous)
    conditions = sc._source_exception_branches(
        ambiguous,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )
    assert conditions
    assert "unless the context otherwise requires" in conditions[0].text
