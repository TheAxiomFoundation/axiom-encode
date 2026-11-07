"""Detached worksheet structure is opt-in, never execution authority."""

import ast
from dataclasses import asdict
from pathlib import Path

import pytest

from axiom_encode.form_numeric_structure import form_numeric_locator_spans
from axiom_encode.harness import source_completeness as s
from axiom_encode.printed_source_rows import (
    mask_printed_heading_dash,
    recognize_printed_rows,
)
from tests.test_actual_floor_evidence import pair, world  # noqa: F401
from tests.test_printed_signed_format_evidence import signed  # noqa: F401

SOURCE = (Path(__file__).parent / "fixtures/detached_schedule2_source.txt").read_text()
CITATION = "ca/policy/cra/t1-2025/federal-spouse-transfer-amounts"


@pytest.mark.parametrize("flag", [False, None, 0, 1, "true"])
def test_detached_is_not_enabled_by_default_or_truthiness(flag):
    assert recognize_printed_rows(SOURCE) is None
    assert recognize_printed_rows(SOURCE, allow_detached=flag) is None
    assert s._unconflicted_printed_rows(SOURCE, ()) is None
    assert mask_printed_heading_dash(SOURCE) == SOURCE


def test_exact_detached_roles_and_optional_heading():
    structure = recognize_printed_rows(SOURCE, allow_detached=True)
    assert structure is not None
    assert structure.heading_dash is None
    assert structure.typography_spans == ()
    assert structure.arithmetic_referral_spans == ()
    assert len(structure.numeric_locator_spans) == 32
    assert structure.numeric_locator_spans == form_numeric_locator_spans(SOURCE)
    assert [SOURCE[a:b] for a, b in structure.additions] == [
        "Add lines 1 to 5.",
        "Add lines 8 to 10.",
    ]
    for start, end in structure.ownership_spans:
        assert 0 <= start < end <= len(SOURCE)
        assert SOURCE[start:end].strip() == SOURCE[start:end]
    # The cap cell cannot acquire its remote caption by consuming rows 4..13.
    cap = next(
        (a, b)
        for a, b in structure.ownership_spans
        if SOURCE[a:b].startswith("(maximum")
    )
    caption = next(
        (a, b)
        for a, b in structure.ownership_spans
        if SOURCE[a:b].startswith("Pension income")
    )
    assert cap[1] < SOURCE.index("Disability amount:") < caption[0]
    branches = s.recognize_source_structure(SOURCE, corpus_citation_path=CITATION)
    assert (
        s._unconflicted_printed_rows(
            SOURCE, branches, corpus_citation_path=CITATION, allow_detached=True
        )
        == structure
    )


@pytest.mark.parametrize(
    "token",
    [
        "$2,000",
        "$5,000",
        "90 days",
        "$57,375",
        "14.5%",
        "1960",
        "under 18",
        "for 2024",
        "in 2025",
        '"0"',
    ],
)
def test_substantive_numbers_are_not_locator_spans(token):
    locators = form_numeric_locator_spans(SOURCE)
    start = SOURCE.index(token)
    end = start + len(token)
    assert not any(a < end and start < b for a, b in locators)


@pytest.mark.parametrize(
    "before,after",
    [
        ("$2,000", "$3,000"),
        ("$5,000", "$6,000"),
        ("90 days", "89 days"),
        ("$57,375", "$57,376"),
        ("14.5%", "15.5%"),
        ("1960", "1961"),
        ("under 18", "under 19"),
        ("for 2024", "for 2023"),
        ("in 2025", "in 2026"),
        ('"0"', '"1"'),
        ("35200", "35300"),
        ("Add lines 1 to 5.", "Add lines 1 to 4."),
        ("Line 7 minus line 11", "Line 11 minus line 7"),
        ("(1)\n\n35700", "(2)\n\n35700"),
        ("cannot transfer to you", "can transfer to you"),
        ("maximum $2,000", "maximum $5,000"),
        ("\n\n3\n\n", "\n\n4\n\n"),
        ("Pension income amount:", "Pension income amount, if separately approved:"),
    ],
)
def test_changed_source_cannot_borrow_structural_credit(before, after):
    changed = SOURCE.replace(before, after, 1)
    assert changed != SOURCE
    assert recognize_printed_rows(changed, allow_detached=True) is None
    assert form_numeric_locator_spans(changed) == ()


@pytest.mark.parametrize(
    "marker",
    [
        "CRA Schedule 2",
        "Complete this schedule",
        "If you were separated",
        "If your spouse or common-law partner is filing",
        "If your spouse or common-law partner is not filing",
        "Attach a copy",
        "Age amount",
        "Canada caregiver amount",
        "(maximum $2,000)",
        "Disability amount:",
        "Tuition amount:",
        "=\n\n6\n\n",
        "Spouse's or common-law partner's adjusted taxable income:",
        "Line 6 minus line 12",
        "Pension income amount:",
        "Add lines 1 to 5.",
        "Spouse's or common-law partner's taxable income:",
        "Amount from line 30000",
        "Amount from line 96",
        "Amount from line 32300",
        "Add lines 8 to 10.",
        "+\n=\n\n10",
        "36100 =",
        "(1) If this is a new claim",
        "(2) The maximum tuition",
        "See the privacy notice",
        "5000-S2 E (25)",
    ],
)
def test_no_unowned_inter_record_qualification(marker):
    start = SOURCE.index(marker)
    changed = SOURCE[:start] + "Only if separately approved.\n" + SOURCE[start:]
    assert recognize_printed_rows(changed, allow_detached=True) is None
    assert form_numeric_locator_spans(changed) == ()


def test_duplicate_or_trailing_source_is_not_partially_recognized():
    for changed in (
        SOURCE + "\nAdditional approval is required.",
        SOURCE + SOURCE,
        SOURCE.replace("Add lines 1 to 5.", "Add lines 1 to 5.\nAdd lines 1 to 5."),
    ):
        assert recognize_printed_rows(changed, allow_detached=True) is None


@pytest.mark.parametrize("step", [2, 3, 4])
def test_step_recognition_is_identical_with_and_without_optin(step):
    from tests.test_printed_source_rows import P

    text = (P / f"step-{step}.txt").read_text()
    original = recognize_printed_rows(text)
    assert original is not None
    assert original.heading_dash is not None
    assert original.numeric_locator_spans == ()
    assert asdict(original) == asdict(recognize_printed_rows(text, allow_detached=True))


def test_only_reviewed_structural_consumers_opt_in():
    root = Path(__file__).parents[1] / "src/axiom_encode"
    calls = []
    for path in root.rglob("*.py"):
        if "allow_detached=True" not in path.read_text():
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and any(
                        k.arg == "allow_detached"
                        and isinstance(k.value, ast.Constant)
                        and k.value.value is True
                        for k in child.keywords
                    ):
                        calls.append((path.relative_to(root).as_posix(), node.name))
    assert sorted(calls) == sorted(
        [
            ("form_numeric_structure.py", "form_numeric_locator_spans"),
            ("harness/source_completeness.py", "_printed_operation_context"),
            ("harness/source_completeness.py", "_source_clause_spans"),
        ]
    )


def test_same_family_detached_source_preserves_actual_floor_evidence(pair):  # noqa: F811
    from axiom_encode.harness import actual_floor_evidence as floor

    expected = floor.witnesses(**pair)
    assert expected
    pair["source_context"]["ca/policy/test/detached"] = SOURCE
    assert floor.witnesses(**pair) == expected


def test_same_family_detached_source_preserves_signed_evidence(signed):  # noqa: F811
    from axiom_encode.harness import printed_signed_format_evidence as fmt

    expected = fmt.witnesses(**signed)
    assert expected
    signed["source_context"]["ca/policy/test/detached"] = SOURCE
    assert fmt.witnesses(**signed) == expected
