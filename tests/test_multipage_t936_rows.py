"""Full-source layout does not grant runtime or numeric coverage."""

import ast
from pathlib import Path

import pytest

from axiom_encode.harness import source_completeness as s
from axiom_encode.printed_source_rows import recognize_printed_rows

SOURCE = (Path(__file__).parent / "fixtures/multipage_t936_source.txt").read_text()


def test_explicit_source_only_and_no_runtime_or_numeric_credit():
    assert recognize_printed_rows(SOURCE) is None
    assert recognize_printed_rows(SOURCE, allow_detached=True) is None
    for value in [False, 1, "true", object()]:
        assert recognize_printed_rows(SOURCE, allow_multipage=value) is None
    rows = recognize_printed_rows(SOURCE, allow_multipage=True)
    assert rows is not None
    assert rows.same_page_floors == ()
    assert len(rows.numeric_locator_spans) == 54
    assert s._unconflicted_printed_rows(SOURCE, ()) is None
    assert rows.note_marker_spans == ((6541, 6544), (6817, 6820))


def test_full_notes_remain_without_invented_legal_children():
    branches = s.recognize_source_structure(SOURCE)
    assert not any(b.path in [("1",), ("2",)] for b in branches)
    clauses = list(s._source_clause_spans(SOURCE, branches=branches))
    for start, end in [(6541, 6816), (6817, 7089)]:
        assert (start, end, SOURCE[start:end]) in clauses
    assert not any(a < 7113 < b for a, b, _ in clauses)
    other = "(1) A benefit is payable.\n(2) A condition applies."
    assert {b.path for b in s.recognize_source_structure(other)} >= {("1",), ("2",)}


def test_categories_keep_include_exclude_context_and_all_caps():
    rows = recognize_printed_rows(SOURCE, allow_multipage=True)
    clauses = list(
        s._source_clause_spans(SOURCE, branches=s.recognize_source_structure(SOURCE))
    )
    for a, b in [(2877, 3984), (3985, 4384), (4385, 5156), (5157, 5467)]:
        assert (a, b) in rows.atomic_spans
        assert (a, b) in rows.ownership_spans
        assert not any(a < point < b for point in rows.boundaries)
        assert (a, b, SOURCE[a:b]) in clauses
    assert "$30,000" in SOURCE[2877:3984]
    assert "March 18, 2019" in SOURCE[2877:3984]


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("$30,000", "$31,000"),
        ("after 1987", "after 1986"),
        ("1988 to 1991", "1988 to 1992"),
        ("× 50%", "× 40%"),
        ("Line 15 minus line 16", "Line 16 minus line 15"),
        ("Page 3 of 4", "Page 3 of 5"),
        ("(2) To calculate", "(3) To calculate"),
        ("Do not include:", "Include:"),
        ("except timber resources", "including timber resources"),
        ('Otherwise, enter "0" on line 19', 'Otherwise, enter "1" on line 19'),
    ],
)
def test_changed_operative_content_or_structure_refuses(old, new):
    assert old in SOURCE
    changed = SOURCE.replace(old, new, 1)
    assert recognize_printed_rows(changed, allow_multipage=True) is None


@pytest.mark.parametrize(
    "changed",
    [
        SOURCE + "\nUnknown requirement.",
        SOURCE[1:],
        SOURCE.replace("Chart A\n", "Chart A\nChart A\n", 1),
    ],
)
def test_full_consumption_is_required(changed):
    assert recognize_printed_rows(changed, allow_multipage=True) is None


def test_only_reviewed_source_callers_opt_in():
    root = Path(s.__file__).parents[1]
    found = set()
    for p in root.rglob("*.py"):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                if any(
                    k.arg == "allow_multipage"
                    and isinstance(k.value, ast.Constant)
                    and k.value.value is True
                    for k in call.keywords
                ):
                    found.add((p.name, node.name))
    assert found == {("form_numeric_structure.py", "form_numeric_locator_spans")} | {
        ("source_completeness.py", name)
        for name in [
            "_recognize_source_structure",
            "_printed_operation_context",
            "_source_clause_spans",
            "_source_exception_branches",
            "_annual_form_source_evidence",
        ]
    }


def test_only_title_dashes_are_typography_not_arithmetic():
    rows = recognize_printed_rows(SOURCE, allow_multipage=True)
    assert len(rows.typography_spans) == 24
    for a, b in [
        (1085, 1086),
        (1143, 1144),
        (1908, 1909),
        (1965, 1966),
        (5529, 5530),
        (5577, 5578),
    ]:
        assert SOURCE[a:b] == "–"
        line = SOURCE[SOURCE.rfind("\n", 0, a) + 1 : SOURCE.find("\n", b)]
        assert line.startswith(("Part 1 – ", "Part 2 – ", "Part 3 – "))
    branches = s.recognize_source_structure(SOURCE)
    formulas = s._source_formula_branches(
        SOURCE, branches=branches, active_branches=branches, deferred_paths=set()
    )
    assert not any(
        x.text.startswith(("Part 1 – ", "Part 2 – ", "Part 3 – ")) for x in formulas
    )
    for clause in [
        "Line 1 minus line 2 = 3",
        "Line 11 minus line 12 = (cid:379) – 13",
    ]:
        assert any(clause in x.text for x in formulas)
    changed = SOURCE.replace("Part 3 – Cumulative", "Part 3 × Cumulative", 1)
    assert recognize_printed_rows(changed, allow_multipage=True) is None


def test_caption_glyphs_preserve_linked_computations_and_substantive_rates():
    rows = recognize_printed_rows(SOURCE, allow_multipage=True)
    assert set(
        (
            (7384, 7385),
            (7850, 7851),
            (7973, 7974),
            (8473, 8474),
        )
    ) <= set(rows.typography_spans)
    assert all(
        not any(c.isdigit() for c in SOURCE[a:b]) for a, b in rows.typography_spans
    )
    branches = s.recognize_source_structure(SOURCE)
    formulas = s._source_formula_branches(
        SOURCE, branches=branches, active_branches=branches, deferred_paths=set()
    )
    captions = [
        "Amount from line 17215 of Form T2048 – 2",
        "Amount from line 17216 of Form T2048 – 12",
        "Inclusion rate × 50% 10",
        "Inclusion rate × 50% 18",
    ]
    assert not any(x.text in captions for x in formulas)
    for start, end in [(7388, 7411), (7978, 8016), (7859, 7935), (8482, 8546)]:
        assert any(
            x.start == start and x.end == end and x.text == SOURCE[start:end]
            for x in formulas
        )
    assert len(rows.numeric_locator_spans) == 54


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("Line 1 minus line 2", "Line 2 minus line 1"),
        ("Line 11 minus line 12", "Line 12 minus line 11"),
        ("percentage on line 10", "percentage on line 18"),
        ("percentage on line 18", "percentage on line 10"),
        ("Inclusion rate × 50% 10", "Inclusion rate 50% × 10"),
        (
            "Amount from line 17215 of Form T2048 – 2",
            "Amount from line 17215 of Form T2048 + 2",
        ),
    ],
)
def test_caption_application_link_drift_refuses(old, new):
    assert (
        recognize_printed_rows(SOURCE.replace(old, new, 1), allow_multipage=True)
        is None
    )


def test_total_rows_keep_application_and_result_caption_together():
    branches = s.recognize_source_structure(SOURCE)
    clauses = list(s._source_clause_spans(SOURCE, branches=branches))
    for a, b in [(1835, 1898), (2717, 2780)]:
        assert (a, b, SOURCE[a:b]) in clauses
    formulas = s._source_formula_branches(
        SOURCE, branches=branches, active_branches=branches, deferred_paths=set()
    )
    for literal in [
        "50% of exploration",
        "annuity payments taxable",
        "50% of income",
        "Add lines 1 to 7.",
        "Add lines 8 to 13.",
    ]:
        assert any(literal in x.text for x in formulas)
    exceptions = s._source_exception_branches(
        SOURCE,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
        formula_branches=formulas,
    )
    for a, b in [(1647, 1834), (2562, 2716)]:
        assert (a, b, SOURCE[a:b]) in [(x.start, x.end, x.text) for x in exceptions]
    for a, b in [(2239, 2465), (1446, 1506), (2466, 2561)]:
        assert (a, b, SOURCE[a:b]) in [(x.start, x.end, x.text) for x in formulas]
    rows = recognize_printed_rows(SOURCE, allow_multipage=True)
    annuity = SOURCE.index("minus the capital portion")
    assert not any(a <= annuity < b for a, b in rows.typography_spans)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("Add lines 1 to 7.", "Add lines 7 to 1."),
        ("Add lines 8 to 13.", "Add lines 8 to 12."),
        ("line 16 minus line 17", "line 17 minus line 16"),
        ("other than allowable capital losses", "including allowable capital losses"),
        ("minus the capital portion", "plus the capital portion"),
        ("Otherwise, enter whichever is less", "Otherwise, enter whichever is more"),
    ],
)
def test_changed_group_arithmetic_or_local_semantics_refuses(old, new):
    assert old in SOURCE
    assert (
        recognize_printed_rows(SOURCE.replace(old, new, 1), allow_multipage=True)
        is None
    )
