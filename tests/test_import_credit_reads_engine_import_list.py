"""An import excuses a finding only when the rules engine loads that import.

Each case reuses the fixture of an existing test in
``tests/test_rulespec_validation.py`` that shows one import-credit check
refusing a module with no import. The case captures that test's call to the
check and runs the check again on three kinds of variant:

- the same module with text that only looks like an imports block (in a
  ``summary`` block scalar, nested under ``module:``, in a top-level block
  scalar): the finding must stay;
- the same module with the same items in a real top-level ``imports`` list:
  the finding must go, so the credit rule itself is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import evals
from axiom_encode.harness.validator_pipeline import (
    ValidatorPipeline,
    _engine_import_items,
    _extract_import_items_from_content,
)
from tests import test_rulespec_validation as fixtures


def _identity(content: str) -> str:
    return content


@dataclass(frozen=True)
class Case:
    fixture_test: str
    check: str
    items: tuple[str, ...]
    issue_marker: str
    is_method: bool = True
    prepare: Callable[[str], str] = field(default=_identity)


CASES = {
    "same_section_subsection_import": Case(
        "test_same_section_subsection_reference_requires_import",
        "find_missing_same_section_subsection_import_issues",
        (
            "us:statutes/7/2015/e"
            "#person_meets_higher_education_student_eligibility_requirements",
        ),
        "Same-section subsection import missing",
        is_method=False,
    ),
    "child_fragment_reencoding": Case(
        "test_child_fragment_reencoding_rejects_parent_copying_child_inputs",
        "find_child_fragment_reencoding_issues",
        ("us:statutes/26/63/c/5",),
        "Child fragment re-encoded",
        is_method=False,
    ),
    "child_exception_import": Case(
        "test_parent_exception_list_requires_child_exception_imports",
        "find_missing_child_exception_import_issues",
        ("us:statutes/26/163",),
        "Parent exception-list child import missing",
        is_method=False,
        prepare=lambda content: content.replace(
            "formula: interest_paid_on_qualifying_indebtedness",
            "formula: interest_paid_on_qualifying_indebtedness "
            "and not fleet_sales_loan_exception_applies",
        ),
    ),
    "cross_reference_exception_placeholder": Case(
        "test_cross_reference_exception_placeholder_requires_import",
        "_check_cross_reference_exception_placeholders",
        ("us:statutes/7/2015/b#section_2015_b_exception_applies",),
        "Cross-reference placeholder",
    ),
    "encoded_cross_reference_placeholder": Case(
        "test_encoded_cross_reference_placeholder_rejects_under_section_input_"
        "when_source_exists",
        "_check_encoded_cross_reference_placeholders",
        ("us:statutes/26/1211#sum_allowed_under_section_1211",),
        "Encoded cross-reference placeholder",
    ),
    "cross_reference_numeric_placeholder": Case(
        "test_cross_reference_numeric_placeholder_rejects_locally_supplied_rates",
        "_check_cross_reference_numeric_placeholders",
        ("us:statutes/26/3101", "us:statutes/26/3241"),
        "Cross-reference numeric placeholder",
    ),
    "cross_reference_numeric_placeholder_title_prefix": Case(
        "test_cross_reference_numeric_placeholder_rejects_locally_supplied_rates",
        "_check_cross_reference_numeric_placeholders",
        ("us:statutes/26",),
        "Cross-reference numeric placeholder",
    ),
    "cross_reference_base_mechanics": Case(
        "test_cross_reference_base_mechanics_raw_compensation_tax_is_rejected",
        "_check_unapplied_cross_reference_base_mechanics",
        ("us:statutes/26/3231#compensation_received_for_services",),
        "Cross-reference base mechanics omitted",
    ),
    "cross_statute_definition_import": Case(
        "test_cross_statute_definition_import_check_uses_cited_title_and_"
        "existing_targets",
        "_check_cross_statute_definition_imports",
        ("us:statutes/7/2014/e#snap_earned_income_deduction",),
        "Cross-statute definition import missing",
    ),
    "embedded_scalar_literal": Case(
        "test_embedded_formula_numeric_guard_allows_imported_parameter_scalar_equality",
        "_check_embedded_scalar_literals",
        (
            "us-nc:policies/dhhs/fns/fns-227-non-citizen-requirements/page-1"
            "#lpr_waiting_period_years",
        ),
        "Embedded scalar literal",
        prepare=lambda content: content.split("\nimports:\n")[0] + "\n",
    ),
}


def _item_lines(items: tuple[str, ...], indent: str) -> str:
    return "".join(f"{indent}- {item}\n" for item in items)


def _after_format_line(content: str, block: str) -> str:
    head, separator, tail = content.partition("\n")
    assert separator and head.startswith("format:")
    return head + "\n" + block + tail


def _fake_in_summary(content: str, items: tuple[str, ...]) -> str:
    block = "    imports:\n" + _item_lines(items, "      ")
    if "  summary: |-\n" in content:
        return content.replace("  summary: |-\n", "  summary: |-\n" + block, 1)
    assert "\nmodule:\n" not in content
    return content.replace(
        "\nrules:\n", "\nmodule:\n  summary: |-\n" + block + "\nrules:\n", 1
    )


def _fake_under_module(content: str, items: tuple[str, ...]) -> str:
    block = "  imports:\n" + _item_lines(items, "    ")
    if "\nmodule:\n" in content:
        return content.replace("\nmodule:\n", "\nmodule:\n" + block, 1)
    return content.replace("\nrules:\n", "\nmodule:\n" + block + "rules:\n", 1)


def _fake_in_top_level_block_scalar(content: str, items: tuple[str, ...]) -> str:
    block = "notes: |\n  imports:\n" + _item_lines(items, "    ")
    return _after_format_line(content, block)


FAKE_SHAPES = {
    "summary_block_scalar": _fake_in_summary,
    "nested_under_module": _fake_under_module,
    "top_level_block_scalar": _fake_in_top_level_block_scalar,
}


def _real_top_level_import(content: str, items: tuple[str, ...]) -> str:
    if "\nimports:\n" in content:
        return content.replace(
            "\nimports:\n", "\nimports:\n" + _item_lines(items, "  "), 1
        )
    return _after_format_line(content, "imports:\n" + _item_lines(items, "  "))


def _run_variants(case: Case, tmp_path: Path, monkeypatch) -> dict[str, object]:
    """Run the fixture test once; inside its call, run the check per variant."""

    seen: dict[str, object] = {}

    def variants(base: str) -> dict[str, str]:
        result = {"base": base, "real": _real_top_level_import(base, case.items)}
        for shape, build in FAKE_SHAPES.items():
            result[shape] = build(base, case.items)
        return result

    def record(name: str, content: str, issues: list[str]) -> None:
        seen[name] = issues
        seen[name + ":imports"] = (yaml.safe_load(content) or {}).get("imports")
        seen[name + ":line_reader"] = _extract_import_items_from_content(content)
        seen[name + ":engine_reader"] = _engine_import_items(content)

    if case.is_method:
        original = getattr(ValidatorPipeline, case.check)

        def method_wrapper(self, rules_file):
            untouched = rules_file.read_text()
            if "base" not in seen:
                for name, content in variants(case.prepare(untouched)).items():
                    rules_file.write_text(content)
                    record(name, content, original(self, rules_file))
                rules_file.write_text(untouched)
            return original(self, rules_file)

        monkeypatch.setattr(ValidatorPipeline, case.check, method_wrapper)
    else:
        original = getattr(fixtures, case.check)

        def function_wrapper(content, **kwargs):
            if "base" not in seen:
                for name, variant in variants(case.prepare(content)).items():
                    record(name, variant, original(variant, **kwargs))
            return original(content, **kwargs)

        monkeypatch.setattr(fixtures, case.check, function_wrapper)

    getattr(fixtures, case.fixture_test)(tmp_path)
    assert "base" in seen, f"{case.fixture_test} no longer calls {case.check}"
    return seen


@pytest.mark.parametrize("name", sorted(CASES))
def test_text_that_looks_like_an_imports_block_excuses_no_finding(
    name, tmp_path, monkeypatch
):
    case = CASES[name]
    seen = _run_variants(case, tmp_path, monkeypatch)

    base = seen["base"]
    assert base and all(case.issue_marker in issue for issue in base)
    for shape in FAKE_SHAPES:
        # The variant differs from the base only in text the engine ignores.
        assert seen[shape + ":imports"] == seen["base:imports"], shape
        assert set(case.items) <= set(seen[shape + ":line_reader"]), shape
        assert seen[shape + ":engine_reader"] == seen["base:engine_reader"], shape
        assert seen[shape] == base, shape


@pytest.mark.parametrize("name", sorted(CASES))
def test_real_top_level_import_still_excuses_the_finding(name, tmp_path, monkeypatch):
    case = CASES[name]
    seen = _run_variants(case, tmp_path, monkeypatch)

    assert set(case.items) <= set(seen["real:imports"])
    assert set(case.items) <= set(seen["real:engine_reader"])
    assert seen["base"]
    assert seen["real"] == []


PROVIDER = """format: rulespec/v1
rules:
  - name: some_amount
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: 4321
"""


def _provider_root(tmp_path: Path) -> Path:
    root = tmp_path / "rulespec-us" / "us"
    provider = root / "statutes" / "1" / "1.yaml"
    provider.parent.mkdir(parents=True)
    provider.write_text(PROVIDER)
    return root


@pytest.mark.parametrize(
    "candidate",
    (
        "format: rulespec/v1\nmodule:\n  summary: |-\n    imports:\n"
        "      - us:statutes/1/1#some_amount\n    The amount is $4,321.\nrules: []\n",
        "format: rulespec/v1\nmodule:\n  imports:\n"
        "    - us:statutes/1/1#some_amount\nrules: []\n",
        "format: rulespec/v1\nnotes: |\n  imports:\n"
        "    - us:statutes/1/1#some_amount\nrules: []\n",
    ),
    ids=sorted(FAKE_SHAPES, reverse=True),
)
def test_untyped_numeric_occurrences_ignore_text_that_looks_like_imports(
    tmp_path, candidate
):
    root = _provider_root(tmp_path)

    # The precedent-context reader still sees the text; numeric credit does not.
    assert evals._extract_import_targets(candidate) == ["us:statutes/1/1"]
    assert evals._engine_import_targets(candidate) == []
    assert evals._imported_named_scalar_occurrences(candidate, root) == {}


def test_untyped_numeric_occurrences_keep_a_real_top_level_import(tmp_path):
    root = _provider_root(tmp_path)
    candidate = (
        "format: rulespec/v1\nimports:\n  - us:statutes/1/1#some_amount\nrules: []\n"
    )

    assert evals._engine_import_targets(candidate) == ["us:statutes/1/1"]
    assert evals._imported_named_scalar_occurrences(candidate, root)[4321.0] == 1
