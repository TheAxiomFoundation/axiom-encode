"""Never turn missing lifetime coverage into an invented scalar scenario."""

from pathlib import Path

import pytest
import yaml

from axiom_encode import cli
from axiom_encode.rulespec_formula_identifiers import formula_calls_lifetime_reduction

PREFIX = "us:statutes/99/3#"


def rule(name, formula, **kwargs):
    return {
        "name": name,
        "kind": "derived",
        "entity": "Person",
        "dtype": "Decimal",
        "versions": [{"effective_from": "2000-01-01", "formula": formula}],
        **kwargs,
    }


@pytest.mark.parametrize(
    "formula, expected",
    [
        ("sum_over_periods(amount)", True),
        ("max_over_periods (amount)", True),
        ("count_over_periods\n(credit)", True),
        ("sum_top_n_over_periods(amount, count_over_periods(credit))", True),
        ("sum_over_periods + amount", False),
        ('if label == "count_over_periods(x)": 1 else: 0', False),
        ("amount # max_over_periods(amount)", False),
        ("calendar_years_to_months(count)", False),
    ],
)
def test_call_detection_distinguishes_functions_from_facts_and_text(formula, expected):
    assert formula_calls_lifetime_reduction(formula) is expected


def test_routing_traverses_all_versions_and_judgment_dependencies_without_cycles(
    tmp_path,
):
    rules = [
        rule(
            "history",
            "0",
            versions=[
                {"effective_from": "2000-01-01", "formula": "0"},
                {
                    "effective_from": "2026-01-01",
                    "formula": "count_over_periods(credit)",
                },
            ],
        ),
        rule("enough", "history > 1", dtype="Judgment"),
        rule("conditional", "if enough: 2 else: 0"),
        rule("cycle_a", "cycle_b + history"),
        rule("cycle_b", "cycle_a"),
        rule("scalar", "amount"),
    ]
    assert cli._non_scalar_fixture_output_names(
        {"rules": rules}, repo_path=tmp_path
    ) == {
        "history",
        "enough",
        "conditional",
        "cycle_a",
        "cycle_b",
    }


@pytest.mark.parametrize("kind", ["output", "zero"])
@pytest.mark.parametrize("existing_lifetime", [False, True])
def test_auto_repair_preserves_history_and_leaves_missing_coverage_for_encoder(
    tmp_path, kind, existing_lifetime
):
    root = tmp_path / "rulespec-us" / "us"
    relative = Path("statutes/99/3.yaml")
    source = root / relative
    source.parent.mkdir(parents=True)
    payload = {
        "format": "rulespec/v1",
        "rules": [
            rule("history", "count_over_periods(credit)"),
            rule("count", "history - dropouts"),
            rule("result", "if eligible: count else: 0"),
            rule("scalar", "if eligible: amount else: 0"),
        ],
    }
    source.write_text(yaml.safe_dump(payload, sort_keys=False))
    test = source.with_name("3.test.yaml")
    prior = (
        [
            {
                "name": "authentic_history",
                "period": {"start": "2026-01-01", "end": "2026-12-31"},
                "lifetime": {"entity": "Person", "periods": [], "batches": []},
                "output": {PREFIX + "history": "2"},
            }
        ]
        if existing_lifetime
        else []
    )
    # The existing mapping is an opaque preservation marker, not a valid or
    # executed fixture. Routing cannot fill in its history or change its values.
    test.write_text(yaml.safe_dump(prior, sort_keys=False))
    source_before = source.read_bytes()
    test_before = test.read_bytes()
    names = ["history", "count", "result"]
    if kind == "output":
        repair = cli._append_generated_derived_output_tests_if_missing
        issues = [
            f"Derived rule missing companion output coverage: `{PREFIX}{name}` is not asserted by the companion `.test.yaml` file."
            for name in names
        ]
    else:
        repair = cli._append_generic_zero_branch_tests_if_missing
        issues = [
            f"Zero branch test coverage missing: `{name}` returns 0." for name in names
        ]
    assert (
        repair(
            rules_file=source,
            test_file=test,
            repo_path=root,
            relative_output=relative,
            issues=issues,
        )
        == []
    )
    assert source.read_bytes() == source_before
    assert test.read_bytes() == test_before
    # A genuine scalar in the same module still gets the existing repair.
    scalar_issue = (
        f"Derived rule missing companion output coverage: `{PREFIX}scalar` is not asserted by the companion `.test.yaml` file."
        if kind == "output"
        else "Zero branch test coverage missing: `scalar` returns 0."
    )
    assert repair(
        rules_file=source,
        test_file=test,
        repo_path=root,
        relative_output=relative,
        issues=[scalar_issue],
    ) == [f"auto_{kind}_scalar"]
    cases = yaml.safe_load(test.read_text())
    assert cases[:-1] == prior
    assert "lifetime" not in cases[-1]
    assert cases[-1]["output"] == {PREFIX + "scalar": 0}


@pytest.mark.parametrize(
    "imported_kind", ["derived", "parameter", "missing", "cross_jurisdiction"]
)
@pytest.mark.parametrize("repair_kind", ["output", "zero"])
def test_imported_lifetime_or_unknown_dependency_cannot_receive_scalar_repair(
    tmp_path, imported_kind, repair_kind
):
    root = tmp_path / "rulespec-us" / "us"
    (root / "statutes/99").mkdir(parents=True)
    upstream = root / "statutes/99/2.yaml"
    prefix = "uk" if imported_kind == "cross_jurisdiction" else "us"
    if imported_kind not in {"missing", "cross_jurisdiction"}:
        upstream.write_text(
            yaml.safe_dump(
                {
                    "format": "rulespec/v1",
                    "rules": [
                        rule(
                            "history",
                            "2"
                            if imported_kind == "parameter"
                            else "count_over_periods(credit)",
                            kind=imported_kind,
                        ),
                    ],
                }
            )
        )
    payload = {
        "format": "rulespec/v1",
        "imports": [f"{prefix}:statutes/99/2#history"],
        "rules": [
            rule("bridge", "history + 1"),
            rule("result", "if eligible: bridge else: 0"),
        ],
    }
    source = root / "statutes/99/3.yaml"
    source.write_text(yaml.safe_dump(payload))
    test = source.with_name("3.test.yaml")
    test.write_text("[]\n")
    before = test.read_bytes()
    repair = (
        cli._append_generated_derived_output_tests_if_missing
        if repair_kind == "output"
        else cli._append_generic_zero_branch_tests_if_missing
    )
    issue = (
        f"Derived rule missing companion output coverage: `{PREFIX}result` is not asserted by the companion `.test.yaml` file."
        if repair_kind == "output"
        else "Zero branch test coverage missing: `result` returns 0."
    )
    repaired = repair(
        rules_file=source,
        test_file=test,
        repo_path=root,
        relative_output=Path("statutes/99/3.yaml"),
        issues=[issue],
    )
    if imported_kind == "parameter":
        assert repaired == [f"auto_{repair_kind}_result"]
    else:
        assert repaired == []
        assert test.read_bytes() == before


@pytest.mark.parametrize("order", ["derived_first", "parameter_first"])
@pytest.mark.parametrize("repair_kind", ["output", "zero"])
def test_duplicate_import_fragments_cannot_be_cleared_by_one_parameter(
    tmp_path, order, repair_kind
):
    root = tmp_path / "rulespec-us" / "us"
    (root / "statutes/99").mkdir(parents=True)
    for number, kind, formula in [
        (1, "derived", "count_over_periods(credit)"),
        (2, "parameter", "2"),
    ]:
        (root / f"statutes/99/{number}.yaml").write_text(
            yaml.safe_dump(
                {
                    "format": "rulespec/v1",
                    "rules": [rule("history", formula, kind=kind)],
                }
            )
        )
    imports = ["us:statutes/99/1#history", "us:statutes/99/2#history"]
    if order == "parameter_first":
        imports.reverse()
    source = root / "statutes/99/3.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "imports": imports,
                "rules": [rule("result", "if eligible: history else: 0")],
            }
        )
    )
    test = source.with_name("3.test.yaml")
    test.write_text("[]\n")
    repair = (
        cli._append_generated_derived_output_tests_if_missing
        if repair_kind == "output"
        else cli._append_generic_zero_branch_tests_if_missing
    )
    issue = (
        f"Derived rule missing companion output coverage: `{PREFIX}result` is not asserted by the companion `.test.yaml` file."
        if repair_kind == "output"
        else "Zero branch test coverage missing: `result` returns 0."
    )
    assert (
        repair(
            rules_file=source,
            test_file=test,
            repo_path=root,
            relative_output=Path("statutes/99/3.yaml"),
            issues=[issue],
        )
        == []
    )
    assert test.read_text() == "[]\n"
