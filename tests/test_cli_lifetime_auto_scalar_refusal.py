"""Never turn missing lifetime coverage into an invented scalar scenario."""

from functools import partial
from pathlib import Path

import pytest
import yaml

from axiom_encode import cli
from axiom_encode.rulespec_formula_identifiers import formula_calls_lifetime_reduction

PREFIX = "us:statutes/99/3#"

# Output repair writes generated cases only after an engine screen accepts
# them. Accepting every world here leaves lifetime routing as the only reason
# a repair can be refused.
OUTPUT_REPAIR = partial(
    cli._append_generated_derived_output_tests_if_missing,
    test_failure_checker=lambda _candidate: [],
)


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
        repair = OUTPUT_REPAIR
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
    "imported_kind",
    ["derived", "scalar_derived", "parameter", "missing", "cross_jurisdiction"],
)
@pytest.mark.parametrize("repair_kind", ["output", "zero"])
def test_only_an_import_calling_a_lifetime_reduction_refuses_scalar_repair(
    tmp_path, imported_kind, repair_kind
):
    """A plain scalar import (``history = credit * 2``), a parameter, or an
    import that can't be resolved keeps main's repairs; only an import whose
    resolved rules call a lifetime reduction is refused."""
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
                            {
                                "parameter": "2",
                                "scalar_derived": "credit * 2",
                            }.get(imported_kind, "count_over_periods(credit)"),
                            kind="parameter"
                            if imported_kind == "parameter"
                            else "derived",
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
        OUTPUT_REPAIR
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
    if imported_kind == "derived":
        assert repaired == []
        assert test.read_bytes() == before
    else:
        assert repaired == [f"auto_{repair_kind}_result"]


def _write_module(root, number, rules, imports=()):
    path = root / f"statutes/99/{number}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "rulespec/v1", "rules": rules}
    if imports:
        payload["imports"] = list(imports)
    path.write_text(yaml.safe_dump(payload))
    return path


def test_lifetime_reduction_behind_two_imports_is_still_refused(tmp_path):
    root = tmp_path / "rulespec-us" / "us"
    _write_module(root, 1, [rule("history", "count_over_periods(credit)")])
    _write_module(
        root, 2, [rule("relay", "history + 1")], imports=["us:statutes/99/1#history"]
    )
    rules_payload = {
        "format": "rulespec/v1",
        "imports": ["us:statutes/99/2#relay"],
        "rules": [rule("result", "relay * 2")],
    }
    assert cli._non_scalar_fixture_output_names(rules_payload, repo_path=root) == {
        "result"
    }


def test_import_cycle_terminates_and_keeps_scalar_repair(tmp_path):
    root = tmp_path / "rulespec-us" / "us"
    _write_module(root, 1, [rule("a", "b + 1")], imports=["us:statutes/99/2#b"])
    _write_module(root, 2, [rule("b", "a + 1")], imports=["us:statutes/99/1#a"])
    rules_payload = {
        "format": "rulespec/v1",
        "imports": ["us:statutes/99/1#a"],
        "rules": [rule("result", "a * 2")],
    }
    assert cli._non_scalar_fixture_output_names(rules_payload, repo_path=root) == set()


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
        OUTPUT_REPAIR
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


@pytest.mark.parametrize("with_rules", [False, True])
@pytest.mark.parametrize("checker", [None, "accept_all"])
def test_positive_judgment_repair_never_clones_or_synthesizes_lifetime_scenarios(
    tmp_path, with_rules, checker
):
    """A Judgment over a lifetime reduction gets no invented positive case.

    With the rules, the dependency itself refuses the repair. Without them,
    the only clonable scenario is a lifetime case, which is never cloned.
    """
    root = tmp_path / "rulespec-us" / "us"
    relative = Path("statutes/99/3.yaml")
    source = root / relative
    source.parent.mkdir(parents=True)
    source.write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "rules": [
                    rule("history", "count_over_periods(credit)"),
                    rule("eligible", "history >= 40", dtype="Judgment"),
                ],
            },
            sort_keys=False,
        )
    )
    test = source.with_name("3.test.yaml")
    lifetime_case = {
        "name": "authentic_history",
        "period": {"start": "2026-01-01", "end": "2026-12-31"},
        "lifetime": {
            "entity": "Person",
            "periods": [{"start": "2026-01-01", "end": "2026-12-31"}],
            "batches": [{"inputs": {"credit": ["1"]}}],
        },
        "output": {PREFIX + "history": "1"},
    }
    test.write_text(yaml.safe_dump([lifetime_case], sort_keys=False))

    repaired = cli._append_generated_judgment_positive_tests_if_missing(
        rules_file=source if with_rules else None,
        test_file=test,
        repo_path=root,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        relative_output=relative,
        issues=[
            "Judgment rule missing positive companion output coverage: "
            f"`{PREFIX}eligible` is not asserted as `holds` by the companion "
            "`.test.yaml` file."
        ],
        test_failure_checker=(lambda *_a, **_k: []) if checker else None,
    )

    assert repaired == []
    # The repair may re-serialize the file; its cases must be unchanged.
    assert yaml.safe_load(test.read_text()) == [lifetime_case]


def test_positive_judgment_repair_skips_lifetime_judgment_even_with_scalar_cases(
    tmp_path,
):
    """The local-lifetime skip holds on its own, with a clonable scalar case."""
    root = tmp_path / "rulespec-us" / "us"
    relative = Path("statutes/99/3.yaml")
    source = root / relative
    source.parent.mkdir(parents=True)
    source.write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "rules": [
                    rule("history", "count_over_periods(credit)"),
                    rule("eligible", "history >= 40", dtype="Judgment"),
                    rule("scalar", "amount + 1"),
                ],
            },
            sort_keys=False,
        )
    )
    test = source.with_name("3.test.yaml")
    scalar_case = {
        "name": "scalar_world",
        "period": {"start": "2026-01-01", "end": "2026-12-31"},
        "input": {PREFIX + "input.amount": 1},
        "output": {PREFIX + "scalar": 2},
    }
    test.write_text(yaml.safe_dump([scalar_case], sort_keys=False))

    repaired = cli._append_generated_judgment_positive_tests_if_missing(
        rules_file=source,
        test_file=test,
        repo_path=root,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        relative_output=relative,
        issues=[
            "Judgment rule missing positive companion output coverage: "
            f"`{PREFIX}eligible` is not asserted as `holds` by the companion "
            "`.test.yaml` file."
        ],
        test_failure_checker=lambda *_a, **_k: [],
    )

    assert repaired == []
    assert yaml.safe_load(test.read_text()) == [scalar_case]


def test_positive_judgment_repair_refuses_an_imported_lifetime_operand(tmp_path):
    """``eligible = history >= 40`` with ``history`` imported from a module that
    counts over periods gets no invented scalar case."""
    root = tmp_path / "rulespec-us" / "us"
    _write_module(root, 2, [rule("history", "count_over_periods(credit)")])
    source = _write_module(
        root,
        3,
        [
            rule("eligible", "history >= 40", dtype="Judgment"),
            rule("scalar", "amount + 1"),
        ],
        imports=["us:statutes/99/2#history"],
    )
    test = source.with_name("3.test.yaml")
    scalar_case = {
        "name": "scalar_world",
        "period": {"start": "2026-01-01", "end": "2026-12-31"},
        "input": {PREFIX + "input.amount": 1},
        "output": {PREFIX + "scalar": 2},
    }
    test.write_text(yaml.safe_dump([scalar_case], sort_keys=False))

    repaired = cli._append_generated_judgment_positive_tests_if_missing(
        rules_file=source,
        test_file=test,
        repo_path=root,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        relative_output=Path("statutes/99/3.yaml"),
        issues=[
            "Judgment rule missing positive companion output coverage: "
            f"`{PREFIX}eligible` is not asserted as `holds` by the companion "
            "`.test.yaml` file."
        ],
        test_failure_checker=lambda *_a, **_k: [],
    )

    assert repaired == []
    assert yaml.safe_load(test.read_text()) == [scalar_case]


def test_diamond_imports_resolve_each_module_once(tmp_path, monkeypatch):
    """B and C both import D, which imports E: each import base resolves once
    within one classification, and the shared module is classified once."""
    root = tmp_path / "rulespec-us" / "us"
    _write_module(root, 5, [rule("e", "count_over_periods(credit)")])
    _write_module(root, 4, [rule("d", "e + 1")], imports=["us:statutes/99/5#e"])
    _write_module(root, 2, [rule("b", "d + 1")], imports=["us:statutes/99/4#d"])
    _write_module(root, 3, [rule("c", "d * 2")], imports=["us:statutes/99/4#d"])
    rules_payload = {
        "format": "rulespec/v1",
        "imports": ["us:statutes/99/2#b", "us:statutes/99/3#c"],
        "rules": [rule("result", "b + c"), rule("plain", "amount + 1")],
    }
    resolved = []
    real = cli._resolve_rulespec_target_file

    def counting(target_ref, repo_path):
        resolved.append(str(target_ref))
        return real(target_ref, repo_path)

    monkeypatch.setattr(cli, "_resolve_rulespec_target_file", counting)
    classified = []
    real_classify = cli._classify_lifetime_dependent_outputs

    def counting_classify(payload, **kwargs):
        classified.append(tuple(r["name"] for r in payload["rules"]))
        return real_classify(payload, **kwargs)

    # The recursion looks the classifier up by name, so this sees every call.
    monkeypatch.setattr(cli, "_classify_lifetime_dependent_outputs", counting_classify)

    assert cli._non_scalar_fixture_output_names(rules_payload, repo_path=root) == {
        "result"
    }
    assert len(resolved) == len(set(resolved)) == 4
    # The top module, B, C, and the shared D and E once each.
    assert sorted(classified) == sorted(
        [("result", "plain"), ("b",), ("c",), ("d",), ("e",)]
    )


def test_scalar_import_beside_an_unrelated_lifetime_rule_keeps_repairs(tmp_path):
    """Only the imported name's own dependencies count, not its module's."""
    root = tmp_path / "rulespec-us" / "us"
    _write_module(
        root,
        2,
        [rule("history", "credit * 2"), rule("other", "count_over_periods(credit)")],
    )
    rules_payload = {
        "format": "rulespec/v1",
        "imports": ["us:statutes/99/2#history"],
        "rules": [rule("result", "history + 1")],
    }
    assert cli._non_scalar_fixture_output_names(rules_payload, repo_path=root) == set()


def _unmemoized_reference(payload, repo_path, seen=frozenset()):
    """The pre-memo recursion, kept as the differential reference."""
    rules = payload.get("rules")
    if not isinstance(rules, list):
        return set()
    by_name = {r["name"]: r for r in rules if isinstance(r, dict) and r.get("name")}
    unsafe = set()
    for raw in payload.get("imports") or ():
        name = raw.rsplit("#", 1)[-1].strip()
        resolved = cli._same_repo_import_base_and_file(raw, repo_path=repo_path)
        if resolved is None:
            continue
        module_file = resolved[1].resolve()
        if module_file in seen:
            continue
        imported = yaml.safe_load(module_file.read_text())
        if name in _unmemoized_reference(imported, repo_path, seen | {module_file}):
            unsafe.add(name)
    seeds = {
        name
        for name, r in by_name.items()
        if cli._generated_rule_formula_identifiers(r) & unsafe
        or any(formula_calls_lifetime_reduction(v["formula"]) for v in r["versions"])
    }
    return cli._expand_affected_generated_rule_dependencies(
        rules_by_name=by_name, seed_rules=seeds
    )


def _random_import_graph(root, rng, count=5):
    """Modules with several rules each: some lifetime, some local, some that
    read one or two imported names. Cycles and diamonds are common, and a
    module's other rules let a cut in one branch hide a reduction that a
    sibling rule or a later path would see."""
    for number in range(1, count + 1):
        names = [f"r{number}_{index}" for index in range(rng.randint(2, 4))]
        targets = rng.sample(
            [n for n in range(1, count + 1) if n != number], rng.randint(0, 3)
        )
        imported = []
        for target in targets:
            for index in rng.sample(range(4), rng.randint(1, 2)):
                imported.append(f"r{target}_{index}")
        imports = sorted(
            {f"us:statutes/99/{name[1:].split('_')[0]}#{name}" for name in imported}
        )
        rules = []
        for position, name in enumerate(names):
            roll = rng.random()
            if roll < 0.15:
                formula = "count_over_periods(credit)"
            elif roll < 0.6 and imported:
                formula = " + ".join(rng.sample(imported, min(2, len(imported))))
            elif position:
                formula = f"{rng.choice(names[:position])} + 1"
            else:
                formula = "amount"
            rules.append(rule(name, formula))
        _write_module(root, number, rules, imports=imports)


@pytest.mark.parametrize("seed", range(40))
def test_memoized_classification_matches_the_unmemoized_recursion(tmp_path, seed):
    """Random multi-rule import graphs, cycles and diamonds included, classify
    the same with and without the memo, from every module.

    Seed 38 builds a graph where a module whose child cut a cycle is later
    imported on a path without that cycle, so storing it would reuse a
    path-dependent answer."""
    import random

    root = tmp_path / "rulespec-us" / "us"
    _random_import_graph(root, random.Random(seed))
    for module in sorted((root / "statutes/99").glob("*.yaml")):
        payload = yaml.safe_load(module.read_text())
        assert cli._non_scalar_fixture_output_names(
            payload, repo_path=root
        ) == _unmemoized_reference(payload, root), module.name


def test_a_childs_cut_is_propagated_to_its_parent(tmp_path):
    """X holds a reduction and imports P; P reads C; C reads X. Classifying
    X first reaches P with X on the path, so C's import of X is cut and P
    looks scalar. P must then not be reused when the top module imports it
    directly, where C reaches X's reduction."""
    root = tmp_path / "rulespec-us" / "us"
    _write_module(
        root,
        1,
        [rule("x", "count_over_periods(credit)"), rule("x_uses_p", "p + 1")],
        imports=["us:statutes/99/2#p"],
    )
    _write_module(
        root,
        2,
        [rule("p", "c + 1"), rule("p_local", "amount + 1")],
        imports=["us:statutes/99/3#c"],
    )
    _write_module(root, 3, [rule("c", "x * 2")], imports=["us:statutes/99/1#x"])
    rules_payload = {
        "format": "rulespec/v1",
        "imports": ["us:statutes/99/1#x", "us:statutes/99/2#p"],
        "rules": [rule("from_p", "p + 1"), rule("plain", "amount + 1")],
    }
    expected = _unmemoized_reference(rules_payload, root)
    assert expected == {"from_p"}
    assert (
        cli._non_scalar_fixture_output_names(rules_payload, repo_path=root) == expected
    )


def test_result_computed_across_a_cut_cycle_is_not_reused(tmp_path):
    """Reached through A, module B sees A as a cut cycle and looks scalar.
    Reached directly, B sees A's reduction. The memo must not reuse the
    first answer for the second import."""
    root = tmp_path / "rulespec-us" / "us"
    _write_module(
        root,
        1,
        [rule("a", "count_over_periods(credit)"), rule("a_plus", "b + 1")],
        imports=["us:statutes/99/2#b"],
    )
    _write_module(root, 2, [rule("b", "a * 2")], imports=["us:statutes/99/1#a"])
    rules_payload = {
        "format": "rulespec/v1",
        "imports": ["us:statutes/99/1#a", "us:statutes/99/2#b"],
        "rules": [rule("from_b", "b + 1"), rule("plain", "amount + 1")],
    }
    expected = _unmemoized_reference(rules_payload, root)
    assert expected == {"from_b"}
    assert (
        cli._non_scalar_fixture_output_names(rules_payload, repo_path=root) == expected
    )
