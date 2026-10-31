"""Pin the independent second-round namespace and companion findings."""

from __future__ import annotations

import pytest
import yaml

from axiom_encode.formula_references import analyze_relation_predicate
from axiom_encode.harness import evals
from axiom_encode.harness import validator_pipeline as validation
from tests.test_companion_declared_entities import _companions
from tests.test_relation_role_review_fixes import relation, scalar_payload

LABEL = "# Synthetic encoder test input; not a RuleSpec module.\n"
TARGET = "us:policies/example/collision"


def collision_payload(formula, legacy=False):
    payload = scalar_payload(formula)
    view = relation("view", "candidate.broken")
    if legacy:
        view["derived_relation"].update(entity="SnapUnit", member_relation="members")
        view["versions"][0]["formula"] = True
    payload["rules"][:0] = [
        {
            "name": "pairs",
            "kind": "data_relation",
            "data_relation": {"arity": 2, "arguments": ["Person", "Person"]},
        },
        view,
    ]
    return payload


@pytest.mark.parametrize("legacy", [False, True], ids=["guarded", "legacy"])
@pytest.mark.parametrize("formula", ["view + 1", "if view: 1 else: 0", "len(view)"])
def test_a1_c2_c3_input_and_relation_namespaces(tmp_path, legacy, formula):
    payload = collision_payload(formula, legacy)
    root = tmp_path / "rulespec-us/us"
    module = root / "policies/example/collision.yaml"
    module.parent.mkdir(parents=True)
    content = yaml.safe_dump(payload)
    module.write_text(LABEL + content)
    scalar_read = formula != "len(view)"
    refs = analyze_relation_predicate(
        formula,
        predicate_terms=False,
        lowering_entry="rule",
        relation_names=frozenset({"view", "pairs"}),
    )
    assert refs.input_reads == ({"view"} if scalar_read else set())
    assert refs.relation_arguments == (set() if scalar_read else {"view"})
    assert (
        "view"
        in validation._rulespec_local_input_slots(
            payload, imports=set(), rules_file=module, policy_repo_path=root
        )
    ) is scalar_read
    assert (
        "view" in validation._rulespec_reference_summary(module).input_slots
    ) is scalar_read
    assert ("view" in evals._context_file_local_inputs(str(module))) is scalar_read
    terminals = ["view", "result"] if scalar_read else ["result"]
    assert (
        validation._rulespec_terminal_executable_names_from_payload(payload)
        == terminals
    )
    assert evals._context_file_terminal_exports(str(module)) == terminals
    cases = [{"output": {TARGET + "#result": 1}}]
    issues = validation.find_missing_derived_companion_output_issues(
        content, cases, rules_file=module, policy_repo_path=root
    )
    assert bool(issues) is (scalar_read and not legacy)
    if issues:
        assert "#view`" in issues[0]


@pytest.mark.parametrize("symbol", ["unused_members", "selection_flag"])
def test_a4_scalar_read_does_not_use_explicit_relation_import(symbol):
    payload = scalar_payload("len(selected)")
    payload["imports"] = ["us:policies/example/provider#relation.unused_members"]
    payload["rules"].insert(0, relation("selected", f"{symbol} > 0"))
    issues = validation.find_unused_import_issues(yaml.safe_dump(payload))
    assert len(issues) == 1
    assert "#relation.unused_members`" in issues[0]


@pytest.mark.parametrize("executor", ["cli", "validator"])
@pytest.mark.parametrize("legacy", [False, True], ids=["guarded", "legacy"])
@pytest.mark.parametrize(
    "formula,value,expected", [("view + 1", 1, 2), ("if view: 1 else: 0", True, 1)]
)
def test_c2_c3_scalar_collision_companions(
    tmp_path, real_engine, executor, legacy, formula, value, expected
):
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/collision.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(LABEL + yaml.safe_dump(collision_payload(formula, legacy)))
    module.with_suffix(".test.yaml").write_text(
        LABEL
        + yaml.safe_dump(
            [
                {
                    "name": "scalar_collision",
                    "period": "2026-01",
                    "input": {
                        TARGET + "#input.view": value,
                        TARGET + "#relation.pairs": [],
                    },
                    "output": {TARGET + "#result": expected},
                }
            ]
        )
    )
    assert (
        _companions(
            executor,
            binary=real_engine.binary,
            engine_root=real_engine.root,
            root=root,
            module=module,
            artifact=tmp_path / "compiled.json",
        )
        == []
    )


def test_a1_unused_members_boolean_read_leaves_relation_uncovered(tmp_path):
    payload = collision_payload("if unused_members: 1 else: 0")
    payload["rules"][1]["name"] = "unused_members"
    payload["rules"][-1]["name"] = "count"
    root = tmp_path / "rulespec-us/us"
    module = root / "policies/example/collision.yaml"
    module.parent.mkdir(parents=True)
    content = yaml.safe_dump(payload)
    module.write_text(LABEL + content)
    issues = validation.find_missing_derived_companion_output_issues(
        content,
        [{"output": {TARGET + "#count": 1}}],
        rules_file=module,
        policy_repo_path=root,
    )
    assert len(issues) == 1 and "#unused_members`" in issues[0]
    assert validation._rulespec_terminal_executable_names_from_payload(payload) == [
        "unused_members",
        "count",
    ]
    assert evals._context_file_terminal_exports(str(module)) == [
        "unused_members",
        "count",
    ]


@pytest.mark.parametrize("symbol", ["max", "count"])
@pytest.mark.parametrize(
    "formula,used",
    [
        ("{symbol} + 1", True),
        ("{symbol} + @", True),
        ("max(a, b)", False),
        ('"{symbol}"', False),
        ("1 # {symbol}", False),
    ],
)
def test_c1_proof_imports_read_bare_builtin_names(symbol, formula, used):
    payload = scalar_payload(formula.format(symbol=symbol))
    payload["rules"][0]["metadata"] = {
        "proof": {
            "atoms": [{"import": {"target": f"us:policies/example/provider#{symbol}"}}]
        }
    }
    issues = validation.find_proof_import_reference_issues(yaml.safe_dump(payload))
    assert bool(issues) is not used


@pytest.mark.parametrize("symbol,value", [("max", 1), ("count", 2)])
@pytest.mark.parametrize("executor", ["cli", "validator"])
def test_c1_builtin_scalar_import_compiles_and_executes(
    tmp_path, real_engine, symbol, value, executor
):
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/collision.yaml"
    module.parent.mkdir(parents=True)
    provider = scalar_payload(value)
    provider["rules"][0].pop("entity")
    provider["rules"][0]["name"] = symbol
    module.with_name("provider.yaml").write_text(LABEL + yaml.safe_dump(provider))
    payload = scalar_payload(f"{symbol} + 1")
    payload["imports"] = [f"us:policies/example/provider#{symbol}"]
    payload["rules"][0]["metadata"] = {
        "proof": {
            "atoms": [{"import": {"target": f"us:policies/example/provider#{symbol}"}}]
        }
    }
    content = yaml.safe_dump(payload)
    assert validation.find_proof_import_reference_issues(content) == []
    module.write_text(LABEL + content)
    module.with_suffix(".test.yaml").write_text(
        LABEL
        + yaml.safe_dump(
            [
                {
                    "name": "builtin_import",
                    "period": "2026-01",
                    "input": {},
                    "output": {TARGET + "#result": value + 1},
                }
            ]
        )
    )
    assert (
        _companions(
            executor,
            binary=real_engine.binary,
            engine_root=real_engine.root,
            root=root,
            module=module,
            artifact=tmp_path / "compiled.json",
        )
        == []
    )


@pytest.mark.parametrize("symbol", ["max", "count"])
def test_c1_builtin_value_read_uses_election_input(symbol):
    payload = scalar_payload(f"if {symbol}: 1 else: 0")
    payload["inputs"] = [
        {
            "name": symbol,
            "dtype": "Judgment",
            "election": {"actor": "Person", "provision": "us/regulation/7/273/1/b/3"},
        }
    ]
    assert validation.find_relation_role_issues(yaml.safe_dump(payload)) == []


@pytest.mark.parametrize("symbol", ["max", "count"])
def test_c1_legacy_predicate_builtin_read_requires_assignment(symbol):
    payload = collision_payload("len(view)", legacy=True)
    payload["module"] = {"proof_validation": {"required": True}}
    payload["rules"][1]["versions"][0]["formula"] = f"{symbol} > 0"
    issues = validation.find_test_input_assignment_issues(
        yaml.safe_dump(payload),
        [
            {
                "name": "missing",
                "period": "2026-01",
                "input": {},
                "output": {TARGET + "#result": 0},
            }
        ],
    )
    assert len(issues) == 1
    assert symbol in issues[0]


def legacy_bare_membership_payload():
    """Build a synthetic legacy filtered predicate with a bare relation read."""
    config = {
        "arity": 2,
        "slot_entities": ["Person", "Household"],
        "current_slot": 1,
        "related_slot": 0,
        "source_relation": "pairs",
    }
    return {
        "format": "rulespec/v1",
        "rules": [
            {
                "name": "pairs",
                "kind": "data_relation",
                "data_relation": {"arity": 2, "arguments": ["Person", "Household"]},
            },
            {
                "name": "view",
                "kind": "derived_relation",
                "derived_relation": dict(config),
                "versions": [{"effective_from": "2026-01-01", "formula": True}],
            },
            {
                "name": "members",
                "kind": "derived_relation",
                "derived_relation": {
                    **config,
                    "entity": "SnapUnit",
                    "member_relation": "selected",
                },
                "versions": [{"effective_from": "2026-01-01", "formula": "view"}],
            },
            {
                "name": "result",
                "kind": "derived",
                "entity": "SnapUnit",
                "dtype": "Integer",
                "versions": [
                    {"effective_from": "2026-01-01", "formula": "len(selected)"}
                ],
            },
        ],
    }


def test_legacy_filtered_bare_relation_uses_judgment_lowering(tmp_path):
    from axiom_encode.formula_references import rule_uses_relation_predicate

    payload = legacy_bare_membership_payload()
    rule = payload["rules"][2]
    assert rule_uses_relation_predicate(rule) is False
    refs = analyze_relation_predicate(
        "view",
        predicate_terms=False,
        lowering_entry="derived_relation",
        relation_names=frozenset({"view", "pairs", "members"}),
    )
    assert refs.relation_arguments == {"view"}
    assert refs.input_reads == set()
    root = tmp_path / "rulespec-us/us"
    module = root / "policies/example/collision.yaml"
    module.parent.mkdir(parents=True)
    content = yaml.safe_dump(payload)
    module.write_text(LABEL + content)
    assert (
        validation.find_missing_derived_companion_output_issues(
            content,
            [{"output": {TARGET + "#result": 1}}],
            rules_file=module,
            policy_repo_path=root,
        )
        == []
    )
    assert validation._rulespec_terminal_executable_names_from_payload(payload) == [
        "result"
    ]
    assert evals._context_file_terminal_exports(str(module)) == ["result"]


def test_legacy_filtered_bare_relation_matches_compiled_artifact(tmp_path, real_engine):
    from tests.test_companion_declared_entities import _compile

    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/collision.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(LABEL + yaml.safe_dump(legacy_bare_membership_payload()))
    artifact = _compile(
        real_engine.binary, real_engine.root, root, module, tmp_path / "compiled.json"
    )
    native = next(
        relation
        for relation in artifact["program"]["relations"]
        if relation["name"] == TARGET + "#relation.members"
    )
    assert native["derivation"]["predicate"] == {
        "kind": "relation_member",
        "relation": TARGET + "#relation.view",
        "current_slot": 1,
        "related_slot": 0,
    }
