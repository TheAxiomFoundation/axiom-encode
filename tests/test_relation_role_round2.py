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
@pytest.mark.parametrize("input_fragment", ["input.view", "result.input.view"])
@pytest.mark.parametrize("legacy", [False, True], ids=["guarded", "legacy"])
@pytest.mark.parametrize(
    "formula,value,expected", [("view + 1", 1, 2), ("if view: 1 else: 0", True, 1)]
)
def test_c2_c3_scalar_collision_companions(
    tmp_path, real_engine, executor, input_fragment, legacy, formula, value, expected
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
                        TARGET + "#" + input_fragment: value,
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
