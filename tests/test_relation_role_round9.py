"""Pin the ninth-round alias, proof and direction findings."""

from __future__ import annotations

import pytest
import yaml

from axiom_encode.formula_references import spelled_names
from axiom_encode.harness import validator_pipeline as validation
from tests.test_companion_declared_entities import _compile
from tests.test_relation_role_round6 import (
    PAIRS,
    PROVIDER,
    TARGET,
    VIEW,
    rule,
    write_module,
)
from tests.test_relation_role_round7 import program, view
from tests.test_relation_role_round8 import (
    EVERYONE,
    certified,
    compose,
    run_composed,
    selector_proof,
)

FIRST = f"{TARGET}#relation.first"
SCALAR = f"{PROVIDER}#view"


def redirected(alias=FIRST):
    """Name another relation's canonical ID as a member alias."""
    return program(
        PAIRS,
        view("first", "pairs", "candidate.uncovered", slots=EVERYONE, entity="Other"),
        view(
            "second",
            "pairs",
            "true",
            slots=EVERYONE,
            entity="Person",
            member_relation=alias,
        ),
        rule("result", "Person", "len(first) + len(second)"),
        rule("keep_first", "Other", "len(first)"),
    )


def alias_issues(content, **paths):
    return [
        issue
        for issue in validation.find_relation_role_issues(content, **paths)
        if issue.startswith("[relation-member-alias]")
    ]


def test_the_role_gate_refuses_a_member_alias_spelled_as_a_relation_id():
    assert alias_issues(yaml.safe_dump(redirected())) == [
        "[relation-member-alias] Derived relation `second` spells member alias "
        f"`{FIRST}` as a relation ID. The engine would send that relation's own "
        "reads through the alias, so use a bare name."
    ]
    assert alias_issues(yaml.safe_dump(redirected("members"))) == []


def test_the_role_gate_refuses_a_relation_id_alias_in_an_import(tmp_path):
    consumer = program(
        view("local", "pairs", "true", slots=EVERYONE),
        imports=[PROVIDER],
    )
    root, module = write_module(tmp_path, consumer, provider=redirected())
    issues = alias_issues(
        module.read_text(), rules_file=module, policy_repo_path=root / "us"
    )
    assert len(issues) == 1 and "`second`" in issues[0]


def test_a_relation_id_alias_takes_the_engines_read_of_that_relation(
    tmp_path, real_engine
):
    root, module = write_module(tmp_path, redirected())
    compiled = _compile(
        real_engine.binary, real_engine.root, root, module, tmp_path / "compiled.json"
    )
    result = next(
        derived
        for derived in compiled["program"]["derived"]
        if derived["id"].endswith("#result")
    )
    assert FIRST not in yaml.safe_dump(result)


@pytest.mark.parametrize(
    "formula,expected",
    [
        ("view + 1", {"view"}),
        ("match view: _ => 1", {"view"}),
        ("len(pairs, view)", {"pairs", "view"}),
        ("len(pairs.view)", {"pairs", "view"}),
        ("view(1)", set()),
        ("1 + (", set()),
    ],
)
def test_spelled_names_keep_what_lowering_drops(formula, expected):
    assert spelled_names(formula) == expected


@pytest.mark.parametrize(
    "formula", ["match view: _ => 1", "len(pairs, view)", "len(pairs.view)"]
)
def test_a_bare_proof_selector_keeps_a_read_lowering_discards(formula):
    bare = {"target": SCALAR, "output": "view"}
    assert certified(selector_proof(formula, imports=[SCALAR], atom=bare))


@pytest.mark.parametrize(
    "formula,atom",
    [
        ("max(1, 2)", {"target": f"{PROVIDER}#max", "output": "max"}),
        ("match view: _ => 1", {"target": VIEW, "output": VIEW}),
    ],
)
def test_a_discarded_read_certifies_only_a_bare_selector(formula, atom):
    assert not certified(selector_proof(formula, imports=[VIEW], atom=atom))


def sourced_proof(atom):
    relation = view("selected", VIEW, "true", slots=EVERYONE)
    relation["metadata"] = {"proof": {"atoms": [{"import": atom}]}}
    return yaml.safe_dump(program(relation, imports=[VIEW]))


@pytest.mark.parametrize(
    "atom,expected",
    [
        ({"target": VIEW, "output": "view"}, True),
        ({"target": VIEW, "output": VIEW}, True),
        ({"target": VIEW}, False),
    ],
)
def test_a_source_certifies_only_a_proof_that_names_an_output(atom, expected):
    assert certified(sourced_proof(atom)) is expected


def test_an_undeclared_key_settles_opposite_reads_of_a_typed_relation(
    tmp_path, real_engine
):
    forward, backward = (
        view(
            name,
            "pairs",
            "candidate == anchor",
            slots=EVERYONE,
            current_slot=current,
            related_slot=1 - current,
        )
        for name, current in (("forward", 0), ("backward", 1))
    )
    case = {
        "name": "probe",
        "entity_id": "case",
        "period": "2026-01",
        "input": {
            "us:policies/example/composition#relation.pairs": [
                {"tuple": ["case", "case"]}
            ]
        },
        "output": {"result": 2},
    }
    root, module, artifact = compose(
        tmp_path,
        real_engine,
        program(rule("unused", "Person", "1")),
        [
            PAIRS,
            forward,
            backward,
            rule("result", "Person", "len(forward) + len(backward)"),
        ],
        case,
    )
    failures = run_composed("validator", tmp_path, real_engine, root, module, artifact)
    assert failures == []
