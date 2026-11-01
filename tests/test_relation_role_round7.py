"""Pin the seventh-round alias, inert declaration and companion findings."""

from __future__ import annotations

import json

import pytest
import yaml

from axiom_encode.companion_entities import declared_input_entities
from axiom_encode.formula_references import (
    member_alias_key,
    member_alias_reads,
    rule_lowers_formulas,
)
from axiom_encode.harness import evals
from axiom_encode.harness import validator_pipeline as validation
from tests.test_companion_declared_entities import _compile
from tests.test_relation_role_review_fixes import relation
from tests.test_relation_role_round6 import (
    PAIRS,
    PROVIDER,
    TARGET,
    UNCOVERED,
    VIEW,
    data_relation,
    rule,
    run,
    write_module,
)

EXECUTORS = ["cli", "validator"]


def view(name, source, formula, *, slots=("Person", "Household"), **config):
    return {
        "name": name,
        "kind": "derived_relation",
        "derived_relation": {
            "arity": 2,
            "slot_entities": list(slots),
            "current_slot": 0,
            "related_slot": 1,
            "source_relation": source,
            **config,
        },
        "versions": [{"effective_from": "2026-01-01", "formula": formula}],
    }


def program(*rules, **fields):
    return {"format": "rulespec/v1", "rules": list(rules), **fields}


def reach(tmp_path, payload, **siblings):
    """Read coverage issues and both terminal inventories of one module."""
    root, module = write_module(tmp_path, payload, **siblings)
    return (
        validation.find_missing_derived_companion_output_issues(
            module.read_text(),
            [{"output": {f"{TARGET}#result": 1}}],
            rules_file=module,
            policy_repo_path=root / "us",
        ),
        validation._rulespec_terminal_executable_names_from_payload(payload),
        evals._context_file_terminal_exports(str(module)),
    )


def outer_reads(payload):
    """Report the input defaults a relation reaches through a scalar helper."""
    return [
        issue
        for issue in validation.find_relation_role_issues(yaml.safe_dump(payload))
        if "derived relation `outer`" in issue
    ]


def outer(*rules, default="hidden"):
    return program(
        PAIRS,
        relation("outer", "candidate.count > 0"),
        *rules,
        inputs=[
            {"name": default, "entity": "Person", "dtype": "Integer", "default": 0}
        ],
    )


def aliased(alias="pairs", read="pairs"):
    """Spell a member alias like a local relation, or give it its own name."""
    return program(
        data_relation("pairs", "Person", "Household"),
        view(
            "uncovered",
            "pairs",
            "candidate.hidden",
            current_slot=1,
            related_slot=0,
            entity="SnapUnit",
            member_relation=alias,
        ),
        rule("result", "SnapUnit", f"len({read})"),
    )


def test_a_local_relation_takes_a_read_before_a_member_alias(tmp_path):
    issues, terminal, context = reach(tmp_path, aliased())
    assert UNCOVERED in issues
    assert terminal == context == ["uncovered", "result"]


def test_a_member_alias_still_reaches_its_relation(tmp_path):
    issues, terminal, context = reach(tmp_path, aliased("members", "members"))
    assert issues == []
    assert terminal == context == ["result"]


def test_the_engine_gives_a_local_relation_the_read(tmp_path, real_engine):
    root, module = write_module(tmp_path, aliased())
    compiled = _compile(
        real_engine.binary,
        real_engine.root,
        root,
        module,
        tmp_path / "compiled.json",
    )
    total = json.dumps(
        [item for item in compiled["program"]["derived"] if item["name"] == "result"]
    )
    assert f"{TARGET}#relation.pairs" in total
    assert "uncovered" not in total


@pytest.mark.parametrize("executor", EXECUTORS)
def test_a_shadowed_alias_reads_no_predicate(executor, tmp_path, real_engine):
    case = {
        "name": "only-the-data-relation",
        "period": "2026-01",
        "entity_id": "household",
        "input": {
            f"{TARGET}#relation.pairs": [{"tuple": ["person", "household"]}],
        },
        "output": {f"{TARGET}#result": 1},
    }
    root, module = write_module(tmp_path, aliased(), [case])
    assert run(executor, tmp_path, real_engine, root, module) == []


def test_a_local_relation_keeps_a_helper_from_an_alias_read():
    filtered = view(
        "filtered",
        "pairs",
        "candidate.hidden > 0",
        slots=("Person", "Person"),
        entity="Person",
        member_relation="members",
    )
    count = rule("count", "Person", "len(members)")
    assert outer_reads(outer(PAIRS, filtered, count))
    local = relation("members", "true")
    assert outer_reads(outer(filtered, count, local)) == []


def test_a_shared_short_name_keeps_a_local_root(tmp_path, real_engine):
    provider = program(data_relation("view", "Person", "Person"))
    payload = program(
        data_relation("pairs", "Person", "Person"),
        relation("view", "true"),
        rule("result", "Person", "len(view)"),
        imports=[PROVIDER],
    )
    root, module = write_module(tmp_path, payload, provider=provider)
    compiled = _compile(
        real_engine.binary,
        real_engine.root,
        root,
        module,
        tmp_path / "compiled.json",
    )
    assert (
        validation.find_relation_role_issues(
            module.read_text(),
            rules_file=module,
            policy_repo_path=root / "us",
            rulespec_dependency_roots=[root],
            compiled_payload=compiled,
        )
        == []
    )


def inert_formula(formula):
    real = data_relation("real", "Person", "Household")
    real["versions"] = [{"effective_from": "2026-01-01", "formula": formula}]
    return real


@pytest.mark.parametrize(
    "kind,lowers",
    [
        ("data_relation", False),
        ("derived_relation", True),
        ("derived", True),
        ("parameter", True),
        (None, True),
    ],
)
def test_only_a_data_relation_leaves_its_formulas_unread(kind, lowers):
    assert rule_lowers_formulas({"kind": kind}) is lowers


def test_rule_lowers_formulas_needs_a_rule():
    assert rule_lowers_formulas(None) is False


def test_an_inert_formula_covers_no_relation(tmp_path):
    payload = program(
        inert_formula("len(uncovered)"),
        view("uncovered", "real", "candidate.flag"),
        rule("result", "Person", "len(real)"),
    )
    issues, terminal, context = reach(tmp_path, payload)
    assert UNCOVERED in issues
    assert terminal == context == ["uncovered", "result"]


def test_a_lowered_formula_still_covers_its_relation(tmp_path):
    payload = program(
        data_relation("real", "Person", "Household"),
        view("uncovered", "real", "candidate.flag"),
        rule("result", "Person", "len(uncovered)"),
    )
    issues, terminal, context = reach(tmp_path, payload)
    assert issues == []
    assert terminal == context == ["result"]


def test_the_engine_compiles_an_inert_formula_away(tmp_path, real_engine):
    payload = program(
        inert_formula("len(uncovered)"),
        view("uncovered", "real", "candidate.flag"),
        rule("result", "Person", "len(real)"),
    )
    root, module = write_module(tmp_path, payload)
    compiled = _compile(
        real_engine.binary,
        real_engine.root,
        root,
        module,
        tmp_path / "compiled.json",
    )
    assert {item["name"] for item in compiled["program"]["derived"]} == {"result"}


def test_a_module_without_a_derived_relation_keeps_its_lexical_terminals():
    payload = program(
        inert_formula("helper"),
        rule("helper", "Person", "1"),
        rule("result", "Person", "2"),
    )
    assert validation._rulespec_terminal_executable_names_from_payload(payload) == [
        "result"
    ]


def test_an_inert_formula_earns_no_explicit_relation_import():
    payload = program(
        inert_formula("len(view)"), rule("result", "Person", "1"), imports=[VIEW]
    )
    assert len(validation.find_unused_import_issues(yaml.safe_dump(payload))) == 1
    payload["rules"][1] = rule("result", "Person", "len(view)")
    assert validation.find_unused_import_issues(yaml.safe_dump(payload)) == []


def test_an_inert_formula_reads_no_relation_spelled_input():
    payload = program(inert_formula("real + 1"), rule("result", "Person", "1"))
    assert validation._rulespec_relation_value_input_names(payload) == set()
    payload["rules"][1] = rule("result", "Person", "real + 1")
    assert validation._rulespec_relation_value_input_names(payload) == {"real"}


def test_an_inert_formula_reaches_no_input_default():
    assert (
        outer_reads(
            outer(
                inert_formula("hidden"),
                rule("count", "Person", "len(real)"),
            )
        )
        == []
    )
    lowered = view("real", "pairs", "candidate.hidden > 0", slots=("Person", "Person"))
    assert outer_reads(outer(lowered, rule("count", "Person", "len(real)")))


@pytest.mark.parametrize(
    "candidate,key",
    [
        (
            {
                "kind": "derived_relation",
                "derived_relation": {"entity": "Person", "member_relation": "members"},
            },
            ("members", "Person"),
        ),
        (
            {
                "kind": "parameter",
                "derived_relation": {"entity": "Person", "member_relation": "members"},
            },
            None,
        ),
        (
            {
                "kind": "derived",
                "derived_relation": {"entity": "Person", "member_relation": "members"},
            },
            None,
        ),
        (
            {"kind": "derived_relation", "derived_relation": {"entity": "Person"}},
            None,
        ),
        (
            {
                "kind": "derived_relation",
                "derived_relation": {"member_relation": "members"},
            },
            None,
        ),
        ({"kind": "derived_relation", "derived_relation": 1}, None),
        (None, None),
    ],
)
def test_only_a_derived_relation_defines_a_member_alias(candidate, key):
    assert member_alias_key(candidate) == key


def test_an_alias_read_needs_the_rules_own_entity():
    aliases = {("members", "Person"): {"selected"}}
    person = {"kind": "derived", "entity": "Person"}
    household = {"kind": "derived", "entity": "Household"}
    assert member_alias_reads({"members"}, aliases, person, ()) == {"selected"}
    assert member_alias_reads({"members"}, aliases, household, ()) == set()
    assert member_alias_reads({"members"}, aliases, person, {"members"}) == set()


def test_an_inert_alias_block_reaches_nothing(tmp_path):
    decoy = rule("decoy", "Person", "len(uncovered)")
    decoy.update(
        kind="parameter",
        derived_relation={"entity": "Person", "member_relation": "real"},
    )
    payload = program(
        data_relation("real", "Person", "Household"),
        view("uncovered", "real", "candidate.flag"),
        decoy,
        rule("result", "Person", "len(real)"),
    )
    issues, _terminal, _context = reach(tmp_path, payload)
    assert UNCOVERED in issues


def proof(formula, *, kind="derived", **selector):
    """Attach one proof import atom to a rule reading `formula`."""
    reader = (
        relation("result", formula)
        if kind == "derived_relation"
        else rule("result", "Person", formula)
    )
    reader["metadata"] = {
        "proof": {"atoms": [{"import": {"target": VIEW, **selector}}]}
    }
    return yaml.safe_dump(program(PAIRS, reader, imports=[VIEW]))


@pytest.mark.parametrize(
    "formula,selector,kind,certified",
    [
        ("view", {"output": VIEW}, "derived", False),
        ("view", {}, "derived", False),
        ("view", {"output": "view"}, "derived", True),
        ("len(view)", {"output": VIEW}, "derived", True),
        ("len(view)", {}, "derived", False),
        ("len(view)", {"output": "view"}, "derived", True),
        ("len(other)", {"output": VIEW}, "derived", False),
        ("foo", {"output": f"{PROVIDER}#foo"}, "derived", False),
        ("view", {"output": VIEW}, "derived_relation", True),
        ("view(anchor, candidate)", {"output": VIEW}, "derived_relation", True),
        ("view +", {"output": VIEW}, "derived", False),
        ("view +", {"output": "view"}, "derived", True),
    ],
)
def test_a_qualified_proof_selector_needs_a_relation_read(
    formula, selector, kind, certified
):
    issues = validation.find_proof_import_reference_issues(
        proof(formula, kind=kind, **selector)
    )
    assert (issues == []) is certified


def test_a_qualified_proof_selector_counts_a_relation_source():
    reader = relation("result", "true")
    reader["derived_relation"]["source_relation"] = VIEW
    reader["metadata"] = {
        "proof": {"atoms": [{"import": {"target": VIEW, "output": VIEW}}]}
    }
    payload = program(PAIRS, reader, imports=[VIEW])
    assert validation.find_proof_import_reference_issues(yaml.safe_dump(payload)) == []


def untyped_compact():
    fact = "fact"
    selected = view(
        "selected_people",
        "people",
        f"candidate.{fact} == 7",
        slots=("Household", "Person"),
    )
    extra = {
        "name": "person_of_extra",
        "kind": "data_relation",
        "data_relation": {"arity": 2},
    }
    return program(
        data_relation("people", "Household", "Person"),
        extra,
        selected,
        rule("result", "Household", "len(selected_people)"),
        inputs=[{"name": fact, "entity": "Person", "dtype": "Integer"}],
    )


@pytest.mark.parametrize("executor", EXECUTORS)
def test_runners_agree_on_an_untyped_compact_member(executor, tmp_path, real_engine):
    case = {
        "name": "untyped",
        "period": "2026-01",
        "input": {
            f"{TARGET}#relation.people": [{"tuple": ["case", "bob"]}],
            f"{TARGET}#relation.person_of_extra": [
                {"id": "bob", f"{TARGET}#input.fact": 7}
            ],
        },
        "output": {f"{TARGET}#result": 1},
    }
    root, module = write_module(tmp_path, untyped_compact(), [case])
    assert run(executor, tmp_path, real_engine, root, module) == []


def inert_inputs_program(inputs):
    return program(
        data_relation("pairs", "Person", "Person"),
        relation("members", "candidate.fact == anchor.fact"),
        rule("result", "Person", "len(members)"),
        inputs=inputs,
    )


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("inputs", [5, True, 1.5])
def test_scalar_inputs_metadata_leaves_a_role_flat_fact_alone(
    executor, inputs, tmp_path, real_engine
):
    case = {
        "name": "flat",
        "period": "2026-01",
        "input": {
            f"{TARGET}#relation.pairs": [{"tuple": ["case", "bob"]}],
            f"{TARGET}#input.fact": "x",
        },
        "tables": {"Person": [{"id": "bob", f"{TARGET}#input.fact": "x"}]},
        "output": {f"{TARGET}#result": 1},
    }
    root, module = write_module(tmp_path, inert_inputs_program(inputs), [case])
    assert run(executor, tmp_path, real_engine, root, module) == []


@pytest.mark.parametrize("inputs", [5, True])
def test_only_role_programs_tolerate_scalar_inputs_metadata(tmp_path, inputs):
    module = tmp_path / "probe.yaml"
    module.write_text(yaml.safe_dump({"inputs": inputs}))

    def entities(**options):
        return declared_input_entities(
            module,
            ["fact"],
            resolve_reference=lambda _key: None,
            resolve_import=lambda _target, _current: None,
            **options,
        )

    assert entities(relation_roles=True) == {}
    with pytest.raises(TypeError):
        entities()
