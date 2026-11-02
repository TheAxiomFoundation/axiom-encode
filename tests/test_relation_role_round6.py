"""Pin the sixth-round source, alias and companion identity findings."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from axiom_encode import cli
from axiom_encode.companion_entities import (
    cross_entity_input_findings,
    foreign_entity_ids,
)
from axiom_encode.companion_relations import compact_relation_row
from axiom_encode.formula_references import (
    local_relation_sources,
    member_alias_entity,
    rule_relation_sources,
)
from axiom_encode.harness import evals, same_case_execution
from axiom_encode.harness import validator_pipeline as validation
from tests.test_companion_declared_entities import _companions, _compile
from tests.test_relation_role_flat_dependencies import (
    CITATION,
    EXCERPT,
    formula_rule,
)
from tests.test_relation_role_review_fixes import relation, scalar_payload

LABEL = "# Synthetic encoder test input; not a RuleSpec module.\n"
TARGET = "us:policies/example/probe"
PROVIDER = "us:policies/example/provider"
EXTERNAL = f"{PROVIDER}#relation.uncovered"
OWN = f"{TARGET}#relation.uncovered"
UNIMPORTED = "us:policies/example/deep#relation.uncovered"
VIEW = f"{PROVIDER}#relation.view"
PAIRS = {
    "name": "pairs",
    "kind": "data_relation",
    "data_relation": {"arity": 2, "arguments": ["Person", "Person"]},
}


def write_module(tmp_path, payload, cases=(), **siblings):
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/probe.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(LABEL + yaml.safe_dump(payload))
    module.with_suffix(".test.yaml").write_text(LABEL + yaml.safe_dump(list(cases)))
    for name, sibling in siblings.items():
        (module.parent / f"{name}.yaml").write_text(LABEL + yaml.safe_dump(sibling))
    return root, module


def run(executor, tmp_path, real_engine, root, module):
    return _companions(
        executor,
        binary=real_engine.binary,
        engine_root=real_engine.root,
        root=root,
        module=module,
        artifact=tmp_path / "compiled.json",
    )


def data_relation(name, *arguments):
    return {
        "name": name,
        "kind": "data_relation",
        "data_relation": {"arity": len(arguments), "arguments": list(arguments)},
    }


def rule(name, entity, formula):
    return {
        "name": name,
        "kind": "derived",
        "entity": entity,
        "dtype": "Integer",
        "versions": [{"effective_from": "2026-01-01", "formula": formula}],
    }


def relation_sources(compiled):
    return {
        (item.get("derivation") or {}).get("source_relation")
        for item in compiled["program"]["relations"]
    }


def source_rule(source):
    view = relation("selected", "true")
    view["derived_relation"]["source_relation"] = source
    return view


def source_beside_local(source, *imports):
    """Read `source` from a module that also holds a local `uncovered`."""
    local = relation("uncovered", "candidate.hidden")
    local["versions"] = [
        {
            "effective_from": "2025-01-01",
            "effective_to": "2025-12-31",
            "formula": "candidate.hidden",
        }
    ]
    payload = scalar_payload("len(selected)")
    payload["imports"] = list(imports)
    payload["rules"][:0] = [PAIRS, local, source_rule(source)]
    return payload


def alias_beside_import(predicate=True):
    """Give a local alias an imported relation's name, then read that relation."""
    aliased = relation("local_view", predicate)
    aliased["derived_relation"].update(entity="SnapUnit", member_relation="view")
    selected = relation("selected", "view(anchor, candidate)")
    selected["entity"] = "SnapUnit"
    payload = scalar_payload("len(selected)")
    payload["imports"] = [VIEW]
    payload["rules"][:0] = [PAIRS, aliased, selected]
    return payload


UNCOVERED = (
    "Derived relation missing companion output coverage: "
    f"`{TARGET}#uncovered` is not reached by an asserted scalar output in the "
    "companion `.test.yaml` file."
)


def coverage_issues(tmp_path, payload):
    root, module = write_module(tmp_path, payload)
    content = module.read_text()
    return validation.find_missing_derived_companion_output_issues(
        content,
        [{"output": {f"{TARGET}#result": 1}}],
        rules_file=module,
        policy_repo_path=root / "us",
    )


@pytest.mark.parametrize(
    "occupied,kinds,foreign",
    [
        ({"a": {"Person"}, "b": {"Household"}}, {"OtherUnit": "Person"}, ["b"]),
        ({"a": {"OtherUnit"}}, {"OtherUnit": "Person"}, []),
        ({"a": {"Person", "Household"}}, {"OtherUnit": "Person"}, ["a"]),
        ({"a": {"Person"}}, {}, ["a"]),
        ({"a": {"OtherUnit"}}, {}, []),
        ({"a": {"Person"}}, None, ["a"]),
    ],
    ids=["kinds", "alias", "mixed", "legacy-other", "legacy-same", "no-map"],
)
def test_foreign_ids_compare_kinds(occupied, kinds, foreign):
    assert foreign_entity_ids(occupied, "OtherUnit", entity_kinds=kinds) == foreign


@pytest.mark.parametrize("executor", ["cli", "validator"])
def test_a_flat_fact_joins_a_tuple_id_of_its_own_kind(tmp_path, real_engine, executor):
    other = relation("other", "candidate.fact == 7")
    other["derived_relation"].update(
        slot_entities=["Household", "Person"],
        current_slot=1,
        related_slot=0,
        source_relation="people",
        entity="OtherUnit",
        member_relation="other_members",
    )
    payload = {
        "format": "rulespec/v1",
        "inputs": [{"name": "fact", "entity": "OtherUnit", "dtype": "Integer"}],
        "rules": [
            data_relation("people", "Household", "Person"),
            other,
            rule("result", "Household", "sum(people.fact)"),
        ],
    }
    case = {
        "name": "probe",
        "period": "2026-01",
        "input": {
            f"{TARGET}#relation.people": [{"tuple": ["case", "case:entity:OtherUnit"]}],
            f"{TARGET}#input.fact": 7,
        },
        "output": {f"{TARGET}#result": 7},
    }
    root, module = write_module(tmp_path, payload, [case])
    assert run(executor, tmp_path, real_engine, root, module) == []


def ternary_program():
    members = relation("members", "triples(anchor, candidate, candidate)")
    return {
        "format": "rulespec/v1",
        "rules": [
            PAIRS,
            data_relation("triples", "Person", "Person", "Person"),
            members,
            rule("result", "Person", "len(members)"),
        ],
    }


@pytest.mark.parametrize("executor", ["cli", "validator"])
def test_tuple_rows_reach_a_ternary_data_relation(tmp_path, real_engine, executor):
    case = {
        "name": "probe",
        "period": "2026-01",
        "input": {
            f"{TARGET}#relation.pairs": [{"tuple": ["case", "bob"]}],
            f"{TARGET}#relation.triples": [{"tuple": ["case", "bob", "bob"]}],
        },
        "output": {f"{TARGET}#result": 1},
    }
    root, module = write_module(tmp_path, ternary_program(), [case])
    assert run(executor, tmp_path, real_engine, root, module) == []


@pytest.mark.parametrize("executor", ["cli", "validator"])
def test_compact_rows_stay_with_binary_relations(tmp_path, real_engine, executor):
    case = {
        "name": "probe",
        "period": "2026-01",
        "input": {
            f"{TARGET}#relation.pairs": [{"tuple": ["case", "bob"]}],
            f"{TARGET}#relation.triples": [{"id": "bob"}],
        },
        "output": {f"{TARGET}#result": 1},
    }
    root, module = write_module(tmp_path, ternary_program(), [case])
    issues = "\n".join(run(executor, tmp_path, real_engine, root, module))
    assert "arity" in issues or "slot" in issues
    assert "neither of which matches" not in issues


def test_declared_slots_keep_every_arity():
    relations = [
        {"name": "one", "slot_entities": ["Person"]},
        {"name": "two", "slot_entities": ["Person", "Household"]},
        {"name": "three", "slot_entities": ["Person", "Person", "Person"]},
        {"name": "none", "slot_entities": []},
        {"name": "partial", "slot_entities": ["Person", ""]},
    ]
    assert validation._rulespec_declared_relation_slots(
        {"program": {"relations": relations}}
    ) == {
        "one": ("Person",),
        "two": ("Person", "Household"),
        "three": ("Person", "Person", "Person"),
    }


@pytest.mark.parametrize(
    "slots,connected",
    [
        (("Person", "Household"), True),
        (("Household", "Person"), True),
        (("Person", "Person", "Household"), False),
        (("Person", "TaxUnit"), False),
        (("TaxUnit", "Household"), False),
    ],
    ids=["forward", "reverse", "ternary", "input-only", "output-only"],
)
def test_only_a_binary_relation_connects_entities_in_a_missing_fact_finding(
    slots, connected
):
    findings = cross_entity_input_findings(
        "missing input fact",
        {"fact": 7},
        {"fact": "Person"},
        {"result": "Household"},
        {"link": slots},
    )
    assert (not findings) is connected


@pytest.mark.parametrize(
    "row,entity_id,explicit",
    [
        ({"id": "alice"}, "alice", True),
        ({"entity_id": "alice"}, "alice", True),
        ({"id": 7}, "7", True),
        ({"id": "alice", "entity_id": "bob"}, "alice", True),
        ({"id": None, "entity_id": "bob"}, "bob", True),
        ({"id": 0, "entity_id": "bob"}, "bob", True),
        ({"id": 0}, "generated", False),
        ({"id": ""}, "generated", False),
        ({"id": False}, "generated", False),
        ({"id": None, "entity_id": ""}, "generated", False),
        ({}, "generated", False),
    ],
)
def test_the_first_truthy_id_names_a_compact_member(row, entity_id, explicit):
    assert compact_relation_row({**row, "fact": 1}, fallback_id="generated") == (
        entity_id,
        explicit,
        {"fact": 1},
    )


@pytest.mark.parametrize(
    "executor,failures",
    [("cli", ["unsupported scalar test value None"]), ("validator", [])],
)
def test_a_legacy_null_id_keeps_each_runners_reading(
    tmp_path, real_engine, executor, failures
):
    payload = {
        "format": "rulespec/v1",
        "rules": [
            data_relation("people", "Household", "Person"),
            rule("result", "Household", "sum(people.fact)"),
        ],
    }
    case = {
        "name": "probe",
        "period": "2026-01",
        "entity_id": "root",
        "input": {f"{TARGET}#relation.people": [{"id": None, "entity_id": "bob"}]},
        "tables": {"Person": [{"id": "bob", f"{TARGET}#input.fact": 7}]},
        "output": {f"{TARGET}#result": 7},
    }
    root, module = write_module(tmp_path, payload, [case])
    assert run(executor, tmp_path, real_engine, root, module) == failures


@pytest.mark.parametrize(
    "source,imports,module,local",
    [
        ("view", [], None, {"view"}),
        (EXTERNAL, [EXTERNAL], None, set()),
        (EXTERNAL, [f" {PROVIDER} "], None, set()),
        (EXTERNAL, [f"{PROVIDER}#other"], None, set()),
        (OWN, [EXTERNAL], None, {"uncovered"}),
        (OWN, "not-a-list", None, {"uncovered"}),
        (OWN, [7], None, {"uncovered"}),
        (OWN, [], TARGET, {"uncovered"}),
        (UNIMPORTED, [], TARGET, set()),
        ("view", [], TARGET, {"view"}),
    ],
    ids=[
        "bare",
        "imported",
        "whole-module-import",
        "another-symbol",
        "own-beside-import",
        "imports-no-list",
        "import-no-string",
        "named-own",
        "named-other",
        "named-bare",
    ],
)
def test_a_full_target_reads_the_module_it_names(source, imports, module, local):
    assert local_relation_sources(source_rule(source), imports, module=module) == local


def test_a_source_path_counts_as_a_source():
    view = source_rule("pairs")
    del view["derived_relation"]["source_relation"]
    view["derived_relation"]["source_path"] = [
        {"relation": "pairs", "from_slot": 0, "to_slot": 1},
        {"relation": EXTERNAL, "from_slot": 1, "to_slot": 0},
    ]
    assert rule_relation_sources(view) == {"pairs", EXTERNAL}
    assert local_relation_sources(view, [EXTERNAL]) == {"pairs"}


@pytest.mark.parametrize("kind", ["derived", "parameter", "data_relation", None])
def test_only_a_derived_relation_has_sources(kind):
    inert = {"kind": kind, "derived_relation": {"source_relation": "view"}}
    assert rule_relation_sources(inert) == set()
    assert local_relation_sources(inert) == set()


@pytest.mark.parametrize("value", [None, "view", 7, ["view"]])
def test_a_rule_that_is_no_mapping_reads_nothing(value):
    assert rule_relation_sources(value) == set()
    assert member_alias_entity(value) is None


def test_full_target_sources_compile_by_the_module_they_name(tmp_path, real_engine):
    """Own and unimported targets compile, so neither one proves ownership."""
    pairs = {
        **PAIRS,
        "name": "uncovered",
        "data_relation": {
            **PAIRS["data_relation"],
            "slot_entities": ["Person", "Person"],
        },
    }
    view = source_rule(UNIMPORTED)
    view["name"] = "view"
    payload = scalar_payload("len(selected)")
    payload["imports"] = [VIEW]
    payload["rules"].insert(0, source_rule(UNIMPORTED))
    root, module = write_module(
        tmp_path,
        payload,
        deep={"format": "rulespec/v1", "rules": [pairs]},
        provider={"format": "rulespec/v1", "imports": [UNIMPORTED], "rules": [view]},
    )
    compiled = _compile(
        real_engine.binary, real_engine.root, root, module, tmp_path / "a.json"
    )
    assert UNIMPORTED in relation_sources(compiled)

    payload = scalar_payload("len(selected)")
    payload["rules"][:0] = [pairs, source_rule(OWN)]
    module.write_text(LABEL + yaml.safe_dump(payload))
    compiled = _compile(
        real_engine.binary, real_engine.root, root, module, tmp_path / "b.json"
    )
    assert OWN in relation_sources(compiled)


@pytest.mark.parametrize(
    "source,imports,covered",
    [
        (EXTERNAL, [EXTERNAL], False),
        (UNIMPORTED, [], False),
        ("uncovered", [], True),
        (OWN, [], True),
    ],
    ids=["imported", "unimported", "bare", "own"],
)
def test_coverage_follows_a_source_to_the_module_it_names(
    tmp_path, source, imports, covered
):
    issues = coverage_issues(tmp_path, source_beside_local(source, *imports))
    assert (UNCOVERED in issues) is not covered


def test_an_inert_source_covers_nothing(tmp_path):
    payload = source_beside_local("pairs")
    payload["rules"][-1]["derived_relation"] = {"source_relation": "uncovered"}
    assert UNCOVERED in coverage_issues(tmp_path, payload)


@pytest.mark.parametrize(
    "source,imports,gaps",
    [
        (EXTERNAL, [EXTERNAL], 0),
        ("uncovered", [], 1),
        (OWN, [], 1),
    ],
    ids=["imported", "bare", "own"],
)
def test_temporal_coverage_follows_a_source_to_its_module(source, imports, gaps):
    content = yaml.safe_dump(source_beside_local(source, *imports))
    issues = validation.find_local_dependency_temporal_coverage_issues(content)
    assert len(issues) == gaps


def test_an_inert_source_adds_no_temporal_dependency():
    payload = source_beside_local("pairs")
    payload["rules"][-1]["derived_relation"] = {"source_relation": "uncovered"}
    content = yaml.safe_dump(payload)
    assert validation.find_local_dependency_temporal_coverage_issues(content) == []


@pytest.mark.parametrize(
    "source,imports,terminal",
    [
        (EXTERNAL, [EXTERNAL], True),
        ("uncovered", [], False),
        (OWN, [], False),
    ],
    ids=["imported", "bare", "own"],
)
def test_terminal_exports_follow_a_source_to_its_module(
    tmp_path, source, imports, terminal
):
    payload = source_beside_local(source, *imports)
    _, module = write_module(tmp_path, payload)
    assert (
        "uncovered"
        in validation._rulespec_terminal_executable_names_from_payload(payload)
    ) is terminal
    assert (
        "uncovered" in evals._context_file_terminal_exports(str(module))
    ) is terminal


def test_an_inert_source_keeps_a_rule_terminal(tmp_path):
    payload = source_beside_local("pairs")
    payload["rules"][-1]["derived_relation"] = {"source_relation": "uncovered"}
    _, module = write_module(tmp_path, payload)
    assert "uncovered" in validation._rulespec_terminal_executable_names_from_payload(
        payload
    )
    assert "uncovered" in evals._context_file_terminal_exports(str(module))


def assignment_issues(payload):
    """Check a proof-required module with one case that assigns `hidden`."""
    payload["module"] = {"proof_validation": {"required": True}}
    output = {f"{TARGET}#result": 1}
    cases = [
        {"name": "set", "input": {f"{TARGET}#input.hidden": 1}, "output": output},
        {"name": "unset", "input": {}, "output": output},
    ]
    return validation.find_test_input_assignment_issues(yaml.safe_dump(payload), cases)


@pytest.mark.parametrize(
    "source,imports,missing",
    [
        (EXTERNAL, [EXTERNAL], False),
        ("uncovered", [], True),
        (OWN, [], True),
    ],
    ids=["imported", "bare", "own"],
)
def test_input_assignment_follows_a_source_to_its_module(source, imports, missing):
    issues = assignment_issues(source_beside_local(source, *imports))
    assert bool(issues) is missing


def test_an_inert_source_is_no_input_read():
    payload = scalar_payload("1")
    payload["rules"][0]["derived_relation"] = {"source_relation": "hidden"}
    assert assignment_issues(payload) == []


def test_an_inert_source_keeps_a_helpers_input_read():
    helper = rule("helper", "Person", "fact")
    helper["derived_relation"] = {"source_relation": "fact"}
    payload = scalar_payload("len(selected)")
    payload["inputs"] = [
        {"name": "fact", "entity": "Person", "dtype": "Integer", "default": 0}
    ]
    payload["rules"][:0] = [
        PAIRS,
        helper,
        relation("selected", "candidate.helper > 0"),
    ]
    issues = validation.find_relation_role_issues(yaml.safe_dump(payload))
    assert any("`fact` read by derived relation `selected`" in i for i in issues)


def test_an_alias_gives_a_predicate_no_input_read():
    payload = alias_beside_import("candidate.fact > 0")
    payload["inputs"] = [
        {"name": "fact", "entity": "Person", "dtype": "Integer", "default": 0}
    ]
    issues = validation.find_relation_role_issues(yaml.safe_dump(payload))
    assert any("relation `local_view`" in i for i in issues)
    assert not any("relation `selected`" in i for i in issues)


def test_only_a_derived_relation_gives_a_generated_rule_a_source():
    formula = {"versions": [{"effective_from": "2026-01-01", "formula": "1"}]}
    block = {"derived_relation": {"source_relation": EXTERNAL}}
    assert cli._generated_rule_formula_identifiers(
        {"kind": "derived_relation", **block, **formula}
    ) == {"uncovered"}
    assert not cli._generated_rule_formula_identifiers(
        {"kind": "derived", **block, **formula}
    )


@pytest.mark.parametrize(
    "kind,consumed", [("derived_relation", True), ("derived", False)]
)
def test_only_a_derived_relation_edge_reaches_a_consumer(kind, consumed):
    step = formula_rule("step", "1", kind=kind)
    step["derived_relation"] = {"source_relation": "reported"}
    payload = {"rules": [step, formula_rule("count", "step", kind="derived")]}
    pipe = SimpleNamespace(
        _source_texts_for_rulespec_content=lambda _: {CITATION: EXCERPT},
        _proof_source_texts_for_rulespec_content=lambda *a, **k: {CITATION: EXCERPT},
    )
    assert (
        same_case_execution._has_consumer(pipe, "", payload, {"reported"}) is consumed
    )


def test_an_inert_source_proves_no_import():
    payload = scalar_payload("1")
    payload["imports"] = [VIEW]
    payload["rules"][0]["metadata"] = {
        "proof": {
            "atoms": [{"import": {"target": VIEW, "output": "view"}}],
        }
    }
    payload["rules"][0]["derived_relation"] = {"source_relation": VIEW}
    assert (
        len(validation.find_proof_import_reference_issues(yaml.safe_dump(payload))) == 1
    )


@pytest.mark.parametrize(
    "rule_,entity",
    [
        ({"kind": "derived", "entity": "SnapUnit"}, "SnapUnit"),
        ({"kind": "parameter", "entity": "SnapUnit"}, "SnapUnit"),
        ({"kind": "derived"}, ""),
        ({"kind": "derived_relation", "entity": "SnapUnit"}, None),
    ],
    ids=["derived", "parameter", "no-entity", "relation"],
)
def test_member_aliases_stay_out_of_derived_relations(rule_, entity):
    assert member_alias_entity(rule_) == entity


def test_an_alias_leaves_a_relation_predicate_reading_its_import():
    content = yaml.safe_dump(alias_beside_import())
    assert validation.find_unused_import_issues(content) == []
    assert cli._prune_unused_imports(content)[1] == []
    assert evals._prune_unused_imports(content)[1] == []


def test_the_engine_reads_the_import_past_an_alias_in_a_predicate(
    tmp_path, real_engine
):
    provider = {"format": "rulespec/v1", "rules": [{**PAIRS, "name": "view"}]}
    root, module = write_module(tmp_path, alias_beside_import(), provider=provider)
    compiled = _compile(
        real_engine.binary, real_engine.root, root, module, tmp_path / "compiled.json"
    )
    relations = {item["name"]: item for item in compiled["program"]["relations"]}
    predicate = relations[f"{TARGET}#relation.selected"]["derivation"]["predicate"]
    assert VIEW in yaml.safe_dump(predicate)


def test_an_alias_gives_a_predicate_no_coverage_edge(tmp_path):
    payload = alias_beside_import()
    root, module = write_module(tmp_path, payload)
    issues = validation.find_missing_derived_companion_output_issues(
        module.read_text(),
        [{"output": {f"{TARGET}#result": 1}}],
        rules_file=module,
        policy_repo_path=root / "us",
    )
    assert any(f"`{TARGET}#local_view`" in issue for issue in issues)
    assert "local_view" in validation._rulespec_terminal_executable_names_from_payload(
        payload
    )
    assert "local_view" in evals._context_file_terminal_exports(str(module))


def count_parses(monkeypatch):
    parses = []
    loader = validation._rulespec_payload_from_file

    def counted(path):
        parses.append(path)
        return loader(path)

    monkeypatch.setattr(validation, "_rulespec_payload_from_file", counted)
    return parses


def test_a_warm_summary_of_a_data_relation_parses_nothing(tmp_path, monkeypatch):
    payload = scalar_payload("1")
    payload["rules"].insert(0, PAIRS)
    _, module = write_module(tmp_path, payload)
    parses = count_parses(monkeypatch)
    for _ in range(4):
        summary = validation._rulespec_reference_summary(module)
    assert summary.relations == {"pairs"}
    assert parses == []


def test_a_warm_summary_skips_a_nameless_derived_relation(tmp_path, monkeypatch):
    payload = scalar_payload("1")
    payload["rules"].insert(0, {"kind": "derived_relation", "derived_relation": {}})
    _, module = write_module(tmp_path, payload)
    parses = count_parses(monkeypatch)
    for _ in range(4):
        validation._rulespec_reference_summary(module)
    assert parses == []


def test_warm_summaries_of_a_relation_role_module_parse_it_once(tmp_path, monkeypatch):
    payload = scalar_payload("len(selected)")
    payload["rules"][:0] = [PAIRS, relation("selected", "candidate.pairs > 0")]
    _, module = write_module(tmp_path, payload)
    parses = count_parses(monkeypatch)
    for _ in range(4):
        summary = validation._rulespec_reference_summary(module)
    assert summary.relations == {"pairs", "selected"}
    assert len(parses) == 1
