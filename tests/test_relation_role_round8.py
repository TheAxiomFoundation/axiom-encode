"""Pin the eighth-round alias, proof, tuple and metadata findings."""

from __future__ import annotations

import copy
import os
import subprocess

import pytest
import yaml

from axiom_encode.cli import _execute_rulespec_test_file
from axiom_encode.companion_entities import declared_input_entities
from axiom_encode.companion_relations import (
    _relations_read_by,
    executable_relation_directions,
    explicit_relation_tuples,
    explicit_tuple_relation_names,
    is_tuple_row,
    pair_relation_assignment_issues,
    reached_relation_names,
)
from axiom_encode.formula_references import member_alias_reads, member_aliases
from axiom_encode.harness import validator_pipeline as validation
from axiom_encode.rules_engine_compat import run_rulespec_compile
from tests.test_companion_declared_entities import _compile
from tests.test_relation_role_datasets import (
    AGE,
    OUTPUT,
    TIES,
    _legacy_program,
    _program,
    _requests,
)
from tests.test_relation_role_datasets import PAIRS as DATASET_PAIRS
from tests.test_relation_role_round6 import (
    LABEL,
    PAIRS,
    PROVIDER,
    TARGET,
    VIEW,
    data_relation,
    rule,
    run,
    write_module,
)
from tests.test_relation_role_round7 import program, reach, view

EXECUTORS = ["cli", "validator"]
EVERYONE = ("Person", "Person")
PEOPLE = f"{PROVIDER}#relation.people"
FLAG = f"{PROVIDER}#input.flag"
PUBLIC_RESULT = f"{PROVIDER}#public_result"
SHARED = (
    "[relation-member-alias] Derived relations `first`, `uncovered` share member "
    "alias `members` on `SnapUnit`. The engine reads only one of them through "
    "it, so give each relation its own alias."
)


def competing(*order, alias="members", entity="SnapUnit"):
    """Give two filtered relations of `pairs` a member alias and entity."""
    relations = {
        "first": view(
            "first",
            "pairs",
            "true",
            slots=EVERYONE,
            entity="SnapUnit",
            member_relation="members",
        ),
        "uncovered": view(
            "uncovered",
            "pairs",
            "candidate.hidden",
            slots=EVERYONE,
            entity=entity,
            member_relation=alias,
        ),
    }
    return program(
        PAIRS,
        *(relations[name] for name in order),
        rule("result", "SnapUnit", "len(members)"),
    )


def alias_case():
    return {
        "name": "probe",
        "entity_id": "a",
        "period": "2026-01",
        "input": {f"{TARGET}#relation.pairs": [{"tuple": ["a", "b"]}]},
        "output": {f"{TARGET}#result": 1},
    }


def test_member_aliases_keep_the_first_declaring_relation():
    rules = {
        relation["name"]: relation
        for relation in competing("first", "uncovered")["rules"]
    }
    assert member_aliases(rules) == {("members", "SnapUnit"): "first"}
    assert member_alias_reads(
        {"members", "other"},
        member_aliases(rules),
        rules["result"],
        (),
    ) == {"first"}


@pytest.mark.parametrize(
    "order,unread",
    [(("first", "uncovered"), "uncovered"), (("uncovered", "first"), "first")],
)
def test_a_later_relation_takes_no_read_through_a_shared_alias(tmp_path, order, unread):
    issues, terminal, context = reach(tmp_path, competing(*order))
    assert [issue for issue in issues if unread in issue]
    assert terminal == context == [unread, "result"]


@pytest.mark.parametrize("executor", EXECUTORS)
def test_the_engine_reads_only_the_first_relation_of_a_shared_alias(
    tmp_path, real_engine, executor
):
    root, module = write_module(
        tmp_path / "first",
        competing("first", "uncovered"),
        [alias_case()],
    )
    assert run(executor, tmp_path / "first", real_engine, root, module) == []
    root, module = write_module(
        tmp_path / "second",
        competing("uncovered", "first"),
        [alias_case()],
    )
    failures = run(executor, tmp_path / "second", real_engine, root, module)
    assert any("missing input `hidden`" in failure for failure in failures)


def test_the_role_gate_refuses_a_shared_member_alias():
    issues = validation.find_relation_role_issues(
        yaml.safe_dump(competing("first", "uncovered"))
    )
    assert SHARED in issues


@pytest.mark.parametrize(
    "difference", [{"alias": "other_members"}, {"entity": "OtherUnit"}]
)
def test_the_role_gate_accepts_aliases_that_differ(difference):
    issues = validation.find_relation_role_issues(
        yaml.safe_dump(competing("first", "uncovered", **difference))
    )
    assert not [issue for issue in issues if "[relation-member-alias]" in issue]


def imported_alias_programs():
    provider = program(
        PAIRS,
        view(
            "selected",
            "pairs",
            "true",
            slots=EVERYONE,
            entity="SnapUnit",
            member_relation="members",
        ),
        rule("own", "SnapUnit", "len(members)"),
    )
    consumer = program(
        data_relation("pairs", "Person", "Person"),
        view(
            "uncovered",
            "pairs",
            "candidate.hidden",
            slots=EVERYONE,
            entity="SnapUnit",
            member_relation="members",
        ),
        rule("result", "SnapUnit", "len(members)"),
        imports=[f"{PROVIDER}#own"],
    )
    return provider, consumer


def test_an_alias_shared_with_an_import_sends_the_read_to_the_import(
    tmp_path, real_engine
):
    provider, consumer = imported_alias_programs()
    root, module = write_module(tmp_path, consumer, provider=provider)
    compiled = _compile(
        real_engine.binary,
        real_engine.root,
        root,
        module,
        tmp_path / "compiled.json",
    )
    result = next(
        rule_
        for rule_ in compiled["program"]["derived"]
        if rule_["id"] == f"{TARGET}#result"
    )
    assert result["versions"][0]["expr"]["relation"] == (
        f"{PROVIDER}#relation.selected"
    )


def test_the_role_gate_refuses_an_alias_shared_with_an_import(tmp_path):
    provider, consumer = imported_alias_programs()
    root, module = write_module(tmp_path, consumer, provider=provider)
    issues = validation.find_relation_role_issues(
        module.read_text(),
        rules_file=module,
        policy_repo_path=root / "us",
    )
    assert [
        issue
        for issue in issues
        if issue.startswith("[relation-member-alias]")
        and "`selected`, `uncovered`" in issue
    ]


def selector_proof(
    formula="len(view)",
    *,
    entity="Household",
    imports=(VIEW,),
    local=(),
    atom=None,
):
    reader = rule("result", entity, formula)
    reader["metadata"] = {
        "proof": {"atoms": [{"import": atom or {"target": VIEW, "output": VIEW}}]}
    }
    return yaml.safe_dump(program(*local, reader, imports=list(imports)))


def certified(content):
    return validation.find_proof_import_reference_issues(content) == []


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, True),
        ({"local": [data_relation("view", "Household", "Person")]}, False),
        ({"imports": [f"{PROVIDER}-other#relation.view"]}, False),
        ({"imports": [PROVIDER]}, False),
        ({"imports": []}, False),
        ({"atom": {"target": f"{PROVIDER}#relation.other", "output": VIEW}}, False),
        ({"atom": {"target": PROVIDER, "output": VIEW}}, True),
        ({"atom": {"target": f"{PROVIDER}-other", "output": VIEW}}, False),
    ],
)
def test_a_qualified_proof_needs_a_read_of_its_own_import(options, expected):
    assert certified(selector_proof(**options)) is expected


def test_a_bare_proof_selector_keeps_mains_lexical_reach():
    content = selector_proof(
        local=[data_relation("view", "Household", "Person")],
        atom={"target": VIEW, "output": "view"},
    )
    assert certified(content)


@pytest.mark.parametrize("entity,expected", [("SnapUnit", False), ("Household", True)])
def test_a_member_alias_takes_a_read_from_a_qualified_proof(entity, expected):
    alias = view(
        "selected",
        "pairs",
        "true",
        slots=EVERYONE,
        entity="SnapUnit",
        member_relation="view",
    )
    content = selector_proof(entity=entity, local=[PAIRS, alias])
    assert certified(content) is expected


@pytest.mark.parametrize(
    "source,local,imports,expected",
    [
        (VIEW, [], [], True),
        ("view", [], [VIEW], True),
        ("view", [data_relation("view", "Person", "Person")], [VIEW], False),
        (f"{PROVIDER}-other#relation.view", [], [VIEW], False),
    ],
)
def test_a_relation_source_certifies_only_the_relation_it_names(
    source, local, imports, expected
):
    reader = view("result", source, "true", slots=EVERYONE)
    reader["metadata"] = {
        "proof": {"atoms": [{"import": {"target": VIEW, "output": VIEW}}]}
    }
    content = yaml.safe_dump(program(*local, reader, imports=imports))
    assert certified(content) is expected


@pytest.mark.parametrize(
    "atom,expected",
    [
        ({"target": VIEW, "output": VIEW}, False),
        ({"target": VIEW, "output": "view"}, True),
    ],
)
def test_an_inert_formula_earns_no_qualified_proof(atom, expected):
    carrier = data_relation("pairs", "Household", "Person")
    carrier["versions"] = [{"effective_from": "2026-01-01", "formula": "len(view)"}]
    carrier["metadata"] = {"proof": {"atoms": [{"import": atom}]}}
    content = yaml.safe_dump(program(carrier, imports=[VIEW]))
    assert certified(content) is expected


def test_a_shadowing_relation_leaves_its_import_unread(tmp_path, real_engine):
    provider = program(
        data_relation("view", "Household", "Person"),
        rule("own", "Household", "len(view)"),
    )
    consumer = program(
        data_relation("view", "Household", "Person"),
        rule("result", "Household", "len(view)"),
        imports=[VIEW],
    )
    consumer["rules"][0]["data_relation"]["slot_entities"] = ["Household", "Person"]
    provider["rules"][0]["data_relation"]["slot_entities"] = ["Household", "Person"]
    root, module = write_module(tmp_path, consumer, provider=provider)
    compiled = _compile(
        real_engine.binary,
        real_engine.root,
        root,
        module,
        tmp_path / "compiled.json",
    )
    result = next(
        rule_
        for rule_ in compiled["program"]["derived"]
        if rule_["id"] == f"{TARGET}#result"
    )
    assert result["versions"][0]["expr"]["relation"] == f"{TARGET}#relation.view"


def metadata_program(name, entity):
    return program(
        PAIRS,
        view("members", "pairs", "anchor.nan == 1", slots=EVERYONE),
        rule("result", "Person", "len(members)"),
        inputs=[{"name": name, "entity": entity, "dtype": "Integer"}],
    )


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize(
    "name,entity",
    [
        (float("nan"), "Household"),
        ("nan", float("nan")),
        (True, "Household"),
        (5, "Household"),
    ],
)
def test_only_text_metadata_routes_a_role_fact(
    tmp_path, real_engine, executor, name, entity
):
    case = {
        "name": "probe",
        "period": "2026-01",
        "input": {
            f"{TARGET}#relation.pairs": [{"tuple": ["case", "bob"]}],
            f"{TARGET}#input.nan": 1,
        },
        "output": {f"{TARGET}#result": 1},
    }
    root, module = write_module(tmp_path, metadata_program(name, entity), [case])
    assert run(executor, tmp_path, real_engine, root, module) == []


@pytest.mark.parametrize(
    "item,role,legacy",
    [
        ({"name": float("nan"), "entity": "Household"}, {}, {"nan": "Household"}),
        ({"name": "nan", "entity": float("nan")}, {}, {"nan": "nan"}),
        ({"name": "nan", "entity": "Household"}, {"nan": "Household"}, None),
    ],
)
def test_legacy_metadata_keeps_mains_reading(tmp_path, item, role, legacy):
    module = tmp_path / "probe.yaml"
    module.write_text(yaml.safe_dump({"inputs": [item]}))

    def entities(**options):
        return declared_input_entities(
            module,
            ["nan"],
            resolve_reference=lambda _key: None,
            resolve_import=lambda _target, _current: None,
            **options,
        )

    assert entities(relation_roles=True) == role
    assert entities() == (role if legacy is None else legacy)


@pytest.mark.parametrize(
    "row,role,legacy",
    [
        ({"tuple": 7}, True, False),
        ({"tuple": "text"}, True, False),
        ({"tuple": None}, True, False),
        ({"tuple": ["a", "b"]}, True, True),
        ({"fact": 7}, False, False),
        ("row", False, False),
    ],
)
def test_only_a_list_opens_the_tuple_form_on_a_legacy_row(row, role, legacy):
    assert is_tuple_row(row, relation_roles=True) is role
    assert is_tuple_row(row, relation_roles=False) is legacy


@pytest.mark.parametrize("slots", [("Household", "Person"), None])
def test_a_legacy_scalar_tuple_stays_a_compact_row(slots):
    rows = [{"tuple": 7}]
    assert (
        explicit_relation_tuples(rows, relation="r", slots=slots, relation_roles=False)
        is None
    )
    with pytest.raises(ValueError, match="tuple row"):
        explicit_relation_tuples(rows, relation="r", slots=slots, relation_roles=True)


def test_only_a_role_program_counts_scalar_and_empty_relations_as_explicit():
    case_input = {
        "scalar": [{"tuple": 7}],
        "empty": [],
        "listed": [{"tuple": ["a", "b"]}],
        "fact": 7,
    }
    assert explicit_tuple_relation_names(case_input, relation_roles=True) == {
        "scalar",
        "empty",
        "listed",
    }
    assert explicit_tuple_relation_names(case_input, relation_roles=False) == {"listed"}


def test_a_scalar_tuple_key_stays_an_assigned_input():
    keys = validation._assignment_keys_from_mapping(
        {"relation": [{"tuple": 7}, {"tuple": ["a", "b"]}]}
    )
    assert keys.count("tuple") == 1


def compose(tmp_path, real_engine, provider, root_rules, case):
    """Compile a composition over a provider, as the composer assembles one."""
    root = tmp_path / "rulespec-us"
    folder = root / "us/policies/example"
    folder.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (folder / "provider.yaml").write_text(LABEL + yaml.safe_dump(provider))
    source = LABEL + yaml.safe_dump(
        program(*root_rules, module={"kind": "composition"}, imports=[PROVIDER])
    )
    composed = tmp_path / "composed.yaml"
    composed.write_text(source)
    module = folder / "composition.yaml"
    module.write_text(source)
    test_file = module.with_suffix(".test.yaml")
    test_file.write_text(yaml.safe_dump([case]))
    artifact = tmp_path / "compiled.json"
    compiled = run_rulespec_compile(
        binary=real_engine.binary,
        program=composed,
        rulespec_roots=[root],
        output=artifact,
        cwd=real_engine.root,
        env=os.environ.copy(),
        composed=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    return root, module, artifact


def run_composed(executor, tmp_path, real_engine, root, module, artifact):
    payload = yaml.safe_load(artifact.read_text())
    test_file = module.with_suffix(".test.yaml")
    pipeline = validation.ValidatorPipeline(
        policy_repo_path=root / "us",
        axiom_rules_path=real_engine.root,
        local_corpus_release=None,
        enable_oracles=False,
        enforce_repository_layout=False,
    )
    pipeline._axiom_rules_binary = lambda binary=None: real_engine.binary
    if executor == "cli":
        result = _execute_rulespec_test_file(
            test_file,
            binary=real_engine.binary,
            pipeline=pipeline,
            axiom_rules_path=real_engine.root,
            env=os.environ.copy(),
            rulespec_roots=[root],
            tmp_path=tmp_path,
            compiled_cache={module: (artifact, payload)},
            policy_repo_path=root / "us",
        )
        return [failure["message"] for failure in result["failures"]]
    return pipeline._run_rulespec_test_cases(
        rules_file=module,
        compiled_path=artifact,
        compiled_payload=payload,
        cases=yaml.safe_load(test_file.read_text()),
    )


def tuple_fact_provider():
    return program(
        data_relation("people", "Household", "Person"),
        rule("public_result", "Household", "sum(people.tuple)"),
    )


def test_a_legacy_composed_cli_row_keeps_its_tuple_fact(tmp_path, real_engine):
    case = {
        "name": "legacy-factual-tuple",
        "period": "2026-01",
        "input": {PEOPLE: [{"tuple": 7}]},
        "output": {PUBLIC_RESULT: 7},
    }
    root, module, artifact = compose(
        tmp_path,
        real_engine,
        tuple_fact_provider(),
        [rule("root_helper", "Person", "tuple")],
        case,
    )
    assert run_composed("cli", tmp_path, real_engine, root, module, artifact) == []


def test_a_legacy_composed_pipeline_row_keeps_mains_refusal(tmp_path, real_engine):
    case = {
        "name": "legacy-factual-tuple",
        "period": "2026-01",
        "input": {PEOPLE: [{"tuple": 7}]},
        "output": {PUBLIC_RESULT: 7},
    }
    root, module, artifact = compose(
        tmp_path,
        real_engine,
        tuple_fact_provider(),
        [rule("root_helper", "Person", "tuple")],
        case,
    )
    failures = run_composed("validator", tmp_path, real_engine, root, module, artifact)
    assert failures
    assert not [failure for failure in failures if "tuple row" in str(failure)]


def shadowed_provider():
    selected = view(
        "selected",
        "people",
        "candidate.flag",
        slots=("Household", "Person"),
    )
    return program(
        data_relation("people", "Household", "Person"),
        selected,
        rule("public_result", "Household", "len(selected)"),
        inputs=[{"name": "flag", "entity": "Person", "dtype": "Boolean"}],
    )


def shadowing_root():
    return [
        data_relation("people", "Person", "TaxUnit"),
        rule("person_helper", "Person", "len(people)"),
        rule("tax_unit_helper", "TaxUnit", "1"),
    ]


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize(
    "rows,tables",
    [
        ([{"tuple": ["case", "bob"]}], {"Person": [{"id": "bob", FLAG: True}]}),
        ([{"id": "bob", FLAG: True}], None),
    ],
)
def test_a_qualified_key_stays_out_of_a_typed_short_relation(
    tmp_path, real_engine, executor, rows, tables
):
    case = {
        "name": "qualified-key",
        "entity_id": "case",
        "period": "2026-01",
        "input": {PEOPLE: rows},
        "output": {PUBLIC_RESULT: 1},
    }
    if tables:
        case["tables"] = tables
    root, module, artifact = compose(
        tmp_path, real_engine, shadowed_provider(), shadowing_root(), case
    )
    assert run_composed(executor, tmp_path, real_engine, root, module, artifact) == []


@pytest.mark.parametrize(
    "declared,typed,expected",
    [
        ({PEOPLE, "people"}, {PEOPLE, "people"}, [PEOPLE]),
        ({PEOPLE, "people"}, {PEOPLE}, [PEOPLE, "people"]),
        ({"people"}, {"people"}, [PEOPLE, "people"]),
    ],
)
def test_a_short_name_stands_for_a_key_only_while_it_is_its_own(
    declared, typed, expected
):
    assert (
        reached_relation_names([PEOPLE, "people"], declared=declared, typed=typed)
        == expected
    )
    assert reached_relation_names([PEOPLE], declared=declared, typed=typed) == [PEOPLE]


def anchored_constant(name):
    return {
        "name": name,
        "arity": 2,
        "derivation": {
            "source_relation": "people",
            "current_slot": 0,
            "related_slot": 1,
            "predicate": {
                "kind": "comparison",
                "op": "eq",
                "left": {"kind": "literal", "value": {"kind": "integer", "value": 1}},
                "right": {"kind": "literal", "value": {"kind": "integer", "value": 1}},
            },
        },
    }


@pytest.mark.parametrize(
    "slots,missing",
    [(["Person", "TaxUnit"], True), (None, False)],
)
def test_a_qualified_key_assigns_only_an_untyped_short_relation(slots, missing):
    short = {"name": "people", "arity": 2}
    if slots:
        short["slot_entities"] = slots
    canonical = {
        "name": PEOPLE,
        "arity": 2,
        "slot_entities": ["Household", "Person"],
    }
    relations = [canonical, short, anchored_constant("selected")]
    case = {"input": {PEOPLE: []}}
    issues = pair_relation_assignment_issues(relations, case, "probe")
    assert bool(issues) is missing


@pytest.mark.parametrize(
    "compiled,declared,expected",
    [
        ([PEOPLE], {"people": ("Person", "TaxUnit")}, {"people": (0, "TaxUnit")}),
        ([], {"people": ("Person", "TaxUnit")}, {}),
        ([PEOPLE], {}, {}),
        ([], {}, {}),
    ],
)
def test_a_key_settles_a_short_relation_only_while_it_feeds_it(
    compiled, declared, expected
):
    derived = {
        "result": {
            "name": "result",
            "entity": "Person",
            "expr": {
                "kind": "count_related",
                "relation": "people",
                "current_slot": 0,
                "related_slot": 1,
                "where": None,
            },
        }
    }
    found = executable_relation_directions(
        derived,
        ["result"],
        {"start": "2026-01-01"},
        "Person",
        declared,
        [{"name": name} for name in compiled],
        explicit_relations={PEOPLE},
    )
    assert found == expected


def test_a_legacy_compact_row_keeps_mains_short_name_copy(tmp_path):
    payload = _legacy_program()
    pairs = next(
        relation
        for relation in payload["program"]["relations"]
        if relation["name"] == DATASET_PAIRS
    )
    short = copy.deepcopy(pairs)
    short["name"] = DATASET_PAIRS.rsplit("#relation.", 1)[1]
    payload["program"]["relations"].append(short)
    case = {
        "input": {DATASET_PAIRS: [{AGE: 20}]},
        "output": {OUTPUT: 1},
    }
    for request in _requests(tmp_path, case, payload):
        assert {record["name"] for record in request["dataset"]["relations"]} == {
            DATASET_PAIRS,
            short["name"],
        }


def test_a_legacy_tuple_stays_out_of_a_typed_short_relation(tmp_path):
    payload = _legacy_program()
    pairs = next(
        relation
        for relation in payload["program"]["relations"]
        if relation["name"] == DATASET_PAIRS
    )
    payload["program"]["relations"].append(
        {**copy.deepcopy(pairs), "name": DATASET_PAIRS.rsplit("#relation.", 1)[1]}
    )
    case = {
        "entity_id": "case",
        "input": {DATASET_PAIRS: [{"tuple": ["case", "bob"]}]},
        "output": {OUTPUT: 1},
    }
    for request in _requests(tmp_path, case, payload):
        assert [record["name"] for record in request["dataset"]["relations"]] == [
            DATASET_PAIRS
        ]


@pytest.mark.parametrize("executor", EXECUTORS)
@pytest.mark.parametrize("relation", ["pairs", f"{TARGET}#relation.pairs"])
def test_a_legacy_empty_relation_keeps_mains_direction_check(
    tmp_path, real_engine, executor, relation
):
    forward, flipped = (
        view(
            name,
            "pairs",
            "true",
            slots=EVERYONE,
            entity="Person",
            member_relation=f"{name}_members",
            current_slot=current,
            related_slot=1 - current,
        )
        for name, current in (("forward", 0), ("flipped", 1))
    )
    case = {
        "name": "probe",
        "entity_id": "a",
        "period": "2026-01",
        "input": {relation: []},
        "output": {f"{TARGET}#result": 0},
    }
    root, module = write_module(
        tmp_path,
        program(
            PAIRS,
            forward,
            flipped,
            rule("result", "Person", "len(forward) + len(flipped)"),
        ),
        [case],
    )
    failures = run(executor, tmp_path, real_engine, root, module)
    assert any("conflicting executable directions" in item for item in failures)


def test_a_role_compact_row_stays_in_its_compiled_relation(tmp_path):
    payload = _program()
    pairs = next(
        relation
        for relation in payload["program"]["relations"]
        if relation["name"] == DATASET_PAIRS
    )
    short = copy.deepcopy(pairs)
    short["name"] = DATASET_PAIRS.rsplit("#relation.", 1)[1]
    payload["program"]["relations"].append(short)
    case = {
        "input": {DATASET_PAIRS: [{"id": "bob", AGE: 20}], TIES: []},
        "output": {OUTPUT: 1},
    }
    cli = _requests(tmp_path, case, payload, cli_only=True)
    assert {record["name"] for record in cli["dataset"]["relations"]} == {DATASET_PAIRS}


def test_a_member_alias_import_read_keeps_its_local_precedence():
    payload = competing("first", "uncovered")
    payload["imports"] = [f"{PROVIDER}#relation.members"]
    assert validation._rulespec_relation_import_reads(payload) == set()


def test_a_full_relation_name_reads_that_relation_alone():
    relations = [
        {"name": PEOPLE, "arity": 2, "slot_entities": ["Household", "Person"]},
        {"name": "people", "arity": 2, "slot_entities": ["Person", "TaxUnit"]},
    ]
    canonical = {"result": {"name": "result", "expr": {"relation": PEOPLE}}}
    short = {"result": {"name": "result", "expr": {"relation": "people"}}}
    assert _relations_read_by({"result": 1}, canonical, relations) == {PEOPLE}
    assert _relations_read_by({"result": 1}, short, relations) == {PEOPLE, "people"}
