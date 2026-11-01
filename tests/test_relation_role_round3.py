"""Pin the third-round lowering-entry and imported-alias findings."""

from __future__ import annotations

import hashlib

import pytest
import yaml

from axiom_encode.formula_references import (
    analyze_relation_predicate,
    rule_lowering_entry,
)
from axiom_encode.harness import validator_pipeline as validation
from tests.test_companion_declared_entities import _companions, _compile
from tests.test_relation_role_review_fixes import relation, scalar_payload

LABEL = "# Synthetic encoder test input; not a RuleSpec module.\n"
TARGET = "us:policies/example/judgment"
PROVIDER = "us:policies/example/provider#relation.unused_view"
PAIRS = {
    "name": "pairs",
    "kind": "data_relation",
    "data_relation": {"arity": 2, "arguments": ["Person", "Person"]},
}


def judgment_payload(formula):
    payload = scalar_payload(formula)
    payload["rules"][0]["dtype"] = "Judgment"
    payload["rules"][:0] = [PAIRS, relation("unused_members", True)]
    return payload


def write_case(root, payload, inputs, output):
    module = root / "us/policies/example/judgment.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(LABEL + yaml.safe_dump(payload))
    case = {
        "name": "case",
        "entity_id": "a",
        "period": "2026-01",
        "input": {
            **{f"{TARGET}#input.{name}": value for name, value in inputs.items()},
            f"{TARGET}#relation.pairs": [],
        },
        "output": {f"{TARGET}#result": output},
    }
    module.with_suffix(".test.yaml").write_text(LABEL + yaml.safe_dump([case]))
    return module


def consumer(formula, *imports):
    payload = scalar_payload(formula)
    payload["imports"] = list(imports)
    return payload


def filtered_view():
    view = relation("local_view", True)
    view["derived_relation"].update(entity="SnapUnit", member_relation="local_members")
    return view


@pytest.mark.parametrize(
    "rule,entry",
    [
        ({"kind": "derived_relation", "dtype": "Judgment"}, "derived_relation"),
        ({"kind": "derived_relation"}, "derived_relation"),
        ({"kind": "derived", "dtype": "Judgment"}, "judgment"),
        ({"kind": "derived", "dtype": "judgment"}, "judgment"),
        ({"kind": "derived", "dtype": "Boolean"}, "rule"),
        ({"kind": "derived", "dtype": "Money"}, "rule"),
        ({"kind": "derived"}, "rule"),
        (None, "rule"),
    ],
)
def test_rule_lowering_entry_follows_the_engine_dispatch(rule, entry):
    assert rule_lowering_entry(rule) == entry


@pytest.mark.parametrize(
    "formula,reads",
    [
        ("not unused_members", {"unused_members"}),
        ("unused_members and flag", {"unused_members", "flag"}),
        ("not (unused_members or flag)", {"unused_members", "flag"}),
    ],
)
def test_judgment_root_accepts_boolean_operators_as_input_reads(formula, reads):
    relation_names = frozenset({"unused_members"})
    analyzed = analyze_relation_predicate(
        formula,
        predicate_terms=False,
        lowering_entry="judgment",
        relation_names=relation_names,
    )
    assert analyzed.analysis_errors == ()
    assert analyzed.input_reads == reads
    assert analyzed.relation_arguments == set()
    assert analyzed.atom_relations == frozenset()


def test_judgment_entry_keeps_the_ordinary_rule_namespace():
    """Only a derived relation's predicate reads a bare relation name."""
    payload = judgment_payload("not unused_members")
    assert "unused_members" in validation._rulespec_local_input_slots(
        payload, imports=set(), rules_file=None, policy_repo_path=None
    )


@pytest.mark.parametrize("executor", ["cli", "validator"])
@pytest.mark.parametrize(
    "formula,inputs,output",
    [
        ("not unused_members", {"unused_members": False}, "holds"),
        ("not unused_members", {"unused_members": True}, "not_holds"),
        ("unused_members or flag", {"unused_members": False, "flag": True}, "holds"),
        (
            "unused_members and flag",
            {"unused_members": True, "flag": False},
            "not_holds",
        ),
    ],
)
def test_judgment_scalar_named_like_a_relation_runs_in_both_runners(
    tmp_path, real_engine, executor, formula, inputs, output
):
    root = tmp_path / "rulespec-us"
    module = write_case(root, judgment_payload(formula), inputs, output)
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


def test_unproven_imported_alias_read_keeps_the_base_refusal():
    payload = consumer("if unused_view: len(members) else: 0", PROVIDER)
    payload["rules"][0]["entity"] = "FilteredPerson"
    assert validation.find_unused_import_issues(yaml.safe_dump(payload)) == [
        f"Unused import `{PROVIDER}`: imported symbol `relation.unused_view` is "
        "not referenced by any formula or proof import."
    ]


def test_bare_imported_alias_read_keeps_the_base_refusal():
    payload = consumer("len(members)", PROVIDER)
    issues = validation.find_unused_import_issues(yaml.safe_dump(payload))
    assert len(issues) == 1
    assert "#relation.unused_view`" in issues[0]


def test_alias_with_a_canonical_proof_keeps_its_import(tmp_path):
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/consumer.yaml"
    module.parent.mkdir(parents=True)
    provider = module.with_name("provider.yaml")
    view = relation("unused_view", True)
    view["derived_relation"].update(entity="SnapUnit", member_relation="members")
    provider.write_text(
        LABEL + yaml.safe_dump({"format": "rulespec/v1", "rules": [PAIRS, view]})
    )
    digest = hashlib.sha256(provider.read_bytes()).hexdigest()
    payload = consumer("if unused_view: len(members) else: 0", PROVIDER)
    payload["rules"][0]["entity"] = "SnapUnit"
    payload["rules"][0]["metadata"] = {
        "proof": {
            "atoms": [
                {
                    "path": "versions[0].formula",
                    "kind": "import",
                    "import": {
                        "target": PROVIDER,
                        "output": "unused_view",
                        "hash": f"sha256:{digest}",
                    },
                }
            ]
        }
    }
    content = LABEL + yaml.safe_dump(payload)
    module.write_text(content)
    assert validation.find_unused_import_issues(content) == []
    assert validation.find_proof_import_reference_issues(content) == []
    assert (
        validation.find_proof_import_hash_consistency_issues(
            content, rules_file=module, policy_repo_path=root / "us"
        )
        == []
    )


def proven(payload, target):
    payload["rules"][0]["metadata"] = {
        "proof": {
            "atoms": [
                {
                    "path": "versions[0].formula",
                    "kind": "import",
                    "import": {"target": target, "output": "unused_view"},
                }
            ]
        }
    }
    return payload


def test_a_proof_keeps_only_the_import_it_names():
    extra = "us:policies/example/provider#relation.unused_extra"
    payload = consumer("if unused_view: len(members) else: 0", PROVIDER, extra)
    issues = validation.find_unused_import_issues(
        yaml.safe_dump(proven(payload, PROVIDER))
    )
    assert len(issues) == 1
    assert "#relation.unused_extra`" in issues[0]


def test_a_local_alias_with_the_same_name_leaves_a_proven_import_alone():
    payload = consumer("if unused_view: len(members) else: 0", PROVIDER)
    view = relation("local_view", True)
    view["derived_relation"].update(entity="LocalUnit", member_relation="members")
    proven(payload, PROVIDER)["rules"][:0] = [PAIRS, view]
    assert validation.find_unused_import_issues(yaml.safe_dump(payload)) == []


@pytest.mark.parametrize(
    "source",
    [
        {"source_relation": "unused_view"},
        {"source_path": [{"relation": "unused_view"}]},
    ],
    ids=["relation", "path"],
)
def test_a_derived_relation_source_keeps_its_import(source):
    payload = consumer("len(selected)", PROVIDER)
    view = relation("selected", "true")
    del view["derived_relation"]["source_relation"]
    view["derived_relation"].update(source)
    payload["rules"].insert(0, view)
    assert validation.find_unused_import_issues(yaml.safe_dump(payload)) == []


def test_a_formula_outside_the_port_keeps_the_base_scan():
    payload = consumer("$ relation.unused_view", PROVIDER)
    assert validation.find_unused_import_issues(yaml.safe_dump(payload)) == []
    payload = consumer("$ unused_view", PROVIDER)
    assert len(validation.find_unused_import_issues(yaml.safe_dump(payload))) == 1


def test_import_stays_unused_beside_a_local_alias_read():
    payload = consumer("len(local_members)", PROVIDER)
    payload["rules"][:0] = [PAIRS, filtered_view()]
    issues = validation.find_unused_import_issues(yaml.safe_dump(payload))
    assert len(issues) == 1
    assert "#relation.unused_view`" in issues[0]


def test_import_stays_unused_beside_a_local_relation_read():
    payload = consumer("len(selected)", PROVIDER)
    payload["rules"][:0] = [PAIRS, relation("selected", "true")]
    issues = validation.find_unused_import_issues(yaml.safe_dump(payload))
    assert len(issues) == 1
    assert "#relation.unused_view`" in issues[0]


@pytest.mark.parametrize(
    "sibling",
    [
        "us:policies/example/other#relation.selected",
        "us:policies/example/other#selected",
    ],
)
def test_import_stays_unused_when_a_sibling_import_explains_the_read(sibling):
    payload = consumer("len(selected)", PROVIDER, sibling)
    issues = validation.find_unused_import_issues(yaml.safe_dump(payload))
    assert any("#relation.unused_view`" in issue for issue in issues)


def test_scalar_import_stays_unused_beside_an_alias_read():
    scalar = "us:policies/example/provider#unused_rate"
    payload = consumer("if unused_view: len(members) else: 0", PROVIDER, scalar)
    issues = validation.find_unused_import_issues(
        yaml.safe_dump(proven(payload, PROVIDER))
    )
    assert len(issues) == 1
    assert "#unused_rate`" in issues[0]


@pytest.mark.parametrize("formula", ["unused_view", "not unused_view"])
def test_predicate_bare_relation_read_uses_its_explicit_import(formula):
    payload = consumer("len(selected)", PROVIDER)
    payload["rules"][:0] = [PAIRS, relation("selected", formula)]
    assert validation.find_unused_import_issues(yaml.safe_dump(payload)) == []


def test_judgment_scalar_read_does_not_use_an_explicit_relation_import():
    payload = consumer("not unused_view", PROVIDER)
    payload["rules"][0]["dtype"] = "Judgment"
    issues = validation.find_unused_import_issues(yaml.safe_dump(payload))
    assert len(issues) == 1
    assert "#relation.unused_view`" in issues[0]


def test_default_check_follows_a_bare_relation_read():
    payload = {
        "format": "rulespec/v1",
        "inputs": {"age": {"dtype": "Integer", "entity": "Person", "default": 0}},
        "rules": [
            PAIRS,
            relation("base_view", "candidate.age > 0"),
            relation("outer_view", "base_view"),
        ],
    }
    issues = validation.find_relation_role_issues(yaml.safe_dump(payload))
    assert issues == [
        f"[relation-input-default] Input `age` read by derived relation `{name}` "
        "must not declare a default. Supply the fact explicitly."
        for name in ("base_view", "outer_view")
    ]


def test_bare_relation_read_compiles_as_a_predicate(tmp_path, real_engine):
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/bare.yaml"
    module.parent.mkdir(parents=True)
    payload = {
        "format": "rulespec/v1",
        "rules": [
            PAIRS,
            relation("base_view", "true"),
            relation("outer_view", "base_view"),
        ],
    }
    module.write_text(LABEL + yaml.safe_dump(payload))
    compiled = _compile(
        real_engine.binary,
        real_engine.root,
        root,
        module,
        tmp_path / "compiled.json",
    )
    assert compiled["program"]


def scalar_module(root, name, payload):
    module = root / f"us/policies/example/{name}.yaml"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(LABEL + yaml.safe_dump({"format": "rulespec/v1", **payload}))
    return module


def scalar_rule(formula, *, dtype="Integer", name="result"):
    return {
        "name": name,
        "kind": "derived",
        "entity": "Person",
        "dtype": dtype,
        "versions": [{"effective_from": "2026-01-01", "formula": formula}],
    }


def parameter_rule(name):
    return {
        "name": name,
        "kind": "parameter",
        "dtype": "Integer",
        "versions": [{"effective_from": "2026-01-01", "formula": 1}],
    }


def role_rules(*rules):
    """Put rules in a relation-role program, the only kind that reads dtypes."""
    return [PAIRS, relation("view", True), *rules]


def scalar_case(module, inputs, output, *, role=False):
    target = "us:policies/example/" + module.stem
    case = {
        "name": "scalar",
        "period": "2026-01",
        "input": {f"{target}#input.{name}": value for name, value in inputs.items()},
        "output": {f"{target}#result": output},
    }
    if role:
        case["entity_id"] = "a"
        case["input"][f"{target}#relation.pairs"] = []
    module.with_suffix(".test.yaml").write_text(LABEL + yaml.safe_dump([case]))


def run_companions(tmp_path, real_engine, root, module, executor):
    return _companions(
        executor,
        binary=real_engine.binary,
        engine_root=real_engine.root,
        root=root,
        module=module,
        artifact=tmp_path / "compiled.json",
    )


@pytest.mark.parametrize(
    "executor,expected", [("cli", 0), ("validator", 1)], ids=["cli", "validator"]
)
def test_an_untyped_quoted_number_keeps_each_runners_own_meaning(
    tmp_path, real_engine, executor, expected
):
    root = tmp_path / "rulespec-us"
    module = scalar_module(
        root, "codes", {"rules": [scalar_rule("if code == other: 1 else: 0")]}
    )
    scalar_case(module, {"code": "001", "other": "1"}, expected)
    assert run_companions(tmp_path, real_engine, root, module, executor) == []


@pytest.mark.parametrize(
    "executor,dtype,expected",
    [
        ("cli", "Integer", 0),
        ("cli", "Text", 0),
        ("validator", "Integer", 1),
        ("validator", "Text", 1),
    ],
)
def test_a_declared_type_leaves_a_legacy_program_alone(
    tmp_path, real_engine, executor, dtype, expected
):
    root = tmp_path / "rulespec-us"
    module = scalar_module(
        root,
        "codes",
        {
            "inputs": [
                {"name": "code", "dtype": dtype},
                {"name": "other", "dtype": dtype},
            ],
            "rules": [scalar_rule("if code == other: 1 else: 0")],
        },
    )
    scalar_case(module, {"code": "001", "other": "1"}, expected)
    assert run_companions(tmp_path, real_engine, root, module, executor) == []


@pytest.mark.parametrize("executor", ["cli", "validator"])
@pytest.mark.parametrize(
    "first,second", [("Text", "Integer"), ("Text", "Text")], ids=["clash", "same"]
)
def test_unread_imported_declarations_leave_a_consumer_input_alone(
    tmp_path, real_engine, executor, first, second
):
    root = tmp_path / "rulespec-us"
    for name, dtype in (("one", first), ("two", second)):
        scalar_module(
            root,
            name,
            {
                "inputs": [{"name": "amount", "dtype": dtype}],
                "rules": [parameter_rule(f"threshold_{name}")],
            },
        )
    consumer = scalar_module(
        root,
        "consumer",
        {
            "imports": [
                "us:policies/example/one#threshold_one",
                "us:policies/example/two#threshold_two",
            ],
            "rules": role_rules(
                scalar_rule(
                    "amount == other and threshold_one == threshold_two",
                    dtype="Judgment",
                )
            ),
        },
    )
    scalar_case(consumer, {"amount": 1, "other": 1}, "holds", role=True)
    assert run_companions(tmp_path, real_engine, root, consumer, executor) == []


@pytest.mark.parametrize(
    "executor,dtype,output",
    [("validator", "Text", "not_holds"), ("cli", "Integer", "holds")],
    ids=["text-stays-text", "number-converts"],
)
def test_a_co_reader_declaration_types_the_shared_slot(
    tmp_path, real_engine, executor, dtype, output
):
    root = tmp_path / "rulespec-us"
    scalar_module(
        root,
        "provider",
        {
            "inputs": [
                {"name": "code", "dtype": dtype},
                {"name": "other", "dtype": dtype},
            ],
            "rules": [scalar_rule("code == other", dtype="Judgment", name="matches")],
        },
    )
    consumer = scalar_module(
        root,
        "consumer",
        {
            "imports": ["us:policies/example/provider#matches"],
            "rules": role_rules(
                scalar_rule("code == other and matches", dtype="Judgment")
            ),
        },
    )
    scalar_case(consumer, {"code": "01", "other": "1"}, output, role=True)
    assert run_companions(tmp_path, real_engine, root, consumer, executor) == []


@pytest.mark.parametrize("executor", ["cli", "validator"])
@pytest.mark.parametrize(
    "first,second,output",
    [("int", "Integer", "holds"), ("Text", "String", "not_holds")],
    ids=["number", "text"],
)
def test_equivalent_reader_spellings_type_a_shared_slot(
    tmp_path, real_engine, executor, first, second, output
):
    root = tmp_path / "rulespec-us"
    for name, dtype in (("one", first), ("two", second)):
        scalar_module(
            root,
            name,
            {
                "inputs": [
                    {"name": "code", "dtype": dtype},
                    {"name": "other", "dtype": dtype},
                ],
                "rules": [
                    scalar_rule("code == other", dtype="Judgment", name=f"{name}_read")
                ],
            },
        )
    consumer = scalar_module(
        root,
        "consumer",
        {
            "imports": [
                "us:policies/example/one#one_read",
                "us:policies/example/two#two_read",
            ],
            "rules": role_rules(
                scalar_rule("code == other and one_read and two_read", dtype="Judgment")
            ),
        },
    )
    scalar_case(consumer, {"code": "01", "other": "1"}, output, role=True)
    assert run_companions(tmp_path, real_engine, root, consumer, executor) == []
