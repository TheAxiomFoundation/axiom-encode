"""Pin the tenth-round import, compact-row, terminal and alias findings."""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from axiom_encode.harness import evals
from axiom_encode.harness import validator_pipeline as validation
from tests.test_companion_declared_entities import _compile
from tests.test_relation_role_datasets import MEMBERS as DATASET_MEMBERS
from tests.test_relation_role_datasets import OUTPUT, TIES, _program, _requests
from tests.test_relation_role_datasets import PAIRS as DATASET_PAIRS
from tests.test_relation_role_round6 import (
    PAIRS,
    PROVIDER,
    TARGET,
    VIEW,
    data_relation,
    rule,
    write_module,
)
from tests.test_relation_role_round7 import program, view
from tests.test_relation_role_round8 import EVERYONE, certified
from tests.test_relation_role_round9 import FIRST, alias_issues

MEMBERS = f"{PROVIDER}#relation.members"
OTHER_VIEW = "us:policies/example/other#relation.view"
AMOUNT = f"{PROVIDER}#amount"
SCALAR = f"{PROVIDER}#view"
LABEL = "# Synthetic encoder test input; not a RuleSpec module.\n"


def provider():
    amount = rule("amount", "Person", "1")
    return program(
        data_relation("view", "Person", "Person"),
        data_relation("members", "Household", "Person"),
        amount,
    )


def parameter(name):
    return {
        "name": name,
        "kind": "parameter",
        "dtype": "Integer",
        "versions": [{"effective_from": "2026-01-01", "values": {0: 1}}],
    }


def predicate(formula, **config):
    """Count a derived relation over `pairs` whose predicate reads `formula`."""
    selected = view("selected", "pairs", formula, slots=EVERYONE, **config)
    if "source_path" in config:
        del selected["derived_relation"]["source_relation"]
    return [PAIRS, selected, rule("result", "Person", "len(selected)")]


IMPORT_FORMS = [
    pytest.param([VIEW], predicate("view(anchor, candidate)"), True, id="atom"),
    pytest.param(
        [VIEW, VIEW], predicate("view(anchor, candidate)"), True, id="exact-duplicate"
    ),
    pytest.param(
        [VIEW],
        predicate("exists(view(anchor, p), p == candidate)"),
        True,
        id="exists",
    ),
    pytest.param([VIEW], predicate("true", source_relation=VIEW), True, id="source"),
    pytest.param(
        [VIEW],
        predicate(
            "true", source_path=[{"relation": VIEW, "from_slot": 0, "to_slot": 1}]
        ),
        True,
        id="path-step",
    ),
    pytest.param(
        [VIEW], predicate("true", source_relation="view"), False, id="bare-source"
    ),
    pytest.param(
        [VIEW],
        predicate(
            "true", source_path=[{"relation": "view", "from_slot": 0, "to_slot": 1}]
        ),
        False,
        id="bare-path-step",
    ),
    pytest.param([VIEW], predicate("view"), False, id="bare-member"),
    pytest.param(
        [MEMBERS], [rule("result", "Household", "len(members)")], False, id="len"
    ),
    pytest.param(
        [VIEW, OTHER_VIEW],
        predicate("view(anchor, candidate)"),
        False,
        id="shared-short-name",
    ),
    pytest.param(
        [VIEW],
        [
            data_relation("view", "Person", "Person"),
            *predicate("view(anchor, candidate)"),
        ],
        True,
        id="local-relation-takes-the-atom",
    ),
    pytest.param(
        [VIEW],
        [rule("result", "Person", "view(result, result)")],
        False,
        id="atom-in-an-ordinary-rule",
    ),
    pytest.param(
        [VIEW],
        [rule("view", "Person", "1"), *predicate("view(anchor, candidate)")],
        False,
        id="local-rule-shares-the-name",
    ),
    pytest.param(
        [VIEW],
        [parameter("view"), *predicate("view(anchor, candidate)")],
        False,
        id="local-parameter-shares-the-name",
    ),
    pytest.param([VIEW], predicate("len(view) > 0"), False, id="len-in-a-predicate"),
    pytest.param(
        [AMOUNT], predicate("true", source_relation=AMOUNT), False, id="scalar-source"
    ),
    pytest.param(
        [AMOUNT],
        predicate(
            "true", source_path=[{"relation": AMOUNT, "from_slot": 0, "to_slot": 1}]
        ),
        False,
        id="scalar-path-step",
    ),
    pytest.param(
        [f"{PROVIDER}#relation."],
        predicate("true", source_relation=f"{PROVIDER}#relation."),
        False,
        id="empty-relation-name",
    ),
]


def engine_reads(tmp_path, real_engine, imports, rules, **siblings):
    """Compile a consumer and name the explicit imports its own rules read."""
    consumer = program(*copy.deepcopy(rules), imports=imports)
    other = program(data_relation("view", "Person", "Person"))
    root, module = write_module(
        tmp_path, consumer, provider=provider(), other=other, **siblings
    )
    try:
        compiled = _compile(
            real_engine.binary,
            real_engine.root,
            root,
            module,
            tmp_path / "compiled.json",
        )
    except AssertionError:
        return None, module.read_text()
    own = [
        item
        for key in ("derived", "relations")
        for item in compiled["program"][key]
        if str(item.get("id") or item.get("name")).startswith(f"{TARGET}#")
    ]
    text = json.dumps(own)
    return {item for item in imports if f'"{item}"' in text}, module.read_text()


@pytest.mark.parametrize("imports,rules,compiles", IMPORT_FORMS)
def test_the_import_scan_counts_exactly_the_reads_the_engine_binds(
    tmp_path, real_engine, imports, rules, compiles
):
    """An explicit relation import is used exactly when the engine binds it."""
    reads, content = engine_reads(tmp_path, real_engine, imports, rules)
    assert (reads is not None) is compiles
    unused = {
        item
        for item in imports
        if any(
            f"`{item}`" in issue
            for issue in validation.find_unused_import_issues(content)
        )
    }
    assert set(imports) - unused == (reads or set())


@pytest.mark.parametrize(
    "formula", ["sum(members.amount)", "count_where(members, amount)"]
)
def test_an_aggregate_over_an_imported_value_keeps_mains_unused_import(
    tmp_path, real_engine, formula
):
    """Intended limit, as on main: the engine binds this aggregate to the import.

    The engine resolves an aggregate's short relation name to an import only
    when its value or condition reads a rule from another module. The scan
    reads one module, so it keeps main's finding for this form.
    """
    imports = [MEMBERS, AMOUNT]
    reads, content = engine_reads(
        tmp_path, real_engine, imports, [rule("result", "Household", formula)]
    )
    assert reads == {MEMBERS}
    issues = validation.find_unused_import_issues(content)
    assert [issue for issue in issues if f"`{MEMBERS}`" in issue]


def test_a_bare_proof_selector_matches_main_across_the_joined_formulas():
    reader = rule("result", "Person", "view")
    reader["versions"].append({"effective_from": "2027-01-01", "formula": "x"})
    reader["metadata"] = {
        "proof": {"atoms": [{"import": {"target": SCALAR, "output": "view\nx"}}]}
    }
    assert certified(yaml.safe_dump(program(reader, imports=[SCALAR])))


def clashing_compact_rows():
    payload = _program()
    for relation in payload["program"]["relations"]:
        if relation["name"] in (DATASET_PAIRS, DATASET_MEMBERS):
            relation["slot_entities"] = ["Person", "Household"]
    payload["program"]["relations"][-1]["derivation"]["slot_entities"] = [
        "Person",
        "Household",
    ]
    return payload


@pytest.mark.parametrize(
    "case,shared",
    [
        ({"input": {DATASET_PAIRS: [{"id": "case"}], TIES: []}}, "case"),
        (
            {
                "input": {DATASET_PAIRS: [{"id": "x"}], TIES: []},
                "tables": {"Person": [{"id": "x"}]},
            },
            "x",
        ),
    ],
    ids=["root-id", "table-row"],
)
@pytest.mark.parametrize("cli_only", [True, False], ids=["cli", "pipeline"])
def test_an_explicit_compact_row_id_joins_the_distinct_id_check(
    tmp_path, case, shared, cli_only
):
    """A Household row may not reuse a Person ID, in either runner."""
    case = {**case, "output": {OUTPUT: 1}}
    message = f"fixture entity ID `{shared}` is shared by `Household`, `Person`"
    with pytest.raises(ValueError, match=message):
        _requests(tmp_path, case, clashing_compact_rows(), cli_only=cli_only)


def unnamed(formula):
    nameless = rule("", "Person", formula)
    del nameless["name"]
    return nameless


TERMINAL_CASES = {
    "unnamed": ([rule("a", "Person", "1"), unnamed("a + 1")], ["a"]),
    "empty": ([rule("a", "Person", "1"), rule("", "Person", "a + 1")], ["a"]),
    "padded": ([rule("a", "Person", "1"), rule(" b ", "Person", "a + 1")], ["b"]),
    "padded-beside-relation": (
        [
            rule("a", "Person", "1"),
            rule(" b ", "Person", "a + 1"),
            PAIRS,
            view("selected", "pairs", "true", slots=EVERYONE),
        ],
        ["b", "selected"],
    ),
    "integer": ([rule("a", "Person", "1"), rule(7, "Person", "a + 1")], ["7"]),
}


@pytest.mark.parametrize("rules,expected", TERMINAL_CASES.values(), ids=TERMINAL_CASES)
def test_terminal_readers_return_a_list_for_a_malformed_name(tmp_path, rules, expected):
    """The engine refuses these names, and both readers still return a list.

    A padded name reads as its trimmed spelling, so the module beside a
    derived relation lists that relation too, as it does with a clean name.
    """
    payload = program(*copy.deepcopy(rules))
    module = tmp_path / "context.yaml"
    module.write_text(LABEL + yaml.safe_dump(payload))
    assert validation._rulespec_terminal_executable_names_from_payload(payload) == (
        expected
    )
    assert evals._context_file_terminal_exports(str(module)) == expected
    trimmed = copy.deepcopy(payload)
    for item in trimmed["rules"]:
        if isinstance(item.get("name"), str):
            item["name"] = item["name"].strip()
    if all(
        isinstance(item.get("name"), str) and item["name"] for item in trimmed["rules"]
    ):
        assert validation._rulespec_terminal_executable_names_from_payload(trimmed) == (
            expected
        )


def test_the_relation_id_alias_refusal_agrees_in_number():
    shared = program(
        PAIRS,
        view("first", "pairs", "true", slots=EVERYONE, entity="Other"),
        *(
            view(
                name,
                "pairs",
                "true",
                slots=EVERYONE,
                entity="Person",
                member_relation=FIRST,
            )
            for name in ("second", "third")
        ),
        rule("result", "Person", "len(second) + len(third)"),
    )
    issues = alias_issues(yaml.safe_dump(shared))
    assert any(
        issue.startswith(
            "[relation-member-alias] Derived relations `second`, `third` spell member "
            f"alias `{FIRST}` as a relation ID."
        )
        for issue in issues
    )
