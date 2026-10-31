"""Pin the independent round-one relation-role review findings."""

from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import evals
from axiom_encode.harness import validator_pipeline as validation


def scalar_payload(formula: str) -> dict:
    """Build synthetic encoder test input, never a corpus module."""
    return {
        "format": "rulespec/v1",
        "rules": [
            {
                "name": "result",
                "kind": "derived",
                "entity": "Person",
                "dtype": "Integer",
                "versions": [{"effective_from": "2026-01-01", "formula": formula}],
            }
        ],
    }


@pytest.mark.parametrize("symbol", ["max", "min", "count", "len", "exists"])
def test_c1_bare_builtin_import_is_used(symbol):
    payload = scalar_payload(f"{symbol} + 1")
    payload["imports"] = [f"us:statutes/example#{symbol}"]
    assert validation.find_unused_import_issues(yaml.safe_dump(payload)) == []


@pytest.mark.parametrize(
    "formula", ["max(a, b)", "1 # max", '""" max """ 1', '"max"', "'max'"]
)
def test_c1_nonreads_do_not_use_builtin_import(formula):
    payload = scalar_payload(formula)
    payload["imports"] = ["us:statutes/example#max"]
    assert validation.find_unused_import_issues(yaml.safe_dump(payload)) == [
        "Unused import `us:statutes/example#max`: imported symbol `max` is "
        "not referenced by any formula or proof import."
    ]


def test_c1_unparsed_formula_preserves_base_import_usage():
    payload = scalar_payload("max + @")
    payload["imports"] = ["us:statutes/example#max"]
    assert validation.find_unused_import_issues(yaml.safe_dump(payload)) == []


def relation(name: str, formula: str) -> dict:
    return {
        "name": name,
        "kind": "derived_relation",
        "derived_relation": {
            "arity": 2,
            "slot_entities": ["Person", "Person"],
            "current_slot": 0,
            "related_slot": 1,
            "source_relation": "pairs",
        },
        "versions": [{"effective_from": "2026-01-01", "formula": formula}],
    }


def test_a1_predicate_value_read_does_not_cover_relation(tmp_path):
    root = tmp_path / "rulespec-us" / "us"
    module = root / "policies/example/coverage.yaml"
    module.parent.mkdir(parents=True)
    payload = scalar_payload("len(selected)")
    payload["rules"][:0] = [
        {
            "name": "pairs",
            "kind": "data_relation",
            "data_relation": {"arity": 2, "arguments": ["Person", "Person"]},
        },
        relation("unused_members", "candidate.broken"),
        relation("selected", "unused_members > 0"),
    ]
    content = yaml.safe_dump(payload)
    module.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n" + content
    )
    cases = [{"output": {"us:policies/example/coverage#result": 1}}]
    assert validation.find_missing_derived_companion_output_issues(
        content, cases, rules_file=module, policy_repo_path=root
    ) == [
        "Derived relation missing companion output coverage: "
        "`us:policies/example/coverage#unused_members` is not reached by an "
        "asserted scalar output in the companion `.test.yaml` file."
    ]
    assert "unused_members" in evals._context_file_terminal_exports(str(module))
    assert (
        "unused_members"
        in validation._rulespec_terminal_executable_names_from_payload(payload)
    )


@pytest.mark.parametrize("executor", ["cli", "validator"])
def test_a1_engine_reads_same_spelling_as_input_in_predicate(
    tmp_path, real_engine, executor
):
    from tests.test_companion_declared_entities import _companions

    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/coverage.yaml"
    module.parent.mkdir(parents=True)
    payload = scalar_payload("len(selected)")
    payload["rules"][:0] = [
        {
            "name": "pairs",
            "kind": "data_relation",
            "data_relation": {"arity": 2, "arguments": ["Person", "Person"]},
        },
        relation("unused_members", "candidate.broken"),
        relation("selected", "unused_members > 0"),
    ]
    module.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        + yaml.safe_dump(payload)
    )
    target = "us:policies/example/coverage"
    cases = [
        {
            "name": "value_read_is_input",
            "period": "2026-01",
            "tables": {
                "Person": [
                    {"id": "a"},
                    {"id": "b", f"{target}#input.unused_members": 1},
                ]
            },
            "input": {f"{target}#relation.pairs": [{"tuple": ["a", "b"]}]},
            "output": {f"{target}#result": [1, 0]},
        }
    ]
    module.with_suffix(".test.yaml").write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        + yaml.safe_dump(cases)
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


def test_c2_scalar_module_skips_import_and_identity_traversal(monkeypatch):
    payload = scalar_payload("age")
    payload["imports"] = ["us:statutes/example#choice"]

    def forbidden(*args, **kwargs):
        pytest.fail("scalar applicability guard traversed imports or identities")

    monkeypatch.setattr(validation, "_resolve_rulespec_import_file_static", forbidden)
    monkeypatch.setattr(validation, "_canonical_rulespec_file_target", forbidden)
    assert (
        validation.find_relation_role_issues(
            yaml.safe_dump(payload),
            rules_file=Path("consumer.yaml"),
            policy_repo_path=Path("us"),
        )
        == []
    )


@pytest.mark.parametrize("election", [None, {}, False, "bad"])
def test_c2_malformed_election_key_still_runs_without_relation(election):
    payload = scalar_payload("choice")
    payload["inputs"] = {"choice": {"election": election}}
    issues = validation.find_relation_role_issues(yaml.safe_dump(payload))
    assert len(issues) == 1
    assert issues[0].startswith("[election-declaration]")


@pytest.mark.parametrize("has_relation", [False, True])
def test_b7_imported_election_default_checked_when_applicable(tmp_path, has_relation):
    root = tmp_path / "rulespec-us" / "us"
    provider = root / "statutes/provider.yaml"
    provider.parent.mkdir(parents=True)
    library = scalar_payload("choice")
    library["inputs"] = {
        "choice": {
            "default": False,
            "election": {
                "actor": "Household",
                "provision": "us/regulation/7/273/1/b/3",
            },
        }
    }
    provider.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        + yaml.safe_dump(library)
    )
    consumer = scalar_payload("result")
    consumer["rules"][0]["name"] = "consumer"
    consumer["imports"] = ["us:statutes/provider#result"]
    if has_relation:
        consumer["rules"].append(relation("members", "true"))
    module = root / "statutes/consumer.yaml"
    issues = validation.find_relation_role_issues(
        yaml.safe_dump(consumer), rules_file=module, policy_repo_path=root
    )
    assert issues == (
        [
            "[election-default] Election input `us:statutes/provider#input.choice` "
            "must not declare a default. Supply the actor's choice explicitly."
        ]
        if has_relation
        else []
    )


def test_b1_predicate_parse_failure_fails_closed_with_position():
    payload = scalar_payload("1")
    payload["rules"].append(relation("members", "candidate.age > @"))
    issues = validation.find_relation_role_issues(yaml.safe_dump(payload))
    assert any(
        issue.startswith("[relation-input-analysis]") and "line 1" in issue
        for issue in issues
    )


def test_c2_nameless_election_is_not_hidden_by_applicability_guard():
    payload = scalar_payload("choice")
    payload["inputs"] = [
        {"election": {"actor": "Household", "provision": "us/regulation/7/273/1/b/3"}}
    ]
    assert validation.find_relation_role_issues(yaml.safe_dump(payload)) == [
        "[election-declaration] Election input must declare a nonempty `name`."
    ]


@pytest.mark.parametrize("name", [17, True, [], {}, "", " "])
@pytest.mark.parametrize("rules", [7, None, []])
def test_b7_malformed_election_names_keep_default_validation(name, rules):
    payload = {
        "rules": rules,
        "inputs": [
            {
                "name": name,
                "default": 0,
                "election": {
                    "actor": "Household",
                    "provision": "us/regulation/7/273/1/b/3",
                },
            }
        ],
    }
    assert validation.find_relation_role_issues(yaml.safe_dump(payload)) == [
        "[election-declaration] Election input must declare a nonempty `name`.",
        "[election-default] Election input `<unnamed>` must not declare a default. "
        "Supply the actor's choice explicitly.",
    ]
