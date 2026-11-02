"""Preserve versioned relation context and reject derived shorthand."""

from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from axiom_encode.harness import evals
from axiom_encode.rule_version_form import find_rule_version_form_issues


def eval_relation_payload():
    """Return a synthetic encoder test input, never a corpus module."""
    return {
        "format": "rulespec/v1",
        "rules": [
            {
                "name": "people",
                "kind": "data_relation",
                "data_relation": {"arity": 2, "arguments": ["Person", "Person"]},
            },
            {
                "name": "under_22",
                "kind": "parameter",
                "dtype": "Integer",
                "versions": [{"effective_from": "2026-01-01", "formula": 22}],
            },
            {
                "name": "selected",
                "kind": "derived_relation",
                "derived_relation": {
                    "source_relation": "people",
                    "current_slot": 0,
                    "related_slot": 1,
                },
                "versions": [
                    {
                        "effective_from": "2026-01-01",
                        "formula": "candidate.age < under_22 and candidate.age < 22",
                    }
                ],
            },
            {
                "name": "relation_count",
                "kind": "derived",
                "entity": "Person",
                "dtype": "Integer",
                "period": "Month",
                "versions": [
                    {"effective_from": "2026-01-01", "formula": "len(selected)"}
                ],
            },
        ],
    }


def context_reader_snapshot(path, payload):
    content = yaml.safe_dump(payload)
    path.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n" + content
    )
    return {
        "inputs": evals._context_file_local_inputs(str(path)),
        "exports": evals._context_file_terminal_exports(str(path)),
        "surfaces": evals._context_file_executable_surfaces(str(path)),
        "concept_numbers": evals._numeric_concept_name_occurrences(content),
        "formula_numbers": evals._inline_table_formula_numeric_occurrences(content),
        "constant": evals._extract_simple_rulespec_constant(content, "under_22"),
    }


@pytest.mark.parametrize("date_alias", [False, True])
def test_eval_relation_readers_follow_versioned_date_aliases(tmp_path, date_alias):
    versioned = eval_relation_payload()
    expected = context_reader_snapshot(tmp_path / "versioned.yaml", versioned)
    aliased = deepcopy(versioned)
    if date_alias:
        for rule in aliased["rules"]:
            for version in rule.get("versions", []):
                version["from"] = version.pop("effective_from")
    actual = context_reader_snapshot(tmp_path / "aliased.yaml", aliased)
    assert actual == expected
    assert "age" in actual["inputs"]
    assert "candidate" not in actual["inputs"]
    assert actual["exports"] == ["relation_count"]
    assert actual["constant"] == 22
    assert actual["formula_numbers"] == {22.0: 1}


@pytest.mark.parametrize("empty_versions", [False, True])
def test_eval_flat_derived_rules_refuse_even_when_versions_exist(empty_versions):
    payload = eval_relation_payload()
    for rule in payload["rules"]:
        if rule.get("versions"):
            rule.update(formula="unexpected_fact < 987", effective_from="1900-01-01")
            if empty_versions:
                rule["versions"] = []
    assert find_rule_version_form_issues(yaml.safe_dump(payload)) == [
        "[rule-version-form] Rule `selected` must put `formula` and its dates under `versions:`.",
        "[rule-version-form] Rule `relation_count` must put `formula` and its dates under `versions:`.",
    ]


@pytest.mark.parametrize("versions", [True, {"formula": "unexpected_fact"}, "wrong"])
def test_eval_context_readers_leave_malformed_versions_to_engine(tmp_path, versions):
    payload = eval_relation_payload()
    for rule in payload["rules"]:
        if "versions" in rule:
            rule["versions"] = versions
    actual = context_reader_snapshot(tmp_path / "malformed.yaml", payload)
    assert actual["inputs"] == set()
    assert actual["formula_numbers"] == {}
    assert actual["constant"] is None


def test_eval_versioned_hint_reader_follows_relation_predicate_dependencies():
    payload = eval_relation_payload()
    payload["rules"][-2]["versions"][0]["formula"] = (
        "candidate.person_is_described_in_previous_mandatory_subclause"
    )
    versioned = yaml.safe_dump(payload)
    expected = evals._policyengine_hint_upstream_composition_issues(
        versioned, "relation_count"
    )
    assert expected
