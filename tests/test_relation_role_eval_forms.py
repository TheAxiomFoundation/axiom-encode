"""Keep native shorthand and versioned relation formulas equally visible."""

from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from axiom_encode.harness import evals


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


@pytest.mark.parametrize("empty_versions", [False, True])
@pytest.mark.parametrize("date_alias", [False, True])
def test_eval_relation_readers_follow_effective_flat_versions(
    tmp_path, empty_versions, date_alias
):
    versioned = eval_relation_payload()
    expected = context_reader_snapshot(tmp_path / "versioned.yaml", versioned)
    flat = deepcopy(versioned)
    for rule in flat["rules"]:
        if "versions" not in rule:
            continue
        version = rule.pop("versions")[0]
        if date_alias:
            version["from"] = version.pop("effective_from")
        rule.update(version)
        if empty_versions:
            rule["versions"] = []
    actual = context_reader_snapshot(tmp_path / "flat.yaml", flat)
    assert actual == expected
    assert "age" in actual["inputs"]
    assert "candidate" not in actual["inputs"]
    assert actual["exports"] == ["relation_count"]
    assert actual["constant"] == 22
    assert actual["formula_numbers"] == {22.0: 1}


def test_eval_relation_readers_ignore_shorthand_when_versions_exist(tmp_path):
    payload = eval_relation_payload()
    expected = context_reader_snapshot(tmp_path / "versioned.yaml", payload)
    for rule in payload["rules"]:
        if rule.get("versions"):
            rule.update(formula="unexpected_fact < 987", effective_from="1900-01-01")
    assert context_reader_snapshot(tmp_path / "both.yaml", payload) == expected


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


def test_eval_flat_hint_reader_follows_relation_predicate_dependencies():
    payload = eval_relation_payload()
    payload["rules"][-2]["versions"][0]["formula"] = (
        "candidate.person_is_described_in_previous_mandatory_subclause"
    )
    versioned = yaml.safe_dump(payload)
    expected = evals._policyengine_hint_upstream_composition_issues(
        versioned, "relation_count"
    )
    assert expected
    for rule in payload["rules"]:
        if rule.get("versions"):
            rule.update(rule.pop("versions")[0])
    assert (
        evals._policyengine_hint_upstream_composition_issues(
            yaml.safe_dump(payload), "relation_count"
        )
        == expected
    )
