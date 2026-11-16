"""Anonymous rows in different relations are different fixture entities."""

import json
import os

import pytest
import yaml

from axiom_encode.cli import _execute_rulespec_test_case, _execute_rulespec_test_file
from axiom_encode.harness.validator_pipeline import ValidatorPipeline
from axiom_encode.rules_engine_compat import run_rulespec_compile


def _request(tmp_path, lengths, *, current_slot=1):
    root = tmp_path / "rulespec-us"
    module = root / "us" / "policies" / "rows.yaml"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text("format: rulespec/v1\ninputs: []\nrules: []\n")
    prefix = "us:policies/rows#"
    relations = [f"rows_{i}" for i in range(len(lengths))]
    inputs = {
        prefix + "relation." + name: [
            {prefix + "input.amount": 100 * (i + 1) + j} for j in range(length)
        ]
        for i, (name, length) in enumerate(zip(relations, lengths))
    }
    # This tests request construction; it does not claim compiled provenance.
    derived = {
        "total": {"id": "total", "entity": "Person", "expr": {"kind": "literal"}}
    }
    return _execute_rulespec_test_case(
        module.with_suffix(".test.yaml"),
        {"period": "2025", "input": inputs, "output": {"total": 0}},
        case_name="distinct-rows",
        compiled_path=tmp_path / "unused.json",
        binary=tmp_path / "unused-engine",
        axiom_rules_path=tmp_path,
        env={},
        parameter_by_id={},
        derived_ids=set(derived),
        derived_by_id=derived,
        declared_relation_names=set(relations),
        declared_relation_slots={
            name: ("SourceEntry", "Person")
            if current_slot == 1
            else ("Person", "SourceEntry")
            for name in relations
        },
        policy_repo_path=root / "us",
        rulespec_roots=[root],
        request_only=True,
    )


@pytest.mark.parametrize("lengths", [(2,), (1, 1), (2, 0, 2), (1, 1, 1)])
@pytest.mark.parametrize("current_slot", [0, 1])
def test_anonymous_rows_are_distinct_across_relations(tmp_path, lengths, current_slot):
    request = _request(tmp_path, lengths, current_slot=current_slot)
    facts = request["dataset"]["inputs"]
    expected = [f"related_{i}" for i in range(sum(lengths))]
    assert [fact["entity_id"] for fact in facts] == expected
    assert {fact["entity"] for fact in facts} == {"SourceEntry"}
    assert [fact["value"]["value"] for fact in facts] == [
        100 * (i + 1) + j for i, length in enumerate(lengths) for j in range(length)
    ]
    edges = request["dataset"]["relations"]
    # Canonical and local relation aliases must refer to the same row.
    assert len(edges) == 2 * len(facts)
    for i, identity in enumerate(expected):
        canonical, local = edges[2 * i : 2 * i + 2]
        assert canonical["tuple"] == local["tuple"]
        assert canonical["tuple"][current_slot] == "case"
        assert canonical["tuple"][1 - current_slot] == identity


@pytest.mark.real_engine
@pytest.mark.parametrize("executor", ["cli", "pipeline"])
def test_distinct_relation_amounts_execute_without_coalescing(
    tmp_path, real_engine, executor
):
    root = tmp_path / "rulespec-us"
    module = root / "us" / "policies" / "rows.yaml"
    module.parent.mkdir(parents=True)
    prefix = "us:policies/rows#"
    rules = [
        {
            "name": name,
            "kind": "data_relation",
            "data_relation": {"arity": 2, "arguments": ["SourceEntry", "Person"]},
        }
        for name in ("shares", "bonds")
    ]
    for name, entity, formula in [
        ("gain", "SourceEntry", "proceeds - cost - expenses"),
        ("zero_gain", "SourceEntry", "(gain * eligible) / proceeds"),
        ("other_gain", "SourceEntry", "gain - zero_gain"),
        (
            "zero_total",
            "Person",
            "sum(shares.zero_gain) + sum(bonds.zero_gain)",
        ),
        (
            "other_total",
            "Person",
            "sum(shares.other_gain) + sum(bonds.other_gain)",
        ),
    ]:
        rules.append(
            {
                "name": name,
                "kind": "derived",
                "entity": entity,
                "dtype": "Money",
                "unit": "CAD",
                "period": "Year",
                "versions": [
                    {
                        "effective_from": "2025-01-01",
                        "effective_to": "2025-12-31",
                        "formula": formula,
                    }
                ],
            }
        )
    module.write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "units": [{"name": "CAD", "kind": "currency", "minor_units": 2}],
                "rules": rules,
            }
        )
    )
    cases = [
        {
            "name": "separate_shares_and_bonds",
            "period": {
                "period_kind": "tax_year",
                "start": "2025-01-01",
                "end": "2025-12-31",
            },
            "input": {
                prefix + "relation." + relation: [
                    {
                        prefix + "input." + name: value
                        for name, value in zip(
                            ("proceeds", "cost", "expenses", "eligible"), amounts
                        )
                    }
                ]
                for relation, amounts in [
                    ("shares", (10000, 6000, 1000, 7000)),
                    ("bonds", (3000, 1800, 0, 2000)),
                ]
            },
            "output": {prefix + "zero_total": 2900, prefix + "other_total": 1300},
        }
    ]
    test_file = module.with_suffix(".test.yaml")
    test_file.write_text(yaml.safe_dump(cases))
    artifact = tmp_path / "compiled.json"
    result = run_rulespec_compile(
        binary=real_engine.binary,
        program=module,
        rulespec_roots=[root],
        output=artifact,
        cwd=real_engine.root,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(artifact.read_text())
    if executor == "cli":
        result = _execute_rulespec_test_file(
            test_file,
            binary=real_engine.binary,
            axiom_rules_path=real_engine.root,
            env=os.environ.copy(),
            rulespec_roots=[root],
            tmp_path=tmp_path,
            compiled_cache={module: (artifact, payload)},
            policy_repo_path=root / "us",
        )
        assert result["cases"] == 1
        assert result["failures"] == []
    else:
        pipeline = ValidatorPipeline(
            root / "us",
            real_engine.root,
            axiom_rules_engine_ref=os.environ.get("AXIOM_RULES_ENGINE_REF"),
            local_corpus_release=None,
            enable_oracles=False,
            enforce_repository_layout=False,
        )
        assert (
            pipeline._run_rulespec_test_cases(
                rules_file=module,
                compiled_path=artifact,
                compiled_payload=payload,
                cases=cases,
            )
            == []
        )
