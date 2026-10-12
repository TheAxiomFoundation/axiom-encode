"""Companion fixtures must preserve independently declared entity scope."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.cli import _execute_rulespec_test_file
from axiom_encode.harness.validator_pipeline import ValidatorPipeline
from axiom_encode.rules_engine_compat import run_rulespec_compile

FIXTURES = Path(__file__).parent / "fixtures" / "companion_declared_entities"
PERIOD = {"period_kind": "month", "start": "2026-01-01", "end": "2026-01-31"}
MODULES = {
    "nm": ("us-nm", "7-2-39", "TaxUnit"),
    "ok": ("us-ok", "68-2906", "Household"),
}


@pytest.fixture(scope="session")
def entity_engine() -> tuple[Path, Path]:
    """Resolve an existing engine without building or crawling checkouts."""
    locator = shutil.which("axiom-locate")
    if locator is None:
        pytest.skip("axiom-locate is unavailable")
    located = subprocess.run(
        [locator, "engine"], capture_output=True, text=True, check=False, timeout=30
    )
    if located.returncode != 0:
        pytest.skip("a built axiom-rules-engine is unavailable")
    binary = Path(located.stdout.strip().splitlines()[0])
    engine_root = binary.parent
    if engine_root.name in {"release", "debug"} and engine_root.parent.name == "target":
        engine_root = engine_root.parent.parent
    return binary, engine_root


def _fixture_module(tmp_path: Path, family: str) -> tuple[Path, Path]:
    jurisdiction, section, _entity = MODULES[family]
    root = tmp_path / "rulespec-us"
    module = root / jurisdiction / "statutes" / f"{section}.yaml"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text((FIXTURES / f"{family}.yaml").read_text())
    module.with_suffix(".test.yaml").write_text(
        (FIXTURES / f"{family}.test.yaml").read_text()
    )
    return root, module


def _compile(
    binary: Path, engine_root: Path, root: Path, module: Path, artifact: Path
) -> dict:
    result = run_rulespec_compile(
        binary=binary,
        program=module,
        rulespec_roots=[root],
        output=artifact,
        cwd=engine_root,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(artifact.read_text())


def _companions(
    executor: str,
    *,
    binary: Path,
    engine_root: Path,
    root: Path,
    module: Path,
    artifact: Path,
) -> list[str]:
    payload = _compile(binary, engine_root, root, module, artifact)
    test_file = module.with_suffix(".test.yaml")
    if executor == "cli":
        result = _execute_rulespec_test_file(
            test_file,
            binary=binary,
            axiom_rules_path=engine_root,
            env=os.environ.copy(),
            rulespec_roots=[root],
            tmp_path=artifact.parent,
            compiled_cache={module: (artifact, payload)},
            policy_repo_path=root / module.relative_to(root).parts[0],
        )
        assert result["cases"] == len(yaml.safe_load(test_file.read_text()))
        return [failure["message"] for failure in result["failures"]]
    pipeline = ValidatorPipeline(
        policy_repo_path=root / module.relative_to(root).parts[0],
        axiom_rules_path=engine_root,
        local_corpus_release=None,
        enable_oracles=False,
        enforce_repository_layout=False,
    )
    return pipeline._run_rulespec_test_cases(
        rules_file=module,
        compiled_path=artifact,
        compiled_payload=payload,
        cases=yaml.safe_load(test_file.read_text()),
    )


@pytest.mark.parametrize("executor", ["cli", "validator"])
@pytest.mark.parametrize("family", ["nm", "ok"])
def test_cross_entity_companion_reports_missing_relation(
    tmp_path, entity_engine, executor, family
):
    binary, engine_root = entity_engine
    root, module = _fixture_module(tmp_path, family)
    issues = _companions(
        executor,
        binary=binary,
        engine_root=engine_root,
        root=root,
        module=module,
        artifact=tmp_path / "compiled.json",
    )
    declaration = yaml.safe_load(module.read_text())
    input_name = declaration["inputs"][0]["name"]
    output_name = declaration["rules"][0]["name"]
    message = "\n".join(issues)
    assert issues, "the old executor silently moves Person facts onto the output entity"
    for required in (input_name, output_name, "Person", MODULES[family][2], "relation"):
        assert required in message


@pytest.mark.parametrize("family", ["nm", "ok"])
def test_rehoming_conceals_real_engine_missing_input(tmp_path, entity_engine, family):
    """Reproduce the false green independently of companion fixture construction."""
    binary, engine_root = entity_engine
    root, module = _fixture_module(tmp_path, family)
    artifact = tmp_path / "compiled.json"
    payload = _compile(binary, engine_root, root, module, artifact)
    case = yaml.safe_load(module.with_suffix(".test.yaml").read_text())[0]
    output_id = payload["program"]["derived"][0]["id"]
    interval = {"start": PERIOD["start"], "end": PERIOD["end"]}

    def execute(entity: str, entity_id: str) -> subprocess.CompletedProcess:
        request = {
            "mode": "explain",
            "dataset": {
                "inputs": [
                    {
                        "name": name,
                        "entity": entity,
                        "entity_id": entity_id,
                        "interval": interval,
                        "value": {"kind": "bool", "value": value},
                    }
                    for name, value in case["input"].items()
                ],
                "relations": [],
            },
            "queries": [
                {"entity_id": "owner-1", "period": PERIOD, "outputs": [output_id]}
            ],
        }
        return subprocess.run(
            [str(binary), "run-compiled", "--artifact", str(artifact)],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )

    rehomed = execute(MODULES[family][2], "owner-1")
    assert rehomed.returncode == 0, rehomed.stderr
    output = json.loads(rehomed.stdout)["results"][0]["outputs"][output_id]
    assert output["outcome"] == "holds"
    scoped = execute("Person", "person-1")
    assert scoped.returncode != 0
    assert "missing input" in scoped.stderr.lower()
    assert "owner-1" in scoped.stderr


@pytest.mark.parametrize("executor", ["cli", "validator"])
def test_same_entity_companions_remain_compatible(tmp_path, entity_engine, executor):
    binary, engine_root = entity_engine
    root, module = _fixture_module(tmp_path, "nm")
    module.write_text(module.read_text().replace("entity: TaxUnit", "entity: Person"))
    assert not _companions(
        executor,
        binary=binary,
        engine_root=engine_root,
        root=root,
        module=module,
        artifact=tmp_path / "compiled.json",
    )


@pytest.mark.parametrize("executor", ["cli", "validator"])
def test_mixed_output_entities_request_separate_cases(
    tmp_path, entity_engine, executor
):
    binary, engine_root = entity_engine
    root, module = _fixture_module(tmp_path, "nm")
    content = yaml.safe_load(module.read_text())
    person_rule = dict(content["rules"][0])
    person_rule.update(name="person_admission", entity="Person")
    content["rules"].append(person_rule)
    module.write_text(yaml.safe_dump(content))
    cases = yaml.safe_load(module.with_suffix(".test.yaml").read_text())
    cases[0]["output"]["us-nm:statutes/7-2-39#person_admission"] = "holds"
    module.with_suffix(".test.yaml").write_text(yaml.safe_dump(cases))
    issues = _companions(
        executor,
        binary=binary,
        engine_root=engine_root,
        root=root,
        module=module,
        artifact=tmp_path / "compiled.json",
    )
    assert any("mixes derived output entities" in issue for issue in issues)
    assert any("separate test cases" in issue for issue in issues)


@pytest.mark.parametrize("executor", ["cli", "validator"])
def test_unused_foreign_flat_fact_does_not_reject_companion(
    tmp_path, entity_engine, executor
):
    binary, engine_root = entity_engine
    root, module = _fixture_module(tmp_path, "nm")
    content = yaml.safe_load(module.read_text())
    content["rules"][0]["versions"][0]["formula"] = (
        "1 == 1 or claimable_as_dependent_on_another_return"
    )
    module.write_text(yaml.safe_dump(content))
    assert not _companions(
        executor,
        binary=binary,
        engine_root=engine_root,
        root=root,
        module=module,
        artifact=tmp_path / "compiled.json",
    )


@pytest.mark.parametrize("executor", ["cli", "validator"])
def test_flat_input_keeps_its_entity_when_relation_is_declared(
    tmp_path, entity_engine, executor, monkeypatch
):
    """A relation declaration cannot excuse reading its fact on the owner id."""
    binary, engine_root = entity_engine
    root, module = _fixture_module(tmp_path, "nm")
    content = yaml.safe_load(module.read_text())
    content["rules"].append(
        {
            "name": "member_of_tax_unit",
            "kind": "data_relation",
            "data_relation": {"arity": 2, "arguments": ["Person", "TaxUnit"]},
        }
    )
    module.write_text(yaml.safe_dump(content))
    requests = []
    original_run = subprocess.run

    def capture_run(command, **kwargs):
        if "run-compiled" in command:
            requests.append(json.loads(kwargs["input"]))
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", capture_run)
    assert _companions(
        executor,
        binary=binary,
        engine_root=engine_root,
        root=root,
        module=module,
        artifact=tmp_path / "compiled.json",
    )
    assert len(requests) == 1
    request = requests[0]
    fact = request["dataset"]["inputs"][0]
    assert fact["entity"] == "Person"
    assert fact["entity_id"] != request["queries"][0]["entity_id"]


@pytest.mark.parametrize("executor", ["cli", "validator"])
@pytest.mark.parametrize("output_entity", ["Person", "Household"])
def test_imported_input_uses_its_declaring_modules_entity(
    tmp_path, entity_engine, executor, output_entity
):
    binary, engine_root = entity_engine
    root = tmp_path / "rulespec-us"
    folder = root / "us" / "policies" / "example"
    folder.mkdir(parents=True)
    producer = folder / "person-facts.yaml"
    producer.write_text(
        """format: rulespec/v1
inputs:
  - name: person_is_eligible
    entity: Person
    dtype: Boolean
rules:
  - name: eligible_person
    kind: derived
    entity: Person
    dtype: Judgment
    versions:
      - effective_from: '2026-01-01'
        formula: person_is_eligible
"""
    )
    module = folder / "consumer.yaml"
    module.write_text(
        """format: rulespec/v1
imports:
  - us:policies/example/person-facts
rules:
  - name: claimant_eligible
    kind: derived
    entity: OUTPUT_ENTITY
    dtype: Judgment
    versions:
      - effective_from: '2026-01-01'
        formula: eligible_person
""".replace("OUTPUT_ENTITY", output_entity)
    )
    module.with_suffix(".test.yaml").write_text(
        """- name: imported_person_fact
  period: 2026-01
  input:
    us:policies/example/person-facts#input.person_is_eligible: true
  output:
    us:policies/example/consumer#claimant_eligible: holds
"""
    )
    issues = _companions(
        executor,
        binary=binary,
        engine_root=engine_root,
        root=root,
        module=module,
        artifact=tmp_path / "compiled.json",
    )
    if output_entity == "Person":
        assert not issues
    else:
        assert issues
        message = "\n".join(issues)
        for required in ("person_is_eligible", "Person", "Household", "relation"):
            assert required in message


@pytest.mark.parametrize("executor", ["cli", "validator"])
@pytest.mark.parametrize("owner_first", [False, True])
def test_declared_relation_traversal_still_passes(
    tmp_path, entity_engine, executor, owner_first
):
    binary, engine_root = entity_engine
    root = tmp_path / "rulespec-us"
    module = root / "us" / "policies" / "example" / "members.yaml"
    module.parent.mkdir(parents=True)
    slots = ["Household", "Person"] if owner_first else ["Person", "Household"]
    module.write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "inputs": [
                    {
                        "name": "person_is_eligible",
                        "entity": "Person",
                        "dtype": "Boolean",
                    }
                ],
                "rules": [
                    {
                        "name": "member_of_household",
                        "kind": "data_relation",
                        "data_relation": {"arity": 2, "arguments": slots},
                    },
                    {
                        "name": "eligible_person",
                        "kind": "derived",
                        "entity": "Person",
                        "dtype": "Judgment",
                        "versions": [
                            {
                                "effective_from": "2026-01-01",
                                "formula": "person_is_eligible",
                            }
                        ],
                    },
                    {
                        "name": "eligible_member_count",
                        "kind": "derived",
                        "entity": "Household",
                        "dtype": "Integer",
                        "versions": [
                            {
                                "effective_from": "2026-01-01",
                                "formula": "count_where(member_of_household, eligible_person)",
                            }
                        ],
                    },
                ],
            }
        )
    )
    module.with_suffix(".test.yaml").write_text(
        """- name: counts_related_people
  period: 2026-01
  input:
    us:policies/example/members#relation.member_of_household:
      - us:policies/example/members#input.person_is_eligible: true
      - us:policies/example/members#input.person_is_eligible: false
  output:
    us:policies/example/members#eligible_member_count: 1
"""
    )
    assert not _companions(
        executor,
        binary=binary,
        engine_root=engine_root,
        root=root,
        module=module,
        artifact=tmp_path / "compiled.json",
    )


@settings(
    max_examples=18,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    input_entity=st.sampled_from(["Person", "TaxUnit", "Household"]),
    output_entity=st.sampled_from(["Person", "TaxUnit", "Household"]),
    relation_present=st.booleans(),
)
def test_flat_companion_pass_implies_correctly_scoped_engine_pass(
    tmp_path, entity_engine, input_entity, output_entity, relation_present
):
    """Use an independent dataset for every generated scope/relation choice."""
    binary, engine_root = entity_engine
    with tempfile.TemporaryDirectory(dir=tmp_path) as example_tmp:
        work = Path(example_tmp)
        root = work / "rulespec-us"
        module = root / "us" / "policies" / "example" / "flat.yaml"
        module.parent.mkdir(parents=True)
        traverses_relation = relation_present and input_entity != output_entity
        rules = [
            {
                "name": "claimant_eligible",
                "kind": "derived",
                "entity": output_entity,
                "dtype": "Judgment",
                "versions": [
                    {
                        "effective_from": "2026-01-01",
                        "formula": (
                            "count_where(member_of_owner, related_person_eligible) > 0"
                            if traverses_relation
                            else "person_is_eligible"
                        ),
                    }
                ],
            }
        ]
        if relation_present:
            rules.append(
                {
                    "name": "member_of_owner",
                    "kind": "data_relation",
                    "data_relation": {
                        "arity": 2,
                        "arguments": [input_entity, output_entity],
                    },
                }
            )
        if traverses_relation:
            rules.append(
                {
                    "name": "related_person_eligible",
                    "kind": "derived",
                    "entity": input_entity,
                    "dtype": "Judgment",
                    "versions": [
                        {
                            "effective_from": "2026-01-01",
                            "formula": "person_is_eligible",
                        }
                    ],
                }
            )
        module.write_text(
            yaml.safe_dump(
                {
                    "format": "rulespec/v1",
                    "inputs": [
                        {
                            "name": "person_is_eligible",
                            "entity": input_entity,
                            "dtype": "Boolean",
                        }
                    ],
                    "rules": rules,
                }
            )
        )
        output_id = "us:policies/example/flat#claimant_eligible"
        input_id = "us:policies/example/flat#input.person_is_eligible"
        relation_id = "us:policies/example/flat#relation.member_of_owner"
        companion_input = {input_id: True}
        if traverses_relation:
            companion_input[relation_id] = [{input_id: True}]
        module.with_suffix(".test.yaml").write_text(
            yaml.safe_dump(
                [
                    {
                        "name": "flat_fact",
                        "period": "2026-01",
                        "input": companion_input,
                        "output": {output_id: "holds"},
                    }
                ]
            )
        )
        artifact = work / "compiled.json"
        _compile(binary, engine_root, root, module, artifact)
        query_id = f"query-{output_entity}"
        fact_id = query_id if input_entity == output_entity else f"fact-{input_entity}"
        interval = {"start": PERIOD["start"], "end": PERIOD["end"]}
        direct_inputs = [
            {
                "name": input_id,
                "entity": input_entity,
                "entity_id": fact_id,
                "interval": interval,
                "value": {"kind": "bool", "value": True},
            }
        ]
        direct_relations = []
        if traverses_relation:
            related_id = f"related-{input_entity}"
            direct_inputs.append({**direct_inputs[0], "entity_id": related_id})
            direct_relations.append(
                {
                    "name": relation_id,
                    "tuple": [related_id, query_id],
                    "interval": interval,
                }
            )
        request = {
            "mode": "explain",
            "dataset": {
                "inputs": direct_inputs,
                "relations": direct_relations,
            },
            "queries": [
                {"entity_id": query_id, "period": PERIOD, "outputs": [output_id]}
            ],
        }
        direct = subprocess.run(
            [str(binary), "run-compiled", "--artifact", str(artifact)],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        direct_passes = direct.returncode == 0
        if direct_passes:
            actual = json.loads(direct.stdout)["results"][0]["outputs"][output_id]
            direct_passes = actual.get("outcome") == "holds"
        assert direct_passes == (input_entity == output_entity or relation_present), (
            direct.stderr or direct.stdout
        )
        for executor in ("cli", "validator"):
            issues = _companions(
                executor,
                binary=binary,
                engine_root=engine_root,
                root=root,
                module=module,
                artifact=artifact,
            )
            assert (not issues) == direct_passes


@pytest.mark.parametrize("executor", ["cli", "validator"])
def test_table_query_id_cannot_rehome_foreign_flat_fact(
    tmp_path, entity_engine, executor
):
    binary, engine_root = entity_engine
    root, module = _fixture_module(tmp_path, "nm")
    cases = yaml.safe_load(module.with_suffix(".test.yaml").read_text())
    # The engine indexes facts by name and ID, not their entity label. A
    # table's explicit owner ID must not claim the generated Person ID.
    cases[0]["id"] = "case"
    cases[0]["tables"] = {"TaxUnit": [{"id": "case:entity:Person"}]}
    cases[0]["output"] = {
        output: [expected] for output, expected in cases[0]["output"].items()
    }
    module.with_suffix(".test.yaml").write_text(yaml.safe_dump(cases))
    issues = _companions(
        executor,
        binary=binary,
        engine_root=engine_root,
        root=root,
        module=module,
        artifact=tmp_path / "compiled.json",
    )
    assert issues, "a table query must not alias a foreign flat fact's ID"
    for required in (
        "claimable_as_dependent_on_another_return",
        "Person",
        "TaxUnit",
        "relation",
    ):
        assert required in "\n".join(issues)


@pytest.mark.parametrize("include_table", [False, True])
def test_foreign_flat_fact_can_share_an_explicit_same_kind_member_id(
    tmp_path, entity_engine, include_table
):
    binary, engine_root = entity_engine
    root, module = _fixture_module(tmp_path, "nm")
    content = yaml.safe_load(module.read_text())
    person_rule = dict(content["rules"][0])
    person_rule.update(name="eligible_person", entity="Person")
    content["rules"][0]["versions"][0]["formula"] = (
        "count_where(member_of_owner, eligible_person) > 0"
    )
    # Copy the versions as well: the helper evaluates the original Person fact.
    person_rule["versions"] = [
        {
            "effective_from": "2026-01-01",
            "formula": "claimable_as_dependent_on_another_return",
        }
    ]
    content["rules"].extend(
        [
            person_rule,
            {
                "name": "member_of_owner",
                "kind": "data_relation",
                "data_relation": {"arity": 2, "arguments": ["Person", "TaxUnit"]},
            },
        ]
    )
    module.write_text(yaml.safe_dump(content))
    cases = yaml.safe_load(module.with_suffix(".test.yaml").read_text())
    cases[0]["id"] = "case"
    cases[0]["input"]["us-nm:statutes/7-2-39#relation.member_of_owner"] = [
        {"id": "case:entity:Person"}
    ]
    if include_table:
        cases[0]["tables"] = {"Person": [{"id": "case:entity:Person"}]}
    module.with_suffix(".test.yaml").write_text(yaml.safe_dump(cases))
    assert not _companions(
        "validator",
        binary=binary,
        engine_root=engine_root,
        root=root,
        module=module,
        artifact=tmp_path / "compiled.json",
    )


@pytest.mark.parametrize("executor", ["cli", "validator"])
def test_imported_input_consumer_alias_uses_declaring_entity(
    tmp_path, entity_engine, executor
):
    binary, engine_root = entity_engine
    root = tmp_path / "rulespec-us"
    folder = root / "us" / "policies" / "example"
    folder.mkdir(parents=True)
    producer = folder / "person-facts.yaml"
    producer.write_text(
        """format: rulespec/v1
inputs:
  - name: person_is_eligible
    entity: Person
    dtype: Boolean
rules:
  - name: eligible_person
    kind: derived
    entity: Person
    dtype: Judgment
    versions:
      - effective_from: '2026-01-01'
        formula: person_is_eligible
"""
    )
    module = folder / "consumer.yaml"
    module.write_text(
        """format: rulespec/v1
imports:
  - us:policies/example/person-facts
rules:
  - name: claimant_eligible
    kind: derived
    entity: Household
    dtype: Judgment
    versions:
      - effective_from: '2026-01-01'
        formula: person_is_eligible
"""
    )
    module.with_suffix(".test.yaml").write_text(
        """- name: imported_person_fact_consumer_alias
  period: 2026-01
  input:
    us:policies/example/consumer#input.person_is_eligible: true
  output:
    us:policies/example/consumer#claimant_eligible: holds
"""
    )
    issues = _companions(
        executor,
        binary=binary,
        engine_root=engine_root,
        root=root,
        module=module,
        artifact=tmp_path / "compiled.json",
    )
    assert issues, "a consumer alias must not rehome an imported Person fact"
    for required in ("person_is_eligible", "Person", "Household", "relation"):
        assert required in "\n".join(issues)


@pytest.mark.parametrize(
    ("executor", "fixture_kind"),
    [("cli", "table"), ("validator", "table"), ("validator", "relation")],
)
def test_explicit_foreign_row_id_cannot_alias_query_entity(
    tmp_path, entity_engine, executor, fixture_kind
):
    """ID-only engine lookup must not let a Person row impersonate a TaxUnit."""
    binary, engine_root = entity_engine
    root, module = _fixture_module(tmp_path, "nm")
    cases = yaml.safe_load(module.with_suffix(".test.yaml").read_text())
    cases[0]["id"] = "case"
    input_key = next(iter(cases[0]["input"]))
    foreign_row = {"id": "case", input_key: True}
    if fixture_kind == "table":
        cases[0]["tables"] = {"Person": [foreign_row]}
    else:
        content = yaml.safe_load(module.read_text())
        content["rules"].append(
            {
                "name": "member_of_tax_unit",
                "kind": "data_relation",
                "data_relation": {"arity": 2, "arguments": ["Person", "TaxUnit"]},
            }
        )
        module.write_text(yaml.safe_dump(content))
        cases[0]["input"]["us-nm:statutes/7-2-39#relation.member_of_tax_unit"] = [
            foreign_row
        ]
    module.with_suffix(".test.yaml").write_text(yaml.safe_dump(cases))
    issues = _companions(
        executor,
        binary=binary,
        engine_root=engine_root,
        root=root,
        module=module,
        artifact=tmp_path / "compiled.json",
    )
    assert issues, (
        "an explicit Person row must not supply facts at the TaxUnit query ID"
    )
    for required in ("Person", "TaxUnit", "case", "distinct"):
        assert required in "\n".join(issues)


@pytest.mark.parametrize("executor", ["cli", "validator"])
@pytest.mark.parametrize("include_late_table", [False, True])
def test_generated_related_ids_preserve_distinct_entity_kinds(
    tmp_path, entity_engine, executor, include_late_table, monkeypatch
):
    """Allocate compact relation IDs after all explicit fixture IDs are known."""
    binary, engine_root = entity_engine
    root, module = _fixture_module(tmp_path, "nm")
    content = yaml.safe_load(module.read_text())
    person_input = content["inputs"][0]["name"]
    content["inputs"].append(
        {"name": "household_is_eligible", "entity": "Household", "dtype": "Boolean"}
    )
    content["rules"] = [
        {
            "name": relation,
            "kind": "data_relation",
            "data_relation": {"arity": 2, "arguments": [entity, "TaxUnit"]},
        }
        for relation, entity in (
            ("member_of_tax_unit", "Person"),
            ("household_for_tax_unit", "Household"),
        )
    ] + [
        {
            "name": name,
            "kind": "derived",
            "entity": entity,
            "dtype": dtype,
            "versions": [{"effective_from": "2026-01-01", "formula": formula}],
        }
        for name, entity, dtype, formula in (
            ("eligible_person", "Person", "Judgment", person_input),
            ("eligible_household", "Household", "Judgment", "household_is_eligible"),
            (
                "eligible_related_entity_count",
                "TaxUnit",
                "Integer",
                "count_where(member_of_tax_unit, eligible_person) "
                "+ count_where(household_for_tax_unit, eligible_household)",
            ),
        )
    ]
    module.write_text(yaml.safe_dump(content))
    person_relation = "us-nm:statutes/7-2-39#relation.member_of_tax_unit"
    household_relation = "us-nm:statutes/7-2-39#relation.household_for_tax_unit"
    case = {
        "name": "counts_distinct_related_entity_kinds",
        "id": "case",
        "period": "2026-01",
        "input": {
            person_relation: [{f"us-nm:statutes/7-2-39#input.{person_input}": True}],
            household_relation: [
                {"us-nm:statutes/7-2-39#input.household_is_eligible": True}
            ],
        },
        "output": {"us-nm:statutes/7-2-39#eligible_related_entity_count": 2},
    }
    table_id = "related_0" if executor == "cli" else f"case-{person_relation}-1"
    if include_late_table:
        case["tables"] = {"Household": [{"id": table_id}]}
    module.with_suffix(".test.yaml").write_text(yaml.safe_dump([case]))
    requests = []
    original_run = subprocess.run

    def capture_run(command, **kwargs):
        if "run-compiled" in command:
            requests.append(json.loads(kwargs["input"]))
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", capture_run)
    assert not _companions(
        executor,
        binary=binary,
        engine_root=engine_root,
        root=root,
        module=module,
        artifact=tmp_path / "compiled.json",
    )
    assert len(requests) == 1
    request = requests[0]
    kinds_by_id = {request["queries"][0]["entity_id"]: {"TaxUnit"}}
    if include_late_table:
        kinds_by_id.setdefault(table_id, set()).add("Household")
    for fact in request["dataset"]["inputs"]:
        kinds_by_id.setdefault(fact["entity_id"], set()).add(fact["entity"])
    assert all(len(kinds) == 1 for kinds in kinds_by_id.values())
    assert {fact["entity"] for fact in request["dataset"]["inputs"]} == {
        "Person",
        "Household",
    }


@pytest.mark.parametrize(
    ("executor", "scenario"),
    [
        ("cli", "case_sensitive"),
        ("validator", "case_sensitive"),
        ("validator", "query_root"),
        ("cli", "queried_table"),
        ("validator", "queried_table"),
    ],
)
def test_generated_table_ids_preserve_distinct_entity_kinds(
    tmp_path, entity_engine, executor, scenario, monkeypatch
):
    """Generated table IDs stay distinct and row queries use their allocated IDs."""
    binary, engine_root = entity_engine
    root, module = _fixture_module(tmp_path, "nm")
    content = yaml.safe_load(module.read_text())
    cases = yaml.safe_load(module.with_suffix(".test.yaml").read_text())
    input_key = next(iter(cases[0]["input"]))
    query_entity = "AB" if scenario == "queried_table" else "TaxUnit"
    content["inputs"][0]["entity"] = query_entity
    content["rules"][0]["entity"] = query_entity

    def fact(name, entity):
        content["inputs"].append({"name": name, "entity": entity, "dtype": "Boolean"})
        content["rules"].append(
            {
                "name": f"declared_{name}",
                "kind": "derived",
                "entity": entity,
                "dtype": "Judgment",
                "versions": [{"effective_from": "2026-01-01", "formula": name}],
            }
        )
        return f"us-nm:statutes/7-2-39#input.{name}"

    if scenario == "case_sensitive":
        cases[0]["tables"] = {
            "AB": [{fact("upper_kind_fact", "AB"): True}],
            "Ab": [{fact("title_kind_fact", "Ab"): True}],
        }
    elif scenario == "query_root":
        cases[0]["tables"] = {"Case": [{fact("case_kind_fact", "Case"): True}]}
    else:
        cases[0]["input"] = {}
        cases[0]["tables"] = {
            "Ab": [{"id": "ab-1", fact("title_kind_fact", "Ab"): True}],
            "AB": [{input_key: True}],
        }
        cases[0]["output"] = {
            output: [expected] for output, expected in cases[0]["output"].items()
        }
    module.write_text(yaml.safe_dump(content))
    module.with_suffix(".test.yaml").write_text(yaml.safe_dump(cases))
    requests = []
    original_run = subprocess.run

    def capture_run(command, **kwargs):
        if "run-compiled" in command:
            requests.append(json.loads(kwargs["input"]))
        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", capture_run)
    assert not _companions(
        executor,
        binary=binary,
        engine_root=engine_root,
        root=root,
        module=module,
        artifact=tmp_path / "compiled.json",
    )
    assert len(requests) == 1
    request = requests[0]
    query_id = request["queries"][0]["entity_id"]
    kinds_by_id = {query_id: {query_entity}}
    facts_by_entity = {}
    for record in request["dataset"]["inputs"]:
        kinds_by_id.setdefault(record["entity_id"], set()).add(record["entity"])
        facts_by_entity[record["entity"]] = record
    assert all(len(kinds) == 1 for kinds in kinds_by_id.values())
    if scenario == "case_sensitive":
        assert facts_by_entity["AB"]["entity_id"] == "ab-1"
        assert facts_by_entity["Ab"]["entity_id"] != "ab-1"
    elif scenario == "query_root":
        assert query_id == "case-1"
        assert facts_by_entity["Case"]["entity_id"] != query_id
    else:
        assert facts_by_entity["Ab"]["entity_id"] == "ab-1"
        assert query_id == facts_by_entity["AB"]["entity_id"] != "ab-1"
