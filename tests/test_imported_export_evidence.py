"""Canonical imported observations retain provider ownership without evaluation."""

import copy
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness.imported_export_evidence import (
    ResolvedImportedExport,
    bind_entered_observations,
)
from axiom_encode.harness.validator_pipeline import ValidatorPipeline

BASE = "ca/policy/forms/annual"
URI = "ca:policies/forms/annual/step-2#observation"
FIXTURES = Path(__file__).parent / "fixtures/form-numeric-locators"


def dump(value):
    return yaml.safe_dump(value, sort_keys=False)


def atom(path, kind, citation, excerpt):
    return {
        "path": path,
        "kind": kind,
        "source": {"corpus_citation_path": citation, "excerpt": excerpt},
    }


def frame():
    return [
        atom(
            f"versions[0].{field}",
            "effective_period",
            BASE + "/primary-document",
            "T1 2031",
        )
        for field in ("effective_from", "effective_to")
    ]


@pytest.fixture
def world(tmp_path):
    context = {
        BASE + f"/step-{n}": (FIXTURES / f"step-{n}.txt").read_text() for n in (2, 3)
    }
    context[BASE + "/primary-document"] = "T1 2031\nIncome Tax and Benefit Return\n"
    row = next(
        x
        for x in context[BASE + "/step-2"].splitlines()
        if x.startswith("Line ") and "line 11900:" in x
    )
    bullet = next(
        x[2:]
        for x in context[BASE + "/step-3"].splitlines()
        if x.startswith("- You entered") and "11900" in x
    )
    version = {
        "effective_from": "2031-01-01",
        "effective_to": "2031-12-31",
        "formula": "observed",
    }
    provider = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": BASE + "/step-2"}},
        "inputs": [
            {"name": "observed", "dtype": "Bool", "entity": "Person", "period": "Year"}
        ],
        "rules": [
            {
                "name": "observation",
                "kind": "derived",
                "dtype": "Judgment",
                "entity": "Person",
                "period": "Year",
                "versions": [version],
                "metadata": {
                    "proof": {
                        "atoms": frame()
                        + [
                            atom(
                                "versions[0].formula", "formula", BASE + "/step-2", row
                            ),
                            atom(
                                "versions[0].formula",
                                "predicate",
                                BASE + "/step-3",
                                bullet,
                            ),
                        ]
                    }
                },
            }
        ],
    }
    consumer = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": BASE + "/step-3"}},
        "imports": [URI],
        "rules": [
            {
                "name": "repayment",
                "kind": "derived",
                "dtype": "Money",
                "entity": "Person",
                "period": "Year",
                "unit": "CAD",
                "versions": [{**version, "formula": "0"}],
                "metadata": {"proof": {"atoms": frame()}},
            }
        ],
    }
    root = tmp_path / "rulespec-ca/ca"
    path = root / "policies/forms/annual/step-2.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(dump(provider))
    return consumer, provider, context, root, path


def records(world):
    consumer, provider, _, root, path = world
    content = dump(provider)
    return (
        ResolvedImportedExport(
            URI,
            URI,
            "observation",
            str(path),
            str(root),
            hashlib.sha256(content.encode()).hexdigest(),
            content,
            tuple(consumer["imports"]),
        ),
    )


def bind(world, record=None):
    consumer, _, context, _, _ = world
    return bind_entered_observations(
        dump(consumer),
        owner="repayment",
        consumer_citation=BASE + "/step-3",
        source_context=context,
        imported_exports=records(world) if record is None else record,
    )


def test_direct_role_uses_source_coordinate_not_symbol_name(world):
    result = bind(world)
    assert len(result) == 1
    assert result[0].coordinate == 11900
    assert result[0].input_uri.endswith("#observed")
    assert result[0].versions[0].year == 2031
    evidence = result[0].versions[0]
    assert world[2][evidence.row_citation][slice(*evidence.row_span)].startswith(
        "Line "
    )
    assert world[2][evidence.predicate_citation][
        slice(*evidence.predicate_span)
    ].startswith("You entered")


@pytest.mark.parametrize(
    "change",
    [
        "computed",
        "default",
        "wrong_type",
        "wrong_entity",
        "wrong_period",
        "unit",
        "missing_predicate",
        "wrong_path",
        "wrong_role",
        "missing_date",
        "wrong_year",
        "extra_version",
        "duplicate_input",
        "input_rule_collision",
    ],
)
def test_provider_refusals(world, change):
    provider = world[1]
    rule = provider["rules"][0]
    atoms = rule["metadata"]["proof"]["atoms"]
    if change == "computed":
        rule["versions"][0]["formula"] = "not observed"
    elif change == "default":
        provider["inputs"][0]["default"] = False
    elif change == "wrong_type":
        provider["inputs"][0]["dtype"] = "Money"
    elif change == "wrong_entity":
        provider["inputs"][0]["entity"] = "Family"
    elif change == "wrong_period":
        provider["inputs"][0]["period"] = "Month"
    elif change == "unit":
        provider["inputs"][0]["unit"] = "CAD"
    elif change == "missing_predicate":
        atoms.pop()
    elif change == "wrong_path":
        atoms[-1]["path"] = "versions[1].formula"
    elif change == "wrong_role":
        atoms[-1]["source"]["excerpt"] = next(
            x[2:]
            for x in world[2][BASE + "/step-3"].splitlines()
            if x.startswith("- You entered") and "11300" in x
        )
    elif change == "missing_date":
        atoms.pop(0)
    elif change == "wrong_year":
        rule["versions"][0]["effective_to"] = "2032-12-31"
    elif change == "extra_version":
        rule["versions"].append(copy.deepcopy(rule["versions"][0]))
    elif change == "duplicate_input":
        provider["inputs"].append(copy.deepcopy(provider["inputs"][0]))
    elif change == "input_rule_collision":
        provider["inputs"][0]["name"] = "observation"
    assert not bind(world)


@pytest.mark.parametrize(
    "change",
    [
        "digest",
        "path",
        "canonical",
        "consumer_imports",
        "duplicate_record",
        "foreign_import",
        "local_alias",
        "duplicate_alias",
    ],
)
def test_identity_and_current_consumer_refusals(world, change):
    record = records(world)[0]
    if change == "digest":
        record = replace(record, content_sha256="0" * 64)
    elif change == "path":
        record = replace(
            record, resolved_path=record.resolved_path.replace("step-2", "step-9")
        )
    elif change == "canonical":
        record = replace(record, canonical_target=URI.replace("ca:", "us:"))
    elif change == "consumer_imports":
        world[0]["imports"] = [URI.replace("step-2", "step-9")]
    elif change == "duplicate_record":
        assert not bind(world, (record, record))
        return
    elif change == "foreign_import":
        world[0]["imports"] = [URI.replace("ca:", "us:")]
    elif change == "local_alias":
        world[0]["inputs"] = [{"name": "observation", "dtype": "Bool"}]
    elif change == "duplicate_alias":
        world[0]["imports"].append(URI.replace("step-2", "step-9"))
    assert not bind(world, (record,))


def test_normal_collector_retains_canonical_file_and_old_projection(world):
    consumer, provider, _, root, path = world
    consumer_path = path.with_name("step-3.yaml")
    consumer_path.write_text(dump(consumer))
    pipeline = ValidatorPipeline(
        root, root.parent / "engine", local_corpus_release=None, enable_oracles=False
    )
    found = pipeline._complete_source_unit_import_bindings(consumer_path)
    assert len(found) == 1
    assert found[0].canonical_target == URI
    assert found[0].resolved_path == str(path)
    assert found[0].content_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert bind(world, found)
    assert pipeline._complete_source_unit_import_symbol_contents(consumer_path) == (
        ("observation", dump(provider)),
    )


def test_shared_bool_cannot_prove_two_distinct_entered_roles(world):
    consumer, provider, context, _, _ = world
    second = copy.deepcopy(provider["rules"][0])
    second["name"] = "other_observation"
    row = next(
        x
        for x in context[BASE + "/step-2"].splitlines()
        if x.startswith("Line ") and "line 11300:" in x
    )
    bullet = next(
        x[2:]
        for x in context[BASE + "/step-3"].splitlines()
        if x.startswith("- You entered") and "11300" in x
    )
    second["metadata"]["proof"]["atoms"][-2:] = [
        atom("versions[0].formula", "formula", BASE + "/step-2", row),
        atom("versions[0].formula", "predicate", BASE + "/step-3", bullet),
    ]
    provider["rules"].append(second)
    other_uri = URI.replace("#observation", "#other_observation")
    consumer["imports"].append(other_uri)
    first = records(world)[0]
    other = replace(
        first,
        import_uri=other_uri,
        canonical_target=other_uri,
        export_name="other_observation",
    )
    assert bind(world, (first,))
    assert bind(world, (other,))
    assert not bind(world, (first, other))
    assert not bind(world, (other, first))


def test_provider_import_alias_cannot_shadow_direct_observation(world):
    world[1]["imports"] = [URI.replace("#observation", "#observed")]
    assert not bind(world)


def test_supplied_consumer_primary_identity_must_match_context(world):
    world[0]["module"]["source_verification"]["corpus_citation_path"] = BASE + "/step-9"
    assert not bind(world)


def test_duplicate_yaml_mapping_is_not_an_observation(world):
    record = records(world)[0]
    content = record.content + "inputs: []\n"
    record = replace(
        record,
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
    )
    assert not bind(world, (record,))


def test_conflicting_provider_snapshots_and_paths_refuse(world):
    world[0]["imports"].append(URI.replace("#observation", "#other"))
    first = records(world)[0]
    content = first.content + "description: different snapshot\n"
    other = replace(
        first,
        import_uri=world[0]["imports"][1],
        canonical_target=world[0]["imports"][1],
        export_name="other",
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
    )
    assert not bind(world, (first, other))
    assert not bind(world, (other, first))
    other = replace(
        other,
        content=first.content,
        content_sha256=first.content_sha256,
        resolved_path=first.resolved_path.replace("step-2.yaml", "different.yaml"),
    )
    assert not bind(world, (first, other))


def test_collector_reads_shared_provider_once(world, monkeypatch):
    consumer, provider, _, root, path = world
    consumer["imports"].append(URI.replace("#observation", "#other"))
    consumer_path = path.with_name("step-3.yaml")
    consumer_path.write_text(dump(consumer))
    original = Path.read_bytes
    reads = []

    def tracked(self):
        if self == path:
            reads.append(self)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", tracked)
    pipeline = ValidatorPipeline(
        root, root.parent / "engine", local_corpus_release=None, enable_oracles=False
    )
    found = pipeline._complete_source_unit_import_bindings(consumer_path)
    assert len(found) == 2
    assert reads == [path]
    assert found[0].content == found[1].content


@pytest.mark.parametrize(
    "mutation",
    [
        "quoted_header",
        "quoted_bullets",
        "duplicate_body",
        "joined_condition",
        "absent_context",
        "wrong_frame_source",
    ],
)
def test_source_refusals(world, mutation):
    consumer, provider, context, _, _ = world
    citation = BASE + "/step-3"
    if mutation == "quoted_header":
        context[BASE + "/primary-document"] = '"T1 2031"\n'
    elif mutation == "quoted_bullets":
        context[citation] = (
            context[citation]
            .replace("- You entered", '"\n- You entered', 1)
            .replace("Line 55 and line 23500:", '"\nLine 55 and line 23500:')
        )
    elif mutation == "duplicate_body":
        context[citation] += "\n" + context[citation]
    elif mutation == "joined_condition":
        old = provider["rules"][0]["metadata"]["proof"]["atoms"][-1]["source"][
            "excerpt"
        ]
        new = old + " unless exempt"
        context[citation] = context[citation].replace(old, new)
        provider["rules"][0]["metadata"]["proof"]["atoms"][-1]["source"]["excerpt"] = (
            new
        )
    elif mutation == "absent_context":
        context[citation] = None
    elif mutation == "wrong_frame_source":
        context[BASE + "/foreign"] = context[BASE + "/primary-document"]
        provider["rules"][0]["metadata"]["proof"]["atoms"][0]["source"][
            "corpus_citation_path"
        ] = BASE + "/foreign"
    assert not bind(world)
