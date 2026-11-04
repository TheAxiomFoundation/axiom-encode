"""Referral collection authenticates source identity, never downstream execution."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.harness import floor_retained_referral_evidence as f
from axiom_encode.harness import validator_pipeline as v


@pytest.fixture
def collected(tmp_path, monkeypatch):
    root = tmp_path / "ca"
    root.mkdir()
    consumer = root / "consumer.yaml"
    target = root / "target.yaml"
    family = "ca/policy/example/target"
    payload = {
        "format": "rulespec/v1",
        "rules": [
            {
                "name": "reference",
                "kind": "source_relation",
                "source_relation": {
                    "type": "cites",
                    "target": "ca:policies/example/target",
                },
            }
        ],
    }
    provider = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": family}},
    }
    content = yaml.safe_dump(payload)
    consumer.write_text(content)
    target.write_text(yaml.safe_dump(provider))
    release = object()
    token = v._AUTHORITATIVE_CORPUS_RELEASE.set(release)
    monkeypatch.setattr(v, "_resolve_rulespec_target_file", lambda *a, **k: target)

    def resolve(citation, selected):
        assert selected is release
        return SimpleNamespace(
            body="source text",
            proof_evidence_text="source text",
            row=SimpleNamespace(expression_date="2031-01-01"),
            to_attestation=lambda: {
                "source_sha256": "a" * 64,
                "corpus_release": "signed-release",
                "corpus_release_content_sha256": "b" * 64,
                "requested_corpus_citation_path": citation,
            },
        )

    monkeypatch.setattr(f, "resolve_local_corpus_source", resolve)
    pipeline = SimpleNamespace(
        _validation_source_root=lambda _: root, rulespec_dependency_roots=()
    )
    sources = {family + "/primary-document": "source text"}

    def collect():
        return f.collect_referral_context(pipeline, content, consumer, sources)

    yield SimpleNamespace(
        root=root,
        consumer=consumer,
        target=target,
        content=content,
        provider=provider,
        collect=collect,
        sources=sources,
        payload=payload,
        pipeline=pipeline,
    )
    v._AUTHORITATIVE_CORPUS_RELEASE.reset(token)


def test_normal_collector_binds_exact_consumer_target_and_authenticated_body(collected):
    binding = collected.collect()
    assert binding is not None
    assert binding.bound(collected.content, collected.sources)
    assert "signed-release" in binding.target_attestation_json
    assert not replace(binding, seal=object()).bound(
        collected.content, collected.sources
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("source_sha256", "c" * 64),
        ("corpus_release", "foreign-release"),
        ("corpus_release_content_sha256", "c" * 64),
        ("corpus_citation_paths", ["ca/policy/example/target"]),
    ],
)
def test_supplied_stale_or_conflicting_target_source_pins_refuse(collected, key, value):
    collected.provider["module"]["source_verification"][key] = value
    collected.target.write_text(yaml.safe_dump(collected.provider))
    assert collected.collect() is None


def test_matching_optional_target_pins_are_supported(collected):
    collected.provider["module"]["source_verification"].update(
        source_sha256="a" * 64,
        corpus_release="signed-release",
        corpus_release_content_sha256="b" * 64,
    )
    collected.target.write_text(yaml.safe_dump(collected.provider))
    assert collected.collect() is not None


@pytest.mark.parametrize("path_name", ["target", "consumer"])
def test_symlink_and_swap_are_not_authenticated(collected, path_name):
    binding = collected.collect()
    path = getattr(collected, path_name)
    other = path.with_suffix(".saved")
    path.rename(other)
    path.symlink_to(other)
    assert collected.collect() is None
    with pytest.raises((OSError, ValueError)):
        binding.bound(collected.content, collected.sources)


def test_changed_target_or_consumer_invalidates_existing_binding(collected):
    binding = collected.collect()
    collected.target.write_text(collected.target.read_text() + "\n")
    assert not binding.bound(collected.content, collected.sources)
    assert (
        f.collect_referral_context(
            collected.pipeline,
            collected.content + "\n",
            collected.consumer,
            collected.sources,
        )
        is None
    )


def test_source_body_mismatch_or_unbound_release_refuses(collected):
    collected.sources[next(iter(collected.sources))] = "tampered source"
    assert collected.collect() is None
    token = v._AUTHORITATIVE_CORPUS_RELEASE.set(None)
    try:
        assert collected.collect() is None
    finally:
        v._AUTHORITATIVE_CORPUS_RELEASE.reset(token)


def test_target_outside_consumer_root_refuses(collected, monkeypatch, tmp_path):
    target = tmp_path / "outside.yaml"
    target.write_text(yaml.safe_dump(collected.provider))
    monkeypatch.setattr(v, "_resolve_rulespec_target_file", lambda *a, **k: target)
    assert collected.collect() is None


@pytest.fixture
def local_pair(tmp_path):
    import copy
    import json

    from axiom_encode.harness import same_case_execution as e
    from tests.test_printed_arithmetic_referral import CITATION, bound_world

    text, printed, branch, rules, _ = bound_world()
    context = dict(printed.proof_sources)
    payload = yaml.safe_load(printed.consumer_content)
    raw = copy.deepcopy(rules["net"])
    raw["name"] = "signed"
    raw["source"] = (
        "Signed difference retained for referred calculations; no loss eligibility."
    )
    raw["versions"][0]["formula"] = "a-b"
    raw["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = branch.text.split(
        " (if negative"
    )[0]
    target_family = "ca/policy/example/loss"
    target_uri = "ca:policies/example/loss"
    local_excerpt = "If negative, you may have a non-capital loss (see Form T1A)"

    def atom(path, kind, citation, excerpt):
        return {
            "path": path,
            "kind": kind,
            "source": {"corpus_citation_path": citation, "excerpt": excerpt},
        }

    relation = {
        "name": "referral",
        "kind": "source_relation",
        "source_relation": {"type": "cites", "target": target_uri},
        "metadata": {
            "proof": {
                "atoms": [
                    atom("source_relation.type", "exception", CITATION, local_excerpt)
                ]
            }
        },
    }
    for suffix, body in [
        ("part-2", "T1A E (25)"),
        (
            "primary-document",
            "Use this form to request a carryback of a loss from 2025 to any of the three previous tax years.",
        ),
    ]:
        citation = target_family + "/" + suffix
        context[citation] = body
        relation["metadata"]["proof"]["atoms"].append(
            atom("source_relation.target", "definition", citation, body)
        )
    for suffix in ("one", "two"):
        citation = "ca/policy/example/" + suffix + "/primary-document"
        body = 'income from line 23600 of the return (or the amount you would have entered if the instructions on line 23600 said "if negative, show in brackets")'
        context[citation] = body
        raw["metadata"]["proof"]["atoms"].append(
            atom("source", "definition", citation, body)
        )
    rules["signed"] = raw
    payload["rules"] = [*rules.values(), relation]
    content = yaml.safe_dump(payload)
    relative = "ca/policies/test/consumer.yaml"
    root = tmp_path / "original"
    original = root / relative
    original.parent.mkdir(parents=True)
    original.write_text(content)
    snapshot = tmp_path / "snapshot"
    (snapshot / relative).parent.mkdir(parents=True)
    (snapshot / relative).write_text(content)
    target = root / "ca/policies/example/loss.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("informational target snapshot")
    binding = f.BoundReferralContext(
        f._sha(content),
        original,
        root,
        "{}",
        target_uri,
        target,
        target.read_text(),
        target_family,
        "referral",
        tuple(f.ReferralSource(c, b, "{}", "2025") for c, b in context.items()),
        f._SEAL,
    )
    module = "ca:policies/test/consumer"
    cases, records = [], []
    for index, x in enumerate((4, 6), 1):
        values = {"a": x, "b": 5, "signed": x - 5, "net": max(0, x - 5)}
        period = {"period_kind": "tax_year", "start": "2025-01-01", "end": "2025-12-31"}
        case = {
            "name": str(index),
            "period": period,
            "input": {module + "#input.x": x, module + "#input.y": 5},
            "output": {module + "#" + k: val for k, val in values.items()},
        }
        query = {
            "entity_id": f"case-{index}",
            "period": period,
            "outputs": list(case["output"]),
        }
        nodes = {
            module + "#" + k: {
                "id": module + "#" + k,
                "entity_id": query["entity_id"],
                "kind": "scalar",
                "dtype": "decimal",
                "unit": "CAD",
                "value": {"kind": "decimal", "value": str(val)},
            }
            for k, val in values.items()
        }
        response = {
            "metadata": {
                "requested_mode": "explain",
                "actual_mode": "explain",
                "fallback_reason": None,
            },
            "results": [
                {
                    "entity_id": query["entity_id"],
                    "period": period,
                    "outputs": nodes,
                    "trace": nodes,
                }
            ],
        }
        records.append(
            e.CaseExecution(
                e.case_key(case),
                json.dumps({"mode": "explain", "queries": [query], "dataset": {}}),
                json.dumps(response),
                index,
            )
        )
        cases.append(case)
    artifact = {
        "program": {
            "derived": [
                {
                    "id": module + "#" + name,
                    "name": name,
                    "entity": "Person",
                    "period": "Year",
                    "dtype": "decimal",
                    "unit": "CAD",
                    "semantics": "scalar",
                }
                for name in rules
            ]
        }
    }
    evidence = e.SameCaseEvidence(
        (e.ClosureFile(relative, original, content.encode()),),
        relative,
        snapshot,
        "a" * 40,
        "b" * 64,
        "c" * 40,
        json.dumps(artifact).encode(),
        tuple(records),
    )
    printed = replace(
        printed, consumer_content=content, proof_sources=tuple(context.items())
    )
    return SimpleNamespace(
        content=content,
        text=text,
        context=context,
        binding=binding,
        evidence=evidence,
        cases=cases,
        rules=rules,
        printed=printed,
        branch=branch,
        citation=CITATION,
    )


def witness(pair, **overrides):
    from axiom_encode.harness.validator_pipeline import (
        extract_typed_numeric_occurrences_from_text,
        numeric_value_is_grounded,
    )

    args = dict(
        content=pair.content,
        source_text=pair.text,
        corpus_citation_path=pair.citation,
        principal_rules=pair.rules,
        asserted_by_rule={name: pair.cases for name in pair.rules},
        formula_environment={},
        referral_context=pair.binding,
        source_context=pair.context,
        printed_operations=pair.printed,
        extract_numeric_occurrences=extract_typed_numeric_occurrences_from_text,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    args.update(overrides)
    return f.witnesses(pair.evidence, pair.branch, **args)


def test_complete_local_pair_retains_signed_value_without_downstream_execution(
    local_pair,
):
    assert witness(local_pair)


def test_citation_alone_and_missing_signed_observation_do_not_cover(local_pair):
    assert not witness(local_pair, referral_context=None)
    assert not witness(local_pair, asserted_by_rule={"net": local_pair.cases})


def rebind(pair):
    """Bind mutated current inputs; refusal must not be merely a stale receipt."""
    from axiom_encode.harness import same_case_execution as e

    payload = yaml.safe_load(pair.content)
    relation = next(r for r in payload["rules"] if r["kind"] == "source_relation")
    payload["rules"] = [*pair.rules.values(), relation]
    pair.content = yaml.safe_dump(payload)
    old = pair.evidence.files[0]
    old.original.write_text(pair.content)
    (pair.evidence.snapshot_root / old.relative).write_text(pair.content)
    pair.evidence = replace(
        pair.evidence,
        files=(replace(old, raw=pair.content.encode()),),
        records=tuple(
            replace(record, case_sha256=e.case_key(case))
            for record, case in zip(pair.evidence.records, pair.cases)
        ),
    )
    pair.binding = replace(pair.binding, consumer_sha256=f._sha(pair.content))
    pair.printed = replace(
        pair.printed,
        consumer_content=pair.content,
        proof_sources=tuple(pair.context.items()),
    )


@pytest.mark.parametrize("value", [[-1], [{"entity_id": "other", "value": -1}], 17])
def test_current_case_wrong_row_or_wrong_scalar_cannot_correlate(local_pair, value):
    local_pair.cases[0]["output"]["ca:policies/test/consumer#signed"] = value
    rebind(local_pair)
    assert local_pair.evidence.bound(local_pair.content, ())
    assert not witness(local_pair)


@pytest.mark.parametrize(
    "name,field,value",
    [
        ("signed", "formula", "b-a"),
        ("net", "formula", "max(0,b-a)"),
        ("signed", "effective_from", "2024-01-01"),
    ],
)
def test_current_corrupted_local_definition_refuses(local_pair, name, field, value):
    local_pair.rules[name]["versions"][0][field] = value
    rebind(local_pair)
    assert not witness(local_pair)


def test_unreplayed_dependency_cannot_be_filled_by_assertion_and_trace(local_pair):
    local_pair.rules["a"]["versions"][0]["formula"] = "unavailable_import"
    rebind(local_pair)
    assert (
        local_pair.evidence.values(local_pair.cases[0], local_pair.content) is not None
    )
    assert not witness(local_pair)


def test_missing_case_receipt_or_positive_only_refuses(local_pair):
    local_pair.evidence = replace(
        local_pair.evidence, records=local_pair.evidence.records[1:]
    )
    assert not witness(local_pair)


def test_duplicate_floor_owner_is_not_allocated_by_name(local_pair):
    import copy

    duplicate = copy.deepcopy(local_pair.rules["net"])
    duplicate["name"] = "duplicate"
    local_pair.rules["duplicate"] = duplicate
    rebind(local_pair)
    assert not witness(local_pair)


def test_additional_changed_primitive_is_not_one_controlled_pair(local_pair):
    local_pair.cases[1]["input"]["ca:policies/test/consumer#input.y"] = 4
    rebind(local_pair)
    assert not witness(local_pair)


def test_wrong_scope_or_partial_advisory_is_not_whole_row_evidence(local_pair):
    local_pair.branch = replace(local_pair.branch, required_principal_output="signed")
    assert not witness(local_pair)


def test_wrong_footer_year_refuses_current_bound_source(local_pair):
    payload = yaml.safe_load(local_pair.content)
    relation = payload["rules"][-1]
    atom = relation["metadata"]["proof"]["atoms"][1]
    citation = atom["source"]["corpus_citation_path"]
    atom["source"]["excerpt"] = "T1A E (24)"
    local_pair.context[citation] = "T1A E (24)"
    local_pair.content = yaml.safe_dump(payload)
    local_pair.binding = replace(
        local_pair.binding,
        sources=tuple(
            replace(s, body=local_pair.context[s.citation])
            for s in local_pair.binding.sources
        ),
    )
    rebind(local_pair)
    assert local_pair.binding.bound(local_pair.content, local_pair.context)
    assert not witness(local_pair)


@pytest.mark.parametrize(
    "which", ["unit", "unselected-version", "unknown-condition", "target-change"]
)
def test_additional_contract_corruption_refuses(local_pair, which):
    import copy

    if which == "unit":
        local_pair.rules["signed"]["unit"] = "USD"
        rebind(local_pair)
    elif which == "unselected-version":
        version = copy.deepcopy(local_pair.rules["net"]["versions"][0])
        version.update(
            effective_from="2024-01-01", effective_to="2024-12-31", formula="max(0,b-a)"
        )
        local_pair.rules["net"]["versions"].append(version)
        rebind(local_pair)
    elif which == "unknown-condition":
        text = local_pair.text.replace(
            "Net income = ^", "Net income unless eligible = ^"
        )
        local_pair.branch = replace(
            local_pair.branch,
            text=local_pair.branch.text.replace(
                "Net income = ^", "Net income unless eligible = ^"
            ),
            end=local_pair.branch.end + len(" unless eligible"),
        )
        local_pair.text = text
    else:
        local_pair.binding.target_file.write_text("changed informational target")
    assert not witness(local_pair)


def test_two_duplicate_full_rows_share_one_allocated_resource(local_pair):
    from axiom_encode.harness import source_completeness as s

    supplied = witness(local_pair)
    assert len(supplied) == 1
    unresolved = s._unmatched_evidence_obligations(
        {
            ("first", local_pair.branch): supplied,
            ("second", local_pair.branch): supplied,
        }
    )
    assert len(unresolved) == 1


def test_pair_needs_negative_and_strict_positive_not_equality_only(local_pair):
    import json

    from axiom_encode.harness import same_case_execution as e

    case = local_pair.cases[1]
    case["input"]["ca:policies/test/consumer#input.x"] = 5
    for name, value in [("a", 5), ("net", 0), ("signed", 0)]:
        case["output"]["ca:policies/test/consumer#" + name] = value
    rebind(local_pair)
    record = local_pair.evidence.records[1]
    response = json.loads(record.response_json)
    for collection in ("outputs", "trace"):
        for name, value in [("a", 5), ("net", 0), ("signed", 0)]:
            response["results"][0][collection]["ca:policies/test/consumer#" + name][
                "value"
            ]["value"] = str(value)
    local_pair.evidence = replace(
        local_pair.evidence,
        records=(
            local_pair.evidence.records[0],
            replace(
                record, case_sha256=e.case_key(case), response_json=json.dumps(response)
            ),
        ),
    )
    assert local_pair.evidence.values(case, local_pair.content) is not None
    assert not witness(local_pair)
