"""Start-date evidence covers the CI engine's explicit and implicit versions.

The real-engine controls compile synthetic parameter modules and inspect the
resulting version dates. They establish syntax/selection behavior, not source
correctness or protected generation provenance.
"""

from __future__ import annotations

import functools
import json
import os
from datetime import date
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import start_date_evidence as sde
from tests.real_engine import binding_receipt
from tests.test_start_date_evidence import (
    CITATION,
    EVIDENCE,
    IMPORT,
    SOURCE,
    import_issues,
    issues,
    module,
)

SHAPES = [
    "version-effective_from",
    "version-from",
    "implicit-effective_from",
    "implicit-from",
    "empty-effective_from",
    "empty-from",
]


def shaped_module(start, shape, **kwargs):
    payload = module(start, **kwargs)
    rule = payload["rules"][0]
    placement, field = shape.split("-", 1)
    version = rule["versions"][0]
    if field == "from":
        version["from"] = version.pop("effective_from")
    if placement != "version":
        rule.pop("versions")
        rule.update(version)
        if placement == "empty":
            rule["versions"] = []
    return payload


def start_holder(payload, shape):
    rule = payload["rules"][0]
    return rule["versions"][0] if shape.startswith("version-") else rule


@pytest.mark.parametrize("shape", SHAPES)
def test_same_unsupported_date_is_refused_in_every_engine_start_shape(shape):
    [issue] = issues(shaped_module("2025-10-01", shape))
    assert issue.found == "2025-10-01"
    field = shape.split("-", 1)[1]
    expected_field = f"versions[0].{field}" if shape.startswith("version-") else field
    assert issue.subject.field == expected_field


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    ("start", "excerpt"),
    [
        ("0001-01-01", None),
        ("2026-07-17", None),
        ("2026-08-01", None),
        ("2025-10-01", "Effective October 1, 2025."),
    ],
)
def test_supported_evidence_passes_in_every_engine_start_shape(shape, start, excerpt):
    assert issues(shaped_module(start, shape, excerpt=excerpt)) == []


@pytest.mark.parametrize("shape", SHAPES)
def test_alias_and_implicit_starts_cannot_use_missing_metadata_as_support(shape):
    evidence = sde.StartDateEvidence(None, None, ("source_attestation is missing",))
    assert len(issues(shaped_module("2026-08-01", shape), evidence=evidence)) == 1
    assert issues(shaped_module("0001-01-01", shape), evidence=evidence) == []


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("missing", [False, True], ids=["null", "absent"])
def test_missing_start_is_not_invented_as_a_supported_date(shape, missing):
    payload = shaped_module(None, shape)
    if missing:
        start_holder(payload, shape).pop(shape.split("-", 1)[1])
    # Missing dates remain the compiler's error, rather than evidence for an
    # arbitrary start or a default supplied by this check.
    assert list(sde._version_starts(payload)) == []
    assert issues(payload) == []


@pytest.mark.parametrize("shape", SHAPES)
def test_imported_export_uses_the_same_alias_and_implicit_version_shapes(shape):
    provider = shaped_module("2025-10-01", shape)
    provider["rules"][0]["name"] = "amount"
    content = yaml.safe_dump(provider)
    assert import_issues("2025-10-01", [("amount", content)]) == []
    assert sde.imported_export_starts(
        module("2025-10-01", imports=[IMPORT]), [("amount", content)]
    ) == frozenset({date(2025, 10, 1)})

    start_holder(provider, shape).pop(shape.split("-", 1)[1])
    assert len(import_issues("2025-10-01", [("amount", yaml.safe_dump(provider))])) == 1


@pytest.mark.parametrize("version_field", ["effective_from", "from"])
@pytest.mark.parametrize("rule_field", ["effective_from", "from"])
@pytest.mark.parametrize(
    ("version_start", "rule_start", "refused"),
    [
        ("2026-08-01", "2025-10-01", False),
        ("2025-10-01", "2026-08-01", True),
    ],
)
def test_nonempty_versions_replace_rule_level_starts(
    version_field, rule_field, version_start, rule_start, refused
):
    payload = shaped_module(version_start, f"version-{version_field}")
    payload["rules"][0].update({rule_field: rule_start, "formula": "false"})
    assert bool(issues(payload)) is refused


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("reverse", [False, True])
def test_both_alias_names_are_duplicate_fields_not_date_precedence(shape, reverse):
    payload = shaped_module("2026-08-01", shape)
    holder = start_holder(payload, shape)
    holder.pop(shape.split("-", 1)[1])
    starts = [("effective_from", "2026-08-01"), ("from", "2025-10-01")]
    holder.update(reversed(starts) if reverse else starts)
    [issue] = issues(payload)
    assert issue.found == "duplicate effective_from/from fields"
    assert issue.subject.field.endswith("effective_from/from")


@pytest.mark.parametrize("shape", SHAPES)
def test_complete_source_analysis_checks_every_engine_start_shape(shape):
    from tests.test_start_date_evidence import _analyze, _start_issues

    bad = _analyze(shaped_module("2025-10-01", shape), EVIDENCE)
    good = _analyze(shaped_module("2026-08-01", shape), EVIDENCE)
    assert len(_start_issues(bad.issues)) == 1
    assert _start_issues(good.issues) == []


def compile_control(real_engine, tmp_path, payload):
    from axiom_encode.rules_engine_compat import run_rulespec_compile

    # A scalar parameter avoids unrelated entity/formula behavior in these
    # controls. The engine uses the same RuleDefinition/RuleVersion reader.
    rule = payload["rules"][0]
    rule.update({"name": "amount", "kind": "parameter", "dtype": "Count"})
    for name in ("entity", "period", "source", "metadata"):
        rule.pop(name, None)
    if rule.get("versions"):
        for version in rule["versions"]:
            version["formula"] = "1"
    else:
        rule["formula"] = "1"
    payload.pop("inputs")
    checkout = tmp_path / "rulespec-us"
    path = checkout / "us-az/policies/test/start-date.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    output = tmp_path / "compiled.json"
    completed = run_rulespec_compile(
        binary=real_engine.binary,
        program=path,
        rulespec_roots=[checkout],
        output=output,
        cwd=real_engine.root,
        env=os.environ.copy(),
    )
    return completed, output


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("start", ["2025-10-01", "2026-08-01", "0001-01-01"])
def test_real_engine_compiles_the_same_start_in_all_supported_shapes(
    real_engine, tmp_path, shape, start
):
    payload = shaped_module(start, shape)
    assert bool(issues(payload)) is (start == "2025-10-01")
    completed, output = compile_control(real_engine, tmp_path, payload)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    [parameter] = json.loads(output.read_text())["program"]["parameters"]
    assert parameter["versions"][0]["effective_from"] == start


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("missing", [False, True], ids=["null", "absent"])
def test_real_engine_does_not_supply_a_default_for_a_missing_start(
    real_engine, tmp_path, shape, missing
):
    payload = shaped_module(None, shape)
    if missing:
        start_holder(payload, shape).pop(shape.split("-", 1)[1])
    if shape.startswith("version-"):
        # An explicit version also cannot inherit the rule's start date.
        payload["rules"][0]["effective_from"] = "2026-08-01"
    completed, _output = compile_control(real_engine, tmp_path, payload)
    assert completed.returncode != 0
    assert "without effective_from" in completed.stderr


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("reverse", [False, True])
def test_real_engine_rejects_both_start_alias_names_in_either_order(
    real_engine, tmp_path, shape, reverse
):
    payload = shaped_module("2026-08-01", shape)
    holder = start_holder(payload, shape)
    holder.pop(shape.split("-", 1)[1])
    starts = [("effective_from", "2026-08-01"), ("from", "2025-10-01")]
    holder.update(reversed(starts) if reverse else starts)
    completed, _output = compile_control(real_engine, tmp_path, payload)
    assert completed.returncode != 0
    assert "duplicate field" in completed.stderr
    assert "effective_from" in completed.stderr


@pytest.mark.parametrize("version_field", ["effective_from", "from"])
@pytest.mark.parametrize("rule_field", ["effective_from", "from"])
def test_real_engine_nonempty_versions_replace_rule_level_start(
    real_engine, tmp_path, version_field, rule_field
):
    payload = shaped_module("2026-08-01", f"version-{version_field}")
    payload["rules"][0].update({rule_field: "2025-10-01", "formula": "2"})
    completed, output = compile_control(real_engine, tmp_path, payload)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    [parameter] = json.loads(output.read_text())["program"]["parameters"]
    assert parameter["versions"][0]["effective_from"] == "2026-08-01"


RAW_SHAPES = ["blank-versions", "comment-versions", "merged-rule", "merged-version"]


def raw_start_module(
    start,
    shape,
    field="effective_from",
    *,
    baseline_start="2026-08-01",
    both_aliases=False,
    null_start=False,
):
    """Keep the exact blank scalar and merge syntax that safe_dump erases."""

    literal = json.dumps(start) if start is not None else "null" if null_start else None
    prefix = "format: rulespec/v1\nrules:\n"
    base = (
        "  - &base_rule\n"
        "    name: baseline\n"
        "    kind: parameter\n"
        "    dtype: Count\n"
        "    versions:\n"
        "      - &base_version\n"
        f"        effective_from: {json.dumps(baseline_start)}\n"
        "        formula: '1'\n"
    )
    other = "from" if field == "effective_from" else "effective_from"

    def starts(indent):
        text = f"{indent}{field}: {literal}\n" if literal is not None else ""
        if both_aliases:
            text += f"{indent}{other}: '2025-10-01'\n"
        return text

    amount = "    name: amount\n    kind: parameter\n    dtype: Count\n"
    if shape == "merged-version":
        return (
            prefix
            + base
            + "  -\n"
            + amount
            + "    versions:\n      - <<: *base_version\n"
            + starts("        ")
            + "        formula: '1'\n"
        )
    if shape == "merged-rule":
        return (
            prefix
            + base
            + "  - <<: *base_rule\n"
            + amount
            + starts("    ")
            + "    formula: '1'\n"
        )
    versions = "versions:" if shape == "blank-versions" else "versions: # none yet"
    return (
        prefix
        + "  -\n"
        + amount
        + starts("    ")
        + f"    formula: '1'\n    {versions}\n"
    )


def raw_start_issues(content):
    return sde.date_silent_start_issues(
        yaml.safe_load(content),
        SOURCE,
        corpus_citation_path=CITATION,
        evidence=EVIDENCE,
        raw_content=content,
    )


@pytest.mark.parametrize("shape", RAW_SHAPES)
@pytest.mark.parametrize("field", ["effective_from", "from"])
@pytest.mark.parametrize("start", ["2025-10-01", "2026-08-01", "0001-01-01"])
def test_raw_engine_start_shapes_check_the_date_the_engine_consumes(
    shape, field, start
):
    found = raw_start_issues(raw_start_module(start, shape, field))
    if start == "2025-10-01":
        [issue] = found
        assert issue.found == start
        assert issue.subject.rule == "amount"
        assert issue.subject.field == (
            f"versions[0].{field}" if shape == "merged-version" else field
        )
    else:
        assert found == []


@pytest.mark.parametrize("shape", RAW_SHAPES)
@pytest.mark.parametrize("field", ["effective_from", "from"])
@pytest.mark.parametrize("null_start", [False, True], ids=["missing", "null"])
def test_raw_empty_or_merged_shape_does_not_invent_a_start(shape, field, null_start):
    content = raw_start_module(None, shape, field, null_start=null_start)
    assert raw_start_issues(content) == []
    assert not any(
        name == "amount"
        for name, _field, _start in sde._version_starts(
            sde._engine_start_payload(content)
        )
    )


@pytest.mark.parametrize("shape", RAW_SHAPES)
@pytest.mark.parametrize("field", ["effective_from", "from"])
def test_raw_own_alias_duplicates_still_refuse_in_either_order(shape, field):
    [issue] = raw_start_issues(
        raw_start_module("2026-08-01", shape, field, both_aliases=True)
    )
    assert issue.found == "duplicate effective_from/from fields"


@pytest.mark.parametrize("shape", RAW_SHAPES)
@pytest.mark.parametrize("field", ["effective_from", "from"])
def test_imported_raw_provider_uses_its_actual_explicit_or_implicit_start(shape, field):
    content = raw_start_module("2025-10-01", shape, field)
    assert import_issues("2025-10-01", [("amount", content)]) == []
    assert sde.imported_export_starts(
        module("2025-10-01", imports=[IMPORT]), [("amount", content)]
    ) == frozenset({date(2025, 10, 1)})


def test_provider_does_not_lend_the_versions_that_only_pyyaml_inherits():
    content = raw_start_module("2026-08-01", "merged-rule", baseline_start="2025-10-01")
    assert len(import_issues("2025-10-01", [("amount", content)])) == 1


@pytest.mark.parametrize("field", ["effective_from", "from"])
@pytest.mark.parametrize("start", ["2025-10-01", "2026-08-01"])
def test_analyzer_propagates_raw_merge_fields_to_the_actual_start_gate(field, start):
    from axiom_encode.harness import source_completeness as sc
    from axiom_encode.harness import validator_pipeline as vp

    content = raw_start_module(start, "merged-rule", field)
    analysis = sc.analyze_complete_source_unit(
        content,
        SOURCE,
        corpus_citation_path=CITATION,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            vp.extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=vp.extract_named_scalar_occurrences,
        numeric_value_is_grounded=vp.numeric_value_is_grounded,
        start_date_evidence=EVIDENCE,
    )
    found = [
        issue
        for issue in analysis.issues
        if getattr(issue, "check", None) == sde.CHECK_START_UNSUPPORTED.id
    ]
    assert len(found) == (1 if start == "2025-10-01" else 0)


def compile_raw_control(real_engine, tmp_path, content):
    from axiom_encode.rules_engine_compat import run_rulespec_compile

    checkout = tmp_path / "rulespec-us"
    path = checkout / "us-az/policies/test/raw-start-date.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(content)
    output = tmp_path / "compiled.json"
    completed = run_rulespec_compile(
        binary=real_engine.binary,
        program=path,
        rulespec_roots=[checkout],
        output=output,
        cwd=real_engine.root,
        env=os.environ.copy(),
    )
    return completed, output


@pytest.mark.parametrize("shape", RAW_SHAPES)
@pytest.mark.parametrize("field", ["effective_from", "from"])
@pytest.mark.parametrize("start", ["2025-10-01", "2026-08-01", "0001-01-01"])
def test_real_engine_raw_start_shape_controls(
    real_engine, tmp_path, shape, field, start
):
    content = raw_start_module(start, shape, field)
    assert bool(raw_start_issues(content)) is (start == "2025-10-01")
    completed, output = compile_raw_control(real_engine, tmp_path, content)
    assert completed.returncode == 0, completed.stderr or completed.stdout
    amount = next(
        item
        for item in json.loads(output.read_text())["program"]["parameters"]
        if item["name"] == "amount"
    )
    assert amount["versions"][0]["effective_from"] == start


@pytest.mark.parametrize("shape", RAW_SHAPES)
@pytest.mark.parametrize("field", ["effective_from", "from"])
@pytest.mark.parametrize("null_start", [False, True], ids=["missing", "null"])
def test_real_engine_raw_shapes_have_no_default_start(
    real_engine, tmp_path, shape, field, null_start
):
    completed, _output = compile_raw_control(
        real_engine,
        tmp_path,
        raw_start_module(None, shape, field, null_start=null_start),
    )
    assert completed.returncode != 0
    assert "without effective_from" in completed.stderr


@pytest.mark.parametrize("shape", RAW_SHAPES)
@pytest.mark.parametrize("field", ["effective_from", "from"])
def test_real_engine_raw_shapes_refuse_own_duplicate_aliases(
    real_engine, tmp_path, shape, field
):
    completed, _output = compile_raw_control(
        real_engine,
        tmp_path,
        raw_start_module("2026-08-01", shape, field, both_aliases=True),
    )
    assert completed.returncode != 0
    assert "duplicate field" in completed.stderr


@pytest.mark.parametrize("spelling", ["null", "~"])
def test_real_engine_explicit_null_versions_are_not_blank_vectors(
    real_engine, tmp_path, spelling
):
    content = raw_start_module("2025-10-01", "blank-versions").replace(
        "    versions:\n", f"    versions: {spelling}\n"
    )
    # Invalid providers cannot lend an implicit date through this container.
    assert (
        sde.imported_export_starts(
            module("2025-10-01", imports=[IMPORT]), [("amount", content)]
        )
        == frozenset()
    )
    completed, _output = compile_raw_control(real_engine, tmp_path, content)
    assert completed.returncode != 0
    assert "expected a sequence" in completed.stderr


def test_public_raw_merge_regression_retains_its_original_inputs():
    import hashlib

    from scripts.replay_us_failure_bundles import load_fixture, replay_case

    root = Path(__file__).parent / "fixtures/us_replay_benchmark"
    directory = root / "local-date-silent-merge-41088317595"
    case = load_fixture(directory)
    mutation = case.provenance["mutation"]
    parent = root / mutation["base_fixture"]
    original = (parent / "candidate.yaml").read_text()
    [replacement] = mutation["replacements"]
    assert original.count(replacement["old"]) == 1
    assert original.replace(replacement["old"], replacement["new"], 1) == case.candidate
    assert mutation["not_historical_encoder_output"] is True
    for name, digest in mutation["original_hashes"].items():
        assert hashlib.sha256((parent / name).read_bytes()).hexdigest() == digest
        if name != "candidate.yaml":
            assert (directory / name).read_bytes() == (parent / name).read_bytes()
    for relative in case.provenance["context_imports"].values():
        assert (directory / relative).read_bytes() == (parent / relative).read_bytes()
    recorded = json.loads((directory / "issues.json").read_text())
    assert len(recorded["issues"]) == 1
    assert "2025-10-01" in recorded["issues"][0]
    assert "complete-source-unit:effective-date" in recorded["issues"][0]
    replayed = replay_case(case)
    assert replayed["error"] is None
    assert replayed["issues"] == []


def test_real_engine_public_raw_merge_preserves_the_program_and_companion(
    real_engine, tmp_path
):
    from axiom_encode.harness.validator_pipeline import ValidatorPipeline
    from axiom_encode.rules_engine_compat import run_rulespec_compile
    from scripts.replay_us_failure_bundles import load_fixture

    root = Path(__file__).parent / "fixtures/us_replay_benchmark"
    directory = root / "local-date-silent-merge-41088317595"
    case = load_fixture(directory)
    original = load_fixture(root / "41088317595")
    checkout = tmp_path / "rulespec-us"
    country = checkout / case.policy_prefix
    program = country / case.module_path
    program.parent.mkdir(parents=True)
    for target, content in case.context.items():
        path = (country / target.split(":", 1)[1]).with_suffix(".yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    receipt = binding_receipt(real_engine)
    assert receipt["binary_sha256"] == real_engine.sha256
    config = checkout / ".axiom/toolchain.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        f'[toolchain]\naxiom_rules_engine_ref = "{receipt["engine_commit"]}"\n'
    )
    program.with_name(program.stem + ".test.yaml").write_text(case.tests)
    payloads = []
    for index, content in enumerate((original.candidate, case.candidate)):
        program.write_text(content)
        output = tmp_path / f"compiled-{index}.json"
        compiled = run_rulespec_compile(
            binary=real_engine.binary,
            program=program,
            rulespec_roots=[checkout],
            output=output,
            cwd=real_engine.root,
            env=os.environ.copy(),
        )
        assert compiled.returncode == 0, compiled.stderr or compiled.stdout
        payload = json.loads(output.read_text())
        payloads.append(output.read_bytes())
        pipeline = ValidatorPipeline(
            policy_repo_path=country,
            axiom_rules_path=real_engine.root,
            local_corpus_release=None,
            enable_oracles=False,
            max_workers=1,
            axiom_rules_engine_ref=receipt["engine_commit"],
        )
        assert (
            pipeline._run_rulespec_test_cases(
                rules_file=program,
                compiled_path=output,
                compiled_payload=payload,
                cases=yaml.safe_load(case.tests),
            )
            == []
        )
    assert payloads[0] == payloads[1]
