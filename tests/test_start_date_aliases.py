"""Start-date evidence covers the CI engine's explicit and implicit versions.

The real-engine controls compile synthetic parameter modules and inspect the
resulting version dates. They establish syntax/selection behavior, not source
correctness or protected generation provenance.
"""

from __future__ import annotations

import json
import os
from datetime import date

import pytest
import yaml

from axiom_encode.harness import start_date_evidence as sde
from tests.test_start_date_evidence import (
    EVIDENCE,
    IMPORT,
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
