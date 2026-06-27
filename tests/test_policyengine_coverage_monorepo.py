"""Coverage classifier ID derivation across monorepo and legacy layouts.

The PolicyEngine oracle-coverage classifier must derive identical canonical
legal IDs whether a jurisdiction's RuleSpec content lives in a country
monorepo (``rulespec-us/us-al/...``) or a legacy standalone checkout
(``rulespec-us-al/...``). Earlier the classifier took the repo directory name
as the prefix and treated everything beneath it as the relative path, which
doubled the jurisdiction in monorepo IDs (``us:us-al/policies/X#r`` instead of
``us-al:policies/X#r``). These tests pin the cross-layout equivalence and the
absence of jurisdiction-doubled IDs.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from axiom_encode.oracles.policyengine.coverage import (
    build_policyengine_coverage_report,
)

# An output with no registry mapping stays ``unmapped`` regardless of layout,
# which keeps these assertions independent of the packaged mapping registry.
_UNMAPPED_US_RULESPEC = """format: rulespec/v1
rules:
  - name: brand_new_state_helper_xyz
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: state_input
"""

_KS_TANF_RULESPEC = """format: rulespec/v1
rules:
  - name: ks_tanf_maximum_benefit
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: tanf_basic_allowance + tanf_shelter_allowance
"""

_UNMAPPED_UK_RULESPEC = """format: rulespec/v1
rules:
  - name: brand_new_local_helper_xyz
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: local_input
"""


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _malformed_doubled_ids(report: dict) -> list[str]:
    """Return IDs whose jurisdiction prefix is doubled (``us:us-...``)."""
    pattern = re.compile(r"^([a-z]{2}):\1[-/]")
    return [
        item["legal_id"] for item in report["items"] if pattern.match(item["legal_id"])
    ]


def test_monorepo_and_legacy_layouts_derive_identical_ids(tmp_path):
    """A file under ``rulespec-us/us-al`` and one under a legacy
    ``rulespec-us-al`` checkout must produce the same ``us-al:policies/X#r``."""
    monorepo_root = tmp_path / "mono"
    _write(
        monorepo_root / "rulespec-us" / "us-al" / "policies" / "dhr" / "poe.yaml",
        _UNMAPPED_US_RULESPEC,
    )

    legacy_root = tmp_path / "legacy"
    _write(
        legacy_root / "rulespec-us-al" / "policies" / "dhr" / "poe.yaml",
        _UNMAPPED_US_RULESPEC,
    )

    monorepo_report = build_policyengine_coverage_report(monorepo_root)
    legacy_report = build_policyengine_coverage_report(legacy_root)

    expected_id = "us-al:policies/dhr/poe#brand_new_state_helper_xyz"
    monorepo_ids = [item["legal_id"] for item in monorepo_report["items"]]
    legacy_ids = [item["legal_id"] for item in legacy_report["items"]]

    assert monorepo_ids == [expected_id]
    assert legacy_ids == [expected_id]
    assert monorepo_ids == legacy_ids

    # The repo attribution is the canonical legacy repo name in both layouts.
    assert {item["repo"] for item in monorepo_report["items"]} == {"rulespec-us-al"}
    assert {item["repo"] for item in legacy_report["items"]} == {"rulespec-us-al"}


def test_direct_monorepo_root_is_enumerated(tmp_path):
    """``--root <rulespec-us>`` should scan that checkout, not only siblings."""
    root = tmp_path / "rulespec-us"
    _write(root / "us-al" / "policies" / "dhr" / "poe.yaml", _UNMAPPED_US_RULESPEC)

    report = build_policyengine_coverage_report(root)

    assert [item["legal_id"] for item in report["items"]] == [
        "us-al:policies/dhr/poe#brand_new_state_helper_xyz"
    ]
    assert {item["repo"] for item in report["items"]} == {"rulespec-us-al"}


def test_direct_legacy_root_is_enumerated(tmp_path):
    """``--root <rulespec-us-al>`` should scan legacy standalone checkouts."""
    root = tmp_path / "rulespec-us-al"
    _write(root / "policies" / "dhr" / "poe.yaml", _UNMAPPED_US_RULESPEC)

    report = build_policyengine_coverage_report(root)

    assert [item["legal_id"] for item in report["items"]] == [
        "us-al:policies/dhr/poe#brand_new_state_helper_xyz"
    ]
    assert {item["repo"] for item in report["items"]} == {"rulespec-us-al"}


def test_kansas_tanf_keesm_prefix_is_classified_not_comparable(tmp_path):
    """Kansas KEESM 7411 source helpers are explicit non-comparable TANF slices."""
    root = tmp_path / "rulespec-us"
    _write(
        root / "us-ks" / "policies" / "dcf" / "keesm" / "keesm7410.yaml",
        _KS_TANF_RULESPEC,
    )

    report = build_policyengine_coverage_report(root, program="tanf")

    assert len(report["items"]) == 1
    item = report["items"][0]
    assert item["legal_id"] == (
        "us-ks:policies/dcf/keesm/keesm7410#ks_tanf_maximum_benefit"
    )
    assert item["repo"] == "rulespec-us-ks"
    assert item["status"] == "known_not_comparable"
    assert item["mapping_type"] == "not_comparable"
    assert item["policyengine_variable"] == "ks_tanf_maximum_benefit"


def test_monorepo_country_directory_is_not_doubled(tmp_path):
    """Country-level content in ``rulespec-us/us`` keeps the ``us:`` prefix."""
    root = tmp_path / "mono"
    _write(
        root / "rulespec-us" / "us" / "statutes" / "26" / "9999.yaml",
        _UNMAPPED_US_RULESPEC,
    )

    report = build_policyengine_coverage_report(root)
    ids = [item["legal_id"] for item in report["items"]]

    assert ids == ["us:statutes/26/9999#brand_new_state_helper_xyz"]
    assert _malformed_doubled_ids(report) == []


def test_monorepo_uk_jurisdiction_directories(tmp_path):
    """``uk`` and ``uk-kingston-upon-thames`` directories keep their prefixes."""
    root = tmp_path / "mono"
    _write(
        root
        / "rulespec-uk"
        / "uk-kingston-upon-thames"
        / "policies"
        / "kingston-upon-thames"
        / "council-tax-reduction.yaml",
        _UNMAPPED_UK_RULESPEC,
    )
    _write(
        root / "rulespec-uk" / "uk" / "policies" / "govuk" / "child-benefit.yaml",
        _UNMAPPED_UK_RULESPEC,
    )

    report = build_policyengine_coverage_report(root)
    ids = {item["legal_id"] for item in report["items"]}

    assert ids == {
        "uk-kingston-upon-thames:policies/kingston-upon-thames/"
        "council-tax-reduction#brand_new_local_helper_xyz",
        "uk:policies/govuk/child-benefit#brand_new_local_helper_xyz",
    }
    assert _malformed_doubled_ids(report) == []
    repos = {item["repo"] for item in report["items"]}
    assert repos == {"rulespec-uk", "rulespec-uk-kingston-upon-thames"}


def test_monorepo_program_directory_is_not_a_jurisdiction(tmp_path):
    """``programs/`` emits program specs without becoming a fake jurisdiction."""
    root = tmp_path / "mono"
    _write(
        root / "rulespec-us" / "us-al" / "policies" / "dhr" / "poe.yaml",
        _UNMAPPED_US_RULESPEC,
    )
    # A shared non-encoding directory holding non-rulespec program manifests.
    _write(
        root / "rulespec-us" / "programs" / "us-al" / "snap" / "fy-2026.yaml",
        "program: us-al/snap\noutputs:\n  - snap_eligible\n",
    )

    report = build_policyengine_coverage_report(root)
    ids = [item["legal_id"] for item in report["items"]]

    assert ids == [
        "us-al:policies/dhr/poe#brand_new_state_helper_xyz",
        "us-al:programs/snap/fy-2026#snap_eligible",
    ]
    assert all(not legal_id.startswith("programs:") for legal_id in ids)


def test_fake_monorepo_produces_no_malformed_country_doubled_ids(tmp_path):
    """A classifier run over a fake multi-jurisdiction monorepo emits zero
    malformed ``<country>:<country>-`` IDs."""
    root = tmp_path / "mono"
    jurisdiction_files = {
        ("rulespec-us", "us"): "statutes/26/100.yaml",
        ("rulespec-us", "us-al"): "policies/dhr/poe/100.yaml",
        ("rulespec-us", "us-ca"): "regulations/mpp/63-300/1.yaml",
        ("rulespec-us", "us-ny"): "policies/otda/snap/100.yaml",
        ("rulespec-us", "us-tx"): "policies/hhsc/snap/100.yaml",
        ("rulespec-uk", "uk"): "policies/govuk/child-benefit.yaml",
        ("rulespec-uk", "uk-kingston-upon-thames"): (
            "policies/kingston-upon-thames/council-tax-reduction.yaml"
        ),
    }
    for (checkout, prefix), rel in jurisdiction_files.items():
        content = (
            _UNMAPPED_UK_RULESPEC if prefix.startswith("uk") else _UNMAPPED_US_RULESPEC
        )
        _write(root / checkout / prefix / rel, content)

    report = build_policyengine_coverage_report(root)

    assert _malformed_doubled_ids(report) == []
    prefixes = {item["legal_id"].split(":", 1)[0] for item in report["items"]}
    assert prefixes == {
        "us",
        "us-al",
        "us-ca",
        "us-ny",
        "us-tx",
        "uk",
        "uk-kingston-upon-thames",
    }


def test_multi_checkout_symlink_layout_matches_ci(tmp_path):
    """Mirror CI: the workspace root holds a real consumer monorepo checkout
    plus a sibling-checkout symlink to a nested second monorepo. Both walk
    correctly, output IDs are not doubled, and the symlinked checkout's outputs
    are not double-counted (the resolved-path dedup collapses the symlink and
    the nested checkout)."""
    workspace = tmp_path / "work"
    workspace.mkdir()
    consumer = workspace / "rulespec-uk"
    _write(
        consumer
        / "uk-kingston-upon-thames"
        / "policies"
        / "kingston-upon-thames"
        / "council-tax-reduction.yaml",
        _UNMAPPED_UK_RULESPEC,
    )
    _write(
        consumer / "uk" / "policies" / "govuk" / "child-benefit.yaml",
        _UNMAPPED_UK_RULESPEC,
    )

    # A second monorepo nested under the consumer checkout's _axiom/ directory,
    # exposed at the workspace root through a sibling-checkout symlink.
    nested_us = consumer / "_axiom" / "rulespec-us"
    _write(
        nested_us / "us-al" / "policies" / "dhr" / "poe.yaml",
        _UNMAPPED_US_RULESPEC,
    )
    _write(
        nested_us / "us" / "statutes" / "26" / "9999.yaml",
        _UNMAPPED_US_RULESPEC,
    )
    os.symlink(nested_us, workspace / "rulespec-us")

    report = build_policyengine_coverage_report(workspace)

    ids = sorted(item["legal_id"] for item in report["items"])
    assert ids == [
        "uk-kingston-upon-thames:policies/kingston-upon-thames/"
        "council-tax-reduction#brand_new_local_helper_xyz",
        "uk:policies/govuk/child-benefit#brand_new_local_helper_xyz",
        "us-al:policies/dhr/poe#brand_new_state_helper_xyz",
        "us:statutes/26/9999#brand_new_state_helper_xyz",
    ]
    # No output is attributed twice despite the symlink + nested checkout.
    assert len(ids) == len(set(ids))
    assert _malformed_doubled_ids(report) == []

    # The reported file path keeps the symlink-name prefix (``rulespec-us/...``)
    # so CI's changed-file matching against ``<consumer-repo>/<path>`` works.
    files_by_id = {item["legal_id"]: item["file"] for item in report["items"]}
    assert (
        files_by_id["us-al:policies/dhr/poe#brand_new_state_helper_xyz"]
        == "rulespec-us/us-al/policies/dhr/poe.yaml"
    )
    assert (
        files_by_id["uk:policies/govuk/child-benefit#brand_new_local_helper_xyz"]
        == "rulespec-uk/uk/policies/govuk/child-benefit.yaml"
    )


def test_legacy_and_monorepo_unmapped_outputs_match_for_real_prefix(tmp_path):
    """A genuinely-new (unmapped) output remains unmapped in both layouts,
    confirming the registry lookup is keyed on the same canonical ID."""
    monorepo_root = tmp_path / "mono"
    _write(
        monorepo_root / "rulespec-us" / "us-zz" / "policies" / "new" / "x.yaml",
        _UNMAPPED_US_RULESPEC,
    )
    legacy_root = tmp_path / "legacy"
    _write(
        legacy_root / "rulespec-us-zz" / "policies" / "new" / "x.yaml",
        _UNMAPPED_US_RULESPEC,
    )

    monorepo_report = build_policyengine_coverage_report(monorepo_root)
    legacy_report = build_policyengine_coverage_report(legacy_root)

    assert monorepo_report["status_counts"] == {"unmapped": 1}
    assert legacy_report["status_counts"] == {"unmapped": 1}
    assert [item["legal_id"] for item in monorepo_report["items"]] == [
        item["legal_id"] for item in legacy_report["items"]
    ]
