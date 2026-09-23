"""Opt-in acceptance test for the rulespec-us#1312 EITC successor repoint.

The repository deliberately vendors no rulespec-us content.  Point
``AXIOM_REPOINT_RULESPEC_US`` at a rulespec-us checkout that contains commit
``c654250f`` to run the real-shape proof and rewrite:

    AXIOM_REPOINT_RULESPEC_US=~/TheAxiomFoundation/rulespec-us \\
        uv run pytest tests/test_successor_repoint_integration.py

Point ``AXIOM_REPOINT_DIAGNOSTIC_DIFF`` at ``diag-26-32-repoint-page-15.diff``
as well to additionally assert the postimage is byte-identical to applying that
diagnostic diff.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from axiom_encode.successor_repoint import (
    ENVELOPE_SCHEMA,
    load_repoint_request_payload,
    prove_concept_map,
    rewrite_repoint_file,
)

RULESPEC_REF = "c654250f07c39c35ca7f3e79975368b019d0e4ca"
LEGACY_PRIMARY = "us/policies/irs/rev-proc-2025-32/earned-income-credit.yaml"
SUCCESSOR_PRIMARY = "us/policies/irs/rev-proc-2025-32/page-15.yaml"
DEPENDENT_PRIMARY = "us/statutes/26/32.yaml"

# `patch -p1 < diag-26-32-repoint-page-15.diff` applied to us/statutes/26/32.yaml
# at rulespec-us c654250f.
EXPECTED_POSTIMAGE_SHA256 = (
    "f370002c97e50a10cb35e08db7847fb114aea7bb3d0a5d3a197d8fc2c93ec4d6"
)
# sha256 of us/policies/irs/rev-proc-2025-32/page-15.yaml at the same commit,
# which is also the repointed proof-import hash.
EXPECTED_SUCCESSOR_SHA256 = (
    "b032822be996093985f5d04fb64477d613fefd239c133e4c3c45192cc281f58e"
)

CONCEPT_MAP = [
    ("eitc_earned_income_amounts", "earned_income_credit_earned_income_amounts"),
    ("eitc_maximum_credit_amounts", "earned_income_credit_maximum_credit_amounts"),
    (
        "eitc_threshold_phaseout_amounts_joint",
        "earned_income_credit_phaseout_threshold_joint_amounts",
    ),
    (
        "eitc_threshold_phaseout_amounts_other",
        "earned_income_credit_phaseout_threshold_other_amounts",
    ),
    (
        "eitc_completed_phaseout_amounts_joint",
        "earned_income_credit_completed_phaseout_joint_amounts",
    ),
    (
        "eitc_completed_phaseout_amounts_other",
        "earned_income_credit_completed_phaseout_other_amounts",
    ),
    (
        "eitc_maximum_investment_income",
        "earned_income_credit_maximum_investment_income",
    ),
]


def _checkout() -> Path:
    raw = os.environ.get("AXIOM_REPOINT_RULESPEC_US")
    if not raw:
        pytest.skip("AXIOM_REPOINT_RULESPEC_US is not set")
    repo = Path(raw).expanduser().resolve()
    if not (repo / ".git").exists():
        pytest.skip(f"{repo} is not a Git checkout")
    probe = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{RULESPEC_REF}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip(f"{repo} does not contain rulespec-us {RULESPEC_REF[:8]}")
    return repo


def _blob(repo: Path, relative: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{RULESPEC_REF}:{relative}"],
        capture_output=True,
        check=True,
    ).stdout


@pytest.fixture(scope="module")
def sources() -> dict[str, bytes]:
    repo = _checkout()
    return {
        path: _blob(repo, path)
        for path in (LEGACY_PRIMARY, SUCCESSOR_PRIMARY, DEPENDENT_PRIMARY)
    }


@pytest.fixture(scope="module")
def request_envelope():
    return load_repoint_request_payload(
        {
            "schema": ENVELOPE_SCHEMA,
            "legacy_primary": LEGACY_PRIMARY,
            "successor_primary": SUCCESSOR_PRIMARY,
            "dependents": [DEPENDENT_PRIMARY],
            "concept_map": [{"from": old, "to": new} for old, new in CONCEPT_MAP],
            "program_scope_updates": [
                {"program_spec": "programs/us/fiit/fy-2026.yaml", "scope": "federal"}
            ],
        }
    )


@pytest.fixture(scope="module")
def proofs(sources, request_envelope):
    return prove_concept_map(
        legacy_raw=sources[LEGACY_PRIMARY],
        successor_raw=sources[SUCCESSOR_PRIMARY],
        request=request_envelope,
        dependent_raws={DEPENDENT_PRIMARY: sources[DEPENDENT_PRIMARY]},
    )


def _postimage(sources, request_envelope, proofs) -> bytes:
    successor_sha256 = hashlib.sha256(sources[SUCCESSOR_PRIMARY]).hexdigest()
    assert successor_sha256 == EXPECTED_SUCCESSOR_SHA256
    rewritten, _replacements = rewrite_repoint_file(
        sources[DEPENDENT_PRIMARY],
        primary=True,
        legacy_identity=request_envelope.legacy_identity,
        successor_identity=request_envelope.successor_identity,
        successor_sha256=successor_sha256,
        renames=proofs.renames,
        label=DEPENDENT_PRIMARY,
    )
    return rewritten


def test_proves_every_declared_rename_over_the_successor_window(proofs):
    assert proofs.successor_window_start == "2026-01-01"
    assert proofs.successor_window_end == "2026-12-31"
    assert len(proofs.proofs) == len(CONCEPT_MAP)
    assert proofs.renames == dict(CONCEPT_MAP)
    by_name = {proof.old: proof for proof in proofs.proofs}
    for old, _new in CONCEPT_MAP:
        proof = by_name[old]
        if old == "eitc_maximum_investment_income":
            assert proof.indexed_by_from is None
            assert proof.keys == (0,)
            continue
        # Every table renames indexed_by, so every use must be literal.
        assert proof.indexed_by_from == "qualifying_child_count"
        assert proof.indexed_by_to == "qualifying_child_count_category"
        assert proof.keys == (0, 1, 2, 3)
        assert all(key in proof.keys for key in proof.literal_subscripts)


def test_records_the_post_window_behaviour_change(proofs):
    # The legacy module silently extended the 2026 amounts forever; the signed
    # successor ends 2026-12-31 and governs.  The receipt must say so.
    assert proofs.post_window_behavior_change is True
    assert proofs.dependent_use_windows
    assert all(
        item["effective_to"] is None and item["extends_past_successor_window"] is True
        for item in proofs.dependent_use_windows
    )


def test_postimage_matches_the_recorded_acceptance_digest(
    sources, request_envelope, proofs
):
    rewritten = _postimage(sources, request_envelope, proofs)
    assert hashlib.sha256(rewritten).hexdigest() == EXPECTED_POSTIMAGE_SHA256
    text = rewritten.decode("utf-8")
    assert "us:policies/irs/rev-proc-2025-32/earned-income-credit" not in text
    # The prose in the deferred-output reason keeps its hyphenated phrase.
    assert "prescribe earned-income-credit" in text


def test_postimage_is_byte_identical_to_the_diagnostic_diff(
    tmp_path, sources, request_envelope, proofs
):
    raw_diff = os.environ.get("AXIOM_REPOINT_DIAGNOSTIC_DIFF")
    if not raw_diff:
        pytest.skip("AXIOM_REPOINT_DIAGNOSTIC_DIFF is not set")
    diff_path = Path(raw_diff).expanduser().resolve()
    if not diff_path.is_file():
        pytest.skip(f"{diff_path} is not a readable diff")
    if shutil.which("patch") is None:
        pytest.skip("patch(1) is unavailable")

    expected = tmp_path / "32.yaml"
    expected.write_bytes(sources[DEPENDENT_PRIMARY])
    result = subprocess.run(
        ["patch", "--posix", "-s", "-p1", str(expected)],
        input=diff_path.read_bytes(),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert _postimage(sources, request_envelope, proofs) == expected.read_bytes()


def test_the_whole_transaction_plans_from_the_real_commit(request_envelope):
    """Replay the command's plan read-only from rulespec-us's Git objects.

    This is exactly what the guard re-derives from a receipt's base commit:
    digest-bound v1 ownership, the successor's model manifest shape, the
    whole-tree reference inventory, the rewrite, and every metadata and
    ProgramSpec reconciliation.  Nothing in the checkout is read or written.
    """

    from axiom_encode.cli import _plan_successor_repoint

    repo = _checkout()
    if repo.name != "rulespec-us":
        pytest.skip("the checkout must be named rulespec-us")
    plan = _plan_successor_repoint(
        repo, commit=RULESPEC_REF, request=request_envelope, verify_live=False
    )
    assert plan.legacy_manifests == (
        {
            "path": (
                ".axiom/encoding-manifests/policies/irs/rev-proc-2025-32/"
                "earned-income-credit.json"
            ),
            "sha256": hashlib.sha256(
                _blob(
                    repo,
                    ".axiom/encoding-manifests/policies/irs/rev-proc-2025-32/"
                    "earned-income-credit.json",
                )
            ).hexdigest(),
            "owner_class": "v1-hmac-untrusted",
        },
    )
    (dependent,) = plan.dependent_records
    assert [(item["path"], item["owner_class"]) for item in dependent["manifests"]] == [
        (
            ".axiom/encoding-manifests/us/statutes/26/32.json",
            "v1-manual-hmac-untrusted",
        ),
        (
            ".axiom/encoding-manifests/statutes/26/32.json",
            "v1-deterministic-hmac-untrusted",
        ),
    ]
    assert hashlib.sha256(plan.postimages[Path(DEPENDENT_PRIMARY)]).hexdigest() == (
        EXPECTED_POSTIMAGE_SHA256
    )
    assert sorted(item["path"] for item in plan.metadata_records) == [
        ".axiom/index/provisions_to_rules.json",
        ".axiom/pending-validation-fingerprints.json",
        ".axiom/toolchain.toml",
        ".axiom/upstream-source-check-baseline.txt",
        "known-missing-money-atoms.yaml",
        "known-validation-gaps.yaml",
    ]
    (program,) = plan.program_records
    assert program["program_spec"] == "programs/us/fiit/fy-2026.yaml"
    assert program["removed"] == ["policies/irs/rev-proc-2025-32/earned-income-credit"]
    assert program["added"] == ["policies/irs/rev-proc-2025-32/page-15"]
    assert sorted(path.as_posix() for path in plan.deletions) == [
        ".axiom/encoding-manifests/policies/irs/rev-proc-2025-32/"
        "earned-income-credit.json",
        ".axiom/encoding-manifests/statutes/26/32.json",
        "us/policies/irs/rev-proc-2025-32/earned-income-credit.test.yaml",
        LEGACY_PRIMARY,
    ]
    semantics = plan.proof_set.receipt_semantics()
    assert semantics["post_window_behavior_change"] is True
    assert semantics["pre_window_behavior_change"] is False
    assert {
        item["engine_lowering"]
        for item in semantics["runtime_behavior_outside_successor_window"]
    } == {"indexed_parameter", "scalar_parameter"}
    assert plan.post_waiver_sha256 != plan.base_waiver_sha256
