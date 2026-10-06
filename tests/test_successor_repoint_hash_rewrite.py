"""Exact proof-pin refreshes and namespace proofs for cascade-only dependents."""

from __future__ import annotations

import hashlib

import pytest

from axiom_encode.successor_repoint import (
    SuccessorRepointError,
    load_repoint_request_payload,
    prove_concept_map,
    rewrite_repoint_file,
)
from tests.test_successor_repoint import (
    DEPENDENT_PRIMARY,
    LEGACY_IDENTITY,
    PAIRS,
    SUCCESSOR_IDENTITY,
    _dependent,
    _envelope,
    _legacy_module,
    _prove,
    _rewrite,
    _successor_module,
)

PINNED_IDENTITY = "us:statutes/26/32"
IMPORTER_PRIMARY = "us/statutes/26/24/d.yaml"
BEFORE = "1" * 64
AFTER = "2" * 64


def _pinning_rule(location: str, *, quote: str = "", identity=PINNED_IDENTITY):
    prefix = "    metadata:\n" if location == "metadata.proof" else ""
    indent = "      " if prefix else "    "
    return (
        "  - name: child_credit\n"
        "    kind: derived\n"
        "    dtype: Money\n"
        "    unit: USD\n"
        "    versions:\n"
        "      - effective_from: '2026-01-01'\n"
        "        formula: earned_income_amount\n"
        f"    note: sha256:{BEFORE}\n"
        f"{prefix}"
        f"{indent}proof:\n"
        f"{indent}  atoms:\n"
        f"{indent}    - kind: import\n"
        f"{indent}      import:\n"
        f"{indent}        target: {identity}#earned_income_amount\n"
        f"{indent}        output: earned_income_amount\n"
        f"{indent}        hash: {quote}sha256:{BEFORE}{quote} # pin comment\n"
    )


def _hash_only_dependent(location, *, quote=""):
    return (
        "format: rulespec/v1\n"
        "imports:\n"
        f"  - {PINNED_IDENTITY}\n"
        f"  - {SUCCESSOR_IDENTITY}\n"
        "rules:\n" + _pinning_rule(location, quote=quote)
    ).encode()


def _refresh(raw, *, refreshes=None):
    return rewrite_repoint_file(
        raw,
        primary=True,
        legacy_identity=LEGACY_IDENTITY,
        successor_identity=SUCCESSOR_IDENTITY,
        successor_sha256=hashlib.sha256(_successor_module()).hexdigest(),
        renames=dict(PAIRS),
        label=IMPORTER_PRIMARY,
        proof_hash_refreshes=(
            {PINNED_IDENTITY: (BEFORE, AFTER)} if refreshes is None else refreshes
        ),
    )


@pytest.mark.parametrize("location", ["metadata.proof", "proof"])
@pytest.mark.parametrize("quote", ["", "'", '"'])
def test_hash_only_refresh_changes_exact_pin_and_records_provenance(location, quote):
    before = _hash_only_dependent(location, quote=quote)
    after, replacements = _refresh(before)
    assert after == before.replace(
        f"hash: {quote}sha256:{BEFORE}{quote}".encode(),
        f"hash: {quote}sha256:{AFTER}{quote}".encode(),
    )
    assert replacements == (
        {
            "operation": "refresh_proof_import_hash",
            "path": f"rules.0.{location}.atoms.0.import.hash",
            "target": f"{PINNED_IDENTITY}#earned_income_amount",
            "from": f"sha256:{BEFORE}",
            "to": f"sha256:{AFTER}",
            "count": 1,
        },
    )


@pytest.mark.parametrize("location", ["metadata.proof", "proof"])
def test_hash_refresh_refuses_already_stale_pin(location):
    before = _hash_only_dependent(location).replace(
        f"hash: sha256:{BEFORE}".encode(), b"hash: sha256:" + b"3" * 64
    )
    with pytest.raises(SuccessorRepointError, match="already-stale proof import hash"):
        _refresh(before)


@pytest.mark.parametrize("refreshes", [{}, {PINNED_IDENTITY: (BEFORE, BEFORE)}])
def test_hash_only_dependent_requires_an_actual_refresh(refreshes):
    with pytest.raises(SuccessorRepointError, match="no legacy reference to rewrite"):
        _refresh(_hash_only_dependent("metadata.proof"), refreshes=refreshes)


@pytest.mark.parametrize("location", ["metadata.proof", "proof"])
def test_mixed_legacy_rewrite_and_dependent_hash_refresh(location):
    legacy = _legacy_module()
    successor = _successor_module()
    dependent = _dependent(legacy)
    unrelated_identity = "us:statutes/26/152/c"
    rule = _pinning_rule(location, identity=unrelated_identity).encode()
    after, replacements = _refresh(
        dependent + rule, refreshes={unrelated_identity: (BEFORE, AFTER)}
    )
    legacy_postimage, legacy_replacements = _rewrite(dependent, successor)
    assert after == legacy_postimage + rule.replace(
        f"hash: sha256:{BEFORE}".encode(), f"hash: sha256:{AFTER}".encode()
    )
    assert replacements[:-1] == legacy_replacements
    assert replacements[-1] == {
        "operation": "refresh_proof_import_hash",
        "path": f"rules.3.{location}.atoms.0.import.hash",
        "target": f"{unrelated_identity}#earned_income_amount",
        "from": f"sha256:{BEFORE}",
        "to": f"sha256:{AFTER}",
        "count": 1,
    }


@pytest.mark.parametrize("location", ["metadata.proof", "proof"])
def test_hash_only_dependent_keeps_its_existing_successor_namespace(location):
    legacy = _legacy_module()
    successor = _successor_module()
    dependent = _dependent(legacy)
    _request, baseline = _prove(legacy, successor, dependent)
    request = load_repoint_request_payload(
        _envelope(dependents=[DEPENDENT_PRIMARY, IMPORTER_PRIMARY])
    )
    proofs = prove_concept_map(
        legacy_raw=legacy,
        successor_raw=successor,
        request=request,
        dependent_raws={
            DEPENDENT_PRIMARY: dependent,
            IMPORTER_PRIMARY: _hash_only_dependent(location),
        },
    )
    assert proofs == baseline


def test_hash_only_namespace_skip_cannot_hide_legacy_formula_uses():
    legacy = _legacy_module()
    importer = _hash_only_dependent("proof").replace(
        f"  - {SUCCESSOR_IDENTITY}\n".encode(), b""
    )
    importer = importer.replace(
        b"formula: earned_income_amount", b"formula: legacy_cap"
    )
    with pytest.raises(SuccessorRepointError, match="without importing it"):
        _prove(legacy, _successor_module(), importer)
