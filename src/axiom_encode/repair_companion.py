"""Exact companion preservation for explicitly constrained artifact repairs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml

from .checks import declare_check

PRESERVATION_CHECK = declare_check(
    "repair:companion-preservation",
    title="The repair must retain the authenticated companion contract",
    requires="All original case facts and typed assertions, bound to signed final bytes.",
    why="A proof repair must not silently weaken the reviewed calculation coverage.",
    fix="Preserve every original companion case and assertion, then regenerate and verify the signed repair with its exact seed hashes.",
    cause="encoder_output",
    gate="guard-generated",
    site="axiom_encode/cli.py:cmd_guard_generated",
)


def guard_preservation_evidence(
    repo: Path, citation: str, original_rulespec_sha256: str, original_tests_sha256: str
) -> dict[str, object]:
    """Run only inside the supervised guard after its complete checks succeed."""
    from .cli import _load_verified_applied_encoding_manifest_payload
    from .corpus_resolver import read_bounded_regular_file
    from .prepare_signed_backfill import citation_rulespec_path

    primary = Path(citation_rulespec_path(citation))
    companion = primary.with_suffix(".test.yaml")
    manifest = Path(".axiom/encoding-manifests") / primary.with_suffix(".json")
    payload, _, verified_digest, issues = (
        _load_verified_applied_encoding_manifest_payload(repo, manifest.as_posix())
    )
    if issues or payload is None or payload.get("citation") != citation:
        raise ValueError(f"Preserved repair manifest failed verification: {issues}")
    raw = read_bounded_regular_file(
        repo,
        repo / manifest,
        label="preserved repair manifest",
        max_bytes=1024 * 1024,
        required_mode=0o644,
    )
    companion_raw = read_bounded_regular_file(
        repo,
        repo / companion,
        label="preserved repair companion",
        max_bytes=512 * 1024,
        required_mode=0o644,
    )
    companion_digest = hashlib.sha256(companion_raw).hexdigest()
    expected = {
        "original_rulespec_sha256": original_rulespec_sha256,
        "original_tests_sha256": original_tests_sha256,
        "final_tests_sha256": companion_digest,
    }
    actual_digest = verified_preservation_manifest(raw, **expected)
    if actual_digest != verified_digest or [
        item.get("sha256")
        for item in payload.get("applied_files", [])
        if item.get("path") == companion.as_posix()
    ] != [companion_digest]:
        raise ValueError("Preservation bytes changed after manifest verification")
    return {
        "schema": "axiom-encode/verified-repair-companion/v1",
        "citation": citation,
        "manifest_path": manifest.as_posix(),
        "manifest_sha256": actual_digest,
        "companion_path": companion.as_posix(),
        **expected,
    }


def verified_preservation_manifest(
    raw: bytes,
    *,
    original_rulespec_sha256: str,
    original_tests_sha256: str,
    final_tests_sha256: str,
) -> str:
    """Verify the exact packaged manifest with the established protected root."""
    from .cli import (
        _applied_encoding_manifest_signature_issue,
        _applied_encoding_manifest_verifier,
    )

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate key in preservation manifest")
            result[key] = value
        return result

    payload = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(payload, dict) or (
        payload.get("schema_version") != "axiom-encode/applied-rulespec/v5"
        or payload.get("tool") != "axiom-encode encode --apply"
    ):
        raise ValueError("Preservation requires an ordinary signed apply manifest")
    verifier = _applied_encoding_manifest_verifier()
    if verifier is None:
        raise ValueError("Preservation manifest protected trust root is unavailable")
    issue = _applied_encoding_manifest_signature_issue(payload, verifier)
    if issue is not None:
        raise ValueError(f"Preservation manifest {issue}")
    verify_preservation_receipt(
        payload.get("repair_companion_preservation"),
        original_rulespec_sha256=original_rulespec_sha256,
        original_tests_sha256=original_tests_sha256,
        final_tests_sha256=final_tests_sha256,
    )
    return hashlib.sha256(raw).hexdigest()


def preserved_companion_record(original: str, proposed: str) -> dict[str, object]:
    """Verify every retained fact and assertion, then bind both byte identities.

    This is opt-in: ordinary repairs may replace source-invalid test cases.
    Case ordering and YAML presentation may change, but no case field or typed
    output may change. The caller must supply the original authenticated seed,
    never the most recent generated retry candidate.
    """
    from .harness.evals import _preserves_companion_test_cases
    from .harness.validator_pipeline import _safe_load_unique_keys

    try:
        before = _safe_load_unique_keys(original)
        after = _safe_load_unique_keys(proposed)

        def cases(value: object) -> list[dict[str, object]]:
            if isinstance(value, dict) and set(value) == {"cases"}:
                value = value["cases"]
            if not isinstance(value, list) or not value:
                raise ValueError("Retained companion must contain named cases")
            if any(not isinstance(case, dict) for case in value):
                raise ValueError("Retained companion cases must be mappings")
            return value

        originals = cases(before)
        proposed_cases = cases(after)
        contracts = [
            {
                "name": case["name"],
                "period": case.get("period"),
                "input": case.get("input"),
                "required_output": case["output"],
            }
            for case in originals
        ]
        # Use the established typed comparison without changing its tests-only
        # empty-contract rejection or allowing new cases/outputs in this mode.
        if not _preserves_companion_test_cases(original, proposed, contracts):
            raise ValueError("Repair changed retained companion facts or assertions")
        outputs = {case["name"]: set(case["output"]) for case in originals}
        if len(proposed_cases) != len(originals) or any(
            case["name"] not in outputs or set(case["output"]) != outputs[case["name"]]
            for case in proposed_cases
        ):
            raise ValueError("Repair changed the retained companion case/output set")
    except (yaml.YAMLError, KeyError, TypeError, RecursionError) as exc:
        raise ValueError("Repair companion preservation could not be verified") from exc
    return {
        "schema": "axiom-encode/repair-companion-preservation/v1",
        "original_tests_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "final_tests_sha256": hashlib.sha256(proposed.encode()).hexdigest(),
        "cases": len(originals),
        "assertions": sum(len(case["output"]) for case in originals),
    }


def verify_preservation_receipt(
    receipt: object,
    *,
    original_rulespec_sha256: str,
    original_tests_sha256: str,
    final_tests_sha256: str,
) -> None:
    """Bind a signature-verified manifest record to this dispatch and its bytes."""
    expected = {
        "schema": "axiom-encode/repair-companion-preservation/v1",
        "original_rulespec_sha256": original_rulespec_sha256,
        "original_tests_sha256": original_tests_sha256,
        "final_tests_sha256": final_tests_sha256,
    }
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for key, value in expected.items()
        if key != "schema"
    ):
        raise ValueError("Companion preservation requires exact dispatch/file hashes")
    if not isinstance(receipt, dict) or set(receipt) != set(expected) | {
        "cases",
        "assertions",
    }:
        raise ValueError("Missing or malformed signed companion preservation record")
    if any(receipt[key] != value for key, value in expected.items()) or any(
        type(receipt[key]) is not int or receipt[key] <= 0
        for key in ("cases", "assertions")
    ):
        raise ValueError(
            "Signed companion preservation differs from dispatch or applied bytes"
        )
