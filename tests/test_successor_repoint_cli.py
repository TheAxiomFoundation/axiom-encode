"""Writer/verifier round trip for the successor-repoint signed manifest classes."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode import __version__ as AXIOM_ENCODE_TEST_VERSION
from axiom_encode.cli import (
    APPLIED_ENCODING_MANIFEST_SCHEMA,
    APPLIED_ENCODING_OFFICIAL_REPOSITORY,
    SUCCESSOR_REPOINT_DEPENDENT_TOOL,
    SUCCESSOR_REPOINT_RECEIPT_DIR,
    SUCCESSOR_REPOINT_SUCCESSOR_TOOL,
    _applied_encoding_manifest_path,
    _applied_manifest_exact_schema_issues,
    _applied_manifest_tool_execution_issues,
    _build_successor_repoint_provenance,
    _successor_repoint_manifest_issues,
    _successor_repoint_v1_ownership_issues,
)
from axiom_encode.successor_repoint import (
    ENVELOPE_SCHEMA,
    ConceptProof,
    ConceptProofSet,
    load_repoint_request_payload,
)
from tests.eval_evidence_fixtures import (
    TEST_APPLY_PRIVATE_KEY_B64,
    TEST_APPLY_PUBLIC_KEY_B64,
)
from tests.signing_broker_fixtures import SigningBrokerFixture

BROKER = SigningBrokerFixture(
    apply_private_key=TEST_APPLY_PRIVATE_KEY_B64,
    apply_public_key=TEST_APPLY_PUBLIC_KEY_B64,
)
ENCODER_IDENTITY = {
    "repository": APPLIED_ENCODING_OFFICIAL_REPOSITORY,
    "commit": "a" * 40,
    "version": AXIOM_ENCODE_TEST_VERSION,
}
ENCODER_PROVENANCE = {
    "root": "/repo/axiom-encode",
    "commit": "a" * 40,
    "dirty_tracked": False,
    "version": AXIOM_ENCODE_TEST_VERSION,
    "version_commit": "b" * 40,
    "identity_source": "git",
}
WAIVER = "c" * 64
LEGACY_PRIMARY = "us/policies/irs/legacy-table.yaml"
LEGACY_COMPANION = "us/policies/irs/legacy-table.test.yaml"
SUCCESSOR_PRIMARY = "us/policies/irs/page-15.yaml"
DEPENDENT_PRIMARY = "us/statutes/26/32.yaml"
LEGACY_MANIFEST = ".axiom/encoding-manifests/policies/irs/legacy-table.json"
SUCCESSOR_MANIFEST = Path(".axiom/encoding-manifests/us/policies/irs/page-15.json")

CORPUS_RELEASE = SimpleNamespace(
    name="us-2026-01",
    content_sha256="1" * 64,
    selector_sha256="2" * 64,
)

VERIFIED_SUCCESSOR = {
    "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
    "tool": "axiom-encode encode --apply",
    "backend": "codex",
    "applied_files": [{"path": SUCCESSOR_PRIMARY, "sha256": "3" * 64}],
}


def _request():
    return load_repoint_request_payload(
        {
            "schema": ENVELOPE_SCHEMA,
            "legacy_primary": LEGACY_PRIMARY,
            "successor_primary": SUCCESSOR_PRIMARY,
            "dependents": [DEPENDENT_PRIMARY],
            "concept_map": [{"from": "legacy_cap", "to": "successor_cap"}],
            "program_scope_updates": [
                {"program_spec": "programs/us/fiit/fy-2026.yaml", "scope": "federal"}
            ],
        }
    )


def _proof_set():
    return ConceptProofSet(
        proofs=(
            ConceptProof(
                old="legacy_cap",
                new="successor_cap",
                indexed_by_from=None,
                indexed_by_to=None,
                keys=(0,),
                window_start="2026-01-01",
                window_end="2026-12-31",
                probes=("2026-01-01", "2026-12-31"),
                formula_uses=1,
                literal_subscripts=(),
                reference_uses=1,
            ),
        ),
        successor_window_start="2026-01-01",
        successor_window_end="2026-12-31",
        dependent_use_windows=(
            {
                "concept": "legacy_cap",
                "effective_from": "2026-01-01",
                "effective_to": None,
                "extends_past_successor_window": True,
            },
        ),
        post_window_behavior_change=True,
    )


DEPENDENT_RECORDS = [
    {
        "primary": DEPENDENT_PRIMARY,
        "manifests": [
            {
                "path": ".axiom/encoding-manifests/statutes/26/32.json",
                "sha256": "4" * 64,
            }
        ],
        "before_files": [{"path": DEPENDENT_PRIMARY, "sha256": "5" * 64}],
        "live_files": [{"path": DEPENDENT_PRIMARY, "sha256": "6" * 64}],
        "rewrites": [
            {
                "path": DEPENDENT_PRIMARY,
                "before_sha256": "5" * 64,
                "after_sha256": "6" * 64,
                "replacements": [{"from": "legacy_cap", "to": "successor_cap"}],
            }
        ],
    }
]
METADATA_RECORDS = [
    {
        "path": "known-validation-gaps.yaml",
        "before_sha256": "7" * 64,
        "after_sha256": "8" * 64,
        "operations": [{"operation": "remove_legacy_validation_gaps", "count": 1}],
    }
]
PROGRAM_RECORDS = [
    {
        "program_spec": "programs/us/fiit/fy-2026.yaml",
        "scope": "federal",
        "before_sha256": "9" * 64,
        "after_sha256": "a" * 64,
        "removed": ["policies/irs/legacy-table"],
        "added": ["policies/irs/page-15"],
    }
]
VALIDATION_EXECUTION = {
    "schema_version": "axiom-encode/apply-validation-execution/v1",
    "scope": "active_jurisdiction_and_country_ancestors",
    "validated_files": [DEPENDENT_PRIMARY],
    "duration_ms": 1234,
    "axiom_encode": dict(ENCODER_IDENTITY),
}


def _build():
    return _build_successor_repoint_provenance(
        request=_request(),
        head_commit="d" * 40,
        base_tree="e" * 40,
        proof_set=_proof_set(),
        legacy_files={LEGACY_PRIMARY: "f" * 64, LEGACY_COMPANION: "0" * 64},
        legacy_manifest_records=[{"path": LEGACY_MANIFEST, "sha256": "1" * 64}],
        successor_primary=Path(SUCCESSOR_PRIMARY),
        successor_manifest_path=SUCCESSOR_MANIFEST,
        successor_manifest_sha256="2" * 64,
        verified_successor=VERIFIED_SUCCESSOR,
        successor_file_entries=[{"path": SUCCESSOR_PRIMARY, "sha256": "3" * 64}],
        dependent_records=DEPENDENT_RECORDS,
        metadata_records=METADATA_RECORDS,
        program_records=PROGRAM_RECORDS,
        validation_execution=VALIDATION_EXECUTION,
        local_corpus_release=CORPUS_RELEASE,
        waiver_sha256=WAIVER,
        encoder_provenance=ENCODER_PROVENANCE,
        signing_broker=BROKER,
    )


@pytest.fixture
def repo(tmp_path):
    receipt_relative, receipt_bytes, manifests, post_waiver = _build()
    root = tmp_path / "rulespec-us"
    for relative, raw in [(receipt_relative, receipt_bytes), *manifests.items()]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        target.chmod(0o644)
    return SimpleNamespace(
        root=root,
        receipt_relative=receipt_relative,
        receipt_bytes=receipt_bytes,
        manifests=manifests,
        post_waiver=post_waiver,
    )


def _verify(repo, relative: Path, **overrides):
    payload = json.loads((repo.root / relative).read_text())
    return _successor_repoint_manifest_issues(
        payload,
        repo_path=repo.root,
        manifest_label=relative.as_posix(),
        signing_broker=BROKER,
        expected_waiver_set_sha256=overrides.get(
            "expected_waiver_set_sha256", repo.post_waiver
        ),
        local_corpus_release=overrides.get("local_corpus_release", CORPUS_RELEASE),
    )


class TestProvenanceRoundTrip:
    def test_receipt_path_is_content_addressed_and_deterministic(self):
        first, first_bytes, _first_manifests, _waiver = _build()
        second, _second_bytes, _second_manifests, _second_waiver = _build()
        assert first == second
        assert first.parent == SUCCESSOR_REPOINT_RECEIPT_DIR
        # generated_at differs, so the bytes may differ; the identity must not.
        assert len(first.stem) == 64
        assert json.loads(first_bytes.decode())["request_sha256"] == (_request().sha256)

    def test_waiver_digest_follows_the_reconciled_waiver_set(self):
        _relative, receipt_bytes, manifests, post_waiver = _build()
        assert post_waiver == "8" * 64
        assert (
            json.loads(receipt_bytes.decode())["validation_waiver_set_sha256"]
            == post_waiver
        )
        for raw in manifests.values():
            assert json.loads(raw.decode())["validation_waiver_set_sha256"] == (
                post_waiver
            )

    def test_successor_manifest_verifies_against_its_receipt(self, repo):
        assert _verify(repo, SUCCESSOR_MANIFEST) == []

    def test_dependent_manifest_verifies_through_the_cascade_owner(self, repo):
        relative = _applied_encoding_manifest_path(Path(DEPENDENT_PRIMARY))
        assert _verify(repo, relative) == []

    def test_successor_manifest_lists_the_retired_group_as_deleted(self, repo):
        payload = json.loads((repo.root / SUCCESSOR_MANIFEST).read_text())
        deleted = {
            item["path"] for item in payload["applied_files"] if item.get("deleted")
        }
        assert deleted == {LEGACY_PRIMARY, LEGACY_COMPANION}
        live = {
            item["path"]: item["sha256"]
            for item in payload["applied_files"]
            if not item.get("deleted")
        }
        assert live == {
            SUCCESSOR_PRIMARY: "3" * 64,
            "known-validation-gaps.yaml": "8" * 64,
            "programs/us/fiit/fy-2026.yaml": "a" * 64,
        }

    def test_receipt_records_the_window_semantics(self, repo):
        receipt = json.loads(repo.receipt_bytes.decode())
        assert receipt["semantics"]["post_window_behavior_change"] is True
        assert receipt["semantics"]["successor_window"] == {
            "effective_from": "2026-01-01",
            "effective_to": "2026-12-31",
        }
        assert (
            "MissingParameterValue"
            in (receipt["semantics"]["runtime_behavior_outside_successor_window"])
        )


class TestVerifierFailsClosed:
    def test_rejects_a_tampered_receipt(self, repo):
        payload = json.loads(repo.receipt_bytes.decode())
        payload["semantics"]["post_window_behavior_change"] = False
        (repo.root / repo.receipt_relative).write_bytes(
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        )
        assert _verify(repo, SUCCESSOR_MANIFEST) == [
            f"{SUCCESSOR_MANIFEST.as_posix()} successor repoint receipt is invalid"
        ]

    def test_rejects_a_missing_receipt(self, repo):
        (repo.root / repo.receipt_relative).unlink()
        assert _verify(repo, SUCCESSOR_MANIFEST) == [
            f"{SUCCESSOR_MANIFEST.as_posix()} successor repoint receipt is unreadable"
        ]

    def test_rejects_a_stale_waiver_binding(self, repo):
        issues = _verify(repo, SUCCESSOR_MANIFEST, expected_waiver_set_sha256="9" * 64)
        assert any("waiver-set binding is stale" in issue for issue in issues)

    def test_rejects_a_stale_corpus_release_binding(self, repo):
        other = SimpleNamespace(
            name="us-2025-01", content_sha256="4" * 64, selector_sha256="5" * 64
        )
        issues = _verify(repo, SUCCESSOR_MANIFEST, local_corpus_release=other)
        assert any("corpus release binding is stale" in issue for issue in issues)

    def test_rejects_tampered_dependent_live_files(self, repo):
        relative = _applied_encoding_manifest_path(Path(DEPENDENT_PRIMARY))
        payload = json.loads((repo.root / relative).read_text())
        payload["applied_files"] = [{"path": DEPENDENT_PRIMARY, "sha256": "7" * 64}]
        assert _successor_repoint_manifest_issues(
            payload,
            repo_path=repo.root,
            manifest_label=relative.as_posix(),
            signing_broker=BROKER,
            expected_waiver_set_sha256=repo.post_waiver,
            local_corpus_release=CORPUS_RELEASE,
        ) == [f"{relative.as_posix()} successor repoint live files are stale"]

    def test_rejects_a_dependent_manifest_at_an_unauthorised_path(self, repo):
        relative = _applied_encoding_manifest_path(Path(DEPENDENT_PRIMARY))
        payload = json.loads((repo.root / relative).read_text())
        assert _successor_repoint_manifest_issues(
            payload,
            repo_path=repo.root,
            manifest_label=".axiom/encoding-manifests/us/statutes/26/99.json",
            signing_broker=BROKER,
            expected_waiver_set_sha256=repo.post_waiver,
            local_corpus_release=CORPUS_RELEASE,
        ) == [
            ".axiom/encoding-manifests/us/statutes/26/99.json is not uniquely "
            "authorized by its repoint receipt"
        ]

    def test_rejects_a_missing_cascade_owner(self, repo):
        (repo.root / SUCCESSOR_MANIFEST).unlink()
        relative = _applied_encoding_manifest_path(Path(DEPENDENT_PRIMARY))
        assert _verify(repo, relative) == [
            f"{relative.as_posix()} successor repoint cascade owner is unreadable"
        ]

    def test_rejects_a_mutated_binding(self, repo):
        payload = json.loads((repo.root / SUCCESSOR_MANIFEST).read_text())
        payload["successor_repoint"]["legacy_manifest_sha256"] = "9" * 64
        assert _successor_repoint_manifest_issues(
            payload,
            repo_path=repo.root,
            manifest_label=SUCCESSOR_MANIFEST.as_posix(),
            signing_broker=BROKER,
            expected_waiver_set_sha256=repo.post_waiver,
            local_corpus_release=CORPUS_RELEASE,
        ) == [
            f"{SUCCESSOR_MANIFEST.as_posix()} successor repoint legacy ownership "
            "binding is stale"
        ]

    def test_ignores_unrelated_manifest_classes(self, repo):
        assert (
            _successor_repoint_manifest_issues(
                {"tool": "axiom-encode encode --apply", "backend": "codex"},
                repo_path=repo.root,
                manifest_label="x.json",
                signing_broker=BROKER,
                expected_waiver_set_sha256=repo.post_waiver,
                local_corpus_release=CORPUS_RELEASE,
            )
            == []
        )


class TestManifestSchemaGates:
    def test_accepts_both_repoint_manifest_classes(self, repo):
        tools = {
            json.loads((repo.root / relative).read_text())["tool"]
            for relative in repo.manifests
        }
        assert tools == {
            SUCCESSOR_REPOINT_SUCCESSOR_TOOL,
            SUCCESSOR_REPOINT_DEPENDENT_TOOL,
        }
        for relative in [
            SUCCESSOR_MANIFEST,
            _applied_encoding_manifest_path(Path(DEPENDENT_PRIMARY)),
        ]:
            payload = json.loads((repo.root / relative).read_text())
            assert (
                _applied_manifest_exact_schema_issues(
                    payload, manifest_label=relative.as_posix()
                )
                == []
            )
            assert (
                _applied_manifest_tool_execution_issues(
                    payload,
                    manifest_label=relative.as_posix(),
                    expected_encoder_identity=ENCODER_IDENTITY,
                )
                == []
            )

    def test_rejects_an_extra_manifest_field(self, repo):
        payload = json.loads((repo.root / SUCCESSOR_MANIFEST).read_text())
        payload["extra"] = 1
        assert _applied_manifest_exact_schema_issues(
            payload, manifest_label=SUCCESSOR_MANIFEST.as_posix()
        )

    def test_rejects_a_dependent_manifest_that_deletes_files(self, repo):
        relative = _applied_encoding_manifest_path(Path(DEPENDENT_PRIMARY))
        payload = json.loads((repo.root / relative).read_text())
        payload["applied_files"] = [{"path": DEPENDENT_PRIMARY, "deleted": True}]
        issues = _applied_manifest_tool_execution_issues(
            payload,
            manifest_label=relative.as_posix(),
            expected_encoder_identity=ENCODER_IDENTITY,
        )
        assert any("malformed file entries" in issue for issue in issues)

    def test_rejects_a_successor_manifest_without_the_embedded_model_manifest(
        self, repo
    ):
        payload = json.loads((repo.root / SUCCESSOR_MANIFEST).read_text())
        payload["repointed_successor_manifest"] = {"tool": "something-else"}
        issues = _applied_manifest_tool_execution_issues(
            payload,
            manifest_label=SUCCESSOR_MANIFEST.as_posix(),
            expected_encoder_identity=ENCODER_IDENTITY,
        )
        assert any("verified signed-v5 successor" in issue for issue in issues)

    def test_rejects_a_manifest_from_another_encoder_build(self, repo):
        payload = json.loads((repo.root / SUCCESSOR_MANIFEST).read_text())
        issues = _applied_manifest_tool_execution_issues(
            payload,
            manifest_label=SUCCESSOR_MANIFEST.as_posix(),
            expected_encoder_identity={**ENCODER_IDENTITY, "commit": "9" * 40},
        )
        assert any("running pinned encoder" in issue for issue in issues)

    def test_rejects_claimed_receipt_free_provenance(self, repo):
        payload = json.loads((repo.root / SUCCESSOR_MANIFEST).read_text())
        payload["validation_execution"] = dict(VALIDATION_EXECUTION)
        issues = _applied_manifest_tool_execution_issues(
            payload,
            manifest_label=SUCCESSOR_MANIFEST.as_posix(),
            expected_encoder_identity=ENCODER_IDENTITY,
        )
        assert any("receipt-bound provenance" in issue for issue in issues)


# ---------------------------------------------------------------------------
# Legacy v1 ownership admission
# ---------------------------------------------------------------------------

# Shaped after the real
# .axiom/encoding-manifests/policies/irs/rev-proc-2025-32/earned-income-credit.json
# at rulespec-us c654250f: jurisdiction-relative applied_files, no
# axiom_encode_git, null run_id and generated_output_sha256.
REAL_SHAPED_V1 = {
    "applied_files": [
        {"path": "policies/irs/legacy-table.yaml", "sha256": "f" * 64},
        {"path": "policies/irs/legacy-table.test.yaml", "sha256": "0" * 64},
    ],
    "axiom_encode_version": "0.2.64",
    "backend": "codex",
    "citation": "policies/irs/legacy-table",
    "context_manifest_file": None,
    "context_manifest_sha256": None,
    "generated_at": "2026-05-11T12:38:16.848631+00:00",
    "generated_output_file": None,
    "generated_output_root": "/repo/rulespec-us",
    "generated_output_sha256": None,
    "generation_prompt_sha256": None,
    "model": "gpt-5.5",
    "run_id": None,
    "runner": "codex-gpt-5.5",
    "schema_version": "axiom-encode/applied-rulespec/v1",
    "signature": {
        "algorithm": "hmac-sha256",
        "key_id": "axiom-encode-apply-v1",
        "value": "1" * 64,
    },
    "tool": "axiom-encode encode --apply",
    "trace_file": None,
    "trace_sha256": None,
}
EXPECTED_GROUP = {LEGACY_PRIMARY: "f" * 64, LEGACY_COMPANION: "0" * 64}


def _ownership(payload, *, expected=None):
    return _successor_repoint_v1_ownership_issues(
        payload,
        expected_files=expected if expected is not None else EXPECTED_GROUP,
        jurisdiction_prefix="us",
        manifest_label=LEGACY_MANIFEST,
    )


class TestLegacyOwnershipAdmission:
    def test_admits_the_real_v1_shape_in_the_relative_scope(self):
        assert _ownership(REAL_SHAPED_V1) == []

    def test_admits_the_jurisdiction_prefixed_scope(self):
        payload = copy.deepcopy(REAL_SHAPED_V1)
        payload["applied_files"] = [
            {"path": LEGACY_PRIMARY, "sha256": "f" * 64},
            {"path": LEGACY_COMPANION, "sha256": "0" * 64},
        ]
        assert _ownership(payload) == []

    def test_admits_the_manual_attestation_class(self):
        payload = copy.deepcopy(REAL_SHAPED_V1)
        payload["tool"] = "axiom-encode sign-applied-files"
        payload["backend"] = "manual"
        payload["runner"] = "manual-attestation"
        payload["manual_exception"] = "repair"
        assert _ownership(payload) == []

    def test_refuses_an_unknown_field(self):
        payload = copy.deepcopy(REAL_SHAPED_V1)
        payload["surprise"] = 1
        assert _ownership(payload) == [
            f"{LEGACY_MANIFEST} has unknown v1 fields: surprise"
        ]

    def test_refuses_claimed_v5_provenance(self):
        payload = copy.deepcopy(REAL_SHAPED_V1)
        payload["validation_execution"] = {}
        issues = _ownership(payload)
        assert any("claims unsupported provenance" in issue for issue in issues)

    def test_refuses_a_non_v1_schema(self):
        payload = copy.deepcopy(REAL_SHAPED_V1)
        payload["schema_version"] = APPLIED_ENCODING_MANIFEST_SCHEMA
        assert any(
            "applied-rulespec schema v1" in issue for issue in _ownership(payload)
        )

    def test_refuses_an_unknown_tool(self):
        payload = copy.deepcopy(REAL_SHAPED_V1)
        payload["tool"] = "axiom-encode retire"
        assert any("known v1 ownership tool" in issue for issue in _ownership(payload))

    def test_refuses_a_non_hmac_signature(self):
        payload = copy.deepcopy(REAL_SHAPED_V1)
        payload["signature"] = {
            "algorithm": "ed25519-domain-v1",
            "key_id": "sha256:" + "0" * 64,
            "value": "2" * 64,
        }
        assert any("v1 signature provenance" in issue for issue in _ownership(payload))

    def test_refuses_coverage_outside_the_retired_group(self):
        payload = copy.deepcopy(REAL_SHAPED_V1)
        payload["applied_files"].append(
            {"path": "policies/irs/unrelated.yaml", "sha256": "3" * 64}
        )
        assert any(
            "outside the group being retired" in issue for issue in _ownership(payload)
        )

    def test_refuses_a_malformed_applied_files_entry(self):
        payload = copy.deepcopy(REAL_SHAPED_V1)
        payload["applied_files"] = [{"path": "policies/irs/legacy-table.yaml"}]
        assert any(
            "applied_files[0] is malformed" in issue for issue in _ownership(payload)
        )

    def test_refuses_a_non_object_payload(self):
        assert _ownership([]) == [f"{LEGACY_MANIFEST} is not a JSON object"]


def test_digest_helpers_agree_with_the_manifest_bytes(repo):
    payload = json.loads((repo.root / SUCCESSOR_MANIFEST).read_text())
    binding = payload["successor_repoint"]
    assert binding["receipt_sha256"] == hashlib.sha256(repo.receipt_bytes).hexdigest()
    assert binding["receipt_path"] == repo.receipt_relative.as_posix()
    assert binding["successor_primary"] == SUCCESSOR_PRIMARY
    assert binding["legacy_primary"] == LEGACY_PRIMARY


# ---------------------------------------------------------------------------
# Workflow envelope routing
# ---------------------------------------------------------------------------

WORKFLOW = Path(__file__).resolve().parents[1] / (
    ".github/workflows/targeted-signed-reencode.yml"
)
REPOINT_ENVELOPE = json.dumps(
    {
        "schema": ENVELOPE_SCHEMA,
        "legacy_primary": LEGACY_PRIMARY,
        "successor_primary": SUCCESSOR_PRIMARY,
        "dependents": [DEPENDENT_PRIMARY],
        "concept_map": [{"from": "legacy_cap", "to": "successor_cap"}],
        "program_scope_updates": [],
    }
)


def _atomic_source_tail() -> str:
    """Return the workflow step that resolves the successor-repoint envelope."""

    import yaml

    workflow = yaml.safe_load(WORKFLOW.read_text())
    step = next(
        item
        for item in workflow["jobs"]["encode"]["steps"]
        if item.get("name") == "Resolve successor repoint request"
    )
    return step["run"]


def _run_tail(tmp_path, *, envelope: str, **env):
    import subprocess
    import sys
    import sysconfig

    root = WORKFLOW.parents[2]
    workspace = tmp_path / "workspace"
    encoder = workspace / "axiom-encode"
    (encoder / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
    (encoder / "scripts").symlink_to(root / "scripts")
    (encoder / ".venv" / "bin" / "python").symlink_to(sys.executable)
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir(exist_ok=True)
    output = tmp_path / "github-output"
    output.touch()
    env.setdefault("EXISTING_SIGNED_IMPORTS_JSON", env.pop("existing_imports", "[]"))
    script = _atomic_source_tail()
    environment = {
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "RUNNER_TEMP": str(runner_temp),
        "GITHUB_OUTPUT": str(output),
        "ATOMIC_SOURCE_JSON": envelope,
        "REPLACE_RULESPEC_PATH": "",
        "REPLACE_LEGACY_RULESPEC_PATH": "",
        "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
        "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
        "DEPENDENT_CITATION": "",
        "SECOND_DEPENDENT_CITATION": "",
        "REPAIR_RUN_ID": "",
        "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": "[]",
        "EXISTING_SIGNED_IMPORTS_JSON": "[]",
        "PYTHONPATH": os.pathsep.join(
            [str(root / "src"), sysconfig.get_paths()["purelib"]]
        ),
    }
    environment.update({key: str(value) for key, value in env.items()})
    completed = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        cwd=workspace,
        env=environment,
    )
    return completed, runner_temp / "successor-repoint-request.json", output


class TestWorkflowEnvelopeRouting:
    def test_emits_the_request_file_and_sets_the_flag(self, tmp_path):
        completed, request, output = _run_tail(tmp_path, envelope=REPOINT_ENVELOPE)
        assert completed.returncode == 0, completed.stderr
        assert "successor_repoint=true" in output.read_text()
        assert json.loads(request.read_text())["schema"] == ENVELOPE_SCHEMA

    def test_clears_the_flag_for_every_other_source_mode(self, tmp_path):
        completed, request, output = _run_tail(tmp_path, envelope="[]")
        assert completed.returncode == 0, completed.stderr
        assert "successor_repoint=false" in output.read_text()
        assert not request.exists()

    @pytest.mark.parametrize(
        "variable",
        [
            "REPLACE_RULESPEC_PATH",
            "REPLACE_LEGACY_RULESPEC_PATH",
            "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
            "DEPENDENT_CITATION",
            "REPAIR_RUN_ID",
        ],
    )
    def test_refuses_to_mix_with_the_model_lanes(self, tmp_path, variable):
        completed, _request, _output = _run_tail(
            tmp_path, envelope=REPOINT_ENVELOPE, **{variable: "x"}
        )
        assert completed.returncode == 1
        assert "cannot mix with replacement" in completed.stderr

    def test_refuses_to_mix_with_existing_signed_imports(self, tmp_path):
        completed, _request, _output = _run_tail(
            tmp_path,
            envelope=REPOINT_ENVELOPE,
            existing_imports='["us/policies/irs/other.yaml"]',
        )
        assert completed.returncode == 1
        assert "cannot mix with replacement" in completed.stderr

    def test_refuses_a_malformed_envelope(self, tmp_path):
        broken = json.loads(REPOINT_ENVELOPE)
        broken["dependents"] = []
        completed, _request, _output = _run_tail(tmp_path, envelope=json.dumps(broken))
        assert completed.returncode != 0
