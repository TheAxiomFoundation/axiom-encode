"""Verifier, schema-gate, v1-admission and install-surface tests for repoints.

The writer/verifier round trip runs against a real repointed fixture checkout
(``tests/successor_repoint_fixtures.py``) because every receipt claim is now
re-derived from its Git base commit.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from axiom_encode.cli import (
    _SUCCESSOR_REPOINT_REPLAY_CACHE,
    APPLIED_ENCODING_MANIFEST_SCHEMA,
    SUCCESSOR_REPOINT_DEPENDENT_TOOL,
    SUCCESSOR_REPOINT_RETIRED_TOOL,
    _applied_manifest_exact_schema_issues,
    _applied_manifest_tool_execution_issues,
    _sign_applied_encoding_manifest,
    _successor_repoint_fresh_receipt_issues,
    _successor_repoint_manifest_issues,
    _successor_repoint_v1_manifest_entries,
)
from axiom_encode.successor_repoint import ENVELOPE_SCHEMA
from tests.successor_repoint_fixtures import (
    BROKER,
    DEPENDENT,
    DEPENDENT_MANIFEST,
    ENCODER_PROVENANCE,
    LEGACY,
    PROGRAM_SPEC,
    RETIRED_MANIFEST,
    SUCCESSOR,
    SUCCESSOR_MANIFEST,
    build_repoint_fixture,
    git,
    install_repoint_signing,
    run_repoint,
)

ENCODER_IDENTITY = {
    "repository": "github.com/TheAxiomFoundation/axiom-encode",
    "commit": ENCODER_PROVENANCE["commit"],
    "version": ENCODER_PROVENANCE["version"],
}
SUCCESSOR_PRIMARY = SUCCESSOR
DEPENDENT_PRIMARY = DEPENDENT
LEGACY_PRIMARY = LEGACY
LEGACY_MANIFEST = ".axiom/encoding-manifests/policies/irs/legacy-table.json"


@pytest.fixture(scope="module")
def _repointed_template(tmp_path_factory):
    """Build and repoint one fixture checkout for the whole module."""

    root = tmp_path_factory.mktemp("repointed")
    with pytest.MonkeyPatch.context() as monkeypatch:
        fixture = build_repoint_fixture(root, monkeypatch)
        run_repoint(fixture)
    return fixture


@pytest.fixture
def repointed(_repointed_template, tmp_path, monkeypatch):
    """Return a private copy of the repointed checkout for one test."""

    install_repoint_signing(monkeypatch)
    copy_root = tmp_path / "copy"
    copy_root.mkdir()
    repo = copy_root / "rulespec-us"
    shutil.copytree(_repointed_template.repo, repo, symlinks=True)
    return type(_repointed_template)(
        tmp_path=copy_root,
        repo=repo,
        corpus=_repointed_template.corpus,
        engine=_repointed_template.engine,
        request=_repointed_template.request,
        base=_repointed_template.base,
        preimages=_repointed_template.preimages,
        successor_manifest_bytes=_repointed_template.successor_manifest_bytes,
    )


def _receipt_path(fixture) -> Path:
    receipts = sorted((fixture.repo / ".axiom/legacy-successor-repoints").iterdir())
    assert len(receipts) == 1
    return receipts[0].relative_to(fixture.repo)


def _verify(fixture, relative: str, payload: dict | None = None):
    return _successor_repoint_manifest_issues(
        payload
        if payload is not None
        else json.loads((fixture.repo / relative).read_text()),
        repo_path=fixture.repo,
        manifest_label=relative,
        signing_broker=BROKER,
        local_corpus_release=None,
    )


def _rewrite_receipt(fixture, mutate) -> dict:
    """Mutate and re-sign the receipt, rebinding every manifest to it."""

    relative = _receipt_path(fixture)
    receipt = json.loads((fixture.repo / relative).read_text())
    mutate(receipt)
    receipt.pop("signature", None)
    _sign_applied_encoding_manifest(receipt, BROKER)
    raw = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    (fixture.repo / relative).write_bytes(raw)
    for manifest in (RETIRED_MANIFEST, DEPENDENT_MANIFEST):
        payload = json.loads((fixture.repo / manifest).read_text())
        payload["successor_repoint"]["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
        payload.pop("signature", None)
        _sign_applied_encoding_manifest(payload, BROKER)
        (fixture.repo / manifest).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    _SUCCESSOR_REPOINT_REPLAY_CACHE.clear()
    return receipt


class TestProvenanceRoundTrip:
    def test_both_manifest_classes_verify_against_the_replayed_receipt(self, repointed):
        assert _verify(repointed, RETIRED_MANIFEST) == []
        assert _verify(repointed, DEPENDENT_MANIFEST) == []

    def test_the_live_tree_is_the_receipts_exact_postimage(self, repointed):
        assert (
            _successor_repoint_fresh_receipt_issues(
                repointed.repo, _receipt_path(repointed), signing_broker=BROKER
            )
            == []
        )

    def test_the_receipt_is_named_by_its_identity_digest(self, repointed):
        relative = _receipt_path(repointed)
        renamed = relative.with_name("0" * 64 + ".json")
        (repointed.repo / relative).rename(repointed.repo / renamed)
        payload = json.loads((repointed.repo / DEPENDENT_MANIFEST).read_text())
        payload["successor_repoint"]["receipt_path"] = renamed.as_posix()
        issues = _verify(repointed, DEPENDENT_MANIFEST, payload)
        assert any("is not named by its identity digest" in issue for issue in issues)


class TestVerifierFailsClosed:
    def test_rejects_an_unsigned_receipt_edit(self, repointed):
        relative = _receipt_path(repointed)
        receipt = json.loads((repointed.repo / relative).read_text())
        receipt["semantics"]["post_window_behavior_change"] = False
        (repointed.repo / relative).write_text(json.dumps(receipt, indent=2))
        issues = _verify(repointed, DEPENDENT_MANIFEST)
        assert any("is not a signed repoint receipt" in issue for issue in issues)

    @pytest.mark.parametrize(
        ("field", "mutate"),
        [
            (
                "semantics",
                lambda receipt: receipt["semantics"].update(
                    post_window_behavior_change=False
                ),
            ),
            (
                "metadata_reconciliations",
                lambda receipt: receipt["metadata_reconciliations"].pop(),
            ),
            (
                "program_scope_reconciliations",
                lambda receipt: receipt["program_scope_reconciliations"][0].update(
                    after_sha256="0" * 64
                ),
            ),
            (
                "legacy",
                lambda receipt: receipt["legacy"]["manifests"].append(
                    {"path": LEGACY_MANIFEST, "sha256": "1" * 64}
                ),
            ),
            (
                "successor",
                lambda receipt: receipt["successor"]["manifest"].update(citation="x"),
            ),
            (
                "dependents",
                lambda receipt: receipt["dependents"][0]["live_files"][0].update(
                    sha256="2" * 64
                ),
            ),
            ("concept_proofs", lambda receipt: receipt["concept_proofs"].pop()),
            (
                "validation_waiver_set_sha256",
                lambda receipt: receipt.update(validation_waiver_set_sha256="3" * 64),
            ),
        ],
    )
    def test_rejects_a_signed_claim_that_does_not_replay(
        self, repointed, field, mutate
    ):
        _rewrite_receipt(repointed, mutate)
        issues = _verify(repointed, DEPENDENT_MANIFEST)
        assert any(
            f"{field} does not match its base-commit replay" in issue
            for issue in issues
        ), issues

    def test_rejects_a_corpus_release_that_is_not_the_base_toolchains(self, repointed):
        _rewrite_receipt(
            repointed,
            lambda receipt: receipt["corpus_release"].update(name="other-release"),
        )
        issues = _verify(repointed, RETIRED_MANIFEST)
        assert any("base-commit toolchain release" in issue for issue in issues)

    def test_rejects_a_missing_receipt(self, repointed):
        (repointed.repo / _receipt_path(repointed)).unlink()
        issues = _verify(repointed, RETIRED_MANIFEST)
        assert any("is unreadable" in issue for issue in issues)

    def test_rejects_a_stale_receipt_digest(self, repointed):
        payload = json.loads((repointed.repo / RETIRED_MANIFEST).read_text())
        payload["successor_repoint"]["receipt_sha256"] = "9" * 64
        assert _verify(repointed, RETIRED_MANIFEST, payload) == [
            f"{RETIRED_MANIFEST} successor repoint receipt binding is stale"
        ]

    def test_rejects_a_waiver_that_is_not_the_post_repoint_digest(self, repointed):
        payload = json.loads((repointed.repo / DEPENDENT_MANIFEST).read_text())
        payload["validation_waiver_set_sha256"] = "9" * 64
        issues = _verify(repointed, DEPENDENT_MANIFEST, payload)
        assert any("post-repoint waiver set" in issue for issue in issues)

    def test_rejects_tampered_dependent_live_files(self, repointed):
        payload = json.loads((repointed.repo / DEPENDENT_MANIFEST).read_text())
        payload["applied_files"][0]["sha256"] = "7" * 64
        assert _verify(repointed, DEPENDENT_MANIFEST, payload) == [
            f"{DEPENDENT_MANIFEST} successor repoint live files are stale"
        ]

    def test_rejects_a_dependent_manifest_at_an_unauthorised_path(self, repointed):
        payload = json.loads((repointed.repo / DEPENDENT_MANIFEST).read_text())
        label = ".axiom/encoding-manifests/us/statutes/26/99.json"
        assert _verify(repointed, label, payload) == [
            f"{label} is not uniquely authorized by its repoint receipt"
        ]

    def test_rejects_a_retirement_that_does_not_cover_the_group(self, repointed):
        payload = json.loads((repointed.repo / RETIRED_MANIFEST).read_text())
        payload["applied_files"] = payload["applied_files"][:1]
        assert _verify(repointed, RETIRED_MANIFEST, payload) == [
            f"{RETIRED_MANIFEST} successor repoint retirement does not cover "
            "exactly the retired legacy group"
        ]

    def test_rejects_a_retirement_at_another_path(self, repointed):
        payload = json.loads((repointed.repo / RETIRED_MANIFEST).read_text())
        label = ".axiom/encoding-manifests/us/policies/irs/other.json"
        assert _verify(repointed, label, payload) == [
            f"{label} is not the receipt's retired-group manifest"
        ]

    def test_rejects_a_changed_successor_primary(self, repointed):
        target = repointed.repo / SUCCESSOR
        target.write_text(target.read_text() + "# edited\n")
        issues = _verify(repointed, DEPENDENT_MANIFEST)
        assert any(
            "successor model manifest does not verify" in issue for issue in issues
        )

    def test_rejects_a_removed_successor_manifest(self, repointed):
        (repointed.repo / SUCCESSOR_MANIFEST).unlink()
        issues = _verify(repointed, DEPENDENT_MANIFEST)
        assert any("successor manifest is unreadable" in issue for issue in issues)

    def test_rejects_a_mutated_binding_shape(self, repointed):
        payload = json.loads((repointed.repo / RETIRED_MANIFEST).read_text())
        payload["successor_repoint"]["legacy_manifest_sha256"] = "9" * 64
        assert _verify(repointed, RETIRED_MANIFEST, payload) == [
            f"{RETIRED_MANIFEST} successor repoint binding is malformed"
        ]

    def test_fresh_check_refuses_a_drifted_postimage(self, repointed):
        spec = repointed.repo / PROGRAM_SPEC
        spec.write_text(spec.read_text() + "# drift\n")
        issues = _successor_repoint_fresh_receipt_issues(
            repointed.repo, _receipt_path(repointed), signing_broker=BROKER
        )
        assert any(f"postimage is not live: {PROGRAM_SPEC}" in i for i in issues)

    def test_ignores_unrelated_manifest_classes(self, repointed):
        assert (
            _successor_repoint_manifest_issues(
                {"tool": "axiom-encode encode --apply", "backend": "codex"},
                repo_path=repointed.repo,
                manifest_label="x.json",
                signing_broker=BROKER,
                local_corpus_release=None,
            )
            == []
        )


class TestManifestSchemaGates:
    def test_accepts_both_repoint_manifest_classes(self, repointed):
        tools = set()
        for relative in (RETIRED_MANIFEST, DEPENDENT_MANIFEST):
            payload = json.loads((repointed.repo / relative).read_text())
            tools.add(payload["tool"])
            assert (
                _applied_manifest_exact_schema_issues(payload, manifest_label=relative)
                == []
            )
            assert (
                _applied_manifest_tool_execution_issues(
                    payload,
                    manifest_label=relative,
                    expected_encoder_identity=ENCODER_IDENTITY,
                )
                == []
            )
        assert tools == {
            SUCCESSOR_REPOINT_RETIRED_TOOL,
            SUCCESSOR_REPOINT_DEPENDENT_TOOL,
        }

    def test_rejects_an_extra_manifest_field(self, repointed):
        payload = json.loads((repointed.repo / RETIRED_MANIFEST).read_text())
        payload["repointed_successor_manifest"] = {}
        assert _applied_manifest_exact_schema_issues(
            payload, manifest_label=RETIRED_MANIFEST
        )

    def test_rejects_a_dependent_manifest_that_deletes_files(self, repointed):
        payload = json.loads((repointed.repo / DEPENDENT_MANIFEST).read_text())
        payload["applied_files"] = [{"path": DEPENDENT_PRIMARY, "deleted": True}]
        issues = _applied_manifest_tool_execution_issues(
            payload,
            manifest_label=DEPENDENT_MANIFEST,
            expected_encoder_identity=ENCODER_IDENTITY,
        )
        assert any("malformed file entries" in issue for issue in issues)

    def test_rejects_a_retirement_that_claims_live_files(self, repointed):
        payload = json.loads((repointed.repo / RETIRED_MANIFEST).read_text())
        payload["applied_files"] = [{"path": LEGACY_PRIMARY, "sha256": "1" * 64}]
        issues = _applied_manifest_tool_execution_issues(
            payload,
            manifest_label=RETIRED_MANIFEST,
            expected_encoder_identity=ENCODER_IDENTITY,
        )
        assert any("malformed file entries" in issue for issue in issues)
        assert _applied_manifest_exact_schema_issues(
            payload, manifest_label=RETIRED_MANIFEST
        )

    def test_rejects_a_manifest_from_another_encoder_build(self, repointed):
        payload = json.loads((repointed.repo / DEPENDENT_MANIFEST).read_text())
        issues = _applied_manifest_tool_execution_issues(
            payload,
            manifest_label=DEPENDENT_MANIFEST,
            expected_encoder_identity={**ENCODER_IDENTITY, "commit": "9" * 40},
        )
        assert any("running pinned encoder" in issue for issue in issues)

    def test_rejects_claimed_receipt_free_provenance(self, repointed):
        payload = json.loads((repointed.repo / DEPENDENT_MANIFEST).read_text())
        payload["validation_execution"] = {}
        issues = _applied_manifest_tool_execution_issues(
            payload,
            manifest_label=DEPENDENT_MANIFEST,
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
PREFIXED_GROUP = {
    "us/policies/irs/legacy-table.yaml": "f" * 64,
    "us/policies/irs/legacy-table.test.yaml": "0" * 64,
}


def _entries(payload):
    return _successor_repoint_v1_manifest_entries(
        payload, jurisdiction_prefix="us", manifest_label=LEGACY_MANIFEST
    )


class TestLegacyOwnershipAdmission:
    def test_admits_the_real_v1_shape_in_the_relative_scope(self):
        assert _entries(REAL_SHAPED_V1) == (PREFIXED_GROUP, [])

    def test_admits_the_jurisdiction_prefixed_scope(self):
        payload = copy.deepcopy(REAL_SHAPED_V1)
        payload["applied_files"] = [
            {"path": path, "sha256": digest} for path, digest in PREFIXED_GROUP.items()
        ]
        assert _entries(payload) == (PREFIXED_GROUP, [])

    @pytest.mark.parametrize(
        ("tool", "backend", "runner"),
        [
            ("axiom-encode sign-applied-files", "manual", "manual-attestation"),
            (
                "axiom-encode deterministic/manual repair",
                "deterministic",
                "deterministic-repair",
            ),
        ],
    )
    def test_admits_the_manual_and_deterministic_repair_classes(
        self, tool, backend, runner
    ):
        payload = copy.deepcopy(REAL_SHAPED_V1)
        payload.update(tool=tool, backend=backend, runner=runner)
        assert _entries(payload) == (PREFIXED_GROUP, [])

    def test_refuses_mixed_path_scopes(self):
        payload = copy.deepcopy(REAL_SHAPED_V1)
        payload["applied_files"][0]["path"] = "us/policies/irs/legacy-table.yaml"
        assert any("mixes jurisdiction" in issue for issue in _entries(payload)[1])

    def test_refuses_an_unknown_field(self):
        payload = copy.deepcopy(REAL_SHAPED_V1)
        payload["surprise"] = 1
        assert _entries(payload)[1] == [
            f"{LEGACY_MANIFEST} has unknown v1 fields: surprise"
        ]

    def test_refuses_claimed_v5_provenance(self):
        payload = copy.deepcopy(REAL_SHAPED_V1)
        payload["validation_execution"] = {}
        assert any(
            "claims unsupported provenance" in issue for issue in _entries(payload)[1]
        )

    def test_refuses_a_non_v1_schema(self):
        payload = copy.deepcopy(REAL_SHAPED_V1)
        payload["schema_version"] = APPLIED_ENCODING_MANIFEST_SCHEMA
        assert any(
            "applied-rulespec schema v1" in issue for issue in _entries(payload)[1]
        )

    def test_refuses_an_unknown_tool(self):
        payload = copy.deepcopy(REAL_SHAPED_V1)
        payload["tool"] = "axiom-encode retire"
        assert any("known v1 ownership tool" in issue for issue in _entries(payload)[1])

    def test_refuses_a_non_hmac_signature(self):
        payload = copy.deepcopy(REAL_SHAPED_V1)
        payload["signature"] = {
            "algorithm": "ed25519-domain-v1",
            "key_id": "sha256:" + "0" * 64,
            "value": "2" * 64,
        }
        assert any("v1 signature provenance" in issue for issue in _entries(payload)[1])

    def test_refuses_a_malformed_applied_files_entry(self):
        payload = copy.deepcopy(REAL_SHAPED_V1)
        payload["applied_files"] = [
            {"path": "../legacy-table.yaml", "sha256": "0" * 64}
        ]
        assert any(
            "applied_files[0] is malformed" in issue for issue in _entries(payload)[1]
        )

    def test_refuses_a_non_object_payload(self):
        assert _entries([]) == ({}, [f"{LEGACY_MANIFEST} is not a JSON object"])


class TestPlanOwnershipBinding:
    """The plan binds v1 digests: exactly for the legacy group, by union for dependents."""

    def _refusal(self, tmp_path, monkeypatch, mutate) -> str:
        fixture = build_repoint_fixture(tmp_path, monkeypatch)
        mutate(fixture.repo)
        git(fixture.repo, "commit", "-q", "-am", "mutate ownership")
        with pytest.raises(SystemExit) as exit_info:
            run_repoint(fixture)
        return str(exit_info.value)

    def test_refuses_a_legacy_manifest_with_a_stale_digest(self, tmp_path, monkeypatch):
        def mutate(repo):
            target = repo / LEGACY
            target.write_text(target.read_text() + "# edited in place\n")

        message = self._refusal(tmp_path, monkeypatch, mutate)
        assert "does not bind the exact bytes of the legacy group" in message

    def test_refuses_a_legacy_manifest_that_covers_a_subset(
        self, tmp_path, monkeypatch
    ):
        def mutate(repo):
            manifest = repo / (
                ".axiom/encoding-manifests/policies/irs/rev-proc-2025-32/"
                "earned-income-credit.json"
            )
            payload = json.loads(manifest.read_text())
            payload["applied_files"] = payload["applied_files"][:1]
            manifest.write_text(json.dumps(payload, indent=2) + "\n")

        message = self._refusal(tmp_path, monkeypatch, mutate)
        assert "does not bind the exact bytes of the legacy group" in message

    def test_refuses_dependent_bytes_no_v1_manifest_binds(self, tmp_path, monkeypatch):
        def mutate(repo):
            target = repo / DEPENDENT
            target.write_text(target.read_text() + "# edited in place\n")

        message = self._refusal(tmp_path, monkeypatch, mutate)
        assert "dependent bytes are not bound by any v1 ownership manifest" in message


# ---------------------------------------------------------------------------
# Workflow envelope routing
# ---------------------------------------------------------------------------

WORKFLOW = Path(__file__).resolve().parents[1] / (
    ".github/workflows/targeted-signed-reencode.yml"
)
SUCCESSOR_CITATION = "us/guidance/irs/rev-proc-2025-32/page-15"
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


def _run_tail(tmp_path, *, envelope: str, before=None, **env):
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
    successor = workspace / "rulespec-us" / SUCCESSOR_PRIMARY
    successor.parent.mkdir(parents=True, exist_ok=True)
    successor.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        f"    corpus_citation_path: {SUCCESSOR_CITATION}\n"
        "rules: []\n"
    )
    successor.chmod(0o644)
    env.setdefault("EXISTING_SIGNED_IMPORTS_JSON", env.pop("existing_imports", "[]"))
    script = _atomic_source_tail()
    environment = {
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "RUNNER_TEMP": str(runner_temp),
        "GITHUB_OUTPUT": str(output),
        "RULESPEC_CHECKOUT": "rulespec-us",
        "CITATION": SUCCESSOR_CITATION,
        "REVIEW_FINDING": "",
        "DEPENDENT_REVIEW_FINDING": "",
        "SECOND_DEPENDENT_REVIEW_FINDING": "",
        "QUEUE_ITEM_ID": "",
        "QUEUE_MANIFEST_SHA256": "",
        "QUEUE_ITEM_GENERATION_SHA256": "",
        "QUEUE_DISPATCHER_RUN_ID": "",
        "ATOMIC_SOURCE_JSON": envelope,
        "REPLACE_RULESPEC_PATH": "",
        "REPLACE_LEGACY_RULESPEC_PATH": "",
        "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
        "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
        "DEPENDENT_CITATION": "",
        "SECOND_DEPENDENT_CITATION": "",
        "REPAIR_RUN_ID": "",
        "QUEUE_ID": "",
        "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": "[]",
        "EXISTING_SIGNED_IMPORTS_JSON": "[]",
        "PYTHONPATH": os.pathsep.join(
            [str(root / "src"), sysconfig.get_paths()["purelib"]]
        ),
    }
    environment.update({key: str(value) for key, value in env.items()})
    if before is not None:
        before()
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
            "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
            "SECOND_DEPENDENT_CITATION",
            "QUEUE_ID",
            "QUEUE_ITEM_ID",
            "QUEUE_MANIFEST_SHA256",
            "QUEUE_ITEM_GENERATION_SHA256",
            "QUEUE_DISPATCHER_RUN_ID",
            "REVIEW_FINDING",
            "DEPENDENT_REVIEW_FINDING",
            "SECOND_DEPENDENT_REVIEW_FINDING",
        ],
    )
    def test_refuses_to_mix_with_the_model_lanes(self, tmp_path, variable):
        completed, _request, _output = _run_tail(
            tmp_path, envelope=REPOINT_ENVELOPE, **{variable: "x"}
        )
        assert completed.returncode == 1
        assert "cannot mix with replacement" in completed.stderr

    def test_refuses_to_mix_with_retained_successors(self, tmp_path):
        completed, _request, _output = _run_tail(
            tmp_path,
            envelope=REPOINT_ENVELOPE,
            LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON='["us/policies/irs/x.yaml"]',
        )
        assert completed.returncode == 1
        assert "cannot mix with replacement" in completed.stderr

    def test_refuses_a_non_canonical_successor_citation(self, tmp_path):
        successor = tmp_path / "workspace/rulespec-us" / SUCCESSOR_PRIMARY
        completed, _request, _output = _run_tail(
            tmp_path,
            envelope=REPOINT_ENVELOPE,
            CITATION="not a citation",
            before=lambda: successor.write_text(
                "module:\n  source_verification:\n"
                "    corpus_citation_path: not a citation\nrules: []\n"
            ),
        )
        assert completed.returncode == 1
        assert "corpus citation is not canonical" in completed.stderr

    def test_refuses_to_mix_with_existing_signed_imports(self, tmp_path):
        completed, _request, _output = _run_tail(
            tmp_path,
            envelope=REPOINT_ENVELOPE,
            existing_imports='["us/policies/irs/other.yaml"]',
        )
        assert completed.returncode == 1
        assert "cannot mix with replacement" in completed.stderr

    def test_binds_the_dispatch_citation_to_the_successor(self, tmp_path):
        completed, _request, output = _run_tail(
            tmp_path,
            envelope=REPOINT_ENVELOPE,
            CITATION="us/guidance/irs/rev-proc-2025-32/earned-income-credit",
        )
        assert completed.returncode == 1
        assert (
            "successor repoint dispatches must use the successor's corpus "
            f"citation: {SUCCESSOR_CITATION}"
        ) in completed.stderr
        assert "successor_repoint=true" not in output.read_text()

    def test_refuses_a_successor_without_one_corpus_citation(self, tmp_path):
        successor = tmp_path / "workspace/rulespec-us" / SUCCESSOR_PRIMARY
        completed, _request, _output = _run_tail(
            tmp_path,
            envelope=REPOINT_ENVELOPE,
            before=lambda: successor.write_text("format: rulespec/v1\nrules: []\n"),
        )
        assert completed.returncode == 1
        assert "declares no single corpus citation path" in completed.stderr

    def test_refuses_a_malformed_envelope(self, tmp_path):
        broken = json.loads(REPOINT_ENVELOPE)
        broken["dependents"] = []
        completed, _request, _output = _run_tail(tmp_path, envelope=json.dumps(broken))
        assert completed.returncode != 0


def test_a_later_legacy_replacement_treats_a_repoint_receipt_as_provenance(
    tmp_path,
):
    """A repoint receipt naming a module is signed history, never a rewrite."""

    from axiom_encode.cli import _legacy_replacement_reference_inventory_issues
    from tests.test_cli import _git, _minimal_legacy_reference_inventory_fixture

    repo, _base_commit, legacy, replacement = (
        _minimal_legacy_reference_inventory_fixture(tmp_path)
    )
    source = repo / "us/statutes/47:32.yaml"
    source.write_text("format: rulespec/v1\nrules: []\n")
    receipt = repo / ".axiom/legacy-successor-repoints" / ("c" * 64 + ".json")
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"dependents": [{"primary": "us:statutes/47:32"}]}))
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "base with a repoint receipt")
    base_commit = _git(repo, "rev-parse", "HEAD").stdout.strip()
    source.unlink()

    issues = _legacy_replacement_reference_inventory_issues(
        repo,
        base_commit=base_commit,
        authoritative_replacements={"us:statutes/47:32": "us:statutes/47/32"},
        legacy=legacy,
        replacement=replacement,
        allow_pending_scheduled=True,
    )
    assert (
        "legacy replacement reference occurs in persisted provenance "
        f".axiom/legacy-successor-repoints/{'c' * 64}.json"
    ) in issues


# ---------------------------------------------------------------------------
# Review round 2: change-set binding, replay robustness, plan hardening
# ---------------------------------------------------------------------------


def _guard(fixture, base: str) -> list[str]:
    from axiom_encode.cli import guard_generated_change_issues

    return guard_generated_change_issues(
        fixture.repo, corpus_path=fixture.corpus, base_ref=base, head_ref="HEAD"
    )


class TestGuardChangeSetBinding:
    def test_refuses_a_receipt_without_its_transaction(self, tmp_path, monkeypatch):
        fixture = build_repoint_fixture(tmp_path, monkeypatch)
        receipt = (
            fixture.repo / ".axiom/legacy-successor-repoints" / ("a" * 64 + ".json")
        )
        receipt.parent.mkdir(parents=True)
        receipt.write_text("{}\n")
        assert _guard(fixture, fixture.base) == [
            f".axiom/legacy-successor-repoints/{'a' * 64}.json changed without the "
            "successor repoint it records"
        ]

    def test_refuses_a_repoint_manifest_restored_without_its_receipt(self, repointed):
        git(repointed.repo, "add", "-A")
        git(repointed.repo, "commit", "-q", "-m", "repoint")
        repoint_commit = git(repointed.repo, "rev-parse", "HEAD").strip()
        # A later change replaces the dependent and drops the repoint manifest.
        dependent = repointed.repo / DEPENDENT
        dependent.write_text(dependent.read_text() + "# re-encoded\n")
        (repointed.repo / DEPENDENT_MANIFEST).unlink()
        git(repointed.repo, "add", "-A")
        git(repointed.repo, "commit", "-q", "-m", "later change")
        later = git(repointed.repo, "rev-parse", "HEAD").strip()
        # Restoring the old signed repoint manifest and bytes must not pass.
        git(
            repointed.repo,
            "checkout",
            repoint_commit,
            "--",
            DEPENDENT,
            DEPENDENT_MANIFEST,
        )
        issues = _guard(repointed, later)
        assert any(
            f"{DEPENDENT_MANIFEST} changed without introducing its successor repoint "
            "receipt" in issue
            for issue in issues
        ), issues

    def test_refuses_removing_a_receipt(self, repointed):
        from axiom_encode.cli import _successor_repoint_change_set_issues

        git(repointed.repo, "add", "-A")
        git(repointed.repo, "commit", "-q", "-m", "repoint")
        head = git(repointed.repo, "rev-parse", "HEAD").strip()
        receipt = _receipt_path(repointed)
        (repointed.repo / receipt).unlink()
        dependent = repointed.repo / DEPENDENT
        dependent.write_text(dependent.read_text() + "# edited\n")
        assert _guard(repointed, head)
        assert _successor_repoint_change_set_issues(
            repointed.repo,
            receipt_changes=[receipt.as_posix()],
            surviving_manifest_paths=[],
        ) == [
            f"{receipt.as_posix()} is a successor repoint receipt and cannot be removed"
        ]

    def test_change_set_check_requires_a_verifier(self, repointed, monkeypatch):
        from axiom_encode.cli import _successor_repoint_change_set_issues

        monkeypatch.setattr(
            "axiom_encode.cli._applied_encoding_manifest_verifier", lambda: None
        )
        issues = _successor_repoint_change_set_issues(
            repointed.repo,
            receipt_changes=[_receipt_path(repointed).as_posix()],
            surviving_manifest_paths=[],
        )
        assert issues == [
            "A protected signing broker is required to verify a successor repoint "
            "receipt"
        ]


class TestReplayRobustness:
    def test_a_misnamed_copy_is_refused_even_after_the_real_receipt_verified(
        self, repointed
    ):
        real = _receipt_path(repointed)
        assert (
            _successor_repoint_fresh_receipt_issues(
                repointed.repo, real, signing_broker=BROKER
            )
            == []
        )
        copy_path = real.with_name("f" * 64 + ".json")
        shutil.copyfile(repointed.repo / real, repointed.repo / copy_path)
        issues = _successor_repoint_fresh_receipt_issues(
            repointed.repo, copy_path, signing_broker=BROKER
        )
        assert any("is not named by its identity digest" in issue for issue in issues)

    def test_replay_does_not_depend_on_the_checkout_name(self, repointed):
        # The replay takes the ProgramSpec country from the request's
        # jurisdiction.  (A dependent manifest's cascade into the successor's
        # model manifest still needs a rulespec-<country> checkout, as every
        # model-manifest verification does.)
        renamed = repointed.tmp_path / "some-worktree"
        shutil.copytree(repointed.repo, renamed, symlinks=True)
        assert (
            _successor_repoint_manifest_issues(
                json.loads((renamed / RETIRED_MANIFEST).read_text()),
                repo_path=renamed,
                manifest_label=RETIRED_MANIFEST,
                signing_broker=BROKER,
                local_corpus_release=None,
            )
            == []
        )

    def test_a_transient_replay_failure_is_refused_but_not_cached(
        self, repointed, monkeypatch
    ):
        from axiom_encode import cli

        real_plan = cli._plan_successor_repoint

        def fail(*_args, **_kwargs):
            raise RuntimeError("base commit is not fetched")

        monkeypatch.setattr(cli, "_plan_successor_repoint", fail)
        issues = _verify(repointed, RETIRED_MANIFEST)
        assert any("could not be replayed from its base" in issue for issue in issues)
        monkeypatch.setattr(cli, "_plan_successor_repoint", real_plan)
        assert _verify(repointed, RETIRED_MANIFEST) == []


class TestPlanHardening:
    def _refusal(self, tmp_path, monkeypatch, **fixture_options) -> str:
        fixture = build_repoint_fixture(tmp_path, monkeypatch, **fixture_options)
        with pytest.raises(SystemExit) as exit_info:
            run_repoint(fixture)
        return str(exit_info.value)

    def test_refuses_a_metadata_record_its_reconciliation_does_not_remove(
        self, tmp_path, monkeypatch
    ):
        def rename_pending_key(repo):
            path = repo / ".axiom/pending-validation-fingerprints.json"
            payload = json.loads(path.read_text())
            payload["modules"] = {
                "us:policies/irs/rev-proc-2025-32/earned-income-credit": {
                    "fingerprint": "sha256:" + "3" * 64
                }
            }
            path.write_text(json.dumps(payload, indent=2) + "\n")

        message = self._refusal(tmp_path, monkeypatch, before_commit=rename_pending_key)
        assert (
            "left a retired record in .axiom/pending-validation-fingerprints.json"
            in message
        )

    def test_refuses_a_file_an_older_signed_v5_manifest_still_claims(
        self, tmp_path, monkeypatch
    ):
        from tests.test_cli import _signed_manifest_payload

        def add_older_owner(repo):
            payload = _signed_manifest_payload(
                {
                    "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
                    "backend": "codex",
                    "applied_files": [
                        {
                            "path": DEPENDENT,
                            "sha256": hashlib.sha256(
                                (repo / DEPENDENT).read_bytes()
                            ).hexdigest(),
                        }
                    ],
                }
            )
            payload["axiom_encode_git"]["commit"] = "d" * 40
            payload["axiom_encode_git"]["version"] = "0.2.1000"
            payload.pop("signature", None)
            _sign_applied_encoding_manifest(payload, BROKER)
            target = repo / ".axiom/encoding-manifests/us/statutes/26/32-older.json"
            target.write_text(json.dumps(payload, indent=2) + "\n")

        message = self._refusal(tmp_path, monkeypatch, before_commit=add_older_owner)
        assert f"cannot retire or rewrite {DEPENDENT}" in message
        assert "claimed by a signed-v5 manifest" in message

    def test_chains_two_declared_scopes_of_one_program_spec(
        self, tmp_path, monkeypatch
    ):
        from axiom_encode.cli import guard_generated_change_issues
        from axiom_encode.prepare_signed_backfill import successor_repoint_changes
        from tests.successor_repoint_fixtures import ENVELOPE

        def two_scopes(repo):
            spec = repo / PROGRAM_SPEC
            spec.write_text(
                spec.read_text()
                + "  us:\n    - policies/irs/rev-proc-2025-32/earned-income-credit\n"
            )

        envelope = dict(
            ENVELOPE,
            program_scope_updates=[
                {"program_spec": PROGRAM_SPEC, "scope": "federal"},
                {"program_spec": PROGRAM_SPEC, "scope": "us"},
            ],
        )
        fixture = build_repoint_fixture(
            tmp_path, monkeypatch, envelope=envelope, before_commit=two_scopes
        )
        run_repoint(fixture)
        spec = (fixture.repo / PROGRAM_SPEC).read_text()
        assert "earned-income-credit" not in spec
        assert spec.count("policies/irs/rev-proc-2025-32/page-15") == 2
        assert (
            guard_generated_change_issues(
                fixture.repo,
                corpus_path=fixture.corpus,
                base_ref=fixture.base,
                head_ref="HEAD",
            )
            == []
        )
        inventory = successor_repoint_changes(
            fixture.repo, fixture.base, request=fixture.request
        )
        assert {item["path"]: item["sha256"] for item in inventory["changes"]}[
            PROGRAM_SPEC
        ] == hashlib.sha256(spec.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Review round 3: --all, uncommitted repoints, pre-signing workflow steps
# ---------------------------------------------------------------------------


class TestLandedRepoint:
    def test_guard_all_does_not_rebind_a_landed_repoint(self, repointed):
        from axiom_encode.cli import guard_generated_change_issues

        git(repointed.repo, "add", "-A")
        git(repointed.repo, "commit", "-q", "-m", "repoint")
        issues = guard_generated_change_issues(
            repointed.repo, corpus_path=repointed.corpus, all_files=True
        )
        assert not [
            issue
            for issue in issues
            if "successor repoint" in issue or "legacy-successor-repoints" in issue
        ], issues

    def test_encoding_over_an_uncommitted_repoint_is_refused(self, repointed):
        from axiom_encode.cli import (
            _ensure_no_unmanifested_preexisting_rulespec_changes,
        )

        # A repoint lands as one reviewed change set; a later encode must not
        # sign over its dependent before that change set is committed.
        git(repointed.repo, "update-ref", "refs/remotes/origin/main", repointed.base)
        with pytest.raises(
            RuntimeError, match="Refusing to sign over pre-existing RuleSpec"
        ):
            _ensure_no_unmanifested_preexisting_rulespec_changes(
                repointed.repo,
                [(Path(DEPENDENT), [repointed.repo / DEPENDENT])],
                corpus_path=repointed.corpus,
            )

    def test_a_malformed_unrelated_program_spec_does_not_stop_a_repoint(
        self, tmp_path, monkeypatch
    ):
        def add_malformed_spec(repo):
            spec = repo / "programs/us/other/fy-2026.yaml"
            spec.parent.mkdir(parents=True)
            spec.write_text("program: us/other\nperiod: 2026-02-30\nscope: {}\n")

        fixture = build_repoint_fixture(
            tmp_path, monkeypatch, before_commit=add_malformed_spec
        )
        run_repoint(fixture)
        assert not (fixture.repo / LEGACY).exists()


class TestRepointModePreSigningSteps:
    """Every step between the resolver and the signer must accept a repoint."""

    # Steps that still run in repoint mode, each checked below or by its own
    # test: none reads the citation except to parse empty source inputs.
    ALLOWED = {
        "Validate dependent cascade",
        "Fetch pinned signed corpus release object",
        "Provision protected signing supervisor",
        "Verify protected RuleSpec routing",
        "Verify existing signed imports",
    }

    def test_no_unreviewed_step_runs_before_the_signer(self):
        import yaml

        steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["encode"]["steps"]
        names = [step.get("name") for step in steps]
        start = names.index("Resolve successor repoint request")
        end = names.index("Repoint legacy successor")
        for step in steps[start + 1 : end]:
            gated = "successor_repoint != 'true'" in str(step.get("if", ""))
            assert gated or step["name"] in self.ALLOWED, step["name"]

    def test_signed_import_helpers_accept_the_repoint_citation(self, repointed):
        from axiom_encode.prepare_signed_backfill import (
            parse_canonical_refresh_bundle,
            parse_existing_signed_imports,
            parse_source_bundle,
            split_atomic_source_input,
        )

        # The successor exists at the pinned base, exactly as in production.
        git(repointed.repo, "checkout", "-q", repointed.base, "--", ".")
        split = split_atomic_source_input(REPOINT_ENVELOPE_REAL)
        citation = "us/guidance/irs/rev-proc-2025-32/page-15"
        assert (
            parse_source_bundle(
                json.dumps(split["source_bundle"]), primary_citation=citation
            )
            == ()
        )
        assert (
            parse_canonical_refresh_bundle(
                repointed.repo,
                json.dumps(split["canonical_refresh_bundle"]),
                primary_citation=citation,
                primary_rulespec_path="",
                primary_required_test_cases_json="[]",
            )
            == ()
        )
        assert (
            parse_existing_signed_imports(
                repointed.repo, "[]", primary_citation=citation
            )
            == ()
        )


REPOINT_ENVELOPE_REAL = json.dumps(
    {
        "schema": ENVELOPE_SCHEMA,
        "legacy_primary": LEGACY_PRIMARY,
        "successor_primary": SUCCESSOR_PRIMARY,
        "dependents": [DEPENDENT_PRIMARY],
        "concept_map": [{"from": "legacy_cap", "to": "successor_cap"}],
        "program_scope_updates": [],
    }
)
