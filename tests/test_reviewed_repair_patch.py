"""No real package is admitted: fixture registry entries are test-local only."""

import subprocess
from types import MappingProxyType

import pytest

from axiom_encode import reviewed_repair_patch as patch


@pytest.mark.parametrize(
    "raw",
    [
        b"a: 1\na: 2",
        b"a: &x [*x]",
        b"<<: {}",
        b"format: rulespec/v1\nrules: []\nunknown: 1",
    ],
)
def test_rule_yaml_rejects_ambiguous_or_unknown(raw):
    with pytest.raises(ValueError):
        patch.strict_yaml(raw, tests=False)


@pytest.mark.parametrize("raw", [b'{"x": 1, "x": 2}', b'{"x": NaN}', b"[]"])
def test_json_closed(raw):
    with pytest.raises(ValueError):
        patch.strict_json(raw)


def test_registry_is_empty_and_caller_cannot_supply_path(tmp_path):
    assert dict(patch.REVIEWED_REPAIR_PATCHES) == {}
    for value in ("unknown", "../escape", "/tmp/patch"):
        with pytest.raises(ValueError):
            patch.load_registered_patch(tmp_path, value)


def manifest():
    return {
        "schema": patch.SCHEMA,
        "patch_id": "fixture",
        "citation": "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit",
        "rulespec_path": "ca/policies/cra/t1-2025/provincial-territorial-foreign-tax-credit.yaml",
        "rulespec_base_commit": "a" * 40,
        "predecessor": {
            "path": ".axiom/encoding-manifests/policies/cra/t1-2025/provincial-territorial-foreign-tax-credit.json",
            "blob": "a" * 40,
            "sha256": "a" * 64,
            "schema": "axiom-encode/applied-rulespec/v1",
            "rulespec_sha256": "b" * 64,
            "tests_sha256": "c" * 64,
            "canonical_manifest": "absent",
        },
        "engine_ref": "b" * 40,
        "contract_sha256": "b" * 64,
        "seed": {
            "run_id": "123",
            "run_attempt": "1",
            "encoder_commit": "c" * 40,
            **{
                k: "c" * 64
                for k in (
                    "archive_sha256",
                    "rulespec_sha256",
                    "tests_sha256",
                    "issues_sha256",
                )
            },
        },
        "source": {
            "corpus_ref": "d" * 40,
            "release_name": "test-release",
            "release_sha256": "d" * 64,
            "body_sha256": "e" * 64,
            "requested_citation": "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit",
            "resolved_citation": "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit",
        },
        "changes": {
            "top_level": [],
            **{
                k: {a: [] for a in ("added", "replaced", "removed")}
                for k in ("rules", "inputs", "cases")
            },
        },
        "expected_postimage": {
            k: "f" * 64
            for k in (
                "rulespec_sha256",
                "tests_sha256",
                "rulespec_object_sha256",
                "tests_object_sha256",
            )
        },
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(unknown=True),
        lambda d: d.update(rulespec_path="../escape"),
        lambda d: d["seed"].update(run_attempt=True),
        lambda d: d["source"].update(requested_citation="unrelated"),
        lambda d: d["changes"]["rules"].update(added=["x"], removed=["x"]),
        lambda d: d["expected_postimage"].update(tests_sha256="bad"),
    ],
)
def test_manifest_refuses_mutations(mutation):
    value = manifest()
    assert patch.validate_manifest(patch.canonical_json(value), "fixture") == value
    mutation(value)
    with pytest.raises(ValueError):
        patch.validate_manifest(patch.canonical_json(value), "fixture")


@pytest.fixture
def registered(tmp_path, monkeypatch):
    def git(*args):
        return (
            subprocess.check_output(["git", "-C", str(tmp_path), *args])
            .decode()
            .strip()
        )

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    git("remote", "add", "origin", patch.OFFICIAL_REPOSITORY)
    folder = tmp_path / "reviewed-repair-patches/fixture"
    folder.mkdir(parents=True)
    raws = [
        patch.canonical_json(manifest()),
        b"format: rulespec/v1\nrules: []\n",
        b"[]\n",
    ]
    for name, raw in zip(patch.MEMBERS, raws, strict=True):
        (folder / name).write_bytes(raw)
    git("add", ".")
    git("commit", "-qm", "fixture")
    commit = git("rev-parse", "HEAD")
    registration = patch.ReviewedPatchRegistration(
        commit,
        tuple(
            (
                git("rev-parse", f"HEAD:reviewed-repair-patches/fixture/{name}"),
                patch.sha256(raw),
                len(raw),
            )
            for name, raw in zip(patch.MEMBERS, raws, strict=True)
        ),
    )
    monkeypatch.setattr(
        patch, "REVIEWED_REPAIR_PATCHES", MappingProxyType({"fixture": registration})
    )
    return tmp_path, git, registration


def test_registered_exact_git_bytes_not_live_worktree(registered):
    root, _git, reg = registered
    package = patch.load_registered_patch(root, "fixture")
    (root / "reviewed-repair-patches/fixture/rules.patch.yaml").write_text("unreviewed")
    assert patch.load_registered_patch(root, "fixture") == package
    assert package.commit == reg.commit
    view = package.manifest
    view["seed"]["run_id"] = "999"
    assert package.manifest["seed"]["run_id"] == "123"


def test_origin_and_blob_size_are_revalidated(registered, monkeypatch):
    root, git, reg = registered
    bad = list(reg.members)
    bad[0] = (bad[0][0], bad[0][1], True)
    monkeypatch.setattr(
        patch,
        "REVIEWED_REPAIR_PATCHES",
        {"fixture": patch.ReviewedPatchRegistration(reg.commit, tuple(bad))},
    )
    with pytest.raises(ValueError, match="size"):
        patch.load_registered_patch(root, "fixture")
    monkeypatch.setattr(patch, "REVIEWED_REPAIR_PATCHES", {"fixture": reg})
    git("remote", "set-url", "origin", "https://example.invalid/clone.git")
    with pytest.raises(ValueError, match="official origin"):
        patch.load_registered_patch(root, "fixture")


def test_full_measured_context_required(registered):
    root, _, _ = registered
    package = patch.load_registered_patch(root, "fixture")
    fields = {
        "citation",
        "rulespec_path",
        "rulespec_base_commit",
        "predecessor",
        "seed",
        "source",
        "contract_sha256",
        "engine_ref",
    }
    actual = {k: package.manifest[k] for k in fields}
    patch.verify_bindings(package, actual)
    actual["source"]["body_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bindings"):
        patch.verify_bindings(package, actual)


def overlay_fixture(tmp_path):
    import yaml

    from axiom_encode.harness.evals import (
        ValidationRetryCandidate,
        _overlay_validation_retry_candidate,
    )

    seed = b"format: rulespec/v1\nrules:\n- name: kept\n  kind: derived\n  versions:\n  - formula: '1'\n- name: changed\n  kind: derived\n  versions:\n  - formula: '2'\n"
    tests = b"- name: original\n  period: '2025-01-01'\n  input: {}\n  output: {changed: 2}\n"
    rule_patch = b"format: rulespec/v1\nrules:\n- name: changed\n  kind: derived\n  versions:\n  - effective_from: '2025-01-01'\n    effective_to: '2025-12-31'\n    formula: '2'\n"
    tests_patch = b"[]\n"
    target = tmp_path / "reference.yaml"
    target.write_bytes(rule_patch)
    target.with_suffix(".test.yaml").write_bytes(tests_patch)
    _overlay_validation_retry_candidate(
        target,
        artifact_root=tmp_path,
        candidate=ValidationRetryCandidate(seed.decode(), tests.decode()),
    )
    expected = (target.read_bytes(), target.with_suffix(".test.yaml").read_bytes())
    value = manifest()
    value["seed"]["rulespec_sha256"] = patch.sha256(seed)
    value["seed"]["tests_sha256"] = patch.sha256(tests)
    value["changes"]["rules"]["replaced"] = ["changed"]
    value["expected_postimage"] = {
        "rulespec_sha256": patch.sha256(expected[0]),
        "tests_sha256": patch.sha256(expected[1]),
        "rulespec_object_sha256": patch.sha256(
            patch.canonical_json(yaml.safe_load(expected[0]))
        ),
        "tests_object_sha256": patch.sha256(
            patch.canonical_json(yaml.safe_load(expected[1]))
        ),
    }
    package = patch.ReviewedRepairPackage(
        "fixture", "1" * 40, patch.canonical_json(value), rule_patch, tests_patch
    )
    return package, seed, tests, expected


def test_existing_overlay_keeps_omitted_objects_and_exact_inputs(tmp_path):
    import yaml

    package, seed, tests, expected = overlay_fixture(tmp_path)
    actual = patch.reconstruct_postimage(package, rulespec=seed, tests=tests)
    assert actual == expected
    before, after = yaml.safe_load(seed), yaml.safe_load(actual[0])
    assert next(x for x in after["rules"] if x["name"] == "kept") == before["rules"][0]
    assert yaml.safe_load(actual[1]) == yaml.safe_load(tests)
    assert (
        next(x for x in after["rules"] if x["name"] == "changed")["versions"][0][
            "formula"
        ]
        == "2"
    )


@pytest.mark.parametrize(
    "mutation", ["seed", "postimage", "inventory", "unknown_retirement"]
)
def test_overlay_refuses_unreviewed_delta(tmp_path, mutation):
    from dataclasses import replace

    package, seed, tests, _ = overlay_fixture(tmp_path)
    value = package.manifest
    if mutation == "seed":
        seed += b"\n"
    elif mutation == "postimage":
        value["expected_postimage"]["tests_sha256"] = "0" * 64
    elif mutation == "inventory":
        value["changes"]["rules"]["replaced"] = []
    else:
        package = replace(
            package,
            rules_bytes=b"format: rulespec/v1\nrules:\n- name: absent\n  repair_remove: true\n",
        )
    package = replace(package, manifest_bytes=patch.canonical_json(value))
    with pytest.raises(ValueError):
        patch.reconstruct_postimage(package, rulespec=seed, tests=tests)


def registered_overlay(tmp_path, monkeypatch):
    package, seed, tests, expected = overlay_fixture(tmp_path)
    registration = patch.ReviewedPatchRegistration(
        package.commit,
        tuple(
            (patch.git_blob_id(raw), patch.sha256(raw), len(raw))
            for raw in (
                package.manifest_bytes,
                package.rules_bytes,
                package.tests_bytes,
            )
        ),
    )
    monkeypatch.setattr(patch, "REVIEWED_REPAIR_PATCHES", {"fixture": registration})
    return package, seed, tests, expected


def test_offline_receipt_reconstructs_exact_postimage(tmp_path, monkeypatch):
    package, seed, tests, expected = registered_overlay(tmp_path, monkeypatch)
    binding = patch.build_repair_binding(package, rulespec=seed, tests=tests)
    restored, actual = patch.verify_repair_binding(binding)
    assert restored == package
    assert actual == expected


@pytest.mark.parametrize(
    "mutation",
    [
        "list_id",
        "bad_commit",
        "member_count",
        "bad_blob",
        "bool_size",
        "base64",
        "seed",
        "extra",
        "oversize",
    ],
)
def test_offline_receipt_refuses_malformed_or_changed_evidence(
    tmp_path, monkeypatch, mutation
):
    from dataclasses import replace

    package, seed, tests, _ = registered_overlay(tmp_path, monkeypatch)
    binding = patch.build_repair_binding(package, rulespec=seed, tests=tests)
    registration = patch.REVIEWED_REPAIR_PATCHES["fixture"]
    if mutation == "list_id":
        binding["patch_id"] = []
    elif mutation == "bad_commit":
        binding["package_commit"] = "HEAD"
    elif mutation == "member_count":
        monkeypatch.setattr(
            patch,
            "REVIEWED_REPAIR_PATCHES",
            {"fixture": replace(registration, members=())},
        )
    elif mutation in {"bad_blob", "bool_size"}:
        members = list(registration.members)
        blob, digest, size = members[0]
        members[0] = (
            "not-a-blob" if mutation == "bad_blob" else blob,
            digest,
            True if mutation == "bool_size" else size,
        )
        monkeypatch.setattr(
            patch,
            "REVIEWED_REPAIR_PATCHES",
            {"fixture": replace(registration, members=tuple(members))},
        )
    elif mutation == "base64":
        binding["seed"]["tests"] += "\n"
    elif mutation == "seed":
        import base64

        binding["seed"]["tests"] = base64.b64encode(tests + b"\n").decode()
    elif mutation == "extra":
        binding["seed"]["caller_authorized"] = True
    else:
        binding["seed"]["tests"] = "A" * patch.MAX_RECEIPT_BYTES
    with pytest.raises(ValueError):
        patch.verify_repair_binding(binding)


def test_explicit_non_model_context_and_metadata_are_bound(tmp_path, monkeypatch):
    from axiom_encode.cli import _apply_result_metadata

    package, seed, tests, _ = registered_overlay(tmp_path, monkeypatch)
    binding = patch.build_repair_binding(package, rulespec=seed, tests=tests)
    result = patch.ReviewedRepairResult(
        "output.yaml", package.manifest["citation"], {}, "context.json", binding
    )
    metadata = _apply_result_metadata(result)
    assert metadata["backend"] is None and metadata["model"] is None
    assert metadata["tool"] == patch.TOOL
    assert metadata["reviewed_repair"] == binding
    result.backend = "openai"
    with pytest.raises(RuntimeError, match="non-model"):
        _apply_result_metadata(result)


def lineage_fixture(tmp_path, monkeypatch):
    from dataclasses import replace

    workspace = tmp_path / "overlay"
    workspace.mkdir()
    package, seed, tests, expected = overlay_fixture(workspace)
    repo = tmp_path / "rulespec-ca"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args]).decode().strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.test")
    git(
        "remote",
        "add",
        "origin",
        "https://github.com/TheAxiomFoundation/rulespec-ca.git",
    )
    value = package.manifest
    target = value["rulespec_path"]
    companion = target.removesuffix(".yaml") + ".test.yaml"
    predecessor = value["predecessor"]["path"]
    old = patch.canonical_json(
        {"schema_version": value["predecessor"]["schema"], "tool": "sign-applied-files"}
    )
    for name, raw in ((target, seed), (companion, tests), (predecessor, old)):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    git("add", ".")
    git("commit", "-qm", "Historical fixture")
    value["rulespec_base_commit"] = git("rev-parse", "HEAD")
    value["predecessor"].update(
        blob=git("rev-parse", f"HEAD:{predecessor}"),
        sha256=patch.sha256(old),
        rulespec_sha256=patch.sha256(seed),
        tests_sha256=patch.sha256(tests),
    )
    package = replace(package, manifest_bytes=patch.canonical_json(value))
    registration = patch.ReviewedPatchRegistration(
        package.commit,
        tuple(
            (patch.git_blob_id(raw), patch.sha256(raw), len(raw))
            for raw in (
                package.manifest_bytes,
                package.rules_bytes,
                package.tests_bytes,
            )
        ),
    )
    monkeypatch.setattr(patch, "REVIEWED_REPAIR_PATCHES", {"fixture": registration})
    binding = patch.build_repair_binding(package, rulespec=seed, tests=tests)
    source = value["source"]
    payload = {
        "citation": value["citation"],
        "reviewed_repair": binding,
        "applied_files": [
            {"path": path, "sha256": patch.sha256(raw)}
            for path, raw in zip((target, companion), expected, strict=True)
        ],
        "source_attestation": {
            "requested_corpus_citation_path": source["requested_citation"],
            "resolved_corpus_citation_path": source["resolved_citation"],
            "corpus_release": source["release_name"],
            "corpus_release_content_sha256": source["release_sha256"],
            "source_sha256": source["body_sha256"],
            "generation_input_sha256": source["body_sha256"],
        },
        "validation_execution": {
            "axiom_rules_engine": {"commit": value["engine_ref"]},
            "rulespec_dependencies": [],
        },
    }
    canonical = ".axiom/encoding-manifests/" + target.removesuffix(".yaml") + ".json"
    return repo, package, payload, canonical


def test_registered_historical_lineage_is_not_relabelled_or_removed(
    tmp_path, monkeypatch
):
    repo, package, payload, canonical = lineage_fixture(tmp_path, monkeypatch)
    before = patch.verify_predecessor(repo, package)
    patch.verify_manifest_reconstruction(payload, repo, canonical)
    assert patch.verify_predecessor(repo, package) == before
    assert not (repo / canonical).exists()


@pytest.mark.parametrize(
    "mutation",
    ["source", "engine", "files", "extra_file", "predecessor", "origin", "path"],
)
def test_manifest_reconstruction_refuses_lineage_or_binding_drift(
    tmp_path, monkeypatch, mutation
):
    repo, package, payload, canonical = lineage_fixture(tmp_path, monkeypatch)
    if mutation == "source":
        payload["source_attestation"]["source_sha256"] = "0" * 64
    elif mutation == "engine":
        payload["validation_execution"]["axiom_rules_engine"]["commit"] = "0" * 40
    elif mutation == "files":
        payload["applied_files"][0]["sha256"] = "0" * 64
    elif mutation == "extra_file":
        payload["applied_files"].append({"path": "ca/other.yaml", "sha256": "0" * 64})
    elif mutation == "predecessor":
        (repo / package.manifest["predecessor"]["path"]).write_bytes(b"{}")
    elif mutation == "origin":
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "remote",
                "set-url",
                "origin",
                "https://example.invalid/other",
            ],
            check=True,
        )
    else:
        canonical = canonical.replace("/ca/", "/us/")
    with pytest.raises(ValueError):
        patch.verify_manifest_reconstruction(payload, repo, canonical)


@pytest.mark.parametrize(
    "failure", ["validation", "supplemental", "postimage", "closure", "shrink", "none"]
)
def test_command_preserves_validation_and_transaction_boundaries(
    tmp_path, monkeypatch, failure
):
    """Wiring/rollback test; mocked execution is not runtime or source clearance."""
    import json
    from contextlib import contextmanager
    from dataclasses import replace
    from types import SimpleNamespace

    from axiom_encode import cli as c

    repo, package, _payload, canonical = lineage_fixture(tmp_path, monkeypatch)
    value = package.manifest
    target = value["rulespec_path"]
    companion = target.removesuffix(".yaml") + ".test.yaml"
    before = {
        p: (repo / p).read_bytes()
        for p in (target, companion, value["predecessor"]["path"])
    }
    source_body = "A bounded source fixture.\n"
    source_hash = patch.sha256(source_body.encode())
    roots = []
    for name in ("engine", "corpus"):
        root = tmp_path / name
        root.mkdir()
        subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.test",
                "commit",
                "--allow-empty",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        roots.append(root)
    engine, corpus = roots
    value["engine_ref"] = subprocess.check_output(
        ["git", "-C", str(engine), "rev-parse", "HEAD"], text=True
    ).strip()
    value["source"].update(
        body_sha256=source_hash,
        corpus_ref=subprocess.check_output(
            ["git", "-C", str(corpus), "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    raw_contract = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v6",
            "repair_mode": "full_artifact",
            "transaction": {
                "schema": "axiom-encode/atomic-source-transaction/v3",
                "source_bundle": [],
                "canonical_refresh_bundle": [],
                "primary_required_test_cases": [
                    {
                        "name": "original",
                        "period": {
                            "period_kind": "tax_year",
                            "start": "2025-01-01",
                            "end": "2025-12-31",
                        },
                        "input": {},
                        "required_output": {"changed": 2},
                    }
                ],
                "require_complete_source_unit": True,
            },
        }
    )
    value["contract_sha256"] = patch.sha256(
        patch.canonical_json(json.loads(raw_contract))
    )
    evidence = tmp_path / "seed"
    evidence.mkdir()
    for name, raw in (
        ("rules.yaml", before[target]),
        ("tests.yaml", before[companion]),
        ("issues.json", b"[]"),
        ("archive.tar", b"authenticated fixture archive"),
    ):
        (evidence / name).write_bytes(raw)
    import io
    import tarfile

    archive_buffer = io.BytesIO()
    prefix = "generated/target/final-rejected-candidate/"
    with tarfile.open(fileobj=archive_buffer, mode="w") as archive:
        for name, raw in (
            (prefix + target.split("/", 1)[1], before[target]),
            (prefix + companion.split("/", 1)[1], before[companion]),
            (prefix + "issues.json", b"[]"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    archive_raw = archive_buffer.getvalue()
    (evidence / "archive.tar").write_bytes(archive_raw)
    value["seed"].update(
        issues_sha256=patch.sha256(b"[]"), archive_sha256=patch.sha256(archive_raw)
    )
    package = replace(package, manifest_bytes=patch.canonical_json(value))
    registration = patch.ReviewedPatchRegistration(
        package.commit,
        tuple(
            (patch.git_blob_id(raw), patch.sha256(raw), len(raw))
            for raw in (
                package.manifest_bytes,
                package.rules_bytes,
                package.tests_bytes,
            )
        ),
    )
    monkeypatch.setattr(patch, "REVIEWED_REPAIR_PATCHES", {"fixture": registration})
    monkeypatch.setattr(patch, "load_registered_patch", lambda *_: package)
    calls = []
    source = value["source"]
    attestation = {
        "requested_corpus_citation_path": source["requested_citation"],
        "resolved_corpus_citation_path": source["resolved_citation"],
        "corpus_release": source["release_name"],
        "corpus_release_content_sha256": source["release_sha256"],
        "source_sha256": source_hash,
    }
    monkeypatch.setattr(
        c,
        "load_rulespec_local_corpus_release",
        lambda *_: SimpleNamespace(
            name=source["release_name"], content_sha256=source["release_sha256"]
        ),
    )
    monkeypatch.setattr(
        c,
        "resolve_corpus_source_unit",
        lambda *_: SimpleNamespace(
            body=source_body,
            requested=value["citation"],
            citation_path=value["citation"],
            source_attestation=attestation,
        ),
    )

    def replacement(*args, **kwargs):
        calls.append("ordinary-replacement")
        return SimpleNamespace(
            relative_output=c.Path(*c.Path(target).parts[1:]),
            legacy_replacement=None,
            retired_source_admission=None,
        )

    monkeypatch.setattr(c, "_resolve_encode_replacement_target", replacement)
    monkeypatch.setattr(
        c, "_manifest_primary_source_verifications", lambda **_: ([], "rulespec-ca/ca")
    )

    def validate(result, **kwargs):
        calls.append("full-validation")
        assert kwargs["require_complete_source_unit"] is True
        assert kwargs["deferred_output_review_contract"].citation == value["citation"]
        setattr(
            result,
            c._APPLY_VALIDATION_SNAPSHOT_ATTR,
            {
                "manifest_validation_execution": {
                    "axiom_rules_engine": {"commit": value["engine_ref"]},
                    "rulespec_dependencies": [],
                }
            },
        )
        if failure == "shrink":
            # Simulate a concurrent canonical receipt gaining extra coverage.
            manifest = repo / canonical
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text(
                json.dumps(
                    {
                        "applied_files": [
                            {"path": target},
                            {"path": companion},
                            {"path": "ca/policies/other.yaml"},
                        ]
                    }
                )
            )
        if failure == "postimage":
            c.Path(result.output_file).write_text("changed")
        return (
            failure != "validation",
            ["fixture rejection"] if failure == "validation" else [],
            {c.Path("other.yaml"): "changed"} if failure == "supplemental" else {},
        )

    monkeypatch.setattr(c, "_run_generated_encoding_overlay_validation", validate)
    monkeypatch.setattr(
        c,
        "_require_unchanged_successful_apply_validation",
        lambda *a, **k: calls.append("snapshot"),
    )
    monkeypatch.setattr(c, "_require_clean_axiom_encode_git_provenance", lambda: {})
    monkeypatch.setattr(c, "verify_rulespec_validation_waiver_set", lambda *_: "a" * 64)

    @contextmanager
    def signer(**kwargs):
        calls.append("signer")
        yield object()

    monkeypatch.setattr(c, "_isolated_apply_manifest_signer", signer)

    def sign(payload, _signer):
        calls.append("sign")
        payload["signature"] = {"fixture": True}

    monkeypatch.setattr(c, "_sign_applied_encoding_manifest", sign)

    def closure(*args, **kwargs):
        calls.append("full-post-install-closure")
        assert kwargs["corpus_path"] == corpus
        assert len(kwargs["planned"]) == 2
        if failure == "closure":
            raise RuntimeError("fixture post-install closure rejection")

    monkeypatch.setattr(c, "_require_apply_post_install_closure", closure)
    monkeypatch.setattr(c, "_current_guard_encoder_execution_identity", lambda: {})
    monkeypatch.setattr(
        c,
        "_load_verified_applied_encoding_manifest_payload",
        lambda *a, **k: ({}, "", "", []),
    )
    args = SimpleNamespace(
        patch_checkout=tmp_path,
        patch_id="fixture",
        repo=repo,
        axiom_rules_path=engine,
        corpus_path=corpus,
        run_id="456",
        source_bundle_json=raw_contract,
        seed_rulespec=evidence / "rules.yaml",
        seed_tests=evidence / "tests.yaml",
        seed_issues=evidence / "issues.json",
        seed_archive=evidence / "archive.tar",
    )
    if failure == "none":
        patch.apply_reviewed_repair(args)
        assert (
            calls.index("full-validation")
            < calls.index("sign")
            < calls.index("full-post-install-closure")
        )
        assert (repo / canonical).exists()
        assert (repo / value["predecessor"]["path"]).read_bytes() == before[
            value["predecessor"]["path"]
        ]
    else:
        with pytest.raises(
            RuntimeError, match="Refusing to shrink" if failure == "shrink" else None
        ):
            patch.apply_reviewed_repair(args)
        assert (repo / canonical).exists() is (failure == "shrink")
        assert {p: (repo / p).read_bytes() for p in before} == before
        if failure != "closure":
            assert "signer" not in calls
    assert calls[0] == "ordinary-replacement"


def test_actual_signed_guard_verifies_reconstruction_and_rejects_signed_wrong_postimage(
    tmp_path, monkeypatch
):
    """Real fixture signature/source release and ordinary manifest verifier, no mocked loader."""
    import json
    from dataclasses import replace

    import yaml

    from axiom_encode import cli as c
    from axiom_encode.harness.evals import (
        ValidationRetryCandidate,
        _overlay_validation_retry_candidate,
    )
    from tests.test_cli import (
        TEST_APPLY_SIGNING_BROKER,
        TEST_PINNED_ENCODER_IDENTITY,
        _bind_test_corpus_release,
    )

    repo, package, _payload, canonical = lineage_fixture(tmp_path, monkeypatch)
    value = package.manifest
    target = value["rulespec_path"]
    companion = target.removesuffix(".yaml") + ".test.yaml"
    seed = (repo / target).read_bytes()
    tests = (repo / companion).read_bytes()
    corpus = tmp_path / "signed-corpus"
    release = _bind_test_corpus_release(
        repo, corpus, citation_path=value["citation"], body="Reviewed source fixture.\n"
    )
    source = c.resolve_corpus_source_unit(value["citation"], release)
    source_hash = patch.sha256(source.body.encode())
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "Bind signed fixture release"],
        check=True,
    )
    value["rulespec_base_commit"] = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    value["source"].update(
        release_name=release.name,
        release_sha256=release.content_sha256,
        body_sha256=source_hash,
    )
    rules = yaml.safe_load(package.rules_bytes)
    rules["module"] = {
        "source_verification": {
            "corpus_citation_path": value["citation"],
            "source_sha256": source_hash,
        }
    }
    rules_patch = yaml.safe_dump(rules, sort_keys=False).encode()
    work = tmp_path / "signed-overlay"
    work.mkdir()
    output = work / "output.yaml"
    output.write_bytes(rules_patch)
    output.with_suffix(".test.yaml").write_bytes(package.tests_bytes)
    _overlay_validation_retry_candidate(
        output,
        artifact_root=work,
        candidate=ValidationRetryCandidate(seed.decode(), tests.decode()),
    )
    post = output.read_bytes(), output.with_suffix(".test.yaml").read_bytes()
    value["changes"]["top_level"] = ["module"]
    value["expected_postimage"] = {
        "rulespec_sha256": patch.sha256(post[0]),
        "tests_sha256": patch.sha256(post[1]),
        "rulespec_object_sha256": patch.sha256(
            patch.canonical_json(yaml.safe_load(post[0]))
        ),
        "tests_object_sha256": patch.sha256(
            patch.canonical_json(yaml.safe_load(post[1]))
        ),
    }
    package = replace(
        package, manifest_bytes=patch.canonical_json(value), rules_bytes=rules_patch
    )
    monkeypatch.setattr(
        patch,
        "REVIEWED_REPAIR_PATCHES",
        {
            "fixture": patch.ReviewedPatchRegistration(
                package.commit,
                tuple(
                    (patch.git_blob_id(raw), patch.sha256(raw), len(raw))
                    for raw in (
                        package.manifest_bytes,
                        package.rules_bytes,
                        package.tests_bytes,
                    )
                ),
            )
        },
    )
    binding = patch.build_repair_binding(package, rulespec=seed, tests=tests)
    identity = TEST_PINNED_ENCODER_IDENTITY
    waiver = c.verify_rulespec_validation_waiver_set(repo)
    attestation = {
        **source.source_attestation,
        "generation_input_sha256": source_hash,
        "rulespec_root": "rulespec-ca/ca",
    }
    # Exercise the real overlay dispatcher/source-context/staging path. The
    # controlled pipeline is explicitly not real engine or source clearance.
    from types import SimpleNamespace

    output_root = tmp_path / "flow-output"
    generated = output_root / "reviewed-repair" / target.split("/", 1)[1]
    generated.parent.mkdir(parents=True)
    generated.write_bytes(post[0])
    generated.with_suffix(".test.yaml").write_bytes(post[1])
    context = output_root / "context.json"
    (output_root / "source.txt").write_text(source.body)
    context.write_text(
        json.dumps(
            {
                "source_text_file": "source.txt",
                "source_metadata": {"source_attestation": attestation},
            }
        )
    )
    result = patch.ReviewedRepairResult(
        str(generated), value["citation"], attestation, str(context), binding
    )
    setattr(result, c._REPLACEMENT_OVERLAY_SCOPE_ATTR, True)
    reached = []

    class ControlledPipeline:
        def __init__(self, **kwargs):
            assert kwargs["require_policy_proofs"] is True
            assert kwargs["require_complete_source_unit"] is True
            assert kwargs["source_metadata"]["source_attestation"] == attestation

        def validate(self, path, *, skip_reviewers):
            assert skip_reviewers is True
            assert path.read_bytes() == post[0]
            assert path.with_suffix(".test.yaml").read_bytes() == post[1]
            reached.append(path)
            return SimpleNamespace(all_passed=True, results={})

    with monkeypatch.context() as scoped:
        scoped.setattr(c, "ValidatorPipeline", ControlledPipeline)
        scoped.setattr(
            c,
            "_record_successful_apply_validation",
            lambda *a, **k: reached.append("snapshot"),
        )
        ok, failures, supplemental = c._run_generated_encoding_overlay_validation(
            result,
            output_root=output_root,
            policy_repo_path=repo / "ca",
            axiom_rules_path=tmp_path / "engine",
            local_corpus_release=release,
            validate_dependents=False,
            require_complete_source_unit=True,
        )
        assert (ok, failures, supplemental) == (True, [], {})
        assert len(reached) == 2 and reached[-1] == "snapshot"
        assert generated.read_bytes() == post[0]
        assert generated.with_suffix(".test.yaml").read_bytes() == post[1]
        # Explicit context mismatch must refuse before reaching the pipeline.
        context.write_text(
            json.dumps(
                {
                    "source_text_file": "source.txt",
                    "source_metadata": {"source_attestation": {}},
                }
            )
        )
        ok, failures, supplemental = c._run_generated_encoding_overlay_validation(
            result,
            output_root=output_root,
            policy_repo_path=repo / "ca",
            axiom_rules_path=tmp_path / "engine",
            local_corpus_release=release,
            validate_dependents=False,
            require_complete_source_unit=True,
        )
        assert not ok and any("does not match" in issue for issue in failures)
        assert len(reached) == 2
    payload = {
        "schema_version": c.APPLIED_ENCODING_MANIFEST_SCHEMA,
        "generated_at": "2026-10-19T00:00:00+00:00",
        "tool": patch.TOOL,
        "citation": value["citation"],
        "run_id": "456",
        "axiom_encode_version": c.__version__,
        "axiom_encode_git": {
            "root": "/fixture/encoder",
            "commit": identity["commit"],
            "version": c.__version__,
            "version_commit": identity["commit"],
            "dirty_tracked": False,
            "identity_source": "git",
        },
        "reviewed_repair": binding,
        c.VALIDATION_WAIVER_SET_SHA256_FIELD: waiver,
        "applied_files": [
            {"path": path, "sha256": patch.sha256(raw)}
            for path, raw in zip((target, companion), post, strict=True)
        ],
        "source_attestation": attestation,
        "validation_execution": {
            "schema": "axiom-encode/apply-validation-execution/v1",
            "axiom_encode": {**identity, "identity_source": "git"},
            "axiom_rules_engine": {
                "repository": "github.com/TheAxiomFoundation/axiom-rules-engine",
                "commit": value["engine_ref"],
            },
            "policy_pre_apply": {
                "rulespec_root": "rulespec-ca/ca",
                "pre_apply_content_sha256": "e" * 64,
                "pre_apply_file_count": 2,
                "toolchain_contract_sha256": "d" * 64,
                "validation_waiver_set_sha256": waiver,
            },
            "rulespec_dependencies": [],
        },
    }
    for path, raw in zip((target, companion), post, strict=True):
        (repo / path).write_bytes(raw)
    manifest_path = repo / canonical
    manifest_path.parent.mkdir(parents=True)
    c._sign_applied_encoding_manifest(payload, TEST_APPLY_SIGNING_BROKER)
    manifest_path.write_bytes(patch.canonical_json(payload))
    verified, _, _, issues = c._load_verified_applied_encoding_manifest_payload(
        repo,
        canonical,
        signing_broker=TEST_APPLY_SIGNING_BROKER,
        expected_encoder_identity=identity,
        expected_waiver_set_sha256=waiver,
        local_corpus_release=release,
    )
    assert issues == []
    assert verified == payload
    monkeypatch.setattr(
        c, "_applied_encoding_manifest_verifier", lambda: TEST_APPLY_SIGNING_BROKER
    )
    monkeypatch.setattr(
        c, "_read_only_guard_encoder_execution_identity", lambda *_args: identity
    )
    changed_issues = c.guard_generated_change_issues(
        repo, corpus_path=corpus, changed_files=[target, companion, canonical]
    )
    assert changed_issues == []
    # Full-repository legacy admission is unchanged: the old v1 receipt is not
    # reclassified as a current signed receipt by this single-target operation.
    all_issues = c.guard_generated_change_issues(
        repo, corpus_path=corpus, all_files=True
    )
    assert all_issues
    payload["applied_files"][0]["sha256"] = "0" * 64
    c._sign_applied_encoding_manifest(payload, TEST_APPLY_SIGNING_BROKER)
    manifest_path.write_bytes(patch.canonical_json(payload))
    _, _, _, issues = c._load_verified_applied_encoding_manifest_payload(
        repo,
        canonical,
        signing_broker=TEST_APPLY_SIGNING_BROKER,
        expected_encoder_identity=identity,
        expected_waiver_set_sha256=waiver,
        local_corpus_release=release,
    )
    assert any("postimage" in issue for issue in issues)


@pytest.mark.parametrize(
    "field",
    ["EXISTING_SIGNED_IMPORTS_JSON", "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON"],
)
@pytest.mark.parametrize(
    "raw_value", ['["ca/policies/other.yaml"]', "{}", "null", "false"]
)
def test_request_rejects_side_bundles_before_registry_or_network(
    tmp_path, monkeypatch, field, raw_value
):
    import json

    from tests.test_atomic_repair_mode import DISPATCH, envelope

    raw = json.dumps(
        {
            "schema": "axiom-encode/reviewed-repair-transaction/v1",
            "patch_id": "fixture",
            "transaction": envelope(),
        }
    )

    def forbidden(*_):
        pytest.fail("Excluded request reached registry acquisition")

    monkeypatch.setattr(patch, "load_registered_patch", forbidden)
    with pytest.raises(ValueError):
        patch.resolve_request(
            raw, {**DISPATCH, "OPEN_PR": "false", field: raw_value}, tmp_path
        )


@pytest.mark.parametrize("empty", ["", "[]", " [ ] "])
def test_request_accepts_only_empty_side_bundles_and_exact_contract(
    tmp_path, monkeypatch, empty
):
    import json
    from types import SimpleNamespace

    from tests.test_atomic_repair_mode import DISPATCH, envelope

    transaction = envelope()
    raw = json.dumps(
        {
            "schema": "axiom-encode/reviewed-repair-transaction/v1",
            "patch_id": "fixture",
            "transaction": transaction,
        }
    )
    value = manifest()
    value.update(
        citation=DISPATCH["CITATION"],
        rulespec_path=DISPATCH["REPLACE_RULESPEC_PATH"],
        contract_sha256=patch.sha256(patch.canonical_json(transaction)),
    )
    value["seed"]["run_id"] = DISPATCH["REPAIR_RUN_ID"]
    monkeypatch.setattr(
        patch,
        "load_registered_patch",
        lambda *_: SimpleNamespace(manifest=value, commit="f" * 40),
    )
    dispatch = {
        **DISPATCH,
        "OPEN_PR": "false",
        "COUNTRY": "ca",
        "RULESPEC_REF": value["rulespec_base_commit"],
        "CORPUS_REF": value["source"]["corpus_ref"],
        "RULES_ENGINE_REF": value["engine_ref"],
        "EXISTING_SIGNED_IMPORTS_JSON": empty,
        "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": empty,
    }
    assert patch.resolve_request(raw, dispatch, tmp_path) == {
        "patch_id": "fixture",
        "package_commit": "f" * 40,
    }
    transaction["transaction"]["primary_required_test_cases"][0]["input"]["cost"] = 101
    with pytest.raises(ValueError, match="transaction differs"):
        patch.resolve_request(
            json.dumps(
                {
                    "schema": "axiom-encode/reviewed-repair-transaction/v1",
                    "patch_id": "fixture",
                    "transaction": transaction,
                }
            ),
            dispatch,
            tmp_path,
        )


def test_unrelated_request_modes_are_not_reinterpreted(tmp_path):
    assert (
        patch.resolve_request('{"schema":"other-mode","unrelated":true}', {}, tmp_path)
        == {}
    )


@pytest.mark.parametrize("mismatch", [False, True])
def test_workflow_queues_exact_reviewed_command_without_model_fallback(
    tmp_path, mismatch
):
    import json
    import os
    from pathlib import Path

    import yaml

    workflow = yaml.safe_load(
        (
            Path(__file__).parents[1] / ".github/workflows/targeted-signed-reencode.yml"
        ).read_text()
    )
    scripts = [
        step.get("run", "")
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
    ]
    script = next(
        text
        for text in scripts
        if "queue_operation command" in text and "apply-reviewed-repair" in text
    )
    start = script.index('reviewed_repair_patch_id="$(jq')
    end = script.index('elif [ "$reviewed_candidate_promotion"', start)
    branch = script[start:end] + "fi\n"
    plan = tmp_path / "reviewed-repair-request.json"
    plan.write_text(json.dumps({"patch_id": "different" if mismatch else "fixture"}))
    log = tmp_path / "queued"
    env = {
        **os.environ,
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_WORKSPACE": str(tmp_path / "workspace"),
        "RULESPEC_CHECKOUT": str(tmp_path / "rulespec-ca"),
        "REPAIR_RUN_LANE": "target",
        "REPAIR_CANDIDATE_ROOT": str(tmp_path / "seed"),
        "REPAIR_CANDIDATE_PATH": "policies/example.yaml",
        "ATOMIC_SOURCE_JSON": '{"exact":"raw wrapper"}',
        "GITHUB_RUN_ID": "456",
        "CITATION": "ca/policy/example",
        "atomic_source_payload": '{"reviewed_repair_patch_id":"fixture"}',
        "QUEUE_LOG": str(log),
    }
    completed = subprocess.run(
        [
            "bash",
            "-eu",
            "-c",
            'queue_operation() { printf "%s\\0" "$@" >> "$QUEUE_LOG"; }\n' + branch,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if mismatch:
        assert completed.returncode != 0 and not log.exists()
        return
    assert completed.returncode == 0, completed.stderr
    args = log.read_bytes().decode().split("\0")[:-1]
    assert args[:3] == [
        "command",
        "/opt/axiom-verification/axiom-encode",
        "apply-reviewed-repair",
    ]
    command = args[3 : args.index("checkpoint")]
    pairs = dict(zip(command[::2], command[1::2], strict=True))
    assert pairs == {
        "--repo": env["RULESPEC_CHECKOUT"],
        "--patch-id": "fixture",
        "--patch-checkout": env["GITHUB_WORKSPACE"] + "/axiom-encode",
        "--axiom-rules-engine-path": env["GITHUB_WORKSPACE"] + "/axiom-rules-engine",
        "--corpus-path": env["GITHUB_WORKSPACE"] + "/axiom-corpus",
        "--seed-rulespec": env["REPAIR_CANDIDATE_ROOT"] + "/policies/example.yaml",
        "--seed-tests": env["REPAIR_CANDIDATE_ROOT"] + "/policies/example.test.yaml",
        "--seed-archive": str(
            tmp_path / "repair-download/targeted-reencode-failure.tar"
        ),
        "--source-bundle-json": env["ATOMIC_SOURCE_JSON"],
        "--run-id": "456",
    }
    assert args[-2:] == ["copy-guard", str(tmp_path / "target-guard-generated.json")]
