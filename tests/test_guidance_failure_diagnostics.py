"""Execute failure packaging after provenance passed but signed upload failed."""

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def record():
    return {
        "schema": "axiom-encode/verified-producer-guidance/v1",
        "manifest_path": "ca/.axiom/encoding-manifests/policies/example.json",
        "manifest_sha256": "a" * 64,
        "context_manifest_sha256": "b" * 64,
        "producer_commit": "c" * 40,
        "path": "review-guidance/ca/example.json",
        "blob_sha256": "d" * 64,
        "size_bytes": 100,
        "citation": "ca/policy/example",
        "source_sha256": "e" * 64,
        "content_sha256": "f" * 64,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "canonical_colon",
        "hidden_blob",
        "manifest_backslash",
        "legacy_success",
        "legacy_failure",
        "extra_top",
        "extra_record",
        "missing",
        "schema",
        "digest",
        "producer",
        "path",
        "manifest",
        "citation",
        "bool_size",
        "oversize",
        "failed_guidance",
        "issues_guidance",
    ],
)
def test_later_upload_failure_retains_exact_guard_or_refuses(tmp_path, mutation):
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    step = next(
        s
        for s in workflow["jobs"]["encode"]["steps"]
        if s.get("name") == "Package failed re-encode diagnostics"
    )
    command = (
        step["run"]
        .replace("/opt/axiom-verification/python/bin/python", sys.executable)
        .replace("/usr/bin/python3", sys.executable)
    )
    guard = {
        "repo": "/runner/rulespec-ca",
        "passed": True,
        "issues": [],
        "producer_review_guidance": record(),
    }
    g = guard["producer_review_guidance"]
    if mutation in {"legacy_success", "legacy_failure"}:
        del guard["producer_review_guidance"]
        guard["passed"] = mutation == "legacy_success"
    elif mutation == "canonical_colon":
        g["citation"] = "us/statute/54a:4-7"
        g["manifest_path"] = ".axiom/encoding-manifests/us/statutes/54a:4-7.json"
    elif mutation == "hidden_blob":
        g["path"] = "review-guidance/.hidden.json"
    elif mutation == "manifest_backslash":
        g["manifest_path"] = "ca/../bad\\path.json"
    elif mutation == "extra_top":
        guard["unknown"] = True
    elif mutation == "extra_record":
        g["unknown"] = True
    elif mutation == "missing":
        del g["source_sha256"]
    elif mutation == "schema":
        g["schema"] = "unknown"
    elif mutation == "digest":
        g["content_sha256"] = "F" * 64
    elif mutation == "producer":
        g["producer_commit"] = "main"
    elif mutation == "path":
        g["path"] = "review-guidance/../example.json"
    elif mutation == "manifest":
        g["manifest_path"] = "/tmp/example.json"
    elif mutation == "citation":
        g["citation"] = "ca/policies/example"
    elif mutation == "bool_size":
        g["size_bytes"] = True
    elif mutation == "oversize":
        g["size_bytes"] = 512 * 1024 + 1
    elif mutation == "failed_guidance":
        guard["passed"] = False
    elif mutation == "issues_guidance":
        guard["issues"] = ["failed"]
    raw = (json.dumps(guard) + "\n").encode()
    (tmp_path / "generated").mkdir()
    (tmp_path / "guard-generated.json").write_bytes(raw)
    result = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CITATION": "ca/policy/example",
            "CORPUS_REF": "corpus-ref",
            "COUNTRY": "ca",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_RUN_ID": "1234",
            "GITHUB_SHA": "c" * 40,
            "RULES_ENGINE_REF": "engine-ref",
            "RULESPEC_REF": "rulespec-ref",
            "RUNNER_TEMP": str(tmp_path),
            "VERIFY_GENERATED_PROVENANCE_CONCLUSION": "success",
            "VERIFY_GENERATED_PROVENANCE_OUTCOME": "success",
            "UPLOAD_SIGNED_REENCODE_ARTIFACT_CONCLUSION": "failure",
            "UPLOAD_SIGNED_REENCODE_ARTIFACT_OUTCOME": "failure",
        },
    )
    archive = tmp_path / "targeted-reencode-failure.tar"
    if mutation not in {None, "canonical_colon", "legacy_success", "legacy_failure"}:
        assert result.returncode != 0
        assert "invalid shape" in result.stderr
        assert not archive.exists()
        return
    assert result.returncode == 0, result.stderr
    with tarfile.open(archive) as bundle:
        assert bundle.extractfile("./guards/guard-generated.json").read() == raw
        metadata = json.loads(bundle.extractfile("./metadata.json").read())
        assert hashlib.sha256(raw).hexdigest() in json.dumps(metadata)
