from __future__ import annotations

import hashlib
import io
import json
import tarfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.extract_repair_candidate import (
    MAX_ARCHIVE_MEMBERS,
    MAX_CANDIDATE_BYTES,
    SINGLE_TARGET_MODE_FIELDS,
    _expected_module_path,
    extract_candidate,
)


def _archive(tmp_path: Path, *, candidate: bytes | None = None) -> tuple[Path, dict]:
    candidate = candidate or b"format: rulespec/v1\nrules: []\n"
    tests = b"[]\n"
    repair = json.dumps(
        {
            "schema_version": "axiom-encode/repair-manifest/v1",
            "citation": "us/statute/42/1437c\u20131",
            "runner": "openai-gpt-5.6-sol",
        }
    ).encode()
    payloads = {
        "target/openai-gpt-5.6-sol/statutes/42/1437c-1.yaml": candidate,
        "target/openai-gpt-5.6-sol/statutes/42/1437c-1.test.yaml": tests,
        "target/openai-gpt-5.6-sol/statutes/42/1437c-1.repair.json": repair,
    }
    metadata = {
        "schema": "axiom-encode/failed-reencode-diagnostics/v1",
        "citation": "us/statute/42/1437c\u20131",
        "country": "us",
        "encoder_commit": "a" * 40,
        "corpus_ref": "b" * 40,
        "rules_engine_ref": "c" * 40,
        "rulespec_ref": "d" * 40,
        "replace_rulespec_path": "us/statutes/42/1437c-1.yaml",
        "workflow_run_id": "1234",
        "workflow_run_attempt": 1,
        "failed_steps": ["encode_apply"],
        "generated_lanes": ["target"],
        **SINGLE_TARGET_MODE_FIELDS,
        "source_bundle_input": "[]",
        "files": [
            {
                "path": path,
                "size": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
            for path, body in sorted(payloads.items())
        ],
    }
    archive = tmp_path / "failure.tar"
    with tarfile.open(archive, "w") as bundle:
        members = {"metadata.json": json.dumps(metadata).encode()}
        members.update({f"generated/{path}": body for path, body in payloads.items()})
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            bundle.addfile(info, io.BytesIO(body))
    return archive, metadata


def _add_retained_candidate(
    source_archive: Path,
    destination: Path,
    metadata: dict,
    *,
    candidate: bytes,
    tests: bytes = b"[]\n",
) -> Path:
    module = "statutes/42/1437c-1.yaml"
    root = "target/final-rejected-candidate"
    issues = json.dumps(
        {
            "schema": "axiom-encode/failed-encode-candidate/v1",
            "citation": "us/statute/42/1437c\u20131",
            "path": module,
            "issues": ["best candidate still needs one repair"],
            "rulespec_sha256": hashlib.sha256(candidate).hexdigest(),
            "tests_sha256": hashlib.sha256(tests).hexdigest(),
            "encoder_version": "0.2.1713",
            "attempt_count": 4,
        }
    ).encode()
    payloads = {
        f"{root}/issues.json": issues,
        f"{root}/{module}": candidate,
        f"{root}/{module.removesuffix('.yaml')}.test.yaml": tests,
    }
    metadata["files"].extend(
        {
            "path": path,
            "size": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        for path, body in sorted(payloads.items())
    )
    metadata["files"].sort(key=lambda entry: entry["path"])
    with (
        tarfile.open(source_archive, "r") as source,
        tarfile.open(destination, "w") as target,
    ):
        for member in source.getmembers():
            extracted = source.extractfile(member)
            assert extracted is not None
            body = extracted.read()
            if member.name == "metadata.json":
                body = json.dumps(metadata).encode()
                member.size = len(body)
            target.addfile(member, io.BytesIO(body))
        for path, body in payloads.items():
            info = tarfile.TarInfo(f"generated/{path}")
            info.size = len(body)
            target.addfile(info, io.BytesIO(body))
    return destination


def _rewrite_metadata(
    source_archive: Path,
    destination: Path,
    metadata: dict,
) -> Path:
    with (
        tarfile.open(source_archive, "r") as source,
        tarfile.open(destination, "w") as target,
    ):
        for member in source.getmembers():
            extracted = source.extractfile(member)
            assert extracted is not None
            body = extracted.read()
            if member.name == "metadata.json":
                body = json.dumps(metadata).encode()
                member.size = len(body)
            target.addfile(member, io.BytesIO(body))
    return destination


def _rewrite_as_dependent_candidate(
    source_archive: Path,
    destination: Path,
    metadata: dict,
) -> Path:
    metadata.pop("source_bundle_input")
    metadata["atomic_source_input"] = "[]"
    metadata["citation"] = "us/guidance/primary/source"
    metadata["replace_rulespec_path"] = "us/guidance/primary/source.yaml"
    metadata["dependent_citation"] = "us/statute/42/1437c\u20131"
    metadata["generated_lanes"] = ["dependent", "target"]
    metadata["files"] = [
        {**entry, "path": entry["path"].replace("target/", "dependent/", 1)}
        for entry in metadata["files"]
    ]
    with (
        tarfile.open(source_archive, "r") as source,
        tarfile.open(destination, "w") as target,
    ):
        for member in source.getmembers():
            extracted = source.extractfile(member)
            assert extracted is not None
            body = extracted.read()
            if member.name == "metadata.json":
                body = json.dumps(metadata).encode()
            name = member.name.replace("generated/target/", "generated/dependent/", 1)
            info = tarfile.TarInfo(name)
            info.size = len(body)
            target.addfile(info, io.BytesIO(body))
    return destination


def _add_generated_payloads(
    source_archive: Path,
    destination: Path,
    metadata: dict,
    payloads: dict[str, bytes],
) -> Path:
    metadata["files"].extend(
        {
            "path": path,
            "size": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        for path, body in sorted(payloads.items())
    )
    metadata["files"].sort(key=lambda entry: entry["path"])
    with (
        tarfile.open(source_archive, "r") as source,
        tarfile.open(destination, "w") as target,
    ):
        for member in source.getmembers():
            extracted = source.extractfile(member)
            assert extracted is not None
            body = extracted.read()
            if member.name == "metadata.json":
                body = json.dumps(metadata).encode()
            info = tarfile.TarInfo(member.name)
            info.size = len(body)
            target.addfile(info, io.BytesIO(body))
        for path, body in payloads.items():
            info = tarfile.TarInfo(f"generated/{path}")
            info.size = len(body)
            target.addfile(info, io.BytesIO(body))
    return destination


def _args(tmp_path: Path, archive: Path, **overrides) -> Namespace:
    values = {
        "archive": archive,
        "destination": tmp_path / "extracted",
        "citation": "us/statute/42/1437c\u20131",
        "country": "us",
        "encoder_commit": "a" * 40,
        "corpus_ref": "b" * 40,
        "rules_engine_ref": "c" * 40,
        "rulespec_ref": "d" * 40,
        "allow_rulespec_base_advance": False,
        "atomic_source_json": "[]",
        "existing_signed_imports_json": "[]",
        "replace_rulespec_path": "us/statutes/42/1437c-1.yaml",
        "workflow_run_id": "1234",
    }
    values.update(overrides)
    return Namespace(**values)


def test_extracts_checksum_bound_final_candidate(tmp_path):
    archive, _ = _archive(tmp_path)

    result = extract_candidate(_args(tmp_path, archive))

    root = Path(result["root"])
    assert result["path"] == "statutes/42/1437c-1.yaml"
    assert result["source_rulespec_ref"] == "d" * 40
    assert (root / result["path"]).read_text() == "format: rulespec/v1\nrules: []\n"
    assert (root / "statutes/42/1437c-1.test.yaml").read_text() == "[]\n"


def test_extracts_candidate_with_exact_artifact_bound_signed_imports(tmp_path):
    archive, metadata = _archive(tmp_path)
    imports = '["us/statutes/7/2015/f.yaml"]'
    metadata["existing_signed_imports_input"] = imports
    replacement = _rewrite_metadata(
        archive,
        tmp_path / "signed-imports.tar",
        metadata,
    )

    result = extract_candidate(
        _args(
            tmp_path,
            replacement,
            existing_signed_imports_json=imports,
        )
    )

    assert result["runner"] == "openai-gpt-5.6-sol"


def test_rejects_candidate_when_signed_imports_change_between_runs(tmp_path):
    archive, metadata = _archive(tmp_path)
    metadata["existing_signed_imports_input"] = '["us/statutes/7/2015/f.yaml"]'
    replacement = _rewrite_metadata(
        archive,
        tmp_path / "changed-signed-imports.tar",
        metadata,
    )

    with pytest.raises(
        ValueError,
        match="compatible single-target run: existing_signed_imports_input",
    ):
        extract_candidate(
            _args(
                tmp_path,
                replacement,
                existing_signed_imports_json=('["us/statutes/7/2015/different.yaml"]'),
            )
        )


def test_prefers_integrity_bound_retained_best_candidate(tmp_path):
    archive, metadata = _archive(tmp_path, candidate=b"final regression\n")
    retained = b"strongest retained candidate\n"
    replacement = _add_retained_candidate(
        archive,
        tmp_path / "retained.tar",
        metadata,
        candidate=retained,
    )

    result = extract_candidate(_args(tmp_path, replacement))

    root = Path(result["root"])
    assert result["runner"] == "retained-best"
    assert (root / result["path"]).read_bytes() == retained


def test_extracts_dependent_candidate_for_atomic_repair(tmp_path):
    archive, metadata = _archive(tmp_path)
    replacement = _rewrite_as_dependent_candidate(
        archive,
        tmp_path / "dependent.tar",
        metadata,
    )

    result = extract_candidate(
        _args(
            tmp_path,
            replacement,
            repair_lane="dependent",
            transaction_citation="us/guidance/primary/source",
            transaction_rulespec_path="us/guidance/primary/source.yaml",
        )
    )

    assert result["runner"] == "openai-gpt-5.6-sol"
    assert result["path"] == "statutes/42/1437c-1.yaml"


def test_rejects_dependent_candidate_with_different_citation(tmp_path):
    archive, metadata = _archive(tmp_path)
    replacement = _rewrite_as_dependent_candidate(
        archive,
        tmp_path / "dependent.tar",
        metadata,
    )
    metadata["dependent_citation"] = "us/statute/42/different"
    mismatch = _rewrite_metadata(
        replacement,
        tmp_path / "dependent-mismatch.tar",
        metadata,
    )

    with pytest.raises(ValueError, match="single-target run: dependent_citation"):
        extract_candidate(
            _args(
                tmp_path,
                mismatch,
                repair_lane="dependent",
                transaction_citation="us/guidance/primary/source",
                transaction_rulespec_path="us/guidance/primary/source.yaml",
            )
        )


def test_rejects_tampered_retained_best_candidate(tmp_path):
    archive, metadata = _archive(tmp_path)
    replacement = _add_retained_candidate(
        archive,
        tmp_path / "retained.tar",
        metadata,
        candidate=b"strongest retained candidate\n",
    )
    candidate_entry = next(
        entry
        for entry in metadata["files"]
        if entry["path"].endswith("final-rejected-candidate/statutes/42/1437c-1.yaml")
    )
    candidate_entry["sha256"] = "0" * 64
    tampered = _rewrite_metadata(
        replacement,
        tmp_path / "tampered-retained.tar",
        metadata,
    )

    with pytest.raises(ValueError, match="digest mismatch"):
        extract_candidate(_args(tmp_path, tampered))


def test_state_rulespec_path_maps_to_lane_relative_candidate_path():
    assert (
        _expected_module_path("us", "us-la/statutes/47/297/4.yaml")
        == "statutes/47/297/4.yaml"
    )


@pytest.mark.parametrize(
    "replace_rulespec_path",
    (
        "ca-la/statutes/47/297/4.yaml",
        "us-/statutes/47/297/4.yaml",
        "us-_/statutes/47/297/4.yaml",
        "us-LA/statutes/47/297/4.yaml",
        "us-la_foo/statutes/47/297/4.yaml",
        "us-la//statutes/47/297/4.yaml",
        "us-la/./statutes/47/297/4.yaml",
        "us-la/other/47/297/4.yaml",
        "us-la/statutes/47/297/4.test.yaml",
        "us-la/statutes/../297/4.yaml",
        "/us-la/statutes/47/297/4.yaml",
    ),
)
def test_rejects_noncanonical_state_rulespec_path(replace_rulespec_path):
    with pytest.raises(ValueError, match="not canonical"):
        _expected_module_path("us", replace_rulespec_path)


@pytest.mark.parametrize(
    "atomic_source_input",
    [
        "[]",
        '{"canonical_refresh_bundle":[]}',
        json.dumps(
            {
                "schema": "axiom-encode/atomic-source-transaction/v2",
                "source_bundle": [],
                "canonical_refresh_bundle": [],
                "primary_required_test_cases": [],
            }
        ),
    ],
)
def test_extracts_new_atomic_source_metadata(tmp_path, atomic_source_input):
    archive, metadata = _archive(tmp_path)
    del metadata["source_bundle_input"]
    metadata["atomic_source_input"] = atomic_source_input
    replacement = _rewrite_metadata(
        archive,
        tmp_path / "new-atomic-source.tar",
        metadata,
    )

    result = extract_candidate(_args(tmp_path, replacement))

    assert result["source_rulespec_ref"] == "d" * 40


def _rewrite_as_source_preflight(
    source_archive: Path,
    destination: Path,
    metadata: dict,
    atomic_source_input: str,
) -> Path:
    metadata.pop("source_bundle_input")
    metadata["atomic_source_input"] = atomic_source_input
    metadata["generated_lanes"] = ["target-preflight"]
    metadata["files"] = [
        {**entry, "path": entry["path"].replace("target/", "target-preflight/", 1)}
        for entry in metadata["files"]
    ]
    with (
        tarfile.open(source_archive, "r") as source,
        tarfile.open(destination, "w") as target,
    ):
        for member in source.getmembers():
            extracted = source.extractfile(member)
            assert extracted is not None
            body = extracted.read()
            if member.name == "metadata.json":
                body = json.dumps(metadata).encode()
            name = member.name.replace(
                "generated/target/", "generated/target-preflight/", 1
            )
            info = tarfile.TarInfo(name)
            info.size = len(body)
            target.addfile(info, io.BytesIO(body))
    return destination


def test_extracts_exactly_bound_source_preflight_candidate(tmp_path):
    atomic_source_input = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v2",
            "source_bundle": ["us/statute/7/2015/f"],
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [],
        }
    )
    archive, metadata = _archive(tmp_path)
    replacement = _rewrite_as_source_preflight(
        archive, tmp_path / "source-preflight.tar", metadata, atomic_source_input
    )

    result = extract_candidate(
        _args(tmp_path, replacement, atomic_source_json=atomic_source_input)
    )

    assert result["runner"] == "openai-gpt-5.6-sol"


def test_extracts_exactly_bound_final_composed_target_candidate(tmp_path):
    atomic_source_input = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v2",
            "source_bundle": [
                "us/statute/7/2015/f",
                "us/guidance/usda/fns/snap-obbb-alien-eligibility-implementation-memo",
            ],
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [],
        }
    )
    archive, metadata = _archive(tmp_path)
    del metadata["source_bundle_input"]
    metadata["atomic_source_input"] = atomic_source_input
    metadata["generated_lanes"] = [
        "source-01",
        "source-02",
        "target",
        "target-preflight",
    ]
    replacement = _rewrite_metadata(
        archive,
        tmp_path / "final-composed-target.tar",
        metadata,
    )

    result = extract_candidate(
        _args(tmp_path, replacement, atomic_source_json=atomic_source_input)
    )

    assert result["runner"] == "openai-gpt-5.6-sol"


def test_extracts_replayed_final_composition_without_redundant_preflight(tmp_path):
    atomic_source_input = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v2",
            "source_bundle": [
                "us/statute/7/2015/f",
                "us/guidance/usda/fns/snap-obbb-alien-eligibility-implementation-memo",
            ],
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [],
        }
    )
    archive, metadata = _archive(tmp_path)
    del metadata["source_bundle_input"]
    metadata["atomic_source_input"] = atomic_source_input
    metadata["generated_lanes"] = ["source-01", "source-02", "target"]
    replacement = _rewrite_metadata(
        archive,
        tmp_path / "replayed-final-composed-target.tar",
        metadata,
    )

    result = extract_candidate(
        _args(tmp_path, replacement, atomic_source_json=atomic_source_input)
    )

    assert result["runner"] == "openai-gpt-5.6-sol"


def test_extracts_digest_bound_source_candidates_from_final_composition(tmp_path):
    source_citations = [
        "us/statute/7/2015/f",
        "us/guidance/usda/fns/snap-obbb-alien-eligibility-implementation-memo",
    ]
    source_paths = [
        "us/statutes/7/2015/f.yaml",
        "us/policies/usda/fns/snap-obbb-alien-eligibility-implementation-memo.yaml",
    ]
    atomic_source_input = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v2",
            "source_bundle": source_citations,
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [],
        }
    )
    archive, metadata = _archive(tmp_path)
    del metadata["source_bundle_input"]
    metadata["atomic_source_input"] = atomic_source_input
    metadata["generated_lanes"] = [
        "source-01",
        "source-02",
        "target",
        "target-preflight",
    ]
    source_one = b"format: rulespec/v1\n# final statute source\nrules: []\n"
    source_two = b"format: rulespec/v1\n# final guidance source\nrules: []\n"
    payloads = {
        "source-01/openai-gpt-5.6-terra/statutes/7/2015/f.yaml": b"older\n",
        "source-01/openai-gpt-5.6-terra/statutes/7/2015/f.test.yaml": b"[]\n",
        "source-01/openai-gpt-5.6-sol/statutes/7/2015/f.yaml": source_one,
        "source-01/openai-gpt-5.6-sol/statutes/7/2015/f.test.yaml": b"[]\n",
        "source-02/openai-gpt-5.6-sol/policies/usda/fns/snap-obbb-alien-eligibility-implementation-memo.yaml": source_two,
        "source-02/openai-gpt-5.6-sol/policies/usda/fns/snap-obbb-alien-eligibility-implementation-memo.test.yaml": b"[]\n",
    }
    replacement = _add_generated_payloads(
        archive,
        tmp_path / "composed-with-sources.tar",
        metadata,
        payloads,
    )

    result = extract_candidate(
        _args(
            tmp_path,
            replacement,
            atomic_source_json=atomic_source_input,
            source_rulespec_paths_json=json.dumps(source_paths),
        )
    )

    assert [candidate["citation"] for candidate in result["source_candidates"]] == (
        source_citations
    )
    first, second = result["source_candidates"]
    assert first["runner"] == "openai-gpt-5.6-sol"
    assert (Path(first["root"]) / first["path"]).read_bytes() == source_one
    assert (Path(second["root"]) / second["path"]).read_bytes() == source_two


def test_extracts_partial_source_repair_after_successful_target_preflight(tmp_path):
    source_citation = "us/guidance/example/source"
    source_path = "us/guidance/example/source.yaml"
    atomic_source_input = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v2",
            "source_bundle": [source_citation],
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [],
        }
    )
    _, metadata = _archive(tmp_path)
    metadata.pop("source_bundle_input")
    metadata["atomic_source_input"] = atomic_source_input
    metadata["generated_lanes"] = ["source-01", "target-preflight"]
    target_candidate = b"format: rulespec/v1\n# successful preflight\nrules: []\n"
    source_candidate = b"format: rulespec/v1\n# source needs repair\nrules: []\n"
    payloads = {
        "target-preflight/openai-gpt-5.6-terra/statutes/42/1437c-1.yaml": (
            target_candidate
        ),
        "target-preflight/openai-gpt-5.6-terra/statutes/42/1437c-1.test.yaml": (
            b"[]\n"
        ),
        f"source-01/openai-gpt-5.6-sol/{source_path.removeprefix('us/')}": (
            source_candidate
        ),
        f"source-01/openai-gpt-5.6-sol/"
        f"{source_path.removeprefix('us/').removesuffix('.yaml')}.test.yaml": (b"[]\n"),
    }
    metadata["files"] = [
        {
            "path": path,
            "size": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        for path, body in sorted(payloads.items())
    ]
    archive = tmp_path / "partial-source-repair.tar"
    with tarfile.open(archive, "w") as bundle:
        members = {"metadata.json": json.dumps(metadata).encode()}
        members.update({f"generated/{path}": body for path, body in payloads.items()})
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            bundle.addfile(info, io.BytesIO(body))

    result = extract_candidate(
        _args(
            tmp_path,
            archive,
            atomic_source_json=atomic_source_input,
            destination=tmp_path / "partial-extracted",
            source_rulespec_paths_json=json.dumps([source_path]),
        )
    )

    assert result["lane"] == "target-preflight"
    assert result["runner"] == "openai-gpt-5.6-terra"
    assert (Path(result["root"]) / result["path"]).read_bytes() == target_candidate
    assert len(result["source_candidates"]) == 1
    source = result["source_candidates"][0]
    assert source["citation"] == source_citation
    assert (Path(source["root"]) / source["path"]).read_bytes() == source_candidate


def test_extracts_source_only_continuation_after_replayed_preflight(tmp_path):
    source_citation = "us/guidance/example/source"
    source_path = "us/guidance/example/source.yaml"
    atomic_source_input = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v2",
            "source_bundle": [source_citation],
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [],
        }
    )
    _, metadata = _archive(tmp_path)
    metadata.pop("source_bundle_input")
    metadata["atomic_source_input"] = atomic_source_input
    metadata["generated_lanes"] = ["source-01"]
    source_candidate = b"format: rulespec/v1\n# improved source repair\nrules: []\n"
    module = source_path.removeprefix("us/")
    payloads = {
        f"source-01/openai-gpt-5.6-sol/{module}": source_candidate,
        f"source-01/openai-gpt-5.6-sol/"
        f"{module.removesuffix('.yaml')}.test.yaml": b"[]\n",
    }
    metadata["files"] = [
        {
            "path": path,
            "size": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        for path, body in sorted(payloads.items())
    ]
    archive = tmp_path / "source-only-repair.tar"
    with tarfile.open(archive, "w") as bundle:
        members = {"metadata.json": json.dumps(metadata).encode()}
        members.update({f"generated/{path}": body for path, body in payloads.items()})
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            bundle.addfile(info, io.BytesIO(body))

    identity = extract_candidate(
        _args(
            tmp_path,
            archive,
            atomic_source_json=atomic_source_input,
            destination=tmp_path / "source-only-identity",
        )
    )
    result = extract_candidate(
        _args(
            tmp_path,
            archive,
            atomic_source_json=atomic_source_input,
            destination=tmp_path / "source-only-extracted",
            source_rulespec_paths_json=json.dumps([source_path]),
        )
    )

    assert identity["lane"] == "source-only"
    assert identity["root"] == ""
    assert identity["source_candidates"] == []
    assert result["lane"] == "source-only"
    assert len(result["source_candidates"]) == 1
    source = result["source_candidates"][0]
    assert (Path(source["root"]) / source["path"]).read_bytes() == source_candidate


def _partial_source_retained_archive(
    tmp_path, *, live_source, mutation=None, missing_companion=False
):
    from axiom_encode.cli import (
        ValidationRetryCandidate,
        _emit_final_rejected_candidate,
    )

    citation = "us/guidance/example/source"
    module = "policies/example/source.yaml"
    source_path = f"us/{module}"
    source_lane = tmp_path / "generated/source-01"
    source_lane.mkdir(parents=True)
    output_file = source_lane / "codex-gpt-5.6-sol" / module
    candidate = b"format: rulespec/v1\n# selected rejected source\nrules: []\n"
    tests = b"[]\n" if missing_companion else b"# selected source companion\n[]\n"
    # The selected attempt's live files disappeared in a later failed call.
    # Emit the real durable contract from its captured pair, without a model.
    retained = _emit_final_rejected_candidate(
        SimpleNamespace(runner="codex-gpt-5.6-sol", output_file=str(output_file)),
        output_root=source_lane,
        destination=source_lane / "final-rejected-candidate",
        citation=citation,
        validation_issues=["selected source still fails deterministic validation"],
        attempt_count=4,
        candidate_override=ValidationRetryCandidate(
            rulespec=candidate.decode(),
            tests=None if missing_companion else tests.decode(),
        ),
    )
    assert not output_file.exists()
    payloads = {
        f"source-01/final-rejected-candidate/{path.relative_to(retained)}": (
            path.read_bytes()
        )
        for path in retained.rglob("*")
        if path.is_file()
    }
    issues_path = "source-01/final-rejected-candidate/issues.json"
    issues = json.loads(payloads[issues_path])
    assert issues["attempt_count"] == 4
    if isinstance(mutation, str):
        if mutation == "missing_metadata":
            del payloads[issues_path]
        elif mutation == "missing_candidate":
            del payloads[f"source-01/final-rejected-candidate/{module}"]
        elif mutation == "missing_tests":
            del payloads[f"source-01/final-rejected-candidate/{module[:-5]}.test.yaml"]
        else:
            payloads["source-01/final-rejected-candidate/unbound.txt"] = b"extra\n"
    elif mutation is not None:
        key, value = mutation
        issues[key] = value
        payloads[issues_path] = json.dumps(issues).encode()
    if live_source:
        payloads[f"source-01/openai-gpt-5.6-sol/{module}"] = b"stale live source\n"
        payloads[f"source-01/openai-gpt-5.6-sol/{module[:-5]}.test.yaml"] = (
            b"# stale live companion\n[]\n"
        )
    target = "target-preflight/openai-gpt-5.6-terra/statutes/42/1437c-1"
    payloads[f"{target}.yaml"] = b"format: rulespec/v1\n# preflight\nrules: []\n"
    payloads[f"{target}.test.yaml"] = b"[]\n"
    _, metadata = _archive(tmp_path)
    metadata.pop("source_bundle_input")
    atomic_source_input = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v2",
            "source_bundle": [citation],
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [],
        }
    )
    metadata["atomic_source_input"] = atomic_source_input
    metadata["generated_lanes"] = ["source-01", "target-preflight"]
    metadata["files"] = [
        {"path": path, "size": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        for path, body in sorted(payloads.items())
    ]
    archive = tmp_path / "partial-source-retained.tar"
    with tarfile.open(archive, "w") as bundle:
        members = {"metadata.json": json.dumps(metadata).encode()}
        members.update({f"generated/{path}": body for path, body in payloads.items()})
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            bundle.addfile(info, io.BytesIO(body))
    args = _args(
        tmp_path,
        archive,
        atomic_source_json=atomic_source_input,
        source_rulespec_paths_json=json.dumps([source_path]),
    )
    return args, candidate, tests


@pytest.mark.parametrize("live_source", [False, True])
@pytest.mark.parametrize(
    ("missing_companion", "encoder_version"), [(False, None), (True, "0.2.1973")]
)
def test_source_repair_prefers_verified_retained_export(
    tmp_path, live_source, missing_companion, encoder_version
):
    args, candidate, tests = _partial_source_retained_archive(
        tmp_path,
        live_source=live_source,
        missing_companion=missing_companion,
        mutation=("encoder_version", encoder_version) if encoder_version else None,
    )

    result = extract_candidate(args)

    assert result["lane"] == "target-preflight"
    assert result["runner"] == "openai-gpt-5.6-terra"
    (source,) = result["source_candidates"]
    assert source["runner"] == "retained-best"
    assert source["citation"] == "us/guidance/example/source"
    assert source["lane"] == "source-01"
    path = Path(source["root"]) / source["path"]
    assert path.read_bytes() == candidate
    assert path.with_suffix(".test.yaml").read_bytes() == tests
    assert source["rulespec_sha256"] == hashlib.sha256(candidate).hexdigest()
    assert source["tests_sha256"] == hashlib.sha256(tests).hexdigest()


@pytest.mark.parametrize("live_source", [False, True])
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_metadata",
        "missing_candidate",
        "missing_tests",
        "extra_file",
        ("citation", "us/guidance/different/source"),
        ("path", "guidance/different/source.yaml"),
        ("attempt_count", 0),
        ("attempt_count", True),
        ("attempt_count", -1),
        ("attempt_count", "4"),
        ("encoder_version", ""),
        ("issues", []),
        ("schema", "axiom-encode/successful-candidate/v1"),
        ("rulespec_sha256", "0" * 64),
        ("tests_sha256", "0" * 64),
        ("accepted", True),
    ],
)
def test_source_repair_rejects_invalid_retained_export_even_with_live_pair(
    tmp_path, live_source, mutation
):
    args, _, _ = _partial_source_retained_archive(
        tmp_path, live_source=live_source, mutation=mutation
    )

    with pytest.raises(ValueError, match="retained repair candidate"):
        extract_candidate(args)

    assert not (Path(args.destination) / "source-candidates").exists()


@pytest.mark.parametrize("live_source", [False, True])
@pytest.mark.parametrize("suffix", ["source.yaml", "source.test.yaml"])
def test_source_repair_rejects_outer_digest_mismatch(tmp_path, live_source, suffix):
    args, _, _ = _partial_source_retained_archive(tmp_path, live_source=live_source)
    with tarfile.open(args.archive, "r") as bundle:
        metadata = json.load(bundle.extractfile("metadata.json"))
    entry = next(
        entry
        for entry in metadata["files"]
        if entry["path"]
        == f"source-01/final-rejected-candidate/policies/example/{suffix}"
    )
    entry["sha256"] = "0" * 64
    args.archive = _rewrite_metadata(
        args.archive, tmp_path / "outer-digest-mismatch.tar", metadata
    )

    with pytest.raises(ValueError, match="repair candidate digest mismatch"):
        extract_candidate(args)

    assert not (Path(args.destination) / "source-candidates").exists()


def _source_only_retained_archive(tmp_path, *, live_source, mutation=None):
    args, candidate, tests = _partial_source_retained_archive(
        tmp_path, live_source=live_source, mutation=mutation
    )
    with tarfile.open(args.archive, "r") as bundle:
        metadata = json.load(bundle.extractfile("metadata.json"))
        payloads = {
            member.name: bundle.extractfile(member).read()
            for member in bundle.getmembers()
            if member.isfile() and member.name.startswith("generated/source-01/")
        }
    metadata["generated_lanes"] = ["source-01"]
    metadata["files"] = [
        entry for entry in metadata["files"] if entry["path"].startswith("source-01/")
    ]
    payloads["metadata.json"] = json.dumps(metadata).encode()
    args.archive = tmp_path / "source-only-retained.tar"
    with tarfile.open(args.archive, "w") as bundle:
        for name, body in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            bundle.addfile(info, io.BytesIO(body))
    return args, candidate, tests


@pytest.mark.parametrize("live_source", [False, True])
def test_source_only_replay_prefers_verified_retained_export(tmp_path, live_source):
    args, candidate, tests = _source_only_retained_archive(
        tmp_path, live_source=live_source
    )

    result = extract_candidate(args)

    assert result["lane"] == "source-only"
    assert all(
        result[key] == ""
        for key in ("root", "path", "runner", "rulespec_sha256", "tests_sha256")
    )
    (source,) = result["source_candidates"]
    assert source["runner"] == "retained-best"
    assert source["citation"] == "us/guidance/example/source"
    assert source["lane"] == "source-01"
    path = Path(source["root"]) / source["path"]
    assert path.read_bytes() == candidate
    assert path.with_suffix(".test.yaml").read_bytes() == tests
    assert source["rulespec_sha256"] == hashlib.sha256(candidate).hexdigest()
    assert source["tests_sha256"] == hashlib.sha256(tests).hexdigest()


@pytest.mark.parametrize("live_source", [False, True])
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_metadata",
        "missing_candidate",
        "missing_tests",
        ("issues", []),
        ("schema", "axiom-encode/successful-candidate/v1"),
        ("citation", "us/guidance/different/source"),
        ("rulespec_sha256", "0" * 64),
        ("tests_sha256", "0" * 64),
    ],
)
def test_source_only_replay_rejects_invalid_retained_export(
    tmp_path, live_source, mutation
):
    args, _, _ = _source_only_retained_archive(
        tmp_path, live_source=live_source, mutation=mutation
    )

    with pytest.raises(ValueError, match="retained repair candidate"):
        extract_candidate(args)

    assert not (Path(args.destination) / "source-candidates").exists()


@pytest.mark.parametrize("live_source", [False, True])
@pytest.mark.parametrize(
    "mutation", [None, "missing_metadata", ("citation", "us/guidance/different/source")]
)
def test_source_only_identity_does_not_extract_or_accept_candidates(
    tmp_path, live_source, mutation
):
    args, _, _ = _source_only_retained_archive(
        tmp_path, live_source=live_source, mutation=mutation
    )
    identity_args = Namespace(
        **{
            **vars(args),
            "source_rulespec_paths_json": None,
            "destination": tmp_path / "source-only-identity",
        }
    )

    identity = extract_candidate(identity_args)

    assert identity == {
        "root": "",
        "lane": "source-only",
        "path": "",
        "rulespec_sha256": "",
        "tests_sha256": "",
        "runner": "",
        "source_rulespec_ref": args.rulespec_ref,
        "source_candidates": [],
    }
    assert list(Path(identity_args.destination).iterdir()) == []
    if mutation is not None:
        with pytest.raises(ValueError, match="retained repair candidate"):
            extract_candidate(args)


def test_rejects_source_candidates_without_final_composed_target(tmp_path):
    atomic_source_input = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v2",
            "source_bundle": ["us/statute/7/2015/f"],
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [],
        }
    )
    archive, metadata = _archive(tmp_path)
    replacement = _rewrite_as_source_preflight(
        archive, tmp_path / "source-preflight.tar", metadata, atomic_source_input
    )

    with pytest.raises(ValueError, match="final composed target artifact"):
        extract_candidate(
            _args(
                tmp_path,
                replacement,
                atomic_source_json=atomic_source_input,
                source_rulespec_paths_json=json.dumps(["us/statutes/7/2015/f.yaml"]),
            )
        )


def test_rejects_final_composed_target_with_incomplete_source_lanes(tmp_path):
    atomic_source_input = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v2",
            "source_bundle": [
                "us/statute/7/2015/f",
                "us/guidance/usda/fns/snap-obbb-alien-eligibility-implementation-memo",
            ],
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [],
        }
    )
    archive, metadata = _archive(tmp_path)
    del metadata["source_bundle_input"]
    metadata["atomic_source_input"] = atomic_source_input
    metadata["generated_lanes"] = [
        "source-01",
        "target",
        "target-preflight",
    ]
    replacement = _rewrite_metadata(
        archive,
        tmp_path / "incomplete-final-composed-target.tar",
        metadata,
    )

    with pytest.raises(ValueError, match="generated lanes do not bind"):
        extract_candidate(
            _args(tmp_path, replacement, atomic_source_json=atomic_source_input)
        )


def test_rejects_source_preflight_candidate_for_different_bundle(tmp_path):
    atomic_source_input = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v2",
            "source_bundle": ["us/statute/7/2015/f"],
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [],
        }
    )
    archive, metadata = _archive(tmp_path)
    replacement = _rewrite_as_source_preflight(
        archive, tmp_path / "source-preflight.tar", metadata, atomic_source_input
    )

    with pytest.raises(ValueError, match="metadata mismatch: atomic_source_input"):
        extract_candidate(
            _args(
                tmp_path,
                replacement,
                atomic_source_json=json.dumps(
                    {
                        "schema": "axiom-encode/atomic-source-transaction/v2",
                        "source_bundle": ["us/guidance/different"],
                        "canonical_refresh_bundle": [],
                        "primary_required_test_cases": [],
                    }
                ),
            )
        )


def test_rejects_source_preflight_lane_without_expected_source(tmp_path):
    archive, metadata = _archive(tmp_path)
    replacement = _rewrite_as_source_preflight(
        archive,
        tmp_path / "source-preflight.tar",
        metadata,
        json.dumps(
            {
                "schema": "axiom-encode/atomic-source-transaction/v2",
                "source_bundle": [],
                "canonical_refresh_bundle": [],
                "primary_required_test_cases": [],
            }
        ),
    )

    with pytest.raises(ValueError, match="only the target lane"):
        extract_candidate(_args(tmp_path, replacement))


def test_rejects_metadata_identity_mismatch(tmp_path):
    archive, _ = _archive(tmp_path)

    with pytest.raises(ValueError, match="metadata mismatch: rulespec_ref"):
        extract_candidate(_args(tmp_path, archive, rulespec_ref="e" * 40))


def test_allows_explicit_rulespec_base_advance_for_later_workflow_proof(tmp_path):
    archive, _ = _archive(tmp_path)

    result = extract_candidate(
        _args(
            tmp_path,
            archive,
            rulespec_ref="e" * 40,
            allow_rulespec_base_advance=True,
        )
    )

    assert result["source_rulespec_ref"] == "d" * 40


def test_rejects_malformed_source_rulespec_ref_during_base_advance(tmp_path):
    archive, metadata = _archive(tmp_path)
    metadata["rulespec_ref"] = "not-a-commit"
    replacement = _rewrite_metadata(
        archive,
        tmp_path / "malformed-rulespec-ref.tar",
        metadata,
    )

    with pytest.raises(ValueError, match="metadata mismatch: rulespec_ref"):
        extract_candidate(
            _args(
                tmp_path,
                replacement,
                rulespec_ref="e" * 40,
                allow_rulespec_base_advance=True,
            )
        )


def test_rejects_candidate_digest_mismatch(tmp_path):
    archive, metadata = _archive(tmp_path)
    candidate_entry = next(
        item for item in metadata["files"] if item["path"].endswith("1437c-1.yaml")
    )
    candidate_entry["sha256"] = "0" * 64
    replacement = _rewrite_metadata(
        archive,
        tmp_path / "tampered.tar",
        metadata,
    )

    with pytest.raises(ValueError, match="digest mismatch"):
        extract_candidate(_args(tmp_path, replacement))


def test_rejects_duplicate_archive_member(tmp_path):
    archive, _ = _archive(tmp_path)
    replacement = tmp_path / "duplicate.tar"
    with tarfile.open(archive, "r") as source, tarfile.open(replacement, "w") as target:
        for member in source.getmembers():
            body = source.extractfile(member).read()
            target.addfile(member, io.BytesIO(body))
            if member.name == "metadata.json":
                target.addfile(member, io.BytesIO(body))

    with pytest.raises(ValueError, match="duplicate member path"):
        extract_candidate(_args(tmp_path, replacement))


def test_rejects_archive_member_flood(tmp_path):
    archive = tmp_path / "flood.tar"
    with tarfile.open(archive, "w") as bundle:
        for index in range(MAX_ARCHIVE_MEMBERS + 1):
            info = tarfile.TarInfo(f"entry-{index}.txt")
            bundle.addfile(info, io.BytesIO())

    with pytest.raises(ValueError, match="member limit"):
        extract_candidate(_args(tmp_path, archive))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dependent_citation", "us/statute/42/other"),
        ("existing_signed_imports_input", '["us/statutes/import.yaml"]'),
        ("legacy_exact_dependent_rulespec_path", "us/statutes/dependent.yaml"),
        ("legacy_retained_successor_rulespec_paths_input", '["old.yaml"]'),
        ("queue_dispatcher_run_id", "999"),
        ("queue_id", "queue"),
        ("queue_item_generation_sha256", "a" * 64),
        ("queue_item_id", "item"),
        ("queue_manifest_sha256", "b" * 64),
        ("replace_legacy_rulespec_path", "us/statutes/old.yaml"),
        ("second_dependent_citation", "us/statute/42/second"),
        (
            "second_legacy_exact_dependent_rulespec_path",
            "us/statutes/second.yaml",
        ),
    ],
)
def test_rejects_incompatible_prior_run_mode(tmp_path, field, value):
    archive, metadata = _archive(tmp_path)
    metadata[field] = value
    replacement = _rewrite_metadata(
        archive,
        tmp_path / "incompatible.tar",
        metadata,
    )

    with pytest.raises(ValueError, match=f"single-target run: {field}"):
        extract_candidate(_args(tmp_path, replacement))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "source_bundle_input",
            '["us/statute/42/source"]',
            "single-target run: source_bundle_input",
        ),
        (
            "canonical_refresh_bundle_input",
            '[{"citation":"x"}]',
            "single-target run: canonical_refresh_bundle_input",
        ),
        (
            "atomic_source_input",
            '["us/statute/42/source"]',
            "metadata mismatch: atomic_source_input",
        ),
        (
            "atomic_source_input",
            '{"canonical_refresh_bundle":[{"citation":"x"}]}',
            "metadata mismatch: atomic_source_input",
        ),
        (
            "atomic_source_input",
            json.dumps(
                {
                    "schema": "axiom-encode/atomic-source-transaction/v2",
                    "source_bundle": [],
                    "canonical_refresh_bundle": [],
                    "primary_required_test_cases": [
                        {
                            "name": "must not replay a review contract",
                            "period": {
                                "period_kind": "tax_year",
                                "start": "2025-01-01",
                                "end": "2025-12-31",
                            },
                            "input": {"us:test#input": True},
                            "required_output": {"us:test#output": 1},
                        }
                    ],
                }
            ),
            "metadata mismatch: atomic_source_input",
        ),
    ],
)
def test_rejects_unbound_nonempty_atomic_source_metadata(
    tmp_path, field, value, message
):
    archive, metadata = _archive(tmp_path)
    if field == "atomic_source_input":
        del metadata["source_bundle_input"]
    metadata[field] = value
    replacement = _rewrite_metadata(
        archive,
        tmp_path / "nonempty-atomic-source.tar",
        metadata,
    )

    with pytest.raises(ValueError, match=message):
        extract_candidate(_args(tmp_path, replacement))


def test_rejects_expected_canonical_refresh_bundle(tmp_path):
    archive, _ = _archive(tmp_path)

    with pytest.raises(ValueError, match="does not support canonical refresh"):
        extract_candidate(
            _args(
                tmp_path,
                archive,
                atomic_source_json=json.dumps(
                    {
                        "schema": "axiom-encode/atomic-source-transaction/v2",
                        "source_bundle": [],
                        "canonical_refresh_bundle": [{"citation": "x"}],
                        "primary_required_test_cases": [],
                    }
                ),
            )
        )


def test_rejects_expected_source_bundle_mixed_with_required_tests(tmp_path):
    archive, _ = _archive(tmp_path)

    with pytest.raises(ValueError, match="expected atomic source input is invalid"):
        extract_candidate(
            _args(
                tmp_path,
                archive,
                atomic_source_json=json.dumps(
                    {
                        "schema": "axiom-encode/atomic-source-transaction/v2",
                        "source_bundle": ["us/statute/7/2015/f"],
                        "canonical_refresh_bundle": [],
                        "primary_required_test_cases": [
                            {
                                "name": "control",
                                "period": {
                                    "period_kind": "tax_year",
                                    "start": "2025-01-01",
                                    "end": "2025-12-31",
                                },
                                "input": {"us:test#input": True},
                                "required_output": {"us:test#output": 1},
                            }
                        ],
                    }
                ),
            )
        )


def test_rejects_missing_legacy_and_atomic_source_metadata(tmp_path):
    archive, metadata = _archive(tmp_path)
    del metadata["source_bundle_input"]
    replacement = _rewrite_metadata(
        archive,
        tmp_path / "missing-source-mode.tar",
        metadata,
    )

    with pytest.raises(ValueError, match="single-target run: source_bundle_input"):
        extract_candidate(_args(tmp_path, replacement))


@pytest.mark.parametrize("field", sorted(SINGLE_TARGET_MODE_FIELDS))
def test_rejects_missing_prior_run_mode_field(tmp_path, field):
    archive, metadata = _archive(tmp_path)
    del metadata[field]
    replacement = _rewrite_metadata(
        archive,
        tmp_path / "missing-mode-field.tar",
        metadata,
    )

    with pytest.raises(ValueError, match=f"single-target run: {field}"):
        extract_candidate(_args(tmp_path, replacement))


def test_candidate_size_bound_is_shared_at_exact_limit(tmp_path):
    archive, _ = _archive(tmp_path, candidate=b"x" * MAX_CANDIDATE_BYTES)

    result = extract_candidate(_args(tmp_path, archive))

    assert Path(result["root"], result["path"]).stat().st_size == MAX_CANDIDATE_BYTES


def test_candidate_size_bound_rejects_limit_plus_one(tmp_path):
    archive, _ = _archive(tmp_path, candidate=b"x" * (MAX_CANDIDATE_BYTES + 1))

    with pytest.raises(ValueError, match="exceeds its size limit"):
        extract_candidate(_args(tmp_path, archive))
