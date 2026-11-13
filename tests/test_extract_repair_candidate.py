from __future__ import annotations

import hashlib
import io
import json
import tarfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode.prepare_signed_backfill import immutable_atomic_source_contract
from scripts.extract_repair_candidate import (
    MAX_ARCHIVE_MEMBERS,
    MAX_CANDIDATE_BYTES,
    SINGLE_TARGET_MODE_FIELDS,
    _expected_module_path,
    extract_candidate,
)


def _archive(
    tmp_path: Path,
    *,
    candidate: bytes | None = None,
    citation: str = "us/statute/42/1437c\u20131",
    module: str = "statutes/42/1437c-1.yaml",
    replace_rulespec_path: str | None = "us/statutes/42/1437c-1.yaml",
) -> tuple[Path, dict]:
    candidate = candidate or b"format: rulespec/v1\nrules: []\n"
    tests = b"[]\n"
    repair = json.dumps(
        {
            "schema_version": "axiom-encode/repair-manifest/v1",
            "citation": citation,
            "runner": "openai-gpt-5.6-sol",
        }
    ).encode()
    payloads = {
        f"target/openai-gpt-5.6-sol/{module}": candidate,
        f"target/openai-gpt-5.6-sol/{module.removesuffix('.yaml')}.test.yaml": tests,
        f"target/openai-gpt-5.6-sol/{module.removesuffix('.yaml')}.repair.json": repair,
    }
    metadata = {
        "schema": "axiom-encode/failed-reencode-diagnostics/v1",
        "citation": citation,
        "country": "us",
        "encoder_commit": "a" * 40,
        "corpus_ref": "b" * 40,
        "rules_engine_ref": "c" * 40,
        "rulespec_ref": "d" * 40,
        "replace_rulespec_path": replace_rulespec_path,
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
    citation: str = "us/statute/42/1437c\u20131",
    module: str = "statutes/42/1437c-1.yaml",
    issue_text: str = "best candidate still needs one repair",
    extra_fields: dict | None = None,
) -> Path:
    root = "target/final-rejected-candidate"
    issues = json.dumps(
        {
            **(extra_fields or {}),
            "schema": "axiom-encode/failed-encode-candidate/v1",
            "citation": citation,
            "path": module,
            "issues": [issue_text],
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


def _add_initial_runner_pair(
    source_archive: Path,
    destination: Path,
    metadata: dict,
    *,
    module: str,
) -> Path:
    runner = "openai-gpt-6-luna"
    payloads = {
        f"target/{runner}/{module}": b"format: rulespec/v1\nrules: [initial]\n",
        f"target/{runner}/{module.removesuffix('.yaml')}.test.yaml": b"[]\n",
    }
    metadata["files"].extend(
        {
            "path": path,
            "size": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        for path, body in payloads.items()
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


def test_accepts_retained_candidate_with_typed_issues(tmp_path):
    archive, metadata = _archive(tmp_path, candidate=b"final regression\n")
    retained = b"strongest retained candidate\n"
    replacement = _add_retained_candidate(
        archive,
        tmp_path / "retained.tar",
        metadata,
        candidate=retained,
        extra_fields={
            "typed_issues": {
                "schema": "axiom-encode/validation-issue/v2",
                "total": 1,
                "typed": 1,
                "issues": [],
            }
        },
    )

    result = extract_candidate(_args(tmp_path, replacement))

    assert result["runner"] == "retained-best"
    assert (Path(result["root"]) / result["path"]).read_bytes() == retained


def test_rejects_retained_candidate_with_malformed_typed_issues(tmp_path):
    archive, metadata = _archive(tmp_path, candidate=b"final regression\n")
    replacement = _add_retained_candidate(
        archive,
        tmp_path / "retained.tar",
        metadata,
        candidate=b"strongest retained candidate\n",
        extra_fields={"typed_issues": {"schema": "other", "issues": []}},
    )

    with pytest.raises(ValueError, match="retained repair candidate metadata"):
        extract_candidate(_args(tmp_path, replacement))


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


@pytest.mark.parametrize("include_initial_runner", [False, True])
def test_extracts_bound_legacy_primary_after_dependent_failure(
    tmp_path, include_initial_runner
):
    archive, metadata = _archive(tmp_path)
    primary_citation = "us/guidance/primary/source"
    primary_path = "us/statutes/42/1437c-1.yaml"
    dependent_citation = "us/statute/42/dependent"
    atomic_source = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v3",
            "source_bundle": [],
            "canonical_refresh_bundle": [],
            "require_complete_source_unit": True,
            "primary_required_test_cases": [
                {
                    "name": "March boundary",
                    "period": {
                        "period_kind": "month",
                        "start": "2026-03-01",
                        "end": "2026-03-31",
                    },
                    "input": {"us:test#input": 1000},
                    "required_output": {"us:test#output": 2.0},
                }
            ],
        }
    )
    metadata.pop("source_bundle_input")
    metadata["atomic_source_input"] = atomic_source
    metadata["citation"] = primary_citation
    metadata["dependent_citation"] = dependent_citation
    metadata["replace_legacy_rulespec_path"] = primary_path
    metadata["generated_lanes"] = ["dependent", "target"]
    metadata["files"] = [
        entry
        for entry in metadata["files"]
        if not entry["path"].endswith(".repair.json")
    ]
    prior = _rewrite_metadata(archive, tmp_path / "legacy-dependent.tar", metadata)
    if include_initial_runner:
        prior = _add_initial_runner_pair(
            prior,
            tmp_path / "legacy-dependent-escalated.tar",
            metadata,
            module="statutes/42/1437c-1.yaml",
        )

    result = extract_candidate(
        _args(
            tmp_path,
            prior,
            citation=primary_citation,
            atomic_source_json=atomic_source,
            dependent_citation=dependent_citation,
            replace_legacy_rulespec_path=primary_path,
            transaction_citation=primary_citation,
            transaction_rulespec_path=primary_path,
        )
    )

    assert result["lane"] == "target"
    assert result["runner"] == "openai-gpt-5.6-sol"
    assert (Path(result["root"]) / "statutes/42/1437c-1.yaml").is_file()


def test_legacy_primary_replay_rejects_empty_boundary_cases_in_both_runs(tmp_path):
    archive, metadata = _archive(tmp_path)
    primary_citation = "us/guidance/primary/source"
    primary_path = "us/statutes/42/1437c-1.yaml"
    dependent_citation = "us/statute/42/dependent"
    atomic_source = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v3",
            "source_bundle": [],
            "canonical_refresh_bundle": [],
            "require_complete_source_unit": True,
            "primary_required_test_cases": [],
        }
    )
    metadata.pop("source_bundle_input")
    metadata["atomic_source_input"] = atomic_source
    metadata["citation"] = primary_citation
    metadata["dependent_citation"] = dependent_citation
    metadata["replace_legacy_rulespec_path"] = primary_path
    metadata["generated_lanes"] = ["dependent", "target"]
    metadata["files"] = [
        entry
        for entry in metadata["files"]
        if not entry["path"].endswith(".repair.json")
    ]
    prior = _rewrite_metadata(archive, tmp_path / "empty-boundaries.tar", metadata)

    with pytest.raises(ValueError, match="boundary tests"):
        extract_candidate(
            _args(
                tmp_path,
                prior,
                citation=primary_citation,
                atomic_source_json=atomic_source,
                dependent_citation=dependent_citation,
                replace_legacy_rulespec_path=primary_path,
                transaction_citation=primary_citation,
                transaction_rulespec_path=primary_path,
            )
        )


def test_legacy_primary_replay_rejects_dependent_lane_before_archive_read(tmp_path):
    primary_path = "us/statutes/42/1437c-1.yaml"
    with pytest.raises(ValueError, match="legacy dependent repair target identity"):
        extract_candidate(
            _args(
                tmp_path,
                tmp_path / "missing-archive.tar",
                citation="us/guidance/primary/source",
                atomic_source_json=json.dumps(
                    {
                        "schema": "axiom-encode/atomic-source-transaction/v3",
                        "source_bundle": [],
                        "canonical_refresh_bundle": [],
                        "require_complete_source_unit": True,
                        "primary_required_test_cases": [{"name": "March boundary"}],
                    }
                ),
                dependent_citation="us/statute/42/dependent",
                replace_legacy_rulespec_path=primary_path,
                transaction_citation="us/guidance/primary/source",
                transaction_rulespec_path=primary_path,
                repair_lane="dependent",
            )
        )


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [
        ("corpus_artifact_run_id", "124"),
        ("provision_export_run_ids", ["456", "790"]),
    ],
)
def test_legacy_primary_replay_binds_corpus_transport_export_ids(
    tmp_path, changed_field, changed_value
):
    archive, metadata = _archive(tmp_path)
    primary_citation = "us/guidance/primary/source"
    primary_path = "us/statutes/42/1437c-1.yaml"
    dependent_citation = "us/statute/42/dependent"
    transaction = {
        "schema": "axiom-encode/atomic-source-transaction/v3",
        "source_bundle": [],
        "canonical_refresh_bundle": [],
        "require_complete_source_unit": True,
        "primary_required_test_cases": [{"name": "March boundary"}],
    }
    transport = {
        "schema": "axiom-encode/corpus-artifact-transport/v1",
        "transaction": transaction,
        "corpus_artifact_run_id": "123",
        "provision_export_run_ids": ["456", "789"],
    }
    prior_source = json.dumps(transport)
    metadata.pop("source_bundle_input")
    metadata["atomic_source_input"] = prior_source
    metadata["citation"] = primary_citation
    metadata["dependent_citation"] = dependent_citation
    metadata["replace_legacy_rulespec_path"] = primary_path
    metadata["generated_lanes"] = ["dependent", "target"]
    metadata["files"] = [
        entry
        for entry in metadata["files"]
        if not entry["path"].endswith(".repair.json")
    ]
    prior = _rewrite_metadata(archive, tmp_path / "transport-dependent.tar", metadata)
    replay_args = {
        "citation": primary_citation,
        "dependent_citation": dependent_citation,
        "replace_legacy_rulespec_path": primary_path,
        "transaction_citation": primary_citation,
        "transaction_rulespec_path": primary_path,
    }

    exact = extract_candidate(
        _args(tmp_path, prior, atomic_source_json=prior_source, **replay_args)
    )
    assert exact["lane"] == "target"

    changed_source = json.dumps({**transport, changed_field: changed_value})
    assert immutable_atomic_source_contract(
        changed_source
    ) == immutable_atomic_source_contract(prior_source)
    with pytest.raises(ValueError, match="exact prior source transport"):
        extract_candidate(
            _args(
                tmp_path,
                prior,
                destination=tmp_path / "changed-export-ids",
                atomic_source_json=changed_source,
                **replay_args,
            )
        )


def test_rejects_legacy_primary_when_required_test_contract_changes(tmp_path):
    archive, metadata = _archive(tmp_path)
    metadata.pop("source_bundle_input")
    metadata["atomic_source_input"] = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v3",
            "source_bundle": [],
            "canonical_refresh_bundle": [],
            "require_complete_source_unit": True,
            "primary_required_test_cases": [{"name": "prior"}],
        }
    )
    metadata["dependent_citation"] = "us/statute/42/dependent"
    metadata["replace_legacy_rulespec_path"] = metadata["replace_rulespec_path"]
    metadata["generated_lanes"] = ["dependent", "target"]
    prior = _rewrite_metadata(archive, tmp_path / "legacy-dependent.tar", metadata)

    with pytest.raises(ValueError, match="metadata mismatch: atomic_source_input"):
        extract_candidate(
            _args(
                tmp_path,
                prior,
                atomic_source_json="[]",
                dependent_citation="us/statute/42/dependent",
                replace_legacy_rulespec_path="us/statutes/42/1437c-1.yaml",
            )
        )


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


@pytest.mark.parametrize(
    ("initial_runner", "escalation_runner"),
    [
        ("openai-gpt-5.6-terra", "openai-gpt-5.6-sol"),
        ("openai-gpt-6-luna", "openai-gpt-6-sol"),
    ],
)
def test_extracts_digest_bound_source_candidates_from_final_composition(
    tmp_path, initial_runner, escalation_runner
):
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
        f"source-01/{initial_runner}/statutes/7/2015/f.yaml": b"older\n",
        f"source-01/{initial_runner}/statutes/7/2015/f.test.yaml": b"[]\n",
        f"source-01/{escalation_runner}/statutes/7/2015/f.yaml": source_one,
        f"source-01/{escalation_runner}/statutes/7/2015/f.test.yaml": b"[]\n",
        f"source-02/{escalation_runner}/policies/usda/fns/snap-obbb-alien-eligibility-implementation-memo.yaml": source_two,
        f"source-02/{escalation_runner}/policies/usda/fns/snap-obbb-alien-eligibility-implementation-memo.test.yaml": b"[]\n",
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
    assert first["runner"] == escalation_runner
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


def test_extracts_new_source_repair_without_replace_path(tmp_path):
    archive, metadata = _archive(
        tmp_path,
        citation="us/guidance/example/new-source",
        module="policies/example/new-source.yaml",
        replace_rulespec_path=None,
    )
    candidate = b"format: rulespec/v1\nrules: []\n"
    replacement = _add_retained_candidate(
        archive,
        tmp_path / "new-source-repair.tar",
        metadata,
        candidate=candidate,
        citation="us/guidance/example/new-source",
        module="policies/example/new-source.yaml",
    )

    result = extract_candidate(
        _args(
            tmp_path,
            replacement,
            citation="us/guidance/example/new-source",
            replace_rulespec_path="",
        )
    )

    assert result["path"] == "policies/example/new-source.yaml"
    assert (Path(result["root"]) / result["path"]).read_bytes() == candidate


def test_extracts_new_source_repair_with_empty_cli_transaction_path(tmp_path):
    archive, metadata = _archive(
        tmp_path,
        citation="us/guidance/example/new-source",
        module="policies/example/new-source.yaml",
        replace_rulespec_path=None,
    )
    replacement = _add_retained_candidate(
        archive,
        tmp_path / "new-source-cli-repair.tar",
        metadata,
        candidate=b"format: rulespec/v1\nrules: []\n",
        citation="us/guidance/example/new-source",
        module="policies/example/new-source.yaml",
    )
    args = _args(
        tmp_path,
        replacement,
        citation="us/guidance/example/new-source",
        replace_rulespec_path="",
    )
    args.transaction_rulespec_path = ""

    result = extract_candidate(args)

    assert result["path"] == "policies/example/new-source.yaml"


@pytest.mark.parametrize(
    "citation",
    [
        "us/guidance/example/new-source/",
        "us//guidance/example/new-source",
    ],
)
def test_rejects_noncanonical_new_source_citation(tmp_path, citation):
    archive, _ = _archive(
        tmp_path,
        citation=citation,
        module="policies/example/new-source.yaml",
        replace_rulespec_path=None,
    )

    with pytest.raises(ValueError, match="exact canonical"):
        extract_candidate(
            _args(
                tmp_path,
                archive,
                citation=citation,
                replace_rulespec_path="",
            )
        )


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


def test_retained_large_diagnostics_do_not_use_yaml_size_bound(tmp_path):
    archive, metadata = _archive(tmp_path)
    retained = _add_retained_candidate(
        archive,
        tmp_path / "large-issues.tar",
        metadata,
        candidate=b"format: rulespec/v1\nrules: []\n",
        issue_text="diagnostic " * 60000,
    )
    result = extract_candidate(_args(tmp_path, retained))
    assert (
        Path(result["root"], result["path"]).read_text()
        == "format: rulespec/v1\nrules: []\n"
    )


def test_retained_diagnostics_reject_shared_limit_plus_one(tmp_path):
    from scripts.extract_repair_candidate import MAX_ISSUES_BYTES

    archive, metadata = _archive(tmp_path)
    retained = _add_retained_candidate(
        archive,
        tmp_path / "oversize-issues.tar",
        metadata,
        candidate=b"format: rulespec/v1\nrules: []\n",
        issue_text="x" * MAX_ISSUES_BYTES,
    )
    with pytest.raises(ValueError, match="exceeds its size limit"):
        extract_candidate(_args(tmp_path, retained))


@pytest.mark.parametrize(
    "prior_lane,changed_primary,incompatible,second_dependent",
    [
        ("target", False, False, False),
        ("target", False, False, True),
        ("dependent", False, False, True),
        ("dependent", False, False, False),
        ("target", False, True, False),
        ("target", False, True, True),
        ("target", True, False, False),
    ],
)
def test_workflow_authenticates_retained_lane_before_new_dependent(
    tmp_path: Path,
    prior_lane: str,
    changed_primary: bool,
    incompatible: bool,
    second_dependent: bool,
) -> None:
    import os
    import subprocess
    import sys

    import yaml

    root = Path(__file__).resolve().parents[1]
    primary_citation = "us/guidance/primary/source"
    primary_path = "us/guidance/primary/source.yaml"
    dependent_citation = "us/statute/42/1437c\u20131"
    dependent_path = "us/statutes/42/1437c-1.yaml"
    if prior_lane == "target":
        archive, metadata = _archive(
            tmp_path,
            citation=primary_citation,
            module="guidance/primary/source.yaml",
            replace_rulespec_path=primary_path,
        )
    else:
        archive, metadata = _archive(tmp_path)
        archive = _rewrite_as_dependent_candidate(
            archive, tmp_path / "dependent.tar", metadata
        )
    repo = tmp_path / "rulespec-us"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True
        ).strip()

    git("init", "--quiet")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    for path in (primary_path, dependent_path):
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("format: rulespec/v1\nrules: []\n")
        target.with_suffix(".test.yaml").write_text("[]\n")
        manifest = repo / ".axiom/encoding-manifests" / Path(path).with_suffix(".json")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}\n")
    git("add", ".")
    git("commit", "-qm", "source base")
    metadata["rulespec_ref"] = git("rev-parse", "HEAD")
    if incompatible:
        metadata["corpus_ref"] = "e" * 40
    archive = _rewrite_metadata(archive, tmp_path / "bound.tar", metadata)
    if changed_primary:
        (repo / primary_path).write_text("changed primary\n")
        git("add", ".")
        git("commit", "-qm", "changed primary")
    workflow = yaml.safe_load(
        (root / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    step = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Resolve trusted prior-run repair candidate"
    )
    command = (
        "set -euo pipefail\n" + step[step.index("extract_authenticated_repair() {") :]
    )
    command = command.replace("axiom-encode/.venv/bin/python", sys.executable)
    for name in ("extract_repair_candidate.py", "verify_repair_base_advance.py"):
        command = command.replace(
            f"axiom-encode/scripts/{name}", str(root / "scripts" / name)
        )
    output_file = tmp_path / "outputs"
    completed = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_OUTPUT": str(output_file),
            "repair_archive": str(archive),
            "repair_encoder_commit": "a" * 40,
            "repair_citation": dependent_citation,
            "repair_rulespec_path": dependent_path,
            "REPAIR_RUN_LANE": "dependent",
            "CITATION": primary_citation,
            "REPLACE_RULESPEC_PATH": primary_path,
            "SECOND_DEPENDENT_CITATION": "us/statute/42/402/w"
            if second_dependent
            else "",
            "COUNTRY": "us",
            "CORPUS_REF": "b" * 40,
            "RULES_ENGINE_REF": "c" * 40,
            "RULESPEC_REF": git("rev-parse", "HEAD"),
            "RULESPEC_CHECKOUT": str(repo),
            "REPAIR_RUN_ID": "1234",
            "ATOMIC_SOURCE_JSON": "[]",
            "source_bundle_json": "[]",
            "primary_required_test_cases_json": "[]",
            "EXISTING_SIGNED_IMPORTS_JSON": "[]",
        },
    )
    if second_dependent and prior_lane == "dependent":
        assert completed.returncode != 0
        assert (
            "two fresh dependents require an authenticated single-target primary"
            in completed.stderr
        )
        assert not (tmp_path / "repair-candidate").exists()
        assert not output_file.exists()
        return
    if incompatible:
        assert completed.returncode != 0
        if second_dependent:
            assert (
                "two fresh dependents require an authenticated single-target primary"
                in completed.stderr
            )
            assert (
                "metadata mismatch: corpus_ref"
                in (tmp_path / "repair-primary-probe.log").read_text()
            )
        else:
            assert "metadata mismatch: corpus_ref" in completed.stderr
        assert not output_file.exists()
        return
    if changed_primary:
        assert completed.returncode != 0
        assert (
            "target identity changed after its source RuleSpec base" in completed.stderr
        )
        assert (tmp_path / "repair-primary-candidate.json").is_file()
        assert not (tmp_path / "repair-candidate").exists()
        assert not output_file.exists()
        return
    assert completed.returncode == 0, completed.stderr
    outputs = dict(line.split("=", 1) for line in output_file.read_text().splitlines())
    expected_path = primary_path if prior_lane == "target" else dependent_path
    assert outputs["lane"] == prior_lane
    assert outputs["rulespec_path"] == expected_path
    assert outputs["path"] == expected_path.removeprefix("us/")
    assert outputs["source_rulespec_ref"] == metadata["rulespec_ref"]
    candidate = Path(outputs["root"]) / outputs["path"]
    assert (
        hashlib.sha256(candidate.read_bytes()).hexdigest() == outputs["rulespec_sha256"]
    )
    assert (
        hashlib.sha256(candidate.with_suffix(".test.yaml").read_bytes()).hexdigest()
        == outputs["tests_sha256"]
    )


@pytest.mark.parametrize("changed_export_id", [False, True])
def test_workflow_replays_legacy_primary_and_reports_probe_failure(
    tmp_path: Path, changed_export_id: bool
) -> None:
    import os
    import subprocess
    import sys

    import yaml

    root = Path(__file__).resolve().parents[1]
    primary_citation = "us/guidance/primary/source"
    primary_path = "us/statutes/42/1437c-1.yaml"
    dependent_citation = "us/guidance/dependent"
    dependent_path = "us/guidance/dependent.yaml"
    transport = {
        "schema": "axiom-encode/corpus-artifact-transport/v1",
        "transaction": {
            "schema": "axiom-encode/atomic-source-transaction/v3",
            "source_bundle": [],
            "canonical_refresh_bundle": [],
            "require_complete_source_unit": True,
            "primary_required_test_cases": [{"name": "March boundary"}],
        },
        "corpus_artifact_run_id": "789",
    }
    prior_source = json.dumps(transport)
    archive, metadata = _archive(tmp_path)
    metadata.pop("source_bundle_input")
    metadata["atomic_source_input"] = prior_source
    metadata["citation"] = primary_citation
    metadata["dependent_citation"] = dependent_citation
    metadata["replace_legacy_rulespec_path"] = primary_path
    metadata["generated_lanes"] = ["dependent", "target"]
    metadata["files"] = [
        entry
        for entry in metadata["files"]
        if not entry["path"].endswith(".repair.json")
    ]

    repo = tmp_path / "rulespec-us"
    repo.mkdir()

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True
        ).strip()

    git("init", "--quiet")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    for path in (primary_path, dependent_path):
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("format: rulespec/v1\nrules: []\n")
        target.with_suffix(".test.yaml").write_text("[]\n")
        manifest = repo / ".axiom/encoding-manifests" / Path(path).with_suffix(".json")
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}\n")
    git("add", ".")
    git("commit", "-qm", "source base")
    metadata["rulespec_ref"] = git("rev-parse", "HEAD")
    archive = _rewrite_metadata(archive, tmp_path / "legacy-transport.tar", metadata)

    workflow = yaml.safe_load(
        (root / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    step = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Resolve trusted prior-run repair candidate"
    )
    command = (
        "set -euo pipefail\n" + step[step.index("extract_authenticated_repair() {") :]
    )
    command = command.replace("axiom-encode/.venv/bin/python", sys.executable)
    for name in ("extract_repair_candidate.py", "verify_repair_base_advance.py"):
        command = command.replace(
            f"axiom-encode/scripts/{name}", str(root / "scripts" / name)
        )
    requested_transport = {
        **transport,
        "corpus_artifact_run_id": "790" if changed_export_id else "789",
    }
    output_file = tmp_path / "outputs"
    completed = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_OUTPUT": str(output_file),
            "repair_archive": str(archive),
            "repair_encoder_commit": "a" * 40,
            "repair_citation": dependent_citation,
            "repair_rulespec_path": dependent_path,
            "REPAIR_RUN_LANE": "dependent",
            "CITATION": primary_citation,
            "REPLACE_RULESPEC_PATH": primary_path,
            "REPLACE_LEGACY_RULESPEC_PATH": primary_path,
            "DEPENDENT_CITATION": dependent_citation,
            "SECOND_DEPENDENT_CITATION": "",
            "COUNTRY": "us",
            "CORPUS_REF": "b" * 40,
            "RULES_ENGINE_REF": "c" * 40,
            "RULESPEC_REF": metadata["rulespec_ref"],
            "RULESPEC_CHECKOUT": str(repo),
            "REPAIR_RUN_ID": "1234",
            "ATOMIC_SOURCE_JSON": json.dumps(requested_transport),
            "source_bundle_json": "[]",
            "primary_required_test_cases_json": '[{"name":"March boundary"}]',
            "EXISTING_SIGNED_IMPORTS_JSON": "[]",
        },
    )
    if changed_export_id:
        assert completed.returncode != 0
        assert "exact prior source transport" in completed.stderr
        assert (
            "legacy dependent repair target identity is invalid" not in completed.stderr
        )
        assert not output_file.exists()
    else:
        assert completed.returncode == 0, completed.stderr
        outputs = dict(
            line.split("=", 1) for line in output_file.read_text().splitlines()
        )
        assert outputs["lane"] == "target"
        assert outputs["rulespec_path"] == primary_path


_PRESERVATION_REQUEST = json.dumps(
    {
        "schema": "axiom-encode/atomic-source-transaction/v7",
        "transaction": [],
        "preserve_repair_companion": True,
    }
)
_ORIGIN_REFUSAL = (
    "^prior v7 repair replay cannot authenticate the originating companion; "
    "select the originating failed run$"
)
_ORIGINAL_COMPANION = (
    b'- name: sample\n  period: "2025"\n  input: {}\n  output: {x: 0, y: 1}\n'
)


def _preservation_archive(tmp_path, *, retained, tests, prior_v7, consumed="valid"):
    archive, metadata = _archive(tmp_path)
    metadata.pop("source_bundle_input")
    metadata["atomic_source_input"] = _PRESERVATION_REQUEST if prior_v7 else "[]"
    if prior_v7:
        metadata["repair_execution"] = {
            "requested_mode": "legacy_inference",
            "mode": "full_artifact",
            "tests_only": False,
            "preserve_repair_companion": True,
        }
        identity = {
            "lane": "target",
            "run_id": "1111",
            "path": "statutes/42/1437c-1.yaml",
            "runner": "openai-gpt-5.6-sol",
            "source_rulespec_ref": "d" * 40,
            "rulespec_sha256": hashlib.sha256(
                b"format: rulespec/v1\nrules: []\n"
            ).hexdigest(),
            "tests_sha256": hashlib.sha256(_ORIGINAL_COMPANION).hexdigest(),
        }
        if consumed == "malformed":
            identity["tests_sha256"] = "invalid"
        elif consumed == "mismatching":
            identity["tests_sha256"] = "f" * 64
        if consumed != "missing":
            metadata["repair_candidate"] = identity
    for entry in metadata["files"]:
        if entry["path"].endswith(".test.yaml"):
            entry.update(size=len(tests), sha256=hashlib.sha256(tests).hexdigest())
    rewritten = tmp_path / "preservation.tar"
    with tarfile.open(archive) as source, tarfile.open(rewritten, "w") as target:
        for member in source:
            body = source.extractfile(member).read()
            if member.name == "metadata.json":
                body = json.dumps(metadata).encode()
            elif member.name.endswith(".test.yaml"):
                body = tests
            member.size = len(body)
            target.addfile(member, io.BytesIO(body))
    if retained:
        rewritten = _add_retained_candidate(
            rewritten,
            tmp_path / "retained.tar",
            metadata,
            candidate=b"format: rulespec/v1\nrules: []\n",
            tests=tests,
        )
    return rewritten


@pytest.mark.parametrize("retained", [False, True], ids=["runner", "retained-best"])
@pytest.mark.parametrize(
    "consumed,unchanged,requested_source",
    [
        ("valid", False, _PRESERVATION_REQUEST),
        ("missing", False, _PRESERVATION_REQUEST),
        ("malformed", False, _PRESERVATION_REQUEST),
        ("mismatching", False, _PRESERVATION_REQUEST),
        ("valid", True, _PRESERVATION_REQUEST),
        ("valid", False, "[]"),
    ],
    ids=["weakened", "missing", "malformed", "mismatching", "unchanged", "downgrade"],
)
def test_prior_v7_replay_refuses_unestablished_origin(
    tmp_path, retained, consumed, unchanged, requested_source
):
    tests = _ORIGINAL_COMPANION
    if not unchanged:
        tests = tests.replace(b"{x: 0, y: 1}", b"{x: 0}")
    archive = _preservation_archive(
        tmp_path, retained=retained, tests=tests, prior_v7=True, consumed=consumed
    )
    args = _args(tmp_path, archive, atomic_source_json=requested_source)
    with pytest.raises(ValueError, match=_ORIGIN_REFUSAL):
        extract_candidate(args)
    assert list(args.destination.rglob("*")) == []


@pytest.mark.parametrize("retained", [False, True], ids=["runner", "retained-best"])
def test_historical_candidate_starts_preservation_with_authenticated_cli_seed(
    tmp_path, retained
):
    from types import SimpleNamespace

    from axiom_encode import cli

    archive = _preservation_archive(
        tmp_path, retained=retained, tests=_ORIGINAL_COMPANION, prior_v7=False
    )
    extracted = extract_candidate(
        _args(tmp_path, archive, atomic_source_json=_PRESERVATION_REQUEST)
    )
    assert extracted["runner"] == (
        "retained-best" if retained else "openai-gpt-5.6-sol"
    )
    args = SimpleNamespace(
        repair_candidate_root=Path(extracted["root"]),
        repair_candidate_path=Path(extracted["path"]),
        repair_candidate_rulespec_sha256=extracted["rulespec_sha256"],
        repair_candidate_tests_sha256=extracted["tests_sha256"],
        preserve_repair_companion=True,
        apply=True,
        require_complete_source_unit=True,
        citation="us/statute/42/1437c–1",
    )
    seed = cli._load_initial_validation_retry_candidate(args)
    cli._validate_preserved_repair_companion(args, seed)
    result = SimpleNamespace()
    setattr(result, cli._PRESERVED_REPAIR_COMPANION_ATTR, seed)
    companion = Path(extracted["root"]) / extracted["path"].replace(
        ".yaml", ".test.yaml"
    )
    record = cli._preserved_repair_companion_record(result, companion)
    assert record["assertions"] == 2
    assert (
        record["original_tests_sha256"]
        == hashlib.sha256(_ORIGINAL_COMPANION).hexdigest()
    )
    companion.write_bytes(_ORIGINAL_COMPANION.replace(b"{x: 0, y: 1}", b"{x: 0}"))
    with pytest.raises(RuntimeError, match="Cannot apply repair"):
        cli._preserved_repair_companion_record(result, companion)
