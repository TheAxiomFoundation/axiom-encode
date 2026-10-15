"""Reproduce the pinned subdivision audit, one bounded shard per invocation.

Run prepare, consecutive shard ranges, then summarize. The default shard is
ten artifacts; use --artifact-limit for slower artifacts. Inputs and outputs
stay in the supplied directory; prepare and shard refuse overwrites.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASELINE_COMMIT = "4f3547461300bfa70569f57c83728b2ad25d9d0b"
BASELINE_SHA256 = "875df7eb77efdaef6add1ca6756989f2881b5825dcf65dd6e66c0f30c43f3b62"
RELEASE_NAME = "us-rulespec-2026-08-08-obbb-alien-snap"
CONTENT_SHA256 = "0d69a0cdbe024fc2276f3c261c00402bb0a47488ac5f482cc20bd3404980adbc"
RELEASE_SHA256 = "8d0fd2993fa667b13205947b71b91aafbb56be73351e9acebbd03255e233190a"
RELEASE_BYTES = 7150325
CORPUS_COMMIT = "e79781bddeaba4d3697c5375c7371b2a038f0f74"
RELEASE_URL = (
    "https://pub-a8952f8657fc49fda358146ac001366c.r2.dev/releases/"
    f"{RELEASE_NAME}/{CONTENT_SHA256}.json"
)
# Published roots in scripts/verify_real_defects.py at BASELINE_COMMIT.
PUBLIC_KEYS = (
    "KwIYEbbs/905yxn/9Yi6jYTF8oyZcq1FlxiG4e0y0tg=",
    "9EJzeLnZsdBoDEDkU8xbi7HjYVVvd4HDX/BxPHPvXTk=",
)
SOURCE = Path("src/axiom_encode/harness/source_completeness.py")
AUDIT = Path("scripts/audit_subdivision_recall_masks.py")
SHARD_SIZE = 10
SHARD_BUDGET_SECONDS = 540
CHILD_TIMEOUT_SECONDS = 480
THREAD_CAPS = {
    name: "4"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
} | {"OMP_WAIT_POLICY": "PASSIVE"}


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def pinned_inputs(directory: Path):
    release_path = directory / "release.json"
    baseline = directory / "baseline.py"
    if (
        release_path.stat().st_size != RELEASE_BYTES
        or digest(release_path) != RELEASE_SHA256
        or digest(baseline) != BASELINE_SHA256
    ):
        raise ValueError("Pinned input bytes differ")
    # Load the repository verifier directly; package initialization is unrelated
    # to release verification and also imports the full encoding toolchain.
    verifier_name = "_subdivision_release_verifier"
    spec = importlib.util.spec_from_file_location(
        verifier_name, Path("src/axiom_encode/corpus_release.py")
    )
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load repository release verifier")
    verifier = importlib.util.module_from_spec(spec)
    sys.modules[verifier_name] = verifier
    spec.loader.exec_module(verifier)
    release = verifier.verify_release_object(
        json.loads(release_path.read_bytes()), public_key=PUBLIC_KEYS
    )
    if (release.name, release.content_sha256, release.git_commit) != (
        RELEASE_NAME,
        CONTENT_SHA256,
        CORPUS_COMMIT,
    ):
        raise ValueError("Signed input identity differs")
    artifacts = [
        dataclasses.asdict(item)
        for item in release.artifacts
        if item.artifact_class == "provisions"
        and item.path.split("/")[4] in {"manual", "guidance", "statute"}
        and (
            item.path.split("/")[3] == "us" or item.path.split("/")[3].startswith("us-")
        )
    ]
    if (
        len(artifacts) != 196
        or sum(item["row_count"] for item in artifacts) != 79687
        or sum(item["byte_count"] for item in artifacts) != 249961962
    ):
        raise ValueError("Signed audit selection differs")
    return artifacts


def prepare(directory: Path):
    directory.mkdir(parents=True, exist_ok=False)
    request = urllib.request.Request(
        RELEASE_URL, headers={"User-Agent": "axiom-encode-subdivision-audit"}
    )
    with (
        urllib.request.urlopen(request, timeout=60) as response,
        (directory / "release.json").open("xb") as output,
    ):
        received = 0
        while chunk := response.read(1024 * 1024):
            received += len(chunk)
            if received > RELEASE_BYTES:
                raise ValueError("Release exceeds pinned byte count")
            output.write(chunk)
    with (directory / "baseline.py").open("xb") as output:
        subprocess.run(
            ["git", "show", f"{BASELINE_COMMIT}:{SOURCE.as_posix()}"],
            stdout=output,
            check=True,
            timeout=60,
        )
    pinned_inputs(directory)
    print("Pinned release signature, input bytes and full selection verified")


def shard(
    directory: Path,
    start: int,
    limit: int,
    row_start: int = 0,
    row_limit: int | None = None,
):
    started = time.monotonic()
    artifacts = pinned_inputs(directory)
    if not 0 <= start < len(artifacts) or limit < 1:
        raise ValueError("Use a nonnegative artifact start and positive limit")
    sliced = row_start != 0 or row_limit is not None
    if sliced and (
        limit != 1
        or row_start < 0
        or row_limit is not None
        and row_limit < 1
        or not row_start
        < (
            row_start + row_limit
            if row_limit is not None
            else artifacts[start]["row_count"]
        )
        <= artifacts[start]["row_count"]
    ):
        raise ValueError("Use a nonempty in-bounds row range in exactly one artifact")
    name = f"batch-{start:03}" + (f"-row-{row_start:05}" if sliced else "")
    stem = directory / name
    if any(stem.with_suffix(suffix).exists() for suffix in (".json", ".log")):
        raise ValueError("Shard output already exists")
    command = [
        sys.executable,
        "-u",
        str(AUDIT),
        "--release",
        str(directory / "release.json"),
        "--release-name",
        RELEASE_NAME,
        "--content-sha256",
        CONTENT_SHA256,
        "--baseline",
        str(directory / "baseline.py"),
        "--output",
        str(stem.with_suffix(".json")),
        "--scratch",
        str(directory / f"download-{name}-{limit:03}"),
        "--artifact-start",
        str(start),
        "--artifact-limit",
        str(limit),
    ]
    if sliced:
        command.extend(("--row-start", str(row_start)))
        if row_limit is not None:
            command.extend(("--row-limit", str(row_limit)))
    for key in PUBLIC_KEYS:
        command.extend(("--public-key", key))
    remaining = SHARD_BUDGET_SECONDS - (time.monotonic() - started)
    if remaining <= 0:
        raise subprocess.TimeoutExpired(command, SHARD_BUDGET_SECONDS)
    with stem.with_suffix(".log").open("x") as log:
        subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=os.environ | THREAD_CAPS,
            timeout=min(CHILD_TIMEOUT_SECONDS, remaining),
            check=True,
        )
    completion = {
        "status": "passed",
        "exit_code": 0,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "summary_sha256": digest(stem.with_suffix(".json")),
        "rows_sha256": digest(stem.with_suffix(".rows.jsonl")),
        "removals_sha256": digest(stem.with_suffix(".removals.jsonl")),
    }
    with stem.with_suffix(".completed.json").open("x") as output:
        output.write(json.dumps(completion, indent=2) + "\n")
    print(f"Artifacts {start} through {min(start + limit, len(artifacts))} passed")


def verify_row_ranges(ranges, artifacts, artifact_cursor, row_cursor):
    """Bind every half-open row range to signed rows and ordered coverage."""
    selected_rows = 0
    for item in ranges:
        index = item["artifact_index"]
        if not 0 <= index < len(artifacts):
            raise ValueError("Row range artifact index differs")
        signed = artifacts[index]
        start, end = item["row_start"], item["row_end"]
        if (
            index != artifact_cursor
            or start != row_cursor
            or not 0 <= start < end <= signed["row_count"]
            or item["path"] != signed["path"]
            or item["artifact_row_count"] != signed["row_count"]
            or item["physical_row_count"] != signed["row_count"]
            or item["selected_row_count"] != end - start
        ):
            raise ValueError("Row range identity or coverage differs")
        selected_rows += end - start
        if end == signed["row_count"]:
            artifact_cursor += 1
            row_cursor = 0
        else:
            row_cursor = end
    return artifact_cursor, row_cursor, selected_rows


def summarize(directory: Path):
    artifacts = pinned_inputs(directory)
    identities = {
        "release": RELEASE_NAME,
        "content_sha256": CONTENT_SHA256,
        "corpus_commit": CORPUS_COMMIT,
        "release_object_sha256": RELEASE_SHA256,
        "script_sha256": digest(AUDIT),
        "baseline_module_sha256": BASELINE_SHA256,
        "current_module_sha256": digest(SOURCE),
    }
    counts = (
        "checked_rows",
        "entry_unchanged_rows",
        "entry_changed_rows",
        "recall_changed_rows",
        "removed_inventory_occurrences",
        "added_inventory_occurrences",
        "glued_quantity_guard_changed_rows",
        "restored_glued_quantity_spans",
        "guard_added_inventory_occurrences",
        "ledger_changed_rows",
    )
    summary = identities | {
        "status": "passed",
        "reproducer_sha256": digest(Path(__file__)),
        "baseline_commit": BASELINE_COMMIT,
        "release_url": RELEASE_URL,
        "release_object_bytes": RELEASE_BYTES,
        "public_keys": [
            {
                "base64_raw_ed25519": key,
                "raw_key_sha256": hashlib.sha256(base64.b64decode(key)).hexdigest(),
            }
            for key in PUBLIC_KEYS
        ],
        "full_release_files": len(artifacts),
        "full_release_rows": sum(item["row_count"] for item in artifacts),
        "declared_bytes": sum(item["byte_count"] for item in artifacts),
        "artifact_manifest_sha256": hashlib.sha256(
            json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "signature_verified": True,
        "scope": "All signed US manual, guidance and statute rows, including headings; no row or length exclusions. Numeric recall only.",
        "removed_entry_identifiers": {"legal-locator": 0, "bounded-geography": 0},
        "entry_removed_inventory_occurrences": 0,
        "entry_removed_inventory_occurrences_by_kind": {
            "legal-locator": 0,
            "bounded-geography": 0,
        },
        "batches": [],
        **dict.fromkeys(counts, 0),
    }
    ledger_paths = {
        "rows": directory / "combined.rows.jsonl",
        "removals": directory / "combined.removals.jsonl",
    }
    with (
        ledger_paths["rows"].open("xb") as rows_out,
        ledger_paths["removals"].open("xb") as removals_out,
    ):
        artifact_cursor = row_cursor = 0
        paths = []
        for path in directory.glob("batch-*.json"):
            if match := re.fullmatch(r"batch-(\d+)(?:-row-(\d+))?", path.stem):
                paths.append((int(match[1]), int(match[2] or 0), path))
        for filename_index, filename_row, path in sorted(paths):
            batch = json.loads(path.read_bytes())
            completion = json.loads(path.with_suffix(".completed.json").read_bytes())
            if (
                completion["status"] != "passed"
                or completion["exit_code"] != 0
                or completion["summary_sha256"] != digest(path)
                or any(
                    completion[f"{kind}_sha256"]
                    != digest(path.with_suffix(f".{kind}.jsonl"))
                    for kind in ("rows", "removals")
                )
            ):
                raise ValueError("Shard subprocess completion differs")
            start, end = batch["artifact_start"], batch["artifact_end"]
            if (
                batch["status"] != "passed"
                or batch["errors"]
                or any(batch[key] != value for key, value in identities.items())
                or not batch["signature_verified"]
                or not batch["baseline_cleanup_suffix_unchanged"]
                or not batch["bounded_glued_quantity_guard_verified"]
                or not batch["all_rows_runtime_compared"]
                or not batch["full_artifact_row_counts_verified"]
                or start != artifact_cursor
                or filename_index != start
                or filename_row != row_cursor
                or not start < end <= len(artifacts)
                or batch["files"] != artifacts[start:end]
                or batch["declared_files"] != end - start
                or batch["declared_bytes"]
                != sum(item["byte_count"] for item in artifacts[start:end])
                or batch["checked_rows"] != batch["declared_rows"]
            ):
                raise ValueError(f"Shard identity or coverage differs: {start}")
            ranges = batch["row_ranges"]
            if len(ranges) != end - start or [
                item["artifact_index"] for item in ranges
            ] != list(range(start, end)):
                raise ValueError("Row ranges differ from selected artifacts")
            artifact_cursor, row_cursor, selected_rows = verify_row_ranges(
                ranges, artifacts, artifact_cursor, row_cursor
            )
            if batch["declared_rows"] != selected_rows:
                raise ValueError("Declared rows differ from signed row selection")
            if not summary["batches"]:
                for key in (
                    "selector_sha256",
                    "python_version",
                    "gil_disabled_build",
                    "existing_definitions_unchanged",
                    "baseline_cleanup_suffix_unchanged",
                    "bounded_glued_quantity_guard_verified",
                    "all_rows_runtime_compared",
                    "full_artifact_row_counts_verified",
                ):
                    summary[key] = batch[key]
            elif any(
                summary[key] != batch[key]
                for key in (
                    "selector_sha256",
                    "python_version",
                    "gil_disabled_build",
                    "existing_definitions_unchanged",
                    "baseline_cleanup_suffix_unchanged",
                    "bounded_glued_quantity_guard_verified",
                    "all_rows_runtime_compared",
                    "full_artifact_row_counts_verified",
                )
            ):
                raise ValueError("Shard environment differs")
            for key in counts:
                summary[key] += batch[key]
            for key in summary["removed_entry_identifiers"]:
                summary["removed_entry_identifiers"][key] += batch[
                    "removed_entry_identifiers"
                ][key]
            record = {
                "artifact_start": start,
                "artifact_end": end,
                "row_ranges": ranges,
                "completion_sha256": digest(path.with_suffix(".completed.json")),
                "elapsed_seconds": completion["elapsed_seconds"],
            }
            for kind, out in (("rows", rows_out), ("removals", removals_out)):
                source = path.with_suffix(f".{kind}.jsonl")
                record[f"{kind}_sha256"] = digest(source)
                record[f"{kind}_bytes"] = source.stat().st_size
                with source.open("rb") as stream:
                    row_count = 0
                    for raw in stream:
                        out.write(raw)
                        row_count += 1
                        if kind == "removals":
                            row = json.loads(raw)
                            for item in row["entry_inventory_delta"]["removed"]:
                                proofs = [
                                    proof
                                    for proof in row["recall_projections"]
                                    if all(
                                        proof[key] == item[key]
                                        for key in ("start", "end", "raw")
                                    )
                                ]
                                if len(proofs) != 1:
                                    raise ValueError("Removal has no unique span proof")
                                summary["entry_removed_inventory_occurrences"] += 1
                                summary["entry_removed_inventory_occurrences_by_kind"][
                                    proofs[0]["kind"]
                                ] += 1
                    expected_rows = batch[
                        "checked_rows" if kind == "rows" else "ledger_changed_rows"
                    ]
                    if row_count != expected_rows:
                        raise ValueError("Shard ledger row count differs")
            summary["batches"].append(record)
    if (
        artifact_cursor != len(artifacts)
        or row_cursor != 0
        or summary["checked_rows"] != summary["full_release_rows"]
    ):
        raise ValueError("Full-release row coverage differs")
    if (
        sum(summary["entry_removed_inventory_occurrences_by_kind"].values())
        != summary["entry_removed_inventory_occurrences"]
    ):
        raise ValueError("Typed removal classification count differs")
    summary["ledgers"] = {
        kind: {"sha256": digest(path), "bytes": path.stat().st_size}
        for kind, path in ledger_paths.items()
    }
    with (directory / "summary.json").open("x") as output:
        output.write(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", *counts)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "shard", "summarize"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--artifact-start", type=int)
    parser.add_argument("--artifact-limit", type=int, default=SHARD_SIZE)
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-limit", type=int)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            prepare(args.directory)
        elif args.action == "shard":
            if args.artifact_start is None:
                parser.error("shard requires --artifact-start")
            shard(
                args.directory,
                args.artifact_start,
                args.artifact_limit,
                args.row_start,
                args.row_limit,
            )
        else:
            summarize(args.directory)
    except subprocess.TimeoutExpired as error:
        parser.exit(
            1,
            f"Subprocess exceeded {error.timeout:g} seconds; partial outputs preserved.\n",
        )


if __name__ == "__main__":
    main()
