"""Audit every US manual/guidance/statute row of a signed corpus release.

Unlike the bounded locator sampler, this audit has no row or source-length
exclusions. Artifacts are read from a supplied corpus or downloaded from the
exact release commit, one at a time, and verified before any row is analyzed.
The JSONL ledgers are streamed once. The final JSON identifies the script,
production module, baseline module, signed release and every input artifact.

This is a numeric-recall differential. It makes no formula-branch, runtime,
or historical replay claim. Unchanged entry-mask inputs are proved equal by
AST equality of the remaining baseline cleanup; changed inputs execute both
public cleanup functions and the same typed inventory extractor.
"""

from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import sysconfig
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from axiom_encode.corpus_release import verify_release_object
from axiom_encode.harness import source_completeness as current
from axiom_encode.harness.validator_pipeline import (
    _numeric_profile_for_citation_path,
    extract_typed_numeric_inventory_occurrences_from_text,
)

DOCUMENT_CLASSES = {"manual", "guidance", "statute"}
LEGAL_PREFIX = re.compile(r"\bsubdivisions?[ \t]+", re.IGNORECASE)
LEGAL_LABEL = re.compile(
    r"(?:(?P<prefix>[A-Za-z])[ \t]*)?(?P<digits>\d+)"
    r"(?P<suffix>[A-Za-z]|[ \t]+[A-Za-z]\b)?"
    r"(?:\([A-Za-z]\))*(?!\w|[.,]\d)"
)
LEGAL_SEPARATOR = re.compile(
    r"[ \t]*(?:,[ \t]*(?:(?:and|or)[ \t]+)?|"
    r"(?:and|or|through|to)[ \t]+|[-–—][ \t]*)",
    re.IGNORECASE,
)
LEGAL_CLOSE = re.compile(
    r"[ \t]*(?:$|[,.;:](?!\d)|(?:hereof|herein|"
    r"for[ \t]+the[ \t]+same[ \t]+taxable[ \t]+year)\b"
    r"(?=[ \t]*(?:$|[,.;:](?!\d))))",
    re.IGNORECASE,
)
REFERENCE_CLOSE = re.compile(
    r"[ \t]*(?:$|[.;:](?!\d)|,[ \t]*(?:claim|use|"
    r"the[ \t]+(?:credit|tax|deduction|subtraction))\b|"
    r"(?:hereof|herein|for[ \t]+the[ \t]+same[ \t]+taxable[ \t]+year)\b"
    r"(?=[ \t]*(?:$|[,.;:](?!\d))))",
    re.IGNORECASE,
)
LEGAL_OBJECT = re.compile(
    r"[ \t]+of[ \t]+(?:(?:this|that|the)[ \t]+"
    r"(?:section|subsection|paragraph|subdivision|chapter|title|code|act|article)\b"
    r"|§[ \t]*\d+(?:[.-]\d+)*(?:\([A-Za-z0-9]+\))*)",
    re.IGNORECASE,
)
PREFIX_LABEL_PREDICATE = re.compile(
    r"[ \t]+(?:(?:is|are)[ \t]+(?:required|applicable|authorized)|"
    r"appl(?:y|ies)|provides?|authorizes?|governs?|requires?)\b(?!-)",
    re.IGNORECASE,
)


def complete_legal_tail(tail: str):
    """An object must close completely, rather than merely start legally."""
    if object_match := LEGAL_OBJECT.match(tail):
        closing = tail[object_match.end() :]
        if LEGAL_CLOSE.match(closing) or re.match(
            r"[ \t]+for[ \t]+taxpayers\b(?=[ \t]*(?:$|[,.;:](?!\d)))",
            closing,
            re.IGNORECASE,
        ):
            return True
    return bool(REFERENCE_CLOSE.match(tail))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_baseline(path: Path):
    name = "axiom_encode.harness._subdivision_audit_baseline"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load baseline module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def isolate_entry_masks(baseline_path: Path):
    """Prove the existing cleanup body is unchanged and extract its new prefix."""
    old_tree = ast.parse(baseline_path.read_text())
    new_tree = ast.parse(Path(current.__file__).read_text())
    existing = {ast.dump(node, include_attributes=False) for node in new_tree.body}
    for node in old_tree.body:
        if getattr(node, "name", None) == "authoritative_numeric_recall_text":
            continue
        if ast.dump(node, include_attributes=False) not in existing:
            raise ValueError("An existing definition outside recall cleanup changed")
    old_function, new_function = (
        next(
            node
            for node in tree.body
            if getattr(node, "name", None) == "authoritative_numeric_recall_text"
        )
        for tree in (old_tree, new_tree)
    )
    old_body = old_function.body[1:]
    new_body = new_function.body[1:]
    prefix_length = len(new_body) - len(old_body)
    if prefix_length < 1 or ast.dump(
        ast.Module(body=old_body, type_ignores=[]), include_attributes=False
    ) != ast.dump(
        ast.Module(body=new_body[prefix_length:], type_ignores=[]),
        include_attributes=False,
    ):
        raise ValueError("Baseline recall cleanup is not an exact unchanged suffix")
    prefix = copy.deepcopy(new_body[:prefix_length])
    wrapper = copy.deepcopy(new_function)
    wrapper.name = "_audit_entry_masks"
    wrapper.body = prefix + [
        ast.Return(value=ast.Name(id="source_text", ctx=ast.Load()))
    ]
    namespace = dict(vars(current))
    tree = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    exec(compile(tree, "<entry-mask-audit>", "exec"), namespace)
    return namespace[wrapper.name], [ast.unparse(statement) for statement in prefix]


def independent_identifier_context(source: str, start: int, end: int):
    """Classify spans with an independent locator grammar, retaining context."""
    if source[start:end] == "48" and re.match(
        r"[ \t]+contiguous[ \t]+states\b", source[end:], re.IGNORECASE
    ):
        prefix = source[:start]
        if not prefix.strip() or re.search(r"\bthe\s+$", prefix, re.IGNORECASE):
            return "bounded-geography", source[start : end + 80]
    for prefix in LEGAL_PREFIX.finditer(source, 0, start):
        cursor = prefix.end()
        labels = []
        while match := LEGAL_LABEL.match(source, cursor):
            labels.append(match)
            cursor = match.end()
            separator = LEGAL_SEPARATOR.match(source, cursor)
            if separator is None or not LEGAL_LABEL.match(source, separator.end()):
                break
            cursor = separator.end()
        # Complete the whole reference positively. A trailing letter can be a
        # unit, so a label carrying one never supplies an identifier proof.
        final_tail = source[labels[-1].end() :] if labels else ""
        complete = (
            bool(labels)
            and all(match.group("suffix") is None for match in labels)
            and complete_legal_tail(final_tail)
        )
        has_range = any(
            re.search(r"[-–—]|\b(?:to|through)\b", source[left.end() : right.start()])
            for left, right in zip(labels, labels[1:])
        )
        if has_range and not (
            prefix.group().strip().lower() == "subdivisions"
            and LEGAL_OBJECT.match(final_tail)
        ):
            complete = False
        for match in labels:
            if match.span("digits") != (start, end):
                continue
            if match.group("suffix") is not None:
                return None
            local_tail = source[match.end() :]
            comma_boundary = (
                match is labels[0]
                and prefix.group().strip().lower() == "subdivision"
                and bool(re.match(r"[ \t]*,", local_tail))
            )
            prefix_predicate = (
                len(labels) == 1
                and match.group("prefix") is not None
                and PREFIX_LABEL_PREDICATE.match(final_tail)
            )
            if complete or comma_boundary or prefix_predicate:
                return "legal-locator", source[prefix.start() : labels[-1].end() + 100]
    return None


def entry_removals(source: str, masked: str):
    if len(source) != len(masked):
        raise ValueError("Entry masks changed source offsets")
    changed = {
        index
        for index, (before, after) in enumerate(zip(source, masked, strict=True))
        if before != after
    }
    if any(not source[index].isdecimal() or masked[index] != " " for index in changed):
        raise ValueError("Entry masks changed nonnumeric source text")
    removals = []
    for match in re.finditer(r"\d+", source):
        span = set(range(match.start(), match.end()))
        if not span & changed:
            continue
        if not span <= changed:
            raise ValueError("Entry masks removed only part of a numeric token")
        classification = independent_identifier_context(source, *match.span())
        if classification is None:
            raise ValueError(
                "Unclassified removal: "
                + repr(source[max(0, match.start() - 80) : match.end() + 100])
            )
        kind, identifier = classification
        removals.append(
            {
                "start": match.start(),
                "end": match.end(),
                "raw": match.group(),
                "kind": kind,
                "identifier_context": identifier,
                "context": source[max(0, match.start() - 100) : match.end() + 120],
            }
        )
    return removals


def changed_recall_spans(before: str, after: str):
    """Require equal offsets and complete digit tokens changed only to spaces."""
    if len(before) != len(after):
        raise ValueError("Recall cleanup changed offsets")
    changed = {
        index
        for index, (left, right) in enumerate(zip(before, after, strict=True))
        if left != right
    }
    if any(not before[index].isdecimal() or after[index] != " " for index in changed):
        raise ValueError("Recall cleanup changed nonnumeric text")
    spans = []
    for match in re.finditer(r"\d+", before):
        positions = set(range(match.start(), match.end()))
        if positions & changed:
            if not positions <= changed:
                raise ValueError("Recall cleanup removed a partial numeric token")
            spans.append(
                {"start": match.start(), "end": match.end(), "raw": match.group()}
            )
    return spans


def project_recall_removals(baseline, source, citation, before, after, removals):
    """Associate original proofs with cleaned spans by exact counterfactuals.

    Existing cleanup may erase a statutory object such as a section citation.
    Blanking only one independently verified original digit span and running
    the baseline must erase exactly one concrete digit span in old recall.
    This proves provenance without reclassifying a now-incomplete reference
    or matching equal-valued occurrences by their numeric value.
    """
    targets = changed_recall_spans(before, after)
    required = {(item["start"], item["end"]): item for item in targets}
    proofs = {}
    for removal in removals:
        if not any(item["raw"] == removal["raw"] for item in targets):
            continue
        start, end = removal["start"], removal["end"]
        isolated = source[:start] + " " * (end - start) + source[end:]
        counterfactual = baseline.authoritative_numeric_recall_text(
            isolated, corpus_citation_path=citation
        )
        projected = changed_recall_spans(before, counterfactual)
        if not projected:
            continue
        if len(projected) != 1 or projected[0]["raw"] != removal["raw"]:
            raise ValueError(
                "An isolated identifier does not project to one recall token"
            )
        item = projected[0]
        key = item["start"], item["end"]
        if key not in required or key in proofs:
            raise ValueError(
                "Identifier projection is ambiguous or changes another token"
            )
        proofs[key] = item | {
            "original_start": start,
            "original_end": end,
            "kind": removal["kind"],
            "method": "Baseline with only the independently verified original span blanked equals old recall with only this exact span blanked.",
        }
        if len(proofs) == len(required):
            break
    if proofs.keys() != required.keys():
        raise ValueError(
            "Original identifier proofs do not explain every recall change"
        )
    reconstructed = before
    for start, end in sorted(proofs, reverse=True):
        reconstructed = (
            reconstructed[:start] + " " * (end - start) + reconstructed[end:]
        )
    if reconstructed != after:
        raise ValueError(
            "Projected identifiers do not reconstruct final recall exactly"
        )
    return [proofs[key] for key in sorted(proofs)]


def require_inventory_projections(removed, projections):
    witnessed = {(item["start"], item["end"], item["raw"]) for item in projections}
    for item in removed:
        if (item["start"], item["end"], item["raw"]) not in witnessed:
            raise ValueError(
                "Removed inventory occurrence lacks exact original-span proof"
            )


def inventory(text: str, citation: str):
    return tuple(
        item
        for item in extract_typed_numeric_inventory_occurrences_from_text(
            text, profile=_numeric_profile_for_citation_path(citation)
        )
        if not current._temporal_occurrence_is_formula_applicability_preface(item, text)
    )


def inventory_key(item):
    record = dataclasses.asdict(item)
    record.pop("start")
    record.pop("end")
    return json.dumps(record, sort_keys=True)


def inventory_delta(before, after, old_text: str, new_text: str):
    old_counts = Counter(inventory_key(item) for item in before)
    new_counts = Counter(inventory_key(item) for item in after)
    result = {"removed": [], "added": []}
    for label, items, counts, text in (
        ("removed", before, old_counts - new_counts, old_text),
        ("added", after, new_counts - old_counts, new_text),
    ):
        other = after if label == "removed" else before
        exact = Counter((inventory_key(item), item.start, item.end) for item in other)
        for item in items:
            key = inventory_key(item)
            exact_key = (key, item.start, item.end)
            if exact[exact_key]:
                exact[exact_key] -= 1
                continue
            if counts[key]:
                record = dataclasses.asdict(item)
                record["context"] = text[max(0, item.start - 100) : item.end + 120]
                result[label].append(record)
                counts[key] -= 1
        if any(counts.values()):
            raise ValueError("Cannot identify changed numeric occurrences")
    return result


def github_token() -> str:
    for variable in ("GH_TOKEN", "GITHUB_TOKEN"):
        if token := os.environ.get(variable):
            return token
    result = subprocess.run(
        ["gh", "auth", "token"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def fetch_artifact(artifact, *, corpus, repository, commit, scratch, token):
    if corpus is not None:
        path = corpus / artifact.path
        cleanup = False
    else:
        path = scratch / "artifact.jsonl"
        cleanup = True
        url = (
            "https://raw.githubusercontent.com/"
            + repository
            + "/"
            + commit
            + "/"
            + urllib.parse.quote(artifact.path, safe="/")
        )
        headers = {"User-Agent": "axiom-encode-subdivision-audit"}
        if token:
            headers["Authorization"] = "Bearer " + token
        request = urllib.request.Request(url, headers=headers)
        received = 0
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            path.open("xb") as out,
        ):
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > artifact.byte_count:
                    raise ValueError("Remote artifact exceeds signed byte count")
                out.write(chunk)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    if size != artifact.byte_count or digest.hexdigest() != artifact.sha256:
        raise ValueError("Corpus artifact differs from signed release bytes")
    return path, cleanup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--release-name", required=True)
    parser.add_argument("--content-sha256", required=True)
    parser.add_argument("--public-key", action="append", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--repository", default="TheAxiomFoundation/axiom-corpus")
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--artifact-start", type=int, default=0)
    parser.add_argument("--artifact-limit", type=int)
    args = parser.parse_args()
    payload = json.loads(args.release.read_bytes())
    release = verify_release_object(payload, public_key=args.public_key)
    if (release.name, release.content_sha256) != (
        args.release_name,
        args.content_sha256,
    ):
        raise ValueError("Signed release does not match the required pin")
    artifacts = tuple(
        artifact
        for artifact in release.artifacts
        if artifact.artifact_class == "provisions"
        and artifact.path.split("/")[4] in DOCUMENT_CLASSES
        and (
            artifact.path.split("/")[3] == "us"
            or artifact.path.split("/")[3].startswith("us-")
        )
    )
    full_release_files = len(artifacts)
    full_release_rows = sum(artifact.row_count or 0 for artifact in artifacts)
    if args.artifact_start < 0 or (
        args.artifact_limit is not None and args.artifact_limit < 1
    ):
        raise ValueError("Artifact selection must be nonnegative and nonempty")
    stop = (
        args.artifact_start + args.artifact_limit
        if args.artifact_limit is not None
        else len(artifacts)
    )
    artifacts = artifacts[args.artifact_start : stop]
    if not artifacts:
        raise ValueError("Artifact selection is empty")
    if sum(artifact.byte_count for artifact in artifacts) > 1024**3:
        raise ValueError("Selected corpus exceeds the one-GiB audit limit")
    # A new directory avoids overwriting any saved evidence from an earlier run.
    args.scratch.mkdir(parents=True, exist_ok=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline(args.baseline)
    masks, prefix = isolate_entry_masks(args.baseline)
    token = github_token() if args.corpus is None else ""
    started = time.monotonic()
    summary = {
        "status": "running",
        "release": release.name,
        "content_sha256": release.content_sha256,
        "selector_sha256": release.selector_sha256,
        "corpus_commit": release.git_commit,
        "release_object_sha256": sha256(args.release.read_bytes()),
        "signature_verified": True,
        "python_version": sys.version.split()[0],
        "gil_disabled_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "script_sha256": sha256(Path(__file__).read_bytes()),
        "baseline_module_sha256": sha256(args.baseline.read_bytes()),
        "current_module_sha256": sha256(Path(current.__file__).read_bytes()),
        "existing_definitions_unchanged": True,
        "baseline_cleanup_suffix_unchanged": True,
        "entry_mask_statements": prefix,
        "scope": "All rows in selected signed US manual, guidance and statute artifacts, including headings; no row or length exclusions. Numeric recall only; no formula-branch or replay claim.",
        "full_release_files": full_release_files,
        "full_release_rows": full_release_rows,
        "artifact_start": args.artifact_start,
        "artifact_end": args.artifact_start + len(artifacts),
        "declared_files": len(artifacts),
        "declared_rows": sum(artifact.row_count or 0 for artifact in artifacts),
        "declared_bytes": sum(artifact.byte_count for artifact in artifacts),
        "files": [],
        "checked_rows": 0,
        "entry_unchanged_rows": 0,
        "entry_changed_rows": 0,
        "recall_changed_rows": 0,
        "removed_entry_identifiers": {"legal-locator": 0, "bounded-geography": 0},
        "removed_inventory_occurrences": 0,
        "added_inventory_occurrences": 0,
        "errors": [],
    }
    try:
        with (
            args.output.with_suffix(".rows.jsonl").open("w") as rows_out,
            args.output.with_suffix(".removals.jsonl").open("w") as removals_out,
        ):
            for index, artifact in enumerate(artifacts, 1):
                path = None
                cleanup = args.corpus is None
                try:
                    path, cleanup = fetch_artifact(
                        artifact,
                        corpus=args.corpus,
                        repository=args.repository,
                        commit=release.git_commit,
                        scratch=args.scratch,
                        token=token,
                    )
                    count = 0
                    with path.open("rb") as stream:
                        for count, raw in enumerate(stream, 1):
                            row = json.loads(raw)
                            source = "\n\n".join(
                                part
                                for part in (row.get("heading"), row.get("body"))
                                if isinstance(part, str) and part
                            )
                            citation = row["citation_path"]
                            record = {
                                "file": artifact.path,
                                "line": count,
                                "citation": citation,
                                "row_sha256": sha256(raw),
                                "source_sha256": sha256(source.encode()),
                            }
                            summary["checked_rows"] += 1
                            masked = masks(source)
                            removals = entry_removals(source, masked)
                            if not removals:
                                summary["entry_unchanged_rows"] += 1
                                record["unchanged_by_exact_cleanup_suffix"] = True
                            else:
                                summary["entry_changed_rows"] += 1
                                for removal in removals:
                                    summary["removed_entry_identifiers"][
                                        removal["kind"]
                                    ] += 1
                                old = baseline.authoritative_numeric_recall_text(
                                    source, corpus_citation_path=citation
                                )
                                new = current.authoritative_numeric_recall_text(
                                    source, corpus_citation_path=citation
                                )
                                record["recall_before_sha256"] = sha256(old.encode())
                                record["recall_after_sha256"] = sha256(new.encode())
                                delta = {"removed": [], "added": []}
                                if old != new:
                                    summary["recall_changed_rows"] += 1
                                    delta = inventory_delta(
                                        inventory(old, citation),
                                        inventory(new, citation),
                                        old,
                                        new,
                                    )
                                projections = project_recall_removals(
                                    baseline, source, citation, old, new, removals
                                )
                                require_inventory_projections(
                                    delta["removed"], projections
                                )
                                if delta["added"]:
                                    raise ValueError(
                                        "Numeric recall obligations were added"
                                    )
                                summary["removed_inventory_occurrences"] += len(
                                    delta["removed"]
                                )
                                summary["added_inventory_occurrences"] += len(
                                    delta["added"]
                                )
                                removals_out.write(
                                    json.dumps(
                                        record
                                        | {
                                            "entry_removals": removals,
                                            "inventory_delta": delta,
                                            "recall_projections": projections,
                                        }
                                    )
                                    + "\n"
                                )
                            rows_out.write(json.dumps(record) + "\n")
                    if count != artifact.row_count:
                        raise ValueError(
                            "Corpus artifact row count differs from signed release"
                        )
                    summary["files"].append(dataclasses.asdict(artifact))
                    print(
                        f"{index}/{len(artifacts)} files; {summary['checked_rows']} rows",
                        flush=True,
                    )
                finally:
                    if cleanup:
                        (path or args.scratch / "artifact.jsonl").unlink(
                            missing_ok=True
                        )
        summary["status"] = "passed"
    except Exception as error:
        summary["status"] = "failed"
        summary["errors"].append(str(error))
        raise
    finally:
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        if args.corpus is None:
            args.scratch.rmdir()
        print(
            json.dumps(
                {
                    key: summary[key]
                    for key in (
                        "status",
                        "checked_rows",
                        "removed_inventory_occurrences",
                        "errors",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
