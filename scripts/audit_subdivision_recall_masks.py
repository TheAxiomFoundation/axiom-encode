"""Audit every US manual/guidance/statute row of a signed corpus release.

Unlike the bounded locator sampler, this audit has no row or source-length
exclusions. Artifacts are read from a supplied corpus or downloaded from the
exact release commit, one at a time, and verified before any row is analyzed.
The JSONL ledgers are streamed once. The final JSON identifies the script,
production module, baseline module, signed release and every input artifact.

This is a numeric-recall differential. It makes no formula-branch, runtime,
or historical replay claim. Every row executes the original, guarded baseline
and current public cleanup. The unchanged cleanup body and one exact guarded
marker definition are checked by AST; independent span proofs explain every
removed identifier and restored glued-unit quantity.
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
AUDIT_TYPOGRAPHY = str.maketrans(
    {
        "\u00a0": " ",
        "\u202f": " ",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "−": "-",
        "–": "-",
        "—": "-",
    }
)
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
    r",[ \t]*the[ \t]+following[ \t]+applies[ \t]*:|"
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
GLUED_QUANTITY_UNIT = re.compile(
    r"(?<!\w)(?P<digits>[1-9]\d?)"
    r"(?P<unit>days?|weeks?|months?|years?|hours?|percent|members?|persons?|"
    r"people|dollars?|cents?|per[ \t\u00a0\u202f]+cent)\b",
    re.IGNORECASE,
)


def complete_legal_tail(tail: str):
    """An object must close completely, rather than merely start legally."""
    tail = tail.translate(AUDIT_TYPOGRAPHY)
    if object_match := LEGAL_OBJECT.match(tail):
        closing = tail[object_match.end() :]
        if LEGAL_CLOSE.match(closing) or re.match(
            r"[ \t]+for[ \t]+taxpayers\b(?=[ \t]*(?:$|[,.;:](?!\d)))",
            closing,
            re.IGNORECASE,
        ):
            return True
    if REFERENCE_CLOSE.match(tail):
        return True
    # A finite definition of a quoted, nonnumeric term starts a new clause.
    # Its comma therefore cannot be joining another member of a quantity list.
    definition = re.match(
        r"[ \t]*,[ \t]+(?P<term>\"[^\"\d\r\n]+\"|“[^”\d\r\n]+”|"
        r"'[^'\d\r\n]+'|‘[^’\d\r\n]+’)[ \t]+"
        r"(?:means|includes|refers[ \t]+to)\b",
        tail,
        re.IGNORECASE,
    )
    return bool(definition and re.search(r"[A-Za-z]", definition.group("term")))


def independent_quantity_clause(tail: str):
    """A complete finite clause proves the preceding comma is a boundary.

    Parse a numeric subject separately from its predicate. Shared comma and
    conjunction lists cannot prove that a preceding number is a locator.
    Only a local numeric range or comparison belongs to one numeric subject.
    """
    tail = tail.translate(AUDIT_TYPOGRAPHY)
    opener = re.match(
        r"[ \t]*,[ \t]+\d+(?:\.\d+)?"
        r"(?:[ \t]*(?:[-–—]|\bto\b|\bthrough\b)[ \t]*\d+(?:\.\d+)?)?"
        r"(?:[ \t]+or[ \t]+(?:more|fewer|less|greater)\b)?[ \t]+",
        tail,
        re.IGNORECASE,
    )
    if opener is None:
        return False
    closing = re.search(r"[.;:](?!\d)|$", tail[opener.end() :])
    if closing is None:
        return False
    clause = tail[opener.end() : opener.end() + closing.start()].strip()
    if not clause or re.search(r"\b(?:and|or)\b", clause, re.IGNORECASE):
        return False
    if re.search(r"[^\w'’% \t-]", clause):
        return False
    words = clause.lower().split()
    for index, word in enumerate(words):
        # A finite predicate can follow a bare numeric subject ("3 qualifies").
        # Shared coordination was rejected above, so it cannot borrow this proof.
        if word in {"qualify", "qualifies"} and index == len(words) - 1:
            return True
        if word in {"must", "shall", "may", "can", "is", "are"}:
            complement = words[index + 1 :]
            if complement and all(
                re.fullmatch(r"[a-z][\w'’-]*", part) for part in complement
            ):
                return True
    return False


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_baseline(path: Path, *, guarded: bool = False):
    name = "axiom_encode.harness._subdivision_audit_" + (
        "guarded_baseline" if guarded else "baseline"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load baseline module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def exact_glued_quantity_guard_assignment(old, new, unit_word):
    """Permit only one known negative lookahead in the marker's pattern."""
    if not (
        isinstance(old, ast.Assign)
        and isinstance(new, ast.Assign)
        and isinstance(old.value, ast.Call)
        and isinstance(new.value, ast.Call)
        and len(old.value.args) == len(new.value.args) == 1
        and isinstance(old.value.args[0], ast.Constant)
        and isinstance(old.value.args[0].value, str)
        and isinstance(new.value.args[0], ast.JoinedStr)
    ):
        return False
    restored = copy.deepcopy(new)
    restored.value.args[0] = old.value.args[0]
    if ast.dump(restored, include_attributes=False) != ast.dump(
        old, include_attributes=False
    ):
        return False
    pieces = []
    substitutions = 0
    for part in new.value.args[0].values:
        if isinstance(part, ast.Constant) and isinstance(part.value, str):
            pieces.append(part.value)
        elif (
            isinstance(part, ast.FormattedValue)
            and isinstance(part.value, ast.Name)
            and part.value.id == "_RECALL_UNIT_WORD"
            and part.conversion == -1
            and part.format_spec is None
        ):
            pieces.append(unit_word)
            substitutions += 1
        else:
            return False
    anchor = "(?=[A-ZÄÖÜ]"
    original = old.value.args[0].value
    guard = "(?!" + unit_word + r"\b|(?i:per[ \t\u00a0\u202f]+cent)\b)"
    return (
        substitutions == 1
        and original.count(anchor) == 1
        and "".join(pieces) == original.replace(anchor, guard + anchor, 1)
    )


def isolate_entry_masks(baseline_path: Path):
    """Prove the existing cleanup body is unchanged and extract its new prefix."""
    old_tree = ast.parse(baseline_path.read_text())
    new_tree = ast.parse(Path(current.__file__).read_text())
    existing = {ast.dump(node, include_attributes=False) for node in new_tree.body}

    def assignment(tree, name):
        found = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ]
        if len(found) != 1:
            raise ValueError("Expected one marker or unit-word assignment")
        return found[0]

    unit_word = assignment(old_tree, "_RECALL_UNIT_WORD").value.value
    old_marker = assignment(old_tree, "_GLUED_SENTENCE_MARKER")
    new_marker = assignment(new_tree, "_GLUED_SENTENCE_MARKER")
    if not exact_glued_quantity_guard_assignment(old_marker, new_marker, unit_word):
        raise ValueError("Glued quantity guard is not the exact permitted extension")
    for node in old_tree.body:
        if getattr(node, "name", None) == "authoritative_numeric_recall_text":
            continue
        if ast.dump(node, include_attributes=False) not in existing:
            if node is old_marker:
                continue
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
    original_source = source
    # Scan a one-for-one copy; proof contexts keep the original source bytes.
    source = source.translate(AUDIT_TYPOGRAPHY)
    if source[start:end] == "48" and re.match(
        r"[ \t]+contiguous[ \t]+states\b", source[end:], re.IGNORECASE
    ):
        prefix = source[:start]
        if not prefix.strip() or re.search(r"\bthe\s+$", prefix, re.IGNORECASE):
            return "bounded-geography", original_source[start : end + 80]
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
            re.search(
                r"[-–—]|\b(?:to|through)\b",
                source[left.end() : right.start()],
                re.IGNORECASE,
            )
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
                and independent_quantity_clause(local_tail)
            )
            prefix_predicate = (
                len(labels) == 1
                and match.group("prefix") is not None
                and PREFIX_LABEL_PREDICATE.match(final_tail)
            )
            if complete or comma_boundary or prefix_predicate:
                return (
                    "legal-locator",
                    original_source[prefix.start() : labels[-1].end() + 100],
                )
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


def prove_glued_quantity_restorations(original, guarded, original_marker):
    """Deleting only independently recognized glued-unit digits restores old text."""
    proofs = []
    for match in GLUED_QUANTITY_UNIT.finditer(guarded):
        marker = original_marker.match(guarded, match.start())
        if marker is None or marker.span("label") != match.span("digits"):
            continue
        proofs.append(
            {
                "start": match.start("digits"),
                "end": match.end("digits"),
                "raw": match.group("digits"),
                "unit_end": match.end("unit"),
                "unit_context": match.group(),
                "kind": "operative-glued-unit",
            }
        )
    restored = guarded
    for proof in reversed(proofs):
        restored = restored[: proof["start"]] + restored[proof["end"] :]
    if restored != original:
        raise ValueError("Guard changes lack an independent glued-unit deletion proof")
    return proofs


def require_quantity_addition_proofs(added, proofs):
    for item in added:
        if not any(
            item["start"] == proof["start"]
            and proof["end"] <= item["end"] <= proof["unit_end"]
            and item["raw"].startswith(proof["raw"])
            for proof in proofs
        ):
            raise ValueError("Added inventory lacks an exact retained quantity proof")


def require_delta_composition(full, guard, entry):
    """Compare signed typed multisets, including equal-valued cancellations."""

    def counts(delta):
        result = Counter()
        for label, sign in (("added", 1), ("removed", -1)):
            for item in delta[label]:
                key = json.dumps(
                    {
                        name: value
                        for name, value in item.items()
                        if name not in {"start", "end", "context"}
                    },
                    sort_keys=True,
                )
                result[key] += sign
        return result

    expected = counts(guard)
    expected.update(counts(entry))
    if counts(full) != expected:
        raise ValueError("Final inventory differs from independently proved deltas")


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


def select_row_ranges(artifacts, *, artifact_start, row_start=0, row_limit=None):
    """Select explicit half-open bounds without changing signed artifact totals."""
    if row_start < 0 or (row_limit is not None and row_limit < 1):
        raise ValueError("Invalid row selection bounds")
    sliced = row_start != 0 or row_limit is not None
    if sliced and len(artifacts) != 1:
        raise ValueError("A row selection requires exactly one artifact")
    ranges = []
    for index, artifact in enumerate(artifacts, artifact_start):
        total = artifact.row_count
        if not isinstance(total, int) or total < 0:
            raise ValueError("Signed artifact lacks a valid row count")
        start = row_start if sliced else 0
        end = start + row_limit if row_limit is not None else total
        if sliced and not (0 <= start < end <= total):
            raise ValueError("Invalid or empty row selection bounds")
        ranges.append(
            {
                "artifact_index": index,
                "path": artifact.path,
                "row_start": start,
                "row_end": end,
                "artifact_row_count": total,
                "selected_row_count": end - start,
            }
        )
    return ranges


def verify_artifact_row_count(path, expected):
    """Count the complete signed file, including rows outside a selected slice."""
    with path.open("rb") as stream:
        physical = sum(1 for _ in stream)
    if physical != expected:
        raise ValueError("Corpus artifact row count differs from signed release")
    return physical


def validate_row_range_coverage(ranges, artifact_row_counts):
    """Require ordered slices to cover every signed artifact row exactly once."""
    artifact_index = 0
    cursor = 0
    for selected in ranges:
        if (
            artifact_index >= len(artifact_row_counts)
            or selected["artifact_index"] != artifact_index
            or selected["row_start"] != cursor
        ):
            raise ValueError("Row coverage has a gap, overlap or out-of-order artifact")
        total = artifact_row_counts[artifact_index]
        end = selected["row_end"]
        if (
            selected["artifact_row_count"] != total
            or not cursor <= end <= total
            or (end == cursor and total != 0)
            or selected["selected_row_count"] != end - cursor
        ):
            raise ValueError("Row coverage has invalid bounds or declared counts")
        cursor = end
        if cursor == total:
            artifact_index += 1
            cursor = 0
    if artifact_index != len(artifact_row_counts):
        raise ValueError("Row coverage is incomplete")


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
    parser.add_argument("--row-start", type=int, default=0)
    parser.add_argument("--row-limit", type=int)
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
    row_ranges = select_row_ranges(
        artifacts,
        artifact_start=args.artifact_start,
        row_start=args.row_start,
        row_limit=args.row_limit,
    )
    if sum(artifact.byte_count for artifact in artifacts) > 1024**3:
        raise ValueError("Selected corpus exceeds the one-GiB audit limit")
    # A new directory avoids overwriting any saved evidence from an earlier run.
    args.scratch.mkdir(parents=True, exist_ok=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline(args.baseline)
    masks, prefix = isolate_entry_masks(args.baseline)
    guard = "(?!" + baseline._RECALL_UNIT_WORD + r"\b|(?i:per[ \t\u00a0\u202f]+cent)\b)"
    expected_marker = baseline._GLUED_SENTENCE_MARKER.pattern.replace(
        "(?=[A-ZÄÖÜ]", guard + "(?=[A-ZÄÖÜ]", 1
    )
    if (
        current._GLUED_SENTENCE_MARKER.pattern != expected_marker
        or current._GLUED_SENTENCE_MARKER.flags != baseline._GLUED_SENTENCE_MARKER.flags
    ):
        raise ValueError("Runtime marker differs from the verified guard extension")
    guarded_baseline = load_baseline(args.baseline, guarded=True)
    guarded_baseline._GLUED_SENTENCE_MARKER = current._GLUED_SENTENCE_MARKER
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
        "existing_definitions_unchanged": False,
        "other_existing_definitions_unchanged": True,
        "baseline_cleanup_suffix_unchanged": True,
        "bounded_glued_quantity_guard_verified": True,
        "all_rows_runtime_compared": True,
        "entry_mask_statements": prefix,
        "scope": "Every row in the declared signed US manual, guidance and statute row ranges, including headings; whole artifact bytes and physical row counts are verified. Numeric recall only; no formula-branch or replay claim.",
        "full_release_files": full_release_files,
        "full_release_rows": full_release_rows,
        "artifact_start": args.artifact_start,
        "artifact_end": args.artifact_start + len(artifacts),
        "declared_files": len(artifacts),
        "declared_rows": sum(selected["selected_row_count"] for selected in row_ranges),
        "row_ranges": row_ranges,
        "full_artifact_row_counts_verified": False,
        "declared_bytes": sum(artifact.byte_count for artifact in artifacts),
        "files": [],
        "checked_rows": 0,
        "entry_unchanged_rows": 0,
        "entry_changed_rows": 0,
        "recall_changed_rows": 0,
        "glued_quantity_guard_changed_rows": 0,
        "restored_glued_quantity_spans": 0,
        "guard_added_inventory_occurrences": 0,
        "ledger_changed_rows": 0,
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
                    selected = row_ranges[index - 1]
                    selected["physical_row_count"] = verify_artifact_row_count(
                        path, artifact.row_count
                    )
                    checked_in_slice = 0
                    with path.open("rb") as stream:
                        for count, raw in enumerate(stream, 1):
                            if count <= selected["row_start"]:
                                continue
                            if count > selected["row_end"]:
                                break
                            checked_in_slice += 1
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
                            old = baseline.authoritative_numeric_recall_text(
                                source, corpus_citation_path=citation
                            )
                            guarded = (
                                guarded_baseline.authoritative_numeric_recall_text(
                                    source, corpus_citation_path=citation
                                )
                            )
                            new = current.authoritative_numeric_recall_text(
                                source, corpus_citation_path=citation
                            )
                            quantity_proofs = prove_glued_quantity_restorations(
                                old, guarded, baseline._GLUED_SENTENCE_MARKER
                            )
                            record["recall_before_sha256"] = sha256(old.encode())
                            record["guarded_recall_sha256"] = sha256(guarded.encode())
                            record["recall_after_sha256"] = sha256(new.encode())
                            record["runtime_compared"] = True
                            if not removals:
                                summary["entry_unchanged_rows"] += 1
                            else:
                                summary["entry_changed_rows"] += 1
                                for removal in removals:
                                    summary["removed_entry_identifiers"][
                                        removal["kind"]
                                    ] += 1
                            inventories = {}

                            def differential(before, after):
                                if before == after:
                                    return {"removed": [], "added": []}
                                for text in (before, after):
                                    if text not in inventories:
                                        inventories[text] = inventory(text, citation)
                                return inventory_delta(
                                    inventories[before],
                                    inventories[after],
                                    before,
                                    after,
                                )

                            guard_delta = differential(old, guarded)
                            if guard_delta["removed"]:
                                raise ValueError(
                                    "Glued quantity guard removed inventory"
                                )
                            require_quantity_addition_proofs(
                                guard_delta["added"], quantity_proofs
                            )
                            entry_delta = differential(guarded, new)
                            projections = []
                            if guarded != new:
                                projections = project_recall_removals(
                                    guarded_baseline,
                                    source,
                                    citation,
                                    guarded,
                                    new,
                                    removals,
                                )
                            require_inventory_projections(
                                entry_delta["removed"], projections
                            )
                            if entry_delta["added"]:
                                raise ValueError("Entry masks added inventory")
                            delta = differential(old, new)
                            require_delta_composition(delta, guard_delta, entry_delta)
                            if old != new:
                                summary["recall_changed_rows"] += 1
                            if old != guarded:
                                summary["glued_quantity_guard_changed_rows"] += 1
                            summary["restored_glued_quantity_spans"] += len(
                                quantity_proofs
                            )
                            summary["guard_added_inventory_occurrences"] += len(
                                guard_delta["added"]
                            )
                            summary["removed_inventory_occurrences"] += len(
                                delta["removed"]
                            )
                            summary["added_inventory_occurrences"] += len(
                                delta["added"]
                            )
                            if removals or quantity_proofs:
                                summary["ledger_changed_rows"] += 1
                                removals_out.write(
                                    json.dumps(
                                        record
                                        | {
                                            "entry_removals": removals,
                                            "inventory_delta": delta,
                                            "entry_inventory_delta": entry_delta,
                                            "guard_inventory_delta": guard_delta,
                                            "recall_projections": projections,
                                            "glued_quantity_proofs": quantity_proofs,
                                        }
                                    )
                                    + "\n"
                                )
                            rows_out.write(json.dumps(record) + "\n")
                    if checked_in_slice != selected["selected_row_count"]:
                        raise ValueError(
                            "Checked row selection differs from declared bounds"
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
        summary["full_artifact_row_counts_verified"] = True
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
