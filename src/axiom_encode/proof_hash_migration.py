"""Base-bound proof-import hash migration for repository schema changes."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from .constants import RULESPEC_ATOMIC_MODULE_ROOTS
from .harness.validator_pipeline import _parse_rulespec_target

REPORT_SCHEMA = "axiom-encode/proof-hash-cascade/v1"
_CHECKOUT_NAME_RE = re.compile(r"rulespec-[a-z]{2}")


@dataclass(frozen=True)
class HashOccurrence:
    importer_path: str
    target: str
    target_path: str
    line: int
    declared_hash: str
    base_target_hash: str | None
    current_target_hash: str | None
    status: Literal[
        "eligible",
        "preexisting_mismatch",
        "base_target_missing",
        "current_target_missing",
    ]
    start_index: int
    end_index: int
    scalar_style: str | None

    def public_record(self) -> dict[str, object]:
        record = asdict(self)
        for key in ("start_index", "end_index", "scalar_style"):
            record.pop(key)
        return record


@dataclass(frozen=True)
class CascadePlan:
    root: Path
    base_commit: str
    head_commit: str
    occurrences: tuple[HashOccurrence, ...]

    @property
    def eligible(self) -> tuple[HashOccurrence, ...]:
        return tuple(item for item in self.occurrences if item.status == "eligible")

    @property
    def ignored(self) -> tuple[HashOccurrence, ...]:
        return tuple(item for item in self.occurrences if item.status != "eligible")


def build_proof_hash_cascade_plan(root: Path, base_ref: str) -> CascadePlan:
    checkout = _validate_checkout(root)
    base_commit = _git_text(checkout, "rev-parse", "--verify", f"{base_ref}^{{commit}}")
    head_commit = _git_text(checkout, "rev-parse", "--verify", "HEAD^{commit}")
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "merge-base",
            "--is-ancestor",
            base_commit,
            head_commit,
        ],
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValueError(f"base ref {base_ref!r} is not an ancestor of HEAD")

    tracked = _git_text(checkout, "ls-files", "-z", strip=False).split("\0")
    module_paths = sorted(
        path
        for path in tracked
        if path
        and _is_atomic_module_path(Path(path))
        and not path.endswith(".test.yaml")
    )
    base_hashes: dict[str, str | None] = {}
    current_hashes: dict[str, str | None] = {}
    occurrences: list[HashOccurrence] = []
    for importer_path in module_paths:
        content = (checkout / importer_path).read_text(encoding="utf-8")
        for target_node, hash_node in _proof_import_scalar_nodes(content):
            target = target_node.value.strip()
            declared_hash = hash_node.value.strip()
            if declared_hash == "sha256:local":
                continue
            target_ref = _parse_rulespec_target(target)
            if target_ref is None:
                continue
            target_path = (
                Path(target_ref.prefix) / target_ref.relative_path
            ).as_posix()
            if target_path not in current_hashes:
                target_file = checkout / target_path
                current_hashes[target_path] = (
                    _sha256(target_file.read_bytes())
                    if target_file.is_file() and not target_file.is_symlink()
                    else None
                )
            if target_path not in base_hashes:
                base_bytes = _git_file(checkout, base_commit, target_path)
                base_hashes[target_path] = (
                    _sha256(base_bytes) if base_bytes is not None else None
                )
            base_hash = base_hashes[target_path]
            current_hash = current_hashes[target_path]
            expected_base = f"sha256:{base_hash}" if base_hash else None
            expected_current = f"sha256:{current_hash}" if current_hash else None
            if expected_current is None:
                status = "current_target_missing"
            elif expected_base is None:
                status = "base_target_missing"
            elif declared_hash == expected_base and declared_hash != expected_current:
                status = "eligible"
            elif declared_hash != expected_current:
                status = "preexisting_mismatch"
            else:
                continue
            occurrences.append(
                HashOccurrence(
                    importer_path=importer_path,
                    target=target,
                    target_path=target_path,
                    line=hash_node.start_mark.line + 1,
                    declared_hash=declared_hash,
                    base_target_hash=expected_base,
                    current_target_hash=expected_current,
                    status=status,
                    start_index=hash_node.start_mark.index,
                    end_index=hash_node.end_mark.index,
                    scalar_style=hash_node.style,
                )
            )
    return CascadePlan(
        root=checkout,
        base_commit=base_commit,
        head_commit=head_commit,
        occurrences=tuple(occurrences),
    )


def apply_proof_hash_cascade(plan: CascadePlan, report_path: Path) -> dict[str, object]:
    current_head = _git_text(plan.root, "rev-parse", "--verify", "HEAD^{commit}")
    if current_head != plan.head_commit:
        raise ValueError("proof hash cascade plan HEAD changed before apply")
    if _git_text(plan.root, "status", "--porcelain"):
        raise ValueError("proof hash cascade apply requires a clean Git checkout")
    report = _validate_report_path(plan.root, report_path)
    grouped: dict[str, list[HashOccurrence]] = defaultdict(list)
    for item in plan.eligible:
        grouped[item.importer_path].append(item)
    updated_files: dict[str, tuple[bytes, bytes, int]] = {}
    for relative_path, items in sorted(grouped.items()):
        path = plan.root / relative_path
        original = path.read_text(encoding="utf-8")
        updated = original
        for item in sorted(items, key=lambda value: value.start_index, reverse=True):
            if item.current_target_hash is None:
                raise ValueError(
                    f"eligible target unexpectedly has no current hash: {item.target_path}"
                )
            replacement = _render_scalar(item.current_target_hash, item.scalar_style)
            updated = (
                updated[: item.start_index] + replacement + updated[item.end_index :]
            )
        if updated == original:
            raise ValueError(
                f"eligible proof hash cascade produced no change: {relative_path}"
            )
        updated_files[relative_path] = (
            original.encode("utf-8"),
            updated.encode("utf-8"),
            len(items),
        )

    applied_files: list[dict[str, object]] = []
    for relative_path, (original, updated, replacement_count) in sorted(
        updated_files.items()
    ):
        path = plan.root / relative_path
        path.write_bytes(updated)
        applied_files.append(
            {
                "path": relative_path,
                "before_sha256": _sha256(original),
                "after_sha256": _sha256(updated),
                "replacements": replacement_count,
            }
        )

    payload: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "base_commit": plan.base_commit,
        "head_commit_before_apply": plan.head_commit,
        "eligible_replacements": len(plan.eligible),
        "ignored_mismatches": len(plan.ignored),
        "applied_files": applied_files,
        "replacements": [item.public_record() for item in plan.eligible],
        "ignored": [item.public_record() for item in plan.ignored],
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def render_proof_hash_cascade_plan(plan: CascadePlan, *, as_json: bool) -> str:
    payload = {
        "schema": REPORT_SCHEMA,
        "base_commit": plan.base_commit,
        "head_commit": plan.head_commit,
        "eligible_replacements": len(plan.eligible),
        "ignored_mismatches": len(plan.ignored),
        "eligible": [item.public_record() for item in plan.eligible],
        "ignored": [item.public_record() for item in plan.ignored],
    }
    if as_json:
        return json.dumps(payload, indent=2, sort_keys=True)
    lines = [
        f"base_commit={plan.base_commit}",
        f"head_commit={plan.head_commit}",
        f"eligible_replacements={len(plan.eligible)}",
        f"ignored_mismatches={len(plan.ignored)}",
    ]
    lines.extend(
        f"eligible {item.importer_path}:{item.line} {item.target_path}"
        for item in plan.eligible
    )
    lines.extend(
        f"ignored[{item.status}] {item.importer_path}:{item.line} {item.target_path}"
        for item in plan.ignored
    )
    return "\n".join(lines)


def _validate_checkout(root: Path) -> Path:
    checkout = Path(root).expanduser().resolve(strict=True)
    if not checkout.is_dir() or not _CHECKOUT_NAME_RE.fullmatch(checkout.name):
        raise ValueError("root must be a canonical rulespec-<country> checkout")
    inside = _git_text(checkout, "rev-parse", "--is-inside-work-tree")
    if inside != "true":
        raise ValueError("root must be a Git worktree")
    top_level = Path(_git_text(checkout, "rev-parse", "--show-toplevel")).resolve(
        strict=True
    )
    if top_level != checkout:
        raise ValueError("root must be the Git worktree root")
    return checkout


def _validate_report_path(root: Path, report_path: Path) -> Path:
    raw = report_path if report_path.is_absolute() else root / report_path
    resolved_parent = raw.parent.resolve(strict=False)
    report = resolved_parent / raw.name
    allowed = root / ".axiom" / "migrations"
    try:
        report.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("report must be written under .axiom/migrations") from exc
    if report.suffix != ".json":
        raise ValueError("report path must end in .json")
    if report.is_symlink():
        raise ValueError("report path must not be a symlink")
    if report.exists() and not report.is_file():
        raise ValueError("report path must be a regular file")
    return report


def _is_atomic_module_path(path: Path) -> bool:
    parts = path.parts
    return (
        len(parts) >= 3
        and re.fullmatch(r"[a-z]{2}(?:-[a-z0-9-]+)*", parts[0]) is not None
        and parts[1] in RULESPEC_ATOMIC_MODULE_ROOTS
        and path.suffix == ".yaml"
    )


def _proof_import_scalar_nodes(content: str) -> list[tuple[ScalarNode, ScalarNode]]:
    root = yaml.compose(content)
    if root is None:
        return []
    found: list[tuple[ScalarNode, ScalarNode]] = []

    def visit(node: Node) -> None:
        if isinstance(node, MappingNode):
            fields = {
                key.value: value
                for key, value in node.value
                if isinstance(key, ScalarNode)
            }
            import_node = fields.get("import")
            if isinstance(import_node, MappingNode):
                import_fields = {
                    key.value: value
                    for key, value in import_node.value
                    if isinstance(key, ScalarNode)
                }
                target = import_fields.get("target")
                hash_node = import_fields.get("hash")
                if isinstance(target, ScalarNode) and isinstance(hash_node, ScalarNode):
                    found.append((target, hash_node))
            for key, value in node.value:
                visit(key)
                visit(value)
        elif isinstance(node, SequenceNode):
            for value in node.value:
                visit(value)

    visit(root)
    return found


def _render_scalar(value: str, style: str | None) -> str:
    if style is None:
        return value
    if style == "'":
        return "'" + value.replace("'", "''") + "'"
    if style == '"':
        return json.dumps(value)
    raise ValueError(f"unsupported proof hash scalar style: {style!r}")


def _git_text(root: Path, *args: str, strip: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip() if strip else completed.stdout


def _git_file(root: Path, commit: str, path: str) -> bytes | None:
    tree_entry = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "-z", commit, "--", path],
        check=False,
        capture_output=True,
    )
    if tree_entry.returncode != 0:
        detail = tree_entry.stderr.decode(errors="replace").strip()
        raise ValueError(f"git ls-tree failed for {commit}:{path}: {detail}")
    if not tree_entry.stdout:
        return None
    metadata, separator, entry_path = tree_entry.stdout.rstrip(b"\0").partition(b"\t")
    fields = metadata.split()
    if separator != b"\t" or len(fields) != 3:
        raise ValueError(f"unexpected git tree entry for {commit}:{path}")
    mode, object_type, object_id = fields
    if entry_path.decode(errors="surrogateescape") != path:
        raise ValueError(f"git tree entry path mismatch for {commit}:{path}")
    if mode == b"120000" or object_type != b"blob":
        raise ValueError(f"base target must be a regular Git blob: {commit}:{path}")
    blob = subprocess.run(
        ["git", "-C", str(root), "cat-file", "blob", object_id.decode("ascii")],
        check=False,
        capture_output=True,
    )
    if blob.returncode != 0:
        detail = blob.stderr.decode(errors="replace").strip()
        raise ValueError(f"git cat-file failed for {commit}:{path}: {detail}")
    return blob.stdout


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
