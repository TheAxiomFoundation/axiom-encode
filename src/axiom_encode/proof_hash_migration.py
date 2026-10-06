"""Base-bound proof-import hash migration for repository schema changes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode
from yaml.tokens import AliasToken, AnchorToken, KeyToken

from .constants import RULESPEC_ATOMIC_MODULE_ROOTS
from .harness.validator_pipeline import _parse_rulespec_target
from .repo_routing import is_composition_policy_repo_root, jurisdiction_country

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
    replacement_hash: str | None
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
class _ProofImportScan:
    importer_path: str
    target: str
    target_path: str
    line: int
    declared_hash: str
    base_target_hash: str
    current_target_hash: str
    start_index: int
    end_index: int
    scalar_style: str | None

    def occurrence(
        self,
        *,
        status: Literal["eligible", "preexisting_mismatch"],
        replacement: str | None,
    ) -> HashOccurrence:
        return HashOccurrence(
            importer_path=self.importer_path,
            target=self.target,
            target_path=self.target_path,
            line=self.line,
            declared_hash=self.declared_hash,
            base_target_hash=self.base_target_hash,
            current_target_hash=self.current_target_hash,
            replacement_hash=replacement,
            status=status,
            start_index=self.start_index,
            end_index=self.end_index,
            scalar_style=self.scalar_style,
        )


@dataclass(frozen=True)
class CascadePlan:
    root: Path
    base_commit: str
    head_commit: str
    occurrences: tuple[HashOccurrence, ...]
    importer_sha256: tuple[tuple[str, str], ...]

    @property
    def eligible(self) -> tuple[HashOccurrence, ...]:
        return tuple(item for item in self.occurrences if item.status == "eligible")

    @property
    def ignored(self) -> tuple[HashOccurrence, ...]:
        return tuple(item for item in self.occurrences if item.status != "eligible")


def build_proof_hash_cascade_plan(
    root: Path,
    base_ref: str,
    *,
    _expected_worktree_bytes: dict[str, bytes] | None = None,
) -> CascadePlan:
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
    country = checkout.name.removeprefix("rulespec-")
    module_paths = sorted(
        path
        for path in tracked
        if path
        and _is_atomic_module_path(Path(path))
        and jurisdiction_country(Path(path).parts[0]) == country
        and not path.endswith(".test.yaml")
    )
    head_entries = _git_tree_entries(checkout, head_commit)
    head_blob_bytes = _git_blob_batch(
        checkout,
        {
            object_id
            for path, (mode, object_type, object_id) in head_entries.items()
            if _is_atomic_module_path(Path(path))
            and jurisdiction_country(Path(path).parts[0]) == country
            and mode.startswith("100")
            and object_type == "blob"
        },
    )
    base_target_bytes: dict[str, bytes | None] = {}
    current_target_bytes: dict[str, bytes | None] = {}
    importer_hashes: dict[str, str] = {}
    module_bytes: dict[str, bytes] = {}
    candidates: list[_ProofImportScan] = []
    current_non_base: list[_ProofImportScan] = []
    ignored: list[HashOccurrence] = []
    for importer_path in module_paths:
        importer_file = checkout / importer_path
        importer_bytes = _read_regular_worktree_file(
            checkout, importer_file, label="importer"
        )
        head_importer = head_entries.get(importer_path)
        if head_importer is None:
            raise ValueError(f"tracked importer is missing from HEAD: {importer_path}")
        head_mode, head_type, _head_object = head_importer
        if not head_mode.startswith("100") or head_type != "blob":
            raise ValueError(
                f"tracked importer is not a regular HEAD blob: {importer_path}"
            )
        expected_importer_bytes = (_expected_worktree_bytes or {}).get(
            importer_path, head_blob_bytes.get(_head_object)
        )
        if importer_bytes != expected_importer_bytes:
            raise ValueError(
                f"importer worktree bytes differ from HEAD: {importer_path}"
            )
        importer_hashes[importer_path] = _sha256(importer_bytes)
        module_bytes[importer_path] = importer_bytes
        content = importer_bytes.decode("utf-8")
        for target_node, hash_node in _proof_import_scalar_nodes(content):
            target = target_node.value.strip()
            declared_hash = hash_node.value.strip()
            if declared_hash == "sha256:local":
                continue
            target_ref = _parse_rulespec_target(target)
            if target_ref is None:
                continue
            if jurisdiction_country(target_ref.prefix) != country:
                raise ValueError(
                    f"proof import target {target!r} does not belong to {checkout.name}"
                )
            target_path = (
                Path(target_ref.prefix) / target_ref.relative_path
            ).as_posix()
            if target_path not in current_target_bytes:
                target_file = checkout / target_path
                current_target_bytes[target_path] = (
                    _read_optional_regular_worktree_file(
                        checkout, target_file, label="proof import target"
                    )
                )
                if current_target_bytes[target_path] is not None:
                    head_target = head_entries.get(target_path)
                    if head_target is None:
                        raise ValueError(
                            f"proof import target is missing from HEAD: {target_path}"
                        )
                    head_mode, head_type, _head_object = head_target
                    if not head_mode.startswith("100") or head_type != "blob":
                        raise ValueError(
                            "proof import target is not a regular HEAD blob: "
                            f"{target_path}"
                        )
                    expected_target_bytes = (_expected_worktree_bytes or {}).get(
                        target_path, head_blob_bytes.get(_head_object)
                    )
                    if current_target_bytes[target_path] != expected_target_bytes:
                        raise ValueError(
                            "proof import target worktree bytes differ from HEAD: "
                            f"{target_path}"
                        )
            if target_path not in base_target_bytes:
                base_target_bytes[target_path] = _git_file(
                    checkout, base_commit, target_path
                )
            base_bytes = base_target_bytes[target_path]
            current_bytes = current_target_bytes[target_path]
            expected_base = (
                f"sha256:{_sha256(base_bytes)}" if base_bytes is not None else None
            )
            expected_current = (
                f"sha256:{_sha256(current_bytes)}"
                if current_bytes is not None
                else None
            )
            if expected_current is None:
                status = "current_target_missing"
            elif expected_base is None:
                status = "base_target_missing"
            else:
                scan = _ProofImportScan(
                    importer_path=importer_path,
                    target=target,
                    target_path=target_path,
                    line=hash_node.start_mark.line + 1,
                    declared_hash=declared_hash,
                    base_target_hash=expected_base,
                    current_target_hash=expected_current,
                    start_index=hash_node.start_mark.index,
                    end_index=hash_node.end_mark.index,
                    scalar_style=hash_node.style,
                )
                if declared_hash == expected_base:
                    candidates.append(scan)
                elif declared_hash == expected_current:
                    current_non_base.append(scan)
                else:
                    ignored.append(
                        scan.occurrence(status="preexisting_mismatch", replacement=None)
                    )
                continue
            ignored.append(
                HashOccurrence(
                    importer_path=importer_path,
                    target=target,
                    target_path=target_path,
                    line=hash_node.start_mark.line + 1,
                    declared_hash=declared_hash,
                    base_target_hash=expected_base,
                    current_target_hash=expected_current,
                    replacement_hash=None,
                    status=status,
                    start_index=hash_node.start_mark.index,
                    end_index=hash_node.end_mark.index,
                    scalar_style=hash_node.style,
                )
            )

    final_module_bytes = _resolve_base_bound_cascade(
        module_bytes=module_bytes,
        candidates=candidates,
        current_target_bytes=current_target_bytes,
    )
    eligible: list[HashOccurrence] = []
    for scan in candidates:
        replacement = _target_hash_for_virtual_files(
            scan.target_path,
            virtual_module_bytes=final_module_bytes,
            current_target_bytes=current_target_bytes,
        )
        if replacement != scan.declared_hash:
            eligible.append(scan.occurrence(status="eligible", replacement=replacement))
    for scan in current_non_base:
        final_target_hash = _target_hash_for_virtual_files(
            scan.target_path,
            virtual_module_bytes=final_module_bytes,
            current_target_bytes=current_target_bytes,
        )
        if final_target_hash != scan.current_target_hash:
            raise ValueError(
                "base-bound cascade would stale a non-base-authorized proof import: "
                f"{scan.importer_path}:{scan.line} -> {scan.target_path}"
            )
    return CascadePlan(
        root=checkout,
        base_commit=base_commit,
        head_commit=head_commit,
        occurrences=tuple([*eligible, *ignored]),
        importer_sha256=tuple(sorted(importer_hashes.items())),
    )


def _resolve_base_bound_cascade(
    *,
    module_bytes: dict[str, bytes],
    candidates: list[_ProofImportScan],
    current_target_bytes: dict[str, bytes | None],
) -> dict[str, bytes]:
    candidates_by_importer: dict[str, list[_ProofImportScan]] = defaultdict(list)
    for scan in candidates:
        candidates_by_importer[scan.importer_path].append(scan)
    virtual = dict(module_bytes)
    iteration_limit = len(candidates_by_importer) + 2
    for _iteration in range(iteration_limit):
        updated_virtual = dict(virtual)
        for importer_path, scans in sorted(candidates_by_importer.items()):
            original = module_bytes[importer_path].decode("utf-8")
            updated = original
            for scan in sorted(
                scans, key=lambda value: value.start_index, reverse=True
            ):
                replacement = _target_hash_for_virtual_files(
                    scan.target_path,
                    virtual_module_bytes=virtual,
                    current_target_bytes=current_target_bytes,
                )
                rendered = _render_scalar(replacement, scan.scalar_style)
                updated = (
                    updated[: scan.start_index] + rendered + updated[scan.end_index :]
                )
            updated_virtual[importer_path] = updated.encode("utf-8")
        if all(
            updated_virtual[path] == virtual[path] for path in candidates_by_importer
        ):
            return updated_virtual
        virtual = updated_virtual
    raise ValueError(
        "base-bound proof hash cascade did not converge; candidate imports contain "
        "a dependency cycle"
    )


def _target_hash_for_virtual_files(
    target_path: str,
    *,
    virtual_module_bytes: dict[str, bytes],
    current_target_bytes: dict[str, bytes | None],
) -> str:
    content = virtual_module_bytes.get(target_path)
    if content is None:
        content = current_target_bytes.get(target_path)
    if content is None:
        raise ValueError(
            f"proof import target disappeared during planning: {target_path}"
        )
    return f"sha256:{_sha256(content)}"


def apply_proof_hash_cascade(plan: CascadePlan, report_path: Path) -> dict[str, object]:
    current_head = _git_text(plan.root, "rev-parse", "--verify", "HEAD^{commit}")
    if current_head != plan.head_commit:
        raise ValueError("proof hash cascade plan HEAD changed before apply")
    if _git_text(plan.root, "status", "--porcelain"):
        raise ValueError("proof hash cascade apply requires a clean Git checkout")
    report = _validate_report_path(plan.root, report_path)
    planned_importer_hashes = dict(plan.importer_sha256)
    grouped: dict[str, list[HashOccurrence]] = defaultdict(list)
    for item in plan.eligible:
        grouped[item.importer_path].append(item)
    updated_files: dict[str, tuple[bytes, bytes, int]] = {}
    for relative_path, items in sorted(grouped.items()):
        path = plan.root / relative_path
        original_bytes = _read_regular_worktree_file(plan.root, path, label="importer")
        if _sha256(original_bytes) != planned_importer_hashes.get(relative_path):
            raise ValueError(
                f"planned importer bytes changed before apply: {relative_path}"
            )
        original = original_bytes.decode("utf-8")
        updated = original
        for item in sorted(items, key=lambda value: value.start_index, reverse=True):
            if item.replacement_hash is None:
                raise ValueError(
                    f"eligible target unexpectedly has no replacement hash: {item.target_path}"
                )
            replacement = _render_scalar(item.replacement_hash, item.scalar_style)
            updated = (
                updated[: item.start_index] + replacement + updated[item.end_index :]
            )
        if updated == original:
            raise ValueError(
                f"eligible proof hash cascade produced no change: {relative_path}"
            )
        updated_files[relative_path] = (
            original_bytes,
            updated.encode("utf-8"),
            len(items),
        )

    for item in plan.eligible:
        target_bytes = _read_regular_worktree_file(
            plan.root,
            plan.root / item.target_path,
            label="proof import target",
        )
        if f"sha256:{_sha256(target_bytes)}" != item.current_target_hash:
            raise ValueError(
                f"planned target bytes changed before apply: {item.target_path}"
            )

    applied_files: list[dict[str, object]] = []
    for relative_path, (original, updated, replacement_count) in sorted(
        updated_files.items()
    ):
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
        "status": "complete",
        "base_commit": plan.base_commit,
        "head_commit_before_apply": plan.head_commit,
        "eligible_replacements": len(plan.eligible),
        "ignored_mismatches": len(plan.ignored),
        "applied_files": applied_files,
        "replacements": [item.public_record() for item in plan.eligible],
        "ignored": [item.public_record() for item in plan.ignored],
    }
    prepared_payload = {**payload, "status": "prepared"}
    created_report_dirs = _create_report_parent(plan.root, report.parent)
    try:
        _write_json_exclusive(report, prepared_payload)
    except BaseException:
        _remove_empty_directories(created_report_dirs)
        raise

    written_paths: list[str] = []
    try:
        for relative_path, (original, updated, _count) in sorted(updated_files.items()):
            path = plan.root / relative_path
            mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
            written_paths.append(relative_path)
            _atomic_replace_bytes(path, updated, mode=mode)
        post_apply = build_proof_hash_cascade_plan(
            plan.root,
            plan.base_commit,
            _expected_worktree_bytes={
                path: updated
                for path, (_original, updated, _count) in updated_files.items()
            },
        )
        if post_apply.eligible:
            raise ValueError(
                "proof hash cascade left eligible migration-induced hashes"
            )
        ignored_identity = {
            (item.importer_path, item.line, item.target_path, item.status)
            for item in plan.ignored
        }
        post_ignored_identity = {
            (item.importer_path, item.line, item.target_path, item.status)
            for item in post_apply.ignored
        }
        if post_ignored_identity != ignored_identity:
            raise ValueError("proof hash cascade changed the ignored mismatch set")
        _atomic_replace_bytes(report, _json_bytes(payload), mode=0o644)
    except BaseException as exc:
        rollback_failures: list[str] = []
        for relative_path in reversed(written_paths):
            original, _updated, _count = updated_files[relative_path]
            path = plan.root / relative_path
            try:
                mode = stat.S_IMODE(path.stat(follow_symlinks=False).st_mode)
                _atomic_replace_bytes(path, original, mode=mode)
            except BaseException as rollback_exc:
                rollback_failures.append(
                    f"{relative_path}: {type(rollback_exc).__name__}: {rollback_exc}"
                )
        rollback_payload = {
            **prepared_payload,
            "status": "rollback_failed" if rollback_failures else "rolled_back",
            "failure": f"{type(exc).__name__}: {exc}",
            "rollback_failures": rollback_failures,
        }
        report_failure: BaseException | None = None
        try:
            _atomic_replace_bytes(report, _json_bytes(rollback_payload), mode=0o644)
        except BaseException as report_exc:
            report_failure = report_exc
        if rollback_failures or report_failure is not None:
            detail = "; ".join(rollback_failures)
            if report_failure is not None:
                report_detail = (
                    f"audit report: {type(report_failure).__name__}: {report_failure}"
                )
                detail = f"{detail}; {report_detail}" if detail else report_detail
            raise RuntimeError(f"proof hash cascade rollback failed: {detail}") from exc
        raise
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
    supplied = Path(root).expanduser()
    if not is_composition_policy_repo_root(supplied):
        raise ValueError("root must be a canonical rulespec-<country> checkout")
    checkout = supplied.resolve(strict=True)
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
    if resolved_parent != allowed:
        raise ValueError("report must be written directly under .axiom/migrations")
    if report.suffix != ".json":
        raise ValueError("report path must end in .json")
    if report.is_symlink():
        raise ValueError("report path must not be a symlink")
    if report.exists():
        raise ValueError("report path must not already exist")
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
    tokens = tuple(yaml.scan(content))
    if any(
        isinstance(token, AliasToken)
        and token_index > 0
        and isinstance(tokens[token_index - 1], KeyToken)
        for token_index, token in enumerate(tokens)
    ):
        raise ValueError("RuleSpec document contains a YAML alias mapping key")
    marker_tokens = tuple(
        token for token in tokens if isinstance(token, (AnchorToken, AliasToken))
    )
    root = yaml.compose(content)
    if root is None:
        return []
    if not isinstance(root, MappingNode):
        raise ValueError("RuleSpec document root must be a mapping")
    found: list[tuple[ScalarNode, ScalarNode]] = []
    root_fields = _mapping_fields(root, context="document root")
    rules = root_fields.get("rules")
    if rules is None:
        return []
    _require_direct_mapping_value(root, "rules", rules, context="document root")
    if not isinstance(rules, SequenceNode):
        raise ValueError("RuleSpec rules must be a sequence")
    seen_hash_spans: set[tuple[int, int]] = set()
    previous_rule_end = rules.start_mark.index
    for rule_index, rule in enumerate(rules.value):
        if rule.start_mark.index < previous_rule_end:
            raise ValueError(
                f"rules[{rule_index}] uses a YAML alias outside its sequence item"
            )
        previous_rule_end = rule.end_mark.index
        _require_node_within(rule, rules, context=f"rules[{rule_index}]")
        if not isinstance(rule, MappingNode):
            raise ValueError(f"RuleSpec rules[{rule_index}] must be a mapping")
        rule_fields = _mapping_fields(rule, context=f"rules[{rule_index}]")
        metadata = rule_fields.get("metadata")
        if metadata is None:
            continue
        _require_direct_mapping_value(
            rule, "metadata", metadata, context=f"rules[{rule_index}]"
        )
        if not isinstance(metadata, MappingNode):
            raise ValueError(f"rules[{rule_index}].metadata must be a mapping")
        metadata_fields = _mapping_fields(
            metadata, context=f"rules[{rule_index}].metadata"
        )
        proof = metadata_fields.get("proof")
        if proof is None:
            continue
        _require_direct_mapping_value(
            metadata, "proof", proof, context=f"rules[{rule_index}].metadata"
        )
        if not isinstance(proof, MappingNode):
            raise ValueError(f"rules[{rule_index}].metadata.proof must be a mapping")
        proof_fields = _mapping_fields(
            proof, context=f"rules[{rule_index}].metadata.proof"
        )
        atoms = proof_fields.get("atoms")
        if atoms is None:
            continue
        _require_direct_mapping_value(
            proof, "atoms", atoms, context=f"rules[{rule_index}].metadata.proof"
        )
        if not isinstance(atoms, SequenceNode):
            raise ValueError(
                f"rules[{rule_index}].metadata.proof.atoms must be a sequence"
            )
        for atom_index, atom in enumerate(atoms.value):
            context = f"rules[{rule_index}].metadata.proof.atoms[{atom_index}]"
            _require_node_within(atom, atoms, context=context)
            if not isinstance(atom, MappingNode):
                raise ValueError(f"{context} must be a mapping")
            atom_fields = _mapping_fields(atom, context=context)
            kind = atom_fields.get("kind")
            if not isinstance(kind, ScalarNode) or kind.value != "import":
                continue
            _require_direct_mapping_value(atom, "kind", kind, context=context)
            import_node = atom_fields.get("import")
            if not isinstance(import_node, MappingNode):
                raise ValueError(f"{context}.import must be a mapping")
            _require_direct_mapping_value(atom, "import", import_node, context=context)
            import_fields = _mapping_fields(import_node, context=f"{context}.import")
            target = import_fields.get("target")
            hash_node = import_fields.get("hash")
            if not isinstance(target, ScalarNode) or not isinstance(
                hash_node, ScalarNode
            ):
                raise ValueError(f"{context}.import requires scalar target and hash")
            _require_direct_mapping_value(
                import_node, "target", target, context=f"{context}.import"
            )
            _require_direct_mapping_value(
                import_node, "hash", hash_node, context=f"{context}.import"
            )
            if hash_node.value.strip() != "sha256:local" and any(
                proof.start_mark.index <= token.start_mark.index < proof.end_mark.index
                for token in marker_tokens
            ):
                raise ValueError(
                    f"rules[{rule_index}].metadata.proof contains YAML anchors or "
                    "aliases and a nonlocal import proof atom"
                )
            span = (hash_node.start_mark.index, hash_node.end_mark.index)
            if span in seen_hash_spans:
                raise ValueError(
                    f"{context}.import hash is duplicated through an alias"
                )
            seen_hash_spans.add(span)
            found.append((target, hash_node))
    return found


def _mapping_fields(node: MappingNode, *, context: str) -> dict[str, Node]:
    fields: dict[str, Node] = {}
    for key, value in node.value:
        if not isinstance(key, ScalarNode):
            raise ValueError(f"{context} contains a non-scalar key")
        if key.tag == "tag:yaml.org,2002:merge":
            raise ValueError(f"{context} contains a YAML merge key")
        if key.value in fields:
            raise ValueError(f"{context} contains duplicate key {key.value!r}")
        fields[key.value] = value
    return fields


def _require_direct_mapping_value(
    mapping: MappingNode, key_name: str, value: Node, *, context: str
) -> None:
    key_node = next(
        (
            key
            for key, candidate in mapping.value
            if isinstance(key, ScalarNode)
            and key.value == key_name
            and candidate is value
        ),
        None,
    )
    if key_node is None or value.start_mark.index < key_node.end_mark.index:
        raise ValueError(f"{context}.{key_name} uses a YAML alias outside its field")


def _require_node_within(node: Node, parent: Node, *, context: str) -> None:
    if (
        node.start_mark.index < parent.start_mark.index
        or node.end_mark.index > parent.end_mark.index
    ):
        raise ValueError(f"{context} uses a YAML alias outside its container")


def _render_scalar(value: str, style: str | None) -> str:
    if style is None:
        return value
    if style == "'":
        return "'" + value.replace("'", "''") + "'"
    if style == '"':
        return json.dumps(value)
    raise ValueError(f"unsupported proof hash scalar style: {style!r}")


def _read_optional_regular_worktree_file(
    root: Path, path: Path, *, label: str
) -> bytes | None:
    relative = path.relative_to(root)
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{label} must not contain a symlink: {relative}")
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file: {relative}")
    return path.read_bytes()


def _read_regular_worktree_file(root: Path, path: Path, *, label: str) -> bytes:
    content = _read_optional_regular_worktree_file(root, path, label=label)
    if content is None:
        raise ValueError(f"{label} does not exist: {path.relative_to(root)}")
    return content


def _create_report_parent(root: Path, parent: Path) -> tuple[Path, ...]:
    allowed = root / ".axiom" / "migrations"
    try:
        parent.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("report must be written under .axiom/migrations") from exc
    missing: list[Path] = []
    cursor = parent
    while cursor != root and not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    created: list[Path] = []
    try:
        for directory in reversed(missing):
            directory.mkdir()
            created.append(directory)
            _fsync_directory(directory.parent)
    except BaseException:
        _remove_empty_directories(tuple(reversed(created)))
        raise
    return tuple(reversed(created))


def _remove_empty_directories(directories: tuple[Path, ...]) -> None:
    for directory in directories:
        try:
            directory.rmdir()
            _fsync_directory(directory.parent)
        except OSError:
            return


def _json_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(path.parent)


def _atomic_replace_bytes(path: Path, content: bytes, *, mode: int) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def _git_tree_entries(root: Path, commit: str) -> dict[str, tuple[str, str, str]]:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-tree", "-r", "-z", commit],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        raise ValueError(f"git ls-tree failed for {commit}: {detail}")
    entries: dict[str, tuple[str, str, str]] = {}
    for raw_entry in completed.stdout.split(b"\0"):
        if not raw_entry:
            continue
        metadata, separator, raw_path = raw_entry.partition(b"\t")
        fields = metadata.split()
        if separator != b"\t" or len(fields) != 3:
            raise ValueError(f"unexpected git tree entry for {commit}")
        mode, object_type, object_id = (field.decode("ascii") for field in fields)
        path = raw_path.decode(errors="surrogateescape")
        entries[path] = (mode, object_type, object_id)
    return entries


def _git_blob_batch(root: Path, object_ids: set[str]) -> dict[str, bytes]:
    if not object_ids:
        return {}
    requested = sorted(object_ids)
    completed = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch"],
        input="".join(f"{object_id}\n" for object_id in requested).encode("ascii"),
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        raise ValueError(f"git cat-file --batch failed: {detail}")
    output = completed.stdout
    offset = 0
    blobs: dict[str, bytes] = {}
    for expected_object_id in requested:
        header_end = output.find(b"\n", offset)
        if header_end < 0:
            raise ValueError("git cat-file --batch returned a truncated header")
        header = output[offset:header_end].split()
        if len(header) != 3:
            raise ValueError("git cat-file --batch returned an invalid header")
        object_id, object_type, raw_size = header
        if object_id.decode("ascii") != expected_object_id or object_type != b"blob":
            raise ValueError(
                f"git cat-file --batch returned an unexpected object for {expected_object_id}"
            )
        try:
            size = int(raw_size)
        except ValueError as exc:
            raise ValueError(
                "git cat-file --batch returned an invalid blob size"
            ) from exc
        content_start = header_end + 1
        content_end = content_start + size
        if content_end >= len(output) or output[content_end : content_end + 1] != b"\n":
            raise ValueError("git cat-file --batch returned truncated blob content")
        blobs[expected_object_id] = output[content_start:content_end]
        offset = content_end + 1
    if offset != len(output):
        raise ValueError("git cat-file --batch returned unexpected trailing output")
    return blobs


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
