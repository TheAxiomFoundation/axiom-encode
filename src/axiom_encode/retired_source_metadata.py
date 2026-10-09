"""Pure, base-bound migration of retired RuleSpec source metadata.

No function in this module installs files, invokes a model, signs, or writes Git.
The caller supplies authenticated base blobs and owns the journaled installation.
Receipt replay regenerates both primary edits and the complete proof-pin cascade.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode
from yaml.tokens import AliasToken, AnchorToken

from .constants import RULESPEC_ATOMIC_MODULE_ROOTS
from .corpus_resolver import normalize_corpus_identifier
from .harness.validator_pipeline import _parse_rulespec_target
from .proof_hash_migration import (
    _proof_import_scalar_nodes,
    _ProofImportScan,
    _render_scalar,
    _resolve_base_bound_cascade,
)

PLAN_SCHEMA: Final = "axiom-encode/retired-source-metadata-migration-plan/v1"
RECEIPT_SCHEMA: Final = "axiom-encode/retired-source-metadata-migration-receipt/v1"
MIGRATION_TOOL: Final = "axiom-encode migrate-retired-source-metadata"
RECEIPT_DIR: Final = Path(".axiom") / "retired-source-metadata-migrations"
_GIT_OBJECT = re.compile(r"[0-9a-f]{40}")
_JURISDICTION = re.compile(r"[a-z]{2}(?:-[a-z0-9_]+)*")
_ROOTS = RULESPEC_ATOMIC_MODULE_ROOTS
_PIN_PREFIX = re.compile(rb"(?<![A-Za-z0-9_])sha256:")
_MAX_RECEIPT_BYTES = 16 * 1024 * 1024
_SHAPE_DECISION = (
    "pending source shape decision: Max's d034 Option A checked_paths, Pavel's "
    "source_documents from rulespec-us#1363, or primary-only collapse from #1307"
)


class RetiredSourceMetadataError(ValueError):
    """A requested edit cannot be proved to be the admitted neutral migration."""


@dataclass(frozen=True, slots=True)
class RetiredSourceMetadataPlan:
    base_commit: str
    modules: tuple[Path, ...]
    payload: Mapping[str, object]
    canonical_bytes: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceMetadataRewrite:
    after: bytes
    removed_values: Mapping[object, object] | None
    removed_values_yaml: str | None
    removed_corpus_citation_paths: tuple[str, ...] | None
    corpus_citation_path: str | None


@dataclass(frozen=True, slots=True)
class MigrationFile:
    path: Path
    before: bytes
    after: bytes
    primary: bool


@dataclass(frozen=True, slots=True)
class RetiredSourceMetadataMigration:
    plan: RetiredSourceMetadataPlan
    base_tree: str
    files: tuple[MigrationFile, ...]
    cascade_rewrites: tuple[dict[str, object], ...]
    receipt: Mapping[str, object]
    receipt_bytes: bytes
    receipt_sha256: str
    receipt_relative: Path


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Return the unambiguous JSON representation used by plan and identity."""

    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError) as exc:
        raise RetiredSourceMetadataError(
            "retired source metadata history is not a finite JSON-compatible mapping"
        ) from exc


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RetiredSourceMetadataError(
                f"JSON object contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _load_json(raw: bytes, *, label: str, limit: int) -> object:
    if len(raw) > limit:
        raise RetiredSourceMetadataError(f"{label} exceeds its byte bound")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_json_object)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise RetiredSourceMetadataError(f"{label} is not valid UTF-8 JSON") from exc


def _primary_path(raw: object, *, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise RetiredSourceMetadataError(f"{field} must be a repository path")
    path = Path(raw)
    if (
        path.is_absolute()
        or path.as_posix() != raw
        or any(part in {"", ".", ".."} for part in path.parts)
        or len(path.parts) < 3
        or _JURISDICTION.fullmatch(path.parts[0]) is None
        or path.parts[1] not in _ROOTS
        or path.suffix != ".yaml"
        or path.name.endswith(".test.yaml")
        or any(ord(character) < 32 or character == "\\" for character in raw)
    ):
        raise RetiredSourceMetadataError(
            f"{field} is not a canonical primary RuleSpec path"
        )
    return path


def load_plan_bytes(raw: bytes) -> RetiredSourceMetadataPlan:
    """Parse exact-schema authority bound to a full RuleSpec base commit."""

    payload = _load_json(raw, label="migration plan", limit=1024 * 1024)
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "base_commit",
        "modules",
    }:
        raise RetiredSourceMetadataError(
            "migration plan must contain exactly schema_version, base_commit, modules"
        )
    if payload["schema_version"] != PLAN_SCHEMA:
        raise RetiredSourceMetadataError("migration plan schema_version is unsupported")
    base = payload["base_commit"]
    if not isinstance(base, str) or _GIT_OBJECT.fullmatch(base) is None:
        raise RetiredSourceMetadataError(
            "migration plan base_commit must be a full Git object ID"
        )
    raw_modules = payload["modules"]
    if not isinstance(raw_modules, list) or not 1 <= len(raw_modules) <= 256:
        raise RetiredSourceMetadataError(
            "migration plan modules must contain 1 to 256 paths"
        )
    modules = tuple(
        _primary_path(item, field=f"modules[{index}]")
        for index, item in enumerate(raw_modules)
    )
    if len(set(modules)) != len(modules):
        raise RetiredSourceMetadataError(
            "migration plan modules contains a duplicated path"
        )
    canonical = canonical_json_bytes(payload)
    return RetiredSourceMetadataPlan(
        base, modules, payload, canonical, _sha256(canonical)
    )


class _UniqueLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: _UniqueLoader, node: MappingNode, deep: bool = False):
    result = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise RetiredSourceMetadataError("RuleSpec contains a YAML merge key")
        key = loader.construct_object(key_node, deep=deep)
        try:
            if key in result:
                raise RetiredSourceMetadataError(
                    f"RuleSpec contains duplicate key {key!r}"
                )
            result[key] = loader.construct_object(value_node, deep=deep)
        except TypeError as exc:
            raise RetiredSourceMetadataError(
                "RuleSpec mapping contains a nonscalar key"
            ) from exc
    return result


_UniqueLoader.add_constructor("tag:yaml.org,2002:map", _construct_mapping)


def _yaml(raw: bytes) -> tuple[str, dict[str, object], MappingNode]:
    try:
        text = raw.decode("utf-8")
        payload = yaml.load(text, Loader=_UniqueLoader)
        root = yaml.compose(text)
    except (UnicodeError, yaml.YAMLError, RecursionError) as exc:
        raise RetiredSourceMetadataError("RuleSpec is not valid UTF-8 YAML") from exc
    if not isinstance(payload, dict) or not isinstance(root, MappingNode):
        raise RetiredSourceMetadataError("RuleSpec root must be a mapping")
    return text, payload, root


def _fields(node: MappingNode) -> dict[str, tuple[ScalarNode, Node]]:
    return {
        key.value: (key, value)
        for key, value in node.value
        if isinstance(key, ScalarNode)
    }


def _last_content_end(node: Node) -> int:
    if isinstance(node, MappingNode):
        return max(
            (_last_content_end(value) for _, value in node.value),
            default=node.end_mark.index,
        )
    if isinstance(node, SequenceNode):
        return max(
            (_last_content_end(value) for value in node.value),
            default=node.end_mark.index,
        )
    return node.end_mark.index


def _require_canonical_value_node(node: Node, *, column: int, text: str) -> None:
    """Require two-space block nesting for the values history we remove."""

    error = "values does not have canonical block YAML indentation"
    if isinstance(node, MappingNode):
        if node.flow_style or node.start_mark.column != column:
            raise RetiredSourceMetadataError(error)
        for key, value in node.value:
            if not isinstance(key, ScalarNode) or key.start_mark.column != column:
                raise RetiredSourceMetadataError(error)
            _require_canonical_value_node(value, column=column + 2, text=text)
    elif isinstance(node, SequenceNode):
        if node.flow_style or node.start_mark.column != column:
            raise RetiredSourceMetadataError(error)
        for value in node.value:
            if value.start_mark.column != column + 2:
                raise RetiredSourceMetadataError(error)
            _require_canonical_value_node(value, column=column + 2, text=text)
    elif isinstance(node, ScalarNode) and node.style in {"|", ">"}:
        lines = text.splitlines()[node.start_mark.line + 1 : node.end_mark.line]
        indents = [len(line) - len(line.lstrip(" ")) for line in lines if line.strip()]
        if indents and min(indents) != column:
            raise RetiredSourceMetadataError(error)


def _block_span(
    text: str, key: ScalarNode, value: Node, *, field: str
) -> tuple[int, int]:
    start = text.rfind("\n", 0, key.start_mark.index) + 1
    header_end = text.find("\n", start)
    if (
        key.style is not None
        or key.start_mark.column != 4
        or text[start : header_end + 1] != f"    {field}:\n"
        or not isinstance(value, (MappingNode, SequenceNode))
        or value.flow_style
    ):
        raise RetiredSourceMetadataError(f"{field} does not have canonical block YAML")
    if field == "values":
        _require_canonical_value_node(value, column=6, text=text)
    content_end = _last_content_end(value)
    # A literal scalar's terminal newline is part of its payload. Ordinary
    # scalar marks precede their newline; preserve unrelated trailing blank lines.
    if content_end > start and text[content_end - 1 : content_end] == "\n":
        end = content_end
    else:
        newline = text.find("\n", content_end)
        if newline < 0:
            raise RetiredSourceMetadataError(f"{field} block must end with a newline")
        end = newline + 1
    return start, end


def _normalized_paths(raw: object, *, label: str) -> tuple[str, ...]:
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(item, str) and item for item in raw)
    ):
        raise RetiredSourceMetadataError(f"{label} must be a non-empty citation list")
    try:
        return tuple(normalize_corpus_identifier(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise RetiredSourceMetadataError(
            f"{label} contains an invalid corpus identifier"
        ) from exc


def _history(value: object) -> object:
    """Preserve YAML types JSON cannot represent, alongside the exact raw block.

    Ordinary string-keyed numeric mappings remain ordinary JSON mappings.
    Numeric mapping keys use an explicit entry list so integer ``1`` and string
    ``"1"`` cannot be merged by JSON's mandatory string-key coercion.
    """

    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: _history(item) for key, item in value.items()}
        return {
            "yaml_mapping_entries": [
                {"key": _history(key), "value": _history(item)}
                for key, item in value.items()
            ]
        }
    if isinstance(value, (list, tuple)):
        return [_history(item) for item in value]
    if isinstance(value, (datetime.date, datetime.datetime)):
        return {"yaml_timestamp": value.isoformat()}
    if isinstance(value, bytes):
        return {"yaml_binary_hex": value.hex()}
    if isinstance(value, set):
        return {"yaml_set": sorted((_history(item) for item in value), key=repr)}
    if isinstance(value, float) and (
        value != value or value in {float("inf"), float("-inf")}
    ):
        return {"yaml_float": repr(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise RetiredSourceMetadataError(
        "removed YAML history contains an unsupported value"
    )


def rewrite_source_metadata(raw: bytes) -> SourceMetadataRewrite:
    """Byte-surgically perform R1/R2, then prove exact structural equality."""

    text, payload, root = _yaml(raw)
    module = payload.get("module")
    verification = (
        module.get("source_verification") if isinstance(module, dict) else None
    )
    if not isinstance(verification, dict) or not (
        {"values", "corpus_citation_paths"} & verification.keys()
    ):
        raise RetiredSourceMetadataError(
            "nothing-to-migrate: module has neither retired shape"
        )
    try:
        if any(
            isinstance(token, (AnchorToken, AliasToken)) for token in yaml.scan(text)
        ):
            raise RetiredSourceMetadataError(
                "migration refuses YAML anchors or aliases"
            )
    except (yaml.YAMLError, RecursionError) as exc:
        raise RetiredSourceMetadataError("RuleSpec is not valid YAML") from exc
    module_node = _fields(root).get("module", (None, None))[1]
    verification_node = (
        _fields(module_node).get("source_verification", (None, None))[1]
        if isinstance(module_node, MappingNode)
        else None
    )
    if not isinstance(verification_node, MappingNode) or verification_node.flow_style:
        raise RetiredSourceMetadataError(
            "source_verification must be canonical block YAML"
        )
    fields = _fields(verification_node)
    expected = copy.deepcopy(payload)
    expected_verification = expected["module"]["source_verification"]
    edits: list[tuple[int, int, str]] = []
    values = None
    values_yaml = None
    removed_paths = None
    ancestor = None
    if "values" in verification:
        values = verification["values"]
        if not isinstance(values, dict) or not values:
            raise RetiredSourceMetadataError(
                "source_verification.values must be a non-empty mapping"
            )
        start, end = _block_span(text, *fields["values"], field="values")
        values_yaml = text[start:end]
        edits.append((start, end, ""))
        del expected_verification["values"]
    if "corpus_citation_paths" in verification:
        if "corpus_citation_path" in verification:
            raise RetiredSourceMetadataError(
                "source_verification mixes singular and plural citation fields"
            )
        if "source_sha256" in verification:
            raise RetiredSourceMetadataError(
                "plural source_sha256 has ambiguous aggregate source identity; "
                + _SHAPE_DECISION
            )
        raw_paths = verification["corpus_citation_paths"]
        paths = _normalized_paths(raw_paths, label="corpus_citation_paths")
        candidates = [
            item
            for item in paths
            if all(item == other or other.startswith(item + "/") for other in paths)
        ]
        if len(candidates) != 1:
            raise RetiredSourceMetadataError(
                "plural citations do not have exactly one ancestor; " + _SHAPE_DECISION
            )
        ancestor = candidates[0]
        if any(
            any(character in item for character in "\r\n#'\"") or item.strip() != item
            for item in raw_paths
        ):
            raise RetiredSourceMetadataError(
                "corpus_citation_paths must use canonical plain YAML scalars"
            )
        upstream = verification.get("upstream_source_check")
        if "upstream_source_check" in verification:
            if not isinstance(upstream, dict):
                raise RetiredSourceMetadataError(
                    "upstream_source_check is inconsistent with plural citation history"
                )
            checked = _normalized_paths(
                upstream.get("checked_paths"),
                label="upstream_source_check.checked_paths",
            )
            if ancestor not in checked or not set(paths).issubset(checked):
                raise RetiredSourceMetadataError(
                    "upstream_source_check is inconsistent with plural citation history"
                )
        start, end = _block_span(
            text, *fields["corpus_citation_paths"], field="corpus_citation_paths"
        )
        block = "    corpus_citation_paths:\n" + "".join(
            f"      - {item}\n" for item in raw_paths
        )
        if text[start:end] != block:
            raise RetiredSourceMetadataError(
                "corpus_citation_paths does not have exact canonical block YAML"
            )
        edits.append((start, end, f"    corpus_citation_path: {ancestor}\n"))
        removed_paths = tuple(raw_paths)
        del expected_verification["corpus_citation_paths"]
        expected_verification["corpus_citation_path"] = ancestor
    updated = text
    for start, end, replacement in sorted(edits, reverse=True):
        updated = updated[:start] + replacement + updated[end:]
    after = updated.encode("utf-8")
    _, actual, _ = _yaml(after)
    if actual != expected:
        raise RetiredSourceMetadataError(
            "migration changed unrelated RuleSpec structure"
        )
    return SourceMetadataRewrite(after, values, values_yaml, removed_paths, ancestor)


def _atomic_path(path: str) -> bool:
    try:
        _primary_path(path, field="base file")
        return True
    except RetiredSourceMetadataError:
        return False


def _cascade(base_files: Mapping[str, bytes], rewritten: Mapping[str, bytes]):
    """Reuse the existing bounded fixed-point resolver against virtual blobs."""

    module_bytes = {path: raw for path, raw in base_files.items() if _atomic_path(path)}
    module_bytes.update(rewritten)
    candidates: list[_ProofImportScan] = []
    ignored = []
    for path, raw in sorted(module_bytes.items()):
        # compose() can obtain a base-authorized sha256 scalar only from its
        # literal text or a quoted YAML escape. Keep every escaped document on
        # the full parser path, including escaped mapping keys and hashes.
        if _PIN_PREFIX.search(raw) is None and b"\\" not in raw:
            continue
        try:
            nodes = _proof_import_scalar_nodes(raw.decode("utf-8"))
        except (ValueError, UnicodeError, yaml.YAMLError, RecursionError) as exc:
            raise RetiredSourceMetadataError(
                f"cannot scan proof imports in {path}: {exc}"
            ) from exc
        for target_node, hash_node in nodes:
            target = target_node.value.strip()
            declared = hash_node.value.strip()
            reference = _parse_rulespec_target(target)
            if declared == "sha256:local" or reference is None:
                continue
            target_path = (Path(reference.prefix) / reference.relative_path).as_posix()
            base = base_files.get(target_path)
            if base is None:
                # Existing unresolved imports are outside the transaction.
                continue
            base_hash = "sha256:" + _sha256(base)
            current = module_bytes.get(target_path, base)
            scan = _ProofImportScan(
                importer_path=path,
                target=target,
                target_path=target_path,
                line=hash_node.start_mark.line + 1,
                declared_hash=declared,
                base_target_hash=base_hash,
                current_target_hash="sha256:" + _sha256(current),
                start_index=hash_node.start_mark.index,
                end_index=hash_node.end_mark.index,
                scalar_style=hash_node.style,
            )
            (candidates if declared == base_hash else ignored).append(scan)
    try:
        final = _resolve_base_bound_cascade(
            module_bytes=module_bytes,
            candidates=candidates,
            current_target_bytes=dict(base_files),
        )
    except ValueError as exc:
        raise RetiredSourceMetadataError(str(exc)) from exc
    # A pin that was already current despite differing from base cannot become
    # stale. Preexisting invalid pins are preserved as preexisting invalid pins.
    for scan in ignored:
        final_target = final.get(scan.target_path, base_files[scan.target_path])
        if (
            scan.declared_hash == scan.current_target_hash
            and scan.declared_hash != "sha256:" + _sha256(final_target)
        ):
            raise RetiredSourceMetadataError(
                "cascade would stale a non-base-authorized proof import: "
                + scan.importer_path
            )
    records = []
    for scan in candidates:
        replacement = "sha256:" + _sha256(
            final.get(scan.target_path, base_files[scan.target_path])
        )
        if replacement != scan.declared_hash:
            records.append(
                scan.occurrence(
                    status="eligible", replacement=replacement
                ).public_record()
            )
    # The existing resolver can only edit scalar hash spans. Independently
    # reconstruct each cascade postimage to pin this locality guarantee.
    for path in final:
        if final[path] == module_bytes[path]:
            continue
        text = module_bytes[path].decode("utf-8")
        for scan in sorted(
            (item for item in candidates if item.importer_path == path),
            key=lambda item: item.start_index,
            reverse=True,
        ):
            replacement = "sha256:" + _sha256(
                final.get(scan.target_path, base_files[scan.target_path])
            )
            text = (
                text[: scan.start_index]
                + _render_scalar(replacement, scan.scalar_style)
                + text[scan.end_index :]
            )
        if text.encode("utf-8") != final[path]:
            raise RetiredSourceMetadataError(
                f"cascade changed bytes outside proof hashes: {path}"
            )
    return final, tuple(records)


def receipt_identity_sha256(payload: Mapping[str, object]) -> str:
    """Hash the complete replay authority, excluding its self-referential ID."""

    return _sha256(
        canonical_json_bytes(
            {key: value for key, value in payload.items() if key != "identity_sha256"}
        )
    )


def _leaves_first(
    paths: set[str], rewrites: tuple[dict[str, object], ...]
) -> tuple[str, ...]:
    """Order changed targets before their pin importers, breaking ties by path."""

    dependencies: dict[str, set[str]] = {path: set() for path in paths}
    for item in rewrites:
        importer = item["importer_path"]
        target = item["target_path"]
        if importer in paths and target in paths:
            dependencies[importer].add(target)
    result = []
    pending = set(paths)
    while pending:
        ready = sorted(path for path in pending if not dependencies[path] & pending)
        if not ready:
            raise RetiredSourceMetadataError(
                "changed proof-pin dependency graph contains a cycle"
            )
        result.extend(ready)
        pending.difference_update(ready)
    return tuple(result)


def build_migration(
    plan: RetiredSourceMetadataPlan,
    *,
    base_tree: str,
    base_files: Mapping[str, bytes],
) -> RetiredSourceMetadataMigration:
    """Plan primaries, leaves-first fixed-point pins, and their durable receipt."""

    if not isinstance(base_tree, str) or _GIT_OBJECT.fullmatch(base_tree) is None:
        raise RetiredSourceMetadataError("base_tree must be a full Git tree object ID")
    # Re-parse frozen authority so a caller cannot hand-build an overbroad plan.
    parsed = load_plan_bytes(plan.canonical_bytes)
    if (
        parsed.payload != plan.payload
        or parsed.modules != plan.modules
        or parsed.sha256 != plan.sha256
        or parsed.base_commit != plan.base_commit
    ):
        raise RetiredSourceMetadataError(
            "migration plan object differs from its canonical authority"
        )
    rewritten = {}
    primary_records = []
    for path in sorted(plan.modules):
        relative = path.as_posix()
        if relative not in base_files:
            raise RetiredSourceMetadataError(
                f"primary is absent from authenticated base: {relative}"
            )
        try:
            rewrite = rewrite_source_metadata(base_files[relative])
        except RetiredSourceMetadataError as exc:
            raise RetiredSourceMetadataError(f"{relative}: {exc}") from exc
        rewritten[relative] = rewrite.after
        primary_records.append(
            {
                "path": relative,
                "removed_values": _history(rewrite.removed_values),
                "removed_values_yaml": rewrite.removed_values_yaml,
                "removed_corpus_citation_paths": list(
                    rewrite.removed_corpus_citation_paths
                )
                if rewrite.removed_corpus_citation_paths is not None
                else None,
                "corpus_citation_path": rewrite.corpus_citation_path,
            }
        )
    final, cascade = _cascade(base_files, rewritten)
    ordered_paths = _leaves_first(
        {path for path, after in final.items() if after != base_files[path]},
        cascade,
    )
    order = {path: index for index, path in enumerate(ordered_paths)}
    cascade = tuple(
        sorted(
            cascade,
            key=lambda item: (
                order[item["importer_path"]],
                item["line"],
                item["target"],
            ),
        )
    )
    files = tuple(
        MigrationFile(Path(path), base_files[path], final[path], path in rewritten)
        for path in ordered_paths
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "base_commit": plan.base_commit,
        "base_tree": base_tree,
        "plan": dict(plan.payload),
        "plan_sha256": plan.sha256,
        "primaries": primary_records,
        "files": [
            {
                "path": item.path.as_posix(),
                "before_sha256": _sha256(item.before),
                "after_sha256": _sha256(item.after),
                "primary": item.primary,
            }
            for item in files
        ],
        "cascade_rewrites": list(cascade),
    }
    identity = receipt_identity_sha256(receipt)
    receipt["identity_sha256"] = identity
    raw = canonical_json_bytes(receipt) + b"\n"
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise RetiredSourceMetadataError("migration receipt exceeds its byte bound")
    return RetiredSourceMetadataMigration(
        plan,
        base_tree,
        files,
        cascade,
        receipt,
        raw,
        _sha256(raw),
        RECEIPT_DIR / f"{identity}.json",
    )


def verify_migration_replay(
    receipt_raw: bytes,
    *,
    base_files: Mapping[str, bytes],
) -> RetiredSourceMetadataMigration:
    """Require exact receipt schema and byte-identical deterministic replay."""

    receipt = _load_json(
        receipt_raw, label="migration receipt", limit=_MAX_RECEIPT_BYTES
    )
    keys = {
        "schema_version",
        "identity_sha256",
        "base_commit",
        "base_tree",
        "plan",
        "plan_sha256",
        "primaries",
        "files",
        "cascade_rewrites",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != keys
        or receipt.get("schema_version") != RECEIPT_SCHEMA
    ):
        raise RetiredSourceMetadataError(
            "migration receipt has an unsupported exact schema"
        )
    if not isinstance(receipt["plan"], dict):
        raise RetiredSourceMetadataError("migration receipt plan is not an object")
    plan = load_plan_bytes(canonical_json_bytes(receipt["plan"]))
    migration = build_migration(
        plan, base_tree=receipt["base_tree"], base_files=base_files
    )
    if migration.receipt_bytes != receipt_raw:
        raise RetiredSourceMetadataError(
            "migration receipt does not equal deterministic replay"
        )
    return migration
