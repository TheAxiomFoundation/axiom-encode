"""Source-hash pinning, staleness checking, and provenance stamping.

RuleSpec modules ground to legal text through
``module.source_verification.corpus_citation_path``. The helpers here add
the mechanical half of that grounding:

- :func:`resolved_source_verification_block` resolves a citation and pins the
  complete stored body, including when encoding used a child slice.
- :func:`provenance_block` builds the ``module.encoding_provenance`` block
  recording the encoder version, model, and run id in module content.
- :func:`check_staleness` recomputes pinned hashes against a local corpus
  checkout and reports every module whose source text no longer matches.

All corpus reads delegate to :mod:`axiom_encode.corpus_resolver`, keeping
generation, validation, proof checking, and staleness on one release-aware
resolution path.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple, Sequence

import yaml

from axiom_encode.constants import (
    RULESPEC_ATOMIC_MODULE_ROOTS,
    RULESPEC_COMPOSITION_SPEC_ROOT,
    RULESPEC_FILE_SUFFIX,
    RULESPEC_TEST_FILE_SUFFIX,
)
from axiom_encode.corpus_resolver import (
    CorpusResolutionError,
    InvalidCorpusCitationError,
    LocalCorpusRelease,
    require_canonical_corpus_citation_path,
    resolve_local_corpus_source,
)
from axiom_encode.repo_routing import (
    canonical_rulespec_root_identity,
    is_policy_repo_root,
)
from axiom_encode.toolchain import load_rulespec_local_corpus_release


class PinnedModule(NamedTuple):
    """A module that declares ``source_verification.source_sha256``."""

    module_path: Path
    citation_path: str | None
    pinned_sha: str


class StaleModule(NamedTuple):
    """A pinned module whose corpus text no longer matches its pin.

    ``current_sha`` is ``None`` when the pinned provision text cannot be
    found in the corpus checkout (including pins that declare no
    ``corpus_citation_path``), which is itself a staleness signal: the pin
    can no longer be verified. ``resolution_error`` preserves the explicit
    fail-closed reason, including active-release ambiguity.
    """

    module_path: Path
    pinned_sha: str
    current_sha: str | None
    resolution_error: str | None = None


class RuleSpecScanError(RuntimeError):
    """A candidate RuleSpec file that could not be inspected safely."""

    def __init__(self, module_path: Path, reason: str) -> None:
        self.module_path = module_path
        self.reason = reason
        super().__init__(f"{module_path}: {reason}")


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


_MAX_RULESPEC_YAML_BYTES = 4 * 1024 * 1024
_MAX_RULESPEC_SCAN_YAML_BYTES = 128 * 1024 * 1024
_MAX_RULESPEC_SCAN_ENTRIES = 100_000
_MAX_RULESPEC_SCAN_DEPTH = 128
_MAX_YAML_TOKENS = 100_000
_MAX_YAML_NODES = 50_000
_MAX_YAML_DEPTH = 128
_MAX_YAML_ALIASES = 1_024
_YAML_MERGE_TAG = "tag:yaml.org,2002:merge"
_IGNORED_RULESPEC_SCAN_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "_axiom",
        "__pycache__",
        "node_modules",
        "venv",
    }
)
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_OPEN_FLAGS = (
    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
)


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    """Construct a mapping without PyYAML's last-key-wins behavior."""
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class _ScanBudget:
    """Mutable entry counter shared by descriptor-relative traversal."""

    def __init__(self) -> None:
        self.entries = 0
        self.yaml_bytes = 0


def _stat_identity(file_stat: os.stat_result) -> tuple[int, int]:
    return file_stat.st_dev, file_stat.st_ino


def _stable_stat_fields(file_stat: os.stat_result) -> tuple[int, ...]:
    """Fields that must remain stable while a file or directory is scanned."""
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _read_bounded_regular_file(
    directory_fd: int,
    name: str,
    path: Path,
    expected_stat: os.stat_result,
    budget: _ScanBudget,
) -> str:
    """Read one stable regular file without following a replaced symlink."""
    try:
        file_fd = os.open(name, _FILE_OPEN_FLAGS, dir_fd=directory_fd)
    except OSError as exc:
        raise RuleSpecScanError(
            path,
            f"could not securely open candidate RuleSpec YAML: {exc}",
        ) from exc
    try:
        opened_stat = os.fstat(file_fd)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise RuleSpecScanError(
                path, "candidate RuleSpec YAML is not a regular file"
            )
        if _stat_identity(opened_stat) != _stat_identity(expected_stat):
            raise RuleSpecScanError(
                path,
                "candidate RuleSpec YAML changed while it was being opened",
            )
        if opened_stat.st_size > _MAX_RULESPEC_YAML_BYTES:
            raise RuleSpecScanError(
                path,
                "candidate RuleSpec YAML exceeds the "
                f"{_MAX_RULESPEC_YAML_BYTES}-byte scan limit",
            )
        if budget.yaml_bytes + opened_stat.st_size > _MAX_RULESPEC_SCAN_YAML_BYTES:
            raise RuleSpecScanError(
                path,
                "candidate RuleSpec YAML exceeds the aggregate "
                f"{_MAX_RULESPEC_SCAN_YAML_BYTES}-byte scan limit",
            )

        chunks: list[bytes] = []
        bytes_read = 0
        while True:
            chunk = os.read(
                file_fd,
                min(64 * 1024, _MAX_RULESPEC_YAML_BYTES + 1 - bytes_read),
            )
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
            if bytes_read > _MAX_RULESPEC_YAML_BYTES:
                raise RuleSpecScanError(
                    path,
                    "candidate RuleSpec YAML exceeds the "
                    f"{_MAX_RULESPEC_YAML_BYTES}-byte scan limit",
                )

        final_stat = os.fstat(file_fd)
        if (
            _stable_stat_fields(final_stat) != _stable_stat_fields(opened_stat)
            or bytes_read != final_stat.st_size
        ):
            raise RuleSpecScanError(
                path,
                "candidate RuleSpec YAML changed while it was being read",
            )
        budget.yaml_bytes += bytes_read
    finally:
        os.close(file_fd)

    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeError as exc:
        raise RuleSpecScanError(
            path,
            f"candidate RuleSpec is not valid UTF-8: {exc}",
        ) from exc


def _iter_yaml_sources_in_directory(
    directory_fd: int,
    directory_path: Path,
    budget: _ScanBudget,
    depth: int,
    scan_root: Path,
    layout: str,
) -> Iterator[tuple[Path, str]]:
    """Yield YAML sources through a symlink-free descriptor-relative walk."""
    if depth > _MAX_RULESPEC_SCAN_DEPTH:
        raise RuleSpecScanError(
            directory_path,
            "RuleSpec directory tree exceeds the "
            f"{_MAX_RULESPEC_SCAN_DEPTH}-level limit",
        )
    before_stat = os.fstat(directory_fd)
    entries: list[os.DirEntry[str]] = []
    try:
        with os.scandir(directory_fd) as scanner:
            for entry in scanner:
                if entry.name in _IGNORED_RULESPEC_SCAN_NAMES:
                    continue
                budget.entries += 1
                if budget.entries > _MAX_RULESPEC_SCAN_ENTRIES:
                    raise RuleSpecScanError(
                        directory_path / entry.name,
                        "RuleSpec scan exceeds the "
                        f"{_MAX_RULESPEC_SCAN_ENTRIES}-entry limit",
                    )
                entries.append(entry)
    except OSError as exc:
        raise RuleSpecScanError(
            directory_path,
            f"could not enumerate RuleSpec directory: {exc}",
        ) from exc
    entries.sort(key=lambda entry: entry.name)

    for entry in entries:
        entry_path = directory_path / entry.name
        if _is_canonical_composition_spec_directory(
            scan_root,
            entry_path,
            layout=layout,
        ):
            continue
        try:
            entry_stat = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise RuleSpecScanError(
                entry_path,
                f"could not inspect RuleSpec path: {exc}",
            ) from exc

        mode = entry_stat.st_mode
        is_yaml = entry.name.endswith((RULESPEC_FILE_SUFFIX, ".yml"))
        is_companion = entry.name.endswith(RULESPEC_TEST_FILE_SUFFIX)
        if stat.S_ISLNK(mode):
            if is_yaml:
                raise RuleSpecScanError(
                    entry_path,
                    "YAML symlinks are not permitted inside a RuleSpec scan root",
                )
            # Match pathlib's non-following traversal semantics for unrelated
            # repository/runtime symlinks without allowing a YAML file escape.
            continue
        if is_yaml and not stat.S_ISREG(mode):
            raise RuleSpecScanError(
                entry_path,
                "YAML paths must be regular files",
            )
        if stat.S_ISDIR(mode):
            try:
                child_fd = os.open(
                    entry.name,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=directory_fd,
                )
            except OSError as exc:
                raise RuleSpecScanError(
                    entry_path,
                    f"could not securely open RuleSpec directory: {exc}",
                ) from exc
            try:
                opened_stat = os.fstat(child_fd)
                if not stat.S_ISDIR(opened_stat.st_mode) or _stat_identity(
                    opened_stat
                ) != _stat_identity(entry_stat):
                    raise RuleSpecScanError(
                        entry_path,
                        "RuleSpec directory changed while it was being opened",
                    )
                yield from _iter_yaml_sources_in_directory(
                    child_fd,
                    entry_path,
                    budget,
                    depth + 1,
                    scan_root,
                    layout,
                )
            finally:
                os.close(child_fd)
        elif is_yaml and not is_companion:
            yield (
                entry_path,
                _read_bounded_regular_file(
                    directory_fd,
                    entry.name,
                    entry_path,
                    entry_stat,
                    budget,
                ),
            )

    after_stat = os.fstat(directory_fd)
    if _stable_stat_fields(after_stat) != _stable_stat_fields(before_stat):
        raise RuleSpecScanError(
            directory_path,
            "RuleSpec directory changed while it was being scanned",
        )


def _iter_rulespec_yaml_sources(
    rulespec_root: Path,
    *,
    layout: str,
) -> Iterator[tuple[Path, str]]:
    """Validate ``rulespec_root`` and securely enumerate its YAML files."""
    root = Path(rulespec_root)
    try:
        expected_stat = os.lstat(root)
    except OSError as exc:
        raise RuleSpecScanError(
            root,
            f"could not inspect RuleSpec root: {exc}",
        ) from exc
    if stat.S_ISLNK(expected_stat.st_mode):
        raise RuleSpecScanError(root, "RuleSpec root must not be a symlink")
    if not stat.S_ISDIR(expected_stat.st_mode):
        raise RuleSpecScanError(root, "RuleSpec root must be a directory")
    try:
        root_fd = os.open(root, _DIRECTORY_OPEN_FLAGS)
    except OSError as exc:
        raise RuleSpecScanError(
            root,
            f"could not securely open RuleSpec root: {exc}",
        ) from exc
    try:
        opened_stat = os.fstat(root_fd)
        if not stat.S_ISDIR(opened_stat.st_mode) or _stat_identity(
            opened_stat
        ) != _stat_identity(expected_stat):
            raise RuleSpecScanError(
                root,
                "RuleSpec root changed while it was being opened",
            )
        try:
            yield from _iter_yaml_sources_in_directory(
                root_fd,
                root,
                _ScanBudget(),
                0,
                root,
                layout,
            )
        except RecursionError as exc:
            raise RuleSpecScanError(
                root,
                "RuleSpec directory tree exceeds the safe recursion limit",
            ) from exc
    finally:
        os.close(root_fd)


def _rulespec_scan_root_layout(root: Path) -> str:
    """Return the one canonical layout represented by a scan root."""

    if canonical_rulespec_root_identity(root) is not None:
        return "jurisdiction"
    if is_policy_repo_root(root):
        return "country"
    raise RuleSpecScanError(
        root,
        "RuleSpec scan root must be an exact canonical country checkout or "
        "jurisdiction content root (rulespec-<country>/<jurisdiction>)",
    )


def _is_canonical_rulespec_module_path(
    root: Path,
    path: Path,
    *,
    layout: str,
) -> bool:
    """Return whether ``path`` occupies one of the four atomic module roots."""

    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if layout == "jurisdiction":
        return (
            len(relative.parts) >= 2
            and relative.parts[0] in RULESPEC_ATOMIC_MODULE_ROOTS
        )
    if layout == "country":
        return (
            len(relative.parts) >= 3
            and relative.parts[1] in RULESPEC_ATOMIC_MODULE_ROOTS
            and canonical_rulespec_root_identity(root / relative.parts[0]) is not None
        )
    return False


def _is_canonical_composition_spec_directory(
    root: Path,
    path: Path,
    *,
    layout: str,
) -> bool:
    """Identify ProgramSpec roots lexically during the secure directory walk."""

    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if layout == "jurisdiction":
        return relative.parts == (RULESPEC_COMPOSITION_SPEC_ROOT,)
    if layout != "country" or len(relative.parts) != 2:
        return False
    country = root.name.removeprefix("rulespec-")
    jurisdiction, source_root = relative.parts
    return (
        source_root == RULESPEC_COMPOSITION_SPEC_ROOT
        and re.fullmatch(rf"{re.escape(country)}(?:-[a-z0-9]+)*", jurisdiction)
        is not None
    )


def _plural_corpus_citation_path_locations(payload: object) -> list[str]:
    """Return every retired plural corpus-source field in a parsed RuleSpec."""

    locations: list[str] = []
    seen: set[int] = set()
    stack: list[tuple[str, object]] = [("$", payload)]
    while stack:
        location, value = stack.pop()
        if isinstance(value, (dict, list)):
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{location}.{key}"
                if key == "corpus_citation_paths":
                    locations.append(child)
                stack.append((child, item))
        elif isinstance(value, list):
            stack.extend(
                (f"{location}[{index}]", item) for index, item in enumerate(value)
            )
    return sorted(locations)


def _validate_yaml_tokens(source: str, path: Path) -> None:
    """Apply token, nesting, and alias budgets before YAML composition."""
    starts = (
        yaml.tokens.BlockMappingStartToken,
        yaml.tokens.BlockSequenceStartToken,
        yaml.tokens.FlowMappingStartToken,
        yaml.tokens.FlowSequenceStartToken,
    )
    ends = (
        yaml.tokens.BlockEndToken,
        yaml.tokens.FlowMappingEndToken,
        yaml.tokens.FlowSequenceEndToken,
    )
    depth = 0
    alias_count = 0
    anchor_count = 0
    try:
        for token_count, token in enumerate(
            yaml.scan(source, Loader=yaml.SafeLoader),
            start=1,
        ):
            if token_count > _MAX_YAML_TOKENS:
                raise RuleSpecScanError(
                    path,
                    f"candidate YAML exceeds the {_MAX_YAML_TOKENS}-token limit",
                )
            if isinstance(token, starts):
                depth += 1
                if depth > _MAX_YAML_DEPTH:
                    raise RuleSpecScanError(
                        path,
                        f"candidate YAML exceeds the {_MAX_YAML_DEPTH}-level limit",
                    )
            elif isinstance(token, ends):
                depth -= 1
            if isinstance(token, yaml.tokens.AliasToken):
                alias_count += 1
                if alias_count > _MAX_YAML_ALIASES:
                    raise RuleSpecScanError(
                        path,
                        f"candidate YAML exceeds the {_MAX_YAML_ALIASES}-alias limit",
                    )
            elif isinstance(token, yaml.tokens.AnchorToken):
                anchor_count += 1
                if anchor_count > _MAX_YAML_ALIASES:
                    raise RuleSpecScanError(
                        path,
                        f"candidate YAML exceeds the {_MAX_YAML_ALIASES}-anchor limit",
                    )
    except yaml.YAMLError as exc:
        raise RuleSpecScanError(
            path,
            f"could not parse candidate RuleSpec YAML: {exc}",
        ) from exc


def _validate_yaml_node_graph(
    root_node: yaml.Node | None,
    path: Path,
    *,
    reject_merges: bool = False,
) -> None:
    """Reject cyclic aliases, optionally reject merges, and bound YAML nodes."""
    if root_node is None:
        return
    visited: set[int] = set()
    active: set[int] = set()
    stack: list[tuple[yaml.Node, int, bool]] = [(root_node, 1, False)]
    node_count = 0
    while stack:
        node, depth, exiting = stack.pop()
        identity = id(node)
        if exiting:
            active.remove(identity)
            visited.add(identity)
            continue
        if identity in active:
            raise RuleSpecScanError(path, "recursive YAML aliases are not permitted")
        if identity in visited:
            continue
        node_count += 1
        if node_count > _MAX_YAML_NODES:
            raise RuleSpecScanError(
                path,
                f"candidate YAML exceeds the {_MAX_YAML_NODES}-node limit",
            )
        if depth > _MAX_YAML_DEPTH:
            raise RuleSpecScanError(
                path,
                f"candidate YAML exceeds the {_MAX_YAML_DEPTH}-level limit",
            )
        active.add(identity)
        stack.append((node, depth, True))
        children: list[yaml.Node] = []
        if isinstance(node, yaml.MappingNode):
            for key_node, value_node in node.value:
                if reject_merges and key_node.tag == _YAML_MERGE_TAG:
                    raise RuleSpecScanError(
                        path,
                        "YAML merge keys are not permitted in RuleSpec scans",
                    )
                children.extend((key_node, value_node))
        elif isinstance(node, yaml.SequenceNode):
            children.extend(node.value)
        stack.extend((child, depth + 1, False) for child in reversed(children))


def _compose_bounded_yaml(source: str, path: Path) -> yaml.Node | None:
    """Compose one bounded YAML document without constructing Python values."""
    _validate_yaml_tokens(source, path)
    try:
        root_node = yaml.compose(source, Loader=yaml.SafeLoader)
    except (yaml.YAMLError, RecursionError) as exc:
        raise RuleSpecScanError(
            path,
            f"could not parse candidate RuleSpec YAML: {exc}",
        ) from exc
    _validate_yaml_node_graph(root_node, path)
    return root_node


def source_text_sha256(source_text: str) -> str:
    """SHA-256 hex digest of provision text, exactly as stored (UTF-8)."""
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()


def resolved_source_verification_block(
    release: LocalCorpusRelease,
    citation_path: str,
) -> dict[str, str]:
    """Resolve ``citation_path`` and pin the complete stored corpus body.

    The requested citation is retained in the block while the hash covers the
    complete selected row (or composed descendant body), matching
    :func:`check_staleness` exactly.
    """

    citation_path = require_canonical_corpus_citation_path(citation_path)
    resolved = resolve_local_corpus_source(citation_path, release)
    return {
        "corpus_citation_path": citation_path,
        "source_sha256": resolved.stored_body_sha256,
    }


def provenance_block(model: str, run_id: str) -> dict[str, str]:
    """Build a ``module.encoding_provenance`` block for the current encoder."""
    return {
        "encoder": f"axiom-encode/{_encoder_version()}",
        "model": model,
        "run_id": run_id,
    }


def _encoder_version() -> str:
    try:
        from axiom_encode import __version__
    except ImportError:  # pragma: no cover - the package always ships one
        return "unknown"
    return __version__


def iter_pinned_modules(rulespec_root: Path) -> Iterator[PinnedModule]:
    """Yield every module, marking missing source hashes for explicit reporting.

    Explicit ``*.test.yaml`` companions are not RuleSpec
    modules and are intentionally skipped. Candidate module files fail closed:
    an unreadable file, malformed YAML, or duplicate mapping key raises
    :class:`RuleSpecScanError` instead of being mistaken for an unpinned file.
    """
    root = Path(rulespec_root)
    layout = _rulespec_scan_root_layout(root)
    for path, source in _iter_rulespec_yaml_sources(
        root,
        layout=layout,
    ):
        canonical_path = _is_canonical_rulespec_module_path(
            root,
            path,
            layout=layout,
        )
        if path.suffix == ".yml" and canonical_path:
            raise RuleSpecScanError(
                path,
                "RuleSpec files must use the canonical .yaml extension; .yml is removed",
            )
        root_node = _compose_bounded_yaml(source, path)
        if not isinstance(root_node, yaml.MappingNode):
            continue
        if any(key_node.tag == _YAML_MERGE_TAG for key_node, _ in root_node.value):
            raise RuleSpecScanError(
                path,
                "a top-level YAML merge key could hide a RuleSpec module",
            )
        if not any(
            isinstance(key_node, yaml.ScalarNode) and key_node.value == "module"
            for key_node, _ in root_node.value
        ):
            # A RuleSpec checkout can contain valid YAML metadata such as bulk
            # worklists. Inspect the syntax, but reserve module validation for
            # documents that actually declare a top-level ``module`` key.
            continue
        if path.suffix != RULESPEC_FILE_SUFFIX:
            raise RuleSpecScanError(
                path,
                "RuleSpec files must use the canonical .yaml extension; .yml is removed",
            )
        if not canonical_path:
            raise RuleSpecScanError(
                path,
                "RuleSpec module is outside the canonical "
                "rulespec-<country>/<jurisdiction>/<content-root> layout",
            )
        _validate_yaml_node_graph(root_node, path, reject_merges=True)
        try:
            payload = yaml.load(source, Loader=_UniqueKeySafeLoader)
        except (yaml.YAMLError, RecursionError) as exc:
            raise RuleSpecScanError(
                path,
                f"could not parse candidate RuleSpec YAML: {exc}",
            ) from exc
        if not isinstance(payload, dict):
            raise RuleSpecScanError(path, "candidate RuleSpec root must be a mapping")
        plural_locations = _plural_corpus_citation_path_locations(payload)
        if plural_locations:
            raise RuleSpecScanError(
                path,
                "retired corpus_citation_paths field is not supported: "
                + ", ".join(plural_locations[:5]),
            )
        module = payload.get("module")
        if not isinstance(module, dict):
            raise RuleSpecScanError(
                path,
                "candidate RuleSpec module must be a mapping",
            )
        if "source_verification" not in module:
            yield PinnedModule(path, None, "<missing>")
            continue
        verification = module["source_verification"]
        if not isinstance(verification, dict):
            raise RuleSpecScanError(
                path,
                "module.source_verification must be a mapping",
            )
        if "source_sha256" not in verification:
            citation_path = _validated_persisted_citation_path(
                path,
                verification.get("corpus_citation_path"),
            )
            yield PinnedModule(path, citation_path, "<missing>")
            continue
        raw_sha = verification["source_sha256"]
        if isinstance(raw_sha, str):
            pinned_sha = raw_sha
        elif raw_sha is None:
            pinned_sha = "<null>"
        else:
            # Keep ``PinnedModule``'s existing three-field tuple API while
            # preserving the fact that this YAML value was not a string.
            # The wrapper is intentionally outside the valid digest alphabet,
            # so ``check_staleness`` can fail closed without source resolution.
            pinned_sha = f"<non-string {type(raw_sha).__name__}: {raw_sha!r}>"
        citation_path = _validated_persisted_citation_path(
            path,
            verification.get("corpus_citation_path"),
        )
        yield PinnedModule(path, citation_path, pinned_sha)


def _validated_persisted_citation_path(
    module_path: Path,
    value: object,
) -> str | None:
    """Return one byte-exact machine identity or reject its alternate spelling."""

    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RuleSpecScanError(
            module_path,
            "module.source_verification.corpus_citation_path must be a non-empty "
            "canonical string",
        )
    try:
        return require_canonical_corpus_citation_path(value)
    except InvalidCorpusCitationError as exc:
        raise RuleSpecScanError(module_path, str(exc)) from exc


def _check_pinned_modules_staleness(
    pinned_modules: Iterable[PinnedModule],
    release: LocalCorpusRelease,
) -> list[StaleModule]:
    """Check already-scanned pins against the active corpus release."""
    stale: list[StaleModule] = []
    for pinned in pinned_modules:
        module_path = pinned.module_path
        citation_path = pinned.citation_path
        pinned_sha = pinned.pinned_sha
        if re.fullmatch(r"[0-9a-f]{64}", pinned_sha) is None:
            reason = (
                "module does not pin module.source_verification.source_sha256"
                if pinned_sha == "<missing>"
                else "module.source_verification.source_sha256 must be exactly "
                "64 lowercase hexadecimal characters"
            )
            stale.append(
                StaleModule(
                    module_path,
                    pinned_sha,
                    None,
                    reason,
                )
            )
            continue
        current_sha: str | None = None
        resolution_error: str | None = None
        if citation_path is not None:
            try:
                resolved = resolve_local_corpus_source(
                    citation_path,
                    release,
                )
            except CorpusResolutionError as exc:
                resolution_error = f"{type(exc).__name__}: {exc}"
            else:
                current_sha = resolved.stored_body_sha256
        else:
            resolution_error = "module does not declare corpus_citation_path"
        if current_sha != pinned_sha:
            stale.append(
                StaleModule(
                    module_path,
                    pinned_sha,
                    current_sha,
                    resolution_error,
                )
            )
    return stale


def check_staleness(
    rulespec_root: Path,
    release: LocalCorpusRelease,
) -> list[StaleModule]:
    """Report every pinned module whose corpus text no longer matches.

    Returns ``(module_path, pinned_sha, current_sha, resolution_error)`` tuples for each
    module declaring ``source_verification.source_sha256`` whose recomputed
    hash differs from the pin. The current hash is the resolver's full stored-body
    digest, even when the requested citation resolves to a slice of a parent row.
    ``current_sha`` is ``None`` when the pinned provision cannot be resolved;
    ``resolution_error`` records the fail-closed reason, including ambiguity.
    Modules whose hashes still match are omitted. Candidate RuleSpec scan
    failures raise :class:`RuleSpecScanError` so callers cannot mistake an
    incomplete scan for a clean result.
    """
    return _check_pinned_modules_staleness(
        iter_pinned_modules(rulespec_root),
        release,
    )


def run_check_source_staleness(argv: Sequence[str] | None = None) -> int:
    """CLI for ``axiom-encode check-source-staleness``."""
    parser = argparse.ArgumentParser(
        prog="axiom-encode check-source-staleness",
        description=(
            "Recompute module.source_verification.source_sha256 pins against "
            "a local corpus checkout and report modules whose source text "
            "changed."
        ),
    )
    parser.add_argument(
        "--rulespec-root",
        required=True,
        type=Path,
        help="Root of a jurisdiction RuleSpec checkout to scan for pinned modules.",
    )
    parser.add_argument(
        "--corpus-path",
        dest="corpus_path",
        required=True,
        type=Path,
        help="Root of a canonical local axiom-corpus checkout.",
    )
    args = parser.parse_args(argv)

    try:
        pinned = list(iter_pinned_modules(args.rulespec_root))
    except RuleSpecScanError as exc:
        print(f"ERROR {exc.module_path}")
        print(f"  error   {exc.reason}")
        return 1

    try:
        release = load_rulespec_local_corpus_release(
            args.rulespec_root,
            args.corpus_path,
        )
    except (CorpusResolutionError, ValueError) as exc:
        print(f"ERROR corpus release: {type(exc).__name__}: {exc}")
        return 1

    if not pinned:
        print(f"No RuleSpec modules found under {args.rulespec_root}.")
        return 0

    stale = _check_pinned_modules_staleness(pinned, release)
    if not stale:
        print(
            f"All {len(pinned)} pinned module(s) match corpus release {release.name!r}."
        )
        return 0

    for entry in stale:
        current = entry.current_sha or "<provision text not found>"
        print(f"STALE {entry.module_path}")
        print(f"  pinned  {entry.pinned_sha}")
        print(f"  current {current}")
        if entry.resolution_error is not None:
            print(f"  error   {entry.resolution_error}")
    print(f"{len(stale)} of {len(pinned)} pinned module(s) are stale.")
    return 1
