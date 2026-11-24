"""Conservative repository dependencies for incremental waiver audits.

The validator reads more than an import graph: upstream placement checks
index all atomic modules in the current and ancestor jurisdictions. These
search dependencies remain broad until the validator itself stops reading
them. Changed paths also stand in for absent members of those searches, so
deleting or introducing a module cannot evade selection.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

from axiom_encode.constants import (
    RULESPEC_ATOMIC_MODULE_ROOTS,
    RULESPEC_COMPOSITION_SPEC_ROOT,
)
from axiom_encode.harness.dependency_stubs import UnsafeRulespecContextPath
from axiom_encode.harness.validator_pipeline import (
    _candidate_rulespec_repo_roots,
    _candidate_upstream_rulespec_roots,
    _extract_import_items_from_content,
    _normalize_rulespec_dependency_roots,
    _normalize_rulespec_import_path_static,
    _parse_rulespec_target,
    _reject_rulespec_checkout_symlinks,
    _rulespec_import_file_candidates_static,
    _rulespec_import_prefix_static,
    _rulespec_resolution_cache_scope,
    _safe_load_unique_keys,
)
from axiom_encode.repo_routing import (
    _canonical_country_checkout_name,
    _git_config_input_paths,
    _git_identity_input_paths,
    _GitProbeError,
    _nearest_git_boundary,
    _rulespec_routing_cache_scope,
    find_policy_repo_root,
    jurisdiction_subdir_names,
    monorepo_checkout_name,
)
from axiom_encode.validation_waivers import (
    DEFAULT_WAIVER_PATH,
    JURISDICTION_RE,
    WaiverSchemaError,
    _contains_control_or_format,
    _module_path,
)

_AliasKey = tuple[object, ...]


@dataclass(frozen=True)
class WaiverAuditScope:
    scope: str
    paths: tuple[str, ...]
    reasons: dict[str, list[str]]
    fallback_reason: str | None = None


class _PathAliases:
    """Compare paths in one fixed tree without losing filesystem aliases."""

    def __init__(self):
        self.identities: dict[Path, tuple[int, int] | None] = {}

    def identity(self, path: Path) -> tuple[int, int] | None:
        path = Path(os.path.abspath(path))
        if path not in self.identities:
            try:
                info = path.stat()
            except (FileNotFoundError, NotADirectoryError):
                identity = None
            else:
                if not info.st_ino:
                    raise ValueError("filesystem did not provide a path identity")
                identity = (info.st_dev, info.st_ino)
            self.identities[path] = identity
        return self.identities[path]

    def same(self, left: Path, right: Path) -> bool:
        left_identity = self.identity(left)
        right_identity = self.identity(right)
        if left_identity is not None and right_identity is not None:
            return left_identity == right_identity
        # An absent resolver candidate can use a different spelling from an
        # existing changed file. Over-selection on case-sensitive trees is safe.
        return os.path.abspath(left).casefold() == os.path.abspath(right).casefold()

    def within(self, path: Path, parent: Path) -> bool:
        return any(self.same(ancestor, parent) for ancestor in (path, *path.parents))

    def key(self, path: Path) -> _AliasKey:
        """Return the key that ``same`` compares an existing path with.

        A present path is its filesystem identity and an absent path is its
        case-folded spelling. ``same(existing, other)`` holds exactly when
        ``key(other)`` is one of ``existing_keys(existing)``.
        """

        identity = self.identity(path)
        if identity is not None:
            return ("identity", *identity)
        return ("name", os.path.abspath(path).casefold())

    def existing_keys(self, path: Path) -> frozenset[_AliasKey]:
        """Return both keys an existing path can be matched by."""

        keys: set[_AliasKey] = {("name", os.path.abspath(path).casefold())}
        identity = self.identity(path)
        if identity is not None:
            keys.add(("identity", *identity))
        return frozenset(keys)


# The signed apply writer stores one manifest per module under this directory,
# at the module's own repository-relative path with a ``.json`` suffix.
_APPLIED_MANIFEST_PARTS = (".axiom", "encoding-manifests")
# Top-level surfaces that can change toolchain, audit, runtime or composition
# behavior. Nothing beneath them is a module, a companion or inert text.
_SHARED_SURFACE_PARTS = frozenset({".axiom", ".github", RULESPEC_COMPOSITION_SPEC_ROOT})


def _applied_manifest_module(path: str) -> str | None:
    """Return the module a signed apply manifest path belongs to, if exact.

    Only ``.axiom/encoding-manifests/<module path>.json`` is recognized, where
    ``<module path>.yaml`` is a canonical atomic module path. Every other
    ``.axiom`` path (toolchain files, ledgers, indexes, legacy manifest
    layouts, manifests of companions or ProgramSpecs) stays unclassified.
    """

    parts = PurePosixPath(path).parts
    prefix = len(_APPLIED_MANIFEST_PARTS)
    if parts[:prefix] != _APPLIED_MANIFEST_PARTS or not path.endswith(".json"):
        return None
    module_parts = parts[prefix:]
    if any(part in _SHARED_SURFACE_PARTS for part in module_parts):
        return None
    module = "/".join(module_parts).removesuffix(".json") + ".yaml"
    try:
        return _module_path(module)
    except WaiverSchemaError:
        return None


def _changed_path_kind(path: str) -> str | None:
    """Recognize only the repository's module contract and inert Markdown."""

    parts = PurePosixPath(path).parts
    if (
        not parts
        or PurePosixPath(path).as_posix() != path
        or PurePosixPath(path).is_absolute()
        or any(part in {"", ".", ".."} for part in parts)
        or "\\" in path
        or path != path.strip()
        or _contains_control_or_format(path)
    ):
        return None
    if _applied_manifest_module(path) is not None:
        # A signed apply manifest is bound to exactly one module. The scope
        # treats it as a change of that module, never as an inert file.
        return "manifest"
    # These surfaces can change toolchain, audit, runtime or composition
    # behavior. Their documentation is deliberately not an inert exception.
    if path == DEFAULT_WAIVER_PATH or any(
        part in _SHARED_SURFACE_PARTS for part in parts
    ):
        return None
    candidate = path
    kind = "module"
    if path.endswith(".test.yaml"):
        candidate = path.removesuffix(".test.yaml") + ".yaml"
        kind = "companion"
    try:
        _module_path(candidate)
    except WaiverSchemaError:
        # Documentation in a jurisdiction could create an otherwise absent
        # content directory and change missing-target resolution diagnostics.
        inert_location = len(parts) == 1 or parts[0] == "docs"
        return (
            "inert" if inert_location and PurePosixPath(path).suffix == ".md" else None
        )
    return kind


def _require_plain_changed_path(root: Path, relative: str) -> None:
    """Fail closed for changed path types that can affect checkout admission."""

    current = root
    parts = PurePosixPath(relative).parts
    for index, part in enumerate(parts):
        current /= part
        if current.is_symlink():
            raise ValueError(f"changed path contains a symlink: {relative}")
        if current.exists():
            expected = (
                current.is_file() if index == len(parts) - 1 else current.is_dir()
            )
            if not expected:
                raise ValueError(f"changed path has an unsafe file type: {relative}")
            if index == len(parts) - 1 and current.stat().st_nlink > 1:
                # A documentation edit may also mutate an execution input
                # outside the checkout through an otherwise invisible alias.
                raise ValueError(f"changed path has multiple hard links: {relative}")


def _yaml_strings(payload: object) -> Iterator[str]:
    """Inspect canonical target strings, including proof and relation targets."""

    pending = [payload]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict | list | tuple):
            if id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, dict):
                pending.extend(value.keys())
                pending.extend(value.values())
            else:
                pending.extend(value)


@dataclass(frozen=True)
class _ParsedFile:
    """One file's dependency declarations, independent of who reads it."""

    imports: tuple[str, ...]
    targets: tuple[tuple[str, str, Path], ...]
    composition: bool


@dataclass(frozen=True)
class _Node:
    """Direct read dependencies of one file as read for one caller root."""

    dependencies: frozenset[Path]
    targets: tuple[Path, ...]


class _ReadSets:
    """Share directory listings and parsed closures within one fixed-tree audit.

    Every repository-wide input is computed once. A file is parsed once, its
    direct dependencies are built once per caller root, and the candidate
    content roots of a jurisdiction are resolved once. A waiver's read set is
    then a walk over that shared graph, so its cost follows the size of its
    own closure and not the size of the repository.
    """

    def __init__(self, root: Path, dependency_roots: Sequence[Path]):
        self.root = Path(root).resolve()
        checkout_name = _canonical_country_checkout_name(
            self.root, allow_composition_specs=True
        )
        if checkout_name is None:
            raise ValueError("waiver audit root is not a canonical country checkout")
        for child in self.root.iterdir():
            if (
                child.is_dir()
                and JURISDICTION_RE.fullmatch(child.name)
                and monorepo_checkout_name(child.name) != checkout_name
            ):
                raise ValueError(f"foreign jurisdiction directory: {child.name}")
        self.dependency_roots = _normalize_rulespec_dependency_roots(dependency_roots)
        self.git_boundary = _nearest_git_boundary(self.root)
        if self.git_boundary is not None and self.git_boundary != self.root:
            raise ValueError("waiver audit root is not an exact Git checkout root")
        self.git_config_inputs = (
            _git_config_input_paths(str(self.root))
            if self.git_boundary is not None
            else ()
        )
        self.atomic_files: dict[Path, set[Path]] = {}
        self.closures: dict[tuple[Path, Path], set[Path]] = {}
        self.aliases = _PathAliases()
        self._parsed: dict[Path, _ParsedFile] = {}
        self._nodes: dict[tuple[Path, Path], _Node] = {}
        self._node_keys: dict[tuple[Path, Path], frozenset[_AliasKey]] = {}
        self._regular: dict[Path, bool] = {}
        self._repo_roots: dict[tuple[str, str, Path], tuple[Path, ...]] = {}
        self._content_roots: dict[Path, Path | None] = {}
        self._upstream_roots: dict[tuple[Path, str], frozenset[Path]] = {}
        self._identity_inputs: tuple[Path, ...] | None = None
        self._keys: dict[Path, _AliasKey | None] = {}
        self._country_keys: dict[Path, frozenset[_AliasKey]] = {}
        self._country_module_keys: frozenset[_AliasKey] | None = None
        self._country_failures: dict[Path, Exception] = {}

    def relative(self, path: Path) -> str | None:
        # Do not resolve an absent target through a different path. The same
        # lexical candidate must remain present after deletion and rename.
        lexical = Path(os.path.abspath(path))
        try:
            return lexical.relative_to(self.root).as_posix()
        except ValueError:
            # Resolvers can retain the case spelling of an import or Git
            # include rather than the spelling used for the checkout root.
            for ancestor in lexical.parents:
                if self.aliases.same(ancestor, self.root):
                    return lexical.relative_to(ancestor).as_posix()
            return None

    def modules(self, content_root: Path) -> set[Path]:
        if content_root not in self.atomic_files:
            self.atomic_files[content_root] = {
                path
                for source_name in RULESPEC_ATOMIC_MODULE_ROOTS
                for path in (content_root / source_name).rglob("*.yaml")
                if not path.name.endswith(".test.yaml") and path.is_file()
            }
        return self.atomic_files[content_root]

    def key(self, path: Path) -> _AliasKey | None:
        """Return a dependency's comparison key, or None outside the checkout.

        A dependency is compared at its repository-relative location, as the
        read set records it. A changed path matches the dependency exactly
        when the dependency's key is one of the changed path's keys.
        """

        if path not in self._keys:
            relative = self.relative(path)
            self._keys[path] = (
                None if relative is None else self.aliases.key(self.root / relative)
            )
        return self._keys[path]

    def _is_regular(self, path: Path) -> bool:
        """Return whether ``path`` is a regular file; refuse any other entry."""

        if path not in self._regular:
            try:
                mode = os.lstat(path).st_mode
            except (FileNotFoundError, NotADirectoryError):
                self._regular[path] = False
            else:
                if not stat.S_ISREG(mode):
                    # The validator's resolver refuses a directory, a symlink
                    # or a special file at a dependency path. Selection must
                    # not guess what such a tree validates as.
                    raise ValueError(
                        "unsafe RuleSpec dependency candidate: "
                        f"{self.relative(path) or path.name}"
                    )
                self._regular[path] = True
        return self._regular[path]

    def content_root(self, path: Path) -> Path | None:
        """Return the jurisdiction root of ``path``, resolved once per directory."""

        directory = path.parent
        if directory not in self._content_roots:
            self._content_roots[directory] = find_policy_repo_root(path)
        return self._content_roots[directory]

    def upstream_roots(self, content_root: Path, jurisdiction: str) -> frozenset[Path]:
        """Return the placement-search roots of one jurisdiction, resolved once."""

        cache_key = (content_root, jurisdiction)
        if cache_key not in self._upstream_roots:
            roots = (
                set(
                    _candidate_upstream_rulespec_roots(
                        content_root,
                        rulespec_dependency_roots=self.dependency_roots,
                    )
                )
                if content_root.is_dir()
                else set()
            )
            # Include nonexistent ancestor jurisdictions as search roots too:
            # adding their first module changes upstream placement validation.
            prefix_parts = jurisdiction.split("-")
            roots.update(
                self.root / "-".join(prefix_parts[:length])
                for length in range(1, len(prefix_parts) + 1)
            )
            self._upstream_roots[cache_key] = frozenset(roots)
        return self._upstream_roots[cache_key]

    def identity_inputs(self) -> tuple[Path, ...]:
        if self._identity_inputs is None:
            self._identity_inputs = _git_identity_input_paths(
                self.root,
                git_boundary=self.git_boundary,
                git_config_inputs=self.git_config_inputs,
            )
        return self._identity_inputs

    def parsed(self, current: Path) -> _ParsedFile:
        """Parse one file's imports and canonical targets exactly once."""

        if current in self._parsed:
            return self._parsed[current]
        content = current.read_text(encoding="utf-8")
        payload = _safe_load_unique_keys(content)
        imports = _extract_import_items_from_content(content)
        declared_imports = payload.get("imports") if isinstance(payload, dict) else None
        if declared_imports is not None and not isinstance(declared_imports, list):
            raise ValueError(
                f"unclassifiable imports in {self.relative(current) or current.name}"
            )
        if isinstance(declared_imports, list) and any(
            not isinstance(item, str) for item in declared_imports
        ):
            raise ValueError(
                f"unclassifiable import item in {self.relative(current) or current.name}"
            )
        if isinstance(declared_imports, list):
            # The engine accepts inline YAML sequences too, while the
            # static line extractor intentionally handles block syntax.
            imports.extend(item for item in declared_imports if isinstance(item, str))
        unique_imports = tuple(dict.fromkeys(imports))
        for imported in unique_imports:
            import_base = imported.split("#", 1)[0].strip().strip("\"'")
            if not _rulespec_import_prefix_static(imported) and import_base.endswith(
                (".yaml", ".yml")
            ):
                raise ValueError(
                    f"ambiguous import {imported!r} in "
                    f"{self.relative(current) or current.name}"
                )
        # Proof import hashes and source_relation verification read targets
        # even when they are absent from the imports block. A target's rule
        # symbol does not change which file is read.
        targets: dict[tuple[str, str, Path], None] = {}
        for value in _yaml_strings(payload):
            target_ref = _parse_rulespec_target(value)
            if target_ref is not None:
                targets[
                    (target_ref.repo_name, target_ref.prefix, target_ref.relative_path)
                ] = None
        module_data = payload.get("module") if isinstance(payload, dict) else None
        result = _ParsedFile(
            imports=unique_imports,
            targets=tuple(targets),
            composition=isinstance(module_data, dict)
            and module_data.get("kind") == "composition",
        )
        self._parsed[current] = result
        return result

    def repo_roots(
        self, repo_name: str, prefix: str, resolution_root: Path
    ) -> tuple[Path, ...]:
        """Return the validator's candidate content roots, resolved once."""

        cache_key = (repo_name, prefix, resolution_root)
        if cache_key not in self._repo_roots:
            self._repo_roots[cache_key] = tuple(
                _candidate_rulespec_repo_roots(
                    repo_name,
                    resolution_root,
                    prefix=prefix,
                    rulespec_dependency_roots=self.dependency_roots,
                )
            )
        return self._repo_roots[cache_key]

    def target_candidates(
        self, target: tuple[str, str, Path], resolution_root: Path
    ) -> tuple[Path, ...]:
        """Return every path a canonical target can resolve to, present or not."""

        repo_name, prefix, relative_path = target
        return tuple(
            content_root / relative_path
            for content_root in self.repo_roots(repo_name, prefix, resolution_root)
        )

    def import_candidates(
        self, imported: str, *, rules_file: Path, resolution_root: Path
    ) -> tuple[Path, ...]:
        """Return the import resolver's ordered candidates, present or not.

        A jurisdiction-prefixed import routes only through candidate content
        roots, which this reader resolves once per jurisdiction. An unprefixed
        import depends on the importing file, so it keeps the validator's own
        candidate function.
        """

        if not _rulespec_import_prefix_static(imported):
            return tuple(
                _rulespec_import_file_candidates_static(
                    imported,
                    rules_file=rules_file,
                    policy_repo_path=resolution_root,
                    rulespec_dependency_roots=self.dependency_roots,
                )
            )
        normalized = _normalize_rulespec_import_path_static(imported)
        if not normalized:
            return ()
        normalized_path = Path(normalized)
        if normalized_path.is_absolute() or any(
            part in {"", ".", ".."} for part in normalized_path.parts
        ):
            raise UnsafeRulespecContextPath(
                f"RuleSpec import target is outside the active policy root: {imported}"
            )
        if (
            not normalized_path.parts
            or normalized_path.parts[0] not in RULESPEC_ATOMIC_MODULE_ROOTS
        ):
            return ()
        target_ref = _parse_rulespec_target(imported)
        if target_ref is None:
            return ()
        return self.target_candidates(
            (target_ref.repo_name, target_ref.prefix, target_ref.relative_path),
            resolution_root,
        )

    def node(self, current: Path, content_root: Path) -> _Node:
        """Return one existing file's direct dependencies for one caller root."""

        cache_key = (current, content_root)
        if cache_key in self._nodes:
            return self._nodes[cache_key]
        parsed = self.parsed(current)
        current_root = self.content_root(current) or content_root
        # Engine import recursion routes from each imported module's own
        # jurisdiction; static source-relation recursion can retain the
        # original caller root. Cover both with the shared resolver.
        resolution_roots = tuple(dict.fromkeys((content_root, current_root)))
        dependencies: dict[Path, None] = {}
        for imported in parsed.imports:
            candidates_found = False
            for resolution_root in resolution_roots:
                candidates = self.import_candidates(
                    imported, rules_file=current, resolution_root=resolution_root
                )
                dependencies.update(dict.fromkeys(candidates))
                candidates_found |= bool(candidates)
            if not candidates_found:
                raise ValueError(
                    f"unclassifiable import {imported!r} in "
                    f"{self.relative(current) or current.name}"
                )
        for target in parsed.targets:
            for resolution_root in resolution_roots:
                dependencies.update(
                    dict.fromkeys(self.target_candidates(target, resolution_root))
                )
        # The validator's resolver reads one of these candidates. Following
        # every present candidate is a superset of whichever one it picks, and
        # keeping the absent ones covers a target that is added or deleted.
        node = _Node(
            dependencies=frozenset(dependencies),
            targets=tuple(
                candidate for candidate in dependencies if self._is_regular(candidate)
            ),
        )
        self._nodes[cache_key] = node
        return node

    def reachable(
        self, starts: Sequence[Path], content_root: Path
    ) -> Iterator[tuple[Path, _Node | None]]:
        """Walk every file the starts transitively read for one caller root."""

        pending = list(starts)
        seen: set[Path] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            if not self._is_regular(current):
                yield current, None
                continue
            node = self.node(current, content_root)
            yield current, node
            pending.extend(node.targets)

    def node_keys(self, current: Path, content_root: Path) -> frozenset[_AliasKey]:
        """Return the comparison keys of one file's direct dependencies.

        The file's own key is not needed: a walked file is either a start,
        which selection compares directly, or a candidate of the file that
        named it.
        """

        cache_key = (current, content_root)
        if cache_key not in self._node_keys:
            dependencies = (
                self.node(current, content_root).dependencies
                if self._is_regular(current)
                else frozenset()
            )
            self._node_keys[cache_key] = frozenset(
                key for path in dependencies if (key := self.key(path)) is not None
            )
        return self._node_keys[cache_key]

    def reachable_keys(
        self,
        starts: Sequence[Path],
        content_root: Path,
        wanted: frozenset[_AliasKey],
    ) -> set[_AliasKey]:
        """Return the wanted keys that the starts transitively read."""

        found: set[_AliasKey] = set()
        for current, _node in self.reachable(starts, content_root):
            found.update(self.node_keys(current, content_root) & wanted)
        return found

    def closure(self, module: Path, content_root: Path) -> set[Path]:
        cache_key = (module, content_root)
        if cache_key in self.closures:
            return self.closures[cache_key]
        dependencies: set[Path] = set()
        for current, node in self.reachable([module], content_root):
            dependencies.add(current)
            if node is not None:
                dependencies.update(node.dependencies)
        self.closures[cache_key] = dependencies
        return dependencies

    def waiver(self, path: str) -> tuple[Path, Path, Path, frozenset[Path], bool]:
        """Return a waiver's module, companion, root, search roots and kind."""

        _module_path(path)
        module = self.root / path
        content_root = self.content_root(module)
        if content_root is None:
            # Removed ledger entries can have an absent module. Their
            # lexical jurisdiction root still determines the search scope.
            content_root = self.root / PurePosixPath(path).parts[0]
        upstream_roots = self.upstream_roots(content_root, PurePosixPath(path).parts[0])
        companion = module.with_name(f"{module.stem}.test.yaml")
        composition = module.is_file() and self.parsed(module).composition
        return module, companion, content_root, upstream_roots, composition

    def shared_dependencies(self) -> set[Path]:
        """Return the inputs every waiver's validation reads."""

        return {
            self.root / DEFAULT_WAIVER_PATH,
            self.root / ".axiom" / "toolchain.toml",
            *self.identity_inputs(),
        }

    def country_modules(self) -> Iterator[Path]:
        """Yield every atomic module of every jurisdiction in the checkout."""

        for jurisdiction in jurisdiction_subdir_names(
            self.root, allow_composition_specs=True
        ):
            yield from self.modules(self.root / jurisdiction)

    def country_module_keys(self) -> frozenset[_AliasKey]:
        """Return the keys of every atomic module in the checkout, once."""

        if self._country_module_keys is None:
            self._country_module_keys = frozenset(
                key
                for candidate in self.country_modules()
                if (key := self.key(candidate)) is not None
            )
        return self._country_module_keys

    def country_keys(self, content_root: Path) -> frozenset[_AliasKey]:
        """Return the keys a composition's program scan can read, once per root.

        Program ownership scans every ProgramSpec and its transitive scope
        before the owning program is composed. Country-wide modules
        conservatively cover those discovered scope targets.
        """

        if content_root in self._country_failures:
            # The walk is the same for every composition of this root. Do not
            # repeat a walk that cannot be classified for each of them.
            raise self._country_failures[content_root]
        if content_root not in self._country_keys:
            keys: set[_AliasKey] = set()
            program_root = self.root / RULESPEC_COMPOSITION_SPEC_ROOT
            for program in program_root.rglob("*.yaml"):
                if (key := self.key(program)) is not None:
                    keys.add(key)
            try:
                for current, _node in self.reachable(
                    sorted(self.country_modules()), content_root
                ):
                    keys.update(self.node_keys(current, content_root))
            except (ValueError, yaml.YAMLError) as exc:
                self._country_failures[content_root] = exc
                raise
            self._country_keys[content_root] = frozenset(keys)
        return self._country_keys[content_root]

    def read_set(
        self, path: str, *, include_index_files: bool = True
    ) -> tuple[set[str], set[Path], bool]:
        module, companion, content_root, upstream_roots, composition = self.waiver(path)
        dependencies = {module, companion, *self.shared_dependencies()}
        if include_index_files:
            for upstream_root in upstream_roots:
                dependencies.update(self.modules(upstream_root))
        if module.is_file():
            dependencies.update(self.closure(module, content_root))
            if companion.is_file():
                # CI checks resolve absolute companion references against
                # their source modules, including targets the module never
                # imports. Inputs, outputs, relations and nested table rows
                # can all carry these references. Keep absent candidates as
                # well so deleting or adding a referenced module is covered.
                dependencies.update(self.closure(companion, content_root))
        if composition:
            # Program ownership scans every ProgramSpec and its transitive
            # scope before the owning program is composed. Country-wide
            # modules conservatively cover those discovered scope targets.
            program_root = self.root / RULESPEC_COMPOSITION_SPEC_ROOT
            dependencies.update(program_root.rglob("*.yaml"))
            for current, node in self.reachable(
                sorted(self.country_modules()), content_root
            ):
                dependencies.add(current)
                if node is not None:
                    dependencies.update(node.dependencies)
        relative_dependencies = {
            relative
            for dependency in dependencies
            if (relative := self.relative(dependency)) is not None
        }
        return relative_dependencies, set(upstream_roots), composition


def validation_waiver_read_set(
    path: str,
    *,
    root: Path,
    rulespec_dependency_roots: Sequence[Path] = (),
) -> set[str]:
    """Return repository read dependencies plus missing resolver candidates."""

    with _rulespec_resolution_cache_scope():
        return _ReadSets(root, rulespec_dependency_roots).read_set(path)[0]


def _waiver_selection_reasons(
    reader: _ReadSets,
    aliases: _PathAliases,
    path: str,
    *,
    meaningful: Sequence[tuple[str, str, str]],
    changed_keys: dict[str, frozenset[_AliasKey]],
    ancestor_keys: dict[str, frozenset[_AliasKey]],
    wanted: frozenset[_AliasKey],
) -> list[str]:
    """Return why the changed paths select one waiver; empty when they do not."""

    module, companion, content_root, upstream_roots, composition = reader.waiver(path)
    starts = [module] if module.is_file() else []
    if starts and companion.is_file():
        starts.append(companion)
    # The walk also proves that every import of this waiver's closure can be
    # classified, whatever changed. The inputs every waiver reads (the ledger,
    # the toolchain file, Git identity files) need no comparison here: none of
    # them can be a classified module or companion, and a change of any of
    # them already forced the full audit above.
    read_keys = reader.reachable_keys(starts, content_root, wanted)
    module_key = reader.key(module)
    companion_key = reader.key(companion)
    search_keys = frozenset(aliases.key(search_root) for search_root in upstream_roots)
    module_changed = bool(ancestor_keys)
    selected_reasons: list[str] = []
    for label, kind, matched in meaningful:
        keys = changed_keys[matched]
        if module_key in keys:
            selected_reasons.append(f"module changed: {label}")
        elif companion_key in keys:
            selected_reasons.append(f"companion changed: {label}")
        elif keys & read_keys:
            selected_reasons.append(f"validation reads changed path: {label}")
        elif kind == "module" and composition:
            # Composing a program indexes every module of the country
            # checkout, so a changed module anywhere is read.
            selected_reasons.append(
                f"validation reads changed path: {label}"
                if keys & reader.country_module_keys()
                else f"validation searches changed module: {label}"
            )
        elif kind == "module" and ancestor_keys[matched] & search_keys:
            selected_reasons.append(f"validation searches changed module: {label}")
        elif composition and not module_changed:
            # Only a companion can reach this point. A program scan can read
            # targets that no module of the waiver's own closure names, so
            # decide the companion from the country-wide walk. With a changed
            # module the composition is selected already, and the walk could
            # add a reason but never a waiver.
            if keys & reader.country_keys(content_root):
                selected_reasons.append(f"validation reads changed path: {label}")
    return selected_reasons


def select_validation_waiver_scope(
    paths: Sequence[str],
    *,
    root: Path,
    changed_paths: set[str],
    rulespec_dependency_roots: Sequence[Path] = (),
    shared_execution_inputs: Sequence[Path] = (),
) -> WaiverAuditScope:
    """Select only when every changed path has a proven dependency class.

    ``paths`` may be one partition of the ledger: each waiver is decided from
    its own read set and the changed paths, never from another waiver, so a
    partition's selection equals its share of the whole ledger's selection.
    A waiver whose own read set cannot be classified is selected. A failure
    that is not one waiver's (the checkout, Git, the filesystem) falls back to
    every path given.
    """

    ordered = tuple(sorted(set(paths)))
    root = Path(root).resolve()

    def full(reason: str) -> WaiverAuditScope:
        return WaiverAuditScope(
            "full",
            ordered,
            {path: [f"full audit fallback: {reason}"] for path in ordered},
            reason,
        )

    if not changed_paths:
        # An empty change set cannot be told from a change set that was never
        # computed (a failed diff, a wrong base, an empty file). Selecting
        # nothing would then pass with no waiver checked. A true empty diff is
        # rare, and the full audit is correct for it.
        return full("no changed path was given")

    shared_input_candidates: set[Path] = set()
    try:
        for raw_input in shared_execution_inputs:
            if not isinstance(raw_input, (str, Path)):
                continue
            # Configured corpus/engine directories and composer executables
            # may live in otherwise inert documentation locations. Retain
            # both lexical and resolved candidates, including deleted inputs,
            # without loading or validating an execution dependency here.
            # Corpus and engine admission expand ~; composer admission keeps
            # it literal. Cover both existing input-resolution contracts.
            for input_path in (Path(raw_input), Path(raw_input).expanduser()):
                lexical = Path(os.path.abspath(input_path))
                shared_input_candidates.update((lexical, lexical.resolve(strict=False)))
    except (OSError, RuntimeError, ValueError) as exc:
        return full(f"could not establish configured execution input paths: {exc}")

    changed_kinds: dict[str, str] = {}
    manifest_modules: dict[str, str] = {}
    aliases = _PathAliases()
    for changed in sorted(changed_paths):
        kind = _changed_path_kind(changed)
        if kind is None:
            return full(f"unclassified or shared validation input changed: {changed}")
        module_relative = changed
        if kind == "companion":
            module_relative = changed.removesuffix(".test.yaml") + ".yaml"
        elif kind == "manifest":
            module_relative = str(_applied_manifest_module(changed))
        if (
            kind in {"module", "companion", "manifest"}
            and monorepo_checkout_name(PurePosixPath(module_relative).parts[0])
            != root.name
        ):
            return full(f"changed path belongs to another country: {changed}")
        try:
            _require_plain_changed_path(root, changed)
            if any(
                aliases.within(root / changed, candidate)
                for candidate in shared_input_candidates
            ):
                return full(f"configured execution input changed: {changed}")
            if aliases.identity(root / changed) is None:
                # A deleted path no longer proves which physical file its
                # spelling denoted. Its old aliases may affect any waiver.
                return full(f"could not establish changed path identity: {changed}")
            if kind in {"companion", "manifest"}:
                _require_plain_changed_path(root, module_relative)
                if not (root / module_relative).is_file():
                    if kind == "manifest":
                        # A manifest whose module is absent proves nothing
                        # about which validation inputs the change touched.
                        return full(
                            f"changed applied manifest has no RuleSpec module: {changed}"
                        )
                    return full(
                        f"changed path is not an adjacent RuleSpec companion: {changed}"
                    )
            if (
                kind in {"module", "companion", "manifest"}
                and (root / module_relative).is_file()
            ):
                payload = _safe_load_unique_keys(
                    (root / module_relative).read_text(encoding="utf-8")
                )
                if (
                    not isinstance(payload, dict)
                    or payload.get("format") != "rulespec/v1"
                ):
                    return full(f"changed path has no valid RuleSpec module: {changed}")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            if kind in {"module", "companion", "manifest"}:
                return full(f"could not classify changed {kind} {changed}: {exc}")
            return full(str(exc))
        changed_kinds[changed] = kind
        if kind == "manifest":
            manifest_modules[changed] = module_relative
    # Each entry is (reason label, kind, repository path that is matched). A
    # manifest stands for its module; when the module changed too, the module
    # entry already selects everything the manifest could.
    meaningful: list[tuple[str, str, str]] = []
    for changed, kind in sorted(changed_kinds.items()):
        if kind == "inert":
            continue
        if kind == "manifest":
            module_relative = manifest_modules[changed]
            if changed_kinds.get(module_relative) == "module":
                continue
            meaningful.append(
                (
                    f"{module_relative} (applied manifest {changed})",
                    "module",
                    module_relative,
                )
            )
        else:
            meaningful.append((changed, kind, changed))
    reasons: dict[str, list[str]] = {}
    try:
        with _rulespec_routing_cache_scope(), _rulespec_resolution_cache_scope():
            # Checkout admission is a shared validation dependency even for
            # an empty selection: a symlink anywhere invalidates the tree.
            _reject_rulespec_checkout_symlinks(root, label="waiver audit checkout")
            reader = _ReadSets(root, rulespec_dependency_roots)
            config_changes = sorted(
                changed
                for changed in changed_paths
                if any(
                    aliases.same(root / changed, config)
                    for config in reader.git_config_inputs
                )
            )
            if config_changes:
                return full(f"Git configuration input changed: {config_changes[0]}")
            # This reader contains only fixed-tree dependency discovery;
            # fingerprints retain their independent validation scopes.
            changed_keys = {
                matched: aliases.existing_keys(root / matched)
                for _label, _kind, matched in meaningful
            }
            wanted = frozenset(key for keys in changed_keys.values() for key in keys)
            # Search-root membership is equivalent to materializing the
            # jurisdiction index for changed modules, including deletes.
            ancestor_keys = {
                matched: frozenset(
                    key
                    for ancestor in (root / matched, *(root / matched).parents)
                    for key in aliases.existing_keys(ancestor)
                )
                for _label, kind, matched in meaningful
                if kind == "module"
            }
            for path in ordered:
                try:
                    selected_reasons = _waiver_selection_reasons(
                        reader,
                        aliases,
                        path,
                        meaningful=meaningful,
                        changed_keys=changed_keys,
                        ancestor_keys=ancestor_keys,
                        wanted=wanted,
                    )
                except (ValueError, yaml.YAMLError) as exc:
                    # A waiver whose own read set cannot be classified is
                    # audited, whatever changed. Every other waiver is still
                    # decided from its own established read set. A filesystem
                    # failure is not a property of one waiver's content: it
                    # leaves this handler and audits every path given.
                    detail = " ".join(str(exc).split())
                    selected_reasons = [
                        f"validation read set could not be established: {detail}"
                    ]
                if selected_reasons:
                    reasons[path] = selected_reasons
    except (OSError, ValueError, yaml.YAMLError, _GitProbeError) as exc:
        return full(f"could not establish validation read set: {exc}")
    return WaiverAuditScope("incremental", tuple(reasons), reasons)
