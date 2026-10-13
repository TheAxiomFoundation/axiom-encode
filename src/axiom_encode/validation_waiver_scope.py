"""Conservative repository dependencies for incremental waiver audits.

The validator reads more than an import graph: upstream placement checks
index all atomic modules in the current and ancestor jurisdictions. These
search dependencies remain broad until the validator itself stops reading
them. Changed paths also stand in for absent members of those searches, so
deleting or introducing a module cannot evade selection.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

from axiom_encode.constants import (
    RULESPEC_ATOMIC_MODULE_ROOTS,
    RULESPEC_COMPOSITION_SPEC_ROOT,
)
from axiom_encode.harness.validator_pipeline import (
    _candidate_upstream_rulespec_roots,
    _extract_import_items_from_content,
    _normalize_rulespec_dependency_roots,
    _parse_rulespec_target,
    _reject_rulespec_checkout_symlinks,
    _resolve_rulespec_import_file_static,
    _resolve_rulespec_target_file,
    _rulespec_import_file_candidates_static,
    _rulespec_import_prefix_static,
    _rulespec_resolution_cache_scope,
    _rulespec_target_file_candidates,
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


@dataclass(frozen=True)
class WaiverAuditScope:
    scope: str
    paths: tuple[str, ...]
    reasons: dict[str, list[str]]
    fallback_reason: str | None = None


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
    # These surfaces can change toolchain, audit, runtime or composition
    # behavior. Their documentation is deliberately not an inert exception.
    if path == DEFAULT_WAIVER_PATH or any(
        part in {".axiom", ".github", RULESPEC_COMPOSITION_SPEC_ROOT}
        for part in parts
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
        return "inert" if inert_location and PurePosixPath(path).suffix == ".md" else None
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
            expected = current.is_file() if index == len(parts) - 1 else current.is_dir()
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


class _ReadSets:
    """Share directory listings and parsed closures within one fixed-tree audit."""

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

    def relative(self, path: Path) -> str | None:
        # Do not resolve an absent target through a different path. The same
        # lexical candidate must remain present after deletion and rename.
        lexical = Path(os.path.abspath(path))
        try:
            return lexical.relative_to(self.root).as_posix()
        except ValueError:
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

    def closure(self, module: Path, content_root: Path) -> set[Path]:
        cache_key = (module, content_root)
        if cache_key in self.closures:
            return self.closures[cache_key]
        dependencies: set[Path] = set()
        pending = [module]
        parsed: set[Path] = set()
        while pending:
            current = pending.pop()
            dependencies.add(current)
            if current in parsed or not current.is_file():
                continue
            parsed.add(current)
            content = current.read_text(encoding="utf-8")
            payload = _safe_load_unique_keys(content)
            current_root = find_policy_repo_root(current) or content_root
            # Engine import recursion routes from each imported module's
            # own jurisdiction; static source-relation recursion can retain
            # the original caller root. Cover both with the shared resolver.
            resolution_roots = tuple(dict.fromkeys((content_root, current_root)))
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
            for imported in dict.fromkeys(imports):
                import_base = imported.split("#", 1)[0].strip().strip("\"'")
                if not _rulespec_import_prefix_static(imported) and import_base.endswith(
                    (".yaml", ".yml")
                ):
                    raise ValueError(
                        f"ambiguous import {imported!r} in "
                        f"{self.relative(current) or current.name}"
                    )
                candidates_found = False
                for resolution_root in resolution_roots:
                    candidates = tuple(
                        _rulespec_import_file_candidates_static(
                            imported,
                            rules_file=current,
                            policy_repo_path=resolution_root,
                            rulespec_dependency_roots=self.dependency_roots,
                        )
                    )
                    dependencies.update(candidates)
                    candidates_found |= bool(candidates)
                    target = _resolve_rulespec_import_file_static(
                        imported,
                        rules_file=current,
                        policy_repo_path=resolution_root,
                        rulespec_dependency_roots=self.dependency_roots,
                    )
                    if target is not None:
                        pending.append(target)
                if not candidates_found:
                    raise ValueError(
                        f"unclassifiable import {imported!r} in "
                        f"{self.relative(current) or current.name}"
                    )
            # Proof import hashes and source_relation verification read
            # targets even when they are absent from the imports block.
            for value in _yaml_strings(payload):
                target_ref = _parse_rulespec_target(value)
                if target_ref is None:
                    continue
                for resolution_root in resolution_roots:
                    dependencies.update(
                        _rulespec_target_file_candidates(
                            target_ref,
                            resolution_root,
                            rulespec_dependency_roots=self.dependency_roots,
                        )
                    )
                    target = _resolve_rulespec_target_file(
                        target_ref,
                        resolution_root,
                        rulespec_dependency_roots=self.dependency_roots,
                    )
                    if target is not None:
                        pending.append(target)
        self.closures[cache_key] = dependencies
        return dependencies

    def read_set(
        self, path: str, *, include_index_files: bool = True
    ) -> tuple[set[str], set[Path], bool]:
        _module_path(path)
        module = self.root / path
        content_root = find_policy_repo_root(module)
        if content_root is None:
            # Removed ledger entries can have an absent module. Their
            # lexical jurisdiction root still determines the search scope.
            content_root = self.root / PurePosixPath(path).parts[0]
        upstream_roots = (
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
        jurisdiction = PurePosixPath(path).parts[0]
        prefix_parts = jurisdiction.split("-")
        upstream_roots.update(
            self.root / "-".join(prefix_parts[:length])
            for length in range(1, len(prefix_parts) + 1)
        )
        dependencies = {
            module,
            module.with_name(f"{module.stem}.test.yaml"),
            self.root / DEFAULT_WAIVER_PATH,
            self.root / ".axiom" / "toolchain.toml",
        }
        dependencies.update(
            _git_identity_input_paths(
                self.root,
                git_boundary=self.git_boundary,
                git_config_inputs=self.git_config_inputs,
            )
        )
        if include_index_files:
            for upstream_root in upstream_roots:
                dependencies.update(self.modules(upstream_root))
        composition = False
        if module.is_file():
            dependencies.update(self.closure(module, content_root))
            companion = module.with_name(f"{module.stem}.test.yaml")
            if companion.is_file():
                # CI checks resolve absolute companion references against
                # their source modules, including targets the module never
                # imports. Inputs, outputs, relations and nested table rows
                # can all carry these references. Keep absent candidates as
                # well so deleting or adding a referenced module is covered.
                dependencies.update(self.closure(companion, content_root))
            payload = _safe_load_unique_keys(module.read_text(encoding="utf-8"))
            module_data = payload.get("module") if isinstance(payload, dict) else None
            composition = (
                isinstance(module_data, dict) and module_data.get("kind") == "composition"
            )
        if composition:
            # Program ownership scans every ProgramSpec and its transitive
            # scope before the owning program is composed. Country-wide
            # modules conservatively cover those discovered scope targets.
            program_root = self.root / RULESPEC_COMPOSITION_SPEC_ROOT
            dependencies.update(program_root.rglob("*.yaml"))
            for jurisdiction in jurisdiction_subdir_names(
                self.root, allow_composition_specs=True
            ):
                for candidate in self.modules(self.root / jurisdiction):
                    dependencies.update(self.closure(candidate, content_root))
        relative_dependencies = {
            relative
            for dependency in dependencies
            if (relative := self.relative(dependency)) is not None
        }
        return relative_dependencies, upstream_roots, composition


def validation_waiver_read_set(
    path: str,
    *,
    root: Path,
    rulespec_dependency_roots: Sequence[Path] = (),
) -> set[str]:
    """Return repository read dependencies plus missing resolver candidates."""

    with _rulespec_resolution_cache_scope():
        return _ReadSets(root, rulespec_dependency_roots).read_set(path)[0]


def select_validation_waiver_scope(
    paths: Sequence[str],
    *,
    root: Path,
    changed_paths: set[str],
    rulespec_dependency_roots: Sequence[Path] = (),
    shared_execution_inputs: Sequence[Path] = (),
) -> WaiverAuditScope:
    """Select only when every changed path has a proven dependency class."""

    ordered = tuple(sorted(set(paths)))
    root = Path(root).resolve()

    def full(reason: str) -> WaiverAuditScope:
        return WaiverAuditScope(
            "full",
            ordered,
            {path: [f"full audit fallback: {reason}"] for path in ordered},
            reason,
        )

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
    for changed in sorted(changed_paths):
        kind = _changed_path_kind(changed)
        if kind is None:
            return full(f"unclassified or shared validation input changed: {changed}")
        if any(
            (root / changed).is_relative_to(candidate)
            for candidate in shared_input_candidates
        ):
            return full(f"configured execution input changed: {changed}")
        if kind in {"module", "companion"} and monorepo_checkout_name(
            PurePosixPath(changed).parts[0]
        ) != root.name:
            return full(f"changed path belongs to another country: {changed}")
        try:
            _require_plain_changed_path(root, changed)
            module_relative = (
                changed.removesuffix(".test.yaml") + ".yaml"
                if kind == "companion"
                else changed
            )
            if kind == "companion":
                _require_plain_changed_path(root, module_relative)
                if not (root / module_relative).is_file():
                    return full(f"changed path is not an adjacent RuleSpec companion: {changed}")
            if kind in {"module", "companion"} and (root / module_relative).is_file():
                payload = _safe_load_unique_keys(
                    (root / module_relative).read_text(encoding="utf-8")
                )
                if not isinstance(payload, dict) or payload.get("format") != "rulespec/v1":
                    return full(f"changed path has no valid RuleSpec module: {changed}")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            if kind in {"module", "companion"}:
                return full(f"could not classify changed {kind} {changed}: {exc}")
            return full(str(exc))
        changed_kinds[changed] = kind
    meaningful = {
        path: kind for path, kind in changed_kinds.items() if kind != "inert"
    }
    reasons: dict[str, list[str]] = {}
    try:
        with _rulespec_routing_cache_scope(), _rulespec_resolution_cache_scope():
            # Checkout admission is a shared validation dependency even for
            # an empty selection: a symlink anywhere invalidates the tree.
            _reject_rulespec_checkout_symlinks(root, label="waiver audit checkout")
            reader = _ReadSets(root, rulespec_dependency_roots)
            config_changes = sorted(
                changed_paths.intersection(
                    relative
                    for config in reader.git_config_inputs
                    if (relative := reader.relative(config)) is not None
                )
            )
            if config_changes:
                return full(f"Git configuration input changed: {config_changes[0]}")
            if not changed_paths:
                return WaiverAuditScope("incremental", (), {})
            for path in ordered:
                # This cache contains only fixed-tree dependency discovery;
                # fingerprints retain their independent validation scopes.
                # Search-root membership is equivalent to materializing the
                # jurisdiction index for changed modules, including deletes.
                # Avoid building the same large string set for every waiver.
                read_set, upstream_roots, composition = reader.read_set(
                    path, include_index_files=False
                )
                selected_reasons: list[str] = []
                companion = path.removesuffix(".yaml") + ".test.yaml"
                for changed, kind in sorted(meaningful.items()):
                    if changed == path:
                        selected_reasons.append(f"module changed: {changed}")
                    elif changed == companion:
                        selected_reasons.append(f"companion changed: {changed}")
                    elif changed in read_set:
                        selected_reasons.append(f"validation reads changed path: {changed}")
                    elif kind == "module" and (
                        composition
                        or any(
                            (root / changed).is_relative_to(search_root)
                            for search_root in upstream_roots
                        )
                    ):
                        selected_reasons.append(f"validation searches changed module: {changed}")
                if selected_reasons:
                    reasons[path] = selected_reasons
    except (OSError, ValueError, yaml.YAMLError, _GitProbeError) as exc:
        return full(f"could not establish validation read set: {exc}")
    return WaiverAuditScope("incremental", tuple(reasons), reasons)
