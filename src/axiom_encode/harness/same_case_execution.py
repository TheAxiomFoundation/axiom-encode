"""Invocation-local, pinned Explain evidence; never a source-proof substitute.

Only an ordinary single-root module closure is supported. Unsupported callers
retain their existing validation, without receiving execution evidence.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator

import yaml

from axiom_encode.engine_pin_diff import effective_versions
from axiom_encode.formula_references import (
    formula_identifiers,
    local_reference_name,
    relation_source_names,
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def case_key(case: dict[str, Any]) -> str | None:
    def supported(value: Any) -> bool:
        if value is None or type(value) in (str, bool, int, float):
            return True
        if isinstance(value, list):
            return all(supported(v) for v in value)
        if isinstance(value, dict):
            return all(type(k) is str and supported(v) for k, v in value.items())
        return False

    if not supported(case):
        return None
    try:
        return digest(
            json.dumps(
                case, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        )
    except (ValueError, TypeError):
        return None


def _payload(raw: bytes) -> dict[str, Any]:
    from .validator_pipeline import _safe_load_unique_keys

    if any(
        isinstance(t, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken))
        for t in yaml.scan(raw)
    ):
        raise ValueError("source aliases are outside the execution snapshot contract")
    value = _safe_load_unique_keys(raw.decode())
    if not isinstance(value, dict) or value.get("format") != "rulespec/v1":
        raise ValueError("ordinary RuleSpec required")
    module = value.get("module")
    if not isinstance(module, dict) or module.get("kind") == "composition":
        raise ValueError("composition cannot supply same-case evidence")
    imports = value.get("imports", [])
    if not isinstance(imports, list) or not all(isinstance(x, str) for x in imports):
        raise ValueError("literal import list required")
    return value


def _formula_names(text: str, *, relation_predicate: bool = False) -> set[str] | None:
    from . import source_completeness as s

    if relation_predicate:
        return formula_identifiers(text, relation_predicate=True)
    node = s._parse_first_formula_branch_node(text)
    if node is not None:
        if text[: node.start].strip() or text[node.end :].strip():
            return None
        parts = [_formula_names(part) for part in (*node.selectors, *node.choices)]
        return None if any(p is None for p in parts) else set().union(*parts)
    expression = s._parse_formula_expression(text)
    return (
        None
        if expression is None
        else {n.id for n in ast.walk(expression) if isinstance(n, ast.Name)}
    )


def _has_consumer(
    pipe: Any, content: str, payload: dict[str, Any], imports: set[str]
) -> bool:
    """Cheap proof-bound dependency test before any clone or compilation."""
    from . import source_completeness as s

    sources = pipe._proof_source_texts_for_rulespec_content(
        content, source_texts=pipe._source_texts_for_rulespec_content(content)
    )
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list) or not all(
        isinstance(r, dict) for r in raw_rules
    ):
        return False
    rules = {r["name"]: r for r in raw_rules if isinstance(r.get("name"), str)}
    edges: dict[str, set[str]] = {}
    for name, rule in rules.items():
        names: set[str] = set()
        versions = effective_versions(rule)
        if not isinstance(versions, list):
            return False
        for version in versions:
            if not isinstance(version, dict) or not isinstance(
                version.get("formula"), str
            ):
                continue
            found = _formula_names(
                version["formula"],
                relation_predicate=rule.get("kind") == "derived_relation",
            )
            if found is not None:
                names.update(found)
        names.update(
            local_reference_name(source)
            for source in relation_source_names(rule.get("derived_relation"))
        )
        edges[name] = names

    def reaches(name: str, seen: set[str]) -> bool:
        if name in seen:
            return False
        return bool(edges.get(name, set()) & imports) or any(
            reaches(n, seen | {name}) for n in edges.get(name, set()) if n in rules
        )

    for name, rule in rules.items():
        if rule.get("kind") != "derived" or not reaches(name, set()):
            continue
        metadata = rule.get("metadata")
        proof = metadata.get("proof") if isinstance(metadata, dict) else None
        atoms = proof.get("atoms") if isinstance(proof, dict) else None
        if not isinstance(atoms, list):
            continue
        for atom in atoms:
            if not isinstance(atom, dict) or atom.get("kind") != "formula":
                continue
            source = atom.get("source")
            if not isinstance(source, dict):
                continue
            citation, excerpt = (
                source.get("corpus_citation_path"),
                source.get("excerpt"),
            )
            if not isinstance(citation, str):
                continue
            body = sources.get(citation) if sources else None
            if (
                not isinstance(body, str)
                or not isinstance(excerpt, str)
                or not excerpt
                or body.count(excerpt) != 1
            ):
                continue
            if s.source_states_explicit_computation(excerpt):
                return True
            rows = s._unconflicted_printed_rows(body, ())
            start = body.index(excerpt)
            if rows and any(
                start <= a < b <= start + len(excerpt) for a, b in rows.additions
            ):
                return True
    return False


@dataclass(frozen=True)
class ClosureFile:
    relative: str
    original: Path
    raw: bytes


@dataclass(frozen=True)
class CaseExecution:
    case_sha256: str
    request_json: str
    response_json: str
    case_index: int | None = None


@dataclass
class ExecutionCapture:
    """Private mutable staging; exposed evidence is frozen after all tests pass."""

    original_root: Path
    snapshot_root: Path
    root_relative: str
    files: tuple[ClosureFile, ...]
    binary: Path
    binary_sha256: str
    engine_pin: str
    clone_seconds: float
    git_head: str
    artifact: Path | None = None
    artifact_sha256: str | None = None
    records: list[CaseExecution] = field(default_factory=list)

    def fresh(self) -> bool:
        try:
            return (
                digest(self.binary.read_bytes()) == self.binary_sha256
                and all(
                    f.original.read_bytes()
                    == f.raw
                    == (self.snapshot_root / f.relative).read_bytes()
                    for f in self.files
                )
                and (
                    self.artifact is None
                    or digest(self.artifact.read_bytes()) == self.artifact_sha256
                )
            )
        except OSError:
            return False

    def compiled(self, artifact: Path) -> None:
        self.records.clear()
        self.artifact = artifact
        self.artifact_sha256 = digest(artifact.read_bytes())

    def stage(
        self,
        case: dict[str, Any],
        request: dict[str, Any],
        response: dict[str, Any],
        binary: Path,
        artifact: Path,
        case_index: int | None = None,
    ) -> None:
        if binary != self.binary or artifact != self.artifact or not self.fresh():
            return
        if request.get("mode") != "explain" or response.get("metadata") != {
            "requested_mode": "explain",
            "actual_mode": "explain",
            "fallback_reason": None,
        }:
            return
        # Initially one Person query, no nested/table/assessment transport.
        queries = request.get("queries")
        rows = response.get("results")
        if (
            not isinstance(queries, list)
            or len(queries) != 1
            or not isinstance(rows, list)
            or len(rows) != 1
        ):
            return
        q, row = queries[0], rows[0]
        if any(
            row.get(k) != q.get(k) for k in ("entity_id", "period", "assessment_date")
        ):
            return
        trace = row.get("trace")
        if not isinstance(trace, dict) or any(
            not isinstance(n, dict)
            or n.get("id") != key
            or n.get("entity_id") != q.get("entity_id")
            for key, n in trace.items()
        ):
            return
        key = case_key(case)
        if key is None:
            return
        self.records.append(
            CaseExecution(
                key,
                json.dumps(request, sort_keys=True),
                json.dumps(response, sort_keys=True),
                case_index if type(case_index) is int and case_index > 0 else None,
            )
        )

    def evidence(self) -> SameCaseEvidence | None:
        if not self.records or not self.fresh() or self.artifact is None:
            return None
        return SameCaseEvidence(
            self.files,
            self.root_relative,
            self.snapshot_root,
            self.git_head,
            self.binary_sha256,
            self.engine_pin,
            self.artifact.read_bytes(),
            tuple(self.records),
        )


@dataclass(frozen=True)
class SameCaseEvidence:
    files: tuple[ClosureFile, ...]
    root_relative: str
    snapshot_root: Path
    git_head: str
    binary_sha256: str
    engine_pin: str
    artifact_bytes: bytes
    records: tuple[CaseExecution, ...]

    def bound(self, content: str, imports: Any) -> bool:
        originals = {str(f.original): f for f in self.files}
        root = next((f for f in self.files if f.relative == self.root_relative), None)
        try:
            if (
                root is None
                or root.raw.decode() != content
                or any(
                    f.original.read_bytes() != f.raw
                    or (self.snapshot_root / f.relative).read_bytes() != f.raw
                    for f in self.files
                )
            ):
                return False
            for record in imports:
                f = originals.get(record.resolved_path)
                if (
                    f is None
                    or record.content.encode() != f.raw
                    or record.content_sha256 != digest(f.raw)
                ):
                    return False
            return True
        except OSError:
            return False

    def values(
        self, case: dict[str, Any], content: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Canonical direct imports only; local trace outputs corroborate replay."""
        key = case_key(case)
        if key is None:
            return None
        matches = [r for r in self.records if r.case_sha256 == key]
        if len(matches) != 1:
            return None
        payload = _payload(content.encode())
        imports = payload.get("imports", [])
        aliases = [x.rsplit("#", 1)[-1] for x in imports]
        local = {r.get("name") for r in payload.get("rules", [])} | {
            i.get("name") for i in payload.get("inputs", [])
        }
        if len(aliases) != len(set(aliases)) or local & set(aliases):
            return None
        artifact = json.loads(self.artifact_bytes)
        compiled = artifact.get("program", {}).get("derived", [])
        by_id = {r.get("id"): r for r in compiled}
        if len(by_id) != len(compiled):
            return None
        response = json.loads(matches[0].response_json)
        trace = response["results"][0]["trace"]
        request = json.loads(matches[0].request_json)
        query = request["queries"][0]
        values = {}
        for ref, node in trace.items():
            declaration = by_id.get(ref)
            if (
                declaration is None
                or declaration.get("entity") != "Person"
                or declaration.get("period") != "Year"
            ):
                continue
            if node.get("entity_id") != query["entity_id"]:
                return None
            if declaration.get("semantics") == "judgment":
                if (
                    node.get("kind") != "judgment"
                    or node.get("unit") != declaration.get("unit")
                    or declaration.get("unit") is not None
                    or ("dtype" in node and node["dtype"] != declaration.get("dtype"))
                ) or node.get("outcome") not in {
                    "holds",
                    "not_holds",
                }:
                    continue
                value = node["outcome"] == "holds"
            else:
                if (
                    node.get("kind") != "scalar"
                    or node.get("dtype") != declaration.get("dtype")
                    or node.get("unit") != declaration.get("unit")
                ):
                    continue
                wire = node.get("value", {})
                if wire.get("kind") == "integer" and type(wire.get("value")) is int:
                    value = Decimal(wire["value"])
                elif wire.get("kind") == "decimal" and isinstance(
                    wire.get("value"), str
                ):
                    try:
                        value = Decimal(wire["value"])
                    except InvalidOperation:
                        continue
                    if not value.is_finite():
                        continue
                else:
                    continue
            values[ref] = value
        module = self.root_relative.removesuffix(".yaml").removesuffix(".yml")
        module = self.snapshot_module(module)
        imported = {
            alias: values[ref]
            for alias, ref in zip(aliases, imports, strict=True)
            if ref in values
        }
        local_values = {
            ref.split("#", 1)[1]: value
            for ref, value in values.items()
            if ref.startswith(module + "#")
        }
        return imported, local_values

    @staticmethod
    def snapshot_module(relative: str) -> str:
        # Relative paths are below the country checkout, including jurisdiction.
        jurisdiction, path = relative.split("/", 1)
        return jurisdiction + ":" + path


def _snapshot_plan(pipe: Any, rules_file: Path) -> tuple | None:
    from . import validator_pipeline as v

    if not pipe.require_complete_source_unit or pipe._declared_engine_pin() is None:
        return None
    try:
        root = pipe._rulespec_compile_roots()[0]
        content = rules_file.read_text()
        payload = _payload(content.encode())
        imports = payload.get("imports", [])
        companion = pipe._rulespec_test_path(rules_file)
        if not imports or pipe.rulespec_dependency_roots or not companion.is_file():
            return None
        exports = pipe._complete_source_unit_import_bindings(rules_file)
        derived = set()
        for export in exports:
            provider = _payload(export.content.encode())
            if any(
                isinstance(r, dict)
                and r.get("name") == export.export_name
                and r.get("kind") == "derived"
                for r in provider.get("rules", [])
            ):
                derived.add(export.export_name)
        if not derived or not _has_consumer(pipe, content, payload, derived):
            return None
        pending = [rules_file]
        files: dict[Path, ClosureFile] = {}
        while pending:
            current = v.validate_rulespec_context_file(
                pending.pop(), pipe.policy_repo_path
            )
            if current in files:
                continue
            raw = current.read_bytes()
            parsed = _payload(raw)
            files[current] = ClosureFile(
                current.relative_to(root).as_posix(), current, raw
            )
            for imported in parsed.get("imports", []):
                if "#" not in imported or v._parse_rulespec_target(imported) is None:
                    raise ValueError("canonical exported imports required")
                target = v._resolve_rulespec_import_file_static(
                    imported,
                    rules_file=current,
                    policy_repo_path=pipe.policy_repo_path,
                    rulespec_dependency_roots=(),
                )
                if target is None:
                    raise ValueError("unresolved import")
                target.relative_to(pipe.policy_repo_path)
                pending.append(target)
        companion = v.validate_rulespec_context_file(companion, pipe.policy_repo_path)
        files[companion] = ClosureFile(
            companion.relative_to(root).as_posix(), companion, companion.read_bytes()
        )
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        binary = pipe._axiom_rules_binary()
        binary_sha = digest(binary.read_bytes())
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        yaml.YAMLError,
        subprocess.SubprocessError,
    ):
        return None
    return root, files, head, binary, binary_sha


def _materialize_snapshot(
    pipe: Any, rules_file: Path, plan: tuple, directory: str
) -> ExecutionCapture | None:
    root, files, head, binary, binary_sha = plan
    snapshot_root = Path(directory).resolve() / root.name
    start = time.monotonic()
    cloned = subprocess.run(
        [
            "git",
            "clone",
            "--no-local",
            "--no-hardlinks",
            "--no-checkout",
            str(root),
            str(snapshot_root),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if cloned.returncode:
        return None
    actual_head = subprocess.check_output(
        ["git", "-C", str(snapshot_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if (
        actual_head != head
        or not (snapshot_root / ".git").is_dir()
        or (snapshot_root / ".git/objects/info/alternates").exists()
        or (snapshot_root / ".git/commondir").exists()
        or any(p.is_symlink() for p in snapshot_root.rglob("*"))
    ):
        return None
    for item in files.values():
        target = snapshot_root / item.relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(item.raw)
    inventory = {
        p.relative_to(snapshot_root).as_posix()
        for p in snapshot_root.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(snapshot_root).parts
    }
    if inventory != {f.relative for f in files.values()}:
        return None
    capture = ExecutionCapture(
        root,
        snapshot_root,
        rules_file.relative_to(root).as_posix(),
        tuple(files.values()),
        binary,
        binary_sha,
        pipe._declared_engine_pin().sha,
        time.monotonic() - start,
        head,
    )
    if not capture.fresh():
        return None
    return capture


@contextmanager
def compilation_snapshot(
    pipe: Any, rules_file: Path
) -> Iterator[ExecutionCapture | None]:
    plan = _snapshot_plan(pipe, rules_file)
    if plan is None:
        yield None
        return
    with tempfile.TemporaryDirectory(prefix="axiom-execution-snapshot-") as directory:
        try:
            capture = _materialize_snapshot(pipe, rules_file, plan, directory)
        except (OSError, ValueError, subprocess.SubprocessError):
            capture = None
        # Never catch exceptions raised by the caller's validation body.
        yield capture


def formula_witnesses(
    evidence: SameCaseEvidence, branch: Any, **kwargs: Any
) -> set[tuple[str, str]]:
    """Replay only this exact case, with private caches and observed imports.

    No imported value enters the legacy caches, case maps, boundary evaluator or
    expected-output bridge. Trace membership alone is not target reachability.
    """
    from . import source_completeness as s

    content = kwargs.pop("content")
    witnesses: set[tuple[str, str]] = set()
    principal = kwargs["principal_rules"]
    for name in kwargs["rule_names"]:
        for case in kwargs["asserted_by_rule"].get(name, ()):
            bound = evidence.values(case, content)
            if bound is None:
                continue
            imported, observed_local = bound
            constants = kwargs["formula_environment"]
            if not imported or imported.keys() & constants.keys():
                continue
            environment = {**constants, **imported}
            # This evaluator never substitutes assertions for missing imports.
            replayed = s._case_dependency_environment(
                principal,
                case,
                formula_environment=environment,
                require_asserted_value=False,
            )
            dependencies = {
                key: value
                for key, value in replayed.items()
                if key in observed_local
                and s._formula_runtime_values_equal(value, observed_local[key])
                and s._asserted_formula_runtime_values_equal(
                    principal[key], value, s._test_case_asserted_output_value(case, key)
                )
            }
            if name not in dependencies:
                continue
            execution = s._case_formula_execution(
                principal[name],
                case,
                formula_environment=environment,
                dependency_environment=dependencies,
            )
            if execution is None:
                continue
            evaluation = s._case_formula_identifier_environment(
                case,
                formula_environment=environment,
                dependency_environment=dependencies,
            )
            if evaluation is None:
                continue
            # Inspect actually selected local operations, not all trace nodes.
            pending = [execution]
            reached_imports: set[str] = set()
            visited = {name}
            while pending:
                current = pending.pop()
                reached = s._reached_formula_expression_identifier_names(
                    current.leaf, environment=evaluation
                )
                for step in current.trace:
                    for selector in step.selectors:
                        reached.update(
                            s._reached_formula_expression_identifier_names(
                                selector, environment=evaluation
                            )
                        )
                reached_imports.update(reached & imported.keys())
                for local in (reached & principal.keys()) - visited:
                    visited.add(local)
                    child = s._case_formula_execution(
                        principal[local],
                        case,
                        formula_environment=environment,
                        dependency_environment=dependencies,
                    )
                    if child is not None:
                        pending.append(child)
            if not reached_imports:
                continue
            private = dict(kwargs)
            private.update(
                rule_names={name},
                asserted_by_rule={name: [case]},
                formula_environment=environment,
                dependency_cache={id(case): dependencies},
                execution_cache={(name, id(case)): execution},
            )
            witnesses.update(s._formula_branch_test_witnesses(branch, **private))
    return witnesses
