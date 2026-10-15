"""Checkpoint the exact paired-obligation differential against pinned main.

Both revisions execute their own condition extraction, procedure predicate,
manual-prose context, numeric filter, and formula suppression. Source-only AST
loading excludes unrelated artifact-validation dependencies; every reachable
repository dependency is loaded from the same pinned revision. No equality of
the two revisions' extraction closures is assumed. Authenticated in-memory
Boolean guard reorders preserve successful classification results; call counts,
exceptions and resource-timeout equivalence are outside the measurement proof.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import inspect
import json
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_MODULE = "axiom_encode.harness.source_completeness"
SOURCE = "src/axiom_encode/harness/source_completeness.py"
LATEST_MAIN = "4f3547461300bfa70569f57c83728b2ad25d9d0b"
SPEC = importlib.util.spec_from_file_location(
    "_latest_main_measurement_shared", HERE / "reproduce-corpus.py"
)
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)
BOUNDARY_SPEC = importlib.util.spec_from_file_location(
    "_latest_main_clause_boundary_measurement", HERE / "clause-boundary-measurement.py"
)
boundary_measurement = importlib.util.module_from_spec(BOUNDARY_SPEC)
BOUNDARY_SPEC.loader.exec_module(boundary_measurement)


class BatchTimeout(Exception):
    pass


def timed_out(signum, frame):
    raise BatchTimeout


def encoded(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


class RevisionLoader:
    """Load exact reachable top-level definitions without unrelated imports."""

    def __init__(self, revision: str | None, key: str, *, accelerate: bool = True):
        self.revision = revision
        self.key = key
        self.accelerate = accelerate
        self.bindings = {}
        self.sources = {}
        self.cache = {}

    def read(self, module_name: str) -> bytes:
        path = "src/" + module_name.replace(".", "/") + ".py"
        if path not in self.sources:
            source = (
                subprocess.check_output(
                    ["git", "show", self.revision + ":" + path], cwd=ROOT
                )
                if self.revision
                else (ROOT / path).read_bytes()
            )
            self.sources[path] = source
        return self.sources[path]

    def load(self, module_name: str, roots: tuple[str, ...]):
        identity = (module_name, tuple(sorted(roots)))
        assert identity not in self.cache or self.cache[identity] is not None, (
            "Recursive source dependency",
            identity,
        )
        if identity in self.cache:
            return self.cache[identity]
        self.cache[identity] = None
        source = self.read(module_name)
        execution = source
        acceleration = None
        boundary_acceleration = None
        if module_name == SOURCE_MODULE and self.accelerate:
            execution, acceleration = shared.accelerated_guidance_source(source)
            execution, boundary_acceleration = (
                boundary_measurement.accelerated_clause_boundaries_source(execution)
            )
            assert (
                boundary_acceleration["source_sha256"]
                == acceleration["execution_source_sha256"]
            )
        tree = ast.parse(execution)
        available = shared.bindings(tree)
        pending = list(roots)
        selected = set()
        while pending:
            name = pending.pop()
            if name in selected:
                continue
            assert name in available or name not in roots, (module_name, name)
            if name not in available:
                continue
            selected.add(name)
            node = available[name]
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                pending.extend(
                    child.id
                    for child in ast.walk(node)
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
                )
        unique_name = (
            "axiom_encode.harness._latest_main_"
            + self.key
            + "_"
            + shared.digest(repr(identity).encode())[:16]
        )
        module = types.ModuleType(unique_name)
        module.__package__ = module_name.rpartition(".")[0]
        sys.modules[unique_name] = module
        statements = [ast.parse("from __future__ import annotations").body[0]]
        selected_ids = {id(available[name]) for name in selected}
        for node in tree.body:
            if id(node) not in selected_ids:
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                aliases = [
                    alias
                    for alias in node.names
                    if (alias.asname or alias.name) in selected
                ]
                if isinstance(node, ast.ImportFrom):
                    imported_name = node.module or ""
                    if node.level:
                        imported_name = importlib.util.resolve_name(
                            "." * node.level + imported_name, module.__package__
                        )
                    if imported_name.startswith("axiom_encode."):
                        dependency = self.load(
                            imported_name, tuple(alias.name for alias in aliases)
                        )
                        for alias in aliases:
                            module.__dict__[alias.asname or alias.name] = getattr(
                                dependency, alias.name
                            )
                        continue
                    node = ast.ImportFrom(
                        module=node.module, names=aliases, level=node.level
                    )
                else:
                    assert all(
                        not alias.name.startswith("axiom_encode") for alias in aliases
                    ), "Repository module imports require an explicit source adapter"
                    node = ast.Import(names=aliases)
            else:
                assert not any(
                    isinstance(child, ast.ImportFrom)
                    and (child.level or (child.module or "").startswith("axiom_encode"))
                    for child in ast.walk(node)
                ), "Function-local repository import outside the pinned loader"
                assert not any(
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id in {"globals", "__import__", "eval", "exec"}
                    for child in ast.walk(node)
                ), "Dynamic global dependency outside the static loader"
            statements.append(node)
        compiled = ast.fix_missing_locations(
            ast.Module(body=statements, type_ignores=[])
        )
        exec(compile(compiled, "<pinned-" + module_name + ">", "exec"), module.__dict__)
        prior = self.bindings.get(module_name, {})
        self.bindings[module_name] = {
            "source_sha256": shared.digest(source),
            "execution_source_sha256": shared.digest(execution),
            "selected_binding_ast_sha256": prior.get("selected_binding_ast_sha256", {})
            | {
                name: shared.digest(shared.node_text(available[name]).encode())
                for name in sorted(selected)
            },
            "guidance_classifier_reordering": acceleration,
            "clause_boundary_reordering": boundary_acceleration,
        }
        self.cache[identity] = module
        return module

    def receipt(self) -> dict:
        return {
            "guidance_classifier_accelerated": self.accelerate,
            "clause_boundary_guards_accelerated": self.accelerate,
            "repository_sources": {
                path: shared.digest(source)
                for path, source in sorted(self.sources.items())
            },
            "modules": self.bindings,
        }


def load_revisions(*, accelerate: bool = True):
    roots = (
        "recognize_source_structure",
        "_source_exception_or_applicability_matches",
        "_source_exception_branches",
        "_source_exception_requires_paired_witness",
        "_source_formula_branches",
        "_source_has_arithmetic_comparison_condition",
        "_EXCEPTION_LANGUAGE",
    )
    prefix = "accelerated_" if accelerate else "original_"
    loaders = {
        "latest_main": RevisionLoader(
            LATEST_MAIN, prefix + "main", accelerate=accelerate
        ),
        "final": RevisionLoader(None, prefix + "final", accelerate=accelerate),
    }
    modules = {}
    for key, loader in loaders.items():
        available = shared.bindings(ast.parse(loader.read(SOURCE_MODULE)))
        optional = tuple(
            name
            for name in ("_us_manual_prose_scope", "_is_us_manual_citation_path")
            if name in available
        )
        modules[key] = loader.load(SOURCE_MODULE, roots + optional)
    return modules, loaders


@contextlib.contextmanager
def clause_span_cache(module, *, enabled: bool):
    """Reuse executed-generator triples within one fixed-scope obligation call."""

    if not enabled:
        yield
        return
    original = module._source_clause_spans
    entries = []

    def cached(source_text, *, branches):
        for known_text, known_branches, triples in entries:
            if source_text is known_text and branches is known_branches:
                return iter(triples)
        triples = tuple(original(source_text, branches=branches))
        # Retain both argument objects so their identities cannot be reused.
        entries.append((source_text, branches, triples))
        return iter(triples)

    module._source_clause_spans = cached
    try:
        yield
    finally:
        module._source_clause_spans = original


def clause_cache_receipt(loaders: dict) -> dict:
    helper = shared.bindings(ast.parse(Path(__file__).read_bytes()))[
        "clause_span_cache"
    ]
    return {
        "enabled": True,
        "helper_ast_sha256": shared.digest(shared.node_text(helper).encode()),
        "original_generator_ast_sha256": {
            key: shared.digest(
                shared.node_text(
                    shared.bindings(ast.parse(loader.read(SOURCE_MODULE)))[
                        "_source_clause_spans"
                    ]
                ).encode()
            )
            for key, loader in loaders.items()
        },
        "executed_generator_ast_sha256": {
            key: loader.bindings[SOURCE_MODULE]["selected_binding_ast_sha256"][
                "_source_clause_spans"
            ]
            for key, loader in loaders.items()
        },
        "key": "exact source-text and branches object identity, with strong references",
        "lifetime": "one revision and body within its fixed manual context; original restored in finally",
        "result": "immutable ordered triples from the executed generator, exposed by a fresh iterator for every call",
        "scope": "ordered successful classification results; call count, exception and resource-timeout equivalence are not asserted",
    }


def obligations(
    module, text: str, citation: str, *, latest: bool, clause_cache: bool = True
) -> dict:
    scope = getattr(module, "_us_manual_prose_scope", None)
    with (
        scope(citation) if scope else contextlib.nullcontext(),
        clause_span_cache(module, enabled=clause_cache),
    ):
        # The production extractor iterates precisely these matches. An empty
        # match tuple proves an empty exception set without formula extraction.
        if not module._source_exception_or_applicability_matches(text):
            return {}
        branches = module.recognize_source_structure(
            text, corpus_citation_path=citation
        )
        options = {
            "branches": branches,
            "active_branches": branches,
            "deferred_paths": set(),
        }
        extraction = dict(options)
        if latest:
            extraction["numeric_limits_only"] = module._is_us_manual_citation_path(
                citation
            )
        exceptions = module._source_exception_branches(text, **extraction)
        if any(
            module._EXCEPTION_LANGUAGE.search(branch.text) is None
            and module._source_has_arithmetic_comparison_condition(branch.text)
            for branch in exceptions
        ):
            formula_options = dict(options)
            if (
                "corpus_citation_path"
                in inspect.signature(module._source_formula_branches).parameters
            ):
                formula_options["corpus_citation_path"] = citation
            formulas = module._source_formula_branches(text, **formula_options)
            exceptions = module._source_exception_branches(
                text, formula_branches=formulas, **extraction
            )
        result = {}
        for branch in exceptions:
            predicate_options = {"branch_path": branch.path}
            if not latest:
                predicate_options.update(
                    corpus_citation_path=citation, source_text=text
                )
            if module._source_exception_requires_paired_witness(
                branch.text, **predicate_options
            ):
                identity = (
                    tuple(branch.path),
                    branch.start,
                    branch.end,
                    shared.digest(branch.text.encode()),
                )
                result[identity] = branch
        return result


def read_ledger(path: Path, header: dict) -> tuple[tuple[int, int], dict]:
    per_file = {}
    position = (-1, -1)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded(header))
        return position, per_file
    with path.open("r+b") as stream:
        assert json.loads(stream.readline()) == header, (
            "Ledger inputs or source changed"
        )
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                stream.truncate(offset)
                break
            row = json.loads(line)
            next_position = (row["file_number"], row["index"])
            assert next_position > position, "Ledger rows must advance strictly"
            position = next_position
            summary = per_file.setdefault(
                row["file"], {"rows": 0, "removed": 0, "added": 0}
            )
            summary["rows"] += 1
            for action in ("removed", "added"):
                summary[action] += sum(
                    change["action"] == action for change in row["delta"]
                )
    return position, per_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--time-limit", type=int, default=525)
    options = parser.parse_args()
    assert 1 <= options.time_limit <= 540
    signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, options.time_limit)
    started = time.monotonic()
    output = None
    try:
        inventory_bytes = (HERE / "corpus-source-binding.json").read_bytes()
        inventory = json.loads(inventory_bytes)
        modules, loaders = load_revisions()
        header = {
            "kind": "complete-latest-main-paired-obligation-differential",
            "latest_main_revision": LATEST_MAIN,
            "corpus_revision": inventory["git_commit"],
            "inventory_sha256": shared.digest(inventory_bytes),
            "execution": {key: loader.receipt() for key, loader in loaders.items()},
            "clause_cache": clause_cache_receipt(loaders),
            "audit_sha256": {
                name: shared.digest((HERE / name).read_bytes())
                for name in (
                    "clause-boundary-measurement.py",
                    "reproduce-latest-main.py",
                    "reproduce-corpus.py",
                    "validate-measurement.py",
                )
            },
            "scope": "raw corpus source obligations with no artifact-specific deferrals, independently covered paths, or companion witnesses; each revision retains its own complete condition extraction and formula suppression",
        }
        validation_bytes = (HERE / "measurement-validation.json").read_bytes()
        validation = json.loads(validation_bytes)
        assert validation["latest_main_revision"] == LATEST_MAIN
        assert validation["inventory_sha256"] == header["inventory_sha256"]
        assert validation["audit_sha256"] == header["audit_sha256"]
        assert validation["accelerated_execution"] == header["execution"]
        assert validation["clause_cache"] == header["clause_cache"]
        header["measurement_validation_sha256"] = shared.digest(validation_bytes)
        position, per_file = read_ledger(options.ledger, header)
        rows = sum(value["rows"] for value in per_file.values())
        output = options.ledger.open("ab", buffering=0)
        for file_number, (name, entry) in enumerate(inventory["files"].items()):
            content = (options.corpus_root / name).read_bytes()
            assert (
                len(content) == entry["bytes"]
                and shared.digest(content) == entry["sha256"]
            ), name
            if file_number < position[0]:
                continue
            for index, line in enumerate(content.decode().splitlines()):
                if (file_number, index) <= position or not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("body") or row.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                citation = row.get("citation_path", "")
                before = obligations(
                    modules["latest_main"], text, citation, latest=True
                )
                after = obligations(modules["final"], text, citation, latest=False)
                delta = []
                for action, identities in (
                    ("removed", before.keys() - after.keys()),
                    ("added", after.keys() - before.keys()),
                ):
                    for path, begin, end, clause_digest in sorted(identities):
                        delta.append(
                            {
                                "action": action,
                                "path": list(path),
                                "start": begin,
                                "end": end,
                                "clause_sha256": clause_digest,
                            }
                        )
                output.write(
                    encoded(
                        {
                            "file_number": file_number,
                            "file": name,
                            "index": index,
                            "body_sha256": shared.digest(text.encode()),
                            "citation": citation,
                            "delta": delta,
                        }
                    )
                )
                rows += 1
                if rows % 100 == 0:
                    print(
                        f"PROGRESS rows={rows} file={file_number} index={index} seconds={time.monotonic() - started:.1f}",
                        flush=True,
                    )
        output.close()
        output = None
        _, per_file = read_ledger(options.ledger, header)
        counts = {
            key: sum(value[key] for value in per_file.values())
            for key in ("rows", "removed", "added")
        }
        assert counts["rows"] == 39375
        assert (HERE / "corpus-source-binding.json").read_bytes() == inventory_bytes
        for path, source in loaders["final"].sources.items():
            assert (ROOT / path).read_bytes() == source, (
                "Final source changed during scan"
            )
        checksum = hashlib.sha256()
        with options.ledger.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                checksum.update(chunk)
        summary = header | {
            "files": inventory["file_count"],
            "input_bytes": inventory["total_bytes"],
            "counts": counts,
            "per_file": per_file,
            "ledger_sha256": checksum.hexdigest(),
            "ledger_bytes": options.ledger.stat().st_size,
        }
        options.summary.parent.mkdir(parents=True, exist_ok=True)
        options.summary.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
        print("COMPLETE " + json.dumps(counts, sort_keys=True), flush=True)
        return 0
    except BatchTimeout:
        print(
            "PAUSED after bounded batch; repeat the same command to resume", flush=True
        )
        return 75
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if output is not None:
            output.close()


if __name__ == "__main__":
    raise SystemExit(main())
