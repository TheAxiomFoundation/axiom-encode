"""Reproduce the source-bound two-clause paired-obligation differential.

All corpus input files are freshly authenticated and read. An AST dependency
proof limits possible differences from historical main to the explicit closed
exemption. Exact extraction and formula suppression run on every possible row.
The current-main comparison covers those selected rows only.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = "src/axiom_encode/harness/source_completeness.py"
HISTORICAL_MAIN = "86a1f911a338cb071e0ed660d1282714034b6dc4"
CURRENT_MAIN = "c2ee166fd1813392022fe5cf4c1e08ae1f09aba0"
PARENT = "f824da90344c6dd5b59811b4f3121f888b0adf85"
sys.path.insert(0, str(ROOT / "src"))
EXEMPTION_GUARD = ast.parse(
    """if (
        corpus_citation_path == _MANUAL_PROCEDURE_CITATION
        and _source_condition_is_manual_procedure(collapsed)
        and (
            _collapse_text(source_text or text).lower() == collapsed.lower()
            or hashlib.sha256(
                _collapse_text(source_text).lower().encode("utf-8")
            ).hexdigest() == _MANUAL_PROCEDURE_SOURCE_SHA256
        )
    ):
        return False
"""
).body[0]


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def node_text(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def normalized(source: bytes, *, numeric_filter: bool = False) -> ast.Module:
    tree = ast.parse(source)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    predicate = functions["_source_exception_requires_paired_witness"]
    extra_arguments = {"corpus_citation_path", "source_text"}
    for index in reversed(range(len(predicate.args.kwonlyargs))):
        if predicate.args.kwonlyargs[index].arg in extra_arguments:
            assert node_text(predicate.args.kw_defaults[index]) == node_text(
                ast.Constant("")
            )
            predicate.args.kwonlyargs.pop(index)
            predicate.args.kw_defaults.pop(index)
    guards = [
        node
        for node in predicate.body
        if (
            isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Name)
                and child.id == "_source_condition_is_manual_procedure"
                for child in ast.walk(node.test)
            )
        )
    ]
    assert len(guards) <= 1
    for guard in guards:
        assert node_text(guard) == node_text(EXEMPTION_GUARD), "Unknown exemption guard"
        predicate.body.remove(guard)
    if numeric_filter:
        exception = functions["_source_exception_branches"]
        arguments = [arg.arg for arg in exception.args.kwonlyargs]
        index = arguments.index("numeric_limits_only")
        assert node_text(exception.args.kw_defaults[index]) == node_text(
            ast.Constant(False)
        )
        exception.args.kwonlyargs.pop(index)
        exception.args.kw_defaults.pop(index)
        expected = ast.parse(
            "if numeric_limits_only and not _STATED_NUMERIC_LIMIT.search(branch_text):\n"
            "    continue\n"
        ).body[0]
        removed = 0
        for parent in ast.walk(exception):
            if isinstance(getattr(parent, "body", None), list):
                kept = []
                for node in parent.body:
                    if node_text(node) == node_text(expected):
                        removed += 1
                    else:
                        kept.append(node)
                parent.body[:] = kept
        assert removed == 1
    # The parent adds numeric metadata to formula branches. Exception
    # suppression observes only their start/end spans; it never reads metadata.
    formula = functions["_source_formula_branches"]
    for index in reversed(range(len(formula.args.kwonlyargs))):
        if formula.args.kwonlyargs[index].arg == "corpus_citation_path":
            formula.args.kwonlyargs.pop(index)
            formula.args.kw_defaults.pop(index)
    formula.body[:] = [
        node
        for node in formula.body
        if not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "worksheet_labels"
                for target in node.targets
            )
        )
    ]
    for call in ast.walk(formula):
        if isinstance(call, ast.Call):
            call.keywords[:] = [
                keyword
                for keyword in call.keywords
                if keyword.arg != "structural_numeric_spans"
            ]
    assert not any(
        isinstance(node, ast.Name) and node.id == "worksheet_labels"
        for node in ast.walk(formula)
    )
    return tree


def bindings(tree: ast.Module) -> dict[str, ast.AST]:
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            result[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            result[node.target.id] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                result[alias.asname or alias.name] = node
    return result


def scope_proof(old: bytes, final: bytes, *, numeric_filter: bool = False) -> dict:
    before = bindings(normalized(old, numeric_filter=numeric_filter))
    after = bindings(normalized(final))
    helper = after["_source_condition_is_manual_procedure"]
    helper_body = [
        node
        for node in helper.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    assert [argument.arg for argument in helper.args.args] == ["text"]
    assert len(helper_body) == 1
    assert node_text(helper_body[0]) == node_text(
        ast.parse(
            "return _collapse_text(text).lower() in _MANUAL_PROCEDURE_CLAUSES"
        ).body[0]
    ), "Unknown procedure classification helper"
    pending = [
        "recognize_source_structure",
        "_source_exception_branches",
        "_source_exception_requires_paired_witness",
        "_source_formula_branches",
        "_source_has_arithmetic_comparison_condition",
    ]
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen or name not in before:
            continue
        seen.add(name)
        assert name in after and node_text(before[name]) == node_text(after[name]), name
        pending.extend(
            node.id
            for node in ast.walk(before[name])
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        )
    # Metadata removal is valid only if suppression reads formula spans alone.
    exception = before["_source_exception_branches"]
    suppression = [
        node
        for node in ast.walk(exception)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Name) and child.id == "formula_branches"
            for child in ast.walk(node.test)
        )
    ]
    assert len(suppression) == 1
    assert {
        node.attr
        for node in ast.walk(suppression[0])
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "formula_branch"
    } == {"start", "end"}
    return {
        "checked_bindings": sorted(seen),
        "checked_binding_count": len(seen),
        "normalized_closure_identical": True,
        "exemption_helper_is_exact_normalized_membership": True,
        "formula_suppression_reads_only_start_and_end": True,
        "normalizations": [
            "closed exemption guard and its source-context arguments",
            "formula passage-locator numeric metadata unobserved by suppression",
        ]
        + (
            [
                "current-main numeric_limits_only parameter and exact numeric-limit filter"
            ]
            if numeric_filter
            else []
        ),
    }


def load_module(key: str, source: bytes):
    name = "axiom_encode.harness._procedure_corpus_" + key
    module = types.ModuleType(name)
    module.__package__ = "axiom_encode.harness"
    sys.modules[name] = module
    exec(compile(source, "<" + key + ">", "exec"), module.__dict__)
    return module


def accelerated_guidance_source(source: bytes) -> tuple[bytes, dict]:
    """Reorder one pure classifier, preserving its Boolean result.

    The proof concerns valid string classification results. It does not assert
    identical call counts, exceptions, or resource timeouts for skipped work.
    """
    tree = ast.parse(source)
    targets = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_source_is_nonoperative_guidance_instruction"
    ]
    assert len(targets) == 1
    function = targets[0]
    expected = ast.parse(
        r'''def _source_is_nonoperative_guidance_instruction(text: str) -> bool:
    """Exclude agency workflow prose that has no claimant-facing outcome."""
    if _source_has_joined_operative_segment(text):
        return False
    return bool(
        re.match(
            r"State\s+agencies\s+must\s+(?:(?:review|check|contact|follow|use|"
            r"verify)\b|provide\s+(?:notice|notification|information|guidance|"
            r"documentation)\b)",
            text,
            flags=re.IGNORECASE,
        )
    )
'''
    ).body[0]
    assert node_text(function) == node_text(expected), "Unknown guidance classifier"
    assert function.col_offset == 0 and not function.decorator_list
    start_line, end_line = function.lineno - 1, function.end_lineno
    original_function = node_text(function)
    joined = copy.deepcopy(function.body[1].test)
    match = copy.deepcopy(function.body[2].value)
    result = ast.Return(ast.BoolOp(ast.And(), [match, ast.UnaryOp(ast.Not(), joined)]))
    function.body[:] = [function.body[0], result]
    ast.fix_missing_locations(function)
    assert node_text(function.body[1].value.values[0]) == node_text(
        expected.body[2].value
    )
    assert node_text(function.body[1].value.values[1].operand) == node_text(
        expected.body[1].test
    )
    lines = source.splitlines(keepends=True)
    original_span = ast.parse(b"".join(lines[start_line:end_line]))
    assert len(original_span.body) == 1
    assert node_text(original_span.body[0]) == original_function
    prefix, suffix = b"".join(lines[:start_line]), b"".join(lines[end_line:])
    replacement = (ast.unparse(function) + "\n").encode()
    replacement_lines = len(replacement.splitlines())
    assert replacement_lines <= end_line - start_line
    replacement += b"\n" * (end_line - start_line - replacement_lines)
    replacement_tree = ast.parse(replacement)
    assert len(replacement_tree.body) == 1
    assert node_text(replacement_tree.body[0]) == node_text(function)
    execution = prefix + replacement + suffix
    assert execution[: len(prefix)] == prefix
    assert execution[len(prefix) + len(replacement) :] == suffix
    assert len(execution.splitlines()) == len(lines)
    return execution, {
        "source_sha256": digest(source),
        "execution_source_sha256": digest(execution),
        "original_helper_ast_sha256": digest(original_function.encode()),
        "execution_helper_ast_sha256": digest(node_text(function).encode()),
        "changed_bindings": ["_source_is_nonoperative_guidance_instruction"],
        "unchanged_prefix_sha256": digest(prefix),
        "unchanged_suffix_sha256": digest(suffix),
        "replaced_source_lines": [start_line + 1, end_line],
        "line_count_preserved": True,
        "proof": "exact original classifier AST; Boolean result (!joined && match) equals (match && !joined), with the original match node and joined call unchanged",
        "scope": "valid-string Boolean classification results; call count, exception and resource-timeout equivalence are not asserted",
    }


def load_accelerated_module(key: str, source: bytes):
    execution, binding = accelerated_guidance_source(source)
    name = "axiom_encode.harness._procedure_corpus_" + key
    module = types.ModuleType(name)
    module.__package__ = "axiom_encode.harness"
    sys.modules[name] = module
    exec(compile(execution, "<" + key + ">", "exec"), module.__dict__)
    return module, binding


def obligations(module, text: str, citation: str, *, current_main=False) -> list[dict]:
    branches = module.recognize_source_structure(text, corpus_citation_path=citation)
    formula_options = {
        "branches": branches,
        "active_branches": branches,
        "deferred_paths": set(),
    }
    if (
        "corpus_citation_path"
        in inspect.signature(module._source_formula_branches).parameters
    ):
        formula_options["corpus_citation_path"] = citation
    formulas = module._source_formula_branches(text, **formula_options)
    exception_options = {
        "branches": branches,
        "active_branches": branches,
        "deferred_paths": set(),
        "formula_branches": formulas,
    }
    if current_main:
        exception_options["numeric_limits_only"] = module._is_us_manual_citation_path(
            citation
        )
    exceptions = module._source_exception_branches(text, **exception_options)
    result = []
    for branch in exceptions:
        options = {"branch_path": branch.path}
        if (
            "corpus_citation_path"
            in inspect.signature(
                module._source_exception_requires_paired_witness
            ).parameters
        ):
            options.update(corpus_citation_path=citation, source_text=text)
        if module._source_exception_requires_paired_witness(branch.text, **options):
            result.append(
                {
                    "path": list(branch.path),
                    "start": branch.start,
                    "end": branch.end,
                    "kind": branch.kind,
                    "text": branch.text,
                }
            )
    return result


def branch_key(branch: dict) -> tuple:
    return tuple(branch["path"]), branch["start"], branch["end"], branch["text"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    inventory_bytes = (HERE / "corpus-source-binding.json").read_bytes()
    inventory = json.loads(inventory_bytes)
    source_bytes = {"final": (ROOT / SOURCE).read_bytes()}
    for key, ref in (
        ("historical_main", HISTORICAL_MAIN),
        ("parent", PARENT),
        ("current_main", CURRENT_MAIN),
    ):
        source_bytes[key] = subprocess.check_output(
            ["git", "show", ref + ":" + SOURCE], cwd=ROOT
        )
    proof = {
        key: scope_proof(source_bytes[key], source_bytes["final"])
        for key in ("historical_main", "parent")
    }
    modules = {key: load_module(key, value) for key, value in source_bytes.items()}
    final = modules["final"]
    assert len(final._MANUAL_PROCEDURE_CLAUSES) == 2
    report = {
        "corpus_revision": inventory["git_commit"],
        "inventory_sha256": digest(inventory_bytes),
        "refs": {
            "historical_main": HISTORICAL_MAIN,
            "parent": PARENT,
            "current_main": CURRENT_MAIN,
        },
        "source_sha256": {key: digest(value) for key, value in source_bytes.items()},
        "normalized_exempt_source_sha256": final._MANUAL_PROCEDURE_SOURCE_SHA256,
        "scope_proof": proof,
        "rows": 0,
        "files": 0,
        "input_bytes": 0,
        "selected_rows": 0,
        "historical_main_differential": {"removed": 0, "added": 0},
        "parent_differential": {"removed": 0, "added": 0},
        "current_main_selected_rows_differential": {"removed": 0, "added": 0},
        "current_main_comparison_scope": "selected rows only; no global equality proof against the broader current-main relaxation",
        "changes": [],
    }
    started = time.monotonic()
    for name, entry in inventory["files"].items():
        if time.monotonic() - started > 525:
            raise TimeoutError("Corpus command exceeded its bounded time budget")
        content = (options.corpus_root / name).read_bytes()
        assert len(content) == entry["bytes"] and digest(content) == entry["sha256"], (
            name
        )
        report["files"] += 1
        report["input_bytes"] += len(content)
        for index, line in enumerate(content.decode().splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("body") or row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            report["rows"] += 1
            citation = row.get("citation_path", "")
            if citation != final._MANUAL_PROCEDURE_CITATION:
                continue
            collapsed = final._collapse_text(text).lower()
            if not any(
                clause in collapsed for clause in final._MANUAL_PROCEDURE_CLAUSES
            ):
                continue
            report["selected_rows"] += 1
            print(f"SELECTED file={name} row={index}", flush=True)
            paired = {
                key: {
                    branch_key(branch): branch
                    for branch in obligations(
                        module, text, citation, current_main=key == "current_main"
                    )
                }
                for key, module in modules.items()
            }
            for key in ("historical_main", "parent", "current_main"):
                counter = report[
                    key
                    + (
                        "_selected_rows_differential"
                        if key == "current_main"
                        else "_differential"
                    )
                ]
                for action, identities in (
                    ("removed", paired[key].keys() - paired["final"].keys()),
                    ("added", paired["final"].keys() - paired[key].keys()),
                ):
                    for identity in sorted(identities):
                        branch = (
                            paired[key][identity]
                            if action == "removed"
                            else paired["final"][identity]
                        )
                        counter[action] += 1
                        report["changes"].append(
                            {
                                "comparison": key,
                                "action": action,
                                "file": name,
                                "index": index,
                                "citation": citation,
                                "body_sha256": digest(text.encode()),
                                "clause_sha256": digest(branch["text"].encode()),
                                "branch": branch,
                            }
                        )
    assert report["files"] == inventory["file_count"] == 117
    assert report["rows"] == 39375
    assert report["input_bytes"] == inventory["total_bytes"]
    assert (ROOT / SOURCE).read_bytes() == source_bytes["final"], (
        "Source changed during scan"
    )
    assert (HERE / "corpus-source-binding.json").read_bytes() == inventory_bytes, (
        "Inventory changed during scan"
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"scope_proof", "changes", "source_sha256"}
            },
            sort_keys=True,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
