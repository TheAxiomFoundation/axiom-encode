"""Authenticate the complete corpus and compare the exact exemption with main.

The whole production module must match pinned main after removing only the
reviewed exemption. Its exact citation guard proves equality for every other
citation. Every row is read and authenticated; rows at that citation execute
each revision's unmodified extraction, formula suppression, and pairing gate.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = "src/axiom_encode/harness/source_completeness.py"
MAIN = "c7fe642eee36ba4e1a766d3eae8a9f3bb31b9ed3"
CITATION = (
    "us-fl/manual/dcf/ess-program-policy-manual/2600-calculation-of-benefits/page-8"
)
SOURCE_SHA256 = "e17c82b40e09f545b80f2cedfad3bfc91daea3eea87801f9cb3c10188fccf0a0"
CLAUSES = frozenset(
    {
        "chapter: 2600 calculation of benefits program: fs 1 2610.0000 food stamps "
        "when the eligibility specialist has determined available income as per "
        "chapters 1800 and 2400, the policy in this chapter must be used to perform "
        "the budgets and tests to determine eligibility for benefits and the benefit "
        "amount. 2610.0100 budgets and test calculations (fs) each program has "
        "budgets and tests that must be executed in order to determine eligibility.",
        "refer to appendix a-1 for the 165% need standard table. 2610.0104.03 gross "
        "income test (fs) when gross income is determined, that income is compared "
        "to the allowable gross income limit for the assistance group's size.",
    }
)
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
sys.path.insert(0, str(ROOT / "src"))


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def node_text(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def bindings(tree: ast.Module) -> dict[str, ast.AST]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def scope_proof(before: bytes, after: bytes) -> dict:
    """Reject every executable module change beyond the exact reviewed hook."""

    changed = (
        subprocess.check_output(
            ["git", "diff", "--name-only", MAIN, "--", "src/axiom_encode"], cwd=ROOT
        )
        .decode()
        .splitlines()
    )
    assert changed == [SOURCE], ("Unexpected production change", changed)
    assert not subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "src/axiom_encode"],
        cwd=ROOT,
    ), "Untracked production code is outside the comparison proof"
    tree = ast.parse(after)
    functions = bindings(tree)
    helper = functions["_source_condition_is_manual_procedure"]
    expected_helper = ast.parse(
        '''def _source_condition_is_manual_procedure(text: str) -> bool:
    """Recognize only the two complete reviewed sequencing instructions."""
    return _collapse_text(text).lower() in _MANUAL_PROCEDURE_CLAUSES
'''
    ).body[0]
    assert node_text(helper) == node_text(expected_helper), "Changed membership helper"
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.startswith(
                "_MANUAL_PROCEDURE_"
            ):
                value = node.value
                if isinstance(value, ast.Call):
                    assert (
                        isinstance(value.func, ast.Name)
                        and value.func.id == "frozenset"
                        and len(value.args) == 1
                        and not value.keywords
                    )
                    value = value.args[0]
                constants[target.id] = ast.literal_eval(value)
    assert constants == {
        "_MANUAL_PROCEDURE_CITATION": CITATION,
        "_MANUAL_PROCEDURE_SOURCE_SHA256": SOURCE_SHA256,
        "_MANUAL_PROCEDURE_CLAUSES": CLAUSES,
    }, "Changed exemption keys"
    tree.body = [
        node
        for node in tree.body
        if node is not helper
        and not (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in constants
        )
    ]
    predicate = functions["_source_exception_requires_paired_witness"]
    for name in ("source_text", "corpus_citation_path"):
        indices = [
            index
            for index, argument in enumerate(predicate.args.kwonlyargs)
            if argument.arg == name
        ]
        assert len(indices) == 1
        index = indices[0]
        assert node_text(predicate.args.kw_defaults[index]) == node_text(
            ast.Constant("")
        )
        predicate.args.kwonlyargs.pop(index)
        predicate.args.kw_defaults.pop(index)
    guards = [
        node for node in predicate.body if node_text(node) == node_text(EXEMPTION_GUARD)
    ]
    assert len(guards) == 1, "Changed exemption guard"
    predicate.body.remove(guards[0])
    calls = [
        node
        for node in ast.walk(functions["_companion_test_issues"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_source_exception_requires_paired_witness"
    ]
    assert len(calls) == 1
    for name in ("corpus_citation_path", "source_text"):
        keywords = [keyword for keyword in calls[0].keywords if keyword.arg == name]
        assert len(keywords) == 1
        assert node_text(keywords[0].value) == node_text(ast.Name(name, ast.Load()))
        calls[0].keywords.remove(keywords[0])
    assert node_text(ast.parse(before)) == node_text(tree), (
        "Production module differs from main beyond the exact reviewed exemption"
    )
    return {
        "entire_module_ast_identical_after_removing_exact_exemption": True,
        "normalized_module_ast_sha256": digest(node_text(tree).encode()),
        "other_production_files_unchanged": True,
        "exact_reviewed_keys_verified_independently": True,
        "noncandidate_equality": "The exact citation guard is false for every row outside the canonical citation; all remaining module code and repository dependencies are unchanged from main.",
        "normalizations": [
            "three independently checked closed-list constants and exact membership helper",
            "exact source-authenticated exemption guard",
            "two empty-default predicate arguments and their one companion-gate call site",
        ],
    }


def load_module(key: str, source: bytes):
    name = "axiom_encode.harness._procedure_corpus_" + key
    module = types.ModuleType(name)
    module.__package__ = "axiom_encode.harness"
    sys.modules[name] = module
    exec(compile(source, "<" + key + ">", "exec"), module.__dict__)
    return module


def obligations(
    module, text: str, citation: str, *, final: bool, numeric_limits_only: bool = True
) -> dict:
    scope = getattr(module, "_us_manual_prose_scope", None)
    with scope(citation) if scope else contextlib.nullcontext():
        branches = module.recognize_source_structure(
            text, corpus_citation_path=citation
        )
        options = {
            "branches": branches,
            "active_branches": branches,
            "deferred_paths": set(),
        }
        formulas = module._source_formula_branches(
            text, corpus_citation_path=citation, **options
        )
        exceptions = module._source_exception_branches(
            text,
            formula_branches=formulas,
            numeric_limits_only=numeric_limits_only
            and module._is_us_manual_citation_path(citation),
            **options,
        )
        result = {}
        for branch in exceptions:
            predicate_options = {"branch_path": branch.path}
            if final:
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
                    digest(branch.text.encode()),
                )
                result[identity] = {
                    "path": list(branch.path),
                    "start": branch.start,
                    "end": branch.end,
                    "kind": branch.kind,
                    "text": branch.text,
                }
        return result


def footer_order_proof(source: bytes) -> dict:
    """A footer-free exact key is unchanged whichever guard runs first."""

    tree = ast.parse(source)
    predicate = bindings(tree)["_source_exception_requires_paired_witness"]
    exemption = next(
        index
        for index, node in enumerate(predicate.body)
        if node_text(node) == node_text(EXEMPTION_GUARD)
    )
    footer_assignment = next(
        (index, node)
        for index, node in enumerate(predicate.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "guidance_footer_fragments"
            for target in node.targets
        )
    )
    footer_index, assignment = footer_assignment
    assert exemption < footer_index
    import re

    pattern = ast.literal_eval(assignment.value)
    assert all(
        re.sub(pattern, " ", clause, flags=re.IGNORECASE) == clause
        for clause in CLAUSES
    )
    return {
        "exemption_precedes_footer_stripping": True,
        "footer_stripping_is_identity_on_both_exact_keys": True,
        "all_other_footer_code_identical_to_main": True,
        "order_scope": "The two exact Florida clauses contain no footer fragment. Every other key fails the exact exemption and reaches main's unchanged footer code.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    inventory_bytes = (HERE / "corpus-source-binding.json").read_bytes()
    inventory = json.loads(inventory_bytes)
    sources = {
        "main": subprocess.check_output(["git", "show", MAIN + ":" + SOURCE], cwd=ROOT),
        "final": (ROOT / SOURCE).read_bytes(),
    }
    proof = scope_proof(sources["main"], sources["final"])
    modules = {key: load_module(key, source) for key, source in sources.items()}
    report = {
        "main_revision": MAIN,
        "corpus_revision": inventory["git_commit"],
        "inventory_sha256": digest(inventory_bytes),
        "source_sha256": {key: digest(source) for key, source in sources.items()},
        "audit_sha256": digest(Path(__file__).read_bytes()),
        "scope_proof": proof,
        "footer_order": footer_order_proof(sources["final"]),
        "scope": "Raw source paired-witness demands, with main's manual numeric filter and formula suppression on both sides, no artifact deferrals or independently covered paths. All corpus rows authenticated; exact-citation candidates extracted; whole-module identity proves equality elsewhere.",
        "files": 0,
        "rows": 0,
        "input_bytes": 0,
        "candidate_rows": 0,
        "differential": {"removed": 0, "added": 0},
        "unfiltered_predicate_differential": {"removed": 0, "added": 0},
        "candidate_obligations": [],
        "changes": [],
    }
    for name, entry in inventory["files"].items():
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
            if citation != CITATION:
                continue
            report["candidate_rows"] += 1
            record = {
                "file": name,
                "index": index,
                "citation": citation,
                "body_sha256": digest(text.encode()),
            }
            candidate = record | {"obligations": {}}
            for scope in ("production", "unfiltered_predicate"):
                paired = {
                    key: obligations(
                        module,
                        text,
                        citation,
                        final=key == "final",
                        numeric_limits_only=scope == "production",
                    )
                    for key, module in modules.items()
                }
                candidate["obligations"][scope] = {
                    key: list(value.values()) for key, value in paired.items()
                }
                counter = report[
                    "differential"
                    if scope == "production"
                    else "unfiltered_predicate_differential"
                ]
                for action, identities in (
                    ("removed", paired["main"].keys() - paired["final"].keys()),
                    ("added", paired["final"].keys() - paired["main"].keys()),
                ):
                    for identity in sorted(identities):
                        branch = paired["main" if action == "removed" else "final"][
                            identity
                        ]
                        counter[action] += 1
                        report["changes"].append(
                            record
                            | {
                                "scope": scope,
                                "action": action,
                                "clause_sha256": identity[-1],
                                "branch": branch,
                            }
                        )
            report["candidate_obligations"].append(candidate)
    assert report["files"] == inventory["file_count"] == 117
    assert report["rows"] == 39375
    assert report["input_bytes"] == inventory["total_bytes"] == 146278276
    assert (ROOT / SOURCE).read_bytes() == sources["final"], (
        "Source changed during census"
    )
    assert (HERE / "corpus-source-binding.json").read_bytes() == inventory_bytes
    assert report["differential"] == {"removed": 0, "added": 0}, report["differential"]
    assert report["unfiltered_predicate_differential"] == {"removed": 2, "added": 0}
    assert len(report["changes"]) == 2
    final = modules["final"]
    for change in report["changes"]:
        normalized_clause = final._collapse_text(change["branch"]["text"]).lower()
        assert normalized_clause in CLAUSES
        assert change["citation"] == CITATION
        assert change["action"] == "removed"
        assert change["scope"] == "unfiltered_predicate"
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "main_revision",
                    "files",
                    "rows",
                    "input_bytes",
                    "candidate_rows",
                    "differential",
                    "unfiltered_predicate_differential",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
