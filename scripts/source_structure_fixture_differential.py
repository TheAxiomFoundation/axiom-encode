"""Compare source-structure and condition-gate verdicts over committed fixtures.

Run ``collect`` once in each checkout, for example a pull request and the main
commit it merges, then ``compare`` the two outputs. Inputs are named relative
to the root they came from, so two checkouts of the same fixtures line up.
``compare`` prints every changed verdict and exits 1 when any changed.

Verdicts for each source text:

- ``branches``: every recognized branch path, kind, label and span;
- ``fact_gates``: the conjunctive fact gates of each sentence with a condition
  cue in every branch's direct text, read with its physical line breaks;
- ``text_fact_gates``: the same gates over the whole text's sentences, keyed by
  absolute offset, so a structure change cannot hide or fake a gate change;
- ``boundaries``: the numeric boundary obligations;
- ``analysis``: for a fixture with a candidate RuleSpec, the complete-source
  issues from ``analyze_complete_source_unit``.

Work is bounded by counts, never by elapsed time. ``--max-chars`` skips a text
longer than the limit and ``--max-sentences`` caps the sentences gated in one
text; the output records every skipped input and every capped text.

``--root`` may name any directory, such as a checkout of a pinned corpus;
``--glob`` then chooses its source texts. Paths in the output stay relative
to that root.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as vp

DEFAULT_GLOBS = (
    "tests/fixtures/**/*.txt",
    "benchmarks/**/*.txt",
)
CANDIDATE_NAMES = (
    "candidate.yaml",
    "rule.yaml",
    "policy.yaml",
    "pre_fix.yaml",
    "post_fix.yaml",
)
CONDITION_CUE = re.compile(r"\b(?:if|when|provided\s+that|unless)\b", re.IGNORECASE)
SENTENCE_END = re.compile(r"(?<=[.;!?])[ \t]+(?=\S)|(?<=[.;!?])\r?\n")


def _extract(text: str) -> Any:
    return vp.extract_typed_numeric_inventory_occurrences_from_text(
        text, profile="en-US"
    )


def _citation_for(text_path: Path, root: Path) -> str:
    """Read the corpus citation a fixture records beside its source text."""

    directory = text_path.parent
    for name, key in (
        ("case.json", "corpus_citation_path"),
        ("issues.json", "citation"),
        ("source-metadata.json", "citation"),
    ):
        candidate = directory / name
        if candidate.is_file():
            try:
                value = json.loads(candidate.read_text()).get(key)
            except (json.JSONDecodeError, AttributeError):
                continue
            if isinstance(value, str) and value:
                return value
    return ""


def _sentences(text: str) -> list[tuple[int, str]]:
    sentences: list[tuple[int, str]] = []
    start = 0
    for match in SENTENCE_END.finditer(text):
        sentences.append((start, text[start : match.start()]))
        start = match.end()
    sentences.append((start, text[start:]))
    return [
        (offset, sentence)
        for offset, sentence in sentences
        if sentence.strip() and CONDITION_CUE.search(sentence)
    ]


def _gate_record(gates: Any) -> list[list[list[str]]]:
    return sorted(
        [sorted(entities), sorted(predicates)] for entities, predicates in gates
    )


def text_verdicts(
    text: str, *, citation: str, max_sentences: int
) -> tuple[dict[str, Any], bool]:
    branches = sc.recognize_source_structure(text, corpus_citation_path=citation)
    verdicts: dict[str, Any] = {
        "branches": [
            [list(branch.path), branch.kind, branch.label, branch.start, branch.end]
            for branch in branches
        ]
    }
    gates: dict[str, Any] = {}
    gated = 0
    capped = False
    for branch in branches:
        direct = sc._source_branch_direct_text(branch, branches=branches)
        for offset, sentence in _sentences(direct):
            if gated >= max_sentences:
                capped = True
                break
            gated += 1
            key = f"{'/'.join(branch.path)}@{branch.start}+{offset}"
            gates[key] = _gate_record(sc._source_conjunctive_fact_gates(sentence))
        if capped:
            break
    verdicts["fact_gates"] = gates
    text_gates: dict[str, Any] = {}
    for offset, sentence in _sentences(text)[:max_sentences]:
        text_gates[str(offset)] = _gate_record(
            sc._source_conjunctive_fact_gates(sentence)
        )
    if len(_sentences(text)) > max_sentences:
        capped = True
    verdicts["text_fact_gates"] = text_gates
    formulas = sc._source_formula_branches(
        text, branches=branches, active_branches=branches, deferred_paths=set()
    )
    verdicts["boundaries"] = sorted(
        [
            list(branch.path),
            branch.start,
            float(occurrence.value),
            occurrence.raw,
            occurrence.start,
            occurrence.end,
        ]
        for branch, occurrence in sc._source_boundary_obligations(
            branches,
            narrative_formula_branches=formulas,
            extract_numeric_occurrences=_extract,
            source_text=text,
        )
    )
    return verdicts, capped


def candidate_verdict(
    candidate: Path, text: str, *, citation: str
) -> dict[str, Any] | None:
    content = candidate.read_text()
    tests_path = candidate.with_name(candidate.stem + ".test.yaml")
    test_cases = (
        yaml.safe_load(tests_path.read_text()) if tests_path.is_file() else None
    )
    analysis = sc.analyze_complete_source_unit(
        content,
        text,
        corpus_citation_path=citation,
        test_cases=test_cases,
        extract_numeric_occurrences=_extract,
        extract_numeric_grounding_occurrences=functools.partial(
            vp.extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=vp.extract_named_scalar_occurrences,
        numeric_value_is_grounded=vp.numeric_value_is_grounded,
    )
    return {"issues": sorted(analysis.issues)}


def _selected(items: list[Path], shard: str | None) -> list[Path]:
    if not shard:
        return items
    index, count = (int(part) for part in shard.split("/"))
    if not 1 <= index <= count:
        raise SystemExit(f"--shard {shard!r} must look like 2/5")
    return items[index - 1 :: count]


def collect(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    texts = sorted(
        {
            path
            for pattern in (args.glob or DEFAULT_GLOBS)
            for path in root.glob(pattern)
        }
    )
    texts = [path for path in texts if path.is_file()]
    output: dict[str, Any] = {
        "root_relative_inputs": True,
        "max_chars": args.max_chars,
        "max_sentences": args.max_sentences,
        "texts": {},
        "candidates": {},
        "skipped": {},
        "capped": [],
        "errors": {},
    }
    for path in _selected(texts, args.shard):
        relative = path.relative_to(root).as_posix()
        text = path.read_text(errors="replace")
        if len(text) > args.max_chars:
            output["skipped"][relative] = f"{len(text)} characters"
            continue
        citation = _citation_for(path, root)
        try:
            verdicts, capped = text_verdicts(
                text, citation=citation, max_sentences=args.max_sentences
            )
        except Exception as error:  # Report, never hide, a changed failure.
            output["errors"][relative] = f"{type(error).__name__}: {error}"
            continue
        output["texts"][relative] = {"citation": citation, **verdicts}
        if capped:
            output["capped"].append(relative)
        if args.skip_candidates:
            continue
        for name in CANDIDATE_NAMES:
            candidate = path.with_name(name)
            if not candidate.is_file():
                continue
            key = f"{candidate.relative_to(root).as_posix()}|{relative}"
            try:
                output["candidates"][key] = candidate_verdict(
                    candidate, text, citation=citation
                )
            except Exception as error:
                output["errors"][key] = f"{type(error).__name__}: {error}"
    Path(args.output).write_text(json.dumps(output, indent=1, sort_keys=True) + "\n")
    print(
        f"{len(output['texts'])} texts, {len(output['candidates'])} candidates, "
        f"{len(output['skipped'])} skipped, {len(output['capped'])} capped, "
        f"{len(output['errors'])} errors",
        file=sys.stderr,
    )
    return 0


def _merged(paths: list[str]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        part = json.loads(Path(path).read_text())
        for section, value in part.items():
            if isinstance(value, dict):
                merged.setdefault(section, {}).update(value)
            elif isinstance(value, list):
                merged.setdefault(section, []).extend(value)
            else:
                merged[section] = value
    return merged


def _multiset_change(before: list[Any], after: list[Any]) -> dict[str, list[Any]]:
    old = Counter(json.dumps(item, sort_keys=True) for item in before)
    new = Counter(json.dumps(item, sort_keys=True) for item in after)
    return {
        "removed": [json.loads(item) for item in sorted((old - new).elements())],
        "added": [json.loads(item) for item in sorted((new - old).elements())],
    }


def compare(args: argparse.Namespace) -> int:
    before = _merged(args.before)
    after = _merged(args.after)
    changes: list[dict[str, Any]] = []
    for section in ("skipped", "errors"):
        for key in sorted(set(before.get(section, {})) | set(after.get(section, {}))):
            if before.get(section, {}).get(key) != after.get(section, {}).get(key):
                changes.append(
                    {
                        "input": key,
                        "verdict": section,
                        "before": before.get(section, {}).get(key),
                        "after": after.get(section, {}).get(key),
                    }
                )
    for key in sorted(set(before["texts"]) | set(after["texts"])):
        old = before["texts"].get(key)
        new = after["texts"].get(key)
        if old is None or new is None:
            changes.append({"input": key, "verdict": "present", "before": bool(old)})
            continue
        for verdict in ("branches", "boundaries"):
            delta = _multiset_change(old[verdict], new[verdict])
            if delta["removed"] or delta["added"]:
                changes.append({"input": key, "verdict": verdict, **delta})
        for verdict in ("fact_gates", "text_fact_gates"):
            for gate_key in sorted(set(old[verdict]) | set(new[verdict])):
                if old[verdict].get(gate_key) != new[verdict].get(gate_key):
                    changes.append(
                        {
                            "input": key,
                            "verdict": verdict,
                            "sentence": gate_key,
                            "before": old[verdict].get(gate_key),
                            "after": new[verdict].get(gate_key),
                        }
                    )
    for key in sorted(set(before["candidates"]) | set(after["candidates"])):
        old = (before["candidates"].get(key) or {}).get("issues", [])
        new = (after["candidates"].get(key) or {}).get("issues", [])
        delta = _multiset_change(old, new)
        if delta["removed"] or delta["added"]:
            changes.append({"input": key, "verdict": "analysis", **delta})
    report = {
        "texts": len(after["texts"]),
        "candidates": len(after["candidates"]),
        "capped": sorted(set(before.get("capped", [])) | set(after.get("capped", []))),
        "changes": changes,
    }
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
    return 1 if changes else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser(
        "collect", help="write this checkout's verdicts"
    )
    collect_parser.add_argument("--root", default=str(Path(__file__).parents[1]))
    collect_parser.add_argument("--glob", action="append")
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--shard", help="k/N: every Nth input, from the kth")
    collect_parser.add_argument("--max-chars", type=int, default=200_000)
    collect_parser.add_argument("--max-sentences", type=int, default=2_000)
    collect_parser.add_argument("--skip-candidates", action="store_true")
    compare_parser = commands.add_parser("compare", help="list every changed verdict")
    compare_parser.add_argument("--before", action="append", required=True)
    compare_parser.add_argument("--after", action="append", required=True)
    compare_parser.add_argument("--output")
    args = parser.parse_args(argv)
    return collect(args) if args.command == "collect" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
