"""Post-run checks for new inputs in an encoded CFR module (A/B tool, not a validator).

Usage:
    python scripts/check_state_set_inputs.py --rulespec-root <rulespec-us/us> \\
        --source <source.txt> --module <candidate.yaml> [--baseline <baseline.yaml>]

Every input of the candidate that the baseline does not have is checked:
root `inputs` entries and implicit inputs (a free name in a formula that is
no rule, no import and no root input, the style of 42 CFR 435.811 on main).

For each new input it prints one owner-table row and a FAIL line for each
failed check; it exits 1 when any check fails:

1. description: a root `inputs` entry whose `description` names a provision
   (`N CFR P.S`, `§ P.S`, `section N`). An implicit input has none and fails.
2. trigger: a proof excerpt of a rule that uses the input either names the
   State or the agency as the actor of an establish-type verb
   ("established by the agency", "the State elects"; "State plan" is not an
   actor), or cites a CFR section whose module in the checkout takes the same
   name as a local input and exports no rule of that name.
3. exported: no rule anywhere under the RuleSpec root has the same name.
4. literals: every dollar amount and "N percent of the Federal poverty level"
   in the source paragraphs that hold those excerpts appears in some
   `versions[].formula` of the module (as N or N/100). Summaries and proof
   excerpts do not count.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from axiom_encode.harness import evals

_VERB = r"(?:establish|elect|specif|set|select|choos)\w*"
_ACTOR = r"(?:the\s+)?(?:State\s+agency|State|agency)(?!\s+plan)\b"
_ACTOR_PHRASE = re.compile(
    rf"\b{_VERB}\s+by\s+{_ACTOR}"
    rf"|\b{_ACTOR}\s+(?:has\s+|may\s+|must\s+|shall\s+|will\s+)?{_VERB}",
    re.IGNORECASE,
)
_PROVISION = re.compile(r"\b\d+\s+C\.?\s?F\.?\s?R\.?\s+\S|§\s*\d|\bsection\s+\d", re.I)
_DOLLAR = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)")
_FPL = re.compile(
    r"(\d+(?:\.\d+)?)\s*percent\s+of\s+the\s+(?:Federal\s+poverty\s+(?:level|line)|FPL)",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _payload(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text()) if path.exists() else None
    return payload if isinstance(payload, dict) else {}


def _rules(payload: dict) -> list[dict]:
    return [rule for rule in payload.get("rules") or [] if isinstance(rule, dict)]


def _formulas(rule: dict) -> list[str]:
    return [
        str(version.get("formula"))
        for version in rule.get("versions") or []
        if isinstance(version, dict) and version.get("formula") is not None
    ]


def module_inputs(path: Path) -> dict[str, dict]:
    """Return every input of a module: root entries and implicit names."""
    payload = _payload(path)
    inputs = {
        str(item["name"]): dict(item)
        for item in payload.get("inputs") or []
        if isinstance(item, dict) and item.get("name")
    }
    for name in sorted(evals._context_file_local_inputs(str(path))):
        inputs.setdefault(name, {"name": name, "implicit": True})
    return inputs


def exported_names(root: Path) -> set[str]:
    names: set[str] = set()
    for path in root.rglob("*.yaml"):
        if path.name.endswith(".test.yaml"):
            continue
        try:
            payload = _payload(path)
        except yaml.YAMLError:
            continue
        names.update(str(rule["name"]) for rule in _rules(payload) if rule.get("name"))
    return names


def actor_sets_value(excerpt: str) -> bool:
    return _ACTOR_PHRASE.search(" ".join(excerpt.split())) is not None


def cited_module_takes_input(
    excerpt: str, name: str, root: Path, unit_citation: str
) -> bool:
    """Whether the excerpt cites a CFR section whose module takes `name` as input."""
    for group in evals._cited_cfr_sections_for_unit(unit_citation, excerpt):
        for module in evals._cfr_section_group_context_modules(root, group):
            if name in evals._context_file_local_inputs(str(module)) and (
                name not in evals._context_file_exports(str(module))
            ):
                return True
    return False


def literal_values(paragraph: str) -> list[tuple[str, set[str]]]:
    values = []
    for raw in _DOLLAR.findall(paragraph):
        number = float(raw.replace(",", ""))
        values.append((f"${raw}", {f"{number:g}"}))
    for raw in _FPL.findall(paragraph):
        number = float(raw)
        values.append((f"{raw} percent FPL", {f"{number:g}", f"{number / 100:g}"}))
    return values


def check(
    root: Path, source: str, module: Path, baseline: Path | None
) -> tuple[list[str], list[str]]:
    """Return (owner-table rows, failures)."""
    payload = _payload(module)
    citation = str(
        (payload.get("module") or {})
        .get("source_verification", {})
        .get("corpus_citation_path", "")
    )
    old = set(module_inputs(baseline)) if baseline is not None else set()
    exports = exported_names(root)
    formula_text = "\n".join(
        formula for rule in _rules(payload) for formula in _formulas(rule)
    )
    rows: list[str] = []
    failures: list[str] = []
    for name, item in module_inputs(module).items():
        if name in old:
            continue
        users = [
            rule
            for rule in _rules(payload)
            if any(name in _IDENTIFIER.findall(f) for f in _formulas(rule))
        ]
        excerpts = [
            str(atom["source"]["excerpt"])
            for rule in users
            for atom in ((rule.get("metadata") or {}).get("proof") or {}).get("atoms")
            or []
            if isinstance(atom.get("source"), dict) and atom["source"].get("excerpt")
        ]
        actor = next((e for e in excerpts if actor_sets_value(e)), None)
        typed = next(
            (e for e in excerpts if cited_module_takes_input(e, name, root, citation)),
            None,
        )
        shown = " ".join(
            (actor or typed or (excerpts[0] if excerpts else "")).split()[:12]
        )
        shape = "implicit" if item.get("implicit") else "root"
        rows.append(
            f"| {module.stem} | {name} | {shape} | "
            f"{item.get('entity', '-')}/{item.get('dtype', '-')}/{item.get('period', '-')}"
            f" | {shown} |"
        )
        if not _PROVISION.search(str(item.get("description") or "")):
            failures.append(
                f"description: `{name}` has no description naming a provision"
            )
        if actor is None and typed is None:
            failures.append(
                f"trigger: no proof excerpt for `{name}` names a State/agency actor "
                "or cites a module that takes it as an input"
            )
        if name in exports:
            failures.append(
                f"exported: `{name}` is a rule name under the RuleSpec root"
            )
        paragraphs = [
            paragraph
            for paragraph in re.split(r"\n\s*\n", source)
            if any(" ".join(e.split()) in " ".join(paragraph.split()) for e in excerpts)
        ]
        for paragraph in paragraphs:
            for label, forms in literal_values(paragraph):
                if not any(
                    re.search(rf"(?<![\d.]){re.escape(form)}(?![\d])", formula_text)
                    for form in forms
                ):
                    failures.append(
                        f"literal: {label} in the cited branch is in no formula "
                        f"(input `{name}`)"
                    )
    return rows, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rulespec-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args(argv)
    rows, failures = check(
        args.rulespec_root, args.source.read_text(), args.module, args.baseline
    )
    print("| unit | input | form | entity/dtype/period | citing excerpt |")
    print("|---|---|---|---|---|")
    print("\n".join(rows))
    for failure in failures:
        print(f"FAIL {failure}")
    print(f"new inputs: {len(rows)}; failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
