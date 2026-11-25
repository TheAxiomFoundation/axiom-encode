"""Post-run checks for new inputs in an encoded CFR module (A/B tool, not a validator).

Usage:
    python scripts/check_state_set_inputs.py --rulespec-root <rulespec-us/us> \\
        --source <source.txt> --module <candidate.yaml> [--baseline <baseline.yaml>]

Every input of the candidate that the baseline does not have is checked:
root `inputs` entries and implicit inputs (a free name in a formula that is
no rule, no import and no root input, the style of 42 CFR 435.811 on main).

For each new input it prints one owner-table row (with the trigger that
applies) and a FAIL line for each failed check; it exits 1 when any check
fails:

1. description: a root `inputs` entry whose `description` names a provision
   (`N CFR P.S`, `§ P.S`, `section N`, or a bare `P.S` of the unit's own CFR
   part such as `435.811`). An implicit input has none and fails.
2. value: a State-set value is a number; a Boolean or Judgment input is a
   fact and fails.
3. federal: the name or description cites a federal benefit program
   standard (SSI, Supplemental Security Income, Federal benefit rate); such
   an amount is never State-set.
4. trigger, recorded per input as the first that holds:
   - `actor`: a proof excerpt of a rule using the input, or the source
     paragraph holding it, names the State or the agency as the actor of an
     establish-type verb ("set by the agency", "the State elects"; "State
     plan" is not an actor);
   - `state-level`: the excerpt, its paragraph or the description names the
     State's own level ("the State's ...", "State approved AFDC plan",
     "State optional supplement", "under the Medicaid State plan",
     "section 1115");
   - `typed:<P.S>`: the description or an excerpt cites a section of the
     unit's CFR part (not its own) that the unit's source text cites as
     "established ... under/in accordance with § P.S", and a module for that
     section exists under the RuleSpec root.
5. exported: no rule anywhere under the RuleSpec root has the same name.
6. literals: every dollar amount and "N percent of the Federal poverty level"
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
_STATE_LEVEL = re.compile(
    r"\bState's\b"
    r"|\bState[-\s]+approved\s+AFDC\b|\bState\s+AFDC\b"
    r"|\bState\s+optional\s+supplement"
    r"|\b(?:under|in)\s+(?:the|its)\s+(?:Medicaid\s+)?State\s+plan\b"
    r"|\bsection\s+1115\b",
    re.IGNORECASE,
)
_FEDERAL_DESCRIPTION = re.compile(
    r"\bSSI\b|\bsupplemental\s+security\s+income\b|\bfederal\s+benefit\s+rate\b"
    r"|\bFBR\b",
    re.IGNORECASE,
)
_FEDERAL_NAME_TOKENS = frozenset({"ssi", "fbr"})
_NUMERIC_DTYPES = frozenset({"money", "rate", "decimal", "integer", "number"})


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
    formulas = "\n".join(f for rule in _rules(payload) for f in _formulas(rule))
    for name in sorted(evals._context_file_local_inputs(str(path))):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*\(", formulas):
            continue  # a formula function such as `count_where(...)`
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


def unit_section_key(citation: str) -> tuple[str, str, str] | None:
    return evals._us_cfr_unit_section_key(citation) if citation else None


def provision_named(description: str, unit_key: tuple[str, str, str] | None) -> bool:
    if _PROVISION.search(description):
        return True
    return unit_key is not None and (
        re.search(rf"(?<![\d.]){unit_key[1]}\.\d{{1,4}}(?![\d])", description)
        is not None
    )


def cited_same_part_sections(text: str, unit_key: tuple[str, str, str]) -> list[str]:
    """Sections of the unit's CFR part cited in text (`§ P.S`, `N CFR P.S`, bare `P.S`)."""
    title, part, own = unit_key
    sections = [
        group.section
        for group in evals._cited_cfr_sections_for_unit(
            f"us/regulation/{title}/{part}/{own}", text
        )
        if (group.title, group.part) == (title, part)
    ]
    for match in re.finditer(rf"(?<![\d.§]){part}\.(\d{{1,4}})(?![\d])", text):
        if match.group(1) != own and match.group(1) not in sections:
            sections.append(match.group(1))
    return sections


def typed_trigger(
    texts: list[str],
    source: str,
    root: Path,
    unit_key: tuple[str, str, str] | None,
) -> str | None:
    """Return `typed:P.S` when a cited same-part section is one the source
    says the value is established under, and its module exists."""
    if unit_key is None:
        return None
    title, part, _own = unit_key
    flat_source = " ".join(source.split())
    windows = []
    for match in re.finditer(r"\bestablish\w*", flat_source, re.IGNORECASE):
        window = flat_source[match.end() : match.end() + 100]
        window = re.split(r";|\.\s", window, maxsplit=1)[0]
        if "§" in window:
            windows.append(window)
    for text in texts:
        for section in cited_same_part_sections(text, unit_key):
            number = re.compile(rf"(?<![\d.]){part}\.{section}(?![\d])")
            if not any(number.search(window) for window in windows):
                continue
            group = evals._CitedCfrSectionGroup(title, part, section, ())
            if evals._cfr_section_group_context_modules(root, group):
                return f"typed:{part}.{section}"
    return None


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
    unit_key = unit_section_key(citation)
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
        flat = lambda text: " ".join(text.split())  # noqa: E731
        paragraphs = [
            paragraph
            for paragraph in re.split(r"\n\s*\n", source)
            if any(flat(e) in flat(paragraph) for e in excerpts)
        ]
        description = str(item.get("description") or "")
        trigger = None
        if any(actor_sets_value(t) for t in [*excerpts, *paragraphs]):
            trigger = "actor"
        elif any(_STATE_LEVEL.search(t) for t in [*excerpts, *paragraphs, description]):
            trigger = "state-level"
        else:
            trigger = typed_trigger([description, *excerpts], source, root, unit_key)
        shown = " ".join((excerpts[0] if excerpts else "").split()[:12])
        shape = "implicit" if item.get("implicit") else "root"
        rows.append(
            f"| {module.stem} | {name} | {shape} | "
            f"{item.get('entity', '-')}/{item.get('dtype', '-')}/{item.get('period', '-')}"
            f" | {trigger or '-'} | {shown} |"
        )
        if not provision_named(description, unit_key):
            failures.append(
                f"description: `{name}` has no description naming a provision"
            )
        if str(item.get("dtype") or "").lower() not in _NUMERIC_DTYPES:
            failures.append(f"value: `{name}` is not a number; a State-set value is")
        if _FEDERAL_DESCRIPTION.search(description) or (
            _FEDERAL_NAME_TOKENS & set(name.lower().split("_"))
        ):
            failures.append(
                f"federal: `{name}` cites a federal benefit program standard; "
                "it is never State-set"
            )
        if trigger is None:
            failures.append(
                f"trigger: no actor, State-level or typed trigger holds for `{name}`"
            )
        if name in exports:
            failures.append(
                f"exported: `{name}` is a rule name under the RuleSpec root"
            )
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
    print("| unit | input | form | entity/dtype/period | trigger | citing excerpt |")
    print("|---|---|---|---|---|---|")
    print("\n".join(rows))
    for failure in failures:
        print(f"FAIL {failure}")
    print(f"new inputs: {len(rows)}; failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
