"""Post-run checks for new inputs in an encoded CFR module (A/B tool, not a validator).

Usage:
    python scripts/check_state_set_inputs.py --rulespec-root <rulespec-us/us> \\
        --source <source.txt> --module <candidate.yaml> [--baseline <baseline.yaml>]

Every input of the candidate that the baseline does not have is checked:
root `inputs` entries and implicit inputs (a free name in a formula that is
no rule, no import and no root input, the style of 42 CFR 435.811 on main).

The tool fails closed. It exits 2 with an `ERROR` line when it cannot do its
work: a path that does not exist (`--rulespec-root`, `--source`, `--module`,
`--baseline`), a module, baseline or RuleSpec-root file that is not parsable
YAML, a RuleSpec root with no module, an empty source, or a module with no
`module.source_verification.corpus_citation_path`.

For each new input it prints one owner-table row (with the trigger that
applies) and a FAIL line for each failed check; it exits 1 when any check
fails:

1. scope: the unit is on the allow-list of CFR parts for which the State-set
   amount rule is emitted (`CFR_STATE_SET_AMOUNT_PARTS` of the encoder
   prompt). A new input in any other unit fails.
2. description: a root `inputs` entry whose `description` names a provision
   (`N CFR P.S`, `§ P.S`, `section N`, or a bare `P.S` of the unit's own CFR
   part such as `435.811`). An implicit input has none and fails. The
   description is model-authored; it never satisfies the trigger.
3. value: a State-set value is a number; a Boolean or Judgment input is a
   fact and fails.
4. excerpt: every proof excerpt of a rule that uses the input is verbatim
   (whitespace-normalised) in the unit's source text. An excerpt that is not
   in the source fails and is not used for the trigger.
5. trigger: acceptance is a positive condition on the source text. A verified
   excerpt must have one of two closed forms, recorded per input as the first
   that holds:
   - `actor`: the State, the State agency or (only in the CFR parts where the
     agency is the State agency) the agency is the actor of a verb of the
     closed list establish, set, elect, specify, select, choose, or of
     "determine ... in/under its State plan" ("set by the agency", "the
     State elects"; "State plan" is not an actor);
   - `plan-level`: a level in effect on a stated calendar date under the
     State's own plan ("the State's AFDC payment standard in effect as of
     July 16, 1996", "under the Medicaid State plan ... as of March 23,
     2010").
   Everything else has no trigger, so an amount that a federal actor or a
   federal formula sets is never accepted; no list of federal names is used.
6. relation: when a new input has a trigger, the module holds a
   `kind: source_relation` record with `source_relation.type: delegates`.
7. exported: no rule anywhere under the RuleSpec root has the same name.
8. literals: every dollar amount and "N percent of the Federal poverty level"
   in the source paragraphs that hold those excerpts appears in some
   `versions[].formula` of the module (as N or N/100). Summaries and proof
   excerpts do not count.

The trigger binds a rule's excerpt, not the input's name, to the source: a
rule that uses two amounts and cites one actor phrase gives both a trigger.
The owner reads every row.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from axiom_encode.harness import evals
from axiom_encode.prompts import encoder as encoder_prompt

_ARTICLE = r"(?:the|a|each)\s+"
_STATE_ACTOR = rf"\b{_ARTICLE}(?:State\s+agency|State)(?!\s+plan\b)\b"
_AGENCY_ACTOR = (
    rf"\b{_ARTICLE}(?:(?:single\s+)?State\s+|Medicaid\s+)?agency(?!\s+plan\b)\b"
)
_BASE = r"(?:establish|set|elect|specify|select|choose)"
_THIRD = r"(?:establishes|sets|elects|specifies|selects|chooses)"
_PARTICIPLE = r"(?:established|set|elected|specified|selected|chosen)"
_MODAL = r"(?:must|may|shall|will)\s+(?:also\s+)?"
_IN_STATE_PLAN = r"(?:in|under)\s+(?:its|the)\s+(?:Medicaid\s+)?State\s+plan\b"


def _actor_phrase(actor: str) -> re.Pattern[str]:
    return re.compile(
        rf"\b{_PARTICIPLE}\s+by\s+(?:{actor})"
        rf"|(?:{actor})\s+(?:{_MODAL}{_BASE}|has\s+{_PARTICIPLE}|{_THIRD})\b"
        rf"|\bdetermined\s+by\s+(?:{actor})\s+{_IN_STATE_PLAN}"
        rf"|(?:{actor})\s+(?:{_MODAL}determine|determines)\b[^.;]{{0,120}}?"
        rf"\b{_IN_STATE_PLAN}",
        re.IGNORECASE,
    )


_STATE_ACTOR_PHRASE = _actor_phrase(_STATE_ACTOR)
_STATE_OR_AGENCY_ACTOR_PHRASE = _actor_phrase(f"{_STATE_ACTOR}|{_AGENCY_ACTOR}")
_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October"
    r"|November|December)"
)
_PLAN_LEVEL_ANCHOR = re.compile(
    r"\bState's\s+AFDC\s+(?:payment|income)\s+standard\b"
    r"|\bunder\s+the\s+(?:Medicaid\s+)?State\s+plan\b"
    r"|\bwaiver\s+of\s+the\s+State\s+plan\b"
)
_PLAN_LEVEL_DATE = re.compile(
    rf"\b(?:as\s+of|in\s+effect\s+on)\s+{_MONTH}\s+\d{{1,2}},\s+\d{{4}}\b"
)
_PROVISION = re.compile(r"\b\d+\s+C\.?\s?F\.?\s?R\.?\s+\S|§\s*\d|\bsection\s+\d", re.I)
_DOLLAR = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)")
_FPL = re.compile(
    r"(\d+(?:\.\d+)?)\s*percent\s+of\s+the\s+(?:Federal\s+poverty\s+(?:level|line)|FPL)",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMERIC_DTYPES = frozenset({"money", "rate", "decimal", "integer", "number"})


class ToolError(Exception):
    """The tool cannot do its work; the run must not count as a pass."""


def _flat(text: str) -> str:
    return " ".join(text.split())


def _read(path: Path, what: str) -> str:
    if not path.is_file():
        raise ToolError(f"{what} not found: {path}")
    try:
        return path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolError(f"{what} is not readable: {path}") from exc


def _payload(path: Path, what: str) -> dict:
    text = _read(path, what)
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ToolError(f"{what} is not parsable YAML: {path}") from exc
    if not isinstance(payload, dict):
        raise ToolError(f"{what} is not a YAML mapping: {path}")
    return payload


def _rules(payload: dict) -> list[dict]:
    return [rule for rule in payload.get("rules") or [] if isinstance(rule, dict)]


def _formulas(rule: dict) -> list[str]:
    return [
        str(version.get("formula"))
        for version in rule.get("versions") or []
        if isinstance(version, dict) and version.get("formula") is not None
    ]


def module_inputs(path: Path, what: str = "module") -> dict[str, dict]:
    """Return every input of a module: root entries and implicit names."""
    payload = _payload(path, what)
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
    if not root.is_dir():
        raise ToolError(f"RuleSpec root not found: {root}")
    names: set[str] = set()
    modules = 0
    for path in sorted(root.rglob("*.yaml")):
        if path.name.endswith(".test.yaml"):
            continue
        text = _read(path, "RuleSpec root file")
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ToolError(f"RuleSpec root file is not parsable YAML: {path}") from exc
        modules += 1
        if isinstance(payload, dict):
            names.update(
                str(rule["name"]) for rule in _rules(payload) if rule.get("name")
            )
    if not modules:
        raise ToolError(f"RuleSpec root holds no module: {root}")
    return names


def actor_sets_value(excerpt: str, *, agency_is_state: bool) -> bool:
    """True when the text names the State (or its agency) as the setting actor."""
    pattern = _STATE_OR_AGENCY_ACTOR_PHRASE if agency_is_state else _STATE_ACTOR_PHRASE
    return pattern.search(_flat(excerpt)) is not None


def plan_level_on_a_date(excerpt: str) -> bool:
    """True for a level in effect on a stated date under the State's own plan."""
    flat = _flat(excerpt)
    return bool(_PLAN_LEVEL_ANCHOR.search(flat) and _PLAN_LEVEL_DATE.search(flat))


def state_set_trigger(
    excerpt: str, unit_key: tuple[str, str, str] | None
) -> str | None:
    """Return the closed form that source text meets, for an allow-listed unit."""
    if (
        unit_key is None
        or unit_key[:2] not in encoder_prompt.CFR_STATE_SET_AMOUNT_PARTS
    ):
        return None
    agency_is_state = unit_key[:2] in encoder_prompt.CFR_PARTS_WHERE_AGENCY_IS_THE_STATE
    if actor_sets_value(excerpt, agency_is_state=agency_is_state):
        return "actor"
    if plan_level_on_a_date(excerpt):
        return "plan-level"
    return None


def unit_section_key(citation: str) -> tuple[str, str, str] | None:
    return encoder_prompt.us_cfr_unit_section_key(citation) if citation else None


def provision_named(description: str, unit_key: tuple[str, str, str] | None) -> bool:
    if _PROVISION.search(description):
        return True
    return unit_key is not None and (
        re.search(rf"(?<![\d.]){unit_key[1]}\.\d{{1,4}}(?![\d])", description)
        is not None
    )


def has_delegates_relation(payload: dict) -> bool:
    return any(
        rule.get("kind") == "source_relation"
        and isinstance(rule.get("source_relation"), dict)
        and rule["source_relation"].get("type") == "delegates"
        for rule in _rules(payload)
    )


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
    """Return (owner-table rows, failures); raise ToolError when it cannot check."""
    payload = _payload(module, "module")
    verification = (payload.get("module") or {}).get("source_verification") or {}
    citation = str(verification.get("corpus_citation_path") or "")
    if not citation:
        raise ToolError(
            f"module has no module.source_verification.corpus_citation_path: {module}"
        )
    flat_source = _flat(source)
    if not flat_source:
        raise ToolError("unit source text is empty")
    unit_key = unit_section_key(citation)
    in_scope = encoder_prompt.cfr_state_set_amount_rule_applies(citation)
    allowed = ", ".join(
        f"{title} CFR part {part}"
        for title, part in sorted(encoder_prompt.CFR_STATE_SET_AMOUNT_PARTS)
    )
    old = set(module_inputs(baseline, "baseline")) if baseline is not None else set()
    exports = exported_names(root)
    formula_text = "\n".join(
        formula for rule in _rules(payload) for formula in _formulas(rule)
    )
    rows: list[str] = []
    failures: list[str] = []

    def fail(message: str) -> None:
        if message not in failures:
            failures.append(message)

    any_trigger = False
    for name, item in module_inputs(module).items():
        if name in old:
            continue
        users = [
            rule
            for rule in _rules(payload)
            if any(name in _IDENTIFIER.findall(f) for f in _formulas(rule))
        ]
        cited = [
            _flat(str(atom["source"]["excerpt"]))
            for rule in users
            for atom in ((rule.get("metadata") or {}).get("proof") or {}).get("atoms")
            or []
            if isinstance(atom, dict)
            and isinstance(atom.get("source"), dict)
            and atom["source"].get("excerpt")
        ]
        # Only text that is verbatim in the unit source counts from here on.
        excerpts = [excerpt for excerpt in cited if excerpt in flat_source]
        paragraphs = [
            paragraph
            for paragraph in re.split(r"\n\s*\n", source)
            if any(excerpt in _flat(paragraph) for excerpt in excerpts)
        ]
        description = str(item.get("description") or "")
        trigger = next(
            (
                kind
                for kind in (state_set_trigger(e, unit_key) for e in excerpts)
                if kind
            ),
            None,
        )
        any_trigger = any_trigger or trigger is not None
        shown = " ".join((excerpts[0] if excerpts else "").split()[:12])
        shape = "implicit" if item.get("implicit") else "root"
        rows.append(
            f"| {module.stem} | {name} | {shape} | "
            f"{item.get('entity', '-')}/{item.get('dtype', '-')}/{item.get('period', '-')}"
            f" | {trigger or '-'} | {shown} |"
        )
        if not in_scope:
            fail(
                f"scope: `{name}` is a new input of `{citation}`, a unit outside "
                f"the allow-list ({allowed})"
            )
        if not provision_named(description, unit_key):
            fail(f"description: `{name}` has no description naming a provision")
        if str(item.get("dtype") or "").lower() not in _NUMERIC_DTYPES:
            fail(f"value: `{name}` is not a number; a State-set value is")
        for excerpt in cited:
            if excerpt not in flat_source:
                fail(
                    f"excerpt: a proof excerpt cited for `{name}` is not verbatim "
                    f'in the unit source: "{" ".join(excerpt.split()[:12])}"'
                )
        if in_scope and trigger is None:
            fail(
                f"trigger: no verified source excerpt names the State as the "
                f"actor that sets `{name}`, or a dated State plan level"
            )
        if name in exports:
            fail(f"exported: `{name}` is a rule name under the RuleSpec root")
        for paragraph in paragraphs:
            for label, forms in literal_values(paragraph):
                if not any(
                    re.search(rf"(?<![\d.]){re.escape(form)}(?![\d])", formula_text)
                    for form in forms
                ):
                    fail(
                        f"literal: {label} in the cited branch is in no formula "
                        f"(input `{name}`)"
                    )
    if any_trigger and not has_delegates_relation(payload):
        fail(
            "relation: the module has no `kind: source_relation` record with "
            "`source_relation.type: delegates` for the State-set amount"
        )
    return rows, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rulespec-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args(argv)
    try:
        rows, failures = check(
            args.rulespec_root,
            _read(args.source, "unit source"),
            args.module,
            args.baseline,
        )
    except ToolError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2
    print("| unit | input | form | entity/dtype/period | trigger | citing excerpt |")
    print("|---|---|---|---|---|---|")
    print("\n".join(rows))
    for failure in failures:
        print(f"FAIL {failure}")
    print(f"new inputs: {len(rows)}; failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
