"""Post-run checks for new inputs in an encoded CFR module (A/B tool, not a validator).

Usage:
    python scripts/check_state_set_inputs.py --rulespec-root <rulespec-us/us> \\
        --unit <us/regulation/42/435/733> --source <source.txt> \\
        --module <candidate.yaml> [--baseline <baseline.yaml>]

Every input of the candidate that the baseline does not have is checked:
root `inputs` entries and implicit inputs (a free name in a formula that is
no rule, no import and no root input, the style of 42 CFR 435.811 on main).

The tool fails closed. It exits 2 with an `ERROR` line when it cannot do its
work: a path that does not exist (`--rulespec-root`, `--source`, `--module`,
`--baseline`), a module, baseline or RuleSpec-root file that is not parsable
YAML, a module whose `inputs` or `rules` is not a list of mappings, a
RuleSpec root with no module, an empty source, or a module whose
`module.source_verification.corpus_citation_path` is missing or is not the
unit given with `--unit`. The unit comes from the caller, not from the
module, so a module cannot declare itself into scope.

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
5. trigger: acceptance is a positive condition that binds one verified
   excerpt to one input. The excerpt must hold a phrase of a closed shape,
   and the head noun phrase of that phrase (its words up to the first
   preposition) must hold every word of the input's name (`amount` and
   `state` are free). The shapes, recorded per input as the first that binds:
   - `actor`: the State, the State agency or (only in the CFR parts where the
     agency is the State agency) the agency is the actor of a verb of the
     closed list establish, set, elect, specify, select, choose, and the
     noun phrase is the object of that verb: "<the> NP established by the
     agency", "<the> NP the State elects", "the agency sets <the> NP". The
     verb determine counts only with "in/under its State plan" after it.
   - `plan-level`: a level in effect on a stated calendar date under the
     State's own plan: "State's AFDC payment|income standard in effect as
     of <date>", or "<the> NP under the [Medicaid] State plan [or waiver of
     the State plan] as of <date>".
   One excerpt serves one input. Because the noun phrase ends at the first
   preposition, in "a percentage of the Federal poverty level that the State
   elects" only `percentage` binds: a federal base (a benefit rate, the
   poverty level, a price index) is never the bound amount. No list of
   federal names is used.
6. relation: when a new input has a trigger, the module holds a
   `kind: source_relation` record with `source_relation.type: delegates`.
7. exported: no rule anywhere under the RuleSpec root has the same name.
8. literals: when the module has a new input, every dollar amount and every
   "N percent" (also "133 1/3 percent") of the unit's source text is a number
   in some `versions[].formula` of the module (N, N/100 or a fraction).
   Summaries and proof excerpts do not count.

The binding is lexical: it proves that the source attributes an amount with
that name to the State, not that the formula uses it correctly. The owner
reads every row.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import yaml

from axiom_encode import corpus_resolver
from axiom_encode.harness import evals
from axiom_encode.prompts import encoder as encoder_prompt

_DETERMINERS = frozenset({"the", "a", "an", "any", "each"})
# A noun phrase ends at the first of these words.
_PHRASE_ENDS = frozenset(
    "of for to at in on under by as than from with per that which who whose "
    "used payable equal and or but if when not is are was were be".split()
)
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
_OBJECT_START = r"\s+(?:(?:the|a|an)\s+)"
_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October"
    r"|November|December)"
)
_DATE = rf"{_MONTH}\s+\d{{1,2}},\s+\d{{4}}\b"
_ON_DATE = rf"(?:in\s+effect\s+)?(?:as\s+of|on)\s+{_DATE}"
_AFDC_LEVEL = re.compile(
    rf"\bState's\s+AFDC\s+(?:payment|income)\s+standard\s+in\s+effect\s+"
    rf"(?:as\s+of|on)\s+{_DATE}"
)
_PLAN_LEVEL = re.compile(
    r"\s+under\s+the\s+(?:Medicaid\s+)?State\s+plan"
    r"(?:\s+or\s+(?:a\s+)?waiver\s+of\s+the\s+(?:State\s+)?plan)?"
    r"(?:\s+under\s+section\s+1115(?:\s+of\s+the\s+Act)?)?"
    rf",?\s+{_ON_DATE}"
)
_CLAUSE_END = re.compile(r"[.;:,()—]")
_PROVISION = re.compile(r"\b\d+\s+C\.?\s?F\.?\s?R\.?\s+\S|§\s*\d|\bsection\s+\d", re.I)
_DOLLAR = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
_PERCENT = re.compile(
    r"(?<![\d./])(\d+(?:\.\d+)?)(?:\s+(\d+)/(\d+))?\s*percent\b", re.IGNORECASE
)
_NUMBER = r"\d+(?:\.\d+)?"
_FORMULA_FRACTION = re.compile(rf"({_NUMBER})\s*/\s*({_NUMBER})")
_FORMULA_NUMBER = re.compile(rf"(?<![A-Za-z_\d.]){_NUMBER}")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMERIC_DTYPES = frozenset({"money", "rate", "decimal", "integer", "number"})
# Name words that need not be in the source phrase.
_FREE_NAME_WORDS = frozenset({"amount", "state"})


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
    for key in ("inputs", "rules"):
        value = payload.get(key)
        if value is not None and not (
            isinstance(value, list) and all(isinstance(item, dict) for item in value)
        ):
            raise ToolError(f"{what} `{key}` is not a list of mappings: {path}")
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
        if item.get("name")
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
        if isinstance(payload, dict) and isinstance(payload.get("rules"), list):
            names.update(
                str(rule["name"]) for rule in _rules(payload) if rule.get("name")
            )
    if not modules:
        raise ToolError(f"RuleSpec root holds no module: {root}")
    return names


def _words(text: str) -> list[str]:
    words = (re.sub(r"['\u2019]s$", "", word.strip(",")) for word in text.split())
    return [word.lower() for word in words if word]


def _head(words: list[str]) -> frozenset[str]:
    """The head noun phrase: the words up to the first preposition.

    "amount of <the> NP" is NP: `amount` is the one transparent head noun.
    """
    if words[:2] == ["amount", "of"]:
        words = words[3:] if words[2:3] and words[2] in _DETERMINERS else words[2:]
    head: list[str] = []
    for word in words:
        if word in _PHRASE_ENDS or word in _DETERMINERS:
            break
        head.append(word)
    return frozenset(head)


def _phrase_before(text: str, end: int) -> frozenset[str]:
    """Head noun phrase that ends at `end`.

    The phrase runs back to a determiner or the clause start, and on through
    "of" ("a percentage of the Federal poverty level" starts at "percentage"),
    so the head is the amount itself, not the quantity it is a share of.
    """
    clause = _words(_CLAUSE_END.split(text[:end])[-1])
    start = len(clause)
    while start > 0:
        word = clause[start - 1]
        if word in _DETERMINERS and not (start > 1 and clause[start - 2] == "of"):
            break
        start -= 1
    return _head(clause[start:])


def _phrase_after(text: str, start: int) -> frozenset[str]:
    """Head noun phrase that starts at `start` (directly after its determiner)."""
    clause = _CLAUSE_END.split(text[start:])[0]
    run: list[str] = []
    for word in _words(clause):
        if word in _DETERMINERS:
            break
        run.append(word)
    return _head(run)


def actor_amount_phrases(
    excerpt: str, *, agency_is_state: bool
) -> list[frozenset[str]]:
    """Noun phrases that the text makes the object of a State actor's verb."""
    text = _flat(excerpt)
    actor = f"(?:{_STATE_ACTOR}|{_AGENCY_ACTOR})" if agency_is_state else _STATE_ACTOR
    phrases: list[frozenset[str]] = []

    def scan(pattern: str, *, before: bool) -> None:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if before:
                phrases.append(_phrase_before(text, match.start()))
            else:
                phrases.append(_phrase_after(text, match.end()))

    active = rf"(?:{_MODAL}{_BASE}|has\s+{_PARTICIPLE}|{_THIRD})\b"
    determine = rf"(?:{_MODAL}determine|determines)\b"
    # "<the> NP established by the agency"
    scan(rf"\s+{_PARTICIPLE}\s+by\s+{actor}", before=True)
    scan(rf"\s+determined\s+by\s+{actor}\s+{_IN_STATE_PLAN}", before=True)
    # "<the> NP [that] the agency has established"
    scan(rf"\s+{actor}\s+{active}", before=True)
    scan(rf"\s+{actor}\s+{determine}\s+{_IN_STATE_PLAN}", before=True)
    # "the agency must establish [in its State plan] <the> NP"
    scan(rf"{actor}\s+{active}(?:\s+{_IN_STATE_PLAN})?{_OBJECT_START}", before=False)
    scan(
        rf"{actor}\s+{determine}{_OBJECT_START}(?=[^.;]{{0,120}}?\b{_IN_STATE_PLAN})",
        before=False,
    )
    return [phrase for phrase in phrases if phrase]


def plan_level_phrases(excerpt: str) -> list[frozenset[str]]:
    """Levels in effect on a stated date under the State's own plan."""
    text = _flat(excerpt)
    phrases = [
        _head(_words(match.group(0))) | frozenset(re.findall(r"\d{4}", match.group(0)))
        for match in _AFDC_LEVEL.finditer(text)
    ]
    for match in _PLAN_LEVEL.finditer(text):
        phrase = _phrase_before(text, match.start())
        if phrase:
            years = re.findall(r"\b\d{4}\b", match.group(0))
            phrases.append(phrase | frozenset(years))
    return phrases


def name_words(name: str) -> frozenset[str]:
    return frozenset(name.lower().split("_")) - _FREE_NAME_WORDS - {""}


def state_set_trigger(
    name: str, excerpt: str, unit_key: tuple[str, str, str] | None
) -> str | None:
    """Return the closed shape that binds this excerpt to this input name."""
    if (
        unit_key is None
        or unit_key[:2] not in encoder_prompt.CFR_STATE_SET_AMOUNT_PARTS
    ):
        return None
    needed = name_words(name)
    if not needed:
        return None
    agency_is_state = unit_key[:2] in encoder_prompt.CFR_PARTS_WHERE_AGENCY_IS_THE_STATE
    shapes = (
        ("actor", actor_amount_phrases(excerpt, agency_is_state=agency_is_state)),
        ("plan-level", plan_level_phrases(excerpt)),
    )
    for kind, phrases in shapes:
        if any(needed <= phrase for phrase in phrases):
            return kind
    return None


def unit_section_key(citation: str) -> tuple[str, str, str] | None:
    return encoder_prompt.us_cfr_unit_section_key(citation) if citation else None


def _normalised_citation(citation: str) -> str:
    try:
        return corpus_resolver.normalize_corpus_identifier(citation)
    except (corpus_resolver.CorpusResolutionError, ValueError):
        return citation.strip()


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


def literal_values(source: str) -> list[tuple[str, list[float], float]]:
    """Dollar amounts and percentages the source states: label, accepted
    forms, and the tolerance (exact, but for a fraction such as 133 1/3)."""
    flat = _flat(source)
    values: list[tuple[str, list[float], float]] = []
    for raw in _DOLLAR.findall(flat):
        values.append((f"${raw}", [float(raw.replace(",", ""))], 1e-9))
    for whole, numerator, denominator in _PERCENT.findall(flat):
        number = float(whole)
        label = whole
        tolerance = 1e-9
        if numerator and float(denominator):
            number += float(numerator) / float(denominator)
            label = f"{whole} {numerator}/{denominator}"
            tolerance = 1e-3
        values.append((f"{label} percent", [number, number / 100], tolerance))
    return values


def formula_numbers(formula_text: str) -> list[float]:
    numbers = [float(raw) for raw in _FORMULA_NUMBER.findall(formula_text)]
    for numerator, denominator in _FORMULA_FRACTION.findall(formula_text):
        if float(denominator):
            numbers.append(float(numerator) / float(denominator))
    return numbers


def check(
    root: Path, unit: str, source: str, module: Path, baseline: Path | None
) -> tuple[list[str], list[str]]:
    """Return (owner-table rows, failures); raise ToolError when it cannot check."""
    payload = _payload(module, "module")
    verification = (payload.get("module") or {}).get("source_verification") or {}
    citation = str(verification.get("corpus_citation_path") or "")
    if not citation:
        raise ToolError(
            f"module has no module.source_verification.corpus_citation_path: {module}"
        )
    if not unit or _normalised_citation(citation) != _normalised_citation(unit):
        raise ToolError(
            f"module corpus_citation_path `{citation}` is not the unit `{unit}`"
        )
    flat_source = _flat(source)
    if not flat_source:
        raise ToolError("unit source text is empty")
    unit_key = unit_section_key(unit)
    in_scope = encoder_prompt.cfr_state_set_amount_rule_applies(unit)
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
    bound: dict[str, str] = {}  # excerpt -> the one input it serves
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
        description = str(item.get("description") or "")
        trigger = None
        taken = None
        shown = ""
        for excerpt in excerpts:
            kind = state_set_trigger(name, excerpt, unit_key)
            if kind is None:
                continue
            if bound.setdefault(excerpt, name) != name:
                taken = bound[excerpt]
                continue
            trigger, shown = kind, excerpt
            break
        any_trigger = any_trigger or trigger is not None
        shown = " ".join((shown or (excerpts[0] if excerpts else "")).split()[:12])
        shape = "implicit" if item.get("implicit") else "root"
        rows.append(
            f"| {module.stem} | {name} | {shape} | "
            f"{item.get('entity', '-')}/{item.get('dtype', '-')}/{item.get('period', '-')}"
            f" | {trigger or '-'} | {shown} |"
        )
        if not in_scope:
            fail(
                f"scope: `{name}` is a new input of `{unit}`, a unit outside "
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
            if taken is not None:
                fail(
                    f"trigger: the only excerpt that names `{name}` already "
                    f"serves `{taken}`; one excerpt serves one input"
                )
            else:
                fail(
                    f"trigger: no verified source excerpt makes `{name}` the "
                    "object of a State actor's verb, or a dated State plan level"
                )
        if name in exports:
            fail(f"exported: `{name}` is a rule name under the RuleSpec root")
    if rows:
        numbers = formula_numbers(formula_text)
        for label, forms, tolerance in literal_values(source):
            if not any(
                math.isclose(number, form, rel_tol=tolerance)
                for number in numbers
                for form in forms
            ):
                fail(f"literal: {label} of the unit source is in no formula")
    if any_trigger and not has_delegates_relation(payload):
        fail(
            "relation: the module has no `kind: source_relation` record with "
            "`source_relation.type: delegates` for the State-set amount"
        )
    return rows, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rulespec-root", type=Path, required=True)
    parser.add_argument(
        "--unit", required=True, help="corpus citation path of the encoded unit"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args(argv)
    try:
        rows, failures = check(
            args.rulespec_root,
            args.unit,
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
