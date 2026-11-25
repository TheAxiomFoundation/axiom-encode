"""A/B report aid: list the new inputs of an encoded CFR module for a reader.

This tool is NOT a gate and NOT a validator. It prints rows that a person
reads after an A/B run. Its checks are lexical; a FAIL or WARN line points the
reader at a row, and a row without one is not proof that the encoding is
right. It still fails closed on bad input, and it still fails the known
federal-amount attacks.

Usage:
    python scripts/check_state_set_inputs.py --rulespec-root <rulespec-us/us> \\
        --unit <us/regulation/42/435/733> --source <source.txt> \\
        --module <candidate.yaml> [--baseline <baseline.yaml>]

Every input of the candidate that the baseline does not have is listed: root
`inputs` entries and implicit inputs (a free name in a formula that is no
rule, no import and no root input).

Two tables:

- State-set amount inputs (checked). A new input is in this table when it
  claims to be a State-set amount: a root `inputs` entry with a numeric
  `dtype` whose `description` names a provision (`N CFR P.S`, `§ P.S`,
  `section N`, a bare `P.S` of the unit's own part) or names the State or the
  agency; or any new input that a verified source phrase binds (see trigger).
- Fact inputs (not checked). Every other new input: Booleans, income,
  household facts, implicit inputs. They never fail. The reader checks that
  no amount hides there.

The last line is machine-readable, for A/B scoring:

    state_set_inputs=<n> failures=<n> warnings=<n>

`state_set_inputs` counts the checked inputs that have a trigger. A branch
counts as encoded with a State-set input only when `state_set_inputs >= 1 and
failures == 0`. A deferred module (`rules: []`) prints `state_set_inputs=0`
and exits 0; that is not a success.

Exit 2 with an `ERROR` line (no table, no summary line) when the tool cannot
do its work: a path that does not exist, a file that is not parsable YAML, a
module or baseline whose `module`, `inputs`, `rules`, `metadata`, `proof`,
`atoms` or `versions` has the wrong shape, an input without a name, a
RuleSpec root with no module, an empty source, or a module whose
`module.source_verification.corpus_citation_path` is missing or is not the
unit given with `--unit`.

Exit 1 when a checked input has a FAIL line:

1. scope: the unit is on the allow-list of CFR parts for which the State-set
   amount rule is emitted (`CFR_STATE_SET_AMOUNT_PARTS`).
2. description: the `description` names a provision.
3. excerpt: every proof excerpt of a rule that uses the input is verbatim
   (whitespace-normalised) in the unit's source text.
4. trigger: one phrase of the unit source, inside a verified excerpt, binds
   the input. The phrase has a closed shape, and the input's name contains
   every word of the head noun phrase (plurals normalised; `amount`, `state`
   and `such` are free; the name may add qualifiers):
   - `actor`: the State, the State agency or (only in the CFR parts where the
     agency is the State agency) the agency is the actor of a verb of the
     closed list establish, set, elect, specify, select, choose, and the head
     noun phrase is the object: "<the> NP established by the agency", "<the>
     NP the State elects", "the agency sets <the> NP". The verb determine
     counts only with "in/under its State plan".
   - `plan-level`: a level in effect on a stated calendar date under the
     State's own plan: "State's AFDC payment|income standard in effect as of
     <date>", "<the> NP under the [Medicaid] State plan ... as of <date>",
     "<the> NP in effect under the State's Medicaid plan on <date>".
   The head noun phrase ends at the first preposition, so in "a percentage of
   the Federal poverty level that the State elects" only a name with
   `percentage` binds. One source span (character range of the phrase in the
   normalised source) binds one input, whatever excerpt cites it.
5. relation: when an input has a trigger, the module holds a
   `kind: source_relation` record with `source_relation.type: delegates`.
6. exported: no rule anywhere under the RuleSpec root has the same name.
7. literals: when the module has a checked input, every dollar amount, every
   "N percent" / "N%" (also "133 1/3 percent") and every "one-half"-style
   fraction of the unit source is a number in some `versions[].formula`.

WARN (no exit code): `qualifier` lists the words of an input's name that are
not in its bound excerpt. Known residual: a name with a federal qualifier
(`ssi_income_standard`) binds a generic phrase ("the income standard
established by the agency"); the reader sees the extra word `ssi`.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import NamedTuple

import yaml

from axiom_encode import corpus_resolver
from axiom_encode.harness import evals
from axiom_encode.prompts import encoder as encoder_prompt

_DETERMINERS = frozenset({"the", "a", "an", "any", "each", "such"})
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
    r"\s+(?:in\s+effect\s+)?under\s+the\s+"
    r"(?:(?:Medicaid\s+)?State\s+plan|State's\s+Medicaid\s+plan)"
    r"(?:\s+or\s+(?:a\s+)?waiver\s+of\s+the\s+(?:State\s+)?plan)?"
    r"(?:\s+under\s+section\s+1115(?:\s+of\s+the\s+Act)?)?"
    rf",?\s+{_ON_DATE}"
)
_CLAUSE_END = re.compile(r"[.;:,()—]")
_PROVISION = re.compile(r"\b\d+\s+C\.?\s?F\.?\s?R\.?\s+\S|§\s*\d|\bsection\s+\d", re.I)
_DOLLAR = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
_PERCENT = re.compile(
    r"(?<![\d./])(\d+(?:\.\d+)?)(?:\s+(\d+)/(\d+))?\s*(?:percent\b|%)",
    re.IGNORECASE,
)
_WORD_FRACTIONS = {
    "one-half": 1 / 2,
    "one-third": 1 / 3,
    "two-thirds": 2 / 3,
    "one-fourth": 1 / 4,
    "one-quarter": 1 / 4,
    "three-fourths": 3 / 4,
    "three-quarters": 3 / 4,
}
_WORD_FRACTION = re.compile(r"\b(" + "|".join(_WORD_FRACTIONS) + r")\b", re.IGNORECASE)
_NUMBER = r"\d+(?:\.\d+)?"
_FORMULA_FRACTION = re.compile(rf"({_NUMBER})\s*/\s*({_NUMBER})")
_FORMULA_NUMBER = re.compile(rf"(?<![A-Za-z_\d.]){_NUMBER}")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMERIC_DTYPES = frozenset({"money", "rate", "decimal", "integer", "number"})
# Phrase words that the input's name need not contain.
_FREE_PHRASE_WORDS = frozenset({"amount", "state", "such"})
_STATE_CLAIM = re.compile(r"\b(?:State|agency)\b")


class ToolError(Exception):
    """The tool cannot do its work; the run must not count as a pass."""


class Report:
    """What the tool found: two tables, FAIL lines, WARN lines, one count."""

    def __init__(self) -> None:
        self.rows: list[str] = []
        self.fact_rows: list[str] = []
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.state_set_inputs = 0

    @property
    def summary(self) -> str:
        return (
            f"state_set_inputs={self.state_set_inputs} "
            f"failures={len(self.failures)} warnings={len(self.warnings)}"
        )


class Phrase(NamedTuple):
    kind: str
    words: frozenset[str]  # head noun phrase
    span: tuple[int, int]  # of the verb or plan construct, in the flat source


def _flat(text: str) -> str:
    return " ".join(text.split())


def _read(path: Path, what: str) -> str:
    if not path.is_file():
        raise ToolError(f"{what} not found: {path}")
    try:
        return path.read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolError(f"{what} is not readable: {path}") from exc


def _mappings(value: object, what: str, path: Path) -> list[dict]:
    if value is None:
        return []
    if not (isinstance(value, list) and all(isinstance(i, dict) for i in value)):
        raise ToolError(f"{what} is not a list of mappings: {path}")
    return value


def _mapping(value: object, what: str, path: Path) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ToolError(f"{what} is not a mapping: {path}")
    return value


def _payload(path: Path, what: str) -> dict:
    """Load a module or baseline and refuse every shape the checks rely on."""
    text = _read(path, what)
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ToolError(f"{what} is not parsable YAML: {path}") from exc
    if not isinstance(payload, dict):
        raise ToolError(f"{what} is not a YAML mapping: {path}")
    module = _mapping(payload.get("module"), f"{what} `module`", path)
    _mapping(module.get("source_verification"), f"{what} `source_verification`", path)
    for item in _mappings(payload.get("inputs"), f"{what} `inputs`", path):
        if not item.get("name"):
            raise ToolError(f"{what} has an input without a name: {path}")
    for rule in _mappings(payload.get("rules"), f"{what} `rules`", path):
        _mappings(rule.get("versions"), f"{what} `versions`", path)
        metadata = _mapping(rule.get("metadata"), f"{what} `metadata`", path)
        proof = _mapping(metadata.get("proof"), f"{what} `proof`", path)
        for atom in _mappings(proof.get("atoms"), f"{what} `atoms`", path):
            _mapping(atom.get("source"), f"{what} atom `source`", path)
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
    inputs = {str(item["name"]): dict(item) for item in payload.get("inputs") or []}
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
    words = (re.sub(r"['’]s$", "", word.strip(",")) for word in text.split())
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

    The phrase runs back to a determiner, and on through "of" ("a percentage
    of the Federal poverty level" starts at "percentage"), so the head is the
    amount itself, not the quantity it is a share of. With no determiner in
    the clause, the phrase starts after the last preposition other than "of"
    ("subject to reasonable limits" is "reasonable limits").
    """
    clause = _words(_CLAUSE_END.split(text[:end])[-1])
    start = len(clause)
    while start > 0:
        word = clause[start - 1]
        if word in _DETERMINERS and not (start > 1 and clause[start - 2] == "of"):
            return _head(clause[start:])
        start -= 1
    for index in range(len(clause) - 1, -1, -1):
        if clause[index] in _PHRASE_ENDS and clause[index] != "of":
            head = _head(clause[index + 1 :])
            if head:
                return head
    return _head(clause)


def _phrase_after(text: str, start: int) -> frozenset[str]:
    """Head noun phrase that starts at `start` (directly after its determiner)."""
    clause = _CLAUSE_END.split(text[start:])[0]
    run: list[str] = []
    for word in _words(clause):
        if word in _DETERMINERS:
            break
        run.append(word)
    return _head(run)


def _actor_phrases(text: str, *, agency_is_state: bool) -> list[Phrase]:
    actor = f"(?:{_STATE_ACTOR}|{_AGENCY_ACTOR})" if agency_is_state else _STATE_ACTOR
    found: list[Phrase] = []

    def scan(pattern: str, *, before: bool) -> None:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            words = (
                _phrase_before(text, match.start())
                if before
                else _phrase_after(text, match.end())
            )
            if words:
                found.append(Phrase("actor", words, match.span()))

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
    return found


def _plan_phrases(text: str) -> list[Phrase]:
    found = [
        Phrase("plan-level", _head(_words(match.group(0))), match.span())
        for match in _AFDC_LEVEL.finditer(text)
    ]
    for match in _PLAN_LEVEL.finditer(text):
        words = _phrase_before(text, match.start())
        if words:
            found.append(Phrase("plan-level", words, match.span()))
    return found


def actor_amount_phrases(
    excerpt: str, *, agency_is_state: bool
) -> list[frozenset[str]]:
    """Noun phrases that the text makes the object of a State actor's verb."""
    found = _actor_phrases(_flat(excerpt), agency_is_state=agency_is_state)
    return [phrase.words for phrase in found]


def plan_level_phrases(excerpt: str) -> list[frozenset[str]]:
    """Levels in effect on a stated date under the State's own plan."""
    return [phrase.words for phrase in _plan_phrases(_flat(excerpt))]


def source_phrases(
    flat_source: str, unit_key: tuple[str, str, str] | None
) -> list[Phrase]:
    """Every phrase of a closed shape in the unit source; none off the allow-list."""
    if (
        unit_key is None
        or unit_key[:2] not in encoder_prompt.CFR_STATE_SET_AMOUNT_PARTS
    ):
        return []
    agency_is_state = unit_key[:2] in encoder_prompt.CFR_PARTS_WHERE_AGENCY_IS_THE_STATE
    return _actor_phrases(flat_source, agency_is_state=agency_is_state) + _plan_phrases(
        flat_source
    )


def _singular(word: str) -> str:
    return word[:-1] if word.endswith("s") and not word.endswith("ss") else word


def name_binds(name: str, phrase_words: frozenset[str]) -> bool:
    """True when the input's name contains every word of the head noun phrase."""
    name_words = {_singular(word) for word in name.lower().split("_") if word}
    core = {_singular(word) for word in phrase_words} - _FREE_PHRASE_WORDS
    return bool(core) and core <= name_words


def state_set_trigger(
    name: str, excerpt: str, unit_key: tuple[str, str, str] | None
) -> str | None:
    """Return the closed shape by which this text binds this input name."""
    for phrase in source_phrases(_flat(excerpt), unit_key):
        if name_binds(name, phrase.words):
            return phrase.kind
    return None


def _occurrences(text: str, part: str) -> list[tuple[int, int]]:
    spans = []
    start = text.find(part)
    while start != -1:
        spans.append((start, start + len(part)))
        start = text.find(part, start + 1)
    return spans


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


def claims_state_set_amount(item: dict, unit_key: tuple[str, str, str] | None) -> bool:
    """A numeric root input whose description cites a provision or the State."""
    if item.get("implicit"):
        return False
    if str(item.get("dtype") or "").lower() not in _NUMERIC_DTYPES:
        return False
    description = str(item.get("description") or "")
    return provision_named(description, unit_key) or bool(
        _STATE_CLAIM.search(description)
    )


def has_delegates_relation(payload: dict) -> bool:
    return any(
        rule.get("kind") == "source_relation"
        and isinstance(rule.get("source_relation"), dict)
        and rule["source_relation"].get("type") == "delegates"
        for rule in _rules(payload)
    )


def literal_values(source: str) -> list[tuple[str, list[float], float]]:
    """Dollar amounts, percentages and word fractions the source states:
    label, accepted forms, and the tolerance (exact, but for thirds)."""
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
    for raw in _WORD_FRACTION.findall(flat):
        values.append((raw.lower(), [_WORD_FRACTIONS[raw.lower()]], 1e-3))
    return values


def formula_numbers(formula_text: str) -> list[float]:
    numbers = [float(raw) for raw in _FORMULA_NUMBER.findall(formula_text)]
    for numerator, denominator in _FORMULA_FRACTION.findall(formula_text):
        if float(denominator):
            numbers.append(float(numerator) / float(denominator))
    return numbers


def check(
    root: Path, unit: str, source: str, module: Path, baseline: Path | None
) -> Report:
    """Return the report; raise ToolError when the tool cannot check."""
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
    phrases = source_phrases(flat_source, unit_key)
    report = Report()

    def fail(message: str) -> None:
        if message not in report.failures:
            report.failures.append(message)

    bound: dict[tuple[int, int], str] = {}  # source span -> the one input it serves
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
            if (atom.get("source") or {}).get("excerpt")
        ]
        # Only text that is verbatim in the unit source counts from here on.
        excerpts = [excerpt for excerpt in cited if excerpt in flat_source]
        trigger = None
        taken = None
        bound_excerpt = ""
        for excerpt in excerpts:
            for start, end in _occurrences(flat_source, excerpt):
                for phrase in phrases:
                    inside = start <= phrase.span[0] and phrase.span[1] <= end
                    if not (inside and name_binds(name, phrase.words)):
                        continue
                    if bound.setdefault(phrase.span, name) != name:
                        taken = bound[phrase.span]
                        continue
                    trigger, bound_excerpt = phrase.kind, excerpt
                    break
                if trigger:
                    break
            if trigger:
                break
        shown = " ".join(
            (bound_excerpt or (excerpts[0] if excerpts else "")).split()[:12]
        )
        shape = "implicit" if item.get("implicit") else "root"
        row = (
            f"| {module.stem} | {name} | {shape} | "
            f"{item.get('entity', '-')}/{item.get('dtype', '-')}/{item.get('period', '-')}"
            f" | {trigger or '-'} | {shown} |"
        )
        if trigger is None and not claims_state_set_amount(item, unit_key):
            report.fact_rows.append(row)
            continue
        report.rows.append(row)
        report.state_set_inputs += trigger is not None
        description = str(item.get("description") or "")
        if not in_scope:
            fail(
                f"scope: `{name}` is a new input of `{unit}`, a unit outside "
                f"the allow-list ({allowed})"
            )
        if not provision_named(description, unit_key):
            fail(f"description: `{name}` has no description naming a provision")
        for excerpt in cited:
            if excerpt not in flat_source:
                fail(
                    f"excerpt: a proof excerpt cited for `{name}` is not verbatim "
                    f'in the unit source: "{" ".join(excerpt.split()[:12])}"'
                )
        if in_scope and trigger is None:
            if taken is not None:
                fail(
                    f"trigger: the only source phrase that names `{name}` "
                    f"already serves `{taken}`; one phrase serves one input"
                )
            else:
                fail(
                    f"trigger: no verified source excerpt makes `{name}` the "
                    "object of a State actor's verb, or a dated State plan level"
                )
        if trigger is not None:
            excerpt_words = {
                _singular(word) for word in re.findall(r"\w+", bound_excerpt.lower())
            }
            extra = sorted(
                word
                for word in name.lower().split("_")
                if word
                and word not in _FREE_PHRASE_WORDS
                and _singular(word) not in excerpt_words
            )
            if extra:
                report.warnings.append(
                    f"qualifier: `{name}` adds words that are not in its source "
                    f"excerpt: {', '.join(extra)}"
                )
        if name in exports:
            fail(f"exported: `{name}` is a rule name under the RuleSpec root")
    if report.rows:
        numbers = formula_numbers(formula_text)
        for label, forms, tolerance in literal_values(source):
            if not any(
                math.isclose(number, form, rel_tol=tolerance)
                for number in numbers
                for form in forms
            ):
                fail(f"literal: {label} of the unit source is in no formula")
    if report.state_set_inputs and not has_delegates_relation(payload):
        fail(
            "relation: the module has no `kind: source_relation` record with "
            "`source_relation.type: delegates` for the State-set amount"
        )
    return report


_TABLE_HEAD = (
    "| unit | input | form | entity/dtype/period | trigger | citing excerpt |\n"
    "|---|---|---|---|---|---|"
)


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
        report = check(
            args.rulespec_root,
            args.unit,
            _read(args.source, "unit source"),
            args.module,
            args.baseline,
        )
    except ToolError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 2
    print("State-set amount inputs (checked):")
    print(_TABLE_HEAD)
    print("\n".join(report.rows))
    print("\nFact inputs (not checked):")
    print(_TABLE_HEAD)
    print("\n".join(report.fact_rows))
    for failure in report.failures:
        print(f"FAIL {failure}")
    for warning in report.warnings:
        print(f"WARN {warning}")
    print(report.summary)
    return 1 if report.failures else 0


if __name__ == "__main__":
    sys.exit(main())
