"""Independent (blind) test author: a shadow tool that gates nothing.

The model that encodes a rule also writes its companion ``.test.yaml`` from
its own formula, so those tests show that the rule agrees with itself. This
module asks a *second* model for test cases written from the source text and
the rule's declared interface only, runs them against the candidate with the
companion-test runner, and reports agree / disagree / not-runnable per case.

Blindness is structural: :func:`extract_rule_interface` copies an allow-list
of declaration fields (names, entities, dtypes, periods, units) and nothing
else. :func:`find_blindness_violations` and :func:`find_interface_leaks` then
re-check the rendered prompt against the module's formulas, values, dates,
prose and existing test cases before any model sees it.

Nothing here is a validation check. No check id is declared, the encode loop
does not call it, and a disagreement changes no validation result. See
``docs/blind-test-author.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Protocol

import yaml

from axiom_encode.constants import REVIEWER_CLI_MODEL

REPORT_SCHEMA = "axiom-encode/blind-test-author-report/v1"

STATUS_AGREE = "agree"
STATUS_DISAGREE = "disagree"
STATUS_NOT_RUNNABLE = "not_runnable"

# A quoted sentence shorter than this proves nothing: "the" is verbatim in
# every source.
MIN_SOURCE_SENTENCE_CHARS = 20
DEFAULT_MAX_CASES = 8
DEFAULT_TIMEOUT_SECONDS = 600

# The start date of a unit whose text states no date is NOT decided. The
# probe below only measures against one candidate convention.
ASSUMPTION_REFERENCE = "TheAxiomFoundation/axiom-encode#2212"
FINDING_NOT_PROBED = "not_probed"
FINDING_BEFORE_ASSUMED_FLOOR = "before_assumed_evidence_floor"
FINDING_NOT_BEFORE_ASSUMED_FLOOR = "not_before_assumed_evidence_floor"
FINDING_BEFORE_STATED_DATE = "before_source_stated_date"
FINDING_NOT_BEFORE_STATED_DATE = "not_before_source_stated_date"

_JUDGMENT_VALUES = ("holds", "not_holds", "undetermined")
_BOOLEAN_DTYPES = {"boolean", "bool"}
_NUMBER_DTYPES = {"money", "integer", "count", "rate", "decimal", "int", "float"}
_TEXT_DTYPES = {"string", "text", "str"}
# Declaration words are a closed vocabulary or a strict shape. Anything else
# in such a field is dropped, never copied: the field is module-controlled.
_KIND_WORDS = {
    "derived",
    "parameter",
    "data_relation",
    "source_relation",
    "derived_relation",
}
_DTYPE_WORDS = _BOOLEAN_DTYPES | _NUMBER_DTYPES | _TEXT_DTYPES | {"judgment", "date"}
_PERIOD_WORDS = {"year", "month", "week", "day", "quarter"}
_UNIT_WORDS = {
    "years",
    "months",
    "weeks",
    "days",
    "hours",
    "calendar_year",
    "percent",
    "persons",
}
_CURRENCY_RE = re.compile(r"[A-Z]{3}")
_ENTITY_RE = re.compile(r"[A-Z][A-Za-z0-9]{0,63}")
_MONTH_RE = re.compile(r"\d{4}-(0[1-9]|1[0-2])")
_ISO_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CITATION_RE = re.compile(r"[a-z0-9][a-z0-9-]*(/[A-Za-z0-9][A-Za-z0-9._()-]*)+")
_NO_FORMULA_VERSION_RE = re.compile(r"has no formula version at")
_ENUMERATION_KEYS = ("values", "allowed_values", "enum")
_CASE_KEYS = {"name", "period", "inputs", "expected", "source_sentence"}
_RESPONSE_KEYS = {"cases", "stated_commencement"}
_COMMENCEMENT_KEYS = {"month", "source_sentence"}


class BlindTestAuthorError(RuntimeError):
    """The tool could not do its job; never a statement about the rule."""


class BlindnessViolation(BlindTestAuthorError):
    """The prompt would show the second model something it must not see."""


class InterfaceLeak(BlindnessViolation):
    """A name or path in the prompt carries a formula value or a date."""


class BlindAuthorBackendError(BlindTestAuthorError):
    """The second-model backend failed or is not available."""


# ---------------------------------------------------------------------------
# Interface extraction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InterfaceInput:
    name: str
    dtype: str | None = None
    entity: str | None = None
    period: str | None = None
    unit: str | None = None
    enumeration: tuple[str, ...] = ()
    declared: bool = True


@dataclass(frozen=True)
class InterfaceOutput:
    name: str
    kind: str
    dtype: str | None = None
    entity: str | None = None
    period: str | None = None
    unit: str | None = None
    enumeration: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuleInterface:
    """What the second model may know about the candidate. Nothing else.

    ``dropped`` names the declaration fields that were not copied because
    their value was not a declaration word. It is for the report, never for
    the prompt, and holds field locations only, not the dropped values.
    """

    module: str
    inputs: tuple[InterfaceInput, ...]
    outputs: tuple[InterfaceOutput, ...]
    relations: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()

    def input(self, name: str) -> InterfaceInput | None:
        return next((item for item in self.inputs if item.name == name), None)

    def output(self, name: str) -> InterfaceOutput | None:
        return next((item for item in self.outputs if item.name == name), None)

    def names(self) -> set[str]:
        names = {item.name for item in (*self.inputs, *self.outputs)}
        names.update(self.relations)
        for item in (*self.inputs, *self.outputs):
            names.update(item.enumeration)
        return names

    def to_json(self) -> dict[str, Any]:
        def clean(item: Any) -> dict[str, Any]:
            return {
                key: (list(value) if isinstance(value, tuple) else value)
                for key, value in asdict(item).items()
                if value not in (None, (), [])
            }

        return {
            "module": self.module,
            "inputs": [clean(item) for item in self.inputs],
            "outputs": [clean(item) for item in self.outputs],
            "relations": list(self.relations),
        }


def _name(value: Any) -> str | None:
    """Return ``value`` only when it is a bare identifier."""
    if isinstance(value, str) and _NAME_RE.fullmatch(value.strip()):
        return value.strip()
    return None


def _type_word(value: Any, field_name: str) -> str | None:
    """Return a declaration word for one field, else None.

    ``kind``, ``dtype`` and ``period`` are closed vocabularies. ``unit`` is a
    three-letter currency code or one of a few unit words. ``entity`` is one
    CamelCase word. Free text, a number or a formula in such a field is
    dropped rather than copied.
    """
    if not isinstance(value, str):
        return None
    word = value.strip()
    if field_name == "kind":
        return word if word in _KIND_WORDS else None
    if field_name == "dtype":
        return word if word.lower() in _DTYPE_WORDS else None
    if field_name == "period":
        return word if word.lower() in _PERIOD_WORDS else None
    if field_name == "unit":
        return word if _CURRENCY_RE.fullmatch(word) or word in _UNIT_WORDS else None
    if field_name == "entity":
        return word if _ENTITY_RE.fullmatch(word) else None
    raise ValueError(f"not a declaration field: {field_name}")


def _enumeration(item: Mapping[str, Any]) -> tuple[str, ...]:
    for key in _ENUMERATION_KEYS:
        values = item.get(key)
        if isinstance(values, list) and values:
            names = tuple(name for name in map(_name, values) if name)
            if len(names) == len(values):
                return names
    return ()


def _compiled_implicit_inputs(
    compiled_payload: Mapping[str, Any] | None,
) -> dict[str, dict[str, set[str]]]:
    """Collect input *names* the compiled program reads, with shallow hints.

    Some modules do not declare boolean facts; the engine accepts them as
    implicit inputs. Only the name, the reading rule's entity and period, and
    whether the fact is compared to a boolean literal are taken. No operator,
    structure or literal value leaves this function.
    """
    found: dict[str, dict[str, set[str]]] = {}
    program = (
        compiled_payload.get("program") if isinstance(compiled_payload, Mapping) else {}
    )
    if not isinstance(program, Mapping):
        return found

    def walk(node: Any, entity: str | None, period: str | None) -> None:
        if isinstance(node, list):
            for child in node:
                walk(child, entity, period)
            return
        if not isinstance(node, Mapping):
            return
        if node.get("kind") == "input":
            name = _name(node.get("name"))
            if name:
                found.setdefault(
                    name, {"entity": set(), "period": set(), "literal": set()}
                )
                if entity:
                    found[name]["entity"].add(entity)
                if period:
                    found[name]["period"].add(period)
        if node.get("kind") == "comparison":
            sides = [node.get("left"), node.get("right")]
            literal_kinds = {
                str(side.get("value", {}).get("kind"))
                for side in sides
                if isinstance(side, Mapping)
                and side.get("kind") == "literal"
                and isinstance(side.get("value"), Mapping)
            }
            for side in sides:
                if isinstance(side, Mapping) and side.get("kind") == "input":
                    name = _name(side.get("name"))
                    if name:
                        found.setdefault(
                            name, {"entity": set(), "period": set(), "literal": set()}
                        )["literal"].update(literal_kinds or {"other"})
        for child in node.values():
            walk(child, entity, period)

    for item in program.get("derived", []) or []:
        if isinstance(item, Mapping):
            walk(
                [item.get("expr"), item.get("versions")],
                _type_word(item.get("entity"), "entity"),
                _type_word(item.get("period"), "period"),
            )
    return found


def extract_rule_interface(
    rulespec_payload: Mapping[str, Any],
    *,
    module: str,
    compiled_payload: Mapping[str, Any] | None = None,
) -> RuleInterface:
    """Derive the declared interface of a RuleSpec module.

    Allow-list only. From ``inputs`` and ``rules`` entries this copies the
    name and the declaration words ``kind``, ``dtype``, ``entity``,
    ``period``, ``unit`` (see :func:`_type_word`) and a list of enumeration
    names. It never reads ``versions``, ``formula``, ``effective_from``,
    ``metadata``, ``source``, ``summary``, labels, descriptions or comments.
    ``compiled_payload`` supplies only the names of implicit (undeclared)
    inputs.
    """
    if not isinstance(rulespec_payload, Mapping):
        raise BlindTestAuthorError("RuleSpec module must be a YAML mapping")
    dropped: list[str] = []

    def words(item: Mapping[str, Any], where: str) -> dict[str, str | None]:
        result: dict[str, str | None] = {}
        for field_name in ("dtype", "entity", "period", "unit"):
            result[field_name] = _type_word(item.get(field_name), field_name)
            if item.get(field_name) is not None and result[field_name] is None:
                dropped.append(f"{where}.{field_name}")
        return result

    inputs: dict[str, InterfaceInput] = {}
    for item in rulespec_payload.get("inputs") or []:
        if not isinstance(item, Mapping):
            continue
        name = _name(item.get("name"))
        if name is None:
            continue
        inputs[name] = InterfaceInput(
            name=name, enumeration=_enumeration(item), **words(item, f"inputs.{name}")
        )
    outputs: list[InterfaceOutput] = []
    relations: list[str] = []
    for item in rulespec_payload.get("rules") or []:
        if not isinstance(item, Mapping):
            continue
        name = _name(item.get("name"))
        kind = _type_word(item.get("kind"), "kind")
        if name is None or kind is None:
            continue
        if kind.endswith("relation"):
            relations.append(name)
            continue
        outputs.append(
            InterfaceOutput(
                name=name,
                kind=kind,
                # A parameter's values are exactly what must stay hidden.
                enumeration=_enumeration(item) if kind == "derived" else (),
                **words(item, f"rules.{name}"),
            )
        )
    output_names = {item.name for item in outputs}
    for name, hints in sorted(_compiled_implicit_inputs(compiled_payload).items()):
        if name in inputs or name in output_names:
            continue
        inputs[name] = InterfaceInput(
            name=name,
            dtype="Boolean" if hints["literal"] == {"bool"} else None,
            entity=next(iter(hints["entity"])) if len(hints["entity"]) == 1 else None,
            period=next(iter(hints["period"])) if len(hints["period"]) == 1 else None,
            declared=False,
        )
    return RuleInterface(
        module=module,
        inputs=tuple(inputs.values()),
        outputs=tuple(outputs),
        relations=tuple(relations),
        dropped=tuple(dropped),
    )


# ---------------------------------------------------------------------------
# Prompt and blindness guard
# ---------------------------------------------------------------------------

_INSTRUCTIONS = """\
You are an independent test author for an encoded legal rule.

You see the SOURCE TEXT of one legal provision and the INTERFACE of a rule
that someone else encoded from it: the names and types of its inputs and
outputs. You do not see the rule's formulas, its values or its existing
tests. Do not guess what they are. Read the source text and decide, from the
text alone, what the outputs must be for concrete inputs.

Write at most {max_cases} test cases. Each case must follow from one sentence
of the source text, which you quote exactly. Cover each stated condition,
each exception, and each stated date or amount boundary. If the source text
does not determine an expected value, do not write that case. An input or
output name is a label chosen by the encoder; when a name seems to disagree
with the source text, follow the source text.

Answer with one JSON object and nothing else. No prose, no markdown fence.

{{
  "cases": [
    {{
      "name": "<short_snake_case_name>",
      "period": "<YYYY-MM>",
      "inputs": {{"<input name>": <value>}},
      "expected": {{"<output name>": <value>}},
      "source_sentence": "<one whole sentence or whole list item, copied exactly>"
    }}
  ],
  "stated_commencement": null
}}

Format rules:
- "inputs" keys and "expected" keys are names from the INTERFACE, exactly.
- Assign every input the expected output depends on, including false facts.
- Boolean values are true or false. Judgment outputs are "holds",
  "not_holds" or "undetermined". Money, Integer, Count, Rate and Decimal
  values are JSON numbers. A rate is a fraction of one, not a percentage.
- "period" is one calendar month, "YYYY-MM".{period_rule}
- "source_sentence" is one whole sentence or one whole list item of the
  source text (or several in a row), copied character for character from its
  first word to its end punctuation, at least {min_chars} characters long.
  Do not start or stop in the middle of a sentence. A case with a quote that
  is not in the source text, or that is cut, is rejected.
- "stated_commencement" stays null unless the source text itself states the
  date from which this provision applies. Then it is
  {{"month": "<YYYY-MM>", "source_sentence": "<the whole sentence that states the date>"}}.
  Do not infer a date from anything else.
"""


@dataclass(frozen=True)
class BlindPrompt:
    """The prompt in parts.

    ``instructions`` is static text plus caller-supplied facts (the corpus
    as-of date, a period hint). ``citation_line`` and ``interface_block``
    are the parts derived from the module. ``source_text`` is corpus text.
    """

    instructions: str
    citation_line: str
    interface_block: str
    source_text: str

    @property
    def module_derived(self) -> str:
        return f"{self.citation_line}{self.interface_block}"

    @property
    def controlled(self) -> str:
        return f"{self.instructions}\n{self.module_derived}"

    @property
    def text(self) -> str:
        return (
            f"{self.controlled}\n=== SOURCE TEXT (begin) ===\n{self.source_text}\n"
            "=== SOURCE TEXT (end) ===\n"
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def require_citation_path(citation_path: str) -> str:
    """Return a canonical corpus citation path, or refuse it.

    The path comes from the module, so it is module-controlled text. Only the
    exact ``<jurisdiction>/<document-class>/<segment>...`` identity that the
    corpus resolver accepts may enter the prompt; free text may not.
    """
    from axiom_encode.corpus_resolver import (
        CorpusResolutionError,
        require_canonical_corpus_citation_path,
    )

    if not isinstance(citation_path, str) or not _CITATION_RE.fullmatch(citation_path):
        raise BlindnessViolation(
            "the citation path is not a canonical corpus citation path; free "
            "text is refused"
        )
    try:
        return require_canonical_corpus_citation_path(citation_path)
    except CorpusResolutionError as exc:
        raise BlindnessViolation(f"the citation path is refused: {exc}") from exc


def build_blind_prompt(
    *,
    source_text: str,
    interface: RuleInterface,
    citation_path: str | None = None,
    source_as_of: str | None = None,
    period_hint: str | None = None,
    max_cases: int = DEFAULT_MAX_CASES,
) -> BlindPrompt:
    """Render the prompt from the source text and the interface only."""
    if not source_text.strip():
        raise BlindTestAuthorError("source text is empty")
    if not interface.outputs:
        raise BlindTestAuthorError("the module declares no derived or parameter rule")
    if period_hint is not None and not _MONTH_RE.fullmatch(period_hint):
        raise BlindTestAuthorError("period hint must be YYYY-MM")
    if source_as_of is not None:
        _parse_as_of(source_as_of)
    if period_hint:
        period_rule = (
            f" Use {period_hint} unless the source text states dates that matter."
        )
    elif source_as_of:
        period_rule = (
            f" The source text was captured on {source_as_of}. Unless the text "
            "states dates that matter, use months after that date."
        )
    else:
        period_rule = ""
    return BlindPrompt(
        instructions=_INSTRUCTIONS.format(
            max_cases=max_cases,
            min_chars=MIN_SOURCE_SENTENCE_CHARS,
            period_rule=period_rule,
        ),
        citation_line=(
            f"Source citation path: {require_citation_path(citation_path)}\n"
            if citation_path
            else ""
        ),
        interface_block="=== INTERFACE ===\n"
        + json.dumps(interface.to_json(), indent=2, sort_keys=True)
        + "\n",
        source_text=source_text,
    )


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _identifier_present(token: str, text: str) -> bool:
    return (
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", text)
        is not None
    )


def _walk_strings(node: Any) -> Iterator[str]:
    if isinstance(node, Mapping):
        for value in node.values():
            yield from _walk_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_strings(value)
    elif isinstance(node, str):
        yield node


@dataclass
class _ModuleSecrets:
    substrings: set[str] = field(default_factory=set)
    numbers: set[str] = field(default_factory=set)
    dates: set[str] = field(default_factory=set)
    test_names: set[str] = field(default_factory=set)


def _module_secrets(
    rulespec_payload: Mapping[str, Any],
    companion_cases: Sequence[Any] | None,
) -> _ModuleSecrets:
    """Collect what a module must not show: text, numbers, dates, test names."""
    secrets = _ModuleSecrets()

    def add_prose(value: Any) -> None:
        for text in _walk_strings(value):
            collapsed = _collapse(text)
            if len(collapsed) >= 12 and not _NAME_RE.fullmatch(collapsed):
                secrets.substrings.add(collapsed)

    def add_formula(value: Any) -> None:
        text = str(value)
        collapsed = _collapse(text)
        if not collapsed or _NAME_RE.fullmatch(collapsed):
            return  # empty, or a bare reference to a name the interface lists
        # Digits inside an identifier (`line_32600`) are a name, not a value.
        secrets.numbers.update(
            re.findall(r"(?<![\w.])\d[\d_,]*(?:\.\d+)?(?!\w)", collapsed)
        )
        secrets.dates.update(m.group(0) for m in _ISO_DATE_RE.finditer(collapsed))
        if re.fullmatch(r"[-+]?[\d_,.]+%?", collapsed):
            return
        secrets.substrings.add(collapsed)
        for line in text.splitlines():
            fragment = _collapse(line)
            if len(fragment) >= 8 and not _NAME_RE.fullmatch(fragment):
                secrets.substrings.add(fragment)

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            for child_key, child in node.items():
                child_key = str(child_key)
                if child_key in {"formula", "value", "amount"} and not isinstance(
                    child, (Mapping, list)
                ):
                    add_formula(child)
                elif child_key in {"effective_from", "effective_to"}:
                    if child is not None:
                        secrets.dates.add(str(child))
                elif child_key in {
                    "summary",
                    "source",
                    "label",
                    "description",
                    "metadata",
                    "notes",
                    "source_url",
                }:
                    add_prose(child)
                else:
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            # A bare number anywhere in a declaration, e.g. `values: [1500]`.
            secrets.numbers.add(repr(node))

    visit(rulespec_payload)
    for case in companion_cases or ():
        if isinstance(case, Mapping):
            name = case.get("name")
            if isinstance(name, str) and name.strip():
                secrets.test_names.add(name.strip())
            add_prose({k: v for k, v in case.items() if k not in {"input", "output"}})
    return secrets


def find_blindness_violations(
    prompt: BlindPrompt,
    *,
    rulespec_payload: Mapping[str, Any],
    companion_cases: Sequence[Any] | None = None,
    interface: RuleInterface | None = None,
) -> list[str]:
    """Return module text that the non-source part of the prompt contains.

    Checked against instructions, citation line and interface: every formula
    (whole and per line) and the module's prose (summary, source labels,
    proof metadata, test prose). Checked against the module-derived parts:
    the name of every existing test case, unless it equals an interface name
    (a test may be named after the output it asserts; that shows nothing
    new). Each hit stops the run. The corpus source text is not checked: it
    does not come from the module.
    """
    collapsed_prompt = _collapse(prompt.controlled)
    secrets = _module_secrets(rulespec_payload, companion_cases)
    # The validated citation path names the unit. The module's proof metadata
    # repeats it; that repetition is not module prose.
    secrets.substrings.discard(
        prompt.citation_line.removeprefix("Source citation path: ").strip()
    )
    violations = [
        f"prompt contains module text: {secret[:80]!r}"
        for secret in sorted(secrets.substrings)
        if secret in collapsed_prompt
    ]
    interface_names = interface.names() if interface is not None else set()
    violations.extend(
        f"prompt contains the name of an existing test case: {name!r}"
        for name in sorted(secrets.test_names - interface_names)
        if _identifier_present(name, prompt.module_derived)
    )
    return violations


def _number_pattern(token: str) -> str | None:
    """A pattern that finds a formula number inside identifiers and paths.

    ``_`` and other separators do not hide the number: ``limit_1500_usd``
    and ``rate_0_3`` match ``1500`` and ``0.3``. A whole number of one digit
    is not searched: 0, 1 and 2 are in every formula and every path.
    """
    digits = token.replace("_", "").replace(",", "").strip(".")
    if not digits or not re.fullmatch(r"\d+(\.\d+)?", digits):
        return None
    whole, _, fraction = digits.partition(".")
    fraction = fraction.rstrip("0")
    if fraction:
        return rf"(?<!\d){int(whole)}[._]{fraction}(?!\d)"
    if len(str(int(whole))) < 2:
        return None
    return rf"(?<!\d)0*{int(whole)}(?!\d)"


def _date_pattern(token: str) -> str | None:
    match = _ISO_DATE_RE.fullmatch(token.strip())
    if match is None:
        return None
    year, month, day = match.groups()
    return rf"(?<!\d){year}[\W_]?{month}[\W_]?{day}(?!\d)"


def find_interface_leaks(
    prompt: BlindPrompt,
    *,
    rulespec_payload: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Return formula numbers and dates that names or paths carry.

    A name such as ``limit_is_1500_usd``, an enumeration member, the module
    path or the citation path can spell a value that the formulas use, or an
    effective date. This cannot be told from a harmless coincidence
    (``fy_2026_...``, ``block-13``), so it is neither passed nor failed in
    silence: each hit is returned with the text that carries it, and the
    caller decides (see ``allow_interface_leaks``).
    """
    secrets = _module_secrets(rulespec_payload, None)
    patterns = [
        ("formula_number", token, _number_pattern(token)) for token in secrets.numbers
    ] + [("effective_date", token, _date_pattern(token)) for token in secrets.dates]
    leaks: dict[tuple[str, str, str], dict[str, str]] = {}
    for where, text in (
        ("citation_path", prompt.citation_line),
        ("interface", prompt.interface_block),
    ):
        for kind, token, pattern in patterns:
            if pattern is None:
                continue
            for match in re.finditer(pattern, text):
                start, end = match.span()
                while start > 0 and re.match(r"[\w./:()-]", text[start - 1]):
                    start -= 1
                while end < len(text) and re.match(r"[\w./:()-]", text[end]):
                    end += 1
                carrier = text[start:end]
                leaks[(kind, token, carrier)] = {
                    "kind": kind,
                    "value": token,
                    "where": where,
                    "carrier": carrier,
                }
    return [leaks[key] for key in sorted(leaks)]


# ---------------------------------------------------------------------------
# Verbatim quotes
# ---------------------------------------------------------------------------

_ITEM_MARKER_RE = re.compile(
    r"(?:[●•◦▪‣*\-–—]"
    r"|\(?[0-9A-Za-z]{1,4}[.)]"
    r"|(?:NOTE|Note|EXAMPLE|Example|EXCEPTION|Exception)[:.]?)\s+"
)
_SENTENCE_END_RE = re.compile(r"[.?!;][\"'”’)\]]*(?=\s)")


@dataclass(frozen=True)
class QuoteMatch:
    found: bool
    at_boundaries: bool = False
    covers_full_item: bool = False


class SourceQuotes:
    """Where a quote may start and end in one source text.

    The source is read as items: each non-empty line is one item (a paragraph
    or a list item). A quote must be verbatim, whitespace runs aside, and
    must start and end at a boundary:

    - start: the start of an item, the text after an item's marker (a bullet,
      an enumerator such as ``(a)`` or ``1.``, a label such as ``NOTE``), or
      the text after a sentence end;
    - end: the end of an item, or a sentence end inside it (``.``, ``?``,
      ``!`` or ``;``, with closing quotes or brackets, before whitespace).

    A quote may span several items. A quote that starts or stops inside a
    sentence is refused: it can drop the condition that the sentence states.
    """

    def __init__(self, source_text: str) -> None:
        pieces: list[str] = []
        self.starts: set[int] = set()
        self.ends: set[int] = set()
        self.item_starts: set[int] = set()
        self.item_ends: set[int] = set()
        position = 0
        for line in source_text.splitlines():
            item = _collapse(line)
            if not item:
                continue
            if pieces:
                position += 1
            marker = _ITEM_MARKER_RE.match(item)
            for start in {position, position + (marker.end() if marker else 0)}:
                self.starts.add(start)
                self.item_starts.add(start)
            for sentence_end in _SENTENCE_END_RE.finditer(item):
                self.ends.add(position + sentence_end.end())
                self.starts.add(position + sentence_end.end() + 1)
            position += len(item)
            self.ends.add(position)
            self.item_ends.add(position)
            pieces.append(item)
        self.text = " ".join(pieces)

    def locate(self, quote: str) -> QuoteMatch:
        needle = _collapse(quote)
        if not needle:
            return QuoteMatch(False)
        best = QuoteMatch(False)
        index = self.text.find(needle)
        while index != -1:
            end = index + len(needle)
            match = QuoteMatch(
                True,
                index in self.starts and end in self.ends,
                index in self.item_starts and end in self.item_ends,
            )
            if (match.at_boundaries, match.covers_full_item) >= (
                best.at_boundaries,
                best.covers_full_item,
            ):
                best = match
            index = self.text.find(needle, index + 1)
        return best


def sentence_in_source(sentence: str, source_text: str) -> bool:
    """True when ``sentence`` occurs in the source, whitespace runs aside.

    Characters must match exactly, including case and punctuation. Only runs
    of whitespace are folded to one space, because a JSON string cannot carry
    the source's line breaks faithfully. Boundaries are a separate question:
    see :class:`SourceQuotes`.
    """
    return SourceQuotes(source_text).locate(sentence).found


def _quote_issue(quote: Any, source: SourceQuotes) -> tuple[str, QuoteMatch]:
    if not isinstance(quote, str):
        return "source_sentence must be a string", QuoteMatch(False)
    if len(_collapse(quote)) < MIN_SOURCE_SENTENCE_CHARS:
        return (
            f"source_sentence is shorter than {MIN_SOURCE_SENTENCE_CHARS} characters",
            QuoteMatch(False),
        )
    match = source.locate(quote)
    if not match.found:
        return "source_sentence is not found verbatim in the source text", match
    if not match.at_boundaries:
        return (
            "source_sentence does not start and end at a sentence or list-item "
            "boundary of the source text",
            match,
        )
    return "", match


# ---------------------------------------------------------------------------
# Strict response parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlindCase:
    name: str
    period: str
    inputs: dict[str, Any]
    expected: dict[str, Any]
    source_sentence: str
    quote_covers_full_item: bool = True

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RejectedCase:
    index: int
    name: str | None
    reason: str
    source_sentence: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParsedResponse:
    cases: tuple[BlindCase, ...]
    rejected: tuple[RejectedCase, ...]
    stated_commencement_month: str | None = None
    stated_commencement_sentence: str | None = None
    response_error: str | None = None
    notes: tuple[str, ...] = ()


def _value_issue(value: Any, dtype: str | None, enumeration: Sequence[str]) -> str:
    if isinstance(value, (dict, list)) or value is None:
        return "value must be a JSON scalar"
    if isinstance(value, float) and not math.isfinite(value):
        return "number must be finite"
    kind = (dtype or "").lower()
    if kind in _BOOLEAN_DTYPES:
        return "" if isinstance(value, bool) else "Boolean value must be true or false"
    if kind == "judgment":
        return (
            ""
            if isinstance(value, str) and value in _JUDGMENT_VALUES
            else "Judgment value must be holds, not_holds or undetermined"
        )
    if kind in _NUMBER_DTYPES:
        return (
            ""
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else f"{dtype} value must be a JSON number"
        )
    if kind in _TEXT_DTYPES or enumeration:
        if not isinstance(value, str):
            return f"{dtype or 'enumeration'} value must be a string"
        if enumeration and value not in enumeration:
            return "value is not in the declared enumeration"
    return ""


def _validate_case(
    index: int, raw: Any, interface: RuleInterface, source: SourceQuotes
) -> BlindCase | RejectedCase:
    if not isinstance(raw, dict):
        return RejectedCase(index, None, "case is not a JSON object")
    name = raw.get("name") if isinstance(raw.get("name"), str) else None
    sentence = (
        raw.get("source_sentence")
        if isinstance(raw.get("source_sentence"), str)
        else None
    )

    def reject(reason: str) -> RejectedCase:
        return RejectedCase(index, name, reason, sentence)

    if set(raw) != _CASE_KEYS:
        extra = sorted(set(raw) - _CASE_KEYS)
        missing = sorted(_CASE_KEYS - set(raw))
        return reject(
            f"case keys must be exactly {sorted(_CASE_KEYS)}"
            f" (missing {missing}, unexpected {extra})"
        )
    if name is None or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,119}", name):
        return reject("name must be a short identifier")
    period = raw["period"]
    if not isinstance(period, str) or not _MONTH_RE.fullmatch(period):
        return reject("period must be a YYYY-MM string")
    inputs, expected = raw["inputs"], raw["expected"]
    if not isinstance(inputs, dict):
        return reject("inputs must be a JSON object")
    if not isinstance(expected, dict) or not expected:
        return reject("expected must be a non-empty JSON object")
    for key, value in inputs.items():
        declared = interface.input(key)
        if declared is None:
            return reject(f"input `{key}` is not in the interface")
        issue = _value_issue(value, declared.dtype, declared.enumeration)
        if issue:
            return reject(f"input `{key}`: {issue}")
    for key, value in expected.items():
        declared_output = interface.output(key)
        if declared_output is None:
            return reject(f"output `{key}` is not in the interface")
        issue = _value_issue(value, declared_output.dtype, declared_output.enumeration)
        if issue:
            return reject(f"output `{key}`: {issue}")
    issue, match = _quote_issue(raw["source_sentence"], source)
    if issue:
        return reject(issue)
    return BlindCase(
        name=name,
        period=period,
        inputs=dict(inputs),
        expected=dict(expected),
        source_sentence=raw["source_sentence"],
        quote_covers_full_item=match.covers_full_item,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate JSON key(s): {', '.join(duplicates)}")
    return dict(pairs)


def _reject_constant(constant: str) -> Any:
    raise ValueError(f"{constant} is not a JSON number")


def parse_blind_response(
    response: str,
    *,
    interface: RuleInterface,
    source_text: str,
    max_cases: int = DEFAULT_MAX_CASES,
) -> ParsedResponse:
    """Parse the typed answer. Free prose is never interpreted.

    The whole answer must be one JSON object (one enclosing code fence is
    tolerated, since it carries no content). Anything else is a response
    error with zero cases. Each case is validated on its own; a case is
    rejected and reported when its quote is not verbatim in the source text
    or is cut inside a sentence, when it repeats an earlier case under
    another name, or when a number is not finite.
    """
    text = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*)\n```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ValueError as exc:
        return ParsedResponse(
            (), (), response_error=f"answer is not one JSON object: {exc}"
        )
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        return ParsedResponse(
            (), (), response_error='answer must be a JSON object with a "cases" list'
        )
    unexpected = sorted(set(payload) - _RESPONSE_KEYS)
    if unexpected:
        return ParsedResponse(
            (), (), response_error=f"unexpected top-level key(s): {unexpected}"
        )
    source = SourceQuotes(source_text)
    cases: list[BlindCase] = []
    rejected: list[RejectedCase] = []
    names: set[str] = set()
    contents: dict[str, str] = {}
    for index, raw in enumerate(payload["cases"], 1):
        if index > max_cases:
            rejected.append(
                RejectedCase(index, None, f"more than {max_cases} cases; not run")
            )
            continue
        result = _validate_case(index, raw, interface, source)
        if isinstance(result, BlindCase):
            content = json.dumps(
                [result.period, result.inputs, result.expected], sort_keys=True
            )
            duplicate = (
                "duplicate case name"
                if result.name in names
                else f"same inputs, period and expected outputs as `{contents[content]}`"
                if content in contents
                else None
            )
            if duplicate:
                result = RejectedCase(
                    index, result.name, duplicate, result.source_sentence
                )
            else:
                names.add(result.name)
                contents[content] = result.name
        if isinstance(result, BlindCase):
            cases.append(result)
        else:
            rejected.append(result)

    notes: list[str] = []
    month = sentence = None
    stated = payload.get("stated_commencement")
    if stated is not None:
        issue = ""
        if not isinstance(stated, dict) or set(stated) != _COMMENCEMENT_KEYS:
            issue = f"keys must be exactly {sorted(_COMMENCEMENT_KEYS)}"
        elif not isinstance(stated["month"], str) or not _MONTH_RE.fullmatch(
            stated["month"]
        ):
            issue = "month must be YYYY-MM"
        else:
            issue, _ = _quote_issue(stated["source_sentence"], source)
            if not issue and stated["month"][:4] not in stated["source_sentence"]:
                issue = "the quoted sentence does not contain the year of the month"
        if issue:
            notes.append(f"stated_commencement ignored: {issue}")
        else:
            month, sentence = stated["month"], stated["source_sentence"]
    return ParsedResponse(
        cases=tuple(cases),
        rejected=tuple(rejected),
        stated_commencement_month=month,
        stated_commencement_sentence=sentence,
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Model backends
# ---------------------------------------------------------------------------


class BlindAuthorBackend(Protocol):
    """A second model: one prompt in, one text answer out."""

    name: str
    model: str

    def complete(self, prompt: str) -> str: ...


@dataclass
class FakeBlindAuthorBackend:
    """Deterministic backend for tests. It records every prompt it is given."""

    response: str | Callable[[str], str]
    name: str = "fake"
    model: str = "fake-blind-author"
    prompts: list[str] = field(default_factory=list)

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response(prompt) if callable(self.response) else self.response


@dataclass
class ClaudeCliBlindAuthorBackend:
    """The ``claude`` CLI on the caller's claude.ai subscription login.

    It reuses the local reviewer's launch rules from
    ``harness/validator_pipeline.py``: the same executable resolution, the
    same explicit-allowlist environment (no ambient API key or token is
    forwarded) and the same tool-less, session-less ``claude --print`` flags.
    Three things differ from the reviewer, on purpose:

    - no fallback to the Codex CLI: the blind author must be a model of
      another family than the generator;
    - the child runs in a fresh, empty temporary directory, so the CLI's own
      context (working directory, git status) cannot describe the candidate;
    - only stdout is the answer. The shared launcher joins stderr to stdout,
      which would turn one CLI warning into an unparseable answer, so this
      backend runs the child itself and keeps stderr in ``last_stderr``.
    """

    model: str = REVIEWER_CLI_MODEL
    timeout: int = DEFAULT_TIMEOUT_SECONDS
    name: str = "claude-cli"
    last_stderr: str = ""

    def complete(self, prompt: str) -> str:
        from axiom_encode.harness import validator_pipeline as vp

        if vp._protected_reviewer_context_active():
            raise BlindAuthorBackendError(
                "the blind test author is a local shadow tool; it does not run "
                "in CI, under the trusted runtime or with a signing broker"
            )
        executable = vp._resolve_local_reviewer_executable("claude")
        if executable is None:
            raise BlindAuthorBackendError("`claude` CLI not found on PATH")
        environment = vp._local_reviewer_environment([executable])
        # Subscription authentication only, never the API credit key.
        environment.pop("ANTHROPIC_API_KEY", None)
        cmd = [
            executable,
            "--print",
            "--permission-mode",
            "dontAsk",
            "--safe-mode",
            "--no-session-persistence",
            "--disable-slash-commands",
            "--no-chrome",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers": {}}',
            "--tools",
            "",
            "--allowed-tools",
            "",
            "--model",
            self.model,
            "-p",
            prompt,
        ]
        try:
            with tempfile.TemporaryDirectory(prefix="blind-author-cwd-") as empty:
                completed = subprocess.run(
                    cmd,
                    cwd=empty,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
        except (OSError, subprocess.SubprocessError) as exc:
            raise BlindAuthorBackendError(f"claude CLI failed: {exc}") from exc
        self.last_stderr = completed.stderr or ""
        if completed.returncode != 0:
            raise BlindAuthorBackendError(
                f"claude CLI exited {completed.returncode}: "
                f"{(completed.stderr or completed.stdout or '')[-400:]}"
            )
        return completed.stdout or ""


# ---------------------------------------------------------------------------
# Running cases with the companion-test runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseOutcome:
    status: str
    issues: tuple[str, ...] = ()
    comparisons: tuple[dict[str, Any], ...] = ()
    not_runnable_kind: str | None = None
    other_issues: tuple[str, ...] = ()


class CaseRunner(Protocol):
    def run(self, case: BlindCase) -> CaseOutcome: ...


def companion_case(case: BlindCase, module: str) -> dict[str, Any]:
    """Render one blind case in the companion ``.test.yaml`` case shape."""
    return {
        "name": case.name,
        "period": case.period,
        "input": {f"{module}#input.{key}": value for key, value in case.inputs.items()},
        "output": {f"{module}#{key}": value for key, value in case.expected.items()},
    }


def classify_outcome(
    issues: Sequence[str], comparisons: Sequence[Mapping[str, Any]]
) -> CaseOutcome:
    """Map runner issues to agree / disagree / not-runnable.

    A case disagrees when an output was compared and differs. Other issues
    of the same case are reported beside the mismatch (``other_issues``);
    they do not hide it. A case with issues and no mismatch did not run
    (binding, a missing input, no formula version at the period, an engine
    failure): that is never a disagreement.
    """
    mismatches = [str(item["mismatch"]) for item in comparisons if item.get("mismatch")]
    records = tuple(dict(item) for item in comparisons)
    if mismatches:
        other = tuple(
            issue
            for issue in issues
            if not any(issue == m or issue.startswith(m) for m in mismatches)
        )
        return CaseOutcome(
            STATUS_DISAGREE, tuple(mismatches), records, other_issues=other
        )
    if not issues:
        if not comparisons:
            return CaseOutcome(
                STATUS_NOT_RUNNABLE,
                ("no output was compared",),
                records,
                "execution_or_binding",
            )
        return CaseOutcome(STATUS_AGREE, (), records)
    kind = (
        "no_formula_version_at_period"
        if any(_NO_FORMULA_VERSION_RE.search(issue) for issue in issues)
        else "execution_or_binding"
    )
    return CaseOutcome(STATUS_NOT_RUNNABLE, tuple(issues), records, kind)


def module_target_for(rules_file: Path, policy_repo_path: Path) -> str:
    """Return ``<jurisdiction>:<path without .yaml>`` for a module file."""
    relative = rules_file.resolve().relative_to(policy_repo_path.resolve())
    return f"{policy_repo_path.resolve().name}:{relative.with_suffix('').as_posix()}"


class EngineCaseRunner:
    """Compile once, then run each blind case as a one-case companion test.

    This is the runner the validator uses for ``.test.yaml`` files
    (``ValidatorPipeline._run_rulespec_test_cases``) on the real rules
    engine. Output comparisons are recorded so a mismatch can be told apart
    from a case that failed to run.
    """

    def __init__(
        self,
        *,
        rules_file: Path,
        policy_repo_path: Path,
        axiom_rules_path: Path,
        work_dir: Path,
        axiom_rules_engine_ref: str | None = None,
    ) -> None:
        from axiom_encode.harness.validator_pipeline import ValidatorPipeline

        comparisons: list[dict[str, Any]] = []

        class _RecordingPipeline(ValidatorPipeline):
            def _compare_rulespec_output(self, **kwargs: Any) -> str | None:
                mismatch = super()._compare_rulespec_output(**kwargs)
                comparisons.append(
                    {
                        "output": kwargs.get("output_name"),
                        "expected": kwargs.get("expected_value"),
                        "actual": kwargs.get("actual_output"),
                        "mismatch": mismatch,
                    }
                )
                return mismatch

        self._comparisons = comparisons
        self.rules_file = Path(rules_file).resolve()
        self.policy_repo_path = Path(policy_repo_path).resolve()
        self.module = module_target_for(self.rules_file, self.policy_repo_path)
        self.compiled_path = Path(work_dir).resolve() / "compiled.json"
        self.compiled_payload: dict[str, Any] | None = None
        self.compile_error: str | None = None
        try:
            self._pipeline = _RecordingPipeline(
                policy_repo_path=self.policy_repo_path,
                axiom_rules_path=Path(axiom_rules_path),
                local_corpus_release=None,
                enable_oracles=False,
                axiom_rules_engine_ref=axiom_rules_engine_ref,
            )
            result, payload = self._pipeline._compile_rulespec_to_artifact(
                self.rules_file, self.compiled_path
            )
        except Exception as exc:
            self.compile_error = f"compile crashed: {exc}"
        else:
            if result.returncode != 0 or not isinstance(payload, dict):
                detail = (result.stderr or result.stdout or "").strip()
                self.compile_error = f"compile failed: {detail[:1000]}"
            else:
                self.compiled_payload = payload

    def run(self, case: BlindCase) -> CaseOutcome:
        if self.compiled_payload is None:
            return CaseOutcome(
                STATUS_NOT_RUNNABLE,
                (self.compile_error or "module did not compile",),
                (),
                "execution_or_binding",
            )
        self._comparisons.clear()
        try:
            issues = self._pipeline._run_rulespec_test_cases(
                rules_file=self.rules_file,
                compiled_path=self.compiled_path,
                compiled_payload=self.compiled_payload,
                cases=[companion_case(case, self.module)],
            )
        except Exception as exc:
            return CaseOutcome(
                STATUS_NOT_RUNNABLE,
                (f"runner crashed: {exc}",),
                (),
                "execution_or_binding",
            )
        return classify_outcome(issues, list(self._comparisons))


# ---------------------------------------------------------------------------
# Period probes
# ---------------------------------------------------------------------------


def _parse_as_of(source_as_of: str) -> date:
    try:
        if not _ISO_DATE_RE.fullmatch(source_as_of):
            raise ValueError(source_as_of)
        return date.fromisoformat(source_as_of)
    except (TypeError, ValueError) as exc:
        raise BlindTestAuthorError("source as-of date must be YYYY-MM-DD") from exc


def assumed_evidence_floor(source_as_of: str) -> str:
    """First full calendar month that starts on or after the as-of date.

    Computed here, never taken from the model. It is one candidate
    convention for a unit whose text states no date ("not earlier than the
    source snapshot"). It is NOT a decided rule: see ``ASSUMPTION_REFERENCE``.
    """
    as_of = _parse_as_of(source_as_of)
    if as_of.day == 1:
        return as_of.strftime("%Y-%m")
    return (as_of.replace(day=1) + timedelta(days=32)).strftime("%Y-%m")


def _previous_month(month: str) -> str:
    first = date.fromisoformat(f"{month}-01")
    return (first - timedelta(days=1)).strftime("%Y-%m")


def _computes_before(
    month: str,
    results: Sequence[tuple[BlindCase, CaseOutcome]],
    runner: CaseRunner,
) -> dict[str, Any]:
    """Run one case that ran again, one month before ``month``."""
    base = next(
        (
            case
            for case, outcome in results
            if outcome.status in (STATUS_AGREE, STATUS_DISAGREE)
        ),
        None,
    )
    if base is None:
        return {"computes": None, "note": "no blind case ran; nothing to re-run"}
    probe_period = _previous_month(month)
    outcome = runner.run(
        BlindCase(
            name=f"{base.name}__period_probe",
            period=probe_period,
            inputs=base.inputs,
            expected=base.expected,
            source_sentence=base.source_sentence,
        )
    )
    result: dict[str, Any] = {"probe_period": probe_period, "probe_case": base.name}
    if outcome.status in (STATUS_AGREE, STATUS_DISAGREE):
        result["computes"] = True
    elif outcome.not_runnable_kind == "no_formula_version_at_period":
        result["computes"] = False
    else:
        result["computes"] = None
        result["note"] = "; ".join(outcome.issues) or "the probe case did not run"
    return result


def probe_periods(
    parsed: ParsedResponse,
    results: Sequence[tuple[BlindCase, CaseOutcome]],
    runner: CaseRunner,
    *,
    source_as_of: str | None,
) -> dict[str, Any]:
    """Does the candidate compute before a date the source supports?

    An output case cannot see a rule that starts too early, so one case that
    ran is run again one month before each of two reference months:

    ``assumed_evidence_floor``
        The first full month after the corpus as-of date, computed in code.
        A hit (``before_assumed_evidence_floor``) is NOT a finding of a
        defect. The start date of a unit whose text states no date is an
        open maintainer decision (``ASSUMPTION_REFERENCE``), and by this
        convention several hundred modules on rulespec-us main "start
        early". The report marks it ``"assumption": true``.

    ``source_stated_date``
        A commencement month the second model read in the source text, with
        the verbatim sentence. A hit (``before_source_stated_date``) rests
        on the model's reading of that sentence.
    """
    floor: dict[str, Any] = {
        "finding": FINDING_NOT_PROBED,
        "assumption": True,
        "reference": ASSUMPTION_REFERENCE,
        "meaning": (
            "Start before the ASSUMED evidence floor (first full month after "
            "the source as-of date). Whether that is a defect depends on the "
            f"decision in {ASSUMPTION_REFERENCE}."
        ),
        "source_as_of": source_as_of,
    }
    if source_as_of:
        floor["floor_month"] = assumed_evidence_floor(source_as_of)
        floor.update(_computes_before(floor["floor_month"], results, runner))
        if floor["computes"] is not None:
            floor["finding"] = (
                FINDING_BEFORE_ASSUMED_FLOOR
                if floor["computes"]
                else FINDING_NOT_BEFORE_ASSUMED_FLOOR
            )
    else:
        floor["note"] = "no source as-of date"
    stated: dict[str, Any] = {
        "finding": FINDING_NOT_PROBED,
        "assumption": False,
        "meaning": (
            "Start before a commencement date that the second model read in "
            "the quoted source sentence. It rests on that reading."
        ),
        "stated_month": parsed.stated_commencement_month,
        "source_sentence": parsed.stated_commencement_sentence,
    }
    if parsed.stated_commencement_month:
        stated.update(
            _computes_before(parsed.stated_commencement_month, results, runner)
        )
        if stated["computes"] is not None:
            stated["finding"] = (
                FINDING_BEFORE_STATED_DATE
                if stated["computes"]
                else FINDING_NOT_BEFORE_STATED_DATE
            )
    else:
        stated["note"] = "the second model found no stated commencement date"
    return {"assumed_evidence_floor": floor, "source_stated_date": stated}


# ---------------------------------------------------------------------------
# Orchestration and report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedPrompt:
    interface: RuleInterface
    prompt: BlindPrompt
    interface_leaks: tuple[dict[str, str], ...]


def prepare_blind_prompt(
    *,
    rulespec_payload: Mapping[str, Any],
    module: str,
    source_text: str,
    compiled_payload: Mapping[str, Any] | None = None,
    companion_cases: Sequence[Any] | None = None,
    citation_path: str | None = None,
    source_as_of: str | None = None,
    period_hint: str | None = None,
    max_cases: int = DEFAULT_MAX_CASES,
    allow_interface_leaks: bool = False,
) -> PreparedPrompt:
    """Build the prompt and prove it blind, or raise before any model call.

    Module text in the prompt always raises :class:`BlindnessViolation`. A
    name or path that carries a formula number or an effective date raises
    :class:`InterfaceLeak` unless ``allow_interface_leaks`` is set; then the
    leaks are returned for the report.
    """
    interface = extract_rule_interface(
        rulespec_payload, module=module, compiled_payload=compiled_payload
    )
    prompt = build_blind_prompt(
        source_text=source_text,
        interface=interface,
        citation_path=citation_path,
        source_as_of=source_as_of,
        period_hint=period_hint,
        max_cases=max_cases,
    )
    violations = find_blindness_violations(
        prompt,
        rulespec_payload=rulespec_payload,
        companion_cases=companion_cases,
        interface=interface,
    )
    if violations:
        raise BlindnessViolation("; ".join(violations))
    leaks = tuple(find_interface_leaks(prompt, rulespec_payload=rulespec_payload))
    if leaks and not allow_interface_leaks:
        listed = "; ".join(
            f"{leak['kind']} {leak['value']} in {leak['where']} `{leak['carrier']}`"
            for leak in leaks
        )
        raise InterfaceLeak(
            "a name or path in the prompt carries a formula value or a date: "
            f"{listed}. If this is a coincidence or an accepted leak, pass "
            "allow_interface_leaks (--allow-interface-leaks); the report "
            "records it."
        )
    return PreparedPrompt(interface, prompt, leaks)


def run_blind_test_author(
    *,
    rulespec_payload: Mapping[str, Any],
    module: str,
    source_text: str,
    backend: BlindAuthorBackend,
    runner: CaseRunner,
    compiled_payload: Mapping[str, Any] | None = None,
    companion_cases: Sequence[Any] | None = None,
    citation_path: str | None = None,
    source_as_of: str | None = None,
    period_hint: str | None = None,
    max_cases: int = DEFAULT_MAX_CASES,
    authored_response: str | None = None,
    allow_interface_leaks: bool = False,
) -> dict[str, Any]:
    """Author blind cases, run them, and return the machine-readable report.

    ``authored_response`` replays a stored answer instead of calling the
    backend, so the same blind cases can run against a second candidate.
    Raises before any model call when the candidate did not compile (the
    interface of a module that does not compile is not known) or when the
    prompt is not blind (see :func:`prepare_blind_prompt`).
    """
    compile_error = getattr(runner, "compile_error", None)
    if compile_error:
        raise BlindTestAuthorError(
            f"the candidate does not compile; no model call was made: {compile_error}"
        )
    prepared = prepare_blind_prompt(
        rulespec_payload=rulespec_payload,
        module=module,
        source_text=source_text,
        compiled_payload=compiled_payload,
        companion_cases=companion_cases,
        citation_path=citation_path,
        source_as_of=source_as_of,
        period_hint=period_hint,
        max_cases=max_cases,
        allow_interface_leaks=allow_interface_leaks,
    )
    interface, prompt = prepared.interface, prepared.prompt
    replayed = authored_response is not None
    response = authored_response if replayed else backend.complete(prompt.text)
    parsed = parse_blind_response(
        response, interface=interface, source_text=source_text, max_cases=max_cases
    )
    results = [(case, runner.run(case)) for case in parsed.cases]
    counts = {
        status: sum(1 for _, outcome in results if outcome.status == status)
        for status in (STATUS_AGREE, STATUS_DISAGREE, STATUS_NOT_RUNNABLE)
    }
    counts["rejected"] = len(parsed.rejected)
    counts["authored"] = len(parsed.cases) + len(parsed.rejected)
    partial = sum(1 for case in parsed.cases if not case.quote_covers_full_item)
    return {
        "schema": REPORT_SCHEMA,
        "shadow": True,
        "gates": "nothing",
        "module": module,
        "citation_path": citation_path,
        "source_text_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "source_as_of": source_as_of,
        "backend": {
            "name": backend.name,
            "model": backend.model,
            "replayed_response": replayed,
            "stderr": getattr(backend, "last_stderr", "")[-2000:],
        },
        "prompt_sha256": prompt.sha256,
        "interface": interface.to_json(),
        "interface_dropped_fields": list(interface.dropped),
        "interface_leaks": [dict(leak) for leak in prepared.interface_leaks],
        "interface_leaks_allowed": bool(prepared.interface_leaks),
        "response_error": parsed.response_error,
        "notes": list(parsed.notes),
        "counts": counts,
        "quote_stats": {
            "accepted_cases": len(parsed.cases),
            "quote_shorter_than_its_source_item": partial,
            "share": round(partial / len(parsed.cases), 4) if parsed.cases else None,
        },
        "cases": [
            {
                **case.to_json(),
                "status": outcome.status,
                "not_runnable_kind": outcome.not_runnable_kind,
                "issues": list(outcome.issues),
                "other_issues": list(outcome.other_issues),
                "comparisons": list(outcome.comparisons),
            }
            for case, outcome in results
        ],
        "rejected_cases": [item.to_json() for item in parsed.rejected],
        "period_probes": probe_periods(
            parsed, results, runner, source_as_of=source_as_of
        ),
        "raw_response": response,
    }


def render_text_summary(report: Mapping[str, Any]) -> str:
    """A short human summary of one report."""
    counts = report["counts"]
    lines = [
        f"Blind test author (shadow, gates nothing): {report['module']}",
        f"  second model: {report['backend']['name']} / {report['backend']['model']}",
        "  cases: "
        f"{counts[STATUS_AGREE]} agree, {counts[STATUS_DISAGREE]} disagree, "
        f"{counts[STATUS_NOT_RUNNABLE]} not runnable, {counts['rejected']} rejected",
    ]
    if report.get("response_error"):
        lines.append(f"  RESPONSE ERROR: {report['response_error']}")
    for leak in report.get("interface_leaks", []):
        lines.append(
            f"  [interface leak, allowed by the caller] {leak['kind']} "
            f"{leak['value']} in {leak['where']} `{leak['carrier']}`"
        )
    for case in report["cases"]:
        if case["status"] == STATUS_AGREE:
            continue
        lines.append(f"  [{case['status']}] {case['name']} ({case['period']})")
        lines.append(f"      source: {_collapse(case['source_sentence'])!r}")
        for issue in case["issues"][:3]:
            lines.append(f"      {issue}")
        for issue in case.get("other_issues", [])[:3]:
            lines.append(f"      also: {issue}")
    for item in report["rejected_cases"]:
        lines.append(f"  [rejected] #{item['index']} {item['name']}: {item['reason']}")
    quotes = report.get("quote_stats") or {}
    if quotes.get("quote_shorter_than_its_source_item"):
        lines.append(
            f"  quotes: {quotes['quote_shorter_than_its_source_item']} of "
            f"{quotes['accepted_cases']} accepted cases quote less than the full "
            "source item they sit in"
        )
    probes = report.get("period_probes") or {}
    floor = probes.get("assumed_evidence_floor") or {}
    if floor.get("finding") == FINDING_BEFORE_ASSUMED_FLOOR:
        lines.append(
            f"  [period, ASSUMPTION, not a defect finding; see {floor['reference']}] "
            f"the candidate computes in {floor['probe_period']}, before the assumed "
            f"evidence floor {floor['floor_month']} (first full month after the "
            f"source as-of date {floor['source_as_of']}). The start-date rule for "
            "a unit whose text states no date is not decided."
        )
    stated = probes.get("source_stated_date") or {}
    if stated.get("finding") == FINDING_BEFORE_STATED_DATE:
        lines.append(
            f"  [period, second model's reading] the candidate computes in "
            f"{stated['probe_period']}, before {stated['stated_month']}, the "
            f"commencement the second model read in: "
            f"{_collapse(stated['source_sentence'])!r}"
        )
    for note in report.get("notes", []):
        lines.append(f"  note: {note}")
    return "\n".join(lines) + "\n"


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BlindTestAuthorError(f"{path} is not a YAML mapping")
    return payload


def real_defect_cases(
    corpus_dir: Path, *, jurisdiction: str = "us", judgeable_only: bool = True
) -> list[dict[str, str]]:
    """List the stored real defect pairs with what a replay needs.

    ``index.json`` has no parent commit; it is in each ``cases/<id>/case.json``.
    """
    corpus_dir = Path(corpus_dir)
    index = json.loads((corpus_dir / "index.json").read_text(encoding="utf-8"))
    rows = []
    for entry in index["cases"]:
        if entry["jurisdiction"] != jurisdiction:
            continue
        if judgeable_only and not entry["judgeable_from_provision"]:
            continue
        case_dir = corpus_dir / "cases" / entry["id"]
        case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "id": case["id"],
                "fix_commit": case["commit"],
                "parent_commit": case["parent_commit"],
                "module_path": case["module_path"],
                "provision": str(case_dir / "provision.txt"),
            }
        )
    return rows
