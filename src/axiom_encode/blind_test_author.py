"""Independent (blind) test author: a shadow tool that gates nothing.

The model that encodes a rule also writes its companion ``.test.yaml`` from
its own formula, so those tests show that the rule agrees with itself. This
module asks a *second* model for test cases written from the source text and
the rule's declared interface only, runs them against the candidate with the
companion-test runner, and reports agree / disagree / not-runnable per case.

Blindness is structural: :func:`extract_rule_interface` copies an allow-list
of declaration fields (names, entities, dtypes, periods, units) and nothing
else, and :func:`find_blindness_violations` re-checks the rendered prompt
against the module's formulas, values, dates, prose and existing test cases
before any model sees it.

Nothing here is a validation check. No check id is declared, the encode loop
does not call it, and a disagreement changes no validation result. See
``docs/blind-test-author.md``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Protocol

import yaml

from axiom_encode.constants import REVIEWER_CLI_MODEL

REPORT_SCHEMA = "axiom-encode/blind-test-author-report/v1"
CASES_SCHEMA = "axiom-encode/blind-test-author-cases/v1"

STATUS_AGREE = "agree"
STATUS_DISAGREE = "disagree"
STATUS_NOT_RUNNABLE = "not_runnable"

# A quoted sentence shorter than this proves nothing: "the" is verbatim in
# every source.
MIN_SOURCE_SENTENCE_CHARS = 20
DEFAULT_MAX_CASES = 8
DEFAULT_TIMEOUT_SECONDS = 600
SOURCE_AS_OF_BASIS = "source_as_of"

_JUDGMENT_VALUES = ("holds", "not_holds", "undetermined")
_BOOLEAN_DTYPES = {"boolean", "bool"}
_NUMBER_DTYPES = {"money", "integer", "count", "rate", "decimal", "int", "float"}
_TEXT_DTYPES = {"string", "text", "str"}
_MONTH_RE = re.compile(r"\d{4}-(0[1-9]|1[0-2])")
_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NO_FORMULA_VERSION_RE = re.compile(r"has no formula version at")
_ENUMERATION_KEYS = ("values", "allowed_values", "enum")
_CASE_KEYS = {"name", "period", "inputs", "expected", "source_sentence"}
_RESPONSE_KEYS = {"cases", "earliest_supported_period", "earliest_period_basis"}


class BlindTestAuthorError(RuntimeError):
    """The tool could not do its job; never a statement about the rule."""


class BlindnessViolation(BlindTestAuthorError):
    """The prompt would show the second model something it must not see."""


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
    """What the second model may know about the candidate. Nothing else."""

    module: str
    inputs: tuple[InterfaceInput, ...]
    outputs: tuple[InterfaceOutput, ...]
    relations: tuple[str, ...] = ()

    def input(self, name: str) -> InterfaceInput | None:
        return next((item for item in self.inputs if item.name == name), None)

    def output(self, name: str) -> InterfaceOutput | None:
        return next((item for item in self.outputs if item.name == name), None)

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


def _type_word(value: Any) -> str | None:
    """Return a declaration word (dtype, entity, period, unit), else None.

    Declaration words are single identifiers. Anything else (prose, a number,
    a formula) is dropped rather than copied.
    """
    return _name(value)


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
                _type_word(item.get("entity")),
                _type_word(item.get("period")),
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
    ``period``, ``unit`` and a list of enumeration names. It never reads
    ``versions``, ``formula``, ``effective_from``, ``metadata``, ``source``,
    ``summary``, labels, descriptions or comments. ``compiled_payload``
    supplies only the names of implicit (undeclared) inputs.
    """
    if not isinstance(rulespec_payload, Mapping):
        raise BlindTestAuthorError("RuleSpec module must be a YAML mapping")
    inputs: dict[str, InterfaceInput] = {}
    for item in rulespec_payload.get("inputs") or []:
        if not isinstance(item, Mapping):
            continue
        name = _name(item.get("name"))
        if name is None:
            continue
        inputs[name] = InterfaceInput(
            name=name,
            dtype=_type_word(item.get("dtype")),
            entity=_type_word(item.get("entity")),
            period=_type_word(item.get("period")),
            unit=_type_word(item.get("unit")),
            enumeration=_enumeration(item),
        )
    outputs: list[InterfaceOutput] = []
    relations: list[str] = []
    for item in rulespec_payload.get("rules") or []:
        if not isinstance(item, Mapping):
            continue
        name = _name(item.get("name"))
        kind = _type_word(item.get("kind"))
        if name is None or kind is None:
            continue
        if kind.endswith("relation"):
            relations.append(name)
            continue
        if kind not in {"derived", "parameter"}:
            continue
        outputs.append(
            InterfaceOutput(
                name=name,
                kind=kind,
                dtype=_type_word(item.get("dtype")),
                entity=_type_word(item.get("entity")),
                period=_type_word(item.get("period")),
                unit=_type_word(item.get("unit")),
                # A parameter's values are exactly what must stay hidden.
                enumeration=_enumeration(item) if kind == "derived" else (),
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
      "source_sentence": "<one sentence or clause copied exactly from the source text>"
    }}
  ],
  "earliest_supported_period": "<YYYY-MM or null>",
  "earliest_period_basis": "<exact source sentence, or \\"source_as_of\\", or null>"
}}

Format rules:
- "inputs" keys and "expected" keys are names from the INTERFACE, exactly.
- Assign every input the expected output depends on, including false facts.
- Boolean values are true or false. Judgment outputs are "holds",
  "not_holds" or "undetermined". Money, Integer, Count, Rate and Decimal
  values are JSON numbers. A rate is a fraction of one, not a percentage.
- "period" is one calendar month, "YYYY-MM".{period_rule}
- "source_sentence" must appear in the source text character for character
  and be at least {min_chars} characters long. A case with a sentence that
  is not in the source text is rejected.
- "earliest_supported_period" is the first month for which the source text
  supports applying this provision. Use a commencement date the text
  states.{as_of_rule} Give the sentence that states the date in
  "earliest_period_basis". If you cannot tell, use null for both.
"""


@dataclass(frozen=True)
class BlindPrompt:
    """The prompt in parts: what we wrote, and the corpus source text.

    ``instructions`` is static text. ``interface_block`` is the only part
    derived from the module. ``source_text`` comes from the corpus.
    """

    instructions: str
    interface_block: str
    source_text: str

    @property
    def controlled(self) -> str:
        return f"{self.instructions}\n{self.interface_block}"

    @property
    def text(self) -> str:
        return (
            f"{self.controlled}\n=== SOURCE TEXT (begin) ===\n{self.source_text}\n"
            "=== SOURCE TEXT (end) ===\n"
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


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
    period_rule = (
        f" Use {period_hint} unless the source text states dates that matter."
        if period_hint
        else ""
    )
    as_of_rule = (
        " If the text states none, use the first full calendar month after "
        f'the source as-of date and give "{SOURCE_AS_OF_BASIS}" as the basis.'
        if source_as_of
        else ""
    )
    instructions = _INSTRUCTIONS.format(
        max_cases=max_cases,
        min_chars=MIN_SOURCE_SENTENCE_CHARS,
        period_rule=period_rule,
        as_of_rule=as_of_rule,
    )
    if citation_path:
        instructions += f"\nSource citation path: {citation_path}"
    if source_as_of:
        instructions += f"\nSource as-of date: {source_as_of}"
    return BlindPrompt(
        instructions=instructions + "\n",
        interface_block="=== INTERFACE ===\n"
        + json.dumps(interface.to_json(), indent=2, sort_keys=True)
        + "\n",
        source_text=source_text,
    )


def _collapse(text: str) -> str:
    return " ".join(text.split())


def _standalone(token: str, text: str) -> bool:
    return (
        re.search(rf"(?<![\w.\-/]){re.escape(token)}(?![\w.\-/])", text, re.ASCII)
        is not None
    )


def _walk_strings(node: Any):
    if isinstance(node, Mapping):
        for value in node.values():
            yield from _walk_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_strings(value)
    elif isinstance(node, str):
        yield node


def _secret_fragments(
    rulespec_payload: Mapping[str, Any],
    companion_cases: Sequence[Any] | None,
) -> tuple[set[str], set[str]]:
    """Return (substring secrets, standalone-token secrets) of a module."""
    substrings: set[str] = set()
    tokens: set[str] = set()

    def add_prose(value: Any) -> None:
        for text in _walk_strings(value):
            collapsed = _collapse(text)
            if len(collapsed) >= 12 and not _NAME_RE.fullmatch(collapsed):
                substrings.add(collapsed)

    def add_formula(value: Any) -> None:
        text = str(value)
        collapsed = _collapse(text)
        if not collapsed:
            return
        if _NAME_RE.fullmatch(collapsed):
            return  # a bare reference to a name the interface already lists
        if re.fullmatch(r"[-+]?[\d_,.]+%?", collapsed):
            tokens.add(collapsed)
            return
        substrings.add(collapsed)
        for line in text.splitlines():
            fragment = _collapse(line)
            if len(fragment) >= 8 and not _NAME_RE.fullmatch(fragment):
                substrings.add(fragment)
        tokens.update(re.findall(r"(?<![\w.])\d[\d_,.]*\d(?![\w.])", collapsed))

    def visit(node: Any, key: str | None = None) -> None:
        if isinstance(node, Mapping):
            for child_key, child in node.items():
                child_key = str(child_key)
                if child_key in {"formula", "value", "amount"} and not isinstance(
                    child, (Mapping, list)
                ):
                    add_formula(child)
                elif child_key in {"effective_from", "effective_to"}:
                    if child is not None:
                        tokens.add(str(child))
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
                    visit(child, child_key)
        elif isinstance(node, list):
            for child in node:
                visit(child, key)

    visit(rulespec_payload)
    for case in companion_cases or ():
        if isinstance(case, Mapping):
            name = case.get("name")
            if isinstance(name, str) and name.strip():
                tokens.add(name.strip())
            add_prose({k: v for k, v in case.items() if k not in {"input", "output"}})
    tokens.discard("")
    return substrings, tokens


def find_blindness_violations(
    prompt: BlindPrompt,
    *,
    rulespec_payload: Mapping[str, Any],
    companion_cases: Sequence[Any] | None = None,
    citation_path: str | None = None,
) -> list[str]:
    """Return what the non-source part of the prompt leaks from the module.

    Checked against instructions plus interface: every formula (whole and
    per line) and the module's prose (summary, source labels, proof
    metadata, test prose). Checked against the interface block, the only
    part derived from the module: every literal number in a formula, every
    parameter value, every effective date and every existing test name, as
    standalone tokens. The corpus source text is not checked: it does not
    come from the module. The citation path names the unit and is allowed.
    """
    collapsed_prompt = _collapse(prompt.controlled)
    substrings, tokens = _secret_fragments(rulespec_payload, companion_cases)
    if citation_path:
        substrings.discard(_collapse(citation_path))
    violations = [
        f"prompt contains module text: {secret[:80]!r}"
        for secret in sorted(substrings)
        if secret in collapsed_prompt
    ]
    violations.extend(
        f"interface contains module value, date or test name: {token!r}"
        for token in sorted(tokens)
        if _standalone(token, prompt.interface_block)
    )
    return violations


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
    earliest_supported_period: str | None = None
    earliest_period_basis: str | None = None
    response_error: str | None = None
    notes: tuple[str, ...] = ()


def sentence_in_source(sentence: str, source_text: str) -> bool:
    """True when ``sentence`` occurs in the source, whitespace runs aside.

    Characters must match exactly, including case and punctuation. Only runs
    of whitespace are folded to one space, because a JSON string cannot carry
    the source's line breaks faithfully.
    """
    needle = _collapse(sentence)
    return bool(needle) and needle in _collapse(source_text)


def _value_issue(value: Any, dtype: str | None, enumeration: Sequence[str]) -> str:
    if isinstance(value, (dict, list)) or value is None:
        return "value must be a JSON scalar"
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
    index: int, raw: Any, interface: RuleInterface, source_text: str
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
    if sentence is None:
        return reject("source_sentence must be a string")
    if len(_collapse(sentence)) < MIN_SOURCE_SENTENCE_CHARS:
        return reject(
            f"source_sentence is shorter than {MIN_SOURCE_SENTENCE_CHARS} characters"
        )
    if not sentence_in_source(sentence, source_text):
        return reject("source_sentence is not found verbatim in the source text")
    return BlindCase(
        name=name,
        period=period,
        inputs=dict(inputs),
        expected=dict(expected),
        source_sentence=sentence,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate JSON key(s): {', '.join(duplicates)}")
    return dict(pairs)


def parse_blind_response(
    response: str,
    *,
    interface: RuleInterface,
    source_text: str,
    source_as_of: str | None = None,
    max_cases: int = DEFAULT_MAX_CASES,
) -> ParsedResponse:
    """Parse the typed answer. Free prose is never interpreted.

    The whole answer must be one JSON object (one enclosing code fence is
    tolerated, since it carries no content). Anything else is a response
    error with zero cases. Each case is validated on its own; a case whose
    quoted sentence is not in the source text is rejected and reported.
    """
    text = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*)\n```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
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
    cases: list[BlindCase] = []
    rejected: list[RejectedCase] = []
    seen: set[str] = set()
    for index, raw in enumerate(payload["cases"], 1):
        if index > max_cases:
            rejected.append(
                RejectedCase(index, None, f"more than {max_cases} cases; not run")
            )
            continue
        result = _validate_case(index, raw, interface, source_text)
        if isinstance(result, BlindCase) and result.name in seen:
            result = RejectedCase(
                index, result.name, "duplicate case name", result.source_sentence
            )
        if isinstance(result, BlindCase):
            seen.add(result.name)
            cases.append(result)
        else:
            rejected.append(result)

    notes: list[str] = []
    earliest = payload.get("earliest_supported_period")
    basis = payload.get("earliest_period_basis")
    if earliest is not None:
        if not isinstance(earliest, str) or not _MONTH_RE.fullmatch(earliest):
            notes.append("earliest_supported_period ignored: not YYYY-MM")
            earliest = basis = None
        elif basis == SOURCE_AS_OF_BASIS and source_as_of:
            pass
        elif not isinstance(basis, str) or not sentence_in_source(basis, source_text):
            notes.append(
                "earliest_supported_period ignored: basis is not a verbatim "
                "source sentence"
            )
            earliest = basis = None
    else:
        basis = None
    return ParsedResponse(
        cases=tuple(cases),
        rejected=tuple(rejected),
        earliest_supported_period=earliest,
        earliest_period_basis=basis,
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

    It reuses the local reviewer's launcher: the same executable resolution,
    the same explicit-allowlist environment (no ambient API key or token is
    forwarded) and the same tool-less, session-less ``claude --print`` flags.
    Unlike the reviewer it never falls back to the Codex CLI: the point of
    the blind author is a model of another family than the generator.
    """

    model: str = REVIEWER_CLI_MODEL
    timeout: int = DEFAULT_TIMEOUT_SECONDS
    cwd: Path | None = None
    name: str = "claude-cli"

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
            result = vp._run_subprocess_with_idle_timeout(
                cmd,
                timeout=self.timeout,
                idle_timeout=self.timeout,
                cwd=self.cwd,
                env=environment,
                env_is_final=True,
            )
        except Exception as exc:  # timeout, missing binary, OS error
            raise BlindAuthorBackendError(f"claude CLI failed: {exc}") from exc
        if result.returncode != 0:
            raise BlindAuthorBackendError(
                f"claude CLI exited {result.returncode}: {result.output[-400:]}"
            )
        return result.output


# ---------------------------------------------------------------------------
# Running cases with the companion-test runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseOutcome:
    status: str
    issues: tuple[str, ...] = ()
    comparisons: tuple[dict[str, Any], ...] = ()
    not_runnable_kind: str | None = None


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

    A case disagrees only when every issue is an output comparison mismatch.
    Any other issue (binding, missing input, no formula version at the
    period, engine failure) means the case did not run: not a disagreement.
    """
    mismatches = [str(item["mismatch"]) for item in comparisons if item.get("mismatch")]
    records = tuple(dict(item) for item in comparisons)
    if not issues:
        if not comparisons:
            return CaseOutcome(
                STATUS_NOT_RUNNABLE,
                ("no output was compared",),
                records,
                "execution_or_binding",
            )
        return CaseOutcome(STATUS_AGREE, (), records)
    if mismatches and all(
        any(issue == m or issue.startswith(m) for m in mismatches) for issue in issues
    ):
        return CaseOutcome(STATUS_DISAGREE, tuple(issues), records)
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
        self._pipeline = _RecordingPipeline(
            policy_repo_path=self.policy_repo_path,
            axiom_rules_path=Path(axiom_rules_path),
            local_corpus_release=None,
            enable_oracles=False,
            axiom_rules_engine_ref=axiom_rules_engine_ref,
        )
        self.compiled_path = Path(work_dir).resolve() / "compiled.json"
        self.compiled_payload: dict[str, Any] | None = None
        self.compile_error: str | None = None
        try:
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
# Orchestration and report
# ---------------------------------------------------------------------------


def _previous_month(month: str) -> str:
    first = date.fromisoformat(f"{month}-01")
    return (first - timedelta(days=1)).strftime("%Y-%m")


def probe_earliest_period(
    parsed: ParsedResponse,
    results: Sequence[tuple[BlindCase, CaseOutcome]],
    runner: CaseRunner,
) -> dict[str, Any]:
    """Does the candidate compute one month before the blind earliest period?

    The blind author names the first month the source text supports. A case
    that ran is run again one month earlier. If the candidate still computes
    an output there, it applies before the period the blind author found
    support for. This is the only way an output-based case can see a rule
    that starts too early.
    """
    probe: dict[str, Any] = {
        "blind_earliest_supported_period": parsed.earliest_supported_period,
        "basis": parsed.earliest_period_basis,
        "finding": "not_probed",
    }
    if parsed.earliest_supported_period is None:
        return probe
    base = next(
        (
            case
            for case, outcome in results
            if outcome.status in (STATUS_AGREE, STATUS_DISAGREE)
        ),
        None,
    )
    if base is None:
        return probe
    probe_period = _previous_month(parsed.earliest_supported_period)
    outcome = runner.run(
        BlindCase(
            name=f"{base.name}__period_probe",
            period=probe_period,
            inputs=base.inputs,
            expected=base.expected,
            source_sentence=base.source_sentence,
        )
    )
    probe["probe_period"] = probe_period
    probe["probe_case"] = base.name
    if outcome.status in (STATUS_AGREE, STATUS_DISAGREE):
        probe["finding"] = "candidate_applies_before_blind_earliest_period"
    elif outcome.not_runnable_kind == "no_formula_version_at_period":
        probe["finding"] = "consistent"
    else:
        probe["issues"] = list(outcome.issues)
    return probe


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
) -> dict[str, Any]:
    """Author blind cases, run them, and return the machine-readable report.

    ``authored_response`` replays a stored answer instead of calling the
    backend, so the same blind cases can run against a second candidate.
    Raises :class:`BlindnessViolation` before any model call when the prompt
    would leak module content.
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
        citation_path=citation_path,
    )
    if violations:
        raise BlindnessViolation("; ".join(violations))
    replayed = authored_response is not None
    response = authored_response if replayed else backend.complete(prompt.text)
    parsed = parse_blind_response(
        response,
        interface=interface,
        source_text=source_text,
        source_as_of=source_as_of,
        max_cases=max_cases,
    )
    results = [(case, runner.run(case)) for case in parsed.cases]
    counts = {
        status: sum(1 for _, outcome in results if outcome.status == status)
        for status in (STATUS_AGREE, STATUS_DISAGREE, STATUS_NOT_RUNNABLE)
    }
    counts["rejected"] = len(parsed.rejected)
    counts["authored"] = len(parsed.cases) + len(parsed.rejected)
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
        },
        "prompt_sha256": prompt.sha256,
        "interface": interface.to_json(),
        "response_error": parsed.response_error,
        "notes": list(parsed.notes),
        "counts": counts,
        "cases": [
            {
                **case.to_json(),
                "status": outcome.status,
                "not_runnable_kind": outcome.not_runnable_kind,
                "issues": list(outcome.issues),
                "comparisons": list(outcome.comparisons),
            }
            for case, outcome in results
        ],
        "rejected_cases": [item.to_json() for item in parsed.rejected],
        "period_probe": probe_earliest_period(parsed, results, runner),
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
    for case in report["cases"]:
        if case["status"] == STATUS_AGREE:
            continue
        lines.append(f"  [{case['status']}] {case['name']} ({case['period']})")
        lines.append(f"      source: {_collapse(case['source_sentence'])!r}")
        for issue in case["issues"][:3]:
            lines.append(f"      {issue}")
    for item in report["rejected_cases"]:
        lines.append(f"  [rejected] #{item['index']} {item['name']}: {item['reason']}")
    probe = report.get("period_probe") or {}
    if probe.get("finding") == "candidate_applies_before_blind_earliest_period":
        lines.append(
            f"  [period] the candidate computes in {probe['probe_period']}, before "
            f"the blind earliest supported period "
            f"{probe['blind_earliest_supported_period']} (basis: {probe['basis']!r})"
        )
    for note in report.get("notes", []):
        lines.append(f"  note: {note}")
    return "\n".join(lines) + "\n"


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BlindTestAuthorError(f"{path} is not a YAML mapping")
    return payload
