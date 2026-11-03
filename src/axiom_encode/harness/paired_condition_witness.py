"""Give every part of a paired source condition its own companion case pair.

The companion-test gate asks one same-period case pair for each source
condition clause. One pair therefore discharged a whole compound clause: a
toggle of ``the applicant is a resident`` stood for ``income is greater than
or equal to the income threshold and the applicant is a resident``, whatever
the formula did with income. A pair that skipped an inclusive bound's equality
point also stood for that bound, so ``>`` passed for ``>=``.

This module reads the closed grammar these clauses use (comparisons and named
flags joined by ``and``/``or``, parenthesized groups, ``both ... and ...``
enumerations, and ``if``/``unless``/``provided`` frames) and asks each part for
its own witness.

Invariants:

* Refusal only. The gate adds the clauses returned here to its missing list and
  never removes one, so no formula it refused before passes now.
* A clause is read only when all of it parses. Any other clause keeps the
  gate's existing verdict.
* A part's witness is one of the gate's executed case pairs. It changes that
  part alone, holds every other part at a value that lets this part decide the
  clause, moves the clause from inactive to active, and changes the asserted
  output in the direction the clause states.
* A comparison is witnessed only through a reached formula comparison of the
  source's own operands, by bare name or the source's literal, with the
  source's relation or its complement. A comparison of two named amounts that
  includes equality also needs its pair to reach equality on the side where the
  comparison holds.
* Two parts never share one case pair.
* Work per clause is capped by the constants below, so finding and parsing
  clauses stays linear in the source text.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from axiom_encode.harness import source_completeness as sc

# Clauses longer than these caps are left to the existing gate, unread.
_MAX_CLAUSE_CHARACTERS = 600
_MAX_CONDITION_TOKENS = 96
_MAX_CONNECTORS_PER_GROUP = 12
_MAX_GROUP_DEPTH = 4
_MAX_PARTS = 8
_MAX_NOUN_PHRASE_WORDS = 8
_MAX_SUBJECT_WORDS = 3
_MAX_DEPENDENCY_RULES = 64

_LEADING_FRAME = re.compile(
    r"(?P<cue>if|when|where|unless|provided(?:\s+that)?)\s+"
    r"(?P<condition>[^,;:]+?)\s*,\s*(?P<effect>[^,;:]+?)"
    r"(?:\s*,\s*provided(?:\s+that)?\s+(?P<proviso>[^,;:]+?))?"
)
_TRAILING_PROVISO_FRAME = re.compile(
    r"(?P<effect>[^,;:]+?)\s*,\s*provided(?:\s+that)?\s+(?P<proviso>[^,;:]+?)"
)
_PROVISO_CUE = re.compile(r"\bprovided\b", re.IGNORECASE)
_EFFECT = re.compile(
    r"(?:(?:the|an?)\s+)?[a-z][a-z'’ -]{0,80}?\s+"
    r"(?:is|are|shall\s+be|will\s+be)\s+(?P<negation>not\s+)?"
    r"(?P<state>eligible|ineligible|allowed|disallowed|entitled|qualified|"
    r"disqualified|denied|available|unavailable)"
)
_NEGATIVE_STATES = frozenset(
    {"ineligible", "disallowed", "disqualified", "denied", "unavailable"}
)
_TOKEN = re.compile(r"\(|\)|\$?\d+(?:,\d{3})*(?:\.\d+)?|[a-z]+(?:['’][a-z]+)?")
_NUMBER = re.compile(r"\$?(?P<digits>\d+(?:,\d{3})*(?:\.\d+)?)")
_DETERMINERS = frozenset({"the", "a", "an"})
_CONNECTORS = frozenset({"and", "or"})
# Words that cannot sit inside an operand: a phrase holding one is a clause.
_NOUN_PHRASE_STOPS = frozenset(
    {
        "are",
        "both",
        "but",
        "do",
        "does",
        "either",
        "except",
        "has",
        "have",
        "if",
        "is",
        "neither",
        "nor",
        "not",
        "or",
        "provided",
        "qualifies",
        "qualify",
        "than",
        "unless",
        "when",
        "where",
        "whether",
        "which",
        "who",
    }
)
_ATTRIBUTE_STOPS = _NOUN_PHRASE_STOPS | _DETERMINERS | {"and", "at", "in", "no"}
_IDENTIFIER_FILLERS = frozenset({"is", "are", "a", "an", "the"})
_DEFAULT_SUBJECTS = frozenset({("applicant",)})

_COMPLEMENT = {">": "<=", ">=": "<", "<": ">=", "<=": ">"}
_MIRROR = {">": "<", ">=": "<=", "<": ">", "<=": ">="}
_AST_RELATIONS = {ast.Gt: ">", ast.GtE: ">=", ast.Lt: "<", ast.LtE: "<="}


def _phrases(relation: str, *templates: str) -> tuple[tuple[tuple[str, ...], str], ...]:
    """Expand ``is|are`` style alternatives into word tuples."""

    expanded: list[tuple[tuple[str, ...], str]] = []
    for template in templates:
        variants: list[tuple[str, ...]] = [()]
        for word in template.split():
            variants = [
                (*prefix, choice) for prefix in variants for choice in word.split("|")
            ]
        expanded.extend((variant, relation) for variant in variants)
    return tuple(expanded)


# ``left RELATION right``. Each phrase is matched as one token, so the ``or`` in
# ``greater than or equal to`` never splits a condition.
_RELATION_PHRASES = (
    *_phrases(
        ">=",
        "is|are greater|more than or equal to",
        "is|are equal to or greater|more than",
        "is|are equal to or exceeds|exceed",
        "equals|equal or exceeds|exceed",
        "exceeds|exceed or equals|equal",
        "is|are at least",
        "is|are not less than",
        "is|are no less than",
        "meets|meet or exceeds|exceed",
    ),
    *_phrases(
        ">",
        "is|are greater|more than",
        "exceeds|exceed",
        "is|are above",
        "is|are in excess of",
    ),
    *_phrases(
        "<=",
        "is|are less than or equal to",
        "is|are equal to or less than",
        "is|are at most",
        "is|are not greater|more than",
        "is|are no greater|more than",
        "does|do not exceed",
    ),
    *_phrases("<", "is|are less than", "is|are below"),
)
_RELATIONS_BY_FIRST_WORD: dict[str, tuple[tuple[tuple[str, ...], str], ...]] = {}
for _words, _relation in sorted(_RELATION_PHRASES, key=lambda item: -len(item[0])):
    _RELATIONS_BY_FIRST_WORD[_words[0]] = (
        *_RELATIONS_BY_FIRST_WORD.get(_words[0], ()),
        (_words, _relation),
    )
_MAX_RELATION_WORDS = max(len(words) for words, _relation in _RELATION_PHRASES)


@dataclass(frozen=True)
class Operand:
    """One side of a source comparison: a named amount or a literal number."""

    text: str
    tokens: tuple[str, ...] = ()
    literal: Decimal | None = None


@dataclass(frozen=True)
class ConditionPart:
    """One comparison or named flag inside a source condition."""

    text: str
    kind: str
    subject: tuple[str, ...] = ()
    attribute: str = ""
    negated: bool = False
    left: Operand | None = None
    relation: str = ""
    right: Operand | None = None

    @property
    def compares_named_amounts(self) -> bool:
        return (
            self.kind == "comparison"
            and self.left is not None
            and self.right is not None
            and self.left.literal is None
            and self.right.literal is None
        )


@dataclass(frozen=True)
class _Node:
    kind: str
    children: tuple[_Node, ...] = ()
    part: ConditionPart | None = None


@dataclass(frozen=True)
class PairedCondition:
    """A fully parsed source clause: when its effect applies, and which effect."""

    parts: tuple[ConditionPart, ...]
    tree: _Node
    requirement: str


@dataclass(frozen=True)
class ConditionPartGap:
    """A source clause and the parts that lack their own executed case pair."""

    branch: sc.SourceStructureBranch
    parts: tuple[ConditionPart, ...]


@dataclass
class _Budget:
    """Counts parser steps; tests read it to bound the work, never the clock."""

    operations: int = 0
    exhausted: bool = field(default=False)

    def spend(self, amount: int = 1) -> None:
        self.operations += amount


@dataclass(frozen=True)
class _Relation:
    relation: str
    words: tuple[str, ...]


def parse_paired_condition(
    text: str, *, budget: _Budget | None = None
) -> PairedCondition | None:
    """Parse one source clause, or return ``None`` when any of it is unknown."""

    budget = budget if budget is not None else _Budget()
    clause = sc._collapse_text(sc._strip_source_clause_marker(text))
    budget.spend(len(clause))
    # Digit-group commas (``$20,000``) are part of a number, not a clause break.
    clause = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", clause.rstrip(" .;:").lower())
    if not clause or len(clause) > _MAX_CLAUSE_CHARACTERS:
        return None
    proviso_text = None
    if match := _LEADING_FRAME.fullmatch(clause):
        cue = re.sub(r"\s+", " ", match.group("cue"))
        condition_text = match.group("condition")
        effect_text = match.group("effect")
        proviso_text = match.group("proviso")
    elif match := _TRAILING_PROVISO_FRAME.fullmatch(clause):
        cue = "provided"
        condition_text = None
        effect_text = match.group("effect")
        proviso_text = match.group("proviso")
    else:
        return None
    # A cue inside the effect, such as ``... if required``, is a qualification
    # this grammar does not read.
    if re.search(
        r"\b(?:if|when|where|unless|provided)\b", _without_parentheses(effect_text)
    ):
        return None
    conditions: list[_Node] = []
    if condition_text is not None:
        condition = _parse_condition(condition_text, budget)
        if condition is None:
            return None
        conditions.append(_Node("not", (condition,)) if cue == "unless" else condition)
    if proviso_text is not None:
        proviso = _parse_condition(proviso_text, budget)
        if proviso is None:
            return None
        conditions.append(proviso)
    tree = conditions[0] if len(conditions) == 1 else _Node("and", tuple(conditions))
    parts = tuple(_leaves(tree))
    if not parts or len(parts) > _MAX_PARTS:
        return None
    return PairedCondition(parts, tree, _effect_requirement(effect_text))


def _without_parentheses(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\([^()]*\)", " ", text)
    return text


def _effect_requirement(text: str) -> str:
    """Read ``<subject> is [not] <state>``; any other effect only must change."""

    effect = sc._collapse_text(_without_parentheses(text))
    match = _EFFECT.fullmatch(effect)
    if match is None:
        return "change"
    negative = (match.group("state") in _NEGATIVE_STATES) != bool(
        match.group("negation")
    )
    return "exclude" if negative else "enable"


def _leaves(node: _Node):
    if node.part is not None:
        yield node.part
    for child in node.children:
        yield from _leaves(child)


def _parse_condition(text: str, budget: _Budget) -> _Node | None:
    tokens = _tokens(text, budget)
    if tokens is None or not tokens or len(tokens) > _MAX_CONDITION_TOKENS:
        return None
    items = _relation_items(tokens, budget)
    nested = _nest(items)
    if nested is None:
        return None
    return _parse_group(nested, budget, depth=0)


def _tokens(text: str, budget: _Budget) -> list[str] | None:
    """Split into words, numbers and parentheses; refuse any other character."""

    tokens = []
    position = 0
    for match in _TOKEN.finditer(text):
        budget.spend()
        if text[position : match.start()].strip():
            return None
        tokens.append(match.group(0))
        position = match.end()
    if text[position:].strip():
        return None
    return tokens


def _relation_items(tokens: list[str], budget: _Budget) -> list[Any]:
    """Replace each longest relation phrase with one relation item."""

    items: list[Any] = []
    index = 0
    while index < len(tokens):
        budget.spend()
        for words, relation in _RELATIONS_BY_FIRST_WORD.get(tokens[index], ()):
            budget.spend()
            if tuple(tokens[index : index + len(words)]) == words:
                items.append(_Relation(relation, words))
                index += len(words)
                break
        else:
            items.append(tokens[index])
            index += 1
    return items


def _nest(items: list[Any]) -> list[Any] | None:
    """Turn parentheses into nested lists; unbalanced groups are unknown."""

    stack: list[list[Any]] = [[]]
    for item in items:
        if item == "(":
            if len(stack) > _MAX_GROUP_DEPTH:
                return None
            stack.append([])
        elif item == ")":
            if len(stack) == 1:
                return None
            group = stack.pop()
            stack[-1].append(group)
        else:
            stack[-1].append(item)
    return stack[0] if len(stack) == 1 else None


def _parse_group(items: list[Any], budget: _Budget, *, depth: int) -> _Node | None:
    """Split one group at a single connector kind into parts, uniquely.

    ``and`` may also join the words of one operand (``earned and unearned
    income``), so every split is tried and exactly one reading must parse.
    Mixed ``and``/``or`` without parentheses has no reading here.
    """

    if depth > _MAX_GROUP_DEPTH or not items:
        return None
    connectors = [
        index
        for index, item in enumerate(items)
        if isinstance(item, str) and item in _CONNECTORS
    ]
    if len(connectors) > _MAX_CONNECTORS_PER_GROUP:
        return None
    readings: list[_Node] = []
    whole = _parse_primary(items, budget, depth=depth)
    if whole is not None:
        readings.append(whole)
    for connector in ("and", "or"):
        points = [index for index in connectors if items[index] == connector]
        if not points:
            continue
        boundaries = [-1, *points, len(items)]
        # ways[j] holds up to two readings of items[: boundaries[j]], each split
        # at chosen boundaries. Two are enough to tell a unique reading apart.
        ways: list[list[tuple[_Node, ...]]] = [[] for _ in boundaries]
        ways[0] = [()]
        for end in range(1, len(boundaries)):
            for start in range(end):
                if not ways[start]:
                    continue
                budget.spend(boundaries[end] - boundaries[start])
                segment = items[boundaries[start] + 1 : boundaries[end]]
                node = _parse_primary(segment, budget, depth=depth)
                if node is None:
                    continue
                for prefix in ways[start]:
                    if len(ways[end]) < 2:
                        ways[end].append((*prefix, node))
        readings.extend(
            _Node(connector, reading) for reading in ways[-1] if len(reading) > 1
        )
    return readings[0] if len(readings) == 1 else None


def _parse_primary(items: list[Any], budget: _Budget, *, depth: int) -> _Node | None:
    budget.spend()
    if len(items) == 1 and isinstance(items[0], list):
        return _parse_group(items[0], budget, depth=depth + 1)
    if any(isinstance(item, list) for item in items):
        return None
    relations = [
        index for index, item in enumerate(items) if isinstance(item, _Relation)
    ]
    if not relations:
        part = _parse_flag(items)
        return _Node("part", part=part) if part is not None else None
    if len(relations) != 1:
        return None
    position = relations[0]
    relation = items[position]
    lefts = _distributed_operands(items[:position])
    rights = _distributed_operands(items[position + 1 :])
    if not lefts or not rights:
        return None
    phrase = " ".join(relation.words)
    distributed = len(lefts) * len(rights) > 1
    parts = tuple(
        _Node(
            "part",
            part=ConditionPart(
                # A distributed part names its own operands; others keep the
                # source words verbatim.
                text=(
                    f"{left.text} {phrase} {right.text}"
                    if distributed
                    else " ".join(
                        " ".join(item.words) if isinstance(item, _Relation) else item
                        for item in items
                    )
                ),
                kind="comparison",
                left=left,
                relation=relation.relation,
                right=right,
            ),
        )
        for left in lefts
        for right in rights
    )
    return parts[0] if len(parts) == 1 else _Node("and", parts)


def _distributed_operands(words: list[Any]) -> tuple[Operand, ...]:
    """Read one operand, or ``both X and Y`` as two that may share a head noun."""

    if any(isinstance(word, (_Relation, list)) for word in words):
        return ()
    if words[:1] != ["both"]:
        operand = _noun_phrase(words)
        return (operand,) if operand is not None else ()
    rest = words[1:]
    if rest.count("and") < 1:
        return ()
    split = rest.index("and")
    first, second = rest[:split], rest[split + 1 :]
    first_core = _strip_determiners(first)
    second_core = _strip_determiners(second)
    if (
        len(first_core) == 1
        and len(second_core) >= 2
        and first_core[0] not in second_core
    ):
        # ``both earned and unearned income``: the modifier shares the head.
        first = [*first_core, *second_core[1:]]
    left, right = _noun_phrase(first), _noun_phrase(second)
    if left is None or right is None or "and" in left.tokens or "and" in right.tokens:
        return ()
    return (left, right)


def _strip_determiners(words: list[str]) -> list[str]:
    index = 0
    while index < len(words) and words[index] in _DETERMINERS:
        index += 1
    return words[index:]


def _noun_phrase(words: list[Any]) -> Operand | None:
    core = _strip_determiners(list(words))
    if not core or len(core) > _MAX_NOUN_PHRASE_WORDS:
        return None
    if len(core) == 1 and (number := _NUMBER.fullmatch(core[0])) is not None:
        try:
            literal = Decimal(number.group("digits").replace(",", ""))
        except InvalidOperation:
            return None
        return Operand(text=core[0], literal=literal)
    if any(not re.fullmatch(r"[a-z]+(?:['’][a-z]+)?", word) for word in core):
        return None
    if any(word in _NOUN_PHRASE_STOPS for word in core):
        return None
    joins = [index for index, word in enumerate(core) if word == "and"]
    # ``earned and unearned income``: one join, with a shared head after it.
    if len(joins) > 1 or any(index == 0 or index + 2 >= len(core) for index in joins):
        return None
    return Operand(text=" ".join(core), tokens=tuple(_word(word) for word in core))


def _parse_flag(words: list[Any]) -> ConditionPart | None:
    """Read ``the S is [not] [a] A``, ``the S qualifies`` or ``does not qualify``."""

    if any(not isinstance(word, str) for word in words):
        return None
    core = _strip_determiners(words)
    text = " ".join(words)
    for verb_index, word in enumerate(core):
        if word in {"is", "are", "qualifies", "qualify", "does", "do"}:
            break
    else:
        return None
    subject = core[:verb_index]
    if (
        not subject
        or len(subject) > _MAX_SUBJECT_WORDS
        or any(
            word in _ATTRIBUTE_STOPS or not re.fullmatch(r"[a-z]+", word)
            for word in subject
        )
    ):
        return None
    predicate = core[verb_index:]
    subject_tokens = tuple(_word(word) for word in subject)
    if predicate in (["qualifies"], ["qualify"]):
        return ConditionPart(
            text=text, kind="flag", subject=subject_tokens, attribute=_word("qualifies")
        )
    if predicate in (["does", "not", "qualify"], ["do", "not", "qualify"]):
        return ConditionPart(
            text=text,
            kind="flag",
            subject=subject_tokens,
            attribute=_word("qualifies"),
            negated=True,
        )
    if predicate[0] not in {"is", "are"}:
        return None
    rest = predicate[1:]
    negated = rest[:1] == ["not"]
    if negated:
        rest = rest[1:]
    if rest[:1] and rest[0] in {"a", "an"}:
        rest = rest[1:]
    if (
        len(rest) != 1
        or rest[0] in _ATTRIBUTE_STOPS
        or not re.fullmatch(r"[a-z]+", rest[0])
    ):
        return None
    return ConditionPart(
        text=text,
        kind="flag",
        subject=subject_tokens,
        attribute=_word(rest[0]),
        negated=negated,
    )


def _word(word: str) -> str:
    """Normalize one word the same way for source text and formula names."""

    word = re.sub(r"['’]s$", "", word.lower())
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    return word


def _identifier_words(name: str) -> tuple[str, ...]:
    normalized = sc._normalized_selector_name(name)
    return tuple(_word(token) for token in re.split(r"[_/.]+", normalized) if token)


def is_active(condition: PairedCondition) -> bool:
    """Compound clauses and comparisons of two named amounts get part witnesses."""

    return len(condition.parts) > 1 or any(
        part.compares_named_amounts for part in condition.parts
    )


def proviso_condition_branches(
    source_text: str,
    *,
    branches: Sequence[sc.SourceStructureBranch],
    active_branches: Sequence[sc.SourceStructureBranch],
    deferred_paths: set[tuple[str, ...]],
    deferred_clauses: Sequence[sc.SourceStructureBranch],
    formula_branches: Sequence[sc.SourceStructureBranch],
    numeric_limits_only: bool,
    exception_branches: Sequence[sc.SourceStructureBranch],
    budget: _Budget | None = None,
) -> tuple[sc.SourceStructureBranch, ...]:
    """Return ``provided`` clauses the gate's condition cues do not reach.

    The gate's cues do not include ``provided``, so ``Provided that A, E,
    provided B`` had no paired demand at all. Only clauses this grammar parses,
    with a readable effect, become demands; the gate's own clause filters
    (deferral, manual numeric limits, formula-owned arithmetic) still apply.
    """

    budget = budget if budget is not None else _Budget()
    budget.spend(len(source_text))
    cues = [
        match
        for match in _PROVISO_CUE.finditer(source_text)
        if not any(
            clause.start <= match.start() and match.end() <= clause.end
            for clause in deferred_clauses
        )
        and not sc._span_is_deferred(
            match.start(),
            match.end(),
            branches=branches,
            deferred_paths=deferred_paths,
        )
        and not any(
            branch.start <= match.start() < branch.end for branch in exception_branches
        )
    ]
    if not cues:
        return ()
    clauses = tuple(sc._source_clause_spans(source_text, branches=branches))
    budget.spend(len(clauses))
    found: dict[tuple[int, int], sc.SourceStructureBranch] = {}
    for match in cues:
        clause = next(
            (
                (start, end, text)
                for start, end, text in clauses
                if start <= match.start() and match.end() <= end
            ),
            None,
        )
        if clause is None:
            continue
        start, end, text = clause
        if (start, end) in found or any(
            branch.start < end and start < branch.end for branch in exception_branches
        ):
            continue
        if (
            sc._EXCEPTION_LANGUAGE.search(text) is None
            and sc._source_has_arithmetic_comparison_condition(text)
            and any(
                formula_branch.start == start and formula_branch.end == end
                for formula_branch in formula_branches
            )
        ):
            continue
        if numeric_limits_only and not sc._STATED_NUMERIC_LIMIT.search(text):
            continue
        condition = parse_paired_condition(text, budget=budget)
        if (
            condition is None
            or not is_active(condition)
            or condition.requirement == "change"
        ):
            continue
        owner = sc._most_specific_containing_branch(
            start, end, branches=active_branches
        ) or sc.SourceStructureBranch(
            (), "source-unit", "source unit", source_text, 0, len(source_text)
        )
        found[(start, end)] = sc.SourceStructureBranch(
            owner.path, "exception-clause", owner.label, text, start, end
        )
    return tuple(found.values())


def condition_part_gaps(
    obligations: Sequence[sc.SourceStructureBranch],
    *,
    principal_rules: dict[str, dict[str, Any]],
    principal_rule_paths: Mapping[str, set[tuple[str, ...]]],
    asserted_by_rule: Mapping[str, list[dict[str, Any]]],
    formula_environment: dict[str, Any] | None,
    toggled_exception_selectors: set[sc._ExceptionWitness],
) -> tuple[ConditionPartGap, ...]:
    """Return each readable clause whose parts lack their own executed pair."""

    formula_environment = formula_environment or {}
    cases_by_identity = {
        id(case): case for cases in asserted_by_rule.values() for case in cases
    }
    context = _WitnessContext(
        principal_rules=principal_rules,
        formula_environment=formula_environment,
        cases_by_identity=cases_by_identity,
    )
    gaps = []
    for branch in obligations:
        condition = parse_paired_condition(branch.text)
        if condition is None or not is_active(condition):
            continue
        missing = _unwitnessed_parts(
            branch,
            condition,
            context=context,
            principal_rule_paths=principal_rule_paths,
            toggled_exception_selectors=toggled_exception_selectors,
        )
        if missing:
            gaps.append(ConditionPartGap(branch, missing))
    return tuple(gaps)


def note_refused_parts(gaps: Sequence[ConditionPartGap]) -> None:
    """Report gaps the gate kept; a test plugin replaces this to audit them."""


def describe_part(part: ConditionPart) -> str:
    """Name what the part's own case pair must show."""

    detail = "its own same-period case pair"
    if part.compares_named_amounts and part.relation in {">=", "<="}:
        detail += (
            " through the formula's comparison of exactly these two amounts, "
            "including a case at equality"
        )
    elif part.kind == "comparison":
        detail += " through the formula's comparison of exactly these operands"
    return (
        f"`{sc._bounded_source_feedback_excerpt(part.text, limit=160)}` needs {detail}"
    )


@dataclass
class _WitnessContext:
    principal_rules: dict[str, dict[str, Any]]
    formula_environment: dict[str, Any]
    cases_by_identity: dict[int, dict[str, Any]]
    environments: dict[int, dict[str, Any] | None] = field(default_factory=dict)
    dependencies: dict[int, dict[str, Any]] = field(default_factory=dict)
    comparisons: dict[tuple[str, int], frozenset[tuple[Any, str, Any]] | None] = field(
        default_factory=dict
    )

    def dependency_environment(self, case: dict[str, Any]) -> dict[str, Any]:
        key = id(case)
        if key not in self.dependencies:
            self.dependencies[key] = sc._case_asserted_dependency_environment(
                self.principal_rules,
                case,
                formula_environment=self.formula_environment,
                allow_bound_row_assertions=True,
            )
        return self.dependencies[key]

    def environment(self, case: dict[str, Any]) -> dict[str, Any] | None:
        key = id(case)
        if key not in self.environments:
            environment = sc._formula_case_runtime_environment(
                case,
                dependency_environment=self.dependency_environment(case),
                formula_environment=self.formula_environment,
            )
            if environment is not None:
                replayed = sc._case_dependency_environment(
                    self.principal_rules,
                    case,
                    formula_environment=self.formula_environment,
                    require_asserted_value=False,
                )
                for name, value in replayed.items():
                    environment.setdefault(name, value)
            self.environments[key] = environment
        return self.environments[key]

    def reached_comparisons(
        self, rule_name: str, case: dict[str, Any]
    ) -> frozenset[tuple[Any, str, Any]] | None:
        key = (rule_name, id(case))
        if key not in self.comparisons:
            rule = self.principal_rules[rule_name]
            execution = sc._case_formula_execution(
                rule,
                case,
                formula_environment=self.formula_environment,
                dependency_environment=self.dependency_environment(case),
            )
            if execution is None:
                # Upstream values the case does not assert still replay locally.
                execution = sc._case_formula_execution(
                    rule,
                    case,
                    formula_environment=self.formula_environment,
                    dependency_environment=sc._case_dependency_environment(
                        self.principal_rules,
                        case,
                        formula_environment=self.formula_environment,
                        require_asserted_value=False,
                    ),
                )
            self.comparisons[key] = (
                None if execution is None else _execution_comparisons(execution)
            )
        return self.comparisons[key]


def _execution_comparisons(execution: Any) -> frozenset[tuple[Any, str, Any]]:
    """Return reached comparisons whose two sides are bare names or numbers."""

    texts = [selector for step in execution.trace for selector in step.selectors]
    texts.append(execution.leaf)
    found = set()
    for text in texts:
        expression = sc._parse_formula_expression(text)
        if expression is None:
            continue
        for node in ast.walk(expression):
            if not isinstance(node, ast.Compare) or len(node.ops) != 1:
                continue
            relation = _AST_RELATIONS.get(type(node.ops[0]))
            left = _bare_operand(node.left)
            right = _bare_operand(node.comparators[0])
            if relation is not None and left is not None and right is not None:
                found.add((left, relation, right))
    return frozenset(found)


def _bare_operand(node: ast.AST) -> Any:
    if isinstance(node, ast.Name):
        return ("name", node.id)
    negative = False
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        negative, node = True, node.operand
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        lexeme = sc._formula_expression_source(node).replace("_", "")
        try:
            value = Decimal(lexeme)
        except InvalidOperation:
            return None
        return ("literal", -value if negative else value)
    return None


def _affecting_rules(
    branch: sc.SourceStructureBranch,
    principal_rule_paths: Mapping[str, set[tuple[str, ...]]],
) -> set[str]:
    return {
        rule_name
        for rule_name, paths in principal_rule_paths.items()
        if not branch.path
        or any(
            path == branch.path[: len(path)] or branch.path == path[: len(branch.path)]
            for path in paths
            if path
        )
    }


def _formula_names(
    rule_names: set[str], principal_rules: Mapping[str, dict[str, Any]]
) -> set[str]:
    """Names the affecting formulas read, through local dependency rules."""

    names: set[str] = set()
    pending = sorted(rule_names)
    visited: set[str] = set()
    while pending and len(visited) < _MAX_DEPENDENCY_RULES:
        rule_name = pending.pop()
        if rule_name in visited or rule_name not in principal_rules:
            continue
        visited.add(rule_name)
        referenced = set(
            sc._FORMULA_IDENTIFIER.findall(
                sc._rule_formula_text(principal_rules[rule_name])
            )
        )
        names |= referenced
        pending.extend(sorted(referenced & principal_rules.keys() - visited))
    return names


@dataclass(frozen=True)
class _Binding:
    """Formula names that may stand for one part's flag or operands."""

    flag: frozenset[str] = frozenset()
    left: frozenset[str] = frozenset()
    right: frozenset[str] = frozenset()


def _bind(
    part: ConditionPart,
    names: set[str],
    default_subjects: frozenset[tuple[str, ...]],
) -> _Binding | None:
    words = {name: _identifier_words(name) for name in names}
    if part.kind == "flag":
        expected = (*part.subject, part.attribute)
        matched = frozenset(
            name
            for name, tokens in words.items()
            if (filtered := tuple(t for t in tokens if t not in _IDENTIFIER_FILLERS))
            == expected
            or (part.subject in default_subjects and filtered == (part.attribute,))
        )
        return _Binding(flag=matched) if matched else None

    def operand_names(operand: Operand | None) -> frozenset[str] | None:
        if operand is None:
            return None
        if operand.literal is not None:
            return frozenset()
        accepted = {operand.tokens}
        for subject in default_subjects:
            if operand.tokens[: len(subject)] == subject and len(operand.tokens) > len(
                subject
            ):
                accepted.add(operand.tokens[len(subject) :])
        matched = frozenset(
            name for name, tokens in words.items() if tokens in accepted
        )
        return matched or None

    left = operand_names(part.left)
    right = operand_names(part.right)
    if left is None or right is None:
        return None
    return _Binding(left=left, right=right)


def _default_subjects(condition: PairedCondition) -> frozenset[tuple[str, ...]]:
    subjects = {part.subject for part in condition.parts if part.kind == "flag"}
    if len(subjects) == 1:
        return _DEFAULT_SUBJECTS | subjects
    return _DEFAULT_SUBJECTS


def _number(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    number = sc._rulespec_runtime_decimal(value)
    if number is None or not number.is_finite():
        return None
    return number


def _operand_value(
    operand: Operand, names: frozenset[str], environment: Mapping[str, Any]
) -> Decimal | None:
    if operand.literal is not None:
        return operand.literal
    values = {_number(environment.get(name)) for name in names if name in environment}
    if len(values) != 1:
        return None
    return next(iter(values))


def _compare(left: Decimal, relation: str, right: Decimal) -> bool:
    return {
        ">": left > right,
        ">=": left >= right,
        "<": left < right,
        "<=": left <= right,
    }[relation]


def _part_value(
    part: ConditionPart, binding: _Binding, environment: Mapping[str, Any]
) -> bool | None:
    if part.kind == "flag":
        values = {
            sc._boolean_value(environment.get(name))
            for name in binding.flag
            if name in environment
        }
        if len(values) != 1 or None in values:
            return None
        value = next(iter(values))
        return (not value) if part.negated else value
    assert part.left is not None and part.right is not None
    left = _operand_value(part.left, binding.left, environment)
    right = _operand_value(part.right, binding.right, environment)
    if left is None or right is None:
        return None
    return _compare(left, part.relation, right)


def _evaluate(node: _Node, values: Mapping[ConditionPart, bool]) -> bool:
    if node.part is not None:
        return values[node.part]
    if node.kind == "not":
        return not _evaluate(node.children[0], values)
    results = (_evaluate(child, values) for child in node.children)
    return all(results) if node.kind == "and" else any(results)


def _operand_matches(operand: Operand, names: frozenset[str], bare: Any) -> bool:
    kind, value = bare
    if operand.literal is not None:
        return kind == "literal" and value == operand.literal
    return kind == "name" and value in names


def _comparison_witnessed(
    part: ConditionPart,
    binding: _Binding,
    comparisons: frozenset[tuple[Any, str, Any]],
) -> tuple[str, str] | None:
    """Find a reached formula comparison that is this part or its complement.

    Returns the matched formula names for the left and right source operands,
    or ``None``. Arithmetic on either side (``threshold + 1``) never matches.
    """

    assert part.left is not None and part.right is not None
    for left, relation, right in sorted(comparisons, key=repr):
        for first, second, formula_relation in (
            (left, right, relation),
            (right, left, _MIRROR[relation]),
        ):
            if formula_relation not in {part.relation, _COMPLEMENT[part.relation]}:
                continue
            if _operand_matches(part.left, binding.left, first) and _operand_matches(
                part.right, binding.right, second
            ):
                return (
                    first[1] if first[0] == "name" else "",
                    second[1] if second[0] == "name" else "",
                )
    return None


def _unwitnessed_parts(
    branch: sc.SourceStructureBranch,
    condition: PairedCondition,
    *,
    context: _WitnessContext,
    principal_rule_paths: Mapping[str, set[tuple[str, ...]]],
    toggled_exception_selectors: set[sc._ExceptionWitness],
) -> tuple[ConditionPart, ...]:
    principal_rules = context.principal_rules
    affecting = _affecting_rules(branch, principal_rule_paths) & principal_rules.keys()
    names = _formula_names(affecting, principal_rules)
    default_subjects = _default_subjects(condition)
    bindings = {part: _bind(part, names, default_subjects) for part in condition.parts}
    unbound = tuple(part for part in condition.parts if bindings[part] is None)
    if unbound:
        # The affecting formulas never name this part's flag or operands, so no
        # pair can show the formula implements it, nor hold it for the others.
        return unbound
    candidates: dict[int, set[tuple[int, ...]]] = {
        index: set() for index in range(len(condition.parts))
    }
    for witness in sorted(toggled_exception_selectors):
        if (
            witness.rule_name not in affecting
            or witness.calendar_attainment_age is not None
        ):
            continue
        witnessed = _witnessed_part(
            witness,
            condition,
            bindings=bindings,  # type: ignore[arg-type]
            context=context,
        )
        if witnessed is not None:
            candidates[witnessed].add(witness.case_pair_identity)
    missing = sc._unmatched_evidence_obligations(candidates)
    return tuple(condition.parts[index] for index in sorted(missing))


def _witnessed_part(
    witness: sc._ExceptionWitness,
    condition: PairedCondition,
    *,
    bindings: Mapping[ConditionPart, _Binding],
    context: _WitnessContext,
) -> int | None:
    """Return the one part this executed pair witnesses, if it witnesses one."""

    if len(witness.case_pair_identity) != 2:
        return None
    cases = [context.cases_by_identity.get(key) for key in witness.case_pair_identity]
    if any(case is None for case in cases):
        return None
    environments = [context.environment(case) for case in cases]  # type: ignore[arg-type]
    if any(environment is None for environment in environments):
        return None
    if witness.numeric_transition is None:
        selected = [
            sc._boolean_value(environment.get(witness.selector_name))  # type: ignore[union-attr]
            for environment in environments
        ]
        if None in selected or selected[0] == selected[1]:
            return None
        exception_index = 0 if selected[0] == witness.active_value else 1
    else:
        values = [
            _number(environment.get(witness.selector_name))  # type: ignore[union-attr]
            for environment in environments
        ]
        if None in values:
            return None
        ordinary_value, exception_value = (
            Decimal(str(value)) for value in witness.numeric_transition
        )
        if values == [ordinary_value, exception_value]:
            exception_index = 1
        elif values == [exception_value, ordinary_value]:
            exception_index = 0
        else:
            return None
    ordinary = environments[1 - exception_index]
    exception = environments[exception_index]
    assert ordinary is not None and exception is not None
    ordinary_values: dict[ConditionPart, bool] = {}
    exception_values: dict[ConditionPart, bool] = {}
    for part in condition.parts:
        before = _part_value(part, bindings[part], ordinary)
        after = _part_value(part, bindings[part], exception)
        if before is None or after is None:
            return None
        ordinary_values[part] = before
        exception_values[part] = after
    changed = [
        index
        for index, part in enumerate(condition.parts)
        if ordinary_values[part] != exception_values[part]
    ]
    # The pair must change one part alone and move the clause from inactive to
    # active; every other part then holds a value that lets this part decide.
    if (
        len(changed) != 1
        or _evaluate(condition.tree, ordinary_values)
        or not _evaluate(condition.tree, exception_values)
    ):
        return None
    index = changed[0]
    part = condition.parts[index]
    binding = bindings[part]
    if part.kind == "flag":
        if (
            witness.numeric_transition is not None
            or witness.selector_name not in binding.flag
        ):
            return None
    else:
        if witness.numeric_transition is None or witness.selector_name not in (
            binding.left | binding.right
        ):
            return None
        matched: tuple[str, str] | None = None
        for case in cases:
            comparisons = context.reached_comparisons(witness.rule_name, case)  # type: ignore[arg-type]
            if comparisons is None:
                return None
            found = _comparison_witnessed(part, binding, comparisons)
            if found is None or (matched is not None and found != matched):
                return None
            matched = found
        if part.compares_named_amounts and part.relation in {">=", "<="}:
            assert matched is not None
            holding = exception if exception_values[part] else ordinary
            left_value = _number(holding.get(matched[0]))
            right_value = _number(holding.get(matched[1]))
            if left_value is None or left_value != right_value:
                return None
    if not sc._exception_witness_satisfies_requirement(
        witness,
        condition.requirement,
        rule=context.principal_rules[witness.rule_name],
    ):
        return None
    return index
