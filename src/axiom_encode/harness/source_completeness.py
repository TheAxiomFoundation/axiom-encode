"""Opt-in authoritative source-unit completeness accounting.

This module deliberately sits beside, rather than inside, the numeric
extractor.  It consumes the extractor and scalar-inventory interfaces supplied
by :mod:`validator_pipeline`, which keeps locale-tokenizer work independent
from the source-unit admission policy implemented here.
"""

from __future__ import annotations

import ast
import bisect
import calendar
import contextlib
import contextvars
import copy
import difflib
import functools
import hashlib
import heapq
import itertools
import math
import re
import textwrap
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import (
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    Decimal,
    InvalidOperation,
    localcontext,
)
from typing import TYPE_CHECKING, Any, Mapping, Protocol

import yaml

from axiom_encode.bare_structure import (
    bare_definition_headers,
    bare_outline,
    mask_bare_labels,
)
from axiom_encode.form_numeric_structure import form_numeric_locator_spans
from axiom_encode.form_source_structure import mask_form_display
from axiom_encode.harness import hebrew_source_text
from axiom_encode.harness.imported_export_evidence import ResolvedImportedExport
from axiom_encode.harness.manual_locator_metadata import (
    MANUAL_CONTENTS_HEADINGS,
    MANUAL_SECTION_CAPTIONS,
    MISSOURI_FOOD_STAMP_BLOCK_SECTIONS,
    OREGON_OPEN_REVISION_CAPTIONS,
    REVIEWED_MANUAL_COMPOSITIONS,
    REVIEWED_MANUAL_LOCATOR_SPANS,
    REVIEWED_MANUAL_REVISION_STAMPS,
)
from axiom_encode.harness.measurement_role_evidence import (
    BoundMeasurementRoles,
    ExecutedMeasurementWitness,
    MeasurementSourceEvidence,
    bind_measurement_roles,
)
from axiom_encode.harness.partnership_reporting_evidence import (
    certify_partnership_reporting,
)
from axiom_encode.harness.proof_validator import (
    _normalize_atom_path,
    _source_proof_evidence_spans,
)
from axiom_encode.harness.repayment_alternative_evidence import (
    RepaymentAlternativeCoverage,
    certify_repayment_alternatives,
)
from axiom_encode.harness.reporting_route_evidence import certify_reporting_routes
from axiom_encode.numeric_equality import rulespec_numeric_values_equal
from axiom_encode.printed_source_rows import (
    PrintedRowStructure,
    recognize_printed_rows,
)
from axiom_encode.statute import (
    CitationParts,
    normalize_rulespec_path_segment,
    parse_usc_citation,
)

from .same_case_execution import SameCaseEvidence, formula_witnesses

if TYPE_CHECKING:
    from axiom_encode.harness.worksheet_composition_evidence import (
        BoundWorksheetComposition,
    )
    from axiom_encode.harness.worksheet_replacement_evidence import (
        BoundWorksheetReplacement,
    )

_UNBOUND_FORMULA_DIAGNOSTIC_RULE_LIMIT = 32
_UNBOUND_FORMULA_DIAGNOSTIC_CASE_LIMIT = 8
_TEMPORAL_WITNESS_NAME_LIMIT = 32
_TEMPORAL_WITNESS_VERSION_LIMIT = 8
_PARAMETER_ALIAS_FORMULA_KEYWORDS = {
    "and",
    "else",
    "false",
    "holds",
    "if",
    "match",
    "none",
    "not",
    "not_holds",
    "null",
    "or",
    "true",
}


class NumericOccurrenceLike(Protocol):
    """Typed numeric source evidence consumed from the shared extractor."""

    value: float
    start: int
    end: int
    raw: str
    has_rate_context: bool
    has_temporal_context: bool
    has_structural_context: bool
    source_value: float | None
    requires_rate_context: bool
    is_word_number: bool
    alternative_values: tuple[float, ...]


@dataclass(frozen=True)
class _NumericOccurrenceView:
    """One typed occurrence rebased into a containing source fragment."""

    value: float
    start: int
    end: int
    raw: str
    has_rate_context: bool
    has_temporal_context: bool
    has_structural_context: bool
    source_value: float | None
    requires_rate_context: bool
    is_word_number: bool
    alternative_values: tuple[float, ...]
    # Carried through rebasing so a month number read from a month name keeps
    # grounding only month comparisons (see validator_pipeline).
    requires_month_context: bool = False


NumericOccurrenceExtractor = Callable[[str], Sequence[NumericOccurrenceLike]]
NamedScalarExtractor = Callable[[str], Sequence[Any]]
NumericGroundingPredicate = Callable[
    [float, Iterable[NumericOccurrenceLike]],
    bool,
]


@dataclass(frozen=True)
class SourceStructureBranch:
    """One operative structural branch in an authoritative source unit."""

    path: tuple[str, ...]
    kind: str
    label: str
    text: str
    start: int
    end: int
    structural_numeric_spans: tuple[tuple[int, int], ...] = ()
    computation_chapeaux: tuple[tuple[int, int, str], ...] = ()
    required_principal_output: str | None = None

    @property
    def computation_chapeau(self) -> tuple[int, int, str] | None:
        """The nearest authenticated cue, when a computation inherits context."""
        return self.computation_chapeaux[-1] if self.computation_chapeaux else None

    @property
    def computation_text(self) -> str:
        """Operation context, separate from the offset-preserving source excerpt."""
        inherited = (
            text
            for start, end, text in self.computation_chapeaux
            if not (self.start <= start and end <= self.end)
        )
        return "\n".join((*inherited, self.text))


@dataclass(frozen=True)
class _OutlineMarker:
    """One resolved marker in a nested parenthesized legal outline."""

    start: int
    path: tuple[str, ...]
    kind: str
    label: str


@dataclass(frozen=True)
class CompleteSourceUnitAnalysis:
    """Deterministic completeness results for one RuleSpec/source pair."""

    issues: tuple[str, ...]
    branches: tuple[SourceStructureBranch, ...]
    source_numeric_occurrence_count: int
    covered_source_numeric_occurrence_count: int
    missing_source_numeric_occurrence_count: int


@dataclass(frozen=True)
class _SourceImperativeArithmeticOperation:
    """A source instruction with raw operands, not resolved RuleSpec values."""

    kind: str
    start: int
    end: int
    first_operand: tuple[int, int]
    second_operand: tuple[int, int]
    left_operand: tuple[int, int]
    right_operand: tuple[int, int]


@dataclass(frozen=True)
class _NumericInterval:
    """One source-stated range retaining exact endpoint evidence."""

    lower: NumericOccurrenceLike | None
    lower_inclusive: bool
    upper: NumericOccurrenceLike | None
    upper_inclusive: bool
    upper_directly_conjoined: bool = False


@dataclass(frozen=True)
class _FormulaTraceStep:
    """One control-flow decision evaluated by a companion case."""

    kind: str
    selectors: tuple[str, ...]
    choice: int


@dataclass(frozen=True)
class _CurrencyRounding:
    mode: str
    minor_units: int


class _CurrencyRoundingRule(dict):
    """A parsed rule with privately resolved module-level currency metadata."""

    def __init__(self, rule: dict[str, Any], rounding: _CurrencyRounding | None):
        super().__init__(rule)
        self.resolved_currency_rounding = rounding


@dataclass(frozen=True)
class _FormulaExecution:
    """The selected control-flow path and reconstructed reachable formula."""

    trace: tuple[_FormulaTraceStep, ...]
    leaf: str
    evaluated_value: tuple[str, str] | None
    evaluates_to_zero: bool
    constant_environment: dict[str, Any]
    currency_rounding: _CurrencyRounding | None = None
    unrounded_value: Decimal | None = None


@dataclass(frozen=True)
class _SourceConditionClause:
    """One exact source proposition owned by a formula proof excerpt."""

    branch_path: tuple[str, ...]
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class _TerminalGateAlternative:
    """Terminal facts used by one active formula path and temporal partition."""

    names: frozenset[str]
    start: str | None
    end: str | None
    resolved: bool = True


@dataclass(frozen=True)
class _TerminalGateEvidence:
    """Semantic evidence from one terminal with fail-closed polarity metadata."""

    entities: frozenset[str]
    predicates: frozenset[str]
    polarity_conflict: bool
    explicit_name_polarities: frozenset[tuple[str, bool]] = frozenset()
    numeric_income_amount: bool = False


@dataclass(frozen=True)
class _UnboundFormulaDiagnostic:
    """Best-effort unbound formula matches and scan completeness."""

    rule_names: tuple[str, ...]
    scan_capped: bool


@dataclass(frozen=True)
class _UncorroboratedFormulaDependencyDiagnostic:
    """Best-effort missing derived assertions for a formula witness."""

    dependency_names: tuple[str, ...]
    case_names: tuple[str, ...]
    scan_capped: bool


@dataclass(frozen=True)
class _TemporalFormulaValue:
    """Literal parameter values selectable by a companion case period."""

    versions: tuple[tuple[str, str, Any], ...]
    version_formula_excerpts: tuple[tuple[str, ...], ...] = ()
    imported_parameter: bool = False


@dataclass(frozen=True)
class _FormulaBranchNode:
    """One parsed RuleSpec conditional or match expression."""

    start: int
    end: int
    kind: str
    selectors: tuple[str, ...]
    patterns: tuple[str, ...]
    choices: tuple[str, ...]


@dataclass(frozen=True, order=True)
class _ExceptionWitness:
    """One isolated exception selector effect demonstrated by paired cases."""

    rule_name: str
    selector_name: str
    active_value: bool
    blocks: bool
    boolean_effect: bool
    zeroes: bool
    numeric_transition: tuple[float, float] | None
    relational_transitions: tuple[tuple[str, str, str], ...] = ()
    case_pair_identity: tuple[int, ...] = ()
    calendar_attainment_age: int | None = None
    maximum_transitions: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class _NamedSourceNumber:
    """A matched source number whose word value may be unsupported."""

    value: float | None


_PARAGRAPH_MARKER = re.compile(
    r"(?m)^[ \t]*(?P<marker>\((?P<label>\d+[a-z]?|[a-z])\))(?=\s|bis\b)",
    flags=re.IGNORECASE,
)
_DOTTED_SUBSECTION_MARKER = re.compile(
    r"(?m)^[ \t]*(?P<marker>(?P<label>[A-Z])\.)(?:[ \t]+|(?=\r?$)|(?=\())"
)
_JOINED_DOTTED_BOUNDARY_MARKER = re.compile(
    r"(?m)^[ \t]*(?P<marker>(?P<label>[A-Z])\.)"
    r"(?=(?:\d+[A-Z]?(?::|\.\d)|(?i:e)\.(?:[ \t]|(?=\r?$))))"
)
_NUMBER_MARKER = re.compile(
    r"(?m)^[ \t]*(?P<marker>(?P<label>\d+[a-z]?)\.)[ \t]+",
    flags=re.IGNORECASE,
)
_INLINE_DOTTED_NUMBER_MARKER = re.compile(
    r"(?<![A-Za-z0-9.])(?P<marker>(?P<label>[1-9]\d{0,2})\.)[ \t]+"
)
_INLINE_DOTTED_LIST_MINIMUM_LENGTH = 3
_INLINE_DOTTED_LIST_MATCH_LIMIT = 256
_INLINE_DOTTED_LIST_CHAPEAU_LIMIT = 512
_INLINE_DOTTED_LIST_STRUCTURAL_CHAPEAU = re.compile(
    r"^\s*(?:(?:(?:all|each)\s+of\s+the|the)\s+)?following\s+(?:tax\s+)?"
    r"(?:allowances?|benefits?|conditions?|credits?|deductions?|eligibility|"
    r"exceptions?|exemptions?|qualifications?|requirements?)\b"
    r"(?:\s*,\s*(?:when|where|if)\s+applicable\s*,|"
    r"\s+(?:when|where|if)\s+applicable)?\s+"
    r"(?:appl(?:y|ies)|(?:shall|must|may)\s+(?:apply|hold|be\s+"
    r"(?:allowed|available|claimed|required|satisfied|met))|"
    r"(?:are|is)\s+(?:allowed|available|required|satisfied|met)|"
    r"(?:shall|must)(?:\s*,\s*(?:when|where|if)\s+applicable\s*,)?\s+"
    r"be\s+deducted(?:\s+from\s+the\s+result\s+obtained\s+under\s+"
    r"subsection\s+\(\d+[a-z]?\)\s+of\s+this\s+section\s+to\s+arrive\s+at\s+"
    r"the\s+annual\s+tax)?)"
    r"(?:\s+(?:(?:only\s+)?when|if)\s+(?:"
    r"no\s+(?:[a-z][a-z'-]*\s+){0,5}"
    r"(?:income|earnings?|wages?|benefits?|credits?|deductions?|allowances?|"
    r"exemptions?)\s+(?:(?:is|are|was|were)|(?:has|have|had)\s+been)\s+"
    r"(?:reported|received|earned|available|claimed|allowed|paid|provided)|"
    r"(?:[a-z][a-z'-]*\s+){0,5}"
    r"(?:income|earnings?|wages?|benefits?|credits?|deductions?|allowances?|"
    r"exemptions?)\s+(?:is|are|was|were)\s+not\s+"
    r"(?:reported|received|earned|available|claimed|allowed|paid|provided)|"
    r"(?:the\s+)?(?:applicant|claimant|dependent|household|individual|person|"
    r"recipient|spouse|taxpayer|taxpayer(?:'s|’s)\s+spouse)\s+"
    r"(?:(?:is|are|was|were)\s+not\s+(?:"
    r"(?:(?:an?|the)\s+)?(?:(?:another\s+)?taxpayer(?:'s|’s)\s+)?"
    r"(?:dependent|eligible|qualified|blind|disabled|married|resident)|"
    r"the\s+dependent\s+of\s+another\s+taxpayer)|"
    r"(?:has|have|had)\s+no\s+(?:[a-z][a-z'-]*\s+){0,5}"
    r"(?:income|earnings?|wages?|benefits?|credits?|deductions?|allowances?|"
    r"exemptions?)|"
    r"(?:is|are|was|were)\s+"
    r"(?:dependent|eligible|qualified|blind|disabled|married|resident))"
    r"))?\s*:\s*$",
    flags=re.IGNORECASE,
)
_LETTER_MARKER = re.compile(
    r"(?m)^[ \t]*(?P<marker>(?P<label>[a-z]{1,2})\))[ \t]+",
    flags=re.IGNORECASE,
)
_PARENTHESIZED_OUTLINE_MARKER = re.compile(
    r"(?m)^[ \t]*(?:[A-Z]\.)?"
    r"(?P<marker>\((?:\d+[a-z]?|[a-z]|[ivxlcdm]{2,15})\)"
    r"(?:[ \t]*\((?:\d+[a-z]?|[a-z]|[ivxlcdm]{2,15})\))*)"
    r"(?=\s|bis\b)",
    flags=re.IGNORECASE,
)
_INLINE_PARENTHESIZED_OUTLINE_MARKER = re.compile(
    r"(?:(?<=[.!?;:])[ \t]+|(?<=—)[ \t]*)"
    r"(?P<marker>\((?:\d+[a-z]?|[a-z]|[ivxlcdm]{2,15})\)"
    r"(?:[ \t]*\((?:\d+[a-z]?|[a-z]|[ivxlcdm]{2,15})\))*)"
    r"(?=\s|bis\b)",
    flags=re.IGNORECASE,
)
_INLINE_OUTLINE_REFERENCE_CONTEXT = re.compile(
    r"\b(?:section|subsection|paragraph|subparagraph|clause|subclause|item|"
    r"subitem|division|subdivision|part|subpart|chapter|subchapter|title|article|"
    r"subarticle)s?"
    r"(?:\s+(?:no\.?|number)(?:\s*[A-Za-z0-9][A-Za-z0-9.-]*)?)?"
    r"\s*[:;,.!?—]?\s*$|"
    r"\b(?:see|compare(?:\s+(?:to|with))?|refer(?:\s+back)?(?:\s+to)?|"
    r"subject\s+to|under|pursuant\s+to|according\s+to|"
    r"in\s+accordance\s+with|for\s+purposes\s+of|"
    r"as\s+(?:provided|specified|described|stated)\s+in)\s*:\s*$",
    flags=re.IGNORECASE,
)
_INLINE_OUTLINE_NAMED_REFERENCE_CONTEXT = re.compile(
    r"\b(?:see|compare(?:\s+(?:to|with))?|refer(?:\s+back)?(?:\s+to)?|"
    r"subject\s+to|under|pursuant\s+to|according\s+to|"
    r"in\s+accordance\s+with|for\s+purposes\s+of|"
    r"as\s+(?:provided|specified|described|stated)\s+in)\s+"
    r"(?:(?!\n|[;!?—]|\.(?=\s+[A-Z])).){1,96}[:;.!?—]\s*$",
    flags=re.IGNORECASE,
)
_INLINE_OUTLINE_STRUCTURAL_CHAPEAU_CONTEXT = re.compile(
    r"\b(?:the\s+)?following\s+"
    r"(?:sections?|subsections?|paragraphs?|subparagraphs?|clauses?|"
    r"subclauses?|items?|subitems?|divisions?|subdivisions?|parts?|subparts?)"
    r"\s*:\s*$",
    flags=re.IGNORECASE,
)
_INLINE_OUTLINE_CHAPEAU_REFERENCE_LEAD = re.compile(
    r"\b(?:see|compare(?:\s+(?:to|with))?|refer(?:\s+back)?(?:\s+to)?|"
    r"subject\s+to|under|pursuant\s+to|according\s+to|"
    r"in\s+accordance\s+with|for\s+purposes\s+of|"
    r"as\s+(?:provided|specified|described|stated)\s+in)"
    r"\s+(?:the\s+)?following\b",
    flags=re.IGNORECASE,
)
_INLINE_OUTLINE_EXPLICIT_REFERENCE_COMMAND = re.compile(
    r"(?:^|[.!?;:—]\s+)(?:"
    r"(?:see(?:\s*,\s*e\.g\.,?)?|refer(?:\s+back)?(?:\s+to)?|"
    r"compare(?:\s+(?:to|with))?|subject\s+to|pursuant\s+to|"
    r"according\s+to|in\s+accordance\s+with|for\s+purposes\s+of|under)\b|"
    r"(?:except\s+)?as\s+(?:provided|specified|described|stated)\s+"
    r"(?:in|under)\b)"
    r".{0,160}[:;.!?—]\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)
_RECALL_UNIT_WORD = (
    r"(?i:days?|weeks?|months?|years?|hours?|percent|members?|persons?|"
    r"people|dollars?|cents?)"
)
_GLUED_SENTENCE_MARKER = re.compile(
    r"(?<![\w])(?P<label>[1-9]\d?)"
    r"(?![A-ZÄÖÜ](?:\d+)?\b)"
    r"(?!(?i:st|nd|rd|th)\b)"
    rf"(?!{_RECALL_UNIT_WORD}\b|(?i:per[ \t\u00a0\u202f]+cent)\b)"
    r"(?=[A-ZÄÖÜ](?!:)(?![ \t]*[.:/\-\u2010-\u2015\u2212\ufe58\ufe63\uff0d]"
    r"[ \t]*\d))"
)
_GLUED_SECTION_SENTENCE_MARKER = re.compile(
    r"(?<![\w])(?P<label>[1-9]\d?)(?=§{1,2}[ \t]*[1-9]\d*)"
)
_EXPLICIT_SENTENCE_MARKER = re.compile(
    r"(?:(?<=^)|(?<=[.;])|(?<=\)))[ \t]*Satz[ \t]+"
    r"(?P<label>[1-9]\d?)(?:[ \t]*:[ \t]*|[ \t]+)(?=[A-ZÄÖÜ])",
    flags=re.IGNORECASE | re.MULTILINE,
)
_EDITORIAL_OMISSION_ONLY = re.compile(
    r"^\s*"
    r"(?:(?:\((?:\d+[a-z]?|[a-z])\)|\d+[a-z]?\.|[a-z]\))\s*)?"
    r"(?:bis\s+\(\d+[a-z]?\)\s*)?"
    r"(?:\(\s*)?"
    r"(?:weggefallen|aufgehoben|omitted|\.{3,})"
    r"(?:\s*\))?"
    r"\s*[.;]?\s*$",
    flags=re.IGNORECASE,
)
# The Hebrew variants of shared patterns are read only behind the wall's gate
# (`_hebrew_script`, #1882); each unprefixed name is main's literal.
_HEBREW_EDITORIAL_OMISSION_ONLY = re.compile(
    r"^\s*"
    r"(?:(?:\((?:\d+[a-z]?|[a-z])\)|\d+[a-z]?\.|[a-z]\))\s*)?"
    rf"(?:\({hebrew_source_text.HEBREW_OUTLINE_LABEL}\)\s*)?"
    r"(?:bis\s+\(\d+[a-z]?\)\s*)?"
    r"(?:\(\s*)?"
    r"(?:weggefallen|aufgehoben|omitted|\.{3,}|"
    rf"{hebrew_source_text.EDITORIAL_OMISSION_WORDS})"
    r"(?:\s*\))?"
    r"\s*[.;]?\s*$",
    flags=re.IGNORECASE,
)
_GERMAN_CARDINAL_VALUES = {
    "ein": 1.0,
    "eins": 1.0,
    "eine": 1.0,
    "einen": 1.0,
    "einem": 1.0,
    "einer": 1.0,
    "eines": 1.0,
    "zwei": 2.0,
    "drei": 3.0,
    "vier": 4.0,
    "fünf": 5.0,
    "sechs": 6.0,
    "sieben": 7.0,
    "acht": 8.0,
    "neun": 9.0,
    "zehn": 10.0,
}
_SLASH_CONJUNCTION = re.compile(r"\b(?:and\s*/\s*or|und\s*/\s*oder)\b", re.IGNORECASE)
# Agency-manual typography that the arithmetic recognizer would otherwise read
# as operators: a month/year date (every page header of the Oregon eligibility
# notebook reads "93 (07/2026)"; `1/2000 of income` stays a fraction), web
# addresses, whose path slashes and hyphens are neither division nor
# subtraction, and footnote asterisks. An asterisk is masked only when it
# stands between a word of two or more letters and whitespace or punctuation
# ("Application Status *includes screenshots", "verification* is required"),
# so `rate* 12`, `2 *x` and `a*b` stay multiplication.
_SLASH_MONTH_YEAR = re.compile(
    r"(?<![\w/.])(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}(?![\w/])(?!\s+of\b)"
)
_WEB_ADDRESS = re.compile(
    r"\bhttps?://\S+"
    r"|(?<![\w.@-])www\.\S+"
    r"|(?<![\w.@-])(?:[A-Za-z0-9-]{1,63}\.){1,10}"
    r"(?:com|gov|org|net|edu|us|info)/\S*"
)
_FOOTNOTE_ASTERISK = re.compile(
    r"(?:(?<=\s)|^)\*(?=[A-Za-z]{2})"
    r"|(?<=[A-Za-z]{2})\*(?=[ \t]+[A-Za-z]{2}|[ \t]*(?:$|[.,;:)\n]))"
)
_LIST_BULLET = "•"
# Masked typography becomes this sentinel, not a space, so the operand and
# operator patterns cannot join the words on either side of it.
_TYPOGRAPHY_MASK = "\x00"
# In a US agency manual a slash or a spaced dash between two words is prose
# ("case/individual", "SNAP/Cash", "Processing - SNAP"), not division or
# subtraction. `_us_manual_prose_scope` sets this flag for one analysis or one
# prompt coverage index of a US manual citation path.
_US_MANUAL_PROSE_SCOPE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_US_MANUAL_PROSE_SCOPE", default=False
)
_MANUAL_WORD_SLASH = re.compile(r"(?<=[A-Za-z])[ \t]*/[ \t]*(?=[A-Za-z])")
# A text that states an equation or a division keeps its slashes and dashes,
# so a formula such as `Family member allowance = (monthly maintenance needs
# standard - family member income) ÷ 3` stays a computation.
_MANUAL_FORMULA_SIGNAL = re.compile(r"[\w)][ \t]*=[ \t]*[\w($]|÷")
_MANUAL_SPACED_WORD_DASH = re.compile(r"(?<=[A-Za-z])[ \t]+[-–][ \t]+(?=[A-Za-z])")
_ARITHMETIC_EXPRESSION = re.compile(
    r"(?:\d+(?:[.,]\d+)?|[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß]*)"
    r"[ \t]*(?:[+*/=×·•∗∙]|(?<!\w)[−–-](?!\w))[ \t]*"
    r"(?:\d+(?:[.,]\d+)?|[A-Za-zÄÖÜäöüß][A-Za-zÄÖÜäöüß]*)"
)
_SYMBOLIC_ARITHMETIC_OPERAND = (
    r"(?:"
    r"[A-Za-zÄÖÜäöüß_][A-Za-z0-9ÄÖÜäöüß_]*|"
    r"\d{1,3}(?:[ .]\d{3})+(?:,\d+)?|"
    r"\d+(?:[.,]\d+)?"
    r")"
)
_WORDED_ARITHMETIC_EXPRESSION = re.compile(
    rf"{_SYMBOLIC_ARITHMETIC_OPERAND}\s+"
    r"(?:plus|minus|mal|less(?!\s+than\b))\s+(?:the\s+)?"
    r"(?!(?:beträgt|gilt|ist|sind|wird|werden|equals?|applies?)\b)"
    rf"{_SYMBOLIC_ARITHMETIC_OPERAND}",
    flags=re.IGNORECASE,
)
_SYMBOLIC_ARITHMETIC_TOKEN = re.compile(
    rf"\b(?:plus|minus|mal)\b|{_SYMBOLIC_ARITHMETIC_OPERAND}|"
    r"[()+*/×·•∗∙−–-]",
    flags=re.IGNORECASE,
)
_COMPUTATION_LANGUAGE = re.compile(
    r"\b(?:"
    r"bemisst\s+sich\s+nach|berechn(?:et|en|ung)|ergibt\s+sich|"
    r"(?:zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn)fache|"
    r"(?:das\s+)?(?:doppelte|dreifache|vierfache|fünffache|sechsfache|"
    r"siebenfache|achtfache|neunfache|zehnfache)\s+(?:des|der|von)|"
    r"hälfte|zehntausendstel|übersteigenden\s+teils|"
    r"(?:ein(?:e[nsrm]?)?\s+)?(?:drittel|viertel|fünftel|sechstel|"
    r"siebtel|achtel|neuntel|zehntel)\s+(?:des|der|von)|"
    r"summe\s+(?:aus|der|von)|produkt\s+(?:aus|der|von)|"
    r"unterschied|differenz|"
    r"geteilt\s+durch|durch\s+(?:\d+|[a-zäöüß]+)\s+geteilt|"
    r"multipliziert\s+mit|"
    r"mit\s+(?:(?:dem|einem)\s+faktor\s+)?"
    r"(?:von\s+)?"
    r"(?:\d+(?:[.,]\d+)?|[a-zäöüß]+)\s+zu\s+"
    r"(?:multiplizieren|vervielfachen)|"
    r"durch\s+multiplikation\s+mit\s+"
    r"(?:(?:(?:dem|einem)\s+)?faktor\s+)?(?:von\s+)?"
    r"(?:\d+(?:[.,]\d+)?|[a-zäöüß]+)\s+zu\s+ermitteln|"
    r"unter\s+anwendung\s+(?:des|eines)\s+faktors?\s+"
    r"(?:\d+(?:[.,]\d+)?|[a-zäöüß]+)\s+zu\s+ermitteln|"
    r"(?:ist|sind)\s+zu\s+(?:"
    r"verdoppeln|verdreifachen|vervierfachen|verfünffachen|"
    r"versechsfachen|versiebenfachen|verachtfachen|verneunfachen|"
    r"verzehnfachen|halbieren)|"
    r"durch\s+halbierung\s+zu\s+ermitteln|"
    r"in\s+(?:2|zwei)\s+gleiche\s+teile\s+zu\s+teilen|"
    r"durch\s+(?:\d+(?:[.,]\d+)?|[a-zäöüß]+)\s+zu\s+teilen|"
    r"um\s+(?:\d+(?:[.,]\d+)?|[a-zäöüß]+)\s+zu\s+"
    r"(?:erhöhen|vermindern|kürzen|vermehren)|"
    r"aus\s+[^.;]{1,100}\s+zu\s+(?:summieren|addieren)|"
    r"\bmal\s+(?:\d+(?:[.,]\d+)?|"
    r"ein(?:s|e[nsrm]?)?|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn)|"
    r"(?:wird|werden)\s+(?:verdoppelt|verdreifacht|vervierfacht)|"
    r"\d+(?:[.,]\d+)?\s*-?fach(?:e[nsrm]?)?|"
    r"(?:vermindert|erhöht|gekürzt|vermehrt)\s+um|"
    r"(?:erhöht|mindert|vermindert|kürzt|vermehrt)\s+sich\s+um|"
    r"(?:abzüglich|zuzüglich)|"
    r"(?:prozent|\d+(?:[.,]\d+)?\s*%)\s+(?:des|der|von|of)|"
    r"\d+(?:[.,]\d+)?\s+(?:vom\s+hundert|v\.?\s*h\.?)\s+"
    r"(?:des|der|von)|"
    r"splitting-verfahren|verfahren\s+nach\s+absatz|"
    r"calculated|computed|computation|multiplied|divided|"
    r"twice|"
    r"amount\s+of\s+(?:the\s+)?excess|"
    r"\d+(?:[.,]\d+)?\s+times\b|"
    r"percentage\s+of|"
    r"equals?[^.;]{0,100}\b(?:plus|minus|times)\b"
    r")\b",
    flags=re.IGNORECASE,
)
_ENGLISH_NUMBER_WORD = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|"
    r"thousand|million|billion|trillion)"
)
_ENGLISH_WORDED_PERCENTAGE_OF = re.compile(
    rf"\b{_ENGLISH_NUMBER_WORD}"
    rf"(?:(?:[-\s]+(?:and[-\s]+)?){_ENGLISH_NUMBER_WORD})*"
    r"\s+(?:percent|per\s+cent)\s+of\b",
    flags=re.IGNORECASE,
)
_EXPLICIT_NUMERIC_PERCENTAGE_OF = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|per\s+cent|percent)\s+of\b",
    flags=re.IGNORECASE,
)
_ENGLISH_CARDINAL_PHRASE = (
    rf"{_ENGLISH_NUMBER_WORD}"
    rf"(?:(?:[-\s]+(?:and[-\s]+)?){_ENGLISH_NUMBER_WORD})*"
)
_EXCESS_BOUND_TAX_RATE_VALUE = (
    rf"(?:\d+(?:[.,]\d+)?\s*(?:%|percent|per\s+cent)|"
    rf"{_ENGLISH_CARDINAL_PHRASE}\s+(?:percent|per\s+cent))"
)
_EXCESS_BOUND_TAX_RATE_PREDICATE = (
    rf"(?:"
    rf"(?:is|are|shall|must|will)\s+(?:be\s+)?taxed\s+at\s+"
    rf"(?:the\s+rate\s+of\s+)?{_EXCESS_BOUND_TAX_RATE_VALUE}|"
    rf"tax\s+(?:is|shall\s+be|must\s+be|will\s+be)\s+imposed\s+at\s+"
    rf"(?:a\s+)?rate\s+of\s+{_EXCESS_BOUND_TAX_RATE_VALUE}"
    rf")"
)
_EXCESS_BOUND_TAX_RATE_LANGUAGE = re.compile(
    rf"\b(?:not\s+)?in\s+excess\s+of\b[^.;]{{0,180}}"
    rf"\b{_EXCESS_BOUND_TAX_RATE_PREDICATE}(?=\W|$)|"
    rf"\b{_EXCESS_BOUND_TAX_RATE_PREDICATE}(?=\W|$)[^.;]{{0,180}}"
    rf"\b(?:not\s+)?in\s+excess\s+of\b",
    flags=re.IGNORECASE,
)
_ENGLISH_ORDINAL_WORD = (
    r"(?:second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|"
    r"eighteenth|nineteenth|twentieth|thirtieth|fortieth|fiftieth|sixtieth|"
    r"seventieth|eightieth|ninetieth|hundredth|thousandth|millionth|billionth|"
    r"trillionth)"
)
_ENGLISH_FRACTION_DENOMINATOR = rf"(?:half|halves|quarters?|{_ENGLISH_ORDINAL_WORD}s?)"
_ENGLISH_FRACTION_PHRASE = (
    rf"(?:half|quarter|(?:a|an)[-\s]+(?:half|quarter|{_ENGLISH_ORDINAL_WORD})|"
    rf"{_ENGLISH_CARDINAL_PHRASE}[-\s]+"
    rf"(?:(?:{_ENGLISH_NUMBER_WORD})[-\s]+)*{_ENGLISH_FRACTION_DENOMINATOR})"
)
_ENGLISH_FRACTION_OF = re.compile(
    rf"\b(?P<fraction>{_ENGLISH_FRACTION_PHRASE})\s+of\s+"
    rf"(?P<target>[^.;:\n]+)",
    flags=re.IGNORECASE,
)
_ENGLISH_CARDINAL_SEQUENCE = rf"{_ENGLISH_NUMBER_WORD}(?:[-\s]+{_ENGLISH_NUMBER_WORD})*"
_ENGLISH_FRACTIONAL_PERCENTAGE_OF = re.compile(
    rf"\b(?:{_ENGLISH_CARDINAL_SEQUENCE}\s+and\s+)?"
    rf"(?:half|quarter|(?:a|an|{_ENGLISH_CARDINAL_SEQUENCE})[-\s]+"
    rf"{_ENGLISH_FRACTION_DENOMINATOR})\s+"
    rf"(?:percent|per\s+cent)\s+of\b",
    flags=re.IGNORECASE,
)
_FORMULA_TABLE_NAME = r"(?:table|schedule)(?:\s+[A-Z0-9]+(?:-[A-Z0-9]+)*)?"
_FORMULA_GENERIC_TABLE_TARGET = (
    r"(?:the\s+)?(?:"
    r"following\s+(?:amounts?|percentages?|rates?|values?|tables?|schedules?)"
    r"(?:\s+for\s+(?:the\s+)?following\s+tax\s+years?)?|"
    rf"following\s+{_FORMULA_TABLE_NAME}|"
    rf"{_FORMULA_TABLE_NAME}\s+(?:set\s+forth\s+)?below|"
    rf"{_FORMULA_TABLE_NAME}|"
    r"(?:amounts?|percentages?|rates?|values?)\s+"
    r"(?:(?:set\s+forth|shown|prescribed)\s+)?in\s+(?:the\s+)?"
    rf"(?:following\s+{_FORMULA_TABLE_NAME}|{_FORMULA_TABLE_NAME}\s+below))"
)
_FORMULA_GENERIC_TABLE_REFERENCE = (
    r"(?:pursuant\s+to|as\s+set\s+forth\s+in|in\s+accordance\s+with|"
    r"from|by\s+reference\s+to)\s+"
    rf"{_FORMULA_GENERIC_TABLE_TARGET}"
)
_FORMULA_NONOPERATIVE_TABLE_HEADING = re.compile(
    rf"(?:\b(?:calculated|computed|determined)\s+(?:"
    r"as\s+follows|"
    rf"{_FORMULA_GENERIC_TABLE_REFERENCE}|"
    rf"(?:using|according\s+to|based\s+on|by|under|in|with|"
    rf"as\s+(?:shown|provided)\s+in|in\s+the\s+manner\s+shown\s+in)\s+"
    rf"{_FORMULA_GENERIC_TABLE_TARGET}"
    rf"|based\s+on\s+[^.;:\n]{{1,240}}?\band\s+(?:\w+\s+)?"
    rf"(?:be\s+)?equal\s+to\s+{_FORMULA_GENERIC_TABLE_TARGET}"
    rf")|\b(?:equals?|(?:be\s+)?equal\s+to)\s+{_FORMULA_GENERIC_TABLE_TARGET})"
    r"(?:\s+for\s+(?:taxable|tax|calendar|fiscal)\s+years?\s+"
    r"(?:\d{4}(?:(?:\s*,\s*(?:(?:and|or)\s+)?|\s+(?:and|or|through|to)\s+|"
    r"\s*[-–]\s*)\d{4})*|beginning\s+[^.;:\n]{0,80}\b\d{4}))?"
    r"\s*:\s*$",
    flags=re.IGNORECASE,
)
_FORMULA_COMPUTED_OPERATION_LANGUAGE = re.compile(
    r"\b(?:calculated|computed|determined)\s+(?:"
    r"(?:by|through)\s+(?:adding|addition|subtracting|subtraction|dividing|"
    r"division|multiplying|multiplication|reducing|reduction|deducting|"
    r"deduction|increasing|increase|decreasing|decrease)\b|"
    r"as\s+(?:the\s+)?(?:addition|average|difference|mean|median|ratio|"
    r"quotient|remainder|product|sum|total|min|max|minimum|maximum|least|"
    r"greatest|lesser|greater|lower|higher|lowest|highest|smallest|largest|"
    r"smaller|larger|percent|percentage)\s+(?:of|between)\b"
    r")",
    flags=re.IGNORECASE,
)
_FORMULA_INDEPENDENT_COORDINATE = (
    r"(?:,\s*(?:(?:although|and|because|but|considering\s+that|given\s+that|"
    r"inasmuch\s+as|nor|notwithstanding\s+that|or|provided(?:\s+that)?|since|"
    r"seeing\s+that|then|though|whereas|while|yet)\s+)?|"
    r"\s+(?:although|and|because|but|considering\s+that|given\s+that|"
    r"inasmuch\s+as|nor|notwithstanding\s+that|or|provided(?:\s+that)?|since|"
    r"seeing\s+that|then|though|whereas|while|yet)\s+)"
    r"(?:the\s+)?"
    r"(?!(?:shall|must|may|is|are|equals?)\b)[^,.;:\n]+?"
    r"\b(?:shall|must|may|is|are|equals?)\b"
    r"|\s+as\s+(?:the\s+)?(?:amounts?|assessments?|benefits?|credits?|"
    r"deductions?|exemptions?|incomes?|liabilit(?:y|ies)|payments?|"
    r"surcharges?|surtaxes?|taxes?)\b[^,.;:\n]*?"
    r"\b(?:shall|must|may|is|are|equals?)\b"
)
_FORMULA_APPLIED_OPERATION_LANGUAGE = re.compile(
    r"\b(?:calculated|computed|determined)\s+(?:"
    r"(?:by|through)\s+(?:applying|application\s+of))\s+"
    r"(?P<operands>[^.;:\n]+?)"
    rf"(?={_FORMULA_INDEPENDENT_COORDINATE}|[.;:\n]|$)",
    flags=re.IGNORECASE,
)
_FORMULA_RESULT_PREDICATE = (
    r"(?:(?:is|are|shall\s+be|must\s+be|may\s+be)\s+equal\s+to|"
    r"(?:shall|must|may)\s+equal|is|are|equals?|shall\s+be|must\s+be|"
    r"may\s+be|means|constitutes?)"
)
_FORMULA_UNCONDITIONAL_OPERATOR_LANGUAGE = re.compile(
    rf"\b(?:calculated|computed)\s+"
    rf"(?!(?:using|according\s+to|under|as\s+follows|based\s+on|"
    rf"by|in|with|without|pursuant\s+to|on\s+the\s+basis\s+of|"
    rf"in\s+the\s+manner|"
    rf"{_FORMULA_GENERIC_TABLE_REFERENCE})\b)|"
    r"\b(?:sum|product)\s+of\b[^.;:\n]{1,120}\band\b[^.;:\n]{1,120}"
    r"\b(?:is|equals?|shall\s+be)\s+\d+(?:[.,]\d+)?\b|"
    r"\bdifference\s+between\b[^.;:\n]{1,120}\band\b[^.;:\n]{1,120}"
    r"\b(?:is|equals?|shall\s+be)\s+\d+(?:[.,]\d+)?\b",
    flags=re.IGNORECASE,
)
_FORMULA_ROUNDED_OPERATION_LANGUAGE = re.compile(
    r"\b(?:(?:sum|product)\s+of|difference\s+between)\b"
    r"[^.;:\n]{1,120}\band\b[^.;:\n]{1,120}"
    r"\b(?:shall|must|may|is|are)\s+(?:be\s+)?rounded\b",
    flags=re.IGNORECASE,
)
_FORMULA_FOLLOWING_OPERAND_PROVISO = re.compile(
    r"\s*(?:(?:and\s+)?provided"
    r"(?:\s*,?\s*(?:always|further|however|in\s+any\s+event|"
    r"nevertheless|nonetheless|only)\s*,?)*\s+that|"
    r"(?:on|upon)\s+(?:the\s+)?condition\s+that|(?:only\s+)?if|when|where|"
    r"(?:so|as)\s+long\s+as|in\s+which\s+case|in\s+the\s+event(?:\s+that)?|"
    r"in\s+(?:(?:a|any|the)\s+)?cases?(?:\s+where)?|save\s+(?:that|where)|"
    r"to\s+the\s+extent\s+that|except\s+as\s+(?:(?:otherwise\s+)?"
    r"(?:provided|specified)|required|prescribed(?:\s+by)?)|"
    r"except\s+(?:that|if|when|where|"
    r"to\s+the\s+extent\s+that|in\s+(?:(?:the|any)\s+)?cases?\s+"
    r"(?:of|where))|"
    r"however|but|unless|subject(?:\s*,?\s*however\s*,?)?\s+to|"
    r"notwithstanding|(?:the|such|a|an)\s+(?:\w+\s+){0,8}"
    r"(?:shall|must|may|is|are|equals?))\b",
    flags=re.IGNORECASE,
)
_FORMULA_NONNEGATIVE_FLOOR_CONTROL = (
    r"(?:(?:can|could|may|might|must|shall|should|will|would)\s+|"
    r"(?:is|are)\s+(?:(?:designed|expected|intended|required|supposed)\s+)?"
    r"to\s+|"
    r"(?:is|are)\s+|(?:has|have|had)\s+(?:been\s+)?(?:"
    r"(?:always|at\s+all\s+times)\s+)?)?(?:"
    r"(?:(?:in\s+(?:no\s+event|no\s+case)\s+|never\s+)?"
    r"(?:(?:not\s+)?(?:be\s+)?|be\s+not\s+)(?:(?:less|lower)\s+than|below|"
    r"no\s+(?:less|lower)\s+than|at\s+least|"
    r"greater\s+than\s+or\s+equal\s+to|equal\s+to\s+or\s+greater\s+than)\s+"
    r"(?:zero|\$?\s*0(?:\.0+)?))|"
    r"(?:(?:always|at\s+all\s+times)\s+)?(?:remain|stay)(?:s|ed)?\s+"
    r"(?:(?:at\s+least|at\s+or\s+above|no\s+(?:less|lower)\s+than|"
    r"greater\s+than\s+or\s+equal\s+to)\s+"
    r"(?:zero|\$?\s*0(?:\.0+)?)|(?:at\s+)?zero\s+or\s+"
    r"(?:above|greater|higher|more))|"
    r"not\s+(?:be\s+(?:an?\s+)?negative(?:\s+amount)?|"
    r"(?:become|turn)\s+negative|"
    r"(?:drop|go|sink)\s+below\s+(?:zero|\$?\s*0(?:\.0+)?)|"
    r"fall\s+below\s+(?:zero|\$?\s*0(?:\.0+)?)|"
    r"result\s+in\s+(?:an?\s+)?negative\s+(?:amount|balance|value))|"
    r"never\s+(?:(?:be|become|remain(?:s|ed)?|turn)\s+)?negative|"
    r"(?:(?:always|never|at\s+all\s+times|in\s+all\s+cases)\s+)?"
    r"(?:(?:be|remain(?:s|ed)?|stay(?:s|ed)?)\s+)?"
    r"(?:not\s+negative|non-?negative)"
    r"(?:\s+(?:at\s+all\s+times|throughout))?|"
    r"(?:does?|do)\s+not\s+(?:become|turn)\s+negative|"
    r"(?:at\s+no\s+time|in\s+no\s+(?:event|case)|never|"
    r"under\s+no\s+circumstances)\s+(?:be|become|turn)\s+negative|"
    r"(?:be\s+)?maintained\s+at\s+no\s+(?:less|lower)\s+than\s+"
    r"(?:zero|\$?\s*0(?:\.0+)?)|"
    r"be\s+(?:(?:treated\s+as|deemed|set\s+(?:at|to))\s+)?zero\s+"
    r"(?:if|when|whenever)\s+(?:(?:(?:it|they)|(?:the\s+)?"
    r"(?:amounts?|benefits?|credits?|incomes?|results?|taxes?|totals?|values?))"
    r"\s+(?:is|are)\s+)?negative|"
    r"(?:at\s+no\s+time|in\s+no\s+(?:event|case)|"
    r"under\s+no\s+circumstances|never)\s+"
    r"(?:(?:can|could|may|might|must|shall|should|will|would|is|are)\s+)?"
    r"(?:(?:it|they)\s+|(?:(?:a|an|any|each|every|the|this|that|such)\s+)?"
    r"(?:[A-Za-z][A-Za-z-]*\s+){0,5}"
    r"(?:amounts?|benefits?|credits?|deductions?|incomes?|"
    r"liabilit(?:y|ies)|payments?|results?|taxes?|totals?|values?)\s+)?"
    r"(?:(?:be|become|turn)\s+negative|"
    r"be\s+(?:less|lower)\s+than\s+(?:zero|\$?\s*0(?:\.0+)?)|"
    r"fall\s+below\s+(?:zero|\$?\s*0(?:\.0+)?)|"
    r"result\s+in\s+(?:an?\s+)?negative\s+(?:amount|balance|value))|"
    r"never\s+(?:carry|have|show)\s+(?:an?\s+)?negative\s+"
    r"(?:amount|balance|value)|"
    r"be\s+(?:zero|\$?\s*0(?:\.0+)?)\s+or\s+(?:above|greater|higher|more)|"
    r"(?:have|has|had)\s+(?:(?:a\s+)?(?:floor|lower\s+bound|"
    r"minimum(?:\s+value)?)\s+(?:at|of)\s+(?:zero|\$?\s*0(?:\.0+)?)|"
    r"(?:a\s+)?(?:zero|\$?\s*0(?:\.0+)?)\s+(?:as\s+a\s+)?lower\s+bound)|"
    r"be\s+bounded\s+(?:from\s+)?below\s+(?:at|by)\s+"
    r"(?:zero|\$?\s*0(?:\.0+)?)|"
    r"(?:cannot|can\s+not)\s+(?:(?:be|become|turn)\s+negative|"
    r"(?:drop|fall|go)\s+below\s+(?:zero|\$?\s*0(?:\.0+)?)|"
    r"(?:carry|have|show)\s+(?:an?\s+)?negative\s+"
    r"(?:amount|balance|value)))"
)
_FORMULA_ADMINISTRATIVE_TITLE_LANGUAGE = re.compile(
    r"\b(?:acts?|agreements?|administration|amendments?|appendices|appendix|"
    r"briefs?|bulletins?|bylaws?|"
    r"archives?|books?|brochures?|catalogs?|chapters?|circulars?|codes?|"
    r"contracts?|criteria|decrees?|directives?|documents?|"
    r"dashboards?|databases?|decisions?|documentation|exhibits?|factsheets?|"
    r"faqs?|files?|flyers?|forms?|guidance|infographics?|"
    r"guides?|guidelines|handbooks?|"
    r"instructions?|letters?|"
    r"laws?|legislation|manuals?|memos?|memorand(?:a|ums?)|methodolog(?:y|ies)|"
    r"news|newsletters?|notes?|notices?|operations?|opinions?|orders?|"
    r"ordinances?|pamphlets?|primers?|"
    r"plans?|polic(?:y|ies)|procedures?|processes?|programs?|protocols?|"
    r"overviews?|portals?|press|publications?|records?|registers?|regulations?|"
    r"references?|releases?|"
    r"reports?|repositories?|requirements?|rules?|rulings?|sites?|summaries?|"
    r"tutorials?|"
    r"websites?|"
    r"advisor(?:y|ies)|articles?|papers?|resolutions?|schedules?|sections?|"
    r"specifications?|standards?|"
    r"statutes?|stud(?:y|ies)|tax|whitepapers?|work|workbooks?|workpapers?|"
    r"worksheets?|workflows?)\b",
    flags=re.IGNORECASE,
)
_FORMULA_ADMINISTRATIVE_TITLE_QUALIFIER_LANGUAGE = re.compile(
    r"\b(?:archived|attachments?|contents?|copies|copy|data|drafts?|excerpts?|"
    r"final|pages?|revs?|revisions?|supplements?|texts?|versions?)\b",
    flags=re.IGNORECASE,
)
_FORMULA_CODE_NAMESPACE_TITLE_LANGUAGE = re.compile(
    r"\b(?:polic(?:y|ies)|rules?|schedules?|sections?|tax|work)\b",
    flags=re.IGNORECASE,
)
_FORMULA_ROUNDING_ACTOR_ROLE_LANGUAGE = re.compile(
    r"\b(?:administrators?|agenc(?:y|ies)|authorit(?:y|ies)|boards?|bureaus?|"
    r"accountants?|advisers?|advisors?|agents?|analysts?|assessors?|attorneys?|"
    r"auditors?|clerks?|collectors?|consultants?|contractors?|inspectors?|"
    r"judges?|liaisons?|preparers?|"
    r"commissions?|commissioners?|committees?|"
    r"councils?|"
    r"chairs?|coordinators?|counsels?|custodians?|departments?|directors?|"
    r"divisions?|employees?|examiners?|leads?|managers?|officers?|offices?|"
    r"representatives?|secretar(?:y|ies)|services?|specialists?|supervisors?|"
    r"trustees?)\b",
    flags=re.IGNORECASE,
)
_FORMULA_RESULT_OPERATION_LANGUAGE = re.compile(
    rf"\b{_FORMULA_RESULT_PREDICATE}\s+(?:the\s+)?(?:"
    r"min|max|minimum|maximum|least|greatest|lesser|greater|lower|higher|"
    r"lowest|highest|smallest|largest|smaller|larger|sum|total|average|mean|"
    r"median|ratio|quotient|difference|product|remainder|percentage|percent)\s+"
    r"(?:amount\s+)?(?:of|between)\b|"
    rf"\b{_FORMULA_RESULT_PREDICATE}\s+(?:the\s+)?(?:"
    r"addition\s+of\b[^.;:\n]{1,120}\band\b|"
    r"(?:subtraction|deduction)\s+of\b[^.;:\n]{1,120}\bfrom\b|"
    r"(?:reduction|increase|decrease|division|multiplication)\s+of\b"
    r"[^.;:\n]{1,120}\bby\b)",
    flags=re.IGNORECASE,
)
_STATUTORY_CREDIT_DIFFERENCE_CONSEQUENCE = (
    r"the taxpayer shall claim from the state not more than the difference "
    r"between the amount of the state child care credit for which the taxpayer "
    r"is eligible and the federal credit for child and dependent care expenses "
    r"the taxpayer is able to deduct from federal tax liability "
    r"for the same taxable year[.;]?"
)
_FORMULA_DIRECT_FOLLOWING_OPERANDS = re.compile(
    r"\b(?:addition|average|difference|greater|greatest|higher|highest|larger|"
    r"largest|least|lesser|lower|lowest|max|maximum|mean|median|min|minimum|"
    r"product|quotient|ratio|remainder|smaller|smallest|sum|total)\s+"
    r"(?:of|between)\s+(?:the\s+)?following\s+"
    r"(?:amounts?|percentages?|rates?|values?)\b",
    flags=re.IGNORECASE,
)
_FORMULA_PARTICIPIAL_RESULT_LANGUAGE = re.compile(
    r"\b(?:(?:is|are|shall\s+be|must\s+be|may\s+be)\s+)?"
    r"(?:reduced|deducted|increased|decreased)\s+by\b",
    flags=re.IGNORECASE,
)
_FORMULA_NUMERIC_RESULT_HEADS = frozenset(
    {
        "allowance",
        "amount",
        "addition",
        "assessment",
        "average",
        "balance",
        "base",
        "benefit",
        "bracket",
        "cap",
        "ceiling",
        "charge",
        "count",
        "credit",
        "decrease",
        "deduction",
        "denominator",
        "deficiency",
        "distribution",
        "difference",
        "divisor",
        "excess",
        "exemption",
        "expense",
        "factor",
        "fee",
        "floor",
        "fraction",
        "income",
        "increase",
        "liability",
        "limit",
        "loss",
        "margin",
        "mean",
        "median",
        "networth",
        "number",
        "numerator",
        "offset",
        "overpayment",
        "payment",
        "percent",
        "percentage",
        "product",
        "proceeds",
        "quantity",
        "quotient",
        "rate",
        "ratio",
        "rebate",
        "reduction",
        "refund",
        "remainder",
        "result",
        "shortfall",
        "surcharge",
        "surtax",
        "sum",
        "tax",
        "threshold",
        "total",
        "value",
        "withholding",
        "worth",
    }
)
_FORMULA_NUMERIC_RESULT_HEAD_PATTERN = "|".join(
    sorted(_FORMULA_NUMERIC_RESULT_HEADS, key=len, reverse=True)
)
_FORMULA_SUBJECT_PHRASE_BREAK = re.compile(
    r"\b(?:of|for|to|under|by|from|in|on|with|without|that|which|who|whose|"
    r"before|after|during|throughout|within)\b",
    flags=re.IGNORECASE,
)
_FORMULA_SUBJECT_TRAILING_MODIFIERS = frozenset(
    {
        "adjusted",
        "allocable",
        "allowed",
        "allowable",
        "applicable",
        "attributable",
        "available",
        "authorized",
        "calculated",
        "computed",
        "currently",
        "described",
        "derived",
        "determined",
        "due",
        "earned",
        "hereby",
        "imposed",
        "incurred",
        "herein",
        "hereunder",
        "now",
        "otherwise",
        "paid",
        "payable",
        "permitted",
        "properly",
        "prescribed",
        "provided",
        "pursuant",
        "received",
        "reported",
        "responsible",
        "shown",
        "so",
        "specified",
        "set",
        "subject",
        "taxable",
        "forth",
        "then",
        "thereon",
        "previously",
        "below",
        "duly",
        "legally",
        "levied",
        "realized",
        "recognized",
        "assessed",
        "worked",
    }
)
_FORMULA_SUBJECT_LEADING_NONHEAD_WORDS = frozenset(
    {
        "a",
        "adjusted",
        "allowable",
        "an",
        "applicable",
        "earned",
        "federal",
        "gross",
        "imposed",
        "individual",
        "operating",
        "qualified",
        "resident",
        "state",
        "taxable",
        "the",
    }
)
_FORMULA_SUBJECT_PARENTHETICAL = re.compile(
    r"(?:as|adjusted|calculated|computed|described|determined|if|provided|"
    r"specified|subject|when|where)\b",
    flags=re.IGNORECASE,
)
_FORMULA_LEGAL_PROVISION_REFERENCE = (
    r"(?:"
    r"(?:(?:this|the)\s+)?(?:section|subsection|paragraph|chapter|part|title|"
    r"article|subdivision)"
    r"(?:\s+(?:(?:\([^)]+\))+|[A-Z]+|[IVXLCDM]+|\d+(?:\.\d+)*))?|"
    r"R\.?\s*S\.?\s*\d+[A-Za-z]?\s*:\s*\d+(?:\.\d+)*(?:\([^)]+\))*|"
    r"§+\s*\d+[A-Za-z]?\s*:\s*\d+(?:\.\d+)*(?:\([^)]+\))*)"
)
_FORMULA_SUBJECT_PREAMBLE_PREFIX = re.compile(
    rf"(?:"
    rf"(?:for\s+purposes\s+of|pursuant\s+to|under|according\s+to|"
    rf"in\s+accordance\s+with|except\s+under|to\s+the\s+extent\s+provided\s+in)"
    rf"\s+{_FORMULA_LEGAL_PROVISION_REFERENCE}|"
    rf"subject\s+to\s+(?:(?:the\s+)?(?:limitations?|provisions?|requirements?)"
    rf"\s+of\s+)?{_FORMULA_LEGAL_PROVISION_REFERENCE}|"
    rf"(?:except\s+)?as\s+(?:otherwise\s+)?provided\s+(?:in|by)\s+"
    rf"{_FORMULA_LEGAL_PROVISION_REFERENCE}|"
    r"for\s+(?:taxable|tax|calendar|fiscal)\s+years?[^,.;:\n]*?"
    r"(?:18|19|20)\d{2}|"
    r"effective\s+for\s+(?:taxable|tax|calendar|fiscal)\s+years?"
    r"[^,.;:\n]*?(?:18|19|20)\d{2}|"
    r"(?:after|beginning\s+after)\s+(?:18|19|20)\d{2}|"
    r"(?:effective\s+)?on\s+or\s+after\s+[A-Za-z]+\s+\d{1,2}\s*,?\s*"
    r"(?:18|19|20)\d{2}|"
    r"beginning\s+on\s+[A-Za-z]+\s+\d{1,2}\s*,?\s*(?:18|19|20)\d{2}"
    r")(?:\s*,\s*|\s+)",
    flags=re.IGNORECASE,
)
_FORMULA_NUMERIC_OPERAND_HEADS = frozenset(
    {
        "allowance",
        "amount",
        "asset",
        "average",
        "base",
        "basis",
        "benefit",
        "bonus",
        "cap",
        "coefficient",
        "contribution",
        "credit",
        "deduction",
        "difference",
        "dividend",
        "divisor",
        "dollar",
        "denominator",
        "expense",
        "earning",
        "factor",
        "fee",
        "formula",
        "fraction",
        "gain",
        "hour",
        "income",
        "index",
        "interest",
        "liability",
        "limit",
        "loss",
        "margin",
        "mean",
        "median",
        "month",
        "multiplier",
        "numerator",
        "number",
        "offset",
        "payment",
        "percent",
        "percentage",
        "product",
        "proceeds",
        "rate",
        "ratio",
        "receipt",
        "salary",
        "schedule",
        "supplement",
        "surcharge",
        "table",
        "tax",
        "threshold",
        "total",
        "unit",
        "value",
        "voucher",
        "wage",
        "quotient",
        "remainder",
        "sum",
    }
)
_FORMULA_NUMERIC_OPERAND_LITERAL_CORE = (
    rf"(?:(?:[$€£]\s*)?-?\d+(?:,\d{{3}})*(?:\.\d+)?|"
    rf"{_ENGLISH_NUMBER_WORD}"
    rf"(?:(?:[-\s]+(?:and[-\s]+)?){_ENGLISH_NUMBER_WORD})*)"
    r"(?:\s+(?:dollars?|cents?|percent(?:age)?(?:\s+points?)?|hours?|units?))?"
)
_FORMULA_NUMERIC_OPERAND_LITERAL = re.compile(
    rf"\s*(?:the\s+)?(?:\(\s*{_FORMULA_NUMERIC_OPERAND_LITERAL_CORE}\s*\)|"
    rf"{_FORMULA_NUMERIC_OPERAND_LITERAL_CORE})\s*",
    flags=re.IGNORECASE,
)
_FORMULA_ADMINISTRATIVE_PARTICIPLE = frozenset(
    {"administering", "handling", "providing"}
)
_FORMULA_OPERAND_IRREGULAR_POSTMODIFIERS = frozenset(
    {
        "borne",
        "bought",
        "built",
        "caught",
        "dealt",
        "done",
        "drawn",
        "driven",
        "exempt",
        "felt",
        "found",
        "given",
        "grown",
        "held",
        "kept",
        "known",
        "laid",
        "led",
        "left",
        "lost",
        "made",
        "paid",
        "read",
        "run",
        "said",
        "set",
        "shown",
        "sold",
        "spent",
        "taught",
        "told",
        "understood",
        "withheld",
        "won",
        "written",
    }
)
_VALID_ROMAN_OUTLINE_LABEL = re.compile(
    r"(?=[ivxlcdm]+\Z)m{0,3}(?:cm|cd|d?c{0,3})"
    r"(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})",
    flags=re.IGNORECASE,
)
_FORMULA_APPLICABILITY_YEAR = r"(?:18|19|20)\d{2}"
_FORMULA_APPLICABILITY_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|"
    r"nov(?:ember)?|dec(?:ember)?)\.?"
)
_FORMULA_APPLICABILITY_DAY = r"(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?"
_FORMULA_APPLICABILITY_DATE = (
    rf"(?:{_FORMULA_APPLICABILITY_MONTH}\s+"
    rf"(?:{_FORMULA_APPLICABILITY_DAY}\s*,?\s*)?"
    rf"{_FORMULA_APPLICABILITY_YEAR}|{_FORMULA_APPLICABILITY_YEAR})"
)
_FORMULA_APPLICABILITY_BOUNDARY = (
    r"(?:after|before|(?:no|not)\s+(?:earlier|later)\s+than|"
    r"prior\s+to|subsequent\s+to|"
    r"on(?:\s+(?:or|and)\s+(?:after|before))?)"
)
_FORMULA_APPLICABILITY_START = (
    r"(?:beginning|commencing|starting|(?:that|which)\s+"
    r"(?:begin|begins|commence|commences|start|starts))"
)
_FORMULA_APPLICABILITY_RANGE_END = (
    rf"(?:,?\s*(?:(?:and|but)\s+(?:ending\s+)?"
    rf"{_FORMULA_APPLICABILITY_BOUNDARY}|through|to|"
    rf"up\s+to\s+and\s+including)\s+"
    rf"{_FORMULA_APPLICABILITY_DATE})"
)
_FORMULA_APPLICABILITY_YEAR_SPAN = (
    rf"(?>{_FORMULA_APPLICABILITY_YEAR}(?:(?:\s*,\s*"
    rf"(?:(?:and|or)\s+)?|\s+(?:and|or|through|to)\s+|\s*[-–]\s*)"
    rf"{_FORMULA_APPLICABILITY_YEAR})*)"
)
_FORMULA_APPLICABILITY_YEAR_SERIES = (
    rf"(?:from\s+)?{_FORMULA_APPLICABILITY_YEAR_SPAN}"
    rf"(?:\s*,?\s*(?:inclusive|and\s+(?:thereafter|later|onward)))?"
)
_FORMULA_APPLICABILITY_PREFACE = re.compile(
    rf"^\s*(?:(?:\([^)]+\)|[A-Z]\.)\s*)*(?:"
    rf"(?:effective\s+)?(?:for|during)\s+(?:the\s+)?"
    rf"(?:tax(?:able)?|calendar|fiscal|assessment)\s+years?\s+"
    rf"{_FORMULA_APPLICABILITY_START}\s+"
    rf"(?:(?:{_FORMULA_APPLICABILITY_BOUNDARY}|in)\s+)?"
    rf"{_FORMULA_APPLICABILITY_DATE}"
    rf"(?:{_FORMULA_APPLICABILITY_RANGE_END})?\s*,?|(?:"
    rf"(?:effective\s+)?(?:for|during)\s+(?:the\s+)?"
    rf"(?:tax(?:able)?|calendar|fiscal|assessment)\s+years?\s+"
    rf"ending\s+(?:(?:{_FORMULA_APPLICABILITY_BOUNDARY}|in)\s+)?"
    rf"{_FORMULA_APPLICABILITY_DATE}"
    rf"(?:{_FORMULA_APPLICABILITY_RANGE_END})?\s*,?|"
    rf"{_FORMULA_APPLICABILITY_START}\s+"
    rf"(?:{_FORMULA_APPLICABILITY_BOUNDARY}\s+)?"
    rf"{_FORMULA_APPLICABILITY_DATE}"
    rf"(?:,?\s+and\s+thereafter)?|"
    rf"(?:for|during)\s+(?:the\s+)?"
    rf"(?:tax(?:able)?|calendar|fiscal|assessment)\s+years?\s+"
    rf"{_FORMULA_APPLICABILITY_YEAR_SERIES}|"
    rf"(?:for|during)\s+(?:the\s+)?"
    rf"(?:tax(?:able)?|calendar|fiscal|assessment)\s+years?\s+"
    rf"{_FORMULA_APPLICABILITY_BOUNDARY}\s+{_FORMULA_APPLICABILITY_DATE}"
    rf"(?:{_FORMULA_APPLICABILITY_RANGE_END})?|"
    rf"(?:effective(?:\s+(?:on|from))?|starting(?:\s+(?:on|from))?|"
    rf"as\s+of|on\s+or\s+after)\s+{_FORMULA_APPLICABILITY_DATE}"
    rf")(?:(?:\s*,)|(?=\s*[.;:\n]|$)))",
    flags=re.IGNORECASE,
)
_STATED_CONVERSION_CUE = re.compile(
    r"\b(?:umgerechnet|converted)\b",
    flags=re.IGNORECASE,
)
_STATED_CONVERSION_RESULT = re.compile(
    r"\b(?P<verb>"
    r"ergibt\s+sich|ergeben\s+sich|entspricht|entsprechen|beträgt|betragen|"
    r"equals?|results?\s+in|is|are"
    r")\b"
    r"[^.!?\n]{0,80}?"
    r"(?P<value>\d{1,3}(?:[ .]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?)",
    flags=re.IGNORECASE,
)
_STATED_CONVERSION_BASE_VALUE = re.compile(
    r"\d{1,3}(?:[ .]\d{3})+(?:,\d+)?|\d+(?:[.,]\d+)?"
)
_STATED_CONVERSION_GERMAN_MONTH = (
    r"(?:jan(?:uar)?|feb(?:ruar)?|märz|maerz|mrz|apr(?:il)?|mai|"
    r"jun(?:i)?|jul(?:i)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"okt(?:ober)?|nov(?:ember)?|dez(?:ember)?)\.?"
)
_STATED_CONVERSION_ENGLISH_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_STATED_CONVERSION_DATE = re.compile(
    rf"(?:"
    rf"(?<!\w)\d{{1,2}}\.\s*{_STATED_CONVERSION_GERMAN_MONTH}\s+"
    rf"(?:des\s+Jahres\s+)?(?:18|19|20)\d{{2}}|"
    rf"\b{_STATED_CONVERSION_ENGLISH_MONTH}\s+\d{{1,2}}"
    rf"(?:st|nd|rd|th)?\s*,?\s*(?:18|19|20)\d{{2}}|"
    rf"(?<!\w)\d{{1,2}}(?:st|nd|rd|th)?\s+"
    rf"{_STATED_CONVERSION_ENGLISH_MONTH}\s+(?:18|19|20)\d{{2}}|"
    rf"(?<!\w)\d{{1,2}}[./]\d{{1,2}}[./]\d{{2,4}}(?!\w)|"
    rf"(?<!\w)(?:18|19|20)\d{{2}}-\d{{2}}-\d{{2}}(?!\w)"
    rf")",
    flags=re.IGNORECASE,
)
_ROUNDING_LANGUAGE = re.compile(
    r"\b(?:"
    r"abgerundet(?:e|en|er|es)?|abzurunden|aufgerundet(?:e|en|er|es)?|"
    r"aufzurunden|gerundet(?:e|en|er|es)?|zu\s+runden|"
    r"kaufmännisch(?:\s+zu)?\s+runden|"
    r"round(?:ed|ing)?(?:\s+(?:down|up|to\s+the\s+nearest))?"
    r")\b",
    flags=re.IGNORECASE,
)
_DOWN_ROUNDING_LANGUAGE = re.compile(
    r"\b(?:abgerundet(?:e|en|er|es)?|abzurunden|"
    r"round(?:ed|ing)?\s+(?:down|to\s+the\s+nearest\s+lower))\b",
    flags=re.IGNORECASE,
)
_UP_ROUNDING_LANGUAGE = re.compile(
    r"\b(?:aufgerundet(?:e|en|er|es)?|aufzurunden|round(?:ed|ing)?\s+up)\b",
    flags=re.IGNORECASE,
)
_NEAREST_ROUNDING_LANGUAGE = re.compile(
    r"\b(?:gerundet(?:e|en|er|es)?|zu\s+runden|"
    r"kaufmännisch(?:\s+zu)?\s+runden|"
    r"round(?:ed|ing)?\s+to\s+the\s+nearest)\b",
    flags=re.IGNORECASE,
)
_EXCEPTION_LANGUAGE = re.compile(
    r"\b(?:"
    r"except(?:ion)?|unless|subject\s+to|shall\s+not\s+apply|"
    r"does\s+not\s+apply|notwithstanding|vorbehaltlich|ausnahme|"
    r"es\s+sei\s+denn|gilt\b[^.;]{0,80}\bnicht(?:\s*,?\s*wenn)?|"
    r"findet\s+keine\s+anwendung|soweit\s+nicht|"
    r"außer|ausgenommen|abweichend\s+von|jedoch\s+nicht|"
    r"(?:[1-9]\d?)?voraussetzung[^.;]{0,160}\bnicht\b"
    r")",
    flags=re.IGNORECASE,
)
_APPLICABILITY_LANGUAGE = re.compile(
    r"\b(?:"
    r"vorausgesetzt\s*,?\s+dass|"
    r"unter\s+der\s+voraussetzung\s*,?\s+dass|"
    r"wenn|falls|sofern|soweit|when|if|"
    r"bei\s+(?:(?:vorliegen|bestehen)\b|"
    r"(?:(?:einer?|bestehender)\s+)?(?:anspruchsberechtigung|berechtigung)\b)"
    r")\b",
    flags=re.IGNORECASE,
)
_SOURCE_SELECTOR_TOKEN_STOPWORDS = frozenset(
    {
        "applies",
        "apply",
        "case",
        "condition",
        "except",
        "federal",
        "flag",
        "for",
        "has",
        "is",
        "no",
        "non",
        "not",
        "status",
        "the",
        "without",
    }
)
_SOURCE_SELECTOR_GENERIC_ENTITY_TOKENS = frozenset(
    {"alien", "applicant", "household", "individual", "member", "person"}
)
_SOURCE_GENERIC_NUMERIC_NAME_TOKENS = frozenset({"amount", "value"})
_SOURCE_ACRONYM_EXPANSIONS = {
    "ca": ("cash", "assistance"),
    "fpl": ("federal", "poverty", "level"),
    "lpr": ("lawful", "permanent", "resident"),
    "ssn": ("social", "security", "number"),
}
_NEGATIVE_NONAPPLICABILITY_LANGUAGE = re.compile(
    r"\b(?:"
    r"(?:shall|does)\s+not\s+apply|"
    r"(?:is|are)\s+not\s+(?:eligible|qualified|allowed|entitled)|"
    r"findet\s+keine\s+anwendung|"
    r"gilt\b[^.;]{0,80}\bnicht|"
    r"nicht\s+berechtigt|"
    r"kein(?:e|en|em|er|es)?\s+(?:anspruch|berechtigung)"
    r")\b",
    flags=re.IGNORECASE,
)
_LOCAL_CONDITION_LANGUAGE = re.compile(
    r"\b(?:"
    r"if|when|where|unless|except|in\s+the\s+case\s+of|"
    r"wenn|falls|sofern|soweit|bei|ohne|mangels|im\s+falle|"
    r"au(?:ß|ss)er|es\s+sei\s+denn"
    r")\b",
    flags=re.IGNORECASE,
)
_PRECISE_DEFERRAL_DEPENDENCY = re.compile(
    r"(?:"
    r"§{1,2}\s*\d+[a-z]?|"
    r"\b(?:section|subsection|paragraph|absatz|abs\.|satz|nummer|nr\.)"
    r"\s*\d+[a-z]?|"
    r"\b[a-z]{2}(?:-[a-z0-9-]+)?/(?:statute|regulation|guidance|manual)/"
    r"[A-Za-z0-9_./-]+|"
    r"\b[a-z]{2}(?:-[a-z0-9-]+)?:"
    r"(?:statutes|regulations|guidance|manuals)/[A-Za-z0-9_./-]+#[A-Za-z0-9_]+"
    r")",
    flags=re.IGNORECASE,
)
_SAME_ACT_SECTION_DEPENDENCY = re.compile(
    r"\bsection\s+(?P<section>\d+[a-z]?)\s+of\s+(?P<act>this|the)\s+act\b",
    flags=re.IGNORECASE,
)
_USC_DEFERRAL_DEPENDENCY = re.compile(
    r"\b(?P<title>\d+)\s+U\.?\s*S\.?\s*C\.?(?:\s*[,;:])?\s*"
    r"(?:(?:§{1,2}|sections?)\s*)?"
    r"(?P<section>\d+[a-z0-9]*(?:[-\u2010\u2011\u2012\u2013\u2014\u2015"
    r"\u2212\ufe58\ufe63\uff0d]\d+)?)"
    r"(?P<tail>(?:\s*\(\s*[A-Za-z0-9]+\s*\))*)",
    flags=re.IGNORECASE,
)
_REVERSED_USC_DEFERRAL_DEPENDENCY = re.compile(
    r"(?:\bsections?|§{1,2})\s*"
    r"(?P<section>\d+[a-z0-9]*(?:[-\u2010\u2011\u2012\u2013\u2014\u2015"
    r"\u2212\ufe58\ufe63\uff0d]\d+)?)"
    r"(?P<tail>(?:\s*\(\s*[A-Za-z0-9]+\s*\))*)"
    r"\s*,?\s*(?:of|in|as\s+codified\s+in)\s+title\s+(?P<title>\d+)\b",
    flags=re.IGNORECASE,
)
_TITLE_FIRST_USC_DEFERRAL_DEPENDENCY = re.compile(
    r"\btitle\s+(?P<title>\d+)\s*,?\s*(?:sections?|§{1,2})\s*"
    r"(?P<section>\d+[a-z0-9]*(?:[-\u2010\u2011\u2012\u2013\u2014\u2015"
    r"\u2212\ufe58\ufe63\uff0d]\d+)?)"
    r"(?P<tail>(?:\s*\(\s*[A-Za-z0-9]+\s*\))*)",
    flags=re.IGNORECASE,
)
_RELATIVE_USC_DEFERRAL_DEPENDENCY = re.compile(
    r"\bsections?\s+"
    r"(?P<section>\d+[a-z0-9]*(?:[-\u2010\u2011\u2012\u2013\u2014\u2015"
    r"\u2212\ufe58\ufe63\uff0d]\d+)?)"
    r"(?P<tail>(?:\s*\(\s*[A-Za-z0-9]+\s*\))*)",
    flags=re.IGNORECASE,
)
_LOUISIANA_RS_DEFERRAL_DEPENDENCY = re.compile(
    r"(?<![A-Za-z0-9_])(?:(?:La\.?\s*)|(?:LSA\s*[-\u2010-\u2015]?\s*))?"
    r"R\.?\s*S\.?\s*(?:§{1,2}\s*)?"
    r"(?P<title>\d+[A-Za-z]?)\s*:\s*"
    r"(?P<section>\d+[A-Za-z]?(?:(?:[-\u2010\u2011\u2012\u2013\u2014\u2015"
    r"\u2212\ufe58\ufe63\uff0d][A-Za-z0-9]+)|(?:\.\d+[A-Za-z]?))*)"
    r"(?P<tail>(?:\s*\(\s*[A-Za-z0-9]+\s*\))*"
    r"(?:\s*(?:"
    r",\s*(?:and\s+)?(?:(?:clauses?|divisions?|items?|lines?|paragraphs?|parts?|"
    r"schedules?|subclauses?|subdivisions?|subitems?|subparagraphs?|subparts?|"
    r"subsections?)\s+)?|"
    r"(?:and/or|and|or|plus|&|as\s+well\s+as|together\s+with|along\s+with|"
    r"in\s+(?:conjunction|combination|tandem)\s+with|in\s+relation\s+(?:to|with)|"
    r"as\s+supplemented\s+by|"
    r"combined\s+with|in\s+addition\s+to|as\s+amended\s+by|including|through|to|"
    r"followed\s+by|as\s+provided\s+in|read\s+with|read\s+together\s+with|"
    r"subject\s+to|except\s+as\s+provided\s+in|as\s+(?:modified|qualified)\s+by|"
    r"[-\u2013\u2014])\s*"
    r"(?:(?:clauses?|divisions?|items?|lines?|paragraphs?|parts?|schedules?|"
    r"subclauses?|subdivisions?|subitems?|subparagraphs?|subparts?|subsections?)\s+)?"
    r"|(?:clauses?|divisions?|items?|lines?|paragraphs?|parts?|schedules?|"
    r"subclauses?|subdivisions?|subitems?|subparagraphs?|subparts?|subsections?)\s+"
    r")\(\s*[A-Za-z0-9]+(?:[-\u2013][A-Za-z0-9]+)*\s*\)"
    r"(?!\s+(?:filers?\b|of\s+(?:the\s+)?(?:calculation|explanatory\s+report|"
    r"form|report|return|tax\s+return|worksheet)\b)))*)"
    r"(?!\s*\()(?![A-Za-z0-9_])(?![./:]\s*(?:[A-Za-z0-9]|\())"
    r"(?![-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]"
    r"\s*(?:[A-Za-z0-9]|\())",
    flags=re.IGNORECASE,
)
_MISSING_DEPENDENCY_LANGUAGE = re.compile(
    r"\b(?:"
    r"requires?|depends?\s+on|missing|not\s+yet\s+encoded|unavailable|"
    r"cannot\s+be\s+(?:computed|encoded|resolved)|until|without|"
    r"benötigt|abhängig|fehlt|nicht\s+codiert"
    r")\b",
    flags=re.IGNORECASE,
)
_HEBREW_MISSING_DEPENDENCY_LANGUAGE = re.compile(
    _MISSING_DEPENDENCY_LANGUAGE.pattern
    + "|"
    + hebrew_source_text.MISSING_DEPENDENCY_LANGUAGE,
    flags=re.IGNORECASE,
)
_ADVERSATIVE_LANGUAGE = re.compile(
    r"\b(?:"
    r"although|but|despite|even\s+though|except|however|nevertheless|nonetheless|"
    r"notwithstanding|though|unless|whereas|while|yet|aber|jedoch|obwohl"
    r")\b",
    flags=re.IGNORECASE,
)
_HEBREW_ADVERSATIVE_LANGUAGE = hebrew_source_text.AdversativePattern(
    _ADVERSATIVE_LANGUAGE.pattern + "|" + hebrew_source_text.ADVERSATIVE_LANGUAGE,
    flags=re.IGNORECASE,
)
# The English and German negators. The English words are the direct negators
# of the English negation guards (``_source_link_scope_has_unreset_negation``,
# ``_louisiana_assertion_is_negative`` and
# ``_same_act_dependency_state_is_reversed``); the German words are those the
# checker reads (``nicht``, ``kein`` with its forms, ``keinesfalls``) and the
# German for each English one, but ``noch``, which also means "still".
# ``without`` is not one: the deferral route reads it as a missing-dependency
# signal. The Hebrew reader takes these as negating a Hebrew signal or link,
# as the translation is read (``not חסר סעיף 2``, "section 2 is not
# missing"); no English or German reason is read with them.
_NEGATION_LANGUAGE = re.compile(
    r"\b(?:"
    r"not|no|never|neither|nor|nothing|nowhere|cannot|(?:false|untrue)\s+that|"
    r"nicht|nichts|kein(?:e|en|em|er|es)?|keinesfalls|keineswegs|nie|niemals|"
    r"weder|nirgends|(?:falsch|unwahr),?\s+dass"
    r")\b|n['’]t\b",
    flags=re.IGNORECASE,
)
# The English negations, prohibitions and exclusions that negate an ``in
# accordance with`` link when they precede it in its clause: the English link
# guard reads a raw scope with this pattern alone, byte for byte main's
# (``test_english_link_negations_are_mains``), so every text outside the
# Hebrew blocks keeps main's verdict.
_ENGLISH_LINK_NEGATION_LANGUAGE = re.compile(
    r"\b(?:not|never|neither|nor|cannot|nowhere|prohibit\w*|barred|"
    r"forbidden|precluded|prevented|excluded|denied|except)\b|"
    r"\b(?:denies?|rejects?|disclaims?)\b|"
    r"\b(?:false|untrue|impossible)\s+that\b|"
    r"(?:^|[,;:]\s*|\b(?:and|but|or|that|while|whereas)\s+)"
    r"no\s+(?:amount|tax|liability|computation|calculation|"
    r"determination|rate|base|statute|rule|provision|requirement)\b"
    r"(?:\s+[A-Za-z-]+){0,3}\s+(?:is|are|was|were|shall|may|must|will|"
    r"would|can|could|does|do|did|states?|provides?|requires?|"
    r"determines?|calculates?|computes?|applies?|arises?)\b|"
    r"\b(?:ca|do|does|did|is|are|was|were|"
    r"shall|will|would|could|should|must|need)n['’]t\b|"
    r"\b(?:fails?|refuses?|unable)\s+to\b|"
    r"\bwithout\s+(?:being\s+)?(?:determin\w*|comput\w*|calculat\w*|"
    r"appl\w*|impos\w*|aris\w*)\b|"
    r"\b(?:in\s+no\s+(?:case|event)|under\s+no\s+circumstances|"
    r"at\s+no\s+time|by\s+no\s+means|in\s+no\s+way|"
    r"on\s+no\s+account|under\s+no\s+conditions?|"
    r"in\s+no\s+(?:manner|respect))\b|"
    r"\b(?:rather|other)\s+than\b",
    flags=re.IGNORECASE,
)
# The negations, prohibitions and exclusions that negate a link in text that
# holds Hebrew: the English link guard reads them only in the cross-language
# copy (``hebrew_source_text.cross_language_text``), which exists only for a
# scope inside the Hebrew folding gate, and the Hebrew reader takes them, with
# ``_NEGATION_LANGUAGE``, as negating a Hebrew source link (``(א) prohibited
# מחושב בהתאם לסעיף 2``), as the translation is read.
#
# The vocabulary is symmetric across English, German and Hebrew (#1882). The
# English lines come first, unchanged. Then come the German words for each of
# them: ``nicht`` for "not", ``ohne`` for "without", ``außer`` for "except",
# ``verboten`` for "prohibited", ``anders als`` for "other than", and so on.
# ``without`` and ``ohne`` exclude a computation word of English or German.
# Before a Hebrew word they exclude whatever follows, as Hebrew ``בלי`` does
# (``משולם without חישוב בהתאם לסעיף 2``, "paid without calculation in
# accordance with section 2").
#
# The wall: a German word is also an English surname (``Tax is computed for
# Nie in accordance with section 2``), so no text without a Hebrew-block code
# point in its reason or source reaches this pattern. A raw scope is read
# with ``_ENGLISH_LINK_NEGATION_LANGUAGE`` alone, and the reason reader takes
# this pattern only once it has a Hebrew citation, so every English and German
# verdict and issue string stays main's
# (tests/test_source_completeness_non_hebrew_wall.py).
_SOURCE_LINK_NEGATION_LANGUAGE = re.compile(
    r"\b(?:not|never|neither|nor|cannot|nowhere|prohibit\w*|barred|"
    r"forbidden|precluded|prevented|excluded|denied|except)\b|"
    r"\b(?:denies?|rejects?|disclaims?)\b|"
    r"\b(?:false|untrue|impossible)\s+that\b|"
    r"(?:^|[,;:]\s*|\b(?:and|but|or|that|while|whereas)\s+)"
    r"no\s+(?:amount|tax|liability|computation|calculation|"
    r"determination|rate|base|statute|rule|provision|requirement)\b"
    r"(?:\s+[A-Za-z-]+){0,3}\s+(?:is|are|was|were|shall|may|must|will|"
    r"would|can|could|does|do|did|states?|provides?|requires?|"
    r"determines?|calculates?|computes?|applies?|arises?)\b|"
    r"\b(?:ca|do|does|did|is|are|was|were|"
    r"shall|will|would|could|should|must|need)n['’]t\b|"
    r"\b(?:fails?|refuses?|unable)\s+to\b|"
    r"\bwithout\s+(?:being\s+)?(?:determin\w*|comput\w*|calculat\w*|"
    r"appl\w*|impos\w*|aris\w*)\b|"
    r"\b(?:in\s+no\s+(?:case|event)|under\s+no\s+circumstances|"
    r"at\s+no\s+time|by\s+no\s+means|in\s+no\s+way|"
    r"on\s+no\s+account|under\s+no\s+conditions?|"
    r"in\s+no\s+(?:manner|respect))\b|"
    r"\b(?:rather|other)\s+than\b|"
    r"\b(?:nicht|nie|niemals|weder|nirgends|nirgendwo|kein(?:e|en|em|er|es)?|"
    r"keinesfalls|keineswegs|au(?:ß|ss)er|ausgenommen|abgesehen\s+von|"
    r"verbiet\w*|verboten\w*|untersag\w*|verwehrt\w*|ausgeschlossen\w*|"
    r"verhindert\w*|verweigert\w*|abgelehnt\w*|bestreit\w*|verneint\w*|"
    r"zurückgewiesen\w*|versäumt\w*|unterl(?:ä|a)sst|unterlassen\w*|weigert\w*|"
    r"au(?:ß|ss)erstande|(?:an)?statt|anders\s+als|"
    r"(?:falsch|unwahr|unmöglich),?\s+dass)\b|"
    r"\b(?:without|ohne)\s+(?:being\s+|zu\s+)?(?:"
    r"determin\w*|comput\w*|calculat\w*|appl\w*|impos\w*|aris\w*|"
    r"bestimm\w*|ermitt?el\w*|ermittl\w*|festgesetzt\w*|festsetz\w*|"
    r"festzusetzen|festgestellt\w*|feststell\w*|festzustellen|berechn\w*|"
    r"errechn\w*|anwend\w*|angewendet\w*|angewandt\w*|anzuwenden|erheb\w*|"
    r"erhoben\w*|auferleg\w*|aufzuerlegen|entsteh\w*|entstanden\w*)\b|"
    r"\b(?:without|ohne)\s+(?:being\s+|zu\s+)?(?=[א-ת])",
    flags=re.IGNORECASE,
)
_CONTEXTUAL_AUTHORITY_LANGUAGE = re.compile(
    r"\b(?:"
    r"(?:included|mentioned|quoted|summarized|discussed|provided|supplied|known)\s+"
    r"only\s+(?:as|for)\s+(?:an?\s+|the\s+)?"
    r"(?:background|comparison|context|historical|history|illustration|"
    r"legislative\s+history|nonbinding\s+authority|orientation)|"
    r"(?:historical|legislative)\s+(?:background|context|history)|"
    r"nonbinding\s+authority|for\s+(?:context|orientation)|"
    r"(?:merely|only)\s+(?:illustrative|nonbinding)"
    r")\b",
    flags=re.IGNORECASE,
)
_COORDINATED_FINITE_CLAUSE = re.compile(
    r"(?:,\s*)?\b(?:and|but|or|yet)\s+"
    r"(?:the\s+|a\s+|an\s+|this\s+|that\s+|these\s+|those\s+)?"
    r"[a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*){0,8}\s+"
    r"(?:can|could|is|are|was|were|has|have|had|does|do|did|may|might|must|"
    r"shall|should|will|would)\b",
    flags=re.IGNORECASE,
)
_DEPENDENCY_CONTEXT_COORDINATION = re.compile(
    r"(?:,\s*)?\b(?:although|and|but|even\s+though|or|though|while|whereas|yet)\s+",
    flags=re.IGNORECASE,
)


def _has_adversative_language(text: str) -> bool:
    without_not_yet = re.sub(
        r"\bnot\s+yet\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return bool(_adversative_language().search(without_not_yet))


_DEPENDENCY_SUBJECT_TERMS = frozenset(
    {
        "amount",
        "assistance",
        "benefit",
        "calendar",
        "calculation",
        "classification",
        "condition",
        "criteria",
        "data",
        "date",
        "designation",
        "determination",
        "document",
        "eligibility",
        "fact",
        "filing",
        "form",
        "guidance",
        "income",
        "information",
        "input",
        "limit",
        "management",
        "notice",
        "plan",
        "policy",
        "payment",
        "participation",
        "process",
        "processing",
        "procedure",
        "program",
        "administration",
        "rate",
        "receipt",
        "record",
        "regulation",
        "retention",
        "requirement",
        "rule",
        "size",
        "standard",
        "status",
        "submission",
        "threshold",
        "timing",
        "verification",
        "workflow",
    }
)
_DEPENDENCY_MODIFIER_TERMS = _DEPENDENCY_SUBJECT_TERMS | {
    "a",
    "an",
    "administrative",
    "adjusted",
    "and",
    "agency",
    "annual",
    "applicable",
    "asset",
    "background-check",
    "failing-score",
    "federal",
    "fiscal",
    "fiscal-year",
    "gross",
    "historical",
    "household",
    "initial",
    "legal",
    "local",
    "of",
    "or",
    "plan",
    "plan-submission",
    "program",
    "public",
    "public-housing",
    "residency",
    "state",
    "tax",
    "taxable",
    "the",
    "troubled",
    "troubled-agency",
    "year",
}
_DEPENDENCY_OBJECT_MODIFIER_TERMS = frozenset(
    {
        "administered",
        "applied",
        "approved",
        "available",
        "authorized",
        "awarded",
        "calculated",
        "cited",
        "defined",
        "described",
        "determined",
        "eligible",
        "senior",
        "established",
        "furnished",
        "issued",
        "implemented",
        "obtained",
        "paid",
        "payable",
        "promulgated",
        "provided",
        "received",
        "referenced",
        "required",
        "set",
        "specified",
        "supplied",
        "verified",
    }
)
_DEPENDENCY_OBJECT_MODIFIER_ADVERBS = frozenset(
    {
        "already",
        "also",
        "not",
        "now",
        "otherwise",
        "still",
        "then",
        "yet",
    }
)
_DEPENDENCY_DETERMINER_TERMS = frozenset(
    {
        "a",
        "all",
        "an",
        "another",
        "any",
        "both",
        "each",
        "either",
        "every",
        "few",
        "her",
        "his",
        "its",
        "many",
        "more",
        "most",
        "my",
        "neither",
        "no",
        "our",
        "one",
        "several",
        "some",
        "that",
        "the",
        "their",
        "these",
        "this",
        "those",
        "your",
    }
)
_DEPENDENCY_ACTOR_AMBIGUOUS_OBJECT_TERMS = frozenset(
    {"benefit", "limit", "process", "program", "rate", "record", "rule"}
)
_DEPENDENCY_ACTOR_ALWAYS_FINITE_PREDICATE_TERMS = frozenset(
    {"based", "condition", "limit", "process", "rate"}
)
_DEPENDENCY_IRREGULAR_PLURAL_TERMS = frozenset({"criteria", "data"})
_DEPENDENCY_NUMBER_AMBIGUOUS_TERMS = frozenset({"data"})
_DEPENDENCY_ZERO_MARKED_FINITE_VERB_TERMS = frozenset(
    {"cost", "cut", "hit", "input", "put", "read", "set", "spread"}
)
_DEPENDENCY_ACTOR_NOMINAL_PREFIX_TERMS = frozenset(
    {
        "amendment",
        "benefit",
        "blind",
        "cash",
        "child",
        "drug",
        "earning",
        "elderly",
        "family",
        "household",
        "immigrant",
        "medical",
        "noncitizen",
        "parent",
        "payment",
        "policy",
        "premium",
        "prescription",
        "program",
        "record",
        "refugee",
        "resource",
        "rule",
        "saving",
        "school",
        "senior",
        "shelter",
        "spenddown",
        "student",
        "utility",
        "wage",
        "waiver",
        "work",
    }
)
_DEPENDENCY_ACTOR_NOMINAL_HEAD_TERMS = frozenset(
    {
        "administration",
        "criteria",
        "eligibility",
        "limit",
        "policy",
        "procedure",
        "program",
        "requirement",
        "rule",
        "standard",
        "threshold",
    }
)
_DEPENDENCY_POLICY_ACRONYM_TERMS = frozenset(
    {
        "abawd",
        "bbce",
        "c",
        "chip",
        "d",
        "dsh",
        "ebt",
        "epsdt",
        "esrd",
        "fmap",
        "fpl",
        "hcbs",
        "hea",
        "hipp",
        "hmo",
        "i",
        "irmaa",
        "ism",
        "lis",
        "lieap",
        "liheap",
        "ltss",
        "ma",
        "magi",
        "mco",
        "moe",
        "msp",
        "pace",
        "pass",
        "pd",
        "pdp",
        "poms",
        "ppo",
        "qdwi",
        "qi",
        "qmb",
        "rsdi",
        "sga",
        "snp",
        "spa",
        "slmb",
        "ssdi",
        "ssp",
        "sua",
        "tfp",
        "tpl",
        "wpr",
    }
)
_DEPENDENCY_POLICY_PROGRAM_SUBJECT_TERMS = frozenset(
    {
        "afdc",
        "chip",
        "liheap",
        "medicaid",
        "medicare",
        "rsdi",
        "snap",
        "ssdi",
        "ssi",
        "tanf",
        "wic",
    }
)
_DEPENDENCY_POLICY_COMPOUND_CONNECTOR_TERMS = frozenset(
    {"based", "non", "part", "waiver"}
)
_DEPENDENCY_POLICY_LEADING_ONLY_TERMS = frozenset(
    {"based", "chip", "pace", "part", "pass"}
)
_DEPENDENCY_COMPOUND_NOMINAL_TERMS = frozenset(
    {
        "administration",
        "calculation",
        "management",
        "participation",
        "payment",
        "processing",
        "retention",
        "verification",
    }
)
_DEPENDENCY_NESTED_NOMINAL_COMPOUNDS = frozenset(
    {
        ("records", "management"),
        ("records", "retention"),
        ("rules", "administration"),
    }
)
_LEGAL_ACTOR_HEAD_TERMS = frozenset(
    {
        "administrator",
        "agency",
        "authority",
        "board",
        "bureau",
        "commission",
        "commissioner",
        "corporation",
        "department",
        "judiciary",
        "office",
        "reserve",
        "secretary",
        "service",
    }
)
_LEGAL_ACTOR_ACRONYMS = frozenset(
    {
        "acf",
        "acl",
        "aphis",
        "atf",
        "bjs",
        "bls",
        "bop",
        "cdc",
        "cfpb",
        "cftc",
        "cia",
        "cms",
        "cpsc",
        "csb",
        "dea",
        "dhs",
        "dod",
        "doe",
        "doj",
        "dol",
        "dot",
        "eeoc",
        "ebsa",
        "epa",
        "faa",
        "fbi",
        "fcc",
        "fda",
        "fdic",
        "fec",
        "fema",
        "fha",
        "fhfa",
        "fincen",
        "fmc",
        "fmcs",
        "fns",
        "fra",
        "ftc",
        "gao",
        "gsa",
        "hhs",
        "hrsa",
        "hud",
        "ice",
        "irs",
        "nara",
        "nasa",
        "ncua",
        "nih",
        "nlrb",
        "nrc",
        "nsa",
        "ntsb",
        "occ",
        "omb",
        "opm",
        "osc",
        "osha",
        "pbgc",
        "sba",
        "samhsa",
        "sec",
        "ssa",
        "tva",
        "tsa",
        "usaid",
        "uscis",
        "usda",
        "usps",
        "usss",
        "ustr",
        "va",
    }
)
_LEGAL_INSTRUMENT_TERMS = frozenset(
    {
        "act",
        "administrative",
        "affordable",
        "assessment",
        "assistance",
        "benefit",
        "benefits",
        "care",
        "choice",
        "code",
        "education",
        "energy",
        "families",
        "fair",
        "federal",
        "food",
        "housing",
        "home",
        "internal",
        "labor",
        "low-income",
        "management",
        "needy",
        "nutrition",
        "patient",
        "procedure",
        "program",
        "protection",
        "public",
        "regulation",
        "regulations",
        "revenue",
        "section",
        "security",
        "social",
        "standards",
        "statute",
        "supplemental",
        "temporary",
        "veterans",
        "voucher",
    }
)
_DEPENDENCY_STATE_VALUE = (
    r"(?:approved|ascertained|available|calculated|computed|determined|encoded|"
    r"established|furnished|implemented|known|made\s+available|missing|"
    r"issued|needed|obtained|produced|provided|received|required|resolved|set|"
    r"supplied|unavailable|verified)"
)
_EXECUTABLE_DEPENDENCY_READY_STATE_VALUE = (
    r"(?:approved|ascertained|available|calculated|computed|determined|encoded|"
    r"established|furnished|implemented|known|made\s+available|obtained|produced|"
    r"provided|received|resolved|set|supplied|verified)"
)
_SOURCE_BOUND_RUNTIME_GAP_LANGUAGE = re.compile(
    r"\b(?:"
    r"administrative|calendar|capabilit(?:y|ies)|classification|complaint|"
    r"content|data|date\s+arithmetic|designation|determination|document|event|"
    r"fact|filing|form|hearing|information|input|investigation|membership|"
    r"notice|process|procedure|record|relation|representation|status|submission|"
    r"timing|workflow"
    r")[a-z-]*\b",
    flags=re.IGNORECASE,
)
_HEBREW_SOURCE_BOUND_RUNTIME_GAP_LANGUAGE = re.compile(
    _SOURCE_BOUND_RUNTIME_GAP_LANGUAGE.pattern
    + "|"
    + hebrew_source_text.RUNTIME_GAP_LANGUAGE,
    flags=re.IGNORECASE,
)
_ADMINISTRATIVE_SOURCE_ARTIFACT_LANGUAGE = re.compile(
    r"\b(?:"
    r"amendment|approval|audit|certification|comment|complaint|consultation|"
    r"copy|document|filing|hearing|inspection|notice|plan|procedure|"
    r"recommendation|record|report|return|submission|waiver"
    r")[a-z-]*\b",
    flags=re.IGNORECASE,
)
_ADMINISTRATIVE_SOURCE_ACTION_LANGUAGE = re.compile(
    r"\b(?:"
    r"adopt|approve|certif|consult|contain|conduct|disapprov|enforce|establish|"
    r"file|investigat|notify|prescrib|publish|recommend|review|submit|waiv"
    r")[a-z-]*\b",
    flags=re.IGNORECASE,
)
_SOURCE_BOUND_RUNTIME_GAP_STOPWORDS = frozenset(
    {
        "administrative",
        "available",
        "because",
        "cannot",
        "capability",
        "capabilities",
        "computed",
        "current",
        "encoded",
        "encoding",
        "exact",
        "input",
        "inputs",
        "missing",
        "required",
        "requires",
        "representation",
        "representations",
        "runtime",
        "section",
        "source",
        "statute",
        "under",
        "unavailable",
        "until",
    }
)
_IMPRECISE_DEFERRAL_RETRY_SHAPE = """\
Required shape (adapt the output and cited dependency to the omitted branch):
module:
  deferred_outputs:
    - output: de:statutes/estg/32a/6#surviving_spouse_splitting_tax
      reason: Cannot be computed until the joint-assessment conditions cited in EStG § 26 are encoded.
`output` and `reason` are required; `blocked_by` is optional and, when present, \
must list exact absolute upstream RuleSpec outputs. When no external legal dependency \
exists, cite the exact current source branch and name its concrete source-stated missing \
input or runtime capability in the `reason` itself; the output path is not a source \
citation, and a generic claim that the branch is unavailable is invalid. When the \
authoritative branch is a bare `Repealed.` tombstone or a finite Louisiana \
`Repealed by Acts ...` session-law tombstone, cite that exact current source branch and \
affirmatively state the repeal in `reason`; do not invent a missing dependency or \
executable rule. Use the bounded form `<exact branch citation> is repealed.`, optionally \
adding the exact authenticated history either as `is repealed by <Acts citation>.` or \
as `is repealed. <Acts citation>.`, and/or adding `and supplies no operative rule`. \
For Louisiana delegated rates, use `no executable RuleSpec output \
for those <source-stated modifiers> rates is supplied in the available context`; do \
not repeat the dependency in that missing object or append a relative clause."""


def _imprecise_deferral_retry_shape(
    *,
    corpus_citation_path: str,
    path: tuple[str, ...],
    reason: str = "",
    hebrew_script: bool,
) -> str:
    """Render branch-specific retry guidance without inventing source facts.

    ``hebrew_script`` is the wall's gate (`_hebrew_script`) narrowed to a
    deferral whose reason or source holds a Hebrew-block code point. Only then
    does the guidance add a native citation or the Hebrew branch-letter
    scheme, so a candidate outside the gate reads main's guidance byte for
    byte, whatever its corpus path (#1882).
    """

    branch_hint = ""
    if path and corpus_citation_path.startswith("us/statute/"):
        with contextlib.suppress(ValueError):
            citation = parse_usc_citation(corpus_citation_path)
            section = normalize_rulespec_path_segment(citation.section)
            fragments = "".join(
                f"({normalize_rulespec_path_segment(part)})"
                for part in (*citation.fragments, *path)
            )
            branch_hint = (
                "\nFor this rejected current-source branch, the literal citation "
                f"required in `reason` is `{citation.title} U.S.C. "
                f"{section}{fragments}`."
            )
    elif (
        path
        and reason
        and _reason_cites_exact_current_statute_branch(
            reason,
            corpus_citation_path=corpus_citation_path,
            path=path,
            strict_terminal=True,
        )
    ):
        branch_hint = (
            "\nThe current-source branch citation in `reason` is already "
            "recognized; correct the bounded dependency, runtime-gap, or repeal "
            "tail without replacing that authenticated citation."
        )
    elif path:
        citation_parts = [
            part for part in corpus_citation_path.strip("/").split("/") if part
        ]
        # `_reason_cites_exact_current_statute_branch` has no CFR-style reader;
        # a regulation branch binds through its literal corpus path, e.g.
        # `us/regulation/42/457/622(a)`, not `42 CFR 457.622(a)`.
        if len(citation_parts) >= 3 and citation_parts[1] in {"statute", "regulation"}:
            fragments = "".join(
                f"({normalize_rulespec_path_segment(part)})" for part in path
            )
            native_citation = (
                hebrew_source_text.native_branch_citation(
                    corpus_citation_path, tuple(path)
                )
                if _hebrew_script() and hebrew_script
                else None
            )
            native_alternative = (
                f" or the native citation `{native_citation}`"
                if native_citation is not None
                else ""
            )
            branch_hint = (
                "\nFor this rejected current-source branch, the literal canonical "
                f"citation required in `reason` is "
                f"`{corpus_citation_path.rstrip('/')}{fragments}`"
                f"{native_alternative} when the reason "
                "cites this branch itself (a repeal, a runtime gap, or a "
                "source-stated input); a reason naming an exact external missing "
                "dependency need not cite it."
            )
    if (
        _hebrew_script()
        and hebrew_script
        and hebrew_source_text.is_hebrew_corpus_path(corpus_citation_path)
    ):
        branch_hint += (
            "\nHebrew branch letters map to output-path segments by gematria "
            "value, not sound: (א)→a, (ב)→b, (ג)→c, (ד)→d, (ה)→e, (ו)→f, (ז)→g, "
            "(ח)→h, (ט)→i, (י)→j, (יא)→k; a numeric suffix is kept, so (ה1)→e1 "
            "and `סעיף 14(ט)(1)` anchors at `…/section-14/i/1`. A Hebrew reason "
            "names an external missing dependency as another section that the "
            "deferred branch itself applies (`לפי סעיף <N>`), of the same law, "
            "for example `חסרה הוראת החישוב שבסעיף <N>`; missing facts of the "
            "branch's own computation are local inputs, not deferral grounds, "
            "and a Hebrew reason cannot defer a branch as a runtime gap."
        )
    return f"{_IMPRECISE_DEFERRAL_RETRY_SHAPE}{branch_hint}"


_ABSATZ_REFERENCE = re.compile(
    r"\b(?:Absatz(?:es)?|Absätze(?:n)?|Abs\.)\s*(?P<label>\d+[a-z]?)\b",
    flags=re.IGNORECASE,
)
_SATZ_REFERENCE = re.compile(
    r"\b(?:Satz(?:es)?|Sätze)\s*(?P<label>\d+)\b",
    flags=re.IGNORECASE,
)
_NUMMER_REFERENCE = re.compile(
    r"\b(?:Nummer|Nr\.)\s*(?P<label>\d+[a-z]?)\b",
    flags=re.IGNORECASE,
)
_BUCHSTABE_REFERENCE = re.compile(
    r"\b(?:Buchstabe|Buchst\.)\s*(?P<label>[a-z])\b",
    flags=re.IGNORECASE,
)
_GERMAN_LEGAL_CITATION = re.compile(
    r"§{1,2}\s*\d+(?:\.\d+)*[a-z]?"
    r"(?:\s*,\s*\d+(?:\.\d+)*[a-z]?)*"
    r"(?:\s*(?:und|bis|[-–—])\s*\d+(?:\.\d+)*[a-z]?)*",
    flags=re.IGNORECASE,
)
_EU_REGULATION_NUMERIC_RECALL_CITATION = re.compile(
    r"\bVerordnung(?:en)?\s+\((?:EU|EG|EWG)\)\s+"
    r"(?:Nr\.\s*)?\d{1,5}/\d{2,5}(?![\w/]|[.,]\d)"
    r"(?:\s*(?:,\s*|und\s+)\((?:EU|EG|EWG)\)\s+"
    r"(?:Nr\.\s*)?\d{1,5}/\d{2,5}(?![\w/]|[.,]\d))*",
    flags=re.IGNORECASE,
)
# Match a complete, court-identified bibliographic parenthesis only. A bare
# slash-separated number or partial docket may still be actual arithmetic.
_BFH_DECISION_NUMERIC_RECALL_CITATION = re.compile(
    r"\(\s*BFH\s+vom\s+\d{1,2}\.\d{1,2}\.\d{4},\s*"
    r"[IVX]+\s+[RB]\s+\d{1,5}/\d{2,4},\s*"
    r"BStBl\s+(?:\d{4}\s+)?II\s+S\.\s*\d+\s*\)",
)
_GERMAN_GAZETTE_NUMERIC_RECALL_CITATION = re.compile(
    r"\(\s*(?:"
    r"ABl\.\s*[LC]\s+\d+\s+vom\s+\d{1,2}\.\d{1,2}\.\d{4},\s*"
    r"|GMBl\.?\s+(?:\d{4}\s*,?\s*)?"
    r")S\.\s*\d+(?:\s*[-–]\s*\d+)?\s*\)",
    flags=re.IGNORECASE,
)
_EXPLICIT_LEGAL_SECTION_REFERENCE = re.compile(
    r"(?:§{1,2}\s*|\b(?:sections?|paragra(?:f|phs?))\s+)"
    r"(?P<section>\d+[a-z]?)",
    flags=re.IGNORECASE,
)
_ENGLISH_LEGAL_CITATION = re.compile(
    r"\b(?:articles?|sections?|secs?\.?|regulations?|paragraphs?)\s+"
    r"\d+(?:\.\d+)*(?:\s*(?:through|to|[-–—]|and|,)\s*\d+(?:\.\d+)*)*",
    flags=re.IGNORECASE,
)
# A complete parenthesis containing only a Statutes at Large volume/page
# reference is bibliographic metadata. Do not accept dollar signs, ranges,
# additional text, or a citation split across lines.
_STATUTES_AT_LARGE_NUMERIC_RECALL_CITATION = re.compile(
    r"\([ \t]*[1-9]\d{0,3}[ \t]+Stat\.[ \t]+[1-9]\d{0,5}[ \t]*\)",
    flags=re.IGNORECASE,
)
_US_CORPUS_CITATION_PATH = re.compile(r"us(?:-[a-z0-9]+)?/", flags=re.IGNORECASE)
# Some US statutes abbreviate a list of U.S. Code references after the first
# title number: `[42 U.S.C. 301 et seq., 401 et seq., ...]`. The continuation
# numbers locate other statutes; they are not operative amounts to encode.
# Require the bracket, U.S.C. marker, and repeated `et seq.` syntax so a nearby
# substantive number remains subject to numeric recall.
_BRACKETED_USC_ET_SEQ_CITATION_LIST = re.compile(
    r"\[\s*\d+\s+U\.?\s*S\.?\s*C\.?\s+\d+[a-z]?\s+et\s+seq\."
    r"(?:\s*,\s*\d+[a-z]?\s+et\s+seq\.)+\s*\]",
    flags=re.IGNORECASE,
)
# US state eligibility manuals cite administrative-code sections such as
# `WAC 388-450-0015` and sibling manual policies such as `policy 770-2`. Both
# are locators. A reference never crosses a line, never stops inside a dotted
# or comma-grouped number (`Policy 24.01` is a structural label, `policy
# 6,740` a value), and takes its whole hyphen chain (`POLICY 05-1-2024`). List
# continuations keep the identifier's hyphenated shape and never take a range
# that a unit word follows, so `policy 254, 3 members`, `policy 254, 10-15
# days` and `policy 10-day notice` keep their values.
_RECALL_UNIT_FOLLOWS = (
    rf"[ \t]*(?:%|-?(?:[A-Za-z]+[ \t-]+){{0,2}}{_RECALL_UNIT_WORD}\b)"
)
# Scan labels first, then prove their reference boundary. Arbitrary connectives
# and quantity words cannot prove that a bare coordinated number is a locator.
_NUMERIC_RECALL_IDENTIFIER_TRANSLATION = str.maketrans(
    {
        "\u00a0": " ",
        "\u202f": " ",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "−": "-",
        "–": "-",
        "—": "-",
    }
)
# Keep locator readers ASCII: Unicode case folding must not prove unit suffixes.
_ENGLISH_SUBDIVISION_NUMERIC_RECALL_LOCATOR = (
    r"(?<![A-Za-z])(?:[A-Za-z][ \t]*)?\d+[A-Za-z]?"
    r"(?:[ \t]+[A-Za-z]\b)?(?:\([A-Za-z]\))*"
    r"(?!\w|[.,]\d)"
)
_ENGLISH_SUBDIVISION_LOCATOR_TOKEN = re.compile(
    _ENGLISH_SUBDIVISION_NUMERIC_RECALL_LOCATOR, flags=re.ASCII
)
_ENGLISH_SUBDIVISION_NUMERIC_RECALL_CITATION = re.compile(
    r"\b(?P<keyword>subdivisions?)[ \t]+(?P<labels>"
    + _ENGLISH_SUBDIVISION_NUMERIC_RECALL_LOCATOR
    + r"(?:[ \t]*(?:,[ \t]*(?:(?:and|or)[ \t]+)?|"
    r"(?:and|or|through|to)[ \t]+|[-–—][ \t]*)"
    + _ENGLISH_SUBDIVISION_NUMERIC_RECALL_LOCATOR
    + r")*)",
    flags=re.IGNORECASE | re.ASCII,
)
_ENGLISH_SUBDIVISION_STATUTORY_TAIL = re.compile(
    r"[ \t]+of[ \t]+(?:(?:this|the|that)[ \t]+"
    r"(?:section|subsection|paragraph|subdivision|act|code|chapter)\b|"
    r"§[ \t]*\d+(?:[.-]\d+)*(?:\([A-Za-z0-9]+\))*)"
    r"(?=[ \t]*(?:$|[,.;:](?!\d)|"
    r"for[ \t]+(?:the[ \t]+same[ \t]+taxable[ \t]+year|taxpayers)\b))",
    flags=re.IGNORECASE,
)
_ENGLISH_SUBDIVISION_REFERENCE_BOUNDARY = re.compile(
    r"[ \t]*(?:$|[.;:](?!\d)|"
    r"(?:for[ \t]+the[ \t]+same[ \t]+taxable[ \t]+year|hereof|herein)"
    r"(?=[ \t]*(?:$|[,.;:](?!\d)))|"
    r",[ \t]*(?:claim\b|use\b|the[ \t]+(?:credit|tax|deduction|subtraction)\b|"
    r"the[ \t]+following[ \t]+applies[ \t]*:|"
    r'(?:"[A-Za-z][A-Za-z -]*"|“[A-Za-z][A-Za-z -]*”)[ \t]+'
    r"(?:means?|includes?|refers?[ \t]+to)\b))",
    flags=re.IGNORECASE,
)
_ENGLISH_SUBDIVISION_QUANTITY_CLAUSE_BOUNDARY = re.compile(
    # A comma alone cannot separate a locator from a shared quantity list.
    # Prove a complete, independent quantity subject and finite predicate.
    # Coordination stays ambiguous; a local comparison or range is one subject.
    r"[ \t]*,[ \t]+\d+(?:\.\d+)?"
    r"(?:[ \t]*(?:[-–—]|to\b|through\b)[ \t]*\d+(?:\.\d+)?)?"
    r"(?:[ \t]+or[ \t]+(?:more|fewer|less|greater)\b)?[ \t]+"
    # Bound positive proof work; longer or ambiguous clauses stay demanded.
    r"(?:(?!(?:and|or)\b)[\w'’%-]+[ \t]+){0,32}"
    r"(?:qualif(?:y|ies)|(?:must|shall|may|can|is|are)[ \t]+"
    r"(?:(?!(?:and|or)\b)[A-Za-z][\w'’-]*[ \t]+){0,32}"
    r"(?!(?:and|or)\b)[A-Za-z][\w'’-]*)"
    r"(?=[ \t]*(?:$|[.;:](?!\d)))",
    flags=re.IGNORECASE,
)
_ENGLISH_SUBDIVISION_SINGLE_REFERENCE_PREDICATE = re.compile(
    r"[ \t]+(?:(?:is|are)[ \t]+(?:required|applicable|authorized)|"
    r"appl(?:y|ies)|provides?|authorizes?|governs?|requires?)\b(?!-)",
    flags=re.IGNORECASE,
)
# The exact phrase can identify poverty-guideline geography. The span helper
# also requires bare or definite reference syntax, retaining unknown units.
_CONTIGUOUS_STATES_NUMERIC_RECALL_IDENTIFIER = re.compile(
    r"(?<![\w.,])48(?=[ \t]+contiguous[ \t]+states\b)",
    flags=re.IGNORECASE,
)
# A table column label that names a jurisdiction set, such as USDA's
# `Household Size 48 States, District of Columbia, Guam, Virgin Islands`.
# The capitalized `States` must be followed directly by a listed jurisdiction
# (District of Columbia, D.C., DC, Guam or Virgin Islands); `48 States` alone,
# `48 states` and `48 months` keep their demand. The span helper also requires
# the column caption `Household Size` directly before it and the table's first
# row (`1 $1,305`) after the column headers: in prose, even `The 48 States and
# DC must approve ...` can state an operative count, and a row such as
# `Maximum Household Size 48 States, ...` is not a header.
_JURISDICTION_SET_LABEL_NUMERIC_RECALL_IDENTIFIER = re.compile(
    r"(?<![\w.,])48(?= States[ \t]*(?:,|&|and\b)[ \t]*(?:the[ \t]+)?"
    r"(?:District[ \t]+of[ \t]+Columbia\b|D\.[ \t]?C\.|DC\b|Guam\b|"
    r"Virgin[ \t]+Islands\b))"
)
_JURISDICTION_SET_LABEL_CAPTION = re.compile(
    r"(?<![\w$])Household[ \t]+Size\Z", re.IGNORECASE
)
# A qualifier before the caption (`Maximum Household Size 48 ...`, across any
# whitespace) makes it a row label with an operative value.
_JURISDICTION_SET_LABEL_CAPTION_QUALIFIER = re.compile(
    r"\b(?:max(?:imum)?|min(?:imum)?|largest|smallest|total|average|eligible|"
    r"limit|up[ \t]+to)\W*\Z",
    re.IGNORECASE,
)
# Only column-header words (no digit, currency sign or second `Household`
# caption, at most 200 characters) may separate the label from row
# `1 $<amount>`.
_JURISDICTION_SET_LABEL_FIRST_ROW = re.compile(
    r"(?:(?!household)[^\d$]){0,200}?(?<!\S)1[ \t]+\$[ \t]*\d", re.IGNORECASE
)
_US_MANUAL_CROSS_REFERENCE_NUMERIC_RECALL_CITATION = re.compile(
    r"\bWAC[ \t]+\d+-\d+-\d+[A-Za-z]?"
    r"(?:[ \t]*(?:,|and|or|through|to)[ \t]*\d+-\d+-\d+[A-Za-z]?)*"
    r"|\bpolic(?:y|ies)[ \t]+\d+(?:-\d+)*(?![.,-]?\d)"
    rf"(?!{_RECALL_UNIT_FOLLOWS})"
    r"(?:[ \t]*(?:,|and|or|through|to)[ \t]*\d+-\d+(?:-\d+)*(?![.,-]?\d)"
    rf"(?!{_RECALL_UNIT_FOLLOWS}))*",
    flags=re.IGNORECASE,
)
# These are heading *candidates*. Capitalization alone cannot distinguish a
# section number from an amount/rate/range at the start of a table row.
# _without_us_manual_locators requires independent numbering evidence and
# gives quantity-table context priority over that evidence.
_MANUAL_CENTS_LABEL = re.compile(r"[ \t]*\d+\.\d{2}")
_MANUAL_DOLLAR_AMOUNT_LINE = re.compile(r"[ \t]*\$[ \t]*\d")
_NUMBERED_MANUAL_HEADING_LABEL = re.compile(
    r"(?m)^[ \t]*(?!\d+\.00(?![.-]\d))\d+(?:[.-]\d+)+"
    r"(?=[ \t]+"
    r"(?!(?i:Percent|Percentage|Times|Dollars?|Cents?|Days?|Weeks?|Months?|"
    r"Years?|Hours?|Persons?|People|Members?|Million|Billion|Thousand|"
    r"Business|Calendar|Working)\b)"
    r"(?P<title>[A-Z][A-Za-z'’&()/-]*(?:[ \t,]+[A-Za-z'’&()/-]+){1,15})[ \t]*$)"
)
# A ZIP+4 extension. PDF extraction can split it after its hyphen
# (`22314-\n1500`) or space it on the same line (`20001 - 3710;`, only
# before punctuation, a line end or a telephone number: `22902 -2964
# 434-970-3160`). A hyphen that opens a new line is a bullet
# (`22314\n- 1500, ...`), never an extension.
_US_ZIP_PLUS_FOUR_EXTENSION = (
    r"(?:-[ \t]*(?:\r?\n[ \t]*)?\d{4}"
    r"|[ \t]+-[ \t]*\d{4}(?=[ \t]*(?:[;,)]|\.(?!\d)|\r?\n|\Z)"
    r"|[ \t]+(?:\(\d{3}\)[ \t]*|\d{3}[-.])\d{3}-\d{4}\b))(?![\d-])"
)
# Telephone numbers, post-office boxes, ZIP+4 codes and letter-prefixed form
# numbers (`Schedule K-210`) on US forms and notices.
_US_FORM_AND_CONTACT_IDENTIFIER_NUMERIC_RECALL = re.compile(
    r"(?:\(\d{3}\)\s*|\b(?:1-)?\d{3}[-.])\d{3}-\d{4}\b"
    r"|\bP\.?\s*O\.?\s*Box\s+\d+\b"
    rf"|\b[A-Z]{{2}}\s+\d{{5}}{_US_ZIP_PLUS_FOUR_EXTENSION}"
    r"|\b(?:Schedule|Form)[ \t]+[A-Z]{1,3}-\d+[A-Z]*(?:-\d+[A-Z]*)*\b"
)
# A complete US postal address: house number, capitalized street name and
# street suffix, an optional directional and suite/room, then `City, ST
# 12345[-6789]` with a USPS state code. Only the whole address is removed, so
# its house number and ZIP code are never demanded. A number after a digit,
# sign, decimal point or comma, or after a currency sign or `USD` across any
# whitespace (`$1,320 Braddock Place ...`, see `_us_street_address_or_amount`),
# is not a house number, and an address without the city/state/ZIP tail keeps
# every number demanded.
_US_STREET_ADDRESS_NUMERIC_RECALL = re.compile(
    r"(?<![\w.,$¢£¥€₹₩+\-−])\d{1,6}"
    r"\s+[A-Z][A-Za-z.'’-]*(?:\s+[A-Z][A-Za-z.'’-]*){0,4}\s+"
    r"(?:Avenue|Ave\.?|Boulevard|Blvd\.?|Center|Drive|Dr\.?|Lane|Ln\.?|Parkway|"
    r"Pkwy\.?|Place|Pl\.?|Plaza|Road|Rd\.?|Street|St\.?|Way)(?![\w'’-])"
    r"(?:,?\s+[NS]\.?[EW]\.?(?![\w'’-]))?"
    r"(?:,\s*(?:Suite|Ste\.?|Room|Rm\.?)\s+(?=[A-Z]*\d)[A-Z\d]{1,6}(?:-\d{1,6})?"
    r"(?![\w'’-]))?"
    r",\s*[A-Za-z .'-]+,\s*"
    r"(?:A[KLRSZ]|C[AOT]|D[CE]|FL|G[AU]|HI|I[ADLN]|K[SY]|LA|M[ADEINOPST]|"
    r"N[CDEHJMVY]|O[HKR]|P[AR]|RI|S[CD]|T[NX]|UT|V[AIT]|W[AIVY])"
    rf"\s+\d{{5}}(?:{_US_ZIP_PLUS_FOUR_EXTENSION})?(?!\d)"
)
_US_STREET_ADDRESS_CURRENCY_PREFIX = re.compile(r"(?:[$¢£¥€₹₩]|\b(?i:usd))\Z")


def _us_street_address_or_amount(match: re.Match[str]) -> str:
    """Remove a postal address unless its first number reads as an amount."""

    before = match.start()
    while before > 0 and (
        match.string[before - 1].isspace() or match.string[before - 1] in "+-−–"
    ):
        before -= 1
    if _US_STREET_ADDRESS_CURRENCY_PREFIX.search(
        match.string, max(0, before - 4), before
    ):
        return match.group(0)
    return ""


# Paginated US manuals open each page with its page number and a revision
# stamp: the Oregon Programs Eligibility Notebook reads `93 (07/2026) Chapter
# 2:Eligibility ...` or `26 Chapter 1: Introduction ... (07/2026)`. These
# shapes are only locator candidates: the complete context must establish
# that neither the leading number nor the stamp is operative. A verified page
# number is replaced with a neutral mark, not removed: most pages are one
# line, and a line that then began with `Chapter N:` would be dropped whole
# as a structural heading, taking the page's real values with it.
_US_MANUAL_PAGE_NUMBER = re.compile(
    r"\A\s*\d{1,4}(?=\s+(?:\((?:0?[1-9]|1[0-2])/(?:19|20)\d{2}\)|Chapter\b))"
)
_US_MANUAL_REVISION_STAMP = re.compile(r"\((?:0?[1-9]|1[0-2])/(?:19|20)\d{2}\)")
_MANUAL_QUANTITY_CAPTION = re.compile(
    r"\b(?:amounts?|rates?|counts?|ages?|percent(?:ages?)?|increments?|"
    r"units?|costs?|quantit(?:y|ies)|currenc(?:y|ies)|USD|EUR|GBP|CAD|AUD|"
    r"coefficients?|factors?|formulas?|"
    r"dollars?|cents?|bands?|brackets?|earnings?|fees?|charges?|"
    r"days?|weeks?|months?|years?|hours?|household[ \t]+size|"
    r"(?:benefit|payment)[ \t]+levels?|standards?|limits?|ceilings?|thresholds?|maximums?|minimums?|"
    r"allotments?|deductions?|allowances?)\b",
    re.IGNORECASE,
)
_MANUAL_OPERATIVE_HEADING_CONTEXT = re.compile(
    _MANUAL_QUANTITY_CAPTION.pattern
    + r"|\b(?:monetary|payments?)\b"
    + r"|\b(?:shown[ \t]+(?:above|below)|(?:month|date|year)[ \t]+shown|"
    r"this[ \t]+date|effective[ \t]+date|as[ \t]+of)\b",
    re.IGNORECASE,
)
# Contents membership alone cannot prove a title harmless: financial topics
# can also caption operative numeric columns. Other titles require an exact
# reviewed complete-source role rather than heuristic context authority.
_MANUAL_NONOPERATIVE_PROCEDURE_TITLES = frozenset(
    {
        "Application Process",
        "Retroactive Applications",
        "Advance Notice of Adverse Action",
        "Warranties, Representations and Covenants",
    }
)
_MANUAL_TABLE_COLUMN = re.compile(
    r"[ \t]*(?:household[ \t]+(?:type|size)|assistance[ \t]+group|"
    r"benefit[ \t]+type|payment[ \t]+type)[ \t]*\r?\n?",
    re.IGNORECASE,
)
_MANUAL_QUANTITY_TITLE = re.compile(
    r"\b(?:monthly|weekly|daily|annual)[ \t]+(?:allowance|rate|amount|payment)\b"
    r"|\b(?:USD|EUR|GBP|CAD|AUD)\b"
    r"|\b(?:eligible[ \t]+children|young[ \t]+adults|payroll[ \t]+contribution|"
    r"child[ \t]+benefit|employer[ \t]+payment|filing[ \t]+charge)\b",
    re.IGNORECASE,
)
_TITLE_SUFFIX_LEGAL_CITATION = re.compile(
    r"\b(?:sections?\s+)?(?:\d+)?[a-z]\s*"
    r"[-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]"
    r"\s*\d+[a-z]?(?:\s*(?:,\s*(?:(?:and|or)\s+)?|"
    r"(?:and|or|through|to)\s+)(?:\d+)?[a-z]\s*"
    r"[-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]"
    r"\s*\d+[a-z]?)*\s+of\s+this\s+title\b",
    flags=re.IGNORECASE,
)
_LOUISIANA_RS_NUMERIC_RECALL_TARGET = (
    r"\d+[A-Za-z]?\s*:\s*"
    r"\d+[A-Za-z]?(?:(?:[-\u2010\u2011\u2012\u2013\u2014\u2015"
    r"\u2212\ufe58\ufe63\uff0d][A-Za-z0-9]+)|(?:\.\d+[A-Za-z]?))*"
    r"(?:\s*\(\s*(?:[A-Za-z]+|\d{1,2})\s*\))*"
)
_LOUISIANA_RS_NUMERIC_RECALL_CITATION = re.compile(
    r"(?<![A-Za-z0-9_])(?:(?:La\.?\s*)|(?:LSA\s*[-\u2010-\u2015]?\s*))?"
    r"R\.?\s*S\.?\s*(?:§{1,2}\s*)?"
    + _LOUISIANA_RS_NUMERIC_RECALL_TARGET
    + r"(?:\s*(?:,\s*(?:(?:and/or|and|or)\s+)?|"
    + r"(?:and/or|and|or|through|to)\s+|&\s*)"
    + _LOUISIANA_RS_NUMERIC_RECALL_TARGET
    + r")*"
    + r"(?![A-Za-z0-9_])(?![./:]\s*(?:[A-Za-z0-9]|\())"
    + r"(?![-\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]"
    + r"\s*(?:[A-Za-z0-9]|\())",
    flags=re.IGNORECASE,
)
_LOUISIANA_SESSION_LAW_CITATION = re.compile(
    r"\bActs?(?:\s+of)?\s+\d{4}\s*,?\s*"
    r"(?:"
    r"(?:(?:\d+\s*(?:st|nd|rd|th|d)|"
    r"[A-Za-z]+(?:[-\s]+[A-Za-z]+){0,3})\s+)?"
    r"(?:E\.?\s*S\.?|Ex\.?|Extra\.?|Extraordinary)\s*"
    r"(?:Sess\.?|Session)?\s*,?\s*"
    r")?"
    r"Nos?\.?\s*\d+(?:-\d+)?\b"
    r"(?:\s*(?:,|and)\s*(?:Nos?\.?\s*)?\d+(?:-\d+)?\b)*"
    r"(?:\s*,?\s*(?:"
    r"§§\s*\d+(?:\s*(?:,|and|through|to|[-–—])\s*\d+)*"
    r"|§\s*\d+"
    r"))?",
    flags=re.IGNORECASE,
)
_LOUISIANA_NOTWITHSTANDING_RS_REFERENCE_PREFIX = re.compile(
    r"^\s*notwithstanding\s+"
    r"(?:(?:the\s+)?provisions?\s+of\s+)?"
    r"R\.S\.\s*\d+[A-Za-z]?:\d+[A-Za-z]?"
    r"(?:-\d+[A-Za-z]?(?:\.\d+)*)?\s*[,;]\s*"
    r"(?P<tail>.+)$",
    flags=re.IGNORECASE,
)
_STRUCTURAL_REFERENCE = re.compile(
    r"\b(?:"
    r"Artikel(?:s|n)?|Art\.|"
    r"Absatz(?:es)?|Absätze(?:n)?|Abs\.|"
    r"Satz(?:es)?|Sätze(?:n)?|"
    r"Nummer(?:n)?|Nr\.|Buchstabe(?:n)?|Buchst\."
    r")\s*\d*[a-z]?"
    r"(?:\s*(?:,|und|bis|[-–—])\s*\d+[a-z]?)*\b",
    flags=re.IGNORECASE,
)
# Session-law history is a finite bibliographic grammar, not a blanket year or
# chapter-number filter. Singular `c.` takes one chapter; plural `cc.` takes
# at least two comma-separated chapters. Special sessions are explicitly
# bounded to Roman I-X. All entries and calendar dates must validate before
# the terminal suffix can be removed from computational recall.
_SESSION_LAW_ACTION = (
    r"amended(?:\s+by)?|as\s+amended|added(?:\s+by)?|"
    r"supplemented(?:\s+by)?|repealed(?:\s+by)?"
)
_SESSION_LAW_PUBLICATION = r"(?:P\.?\s*L\.?|L\.|Laws\b)\s*"
_SESSION_LAW_YEAR_CHAPTER = (
    r"\d{4}\s*,\s*"
    r"(?:Sp\.\s*Sess\.\s*(?:VIII|VII|VI|IV|III|II|IX|V|I|X)\s*,\s*)?"
    r"(?:c\.\s*\d+|cc\.\s*\d+(?:\s*,\s*\d+)+)"
)
_SESSION_LAW_PREFIX = (
    r"(?:(?P<history_label>history|source)\s*[:.\-–—]+\s*)?"
    rf"(?:(?P<action>{_SESSION_LAW_ACTION})\s+)?"
    rf"(?:{_SESSION_LAW_PUBLICATION})?"
)
# An explicit History label or an action plus a publication authenticates a
# history start even when its payload is empty or malformed. Weaker leaders
# (ordinary Source metadata and bare actions) need a year plus chapter marker.
# Discovery does not validate entries: every discovered suffix must pass the
# strict grammar below, so a malformed first entry cannot be bypassed.
_SESSION_LAW_HISTORY_LEADER = (
    r"(?:history|source)\s*[:.\-–—]+\s*|"
    rf"(?:{_SESSION_LAW_ACTION})\s+|{_SESSION_LAW_PUBLICATION}"
)
_SESSION_LAW_UNNAMED_PREFIX = (
    r"(?:(?:history|source)\s*[:.\-–—]+\s*)?"
    rf"(?:(?:{_SESSION_LAW_ACTION})\s+)?"
    rf"(?:{_SESSION_LAW_PUBLICATION})?"
)
_SESSION_LAW_EXPLICIT_HISTORY_LABEL = r"history\s*[:.\-–—]+\s*"
_SESSION_LAW_ACTION_PUBLICATION_START = (
    r"(?:(?:history|source)\s*[:.\-–—]+\s*)?"
    rf"(?:{_SESSION_LAW_ACTION})\s+{_SESSION_LAW_PUBLICATION}"
)
_SESSION_LAW_YEAR_CHAPTER_MARKER = (
    r"\d[A-Za-z\d]{0,7}\s*,\s*(?:Sp\.\s*Sess\.\s*[IVX]+\s*,\s*)?cc?\."
)
_SESSION_LAW_TAIL_START = re.compile(
    rf"(?<!\w)(?={_SESSION_LAW_EXPLICIT_HISTORY_LABEL}|"
    rf"{_SESSION_LAW_ACTION_PUBLICATION_START}|"
    rf"(?={_SESSION_LAW_HISTORY_LEADER})"
    rf"{_SESSION_LAW_UNNAMED_PREFIX}{_SESSION_LAW_YEAR_CHAPTER_MARKER}|"
    rf"{_SESSION_LAW_YEAR_CHAPTER_MARKER})"
    rf"{_SESSION_LAW_PREFIX}(?:\d{{4}}(?!\d))?",
    flags=re.IGNORECASE,
)
_SESSION_LAW_MONTH = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|"
    r"Aug(?:ust)?|Sept(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
)
_SESSION_LAW_ENTRY = re.compile(
    rf"\s*{_SESSION_LAW_PREFIX}{_SESSION_LAW_YEAR_CHAPTER}"
    r"(?:\s*(?:,|\.)\s*(?:(?:s|ss)\.|§{1,2})\s*[A-Za-z0-9]+"
    r"(?:[.:\-–—][A-Za-z0-9]+)*"
    r"(?:\s*(?:through|to|[-–—])\s*[A-Za-z0-9]+)?)?"
    r"(?:\s*,\s*eff(?:ective)?\.\s*"
    rf"(?P<effective_month>{_SESSION_LAW_MONTH})"
    r"\.?\s+(?P<effective_day>\d{1,2})\s*,\s*(?P<effective_year>\d{4}))?"
    r"(?:\s*,\s*operative\s+"
    rf"(?P<operative_month>{_SESSION_LAW_MONTH})"
    r"\.?\s+(?P<operative_day>\d{1,2})\s*,\s*(?P<operative_year>\d{4}))?"
    r"\s*\.?\s*",
    flags=re.IGNORECASE,
)
_SESSION_LAW_SENTENCE_SEPARATOR = re.compile(
    r"(?<=\d)\.\s+(?="
    rf"(?:(?:{_SESSION_LAW_ACTION})\s+)?"
    rf"(?:{_SESSION_LAW_PUBLICATION})?{_SESSION_LAW_YEAR_CHAPTER})",
    flags=re.IGNORECASE,
)
_SESSION_LAW_PARAGRAPH_BREAK = re.compile(r"\r?\n[ \t]*\r?\n")
_SESSION_LAW_CONTINUATION_WRAPPER = re.compile(r"(?:[('\"“‘][ \t]*){0,2}")
# A validated citation can introduce body prose in an earlier paragraph.
# The opening must be ordinary prose, currency or a number with a unit or
# arithmetic operator, with finite history continuations excluded. Dates,
# section/chapter lists, qualifiers and actions cannot release a malformed
# chain. The body prose is retained in full.
_SESSION_LAW_OPERATIVE_CONTINUATION = re.compile(
    r"^(?!"
    r"(?:history|source)\s*[:.\-–—]|"
    rf"(?:{_SESSION_LAW_ACTION})\s+(?:{_SESSION_LAW_PUBLICATION}|\d|"
    r"unknown\b|incomplete\b|missing\b)|"
    rf"(?:{_SESSION_LAW_ACTION})(?=\s*[.;]|\s*$)|"
    rf"{_SESSION_LAW_PUBLICATION}(?:\d|[.;,]|unknown\b|incomplete\b|missing\b|$)|"
    r"(?:s|ss|c|cc|eff|effective|Sp|Sess)\.|"
    r"(?:s|ss|c|cc|sections?|chapters?)\s+(?:\d|unknown\b|incomplete\b|missing\b)|"
    r"Sp\.?\s+Sess\.?(?:\s|$)|"
    r"Acts?\s+\d|"
    rf"(?:eff|effective|operative)\s+(?:{_SESSION_LAW_MONTH}\b|\d|"
    r"unknown\b|incomplete\b|missing\b)|"
    r"(?:effective|operative)(?=\s*[.;]|\s*$)|"
    r"(?:unknown|incomplete|missing)\b|"
    rf"(?:{_SESSION_LAW_MONTH})\.?\s+\d"
    r")"
    r"(?:[A-Za-z]+\b(?=[ \t\r\n$])|\$\d|"
    r"\d+(?:[.,]\d+)?[ \t]+"
    r"(?!(?:c|cc|s|ss|Sp|Sess|Laws?|history|source|amended|added|supplemented|repealed)\b)"
    r"[A-Za-z]+\b|\d+\s*[-+*/]\s*\d+\b)",
    flags=re.IGNORECASE,
)
# The same predicate anchored at a ``pos`` rather than the string start, so a
# caller can match inside the source without copying the remaining suffix.
_SESSION_LAW_OPERATIVE_CONTINUATION_AT = re.compile(
    _SESSION_LAW_OPERATIVE_CONTINUATION.pattern.removeprefix("^"),
    flags=_SESSION_LAW_OPERATIVE_CONTINUATION.flags,
)
_ALABAMA_TERMINAL_ACT_HISTORY_ENTRY = re.compile(
    r"\s*Acts?\s+"
    r"(?:"
    r"\d{4}\s*,\s*"
    r"(?:"
    r"(?:(?:\d+\s*(?:st|nd|rd|th|d)|"
    r"[A-Za-z]+(?:[-\s]+[A-Za-z]+){0,3})\s+)?"
    r"(?:E\.?\s*S\.?|Ex\.?|Extra\.?|Extraordinary)\s*"
    r"(?:Sess\.?|Session)\s*,\s*"
    r")?"
    r"Nos?\.?\s*\d+(?:-\d+)?"
    r"|\d{2}-\d+"
    r")"
    r"\s*,\s*p\.?\s*\d+"
    r"(?:\s*,\s*§{1,2}\s*\d+(?:\s*(?:,|and|through|to|[-–—])\s*\d+)*)?"
    r"\s*\.?\s*",
    flags=re.IGNORECASE,
)
_ALABAMA_TERMINAL_CODE_HISTORY_ENTRY = re.compile(
    r"\s*Code\s+\d{4}\s*,\s*T\.?\s*\d+\s*,\s*"
    r"§{1,2}\s*\d+(?:\s*(?:,|and|through|to|[-–—])\s*\d+)*"
    r"\s*\.?\s*",
    flags=re.IGNORECASE,
)
_ARMENIAN_AMENDMENT_HISTORY_DASH = (
    r"[-\u058a\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]"
)
# The one dot class, used for the article label's decimal point, the action's
# optional period and both date separators. The encoder prompt names exactly
# these characters.
_ARMENIAN_AMENDMENT_HISTORY_DOT = r"[.\u2024]"
# ASCII digits only, as the prompt's `DD.MM.YY` spells them. Matched against
# the stripped body: no lazy quantifier and no trailing
# whitespace run, so a long whitespace run cannot make the match quadratic.
_ARMENIAN_AMENDMENT_HISTORY_ARTICLE = re.compile(
    rf"[0-9]+(?:{_ARMENIAN_AMENDMENT_HISTORY_DOT}[0-9]+)?"
    rf"{_ARMENIAN_AMENDMENT_HISTORY_DASH}"
    r"(?:ին|րդ)\s+հոդվածը\s+(?P<history>\S.*)",
    flags=re.DOTALL,
)
# Amendment-action abbreviations are matched case-insensitively, but the paired
# `ՀՕ-N[-N]-Ն` law identifier is not: scope the case folding to the action
# alternation so recognizing an uppercase action cannot also fold the
# identifier's casing and admit a citation the prompt contract does not permit.
_ARMENIAN_AMENDMENT_HISTORY_ACTION_TOKEN = (
    rf"(?i:փոփ|լրաց|խմբ)(?:{_ARMENIAN_AMENDMENT_HISTORY_DOT})?"
)
_ARMENIAN_AMENDMENT_HISTORY_CITATION_TOKEN = (
    rf"[0-9]{{2}}{_ARMENIAN_AMENDMENT_HISTORY_DOT}[0-9]{{2}}"
    rf"{_ARMENIAN_AMENDMENT_HISTORY_DOT}[0-9]{{2}}(?:[0-9]{{2}})?\s+Հ(?:Օ)?"
    rf"{_ARMENIAN_AMENDMENT_HISTORY_DASH}[0-9]+(?:"
    rf"{_ARMENIAN_AMENDMENT_HISTORY_DASH}[0-9]+)?"
    rf"{_ARMENIAN_AMENDMENT_HISTORY_DASH}Ն"
)
_ARMENIAN_AMENDMENT_HISTORY_ACTION = re.compile(
    _ARMENIAN_AMENDMENT_HISTORY_ACTION_TOKEN,
)
_ARMENIAN_AMENDMENT_HISTORY_CITATION = re.compile(
    _ARMENIAN_AMENDMENT_HISTORY_CITATION_TOKEN,
)
# Possessive: no action or citation token can begin with a separator character,
# so giving separator characters back never enables a match. Committing to the
# longest run keeps the sequence scan linear on adversarial separator runs.
_ARMENIAN_AMENDMENT_HISTORY_SEPARATOR = r"[\s,;]++"
_ARMENIAN_AMENDMENT_HISTORY_SEQUENCE = re.compile(
    rf"\s*(?:{_ARMENIAN_AMENDMENT_HISTORY_ACTION_TOKEN}|"
    rf"{_ARMENIAN_AMENDMENT_HISTORY_CITATION_TOKEN})(?:"
    rf"{_ARMENIAN_AMENDMENT_HISTORY_SEPARATOR}(?:"
    rf"{_ARMENIAN_AMENDMENT_HISTORY_ACTION_TOKEN}|"
    rf"{_ARMENIAN_AMENDMENT_HISTORY_CITATION_TOKEN}))*\s*",
)
# One parenthetical alone on its line. Any parenthesis inside the body fails
# the candidate, so nested ledgers are never consumed whole.
_ARMENIAN_AMENDMENT_HISTORY_CANDIDATE = re.compile(
    r"(?:^|\r?\n)[ \t]*\((?P<body>[^()]*)\)(?=[ \t]*(?:\r?\n|$))"
)
_FORMULA_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _glued_section_sentence_matches(text: str) -> tuple[re.Match[str], ...]:
    """Recognize a printed sentence number glued to its opening section sign."""

    matches = []
    for match in _GLUED_SECTION_SENTENCE_MARKER.finditer(text):
        prefix = text[: match.start()].rstrip()
        if prefix and not re.fullmatch(r"\(\d+[a-z]?\)", prefix):
            if not prefix.endswith((".", "!", "?")):
                continue
            # A legal address such as "Art. 2§ 3" is not a new sentence.
            if re.search(r"\b(?:Art|Abs|Nr|S|Sec|Sect)\.$", prefix, re.IGNORECASE):
                continue
        matches.append(match)
    return tuple(matches)


_IRS_REVENUE_PROCEDURE_CITATION = re.compile(
    r"us/guidance/irs/rev-proc-\d{4}-\d+(?:/page-\d+)?"
)
_IRS_REVENUE_PROCEDURE_SUBSECTION = re.compile(
    r"(?:^|(?<=[.!?])\s+|\n[ \t]*)"
    r"(?P<marker>\.(?P<label>\d{2})[ \t]+[A-Z][A-Za-z'-]*"
    r"(?:[ \t]+(?:[A-Z][A-Za-z'-]*|as|of|for|and|or|the|to|in)){1,15}\.)"
)
_IRS_REVENUE_PROCEDURE_PARAGRAPH = re.compile(
    r"(?:^|(?<=[.!?])\s+|\n[ \t]*)"
    r"(?P<marker>\((?P<label>\d+[a-z]?)\))(?=\s+[A-Z])"
)


def _irs_revenue_procedure_structure(
    source_text: str,
) -> tuple[SourceStructureBranch, ...]:
    """Recognize titled .NN sections and printed paragraphs after PDF flattening."""

    headings = tuple(_IRS_REVENUE_PROCEDURE_SUBSECTION.finditer(source_text))
    branches: list[SourceStructureBranch] = []
    containers: list[tuple[tuple[str, ...], int, int]] = [
        ((), 0, headings[0].start("marker") if headings else len(source_text))
    ]
    for index, heading in enumerate(headings):
        start = heading.start("marker")
        end = (
            headings[index + 1].start("marker")
            if index + 1 < len(headings)
            else len(source_text)
        )
        path = (heading.group("label"),)
        branches.append(
            SourceStructureBranch(
                path,
                "guidance-subsection",
                heading.group("marker"),
                source_text[start:end].strip(),
                start,
                end,
            )
        )
        containers.append((path, start, end))
    for path, start, end in containers:
        # Restore only authenticated PDF-flattened paragraph boundaries, with
        # identical offsets. Parse the whole subsection once so nested numeric
        # markers retain their U.S. outline ancestors rather than becoming roots.
        local_source = source_text[start:end]
        paragraph_boundaries = {
            match.start("marker") - 1
            for match in _IRS_REVENUE_PROCEDURE_PARAGRAPH.finditer(local_source)
            if match.start("marker") > 0
            and local_source[match.start("marker") - 1].isspace()
        }
        outlined_source = "".join(
            "\n" if index in paragraph_boundaries else char
            for index, char in enumerate(local_source)
        )
        for nested in recognize_source_structure(outlined_source):
            if nested.kind != "sentence":
                branches.append(
                    replace(
                        nested,
                        path=(*path, *nested.path),
                        text=source_text[
                            start + nested.start : start + nested.end
                        ].strip(),
                        start=start + nested.start,
                        end=start + nested.end,
                        structural_numeric_spans=tuple(
                            (start + span_start, start + span_end)
                            for span_start, span_end in nested.structural_numeric_spans
                        ),
                    )
                )
    return tuple(sorted(branches, key=lambda branch: (branch.start, len(branch.path))))


def recognize_source_structure(
    source_text: str, *, corpus_citation_path: str = ""
) -> tuple[SourceStructureBranch, ...]:
    """Recognize paragraph, list, letter, and glued German sentence markers."""

    if _IRS_REVENUE_PROCEDURE_CITATION.fullmatch(corpus_citation_path):
        return _irs_revenue_procedure_structure(source_text)

    outline = bare_outline(source_text)
    if outline:
        masked = mask_bare_labels(source_text, outline)
        branches = [
            SourceStructureBranch(
                tuple(part.lower() for part in label.path),
                "paragraph" if label.token[0].isdigit() else "letter",
                f"({label.token})",
                masked[label.start : label.end],
                label.start,
                label.end,
            )
            for label in outline
        ]
        definitions = bare_definition_headers(source_text, outline)
        for index, (start, term) in enumerate(definitions):
            owner = max(
                (label for label in outline if label.start <= start < label.end),
                key=lambda label: (len(label.path), label.start),
                default=None,
            )
            end = (
                definitions[index + 1][0]
                if index + 1 < len(definitions)
                else len(source_text)
            )
            if owner is not None:
                end = min(end, owner.end)
            branches.append(
                SourceStructureBranch(
                    tuple(part.lower() for part in owner.path) if owner else (),
                    "definition",
                    term,
                    masked[start:end],
                    start,
                    end,
                )
            )
        prefix_end = min(branch.start for branch in branches)
        if masked[:prefix_end].strip():
            # A bare outline's unlabelled chapeau remains an active source
            # branch: its thresholds and conditions are independent of the
            # following child bodies. Keep offsets aligned with source text.
            branches.append(
                SourceStructureBranch(
                    (),
                    "source-unit",
                    "source unit",
                    masked[:prefix_end],
                    0,
                    prefix_end,
                )
            )
        return tuple(
            sorted(branches, key=lambda branch: (branch.start, len(branch.path)))
        )

    branches: list[SourceStructureBranch] = []
    paragraph_segments: list[tuple[tuple[str, ...], int, int, str]] = []
    hebrew_script = _hebrew_script()
    hebrew_markers = (
        tuple(
            _OutlineMarker(start, path, kind, label)
            for start, path, kind, label in hebrew_source_text.hebrew_outline_markers(
                source_text
            )
        )
        if hebrew_script
        else ()
    )
    dotted_matches = _qualified_dotted_subsection_matches(source_text)
    dotted_candidates = _dotted_subsection_boundary_matches(source_text)
    outer_segments = [
        (
            match,
            match.start(),
            next(
                (
                    candidate.start()
                    for candidate in dotted_candidates
                    if candidate.start() > match.start()
                ),
                len(source_text),
            ),
        )
        for match in dotted_matches
    ]

    if hebrew_script and hebrew_markers:
        # Israeli statutes: (א) subsections, (1) paragraphs, (א) subparagraphs,
        # with paths transliterated by gematria value ((ב)(2) -> b/2).
        for marker, end in _nested_outline_marker_spans(
            hebrew_markers,
            outer_end=len(source_text),
        ):
            text = source_text[marker.start : end].strip()
            # The label may carry points or direction marks (``(א‏) (בוטל)``).
            if _is_editorial_omission(hebrew_source_text.fold_marks(text)):
                continue
            branches.append(
                SourceStructureBranch(
                    marker.path,
                    marker.kind,
                    marker.label,
                    text,
                    marker.start,
                    end,
                )
            )
            paragraph_segments.append((marker.path, marker.start, end, text))
    elif outer_segments:
        prefix_end = outer_segments[0][1]
        prefix_matches = list(_PARAGRAPH_MARKER.finditer(source_text, 0, prefix_end))
        for index, match in enumerate(prefix_matches):
            start = match.start()
            end = (
                prefix_matches[index + 1].start()
                if index + 1 < len(prefix_matches)
                else prefix_end
            )
            label = match.group("label").lower()
            text = source_text[start:end].strip()
            if _is_editorial_omission(text):
                continue
            path = (label,)
            branches.append(
                SourceStructureBranch(
                    path, "paragraph", match.group("marker"), text, start, end
                )
            )
            paragraph_segments.append((path, start, end, text))

        for outer_match, outer_start, outer_end in outer_segments:
            outer_label = outer_match.group("label").lower()
            outer_path = (outer_label,)
            outer_text = source_text[outer_start:outer_end].strip()
            if not _is_editorial_omission(outer_text):
                branches.append(
                    SourceStructureBranch(
                        outer_path,
                        "paragraph",
                        outer_match.group("marker"),
                        outer_text,
                        outer_start,
                        outer_end,
                    )
                )
            nested_markers = _nested_parenthesized_outline_markers(
                source_text,
                outer_path=outer_path,
                start=outer_start,
                end=outer_end,
            )
            if not nested_markers:
                paragraph_segments.append(
                    (outer_path, outer_start, outer_end, outer_text)
                )
                continue
            for marker, end in _nested_outline_marker_spans(
                nested_markers,
                outer_end=outer_end,
            ):
                text = source_text[marker.start : end].strip()
                if _is_editorial_omission(text):
                    continue
                branches.append(
                    SourceStructureBranch(
                        marker.path,
                        marker.kind,
                        marker.label,
                        text,
                        marker.start,
                        end,
                    )
                )
                paragraph_segments.append((marker.path, marker.start, end, text))
    else:
        nested_markers = _qualified_parenthesized_legal_outline_markers(source_text)
        if nested_markers:
            for marker, end in _nested_outline_marker_spans(
                nested_markers,
                outer_end=len(source_text),
            ):
                text = source_text[marker.start : end].strip()
                if _is_editorial_omission(text):
                    continue
                branches.append(
                    SourceStructureBranch(
                        marker.path,
                        marker.kind,
                        marker.label,
                        text,
                        marker.start,
                        end,
                    )
                )
                paragraph_segments.append((marker.path, marker.start, end, text))
        else:
            paragraph_matches = list(_PARAGRAPH_MARKER.finditer(source_text))
            for index, match in enumerate(paragraph_matches):
                start = match.start()
                end = (
                    paragraph_matches[index + 1].start()
                    if index + 1 < len(paragraph_matches)
                    else len(source_text)
                )
                label = match.group("label").lower()
                text = source_text[start:end].strip()
                if _is_editorial_omission(text):
                    continue
                path = (label,)
                branches.append(
                    SourceStructureBranch(
                        path, "paragraph", match.group("marker"), text, start, end
                    )
                )
                paragraph_segments.append((path, start, end, text))

    if not paragraph_segments:
        paragraph_segments = [((), 0, len(source_text), source_text)]

    legacy_marker_offsets = {
        *(match.start() for match in _NUMBER_MARKER.finditer(source_text)),
        *(match.start() for match in _LETTER_MARKER.finditer(source_text)),
        *(match.start() for match in _GLUED_SENTENCE_MARKER.finditer(source_text)),
        *(
            match.start()
            for match in _GLUED_SECTION_SENTENCE_MARKER.finditer(source_text)
        ),
        *(match.start() for match in _EXPLICIT_SENTENCE_MARKER.finditer(source_text)),
    }
    owner_paths = _most_specific_segment_paths_at_offsets(
        paragraph_segments,
        legacy_marker_offsets,
    )

    for paragraph_path, paragraph_start, paragraph_end, _ in paragraph_segments:
        paragraph_text = source_text[paragraph_start:paragraph_end]
        number_matches = [
            match
            for match in _NUMBER_MARKER.finditer(paragraph_text)
            if owner_paths.get(paragraph_start + match.start()) == paragraph_path
        ]
        number_segments: list[tuple[tuple[str, ...], int, int]] = []
        for index, match in enumerate(number_matches):
            start = paragraph_start + match.start()
            end = (
                paragraph_start + number_matches[index + 1].start()
                if index + 1 < len(number_matches)
                else paragraph_end
            )
            label = match.group("label").lower()
            path = (*paragraph_path, label)
            text = source_text[start:end].strip()
            if _is_editorial_omission(text):
                continue
            branches.append(
                SourceStructureBranch(
                    path, "number", match.group("marker"), text, start, end
                )
            )
            number_segments.append((path, start, end))

        letter_containers = number_segments or [
            (paragraph_path, paragraph_start, paragraph_end)
        ]
        for container_path, container_start, container_end in letter_containers:
            container_text = source_text[container_start:container_end]
            letter_matches = [
                match
                for match in _LETTER_MARKER.finditer(container_text)
                if owner_paths.get(container_start + match.start()) == paragraph_path
            ]
            for index, match in enumerate(letter_matches):
                start = container_start + match.start()
                end = (
                    container_start + letter_matches[index + 1].start()
                    if index + 1 < len(letter_matches)
                    else container_end
                )
                label = match.group("label").lower()
                path = (*container_path, label)
                text = source_text[start:end].strip()
                if _is_editorial_omission(text):
                    continue
                branches.append(
                    SourceStructureBranch(
                        path, "letter", match.group("marker"), text, start, end
                    )
                )

        sentence_matches = [
            match
            for match in sorted(
                (
                    *_GLUED_SENTENCE_MARKER.finditer(paragraph_text),
                    *_glued_section_sentence_matches(paragraph_text),
                    *_EXPLICIT_SENTENCE_MARKER.finditer(paragraph_text),
                ),
                key=lambda item: item.start(),
            )
            if owner_paths.get(paragraph_start + match.start()) == paragraph_path
        ]
        for index, match in enumerate(sentence_matches):
            start = paragraph_start + match.start()
            end = (
                paragraph_start + sentence_matches[index + 1].start()
                if index + 1 < len(sentence_matches)
                else paragraph_end
            )
            label = match.group("label")
            path = (*paragraph_path, f"satz-{label}")
            text = source_text[start:end].strip()
            if _is_editorial_omission(text):
                continue
            branches.append(
                SourceStructureBranch(
                    path, "sentence", f"Satz {label}", text, start, end
                )
            )

    unique = {(branch.path, branch.kind, branch.start): branch for branch in branches}
    return tuple(
        sorted(
            unique.values(),
            key=lambda branch: (branch.start, len(branch.path), branch.kind),
        )
    )


def _most_specific_segment_paths_at_offsets(
    segments: Sequence[tuple[tuple[str, ...], int, int, str]],
    offsets: Iterable[int],
) -> dict[int, tuple[str, ...]]:
    """Assign legacy markers to their deepest active structural segment."""

    ordered_segments = sorted(segments, key=lambda item: item[1])
    active: list[tuple[int, int, int, tuple[str, ...]]] = []
    owners: dict[int, tuple[str, ...]] = {}
    segment_index = 0
    for offset in sorted(set(offsets)):
        while (
            segment_index < len(ordered_segments)
            and ordered_segments[segment_index][1] <= offset
        ):
            path, start, end, _text = ordered_segments[segment_index]
            heapq.heappush(active, (-len(path), -start, end, path))
            segment_index += 1
        while active and active[0][2] <= offset:
            heapq.heappop(active)
        if active:
            owners[offset] = active[0][3]
    return owners


def _qualified_dotted_subsection_matches(source_text: str) -> tuple[re.Match[str], ...]:
    """Return an ordered legal subsection sequence, excluding isolated initials."""

    candidates = tuple(_DOTTED_SUBSECTION_MARKER.finditer(source_text))
    for start, match in enumerate(candidates):
        if match.group("label") != "A":
            continue
        sequence = [match]
        for candidate_index in range(start + 1, len(candidates)):
            candidate = candidates[candidate_index]
            expected = chr(ord(sequence[-1].group("label")) + 1)
            if candidate.group("label") == expected:
                if len(sequence) >= 2 and _dotted_marker_is_ignorable(
                    source_text, candidate
                ):
                    continue
                sequence.append(candidate)
                continue
            if _dotted_marker_is_ignorable(source_text, candidate):
                continue
            break
        if len(sequence) >= 2:
            return tuple(sequence)
    return ()


def _nested_parenthesized_outline_markers(
    source_text: str,
    *,
    outer_path: tuple[str, ...],
    start: int,
    end: int,
) -> tuple[_OutlineMarker, ...]:
    """Resolve attached and line-leading U.S. subsection marker chains."""

    matches = tuple(_PARENTHESIZED_OUTLINE_MARKER.finditer(source_text, start, end))
    markers: list[_OutlineMarker] = []
    active: list[str] = []
    active_roman = False
    for index, match in enumerate(matches):
        labels = tuple(
            label.lower()
            for label in re.findall(r"\(([A-Za-z0-9]+)\)", match.group("marker"))
        )
        if not labels or not all(
            _is_parenthesized_outline_label(label) for label in labels
        ):
            continue
        marker_start = match.start("marker")
        if len(labels) > 1:
            active = []
            for label in labels:
                active.append(label)
                markers.append(
                    _OutlineMarker(
                        marker_start,
                        (*outer_path, *active),
                        "paragraph" if label[0].isdigit() else "letter",
                        f"({label})",
                    )
                )
            active_roman = len(active) >= 3 and _is_roman_outline_label(active[-1])
            continue

        label = labels[0]
        if label[0].isdigit():
            numeric_depth = next(
                (
                    position
                    for position in range(len(active) - 1, -1, -1)
                    if active[position][0].isdigit()
                ),
                None,
            )
            if active and not active[-1][0].isdigit() and label == "1":
                active = [*active, label]
            elif numeric_depth is None:
                active = [*active, label] if active else [label]
            else:
                active = [*active[:numeric_depth], label]
            active_roman = False
        else:
            next_labels = (
                tuple(
                    item.lower()
                    for item in re.findall(
                        r"\(([A-Za-z0-9]+)\)", matches[index + 1].group("marker")
                    )
                )
                if index + 1 < len(matches)
                else ()
            )
            begins_roman_children = (
                label == "i"
                and len(active) >= 2
                and bool(next_labels)
                and next_labels[0] == "ii"
            )
            continues_roman_children = (
                active_roman and len(active) >= 3 and _is_roman_outline_label(label)
            )
            if begins_roman_children:
                active = [*active, label]
                active_roman = True
            elif continues_roman_children:
                active = [*active[:-1], label]
                active_roman = True
            else:
                numeric_depth = next(
                    (
                        position
                        for position in range(len(active) - 1, -1, -1)
                        if active[position][0].isdigit()
                    ),
                    None,
                )
                if active and active[-1][0].isdigit():
                    if label == "a":
                        active = [*active, label]
                    else:
                        parent_letter_depth = next(
                            (
                                position
                                for position in range(len(active) - 2, -1, -1)
                                if not active[position][0].isdigit()
                            ),
                            None,
                        )
                        active = (
                            [*active[:parent_letter_depth], label]
                            if parent_letter_depth is not None
                            else [label]
                        )
                elif numeric_depth is not None:
                    active = [*active[: numeric_depth + 1], label]
                else:
                    active = [label]
                active_roman = False
        markers.append(
            _OutlineMarker(
                marker_start,
                (*outer_path, *active),
                "paragraph" if label[0].isdigit() else "letter",
                f"({label})",
            )
        )
    return tuple(markers)


_PARENTHESIZED_LEGAL_OUTLINE_LEVELS = (
    "lower-alpha",
    "numeric",
    "roman",
    "upper-alpha",
    "numeric",
    "roman",
    "upper-alpha",
    "numeric",
    "roman",
)
_NUMERIC_ROOT_PARENTHESIZED_LEGAL_OUTLINE_LEVELS = (
    "numeric",
    "lower-alpha",
    "roman",
    "upper-alpha",
    "numeric",
    "roman",
    "upper-alpha",
    "numeric",
    "roman",
)


def _qualified_parenthesized_legal_outline_markers(
    source_text: str,
) -> tuple[_OutlineMarker, ...]:
    """Resolve a proven all-parenthesized U.S. legal outline.

    A flat parenthesized scan cannot distinguish repeated labels such as
    ``(a)(3)``, ``(a)(6)(ii)(G)(3)``, and ``(c)(3)``.  Use the conventional
    U.S. legal hierarchy only when sequential root subsections prove that the
    source is an outline; otherwise retain the legacy flat recognizer.
    """

    line_matches = tuple(_PARENTHESIZED_OUTLINE_MARKER.finditer(source_text))
    all_inline_matches = tuple(
        _INLINE_PARENTHESIZED_OUTLINE_MARKER.finditer(source_text)
    )
    flattened_root_starts = _flattened_inline_numeric_root_starts(
        source_text,
        line_matches,
        all_inline_matches,
    )
    inline_matches = tuple(
        match
        for match in all_inline_matches
        if (
            match.start("marker") in flattened_root_starts
            or not _inline_outline_marker_has_reference_context(source_text, match)
        )
    )
    inline_marker_starts = {match.start("marker") for match in inline_matches}
    matches = tuple(
        heapq.merge(
            line_matches,
            inline_matches,
            key=lambda match: match.start("marker"),
        )
    )
    if not matches:
        return ()

    first_labels = tuple(re.findall(r"\(([A-Za-z0-9]+)\)", matches[0].group("marker")))
    outline_levels = (
        _NUMERIC_ROOT_PARENTHESIZED_LEGAL_OUTLINE_LEVELS
        if first_labels and first_labels[0][0].isdigit()
        else _PARENTHESIZED_LEGAL_OUTLINE_LEVELS
    )

    markers: list[_OutlineMarker] = []
    emitted_paths: set[tuple[str, ...]] = set()
    active: list[tuple[int, str, str, str]] = []
    for match in matches:
        raw_labels = tuple(re.findall(r"\(([A-Za-z0-9]+)\)", match.group("marker")))
        if not raw_labels or not all(
            _is_parenthesized_outline_label(label) for label in raw_labels
        ):
            continue
        marker_start = match.start("marker")
        attached_parent_level: int | None = None
        for label_index, raw_label in enumerate(raw_labels):
            level, category = _parenthesized_legal_outline_level(
                raw_label,
                active=active,
                attached_parent_level=attached_parent_level,
                outline_levels=outline_levels,
            )
            if (
                marker_start in inline_marker_starts
                and marker_start not in flattened_root_starts
                and label_index == 0
                and level == 0
            ):
                break
            active = [entry for entry in active if entry[0] < level]
            normalized = raw_label.lower()
            active.append((level, raw_label, normalized, category))
            path = tuple(entry[2] for entry in active)
            is_repeated_attached_ancestor = (
                len(raw_labels) > 1
                and label_index < len(raw_labels) - 1
                and path in emitted_paths
            )
            if not is_repeated_attached_ancestor:
                markers.append(
                    _OutlineMarker(
                        marker_start,
                        path,
                        (
                            "paragraph"
                            if category == "numeric" or level == 0
                            else "letter"
                        ),
                        f"({raw_label})",
                    )
                )
                emitted_paths.add(path)
            attached_parent_level = level

    root_category = outline_levels[0]
    root_labels = [marker.label[1:-1] for marker in markers if len(marker.path) == 1]
    has_sequential_roots = any(
        _parenthesized_outline_label_is_first(first, root_category)
        and _parenthesized_outline_label_follows(first, second, root_category)
        for first, second in itertools.pairwise(root_labels)
    )
    return tuple(markers) if has_sequential_roots else ()


def _flattened_inline_numeric_root_starts(
    source_text: str,
    line_matches: Sequence[re.Match[str]],
    inline_matches: Sequence[re.Match[str]],
) -> frozenset[int]:
    """Recover a proven numeric root sequence from flattened statutory prose.

    Official PDF extraction can place ``(1)`` through ``(N)`` after sentence
    punctuation on one physical line.  Those markers are ambiguous in
    isolation, so restore them only when the first outline candidate is an
    inline ``(1)`` and independently qualified inline candidates prove
    contiguous ``(2)`` and ``(3)`` successors.  Every recovered root must be
    free of direct cross-reference context.  A trailing ``of/under this
    section.`` from the preceding statutory sentence is treated as stale only
    when that sentence has no explicit see/refer/compare command.  Attached
    children and capitalization never override an explicit reference.  The
    initial ``(1)`` remains provisional until those successors are found.
    """

    ordered_matches = tuple(
        heapq.merge(
            line_matches,
            inline_matches,
            key=lambda match: match.start("marker"),
        )
    )
    if not ordered_matches or ordered_matches[0] not in inline_matches:
        return frozenset()
    has_inline_first_root = any(
        tuple(re.findall(r"\(([A-Za-z0-9]+)\)", match.group("marker")))[:1] == ("1",)
        for match in inline_matches
    )
    if not has_inline_first_root:
        return frozenset()

    expected = 1
    selected_starts: list[int] = []
    for match in inline_matches:
        labels = tuple(re.findall(r"\(([A-Za-z0-9]+)\)", match.group("marker")))
        if not labels or re.fullmatch(r"\d+", labels[0]) is None:
            continue
        value = int(labels[0])
        if value == expected:
            candidate_is_proven = (
                _flattened_inline_numeric_first_root_is_proven(source_text, match)
                if expected == 1
                else _flattened_inline_numeric_root_is_proven(source_text, match)
            )
            if candidate_is_proven:
                selected_starts.append(match.start("marker"))
                expected += 1
            continue
        if value > expected and _flattened_inline_numeric_root_is_proven(
            source_text,
            match,
        ):
            break
    return frozenset(selected_starts) if expected >= 4 else frozenset()


def _flattened_inline_numeric_root_has_explicit_reference_command(
    source_text: str,
    match: re.Match[str],
) -> bool:
    prefix = _flattened_inline_numeric_root_immediate_clause(source_text, match)
    return _INLINE_OUTLINE_EXPLICIT_REFERENCE_COMMAND.search(prefix) is not None


def _flattened_inline_numeric_first_root_is_proven(
    source_text: str,
    match: re.Match[str],
) -> bool:
    """Keep the first root provisional without admitting a reference marker."""

    prefix = _flattened_inline_numeric_root_immediate_clause(source_text, match)
    if _INLINE_OUTLINE_EXPLICIT_REFERENCE_COMMAND.search(prefix):
        return False
    has_reference_context = bool(
        _INLINE_OUTLINE_REFERENCE_CONTEXT.search(prefix)
        or _INLINE_OUTLINE_NAMED_REFERENCE_CONTEXT.search(prefix)
    )
    if not has_reference_context:
        return True
    return (
        re.search(
            r"\bKRS\s+\d+[A-Za-z]?(?:\.\d+[A-Za-z]?)*[.!?]\s*$",
            prefix,
            re.IGNORECASE,
        )
        is not None
    )


def _flattened_inline_numeric_root_context_prefix(
    source_text: str,
    match: re.Match[str],
) -> str:
    """Return local context without bleeding through a prior inline marker."""

    marker_start = match.start("marker")
    context_start = max(0, marker_start - 256)
    for previous in _INLINE_PARENTHESIZED_OUTLINE_MARKER.finditer(
        source_text,
        context_start,
        marker_start,
    ):
        context_start = max(context_start, previous.end("marker"))
    return source_text[context_start:marker_start]


def _flattened_inline_numeric_root_immediate_clause(
    source_text: str,
    match: re.Match[str],
) -> str:
    """Return the final clause while preserving periods in legal abbreviations."""

    prefix = _flattened_inline_numeric_root_context_prefix(source_text, match)
    masked = list(prefix)
    abbreviation_patterns = (
        re.compile(r"\b(?:e\.g|i\.e|U\.S)\.", re.IGNORECASE),
        re.compile(r"\b(?:[A-Z]\.){2,}"),
        re.compile(r"\b[A-Z][A-Za-z]{0,7}\.(?=\s+(?:Code|Rev|Stat|Comp|Gen)\b)"),
        re.compile(r"\b(?:Code|Rev|Stat|Comp|Gen)\.(?=\s+\d)"),
    )
    for pattern in abbreviation_patterns:
        for abbreviation in pattern.finditer(prefix):
            for index in range(abbreviation.start(), abbreviation.end()):
                if masked[index] == ".":
                    masked[index] = "\u2024"
    masked_text = "".join(masked)
    content_end = len(masked_text.rstrip())
    boundary_end = content_end
    if content_end and masked_text[content_end - 1] in ".!?":
        boundary_end -= 1
    clause_start = 0
    for boundary in re.finditer(
        r"[.!?]\s+(?=[A-Z\dÄÖÜ\"“])",
        masked_text[:boundary_end],
    ):
        clause_start = boundary.end()
    return prefix[clause_start:content_end]


def _flattened_inline_numeric_root_is_proven(
    source_text: str,
    match: re.Match[str],
) -> bool:
    """Require local structural evidence before restoring one inline root."""

    if _flattened_inline_numeric_root_has_explicit_reference_command(
        source_text,
        match,
    ):
        return False
    prefix = _flattened_inline_numeric_root_immediate_clause(source_text, match)
    if _INLINE_OUTLINE_STRUCTURAL_CHAPEAU_CONTEXT.search(prefix) and not (
        _INLINE_OUTLINE_CHAPEAU_REFERENCE_LEAD.search(prefix)
    ):
        return True
    has_reference_context = bool(
        _INLINE_OUTLINE_REFERENCE_CONTEXT.search(prefix)
        or _INLINE_OUTLINE_NAMED_REFERENCE_CONTEXT.search(prefix)
    )
    if not has_reference_context:
        return True
    stale_section_tail = re.search(
        r"\b(?:of|under)\s+this\s+section[.!?]\s*$",
        prefix,
        re.IGNORECASE,
    )
    if stale_section_tail is not None:
        return True
    return False


def _inline_outline_marker_has_reference_context(
    source_text: str,
    match: re.Match[str],
) -> bool:
    prefix = source_text[max(0, match.start("marker") - 128) : match.start("marker")]
    if _INLINE_OUTLINE_STRUCTURAL_CHAPEAU_CONTEXT.search(prefix) and not (
        _INLINE_OUTLINE_CHAPEAU_REFERENCE_LEAD.search(prefix)
    ):
        return False
    return bool(
        _INLINE_OUTLINE_REFERENCE_CONTEXT.search(prefix)
        or _INLINE_OUTLINE_NAMED_REFERENCE_CONTEXT.search(prefix)
    )


def _parenthesized_legal_outline_level(
    raw_label: str,
    *,
    active: Sequence[tuple[int, str, str, str]],
    attached_parent_level: int | None,
    outline_levels: Sequence[str],
) -> tuple[int, str]:
    categories = _parenthesized_legal_outline_categories(raw_label)
    candidate_levels = tuple(
        level for level, category in enumerate(outline_levels) if category in categories
    )
    if not candidate_levels:
        return 0, categories[0]

    if attached_parent_level is not None:
        level = next(
            (level for level in candidate_levels if level > attached_parent_level),
            candidate_levels[-1],
        )
        return level, outline_levels[level]

    continuing = [
        (level, category)
        for level, previous, _normalized, category in active
        if level in candidate_levels
        and _parenthesized_outline_label_follows(previous, raw_label, category)
    ]
    if continuing:
        return max(continuing, key=lambda item: item[0])

    parent_level = active[-1][0] if active else -1
    first_descending = [
        level
        for level in candidate_levels
        if level > parent_level
        and _parenthesized_outline_label_is_first(
            raw_label,
            outline_levels[level],
        )
    ]
    if first_descending:
        level = first_descending[0]
        return level, outline_levels[level]

    same_category = [
        (level, category)
        for level, _previous, _normalized, category in active
        if category in categories
    ]
    if same_category:
        return max(same_category, key=lambda item: item[0])

    descending = [level for level in candidate_levels if level > parent_level]
    if descending:
        level = descending[0]
        return level, outline_levels[level]

    level = min(candidate_levels, key=lambda item: abs(item - parent_level))
    return level, outline_levels[level]


def _parenthesized_legal_outline_categories(raw_label: str) -> tuple[str, ...]:
    if raw_label[0].isdigit():
        return ("numeric",)
    if raw_label.isupper():
        return ("upper-alpha",)
    categories = ["lower-alpha"]
    if _is_roman_outline_label(raw_label):
        categories.append("roman")
    return tuple(categories)


def _parenthesized_outline_label_follows(
    previous: str,
    current: str,
    category: str,
) -> bool:
    if category == "numeric":
        previous_match = re.fullmatch(r"(\d+)([a-z]?)", previous, re.IGNORECASE)
        current_match = re.fullmatch(r"(\d+)([a-z]?)", current, re.IGNORECASE)
        if previous_match is None or current_match is None:
            return False
        previous_number = int(previous_match.group(1))
        current_number = int(current_match.group(1))
        previous_suffix = previous_match.group(2).lower()
        current_suffix = current_match.group(2).lower()
        return (current_number == previous_number + 1 and not current_suffix) or (
            current_number == previous_number
            and len(current_suffix) == 1
            and (
                (not previous_suffix and current_suffix == "a")
                or (
                    len(previous_suffix) == 1
                    and ord(current_suffix) == ord(previous_suffix) + 1
                )
            )
        )
    if category in {"lower-alpha", "upper-alpha"}:
        return (
            len(previous) == len(current) == 1
            and ord(current.lower()) == ord(previous.lower()) + 1
        )
    if category == "roman":
        previous_value = _roman_outline_value(previous)
        current_value = _roman_outline_value(current)
        return previous_value is not None and current_value == previous_value + 1
    return False


def _parenthesized_outline_label_is_first(label: str, category: str) -> bool:
    if category == "numeric":
        match = re.fullmatch(r"(\d+)([a-z]?)", label, re.IGNORECASE)
        return match is not None and int(match.group(1)) == 1 and not match.group(2)
    if category in {"lower-alpha", "upper-alpha"}:
        return label.lower() == "a"
    if category == "roman":
        return _roman_outline_value(label) == 1
    return False


def _roman_outline_value(label: str) -> int | None:
    if not _is_roman_outline_label(label):
        return None
    values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    previous = 0
    for character in reversed(label.lower()):
        value = values[character]
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    return total


def _nested_outline_marker_spans(
    markers: Sequence[_OutlineMarker],
    *,
    outer_end: int,
) -> tuple[tuple[_OutlineMarker, int], ...]:
    """Close nested marker spans in one forward stack pass."""

    ends: dict[_OutlineMarker, int] = {}
    active: list[_OutlineMarker] = []
    for start, grouped in itertools.groupby(markers, key=lambda marker: marker.start):
        group = tuple(grouped)
        shallowest_depth = min(len(marker.path) for marker in group)
        while active and len(active[-1].path) >= shallowest_depth:
            ends[active.pop()] = start
        active.extend(group)
    while active:
        ends[active.pop()] = outer_end
    return tuple((marker, ends[marker]) for marker in markers)


def _is_parenthesized_outline_label(label: str) -> bool:
    return bool(
        re.fullmatch(r"\d+[a-z]?", label, flags=re.IGNORECASE)
        or re.fullmatch(r"[a-z]", label, flags=re.IGNORECASE)
        or _is_roman_outline_label(label)
    )


def _is_roman_outline_label(label: str) -> bool:
    return bool(_VALID_ROMAN_OUTLINE_LABEL.fullmatch(label))


def _dotted_subsection_boundary_matches(source_text: str) -> tuple[re.Match[str], ...]:
    """Return structural and joined rejected markers as ownership boundaries."""

    return tuple(
        heapq.merge(
            _DOTTED_SUBSECTION_MARKER.finditer(source_text),
            _JOINED_DOTTED_BOUNDARY_MARKER.finditer(source_text),
            key=lambda match: match.start(),
        )
    )


def _dotted_marker_starts_citation(source_text: str, match: re.Match[str]) -> bool:
    line_end = source_text.find("\n", match.end())
    if line_end < 0:
        line_end = len(source_text)
    remainder = source_text[match.end() : line_end].lstrip()
    return bool(re.match(r"\d+[A-Z]?(?::|\.\d)", remainder, flags=re.IGNORECASE))


def _dotted_marker_is_ignorable(source_text: str, match: re.Match[str]) -> bool:
    if _dotted_marker_starts_citation(source_text, match):
        return True
    line_end = source_text.find("\n", match.end())
    if line_end < 0:
        line_end = len(source_text)
    remainder = source_text[match.end() : line_end].lstrip()
    return match.group("label") == "I" and bool(
        re.match(r"e\.(?:\s|$)", remainder, flags=re.IGNORECASE)
    )


def _is_editorial_omission(text: str) -> bool:
    return bool(_editorial_omission_only().fullmatch(text))


def _normalize_formula_result_head(head: str) -> str:
    head = head.lower()
    if head in _FORMULA_NUMERIC_RESULT_HEADS:
        return head
    if head.endswith("ies") and head[:-3] + "y" in _FORMULA_NUMERIC_RESULT_HEADS:
        return head[:-3] + "y"
    if head.endswith("es") and head[:-2] in _FORMULA_NUMERIC_RESULT_HEADS:
        return head[:-2]
    if head.endswith("s") and head[:-1] in _FORMULA_NUMERIC_RESULT_HEADS:
        return head[:-1]
    return ""


def _formula_subject_segment_head(subject: str) -> str:
    """Return the grammatical head of one comma-delimited subject segment."""

    subject = _FORMULA_SUBJECT_PHRASE_BREAK.split(subject, maxsplit=1)[0]
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", subject)
    for index, word in enumerate(words):
        lowered = word.lower()
        if not lowered.endswith("ing"):
            continue
        preceding_has_head = any(
            preceding.lower() not in _FORMULA_SUBJECT_LEADING_NONHEAD_WORDS
            for preceding in words[:index]
        )
        following = words[index + 1 :]
        following_has_numeric_head = any(
            _normalize_formula_result_head(candidate) for candidate in following
        )
        if preceding_has_head and (
            lowered in _FORMULA_ADMINISTRATIVE_PARTICIPLE
            or not following_has_numeric_head
        ):
            words = words[:index]
            break
    while len(words) > 1 and (
        words[-1].lower() in _FORMULA_SUBJECT_TRAILING_MODIFIERS
        or words[-1].lower().endswith("ly")
    ):
        words.pop()
    return words[-1].lower() if words else ""


def _formula_result_subject_head(prefix: str) -> str:
    """Return the grammatical head of a formula predicate's subject phrase."""

    subject = _strip_source_clause_marker(prefix).strip()
    preamble = _FORMULA_SUBJECT_PREAMBLE_PREFIX.match(subject)
    if preamble is not None:
        subject = subject[preamble.end() :].strip()

    segments = [segment.strip() for segment in subject.split(",")]
    while segments and not segments[-1]:
        segments.pop()
    if subject.rstrip().endswith(","):
        while len(segments) > 1 and _FORMULA_SUBJECT_PARENTHETICAL.match(segments[-1]):
            segments.pop()
    selected = segments[-1] if segments else ""
    if not _normalize_formula_result_head(_formula_subject_segment_head(selected)):
        for segment in segments[:-1]:
            if not _FORMULA_SUBJECT_PHRASE_BREAK.search(segment):
                continue
            if _normalize_formula_result_head(_formula_subject_segment_head(segment)):
                selected = segment
                break
    return _formula_subject_segment_head(selected)


def _formula_operand_is_numeric(operand: str) -> bool:
    if _FORMULA_NUMERIC_OPERAND_LITERAL.fullmatch(operand):
        return True
    literal_operand = re.fullmatch(
        rf"\s*(?:(?:the\s+)?(?:first|next|each|every|any)\s+)?"
        rf"{_FORMULA_NUMERIC_OPERAND_LITERAL_CORE}"
        rf"(?:\s+of\s+(?P<base>.+))?\s*",
        operand,
        flags=re.IGNORECASE,
    )
    if literal_operand is not None:
        base = literal_operand.group("base")
        return base is None or _formula_operand_is_numeric(base)
    bracket_base = re.fullmatch(
        r"\s*(?:(?:the|that)\s+(?:part|portion)\s+of\s+(?P<portion>.+?)\s+"
        r"(?:over|above|exceeding|not\s+(?:exceeding|to\s+exceed)|"
        r"in\s+excess\s+of|(?:which|that)\s+"
        r"(?:equals\s+or\s+exceeds|exceeds\s+or\s+equals|exceeds|"
        r"(?:is\s+)?greater\s+than|does\s+not\s+exceed|"
        r"(?:is\s+)?(?:not|no)\s+more\s+than|(?:is\s+)?at\s+most|"
        r"(?:is\s+)?less\s+than(?:\s+or\s+equal\s+to)?))\s+"
        r"(?P<portion_limit>.+)|so\s+much\s+of\s+(?P<much>.+?)\s+as\s+"
        r"(?:equals\s+or\s+exceeds|exceeds\s+or\s+equals|exceeds|"
        r"is\s+in\s+excess\s+of|(?:is\s+)?greater\s+than|"
        r"does\s+not\s+exceed|(?:is\s+)?(?:not|no)\s+more\s+than|"
        r"(?:is\s+)?at\s+most|(?:is\s+)?less\s+than"
        r"(?:\s+or\s+equal\s+to)?)\s+"
        r"(?P<much_limit>.+)|(?:the\s+)?excess\s+of\s+(?P<excess>.+?)\s+"
        r"(?:over|above)\s+(?P<excess_limit>.+))\s*",
        operand,
        flags=re.IGNORECASE,
    )
    if bracket_base is not None:
        base = next(
            bracket_base.group(name)
            for name in ("portion", "much", "excess")
            if bracket_base.group(name) is not None
        )
        limit = next(
            bracket_base.group(name)
            for name in ("portion_limit", "much_limit", "excess_limit")
            if bracket_base.group(name) is not None
        )
        return _formula_operand_is_numeric(base) and _formula_operand_is_numeric(limit)
    if re.fullmatch(
        rf"\s*(?:the\s+)?{_FORMULA_TABLE_NAME}\s*",
        operand,
        flags=re.IGNORECASE,
    ):
        return True
    if re.match(
        r"\s*(?:the\s+)?(?:addition|average|difference|greater|greatest|higher|"
        r"highest|larger|largest|least|lesser|lower|lowest|max|maximum|mean|"
        r"median|min|minimum|product|quotient|ratio|remainder|smaller|smallest|"
        r"sum|total)\s+(?:of|between)\b",
        operand,
        flags=re.IGNORECASE,
    ):
        return _formula_operation_has_numeric_operands(operand)
    head = _formula_subject_segment_head(operand)
    if _normalize_formula_operand_head(head):
        return True
    operand_segment = _FORMULA_SUBJECT_PHRASE_BREAK.split(operand, maxsplit=1)[0]
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", operand_segment)
    for index, word in enumerate(words):
        normalized_word = _normalize_formula_operand_head(word)
        if not normalized_word:
            continue
        trailing = words[index + 1 :]
        if trailing:
            if all(
                candidate.lower().endswith(("ed", "en"))
                or candidate.lower().endswith("ly")
                or candidate.lower() in _FORMULA_OPERAND_IRREGULAR_POSTMODIFIERS
                for candidate in trailing
            ):
                return True
            if normalized_word in {"coefficient", "formula", "index"} and all(
                re.fullmatch(
                    r"(?:[A-Z0-9]+(?:-[A-Z0-9]+)*|[IVXLCDM]+)",
                    candidate,
                )
                is not None
                for candidate in trailing
            ):
                return True
    return False


def _normalize_formula_operand_head(head: str) -> str:
    head = head.lower()
    if head in _FORMULA_NUMERIC_OPERAND_HEADS:
        return head
    candidates = []
    if head.endswith("ies"):
        candidates.append(head[:-3] + "y")
    if head.endswith("es"):
        candidates.append(head[:-2])
    if head.endswith("s"):
        candidates.append(head[:-1])
    return next(
        (
            candidate
            for candidate in candidates
            if candidate in _FORMULA_NUMERIC_OPERAND_HEADS
        ),
        "",
    )


def _formula_aggregate_operands_are_numeric(operands: str) -> bool:
    comma_parts = [
        re.sub(r"^(?:and|or)\s+", "", part.strip(), flags=re.IGNORECASE)
        for part in re.split(r"\s*,\s*", operands)
        if part.strip()
    ]
    if len(comma_parts) >= 2:
        return all(_formula_operand_is_numeric(part) for part in comma_parts)
    conjunctions = list(re.finditer(r"\s+(?:and|or)\s+", operands, flags=re.IGNORECASE))
    for conjunction in conjunctions:
        left = operands[: conjunction.start()]
        right = operands[conjunction.end() :]
        if _formula_operand_is_numeric(left) and _formula_operand_is_numeric(right):
            return True
        left_words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", left)
        if (
            _formula_operand_is_numeric(right)
            and left_words
            and all(
                word.lower() in _FORMULA_SUBJECT_LEADING_NONHEAD_WORDS
                for word in left_words
            )
        ):
            return True
    return False


def _formula_inline_operands_are_numeric(operands: str) -> bool:
    operands = operands.strip().strip(".;:, ")
    semicolon_parts = [part.strip() for part in operands.split(";") if part.strip()]
    if len(semicolon_parts) >= 2:
        return all(_formula_operand_is_numeric(part) for part in semicolon_parts)
    return _formula_operand_is_numeric(operands) or (
        _formula_aggregate_operands_are_numeric(operands)
    )


def _truncate_following_operand_tail(tail: str) -> str:
    """Stop a following-values list before a proviso or sibling outline."""

    tail = re.split(r"\n(?=\s*[A-Z]\.)", tail, maxsplit=1)[0]
    proviso = re.search(
        rf";(?={_FORMULA_FOLLOWING_OPERAND_PROVISO.pattern})",
        tail,
        flags=re.IGNORECASE,
    )
    if proviso is not None:
        tail = tail[: proviso.start()]

    markers = tuple(
        re.finditer(
            r"(?m)^\s*(?:\((?P<paren>[^)]+)\)|(?P<number>\d+)\.)\s*",
            tail,
        )
    )
    if not markers:
        return tail
    first_is_numeric = bool(markers[0].group("number")) or bool(
        (markers[0].group("paren") or "")[0:1].isdigit()
    )
    for marker in markers[1:]:
        label = marker.group("number") or marker.group("paren") or ""
        if bool(label[0:1].isdigit()) != first_is_numeric:
            return tail[: marker.start()]
    return tail


def _applied_operation_match_is_numeric(text: str, match: re.Match[str]) -> bool:
    """Return whether one applying/application clause states numeric work."""

    prefix = re.sub(
        r"\b(?:(?:shall|must|may)\s+be|is|are)\s*$",
        "",
        text[: match.start()],
        flags=re.IGNORECASE,
    )
    if not _normalize_formula_result_head(_formula_result_subject_head(prefix)):
        return False
    operands = re.split(
        _FORMULA_INDEPENDENT_COORDINATE,
        match.group("operands"),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    target_match = re.search(r"\s+to\s+", operands, flags=re.IGNORECASE)
    applied = operands[: target_match.start()] if target_match is not None else operands
    target = operands[target_match.end() :] if target_match is not None else ""
    # A named formula or index must end in a bounded code identifier, including
    # hyphenated codes such as CPI-U.  Prose document titles remain
    # administrative even when they name a numeric target.  Coefficients also
    # permit one targeted proper name ("coefficient Alpha to income").
    named_match = re.search(
        r"\b(?P<head>coefficient|formula|index)\b(?P<tail>.*)\Z",
        applied,
        flags=re.IGNORECASE,
    )
    named_reference_is_numeric = False
    if named_match is not None:
        identifier_tail = named_match.group("tail").strip()
        if identifier_tail:
            administrative_title = _FORMULA_ADMINISTRATIVE_TITLE_LANGUAGE.search(
                identifier_tail
            )
            # Code length does not distinguish an identifier from a titled
            # administrative document.  Accept the complete bounded uppercase
            # code grammar, then reject semantic document-title heads above.
            bounded_code = re.fullmatch(r"[A-Z0-9]+(?:-[A-Z0-9]+)*", identifier_tail)
            code_segments = identifier_tail.split("-")
            administrative_segments = [
                segment
                for segment in code_segments
                if _FORMULA_ADMINISTRATIVE_TITLE_LANGUAGE.fullmatch(segment)
            ]
            semantic_document_segment = any(
                not _FORMULA_CODE_NAMESPACE_TITLE_LANGUAGE.fullmatch(segment)
                for segment in administrative_segments
            )
            title_qualifier_segment = any(
                _FORMULA_ADMINISTRATIVE_TITLE_QUALIFIER_LANGUAGE.fullmatch(segment)
                for segment in code_segments
            )
            numeric_code_tail = code_segments[-1].lower() in (
                _FORMULA_NUMERIC_RESULT_HEADS
                | {
                    "adjustment",
                    "coefficient",
                    "index",
                    "multiplier",
                    "scale",
                    "weight",
                }
            )
            # Ambiguous namespace atoms (RULE, SECTION, SCHEDULE, POLICY, TAX,
            # WORK) may participate in a multi-atom code.  An unambiguous
            # document atom such as MANUAL, REPORT, GUIDE, or WEBSITE wins over
            # every acronym, digit, and payload suffix; those signals cannot
            # turn a titled document into executable work.
            code_like = bool(
                bounded_code is not None
                and not semantic_document_segment
                and (not title_qualifier_segment or numeric_code_tail)
                and (not administrative_segments or len(code_segments) > 1)
            )
            targeted_coefficient_name = (
                named_match.group("head").lower() == "coefficient"
                and target_match is not None
                and (administrative_title is None or numeric_code_tail)
                and re.fullmatch(
                    r"[A-Z][A-Za-z0-9]*(?:-[A-Z][A-Za-z0-9]*)*",
                    identifier_tail,
                )
            )
            uppercase_coefficient_code = (
                named_match.group("head").lower() == "coefficient"
                and (administrative_title is None or numeric_code_tail)
                and re.fullmatch(
                    r"[A-Z0-9]+(?:-[A-Z0-9]+)*",
                    identifier_tail,
                )
            )
            if not (
                code_like or targeted_coefficient_name or uppercase_coefficient_code
            ):
                return False
            named_reference_is_numeric = True
    return (named_reference_is_numeric or _formula_operand_is_numeric(applied)) and (
        target_match is None or _formula_operand_is_numeric(target)
    )


def _formula_states_applied_operation(text: str) -> bool:
    """Recognize applying a numeric rate/formula, not administrative criteria."""

    return any(
        _applied_operation_match_is_numeric(text, match)
        for match in _FORMULA_APPLIED_OPERATION_LANGUAGE.finditer(text)
    )


def _without_unproven_applied_operations(text: str) -> str:
    """Mask administrative applying clauses from generic computation cues."""

    characters = list(text)
    for match in _FORMULA_APPLIED_OPERATION_LANGUAGE.finditer(text):
        if _applied_operation_match_is_numeric(text, match):
            continue
        characters[match.start() : match.end()] = " " * (match.end() - match.start())
    return "".join(characters)


def _formula_numeric_noun_phrase_head(phrase: str) -> str:
    """Return the normalized numeric result head of a bounded noun phrase."""

    phrase = re.sub(
        r"^(?:that|this|such)\s+",
        "the ",
        phrase.strip(),
        flags=re.IGNORECASE,
    )
    return _normalize_formula_result_head(_formula_subject_segment_head(phrase))


def _formula_rounding_noun_phrase_is_numeric(phrase: str) -> bool:
    """Recognize a numeric antecedent without admitting a titled document."""

    last_word = phrase.strip().rsplit(maxsplit=1)[-1]
    administrative_titles = tuple(
        _FORMULA_ADMINISTRATIVE_TITLE_LANGUAGE.finditer(phrase)
    )
    return bool(
        not _FORMULA_ADMINISTRATIVE_TITLE_LANGUAGE.fullmatch(last_word)
        and all(match.group().lower() == "tax" for match in administrative_titles)
        and _formula_numeric_noun_phrase_head(phrase)
    )


def _formula_rounding_actor_is_bounded(actor: str) -> bool:
    """Recognize a finite-clause actor while rejecting titled documents."""

    actor = actor.strip().lstrip(",").strip()
    if not actor:
        return True
    if re.fullmatch(r"(?:it|they)", actor, flags=re.IGNORECASE):
        return True
    determined_actor = re.fullmatch(
        r"(?:the|a|an|each|any|every|this|that|such)\s+"
        r"[A-Za-z][A-Za-z'-]*(?:\s+(?:(?:of|for)\s+(?:the\s+)?)?"
        r"[A-Za-z][A-Za-z'-]*){0,5}",
        actor,
        flags=re.IGNORECASE,
    )
    proper_actor = re.fullmatch(
        r"[A-Z][A-Za-z'-]*(?:\s+(?:(?:of|for)\s+(?:the\s+)?|and\s+)?"
        r"[A-Z][A-Za-z'-]*){0,5}",
        actor,
    )
    bare_actor = re.fullmatch(
        r"(?!.*\b(?:a|an|any|each|every|that|the|this)\b)"
        r"[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){0,5}",
        actor,
    )
    if not (determined_actor or proper_actor or bare_actor):
        return False
    actor_core = re.sub(
        r"^(?:the|a|an|each|any|every|this|that|such)\s+",
        "",
        actor,
        flags=re.IGNORECASE,
    )
    actor_starts_with_role = bool(
        _FORMULA_ROUNDING_ACTOR_ROLE_LANGUAGE.match(actor_core)
    )
    actor_last_word = actor_core.rsplit(maxsplit=1)[-1]
    actor_ends_with_role = bool(
        _FORMULA_ROUNDING_ACTOR_ROLE_LANGUAGE.fullmatch(actor_last_word)
    )
    actor_has_title = bool(_FORMULA_ADMINISTRATIVE_TITLE_LANGUAGE.search(actor))
    role_starts_scoped_actor = bool(
        actor_starts_with_role
        and re.match(
            r"^[A-Za-z][A-Za-z'-]*\s+(?:of|for)\b",
            actor_core,
            flags=re.IGNORECASE,
        )
    )
    if not actor_has_title:
        return True
    return bool(role_starts_scoped_actor or actor_ends_with_role)


def _formula_passive_rounding_modifier_is_bounded(modifier: str) -> bool:
    """Recognize bounded adverbial, actor, and legal passive modifiers."""

    adverbial_by_actor = re.fullmatch(
        r"[A-Za-z]+ly(?:\s+and\s+[A-Za-z]+ly)*\s+by\s+"
        r"(?P<actor>[^,.;:\n]{1,60})",
        modifier,
        flags=re.IGNORECASE,
    )
    if re.fullmatch(
        r"[A-Za-z]+ly(?:\s+and\s+[A-Za-z]+ly)*",
        modifier,
        flags=re.IGNORECASE,
    ):
        return True
    if adverbial_by_actor is not None:
        return _formula_rounding_actor_is_bounded(adverbial_by_actor.group("actor"))
    if modifier.lower().startswith("by "):
        return _formula_rounding_actor_is_bounded(modifier[3:])
    legal_reference = (
        r"(?:(?:the|this|that|such)\s+)?"
        r"[A-Za-z§][A-Za-z0-9 .():§'-]{0,60}"
    )
    legal_connector = (
        r"(?:according\s+to|consistent\s+with|for\s+(?:the\s+)?purposes?\s+of|"
        r"in\s+accordance\s+with|in\s+compliance\s+with|"
        r"in\s+conformity\s+with|per|subject\s+to|under|pursuant\s+to)"
    )
    return bool(
        re.fullmatch(
            rf"as\s+(?:otherwise\s+)?(?:authorized|described|directed|"
            rf"established|mandated|outlined|required|provided|specified|"
            rf"prescribed|stipulated)"
            rf"(?:\s+(?:by|in|under|{legal_connector})\s+{legal_reference})?",
            modifier,
            flags=re.IGNORECASE,
        )
        or re.fullmatch(
            rf"as\s+set\s+forth\s+in\s+{legal_reference}",
            modifier,
            flags=re.IGNORECASE,
        )
        or re.fullmatch(
            rf"as\s+provided\s+for(?:\s+in\s+{legal_reference})?",
            modifier,
            flags=re.IGNORECASE,
        )
        or re.fullmatch(
            rf"as\s+{legal_reference}\s+requires",
            modifier,
            flags=re.IGNORECASE,
        )
        or re.fullmatch(
            rf"in\s+(?:the\s+)?manner\s+(?:authorized|established|mandated|"
            rf"outlined|provided|required|specified|prescribed|stipulated)\s+"
            rf"(?:by|in|under)\s+"
            rf"{legal_reference}",
            modifier,
            flags=re.IGNORECASE,
        )
        or re.fullmatch(
            rf"in\s+(?:the\s+)?manner\s+set\s+forth\s+in\s+{legal_reference}",
            modifier,
            flags=re.IGNORECASE,
        )
        or re.fullmatch(
            rf"{legal_connector}\s+{legal_reference}",
            modifier,
            flags=re.IGNORECASE,
        )
    )


def _formula_passive_rounding_actor(tail: str) -> str | None:
    """Separate a passive determination modifier from the rounding actor."""

    tail = tail.strip()
    if not tail:
        return ""
    if _formula_passive_rounding_modifier_is_bounded(tail):
        return ""
    if "," in tail:
        modifier, actor = tail.rsplit(",", maxsplit=1)
        modifier = modifier.strip()
        if modifier and not _formula_passive_rounding_modifier_is_bounded(modifier):
            return None
        return actor.strip()
    adverbial_actor = re.fullmatch(
        r"[A-Za-z]+ly\s+(?P<actor>[^,.;:\n]{1,60})",
        tail,
        flags=re.IGNORECASE,
    )
    if adverbial_actor is not None:
        return adverbial_actor.group("actor")
    return tail


def _formula_rounding_antecedent_is_numeric(prefix: str) -> bool:
    """Recognize a bounded numeric antecedent before a rounding pronoun."""

    gerund = re.search(
        r"\b(?:after|before|once|upon|when)\s+"
        r"(?:calculating|computing|determining)\s+"
        r"(?P<body>[^.;:\n]{1,120})$",
        prefix,
        flags=re.IGNORECASE,
    )
    passive = re.search(
        r"\b(?:after|before|once|upon|when)\s+"
        r"(?P<noun_phrase>[^,.;:\n]{1,80}?)\s+"
        r"(?:(?:is|are)\s+|(?:has|have|had)\s+been\s+)"
        r"(?:calculated|computed|determined)"
        r"(?P<tail>\s*[^.;:\n]{0,100})$",
        prefix,
        flags=re.IGNORECASE,
    )
    if passive is not None:
        passive_actor = _formula_passive_rounding_actor(passive.group("tail"))
        if (
            passive_actor is not None
            and _formula_rounding_noun_phrase_is_numeric(passive.group("noun_phrase"))
            and _formula_rounding_actor_is_bounded(passive_actor)
        ):
            return True
    if gerund is None:
        return False
    body = gerund.group("body").strip()
    if "," in body:
        body = body.split(",", maxsplit=1)[0].strip()
        return _formula_rounding_noun_phrase_is_numeric(body)
    # Without punctuation, find a numeric result phrase followed by any
    # bounded finite-clause actor.  This avoids hard-coding agency titles such
    # as department, commissioner, secretary, or board.
    words = body.split()
    for split_at in range(len(words) - 1, 0, -1):
        noun_phrase = " ".join(words[:split_at])
        actor = " ".join(words[split_at:])
        if _formula_rounding_actor_is_bounded(
            actor
        ) and _formula_rounding_noun_phrase_is_numeric(noun_phrase):
            return True
    return False


def _formula_nonnegative_floor_tail_start(clause: str) -> int | None:
    """Return the final coordinated zero-floor start for a numeric result."""

    connectors = tuple(
        re.finditer(
            r"(?:,\s*)?\bbut\b|\s+\band\b|,\s*\bwhich\b",
            clause,
            flags=re.IGNORECASE,
        )
    )
    for connector in reversed(connectors):
        tail = clause[connector.end() :].strip()
        if re.fullmatch(
            _FORMULA_NONNEGATIVE_FLOOR_CONTROL,
            tail,
            flags=re.IGNORECASE,
        ) or re.fullmatch(
            rf"it\s+{_FORMULA_NONNEGATIVE_FLOOR_CONTROL}",
            tail,
            flags=re.IGNORECASE,
        ):
            return connector.start()
        subject_floor = re.fullmatch(
            rf"(?P<subject>(?:the\s+)?[A-Za-z][A-Za-z '-]{{0,40}}?)\s+"
            rf"{_FORMULA_NONNEGATIVE_FLOOR_CONTROL}",
            tail,
            flags=re.IGNORECASE,
        )
        if subject_floor is not None and _formula_numeric_noun_phrase_head(
            subject_floor.group("subject")
        ):
            return connector.start()
    return None


def _rounding_language_is_computational(text: str) -> bool:
    """Require a numeric subject, operation, or precision for rounding prose."""

    for match in _ROUNDING_LANGUAGE.finditer(text):
        if not re.match(r"round", match.group(), flags=re.IGNORECASE):
            return True
        rounding_word = match.group().lower()
        active_target = None
        if (
            rounding_word == "round"
            or rounding_word.startswith("rounding")
            or (rounding_word.startswith("round "))
        ):
            clause_start = max(
                text.rfind(".", 0, match.start()),
                text.rfind(";", 0, match.start()),
                text.rfind("\n", 0, match.start()),
            )
            active_prefix = _strip_source_clause_marker(
                text[clause_start + 1 : match.start()]
            ).strip()
            modal_directive = re.search(
                r"\b(?:can|may|must|shall|should|will)\b"
                r"(?:(?:\s+(?:and\s+)?(?:then|\w+ly))|"
                r"(?:\s+(?:as\s+necessary|from\s+time\s+to\s+time|"
                r"without\s+delay|in\s+all\s+cases))|"
                r"(?:\s*,\s*[^,]{1,80},))*\s*$",
                active_prefix,
                flags=re.IGNORECASE,
            )
            to_directive = re.search(
                r"\b(?:is|are)\s+to\s*$",
                active_prefix,
                flags=re.IGNORECASE,
            )
            conditional_directive = re.fullmatch(
                r"(?:after|before|for|if|once|unless|upon|when|where)\b"
                r"[^.;:\n]{0,120},\s*"
                r"(?:(?:and\s+)?(?:then|\w+ly)\s*)*",
                active_prefix,
                flags=re.IGNORECASE,
            )
            coordinated_directive = re.search(
                r"\b(?:can|may|must|shall|should|will)\s+"
                r"(?:(?:then|thereafter|[A-Za-z]+ly|first|next)\s+)*"
                r"(?:calculate|compute|determine)\s+"
                r"(?P<noun_phrase>[^,.;:\n]{1,80}?)\s*,?\s+and"
                r"(?:\s+(?:then|thereafter|[A-Za-z]+ly|first|next))*"
                r"(?:\s+(?:can|may|must|shall|should|will)"
                r"(?:\s+(?:then|thereafter|[A-Za-z]+ly|first|next))*)?\s*$",
                re.sub(r"\s*,\s*", " ", active_prefix),
                flags=re.IGNORECASE,
            )
            shared_object_directive = re.search(
                r"\b(?:can|may|must|shall|should|will)\s+"
                r"(?:(?:then|thereafter|[A-Za-z]+ly|first|next)\s+)*"
                r"(?:calculate|compute|determine)\s*,?\s+and"
                r"(?:\s+(?:then|thereafter|[A-Za-z]+ly|first|next))*"
                r"(?:\s+(?:can|may|must|shall|should|will)"
                r"(?:\s+(?:then|thereafter|[A-Za-z]+ly|first|next))*)?\s*$",
                re.sub(r"\s*,\s*", " ", active_prefix),
                flags=re.IGNORECASE,
            )
            if active_prefix and not (
                modal_directive
                or to_directive
                or conditional_directive
                or coordinated_directive
                or shared_object_directive
            ):
                continue
            target_match = re.match(
                r"\s*(?:,\s*[^,]{1,80},\s*)?"
                r"(?P<target>[^.;\n]{1,120}?)(?=[.;\n]|$)",
                text[match.end() :],
                flags=re.IGNORECASE,
            )
            if target_match is None:
                continue
            active_target = re.sub(
                r"^(?:down(?:ward)?|up(?:ward)?|off)\s+",
                "",
                target_match.group("target").strip(),
                flags=re.IGNORECASE,
            )
            active_target = re.split(
                r"\s+\b(?:is|are|shall|must|may)\b",
                active_target,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
            active_target = re.sub(
                r"\s+(?:down(?:ward)?|up(?:ward)?|off)\s*$",
                "",
                active_target,
                flags=re.IGNORECASE,
            )
            active_target = re.sub(
                r",?\s*(?:as|if|unless|when|whenever|where)\b[^,.;:\n]*$",
                "",
                active_target,
                flags=re.IGNORECASE,
            ).strip()
        if re.match(r"\s+out\b", text[match.end() :], flags=re.IGNORECASE):
            continue
        if active_target is not None:
            if _formula_operand_is_numeric(active_target) or (
                _normalize_formula_result_head(
                    _formula_subject_segment_head(active_target)
                )
            ):
                return True
            if re.fullmatch(r"(?:it|them)", active_target, flags=re.IGNORECASE):
                prefix_without_modal = re.sub(
                    r"\b(?:can|may|must|shall|should|will)\b"
                    r"(?:(?:\s+(?:and\s+)?(?:then|\w+ly))|"
                    r"(?:\s+(?:as\s+necessary|from\s+time\s+to\s+time|"
                    r"without\s+delay|in\s+all\s+cases))|"
                    r"(?:\s*,\s*[^,]{1,80},))*\s*$",
                    "",
                    active_prefix,
                    flags=re.IGNORECASE,
                )
                antecedent_head = _normalize_formula_result_head(
                    _formula_result_subject_head(prefix_without_modal)
                )
                coordinated_antecedent_head = (
                    _formula_numeric_noun_phrase_head(
                        re.sub(
                            r"\s+(?:first|next|then|thereafter|[A-Za-z]+ly)\s*$",
                            "",
                            coordinated_directive.group("noun_phrase"),
                            flags=re.IGNORECASE,
                        )
                    )
                    if coordinated_directive is not None
                    else ""
                )
                if (
                    antecedent_head
                    or _formula_rounding_antecedent_is_numeric(prefix_without_modal)
                    or coordinated_antecedent_head
                ):
                    return True
        clause_start = max(
            text.rfind(".", 0, match.start()),
            text.rfind(";", 0, match.start()),
            text.rfind("\n", 0, match.start()),
        )
        clause_end_match = re.search(r"[.;\n]", text[match.end() :])
        clause_end = (
            match.end() + clause_end_match.start()
            if clause_end_match is not None
            else len(text)
        )
        clause = text[clause_start + 1 : clause_end]
        prefix = re.sub(
            r"\b(?:(?:shall|must|may|should|will)\s+be|"
            r"(?:is|are)(?:\s+required)?\s+to\s+be|is|are)\s*$",
            "",
            text[clause_start + 1 : match.start()],
            flags=re.IGNORECASE,
        )
        passive_pronoun = re.fullmatch(
            r"(?P<antecedent>.+?)(?:,\s*|\s+)(?:it|they)\s*",
            prefix,
            flags=re.IGNORECASE,
        )
        if passive_pronoun is not None and _formula_rounding_antecedent_is_numeric(
            passive_pronoun.group("antecedent")
        ):
            return True
        if _normalize_formula_result_head(_formula_result_subject_head(prefix)):
            return True
        if _formula_operation_has_numeric_operands(clause):
            return True
        if re.search(
            r"\b(?:down|up|to\s+the\s+nearest)\b[^.;\n]{0,80}"
            r"\b(?:dollars?|cents?|percent(?:age)?(?:\s+points?)?|hours?|units?)\b",
            text[match.start() : clause_end],
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _formula_operation_has_numeric_operands(text: str) -> bool:
    """Require actual numeric operand heads for a noun-form operation."""

    without_parentheticals = re.sub(
        r",\s*(?:(?:as|adjusted|calculated|computed|described|determined|"
        r"provided|subject|whether)\s+[^,]+|if\s+[^,]+|net\s+of\s+[^,]+|"
        r"including\s+[^,]+|excluding\s+[^,]+)\s*,",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    direct_following = _FORMULA_DIRECT_FOLLOWING_OPERANDS.search(without_parentheticals)
    if direct_following is not None:
        tail = without_parentheticals[direct_following.end() :].lstrip()
        if tail.lstrip().startswith(":"):
            tail = tail.lstrip()[1:]
        elif tail.startswith(","):
            tail = tail[1:]
        tail = _truncate_following_operand_tail(tail)
        marker = re.search(r"(?m)^\s*(?:\([^)]+\)|\d+\.)\s*", tail)
        if marker is not None:
            entries = [
                entry.strip()
                for entry in re.split(
                    r"(?m)^\s*(?:\([^)]+\)|\d+\.)\s*",
                    tail[marker.start() :],
                )
                if entry.strip()
            ]
            return bool(entries) and all(
                source_states_explicit_computation(entry)
                or _formula_inline_operands_are_numeric(entry)
                for entry in entries
            )
        inline = re.split(r"\n(?=\s*[A-Z]\.)", tail, maxsplit=1)[0]
        inline = inline.strip().strip(".;:, ")
        if inline:
            return _formula_inline_operands_are_numeric(inline)
        return True

    clause = re.split(r"[.;:\n]", without_parentheticals, maxsplit=1)[0]
    floor_tail_start = _formula_nonnegative_floor_tail_start(clause)
    if floor_tail_start is not None:
        clause = clause[:floor_tail_start]
    clause = re.sub(
        r",\s*(?:whichever[^,]{0,80}\b(?:is|shall\s+be|may\s+be|would\s+be)\s+[^,]+|as\s+applicable|"
        r"if\s+applicable|"
        r"(?:in\s+)?no\s+case\s+(?:less\s+than|below)\s+"
        r"(?:zero|\$?\s*0(?:\.0+)?)|except\s+that\s+it\s+"
        r"(?:shall|must|may)\s+not\s+be\s+less\s+than\s+"
        r"(?:zero|\$?\s*0(?:\.0+)?)|with\s+a\s+minimum\s+of\s+"
        r"(?:zero|\$?\s*0(?:\.0+)?)|subject\s+to\s+a\s+floor\s+of\s+"
        r"(?:zero|\$?\s*0(?:\.0+)?))\s*$",
        "",
        clause,
        flags=re.IGNORECASE,
    )
    if re.search(
        r"\b(?:addition|average|difference|greater|greatest|higher|highest|"
        r"larger|largest|least|lesser|lower|lowest|max|maximum|mean|median|min|"
        r"minimum|product|quotient|ratio|remainder|smaller|smallest|sum|total)"
        r"\s+(?:of|between)\s*$",
        clause,
        flags=re.IGNORECASE,
    ):
        return True
    remainder = re.search(
        r"\bremainder\s+of\s+(?P<left>.+?)\s+after\s+(?P<right>.+)$",
        clause,
        flags=re.IGNORECASE,
    )
    if remainder is not None:
        return _formula_operand_is_numeric(
            remainder.group("left")
        ) and _formula_operand_is_numeric(remainder.group("right"))

    aggregate = re.search(
        r"\b(?P<operation>addition|average|greater|greatest|higher|highest|largest|least|"
        r"larger|lesser|lower|lowest|max|maximum|mean|median|min|minimum|product|"
        r"remainder|smaller|smallest|sum|total|difference)\s+of\s+"
        r"(?P<operands>.+)$",
        clause,
        flags=re.IGNORECASE,
    )
    if aggregate is not None:
        operands = aggregate.group("operands")
        if _formula_aggregate_operands_are_numeric(operands):
            return True
        return aggregate.group("operation").lower() in {
            "addition",
            "average",
            "mean",
            "median",
            "sum",
            "total",
        } and _formula_operand_is_numeric(operands)

    patterns = (
        r"\bdifference\s+between\s+(?P<left>.+?)\s+and\s+(?P<right>.+)$",
        r"\b(?:quotient|ratio)\s+between\s+(?P<left>.+?)\s+and\s+"
        r"(?P<right>.+)$",
        r"\b(?:quotient|ratio)\s+of\s+(?P<left>.+?)\s+"
        r"(?:and|by|to)\s+(?P<right>.+)$",
        r"\b(?:decrease|division|increase|multiplication|reduction)\s+of\s+"
        r"(?P<left>.+?)\s+by\s+(?P<right>.+)$",
        r"\b(?:deduction|subtraction)\s+of\s+(?P<left>.+?)\s+from\s+"
        r"(?P<right>.+)$",
    )
    for pattern in patterns:
        operation = re.search(pattern, clause, flags=re.IGNORECASE)
        if operation is not None:
            right = operation.group("right").split(",", maxsplit=1)[0]
            return _formula_operand_is_numeric(
                operation.group("left")
            ) and _formula_operand_is_numeric(right)
    unary = re.search(
        r"\b(?:percent|percentage)\s+of\s+(?P<operand>.+)$",
        clause,
        flags=re.IGNORECASE,
    )
    if unary is not None:
        operand = unary.group("operand").split(",", maxsplit=1)[0]
        return _formula_operand_is_numeric(operand)
    return False


def _formula_states_contextual_operator(
    source_text: str,
    *,
    include_unconditional: bool = True,
) -> bool:
    """Recognize result operations only when their grammatical subject is numeric."""

    if include_unconditional and _FORMULA_UNCONDITIONAL_OPERATOR_LANGUAGE.search(
        source_text
    ):
        return True
    # This closed statutory instruction names both numeric credit operands.
    # Its claimant subject does not supply the generic numeric result head.
    if re.search(
        _STATUTORY_CREDIT_DIFFERENCE_CONSEQUENCE + r"(?=\s*(?:[.;]|$))",
        " ".join(source_text.split()),
        flags=re.IGNORECASE,
    ):
        return True
    for pattern in (
        _FORMULA_RESULT_OPERATION_LANGUAGE,
        _FORMULA_PARTICIPIAL_RESULT_LANGUAGE,
    ):
        for match in pattern.finditer(source_text):
            prefix = source_text[: match.start()]
            if pattern is _FORMULA_PARTICIPIAL_RESULT_LANGUAGE and re.search(
                r"\bbefore(?:\s+\w+){0,3}\s*$",
                prefix,
                flags=re.IGNORECASE,
            ):
                continue
            head = _normalize_formula_result_head(_formula_result_subject_head(prefix))
            if head and (
                pattern is _FORMULA_PARTICIPIAL_RESULT_LANGUAGE
                or _formula_operation_has_numeric_operands(source_text[match.start() :])
            ):
                return True
    return False


def _english_fraction_target_is_duration(
    target: str, *, numeric_result_subject: bool = False
) -> bool:
    """Recognize a short duration noun phrase without swallowing its later base."""

    duration = re.match(
        r"^\s*(?:(?!(?:as|for|from|in|of|over|under)\b)"
        r"[A-Za-z0-9$€£'’.-]+\s+){0,8}"
        r"(?P<unit>seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b"
        r"(?P<tail>.*)$",
        target,
        flags=re.IGNORECASE,
    )
    if duration is None:
        return False
    tail = duration.group("tail").strip()
    if not tail:
        return True
    if re.match(r"^of\b", tail, flags=re.IGNORECASE) is None:
        return False
    return not (numeric_result_subject and duration.group("unit").lower().endswith("s"))


def _english_fraction_of_is_computational(source_text: str) -> bool:
    """Require a numeric target and reject ordinary fractional durations."""

    for match in _ENGLISH_FRACTION_OF.finditer(source_text):
        target = re.split(
            _FORMULA_INDEPENDENT_COORDINATE,
            match.group("target"),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        target = target.split(",", maxsplit=1)[0].strip()
        prefix = source_text[: match.start()]
        result_predicates = tuple(
            re.finditer(rf"\b{_FORMULA_RESULT_PREDICATE}\b", prefix, re.IGNORECASE)
        )
        result_predicate = result_predicates[-1] if result_predicates else None
        intervening = (
            prefix[result_predicate.end() :].strip() if result_predicate else ""
        )
        predicate_directly_governs_fraction = result_predicate is not None and (
            not intervening
            or bool(
                re.fullmatch(
                    rf"(?:a|an|{_ENGLISH_CARDINAL_PHRASE})",
                    intervening,
                    flags=re.IGNORECASE,
                )
            )
        )
        subject = prefix[: result_predicate.start()] if result_predicate else prefix
        numeric_result_subject = predicate_directly_governs_fraction and bool(
            _formula_result_subject_head(subject)
        )
        if _english_fraction_target_is_duration(
            target, numeric_result_subject=numeric_result_subject
        ):
            continue
        if _formula_operand_is_numeric(target):
            return True
    return False


_FNS_FY2026_INCOME_STANDARDS_PAGE_TWO_PATH = (
    "us/guidance/usda/fns/snap-fy2026-income-eligibility-standards/page-2"
)
_FNS_FY2026_INCOME_STANDARDS_PAGE_TWO_SHA256 = (
    "f5d38b6b863f107f0c0e9a4c08ed8443bf4f3432b624860e66c1a899d251bcfc"
)
_FNS_FY2026_INCOME_STANDARDS_COMPOSED_PATH = (
    "us/guidance/usda/fns/snap-fy2026-income-eligibility-standards"
)
_FNS_FY2026_INCOME_STANDARDS_COMPOSED_SHA256 = (
    "4145775814ee480173e064c402eb67fc270c05227073b93f23a3e200ddda23e7"
)


def _without_precomputed_income_table_percentage_captions(
    source_text: str, *, corpus_citation_path: str = ""
) -> str:
    """Mask only descriptive rates above a complete published income table.

    The table supplies independently rounded amounts, not an instruction to
    recompute them from another table. Keep all source bytes elsewhere available
    to numeric recall and proof validation; this view is only for computation
    classification. Incomplete or prose-shaped tables remain conservative.
    """

    amount = r"\$\d+(?:,\d{3})*(?:\.\d+)?"
    three_amounts = rf"{amount}\s+{amount}\s+{amount}"
    rows = r"\s+".join(rf"{size}\s+{three_amounts}" for size in range(1, 9))
    captioned_table = re.compile(
        r"^[ \t]*(?:Net|Gross) Monthly Income Eligibility Standards"
        r"(?: for Households Where Elderly Disabled Are a Separate Household)?"
        r"[ \t]+(?P<caption>\(\d+(?:\.\d+)? Percent of Poverty Level\))"
        r"[ \t]*\r?\n\s*Household Size 48 States, DC, Guam, Virgin Islands Alaska Hawaii"
        rf"\s+{rows}\s+Each additional person\s+{three_amounts}[ \t]*(?=\r?$)",
        flags=re.MULTILINE | re.IGNORECASE,
    )
    flattened_captioned_table = re.compile(
        r"(?:Net|Gross) Monthly Income Eligibility Standards"
        r"(?: for Households Where Elderly Disabled Are a Separate Household)?"
        r"\s+(?P<caption>\(\d+(?:\.\d+)? Percent of Poverty Level\))"
        r"\s+Household Size 48 States, D\.C\., Guam, Virgin Islands Alaska Hawaii"
        rf"\s+{rows}\s+Each additional member\s+{three_amounts}",
        flags=re.IGNORECASE,
    )
    masked = list(source_text)
    for pattern in (captioned_table, flattened_captioned_table):
        for match in pattern.finditer(source_text):
            start, end = match.span("caption")
            masked[start:end] = " " * (end - start)
    if (
        corpus_citation_path == _FNS_FY2026_INCOME_STANDARDS_PAGE_TWO_PATH
        and hashlib.sha256(source_text.encode()).hexdigest()
        == _FNS_FY2026_INCOME_STANDARDS_PAGE_TWO_SHA256
    ):
        # This signed, complete FNS page is flattened into one OCR line. Its
        # title describes independently published dollar cells; neither the
        # 165% caption nor "Elderly/Disabled" is an arithmetic instruction.
        # Exact source identity fails closed if the page changes or loses a row.
        title = re.search(
            r"Table 3: Gross Monthly Income Limit Where Elderly/Disabled "
            r"Members Are a Separate Household \(165% of Federal Poverty Level\)",
            source_text,
        )
        if title is not None:
            start, end = title.span()
            masked[start:end] = " " * (end - start)
    if (
        corpus_citation_path == _FNS_FY2026_INCOME_STANDARDS_COMPOSED_PATH
        and hashlib.sha256(source_text.encode()).hexdigest()
        == _FNS_FY2026_INCOME_STANDARDS_COMPOSED_SHA256
    ):
        # The signed composed source joins three independently published dollar
        # tables into a single OCR-flattened paragraph. Their percentage titles
        # describe the supplied cells, not an instruction to recompute them.
        captions = (
            "Table 1: Net Monthly Income Limit (100% of Federal Poverty Level)",
            "Table 2: Gross Monthly Income Limit (130% of Federal Poverty Level)",
            "Table 3: Gross Monthly Income Limit Where Elderly/Disabled "
            "Members Are a Separate Household (165% of Federal Poverty Level)",
        )
        matches = tuple(
            re.search(re.escape(caption), source_text) for caption in captions
        )
        if all(match is not None for match in matches):
            for match in matches:
                assert match is not None
                start, end = match.span()
                masked[start:end] = " " * (end - start)
    return "".join(masked)


def _imperative_arithmetic_operand_is_numeric(operand: str) -> bool:
    """Admit quantities and exact row references, never identifier numbers."""

    if re.search(
        r"\b(?:if|unless|except|when|where|provided|only|and|or|but|then)\b"
        r"|[,;][ \t]*[A-Za-z]",
        operand,
        flags=re.IGNORECASE,
    ):
        return False
    if re.fullmatch(
        r"(?:the[ \t]+amount[ \t]+(?:on|from)[ \t]+)?"
        r"(?:line|row)[ \t]+[1-9]\d{0,3}",
        operand.strip(),
        flags=re.IGNORECASE,
    ):
        return True
    # A trailing "number" can name an identifier, even after a monetary word
    # ("tax number", "benefit number"). Counts must state what is counted.
    if re.search(r"\bnumbers?[ \t]*$", operand, flags=re.IGNORECASE):
        return False
    if re.search(
        r"\b(?:telephone|phone|fax|social[ \t]+insurance|account|identification|"
        r"registration|reference|form|document|application|contact|section|page)"
        r"[ \t]+numbers?\b",
        operand,
        flags=re.IGNORECASE,
    ):
        return False
    return _formula_operand_is_numeric(operand)


def _source_imperative_arithmetic_operations(
    source_text: str,
) -> tuple[_SourceImperativeArithmeticOperation, ...]:
    """Recognize four complete binary imperatives without truncating context."""

    return _cached_source_imperative_arithmetic_operations(
        source_text, hebrew_script=_hebrew_script()
    )


@functools.lru_cache(maxsize=4096)
def _cached_source_imperative_arithmetic_operations(
    source_text: str,
    *,
    hebrew_script: bool,
) -> tuple[_SourceImperativeArithmeticOperation, ...]:
    """`_source_imperative_arithmetic_operations`, cached per wall gate.

    The body reaches the Hebrew readers, which read the gate themselves;
    ``hebrew_script`` keys the cache on it, so a result read behind an open
    gate never answers a candidate outside it (#1882).
    """

    del hebrew_script

    quoted = _annual_source_quoted_spans(source_text)
    if quoted is None:
        return ()
    operations: list[_SourceImperativeArithmeticOperation] = []
    pattern = (
        r"(?:^|(?<=[.;!?]))[ \t\r\n]*"
        r"(?P<verb>Add|Subtract|Multiply|Divide)[ \t]+"
        r"(?P<first>(?:\.(?=\d)|[^.;!?\n]){1,160}?)[ \t]+"
        r"(?P<connector>to|from|by)[ \t]+"
        r"(?P<second>(?:\.(?=\d)|[^.;!?\n]){1,160}?)[ \t]*"
        r"(?=\.(?!\d)|[;!?\n]|$)"
    )
    connectors = {"add": "to", "subtract": "from", "multiply": "by", "divide": "by"}
    for match in re.finditer(pattern, source_text, flags=re.IGNORECASE):
        kind = match.group("verb").lower()
        if match.group("connector").lower() != connectors[kind]:
            continue
        if any(start <= match.start("verb") < end for start, end in quoted):
            continue
        if not all(
            _imperative_arithmetic_operand_is_numeric(match.group(group))
            for group in ("first", "second")
        ):
            continue
        first, second = match.span("first"), match.span("second")
        left, right = (second, first) if kind == "subtract" else (first, second)
        operations.append(
            _SourceImperativeArithmeticOperation(
                kind, match.start("verb"), second[1], first, second, left, right
            )
        )
    return tuple(operations)


def source_states_explicit_computation(
    source_text: str, *, corpus_citation_path: str = ""
) -> bool:
    """Return whether text states a computation rather than only a scalar."""

    # A parenthesized cross-reference can contain title punctuation that looks
    # arithmetic to the generic expression recognizer (for example
    # ``(See Elderly or Have a Disability - NA Special Considerations)``).
    # It points to another source unit; it does not itself direct a computation.
    if re.fullmatch(
        r"\s*\(\s*(?:see(?:\s+also)?|refer\s+to)\b[^()]{1,300}\)\s*[.!]?\s*",
        source_text,
        flags=re.IGNORECASE,
    ):
        return False

    computation_text = _without_unproven_applied_operations(
        _without_stated_conversion_results(
            _without_precomputed_income_table_percentage_captions(
                source_text, corpus_citation_path=corpus_citation_path
            )
        )
    )
    return bool(
        _has_substantive_arithmetic_expression(computation_text)
        or _source_imperative_arithmetic_operations(computation_text)
        or _COMPUTATION_LANGUAGE.search(computation_text)
        or _EXCESS_BOUND_TAX_RATE_LANGUAGE.search(computation_text)
        or _ENGLISH_WORDED_PERCENTAGE_OF.search(computation_text)
        or _EXPLICIT_NUMERIC_PERCENTAGE_OF.search(computation_text)
        or _english_fraction_of_is_computational(computation_text)
        or _ENGLISH_FRACTIONAL_PERCENTAGE_OF.search(computation_text)
        or _FORMULA_COMPUTED_OPERATION_LANGUAGE.search(computation_text)
        or _formula_states_applied_operation(computation_text)
        or _rounding_language_is_computational(computation_text)
        or _formula_states_contextual_operator(computation_text)
    )


def _source_states_nonrounding_computation(
    source_text: str, *, corpus_citation_path: str = ""
) -> bool:
    """Return whether text states a computation other than rounding."""

    computation_text = _without_unproven_applied_operations(
        _without_stated_conversion_results(
            _without_precomputed_income_table_percentage_captions(
                source_text, corpus_citation_path=corpus_citation_path
            )
        )
    )
    return bool(
        _has_substantive_arithmetic_expression(computation_text)
        or _source_imperative_arithmetic_operations(computation_text)
        or _COMPUTATION_LANGUAGE.search(computation_text)
        or _EXCESS_BOUND_TAX_RATE_LANGUAGE.search(computation_text)
        or _ENGLISH_WORDED_PERCENTAGE_OF.search(computation_text)
        or _EXPLICIT_NUMERIC_PERCENTAGE_OF.search(computation_text)
        or _english_fraction_of_is_computational(computation_text)
        or _ENGLISH_FRACTIONAL_PERCENTAGE_OF.search(computation_text)
        or _FORMULA_COMPUTED_OPERATION_LANGUAGE.search(computation_text)
        or _formula_states_applied_operation(computation_text)
        or _formula_states_contextual_operator(computation_text)
    )


def source_states_stated_conversion_result(source_text: str) -> bool:
    """Return whether text states both a base scalar and its converted result."""

    return bool(_stated_conversion_result_spans(source_text))


def _stated_conversion_result_spans(source_text: str) -> tuple[tuple[int, int], ...]:
    """Locate result clauses such as ``Umgerechnet ... ergibt sich 6 150``."""

    spans: list[tuple[int, int]] = []
    for cue in _STATED_CONVERSION_CUE.finditer(source_text):
        result = _STATED_CONVERSION_RESULT.search(
            source_text,
            cue.end(),
            min(len(source_text), cue.end() + 180),
        )
        if result is None:
            continue
        if _stated_conversion_candidate_contains_formula(source_text, cue, result):
            continue
        base_window = source_text[max(0, cue.start() - 300) : cue.start()]
        paragraph_break = base_window.rfind("\n\n")
        if paragraph_break >= 0:
            base_window = base_window[paragraph_break + 2 :]
        base_values = list(_STATED_CONVERSION_BASE_VALUE.finditer(base_window))
        substantive_base = next(
            (
                match
                for match in reversed(base_values)
                if not _looks_like_temporal_or_structural_number(base_window, match)
            ),
            None,
        )
        if substantive_base is None:
            continue
        spans.append((cue.start(), result.end("value")))
    return tuple(spans)


def _stated_conversion_candidate_contains_formula(
    source_text: str,
    cue: re.Match[str],
    result: re.Match[str],
) -> bool:
    """Keep arithmetic inside a conversion clause out of the scalar exemption."""

    start = cue.start()
    suffix_limit = min(len(source_text), result.end("value") + 160)
    clause_suffix = source_text[result.end("value") : suffix_limit]
    clause_boundary = re.search(r"[.;!?\n]", clause_suffix)
    end = (
        result.end("value") + clause_boundary.start()
        if clause_boundary is not None
        else suffix_limit
    )
    characters = list(source_text[start:end])
    verb_start = result.start("verb") - start
    verb_end = result.end("verb") - start
    characters[verb_start:verb_end] = " " * (verb_end - verb_start)
    candidate = "".join(characters)
    return bool(
        _source_states_nonrounding_computation(candidate)
        or _ROUNDING_LANGUAGE.search(candidate)
    )


def _looks_like_temporal_or_structural_number(
    text: str,
    match: re.Match[str],
) -> bool:
    """Return whether a prospective base value is only metadata or structure."""

    for pattern in (
        _STATED_CONVERSION_DATE,
        _GERMAN_LEGAL_CITATION,
        _ENGLISH_LEGAL_CITATION,
        _STRUCTURAL_REFERENCE,
    ):
        if any(
            metadata.start() <= match.start() and match.end() <= metadata.end()
            for metadata in pattern.finditer(text)
        ):
            return True

    raw = match.group(0).replace(" ", "").replace(".", "").replace(",", ".")
    with contextlib.suppress(ValueError):
        value = float(raw)
        if value.is_integer() and 1900 <= value <= 2100:
            prefix = text[max(0, match.start() - 24) : match.start()]
            if re.search(
                r"\b(?:jahr(?:es)?|year|in|for|für(?:\s+das)?)\s*$",
                prefix,
                re.IGNORECASE,
            ):
                return True

    prefix = text[max(0, match.start() - 32) : match.start()]
    suffix = text[match.end() : min(len(text), match.end() + 8)]
    if re.search(
        r"(?:§+|artikel|art\.?|absatz|abs\.?|satz|nummer|nr\.?|"
        r"section|sec\.?|subsection|paragraph|clause|item)\s*$",
        prefix,
        re.IGNORECASE,
    ):
        return True
    if prefix.endswith("(") and re.match(r"\s*\)", suffix):
        return True
    line_prefix = prefix.rsplit("\n", 1)[-1]
    return bool(not line_prefix.strip() and re.match(r"\s*\.", suffix))


def _without_stated_conversion_results(source_text: str) -> str:
    """Blank stated-result clauses without hiding other source computations."""

    spans = _stated_conversion_result_spans(source_text)
    if not spans:
        return source_text
    characters = list(source_text)
    for start, end in spans:
        characters[start:end] = " " * (end - start)
    return "".join(characters)


def _without_slash_conjunction_operators(source_text: str) -> str:
    """Keep prose and offsets intact while masking coordinating slashes."""

    return _SLASH_CONJUNCTION.sub(
        lambda match: match.group().replace("/", " "), source_text
    )


def _is_list_bullet(source_text: str, start: int, end: int) -> bool:
    """Return whether a ``•`` marks a list item rather than multiplication.

    Between words it is a list marker ("Eligibility • Section", "applications:
    • DHS 0415F"). Next to a number, a parenthesis or a one-letter variable it
    is multiplication, as in the § 32a EStG tariff "(914,51 • y + 1 400) • y".
    """

    # Bounded windows keep this linear on long bodies with many bullets.
    following = re.match(
        r"[A-Za-zÄÖÜäöüß]+|\S", source_text[end : end + 64].lstrip(" \t")
    )
    if following is None:
        return False
    following_token = following.group(0)
    if not following_token.isalpha() or len(following_token) < 2:
        return False
    previous = re.search(
        r"(?:[A-Za-zÄÖÜäöüß]+|\S)\Z",
        source_text[max(0, start - 64) : start].rstrip(" \t"),
    )
    if previous is None:
        return True
    previous_token = previous.group(0)
    if previous_token.isalpha():
        return len(previous_token) >= 2
    return previous_token in {":", ";", ",", "."}


def _worksheet_referral_typography_spans(
    source_text: str,
) -> tuple[tuple[int, int], ...]:
    """Identify punctuation in corroborated referrals, not worksheet arithmetic."""

    spans: list[tuple[int, int]] = []
    heading = re.match(
        r"\s*(?P<line>[1-9]\d{0,3})\s+WORK CHART\s+(?P<dash>[–—-])\s+"
        r"[^.;\n\"“”]{3,160}?\s+Read the instructions for line (?P=line) "
        r"in the guide before completing this work chart\b",
        source_text,
    )
    if heading is not None and not re.search(
        r"[+*/=×%]|\b(?:add|subtract|multiply|divide|plus|minus|maximum|minimum|"
        r"less|greater|equal|exceed|negative|positive|percent)\b",
        source_text,
        flags=re.IGNORECASE,
    ):
        spans.append(heading.span("dash"))

    # A complete unqualified carry instruction followed by a printed output
    # caption names a destination and row. Qualified or compound instructions
    # remain untouched; e.g. a maximum amount must still require computation.
    carry = re.fullmatch(
        r"\s*Carry (?:the result|the amount|this amount) to line [1-9]\d{0,3} "
        r"of your return\.\s*\x08\s*"
        r"(?P<caption>[A-Za-z][A-Za-z ’'–—-]{2,160}?)\s*(?P<equals>=)\s*[1-9]\d{0,2}"
        r"(?:\s+(?:T[ \t]+)?[A-Z]{2,}(?:[-.][A-Z0-9]+)+[ \t]+"
        r"\(\d{4}-\d{2}\)[ \t]+[1-9]\d?[ \t]+of[ \t]+[1-9]\d?"
        r"[ \t]+Keep these pages for your files\.)?\s*",
        source_text,
    )
    if carry is not None and not re.search(
        r"\b(?:plus|minus|maximum|minimum|less|greater|equal|percent)\b",
        carry.group("caption"),
        flags=re.IGNORECASE,
    ):
        spans.append(carry.span("equals"))
    return tuple(spans)


def _without_manual_typography_operators(source_text: str) -> str:
    """Blank manual typography that only looks like arithmetic, keeping offsets."""

    masked = list(source_text)
    for pattern in (_SLASH_MONTH_YEAR, _WEB_ADDRESS, _FOOTNOTE_ASTERISK):
        for match in pattern.finditer(source_text):
            masked[match.start() : match.end()] = _TYPOGRAPHY_MASK * (
                match.end() - match.start()
            )
    for start, end in _worksheet_referral_typography_spans(source_text):
        masked[start:end] = _TYPOGRAPHY_MASK * (end - start)
    for match in re.finditer(_LIST_BULLET, source_text):
        if _is_list_bullet(source_text, match.start(), match.end()):
            masked[match.start()] = _TYPOGRAPHY_MASK
    if _US_MANUAL_PROSE_SCOPE.get() and not _MANUAL_FORMULA_SIGNAL.search(
        "".join(masked)
    ):
        for pattern in (_MANUAL_WORD_SLASH, _MANUAL_SPACED_WORD_DASH):
            for match in pattern.finditer(source_text):
                masked[match.start() : match.end()] = _TYPOGRAPHY_MASK * (
                    match.end() - match.start()
                )
    return "".join(masked)


def _has_substantive_arithmetic_expression(source_text: str) -> bool:
    """Ignore prose conjunctions and year spans, retaining actual arithmetic."""

    arithmetic_text = list(
        _without_slash_conjunction_operators(
            _without_manual_typography_operators(source_text)
        )
    )
    for metadata_pattern in (
        _STATED_CONVERSION_DATE,
        _EU_REGULATION_NUMERIC_RECALL_CITATION,
        _BFH_DECISION_NUMERIC_RECALL_CITATION,
    ):
        for metadata in metadata_pattern.finditer(source_text):
            if metadata_pattern is _EU_REGULATION_NUMERIC_RECALL_CITATION and (
                re.match(
                    r"\s*(?:[+*/=×·•∗∙−–-]|(?:plus|minus|mal|less)\b)"
                    r"\s*[+−-]?\s*(?:\d|[.,]\d)",
                    source_text[metadata.end() :],
                    flags=re.IGNORECASE,
                )
                or re.search(
                    r"\d(?:[.,]\d+)?\s*"
                    r"(?:[+*/=×·•∗∙−–-]|\b(?:plus|minus|mal|less))\s*$",
                    source_text[: metadata.start()],
                    flags=re.IGNORECASE,
                )
            ):
                # A numeric operator attached to the reference is ambiguous:
                # preserve it rather than hide an operand with the citation.
                continue
            arithmetic_text[metadata.start() : metadata.end()] = " " * (
                metadata.end() - metadata.start()
            )
    masked_source_text = "".join(arithmetic_text)
    if _WORDED_ARITHMETIC_EXPRESSION.search(masked_source_text):
        return True
    for match in _ARITHMETIC_EXPRESSION.finditer(masked_source_text):
        title_prefix = masked_source_text[max(0, match.start() - 120) : match.start()]
        if re.fullmatch(r"\d+\+\s+(?:\d+|[A-Za-z]+)", match.group(0)) and (
            re.search(r"\bage\s+$", title_prefix, flags=re.IGNORECASE)
            or re.search(
                r"\bhousehold\s+size(?:\s+\d+)+\s+$",
                title_prefix,
                flags=re.IGNORECASE,
            )
        ):
            # An age threshold ("age 60+ or disabled") or the last column
            # of a household-size table ("1 2 3 4 5 6+ 48 States") is a
            # range label, not addition to the following prose/table cell.
            continue
        if re.fullmatch(
            r"(?:19|20)\d{2}\s+[–—]\s+[A-Z][A-Za-z'-]*",
            match.group(0),
        ) and re.search(
            r"\b(?:Act|Bill|Memorandum|Program|Report|Title)\s+of\s+$",
            title_prefix,
            flags=re.IGNORECASE,
        ):
            # PDF text commonly preserves an en/em dash in a title such as
            # ``Act of 2025 – Information Memorandum``.  A year followed by a
            # capitalized title word is typography, not subtraction.
            continue
        expression = re.sub(r"\s+", "", match.group(0))
        if re.fullmatch(r"(?:19|20)\d{2}/(?:19|20)\d{2}", expression):
            continue
        return True
    return False


def analyze_complete_source_unit(
    content: str,
    authoritative_source_text: str,
    *,
    corpus_citation_path: str,
    test_cases: Sequence[object] | None,
    extract_numeric_occurrences: NumericOccurrenceExtractor,
    extract_numeric_grounding_occurrences: NumericOccurrenceExtractor | None = None,
    extract_named_scalars: NamedScalarExtractor,
    numeric_value_is_grounded: NumericGroundingPredicate,
    artifact_numeric_values: Sequence[float] | None = None,
    artifact_numeric_bindings: Sequence[tuple[str, float]] | None = None,
    authenticated_same_act_aliases: Sequence[str] = (),
    imported_symbol_contents: Sequence[tuple[str, str]] = (),
    imported_export_bindings: Sequence[ResolvedImportedExport] = (),
    imported_proof_source_context: Mapping[str, str | None] | None = None,
    resolved_dependency_outputs: Sequence[str] = (),
    source_context: Mapping[str, str | None] | None = None,
    execution_evidence: SameCaseEvidence | None = None,
    referral_context: Any = None,
) -> CompleteSourceUnitAnalysis:
    """Analyze one artifact against its authoritative, resolver-owned body."""

    with _us_manual_prose_scope(corpus_citation_path):
        source_text = authoritative_source_text.strip()
        if not source_text:
            issue = (
                "[complete-source-unit:authoritative-source] "
                "The authoritative corpus body is required; `module.summary` is "
                "never completeness evidence."
            )
            return CompleteSourceUnitAnalysis((issue,), (), 0, 0, 0)

        try:
            payload = yaml.safe_load(content)
        except (yaml.YAMLError, TypeError, ValueError) as exc:
            # PyYAML raises ValueError for an impossible timestamp such as
            # `2026-02-30`, not a YAMLError.
            return _unanalyzable_complete_source_unit(exc)
        if not (isinstance(payload, dict) and payload.get("format") == "rulespec/v1"):
            return _unanalyzable_complete_source_unit(
                None, reason="the artifact is not a `format: rulespec/v1` mapping"
            )
        shape_problem = _rulespec_payload_shape_problem(payload)
        if shape_problem is not None:
            return _unanalyzable_complete_source_unit(None, reason=shape_problem)
        try:
            with _hebrew_reading(payload, source_text):
                return _analyze_rulespec_payload(
                    payload,
                    content=content,
                    execution_evidence=execution_evidence,
                    referral_context=referral_context,
                    source_text=source_text,
                    raw_source_text=authoritative_source_text,
                    corpus_citation_path=corpus_citation_path,
                    test_cases=test_cases,
                    extract_numeric_occurrences=extract_numeric_occurrences,
                    extract_numeric_grounding_occurrences=(
                        extract_numeric_grounding_occurrences or extract_numeric_occurrences
                    ),
                    extract_named_scalars=extract_named_scalars,
                    numeric_value_is_grounded=numeric_value_is_grounded,
                    artifact_numeric_values=artifact_numeric_values,
                    artifact_numeric_bindings=artifact_numeric_bindings,
                    authenticated_same_act_aliases=authenticated_same_act_aliases,
                    imported_symbol_contents=imported_symbol_contents,
                    imported_export_bindings=imported_export_bindings,
                    imported_proof_source_context=imported_proof_source_context,
                    resolved_dependency_outputs=resolved_dependency_outputs,
                    source_context=source_context,
                )
        except (yaml.YAMLError, TypeError, ValueError) as exc:
            # A gate that cannot read its input must report it, never pass it.
            return _unanalyzable_complete_source_unit(exc)
