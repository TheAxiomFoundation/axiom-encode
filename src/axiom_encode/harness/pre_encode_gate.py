"""Pre-encode gate: deterministic target checks that run before any model call.

See docs/design/encode-failure-classes.md. Each check reads only data the
encoder already resolved for the target (its corpus source unit, the RuleSpec
checkout, and the corpus release). ``enforce`` checks stop an encode in the
default ``enforce`` mode; ``report`` checks only record a finding. In
``report`` mode every check is advisory.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..failure_classes import CORPUS, DEPENDENCY, SELECTION
from ..source_text_signals import (
    CONTENTS_MARKER,
    TOC_LEADER,
    TOC_MIN_LEADERS,
    is_table_of_contents,
    rule_signal_outside_contents_entries,
    text_outside_contents_entries,
)

PRE_ENCODE_GATE_SCHEMA = "axiom-encode/pre-encode-gate/v1"
PRE_ENCODE_GATE_MODES = ("report", "enforce")
DEFAULT_PRE_ENCODE_GATE_MODE = "enforce"

CHECK_TOC_PAGE = "toc_page"
CHECK_NO_OPERATIVE_TEXT = "no_operative_text"
CHECK_REPEALED_TARGET = "repealed_target"
CHECK_EXISTING_MODULE = "existing_module"
CHECK_UNENCODED_REFERENCE = "unencoded_reference"

# check -> (failure class, may enforce)
PRE_ENCODE_CHECKS: Mapping[str, tuple[str, bool]] = {
    # Report-only: a judgement heuristic never stops an encode. It may move to
    # enforce only on measured evidence from real rows.
    CHECK_TOC_PAGE: (SELECTION, False),
    CHECK_NO_OPERATIVE_TEXT: (SELECTION, True),
    CHECK_REPEALED_TARGET: (SELECTION, True),
    CHECK_EXISTING_MODULE: (SELECTION, False),
    CHECK_UNENCODED_REFERENCE: (DEPENDENCY, False),
}

# A citation locator left after the heading is removed: an optional unit word
# and one marker. The marker counts only when it names the target itself (for
# example "(a)" or "273.9" for us/regulation/7/273/9); a marker that names
# another provision, such as "Section 401(a)(9)(A).", is a cross-reference.
_LOCATOR_RE = re.compile(
    r"^(?:(?:§+|sec(?:tion)?\.?|part|subpart|chapter|ch\.|article|art\.|"
    r"rule|title|page|p\.)\s*)?(?P<marker>[^\s]+)$",
    re.IGNORECASE,
)
_EDGE_PUNCTUATION = " \t\n.,;:-–—|*#_"

# Explicit federal citations in source text. Only USC sections and CFR
# sections are recognised; a state citation is never read as a reference.
_USC_TEXT_RE = re.compile(
    r"\b(?P<title>\d{1,2})\s+U\.?\s?S\.?\s?C\.?\s+(?:§+\s*)?"
    r"(?P<section>\d+[A-Za-z]?(?:-\d+[A-Za-z]?)?)\b"
)
_CFR_TEXT_RE = re.compile(
    r"\b(?P<title>\d{1,2})\s+C\.?\s?F\.?\s?R\.?\s+(?:§+\s*)?"
    r"(?P<section>\d+\.\d+[A-Za-z]?)\b"
)
MAX_REFERENCE_LOOKUPS = 8
# toc_page also needs the body to be mostly contents entries. Measured on the
# all-states SNAP queue, contents pages keep at most 12% of their text outside
# entries; a contents list followed by a change-history log keeps 21-52%, and
# a whole chapter that opens with its contents list keeps 77% or more.
TOC_MAX_TEXT_OUTSIDE_ENTRIES = 0.15


@dataclass(frozen=True)
class GateSource:
    """The target text and identity the gate judges."""

    requested: str
    citation_path: str
    body: str
    labels: tuple[str, ...] = ()
    # The encoder also gets amendment documents for this target (a principal
    # provision with an amendment, or an amending provision). Its own body
    # is then not the whole source, so content checks only report.
    amendment_context: bool = False

    @classmethod
    def from_source_unit(cls, source_unit: Any) -> GateSource:
        labels: list[str] = []
        resolved = getattr(source_unit, "resolved_source", None)
        selected = getattr(resolved, "selected_record", None)
        for value in (
            getattr(selected, "heading", None),
            getattr(selected, "citation_label", None),
            (getattr(selected, "metadata", None) or {}).get("title"),
            (getattr(source_unit, "provision_metadata", None) or {}).get("title"),
        ):
            if isinstance(value, str) and value.strip():
                labels.append(value)
        return cls(
            requested=str(source_unit.requested),
            citation_path=str(source_unit.citation_path),
            body=str(source_unit.body or ""),
            labels=tuple(dict.fromkeys(labels)),
            amendment_context=bool(
                getattr(source_unit, "amendment_documents", ())
                or getattr(source_unit, "amendment_request", None)
            ),
        )


@dataclass(frozen=True)
class GateFinding:
    check: str
    failure_class: str
    subclass: str | None
    enforce: bool
    reason: str
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "class": self.failure_class,
            "subclass": self.subclass,
            "enforce": self.enforce,
            "reason": self.reason,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class GateResult:
    mode: str
    findings: tuple[GateFinding, ...] = field(default_factory=tuple)

    @property
    def blocked(self) -> bool:
        return any(finding.enforce for finding in self.findings)

    @property
    def blocking_findings(self) -> tuple[GateFinding, ...]:
        return tuple(finding for finding in self.findings if finding.enforce)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PRE_ENCODE_GATE_SCHEMA,
            "mode": self.mode,
            "blocked": self.blocked,
            "findings": [finding.to_dict() for finding in self.findings],
        }


def _finding(
    check: str,
    *,
    mode: str,
    reason: str,
    evidence: Sequence[str] = (),
    failure_class: str | None = None,
    subclass: str | None = None,
    advisory: bool = False,
) -> GateFinding:
    default_class, may_enforce = PRE_ENCODE_CHECKS[check]
    return GateFinding(
        check=check,
        failure_class=failure_class or default_class,
        subclass=subclass,
        enforce=may_enforce and mode == "enforce" and not advisory,
        reason=reason,
        evidence=tuple(str(item)[:200] for item in evidence),
    )


def _normalize_space(text: str) -> str:
    return " ".join(text.split())


def check_toc_page(body: str, labels: Sequence[str] = ()) -> str | None:
    if not is_table_of_contents(body):
        return None
    outside = text_outside_contents_entries(body)
    if outside > TOC_MAX_TEXT_OUTSIDE_ENTRIES:
        return None
    # Forms and value tables also use dot leaders ("Tax liability ....... 1."),
    # so the page must also call itself a table of contents.
    if CONTENTS_MARKER.search(body) is None:
        return None
    # A rule next to a contents list is still a rule, however short.
    if rule_signal_outside_contents_entries(body, labels) is not None:
        return None
    leaders = len(TOC_LEADER.findall(body))
    return (
        f"source has {leaders} dot-leader runs (threshold {TOC_MIN_LEADERS}) "
        f"and {outside:.0%} of its text outside contents entries "
        f"(limit {TOC_MAX_TEXT_OUTSIDE_ENTRIES:.0%})"
    )


# An empty or punctuation-only body is a corpus extraction gap (the row lost
# its text), not a target-selection error; a heading-only body is selection.
EMPTY_BODY_REASON = "source body is empty"
PUNCTUATION_BODY_REASON = "source body has only punctuation"
_CORPUS_CONTENT_REASONS = frozenset({EMPTY_BODY_REASON, PUNCTUATION_BODY_REASON})


def _marker_key(value: str) -> str:
    return re.sub(r"[^0-9a-z]", "", value.casefold())


def _target_markers(citation_path: str) -> frozenset[str]:
    """Return the keys a locator may use to name the target itself."""

    segments = [segment for segment in citation_path.split("/") if segment]
    keys = set()
    for count in range(1, min(4, len(segments)) + 1):
        key = _marker_key("".join(segments[-count:]))
        if key:
            keys.add(key)
    return frozenset(keys)


def check_no_operative_text(
    body: str, labels: Sequence[str] = (), citation_path: str = ""
) -> str | None:
    """Return a reason when the body is empty or only a heading and locator.

    A body counts as heading-only only when one of the row's own labels was
    found in it; text with no label match is never judged by its length.
    """

    remaining = _normalize_space(body)
    if not remaining:
        return EMPTY_BODY_REASON
    if not any(character.isalnum() for character in remaining):
        return PUNCTUATION_BODY_REASON
    label_found = False
    for label in sorted(
        {_normalize_space(item) for item in labels if item}, key=len, reverse=True
    ):
        if label and re.search(re.escape(label), remaining, flags=re.IGNORECASE):
            label_found = True
            remaining = re.sub(re.escape(label), " ", remaining, flags=re.IGNORECASE)
    if not label_found:
        return None
    remaining = _normalize_space(remaining).strip(_EDGE_PUNCTUATION)
    if not remaining:
        return "source body is only its heading"
    locator = _LOCATOR_RE.fullmatch(remaining)
    if locator is not None and citation_path:
        marker = _marker_key(locator.group("marker"))
        if marker and marker in _target_markers(citation_path):
            return "source body is only its heading and its own citation locator"
    return None


def explicit_federal_references(body: str) -> tuple[str, ...]:
    """Return explicit USC/CFR section citations in reading order."""

    found: list[tuple[int, str]] = []
    for match in _USC_TEXT_RE.finditer(body):
        found.append(
            (match.start(), f"{match.group('title')} USC {match.group('section')}")
        )
    for match in _CFR_TEXT_RE.finditer(body):
        found.append(
            (match.start(), f"{match.group('title')} CFR {match.group('section')}")
        )
    return tuple(dict.fromkeys(citation for _start, citation in sorted(found)))


def _related_paths(left: str, right: str) -> bool:
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def _module_exists(content_root: Path, relative: Path) -> bool:
    module = content_root / relative
    if module.is_file():
        return True
    directory = module.with_suffix("")
    return directory.is_dir() and any(directory.rglob("*.yaml"))


def evaluate_pre_encode_gate(
    source: GateSource | None,
    *,
    mode: str = DEFAULT_PRE_ENCODE_GATE_MODE,
    policy_repo_path: Path | None = None,
    replacement_target: bool = False,
    repealed_reason: str | None = None,
    rulespec_checkout_root: Path | None = None,
    corpus_row_exists: Callable[[str], bool] | None = None,
) -> GateResult:
    """Run the v1 checks on one resolved target.

    ``source`` is ``None`` when the resolver refused the target as repealed;
    ``repealed_reason`` then carries the resolver's message.
    """

    if mode not in PRE_ENCODE_GATE_MODES:
        raise ValueError(f"pre-encode gate mode must be one of {PRE_ENCODE_GATE_MODES}")
    findings: list[GateFinding] = []
    if repealed_reason is not None:
        findings.append(
            _finding(
                CHECK_REPEALED_TARGET,
                mode=mode,
                reason="corpus marks the target row repealed",
                evidence=(repealed_reason,),
            )
        )
    if source is None:
        return GateResult(mode=mode, findings=tuple(findings))

    toc_reason = check_toc_page(source.body, source.labels)
    if toc_reason is not None:
        findings.append(
            _finding(
                CHECK_TOC_PAGE,
                mode=mode,
                reason=toc_reason,
                advisory=source.amendment_context,
                evidence=(source.citation_path,),
            )
        )
    empty_reason = check_no_operative_text(
        source.body, source.labels, source.citation_path
    )
    if empty_reason is not None:
        findings.append(
            _finding(
                CHECK_NO_OPERATIVE_TEXT,
                mode=mode,
                reason=empty_reason,
                evidence=(source.citation_path,),
                advisory=source.amendment_context,
                **(
                    {"failure_class": CORPUS, "subclass": "content"}
                    if empty_reason in _CORPUS_CONTENT_REASONS
                    else {}
                ),
            )
        )

    # Report checks never stop an encode, so a failure to evaluate one is no
    # finding rather than an error.
    if policy_repo_path is not None and not replacement_target:
        try:
            from .evals import _resolve_eval_output_path

            relative = _resolve_eval_output_path(source.requested)
            if (Path(policy_repo_path) / relative).is_file():
                findings.append(
                    _finding(
                        CHECK_EXISTING_MODULE,
                        mode=mode,
                        reason=(
                            "canonical RuleSpec module already exists and no "
                            "replacement target was given"
                        ),
                        evidence=(relative.as_posix(),),
                    )
                )
        except Exception:  # noqa: BLE001 - report check, never fatal
            pass

    if rulespec_checkout_root is not None and corpus_row_exists is not None:
        try:
            unencoded = _unencoded_references(
                source,
                rulespec_checkout_root=Path(rulespec_checkout_root),
                corpus_row_exists=corpus_row_exists,
            )
        except Exception:  # noqa: BLE001 - report check, never fatal
            unencoded = ()
        if unencoded:
            findings.append(
                _finding(
                    CHECK_UNENCODED_REFERENCE,
                    mode=mode,
                    reason=(
                        "source cites a provision that has a corpus row but no "
                        "RuleSpec module"
                    ),
                    evidence=unencoded,
                )
            )
    return GateResult(mode=mode, findings=tuple(findings))


def _unencoded_references(
    source: GateSource,
    *,
    rulespec_checkout_root: Path,
    corpus_row_exists: Callable[[str], bool],
) -> tuple[str, ...]:
    from .evals import _citation_to_corpus_citation_path, _resolve_eval_output_path

    unencoded: list[str] = []
    lookups = 0
    for citation in explicit_federal_references(source.body):
        try:
            cited_path = _citation_to_corpus_citation_path(citation)
            relative = _resolve_eval_output_path(cited_path)
        except ValueError:
            continue
        if _related_paths(cited_path, source.citation_path):
            continue
        if lookups >= MAX_REFERENCE_LOOKUPS:
            break
        content_root = rulespec_checkout_root / cited_path.split("/", 1)[0]
        if _module_exists(content_root, relative):
            continue
        lookups += 1
        try:
            exists = corpus_row_exists(cited_path) is True
        except Exception:  # noqa: BLE001 - unknown is no finding
            exists = False
        if exists:
            unencoded.append(cited_path)
    return tuple(unencoded)
