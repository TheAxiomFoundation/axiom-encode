"""Certified closed worksheet path evidence, separate from structural diagnostics.

No annual spans, cap or transfer coverage is granted here. Unsupported source
families/callers return empty evidence and retain existing validator behavior.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import source_path_shadow as paths
from axiom_encode.harness.proof_validator import validate_rulespec_proofs

_LINE2 = "provincial_territorial_2025_t2036_federal_credit_line_2"
_TAX = "provincial_or_territorial_tax_otherwise_payable"


@dataclass(frozen=True)
class PathObligationEvidence:
    identity: str
    owner: str
    source_sha256: str
    citation: str
    interval: tuple[str, str]
    source_spans: tuple[paths.RawSpan, ...]
    compatible_paths: tuple[int, ...]
    selected_dependencies: tuple[paths.SelectedDependency, ...]
    case_links: tuple[int, ...]
    predicate_pairs: tuple[tuple[str, int, int], ...]
    unresolved: tuple[str, ...]

    @property
    def certified(self) -> bool:
        return bool(self.compatible_paths and self.case_links) and not self.unresolved


@dataclass(frozen=True)
class CertifiedPathCoverage:
    obligations: tuple[PathObligationEvidence, ...] = ()
    # Only a complete closed frame can replace its old names-only attribution.
    condition_spans: tuple[tuple[str, int, int, str, str], ...] = ()
    unresolved: tuple[str, ...] = ()
    input_identity: str = ""

    def owns_condition(
        self,
        owner: str,
        start: int,
        end: int,
        interval_start: str | None,
        interval_end: str | None,
    ) -> bool:
        return any(
            name == owner
            and a <= start < end <= b
            and interval_start is not None
            and interval_end is not None
            and lo <= interval_start <= interval_end <= hi
            for name, a, b, lo, hi in self.condition_spans
        )

    def owns_exception(
        self,
        start: int,
        end: int,
        *,
        source_text: str = "",
        principal_rules: Mapping[str, dict[str, Any]] | None = None,
    ) -> bool:
        if not source_text or principal_rules is None:
            return False
        for owner, a, b, lo, hi in self.condition_spans:
            if not a <= start < end <= b:
                continue
            records = tuple(
                r
                for r in self.obligations
                if r.owner == owner
                and r.certified
                and any(s.start <= start < end <= s.end for s in r.source_spans)
            )
            if not records:
                continue
            dependencies = {
                (d.name, d.version, d.start, d.end)
                for r in records
                for d in r.selected_dependencies
            }
            claimed = False
            invalid = False
            for name, rule in principal_rules.items():
                if rule.get("dtype") not in {
                    "Money",
                    "Decimal",
                    "Rate",
                    "Count",
                    "Integer",
                }:
                    continue
                for path, citation, excerpt in sc._rule_source_excerpt_atoms(rule):
                    match = re.fullmatch(r"versions\[(\d+)\]\.formula", path)
                    if match is None or not excerpt or source_text.count(excerpt) != 1:
                        continue
                    x = source_text.index(excerpt)
                    if not (x < end and start < x + len(excerpt)):
                        continue
                    if citation not in {r.citation for r in records}:
                        invalid = True
                        continue
                    claimed = True
                    index = int(match[1])
                    intervals = tuple(
                        (lower, upper)
                        for i, _, lower, upper in sc._effective_formula_version_intervals(
                            rule
                        )
                        if i == index
                    )
                    if not intervals or any(
                        lower is None
                        or upper is None
                        or lower < lo
                        or upper > hi
                        or (name, index, lower, upper) not in dependencies
                        for lower, upper in intervals
                    ):
                        invalid = True
            if claimed and not invalid:
                return True
        return False


def _same_world(a: dict[str, Any], b: dict[str, Any], changed: str) -> bool:
    if (
        a.get("period") != b.get("period")
        or not isinstance(a.get("output"), dict)
        or not isinstance(b.get("output"), dict)
        or set(a["output"]) != set(b["output"])
        or not isinstance(a.get("input"), dict)
        or not isinstance(b.get("input"), dict)
        or set(a["input"]) != set(b["input"])
        or not sc._cases_differ_by_one_input(a, b)
    ):
        return False
    left, right = (
        sc._case_input_formula_environment(a),
        sc._case_input_formula_environment(b),
    )
    if left is None or right is None or set(left) != set(right):
        return False
    return {key for key in left if left[key] != right[key]} == {changed}


def _proof_ranges(
    rules: Mapping[str, dict[str, Any]],
    dependencies: Sequence[paths.SelectedDependency],
    source: str,
    citation: str,
) -> tuple[tuple[int, int], ...]:
    ranges = []
    for dependency in dependencies:
        rule = rules[dependency.name]
        for path, cited, excerpt in sc._rule_source_excerpt_atoms(rule):
            if path != f"versions[{dependency.version}].formula" or cited != citation:
                continue
            if not excerpt or source.count(excerpt) != 1:
                continue
            start = source.index(excerpt)
            ranges.append((start, start + len(excerpt)))
    return tuple(ranges)


def _covered_operation(
    span: paths.RawSpan, ranges: Sequence[tuple[int, int]], source: str
) -> bool:
    # Only exact source-owned ordinal markers may be outside operation atoms.
    # Keep operands, predicate words and every other non-whitespace byte.
    labels = {
        i
        for match in re.finditer(
            r"(?m)^\s*[1-9][0-9]*\.\s", source[span.start : span.end]
        )
        for i in range(span.start + match.start(), span.start + match.end())
    }
    return all(
        char.isspace() or i in labels or any(a <= i < b for a, b in ranges)
        for i, char in enumerate(source[span.start : span.end], span.start)
    )


def path_input_identity(
    payload: Mapping[str, Any], cases: Sequence[object] | None
) -> str:
    """Bind an internal certificate to its exact typed candidate and cases."""
    rules = paths._index(payload.get("rules")) or {}
    normalized = sc._typed_numeric_expected_cases(cases, rules)
    encoded = json.dumps(
        {"payload": payload, "cases": normalized},
        sort_keys=True,
        default=lambda value: {
            "python_type": type(value).__name__,
            "value": str(value),
        },
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def certify_worksheet_paths(
    payload: Mapping[str, Any],
    *,
    source_text: str,
    corpus_citation_path: str,
    test_cases: Sequence[object] | None,
    numeric_value_is_grounded: Any = None,
    extract_numeric_occurrences: Any = None,
) -> CertifiedPathCoverage:
    """Use closed source operations + selected proofs + real same-world pairs.

    The existing narrow upstream field contract is preserved. No imported body,
    guessed alias, unproved path, or nonexecuted definition can grant evidence.
    """
    rules = paths._index(payload.get("rules"))
    if rules is None or not {_LINE2, _TAX} <= rules.keys() or not test_cases:
        return CertifiedPathCoverage()
    contract = paths.worksheet_shadow_contract(
        source_text,
        citation=corpus_citation_path,
        line2_owner=_LINE2,
        tax_otherwise_owner=_TAX,
    )
    if contract.unresolved:
        return CertifiedPathCoverage(unresolved=contract.unresolved)
    cases = [
        c
        for c in (sc._typed_numeric_expected_cases(test_cases, rules) or ())
        if isinstance(c, dict)
    ]
    if any(
        sc._case_input_principal_output_collisions(case, set(rules)) for case in cases
    ):
        return CertifiedPathCoverage(
            unresolved=("case supplies derived output as input",)
        )
    records = []
    coverage = []
    global_errors = []
    frame_by_owner = {
        _LINE2: tuple(f for f in contract.frames if f.kind == "minimum_tax"),
        _TAX: tuple(f for f in contract.frames if f.kind != "minimum_tax"),
    }
    # Frame kinds are data from the closed source parser, not candidate labels.
    amt = paths.closed_amt_frame(source_text)
    if amt is None:
        return CertifiedPathCoverage()
    frame_by_owner[_LINE2] = (amt,)
    frame_by_owner[_TAX] = paths.closed_result_frames(source_text)
    for owner in (_LINE2, _TAX):
        result = paths.inspect_source_paths(
            payload,
            source_text=source_text,
            owner=owner,
            obligations=contract.obligations,
        )
        if result.unresolved:
            global_errors.extend(result.unresolved)
            continue
        links = paths.corroborated_case_links(payload, result, cases)
        if not links or any(not link["corroborated"] for link in links):
            global_errors.append(owner + ": failed asserted case link")
            continue
        selected = tuple({d for p in result.paths for d in p.dependencies})
        selected_names = {d.name for d in selected}
        proof_payload = {
            **payload,
            "rules": [rules[name] for name in sorted(selected_names)],
        }
        intro = re.search(
            r"(?m)^Use this form to calculate the foreign non-business income tax credit\s+for (?P<year>20[0-9]{2}) that you can deduct from the income tax",
            source_text,
        )
        if intro is None:
            continue
        year = intro["year"]
        prefix = sc._collapse_text(intro.group().split(" that you can deduct", 1)[0])
        quoted = sc._annual_source_quoted_spans(source_text)
        header = re.fullmatch(
            r"(?:[A-Z][A-Za-z]* )?Form [A-Z]*[1-9]\d* "
            + re.escape(year)
            + r" (?P<title>[A-Za-z -]{1,120})\n\s*(?:Protected [A-Z] when completed\s+)?(?P=title)\s*",
            source_text[: intro.start()],
        )
        if (
            header is None
            or quoted is None
            or any(a < intro.end() and b > 0 for a, b in quoted)
        ):
            global_errors.append(owner + ": unowned annual source header")
            continue
        date_errors = []
        for dependency in selected:
            actual_intervals = sc._effective_formula_version_intervals(
                rules[dependency.name]
            )
            selected_interval = next(
                ((a, b) for i, _, a, b in actual_intervals if i == dependency.version),
                None,
            )
            if selected_interval != (year + "-01-01", year + "-12-31"):
                date_errors.append(
                    dependency.name
                    + ": selected interval is not the source annual interval"
                )
            for field in ("effective_from", "effective_to"):
                if not any(
                    path == f"versions[{dependency.version}].{field}"
                    and cite == corpus_citation_path
                    and prefix in sc._collapse_text(ex)
                    and ex in source_text
                    for path, cite, ex in sc._rule_source_excerpt_atoms(
                        rules[dependency.name]
                    )
                ):
                    date_errors.append(
                        dependency.name + ": missing owned annual date proof"
                    )
        if date_errors:
            global_errors.extend(date_errors)
            continue
        proof = validate_rulespec_proofs(
            yaml.safe_dump(proof_payload),
            require_policy_proofs=True,
            source_texts={corpus_citation_path: source_text},
        )
        if not proof.passed:
            global_errors.extend(owner + ": " + issue for issue in proof.issues)
            continue
        # Numeric grounding remains the caller's normal source-specific API.
        if numeric_value_is_grounded is None or extract_numeric_occurrences is None:
            global_errors.append(owner + ": numeric certification unavailable")
            continue
        numeric_errors = []
        for dependency in selected:
            if str(rules[dependency.name].get("kind")) != "parameter":
                continue
            for occurrence in extract_numeric_occurrences(dependency.formula):
                excerpts = [
                    ex
                    for path, cite, ex in sc._rule_source_excerpt_atoms(
                        rules[dependency.name]
                    )
                    if path == f"versions[{dependency.version}].formula"
                    and cite == corpus_citation_path
                ]
                if not excerpts or not any(
                    numeric_value_is_grounded(
                        occurrence.value, extract_numeric_occurrences(ex)
                    )
                    for ex in excerpts
                ):
                    numeric_errors.append(
                        dependency.name + ": ungrounded selected parameter"
                    )
        if numeric_errors:
            global_errors.extend(numeric_errors)
            continue
        ranges = _proof_ranges(rules, selected, source_text, corpus_citation_path)
        owner_records = []
        for obligation in contract.obligations:
            if obligation.owner != owner or obligation.identity == "ordinary_line2":
                continue  # Not one of the seven requested monetary clauses.
            rows = [
                row
                for row in result.rows
                if row["obligation"] == obligation.identity
                and row["status"] != "excluded"
            ]
            compatible = tuple(row["path"] for row in rows)
            errors = []
            if not compatible or any(
                row["status"] != "diagnostic_match" for row in rows
            ):
                errors.append("not all compatible source-positive paths match")
            operation_spans = tuple(
                s
                for s in obligation.spans
                if s
                not in tuple(
                    f.owner for f in frame_by_owner[owner] if f.owner != f.body
                )
            )
            if not operation_spans or not all(
                _covered_operation(s, ranges, source_text) for s in operation_spans
            ):
                errors.append(
                    "selected formula proofs do not cover owned operation/predicates"
                )
            active = [link for link in links if link["path"] in compatible]
            if set(link["path"] for link in active) != set(compatible):
                errors.append("unwitnessed compatible path/interval")
            pairs = []
            for predicate in obligation.predicates:
                # A source OR names two permitted form operations, not an
                # eligibility/amount-change requirement. Both paths still need
                # actual independently corroborated operation executions.
                if (
                    predicate.name == "alberta_applicable_form_is_428mj"
                    and obligation.identity.startswith("alberta_")
                ):
                    continue
                source_frames = [
                    frame
                    for frame in frame_by_owner[owner]
                    if frame.body in obligation.spans
                ]
                siblings = [
                    o
                    for o in contract.obligations
                    if o.owner == owner
                    and any(frame.body in o.spans for frame in source_frames)
                    and predicate in o.predicates
                ]
                parent_paths = {
                    row["path"]
                    for row in result.rows
                    if row["status"] == "diagnostic_match"
                    and row["obligation"] in {o.identity for o in siblings}
                }
                parent_active = [link for link in links if link["path"] in parent_paths]
                found = []
                for a in parent_active:
                    for b in links:
                        if a["case_index"] == b["case_index"] or not _same_world(
                            cases[a["case_index"]],
                            cases[b["case_index"]],
                            predicate.name,
                        ):
                            continue
                        env = sc._case_input_formula_environment(cases[b["case_index"]])
                        if (
                            env is None
                            or paths._predicate_value(predicate, env) is not False
                            or (a["actual"] == b["actual"] and a["path"] == b["path"])
                        ):
                            continue
                        found.append((predicate.name, a["case_index"], b["case_index"]))
                if not found:
                    errors.append(
                        "no corroborated same-world predicate pair: " + predicate.name
                    )
                else:
                    pairs.append(sorted(found)[0])
            record = PathObligationEvidence(
                obligation.identity,
                owner,
                result.source_sha256,
                corpus_citation_path,
                (obligation.start, obligation.end),
                obligation.spans,
                compatible,
                selected,
                tuple(sorted({x["case_index"] for x in active})),
                tuple(pairs),
                tuple(errors),
            )
            records.append(record)
            owner_records.append(record)
        for frame in frame_by_owner[owner]:
            relevant = [r for r in owner_records if frame.body in r.source_spans]
            if relevant and all(r.certified for r in relevant):
                intervals = {r.interval for r in relevant}
                if len(intervals) == 1:
                    lo, hi = intervals.pop()
                    start = frame.body.start
                    while start > 0 and source_text[start - 1].isspace():
                        start -= 1
                    # Include only the closed parser's immediate structural
                    # note heading, never an intervening operation/condition.
                    if frame.owner != frame.body and re.fullmatch(
                        r"\([1-9][0-9]*\) Provincial or territorial tax otherwise payable\s+",
                        source_text[frame.owner.start : frame.body.start],
                    ):
                        start = frame.owner.start
                    end = frame.body.end
                    while end < len(source_text) and source_text[end].isspace():
                        end += 1
                    coverage.append((owner, start, end, lo, hi))
    return CertifiedPathCoverage(
        tuple(records),
        tuple(coverage),
        tuple(global_errors),
        path_input_identity(payload, test_cases),
    )
