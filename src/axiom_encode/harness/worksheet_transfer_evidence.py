"""Certified bounded worksheet transfer evidence with identity-scoped consumers.

Children are computed from their own facts. Parent expectations corroborate the
computed sum and routes; they never supply a child or aggregate value.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import source_path_evidence as paths
from axiom_encode.harness import source_path_shadow as shadow
from axiom_encode.harness import worksheet_operation_evidence as operations
from axiom_encode.harness.proof_validator import validate_rulespec_proofs

RELATION = "has_t2036_calculation_unit"
UNIT = "t2036_calculation_unit_credit"
TOTAL = "t2036_calculation_units_credit_total"
ORDINARY = "t2036_amount_to_form_428"
MULTI = "t2036_amount_to_form_428mj"
PAYABLE = "jurisdictions_tax_payable_count"
OUTPUTS = (TOTAL, ORDINARY, MULTI)
_MODULE = "ca:policies/cra/t1-2025/provincial-territorial-foreign-tax-credit"
_ORDINARY_TEXT = "Enter the total from line 5 (for each country if applicable) on the line for the provincial or territorial foreign tax credit of\nForm 428."
_MULTI_TEXT = "If you have to pay tax to more than one jurisdiction, enter the amount from line 5 on the applicable line in Part 4,\nSection 428MJ of Form T2203, Provincial and Territorial Taxes for Multiple Jurisdictions, only for the province or territory\nyou resided in on the last day of the tax year."
_GROUP_TEXT = "If you paid tax to more than one foreign country, and the total of the non-business income taxes that you paid to all foreign\ncountries was more than $200, do the calculation on a separate form for each country in Canadian dollars.\nCountry or countries for which you are making this claim:"


@dataclass(frozen=True)
class TransferCaseEvidence:
    case_index: int
    children: tuple[Decimal, ...]
    total: Decimal
    payable_count: int
    ordinary: Decimal
    multi: Decimal


@dataclass(frozen=True)
class WorksheetTransferEvidence:
    source_spans: tuple[tuple[int, int], ...] = ()
    cases: tuple[TransferCaseEvidence, ...] = ()
    routing_pairs: tuple[tuple[int, int], ...] = ()
    unresolved: tuple[str, ...] = ()
    input_identity: str = ""
    source_identity: str = ""
    context_identity: str = ""
    owner_identities: tuple[tuple[str, str], ...] = ()
    all_rule_identities: tuple[tuple[str, str], ...] = ()
    flat_case_indices: tuple[int, ...] = ()
    scalar_paths: paths.CertifiedPathCoverage | None = None
    scalar_operations: operations.WorksheetOperationCoverage | None = None

    def matches(self, payload, cases, source_text, source_context) -> bool:
        return (
            bool(self.cases and self.routing_pairs)
            and not self.unresolved
            and (
                self.input_identity == paths.path_input_identity(payload, cases)
                and self.source_identity
                == hashlib.sha256(source_text.encode()).hexdigest()
                and self.context_identity
                == paths.path_input_identity(source_context or {}, ())
            )
        )

    def owns_output(self, owner, rule) -> bool:
        return (
            bool(self.cases and self.routing_pairs)
            and not self.unresolved
            and (
                (owner, paths.path_input_identity({"rules": [rule]}, ()))
                in self.owner_identities
            )
        )

    def owns_condition(self, owner, rule, start, end, lower, upper, *, source_text):
        if (
            not self.owns_output(owner, rule)
            or (lower, upper) != ("2025-01-01", "2025-12-31")
            or hashlib.sha256(source_text.encode()).hexdigest() != self.source_identity
        ):
            return False
        while start < end and source_text[start].isspace():
            start += 1
        while end > start and source_text[end - 1].isspace():
            end -= 1
        if (start, end) in self.source_spans:
            return True
        # Legacy proposition attribution joins the numeric cap to ordinary
        # carry. Both operations are independently certified for this computed
        # child closure; no other expanded prefix or suffix is accepted.
        return bool(
            self.scalar_operations is not None
            and not self.scalar_operations.unresolved
            and self.scalar_operations.cap_span
            and start == self.scalar_operations.cap_span[0]
            and end == self.source_spans[0][1]
            and not source_text[
                self.scalar_operations.cap_span[1] : self.source_spans[0][0]
            ].strip()
        )

    def owns_exception(
        self, branch, *, source_text, principal_rules, corpus_citation_path
    ):
        if (
            corpus_citation_path != operations._PRIMARY
            or hashlib.sha256(source_text.encode()).hexdigest() != self.source_identity
            or (branch.start, branch.end) not in self.source_spans
            or any(
                (name, paths.path_input_identity({"rules": [rule]}, ()))
                not in self.all_rule_identities
                for name, rule in principal_rules.items()
            )
            or not all(
                self.owns_output(name, principal_rules.get(name, {}))
                for name in (UNIT, *OUTPUTS)
            )
        ):
            return False
        # Recheck the actual current claimant set, including partial proofs.
        return not _additional_claimant(
            principal_rules,
            source_text,
            corpus_citation_path,
            self.source_spans,
            self.scalar_operations,
        )


def _additional_claimant(rules, source_text, citation, spans, core=None):
    branches = sc.recognize_source_structure(source_text)
    for name, rule in rules.items():
        if name in {UNIT, *OUTPUTS}:
            continue
        for path, claimed_citation, excerpt in sc._rule_source_excerpt_atoms(rule):
            if sc._formula_proof_version_index(path) is None or not excerpt:
                continue
            if (
                core is not None
                and not core.unresolved
                and core.cap_span
                and name == operations._LINE5
                and claimed_citation == citation
                and source_text.count(excerpt) == 1
                and excerpt == operations._SOURCE_OPERATIONS["line5"][0][1]
                and source_text.index(excerpt) + len(excerpt) == core.cap_span[1]
                and any(
                    i == sc._formula_proof_version_index(path)
                    and lo == "2025-01-01"
                    and hi == "2025-12-31"
                    for i, _, lo, hi in sc._effective_formula_version_intervals(rule)
                )
            ):
                # The exact, independently certified worksheet operation ends
                # at the cap. Legacy proposition expansion adds transfer text
                # that this selected LINE5 atom does not actually quote.
                continue
            # The source owner resolver retains expanded parent/child ownership;
            # a short quote cannot evade the whole transfer claimant check.
            clauses, ambiguous = sc._source_condition_clauses_owned_by_excerpt(
                excerpt,
                rule=rule,
                source_text=source_text,
                branches=branches,
                corpus_citation_path=citation,
                narrow_conjunctive_excerpt=False,
            )
            overlaps = any(c.start < b and a < c.end for c in clauses for a, b in spans)
            raw_overlap = source_text.count(excerpt) == 1 and any(
                source_text.index(excerpt) < b
                and a < source_text.index(excerpt) + len(excerpt)
                for a, b in spans
            )
            if overlaps or raw_overlap:
                return name
            if (
                ambiguous
                and claimed_citation == citation
                and any(excerpt in source_text[a:b] for a, b in spans)
            ):
                return name
    return None


def _failure(reason: str) -> WorksheetTransferEvidence:
    return WorksheetTransferEvidence(unresolved=(reason,))


def _exact_formula(actual: str, expected: str) -> bool:
    a, b = operations._formula_ast(actual), operations._formula_ast(expected)
    return a is not None and b is not None and ast.dump(a) == ast.dump(b)


def _is_i64_integer(value):
    """Match pinned engine ScalarValueSpec::Integer, without Boolean coercion."""
    return type(value) is int and -(2**63) <= value < 2**63


def _input_value_matches(declaration, value):
    if (declaration.get("entity"), declaration.get("period")) != ("Person", "Year"):
        return False
    dtype = declaration.get("dtype")
    if declaration.get("unit") != ("CAD" if dtype == "Money" else None):
        return False
    if dtype == "Boolean":
        return type(value) is bool
    if dtype == "Integer":
        return _is_i64_integer(value)
    if dtype == "Text":
        return isinstance(value, str)
    if dtype in {"Money", "Decimal", "Number"}:
        return (
            type(value) in {int, float, Decimal}
            and sc._rulespec_runtime_decimal(value) is not None
        )
    return False


def _companion_sum(children: Sequence[Decimal]) -> Decimal | None:
    """Honor both supported caller ID orders, never round away divergence.

    CLI emits related_0..N; pipeline emits common-prefix-1..N. Engine895 sum
    visits distinct related IDs in sorted order (execution-semantics.md).
    Compact rows cannot override these IDs in this supported common subset.
    """
    results = []
    try:
        for offset in (0, 1):
            total = Decimal(0)
            for index in sorted(range(len(children)), key=lambda i: str(i + offset)):
                total = sc._rulespec_decimal_binary_operation(
                    total, children[index], "PLUS"
                )
            results.append(total)
    except (ArithmeticError, InvalidOperation, TypeError, ValueError):
        return None
    return results[0] if results[0] == results[1] else None


def certify_worksheet_transfers(
    payload: Mapping[str, Any],
    *,
    source_text: str,
    corpus_citation_path: str,
    source_context: Mapping[str, str | None] | None,
    test_cases: Sequence[object] | None,
    numeric_value_is_grounded,
    extract_numeric_occurrences,
) -> WorksheetTransferEvidence:
    """Compute closed component evidence without modifying completeness findings."""
    if corpus_citation_path != operations._PRIMARY or not test_cases:
        return _failure("unsupported source or absent cases")
    rules = shadow._index(payload.get("rules"))
    inputs = shadow._index(payload.get("inputs"))
    if rules is None or inputs is None:
        return _failure("invalid declarations")
    if not {RELATION, UNIT, *OUTPUTS} <= rules.keys():
        return _failure("missing transfer declarations")
    quotes = sc._annual_source_quoted_spans(source_text)
    if quotes is None:
        return _failure("malformed source quotes")
    spans = []
    for text in (_ORDINARY_TEXT, _MULTI_TEXT, _GROUP_TEXT):
        if source_text.count(text) != 1:
            return _failure("unknown transfer or grouping source")
        start = source_text.index(text)
        if any(a <= start < b for a, b in quotes):
            return _failure("quoted transfer source")
        spans.append((start, start + len(text)))
    if source_text[spans[0][1] : spans[1][0]].strip():
        return _failure("unowned routing context")
    if not source_text[spans[1][1] :].startswith(
        "\nT2036 E (25) (Ce formulaire est disponible en français.) Page 1 of 2\n\n"
        "(1) If you must pay minimum tax, follow the instructions below:"
    ):
        return _failure("unknown transfer tail ownership")
    relation = rules[RELATION]
    if relation.get("kind") != "data_relation" or relation.get("data_relation") != {
        "arity": 2,
        "arguments": ["Person", "T2036CalculationUnit"],
    }:
        return _failure("wrong relation contract")
    if not any(
        p == "data_relation" and c == corpus_citation_path and _GROUP_TEXT in ex
        for p, c, ex in sc._rule_source_excerpt_atoms(relation)
    ):
        return _failure("missing relation grouping proof")
    intro = operations._annual_intro(source_text)
    if intro is None:
        return _failure("missing annual context")
    prefix = sc._collapse_text(intro.group().split(" that you can deduct", 1)[0])
    expected = {
        UNIT: operations._LINE5,
        TOTAL: f"sum({RELATION}.{UNIT})",
        ORDINARY: f"if {PAYABLE} > 1: 0 else: {TOTAL}",
        MULTI: f"if {PAYABLE} > 1: {TOTAL} else: 0",
    }
    for name, expression in expected.items():
        rule = rules[name]
        if (
            rule.get("kind"),
            rule.get("entity"),
            rule.get("period"),
            rule.get("dtype"),
            rule.get("unit"),
        ) != (
            "derived",
            "T2036CalculationUnit" if name == UNIT else "Person",
            "Year",
            "Money",
            "CAD",
        ):
            return _failure("wrong transfer type: " + name)
        selected = sc._effective_formula_version_intervals(rule)
        if len(selected) != 1:
            return _failure("ambiguous transfer versions: " + name)
        index, version, start, end = selected[0]
        if (start, end) != ("2025-01-01", "2025-12-31") or not _exact_formula(
            version, expression
        ):
            return _failure("wrong selected transfer operation: " + name)
        atoms = tuple(sc._rule_source_excerpt_atoms(rule))
        owned = (
            (_ORDINARY_TEXT, _MULTI_TEXT)
            if name in (ORDINARY, MULTI)
            else (_ORDINARY_TEXT,)
        )
        if any(
            not any(
                p == f"versions[{index}].formula"
                and c == corpus_citation_path
                and text in ex
                for p, c, ex in atoms
            )
            for text in owned
        ):
            return _failure("missing transfer operation proof: " + name)
        if any(
            not any(
                p == f"versions[{index}].{field}"
                and c == corpus_citation_path
                and prefix in sc._collapse_text(ex)
                and ex in source_text
                for p, c, ex in atoms
            )
            for field in ("effective_from", "effective_to")
        ):
            return _failure("unowned transfer date: " + name)
    if (
        inputs.get(PAYABLE, {}).get("dtype"),
        inputs.get(PAYABLE, {}).get("entity"),
        inputs.get(PAYABLE, {}).get("period"),
        inputs.get(PAYABLE, {}).get("unit"),
    ) != ("Integer", "Person", "Year", None):
        return _failure("wrong source payable count contract")
    new_rules = [rules[n] for n in (RELATION, UNIT, *OUTPUTS)]
    issues = validate_rulespec_proofs(
        yaml.safe_dump({**payload, "rules": new_rules}),
        require_policy_proofs=True,
        source_texts=source_context,
    ).issues
    if issues:
        return _failure("invalid transfer proof: " + issues[0])
    # The existing core accepts scalar fixtures only. Nested transfer fixtures
    # are never treated as scalar witnesses: every one is validated below.
    flat_cases = []
    flat_indices = []
    for case_index, case in enumerate(test_cases):
        if not isinstance(case, dict) or not isinstance(case.get("input"), dict):
            return _failure("malformed fixture")
        outputs = case.get("output")
        if not isinstance(outputs, dict):
            return _failure("malformed fixture outputs")
        is_transfer = any(str(k).rsplit("#", 1)[-1] in OUTPUTS for k in outputs)
        nested = any(isinstance(v, (dict, list)) for v in case["input"].values())
        if not is_transfer and nested:
            return _failure("unrecognized nested fixture")
        if not is_transfer:
            flat_cases.append(case)
            flat_indices.append(case_index)
    path_evidence = paths.certify_worksheet_paths(
        payload,
        source_text=source_text,
        corpus_citation_path=corpus_citation_path,
        test_cases=flat_cases,
        numeric_value_is_grounded=numeric_value_is_grounded,
        extract_numeric_occurrences=extract_numeric_occurrences,
    )
    core = operations.certify_worksheet_operations(
        payload,
        source_text=source_text,
        corpus_citation_path=corpus_citation_path,
        source_context=source_context,
        test_cases=flat_cases,
        paths_evidence=path_evidence,
    )
    if core.unresolved or core.outputs != operations._OUTPUTS:
        return _failure("uncertified child source operations")
    claimant = _additional_claimant(
        rules, source_text, corpus_citation_path, spans[:2], core
    )
    if claimant:
        return _failure("additional transfer claimant: " + claimant)
    cases = sc._typed_numeric_expected_cases(test_cases, rules) or ()
    constants = sc._constant_rule_environment(payload)
    allowed_names = {
        d.name
        for record in path_evidence.obligations
        if record.certified
        for d in record.selected_dependencies
    }
    for owner in operations._expected_operations():
        resolver = operations._OperationResolver(
            payload,
            owner,
            "2025-01-01",
            "2025-12-31",
            source_text,
            corpus_citation_path,
        )
        if resolver.expand(ast.Name(id=owner)) is None or resolver.errors:
            return _failure("uncertified selected child closure")
        allowed_names.update(resolver.dependencies)
    local = {
        n: r
        for n, r in rules.items()
        if r.get("kind") == "derived" and n in allowed_names
    }
    links = []
    selected_cases = {}
    for i, case in enumerate(cases):
        output = case.get("output")
        if not isinstance(output, dict):
            continue
        names = {str(k).rsplit("#", 1)[-1] for k in output}
        if not names.intersection(OUTPUTS):
            continue
        if set(output) != {_MODULE + "#" + n for n in OUTPUTS}:
            return _failure("unsupported parent assertion set")
        if case.get("period") != {
            "period_kind": "tax_year",
            "start": "2025-01-01",
            "end": "2025-12-31",
        }:
            return _failure("unsupported parent period")
        raw = case.get("input")
        if not isinstance(raw, dict):
            return _failure("missing parent input")
        relation_keys = [k for k in raw if k == _MODULE + "#relation." + RELATION]
        count_keys = [k for k in raw if k == _MODULE + "#input." + PAYABLE]
        if len(relation_keys) != 1 or len(count_keys) != 1 or len(raw) != 2:
            return _failure("unsupported parent input layout")
        rows, count = raw[relation_keys[0]], raw[count_keys[0]]
        if (
            not isinstance(rows, list)
            or not rows
            or len(rows) > 64
            or not _is_i64_integer(count)
            or count < 1
        ):
            return _failure("empty or invalid factual calculation domain")
        children = []
        for row in rows:
            if not isinstance(row, dict) or not row:
                return _failure("invalid child record")
            seen = set()
            for key, value in row.items():
                name = str(key).rsplit("#input.", 1)[-1]
                if (
                    key != _MODULE + "#input." + name
                    or name in rules
                    or name in seen
                    or name not in inputs
                    or not _input_value_matches(inputs[name], value)
                ):
                    return _failure("invalid typed child fact")
                seen.add(name)
            child = {"input": row, "period": case["period"], "output": {}}
            environment = sc._case_dependency_environment(
                local,
                child,
                formula_environment=constants,
                require_asserted_value=False,
                allowed_names=allowed_names,
            )
            value = sc._rulespec_runtime_decimal(environment.get(operations._LINE5))
            if value is None:
                return _failure("unresolved reached child result")
            children.append(value)
        total = _companion_sum(children)
        if total is None:
            return _failure("unrepresentable or caller-order-dependent child sum")
        ordinary, multi = (Decimal(0), total) if count > 1 else (total, Decimal(0))
        for name, value in zip(OUTPUTS, (total, ordinary, multi)):
            asserted = sc._test_case_asserted_output_value(case, name)
            if not sc._asserted_formula_runtime_values_equal(
                rules[name], value, asserted
            ):
                return _failure("incorrect computed parent assertion")
        links.append(
            TransferCaseEvidence(i, tuple(children), total, count, ordinary, multi)
        )
        selected_cases[i] = (case, count_keys[0])
    if not links:
        return _failure("no executed nonempty child cases")
    pairs = []
    for a in links:
        for b in links:
            if (
                a.payable_count != 1
                or b.payable_count != 2
                or a.total <= 0
                or a.children != b.children
            ):
                continue
            ca, key = selected_cases[a.case_index]
            cb, other_key = selected_cases[b.case_index]
            if key == other_key and paths._same_world(ca, cb, PAYABLE):
                pairs.append((a.case_index, b.case_index))
    if not pairs:
        return _failure("missing same-world root routing pair")
    return WorksheetTransferEvidence(
        tuple(spans[:2]),
        tuple(links),
        tuple(pairs),
        (),
        paths.path_input_identity(payload, test_cases),
        hashlib.sha256(source_text.encode()).hexdigest(),
        paths.path_input_identity(source_context or {}, ()),
        tuple(
            (n, paths.path_input_identity({"rules": [rules[n]]}, ()))
            for n in (UNIT, *OUTPUTS)
        ),
        tuple(
            (n, paths.path_input_identity({"rules": [r]}, ())) for n, r in rules.items()
        ),
        tuple(flat_indices),
        path_evidence,
        core,
    )
