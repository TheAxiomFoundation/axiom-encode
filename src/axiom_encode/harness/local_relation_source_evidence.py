"""Invocation-local dependency observations; persisted diagnostics are not evidence.

This adapter does not authenticate source text or discharge source obligations.
Its token exists only while the normal source-analysis call holds this scope.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, DecimalException
from typing import Any, Iterator

from . import local_compile_custody as custody
from .local_case_custody import _live_seal, _Record, _request, _typed
from .local_ir_replay import check_compiled_replay

_DECLINE = (ValueError, TypeError, KeyError, AttributeError, OSError, DecimalException)


def _source_key(value: Any) -> tuple:
    """Exact identity for existing analyzer normalization, never caller coercion."""
    from .source_completeness import _CurrencyRounding, _CurrencyRoundingRule

    if type(value) is Decimal:
        if not value.is_finite():
            raise ValueError("nonfinite normalized number")
        return ("Decimal", value.as_tuple())
    if type(value) is _CurrencyRoundingRule:
        rounding = value.resolved_currency_rounding
        if rounding is None:
            stamp = None
        elif (
            type(rounding) is _CurrencyRounding
            and type(rounding.mode) is str
            and type(rounding.minor_units) is int
        ):
            stamp = (rounding.mode, rounding.minor_units)
        else:
            raise ValueError("unsupported rounding stamp")
        return ("currency_rule", _source_key(dict(value)), stamp)
    if type(value) is dict and all(type(k) is str for k in value):
        return ("dict", tuple((k, _source_key(v)) for k, v in value.items()))
    if type(value) is list:
        return ("list", tuple(_source_key(v) for v in value))
    return _typed(value)


def _normalized_source_inputs(policy: dict, cases: list) -> tuple[dict, list]:
    from . import source_completeness as source

    normalized_policy = source._bind_currency_rounding_rules(policy)
    named = {
        str(r.get("name") or "").strip(): r
        for r in normalized_policy.get("rules", [])
        if isinstance(r, dict) and str(r.get("name") or "").strip()
    }
    normalized_cases = source._typed_date_cases(
        source._typed_numeric_expected_cases(cases, named), normalized_policy
    )
    if type(normalized_cases) is not list or len(normalized_cases) != len(cases):
        raise ValueError("normalization changed case sequence")
    return normalized_policy, normalized_cases


@dataclass(frozen=True)
class _Bound:
    owner: Any
    generation: int
    receipt: Any
    records: tuple[_Record, ...]
    content: str
    cases: Any
    cases_identity: tuple
    sources: Any
    sources_identity: tuple
    policy: dict
    # Serialized case keys and engine records retain original ordinals.
    observations: tuple[tuple[frozenset[tuple], dict, dict, dict], ...]
    seal: Any

    def fresh(self) -> bool:
        try:
            return (
                custody._owner(self.owner.capture) is self.owner
                and custody._fresh(self.owner)
                and self.owner.generation == self.generation
                and self.owner.receipt is self.receipt
                and _live_seal(self.owner, self.receipt.pipeline) is self.seal
                and self.seal is not None
                and self.owner.consumed
                and self.owner.case_records is self.records
                and self.receipt.output.read_bytes() == self.receipt.raw
                and not self.receipt.output.is_symlink()
                and _typed(self.cases) == self.cases_identity
                and _typed(self.sources) == self.sources_identity
            )
        except _DECLINE:
            return False


_contexts: dict[object, _Bound] = {}
# Opaque operation replacements live only within the owning source invocation.
_site_replacements: dict[object, tuple[object, Any, tuple, tuple]] = {}


@contextmanager
def local_relation_source_scope(
    capture: Any,
    pipeline: Any,
    content: str,
    cases: Any,
    source_context: Any,
) -> Iterator[object | None]:
    """Register only an existing successful materializer/compile/suite chain."""
    from .validator_pipeline import _safe_load_unique_keys

    token = None
    try:
        owner = custody._owner(capture)
        if (
            owner is None
            or not custody._fresh(owner)
            or owner.receipt is None
            or owner.receipt.pipeline is not pipeline
            or not owner.consumed
            or type(owner.case_records) is not tuple
            or not owner.case_records
            or not isinstance(source_context, dict)
            or not source_context
        ):
            raise ValueError("no live completed source invocation")
        seal = _live_seal(owner, pipeline)
        if seal is None:
            raise ValueError("no immutable suite seal")
        receipt = owner.receipt
        if receipt.output.is_symlink() or receipt.output.read_bytes() != receipt.raw:
            raise ValueError("changed compiled artifact")
        root_relative = owner.binding[2]
        source = [f for f in owner.binding[3] if f.relative == root_relative]
        if len(source) != 1 or source[0].raw.decode() != content:
            raise ValueError("different authored content")
        companion = pipeline._rulespec_test_path(owner.binding[1] / root_relative)
        companion_relative = companion.relative_to(owner.binding[1]).as_posix()
        companion_files = [
            f for f in owner.binding[3] if f.relative == companion_relative
        ]
        if len(companion_files) != 1:
            raise ValueError("unbound companion")
        original_cases = _safe_load_unique_keys(companion_files[0].raw.decode())
        if isinstance(original_cases, dict):
            original_cases = original_cases.get("cases")
        if type(original_cases) is not list or _typed(original_cases) != _typed(cases):
            raise ValueError("different companion cases")
        policy = _safe_load_unique_keys(content)
        normalized_policy, normalized_cases = _normalized_source_inputs(
            policy, original_cases
        )
        artifact = json.loads(receipt.raw, object_pairs_hook=custody._unique_object)
        expected = {}
        for ordinal, case in enumerate(original_cases, 1):
            request = _request(
                pipeline, owner.binding[1] / root_relative, artifact, case, ordinal
            )
            if request is not None:
                expected[ordinal] = _typed(request)
        observations = []
        ordinals = set()
        for record in owner.case_records:
            if (
                type(record) is not _Record
                or type(record.ordinal) is not int
                or not 1 <= record.ordinal <= len(original_cases)
                or record.ordinal in ordinals
            ):
                raise ValueError("invalid sealed record")
            ordinals.add(record.ordinal)
            request = json.loads(
                record.request_json, object_pairs_hook=custody._unique_object
            )
            response = json.loads(
                record.response_json, object_pairs_hook=custody._unique_object
            )
            if expected.get(record.ordinal) != _typed(request):
                raise ValueError("record belongs to another companion case")
            values = check_compiled_replay(artifact, policy, request, response)
            observations.append(
                (
                    frozenset(
                        {
                            _source_key(original_cases[record.ordinal - 1]),
                            _source_key(normalized_cases[record.ordinal - 1]),
                        }
                    ),
                    request,
                    response,
                    values,
                )
            )
        if ordinals != set(expected):
            raise ValueError("incomplete ordinary suite")
        bound = _Bound(
            owner,
            owner.generation,
            receipt,
            owner.case_records,
            content,
            cases,
            _typed(cases),
            source_context,
            _typed(source_context),
            normalized_policy,
            tuple(observations),
            seal,
        )
        if not bound.fresh():
            raise ValueError("expired source invocation")
        token = object()
        _contexts[token] = bound
    except _DECLINE:
        token = None
    try:
        yield token
    finally:
        if token is not None:
            _contexts.pop(token, None)
            for replacement, (context, _, _, _) in tuple(_site_replacements.items()):
                if context is token:
                    _site_replacements.pop(replacement, None)


def observed_dependencies(
    token: object | None,
    *,
    content: str,
    case: dict,
    principal: str,
    source_text: str,
    citation: str,
    principal_rules: dict,
) -> dict | None:
    """Return a copy of reached ROOT values only; never flatten provider IDs."""
    try:
        bound = _contexts.get(token)
        if bound is None or not bound.fresh() or content != bound.content:
            return None
        body = bound.sources.get(citation)
        if not isinstance(body, str) or body.strip() != source_text:
            return None
        declarations = {r["name"]: r for r in bound.policy["rules"]}
        if any(
            _source_key(rule) != _source_key(declarations[name])
            for name, rule in principal_rules.items()
        ):
            return None
        matches = [o for o in bound.observations if _source_key(case) in o[0]]
        if len(matches) != 1:
            return None
        _, request, response, values = matches[0]
        query = request["queries"][0]
        root = query["entity_id"]
        trace = response["results"][0]["trace"]
        refs = [r for r in query["outputs"] if r.split("#", 1)[1] == principal]
        if len(refs) != 1 or not any(
            key in case.get("output", {}) for key in (principal, refs[0])
        ):
            return None
        pending, visited = refs[:], set()
        while pending:
            ref = pending.pop()
            if ref in visited:
                continue
            visited.add(ref)
            pending.extend(trace[ref].get("dependencies", []))
        result = {}
        for ref in visited:
            node = trace[ref]
            if node["entity_id"] == root:
                name = node["name"]
                result[name] = values[(name, root)]
        return result if bound.fresh() else None
    except _DECLINE:
        return None


@dataclass(frozen=True)
class _Witness:
    context: object
    content: str
    case: dict
    principal: str
    source_text: str
    citation: str
    principal_rules: dict


_witnesses: dict[object, _Witness] = {}


def corroborated_dependencies(witness: object | None, case: dict) -> dict | None:
    """Only a currently registered branch-local witness can supply values."""
    try:
        record = _witnesses.get(witness)
        if record is None or _source_key(case) != _source_key(record.case):
            return None
        return observed_dependencies(
            record.context,
            content=record.content,
            case=case,
            principal=record.principal,
            source_text=record.source_text,
            citation=record.citation,
            principal_rules=record.principal_rules,
        )
    except _DECLINE:
        return None


def corroborates(witness: object | None, case: dict, name: str, value: Any) -> bool:
    from .source_completeness import _formula_runtime_values_equal

    values = corroborated_dependencies(witness, case)
    return (
        values is not None
        and name in values
        and _formula_runtime_values_equal(values[name], value)
    )


def _canonical_formula_context(witness: object, case: dict) -> tuple[dict, dict] | None:
    from . import source_completeness as source
    from .actual_local_relation_evidence import _closed_interval, _wire_value

    record = _witnesses.get(witness)
    if record is None or _source_key(case) != _source_key(record.case):
        return None
    bound = _contexts.get(record.context)
    if bound is None or not bound.fresh():
        return None
    matches = [o for o in bound.observations if _source_key(case) in o[0]]
    if len(matches) != 1:
        return None
    request = matches[0][1]
    query = request["queries"][0]
    period = query["period"]
    frame = _closed_interval({k: period[k] for k in ("start", "end")})
    facts = {}
    for fact in request["dataset"]["inputs"]:
        interval = _closed_interval(fact["interval"])
        if (
            fact["entity_id"] == query["entity_id"]
            and interval is not None
            and frame is not None
            and interval[0] <= frame[0] <= frame[1] <= interval[1]
        ):
            name = fact["name"].split("#input.", 1)[1]
            if name in facts:
                return None
            facts[name] = _wire_value(fact["value"])
    return source._constant_rule_environment(bound.policy), {
        "period": period,
        "input": facts,
    }


def corroborated_execution(
    witness: object | None,
    rule: dict,
    case: dict,
    formula_environment: dict,
) -> Any:
    """Reconstruct an authored path with verified values, never a literal rewrite."""
    import re

    from . import source_completeness as source

    try:
        record = _witnesses.get(witness)
        values = corroborated_dependencies(witness, case)
        name = rule.get("name")
        if (
            record is None
            or values is None
            or name not in values
            or _source_key(rule) != _source_key(record.principal_rules[name])
        ):
            return None
        canonical = _canonical_formula_context(witness, case)
        if canonical is None:
            return None
        constants, canonical_case = canonical
        execution = source._case_formula_execution(
            rule,
            canonical_case,
            formula_environment=constants,
            dependency_environment=values,
        )
        source_execution = source._case_formula_execution(
            rule,
            case,
            formula_environment=constants,
            dependency_environment=values,
        )
        if (
            execution is not None
            and source_execution is not None
            and execution.trace == source_execution.trace
            and execution.leaf == source_execution.leaf
            and source._formula_runtime_values_equal(
                source._formula_execution_runtime_value(execution), values[name]
            )
            and source._formula_runtime_values_equal(
                source._formula_execution_runtime_value(source_execution), values[name]
            )
        ):
            return execution
        formula = source._rule_formula_text_for_case(rule, case)
        if (
            not isinstance(formula, str)
            or rule.get("rounding") is not None
            or re.fullmatch(
                r"sum\([A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\)",
                formula.strip(),
            )
            is None
        ):
            return None
        # Both authored provider replay and compiled path replay already checked
        # this exact aggregate, including every provider and edge. Keep its AST.
        value = values[name]
        return source._FormulaExecution(
            (),
            formula.strip(),
            (type(value).__name__, repr(value)),
            value == 0,
            source._formula_environment_for_case(constants, canonical_case),
        )
    except _DECLINE:
        return None


def _operation_site(
    witness: object, rule: dict, case: dict, branch: Any
) -> tuple | None:
    from .local_fraction_site_evidence import fraction_site

    try:
        record = _witnesses.get(witness)
        if record is None:
            return None
        bound = _contexts.get(record.context)
        canonical = _canonical_formula_context(witness, case)
        if bound is None or canonical is None or not bound.fresh():
            return None
        observations = [o for o in bound.observations if _source_key(case) in o[0]]
        if len(observations) != 1:
            return None
        _, request, response, _ = observations[0]
        refs = [
            r
            for r in request["queries"][0]["outputs"]
            if r.split("#", 1)[1] == rule["name"]
        ]
        if len(refs) != 1:
            return None
        site = fraction_site(
            bound.policy,
            rule,
            case,
            branch,
            record.source_text,
            record.citation,
            response["results"][0]["trace"][refs[0]],
            canonical[1]["input"],
        )
        return site if site is not None and bound.fresh() else None
    except _DECLINE:
        return None


def apply_operation_replacements(
    context: object, branch: Any, candidates: set, tokens: list
) -> None:
    """Consume only live, privately minted replacements before graph allocation."""
    for token in tokens:
        record = _site_replacements.pop(token, None)
        if record is None:
            continue
        owner, owned_branch, old, new = record
        bound = _contexts.get(context)
        if (
            owner is context
            and owned_branch is branch
            and bound is not None
            and bound.fresh()
            and old in candidates
        ):
            candidates.remove(old)
            candidates.add(new)


def formula_witnesses(
    context: object, branch: Any, **kwargs: Any
) -> set[tuple[str, str]]:
    """Add execution corroboration using private per-principal caches only."""
    from . import source_completeness as source

    content = kwargs.pop("content")
    replacement_tokens = kwargs.pop("_replacement_tokens", None)
    result = set()
    principal_rules = kwargs["principal_rules"]
    for name in sorted(kwargs["rule_names"]):
        for case in kwargs["asserted_by_rule"].get(name, ()):
            token = object()
            _witnesses[token] = _Witness(
                context,
                content,
                case,
                name,
                kwargs["source_text"],
                kwargs["corpus_citation_path"],
                principal_rules,
            )
            try:
                values = corroborated_dependencies(token, case)
                if values is None or name not in values:
                    continue
                rule = principal_rules[name]
                if not source._asserted_formula_runtime_values_equal(
                    rule,
                    values[name],
                    source._test_case_asserted_output_value(case, name),
                ):
                    continue
                execution = corroborated_execution(
                    token, rule, case, kwargs["formula_environment"]
                )
                if execution is None:
                    continue
                canonical = _canonical_formula_context(token, case)
                if canonical is None:
                    continue
                private = dict(kwargs)
                private.update(
                    formula_environment=canonical[0],
                    rule_names={name},
                    asserted_by_rule={name: [case]},
                    dependency_cache={id(case): values},
                    execution_cache={(name, id(case)): execution},
                    local_corroboration=token,
                )
                # Coverage side effects share the same commit boundary as the
                # witness: an expired observation must leave no numeric credit.
                coverage_sets = {}
                for key in (
                    "represented_annual_spans",
                    "represented_measurement_spans",
                    "measurement_witnesses",
                ):
                    if isinstance(private.get(key), set):
                        coverage_sets[key] = private[key]
                        private[key] = set(private[key])
                candidates = source._formula_branch_test_witnesses(branch, **private)
                old = (name, f"case:{id(case)}")
                site = (
                    _operation_site(token, rule, case, branch)
                    if type(replacement_tokens) is list and old in candidates
                    else None
                )
                # Source checks cannot outlive or mutate the observation.
                if corroborated_dependencies(token, case) is not None:
                    result.update(candidates)
                    if site is not None:
                        replacement = object()
                        new = (
                            name,
                            f"case:{id(case)}|version:{site[0]}|site:{site[1]}|node:{site[2]}",
                        )
                        _site_replacements[replacement] = (context, branch, old, new)
                        replacement_tokens.append(replacement)
                    for key, original in coverage_sets.items():
                        original.update(private[key])
            except _DECLINE:
                continue
            finally:
                _witnesses.pop(token, None)
    return result


def _year_occurrence_context(
    context: object | None,
    *,
    content: str,
    cases: Any,
    source_text: str,
    citation: str,
    raw_span: tuple[int, int],
    occurrence: Any,
    recall_text: str,
    recall_map: Any,
) -> tuple[_Bound, list] | None:
    """Shared strict origin/custody check; callers must bind a closed source role."""
    from . import source_completeness as source
    from .validator_pipeline import (
        NumericOccurrence,
        _numeric_profile_for_citation_path,
        extract_typed_numeric_inventory_occurrences_from_text,
    )

    try:
        if (
            type(raw_span) is not tuple
            or len(raw_span) != 2
            or any(type(i) is not int for i in raw_span)
        ):
            return None
        bound = _contexts.get(context)
        if (
            bound is None
            or not bound.fresh()
            or content != bound.content
            or bound.sources.get(citation, "").strip() != source_text
            or type(occurrence) is not NumericOccurrence
            or type(recall_map) is not source._NumericRecallSourceMap
            or any(type(i) is not int for i in (occurrence.start, occurrence.end))
            or not 0 <= occurrence.start < occurrence.end <= len(recall_text)
            or any(
                type(i) is not int
                for i in recall_map.offsets[occurrence.start : occurrence.end]
            )
            or recall_map.source_span(occurrence.start, occurrence.end) != raw_span
            or recall_text[occurrence.start : occurrence.end] != occurrence.raw
            or source_text[raw_span[0] : raw_span[1]] != occurrence.raw
            or occurrence.raw not in {"2022", "2023", "2024"}
            or type(occurrence.value) is not float
            or type(occurrence.source_value) is not float
            or occurrence.value != float(occurrence.raw)
            or occurrence.source_value != occurrence.value
            or any(
                getattr(occurrence, flag) is not False
                for flag in (
                    "has_rate_context",
                    "has_temporal_context",
                    "has_structural_context",
                    "requires_rate_context",
                    "requires_month_context",
                    "is_word_number",
                )
            )
            or type(occurrence.alternative_values) is not tuple
            or occurrence.alternative_values != ()
        ):
            return None
        # Independently recover this occurrence from the original authoritative
        # body. Caller masks may shift offsets, never change its numeric meaning.
        original_map = source._NumericRecallSourceMap(list(range(len(source_text))))
        original_recall = source.authoritative_numeric_recall_text(
            source_text,
            corpus_citation_path=citation,
            _source_map=original_map,
        )
        original_occurrences = extract_typed_numeric_inventory_occurrences_from_text(
            original_recall,
            profile=_numeric_profile_for_citation_path(citation),
        )
        originals = [
            o
            for o in original_occurrences
            if original_map.source_span(o.start, o.end) == raw_span
        ]
        if len(originals) != 1 or any(
            type(getattr(originals[0], field)) is not type(getattr(occurrence, field))
            or getattr(originals[0], field) != getattr(occurrence, field)
            for field in NumericOccurrence.__dataclass_fields__
            if field not in {"start", "end"}
        ):
            return None
        _, normalized = _normalized_source_inputs(bound.policy, bound.cases)
        if _source_key(cases) != _source_key(normalized):
            return None
        return (bound, normalized) if bound.fresh() else None
    except _DECLINE:
        return None


def vehicle_year_occurrence(
    context: object | None,
    *,
    content: str,
    cases: Any,
    source_text: str,
    citation: str,
    raw_span: tuple[int, int] | None,
    occurrence: Any,
    recall_text: str,
    recall_map: Any,
) -> bool:
    """Consume a single mapped source occurrence from live direct-root cases.

    No descriptor, stored trace or same-valued scalar is an authority. Require
    below/equal/above-cap effects in the authenticated complete suite, using
    only the queried SourceEntry root and its immutable ordinary request.
    """
    from datetime import date

    from . import source_completeness as source
    from .local_vehicle_date_evidence import _ROLES, vehicle_date_role_binding

    try:
        if (
            type(raw_span) is not tuple
            or len(raw_span) != 2
            or any(type(i) is not int for i in raw_span)
            or raw_span not in {(r[0], r[0] + 4) for r in _ROLES}
        ):
            return False
        checked = _year_occurrence_context(
            context,
            content=content,
            cases=cases,
            source_text=source_text,
            citation=citation,
            raw_span=raw_span,
            occurrence=occurrence,
            recall_text=recall_text,
            recall_map=recall_map,
        )
        if checked is None:
            return False
        bound, normalized = checked
        rules = {
            r["name"]: r for r in bound.policy["rules"] if r.get("kind") == "derived"
        }
        principal = "t936_asset_included_uncollectible_portion"
        controls = set()
        for case in normalized:
            if not any(k.split("#")[-1] == principal for k in case.get("output", {})):
                continue
            role = vehicle_date_role_binding(
                bound.policy, source_text, citation, case, raw_span
            )
            if role is None or not any(
                k.split("#")[-1] == role.selector for k in case.get("output", {})
            ):
                continue
            token = object()
            _witnesses[token] = _Witness(
                context, content, case, principal, source_text, citation, rules
            )
            try:
                canonical = _canonical_formula_context(token, case)
                values = corroborated_dependencies(token, case)
                if canonical is None or values is None:
                    continue
                facts = canonical[1]["input"]
                acquired = facts.get("t936_asset_acquisition_date")
                cost = facts.get("t936_asset_acquisition_cost")
                portion = facts.get("t936_asset_uncollectible_portion")
                ordinary = role.subtype == "ordinary"
                if (
                    type(acquired) is not date
                    or (role.lower is not None and acquired < role.lower)
                    or (role.upper is not None and acquired >= role.upper)
                    or facts.get("t936_asset_is_timber_resource") is not False
                    or facts.get("t936_asset_is_ordinary_passenger_vehicle")
                    is not ordinary
                    or facts.get("t936_asset_is_zero_emission_passenger_vehicle")
                    is not (not ordinary)
                    or type(cost) not in (int, Decimal)
                    or type(portion) not in (int, Decimal)
                    or not Decimal(cost).is_finite()
                    or not Decimal(portion).is_finite()
                    or portion <= 0
                ):
                    continue
                # Reached root values are linked to the ordinary queried root;
                # the factory independently checked its authored/compiled path.
                excluded = cost > role.cap
                if values.get(role.selector) is not excluded:
                    continue
                expected = 0 if excluded else portion
                if values.get(principal) != expected:
                    continue
                if any(
                    not source._asserted_formula_runtime_values_equal(
                        rules[name],
                        values[name],
                        source._test_case_asserted_output_value(case, name),
                    )
                    for name in (principal, role.selector)
                ):
                    continue
                if (
                    corroborated_execution(token, rules[principal], case, canonical[0])
                    is None
                ):
                    continue
                if corroborated_dependencies(token, case) is None:
                    continue
                controls.add(-1 if cost < role.cap else 0 if cost == role.cap else 1)
            finally:
                _witnesses.pop(token, None)
        return controls == {-1, 0, 1} and bound.fresh()
    except _DECLINE:
        return False


def prior_form_year_occurrence(
    context: object | None,
    *,
    content: str,
    cases: Any,
    source_text: str,
    citation: str,
    raw_span: tuple[int, int] | None,
    occurrence: Any,
    recall_text: str,
    recall_map: Any,
) -> bool:
    """Prove both source-owned prior-report routes from existing actual roots."""
    from . import source_completeness as source
    from .local_prior_form_evidence import _SPANS, prior_form_reference_binding

    try:
        if (
            type(raw_span) is not tuple
            or len(raw_span) != 2
            or any(type(i) is not int for i in raw_span)
            or raw_span not in {(v, v + 4) for v in _SPANS}
        ):
            return False
        checked = _year_occurrence_context(
            context,
            content=content,
            cases=cases,
            source_text=source_text,
            citation=citation,
            raw_span=raw_span,
            occurrence=occurrence,
            recall_text=recall_text,
            recall_map=recall_map,
        )
        if checked is None:
            return False
        bound, normalized = checked
        rules = {
            r["name"]: r for r in bound.policy["rules"] if r.get("kind") == "derived"
        }
        principal = f"t936_previous_{_SPANS[raw_span[0]]}"
        copied = False
        reconstructed = False
        positive_years = set()
        for case in normalized:
            if not any(k.split("#")[-1] == principal for k in case.get("output", {})):
                continue
            role = prior_form_reference_binding(
                bound.policy, source_text, citation, case, raw_span
            )
            if role is None:
                continue
            token = object()
            _witnesses[token] = _Witness(
                context, content, case, principal, source_text, citation, rules
            )
            try:
                canonical = _canonical_formula_context(token, case)
                values = corroborated_dependencies(token, case)
                if canonical is None or values is None or principal not in values:
                    continue
                facts = canonical[1]["input"]
                selector = facts.get(role.selector)
                if type(selector) is not bool:
                    continue
                execution = corroborated_execution(
                    token, rules[principal], case, canonical[0]
                )
                if (
                    execution is None
                    or not source._asserted_formula_runtime_values_equal(
                        rules[principal],
                        values[principal],
                        source._test_case_asserted_output_value(case, principal),
                    )
                ):
                    continue
                required = (role.copy_input,) if selector else role.annual_inputs
                if any(
                    type(facts.get(n)) not in (int, Decimal)
                    or not Decimal(facts[n]).is_finite()
                    for n in required
                ):
                    continue
                expected_leaf = (
                    role.copy_input if selector else " + ".join(role.annual_inputs)
                )
                if execution.leaf.strip() != expected_leaf:
                    continue
                if corroborated_dependencies(token, case) is None:
                    continue
                if selector:
                    copied |= (
                        values[principal] == facts[role.copy_input]
                        and facts[role.copy_input] > 0
                    )
                else:
                    reconstructed = True
                    positive_years.update(
                        y
                        for y, n in zip(
                            range(1988, 2025), role.annual_inputs, strict=True
                        )
                        if facts[n] > 0
                    )
            finally:
                _witnesses.pop(token, None)
        return (
            copied
            and reconstructed
            and {1988, 2024}.issubset(positive_years)
            and any(1988 < y < 2024 for y in positive_years)
            and bound.fresh()
        )
    except _DECLINE:
        return False


def _pair_changed_fact(left: dict, right: dict, *, boolean: bool = False) -> str | None:
    """Compare complete requests, allowing query selection and one root value only."""
    from copy import deepcopy

    from .actual_local_relation_evidence import _wire_value

    try:
        a, b = deepcopy(left), deepcopy(right)
        if len(a["queries"]) != 1 or len(b["queries"]) != 1:
            return None
        qa, qb = a["queries"][0], b["queries"][0]
        qa.pop("outputs")
        qb.pop("outputs")
        if _typed(qa) != _typed(qb):
            return None
        ai, bi = a["dataset"].pop("inputs"), b["dataset"].pop("inputs")
        if _typed(a) != _typed(b) or len(ai) != len(bi):
            return None
        changed = []
        for x, y in zip(ai, bi, strict=True):
            xv, yv = x.pop("value"), y.pop("value")
            if _typed(x) != _typed(y):
                return None
            if _typed(xv) == _typed(yv):
                continue
            if x["entity_id"] != qa["entity_id"] or xv["kind"] != yv["kind"]:
                return None
            vx, vy = _wire_value(xv), _wire_value(yv)
            if type(boolean) is not bool:
                return None
            if boolean:
                if type(vx) is not bool or type(vy) is not bool or vx is vy:
                    return None
            elif (
                type(vx) not in (int, Decimal)
                or type(vy) is not type(vx)
                or not Decimal(vx).is_finite()
                or not Decimal(vy).is_finite()
                or vx == vy
            ):
                return None
            changed.append(x["name"].split("#input.", 1)[1])
        return changed[0] if len(changed) == 1 else None
    except (*_DECLINE, IndexError):
        return None


def numeric_pair_witnesses(
    context: object | None,
    *,
    content: str,
    cases: Any,
    principal_rules: dict,
    source_text: str,
    citation: str,
) -> set:
    """Fresh two-record common-principal observations; original queries stay intact."""
    from itertools import combinations

    from . import source_completeness as source

    found = set()
    try:
        bound = _contexts.get(context)
        if bound is None or not bound.fresh() or content != bound.content:
            return found
        _, normalized = _normalized_source_inputs(bound.policy, bound.cases)
        if _source_key(cases) != _source_key(normalized):
            return found
        for name, rule in principal_rules.items():
            selector_names = source._rule_numeric_selector_names(rule)
            if not selector_names:
                continue
            endpoints = []
            for case in cases:
                # Cheap filter only; observed_dependencies verifies the exact query.
                if not any(k.split("#", 1)[-1] == name for k in case.get("output", {})):
                    continue
                token = object()
                _witnesses[token] = _Witness(
                    context, content, case, name, source_text, citation, principal_rules
                )
                try:
                    canonical = _canonical_formula_context(token, case)
                    values = corroborated_dependencies(token, case)
                    if canonical is None or values is None or name not in values:
                        continue
                    execution = corroborated_execution(token, rule, case, canonical[0])
                    if (
                        execution is None
                        or not source._asserted_formula_runtime_values_equal(
                            rule,
                            values[name],
                            source._test_case_asserted_output_value(case, name),
                        )
                    ):
                        continue
                    observations = [
                        o for o in bound.observations if _source_key(case) in o[0]
                    ]
                    if len(observations) != 1:
                        continue
                    endpoints.append(
                        (case, observations[0][1], values, canonical, execution)
                    )
                finally:
                    _witnesses.pop(token, None)
            for left, right in combinations(endpoints, 2):
                lc, lr, lv, lcanon, le = left
                rc, rr, rv, rcanon, re = right
                changed = _pair_changed_fact(lr, rr)
                if changed is None:
                    continue
                if not source._exception_effect_changes(lv[name], rv[name]):
                    continue
                selector_values = {
                    n: (lv[n], rv[n])
                    for n in selector_names & lv.keys() & rv.keys()
                    if type(lv[n]) in (int, Decimal)
                    and type(rv[n]) in (int, Decimal)
                    and lv[n] != rv[n]
                }
                if not selector_values:
                    continue
                transitions = source._formula_relational_transitions(
                    le,
                    re,
                    left_case=lcanon[1],
                    right_case=rcanon[1],
                    left_dependencies=lv,
                    right_dependencies=rv,
                    formula_environment=lcanon[0],
                    changed_names=set(selector_values),
                )
                arithmetic_relations = source._formula_arithmetic_relation_descriptors(
                    le.leaf
                ) | source._formula_arithmetic_relation_descriptors(re.leaf)
                if (
                    (le.trace or re.trace)
                    and source._formula_leaf_semantic_key(
                        le.leaf, formula_environment=le.constant_environment
                    )
                    == source._formula_leaf_semantic_key(
                        re.leaf, formula_environment=re.constant_environment
                    )
                    and not arithmetic_relations.intersection(
                        relation for group in transitions for relation in group
                    )
                ):
                    continue
                for selector, pair in selector_values.items():
                    # A reached comparison is a property of the complete pair.
                    # Preserve both causal directions, including the zeroing arm.
                    if not (
                        (
                            source._formula_execution_reaches_selector(le, selector)
                            and source._formula_execution_reaches_selector(re, selector)
                        )
                        or any(
                            selector in (a, b)
                            for group in transitions
                            for a, _, b in group
                        )
                    ):
                        continue
                    for i, (ordinary, exceptional, relations, maxima) in enumerate(
                        (
                            (lv[name], rv[name], transitions[0], transitions[2]),
                            (rv[name], lv[name], transitions[1], transitions[3]),
                        )
                    ):
                        found.add(
                            source._ExceptionWitness(
                                name,
                                selector,
                                True,
                                source._exception_effect_is_blocking(
                                    ordinary, exceptional
                                ),
                                False,
                                source._exception_effect_is_zero(exceptional),
                                pair if i == 0 else pair[::-1],
                                relations,
                                source._case_pair_identity(lc, rc),
                                maximum_transitions=maxima,
                            )
                        )
        return found if bound.fresh() else set()
    except _DECLINE:
        return set()


def boolean_pair_witnesses(
    context: object | None,
    *,
    content: str,
    cases: Any,
    principal_rules: dict,
    source_text: str,
    citation: str,
) -> set:
    """Direct raw-Boolean interventions between two completely observed worlds."""
    from copy import deepcopy
    from itertools import combinations

    from . import source_completeness as source

    found = set()
    try:
        bound = _contexts.get(context)
        if bound is None or not bound.fresh() or content != bound.content:
            return found
        _, normalized = _normalized_source_inputs(bound.policy, bound.cases)
        if _source_key(cases) != _source_key(normalized):
            return found
        declared_booleans = {
            r["name"]
            for r in bound.policy.get("inputs", [])
            if r.get("dtype") == "Boolean"
        }
        derived_names = {
            r["name"] for r in bound.policy["rules"] if r.get("kind") == "derived"
        }
        for name, rule in principal_rules.items():
            selectors = source._rule_exception_selector_names(rule) & declared_booleans
            if not selectors:
                continue
            endpoints = []
            for case in cases:
                if not any(k.split("#", 1)[-1] == name for k in case.get("output", {})):
                    continue
                token = object()
                _witnesses[token] = _Witness(
                    context, content, case, name, source_text, citation, principal_rules
                )
                try:
                    canonical = _canonical_formula_context(token, case)
                    values = corroborated_dependencies(token, case)
                    if canonical is None or values is None or name not in values:
                        continue
                    execution = corroborated_execution(token, rule, case, canonical[0])
                    if (
                        execution is None
                        or not source._asserted_formula_runtime_values_equal(
                            rule,
                            values[name],
                            source._test_case_asserted_output_value(case, name),
                        )
                    ):
                        continue
                    observations = [
                        o for o in bound.observations if _source_key(case) in o[0]
                    ]
                    if len(observations) != 1:
                        continue
                    # Raw facts belong to canonical inputs, never to the derived
                    # dependency interface used by the intervention helper.
                    dependencies = {
                        n: v for n, v in values.items() if n in derived_names
                    }
                    endpoints.append(
                        (case, observations[0][1], dependencies, canonical, execution)
                    )
                finally:
                    _witnesses.pop(token, None)
            for left, right in combinations(endpoints, 2):
                lc, lr, ld, lcanon, le = left
                rc, rr, rd, rcanon, re = right
                changed = _pair_changed_fact(lr, rr, boolean=True)
                if changed not in selectors or changed in derived_names:
                    continue
                lvalue, rvalue = (
                    lcanon[1]["input"].get(changed),
                    rcanon[1]["input"].get(changed),
                )
                if (
                    type(lvalue) is not bool
                    or type(rvalue) is not bool
                    or lvalue is rvalue
                ):
                    continue
                counterfactual = deepcopy(lcanon[1])
                counterfactual["input"][changed] = rvalue
                if _source_key(counterfactual) != _source_key(rcanon[1]):
                    continue
                if not all(
                    source._formula_execution_reaches_selector(e, changed)
                    for e in (le, re)
                ):
                    continue
                for ordinary, exception in ((left, right), (right, left)):
                    oc, _, od, ocanon, _ = ordinary
                    ec, _, ed, ecanon, _ = exception
                    active = ecanon[1]["input"][changed]
                    if not source._direct_boolean_intervention_matches_case(
                        oc,
                        ec,
                        selector_name=changed,
                        active_value=active,
                        ordinary_dependencies=od,
                        exception_dependencies=ed,
                    ):
                        continue
                    witness = source._exception_witness_for_case_pair(
                        name,
                        rule,
                        principal_rules=principal_rules,
                        selector_name=changed,
                        active_value=active,
                        left_case=oc,
                        left_dependencies=od,
                        left_selector_value=ocanon[1]["input"][changed],
                        right_case=ec,
                        right_dependencies=ed,
                        formula_environment=ocanon[0],
                    )
                    if witness is not None:
                        found.add(witness)
        return found if bound.fresh() else set()
    except _DECLINE:
        return set()


def paired_branch_witnesses(
    context: object | None,
    branch: Any,
    *,
    content: str,
    cases: Any,
    principal_rules: dict,
    source_text: str,
    citation: str,
) -> set:
    """Fresh source-owned operation pairs; persisted observations grant nothing."""
    import ast
    from decimal import Context, localcontext
    from itertools import combinations

    from . import source_completeness as source
    from .actual_local_relation_evidence import _exact_engine_decimal
    from .local_pair_source_evidence import bind_pair_role

    found = set()
    try:
        bound = _contexts.get(context)
        if bound is None or not bound.fresh() or content != bound.content:
            return found
        _, normalized = _normalized_source_inputs(bound.policy, bound.cases)
        if _source_key(cases) != _source_key(normalized):
            return found
        for name, rule in principal_rules.items():
            role = bind_pair_role(bound.policy, branch, source_text, citation, name)
            if role is None:
                continue
            endpoints = []
            for case in cases:
                token = object()
                _witnesses[token] = _Witness(
                    context, content, case, name, source_text, citation, principal_rules
                )
                try:
                    canonical = _canonical_formula_context(token, case)
                    values = corroborated_dependencies(token, case)
                    if canonical is None or values is None or name not in values:
                        continue
                    if canonical[1]["period"] != {
                        "period_kind": "tax_year",
                        "start": "2025-01-01",
                        "end": "2025-12-31",
                    }:
                        continue
                    execution = corroborated_execution(token, rule, case, canonical[0])
                    if (
                        execution is None
                        or not source._asserted_formula_runtime_values_equal(
                            rule,
                            values[name],
                            source._test_case_asserted_output_value(case, name),
                        )
                    ):
                        continue
                    observations = [
                        o for o in bound.observations if _source_key(case) in o[0]
                    ]
                    if len(observations) != 1:
                        continue
                    if role.operation == "boolean":
                        selector = role.operands[0]
                        state = canonical[1]["input"].get(selector)
                        if type(
                            state
                        ) is not bool or not source._formula_execution_reaches_selector(
                            execution, selector
                        ):
                            continue
                        signed = state
                    else:
                        left, right = role.operands
                        operands = dict(values)
                        if role.operation == "multiply":
                            from .actual_local_relation_evidence import _wire_value

                            response = observations[0][2]
                            nodes = [
                                n
                                for result in response["results"]
                                for n in result["trace"].values()
                                if n.get("name") == name
                            ]
                            if len(nodes) != 1:
                                continue
                            reads = [
                                r
                                for r in nodes[0].get("parameter_reads", [])
                                if r.get("name") == right
                            ]
                            if (
                                len(reads) != 1
                                or type(reads[0].get("index")) is not int
                                or reads[0]["index"] != 0
                            ):
                                continue
                            rate = _wire_value(reads[0]["value"])
                            if (
                                rate != Decimal("0.5")
                                or source._rulespec_runtime_decimal(
                                    source._formula_environment_for_case(
                                        canonical[0], case
                                    ).get(right)
                                )
                                != rate
                            ):
                                continue
                            operands[right] = rate
                        if left not in operands or right not in operands:
                            continue
                        a, b = (
                            _exact_engine_decimal(operands[left]),
                            _exact_engine_decimal(operands[right]),
                        )
                        if a is None or b is None:
                            continue
                        # A shortcut output of zero is never a reached floor.
                        op = "-" if role.operation == "subtract" else "*"
                        expected = f"max(0, {left} {op} {right})"
                        if ast.dump(ast.parse(execution.leaf)) != ast.dump(
                            ast.parse(expected)
                        ):
                            continue
                        with localcontext(Context(prec=80)):
                            signed = _exact_engine_decimal(
                                (a - b if op == "-" else a * b).normalize()
                            )
                        if signed is None or not source._formula_runtime_values_equal(
                            values[name], max(Decimal(0), signed)
                        ):
                            continue
                    endpoints.append(
                        (case, observations[0][1], values, canonical, signed)
                    )
                finally:
                    _witnesses.pop(token, None)
            for left, right in combinations(endpoints, 2):
                lc, lr, lv, lcanon, ls = left
                rc, rr, rv, rcanon, rs = right
                changed = _pair_changed_fact(
                    lr, rr, boolean=role.operation == "boolean"
                )
                if changed is None:
                    continue
                if role.operation == "boolean":
                    if changed != role.operands[0] or ls is rs:
                        continue
                    # Both actual arms and their complete unchanged factual world.
                    off, on = (
                        (lv[name], rv[name]) if ls is False else (rv[name], lv[name])
                    )
                    if _exact_engine_decimal(off) != 0 or (
                        _exact_engine_decimal(on) is None or on <= 0
                    ):
                        continue
                elif not (ls < 0 < rs or rs < 0 < ls):
                    continue
                found.add((name, source._case_pair_identity(lc, rc)))
        return found if bound.fresh() else set()
    except (*_DECLINE, SyntaxError):
        return set()


def signed_branch_witnesses(
    context: object | None,
    branch: Any,
    *,
    content: str,
    cases: Any,
    principal_rules: dict,
    source_text: str,
    citation: str,
) -> set:
    """Faithful signed own-outputs; mixed row8 additionally requires its gate.

    This proves numeric representation, not how a form renders parentheses.
    It cannot supply a notation-only token for the conditional copied row.
    """
    import ast
    from copy import deepcopy

    from . import source_completeness as source
    from .actual_local_relation_evidence import _checked_sum, _exact_engine_decimal
    from .local_pair_source_evidence import bind_signed_role

    found = set()
    try:
        bound = _contexts.get(context)
        if bound is None or not bound.fresh() or content != bound.content:
            return found
        _, normalized = _normalized_source_inputs(bound.policy, bound.cases)
        if _source_key(cases) != _source_key(normalized):
            return found
        for name, rule in principal_rules.items():
            role = bind_signed_role(bound.policy, branch, source_text, citation, name)
            if role is None:
                continue
            endpoints = []
            for case in cases:
                token = object()
                _witnesses[token] = _Witness(
                    context, content, case, name, source_text, citation, principal_rules
                )
                try:
                    canonical = _canonical_formula_context(token, case)
                    values = corroborated_dependencies(token, case)
                    if canonical is None or values is None or name not in values:
                        continue
                    period = {
                        "period_kind": "tax_year",
                        "start": "2025-01-01",
                        "end": "2025-12-31",
                    }
                    if canonical[1]["period"] != period:
                        continue
                    execution = corroborated_execution(token, rule, case, canonical[0])
                    observed = _exact_engine_decimal(values[name])
                    if (
                        execution is None
                        or observed is None
                        or not source._asserted_formula_runtime_values_equal(
                            rule,
                            values[name],
                            source._test_case_asserted_output_value(case, name),
                        )
                    ):
                        continue
                    observations = [
                        o for o in bound.observations if _source_key(case) in o[0]
                    ]
                    if len(observations) != 1:
                        continue
                    request = observations[0][1]
                    query = deepcopy(request["queries"][0])
                    query.pop("outputs")
                    query_identity = _typed(query)
                    state = True
                    if role.selector is not None:
                        state = canonical[1]["input"].get(role.selector)
                        if type(
                            state
                        ) is not bool or not source._formula_execution_reaches_selector(
                            execution, role.selector
                        ):
                            continue
                    if not state:
                        if observed != 0 or ast.dump(
                            ast.parse(execution.leaf)
                        ) != ast.dump(ast.parse("0")):
                            continue
                    else:
                        expected = " + ".join(role.operands)
                        if ast.dump(ast.parse(execution.leaf)) != ast.dump(
                            ast.parse(expected)
                        ):
                            continue
                        environment = {**canonical[1]["input"], **values}
                        calculated = _checked_sum(environment[n] for n in role.operands)
                        if calculated is None or calculated != observed:
                            continue
                    endpoints.append(
                        (case, request, canonical, query_identity, observed, state)
                    )
                finally:
                    _witnesses.pop(token, None)
            negative = [e for e in endpoints if e[5] and e[4] < 0]
            nonnegative = [e for e in endpoints if e[5] and e[4] >= 0]
            for neg in negative:
                for pos in nonnegative:
                    if neg[3] != pos[3]:
                        continue
                    pair = source._case_pair_identity(neg[0], pos[0])
                    if role.selector is None:
                        found.add((name, pair))
                        continue
                    if pos[4] <= 0:
                        continue
                    for off in endpoints:
                        if off[5] or off[3] != pos[3] or off[4] != 0:
                            continue
                        if (
                            _pair_changed_fact(pos[1], off[1], boolean=True)
                            != role.selector
                        ):
                            continue
                        counterfactual = deepcopy(pos[2][1])
                        counterfactual["input"][role.selector] = False
                        if _source_key(counterfactual) != _source_key(off[2][1]):
                            continue
                        identity = tuple(
                            sorted({*pair, *source._case_pair_identity(pos[0], off[0])})
                        )
                        if len(identity) == 3:
                            found.add((name, identity))
        # Alternative successful endpoint sets prove the same source duty.
        # They must not create capacity for a duplicate parent obligation.
        return {min(found)} if found and bound.fresh() else set()
    except (*_DECLINE, SyntaxError):
        return set()


def chart_duty_witnesses(
    context: object | None,
    branch: Any,
    *,
    content: str,
    cases: Any,
    principal_rules: dict,
    source_text: str,
    citation: str,
) -> set:
    """Source-owned Chart duty using the existing live Boolean intervention proof."""
    from .local_pair_source_evidence import bind_chart_duty

    try:
        bound = _contexts.get(context)
        if bound is None or not bound.fresh() or content != bound.content:
            return set()
        role = bind_chart_duty(bound.policy, branch, source_text, citation)
        if role is None or role[0] not in principal_rules:
            return set()
        _, normalized = _normalized_source_inputs(bound.policy, bound.cases)
        if _source_key(cases) != _source_key(normalized):
            return set()
        eligible = set()
        for case in cases:
            observations = [o for o in bound.observations if _source_key(case) in o[0]]
            if len(observations) != 1:
                continue
            queries = observations[0][1]["queries"]
            if len(queries) == 1 and queries[0].get("period") == {
                "period_kind": "tax_year",
                "start": "2025-01-01",
                "end": "2025-12-31",
            }:
                eligible.add(id(case))
        witnesses = boolean_pair_witnesses(
            context,
            content=content,
            cases=cases,
            principal_rules=principal_rules,
            source_text=source_text,
            citation=citation,
        )
        found = {
            (role[0], w.case_pair_identity)
            for w in witnesses
            if w.rule_name == role[0]
            and w.selector_name == role[1]
            and w.boolean_effect
            and w.active_value is True
            and len(w.case_pair_identity) == 2
            and len(set(w.case_pair_identity)) == 2
            and set(w.case_pair_identity) <= eligible
        }
        return {min(found)} if found and bound.fresh() else set()
    except _DECLINE:
        return set()


def _ordered_conditional_path(trace, selectors: tuple[str, ...], choice: int) -> bool:
    """Recognize flat or exact nested arms after a complete static source bind.

    This does not rewrite captured traces or grant evidence on its own. Callers
    retain exact authored formulas, reached leaves, custody and freshness.
    """
    if type(choice) is not int or not 0 <= choice <= len(selectors) or not selectors:
        return False
    if any(type(step.choice) is not int for step in trace):
        return False
    if len(trace) == 1:
        step = trace[0]
        if (
            step.kind == "if"
            and step.choice == choice
            and step.selectors == selectors[: choice + 1]
        ):
            return True
    expected_length = min(choice + 1, len(selectors))
    if len(trace) != expected_length:
        return False
    return all(
        step.kind == "if"
        and step.selectors == (selectors[index],)
        and step.choice == (0 if index == choice else 1)
        for index, step in enumerate(trace)
    )


def remaining_chart_witnesses(
    context: object | None,
    branch: Any,
    *,
    content: str,
    cases: Any,
    principal_rules: dict,
    source_text: str,
    citation: str,
) -> set:
    """Four closed Chart roles backed by original complete execution records."""
    import ast
    from itertools import combinations

    from . import source_completeness as source
    from .actual_local_relation_evidence import (
        _checked_sum,
        _exact_engine_decimal,
        _wire_value,
    )
    from .local_pair_source_evidence import bind_remaining_chart_role

    row3 = "federal_2025_t936_chart_a_line_3"
    row13 = "federal_2025_t936_chart_a_line_13"
    row14 = "t936_chart_a_line_14_non_eligible_taxable_capital_gains"
    row19 = "federal_2025_t936_chart_a_line_19"
    row20 = "federal_2025_t936_chart_a_additional_investment_income_line_20"
    raw = "schedule_3_line_19900"
    selector = "t936_chart_a_line_14_includes_t3"
    rate_name = "federal_2025_t936_chart_a_inclusion_rate"
    found = set()
    try:
        bound = _contexts.get(context)
        if bound is None or not bound.fresh() or content != bound.content:
            return set()
        _, normalized = _normalized_source_inputs(bound.policy, bound.cases)
        if _source_key(cases) != _source_key(normalized):
            return set()
        role = bind_remaining_chart_role(bound.policy, branch, source_text, citation)
        required = {
            "floor14": (row14,),
            "second_shortcut": (row20,),
            "t3_conditional": (row19, row20),
            "line1_mixed": (row3, row20),
        }.get(role)
        if required is None or any(n not in principal_rules for n in required):
            return set()
        endpoints = {name: {} for name in required}
        for name in required:
            rule = principal_rules[name]
            for case in cases:
                token = object()
                _witnesses[token] = _Witness(
                    context, content, case, name, source_text, citation, principal_rules
                )
                try:
                    canonical = _canonical_formula_context(token, case)
                    values = corroborated_dependencies(token, case)
                    if canonical is None or values is None or name not in values:
                        continue
                    if canonical[1]["period"] != {
                        "period_kind": "tax_year",
                        "start": "2025-01-01",
                        "end": "2025-12-31",
                    }:
                        continue
                    execution = corroborated_execution(token, rule, case, canonical[0])
                    if (
                        execution is None
                        or not source._asserted_formula_runtime_values_equal(
                            rule,
                            values[name],
                            source._test_case_asserted_output_value(case, name),
                        )
                    ):
                        continue
                    obs = [o for o in bound.observations if _source_key(case) in o[0]]
                    if len(obs) != 1 or _exact_engine_decimal(values[name]) is None:
                        continue
                    endpoints[name][id(case)] = (
                        case,
                        obs[0][1],
                        canonical[1],
                        values,
                        execution,
                        obs[0][2],
                        canonical[0],
                    )
                finally:
                    _witnesses.pop(token, None)

        def leaf(endpoint, expected):
            return ast.dump(ast.parse(endpoint[4].leaf)) == ast.dump(
                ast.parse(expected)
            )

        def path(endpoint, choice, principal):
            selectors = (
                (f"{raw} == 0", f"{row14} == 0") if principal == row20 else (selector,)
            )
            return _ordered_conditional_path(endpoint[4].trace, selectors, choice)

        def amount(endpoint, name):
            return _exact_engine_decimal(
                {**endpoint[2]["input"], **endpoint[3]}.get(name)
            )

        if role == "floor14":
            good = []
            for e in endpoints[row14].values():
                a, b = amount(e, row3), amount(e, row13)
                if a is None or b is None or not leaf(e, f"max(0, {row3} - {row13})"):
                    continue
                delta = _checked_sum((a, -b))
                if delta is not None and amount(e, row14) == max(Decimal(0), delta):
                    good.append((e, delta))
            for (a, av), (b, bv) in combinations(good, 2):
                if (av < 0 < bv or bv < 0 < av) and _pair_changed_fact(
                    a[1], b[1], boolean=False
                ) == raw:
                    found.add((row14, source._case_pair_identity(a[0], b[0])))
        elif role == "second_shortcut":
            good = list(endpoints[row20].values())
            for on in good:
                if not (
                    path(on, 2, row20)
                    and leaf(on, f"max(0, {row14} - {row19})")
                    and amount(on, raw) not in (None, 0)
                    and amount(on, row14) is not None
                    and amount(on, row14) > 0
                    and amount(on, row20) > 0
                ):
                    continue
                for off in good:
                    if (
                        path(off, 1, row20)
                        and leaf(off, "0")
                        and amount(off, raw) not in (None, 0)
                        and amount(off, row14) == 0
                        and amount(off, row20) == 0
                        and _pair_changed_fact(on[1], off[1], boolean=False) == raw
                    ):
                        found.add((row20, source._case_pair_identity(on[0], off[0])))
        elif role == "line1_mixed":
            signed = []
            for e in endpoints[row3].values():
                a, b = amount(e, raw), amount(e, "t2048_line_17215")
                if (
                    a is not None
                    and b is not None
                    and leaf(e, f"{raw} - t2048_line_17215")
                    and _checked_sum((a, -b)) == amount(e, row3)
                ):
                    signed.append(e)
            for pos in signed:
                on = endpoints[row20].get(id(pos[0]))
                if (
                    amount(pos, raw) < 0
                    or on is None
                    or not path(on, 2, row20)
                    or not leaf(on, f"max(0, {row14} - {row19})")
                    or amount(on, row20) <= 0
                ):
                    continue
                for neg in signed:
                    if (
                        amount(neg, raw) >= 0
                        or _pair_changed_fact(pos[1], neg[1], boolean=False) != raw
                    ):
                        continue
                    for off in endpoints[row20].values():
                        if (
                            path(off, 0, row20)
                            and leaf(off, "0")
                            and amount(off, raw) == 0
                            and amount(off, row20) == 0
                            and _pair_changed_fact(on[1], off[1], boolean=False) == raw
                        ):
                            ids = tuple(sorted({id(pos[0]), id(neg[0]), id(off[0])}))
                            if len(ids) == 3:
                                found.add((row3, ids))
        else:
            witnesses = boolean_pair_witnesses(
                context,
                content=content,
                cases=cases,
                principal_rules=principal_rules,
                source_text=source_text,
                citation=citation,
            )
            pairs = {
                w.case_pair_identity
                for w in witnesses
                if w.rule_name == row19
                and w.selector_name == selector
                and w.active_value is False
                and w.zeroes
                and not w.boolean_effect
            }
            for pair in pairs:
                if len(set(pair)) != 2 or any(
                    i not in endpoints[n] for i in pair for n in required
                ):
                    continue
                left, right = (endpoints[row20][i] for i in pair)
                if _pair_changed_fact(left[1], right[1], boolean=True) != selector:
                    continue
                from copy import deepcopy

                changed = deepcopy(left[2])
                changed["input"][selector] = right[2]["input"].get(selector)
                if (
                    _source_key(changed) != _source_key(right[2])
                    or amount(left, row14) != amount(right, row14)
                    or amount(left, row20) == amount(right, row20)
                ):
                    continue
                valid = True
                for i in pair:
                    e, downstream = endpoints[row19][i], endpoints[row20][i]
                    state = e[2]["input"].get(selector)
                    if (
                        type(state) is not bool
                        or amount(downstream, raw) in (None, 0)
                        or amount(downstream, row14) is None
                        or amount(downstream, row14) <= 0
                        or not path(downstream, 2, row20)
                        or not leaf(downstream, f"max(0, {row14} - {row19})")
                        or not path(e, 0 if state else 1, row19)
                    ):
                        valid = False
                        break
                    if not state:
                        valid = (
                            leaf(e, "0")
                            and amount(e, row19) == 0
                            and amount(downstream, row20) == amount(downstream, row14)
                        )
                    else:
                        nodes = [
                            n
                            for result in e[5]["results"]
                            for n in result["trace"].values()
                            if n.get("name") == row19
                        ]
                        reads = [
                            r
                            for n in nodes
                            for r in n.get("parameter_reads", [])
                            if r.get("name") == rate_name
                        ]
                        valid = (
                            len(nodes) == len(reads) == 1
                            and type(reads[0].get("index")) is int
                            and reads[0]["index"] == 0
                            and _wire_value(reads[0]["value"]) == Decimal("0.5")
                            and source._rulespec_runtime_decimal(
                                source._formula_environment_for_case(e[6], e[0]).get(
                                    rate_name
                                )
                            )
                            == Decimal("0.5")
                            and leaf(
                                e, f"federal_2025_t936_chart_a_line_17 * {rate_name}"
                            )
                            and amount(e, row19) > 0
                        )
                    if not valid:
                        break
                if valid:
                    found.add((row19, pair))
        return {min(found)} if found and bound.fresh() else set()
    except (*_DECLINE, SyntaxError):
        return set()


def expense_group_witnesses(
    context: object | None,
    branch: Any,
    *,
    content: str,
    cases: Any,
    principal_rules: dict,
    source_text: str,
    citation: str,
) -> set:
    """One whole expense obligation, requiring every source-owned effect role."""
    import ast
    from copy import deepcopy
    from datetime import date
    from itertools import combinations, product

    from . import source_completeness as source
    from .actual_local_relation_evidence import (
        _checked_sum,
        _exact_engine_decimal,
        _related_ids,
        _wire_value,
    )
    from .local_pair_source_evidence import _EXPENSE_CATEGORIES, bind_expense_group

    included = "t936_asset_included_uncollectible_portion"
    aggregate = "t936_uncollectible_dispositions_total"
    ordinary = "t936_ordinary_vehicle_cost_excluded"
    zepv = "t936_zepv_cost_excluded"
    cost = "t936_asset_acquisition_cost"
    portion = "t936_asset_uncollectible_portion"
    timber = "t936_asset_is_timber_resource"
    oflag = "t936_asset_is_ordinary_passenger_vehicle"
    zflag = "t936_asset_is_zero_emission_passenger_vehicle"
    acquired = "t936_asset_acquisition_date"
    try:
        bound = _contexts.get(context)
        if bound is None or not bound.fresh() or content != bound.content:
            return set()
        _, normalized = _normalized_source_inputs(bound.policy, bound.cases)
        if _source_key(cases) != _source_key(normalized):
            return set()
        principal = bind_expense_group(bound.policy, branch, source_text, citation)
        if principal is None:
            return set()
        required = (principal, aggregate, included, ordinary, zepv)
        if any(n not in principal_rules for n in required):
            return set()
        endpoints = {n: {} for n in required}
        for name in required:
            for case in cases:
                token = object()
                _witnesses[token] = _Witness(
                    context, content, case, name, source_text, citation, principal_rules
                )
                try:
                    canonical = _canonical_formula_context(token, case)
                    values = corroborated_dependencies(token, case)
                    if canonical is None or values is None or name not in values:
                        continue
                    if canonical[1]["period"] != {
                        "period_kind": "tax_year",
                        "start": "2025-01-01",
                        "end": "2025-12-31",
                    }:
                        continue
                    execution = corroborated_execution(
                        token, principal_rules[name], case, canonical[0]
                    )
                    obs = [o for o in bound.observations if _source_key(case) in o[0]]
                    if (
                        execution is None
                        or len(obs) != 1
                        or not source._asserted_formula_runtime_values_equal(
                            principal_rules[name],
                            values[name],
                            source._test_case_asserted_output_value(case, name),
                        )
                    ):
                        continue
                    endpoints[name][id(case)] = (
                        case,
                        obs[0][1],
                        canonical[1],
                        values,
                        execution,
                        obs[0][2],
                        obs[0][3],
                    )
                finally:
                    _witnesses.pop(token, None)

        def leaf(e, text):
            return ast.dump(ast.parse(e[4].leaf)) == ast.dump(ast.parse(text))

        def number(e, name):
            return _exact_engine_decimal({**e[2]["input"], **e[3]}.get(name))

        def arm(e, choice):
            return _ordered_conditional_path(
                e[4].trace,
                (
                    timber,
                    f"{zflag} and {zepv} == true",
                    f"{oflag} and {ordinary} == true",
                ),
                choice,
            )

        roles = {}

        def add(role, *es):
            ids = tuple(sorted({id(e[0]) for e in es}))
            roles[role] = min(roles.get(role, ids), ids)

        for e in endpoints[principal].values():
            amounts = [number(e, n) for n in _EXPENSE_CATEGORIES]
            total = number(e, aggregate)
            if (
                all(v is not None and v > 0 for v in amounts)
                and total is not None
                and total > 0
                and aggregate in e[3]
                and all(n in e[2]["input"] for n in _EXPENSE_CATEGORIES)
                and leaf(e, " + ".join((*_EXPENSE_CATEGORIES, aggregate)))
                and _checked_sum((*amounts, total)) == number(e, principal)
            ):
                add("categories", e)

        # Full provider identities stay in the replayed (name,entity_id) map.
        aggregates = []
        for e in endpoints[aggregate].values():
            request, response = e[1], e[5]
            query = request["queries"][0]
            refs = [r for r in query["outputs"] if r.split("#", 1)[1] == aggregate]
            if len(refs) != 1 or not leaf(
                e, f"sum(t936_depreciable_disposition_for_return.{included})"
            ):
                continue
            ref = refs[0]
            trace = response["results"][0]["trace"]
            providers = trace[ref]["dependencies"]
            relation = (
                ref.split("#")[0] + "#relation.t936_depreciable_disposition_for_return"
            )
            ids = _related_ids(
                request["dataset"]["relations"],
                relation=relation,
                root=query["entity_id"],
                interval={k: query["period"][k] for k in ("start", "end")},
                current_slot=1,
                related_slot=0,
            )
            if (
                ids is None
                or len(ids) != 2
                or len(providers) != 2
                or len(set(providers)) != 2
            ):
                continue
            nodes = [trace[r] for r in providers]
            if tuple(sorted(n["entity_id"] for n in nodes)) != ids or any(
                n["name"] != included or n.get("unit") != "CAD" for n in nodes
            ):
                continue
            amounts = {
                n["entity_id"]: _exact_engine_decimal(
                    e[6].get((included, n["entity_id"]))
                )
                for n in nodes
            }
            if any(v is None for v in amounts.values()) or _checked_sum(
                amounts.values()
            ) != number(e, aggregate):
                continue
            if any(
                not source._formula_runtime_values_equal(
                    _wire_value(n["value"]), amounts[n["entity_id"]]
                )
                for n in nodes
            ):
                continue
            aggregates.append((e, ids, amounts))

        def request_without_outputs(request):
            result = deepcopy(request)
            result["queries"][0].pop("outputs")
            return result

        for a, b in combinations(aggregates, 2):
            ea, ids, av = a
            eb, bids, bv = b
            if (
                ids != bids
                or sorted(av.values()) != sorted(bv.values())
                or not (min(av.values()) == 0 < max(av.values()))
            ):
                continue
            swapped = request_without_outputs(ea[1])
            for f in swapped["dataset"]["inputs"]:
                if f["entity_id"] in ids:
                    f["entity_id"] = ids[1] if f["entity_id"] == ids[0] else ids[0]
            # Input order is transport only; preserve every typed fact and edge.
            target = request_without_outputs(eb[1])
            for q in (swapped, target):
                q["dataset"]["inputs"].sort(key=lambda f: (f["entity_id"], f["name"]))
            if _typed(swapped) != _typed(target) or any(
                av[i] != bv[ids[1] if i == ids[0] else ids[0]] for i in ids
            ):
                continue
            for ec, cids, cv in aggregates:
                if cids != ids or not all(v > 0 for v in cv.values()):
                    continue
                before, after = (
                    request_without_outputs(ea[1]),
                    request_without_outputs(ec[1]),
                )
                ai, bi = before["dataset"].pop("inputs"), after["dataset"].pop("inputs")
                if _typed(before) != _typed(after) or len(ai) != len(bi):
                    continue
                differences = []
                for x, y in zip(ai, bi, strict=True):
                    xv, yv = x.pop("value"), y.pop("value")
                    if _typed(x) != _typed(y):
                        differences.append(None)
                    elif _typed(xv) != _typed(yv):
                        differences.append(
                            (
                                x["name"].split("#input.")[-1],
                                x["entity_id"],
                                _exact_engine_decimal(_wire_value(xv)),
                                _exact_engine_decimal(_wire_value(yv)),
                            )
                        )
                if len(differences) != 1 or differences[0] is None:
                    continue
                name, provider, old, new = differences[0]
                if (
                    name != cost
                    or provider not in ids
                    or old is None
                    or new is None
                    or not old > new
                    or av[provider] != 0
                    or cv[provider] <= 0
                ):
                    continue
                others = [i for i in ids if i != provider]
                if any(av[i] != cv[i] for i in others):
                    continue
                # This aggregation intervention must land exactly on the actual
                # provider's selected cap, not merely any admitted lower cost.
                trace = ea[5]["results"][0]["trace"]
                cap_reads = [
                    read
                    for node in trace.values()
                    if node["entity_id"] == provider
                    and node["name"] in (ordinary, zepv)
                    for read in node.get("parameter_reads", [])
                    if read.get("name", "").startswith("t936_vehicle_cost_")
                ]
                if (
                    len(cap_reads) != 1
                    or _exact_engine_decimal(_wire_value(cap_reads[0]["value"])) != new
                ):
                    continue
                add("aggregate", ea, eb, ec)

        bands = [
            (ordinary, None, "2022-01-01", 30000, 0),
            (ordinary, "2022-01-01", "2023-01-01", 34000, 1),
            (ordinary, "2023-01-01", "2024-01-01", 36000, 2),
            (ordinary, "2024-01-01", "2025-01-01", 37000, 3),
            (ordinary, "2025-01-01", "2026-01-01", 38000, 4),
            (zepv, "2019-03-18", "2022-01-01", 55000, 2),
            (zepv, "2022-01-01", "2023-01-01", 59000, 1),
            (zepv, "2023-01-01", None, 61000, 0),
        ]
        for selector, low, high, cap, choice in bands:
            buckets = {-1: [], 0: [], 1: []}
            for key, e in endpoints[included].items():
                se = endpoints[selector].get(key)
                facts = e[2]["input"]
                when = facts.get(acquired)
                if (
                    se is None
                    or type(when) is not date
                    or facts.get(timber) is not False
                ):
                    continue
                is_z = selector == zepv
                if facts.get(oflag) is not (not is_z) or facts.get(zflag) is not is_z:
                    continue
                if (
                    low
                    and (
                        when <= date.fromisoformat(low)
                        if cap == 55000
                        else when < date.fromisoformat(low)
                    )
                    or high
                    and when >= date.fromisoformat(high)
                ):
                    continue
                value, entered = number(e, cost), number(e, portion)
                if (
                    value is None
                    or entered is None
                    or entered <= 0
                    or not leaf(se, "true" if value > cap else "false")
                ):
                    continue
                trace = se[4].trace
                if (
                    len(trace) != 2
                    or trace[0].choice != choice
                    or trace[1].choice != (0 if value > cap else 1)
                ):
                    continue
                if (
                    se[3].get(selector) is not (value > cap)
                    or _exact_engine_decimal(
                        se[4].constant_environment.get(f"t936_vehicle_cost_{cap}")
                    )
                    != cap
                ):
                    continue
                if not arm(e, (1 if is_z else 2) if value > cap else 3) or not leaf(
                    e, "0" if value > cap else portion
                ):
                    continue
                if number(e, included) != (0 if value > cap else entered):
                    continue
                buckets[(value > cap) - (value < cap)].append(e)
            for below, equal, above in product(buckets[-1], buckets[0], buckets[1]):
                if (
                    _pair_changed_fact(below[1], equal[1]) == cost
                    and _pair_changed_fact(equal[1], above[1]) == cost
                ):
                    add(f"band{cap}", below, equal, above)

        ordinary_points = []
        for e in endpoints[included].values():
            f = e[2]["input"]
            entered = number(e, portion)
            if entered is None or entered <= 0:
                continue
            if (
                f.get(timber) is False
                and f.get(oflag) is False
                and f.get(zflag) is False
                and arm(e, 3)
                and leaf(e, portion)
                and number(e, included) == entered
            ):
                ordinary_points.append(e)
                add("nonvehicle", e)
            if (
                f.get(timber) is False
                and f.get(oflag) is False
                and f.get(zflag) is True
                and f.get(acquired) == date(2019, 3, 18)
                and e[3].get(zepv) is False
                and arm(e, 3)
                and leaf(e, portion)
                and number(e, included) == entered
            ):
                add("march18", e)
        for on in endpoints[included].values():
            if (
                not arm(on, 0)
                or not leaf(on, "0")
                or number(on, included) != 0
                or on[2]["input"].get(timber) is not True
            ):
                continue
            for off in ordinary_points:
                if _pair_changed_fact(off[1], on[1], boolean=True) == timber:
                    add("timber", off, on)
        needed = {
            "categories",
            "aggregate",
            "nonvehicle",
            "march18",
            "timber",
            *(f"band{b[3]}" for b in bands),
        }
        if set(roles) != needed or not bound.fresh():
            return set()
        return {(principal, tuple(sorted({i for ids in roles.values() for i in ids})))}
    except (*_DECLINE, SyntaxError):
        return set()
