"""Bind authenticated note (3) replacement operations to existing witnesses.

Only successful universal worksheet evidence may supply the replacement's
subtraction kind. Numeric, topology, dependency, temporal and witness-allocation
checks remain in the ordinary formula consumer.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import source_path_evidence as paths
from axiom_encode.harness import worksheet_operation_evidence as op
from axiom_encode.harness import worksheet_transfer_evidence as tr

OWNER = "net_income_for_multiple_jurisdictions"
SPAN = (4334, 4582)
YEAR = ("2025-01-01", "2025-12-31")
PRIMARY_SHA = "7637ab494fc385b938e9e3ce4d5f3b1af8bc331ad644fe3cedefaee8434028ee"
NET = "t2036_net_income_for_credit"


def identity(value):
    return paths.path_input_identity(value, ())


def rule_context_identity(payload):
    """Freeze private rule context omitted by plain mapping serialization."""
    rules = op.paths._index(payload.get("rules")) or {}
    return identity(
        {
            name: getattr(rule, "resolved_currency_rounding", None)
            for name, rule in rules.items()
        }
    )


def leaf_key(text):
    if not isinstance(text, str):
        return None
    parsed = sc._parse_formula_expression(text)
    return ast.dump(parsed, include_attributes=False) if parsed is not None else None


@dataclasses.dataclass(frozen=True)
class ExecutionRecord:
    master_index: int
    case_identity: str
    version: int
    interval: tuple[str, str]
    leaf: str
    owner_identity: str
    execution_identity: str


@dataclasses.dataclass(frozen=True)
class WorksheetReplacementCoverage:
    master_identity: str = ""
    context_identity: str = ""
    source_identity: str = ""
    flat_indices: tuple[int, ...] = ()
    scalar_identity: str = ""
    records: tuple[ExecutionRecord, ...] = ()
    unresolved: tuple[str, ...] = ()
    rule_context: str = ""

    def matches(self, payload, cases, source, context, bundle):
        try:
            return (
                bool(self.records)
                and not self.unresolved
                and self.rule_context == rule_context_identity(payload)
                and (
                    bundle.matches(payload, cases, source, context)
                    and self.master_identity
                    == paths.path_input_identity(payload, cases)
                    and self.context_identity == identity(context)
                    and self.source_identity
                    == hashlib.sha256(source.encode()).hexdigest()
                    and self.flat_indices == bundle.flat_case_indices
                    and isinstance(
                        bundle.scalar_operations, op.WorksheetOperationCoverage
                    )
                    and bundle.scalar_operations.matches(
                        payload, [cases[i] for i in self.flat_indices], source, context
                    )
                    and self.scalar_identity
                    == paths.path_input_identity(
                        payload, [cases[i] for i in self.flat_indices]
                    )
                )
            )
        except (TypeError, ValueError, KeyError, IndexError, AttributeError):
            return False

    def operation_kinds(
        self, *, payload, cases, source, context, bundle, rule, case, branch, execution
    ):
        try:
            if not self.matches(payload, cases, source, context, bundle):
                return None
            if (branch.start, branch.end) != SPAN or branch.text != source[
                slice(*SPAN)
            ]:
                return None
            if rule.get("name") != OWNER or not isinstance(
                execution, sc._FormulaExecution
            ):
                return None
            selected = sc._selected_rule_formula_version_index(rule, case)
            for r in self.records:
                if (
                    case is cases[r.master_index]
                    and r.case_identity == identity(case)
                    and r.version == selected
                    and r.owner_identity == identity(rule)
                    and r.leaf == leaf_key(execution.leaf)
                    and r.execution_identity == identity(dataclasses.asdict(execution))
                ):
                    return frozenset({"subtract"})
            return None
        except (TypeError, ValueError, KeyError, IndexError, AttributeError):
            return None


def certify_worksheet_replacement(payload, cases, source, context, bundle):
    """Scope existing successful universal evidence; never reconstruct success."""

    def fail(reason):
        return WorksheetReplacementCoverage(unresolved=(reason,))

    try:
        if not bundle.matches(payload, cases, source, context):
            return fail("full transfer identity invalid")
        core = bundle.scalar_operations
        if core is None or core.unresolved or set(core.outputs) != set(op._OUTPUTS):
            return fail("universal operation certificate absent")
        if hashlib.sha256(source.encode()).hexdigest() != PRIMARY_SHA:
            return fail("unsupported exact replacement source")
        if context.get(op._PRIMARY, "").strip() != source:
            return fail("primary context mismatch")
        indices = bundle.flat_case_indices
        expected_indices = tuple(
            i
            for i, case in enumerate(cases)
            if not any(
                sc._test_case_asserted_output_value(case, n)
                is not sc._UNRESOLVED_CONDITION_VALUE
                for n in tr.OUTPUTS
            )
        )
        if indices != expected_indices or len(set(indices)) != len(indices):
            return fail("flat projection differs")
        flat = [cases[i] for i in indices]
        if not isinstance(core, op.WorksheetOperationCoverage) or not core.matches(
            payload, flat, source, context
        ):
            return fail("stale universal core identity")
        if bundle.scalar_paths.input_identity != paths.path_input_identity(
            payload, flat
        ):
            return fail("scalar projection identity differs")
        rules = op.paths._index(payload.get("rules"))
        if not rules or OWNER not in rules:
            return fail("owner absent")
        rule = rules[OWNER]
        if (
            rule.get("dtype"),
            rule.get("unit"),
            rule.get("entity"),
            rule.get("period"),
        ) != ("Money", "CAD", "Person", "Year"):
            return fail("owner type differs")
        selected_intervals = sc._effective_formula_version_intervals(rule)
        constants = sc._constant_rule_environment(payload)
        records = []
        for flat_i, master_i in enumerate(indices):
            if (op._LINE4, flat_i) not in core.case_links:
                continue
            case = cases[master_i]
            env = sc._case_input_formula_environment(case) or {}
            count = env.get("jurisdictions_tax_paid_count")
            if type(count) is not int or count <= 1:
                continue
            version = sc._selected_rule_formula_version_index(rule, case)
            if not any(
                i == version and (lo, hi) == YEAR for i, _, lo, hi in selected_intervals
            ):
                continue
            dep = sc._case_asserted_dependency_environment(
                rules, case, formula_environment=constants
            )
            execution = sc._case_formula_execution(
                rule, case, formula_environment=constants, dependency_environment=dep
            )
            asserted = sc._test_case_asserted_output_value(case, OWNER)
            if (
                execution is None
                or asserted is sc._UNRESOLVED_CONDITION_VALUE
                or not sc._asserted_formula_runtime_values_equal(
                    rule, sc._formula_execution_runtime_value(execution), asserted
                )
            ):
                continue
            if not any(
                path == f"versions[{version}].formula"
                and citation == op._PRIMARY
                and source[slice(*SPAN)] in excerpt
                for path, citation, excerpt in sc._rule_source_excerpt_atoms(rule)
            ):
                continue
            if leaf_key(execution.leaf) is None:
                continue
            records.append(
                ExecutionRecord(
                    master_i,
                    identity(case),
                    version,
                    YEAR,
                    leaf_key(execution.leaf),
                    identity(rule),
                    identity(dataclasses.asdict(execution)),
                )
            )
        if not records:
            return fail("no applicable reached asserted owner")
        return WorksheetReplacementCoverage(
            paths.path_input_identity(payload, cases),
            identity(context),
            hashlib.sha256(source.encode()).hexdigest(),
            indices,
            paths.path_input_identity(payload, flat),
            tuple(records),
            rule_context=rule_context_identity(payload),
        )
    except (TypeError, ValueError, KeyError, IndexError, AttributeError) as error:
        return fail(
            "malformed evidence inputs: " + type(error).__name__ + ": " + str(error)
        )


@dataclasses.dataclass(frozen=True)
class BoundWorksheetReplacement:
    """In-process inputs are rechecked at every execution; never a scope flag."""

    evidence: WorksheetReplacementCoverage
    payload: object
    cases: object
    source: str
    context: object
    bundle: object

    def operation_kinds(self, *, rule, case, branch, execution):
        return self.evidence.operation_kinds(
            payload=self.payload,
            cases=self.cases,
            source=self.source,
            context=self.context,
            bundle=self.bundle,
            rule=rule,
            case=case,
            branch=branch,
            execution=execution,
        )


def bind_worksheet_replacement(payload, cases, source, context, bundle):
    evidence = certify_worksheet_replacement(payload, cases, source, context, bundle)
    return BoundWorksheetReplacement(evidence, payload, cases, source, context, bundle)
