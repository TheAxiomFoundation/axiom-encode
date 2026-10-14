"""Bind existing equality witnesses to authenticated worksheet executions."""

import ast
import math
from decimal import Decimal

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import worksheet_operation_evidence as op

SPAN = (624, 834)
YEAR = ("2025-01-01", "2025-12-31")


def existing_witnesses(bound, branch, witnesses, *, principal_rules, asserted_by_rule):
    """Return exact existing objects only; no new witness or case resource."""
    accepted = set()
    try:
        if not bound.evidence.matches(
            bound.payload, bound.cases, bound.source, bound.context, bound.bundle
        ):
            return accepted
        if (branch.start, branch.end) != SPAN or branch.text != bound.source[
            slice(*SPAN)
        ]:
            return accepted
        quotes = sc._annual_source_quoted_spans(bound.source)
        if quotes is None or any(
            a < branch.end and b > branch.start for a, b in quotes
        ):
            return accepted
        # Full paragraph and alias proof/computation already required by universal core.
        rule = principal_rules.get(op._LINE5)
        original_rules = op.paths._index(bound.payload.get("rules")) or {}
        if rule is None or rule != original_rules.get(op._LINE5):
            return accepted
        if any(
            name not in original_rules
            or supplied != original_rules[name]
            or getattr(supplied, "resolved_currency_rounding", None)
            != getattr(original_rules[name], "resolved_currency_rounding", None)
            for name, supplied in principal_rules.items()
        ):
            return accepted
        # The universal operation core does not admit unrelated partial claimants.
        for name, claimant in original_rules.items():
            for path, citation, excerpt in sc._rule_source_excerpt_atoms(claimant):
                version = sc._formula_proof_version_index(path)
                if version is None or not excerpt:
                    continue
                if bound.source.count(excerpt) == 1:
                    start = bound.source.index(excerpt)
                    overlaps = start < branch.end and branch.start < start + len(
                        excerpt
                    )
                else:
                    a = sc._collapse_text(excerpt)
                    b = sc._collapse_text(branch.text)
                    overlaps = bool(a) and (a in b or b in a)
                if not overlaps:
                    continue
                if (
                    name != op._LINE5
                    or citation != op._PRIMARY
                    or not any(
                        i == version and (lo, hi) == YEAR
                        for i, _, lo, hi in sc._effective_formula_version_intervals(
                            claimant
                        )
                    )
                ):
                    return accepted
        alias = original_rules.get(op._LINE1)
        if alias is None or (
            rule.get("dtype"),
            rule.get("unit"),
            rule.get("entity"),
            rule.get("period"),
        ) != ("Money", "CAD", "Person", "Year"):
            return accepted
        env = sc._constant_rule_environment(bound.payload)
        master = {id(case): (i, case) for i, case in enumerate(bound.cases)}
        asserted = {id(case): case for case in asserted_by_rule.get(op._LINE5, ())}
        flat_index = {
            master_i: flat_i
            for flat_i, master_i in enumerate(bound.bundle.flat_case_indices)
        }
        for witness in witnesses:
            if (
                witness.rule_name != op._LINE5
                or witness.selector_name not in {op._PAID, op._CREDIT}
                or witness.active_value is not True
                or not witness.zeroes
                or not witness.blocks
                or witness.boolean_effect
                or witness.calendar_attainment_age is not None
                or witness.maximum_transitions
                or witness.numeric_transition is None
                or witness.relational_transitions != ((op._LINE1, "==", op._CREDIT),)
                or len(witness.case_pair_identity) != 2
                or len(set(witness.case_pair_identity)) != 2
            ):
                continue
            transition_metadata = witness.numeric_transition
            if (
                not isinstance(transition_metadata, tuple)
                or len(transition_metadata) != 2
                or any(
                    isinstance(x, bool)
                    or not isinstance(x, (int, float, Decimal))
                    or not math.isfinite(x)
                    for x in transition_metadata
                )
            ):
                continue
            ids = witness.case_pair_identity
            if any(i not in master or i not in asserted for i in ids):
                continue
            pairs = [master[i] for i in ids]
            cases = [x[1] for x in pairs]
            if any(asserted[i] is not master[i][1] for i in ids):
                continue
            if (
                any("period" not in c for c in cases)
                or cases[0]["period"] != cases[1]["period"]
                or not sc._cases_differ_by_one_input(*cases)
                or not sc._cases_have_same_output_keys(*cases)
            ):
                continue
            inputs = [sc._case_input_formula_environment(c) for c in cases]
            if any(x is None for x in inputs):
                continue
            changed = {
                key
                for key in inputs[0].keys() | inputs[1].keys()
                if inputs[0].get(key) != inputs[1].get(key)
            }
            if changed != {witness.selector_name}:
                continue
            transition = [x.get(witness.selector_name) for x in inputs]
            if any(
                isinstance(x, bool)
                or not isinstance(x, (int, float, Decimal))
                or not math.isfinite(x)
                for x in transition
            ):
                continue
            if tuple(sorted(transition)) != tuple(sorted(witness.numeric_transition)):
                continue
            if any(
                (op._LINE5, flat_index.get(i))
                not in bound.bundle.scalar_operations.case_links
                for i, _ in pairs
            ):
                continue
            equality_values = []
            outputs = []
            valid = True
            for case in cases:
                selected = sc._selected_rule_formula_version_index(rule, case)
                alias_selected = sc._selected_rule_formula_version_index(alias, case)
                if not any(
                    i == selected and (lo, hi) == YEAR
                    for i, _, lo, hi in sc._effective_formula_version_intervals(rule)
                ):
                    valid = False
                    break
                if not any(
                    i == alias_selected and (lo, hi) == YEAR
                    for i, _, lo, hi in sc._effective_formula_version_intervals(alias)
                ):
                    valid = False
                    break
                node = sc._parse_formula_expression(
                    alias["versions"][alias_selected]["formula"]
                )
                if not isinstance(node, ast.Name) or node.id != op._PAID:
                    valid = False
                    break
                if not any(
                    path == f"versions[{selected}].formula"
                    and citation == op._PRIMARY
                    and branch.text in excerpt
                    for path, citation, excerpt in sc._rule_source_excerpt_atoms(rule)
                ):
                    valid = False
                    break
                # Independently compute the selected closure: assertions cannot supply dependencies.
                dependencies = sc._case_dependency_environment(
                    original_rules,
                    case,
                    formula_environment=env,
                    require_asserted_value=False,
                )
                runtime_env = sc._formula_case_runtime_environment(
                    case, dependency_environment=dependencies, formula_environment=env
                )
                execution = sc._case_formula_execution(
                    rule,
                    case,
                    formula_environment=env,
                    dependency_environment=dependencies,
                )
                if runtime_env is None or execution is None:
                    valid = False
                    break
                values = [runtime_env.get(name) for name in (op._LINE1, op._CREDIT)]
                if any(
                    isinstance(x, bool)
                    or not isinstance(x, (int, float, Decimal))
                    or not math.isfinite(x)
                    for x in values
                ):
                    valid = False
                    break
                if runtime_env.get(op._PAID) != values[0]:
                    valid = False
                    break
                actual = sc._formula_execution_runtime_value(execution)
                expected = sc._test_case_asserted_output_value(case, op._LINE5)
                if not sc._asserted_formula_runtime_values_equal(
                    rule, actual, expected
                ):
                    valid = False
                    break
                reached = []
                for step in execution.trace:
                    if step.kind != "if" or len(step.selectors) != 1:
                        continue
                    expression = sc._parse_formula_expression(step.selectors[0])
                    if expression is None:
                        continue
                    relations = tuple(
                        (a.id, cmp, b.id)
                        for a, cmp, b in sc._formula_relational_expressions(expression)
                        if cmp == "=="
                        and isinstance(a, ast.Name)
                        and isinstance(b, ast.Name)
                    )
                    if relations == witness.relational_transitions:
                        reached.append(step.choice)
                equal = values[0] == values[1]
                if (
                    reached != [0 if equal else 1]
                    or (equal and actual != 0)
                    or (not equal and actual == 0)
                ):
                    valid = False
                    break
                equality_values.append(equal)
                outputs.append(actual)
            if valid and set(equality_values) == {False, True}:
                accepted.add(witness)
        return accepted
    except (TypeError, ValueError, KeyError, IndexError, AttributeError):
        return set()
