"""Pure, closed source/AST bindings; returned sites carry no runtime authority."""

from __future__ import annotations

import ast
import hashlib
from decimal import Decimal
from typing import Any

from axiom_encode.printed_source_rows import recognize_printed_rows

_PROFILES = (
    (
        "50% of exploration and development expenses (line 22400) + 5",
        "Add lines 1 to 7. Total investment expenses claimed in 2025 = A",
        "exploration_and_development_expenses_line_22400",
        "Full line22400 expenses before T936 50% computation.",
        7,
    ),
    (
        "50% of income from the recovery of exploration and development expenses (line 13000) 68110 + 12",
        "Add lines 8 to 13. Total investment income reported in 2025 = B",
        "recovery_of_exploration_and_development_expenses_line_13000",
        "Full recovery income before T936 50% computation.",
        6,
    ),
)


def fraction_site(
    policy: dict,
    rule: dict,
    case: dict,
    branch: Any,
    source_text: str,
    citation: str,
    trace_node: dict,
    facts: dict,
) -> tuple[int, str, str] | None:
    """Bind one reached fifth-row fraction, not another same-valued operation.

    Caller must already own an authenticated replay and a successful complete
    source witness. This pure function cannot authenticate a supplied trace.
    """
    from . import source_completeness as source
    from .actual_local_relation_evidence import _wire_value

    try:
        if recognize_printed_rows(source_text, allow_multipage=True) is None:
            return None
        profiles = [p for p in _PROFILES if branch.text == p[0]]
        if len(profiles) != 1:
            return None
        row, total, amount_name, description, count = profiles[0]
        if (
            source_text.count(row) != 1
            or source_text[branch.start : branch.end] != row
            or source_text.count(total) != 1
        ):
            return None
        if (
            any(
                rule.get(k) != v
                for k, v in {
                    "kind": "derived",
                    "dtype": "Money",
                    "unit": "CAD",
                    "entity": "Person",
                    "period": "Year",
                }.items()
            )
            or rule.get("rounding") is not None
        ):
            return None
        version = source._selected_rule_formula_version_index(rule, case)
        if version is None:
            return None
        selected = rule["versions"][version]
        if (selected.get("effective_from"), selected.get("effective_to")) != (
            "2025-01-01",
            "2025-12-31",
        ):
            return None
        header = source_text[: source_text.index("\n")]

        def proves(declaration: dict, index: int, field: str, excerpt: str) -> bool:
            return any(
                path == f"versions[{index}].{field}"
                and cite == citation
                and text.strip() == excerpt
                for path, cite, text in source._rule_source_excerpt_atoms(declaration)
            )

        def window_proved(declaration: dict, index: int) -> bool:
            return all(
                proves(declaration, index, field, header)
                for field in ("effective_from", "effective_to")
            )

        # These are exact row-group boundaries in the fully recognized source,
        # not a general enclosing-excerpt allowance. The rate still proves the
        # individual fractional row; the group grants no other operation sites.
        group_start, group_end = (1193, 1647) if count == 7 else (2014, 2562)
        fraction_owned = proves(rule, version, "formula", row) or (
            group_start <= branch.start < branch.end <= group_end
            and proves(
                rule, version, "formula", source_text[group_start:group_end].strip()
            )
        )
        if not (
            window_proved(rule, version)
            and fraction_owned
            and proves(rule, version, "formula", total)
        ):
            return None
        expression = selected.get("formula")
        if not isinstance(expression, str):
            return None
        tree = ast.parse(expression, mode="eval").body
        terms = []

        def flatten(node: ast.AST, path: tuple[str, ...]) -> None:
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                flatten(node.left, (*path, "left"))
                flatten(node.right, (*path, "right"))
            else:
                terms.append((node, path))

        flatten(tree, ())
        if len(terms) != count:
            return None
        node, path = terms[4]
        if not (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mult)
            and isinstance(node.left, ast.Name)
            and node.left.id == amount_name
            and isinstance(node.right, ast.Name)
        ):
            return None
        if any(
            not isinstance(term, ast.Name)
            for i, (term, _) in enumerate(terms)
            if i != 4
        ):
            return None
        names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
        if len(names) != len(set(names)):
            return None
        inputs = policy.get("inputs", [])
        amounts = [d for d in inputs if d.get("name") == amount_name]
        if len(amounts) != 1:
            return None
        amount = amounts[0]
        # Complete substantive identity AND full-before-fraction contract are
        # required. A line-number suffix or an unrelated Money input is not.
        if (
            amount.get("description") != description
            or any(
                amount.get(k) != v
                for k, v in {
                    "dtype": "Money",
                    "unit": "CAD",
                    "entity": "Person",
                    "period": "Year",
                }.items()
            )
            or "default" in amount
        ):
            return None
        if sum(d.get("description") == description for d in inputs) != 1:
            return None
        rates = [d for d in policy.get("rules", []) if d.get("name") == node.right.id]
        if (
            len(rates) != 1
            or rates[0].get("kind") != "parameter"
            or rates[0].get("dtype") != "Rate"
        ):
            return None
        rate = rates[0]
        index = source._selected_rule_formula_version_index(rate, case)
        if index is None:
            return None
        cell = rate["versions"][index]
        if (cell.get("effective_from"), cell.get("effective_to")) != (
            "2025-01-01",
            "2025-12-31",
        ):
            return None
        if not proves(rate, index, "formula", row) or not window_proved(rate, index):
            return None
        rate_value = source._rulespec_runtime_decimal(
            source._formula_environment_for_case(
                source._constant_rule_environment(policy), case
            ).get(node.right.id)
        )
        if rate_value != Decimal("0.5"):
            return None
        reads = [
            r
            for r in trace_node.get("parameter_reads", [])
            if r.get("name") == node.right.id
        ]
        if len(reads) != 1:
            return None
        read = reads[0]
        if (
            read.get("index") != index
            or type(read.get("index")) is not int
            or _wire_value(read["value"]) != rate_value
            or (read.get("effective_from"), read.get("effective_to"))
            != (cell["effective_from"], cell["effective_to"])
        ):
            return None
        value = facts.get(amount_name)
        if type(value) not in (int, Decimal) or (
            type(value) is Decimal and not value.is_finite()
        ):
            return None
        # The registered factory independently replayed every intermediate in
        # this exact authored expression and its compiled execution.
        digest = hashlib.sha256(
            ast.dump(node, include_attributes=False).encode()
        ).hexdigest()
        return version, "/".join(path), digest
    except (KeyError, TypeError, ValueError, SyntaxError, AttributeError, IndexError):
        return None
