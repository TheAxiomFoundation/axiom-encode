"""Closed printed-row bindings; descriptors alone carry no execution authority."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from axiom_encode.printed_source_rows import recognize_printed_rows

_CITATION = "ca/policy/cra/t1-2025/cumulative-net-investment-loss"
_PROFILES = {
    "chart_completed_expense": (
        1647,
        1834,
        "federal_2025_t936_additional_expenses_line_7",
        "boolean",
        ("t936_chart_a_completed",),
        "Additional investment expenses: if you did not complete Chart A "
        'on page 4 of this form, enter "0".\n'
        "Otherwise, enter whichever is less: line 20 in Chart A or line "
        "25300 of your return. + 7",
    ),
    "chart_completed_income": (
        2562,
        2716,
        "federal_2025_t936_additional_income_line_13",
        "boolean",
        ("t936_chart_a_completed",),
        "Additional investment income: if you did not complete Chart A on "
        'page 4 of this form, enter "0".\n'
        "Otherwise, enter the amount from line 20 in Chart A. + 13",
    ),
    "cnil_floor": (
        6332,
        6408,
        "cumulative_net_investment_loss_t936_part_3_line_c",
        "subtract",
        (
            "federal_2025_t936_cumulative_investment_expenses_line_18",
            "federal_2025_t936_cumulative_investment_income_line_21",
        ),
        'Line 18 minus line 21 (if negative, enter "0") CNIL to December 31, 2025 = C',
    ),
    "chart11_floor": (
        7859,
        7935,
        "federal_2025_t936_chart_a_line_11",
        "multiply",
        (
            "federal_2025_t936_chart_a_line_9",
            "federal_2025_t936_chart_a_inclusion_rate",
        ),
        'Line 9 multiplied by the percentage on line 10\n(if negative, enter "0") = 11',
    ),
    "chart20_floor": (
        8547,
        8627,
        "federal_2025_t936_chart_a_additional_investment_income_line_20",
        "subtract",
        (
            "t936_chart_a_line_14_non_eligible_taxable_capital_gains",
            "federal_2025_t936_chart_a_line_19",
        ),
        'Line 14 minus line 19 (if negative, enter "0") Additional investment '
        "income = 20",
    ),
}
_RULES = {
    "federal_2025_t936_chart_a_inclusion_rate": {
        "formula": "0.50",
        "dtype": "Rate",
        "proofs": (
            "Inclusion rate × 50% 10",
            "Inclusion rate × 50% 18",
            "50% of exploration and development expenses (line 22400) + 5",
            "50% of income from the recovery of "
            "exploration and development "
            "expenses (line 13000) 68110 + 12",
        ),
    },
    "federal_2025_t936_chart_a_line_9": {
        "formula": "federal_2025_t936_chart_a_line_7 + t936_chart_a_line_8",
        "dtype": "Money",
        "proofs": ("Line 7 plus line 8 (if negative, show in brackets) = 9",),
    },
    "federal_2025_t936_chart_a_line_11": {
        "formula": "max(0, federal_2025_t936_chart_a_line_9 * "
        "federal_2025_t936_chart_a_inclusion_rate)",
        "dtype": "Money",
        "proofs": (
            "Line 9 multiplied by the percentage on "
            "line 10\n"
            '(if negative, enter "0") = 11',
        ),
    },
    "t936_chart_a_line_14_non_eligible_taxable_capital_gains": {
        "formula": "max(0, "
        "federal_2025_t936_chart_a_line_3 "
        "- "
        "federal_2025_t936_chart_a_line_13)",
        "dtype": "Money",
        "proofs": ('Line 3 minus line 13 (if negative, enter "0") = 14',),
    },
    "federal_2025_t936_chart_a_line_19": {
        "formula": "if t936_chart_a_line_14_includes_t3:\n"
        "  federal_2025_t936_chart_a_line_17 * "
        "federal_2025_t936_chart_a_inclusion_rate\n"
        "else:\n"
        "  0",
        "dtype": "Money",
        "proofs": (
            "If the total from line 14 includes an "
            "amount from a T3 slip, complete lines 15 "
            "to 19 below.\n"
            'Otherwise, enter "0" on line 19 and '
            "continue at line 20.",
            "Inclusion rate × 50% 18\n"
            "Line 17 multiplied by the percentage on "
            "line 18 = (cid:379) – 19",
        ),
    },
    "federal_2025_t936_chart_a_additional_investment_income_line_20": {
        "formula": "if "
        "schedule_3_line_19900 "
        "== 0:\n"
        "  0\n"
        "elif "
        "t936_chart_a_line_14_non_eligible_taxable_capital_gains "
        "== 0:\n"
        "  0\n"
        "else:\n"
        "  max(0, "
        "t936_chart_a_line_14_non_eligible_taxable_capital_gains "
        "- "
        "federal_2025_t936_chart_a_line_19)",
        "dtype": "Money",
        "proofs": (
            "If the amount "
            'is "0", do '
            "not complete "
            "lines 2 to "
            "17, and enter "
            '"0" on line '
            "20. 1",
            "If the total "
            "from line 14 "
            'is "0", do '
            "not complete "
            "lines 15 to "
            "19, and enter "
            '"0" on line '
            "20.",
            "Line 14 minus "
            "line 19 (if "
            "negative, "
            'enter "0") '
            "Additional "
            "investment "
            "income = 20",
        ),
    },
    "federal_2025_t936_additional_expenses_line_7": {
        "formula": "if t936_chart_a_completed:\n"
        "  "
        "min(federal_2025_t936_chart_a_additional_investment_income_line_20, "
        "t1_line_25300_net_capital_losses_of_other_years)\n"
        "else:\n"
        "  0",
        "dtype": "Money",
        "proofs": (
            "Additional investment expenses: "
            "if you did not complete Chart A "
            "on page 4 of this form, enter "
            '"0".\n'
            "Otherwise, enter whichever is "
            "less: line 20 in Chart A or "
            "line 25300 of your return. + "
            "7",
        ),
    },
    "federal_2025_t936_additional_income_line_13": {
        "formula": "if t936_chart_a_completed:\n"
        "  "
        "federal_2025_t936_chart_a_additional_investment_income_line_20\n"
        "else:\n"
        "  0",
        "dtype": "Money",
        "proofs": (
            "Additional investment income: if "
            "you did not complete Chart A on "
            'page 4 of this form, enter "0".\n'
            "Otherwise, enter the amount from "
            "line 20 in Chart A. + 13",
        ),
    },
    "federal_2025_t936_cumulative_investment_expenses_line_18": {
        "formula": "federal_2025_t936_total_expenses_after_1987_line_16 "
        "- "
        "t2048_2024_to_2025_line_17218",
        "dtype": "Money",
        "proofs": (
            "Amount from line "
            "17218 of Form T2048 "
            "for 2024 to 2025 – "
            "17\n"
            "Cumulative "
            "investment expenses "
            "(line 16 minus line "
            "17) = (cid:379) "
            "18",
        ),
    },
    "federal_2025_t936_cumulative_investment_income_line_21": {
        "formula": "federal_2025_t936_total_investment_income_line_b + t936_previous_b",
        "dtype": "Money",
        "proofs": (
            "Total investment "
            "income reported in "
            "2025 (line B of Part "
            "2) 19\n"
            "Total investment "
            "income reported in "
            "previous years after "
            "1987:\n"
            "Amount from line 19 "
            "of Part 3 of Form "
            "T936 for 2024. (if "
            "you did not\n"
            "complete Form T936 "
            "for 2024, see note "
            "(2)) + 20\n"
            "Cumulative investment "
            "income (total of "
            "lines 19 and 20) = "
            "(cid:379) – 21",
        ),
    },
    "cumulative_net_investment_loss_t936_part_3_line_c": {
        "formula": "max(0, "
        "federal_2025_t936_cumulative_investment_expenses_line_18 "
        "- "
        "federal_2025_t936_cumulative_investment_income_line_21)",
        "dtype": "Money",
        "proofs": (
            "Line 18 minus line 21 (if "
            'negative, enter "0") CNIL '
            "to December 31, 2025 = "
            "C",
        ),
    },
}


@dataclass(frozen=True)
class PairRole:
    role: str
    principal: str
    operation: str
    operands: tuple[str, ...]
    span: tuple[int, int]


def _tree(expression: str) -> str:
    return ast.dump(ast.parse(expression), include_attributes=False)


def bind_pair_role(
    policy: dict, branch: Any, source_text: str, citation: str, principal: str
) -> PairRole | None:
    """Recognize five complete source instructions, never waive child obligations.

    Exact source-owned immediate operands and ordered formulas are required.
    The private caller must separately prove the actual reached operation and
    two complete captured executions. This helper grants no coverage itself.
    """
    try:
        if (
            citation != _CITATION
            or recognize_printed_rows(source_text, allow_multipage=True) is None
        ):
            return None
        matches = [
            (role, p)
            for role, p in _PROFILES.items()
            if p[2] == principal
            and (branch.start, branch.end) == p[:2]
            and branch.text == p[5]
            and source_text[p[0] : p[1]] == p[5]
        ]
        if len(matches) != 1:
            return None
        role, profile = matches[0]
        rules, inputs = policy["rules"], policy["inputs"]
        all_declarations = [*rules, *inputs]
        by_name = {r["name"]: r for r in all_declarations}
        if len(by_name) != len(all_declarations):
            return None
        required = {principal, *profile[4]}
        if profile[3] == "boolean":
            required.add(
                "federal_2025_t936_chart_a_additional_investment_income_line_20"
            )
        # Close only the explicitly recognized source roles; underlying tax
        # computations outside this profile retain their independent obligations.
        pending = list(required)
        while pending:
            name = pending.pop()
            if name not in _RULES:
                continue
            for node in ast.walk(ast.parse(_RULES[name]["formula"])):
                if (
                    isinstance(node, ast.Name)
                    and node.id in _RULES
                    and node.id not in required
                ):
                    required.add(node.id)
                    pending.append(node.id)
        # Verify each recognized owner by full printed proof, not name suffix.
        for name in tuple(required):
            if name not in _RULES:
                continue
            r, spec = by_name[name], _RULES[name]
            if r not in rules or any(
                k in r
                for k in (
                    "formula",
                    "default",
                    "rounding",
                    "resolved_currency_rounding",
                    "indexed_by",
                    "values",
                )
            ):
                return None
            dtype = spec["dtype"]
            if r.get("dtype") != dtype or r.get("kind") != (
                "parameter" if dtype == "Rate" else "derived"
            ):
                return None
            if dtype != "Rate" and any(
                r.get(k) != v
                for k, v in {
                    "entity": "Person",
                    "period": "Year",
                    "unit": "CAD",
                }.items()
            ):
                return None
            if dtype == "Rate" and any(k in r for k in ("entity", "period", "unit")):
                return None
            versions = r.get("versions")
            if not isinstance(versions, list) or len(versions) != 1:
                return None
            v = versions[0]
            if (v.get("effective_from"), v.get("effective_to")) != (
                "2025-01-01",
                "2025-12-31",
            ) or set(v) != {"effective_from", "effective_to", "formula"}:
                return None
            if _tree(v["formula"]) != _tree(spec["formula"]):
                return None
            atoms = r["metadata"]["proof"]["atoms"]
            obligations = [
                ("formula", "parameter" if dtype == "Rate" else "formula", x)
                for x in spec["proofs"]
            ]
            obligations += [
                (f, "effective_period", source_text.splitlines()[0])
                for f in ("effective_from", "effective_to")
            ]
            for field, kind, excerpt in obligations:
                if source_text.count(excerpt) != 1 or not any(
                    a.get("path") == "versions[0]." + field
                    and a.get("kind") == kind
                    and a.get("source", {}).get("corpus_citation_path") == citation
                    and a.get("source", {}).get("excerpt") in (excerpt, excerpt + "\n")
                    for a in atoms
                ):
                    return None
            # All expression references must be explicit typed declarations.
            # Their underlying source obligations remain independently checked.
            for n in ast.walk(ast.parse(v["formula"])):
                if not isinstance(n, ast.Name) or n.id in ("max", "min"):
                    continue
                d = by_name[n.id]
                if any(
                    k in d
                    for k in (
                        "default",
                        "rounding",
                        "resolved_currency_rounding",
                        "indexed_by",
                    )
                ):
                    return None
                expected_type = (
                    "Rate"
                    if n.id == "federal_2025_t936_chart_a_inclusion_rate"
                    else "Boolean"
                    if n.id
                    in ("t936_chart_a_completed", "t936_chart_a_line_14_includes_t3")
                    else "Money"
                )
                if d.get("dtype") != expected_type:
                    return None
                if expected_type == "Rate" and (
                    d not in rules or any(k in d for k in ("entity", "period", "unit"))
                ):
                    return None
                if d.get("dtype") != "Rate" and (
                    d.get("entity"),
                    d.get("period"),
                    d.get("unit"),
                ) != ("Person", "Year", "CAD" if d["dtype"] == "Money" else None):
                    return None
                if d in inputs and any(
                    k in d for k in ("formula", "versions", "values")
                ):
                    return None
        if profile[3] == "boolean":
            selector = by_name["t936_chart_a_completed"]
            if (
                selector not in inputs
                or selector.get("dtype") != "Boolean"
                or selector.get("description")
                != "Chart A was completed, separate from required-to-complete duty."
            ):
                return None
            if role == "chart_completed_expense":
                amount = by_name["t1_line_25300_net_capital_losses_of_other_years"]
                if (
                    amount not in inputs
                    or amount.get("description")
                    != "Explicit completed 2025 source-line amount: t1_line_25300_net_capital_losses_of_other_years. No upstream recalculation or eligibility inference."
                ):
                    return None
        return PairRole(role, principal, profile[3], profile[4], profile[:2])
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, SyntaxError):
        return None


_SIGNED_PROFILES = {
    "federal_2025_t936_chart_a_line_7": (
        (7533, 7586),
        "schedule_3_line_10700 + schedule_3_line_11000 + schedule_3_line_12400",
        (
            "Amount from line 10700 of Schedule 3 4\n"
            "Amount from line 11000 of Schedule 3 + 5\n"
            "Amount from line 12400 of Schedule 3 + 6\n"
            "Add lines 4 to 6 (if negative, show in brackets). = "
            "7",
        ),
    ),
    "t936_chart_a_line_8": (
        (7587, 7779),
        "if schedule_3_line_19200_reported:\n"
        "  t2017_line_13_reserves_qffp_qsbcs\n"
        "else:\n"
        "  0",
        (
            "If you reported an amount on line 19200 of Schedule 3, enter the "
            "amount\n"
            "from line 13 of Form T2017, Summary of Reserves on Dispositions "
            "of\n"
            "Capital Property (if negative, show in brackets). + 8",
        ),
    ),
    "federal_2025_t936_chart_a_line_9": (
        (7780, 7834),
        "federal_2025_t936_chart_a_line_7 + t936_chart_a_line_8",
        ("Line 7 plus line 8 (if negative, show in brackets) = 9",),
    ),
}


@dataclass(frozen=True)
class SignedRole:
    principal: str
    operands: tuple[str, ...]
    selector: str | None
    span: tuple[int, int]


def bind_signed_role(
    policy: dict, branch: Any, source_text: str, citation: str, principal: str
) -> SignedRole | None:
    """Bind signed numeric representation; no claim about rendered brackets.

    Row8's returned descriptor additionally requires a separate actual reporting
    gate pair. A caller must never treat the descriptor as execution authority.
    """
    try:
        if (
            citation != _CITATION
            or recognize_printed_rows(source_text, allow_multipage=True) is None
        ):
            return None
        span, formula, proofs = _SIGNED_PROFILES[principal]
        if (
            type(branch.start) is not int
            or type(branch.end) is not int
            or (branch.start, branch.end) != span
        ):
            return None
        if (
            branch.text != source_text[span[0] : span[1]]
            or branch.text != proofs[0].split("\n")[-1]
            and principal.endswith("line_7")
        ):
            return None
        # For mixedcopy andsum9 the whole row is the exact sourceproof.
        if not principal.endswith("line_7") and branch.text != proofs[0]:
            return None
        rules, inputs = policy["rules"], policy["inputs"]
        declarations = [*rules, *inputs]
        named = {r["name"]: r for r in declarations}
        if len(named) != len(declarations):
            return None
        required = {principal}
        if principal == "federal_2025_t936_chart_a_line_9":
            required |= {"federal_2025_t936_chart_a_line_7", "t936_chart_a_line_8"}
        header = source_text.splitlines()[0]
        for name in required:
            r = named[name]
            if r not in rules or any(
                r.get(k) != v
                for k, v in {
                    "kind": "derived",
                    "dtype": "Money",
                    "unit": "CAD",
                    "entity": "Person",
                    "period": "Year",
                }.items()
            ):
                return None
            if any(
                k in r
                for k in (
                    "formula",
                    "default",
                    "rounding",
                    "resolved_currency_rounding",
                    "indexed_by",
                    "values",
                )
            ):
                return None
            versions = r["versions"]
            if not isinstance(versions, list) or len(versions) != 1:
                return None
            v = versions[0]
            if set(v) != {"formula", "effective_from", "effective_to"} or (
                v["effective_from"],
                v["effective_to"],
            ) != ("2025-01-01", "2025-12-31"):
                return None
            _, expected, owned = _SIGNED_PROFILES[name]
            if _tree(v["formula"]) != _tree(expected):
                return None
            obligations = [("formula", "formula", text) for text in owned] + [
                (field, "effective_period", header)
                for field in ("effective_from", "effective_to")
            ]
            for field, kind, text in obligations:
                if source_text.count(text) != 1 or not any(
                    atom.get("path") == "versions[0]." + field
                    and atom.get("kind") == kind
                    and atom.get("source", {}).get("corpus_citation_path") == citation
                    and atom.get("source", {}).get("excerpt") in (text, text + "\n")
                    for atom in r["metadata"]["proof"]["atoms"]
                ):
                    return None
            for node in ast.walk(ast.parse(expected)):
                if not isinstance(node, ast.Name) or node.id in required:
                    continue
                d = named[node.id]
                boolean = node.id == "schedule_3_line_19200_reported"
                dtype = "Boolean" if boolean else "Money"
                description = (
                    "An amount was reported on Schedule3 line19200, including an explicitly reported zero."
                    if boolean
                    else f"Explicit completed 2025 source-line amount: {node.id}. No upstream recalculation or eligibility inference."
                )
                if d not in inputs or any(
                    d.get(k) != val
                    for k, val in {
                        "dtype": dtype,
                        "entity": "Person",
                        "period": "Year",
                        "unit": None if boolean else "CAD",
                        "description": description,
                    }.items()
                ):
                    return None
                if any(
                    k in d
                    for k in (
                        "formula",
                        "versions",
                        "default",
                        "rounding",
                        "resolved_currency_rounding",
                        "indexed_by",
                        "values",
                    )
                ):
                    return None
        selector = (
            "schedule_3_line_19200_reported"
            if principal == "t936_chart_a_line_8"
            else None
        )
        operands = tuple(
            n.id
            for n in sorted(
                (n for n in ast.walk(ast.parse(formula)) if isinstance(n, ast.Name)),
                key=lambda n: (n.lineno, n.col_offset),
            )
            if n.id != selector
        )
        return SignedRole(principal, operands, selector, span)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, SyntaxError):
        return None


def reserves_signed_copy_branch(branch: Any, source_text: str, citation: str) -> bool:
    """Keep this whole mixed source duty out of gate-only evidence routes."""
    span, _, proofs = _SIGNED_PROFILES["t936_chart_a_line_8"]
    return (
        citation == _CITATION
        and type(source_text) is str
        and type(getattr(branch, "start", None)) is int
        and type(getattr(branch, "end", None)) is int
        and (branch.start, branch.end) == span
        and getattr(branch, "text", None) == proofs[0]
        and source_text[span[0] : span[1]] == proofs[0]
        and recognize_printed_rows(source_text, allow_multipage=True) is not None
    )


_CHART_DUTY_TEXT = (
    "Note: If, in 2025, you have capital gains other than from the disposition "
    "of qualified farm or fishing property or qualified small\n"
    "business corporation shares, first complete Chart A on page 4 of this form "
    "to determine if you have additional investment\n"
    "income to include when you calculate your CNIL."
)
_CHART_PRINCIPAL = "t936_should_first_complete_chart_a"
_CHART_SELECTOR = "t936_other_than_qffp_qsbcs_gains_2025"


def reserves_chart_duty(branch: Any, source_text: str, citation: str) -> bool:
    """Recognize the source duty independently of candidate or execution validity."""
    return (
        citation == _CITATION
        and type(source_text) is str
        and type(getattr(branch, "start", None)) is int
        and type(getattr(branch, "end", None)) is int
        and (branch.start, branch.end) == (775, 1075)
        and getattr(branch, "text", None) == _CHART_DUTY_TEXT
        and source_text[775:1075] == _CHART_DUTY_TEXT
        and recognize_printed_rows(source_text, allow_multipage=True) is not None
    )


def bind_chart_duty(
    policy: dict, branch: Any, source_text: str, citation: str
) -> tuple[str, str] | None:
    """Bind the whole qualified Chart instruction to its direct Judgment output."""
    try:
        if not reserves_chart_duty(branch, source_text, citation):
            return None
        rules, inputs = policy["rules"], policy["inputs"]
        declarations = [*rules, *inputs]
        named = {r["name"]: r for r in declarations}
        if len(named) != len(declarations):
            return None
        r, d = named[_CHART_PRINCIPAL], named[_CHART_SELECTOR]
        if r not in rules or d not in inputs:
            return None
        for decl, expected in (
            (
                r,
                {
                    "kind": "derived",
                    "dtype": "Judgment",
                    "entity": "Person",
                    "period": "Year",
                },
            ),
            (
                d,
                {
                    "dtype": "Boolean",
                    "entity": "Person",
                    "period": "Year",
                    "description": "Had2025 capital gains other than the two source-excluded property classes.",
                },
            ),
        ):
            if any(decl.get(k) != v for k, v in expected.items()):
                return None
            if decl.get("unit") is not None or any(
                k in decl
                for k in (
                    "default",
                    "formula",
                    "rounding",
                    "resolved_currency_rounding",
                    "indexed_by",
                    "values",
                )
            ):
                return None
        if "versions" in d:
            return None
        versions = r["versions"]
        if not isinstance(versions, list) or len(versions) != 1:
            return None
        v = versions[0]
        if type(v.get("formula")) is not str:
            return None
        if (
            set(v) != {"formula", "effective_from", "effective_to"}
            or (v["effective_from"], v["effective_to"]) != ("2025-01-01", "2025-12-31")
            or _tree(v["formula"]) != _tree(_CHART_SELECTOR)
        ):
            return None
        header = source_text.splitlines()[0]
        for field, kind, excerpt in (
            ("formula", "formula", _CHART_DUTY_TEXT),
            ("effective_from", "effective_period", header),
            ("effective_to", "effective_period", header),
        ):
            if source_text.count(excerpt) != 1 or not any(
                a.get("path") == "versions[0]." + field
                and a.get("kind") == kind
                and a.get("source", {}).get("corpus_citation_path") == citation
                and a.get("source", {}).get("excerpt") in (excerpt, excerpt + "\n")
                for a in r["metadata"]["proof"]["atoms"]
            ):
                return None
        return _CHART_PRINCIPAL, _CHART_SELECTOR
    except (KeyError, TypeError, ValueError, AttributeError, SyntaxError):
        return None


_REMAINING_CHART_SPANS = {
    "line1_mixed": (7150, 7346),
    "floor14": (8017, 8067),
    "second_shortcut": (8068, 8159),
    "t3_conditional": (8160, 8308),
}
_CHART_ADDITIONAL_RULES = {
    "federal_2025_t936_chart_a_line_3": (
        "schedule_3_line_19900 - t2048_line_17215",
        "Line 1 minus line 2 = 3",
    ),
    "federal_2025_t936_chart_a_line_13": (
        "federal_2025_t936_chart_a_line_11 - t2048_line_17216",
        "Line 11 minus line 12 = (cid:379) – 13",
    ),
    "federal_2025_t936_chart_a_line_17": (
        "t3_box_21_capital_gains - t3_box_30_capital_gains_eligible_for_deduction",
        "Enter the amount from box 21 of all 2025 T3 slips. 68140 15\n"
        "Enter the amount from box 30 of all 2025 T3 slips. – 16\n"
        "Line 15 minus line 16 68150 = 17",
    ),
}


def bind_remaining_chart_role(
    policy: dict, branch: Any, source_text: str, citation: str
) -> str | None:
    """Static closure only: callers still owe each role's actual execution proof.

    Child binders validate existing source contracts; their descriptors are not
    coverage tokens and do not establish execution of those child paragraphs.
    """
    from types import SimpleNamespace

    try:
        if (
            citation != _CITATION
            or type(source_text) is not str
            or type(branch.start) is not int
            or type(branch.end) is not int
            or recognize_printed_rows(source_text, allow_multipage=True) is None
        ):
            return None
        role = next(
            (
                name
                for name, span in _REMAINING_CHART_SPANS.items()
                if span == (branch.start, branch.end)
            ),
            None,
        )
        if role is None or branch.text != source_text[branch.start : branch.end]:
            return None
        rules, inputs = policy["rules"], policy["inputs"]
        declarations = [*rules, *inputs]
        named = {d["name"]: d for d in declarations}
        if len(named) != len(declarations):
            return None
        # ast.parse accepts bytes and ASTs; guard actual authored formula strings
        # before invoking either existing static binder.
        required = {
            "federal_2025_t936_chart_a_inclusion_rate",
            "federal_2025_t936_chart_a_line_7",
            "t936_chart_a_line_8",
            "federal_2025_t936_chart_a_line_9",
            "federal_2025_t936_chart_a_line_11",
            "t936_chart_a_line_14_non_eligible_taxable_capital_gains",
            "federal_2025_t936_chart_a_line_19",
            "federal_2025_t936_chart_a_additional_investment_income_line_20",
            *_CHART_ADDITIONAL_RULES,
        }
        for name in required:
            versions = named[name]["versions"]
            if not isinstance(versions, list) or len(versions) != 1:
                return None
            if type(versions[0].get("formula")) is not str:
                return None
        floor = _PROFILES["chart20_floor"]
        if (
            bind_pair_role(
                policy,
                SimpleNamespace(start=floor[0], end=floor[1], text=floor[5]),
                source_text,
                citation,
                floor[2],
            )
            is None
        ):
            return None
        signed_name = "federal_2025_t936_chart_a_line_9"
        span, _, proofs = _SIGNED_PROFILES[signed_name]
        if (
            bind_signed_role(
                policy,
                SimpleNamespace(start=span[0], end=span[1], text=proofs[0]),
                source_text,
                citation,
                signed_name,
            )
            is None
        ):
            return None
        forbidden = (
            "formula",
            "default",
            "rounding",
            "resolved_currency_rounding",
            "indexed_by",
            "values",
        )
        for name, (formula, proof) in _CHART_ADDITIONAL_RULES.items():
            d = named[name]
            if d not in rules or any(k in d for k in forbidden):
                return None
            if any(
                d.get(k) != v
                for k, v in {
                    "kind": "derived",
                    "dtype": "Money",
                    "unit": "CAD",
                    "entity": "Person",
                    "period": "Year",
                }.items()
            ):
                return None
            v = d["versions"][0]
            if (
                set(v) != {"formula", "effective_from", "effective_to"}
                or (v["effective_from"], v["effective_to"])
                != ("2025-01-01", "2025-12-31")
                or _tree(v["formula"]) != _tree(formula)
            ):
                return None
            for field, kind, excerpt in (
                ("formula", "formula", proof),
                ("effective_from", "effective_period", source_text.splitlines()[0]),
                ("effective_to", "effective_period", source_text.splitlines()[0]),
            ):
                if source_text.count(excerpt) != 1 or not any(
                    a.get("path") == "versions[0]." + field
                    and a.get("kind") == kind
                    and a.get("source", {}).get("corpus_citation_path") == citation
                    and a.get("source", {}).get("excerpt") in (excerpt, excerpt + "\n")
                    for a in d["metadata"]["proof"]["atoms"]
                ):
                    return None
        money_inputs = (
            "schedule_3_line_19900",
            "t2048_line_17215",
            "t2048_line_17216",
            "t3_box_21_capital_gains",
            "t3_box_30_capital_gains_eligible_for_deduction",
        )
        for name in (*money_inputs, "t936_chart_a_line_14_includes_t3"):
            d = named[name]
            boolean = name not in money_inputs
            description = (
                "Line14 includes a T3-slip amount; source-specific membership fact, not aggregate eligibility."
                if boolean
                else f"Explicit completed 2025 source-line amount: {name}. No upstream recalculation or eligibility inference."
            )
            if (
                d not in inputs
                or any(k in d for k in (*forbidden, "versions"))
                or any(
                    d.get(k) != v
                    for k, v in {
                        "dtype": "Boolean" if boolean else "Money",
                        "entity": "Person",
                        "period": "Year",
                        "unit": None if boolean else "CAD",
                        "description": description,
                    }.items()
                )
            ):
                return None
        return role
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, SyntaxError):
        return None


def reserves_chart_line1(branch: Any, source_text: str, citation: str) -> bool:
    """Reserve the complete signed-input plus first-shortcut duty, source-only."""
    text = (
        "Enter the amount from line 19900 of Schedule 3, Capital Gains or Losses (if negative, show the\n"
        'amount in brackets). If the amount is "0", do not complete lines 2 to 17, and enter "0" on line 20. 1'
    )
    return (
        citation == _CITATION
        and type(source_text) is str
        and type(getattr(branch, "start", None)) is int
        and type(getattr(branch, "end", None)) is int
        and (branch.start, branch.end) == (7150, 7346)
        and getattr(branch, "text", None) == text
        and source_text[7150:7346] == text
        and recognize_printed_rows(source_text, allow_multipage=True) is not None
    )


_EXPENSE_CATEGORIES = {
    "t936_repaid_inducements": "Repayments of inducements.",
    "t936_repaid_refund_interest": "Repayments of refund interest.",
    "t936_prior_proceeds_sale_agreement_mortgage_claim": "Sale of agreement for sale or mortgage/hypothecary claim included in proceeds in a previous year under20(5); completed category amount, not locally reconstructed prior sale.",
    "t936_foreign_nonbusiness_tax_20_11_12": "Foreign nonbusiness tax under20(11)/(12).",
    "t936_life_insurance_premiums_property_income": "Life insurance premiums deducted from property income.",
    "t936_cca_certified_films_videotapes": "CCA claimed on certified films/videotapes.",
    "t936_farming_fishing_losses_nonactive_limited_partner": "Farming/fishing losses claimed by nonactive/limited partner.",
}


def bind_expense_group(
    policy: dict, branch: Any, source_text: str, citation: str
) -> str | None:
    """Close the entire expense source contract, without granting execution credit.

    All six vehicle descriptors are static dependencies only. Actual category,
    relation, eight-band inclusion and timber effects remain the caller's duty.
    """
    from .local_vehicle_date_evidence import vehicle_date_role_binding

    try:
        if (
            citation != _CITATION
            or type(source_text) is not str
            or type(branch.start) is not int
            or type(branch.end) is not int
            or (branch.start, branch.end) != (2877, 3984)
            or branch.text != source_text[2877:3984]
            or recognize_printed_rows(source_text, allow_multipage=True) is None
        ):
            return None
        rules, inputs = policy["rules"], policy["inputs"]
        declarations = [*rules, *inputs]
        named = {d["name"]: d for d in declarations}
        if len(named) != len(declarations):
            return None
        period = {"period_kind": "tax_year", "start": "2025-01-01", "end": "2025-12-31"}
        for start in (3630, 3669, 3708, 3747, 3937, 3979):
            if (
                vehicle_date_role_binding(
                    policy,
                    source_text,
                    citation,
                    {"period": period},
                    (start, start + 4),
                )
                is None
            ):
                return None
        header = source_text.splitlines()[0]
        begin = source_text.index("the uncollectible portion of proceeds")
        end = source_text.index("after 2022)", begin) + len("after 2022)")
        paragraph = source_text[begin:end]
        split = paragraph.index("zero-emission")
        # Preserve the literal ordinary excerpt's trailing space before 'and'.
        ordinary = paragraph[:split].removesuffix("and ")
        proofs = {
            "t936_source_year": ("parameter", header),
            "t936_asset_included_uncollectible_portion": ("formula", paragraph),
            "t936_ordinary_vehicle_cost_excluded": ("formula", ordinary),
            "t936_zepv_cost_excluded": ("formula", paragraph[split:]),
            "t936_uncollectible_dispositions_total": ("formula", paragraph),
        }
        caps = {
            30000: "$30,000 for vehicles acquired prior to 2022",
            34000: "$34,000 for vehicles acquired in 2022",
            36000: "$36,000 for vehicles\nacquired in 2023",
            37000: "$37,000 for vehicles acquired in 2024",
            38000: "$38,000 for vehicles acquired in 2025",
            55000: "$55,000 for ZEPV acquired after March 18, 2019",
            59000: "$59,000 for ZEPV acquired\nin 2022",
            61000: "$61,000 for ZEPV acquired after 2022",
        }
        proofs.update(
            {f"t936_vehicle_cost_{v}": ("parameter", e) for v, e in caps.items()}
        )
        dates = [*(f"t936_start_{y}" for y in range(2022, 2027)), "t936_march_18_2019"]
        proofs.update({name: ("definition", paragraph) for name in dates})
        principal = "other_investment_expenses_line_68080"
        printed = 'Any other investment expenses claimed in 2025 to earn property income\n(see the list "Other investment expenses" on the next page) 68080 + 6'
        proofs[principal] = ("formula", printed)

        def proves(d: dict, path: str, kind: str, excerpt: str) -> bool:
            return source_text.count(excerpt) == 1 and any(
                a.get("path") == path
                and a.get("kind") == kind
                and a.get("source", {}).get("corpus_citation_path") == citation
                and a.get("source", {}).get("excerpt") in (excerpt, excerpt + "\n")
                for a in d["metadata"]["proof"]["atoms"]
            )

        for name, (proof_kind, excerpt) in proofs.items():
            d = named[name]
            table = name in dates
            derived = proof_kind == "formula"
            allowed = {"name", "kind", "dtype", "source", "metadata", "versions"}
            if table:
                allowed.add("indexed_by")
            if derived:
                allowed.update(("entity", "period"))
            money = name.startswith("t936_vehicle_cost_") or name in (
                principal,
                "t936_uncollectible_dispositions_total",
                "t936_asset_included_uncollectible_portion",
            )
            if money:
                allowed.add("unit")
            if d not in rules or not set(d) <= allowed:
                return None
            expected_dtype = (
                "Date"
                if table
                else "Money"
                if money
                else "Boolean"
                if derived
                else "Integer"
            )
            if d.get("dtype") != expected_dtype or d.get("kind") != (
                "derived" if derived else "parameter"
            ):
                return None
            if money and d.get("unit") != "CAD":
                return None
            if derived and (d.get("entity"), d.get("period")) != (
                "Person"
                if name in (principal, "t936_uncollectible_dispositions_total")
                else "SourceEntry",
                "Year",
            ):
                return None
            versions = d["versions"]
            field = "values" if table else "formula"
            if type(versions) is not list or len(versions) != 1:
                return None
            v = versions[0]
            if set(v) != {field, "effective_from", "effective_to"} or (
                v["effective_from"],
                v["effective_to"],
            ) != ("2025-01-01", "2025-12-31"):
                return None
            if not table and type(v[field]) is not str:
                return None
            if not proves(d, "versions[0]." + field, proof_kind, excerpt) or not all(
                proves(d, "versions[0]." + f, "effective_period", header)
                for f in ("effective_from", "effective_to")
            ):
                return None
        relation = named["t936_depreciable_disposition_for_return"]
        if (
            relation not in rules
            or set(relation) != {"name", "kind", "source", "data_relation", "metadata"}
            or relation["kind"] != "data_relation"
            or set(relation["data_relation"]) != {"arity", "arguments"}
            or type(relation["data_relation"]["arity"]) is not int
            or relation["data_relation"]["arity"] != 2
            or relation["data_relation"]["arguments"] != ["SourceEntry", "Person"]
            or not proves(relation, "data_relation", "definition", paragraph)
        ):
            return None
        aggregate = "t936_uncollectible_dispositions_total"
        if _tree(named[aggregate]["versions"][0]["formula"]) != _tree(
            "sum(t936_depreciable_disposition_for_return.t936_asset_included_uncollectible_portion)"
        ) or _tree(named[principal]["versions"][0]["formula"]) != _tree(
            " + ".join((*_EXPENSE_CATEGORIES, aggregate))
        ):
            return None
        if not proves(
            named[principal],
            "versions[0].formula",
            "formula",
            source_text[2877:4385].removesuffix("\n"),
        ):
            return None
        exclusion = " Completed property-income deduction category. Excludes business-income expenses,20(1)(j) loan repayments, and interest for income-averaging annuity/RRSP/RPP/DPSP uses as listed by source."
        for name, description in _EXPENSE_CATEGORIES.items():
            d = named[name]
            if d not in inputs or d != {
                "name": name,
                "entity": "Person",
                "dtype": "Money",
                "period": "Year",
                "description": description + exclusion,
                "unit": "CAD",
            }:
                return None
        for name in (
            "t936_asset_acquisition_date",
            "t936_asset_acquisition_cost",
            "t936_asset_uncollectible_portion",
            "t936_asset_is_timber_resource",
            "t936_asset_is_ordinary_passenger_vehicle",
            "t936_asset_is_zero_emission_passenger_vehicle",
        ):
            d = named[name]
            if not set(d) <= {
                "name",
                "entity",
                "dtype",
                "period",
                "description",
                "unit",
            }:
                return None
        return principal
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, SyntaxError):
        return None


def reserves_expense_group(branch: Any, source_text: str, citation: str) -> bool:
    """The complete expense paragraph cannot borrow a single-child witness."""
    return (
        citation == _CITATION
        and type(source_text) is str
        and type(getattr(branch, "start", None)) is int
        and type(getattr(branch, "end", None)) is int
        and (branch.start, branch.end) == (2877, 3984)
        and getattr(branch, "text", None) == source_text[2877:3984]
        and recognize_printed_rows(source_text, allow_multipage=True) is not None
    )
