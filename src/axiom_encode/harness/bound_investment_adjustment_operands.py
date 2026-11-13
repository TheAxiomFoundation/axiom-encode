"""Complete Schedule N static operand contract; not execution/source authority."""

from __future__ import annotations

import ast
import hashlib
import io
import re
import tokenize
from dataclasses import dataclass

import yaml

from .bound_investment_adjustment_structure import (
    CITATION,
    recognize_investment_adjustment_source,
)

GUIDE = "ca/policy/revenu-quebec/tp1-2025/main-return/guide"
GUIDE_SHA256 = "d399dff4ffd095abf64a22f40c8834d0129721b6cb22ab28caeebfa6e76dd4af"
PREFIX = "schedule_n_"
RATE = "quebec_2025_schedule_n_resource_recovery_rate"
# These roles come from the printed rows and their source captions, not a policy fixture.
INPUT_ROWS = {
    10: "specified_member_partnership_loss_line_10",
    12: "carrying_charges_interest_line_12",
    14: "exploration_development_expenses_line_14",
    16: "other_property_income_expenses_line_16",
    20: "taxable_canadian_dividends_line_20",
    22: "interest_other_investment_income_line_22",
    24: "specified_member_partnership_income_line_24",
    26: "life_insurance_policy_income_line_26",
    30: "ordinary_annuities_ro_line_30",
    32: "other_trust_shareholder_property_income_line_32",
    34: "taxable_capital_gains_line_34",
    50: "limited_partnership_loss_line_50",
    52: "net_capital_losses_other_years_line_52",
    70: "unused_adjustment_prior_years_line_70",
    78: "amount_from_return_line_252_line_78",
}
OUTPUT_ROWS = {
    18: "investment_expenses_line_18",
    28: "resource_deduction_recovery_line_28",
    36: "investment_income_line_36",
    40: "investment_expense_adjustment_line_260",
    54: "other_investment_expenses_line_54",
    60: "investment_income_excess_line_60",
    64: "other_investment_expense_adjustment_line_276",
    76: "adjustment_available_line_76",
    78: "adjustment_used_line_252",
    80: "adjustment_carryforward_line_80",
}
FLAGS = (
    "limited_partnership_loss_deduction_entered",
    "net_capital_loss_deduction_entered",
    "deducting_net_capital_loss_from_qualified_capital_gains",
    "wants_copy_for_records",
    "completing_schedule_n",
)
MONEY_DESCRIPTION = "Explicit amount entered at the named Schedule N line for this individual; no implicit default or upstream eligibility inference."
MONEY_ROLE_DESCRIPTIONS = {
    PREFIX
    + INPUT_ROWS[
        78
    ]: "Already computed amount entered on return252, copied to Schedule N line78; not a requested deduction or a local entitlement assertion. No default.",
    PREFIX
    + INPUT_ROWS[
        14
    ]: "Already calculated line14 result under guide260: 50% of (return241 minus the sum of the three specified Quebec expense amounts); supplied result, not raw241.",
    PREFIX
    + INPUT_ROWS[
        16
    ]: "Entered source-defined property-income expenses excluding bad debts as guide260 directs.",
    PREFIX
    + INPUT_ROWS[
        30
    ]: "Entered ordinary-annuity subtotal from RL2 boxB only where provenance codeRO applies.",
    PREFIX
    + INPUT_ROWS[
        34
    ]: "Already calculated line34 result under guide260 and its qualified-property work chart; supplied result, not arbitrary raw139.",
    PREFIX
    + INPUT_ROWS[
        70
    ]: "Supplied unused historical adjustment: total lines40/64 since2004 minus amounts already used in previous years; not inferred from current amounts.",
}
FLAG_DESCRIPTIONS = dict(
    zip(
        FLAGS,
        (
            "Explicit limited-partnership-loss deduction presence; other return289 losses do not establish this fact.",
            "Explicit net-capital-loss deduction presence, including entered zero; not positive amount inference.",
            "Explicit source-defined circumstance triggering contact; upstream gain qualification is supplied, not locally calculated.",
            "Explicit desire for a records copy, not compliance evidence.",
            "Explicit schedule-use fact for enclosure duty, not eligibility.",
        ),
        strict=True,
    )
)
RECOVERY = "Recovery of resource deductions (see line 154, point 11, in the guide)\nmultiplied by 50%"
CONTACT = "However, if you are deducting a net capital loss from capital gains that qualify for the capital gains deduction, contact us."
PART_C = "09 Adjustment of other investment expenses The amount that you deduct as investment expenses cannot exceed your investment income. If you are deducting either of the following, complete Schedule N to calculate the amount to enter on line 276 of your return: • a limited partnership loss (included on line 289 of your return); • net capital losses from other years. On line 52 of Schedule N, enter the amount from line 290 of your return."
CARRYOVER = "If, for 2025, your investment income (line 36 of Schedule N) is greater than your investment expenses (lines 18 and 54 of Schedule N), you can deduct the unused portion of the adjustment of investment expenses (up to the difference between your investment income and your investment expenses). Unused portion of the adjustment of investment expenses (line 70 of Schedule N) The unused portion of the adjustment of investment expenses is equal to the total of the amounts entered since 2004 on lines 40 and 64 of Schedule N, minus the amount already used to reduce your investment income for a previous year. "


@dataclass(frozen=True)
class InvestmentAdjustmentBinding:
    content_sha256: str
    parent_sha256: str
    guide_sha256: str
    rule_expressions: tuple
    # The copy is not a local entitlement decision; factual truth stays upstream.
    residuals: tuple[str, ...] = ()
    upstream_boundaries: tuple[str, ...] = (
        "return252_is_already_computed_supplied_fact",
        "source_upper_conditions_report_is_not_comprehensive_eligibility",
    )

    def bound(self, content: str, parent: str, guide: str) -> bool:
        return bind_investment_adjustment(content, parent, guide) == self


def _expression(text: str) -> str:
    """Parse the closed scalar grammar without executing candidate expressions."""
    if type(text) is not str or len(text) > 12000:
        raise ValueError("bounded formula required")
    if any(
        t.type == tokenize.COMMENT
        for t in tokenize.generate_tokens(io.StringIO(text).readline)
    ):
        raise ValueError("comments outside grammar")
    match = re.fullmatch(r"match ([A-Za-z_]\w*):\n +true => ([^\n]+)\n?", text)
    if match:
        return "match_true:" + match[1] + ":" + _expression(match[2])
    # RuleSpec's one-line conditional uses a colon before both expression arms.
    inline = re.fullmatch(r"if ([^\n]+?): ([^\n]+?) else: ([^\n]+)", text)
    if inline:
        text = f"if {inline[1]}:\n    {inline[2]}\nelse:\n    {inline[3]}"
    tree = ast.parse(text, mode="exec")
    allowed = (
        ast.Module,
        ast.Expr,
        ast.If,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.Compare,
        ast.Eq,
        ast.Gt,
        ast.GtE,
        ast.LtE,
        ast.Call,
    )
    if any(type(n) not in allowed for n in ast.walk(tree)):
        raise ValueError("node outside closed grammar")
    if len(tree.body) != 1:
        raise ValueError("one expression required")
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) not in (int, float):
            raise ValueError("numeric literal required; Boolean is a RuleSpec name")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name)
            or node.func.id not in ("min", "max")
            or len(node.args) != 2
            or node.keywords
        ):
            raise ValueError("two operand min/max only")
        if isinstance(node, ast.If) and (
            len(node.body) != 1
            or len(node.orelse) != 1
            or not isinstance(node.body[0], ast.Expr)
            or not isinstance(node.orelse[0], ast.Expr)
        ):
            raise ValueError("expression arms required")
    return ast.dump(tree, include_attributes=False)


def _contracts(parent: str) -> tuple:
    """Build ordered formulas independently from the reviewed printed row graph."""
    inp = {n: PREFIX + role for n, role in INPUT_ROWS.items()}
    out = {n: PREFIX + role for n, role in OUTPUT_ROWS.items()}
    applies = PREFIX + "part_c_applies"
    consistent = PREFIX + "part_c_input_consistent"
    net = PREFIX + "net_investment_income_for_carryover"
    maximum = PREFIX + "maximum_allowable_return_252"
    selected = PREFIX + "entered_return252_within_source_upper_conditions"
    part_c_start = parent.index("Complete this part only if")
    part_c_end = parent.index("\n\nD\n", part_c_start)
    part_c = parent[part_c_start:part_c_end] + "\n"
    return (
        (RATE, "Rate", "0.5", ((CITATION, RECOVERY),)),
        (
            applies,
            "Boolean",
            f"if {FLAGS[0]} == true or {FLAGS[1]} == true:\n    true\nelse:\n    false",
            ((GUIDE, PART_C),),
        ),
        (
            consistent,
            "Boolean",
            f"if {applies} == true or ({inp[50]} == 0 and {inp[52]} == 0):\n    true\nelse:\n    false",
            ((CITATION, part_c),),
        ),
        (net, "Money", f"{out[36]} - {out[18]} - {out[54]}", ((GUIDE, CARRYOVER),)),
        (
            maximum,
            "Money",
            f"if {net} > 0: min({inp[70]}, {net}) else: 0",
            ((GUIDE, CARRYOVER),),
        ),
        (
            selected,
            "Boolean",
            f"if {inp[78]} == 0 or ({net} > 0 and {inp[78]} <= {inp[70]} and {inp[78]} <= {net}):\n    true\nelse:\n    false",
            ((GUIDE, CARRYOVER),),
        ),
        (
            out[18],
            "Money",
            " + ".join(inp[n] for n in (10, 12, 14, 16)),
            ((CITATION, "Add lines 10 through 16."),),
        ),
        (
            out[28],
            "Money",
            f"{PREFIX}resource_deduction_recovery_base * {RATE}",
            ((CITATION, RECOVERY),),
        ),
        (
            out[36],
            "Money",
            " + ".join(
                out[28] if n == 28 else inp[n] for n in (20, 22, 24, 26, 28, 30, 32, 34)
            ),
            ((CITATION, "Add lines 20 through 34."),),
        ),
        (
            out[40],
            "Money",
            f"max(0, {out[18]} - {out[36]})",
            (
                (
                    CITATION,
                    "Subtract line 36 from line 18. If the result is negative, enter 0.\nCarry the result to line 260 of your return.",
                ),
            ),
        ),
        (
            out[54],
            "Money",
            f"match {consistent}:\n    true => {inp[50]} + {inp[52]}",
            ((CITATION, part_c),),
        ),
        (
            out[60],
            "Money",
            f"max(0, {out[36]} - {out[18]})",
            (
                (
                    CITATION,
                    "Subtract line 58 from line 56. If the result is negative, enter 0.",
                ),
            ),
        ),
        (
            out[64],
            "Money",
            f"max(0, {out[54]} - {out[60]})",
            (
                (
                    CITATION,
                    "Subtract line 60 from line 54. If the result is negative, enter 0.\nCarry the result to line 276 of your return.",
                ),
            ),
        ),
        (
            out[76],
            "Money",
            f"{inp[70]} + {out[40]} + {out[64]}",
            ((CITATION, "Add lines 70 through 74."),),
        ),
        (
            out[78],
            "Money",
            inp[78],
            ((CITATION, "Amount from line 252 of your 2025 return"),),
        ),
        (
            out[80],
            "Money",
            f"{out[76]} - {out[78]}",
            ((CITATION, "Subtract line 78 from line 76."),),
        ),
        (
            PREFIX + "enclosure_required",
            "Judgment",
            "completing_schedule_n == true",
            ((CITATION, "Enclose this schedule with your return."),),
        ),
        (
            PREFIX + "copy_guide_reference_applies",
            "Judgment",
            "wants_copy_for_records == true",
            (
                (
                    CITATION,
                    "If you want to keep a copy for your records, see the guide.",
                ),
            ),
        ),
        (
            PREFIX + "qualified_gain_loss_contact_required",
            "Judgment",
            FLAGS[2] + " == true",
            ((GUIDE, CONTACT),),
        ),
    )


def bind_investment_adjustment(
    content: str, parent: str, guide: str
) -> InvestmentAdjustmentBinding | None:
    """Match the full static contract, retaining explicit unresolved obligations."""
    from .validator_pipeline import _safe_load_unique_keys

    try:
        source = recognize_investment_adjustment_source(parent)
        if source is None or hashlib.sha256(guide.encode()).hexdigest() != GUIDE_SHA256:
            return None
        if type(content) is not str or len(content) > 100000:
            return None
        doc = _safe_load_unique_keys(content)
        if type(doc) is not dict or set(doc) != {
            "format",
            "imports",
            "module",
            "units",
            "inputs",
            "rules",
        }:
            return None
        if (
            doc["format"] != "rulespec/v1"
            or doc["imports"] != []
            or doc["units"] != [{"name": "CAD", "kind": "currency", "minor_units": 2}]
        ):
            return None
        if type(doc["units"][0]["minor_units"]) is not int:
            return None
        module = doc["module"]
        if (
            set(module) != {"summary", "proof_validation", "source_verification"}
            or type(module["summary"]) is not str
        ):
            return None
        if (
            module["proof_validation"] != {"required": True}
            or type(module["proof_validation"]["required"]) is not bool
        ):
            return None
        if module["source_verification"] != {
            "corpus_citation_path": CITATION,
            "source_sha256": source.source_sha256,
        }:
            return None
        expected_inputs = {PREFIX + v: "Money" for v in INPUT_ROWS.values()}
        expected_inputs[PREFIX + "resource_deduction_recovery_base"] = "Money"
        expected_inputs.update(dict.fromkeys(FLAGS, "Boolean"))
        if type(doc["inputs"]) is not list or len(doc["inputs"]) != len(
            expected_inputs
        ):
            return None
        seen = set()
        for item in doc["inputs"]:
            name = item["name"]
            if (
                name in seen
                or name not in expected_inputs
                or type(item.get("description")) is not str
            ):
                return None
            seen.add(name)
            expected = {
                "name": name,
                "entity": "Person",
                "dtype": expected_inputs[name],
                "period": "Year",
                "description": (
                    MONEY_ROLE_DESCRIPTIONS.get(name, MONEY_DESCRIPTION)
                    if expected_inputs[name] == "Money"
                    else FLAG_DESCRIPTIONS[name]
                ),
            }
            if expected_inputs[name] == "Money":
                expected["unit"] = "CAD"
            if item != expected:
                return None
        contracts = _contracts(parent)
        if type(doc["rules"]) is not list or len(doc["rules"]) != len(contracts):
            return None
        rules = {r["name"]: r for r in doc["rules"]}
        if len(rules) != len(contracts) or set(rules) != {r[0] for r in contracts}:
            return None
        title = parent.split("\n", 1)[0]
        expressions = []
        for name, dtype, formula, proofs in contracts:
            rule = rules[name]
            versions = rule["versions"]
            if type(versions) is not list or len(versions) != 1:
                return None
            version = versions[0]
            if (
                set(version) != {"effective_from", "effective_to", "formula"}
                or version["effective_from"] != "2025-01-01"
                or version["effective_to"] != "2025-12-31"
            ):
                return None
            expression = _expression(version["formula"])
            if expression != _expression(formula):
                return None
            atoms = [
                {
                    "path": "versions[0].formula",
                    "kind": "formula",
                    "source": {"corpus_citation_path": citation, "excerpt": excerpt},
                }
                for citation, excerpt in proofs
            ]
            for citation, excerpt in proofs:
                if excerpt not in (parent if citation == CITATION else guide):
                    return None
            atoms += [
                {
                    "path": "versions[0]." + key,
                    "kind": "effective_period",
                    "source": {"corpus_citation_path": CITATION, "excerpt": title},
                }
                for key in ("effective_from", "effective_to")
            ]
            expected_rule = {
                "name": name,
                "kind": "parameter" if name == RATE else "derived",
                "dtype": dtype,
                "source": proofs[0][1],
                "versions": versions,
                "metadata": {"proof": {"atoms": atoms}},
            }
            if name != RATE:
                expected_rule.update(entity="Person", period="Year")
            if dtype == "Money":
                expected_rule["unit"] = "CAD"
            if rule != expected_rule:
                return None
            expressions.append((name, expression))
        return InvestmentAdjustmentBinding(
            hashlib.sha256(content.encode()).hexdigest(),
            source.source_sha256,
            GUIDE_SHA256,
            tuple(expressions),
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        SyntaxError,
        RecursionError,
        yaml.YAMLError,
        tokenize.TokenError,
    ):
        return None
