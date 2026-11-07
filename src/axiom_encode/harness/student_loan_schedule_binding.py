"""Closed source/AST ownership; neither input truth nor execution evidence."""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass

import yaml

from axiom_encode.student_loan_schedule_source import StudentLoanScheduleSource


def _normalized(text):
    return " ".join(text.split())


def _expression(formula):
    if not isinstance(formula, str) or "#" in formula or ";" in formula:
        return None
    try:
        if formula.startswith("if "):
            # The supported RuleSpec branch form is not a Python conditional expression.
            head, rest = formula.split(":", 1)
            yes, no = rest.split(" else:", 1)
            node = ast.parse(head + ":\n " + yes.strip() + "\nelse:\n " + no.strip())
            if (
                len(node.body) != 1
                or not isinstance(node.body[0], ast.If)
                or len(node.body[0].body) != 1
                or len(node.body[0].orelse) != 1
                or not isinstance(node.body[0].body[0], ast.Expr)
                or not isinstance(node.body[0].orelse[0], ast.Expr)
            ):
                return None
        else:
            node = ast.parse(formula, mode="eval")
        return ast.dump(node, include_attributes=False)
    except (SyntaxError, ValueError):
        return None


def _names(items):
    if not isinstance(items, list) or any(not isinstance(i, dict) for i in items):
        return None
    names = [i.get("name") for i in items]
    if any(
        not isinstance(n, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", n) for n in names
    ):
        return None
    if len(set(names)) != len(names):
        return None
    return dict(zip(names, items, strict=True))


def _proof(rule, source, required):
    metadata = rule.get("metadata")
    if not isinstance(metadata, dict) or set(metadata) != {"proof"}:
        return False
    proof = metadata["proof"]
    if not isinstance(proof, dict) or set(proof) != {"atoms"}:
        return False
    atoms = proof["atoms"]
    if not isinstance(atoms, list) or not atoms:
        return False
    formulas = set()
    dates = set()
    raw = _normalized(source.source)
    for atom in atoms:
        if not isinstance(atom, dict) or set(atom) != {"path", "kind", "source"}:
            return False
        if not isinstance(atom["path"], str) or not isinstance(atom["kind"], str):
            return False
        evidence = atom["source"]
        if (
            not isinstance(evidence, dict)
            or set(evidence) != {"corpus_citation_path", "excerpt"}
            or evidence["corpus_citation_path"] != source.citation
            or not isinstance(evidence["excerpt"], str)
            or not evidence["excerpt"].strip()
            or _normalized(evidence["excerpt"]) not in raw
        ):
            return False
        if atom["path"] == "versions[0].formula":
            if atom["kind"] not in {"formula", "condition", "definition", "unit"}:
                return False
            if atom["kind"] != "unit":
                formulas.add(_normalized(evidence["excerpt"]))
        elif atom["path"] in {"versions[0].effective_from", "versions[0].effective_to"}:
            if atom["kind"] != "effective_period" or _normalized(
                evidence["excerpt"]
            ) not in {
                f"Revenu Quebec Schedule M {source.year} - Interest Paid on a Student Loan",
                f"T P -1. D. M -V ({source.year}-12)",
                f"Interest paid in {source.year} on a student loan",
                f"Amount carried to line 385 of your {source.year} return",
            }:
                return False
            if _normalized(evidence["excerpt"]) in {
                f"Revenu Quebec Schedule M {source.year} - Interest Paid on a Student Loan",
                f"T P -1. D. M -V ({source.year}-12)",
            }:
                dates.add(atom["path"])
        else:
            return False
    return (
        dates == {"versions[0].effective_from", "versions[0].effective_to"}
        and {_normalized(x) for x in required} <= formulas
    )


@dataclass(frozen=True)
class StudentLoanScheduleBinding:
    source: StudentLoanScheduleSource
    content_sha256: str
    input_roles: tuple[tuple[str, str], ...]
    rule_roles: tuple[tuple[str, str], ...]
    formulas: tuple[tuple[str, str], ...]


def bind_student_loan_schedule(source, content):
    """Require every declaration to belong to the reviewed source/AST graph.

    Conservative names corroborate the reviewed fact mapping, not factual truth.
    Input metadata is never used as authenticated proof. No capture is admitted.
    """
    if not isinstance(source, StudentLoanScheduleSource) or not source.bound(
        source.source, source.citation
    ):
        return None
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict) or set(data) != {
        "format",
        "module",
        "units",
        "inputs",
        "rules",
    }:
        return None
    if data["format"] != "rulespec/v1" or data["units"] != [
        {"name": "CAD", "kind": "currency", "minor_units": 2}
    ]:
        return None
    if type(data["units"][0]["minor_units"]) is not int:
        return None
    module = data["module"]
    if (
        not isinstance(module, dict)
        or set(module) != {"summary", "proof_validation", "source_verification"}
        or module["proof_validation"] != {"required": True}
        or type(module["proof_validation"].get("required")) is not bool
        or module["source_verification"]
        != {
            "corpus_citation_path": source.citation,
            "source_sha256": source.source_sha256,
        }
    ):
        return None
    inputs, rules = _names(data["inputs"]), _names(data["rules"])
    if (
        inputs is None
        or rules is None
        or len(inputs) != 9
        or len(rules) != 7
        or inputs.keys() & rules.keys()
    ):
        return None
    suffixes = {
        "calculated": f"carryforward_calculated_for_{source.prior_year}",
        "prior_unused": "unused_student_loan_interest_line_46",
        "history_paid": f"student_loan_interest_paid_{source.history_first_year}_through_{source.prior_year}",
        "history_used": "student_loan_interest_used_for_previous_year_credits",
        "current_paid": f"student_loan_interest_paid_{source.year}_line_48",
        "requested": "student_loan_interest_claimed_line_60",
        "decision": f"decided_to_carry_{source.prior_year}_student_loan_interest_to_future_year",
        "form_use": "is_completing_student_loan_interest_schedule",
        "copy_wish": "wants_to_keep_student_loan_interest_schedule_copy",
    }
    roles = {}
    booleans = {"calculated", "decision", "form_use", "copy_wish"}
    for role, suffix in suffixes.items():
        candidates = [
            n
            for n in inputs
            if (n == suffix or n.endswith("_" + suffix))
            and not set(n.split("_")) & {"no", "not", "never", "without"}
        ]
        if len(candidates) != 1:
            return None
        name = candidates[0]
        item = inputs[name]
        expected = {
            "name": name,
            "entity": "Person",
            "dtype": "Boolean" if role in booleans else "Money",
            "period": "Year",
        }
        if role not in booleans:
            expected["unit"] = "CAD"
        if {k: v for k, v in item.items() if k != "description"} != expected:
            return None
        if "description" in item:
            description = item["description"]
            if not isinstance(description, str):
                return None
            # Closed reviewed descriptions, not token intersections. Unknown
            # prose refuses, including negated definitions with copied keywords.
            descriptions = {
                "calculated": f"For {source.prior_year}, you calculated an amount for interest paid on a student loan.",
                "prior_unused": "Unused portion of the interest paid on a student loan.",
                "history_paid": f"Interest paid for {source.history_first_year} through {source.prior_year}.",
                "history_used": "Interest already used to calculate a credit for interest paid on a student loan for previous years.",
                "current_paid": f"Interest paid in {source.year} on a student loan.",
                "requested": f"Amount carried to line 385 of your {source.year} return.",
                "decision": f"Explicit prior carry decision separate from whether{source.prior_year} calculation occurred; never inferred from positive money or treated alone as entitlement.",
                "form_use": "Explicit claimant form-use context, not eligibility, filing completed or enclosure compliance.",
                "copy_wish": "Explicit wish to keep a copy of this schedule; not actual retention or guide consultation.",
            }
            if _normalized(description) != descriptions[role]:
                return None
        roles[role] = name
    if set(roles.values()) != set(inputs):
        return None
    text = {r.role: source.source[slice(*r.span)] for r in source.records}
    required = {
        "line46": [text["historical_reconstruction"], text["entered_prior"]],
        "line52": [
            "Add lines 46 and 48.",
            f"Interest paid in {source.year} on a student loan",
        ],
        "line385": [
            "Carry all or part of this amount to line 385 of your return.",
            f"Amount carried to line 385 of your {source.year} return",
        ],
        "line62": ["Subtract line 60 from line 52."],
        "completion": [text["current_completion"], text["prior_completion"]],
        "enclosure": [text["enclosure"]],
        "guide": [text["guide_referral"]],
    }
    owners = {}
    for role, excerpts in required.items():
        candidates = [n for n, rule in rules.items() if _proof(rule, source, excerpts)]
        if len(candidates) != 1:
            return None
        owners[role] = candidates[0]
    if set(owners.values()) != set(rules):
        return None
    f = roles
    expected_formulas = {
        "line46": f"if {f['calculated']}: {f['prior_unused']} else: {f['history_paid']} - {f['history_used']}",
        "line52": f"{owners['line46']} + {f['current_paid']}",
        "line385": f"min({owners['line52']}, max(0, {f['requested']}))",
        "line62": f"{owners['line52']} - {owners['line385']}",
        "completion": f"({f['current_paid']} > 0) or ({f['calculated']} and {f['decision']})",
        "enclosure": f["form_use"],
        "guide": f["copy_wish"],
    }
    for role, name in owners.items():
        rule = rules[name]
        money = role.startswith("line")
        keys = {
            "name",
            "kind",
            "entity",
            "dtype",
            "period",
            "source",
            "metadata",
            "versions",
        } | ({"unit"} if money else set())
        if (
            set(rule) != keys
            or rule["kind"] != "derived"
            or rule["entity"] != "Person"
            or rule["period"] != "Year"
            or rule["dtype"] != ("Money" if money else "Judgment")
            or (money and rule["unit"] != "CAD")
        ):
            return None
        versions = rule["versions"]
        if (
            not isinstance(versions, list)
            or len(versions) != 1
            or not isinstance(versions[0], dict)
        ):
            return None
        v = versions[0]
        if (
            set(v) != {"effective_from", "effective_to", "formula"}
            or v["effective_from"] != f"{source.year}-01-01"
            or v["effective_to"] != f"{source.year}-12-31"
            or _expression(v["formula"]) != _expression(expected_formulas[role])
        ):
            return None
    return StudentLoanScheduleBinding(
        source,
        hashlib.sha256(content.encode()).hexdigest(),
        tuple(roles.items()),
        tuple(owners.items()),
        tuple((owners[k], v) for k, v in expected_formulas.items()),
    )
