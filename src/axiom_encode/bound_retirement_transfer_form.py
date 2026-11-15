"""Closed source/AST/type contract for a selected-pair retirement transfer.

This standalone binding authenticates no factual truth, execution, relation
membership or publication. It grants no numeric or condition coverage.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass

import yaml

from .retirement_transfer_source import (
    RetirementTransferSource,
    recognize_retirement_transfer_schedule,
)

# Reviewed source-role declarations. Descriptions corroborate the declared role;
# neither names nor descriptions prove that a caller's fact is true.
_INPUTS = (
    {
        "name": "schedule_q_retirement_income_line_122_line_10",
        "entity": "Person",
        "dtype": "Money",
        "period": "Year",
        "description": "Entered amount from line122 on line10.",
        "unit": "CAD",
    },
    {
        "name": "schedule_q_annuity_rrsp_rrif_prpp_transfer_line_12",
        "entity": "Person",
        "dtype": "Money",
        "period": "Year",
        "description": "Entered deduction included in amount on line10, 12.",
        "unit": "CAD",
    },
    {
        "name": "schedule_q_unused_prpp_vrsp_refund_deduction_line_13",
        "entity": "Person",
        "dtype": "Money",
        "period": "Year",
        "description": "Entered deduction included in amount on line10, 13.",
        "unit": "CAD",
    },
    {
        "name": "schedule_q_line_293_deduction_line_14",
        "entity": "Person",
        "dtype": "Money",
        "period": "Year",
        "description": "Entered deduction included in amount on line10, 14.",
        "unit": "CAD",
    },
    {
        "name": "schedule_q_line_297_point_12_deduction_line_16",
        "entity": "Person",
        "dtype": "Money",
        "period": "Year",
        "description": "Entered deduction included in amount on line10, 16.",
        "unit": "CAD",
    },
    {
        "name": "schedule_q_requested_retirement_income_transfer_line_22",
        "entity": "Person",
        "dtype": "Money",
        "period": "Year",
        "description": "Requested selected retirement transfer, not automatic half.",
        "unit": "CAD",
    },
    {
        "name": "quebec_rl2_box_j_from_line20",
        "entity": "Person",
        "dtype": "Money",
        "period": "Year",
        "description": "Quebec withholding from income on line20, RL2 boxJ.",
        "unit": "CAD",
    },
    {
        "name": "quebec_rl1_box_e_from_included_compensation",
        "entity": "Person",
        "dtype": "Money",
        "period": "Year",
        "description": "RL1 boxE for retirement compensation included in line20.",
        "unit": "CAD",
    },
    {
        "name": "has_selected_spouse_at_reference_date",
        "entity": "Person",
        "dtype": "Boolean",
        "period": "Year",
        "description": "Explicit association to selected spouse at reference date, not household "
        "inference.",
    },
    {
        "name": "jointly_decided_retirement_income_transfer",
        "entity": "Person",
        "dtype": "Boolean",
        "period": "Year",
        "description": "Explicit joint decision of selected transferor and recipient.",
    },
    {
        "name": "compensation_included_in_line20",
        "entity": "Person",
        "dtype": "Boolean",
        "period": "Year",
        "description": "Whether income on line20 includes retirement compensation arrangement amounts.",
    },
    {
        "name": "received_retirement_compensation_amounts",
        "entity": "Person",
        "dtype": "Boolean",
        "period": "Year",
        "description": "Whether claimant received retirement compensation arrangement amounts; guide "
        "referral premise.",
    },
    {
        "name": "is_completing_retirement_transfer_schedule",
        "entity": "Person",
        "dtype": "Boolean",
        "period": "Year",
        "description": "Explicit transferor form-use, not actual enclosure or filing.",
    },
    {
        "name": "wants_retirement_transfer_schedule_copy",
        "entity": "Person",
        "dtype": "Boolean",
        "period": "Year",
        "description": "Explicit wish to keep a copy, not compliance.",
    },
    {
        "name": "transferor_birth_date",
        "entity": "Person",
        "dtype": "Date",
        "period": "Year",
        "description": "Actual transferor date of birth.",
    },
    {
        "name": "spouse_reference_date",
        "entity": "Person",
        "dtype": "Date",
        "period": "Year",
        "description": "Date at which selected spouse association fact is asserted.",
    },
)

_RULES = (
    (
        "cap_rate",
        "quebec_2025_schedule_q_transfer_cap_rate",
        {
            "kind": "parameter",
            "dtype": "Rate",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": 0.5,
                }
            ],
        },
        (
            (
                "versions[0].formula",
                "formula",
                "selected_transfer",
                "recipient_income_destination",
                "",
            ),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
    (
        "deductions",
        "schedule_q_retirement_income_deductions_line_18",
        {
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "schedule_q_annuity_rrsp_rrif_prpp_transfer_line_12 + "
                    "schedule_q_unused_prpp_vrsp_refund_deduction_line_13 + "
                    "schedule_q_line_293_deduction_line_14 + "
                    "schedule_q_line_297_point_12_deduction_line_16",
                }
            ],
            "entity": "Person",
            "period": "Year",
            "unit": "CAD",
        },
        (
            (
                "versions[0].formula",
                "formula",
                "deduction12",
                "deductions_total",
                "\n=\n",
            ),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
    (
        "net_income",
        "schedule_q_eligible_retirement_income_line_20",
        {
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "schedule_q_retirement_income_line_122_line_10 - "
                    "schedule_q_retirement_income_deductions_line_18",
                }
            ],
            "entity": "Person",
            "period": "Year",
            "unit": "CAD",
        },
        (
            ("versions[0].formula", "formula", "net_income", "net_income", ""),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
    (
        "normalized_request",
        "normalized_requested_retirement_income_transfer",
        {
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "min(max(0, schedule_q_requested_retirement_income_transfer_line_22), "
                    "schedule_q_eligible_retirement_income_line_20 * "
                    "quebec_2025_schedule_q_transfer_cap_rate)",
                }
            ],
            "entity": "Person",
            "period": "Year",
            "unit": "CAD",
        },
        (
            (
                "versions[0].formula",
                "formula",
                "selected_transfer",
                "recipient_income_destination",
                "",
            ),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
    (
        "permission",
        "retirement_income_transfer_permission_applies",
        {
            "kind": "derived",
            "dtype": "Judgment",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "transferor_birth_date < "
                    "retirement_transfer_birth_cutoff[schedule_q_source_year] and "
                    "has_selected_spouse_at_reference_date and spouse_reference_date == "
                    "retirement_transfer_spouse_reference_date[schedule_q_source_year] and "
                    "jointly_decided_retirement_income_transfer",
                }
            ],
            "entity": "Person",
            "period": "Year",
        },
        (
            ("versions[0].formula", "condition", "permission", "permission", " "),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
    (
        "authorized_transfer",
        "schedule_q_retirement_income_transferred_line_245",
        {
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "if retirement_income_transfer_permission_applies:\n"
                    "  normalized_requested_retirement_income_transfer\n"
                    "else:\n"
                    "  0",
                }
            ],
            "entity": "Person",
            "period": "Year",
            "unit": "CAD",
        },
        (
            (
                "versions[0].formula",
                "formula",
                "selected_transfer",
                "recipient_income_destination",
                "",
            ),
            ("versions[0].formula", "condition", "permission", "permission", " "),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
    (
        "entered_withholding",
        "entered_quebec_withholding_line50",
        {
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "if compensation_included_in_line20:\n"
                    "  quebec_rl2_box_j_from_line20 + "
                    "quebec_rl1_box_e_from_included_compensation\n"
                    "else:\n"
                    "  quebec_rl2_box_j_from_line20",
                }
            ],
            "entity": "Person",
            "period": "Year",
            "unit": "CAD",
        },
        (
            (
                "versions[0].formula",
                "formula",
                "entered_rl2_withholding",
                "conditional_rl1_withholding",
                "\n",
            ),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
    (
        "transferred_withholding",
        "schedule_q_quebec_withholding_transferred_line_451_1",
        {
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "if schedule_q_retirement_income_transferred_line_245 > 0:\n"
                    "  entered_quebec_withholding_line50 * "
                    "schedule_q_retirement_income_transferred_line_245 / "
                    "schedule_q_eligible_retirement_income_line_20\n"
                    "else:\n"
                    "  0",
                }
            ],
            "entity": "Person",
            "period": "Year",
            "unit": "CAD",
        },
        (
            (
                "versions[0].formula",
                "formula",
                "withholding_numerator",
                "recipient_withholding_destination",
                "",
            ),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
    (
        "selected_relation",
        "retirement_income_transferred_to",
        {
            "kind": "data_relation",
            "data_relation": {"arity": 2, "arguments": ["Person", "Person"]},
        },
        (("data_relation", "definition", "permission", "permission", " "),),
    ),
    (
        "recipient_income",
        "retirement_income_received_line123",
        {
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "sum(retirement_income_transferred_to.schedule_q_retirement_income_transferred_line_245)",
                }
            ],
            "entity": "Person",
            "period": "Year",
            "unit": "CAD",
        },
        (
            (
                "versions[0].formula",
                "formula",
                "recipient_income_destination",
                "recipient_income_destination",
                "",
            ),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
    (
        "recipient_withholding",
        "quebec_withholding_received_line451_3",
        {
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "sum(retirement_income_transferred_to.schedule_q_quebec_withholding_transferred_line_451_1)",
                }
            ],
            "entity": "Person",
            "period": "Year",
            "unit": "CAD",
        },
        (
            (
                "versions[0].formula",
                "formula",
                "recipient_withholding_destination",
                "recipient_withholding_destination",
                "",
            ),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
    (
        "withholding_duty",
        "corresponding_withholding_transfer_required",
        {
            "kind": "derived",
            "dtype": "Judgment",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "schedule_q_retirement_income_transferred_line_245 > 0",
                }
            ],
            "entity": "Person",
            "period": "Year",
        },
        (
            (
                "versions[0].formula",
                "condition",
                "corresponding_withholding",
                "corresponding_withholding",
                "",
            ),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
    (
        "enclosure",
        "retirement_transfer_schedule_enclosure_required",
        {
            "kind": "derived",
            "dtype": "Judgment",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "is_completing_retirement_transfer_schedule and "
                    "schedule_q_retirement_income_transferred_line_245 > 0",
                }
            ],
            "entity": "Person",
            "period": "Year",
        },
        (
            (
                "versions[0].formula",
                "condition",
                "transferor_enclosure",
                "transferor_enclosure",
                "",
            ),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
    (
        "identification",
        "spouse_identification_reporting_required",
        {
            "kind": "derived",
            "dtype": "Judgment",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "is_completing_retirement_transfer_schedule",
                }
            ],
            "entity": "Person",
            "period": "Year",
        },
        (
            ("versions[0].formula", "condition", "spouse_sin", "spouse_name", "\n3"),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
    (
        "compensation_guide",
        "compensation_eligible_income_guide_consultation_required",
        {
            "kind": "derived",
            "dtype": "Judgment",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "received_retirement_compensation_amounts",
                }
            ],
            "entity": "Person",
            "period": "Year",
        },
        (
            (
                "versions[0].formula",
                "condition",
                "entered_income",
                "compensation_guide",
                "\n",
            ),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
    (
        "copy_guide",
        "retirement_transfer_copy_guide_consultation_required",
        {
            "kind": "derived",
            "dtype": "Judgment",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "wants_retirement_transfer_schedule_copy",
                }
            ],
            "entity": "Person",
            "period": "Year",
        },
        (
            ("versions[0].formula", "condition", "copy_guide", "copy_guide", ""),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
    (
        "recipient_income_duty",
        "recipient_income_reporting_required",
        {
            "kind": "derived",
            "dtype": "Judgment",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "retirement_income_received_line123 > 0",
                }
            ],
            "entity": "Person",
            "period": "Year",
        },
        (
            (
                "versions[0].formula",
                "condition",
                "recipient_income_destination",
                "recipient_income_destination",
                "",
            ),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
    (
        "recipient_withholding_duty",
        "recipient_withholding_reporting_required",
        {
            "kind": "derived",
            "dtype": "Judgment",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "retirement_income_received_line123 > 0",
                }
            ],
            "entity": "Person",
            "period": "Year",
        },
        (
            (
                "versions[0].formula",
                "condition",
                "recipient_withholding_destination",
                "recipient_withholding_destination",
                "",
            ),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
    (
        "birth_cutoff",
        "retirement_transfer_birth_cutoff",
        {
            "kind": "parameter",
            "dtype": "Date",
            "indexed_by": "source_year",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "values": {"2025": "1961-01-01"},
                }
            ],
        },
        (
            ("versions[0].values", "definition", "permission", "permission", " "),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
    (
        "spouse_reference",
        "retirement_transfer_spouse_reference_date",
        {
            "kind": "parameter",
            "dtype": "Date",
            "indexed_by": "source_year",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "values": {"2025": "2025-12-31"},
                }
            ],
        },
        (
            ("versions[0].values", "definition", "permission", "permission", " "),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
    (
        "source_year",
        "schedule_q_source_year",
        {
            "kind": "parameter",
            "dtype": "Integer",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": 2025,
                }
            ],
        },
        (
            ("versions[0].formula", "definition", "dated_anchor", None, ""),
            (
                "versions[0].effective_from",
                "effective_period",
                "dated_anchor",
                None,
                "",
            ),
            ("versions[0].effective_to", "effective_period", "dated_anchor", None, ""),
        ),
    ),
)


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if type(key) not in (str, int) or key in result:
            raise ValueError("duplicate or unsupported mapping key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _typed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _tree(value):
    if not isinstance(value, str) or "#" in value or len(value) > 3000:
        return None
    try:
        return ast.dump(ast.parse(value.strip(), mode="exec"), include_attributes=False)
    except (SyntaxError, ValueError, RecursionError):
        return None


def _span(source, role):
    return next(r.span for r in source.records if r.role == role)


def _anchors(source):
    a, b = _span(source, "header")
    result = {source.source[a:b]}
    # The complete recognizer validates any resolver wrapper. Only its full
    # exact title is an alternative anchor; a bare year cannot anchor a date.
    if a:
        result.add(source.source[:a].rstrip("\r\n"))
    return result


def _proof(rule, requirements, source, citation):
    metadata = rule.get("metadata")
    if not isinstance(metadata, dict) or set(metadata) != {"proof"}:
        return False
    proof = metadata["proof"]
    if not isinstance(proof, dict) or set(proof) != {"atoms"}:
        return False
    atoms = proof["atoms"]
    if not isinstance(atoms, list) or len(atoms) != len(requirements):
        return False
    permitted = {}
    for path, kind, first, last, tail in requirements:
        if first == "dated_anchor":
            excerpts = _anchors(source)
        else:
            a = _span(source, first)[0]
            b = _span(source, last)[1]
            pattern = re.escape(tail).replace("\\" + "\n", r"\r?\n")
            match = re.match(pattern, source.source[b:])
            if match is None:
                return False
            excerpts = {source.source[a : b + match.end()]}
        permitted[(path, kind)] = permitted.get((path, kind), set()) | excerpts
    found = set()
    for atom in atoms:
        if not isinstance(atom, dict) or set(atom) != {"path", "kind", "source"}:
            return False
        path, kind, origin = atom["path"], atom["kind"], atom["source"]
        if (
            type(path) is not str
            or type(kind) is not str
            or not isinstance(origin, dict)
            or set(origin) != {"corpus_citation_path", "excerpt"}
        ):
            return False
        excerpt = origin["excerpt"]
        if (
            origin["corpus_citation_path"] != citation
            or type(excerpt) is not str
            or not excerpt
            or source.source.count(excerpt) != 1
        ):
            return False
        token = (path, kind, excerpt)
        if token in found or excerpt not in permitted.get((path, kind), set()):
            return False
        found.add(token)
    # Every independently required role must occur, even when path/kind agree.
    for path, kind, first, last, tail in requirements:
        if first == "dated_anchor":
            expected = _anchors(source)
        else:
            a = _span(source, first)[0]
            b = _span(source, last)[1]
            match = re.match(
                re.escape(tail).replace("\\" + "\n", r"\r?\n"), source.source[b:]
            )
            expected = {source.source[a : b + match.end()]}
        if not any((path, kind, excerpt) in found for excerpt in expected):
            return False
    return True


def _expected(role, declaration, source):
    result = deepcopy(declaration)
    for v in result.get("versions", []):
        v["effective_from"] = f"{source.year}-01-01"
        v["effective_to"] = f"{source.year}-12-31"
        if role == "source_year":
            v["formula"] = source.year
        if role == "birth_cutoff":
            v["values"] = {source.year: source.birth_cutoff.isoformat()}
        if role == "spouse_reference":
            v["values"] = {source.year: source.spouse_reference_date.isoformat()}
        if role == "cap_rate":
            v["formula"] = float(source.cap_percent / 100)
    return result


@dataclass(frozen=True)
class BoundRetirementTransferForm:
    content_sha256: str
    source: RetirementTransferSource
    rule_roles: tuple[tuple[str, str], ...]
    input_names: tuple[str, ...]
    relation_arguments: tuple[str, str]
    declaration_json: str

    def bound(self, content, source, citation):
        return bind_retirement_transfer_form(content, source, citation) == self


def bind_retirement_transfer_form(content, source_text, citation):
    """Bind the complete source graph, without creating execution authority."""
    source = recognize_retirement_transfer_schedule(source_text, citation)
    if source is None or not isinstance(content, str) or len(content) > 200_000:
        return None
    try:
        if any(
            isinstance(t, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken))
            for t in yaml.scan(content)
        ):
            return None
        payload = yaml.load(content, Loader=_UniqueLoader)
        if (
            not isinstance(payload, dict)
            or set(payload)
            - {"format", "module", "units", "inputs", "rules", "imports"}
            or payload.get("format") != "rulespec/v1"
            or payload.get("imports", []) != []
        ):
            return None
        module = payload.get("module")
        if (
            not isinstance(module, dict)
            or set(module) != {"summary", "proof_validation", "source_verification"}
            or type(module["summary"]) is not str
        ):
            return None
        if _typed(module["proof_validation"]) != _typed({"required": True}):
            return None
        if _typed(module["source_verification"]) != _typed(
            {"corpus_citation_path": citation, "source_sha256": source.source_sha256}
        ):
            return None
        if _typed(payload.get("units")) != _typed(
            [{"name": "CAD", "kind": "currency", "minor_units": 2}]
        ):
            return None
        inputs, rules = payload.get("inputs"), payload.get("rules")
        if (
            not isinstance(inputs, list)
            or len(inputs) != len(_INPUTS)
            or not isinstance(rules, list)
            or len(rules) != len(_RULES)
        ):
            return None
        by_input = {}
        for item in inputs:
            if (
                not isinstance(item, dict)
                or type(item.get("name")) is not str
                or item["name"] in by_input
            ):
                return None
            by_input[item["name"]] = item
        for expected in _INPUTS:
            actual = by_input.get(expected["name"])
            if actual is None:
                return None
            wanted = dict(expected)
            if "description" not in actual:
                wanted.pop("description")
            if _typed(actual) != _typed(wanted):
                return None
        by_rule = {}
        for rule in rules:
            if (
                not isinstance(rule, dict)
                or type(rule.get("name")) is not str
                or rule["name"] in by_rule
                or rule["name"] in by_input
            ):
                return None
            by_rule[rule["name"]] = rule
        for role, name, declaration, proof in _RULES:
            expected = _expected(role, declaration, source)
            rule = by_rule.get(name)
            if (
                rule is None
                or set(rule) != set(expected) | {"name", "source", "metadata"}
                or type(rule["source"]) is not str
                or not rule["source"].strip()
            ):
                return None
            if not _proof(rule, proof, source, citation):
                return None
            for key, wanted in expected.items():
                if key != "versions":
                    if _typed(rule[key]) != _typed(wanted):
                        return None
                    continue
                actual = rule[key]
                if (
                    not isinstance(actual, list)
                    or len(actual) != 1
                    or not isinstance(actual[0], dict)
                    or set(actual[0]) != set(wanted[0])
                ):
                    return None
                v, w = actual[0], wanted[0]
                if (
                    type(v["effective_from"]) is not str
                    or type(v["effective_to"]) is not str
                    or v["effective_from"] != w["effective_from"]
                    or v["effective_to"] != w["effective_to"]
                ):
                    return None
                if "values" in w:
                    values = v["values"]
                    if (
                        not isinstance(values, dict)
                        or len(values) != 1
                        or any(type(k) is not int for k in values)
                        or set(values) != {source.year}
                        or type(values[source.year]) is not str
                        or values[source.year] != w["values"][source.year]
                    ):
                        return None
                elif type(w["formula"]) in (int, float):
                    if (
                        type(v["formula"]) is not type(w["formula"])
                        or v["formula"] != w["formula"]
                    ):
                        return None
                elif _tree(v["formula"]) is None or _tree(v["formula"]) != _tree(
                    w["formula"]
                ):
                    return None
        return BoundRetirementTransferForm(
            hashlib.sha256(content.encode()).hexdigest(),
            source,
            tuple((r, n) for r, n, _, _ in _RULES),
            tuple(i["name"] for i in _INPUTS),
            ("Person", "Person"),
            _typed(payload),
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        RecursionError,
        OverflowError,
        yaml.YAMLError,
    ):
        return None
