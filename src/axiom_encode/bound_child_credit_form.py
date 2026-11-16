"""Closed static contract for the reviewed child-credit form graph.

No execution/source-coverage authority. Declaration vocabulary is deliberately
conservative, not alpha-invariant. Input descriptions corroborate roles only;
neither descriptions nor proof metadata authenticate factual truth.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import yaml

from axiom_encode.child_credit_source import (
    ChildCreditForm,
    recognize_child_credit_form,
)

_INPUTS = (
    {
        "name": "ns479_volunteer_firefighters_ground_search_rescue_credit_eligible",
        "entity": "Person",
        "dtype": "Boolean",
        "period": "Year",
        "description": "Explicit externally determined applicability of the volunteer credit; this form does not determine detailed eligibility from the linked guidance.",
    },
    {
        "name": "completing_ns479",
        "entity": "Person",
        "dtype": "Boolean",
        "period": "Year",
        "description": "Explicit use of this form, not a claim that calculations or filing have been completed.",
    },
    {
        "name": "filing_paper_return",
        "entity": "Person",
        "dtype": "Boolean",
        "period": "Year",
        "description": "Explicit paper-return filing context.",
    },
    {
        "name": "using_children_credit_section",
        "entity": "Person",
        "dtype": "Boolean",
        "period": "Year",
        "description": "Explicit use of this section; not an eligibility determination or truth of completed table.",
    },
    {
        "name": "needs_additional_space",
        "entity": "Person",
        "dtype": "Boolean",
        "period": "Year",
        "description": "Explicit need for additional space; not inferred from number of child rows.",
    },
    {
        "name": "taxes_to_pay",
        "entity": "Person",
        "dtype": "Money",
        "period": "Year",
        "description": "Explicit amount of taxes to pay for conditional credit excess; no default and no claim of final refund settlement.",
        "unit": "CAD",
    },
    {
        "name": "child_name",
        "entity": "SourceEntry",
        "dtype": "Text",
        "period": "Year",
        "description": "Column A child name for this annual per-child table row; not a unique global identity or kinship inference.",
    },
    {
        "name": "eligible_expenses_b",
        "entity": "SourceEntry",
        "dtype": "Money",
        "period": "Year",
        "description": "Column B entered annual eligible expenses for this one child; detailed program eligibility is external to this form.",
        "unit": "CAD",
    },
    {
        "name": "assistance_received_or_receivable_c",
        "entity": "SourceEntry",
        "dtype": "Money",
        "period": "Year",
        "description": "Column C assistance received or receivable for expenses in column B, including reimbursements and allowances.",
        "unit": "CAD",
    },
    {
        "name": "amount_from_d_claimed_by_others_e",
        "entity": "SourceEntry",
        "dtype": "Money",
        "period": "Year",
        "description": "Column E entered amount from this child row column D claimed by others.",
        "unit": "CAD",
    },
)
_RULES = (
    (
        "volunteer_amount",
        "ns479_2025_volunteer_credit_amount",
        {
            "kind": "parameter",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": 500,
                }
            ],
            "unit": "CAD",
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "formula",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "Line 1 and line 62400: Volunteer firefighters and ground search and rescue\ntax credit Claim $500 ^",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
    (
        "child_cap",
        "ns479_2025_child_sports_arts_expense_cap",
        {
            "kind": "parameter",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": 500,
                }
            ],
            "unit": "CAD",
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "formula",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "Column D Column B minus column C (Maximum $500 per child) ^",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
    (
        "rate",
        "ns479_2025_child_sports_arts_rate",
        {
            "kind": "parameter",
            "dtype": "Rate",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": 0.0879,
                }
            ],
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "formula",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "Line 3: Applicable rate 8.79%",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
    (
        "row_membership",
        "annual_child_row_for_claimant",
        {
            "kind": "data_relation",
            "data_relation": {"arity": 2, "arguments": ["SourceEntry", "Person"]},
        },
        [
            {
                "path": "data_relation",
                "kind": "definition",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "Line 2 and line 62410: Your children's sports and arts expenses Total of all\namounts in column F of the table above = ^",
                },
            }
        ],
    ),
    (
        "column_a",
        "child_name_column_a",
        {
            "kind": "derived",
            "dtype": "Text",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "child_name",
                }
            ],
            "entity": "SourceEntry",
            "period": "Year",
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "definition",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "Column A Child's name (one child per line) ^",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
    (
        "column_b",
        "eligible_expenses_column_b",
        {
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "eligible_expenses_b",
                }
            ],
            "entity": "SourceEntry",
            "period": "Year",
            "unit": "CAD",
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "definition",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "Column B All eligible expenses per child for the year ^",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
    (
        "column_c",
        "assistance_column_c",
        {
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "assistance_received_or_receivable_c",
                }
            ],
            "entity": "SourceEntry",
            "period": "Year",
            "unit": "CAD",
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "definition",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "Column C Assistance received or receivable for amount in column B ^",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
    (
        "column_e",
        "claimed_by_others_column_e",
        {
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "amount_from_d_claimed_by_others_e",
                }
            ],
            "entity": "SourceEntry",
            "period": "Year",
            "unit": "CAD",
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "definition",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "Column E Amounts from column D claimed by others ^",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
    (
        "column_d",
        "per_child_column_d",
        {
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "min(eligible_expenses_column_b - assistance_column_c, ns479_2025_child_sports_arts_expense_cap)",
                }
            ],
            "entity": "SourceEntry",
            "period": "Year",
            "unit": "CAD",
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "formula",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "Column D: Column B minus column C (Maximum $500 per\nchild);",
                },
            },
            {
                "path": "versions[0].formula",
                "kind": "formula",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "Column D Column B minus column C (Maximum $500 per child) ^",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
    (
        "column_f",
        "per_child_column_f",
        {
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "per_child_column_d - claimed_by_others_column_e",
                }
            ],
            "entity": "SourceEntry",
            "period": "Year",
            "unit": "CAD",
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "formula",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "Column F Column D minus column E ^",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
    (
        "aggregate_d_export",
        "ns479_children_sports_arts_column_d_capped_expenses",
        {
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "sum(annual_child_row_for_claimant.per_child_column_d)",
                }
            ],
            "entity": "Person",
            "period": "Year",
            "unit": "CAD",
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "formula",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "Column D Column B minus column C (Maximum $500 per child) ^\n\nColumn E Amounts from column D claimed by others ^\n\nColumn F Column D minus column E ^",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
    (
        "line2",
        "ns479_children_sports_arts_expenses_line_62410",
        {
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "sum(annual_child_row_for_claimant.per_child_column_f)",
                }
            ],
            "entity": "Person",
            "period": "Year",
            "unit": "CAD",
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "formula",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "Line 2 and line 62410: Your children's sports and arts expenses Total of all\namounts in column F of the table above = ^",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
    (
        "line1",
        "ns479_volunteer_firefighters_ground_search_rescue_tax_credit_line_62400",
        {
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "if ns479_volunteer_firefighters_ground_search_rescue_credit_eligible:\n  ns479_2025_volunteer_credit_amount\nelse:\n  0",
                }
            ],
            "entity": "Person",
            "period": "Year",
            "unit": "CAD",
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "formula",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "Line 1 and line 62400: Volunteer firefighters and ground search and rescue\ntax credit Claim $500 ^",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
    (
        "line4",
        "ns479_children_sports_arts_tax_credit_line_4",
        {
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "ns479_children_sports_arts_expenses_line_62410 * ns479_2025_child_sports_arts_rate",
                }
            ],
            "entity": "Person",
            "period": "Year",
            "unit": "CAD",
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "formula",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "Line 4: Line 2 multiplied by the percentage from line 3 Children's sports and\narts tax credit = ^",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
    (
        "line5",
        "nova_scotia_credits_line_47900",
        {
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "ns479_volunteer_firefighters_ground_search_rescue_tax_credit_line_62400 + ns479_children_sports_arts_tax_credit_line_4",
                }
            ],
            "entity": "Person",
            "period": "Year",
            "unit": "CAD",
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "formula",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "Line 5: Line 1 plus line 4. Enter this amount on line 47900 of your return.\nNova Scotia Credits = ^",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
    (
        "conditional_excess",
        "conditional_credit_excess_over_taxes",
        {
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "if nova_scotia_credits_line_47900 > taxes_to_pay:\n  nova_scotia_credits_line_47900 - taxes_to_pay\nelse:\n  0",
                }
            ],
            "entity": "Person",
            "period": "Year",
            "unit": "CAD",
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "formula",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "You can claim these refundable tax credits even if you do not have to pay any\ntax. If the total of these credits is more than the amount of taxes you have\nto pay, you may get a refund for the difference.",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
    (
        "paper_attachment",
        "paper_return_form_attachment_required",
        {
            "kind": "derived",
            "dtype": "Judgment",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "completing_ns479 and filing_paper_return",
                }
            ],
            "entity": "Person",
            "period": "Year",
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "definition",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "Complete the calculations that apply to you and attach a copy of this form to\nyour paper return.",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
    (
        "table_completion",
        "children_expense_table_completion_required",
        {
            "kind": "derived",
            "dtype": "Judgment",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "using_children_credit_section",
                }
            ],
            "entity": "Person",
            "period": "Year",
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "definition",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "If applicable, complete the following table and include all of your\nchildren's expenses for programs of artistic, cultural, recreational, or\ndevelopmental activity or physical activity. Assistance includes\nreimbursements, allowances, and any form of assistance received or\nreceivable.",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
    (
        "receipts_retention",
        "children_receipts_retention_required",
        {
            "kind": "derived",
            "dtype": "Judgment",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "using_children_credit_section",
                }
            ],
            "entity": "Person",
            "period": "Year",
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "definition",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "Do not include your receipts but keep them in case you are asked to provide\nthem later.",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
    (
        "receipts_non_enclosure",
        "children_receipts_non_enclosure_required",
        {
            "kind": "derived",
            "dtype": "Judgment",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "using_children_credit_section",
                }
            ],
            "entity": "Person",
            "period": "Year",
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "definition",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "Do not include your receipts but keep them in case you are asked to provide\nthem later.",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
    (
        "additional_sheet",
        "additional_sheet_attachment_required",
        {
            "kind": "derived",
            "dtype": "Judgment",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "needs_additional_space",
                }
            ],
            "entity": "Person",
            "period": "Year",
        },
        [
            {
                "path": "versions[0].formula",
                "kind": "definition",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "If you need more space, attach an additional sheet of paper.",
                },
            },
            {
                "path": "versions[0].effective_from",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
            {
                "path": "versions[0].effective_to",
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/cra/t1-2025/ns479-credits",
                    "excerpt": "CRA Form NS479 2025 Nova Scotia Credits",
                },
            },
        ],
    ),
)


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError("duplicate or non-string key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _typed(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _formula(value):
    if type(value) in (int, float):
        return ("number", Decimal(str(value)))
    if not isinstance(value, str) or "#" in value or len(value) > 2000:
        return None
    try:
        return (
            "ast",
            ast.dump(ast.parse(value.strip(), mode="exec"), include_attributes=False),
        )
    except (SyntaxError, ValueError):
        return None


def _endpoint(value, year, end):
    expected = date(year, 12, 31) if end else date(year, 1, 1)
    return type(value) is str and value == expected.isoformat()


def _proof(rule, expected, source, citation, form):
    metadata = rule.get("metadata")
    if not isinstance(metadata, dict) or set(metadata) != {"proof"}:
        return False
    proof = metadata["proof"]
    if not isinstance(proof, dict) or set(proof) != {"atoms"}:
        return False
    atoms = proof["atoms"]
    if not isinstance(atoms, list) or not atoms:
        return False
    found = set()
    endpoints = set()
    header = next(r for r in form.records if r.role == "header")
    anchors = {
        source[slice(*header.span)],
        f"CRA Form NS479 {form.year} Nova Scotia Credits",
    }
    for atom in atoms:
        if not isinstance(atom, dict) or set(atom) != {"path", "kind", "source"}:
            return False
        path, kind, origin = atom["path"], atom["kind"], atom["source"]
        if (
            not isinstance(path, str)
            or not isinstance(kind, str)
            or not isinstance(origin, dict)
            or set(origin) != {"corpus_citation_path", "excerpt"}
        ):
            return False
        excerpt = origin["excerpt"]
        if (
            origin["corpus_citation_path"] != citation
            or not isinstance(excerpt, str)
            or not excerpt
            or excerpt not in source
        ):
            return False
        token = (path, kind, excerpt)
        if token in found:
            return False
        found.add(token)
        if path in {"versions[0].effective_from", "versions[0].effective_to"}:
            if kind != "effective_period" or excerpt not in anchors:
                return False
            endpoints.add(path)
        elif not any(
            path == e["path"]
            and kind == e["kind"]
            and excerpt == e["source"]["excerpt"]
            for e in expected
        ):
            return False
    for atom in expected:
        if atom["kind"] == "effective_period":
            if atom["path"] not in endpoints:
                return False
        elif (atom["path"], atom["kind"], atom["source"]["excerpt"]) not in found:
            return False
    return True


@dataclass(frozen=True)
class BoundChildCreditForm:
    content_sha256: str
    source: ChildCreditForm
    rule_roles: tuple[tuple[str, str], ...]
    input_roles: tuple[str, ...]
    relation_arguments: tuple[str, str]
    declaration_json: str

    def bound(self, content, source, citation):
        return bind_child_credit_form(content, source, citation) == self


def bind_child_credit_form(
    content: str, source: str, citation: str
) -> BoundChildCreditForm | None:
    """Match all declarations; no snapshot, trace or source-coverage admission."""
    if not isinstance(content, str) or len(content) > 150000:
        return None
    form = recognize_child_credit_form(source, citation)
    if form is None:
        return None
    try:
        payload = yaml.load(content, Loader=_UniqueLoader)
        if (
            not isinstance(payload, dict)
            or set(payload)
            != {"format", "module", "units", "inputs", "imports", "rules"}
            or payload["format"] != "rulespec/v1"
            or payload["imports"] != []
        ):
            return None
        module = payload["module"]
        if not isinstance(module, dict) or set(module) - {
            "summary",
            "proof_validation",
            "source_verification",
        }:
            return None
        if "summary" in module and not isinstance(module["summary"], str):
            return None
        if _typed(module.get("proof_validation")) != _typed({"required": True}):
            return None
        if module.get("source_verification") != {
            "corpus_citation_path": citation,
            "source_sha256": form.source_sha256,
        }:
            return None
        if _typed(payload["units"]) != _typed(
            [{"name": "CAD", "kind": "currency", "minor_units": 2}]
        ):
            return None
        inputs = payload["inputs"]
        rules = payload["rules"]
        if (
            not isinstance(inputs, list)
            or not isinstance(rules, list)
            or len(inputs) != len(_INPUTS)
            or len(rules) != len(_RULES)
        ):
            return None
        by_input = {}
        for item in inputs:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or item["name"] in by_input
            ):
                return None
            by_input[item["name"]] = item
        for expected in _INPUTS:
            actual = by_input.get(expected["name"])
            if actual is None:
                return None
            # Exact reviewed optional descriptions, not arbitrary token matches.
            compare = dict(expected)
            if "description" not in actual:
                compare.pop("description")
            if _typed(actual) != _typed(compare):
                return None
        by_rule = {}
        for rule in rules:
            if (
                not isinstance(rule, dict)
                or not isinstance(rule.get("name"), str)
                or rule["name"] in by_rule
                or rule["name"] in by_input
            ):
                return None
            by_rule[rule["name"]] = rule
        for role, name, expected, atoms in _RULES:
            rule = by_rule.get(name)
            if (
                rule is None
                or set(rule) != set(expected) | {"name", "source", "metadata"}
                or not isinstance(rule["source"], str)
                or not rule["source"].strip()
            ):
                return None
            if not _proof(rule, atoms, source, citation, form):
                return None
            for key, value in expected.items():
                if key == "versions":
                    versions = rule[key]
                    if (
                        not isinstance(versions, list)
                        or len(versions) != 1
                        or not isinstance(versions[0], dict)
                        or set(versions[0])
                        != {"effective_from", "effective_to", "formula"}
                    ):
                        return None
                    version = versions[0]
                    if not _endpoint(
                        version["effective_from"], form.year, False
                    ) or not _endpoint(version["effective_to"], form.year, True):
                        return None
                    tree = _formula(version["formula"])
                    if tree is None or tree != _formula(value[0]["formula"]):
                        return None
                elif _typed(rule[key]) != _typed(value):
                    return None
        return BoundChildCreditForm(
            hashlib.sha256(content.encode()).hexdigest(),
            form,
            tuple((role, name) for role, name, _, _ in _RULES),
            tuple(i["name"] for i in _INPUTS),
            ("SourceEntry", "Person"),
            _typed(payload),
        )
    except (ValueError, TypeError, KeyError, RecursionError, yaml.YAMLError):
        return None
