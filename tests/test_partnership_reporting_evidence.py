from __future__ import annotations

import copy
import functools
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.harness import partnership_reporting_evidence as e
from axiom_encode.harness import source_completeness as s
from axiom_encode.harness import validator_pipeline as v

MATERIAL = {
    "payload": {
        "format": "rulespec/v1",
        "module": {
            "summary": "Revenu Quebec Schedule L 2025 business income: "
            "business, farming,\n"
            "fishing, professional, commission, cryptoasset and "
            "partnership income\n"
            "aggregation for TP-1 line 164.",
            "source_verification": {
                "corpus_citation_path": "ca/policy/revenu-quebec/tp1-2025/schedule-l-business-income",
                "upstream_source_check": {
                    "status": "official_parameter_source",
                    "checked_paths": ["quebec/statute/taxation-act"],
                    "rationale": "Revenu "
                    "Quebec "
                    "Schedule "
                    "L "
                    "is "
                    "the "
                    "official "
                    "2025 "
                    "schedule "
                    "for "
                    "aggregating "
                    "net "
                    "business "
                    "income "
                    "for "
                    "the "
                    "TP-1 "
                    "return.",
                },
            },
        },
        "rules": [
            {
                "name": "schedule_l_reported_cryptoassets_gross_income_line_16_1",
                "kind": "derived",
                "entity": "Person",
                "dtype": "Money",
                "unit": "CAD",
                "period": "Year",
                "versions": [
                    {
                        "effective_from": "2025-01-01",
                        "effective_to": "2025-12-31",
                        "formula": "if "
                        "schedule_l_is_member_of_partnership: "
                        "schedule_l_rl15_box_14_7_amount "
                        "else: "
                        "schedule_l_cryptoassets_gross_income_line_16_1",
                    }
                ],
                "metadata": {
                    "proof": {
                        "atoms": [
                            {
                                "path": "versions[0].formula",
                                "kind": "formula",
                                "source": {
                                    "corpus_citation_path": "ca/policy/revenu-quebec/tp1-2025/schedule-l-business-income",
                                    "excerpt": "Enter "
                                    "the "
                                    "amount "
                                    "from "
                                    "box "
                                    "14-7 "
                                    "of "
                                    "your "
                                    "RL-15 "
                                    "slip "
                                    "on "
                                    "line "
                                    "16.1.",
                                },
                            },
                            {
                                "path": "versions[0].effective_from",
                                "kind": "effective_period",
                                "source": {
                                    "corpus_citation_path": "ca/policy/revenu-quebec/tp1-2025/schedule-l-business-income",
                                    "excerpt": "Revenu "
                                    "Quebec "
                                    "Schedule "
                                    "L "
                                    "2025 "
                                    "- "
                                    "Business "
                                    "Income",
                                },
                            },
                            {
                                "path": "versions[0].effective_to",
                                "kind": "effective_period",
                                "source": {
                                    "corpus_citation_path": "ca/policy/revenu-quebec/tp1-2025/schedule-l-business-income",
                                    "excerpt": "Revenu "
                                    "Quebec "
                                    "Schedule "
                                    "L "
                                    "2025 "
                                    "- "
                                    "Business "
                                    "Income",
                                },
                            },
                            {
                                "path": "versions[0].formula",
                                "kind": "formula",
                                "source": {
                                    "corpus_citation_path": "ca/policy/revenu-quebec/tp1-2025/schedule-l-business-income",
                                    "excerpt": "If you are a member of a partnership:",
                                },
                            },
                        ]
                    }
                },
            },
            {
                "name": "schedule_l_reported_cryptoassets_net_income_line_26_1",
                "kind": "derived",
                "entity": "Person",
                "dtype": "Money",
                "unit": "CAD",
                "period": "Year",
                "versions": [
                    {
                        "effective_from": "2025-01-01",
                        "effective_to": "2025-12-31",
                        "formula": "if "
                        "schedule_l_is_member_of_partnership: "
                        "schedule_l_rl15_box_1_12_amount "
                        "else: "
                        "schedule_l_cryptoassets_net_income_line_26_1",
                    }
                ],
                "metadata": {
                    "proof": {
                        "atoms": [
                            {
                                "path": "versions[0].formula",
                                "kind": "formula",
                                "source": {
                                    "corpus_citation_path": "ca/policy/revenu-quebec/tp1-2025/schedule-l-business-income",
                                    "excerpt": "Enter "
                                    "the "
                                    "amount "
                                    "from "
                                    "box "
                                    "1-12 "
                                    "of "
                                    "the "
                                    "RL-15 "
                                    "slip "
                                    "on "
                                    "line "
                                    "26.1.",
                                },
                            },
                            {
                                "path": "versions[0].effective_from",
                                "kind": "effective_period",
                                "source": {
                                    "corpus_citation_path": "ca/policy/revenu-quebec/tp1-2025/schedule-l-business-income",
                                    "excerpt": "Revenu "
                                    "Quebec "
                                    "Schedule "
                                    "L "
                                    "2025 "
                                    "- "
                                    "Business "
                                    "Income",
                                },
                            },
                            {
                                "path": "versions[0].effective_to",
                                "kind": "effective_period",
                                "source": {
                                    "corpus_citation_path": "ca/policy/revenu-quebec/tp1-2025/schedule-l-business-income",
                                    "excerpt": "Revenu "
                                    "Quebec "
                                    "Schedule "
                                    "L "
                                    "2025 "
                                    "- "
                                    "Business "
                                    "Income",
                                },
                            },
                            {
                                "path": "versions[0].formula",
                                "kind": "formula",
                                "source": {
                                    "corpus_citation_path": "ca/policy/revenu-quebec/tp1-2025/schedule-l-business-income",
                                    "excerpt": "If you are a member of a partnership:",
                                },
                            },
                        ]
                    }
                },
            },
            {
                "name": "schedule_l_primary_net_business_income_line_27",
                "kind": "derived",
                "entity": "Person",
                "dtype": "Money",
                "unit": "CAD",
                "period": "Year",
                "versions": [
                    {
                        "effective_from": "2025-01-01",
                        "effective_to": "2025-12-31",
                        "formula": "schedule_l_business_other_net_income_line_22 "
                        "+ "
                        "schedule_l_farming_net_income_line_23 "
                        "+ "
                        "schedule_l_fishing_net_income_line_24 "
                        "+ "
                        "schedule_l_profession_net_income_line_25 "
                        "+ "
                        "schedule_l_commission_net_income_line_26 "
                        "+ "
                        "schedule_l_reported_cryptoassets_net_income_line_26_1",
                    }
                ],
                "metadata": {
                    "proof": {
                        "atoms": [
                            {
                                "path": "versions[0].formula",
                                "kind": "formula",
                                "source": {
                                    "corpus_citation_path": "ca/policy/revenu-quebec/tp1-2025/schedule-l-business-income",
                                    "excerpt": "Add lines 22 through 26.1.",
                                },
                            },
                            {
                                "path": "versions[0].effective_from",
                                "kind": "effective_period",
                                "source": {
                                    "corpus_citation_path": "ca/policy/revenu-quebec/tp1-2025/schedule-l-business-income",
                                    "excerpt": "Revenu "
                                    "Quebec "
                                    "Schedule "
                                    "L "
                                    "2025 "
                                    "- "
                                    "Business "
                                    "Income",
                                },
                            },
                            {
                                "path": "versions[0].effective_to",
                                "kind": "effective_period",
                                "source": {
                                    "corpus_citation_path": "ca/policy/revenu-quebec/tp1-2025/schedule-l-business-income",
                                    "excerpt": "Revenu "
                                    "Quebec "
                                    "Schedule "
                                    "L "
                                    "2025 "
                                    "- "
                                    "Business "
                                    "Income",
                                },
                            },
                        ]
                    }
                },
            },
            {
                "name": "schedule_l_net_business_income_line_164",
                "kind": "derived",
                "entity": "Person",
                "dtype": "Money",
                "unit": "CAD",
                "period": "Year",
                "versions": [
                    {
                        "effective_from": "2025-01-01",
                        "effective_to": "2025-12-31",
                        "formula": "schedule_l_primary_net_business_income_line_27 "
                        "+ "
                        "schedule_l_retiring_partner_or_physician_allowance_line_28 "
                        "+ "
                        "schedule_l_specified_member_partnership_income_line_29 "
                        "+ "
                        "schedule_l_post_operation_commissions_line_30",
                    }
                ],
                "metadata": {
                    "proof": {
                        "atoms": [
                            {
                                "path": "versions[0].formula",
                                "kind": "formula",
                                "source": {
                                    "corpus_citation_path": "ca/policy/revenu-quebec/tp1-2025/schedule-l-business-income",
                                    "excerpt": "Add lines 27 through 30.",
                                },
                            },
                            {
                                "path": "versions[0].effective_from",
                                "kind": "effective_period",
                                "source": {
                                    "corpus_citation_path": "ca/policy/revenu-quebec/tp1-2025/schedule-l-business-income",
                                    "excerpt": "Revenu "
                                    "Quebec "
                                    "Schedule "
                                    "L "
                                    "2025 "
                                    "- "
                                    "Business "
                                    "Income",
                                },
                            },
                            {
                                "path": "versions[0].effective_to",
                                "kind": "effective_period",
                                "source": {
                                    "corpus_citation_path": "ca/policy/revenu-quebec/tp1-2025/schedule-l-business-income",
                                    "excerpt": "Revenu "
                                    "Quebec "
                                    "Schedule "
                                    "L "
                                    "2025 "
                                    "- "
                                    "Business "
                                    "Income",
                                },
                            },
                        ]
                    }
                },
            },
            {
                "name": "schedule_l_family_resource_qpip_qpp_earnings_line_40",
                "kind": "derived",
                "entity": "Person",
                "dtype": "Money",
                "unit": "CAD",
                "period": "Year",
                "versions": [
                    {
                        "effective_from": "2025-01-01",
                        "effective_to": "2025-12-31",
                        "formula": "schedule_l_lm53_line_8_family_resource_qpip_qpp_earnings",
                    }
                ],
                "metadata": {
                    "proof": {
                        "atoms": [
                            {
                                "path": "versions[0].formula",
                                "kind": "formula",
                                "source": {
                                    "corpus_citation_path": "ca/policy/revenu-quebec/tp1-2025/schedule-l-business-income",
                                    "excerpt": "Insurable "
                                    "earnings "
                                    "(QPIP) "
                                    "and "
                                    "pensionable "
                                    "earnings "
                                    "(QPP) "
                                    "of a "
                                    "person "
                                    "responsible "
                                    "for a "
                                    "family-type "
                                    "resource\n"
                                    "or an "
                                    "intermediate "
                                    "resource "
                                    "(line "
                                    "8 of "
                                    "form "
                                    "LM-53-V)",
                                },
                            },
                            {
                                "path": "versions[0].effective_from",
                                "kind": "effective_period",
                                "source": {
                                    "corpus_citation_path": "ca/policy/revenu-quebec/tp1-2025/schedule-l-business-income",
                                    "excerpt": "Revenu "
                                    "Quebec "
                                    "Schedule "
                                    "L "
                                    "2025 "
                                    "- "
                                    "Business "
                                    "Income",
                                },
                            },
                            {
                                "path": "versions[0].effective_to",
                                "kind": "effective_period",
                                "source": {
                                    "corpus_citation_path": "ca/policy/revenu-quebec/tp1-2025/schedule-l-business-income",
                                    "excerpt": "Revenu "
                                    "Quebec "
                                    "Schedule "
                                    "L "
                                    "2025 "
                                    "- "
                                    "Business "
                                    "Income",
                                },
                            },
                        ]
                    }
                },
            },
        ],
        "units": [{"name": "CAD", "kind": "currency", "minor_units": 2}],
        "inputs": [
            {
                "name": "schedule_l_business_other_net_income_line_22",
                "entity": "Person",
                "dtype": "Money",
                "period": "Year",
                "unit": "CAD",
            },
            {
                "name": "schedule_l_commission_net_income_line_26",
                "entity": "Person",
                "dtype": "Money",
                "period": "Year",
                "unit": "CAD",
            },
            {
                "name": "schedule_l_cryptoassets_gross_income_line_16_1",
                "entity": "Person",
                "dtype": "Money",
                "period": "Year",
                "unit": "CAD",
            },
            {
                "name": "schedule_l_cryptoassets_net_income_line_26_1",
                "entity": "Person",
                "dtype": "Money",
                "period": "Year",
                "unit": "CAD",
            },
            {
                "name": "schedule_l_farming_net_income_line_23",
                "entity": "Person",
                "dtype": "Money",
                "period": "Year",
                "unit": "CAD",
            },
            {
                "name": "schedule_l_fishing_net_income_line_24",
                "entity": "Person",
                "dtype": "Money",
                "period": "Year",
                "unit": "CAD",
            },
            {
                "name": "schedule_l_is_member_of_partnership",
                "entity": "Person",
                "dtype": "Boolean",
                "period": "Year",
            },
            {
                "name": "schedule_l_lm53_line_8_family_resource_qpip_qpp_earnings",
                "entity": "Person",
                "dtype": "Money",
                "period": "Year",
                "unit": "CAD",
            },
            {
                "name": "schedule_l_post_operation_commissions_line_30",
                "entity": "Person",
                "dtype": "Money",
                "period": "Year",
                "unit": "CAD",
            },
            {
                "name": "schedule_l_profession_net_income_line_25",
                "entity": "Person",
                "dtype": "Money",
                "period": "Year",
                "unit": "CAD",
            },
            {
                "name": "schedule_l_retiring_partner_or_physician_allowance_line_28",
                "entity": "Person",
                "dtype": "Money",
                "period": "Year",
                "unit": "CAD",
            },
            {
                "name": "schedule_l_rl15_box_14_7_amount",
                "entity": "Person",
                "dtype": "Money",
                "period": "Year",
                "unit": "CAD",
            },
            {
                "name": "schedule_l_rl15_box_1_12_amount",
                "entity": "Person",
                "dtype": "Money",
                "period": "Year",
                "unit": "CAD",
            },
            {
                "name": "schedule_l_specified_member_partnership_income_line_29",
                "entity": "Person",
                "dtype": "Money",
                "period": "Year",
                "unit": "CAD",
            },
        ],
    },
    "cases": [
        {
            "name": "schedule_l_nonpartner_completed_amounts",
            "period": {
                "period_kind": "tax_year",
                "start": "2025-01-01",
                "end": "2025-12-31",
            },
            "input": {
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_business_other_net_income_line_22": 10000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_farming_net_income_line_23": 2000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_fishing_net_income_line_24": 3000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_profession_net_income_line_25": 4000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_commission_net_income_line_26": 5000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_retiring_partner_or_physician_allowance_line_28": 700,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_specified_member_partnership_income_line_29": 800,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_post_operation_commissions_line_30": 900,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_lm53_line_8_family_resource_qpip_qpp_earnings": 12000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_is_member_of_partnership": False,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_cryptoassets_net_income_line_26_1": 600,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_cryptoassets_gross_income_line_16_1": 1000,
            },
            "output": {
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_reported_cryptoassets_gross_income_line_16_1": 1000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_reported_cryptoassets_net_income_line_26_1": 600,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_primary_net_business_income_line_27": 24600,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_net_business_income_line_164": 27000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_family_resource_qpip_qpp_earnings_line_40": 12000,
            },
        },
        {
            "name": "schedule_l_partner_slip_transfers",
            "period": {
                "period_kind": "tax_year",
                "start": "2025-01-01",
                "end": "2025-12-31",
            },
            "input": {
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_business_other_net_income_line_22": 10000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_farming_net_income_line_23": 2000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_fishing_net_income_line_24": 3000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_profession_net_income_line_25": 4000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_commission_net_income_line_26": 5000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_retiring_partner_or_physician_allowance_line_28": 700,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_specified_member_partnership_income_line_29": 800,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_post_operation_commissions_line_30": 900,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_lm53_line_8_family_resource_qpip_qpp_earnings": 12000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_is_member_of_partnership": True,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_rl15_box_14_7_amount": 1400,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_rl15_box_1_12_amount": 400,
            },
            "output": {
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_reported_cryptoassets_gross_income_line_16_1": 1400,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_reported_cryptoassets_net_income_line_26_1": 400,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_primary_net_business_income_line_27": 24400,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_net_business_income_line_164": 26800,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_family_resource_qpip_qpp_earnings_line_40": 12000,
            },
        },
        {
            "name": "schedule_l_gross_does_not_enter_net_total",
            "period": {
                "period_kind": "tax_year",
                "start": "2025-01-01",
                "end": "2025-12-31",
            },
            "input": {
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_business_other_net_income_line_22": 10000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_farming_net_income_line_23": 2000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_fishing_net_income_line_24": 3000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_profession_net_income_line_25": 4000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_commission_net_income_line_26": 5000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_retiring_partner_or_physician_allowance_line_28": 700,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_specified_member_partnership_income_line_29": 800,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_post_operation_commissions_line_30": 900,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_lm53_line_8_family_resource_qpip_qpp_earnings": 12000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_is_member_of_partnership": True,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_rl15_box_14_7_amount": 9999,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_rl15_box_1_12_amount": 400,
            },
            "output": {
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_reported_cryptoassets_gross_income_line_16_1": 9999,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_reported_cryptoassets_net_income_line_26_1": 400,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_primary_net_business_income_line_27": 24400,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_net_business_income_line_164": 26800,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_family_resource_qpip_qpp_earnings_line_40": 12000,
            },
        },
        {
            "name": "schedule_l_net_business_loss",
            "period": {
                "period_kind": "tax_year",
                "start": "2025-01-01",
                "end": "2025-12-31",
            },
            "input": {
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_business_other_net_income_line_22": -1000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_farming_net_income_line_23": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_fishing_net_income_line_24": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_profession_net_income_line_25": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_commission_net_income_line_26": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_retiring_partner_or_physician_allowance_line_28": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_specified_member_partnership_income_line_29": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_post_operation_commissions_line_30": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_lm53_line_8_family_resource_qpip_qpp_earnings": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_is_member_of_partnership": False,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_cryptoassets_net_income_line_26_1": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_cryptoassets_gross_income_line_16_1": 0,
            },
            "output": {
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_reported_cryptoassets_gross_income_line_16_1": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_reported_cryptoassets_net_income_line_26_1": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_primary_net_business_income_line_27": -1000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_net_business_income_line_164": -1000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_family_resource_qpip_qpp_earnings_line_40": 0,
            },
        },
        {
            "name": "schedule_l_special_income_excluded_from_line27",
            "period": {
                "period_kind": "tax_year",
                "start": "2025-01-01",
                "end": "2025-12-31",
            },
            "input": {
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_business_other_net_income_line_22": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_farming_net_income_line_23": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_fishing_net_income_line_24": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_profession_net_income_line_25": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_commission_net_income_line_26": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_retiring_partner_or_physician_allowance_line_28": 700,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_specified_member_partnership_income_line_29": 800,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_post_operation_commissions_line_30": 900,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_lm53_line_8_family_resource_qpip_qpp_earnings": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_is_member_of_partnership": False,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_cryptoassets_net_income_line_26_1": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_cryptoassets_gross_income_line_16_1": 0,
            },
            "output": {
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_reported_cryptoassets_gross_income_line_16_1": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_reported_cryptoassets_net_income_line_26_1": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_primary_net_business_income_line_27": 0,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_net_business_income_line_164": 2400,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_family_resource_qpip_qpp_earnings_line_40": 0,
            },
        },
        {
            "name": "schedule_l_lm53_direct_transfer",
            "period": {
                "period_kind": "tax_year",
                "start": "2025-01-01",
                "end": "2025-12-31",
            },
            "input": {
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_business_other_net_income_line_22": 10000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_farming_net_income_line_23": 2000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_fishing_net_income_line_24": 3000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_profession_net_income_line_25": 4000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_commission_net_income_line_26": 5000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_retiring_partner_or_physician_allowance_line_28": 700,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_specified_member_partnership_income_line_29": 800,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_post_operation_commissions_line_30": 900,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_lm53_line_8_family_resource_qpip_qpp_earnings": 12345,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_is_member_of_partnership": False,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_cryptoassets_net_income_line_26_1": 600,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input.schedule_l_cryptoassets_gross_income_line_16_1": 1000,
            },
            "output": {
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_reported_cryptoassets_gross_income_line_16_1": 1000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_reported_cryptoassets_net_income_line_26_1": 600,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_primary_net_business_income_line_27": 24600,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_net_business_income_line_164": 27000,
                "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#schedule_l_family_resource_qpip_qpp_earnings_line_40": 12345,
            },
        },
    ],
    "source": "Revenu Quebec Schedule L 2025 - Business Income\n"
    "\n"
    "T P -1. D. L-V (2025-12)\n"
    "T\n"
    "\n"
    "Business Income\n"
    "\n"
    "SCHEDULE\n"
    "\n"
    "L\n"
    "\n"
    "Net business income\n"
    "\n"
    "Enter your net business income on lines 22 through 26.1 (do not include "
    "income covered by lines 28 through 30). Enclose form TP-21.4.39-V,\n"
    "TP-80-V and/or TP-80.AP-V, or your financial statements, as applicable.\n"
    "Gross income\n"
    "Net income\n"
    "Business other than the types listed below\n"
    "12\n"
    "Farming\n"
    "13\n"
    "Fishing\n"
    "14\n"
    "Profession\n"
    "15\n"
    "Work remunerated on a commission basis\n"
    "16\n"
    "Cryptoassets. Complete form TP-21.4.39-V.\n"
    "If you are a member of a partnership:\n"
    "• Enter the amount from box 14-7 of your RL-15 slip on line 16.1.\n"
    "• Enter the amount from box 1-12 of the RL-15 slip on line 26.1.\n"
    "16.1\n"
    "Add lines 22 through 26.1.\n"
    "Income from a partnership allocated to a retiring partner or end-of-career "
    "allowance received\n"
    "by a physician who did not practise medicine in the year the allowance was "
    "received\n"
    "Income from a partnership of which you were a specified member\n"
    "Commissions you received from a business in the years following the end of "
    "its operations\n"
    "Add lines 27 through 30.\n"
    "Carry the result to line 164 of your return. \x08\n"
    "\n"
    "22\n"
    "\n"
    "+\n"
    "+\n"
    "+\n"
    "+\n"
    "\n"
    "24\n"
    "25\n"
    "26\n"
    "\n"
    "+ 26.1\n"
    "= 27\n"
    "+ 28\n"
    "+ 29\n"
    "+ 30\n"
    "Net business income = 34\n"
    "\n"
    "Insurable earnings (QPIP) and pensionable earnings (QPP) of a person "
    "responsible for a family-type resource\n"
    "or an intermediate resource (line 8 of form LM-53-V)\n"
    "\n"
    "Enclose this schedule with your return.\n"
    "If you want to keep a copy for your records, see the guide.\n"
    "\n"
    "23\n"
    "\n"
    "Y5L1 ZZ 89537649\n"
    "\n"
    "40\n"
    "\n"
    "L",
}

CITATION = "ca/policy/revenu-quebec/tp1-2025/schedule-l-business-income"


def material():
    d = copy.deepcopy(MATERIAL)
    return d["payload"], d["cases"], d["source"]


def certify(doc, cases, source):
    return e.certify_partnership_reporting(
        yaml.safe_dump(doc),
        source,
        CITATION,
        cases,
        principal_rules={r["name"]: r for r in doc["rules"]},
        formula_environment=s._constant_rule_environment(doc),
    )


def test_actual_sparse_worlds_execute_without_filler():
    doc, cases, source = material()
    before = copy.deepcopy((doc, cases))
    result = certify(doc, cases, source)
    assert result is not None
    assert {r.partner for r in result.witnesses} == {True, False}
    assert (doc, cases) == before
    branch = SimpleNamespace(
        start=result.start,
        end=result.end,
        text=source[result.start : result.end],
        required_principal_output=None,
    )
    assert result.unmatched([branch]) == ()
    assert result.unmatched([branch, branch]) == (branch,)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_true",
        "missing_false",
        "missing_selected",
        "missing_selector",
        "wrong_orientation",
        "missing_output",
        "wrong_output",
        "derived_input",
        "foreign_namespace",
        "inactive_filler",
        "wrong_owner",
        "wrong_unit",
        "wrong_period",
        "default",
        "duplicate_input",
        "duplicate_rule",
        "wrong_ast",
        "swapped_boxes",
        "missing_proof",
        "reused_proof",
        "wrong_proof_citation",
        "wrong_dates",
        "wrong_cad",
        "quoted",
        "duplicate_condition",
        "joined_condition",
        "duplicate_operation",
        "wrong_header",
        "wrong_revision",
        "boolean_money",
        "wrong_row",
    ],
)
def test_invalid_evidence_cannot_discharge_partnership(mutation):
    doc, cases, source = material()
    before = copy.deepcopy((doc, cases, source))

    def key(n):
        return (
            "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income#input." + n
        )

    if mutation in ("missing_true", "missing_false"):
        cases = [
            c
            for c in cases
            if c["input"][key(e.SELECTOR)] is (mutation == "missing_false")
        ]
    elif mutation in (
        "missing_selected",
        "missing_selector",
        "wrong_orientation",
        "inactive_filler",
        "boolean_money",
        "wrong_output",
        "missing_output",
        "derived_input",
        "foreign_namespace",
        "wrong_period",
        "wrong_row",
    ):
        for c in cases:
            if mutation == "missing_selected":
                c["input"].pop(
                    key(e.PARTNER[1] if c["input"][key(e.SELECTOR)] else e.DIRECT[1])
                )
            elif mutation == "missing_selector":
                del c["input"][key(e.SELECTOR)]
            elif mutation == "wrong_orientation":
                c["input"][key(e.SELECTOR)] = not c["input"][key(e.SELECTOR)]
            elif mutation == "inactive_filler":
                c["input"][
                    key(e.DIRECT[0] if c["input"][key(e.SELECTOR)] else e.PARTNER[0])
                ] = 0
            elif mutation == "boolean_money":
                c["input"][key(e.LM53_INPUT)] = False
            elif mutation == "wrong_output":
                c["output"][
                    next(k for k in c["output"] if k.endswith("#" + e.TOTAL))
                ] += 1
            elif mutation == "missing_output":
                del c["output"][
                    next(k for k in c["output"] if k.endswith("#" + e.TOTAL))
                ]
            elif mutation == "derived_input":
                c["input"].update(c["output"])
            elif mutation == "foreign_namespace":
                c["input"] = {
                    k.replace("ca:policies", "ca:foreign"): val
                    for k, val in c["input"].items()
                }
            elif mutation == "wrong_period":
                c["period"]["start"] = "2024-01-01"
            elif mutation == "wrong_row":
                c["tables"] = {"Person": [{"irrelevant": 99}]}
                c["output"] = {k: [v] for k, v in c["output"].items()}
    elif mutation == "wrong_owner":
        doc["inputs"][0]["entity"] = "Family"
    elif mutation == "wrong_unit":
        doc["inputs"][0]["unit"] = "USD"
    elif mutation == "default":
        doc["inputs"][0]["default"] = 0
    elif mutation == "duplicate_input":
        doc["inputs"].append(copy.deepcopy(doc["inputs"][0]))
    elif mutation == "duplicate_rule":
        doc["rules"].append(copy.deepcopy(doc["rules"][0]))
    elif mutation == "wrong_ast":
        doc["rules"][2]["versions"][0]["formula"] += " + 0"
    elif mutation == "swapped_boxes":
        doc["rules"][0]["versions"][0]["formula"] = doc["rules"][0]["versions"][0][
            "formula"
        ].replace(e.PARTNER[0], e.PARTNER[1])
    elif mutation == "missing_proof":
        doc["rules"][0]["metadata"]["proof"]["atoms"] = []
    elif mutation == "reused_proof":
        doc["rules"][1]["metadata"] = copy.deepcopy(doc["rules"][0]["metadata"])
    elif mutation == "wrong_proof_citation":
        doc["rules"][0]["metadata"]["proof"]["atoms"][0]["source"][
            "corpus_citation_path"
        ] = "ca/policy/other"
    elif mutation == "wrong_dates":
        doc["rules"][0]["versions"][0]["effective_to"] = "2026-12-31"
    elif mutation == "wrong_cad":
        doc["units"][0]["minor_units"] = 3
    elif mutation == "quoted":
        source = source.replace(e.CONDITION, '"' + e.CONDITION + '"')
    elif mutation == "duplicate_condition":
        source += "\n" + e.CONDITION
    elif mutation == "joined_condition":
        source = source.replace(
            e.CONDITION, e.CONDITION[:-1] + " and you are eligible:"
        )
    elif mutation == "duplicate_operation":
        source += "\n" + e.GROSS_QUOTE
    elif mutation == "wrong_header":
        source = source.replace(e.HEADER, e.HEADER.replace("2025", "2024"))
    elif mutation == "wrong_revision":
        source = source.replace("L-V (2025-12)", "L-V (2024-12)")
    assert (doc, cases, source) != before
    assert certify(doc, cases, source) is None


def issues(doc, cases, source):
    content = yaml.safe_dump(doc, sort_keys=False)
    profile = v._numeric_profile_for_citation_path(CITATION)
    bindings = s.collect_artifact_numeric_bindings(
        content,
        extract_named_scalars=v.extract_named_scalar_occurrences,
        imported_symbol_contents=(),
    )
    return s.analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            v.extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            v.extract_typed_numeric_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=v.extract_named_scalar_occurrences,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        artifact_numeric_values=tuple(x[1] for x in bindings),
        artifact_numeric_bindings=bindings,
    ).issues


def test_normal_source_gate_consumes_only_exact_partnership_condition(
    monkeypatch, request
):
    from axiom_encode import form_source_structure

    # The production tokenizer caches by text/profile; this test explicitly
    # changes the layout recognizer, so isolate both sides of that test seam.
    v._tokenize_numeric_occurrences_from_text.cache_clear()
    request.addfinalizer(v._tokenize_numeric_occurrences_from_text.cache_clear)
    monkeypatch.setattr(
        form_source_structure, "_schedule_l_printed_columns", lambda _text: None
    )
    doc, cases, source = material()
    good = issues(doc, cases, source)
    assert not any("paired" in x and "partnership" in x for x in good)
    assert any(
        "numeric" in x for x in good
    )  # Deliberately incomplete coordinate skeleton.
    cases = [
        c
        for c in cases
        if c["input"][next(k for k in c["input"] if k.endswith(e.SELECTOR))] is False
    ]
    bad = issues(doc, cases, source)
    assert any("partnership" in x and "paired" in x for x in bad)


def test_duplicate_yaml_key_refuses_before_name_projection():
    doc, cases, source = material()
    content = yaml.safe_dump(doc) + "\ninputs: []\n"
    assert (
        e.certify_partnership_reporting(
            content,
            source,
            CITATION,
            cases,
            principal_rules={r["name"]: r for r in doc["rules"]},
            formula_environment=s._constant_rule_environment(doc),
        )
        is None
    )


def test_actual_scalar_assertions_and_unbound_listed_row_refusal():
    doc, cases, source = material()
    assert certify(doc, cases, source) is not None
    for case in cases:
        case["output"] = {k: [v] for k, v in case["output"].items()}
    # No row identity was supplied: existing ordinary dependency evidence
    # deliberately refuses these lists instead of flattening them.
    assert certify(doc, cases, source) is None
    for case in cases:
        case["output"] = {k: [v[0] + 1] for k, v in case["output"].items()}
    assert certify(doc, cases, source) is None


def test_missing_gross_only_control_retains_partnership_obligation():
    doc, cases, source = material()
    cases = [
        case
        for case in cases
        if case["name"] != "schedule_l_gross_does_not_enter_net_total"
    ]
    assert len(cases) == 5
    assert certify(doc, cases, source) is None
    assert any(
        "paired" in issue and "partnership" in issue
        for issue in issues(doc, cases, source)
    )


@pytest.mark.parametrize(
    "mutation", ["payload_citation", "all_foreign", "wrong_row_proof"]
)
def test_source_owner_and_proof_identity_are_bound(mutation):
    doc, cases, source = material()
    before = copy.deepcopy((doc, cases, source))
    citation = CITATION
    if mutation == "payload_citation":
        doc["module"]["source_verification"]["corpus_citation_path"] = (
            "ca/policy/unrelated"
        )
    elif mutation == "all_foreign":
        citation = "ca/policy/unrelated"
        doc["module"]["source_verification"]["corpus_citation_path"] = citation
        for rule in doc["rules"]:
            for atom in rule["metadata"]["proof"]["atoms"]:
                atom["source"]["corpus_citation_path"] = citation
        for case in cases:
            for field in ("input", "output"):
                case[field] = {
                    key.replace(
                        "ca:policies/revenu-quebec/tp1-2025/schedule-l-business-income",
                        "ca:policies/unrelated",
                    ): value
                    for key, value in case[field].items()
                }
    else:
        # Same source/citation but gross ownership proof is the other row's box.
        doc["rules"][0]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
            e.NET_QUOTE
        )
    assert (doc, cases, source) != before
    assert (
        e.certify_partnership_reporting(
            yaml.safe_dump(doc),
            source,
            citation,
            cases,
            principal_rules={r["name"]: r for r in doc["rules"]},
            formula_environment=s._constant_rule_environment(doc),
        )
        is None
    )
