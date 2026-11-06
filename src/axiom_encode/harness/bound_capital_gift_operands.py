"""Source-role/AST binding for the closed 2025 T1170 diagnostic contract.

This is standalone static evidence, not source authentication, factual truth,
execution admission, or coverage. Names use a conservative reviewed vocabulary;
there is no general alpha-renaming claim. P113 ownership is limited to six exact
cited records. No generated candidate fixture is read by production code.
"""

from __future__ import annotations

import ast
import hashlib
import io
import re
import tokenize
from dataclasses import dataclass
from typing import Any

import yaml

from .bound_capital_gift_structure import (
    CapitalGiftSource,
    recognize_capital_gift_source,
)

PARENT = "ca/policy/cra/t1-2025/capital-gains-on-gifts"
GUIDE = "ca/policy/cra/t1-2025/federal-donations-and-gifts/p113"

# Role names and present descriptions corroborate the source/typed AST contract.
# Absent descriptions are permitted; present unknown/negated descriptions refuse.
_INPUTS = {
    "t1170_proceeds_of_disposition_column_2": (
        "SourceEntry",
        "Money",
        "Exact entered per-gift column amount; externally determined tax/factual value, not inferred from table totals or legal eligibility.",
    ),
    "t1170_adjusted_cost_base_column_3": (
        "SourceEntry",
        "Money",
        "Exact entered per-gift column amount; externally determined tax/factual value, not inferred from table totals or legal eligibility.",
    ),
    "t1170_outlays_and_expenses_column_4": (
        "SourceEntry",
        "Money",
        "Exact entered per-gift column amount; externally determined tax/factual value, not inferred from table totals or legal eligibility.",
    ),
    "t1170_eligible_amount_of_gift_column_6": (
        "SourceEntry",
        "Money",
        "Exact entered per-gift column amount; externally determined tax/factual value, not inferred from table totals or legal eligibility.",
    ),
    "t1170_year_acquired_column_1": (
        "SourceEntry",
        "Integer",
        "Printed acquisition year; not a cutoff used to infer gift eligibility.",
    ),
    "t1170_disposition_date": (
        "SourceEntry",
        "Date",
        "Actual gift/disposition date used for explicit2025 form applicability.",
    ),
    "t1170_qualified_donee": (
        "SourceEntry",
        "Boolean",
        "Externally established qualified-donee status, not aggregate gift eligibility.",
    ),
    "t1170_property_source_class": (
        "SourceEntry",
        "Text",
        "Explicit selected source category from six page1 classes, not generic eligible=True.",
    ),
    "t1170_shares_number": ("SourceEntry", "Integer", "Number field in shares table."),
    "t1170_fund_corporation_share_class": (
        "SourceEntry",
        "Text",
        "Name of fund/corporation and class of shares.",
    ),
    "t1170_bond_face_value": (
        "SourceEntry",
        "Money",
        "Printed face value, not proceeds.",
    ),
    "t1170_bond_maturity_date": (
        "SourceEntry",
        "Date",
        "Printed maturity date, not disposition date.",
    ),
    "t1170_bond_issuer": ("SourceEntry", "Text", "Printed issuer."),
    "t1170_land_address_or_legal_description": (
        "SourceEntry",
        "Text",
        "Printed land identification; no title/ownership inference.",
    ),
    "t1170_exchange_route_used": (
        "SourceEntry",
        "Boolean",
        "Whether source exchange-extension route is invoked for this transaction.",
    ),
    "t1170_exchange_right_at_issue": (
        "SourceEntry",
        "Boolean",
        "P113 exchange condition in original shares at issuance.",
    ),
    "t1170_exchange_right_at_disposition": (
        "SourceEntry",
        "Boolean",
        "P113 exchange condition still present at disposition.",
    ),
    "t1170_securities_only_exchange_consideration": (
        "SourceEntry",
        "Boolean",
        "P113 securities-only consideration condition.",
    ),
    "t1170_exchange_date": (
        "SourceEntry",
        "Date",
        "Actual exchange date, not period_start-derived.",
    ),
    "completing_t1170": (
        "Person",
        "Boolean",
        "Explicit use of this form, not completed filing.",
    ),
    "needs_t1170_additional_space": (
        "Person",
        "Boolean",
        "Actual additional-space need, not inferred from number of rows.",
    ),
    "t1170_fts_schedule3_additional_gain_applies": (
        "Person",
        "Boolean",
        "Source FTS additional-gain reporting circumstance; does not calculate threshold.",
    ),
    "t1170_completed_schedule3_fts_gain_line68231": (
        "Person",
        "Money",
        "Completed upstream Schedule3 deemed gain already included there.",
    ),
    "t1170_nonprescribed_partnership_exchange_applies": (
        "Person",
        "Boolean",
        "Specific source branch: exchanged partnership interest other than prescribed interests.",
    ),
    "t1170_advantage_in_respect_of_gift": (
        "SourceEntry",
        "Money",
        "Explicit advantage received for this new diagnostic row. Does not replace entered eligible amount or infer gift eligibility; new values are labelled new facts.",
    ),
    "t1170_property_instrument_kind": (
        "SourceEntry",
        "Text",
        "Explicit instrument discriminator for designated-exchange source class, not inferred from amount or owner.",
    ),
    "t1170_ecological_donee_not_private_foundation": (
        "SourceEntry",
        "Boolean",
        "Explicit P113 ecological-donee restriction, independent of generic qualified_donee. Only applicable ecological rows supply it.",
    ),
    "t1170_exchanged_asset_kind": (
        "SourceEntry",
        "Text",
        "Explicit kind of the property surrendered in this selected prior exchange; corporate_capital_stock_share identifies capital-stock shares of a corporation, never inferred from the donated security.",
    ),
    "t1170_gift_is_of_securities_received_in_exchange": (
        "SourceEntry",
        "Boolean",
        "Explicit selected-transaction fact that the donated securities in this row are the securities received in the prior exchange; no ownership or factual-truth inference.",
    ),
    "t1170_land_interest_subtype": (
        "SourceEntry",
        "Text",
        "Explicit newly stated ecological property/donee attribute: t1170_land_interest_subtype. No inference from the class label.",
    ),
    "t1170_land_location_jurisdiction": (
        "SourceEntry",
        "Text",
        "Actual jurisdiction of land; Quebec required for both servitude forms, not donor residence.",
    ),
    "t1170_personal_servitude_term_years": (
        "SourceEntry",
        "Integer",
        "Actual term of personal-servitude rights; reached only for Quebec personal subtype.",
    ),
    "t1170_ecological_land_certified_by_eccc": (
        "SourceEntry",
        "Boolean",
        "Minister or designated person has actually certified ecological sensitivity for this land, not caller self-assessment.",
    ),
    "t1170_ecological_donee_category": (
        "SourceEntry",
        "Text",
        "Explicit newly stated ecological property/donee attribute: t1170_ecological_donee_category. No inference from the class label.",
    ),
    "t1170_ecological_donee_approved_in_respect_of_gift": (
        "SourceEntry",
        "Boolean",
        "Minister/designate approval for this gift; required for municipality/public body/registered charity branch, not government branch.",
    ),
    "t1170_prior_exchange_capital_gain": (
        "SourceEntry",
        "Money",
        "Actual capital gain on this earlier corporation-share exchange, separately determined from the later donated-property column5. Explicit hypothetical transaction amount, no gift-ratio inference.",
    ),
    "t1170_partnership_exchange_gain_otherwise_determined": (
        "SourceEntry",
        "Money",
        "Capital gain on this exchanged non-prescribed interest determined before this special calculation; not ordinary gift gain.",
    ),
    "t1170_exchanged_partnership_interest_cost": (
        "SourceEntry",
        "Money",
        "Cost to this donor of this exchanged interest.",
    ),
    "t1170_donor_partnership_capital_contributions": (
        "SourceEntry",
        "Money",
        "Contributions to partnership capital by this donor for this interest; source-attributed amount, not arbitrary all partnership cashflows.",
    ),
    "t1170_partnership_adjusted_basis_without_distributions": (
        "SourceEntry",
        "Money",
        "Adjusted cost base of this interest determined without distributions of partnership profits or capital; externally determined basis fact.",
    ),
    "t1170_partnership_exchange_right_at_issue": (
        "SourceEntry",
        "Boolean",
        "Actual exchange right in the selected non-prescribed partnership interest at issuance; partnership-specific counterfactual condition, not a corporation-share fact.",
    ),
    "t1170_partnership_exchange_right_at_disposition": (
        "SourceEntry",
        "Boolean",
        "Actual exchange right in the selected non-prescribed partnership interest at disposition; partnership-specific counterfactual condition, not a corporation-share fact.",
    ),
}

_GUIDE_RECORDS = {
    "guide_property_scope": "You may be entitled to an inclusion rate of zero on any capital gain realized on such gifts if you donate:\n\na share of the capital stock of a mutual fund corporation\na unit of a mutual fund trust\nan interest in a related segregated fund trust\na prescribed debt obligation\na share, debt obligation, or right listed on a designated stock exchange\ncertified ecologically sensitive land, including a covenant or an easement to which land is subject or, in the case of land in Quebec, a personal servitude where certain conditions are met or real servitude gifted to a qualified donee other than a private foundation. For more details, see&nbsp; Ecological gifts\n\n",
    "guide_fts_ratio": "If you donate property to a qualified donee that is, at the time of the donation, included in a flow-through share (FTS) class of property, you are deemed to have a capital gain from the disposition of another capital property. In addition to any capital gain that would otherwise be subject to the zero inclusion rate discussed earlier in this section, you are deemed to have a capital gain from the disposition of another capital property equal to the whichever of the following amounts is less :\n\nthe amount of your exemption threshold, at the time of the donation, in respect of the FTS class of property\nthe total capital gains from the actual disposition\n\nIf there is no advantage received in respect of the gift, the full amount of the capital gain is eligible for the inclusion rate of zero. However, if you receive an advantage in respect of the gift, only a portion of the capital gain is eligible for the inclusion rate of zero. The rest is subject to the inclusion rate calculated on&nbsp; Schedule 3, Capital Gains or Losses .\nThe amount subject to the inclusion rate of zero is calculated using the following formula:\nA × (B ÷ C)\nWhere A = the capital gain B = the eligible amount of the gift C = the proceeds of disposition\n",
    "guide_partnership": "If the exchanged property is a partnership interest (other than prescribed interests in a partnership), the capital gain will generally be whichever of the following amounts is less :\n\nthe capital gain otherwise determined\nthe amount, if any, by which the cost to the donor of the exchanged interests, plus any contributions to partnership capital by the donor, exceeds the ACB of those interests (determined without reference to distributions of partnership profits or capital)",
    "guide_exchange": "The inclusion rate of zero is extended to any capital gain realized on the exchange of shares of the capital stock of a corporation for securities listed in the first five bullets above that are then donated if the capital stock shares meet all of the following conditions:\n\nat the time they were issued and at the time of disposition, the shares included a condition allowing the holder to exchange them for the securities\nthe securities are the only consideration received from the exchange\nthe securities are donated within 30 days of the exchange\n\nIf the exchanged property is a partnership interest (other than prescribed interests in a partnership), the capital gain will generally be whichever of the following amounts is less :\n\nthe capital gain otherwise determined\nthe amount, if any, by which the cost to the donor of the exchanged interests, plus any contributions to partnership capital by the donor, exceeds the ACB of those interests (determined without reference to distributions of partnership profits or capital)\n\n",
    "guide_ecological": "You may be able to claim a tax credit based on the eligible amount of a gift of ecologically sensitive land, including a covenant, an easement, or in the case of land in Quebec, a real servitude or a personal servitude (the rights to which the land is subject and which has a term of not less than 100 years)&nbsp;you made to any of the following:&nbsp;\n\nthe Government&nbsp;of Canada, a province or a territory\na municipality in Canada, or a municipal or public body performing a function of government in Canada, that is approved by the Minister of Environment and Climate Change Canada (ECCC) or a person designated by that Minister in respect of the gift\na registered charity approved by the Minister of ECCC or the designated person in respect of the gift",
    "guide_ecological_restrictions": "A&nbsp;gift of ecologically sensitive land cannot be made to a private foundation.\nThe Minister of ECCC , or a person designated by that Minister, has to certify that the land is ecologically sensitive land, the conservation and protection of which is important to the preservation of Canada's environmental heritage. The Minister also determines the FMV of the gift.",
}


@dataclass(frozen=True)
class CapitalGiftRule:
    name: str
    kind: str
    dtype: str | None
    entity: str | None
    expression: tuple | None
    dependencies: tuple[str, ...]
    proof_roles: tuple[str, ...]


@dataclass(frozen=True)
class CapitalGiftBinding:
    content_sha256: str
    source: CapitalGiftSource
    guide_sha256: str
    guide_spans: tuple[tuple[str, tuple[int, int]], ...]
    inputs: tuple[tuple[str, str, str], ...]
    rules: tuple[CapitalGiftRule, ...]

    def bound(self, content: str, source: str, guide: str) -> bool:
        return bind_capital_gift(content, source, guide) == self


@dataclass(frozen=True)
class _Rule:
    name: str
    formula: str | None
    roles: tuple[str, ...]
    dtype: str = "Money"
    entity: str | None = "SourceEntry"
    kind: str = "derived"
    proof_kind: str = "formula"
    value: str | None = None
    parameter_period: bool = False


def _contracts() -> tuple[_Rule, ...]:
    """Build the reviewed semantic graph from printed column and branch roles.

    This intentionally spells out ordered operations, scalar-vs-Judgment branch
    semantics, category mappings and source owners, independently of input YAML.
    """
    rules: list[_Rule] = []
    tables = ("shares_table", "bonds_table", "land_table")
    prop = ("property_scope", "guide_property_scope")
    gain = "federal_2025_t1170_gift_capital_gain_column_5"
    zero = "federal_2025_t1170_gain_eligible_zero_inclusion_column_7"
    other = "federal_2025_t1170_gain_subject_to_50_percent_inclusion_column_8"
    rate = "federal_2025_t1170_taxable_inclusion_rate"

    def add(name, formula, roles, **kwargs):
        rules.append(_Rule(name, formula, tuple(roles), **kwargs))

    def boolean(name, condition, roles):
        add(
            name, f"if {condition}:\n    true\nelse:\n    false", roles, dtype="Boolean"
        )

    def guarded(selector, expression):
        return f"match {selector}:\n    true => {expression}"

    add(rate, "0.50", ("gift_inclusion",), dtype="Rate", entity=None, kind="parameter")
    add(
        gain,
        guarded(
            "t1170_gift_within_form_scope",
            "t1170_proceeds_of_disposition_column_2 - t1170_adjusted_cost_base_column_3 - t1170_outlays_and_expenses_column_4",
        ),
        prop + tables,
    )
    add(
        zero,
        f"{gain} * t1170_eligible_amount_of_gift_column_6 / t1170_proceeds_of_disposition_column_2",
        tables + ("guide_fts_ratio",),
    )
    add(other, f"{gain} - {zero}", tables + ("guide_fts_ratio",))
    add(
        "federal_2025_t1170_taxable_capital_gain_from_gift",
        f"{other} * {rate}",
        ("gift_inclusion",),
    )
    add(
        "t1170_entered_proceeds_column_2",
        "t1170_proceeds_of_disposition_column_2",
        tables,
    )
    for name, operand, dtype, roles in (
        ("year_acquired_column_1", "year_acquired_column_1", "Integer", tables),
        ("shares_number", "shares_number", "Integer", tables[:1]),
        (
            "fund_corporation_share_class",
            "fund_corporation_share_class",
            "Text",
            tables[:1],
        ),
        ("bond_face_value", "bond_face_value", "Money", tables[1:2]),
        ("bond_maturity_date", "bond_maturity_date", "Date", tables[1:2]),
        ("bond_issuer", "bond_issuer", "Text", tables[1:2]),
        (
            "land_address_or_legal_description",
            "land_address_or_legal_description",
            "Text",
            tables[2:],
        ),
    ):
        add(
            f"t1170_reported_{name}",
            f"t1170_{operand}",
            roles,
            dtype=dtype,
            proof_kind="formula" if dtype == "Money" else "definition",
        )

    # Table memberships are data associations, not inferred legal ownership.
    categories = (
        (
            "shares_units",
            "shares_table",
            "federal_2025_t1170_shares_and_units_schedule_3_line_13200",
            "68220",
            "68230",
        ),
        (
            "bonds_other",
            "bonds_table",
            "federal_2025_t1170_bonds_debentures_and_other_properties_schedule_3_line_15300",
            "68240",
            "68250",
        ),
        (
            "ecological_land",
            "land_table",
            "federal_2025_t1170_ecological_land_schedule_3_line_13800",
            "68251",
            "68252",
        ),
    )
    for cat, role, destination, proceeds_field, zero_field in categories:
        relation = f"t1170_{cat}_row_for_return"
        add(
            relation,
            None,
            (role,),
            kind="data_relation",
            entity=None,
            dtype=None,
            proof_kind="definition",
        )
        boolean(
            f"t1170_{cat}_row_within_table_scope",
            f"t1170_gift_within_form_scope == true and t1170_{cat}_classification_matches == true",
            ("property_scope", role),
        )
        for suffix, operand, export in (
            (
                "proceeds",
                "t1170_proceeds_of_disposition_column_2",
                f"t1170_{cat}_proceeds_line_{proceeds_field}",
            ),
            ("zero_inclusion", zero, f"t1170_{cat}_zero_inclusion_line_{zero_field}"),
            ("other_gain", other, destination),
        ):
            provider = f"t1170_{cat}_table_{suffix}"
            add(
                provider,
                guarded(f"t1170_{cat}_row_within_table_scope", operand),
                ("property_scope", role),
            )
            add(export, f"sum({relation}.{provider})", (role,), entity="Person")
    add(
        "federal_2025_t1170_total_gain_eligible_zero_inclusion",
        "t1170_shares_units_zero_inclusion_line_68230 + t1170_bonds_other_zero_inclusion_line_68250",
        tables[:2],
        entity="Person",
    )
    add(
        "federal_2025_t1170_total_gain_subject_to_50_percent_inclusion",
        f"{categories[0][2]} + {categories[1][2]}",
        tables[:2],
        entity="Person",
    )
    add(
        "t1170_fts_gain_for_t691_part1_line25",
        guarded(
            "t1170_fts_schedule3_additional_gain_applies",
            "t1170_completed_schedule3_fts_gain_line68231",
        ),
        ("flow_through_route",),
        entity="Person",
    )
    add(
        "t1170_partnership_gain_for_schedule3_line17400",
        guarded(
            "t1170_nonprescribed_partnership_exchange_applies",
            "sum(t1170_partnership_exchange_for_return.t1170_partnership_special_capital_gain_per_entry)",
        ),
        ("partnership", "guide_partnership"),
        entity="Person",
    )
    for name, operand, roles in (
        ("enclosure_required", "completing_t1170", ("document_duties",)),
        (
            "additional_sheet_required",
            "needs_t1170_additional_space",
            ("document_duties",),
        ),
        (
            "fts_t691_reporting_required",
            "t1170_fts_schedule3_additional_gain_applies",
            ("flow_through_route",),
        ),
        (
            "partnership_special_calculation_required",
            "t1170_nonprescribed_partnership_exchange_applies",
            ("partnership",),
        ),
        (
            "partnership_schedule3_reporting_required",
            "t1170_nonprescribed_partnership_exchange_applies",
            ("partnership",),
        ),
    ):
        add(
            f"t1170_{name}",
            operand,
            roles,
            entity="Person",
            dtype="Judgment",
            proof_kind="definition",
        )
    add(
        "t1170_no_advantage_in_respect_of_gift",
        "t1170_advantage_in_respect_of_gift == 0",
        ("gift_inclusion",),
        dtype="Judgment",
        proof_kind="definition",
    )
    add(
        "t1170_form_year",
        "2025",
        ("resolver_title",),
        dtype="Integer",
        entity=None,
        kind="parameter",
    )
    for side, date in (("start", "2025-01-01"), ("end", "2025-12-31")):
        add(
            f"t1170_form_year_{side}",
            None,
            ("resolver_title",),
            dtype="Date",
            entity=None,
            kind="parameter",
            proof_kind="definition",
            value=date,
        )
    add(
        "t1170_exchange_gift_deadline_days",
        "30",
        ("guide_exchange",),
        dtype="Integer",
        entity=None,
        kind="parameter",
    )
    add(
        "t1170_minimum_quebec_personal_servitude_term_years",
        "100",
        ("guide_ecological",),
        dtype="Integer",
        entity=None,
        kind="parameter",
        parameter_period=True,
    )
    for cat, role, alternatives in (
        (
            "shares_units",
            "shares_table",
            (
                ("designated_exchange_share_debt_or_right", "share"),
                ("designated_exchange_share_debt_or_right", "right"),
                ("mutual_fund_corporation_share", "share"),
                ("mutual_fund_trust_unit", "fund_unit"),
                ("related_segregated_fund_trust_interest", "fund_interest"),
            ),
        ),
        (
            "bonds_other",
            "bonds_table",
            (
                ("designated_exchange_share_debt_or_right", "debt_obligation"),
                ("prescribed_debt_obligation", "debt_obligation"),
            ),
        ),
        (
            "ecological_land",
            "land_table",
            (("certified_ecologically_sensitive_land", "land_interest"),),
        ),
    ):
        condition = " or ".join(
            f'(t1170_property_source_class == "{cls}" and t1170_property_instrument_kind == "{instrument}")'
            for cls, instrument in alternatives
        )
        boolean(f"t1170_{cat}_classification_matches", condition, prop + (role,))
    boolean(
        "t1170_gift_within_form_scope",
        "t1170_qualified_donee and t1170_disposition_date >= t1170_form_year_start[t1170_form_year] and t1170_disposition_date <= t1170_form_year_end[t1170_form_year] and (t1170_shares_units_classification_matches == true or t1170_bonds_other_classification_matches == true or (t1170_ecological_land_classification_matches == true and t1170_ecological_source_conditions_met == true))",
        ("resolver_title",)
        + prop
        + ("guide_ecological", "guide_ecological_restrictions"),
    )
    boolean(
        "t1170_ecological_interest_subtype_matches",
        't1170_land_interest_subtype == "land" or t1170_land_interest_subtype == "covenant" or t1170_land_interest_subtype == "easement" or (t1170_land_location_jurisdiction == "Quebec" and (t1170_land_interest_subtype == "real_servitude" or (t1170_land_interest_subtype == "personal_servitude" and t1170_personal_servitude_term_years >= t1170_minimum_quebec_personal_servitude_term_years)))',
        ("guide_ecological",),
    )
    boolean(
        "t1170_ecological_donee_category_matches",
        't1170_ecological_donee_category == "canada_government" or t1170_ecological_donee_category == "provincial_government" or t1170_ecological_donee_category == "territorial_government" or ((t1170_ecological_donee_category == "canadian_municipality" or t1170_ecological_donee_category == "canadian_municipal_or_public_government_body" or t1170_ecological_donee_category == "registered_charity") and t1170_ecological_donee_approved_in_respect_of_gift)',
        ("guide_ecological",),
    )
    boolean(
        "t1170_ecological_source_conditions_met",
        "t1170_ecological_interest_subtype_matches == true and t1170_ecological_land_certified_by_eccc and t1170_ecological_donee_not_private_foundation and t1170_ecological_donee_category_matches == true",
        ("guide_ecological", "guide_ecological_restrictions"),
    )

    def exchange_condition(asset, rights):
        return f't1170_exchange_route_used and t1170_exchanged_asset_kind == "{asset}" and t1170_gift_is_of_securities_received_in_exchange and t1170_gift_within_form_scope == true and (t1170_shares_units_classification_matches == true or t1170_bonds_other_classification_matches == true) and {rights}_at_issue and {rights}_at_disposition and t1170_securities_only_exchange_consideration and days_between(t1170_exchange_date, t1170_disposition_date) >= 0 and days_between(t1170_exchange_date, t1170_disposition_date) <= t1170_exchange_gift_deadline_days'

    exchange = exchange_condition(
        "corporate_capital_stock_share", "t1170_exchange_right"
    )
    add(
        "t1170_exchange_extension_conditions_met",
        exchange,
        ("guide_exchange",) + prop,
        dtype="Judgment",
    )
    exchange_roles = ("prior_exchange", "guide_exchange")
    boolean(
        "t1170_prior_exchange_within_zero_inclusion_scope", exchange, exchange_roles
    )
    add(
        "t1170_prior_exchange_gain_at_zero_inclusion",
        guarded(
            "t1170_prior_exchange_within_zero_inclusion_scope",
            "t1170_prior_exchange_capital_gain",
        ),
        exchange_roles,
    )
    add(
        "t1170_prior_exchange_taxable_capital_gain",
        guarded("t1170_prior_exchange_within_zero_inclusion_scope", "0"),
        exchange_roles,
    )
    boolean(
        "t1170_partnership_exchange_source_conditions_met",
        exchange_condition(
            "nonprescribed_partnership_interest", "t1170_partnership_exchange_right"
        ),
        ("guide_partnership", "guide_exchange", "partnership"),
    )
    add(
        "t1170_partnership_special_capital_gain_per_entry",
        guarded(
            "t1170_partnership_exchange_source_conditions_met",
            "min(t1170_partnership_exchange_gain_otherwise_determined, max(t1170_exchanged_partnership_interest_cost + t1170_donor_partnership_capital_contributions - t1170_partnership_adjusted_basis_without_distributions, 0))",
        ),
        ("guide_partnership", "partnership"),
    )
    add(
        "t1170_partnership_exchange_for_return",
        None,
        ("partnership",),
        kind="data_relation",
        entity=None,
        dtype=None,
        proof_kind="definition",
    )
    return tuple(rules)


def _node(node: ast.AST) -> tuple:
    """Closed ordered scalar/condition grammar; never evaluate source code."""
    if isinstance(node, ast.Name):
        return ("name", node.id)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float, str):
        return ("literal", type(node.value).__name__, node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
    ):
        return (type(node.op).__name__, _node(node.left), _node(node.right))
    if isinstance(node, ast.BoolOp) and type(node.op) in (ast.And, ast.Or):
        return (type(node.op).__name__, *(_node(v) for v in node.values))
    if (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and type(node.ops[0]) in (ast.Eq, ast.GtE, ast.LtE)
    ):
        return (
            type(node.ops[0]).__name__,
            _node(node.left),
            _node(node.comparators[0]),
        )
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return ("provider", node.value.id, node.attr)
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and isinstance(node.slice, ast.Name)
    ):
        return ("indexed", node.value.id, node.slice.id)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and not node.keywords
        and node.func.id in ("min", "max", "sum", "days_between")
    ):
        expected = 1 if node.func.id == "sum" else 2
        if len(node.args) == expected:
            return ("call", node.func.id, *(_node(a) for a in node.args))
    raise ValueError("unsupported formula node")


def _expression(formula: Any) -> tuple:
    if not isinstance(formula, str) or len(formula) > 12_000:
        raise ValueError("formula must be bounded text")
    if any(
        t.type == tokenize.COMMENT
        for t in tokenize.generate_tokens(io.StringIO(formula).readline)
    ):
        raise ValueError("formula comments not in closed grammar")
    if formula.startswith("match "):
        match = re.fullmatch(
            r"match ([A-Za-z_][A-Za-z_0-9]*):\n +true => ([^\n]+)\n?", formula
        )
        if not match:
            raise ValueError("only one true arm with no wildcard is supported")
        return ("match_true", match[1], _node(ast.parse(match[2], mode="eval").body))
    parsed = ast.parse(formula, mode="exec")
    if len(parsed.body) != 1:
        raise ValueError("multiple formula statements")
    body = parsed.body[0]
    if isinstance(body, ast.Expr):
        return _node(body.value)
    if (
        isinstance(body, ast.If)
        and len(body.body) == len(body.orelse) == 1
        and isinstance(body.body[0], ast.Expr)
        and isinstance(body.orelse[0], ast.Expr)
    ):
        return (
            "if",
            _node(body.test),
            _node(body.body[0].value),
            _node(body.orelse[0].value),
        )
    raise ValueError("unsupported formula statement")


def _dependencies(expression: tuple | None) -> tuple[str, ...]:
    found: set[str] = set()

    def visit(node):
        if not isinstance(node, tuple):
            return
        if node[0] == "name" and node[1] not in ("true", "false"):
            found.add(node[1])
        elif node[0] in ("provider", "indexed"):
            found.update(node[1:])
        elif node[0] == "match_true":
            found.add(node[1])
        for child in node[1:]:
            visit(child)

    visit(expression)
    return tuple(sorted(found))


def _exact(actual: Any, expected: Any) -> bool:
    """Python equality alone conflates bool/int and int/float."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return len(actual) == len(expected) and all(
            any(_exact(k, key) and _exact(actual[k], value) for k in actual)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact(a, b) for a, b in zip(actual, expected, strict=True)
        )
    return actual == expected


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError("duplicate mapping key")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _load(content: str) -> dict:
    if not isinstance(content, str) or len(content) > 500_000:
        raise ValueError("bounded policy text required")
    if any(
        isinstance(t, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken))
        for t in yaml.scan(content)
    ):
        raise ValueError("policy aliases are outside the closed contract")
    result = yaml.load(content, Loader=_UniqueLoader)
    if not isinstance(result, dict):
        raise ValueError("policy object required")
    return result


def _sources(source: CapitalGiftSource, guide: str):
    # Identity pin is edition restriction, not sufficient proof of a consumer.
    # Every declaration and formula below is independently bound to source roles.
    if (
        not isinstance(guide, str)
        or hashlib.sha256(guide.encode()).hexdigest()
        != "bdc817872594a6da5785539d68370aa62f9c8dcd71e6d81d234958d1c8f417f7"
    ):
        raise ValueError("unreviewed P113 resolver body")
    roles = {r.role: (PARENT, source.source[slice(*r.span)]) for r in source.records}
    spans = []
    for role, text in _GUIDE_RECORDS.items():
        if guide.count(text) != 1:
            raise ValueError("missing or duplicated dependency record")
        left = guide.index(text)
        spans.append((role, (left, left + len(text))))
        roles[role] = (GUIDE, text)
    return roles, tuple(spans)


def _proof(rule: dict, contract: _Rule, roles: dict, source: str, guide: str) -> None:
    metadata = rule.get("metadata")
    if not isinstance(metadata, dict) or set(metadata) != {"proof"}:
        raise ValueError("exact proof metadata required")
    proof = metadata["proof"]
    if (
        not isinstance(proof, dict)
        or set(proof) != {"atoms"}
        or not isinstance(proof["atoms"], list)
    ):
        raise ValueError("proof atoms required")
    path = (
        "data_relation"
        if contract.kind == "data_relation"
        else "versions[0].values"
        if contract.value is not None
        else "versions[0].formula"
    )
    required = {(path, contract.proof_kind, role) for role in contract.roles}
    if contract.kind != "data_relation":
        required.update(
            (f"versions[0].{endpoint}", "effective_period", "resolver_title")
            for endpoint in ("effective_from", "effective_to")
        )
    seen = set()
    for atom in proof["atoms"]:
        if (
            not isinstance(atom, dict)
            or set(atom) != {"path", "kind", "source"}
            or type(atom["path"]) is not str
            or type(atom["kind"]) is not str
        ):
            raise ValueError("malformed proof atom")
        record = atom["source"]
        if (
            not isinstance(record, dict)
            or set(record) != {"corpus_citation_path", "excerpt"}
            or not all(type(v) is str for v in record.values())
        ):
            raise ValueError("malformed proof source")
        citation, excerpt = record["corpus_citation_path"], record["excerpt"]
        body = source if citation == PARENT else guide if citation == GUIDE else ""
        if not excerpt or body.count(excerpt) != 1:
            raise ValueError("proof excerpt not unique in exact source")
        # Only the record's one literal trailing LF may be omitted/retained.
        # No arbitrary whitespace stripping or partial-clause proof recognition.
        matches = [
            role
            for role, (owner, text) in roles.items()
            if owner == citation and excerpt in (text, text + "\n")
        ]
        selected = [
            (atom["path"], atom["kind"], role)
            for role in matches
            if (atom["path"], atom["kind"], role) in required
        ]
        if len(selected) != 1 or selected[0] in seen:
            raise ValueError("proof has wrong role/path/kind or duplicates")
        seen.add(selected[0])
    if seen != required:
        raise ValueError("incomplete proof role allocation")


def _input(item: dict) -> tuple[str, str, str]:
    if (
        not isinstance(item, dict)
        or type(item.get("name")) is not str
        or item["name"] not in _INPUTS
    ):
        raise ValueError("unknown factual role")
    name = item["name"]
    entity, dtype, description = _INPUTS[name]
    expected = {"name": name, "entity": entity, "dtype": dtype, "period": "Year"}
    if dtype == "Money":
        expected["unit"] = "CAD"
    if "description" in item:
        expected["description"] = description
    if not _exact(item, expected):
        raise ValueError("factual role type/entity/unit/description mismatch")
    return name, entity, dtype


def _rule(
    item: dict, contract: _Rule, roles: dict, source: str, guide: str
) -> CapitalGiftRule:
    expected = {
        "name": contract.name,
        "kind": contract.kind,
        "source": roles["resolver_title"][1],
    }
    if contract.kind == "data_relation":
        expected["data_relation"] = {"arity": 2, "arguments": ["SourceEntry", "Person"]}
        expression = None
    else:
        expected["dtype"] = contract.dtype
        if contract.entity is not None:
            expected["entity"] = contract.entity
        if contract.kind == "derived" or contract.parameter_period:
            expected["period"] = "Year"
        if contract.dtype == "Money":
            expected["unit"] = "CAD"
        versions = item.get("versions")
        if (
            not isinstance(versions, list)
            or len(versions) != 1
            or not isinstance(versions[0], dict)
        ):
            raise ValueError("one complete 2025 version required")
        version = versions[0]
        expected_version = {
            "effective_from": "2025-01-01",
            "effective_to": "2025-12-31",
        }
        if contract.value is not None:
            expected["indexed_by"] = "t1170_form_year"
            expected_version["values"] = {2025: contract.value}
            expression = ("date", 2025, contract.value, "t1170_form_year")
        else:
            expression = _expression(version.get("formula"))
            if expression != _expression(contract.formula):
                raise ValueError("source-role ordered AST mismatch")
            # Formula syntax may differ in whitespace/quotes only after AST equality.
            expected_version["formula"] = version["formula"]
        if not _exact(version, expected_version):
            raise ValueError("period/index/value/version metadata mismatch")
        expected["versions"] = versions
    metadata = item.get("metadata")
    expected["metadata"] = metadata
    if not _exact(item, expected):
        raise ValueError("declaration schema/type/entity/relation mismatch")
    _proof(item, contract, roles, source, guide)
    dependencies = (
        ("t1170_form_year",)
        if contract.value is not None
        else _dependencies(expression)
    )
    return CapitalGiftRule(
        contract.name,
        contract.kind,
        contract.dtype,
        contract.entity,
        expression,
        dependencies,
        contract.roles,
    )


def bind_capital_gift(
    content: str, source: str, guide: str
) -> CapitalGiftBinding | None:
    """Bind the whole reviewed source/typed declaration graph, or refuse.

    Caller-supplied source text is not authenticated here. A future private
    execution context must independently resolve/authenticate both source bodies.
    The returned frozen record has no source-completeness or execution authority.
    """
    try:
        structure = recognize_capital_gift_source(source, PARENT)
        if structure is None:
            return None
        roles, spans = _sources(structure, guide)
        payload = _load(content)
        if set(payload) != {"format", "module", "imports", "inputs", "rules", "units"}:
            return None
        if (
            payload["format"] != "rulespec/v1"
            or not _exact(payload["imports"], [])
            or not _exact(
                payload["units"],
                [{"name": "CAD", "kind": "currency", "minor_units": 2}],
            )
        ):
            return None
        module = payload["module"]
        if (
            not isinstance(module, dict)
            or set(module) != {"summary", "proof_validation", "source_verification"}
            or type(module["summary"]) is not str
        ):
            return None
        if not _exact(module["proof_validation"], {"required": True}) or not _exact(
            module["source_verification"],
            {"corpus_citation_path": PARENT, "source_sha256": structure.source_sha256},
        ):
            return None
        if not isinstance(payload["inputs"], list) or not isinstance(
            payload["rules"], list
        ):
            return None
        inputs = tuple(_input(i) for i in payload["inputs"])
        if len(inputs) != len(_INPUTS) or {i[0] for i in inputs} != set(_INPUTS):
            return None
        contracts = {r.name: r for r in _contracts()}
        rules = []
        for item in payload["rules"]:
            if (
                not isinstance(item, dict)
                or type(item.get("name")) is not str
                or item["name"] not in contracts
            ):
                return None
            rules.append(_rule(item, contracts[item["name"]], roles, source, guide))
        if len(rules) != len(contracts) or {r.name for r in rules} != set(contracts):
            return None
        names = set(contracts) | set(_INPUTS)
        if any(not set(rule.dependencies) <= names for rule in rules):
            return None
        # Contract construction, not caller aliases, fixes the acyclic graph.
        return CapitalGiftBinding(
            hashlib.sha256(content.encode()).hexdigest(),
            structure,
            hashlib.sha256(guide.encode()).hexdigest(),
            spans,
            tuple(sorted(inputs)),
            tuple(sorted(rules, key=lambda r: r.name)),
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        SyntaxError,
        RecursionError,
        yaml.YAMLError,
        tokenize.TokenError,
    ):
        return None
