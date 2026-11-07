"""Complete source-role witnesses for privately observed T1170 execution.

This module supplies no caller-created certificate. Source ownership and actual
execution completeness are separate, and unobserved owned obligations remain.
"""

import weakref
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal

from .capital_gift_execution import PREFIX
from .capital_gift_provider_refusal import TARGETS

_GIFT = "federal_2025_t1170_gift_capital_gain_column_5"
_PRIOR = "t1170_prior_exchange_gain_at_zero_inclusion"
_PARTNERSHIP = "t1170_partnership_special_capital_gain_per_entry"


def source_role_witnesses(binding, observed):
    """Derive roles from validated actual observations, never names alone.

    This internal helper is not a capture grant; the factory must obtain its
    arguments from the registered fresh materialized context and reobserve them.
    """
    records = observed.positive.records
    queried = {name.removeprefix(PREFIX) for r in records for name, _ in r.outputs}
    if queried != {r.name for r in binding.rules if r.kind == "derived"}:
        raise ValueError("every bound derived output must be actually queried")
    positive = {r.case_key: r for r in records}
    negative = {r.case_key: r for r in observed.refusals}
    roles = {}

    def require(role, keys):
        keys = frozenset(keys)
        if not keys:
            raise ValueError("missing source-owned role: " + role)
        roles[role] = keys

    def direct(principal, fact, predicate=lambda _: True):
        return {
            (a, b)
            for output, changed, a, b in observed.pairs
            if output == principal
            and changed == fact
            and b in negative
            and a in positive
            and predicate(dict(negative[b].facts))
        }

    def local(record):
        return {
            name: value
            for entity, name, value in record.facts
            if entity == record.root_entity_id
        }

    for fact in (
        "t1170_qualified_donee",
        "t1170_property_source_class",
        "t1170_property_instrument_kind",
    ):
        require("gift:" + fact, direct(_GIFT, fact))
    for label, predicate in (
        ("before_year", lambda f: f["t1170_disposition_date"] < date(2025, 1, 1)),
        ("after_year", lambda f: f["t1170_disposition_date"] > date(2025, 12, 31)),
    ):
        require(label, direct(_GIFT, "t1170_disposition_date", predicate))
    for day in (date(2025, 1, 1), date(2025, 12, 31)):
        require(
            "included_date:" + day.isoformat(),
            (
                r.case_key
                for r in records
                if PREFIX + _GIFT in dict(r.outputs)
                and local(r).get("t1170_disposition_date") == day
            ),
        )
    for source_class in (
        "designated_exchange_share_debt_or_right",
        "mutual_fund_corporation_share",
        "mutual_fund_trust_unit",
        "prescribed_debt_obligation",
        "related_segregated_fund_trust_interest",
        "certified_ecologically_sensitive_land",
    ):
        require(
            "property_class:" + source_class,
            (
                r.case_key
                for r in records
                if PREFIX + _GIFT in dict(r.outputs)
                and local(r).get("t1170_property_source_class") == source_class
            ),
        )
    for fact in (
        "t1170_personal_servitude_term_years",
        "t1170_ecological_land_certified_by_eccc",
        "t1170_ecological_donee_not_private_foundation",
        "t1170_ecological_donee_approved_in_respect_of_gift",
        "t1170_land_interest_subtype",
        "t1170_ecological_donee_category",
    ):
        require("ecological:" + fact, direct(_GIFT, fact))
    for subtype in ("personal_servitude", "real_servitude"):
        require(
            "ecological_jurisdiction:" + subtype,
            direct(
                _GIFT,
                "t1170_land_location_jurisdiction",
                lambda f, subtype=subtype: f["t1170_land_interest_subtype"] == subtype,
            ),
        )
    for fact in (
        "t1170_exchange_right_at_issue",
        "t1170_exchange_right_at_disposition",
        "t1170_securities_only_exchange_consideration",
        "t1170_gift_is_of_securities_received_in_exchange",
        "t1170_exchanged_asset_kind",
        "received_property_class_and_instrument",
    ):
        require("prior:" + fact, direct(_PRIOR, fact))
    for label, predicate in (
        (
            "before_exchange",
            lambda f: f["t1170_disposition_date"] < f["t1170_exchange_date"],
        ),
        (
            "after_thirty_days",
            lambda f: (
                (f["t1170_disposition_date"] - f["t1170_exchange_date"]).days > 30
            ),
        ),
    ):
        require(label, direct(_PRIOR, "t1170_disposition_date", predicate))
    for days in (0, 30):
        require(
            "included_exchange_days:" + str(days),
            (
                r.case_key
                for r in records
                if PREFIX + _PRIOR in dict(r.outputs)
                and (
                    local(r)["t1170_disposition_date"] - local(r)["t1170_exchange_date"]
                ).days
                == days
            ),
        )
    for fact in (
        "t1170_partnership_exchange_right_at_issue",
        "t1170_partnership_exchange_right_at_disposition",
        "t1170_exchanged_asset_kind",
        "t1170_disposition_date",
    ):
        require("partnership:" + fact, direct(_PARTNERSHIP, fact))
    # Each aggregate needs its own actual selected-provider failure propagation.
    # No direct root failure or unrelated provider can satisfy these roles.
    for target in TARGETS:
        require(
            "provider:" + target,
            (
                (p.positive_key, p.negative_key, p.provider)
                for p in observed.provider_pairs
                if p.target == target
            ),
        )

    zero = PREFIX + "federal_2025_t1170_gain_eligible_zero_inclusion_column_7"
    other = PREFIX + "federal_2025_t1170_gain_subject_to_50_percent_inclusion_column_8"
    taxable = PREFIX + "federal_2025_t1170_taxable_capital_gain_from_gift"
    no_advantage, advantage = [], []
    for r in records:
        outputs, facts = dict(r.outputs), local(r)
        if not {PREFIX + _GIFT, zero, other, taxable} <= outputs.keys():
            continue
        if "t1170_advantage_in_respect_of_gift" not in facts:
            # Unused inputs can be absent in legitimate arithmetic-only
            # records. They supply no evidence of the source's advantage
            # premise; select explicit-fact examples without adding defaults.
            continue
        proceeds = facts["t1170_proceeds_of_disposition_column_2"]
        eligible = facts["t1170_eligible_amount_of_gift_column_6"]
        actual_advantage = facts["t1170_advantage_in_respect_of_gift"]
        # These are explicit source-coherent example premises, not a universal
        # legal input restriction or a definition of zero-denominator division.
        if proceeds <= 0 or eligible + actual_advantage != proceeds:
            continue
        if (
            actual_advantage == 0
            and outputs[zero] == outputs[PREFIX + _GIFT]
            and outputs[other] == outputs[taxable] == 0
        ):
            no_advantage.append(r.case_key)
        if (
            0 < eligible < proceeds
            and actual_advantage > 0
            and outputs[other] > 0
            and outputs[taxable] == outputs[other] * Decimal("0.5")
        ):
            advantage.append(r.case_key)
    require("no_advantage_arithmetic", no_advantage)
    require("advantage_and_half_arithmetic", advantage)
    # Existing positive observer independently validates the full graph, each
    # document/reporting pair, source-shaped per-row arithmetic and anti-pooling
    # controls. Keep those exact keyed pairs attached to their own output roles.
    for fact, output in (
        ("completing_t1170", "t1170_enclosure_required"),
        ("needs_t1170_additional_space", "t1170_additional_sheet_required"),
        (
            "t1170_fts_schedule3_additional_gain_applies",
            "t1170_fts_t691_reporting_required",
        ),
        (
            "t1170_nonprescribed_partnership_exchange_applies",
            "t1170_partnership_special_calculation_required",
        ),
        (
            "t1170_nonprescribed_partnership_exchange_applies",
            "t1170_partnership_schedule3_reporting_required",
        ),
    ):
        require(
            output,
            (
                (a, b)
                for f, o, a, b in observed.positive.root_role_pairs
                if f == PREFIX + "input." + fact and o == PREFIX + output
            ),
        )
    require(
        "completed_fts_copy",
        (
            r.case_key
            for r in records
            if PREFIX + "t1170_fts_gain_for_t691_part1_line25" in dict(r.outputs)
            and local(r).get("t1170_fts_schedule3_additional_gain_applies") is True
            and dict(r.outputs)[PREFIX + "t1170_fts_gain_for_t691_part1_line25"]
            == local(r)["t1170_completed_schedule3_fts_gain_line68231"]
        ),
    )
    return roles


# Closed extraction compositions are derived from the literal source records.
# Their coordinates are not accepted from a caller or from a generated policy.
def _branch_owners(source):
    records = {r.role: r for r in source.records}
    subrecords = {r.role: r for r in source.subrecords}
    text = source.source
    no_advantage_end = subrecords["no_advantage"].span[1]
    advantage_end = subrecords["advantage"].span[1]
    if text[no_advantage_end - 1] != " " or text[advantage_end - 1] != " ":
        raise ValueError("closed known extraction separators required")
    partnership = records["partnership"]
    partnership_text = text[slice(*partnership.span)]
    first_sentence_end = (
        partnership.span[0] + partnership_text.index(". This amount") + 1
    )
    fts = records["flow_through_scope"]
    fts_text = text[slice(*fts.span)]
    fts_start = fts.span[0] + fts_text.index("If this is the case,")
    fts_end = fts.span[0] + fts_text.index(" This additional capital gain")
    return {
        (0, no_advantage_end - 1): ("property_scope", "no_advantage_arithmetic"),
        (subrecords["advantage"].span[0], advantage_end - 1): (
            "advantage_and_half_arithmetic",
        ),
        subrecords["half_inclusion"].span: ("advantage_and_half_arithmetic",),
        records["prior_exchange"].span: ("prior_exchange",),
        (partnership.span[0], first_sentence_end): ("partnership",),
        (fts_start, fts_end): (
            "completed_fts_copy",
            "t1170_fts_t691_reporting_required",
        ),
    }


def _gate_spans(source):
    subrecords = {r.role: r for r in source.subrecords}
    start = subrecords["property_alternative_1"].span[0] - 1
    if source.source[start] != "\n":
        raise ValueError("closed property-list separator required")
    return (
        (0, subrecords["no_advantage"].span[1] - 1),
        (start, subrecords["half_inclusion"].span[1]),
    )


# Exact typed source occurrences relative to closed physical records.
# Each literal has its own role, offset and interpretation; no enclosing credit.
_NUMERIC_SOURCE_ROLES = (
    ("gift_inclusion", 308, 310, "50", 0.5, 50.0, True),
    ("shares_table", 498, 499, "1", 1.0, 1.0, False),
    ("bonds_table", 460, 461, "2", 2.0, 2.0, False),
    ("land_table", 534, 535, "3", 3.0, 3.0, False),
    ("property_scope", 122, 126, "2025", 2025.0, 2025.0, False),
    ("prior_exchange", 210, 212, "30", 30.0, 30.0, False),
    ("disposition_year", 0, 4, "5202", 5202.0, 5202.0, False),
    ("shares_table", 201, 202, "5", 5.0, 5.0, False),
    ("shares_table", 222, 223, "5", 5.0, 5.0, False),
    ("shares_table", 329, 330, "6", 6.0, 6.0, False),
    ("shares_table", 375, 376, "2", 2.0, 2.0, False),
    ("shares_table", 470, 471, "3", 3.0, 3.0, False),
    ("shares_table", 500, 505, "03286", 3286.0, 3286.0, False),
    ("shares_table", 512, 517, "02286", 2286.0, 2286.0, False),
    ("shares_table", 524, 526, ".3", 0.3, 0.3, False),
    ("shares_table", 539, 540, "3", 3.0, 3.0, False),
    ("shares_table", 549, 554, "00231", 231.0, 231.0, False),
    ("shares_table", 563, 564, "8", 8.0, 8.0, False),
    ("shares_table", 575, 576, "1", 1.0, 1.0, False),
    ("bonds_table", 167, 168, "5", 5.0, 5.0, False),
    ("bonds_table", 188, 189, "5", 5.0, 5.0, False),
    ("bonds_table", 300, 301, "6", 6.0, 6.0, False),
    ("bonds_table", 337, 338, "2", 2.0, 2.0, False),
    ("bonds_table", 432, 433, "3", 3.0, 3.0, False),
    ("bonds_table", 462, 467, "05286", 5286.0, 5286.0, False),
    ("bonds_table", 474, 479, "04286", 4286.0, 4286.0, False),
    ("bonds_table", 486, 488, ".3", 0.3, 0.3, False),
    ("bonds_table", 501, 502, "3", 3.0, 3.0, False),
    ("bonds_table", 511, 516, "00351", 351.0, 351.0, False),
    ("bonds_table", 525, 526, "8", 8.0, 8.0, False),
    ("bonds_table", 537, 538, "2", 2.0, 2.0, False),
    ("land_table", 255, 256, "5", 5.0, 5.0, False),
    ("land_table", 276, 277, "5", 5.0, 5.0, False),
    ("land_table", 385, 386, "6", 6.0, 6.0, False),
    ("land_table", 411, 412, "2", 2.0, 2.0, False),
    ("land_table", 506, 507, "3", 3.0, 3.0, False),
    ("land_table", 536, 541, "25286", 25286.0, 25286.0, False),
    ("land_table", 548, 553, "15286", 15286.0, 15286.0, False),
    ("land_table", 560, 562, ".3", 0.3, 0.3, False),
    ("land_table", 575, 576, "3", 3.0, 3.0, False),
    ("land_table", 585, 590, "00831", 831.0, 831.0, False),
    ("land_table", 599, 600, "8", 8.0, 8.0, False),
    ("land_table", 611, 612, "3", 3.0, 3.0, False),
    ("flow_through_route", 0, 1, "3", 3.0, 3.0, False),
    ("flow_through_route", 154, 155, "4", 4.0, 4.0, False),
    ("flow_through_route", 156, 161, "13286", 13286.0, 13286.0, False),
    ("flow_through_route", 201, 202, "1", 1.0, 1.0, False),
    ("flow_through_route", 211, 213, "52", 52.0, 52.0, False),
    ("flow_through_route", 222, 223, "4", 4.0, 4.0, False),
    ("page_two_display", 39, 40, "2", 2.0, 2.0, False),
    ("page_two_display", 44, 45, "2", 2.0, 2.0, False),
    ("prior_exchange", 167, 171, "five", 5.0, 5.0, False),
)


# These registries are analysis-local custody, not environment or caller grants.

_RESERVED = weakref.WeakSet()
_CERTIFICATES = weakref.WeakKeyDictionary()


@dataclass(eq=False, frozen=True)
class CapitalGiftEvidence:
    source: object | None
    complete: bool = False
    reserved_parent: bool = False

    def live(self):
        from .capital_gift_capture import source_binding_live

        entry = _CERTIFICATES.get(self)
        return bool(
            self.complete
            and entry is not None
            and source_binding_live(entry[0], entry[1])
        )

    def owns_branch(self, branch, kind):
        if self not in _RESERVED or self.source is None:
            return False
        owners = _branch_owners(self.source)
        spans = set(owners)
        if kind == "formula":
            spans = {
                span for span, roles in owners.items() if "completed_fts_copy" in roles
            }
        elif kind != "control":
            return False
        return bool(
            type(branch.start) is int
            and type(branch.end) is int
            and (branch.start, branch.end) in spans
            and branch.path == ()
            and branch.kind
            == ("formula-clause" if kind == "formula" else "exception-clause")
            and branch.text == self.source.source[branch.start : branch.end]
            and branch.required_principal_output is None
            and not branch.computation_chapeaux
            and not branch.structural_numeric_spans
        )

    def partition(self, branches, kind):
        counts = Counter((b.path, b.kind, b.start, b.end, b.text) for b in branches)
        remaining, missing = [], []
        live = self.live()
        for branch in branches:
            if not self.owns_branch(branch, kind):
                remaining.append(branch)
            elif (
                not live
                or counts[
                    (branch.path, branch.kind, branch.start, branch.end, branch.text)
                ]
                != 1
            ):
                missing.append(branch)
        return tuple(remaining), tuple(missing)

    def covers_gate(self, rule_name, rule, index, formula, start, end, clause):
        from .bound_capital_gift_operands import _exact, _load

        if not self.live() or type(index) is not int or index != 0:
            return False
        property_names = {
            _GIFT,
            "t1170_exchange_extension_conditions_met",
            *(
                "t1170_" + category + "_table_" + output
                for category in ("shares_units", "bonds_other", "ecological_land")
                for output in ("proceeds", "zero_inclusion", "other_gain")
            ),
        }
        arithmetic_names = {
            "federal_2025_t1170_taxable_capital_gain_from_gift",
            "t1170_no_advantage_in_respect_of_gift",
        }
        if rule_name not in property_names | arithmetic_names:
            return False
        span = _gate_spans(self.source)[int(rule_name in arithmetic_names)]
        if (
            type(clause.start) is not int
            or type(clause.end) is not int
            or (clause.start, clause.end) != span
            or clause.branch_path != ()
            or clause.text != self.source.source[slice(*span)]
        ):
            return False
        _, context, _ = _CERTIFICATES[self]
        expected = next(
            (r for r in _load(context.content)["rules"] if r["name"] == rule_name), None
        )
        if expected is None or not _exact(rule, expected):
            return False
        version = expected["versions"][0]
        return (formula, start, end) == (version["formula"], "2025-01-01", "2025-12-31")

    def numeric_owner(self, occurrence, *, recall_text=None):
        """Own raw coordinates, or explicitly map the actual recall input.

        Recall substitutions can delete a page label. The caller must supply
        the exact text it extracted; masked or otherwise different inputs do
        not inherit the raw source's coordinate authority.
        """
        from .validator_pipeline import NumericOccurrence

        if not self.live() or type(occurrence) is not NumericOccurrence:
            return None
        if recall_text is not None:
            from .bound_capital_gift_operands import PARENT
            from .source_completeness import (
                _NumericRecallSourceMap,
                authoritative_numeric_recall_text,
            )

            if (
                type(recall_text) is not str
                or type(occurrence.start) is not int
                or type(occurrence.end) is not int
            ):
                return None
            source_map = _NumericRecallSourceMap(list(range(len(self.source.source))))
            expected = authoritative_numeric_recall_text(
                self.source.source,
                corpus_citation_path=PARENT,
                _source_map=source_map,
            )
            if recall_text != expected:
                return None
            span = source_map.source_span(occurrence.start, occurrence.end)
            if (
                span is None
                or recall_text[occurrence.start : occurrence.end] != occurrence.raw
                or self.source.source[slice(*span)] != occurrence.raw
            ):
                return None
            occurrence = replace(occurrence, start=span[0], end=span[1])
        records = {r.role: r for r in self.source.records}
        for role, left, right, raw, value, source_value, rate in _NUMERIC_SOURCE_ROLES:
            start = records[role].span[0] + left
            end = records[role].span[0] + right
            if (
                type(occurrence.start) is int
                and type(occurrence.end) is int
                and (occurrence.start, occurrence.end, occurrence.raw)
                == (start, end, raw)
                and self.source.source[start:end] == raw
                and type(occurrence.value) is float
                and occurrence.value == value
                and type(occurrence.source_value) is float
                and occurrence.source_value == source_value
                and occurrence.has_rate_context is rate
                and occurrence.requires_rate_context is rate
                and occurrence.has_temporal_context is False
                and occurrence.has_structural_context is False
                and occurrence.is_word_number is False
                and occurrence.requires_month_context is False
                and type(occurrence.alternative_values) is tuple
                and occurrence.alternative_values == ()
            ):
                return role, start, end, raw
        return None

    def missing(self):
        if self.reserved_parent and not self.live():
            return (
                "[complete-source-unit:t1170-evidence] Complete private T1170 source-role, typed principal and direct/provider refusal evidence is required; generic witnesses cannot replace absent or stale owned evidence.",
            )
        return ()


def certify_capital_gift(content, source, citation, raw_cases, evidence):
    from .bound_capital_gift_operands import PARENT
    from .bound_capital_gift_structure import recognize_capital_gift_source
    from .capital_gift_capture import source_binding

    recognized = recognize_capital_gift_source(source, citation)
    absent = CapitalGiftEvidence(recognized, reserved_parent=citation == PARENT)
    _RESERVED.add(absent)
    if recognized is None:
        return absent
    actual = source_binding(evidence, content, source, raw_cases)
    if actual is None:
        return absent
    context, observation = actual
    try:
        roles = source_role_witnesses(context.binding, observation)
    except (ValueError, TypeError, KeyError, ArithmeticError):
        return absent
    result = CapitalGiftEvidence(recognized, True, reserved_parent=True)
    _RESERVED.add(result)
    _CERTIFICATES[result] = (evidence, context, roles)
    return result
