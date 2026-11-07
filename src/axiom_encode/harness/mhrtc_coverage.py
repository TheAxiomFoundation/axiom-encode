"""Fact-sensitive MHRtC witness observations, never execution authority.

Only a private owner can authenticate the context and receipts. This module
reconstructs their transport and reports observations and unfinished obligations;
constructing its dataclasses grants no source coverage. Analyze a suite once,
then let the private owner recheck liveness when consuming the result.
"""

from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal

from .bound_mhrtc_program import bind_mhrtc_program
from .mhrtc_execution import same_typed_json, strict_json
from .mhrtc_otherclaim_obligations import mhrtc_otherclaim_obligations
from .mhrtc_otherclaim_request import reconstruct_mhrtc_other_claim_request
from .mhrtc_request import _PREFIX, parse_mhrtc_companion, reconstruct_mhrtc_request
from .mhrtc_results import (
    _assert_output,
    _declarations,
    _derived,
    _number,
    _numeric_wire,
    observe_mhrtc_outputs,
    validate_mhrtc_outputs,
    validate_mhrtc_parameters,
)

_INPUT = _PREFIX + "input."
_CREDIT = "multigenerational_home_renovation_tax_credit_line_45355"
_TOTAL = "federal_2025_mhrtc_total_qualifying_expenditures_line_45354"
_ELIGIBILITY = (
    "qualifying_individual",
    "qualifying_relation",
    "eligible_claimant",
    "eligible_dwelling",
    "secondary_unit",
    "qualifying_renovation",
    "claim_source_eligibility",
)
# Explicitly unfinished semantic refinements cannot disappear behind successful
# suite counts or broad positive/negative eligibility outcomes.
_UNFINISHED = (
    "eligibility.isolated_source_predicate_contrasts",
    "expense.coherent_event_payer_completion_contrasts",
    "expense.reasonableness_attribution_other_credit_contrasts",
    "land.building_and_physical_scope_contrasts",
    "history.same_different_individual_renovation_contrasts",
    "selection.record_order_independence",
    "otherclaim.local_expense_land_allocation_contrasts",
    "public_total.foreign_claim_exclusion_contrast",
)


@dataclass(frozen=True)
class MhrtcWitness:
    role: str
    ordinal: int
    entities: tuple[str, ...]
    trace_keys: tuple[str, ...]
    axis: str = ""


@dataclass(frozen=True)
class _AuxiliaryObservation:
    target: str
    role: str
    qualification: tuple
    capacity: tuple | None


@dataclass(frozen=True)
class MhrtcCoverage:
    witnesses: tuple[MhrtcWitness, ...]
    missing: tuple[str, ...]


def _trace_key(rule, entity, root):
    return (
        _PREFIX + rule + ("" if entity == root else "@entity:" + entity.encode().hex())
    )


def _trace(payload, request, response):
    """Validate evaluated declaration nodes; never use not-evaluated markers."""
    root = request["queries"][0]["entity_id"]
    declarations = _declarations(payload, "derived")
    entities = {
        (_derived(key, declarations)["entity"], root)
        for key in request["queries"][0]["outputs"]
    }
    dataset = request.get("dataset", {})
    entities.update(
        (row["entity"], row["entity_id"]) for row in dataset.get("inputs", [])
    )
    relation_kinds = {
        row["name"]: row["slot_entities"]
        for row in payload["program"].get("relations", [])
    }
    for edge in dataset.get("relations", []):
        kinds = relation_kinds.get(edge["name"])
        if kinds is None or len(kinds) != len(edge["tuple"]):
            raise ValueError("declared trace relation endpoints required")
        entities.update(zip(kinds, edge["tuple"], strict=True))
    result = {}
    for key, node in response["results"][0]["trace"].items():
        if type(node) is not dict or type(node.get("id")) is not str:
            raise ValueError("typed trace declaration required")
        declaration = _derived(node["id"], declarations)
        entity = node.get("entity_id")
        if type(entity) is not str or not entity:
            raise ValueError("explicit trace entity required")
        if (declaration["entity"], entity) not in entities:
            raise ValueError("trace entity kind must belong to frozen dataset")
        name = declaration["name"]
        if key != _trace_key(name, entity, root):
            raise ValueError("canonical trace identity required")
        dtype = declaration["dtype"]
        if dtype in ("integer", "decimal"):
            value = _numeric_wire(node.get("value"))
            if dtype == "integer":
                if value != value.to_integral_value():
                    raise ValueError("integral Integer trace required")
                value = int(value)
        elif dtype == "judgment":
            if "dtype" in node:
                raise ValueError("Judgment trace has no scalar dtype")
            value = node.get("outcome")
        elif dtype == "date":
            wire = node.get("value")
            value = wire.get("value") if type(wire) is dict else None
        else:
            raise ValueError("unsupported trace type")
        fields = (
            ("kind", "name", "id", "unit", "outcome")
            if dtype == "judgment"
            else ("kind", "name", "id", "dtype", "unit", "value")
        )
        if any(field not in node for field in fields):
            raise ValueError("complete typed trace projection required")
        _assert_output(
            node["id"], {field: node[field] for field in fields}, declaration, value
        )
        identity = (name, entity)
        if identity in result:
            raise ValueError("duplicate trace identity")
        result[identity] = value
    return result


def _fact(row, name):
    return row.get(_INPUT + name)


def _edges(case):
    return {
        (edge["name"].removeprefix(_PREFIX + "relation."), tuple(edge["tuple"]))
        for edge in case["relations"]
    }


def _rows(case, kind):
    rows = case["tables"].get(kind, [])
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("unique coverage entity rows required")
    return rows


def _owned(case, root, payload, plan):
    declaration = _declarations(payload, "derived")[plan.derived_outputs[0]]
    rows = _rows(case, "Claim")
    if declaration["entity"] == "Claim":
        merged = {"id": root, **case["input"]}
        for row in rows:
            if row["id"] != root:
                continue
            for key, value in row.items():
                if key in merged and not same_typed_json(merged[key], value):
                    raise ValueError("conflicting root Claim facts")
                merged[key] = value
        return [merged]
    if declaration["entity"] != "Family":
        raise ValueError("supported coverage root required")
    filer = _fact(case["input"], "filing_claimant_key")
    if type(filer) is not int:
        raise ValueError("exact filing claimant identity required")
    edges = _edges(case)
    linked = [r for r in rows if ("claim_records", (r["id"], root)) in edges]
    if any(type(_fact(r, "claimant_key")) is not int for r in linked):
        raise ValueError("explicit linked claimant identity required")
    return [r for r in linked if _fact(r, "claimant_key") == filer]


def _auxiliary(case, payload, receipts, *, pipe, rules_file, ordinal):
    targets = mhrtc_otherclaim_obligations(
        case, payload, pipe=pipe, rules_file=rules_file, case_index=ordinal
    )
    if (
        type(receipts) is not tuple
        or any(type(r) is not tuple or len(r) != 3 for r in receipts)
        or tuple(r[0] for r in receipts) != targets
    ):
        raise ValueError("complete ordered auxiliary obligations required")
    rows = {row["id"]: row for row in _rows(case, "OtherClaim")}
    observations = []
    for target, qualification, capacity in receipts:
        values = []
        traces = []
        for kind, receipt in (("qualification", qualification), ("capacity", capacity)):
            if receipt is None:
                values.append(None)
                traces.append(None)
                continue
            plan = reconstruct_mhrtc_other_claim_request(
                case,
                payload,
                pipe=pipe,
                rules_file=rules_file,
                case_index=ordinal,
                other_claim_id=target,
                witness_kind=kind,
            )
            if not same_typed_json(strict_json(receipt.request_json), plan.request):
                raise ValueError("same-facts auxiliary request required")
            response = strict_json(receipt.stdout)
            values.append(dict(observe_mhrtc_outputs(payload, plan.request, response)))
            traces.append(tuple(_trace(payload, plan.request, response).items()))
        qualification_values, capacity_values = values
        if qualification_values is None:
            raise ValueError("qualification execution required")
        eligible = qualification_values[_PREFIX + "other_claim_source_eligibility"]
        selected = qualification_values[_PREFIX + "other_selected_for_individual"]
        if eligible == "not_holds":
            if capacity_values is not None:
                raise ValueError("unexpected ineligible capacity witness")
            role = "shared.source_ineligible"
        else:
            amount = _number(_fact(rows[target], "other_actual_claimed_expenses"))
            if selected == "not_holds":
                if amount != 0 or capacity_values is not None:
                    raise ValueError("nonselected actual claim must be zero")
                role = "shared.nonselected_zero"
            else:
                if capacity_values is None:
                    raise ValueError("selected eligible capacity required")
                limit = capacity_values[_PREFIX + "other_allocated_local_capacity"]
                if amount < 0 or limit < 0 or amount > limit:
                    raise ValueError("actual claim exceeds eligible capacity")
                role = (
                    "shared.selected_positive" if amount > 0 else "shared.selected_zero"
                )
        observations.append(_AuxiliaryObservation(target, role, traces[0], traces[1]))
    return observations


def _recognize(views):
    witnesses = []
    required = set(_UNFINISHED) | {
        "completion.year_boundary",
        "cap.local_boundaries",
        "rate.nonzero",
        "public.independent_per_renovation_caps",
        "parameters.source_constants",
    }

    def add(role, view, entities, names, condition):
        required.add(role)
        if condition:
            keys = tuple(_trace_key(n, e, view["root"]) for n, e in names)
            if any((n, e) not in view["trace"] for n, e in names):
                raise ValueError("witness requires evaluated trace")
            witnesses.append(MhrtcWitness(role, view["ordinal"], tuple(entities), keys))

    # Register requirements independently of whether the suite has any Claims.
    for rule in _ELIGIBILITY:
        required.update(f"eligibility.{rule}.{v}" for v in ("holds", "not_holds"))
    required.update(
        (
            "expense.cutoff_rejected",
            "expense.paid_cutoff_first",
            "expense.incurred_cutoff_first",
            "reimbursement.zero_assistance",
            "reimbursement.partial",
            "land.at",
            "land.above",
            "land.actual_additional_scope",
            "minister.absent",
            "minister.applies",
            "minister.wrong_identity",
            "minister.lower_allocation",
            "minister.capacity_below_allocation",
            "selection.holds",
            "selection.not_holds",
            "history.none",
            "history.distinct",
            "documents.paper",
            "documents.electronic",
            "sheet.holds",
            "sheet.not_holds",
            "shared.selected_positive",
            "shared.selected_zero",
            "shared.source_ineligible",
            "shared.nonselected_zero",
            "shared.unrelated_or_self",
            "public.multiple_selected",
            "public.single_selected",
        )
    )
    for view in views:
        case, root, trace, parents = (
            view[k] for k in ("case", "root", "trace", "owned")
        )

        def t(name, entity=root):
            return trace.get((name, entity))

        for paper in (True, False):
            names = (
                "attach_schedule_to_return",
                "retain_supporting_documents",
                "include_supporting_documents_with_return",
            )
            add(
                "documents.paper" if paper else "documents.electronic",
                view,
                [root],
                [(n, root) for n in names],
                _fact(case["input"], "filing_paper_return") is paper
                and t(names[0]) == ("holds" if paper else "not_holds")
                and t(names[1]) == "holds"
                and t(names[2]) == "not_holds",
            )
        for outcome in ("holds", "not_holds"):
            indicators = [
                t("selected_renovation_sheet_count", p["id"]) for p in parents
            ]
            present = bool(indicators) and all(
                type(v) is int and v in (0, 1) for v in indicators
            )
            add(
                "sheet." + outcome,
                view,
                [root, *(p["id"] for p in parents)],
                [
                    ("additional_renovation_sheet_required", root),
                    *(("selected_renovation_sheet_count", p["id"]) for p in parents),
                ],
                present
                and t("additional_renovation_sheet_required") == outcome
                and (sum(indicators) > 1) == (outcome == "holds"),
            )
        selected = [
            p
            for p in parents
            if t("selected_for_individual", p["id"]) == "holds"
            and t("claim_source_eligibility", p["id"]) == "holds"
        ]
        for role, count in (
            ("public.single_selected", 1),
            ("public.multiple_selected", 2),
        ):
            individuals = {_fact(p, "individual_key") for p in selected}
            renovations = {_fact(p, "renovation_key") for p in selected}
            add(
                role,
                view,
                [root],
                [(_TOTAL, root), (_CREDIT, root)],
                (len(selected) == 1 if count == 1 else len(selected) >= count)
                and all(type(key) is int for key in individuals | renovations)
                and len(individuals) >= count
                and len(renovations) >= count
                and t(_TOTAL) is not None
                and t(_CREDIT) is not None,
            )
        for auxiliary in view["auxiliary"]:
            target, role = auxiliary.target, auxiliary.role
            # These canonical query outputs were validated separately above.
            outputs = (
                _PREFIX + "other_claim_source_eligibility",
                _PREFIX + "other_selected_for_individual",
            )
            if role in ("shared.selected_positive", "shared.selected_zero"):
                outputs += (_PREFIX + "other_allocated_local_capacity",)
            witnesses.append(
                MhrtcWitness(
                    role,
                    view["ordinal"],
                    (target,),
                    outputs,
                )
            )
        for parent in parents:
            entity = parent["id"]

            def p(name):
                return t(name, entity)

            for rule in _ELIGIBILITY:
                for outcome in ("holds", "not_holds"):
                    add(
                        f"eligibility.{rule}.{outcome}",
                        view,
                        [entity],
                        [(rule, entity)],
                        p(rule) == outcome,
                    )
            for outcome in ("holds", "not_holds"):
                add(
                    "selection." + outcome,
                    view,
                    [entity],
                    [("selected_for_individual", entity)],
                    p("selected_for_individual") == outcome,
                )
            prior = p("prior_distinct_renovation_count")
            for role, positive in (("history.none", False), ("history.distinct", True)):
                add(
                    role,
                    view,
                    [entity],
                    [("prior_distinct_renovation_count", entity)],
                    type(prior) is int and (prior > 0 if positive else prior == 0),
                )
            one, two, three = (p(n) for n in ("line_one", "line_two", "line_three"))
            if all(v is not None for v in (one, two, three)):
                for role, condition in (
                    ("reimbursement.zero_assistance", one > 0 and two == 0),
                    ("reimbursement.partial", 0 < two < one),
                ):
                    add(
                        role,
                        view,
                        [entity],
                        [(n, entity) for n in ("line_one", "line_two", "line_three")],
                        condition,
                    )
            decision = _fact(parent, "actual_minister_decision_exists")
            applies = p("actual_allocation_applies")
            for role, condition in (
                ("minister.absent", decision is False and applies == "not_holds"),
                ("minister.applies", decision is True and applies == "holds"),
                (
                    "minister.wrong_identity",
                    decision is True
                    and applies == "not_holds"
                    and (
                        _fact(parent, "decision_claimant_key")
                        != _fact(parent, "claimant_key")
                        or _fact(parent, "decision_renovation_key")
                        != _fact(parent, "renovation_key")
                    ),
                ),
            ):
                add(
                    role,
                    view,
                    [entity],
                    [("actual_allocation_applies", entity)],
                    condition,
                )
            allocation = _fact(parent, "allocated_expenditure_portion")
            capacity = p("allocated_local_capacity")
            if (
                decision is True
                and applies == "holds"
                and type(allocation) in (int, Decimal)
                and three is not None
                and capacity is not None
            ):
                for role, condition in (
                    (
                        "minister.lower_allocation",
                        0 <= allocation < three and capacity == allocation,
                    ),
                    (
                        "minister.capacity_below_allocation",
                        0 <= three < allocation and capacity == three,
                    ),
                ):
                    add(
                        role,
                        view,
                        [entity],
                        [("line_three", entity), ("allocated_local_capacity", entity)],
                        condition,
                    )
            for other in _rows(case, "OtherClaim"):
                linked = ("candidate_other_claims", (other["id"], entity)) in view[
                    "edges"
                ]
                keys = [
                    _fact(other, n)
                    for n in ("other_renovation_key", "other_claimant_key")
                ] + [_fact(parent, n) for n in ("renovation_key", "claimant_key")]
                unrelated = all(type(k) is int for k in keys) and (
                    keys[0] != keys[2] or keys[1] == keys[3]
                )
                add(
                    "shared.unrelated_or_self",
                    view,
                    [entity, other["id"]],
                    [("selected_other_claim_amount", entity)],
                    linked and unrelated and p("selected_other_claim_amount") == 0,
                )
            for expense in _rows(case, "Expense"):
                eid = expense["id"]
                if ("receipts", (eid, entity)) not in view["edges"]:
                    continue
                paid, incurred = (
                    _fact(expense, n) for n in ("paid_date", "incurred_date")
                )
                basic = t("receipt_basic_conditions", eid) == "holds"
                cutoff_names = [
                    ("paid_after_source_cutoff", eid),
                    ("incurred_after_source_cutoff", eid),
                    ("line_one", entity),
                ]
                add(
                    "expense.cutoff_rejected",
                    view,
                    [entity, eid],
                    cutoff_names,
                    basic
                    and paid == incurred == "2022-12-31"
                    and all(trace.get(k) == "not_holds" for k in cutoff_names[:2])
                    and one == 0,
                )
                for kind, own, other_date in (
                    ("paid", paid, incurred),
                    ("incurred", incurred, paid),
                ):
                    add(
                        f"expense.{kind}_cutoff_first",
                        view,
                        [entity, eid],
                        [(kind + "_after_source_cutoff", eid), ("line_one", entity)],
                        basic
                        and own == "2023-01-01"
                        and other_date == "2022-12-31"
                        and t(kind + "_after_source_cutoff", eid) == "holds"
                        and one is not None
                        and one > 0,
                    )
                scope = _fact(expense, "expense_land_scope_key")
                land = (
                    _fact(expense, "expense_is_land") is True
                    and type(scope) is int
                    and scope == _fact(parent, "dwelling_land_scope_key")
                )
                area = _fact(parent, "dwelling_land_area_hectares")
                if land and type(area) in (int, Decimal):
                    extra = _fact(parent, "additional_land_scope_applies")
                    for role, condition in (
                        (
                            "land.at",
                            area == Decimal(".5")
                            and p("land_extent_eligible") == "holds",
                        ),
                        (
                            "land.above",
                            area > Decimal(".5")
                            and extra is False
                            and p("land_extent_eligible") == "not_holds",
                        ),
                        (
                            "land.actual_additional_scope",
                            area > Decimal(".5")
                            and extra is True
                            and p("land_extent_eligible") == "holds",
                        ),
                    ):
                        add(
                            role,
                            view,
                            [entity, eid],
                            [("land_extent_eligible", entity)],
                            condition,
                        )
    refined, unfinished = _refinements(views)
    witnesses.extend(refined)
    required.update(unfinished)
    found = {w.role for w in witnesses}
    return MhrtcCoverage(tuple(witnesses), tuple(sorted(required - found)))


def _typed_signature(value):
    if type(value) is dict:
        return (dict, tuple(sorted((k, _typed_signature(v)) for k, v in value.items())))
    if type(value) in (list, tuple):
        return (type(value), tuple(_typed_signature(v) for v in value))
    return (type(value), value)


def _graph(case, masks=(), *, unordered=False):
    """An exact factual graph, excluding only names/assertions and closed masks."""
    graph = deepcopy({k: v for k, v in case.items() if k not in ("name", "output")})
    for kind, entity, fields in masks:
        rows = (
            [graph["input"]]
            if kind == "root"
            else [r for r in graph["tables"].get(kind, []) if r["id"] == entity]
        )
        for row in rows:
            for field in fields:
                row.pop(_INPUT + field, None)
    if unordered:
        for rows in graph["tables"].values():
            if len({r["id"] for r in rows}) != len(rows):
                raise ValueError("ambiguous order-equivalence identity")
            rows.sort(key=lambda row: row["id"])
        graph["relations"].sort(key=lambda edge: (edge["name"], tuple(edge["tuple"])))
    return _typed_signature(graph)


def _tv(view, name, entity):
    return view["trace"].get((name, entity))


def _participants(views, kind):
    for view in views:
        if kind == "Claim":
            for row in view["owned"]:
                yield view, row
        else:
            relation = {"Expense": "receipts", "History": "prior_renovations"}[kind]
            for row in _rows(view["case"], kind):
                if any(
                    (relation, (row["id"], p["id"])) in view["edges"]
                    for p in view["owned"]
                ):
                    yield view, row


def _parent(view, row, relation="receipts"):
    parents = [
        p for p in view["owned"] if (relation, (row["id"], p["id"])) in view["edges"]
    ]
    # A contrast involving shared rows must select a precise association first.
    return parents[0] if len(parents) == 1 else None


def _pairs(views, kind, fields, left, right):
    rows = list(_participants(views, kind))
    for a, ar in rows:
        if not left(a, ar):
            continue
        signature = _graph(a["case"], ((kind, ar["id"], fields),))
        for b, br in rows:
            if ar["id"] != br["id"] or not right(b, br):
                continue
            if _graph(a["case"]) == _graph(b["case"]):
                continue
            if signature == _graph(b["case"], ((kind, br["id"], fields),)):
                return ((a, ar), (b, br))
    return ()


def _refinements(views):
    """Closed factual axes, never case labels or a general formula interpreter."""
    obligations = {
        role: {}
        for role in (
            *_UNFINISHED,
            "completion.year_boundary",
            "cap.local_boundaries",
            "rate.nonzero",
            "public.independent_per_renovation_caps",
        )
    }

    def need(role, axis, matches, rule=None):
        records = []
        for view, row in matches:
            names = () if rule is None else (_trace_key(rule, row["id"], view["root"]),)
            records.append(
                MhrtcWitness(role, view["ordinal"], (row["id"],), names, axis)
            )
        obligations[role][axis] = tuple(records)

    def completion(v, r, inside):
        value = _fact(r, "completion_date")
        return (
            type(value) is str
            and (
                ("2025-01-01" <= value <= "2025-12-31")
                if inside
                else value >= "2026-01-01"
            )
            and _tv(v, "completion_in_tax_year", r["id"])
            == ("holds" if inside else "not_holds")
            and _tv(v, "claim_source_eligibility", r["id"])
            == ("holds" if inside else "not_holds")
        )

    need(
        "completion.year_boundary",
        "2025_vs_later_completion",
        _pairs(
            views,
            "Claim",
            ("completion_date",),
            lambda v, r: completion(v, r, True),
            lambda v, r: completion(v, r, False),
        ),
        "completion_in_tax_year",
    )
    for axis, compare in (
        ("below", lambda x: 0 < x < 50000),
        ("at", lambda x: x == 50000),
        ("above", lambda x: x > 50000),
    ):
        matches = []
        for v, r in _participants(views, "Claim"):
            one, two = (_tv(v, name, r["id"]) for name in ("line_one", "line_two"))
            raw = one - two if one is not None and two is not None else None
            limited = _tv(v, "selected_qualifying_expenditures", r["id"])
            if (
                raw is not None
                and compare(raw)
                and _tv(v, "selected_other_claim_amount", r["id"]) == 0
                and _tv(v, "actual_allocation_applies", r["id"]) == "not_holds"
                and limited == min(raw, Decimal(50000))
            ):
                matches.append((v, r))
        need(
            "cap.local_boundaries",
            axis,
            matches[:1],
            "selected_qualifying_expenditures",
        )
    rate, multiple = [], []
    for v in views:
        total = _tv(v, _TOTAL, v["root"])
        credit = _tv(v, _CREDIT, v["root"])
        if total is not None and total > 0 and credit == total * Decimal("0.145"):
            rate.append((v, {"id": v["root"]}))
            amounts = [
                _tv(v, "selected_qualifying_expenditures", p["id"]) for p in v["owned"]
            ]
            if (
                total > 50000
                and len(amounts) > 1
                and all(a is not None and 0 <= a <= 50000 for a in amounts)
                and sum(amounts) == total
            ):
                multiple.append((v, {"id": v["root"]}))
    need("rate.nonzero", "exact_0145", rate[:1], _CREDIT)
    need(
        "public.independent_per_renovation_caps",
        "summed_total_above_single_cap",
        multiple[:1],
        _TOTAL,
    )

    eligibility = _UNFINISHED[0]
    axes = {
        "qualifying_relation": (
            "relationship_to_individual_or_cohabiting_partner_during_year",
        ),
        "eligible_claimant": (
            "canada_resident_full_year",
            "claimant_is_relation",
            "claimant_is_individual",
            "claimant_is_individual_cohabiting_partner",
        ),
        "eligible_dwelling": (
            "dwelling_in_canada",
            "individual_dwelling_occupancy_within_twelve_months",
            "relation_dwelling_occupancy_within_twelve_months",
            "dwelling_owned_by_individual_or_relation_during_year",
            "dwelling_is_housing_unit",
            "dwelling_owner_trust_has_individual_or_relation_beneficiary_during_year",
        ),
        "secondary_unit": (
            "secondary_self_contained",
            "secondary_kitchen",
            "secondary_private_entrance",
            "secondary_bathroom",
            "secondary_sleeping_area",
            "local_requirements_met",
        ),
        "qualifying_renovation": (
            "renovation_enduring",
            "renovation_integral",
            "renovation_enables_coresidence",
            "secondary_occupancy_by_individual",
            "secondary_occupancy_by_relation",
        ),
    }
    for rule, fields in axes.items():
        for field in fields:
            pair = _pairs(
                views,
                "Claim",
                (field,),
                lambda v, r, f=field, n=rule: (
                    _fact(r, f) is True
                    and _tv(v, n, r["id"]) == "holds"
                    and _tv(v, "claim_source_eligibility", r["id"]) == "holds"
                ),
                lambda v, r, f=field, n=rule: (
                    _fact(r, f) is False
                    and _tv(v, n, r["id"]) == "not_holds"
                    and _tv(v, "claim_source_eligibility", r["id"]) == "not_holds"
                ),
            )
            need(eligibility, field, pair, rule)
    for field, rule, lower, upper, companion in (
        ("individual_age_at_year_end", "qualifying_individual", 17, 18, True),
        ("individual_age_at_year_end", "qualifying_individual", 64, 65, False),
        ("relation_age_at_year_end", "qualifying_relation", 17, 18, None),
    ):

        def boundary(v, r, value, outcome):
            return (
                type(_fact(r, field)) is int
                and _fact(r, field) == value
                and _tv(v, rule, r["id"]) == outcome
                and _tv(v, "claim_source_eligibility", r["id"]) == outcome
                and (companion is None or _fact(r, "individual_dtc") is companion)
            )

        need(
            eligibility,
            field + str(upper),
            _pairs(
                views,
                "Claim",
                (field,),
                lambda v, r: boundary(v, r, lower, "not_holds"),
                lambda v, r: boundary(v, r, upper, "holds"),
            ),
            rule,
        )
    need(
        eligibility,
        "dtc_alternative",
        _pairs(
            views,
            "Claim",
            ("individual_dtc",),
            lambda v, r: (
                _fact(r, "individual_dtc") is False
                and _fact(r, "individual_age_at_year_end") == 18
                and _tv(v, "qualifying_individual", r["id"]) == "not_holds"
                and _tv(v, "claim_source_eligibility", r["id"]) == "not_holds"
            ),
            lambda v, r: (
                _fact(r, "individual_dtc") is True
                and _fact(r, "individual_age_at_year_end") == 18
                and _tv(v, "qualifying_individual", r["id"]) == "holds"
                and _tv(v, "claim_source_eligibility", r["id"]) == "holds"
            ),
        ),
        "qualifying_individual",
    )
    need(
        eligibility,
        "local_requirements_apply",
        _pairs(
            views,
            "Claim",
            ("local_requirements_apply",),
            lambda v, r: (
                _fact(r, "local_requirements_apply") is False
                and _tv(v, "secondary_unit", r["id"]) == "holds"
                and _tv(v, "claim_source_eligibility", r["id"]) == "holds"
            ),
            lambda v, r: (
                _fact(r, "local_requirements_apply") is True
                and _tv(v, "secondary_unit", r["id"]) == "not_holds"
                and _tv(v, "claim_source_eligibility", r["id"]) == "not_holds"
            ),
        ),
        "secondary_unit",
    )
    relationships = (
        "parent",
        "grandparent",
        "child",
        "grandchild",
        "brother",
        "sister",
        "aunt",
        "uncle",
        "niece",
        "nephew",
    )
    for relationship in relationships:
        matches = [
            (v, r)
            for v, r in _participants(views, "Claim")
            if _fact(r, "family_relationship") == relationship
            and _tv(v, "qualifying_relation", r["id"]) == "holds"
            and _tv(v, "claim_source_eligibility", r["id"]) == "holds"
        ]
        need(eligibility, "family_" + relationship, matches[:1], "qualifying_relation")
    need(
        eligibility,
        "unrelated_family",
        _pairs(
            views,
            "Claim",
            ("family_relationship",),
            lambda v, r: (
                _fact(r, "family_relationship") in relationships
                and _tv(v, "qualifying_relation", r["id"]) == "holds"
                and _tv(v, "claim_source_eligibility", r["id"]) == "holds"
            ),
            lambda v, r: (
                _fact(r, "family_relationship") == "unrelated"
                and _tv(v, "qualifying_relation", r["id"]) == "not_holds"
                and _tv(v, "claim_source_eligibility", r["id"]) == "not_holds"
            ),
        ),
        "qualifying_relation",
    )
    for trust in (False, True):
        matches = [
            (v, r)
            for v, r in _participants(views, "Claim")
            if _fact(r, "claimant_dwelling_occupancy_within_twelve_months") is False
            and _fact(r, "claimant_is_relation") is True
            and _fact(r, "claimant_owns_dwelling") is (not trust)
            and _fact(r, "claimant_owning_trust_beneficiary") is trust
            and _tv(v, "eligible_claimant", r["id"]) == "holds"
            and _tv(v, "claim_source_eligibility", r["id"]) == "holds"
        ]
        need(
            eligibility,
            "nonoccupying_" + ("trust" if trust else "owner"),
            matches[:1],
            "eligible_claimant",
        )

    exclusions = _UNFINISHED[2]

    def receipt_result(v, r, basic, positive):
        parent = _parent(v, r)
        one = _tv(v, "line_one", parent["id"]) if parent else None
        return (
            parent is not None
            and _tv(v, "receipt_basic_conditions", r["id"]) == basic
            and one is not None
            and (one > 0 if positive else one == 0)
        )

    for field, allowed in (
        ("expense_reasonable", True),
        ("expense_directly_attributable", True),
        ("medical_credit_claimed", False),
        ("home_accessibility_credit_claimed", False),
    ):
        pair = _pairs(
            views,
            "Expense",
            (field,),
            lambda v, r, f=field, a=allowed: (
                _fact(r, f) is a and receipt_result(v, r, "holds", True)
            ),
            lambda v, r, f=field, a=allowed: (
                _fact(r, f) is (not a) and receipt_result(v, r, "not_holds", False)
            ),
        )
        need(exclusions, field, pair, "receipt_basic_conditions")
    mixed = []
    for view in views:
        for parent in view["owned"]:
            expenses = [
                r
                for r in _rows(view["case"], "Expense")
                if ("receipts", (r["id"], parent["id"])) in view["edges"]
            ]
            excluded = [
                r
                for r in expenses
                if _tv(view, "receipt_basic_conditions", r["id"]) == "not_holds"
                and _number(_fact(r, "reimbursement_received_or_receivable")) > 0
            ]
            included = [
                r
                for r in expenses
                if _tv(view, "receipt_basic_conditions", r["id"]) == "holds"
                and _fact(r, "reimbursement_received_or_receivable") == 0
            ]
            if (
                excluded
                and included
                and _tv(view, "line_one", parent["id"]) is not None
                and _tv(view, "line_one", parent["id"]) > 0
                and _tv(view, "line_two", parent["id"]) == 0
            ):
                mixed.append((view, parent))
    need(exclusions, "excluded_assistance_not_subtracted", mixed[:1], "line_two")

    history = _UNFINISHED[4]

    def historical(v, r, same_individual, same_renovation):
        parent = _parent(v, r, "prior_renovations")
        if parent is None:
            return False
        values = [
            _fact(r, "historical_individual_key"),
            _fact(r, "historical_renovation_key"),
            _fact(parent, "individual_key"),
            _fact(parent, "renovation_key"),
        ]
        if not all(type(value) is int for value in values):
            return False
        blocked = same_individual and not same_renovation
        return (
            (values[0] == values[2]) == same_individual
            and (values[1] == values[3]) == same_renovation
            and _tv(v, "prior_distinct_renovation_count", parent["id"])
            == (1 if blocked else 0)
            and _tv(v, "claim_source_eligibility", parent["id"])
            == ("not_holds" if blocked else "holds")
        )

    for field, other in (
        ("historical_renovation_key", (True, True)),
        ("historical_individual_key", (False, False)),
    ):
        need(
            history,
            field,
            _pairs(
                views,
                "History",
                (field,),
                lambda v, r: historical(v, r, True, False),
                lambda v, r, args=other: historical(v, r, *args),
            ),
        )

    _event_refinements(views, need, receipt_result)
    _land_refinements(views, need)
    _order_and_foreign_refinements(views, need)
    _otherclaim_refinements(views, need)
    witnesses, missing = [], []
    for role, axes in obligations.items():
        absent = [axis for axis, records in axes.items() if not records]
        if not axes:
            absent = ["not_implemented"]
        if absent:
            missing.extend(role + ":" + axis for axis in absent)
        else:
            witnesses.extend(
                witness for records in axes.values() for witness in records
            )
    return witnesses, missing


def _event_valid(expense, parent, event):
    date = _fact(expense, event + "_date")
    completion = _fact(parent, "completion_date")
    return (
        _fact(expense, event + "_event_exists") is True
        and (
            _fact(expense, event + "_by_claimant") is True
            or _fact(expense, event + "_by_claimant_beneficiary_trust") is True
        )
        and type(date) is str
        and type(completion) is str
        and date > "2022-12-31"
        and (
            date < completion
            or (
                date == completion
                and _fact(expense, event + "_same_day_before_completion") is True
            )
        )
    )


def _event_refinements(views, need, receipt_result):
    role = _UNFINISHED[1]

    def valid(v, r, positive):
        parent = _parent(v, r)
        return (
            parent is not None
            and receipt_result(v, r, "holds", positive)
            and (
                any(_event_valid(r, parent, e) for e in ("paid", "incurred"))
                == positive
            )
        )

    for event in ("paid", "incurred"):
        other = "incurred" if event == "paid" else "paid"
        pair = _pairs(
            views,
            "Expense",
            (event + "_date",),
            lambda v, r: (
                valid(v, r, False)
                and _fact(r, "paid_date") == _fact(r, "incurred_date") == "2022-12-31"
            ),
            lambda v, r, e=event, o=other: (
                valid(v, r, True)
                and _fact(r, e + "_date") == "2023-01-01"
                and _fact(r, o + "_date") == "2022-12-31"
            ),
        )
        need(role, event + "_cutoff", pair)
    flags = ("paid_same_day_before_completion", "incurred_same_day_before_completion")

    def same_day(v, r, before):
        parent = _parent(v, r)
        return (
            parent is not None
            and _fact(r, "paid_date")
            == _fact(r, "incurred_date")
            == _fact(parent, "completion_date")
            and all(_fact(r, f) is before for f in flags)
            and valid(v, r, before)
        )

    need(
        role,
        "same_day_ordering",
        _pairs(
            views,
            "Expense",
            flags,
            lambda v, r: same_day(v, r, False),
            lambda v, r: same_day(v, r, True),
        ),
    )
    for field in ("paid_by_claimant_beneficiary_trust", "incurred_by_claimant"):
        need(
            role,
            field,
            _pairs(
                views,
                "Expense",
                (field,),
                lambda v, r, f=field: _fact(r, f) is False and valid(v, r, False),
                lambda v, r, f=field: _fact(r, f) is True and valid(v, r, True),
            ),
        )
    payer_fields = ("paid_by_claimant", "incurred_by_claimant")
    need(
        role,
        "claimant_paid_route",
        _pairs(
            views,
            "Expense",
            payer_fields,
            lambda v, r: (
                all(_fact(r, f) is False for f in payer_fields) and valid(v, r, False)
            ),
            lambda v, r: _fact(r, "paid_by_claimant") is True and valid(v, r, True),
        ),
    )
    need(
        role,
        "date_payer_not_mixed",
        _pairs(
            views,
            "Expense",
            ("incurred_date",),
            lambda v, r: (
                _fact(r, "paid_by_claimant") is False
                and _fact(r, "paid_by_claimant_beneficiary_trust") is False
                and _fact(r, "incurred_by_claimant") is True
                and valid(v, r, True)
            ),
            lambda v, r: (
                _fact(r, "paid_by_claimant") is False
                and _fact(r, "paid_by_claimant_beneficiary_trust") is False
                and _fact(r, "incurred_by_claimant") is True
                and _fact(r, "incurred_date") == "2022-12-31"
                and valid(v, r, False)
            ),
        ),
    )

    def split_bounds(v, r):
        parent = _parent(v, r)
        return (
            parent is not None
            and all(
                type(value) is str
                for value in (
                    _fact(r, "paid_date"),
                    _fact(r, "incurred_date"),
                    _fact(parent, "completion_date"),
                )
            )
            and _fact(r, "paid_date") <= "2022-12-31"
            and _fact(r, "incurred_date") > _fact(parent, "completion_date")
            and valid(v, r, False)
        )

    need(
        role,
        "date_bounds_not_mixed",
        _pairs(
            views,
            "Expense",
            ("paid_date", "incurred_date"),
            lambda v, r: (
                _fact(r, "paid_date") == _fact(r, "incurred_date") == "2022-12-31"
                and valid(v, r, False)
            ),
            split_bounds,
        ),
    )
    for axis, predicate in (
        (
            "incurred_trust_only",
            lambda v, r: (
                _fact(r, "paid_event_exists") is False
                and _fact(r, "incurred_event_exists") is True
                and _fact(r, "incurred_by_claimant") is False
                and _fact(r, "incurred_by_claimant_beneficiary_trust") is True
                and valid(v, r, True)
            ),
        ),
        (
            "no_event",
            lambda v, r: (
                _fact(r, "paid_event_exists") is False
                and _fact(r, "incurred_event_exists") is False
                and valid(v, r, False)
            ),
        ),
    ):
        need(
            role,
            axis,
            [(v, r) for v, r in _participants(views, "Expense") if predicate(v, r)][:1],
        )


def _land_refinements(views, need):
    role = _UNFINISHED[3]

    def land(v, r, match, positive):
        parent = _parent(v, r)
        if parent is None:
            return False
        keys = (
            _fact(r, "expense_land_scope_key"),
            _fact(parent, "dwelling_land_scope_key"),
        )
        one = _tv(v, "line_one", parent["id"])
        return (
            _fact(r, "expense_is_land") is True
            and _fact(r, "expense_is_building") is False
            and all(type(k) is int for k in keys)
            and (keys[0] == keys[1]) == match
            and _number(_fact(r, "amount_including_taxes")) > 0
            and one is not None
            and (one > 0 if positive else one == 0)
        )

    need(
        role,
        "physical_scope",
        _pairs(
            views,
            "Expense",
            ("expense_land_scope_key",),
            lambda v, r: land(v, r, True, True),
            lambda v, r: land(v, r, False, False),
        ),
    )

    def extra_scope(v, r, match):
        parent = _parent(v, r)
        return (
            parent is not None
            and _fact(parent, "additional_land_scope_applies") is True
            and land(v, r, match, match)
        )

    need(
        role,
        "additional_scope_is_local",
        _pairs(
            views,
            "Expense",
            ("expense_land_scope_key",),
            lambda v, r: extra_scope(v, r, True),
            lambda v, r: extra_scope(v, r, False),
        ),
    )

    def threshold(v, r, above):
        area = _fact(r, "dwelling_land_area_hectares")
        expenses = [
            e
            for e in _rows(v["case"], "Expense")
            if ("receipts", (e["id"], r["id"])) in v["edges"]
        ]
        return (
            type(area) in (int, Decimal)
            and (area > Decimal(".5") if above else area == Decimal(".5"))
            and _fact(r, "additional_land_scope_applies") is False
            and any(land(v, e, True, not above) for e in expenses)
            and _tv(v, "land_extent_eligible", r["id"])
            == ("not_holds" if above else "holds")
        )

    need(
        role,
        "at_above_general_limit",
        _pairs(
            views,
            "Claim",
            ("dwelling_land_area_hectares",),
            lambda v, r: threshold(v, r, False),
            lambda v, r: threshold(v, r, True),
        ),
        "land_extent_eligible",
    )
    building, mixed, isolated = [], [], []
    for view in views:
        scope_results = []
        for parent in view["owned"]:
            expenses = [
                e
                for e in _rows(view["case"], "Expense")
                if ("receipts", (e["id"], parent["id"])) in view["edges"]
            ]
            amount = _tv(view, "line_one", parent["id"])
            area = _fact(parent, "dwelling_land_area_hectares")
            if (
                type(area) in (int, Decimal)
                and area > Decimal(".5")
                and amount is not None
                and amount > 0
                and any(
                    _fact(e, "expense_is_building") is True
                    and _fact(e, "expense_is_land") is False
                    for e in expenses
                )
            ):
                building.append((view, parent))
            included = [
                e
                for e in expenses
                if _fact(e, "expense_is_building") is True
                and _fact(e, "expense_is_land") is False
            ]
            outside = [
                e
                for e in expenses
                if _fact(e, "expense_is_land") is True
                and _fact(e, "expense_land_scope_key")
                != _fact(parent, "dwelling_land_scope_key")
            ]
            if (
                included
                and outside
                and amount
                == sum(
                    (_number(_fact(e, "amount_including_taxes")) for e in included),
                    Decimal(0),
                )
            ):
                mixed.append((view, parent))
            matching = [
                e
                for e in expenses
                if _fact(e, "expense_is_land") is True
                and _fact(e, "expense_land_scope_key")
                == _fact(parent, "dwelling_land_scope_key")
            ]
            if outside and not matching and amount == 0:
                scope_results.append((parent, outside))
        if len(view["owned"]) > 1 and scope_results:
            for parent, outside in scope_results:
                if any(
                    _fact(e, "expense_land_scope_key")
                    == _fact(other, "dwelling_land_scope_key")
                    for e in outside
                    for other in view["owned"]
                    if other["id"] != parent["id"]
                    and _tv(view, "line_one", other["id"]) is not None
                    and _tv(view, "line_one", other["id"]) > 0
                    and any(
                        ("receipts", (receipt["id"], other["id"])) in view["edges"]
                        and _fact(receipt, "expense_is_land") is True
                        and _fact(receipt, "expense_land_scope_key")
                        == _fact(other, "dwelling_land_scope_key")
                        for receipt in _rows(view["case"], "Expense")
                    )
                ):
                    isolated.append((view, parent))
    need(role, "building_large_property", building[:1], "line_one")
    need(role, "mixed_building_outside_land", mixed[:1], "line_one")
    need(role, "two_claim_scope_isolation", isolated[:1], "line_one")


def _foreign_matches_executed_owned(view, foreign):
    """A bounded factual isomorphism, not an evaluator of foreign tax rules.

    Only claimant identity and expense entity labels may differ. Shared physical,
    individual, selection and history identities remain exact. An actual owned
    trace supplies the positive eligibility/selection/capacity observation.
    """
    case, edges = view["case"], view["edges"]

    def facts(row, omitted=()):
        return _typed_signature({k: v for k, v in row.items() if k not in omitted})

    def children(claim, kind, relation, *, omit_id=False):
        rows = {r["id"]: r for r in _rows(case, kind)}
        ids = [
            edge[0]
            for name, edge in edges
            if name == relation and edge[1] == claim["id"]
        ]
        if any(entity not in rows for entity in ids):
            return None
        return sorted(
            (facts(rows[entity], ("id",) if omit_id else ()) for entity in ids),
            key=repr,
        )

    for owned in view["owned"]:
        amount = _tv(view, "selected_qualifying_expenditures", owned["id"])
        if (
            _tv(view, "claim_source_eligibility", owned["id"]) != "holds"
            or _tv(view, "selected_for_individual", owned["id"]) != "holds"
            or amount is None
            or amount <= 0
            or facts(foreign, ("id", _INPUT + "claimant_key"))
            != facts(owned, ("id", _INPUT + "claimant_key"))
        ):
            continue
        # Other-claim matching depends on claimant identity; this narrow witness
        # does not rename those associations or infer their counterfactual result.
        if any(
            name == "candidate_other_claims" and edge[1] in (foreign["id"], owned["id"])
            for name, edge in edges
        ):
            continue
        same = True
        for kind, relation, omit_id in (
            ("Expense", "receipts", True),
            ("QualifyingIndividual", "claim_selection_records", False),
            ("History", "prior_renovations", False),
        ):
            left = children(foreign, kind, relation, omit_id=omit_id)
            right = children(owned, kind, relation, omit_id=omit_id)
            if left is None or right is None or left != right:
                same = False
                break
        if same:
            return True
    return False


def _order_and_foreign_refinements(views, need):
    order_role, foreign_role = _UNFINISHED[5], _UNFINISHED[7]
    reordered = []
    for a in views:
        selected = {
            p["id"]: _tv(a, "selected_for_individual", p["id"]) for p in a["owned"]
        }
        if set(selected.values()) != {"holds", "not_holds"}:
            continue
        if not any(
            _fact(p, "individual_key") == _fact(q, "individual_key")
            and _fact(p, "renovation_key") != _fact(q, "renovation_key")
            for p in a["owned"]
            for q in a["owned"]
        ):
            continue
        for b in views:
            if _graph(a["case"]) == _graph(b["case"]) or _graph(
                a["case"], unordered=True
            ) != _graph(b["case"], unordered=True):
                continue
            other = {
                p["id"]: _tv(b, "selected_for_individual", p["id"]) for p in b["owned"]
            }
            if selected == other and all(
                _tv(a, name, a["root"]) is not None
                and _tv(a, name, a["root"]) == _tv(b, name, b["root"])
                for name in (_TOTAL, _CREDIT)
            ):
                reordered = [(a, {"id": a["root"]}), (b, {"id": b["root"]})]
                break
        if reordered:
            break
    need(order_role, "identities_and_outputs_under_permutation", reordered)
    foreign = []
    for large in views:
        case = large["case"]
        if (
            _tv(large, _TOTAL, large["root"]) is None
            or _tv(large, _TOTAL, large["root"]) <= 0
        ):
            continue
        filer = _fact(case["input"], "filing_claimant_key")
        added_claims = [
            r
            for r in _rows(case, "Claim")
            if type(_fact(r, "claimant_key")) is int
            and _fact(r, "claimant_key") != filer
            and ("claim_records", (r["id"], large["root"])) in large["edges"]
        ]
        if not added_claims:
            continue
        positive_foreign = any(
            ("receipts", (expense["id"], claim["id"])) in large["edges"]
            and _foreign_matches_executed_owned(large, claim)
            and type(_fact(expense, "amount_including_taxes")) in (int, Decimal)
            and type(_fact(expense, "reimbursement_received_or_receivable"))
            in (int, Decimal)
            and _fact(expense, "amount_including_taxes")
            > _fact(expense, "reimbursement_received_or_receivable")
            >= 0
            and _fact(expense, "expense_reasonable") is True
            and _fact(expense, "expense_directly_attributable") is True
            and _fact(expense, "medical_credit_claimed") is False
            and _fact(expense, "home_accessibility_credit_claimed") is False
            and any(
                _event_valid(expense, claim, event) for event in ("paid", "incurred")
            )
            for claim in added_claims
            for expense in _rows(case, "Expense")
        )
        if not positive_foreign:
            continue
        claim_ids = {r["id"] for r in added_claims}
        for small in views:
            if not all(
                _tv(large, n, large["root"]) is not None
                and _tv(large, n, large["root"]) == _tv(small, n, small["root"])
                for n in (_TOTAL, _CREDIT)
            ):
                continue
            reduced = deepcopy(case)
            allowed = {
                "Claim": claim_ids,
                "Expense": set(),
                "QualifyingIndividual": set(),
                "History": set(),
            }
            for relation, kind in (
                ("receipts", "Expense"),
                ("claim_selection_records", "QualifyingIndividual"),
                ("prior_renovations", "History"),
            ):
                for name, edge in large["edges"]:
                    if name == relation and edge[1] in claim_ids:
                        # A shared unchanged identity row may remain in small graph.
                        if not any(
                            r["id"] == edge[0]
                            for r in small["case"]["tables"].get(kind, [])
                        ):
                            allowed[kind].add(edge[0])
            for kind, ids in allowed.items():
                if kind in reduced["tables"]:
                    reduced["tables"][kind] = [
                        r for r in reduced["tables"][kind] if r["id"] not in ids
                    ]
            removed_edges = []
            kept_edges = []
            for edge in reduced["relations"]:
                name = edge["name"].removeprefix(_PREFIX + "relation.")
                tup = tuple(edge["tuple"])
                if (
                    name == "claim_records"
                    and tup[0] in claim_ids
                    and tup[1] == large["root"]
                ) or (
                    name in ("receipts", "claim_selection_records", "prior_renovations")
                    and tup[1] in claim_ids
                ):
                    removed_edges.append(edge)
                else:
                    kept_edges.append(edge)
            reduced["relations"] = kept_edges
            if (
                removed_edges
                and _graph(reduced) == _graph(small["case"])
                and _tv(large, "selected_total_qualifying_expenditures", large["root"])
                is not None
            ):
                foreign = [
                    (small, {"id": small["root"]}),
                    (large, {"id": large["root"]}),
                ]
                break
        if foreign:
            break
    need(foreign_role, "unchanged_owned_graph_plus_foreign_subtree", foreign)


def _otherclaim_refinements(views, need):
    role = _UNFINISHED[6]
    candidates = []
    for view in views:
        for auxiliary in view["auxiliary"]:
            if auxiliary.capacity is None:
                continue
            trace = dict(auxiliary.capacity)
            target = auxiliary.target
            row = next(
                r for r in _rows(view["case"], "OtherClaim") if r["id"] == target
            )
            expenses = [
                r
                for r in _rows(view["case"], "OtherExpense")
                if ("other_receipts", (r["id"], target)) in view["edges"]
            ]
            if len(expenses) != 1:
                continue
            expense = expenses[0]
            # Positive local capacity witnesses must use coherent event facts,
            # including the local typed date trace; no principal-total shortcut.
            local_expense = {
                k.replace(_INPUT + "other_", _INPUT): value
                for k, value in expense.items()
            }
            local_claim = {
                k.replace(_INPUT + "other_", _INPUT): value for k, value in row.items()
            }
            coherent_events = [
                e
                for e in ("paid", "incurred")
                if _event_valid(local_expense, local_claim, e)
            ]
            if (
                not coherent_events
                or trace.get(("other_receipt_basic_conditions", expense["id"]))
                != "holds"
            ):
                continue
            if trace.get(("other_line_one", target), 0) > 0 and not any(
                trace.get(("other_" + event + "_after_source_cutoff", expense["id"]))
                == "holds"
                for event in coherent_events
            ):
                continue
            candidates.append((view, row, expense, trace))

    def local(v, row, expense, trace, kind, matching, allocation=False):
        target = row["id"]

        def get(name):
            return trace.get(("other_" + name, target))

        amount = _number(_fact(expense, "other_amount_including_taxes"))
        claimed = _number(_fact(row, "other_actual_claimed_expenses"))
        if amount <= 0 or _fact(expense, "other_expense_is_" + kind) is not True:
            return False
        if kind == "land":
            keys = (
                _fact(expense, "other_expense_land_scope_key"),
                _fact(row, "other_dwelling_land_scope_key"),
            )
            if (
                not all(type(k) is int for k in keys)
                or (keys[0] == keys[1]) != matching
            ):
                return False
        if allocation:
            if _fact(row, "other_actual_minister_decision_exists") is not True:
                return False
            portion = _number(_fact(row, "other_allocated_expenditure_portion"))
            return (
                _fact(row, "other_actual_minister_decision_exists") is True
                and _fact(row, "other_decision_claimant_key")
                == _fact(row, "other_claimant_key")
                and _fact(row, "other_decision_renovation_key")
                == _fact(row, "other_renovation_key")
                and get("actual_allocation_applies") == "holds"
                and get("line_three") is not None
                and 0 <= portion < get("line_three")
                and get("allocated_local_capacity") == portion == claimed
            )
        return (
            _fact(row, "other_actual_minister_decision_exists") is False
            and get("actual_allocation_applies") == "not_holds"
            and get("line_one") == (amount if matching else 0)
            and get("line_two") == 0
            and get("line_three")
            == get("allocated_local_capacity")
            == claimed
            == (amount if matching else 0)
        )

    groups = {
        "building": [x for x in candidates if local(*x, "building", True)],
        "land": [x for x in candidates if local(*x, "land", True)],
        "outside": [x for x in candidates if local(*x, "land", False)],
        "allocation": [x for x in candidates if local(*x, "building", True, True)],
    }
    for axis, left, right, claim_fields, expense_fields in (
        (
            "building_land_same_capacity",
            "building",
            "land",
            (),
            (
                "other_expense_is_building",
                "other_expense_is_land",
                "other_expense_land_scope_key",
            ),
        ),
        (
            "local_scope_zero_capacity",
            "land",
            "outside",
            ("other_actual_claimed_expenses",),
            ("other_expense_land_scope_key",),
        ),
        (
            "actual_allocation_local_ceiling",
            "building",
            "allocation",
            (
                "other_actual_minister_decision_exists",
                "other_decision_claimant_key",
                "other_decision_renovation_key",
                "other_allocated_expenditure_portion",
                "other_actual_claimed_expenses",
            ),
            (),
        ),
    ):
        pair = []
        for a, ar, ae, _ in groups[left]:
            for b, br, be, _ in groups[right]:
                if ar["id"] != br["id"] or ae["id"] != be["id"]:
                    continue
                masks = (
                    ("OtherClaim", ar["id"], claim_fields),
                    ("OtherExpense", ae["id"], expense_fields),
                )
                if _graph(a["case"], masks) == _graph(b["case"], masks):
                    pair = [(a, ar), (b, br)]
                    break
            if pair:
                break
        need(role, axis, pair)


def observe_mhrtc_coverage(context, receipts, *, pipe, rules_file):
    """Observe one complete suite; caller must authenticate private provenance."""
    if bind_mhrtc_program(context.content.decode(), context.source) is None:
        raise ValueError("closed source/formula binding required")
    cases = parse_mhrtc_companion(context.companion.decode())
    payload = strict_json(context.rebuilt)
    if (
        type(receipts) is not tuple
        or any(type(getattr(r, "ordinal", None)) is not int for r in receipts)
        or tuple(r.ordinal for r in receipts) != context.principal_ordinals
    ):
        raise ValueError("complete ordered principal receipts required")
    by_ordinal = {r.ordinal: r for r in receipts}
    if len(by_ordinal) != len(receipts):
        raise ValueError("unique principal ordinals required")
    principal, parameters, views = [], [], []
    for ordinal, case in enumerate(cases, 1):
        plan = reconstruct_mhrtc_request(
            case, payload, pipe=pipe, rules_file=rules_file, case_index=ordinal
        )
        if plan.request is None:
            validate_mhrtc_parameters(payload, case["output"])
            parameters.append(ordinal)
            continue
        principal.append(ordinal)
        receipt = by_ordinal.get(ordinal)
        if (
            receipt is None
            or type(receipt.record.case_index) is not int
            or receipt.record.case_index != ordinal
        ):
            raise ValueError("matching principal record required")
        request = strict_json(receipt.principal.request_json)
        response = strict_json(receipt.principal.stdout)
        if (
            not same_typed_json(request, plan.request)
            or not same_typed_json(strict_json(receipt.record.request_json), request)
            or not same_typed_json(strict_json(receipt.record.response_json), response)
        ):
            raise ValueError("same frozen case request and response required")
        validate_mhrtc_outputs(payload, request, response, case["output"])
        root = request["queries"][0]["entity_id"]
        views.append(
            {
                "ordinal": ordinal,
                "case": case,
                "root": root,
                "trace": _trace(payload, request, response),
                "owned": _owned(case, root, payload, plan),
                "edges": _edges(case),
                "auxiliary": _auxiliary(
                    case,
                    payload,
                    receipt.auxiliary,
                    pipe=pipe,
                    rules_file=rules_file,
                    ordinal=ordinal,
                ),
            }
        )
    if (
        tuple(principal) != context.principal_ordinals
        or tuple(parameters) != context.parameter_ordinals
    ):
        raise ValueError("frozen principal/parameter closure differs")
    observed = _recognize(views)
    # Static parameter assertions are distinct from runtime CaseExecution rows.
    static = tuple(
        MhrtcWitness(
            "parameters.source_constants", ordinal, (), (), "static_no_runtime"
        )
        for ordinal in parameters
    )
    missing = tuple(
        role
        for role in observed.missing
        if role != "parameters.source_constants" or not static
    )
    return MhrtcCoverage(observed.witnesses + static, missing)
