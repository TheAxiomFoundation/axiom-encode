"""Bound provider failure attribution; raw errors never identify a provider."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from .capital_gift_execution import (
    PREFIX,
    _compiled,
    _number,
    _request,
    capital_gift_case_key,
)

TARGETS = {
    "t1170_shares_units_zero_inclusion_line_68230": (
        "t1170_shares_units_row_for_return",
        "t1170_shares_units_table_zero_inclusion",
        "t1170_shares_units_row_within_table_scope",
    ),
    "t1170_bonds_other_zero_inclusion_line_68250": (
        "t1170_bonds_other_row_for_return",
        "t1170_bonds_other_table_zero_inclusion",
        "t1170_bonds_other_row_within_table_scope",
    ),
    "t1170_ecological_land_zero_inclusion_line_68252": (
        "t1170_ecological_land_row_for_return",
        "t1170_ecological_land_table_zero_inclusion",
        "t1170_ecological_land_row_within_table_scope",
    ),
    "t1170_partnership_gain_for_schedule3_line17400": (
        "t1170_partnership_exchange_for_return",
        "t1170_partnership_special_capital_gain_per_entry",
        "t1170_partnership_exchange_source_conditions_met",
    ),
}


class _NoMatch(ValueError):
    def __init__(self, entity, principal, selector):
        super().__init__("independently evaluated false source selector")
        self.identity = entity, principal, selector


class _Evaluation:
    """Pure evaluation of an independently lowered closed compiled graph.

    This produces neither a successful response nor an engine trace. Failures
    are attributed by actual typed facts and edges, never parsed out of stderr.
    """

    def __init__(self, derived, parameters, values, edges):
        self.derived, self.parameters = derived, parameters
        self.values, self.edges = values, edges
        self.computed, self.visiting = {}, set()

    def rule(self, name, entity):
        key = entity, name
        if key in self.computed:
            return self.computed[key]
        if key in self.visiting:
            raise ValueError("provider dependency cycle")
        self.visiting.add(key)
        try:
            value = self.ev(self.derived[name]["expr"], entity, name)
            self.computed[key] = value
            return value
        finally:
            self.visiting.remove(key)

    def ev(self, node, entity, principal):
        kind = node["kind"]
        if kind == "literal":
            wire = node["value"]
            return (
                _number(wire["value"]) if wire["kind"] == "decimal" else wire["value"]
            )
        if kind == "input":
            return self.values[(entity, node["name"])]
        if kind == "derived":
            return self.rule(node["name"], entity)
        if kind == "parameter_lookup":
            index = self.ev(node["index"], entity, principal)
            if type(index) is not int:
                raise ValueError("parameter index must be integer")
            wire = self.parameters[node["parameter"]]["versions"][0]["values"][
                str(index)
            ]
            if wire["kind"] == "date":
                return date.fromisoformat(wire["value"])
            return (
                _number(wire["value"]) if wire["kind"] == "decimal" else wire["value"]
            )
        if kind == "comparison":
            left = self.ev(node["left"], entity, principal)
            right = self.ev(node["right"], entity, principal)
            return {
                "eq": lambda: left == right,
                "gte": lambda: left >= right,
                "lte": lambda: left <= right,
            }[node["op"]]()
        if kind in ("and", "or"):
            for child in node["items"]:
                value = self.ev(child, entity, principal)
                if type(value) is not bool:
                    raise ValueError("Boolean condition required")
                if (kind == "and" and not value) or (kind == "or" and value):
                    return value
            return kind == "and"
        if kind == "if":
            condition = self.ev(node["condition"], entity, principal)
            if type(condition) is not bool:
                raise ValueError("Boolean branch condition required")
            return self.ev(
                node["then_expr"] if condition else node["else_expr"], entity, principal
            )
        if kind == "no_match":
            subject = node["subject"]
            if (
                subject["kind"] != "derived"
                or self.rule(subject["name"], entity) is not False
            ):
                raise ValueError("owned false derived selector required")
            raise _NoMatch(entity, principal, subject["name"])
        if kind in ("add", "min", "max"):
            items = [self.ev(x, entity, principal) for x in node["items"]]
            return (
                sum(items)
                if kind == "add"
                else min(items)
                if kind == "min"
                else max(items)
            )
        if kind in ("sub", "mul", "div"):
            left = self.ev(node["left"], entity, principal)
            right = self.ev(node["right"], entity, principal)
            return (
                left - right
                if kind == "sub"
                else left * right
                if kind == "mul"
                else left / right
            )
        if kind == "days_between":
            return (
                self.ev(node["to"], entity, principal)
                - self.ev(node["from"], entity, principal)
            ).days
        if kind == "sum_related":
            return sum(
                (
                    self.rule(node["value"]["name"], edge["tuple"][0])
                    for edge in self.edges
                    if edge["name"] == node["relation"] and edge["tuple"][1] == entity
                ),
                Decimal(0),
            )
        raise ValueError("unknown bound compiled expression")


@dataclass(frozen=True)
class CapitalGiftProviderRefusal:
    case_key: str
    root: str
    target: str
    relation: str
    provider: str
    provider_index: int
    principal: str
    selector: str
    facts: tuple
    edges: tuple


def observe_provider_refusal(binding, artifact, engine_pin, case, call):
    """Validate one unique selected-provider failure, without capture authority."""
    derived, parameters = _compiled(binding, artifact, engine_pin)
    if (
        type(call) is not dict
        or set(call) != {"request", "returncode", "stdout", "stderr"}
        or type(call["returncode"]) is not int
        or call["returncode"] != 1
        or call["stdout"] != ""
        or type(call["stderr"]) is not str
    ):
        raise ValueError("exact actual refusal process required")
    values, edges, root = _request(
        case, call["request"], binding, refusal=True, provider_refusal=True
    )
    target = case["expected_refusal"]["output"].removeprefix(PREFIX)
    if target not in TARGETS:
        raise ValueError("unsupported provider aggregate")
    relation, principal, selector = TARGETS[target]
    evaluation = _Evaluation(derived, parameters, values, edges)
    expression = derived[target]["expr"]
    if expression["kind"] == "if":
        if evaluation.ev(expression["condition"], root, target) is not True:
            raise ValueError("root refusal is not provider refusal")
        expression = expression["then_expr"]
    if (
        expression["kind"] != "sum_related"
        or expression["relation"] != PREFIX + "relation." + relation
        or expression["current_slot"] != 1
        or expression["related_slot"] != 0
        or expression["value"] != {"kind": "derived", "name": principal}
    ):
        raise ValueError("exact bound aggregate/provider direction required")
    selected = [
        edge["tuple"][0]
        for edge in edges
        if edge["name"] == expression["relation"] and edge["tuple"][1] == root
    ]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("unique reached provider identities required")
    failures = []
    for index, provider in enumerate(selected):
        try:
            evaluation.rule(principal, provider)
        except _NoMatch as failure:
            if failure.identity != (provider, principal, selector):
                raise ValueError(
                    "failure is not the owned provider selector"
                ) from failure
            failures.append((index, provider))
    if len(failures) != 1:
        raise ValueError("exactly one selected provider must fail")
    expected = (
        f"no `match` arm in `{PREFIX}{principal}` covers `{selector}` = false "
        "(arms: true); add an arm for it or a final `_ =>` arm\n"
    )
    if call["stderr"] != expected:
        raise ValueError("actual error differs from independently derived error")
    index, provider = failures[0]
    return CapitalGiftProviderRefusal(
        capital_gift_case_key(case, binding),
        root,
        target,
        relation,
        provider,
        index,
        principal,
        selector,
        tuple(
            (entity, name, value) for (entity, name), value in sorted(values.items())
        ),
        tuple((edge["name"], *edge["tuple"]) for edge in edges),
    )


@dataclass(frozen=True)
class CapitalGiftProviderPair:
    positive_key: str
    negative_key: str
    target: str
    relation: str
    provider: str
    provider_index: int
    principal: str
    selector: str
    changed_fact: str


def provider_refusal_pairs(
    binding, artifact, engine_pin, positive_cases, positive_calls, refusals
):
    """Revalidate positive raw records; preserve literal provider identities.

    The private consumer must independently revalidate each raw refusal before
    calling this helper. Its dataclasses alone carry no custody/source authority.
    """
    from .capital_gift_execution import _record

    derived, parameters = _compiled(binding, artifact, engine_pin)
    validated = []
    for case, call in zip(positive_cases, positive_calls, strict=True):
        record = _record(binding, derived, parameters, case, call)
        values, edges, root = _request(case, call["request"], binding)
        validated.append(
            (
                record,
                values,
                tuple((edge["name"], *edge["tuple"]) for edge in edges),
                root,
            )
        )
    pairs = []
    for negative in refusals:
        facts = {(entity, name): value for entity, name, value in negative.facts}
        matches = []
        for positive, values, edges, root in validated:
            if (
                root != negative.root
                or edges != negative.edges
                or set(values) != set(facts)
            ):
                continue
            if set(dict(positive.outputs)) != {PREFIX + negative.target}:
                continue
            changed = [
                key
                for key in facts
                if type(facts[key]) is not type(values[key])
                or facts[key] != values[key]
            ]
            if len(changed) != 1:
                continue
            entity, name = changed[0]
            if (
                entity != negative.provider
                or values[(entity, name)] is not True
                or facts[(entity, name)] is not False
            ):
                continue
            # The actual positive principal and selector must both be reached on
            # that same provider. A local Boolean-only query is not propagation.
            reached = {(e, n): v for e, n, v in positive.reached}
            if (
                reached.get((entity, negative.selector)) is not True
                or (entity, negative.principal) not in reached
            ):
                continue
            matches.append(
                CapitalGiftProviderPair(
                    positive.case_key,
                    negative.case_key,
                    negative.target,
                    negative.relation,
                    entity,
                    negative.provider_index,
                    negative.principal,
                    negative.selector,
                    name,
                )
            )
        if not matches:
            raise ValueError(
                "exact same-provider one-fact positive propagation pair required"
            )
        pairs.extend(matches)
    return tuple(pairs)
