"""Reconstruct the closed MHRtC companion transport without running an engine.

The caller must already bind the compiled artifact, policy, source, companion
and ordinal to a fresh private context. A plan is public data, not authority.
Expected-output checks, Explain metadata and full coverage remain separate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from axiom_encode.companion_relations import executable_relation_directions

from ..repo_routing import _rulespec_routing_cache_scope
from .bound_mhrtc_program import _UniqueLoader
from .mhrtc_program_contract import INPUTS, POLICY_PATH, RULES
from .validator_pipeline import (
    _rulespec_declared_relation_names,
    _rulespec_declared_relation_slots,
    _rulespec_module_target,
    _rulespec_program_has_legal_ids,
)

_TARGET = "ca:" + POLICY_PATH.removeprefix("ca/").removesuffix(".yaml")
_PREFIX = _TARGET + "#"
_PERIOD = {
    "period_kind": "tax_year",
    "start": "2025-01-01",
    "end": "2025-12-31",
}
_INPUTS = {_PREFIX + "input." + item["name"]: item for item in INPUTS}
_RULES = {_PREFIX + item["name"]: item for item in RULES}
_RELATIONS = {
    _PREFIX + "relation." + item["name"]: item
    for item in RULES
    if item["kind"] in {"data_relation", "derived_relation"}
}
_PUBLIC_PARAMETERS = frozenset(
    _PREFIX + name
    for name in (
        "federal_2025_mhrtc_qualifying_expenditure_limit_per_renovation",
        "federal_2025_mhrtc_rate",
        "federal_2025_mhrtc_maximum_credit_per_renovation",
    )
)


@dataclass(frozen=True)
class MhrtcRequestPlan:
    """A parameter-only plan has no request and no fabricated runtime receipt."""

    request: dict[str, Any] | None
    derived_outputs: tuple[str, ...]
    parameter_outputs: tuple[str, ...]


def _validate_facts(facts: object, entity: str) -> None:
    if type(facts) is not dict:
        raise ValueError("MHRtC factual input mapping required")
    for key, value in facts.items():
        if type(key) is not str or key not in _INPUTS:
            raise ValueError("canonical declared MHRtC input required")
        item = _INPUTS[key]
        if item["entity"] != entity:
            raise ValueError("input belongs to another declared entity")
        dtype = item["dtype"]
        valid = (
            (dtype == "Judgment" and type(value) is bool)
            or (dtype == "Integer" and type(value) is int)
            or (
                dtype in {"Money", "Decimal", "Rate"}
                and (
                    type(value) is int
                    or (type(value) is float and math.isfinite(value))
                    or (type(value) is Decimal and value.is_finite())
                )
            )
            or (dtype in {"Text", "Date"} and type(value) is str)
        )
        if not valid:
            raise ValueError("factual value differs from declared scalar type")
        if dtype == "Date" and date.fromisoformat(value).isoformat() != value:
            raise ValueError("canonical ISO factual date required")


def _validate_case(case: object) -> None:
    base = {"name", "entity_id", "period", "input", "output"}
    if type(case) is not dict or set(case) not in (
        base,
        base | {"tables", "relations"},
    ):
        raise ValueError("closed MHRtC companion fields required")
    if any(
        type(case[k]) is not str or not case[k].strip() for k in ("name", "entity_id")
    ):
        raise ValueError("explicit case name and root identity required")
    if type(case["period"]) is not dict or case["period"] != _PERIOD:
        raise ValueError("exact 2025 tax-year period required")
    if (
        type(case["input"]) is not dict
        or type(case["output"]) is not dict
        or not case["output"]
    ):
        raise ValueError("input/output mappings required")
    for key, value in case["output"].items():
        if type(key) is not str or key not in _RULES:
            raise ValueError("canonical MHRtC output required; no aliases")
        if (
            type(value) not in (str, bool, int, float, Decimal)
            or (type(value) is float and not math.isfinite(value))
            or (type(value) is Decimal and not value.is_finite())
        ):
            raise ValueError("scalar expected outputs required; no row-list outputs")
    if "relations" not in case:
        if set(case["output"]) != _PUBLIC_PARAMETERS or case["input"]:
            raise ValueError(
                "only the three public parameter assertions omit the graph"
            )
        return
    if type(case["relations"]) is not list or type(case["tables"]) is not dict:
        raise ValueError("explicit relation list and table mapping required")
    for edge in case["relations"]:
        if (
            type(edge) is not dict
            or set(edge) != {"name", "tuple"}
            or type(edge["name"]) is not str
            or edge["name"] not in _RELATIONS
            or _RELATIONS[edge["name"]]["kind"] != "data_relation"
        ):
            raise ValueError("explicit declared MHRtC data-relation edge required")
    known_entities = {item["entity"] for item in INPUTS}
    for entity, rows in case["tables"].items():
        if (
            type(entity) is not str
            or entity not in known_entities
            or type(rows) is not list
        ):
            raise ValueError("declared table entity and row list required")
        for row in rows:
            if (
                type(row) is not dict
                or type(row.get("id")) is not str
                or not row["id"].strip()
            ):
                raise ValueError("table rows require explicit id; no identity aliases")
            _validate_facts(
                {key: value for key, value in row.items() if key != "id"}, entity
            )


def parse_mhrtc_companion(content: str) -> tuple[dict, ...]:
    """Load raw companion bytes before duplicate-key/alias information is lost."""
    if type(content) is not str or len(content) > 4 * 1024 * 1024:
        raise ValueError("bounded MHRtC companion text required")
    for token in yaml.scan(content):
        if isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken)):
            raise ValueError("companion anchors and aliases are not supported")
    cases = yaml.load(content, Loader=_UniqueLoader)
    if type(cases) is not list or not cases or len(cases) > 1024:
        raise ValueError("bounded nonempty case list required")
    names = set()
    for case in cases:
        _validate_case(case)
        if case["name"] in names:
            raise ValueError("duplicate case names")
        names.add(case["name"])
    return tuple(cases)


def _compiled_maps(pipe, payload: dict) -> tuple[dict, dict]:
    if type(payload) is not dict or type(payload.get("program")) is not dict:
        raise ValueError("bound compiled program required")
    program = payload["program"]
    seen = set()
    for collection, kind in (("derived", "derived"), ("parameters", "parameter")):
        if type(program.get(collection)) is not list:
            raise ValueError("compiled declaration lists required")
        for item in program[collection]:
            if type(item) is not dict or type(item.get("id")) is not str:
                raise ValueError("canonical compiled identifier required")
            key = item["id"]
            expected = _RULES.get(key)
            if (
                key in seen
                or expected is None
                or expected["kind"] != kind
                or item.get("name") != expected["name"]
                or (kind == "derived" and item.get("entity") != expected["entity"])
            ):
                raise ValueError("compiled output identity/role is not canonical")
            seen.add(key)
    if _rulespec_module_target(
        payload
    ) != _TARGET or not _rulespec_program_has_legal_ids(payload):
        raise ValueError("canonical MHRtC compiled module required")
    relations = program.get("relations")
    if type(relations) is not list:
        raise ValueError("compiled relation declarations required")
    relation_names = set()
    for relation in relations:
        if type(relation) is not dict or type(relation.get("name")) is not str:
            raise ValueError("canonical compiled relation required")
        name = relation["name"]
        expected = _RELATIONS.get(name)
        if expected is None or name in relation_names:
            raise ValueError("unknown or duplicate compiled relation")
        relation_names.add(name)
        declaration = expected[expected["kind"]]
        slots = declaration.get("arguments", declaration.get("slot_entities"))
        if (
            type(relation.get("arity")) is not int
            or relation["arity"] != 2
            or relation.get("slot_entities") != slots
            or ("derivation" in relation) != (expected["kind"] == "derived_relation")
        ):
            raise ValueError("compiled relation roles differ from static contract")
    return pipe._rulespec_program_maps(payload)


@_rulespec_routing_cache_scope()
def reconstruct_mhrtc_request(
    case: dict,
    compiled_payload: dict,
    *,
    pipe,
    rules_file: Path,
    case_index: int,
) -> MhrtcRequestPlan:
    """Use the ordinary dataset builder on a caller-bound frozen M case.

    No case-name/hash whitelist, engine call or claimed capture authentication.
    The caller must strict-parse compiled JSON and seal these inputs externally.
    """
    _validate_case(case)
    if type(case_index) is not int or not 1 <= case_index <= 1024:
        raise ValueError("positive bounded exact case ordinal required")
    if (
        not rules_file.as_posix().endswith("/" + POLICY_PATH)
        or not rules_file.is_file()
    ):
        raise ValueError("canonical materialized policy path required")
    derived, parameters = _compiled_maps(pipe, compiled_payload)
    output_names = list(case["output"])
    if "relations" not in case:
        if not all(name in parameters for name in output_names):
            raise ValueError(
                "public parameter declarations absent from compiled program"
            )
        return MhrtcRequestPlan(None, (), tuple(output_names))
    if not all(name in derived for name in output_names):
        raise ValueError("graph cases require only compiled derived queries")
    entities = {derived[name]["entity"] for name in output_names}
    if len(entities) != 1 or not entities <= {"Family", "Claim"}:
        raise ValueError("one supported MHRtC query entity required")
    query_entity = next(iter(entities))
    _validate_facts(case["input"], query_entity)
    period = pipe._coerce_rulespec_period(case["period"])
    query_id = pipe._rulespec_case_query_entity_id(case, query_entity, case_index)
    slots = _rulespec_declared_relation_slots(compiled_payload)
    relations = compiled_payload["program"].get("relations")
    if type(relations) is not list:
        raise ValueError("compiled relation list required")
    directions = executable_relation_directions(
        derived, output_names, period, query_entity, slots, relations
    )
    dataset = pipe._build_rulespec_dataset(
        case["input"],
        case_tables=case["tables"],
        explicit_relations_case=case,
        compiled_relations=relations,
        period=period,
        query_entity=query_entity,
        query_entity_id=query_id,
        require_legal_input_keys=True,
        legal_ids_by_friendly_name=pipe._rulespec_legal_ids_by_friendly_output_name(
            compiled_payload
        ),
        module_target=_TARGET,
        declared_relation_names=_rulespec_declared_relation_names(compiled_payload),
        declared_relation_slots=slots,
        executable_directions=directions,
        input_entities=pipe._rulespec_companion_input_entities(
            rules_file, case["input"]
        ),
        table_entity_ids={},
    )
    return MhrtcRequestPlan(
        {
            "mode": "explain",
            "dataset": dataset,
            "queries": [
                {"entity_id": query_id, "period": period, "outputs": output_names}
            ],
        },
        tuple(output_names),
        (),
    )
