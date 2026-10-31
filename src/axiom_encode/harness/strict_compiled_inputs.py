"""Conservative structural recognition of compiled programs with strict reads.

This does not evaluate formulas or certify companion outcomes. Unknown/default-
bearing IR keeps the existing conservative source-only input check. Only the
normal compiler's artifact is used by the caller, before ordinary execution.
"""


def _keys(value: object, required: set[str], optional: set[str] = frozenset()) -> bool:
    return isinstance(value, dict) and required <= value.keys() <= required | optional


def _name(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _literal(value: object) -> bool:
    if not _keys(value, {"kind", "value"}):
        return False
    kind, item = value["kind"], value["value"]
    if kind == "bool":
        return isinstance(item, bool)
    if kind == "integer":
        return type(item) is int
    if kind in {"decimal", "text", "date"}:
        return isinstance(item, str)
    return False


def _term(value: object, depth: int) -> bool:
    if isinstance(value, str):
        return value in {"anchor", "candidate"}
    return (
        _keys(value, {"var"})
        and type(value["var"]) is int
        and 0 <= value["var"] < depth
    )


def _expression(value: object, *, predicate: bool = False, depth: int = 0) -> bool:
    if not isinstance(value, dict):
        return False
    kind = value.get("kind")
    if kind in {"input", "derived"}:
        return _keys(value, {"kind", "name"}) and _name(value["name"])
    if kind in {"role_input", "role_derived"}:
        return (
            predicate
            and _keys(value, {"kind", "name", "term"})
            and _name(value["name"])
            and _term(value["term"], depth)
        )
    if kind == "literal":
        return _keys(value, {"kind", "value"}) and _literal(value["value"])
    if kind in {"add", "min", "max", "and", "or"}:
        return (
            _keys(value, {"kind", "items"})
            and isinstance(value["items"], list)
            and bool(value["items"])
            and all(
                _expression(item, predicate=predicate, depth=depth)
                for item in value["items"]
            )
        )
    if kind in {"sub", "mul", "div", "comparison"}:
        fields = {"kind", "left", "right"}
        if kind == "comparison":
            fields.add("op")
        return (
            _keys(value, fields)
            and (
                kind != "comparison"
                or value["op"] in {"eq", "ne", "lt", "lte", "gt", "gte"}
            )
            and _expression(value["left"], predicate=predicate, depth=depth)
            and _expression(value["right"], predicate=predicate, depth=depth)
        )
    if kind == "not":
        return _keys(value, {"kind", "item"}) and _expression(
            value["item"], predicate=predicate, depth=depth
        )
    if kind == "if":
        return _keys(value, {"kind", "condition", "then_expr", "else_expr"}) and all(
            _expression(value[key], predicate=predicate, depth=depth)
            for key in ("condition", "then_expr", "else_expr")
        )
    if kind == "parameter_lookup":
        return (
            _keys(value, {"kind", "parameter", "index"})
            and _name(value["parameter"])
            and _expression(value["index"], predicate=predicate, depth=depth)
        )
    if kind == "relation_member":
        return predicate and _keys(
            value, {"kind", "relation", "current_slot", "related_slot"}
        ) and _coordinates(value)
    if kind == "relation_atom":
        return (
            predicate
            and _keys(value, {"kind", "relation", "terms"})
            and _name(value["relation"])
            and isinstance(value["terms"], list)
            and bool(value["terms"])
            and all(_term(term, depth) for term in value["terms"])
        )
    if kind == "same_entity":
        return (
            predicate
            and _keys(value, {"kind", "left", "right", "negated"})
            and type(value["negated"]) is bool
            and _term(value["left"], depth)
            and _term(value["right"], depth)
        )
    if kind == "exists":
        if not (
            predicate
            and _keys(value, {"kind", "relation", "terms", "variable_slot", "body"})
            and _name(value["relation"])
            and isinstance(value["terms"], list)
            and type(value["variable_slot"]) is int
            and 0 <= value["variable_slot"] < len(value["terms"])
        ):
            return False
        # A generator marks its newly bound slot with Var(0). Other generator
        # terms read the enclosing environment; the body adds one binding.
        return all(
            (_term(term, 1) and term == {"var": 0})
            if slot == value["variable_slot"]
            else _term(term, depth)
            for slot, term in enumerate(value["terms"])
        ) and _expression(value["body"], predicate=True, depth=depth + 1)
    if kind == "count_related":
        return (
            _keys(value, {"kind", "relation", "current_slot", "related_slot", "where"})
            and _coordinates(value)
            # Aggregate children have no relation-role context, even when the
            # aggregate occurs inside a derived relation's predicate.
            and (value["where"] is None or _expression(value["where"]))
        )
    if kind == "sum_related":
        return (
            _keys(
                value,
                {"kind", "relation", "current_slot", "related_slot", "value", "where"},
            )
            and _name(value["relation"])
            and type(value["current_slot"]) is int
            and type(value["related_slot"]) is int
            and value["current_slot"] >= 0
            and value["related_slot"] >= 0
            and _keys(value["value"], {"kind", "name"})
            and value["value"]["kind"] in {"input", "derived"}
            and _name(value["value"]["name"])
            and (value["where"] is None or _expression(value["where"]))
        )
    # Includes input_or_else and future operators: no default-free assumption.
    return False


def _coordinates(value: dict) -> bool:
    return (
        _name(value["relation"])
        and type(value["current_slot"]) is int
        and type(value["related_slot"]) is int
        and value["current_slot"] >= 0
        and value["related_slot"] >= 0
        and value["current_slot"] != value["related_slot"]
    )


def _relation(value: object) -> bool:
    if not (
        _keys(value, {"name", "arity", "slot_entities"}, {"derivation"})
        and _name(value["name"])
        and type(value["arity"]) is int
        and value["arity"] > 0
        and isinstance(value["slot_entities"], list)
        and len(value["slot_entities"]) == value["arity"]
        and all(_name(entity) for entity in value["slot_entities"])
    ):
        return False
    derivation = value.get("derivation")
    if derivation is None:
        return True
    if not (
        value["arity"] == 2
        and _keys(
            derivation,
            {"current_slot", "related_slot", "predicate"},
            {"source_relation", "source_path", "entity", "member_relation", "slot_entities"},
        )
        and type(derivation["current_slot"]) is int
        and type(derivation["related_slot"]) is int
        and {derivation["current_slot"], derivation["related_slot"]} == {0, 1}
    ):
        return False
    slots = derivation.get("slot_entities", [])
    if not isinstance(slots, list) or (
        slots and (len(slots) != 2 or not all(_name(kind) for kind in slots))
    ):
        return False
    if any(
        derivation.get(key) is not None and not _name(derivation[key])
        for key in ("entity", "member_relation")
    ):
        return False
    source = derivation.get("source_relation")
    path = derivation.get("source_path")
    if _name(source):
        if path is not None:
            return False
    elif source is not None or not (
        isinstance(path, list)
        and bool(path)
        and len(slots) == 2
        and not derivation.get("entity")
        and not derivation.get("member_relation")
        and all(
            _keys(step, {"relation", "from_slot", "to_slot"})
            and _name(step["relation"])
            and type(step["from_slot"]) is int
            and type(step["to_slot"]) is int
            and {step["from_slot"], step["to_slot"]} == {0, 1}
            for step in path
        )
    ):
        return False
    return _expression(derivation["predicate"], predicate=True)


_SOURCE_FIELDS = {"id", "source", "source_url", "corpus_citation_path", "unit"}


def _derived(rule: object) -> bool:
    if not _keys(
        rule,
        {"name", "entity", "dtype", "period", "semantics", "expr", "versions"},
        _SOURCE_FIELDS,
    ):
        return False
    if rule["semantics"] not in {"scalar", "judgment"} or not _expression(rule["expr"]):
        return False
    versions = rule["versions"]
    return isinstance(versions, list) and all(
        _keys(v, {"effective_from", "semantics", "expr"}, {"effective_to"})
        and v["semantics"] in {"scalar", "judgment"}
        and _expression(v["expr"])
        for v in versions
    )


def _parameter(parameter: object) -> bool:
    if not _keys(parameter, {"name", "versions"}, _SOURCE_FIELDS | {"indexed_by"}):
        return False
    index = parameter.get("indexed_by")
    if index is not None and not (
        _name(index) or (isinstance(index, list) and all(_name(x) for x in index))
    ):
        return False
    versions = parameter["versions"]
    return (
        isinstance(versions, list)
        and bool(versions)
        and all(
            _keys(v, {"effective_from", "values"}, {"effective_to"})
            and isinstance(v["values"], dict)
            and all(_literal(item) for item in v["values"].values())
            for v in versions
        )
    )


def compiled_program_has_strict_input_reads(payload: object) -> bool:
    """Recognize strict scalar and predicate reads in formats 2 and 3.

    Format 3 adds an inert interface and producer identities. The engine checks
    them when it loads the artifact; all executable expressions still pass the
    same structural walk below.
    """
    if not isinstance(payload, dict):
        return False
    version = payload.get("artifact_format_version")
    if type(version) is not int or version not in {2, 3}:
        return False
    identities = {"engine_build", "source_closure"} if version == 3 else set()
    if not _keys(
        payload,
        {"artifact_format_version", "engine_version", "program", "metadata"},
        identities,
    ):
        return False
    if version == 3 and not _keys(payload.get("engine_build"), {"commit", "dirty"}):
        return False
    program = payload["program"]
    interface = {"interface"} if version == 3 else set()
    if not _keys(
        program,
        {"derived", "parameters", "relations"} | interface,
        {"module", "units"} | ({"input_declarations"} if version == 3 else set()),
    ):
        return False
    if version == 3 and not (
        _keys(program["interface"], {"entities", "inputs", "relations"})
        and all(isinstance(value, list) for value in program["interface"].values())
    ):
        return False
    if not all(
        isinstance(program[key], list) for key in ("derived", "parameters", "relations")
    ):
        return False
    try:
        declarations = program.get("input_declarations", [])
        if not isinstance(declarations, list) or not all(
            _keys(item, {"input", "election"})
            and _name(item["input"])
            and _keys(item["election"], {"actor", "provision"})
            and all(_name(v) for v in item["election"].values())
            for item in declarations
        ):
            return False
        # Walk the entire compiled closure, including imported predicates and
        # inactive scalar versions. Unknown or default-bearing nodes still refuse.
        return (
            bool(program["derived"])
            and all(_relation(r) for r in program["relations"])
            and all(_derived(r) for r in program["derived"])
            and all(_parameter(p) for p in program["parameters"])
        )
    except (RecursionError, TypeError, KeyError):
        return False
