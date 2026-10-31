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


def _expression(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    kind = value.get("kind")
    if kind in {"input", "derived"}:
        return _keys(value, {"kind", "name"}) and _name(value["name"])
    if kind == "literal":
        return _keys(value, {"kind", "value"}) and _literal(value["value"])
    if kind in {"add", "min", "max", "and", "or"}:
        return (
            _keys(value, {"kind", "items"})
            and isinstance(value["items"], list)
            and bool(value["items"])
            and all(_expression(item) for item in value["items"])
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
            and _expression(value["left"])
            and _expression(value["right"])
        )
    if kind == "not":
        return _keys(value, {"kind", "item"}) and _expression(value["item"])
    if kind == "if":
        return _keys(value, {"kind", "condition", "then_expr", "else_expr"}) and all(
            _expression(value[key]) for key in ("condition", "then_expr", "else_expr")
        )
    if kind == "parameter_lookup":
        return (
            _keys(value, {"kind", "parameter", "index"})
            and _name(value["parameter"])
            and _expression(value["index"])
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
    """Recognize strict scalar reads in formats 2 and 3, not execution outcomes.

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
        {"module", "units"},
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
    # Entire lowered program, including imports and inactive versions. Derived
    # relation predicates are deliberately outside this first recognized subset.
    if not all(
        _keys(r, {"name", "arity", "slot_entities"})
        and _name(r["name"])
        and type(r["arity"]) is int
        and r["arity"] > 0
        and isinstance(r["slot_entities"], list)
        and len(r["slot_entities"]) == r["arity"]
        and all(_name(entity) for entity in r["slot_entities"])
        for r in program["relations"]
    ):
        return False
    try:
        return (
            bool(program["derived"])
            and all(_derived(r) for r in program["derived"])
            and all(_parameter(p) for p in program["parameters"])
        )
    except (RecursionError, TypeError, KeyError):
        return False
