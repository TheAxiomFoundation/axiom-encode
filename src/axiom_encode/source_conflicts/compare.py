"""The compare step: run two linked modules on generated households.

``compare`` compiles both modules, runs the link's gates, builds the household
grid, runs every household on each side and writes the conflict record. A
household for which the two sides give different results is a witness.

A witness proves that two *encodings* differ. Whether the two *sources*
differ is a review decision, so the record's ``review_state`` is always
``unreviewed`` here and each side carries the citations and excerpts of the
rules that decided, for a reviewer to open the texts.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

import yaml

from .artifact import Artifact, ArtifactError
from .gates import LinkGateError, check_link, side_inputs
from .link import SIDES, VALUE, Link, canonical_decimal
from .values import (
    SAMPLING_METHOD,
    combination,
    generate_values,
    grid_size,
    select_indices,
)

RECORD_FORMAT = "axiom-source-conflict-record/v1"
REVIEW_STATES = (
    "unreviewed",
    "source_conflict",
    "encoding_error",
    "refinement_or_option",
)
BATCH = 400


class CompareError(RuntimeError):
    """The engine gave a result the compare step cannot read."""


class EngineLike(Protocol):
    commit: str

    def compile(
        self, module: Path, rulespec_root: Path, output: Path
    ) -> dict[str, Any]: ...

    def run(self, artifact: Path, request: dict[str, Any]) -> dict[str, Any]: ...


def repository_state(rulespec_root: Path) -> dict[str, Any]:
    """The commit of the RuleSpec checkout, and whether tracked files differ from it."""

    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(rulespec_root), *args],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def _engine_value(kind: str, value: object) -> dict[str, Any]:
    if kind == "decimal":
        return {"kind": "decimal", "value": str(value)}
    return {"kind": kind, "value": value}


def _truth(output: Mapping[str, Any], name: str) -> bool:
    if output.get("kind") == "judgment" and output.get("outcome") in (
        "holds",
        "not_holds",
    ):
        return output["outcome"] == "holds"
    value = output.get("value")
    if isinstance(value, dict) and isinstance(value.get("value"), bool):
        return value["value"]
    raise CompareError(
        f"the engine result of '{name}' is not a yes/no value: "
        f"{json.dumps(output, sort_keys=True)[:300]}"
    )


def _number(output: Mapping[str, Any], name: str) -> Decimal:
    value = output.get("value")
    inner = value.get("value") if isinstance(value, dict) else None
    if isinstance(inner, (int, str)) and not isinstance(inner, bool):
        try:
            number = Decimal(str(inner))
        except ArithmeticError:
            number = None
        if number is not None and number.is_finite():
            return number
    raise CompareError(
        f"the engine result of '{name}' is not a number: "
        f"{json.dumps(output, sort_keys=True)[:300]}"
    )


def _is_zero(value: object) -> bool:
    return value is False or (not isinstance(value, bool) and Decimal(str(value)) == 0)


class _ModuleRules:
    """Citations and proof excerpts of rules, read from the module files."""

    def __init__(self, rulespec_root: Path):
        self.root = Path(rulespec_root)
        self._modules: dict[str, dict[str, Any]] = {}

    def _module(self, prefix: str) -> dict[str, Any]:
        if prefix not in self._modules:
            rules: dict[str, Any] = {}
            path = self.root / (prefix.replace(":", "/", 1) + ".yaml")
            try:
                document = yaml.safe_load(path.read_text(encoding="utf-8"))
                for rule in document.get("rules") or []:
                    if isinstance(rule, dict) and isinstance(rule.get("name"), str):
                        rules[rule["name"]] = rule
            except (OSError, UnicodeError, yaml.YAMLError, AttributeError):
                rules = {}
            self._modules[prefix] = rules
        return self._modules[prefix]

    def atoms(self, rule_id: str) -> list[dict[str, Any]]:
        prefix, _, name = rule_id.partition("#")
        rule = self._module(prefix).get(name) or {}
        atoms = ((rule.get("metadata") or {}).get("proof") or {}).get("atoms") or []
        found: list[dict[str, Any]] = []
        for atom in atoms:
            source = atom.get("source") if isinstance(atom, dict) else None
            if not isinstance(source, dict):
                continue
            entry = {
                "kind": atom.get("kind"),
                "corpus_citation_path": source.get("corpus_citation_path"),
                "excerpt": source.get("excerpt"),
            }
            if entry not in found:
                found.append(entry)
        return found


def _rule_records(
    artifact: Artifact, deciding: Sequence[str], modules: _ModuleRules
) -> dict[str, Any]:
    """Every rule the deciding rules depend on, with its citation and excerpts."""

    derived_names, parameter_names, _ = artifact.closure(deciding)
    # A deciding rule can be a parameter itself (a number read from the artifact).
    parameter_names = list(
        dict.fromkeys(
            [
                *(
                    artifact.parameter(rule_id)["name"]
                    for rule_id in deciding
                    if artifact.parameter(rule_id) is not None
                ),
                *parameter_names,
            ]
        )
    )
    records: dict[str, Any] = {}
    for kind, names in (("derived", derived_names), ("parameter", parameter_names)):
        for name in names:
            rule_id = artifact.rule_id(name)
            rule = artifact.rule(rule_id) or {}
            record = {
                "kind": kind,
                "source": rule.get("source"),
                "corpus_citation_path": rule.get("corpus_citation_path"),
                "proof_atoms": modules.atoms(rule_id),
            }
            if kind == "derived":
                record["depends_on"] = artifact.direct_dependencies(name)
            records[rule_id] = record
    return dict(sorted(records.items()))


def compare(
    link: Link,
    *,
    rulespec_root: Path,
    engine: EngineLike,
    workdir: Path,
    repository: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the compare step and return the conflict record.

    Raises :class:`LinkGateError` when the link fails a gate, and
    ``EngineError`` / :class:`CompareError` when the engine fails.
    """

    rulespec_root = Path(rulespec_root)
    start = link.period["start"]
    artifacts: dict[str, Artifact] = {}
    artifact_paths: dict[str, Path] = {}
    module_hashes: dict[str, str] = {}
    missing = []
    for side in SIDES:
        module = rulespec_root / link.sides[side].module
        if not module.is_file():
            missing.append(
                f"{side}.module: {link.sides[side].module} is not a file in {rulespec_root}"
            )
            continue
        module_hashes[side] = hashlib.sha256(module.read_bytes()).hexdigest()
        artifact_paths[side] = Path(workdir) / f"{side}.json"
        try:
            artifacts[side] = Artifact(
                engine.compile(module, rulespec_root, artifact_paths[side]), start
            )
        except ArtifactError as error:
            raise CompareError(f"side {side}: {error}") from error
    if missing:
        raise LinkGateError(missing)

    warnings = check_link(link, artifacts)

    # Values: explicit, or generated from the thresholds of both modules.
    fixed_slots = {
        side: {
            artifacts[side].slot_by_request_name[name]: entry.value
            for name, entry in link.fixed.items()
            if name in artifacts[side].slot_by_request_name
        }
        for side in SIDES
    }
    supplied = {side: side_inputs(link, side, artifacts[side]) for side in SIDES}
    thresholds: dict[str, list[dict[str, str]]] = {name: [] for name in link.facts}
    for side in SIDES:
        spec, artifact = link.sides[side], artifacts[side]
        derived_names, _, _ = artifact.closure([*spec.outputs, *spec.guards])
        for threshold in artifact.thresholds(
            derived_names,
            fixed_slots[side],
            bounds=all(link.sides[each].combine == VALUE for each in SIDES),
        ):
            reached: dict[str, None] = {}
            for slot in sorted(threshold.inputs):
                group, name = supplied[side].get(
                    artifact.request_name_by_slot.get(slot, ""), (None, None)
                )
                if group == "facts":
                    reached[name] = None
                elif group == "derived":
                    reached.update(dict.fromkeys(link.derived[name].of))
            for fact_name in reached:
                entry = {
                    "value": canonical_decimal(threshold.value),
                    "side": side,
                    "rule": threshold.rule,
                }
                if entry not in thresholds[fact_name]:
                    thresholds[fact_name].append(entry)
    for entries in thresholds.values():
        entries.sort(
            key=lambda entry: (Decimal(entry["value"]), entry["side"], entry["rule"])
        )

    # A side reads a fact through the input the fact maps, or through a
    # derived input over the fact that the side's artifact accepts.
    # An input that no compared rule reads (a gate warning) does not count.
    read_by: dict[str, set[str]] = {side: set() for side in SIDES}
    for side in SIDES:
        spec, artifact = link.sides[side], artifacts[side]
        _, _, slots = artifact.closure([*spec.outputs, *spec.guards])
        read_names = {artifact.request_name_by_slot.get(slot) for slot in slots}
        for request_name, (group, name) in supplied[side].items():
            if request_name not in read_names:
                continue
            if group == "facts":
                read_by[side].add(name)
            elif group == "derived":
                read_by[side].update(link.derived[name].of)

    fact_names = list(link.facts)
    value_lists: list[list[Any]] = []
    fact_records: dict[str, Any] = {}
    for name, fact in link.facts.items():
        explicit = fact.values is not None
        values = (
            list(fact.values)
            if explicit
            else generate_values(
                fact, [Decimal(entry["value"]) for entry in thresholds[name]]
            )
        )
        value_lists.append(values)
        record: dict[str, Any] = {"kind": fact.kind}
        for side in SIDES:
            record[side] = fact.inputs.get(side)
        record["read_by"] = [side for side in SIDES if name in read_by[side]]
        record["values"] = values
        record["values_from"] = "link" if explicit else "modules"
        if not explicit and fact.kind != "bool":
            record["thresholds"] = thresholds[name]
        fact_records[name] = record

    empty = [name for name, values in zip(fact_names, value_lists) if not values]
    if empty:
        raise LinkGateError(
            [
                f"facts.{name}: no value is left between its min and max, so there "
                "is no household to compare"
                for name in empty
            ]
        )
    size = grid_size(value_lists)
    indices, sampled = select_indices(size, link.max_combinations, link.seed)
    cases = [
        dict(zip(fact_names, combination(index, value_lists))) for index in indices
    ]

    # Constraints: which combinations are not real households.
    broken: list[list[int]] = []  # per case, the constraints it breaks
    for case in cases:
        broken.append(
            [
                number
                for number, constraint in enumerate(link.constraints)
                if not constraint.expression.holds(case)
            ]
        )
    constraint_problems = []
    scope = (
        f"{len(cases)} combinations sampled from {size} "
        "(raise sampling.max_combinations to test more)"
        if sampled
        else f"{len(cases)} combinations"
    )
    for number, constraint in enumerate(link.constraints):
        if all(number in found for found in broken):
            constraint_problems.append(
                f"constraints[{number}] ({constraint.rule}) removes every one of the "
                f"{scope}; it is too strong for these values"
            )
    if not constraint_problems and all(broken):
        constraint_problems.append(
            f"the constraints together remove every one of the {scope}"
        )
    if constraint_problems:
        raise LinkGateError(constraint_problems)

    # Run every combination on each side, the removed ones too: the record
    # reports how many differing households each constraint removes.
    rows: dict[str, list[dict[str, bool]]] = {}
    interval = {"start": link.period["start"], "end": link.period["end"]}
    for side in SIDES:
        spec, artifact = link.sides[side], artifacts[side]
        # A parameter output is a constant of the artifact; the engine is
        # asked for the derived rules only.
        constants = {
            name: artifact.parameter_number(name)[0]
            for name in spec.outputs
            if spec.combine == VALUE and artifact.derived(name) is None
        }
        queried = [
            name for name in (*spec.outputs, *spec.guards) if name not in constants
        ]
        numbers = set(spec.outputs) if spec.combine == VALUE else set()
        side_rows: list[dict[str, Any]] = []
        if not queried:
            rows[side] = [dict(constants) for _ in cases]
            continue
        for offset in range(0, len(cases), BATCH):
            inputs, queries = [], []
            for position in range(offset, min(offset + BATCH, len(cases))):
                case, entity_id = cases[position], f"case:{indices[position]}"
                for request_name, (group, name) in supplied[side].items():
                    if group == "facts":
                        kind, value = link.facts[name].kind, case[name]
                    elif group == "derived":
                        derived = link.derived[name]
                        total = sum(
                            (Decimal(str(case[fact])) for fact in derived.of),
                            Decimal(0),
                        )
                        kind = derived.kind
                        value = (
                            int(total)
                            if kind == "integer"
                            else canonical_decimal(total)
                        )
                    else:
                        kind, value = link.fixed[name].kind, link.fixed[name].value
                    inputs.append(
                        {
                            "name": request_name,
                            "entity": link.entity,
                            "entity_id": entity_id,
                            "interval": interval,
                            "value": _engine_value(kind, value),
                        }
                    )
                queries.append(
                    {
                        "entity_id": entity_id,
                        "period": dict(link.period),
                        "outputs": queried,
                    }
                )
            response = engine.run(
                artifact_paths[side],
                {
                    "mode": "fast",
                    "dataset": {"inputs": inputs, "relations": []},
                    "queries": queries,
                },
            )
            results = response.get("results") if isinstance(response, dict) else None
            if not isinstance(results, list) or len(results) != len(queries):
                raise CompareError(
                    f"side {side}: the engine returned {len(results or [])} results "
                    f"for {len(queries)} households"
                )
            for query, result in zip(queries, results):
                if result.get("entity_id") != query["entity_id"]:
                    raise CompareError(
                        f"side {side}: the engine returned household "
                        f"{result.get('entity_id')} where {query['entity_id']} was asked"
                    )
                outputs = result.get("outputs") or {}
                try:
                    side_rows.append(
                        {
                            **constants,
                            **{
                                name: (_number if name in numbers else _truth)(
                                    outputs[name], name
                                )
                                for name in queried
                            },
                        }
                    )
                except KeyError as error:
                    raise CompareError(
                        f"side {side}: the engine result lacks the output {error}"
                    ) from error
        rows[side] = side_rows

    value_mode = all(link.sides[side].combine == VALUE for side in SIDES)

    def decide(side: str, row: Mapping[str, Any]) -> tuple[Any, list[str]]:
        spec = link.sides[side]
        if spec.combine == VALUE:
            return row[spec.outputs[0]], list(spec.outputs)
        truths = [row[name] for name in spec.outputs]
        if spec.combine == "any":
            result = any(truths)
            deciding = (
                [name for name in spec.outputs if row[name]]
                if result
                else list(spec.outputs)
            )
        else:
            result = all(truths)
            deciding = (
                list(spec.outputs)
                if result
                else [name for name in spec.outputs if not row[name]]
            )
        return result, deciding

    one_side_facts = [
        name for name in fact_names if sum(name in read_by[side] for side in SIDES) == 1
    ]

    def one_side_value(name: str, value: object) -> object:
        # A number that one side reads splits a group by zero or not zero,
        # not by each value, so one difference stays one group.
        if link.facts[name].kind == "bool":
            return value
        return "zero" if _is_zero(value) else "nonzero"

    # Per direction and deciding rules: the values each one-side fact takes
    # in the witnesses. A fact with one value there is a condition.
    conditions_seen: dict[tuple, dict[str, set]] = {}
    groups: dict[tuple, dict[str, Any]] = {}
    run = skipped_by_guard = differ = removed = removed_differ = 0
    removed_differ_by_constraint = [0] * len(link.constraints)
    for position, case in enumerate(cases):
        guarded = all(
            rows[side][position][guard]
            for side in SIDES
            for guard in link.sides[side].guards
        )
        decided = {side: decide(side, rows[side][position]) for side in SIDES}
        if value_mode:
            difference = decided["b"][0] - decided["a"][0]
            differs = guarded and abs(difference) > link.tolerance
            directions = ("lower", "higher") if difference > 0 else ("higher", "lower")
        else:
            differs = guarded and decided["a"][0] != decided["b"][0]
            directions = (decided["a"][0], decided["b"][0])
        if broken[position]:
            removed += 1
            if differs:
                removed_differ += 1
                for number in broken[position]:
                    removed_differ_by_constraint[number] += 1
            continue
        run += 1
        if not guarded:
            skipped_by_guard += 1
            continue
        if not differs:
            continue
        differ += 1
        key = (
            *directions,
            tuple(decided["a"][1]),
            tuple(decided["b"][1]),
            tuple((name, one_side_value(name, case[name])) for name in one_side_facts),
        )
        seen = conditions_seen.setdefault(
            key[:4], {name: set() for name in one_side_facts}
        )
        for name, value in key[4]:
            seen[name].add(value)
        weight = sum(1 for value in case.values() if not _is_zero(value))
        group = groups.get(key)
        if group is None:
            group = groups[key] = {
                "count": 0,
                "weight": None,
                "values": {name: [] for name in fact_names},
            }
        group["count"] += 1
        if value_mode:
            group["differences"] = (
                min(group.get("differences", (difference,))[0], difference),
                max(group.get("differences", (difference,))[-1], difference),
            )
        for name in fact_names:
            if case[name] not in group["values"][name]:
                group["values"][name].append(case[name])
        if group["weight"] is None or weight < group["weight"]:
            group.update(weight=weight, position=position)

    if run and skipped_by_guard == run:
        tested = (
            f"{run} households of the sample of {len(cases)} from {size} "
            "combinations (raise sampling.max_combinations to test more)"
            if sampled
            else f"{run} households"
        )
        raise LinkGateError(
            [
                f"the guards skip every one of the {tested} that meet the "
                "constraints, so nothing is compared; check the guards and the "
                "fixed inputs they read"
            ]
        )

    modules = _ModuleRules(rulespec_root)
    ordered = sorted(
        groups.items(), key=lambda item: (item[1]["weight"], item[1]["position"])
    )

    def shown(value: Any) -> Any:
        return canonical_decimal(value) if isinstance(value, Decimal) else value

    witness_groups = []
    deciding_all: dict[str, dict[str, None]] = {side: {} for side in SIDES}
    for key, group in ordered:
        position = group["position"]
        entry: dict[str, Any] = {
            "direction": {"a": key[0], "b": key[1]},
            "deciding_rules": {"a": list(key[2]), "b": list(key[3])},
            "one_side_facts": dict(key[4]),
            "conditions": [
                {
                    "fact": name,
                    "read_by": next(side for side in SIDES if name in read_by[side]),
                    "value": next(iter(conditions_seen[key[:4]][name])),
                }
                for name in one_side_facts
                if len(conditions_seen[key[:4]][name]) == 1
            ],
            "count": group["count"],
            "witness": {
                "facts": cases[position],
                **{
                    side: {
                        "result": shown(decide(side, rows[side][position])[0]),
                        "outputs": {
                            name: shown(rows[side][position][name])
                            for name in link.sides[side].outputs
                        },
                    }
                    for side in SIDES
                },
            },
            "fact_values": {
                name: [value for value in values if value in group["values"][name]]
                for name, values in zip(fact_names, value_lists)
            },
        }
        if value_mode:
            a_value, b_value = (decide(side, rows[side][position])[0] for side in SIDES)
            entry["witness"]["difference"] = canonical_decimal(b_value - a_value)
            low, high = group["differences"]
            entry["difference"] = {
                "min": canonical_decimal(low),
                "max": canonical_decimal(high),
            }
        witness_groups.append(entry)
        for side, names in (("a", key[2]), ("b", key[3])):
            deciding_all[side].update(dict.fromkeys(names))

    sides: dict[str, Any] = {}
    for side in SIDES:
        spec = link.sides[side]
        sides[side] = {
            "label": spec.label,
            "module": spec.module,
            "module_sha256": module_hashes[side],
            "outputs": list(spec.outputs),
            "combine": spec.combine,
            "guards": list(spec.guards),
            "rules": _rule_records(artifacts[side], list(deciding_all[side]), modules),
        }

    assumptions = [
        {"where": where, "note": note}
        for where, note in (
            *((f"sides.{side}", link.sides[side].note) for side in SIDES),
            *((f"facts.{name}", fact.note) for name, fact in link.facts.items()),
            *((f"derived.{name}", entry.note) for name, entry in link.derived.items()),
            *((f"fixed.{name}", entry.note) for name, entry in link.fixed.items()),
            *(
                (f"constraints[{number}]", constraint.note)
                for number, constraint in enumerate(link.constraints)
            ),
        )
        if note
    ]
    record = {
        "format": RECORD_FORMAT,
        "id": link.id,
        "review_state": "unreviewed",
        "conflict": bool(witness_groups),
        "question": link.question,
        "link": {"sha256": link.sha256},
        "rulespec": dict(
            repository if repository is not None else repository_state(rulespec_root)
        ),
        "engine": {
            "commit": engine.commit,
            "version": artifacts["a"].engine_version,
        },
        "entity": link.entity,
        "period": dict(link.period),
        "sides": sides,
        "facts": fact_records,
        "assumptions": assumptions,
        "constraints": [
            {
                "rule": constraint.rule,
                "note": constraint.note,
                "removed": sum(1 for found in broken if number in found),
                "removed_by_this_alone": sum(
                    1 for found in broken if found == [number]
                ),
                "removed_that_differ": removed_differ_by_constraint[number],
            }
            for number, constraint in enumerate(link.constraints)
        ],
        "counts": {
            "grid": size,
            "combinations": len(cases),
            "removed_by_constraints": removed,
            "removed_that_differ": removed_differ,
            "run": run,
            "skipped_by_guard": skipped_by_guard,
            "differ": differ,
            "witness_groups": len(witness_groups),
        },
        "sampled": sampled,
        "sampling": {
            "method": SAMPLING_METHOD if sampled else None,
            "max_combinations": link.max_combinations,
            "seed": link.seed,
        },
        "warnings": warnings,
        "witness_groups": witness_groups,
    }
    if value_mode:
        # A yes/no record has no such key, so its bytes are unchanged.
        record["comparison"] = {
            "kind": "value",
            "tolerance": canonical_decimal(link.tolerance),
            "unit": next(
                (
                    unit
                    for unit in (
                        (artifacts[side].rule(link.sides[side].outputs[0]) or {}).get(
                            "unit"
                        )
                        for side in SIDES
                    )
                    if unit
                ),
                None,
            ),
        }
        if not link.facts and len(cases) == 1:
            # One evaluation with no fact: give both numbers, conflict or not.
            record["comparison"]["values"] = {
                side: shown(decide(side, rows[side][0])[0]) for side in SIDES
            }
    return record


def render_record(record: Mapping[str, Any]) -> str:
    """The record as the bytes the step writes: stable for the same inputs."""

    return json.dumps(record, indent=2, ensure_ascii=False) + "\n"
