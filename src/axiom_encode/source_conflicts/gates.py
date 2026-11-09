"""Deterministic gates on a link, run before any household is compared.

A link is the weak point of the compare step: a wrong name or a missing input
gives a result that looks like a finding. Every gate here reads only the link
and the two compiled artifacts, and every failure is a clear error.
"""

from __future__ import annotations

from collections.abc import Mapping

from .artifact import BOOL, NUMERIC, Artifact
from .link import SIDES, Link

_BOOL_DTYPES = ("judgment", "bool", "boolean")


def _period_name(value: object) -> str:
    # The artifact writes "TaxYear" where a request writes "tax_year".
    return str(value).replace("_", "").lower()


class LinkGateError(ValueError):
    """The link does not fit the compiled modules. ``problems`` lists why."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__(
            "the link failed its gates:\n"
            + "\n".join(f"- {problem}" for problem in self.problems)
        )


def side_inputs(
    link: Link, side: str, artifact: Artifact
) -> dict[str, tuple[str, str]]:
    """Request name -> (supplier group, supplier name) for one side.

    A fact supplies the input it maps on this side. A derived or fixed input
    is supplied to every side whose artifact accepts its name.
    """

    supplied: dict[str, tuple[str, str]] = {}
    for fact in link.facts.values():
        if side in fact.inputs:
            supplied[fact.inputs[side]] = ("facts", fact.name)
    for group, entries in (("derived", link.derived), ("fixed", link.fixed)):
        for name in entries:
            if name in artifact.slot_by_request_name:
                supplied[name] = (group, name)
    return supplied


def check_link(link: Link, artifacts: Mapping[str, Artifact]) -> list[str]:
    """Run the static gates. Raise :class:`LinkGateError`; return warnings."""

    problems: list[str] = []
    warnings: list[str] = []
    for side in SIDES:
        spec, artifact = link.sides[side], artifacts[side]
        queried = [*spec.outputs, *spec.guards]
        for group, names in (("outputs", spec.outputs), ("guards", spec.guards)):
            for name in names:
                rule = artifact.derived(name)
                if rule is None:
                    problems.append(
                        f"{side}.{group}: '{name}' is not a derived rule of the compiled "
                        f"module {spec.module}"
                    )
                    continue
                if str(rule.get("dtype")).lower() not in _BOOL_DTYPES:
                    problems.append(
                        f"{side}.{group}: '{name}' has dtype {rule.get('dtype')}; the compare "
                        "step combines yes/no rules only"
                    )
                if rule.get("entity") and rule["entity"] != link.entity:
                    problems.append(
                        f"{side}.{group}: '{name}' is a rule of entity {rule['entity']}, "
                        f"but the link's entity is {link.entity}"
                    )
                period = rule.get("period")
                if period and _period_name(period) != _period_name(
                    link.period["period_kind"]
                ):
                    problems.append(
                        f"{side}.{group}: '{name}' has period {period}, but the link's "
                        f"period_kind is {link.period['period_kind']}"
                    )

        for fact in link.facts.values():
            name = fact.inputs.get(side)
            if name is not None and name not in artifact.slot_by_request_name:
                problems.append(
                    f"facts.{fact.name}.{side}: '{name}' is not an input of the compiled "
                    f"module {spec.module}"
                )

        derived_names, _, required_slots = artifact.closure(queried)
        supplied = side_inputs(link, side, artifact)
        for slot in required_slots:
            request_name = artifact.request_name_by_slot.get(slot)
            if request_name is None:
                problems.append(
                    f"side {side}: the rules read input slot '{slot}', which the "
                    "artifact's input catalog does not list"
                )
            elif request_name not in supplied:
                problems.append(
                    f"side {side}: input '{request_name}' is read by the compared rules "
                    "but no fact, derived input or fixed input supplies it"
                )

        kinds = {name: fact.kind for name, fact in link.facts.items()}
        classes = artifact.input_classes(derived_names)
        required = set(required_slots)
        for request_name, (group, name) in supplied.items():
            slot = artifact.slot_by_request_name.get(request_name)
            if slot is None:
                continue
            if slot not in required:
                warnings.append(
                    f"side {side}: input '{request_name}' ({group}) is not read by the "
                    "compared rules"
                )
                continue
            kind = {
                "facts": kinds.get(name),
                "derived": link.derived[name].kind if group == "derived" else None,
                "fixed": link.fixed[name].kind if group == "fixed" else None,
            }[group]
            used_as = classes.get(slot) or set()
            mine = BOOL if kind == "bool" else NUMERIC
            if used_as and mine not in used_as:
                where = f"facts.{name}" if group == "facts" else f"{group}.{name}"
                problems.append(
                    f"{where}: kind {kind} does not match side {side}, where the rules use "
                    f"'{request_name}' as a {' and '.join(sorted(used_as))} value"
                )

    for group, entries in (("derived", link.derived), ("fixed", link.fixed)):
        for name in entries:
            if not any(name in artifacts[side].slot_by_request_name for side in SIDES):
                problems.append(
                    f"{group}.{name}: not an input of either compiled module"
                )
    if problems:
        raise LinkGateError(problems)
    return warnings
