"""Closed physical allowance lists, with shared facts and alternative groups.

This reader deliberately supports only repeated-object, same-period lists with
coordinated action and billing facts. It neither guesses unread English nor
certifies raw caller facts. Unsupported source text keeps the existing checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .signed_condition_paths import SignedPath

_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
_QUALIFIERS = re.compile(
    r"\b(?:not|no|never|without|unless|except|provided|if|only|neither|nor|but|until|while)\b",
    re.I,
)


def _words(text: str) -> frozenset[str]:
    # Exact spelling modulo ordinary plurals; no semantic synonym expansion.
    return frozenset(
        word[:-1] if word.endswith("s") else word
        for word in re.findall(r"[a-z]+", text.lower())
    )


def _number(text: str) -> int | None:
    return (
        int(text) if text.isdigit() and len(text) <= 2 else _NUMBERS.get(text.lower())
    )


def _expense_words(text: str) -> frozenset[str]:
    return frozenset("expense" if word == "cost" else word for word in _words(text))


@dataclass(frozen=True)
class AllowanceBranch:
    source: str
    action: str
    causes: tuple[str, ...]


@dataclass(frozen=True)
class PhysicalAllowanceCondition:
    source: str
    shared: str
    objects: str
    minimum: int
    period: str
    branches: tuple[AllowanceBranch, ...]

    @property
    def gate_count(self) -> int:
        return max(3 + bool(branch.causes) for branch in self.branches)

    def owns_excerpts(self, excerpts: tuple[str, ...]) -> bool:
        """Require exact unambiguous source coverage, not invented proof text."""
        normalized = " ".join(self.source.split())
        proofs = tuple(" ".join(excerpt.split()) for excerpt in excerpts)
        if not proofs or any(not p or normalized.count(p) != 1 for p in proofs):
            return False
        demands = (self.shared, *(branch.source for branch in self.branches))
        return all(
            any(" ".join(demand.split()) in proof for proof in proofs)
            for demand in demands
        )

    def corroborates(
        self, paths: tuple[SignedPath, ...] | None, inputs: dict, *, entity: str | None
    ) -> bool:
        if paths is None or not paths:
            return False

        def same_object(name):
            item = inputs.get(name, {})
            return (
                str(item.get("period", "")).lower() == self.period
                # This closed reader consumes predicate-shaped identifiers,
                # not arbitrary prose contracts. Unknown descriptions cannot
                # silently redefine a count, object set, or causal assertion.
                and not str(item.get("description", "")).strip()
            )

        objects = self.objects.lower()
        singular = objects[:-1] if objects.endswith("s") else objects
        count_suffix = "_distinct_expense_" + singular + "_count"
        for path in paths:
            counts = {
                f.name
                for f in path.facts
                if str(inputs.get(f.name, {}).get("dtype", "")).lower() == "count"
                and same_object(f.name)
                and f.name.endswith(count_suffix)
                and _words(objects)
                <= _words(f.name[: -len(count_suffix)].replace("_", " "))
                and (
                    f.positive
                    and (f.relation, f.threshold)
                    in {("ge", self.minimum), ("gt", self.minimum - 1)}
                    or not f.positive
                    and (f.relation, f.threshold)
                    in {("lt", self.minimum), ("le", self.minimum - 1)}
                )
            }
            positive = {
                f.name
                for f in path.facts
                if f.positive
                and f.relation == "boolean"
                and str(inputs.get(f.name, {}).get("dtype", "")).lower()
                in {"boolean", "bool"}
                and same_object(f.name)
            }
            matched = False
            for count in counts:
                if not entity or inputs[count].get("entity") != entity:
                    continue
                group = count[: -len(count_suffix)]
                names = {
                    name for name in positive if inputs[name].get("entity") == entity
                }
                billed = names & {group + "_billed_for_those_" + objects}
                for branch in self.branches:
                    connectors = (
                        ("_between_those_",)
                        if branch.action.lower() == "moves"
                        else ("_those_",)
                    )
                    actions = names & {
                        group + "_" + branch.action.lower() + connector + objects
                        for connector in connectors
                    }
                    causes = names & {
                        group + "_because_of_" + "_".join(cause.lower().split())
                        for cause in branch.causes
                    }
                    # Distinct facts cannot be replaced by one aggregate flag.
                    if any(
                        a != b
                        and (not branch.causes or any(c not in {a, b} for c in causes))
                        for a in actions
                        for b in billed
                    ):
                        matched = True
            if not matched:
                return False
        return True


def read_physical_allowance_condition(
    source: str, *, limit: int = 32
) -> PhysicalAllowanceCondition | None:
    """Consume an entire bounded repeated-object allowance-list source unit."""
    if len(source) > 12000:
        return None
    lines = tuple(line.strip() for line in source.splitlines() if line.strip())
    if not 3 <= len(lines) <= min(limit, 10):
        return None
    header = re.fullmatch(
        r"(?P<effect>[A-Za-z ]+) for (?P<subject>[A-Za-z ]+) when "
        r"(?P<shared>expenses of (?P<count>one|two|three|four|five|[1-9][0-9]?) or more "
        r"(?P<objects>[A-Za-z-]+) occur in the same (?P<period>month|year)) "
        r"are allowable(?: for [A-Z][A-Z ]*)? only when any of the following applies:",
        lines[0],
        re.I,
    )
    if header is None or _QUALIFIERS.search(header["effect"] + " " + header["subject"]):
        return None
    minimum = _number(header["count"])
    if minimum is None or minimum < 2:
        return None
    branches = []
    for line in lines[1:]:
        branch = re.fullmatch(
            r"[●•]\s+(?P<subject>[A-Za-z ]+) (?P<action>[A-Za-z]+) and is billed for "
            r"(?P<expense>[A-Za-z ]+) (?:on|for) more than (?P<count>one|two|three|four|five|[1-9][0-9]?) "
            r"(?P<objects>[A-Za-z-]+) (?P<tail>[^.;:!?]+)\.",
            line,
            re.I,
        )
        if branch is None or _QUALIFIERS.search(line):
            return None
        if _expense_words(branch["expense"]) != _expense_words(header["effect"]):
            return None
        if (
            branch["subject"].lower() != header["subject"].lower()
            or _words(branch["objects"]) != _words(header["objects"])
            or _number(branch["count"]) != minimum - 1
        ):
            return None
        tail = branch["tail"]
        causes = ()
        if tail.lower() == "during the same " + header["period"].lower():
            pass
        elif tail.lower().startswith("because of "):
            cause = tail[len("because of ") :]
            alternatives = re.fullmatch(
                r"([A-Za-z]+) or ([A-Za-z]+)( away from [A-Za-z]+)", cause, re.I
            )
            if alternatives:
                causes = (
                    alternatives[1] + alternatives[3],
                    alternatives[2] + alternatives[3],
                )
            elif re.fullmatch(r"[A-Za-z]+ away from [A-Za-z]+", cause, re.I):
                causes = (cause,)
            else:
                return None
        else:
            return None
        branches.append(AllowanceBranch(line[2:].strip(), branch["action"], causes))
    if len(set(branches)) != len(branches):
        return None
    return PhysicalAllowanceCondition(
        source,
        header["shared"],
        header["objects"],
        minimum,
        header["period"].lower(),
        tuple(branches),
    )
