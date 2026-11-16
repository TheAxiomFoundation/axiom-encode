"""The v33 protected-delta predicate: atomic records, unique assignment, replay.

Inputs have already passed tree totality and authenticated lineage eligibility.
No candidate code, model, private key, or network is involved here.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

from .lineage import CORRECTION, EligibleRecord, PathPolicy
from .manifest import Manifest, manifest_diff
from .refusal import Refusal

type State = tuple[str, str] | None  # (raw blob sha256, mode)

DETAILS = {
    "uncovered-path": "no eligible transition covers this path",
    "inconsistent-chain": "eligible transitions exist but no valid chain",
    "record-cycle": "consumed records admit no execution order",
    "no-valid-execution": "no topological order yields realizable trees",
    "ambiguous-assignment": "more than one valid assignment",
    "inadmissible-entry": "entry mode or type inadmissible",
}


@dataclass(frozen=True)
class Coverage:
    assignment: tuple[tuple[str, tuple[str, ...]], ...]
    unused_eligible_records: tuple[str, ...]
    unprotected_changes: tuple[str, ...]

    def assignment_json(self) -> list[dict]:
        return [
            {"path": path, "record_sha256s": list(records)}
            for path, records in self.assignment
        ]


def _endpoint(transition: dict, side: str) -> State:
    digest = transition[f"{side}_blob_sha256"]
    return None if digest is None else (digest, transition[f"{side}_mode"])


def _projection(manifest: Manifest, policy: PathPolicy) -> dict[str, State]:
    return {p: (digest, mode) for p, mode, digest in manifest if policy.protects(p)}


def _realizable(state: Mapping[str, State]) -> bool:
    paths = set(state)
    return not any(
        "/".join(parts[:end]) in paths
        for path in paths
        for parts in [path.split("/")]
        for end in range(1, len(parts))
    )


def protected_mode_refusal(
    base: Manifest,
    subject: Manifest,
    records: Sequence[EligibleRecord],
    policy: PathPolicy,
) -> Refusal | None:
    """The wall is domain-total, including unchanged paths and unused records."""
    bad = {
        p for p, mode, _ in [*base, *subject] if policy.protects(p) and mode != "100644"
    }
    for record in records:
        for transition in record.body["transitions"]:
            if any(
                transition[f"{side}_mode"] == "100755" for side in ("before", "after")
            ):
                bad.add(transition["path"])
    if bad:
        return Refusal(
            "inadmissible-entry",
            min(bad, key=str.encode),
            DETAILS["inadmissible-entry"],
        )
    return None


def _chains(
    start: State, end: State, edges: Sequence[tuple[str, dict, dict]]
) -> Iterator[tuple[str, ...]]:
    # Finite simple paths, ordered by record address. Neither record nor state
    # can repeat. Corrections bind their immediate candidate-local predecessor.
    stack = [(start, (), frozenset([start]))]
    while stack:
        state, chain, visited = stack.pop()
        if state == end:
            yield chain
            continue
        for address, transition, body in reversed(edges):
            after = _endpoint(transition, "after")
            if (
                address in chain
                or after in visited
                or _endpoint(transition, "before") != state
            ):
                continue
            if body["schema"] == CORRECTION:
                predecessor = body["predecessor_record_sha256"]
                if predecessor is not None and (not chain or predecessor != chain[-1]):
                    continue
            stack.append((after, (*chain, address), visited | {after}))


def _dependencies(
    assignment: Mapping[str, tuple[str, ...]],
) -> dict[str, frozenset[str]]:
    parents: dict[str, set[str]] = {}
    for chain in assignment.values():
        for index, address in enumerate(chain):
            parents.setdefault(address, set()).update(
                chain[index - 1 : index] if index else ()
            )
    return {address: frozenset(values) for address, values in parents.items()}


def _acyclic(parents: Mapping[str, frozenset[str]]) -> bool:
    remaining = set(parents)
    while remaining:
        ready = {address for address in remaining if not (parents[address] & remaining)}
        if not ready:
            return False
        remaining -= ready
    return True


def _can_replay(
    base: dict[str, State],
    subject: dict[str, State],
    parents: Mapping[str, frozenset[str]],
    bodies: Mapping[str, dict],
) -> bool:
    # One arbitrary topological ordering is insufficient: deleting a terminal
    # before adding its children can succeed when the opposite order cannot.
    # For a fixed executed-record set, endpoint chains determine the same state;
    # memoizing that set therefore preserves the existential predicate.
    stack = [(frozenset(), base)]
    visited = set()
    while stack:
        done, state = stack.pop()
        if done in visited:
            continue
        visited.add(done)
        if len(done) == len(parents):
            if state == subject:
                return True
            continue
        for address in sorted(parents.keys() - done, reverse=True):
            if not parents[address] <= done:
                continue
            transitions = bodies[address]["transitions"]
            if any(state.get(t["path"]) != _endpoint(t, "before") for t in transitions):
                continue
            next_state = dict(state)
            for transition in transitions:
                after = _endpoint(transition, "after")
                if after is None:
                    next_state.pop(transition["path"], None)
                else:
                    next_state[transition["path"]] = after
            if _realizable(next_state):
                stack.append((done | {address}, next_state))
    return False


def compute_coverage(
    base: Manifest,
    subject: Manifest,
    records: Sequence[EligibleRecord],
    policy: PathPolicy,
) -> Coverage | Refusal:
    """Return the sole whole-record assignment, or the first normative refusal.

    Runtime limits belong to the invoking worker; interruption must never be
    translated into a pass. Search stops only after proving a second valid
    assignment. An unusable retry does not create ambiguity.
    """
    before, after = _projection(base, policy), _projection(subject, policy)
    changed = [entry.path for entry in manifest_diff(base, subject)]
    protected = [path for path in changed if policy.protects(path)]
    unprotected = tuple(path for path in changed if not policy.protects(path))
    bodies = {r.body_sha256: r.body for r in records}
    edges: dict[str, list[tuple[str, dict, dict]]] = {path: [] for path in protected}
    for address, body in sorted(bodies.items()):
        for transition in body["transitions"]:
            if transition["path"] in edges:
                edges[transition["path"]].append((address, transition, body))
    # Whole-domain check before any chain check, per the refusal precedence.
    for path in protected:
        if not edges[path]:
            return Refusal("uncovered-path", path, DETAILS["uncovered-path"])
    choices = {}
    permanently_unusable = {
        address
        for address, body in bodies.items()
        if any(t["path"] not in edges for t in body["transitions"])
    }
    for path in protected:
        choices[path] = tuple(
            _chains(
                before.get(path),
                after.get(path),
                [edge for edge in edges[path] if edge[0] not in permanently_unusable],
            )
        )
        if not choices[path]:
            return Refusal("inconsistent-chain", path, DETAILS["inconsistent-chain"])
    atomic = acyclic = False
    solution = None
    for chains in product(*(choices[path] for path in protected)):
        assignment = dict(zip(protected, chains, strict=True))
        consumed = {address for chain in chains for address in chain}
        incomplete = {
            address
            for address in consumed
            if any(
                address not in assignment.get(t["path"], ())
                for t in bodies[address]["transitions"]
            )
        }
        if incomplete:
            continue
        atomic = True
        parents = _dependencies(assignment)
        if not _acyclic(parents):
            continue
        acyclic = True
        if not _can_replay(before, after, parents, bodies):
            continue
        if solution is not None:
            return Refusal(
                "ambiguous-assignment", None, DETAILS["ambiguous-assignment"]
            )
        solution = Coverage(
            tuple(assignment.items()),
            tuple(sorted(bodies.keys() - consumed)),
            unprotected,
        )
    if solution is not None:
        return protected_mode_refusal(base, subject, records, policy) or solution
    code = (
        "inconsistent-chain"
        if not atomic
        else "record-cycle"
        if not acyclic
        else "no-valid-execution"
    )
    return Refusal(
        code,
        None,  # Remaining atomic contradictions are global, not one failed retry.
        DETAILS[code],
    )
