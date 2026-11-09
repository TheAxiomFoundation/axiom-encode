"""The NZ bare-structure differential compares gate obligations as a multiset (#1949).

Two equal obligations used to collapse into one dictionary entry, so dropping
one of them reported no gate change and no failure. Equal obligations are
independent demands in production, so each surplus copy is a removal, and a
surviving equal copy, already accounted to its own twin, cannot certify it.
"""

from __future__ import annotations

import copy
import sys
from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.bare_structure import bare_outline, mask_bare_labels
from axiom_encode.harness.source_completeness import SourceStructureBranch

SCRIPT = Path(__file__).parents[1] / "scripts" / "nz_bare_structure_differential.py"
SPEC = spec_from_file_location("_nz_bare_differential_multiset_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)

SOURCE = "a\nIncome must be below 450;\nb\nPayment is 700."


def _boundary_world():
    outline = bare_outline(SOURCE)
    assert outline is not None
    masked = mask_bare_labels(SOURCE, outline)
    start = SOURCE.index("450")
    requirement = {
        "raw": "450",
        "value": 450.0,
        # An unknown raw anchor: the obligation carries no source offsets of
        # its own, so equal obligations have equal comparison keys.
        "source_start": None,
        "source_end": None,
        "alternative_values": (),
    }
    boundary = AUDIT.BoundaryGateBranch(
        ("a",), "numeric-boundary", "bound", masked, 0, len(SOURCE), requirement
    )
    clause = SourceStructureBranch(
        (), "excerpt", "source clause", masked, 0, len(SOURCE)
    )
    assert start > 0
    return outline, boundary, clause


def _gates(**overrides):
    gates = {gate: () for gate in AUDIT.GATES}
    gates["_condition_cues"] = ()
    gates.update(overrides)
    return gates


def test_dropping_one_of_two_equal_obligations_is_reported_and_fails():
    outline, boundary, clause = _boundary_world()
    twin = copy.deepcopy(boundary)
    old = _gates(tests_boundaries=(boundary, twin), clauses=(clause,))
    new = _gates(tests_boundaries=(boundary,), clauses=(clause,))

    changes = AUDIT.compare_gates(old, new, SOURCE, outline)

    removed = changes["tests_boundaries"]["removed"]
    assert len(removed) == 1
    assert changes["tests_boundaries"]["added"] == []
    # The surviving copy belongs to the other obligation; it cannot explain
    # this one away, so the run counts an unexplained removal.
    assert not removed[0]["evidence"]["structural_artifact"]
    counts: Counter[str] = Counter()
    AUDIT.add_counts(
        counts,
        {
            **_EMPTY_RESULT,
            "gate_changes": changes,
        },
    )
    assert counts["tests_boundaries_unexplained_removals"] == 1
    assert AUDIT.failure_count(dict(counts)) >= 1


def test_equal_obligations_kept_in_full_report_no_change():
    outline, boundary, clause = _boundary_world()
    twin = copy.deepcopy(boundary)
    old = _gates(tests_boundaries=(boundary, twin), clauses=(clause,))

    assert AUDIT.compare_gates(old, copy.deepcopy(old), SOURCE, outline) == {}


def test_adding_a_second_equal_obligation_is_reported_as_added():
    outline, boundary, clause = _boundary_world()
    old = _gates(tests_boundaries=(boundary,), clauses=(clause,))
    new = _gates(
        tests_boundaries=(boundary, copy.deepcopy(boundary)), clauses=(clause,)
    )

    changes = AUDIT.compare_gates(old, new, SOURCE, outline)

    assert len(changes["tests_boundaries"]["added"]) == 1
    assert changes["tests_boundaries"]["removed"] == []


def test_dropped_equal_branch_is_not_certified_by_its_surviving_twin():
    outline, pool, clause = _branch_pool()
    branch = pool[1]
    old = _gates(branches=(branch, copy.deepcopy(branch)), clauses=(clause,))
    new = _gates(branches=(copy.deepcopy(branch),), clauses=(clause,))

    removed = AUDIT.compare_gates(old, new, _BRANCH_SOURCE, outline)["branches"][
        "removed"
    ]

    assert len(removed) == 1
    assert removed[0]["evidence"]["overlapping_replacements"] == []
    assert not removed[0]["evidence"]["structural_artifact"]


def test_dropped_twin_evidence_sees_new_gates_without_its_twin(monkeypatch):
    """Pins the `new_gates` override, not only the `replacements` filter.

    Cross-gate checks in `structural_removal_evidence` read
    ``new_gates[gate]``, so the surviving twin must be absent there too. A
    removal with no surviving twin gets the unmodified new gates.
    """

    outline, pool, clause = _branch_pool()
    twin = pool[1]
    old = _gates(branches=(twin, copy.deepcopy(twin), pool[2]), clauses=(clause,))
    survivor = copy.deepcopy(twin)
    new = _gates(branches=(survivor, copy.deepcopy(pool[0])), clauses=(clause,))
    seen = []
    real = AUDIT.structural_removal_evidence

    def spy(removed, replacements, clauses, text, outline, gate, **kwargs):
        seen.append((removed, replacements, kwargs["new_gates"]))
        return real(removed, replacements, clauses, text, outline, gate, **kwargs)

    monkeypatch.setattr(AUDIT, "structural_removal_evidence", spy)

    AUDIT.compare_gates(old, new, _BRANCH_SOURCE, outline)

    by_label = {
        removed.label: (replacements, gates) for removed, replacements, gates in seen
    }
    assert set(by_label) == {"(b)", "(c)"}
    twin_replacements, twin_gates = by_label["(b)"]
    assert survivor not in twin_replacements
    assert survivor not in twin_gates["branches"]
    assert twin_gates["clauses"] == new["clauses"]
    _replacements, plain_gates = by_label["(c)"]
    assert plain_gates is new


# --- differential against main's dictionary comparison --------------------------


def _legacy_compare_gates(old, new, text, outline):
    """main's dictionary comparison, for items without a ``requirement``.

    The body is main's `compare_gates` apart from the name. `_gate_key` puts
    ``None`` where main's key has the requirement tuple, so this matches main
    only for items without ``requirement``, which are the only items the
    property below generates.
    """

    changes = {}
    key = _gate_key
    for gate in AUDIT.GATES:
        old_by_key = {key(item): item for item in old[gate]}
        new_by_key = {key(item): item for item in new[gate]}
        removed = [old_by_key[item] for item in old_by_key.keys() - new_by_key.keys()]
        added = [new_by_key[item] for item in new_by_key.keys() - old_by_key.keys()]
        if removed or added:
            changes[gate] = {
                "removed": [
                    {
                        **AUDIT.branch_record(item),
                        "evidence": AUDIT.structural_removal_evidence(
                            item,
                            new[gate],
                            new["clauses"],
                            text,
                            outline,
                            gate,
                            old_gates=old,
                            new_gates=new,
                        ),
                    }
                    for item in sorted(
                        removed, key=lambda item: (item.start, item.path)
                    )
                ],
                "added": [
                    AUDIT.branch_record(item)
                    for item in sorted(added, key=lambda item: (item.start, item.path))
                ],
            }
    return changes


def _gate_key(item):
    return (
        item.path,
        item.kind,
        item.label,
        item.start,
        item.end,
        AUDIT.collapse(item.text),
        tuple(getattr(item, "computation_chapeaux", ())),
        None,
    )


_BRANCH_SOURCE = (
    "a\nThe first rule applies.\nb\nThe second rule applies.\n"
    "c\nThe third rule applies."
)


def _branch_pool():
    outline = bare_outline(_BRANCH_SOURCE)
    assert outline is not None
    masked = mask_bare_labels(_BRANCH_SOURCE, outline)
    pool = []
    for label in ("a", "b", "c"):
        start = _BRANCH_SOURCE.index(f"{label}\n")
        end = _BRANCH_SOURCE.find("\n", start + 2)
        end = len(_BRANCH_SOURCE) if end == -1 else end
        pool.append(
            SourceStructureBranch(
                (label,), "branch", f"({label})", masked[start:end], start, end
            )
        )
    clause = SourceStructureBranch(
        (), "excerpt", "source clause", masked, 0, len(_BRANCH_SOURCE)
    )
    return outline, pool, clause


@settings(max_examples=150, deadline=None)
@given(
    old_picks=st.lists(st.integers(0, 2), max_size=5),
    new_picks=st.lists(st.integers(0, 2), max_size=5),
)
def test_property_multiset_comparison(old_picks, new_picks):
    """Distinct obligations get main's verdict; copies count one by one."""

    outline, pool, clause = _branch_pool()
    old = _gates(
        branches=tuple(copy.deepcopy(pool[i]) for i in old_picks), clauses=(clause,)
    )
    new = _gates(
        branches=tuple(copy.deepcopy(pool[i]) for i in new_picks), clauses=(clause,)
    )

    changes = AUDIT.compare_gates(old, new, _BRANCH_SOURCE, outline)

    old_counts = Counter(old_picks)
    new_counts = Counter(new_picks)
    change = changes.get("branches", {"removed": [], "added": []})
    labels = {"(a)": 0, "(b)": 1, "(c)": 2}
    assert Counter(labels[item["label"]] for item in change["removed"]) == (
        old_counts - new_counts
    )
    assert Counter(labels[item["label"]] for item in change["added"]) == (
        new_counts - old_counts
    )
    if len(set(old_picks)) == len(old_picks) and len(set(new_picks)) == len(new_picks):
        assert changes == _legacy_compare_gates(old, new, _BRANCH_SOURCE, outline)


_EMPTY_RESULT = {
    "authenticated_labels": 1,
    "candidate_bare_lines": 1,
    "ambiguous_unchanged": True,
    "numeric_removed": [],
    "numeric_added": [],
    "numeric_demand_values_removed": [],
    "numeric_demand_values_added": [],
    "recall_matches_baseline_of_authenticated_mask": True,
    "numeric_occurrences_before": 0,
    "numeric_occurrences_after": 0,
    "lost_condition_cues": [],
    "unanchored_changed_numeric_occurrences": 0,
    "slice_layout_audits": [],
    "unanchored_numeric_occurrences_before": 0,
    "unanchored_numeric_occurrences_after": 0,
    "slices": [],
    "gate_counts_before": {"tests_boundaries": 2},
    "gate_counts_after": {"tests_boundaries": 1},
    "glue_violations": [],
    "formula_contexts": [],
    "ambiguous_restored_numeric_gate_additions": 0,
}
