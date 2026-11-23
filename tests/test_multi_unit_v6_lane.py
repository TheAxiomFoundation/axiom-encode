"""The multi-unit ("batch") form of the v6 lane of the merge guard. OFF.

Issue #2301, option A; design: ``docs/content-bound-merge-guard.md``, section
5, "Multi-unit candidates". With
``reviewed_pr_admission.MULTI_UNIT_V6_ENABLED`` off (the committed state) a
pull request with two v6 manifests is refused as before. On-state behaviour is
reached in tests only, by patching the constant.

Every case runs over the injected ``fetch`` of the single-unit suite's fake
(``tests/test_content_merge_guard.py``); no test reaches the network.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import random
from pathlib import Path

import pytest

from axiom_encode import content_merge_guard as guard
from axiom_encode import reviewed_pr_admission as admission
from axiom_encode import rulespec_citation_paths
from tests.test_content_merge_guard import (
    B2,
    BASE_FILES,
    ENCODER_COMMIT,
    ENCODER_MAIN,
    ENCODER_TIP,
    EXECUTABLE,
    LINK,
    MAX,
    NAME,
    OUTSIDER,
    PAGE,
    PAVEL,
    PR,
    REPO,
    ROOT,
    SUBMODULE,
    WEB_FLOW,
    B,
    World,
    blob_id,
    commit,
    manifest_bytes,
    oid,
    review,
    second_read,
    v5_manifest,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HEAD = oid("batch-head")
HEAD2 = oid("batch-head-2")
ELSEWHERE = oid("not-a-commit-of-this-pr")


def stem(n):
    return f"us-az/policies/des/faa5/medical/block-{n}"


def citation(n):
    return f"us-az/manual/des/faa5/medical/block-{n}"


def P(n):
    return stem(n) + ".yaml"


def T(n):
    return stem(n) + ".test.yaml"


def M(n):
    return guard.MANIFEST_PREFIX + stem(n) + ".json"


def primary(n):
    return f"format: rulespec/v1\n# unit {n}\nrules: []\n".encode()


def companion(n):
    return f"- name: fixture {n}\n".encode()


def run_id(n):
    return str(7000 + n)


def candidate(index):
    return oid(f"batch-candidate-{index}")


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def signing_run_path(n):
    return f"{guard.ENCODER_ROOT}/actions/runs/{run_id(n)}"


def unit_manifest(
    n,
    *,
    candidate_head,
    candidate_base=B,
    author=PAVEL,
    dispatcher=PAVEL,
    mutate=None,
):
    """The signed v6 manifest of unit ``n``: its own citation, pair and run."""

    def shape(payload):
        payload["citation"] = citation(n)
        payload["run_id"] = run_id(n)
        payload["applied_files"] = [
            {"path": P(n), "sha256": sha256(primary(n))},
            {"path": T(n), "sha256": sha256(companion(n))},
        ]
        payload["attestation_scope"]["candidate"]["base_sha"] = candidate_base
        if mutate:
            mutate(payload)

    return manifest_bytes(
        mutate=shape, author=author, dispatcher=dispatcher, candidate=candidate_head
    )


def signing_run(world, n, dispatcher=PAVEL, encoder_commit=ENCODER_COMMIT, **changes):
    world.api[signing_run_path(n)] = {
        "id": int(run_id(n)),
        "event": "workflow_dispatch",
        "path": ".github/workflows/targeted-signed-reencode.yml",
        "head_branch": "main",
        "head_sha": encoder_commit,
        "run_attempt": 1,
        "status": "completed",
        "conclusion": "success",
        "repository": {"full_name": "TheAxiomFoundation/axiom-encode"},
        "actor": {"id": dispatcher[0], "login": dispatcher[1]},
        "triggering_actor": {"id": dispatcher[0], "login": dispatcher[1]},
    } | changes
    world.api[f"{guard.ENCODER_ROOT}/compare/{encoder_commit}...{ENCODER_TIP}"] = {
        "status": "ahead",
        "merge_base_commit": {"sha": encoder_commit},
    }


def batch(
    units=(1, 2, 3),
    *,
    flow="parallel",
    author=PAVEL,
    dispatchers=None,
    approver=MAX,
    mutate=None,
    manifests=None,
    head_extra=None,
    base_extra=None,
    drop=(),
    candidate_extra=None,
    candidate_bases=None,
):
    """One candidate PR that carries ``units``, each signed by its own run.

    ``flow="parallel"``: every run was dispatched on the first commit, which
    holds every pair and no manifest. ``flow="serial"``: run ``i`` was
    dispatched on a commit that also holds the manifests of the units before
    it. ``mutate`` and ``manifests`` map a unit number to a payload mutator or
    to manifest bytes; ``dispatchers`` maps a unit number to its dispatcher.
    """
    world = World()
    units = tuple(units)
    dispatchers = dispatchers or {}
    base_files = BASE_FILES | (base_extra or {})
    world.tree(B, base_files)
    pairs = {}
    for n in units:
        pairs |= {P(n): primary(n), T(n): companion(n)}
    signed, commits, candidates = {}, [], {}
    for index, n in enumerate(units):
        dispatcher = dispatchers.get(n, PAVEL)
        head_sha = candidate(0 if flow == "parallel" else index)
        if head_sha not in world.trees:
            world.tree(head_sha, base_files | pairs | signed | (candidate_extra or {}))
            commits.append(
                commit(head_sha, author)
                if index == 0
                else commit(head_sha, dispatcher, WEB_FLOW)
            )
        candidates[n] = head_sha
        signed[M(n)] = (manifests or {}).get(n) or unit_manifest(
            n,
            candidate_head=head_sha,
            candidate_base=(candidate_bases or {}).get(n, B),
            author=author,
            dispatcher=dispatcher,
            mutate=(mutate or {}).get(n),
        )
        signing_run(world, n, dispatcher)
    head_files = base_files | pairs | signed | (head_extra or {})
    for path in drop:
        del head_files[path]
    world.tree(HEAD, head_files)
    commits.append(commit(HEAD, dispatchers.get(units[-1], PAVEL), WEB_FLOW))
    world.publish(HEAD, B, commits, author=author)
    world.reviews([review(11, approver, HEAD)])
    world.api[ENCODER_MAIN] = {"object": {"sha": ENCODER_TIP}}
    world.units, world.candidates, world.commits = units, candidates, commits
    world.head_files = head_files
    return world


def expect(world, verdict, reason):
    result = world.evaluate()
    assert (result.verdict, result.reason) == (verdict, reason), result
    return result


def fails(world, reason):
    return expect(world, guard.FAIL, reason)


def passes(world):
    return expect(world, guard.PASS, guard.R_MULTI_UNIT_PASS)


ROOMY_LIMIT = 10


@pytest.fixture
def on(monkeypatch):
    """The multi-unit switch open, with a limit of ten units. Tests only.

    The committed limit is 2 (the request budget); the rules are driven with
    batches of up to ten so that first, middle and last positions exist. The
    tests of the committed limit patch the switch alone.
    """
    monkeypatch.setattr(admission, "MULTI_UNIT_V6_ENABLED", True)
    monkeypatch.setattr(admission, "MAX_V6_UNITS", ROOMY_LIMIT)


@pytest.fixture
def switch_only(monkeypatch):
    """The switch open with the committed limit."""
    monkeypatch.setattr(admission, "MULTI_UNIT_V6_ENABLED", True)


# --- the switch ---------------------------------------------------------------


def test_the_switch_is_off_and_is_one_literal_constant_with_no_runtime_override():
    assert admission.MULTI_UNIT_V6_ENABLED is False
    source = Path(admission.__file__).read_text("utf-8")
    assert "os.environ" not in source and "getenv" not in source
    assert "import os" not in source and "argparse" not in source
    assigned = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))
        and any(
            isinstance(target, ast.Name) and target.id == "MULTI_UNIT_V6_ENABLED"
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
        )
    ]
    assert len(assigned) == 1
    assert isinstance(assigned[0].value, ast.Constant)
    assert assigned[0].value.value is False
    # Nothing but the admission module assigns the switch, and the guard
    # reads it from that module at each evaluation.
    for path in sorted((REPOSITORY_ROOT / "src" / "axiom_encode").rglob("*.py")):
        if path.name == "reviewed_pr_admission.py":
            continue
        for node in ast.walk(ast.parse(path.read_text("utf-8"))):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, (ast.AnnAssign, ast.AugAssign))
                else []
            )
            for target in targets:
                name = getattr(target, "id", getattr(target, "attr", None))
                assert name != "MULTI_UNIT_V6_ENABLED", path
    # Every read of the switch compares with ``is True`` or ``is not True``.
    for module in (admission, guard):
        reads = [
            line.strip()
            for line in Path(module.__file__).read_text("utf-8").splitlines()
            if "MULTI_UNIT_V6_ENABLED" in line
            and not line.lstrip().startswith(("#", "MULTI_UNIT_V6_ENABLED:"))
            and "``" not in line
        ]
        assert reads, module
        assert all(
            "MULTI_UNIT_V6_ENABLED is True" in line
            or "MULTI_UNIT_V6_ENABLED is not True" in line
            for line in reads
        ), reads


def test_off_a_valid_batch_is_refused_as_two_v6_manifests_always_were():
    result = fails(batch((1, 2)), guard.R_MULTIPLE_V6)
    assert result.detail == "exactly one v6 manifest is allowed per PR"
    assert "units" not in result.evidence


@pytest.mark.parametrize("value", [1, "true", "True", [True], None, 0, ""])
def test_only_an_exact_true_switch_admits_a_batch(monkeypatch, value):
    monkeypatch.setattr(admission, "MULTI_UNIT_V6_ENABLED", value)
    fails(batch((1, 2)), guard.R_MULTIPLE_V6)


def worst(units):
    """The most requests of one evaluation of ``units`` v6 units (measured
    below, ``test_the_worst_case_evaluation_costs_20_plus_10_requests_per_unit``)."""
    return 20 + 10 * units


# ``merge_guard_runner`` of the guard stack (#2286), which this branch does not
# contain; read at ``cbb47fbb6``: ``MAX_ENFORCE_REQUESTS``, ``MAX_CI_POLLS``
# (and ``MAX_MERGE_GROUP_POLLS``), ``MERGE_GROUP_POLL_READS`` (and
# ``GROUP_CI_RECHECK_READS``). The merge-group evaluation adds four reads to
# ``evaluate``.
RUNNER_CAP, RUNNER_POLLS, GROUP_POLL_READS, GROUP_EVALUATION_READS = 350, 101, 3, 4


def pull_request_run(units):
    return 2 * worst(units) + RUNNER_POLLS


def merge_group_run(units, poll_reads=GROUP_POLL_READS):
    if poll_reads == 1:
        # One listed check: 100 polls of one read, a last poll of two, the
        # evaluation, and a second read of two.
        return 100 + 2 + worst(units) + GROUP_EVALUATION_READS + 2
    return (
        RUNNER_POLLS * poll_reads
        + worst(units)
        + GROUP_EVALUATION_READS
        + GROUP_POLL_READS
    )


def test_the_committed_limit_fits_both_bounds_of_the_enforcing_run():
    """The arithmetic of the design, "Request budget"."""
    limit = admission.MAX_V6_UNITS
    assert limit == 2 and type(limit) is int
    # A pull-request run: two whole evaluations and a full wait.
    assert pull_request_run(limit) == 181 <= RUNNER_CAP
    assert pull_request_run(10) == 341 <= RUNNER_CAP < pull_request_run(11) == 361
    # A merge-group run with three reads at every poll: the binding bound.
    # It is met exactly: the run has no request to spare.
    assert merge_group_run(limit) == 350 == RUNNER_CAP
    assert merge_group_run(limit + 1) == 360 > RUNNER_CAP
    # A merge-group run with one listed check.
    assert merge_group_run(limit, 1) == 148 and merge_group_run(10, 1) == 228
    assert merge_group_run(22, 1) == 348 <= RUNNER_CAP < merge_group_run(23, 1)
    # What ten units would need of the runner's cap.
    assert merge_group_run(10) == 430 and merge_group_run(20) == 530
    # The costliest hour of a pull-request job: 20 polls and two evaluations.
    assert 20 + 2 * worst(10) == 260 and 4 * 260 > 1000 >= 3 * 260


def test_a_batch_of_the_committed_limit_passes_and_one_more_is_refused(switch_only):
    passes(batch((1, 2)))
    fails(batch((1, 2, 3)), guard.R_TOO_MANY_UNITS)


# --- the passing batch ----------------------------------------------------------


def test_a_batch_of_two_passes_with_its_own_reason_and_per_unit_evidence(on):
    world = batch((1, 2))
    result = passes(world)
    assert result.as_dict() == {
        "schema": guard.VERDICT_SCHEMA,
        "verdict": "pass",
        "reason": "content_bound_approval_multi_unit",
        "detail": "",
        "evidence": {
            "repository": REPO,
            "pull_request": 42,
            "head": HEAD,
            "base": B,
            "units": [
                {
                    "manifest": M(n),
                    "paths": {"primary": P(n), "companion": T(n), "manifest": M(n)},
                    "primary_sha256": sha256(primary(n)),
                    "companion_sha256": sha256(companion(n)),
                    "manifest_sha256": sha256(world.head_files[M(n)]),
                    "dispatcher_id": PAVEL[0],
                    "run_id": run_id(n),
                    "candidate_head": candidate(0),
                }
                for n in (1, 2)
            ],
            "review_id": 11,
            "reviewer_id": MAX[0],
            "approvals": [{"review_id": 11, "reviewer_id": MAX[0], "commit_id": HEAD}],
        },
    }


@pytest.mark.parametrize("flow", ["parallel", "serial"])
@pytest.mark.parametrize("count", [2, 3, 5, 10])
def test_batches_up_to_the_limit_pass_in_both_dispatch_orders(on, flow, count):
    assert count <= admission.MAX_V6_UNITS == ROOMY_LIMIT
    result = passes(batch(range(1, count + 1), flow=flow))
    assert len(result.evidence["units"]) == count


def test_units_are_judged_in_the_order_of_their_manifest_paths(on):
    result = passes(batch((3, 1, 2)))
    assert [unit["manifest"] for unit in result.evidence["units"]] == [
        M(1),
        M(2),
        M(3),
    ]


def test_a_batch_pass_is_not_the_single_unit_pass_reason(on):
    """A consumer that accepts by verdict and reason must name the batch."""
    assert guard.R_MULTI_UNIT_PASS != guard.R_PASS
    assert passes(batch((1, 2))).reason == guard.R_MULTI_UNIT_PASS
    assert expect(batch((1,)), guard.PASS, guard.R_PASS).reason == guard.R_PASS


# --- the limit ------------------------------------------------------------------


def test_one_unit_more_than_the_limit_is_refused(on):
    limit = admission.MAX_V6_UNITS
    passes(batch(range(1, limit + 1)))
    world = batch(range(1, limit + 2))
    result = fails(world, guard.R_TOO_MANY_UNITS)
    # Refused from the manifest count: no file list, commit or run is read.
    assert PR + "/files" + PAGE not in world.calls
    assert "units" not in result.evidence


@pytest.mark.parametrize("limit", [True, "10", None, 10.0, 1, 0, -1])
def test_a_limit_that_is_not_a_large_enough_integer_refuses(on, monkeypatch, limit):
    monkeypatch.setattr(admission, "MAX_V6_UNITS", limit)
    fails(batch((1, 2)), guard.R_TOO_MANY_UNITS)


def test_the_limit_is_read_at_each_evaluation(on, monkeypatch):
    monkeypatch.setattr(admission, "MAX_V6_UNITS", 2)
    passes(batch((1, 2)))
    fails(batch((1, 2, 3)), guard.R_TOO_MANY_UNITS)


# --- scope: exactly a set of complete new triples -------------------------------

WORKFLOW = ".github/workflows/repository-checks.yml"
INDEX = ".axiom/index/provisions_to_rules.json"


@pytest.mark.parametrize(
    "path",
    [
        "README.md",  # allowlisted, and still not part of any unit
        "NOTICE",
        INDEX,
        ".axiom/toolchain.toml",
        "known-validation-gaps.yaml",
        "programs/us-az/snap/fy-2026.yaml",
        "tests/test_new.py",
        "us-az/policies/des/faa5/medical/notes.md",
        "us-az/policies/des/faa5/medical/block-1.yaml.bak",
        P(9),  # a primary with no companion and no manifest
        T(9),
    ],
)
def test_any_path_of_no_unit_refuses_the_whole_batch(on, path):
    result = fails(batch(head_extra={path: b"extra\n"}), guard.R_BATCH_EXTRA_PATH)
    assert result.detail.endswith(": " + path)


def test_any_other_file_in_the_manifest_directory_is_read_as_a_manifest(on):
    stray = ".axiom/encoding-manifests/us-az/policies/des/faa5/medical/block-1.sig"
    fails(batch(head_extra={stray: b"extra\n"}), guard.R_MANIFEST_UNPARSEABLE)


def test_a_workflow_change_beside_a_batch_is_refused(on):
    fails(
        batch(
            head_extra={WORKFLOW: b"changed\n"},
            base_extra={WORKFLOW: b"original\n"},
        ),
        guard.R_BATCH_EXTRA_PATH,
    )


def test_a_pair_without_its_manifest_is_a_partial_batch(on):
    extra = {P(9): primary(9), T(9): companion(9)}
    result = fails(batch(head_extra=extra), guard.R_BATCH_EXTRA_PATH)
    assert result.detail.endswith(": " + T(9))  # the first in path order
    assert len(result.evidence["units"]) == 3


@pytest.mark.parametrize("unit", [1, 2, 3])
@pytest.mark.parametrize("member", [P, T])
def test_a_unit_without_its_primary_or_companion_is_incomplete(on, unit, member):
    result = fails(batch(drop=[member(unit)]), guard.R_UNIT_INCOMPLETE)
    assert result.detail.endswith(": " + member(unit))


def test_a_manifest_alone_beside_a_complete_unit_is_incomplete(on):
    fails(batch((1, 2), drop=[P(2), T(2)]), guard.R_UNIT_INCOMPLETE)


def test_removing_a_whole_unit_after_signing_refuses_the_rest(on):
    """The files of unit 3 were in the commit that the runs of units 1 and 2
    validated, and are gone from the head. Units 1 and 2 were validated
    beside bytes that this head does not land: refused. The remedy is a new
    candidate and new signing runs."""
    result = fails(batch(drop=[P(3), T(3), M(3)]), guard.R_CANDIDATE_UNJUDGED)
    assert result.detail.startswith(M(1) + ": ")


@pytest.mark.parametrize("unit", [1, 2, 3])
@pytest.mark.parametrize("member", [P, T, M])
def test_a_unit_path_that_is_already_on_main_refuses_the_batch(on, unit, member):
    result = fails(
        batch(base_extra={member(unit): b"already on main\n"}), guard.R_SCOPE_NOT_NEW
    )
    assert result.detail == "promotion paths already exist on main"


def test_a_unit_that_is_already_on_main_unchanged_leaves_no_manifest_for_it(on):
    """Unit 2 is byte-identical on main: its files are not in the delta, so
    the head is a batch of units 1 and 3 only."""
    world = batch()
    on_main = {path: world.head_files[path] for path in (P(2), T(2), M(2))}
    result = passes(batch(base_extra=on_main))
    assert [unit["manifest"] for unit in result.evidence["units"]] == [M(1), M(3)]


def test_removing_a_file_of_main_beside_a_batch_is_refused(on):
    world = batch()
    files = dict(world.head_files)
    del files["us-az/policies/existing.yaml"]
    world.tree(HEAD, files)
    world.publish(HEAD, B, world.commits)
    result = fails(world, guard.R_BATCH_EXTRA_PATH)
    assert result.detail.endswith(": us-az/policies/existing.yaml")


def test_a_rename_of_a_main_module_into_the_batch_is_refused(on):
    """Git has no rename: the tree delta is a removal and an addition. The
    removal is a path of no unit. A file list that calls it ``renamed``
    disagrees with the tree delta and is refused earlier."""
    world = batch((1, 2))
    files = dict(world.head_files)
    files[P(2)] = files.pop("us-az/policies/existing.yaml")
    world.tree(HEAD, files)
    world.publish(HEAD, B, world.commits)
    fails(world, guard.R_BATCH_EXTRA_PATH)
    listing = world.api[PR + "/files" + PAGE]
    renamed = [
        item | {"status": "renamed"} if item["filename"] == P(2) else item
        for item in listing
        if item["filename"] != "us-az/policies/existing.yaml"
    ]
    world.api[PR + "/files" + PAGE] = renamed
    world.api[PR]["changed_files"] = len(renamed)
    fails(world, guard.R_DELTA_DISAGREES)


@pytest.mark.parametrize("entry", [LINK, SUBMODULE, EXECUTABLE])
@pytest.mark.parametrize("unit", [1, 3])
@pytest.mark.parametrize("member", [P, T, M])
def test_a_unit_file_that_is_not_a_regular_blob_refuses_the_batch(
    on, unit, member, entry
):
    fails(batch(head_extra={member(unit): entry}), guard.R_GUARDED_LEAF)


@pytest.mark.parametrize("evidence", ["files", "compare", "count"])
def test_mutable_file_lists_cannot_hide_a_change_beside_a_batch(on, evidence):
    world = batch(head_extra={WORKFLOW: b"changed\n"})
    comparison = world.api[f"{ROOT}/compare/{B}...{HEAD}"]
    hidden = [item for item in comparison["files"] if item["filename"] != WORKFLOW]
    if evidence == "files":
        world.api[PR + "/files" + PAGE] = hidden
        world.api[PR]["changed_files"] = len(hidden)
    elif evidence == "compare":
        comparison["files"] = hidden
    else:
        world.api[PR]["changed_files"] = len(hidden)
    fails(world, guard.R_DELTA_DISAGREES)


# --- mixed lanes and manifest positions ----------------------------------------


def test_a_v5_manifest_beside_v6_units_is_refused(on):
    v5_primary = "us/policies/ssa/poms/new.yaml"
    v5_path = ".axiom/encoding-manifests/us/policies/ssa/poms/new.json"
    world = batch(
        (1, 2),
        head_extra={
            v5_primary: b"x\n",
            v5_path: v5_manifest(v5_primary),
        },
    )
    fails(world, guard.R_MIXED_SCHEMAS)


def test_an_unknown_schema_beside_v6_units_is_refused(on):
    world = batch(
        (1, 2),
        manifests={2: b'{"schema_version": "axiom-encode/applied-rulespec/v7"}'},
    )
    fails(world, guard.R_UNKNOWN_SCHEMA)


def test_an_unparseable_manifest_beside_v6_units_is_refused(on):
    fails(batch((1, 2), manifests={2: b"{"}), guard.R_MANIFEST_UNPARSEABLE)


def test_a_removed_manifest_of_main_beside_v6_units_is_refused(on):
    old = ".axiom/encoding-manifests/us/policies/existing.json"
    world = batch((1, 2), base_extra={old: b"{}"})
    files = dict(world.head_files)
    del files[old]
    world.tree(HEAD, files)
    world.publish(HEAD, B, world.commits)
    fails(world, guard.R_MANIFEST_REMOVED)


def test_a_v6_manifest_at_a_jurisdiction_local_position_is_refused(on):
    local = "us-az/.axiom/encoding-manifests/policies/des/faa5/medical/block-2.json"
    world = batch((1, 2))
    files = dict(world.head_files)
    files[local] = files.pop(M(2))
    world.tree(HEAD, files)
    world.publish(HEAD, B, world.commits)
    result = fails(world, guard.R_MANIFEST_PATH)
    assert result.detail == f"{local}: {local}"


def test_two_manifests_for_one_citation_are_refused(on):
    """The second manifest names unit 1 and sits at the position of unit 2."""
    clone = unit_manifest(
        1,
        candidate_head=candidate(0),
        mutate=lambda payload: payload.update(run_id=run_id(2)),
    )
    result = fails(batch((1, 2), manifests={2: clone}), guard.R_MANIFEST_PATH)
    assert result.detail.startswith(M(2) + ": ")


# --- each manifest binds exactly its own pair ------------------------------------


def swap_applied(source):
    def mutate(payload):
        payload["applied_files"] = [
            {"path": P(source), "sha256": sha256(primary(source))},
            {"path": T(source), "sha256": sha256(companion(source))},
        ]

    return mutate


def test_a_manifest_that_binds_the_files_of_another_unit_is_cross_bound(on):
    result = fails(batch(mutate={2: swap_applied(1)}), guard.R_UNIT_CROSS_BOUND)
    assert result.detail.startswith(M(2) + ": ")


def test_a_manifest_that_binds_its_primary_and_a_sibling_companion_is_cross_bound(on):
    def mutate(payload):
        payload["applied_files"][1] = {
            "path": T(3),
            "sha256": sha256(companion(3)),
        }

    fails(batch(mutate={2: mutate}), guard.R_UNIT_CROSS_BOUND)


def test_own_paths_with_the_digests_of_another_unit_are_a_digest_mismatch(on):
    def mutate(payload):
        payload["applied_files"] = [
            {"path": P(2), "sha256": sha256(primary(1))},
            {"path": T(2), "sha256": sha256(companion(1))},
        ]

    fails(batch(mutate={2: mutate}), guard.R_APPLIED_FILES)


def test_a_manifest_that_binds_a_file_outside_the_batch_is_a_mismatch(on):
    def mutate(payload):
        payload["applied_files"][1]["path"] = "us-az/policies/existing.test.yaml"

    fails(batch(mutate={2: mutate}), guard.R_APPLIED_FILES)


def test_two_units_of_one_signing_run_are_refused(on):
    world = batch(mutate={3: lambda payload: payload.update(run_id=run_id(1))})
    fails(world, guard.R_UNITS_SHARE_RUN)
    # Refused from the manifests: no blob of a module and no run is read.
    assert not any("/actions/runs/" in path for path in world.calls)


# --- the scope function, directly -------------------------------------------------


def triple(n):
    return (P(n), T(n), M(n))


def added(*units):
    return {path: "added" for n in units for path in triple(n)}


def test_scope_function_admits_exactly_a_disjoint_set_of_new_triples():
    refusal = guard._v6_scope_refusal
    assert refusal(added(1), [triple(1)]) is None
    assert refusal(added(1, 2, 3), [triple(1), triple(2), triple(3)]) is None
    assert refusal(added(1, 2), [triple(1)])[0] == guard.R_SCOPE_PATHS
    assert refusal(added(1, 2, 3), [triple(1), triple(2)])[0] == (
        guard.R_BATCH_EXTRA_PATH
    )
    assert refusal(added(1), [triple(1), triple(2)])[0] == guard.R_UNIT_INCOMPLETE
    assert refusal({}, [triple(1)])[0] == guard.R_SCOPE_PATHS
    assert refusal({}, [triple(1), triple(2)])[0] == guard.R_UNIT_INCOMPLETE


def test_scope_function_refuses_units_that_share_a_path():
    """Unreachable through ``evaluate`` (a manifest position decides its
    primary and companion, and two manifests have two positions), so the rule
    is driven here."""
    refusal = guard._v6_scope_refusal
    shared_primary = (P(1), T(2), M(2))
    assert refusal(added(1, 2), [triple(1), shared_primary]) == (
        guard.R_UNITS_SHARE_PATH,
        "two units name the path: " + P(1),
    )
    companion_as_primary = (T(1), "x/y.test.yaml", M(2))
    assert (
        refusal(
            added(1) | {"x/y.test.yaml": "added", M(2): "added"},
            [triple(1), companion_as_primary],
        )[0]
        == guard.R_UNITS_SHARE_PATH
    )
    assert refusal(added(1), [triple(1), triple(1)])[0] == guard.R_UNITS_SHARE_PATH


@pytest.mark.parametrize("status", ["modified", "removed", "renamed", None])
def test_scope_function_refuses_any_path_that_is_not_added(status):
    delta = added(1, 2) | {T(2): status}
    assert guard._v6_scope_refusal(delta, [triple(1), triple(2)]) == (
        guard.R_SCOPE_NOT_NEW,
        "promotion paths already exist on main",
    )
    assert guard._v6_scope_refusal(added(1) | {M(1): status}, [triple(1)]) == (
        guard.R_SCOPE_NOT_NEW,
        "promotion paths already exist on main",
    )


def test_scope_function_keeps_the_single_unit_reason_and_words():
    words = "delta is not exactly the primary, companion and manifest"
    refusal = guard._v6_scope_refusal
    assert refusal(added(1) | {"x": "added"}, [triple(1)]) == (
        guard.R_SCOPE_PATHS,
        words,
    )
    assert refusal({P(1): "added", M(1): "added"}, [triple(1)]) == (
        guard.R_SCOPE_PATHS,
        words,
    )
    # A missing file wins over an extra one, for one unit and for many.
    assert refusal({P(1): "added", "x": "added"}, [triple(1)])[0] == (
        guard.R_SCOPE_PATHS
    )
    delta = added(1, 2) | {"x": "added"}
    del delta[T(2)]
    assert refusal(delta, [triple(1), triple(2)])[0] == guard.R_UNIT_INCOMPLETE


# --- every single-unit rule holds for each unit of a batch -----------------------


def scope_field(*keys_and_value):
    *keys, value = keys_and_value

    def mutate(payload):
        target = payload["attestation_scope"]
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value

    return mutate


def drop_field(name):
    return lambda payload: payload.pop(name)


def set_field(name, value):
    return lambda payload: payload.update({name: value})


def other_candidate(payload):
    payload["candidate_rulespec_ref"] = ELSEWHERE
    payload["attestation_scope"]["candidate"]["head_sha"] = ELSEWHERE


def wrong_digest(payload):
    payload["applied_files"][0]["sha256"] = "0" * 64


def one_applied_file(payload):
    payload["applied_files"] = payload["applied_files"][:1]


# name -> (payload mutator, expected reason)
PAYLOAD_RULES = {
    "no signature": (drop_field("signature"), guard.R_MANIFEST_SHAPE),
    "an extra field": (set_field("backend", "codex"), guard.R_MANIFEST_SHAPE),
    "another tool": (set_field("tool", "axiom-encode encode"), guard.R_MANIFEST_SHAPE),
    "a run id that is no number": (set_field("run_id", "x"), guard.R_MANIFEST_SHAPE),
    "one applied file": (one_applied_file, guard.R_MANIFEST_SHAPE),
    "owner review claimed": (
        scope_field("owner_review", "approved"),
        guard.R_MANIFEST_SHAPE,
    ),
    "another merge requirement": (
        scope_field("merge_requirement", "none"),
        guard.R_MANIFEST_SHAPE,
    ),
    "another pull request": (
        scope_field("candidate", "pull_request", 43),
        guard.R_MANIFEST_CANDIDATE,
    ),
    "another repository": (
        scope_field("candidate", "repository", "TheAxiomFoundation/rulespec-uk"),
        guard.R_MANIFEST_CANDIDATE,
    ),
    "another author": (
        scope_field("candidate", "author_id", OUTSIDER[0]),
        guard.R_MANIFEST_CANDIDATE,
    ),
    "candidate ref is not the candidate head": (
        set_field("candidate_rulespec_ref", ELSEWHERE),
        guard.R_MANIFEST_CANDIDATE,
    ),
    "candidate is no commit of this PR": (other_candidate, guard.R_MANIFEST_CANDIDATE),
    "citation of another country": (
        set_field("citation", "uk/legislation/ukpga/2012/5/section/1"),
        guard.R_CITATION,
    ),
    "citation that is no citation": (set_field("citation", "../x"), guard.R_CITATION),
    "citation of another module": (
        set_field("citation", "us-az/manual/des/faa5/medical/block-99"),
        guard.R_MANIFEST_PATH,
    ),
    "a digest that is not the head blob": (wrong_digest, guard.R_APPLIED_FILES),
    "a dispatcher that is not the run's": (
        scope_field("dispatcher", "login", "MaxGhenis"),
        guard.R_DISPATCHER_MISMATCH,
    ),
}


def run_field(field, value):
    def change(world, n):
        world.api[signing_run_path(n)][field] = value

    return change


def run_off_main(world, n):
    signing_run(world, n, encoder_commit=oid(f"encoder-side-{n}"))
    world.api[
        f"{guard.ENCODER_ROOT}/compare/{oid(f'encoder-side-{n}')}...{ENCODER_TIP}"
    ] = {"status": "diverged", "merge_base_commit": {"sha": ENCODER_COMMIT}}


def run_missing(world, n):
    del world.api[signing_run_path(n)]


def primary_blob_corrupt(world, n):
    world.api[f"{ROOT}/git/blobs/{blob_id(primary(n))}"]["size"] += 1


def candidate_lost_the_primary(world, n):
    """The commit the run validated carried other primary bytes for unit n."""
    sha = world.candidates[n]
    files = {path: dict(entry) for path, entry in world.trees[sha].items()}
    files[P(n)] = b"other primary bytes\n"
    world.tree(sha, files)


def stray_base(n):
    return oid(f"stray-base-{n}")


def run_changes_between_reads(world, n):
    later = dict(world.api[signing_run_path(n)]) | {"run_attempt": 2}
    second_read(world, signing_run_path(n), later)


# name -> (world change for unit n, expected reason)
WORLD_RULES = {
    "run did not succeed": (run_field("conclusion", "failure"), guard.R_RUN),
    "run still in progress": (run_field("status", "in_progress"), guard.R_RUN),
    "run of another event": (run_field("event", "push"), guard.R_RUN),
    "run of another workflow": (
        run_field("path", ".github/workflows/ci.yml"),
        guard.R_RUN,
    ),
    "run on another branch": (run_field("head_branch", "feature"), guard.R_RUN),
    "run commit not on encoder main": (run_off_main, guard.R_RUN),
    "run unreadable": (run_missing, guard.R_READ_FAILED),
    # Built by ``judged``: an account that is not enrolled dispatched the run
    # of unit n and its manifest names it.
    "dispatcher not enrolled": (None, guard.R_DISPATCHER_NOT_ENROLLED),
    "primary blob does not hash": (primary_blob_corrupt, guard.R_BLOB_UNREADABLE),
    "candidate commit has other primary bytes": (
        candidate_lost_the_primary,
        guard.R_MANIFEST_CANDIDATE,
    ),
    # Built by ``judged``: the manifest of unit n names a base that ``main``
    # does not descend from.
    "candidate base is no ancestor of main": (None, guard.R_MANIFEST_CANDIDATE),
    "run changes between the two reads": (run_changes_between_reads, guard.R_CHANGED),
}


def judged(units, n, rule, enabled, monkeypatch):
    monkeypatch.setattr(admission, "MULTI_UNIT_V6_ENABLED", enabled)
    monkeypatch.setattr(admission, "MAX_V6_UNITS", ROOMY_LIMIT)
    if rule in PAYLOAD_RULES:
        mutate, reason = PAYLOAD_RULES[rule]
        world = batch(units, flow="serial", mutate={n: mutate})
    elif rule == "dispatcher not enrolled":
        reason = WORLD_RULES[rule][1]
        world = batch(units, flow="serial", dispatchers={n: OUTSIDER})
    elif rule == "candidate base is no ancestor of main":
        reason = WORLD_RULES[rule][1]
        world = batch(units, flow="serial", candidate_bases={n: stray_base(n)})
        world.ancestry(stray_base(n), B, oid("older"), status="diverged")
    else:
        change, reason = WORLD_RULES[rule]
        world = batch(units, flow="serial")
        change(world, n)
    return world.evaluate(), reason


@pytest.mark.parametrize("rule", sorted(PAYLOAD_RULES) + sorted(WORLD_RULES))
def test_each_single_unit_rule_refuses_alone_and_at_every_position_of_a_batch(
    monkeypatch, rule
):
    """The rule's reason for one unit (switch off, and on) is its reason for
    the first, a middle and the last unit of a batch of three."""
    for enabled in (False, True):
        result, reason = judged((2,), 2, rule, enabled, monkeypatch)
        assert (result.verdict, result.reason) == (guard.FAIL, reason), (rule, result)
    single_detail = result.detail
    for n in (1, 2, 3):
        result, reason = judged((1, 2, 3), n, rule, True, monkeypatch)
        assert (result.verdict, result.reason) == (guard.FAIL, reason), (rule, n)
        if reason not in (guard.R_READ_FAILED, guard.R_CHANGED):
            # The refusal names the unit; the rest is the single unit's detail.
            assert result.detail.startswith(M(n) + ": "), result
            assert result.detail == f"{M(n)}: " + single_detail.replace(
                M(2), M(n)
            ).replace(run_id(2), run_id(n)), (result, single_detail)


def test_the_rule_tables_cover_every_reason_of_the_per_unit_steps():
    covered = {reason for _change, reason in PAYLOAD_RULES.values()} | {
        reason for _change, reason in WORLD_RULES.values()
    }
    assert covered == {
        guard.R_MANIFEST_SHAPE,
        guard.R_MANIFEST_CANDIDATE,
        guard.R_CITATION,
        guard.R_MANIFEST_PATH,
        guard.R_APPLIED_FILES,
        guard.R_DISPATCHER_MISMATCH,
        guard.R_DISPATCHER_NOT_ENROLLED,
        guard.R_RUN,
        guard.R_READ_FAILED,
        guard.R_BLOB_UNREADABLE,
        guard.R_CHANGED,
    }


# --- the batch is frozen before its first signing run ----------------------------


def test_a_candidate_commit_with_other_bytes_of_a_sibling_is_refused(on):
    """Parallel flow: the one candidate commit had another primary of unit 3.
    Unit 1 is judged first; its own pair is intact, the batch is not."""
    world = batch()
    candidate_lost_the_primary(world, 3)
    result = fails(world, guard.R_BATCH_NOT_FROZEN)
    assert result.detail.startswith(M(1) + ": ")


@pytest.mark.parametrize("late,early", [(1, 2), (1, 3), (2, 3), (2, 1), (3, 1), (3, 2)])
@pytest.mark.parametrize("how", ["other bytes", "absent"])
def test_the_freeze_rule_holds_for_a_unit_at_any_position(on, late, early, how):
    """Independent review of cc09b513a, P2-2. Unit ``early`` was signed on a
    commit where unit ``late`` had other bytes or was absent; the other units
    were signed on a later, complete commit. The refusal names ``early``: each
    unit is judged against its own candidate commit, not the first unit's."""
    world = batch()
    complete = oid("batch-complete-candidate")
    world.tree(complete, world.trees[candidate(0)])
    files = candidate_files(world, candidate(0))
    if how == "absent":
        del files[P(late)], files[T(late)]
    else:
        files[P(late)] = b"other bytes when the early unit was signed\n"
    world.tree(candidate(0), files)
    head_files = dict(world.head_files)
    for n in world.units:
        if n != early:
            head_files[M(n)] = unit_manifest(n, candidate_head=complete)
    world.tree(HEAD, head_files)
    world.commits.insert(1, commit(complete))
    world.publish(HEAD, B, world.commits)
    result = fails(world, guard.R_BATCH_NOT_FROZEN)
    assert result.detail.startswith(M(early) + ": ")


@pytest.mark.parametrize("member", [P, T])
@pytest.mark.parametrize("sibling", [2, 3])
def test_the_freeze_rule_covers_primaries_and_companions_of_every_sibling(
    on, member, sibling
):
    world = batch()
    files = {path: dict(entry) for path, entry in world.trees[candidate(0)].items()}
    files[member(sibling)] = b"other bytes at the candidate commit\n"
    world.tree(candidate(0), files)
    result = fails(world, guard.R_BATCH_NOT_FROZEN)
    assert result.detail.startswith(M(1) + ": ")


def test_a_unit_added_after_the_first_signing_run_is_refused(on):
    """Serial flow: unit 3 joined after units 1 and 2 were signed, so their
    candidate commits do not carry it."""
    world = batch(flow="serial")
    for index in (0, 1):
        sha = candidate(index)
        files = {
            path: {key: value for key, value in entry.items() if key != "path"}
            for path, entry in world.trees[sha].items()
            if path not in (P(3), T(3))
        }
        world.tree(sha, files)
    result = fails(world, guard.R_BATCH_NOT_FROZEN)
    assert result.detail.startswith(M(1) + ": ")


# --- the candidate rule: nothing unjudged beside a validated unit -----------------

HELPER = "us-az/policies/des/faa5/medical/helper.yaml"


def test_review_case_a_unit_signed_beside_a_sibling_that_is_dropped(on):
    """Independent review of cc09b513a, P2-1. Admission (switch on) signed
    unit 1 while the candidate held the unsigned pair of unit 8. The pair is
    removed; the head is the triple of unit 1 only. Before the fix the guard
    passed it with the single-unit reason."""
    extra = {P(8): primary(8), T(8): companion(8)}
    result = fails(batch((1,), candidate_extra=extra), guard.R_CANDIDATE_UNJUDGED)
    assert result.detail == (
        "the candidate commit holds a change that is not a file of a unit of this head"
    )
    result = fails(batch((1, 2), candidate_extra=extra), guard.R_CANDIDATE_UNJUDGED)
    assert result.detail.startswith(M(1) + ": ")


def test_review_case_a_candidate_with_a_helper_module_that_the_head_drops(on):
    """Independent review of cc09b513a
    (``test_freeze_candidate_holds_extra_file_removed_later``): the commit the
    run validated held a module a unit could import; it is gone at the head."""
    for units in ((1, 2), (1,)):
        fails(
            batch(units, candidate_extra={HELPER: b"helper\n"}),
            guard.R_CANDIDATE_UNJUDGED,
        )


def test_off_the_single_unit_lane_does_not_apply_the_candidate_rule():
    """With the switch off admission never signs beside another file but the
    optional derived index, and the evaluation is the base's: this head passes
    as it always did. The rule exists only where the switch is on."""
    world = batch((1,), candidate_extra={HELPER: b"helper\n"})
    expect(world, guard.PASS, guard.R_PASS)


def candidate_files(world, sha):
    return {path: dict(entry) for path, entry in world.trees[sha].items()}


def held_a_workflow(files):
    files[WORKFLOW] = b"changed for the run\n"


def held_another_toolchain_pin(files):
    files[".axiom/toolchain.toml"] = b"pin = 'other'\n"


def held_other_bytes_of_a_main_module(files):
    files["us-az/policies/existing.yaml"] = b"changed for the run\n"


def lacked_a_main_module(files):
    del files["us-az/policies/existing.yaml"]


def held_the_derived_index(files):
    files[INDEX] = b"{}\n"


def held_a_readme_change(files):
    files["README.md"] = b"changed\n"


@pytest.mark.parametrize(
    "change",
    [
        held_a_workflow,
        held_another_toolchain_pin,
        held_other_bytes_of_a_main_module,
        lacked_a_main_module,
        held_the_derived_index,
        held_a_readme_change,
    ],
)
@pytest.mark.parametrize("units,position", [((1,), 0), ((1, 2, 3), 0), ((1, 2, 3), 2)])
def test_any_change_at_the_candidate_that_the_head_does_not_land_refuses(
    on, change, units, position
):
    """For one unit and for a batch, and for a unit that is not the first:
    each unit's own candidate commit is compared with its own base."""
    world = batch(units, flow="serial")
    sha = candidate(position)
    files = candidate_files(world, sha)
    change(files)
    world.tree(sha, files)
    result = fails(world, guard.R_CANDIDATE_UNJUDGED)
    if len(units) > 1:
        assert result.detail.startswith(M(units[position]) + ": ")


def test_a_sibling_manifest_at_the_candidate_must_be_the_one_of_the_head(on):
    """Serial order: the run of unit 2 validated beside a manifest of unit 1
    that was signed again later."""
    world = batch((1, 2), flow="serial")
    files = candidate_files(world, candidate(1))
    files[M(1)] = unit_manifest(
        1,
        candidate_head=candidate(0),
        mutate=lambda payload: payload.update(generated_at="2026-11-01T00:00:00+00:00"),
    )
    world.tree(candidate(1), files)
    result = fails(world, guard.R_CANDIDATE_UNJUDGED)
    assert result.detail.startswith(M(2) + ": ")


def test_the_candidate_is_compared_with_the_main_commit_it_was_cut_from(on):
    """``main`` moved after signing: the candidate is compared with its own
    base, so a file that only the newer ``main`` holds is not a change of the
    candidate, and a base-only update keeps a clean batch passing."""
    world = batch()
    update_branch(world)
    passes(world)
    assert f"{ROOT}/git/trees/{B}?recursive=1" in world.calls


@pytest.mark.parametrize("units", [(1,), (1, 2)])
def test_a_unit_file_that_the_candidate_base_already_held_refuses(on, units):
    """The ``main`` commit the candidate was cut from held other bytes at the
    primary of unit 1, and ``main`` dropped that file before this head: at
    the head the path is new, at the validated commit it was a replacement."""
    older = oid("older-main-with-the-path")
    world = batch(units, candidate_bases={n: older for n in units})
    world.tree(older, BASE_FILES | {P(1): b"an earlier module at this path\n"})
    world.ancestry(older, B, older)
    fails(world, guard.R_CANDIDATE_UNJUDGED)
    world.tree(older, BASE_FILES)
    expect(
        world, guard.PASS, guard.R_PASS if len(units) == 1 else guard.R_MULTI_UNIT_PASS
    )


@pytest.mark.parametrize("units", [(1,), (1, 2)])
def test_narrow_review_p3_2_an_allowlisted_file_added_at_the_candidate_refuses(
    on, units
):
    """Narrow review of e058ec1ca, P3-2. The candidate added ``NOTICE``, a
    file of the allowlist. ``main`` added the same bytes later, so the head
    holds the file with that blob and its delta is the units only. The
    candidate still differed from the commit it was cut from by a file of no
    unit: the rule has no allowlist."""
    older = oid("older-main-without-notice")
    notice = {"NOTICE": b"notice\n"}
    world = batch(units, base_extra=notice, candidate_bases={n: older for n in units})
    world.tree(older, BASE_FILES)
    world.ancestry(older, B, older)
    fails(world, guard.R_CANDIDATE_UNJUDGED)
    # The same head passes when the commit it was cut from held the file.
    world.tree(older, BASE_FILES | notice)
    expect(
        world, guard.PASS, guard.R_PASS if len(units) == 1 else guard.R_MULTI_UNIT_PASS
    )


@pytest.mark.parametrize("path", ["README.md", "LICENSE"])
def test_an_allowlisted_file_removed_at_the_candidate_refuses(on, path):
    world = batch(base_extra={"LICENSE": b"license\n"})
    files = candidate_files(world, candidate(0))
    del files[path]
    world.tree(candidate(0), files)
    fails(world, guard.R_CANDIDATE_UNJUDGED)


def test_an_unreadable_candidate_base_tree_refuses(on):
    world = batch()
    update_branch(world)
    world.api[f"{ROOT}/git/trees/{B}?recursive=1"]["truncated"] = True
    fails(world, guard.R_DELTA_INCOMPLETE)


# --- one approval for the whole batch# --- one approval for the whole batch --------------------------------------------


def test_an_approval_before_the_last_manifest_does_not_cover_the_batch(on):
    world = batch(flow="serial")
    # The approved commit holds every pair and the manifests of units 1, 2.
    approved = candidate(2)
    world.reviews([review(11, MAX, approved)])
    world.ancestry(B, approved, B)
    result = fails(world, guard.R_NOT_CONTENT_BOUND)
    assert result.evidence["rejected_approvals"] == [
        {"review_id": 11, "reviewer_id": MAX[0], "reason": guard.R_NOT_CONTENT_BOUND}
    ]


def test_an_approval_of_the_unsigned_candidate_does_not_count(on):
    world = batch()
    world.reviews([review(11, MAX, candidate(0))])
    world.ancestry(B, candidate(0), B)
    fails(world, guard.R_NOT_CONTENT_BOUND)


def update_branch(world, changed=None):
    """A merge of a newer ``main`` into the batch, by someone else."""
    base_files = BASE_FILES | {"docs/other.md": b"merged elsewhere\n"}
    world.tree(B2, base_files)
    world.ancestry(B, B2, B)
    files = dict(world.head_files) | {"docs/other.md": b"merged elsewhere\n"}
    files |= changed or {}
    world.tree(HEAD2, files)
    commits = world.commits + [commit(HEAD2, PAVEL, WEB_FLOW)]
    world.publish(HEAD2, B2, commits)
    world.ancestry(B2, HEAD, B)


def test_a_base_only_update_keeps_the_one_approval_of_the_batch(on):
    world = batch()
    update_branch(world)
    result = passes(world)
    assert result.evidence["approvals"] == [
        {"review_id": 11, "reviewer_id": MAX[0], "commit_id": HEAD}
    ]


@pytest.mark.parametrize("unit", [1, 2, 3])
def test_a_byte_change_in_any_unit_after_the_approval_voids_it(on, unit):
    world = batch()
    update_branch(world, {T(unit): b"- name: changed after approval\n"})
    # The manifest of that unit no longer matches its companion.
    fails(world, guard.R_APPLIED_FILES)


def test_a_resigned_unit_after_the_approval_voids_it(on):
    world = batch()
    resigned = unit_manifest(
        2,
        candidate_head=candidate(0),
        mutate=lambda payload: payload.update(generated_at="2026-11-13T00:00:00+00:00"),
    )
    update_branch(world, {M(2): resigned})
    fails(world, guard.R_NOT_CONTENT_BOUND)


def test_no_approval_at_all(on):
    world = batch()
    world.reviews([])
    fails(world, guard.R_NO_APPROVAL)


def test_the_author_cannot_approve_the_batch(on):
    fails(batch(author=MAX, approver=MAX), guard.R_IS_AUTHOR)


@pytest.mark.parametrize("unit", [1, 2, 3])
def test_the_dispatcher_of_any_one_unit_cannot_approve_the_batch(on, unit):
    """Max dispatched one of the three runs; Max is then no approver, and
    Pavel is the author: nobody can approve this batch."""
    world = batch(dispatchers={unit: MAX})
    world.reviews([review(11, MAX, HEAD)])
    # The head commit is attributed to the last dispatcher; keep Max out of
    # the commit list so that the dispatcher rule is the one that refuses.
    world.commits[:] = [commit(candidate(0), PAVEL), commit(HEAD, PAVEL, WEB_FLOW)]
    world.publish(HEAD, B, world.commits)
    fails(world, guard.R_IS_DISPATCHER)


def test_the_run_actor_of_any_one_unit_cannot_approve_the_batch(on):
    world = batch()
    world.api[signing_run_path(3)]["actor"] = {"id": MAX[0], "login": MAX[1]}
    fails(world, guard.R_IS_DISPATCHER)


def test_an_approver_who_wrote_a_commit_cannot_approve_the_batch(on):
    world = batch()
    world.commits.insert(1, commit(oid("by-max"), MAX))
    world.publish(HEAD, B, world.commits)
    fails(world, guard.R_AUTHORED_COMMITS)


def test_an_approver_who_is_not_enrolled_cannot_approve_the_batch(on):
    fails(batch(approver=OUTSIDER), guard.R_NOT_ENROLLED)


def test_changes_requested_blocks_the_batch(on):
    world = batch()
    world.reviews(
        [
            review(11, MAX, HEAD),
            review(12, OUTSIDER, HEAD, "CHANGES_REQUESTED", "COLLABORATOR"),
        ]
    )
    fails(world, guard.R_CHANGES_REQUESTED)


def test_a_dismissed_approval_does_not_cover_the_batch(on):
    world = batch()
    world.reviews([review(11, MAX, HEAD, "DISMISSED")])
    fails(world, guard.R_NO_APPROVAL)


def test_an_unattributed_commit_refuses_the_batch(on):
    world = batch()
    world.commits.insert(1, commit(oid("nobody"), None, PAVEL))
    world.publish(HEAD, B, world.commits)
    fails(world, guard.R_UNATTRIBUTED)


# --- the rules of the whole pull request -------------------------------------------


def test_a_draft_batch_has_no_verdict(on):
    world = batch()
    world.api[PR]["draft"] = True
    fails(world, guard.R_PR_DRAFT)


def test_a_batch_needs_the_required_check_green_on_its_head(on):
    world = batch()
    world.ci(HEAD, conclusion="failure")
    fails(world, guard.R_CI)


def test_a_batch_behind_main_is_refused(on):
    world = batch()
    world.ancestry(B, HEAD, B, status="diverged")
    fails(world, guard.R_BASE_NOT_CURRENT)


def test_a_batch_in_a_subdivision_repository_is_unsupported(on):
    world = batch((1, 2))
    world.api = json.loads(json.dumps(world.api).replace(NAME, "rulespec-us-tn"))
    result = guard.evaluate("rulespec-us-tn", 42, fetch=world.fetch)
    assert (result.verdict, result.reason) == (
        guard.FAIL,
        guard.R_UNSUPPORTED_REPOSITORY,
    )


def test_main_moving_during_the_evaluation_refuses_the_batch(on):
    world = batch()
    second_read(world, ROOT + "/git/ref/heads/main", {"object": {"sha": B2}})
    fails(world, guard.R_CHANGED)


def test_encoder_main_moving_between_the_two_passes_refuses_the_batch(on):
    """The second pass reads the encoder tip again: a run commit that is no
    longer on encoder ``main`` refuses, as it does for one unit."""
    world = batch()
    moved = oid("encoder-main-rewritten")
    second_read(world, ENCODER_MAIN, {"object": {"sha": moved}})
    world.api[f"{guard.ENCODER_ROOT}/compare/{ENCODER_COMMIT}...{moved}"] = {
        "status": "diverged",
        "merge_base_commit": {"sha": oid("older-encoder")},
    }
    fails(world, guard.R_CHANGED)


def passing_reads(units=(1, 2, 3), flow="serial"):
    world = batch(units, flow=flow)
    assert world.evaluate().verdict == guard.PASS
    return list(world.calls)


def test_any_failed_or_malformed_read_fails_the_batch_closed(on):
    reads = passing_reads()
    assert len(reads) == len(
        [path for path in reads if path.startswith("/repos/TheAxiomFoundation/")]
    )
    for index in range(len(reads)):
        for failure in ("error", "malformed"):
            world = batch(flow="serial")
            seen = []

            def fetch(path, index=index, failure=failure, world=world, seen=seen):
                seen.append(path)
                if len(seen) - 1 == index:
                    if failure == "error":
                        raise OSError("read failed")
                    return "not the expected JSON"
                return world.fetch(path)

            result = guard.evaluate(NAME, 42, fetch=fetch)
            assert result.verdict == guard.FAIL, (index, reads[index], failure)


# --- request volume ---------------------------------------------------------------


def worst_case(count):
    """Every read the evaluation can make, with one page per list: each unit
    with its own candidate commit and candidate base (and that base's tree),
    each signing run on its own encoder commit behind the tip, and the
    approval on a commit that is not the head."""
    units = range(1, count + 1)
    bases = {n: oid(f"older-main-{n}") for n in units}
    world = batch(units, flow="serial", candidate_bases=bases)
    for n in units:
        world.ancestry(bases[n], B2, bases[n])
        world.tree(bases[n], BASE_FILES)
        signing_run(world, n, encoder_commit=oid(f"encoder-commit-{n}"))
    update_branch(world)
    return world


@pytest.mark.parametrize("count", [1, 2, 3, 5, 10])
def test_the_worst_case_evaluation_costs_20_plus_10_requests_per_unit(on, count):
    world = worst_case(count)
    assert world.evaluate().verdict == guard.PASS
    assert len(world.calls) == 20 + 10 * count == worst(count)


@pytest.mark.parametrize("count,expected", [(2, 30), (3, 35), (5, 45), (10, 70)])
def test_a_parallel_batch_approved_on_its_head_costs_20_plus_5_per_unit(
    on, count, expected
):
    """One candidate commit, one encoder commit: 3 blobs and 2 run reads for
    each unit. (A single unit on the same terms costs 25.)"""
    world = batch(range(1, count + 1))
    passes(world)
    assert len(world.calls) == expected == 20 + 5 * count


def test_the_encoder_main_tip_is_read_once_in_each_pass_over_the_runs(on):
    world = batch(range(1, 6))
    passes(world)
    assert world.calls.count(ENCODER_MAIN) == 2
    for n in range(1, 6):
        assert world.calls.count(signing_run_path(n)) == 2


def test_the_encoder_reader_caches_only_the_tip_and_exact_ancestries():
    calls = []

    def read(path):
        calls.append(path)
        return {"n": len(calls)}

    cached = guard._encoder_main_reader(read)
    tip = guard.ENCODER_ROOT + "/git/ref/heads/main"
    compare = guard.ENCODER_ROOT + "/compare/" + "a" * 40 + "..." + "b" * 40
    run = guard.ENCODER_ROOT + "/actions/runs/7"
    other_tip = ROOT + "/git/ref/heads/main"
    other_compare = ROOT + "/compare/" + "a" * 40 + "..." + "b" * 40
    assert cached(tip) is cached(tip)
    assert cached(compare) is cached(compare)
    assert cached(run) != cached(run)
    assert cached(other_tip) != cached(other_tip)
    assert cached(other_compare) != cached(other_compare)
    assert calls.count(tip) == 1 and calls.count(compare) == 1
    assert calls.count(run) == 2
    assert calls.count(other_tip) == 2 and calls.count(other_compare) == 2
    # A second reader is a second pass: it reads the tip again.
    guard._encoder_main_reader(read)(tip)
    assert calls.count(tip) == 2


def test_a_failed_tip_read_is_not_cached():
    attempts = []

    def read(path):
        attempts.append(path)
        raise OSError("down")

    cached = guard._encoder_main_reader(read)
    for _ in range(2):
        with pytest.raises(OSError):
            cached(guard.ENCODER_ROOT + "/git/ref/heads/main")
    assert len(attempts) == 2


# --- property: nothing but a valid set of triples is admitted ----------------------

FUZZ_CASES = int(os.environ.get("MULTI_UNIT_FUZZ_CASES", "3000"))
STRAY_PATHS = (
    "README.md",
    INDEX,
    WORKFLOW,
    ".axiom/toolchain.toml",
    "known-validation-gaps.yaml",
    "programs/us/snap.yaml",
    "tests/fixtures/case.yaml",
    "us-az/policies/existing.yaml",
    "us-az/policies/des/faa5/medical/notes.md",
    "us-az/policies/des/faa5/medical/block-1.yaml.orig",
    ".axiom/encoding-manifests/us-az/policies/des/faa5/medical/block-1.json.bak",
    "us-zz/policies/new.yaml",
    "newroot/file.txt",
)


def fuzz_world(rng):
    """A random head tree and the oracle's answer, from the recipe alone.

    Each of up to ``MAX_V6_UNITS + 2`` units independently holds a random
    subset of its primary, companion and manifest; some of its paths may
    already be on ``main``; and the head may hold stray paths. The oracle
    never looks at the delta: the candidate is admissible exactly when every
    present unit is complete and new, at least one unit is present, at most
    ``MAX_V6_UNITS`` are, no stray path changed, and the validated commit held
    no file that the head does not land.
    """
    limit = admission.MAX_V6_UNITS
    count = rng.choice([1, 1, 2, 2, 3, 4, 5, limit, limit + 1, limit + 2])
    units = list(range(1, count + 1))
    clean = rng.random() < 0.35
    present, on_main = {}, {}
    for n in units:
        members = {"P", "T", "M"}
        if not clean and rng.random() < 0.25:
            members = {m for m in members if rng.random() < 0.6}
        present[n] = members
        on_main[n] = set()
        if not clean and rng.random() < 0.08 and members:
            on_main[n] = {rng.choice(sorted(members))}
    strays = {}
    if not clean and rng.random() < 0.5:
        for path in rng.sample(STRAY_PATHS, rng.randint(1, 3)):
            strays[path] = rng.choice([b"stray\n", None])
    candidate_stray = None
    if not clean and rng.random() < 0.12:
        candidate_stray = rng.choice(STRAY_PATHS + (P(90), T(90)))
    mode = None
    if not clean and rng.random() < 0.05:
        mode = (rng.choice(units), rng.choice("PTM"), rng.choice([LINK, EXECUTABLE]))

    world = World()
    base_files = dict(BASE_FILES)
    world_units = [n for n in units if present[n]]
    pairs, signed = {}, {}
    for n in units:
        if "P" in present[n]:
            pairs[P(n)] = primary(n)
        if "T" in present[n]:
            pairs[T(n)] = companion(n)
        if "M" in present[n]:
            signed[M(n)] = unit_manifest(n, candidate_head=candidate(0))
        signing_run(world, n)
    path_of = {"P": P, "T": T, "M": M}
    for n in units:
        for member in on_main[n]:
            base_files[path_of[member](n)] = b"already on main\n"
    head_files = base_files | pairs | signed
    stray_changed = False
    for path, value in strays.items():
        if value is None:
            if path in head_files and path in base_files:
                del head_files[path]
                stray_changed = True
        elif head_files.get(path) != value:
            head_files[path] = value
            stray_changed = True
    mode_changed = False
    if mode is not None:
        n, member, entry = mode
        path = path_of[member](n)
        if path in head_files:
            head_files[path] = entry
            mode_changed = True
    world.tree(B, base_files)
    # The candidate commit carries every module the head carries.
    world.tree(
        candidate(0),
        BASE_FILES
        | {
            path: value
            for path, value in head_files.items()
            if path in pairs and isinstance(value, bytes)
        },
    )
    if candidate_stray is not None:
        # A file that only the validated commit held (or held otherwise).
        files = candidate_files(world, candidate(0))
        files[candidate_stray] = b"only at the candidate\n"
        world.tree(candidate(0), files)
    world.tree(HEAD, head_files)
    commits = [commit(candidate(0), PAVEL), commit(HEAD, PAVEL, WEB_FLOW)]
    world.publish(HEAD, B, commits)
    world.reviews([review(11, MAX, HEAD)])
    world.api[ENCODER_MAIN] = {"object": {"sha": ENCODER_TIP}}

    complete = [n for n in world_units if present[n] == {"P", "T", "M"}]
    scope_valid = (
        1 <= len(world_units) <= limit
        and len(complete) == len(world_units)
        and not any(on_main[n] for n in world_units)
        and not stray_changed
        and not mode_changed
    )
    # With the switch off the candidate rule of the multi-unit form is not
    # applied (the base's single-unit evaluation).
    world.admissible_with_the_switch_off = scope_valid and len(world_units) == 1
    return world, scope_valid and candidate_stray is None, len(world_units)


def test_fuzz_only_a_valid_set_of_new_triples_is_admitted(on):
    """Random head trees: the guard passes exactly the admissible ones.

    ``MULTI_UNIT_FUZZ_CASES`` sets the number of cases (default 3,000; the
    change was run with 120,000, see the design)."""
    rng = random.Random(2301)
    admitted = refused = 0
    reasons: dict[str, int] = {}
    for case in range(FUZZ_CASES):
        world, admissible, count = fuzz_world(rng)
        result = world.evaluate()
        if admissible:
            expected = guard.R_PASS if count == 1 else guard.R_MULTI_UNIT_PASS
            assert (result.verdict, result.reason) == (guard.PASS, expected), (
                case,
                result,
            )
            admitted += 1
        else:
            # ``not_applicable`` is possible (a head that changes only an
            # allowlisted file); a pass is not.
            assert result.verdict != guard.PASS, (case, result)
            reasons[result.reason] = reasons.get(result.reason, 0) + 1
            refused += 1
    # The run exercised both outcomes and every scope reason.
    assert admitted >= FUZZ_CASES // 10 and refused >= FUZZ_CASES // 4
    if FUZZ_CASES >= 3000:
        assert {
            guard.R_BATCH_EXTRA_PATH,
            guard.R_UNIT_INCOMPLETE,
            guard.R_SCOPE_NOT_NEW,
            guard.R_SCOPE_PATHS,
            guard.R_TOO_MANY_UNITS,
            guard.R_GUARDED_LEAF,
            guard.R_CANDIDATE_UNJUDGED,
        } <= set(reasons), reasons


def test_fuzz_with_the_switch_off_admits_single_units_only():
    rng = random.Random(2301)
    for case in range(min(FUZZ_CASES, 1500)):
        world, admissible, count = fuzz_world(rng)
        result = world.evaluate()
        del admissible, count
        if world.admissible_with_the_switch_off:
            assert (result.verdict, result.reason) == (guard.PASS, guard.R_PASS), case
        else:
            assert result.verdict != guard.PASS, (case, result)


def test_fuzz_scope_function_against_an_independent_oracle():
    """The scope rule alone, over random deltas and random unit lists."""
    rng = random.Random(42)
    pool = [triple(n) for n in range(1, 9)]
    statuses = ["added"] * 12 + ["modified", "removed"]
    admitted = 0
    for _case in range(max(FUZZ_CASES * 10, 30000)):
        units = rng.sample(pool, rng.randint(1, 5))
        if rng.random() < 0.05:
            units.append(rng.choice(units))
        delta = {path: "added" for unit in units for path in unit}
        if rng.random() < 0.6:
            for _ in range(rng.randint(1, 3)):
                action = rng.random()
                if action < 0.35 and delta:
                    del delta[rng.choice(sorted(delta))]
                elif action < 0.7:
                    extra = rng.choice(
                        [path for unit in pool for path in unit] + list(STRAY_PATHS)
                    )
                    delta[extra] = rng.choice(statuses)
                elif delta:
                    delta[rng.choice(sorted(delta))] = rng.choice(statuses)
        claimed = [path for unit in units for path in unit]
        valid = (
            len(set(claimed)) == len(claimed)
            and set(delta) == set(claimed)
            and all(status == "added" for status in delta.values())
        )
        refusal = guard._v6_scope_refusal(delta, units)
        assert (refusal is None) == valid, (delta, units, refusal)
        admitted += valid
    assert admitted > 1000


# --- pinned copies and source facts -----------------------------------------------


def test_the_admission_modules_copies_match_their_sources():
    assert admission._BATCH_MANIFEST_PREFIX == guard.MANIFEST_PREFIX
    assert admission._BATCH_ATOMIC_ROOTS == (
        rulespec_citation_paths.RULESPEC_ATOMIC_ROOTS
    )
    assert admission._BATCH_ATOMIC_ROOTS == guard.RULESPEC_ATOMIC_ROOTS
    assert admission._BATCH_JURISDICTION.pattern == guard._JURISDICTION.pattern


def test_the_admission_module_still_imports_nothing_from_the_package():
    """The root provisioner loads it by path, outside the package."""
    tree = ast.parse(Path(admission.__file__).read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and not (node.module or "").startswith(
                "axiom_encode"
            )
        if isinstance(node, ast.Import):
            assert not any(
                alias.name.startswith("axiom_encode") for alias in node.names
            )


def test_the_multi_unit_reasons_are_distinct_reason_codes():
    reasons = {
        name: value
        for name, value in vars(guard).items()
        if name.startswith("R_") and isinstance(value, str)
    }
    assert len(set(reasons.values())) == len(reasons)
    assert {
        "content_bound_approval_multi_unit",
        "too_many_v6_units",
        "v6_unit_incomplete",
        "v6_batch_extra_path",
        "v6_units_share_a_path",
        "v6_units_share_a_signing_run",
        "v6_unit_cross_bound",
        "v6_batch_not_frozen_at_candidate",
        "v6_candidate_holds_unjudged_change",
    } <= set(reasons.values())


def test_one_function_judges_one_unit_and_many():
    """No second copy of the per-unit rules: ``_evaluate`` calls each helper
    of the v6 steps from one place."""
    source = inspect.getsource(guard._evaluate)
    for call in (
        "_parse_manifest(",
        "citation_rulespec_path(",
        "_v6_scope_refusal(",
        "_signing_run(run_read",
        "_signing_run(recheck_read",
        "R_MANIFEST_CANDIDATE",
        "R_APPLIED_FILES",
        "R_DISPATCHER_MISMATCH",
        "R_DISPATCHER_NOT_ENROLLED",
    ):
        assert source.count(call) == 1, call


# --- the design and the changelog state what the code does -----------------------


def flat(path):
    text = (REPOSITORY_ROOT / path).read_text("utf-8")
    return " ".join(text.replace(">", " ").split())


def design_section():
    design = flat("docs/content-bound-merge-guard.md")
    start = design.index(
        "### Multi-unit candidates (change 5d, present and switched off)"
    )
    end = design.index("## 6. Merge queue and `notary-guard`")
    assert start < end
    return design[start:end]


def test_the_design_states_the_switch_and_the_limit_as_committed():
    section = design_section()
    assert admission.MULTI_UNIT_V6_ENABLED is False
    assert "`reviewed_pr_admission.MULTI_UNIT_V6_ENABLED` is `False`" in section
    assert f"`MAX_V6_UNITS` is {admission.MAX_V6_UNITS} (" in section
    assert "every read compares it with `is True`" in section
    assert "The switch is not the only lock, and the two halves are locked" in section
    assert "The runner's lock below does not cover this half." in section
    assert "So before the switch is set back to `False`, close every open" in section
    # The header of the design says that the form exists and is off.
    design = flat("docs/content-bound-merge-guard.md")
    assert (
        "A multi-unit form of the v6 lane (change 5d of §8) is present and "
        "switched off: `MULTI_UNIT_V6_ENABLED` is `False`"
    ) in design[: design.index("## 1. What exists today")]
    source = flat("src/axiom_encode/reviewed_pr_admission.py").replace("# ", "")
    assert f"MAX_V6_UNITS: Final[int] = {admission.MAX_V6_UNITS}" in source
    assert "2 x (20 + 10u) + 101 <= 350 gives u <= 10" in source
    assert "3 x 101 + (24 + 10u) + 3 <= 350 gives u <= 2 (exactly 350)" in source
    assert f"(10 units: 3 x 101 + 124 + 3 = {merge_group_run(10)})" in source
    assert "ROLLBACK ORDER: that rule exists only while the switch is on." in source


def test_the_design_names_every_multi_unit_reason_with_its_value():
    section = design_section()
    multi_unit = {
        guard.R_MULTI_UNIT_PASS,
        guard.R_TOO_MANY_UNITS,
        guard.R_UNIT_INCOMPLETE,
        guard.R_BATCH_EXTRA_PATH,
        guard.R_UNITS_SHARE_PATH,
        guard.R_UNITS_SHARE_RUN,
        guard.R_UNIT_CROSS_BOUND,
        guard.R_BATCH_NOT_FROZEN,
        guard.R_CANDIDATE_UNJUDGED,
    }
    source = Path(guard.__file__).read_text("utf-8")
    block = source[
        source.index("# The multi-unit form of the v6 lane") : source.index(
            "R_APPLIED_FILES ="
        )
    ]
    declared = {
        node.value.value
        for node in ast.walk(ast.parse(block))
        if isinstance(node, ast.Assign)
    }
    assert declared == multi_unit
    for reason in sorted(multi_unit):
        assert f"| `{reason}` |" in section, reason
    for reason in (
        guard.R_SCOPE_NOT_NEW,
        guard.R_GUARDED_LEAF,
        guard.R_DELTA_DISAGREES,
        guard.R_MANIFEST_PATH,
        guard.R_MIXED_SCHEMAS,
        guard.R_MULTIPLE_V6,
        guard.R_MANIFEST_CANDIDATE,
        guard.R_IS_DISPATCHER,
    ):
        assert f"`{reason}`" in section, reason


def test_the_design_states_the_request_budget_this_file_measures():
    section = design_section()
    limit = admission.MAX_V6_UNITS
    counts = ", ".join(str(worst(units)) for units in (1, 2, 3, 5, 10))
    assert f"`20 + 10u` ({counts} for 1, 2, 3, 5, 10 units)" in section
    parallel = ", ".join(str(20 + 5 * units) for units in (2, 3, 5, 10))
    assert f"`20 + 5u` ({parallel} for 2, 3, 5, 10 units)" in section
    assert (
        f"`2 x (20 + 10u) + 101 <= 350` | 10 ({pull_request_run(10)}); 11 gives "
        f"{pull_request_run(11)} |"
    ) in section
    assert (
        f"`3 x 101 + (24 + 10u) + 3 <= 350` | **{limit}** ({merge_group_run(limit)}); "
        f"{limit + 1} gives {merge_group_run(limit + 1)} |"
    ) in section
    assert (
        f"`100 + 2 + (24 + 10u) + 2 <= 350` | 22 ({merge_group_run(22, 1)}); 2 units "
        f"give {merge_group_run(2, 1)}, 10 give {merge_group_run(10, 1)} |"
    ) in section
    assert (
        f"`MAX_V6_UNITS` is {limit}: the merge-group run with three reads at every "
        f"poll is the binding bound, and {limit} is the largest size that every row "
        "covers. In that row two units use the whole cap: the run has no request "
        "to spare"
    ) in section
    assert merge_group_run(limit) == RUNNER_CAP
    # The best layout: four units fit that row exactly, five do not.
    assert "(`3 x 101 + (24 + 5 x 4) + 3 = 350`)" in section
    assert 3 * 101 + (24 + 5 * 4) + 3 == RUNNER_CAP < 3 * 101 + (24 + 5 * 5) + 3
    # What #2301's sizes need of the runner.
    assert f"`3 x 101 + (24 + 100) + 3 = {merge_group_run(10)}`" in section
    assert f"`2 x (20 + 100) + 101 = {pull_request_run(10)}`" in section
    assert (
        f"`MAX_ENFORCE_REQUESTS` of at least {merge_group_run(10)} (450 leaves "
        f"{450 - merge_group_run(10)} for further pages)"
    ) in section
    assert f"`3 x 101 + (24 + 200) + 3 = {merge_group_run(20)}`" in section
    assert f"`2 x (20 + 200) + 101 = {pull_request_run(20)}`" in section
    assert "Neither is in this change." in section
    assert "no value of it gives a pass" in section
    assert "Each unit adds at least 5" in section
    assert "Each unit adds at most 10" in section
    assert f"three reads at every one of {RUNNER_POLLS} polls" in section
    assert "With the best layout (`20 + 5u`) that row covers 4 units" in section
    assert (
        f"The committed {limit} is what the runner's cap of {RUNNER_CAP} covers in "
        f"every case. 10 needs a cap of at least {merge_group_run(10)} and 20 a cap "
        f"of at least {merge_group_run(20)}"
    ) in section
    assert "without that case, 10 units fit" in section
    assert (
        "At 30 to 40 units a day that is 90 to 120 actions a day by the dispatcher "
        "and the merger"
    ) in section
    assert (3 * 30, 3 * 40) == (90, 120)
    # The rate limit of the job token.
    assert (
        f"`20 + 2 x {worst(10)} = {20 + 2 * worst(10)}` requests for 10 units, "
        "against 80 today (two evaluations of 30 and 20 polls). So 3 such jobs fit "
        "one hour and a fourth reaches the limit"
    ) in section
    assert '"1,000 requests per hour per repository"' in section
    assert "pull-request job holds 20 polls and both evaluations" in section
    assert 3 * (20 + 2 * worst(10)) <= 1000 < 4 * (20 + 2 * worst(10))


def test_the_design_states_why_a_batch_is_one_queue_entry():
    section = design_section()
    for sentence in (
        "**Why this lane is the only way to land several units at one time.**",
        "requires four settings of every `merge_queue` rule: one entry to merge at "
        "most and at least, one entry to build, and the `ALLGREEN` grouping "
        "strategy",
        "cannot prove that the group holds only that pull request",
        "each entry needs its own fresh judgment",
        "A batch pull request is one queue entry: one merge-group run judges all "
        "of its units.",
        "| Add to the merge queue | 1 | 1: one queue entry |",
    ):
        assert sentence in section, sentence
    assert dict(admission._MERGE_QUEUE_REQUIRED) == {
        "grouping_strategy": "ALLGREEN",
        "max_entries_to_build": 1,
        "min_entries_to_merge": 1,
        "max_entries_to_merge": 1,
    }


def test_the_design_states_the_review_granularity_and_the_human_steps():
    section = design_section()
    for sentence in (
        "under **one** owner approval",
        "**The intended change: review granularity.**",
        "one approval covers every unit of the batch",
        "every primary and every companion of the batch, not a sample",
        "Who may approve does not change",
        "or run actor of **any** unit's signing run",
        "a batch whose units were dispatched by both has no possible approver",
        "All or nothing: one bad unit refuses the whole candidate.",
        "A candidate of one unit is judged as a single unit",
        "One approval covers the batch.",
        "Signing stays one run for one unit. The workflow file is not changed.",
        "A one-run form (one dispatch, one `production-signing` approval, `k` "
        "manifests) is **not built**.",
        "| Approve the green candidate | 1 | **1 for the batch** (the intended "
        "change) |",
        "| Dispatch `targeted-signed-reencode.yml` | 1 | `k`, all by one enrolled "
        "identity |",
        "| Approve the `production-signing` environment | 1 | `k` |",
        "| Actions of a person | 7 | `3k + 4`: 10 for 2 units, against 14; 34 for "
        "10 units, against 70 |",
        "| 2 (`MAX_V6_UNITS` as committed) | 2 | at most 6 to 8 | the switch and "
        "the runner's success set |",
        "| 10 | 10 | at most 30 to 40 | also a request cap of at least 430 in the "
        "runner and a new `MAX_V6_UNITS` |",
        "The lane without owner approval that #2301 also discusses (option B and "
        "its narrow form) is not designed or built here",
        "neither is tested against the hosted workflow [not verified]",
        "No batch ran on the hosted workflow.",
    ):
        assert sentence in section, sentence
    assert (3 * 2 + 4, 7 * 2, 3 * 10 + 4, 7 * 10) == (10, 14, 34, 70)
    assert admission.MAX_V6_UNITS == 2


def test_the_design_states_the_freeze_rule_and_the_second_lock():
    section = design_section()
    for sentence in (
        "**Freeze rule (new, batch only).**",
        "must already carry the primary and the companion of **every** unit of "
        "the batch",
        "A unit that is added after a sibling was signed refuses the batch",
        "Each unit is judged against its own candidate commit.",
        "**Candidate rule of the multi-unit form (new; with the switch on, for one "
        "unit and for many).**",
        "may differ from the `main` commit it was cut from (`candidate.base_sha`) "
        "only by files that this head lands",
        "**no unit passes after it was validated beside bytes that are not part of "
        "the final, judged batch or of the `main` commit it was cut from.**",
        "Removing a whole unit after a sibling was signed therefore refuses the rest",
        "With the switch off the rule is not applied",
        "**Turning the switch on also tightens the single-unit lane, in this one "
        "way.**",
        "With the switch off a single-unit candidate may carry the derived index",
        "so admission refuses that candidate first and no signing run is spent",
        "the operator removes the index from the candidate (merge `main`, delete "
        "the file) and dispatches again",
        "so complete sibling pairs stay admitted",
        "A file of no unit is refused before signing",
        "keeping them is question 2 below",
        "A unit whose run fails cannot be removed from a candidate whose other "
        "units are already signed",
        "so a batch verdict ends that job with failure until a reviewed change "
        "adds the reason there",
        "Add the new reasons to `_DECIDED_REASONS` of the same file",
        "each is a decision, none is an incomplete read",
        "Raise the request cap to at least 430 before `MAX_V6_UNITS` is raised to 10",
        "Whether `validate / validate` fails a head on which a module imports a "
        "module that the head does not hold.",
        "A pass of two or more units has its own reason code, "
        "`content_bound_approval_multi_unit`, not `content_bound_approval`.",
        "add `(pass, content_bound_approval_multi_unit)` to `SUCCESS_VERDICTS`",
        "This branch has no `BUNDLE_SHA256` and changes none.",
        "Two manifests that name one signing run: `fail`. One run signs one unit.",
    ):
        assert sentence in section, sentence
    assert guard.R_PASS == "content_bound_approval"
    # The follow-up names every new refusal of the evaluator for the runner.
    follow_up = section[section.index("Add the new reasons to `_DECIDED_REASONS`") :]
    follow_up = follow_up[: follow_up.index("Until then")]
    for reason in (
        guard.R_MULTI_UNIT_PASS,
        guard.R_TOO_MANY_UNITS,
        guard.R_UNIT_INCOMPLETE,
        guard.R_BATCH_EXTRA_PATH,
        guard.R_UNITS_SHARE_PATH,
        guard.R_UNITS_SHARE_RUN,
        guard.R_UNIT_CROSS_BOUND,
        guard.R_BATCH_NOT_FROZEN,
        guard.R_CANDIDATE_UNJUDGED,
    ):
        assert f"`{reason}`" in follow_up, reason
    assert "under `fail` the eight refusals" in follow_up
    # The statement about this branch is true of this branch.
    for path in (REPOSITORY_ROOT / "src").rglob("*.py"):
        assert "BUNDLE_SHA256" not in path.read_text("utf-8"), path
    for path in (REPOSITORY_ROOT / ".github" / "workflows").glob("*.yml"):
        assert "BUNDLE_SHA256" not in path.read_text("utf-8"), path


def test_the_design_states_the_fuzz_size_this_file_defaults_to():
    section = design_section()
    assert inspect.getsource(test_fuzz_only_a_valid_set_of_new_triples_is_admitted)
    assert 'os.environ.get("MULTI_UNIT_FUZZ_CASES", "3000")' in Path(
        __file__
    ).read_text("utf-8")
    assert "(default 3,000 cases; run with 120,000 for this change)" in section


def test_the_changelog_names_the_switch_and_the_limit():
    fragment = flat("changelog.d/2301-multi-unit-v6-lane.added.md")
    assert "`MULTI_UNIT_V6_ENABLED` (off)" in fragment
    assert f"`MAX_V6_UNITS` ({admission.MAX_V6_UNITS}, " in fragment
    assert "With the switch off every verdict and admission result is unchanged." in (
        fragment
    )
