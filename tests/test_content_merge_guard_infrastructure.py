"""Proposed three-class lane of the content-bound merge guard. OFF.

Design section 10, question 0. With ``INFRASTRUCTURE_LANE_ENABLED`` off (the
default) the classifier is unchanged. With it on:

- content keeps its lanes;
- a control-plane change needs one content-bound approval from an enrolled
  owner who did not author it;
- an infrastructure-only change is ``not_applicable``;
- content mixed with anything else, and any unknown path, is refused.
"""

from __future__ import annotations

import urllib.error

import pytest

from axiom_encode import content_merge_guard as guard
from tests.test_content_merge_guard import (
    B2,
    BASE_FILES,
    C1,
    COMPANION,
    EXECUTABLE,
    H2,
    INDEX,
    LEDGER,
    LINK,
    MAX,
    OUTSIDER,
    PAVEL,
    PIN,
    PR,
    PRIMARY,
    ROOT,
    SUBMODULE,
    V5_UNIT,
    WEB_FLOW,
    WORKFLOW,
    B,
    H,
    M,
    P,
    T,
    World,
    build,
    commit,
    expect,
    fails,
    plain,
    review,
    second_read,
)

CONTROL_PLANE = [
    WORKFLOW,
    ".github/CODEOWNERS",
    ".github/.gitignore",
    PIN,
    ".axiom/toolchain.toml",
    ".axiom/repository-structure.yaml",
    ".axiom/target-validation-passes.json",
    ".axiom/pending-validation-fingerprints.json",
    ".axiom/reviewed-migrations.json",
    ".axiom/legacy-replacements/abc.json",
    ".axiom/upstream-source-check-baseline.txt",
    ".axiom/anything-new.toml",  # default-deny inside .axiom
    INDEX,
    LEDGER,
    "known-dangling.yaml",
    "known-missing-money-atoms.yaml",
    "known-source-conflicts.yaml",
    "oracle-coverage-pending.yaml",
    "household-choices.yaml",
    "tools/known-broken-specs.txt",
    "tools/known-engine-load-failures.txt",
    "programs/us-az/snap/fy-2026.yaml",
    ".gitattributes",
    "CLAUDE.md",
    ".claude/settings.json",
    ".claude/commands/approve.md",
    "NEWFILE.md",  # a root file no class names is control plane
    "pyproject.toml",
]
INFRASTRUCTURE = [
    "tools/generate.py",
    "tools/README.md",
    "tools/tests/test_manifest_provenance.py",
    "tools/pin_oracle/axiom_pin_oracle.py",
    "tools/known-other.txt",
    "bulk/compute_matrix.py",
    "bulk/worklist.yaml",
    "tests/test_x.py",
    "tests/fixtures/case.yaml",
]
CONTENT = [
    P,
    T,
    M,
    "us-mo/manual/dss/snap/block-1.yaml",
    "us-zz/newroot/x.yaml",  # a new jurisdiction is content with no lane
    "us/ca/policies/x.yaml",
    "us-az/policies/notes.md",
    "us-az/x.yaml",
    "us-az/.gitattributes",
    "us-az/CLAUDE.md",
    "us-az/README.md",
    "us-az/tests/test_x.py",
    "us-az/.github/workflows/x.yml",
    "us-co/.axiom/notes.md",
    "us-az/.axiom",
    "us-az/.axiom/encoding-manifests",
    "us-az/.axiom/encoding-manifests/policies/x.json",
    ".axiom",
    ".axiom/encoding-manifests",
    ".axiom/encoding-manifests/us-az/policies/x.json",
    "policies/x.yaml",
    "statutes/26/1.yaml",
    "us-az",
    "policies",
    "nz",
]
UNKNOWN = [
    "docs/guide.md",
    "newtop/x.txt",
    "sub/README.md",
    ".clauded/x",
    "src/x.py",
    "US-AZ/policies/x.yaml",
]
MALFORMED = [
    "",
    "/",
    "/tests/x.py",
    "tests/",
    "tests//x.py",
    "tests/../us-az/policies/x.yaml",
    "tests/./x.py",
    "..",
    "us-az\\policies\\x.yaml",
    "tests\\x.py",
    "tests/x y.py",
    "tests/x\ty.py",
    "tests/x\ny.py",
    "tests/x\x00.py",
    "tests/x\x7f.py",
    "tests/café.py",
    "tests/‮x.py",
    "уs-az/policies/x.yaml",  # Cyrillic first letter
    ".axiom/ENCODING-MANIFESTS/us-az/x.json",
    ".AXIOM/toolchain.toml",
    ".GitHub/workflows/x.yml",
    "Tests/test_x.py",
    "TOOLS/generate.py",
    "tools/Known-Broken-Specs.txt",
    "Programs/us/snap.yaml",
    "us-az/Policies/x.yaml",
    "STATUTES/26/1.yaml",
    "readme.md",
    "claude.md",
    ".GITATTRIBUTES",
    "license",
    None,
    7,
    b"tests/x.py",
]
JURISDICTIONS = frozenset({"us-az", "us"})


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(guard, "INFRASTRUCTURE_LANE_ENABLED", True)


def approved(files, *, approver=MAX, author=PAVEL, **kwargs):
    """A PR of ``files`` by ``author``, approved on its head by ``approver``."""
    world = plain(files, **kwargs)
    world.publish(H, B, [commit(H, author)], author=author)
    world.reviews([review(11, approver, H)])
    return world


# --- the classifier ----------------------------------------------------------


@pytest.mark.parametrize("path", CONTROL_PLANE)
def test_control_plane_paths(path):
    assert guard.classify_path(path, JURISDICTIONS) == guard.PATH_CONTROL_PLANE


@pytest.mark.parametrize("path", INFRASTRUCTURE)
def test_infrastructure_paths(path):
    assert guard.classify_path(path, JURISDICTIONS) == guard.PATH_INFRASTRUCTURE


@pytest.mark.parametrize("path", CONTENT)
def test_content_paths(path):
    assert guard.classify_path(path, JURISDICTIONS) == guard.PATH_CONTENT
    # Content does not depend on the base list: a jurisdiction-shaped name
    # that the base lacks is still content.
    assert guard.classify_path(path, frozenset()) == guard.PATH_CONTENT


@pytest.mark.parametrize("path", UNKNOWN)
def test_unknown_top_level_directories(path):
    assert guard.classify_path(path, JURISDICTIONS) == guard.PATH_UNKNOWN


@pytest.mark.parametrize("path", MALFORMED, ids=repr)
def test_malformed_or_case_variant_paths_are_unknown(path):
    assert guard.classify_path(path, JURISDICTIONS) == guard.PATH_UNKNOWN


@pytest.mark.parametrize("path", sorted(guard.UNGUARDED_ROOT_FILES))
def test_allowlisted_paths(path):
    assert guard.classify_path(path, JURISDICTIONS) == guard.PATH_UNGUARDED


def test_a_base_jurisdiction_of_any_name_is_content():
    assert (
        guard.classify_path("tribal/policies/x.yaml", frozenset({"tribal"}))
        == guard.PATH_CONTENT
    )
    assert guard.classify_path("tribal/policies/x.yaml", frozenset()) == (
        guard.PATH_UNKNOWN
    )


def test_no_named_directory_can_be_a_jurisdiction_or_a_root():
    named = guard.INFRASTRUCTURE_ROOT_DIRECTORIES | guard.CONTROL_PLANE_ROOT_DIRECTORIES
    assert not guard.INFRASTRUCTURE_ROOT_DIRECTORIES & (
        guard.CONTROL_PLANE_ROOT_DIRECTORIES
    )
    for name in named:
        assert guard._JURISDICTION.fullmatch(name) is None, name
        assert name not in guard.RULESPEC_ATOMIC_ROOTS, name
    for path in guard.CONTROL_PLANE_FILES:
        assert path.split("/")[0] in guard.INFRASTRUCTURE_ROOT_DIRECTORIES


# --- off (default): nothing changes -------------------------------------------


def test_the_lane_is_off_by_default():
    assert guard.INFRASTRUCTURE_LANE_ENABLED is False


@pytest.mark.parametrize("path", CONTROL_PLANE + INFRASTRUCTURE)
def test_off_no_lane_also_with_an_enrolled_approval(path):
    fails(approved({path: b"x\n"}), guard.R_NO_LANE)


def test_only_an_exact_true_enables(monkeypatch):
    monkeypatch.setattr(guard, "INFRASTRUCTURE_LANE_ENABLED", 1)
    fails(approved({PIN: b"x\n"}), guard.R_NO_LANE)
    fails(plain({"tests/test_x.py": b"x\n"}), guard.R_NO_LANE)


# --- on: infrastructure only --------------------------------------------------


@pytest.mark.parametrize("path", INFRASTRUCTURE)
def test_on_infrastructure_only_is_not_applicable(enabled, path):
    world = plain({path: b"x\n"})
    expect(world, guard.NOT_APPLICABLE, guard.R_INFRASTRUCTURE)
    # Decided from the two trees: no review, CI or manifest is read.
    assert not any("/reviews" in call or "/check-runs" in call for call in world.calls)


def test_on_many_infrastructure_paths_with_an_allowlisted_file(enabled):
    world = plain({path: b"x\n" for path in INFRASTRUCTURE} | {"README.md": b"x\n"})
    expect(world, guard.NOT_APPLICABLE, guard.R_INFRASTRUCTURE)


def test_on_removed_infrastructure_is_not_applicable(enabled):
    removed = "tests/fixtures/case.yaml"
    assert removed in BASE_FILES
    expect(plain({}, removed=[removed]), guard.NOT_APPLICABLE, guard.R_INFRASTRUCTURE)


def test_on_allowlist_only_keeps_its_own_reason(enabled):
    expect(
        plain({"README.md": b"x\n"}), guard.NOT_APPLICABLE, guard.R_NO_GUARDED_CONTENT
    )


@pytest.mark.parametrize("entry", [LINK, SUBMODULE, EXECUTABLE])
@pytest.mark.parametrize("path", [WORKFLOW, "tools/run.sh", "bulk/compute_matrix.py"])
def test_on_changed_leaf_that_is_not_a_regular_file_fails(enabled, path, entry):
    # The evaluator's leaf rule stays: a changed executable tool has no lane.
    fails(approved({path: entry}), guard.R_GUARDED_LEAF)


# --- on: the control plane needs one content-bound enrolled approval -----------


@pytest.mark.parametrize("path", CONTROL_PLANE)
def test_on_control_plane_without_approval_fails(enabled, path):
    fails(plain({path: b"x\n"}), guard.R_NO_APPROVAL)


@pytest.mark.parametrize("path", CONTROL_PLANE)
def test_on_control_plane_with_an_enrolled_independent_approval_passes(enabled, path):
    result = expect(approved({path: b"x\n"}), guard.PASS, guard.R_CONTROL_PLANE_PASS)
    assert result.evidence["reviewer_id"] == MAX[0]
    assert result.evidence["paths"] == [path]


def test_on_two_pr_split_first_half_is_not_waved_through(enabled):
    """The review's P1: a workflow, pin or register change must not be free.

    Otherwise an infrastructure-only PR relaxes validation and a second PR
    with a v5-labelled manifest takes the earlier lane with no approval.
    """
    for files in (
        {WORKFLOW: b"jobs: {}\n"},
        {PIN: b"x\n"},
        {LEDGER: b"x\n", ".axiom/toolchain.toml": b"x\n"},
        {LEDGER: b"x\n", "tests/test_x.py": b"x\n"},
    ):
        fails(plain(files), guard.R_NO_APPROVAL)


def test_on_control_plane_with_infrastructure_is_all_control_plane(enabled):
    files = {PIN: b"x\n", "tests/test_x.py": b"x\n", "README.md": b"x\n"}
    fails(plain(files), guard.R_NO_APPROVAL)
    result = expect(approved(files), guard.PASS, guard.R_CONTROL_PLANE_PASS)
    assert result.evidence["paths"] == sorted(files)


def test_on_author_cannot_approve_the_control_plane(enabled):
    world = approved({PIN: b"x\n"}, approver=PAVEL)
    fails(world, guard.R_IS_AUTHOR)


def test_on_non_enrolled_approver_does_not_count(enabled):
    fails(approved({PIN: b"x\n"}, approver=OUTSIDER), guard.R_NOT_ENROLLED)


def test_on_enrolled_login_with_another_id_does_not_count(enabled):
    fails(approved({PIN: b"x\n"}, approver=(777, "MaxGhenis")), guard.R_NOT_ENROLLED)


def test_on_approver_who_committed_does_not_count(enabled):
    world = approved({PIN: b"x\n"}, author=OUTSIDER)
    world.publish(H, B, [commit(C1, OUTSIDER), commit(H, MAX)], author=OUTSIDER)
    world.tree(C1, BASE_FILES | {PIN: b"x\n"})
    fails(world, guard.R_AUTHORED_COMMITS)


def test_on_unattributed_commit_fails(enabled):
    world = approved({PIN: b"x\n"})
    world.publish(H, B, [commit(H, None, WEB_FLOW)])
    fails(world, guard.R_UNATTRIBUTED)


def test_on_enrolled_changes_requested_blocks(enabled):
    world = approved({PIN: b"x\n"}, author=OUTSIDER)
    world.reviews([review(11, MAX, H), review(12, PAVEL, H, state="CHANGES_REQUESTED")])
    fails(world, guard.R_CHANGES_REQUESTED)


def test_on_dismissed_approval_does_not_count(enabled):
    world = approved({PIN: b"x\n"})
    world.reviews([review(11, MAX, H), review(12, MAX, H, state="DISMISSED")])
    fails(world, guard.R_NO_APPROVAL)


def test_on_draft_control_plane_pr_fails(enabled):
    world = approved({PIN: b"x\n"})
    world.api[PR]["draft"] = True
    fails(world, guard.R_PR_DRAFT)


def test_on_red_ci_fails(enabled):
    world = approved({PIN: b"x\n"})
    world.ci(H, conclusion="failure")
    fails(world, guard.R_CI)


def _moved(files_at_approval, files_at_head, *, base_at_head=B, base_files=None):
    """Approval on ``H``; the PR then moves to ``H2``."""
    world = World()
    world.tree(B, BASE_FILES)
    world.tree(H, BASE_FILES | files_at_approval)
    head_base = BASE_FILES | (base_files or {})
    if base_at_head != B:
        world.tree(base_at_head, head_base)
    world.tree(H2, head_base | files_at_head)
    world.publish(H2, base_at_head, [commit(H), commit(H2, PAVEL, WEB_FLOW)])
    world.ancestry(base_at_head, H, B)
    world.reviews([review(11, MAX, H)])
    return world


def test_on_base_only_update_keeps_the_control_plane_approval(enabled):
    world = _moved(
        {PIN: b"x\n"},
        {PIN: b"x\n"},
        base_at_head=B2,
        base_files={"tests/merged_elsewhere.py": b"y\n"},
    )
    result = expect(world, guard.PASS, guard.R_CONTROL_PLANE_PASS)
    assert result.evidence["approvals"][0]["commit_id"] == H


def test_on_changed_byte_after_approval_voids_it(enabled):
    world = _moved({PIN: b"x\n"}, {PIN: b"y\n"})
    fails(world, guard.R_NOT_CONTENT_BOUND)


def test_on_added_path_after_approval_voids_it(enabled):
    world = _moved({PIN: b"x\n"}, {PIN: b"x\n", WORKFLOW: b"jobs: {}\n"})
    fails(world, guard.R_NOT_CONTENT_BOUND)


def test_on_added_infrastructure_path_after_approval_voids_it(enabled):
    world = _moved({PIN: b"x\n"}, {PIN: b"x\n", "tests/test_x.py": b"x\n"})
    fails(world, guard.R_NOT_CONTENT_BOUND)


def test_on_dropped_path_after_approval_voids_it(enabled):
    world = _moved({PIN: b"x\n", LEDGER: b"x\n"}, {PIN: b"x\n"})
    fails(world, guard.R_NOT_CONTENT_BOUND)


def test_on_removal_is_bound_like_a_change(enabled):
    removed = "programs/us/snap.yaml"
    assert removed in BASE_FILES
    world = approved({}, removed=[removed])
    expect(world, guard.PASS, guard.R_CONTROL_PLANE_PASS)
    # Approved while the file was edited; the head now removes it.
    world = World()
    world.tree(B, BASE_FILES)
    world.tree(H, BASE_FILES | {removed: b"edited\n"})
    world.tree(H2, {k: v for k, v in BASE_FILES.items() if k != removed})
    world.publish(H2, B, [commit(H), commit(H2, PAVEL, WEB_FLOW)])
    world.ancestry(B, H, B)
    world.reviews([review(11, MAX, H)])
    fails(world, guard.R_NOT_CONTENT_BOUND)


def test_on_approval_on_a_commit_outside_the_pr_does_not_count(enabled):
    world = approved({PIN: b"x\n"})
    world.reviews([review(11, MAX, C1)])
    fails(world, guard.R_NOT_CONTENT_BOUND)


def test_on_review_dismissed_between_the_two_reads_fails(enabled):
    world = approved({PIN: b"x\n"})
    reviews = PR + "/reviews?per_page=100&page=1"
    assert reviews in world.api
    second_read(
        world, reviews, [review(11, MAX, H), review(12, MAX, H, state="DISMISSED")]
    )
    fails(world, guard.R_CHANGED)


def _control_plane_reads():
    world = approved({PIN: b"x\n"})
    monkey = pytest.MonkeyPatch()
    monkey.setattr(guard, "INFRASTRUCTURE_LANE_ENABLED", True)
    try:
        expect(world, guard.PASS, guard.R_CONTROL_PLANE_PASS)
    finally:
        monkey.undo()
    return sorted(set(world.calls))


CONTROL_PLANE_READS = _control_plane_reads()


@pytest.mark.parametrize("index", range(len(CONTROL_PLANE_READS)))
def test_on_any_failed_read_of_a_passing_evaluation_fails(enabled, index):
    world = approved({PIN: b"x\n"})

    def refuse():
        raise urllib.error.HTTPError("https://api.github.com", 500, "error", {}, None)

    world.api[CONTROL_PLANE_READS[index]] = refuse
    fails(world, guard.R_READ_FAILED)


def test_the_passing_control_plane_evaluation_reads_reviews_commits_and_ci():
    assert len(CONTROL_PLANE_READS) >= 8
    for part in (
        "/reviews",
        "/commits",
        "/check-runs",
        "/actions/runs/",
        "/git/trees/",
    ):
        assert any(part in call for call in CONTROL_PLANE_READS), part
    assert all(call.startswith(ROOT) for call in CONTROL_PLANE_READS)


# --- on: content and mixed pull requests are refused as before -----------------


@pytest.mark.parametrize(
    "path",
    [path for path in CONTENT if path not in {P, T, M}] + UNKNOWN,
)
def test_on_content_or_unknown_path_alone_still_fails(enabled, path):
    result = approved({path: b"x\n"}).evaluate()
    assert result.verdict == guard.FAIL, result


@pytest.mark.parametrize("other", [WORKFLOW, PIN, LEDGER, "CLAUDE.md", "tests/x.py"])
@pytest.mark.parametrize(
    "content",
    [P, "us-mo/manual/dss/snap/block-1.yaml", "us-zz/newroot/x.yaml", "docs/x.md"],
)
def test_on_content_mixed_with_anything_fails_also_when_approved(
    enabled, other, content
):
    fails(approved({other: b"x\n", content: b"x\n"}), guard.R_NO_LANE)


@pytest.mark.parametrize("path", ["tests/x y.py", "tests/café.py", "Tests/x.py"])
def test_on_malformed_path_in_the_delta_fails_also_when_approved(enabled, path):
    fails(approved({path: b"x\n"}), guard.R_NO_LANE)
    fails(approved({path: b"x\n", PIN: b"x\n"}), guard.R_NO_LANE)


@pytest.mark.parametrize("other", [WORKFLOW, PIN, LEDGER, "tests/test_x.py"])
def test_on_signed_v5_unit_with_anything_else_fails(enabled, other):
    # The shape of rulespec-us#1470: a signed manifest together with a pin, a
    # ledger and a test. Still no lane: split the pull request.
    fails(approved(V5_UNIT | {other: b"x\n"}), guard.R_NO_LANE)


def test_on_v5_unit_alone_is_still_the_earlier_lane(enabled):
    expect(plain(V5_UNIT), guard.NOT_APPLICABLE, guard.R_EARLIER_LANE)


def test_on_v6_unit_alone_still_needs_its_approval(enabled):
    expect(build(), guard.PASS, guard.R_PASS)
    world = build()
    world.reviews([])
    fails(world, guard.R_NO_APPROVAL)


@pytest.mark.parametrize("other", [WORKFLOW, PIN, LEDGER, "tests/test_x.py"])
def test_on_v6_unit_with_anything_else_fails(enabled, other):
    result = build(head_extra={other: b"x\n"}).evaluate()
    assert result.verdict == guard.FAIL, result


def test_on_content_removed_under_a_control_plane_change_fails(enabled):
    world = approved({PIN: b"x\n"}, base_extra={P: PRIMARY, T: COMPANION}, removed=[P])
    fails(world, guard.R_NO_LANE)


def test_on_unreadable_tree_fails(enabled):
    world = plain({"tests/test_x.py": b"x\n"})
    inner = world.fetch

    def fetch(path):
        if "/git/trees/" in path:
            raise OSError("HTTP 500")
        return inner(path)

    world.fetch = fetch
    result = world.evaluate()
    assert result.verdict == guard.FAIL, result
