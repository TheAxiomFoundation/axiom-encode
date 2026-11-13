"""Proposed infrastructure lane of the content-bound merge guard. OFF.

Design section 10, question 0. With ``INFRASTRUCTURE_LANE_ENABLED`` off (the
default) the classifier is unchanged. With it on, a pull request whose whole
delta is infrastructure is ``not_applicable``; RuleSpec content keeps its
lanes, and a pull request that mixes the two is refused.
"""

from __future__ import annotations

import pytest

from axiom_encode import content_merge_guard as guard
from tests.test_content_merge_guard import (
    COMPANION,
    EXECUTABLE,
    INDEX,
    LEDGER,
    LINK,
    PIN,
    PRIMARY,
    SUBMODULE,
    V5_UNIT,
    WORKFLOW,
    M,
    P,
    T,
    build,
    expect,
    fails,
    plain,
)

INFRASTRUCTURE = [
    WORKFLOW,
    ".github/CODEOWNERS",
    ".github/.gitignore",
    PIN,
    ".axiom/toolchain.toml",
    ".axiom/repository-structure.yaml",
    ".axiom/target-validation-passes.json",
    INDEX,
    LEDGER,
    "known-dangling.yaml",
    "oracle-coverage-pending.yaml",
    "household-choices.yaml",
    "tools/generate.py",
    "bulk/compute_matrix.py",
    "tests/test_x.py",
    "programs/us-az/snap/fy-2026.yaml",
    ".gitattributes",
    "CLAUDE.md",
    ".claude/settings.json",
    ".claude/commands/approve.md",
    "NEWFILE.md",
    "readme.md",
]
CONTENT = [
    P,
    T,
    M,
    "us-mo/manual/dss/snap/block-1.yaml",
    "us-zz/newroot/x.yaml",  # a new jurisdiction is never infrastructure
    "us/ca/policies/x.yaml",
    "us-az/Policies/x.yaml",
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
UNKNOWN = ["docs/guide.md", "newtop/x.txt", "sub/README.md", ".clauded/x", "src/x.py"]
JURISDICTIONS = frozenset({"us-az", "us"})


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(guard, "INFRASTRUCTURE_LANE_ENABLED", True)


# --- the classifier ----------------------------------------------------------


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


def test_no_infrastructure_directory_can_be_a_jurisdiction_or_a_root():
    for name in guard.INFRASTRUCTURE_ROOT_DIRECTORIES:
        assert guard._JURISDICTION.fullmatch(name) is None, name
        assert name not in guard.RULESPEC_ATOMIC_ROOTS, name


# --- off (default): nothing changes -------------------------------------------


def test_the_lane_is_off_by_default():
    assert guard.INFRASTRUCTURE_LANE_ENABLED is False


@pytest.mark.parametrize("path", INFRASTRUCTURE)
def test_off_infrastructure_has_no_lane(path):
    fails(plain({path: b"x\n"}), guard.R_NO_LANE)


def test_only_an_exact_true_enables(monkeypatch):
    monkeypatch.setattr(guard, "INFRASTRUCTURE_LANE_ENABLED", 1)
    fails(plain({PIN: b"x\n"}), guard.R_NO_LANE)


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
    world = plain({}, base_extra={LEDGER: b"x\n"}, removed=[LEDGER])
    expect(world, guard.NOT_APPLICABLE, guard.R_INFRASTRUCTURE)


def test_on_allowlist_only_keeps_its_own_reason(enabled):
    expect(
        plain({"README.md": b"x\n"}), guard.NOT_APPLICABLE, guard.R_NO_GUARDED_CONTENT
    )


# --- on: content and mixed pull requests are refused as before -----------------


@pytest.mark.parametrize(
    "path",
    [path for path in CONTENT if path not in {P, T, M}] + UNKNOWN,
)
def test_on_content_or_unknown_path_alone_still_fails(enabled, path):
    result = plain({path: b"x\n"}).evaluate()
    assert result.verdict == guard.FAIL, result


@pytest.mark.parametrize("infrastructure", [WORKFLOW, PIN, LEDGER, "CLAUDE.md"])
@pytest.mark.parametrize(
    "content",
    [P, "us-mo/manual/dss/snap/block-1.yaml", "us-zz/newroot/x.yaml", "docs/x.md"],
)
def test_on_mixed_infrastructure_and_content_fails(enabled, infrastructure, content):
    fails(plain({infrastructure: b"x\n", content: b"x\n"}), guard.R_NO_LANE)


@pytest.mark.parametrize("infrastructure", [WORKFLOW, PIN, LEDGER, "tests/test_x.py"])
def test_on_signed_v5_unit_with_infrastructure_fails(enabled, infrastructure):
    # The shape of rulespec-us#1470: a signed manifest together with a pin, a
    # ledger and a test. Still no lane: split the pull request.
    world = plain(V5_UNIT | {infrastructure: b"x\n"})
    fails(world, guard.R_NO_LANE)


def test_on_v5_unit_alone_is_still_the_earlier_lane(enabled):
    world = plain(V5_UNIT)
    expect(world, guard.NOT_APPLICABLE, guard.R_EARLIER_LANE)


def test_on_v6_unit_alone_still_needs_its_approval(enabled):
    expect(build(), guard.PASS, guard.R_PASS)
    world = build()
    world.reviews([])
    fails(world, guard.R_NO_APPROVAL)


@pytest.mark.parametrize("infrastructure", [WORKFLOW, PIN, LEDGER])
def test_on_v6_unit_with_infrastructure_fails(enabled, infrastructure):
    result = build(head_extra={infrastructure: b"x\n"}).evaluate()
    assert result.verdict == guard.FAIL, result


def test_on_content_removed_under_an_infrastructure_change_fails(enabled):
    world = plain({PIN: b"x\n"}, base_extra={P: PRIMARY, T: COMPANION}, removed=[P])
    fails(world, guard.R_NO_LANE)


@pytest.mark.parametrize("entry", [LINK, SUBMODULE, EXECUTABLE])
@pytest.mark.parametrize("path", [WORKFLOW, "tools/run.sh", LEDGER])
def test_on_non_regular_infrastructure_leaf_fails(enabled, path, entry):
    fails(plain({path: entry}), guard.R_GUARDED_LEAF)


def test_on_unreadable_tree_fails(enabled):
    world = plain({PIN: b"x\n"})
    inner = world.fetch

    def fetch(path):
        if "/git/trees/" in path:
            raise OSError("HTTP 500")
        return inner(path)

    world.fetch = fetch
    result = world.evaluate()
    assert result.verdict == guard.FAIL, result
