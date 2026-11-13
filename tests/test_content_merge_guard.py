"""Adversarial cases for the content-bound merge guard evaluator.

Case numbers refer to the table in ``docs/content-bound-merge-guard.md``.
Every case runs over an injected ``fetch``; no test reaches the network.
"""

import argparse
import base64
import copy
import hashlib
import json
import urllib.error
from pathlib import Path

import pytest

from axiom_encode import content_merge_guard as guard
from axiom_encode.reviewed_pr_admission import AdmissionError

NAME = "rulespec-us"
REPO = "TheAxiomFoundation/" + NAME
ROOT = "/repos/" + REPO
PR = ROOT + "/pulls/42"
CITATION = "us-az/manual/des/faa5/medical/block-7"
P = "us-az/policies/des/faa5/medical/block-7.yaml"
T = "us-az/policies/des/faa5/medical/block-7.test.yaml"
M = ".axiom/encoding-manifests/us-az/policies/des/faa5/medical/block-7.json"
INDEX = ".axiom/index/provisions_to_rules.json"
V5 = "axiom-encode/applied-rulespec/v5"
PRIMARY = b"format: rulespec/v1\nrules: []\n"
COMPANION = b"- name: fixture\n"
MAX = (6076111, "MaxGhenis")
PAVEL = (110687043, "PavelMakarchuk")
OUTSIDER = (555, "outside-author")
# GitHub's own committer for commits made through the API or the web UI. The
# signing workflow's manifest commit is authored by the dispatcher's account
# with this committer (observed on rulespec-us#2045, commit 9be9d91e5).
WEB_FLOW = (19864447, "web-flow")
RUN_ID = "777"
CI_RUN = "888"
SIGNING_RUN = f"{guard.ENCODER_ROOT}/actions/runs/{RUN_ID}"
ENCODER_MAIN = guard.ENCODER_ROOT + "/git/ref/heads/main"
CHECKS = "/check-runs?check_name=validate%20%2F%20validate&filter=latest&per_page=100"
PAGE = "?per_page=100&page=1"
# ``main`` already holds modules under these jurisdictions.
BASE_FILES = {
    "README.md": b"readme\n",
    "us/policies/existing.yaml": b"existing\n",
    "us-az/policies/existing.yaml": b"existing\n",
    "us-ca/regulations/existing.yaml": b"existing\n",
    "us-mo/policies/existing.yaml": b"existing\n",
    "programs/us/snap.yaml": b"existing\n",
    "tests/fixtures/case.yaml": b"existing\n",
}


def oid(label):
    return hashlib.sha1(label.encode()).hexdigest()


B, C1, H = oid("main"), oid("candidate"), oid("signed")
B2, C2, H2 = oid("main-2"), oid("candidate-2"), oid("signed-2")
ENCODER_COMMIT, ENCODER_TIP = oid("encoder-run"), oid("encoder-main")


def blob_id(raw):
    return hashlib.sha1(b"blob %d\x00" % len(raw) + raw).hexdigest()


def manifest_bytes(
    primary=PRIMARY,
    companion=COMPANION,
    mutate=None,
    author=PAVEL,
    dispatcher=PAVEL,
    candidate=C1,
):
    payload = {
        "schema_version": guard.MANIFEST_SCHEMA_V6,
        "generated_at": "2026-11-12T00:00:00+00:00",
        "tool": guard.PROMOTION_TOOL,
        "axiom_encode_version": "0.0.0",
        "axiom_encode_git": {"commit": "e" * 40},
        "run_id": RUN_ID,
        "citation": CITATION,
        "candidate_rulespec_ref": candidate,
        "validation_waiver_set_sha256": "0" * 64,
        "applied_files": [
            {"path": P, "sha256": hashlib.sha256(primary).hexdigest()},
            {"path": T, "sha256": hashlib.sha256(companion).hexdigest()},
        ],
        "source_attestation": {"fixture": True},
        "validation_execution": {"fixture": True},
        "signature": {"fixture": True},
        "attestation_scope": {
            "schema": guard.ATTESTATION_SCOPE_SCHEMA,
            "validation": "encoder-main-complete-source",
            "owner_review": "none",
            "origin": "local-candidate",
            "candidate": {
                "repository": REPO,
                "pull_request": 42,
                "head_sha": candidate,
                "base_sha": B,
                "author_id": author[0],
            },
            "dispatcher": {"id": dispatcher[0], "login": dispatcher[1]},
            "merge_requirement": guard.MERGE_REQUIREMENT,
        },
    }
    if mutate:
        mutate(payload)
    return json.dumps(payload, sort_keys=True).encode()


def v5_manifest(*paths, content=None, **fields):
    """A v5-labelled manifest listing ``paths`` with the digest of their bytes."""
    known = {
        P: PRIMARY,
        T: COMPANION,
        "regulations/mpp/63-503/433.yaml": PRIMARY,
        "regulations/mpp/63-503/433.test.yaml": COMPANION,
    } | (content or {})
    payload = {
        "schema_version": V5,
        "applied_files": [
            {
                "path": path,
                "sha256": hashlib.sha256(known.get(path, b"x\n")).hexdigest(),
            }
            for path in paths
        ],
    }
    return json.dumps(payload | fields, sort_keys=True).encode()


def review(identity, user, commit, state="APPROVED", association="MEMBER", **extra):
    return {
        "id": identity,
        "state": state,
        "commit_id": commit,
        "author_association": association,
        "user": {"id": user[0], "login": user[1], "type": "User"} | extra,
    }


def commit(sha, author=PAVEL, committer=None):
    def account(user):
        return None if user is None else {"id": user[0], "login": user[1]}

    committer = author if committer is None else committer
    return {"sha": sha, "author": account(author), "committer": account(committer)}


class World:
    """A consistent fake of the GitHub REST reads the evaluator makes."""

    def __init__(self):
        self.api = {}
        self.calls = []
        self.trees = {}

    def fetch(self, path):
        self.calls.append(path)
        value = self.api[path]
        return copy.deepcopy(value() if callable(value) else value)

    def tree(self, sha, files, truncated=False):
        """``files`` maps a path to bytes, or to a raw tree entry dict."""
        entries = {}
        for path, value in files.items():
            if isinstance(value, bytes):
                digest = blob_id(value)
                self.api[f"{ROOT}/git/blobs/{digest}"] = {
                    "sha": digest,
                    "size": len(value),
                    "encoding": "base64",
                    "content": base64.encodebytes(value).decode(),
                }
                value = {"type": "blob", "mode": "100644", "sha": digest}
            entries[path] = {"path": path} | value
        self.trees[sha] = entries
        self.api[f"{ROOT}/git/trees/{sha}?recursive=1"] = {
            "truncated": truncated,
            "tree": list(entries.values()),
        }

    def delta(self, base, head):
        old, new = self.trees[base], self.trees[head]
        return [
            {
                "filename": path,
                "status": "added"
                if path not in old
                else "removed"
                if path not in new
                else "modified",
            }
            for path in sorted(old.keys() | new.keys())
            if old.get(path) != new.get(path)
        ]

    def ancestry(self, base, sha, fork, files=None, status="ahead"):
        self.api[f"{ROOT}/compare/{base}...{sha}"] = {
            "status": status,
            "merge_base_commit": {"sha": fork},
            "files": files or [],
        }

    def publish(self, head, base, commits, author=PAVEL):
        """Point the pull request at ``head`` over a ``main`` tip ``base``."""
        files = self.delta(base, head)
        self.api[PR] = {
            "number": 42,
            "state": "open",
            "merged": False,
            "draft": False,
            "changed_files": len(files),
            "commits": len(commits),
            "head": {"sha": head, "ref": "codex/medical", "repo": {"full_name": REPO}},
            "base": {"sha": base, "ref": "main", "repo": {"full_name": REPO}},
            "user": {"id": author[0], "login": author[1]},
        }
        self.api[ROOT + "/git/ref/heads/main"] = {"object": {"sha": base}}
        self.ancestry(base, head, base, files)
        self.api[PR + "/files" + PAGE] = files
        self.api[PR + "/commits" + PAGE] = commits
        self.ci(head)

    def reviews(self, items):
        self.api[PR + "/reviews" + PAGE] = items

    def ci(self, head, **changes):
        self.api[f"{ROOT}/commits/{head}{CHECKS}"] = {
            "total_count": 1,
            "check_runs": [
                {
                    "id": 1,
                    "name": "validate / validate",
                    "app": {"id": 15368},
                    "head_sha": head,
                    "status": "completed",
                    "conclusion": "success",
                    "details_url": (
                        f"https://github.com/{REPO}/actions/runs/{CI_RUN}/job/1"
                    ),
                }
                | changes
            ],
        }
        # The shape of a real pull_request run of repository-checks.yml.
        self.api[f"{ROOT}/actions/runs/{CI_RUN}"] = {
            "id": int(CI_RUN),
            "event": "pull_request",
            "path": ".github/workflows/repository-checks.yml",
            "head_sha": head,
            "run_attempt": 1,
            "status": "completed",
            "conclusion": "success",
            "repository": {"full_name": REPO},
            "head_repository": {"full_name": REPO},
            "pull_requests": [{"number": 42, "head": {"sha": head}}],
        }

    def evaluate(self):
        return guard.evaluate(NAME, 42, fetch=self.fetch)


def build(
    *,
    author=PAVEL,
    dispatcher=PAVEL,
    run_actor=None,
    candidate_author=PAVEL,
    approver=MAX,
    manifest=None,
    mutate=None,
    head_extra=None,
    base_extra=None,
):
    """PR by ``author``; ``dispatcher`` signs; ``approver`` approves the head."""
    world = World()
    base_files = BASE_FILES | (base_extra or {})
    world.tree(B, base_files)
    world.tree(C1, base_files | {P: PRIMARY, T: COMPANION})
    if manifest is None:
        manifest = manifest_bytes(mutate=mutate, author=author, dispatcher=dispatcher)
    world.tree(
        H, base_files | {P: PRIMARY, T: COMPANION, M: manifest} | (head_extra or {})
    )
    commits = [commit(C1, candidate_author), commit(H, dispatcher, WEB_FLOW)]
    world.publish(H, B, commits, author=author)
    world.reviews([review(11, approver, H)])
    run_actor = run_actor or dispatcher
    # The shape of a real targeted-signed-reencode.yml dispatch run.
    world.api[SIGNING_RUN] = {
        "id": int(RUN_ID),
        "event": "workflow_dispatch",
        "path": ".github/workflows/targeted-signed-reencode.yml",
        "head_branch": "main",
        "head_sha": ENCODER_COMMIT,
        "run_attempt": 1,
        "status": "completed",
        "conclusion": "success",
        "repository": {"full_name": "TheAxiomFoundation/axiom-encode"},
        "actor": {"id": run_actor[0], "login": run_actor[1]},
        "triggering_actor": {"id": dispatcher[0], "login": dispatcher[1]},
    }
    world.api[ENCODER_MAIN] = {"object": {"sha": ENCODER_TIP}}
    world.api[f"{guard.ENCODER_ROOT}/compare/{ENCODER_COMMIT}...{ENCODER_TIP}"] = {
        "status": "ahead",
        "merge_base_commit": {"sha": ENCODER_COMMIT},
    }
    return world


def advance(world, *, primary=PRIMARY, manifest=None, extra=None, new_base=False):
    """Move the PR to a second signed head, optionally over a newer ``main``."""
    base = B
    base_files = dict(BASE_FILES)
    commits = [commit(C1), commit(H, PAVEL, WEB_FLOW)]
    candidate = C1
    if new_base:
        base = B2
        base_files |= {"docs/other.md": b"merged elsewhere\n"}
        world.tree(B2, base_files)
        world.ancestry(B, B2, B)  # the manifest's base is an ancestor of main
    if primary != PRIMARY:
        candidate = C2
        commits.append(commit(C2))
        world.tree(C2, base_files | {P: primary, T: COMPANION})
    commits.append(commit(H2, PAVEL, WEB_FLOW))
    manifest = manifest or manifest_bytes(primary=primary, candidate=candidate)
    world.tree(H2, base_files | {P: primary, T: COMPANION, M: manifest} | (extra or {}))
    world.publish(H2, base, commits)
    # The approved commit ``H`` forked from the first ``main`` tip.
    world.ancestry(base, H, B)


def second_read(world, path, later):
    first = world.api[path]
    seen = []

    def read():
        seen.append(1)
        return first if len(seen) == 1 else later

    world.api[path] = read


def expect(world, verdict, reason):
    result = world.evaluate()
    assert (result.verdict, result.reason) == (verdict, reason), result
    return result


def fails(world, reason):
    return expect(world, guard.FAIL, reason)


# --- the passing baseline ---------------------------------------------------


def test_independent_enrolled_approval_on_the_signed_head_passes():
    result = expect(build(), guard.PASS, guard.R_PASS)
    assert result.as_dict() == {
        "schema": guard.VERDICT_SCHEMA,
        "verdict": "pass",
        "reason": "content_bound_approval",
        "detail": "",
        "evidence": {
            "repository": REPO,
            "pull_request": 42,
            "head": H,
            "base": B,
            "primary_sha256": hashlib.sha256(PRIMARY).hexdigest(),
            "companion_sha256": hashlib.sha256(COMPANION).hexdigest(),
            "manifest_sha256": hashlib.sha256(manifest_bytes()).hexdigest(),
            "paths": {"primary": P, "companion": T, "manifest": M},
            "review_id": 11,
            "reviewer_id": MAX[0],
            "dispatcher_id": PAVEL[0],
            "run_id": RUN_ID,
            "candidate_head": C1,
            "approvals": [{"review_id": 11, "reviewer_id": MAX[0], "commit_id": H}],
        },
    }


def test_per_file_waiver_variant_passes():
    def swap(payload):
        del payload["validation_waiver_set_sha256"]
        payload["validation_waiver_bindings"] = {"fixture": True}

    expect(build(mutate=swap), guard.PASS, guard.R_PASS)


# --- F1, R2-1, R2-2: the guard stands aside only on an allowlist or a lane ---


def plain(files, base_extra=None, removed=()):
    """A PR with no promotion unit: ``files`` over the base, one commit."""
    world = World()
    base_files = BASE_FILES | (base_extra or {})
    world.tree(B, base_files)
    head_files = base_files | files
    for path in removed:
        del head_files[path]
    world.tree(H, head_files)
    world.publish(H, B, [commit(H)])
    world.reviews([])
    return world


WORKFLOW = ".github/workflows/repository-checks.yml"
PIN = ".axiom/workflow-toolchain.toml"
LEDGER = "known-validation-gaps.yaml"
LINK = {"type": "blob", "mode": "120000", "sha": "d" * 40}
SUBMODULE = {"type": "commit", "mode": "160000", "sha": "d" * 40}
EXECUTABLE = {"type": "blob", "mode": "100755", "sha": "d" * 40}
V5_UNIT = {P: PRIMARY, T: COMPANION, M: v5_manifest(P, T)}


def test_docs_only_pr_is_not_applicable():
    world = plain(
        {
            "README.md": b"changed\n",
            "NOTICE": b"changed\n",
            "LICENSE": b"changed\n",
            "LICENSE-CODE": b"changed\n",
            ".gitignore": b"changed\n",
        }
    )
    world.api[PR]["draft"] = True
    result = expect(world, guard.NOT_APPLICABLE, guard.R_NO_GUARDED_CONTENT)
    assert result.evidence == {
        "repository": REPO,
        "pull_request": 42,
        "head": H,
        "base": B,
    }
    # Applicability is decided from the two trees alone.
    assert sorted(world.calls) == sorted(
        [
            PR,
            ROOT + "/git/ref/heads/main",
            f"{ROOT}/compare/{B}...{H}",
            f"{ROOT}/git/trees/{B}?recursive=1",
            f"{ROOT}/git/trees/{H}?recursive=1",
        ]
    )


def test_removed_allowlisted_file_is_not_applicable():
    world = plain({}, base_extra={"NOTICE": b"x\n"}, removed=["NOTICE"])
    expect(world, guard.NOT_APPLICABLE, guard.R_NO_GUARDED_CONTENT)


@pytest.mark.parametrize(
    "path",
    [
        # R2-2: content the encoder's atomic-root list does not name.
        "us-mo/manual/dss/snap/block-1.yaml",
        "us-mo/manual/dss/snap/block-1.test.yaml",
        "programs/us-az/snap/fy-2026.yaml",
        "us-zz/newroot/x.yaml",
        "us/ca/policies/x.yaml",
        "us-az/Policies/x.yaml",
        "us-az/policies/x.yaml",
        "us-az/statutes/x.yaml",
        "us-az/regulations/x.yaml",
        "us-az/legislation/x.yaml",
        "policies/x.yaml",
        "us-az/policies/notes.md",
        "us-az/x.yaml",
        # R2-2: configuration, pins and ledgers that validation reads.
        PIN,
        ".axiom/toolchain.toml",
        ".axiom/repository-structure.yaml",
        ".axiom/target-validation-passes.json",
        INDEX,
        ".axiom",
        ".axiom/encoding-manifests",
        "us-az/.axiom",
        "us-az/.axiom/encoding-manifests",
        "us-co/.axiom/notes.md",
        LEDGER,
        "known-dangling.yaml",
        "oracle-coverage-pending.yaml",
        "household-choices.yaml",
        WORKFLOW,
        ".github/CODEOWNERS",
        # Tooling that CI executes, and anything new at the top level.
        "tools/generate.py",
        "bulk/compute_matrix.py",
        "tests/test_x.py",
        "docs/guide.md",
        "newtop/x.txt",
        "NEWFILE.md",
        "readme.md",  # the allowlist is exact, also in case
        # R3-1: Git attributes change checked-out bytes and review diffs.
        ".gitattributes",
        "us-az/.gitattributes",
        # R3-2: files that steer the agent sessions.
        "CLAUDE.md",
        ".claude/settings.json",
        ".claude/commands/approve.md",
        # The allowlist matches the whole path, not the basename.
        "sub/README.md",
        "us-az/CLAUDE.md",
        "us-az/README.md",
        "docs/LICENSE",
        "us-az/policies/NOTICE",
        ".github/.gitignore",
        ".claude",
        ".clauded/x",
    ],
)
def test_r2_2_guarded_path_with_no_manifest_has_no_lane(path):
    fails(plain({path: b"x\n"}), guard.R_NO_LANE)


def test_r2_2_guarded_path_next_to_allowlisted_paths_has_no_lane():
    fails(plain({"README.md": b"x\n", PIN: b"x\n"}), guard.R_NO_LANE)


@pytest.mark.parametrize(
    "path",
    [
        "us-az/manual",
        "us-az/policies2",
        ".axiom/encoding-manifests",
        ".axiom",
        "us-az/.axiom/encoding-manifests",
        "us-az/.axiom",
        "us-az",
        "us-az/policies",
        "policies",
        P,
        M,
        "README.md",  # also an allowlisted path must be a regular file
        ".claude/link",
        "scripts/run.sh",
    ],
)
@pytest.mark.parametrize("entry", [LINK, SUBMODULE, EXECUTABLE])
def test_any_changed_leaf_that_is_not_a_regular_file_fails(path, entry):
    world = plain({path: entry, "elsewhere/manifest.json": manifest_bytes()})
    fails(world, guard.R_GUARDED_LEAF)


def test_unchanged_non_regular_leaf_on_main_is_not_judged():
    world = plain({"README.md": b"x\n"}, base_extra={"bulk/run.py": EXECUTABLE})
    expect(world, guard.NOT_APPLICABLE, guard.R_NO_GUARDED_CONTENT)


def test_f1a_content_and_workflow_with_no_manifest_fails():
    world = plain(
        {P: PRIMARY, T: COMPANION, WORKFLOW: b"jobs: {}\n", ".github/CODEOWNERS": b"\n"}
    )
    fails(world, guard.R_NO_LANE)


@pytest.mark.parametrize(
    "schema", ["axiom-encode/applied-rulespec/v7", "", None, 6, ["v6"]]
)
def test_f1b_relabelled_manifest_fails(schema):
    def relabel(payload):
        payload["schema_version"] = schema

    world = plain({P: PRIMARY, T: COMPANION, M: manifest_bytes(mutate=relabel)})
    fails(world, guard.R_UNKNOWN_SCHEMA)


def test_f1b_manifest_without_a_schema_fails():
    world = plain({P: PRIMARY, T: COMPANION, M: b'{"applied_files": []}'})
    fails(world, guard.R_UNKNOWN_SCHEMA)


def test_f1_removed_content_with_no_manifest_fails():
    fails(plain({}, base_extra={P: PRIMARY}, removed=[P]), guard.R_NO_LANE)


def test_f1_removed_manifest_fails():
    world = plain({}, base_extra={M: v5_manifest()}, removed=[M])
    fails(world, guard.R_MANIFEST_REMOVED)


def test_f1_mixed_v5_and_v6_manifests_fail():
    other = M.replace("block-7", "block-8")
    fails(build(head_extra={other: v5_manifest()}), guard.R_MIXED_SCHEMAS)


# The earlier (v5) lane. Its verdict rests on the lane's scope and on this
# PR's own green ``validate / validate`` run, never on the manifest's label.


def test_v5_promotion_shape_of_rulespec_us_2045_is_the_earlier_lane():
    # rulespec-us#2045: a v5 promotion manifest at the repository root and
    # its two module files, with repository-relative applied paths.
    world = plain(V5_UNIT)
    result = expect(world, guard.NOT_APPLICABLE, guard.R_EARLIER_LANE)
    assert f"{ROOT}/commits/{H}{CHECKS}" in world.calls
    assert f"{ROOT}/actions/runs/{CI_RUN}" in world.calls
    assert world.calls.count(PR) == 2  # re-read
    assert "primary_sha256" not in result.evidence


LOCAL_P = "us-ca/regulations/mpp/63-503/433.yaml"
LOCAL_T = "us-ca/regulations/mpp/63-503/433.test.yaml"
LOCAL_M = "us-ca/.axiom/encoding-manifests/regulations/mpp/63-503/433.json"
LOCAL_MANIFEST = v5_manifest(
    "regulations/mpp/63-503/433.yaml", "regulations/mpp/63-503/433.test.yaml"
)


def test_v5_lane_with_a_jurisdiction_local_manifest():
    # The layout of most manifests on rulespec-us main: the manifest lives
    # under the jurisdiction and lists paths relative to it.
    world = plain({LOCAL_P: PRIMARY, LOCAL_T: COMPANION, LOCAL_M: LOCAL_MANIFEST})
    expect(world, guard.NOT_APPLICABLE, guard.R_EARLIER_LANE)


def test_v5_lane_modifying_a_module():
    older = {P: b"older\n", T: b"older test\n", M: v5_manifest(P, T, generated_at="1")}
    expect(plain(V5_UNIT, base_extra=older), guard.NOT_APPLICABLE, guard.R_EARLIER_LANE)


@pytest.mark.parametrize(
    "primary",
    [
        "us-mo/policies/dss/snap/block-1.yaml",
        "us/policies/ca/x.yaml",
        "policies/x.yaml",
    ],
)
def test_v5_lane_admits_atomic_roots_at_either_layout_and_any_depth(primary):
    world = plain(
        {
            primary: b"x\n",
            guard.MANIFEST_PREFIX + primary[:-5] + ".json": v5_manifest(primary),
        }
    )
    expect(world, guard.NOT_APPLICABLE, guard.R_EARLIER_LANE)


@pytest.mark.parametrize("root", sorted(guard.RULESPEC_ATOMIC_ROOTS))
@pytest.mark.parametrize(
    "local", [False, True], ids=["root-manifest", "local-manifest"]
)
def test_v5_lane_admits_each_atomic_root_with_both_manifest_layouts(root, local):
    primary = f"us-az/{root}/nonexistent-review-fixture.yaml"
    listed = primary.split("/", 1)[1] if local else primary
    manifest = (
        ("us-az/" if local else "") + guard.MANIFEST_PREFIX + listed[:-5] + ".json"
    )
    world = plain({primary: b"x\n", manifest: v5_manifest(listed)})
    expect(world, guard.NOT_APPLICABLE, guard.R_EARLIER_LANE)


@pytest.mark.parametrize("root", ["tools", "manual", "programs", "Policies", "newroot"])
@pytest.mark.parametrize(
    "local", [False, True], ids=["root-manifest", "local-manifest"]
)
@pytest.mark.parametrize("suffix", [".yaml", ".test.yaml"])
def test_v5_lane_refuses_non_atomic_nested_roots_even_with_own_manifest(
    root, local, suffix
):
    path = f"us-az/{root}/nonexistent-review-fixture{suffix}"
    listed = path.split("/", 1)[1] if local else path
    primary = (
        listed.removesuffix(".test.yaml") + ".yaml"
        if suffix == ".test.yaml"
        else listed
    )
    manifest = (
        ("us-az/" if local else "") + guard.MANIFEST_PREFIX + primary[:-5] + ".json"
    )
    world = plain({path: b"x\n", manifest: v5_manifest(listed)})
    fails(world, guard.R_NO_LANE)


@pytest.mark.parametrize("listing", [(), (P, T)])
def test_r4_v5_manifest_that_lists_no_changed_module_file_fails(listing):
    # An empty listing, or a listing of modules the PR does not change.
    base = {P: PRIMARY, T: COMPANION} if listing else {}
    world = plain({M: v5_manifest(*listing)}, base_extra=base)
    fails(world, guard.R_CONTENT_NOT_COVERED)


def test_r4_every_changed_manifest_must_list_a_changed_module_file():
    idle = M.replace("block-7", "existing").replace("des/faa5/medical/", "")
    assert idle == guard.MANIFEST_PREFIX + "us-az/policies/existing.json"
    world = plain(V5_UNIT | {idle: v5_manifest()})
    fails(world, guard.R_CONTENT_NOT_COVERED)


def test_r4_modified_module_needs_the_digest_of_its_new_bytes():
    older = {P: b"older\n", T: COMPANION, M: v5_manifest(P, T, generated_at="1")}
    stale = v5_manifest(P, T, content={P: b"older\n"})
    world = plain({P: PRIMARY, M: stale}, base_extra=older)
    fails(world, guard.R_CONTENT_NOT_COVERED)


@pytest.mark.parametrize(
    "extra",
    [
        WORKFLOW,
        ".github/CODEOWNERS",
        PIN,
        LEDGER,
        INDEX,
        "programs/us-az/snap/fy-2026.yaml",
        "tools/generate.py",
        "README.md",  # the lane claims the whole delta, allowlisted paths too
        "us-az/policies/notes.md",
        "us-az/x.yaml",
        "us-az/.hidden/policies/x.yaml",
        "us-co/.axiom/notes.yaml",
        "Us-az/policies/x.yaml",
        "tests/policies/x.yaml",
    ],
)
def test_r2_1_i_v5_label_with_any_path_outside_the_lane_has_no_lane(extra):
    # Every extra is listed by the hand-written manifest: coverage is not the
    # control here, the lane's scope is.
    manifest = v5_manifest(P, T, extra)
    world = plain({P: PRIMARY, T: COMPANION, M: manifest, extra: b"x\n"})
    fails(world, guard.R_NO_LANE)


def test_r2_1_i_the_reproduced_bundle_has_no_lane():
    world = plain(
        V5_UNIT
        | {
            WORKFLOW: b"jobs: {}\n",
            ".github/CODEOWNERS": b"\n",
            PIN: b"x\n",
            LEDGER: b"x\n",
        }
    )
    fails(world, guard.R_NO_LANE)


def test_r2_1_ii_v5_lane_reads_ci_and_fails_without_it():
    world = plain(V5_UNIT)
    del world.api[f"{ROOT}/commits/{H}{CHECKS}"]
    fails(world, guard.R_READ_FAILED)


@pytest.mark.parametrize(
    "changes",
    [
        {"conclusion": "failure"},
        {"status": "in_progress", "conclusion": None},
        {"app": {"id": 99999}},
        {"head_sha": C1},
    ],
)
def test_r2_1_ii_v5_lane_needs_the_required_check_green(changes):
    world = plain(V5_UNIT)
    world.ci(H, **changes)
    fails(world, guard.R_CI)


@pytest.mark.parametrize(
    "field,value",
    [
        ("event", "workflow_dispatch"),
        ("pull_requests", []),
        ("repository", {"full_name": "fork/rulespec-us"}),
        ("path", ".github/workflows/imitation.yml"),
    ],
)
def test_r2_1_ii_v5_lane_needs_this_prs_pull_request_run(field, value):
    world = plain(V5_UNIT)
    world.api[f"{ROOT}/actions/runs/{CI_RUN}"][field] = value
    fails(world, guard.R_CI)


def test_r2_1_iii_hand_edit_with_a_doctored_v5_manifest_rests_on_ci():
    # An unsigned v5 manifest cannot be told from a signed one here. The
    # required check verifies the signature, so the verdict follows CI.
    older = {P: b"older\n", T: COMPANION, M: v5_manifest(P, T, generated_at="1")}
    edit = b"hand edit\n"
    world = plain({P: edit, M: v5_manifest(P, T, content={P: edit})}, base_extra=older)
    world.ci(H, conclusion="failure")
    fails(world, guard.R_CI)


def deleting_manifest(*paths):
    items = [{"path": path, "deleted": True} for path in paths]
    return json.dumps({"schema_version": V5, "applied_files": items}).encode()


def test_r2_1_iv_v5_manifest_that_lists_its_own_deletion_rests_on_ci():
    world = plain(
        {M: deleting_manifest(P, T)},
        base_extra={P: PRIMARY, T: COMPANION},
        removed=[P, T],
    )
    world.ci(H, conclusion="failure")
    fails(world, guard.R_CI)
    world.ci(H)
    expect(world, guard.NOT_APPLICABLE, guard.R_EARLIER_LANE)


def test_r2_1_iv_v5_manifest_cannot_delete_another_module():
    other = "us-az/policies/des/other.yaml"
    world = plain(
        {M: deleting_manifest(P, other)},
        base_extra={P: PRIMARY, other: b"other\n"},
        removed=[P, other],
    )
    fails(world, guard.R_CONTENT_NOT_COVERED)


# Round 3 hardening of the earlier lane: digest, position, base jurisdiction.


def test_r3_v5_manifest_must_list_the_digest_of_the_head_bytes():
    wrong = v5_manifest(P, T, content={P: b"other bytes\n"})
    fails(plain({P: PRIMARY, T: COMPANION, M: wrong}), guard.R_CONTENT_NOT_COVERED)
    missing = json.dumps(
        {"schema_version": V5, "applied_files": [{"path": P}, {"path": T}]}
    )
    world = plain({P: PRIMARY, T: COMPANION, M: missing.encode()})
    fails(world, guard.R_CONTENT_NOT_COVERED)


def test_r3_modules_covered_by_an_unrelated_manifest_position():
    unit = {P: PRIMARY, T: COMPANION}
    elsewhere = guard.MANIFEST_PREFIX + "us-az/policies/unrelated.json"
    fails(plain(unit | {elsewhere: v5_manifest(P, T)}), guard.R_CONTENT_NOT_COVERED)
    foreign = guard.MANIFEST_PREFIX + "zz/unrelated.json"
    fails(plain(unit | {foreign: v5_manifest(P, T)}), guard.R_NO_LANE)


def test_r3_companion_belongs_to_the_manifest_of_its_primary():
    own = M.replace(".json", ".test.json")
    world = plain({P: PRIMARY, T: COMPANION, M: v5_manifest(P), own: v5_manifest(T)})
    fails(world, guard.R_CONTENT_NOT_COVERED)


@pytest.mark.parametrize(
    "primary", ["zz/a/b.yaml", "ci/workflows/x.yaml", "us-zz/policies/x.yaml"]
)
def test_r3_module_under_a_jurisdiction_that_main_lacks_has_no_lane(primary):
    manifest = guard.MANIFEST_PREFIX + primary[:-5] + ".json"
    fails(plain({primary: b"x\n", manifest: v5_manifest(primary)}), guard.R_NO_LANE)


@pytest.mark.parametrize(
    "path",
    [
        "tools/.axiom/encoding-manifests/x.json",
        "programs/.axiom/encoding-manifests/us/snap.json",
        ".axiom/encoding-manifests/x.json",
        ".axiom/encoding-manifests/us-az/policies/x.txt",
        ".axiom/encoding-manifests/us-az/x.json",
        "us-zz/.axiom/encoding-manifests/policies/x.json",
    ],
)
def test_r3_v5_manifest_alone_at_a_position_of_no_module_has_no_lane(path):
    fails(plain({path: v5_manifest()}), guard.R_NO_LANE)


def test_r3_removed_file_must_be_listed_by_its_own_manifest():
    older = {P: PRIMARY, T: COMPANION, M: v5_manifest(P, T, generated_at="1")}
    world = plain({M: v5_manifest(P)}, base_extra=older, removed=[T])
    fails(world, guard.R_CONTENT_NOT_COVERED)


def test_r3_lookalike_directory_is_not_a_manifest_directory():
    # Past the length of the real prefix this path reads as a module position.
    path = "us-az/encoding-manifests/pus-az/policies/existing.json"
    assert path[len(guard.MANIFEST_PREFIX) :] == "us-az/policies/existing.json"
    fails(plain({path: v5_manifest()}), guard.R_NO_LANE)


def test_r3_jurisdictions_come_from_the_base_not_from_the_head():
    assert guard._base_jurisdictions(
        {path: {} for path in [*BASE_FILES, "policies/root.yaml", ".x/a/b.yaml"]}
    ) == {"us", "us-az", "us-ca", "us-mo"}


@pytest.mark.parametrize(
    "manifest",
    [
        v5_manifest(P),  # the companion is not covered
        v5_manifest(),
        json.dumps({"schema_version": V5}).encode(),
        json.dumps({"schema_version": V5, "applied_files": [P, T]}).encode(),
    ],
)
def test_v5_lane_with_uncovered_content_fails(manifest):
    world = plain({P: PRIMARY, T: COMPANION, M: manifest})
    fails(world, guard.R_CONTENT_NOT_COVERED)


def test_jurisdiction_local_manifest_does_not_cover_another_jurisdiction():
    # Repository-relative paths with the right digests: in this layout they
    # read as us-ca/us-ca/..., so they cover nothing.
    manifest = v5_manifest(
        LOCAL_P, LOCAL_T, content={LOCAL_P: PRIMARY, LOCAL_T: COMPANION}
    )
    world = plain({LOCAL_P: PRIMARY, LOCAL_T: COMPANION, LOCAL_M: manifest})
    fails(world, guard.R_CONTENT_NOT_COVERED)


def test_jurisdiction_local_manifest_alone_is_guarded():
    relabelled = json.dumps({"schema_version": "x"}).encode()
    fails(plain({LOCAL_M: relabelled}), guard.R_UNKNOWN_SCHEMA)


@pytest.mark.parametrize(
    "path",
    [
        "us-az/encoding-manifests/policies/x.json",
        "encoding-manifests/x.json",
        "us/az/.axiom/encoding-manifests/x.json",
        ".axiom/manifests/x.json",
    ],
)
def test_manifest_outside_a_manifest_directory_is_not_a_manifest(path):
    world = plain({P: PRIMARY, T: COMPANION, path: v5_manifest(P, T)})
    fails(world, guard.R_NO_LANE)


def test_v5_lane_refuses_when_the_head_or_main_moves():
    world = plain(V5_UNIT)
    moved = copy.deepcopy(world.api[PR])
    moved["head"]["sha"] = H2
    second_read(world, PR, moved)
    fails(world, guard.R_CHANGED)
    world = plain(V5_UNIT)
    second_read(world, ROOT + "/git/ref/heads/main", {"object": {"sha": B2}})
    fails(world, guard.R_CHANGED)


def test_v6_manifest_at_a_jurisdiction_local_path_fails():
    world = plain({P: PRIMARY, T: COMPANION, LOCAL_M: manifest_bytes()})
    fails(world, guard.R_MANIFEST_PATH)


def test_f1_v6_unit_with_any_extra_path_fails():
    for extra in (WORKFLOW, ".github/CODEOWNERS", "README.md", "docs/new.md", PIN):
        fails(build(head_extra={extra: b"changed\n"}), guard.R_SCOPE_PATHS)


def test_f1_v6_unit_with_a_second_module_fails():
    second = {"us-az/policies/des/other.yaml": b"x\n"}
    fails(build(head_extra=second), guard.R_SCOPE_PATHS)


# --- F2: the index is not part of a v6 unit ---------------------------------


def test_f2_index_change_in_a_v6_unit_fails():
    fails(build(head_extra={INDEX: b"{}\n"}), guard.R_SCOPE_PATHS)


def test_f2_new_head_that_changes_only_the_index_after_approval_fails():
    world = build()
    advance(world, extra={INDEX: b'{"changed": true}\n'})
    fails(world, guard.R_SCOPE_PATHS)


def test_f2_approval_on_a_commit_that_also_changed_the_index_does_not_count():
    world = build()
    advance(world, new_base=True)
    world.tree(
        H,
        BASE_FILES | {P: PRIMARY, T: COMPANION, M: manifest_bytes(), INDEX: b"{}\n"},
    )
    fails(world, guard.R_NOT_CONTENT_BOUND)


# --- cases 1-5: the approval is bound to content ----------------------------


def test_case_01_approval_before_signing_does_not_count():
    world = build()
    world.reviews([review(11, MAX, C1)])
    world.ancestry(B, C1, B)
    fails(world, guard.R_NOT_CONTENT_BOUND)


def test_case_02_primary_changes_after_approval():
    world = build()
    advance(world, primary=PRIMARY + b"#")  # re-signed, so the manifest agrees
    fails(world, guard.R_NOT_CONTENT_BOUND)


def test_case_02_primary_changes_by_one_byte_without_resign():
    world = build()
    advance(world, primary=PRIMARY + b"#", manifest=manifest_bytes())
    fails(world, guard.R_APPLIED_FILES)


def test_case_03_resign_voids_the_approval():
    def later(payload):
        payload["generated_at"] = "2026-11-13T00:00:00+00:00"

    world = build()
    advance(world, manifest=manifest_bytes(mutate=later))
    fails(world, guard.R_NOT_CONTENT_BOUND)


def test_case_04_base_only_update_retains_the_approval():
    world = build()
    advance(world, new_base=True)
    result = expect(world, guard.PASS, guard.R_PASS)
    assert result.evidence["head"] == H2 and result.evidence["base"] == B2
    assert result.evidence["approvals"][0]["commit_id"] == H


def test_case_05_base_update_that_adds_a_workflow_fails_scope():
    world = build()
    advance(world, new_base=True, extra={".github/workflows/x.yml": b"on: push\n"})
    fails(world, guard.R_SCOPE_PATHS)


def test_approved_commit_without_a_merge_base_fails():
    world = build()
    advance(world, new_base=True)
    world.ancestry(B2, H, None)
    fails(world, guard.R_DELTA_INCOMPLETE)


def test_approval_on_a_commit_outside_the_pr_does_not_count():
    world = build()
    world.reviews([review(11, MAX, oid("elsewhere"))])
    fails(world, guard.R_NOT_CONTENT_BOUND)


def test_approved_commit_with_a_wider_delta_does_not_count():
    world = build()
    advance(world, new_base=True)
    world.tree(
        H,
        BASE_FILES
        | {P: PRIMARY, T: COMPANION, M: manifest_bytes(), "extra.md": b"x\n"},
    )
    fails(world, guard.R_NOT_CONTENT_BOUND)


# --- cases 6-9, 27 and F4: who may approve ----------------------------------


def test_case_06_author_is_the_only_approver():
    fails(build(author=MAX, approver=MAX), guard.R_IS_AUTHOR)


def test_case_07_dispatcher_is_the_only_approver():
    fails(build(dispatcher=MAX, approver=MAX), guard.R_IS_DISPATCHER)


def test_case_07_rerun_original_actor_is_the_only_approver():
    # Max dispatched; Pavel re-ran, so Pavel is ``triggering_actor``.
    world = build(author=OUTSIDER, dispatcher=PAVEL, run_actor=MAX, approver=MAX)
    fails(world, guard.R_IS_DISPATCHER)


def test_case_08_write_collaborator_who_is_not_enrolled():
    world = build()
    world.reviews([review(11, (424242, "collaborator"), H, association="COLLABORATOR")])
    fails(world, guard.R_NOT_ENROLLED)


@pytest.mark.parametrize(
    "user,extra",
    [
        ((424242, "MaxGhenis"), {}),  # recycled login, another account
        ((MAX[0], "renamed-account"), {}),  # enrolled id, another login
        (MAX, {"type": "Bot"}),
        ((424242, None), {}),  # no login must not equal "not enrolled"
        (MAX, {"type": None}),
    ],
)
def test_case_09_enrolment_is_id_login_and_account_type(user, extra):
    world = build()
    world.reviews([review(11, user, H, **extra)])
    fails(world, guard.R_NOT_ENROLLED)


def test_case_27_enrolled_owner_authored_the_candidate_commits():
    world = build(author=OUTSIDER, candidate_author=MAX, approver=MAX)
    fails(world, guard.R_AUTHORED_COMMITS)


@pytest.mark.parametrize("role", ["author", "committer"])
@pytest.mark.parametrize("index", [0, 1])
def test_f4_approver_wrote_any_commit_of_the_pr(index, role):
    # Index 1 is the commit that adds only the manifest.
    world = build(author=OUTSIDER)
    world.api[PR + "/commits" + PAGE][index][role] = {"id": MAX[0], "login": MAX[1]}
    fails(world, guard.R_AUTHORED_COMMITS)


def test_f4_approver_clicked_update_branch():
    # Stated limit: the merge commit is attributed to the approver.
    world = build()
    advance(world, new_base=True)
    world.api[PR + "/commits" + PAGE][-1] = commit(H2, MAX, WEB_FLOW)
    fails(world, guard.R_AUTHORED_COMMITS)


@pytest.mark.parametrize(
    "change",
    [
        {"author": None, "committer": None},
        {"author": None, "committer": {"id": WEB_FLOW[0], "login": WEB_FLOW[1]}},
        {"committer": None},
    ],
)
@pytest.mark.parametrize("index", [0, 1])
def test_r2_3_unattributed_commit_fails(index, change):
    # Git author metadata is free text. When GitHub attributes a commit to no
    # account, the evaluator cannot tell that the approver did not write it.
    world = build(author=OUTSIDER, approver=MAX)
    world.api[PR + "/commits" + PAGE][index].update(change)
    fails(world, guard.R_UNATTRIBUTED)


def test_r2_3_web_flow_committer_with_an_attributed_author_is_allowed():
    world = build()
    assert world.api[PR + "/commits" + PAGE][1]["committer"]["id"] == WEB_FLOW[0]
    expect(world, guard.PASS, guard.R_PASS)


@pytest.mark.parametrize(
    "change",
    [
        lambda world: world.api[PR].update(commits=251),
        # The head is not in the list.
        lambda world: world.api[PR + "/commits" + PAGE].__setitem__(1, commit(C2)),
        # A duplicate entry, with the count adjusted to match.
        lambda world: (
            world.api[PR + "/commits" + PAGE].append(commit(C1)),
            world.api[PR].update(commits=3),
        ),
        lambda world: world.api[PR + "/commits" + PAGE][0].update(sha="main"),
        lambda world: world.api[PR + "/commits" + PAGE][0].update(
            author={"id": "110687043"}
        ),
    ],
    ids=["count", "head-missing", "duplicate", "not-a-sha", "author-id-not-int"],
)
def test_commit_list_must_be_complete_and_well_formed(change):
    world = build()
    change(world)
    fails(world, guard.R_COMMITS)


# --- cases 10-12: review lifecycle ------------------------------------------


def test_case_10_dismissed_approval():
    world = build()
    world.reviews([review(11, MAX, H), review(12, MAX, H, state="DISMISSED")])
    fails(world, guard.R_NO_APPROVAL)


@pytest.mark.parametrize(
    "user,association",
    [(PAVEL, "MEMBER"), ((424242, "collaborator"), "COLLABORATOR"), (OUTSIDER, None)],
)
def test_case_11_changes_requested_blocks(user, association):
    world = build()
    blocking = review(12, user, H, state="CHANGES_REQUESTED", association=association)
    world.reviews([review(11, MAX, H), blocking])
    fails(world, guard.R_CHANGES_REQUESTED)


def test_case_11_outside_account_cannot_block():
    world = build()
    drive_by = review(12, OUTSIDER, H, state="CHANGES_REQUESTED", association="NONE")
    world.reviews([review(11, MAX, H), drive_by])
    expect(world, guard.PASS, guard.R_PASS)


def test_case_12_later_comment_keeps_the_approval():
    world = build()
    world.reviews([review(11, MAX, H), review(12, MAX, H, state="COMMENTED")])
    expect(world, guard.PASS, guard.R_PASS)


def test_unknown_review_state_fails():
    world = build()
    world.reviews([review(11, MAX, H), review(12, OUTSIDER, H, state="NEW_STATE")])
    fails(world, guard.R_REVIEWS)


@pytest.mark.parametrize("identity", ["11", True, 0, None])
def test_review_identity_must_be_a_positive_integer(identity):
    world = build()
    world.reviews([review(identity, MAX, H)])
    fails(world, guard.R_REVIEWS)


def test_reviewer_identity_must_be_a_positive_integer():
    world = build()
    world.reviews([review(11, ("6076111", "MaxGhenis"), H)])
    fails(world, guard.R_REVIEWS)


@pytest.mark.parametrize("state", ["COMMENTED", "CHANGES_REQUESTED", "APPROVED"])
def test_review_from_a_deleted_account_is_ignored(state):
    world = build()
    ghost = review(12, OUTSIDER, H, state=state) | {"user": None}
    world.reviews([review(11, MAX, H), ghost])
    expect(world, guard.PASS, guard.R_PASS)


def test_approval_from_a_deleted_account_never_counts():
    world = build()
    world.reviews([review(11, MAX, H) | {"user": None}])
    fails(world, guard.R_NO_APPROVAL)


def test_no_review_at_all():
    world = build()
    world.reviews([])
    fails(world, guard.R_NO_APPROVAL)


# --- cases 13, 14, 16, 20, 21, 23: scope and canonical paths ----------------


@pytest.mark.parametrize(
    "entry",
    [
        {"type": "blob", "mode": "120000"},
        {"type": "blob", "mode": "100755"},
        {"type": "commit", "mode": "160000"},
    ],
)
@pytest.mark.parametrize("path", [P, T, M])
def test_case_13_unit_file_is_not_a_regular_blob(path, entry):
    world = build(head_extra={path: entry | {"sha": "d" * 40}})
    fails(world, guard.R_GUARDED_LEAF)


def test_case_14_primary_at_a_non_canonical_path():
    def elsewhere(payload):
        payload["citation"] = "us-az/manual/des/faa5/medical/block-8"

    fails(build(mutate=elsewhere), guard.R_MANIFEST_PATH)


def test_case_14_citation_of_another_country():
    def foreign(payload):
        payload["citation"] = "uk/manual/des/faa5/medical/block-7"

    fails(build(mutate=foreign), guard.R_CITATION)


def test_case_16_pr_rewrites_the_required_workflow():
    fails(build(head_extra={WORKFLOW: b"jobs: {}\n"}), guard.R_SCOPE_PATHS)


def test_case_20_only_a_manifest():
    world = build()
    world.tree(H, BASE_FILES | {M: manifest_bytes()})
    world.publish(H, B, [commit(C1), commit(H, PAVEL, WEB_FLOW)])
    fails(world, guard.R_SCOPE_PATHS)


def test_case_20_two_v6_manifests():
    second = M.replace("block-7", "block-8")
    fails(build(head_extra={second: manifest_bytes()}), guard.R_MULTIPLE_V6)


def test_case_21_applied_files_digest_differs():
    def wrong(payload):
        payload["applied_files"][0]["sha256"] = "1" * 64

    fails(build(mutate=wrong), guard.R_APPLIED_FILES)


@pytest.mark.parametrize("path", [P, T, M])
def test_case_23_unit_path_already_exists_on_main(path):
    fails(build(base_extra={path: b"older bytes\n"}), guard.R_SCOPE_NOT_NEW)


@pytest.mark.parametrize("evidence", ["files", "compare", "count"])
def test_mutable_file_lists_cannot_hide_a_change(evidence):
    world = build()
    hidden = [item for item in world.api[PR + "/files" + PAGE] if item["filename"] != T]
    if evidence == "files":
        world.api[PR + "/files" + PAGE] = hidden
        world.api[PR]["changed_files"] = len(hidden)
    elif evidence == "compare":
        world.api[f"{ROOT}/compare/{B}...{H}"]["files"] = hidden
    else:
        world.api[PR]["changed_files"] = 4
    fails(world, guard.R_DELTA_DISAGREES)


def test_v6_unit_in_a_subdivision_repository_is_unsupported():
    world = build()
    text = json.dumps(world.api).replace(NAME, "rulespec-us-tn")
    world.api = json.loads(text)
    result = guard.evaluate("rulespec-us-tn", 42, fetch=world.fetch)
    assert (result.verdict, result.reason) == (
        guard.FAIL,
        guard.R_UNSUPPORTED_REPOSITORY,
    )
    assert result.evidence["repository"] == "TheAxiomFoundation/rulespec-us-tn"


# --- cases 19, 26, F5 (manifest fields) and the exact v6 shape ---------------


def test_case_19_manifest_names_another_pr():
    def other(payload):
        payload["attestation_scope"]["candidate"]["pull_request"] = 43

    fails(build(mutate=other), guard.R_MANIFEST_CANDIDATE)


@pytest.mark.parametrize(
    "field,value", [("repository", "TheAxiomFoundation/rulespec-uk"), ("author_id", 7)]
)
def test_case_19_manifest_names_another_repository_or_author(field, value):
    def other(payload):
        payload["attestation_scope"]["candidate"][field] = value

    fails(build(mutate=other), guard.R_MANIFEST_CANDIDATE)


def test_f5_candidate_ref_must_be_the_candidate_head():
    def other(payload):
        payload["candidate_rulespec_ref"] = H

    fails(build(mutate=other), guard.R_MANIFEST_CANDIDATE)


def test_f5_candidate_head_must_be_a_commit_of_this_pr():
    def other(payload):
        payload["candidate_rulespec_ref"] = C2
        payload["attestation_scope"]["candidate"]["head_sha"] = C2

    world = build(mutate=other)
    world.tree(C2, BASE_FILES | {P: PRIMARY, T: COMPANION})
    fails(world, guard.R_MANIFEST_CANDIDATE)


@pytest.mark.parametrize(
    "files",
    [
        {P: PRIMARY + b"#", T: COMPANION},
        {P: PRIMARY, T: COMPANION + b"#"},
        {P: PRIMARY},
        {P: PRIMARY, T: {"type": "blob", "mode": "100755", "sha": blob_id(COMPANION)}},
    ],
)
def test_f5_candidate_commit_must_carry_the_head_content(files):
    world = build()
    world.tree(C1, BASE_FILES | files)
    fails(world, guard.R_MANIFEST_CANDIDATE)


def test_f5_candidate_base_must_be_an_ancestor_of_the_base():
    def other(payload):
        payload["attestation_scope"]["candidate"]["base_sha"] = B2

    world = build(mutate=other)
    for status, fork in (("diverged", C1), ("behind", B2), ("ahead", C1)):
        world.ancestry(B2, B, fork, status=status)
        fails(world, guard.R_MANIFEST_CANDIDATE)
    world.ancestry(B2, B, B2)
    expect(world, guard.PASS, guard.R_PASS)


@pytest.mark.parametrize(
    "raw",
    [
        b"{not json",
        b"[]",
        b"\xff\xfe",
        b'{"schema_version": "x", "schema_version": "' + b"y" + b'"}',
        b'{"schema_version": NaN}',
    ],
)
def test_case_26_unparseable_manifest(raw):
    fails(build(manifest=raw), guard.R_MANIFEST_UNPARSEABLE)


def test_case_26_duplicate_schema_key_cannot_relabel_a_v6_manifest():
    raw = manifest_bytes()
    hidden = b'{"schema_version": "axiom-encode/applied-rulespec/v5", ' + raw[1:]
    fails(build(manifest=hidden), guard.R_MANIFEST_UNPARSEABLE)


def scope_change(*keys_and_value):
    *keys, value = keys_and_value

    def change(payload):
        target = payload
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value

    return change


def rename(old, new):
    def change(payload):
        payload[new] = payload.pop(old)

    return change


def drop(*keys):
    def change(payload):
        target = payload
        for key in keys[:-1]:
            target = target[key]
        del target[keys[-1]]

    return change


SCOPE = "attestation_scope"


@pytest.mark.parametrize(
    "mutate",
    [
        scope_change("unexpected", 1),
        drop(SCOPE),
        drop("signature"),
        rename("candidate_rulespec_ref", "reviewed_rulespec_ref"),
        scope_change("backend", "codex"),
        scope_change("validation_waiver_bindings", {}),
        scope_change("tool", "axiom-encode retire"),
        scope_change("run_id", 777),
        scope_change("run_id", "0777"),
        scope_change("candidate_rulespec_ref", "main"),
        scope_change("signature", {}),
        scope_change("applied_files", []),
        scope_change("applied_files", [{"path": P, "sha256": "0" * 64}] * 2),
        scope_change("applied_files", 0, "unexpected", 1),
        scope_change("applied_files", 0, "path", 5),
        scope_change("applied_files", 0, "sha256", 5),
        scope_change("applied_files", 0, "sha256", "abc"),
        scope_change(SCOPE, "dispatcher", "login", 5),
        scope_change(SCOPE, "schema", "axiom-encode/attestation-scope/v2"),
        scope_change(SCOPE, "validation", "none"),
        scope_change(SCOPE, "owner_review", "approved"),
        scope_change(SCOPE, "origin", "model"),
        scope_change(SCOPE, "merge_requirement", "none"),
        scope_change(SCOPE, "unexpected", 1),
        drop(SCOPE, "merge_requirement"),
        scope_change(SCOPE, "candidate", "unexpected", 1),
        scope_change(SCOPE, "candidate", "pull_request", "42"),
        scope_change(SCOPE, "candidate", "pull_request", True),
        scope_change(SCOPE, "candidate", "head_sha", "HEAD"),
        scope_change(SCOPE, "candidate", "base_sha", "main"),
        scope_change(SCOPE, "dispatcher", "unexpected", 1),
        scope_change(SCOPE, "dispatcher", "id", "110687043"),
        scope_change(SCOPE, "dispatcher", "login", "not a login"),
    ],
)
def test_anything_but_the_exact_v6_shape_fails(mutate):
    fails(build(mutate=mutate), guard.R_MANIFEST_SHAPE)


# --- dispatcher and signing run (decision 3, F5) -----------------------------


def test_manifest_author_id_must_be_an_integer():
    # ``True == 1`` in Python; a boolean must not pass as the author id 1.
    world = build(
        author=(1, "first-account"),
        mutate=scope_change(SCOPE, "candidate", "author_id", True),
    )
    fails(world, guard.R_MANIFEST_SHAPE)


def test_dispatcher_must_be_enrolled():
    fails(build(dispatcher=OUTSIDER), guard.R_DISPATCHER_NOT_ENROLLED)


def test_enrolled_dispatcher_id_with_another_login_is_not_enrolled():
    # The manifest and the run agree on a login the enrolment table lacks.
    world = build(dispatcher=(PAVEL[0], "renamed-account"))
    fails(world, guard.R_DISPATCHER_NOT_ENROLLED)


@pytest.mark.parametrize(
    "trigger",
    [
        {"id": MAX[0], "login": MAX[1]},  # a re-run by another person
        {"id": PAVEL[0], "login": "renamed-account"},
        {"id": PAVEL[0], "login": None},
        {"id": PAVEL[0]},
    ],
)
def test_manifest_dispatcher_must_be_the_run_trigger(trigger):
    world = build()
    world.api[SIGNING_RUN]["triggering_actor"] = trigger
    fails(world, guard.R_DISPATCHER_MISMATCH)


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", 778),
        ("event", "pull_request"),
        ("path", ".github/workflows/other.yml"),
        ("head_branch", "feature"),
        ("head_sha", "main"),
        ("head_sha", None),
        ("status", "in_progress"),
        ("conclusion", "failure"),
        ("repository", {"full_name": "TheAxiomFoundation/rulespec-us"}),
        ("repository", None),
        ("actor", None),
        ("actor", {"id": "110687043"}),
        ("triggering_actor", {"login": "PavelMakarchuk"}),
        ("triggering_actor", {"id": "110687043", "login": "PavelMakarchuk"}),
    ],
)
def test_manifest_run_must_be_a_successful_central_main_dispatch(field, value):
    world = build()
    world.api[SIGNING_RUN][field] = value
    fails(world, guard.R_RUN)


def test_f5_string_actor_id_cannot_hide_the_original_dispatcher():
    # Without the strict integer check, "6076111" never equals the reviewer id
    # and the independence check is skipped.
    world = build(author=OUTSIDER, dispatcher=PAVEL, run_actor=MAX, approver=MAX)
    world.api[SIGNING_RUN]["actor"]["id"] = str(MAX[0])
    fails(world, guard.R_RUN)


@pytest.mark.parametrize("status,fork", [("diverged", B), ("behind", ENCODER_TIP)])
def test_f5_signing_run_commit_must_be_on_encoder_main(status, fork):
    # ``head_branch`` is only a name: a tag or a deleted branch can carry it.
    world = build()
    key = f"{guard.ENCODER_ROOT}/compare/{ENCODER_COMMIT}...{ENCODER_TIP}"
    world.api[key] = {"status": status, "merge_base_commit": {"sha": fork}}
    fails(world, guard.R_RUN)


def test_f5_signing_run_at_the_encoder_main_tip_needs_no_comparison():
    world = build()
    world.api[ENCODER_MAIN] = {"object": {"sha": ENCODER_COMMIT}}
    del world.api[f"{guard.ENCODER_ROOT}/compare/{ENCODER_COMMIT}...{ENCODER_TIP}"]
    expect(world, guard.PASS, guard.R_PASS)


# --- case 22 and F3: current-head CI ----------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "in_progress", "conclusion": None},
        {"status": "in_progress"},
        {"conclusion": "failure"},
        {"conclusion": "skipped"},
        {"app": {"id": 99999}},
        {"head_sha": C1},
        {"name": "validate"},
        {"details_url": f"https://github.com/{REPO}/runs/1"},
        {"details_url": f"https://github.com/evil/repo/actions/runs/{CI_RUN}/job/1"},
    ],
)
def test_case_22_required_check_is_not_green_on_the_head(changes):
    world = build()
    world.ci(H, **changes)
    fails(world, guard.R_CI)


def test_case_22_green_only_on_an_older_head():
    world = build()
    world.api[f"{ROOT}/commits/{H}{CHECKS}"] = {"total_count": 0, "check_runs": []}
    fails(world, guard.R_CI)


def test_case_22_one_of_two_same_named_check_runs_is_not_green():
    world = build()
    listing = world.api[f"{ROOT}/commits/{H}{CHECKS}"]
    listing["check_runs"].append(listing["check_runs"][0] | {"conclusion": "skipped"})
    listing["total_count"] = 2
    fails(world, guard.R_CI)


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", ".github/workflows/imitation.yml"),
        ("head_sha", C1),
        ("id", 889),
        ("event", "workflow_dispatch"),
        ("event", "pull_request_target"),
        ("repository", {"full_name": "fork/rulespec-us"}),
        ("head_repository", {"full_name": "fork/rulespec-us"}),
        ("head_repository", None),
        ("pull_requests", []),
        ("pull_requests", None),
        ("pull_requests", [{"number": 43, "head": {"sha": H}}]),
        ("pull_requests", [{"number": 42, "head": {"sha": C1}}]),
    ],
)
def test_f3_check_must_come_from_this_prs_pull_request_run(field, value):
    world = build()
    world.api[f"{ROOT}/actions/runs/{CI_RUN}"][field] = value
    fails(world, guard.R_CI)


def test_case_22_truncated_check_listing():
    world = build()
    world.api[f"{ROOT}/commits/{H}{CHECKS}"]["total_count"] = 2
    fails(world, guard.R_CI)


@pytest.mark.parametrize("lane", ["v5", "v6"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "queued"),
        ("status", "in_progress"),
        ("status", None),
        ("conclusion", "failure"),
        ("conclusion", "cancelled"),
        ("conclusion", "skipped"),
        ("conclusion", None),
    ],
)
def test_required_ci_workflow_must_finish_successfully_even_if_check_is_green(
    lane, field, value
):
    world = build() if lane == "v6" else plain(V5_UNIT)
    world.api[f"{ROOT}/actions/runs/{CI_RUN}"][field] = value
    fails(world, guard.R_CI)


# --- case 17: every read fails closed ---------------------------------------


def passing_reads():
    world = build()
    assert world.evaluate().verdict == guard.PASS
    return sorted(set(world.calls))


PASSING_READS = passing_reads()


def test_case_17_the_passing_evaluation_reads_only_the_fixed_organisation():
    assert len(PASSING_READS) == 17
    assert all(path.startswith("/repos/TheAxiomFoundation/") for path in PASSING_READS)


@pytest.mark.parametrize("index", range(len(PASSING_READS)))
@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError("https://api.github.com", 403, "rate limit", {}, None),
        urllib.error.HTTPError("https://api.github.com", 500, "error", {}, None),
        AdmissionError("GitHub admission requests must not redirect"),
        TimeoutError(),
    ],
    ids=["403", "500", "redirect", "timeout"],
)
def test_case_17_any_failed_read_fails_closed(index, error):
    world = build()

    def refuse():
        raise error

    world.api[PASSING_READS[index]] = refuse
    fails(world, guard.R_READ_FAILED)


@pytest.mark.parametrize("index", range(len(PASSING_READS)))
def test_case_17_any_missing_or_malformed_read_fails_closed(index):
    for value in (None, "", [], {}):
        world = build()
        world.api[PASSING_READS[index]] = value
        assert world.evaluate().verdict == guard.FAIL, (PASSING_READS[index], value)


def test_case_17_no_pass_is_cached_across_evaluations():
    world = build()
    assert world.evaluate().verdict == guard.PASS

    def refuse():
        raise urllib.error.HTTPError("https://api.github.com", 500, "error", {}, None)

    world.api[PR + "/reviews" + PAGE] = refuse
    fails(world, guard.R_READ_FAILED)


@pytest.mark.parametrize("sha", [B, H])
def test_case_17_truncated_tree(sha):
    world = build()
    world.api[f"{ROOT}/git/trees/{sha}?recursive=1"]["truncated"] = True
    fails(world, guard.R_DELTA_INCOMPLETE)


def test_case_17_truncated_tree_fails_a_pr_with_no_guarded_content():
    world = plain({"README.md": b"changed\n"})
    world.api[f"{ROOT}/git/trees/{H}?recursive=1"]["truncated"] = True
    fails(world, guard.R_DELTA_INCOMPLETE)


def test_case_17_truncated_tree_of_the_approved_commit():
    world = build()
    advance(world, new_base=True)
    world.api[f"{ROOT}/git/trees/{H}?recursive=1"]["truncated"] = True
    fails(world, guard.R_DELTA_INCOMPLETE)


def test_case_17_truncated_tree_of_the_candidate_commit():
    world = build()
    world.api[f"{ROOT}/git/trees/{C1}?recursive=1"]["truncated"] = True
    fails(world, guard.R_MANIFEST_CANDIDATE)


def test_case_17_eleventh_page_of_reviews():
    # Stated limit: 1000 reviews, also comment-only ones from outside
    # accounts, exhaust the page bound and the verdict stays ``fail``.
    world = build()
    comments = [
        review(1000 + number, OUTSIDER, H, state="COMMENTED") for number in range(100)
    ]
    for page in range(1, 11):
        world.api[f"{PR}/reviews?per_page=100&page={page}"] = comments
    fails(world, guard.R_REVIEWS)


@pytest.mark.parametrize(
    "change",
    [
        {"content": base64.b64encode(b"other bytes").decode(), "size": 11},
        {"encoding": "utf-8"},  # same text, another declared encoding
        {"encoding": "none", "content": ""},
        {"content": "!not base64!"},
        {"content": None},
        {"size": 1},
        {"sha": "0" * 40},
    ],
)
def test_case_17_blob_bytes_must_hash_to_the_tree_object_id(change):
    world = build()
    world.api[f"{ROOT}/git/blobs/{blob_id(PRIMARY)}"].update(change)
    fails(world, guard.R_BLOB_UNREADABLE)


def test_case_17_blob_larger_than_the_bound(monkeypatch):
    monkeypatch.setattr(guard, "MAX_BLOB_BYTES", len(PRIMARY) - 1)
    fails(build(), guard.R_BLOB_UNREADABLE)


def test_unforeseen_error_fails_closed(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("unforeseen")

    monkeypatch.setattr(guard, "_parse_manifest", broken)
    fails(build(), guard.R_EVALUATION_ERROR)


@pytest.mark.parametrize(
    "name,number",
    [
        ("us", 42),
        ("rulespec-US", 42),
        ("rulespec-usa", 42),
        ("rulespec-validators", 42),
        ("rulespec-us/../axiom-encode", 42),
        ("TheAxiomFoundation/rulespec-us", 42),
        ("rulespec-us-", 42),
        (None, 42),
        ("rulespec-us", 0),
        ("rulespec-us", True),
        ("rulespec-us", "42"),
    ],
)
def test_invalid_request_fails_without_a_read(name, number):
    world = build()
    result = guard.evaluate(name, number, fetch=world.fetch)
    assert (result.verdict, result.reason) == (guard.FAIL, guard.R_INVALID_REQUEST)
    assert world.calls == []


@pytest.mark.parametrize(
    "name", ["rulespec-us-tn", "rulespec-uk-kingston-upon-thames", "rulespec-tz-znz"]
)
def test_subdivision_repository_names_are_accepted(name):
    calls = []

    def fetch(path):
        calls.append(path)
        raise KeyError(path)

    result = guard.evaluate(name, 42, fetch=fetch)
    assert (result.verdict, result.reason) == (guard.FAIL, guard.R_READ_FAILED)
    assert calls == [f"/repos/TheAxiomFoundation/{name}/pulls/42"]


# --- case 18: state moves during the evaluation -----------------------------


def test_case_18_review_dismissed_between_the_two_reads():
    world = build()
    dismissed = [review(11, MAX, H), review(12, MAX, H, state="DISMISSED")]
    second_read(world, PR + "/reviews" + PAGE, dismissed)
    fails(world, guard.R_CHANGED)


def test_case_18_head_moves_between_the_two_reads():
    world = build()
    moved = copy.deepcopy(world.api[PR])
    moved["head"]["sha"] = H2
    second_read(world, PR, moved)
    fails(world, guard.R_CHANGED)


def test_case_18_main_moves_between_the_two_reads():
    world = build()
    second_read(world, ROOT + "/git/ref/heads/main", {"object": {"sha": B2}})
    fails(world, guard.R_CHANGED)


def test_case_18_converted_to_draft_between_the_two_reads():
    world = build()
    second_read(world, PR, world.api[PR] | {"draft": True})
    fails(world, guard.R_CHANGED)


def mutate_during_pr_recheck(world, mutation):
    original = world.api[PR]

    def recheck():
        if world.calls.count(PR) == 2:
            mutation()
        return original

    world.api[PR] = recheck


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "queued"),
        ("status", "in_progress"),
        ("conclusion", "failure"),
        ("triggering_actor", {"id": MAX[0], "login": MAX[1]}),
        ("actor", {"id": MAX[0], "login": MAX[1]}),
        ("head_sha", ENCODER_TIP),
        ("run_attempt", 2),
    ],
)
def test_case_18_signing_run_moves_during_final_pr_recheck(field, value):
    world = build()
    world.api[SIGNING_RUN]["run_attempt"] = 1
    mutate_during_pr_recheck(
        world, lambda: world.api[SIGNING_RUN].update({field: value})
    )
    fails(world, guard.R_CHANGED)
    assert world.calls.count(SIGNING_RUN) == 2


@pytest.mark.parametrize("lane", ["v5", "v6"])
@pytest.mark.parametrize(
    "target,field,value",
    [
        ("check", "id", 2),
        ("check", "status", "queued"),
        ("check", "conclusion", "failure"),
        (
            "check",
            "details_url",
            f"https://github.com/{REPO}/actions/runs/{CI_RUN}/job/2",
        ),
        ("workflow", "run_attempt", 2),
        ("workflow", "event", "workflow_dispatch"),
        ("workflow", "status", "queued"),
        ("workflow", "status", "in_progress"),
        ("workflow", "conclusion", "failure"),
        ("workflow", "conclusion", "cancelled"),
    ],
)
def test_case_18_required_ci_moves_during_final_pr_recheck(lane, target, field, value):
    world = build() if lane == "v6" else plain(V5_UNIT)
    checks = world.api[f"{ROOT}/commits/{H}{CHECKS}"]["check_runs"]
    checks[0]["id"] = 1
    workflow = world.api[f"{ROOT}/actions/runs/{CI_RUN}"]
    workflow["run_attempt"] = 1
    record = checks[0] if target == "check" else workflow
    mutate_during_pr_recheck(world, lambda: record.update({field: value}))
    fails(world, guard.R_CHANGED)


def test_stable_native_run_and_check_identities_keep_the_approval():
    world = build()
    world.api[SIGNING_RUN]["run_attempt"] = 1
    world.api[f"{ROOT}/actions/runs/{CI_RUN}"]["run_attempt"] = 1
    world.api[f"{ROOT}/commits/{H}{CHECKS}"]["check_runs"][0]["id"] = 1
    expect(world, guard.PASS, guard.R_PASS)


@pytest.mark.parametrize("value", [None, True, 0, "1"])
def test_present_signing_run_attempt_requires_a_numeric_identity(value):
    world = build()
    world.api[SIGNING_RUN]["run_attempt"] = value
    fails(world, guard.R_RUN)


@pytest.mark.parametrize("value", [None, True, 0, "1"])
@pytest.mark.parametrize("target", ["check", "workflow"])
def test_present_ci_identity_requires_a_numeric_identity(target, value):
    world = build()
    if target == "check":
        world.api[f"{ROOT}/commits/{H}{CHECKS}"]["check_runs"][0]["id"] = value
    else:
        world.api[f"{ROOT}/actions/runs/{CI_RUN}"]["run_attempt"] = value
    fails(world, guard.R_CI)


# --- PR identity (step 1) ---------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        {"state": "closed"},
        {"merged": True},
        {"number": 43},
        {"user": {"login": "no-id"}},
        {"user": {"id": 0, "login": "zero"}},
        {"head": {"sha": H, "ref": "codex/medical", "repo": {"full_name": "fork/x"}}},
        {"head": {"sha": H, "ref": "", "repo": {"full_name": REPO}}},
        {"head": {"sha": H, "ref": None, "repo": {"full_name": REPO}}},
        {"head": {"sha": H, "ref": 5, "repo": {"full_name": REPO}}},
        {"base": {"sha": B, "ref": "release", "repo": {"full_name": REPO}}},
        {"head": {"sha": "main", "ref": "codex/medical", "repo": {"full_name": REPO}}},
    ],
)
def test_pr_identity(change):
    world = build()
    world.api[PR].update(change)
    fails(world, guard.R_PR_IDENTITY)


@pytest.mark.parametrize("status,fork", [("diverged", B), ("behind", B), ("ahead", C1)])
def test_head_must_descend_from_the_live_main_tip(status, fork):
    world = build()
    world.api[f"{ROOT}/compare/{B}...{H}"].update(
        status=status, merge_base_commit={"sha": fork}
    )
    fails(world, guard.R_BASE_NOT_CURRENT)


def test_main_ref_must_be_an_exact_commit():
    world = build()
    world.api[ROOT + "/git/ref/heads/main"] = {"object": {"sha": "main"}}
    fails(world, guard.R_BASE_NOT_CURRENT)


@pytest.mark.parametrize("draft", [True, None])
def test_draft_promotion_has_no_verdict(draft):
    world = build()
    world.api[PR]["draft"] = draft
    fails(world, guard.R_PR_DRAFT)


# --- cases 16 and 25: no manifest ------------------------------------------------


def test_case_16_workflow_only_pr_has_no_lane():
    # Decided by round 2: a guarded change that no lane claims fails closed.
    world = plain({WORKFLOW: b"jobs: {}\n", ".github/CODEOWNERS": b"\n"})
    fails(world, guard.R_NO_LANE)


def test_changed_manifests_beyond_the_read_bound_fail(monkeypatch):
    monkeypatch.setattr(guard, "MAX_MANIFEST_BLOBS", 1)
    world = plain(
        {
            guard.MANIFEST_PREFIX + "us/a.json": v5_manifest(),
            guard.MANIFEST_PREFIX + "us/b.json": v5_manifest() + b"\n",
        }
    )
    fails(world, guard.R_TOO_MANY_MANIFESTS)


# --- cases that this evaluator cannot decide --------------------------------


@pytest.mark.skip(
    reason="Case 15 needs the publisher binding: branch protection must require "
    "the guard App's app_id. No App, workflow or check-run writer exists yet."
)
def test_case_15_imitated_check_name_from_a_pr_workflow():
    raise AssertionError("not decidable by the single-PR evaluator")


@pytest.mark.skip(
    reason="Case 24 needs merge-group evaluation, which is a stub in this change."
)
def test_case_24_merge_group_with_one_unapproved_unit():
    raise AssertionError("merge-group evaluation is not implemented")


def test_merge_group_stub_fails_closed():
    result = guard.evaluate_merge_group(NAME, "gh-readonly-queue/main/pr-42-" + B)
    assert (result.verdict, result.reason) == (guard.FAIL, guard.R_MERGE_GROUP)


# --- contract with the live encoder -----------------------------------------


def test_constants_follow_the_live_encoder_contract():
    from axiom_encode import cli
    from axiom_encode.prepare_signed_backfill import citation_rulespec_path

    assert str(citation_rulespec_path(CITATION)) == P
    assert cli._applied_encoding_manifest_path(Path(P)).as_posix() == M
    assert cli.APPLIED_ENCODING_MANIFEST_DIR.as_posix() == guard.MANIFEST_DIR
    assert guard.PROMOTION_TOOL == cli.APPLIED_ENCODING_REVIEWED_CANDIDATE_TOOL
    # No reader or writer knows v6 yet: this module must not make it live.
    assert cli.APPLIED_ENCODING_MANIFEST_SCHEMA == V5
    assert guard.EARLIER_MANIFEST_SCHEMAS == {cli.APPLIED_ENCODING_MANIFEST_SCHEMA}
    assert guard.MANIFEST_SCHEMA_V6 == "axiom-encode/applied-rulespec/v6"
    renamed = {"candidate_rulespec_ref", "attestation_scope"}
    assert set(guard.V6_MANIFEST_FIELD_VARIANTS) == {
        (cli._REVIEWED_CANDIDATE_MANIFEST_FIELDS - {"reviewed_rulespec_ref"}) | renamed,
        (
            cli._REVIEWED_CANDIDATE_PER_FILE_WAIVER_MANIFEST_FIELDS
            - {"reviewed_rulespec_ref"}
        )
        | renamed,
    }


def test_nothing_in_the_encoder_calls_the_guard():
    root = Path(__file__).resolve().parents[1]
    callers = [
        path.relative_to(root).as_posix()
        for folder in ("src", "scripts", ".github")
        for path in (root / folder).rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".yml", ".yaml"}
        and path.name != "content_merge_guard.py"
        and (
            "content_merge_guard" in path.read_text(errors="ignore")
            or "merge-guard" in path.read_text(errors="ignore")
        )
    ]
    assert callers == ["src/axiom_encode/cli.py"]


# --- CLI --------------------------------------------------------------------


def run_command(monkeypatch, capsys, world):
    monkeypatch.setattr(guard, "github_json", world.fetch)
    code = guard.command(argparse.Namespace(repository=NAME, pr=42))
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and out.endswith("\n")  # one JSON line
    return code, json.loads(out)


def test_command_exit_codes_and_one_json_line(monkeypatch, capsys):
    code, printed = run_command(monkeypatch, capsys, build())
    assert (code, printed["verdict"], printed["reason"]) == (
        0,
        "pass",
        "content_bound_approval",
    )
    assert printed["schema"] == guard.VERDICT_SCHEMA
    assert printed["evidence"]["head"] == H

    unapproved = build()
    unapproved.reviews([])
    code, printed = run_command(monkeypatch, capsys, unapproved)
    assert (code, printed["verdict"], printed["reason"]) == (1, "fail", "no_approval")

    code, printed = run_command(monkeypatch, capsys, plain({"README.md": b"x\n"}))
    assert (code, printed["verdict"], printed["reason"]) == (
        3,
        "not_applicable",
        "no_guarded_content",
    )


def test_cli_registers_the_command(monkeypatch, capsys):
    from axiom_encode import cli

    monkeypatch.setattr(guard, "github_json", plain({"README.md": b"x\n"}).fetch)
    monkeypatch.setattr(
        "sys.argv",
        ["axiom-encode", "merge-guard", "--repository", NAME, "--pr", "42"],
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 3
    assert json.loads(capsys.readouterr().out)["verdict"] == "not_applicable"
