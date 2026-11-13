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
from axiom_encode.reviewed_pr_admission import INDEX_PATH, AdmissionError

REPO = "TheAxiomFoundation/rulespec-us"
ROOT = "/repos/" + REPO
PR = ROOT + "/pulls/42"
CITATION = "us-az/manual/des/faa5/medical/block-7"
P = "us-az/policies/des/faa5/medical/block-7.yaml"
T = "us-az/policies/des/faa5/medical/block-7.test.yaml"
M = ".axiom/encoding-manifests/us-az/policies/des/faa5/medical/block-7.json"
PRIMARY = b"format: rulespec/v1\nrules: []\n"
COMPANION = b"- name: fixture\n"
MAX = (6076111, "MaxGhenis")
PAVEL = (110687043, "PavelMakarchuk")
OUTSIDER = (555, "outside-author")
BOT = (900001, "axiom-signing[bot]")
RUN_ID = "777"
CI_RUN = "888"
SIGNING_RUN = f"{guard.ENCODER_ROOT}/actions/runs/{RUN_ID}"
CHECKS = "/check-runs?check_name=validate%20%2F%20validate&filter=latest&per_page=100"
PAGE = "?per_page=100&page=1"
BASE_FILES = {"README.md": b"readme\n", INDEX_PATH: b"{}\n"}


def oid(label):
    return hashlib.sha1(label.encode()).hexdigest()


B, C1, H = oid("main"), oid("candidate"), oid("signed")
B2, C2, H2 = oid("main-2"), oid("candidate-2"), oid("signed-2")


def blob_id(raw):
    return hashlib.sha1(b"blob %d\x00" % len(raw) + raw).hexdigest()


def manifest_bytes(primary=PRIMARY, companion=COMPANION, mutate=None, **scope):
    payload = {
        "schema_version": guard.MANIFEST_SCHEMA_V6,
        "generated_at": "2026-11-12T00:00:00+00:00",
        "tool": guard.PROMOTION_TOOL,
        "axiom_encode_version": "0.0.0",
        "axiom_encode_git": {"commit": "e" * 40},
        "run_id": RUN_ID,
        "citation": CITATION,
        "candidate_rulespec_ref": C1,
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
                "head_sha": C1,
                "base_sha": B,
                "author_id": scope.get("author", PAVEL)[0],
            },
            "dispatcher": {
                "id": scope.get("dispatcher", PAVEL)[0],
                "login": scope.get("dispatcher", PAVEL)[1],
            },
            "merge_requirement": guard.MERGE_REQUIREMENT,
        },
    }
    if mutate:
        mutate(payload)
    return json.dumps(payload, sort_keys=True).encode()


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

    def ancestry(self, base, sha, fork, files=None):
        self.api[f"{ROOT}/compare/{base}...{sha}"] = {
            "status": "ahead",
            "merge_base_commit": {"sha": fork},
            "files": files or [],
        }

    def publish(self, head, base, commits, author=PAVEL, history=None):
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
        by_sha = {item["sha"]: item for item in commits}
        for path in (P, T):
            quoted = path.replace("/", "%2F")
            shas = (history or {}).get(path, [C1])
            self.api[f"{ROOT}/commits?sha={head}&path={quoted}&per_page=100&page=1"] = [
                by_sha[sha] for sha in shas if sha in by_sha
            ]
        self.ci(head)

    def reviews(self, items):
        self.api[PR + "/reviews" + PAGE] = items

    def ci(self, head, **changes):
        self.api[f"{ROOT}/commits/{head}{CHECKS}"] = {
            "total_count": 1,
            "check_runs": [
                {
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
        self.api[f"{ROOT}/actions/runs/{CI_RUN}"] = {
            "id": int(CI_RUN),
            "path": ".github/workflows/repository-checks.yml",
            "head_sha": head,
        }

    def evaluate(self):
        return guard.evaluate("us", 42, fetch=self.fetch)


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
    world.publish(H, B, [commit(C1, candidate_author), commit(H, BOT)], author=author)
    world.reviews([review(11, approver, H)])
    run_actor = run_actor or dispatcher
    world.api[SIGNING_RUN] = {
        "id": int(RUN_ID),
        "event": "workflow_dispatch",
        "path": ".github/workflows/targeted-signed-reencode.yml",
        "head_branch": "main",
        "head_sha": "e" * 40,
        "status": "completed",
        "conclusion": "success",
        "actor": {"id": run_actor[0], "login": run_actor[1]},
        "triggering_actor": {"id": dispatcher[0], "login": dispatcher[1]},
    }
    return world


def advance(world, *, primary=PRIMARY, manifest=None, extra=None, new_base=False):
    """Move the PR to a second signed head, optionally over a newer ``main``."""
    base = B
    base_files = dict(BASE_FILES)
    commits = [commit(C1), commit(H, BOT)]
    history = {}
    if new_base:
        base = B2
        base_files |= {"other/module.yaml": b"merged elsewhere\n"}
        world.tree(B2, base_files)
    if primary != PRIMARY:
        commits.append(commit(C2))
        history[P] = [C1, C2]
    commits.append(commit(H2, BOT))
    manifest = manifest or manifest_bytes(primary=primary)
    world.tree(H2, base_files | {P: primary, T: COMPANION, M: manifest} | (extra or {}))
    world.publish(H2, base, commits, history=history)
    # The approved commit ``H`` forked from the first ``main`` tip.
    world.ancestry(base, H, B)


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
            "approvals": [{"review_id": 11, "reviewer_id": MAX[0], "commit_id": H}],
        },
    }


def test_per_file_waiver_variant_and_index_change_pass():
    def swap(payload):
        del payload["validation_waiver_set_sha256"]
        payload["validation_waiver_bindings"] = {"fixture": True}

    world = build(mutate=swap, head_extra={INDEX_PATH: b'{"changed": true}\n'})
    expect(world, guard.PASS, guard.R_PASS)


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


# --- cases 6-9, 27: who may approve -----------------------------------------


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
    ],
)
def test_case_09_enrolment_is_id_login_and_account_type(user, extra):
    world = build()
    world.reviews([review(11, user, H, **extra)])
    fails(world, guard.R_NOT_ENROLLED)


def test_case_27_enrolled_owner_authored_the_candidate_commits():
    world = build(author=OUTSIDER, candidate_author=MAX, approver=MAX)
    fails(world, guard.R_AUTHORED_COMMITS)


def test_case_27_unattributed_commit_author_is_a_documented_gap():
    # Git author metadata is free text. When GitHub attributes the commit to
    # no account, the evaluator cannot see who wrote it. This test records the
    # gap; it is not a defence.
    world = build(author=OUTSIDER, approver=MAX)
    world.api[PR + "/commits" + PAGE][0].update(author=None, committer=None)
    for path in (P, T):
        quoted = path.replace("/", "%2F")
        key = f"{ROOT}/commits?sha={H}&path={quoted}&per_page=100&page=1"
        world.api[key][0].update(author=None, committer=None)
    expect(world, guard.PASS, guard.R_PASS)


def test_path_history_outside_the_pr_commits_fails():
    world = build()
    key = f"{ROOT}/commits?sha={H}&path={P.replace('/', '%2F')}&per_page=100&page=1"
    world.api[key] = [commit(oid("not-in-pr"))]
    fails(world, guard.R_COMMITS)


def test_incomplete_commit_list_fails():
    world = build()
    world.api[PR]["commits"] = 251
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
def test_case_13_primary_is_not_a_regular_blob(entry):
    world = build(head_extra={P: entry | {"sha": blob_id(PRIMARY)}})
    fails(world, guard.R_DELTA_DISAGREES)


def test_case_13_manifest_is_a_symlink():
    link = {"type": "blob", "mode": "120000", "sha": blob_id(manifest_bytes())}
    fails(build(head_extra={M: link}), guard.R_MANIFEST_NOT_REGULAR)


def test_case_14_primary_at_a_non_canonical_path():
    def elsewhere(payload):
        payload["citation"] = "us-az/manual/des/faa5/medical/block-8"

    fails(build(mutate=elsewhere), guard.R_MANIFEST_PATH)


def test_case_14_citation_of_another_country():
    def foreign(payload):
        payload["citation"] = "uk/manual/des/faa5/medical/block-7"

    fails(build(mutate=foreign), guard.R_CITATION)


def test_case_16_pr_rewrites_the_required_workflow():
    workflow = {".github/workflows/repository-checks.yml": b"jobs: {}\n"}
    fails(build(head_extra=workflow), guard.R_SCOPE_PATHS)


def test_case_20_only_a_manifest():
    world = build()
    world.tree(H, BASE_FILES | {M: manifest_bytes()})
    world.publish(H, B, [commit(C1), commit(H, BOT)])
    fails(world, guard.R_SCOPE_PATHS)


def test_case_20_two_v6_manifests():
    second = M.replace("block-7", "block-8")
    fails(build(head_extra={second: manifest_bytes()}), guard.R_MULTIPLE_V6)


def test_case_21_applied_files_digest_differs():
    def wrong(payload):
        payload["applied_files"][0]["sha256"] = "1" * 64

    fails(build(mutate=wrong), guard.R_APPLIED_FILES)


def test_case_23_module_already_exists_on_main():
    fails(build(base_extra={P: b"older module\n"}), guard.R_SCOPE_NOT_NEW)


def test_index_removal_fails_scope():
    world = build()
    files = dict(BASE_FILES) | {P: PRIMARY, T: COMPANION, M: manifest_bytes()}
    del files[INDEX_PATH]
    world.tree(H, files)
    world.publish(H, B, [commit(C1), commit(H, BOT)])
    fails(world, guard.R_SCOPE_PATHS)


@pytest.mark.parametrize("evidence", ["files", "compare"])
def test_mutable_file_lists_cannot_hide_a_change(evidence):
    world = build()
    hidden = [item for item in world.api[PR + "/files" + PAGE] if item["filename"] != T]
    if evidence == "files":
        world.api[PR + "/files" + PAGE] = hidden
        world.api[PR]["changed_files"] = len(hidden)
    else:
        world.api[f"{ROOT}/compare/{B}...{H}"]["files"] = hidden
    fails(world, guard.R_DELTA_DISAGREES)


# --- cases 19, 26 and the exact v6 shape ------------------------------------


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


def test_case_26_duplicate_schema_key_cannot_hide_a_v6_manifest():
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
        scope_change(SCOPE, "dispatcher", "unexpected", 1),
        scope_change(SCOPE, "dispatcher", "id", "110687043"),
        scope_change(SCOPE, "dispatcher", "login", "not a login"),
    ],
)
def test_anything_but_the_exact_v6_shape_fails(mutate):
    fails(build(mutate=mutate), guard.R_MANIFEST_SHAPE)


# --- dispatcher and signing run (decision 3) --------------------------------


def test_dispatcher_must_be_enrolled():
    fails(build(dispatcher=OUTSIDER), guard.R_DISPATCHER_NOT_ENROLLED)


def test_manifest_dispatcher_must_be_the_run_trigger():
    world = build()
    world.api[SIGNING_RUN]["triggering_actor"] = {"id": MAX[0], "login": MAX[1]}
    fails(world, guard.R_DISPATCHER_MISMATCH)


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", 778),
        ("event", "pull_request"),
        ("path", ".github/workflows/other.yml"),
        ("head_branch", "feature"),
        ("status", "in_progress"),
        ("conclusion", "failure"),
        ("actor", None),
        ("triggering_actor", {"login": "PavelMakarchuk"}),
    ],
)
def test_manifest_run_must_be_a_successful_central_main_dispatch(field, value):
    world = build()
    world.api[SIGNING_RUN][field] = value
    fails(world, guard.R_RUN)


# --- case 22: current-head CI -----------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "in_progress", "conclusion": None},
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


@pytest.mark.parametrize(
    "field,value",
    [
        ("path", ".github/workflows/imitation.yml"),
        ("head_sha", C1),
        ("id", 889),
    ],
)
def test_case_22_check_must_come_from_the_protected_workflow_run(field, value):
    world = build()
    world.api[f"{ROOT}/actions/runs/{CI_RUN}"][field] = value
    fails(world, guard.R_CI)


def test_case_22_truncated_check_listing():
    world = build()
    world.api[f"{ROOT}/commits/{H}{CHECKS}"]["total_count"] = 2
    fails(world, guard.R_CI)


# --- case 17: every read fails closed ---------------------------------------


def passing_reads():
    world = build()
    assert world.evaluate().verdict == guard.PASS
    return sorted(set(world.calls))


def test_case_17_the_passing_evaluation_reads_the_expected_endpoints():
    reads = passing_reads()
    assert len(reads) == 16
    assert all(path.startswith("/repos/TheAxiomFoundation/") for path in reads)


@pytest.mark.parametrize("index", range(16))
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
    path = passing_reads()[index]

    def refuse():
        raise error

    world.api[path] = refuse
    fails(world, guard.R_READ_FAILED)


@pytest.mark.parametrize("index", range(16))
def test_case_17_any_missing_or_malformed_read_fails_closed(index):
    path = passing_reads()[index]
    for value in (None, "", [], {}):
        world = build()
        world.api[path] = value
        assert world.evaluate().verdict == guard.FAIL, (path, value)


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


def test_case_17_truncated_tree_of_the_approved_commit():
    world = build()
    advance(world, new_base=True)
    world.api[f"{ROOT}/git/trees/{H}?recursive=1"]["truncated"] = True
    fails(world, guard.R_DELTA_INCOMPLETE)


def test_case_17_eleventh_page_of_reviews():
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
        {"encoding": "none", "content": ""},
        {"size": 1},
        {"sha": "0" * 40},
    ],
)
def test_case_17_blob_bytes_must_hash_to_the_tree_object_id(change):
    world = build()
    world.api[f"{ROOT}/git/blobs/{blob_id(PRIMARY)}"].update(change)
    fails(world, guard.R_BLOB_UNREADABLE)


def test_unforeseen_error_fails_closed(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("unforeseen")

    monkeypatch.setattr(guard, "_parse_manifest", broken)
    fails(build(), guard.R_EVALUATION_ERROR)


@pytest.mark.parametrize("country,number", [("US", 42), ("usa", 42), ("us", 0)])
def test_invalid_request_fails(country, number):
    result = guard.evaluate(country, number, fetch=build().fetch)
    assert (result.verdict, result.reason) == (guard.FAIL, guard.R_INVALID_REQUEST)


def test_boolean_pull_request_number_fails():
    result = guard.evaluate("us", True, fetch=build().fetch)
    assert (result.verdict, result.reason) == (guard.FAIL, guard.R_INVALID_REQUEST)


# --- case 18: state moves during the evaluation -----------------------------


def second_read(world, path, later):
    first = world.api[path]
    seen = []

    def read():
        seen.append(1)
        return first if len(seen) == 1 else later

    world.api[path] = read


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


# --- PR identity (step 1) ---------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        {"state": "closed"},
        {"merged": True},
        {"number": 43},
        {"user": {"login": "no-id"}},
        {"head": {"sha": H, "ref": "codex/medical", "repo": {"full_name": "fork/x"}}},
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


@pytest.mark.parametrize("draft", [True, None])
def test_draft_promotion_has_no_verdict(draft):
    world = build()
    world.api[PR]["draft"] = draft
    fails(world, guard.R_PR_DRAFT)


# --- case 25: not applicable ------------------------------------------------


def maintenance(files):
    world = World()
    world.tree(B, BASE_FILES)
    world.tree(H, BASE_FILES | files)
    world.publish(H, B, [commit(H)])
    world.reviews([])
    return world


def test_case_25_maintenance_pr_is_not_applicable():
    script = {"type": "blob", "mode": "100755", "sha": "d" * 40}
    world = maintenance({"README.md": b"changed\n", "scripts/run.sh": script})
    world.api[PR]["draft"] = True
    result = expect(world, guard.NOT_APPLICABLE, guard.R_NO_V6_MANIFEST)
    assert result.evidence == {
        "repository": REPO,
        "pull_request": 42,
        "head": H,
        "base": B,
    }
    # Applicability is decided from the two trees and the changed manifests.
    assert not any("/reviews" in path or "/files" in path for path in world.calls)


def test_case_25_v5_manifest_is_not_applicable():
    def legacy(payload):
        payload["schema_version"] = "axiom-encode/applied-rulespec/v5"

    world = maintenance({P: PRIMARY, T: COMPANION, M: manifest_bytes(mutate=legacy)})
    expect(world, guard.NOT_APPLICABLE, guard.R_NO_V6_MANIFEST)


def test_case_25_removed_manifest_is_not_applicable():
    world = World()
    world.tree(B, BASE_FILES | {M: manifest_bytes()})
    world.tree(H, BASE_FILES)
    world.publish(H, B, [commit(H)])
    expect(world, guard.NOT_APPLICABLE, guard.R_NO_V6_MANIFEST)


def test_changed_manifests_beyond_the_read_bound_fail(monkeypatch):
    monkeypatch.setattr(guard, "MAX_MANIFEST_BLOBS", 1)
    legacy = b'{"schema_version": "axiom-encode/applied-rulespec/v5"}'
    world = maintenance(
        {
            guard.MANIFEST_PREFIX + "us/a.json": legacy,
            guard.MANIFEST_PREFIX + "us/b.json": legacy + b"\n",
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
    reason="Case 16, second half: a maintenance PR that rewrites the required "
    "workflow is not_applicable here by design. That gap is in the binding of "
    "validate / validate, outside this evaluator."
)
def test_case_16_maintenance_pr_rewrites_the_required_workflow():
    raise AssertionError("not decidable by the single-PR evaluator")


@pytest.mark.skip(
    reason="Case 24 needs merge-group evaluation, which is a stub in this change."
)
def test_case_24_merge_group_with_one_unapproved_unit():
    raise AssertionError("merge-group evaluation is not implemented")


def test_merge_group_stub_fails_closed():
    result = guard.evaluate_merge_group("us", "gh-readonly-queue/main/pr-42-" + B)
    assert (result.verdict, result.reason) == (guard.FAIL, guard.R_MERGE_GROUP)


# --- contract with the live encoder -----------------------------------------


def test_constants_follow_the_live_encoder_contract():
    from pathlib import Path

    from axiom_encode import cli
    from axiom_encode.prepare_signed_backfill import citation_rulespec_path

    assert str(citation_rulespec_path(CITATION)) == P
    assert cli._applied_encoding_manifest_path(Path(P)).as_posix() == M
    assert guard.PROMOTION_TOOL == cli.APPLIED_ENCODING_REVIEWED_CANDIDATE_TOOL
    # No reader or writer knows v6 yet: this module must not make it live.
    assert cli.APPLIED_ENCODING_MANIFEST_SCHEMA == "axiom-encode/applied-rulespec/v5"
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


@pytest.mark.parametrize("approved,code", [(True, 0), (False, 1)])
def test_command_prints_one_verdict(monkeypatch, capsys, approved, code):
    world = build()
    if not approved:
        world.reviews([])
    monkeypatch.setattr(guard, "github_json", world.fetch)
    assert guard.command(argparse.Namespace(country="us", pr=42)) == code
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema"] == guard.VERDICT_SCHEMA
    assert printed["verdict"] == ("pass" if approved else "fail")
    assert printed["evidence"]["head"] == H


def test_cli_registers_the_command(monkeypatch, capsys):
    from axiom_encode import cli

    monkeypatch.setattr(guard, "github_json", maintenance({"README.md": b"x\n"}).fetch)
    monkeypatch.setattr(
        "sys.argv", ["axiom-encode", "merge-guard", "--country", "us", "--pr", "42"]
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "not_applicable"
