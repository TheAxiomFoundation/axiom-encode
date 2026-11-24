"""Fail-closed review evidence for reusable protected candidate promotion."""

import base64
import copy
import hashlib
import inspect
from dataclasses import replace

import pytest

from axiom_encode import reviewed_pr_admission as admission_module
from axiom_encode.reviewed_pr_admission import (
    AdmissionError,
    PromotionSelector,
    revalidate_admission,
    verify_reviewed_pr,
)

HEAD = "a" * 40
BASE = "b" * 40
PATH = "us-az/policies/des/faa5/medical/block-7.yaml"
SELECTOR = PromotionSelector(
    "us",
    42,
    HEAD,
    "us-az/manual/des/faa5/medical/block-7",
    PATH,
    "codex/medical",
    "PavelMakarchuk",
    "12345",
)
ROOT = "/repos/TheAxiomFoundation/rulespec-us"
PR = ROOT + "/pulls/42"
RUN = "/repos/TheAxiomFoundation/axiom-encode/actions/runs/12345"
REVIEWS = PR + "/reviews?per_page=100&page=1"
PAVEL = {"id": 110687043, "login": "PavelMakarchuk"}


@pytest.fixture(
    autouse=True, params=[True, False], ids=["approval-required", "owner-dispatch"]
)
def approval_required(request, monkeypatch):
    """Every test runs under both policies unless it pins one."""
    monkeypatch.setattr(admission_module, "OWNER_APPROVAL_REQUIRED", request.param)
    return request.param


# The rule in force before #2301: these tests are unchanged and pin the policy.
approval_policy = pytest.mark.parametrize(
    "approval_required", [True], indirect=True, ids=["approval-required"]
)
dispatch_policy = pytest.mark.parametrize(
    "approval_required", [False], indirect=True, ids=["owner-dispatch"]
)


@pytest.fixture
def api():
    files = [PATH, SELECTOR.companion]
    review = {
        "id": 1,
        "state": "APPROVED",
        "commit_id": HEAD,
        "user": {"id": 6076111, "login": "MaxGhenis", "type": "User"},
    }
    result = {
        "/repos/TheAxiomFoundation/axiom-encode/actions/runs/12345": {
            "id": 12345,
            "event": "workflow_dispatch",
            "path": ".github/workflows/targeted-signed-reencode.yml",
            "head_branch": "main",
            "head_sha": "e" * 40,
            "status": "in_progress",
            "triggering_actor": {**PAVEL, "type": "User"},
        },
        PR: {
            "number": 42,
            "state": "open",
            "merged": False,
            "changed_files": 2,
            "head": {
                "sha": HEAD,
                "ref": SELECTOR.branch,
                "repo": {"full_name": SELECTOR.repository},
            },
            "base": {
                "sha": BASE,
                "ref": "main",
                "repo": {"full_name": SELECTOR.repository},
            },
            "user": {"id": 110687043},
        },
        ROOT + "/git/ref/heads/main": {"object": {"sha": BASE}},
        ROOT + f"/compare/{BASE}...{HEAD}": {
            "status": "ahead",
            "merge_base_commit": {"sha": BASE},
            "files": [{"filename": p, "status": "added"} for p in files],
        },
        ROOT + f"/git/trees/{BASE}?recursive=1": {"truncated": False, "tree": []},
        ROOT + "/git/ref/heads/codex%2Fmedical": {"object": {"sha": HEAD}},
        PR + "/files?per_page=100&page=1": [
            {"filename": p, "status": "added"} for p in files
        ],
        PR + "/reviews?per_page=100&page=1": [review],
        ROOT + f"/git/trees/{HEAD}?recursive=1": {
            "truncated": False,
            "tree": [
                {"path": p, "type": "blob", "mode": "100644", "sha": "c" * 40}
                for p in files
            ],
        },
    }
    for path in files:
        raw = ("fixture " + path).encode()
        result[f"{ROOT}/contents/{path}?ref={HEAD}"] = {
            "type": "file",
            "path": path,
            "encoding": "base64",
            "content": base64.b64encode(raw).decode(),
            "size": len(raw),
            "sha": "c" * 40,
        }
    return result


@approval_policy
def test_exact_bound_admission(api):
    result = verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    assert result["record"]["base_sha"] == BASE
    assert (
        result["record"]["binding"]["rulespec_sha256"]
        == hashlib.sha256(("fixture " + PATH).encode()).hexdigest()
    )
    assert result["record"]["pr_base_branch"] == SELECTOR.branch
    assert result["approvals"] == [{"review_id": 1, "reviewer_id": 6076111}]
    assert result["admission_basis"] == "owner_approval"
    assert "dispatcher" not in result
    assert "approval" in result["record"]["purpose"]


@pytest.mark.parametrize(
    "field,value",
    [("state", "closed"), ("merged", True), ("number", 43), ("changed_files", 3)],
)
def test_wrong_pr(api, field, value):
    api[PR][field] = value
    with pytest.raises(AdmissionError):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@pytest.mark.parametrize(
    "side,field,value",
    [
        ("head", "sha", BASE),
        ("head", "ref", "other"),
        ("base", "ref", "stacked"),
        ("base", "sha", HEAD),
    ],
)
def test_wrong_head_or_base(api, side, field, value):
    api[PR][side][field] = value
    with pytest.raises(AdmissionError):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


def test_fork(api):
    api[PR]["head"]["repo"]["full_name"] = "attacker/rulespec-us"
    with pytest.raises(AdmissionError):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@approval_policy
@pytest.mark.parametrize(
    "state", ["DISMISSED", "CHANGES_REQUESTED", "COMMENTED", "PENDING"]
)
def test_nonapproval(api, state):
    api[PR + "/reviews?per_page=100&page=1"][0]["state"] = state
    with pytest.raises(AdmissionError):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@approval_policy
@pytest.mark.parametrize(
    "change", ["stale", "author", "actor", "bot", "recycled_login", "unknown_reviewer"]
)
def test_independent_owner_approval(api, change):
    review = api[PR + "/reviews?per_page=100&page=1"][0]
    selector = SELECTOR
    if change == "stale":
        review["commit_id"] = BASE
    if change == "author":
        api[PR]["user"]["id"] = 6076111
    if change == "actor":
        selector = replace(SELECTOR, actor="MaxGhenis")
        api["/repos/TheAxiomFoundation/axiom-encode/actions/runs/12345"][
            "triggering_actor"
        ]["login"] = "MaxGhenis"
    if change == "bot":
        review["user"]["type"] = "Bot"
    if change == "recycled_login":
        review["user"]["id"] = 123
    if change == "unknown_reviewer":
        review["user"]["login"] = "unknown"
    with pytest.raises(AdmissionError):
        verify_reviewed_pr(selector, fetch=api.__getitem__)


@approval_policy
@pytest.mark.parametrize("state", ["DISMISSED", "CHANGES_REQUESTED"])
def test_later_decisive_review_revokes(api, state):
    reviews = api[PR + "/reviews?per_page=100&page=1"]
    later = copy.deepcopy(reviews[0])
    later.update(id=2, state=state)
    reviews.append(later)
    with pytest.raises(AdmissionError):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


def test_comment_does_not_erase_approval(api):
    reviews = api[PR + "/reviews?per_page=100&page=1"]
    later = copy.deepcopy(reviews[0])
    later.update(id=2, state="COMMENTED")
    reviews.append(later)
    verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@pytest.mark.parametrize(
    "path",
    [".github/workflows/ci.yml", "other.yaml", ".axiom/encoding-manifests/us/x.json"],
)
def test_extra_file_refused(api, path):
    api[PR]["changed_files"] = 3
    api[PR + "/files?per_page=100&page=1"].append({"filename": path, "status": "added"})
    with pytest.raises(AdmissionError):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@pytest.mark.parametrize("status", ["modified", "renamed", "removed"])
def test_existing_modules_not_admitted(api, status):
    api[PR + "/files?per_page=100&page=1"][0]["status"] = status
    with pytest.raises(AdmissionError):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@pytest.mark.parametrize("mode", ["120000", "100755", "160000"])
def test_symlink_executable_submodule_refused(api, mode):
    api[ROOT + f"/git/trees/{HEAD}?recursive=1"]["tree"][0]["mode"] = mode
    with pytest.raises(AdmissionError):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


def test_truncated_tree_refused(api):
    api[ROOT + f"/git/trees/{HEAD}?recursive=1"]["truncated"] = True
    with pytest.raises(AdmissionError):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


def test_head_moves_during_query(api):
    count = 0

    def fetch(path):
        nonlocal count
        if path == PR:
            count += 1
            if count == 2:
                changed = copy.deepcopy(api[path])
                changed["head"]["sha"] = BASE
                return changed
        return api[path]

    with pytest.raises(AdmissionError):
        verify_reviewed_pr(SELECTOR, fetch=fetch)


def test_dismissal_during_query(api):
    count = 0

    def fetch(path):
        nonlocal count
        if path == PR + "/reviews?per_page=100&page=1":
            count += 1
            if count == 2:
                changed = copy.deepcopy(api[path])
                changed[0]["state"] = "DISMISSED"
                return changed
        return api[path]

    with pytest.raises(AdmissionError):
        verify_reviewed_pr(SELECTOR, fetch=fetch)


def test_api_error_fails_closed():
    def fetch(path):
        raise OSError("API unavailable")

    with pytest.raises(OSError):
        verify_reviewed_pr(SELECTOR, fetch=fetch)


def test_revalidate_reads_only_mutable_authority(api):
    admission = verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    requested = []

    def fetch(path):
        requested.append(path)
        return api[path]

    assert revalidate_admission(admission, fetch=fetch) == admission
    assert len(requested) == 4
    assert not any("/contents/" in path or "/git/trees/" in path for path in requested)


@approval_policy
@pytest.mark.parametrize(
    "change", ["dismissed", "head", "main", "base", "closed", "run", "actor", "encoder"]
)
def test_revalidate_rejects_changed_authority(api, change):
    admission = verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    run = api["/repos/TheAxiomFoundation/axiom-encode/actions/runs/12345"]
    if change == "dismissed":
        api[PR + "/reviews?per_page=100&page=1"][0]["state"] = "DISMISSED"
    elif change == "head":
        api[PR]["head"]["sha"] = "f" * 40
    elif change == "main":
        api[ROOT + "/git/ref/heads/main"]["object"]["sha"] = "f" * 40
    elif change == "base":
        api[PR]["base"]["sha"] = "f" * 40
    elif change == "closed":
        api[PR]["state"] = "closed"
    elif change == "run":
        run["status"] = "completed"
    elif change == "actor":
        run["triggering_actor"]["login"] = "someone-else"
    else:
        run["head_sha"] = "f" * 40
    with pytest.raises(AdmissionError):
        revalidate_admission(admission, fetch=api.__getitem__)


def test_stale_merge_base_refused(api):
    api[ROOT + f"/compare/{BASE}...{HEAD}"]["merge_base_commit"]["sha"] = "d" * 40
    with pytest.raises(AdmissionError):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


def test_path_now_exists_on_main_despite_added_diff(api):
    api[ROOT + f"/git/trees/{BASE}?recursive=1"]["tree"] = [{"path": PATH}]
    with pytest.raises(AdmissionError):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@pytest.mark.parametrize("hide_in_comparison", [False, True])
def test_mutable_files_cannot_hide_changed_workflow(api, hide_in_comparison):
    workflow = ".github/workflows/ci.yml"
    old = {"path": workflow, "type": "blob", "mode": "100644", "sha": "d" * 40}
    new = {**old, "sha": "e" * 40}
    api[ROOT + f"/git/trees/{BASE}?recursive=1"]["tree"].append(old)
    api[ROOT + f"/git/trees/{HEAD}?recursive=1"]["tree"].append(new)
    api[PR]["changed_files"] = 3
    api[PR + "/files?per_page=100&page=1"].append(
        {"filename": ".axiom/index/provisions_to_rules.json", "status": "modified"}
    )
    api[ROOT + f"/compare/{BASE}...{HEAD}"]["files"].append(
        {
            "filename": ".axiom/index/provisions_to_rules.json"
            if hide_in_comparison
            else workflow,
            "status": "modified",
        }
    )
    with pytest.raises(AdmissionError, match="immutable tree delta"):
        receipt = verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
        # The reported two-call exploit must fail before a receipt exists.
        # Revalidation deliberately checks mutable authority, not immutable
        # objects already inspected and bound by exact commit IDs.
        revalidate_admission(receipt, fetch=api.__getitem__)


def test_allowed_index_delta_and_unchanged_leaves(api):
    index = ".axiom/index/provisions_to_rules.json"
    old = {"path": index, "type": "blob", "mode": "100644", "sha": "d" * 40}
    unchanged = {
        "path": "unrelated.yaml",
        "type": "blob",
        "mode": "100644",
        "sha": "f" * 40,
    }
    api[ROOT + f"/git/trees/{BASE}?recursive=1"]["tree"] += [old, unchanged]
    api[ROOT + f"/git/trees/{HEAD}?recursive=1"]["tree"] += [
        {**old, "sha": "e" * 40},
        unchanged,
    ]
    api[PR]["changed_files"] = 3
    for files in (
        api[PR + "/files?per_page=100&page=1"],
        api[ROOT + f"/compare/{BASE}...{HEAD}"]["files"],
    ):
        files.append({"filename": index, "status": "modified"})
    verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "nonobject",
        "bad_sha",
        "truncated_compare",
        "type_change",
        "mode_change",
        "removed",
    ],
)
def test_complete_tree_delta_fails_closed(api, kind):
    tree = api[ROOT + f"/git/trees/{HEAD}?recursive=1"]["tree"]
    if kind == "duplicate":
        tree.append(copy.deepcopy(tree[0]))
    elif kind == "nonobject":
        tree.append(None)
    elif kind == "bad_sha":
        tree.append({"path": "other", "type": "blob", "mode": "100644", "sha": "bad"})
    elif kind == "truncated_compare":
        api[ROOT + f"/compare/{BASE}...{HEAD}"]["files"].pop()
    else:
        old = {"path": "other", "type": "blob", "mode": "100644", "sha": "e" * 40}
        api[ROOT + f"/git/trees/{BASE}?recursive=1"]["tree"].append(old)
        if kind == "type_change":
            tree.append({**old, "type": "commit", "mode": "160000"})
        elif kind == "mode_change":
            tree.append({**old, "mode": "100755"})
    with pytest.raises(AdmissionError):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@approval_policy
def test_actor_regression_detects_removed_independence_guard(api, monkeypatch):
    from axiom_encode import reviewed_pr_admission as module

    source = inspect.getsource(module._approved_reviews)
    guard = '            and reviewer["login"].lower() != selector.actor.lower()\n'
    assert guard in source
    namespace = dict(vars(module))
    exec(
        compile(source.replace(guard, ""), "<actor-guard-mutation>", "exec"), namespace
    )
    monkeypatch.setattr(module, "_approved_reviews", namespace["_approved_reviews"])
    # The negative test must fail when only actor exclusion is removed; a
    # mismatch with the mocked workflow actor must not mask that mutation.
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        test_independent_owner_approval(api, "actor")


def test_scope_regression_detects_removed_immutable_delta_guard(api, monkeypatch):
    from axiom_encode import reviewed_pr_admission as module

    monkeypatch.setattr(
        module,
        "_require_immutable_delta",
        lambda base, head, files, comparison: module._leaf_entries(head),
    )
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        test_mutable_files_cannot_hide_changed_workflow(api, False)


def test_promotion_wrapper_preserves_contract():
    import json

    from axiom_encode.prepare_signed_backfill import (
        immutable_atomic_source_contract,
        split_atomic_source_input,
    )

    transaction = {
        "schema": "axiom-encode/atomic-source-transaction/v5",
        "source_bundle": [],
        "canonical_refresh_bundle": [],
        "primary_required_test_cases": [],
        "require_complete_source_unit": True,
        "manifest_only_refresh": False,
        "reviewed_candidate_promotion": True,
    }
    wrapper = {
        "schema": "axiom-encode/reviewed-pr-promotion/v1",
        "transaction": transaction,
        "rulespec_pr": 42,
    }
    assert split_atomic_source_input(json.dumps(wrapper))["rulespec_pr"] == 42
    assert immutable_atomic_source_contract(
        json.dumps(wrapper)
    ) == split_atomic_source_input(json.dumps(transaction))
    for injected in ("selector", "approvals", "record", "workflow_commit"):
        forged = {**wrapper, injected: {"approved": True}}
        with pytest.raises(ValueError):
            split_atomic_source_input(json.dumps(forged))
    for invalid in [False, 0, -1, "42", None]:
        wrapper["rulespec_pr"] = invalid
        with pytest.raises(ValueError):
            split_atomic_source_input(json.dumps(wrapper))
    wrapper["rulespec_pr"] = 42
    transaction["reviewed_candidate_promotion"] = False
    with pytest.raises(ValueError):
        split_atomic_source_input(json.dumps(wrapper))


# --- #2301: admission by enrolled-owner dispatch, no approval -----------------


def _actor(api):
    return api[RUN]["triggering_actor"]


def test_constant_alone_decides_whether_an_approval_is_required(api, approval_required):
    api[REVIEWS].clear()
    if approval_required:
        with pytest.raises(AdmissionError, match="lacks an independent"):
            verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    else:
        assert verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)[
            "admission_basis"
        ] == ("owner_dispatch")


@pytest.mark.parametrize("value", [None, 0, "", "False", 1])
def test_only_exactly_false_drops_the_approval(api, monkeypatch, value):
    monkeypatch.setattr(admission_module, "OWNER_APPROVAL_REQUIRED", value)
    api[REVIEWS].clear()
    with pytest.raises(AdmissionError, match="lacks an independent"):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


def test_shipped_policy_is_admission_without_owner_approval():
    source = inspect.getsource(admission_module)
    assert "\nOWNER_APPROVAL_REQUIRED: Final[bool] = False\n" in source


@dispatch_policy
def test_owner_dispatch_admission_names_its_basis_and_dispatcher(api):
    # No review at all, and the dispatching owner is also the PR author.
    api[REVIEWS].clear()
    assert api[PR]["user"]["id"] == PAVEL["id"]
    result = verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    assert set(result) == {
        "schema",
        "workflow_commit",
        "selector",
        "admission_basis",
        "dispatcher",
        "record",
    }
    assert result["schema"] == "axiom-encode/live-reviewed-pr-admission/v2"
    assert result["admission_basis"] == "owner_dispatch"
    assert result["dispatcher"] == PAVEL
    assert result["record"]["purpose"] == (
        "Enrolled signing-owner dispatch of the protected signing run; no "
        "owner approval of the RuleSpec PR; new-module promotion only."
    )
    assert result["record"]["binding"]["rulespec_path"] == PATH


def _other_writer(api, **actor):
    api[RUN]["triggering_actor"] = {
        "id": 999,
        "login": "other-writer",
        "type": "User",
        **actor,
    }
    return replace(SELECTOR, actor=api[RUN]["triggering_actor"]["login"])


def test_non_owner_dispatcher_lands_only_with_an_owner_approval(api, approval_required):
    selector = _other_writer(api)
    if approval_required:
        assert verify_reviewed_pr(selector, fetch=api.__getitem__)["approvals"]
    else:
        # Max's exact-head approval is present and does not help.
        with pytest.raises(AdmissionError, match="enrolled signing owner"):
            verify_reviewed_pr(selector, fetch=api.__getitem__)


@dispatch_policy
@pytest.mark.parametrize(
    "change",
    [
        {"id": 123},  # recycled login
        {"id": 6076111},  # another owner's id under this login
        {"id": "110687043"},
        {"id": 110687043.0},
        {"id": True},
        {"id": None},
        {"type": "Bot"},
        {"type": "Organization"},
        {"type": None},
    ],
    ids=str,
)
def test_owner_dispatcher_identity_must_verify(api, change):
    _actor(api).update(change)
    with pytest.raises(AdmissionError, match="enrolled signing owner"):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@dispatch_policy
@pytest.mark.parametrize("missing", ["id", "type"])
def test_dispatcher_that_cannot_be_verified_is_refused(api, missing):
    del _actor(api)[missing]
    with pytest.raises(AdmissionError, match="enrolled signing owner"):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@dispatch_policy
@pytest.mark.parametrize("actor", [None, "PavelMakarchuk", []])
def test_dispatcher_object_absent_is_refused(api, actor):
    api[RUN]["triggering_actor"] = actor
    with pytest.raises(AdmissionError):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@dispatch_policy
def test_renamed_owner_login_is_refused(api):
    # The enrolled id under a login that is not the enrolled login.
    selector = _other_writer(api, id=PAVEL["id"])
    with pytest.raises(AdmissionError, match="enrolled signing owner"):
        verify_reviewed_pr(selector, fetch=api.__getitem__)


def test_enrolled_dispatcher_binds_the_selector_actor(api):
    # The helper does not rely on its caller's login comparison.
    selector = replace(SELECTOR, actor="MaxGhenis")
    assert admission_module._enrolled_dispatcher(api[RUN], SELECTOR) == PAVEL
    with pytest.raises(AdmissionError, match="enrolled signing owner"):
        admission_module._enrolled_dispatcher(api[RUN], selector)


@dispatch_policy
def test_rerun_by_a_non_owner_is_refused(api):
    # GitHub reports the account that re-ran as the triggering actor.
    api[RUN]["actor"] = {**PAVEL, "type": "User"}
    selector = _other_writer(api)
    with pytest.raises(AdmissionError, match="enrolled signing owner"):
        verify_reviewed_pr(selector, fetch=api.__getitem__)


@dispatch_policy
@pytest.mark.parametrize(
    "reviewer",
    [
        {"id": 6076111, "login": "MaxGhenis", "type": "User"},
        {"id": 555, "login": "outside-reader", "type": "User"},
        {"id": 110687043, "login": "PavelMakarchuk", "type": "User"},
    ],
    ids=["other-owner", "any-account", "dispatcher"],
)
def test_changes_requested_still_refuses_owner_dispatch(api, reviewer):
    api[REVIEWS][:] = [
        {"id": 7, "state": "CHANGES_REQUESTED", "commit_id": BASE, "user": reviewer}
    ]
    with pytest.raises(AdmissionError, match="unresolved changes requested"):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@dispatch_policy
@pytest.mark.parametrize("state", ["APPROVED", "DISMISSED"])
def test_veto_ends_when_its_reviewer_withdraws_it(api, state):
    reviewer = {"id": 555, "login": "outside-reader", "type": "User"}
    api[REVIEWS][:] = [
        {"id": 7, "state": "CHANGES_REQUESTED", "commit_id": HEAD, "user": reviewer},
        {"id": 8, "state": state, "commit_id": HEAD, "user": reviewer},
    ]
    assert verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)["dispatcher"] == PAVEL


@dispatch_policy
@pytest.mark.parametrize("state", ["DISMISSED", "COMMENTED", "PENDING"])
def test_owner_dispatch_needs_no_approval(api, state):
    api[REVIEWS][0]["state"] = state
    assert verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)["dispatcher"] == PAVEL


def test_unknown_review_state_refuses_under_both_policies(api):
    api[REVIEWS].append(
        {"id": 9, "state": "SUPERSEDED", "user": {"id": 555, "login": "x"}}
    )
    with pytest.raises(AdmissionError, match="unknown review state"):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@dispatch_policy
def test_owner_dispatch_revalidation_reads_only_mutable_authority(api):
    admission = verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    requested = []

    def fetch(path):
        requested.append(path)
        return api[path]

    assert revalidate_admission(admission, fetch=fetch) == admission
    assert requested == [RUN, PR, ROOT + "/git/ref/heads/main", REVIEWS]
    # The approval is not this admission's authority: its dismissal is no change.
    api[REVIEWS][0]["state"] = "DISMISSED"
    assert revalidate_admission(admission, fetch=api.__getitem__) == admission


@dispatch_policy
@pytest.mark.parametrize(
    "change",
    [
        "changes_requested",
        "head",
        "main",
        "base",
        "closed",
        "run",
        "actor",
        "actor_id",
        "actor_type",
        "actor_unverifiable",
        "encoder",
    ],
)
def test_owner_dispatch_revalidation_rejects_changed_authority(api, change):
    admission = verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    run = api[RUN]
    if change == "changes_requested":
        api[REVIEWS].append(
            {
                "id": 9,
                "state": "CHANGES_REQUESTED",
                "user": {"id": 555, "login": "outside-reader", "type": "User"},
            }
        )
    elif change == "head":
        api[PR]["head"]["sha"] = "f" * 40
    elif change == "main":
        api[ROOT + "/git/ref/heads/main"]["object"]["sha"] = "f" * 40
    elif change == "base":
        api[PR]["base"]["sha"] = "f" * 40
    elif change == "closed":
        api[PR]["state"] = "closed"
    elif change == "run":
        run["status"] = "completed"
    elif change == "actor":
        run["triggering_actor"]["login"] = "someone-else"
    elif change == "actor_id":
        run["triggering_actor"]["id"] = 123
    elif change == "actor_type":
        run["triggering_actor"]["type"] = "Bot"
    elif change == "actor_unverifiable":
        del run["triggering_actor"]["id"]
    else:
        run["head_sha"] = "f" * 40
    with pytest.raises(AdmissionError):
        revalidate_admission(admission, fetch=api.__getitem__)


@dispatch_policy
@pytest.mark.parametrize(
    "dispatcher",
    [
        {"id": 6076111, "login": "MaxGhenis"},
        {"id": 110687043, "login": "MaxGhenis"},
        {"id": 110687043},
        None,
    ],
)
def test_revalidation_compares_the_recorded_dispatcher(api, dispatcher):
    admission = verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    admission["dispatcher"] = dispatcher
    with pytest.raises(AdmissionError, match="dispatching owner changed"):
        revalidate_admission(admission, fetch=api.__getitem__)


def _no_fetch(path):
    raise AssertionError("a refused record must not reach GitHub: " + path)


@pytest.mark.parametrize(
    "basis", ["owner_waiver", "", "OWNER_DISPATCH", None, 0, ["owner_dispatch"]]
)
def test_unknown_admission_basis_is_refused(api, basis):
    admission = verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    admission["admission_basis"] = basis
    with pytest.raises(AdmissionError, match="unknown admission basis"):
        revalidate_admission(admission, fetch=_no_fetch)


def test_admission_without_a_basis_is_refused(api):
    admission = verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    del admission["admission_basis"]
    with pytest.raises(AdmissionError, match="unknown admission basis"):
        revalidate_admission(admission, fetch=_no_fetch)


def test_basis_must_be_the_policy_in_force(api, approval_required, monkeypatch):
    admission = verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    assert revalidate_admission(admission, fetch=api.__getitem__) == admission
    monkeypatch.setattr(
        admission_module, "OWNER_APPROVAL_REQUIRED", not approval_required
    )
    with pytest.raises(AdmissionError, match="not the policy in force"):
        revalidate_admission(admission, fetch=_no_fetch)


def test_basis_and_evidence_field_must_agree(api, approval_required):
    admission = verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    own, other = (
        ("approvals", "dispatcher")
        if approval_required
        else ("dispatcher", "approvals")
    )
    swapped = {key: value for key, value in admission.items() if key != own}
    for forged in (
        {**admission, other: []},  # both evidence fields
        {**swapped, other: []},  # the other basis's field, empty
        swapped,  # no evidence field
        {**admission, "schema": "axiom-encode/live-reviewed-pr-admission/v1"},
        {**admission, "extra": True},
    ):
        with pytest.raises(AdmissionError, match="invalid provisioned admission"):
            revalidate_admission(forged, fetch=_no_fetch)


@approval_policy
def test_pre_basis_record_shape_is_refused(api):
    admission = verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    legacy = {
        **{key: value for key, value in admission.items() if key != "admission_basis"},
        "schema": "axiom-encode/live-reviewed-pr-admission/v1",
    }
    with pytest.raises(AdmissionError, match="unknown admission basis"):
        revalidate_admission(legacy, fetch=_no_fetch)
