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
# A promotable primary: the citation and a 64-hex pin, in the plain block form.
PIN = "d" * 64
PRIMARY = (
    "format: rulespec/v1\n"
    "module:\n"
    "  source_verification:\n"
    f"    corpus_citation_path: {SELECTOR.citation}\n"
    f"    source_sha256: {PIN}\n"
    "  summary: |-\n"
    "    Synthetic fixture text.\n"
    "rules: []\n"
).encode()


def _fixture_bytes(path: str) -> bytes:
    return PRIMARY if path == PATH else ("fixture " + path).encode()


def _set_primary(api: dict, raw: bytes) -> None:
    entry = api[f"{ROOT}/contents/{PATH}?ref={HEAD}"]
    entry["content"] = base64.b64encode(raw).decode()
    entry["size"] = len(raw)


ROOT = "/repos/TheAxiomFoundation/rulespec-us"
PR = ROOT + "/pulls/42"
RUN = "/repos/TheAxiomFoundation/axiom-encode/actions/runs/12345"
REVIEWS = PR + "/reviews?per_page=100&page=1"
PAVEL = {"id": 110687043, "login": "PavelMakarchuk"}
DISPATCHER = {
    "login": "PavelMakarchuk",
    "actor_id": 110687043,
    "triggering_actor_id": 110687043,
}


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
            "actor": {**PAVEL, "type": "User"},
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
        raw = _fixture_bytes(path)
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
        == hashlib.sha256(PRIMARY).hexdigest()
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
    assert result["dispatcher"] == DISPATCHER
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
    assert admission_module._enrolled_dispatcher(api[RUN], SELECTOR) == DISPATCHER
    with pytest.raises(AdmissionError, match="enrolled signing owner"):
        admission_module._enrolled_dispatcher(api[RUN], selector)


def test_enrolled_dispatcher_checks_the_login_of_the_live_actor(api):
    # The enrolled id under another login, while the selector names the owner.
    _actor(api)["login"] = "renamed-account"
    with pytest.raises(AdmissionError, match="enrolled signing owner"):
        admission_module._enrolled_dispatcher(api[RUN], SELECTOR)


def test_enrolled_dispatcher_reads_the_triggering_actor_not_the_first_actor(api):
    # An owner started the run; another account with a recycled login re-ran it.
    api[RUN]["actor"] = {**PAVEL, "type": "User"}
    _actor(api)["id"] = 999
    with pytest.raises(AdmissionError, match="enrolled signing owner"):
        admission_module._enrolled_dispatcher(api[RUN], SELECTOR)


@approval_policy
def test_revalidate_rejects_a_different_approval(api):
    admission = verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    again = copy.deepcopy(api[REVIEWS][0])
    again["id"] = 2
    api[REVIEWS].append(again)
    with pytest.raises(AdmissionError, match="differs from provisioned"):
        revalidate_admission(admission, fetch=api.__getitem__)


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
    assert (
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)["dispatcher"] == DISPATCHER
    )


@dispatch_policy
@pytest.mark.parametrize("state", ["DISMISSED", "COMMENTED", "PENDING"])
def test_owner_dispatch_needs_no_approval(api, state):
    api[REVIEWS][0]["state"] = state
    assert (
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)["dispatcher"] == DISPATCHER
    )


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
        {"login": "MaxGhenis", "actor_id": 6076111, "triggering_actor_id": 6076111},
        {**DISPATCHER, "login": "MaxGhenis"},
        {**DISPATCHER, "actor_id": 6076111},
        {**DISPATCHER, "triggering_actor_id": 6076111},
        {"login": "PavelMakarchuk", "triggering_actor_id": 110687043},
        PAVEL,
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


# --- review round 1 -----------------------------------------------------------


def _max_account():
    return {"id": 6076111, "login": "MaxGhenis", "type": "User"}


@dispatch_policy
@pytest.mark.parametrize(
    "run_actor",
    [
        {"id": 999, "login": "other-writer", "type": "User"},
        {"id": 999, "login": "PavelMakarchuk", "type": "User"},
        {"id": 6076111, "login": "MaxGhenis", "type": "User"},
        {"id": 110687043, "login": "PavelMakarchuk", "type": "Bot"},
        {"id": 110687043, "login": "PavelMakarchuk"},
        {"login": "PavelMakarchuk", "type": "User"},
        {"id": "110687043", "login": "PavelMakarchuk", "type": "User"},
        "PavelMakarchuk",
        None,
    ],
    ids=str,
)
def test_owner_rerun_of_a_run_that_another_account_dispatched_is_refused(
    api, run_actor
):
    # ``actor`` chose the inputs; ``triggering_actor`` is the enrolled owner.
    api[RUN]["actor"] = run_actor
    with pytest.raises(AdmissionError):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@dispatch_policy
@pytest.mark.parametrize(
    "writer",
    [
        {"id": 999, "login": "other-writer", "type": "User"},
        {"id": 6076111, "login": "other-writer", "type": "User"},
        {"id": 999, "login": "MaxGhenis", "type": "User"},
    ],
    ids=["not-enrolled", "enrolled-id-other-login", "enrolled-login-other-id"],
)
def test_one_account_that_is_not_enrolled_in_both_run_fields_is_refused(api, writer):
    # Both run accounts and the selector agree, so only the enrolment rule
    # stands between this account and an admission.
    api[RUN]["actor"] = dict(writer)
    api[RUN]["triggering_actor"] = dict(writer)
    selector = replace(SELECTOR, actor=writer["login"])
    with pytest.raises(AdmissionError, match="enrolled signing owner"):
        verify_reviewed_pr(selector, fetch=api.__getitem__)
    with pytest.raises(AdmissionError, match="enrolled signing owner"):
        admission_module._enrolled_dispatcher(api[RUN], selector)


@dispatch_policy
def test_revalidation_refuses_one_unenrolled_account_in_both_run_fields(api):
    import json

    admission = json.loads(
        json.dumps(verify_reviewed_pr(SELECTOR, fetch=api.__getitem__))
    )
    writer = {"id": 999, "login": "other-writer", "type": "User"}
    api[RUN]["actor"] = dict(writer)
    api[RUN]["triggering_actor"] = dict(writer)
    # A record forged to agree with the live run does not help either.
    forged = copy.deepcopy(admission)
    forged["selector"]["actor"] = writer["login"]
    assert forged["dispatcher"]["login"] == "PavelMakarchuk"
    forged["dispatcher"] = {
        "login": writer["login"],
        "actor_id": writer["id"],
        "triggering_actor_id": writer["id"],
    }
    for candidate in (admission, forged):
        with pytest.raises(AdmissionError):
            revalidate_admission(candidate, fetch=api.__getitem__)


@dispatch_policy
def test_run_without_an_actor_field_is_refused(api):
    del api[RUN]["actor"]
    with pytest.raises(AdmissionError, match="run actor"):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@dispatch_policy
def test_run_without_a_triggering_actor_does_not_fall_back_to_the_actor(api):
    del api[RUN]["triggering_actor"]
    with pytest.raises(AdmissionError, match="triggering actor"):
        admission_module._enrolled_dispatcher(api[RUN], SELECTOR)


@approval_policy
def test_approval_policy_does_not_read_the_run_actor(api):
    # The rule in force before #2301: any account may have dispatched.
    api[RUN]["actor"] = {"id": 999, "login": "other-writer", "type": "User"}
    assert verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)["approvals"]


@dispatch_policy
@pytest.mark.parametrize("field", ["actor", "triggering_actor"])
def test_revalidation_rereads_both_run_accounts(api, field):
    admission = verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    api[RUN][field]["id"] = 999
    with pytest.raises(AdmissionError, match="enrolled signing owner"):
        revalidate_admission(admission, fetch=api.__getitem__)


def test_logins_are_compared_exactly(api):
    lower = "pavelmakarchuk"
    with pytest.raises(AdmissionError, match="enrolled signing owner"):
        admission_module._enrolled_dispatcher(api[RUN], replace(SELECTOR, actor=lower))
    for field in ("actor", "triggering_actor"):
        run = copy.deepcopy(api[RUN])
        run[field]["login"] = lower
        with pytest.raises(AdmissionError, match="enrolled signing owner"):
            admission_module._enrolled_dispatcher(run, SELECTOR)
    run = copy.deepcopy(api[RUN])
    run["actor"]["login"] = run["triggering_actor"]["login"] = lower
    with pytest.raises(AdmissionError, match="enrolled signing owner"):
        admission_module._enrolled_dispatcher(run, replace(SELECTOR, actor=lower))


@dispatch_policy
def test_revalidation_of_a_record_read_back_from_json(api):
    # Production reads the record from the runtime attestation file.
    import json

    admission = json.loads(
        json.dumps(verify_reviewed_pr(SELECTOR, fetch=api.__getitem__))
    )
    assert admission["admission_basis"] is not admission_module.BASIS_OWNER_DISPATCH
    assert revalidate_admission(admission, fetch=api.__getitem__) == admission
    api[RUN]["triggering_actor"]["id"] = 999
    with pytest.raises(AdmissionError, match="enrolled signing owner"):
        revalidate_admission(admission, fetch=api.__getitem__)
    api[RUN]["triggering_actor"]["id"] = PAVEL["id"]
    admission["dispatcher"]["actor_id"] = 6076111
    with pytest.raises(AdmissionError, match="dispatching owner changed"):
        revalidate_admission(admission, fetch=api.__getitem__)


def _comments(count):
    return [
        {
            "id": number,
            "state": "COMMENTED",
            "user": {"id": 555, "login": "outside-reader", "type": "User"},
        }
        for number in range(100, 100 + count)
    ]


@dispatch_policy
@pytest.mark.parametrize("stage", ["admission", "revalidation"])
def test_changes_requested_on_the_second_review_page_refuses(api, stage):
    veto = {
        "id": 500,
        "state": "CHANGES_REQUESTED",
        "user": {"id": 777, "login": "second-page", "type": "User"},
    }
    api[REVIEWS][:] = _comments(100)
    api[PR + "/reviews?per_page=100&page=2"] = []
    if stage == "admission":
        api[PR + "/reviews?per_page=100&page=2"] = [veto]
        with pytest.raises(AdmissionError, match="unresolved changes requested"):
            verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
        return
    admission = verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    assert revalidate_admission(admission, fetch=api.__getitem__) == admission
    api[PR + "/reviews?per_page=100&page=2"] = [veto]
    with pytest.raises(AdmissionError, match="unresolved changes requested"):
        revalidate_admission(admission, fetch=api.__getitem__)


def test_review_order_is_by_review_id_not_by_api_order(api):
    reviewer = {"id": 555, "login": "outside-reader", "type": "User"}
    api[REVIEWS].extend(
        [
            {"id": 9, "state": "CHANGES_REQUESTED", "user": reviewer},
            {"id": 8, "state": "APPROVED", "commit_id": HEAD, "user": reviewer},
        ]
    )
    with pytest.raises(AdmissionError, match="unresolved changes requested"):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@pytest.mark.parametrize("bad", [None, 0, -1, True, "7", 7.0], ids=repr)
@pytest.mark.parametrize("where", ["review", "reviewer"])
def test_review_with_a_malformed_identity_refuses(api, where, bad):
    veto = {
        "id": 7,
        "state": "CHANGES_REQUESTED",
        "user": {"id": 555, "login": "outside-reader", "type": "User"},
    }
    if where == "review":
        veto["id"] = bad
    else:
        veto["user"]["id"] = bad
    api[REVIEWS].append(veto)
    with pytest.raises(AdmissionError, match="identity is malformed"):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@pytest.mark.parametrize("key", ["id", "user"])
def test_review_without_an_identity_refuses(api, key):
    del api[REVIEWS][0][key]
    with pytest.raises(AdmissionError):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@dispatch_policy
def test_an_approval_without_a_reviewer_id_cannot_hide_a_veto(api):
    # Reviewer probe: two reviewers with no id collided on one key.
    api[REVIEWS][:] = [
        {"id": 7, "state": "CHANGES_REQUESTED", "user": {"login": "first"}},
        {"id": 8, "state": "APPROVED", "commit_id": HEAD, "user": {"login": "second"}},
    ]
    with pytest.raises(AdmissionError, match="identity is malformed"):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@dispatch_policy
def test_a_veto_without_a_review_id_cannot_lose_to_an_earlier_approval(api):
    # Reviewer probe: a missing review id sorted first.
    reviewer = {"id": 555, "login": "outside-reader", "type": "User"}
    api[REVIEWS][:] = [
        {"id": 7, "state": "APPROVED", "commit_id": HEAD, "user": reviewer},
        {"state": "CHANGES_REQUESTED", "user": reviewer},
    ]
    with pytest.raises(AdmissionError, match="identity is malformed"):
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)


@dispatch_policy
def test_malformed_review_refuses_at_revalidation(api):
    admission = verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    api[REVIEWS].append({"state": "COMMENTED", "user": {"id": 555, "login": "x"}})
    with pytest.raises(AdmissionError, match="identity is malformed"):
        revalidate_admission(admission, fetch=api.__getitem__)


# --- the candidate source pin (signing run 45328443323) ---


def _primary(block: str, *, head: str = "format: rulespec/v1\nmodule:\n") -> bytes:
    return (head + block + "rules: []\n").encode()


BOUND_BLOCK = (
    "  source_verification:\n"
    f"    corpus_citation_path: {SELECTOR.citation}\n"
    f"    source_sha256: {PIN}\n"
)


def test_admission_refuses_a_candidate_without_the_pin_and_names_the_command(api):
    _set_primary(
        api,
        _primary(
            f"  source_verification:\n    corpus_citation_path: {SELECTOR.citation}\n"
        ),
    )
    with pytest.raises(AdmissionError) as caught:
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    message = str(caught.value)
    assert "source_sha256` is missing" in message
    assert "axiom-encode bind-source-verification" in message
    assert admission_module.SOURCE_PIN_COMMAND in message


@pytest.mark.parametrize(
    "block, detail",
    [
        ("  summary: x\n", "no `module.source_verification` block"),
        (
            f"  source_verification:\n    source_sha256: {PIN}\n",
            "corpus_citation_path` is missing",
        ),
        (
            "  source_verification:\n"
            f"    corpus_citation_path: us-az/manual/other\n    source_sha256: {PIN}\n",
            "not the promotion citation",
        ),
        (
            BOUND_BLOCK.replace(PIN, PIN[:-1]),
            "not 64 lowercase hex",
        ),
        (BOUND_BLOCK.replace(PIN, PIN.upper()), "not 64 lowercase hex"),
        (BOUND_BLOCK.replace(PIN, "1" * 64), "not a plain string"),
        (BOUND_BLOCK.replace(PIN, PIN + " extra"), "plain block form"),
        (BOUND_BLOCK + f"    source_sha256: {PIN}\n", "no duplicate key"),
        (BOUND_BLOCK + "    nested:\n      key: value\n", "plain block form"),
        # One column short of the block's entries: not a key of the block.
        (
            BOUND_BLOCK.replace("    source_sha256", "   xsource_sha256"),
            "plain block form",
        ),
        (BOUND_BLOCK.replace(PIN, "&a " + PIN), "plain block form"),
        (BOUND_BLOCK.replace(PIN, "!!str " + PIN), "plain block form"),
        (BOUND_BLOCK.replace(PIN, "|\n      " + PIN), "not a plain string"),
        (BOUND_BLOCK.replace("    source", "\tsource"), "indented with tabs"),
        (BOUND_BLOCK.replace("    source", "    \tsource"), "plain block form"),
        ("  \tsummary: x\n" + BOUND_BLOCK, "indented with tabs"),
        (
            "  source_verification: {corpus_citation_path: "
            f"{SELECTOR.citation}, source_sha256: {PIN}}}\n",
            "not a single plain block mapping",
        ),
        (BOUND_BLOCK + BOUND_BLOCK, "not a single plain block mapping"),
        (
            BOUND_BLOCK.replace("source_verification:", "source_verification: &a"),
            "an anchor, an alias or a tag",
        ),
        (
            BOUND_BLOCK.replace("source_verification:", '"source_verification":'),
            "not a plain `name:` key",
        ),
        ("  source_verification:\n  summary: x\n", "is empty"),
        ("  <<: *base\n" + BOUND_BLOCK, "an anchor, an alias or a tag"),
        ("  <<:\n    a: 1\n" + BOUND_BLOCK, "a merge key"),
        (" odd: 1\n".join(["  summary: x\n", BOUND_BLOCK]), "indented evenly"),
    ],
)
def test_admission_refuses_a_missing_or_malformed_pin(api, block, detail):
    _set_primary(api, _primary(block))
    with pytest.raises(AdmissionError) as caught:
        verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    assert detail in str(caught.value)
    assert "bind-source-verification" in str(caught.value)


@pytest.mark.parametrize(
    "raw, detail",
    [
        (b"\xff\xfe", "not UTF-8"),
        (_primary(BOUND_BLOCK).replace(b"\n", b"\r"), "line break"),
        (b"---\n" + _primary(BOUND_BLOCK), "document marker"),
        (b"\xef\xbb\xbf" + _primary(BOUND_BLOCK), "byte order mark"),
        (_primary(BOUND_BLOCK, head="format: rulespec/v1\n"), "no top-level plain"),
        (_primary(BOUND_BLOCK) + _primary(BOUND_BLOCK), "`module` occurs twice"),
        (
            _primary("", head="module: {source_verification: {}}\n"),
            "`module:` is not a plain block mapping",
        ),
        (_primary("", head="module:\n"), "`module` is empty"),
        (_primary("\tsummary: x\n" + BOUND_BLOCK, head="module:\n  a: 1\n"), "tabs"),
    ],
)
def test_the_strict_reader_refuses_every_other_file_shape(raw, detail):
    with pytest.raises(AdmissionError) as caught:
        admission_module.require_candidate_source_pin(raw, SELECTOR.citation)
    assert detail in str(caught.value)


@pytest.mark.parametrize(
    "block",
    [
        BOUND_BLOCK,
        BOUND_BLOCK.replace("\n", "\r\n"),
        BOUND_BLOCK.replace(PIN, f"'{PIN}'  # pinned"),
        BOUND_BLOCK.replace(SELECTOR.citation, f'"{SELECTOR.citation}"'),
        "  summary: |-\n    source_verification: not a key\n\n  # note\n" + BOUND_BLOCK,
        "      source_verification:\n"
        f"          # lead\n          source_sha256:   {PIN}\n"
        f"          corpus_citation_path: {SELECTOR.citation}\n",
    ],
)
def test_the_strict_reader_accepts_the_plain_block_form(block):
    raw = _primary(block, head="format: rulespec/v1\n# c\nmodule:\n")
    assert admission_module.read_candidate_source_pin(raw) == (SELECTOR.citation, PIN)
    admission_module.require_candidate_source_pin(raw, SELECTOR.citation)


def test_the_pin_check_is_the_only_change_to_admission_for_a_pinned_candidate(
    api, monkeypatch
):
    """Differential: with the check switched off the admission is the same."""
    with_check = verify_reviewed_pr(SELECTOR, fetch=copy.deepcopy(api).__getitem__)
    monkeypatch.setattr(
        admission_module, "require_candidate_source_pin", lambda raw, citation: None
    )
    without_check = verify_reviewed_pr(SELECTOR, fetch=copy.deepcopy(api).__getitem__)
    assert with_check == without_check
    assert with_check["record"]["binding"] == {
        "citation": SELECTOR.citation,
        "rulespec_path": PATH,
        "rulespec_sha256": hashlib.sha256(PRIMARY).hexdigest(),
        "companion_sha256": hashlib.sha256(
            _fixture_bytes(SELECTOR.companion)
        ).hexdigest(),
    }


def test_the_pin_check_reads_the_primary_and_not_the_companion(api, monkeypatch):
    seen = []
    monkeypatch.setattr(
        admission_module,
        "require_candidate_source_pin",
        lambda raw, citation: seen.append((raw, citation)),
    )
    verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    assert seen == [(PRIMARY, SELECTOR.citation)]


def test_the_admission_module_stays_standard_library_only():
    source = inspect.getsource(admission_module)
    assert "import yaml" not in source
    assert "from axiom_encode" not in source and "from ." not in source


# --- the reader agrees with a YAML parser or refuses (review round 1) ---

GOOD = _primary(BOUND_BLOCK)
OTHER = "    corpus_citation_path: us-az/manual/other\n"


@pytest.mark.parametrize(
    "raw",
    [
        # A second top-level key that YAML reads as `module` replaces the block.
        GOOD + b'"modul\\x65":\n  source_verification:\n' + OTHER.encode(),
        GOOD + b"? module\n: source_verification:\n" + OTHER.encode(),
        GOOD + b"!!str module:\n  status: x\n",
        GOOD + b"&a module:\n  status: x\n",
        GOOD + b"'module':\n  status: x\n",
        GOOD + b"{module: {status: x}}\n",
        # A second key under `module` that YAML reads as `source_verification`.
        _primary(BOUND_BLOCK + '  "source_verificatio\\x6e":\n' + OTHER),
        _primary(BOUND_BLOCK + "  !!str source_verification:\n" + OTHER),
        _primary(BOUND_BLOCK + "  ? source_verification\n  :" + OTHER.lstrip(" ")),
        # The block sits inside a scalar: YAML reads no block at all.
        _primary('  summary: "x\n' + BOUND_BLOCK + '  y"\n'),
        _primary("  summary: 'x\n" + BOUND_BLOCK + "  y'\n"),
        _primary("  summary: |\n  " + BOUND_BLOCK.replace("\n", "\n  ")),
        _primary(BOUND_BLOCK, head='doc: "x\nmodule:\n').replace(
            b"rules: []", b'  y"\nrules: []'
        ),
        _primary("  deferred:\n    - reason: 'x\n" + BOUND_BLOCK + "      y'\n"),
        _primary("  list: [a,\n" + BOUND_BLOCK + "  b]\n"),
        # The pin line continues: YAML reads a longer value.
        _primary(BOUND_BLOCK + "      extra\n"),
        _primary(BOUND_BLOCK.replace("    source_sha256", "     source_sha256")),
        # An anchored block reused by an alias, a tag on the mapping.
        _primary("  base: &b\n    a: 1\n" + BOUND_BLOCK + "  other: *b\n"),
        _primary(
            BOUND_BLOCK.replace("source_verification:", "source_verification: !!map")
        ),
        # One quoted scalar from the first line to the last: no module at all.
        b"doc: 'x\nmodule:\n" + BOUND_BLOCK.encode() + b"rules: []'\n",
        # A quote inside a flow sequence hides the block up to `"]`.
        _primary('  x: ["a\n' + BOUND_BLOCK + '  b: 1 # "]\n'),
        # A key before `module` that YAML also reads as `module`.
        b'"modul\\x65":\n  status: x\n' + GOOD,
        b"? module\n: {}\n" + GOOD,
        # Text after a quoted scalar that closed on a later line.
        _primary("  summary: 'a\n    b' trailing\n" + BOUND_BLOCK),
        # A tagged or anchored key deeper in `module`.
        _primary("  nested:\n    &a k: 1\n" + BOUND_BLOCK),
        _primary("  nested:\n    !!str k: 1\n" + BOUND_BLOCK),
        # A root mapping that starts indented and then dedents is not YAML.
        b" module:\n" + GOOD,
        b"# c\n\n  format: x\n" + GOOD,
        # NEL is a line break for a YAML parser.
        _primary(BOUND_BLOCK).replace(b"\nrules", "\u0085rules".encode()),
    ],
)
def test_the_reader_refuses_what_a_yaml_parser_could_read_differently(raw):
    with pytest.raises(AdmissionError):
        admission_module.require_candidate_source_pin(raw, SELECTOR.citation)


@pytest.mark.parametrize(
    "block",
    [
        # The shapes of the encoder's own YAML dump: list items at the key
        # column, and a quoted scalar that continues over lines.
        '  deferred_outputs:\n  - output: x\n    reason: \'it is "q" and\n'
        "      source_verification: text '\n" + BOUND_BLOCK,
        "  summary: 'a\n\n    b: c'\n" + BOUND_BLOCK,
        '  summary: "a \\"q\\" \\\n    b"\n' + BOUND_BLOCK,
        "  summary: plain text\n    continued 'here\n" + BOUND_BLOCK,
        # A double-quoted scalar may continue at the key column.
        '  summary: "a\n  b"\n' + BOUND_BLOCK,
        # Block scalar text is not lexed: an open quote there opens nothing.
        "  summary: |-\n    \"unclosed quote text\n    it's\n" + BOUND_BLOCK,
        # `''` inside a single-quoted scalar is one quote character.
        "  summary: 'it''s fine'\n" + BOUND_BLOCK,
        '  summary: "a \\" b"\n' + BOUND_BLOCK,
        # A continuation line of a plain scalar may start with a quote.
        "  summary: plain\n    'here and\n    \"there\n" + BOUND_BLOCK,
        "  summary: >-\n    source_verification:\n      source_sha256: x\n"
        + BOUND_BLOCK,
        "  flags: [a, b]\n  empty: {}\n" + BOUND_BLOCK + "    note: true\n",
    ],
)
def test_the_reader_follows_scalars_over_lines_and_agrees_with_yaml(block):
    import yaml

    raw = _primary(block).replace(b"rules: []", b"rules:\n- name: r\n  x: 1")
    parsed = yaml.safe_load(raw)["module"]["source_verification"]
    assert (parsed["corpus_citation_path"], parsed["source_sha256"]) == (
        SELECTOR.citation,
        PIN,
    )
    assert admission_module.read_candidate_source_pin(raw) == (SELECTOR.citation, PIN)


@pytest.mark.parametrize(
    "value, plain",
    [
        ("us-az/manual/x", True),
        ("3f" * 32, True),
        ("1" * 64, False),
        ("12e" + "4" * 61, False),
        ("1.e3", False),
        ("1.5e3", False),
        ("0o17", False),
        ("0x1f", False),
        ("1_000", False),
        ("1:30", False),
        (".inf", False),
        ("-.5", False),
        ("true", False),
        ("Yes", False),
        ("null", False),
        ("~", False),
        ("a b", False),
        ("a#b", False),
    ],
)
def test_plain_scalar_is_string(value, plain):
    assert admission_module.plain_scalar_is_string(value) is plain


def test_a_refusal_carries_its_detail_and_says_plain_block_form():
    with pytest.raises(AdmissionError) as caught:
        admission_module.require_candidate_source_pin(
            b"---\n" + GOOD, SELECTOR.citation
        )
    assert caught.value.detail.startswith("the module has a byte order mark")
    assert "must be in the plain block form" in str(caught.value)
    assert "rewrite the block by hand" in str(caught.value)
