"""Change 5 of docs/content-bound-merge-guard.md (#2172): switched off.

The unreviewed-candidate lane admits a candidate without the owner approval
before signing and writes ``applied-rulespec/v6``. Three locks keep it shut:
``V6_WRITER_ENABLED``, ``GUARD_APP_ID`` and the live read that the merge guard
is a required check of ``main``. On-state behaviour is reached in tests only,
by patching the constants.
"""

from __future__ import annotations

import copy
import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode import applied_manifest_v6 as v6
from axiom_encode import cli
from axiom_encode import content_merge_guard as guard
from axiom_encode import reviewed_pr_admission as admission
from axiom_encode.cli import (
    _APPLY_VALIDATION_SNAPSHOT_ATTR,
    _TRUSTED_RUNTIME_ENV,
    APPLIED_ENCODING_OFFICIAL_REPOSITORY,
    _applied_encoding_manifest_path,
    cmd_promote_reviewed_candidate,
)
from axiom_encode.reviewed_pr_admission import AdmissionError
from tests.test_applied_manifest_v6 import _sign, _to_v6
from tests.test_cli import (
    AXIOM_ENCODE_TEST_VERSION,
    TEST_APPLY_SIGNING_BROKER,
    _manifest_refresh_source_unit,
    _reviewed_candidate_binding,
    _reviewed_candidate_checkout,
    _write_reviewed_rulespec_records,
)
from tests.test_encoder_manifest_ancestry import (  # noqa: F401 (fixtures)
    _expected,
    _load,
    _reviewed_promotion,
    _set_producer,
    graph,
    manifest_fixture,
)
from tests.test_reviewed_pr_admission import (  # noqa: F401 (fixture)
    BASE,
    HEAD,
    PR,
    ROOT,
    SELECTOR,
    api,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
APP = 987654
PAVEL = {"id": 110687043, "login": "PavelMakarchuk", "type": "User"}
MAX = {"id": 6076111, "login": "MaxGhenis", "type": "User"}
RUN = "/repos/TheAxiomFoundation/axiom-encode/actions/runs/12345"
BRANCH = ROOT + "/branches/main"
REVIEWS = PR + "/reviews?per_page=100&page=1"
RECORDED_BRANCH = json.loads(
    (
        REPOSITORY_ROOT
        / "tests/fixtures/merge_guard_requirement/rulespec_us_branch_main.json"
    ).read_text("utf-8")
)["response"]
REVIEWED_KEYS = {"schema", "selector", "record", "approvals", "workflow_commit"}


def required(*checks, level="everyone"):
    """The public branch record with these required checks."""
    record = copy.deepcopy(RECORDED_BRANCH)
    record["protection"]["required_status_checks"] = {
        "enforcement_level": level,
        "contexts": [check["context"] for check in checks],
        "checks": list(checks),
    }
    return record


VALIDATE = {"context": "validate / validate", "app_id": 15368}
GUARD = {"context": admission.GUARD_CHECK_NAME, "app_id": APP}


@pytest.fixture
def on(monkeypatch):
    """All three locks open. Tests only."""
    monkeypatch.setattr(admission, "V6_WRITER_ENABLED", True)
    monkeypatch.setattr(admission, "GUARD_APP_ID", APP)


@pytest.fixture
def world(api):  # noqa: F811
    """The reviewed-lane fixture, with no review, an enrolled dispatcher by
    numeric id, and the guard required on main."""
    api[REVIEWS] = []
    api[RUN]["triggering_actor"] = dict(PAVEL)
    api[BRANCH] = required(VALIDATE, GUARD)
    return api


class Recorder:
    def __init__(self, api):  # noqa: F811
        self.api, self.calls = api, []

    def __call__(self, path):
        self.calls.append(path)
        value = self.api[path]
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)


def outcome(function, *args, **kwargs):
    try:
        return ("ok", function(*args, **kwargs))
    except Exception as exc:  # the comparison includes the error text
        return (type(exc).__name__, str(exc))


# --- lock 1: the switch -------------------------------------------------------


def test_lane_is_off_by_default_and_has_no_runtime_override(monkeypatch):
    assert admission.V6_WRITER_ENABLED is False
    assert admission.GUARD_APP_ID is None
    assert admission.candidate_lane_enabled() is False
    source = Path(admission.__file__).read_text()
    assert "os.environ" not in source and "getenv" not in source
    assert "import os" not in source and "argparse" not in source
    # Every read of the switch compares with ``is True``.
    reads = [
        line.strip()
        for line in source.splitlines()
        if "V6_WRITER_ENABLED" in line and not line.lstrip().startswith("#")
    ]
    assert len(reads) == 6
    assert reads[0] == "V6_WRITER_ENABLED: Final[bool] = False"
    for line in reads[1:]:
        assert "V6_WRITER_ENABLED is True" in line or (
            "V6_WRITER_ENABLED is not True" in line
        ), line
    monkeypatch.setenv("AXIOM_ENCODE_V6_WRITER", "1")
    monkeypatch.setenv("V6_WRITER_ENABLED", "true")
    assert admission.candidate_lane_enabled() is False


@pytest.mark.parametrize("value", [1, "true", "True", [True], object()])
def test_only_an_exact_true_switch_enables(monkeypatch, world, value):
    monkeypatch.setattr(admission, "V6_WRITER_ENABLED", value)
    monkeypatch.setattr(admission, "GUARD_APP_ID", APP)
    assert admission.candidate_lane_enabled() is False
    assert admission.is_candidate_admission({"schema": admission.CANDIDATE_SCHEMA}) is (
        False
    )
    with pytest.raises(AdmissionError, match="exact-head signing-owner approval"):
        admission.verify_reviewed_pr(SELECTOR, fetch=world.__getitem__)
    with pytest.raises(ValueError, match="switched off"):
        v6.require_candidate_writer()


def test_off_a_candidate_without_approval_is_refused_as_today(world):
    fetch = Recorder(world)
    with pytest.raises(AdmissionError, match="exact-head signing-owner approval"):
        admission.verify_reviewed_pr(SELECTOR, fetch=fetch)
    assert BRANCH not in fetch.calls


def test_off_the_reviewed_admission_is_unchanged_and_never_reads_protection(api):  # noqa: F811
    fetch = Recorder(api)
    result = admission.verify_reviewed_pr(SELECTOR, fetch=fetch)
    assert set(result) == REVIEWED_KEYS
    assert result["schema"] == admission.SCHEMA
    assert result["approvals"] == [{"reviewer_id": 6076111, "review_id": 1}]
    assert "approval" in result["record"]["purpose"]
    assert BRANCH not in fetch.calls
    again = Recorder(api)
    api[RUN]["triggering_actor"] = {"login": "PavelMakarchuk"}
    assert admission.revalidate_admission(result, fetch=again) == result
    assert BRANCH not in again.calls


def test_off_a_candidate_admission_record_is_an_invalid_admission(
    world, on, monkeypatch
):
    record = admission.verify_reviewed_pr(SELECTOR, fetch=world.__getitem__)
    monkeypatch.setattr(admission, "V6_WRITER_ENABLED", False)
    fetch = Recorder(world)
    with pytest.raises(AdmissionError, match="^invalid provisioned admission$"):
        admission.revalidate_admission(record, fetch=fetch)
    assert fetch.calls == []
    assert admission.is_candidate_admission(record) is False
    with pytest.raises(AdmissionError, match="^invalid provisioned admission$"):
        admission._revalidate_candidate_admission(record, fetch=fetch)
    assert fetch.calls == []


# --- lock 2: the App id; lock 3: the required check ---------------------------


def test_switch_on_alone_admits_nothing_while_the_app_is_not_recorded(
    monkeypatch, world
):
    monkeypatch.setattr(admission, "V6_WRITER_ENABLED", True)
    fetch = Recorder(world)
    with pytest.raises(AdmissionError, match="GUARD_APP_ID"):
        admission.verify_reviewed_pr(SELECTOR, fetch=fetch)
    # Refused after the run read and before any candidate or protection read.
    assert fetch.calls == [RUN]


@pytest.mark.parametrize("app_id", [None, 0, -1, True, "987654", 15368, 987654.0])
def test_guard_app_id_must_be_a_positive_integer_that_is_not_actions(
    monkeypatch, world, app_id
):
    monkeypatch.setattr(admission, "V6_WRITER_ENABLED", True)
    monkeypatch.setattr(admission, "GUARD_APP_ID", app_id)
    world[BRANCH] = required(
        VALIDATE, {"context": admission.GUARD_CHECK_NAME, "app_id": app_id}
    )
    fetch = Recorder(world)
    with pytest.raises(AdmissionError, match="GUARD_APP_ID"):
        admission.require_guard_enforced(SELECTOR.repository, fetch=fetch)
    assert fetch.calls == []


def test_recorded_protection_of_rulespec_us_does_not_satisfy_the_interlock(on, world):
    # The state read live on 2026-11-13: only validate / validate is required.
    world[BRANCH] = copy.deepcopy(RECORDED_BRANCH)
    assert RECORDED_BRANCH["protection"]["required_status_checks"]["checks"] == [
        VALIDATE
    ]
    with pytest.raises(AdmissionError, match="not a required check"):
        admission.verify_reviewed_pr(SELECTOR, fetch=world.__getitem__)


def test_interlock_passes_only_for_the_guard_check_from_the_guard_app(on, world):
    fetch = Recorder(world)
    assert admission.require_guard_enforced(SELECTOR.repository, fetch=fetch) == {
        "check": "content-bound-merge-guard",
        "app_id": APP,
    }
    assert fetch.calls == [BRANCH]


def _drop(*keys):
    def change(record):
        target = record
        for key in keys[:-1]:
            target = target[key]
        del target[keys[-1]]

    return change


def _set(value, *keys):
    def change(record):
        target = record
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value

    return change


CHECKS = ("protection", "required_status_checks", "checks")
SHADOW = {"context": "content-bound-merge-guard (shadow)", "app_id": APP}


@pytest.mark.parametrize(
    "change",
    [
        _set([VALIDATE], *CHECKS),
        _set([], *CHECKS),
        _set(None, *CHECKS),
        _drop(*CHECKS),
        _set([VALIDATE, {"context": GUARD["context"], "app_id": 15368}], *CHECKS),
        _set([VALIDATE, {"context": GUARD["context"], "app_id": APP + 1}], *CHECKS),
        _set([VALIDATE, {"context": GUARD["context"], "app_id": None}], *CHECKS),
        _set([VALIDATE, {"context": GUARD["context"], "app_id": str(APP)}], *CHECKS),
        _set([VALIDATE, {"context": GUARD["context"], "app_id": float(APP)}], *CHECKS),
        _set([VALIDATE, {"context": GUARD["context"]}], *CHECKS),
        _set([VALIDATE, GUARD, GUARD], *CHECKS),
        _set(
            [VALIDATE, GUARD, {"context": GUARD["context"], "app_id": 15368}], *CHECKS
        ),
        _set([VALIDATE, SHADOW], *CHECKS),
        _set([VALIDATE, {"context": GUARD["context"] + " ", "app_id": APP}], *CHECKS),
        _set([VALIDATE, {"context": GUARD["context"].upper(), "app_id": APP}], *CHECKS),
        _set([VALIDATE, GUARD, "x"], *CHECKS),
        _set("non_admins", "protection", "required_status_checks", "enforcement_level"),
        _set("off", "protection", "required_status_checks", "enforcement_level"),
        _drop("protection", "required_status_checks", "enforcement_level"),
        _set(None, "protection", "required_status_checks"),
        _drop("protection", "required_status_checks"),
        _set(False, "protection", "enabled"),
        _set("true", "protection", "enabled"),
        _drop("protection"),
        _set(False, "protected"),
        _set(1, "protected"),
        _drop("protected"),
        _set("master", "name"),
        _drop("name"),
    ],
)
def test_interlock_refuses_every_other_protection_state(on, world, change):
    change(world[BRANCH])
    with pytest.raises(AdmissionError):
        admission.require_guard_enforced(SELECTOR.repository, fetch=world.__getitem__)
    with pytest.raises(AdmissionError):
        admission.verify_reviewed_pr(SELECTOR, fetch=world.__getitem__)


def test_a_check_named_only_in_contexts_does_not_count(on, world):
    world[BRANCH]["protection"]["required_status_checks"]["checks"] = [VALIDATE]
    assert (
        GUARD["context"]
        in (world[BRANCH]["protection"]["required_status_checks"]["contexts"])
    )
    with pytest.raises(AdmissionError, match="not a required check"):
        admission.require_guard_enforced(SELECTOR.repository, fetch=world.__getitem__)


@pytest.mark.parametrize(
    "response",
    [
        urllib.error.HTTPError("u", 403, "rate limited", {}, None),
        urllib.error.HTTPError("u", 404, "Not Found", {}, None),
        urllib.error.HTTPError("u", 500, "error", {}, None),
        urllib.error.URLError("unreachable"),
        TimeoutError(),
        ValueError("bad JSON"),
        AdmissionError("GitHub admission requests must not redirect"),
        None,
        [],
        "main",
        {"message": "Not Found"},
    ],
)
def test_unreadable_protection_refuses_and_is_never_read_as_absent(on, world, response):
    world[BRANCH] = response
    with pytest.raises(AdmissionError):
        admission.require_guard_enforced(SELECTOR.repository, fetch=Recorder(world))
    with pytest.raises(AdmissionError):
        admission.verify_reviewed_pr(SELECTOR, fetch=Recorder(world))


def test_missing_protection_response_refuses(on, world):
    del world[BRANCH]
    with pytest.raises(AdmissionError, match="cannot be read"):
        admission.verify_reviewed_pr(SELECTOR, fetch=world.__getitem__)


@pytest.mark.parametrize(
    "repository",
    [
        "TheAxiomFoundation/rulespec-us-tn",
        "TheAxiomFoundation/axiom-encode",
        "other/rulespec-us",
        "rulespec-us",
        None,
    ],
)
def test_interlock_refuses_a_repository_outside_the_promotion_lane(on, repository):
    def fetch(path):
        raise AssertionError("no read for an invalid repository")

    with pytest.raises(AdmissionError, match="invalid repository"):
        admission.require_guard_enforced(repository, fetch=fetch)


# --- on: admission without the approval ---------------------------------------


def test_on_a_candidate_without_any_review_is_admitted_as_unreviewed(on, world):
    fetch = Recorder(world)
    result = admission.verify_reviewed_pr(SELECTOR, fetch=fetch)
    assert set(result) == admission.CANDIDATE_FIELDS
    assert "approvals" not in result
    assert result["schema"] == admission.CANDIDATE_SCHEMA != admission.SCHEMA
    assert result["owner_review"] == "none"
    assert result["dispatcher"] == {"id": 110687043, "login": "PavelMakarchuk"}
    assert result["candidate"] == {"author_id": 110687043}
    assert result["guard"] == {"check": admission.GUARD_CHECK_NAME, "app_id": APP}
    assert result["workflow_commit"] == "e" * 40
    assert result["record"]["base_sha"] == BASE
    assert "NO owner approval" in result["record"]["purpose"]
    # The requirement is read before the candidate and again at the end.
    assert fetch.calls[:2] == [RUN, BRANCH]
    assert fetch.calls[-1] == BRANCH
    assert admission.is_candidate_admission(result) is True


def test_on_the_candidate_record_is_a_valid_admission_record(on, world):
    from axiom_encode.prepare_signed_backfill import _reviewed_rulespec_record_issue

    result = admission.verify_reviewed_pr(SELECTOR, fetch=world.__getitem__)
    assert (
        _reviewed_rulespec_record_issue(
            result["record"], country="us", rulespec_ref=HEAD
        )
        is None
    )


def test_on_an_existing_approval_does_not_make_the_admission_reviewed(on, api):  # noqa: F811
    api[RUN]["triggering_actor"] = dict(PAVEL)
    api[BRANCH] = required(VALIDATE, GUARD)
    result = admission.verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    assert result["schema"] == admission.CANDIDATE_SCHEMA
    assert "approvals" not in result


@pytest.mark.parametrize(
    "reviewer", [MAX, {"id": 555, "login": "outside", "type": "User"}]
)
def test_on_changes_requested_still_blocks(on, world, reviewer):
    world[REVIEWS] = [{"id": 5, "state": "CHANGES_REQUESTED", "user": reviewer}]
    with pytest.raises(AdmissionError, match="unresolved changes requested"):
        admission.verify_reviewed_pr(SELECTOR, fetch=world.__getitem__)


def test_on_superseded_changes_requested_and_comments_do_not_block(on, world):
    world[REVIEWS] = [
        {"id": 5, "state": "CHANGES_REQUESTED", "user": MAX},
        {"id": 6, "state": "DISMISSED", "user": MAX},
        {"id": 7, "state": "COMMENTED", "user": MAX},
    ]
    assert admission.verify_reviewed_pr(SELECTOR, fetch=world.__getitem__)


@pytest.mark.parametrize(
    "review",
    [
        {"id": 5, "state": "SOMETHING_NEW", "user": MAX},
        {"id": 5, "state": "APPROVED", "user": None},
        {"id": 5, "state": "APPROVED"},
    ],
)
def test_on_unreadable_reviews_refuse(on, world, review):
    world[REVIEWS] = [review]
    with pytest.raises(AdmissionError):
        admission.verify_reviewed_pr(SELECTOR, fetch=world.__getitem__)


@pytest.mark.parametrize(
    "actor",
    [
        {"login": "PavelMakarchuk"},
        {"id": 555, "login": "PavelMakarchuk", "type": "User"},
        {"id": 6076111, "login": "PavelMakarchuk", "type": "User"},
        {"id": 110687043, "login": "PavelMakarchuk", "type": "Bot"},
        {"id": 110687043, "login": "PavelMakarchuk"},
        {"id": "110687043", "login": "PavelMakarchuk", "type": "User"},
        {"id": True, "login": "PavelMakarchuk", "type": "User"},
    ],
)
def test_on_the_dispatcher_must_be_enrolled_by_numeric_id(on, world, actor):
    world[RUN]["triggering_actor"] = actor
    fetch = Recorder(world)
    with pytest.raises(AdmissionError, match="not an enrolled identity"):
        admission.verify_reviewed_pr(SELECTOR, fetch=fetch)
    assert fetch.calls == [RUN]


def test_on_a_write_access_account_without_enrolment_cannot_dispatch(on, world):
    from dataclasses import replace

    world[RUN]["triggering_actor"] = {
        "id": 555,
        "login": "collaborator",
        "type": "User",
    }
    selector = replace(SELECTOR, actor="collaborator")
    with pytest.raises(AdmissionError, match="not an enrolled identity"):
        admission.verify_reviewed_pr(selector, fetch=world.__getitem__)


def _extra_file(world):
    extra = {"filename": ".github/workflows/x.yml", "status": "added"}
    world[PR + "/files?per_page=100&page=1"].append(extra)
    world[PR]["changed_files"] = 3


def _fork(world):
    world[PR]["head"]["repo"]["full_name"] = "attacker/rulespec-us"


def _moved_main(world):
    world[ROOT + "/git/ref/heads/main"] = {"object": {"sha": "9" * 40}}


def _truncated(world):
    world[ROOT + f"/git/trees/{HEAD}?recursive=1"]["truncated"] = True


def _closed(world):
    world[PR]["state"] = "closed"


def _run_done(world):
    world[RUN]["status"] = "completed"


def _other_workflow(world):
    world[RUN]["path"] = ".github/workflows/ci.yml"


def _not_main(world):
    world[RUN]["head_branch"] = "feature"


@pytest.mark.parametrize(
    "mutate",
    [
        _extra_file,
        _fork,
        _moved_main,
        _truncated,
        _closed,
        _run_done,
        _other_workflow,
        _not_main,
    ],
)
def test_scope_and_run_rules_are_the_same_in_both_lanes(monkeypatch, api, mutate):  # noqa: F811
    api[RUN]["triggering_actor"] = dict(PAVEL)
    api[BRANCH] = required(VALIDATE, GUARD)
    mutate(api)
    off = outcome(admission.verify_reviewed_pr, SELECTOR, fetch=api.__getitem__)
    monkeypatch.setattr(admission, "V6_WRITER_ENABLED", True)
    monkeypatch.setattr(admission, "GUARD_APP_ID", APP)
    on_state = outcome(admission.verify_reviewed_pr, SELECTOR, fetch=api.__getitem__)
    assert off[0] == "AdmissionError"
    assert on_state == off


def test_on_a_requirement_removed_during_admission_refuses(on, world):
    reads = []
    enforced = world[BRANCH]

    class Api(dict):
        def __getitem__(self, path):
            if path == BRANCH:
                reads.append(path)
                return enforced if len(reads) == 1 else required(VALIDATE)
            return dict.__getitem__(self, path)

    with pytest.raises(AdmissionError):
        admission.verify_reviewed_pr(SELECTOR, fetch=Api(world).__getitem__)
    assert len(reads) == 2


# --- on: revalidation before validation, signing and publication --------------


@pytest.fixture
def admitted(on, world):
    return admission.verify_reviewed_pr(SELECTOR, fetch=world.__getitem__)


def test_on_revalidation_returns_the_same_record(admitted, world):
    fetch = Recorder(world)
    assert admission.revalidate_admission(admitted, fetch=fetch) == admitted
    assert BRANCH in fetch.calls


def _changes_requested(world):
    world[REVIEWS] = [{"id": 9, "state": "CHANGES_REQUESTED", "user": MAX}]


def _guard_dropped(world):
    world[BRANCH] = required(VALIDATE)


def _guard_not_for_admins(world):
    world[BRANCH] = required(VALIDATE, GUARD, level="non_admins")


def _actor_changed(world):
    world[RUN]["triggering_actor"] = dict(MAX)


def _actor_unenrolled(world):
    world[RUN]["triggering_actor"] = {"id": 555, "login": "PavelMakarchuk"}


def _run_commit(world):
    world[RUN]["head_sha"] = "f" * 40


def _author_changed(world):
    world[PR]["user"] = {"id": 6076111}


def _head_moved(world):
    world[PR]["head"]["sha"] = "f" * 40


def _protection_unreadable(world):
    world[BRANCH] = urllib.error.HTTPError("u", 403, "rate limited", {}, None)


@pytest.mark.parametrize(
    "mutate",
    [
        _changes_requested,
        _guard_dropped,
        _guard_not_for_admins,
        _actor_changed,
        _actor_unenrolled,
        _run_done,
        _run_commit,
        _moved_main,
        _author_changed,
        _head_moved,
        _closed,
        _protection_unreadable,
    ],
)
def test_on_revalidation_refuses_any_live_change(admitted, world, mutate):
    mutate(world)
    with pytest.raises(AdmissionError):
        admission.revalidate_admission(admitted, fetch=Recorder(world))


def _with(key, value):
    def change(record):
        record[key] = value

    return change


def _without(key):
    def change(record):
        del record[key]

    return change


def _candidate_extra(record):
    record["candidate"]["approved"] = True


def _other_app(record):
    record["guard"]["app_id"] = 15368


def _other_dispatcher(record):
    record["dispatcher"] = {"id": 6076111, "login": "MaxGhenis"}


def _other_base(record):
    record["record"]["base_sha"] = "9" * 40


def _other_author(record):
    record["candidate"]["author_id"] = 6076111


@pytest.mark.parametrize(
    "change",
    [
        _with("approvals", [{"reviewer_id": 6076111, "review_id": 1}]),
        _with("owner_review", "before-signing"),
        _with("owner_review", None),
        _with("extra", 1),
        _without("guard"),
        _without("dispatcher"),
        _without("candidate"),
        _without("owner_review"),
        _with("candidate", None),
        _with("record", None),
        _with("workflow_commit", "f" * 40),
        _candidate_extra,
        _other_app,
        _other_dispatcher,
        _other_base,
        _other_author,
    ],
)
def test_on_revalidation_refuses_an_edited_record(admitted, world, change):
    record = copy.deepcopy(admitted)
    change(record)
    with pytest.raises(AdmissionError):
        admission.revalidate_admission(record, fetch=Recorder(world))


def test_on_a_reviewed_admission_record_is_still_revalidated_as_reviewed(
    monkeypatch,
    api,  # noqa: F811
):
    reviewed = admission.verify_reviewed_pr(SELECTOR, fetch=api.__getitem__)
    monkeypatch.setattr(admission, "V6_WRITER_ENABLED", True)
    monkeypatch.setattr(admission, "GUARD_APP_ID", APP)
    fetch = Recorder(api)
    assert admission.revalidate_admission(reviewed, fetch=fetch) == reviewed
    assert admission.is_candidate_admission(reviewed) is False
    assert BRANCH not in fetch.calls
    # A reviewed record cannot be relabelled as a candidate, or the reverse.
    relabelled = reviewed | {"schema": admission.CANDIDATE_SCHEMA}
    with pytest.raises(AdmissionError, match="invalid provisioned admission"):
        admission.revalidate_admission(relabelled, fetch=fetch)


@pytest.mark.parametrize("payload", [None, [], "x", {}, {"schema": None}])
def test_is_candidate_admission_refuses_non_records(on, payload):
    assert admission.is_candidate_admission(payload) is False


def test_admission_module_still_loads_standalone_with_the_standard_library_only():
    # The root provisioner loads this one file from the attested snapshot.
    import ast

    tree = ast.parse(Path(admission.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative import"
            imported.add((node.module or "").split(".")[0])
    assert imported <= {
        "__future__",
        "base64",
        "hashlib",
        "json",
        "re",
        "urllib",
        "dataclasses",
        "typing",
    }


# --- the writer: one definition with the reader --------------------------------

SCOPE_ARGUMENTS = {
    "v5_ref_field": "reviewed_rulespec_ref",
    "repository": "TheAxiomFoundation/rulespec-us",
    "pull_request": 2044,
    "base_sha": "a" * 40,
    "author_id": 110687043,
    "dispatcher": {"id": 110687043, "login": "PavelMakarchuk"},
}


@pytest.fixture
def writable(on, monkeypatch):
    monkeypatch.setattr(v6, "V6_READER_ENABLED", True)


@pytest.fixture
def promotion(manifest_fixture, graph):  # noqa: F811
    """An unsigned v5 promotion payload the live verifier admits."""
    _fixtures, repo, _rule, payload = manifest_fixture
    _set_producer(payload, graph.commits[1])
    v5 = _reviewed_promotion(payload)
    v5.pop("signature", None)
    return repo, v5, _expected(graph)


def test_writer_needs_both_switches(monkeypatch, promotion):
    _repo, v5, _identity = promotion
    refusals = []
    for writer in (False, True):
        for reader in (False, True):
            monkeypatch.setattr(admission, "V6_WRITER_ENABLED", writer)
            monkeypatch.setattr(v6, "V6_READER_ENABLED", reader)
            result = outcome(v6.as_v6_candidate_payload, v5, **SCOPE_ARGUMENTS)
            refusals.append((writer, reader, result[0]))
            assert outcome(v6.require_candidate_writer)[0] == result[0] or (
                writer and reader
            )
    assert refusals == [
        (False, False, "ValueError"),
        (False, True, "ValueError"),
        (True, False, "ValueError"),
        (True, True, "ok"),
    ]


@pytest.mark.parametrize("value", [1, "true"])
def test_writer_switches_need_an_exact_true(monkeypatch, promotion, value):
    monkeypatch.setattr(admission, "V6_WRITER_ENABLED", value)
    monkeypatch.setattr(v6, "V6_READER_ENABLED", True)
    with pytest.raises(ValueError, match="writer is switched off"):
        v6.require_candidate_writer()
    monkeypatch.setattr(admission, "V6_WRITER_ENABLED", True)
    monkeypatch.setattr(v6, "V6_READER_ENABLED", value)
    with pytest.raises(ValueError, match="reader is switched off"):
        v6.require_candidate_writer()


def test_writer_emits_exactly_the_shape_the_reader_contract_defines(
    writable, promotion
):
    _repo, v5, _identity = promotion
    written = v6.as_v6_candidate_payload(v5, **SCOPE_ARGUMENTS)
    # The reader's own test builder, from the reader change: same bytes.
    assert written == _to_v6(v5)
    assert json.dumps(written, sort_keys=True) == json.dumps(_to_v6(v5), sort_keys=True)
    scope = written["attestation_scope"]
    # Writer to reader and reader to writer: the field sets are the reader's.
    assert set(scope) == v6.ATTESTATION_SCOPE_FIELDS
    assert set(scope["candidate"]) == v6.ATTESTATION_CANDIDATE_FIELDS
    assert set(scope["dispatcher"]) == v6.ATTESTATION_DISPATCHER_FIELDS
    signed = _sign(copy.deepcopy(written))
    assert frozenset(signed) in cli._REVIEWED_CANDIDATE_V6_MANIFEST_VARIANTS
    assert frozenset(signed) in guard.V6_MANIFEST_FIELD_VARIANTS
    assert (
        v6.contract_issues(
            signed,
            manifest_label="m",
            field_variants=cli._REVIEWED_CANDIDATE_V6_MANIFEST_VARIANTS,
        )
        == []
    )
    assert written["schema_version"] == v6.MANIFEST_SCHEMA_V6
    assert "reviewed_rulespec_ref" not in written
    assert written["candidate_rulespec_ref"] == v5["reviewed_rulespec_ref"]
    assert v6.owner_review_state(written) == "none"
    # The input is not changed.
    assert "reviewed_rulespec_ref" in v5 and "attestation_scope" not in v5


def test_every_reader_scope_field_is_required_of_the_writer_output(writable, promotion):
    _repo, v5, _identity = promotion
    written = v6.as_v6_candidate_payload(v5, **SCOPE_ARGUMENTS)
    for name in v6.ATTESTATION_SCOPE_FIELDS:
        broken = copy.deepcopy(written)
        del broken["attestation_scope"][name]
        assert v6.attestation_scope_issues(broken, manifest_label="m")
    for group, names in (
        ("candidate", v6.ATTESTATION_CANDIDATE_FIELDS),
        ("dispatcher", v6.ATTESTATION_DISPATCHER_FIELDS),
    ):
        for name in names:
            broken = copy.deepcopy(written)
            del broken["attestation_scope"][group][name]
            assert v6.attestation_scope_issues(broken, manifest_label="m")


def test_written_manifest_round_trips_through_the_reader(writable, promotion):
    repo, v5, expected = promotion
    written = _sign(v6.as_v6_candidate_payload(v5, **SCOPE_ARGUMENTS))
    verified, _root, _digest, issues = _load(repo, written, expected)
    assert issues == []
    assert verified == written


def test_written_manifest_is_refused_by_the_reader_while_the_reader_is_off(
    writable, promotion, monkeypatch
):
    repo, v5, expected = promotion
    written = _sign(v6.as_v6_candidate_payload(v5, **SCOPE_ARGUMENTS))
    monkeypatch.setattr(v6, "V6_READER_ENABLED", False)
    verified, _root, _digest, issues = _load(repo, written, expected)
    assert verified is None and issues


def test_written_scope_is_one_the_merge_guard_parses(writable, promotion):
    _repo, v5, _identity = promotion
    written = _sign(v6.as_v6_candidate_payload(v5, **SCOPE_ARGUMENTS))
    written["applied_files"] = [
        {"path": "us/statutes/26/1.yaml", "sha256": "0" * 64},
        {"path": "us/statutes/26/1.test.yaml", "sha256": "1" * 64},
    ]
    written["run_id"] = "12345"
    parsed = guard._parse_manifest(
        written,
        repository="TheAxiomFoundation/rulespec-us",
        number=2044,
        author_id=110687043,
    )
    assert (parsed.dispatcher_id, parsed.dispatcher_login) == (
        110687043,
        "PavelMakarchuk",
    )
    assert parsed.candidate_base == "a" * 40


@pytest.mark.parametrize(
    "change",
    [
        {"repository": "TheAxiomFoundation/rulespec-uk"},
        {"repository": "TheAxiomFoundation/rulespec-us-tn"},
        {"repository": "rulespec-us"},
        {"pull_request": 0},
        {"pull_request": True},
        {"pull_request": "2044"},
        {"base_sha": "A" * 40},
        {"base_sha": "a" * 39},
        {"base_sha": None},
        {"author_id": 0},
        {"author_id": None},
        {"author_id": "110687043"},
        {"dispatcher": {"id": 0, "login": "PavelMakarchuk"}},
        {"dispatcher": {"id": 110687043}},
        {"dispatcher": {"id": 110687043, "login": "bad login"}},
        {"dispatcher": None},
        {"dispatcher": ["id", "login"]},
        {"v5_ref_field": "missing_field"},
    ],
)
def test_writer_refuses_a_scope_the_reader_would_refuse(writable, promotion, change):
    _repo, v5, _identity = promotion
    with pytest.raises(ValueError):
        v6.as_v6_candidate_payload(v5, **(SCOPE_ARGUMENTS | change))


def test_writer_keeps_only_the_named_dispatcher_fields(writable, promotion):
    _repo, v5, _identity = promotion
    dispatcher = {"id": 110687043, "login": "PavelMakarchuk", "type": "User"}
    written = v6.as_v6_candidate_payload(
        v5, **(SCOPE_ARGUMENTS | {"dispatcher": dispatcher})
    )
    assert written["attestation_scope"]["dispatcher"] == SCOPE_ARGUMENTS["dispatcher"]


def _signed(payload):
    payload["signature"] = {"algorithm": "x", "key_id": "y", "value": "z"}


def _model_tool(payload):
    payload["tool"] = cli.APPLIED_ENCODING_MODEL_TOOL


def _backend(payload):
    payload["backend"] = "openai"


def _already_v6(payload):
    payload["schema_version"] = v6.MANIFEST_SCHEMA_V6


def _has_candidate_ref(payload):
    payload["candidate_rulespec_ref"] = "d" * 40


def _has_scope(payload):
    payload["attestation_scope"] = {}


def _bad_ref(payload):
    payload["reviewed_rulespec_ref"] = "main"


def _other_root(payload):
    payload["source_attestation"] = dict(
        payload["source_attestation"], rulespec_root="rulespec-uk/uk"
    )


@pytest.mark.parametrize(
    "change",
    [
        _signed,
        _model_tool,
        _backend,
        _already_v6,
        _has_candidate_ref,
        _has_scope,
        _bad_ref,
        _other_root,
    ],
)
def test_writer_relabels_only_an_unsigned_promotion_payload(
    writable, promotion, change
):
    _repo, v5, _identity = promotion
    payload = copy.deepcopy(v5)
    change(payload)
    with pytest.raises(ValueError):
        v6.as_v6_candidate_payload(payload, **SCOPE_ARGUMENTS)


def test_the_writer_is_reached_from_one_place_behind_the_candidate_lane():
    source = Path(cli.__file__).read_text()
    assert source.count("_manifest_v6.as_v6_candidate_payload(") == 1
    assert source.count("_manifest_v6.require_candidate_writer()") == 1
    assert source.count("is_candidate_admission(") == 1
    before, after = source.split("_manifest_v6.as_v6_candidate_payload(")
    assert before.rstrip().endswith("payload =")
    assert "if candidate_lane:" in before[-400:]
    # The lane flag has one assignment after its ``False`` default.
    assert source.count("candidate_lane = ") == 2
    assert "candidate_lane = False" in source
    assert "candidate_lane = is_candidate_admission(live_admission)" in source
    # No other module writes v6.
    root = Path(cli.__file__).parent
    writers = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "as_v6_candidate_payload" in path.read_text()
    )
    assert writers == ["applied_manifest_v6.py", "cli.py"]


def test_the_switch_rule_is_stated_at_the_writer_switch():
    rule = (
        "The reader may be switched on while no v6 writer exists: nothing new can "
        "land. The v6 writer and the admission without approval must stay off "
        "until the guard check is required."
    )
    flat = " ".join(Path(admission.__file__).read_text().replace("#", " ").split())
    assert rule in flat
    assert "OPERATOR PRECONDITION" in flat


# --- the promotion command ------------------------------------------------------


def _promote(tmp_path, monkeypatch, *, live, per_file=False, revalidate=None):
    """Run ``promote-reviewed-candidate`` with ``live`` as the provisioned
    admission. Returns (manifest path, signer context mock, revalidate mock)."""

    repo, rule, companion, ref = _reviewed_candidate_checkout(tmp_path)
    binding = _reviewed_candidate_binding(rule, companion)
    record_root = _write_reviewed_rulespec_records(
        tmp_path / "reviewed-rulespec-refs", ("us", ref, binding)
    )
    record_file = next(record_root.rglob("*.json"))
    record = json.loads(record_file.read_text())
    record_file.unlink()
    args = SimpleNamespace(
        repo=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        corpus_path=tmp_path / "axiom-corpus",
        rulespec_dependency_root=[],
        rulespec_path="us/statutes/7/2015/f.yaml",
        citation="us/statute/7/2015/f",
        reviewed_rulespec_ref=ref,
        run_id="12345",
    )
    args.axiom_rules_path.mkdir()
    args.corpus_path.mkdir()
    selector = {
        "country": "us",
        "pr": 42,
        "head": ref,
        "citation": args.citation,
        "rulespec_path": args.rulespec_path,
        "branch": "codex/candidate",
        "actor": "PavelMakarchuk",
        "run_id": args.run_id,
    }
    attestation = tmp_path / "runtime-attestation.json"
    attestation.write_text(
        json.dumps({"reviewed_pr_admission": live(selector, record)})
    )
    monkeypatch.setenv(_TRUSTED_RUNTIME_ENV, "1")
    execution = {"schema": "axiom-encode/apply-validation-execution/v2"}

    def validate(result, **_kwargs):
        setattr(
            result,
            _APPLY_VALIDATION_SNAPSHOT_ATTR,
            {"manifest_validation_execution": execution},
        )
        return True, [], {}

    identity = {
        "repository": APPLIED_ENCODING_OFFICIAL_REPOSITORY,
        "commit": "b" * 40,
        "version": AXIOM_ENCODE_TEST_VERSION,
        "identity_source": "trusted-runtime-attestation",
    }
    provenance = {
        "root": "/opt/axiom-verification",
        "commit": "b" * 40,
        "dirty_tracked": False,
        "version": AXIOM_ENCODE_TEST_VERSION,
        "version_commit": "b" * 40,
        "identity_source": "trusted-runtime-attestation",
    }
    bindings = {
        "schema": "axiom-encode/validation-waiver-bindings/v1",
        "files": [
            {"path": path, "fingerprint": None, "record_sha256": None}
            for path in ("us/statutes/7/2015/f.test.yaml", "us/statutes/7/2015/f.yaml")
        ],
    }
    manifest = repo / _applied_encoding_manifest_path(Path(args.rulespec_path))
    error = None
    with (
        patch(
            "axiom_encode.cli._load_trusted_runtime_attestation",
            return_value={"path": str(attestation), "commit": "e" * 40},
        ),
        patch(
            "axiom_encode.reviewed_pr_admission.revalidate_admission",
            **({} if revalidate is None else {"side_effect": revalidate}),
        ) as live_check,
        patch(
            "axiom_encode.cli._promotion_reviewed_rulespec_record_root",
            return_value=record_root,
        ),
        patch("axiom_encode.cli._recover_apply_transaction"),
        patch("axiom_encode.cli._isolated_apply_manifest_signer") as signer_context,
        patch(
            "axiom_encode.cli.load_rulespec_local_corpus_release",
            return_value=object(),
        ),
        patch(
            "axiom_encode.cli._manifest_primary_source_verifications",
            return_value=([], "rulespec-us/us"),
        ),
        patch(
            "axiom_encode.cli.resolve_corpus_source_unit",
            return_value=_manifest_refresh_source_unit(),
        ),
        patch(
            "axiom_encode.cli._run_generated_encoding_overlay_validation",
            side_effect=validate,
        ) as overlay,
        patch(
            "axiom_encode.cli._require_clean_axiom_encode_git_provenance",
            return_value=provenance,
        ),
        patch(
            "axiom_encode.cli.verify_rulespec_validation_waiver_set",
            return_value="c" * 64,
        ),
        patch(
            "axiom_encode.cli._target_uses_per_file_waiver_bindings",
            return_value=per_file,
        ),
        patch(
            "axiom_encode.cli._build_applied_manifest_file_waiver_bindings",
            return_value=bindings,
        ),
        patch(
            "axiom_encode.cli._current_guard_encoder_execution_identity",
            return_value=identity,
        ),
        patch(
            "axiom_encode.cli._load_verified_applied_encoding_manifest_payload",
            return_value=({}, "", "d" * 64, []),
        ),
    ):
        signer_context.return_value.__enter__.return_value = TEST_APPLY_SIGNING_BROKER
        try:
            cmd_promote_reviewed_candidate(args)
        except Exception as exc:
            error = exc
    return SimpleNamespace(
        manifest=manifest,
        signer=signer_context,
        overlay=overlay,
        live_check=live_check,
        error=error,
        ref=ref,
        record=record,
    )


def _candidate_record(selector, record):
    return {
        "schema": admission.CANDIDATE_SCHEMA,
        "workflow_commit": "e" * 40,
        "selector": selector,
        "owner_review": "none",
        # Not the dispatcher: the two identities must not be confused.
        "candidate": {"author_id": 555},
        "dispatcher": {"id": 110687043, "login": "PavelMakarchuk"},
        "guard": {"check": admission.GUARD_CHECK_NAME, "app_id": APP},
        "record": dict(record, base_sha="a" * 40, rulespec_pr=42),
    }


def _reviewed_record(selector, record):
    return {
        "schema": admission.SCHEMA,
        "workflow_commit": "e" * 40,
        "selector": selector,
        "approvals": [{"reviewer_id": 6076111, "review_id": 1}],
        "record": record,
    }


@pytest.mark.parametrize("per_file", [False, True])
def test_on_the_command_writes_a_signed_v6_manifest(
    tmp_path, monkeypatch, writable, per_file
):
    run = _promote(tmp_path, monkeypatch, live=_candidate_record, per_file=per_file)
    assert run.error is None
    # Before validation and again immediately before signing.
    assert run.live_check.call_count == 2
    payload = json.loads(run.manifest.read_text())
    assert payload["schema_version"] == v6.MANIFEST_SCHEMA_V6
    assert "reviewed_rulespec_ref" not in payload
    assert payload["candidate_rulespec_ref"] == run.ref
    assert payload["attestation_scope"] == {
        "schema": v6.ATTESTATION_SCOPE_SCHEMA,
        "validation": "encoder-main-complete-source",
        "owner_review": "none",
        "origin": "local-candidate",
        "candidate": {
            "repository": "TheAxiomFoundation/rulespec-us",
            "pull_request": 42,
            "head_sha": run.ref,
            "base_sha": "a" * 40,
            "author_id": 555,
        },
        "dispatcher": {"id": 110687043, "login": "PavelMakarchuk"},
        "merge_requirement": "axiom-encode/content-bound-merge-guard/v1",
    }
    unsigned = frozenset(payload) - {"signature"}
    variants = {
        fields - {"signature"}
        for fields in cli._REVIEWED_CANDIDATE_V6_MANIFEST_VARIANTS
    }
    assert unsigned in variants
    assert ("validation_waiver_bindings" in payload) is per_file
    assert (
        cli._applied_encoding_manifest_signature_issue(
            payload, TEST_APPLY_SIGNING_BROKER
        )
        is None
    )
    assert (
        v6.contract_issues(
            payload,
            manifest_label="m",
            field_variants=cli._REVIEWED_CANDIDATE_V6_MANIFEST_VARIANTS,
        )
        == []
    )
    parsed = guard._parse_manifest(
        payload,
        repository="TheAxiomFoundation/rulespec-us",
        number=42,
        author_id=555,
    )
    assert parsed.candidate_head == run.ref


def test_on_a_reviewed_admission_still_writes_v5(tmp_path, monkeypatch, writable):
    run = _promote(tmp_path, monkeypatch, live=_reviewed_record)
    assert run.error is None
    payload = json.loads(run.manifest.read_text())
    assert payload["schema_version"] == cli.APPLIED_ENCODING_MANIFEST_SCHEMA
    assert payload["reviewed_rulespec_ref"] == run.ref
    assert "attestation_scope" not in payload


def test_off_the_command_refuses_a_candidate_admission_before_any_work(
    tmp_path, monkeypatch
):
    # The real revalidation: with the switch off the record is invalid, and
    # no request is made.
    run = _promote(
        tmp_path,
        monkeypatch,
        live=_candidate_record,
        revalidate=admission.revalidate_admission,
    )
    assert isinstance(run.error, AdmissionError)
    assert str(run.error) == "invalid provisioned admission"
    run.signer.assert_not_called()
    run.overlay.assert_not_called()
    assert not run.manifest.exists()


def test_off_a_candidate_record_never_becomes_a_v6_manifest(tmp_path, monkeypatch):
    # Even if revalidation were bypassed, the switch decides the lane.
    run = _promote(tmp_path, monkeypatch, live=_candidate_record)
    assert run.error is None
    payload = json.loads(run.manifest.read_text())
    assert payload["schema_version"] == cli.APPLIED_ENCODING_MANIFEST_SCHEMA
    assert "attestation_scope" not in payload


def test_writer_on_and_reader_off_refuses_before_validation(tmp_path, monkeypatch, on):
    assert v6.V6_READER_ENABLED is False
    run = _promote(tmp_path, monkeypatch, live=_candidate_record)
    assert isinstance(run.error, ValueError)
    assert "reader is switched off" in str(run.error)
    run.signer.assert_not_called()
    run.overlay.assert_not_called()
    assert not run.manifest.exists()


def test_on_a_revoked_candidate_is_not_signed(tmp_path, monkeypatch, writable):
    calls = []

    def revalidate(_record):
        calls.append(1)
        if len(calls) == 2:
            raise AdmissionError("candidate has unresolved changes requested")

    run = _promote(tmp_path, monkeypatch, live=_candidate_record, revalidate=revalidate)
    assert isinstance(run.error, AdmissionError)
    assert len(calls) == 2
    assert not run.manifest.exists()


@pytest.mark.parametrize(
    "change",
    [
        lambda live: live["candidate"].update(author_id=0),
        lambda live: live["dispatcher"].update(login="bad login"),
        lambda live: live["record"].update(base_sha="not-a-sha"),
        lambda live: live.pop("dispatcher"),
        lambda live: live.pop("candidate"),
    ],
)
def test_on_an_unusable_candidate_record_is_not_signed(
    tmp_path, monkeypatch, writable, change
):
    def live(selector, record):
        result = _candidate_record(selector, record)
        change(result)
        return result

    run = _promote(tmp_path, monkeypatch, live=live)
    assert run.error is not None
    assert not run.manifest.exists()


def test_on_a_dispatcher_login_that_differs_from_the_selector_refuses(on, world):
    world[RUN]["triggering_actor"] = {
        "id": 110687043,
        "login": "pavelmakarchuk",
        "type": "User",
    }
    with pytest.raises(AdmissionError):
        admission.verify_reviewed_pr(SELECTOR, fetch=world.__getitem__)
    # The enrolment check itself also binds the selector's actor.
    with pytest.raises(AdmissionError, match="not an enrolled identity"):
        admission._enrolled_dispatcher(
            {"triggering_actor": dict(PAVEL)},
            SimpleNamespace(actor="MaxGhenis"),
        )
