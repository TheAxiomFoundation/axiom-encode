"""Change 5c of docs/content-bound-merge-guard.md (#2172): locked.

A second accepted form of locks 2 and 3 of the v6 writer: ``GUARD_WORKFLOW``
and a live read that an organization ruleset requires the guard workflow, at
a recorded commit, on ``main`` of the RuleSpec repository, with a merge queue.
Exactly one of ``GUARD_APP_ID`` and ``GUARD_WORKFLOW`` may be recorded. Both
are ``None`` on encoder main; on-state behaviour is reached in tests only, by
patching the constants.

The listing with the ``workflows`` rule is SYNTHETIC. [experiment X14]: it
must be replaced by the recorded live response before activation.
"""

from __future__ import annotations

import copy
import itertools
import json
import urllib.error
from pathlib import Path

import pytest

from axiom_encode import applied_manifest_v6 as v6
from axiom_encode import candidate_promotion_evidence as evidence
from axiom_encode import reviewed_pr_admission as admission
from axiom_encode.reviewed_pr_admission import AdmissionError, GuardWorkflow
from tests.test_candidate_pr_admission import (
    APP,
    BRANCH,
    GUARD,
    PAVEL,
    REVIEWS,
    RUN,
    VALIDATE,
    Recorder,
    required,
)
from tests.test_candidate_promotion_evidence import (
    CITATION,
    LIVE_PROMOTION,
    REF,
    written_v6,
)
from tests.test_reviewed_pr_admission import (  # noqa: F401 (fixture)
    ROOT,
    SELECTOR,
    api,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPOSITORY_ROOT / "tests/fixtures/merge_guard_requirement"
SYNTHETIC_FILE = FIXTURES / "rules_branches_main_workflows_SYNTHETIC.json"
SYNTHETIC = json.loads(SYNTHETIC_FILE.read_text("utf-8"))
RECORDED = json.loads((FIXTURES / "rules_branches_main_recorded.json").read_text())
X14 = (
    "[experiment X14]: must be replaced by the recorded live response before activation"
)

RULES_PATH = ROOT + "/rules/branches/main"
RULES = RULES_PATH + "?per_page=100&page=1"
RULES_2 = RULES_PATH + "?per_page=100&page=2"
REPOSITORY_ID = 1125775893
SHA_A = "a1" * 20
SHA_B = "b2" * 20
RULESET_ID = 4242
WORKFLOW = GuardWorkflow(
    repository_id=REPOSITORY_ID,
    path=admission.GUARD_WORKFLOW_PATH,
    shas=frozenset({SHA_A, SHA_B}),
)
ACCEPTED = {
    "form": "organization_ruleset_workflow",
    "ruleset_id": RULESET_ID,
    "repository_id": REPOSITORY_ID,
    "path": ".github/workflows/content-bound-merge-guard.yml",
    "sha": SHA_A,
}
FILLER = {
    "type": "deletion",
    "ruleset_source_type": "Repository",
    "ruleset_source": "TheAxiomFoundation/rulespec-us",
    "ruleset_id": 7,
}


def listing() -> list:
    """The SYNTHETIC listing: the workflows rule, then the merge queue."""

    return copy.deepcopy(SYNTHETIC["response"])


@pytest.fixture
def ruleset_on(monkeypatch):
    """Lock 1 open and the ruleset form of lock 2 recorded. Tests only."""

    monkeypatch.setattr(admission, "V6_WRITER_ENABLED", True)
    monkeypatch.setattr(admission, "GUARD_WORKFLOW", WORKFLOW)


@pytest.fixture
def world(api):  # noqa: F811
    """The reviewed-lane fixture with no review, an enrolled dispatcher, and
    the guard required in BOTH forms, so that a refusal never comes from a
    missing response."""

    api[REVIEWS] = []
    api[RUN]["triggering_actor"] = dict(PAVEL)
    api[BRANCH] = required(VALIDATE, GUARD)
    api[RULES] = listing()
    return api


def enforced(fetch):
    return admission.require_guard_enforced(SELECTOR.repository, fetch=fetch)


# --- the fixture is synthetic, and says so --------------------------------------


def test_the_workflows_listing_is_marked_synthetic_everywhere():
    assert SYNTHETIC["synthetic"] is True
    assert SYNTHETIC["note"].startswith("SYNTHETIC. Not a recorded response.")
    assert X14 in SYNTHETIC["note"]
    assert "SYNTHETIC" in SYNTHETIC_FILE.name

    def flat(path):
        text = (REPOSITORY_ROOT / path).read_text("utf-8").replace("#", " ")
        return " ".join(text.split())

    for path in (
        "src/axiom_encode/reviewed_pr_admission.py",
        "docs/content-bound-merge-guard.md",
        "tests/test_guard_ruleset_lock.py",
    ):
        text = flat(path)
        assert "SYNTHETIC" in text, path
        assert X14 in text, path


def test_the_synthetic_listing_uses_only_fields_of_the_rest_schema():
    workflows, queue = SYNTHETIC["response"]
    info = {"ruleset_source_type", "ruleset_source", "ruleset_id"}
    assert set(workflows) == {"type", "parameters"} | info
    assert set(workflows["parameters"]) == {"workflows"}
    (entry,) = workflows["parameters"]["workflows"]
    assert set(entry) == {"repository_id", "path", "sha"}
    assert set(entry) <= admission._WORKFLOW_ENTRY_FIELDS
    # The merge-queue rule is the recorded one of axiom-encode, with another
    # (invented) source and id.
    recorded = RECORDED["reads"][1]["response"][0]
    assert set(queue) == set(recorded)
    assert queue["type"] == recorded["type"] == "merge_queue"
    assert queue["parameters"] == recorded["parameters"]


# --- lock 2: exactly one form is recorded ---------------------------------------


def test_both_forms_are_committed_unrecorded_with_no_runtime_override():
    assert admission.V6_WRITER_ENABLED is False
    assert admission.GUARD_APP_ID is None
    assert admission.GUARD_WORKFLOW is None
    source = Path(admission.__file__).read_text()
    for forbidden in ("os.environ", "getenv", "import os", "argparse", "sys.argv"):
        assert forbidden not in source
    # One assignment each, at module level, to None.
    assert source.count("\nGUARD_WORKFLOW: Final[GuardWorkflow | None] = None\n") == 1
    assert source.count("GUARD_WORKFLOW =") == source.count("GUARD_WORKFLOW:") - 1 == 0
    assert source.count("\nGUARD_APP_ID: Final[int | None] = None\n") == 1


def test_committed_constants_refuse_before_any_read():
    fetch = Recorder({})
    with pytest.raises(AdmissionError) as refusal:
        enforced(fetch)
    assert "GUARD_APP_ID" in str(refusal.value)
    assert "GUARD_WORKFLOW" in str(refusal.value)
    assert fetch.calls == []
    with pytest.raises(ValueError, match="writer is switched off"):
        v6.require_candidate_writer()
    assert admission.candidate_lane_enabled() is False
    assert (
        admission.is_candidate_admission({"schema": admission.CANDIDATE_SCHEMA})
        is False
    )


def test_committed_constants_the_reviewed_lane_reads_no_guard_record(api):  # noqa: F811
    fetch = Recorder(api)
    record = admission.verify_reviewed_pr(SELECTOR, fetch=fetch)
    assert record["schema"] == admission.SCHEMA and "guard" not in record
    admission.revalidate_admission(record, fetch=fetch)
    assert not [path for path in fetch.calls if "/rules/" in path or path == BRANCH]


@pytest.mark.parametrize(
    ("app_id", "workflow", "expected", "reads"),
    [
        pytest.param(
            None, None, "neither GUARD_APP_ID nor GUARD_WORKFLOW", [], id="neither"
        ),
        pytest.param(
            APP,
            WORKFLOW,
            "both GUARD_APP_ID and GUARD_WORKFLOW are recorded",
            [],
            id="both",
        ),
        pytest.param(
            APP,
            None,
            {"check": "content-bound-merge-guard", "app_id": APP},
            [BRANCH],
            id="app",
        ),
        pytest.param(None, WORKFLOW, ACCEPTED, [RULES], id="workflow"),
    ],
)
def test_truth_table_of_the_two_forms(
    monkeypatch, world, app_id, workflow, expected, reads
):
    monkeypatch.setattr(admission, "V6_WRITER_ENABLED", True)
    monkeypatch.setattr(admission, "GUARD_APP_ID", app_id)
    monkeypatch.setattr(admission, "GUARD_WORKFLOW", workflow)
    fetch = Recorder(world)
    if isinstance(expected, dict):
        assert enforced(fetch) == expected
        assert fetch.calls == reads
        record = admission.verify_reviewed_pr(SELECTOR, fetch=Recorder(world))
        assert record["guard"] == expected
        assert record["schema"] == admission.CANDIDATE_SCHEMA
    else:
        with pytest.raises(AdmissionError, match=expected):
            enforced(fetch)
        assert fetch.calls == []
        with pytest.raises(AdmissionError, match=expected):
            admission.verify_reviewed_pr(SELECTOR, fetch=fetch)
        # Refused after the run read, before any candidate or guard read.
        assert fetch.calls == [RUN]


@pytest.mark.parametrize(
    ("app_id", "workflow"),
    [
        (0, WORKFLOW),
        (False, WORKFLOW),
        (15368, WORKFLOW),
        ("987654", WORKFLOW),
        (APP, {}),
        (APP, False),
        (APP, ()),
        (APP, "x"),
    ],
)
def test_any_two_recorded_values_are_ambiguous_also_malformed_ones(
    monkeypatch, world, app_id, workflow
):
    monkeypatch.setattr(admission, "GUARD_APP_ID", app_id)
    monkeypatch.setattr(admission, "GUARD_WORKFLOW", workflow)
    fetch = Recorder(world)
    with pytest.raises(AdmissionError, match="^ambiguous merge guard configuration"):
        enforced(fetch)
    assert fetch.calls == []


@pytest.mark.parametrize("app_id", [0, -1, True, "987654", 15368, 987654.0])
def test_app_form_alone_keeps_its_own_refusal_for_a_malformed_app_id(
    monkeypatch, world, app_id
):
    monkeypatch.setattr(admission, "GUARD_APP_ID", app_id)
    fetch = Recorder(world)
    with pytest.raises(
        AdmissionError,
        match=r"^the merge guard App is not recorded on encoder main \(GUARD_APP_ID\)$",
    ):
        enforced(fetch)
    assert fetch.calls == []


def _workflow(**changes):
    fields = {
        "repository_id": REPOSITORY_ID,
        "path": admission.GUARD_WORKFLOW_PATH,
        "shas": frozenset({SHA_A}),
    } | changes
    return GuardWorkflow(**fields)


class _Subclass(GuardWorkflow):
    pass


class _Lookalike:
    repository_id = REPOSITORY_ID
    path = admission.GUARD_WORKFLOW_PATH
    shas = frozenset({SHA_A})


class _Text(str):
    pass


@pytest.mark.parametrize(
    "workflow",
    [
        pytest.param(
            {
                "repository_id": REPOSITORY_ID,
                "path": admission.GUARD_WORKFLOW_PATH,
                "shas": frozenset({SHA_A}),
            },
            id="a-dict-with-the-right-keys",
        ),
        pytest.param(
            (REPOSITORY_ID, admission.GUARD_WORKFLOW_PATH, frozenset({SHA_A})),
            id="tuple",
        ),
        pytest.param(
            _Subclass(REPOSITORY_ID, admission.GUARD_WORKFLOW_PATH, frozenset({SHA_A})),
            id="subclass",
        ),
        pytest.param(_Lookalike(), id="lookalike"),
        pytest.param(True, id="true"),
        pytest.param(_workflow(repository_id=True), id="bool-id"),
        pytest.param(_workflow(repository_id=0), id="zero-id"),
        pytest.param(_workflow(repository_id=-REPOSITORY_ID), id="negative-id"),
        pytest.param(_workflow(repository_id=str(REPOSITORY_ID)), id="string-id"),
        pytest.param(_workflow(repository_id=float(REPOSITORY_ID)), id="float-id"),
        pytest.param(_workflow(repository_id=None), id="no-id"),
        pytest.param(_workflow(path=".github/workflows/other.yml"), id="other-path"),
        pytest.param(
            _workflow(path="/" + admission.GUARD_WORKFLOW_PATH), id="leading-slash"
        ),
        pytest.param(
            _workflow(path=admission.GUARD_WORKFLOW_PATH.upper()), id="path-case"
        ),
        pytest.param(
            _workflow(path=_Text(admission.GUARD_WORKFLOW_PATH)), id="str-subclass-path"
        ),
        pytest.param(_workflow(path=None), id="no-path"),
        pytest.param(_workflow(shas=[SHA_A]), id="list-of-shas"),
        pytest.param(_workflow(shas={SHA_A}), id="mutable-set"),
        pytest.param(_workflow(shas=(SHA_A,)), id="tuple-of-shas"),
        pytest.param(_workflow(shas=SHA_A), id="one-string"),
        pytest.param(_workflow(shas=frozenset()), id="empty-set"),
        pytest.param(_workflow(shas=None), id="no-shas"),
        pytest.param(_workflow(shas=frozenset({SHA_A, SHA_A[:39]})), id="short-sha"),
        pytest.param(_workflow(shas=frozenset({SHA_A.upper()})), id="uppercase-sha"),
        pytest.param(_workflow(shas=frozenset({SHA_A + "\n"})), id="newline-sha"),
        pytest.param(_workflow(shas=frozenset({SHA_A, 1})), id="integer-sha"),
        pytest.param(_workflow(shas=frozenset({_Text(SHA_A)})), id="str-subclass-sha"),
        pytest.param(_workflow(shas=frozenset({"main"})), id="a-ref"),
    ],
)
def test_a_malformed_recorded_workflow_refuses_before_any_read(
    monkeypatch, world, workflow
):
    monkeypatch.setattr(admission, "V6_WRITER_ENABLED", True)
    monkeypatch.setattr(admission, "GUARD_WORKFLOW", workflow)
    # The listing would satisfy the value, were it believed.
    entry = world[RULES][0]["parameters"]["workflows"][0]
    for name in ("repository_id", "path"):
        value = getattr(workflow, name, None)
        if isinstance(value, (int, str)):
            entry[name] = value
    fetch = Recorder(world)
    with pytest.raises(AdmissionError, match=r"malformed \(GUARD_WORKFLOW\)$"):
        enforced(fetch)
    assert fetch.calls == []
    with pytest.raises(AdmissionError, match=r"malformed \(GUARD_WORKFLOW\)$"):
        admission.verify_reviewed_pr(SELECTOR, fetch=fetch)
    assert fetch.calls == [RUN]


def test_the_recorded_workflow_cannot_be_changed_or_extended():
    with pytest.raises(AttributeError):
        WORKFLOW.repository_id = 1
    with pytest.raises((AttributeError, TypeError)):
        WORKFLOW.extra = 1
    with pytest.raises(TypeError):
        GuardWorkflow(
            REPOSITORY_ID, admission.GUARD_WORKFLOW_PATH, frozenset({SHA_A}), "x"
        )
    with pytest.raises(TypeError):
        GuardWorkflow(
            repository_id=REPOSITORY_ID,
            path=admission.GUARD_WORKFLOW_PATH,
            shas=frozenset({SHA_A}),
            ref="main",
        )


# --- lock 3, ruleset form: what is accepted --------------------------------------


def test_the_synthetic_listing_is_accepted_with_one_read(ruleset_on, world):
    fetch = Recorder(world)
    assert enforced(fetch) == ACCEPTED
    assert fetch.calls == [RULES]
    assert json.loads(json.dumps(ACCEPTED)) == ACCEPTED


def test_each_recorded_sha_is_accepted_and_named_in_the_record(ruleset_on, world):
    world[RULES][0]["parameters"]["workflows"][0]["sha"] = SHA_B
    assert enforced(world.__getitem__) == ACCEPTED | {"sha": SHA_B}


def _entry(listing_):
    return listing_[0]["parameters"]["workflows"][0]


def _null_ref(rules):
    _entry(rules)["ref"] = None


def _not_enforced_on_create(rules):
    rules[0]["parameters"]["do_not_enforce_on_create"] = False


def _queue_first(rules):
    rules.reverse()


def _two_queues(rules):
    rules.append(copy.deepcopy(rules[1]) | {"ruleset_id": 9})


def _other_rules(rules):
    rules.insert(0, dict(FILLER))
    rules.append(dict(FILLER, type="pull_request", parameters={}))


def _unknown_rule_field(rules):
    rules[0]["future_field"] = 1


@pytest.mark.parametrize(
    "change",
    [
        _null_ref,
        _not_enforced_on_create,
        _queue_first,
        _two_queues,
        _other_rules,
        _unknown_rule_field,
    ],
)
def test_harmless_variants_of_the_listing_are_accepted(ruleset_on, world, change):
    change(world[RULES])
    assert enforced(world.__getitem__) == ACCEPTED
    assert (
        admission.verify_reviewed_pr(SELECTOR, fetch=world.__getitem__)["guard"]
        == ACCEPTED
    )


# --- lock 3, ruleset form: what refuses -------------------------------------------


def _rule(**changes):
    def change(rules):
        rules[0].update(changes)

    return change


def _rule_without(key):
    def change(rules):
        del rules[0][key]

    return change


def _parameters(**changes):
    def change(rules):
        rules[0]["parameters"].update(changes)

    return change


def _entry_with(**changes):
    def change(rules):
        _entry(rules).update(changes)

    return change


def _entry_without(key, **changes):
    def change(rules):
        del _entry(rules)[key]
        _entry(rules).update(changes)

    return change


def _replace(value):
    def change(rules):
        rules[:] = value

    return change


def _append(value):
    def change(rules):
        rules.append(value)

    return change


def _second_workflows_rule(**changes):
    def change(rules):
        rules.append(copy.deepcopy(rules[0]) | changes)

    return change


def _no_queue(rules):
    del rules[1]


def _queue_type(value):
    def change(rules):
        rules[1]["type"] = value

    return change


GOOD_ENTRY = SYNTHETIC["response"][0]["parameters"]["workflows"][0]
NONE = "exactly one active rule that requires workflows; the branch rules show 0"
TWO = "exactly one active rule that requires workflows; the branch rules show 2"
SOURCE = "not from a ruleset of the organization TheAxiomFoundation$"
SHAPE = "required workflows rule has an unknown shape"
IDENTITY = r"not the recorded merge guard workflow \(repository id or path differs\)"
REF_FOLLOWING = "follows a ref; only a pinned commit SHA is accepted"
NOT_PINNED = "is not pinned to a commit SHA$"
NOT_RECORDED = r"pinned to a commit that is not recorded \(GUARD_WORKFLOW\)"
QUEUE = "main does not require a merge queue"
REFUSALS = {
    # Evaluate or disabled enforcement: the rule is not in the listing.
    "empty-listing-as-in-evaluate-mode": (_replace([]), NONE),
    "recorded-rulespec-us": (_replace(RECORDED["reads"][0]["response"]), NONE),
    "recorded-axiom-encode-queue-only": (
        _replace(RECORDED["reads"][1]["response"]),
        NONE,
    ),
    "queue-present-workflows-missing": (_replace(SYNTHETIC["response"][1:]), NONE),
    "only-other-rules": (_replace([FILLER]), NONE),
    "type-case-variant": (_rule(type="Workflows"), NONE),
    "type-with-space": (_rule(type="workflows "), NONE),
    "required-status-checks-is-not-workflows": (
        _rule(type="required_status_checks"),
        NONE,
    ),
    # The source of the ruleset.
    "repository-source": (
        _rule(
            ruleset_source_type="Repository",
            ruleset_source="TheAxiomFoundation/rulespec-us",
        ),
        SOURCE,
    ),
    "repository-source-type-with-the-organization-name": (
        _rule(ruleset_source_type="Repository"),
        SOURCE,
    ),
    "another-organization": (_rule(ruleset_source="OtherOrg"), SOURCE),
    "organization-name-case": (_rule(ruleset_source="theaxiomfoundation"), SOURCE),
    "organization-name-as-repository": (
        _rule(ruleset_source="TheAxiomFoundation/rulespec-us"),
        SOURCE,
    ),
    "source-type-case": (_rule(ruleset_source_type="organization"), SOURCE),
    "enterprise-source": (_rule(ruleset_source_type="Enterprise"), SOURCE),
    "no-source-type": (_rule_without("ruleset_source_type"), SOURCE),
    "no-source": (_rule_without("ruleset_source"), SOURCE),
    "no-ruleset-id": (_rule_without("ruleset_id"), SOURCE),
    "null-ruleset-id": (_rule(ruleset_id=None), SOURCE),
    "string-ruleset-id": (_rule(ruleset_id="4242"), SOURCE),
    "bool-ruleset-id": (_rule(ruleset_id=True), SOURCE),
    "float-ruleset-id": (_rule(ruleset_id=4242.0), SOURCE),
    "zero-ruleset-id": (_rule(ruleset_id=0), SOURCE),
    "negative-ruleset-id": (_rule(ruleset_id=-4242), SOURCE),
    # More than one rule, more than one entry.
    "two-identical-workflows-rules": (_second_workflows_rule(), TWO),
    "a-second-workflows-rule-of-the-repository": (
        _second_workflows_rule(
            ruleset_source_type="Repository",
            ruleset_source="TheAxiomFoundation/rulespec-us",
            ruleset_id=5,
        ),
        TWO,
    ),
    "two-entries": (_parameters(workflows=[GOOD_ENTRY, GOOD_ENTRY]), "it names 2$"),
    "a-second-entry-of-another-workflow": (
        _parameters(
            workflows=[GOOD_ENTRY, GOOD_ENTRY | {"path": ".github/workflows/x.yml"}]
        ),
        "it names 2$",
    ),
    "no-entry": (_parameters(workflows=[]), "it names 0$"),
    # Unknown shapes.
    "no-parameters": (_rule_without("parameters"), "parameters is not an object"),
    "null-parameters": (_rule(parameters=None), "parameters is not an object"),
    "list-parameters": (_rule(parameters=[GOOD_ENTRY]), "parameters is not an object"),
    "no-workflows-list": (_rule(parameters={}), SHAPE),
    "null-workflows": (_parameters(workflows=None), SHAPE),
    "workflows-as-object": (_parameters(workflows=GOOD_ENTRY), SHAPE),
    "workflows-as-tuple": (_parameters(workflows=(GOOD_ENTRY,)), SHAPE),
    "unknown-parameter": (_parameters(required_workflows=[GOOD_ENTRY]), SHAPE),
    "entry-is-a-string": (
        _parameters(workflows=["x"]),
        "required workflow is not an object",
    ),
    "entry-is-null": (
        _parameters(workflows=[None]),
        "required workflow is not an object",
    ),
    "entry-is-a-list": (
        _parameters(workflows=[[GOOD_ENTRY]]),
        "required workflow is not an object",
    ),
    "unknown-entry-field": (_entry_with(branch="main"), "entry has an unknown field"),
    "rule-is-a-string": (_append("x"), "list entry is not an object"),
    "rule-is-null": (_append(None), "list entry is not an object"),
    "rule-without-type": (
        _append({"ruleset_id": 1}),
        "a branch rule of main has no type",
    ),
    "rule-with-null-type": (_append(dict(FILLER, type=None)), "has no type"),
    "rule-with-integer-type": (_append(dict(FILLER, type=5)), "has no type"),
    "rule-with-list-type": (_append(dict(FILLER, type=["workflows"])), "has no type"),
    # The workflow: repository id and path.
    "repository-id-as-string": (
        _entry_with(repository_id=str(REPOSITORY_ID)),
        IDENTITY,
    ),
    "repository-id-as-float": (
        _entry_with(repository_id=float(REPOSITORY_ID)),
        IDENTITY,
    ),
    "repository-id-true": (_entry_with(repository_id=True), IDENTITY),
    "repository-id-null": (_entry_with(repository_id=None), IDENTITY),
    "no-repository-id": (_entry_without("repository_id"), IDENTITY),
    "another-repository": (_entry_with(repository_id=1114692039), IDENTITY),
    "another-workflow-file": (
        _entry_with(path=".github/workflows/merge-guard-runner.yml"),
        IDENTITY,
    ),
    "path-case-variant": (
        _entry_with(path=".github/workflows/Content-Bound-Merge-Guard.yml"),
        IDENTITY,
    ),
    "path-leading-slash": (_entry_with(path="/" + GOOD_ENTRY["path"]), IDENTITY),
    "path-dot-slash": (_entry_with(path="./" + GOOD_ENTRY["path"]), IDENTITY),
    "path-trailing-space": (_entry_with(path=GOOD_ENTRY["path"] + " "), IDENTITY),
    "path-yaml-extension": (
        _entry_with(path=GOOD_ENTRY["path"][:-3] + "yaml"),
        IDENTITY,
    ),
    "path-null": (_entry_with(path=None), IDENTITY),
    "path-as-list": (_entry_with(path=[GOOD_ENTRY["path"]]), IDENTITY),
    "no-path": (_entry_without("path"), IDENTITY),
    # The pin.
    "ref-only": (_entry_without("sha", ref="refs/heads/main"), REF_FOLLOWING),
    "ref-next-to-a-recorded-sha": (_entry_with(ref="refs/heads/main"), REF_FOLLOWING),
    "ref-that-is-a-recorded-sha": (_entry_without("sha", ref=SHA_A), REF_FOLLOWING),
    "empty-ref": (_entry_with(ref=""), REF_FOLLOWING),
    "false-ref": (_entry_with(ref=False), REF_FOLLOWING),
    "no-sha": (_entry_without("sha"), NOT_PINNED),
    "null-sha": (_entry_with(sha=None), NOT_PINNED),
    "null-sha-and-null-ref": (_entry_with(sha=None, ref=None), NOT_PINNED),
    "short-sha": (_entry_with(sha=SHA_A[:39]), NOT_PINNED),
    "abbreviated-sha": (_entry_with(sha=SHA_A[:7]), NOT_PINNED),
    "long-sha": (_entry_with(sha=SHA_A + "a"), NOT_PINNED),
    "uppercase-sha": (_entry_with(sha=SHA_A.upper()), NOT_PINNED),
    "sha-with-newline": (_entry_with(sha=SHA_A + "\n"), NOT_PINNED),
    "sha-with-space": (_entry_with(sha=" " + SHA_A), NOT_PINNED),
    "sha-as-list": (_entry_with(sha=[SHA_A]), NOT_PINNED),
    "sha-as-integer": (_entry_with(sha=1), NOT_PINNED),
    "empty-sha": (_entry_with(sha=""), NOT_PINNED),
    "sha-256-length": (_entry_with(sha="a1" * 32), NOT_PINNED),
    "unrecorded-sha": (_entry_with(sha="c3" * 20), NOT_RECORDED),
    # The merge queue.
    "no-merge-queue": (_no_queue, QUEUE),
    "merge-queue-type-case": (_queue_type("Merge_Queue"), QUEUE),
    "merge-queue-other-name": (_queue_type("merge-queue"), QUEUE),
}


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_every_other_listing_refuses_and_names_what_is_wrong(ruleset_on, world, name):
    change, message = REFUSALS[name]
    change(world[RULES])
    fetch = Recorder(world)
    with pytest.raises(AdmissionError, match=message):
        enforced(fetch)
    assert fetch.calls == [RULES]
    with pytest.raises(AdmissionError, match=message):
        admission.verify_reviewed_pr(SELECTOR, fetch=Recorder(world))


def test_a_true_repository_id_is_not_the_recorded_id_one(monkeypatch, world):
    # ``True == 1``: with repository id 1 recorded, ``true`` is not that id.
    monkeypatch.setattr(admission, "GUARD_WORKFLOW", _workflow(repository_id=1))
    _entry(world[RULES])["repository_id"] = 1
    assert enforced(world.__getitem__) == ACCEPTED | {"repository_id": 1}
    _entry(world[RULES])["repository_id"] = True
    with pytest.raises(AdmissionError, match=IDENTITY):
        enforced(world.__getitem__)


def test_a_true_ruleset_id_is_not_ruleset_one(ruleset_on, world):
    world[RULES][0]["ruleset_id"] = 1
    assert enforced(world.__getitem__) == ACCEPTED | {"ruleset_id": 1}
    world[RULES][0]["ruleset_id"] = True
    with pytest.raises(AdmissionError, match=SOURCE):
        enforced(world.__getitem__)


def test_the_app_forms_branch_record_never_satisfies_the_ruleset_form(
    ruleset_on, world
):
    # The guard App check is required in classic protection; no rule exists.
    world[RULES] = []
    fetch = Recorder(world)
    with pytest.raises(AdmissionError, match=NONE):
        enforced(fetch)
    assert BRANCH not in fetch.calls


def test_the_ruleset_listing_never_satisfies_the_app_form(monkeypatch, world):
    monkeypatch.setattr(admission, "GUARD_APP_ID", APP)
    world[BRANCH] = required(VALIDATE)
    fetch = Recorder(world)
    with pytest.raises(AdmissionError, match="not a required check of main"):
        enforced(fetch)
    assert fetch.calls == [BRANCH]


# --- the read: errors, pages, bounds ----------------------------------------------


@pytest.mark.parametrize(
    "response",
    [
        urllib.error.HTTPError("u", 403, "rate limited", {}, None),
        urllib.error.HTTPError("u", 429, "secondary rate limit", {}, None),
        urllib.error.HTTPError("u", 404, "Not Found", {}, None),
        urllib.error.HTTPError("u", 500, "error", {}, None),
        urllib.error.URLError("unreachable"),
        TimeoutError(),
        ValueError("bad JSON"),
        json.JSONDecodeError("Expecting value", "<html>", 0),
        RecursionError(),
        AdmissionError("GitHub admission requests must not redirect"),
        AdmissionError("GitHub admission response exceeds its bound"),
        None,
        {},
        "[]",
        {"message": "Not Found"},
        {"type": "workflows"},
        (),
    ],
)
def test_an_unreadable_listing_refuses_and_is_never_read_as_absent(
    ruleset_on, world, response
):
    world[RULES] = response
    with pytest.raises(AdmissionError):
        enforced(Recorder(world))
    with pytest.raises(AdmissionError):
        admission.verify_reviewed_pr(SELECTOR, fetch=Recorder(world))


def test_a_read_error_is_reported_as_unreadable_with_its_cause(ruleset_on, world):
    error = urllib.error.HTTPError("u", 403, "rate limited", {}, None)
    world[RULES] = error
    with pytest.raises(
        AdmissionError, match="^branch rules of the RuleSpec repository cannot be read$"
    ) as refusal:
        enforced(Recorder(world))
    assert refusal.value.__cause__ is error
    del world[RULES]
    with pytest.raises(AdmissionError, match="cannot be read"):
        admission.verify_reviewed_pr(SELECTOR, fetch=world.__getitem__)


class _Response:
    def __init__(self, status, raw):
        self.status, self.raw = status, raw

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, _limit):
        return self.raw


@pytest.mark.parametrize(
    ("status", "raw"),
    [
        (200, b"<html>rate limited</html>"),
        (200, b""),
        (202, json.dumps(SYNTHETIC["response"]).encode()),
        (304, b"[]"),
        (200, b"[" + b" " * admission.MAX_RESPONSE_BYTES + b"]"),
    ],
)
def test_the_production_reader_refuses_non_json_and_non_200(
    ruleset_on, monkeypatch, status, raw
):
    requests = []

    class Opener:
        def open(self, request, timeout):
            requests.append(request.full_url)
            return _Response(status, raw)

    monkeypatch.setattr(
        admission.urllib.request, "build_opener", lambda *handlers: Opener()
    )
    with pytest.raises(AdmissionError):
        enforced(admission.github_json)
    assert requests == ["https://api.github.com" + RULES]


def test_the_production_reader_reads_the_synthetic_listing(ruleset_on, monkeypatch):
    # The control of the test above: the same stand-in, a good response.
    raw = json.dumps(SYNTHETIC["response"]).encode()

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "https://api.github.com" + RULES
            assert request.get_header("Authorization") is None
            return _Response(200, raw)

    monkeypatch.setattr(
        admission.urllib.request, "build_opener", lambda *handlers: Opener()
    )
    assert enforced(admission.github_json) == ACCEPTED


def test_the_production_reader_refuses_a_redirect():
    handler = admission._NoRedirect()
    with pytest.raises(AdmissionError, match="must not redirect"):
        handler.redirect_request(None, None, 302, "Found", {}, "https://example.org/")


def test_the_whole_listing_is_read_a_second_page_counts(ruleset_on, world):
    rules = listing()
    # Page 1 is full: 99 other rules and the merge queue. The workflows rule
    # is on page 2.
    world[RULES] = [dict(FILLER) for _ in range(99)] + [rules[1]]
    world[RULES_2] = [rules[0]]
    fetch = Recorder(world)
    assert enforced(fetch) == ACCEPTED
    assert fetch.calls == [RULES, RULES_2]
    # A second workflows rule on page 2 is seen, and refuses.
    world[RULES] = [dict(FILLER) for _ in range(98)] + rules
    world[RULES_2] = [copy.deepcopy(rules[0])]
    with pytest.raises(AdmissionError, match=TWO):
        enforced(world.__getitem__)
    # The merge queue on page 2 is seen.
    world[RULES] = [dict(FILLER) for _ in range(99)] + [rules[0]]
    world[RULES_2] = [rules[1]]
    assert enforced(world.__getitem__) == ACCEPTED
    world[RULES_2] = []
    with pytest.raises(AdmissionError, match=QUEUE):
        enforced(world.__getitem__)
    # A missing or unreadable second page refuses.
    del world[RULES_2]
    with pytest.raises(AdmissionError, match="cannot be read"):
        enforced(world.__getitem__)


def test_the_listing_is_read_within_a_bound_and_refused_past_it(ruleset_on, world):
    full = [dict(FILLER) for _ in range(98)] + listing()
    calls = []

    def fetch(path):
        calls.append(path)
        return copy.deepcopy(full)

    with pytest.raises(AdmissionError, match="pagination limit reached"):
        enforced(fetch)
    assert calls == [
        f"{RULES_PATH}?per_page=100&page={page}"
        for page in range(1, admission.MAX_PAGES + 1)
    ]
    # More than a page's worth in one response is not a page.
    world[RULES] = [dict(FILLER) for _ in range(99)] + listing()
    assert len(world[RULES]) == 101
    with pytest.raises(AdmissionError, match="invalid paginated GitHub response"):
        enforced(world.__getitem__)


@pytest.mark.parametrize(
    "repository",
    [
        "TheAxiomFoundation/rulespec-us-tn",
        "TheAxiomFoundation/axiom-encode",
        "other/rulespec-us",
        "rulespec-us",
        "TheAxiomFoundation/rulespec-us/../axiom-encode",
        None,
    ],
)
def test_ruleset_form_refuses_a_repository_outside_the_promotion_lane(
    ruleset_on, repository
):
    def fetch(path):
        raise AssertionError("no read for an invalid repository")

    with pytest.raises(AdmissionError, match="invalid repository"):
        admission.require_guard_enforced(repository, fetch=fetch)


# --- admission and revalidation carry the record of either form -------------------


@pytest.fixture
def admitted(ruleset_on, world):
    return admission.verify_reviewed_pr(SELECTOR, fetch=world.__getitem__)


def test_admission_reads_the_rules_first_and_last_and_records_the_form(
    ruleset_on, world
):
    fetch = Recorder(world)
    record = admission.verify_reviewed_pr(SELECTOR, fetch=fetch)
    assert set(record) == admission.CANDIDATE_FIELDS
    assert record["guard"] == ACCEPTED
    assert record["owner_review"] == "none" and "approvals" not in record
    assert fetch.calls[:2] == [RUN, RULES]
    assert fetch.calls[-1] == RULES
    assert fetch.calls.count(RULES) == 2 and BRANCH not in fetch.calls
    # The record survives the JSON round trip of the runtime attestation.
    assert json.loads(json.dumps(record)) == record


def test_revalidation_returns_the_same_record_and_reads_the_rules(admitted, world):
    fetch = Recorder(world)
    assert admission.revalidate_admission(admitted, fetch=fetch) == admitted
    assert fetch.calls.count(RULES) == 1 and BRANCH not in fetch.calls
    assert admission.is_candidate_admission(admitted) is True


def _pin_moved(api_):
    _entry(api_[RULES])["sha"] = SHA_B


def _ruleset_replaced(api_):
    api_[RULES][0]["ruleset_id"] = RULESET_ID + 1


def _rule_gone(api_):
    api_[RULES] = api_[RULES][1:]


def _queue_gone(api_):
    api_[RULES] = api_[RULES][:1]


def _unreadable(api_):
    api_[RULES] = urllib.error.HTTPError("u", 403, "rate limited", {}, None)


LIVE_CHANGES = [_pin_moved, _ruleset_replaced, _rule_gone, _queue_gone, _unreadable]


@pytest.mark.parametrize("change", LIVE_CHANGES)
def test_revalidation_refuses_any_change_of_the_requirement(admitted, world, change):
    change(world)
    with pytest.raises(AdmissionError):
        admission.revalidate_admission(admitted, fetch=Recorder(world))


@pytest.mark.parametrize("change", LIVE_CHANGES)
def test_a_requirement_that_changes_during_admission_refuses(ruleset_on, world, change):
    before = copy.deepcopy(world[RULES])
    after = copy.deepcopy(world)
    change(after)
    reads = []

    class Api(dict):
        def __getitem__(self, path):
            if path == RULES:
                reads.append(path)
                value = before if len(reads) == 1 else after[RULES]
                if isinstance(value, Exception):
                    raise value
                return copy.deepcopy(value)
            return super().__getitem__(path)

    with pytest.raises(AdmissionError):
        admission.verify_reviewed_pr(SELECTOR, fetch=Api(world).__getitem__)
    assert len(reads) == 2


def _guard(**changes):
    def change(record):
        record["guard"].update(changes)

    return change


def _app_form_guard(record):
    record["guard"] = {"check": admission.GUARD_CHECK_NAME, "app_id": APP}


def _guard_without(key):
    def change(record):
        del record["guard"][key]

    return change


@pytest.mark.parametrize(
    "change",
    [
        _guard(sha=SHA_B),
        _guard(ruleset_id=RULESET_ID + 1),
        _guard(repository_id=1114692039),
        _guard(path=".github/workflows/x.yml"),
        _guard(form="classic_branch_protection"),
        _guard(app_id=APP),
        _guard_without("form"),
        _guard_without("sha"),
        _app_form_guard,
    ],
)
def test_revalidation_refuses_an_edited_guard_record(admitted, world, change):
    record = copy.deepcopy(admitted)
    change(record)
    with pytest.raises(AdmissionError, match="live candidate differs"):
        admission.revalidate_admission(record, fetch=Recorder(world))


def test_a_record_of_one_form_is_refused_under_the_other(monkeypatch, admitted, world):
    monkeypatch.setattr(admission, "GUARD_WORKFLOW", None)
    monkeypatch.setattr(admission, "GUARD_APP_ID", APP)
    with pytest.raises(AdmissionError, match="live candidate differs"):
        admission.revalidate_admission(admitted, fetch=Recorder(world))
    by_app = admission.verify_reviewed_pr(SELECTOR, fetch=world.__getitem__)
    assert by_app["guard"] == {"check": admission.GUARD_CHECK_NAME, "app_id": APP}
    monkeypatch.setattr(admission, "GUARD_APP_ID", None)
    monkeypatch.setattr(admission, "GUARD_WORKFLOW", WORKFLOW)
    with pytest.raises(AdmissionError, match="live candidate differs"):
        admission.revalidate_admission(by_app, fetch=Recorder(world))
    # With neither recorded again, both records refuse at the lock.
    monkeypatch.setattr(admission, "GUARD_WORKFLOW", None)
    for record in (admitted, by_app):
        with pytest.raises(
            AdmissionError, match="neither GUARD_APP_ID nor GUARD_WORKFLOW"
        ):
            admission.revalidate_admission(record, fetch=Recorder(world))


# --- the evidence step of change 5b uses the same lock ----------------------------


class Reads:
    """The two anonymous reads of lock 3, one per form."""

    def __init__(self, branch, rules):
        self.values, self.calls = {BRANCH: branch, RULES: rules}, []

    def __call__(self, path):
        self.calls.append(path)
        value = self.values[path]
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)


def _evidence(fetch):
    try:
        return (
            "ok",
            evidence.require_candidate_promotion_evidence(
                written_v6(REF),
                atomic_source_json=LIVE_PROMOTION,
                country="us",
                citation=CITATION,
                rulespec_ref=REF,
                fetch=fetch,
            ),
        )
    except ValueError as exc:
        return ("refused", str(exc))


BRANCH_STATES = {
    "required": required(VALIDATE, GUARD),
    "not-required": required(VALIDATE),
    "unreadable": OSError("connection reset"),
}
RULES_STATES = {
    "required": SYNTHETIC["response"],
    "empty": [],
    "no-queue": SYNTHETIC["response"][:1],
    "rate-limited": AdmissionError("rate limited"),
}


def _lock_states():
    return itertools.product(
        (False, True),  # lock 1
        (False, True),  # the reader switch
        (None, 15368, APP),  # lock 2, App form (15368 is GitHub Actions)
        (None, WORKFLOW),  # lock 2, ruleset form
        sorted(BRANCH_STATES),  # lock 3, App form
        sorted(RULES_STATES),  # lock 3, ruleset form
    )


@pytest.mark.parametrize(
    ("writer", "reader", "app_id", "workflow", "branch", "rules"), _lock_states()
)
def test_evidence_is_admitted_only_with_every_lock_open_in_exactly_one_form(
    monkeypatch, writer, reader, app_id, workflow, branch, rules
):
    monkeypatch.setattr(admission, "V6_WRITER_ENABLED", writer)
    monkeypatch.setattr(v6, "V6_READER_ENABLED", reader)
    monkeypatch.setattr(admission, "GUARD_APP_ID", app_id)
    monkeypatch.setattr(admission, "GUARD_WORKFLOW", workflow)
    fetch = Reads(BRANCH_STATES[branch], RULES_STATES[rules])
    result = _evidence(fetch)
    by_app = app_id == APP and workflow is None
    by_ruleset = app_id is None and workflow is WORKFLOW
    opened = (
        writer
        and reader
        and ((by_app and branch == "required") or (by_ruleset and rules == "required"))
    )
    assert (result == ("ok", REF)) is bool(opened)
    assert result[0] == ("ok" if opened else "refused")
    # Never "nothing there": one read, of the recorded form only, and only
    # behind locks 1 and 2.
    if writer and reader and by_app:
        assert fetch.calls == [BRANCH]
    elif writer and reader and by_ruleset:
        assert fetch.calls == [RULES]
    else:
        assert fetch.calls == []


def test_the_lock_state_table_has_exactly_the_expected_admitting_rows():
    states = list(_lock_states())
    assert len(states) == 2 * 2 * 3 * 2 * 3 * 4 == 288
    admitting = [
        state
        for state in states
        if state[0]
        and state[1]
        and (
            (state[2] == APP and state[3] is None and state[4] == "required")
            or (state[2] is None and state[3] is WORKFLOW and state[5] == "required")
        )
    ]
    # App form: 4 states of the unread rules listing. Ruleset form: 3 states
    # of the unread branch record.
    assert len(admitting) == 4 + 3
