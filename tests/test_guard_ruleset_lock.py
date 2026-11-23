"""Change 5c of docs/content-bound-merge-guard.md (#2172): locked.

A second accepted form of locks 2 and 3 of the v6 writer: ``GUARD_WORKFLOW``
and a live read that an organization ruleset requires the guard workflow, at
a recorded commit, on ``main`` of the RuleSpec repository, with a merge queue.
Exactly one of ``GUARD_APP_ID`` and ``GUARD_WORKFLOW`` may be recorded. Both
are ``None`` on encoder main; on-state behaviour is reached in tests only, by
patching the constants.

The listing with the ``workflows`` rule is SYNTHETIC.
[experiment X14]: must be replaced by the recorded live response before
activation.
"""

from __future__ import annotations

import ast
import copy
import itertools
import json
import re
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
    ):
        text = flat(path)
        assert "SYNTHETIC" in text, path
        assert X14 in text, path
    # This file: the disclosure must stand in the MODULE DOCSTRING. A scan of
    # the file's own text would find the two strings in this test and in the
    # constant ``X14`` above, also with the docstring removed.
    for problem in _disclosure_problems(Path(__file__).read_text("utf-8")):
        raise AssertionError(problem)


def _disclosure_problems(source: str) -> list[str]:
    """What is missing from the module docstring of a test module's source."""

    docstring = ast.get_docstring(ast.parse(source)) or ""
    text = " ".join(docstring.split())
    return [
        f"the module docstring does not say {needed!r}"
        for needed in ("SYNTHETIC", X14)
        if needed not in text
    ]


def test_the_disclosure_check_reads_the_module_docstring_and_nothing_else():
    """Review of ``e2d12f5c2``: the earlier check scanned this file's whole
    text, which names both strings outside the docstring too."""

    source = Path(__file__).read_text("utf-8")
    assert _disclosure_problems(source) == []
    tree = ast.parse(source)
    first = tree.body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
    lines = source.splitlines(keepends=True)
    without = "".join(lines[: first.lineno - 1] + lines[first.end_lineno :])
    # The strings are still in the text, and the check is not fooled.
    assert "SYNTHETIC" in without and X14 in " ".join(without.split())
    assert len(_disclosure_problems(without)) == 2
    # A docstring that names only one of the two is one problem.
    assert len(_disclosure_problems('"""SYNTHETIC only."""\n' + without)) == 1
    # A string that is not the docstring does not count.
    assert len(_disclosure_problems("import os\n" + source)) == 2


def test_the_synthetic_listing_uses_only_fields_of_the_rest_schema():
    workflows, queue = SYNTHETIC["response"]
    info = {"ruleset_source_type", "ruleset_source", "ruleset_id"}
    assert set(workflows) == {"type", "parameters"} | info
    assert set(workflows["parameters"]) == {"workflows"}
    (entry,) = workflows["parameters"]["workflows"]
    assert set(entry) == {"repository_id", "path", "sha"}
    assert set(entry) <= admission._WORKFLOW_ENTRY_FIELDS
    # The rule-level field set of the lock is the schema's, and it is exactly
    # the key set of the one recorded live rule of this endpoint.
    assert admission._BRANCH_RULE_FIELDS == {"type", "parameters"} | info
    assert set(RECORDED["reads"][1]["response"][0]) == admission._BRANCH_RULE_FIELDS
    # The merge-queue rule is the recorded one of axiom-encode, with another
    # (invented) source and id.
    recorded = RECORDED["reads"][1]["response"][0]
    assert set(queue) == set(recorded)
    assert queue["type"] == recorded["type"] == "merge_queue"
    # ... and with the two size parameters set to what the lock requires. The
    # recorded queue of axiom-encode builds and merges up to 5 entries.
    assert set(queue["parameters"]) == set(recorded["parameters"])
    assert set(queue["parameters"]) == admission._MERGE_QUEUE_PARAMETER_FIELDS
    assert queue["parameters"] == recorded["parameters"] | {
        "max_entries_to_build": 1,
        "max_entries_to_merge": 1,
    }
    assert recorded["parameters"]["max_entries_to_build"] == 5
    assert recorded["parameters"]["max_entries_to_merge"] == 5
    assert (
        "`max_entries_to_build` and `max_entries_to_merge` changed"
        in (SYNTHETIC["note"])
    )


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


class _EverySha(frozenset):
    """Review round 1, P3-2: a ``frozenset`` subclass answers ``in`` as it
    likes. Here it claims to hold every SHA."""

    def __contains__(self, _sha):
        return True


def _without_fields(*names):
    """Review round 1, P3-3: an instance made without ``__init__``; only the
    named slots are set."""
    value = object.__new__(GuardWorkflow)
    fields = {
        "repository_id": REPOSITORY_ID,
        "path": admission.GUARD_WORKFLOW_PATH,
        "shas": frozenset({SHA_A}),
    }
    for name in names:
        object.__setattr__(value, name, fields[name])
    return value


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
        pytest.param(_workflow(shas=_EverySha({SHA_A})), id="frozenset-subclass"),
        pytest.param(
            _workflow(shas=_EverySha({"c3" * 20})), id="frozenset-subclass-other-sha"
        ),
        pytest.param(_without_fields(), id="instance-without-fields"),
        pytest.param(
            _without_fields("repository_id", "path"), id="instance-without-shas"
        ),
        pytest.param(_without_fields("path", "shas"), id="instance-without-id"),
        pytest.param(
            _without_fields("repository_id", "shas"), id="instance-without-path"
        ),
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


def _not_enforced_on_create(rules):
    rules[0]["parameters"]["do_not_enforce_on_create"] = False


def _queue_first(rules):
    rules.reverse()


def _two_queues(rules):
    rules.append(copy.deepcopy(rules[1]) | {"ruleset_id": 9})


def _other_rules(rules):
    rules.insert(0, dict(FILLER))
    rules.append(dict(FILLER, type="pull_request", parameters={}))


def _queue_with(**changes):
    def change(rules):
        rules[1].update(changes)

    return change


def _queue_parameters(**changes):
    def change(rules):
        rules[1]["parameters"].update(changes)

    return change


def _queue_without_parameter(name):
    def change(rules):
        del rules[1]["parameters"][name]

    return change


def _second_queue(position, **parameters):
    """One more queue rule of the same repository, with other parameters."""

    def change(rules):
        queue = copy.deepcopy(rules[1]) | {"ruleset_id": 99}
        queue["parameters"].update(parameters)
        rules.insert(position, queue)

    return change


@pytest.mark.parametrize(
    "change",
    [
        _not_enforced_on_create,
        _queue_first,
        _two_queues,
        _other_rules,
        # The queue parameters that the lock does not require (review of
        # ``e2d12f5c2``; see ``_require_guard_workflow_ruleset``).
        _queue_parameters(merge_method="SQUASH"),
        _queue_parameters(merge_method="REBASE"),
        _queue_parameters(check_response_timeout_minutes=1),
        _queue_parameters(check_response_timeout_minutes=360),
        _queue_parameters(min_entries_to_merge_wait_minutes=0),
        _queue_without_parameter("merge_method"),
        _queue_without_parameter("check_response_timeout_minutes"),
        _queue_without_parameter("min_entries_to_merge_wait_minutes"),
        _queue_with(ruleset_id=1),
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


def _queue_without(key):
    def change(rules):
        del rules[1][key]

    return change


def _queue_is(value):
    def change(rules):
        rules[1] = copy.deepcopy(value)

    return change


def _queue_before(value):
    """A second merge queue rule, listed before the good one."""

    def change(rules):
        rules.insert(1, copy.deepcopy(value))

    return change


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
REF_FOLLOWING = (
    "the required workflow entry has a ref; only a pinned commit SHA with no ref "
    "is accepted$"
)
QUEUE_PARAMETER = "a merge queue rule of main has an unknown parameter$"


def _unjudged(name, wanted):
    return (
        "a merge queue rule of main can merge a pull request without its own "
        f"passing merge-group check: {name} must be {wanted}$"
    )


STRATEGY = _unjudged("grouping_strategy", '"ALLGREEN"')
BUILD = _unjudged("max_entries_to_build", 1)
MIN_MERGE = _unjudged("min_entries_to_merge", 1)
MAX_MERGE = _unjudged("max_entries_to_merge", 1)
NOT_PINNED = "is not pinned to a commit SHA$"
NOT_RECORDED = r"pinned to a commit that is not recorded \(GUARD_WORKFLOW\)"
QUEUE = "main does not require a merge queue"
RULE_FIELD = "the rule that requires workflows on main has an unknown field$"
ON_CREATE = r"is not enforced on creation \(do_not_enforce_on_create\)$"
QUEUE_SHAPE = "a merge queue rule of main has an unknown shape$"
QUEUE_SOURCE = (
    "a merge queue rule of main is not from a ruleset of the RuleSpec "
    "repository itself$"
)
GOOD_QUEUE = SYNTHETIC["response"][1]
# The one recorded live rule of this endpoint: the queue of ANOTHER repository.
RECORDED_QUEUE = RECORDED["reads"][1]["response"][0]
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
    "ref-that-is-a-recorded-sha-next-to-it": (_entry_with(ref=SHA_A), REF_FOLLOWING),
    "empty-ref": (_entry_with(ref=""), REF_FOLLOWING),
    "false-ref": (_entry_with(ref=False), REF_FOLLOWING),
    # Review of ``e2d12f5c2``, blocking 2: the PRESENCE of ``ref`` refuses.
    # ``"ref": null`` next to the recorded SHA was accepted before; the
    # schema's ``ref`` is an optional string that is not nullable.
    "null-ref-next-to-a-recorded-sha": (_entry_with(ref=None), REF_FOLLOWING),
    "null-ref-only": (_entry_without("sha", ref=None), REF_FOLLOWING),
    "zero-ref": (_entry_with(ref=0), REF_FOLLOWING),
    "true-ref": (_entry_with(ref=True), REF_FOLLOWING),
    "list-ref": (_entry_with(ref=[]), REF_FOLLOWING),
    "list-ref-with-a-name": (_entry_with(ref=["refs/heads/main"]), REF_FOLLOWING),
    "dict-ref": (_entry_with(ref={}), REF_FOLLOWING),
    "dict-ref-with-a-sha": (_entry_with(ref={"sha": SHA_A}), REF_FOLLOWING),
    "space-ref": (_entry_with(ref=" "), REF_FOLLOWING),
    "default-branch-ref": (_entry_with(ref="main"), REF_FOLLOWING),
    "tag-ref": (_entry_with(ref="refs/tags/v1"), REF_FOLLOWING),
    "no-sha": (_entry_without("sha"), NOT_PINNED),
    "null-sha": (_entry_with(sha=None), NOT_PINNED),
    "null-sha-and-null-ref": (_entry_with(sha=None, ref=None), REF_FOLLOWING),
    "short-sha": (_entry_with(sha=SHA_A[:39]), NOT_PINNED),
    "abbreviated-sha": (_entry_with(sha=SHA_A[:7]), NOT_PINNED),
    "long-sha": (_entry_with(sha=SHA_A + "a"), NOT_PINNED),
    "uppercase-sha": (_entry_with(sha=SHA_A.upper()), NOT_PINNED),
    "sha-with-newline": (_entry_with(sha=SHA_A + "\n"), NOT_PINNED),
    "sha-with-space": (_entry_with(sha=" " + SHA_A), NOT_PINNED),
    "sha-as-list": (_entry_with(sha=[SHA_A]), NOT_PINNED),
    "sha-as-integer": (_entry_with(sha=1), NOT_PINNED),
    # Its decimal form is 40 hex characters; it is still not a string.
    "sha-as-40-digit-integer": (_entry_with(sha=int("1" * 40)), NOT_PINNED),
    "empty-sha": (_entry_with(sha=""), NOT_PINNED),
    "sha-256-length": (_entry_with(sha="a1" * 32), NOT_PINNED),
    "unrecorded-sha": (_entry_with(sha="c3" * 20), NOT_RECORDED),
    # Review round 1, P2-2: a key of the workflows rule outside the schema's
    # set. It could say whether the rule binds.
    "rule-level-enforcement-evaluate": (_rule(enforcement="evaluate"), RULE_FIELD),
    "rule-level-enforcement-active": (_rule(enforcement="active"), RULE_FIELD),
    "rule-level-bypass-actors": (
        _rule(bypass_actors=[{"actor_id": 5, "actor_type": "RepositoryRole"}]),
        RULE_FIELD,
    ),
    "rule-level-empty-bypass-actors": (_rule(bypass_actors=[]), RULE_FIELD),
    "rule-level-current-user-can-bypass": (
        _rule(current_user_can_bypass="always"),
        RULE_FIELD,
    ),
    "rule-level-future-field": (_rule(future_field=1), RULE_FIELD),
    "rule-level-null-future-field": (_rule(future_field=None), RULE_FIELD),
    # Review round 1, P2-2: only absent or exactly ``False``.
    "not-enforced-on-create-true": (
        _parameters(do_not_enforce_on_create=True),
        ON_CREATE,
    ),
    "not-enforced-on-create-string": (
        _parameters(do_not_enforce_on_create="yes"),
        ON_CREATE,
    ),
    "not-enforced-on-create-false-string": (
        _parameters(do_not_enforce_on_create="false"),
        ON_CREATE,
    ),
    "not-enforced-on-create-null": (
        _parameters(do_not_enforce_on_create=None),
        ON_CREATE,
    ),
    "not-enforced-on-create-zero": (_parameters(do_not_enforce_on_create=0), ON_CREATE),
    "not-enforced-on-create-one": (_parameters(do_not_enforce_on_create=1), ON_CREATE),
    "not-enforced-on-create-empty-list": (
        _parameters(do_not_enforce_on_create=[]),
        ON_CREATE,
    ),
    # The merge queue.
    "no-merge-queue": (_no_queue, QUEUE),
    "merge-queue-type-case": (_queue_type("Merge_Queue"), QUEUE),
    "merge-queue-other-name": (_queue_type("merge-queue"), QUEUE),
    "merge-queue-name-with-suffix": (_queue_type("merge_queue_v2"), QUEUE),
    "merge-queue-name-with-prefix": (_queue_type("no_merge_queue"), QUEUE),
    "merge-queue-name-with-space": (_queue_type("merge_queue "), QUEUE),
    # Review round 1, P2-1: the queue rule is a well-formed rule object.
    "queue-of-type-alone": (_queue_is({"type": "merge_queue"}), QUEUE_SHAPE),
    "queue-without-parameters": (_queue_without("parameters"), QUEUE_SHAPE),
    "queue-null-parameters": (_queue_with(parameters=None), QUEUE_SHAPE),
    "queue-list-parameters": (_queue_with(parameters=[]), QUEUE_SHAPE),
    "queue-string-parameters": (_queue_with(parameters="ALLGREEN"), QUEUE_SHAPE),
    "queue-enforcement-evaluate": (_queue_with(enforcement="evaluate"), QUEUE_SHAPE),
    "queue-bypass-actors": (_queue_with(bypass_actors=[]), QUEUE_SHAPE),
    "queue-current-user-can-bypass": (
        _queue_with(current_user_can_bypass="always"),
        QUEUE_SHAPE,
    ),
    "queue-future-field": (_queue_with(future_field=None), QUEUE_SHAPE),
    # ... of the ruleset of THIS repository.
    "queue-of-another-repository-as-recorded": (
        _queue_is(RECORDED_QUEUE),
        QUEUE_SOURCE,
    ),
    "queue-of-another-rulespec-repository": (
        _queue_with(ruleset_source="TheAxiomFoundation/rulespec-uk"),
        QUEUE_SOURCE,
    ),
    "queue-of-a-longer-repository-name": (
        _queue_with(ruleset_source="TheAxiomFoundation/rulespec-us-tn"),
        QUEUE_SOURCE,
    ),
    "queue-source-short-name": (
        _queue_with(ruleset_source="rulespec-us"),
        QUEUE_SOURCE,
    ),
    "queue-source-case": (
        _queue_with(ruleset_source="theaxiomfoundation/rulespec-us"),
        QUEUE_SOURCE,
    ),
    "queue-source-trailing-space": (
        _queue_with(ruleset_source="TheAxiomFoundation/rulespec-us "),
        QUEUE_SOURCE,
    ),
    "queue-source-another-owner": (
        _queue_with(ruleset_source="other/rulespec-us"),
        QUEUE_SOURCE,
    ),
    "queue-source-as-list": (
        _queue_with(ruleset_source=["TheAxiomFoundation/rulespec-us"]),
        QUEUE_SOURCE,
    ),
    "queue-source-null": (_queue_with(ruleset_source=None), QUEUE_SOURCE),
    "queue-without-source": (_queue_without("ruleset_source"), QUEUE_SOURCE),
    # GitHub documents no merge queue rule of an organization ruleset; an
    # unknown state refuses.
    "queue-organization-source": (
        _queue_with(
            ruleset_source_type="Organization", ruleset_source="TheAxiomFoundation"
        ),
        QUEUE_SOURCE,
    ),
    "queue-organization-type-with-the-repository-name": (
        _queue_with(ruleset_source_type="Organization"),
        QUEUE_SOURCE,
    ),
    "queue-source-type-case": (
        _queue_with(ruleset_source_type="repository"),
        QUEUE_SOURCE,
    ),
    "queue-enterprise-source": (
        _queue_with(ruleset_source_type="Enterprise"),
        QUEUE_SOURCE,
    ),
    "queue-source-type-null": (_queue_with(ruleset_source_type=None), QUEUE_SOURCE),
    "queue-without-source-type": (_queue_without("ruleset_source_type"), QUEUE_SOURCE),
    "queue-without-ruleset-id": (_queue_without("ruleset_id"), QUEUE_SOURCE),
    "queue-null-ruleset-id": (_queue_with(ruleset_id=None), QUEUE_SOURCE),
    "queue-string-ruleset-id": (_queue_with(ruleset_id="4343"), QUEUE_SOURCE),
    "queue-bool-ruleset-id": (_queue_with(ruleset_id=True), QUEUE_SOURCE),
    "queue-float-ruleset-id": (_queue_with(ruleset_id=4343.0), QUEUE_SOURCE),
    "queue-zero-ruleset-id": (_queue_with(ruleset_id=0), QUEUE_SOURCE),
    "queue-negative-ruleset-id": (_queue_with(ruleset_id=-4343), QUEUE_SOURCE),
    # Review of ``e2d12f5c2``, blocking 1: the queue must make every entry
    # merge alone and only on its own passing merge-group checks.
    "queue-headgreen": (_queue_parameters(grouping_strategy="HEADGREEN"), STRATEGY),
    "queue-strategy-case": (_queue_parameters(grouping_strategy="allgreen"), STRATEGY),
    "queue-strategy-with-space": (
        _queue_parameters(grouping_strategy="ALLGREEN "),
        STRATEGY,
    ),
    "queue-strategy-null": (_queue_parameters(grouping_strategy=None), STRATEGY),
    "queue-strategy-list": (
        _queue_parameters(grouping_strategy=["ALLGREEN"]),
        STRATEGY,
    ),
    "queue-strategy-true": (_queue_parameters(grouping_strategy=True), STRATEGY),
    "queue-strategy-unknown": (
        _queue_parameters(grouping_strategy="ANYGREEN"),
        STRATEGY,
    ),
    "queue-without-strategy": (_queue_without_parameter("grouping_strategy"), STRATEGY),
    "queue-empty-parameters": (_queue_with(parameters={}), STRATEGY),
    "queue-merges-two": (_queue_parameters(max_entries_to_merge=2), MAX_MERGE),
    "queue-merges-three": (_queue_parameters(max_entries_to_merge=3), MAX_MERGE),
    "queue-merges-a-hundred": (
        _queue_parameters(max_entries_to_merge=100, min_entries_to_merge=1),
        MAX_MERGE,
    ),
    "queue-merges-zero": (_queue_parameters(max_entries_to_merge=0), MAX_MERGE),
    "queue-merges-true": (_queue_parameters(max_entries_to_merge=True), MAX_MERGE),
    "queue-merges-one-as-float": (
        _queue_parameters(max_entries_to_merge=1.0),
        MAX_MERGE,
    ),
    "queue-merges-one-as-string": (
        _queue_parameters(max_entries_to_merge="1"),
        MAX_MERGE,
    ),
    "queue-merges-null": (_queue_parameters(max_entries_to_merge=None), MAX_MERGE),
    "queue-merges-list": (_queue_parameters(max_entries_to_merge=[1]), MAX_MERGE),
    "queue-without-max-merge": (
        _queue_without_parameter("max_entries_to_merge"),
        MAX_MERGE,
    ),
    "queue-minimum-two": (_queue_parameters(min_entries_to_merge=2), MIN_MERGE),
    "queue-minimum-zero": (_queue_parameters(min_entries_to_merge=0), MIN_MERGE),
    "queue-minimum-true": (_queue_parameters(min_entries_to_merge=True), MIN_MERGE),
    "queue-minimum-null": (_queue_parameters(min_entries_to_merge=None), MIN_MERGE),
    "queue-without-minimum": (
        _queue_without_parameter("min_entries_to_merge"),
        MIN_MERGE,
    ),
    "queue-builds-two": (_queue_parameters(max_entries_to_build=2), BUILD),
    "queue-builds-five-as-recorded": (_queue_parameters(max_entries_to_build=5), BUILD),
    "queue-builds-zero": (_queue_parameters(max_entries_to_build=0), BUILD),
    "queue-builds-true": (_queue_parameters(max_entries_to_build=True), BUILD),
    "queue-builds-one-as-string": (_queue_parameters(max_entries_to_build="1"), BUILD),
    "queue-builds-null": (_queue_parameters(max_entries_to_build=None), BUILD),
    "queue-without-build": (_queue_without_parameter("max_entries_to_build"), BUILD),
    "queue-headgreen-of-three": (
        _queue_parameters(grouping_strategy="HEADGREEN", max_entries_to_merge=3),
        STRATEGY,
    ),
    "queue-recorded-parameters-of-axiom-encode": (
        _queue_with(parameters=dict(RECORDED_QUEUE["parameters"])),
        BUILD,
    ),
    "queue-unknown-parameter": (_queue_parameters(future_parameter=1), QUEUE_PARAMETER),
    "queue-unknown-null-parameter": (
        _queue_parameters(future_parameter=None),
        QUEUE_PARAMETER,
    ),
    "queue-parameter-name-case": (
        _queue_parameters(Grouping_Strategy="HEADGREEN"),
        QUEUE_PARAMETER,
    ),
    "queue-group-size-under-another-name": (
        _queue_parameters(max_group_size=5),
        QUEUE_PARAMETER,
    ),
    # A second queue rule that could merge unjudged entries, in any position.
    "a-second-queue-headgreen-listed-last": (
        _second_queue(2, grouping_strategy="HEADGREEN"),
        STRATEGY,
    ),
    "a-second-queue-headgreen-listed-first": (
        _second_queue(1, grouping_strategy="HEADGREEN"),
        STRATEGY,
    ),
    "a-second-queue-of-two-listed-last": (
        _second_queue(2, max_entries_to_merge=2),
        MAX_MERGE,
    ),
    "a-second-queue-of-two-listed-before-the-workflows-rule": (
        _second_queue(0, max_entries_to_merge=2),
        MAX_MERGE,
    ),
    "a-second-queue-with-an-unknown-parameter": (
        _second_queue(2, bypass=True),
        QUEUE_PARAMETER,
    ),
    # Every merge queue rule of the listing is held to this, in any position.
    "a-second-queue-of-type-alone-listed-last": (
        _append({"type": "merge_queue"}),
        QUEUE_SHAPE,
    ),
    "a-second-queue-of-type-alone-listed-first": (
        _queue_before({"type": "merge_queue"}),
        QUEUE_SHAPE,
    ),
    "a-second-queue-of-another-repository-listed-last": (
        _append(RECORDED_QUEUE),
        QUEUE_SOURCE,
    ),
    "a-second-queue-of-another-repository-listed-first": (
        _queue_before(RECORDED_QUEUE),
        QUEUE_SOURCE,
    ),
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


# --- hostile listings (review of ``e2d12f5c2``) -----------------------------------

_MISSING = object()
REF_VALUES = {
    "null": None,
    "empty": "",
    "space": " ",
    "zero": 0,
    "false": False,
    "true": True,
    "list": [],
    "list-with-sha": [SHA_A],
    "dict": {},
    "dict-with-sha": {"sha": SHA_A},
    "branch": "main",
    "full-ref": "refs/heads/main",
    "recorded-sha": SHA_A,
    "float": 1.5,
}


def _hostile_listings():
    """Name -> (pages, expected refusal or None for "any refusal")."""

    cases = {}

    def one_page(name, rules, message=None):
        cases[name] = ({RULES: rules}, message)

    # ``ref`` present, in every form, with and without the recorded SHA.
    for label, value in REF_VALUES.items():
        rules = listing()
        _entry(rules)["ref"] = value
        one_page(f"ref-{label}-next-to-the-recorded-sha", rules, REF_FOLLOWING)
        rules = listing()
        del _entry(rules)["sha"]
        _entry(rules)["ref"] = value
        one_page(f"ref-{label}-without-a-sha", rules, REF_FOLLOWING)
    # Unknown keys beside a good pin.
    for key in ("ref_name", "branch", "sha256", "REF", "ref ", "commit", "tag"):
        rules = listing()
        _entry(rules)[key] = "main"
        one_page(f"entry-key-{key!r}", rules, "entry has an unknown field")
    # Duplicated workflow entries, identical and not.
    for label, second in {
        "identical": GOOD_ENTRY,
        "null-ref": GOOD_ENTRY | {"ref": None},
        "other-recorded-sha": GOOD_ENTRY | {"sha": SHA_B},
        "unrecorded-sha": GOOD_ENTRY | {"sha": "c3" * 20},
        "other-repository": GOOD_ENTRY | {"repository_id": 1},
    }.items():
        for order in ("first", "second"):
            rules = listing()
            entries = [dict(GOOD_ENTRY), dict(second)]
            rules[0]["parameters"]["workflows"] = (
                entries if order == "second" else entries[::-1]
            )
            one_page(f"duplicate-entry-{label}-{order}", rules, "it names 2$")
    # Two rules of type workflows.
    for label, changes in {
        "identical": {},
        "null-ref-entry": {"parameters": {"workflows": [GOOD_ENTRY | {"ref": None}]}},
        "repository-ruleset": {
            "ruleset_source_type": "Repository",
            "ruleset_source": "TheAxiomFoundation/rulespec-us",
        },
        "other-ruleset-id": {"ruleset_id": 1},
        "empty-parameters": {"parameters": {}},
    }.items():
        for position in (0, 1, 2):
            rules = listing()
            rules.insert(position, copy.deepcopy(rules[0]) | changes)
            one_page(f"two-workflows-rules-{label}-at-{position}", rules, TWO)
    # Repository id types.
    for label, value in {
        "string": str(REPOSITORY_ID),
        "float": float(REPOSITORY_ID),
        "true": True,
        "false": False,
        "null": None,
        "list": [REPOSITORY_ID],
        "dict": {"id": REPOSITORY_ID},
        "negative": -REPOSITORY_ID,
        "zero": 0,
        "off-by-one": REPOSITORY_ID + 1,
        "missing": _MISSING,
    }.items():
        rules = listing()
        if value is _MISSING:
            del _entry(rules)["repository_id"]
        else:
            _entry(rules)["repository_id"] = value
        one_page(f"repository-id-{label}", rules, IDENTITY)
    # ``do_not_enforce_on_create``.
    for label, value in {
        "true": True,
        "null": None,
        "zero": 0,
        "one": 1,
        "string-false": "false",
        "empty-string": "",
        "list": [],
        "dict": {},
    }.items():
        rules = listing()
        rules[0]["parameters"]["do_not_enforce_on_create"] = value
        one_page(f"not-enforced-on-create-{label}", rules, ON_CREATE)
    # Pagination: the hostile part is on the second page.
    filler = [dict(FILLER) for _ in range(98)]

    def two_pages(name, second_page, message):
        cases[name] = ({RULES: filler + listing(), RULES_2: second_page}, message)

    good = listing()
    two_pages("page-2-second-workflows-rule", [good[0]], TWO)
    two_pages(
        "page-2-second-workflows-rule-with-a-null-ref",
        [copy.deepcopy(good[0]) | {"parameters": {"workflows": [{"ref": None}]}}],
        TWO,
    )
    for label, parameters, message in (
        ("headgreen", {"grouping_strategy": "HEADGREEN"}, STRATEGY),
        ("of-two", {"max_entries_to_merge": 2}, MAX_MERGE),
        ("building-two", {"max_entries_to_build": 2}, BUILD),
        ("unknown-parameter", {"future": 1}, QUEUE_PARAMETER),
    ):
        queue = copy.deepcopy(good[1]) | {"ruleset_id": 77}
        queue["parameters"].update(parameters)
        two_pages(f"page-2-second-queue-{label}", [queue], message)
    two_pages("page-2-queue-of-type-alone", [{"type": "merge_queue"}], QUEUE_SHAPE)
    two_pages("page-2-rule-without-type", [{"ruleset_id": 1}], "has no type")
    # The pinned rule itself on page 2, with a hostile entry.
    for label, value in (("null", None), ("empty", ""), ("branch", "main")):
        rules = listing()
        _entry(rules)["ref"] = value
        cases[f"page-2-pinned-rule-with-a-{label}-ref"] = (
            {
                RULES: [dict(FILLER) for _ in range(99)] + [rules[1]],
                RULES_2: [rules[0]],
            },
            REF_FOLLOWING,
        )
    return cases


HOSTILE = _hostile_listings()


def test_the_hostile_corpus_is_large_and_its_control_passes(ruleset_on, world):
    assert len(HOSTILE) == 90
    assert sum(name.startswith("ref-") for name in HOSTILE) == 2 * len(REF_VALUES)
    assert sum(name.startswith("page-2-") for name in HOSTILE) >= 10
    # The control: the same reader accepts the unchanged listing, on one page
    # and across two.
    assert enforced(world.__getitem__) == ACCEPTED
    rules = listing()
    world[RULES] = [dict(FILLER) for _ in range(99)] + [rules[1]]
    world[RULES_2] = [rules[0]]
    assert enforced(world.__getitem__) == ACCEPTED


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_no_hostile_listing_is_accepted(ruleset_on, world, name):
    pages, message = HOSTILE[name]
    world.update(copy.deepcopy(pages))
    fetch = Recorder(world)
    with pytest.raises(AdmissionError, match=message):
        enforced(fetch)
    assert fetch.calls == sorted(pages)
    # Admission and revalidation refuse it too.
    with pytest.raises(AdmissionError, match=message):
        admission.verify_reviewed_pr(SELECTOR, fetch=Recorder(world))


QUEUE_GRID = {
    "grouping_strategy": (
        _MISSING,
        "ALLGREEN",
        "HEADGREEN",
        "allgreen",
        "",
        None,
        True,
        1,
        ["ALLGREEN"],
    ),
    "max_entries_to_merge": (_MISSING, 1, 0, 2, 3, 100, True, 1.0, "1", None, [1]),
    "min_entries_to_merge": (_MISSING, 1, 0, 2, True, 1.0, "1", None),
    "max_entries_to_build": (_MISSING, 1, 0, 2, 5, True, 1.0, "1", None),
}


def _exactly(value, wanted):
    return value is not _MISSING and type(value) is type(wanted) and value == wanted


def test_the_lock_accepts_exactly_one_queue_configuration(ruleset_on, world):
    """Every combination of the four required parameters, 7128 queues.

    The lock accepts a queue only when it merges every entry alone
    (``max_entries_to_merge`` and ``min_entries_to_merge`` exactly 1), on its
    own passing checks (``ALLGREEN``), with a build concurrency of 1. So no
    accepted queue is one in which GitHub's documented rules merge a pull
    request whose own merge-group check did not pass: that needs
    ``HEADGREEN`` and a group of two or more.
    """

    names = list(QUEUE_GRID)
    accepted = []
    total = 0
    for values in itertools.product(*QUEUE_GRID.values()):
        total += 1
        rules = listing()
        for name, value in zip(names, values):
            if value is _MISSING:
                del rules[1]["parameters"][name]
            else:
                rules[1]["parameters"][name] = value
        world[RULES] = rules
        try:
            result = enforced(world.__getitem__)
        except AdmissionError as refusal:
            assert "a merge queue rule of main can merge a pull request" in str(refusal)
            continue
        assert result == ACCEPTED
        accepted.append(dict(zip(names, values)))
    assert total == 9 * 11 * 8 * 9 == 7128
    assert accepted == [dict(admission._MERGE_QUEUE_REQUIRED)]
    (only,) = accepted
    assert all(
        _exactly(only[name], wanted) for name, wanted in admission._MERGE_QUEUE_REQUIRED
    )
    assert only["grouping_strategy"] == "ALLGREEN" and only["max_entries_to_merge"] == 1


def test_a_second_queue_rule_is_held_to_the_same_configuration(ruleset_on, world):
    # 2 strategies x 3 sizes for each of two queue rules, in both orders: only
    # the pair in which BOTH are the required configuration is accepted.
    options = [
        {"grouping_strategy": strategy, "max_entries_to_merge": size}
        for strategy in ("ALLGREEN", "HEADGREEN")
        for size in (1, 2, 3)
    ]
    accepted = 0
    for first, second in itertools.product(options, repeat=2):
        rules = listing()
        rules[1]["parameters"].update(first)
        other = copy.deepcopy(listing()[1]) | {"ruleset_id": 9}
        other["parameters"].update(second)
        rules.append(other)
        world[RULES] = rules
        good = first == second == options[0]
        if good:
            assert enforced(world.__getitem__) == ACCEPTED
            accepted += 1
        else:
            with pytest.raises(AdmissionError, match="can merge a pull request"):
                enforced(world.__getitem__)
    assert accepted == 1


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


def test_the_queue_rule_must_name_the_repository_that_is_checked(ruleset_on, world):
    """Review round 1, P2-1: "this repository" is the argument, not a constant.
    The same listing is read for another RuleSpec repository: the queue rule of
    ``rulespec-us`` does not count there, and that repository's own does."""
    other = "TheAxiomFoundation/rulespec-uk"
    path = f"/repos/{other}/rules/branches/main?per_page=100&page=1"
    world[path] = listing()
    fetch = Recorder(world)
    with pytest.raises(AdmissionError, match=QUEUE_SOURCE):
        admission.require_guard_enforced(other, fetch=fetch)
    assert fetch.calls == [path]
    world[path][1]["ruleset_source"] = other
    assert admission.require_guard_enforced(other, fetch=world.__getitem__) == ACCEPTED
    # And the reverse: that listing does not satisfy the lock for rulespec-us.
    world[RULES] = copy.deepcopy(world[path])
    with pytest.raises(AdmissionError, match=QUEUE_SOURCE):
        enforced(world.__getitem__)


def test_a_true_queue_ruleset_id_is_not_ruleset_one(ruleset_on, world):
    world[RULES][1]["ruleset_id"] = 1
    assert enforced(world.__getitem__) == ACCEPTED
    world[RULES][1]["ruleset_id"] = True
    with pytest.raises(AdmissionError, match=QUEUE_SOURCE):
        enforced(world.__getitem__)


def test_the_recorded_live_queue_rule_is_well_formed_for_its_own_repository(
    ruleset_on, world
):
    """The only recorded live rule (the queue of axiom-encode) passes the
    queue half when the source it names is the repository that is checked:
    here with nothing but ``ruleset_source`` changed."""
    world[RULES][1] = copy.deepcopy(RECORDED_QUEUE) | {
        "ruleset_source": SELECTOR.repository
    }
    assert world[RULES][1]["ruleset_id"] == 26573200
    # Its rule shape and source are accepted; its parameters are not: the
    # queue of axiom-encode builds and merges up to 5 entries.
    with pytest.raises(AdmissionError, match=BUILD):
        enforced(world.__getitem__)
    world[RULES][1]["parameters"] |= {
        "max_entries_to_build": 1,
        "max_entries_to_merge": 1,
    }
    assert enforced(world.__getitem__) == ACCEPTED


def test_the_limits_of_the_queue_half_are_stated_where_the_lock_is_described():
    """Review round 1, P2-1 (ii), P3-1, P3-5: the text, not the behaviour."""

    def flat(path):
        text = (REPOSITORY_ROOT / path).read_text("utf-8").replace("#", " ")
        return " ".join(text.replace(">", " ").split())

    source = flat("src/axiom_encode/reviewed_pr_admission.py")
    design = flat("docs/content-bound-merge-guard.md")
    quoted = (
        '"This rule is not available for rulesets created at the organization level."'
    )
    for text in (source, design):
        assert quoted in text
        assert "edit, disable or give a bypass list" in text
        assert "present at each of its reads" in text
        assert "ONLY evidence" in text
    # The module header (comment) and the docstring both carry the limit.
    assert source.count("present at each of its reads") == 2
    assert "A bypass entry that exists before the dispatch is invisible" in " ".join(
        source.replace("a bypass entry", "A bypass entry").split()
    )
    # The activation conditions name both bypass lists and say "once".
    assert "the bypass list of the repository ruleset that holds the merge queue" in (
        design
    )
    assert "This confirmation happens once" in design


def test_the_queue_safety_argument_is_the_final_tree_binding_and_no_more():
    """Review of ``e2d12f5c2``, blocking 1: the text, where the lock is
    described. The earlier text said that a group of two or more pull requests
    always fails at the guard and that no queue parameter matters."""

    def flat(path):
        text = (REPOSITORY_ROOT / path).read_text("utf-8").replace("#", " ")
        return " ".join(text.replace(">", " ").split())

    source = flat("src/axiom_encode/reviewed_pr_admission.py")
    design = flat("docs/content-bound-merge-guard.md")
    for text in (source, design):
        # The withdrawn claims are gone.
        assert "parameters are NOT checked" not in text
        assert "parameters are not checked" not in text
        assert "has another tree or another base and fails at the guard" not in text
        assert "has another tree or another base and fails" not in text
        assert "no value lets a merge pass" not in text
        assert "no parameter value lets a merge pass" not in text
        # What the evaluation proves, and what it does not.
        assert "does not prove that the group holds only that pull request" in (
            text.replace("It does not", "does not")
        )
        assert "has read only the second pull request" in text
        assert "519603227" in text
        assert "979124f10" not in text
        assert "experiment X5" in text
        # What safety requires, and that the lock refuses the rest.
        assert "own fresh passing judgment at merge time" in text
        for name, wanted in admission._MERGE_QUEUE_REQUIRED:
            assert name in text, name
    assert "FINAL TREE" in source and "**final tree**" in design
    assert (
        "EVERY queue entry that merges must have its own fresh passing judgment "
        "at merge time"
    ) in source
    assert (
        "**every queue entry that merges must have its own fresh passing "
        "judgment at merge time**"
    ) in design
    assert "A model run, not evidence about GitHub" in design


def test_the_ruleset_activation_conditions_cite_2287_and_its_experiments():
    """Review of ``e2d12f5c2``, blocking 1, second part."""

    text = (REPOSITORY_ROOT / "docs/content-bound-merge-guard.md").read_text("utf-8")
    start = text.index("If the maintainers accept the ruleset mechanism instead")
    end = text.index("What fixes the order.", start)
    section = " ".join(text[start:end].split())
    assert "are in **#2287**" in section
    assert "`docs/merge-guard-ruleset-experiment.md`" in section
    # The document on main recommends the App mechanism; the section says so
    # and does not send the reader there for the ruleset steps.
    assert "is the App-based mechanism (b), not the ruleset" in section
    assert "as that proposal describes" not in section
    assert "X14 alone is not enough" in section
    for needed in (
        "**The pin (X1 of #2287; X5 for the merge group).**",
        "**A missing run blocks (X8 of #2287, parts (a) and (b)).**",
        "**A dismissal and the queue (X15 of #2287, parts (a), (b) and (c)).**",
        "must have the result `pass`",
        "only merge non-failing pull requests (`ALLGREEN`), minimum and maximum "
        "group size 1, build concurrency 1",
    ):
        assert needed in section, needed
    assert X14 in section


def test_the_ref_rule_is_stated_as_presence_where_the_lock_is_described():
    """Review of ``e2d12f5c2``, blocking 2: the text follows the code."""

    def flat(path):
        text = (REPOSITORY_ROOT / path).read_text("utf-8").replace("#", " ")
        return " ".join(text.split())

    source = flat("src/axiom_encode/reviewed_pr_admission.py")
    design = flat("docs/content-bound-merge-guard.md")
    assert "An entry with a ``ref`` that is not null refuses" not in source
    assert "An entry that HAS a ``ref`` key refuses, whatever its value" in source
    assert (
        "no `ref` key (a `ref` key refuses whatever its value, `null` included"
    ) in design
    for text in (source, design):
        assert "optional string that is not nullable" in text
    code = Path(admission.__file__).read_text()
    assert code.count('if "ref" in entry:') == 1
    assert 'entry.get("ref")' not in code


def test_the_zero_read_claim_names_require_guard_enforced(world):
    """Review of ``e2d12f5c2``, non-blocking: with neither form recorded the
    lock function reads nothing; admission reads the signing run first."""

    fragment = (
        REPOSITORY_ROOT / "changelog.d/2172-guard-ruleset-lock.added.md"
    ).read_text()
    assert "the lane refuses before any read" not in fragment
    assert (
        "`require_guard_enforced` refuses before any read of its own (admission "
        "reads the signing run first and makes no guard read)"
    ) in fragment
    design = " ".join(
        (REPOSITORY_ROOT / "docs/content-bound-merge-guard.md").read_text().split()
    )
    assert "the lane refuses before any read" not in design
    assert (
        "A reviewed-lane run makes no guard read at admission or at revalidation"
        in (design)
    )
    source = " ".join(Path(admission.__file__).read_text().replace("#", " ").split())
    assert "the lane refuses before any read" not in source
    # The behaviour the sentences describe, with the committed constants and
    # lock 1 open.
    fetch = Recorder(world)
    with pytest.raises(AdmissionError):
        enforced(fetch)
    assert fetch.calls == []


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


class _LyingRepository:
    """Prints as ``rulespec-us`` and compares equal to every value."""

    name = "TheAxiomFoundation/rulespec-us"

    def __str__(self):
        return self.name

    def __format__(self, spec):
        return self.name

    def __eq__(self, other):
        return True

    def __hash__(self):
        return hash(self.name)


def test_ruleset_form_refuses_a_repository_that_is_not_an_exact_str(ruleset_on, world):
    """Review round 2, P3-1: the queue rule is compared with the argument, so
    an object that only prints as the repository must not reach that compare.
    Without the type check this listing, whose queue belongs to another
    repository, is accepted."""
    world[RULES][1]["ruleset_source"] = "evil/other"
    fetch = Recorder(world)
    with pytest.raises(AdmissionError, match="invalid repository"):
        admission.require_guard_enforced(_LyingRepository(), fetch=fetch)
    assert fetch.calls == []


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


def _queue_shape_lost(api_):
    api_[RULES][1]["parameters"] = None


def _queue_moved_to_another_repository(api_):
    api_[RULES][1]["ruleset_source"] = "TheAxiomFoundation/axiom-encode"


LIVE_CHANGES += [_queue_shape_lost, _queue_moved_to_another_repository]


def test_a_replaced_queue_ruleset_of_the_same_repository_is_not_seen(admitted, world):
    """A stated limit, pinned: the record does not name the queue's ruleset.
    Another well-formed queue rule of the same repository passes revalidation."""
    world[RULES][1]["ruleset_id"] += 1
    world[RULES][1]["parameters"]["merge_method"] = "SQUASH"
    assert admission.revalidate_admission(admitted, fetch=Recorder(world)) == admitted
    # ... but not one that can merge a pull request without its own check.
    world[RULES][1]["parameters"]["grouping_strategy"] = "HEADGREEN"
    with pytest.raises(AdmissionError, match=STRATEGY):
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


# --- repeated JSON keys in the response text (independent review of 4c7da64f0) ----

GOOD_RAW = json.dumps(SYNTHETIC["response"], indent=1)
DUPLICATE = "GitHub response repeats a JSON object key$"


def _raw(old: str, new: str) -> str:
    assert GOOD_RAW.count(old) == 1, old
    return GOOD_RAW.replace(old, new)


RAW_DUPLICATES = {
    # The reviewer's two inputs: ``json.loads`` keeps the last value.
    "group-size-five-then-one": _raw(
        '"max_entries_to_merge": 1',
        '"max_entries_to_merge": 5, "max_entries_to_merge": 1',
    ),
    "queue-type-then-deletion": None,  # built below: a second, weak queue rule
    "group-size-one-then-five": _raw(
        '"max_entries_to_merge": 1',
        '"max_entries_to_merge": 1, "max_entries_to_merge": 5',
    ),
    "strategy-headgreen-then-allgreen": _raw(
        '"grouping_strategy": "ALLGREEN"',
        '"grouping_strategy": "HEADGREEN", "grouping_strategy": "ALLGREEN"',
    ),
    "same-value-twice": _raw(
        '"grouping_strategy": "ALLGREEN"',
        '"grouping_strategy": "ALLGREEN", "grouping_strategy": "ALLGREEN"',
    ),
    # Nested: the workflow entry, three levels down.
    "ref-then-absent-by-a-second-sha": _raw(
        f'"sha": "{SHA_A}"', f'"sha": "{"c3" * 20}", "sha": "{SHA_A}"'
    ),
    "repository-id-twice": _raw(
        f'"repository_id": {REPOSITORY_ID}',
        f'"repository_id": 1, "repository_id": {REPOSITORY_ID}',
    ),
    "workflows-list-twice": _raw(
        '"workflows": [', '"workflows": [{"path": "x"}], "workflows": ['
    ),
    "parameters-twice-in-the-workflows-rule": _raw(
        '"parameters": {\n   "workflows"',
        '"parameters": {"do_not_enforce_on_create": true}, "parameters": {\n   "workflows"',
    ),
    "source-type-repository-then-organization": _raw(
        '"ruleset_source_type": "Organization"',
        '"ruleset_source_type": "Repository", "ruleset_source_type": "Organization"',
    ),
    "ruleset-id-twice": _raw(
        '"ruleset_id": 4242', '"ruleset_id": 1, "ruleset_id": 4242'
    ),
    # The same key spelled with an escape is the same key.
    "escaped-spelling-of-the-same-key": _raw(
        '"max_entries_to_merge": 1',
        '"max_entries_to_merge": 5, "max_entries_to_\\u006derge": 1',
    ),
    # A repeated key inside a value that the lock otherwise ignores.
    "duplicate-in-an-unrelated-rule": GOOD_RAW[: GOOD_RAW.rindex("]")]
    + ', {"type": "deletion", "x": 1, "x": 2}]',
}
_WEAK = copy.deepcopy(SYNTHETIC["response"][1])
_WEAK["parameters"] |= {"grouping_strategy": "HEADGREEN", "max_entries_to_merge": 5}
RAW_DUPLICATES["queue-type-then-deletion"] = (
    GOOD_RAW[: GOOD_RAW.rindex("]")]
    + ", "
    + json.dumps(_WEAK).replace(
        '"type": "merge_queue"', '"type": "merge_queue", "type": "deletion"'
    )
    + "]"
)


def _serve(monkeypatch, raw: str):
    requests = []

    class Opener:
        def open(self, request, timeout):
            requests.append(request.full_url)
            return _Response(200, raw.encode())

    monkeypatch.setattr(
        admission.urllib.request, "build_opener", lambda *handlers: Opener()
    )
    return requests


@pytest.mark.parametrize("name", sorted(RAW_DUPLICATES))
def test_a_repeated_key_in_the_listing_text_refuses(ruleset_on, monkeypatch, name):
    raw = RAW_DUPLICATES[name]
    # The text is JSON, and the lenient parser would have read it.
    lenient = json.loads(raw)
    assert isinstance(lenient, list)
    requests = _serve(monkeypatch, raw)
    with pytest.raises(AdmissionError, match=DUPLICATE):
        enforced(admission.github_json)
    assert requests == ["https://api.github.com" + RULES]


def test_without_the_refusal_the_reviewers_two_texts_would_be_accepted(
    ruleset_on, monkeypatch
):
    """What the refusal is for: read with ``json.loads`` alone, the last value
    wins, and both texts satisfy the lock."""

    for name in ("group-size-five-then-one", "queue-type-then-deletion"):
        parsed = json.loads(RAW_DUPLICATES[name])
        assert enforced({RULES: parsed}.__getitem__) == ACCEPTED, name
    # The control: the same text without the repetition is accepted by the
    # production reader, so the refusals above come from the repeated key.
    requests = _serve(monkeypatch, GOOD_RAW)
    assert enforced(admission.github_json) == ACCEPTED
    assert len(requests) == 1


def test_the_unique_key_reader_replaces_only_the_shared_production_reader(
    ruleset_on, monkeypatch
):
    assert admission._SHARED_GITHUB_READER is admission.github_json
    # A reader that is merely NAMED ``github_json`` (a stand-in set on the
    # module) is an injected reader: it is called, not replaced.
    calls = []

    def stand_in(path):
        calls.append(path)
        return listing()

    monkeypatch.setattr(admission, "github_json", stand_in)
    assert enforced(admission.github_json) == ACCEPTED
    assert calls == [RULES]


def test_the_shared_reader_of_the_active_lane_is_unchanged(monkeypatch):
    """The reviewed lane's reader still parses as before (last value wins):
    this change gives the active lane no new refusal."""

    _serve(monkeypatch, '{"a": 1, "a": 2, "b": {"c": 1, "c": 3}}')
    path = "/repos/TheAxiomFoundation/rulespec-us/pulls/1"
    assert admission.github_json(path) == {"a": 2, "b": {"c": 3}}
    with pytest.raises(AdmissionError, match=DUPLICATE):
        admission._github_json_unique_keys(path)
    _serve(monkeypatch, '{"a": 1, "b": [{"c": 1}, {"c": 1}]}')
    assert admission._github_json_unique_keys(path) == {
        "a": 1,
        "b": [{"c": 1}, {"c": 1}],
    }


def test_the_limits_added_by_the_independent_review_are_stated():
    def flat(path):
        text = (REPOSITORY_ROOT / path).read_text("utf-8").replace("#", " ")
        return " ".join(text.split())

    source = flat("src/axiom_encode/reviewed_pr_admission.py")
    design = flat("docs/content-bound-merge-guard.md")
    for text in (source, design):
        assert "no merge queue in classic branch protection of" in text
        assert "which of the two governs a merge is not known" in text
        assert "max-age=60, s-maxage=60" in text
        assert "ends at the first page with fewer than 100 rules" in text
        assert "between the end of the guard's merge-group job and the merge" in text
        assert "keeps the last value" in text


def test_the_queue_sentences_are_pinned_word_for_word():
    """Independent review of ``4c7da64f0``: five rewordings survived the
    looser pins (the withdrawn claim reworded, another build concurrency, the
    section 6 sentence removed, the changelog without the settings, "may
    merge" for the missing-run experiment)."""

    def flat(path):
        text = (REPOSITORY_ROOT / path).read_text("utf-8").replace("#", " ")
        return " ".join(text.split())

    design = flat("docs/content-bound-merge-guard.md")
    source = flat("src/axiom_encode/reviewed_pr_admission.py")
    changelog = flat("changelog.d/2172-guard-ruleset-lock.added.md")
    # The withdrawn claim, in any wording: no sentence says that such a group
    # always fails.
    for text in (design, source):
        lowered = text.lower()
        for phrase in ("always fails", "always fail at", "always fail there"):
            assert lowered.count(phrase) == lowered.count("not** " + phrase) + (
                lowered.count("not " + phrase)
            ), phrase
        assert "never passes a group of two" not in lowered
    assert (
        "A group of two pull requests therefore does **not** always fail at the guard."
    ) in design
    assert "So a group of two pull requests does NOT always fail there." in source
    # The four required values, as the code has them.
    assert dict(admission._MERGE_QUEUE_REQUIRED) == {
        "grouping_strategy": "ALLGREEN",
        "max_entries_to_build": 1,
        "min_entries_to_merge": 1,
        "max_entries_to_merge": 1,
    }
    assert (
        'requires four of them, present and exact: `grouping_strategy: "ALLGREEN"`, '
        "`max_entries_to_merge: 1`, `min_entries_to_merge: 1` and "
        "`max_entries_to_build: 1`."
    ) in design
    assert (
        '``grouping_strategy`` ``"ALLGREEN"``, ``max_entries_to_merge`` ``1``, '
        "``min_entries_to_merge`` ``1`` and ``max_entries_to_build`` ``1``."
    ) in source
    for text in (design, source):
        assert not re.search(r"max_entries_to_(?:build|merge)`*:? `*[02-9]", text)
    # Section 6.
    assert (
        "In the ruleset form the writer's lock requires it, with `ALLGREEN` and a "
        "build concurrency of 1: the merge-group evaluation binds the final tree "
        "to one judged pull request and does not prove that a group holds only "
        'that pull request (§5, "Encoder").'
    ) in design
    # The changelog names the settings.
    assert (
        "with a merge queue rule of that repository's own ruleset that merges "
        "every entry alone and only on its own passing checks (`ALLGREEN`, group "
        "size 1, build concurrency 1)."
    ) in changelog
    # The three experiments, with what must hold.
    assert (
        "**A missing run blocks (X8 of 2287, parts (a) and (b)).** A head that "
        "gets no run of the required workflow (a push by a job token, `[skip "
        "ci]`) cannot merge."
    ) in design
    assert (
        "pinned to a commit SHA runs exactly the file of that commit, also after "
        "the source branch moves, for a pull request and for a `merge_group` run."
    ) in design
    assert (
        "A pull request whose approval is dismissed before or in the queue does "
        "not merge, and the queue merges only after the guard's merge-group job "
        "has ended."
    ) in design
    assert (
        "**No classic merge queue.** A `rulespec-us` administrator confirms that "
        "the classic branch protection of `main` has no merge queue."
    ) in design
    # The model is called what it is.
    assert ("This is a consistency check of the argument above and no more.") in design
    assert "16 distinct calls of the evaluator" in design
    assert "does not model the time between a judgment and the merge" in design
    # An entry, not every pull request.
    for text in (design, source):
        assert "every pull request that merges must" not in text.lower()
        assert "An entry, not every pull request" in text
