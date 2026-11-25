"""A candidate on an older main base: the switched-off base-advance rule.

``BASE_ADVANCE_ADMISSION_ENABLED`` is off on main. Three groups of tests:

- switch off: the module reads, records and refuses exactly as encoder main
  did before the switch existed. ``fixtures/admission_base_advance_main.json``
  holds the outcomes of the module at the commit this change started from; the
  switched-off module must reproduce every one of them. Recreate the file with
  ``python -m tests.test_admission_base_advance <path of that module> <output>``;
- switch on, ``main`` at the base: the same reads and results as switch off;
- switch on, ``main`` ahead of the base: one test per rule and per refusal.

Review round 1 (#2360): a new module in a jurisdiction that a unit reads is
refused, the bound in time is gone, and the last proven head of ``main`` is
written for the lane pull request.
"""

import copy
import hashlib
import importlib.util
import json
import sys
import urllib.error
from pathlib import Path

import pytest

from axiom_encode import constants, prepare_signed_backfill
from axiom_encode import reviewed_pr_admission as admission
from axiom_encode.reviewed_pr_admission import (
    AdmissionError,
    base_advance_issue,
    revalidate_admission,
    revalidate_admission_with_main,
    verify_reviewed_pr,
)
from tests.test_multi_unit_promotion_admission import (
    BASE,
    BASE_TREE,
    COMPARE,
    HEAD,
    HEAD_TREE,
    PR,
    REVIEWS,
    ROOT,
    SHARED,
    Reads,
    _single_unit_corpus,
    make_api,
    outcome,
    unit,
)

TIP = "9" * 40
TIP2 = "8" * 40
MAIN_REF = ROOT + "/git/ref/heads/main"
GOLDEN = Path(__file__).parent / "fixtures" / "admission_base_advance_main.json"

DEPENDENCY = "us-az/policies/des/faa5/medical/dependency.yaml"
TOOLCHAIN = ".axiom/toolchain.toml"
PROGRAM = "programs/us-az/snap/fy-2026.yaml"
# What main holds at the candidate's base, besides the candidate.
MAIN_FILES = (
    DEPENDENCY,
    "us/statute/7/2014/e.yaml",
    TOOLCHAIN,
    PROGRAM,
    "oracle-coverage-pending.yaml",
    "LICENSE",
    ".github/workflows/repository-checks.yml",
)
# A unit of a jurisdiction that the candidate (us-az) does not read.
OTHER_UNIT = "us-co/regulations/10-ccr-2506-1/4.407/block-09"
LANDED = (
    OTHER_UNIT + ".yaml",
    OTHER_UNIT + ".test.yaml",
    ".axiom/encoding-manifests/" + OTHER_UNIT + ".json",
)


def leaf(path, sha="1" * 40, mode="100644", kind="blob"):
    return {"path": path, "type": kind, "mode": mode, "sha": sha}


def tree_of(leaves):
    return {"truncated": False, "tree": [copy.deepcopy(entry) for entry in leaves]}


def chain_commits(base, tips):
    """First-parent commits from ``base`` to the last of ``tips``."""
    commits, parent = [], base
    for sha in tips:
        commits.append({"sha": sha, "parents": [{"sha": parent}]})
        parent = sha
    return commits


def comparison_of(base, commits):
    return {
        "status": "ahead",
        "merge_base_commit": {"sha": base},
        "ahead_by": len(commits),
        "behind_by": 0,
        "total_commits": len(commits),
        "commits": commits,
    }


def candidate(units=None):
    """A good candidate whose base holds ``MAIN_FILES``; main is at the base."""
    units = units or [unit(1)]
    api = make_api(units)
    for path in MAIN_FILES:
        api[BASE_TREE]["tree"].append(leaf(path))
        api[HEAD_TREE]["tree"].append(leaf(path))
    return api


def move_main(api, changes=(), *, tip=TIP, commits=None):
    """Move main from the base to ``tip`` through ``changes``.

    A change is ``(path, None)`` for a deletion or ``(path, entry fields)``
    for an added or changed leaf.
    """
    leaves = {entry["path"]: entry for entry in api[BASE_TREE]["tree"]}
    for path, fields in changes:
        if fields is None:
            leaves.pop(path)
        else:
            leaves[path] = leaf(path, **{"sha": "2" * 40, **fields})
    api[MAIN_REF] = {"object": {"sha": tip}}
    api[ROOT + f"/git/trees/{tip}?recursive=1"] = tree_of(leaves.values())
    api[ROOT + f"/compare/{BASE}...{tip}"] = comparison_of(
        BASE,
        commits if commits is not None else chain_commits(BASE, [tip]),
    )
    return api


def selector_of(units=None, module=admission):
    units = units or [unit(1)]
    if len(units) == 1:
        return module.PromotionSelector(
            citation=units[0][0], rulespec_path=units[0][1], **SHARED
        )
    return module.BatchPromotionSelector(units=tuple(units), **SHARED)


def admit(api, units=None, reads=None):
    return verify_reviewed_pr(selector_of(units), fetch=reads or Reads(api))


PLUMBING = [("LICENSE", {}), (".github/workflows/repository-checks.yml", {})]
LANDING = [(path, {}) for path in LANDED]


@pytest.fixture(
    autouse=True, params=[True, False], ids=["approval-required", "owner-dispatch"]
)
def approval_required(request, monkeypatch):
    """Every test runs under both admission bases."""
    monkeypatch.setattr(admission, "OWNER_APPROVAL_REQUIRED", request.param)
    return request.param


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(admission, "BASE_ADVANCE_ADMISSION_ENABLED", True)


# --- The shipped constants ----------------------------------------------------


def test_the_switch_is_off():
    """The one pin of the shipped switch value."""
    source = Path(admission.__file__).read_text()
    assert admission.BASE_ADVANCE_ADMISSION_ENABLED is False
    assert "\nBASE_ADVANCE_ADMISSION_ENABLED: Final[bool] = False\n" in source
    assert admission.MAX_BASE_ADVANCE_COMMITS == 20
    assert admission.MAX_PROMOTION_UNITS_ON_ADVANCED_BASE == 3
    # No environment variable, flag, setting or clock reads into the rule.
    assert "os.environ" not in source and "getenv" not in source
    assert "import time" not in source and "datetime" not in source


def test_the_path_lists_are_those_of_their_owners():
    """Lists repeated because this module is loaded alone (#1892)."""
    assert (
        admission.BASE_ADVANCE_PLUMBING_PREFIXES
        == prepare_signed_backfill.PUBLICATION_BASE_ADVANCE_PREFIXES
    )
    assert (
        admission.BASE_ADVANCE_PLUMBING_FILES
        == prepare_signed_backfill.PUBLICATION_BASE_ADVANCE_FILES
    )
    assert admission.BASE_ADVANCE_MODULE_ROOTS == constants.RULESPEC_ATOMIC_MODULE_ROOTS


def test_the_document_states_the_shipped_switch_and_limits():
    text = (
        Path(__file__).resolve().parents[1] / "docs/reviewed-pr-promotion.md"
    ).read_text()
    section = text.split("## A candidate on an older main base")[1].split("\n## ")[0]
    assert ("**Switched off.**" in section) is (
        admission.BASE_ADVANCE_ADMISSION_ENABLED is False
    )
    limit = admission.MAX_PROMOTION_UNITS_ON_ADVANCED_BASE
    assert f"so `k <= {limit}` ({47 + 4 * limit} reads)" in section
    assert (
        f"`MAX_BASE_ADVANCE_COMMITS` ({admission.MAX_BASE_ADVANCE_COMMITS})" in section
    )
    assert admission.BASE_ADVANCE_SCHEMA in section
    assert "### Before the switch goes on" in section


# --- Switch off: the module of encoder main -----------------------------------


def _set_main(value):
    def change(api):
        api[MAIN_REF] = value
        return api

    return change


# States of the main ref that are not a commit (review round 1, finding 6).
MALFORMED_MAIN = {
    "no-object": {},
    "object-not-a-mapping": {"object": 3},
    "sha-missing": {"object": {}},
    "sha-not-text": {"object": {"sha": 5}},
    "not-a-mapping": [],
    "null": None,
}


def _corpus():
    """Candidates and moves of main, good and bad, as ``{name: (units, api)}``."""
    target, singles = _single_unit_corpus()
    cases = {f"single/{name}": ([target], api) for name, api in singles.items()}
    cases["batch/good"] = (
        [unit(1), unit(2), unit(3)],
        make_api([unit(n) for n in (1, 2, 3)]),
    )
    cases["with-main-files"] = ([target], candidate())
    cases["advance/plumbing"] = ([target], move_main(candidate(), PLUMBING))
    cases["advance/landing"] = ([target], move_main(candidate(), LANDING))
    cases["advance/dependency"] = ([target], move_main(candidate(), [(DEPENDENCY, {})]))
    cases["advance/batch"] = (
        [unit(1), unit(2)],
        move_main(candidate([unit(1), unit(2)]), PLUMBING),
    )
    for name, value in MALFORMED_MAIN.items():
        cases[f"main/{name}"] = ([target], _set_main(value)(candidate()))
    return cases


def _later_states(api):
    """States of GitHub at a revalidation, as ``{name: api}``."""
    later = {
        "unchanged": copy.deepcopy(api),
        "plumbing": move_main(copy.deepcopy(api), PLUMBING, tip=TIP2),
        "dependency": move_main(copy.deepcopy(api), [(DEPENDENCY, {})], tip=TIP2),
        "unknown": _set_main({"object": {"sha": "f" * 40}})(copy.deepcopy(api)),
    }
    for name, value in MALFORMED_MAIN.items():
        later[f"main-{name}"] = _set_main(value)(copy.deepcopy(api))
    # Another approval and a main ref that is no commit: which one refuses.
    both = _set_main({"object": 3})(copy.deepcopy(api))
    both[REVIEWS][0]["id"] = 7
    later["other-approval-and-main-object-not-a-mapping"] = both
    return later


def _changed_records(admitted):
    """Provisioned admissions that differ from the live state, by name."""

    def other_base(payload):
        for record in payload.get("records", [payload.get("record")]):
            record["base_sha"] = TIP
        return payload

    def with_advance(payload):
        payload["base_advance"] = {
            "schema": "axiom-encode/reviewed-pr-base-advance/v1",
            "base_sha": BASE,
            "main_sha": TIP,
        }
        return payload

    return {
        "as-admitted": copy.deepcopy(admitted),
        "other-base": other_base(copy.deepcopy(admitted)),
        "with-advance": with_advance(copy.deepcopy(admitted)),
    }


def collect_outcomes(module) -> dict:
    """What ``module`` reads and returns for every case, as comparable text.

    For each case and each admission basis: the admission and its reads, and
    for a good admission its revalidation in every later state of main, for
    the admission as provisioned and for two changed records of it.
    """

    def digest(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, default=str).encode()
        ).hexdigest()

    def summary(result, reads):
        shown = result if result[0] == "error" else ("ok", digest(result[1]))
        return [list(shown), len(reads.paths), digest(reads.paths)]

    seen = {}
    shipped = module.OWNER_APPROVAL_REQUIRED
    try:
        for required in (True, False):
            module.OWNER_APPROVAL_REQUIRED = required
            for name, (units, api) in sorted(_corpus().items()):
                reads = Reads(api)
                selector = selector_of(units, module)
                result = outcome(
                    lambda: module.verify_reviewed_pr(selector, fetch=reads)
                )
                entry = {"admission": summary(result, reads)}
                if result[0] == "ok":
                    for record, payload in _changed_records(result[1]).items():
                        for state, changed in _later_states(api).items():
                            again = Reads(changed)
                            entry[f"revalidation/{record}/{state}"] = summary(
                                outcome(
                                    lambda: module.revalidate_admission(
                                        copy.deepcopy(payload), fetch=again
                                    )
                                ),
                                again,
                            )
                seen[f"{'approval' if required else 'dispatch'}/{name}"] = entry
    finally:
        module.OWNER_APPROVAL_REQUIRED = shipped
    return seen


def test_switch_off_is_the_module_of_encoder_main():
    """Every read, record and error equals the recorded outcome of main."""
    assert admission.BASE_ADVANCE_ADMISSION_ENABLED is False
    recorded = json.loads(GOLDEN.read_text())["outcomes"]
    assert collect_outcomes(admission) == recorded
    # The recorded cases include admissions, refusals and moves of main.
    assert {entry["admission"][0][0] for entry in recorded.values()} == {"ok", "error"}
    assert sum(len(entry) for entry in recorded.values()) == 306
    assert recorded["dispatch/advance/plumbing"]["admission"][0] == [
        "error",
        "AdmissionError",
        "PR base is not the current main head",
    ]
    good = recorded["dispatch/with-main-files"]
    assert good["revalidation/as-admitted/plumbing"][0] == [
        "error",
        "AdmissionError",
        "live approval or candidate differs from provisioned admission",
    ]
    # Finding 6: a record on another base refuses before the main ref is
    # looked at; with the record's own base the malformed ref is named.
    assert good["revalidation/other-base/main-object-not-a-mapping"][0] == [
        "error",
        "AdmissionError",
        "live approval or candidate differs from provisioned admission",
    ]
    assert good["revalidation/as-admitted/main-object-not-a-mapping"][0] == [
        "error",
        "AdmissionError",
        "GitHub main object is not an object",
    ]
    assert good["revalidation/with-advance/unchanged"][0] == [
        "error",
        "AdmissionError",
        "invalid provisioned admission",
    ]


def test_switch_off_refuses_a_moved_main_after_three_reads():
    reads = Reads(move_main(candidate(), PLUMBING))
    with pytest.raises(AdmissionError, match="^PR base is not the current main head$"):
        admit(None, reads=reads)
    assert len(reads.paths) == 3


def test_switch_off_revalidation_refuses_a_moved_main_after_four_reads():
    api = candidate()
    provisioned = admit(api)
    reads = Reads(move_main(api, PLUMBING))
    with pytest.raises(
        AdmissionError,
        match="^live approval or candidate differs from provisioned admission$",
    ):
        revalidate_admission(provisioned, fetch=reads)
    assert len(reads.paths) == 4


def test_switch_off_names_no_proven_main():
    api = candidate()
    provisioned = admit(api)
    assert revalidate_admission_with_main(provisioned, fetch=Reads(api)) == (
        provisioned,
        None,
    )


@pytest.mark.parametrize("value", [None, 0, 1, "True", "true", [True]])
def test_only_exactly_true_turns_the_switch_on(monkeypatch, value):
    monkeypatch.setattr(admission, "BASE_ADVANCE_ADMISSION_ENABLED", value)
    with pytest.raises(AdmissionError, match="^PR base is not the current main head$"):
        admit(move_main(candidate(), PLUMBING))
    api = candidate()
    provisioned = admit(api)
    assert revalidate_admission_with_main(provisioned, fetch=Reads(api))[1] is None


def test_switch_off_refuses_an_admission_that_records_an_advance(monkeypatch):
    monkeypatch.setattr(admission, "BASE_ADVANCE_ADMISSION_ENABLED", True)
    api = move_main(candidate(), PLUMBING)
    provisioned = admit(api)
    assert "base_advance" in provisioned
    monkeypatch.setattr(admission, "BASE_ADVANCE_ADMISSION_ENABLED", False)
    reads = Reads(api)
    with pytest.raises(AdmissionError, match="^invalid provisioned admission$"):
        revalidate_admission(provisioned, fetch=reads)
    assert reads.paths == []


# --- Switch on, main at the base: nothing changes -----------------------------


@pytest.mark.parametrize(
    "name",
    sorted(
        name
        for name in _corpus()
        if not name.startswith("advance/") and name != "single/main-moved"
    ),
)
def test_the_switch_does_not_change_an_admission_at_the_head_of_main(monkeypatch, name):
    """Same reads, same record or same error, with the switch on and off."""
    units, api = _corpus()[name]
    seen = []
    for enabled in (False, True):
        monkeypatch.setattr(admission, "BASE_ADVANCE_ADMISSION_ENABLED", enabled)
        reads = Reads(api)
        result = outcome(lambda: admit(None, units, reads=reads))
        again = Reads(api)
        revalidated = (
            outcome(lambda: revalidate_admission(result[1], fetch=again))
            if result[0] == "ok"
            else None
        )
        seen.append((result, reads.paths, revalidated, again.paths))
    if name.startswith("main/"):
        # A main ref that is not a commit refuses under both values; with the
        # switch on a text that is no commit is refused by the proof.
        assert seen[0][0][0] == seen[1][0][0] == "error"
        return
    assert seen[0] == seen[1]
    if seen[0][0][0] == "ok":
        assert "base_advance" not in seen[0][0][1]


# --- Switch on, main ahead of the base: what is admitted -----------------------


def _delta_digest(classes):
    return hashlib.sha256(
        json.dumps(sorted(classes.items()), separators=(",", ":")).encode()
    ).hexdigest()


def test_an_advance_through_plumbing_is_admitted_and_recorded(on):
    api = move_main(candidate(), PLUMBING)
    reads = Reads(api)
    result = admit(None, reads=reads)
    assert result["record"]["base_sha"] == BASE
    assert result["base_advance"] == {
        "schema": admission.BASE_ADVANCE_SCHEMA,
        "base_sha": BASE,
        "main_sha": TIP,
        "first_parent_commits": 1,
        "plumbing_paths": 2,
        "added_paths": 0,
        "delta_sha256": _delta_digest({path: "plumbing" for path, _f in PLUMBING}),
    }
    # Two reads more than an admission at the head of main.
    assert len(reads.paths) == 13 + 2
    assert reads.paths.count(ROOT + f"/compare/{BASE}...{TIP}") == 1
    assert reads.paths.count(ROOT + f"/git/trees/{TIP}?recursive=1") == 1


def test_the_record_differs_from_an_admission_at_the_head_only_by_the_advance(on):
    at_head = admit(candidate())
    advanced = admit(move_main(candidate(), PLUMBING))
    assert {**advanced, "base_advance": None} == {**at_head, "base_advance": None}


def test_units_that_landed_in_an_unrelated_jurisdiction_are_admitted(on):
    result = admit(move_main(candidate(), LANDING + PLUMBING))
    assert result["base_advance"]["added_paths"] == 3
    assert result["base_advance"]["plumbing_paths"] == 2


def test_empty_commits_of_main_are_admitted(on):
    """A commit that changes no file still moves the head of main."""
    api = move_main(candidate(), commits=chain_commits(BASE, ["7" * 40, TIP]))
    result = admit(api)
    assert result["base_advance"]["first_parent_commits"] == 2
    assert result["base_advance"]["plumbing_paths"] == 0


def test_a_merge_commit_on_the_first_parent_history_is_admitted(on):
    commits = chain_commits(BASE, [TIP])
    commits[0]["parents"].append({"sha": "6" * 40})
    side = chain_commits(BASE, ["6" * 40])
    assert admit(move_main(candidate(), LANDING, commits=side + commits))


def test_no_commit_date_is_read(on):
    """A date is set by the committer; the rule has no bound in time."""
    commits = chain_commits(BASE, [TIP])
    commits[0]["commit"] = {"committer": {"date": "1970-01-01T00:00:00Z"}}
    old = admit(move_main(candidate(), PLUMBING, commits=commits))
    assert old == admit(move_main(candidate(), PLUMBING))


def test_a_batch_within_the_limit_is_admitted(on):
    units = [unit(n) for n in (1, 2, 3)]
    result = admit(move_main(candidate(units), PLUMBING), units)
    assert result["schema"] == admission.BATCH_SCHEMA
    assert result["base_advance"]["main_sha"] == TIP
    assert {record["base_sha"] for record in result["records"]} == {BASE}


# --- Switch on: every refusal --------------------------------------------------

UNIT_DIR = "us-az/policies/des/faa5/medical"
BINDING = {
    "dependency-modified": [(DEPENDENCY, {})],
    "dependency-deleted": [(DEPENDENCY, None)],
    "federal-module-modified": [("us/statute/7/2014/e.yaml", {})],
    "toolchain-modified": [(TOOLCHAIN, {})],
    "program-modified": [(PROGRAM, {})],
    "program-added": [("programs/us-az/snap/fy-2027.yaml", {})],
    "ledger-modified": [("oracle-coverage-pending.yaml", {})],
    "waiver-added": [(".axiom/waivers/us-co/x.json", {})],
    "index-added": [(admission.INDEX_PATH, {})],
    "root-file-added": [("x.yaml", {})],
    "not-yaml-added": [("us-co/policies/notes.md", {})],
    "other-country-added": [("uk/statutes/x/y.yaml", {})],
    "other-country-manifest-added": [
        (".axiom/encoding-manifests/uk/statutes/x/y.json", {})
    ],
    "manifest-not-json-added": [
        (".axiom/encoding-manifests/us-co/statutes/x/y.yaml", {})
    ],
    "symlink-added": [(LANDED[0], {"mode": "120000"})],
    "executable-added": [(LANDED[0], {"mode": "100755"})],
    "submodule-added": [(LANDED[0], {"mode": "160000", "kind": "commit"})],
    "mode-changed": [(DEPENDENCY, {"sha": "1" * 40, "mode": "100755"})],
    "renamed": [(DEPENDENCY, None), (DEPENDENCY.replace("dependency", "moved"), {})],
    # Finding 1: a new module where a check of the unit can read it.
    "child-of-the-unit": [(unit(1)[1].removesuffix(".yaml") + "/c.yaml", {})],
    "sibling-in-the-directory": [(UNIT_DIR + "/block-77.yaml", {})],
    "sibling-companion": [(UNIT_DIR + "/block-77.test.yaml", {})],
    "same-section": [("us-az/policies/des/faa5/shelter/block-09.yaml", {})],
    "same-jurisdiction": [("us-az/statutes/ars/46/x.yaml", {})],
    "shadowing-import-path": [("us-az/policies/dependency.yaml", {})],
    "federal-module-added": [("us/statutes/7/2014/f.yaml", {})],
    "below-the-jurisdiction": [("us-az-phoenix/policies/x/y.yaml", {})],
    "manifest-in-the-jurisdiction": [
        (".axiom/encoding-manifests/us-az/policies/des/faa5/shelter/block-09.json", {})
    ],
    "federal-manifest-added": [
        (".axiom/encoding-manifests/us/statutes/7/2014/f.json", {})
    ],
    # Finding 4: only the canonical path of a module, companion or manifest.
    "toolchain-under-a-jurisdiction": [("us-co/.axiom/toolchain.yaml", {})],
    "ledger-under-a-jurisdiction": [("us-co/known-validation-gaps.yaml", {})],
    "program-under-a-jurisdiction": [("us-co/programs/snap/fy-2026.yaml", {})],
    "hidden-directory": [("us-co/policies/.hidden/x.yaml", {})],
    "hidden-file": [("us-co/policies/x/.y.yaml", {})],
    "no-module-root": [("us-co/x.yaml", {})],
    "manifest-outside-a-module-root": [
        (".axiom/encoding-manifests/us-co/programs/x.json", {})
    ],
    "jurisdiction-of-another-shape": [("us_co/policies/x/y.yaml", {})],
    "git-attributes-added": [(".gitattributes", {})],
    "workflow-toolchain-added": [(".axiom/workflow-toolchain.toml", {})],
    "manifest-in-a-legacy-place": [
        ("us-co/.axiom/encoding-manifests/policies/x/y.json", {})
    ],
}


@pytest.mark.parametrize("name", sorted(BINDING))
def test_an_advance_through_a_binding_path_is_refused_and_named(on, name):
    changes = BINDING[name]
    with pytest.raises(AdmissionError) as refusal:
        admit(move_main(candidate(), PLUMBING + LANDING + changes))
    message = str(refusal.value)
    assert message.startswith(
        "PR base is not the current main head, and main advanced through paths "
        "the candidate's validation may read: "
    )
    assert changes[-1][0] in message
    for path, _fields in PLUMBING + LANDING:
        if path not in dict(changes):
            assert path not in message


def test_a_federal_unit_admits_no_module_of_any_state(on):
    """Every state root is below ``us``."""
    federal = ("us/statutes/7/2014/e/1", "us/statutes/7/2014/e/1.yaml")
    api = make_api([federal])
    with pytest.raises(AdmissionError, match="validation may read: .*" + LANDED[0]):
        admit(move_main(api, LANDING), [federal])
    assert admit(move_main(make_api([federal]), PLUMBING), [federal])


def test_a_batch_refuses_a_landing_related_to_any_of_its_units(on):
    az, co = (
        unit(1),
        (
            "us-co/regulations/10-ccr-2506-1/4.407/block-01",
            "us-co/regulations/10-ccr-2506-1/4.407/block-01.yaml",
        ),
    )
    units = sorted([az, co], key=lambda target: target[1])
    api = make_api(units)
    nc = [("us-nc/statutes/108a/x.yaml", {})]
    assert admit(move_main(copy.deepcopy(api), nc), units)
    with pytest.raises(AdmissionError, match="validation may read: .*" + LANDED[0]):
        admit(move_main(copy.deepcopy(api), LANDING), units)


def test_a_long_list_of_binding_paths_is_shortened(on):
    changes = [(f"programs/p{index:02d}.yaml", {}) for index in range(25)]
    with pytest.raises(AdmissionError, match=r"programs/p19\.yaml \(\+5 more\)"):
        admit(move_main(candidate(), changes))


@pytest.mark.parametrize(
    "path",
    [
        unit(1)[1],
        unit(1)[1].removesuffix(".yaml") + ".test.yaml",
        ".axiom/encoding-manifests/" + unit(1)[1].removesuffix(".yaml") + ".json",
        unit(1)[1].upper(),
        unit(1)[1] + "/nested.yaml",
        "us-az/policies/des/faa5/medical",
    ],
    ids=["primary", "companion", "manifest", "other-case", "below", "above"],
)
def test_a_unit_path_that_exists_on_main_is_refused_and_named(on, path):
    with pytest.raises(AdmissionError) as refusal:
        admit(move_main(candidate(), [(path, {})]))
    message = str(refusal.value)
    assert "already exists on main, or a path of main collides with it" in message
    assert f"unit {unit(1)[0]} ({unit(1)[1]})" in message and path in message


def test_a_base_that_is_not_on_the_first_parent_history_is_refused(on):
    """The base was merged into main; it never was a head of main."""
    commits = [{"sha": TIP, "parents": [{"sha": "5" * 40}, {"sha": BASE}]}]
    with pytest.raises(AdmissionError, match="not on the first-parent history of main"):
        admit(move_main(candidate(), PLUMBING, commits=commits))


def test_the_commit_limit_admits_its_bound_and_refuses_one_more(on):
    limit = admission.MAX_BASE_ADVANCE_COMMITS

    def tips(count):
        return [f"{index:040x}" for index in range(1, count)] + [TIP]

    admitted = admit(move_main(candidate(), commits=chain_commits(BASE, tips(limit))))
    assert admitted["base_advance"]["first_parent_commits"] == limit
    with pytest.raises(AdmissionError, match=f"more than {limit} first-parent commits"):
        admit(move_main(candidate(), commits=chain_commits(BASE, tips(limit + 1))))


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "diverged"),
        ("status", "behind"),
        ("status", "identical"),
        ("merge_base_commit", {"sha": "4" * 40}),
        ("behind_by", 1),
        ("behind_by", None),
        ("total_commits", 2),
        ("total_commits", "1"),
        ("total_commits", True),
        ("ahead_by", 2),
        ("commits", []),
        ("commits", None),
    ],
)
def test_a_comparison_that_is_not_a_complete_descent_is_refused(on, field, value):
    api = move_main(candidate(), PLUMBING)
    api[ROOT + f"/compare/{BASE}...{TIP}"][field] = value
    with pytest.raises(
        AdmissionError, match="does not descend from it by a history that can be read"
    ):
        admit(api)


def test_a_comparison_with_more_commits_than_one_page_is_refused(on):
    """GitHub returns at most 250 commits; the total then exceeds the list."""
    api = move_main(candidate(), PLUMBING)
    api[ROOT + f"/compare/{BASE}...{TIP}"].update(total_commits=251, ahead_by=251)
    with pytest.raises(AdmissionError, match="can be read completely"):
        admit(api)


@pytest.mark.parametrize("fault", ["duplicate", "no-parents", "bad-sha", "bad-parent"])
def test_an_invalid_commit_of_the_advance_is_refused(on, fault):
    commits = chain_commits(BASE, ["7" * 40, TIP])
    if fault == "duplicate":
        commits[0]["sha"] = TIP
    elif fault == "no-parents":
        commits[1]["parents"] = []
    elif fault == "bad-sha":
        commits[0]["sha"] = "7" * 39
    else:
        commits[1]["parents"] = [{"sha": "main"}]
    with pytest.raises(AdmissionError, match="invalid or duplicate commit"):
        admit(move_main(candidate(), PLUMBING, commits=commits))


@pytest.mark.parametrize("sha", [None, "main", "9" * 39, 9])
def test_a_main_head_that_is_not_a_commit_is_refused(on, sha):
    api = move_main(candidate(), PLUMBING)
    api[MAIN_REF] = {"object": {"sha": sha}}
    with pytest.raises(AdmissionError, match="main head is not an exact commit"):
        admit(api)


def test_a_main_tree_that_cannot_be_read_completely_is_refused(on):
    api = move_main(candidate(), PLUMBING)
    api[ROOT + f"/git/trees/{TIP}?recursive=1"]["truncated"] = True
    with pytest.raises(AdmissionError, match="cannot be inspected completely"):
        admit(api)


def _failing(api, fragment, error):
    """A fetch that raises ``error`` for the first path that holds ``fragment``."""
    reads = Reads(api)

    def fetch(path):
        if fragment in path:
            reads.paths.append(path)
            raise error
        return reads(path)

    fetch.reads = reads
    return fetch


PROOF_READS = {
    "compare": f"/compare/{BASE}...",
    "tree": "/git/trees/" + TIP[:8],
}
API_ERRORS = {
    "rate-limit": urllib.error.HTTPError(
        "https://api.github.com", 403, "rate", {}, None
    ),
    "server": urllib.error.HTTPError("https://api.github.com", 502, "bad", {}, None),
    "network": urllib.error.URLError("timed out"),
    "not-200": AdmissionError("GitHub admission query did not succeed"),
    "missing": KeyError("not served"),
}


@pytest.mark.parametrize("read", sorted(PROOF_READS))
@pytest.mark.parametrize("error", sorted(API_ERRORS))
def test_an_api_error_in_the_proof_fails_closed(on, read, error):
    """HTTP errors and rate limits refuse; nothing is admitted from a cache."""
    fragment = PROOF_READS[read] + (TIP if read == "compare" else "")
    fetch = _failing(move_main(candidate(), PLUMBING), fragment, API_ERRORS[error])
    with pytest.raises(type(API_ERRORS[error])):
        verify_reviewed_pr(selector_of(), fetch=fetch)
    # The failed read is the last one: nothing is tried again or skipped.
    assert fragment in fetch.reads.paths[-1]


@pytest.mark.parametrize(
    "read", ["/git/trees/" + BASE, f"/compare/{BASE}...{TIP2}", TIP2 + "?"]
)
@pytest.mark.parametrize("error", sorted(API_ERRORS))
def test_an_api_error_in_the_proof_of_a_revalidation_fails_closed(on, read, error):
    api = candidate()
    provisioned = admit(api)
    moved = move_main(api, PLUMBING, tip=TIP2)
    fetch = _failing(moved, read, API_ERRORS[error])
    with pytest.raises(type(API_ERRORS[error])):
        revalidate_admission_with_main(provisioned, fetch=fetch)
    assert read in fetch.reads.paths[-1]


def test_a_candidate_that_does_not_descend_from_its_base_is_still_refused(on):
    api = move_main(candidate(), PLUMBING)
    api[COMPARE]["merge_base_commit"]["sha"] = "d" * 40
    with pytest.raises(
        AdmissionError, match="^candidate must descend from its main base$"
    ):
        admit(api)


def test_the_scope_of_the_candidate_is_still_its_delta_from_the_base(on):
    """A candidate that also changes a file of main is refused as before."""
    api = move_main(candidate(), PLUMBING)
    api[HEAD_TREE]["tree"] = [
        {**entry, "sha": "3" * 40} if entry["path"] == DEPENDENCY else entry
        for entry in api[HEAD_TREE]["tree"]
    ]
    with pytest.raises(AdmissionError, match="file scope differs"):
        admit(api)


def test_a_unit_that_exists_at_the_base_is_refused_before_the_proof(on):
    api = move_main(candidate(), PLUMBING)
    api[BASE_TREE]["tree"].append(leaf(unit(1)[1]))
    reads = Reads(api)
    with pytest.raises(AdmissionError, match="already exist on main"):
        admit(None, reads=reads)
    assert not any(TIP in path for path in reads.paths)


def test_the_veto_still_refuses_on_an_advanced_base(on):
    api = move_main(candidate(), PLUMBING)
    api[REVIEWS].append(
        {
            "id": 2,
            "state": "CHANGES_REQUESTED",
            "commit_id": HEAD,
            "user": {"id": 5, "login": "someone", "type": "User"},
        }
    )
    with pytest.raises(AdmissionError, match="unresolved changes requested"):
        admit(api)


def test_a_batch_over_the_limit_of_an_advanced_base_is_refused(on):
    limit = admission.MAX_PROMOTION_UNITS_ON_ADVANCED_BASE
    units = [unit(index) for index in range(1, limit + 2)]
    api = candidate(units)
    assert "base_advance" not in admit(api, units)
    reads = Reads(move_main(api, PLUMBING))
    with pytest.raises(
        AdmissionError,
        match=f"at most {limit} units \\(this one names {limit + 1}\\)",
    ):
        admit(None, units, reads=reads)
    # Refused before the two reads of the proof.
    assert not any(TIP in path for path in reads.paths)


@pytest.mark.parametrize("limit", [None, True, "3", 3.0])
def test_a_limit_that_is_not_an_integer_refuses(on, monkeypatch, limit):
    monkeypatch.setattr(admission, "MAX_PROMOTION_UNITS_ON_ADVANCED_BASE", limit)
    with pytest.raises(AdmissionError, match="may carry at most"):
        admit(move_main(candidate(), PLUMBING))


# --- Switch on: revalidation, and the head of main it proves -------------------


def test_revalidation_reads_nothing_more_while_main_stands(on):
    api = move_main(candidate(), PLUMBING)
    provisioned = admit(api)
    reads = Reads(api)
    assert revalidate_admission(provisioned, fetch=reads) == provisioned
    assert len(reads.paths) == 4


def test_revalidation_at_the_base_names_the_base_as_the_proven_main(on):
    api = candidate()
    provisioned = admit(api)
    reads = Reads(api)
    admitted, proof = revalidate_admission_with_main(provisioned, fetch=reads)
    assert admitted == provisioned and len(reads.paths) == 4
    assert proof == {
        "schema": admission.BASE_ADVANCE_SCHEMA,
        "base_sha": BASE,
        "main_sha": BASE,
        "first_parent_commits": 0,
        "plumbing_paths": 0,
        "added_paths": 0,
        "delta_sha256": _delta_digest({}),
    }


def test_revalidation_while_main_stands_names_the_recorded_main(on):
    api = move_main(candidate(), PLUMBING)
    provisioned = admit(api)
    proof = revalidate_admission_with_main(provisioned, fetch=Reads(api))[1]
    assert proof == provisioned["base_advance"] and proof["main_sha"] == TIP


def test_revalidation_of_a_record_read_back_from_json(on):
    api = move_main(candidate(), PLUMBING + LANDING)
    provisioned = json.loads(json.dumps(admit(api)))
    assert revalidate_admission(provisioned, fetch=Reads(api)) == provisioned


@pytest.mark.parametrize("start", ["at-head", "advanced"])
def test_revalidation_proves_a_further_move_and_names_the_new_main(on, start):
    """Finding 3: the proof that is returned is the last one, not the first."""
    api = candidate() if start == "at-head" else move_main(candidate(), PLUMBING)
    provisioned = admit(api)
    commits = chain_commits(BASE, [TIP, TIP2])
    reads = Reads(move_main(api, PLUMBING + LANDING, tip=TIP2, commits=commits))
    admitted, proof = revalidate_admission_with_main(provisioned, fetch=reads)
    assert admitted == provisioned
    assert proof == {
        "schema": admission.BASE_ADVANCE_SCHEMA,
        "base_sha": BASE,
        "main_sha": TIP2,
        "first_parent_commits": 2,
        "plumbing_paths": 2,
        "added_paths": 3,
        "delta_sha256": _delta_digest(
            {
                **{path: "plumbing" for path, _f in PLUMBING},
                **{path: "added" for path in LANDED},
            }
        ),
    }
    # The four reads of every revalidation, and three for the proof.
    assert len(reads.paths) == 7
    assert reads.paths[4:] == [
        BASE_TREE,
        ROOT + f"/compare/{BASE}...{TIP2}",
        ROOT + f"/git/trees/{TIP2}?recursive=1",
    ]


@pytest.mark.parametrize("start", ["at-head", "advanced"])
@pytest.mark.parametrize(
    "change", [DEPENDENCY, UNIT_DIR + "/block-77.yaml", "us/statutes/7/2014/f.yaml"]
)
def test_revalidation_refuses_a_further_move_through_a_binding_path(on, start, change):
    api = candidate() if start == "at-head" else move_main(candidate(), PLUMBING)
    provisioned = admit(api)
    moved = move_main(
        api, [(change, {})], tip=TIP2, commits=chain_commits(BASE, [TIP, TIP2])
    )
    with pytest.raises(AdmissionError, match="validation may read: " + change):
        revalidate_admission(provisioned, fetch=Reads(moved))


def test_revalidation_refuses_a_unit_path_that_landed_meanwhile(on):
    api = candidate()
    provisioned = admit(api)
    moved = move_main(api, [(unit(1)[1], {})])
    with pytest.raises(AdmissionError, match="already exists on main"):
        revalidate_admission(provisioned, fetch=Reads(moved))


def test_revalidation_refuses_a_main_that_left_the_recorded_history(on):
    """Main was moved to a commit that does not descend from the base."""
    api = move_main(candidate(), PLUMBING)
    provisioned = admit(api)
    moved = move_main(api, PLUMBING, tip=TIP2)
    moved[ROOT + f"/compare/{BASE}...{TIP2}"]["status"] = "diverged"
    with pytest.raises(AdmissionError, match="does not descend from it"):
        revalidate_admission(provisioned, fetch=Reads(moved))


def test_revalidation_refuses_a_main_that_went_back_to_the_base(on):
    api = move_main(candidate(), PLUMBING)
    provisioned = admit(api)
    api[MAIN_REF] = {"object": {"sha": BASE}}
    with pytest.raises(AdmissionError, match="differs from provisioned admission"):
        revalidate_admission(provisioned, fetch=Reads(api))


@pytest.mark.parametrize("value", sorted(MALFORMED_MAIN))
def test_revalidation_refuses_a_main_ref_that_is_not_a_commit(on, value):
    api = candidate()
    provisioned = admit(api)
    api[MAIN_REF] = MALFORMED_MAIN[value]
    with pytest.raises(AdmissionError):
        revalidate_admission(provisioned, fetch=Reads(api))


def test_revalidation_refuses_a_changed_pr_base(on):
    api = move_main(candidate(), PLUMBING)
    provisioned = admit(api)
    api[PR]["base"]["sha"] = TIP
    with pytest.raises(AdmissionError, match="differs from provisioned admission"):
        revalidate_admission(provisioned, fetch=Reads(api))


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "axiom-encode/reviewed-pr-base-advance/v0"},
        {"base_sha": TIP2},
        {"main_sha": BASE},
        {"main_sha": "main"},
        {"main_sha": None},
        {"main_moved_at": 1},
        {"extra": 1},
    ],
)
def test_revalidation_refuses_a_malformed_record_of_the_advance(on, change):
    api = move_main(candidate(), PLUMBING)
    provisioned = admit(api)
    provisioned["base_advance"].update(change)
    with pytest.raises(AdmissionError, match="^invalid provisioned base advance$"):
        revalidate_admission(provisioned, fetch=Reads(api))


@pytest.mark.parametrize("value", [None, [], "advance", 1])
def test_revalidation_refuses_an_advance_that_is_not_an_object(on, value):
    api = move_main(candidate(), PLUMBING)
    provisioned = admit(api)
    provisioned["base_advance"] = value
    reads = Reads(api)
    with pytest.raises(AdmissionError, match="^invalid provisioned base advance$"):
        revalidate_admission(provisioned, fetch=reads)
    assert reads.paths == []


def test_revalidation_refuses_a_batch_over_the_limit_after_main_moved(on):
    limit = admission.MAX_PROMOTION_UNITS_ON_ADVANCED_BASE
    units = [unit(index) for index in range(1, limit + 2)]
    api = candidate(units)
    provisioned = admit(api, units)
    assert revalidate_admission(provisioned, fetch=Reads(api)) == provisioned
    reads = Reads(move_main(api, PLUMBING))
    with pytest.raises(AdmissionError, match=f"at most {limit} units"):
        revalidate_admission(provisioned, fetch=reads)


# --- The proven head of main reaches the lane pull request ---------------------


@pytest.fixture
def publication(monkeypatch, tmp_path):
    """Run ``verify-publication --main-proof`` on an admission and an API."""
    path = (
        Path(__file__).resolve().parents[1] / "scripts/prepare_reviewed_pr_promotion.py"
    )
    spec = importlib.util.spec_from_file_location("_publication_recheck", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    def run(provisioned, api):
        loaded = script.load("reviewed_pr_admission")
        for name in ("OWNER_APPROVAL_REQUIRED", "BASE_ADVANCE_ADMISSION_ENABLED"):
            setattr(loaded, name, getattr(admission, name))
        reads = Reads(api)
        loaded.github_json = reads
        loaded.revalidate_admission_with_main.__kwdefaults__["fetch"] = reads
        loaded.revalidate_admission.__kwdefaults__["fetch"] = reads
        monkeypatch.setattr(script, "load", lambda _name: loaded)
        attestation = tmp_path / "runtime-attestation.json"
        attestation.write_text(json.dumps({"reviewed_pr_admission": provisioned}))
        proof = tmp_path / "main-proof.json"
        monkeypatch.setattr(
            "sys.argv",
            [
                "prepare",
                "verify-publication",
                "--attestation",
                str(attestation),
                "--main-proof",
                str(proof),
            ],
        )
        script.main()
        return proof

    return run


def test_the_publication_check_writes_no_proof_while_the_switch_is_off(publication):
    api = candidate()
    assert not publication(admit(api), api).exists()


def test_the_publication_check_writes_the_last_proven_main(on, publication):
    api = candidate()
    provisioned = admit(api)
    moved = move_main(api, PLUMBING, tip=TIP2)
    proof = json.loads(publication(provisioned, moved).read_text())
    assert (proof["base_sha"], proof["main_sha"]) == (BASE, TIP2)
    assert proof["plumbing_paths"] == 2


def test_the_publication_check_refuses_and_writes_nothing(on, publication, tmp_path):
    api = candidate()
    provisioned = admit(api)
    moved = move_main(api, [(DEPENDENCY, {})], tip=TIP2)
    with pytest.raises(ValueError, match="validation may read"):
        publication(provisioned, moved)
    assert not (tmp_path / "main-proof.json").exists()


def test_the_workflow_puts_the_proven_main_on_the_lane_pull_request():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github/workflows/targeted-signed-reencode.yml"
    ).read_text()
    recheck = workflow.split(
        "- name: Recheck live reviewed PR authorization before publication"
    )[1].split("\n      - name: ")[0]
    assert '--main-proof "$main_proof"' in recheck
    assert 'rm -f "$main_proof"' in recheck and "GITHUB_STEP_SUMMARY" in recheck
    # The body gains the line only when the proof file exists (switch on).
    publish = workflow.split('gh api --method POST "repos/${repo}/pulls"')[0]
    block = publish.rsplit('main_proof="$RUNNER_TEMP/reviewed-pr-main-proof.json"', 1)[
        1
    ]
    assert block.lstrip().startswith('if test -s "$main_proof"; then')
    assert "Proven main head: `" in block and ".main_sha" in block
    assert workflow.count("reviewed-pr-main-proof.json") == 2


# --- The request budget --------------------------------------------------------


def _reads_of_one_run_on_an_advanced_base(count: int) -> int:
    """GitHub reads of one signing run whose main moves before every check.

    Two admissions on an advanced base, and three revalidations that each
    find main moved again.
    """
    units = [unit(index) for index in range(1, count + 1)]
    api = move_main(candidate(units), PLUMBING)
    first = Reads(api)
    provisioned = admit(None, units, reads=first)
    moved = move_main(
        copy.deepcopy(api), PLUMBING, tip=TIP2, commits=chain_commits(BASE, [TIP, TIP2])
    )
    again = Reads(moved)
    revalidate_admission(provisioned, fetch=again)
    assert len(first.paths) == 13 + 2 * count
    assert len(again.paths) == 7
    return 2 * len(first.paths) + 3 * len(again.paths)


def test_the_limit_is_the_anonymous_request_budget_of_one_run(on, monkeypatch):
    limit = admission.MAX_PROMOTION_UNITS_ON_ADVANCED_BASE
    monkeypatch.setattr(admission, "MAX_PROMOTION_UNITS_ON_ADVANCED_BASE", limit + 1)
    budget = admission.ANONYMOUS_REQUESTS_PER_HOUR
    assert _reads_of_one_run_on_an_advanced_base(limit) == 47 + 4 * limit == 59
    assert _reads_of_one_run_on_an_advanced_base(limit) <= budget
    assert _reads_of_one_run_on_an_advanced_base(limit + 1) > budget
    assert limit <= admission.MAX_PROMOTION_UNITS


# --- The pure rule -------------------------------------------------------------


def _entries(paths):
    return {path: leaf(path) for path in paths}


def test_the_rule_names_every_class_of_a_differing_path():
    units = (selector_of(),)
    base = _entries(MAIN_FILES)
    tip = {**base, **{path: leaf(path, sha="2" * 40) for path in ("LICENSE", *LANDED)}}
    assert base_advance_issue(base, tip, units) == (
        None,
        {"LICENSE": "plumbing", **{path: "added" for path in LANDED}},
    )
    assert base_advance_issue(base, dict(base), units) == (None, {})


@pytest.mark.parametrize(
    "root,admitted",
    [
        ("us-co", True),
        ("us-nc", True),
        ("us-a", True),
        ("us-az", False),
        ("us", False),
        ("us-az-phoenix", False),
        ("usa", False),
        ("uk", False),
        ("US-CO", False),
    ],
)
def test_the_rule_admits_a_new_module_only_in_an_unrelated_jurisdiction(root, admitted):
    """Own root, the roots above it and the roots below it are read by checks."""
    units = (selector_of(),)
    base = _entries(MAIN_FILES)
    for path in (
        f"{root}/statutes/x/y.yaml",
        f"{root}/policies/x/y.test.yaml",
        f".axiom/encoding-manifests/{root}/regulations/x/y.json",
    ):
        issue, _classes = base_advance_issue(base, {**base, path: leaf(path)}, units)
        assert (issue is None) is admitted, path


def test_the_rule_is_independent_of_what_a_validation_reads():
    """Every non-plumbing file of the base must be the same object."""
    units = (selector_of(),)
    base = _entries(MAIN_FILES)
    for path in MAIN_FILES:
        changed = {**base, path: leaf(path, sha="2" * 40)}
        issue, _classes = base_advance_issue(base, changed, units)
        assert (issue is None) is admission._is_base_advance_plumbing(path)


# --- The recorded outcomes of main ---------------------------------------------


def _write_recorded_outcomes(module_path: str, output: str) -> None:
    spec = importlib.util.spec_from_file_location("_admission_of_main", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    Path(output).write_text(
        json.dumps(
            {
                "comment": (
                    "Outcomes of reviewed_pr_admission.py as it was on encoder "
                    "main before the base-advance switch existed. See "
                    "tests/test_admission_base_advance.py."
                ),
                "outcomes": collect_outcomes(module),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    _write_recorded_outcomes(sys.argv[1], sys.argv[2])
