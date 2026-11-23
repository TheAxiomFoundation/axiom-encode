"""Admission of one unit out of a multi-unit ("batch") candidate. OFF.

Issue #2301, option A; design: ``docs/content-bound-merge-guard.md``, section
5, "Multi-unit candidates". ``verify_reviewed_pr`` admits one unit of a
candidate pull request that carries further complete new units only with
every lock of the v6 writer open AND ``MULTI_UNIT_V6_ENABLED`` on. Both are
off in the committed code; tests patch the constants.
"""

from __future__ import annotations

import copy
import random

import pytest

from axiom_encode import content_merge_guard as guard
from axiom_encode import reviewed_pr_admission as admission
from axiom_encode.reviewed_pr_admission import AdmissionError
from tests.test_candidate_pr_admission import (  # noqa: F401 (fixtures)
    Recorder,
    on,
    outcome,
    world,
)
from tests.test_reviewed_pr_admission import (  # noqa: F401 (fixture)
    BASE,
    HEAD,
    PR,
    ROOT,
    SELECTOR,
    api,
)

FILES = PR + "/files?per_page=100&page=1"
COMPARE = ROOT + f"/compare/{BASE}...{HEAD}"
HEAD_TREE = ROOT + f"/git/trees/{HEAD}?recursive=1"
BASE_TREE = ROOT + f"/git/trees/{BASE}?recursive=1"
OUT_OF_SCOPE = "candidate PR changes files outside its new module scope"
NO_UNIT = "batch candidate changes a file that belongs to no promotion unit"
INCOMPLETE = "batch candidate holds an incomplete sibling unit"
OWN_MANIFEST = "batch candidate already holds a manifest for this unit"
TOO_MANY = "batch candidate holds more units than the batch limit"
INDEX = admission.INDEX_PATH


def stem(n):
    return f"us-az/policies/des/faa5/medical/block-{n}"


def P(n):
    return stem(n) + ".yaml"


def T(n):
    return stem(n) + ".test.yaml"


def M(n):
    return guard.MANIFEST_PREFIX + stem(n) + ".json"


OWN_M = guard.MANIFEST_PREFIX + SELECTOR.rulespec_path.removesuffix(".yaml") + ".json"


def add(fake, *paths, status="added", entry=None):
    """Add changed files to the candidate, consistently in every listing."""
    for path in paths:
        item = {"filename": path, "status": status}
        fake[FILES].append(dict(item))
        fake[COMPARE]["files"].append(dict(item))
        if status != "removed":
            fake[HEAD_TREE]["tree"].append(
                {"path": path, "type": "blob", "mode": "100644", "sha": "d" * 40}
                | (entry or {})
            )
        if status != "added":
            fake[BASE_TREE]["tree"].append(
                {"path": path, "type": "blob", "mode": "100644", "sha": "9" * 40}
            )
    fake[PR]["changed_files"] = len(fake[FILES])
    return fake


def sibling(n, manifest=False):
    return (P(n), T(n)) + ((M(n),) if manifest else ())


@pytest.fixture
def batch_on(on, monkeypatch):  # noqa: F811
    """Every lock of the v6 writer open and the multi-unit switch on."""
    monkeypatch.setattr(admission, "MULTI_UNIT_V6_ENABLED", True)
    monkeypatch.setattr(admission, "MAX_V6_UNITS", 10)


def admit(fake):
    return admission.verify_reviewed_pr(SELECTOR, fetch=fake.__getitem__)


# --- off: nothing changes ----------------------------------------------------


def test_off_a_batch_candidate_is_refused_in_the_reviewed_lane(api):  # noqa: F811
    add(api, *sibling(8))
    with pytest.raises(AdmissionError, match=OUT_OF_SCOPE):
        admit(api)


def test_writer_on_and_multi_unit_off_refuses_a_batch_candidate(on, world):  # noqa: F811
    add(world, *sibling(8))
    with pytest.raises(AdmissionError, match=OUT_OF_SCOPE):
        admit(world)


def test_multi_unit_on_alone_admits_no_batch_in_the_reviewed_lane(monkeypatch, api):  # noqa: F811
    """Without the v6 writer the switch does nothing: the reviewed lane keeps
    its single-unit scope, with the same reads and the same error."""
    add(api, *sibling(8))
    off = outcome(admit, api)
    recorder_off, recorder_on = Recorder(api), Recorder(api)
    outcome(admission.verify_reviewed_pr, SELECTOR, fetch=recorder_off)
    monkeypatch.setattr(admission, "MULTI_UNIT_V6_ENABLED", True)
    assert outcome(admit, api) == off == ("AdmissionError", OUT_OF_SCOPE)
    outcome(admission.verify_reviewed_pr, SELECTOR, fetch=recorder_on)
    assert recorder_on.calls == recorder_off.calls


@pytest.mark.parametrize("value", [1, "true", [True], None, 0])
def test_only_an_exact_true_switch_admits_a_batch_candidate(
    on,  # noqa: F811
    world,  # noqa: F811
    monkeypatch,
    value,
):
    monkeypatch.setattr(admission, "MULTI_UNIT_V6_ENABLED", value)
    add(world, *sibling(8))
    with pytest.raises(AdmissionError, match=OUT_OF_SCOPE):
        admit(world)


# --- a batch of one is a single unit --------------------------------------------


def _nothing(fake):
    return None


def _index_added(fake):
    add(fake, INDEX)


def _index_modified(fake):
    add(fake, INDEX, status="modified")


def _workflow(fake):
    add(fake, ".github/workflows/x.yml")


def _companion_missing(fake):
    for listing in (fake[FILES], fake[COMPARE]["files"]):
        listing[:] = [item for item in listing if item["filename"] != T(7)]
    fake[HEAD_TREE]["tree"] = [
        entry for entry in fake[HEAD_TREE]["tree"] if entry["path"] != T(7)
    ]
    fake[PR]["changed_files"] = 1


def _primary_on_main(fake):
    fake[BASE_TREE]["tree"].append(
        {"path": P(7), "type": "blob", "mode": "100644", "sha": "9" * 40}
    )


def _duplicate_file(fake):
    fake[FILES].append({"filename": P(7), "status": "added"})
    fake[PR]["changed_files"] = 3


def _moved_main(fake):
    fake[ROOT + "/git/ref/heads/main"] = {"object": {"sha": "9" * 40}}


def _own_manifest(fake):
    add(fake, OWN_M)


def _readme(fake):
    add(fake, "README.md")


def _ledger_and_index(fake):
    add(fake, "known-validation-gaps.yaml", INDEX)


def _nameless_file(fake):
    fake[FILES].append({"status": "added"})
    fake[PR]["changed_files"] = len(fake[FILES])


@pytest.mark.parametrize(
    "mutate",
    [
        _nothing,
        _index_added,
        _index_modified,
        _workflow,
        _companion_missing,
        _primary_on_main,
        _duplicate_file,
        _moved_main,
        _own_manifest,
        _readme,
        _ledger_and_index,
        _nameless_file,
    ],
)
def test_a_single_unit_candidate_is_judged_the_same_with_the_switch_on(
    on,  # noqa: F811
    world,  # noqa: F811
    monkeypatch,
    mutate,
):
    """Same result or error text, and the same reads in the same order."""
    assert SELECTOR.rulespec_path == P(7)
    mutate(world)
    off_reads, on_reads = Recorder(world), Recorder(world)
    off = outcome(admission.verify_reviewed_pr, SELECTOR, fetch=off_reads)
    monkeypatch.setattr(admission, "MULTI_UNIT_V6_ENABLED", True)
    assert outcome(admission.verify_reviewed_pr, SELECTOR, fetch=on_reads) == off
    assert on_reads.calls == off_reads.calls


# --- on: one unit of a batch ------------------------------------------------------


def test_a_unit_of_a_batch_of_unsigned_pairs_is_admitted(batch_on, world):  # noqa: F811
    single = admit(copy.deepcopy(world))
    add(world, *sibling(8), *sibling(9))
    result = admit(world)
    # The record binds this run's own unit only; siblings leave no trace.
    assert result == single
    assert result["schema"] == admission.CANDIDATE_SCHEMA
    assert result["record"]["binding"]["rulespec_path"] == P(7)


def test_siblings_that_earlier_runs_already_signed_are_admitted(batch_on, world):  # noqa: F811
    add(world, *sibling(8, manifest=True), *sibling(9), *sibling(10, manifest=True))
    assert admit(world)["record"]["binding"]["citation"] == SELECTOR.citation


@pytest.mark.parametrize(
    "path",
    [
        P(8),
        T(8),
        M(8),
        INDEX,
        "README.md",
        ".github/workflows/x.yml",
        ".axiom/toolchain.toml",
        "known-validation-gaps.yaml",
        "programs/us-az/snap/fy-2026.yaml",
        "us-az/policies/des/faa5/medical/notes.md",
        "us-az/manual/des/faa5/medical/block-8.yaml",
        "us-az/policies/.hidden/block-8.yaml",
        "us-az/policies/des//block-8.yaml",
        "policies.yaml",
        "newroot/policies/x.yaml",
        "US-AZ/policies/x.yaml",
        "us-az/.axiom/encoding-manifests/policies/des/faa5/medical/block-8.json",
        guard.MANIFEST_PREFIX + "us-az/policies/des/faa5/medical/block-8.sig",
        guard.MANIFEST_PREFIX + "us-az/manual/x.json",
        "us-az/policies/des/faa5/medical/block-8.test.test.yaml",
    ],
)
def test_a_path_that_completes_no_sibling_unit_refuses(batch_on, world, path):  # noqa: F811
    add(world, *sibling(9), path)
    with pytest.raises(AdmissionError) as refusal:
        admit(world)
    member = admission._batch_unit_member(path)
    assert str(refusal.value) == (NO_UNIT if member is None else INCOMPLETE)


@pytest.mark.parametrize(
    "paths,message",
    [
        ((P(8),), INCOMPLETE),
        ((T(8),), INCOMPLETE),
        ((M(8),), INCOMPLETE),
        ((P(8), M(8)), INCOMPLETE),
        ((T(8), M(8)), INCOMPLETE),
        ((*sibling(8), P(9)), INCOMPLETE),
        ((*sibling(8), INDEX), NO_UNIT),
        ((*sibling(8), ".github/workflows/x.yml"), NO_UNIT),
        ((OWN_M,), OUT_OF_SCOPE),
        ((*sibling(8), OWN_M), OWN_MANIFEST),
    ],
)
def test_each_batch_refusal_has_its_own_words(batch_on, world, paths, message):  # noqa: F811
    add(world, *paths)
    with pytest.raises(AdmissionError) as refusal:
        admit(world)
    assert str(refusal.value) == message


def test_the_index_is_admitted_only_beside_a_single_unit(batch_on, world):  # noqa: F811
    """The reviewed lane's optional index stays for a candidate with no
    sibling; a batch may not carry it (the merge guard refuses it anyway)."""
    alone = add(copy.deepcopy(world), INDEX)
    assert admit(alone)["schema"] == admission.CANDIDATE_SCHEMA
    add(world, INDEX, *sibling(8))
    with pytest.raises(AdmissionError, match=NO_UNIT):
        admit(world)


@pytest.mark.parametrize("status", ["modified", "removed", "renamed"])
@pytest.mark.parametrize("member", [P, T, M])
def test_a_sibling_file_that_is_not_new_refuses(batch_on, world, member, status):  # noqa: F811
    others = [path for path in sibling(8, manifest=True) if path != member(8)]
    add(world, *others)
    add(world, member(8), status=status)
    with pytest.raises(AdmissionError, match=OUT_OF_SCOPE):
        admit(world)


@pytest.mark.parametrize(
    "entry",
    [
        {"mode": "120000"},
        {"mode": "100755"},
        {"type": "commit", "mode": "160000"},
    ],
)
@pytest.mark.parametrize("member", [P, T, M])
def test_a_sibling_file_that_is_not_a_regular_blob_refuses(
    batch_on,
    world,  # noqa: F811
    member,
    entry,
):
    others = [path for path in sibling(8, manifest=True) if path != member(8)]
    add(world, *others)
    add(world, member(8), entry=entry)
    with pytest.raises(AdmissionError, match="not a regular blob"):
        admit(world)


def test_a_sibling_hidden_from_the_file_list_refuses(batch_on, world):  # noqa: F811
    add(world, *sibling(8))
    world[HEAD_TREE]["tree"].append(
        {
            "path": ".github/workflows/x.yml",
            "type": "blob",
            "mode": "100644",
            "sha": "d" * 40,
        }
    )
    with pytest.raises(AdmissionError, match="complete immutable tree delta"):
        admit(world)


def test_the_batch_limit_counts_this_unit_and_its_siblings(batch_on, world):  # noqa: F811
    limit = admission.MAX_V6_UNITS
    full = copy.deepcopy(world)
    for n in range(100, 100 + limit - 1):
        add(full, *sibling(n))
    assert admit(full)["schema"] == admission.CANDIDATE_SCHEMA
    add(full, *sibling(999))
    with pytest.raises(AdmissionError, match=TOO_MANY):
        admit(full)


def test_the_committed_limit_admits_one_sibling_and_refuses_two(
    on,  # noqa: F811
    world,  # noqa: F811
    monkeypatch,
):
    monkeypatch.setattr(admission, "MULTI_UNIT_V6_ENABLED", True)
    assert admission.MAX_V6_UNITS == 2
    add(world, *sibling(8))
    assert admit(world)["schema"] == admission.CANDIDATE_SCHEMA
    add(world, *sibling(9))
    with pytest.raises(AdmissionError, match=TOO_MANY):
        admit(world)


def test_the_index_beside_siblings_is_outside_the_allowed_set_too(
    batch_on,
    world,  # noqa: F811
    monkeypatch,
):
    """Defence in depth: were the sibling rule to let the index through, the
    allowed set of the caller still has no index beside siblings."""
    add(world, INDEX, *sibling(8))
    monkeypatch.setattr(
        admission, "_batch_sibling_paths", lambda paths, selector: frozenset(sibling(8))
    )
    with pytest.raises(AdmissionError, match=OUT_OF_SCOPE):
        admit(world)


@pytest.mark.parametrize("limit", [True, "10", None, 10.0, 1, 0])
def test_a_limit_that_is_not_a_large_enough_integer_refuses(
    batch_on,
    world,  # noqa: F811
    monkeypatch,
    limit,
):
    monkeypatch.setattr(admission, "MAX_V6_UNITS", limit)
    add(world, *sibling(8))
    with pytest.raises(AdmissionError, match=TOO_MANY):
        admit(world)


def test_a_duplicate_entry_in_the_file_list_refuses_a_batch(batch_on, world):  # noqa: F811
    add(world, *sibling(8))
    world[FILES].append({"filename": P(8), "status": "added"})
    world[PR]["changed_files"] = len(world[FILES])
    with pytest.raises(AdmissionError, match=OUT_OF_SCOPE):
        admit(world)


def test_a_file_entry_without_a_name_refuses_a_batch(batch_on, world):  # noqa: F811
    add(world, *sibling(8))
    world[FILES].append({"status": "added"})
    world[PR]["changed_files"] = len(world[FILES])
    with pytest.raises(AdmissionError, match=NO_UNIT):
        admit(world)


def test_the_lock_of_the_guard_is_still_read_for_a_batch(batch_on, world):  # noqa: F811
    add(world, *sibling(8))
    fetch = Recorder(world)
    admission.verify_reviewed_pr(SELECTOR, fetch=fetch)
    branch = ROOT + "/branches/main"
    assert fetch.calls[1] == branch and fetch.calls[-1] == branch
    world[branch]["protection"]["required_status_checks"]["checks"] = []
    with pytest.raises(AdmissionError):
        admit(world)


def test_changes_requested_still_blocks_a_batch(batch_on, world):  # noqa: F811
    add(world, *sibling(8))
    world[PR + "/reviews?per_page=100&page=1"] = [
        {
            "id": 5,
            "state": "CHANGES_REQUESTED",
            "commit_id": HEAD,
            "user": {"id": 6076111, "login": "MaxGhenis", "type": "User"},
        }
    ]
    with pytest.raises(AdmissionError, match="unresolved changes requested"):
        admit(world)


@pytest.mark.parametrize(
    "paths",
    [
        sibling(8),  # an unsigned sibling pair
        sibling(8, manifest=True),  # its manifest is not read here
        ("zz/policies/a.yaml", "zz/policies/a.test.yaml"),  # a new jurisdiction
        ("statutes/a.yaml", "statutes/a.test.yaml"),  # the root layout
    ],
)
def test_review_cases_admission_signs_beside_complete_sibling_units(
    batch_on,
    world,  # noqa: F811
    paths,
):
    """Independent review of cc09b513a, P2-1: with the switch on, admission
    signs one unit beside sibling units that nobody has judged. This is the
    design. It cannot know which siblings will land. The guard covers it:
    ``test_review_case_a_unit_signed_beside_a_sibling_that_is_dropped`` in
    ``tests/test_multi_unit_v6_lane.py`` (a unit that was validated beside a
    file the head does not land never passes while the switch is on)."""
    add(world, *paths)
    assert admit(world)["schema"] == admission.CANDIDATE_SCHEMA


# --- the path rule, directly --------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        (P(8), (stem(8), "primary")),
        (T(8), (stem(8), "companion")),
        (M(8), (stem(8), "manifest")),
        ("statutes/7/2014.yaml", ("statutes/7/2014", "primary")),
        (
            "us/regulations/7-cfr/273/9.test.yaml",
            ("us/regulations/7-cfr/273/9", "companion"),
        ),
        ("us-az-x1/legislation/a.yaml", ("us-az-x1/legislation/a", "primary")),
        ("policies.yaml", None),
        ("statutes.yaml", None),
        ("us/policies.yaml", None),
        ("us/manual/a.yaml", None),
        ("programs/us/snap.yaml", None),
        ("us/policies/a.yml", None),
        ("us/policies/a.json", None),
        ("us/policies/.a.yaml", None),
        ("us/policies//a.yaml", None),
        ("/us/policies/a.yaml", None),
        ("us/policies/a.test.test.yaml", None),
        ("USA/policies/a.yaml", None),
        ("u/policies/a.yaml", None),
        (guard.MANIFEST_PREFIX + "us/policies/a.yaml", None),
        (guard.MANIFEST_PREFIX + "us/policies/a.test.json", None),
        (guard.MANIFEST_PREFIX + "programs/a.json", None),
        ("x/" + guard.MANIFEST_PREFIX + "us/policies/a.json", None),
        # As long as the manifest prefix, and not it.
        ("a" * (len(guard.MANIFEST_PREFIX) - 1) + "/us/policies/a.json", None),
        ("", None),
    ],
)
def test_which_paths_can_belong_to_a_promotion_unit(path, expected):
    assert admission._batch_unit_member(path) == expected


def test_a_unit_member_of_admission_is_a_canonical_unit_path_of_the_guard():
    """The admission rule is path-only and never wider than the guard's
    module shape: every primary and companion it accepts has that shape, and
    its manifest is at the one position the guard's v6 lane accepts."""
    rng = random.Random(7)
    segments = ["us", "us-az", "policies", "statutes", "manual", ".x", "a", "b-1", ""]
    suffixes = [".yaml", ".test.yaml", ".json", ".yml", ".test.test.yaml", ""]
    accepted = 0
    for _ in range(20000):
        parts = [rng.choice(segments) for _ in range(rng.randint(1, 5))]
        path = "/".join(parts) + rng.choice(suffixes)
        if rng.random() < 0.3:
            path = guard.MANIFEST_PREFIX + path
        member = admission._batch_unit_member(path)
        if member is None:
            continue
        accepted += 1
        unit_stem, role = member
        assert guard._has_module_shape(unit_stem + ".yaml"), path
        assert guard._has_module_shape(unit_stem + ".test.yaml"), path
        assert (
            path
            == {
                "primary": unit_stem + ".yaml",
                "companion": unit_stem + ".test.yaml",
                "manifest": guard.MANIFEST_PREFIX + unit_stem + ".json",
            }[role]
        )
    assert accepted > 200
