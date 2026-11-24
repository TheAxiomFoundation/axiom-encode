"""Several new units in one candidate: the admission half (#2301).

``MULTI_UNIT_PROMOTION_ENABLED`` is off on main. These tests turn it on with
``monkeypatch`` and prove each rule of the batch form, and prove that the
single-unit form does not change with either value.
"""

import base64
import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from axiom_encode import reviewed_pr_admission as admission
from axiom_encode.reviewed_pr_admission import (
    AdmissionError,
    BatchPromotionSelector,
    PromotionSelector,
    revalidate_admission,
    selector_from_payload,
    verify_reviewed_pr,
)

HEAD = "a" * 40
BASE = "b" * 40
ROOT = "/repos/TheAxiomFoundation/rulespec-us"
PR = ROOT + "/pulls/42"
RUN = "/repos/TheAxiomFoundation/axiom-encode/actions/runs/12345"
FILES = PR + "/files?per_page=100&page=1"
REVIEWS = PR + "/reviews?per_page=100&page=1"
COMPARE = ROOT + f"/compare/{BASE}...{HEAD}"
BASE_TREE = ROOT + f"/git/trees/{BASE}?recursive=1"
HEAD_TREE = ROOT + f"/git/trees/{HEAD}?recursive=1"
PAVEL = {"id": 110687043, "login": "PavelMakarchuk", "type": "User"}
SHARED = {
    "country": "us",
    "pr": 42,
    "head": HEAD,
    "branch": "codex/batch",
    "actor": "PavelMakarchuk",
    "run_id": "12345",
}


def unit(index: int) -> tuple[str, str]:
    """``(citation, primary path)`` of the unit with this number."""
    return (
        f"us-az/manual/des/faa5/medical/block-{index:02d}",
        f"us-az/policies/des/faa5/medical/block-{index:02d}.yaml",
    )


def companion(path: str) -> str:
    return path.removesuffix(".yaml") + ".test.yaml"


def unit_files(units) -> list[str]:
    return [p for _c, path in units for p in (path, companion(path))]


def blob_sha(path: str) -> str:
    return hashlib.sha1(path.encode()).hexdigest()


def batch(units) -> BatchPromotionSelector:
    return BatchPromotionSelector(units=tuple(units), **SHARED)


def single(target) -> PromotionSelector:
    return PromotionSelector(citation=target[0], rulespec_path=target[1], **SHARED)


def make_api(units, *, extra=()) -> dict:
    """The GitHub state of a candidate that adds these units and ``extra``.

    ``extra`` is ``(path, status)`` for further changed files. An ``added``
    or ``modified`` one is in the head tree; a ``modified`` or ``removed``
    one is in the base tree.
    """
    files = [(path, "added") for path in unit_files(units)] + list(extra)
    listing = [{"filename": path, "status": status} for path, status in files]

    def leaf(path, sha):
        return {"path": path, "type": "blob", "mode": "100644", "sha": sha}

    api = {
        RUN: {
            "id": 12345,
            "event": "workflow_dispatch",
            "path": ".github/workflows/targeted-signed-reencode.yml",
            "head_branch": "main",
            "head_sha": "e" * 40,
            "status": "in_progress",
            "actor": dict(PAVEL),
            "triggering_actor": dict(PAVEL),
        },
        PR: {
            "number": 42,
            "state": "open",
            "merged": False,
            "changed_files": len(files),
            "head": {
                "sha": HEAD,
                "ref": SHARED["branch"],
                "repo": {"full_name": "TheAxiomFoundation/rulespec-us"},
            },
            "base": {
                "sha": BASE,
                "ref": "main",
                "repo": {"full_name": "TheAxiomFoundation/rulespec-us"},
            },
            "user": {"id": 110687043},
        },
        ROOT + "/git/ref/heads/main": {"object": {"sha": BASE}},
        COMPARE: {
            "status": "ahead",
            "merge_base_commit": {"sha": BASE},
            "files": copy.deepcopy(listing),
        },
        BASE_TREE: {
            "truncated": False,
            "tree": [
                leaf(path, "d" * 40)
                for path, status in files
                if status in {"modified", "removed"}
            ],
        },
        ROOT + "/git/ref/heads/codex%2Fbatch": {"object": {"sha": HEAD}},
        FILES: listing,
        REVIEWS: [
            {
                "id": 1,
                "state": "APPROVED",
                "commit_id": HEAD,
                "user": {"id": 6076111, "login": "MaxGhenis", "type": "User"},
            }
        ],
        HEAD_TREE: {
            "truncated": False,
            "tree": [
                leaf(path, blob_sha(path))
                for path, status in files
                if status in {"added", "modified"}
            ],
        },
    }
    for path in unit_files(units):
        raw = ("fixture " + path).encode()
        api[f"{ROOT}/contents/{path}?ref={HEAD}"] = {
            "type": "file",
            "path": path,
            "encoding": "base64",
            "content": base64.b64encode(raw).decode(),
            "size": len(raw),
            "sha": blob_sha(path),
        }
    return api


class Reads:
    """A fetch that records every path it serves."""

    def __init__(self, api):
        self.api = api
        self.paths = []

    def __call__(self, path):
        self.paths.append(path)
        return copy.deepcopy(self.api[path])


def outcome(call):
    """A comparable result: the value, or the error's type and text."""
    try:
        return ("ok", call())
    except Exception as exc:  # noqa: BLE001 - the comparison is the point
        return ("error", type(exc).__name__, str(exc))


@pytest.fixture(
    autouse=True, params=[True, False], ids=["approval-required", "owner-dispatch"]
)
def approval_required(request, monkeypatch):
    """Every test runs under both admission bases."""
    monkeypatch.setattr(admission, "OWNER_APPROVAL_REQUIRED", request.param)
    return request.param


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(admission, "MULTI_UNIT_PROMOTION_ENABLED", True)


UNITS3 = [unit(1), unit(2), unit(3)]


# --- The shipped constants ----------------------------------------------------


def test_the_switch_is_off_and_the_limit_is_six():
    source = Path(admission.__file__).read_text()
    assert admission.MULTI_UNIT_PROMOTION_ENABLED is False
    assert "\nMULTI_UNIT_PROMOTION_ENABLED: Final[bool] = False\n" in source
    assert admission.MAX_PROMOTION_UNITS == 6
    assert "\nMAX_PROMOTION_UNITS: Final[int] = 6\n" in source
    # No environment variable, flag or setting reads the switch.
    assert "os.environ" not in source and "getenv" not in source


# --- Switch off: the batch forms do not exist ---------------------------------


def test_switch_off_refuses_a_batch_selector_payload_as_before():
    payload = batch(UNITS3).payload()
    with pytest.raises(AdmissionError, match="promotion selector has unexpected"):
        selector_from_payload(payload)


def test_switch_off_refuses_a_batch_selector_without_any_read():
    reads = Reads(make_api(UNITS3))
    with pytest.raises(AdmissionError, match="multi-unit promotion is switched off"):
        verify_reviewed_pr(batch(UNITS3), fetch=reads)
    assert reads.paths == []


@pytest.mark.parametrize("value", [None, 0, 1, "True", "true", [True]])
def test_only_exactly_true_turns_the_switch_on(monkeypatch, value):
    monkeypatch.setattr(admission, "MULTI_UNIT_PROMOTION_ENABLED", value)
    reads = Reads(make_api(UNITS3))
    with pytest.raises(AdmissionError, match="switched off"):
        verify_reviewed_pr(batch(UNITS3), fetch=reads)
    with pytest.raises(AdmissionError, match="unexpected fields"):
        selector_from_payload(batch(UNITS3).payload())
    assert reads.paths == []


def test_switch_off_refuses_a_batch_admission_as_before(monkeypatch):
    monkeypatch.setattr(admission, "MULTI_UNIT_PROMOTION_ENABLED", True)
    api = make_api(UNITS3)
    provisioned = verify_reviewed_pr(batch(UNITS3), fetch=Reads(api))
    monkeypatch.setattr(admission, "MULTI_UNIT_PROMOTION_ENABLED", False)
    reads = Reads(api)
    with pytest.raises(AdmissionError, match="^invalid provisioned admission$"):
        revalidate_admission(provisioned, fetch=reads)
    assert reads.paths == []


@pytest.mark.parametrize("enabled", [False, True])
def test_a_single_unit_selector_never_admits_a_candidate_of_two_units(
    monkeypatch, enabled
):
    """k > 1 without the batch form is refused exactly as today, on or off."""
    monkeypatch.setattr(admission, "MULTI_UNIT_PROMOTION_ENABLED", enabled)
    with pytest.raises(
        AdmissionError,
        match="^candidate PR changes files outside its new module scope$",
    ):
        verify_reviewed_pr(single(unit(1)), fetch=Reads(make_api([unit(1), unit(2)])))


def _single_unit_corpus():
    """Single-unit candidates, good and bad, for the on/off comparison."""
    target = unit(1)
    cases = {"good": make_api([target])}
    cases["index"] = make_api([target], extra=[(admission.INDEX_PATH, "added")])
    cases["two-units"] = make_api([target, unit(2)])
    cases["extra"] = make_api([target], extra=[("README.md", "added")])
    cases["modified"] = make_api([target], extra=[("us/x.yaml", "modified")])
    cases["removed"] = make_api([target], extra=[("us/x.yaml", "removed")])
    moved = make_api([target])
    moved[ROOT + "/git/ref/heads/main"]["object"]["sha"] = "f" * 40
    cases["main-moved"] = moved
    veto = make_api([target])
    veto[REVIEWS].append(
        {
            "id": 2,
            "state": "CHANGES_REQUESTED",
            "commit_id": HEAD,
            "user": {"id": 5, "login": "someone", "type": "User"},
        }
    )
    cases["veto"] = veto
    no_companion = make_api([target])
    no_companion[FILES].pop()
    no_companion[PR]["changed_files"] = 1
    cases["no-companion"] = no_companion
    return target, cases


@pytest.mark.parametrize("name", sorted(_single_unit_corpus()[1]))
def test_the_switch_does_not_change_a_single_unit_admission(monkeypatch, name):
    """Same reads, same record or same error, with the switch on and off."""
    target, cases = _single_unit_corpus()
    seen = []
    for enabled in (False, True):
        monkeypatch.setattr(admission, "MULTI_UNIT_PROMOTION_ENABLED", enabled)
        reads = Reads(cases[name])
        result = outcome(lambda: verify_reviewed_pr(single(target), fetch=reads))
        again = Reads(cases[name])
        revalidated = (
            outcome(lambda: revalidate_admission(result[1], fetch=again))
            if result[0] == "ok"
            else None
        )
        seen.append((result, reads.paths, revalidated, again.paths))
    assert seen[0] == seen[1]
    assert (seen[0][0][0] == "ok") is (name in {"good", "index"})


def test_single_unit_admission_keeps_its_exact_shape(approval_required):
    result = verify_reviewed_pr(single(unit(1)), fetch=Reads(make_api([unit(1)])))
    evidence = "approvals" if approval_required else "dispatcher"
    assert list(result) == [
        "schema",
        "workflow_commit",
        "selector",
        "admission_basis",
        evidence,
        "record",
    ]
    assert result["schema"] == "axiom-encode/live-reviewed-pr-admission/v2"
    assert list(result["selector"]) == [
        "country",
        "pr",
        "head",
        "citation",
        "rulespec_path",
        "branch",
        "actor",
        "run_id",
    ]
    assert list(result["record"]) == [
        "schema",
        "country",
        "rulespec_ref",
        "rulespec_pr",
        "pr_base_branch",
        "base_sha",
        "binding",
        "declared",
        "purpose",
    ]


# --- Switch on: what a batch admits -------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 6])
def test_a_batch_admits_one_record_per_unit_equal_to_a_single_admission(on, count):
    """Each record is what a single-unit promotion of that unit records."""
    units = [unit(index) for index in range(1, count + 1)]
    result = verify_reviewed_pr(batch(units), fetch=Reads(make_api(units)))
    assert result["schema"] == admission.BATCH_SCHEMA
    assert result["selector"] == batch(units).payload()
    assert "record" not in result
    assert len(result["records"]) == count
    for target, record in zip(units, result["records"]):
        alone = verify_reviewed_pr(single(target), fetch=Reads(make_api([target])))
        assert record == alone["record"]
        assert json.dumps(record) == json.dumps(alone["record"])
        assert record["binding"]["rulespec_sha256"] == (
            hashlib.sha256(("fixture " + target[1]).encode()).hexdigest()
        )
        for field in ("workflow_commit", "admission_basis"):
            assert result[field] == alone[field]
        evidence = admission._BASIS_EVIDENCE_FIELD[result["admission_basis"]]
        assert result[evidence] == alone[evidence]


def test_a_batch_of_one_reads_what_a_single_admission_reads(on):
    target = unit(1)
    in_batch, alone = Reads(make_api([target])), Reads(make_api([target]))
    verify_reviewed_pr(batch([target]), fetch=in_batch)
    verify_reviewed_pr(single(target), fetch=alone)
    assert in_batch.paths == alone.paths


def test_the_limit_admits_max_units_and_refuses_one_more(on):
    limit = admission.MAX_PROMOTION_UNITS
    units = [unit(index) for index in range(1, limit + 2)]
    assert (
        len(
            verify_reviewed_pr(
                batch(units[:limit]), fetch=Reads(make_api(units[:limit]))
            )["records"]
        )
        == limit
    )
    reads = Reads(make_api(units))
    with pytest.raises(AdmissionError, match=f"names {limit + 1} units; the limit is"):
        verify_reviewed_pr(batch(units), fetch=reads)
    assert reads.paths == []
    with pytest.raises(AdmissionError, match="the limit is"):
        selector_from_payload(batch(units).payload())


@pytest.mark.parametrize("limit", [None, "6", 6.0, True])
def test_a_limit_that_is_not_an_integer_refuses(on, monkeypatch, limit):
    monkeypatch.setattr(admission, "MAX_PROMOTION_UNITS", limit)
    with pytest.raises(AdmissionError, match="the limit is"):
        verify_reviewed_pr(batch([unit(1)]), fetch=Reads(make_api([unit(1)])))


@pytest.mark.parametrize(
    "extra",
    [
        ("README.md", "added"),
        (".github/workflows/ci.yml", "added"),
        (".axiom/encoding-manifests/us-az/policies/x.json", "added"),
        ("us-az/policies/des/faa5/medical/other.yaml", "added"),
    ],
)
def test_an_extra_file_is_refused_by_name(on, extra):
    with pytest.raises(AdmissionError, match=re.escape(extra[0]) + ".*no unit named"):
        verify_reviewed_pr(batch(UNITS3), fetch=Reads(make_api(UNITS3, extra=[extra])))


@pytest.mark.parametrize("status", ["modified", "removed", "renamed"])
def test_a_changed_or_deleted_existing_file_is_refused(on, status):
    existing = "us-az/policies/des/faa5/medical/existing.yaml"
    with pytest.raises(AdmissionError, match=re.escape(existing)):
        verify_reviewed_pr(
            batch(UNITS3), fetch=Reads(make_api(UNITS3, extra=[(existing, status)]))
        )


@pytest.mark.parametrize("missing", ["companion", "primary"])
def test_a_unit_without_its_pair_is_refused_and_named(on, missing):
    """The candidate carries only half of unit 2."""
    api = make_api(UNITS3)
    lost = companion(unit(2)[1]) if missing == "companion" else unit(2)[1]
    for key in (FILES, COMPARE):
        listing = api[key] if key == FILES else api[key]["files"]
        listing[:] = [item for item in listing if item["filename"] != lost]
    api[HEAD_TREE]["tree"] = [
        entry for entry in api[HEAD_TREE]["tree"] if entry["path"] != lost
    ]
    api[PR]["changed_files"] = 5
    with pytest.raises(AdmissionError) as refusal:
        verify_reviewed_pr(batch(UNITS3), fetch=Reads(api))
    assert f"lacks {lost} of unit {unit(2)[0]}" in str(refusal.value)


def test_a_candidate_with_an_unpaired_file_of_an_unnamed_unit_is_refused(on):
    stray = companion(unit(9)[1])
    with pytest.raises(AdmissionError, match=re.escape(stray) + ".*no unit named"):
        verify_reviewed_pr(
            batch(UNITS3), fetch=Reads(make_api(UNITS3, extra=[(stray, "added")]))
        )


def test_an_operator_list_that_is_a_subset_of_the_candidate_is_refused(on):
    """The dispatch cannot select a part of the candidate."""
    with pytest.raises(AdmissionError) as refusal:
        verify_reviewed_pr(batch(UNITS3[:2]), fetch=Reads(make_api(UNITS3)))
    assert f"changes {companion(unit(3)[1])}, which belongs to no unit named" in str(
        refusal.value
    )


def test_an_operator_list_that_is_a_superset_of_the_candidate_is_refused(on):
    with pytest.raises(AdmissionError) as refusal:
        verify_reviewed_pr(batch(UNITS3), fetch=Reads(make_api(UNITS3[:2])))
    assert f"lacks {unit(3)[1]} of unit {unit(3)[0]}" in str(refusal.value)


def test_a_primary_named_twice_is_refused():
    citation, path = unit(1)
    with pytest.raises(AdmissionError, match="name no primary twice"):
        batch([(citation, path), (citation + "-again", path)])


def test_units_out_of_path_order_are_refused():
    with pytest.raises(AdmissionError, match="ascending path order"):
        batch([unit(2), unit(1)])


def test_a_citation_named_twice_is_refused():
    with pytest.raises(AdmissionError, match="citation twice"):
        batch([unit(1), (unit(1)[0], unit(2)[1])])


@pytest.mark.parametrize(
    "units", [(), [("us/x",)], [("us/x", 1)], [["us/x", "us/x.yaml"]], "us/x"]
)
def test_a_batch_without_wellformed_units_is_refused(units):
    with pytest.raises(AdmissionError):
        BatchPromotionSelector(units=units, **SHARED)


@pytest.mark.parametrize(
    "target",
    [
        ("nz/statute/x", "nz/statutes/x.yaml"),
        ("us/statute/x", "us/statutes/x.test.yaml"),
        ("us/statute/x", "us/statutes/../x.yaml"),
        ("us/statute/x", "us/statutes/x.json"),
    ],
)
def test_every_unit_is_checked_as_a_single_selector_is(target):
    with pytest.raises(AdmissionError):
        batch([target])


def test_a_unit_path_that_exists_on_main_is_refused_and_named(on):
    api = make_api(UNITS3)
    api[BASE_TREE]["tree"].append(
        {"path": unit(2)[1], "type": "blob", "mode": "100644", "sha": "d" * 40}
    )
    with pytest.raises(AdmissionError) as refusal:
        verify_reviewed_pr(batch(UNITS3), fetch=Reads(api))
    assert f"{unit(2)[1]} of unit {unit(2)[0]}" in str(refusal.value)
    assert "already exists on main" in str(refusal.value)


def test_a_unit_file_that_is_not_added_is_refused_and_named(on):
    api = make_api(UNITS3)
    api[FILES][2]["status"] = "modified"
    with pytest.raises(AdmissionError) as refusal:
        verify_reviewed_pr(batch(UNITS3), fetch=Reads(api))
    assert f"does not add {unit(2)[1]} of unit {unit(2)[0]}" in str(refusal.value)


def test_a_duplicate_changed_file_is_refused(on):
    api = make_api(UNITS3)
    api[FILES].append(dict(api[FILES][0]))
    api[PR]["changed_files"] = 7
    with pytest.raises(AdmissionError, match="invalid or duplicate changed file"):
        verify_reviewed_pr(batch(UNITS3), fetch=Reads(api))


@pytest.mark.parametrize("mode", ["120000", "100755", "160000"])
def test_a_unit_that_is_not_a_regular_blob_is_refused(on, mode):
    api = make_api(UNITS3)
    api[HEAD_TREE]["tree"][3]["mode"] = mode
    with pytest.raises(AdmissionError):
        verify_reviewed_pr(batch(UNITS3), fetch=Reads(api))


def test_the_tree_delta_still_decides_the_scope(on):
    """A file hidden from both file lists is found in the trees."""
    api = make_api(UNITS3)
    api[HEAD_TREE]["tree"].append(
        {
            "path": ".github/workflows/ci.yml",
            "type": "blob",
            "mode": "100644",
            "sha": "9" * 40,
        }
    )
    with pytest.raises(AdmissionError, match="complete immutable tree delta"):
        verify_reviewed_pr(batch(UNITS3), fetch=Reads(api))


@pytest.mark.parametrize("status", ["added", "modified"])
def test_the_derived_index_is_allowed_as_for_a_single_unit(on, status):
    """The index is one repository-wide derived file, not a per-unit file."""
    api = make_api(UNITS3, extra=[(admission.INDEX_PATH, status)])
    assert len(verify_reviewed_pr(batch(UNITS3), fetch=Reads(api))["records"]) == 3


def test_content_that_differs_from_its_tree_entry_is_refused(on):
    api = make_api(UNITS3)
    api[f"{ROOT}/contents/{unit(3)[1]}?ref={HEAD}"]["sha"] = "0" * 40
    with pytest.raises(AdmissionError, match="not a regular file"):
        verify_reviewed_pr(batch(UNITS3), fetch=Reads(api))


@pytest.mark.parametrize(
    ("path", "field", "value", "message"),
    [
        (PR, "state", "closed", "identity, head, or main base"),
        (RUN, "head_branch", "feature", "live central main signing run"),
        (RUN, "status", "completed", "live central main signing run"),
        (COMPARE, "status", "diverged", "descend from the current main head"),
    ],
)
def test_every_other_rule_still_applies_to_a_batch(on, path, field, value, message):
    api = make_api(UNITS3)
    api[path][field] = value
    with pytest.raises(AdmissionError, match=message):
        verify_reviewed_pr(batch(UNITS3), fetch=Reads(api))


def test_a_batch_on_a_stale_main_is_refused(on):
    api = make_api(UNITS3)
    api[ROOT + "/git/ref/heads/main"]["object"]["sha"] = "f" * 40
    with pytest.raises(AdmissionError, match="not the current main head"):
        verify_reviewed_pr(batch(UNITS3), fetch=Reads(api))


def test_a_batch_whose_branch_moved_is_refused(on):
    api = make_api(UNITS3)
    api[ROOT + "/git/ref/heads/codex%2Fbatch"]["object"]["sha"] = "f" * 40
    with pytest.raises(AdmissionError, match="candidate branch moved"):
        verify_reviewed_pr(batch(UNITS3), fetch=Reads(api))


def test_the_veto_refuses_a_batch(on):
    api = make_api(UNITS3)
    api[REVIEWS].append(
        {
            "id": 2,
            "state": "CHANGES_REQUESTED",
            "commit_id": HEAD,
            "user": {"id": 5, "login": "someone", "type": "User"},
        }
    )
    with pytest.raises(AdmissionError, match="unresolved changes requested"):
        verify_reviewed_pr(batch(UNITS3), fetch=Reads(api))


@pytest.mark.parametrize("approval_required", [False], indirect=True)
def test_a_dispatcher_who_is_not_an_enrolled_owner_cannot_admit_a_batch(on):
    api = make_api(UNITS3)
    api[RUN]["actor"] = {"id": 7, "login": "PavelMakarchuk", "type": "User"}
    with pytest.raises(AdmissionError, match="not a verified enrolled signing owner"):
        verify_reviewed_pr(batch(UNITS3), fetch=Reads(api))


# --- The selector's JSON form and the provisioner -----------------------------


def test_the_selector_survives_its_json_form(on):
    selector = batch(UNITS3)
    text = json.dumps(selector.payload(), sort_keys=True)
    assert selector_from_payload(json.loads(text)) == selector
    assert set(selector.payload()) == admission._BATCH_SELECTOR_FIELDS


@pytest.mark.parametrize(
    "change",
    [
        lambda p: p.update(extra=1),
        lambda p: p.pop("run_id"),
        lambda p: p.update(units="us/x"),
        lambda p: p["units"][0].update(extra=1),
        lambda p: p["units"][0].pop("citation"),
        lambda p: p["units"].append("us/x"),
        lambda p: p["units"].reverse(),
    ],
)
def test_a_malformed_batch_selector_payload_is_refused(on, change):
    payload = batch(UNITS3).payload()
    change(payload)
    with pytest.raises(AdmissionError):
        selector_from_payload(payload)


def test_a_selector_larger_than_the_provisioner_reads_is_refused(on, monkeypatch):
    monkeypatch.setattr(admission, "MAX_PROMOTION_UNITS", 50)
    units = [
        (
            f"us/statute/{index:02d}/" + "x" * 150,
            f"us/statutes/{index:02d}/" + "x" * 150 + ".yaml",
        )
        for index in range(12)
    ]
    selector = batch(units)
    assert len(json.dumps(selector.payload(), sort_keys=True)) + 1 > 4096
    with pytest.raises(AdmissionError, match="size bound"):
        verify_reviewed_pr(selector, fetch=Reads(make_api(units)))
    # One byte below the bound passes the size rule.
    monkeypatch.setattr(
        admission,
        "MAX_SELECTOR_BYTES",
        len(json.dumps(selector.payload(), sort_keys=True)) + 1,
    )
    assert (
        len(verify_reviewed_pr(selector, fetch=Reads(make_api(units)))["records"]) == 12
    )


def test_the_size_bound_is_the_provisioners():
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/provision_verification_supervisor.py"
    ).read_text()
    assert "raw_selector = selector_stream.read(4097)" in source
    assert "if len(raw_selector) > 4096:" in source
    assert admission.MAX_SELECTOR_BYTES == 4096
    # The provisioner needs only these two names and one key of the result.
    assert "module.selector_from_payload(json.loads(raw_selector))" in source
    assert "module.verify_reviewed_pr(selector)" in source
    assert 'admission["workflow_commit"]' in source


# --- Revalidation ---------------------------------------------------------------


@pytest.fixture
def provisioned(on):
    api = make_api(UNITS3)
    return api, verify_reviewed_pr(batch(UNITS3), fetch=Reads(api))


def test_revalidation_of_a_batch_reads_only_mutable_authority(provisioned):
    api, payload = provisioned
    reads = Reads(api)
    assert revalidate_admission(payload, fetch=reads) == payload
    assert reads.paths == [RUN, PR, ROOT + "/git/ref/heads/main", REVIEWS]


def test_revalidation_refuses_a_batch_after_main_moved(provisioned):
    api, payload = provisioned
    api[ROOT + "/git/ref/heads/main"]["object"]["sha"] = "f" * 40
    with pytest.raises(AdmissionError, match="differs from provisioned admission"):
        revalidate_admission(payload, fetch=Reads(api))


def test_revalidation_refuses_a_batch_after_the_head_moved(provisioned):
    api, payload = provisioned
    api[PR]["head"]["sha"] = "f" * 40
    with pytest.raises(AdmissionError, match="identity, head, or main base"):
        revalidate_admission(payload, fetch=Reads(api))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda p: p["records"].pop(), "lacks one record per unit"),
        (lambda p: p["records"].append(p["records"][0]), "lacks one record per unit"),
        (lambda p: p.update(records={}), "lacks one record per unit"),
        (
            lambda p: p["records"][1]["binding"].update(citation="us/other"),
            "not bound to its unit " + re.escape(unit(2)[0]),
        ),
        (
            lambda p: p["records"][2]["binding"].update(rulespec_path="us/o.yaml"),
            "not bound to its unit " + re.escape(unit(3)[0]),
        ),
        (lambda p: p["records"].reverse(), "not bound to its unit"),
        (lambda p: p["records"][1].update(base_sha="f" * 40), "differ in their main"),
        (lambda p: [r.update(base_sha="f" * 40) for r in p["records"]], "differs"),
        (lambda p: [r.update(base_sha=None) for r in p["records"]], "differ in"),
        (lambda p: p.update(schema=admission.SCHEMA), "invalid provisioned"),
        (lambda p: p.update(record=p["records"][0]), "invalid provisioned"),
        (lambda p: p["selector"].pop("units"), "selector has unexpected fields"),
        (lambda p: p["selector"]["units"].pop(), "lacks one record per unit"),
    ],
)
def test_revalidation_refuses_a_changed_batch_admission(provisioned, change, message):
    api, payload = provisioned
    payload = copy.deepcopy(payload)
    change(payload)
    with pytest.raises(AdmissionError, match=message):
        revalidate_admission(payload, fetch=Reads(api))


def test_a_single_admission_cannot_carry_a_batch_selector(on):
    api = make_api([unit(1)])
    payload = verify_reviewed_pr(single(unit(1)), fetch=Reads(api))
    payload["selector"] = batch([unit(1)]).payload()
    reads = Reads(api)
    with pytest.raises(AdmissionError, match="^invalid provisioned admission$"):
        revalidate_admission(payload, fetch=reads)
    assert reads.paths == []


def test_revalidation_refuses_a_batch_over_a_lowered_limit(provisioned, monkeypatch):
    api, payload = provisioned
    monkeypatch.setattr(admission, "MAX_PROMOTION_UNITS", 2)
    with pytest.raises(AdmissionError, match="the limit is 2"):
        revalidate_admission(payload, fetch=Reads(api))


# --- The request budget behind MAX_PROMOTION_UNITS ----------------------------


def _reads_of_one_run(count: int) -> int:
    """GitHub reads of one signing run for a batch of ``count`` units.

    Two admissions (preflight, root provisioner) and three revalidations
    (before validation, before signing, before publication).
    """
    units = [unit(index) for index in range(1, count + 1)]
    api = make_api(units)
    admit = Reads(api)
    payload = verify_reviewed_pr(batch(units), fetch=admit)
    again = Reads(api)
    revalidate_admission(payload, fetch=again)
    assert len(admit.paths) == 11 + 2 * count
    assert len(again.paths) == 4
    return 2 * len(admit.paths) + 3 * len(again.paths)


def test_the_limit_is_the_anonymous_request_budget_of_one_run(on, monkeypatch):
    limit = admission.MAX_PROMOTION_UNITS
    monkeypatch.setattr(admission, "MAX_PROMOTION_UNITS", limit + 1)
    assert admission.ANONYMOUS_REQUESTS_PER_HOUR == 60
    assert _reads_of_one_run(limit) == 34 + 4 * limit == 58
    assert _reads_of_one_run(limit) <= admission.ANONYMOUS_REQUESTS_PER_HOUR
    assert _reads_of_one_run(limit + 1) > admission.ANONYMOUS_REQUESTS_PER_HOUR


def test_a_single_unit_run_reads_38_times(approval_required):
    target = unit(1)
    api = make_api([target])
    admit, again = Reads(api), Reads(api)
    revalidate_admission(verify_reviewed_pr(single(target), fetch=admit), fetch=again)
    assert 2 * len(admit.paths) + 3 * len(again.paths) == 38


def test_the_admission_sends_no_credential():
    """The budget is the anonymous one because no token is sent."""
    source = Path(admission.__file__).read_text()
    assert "Authorization" not in source and "token" not in source.split(
        "def github_json"
    )[1].split("\n\n\n")[0].replace("no repository token", "")


def test_the_document_states_the_shipped_switch_and_limit():
    text = (
        Path(__file__).resolve().parents[1] / "docs/reviewed-pr-promotion.md"
    ).read_text()
    section = text.split("## Several units in one candidate")[1].split("\n## ")[0]
    assert ("**Switched off.**" in section) is (
        admission.MULTI_UNIT_PROMOTION_ENABLED is False
    )
    limit = admission.MAX_PROMOTION_UNITS
    assert f"`MAX_PROMOTION_UNITS` is {limit}." in section
    assert f"so `k <= {limit}` ({34 + 4 * limit} reads)" in section
    assert admission.BATCH_SCHEMA in section
