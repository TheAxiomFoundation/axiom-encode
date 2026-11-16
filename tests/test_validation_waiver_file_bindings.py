"""Per-file waiver bindings for signed manifests (#2092)."""

from __future__ import annotations

import copy
from datetime import date, timedelta
from types import MappingProxyType

import pytest

from axiom_encode.validation_waivers import (
    FILE_WAIVER_BINDINGS_SCHEMA,
    MAX_WAIVER_DAYS,
    ValidationWaiverError,
    ValidationWaiverSet,
    WaiverEntry,
    WaiverMetadata,
    WaiverSchemaError,
    active_waiver_record_sha256,
    build_file_waiver_bindings,
    file_waiver_binding_issues,
    load_validation_waivers_bytes,
)

TODAY = date(2026, 11, 6)
EXPIRY = (TODAY + timedelta(days=30)).isoformat()
COVERED = "us/statutes/26/21.yaml"
COVERED_PLAIN = "us/statutes/26/22.yaml"
UNCOVERED = "us-ma/regulations/106-cmr/1.yaml"
LABEL = "m.json validation_waiver_bindings"


def _metadata(marker: str, *, expires: str = EXPIRY) -> WaiverMetadata:
    return WaiverMetadata(
        fingerprint=f"sha256:{marker * 64}",
        owner="@MaxGhenis",
        issue="https://github.com/TheAxiomFoundation/rulespec-us/issues/1980",
        expires=expires,
    )


def _waivers(
    *, covered=_metadata("a"), uncovered=_metadata("b"), pending=None
) -> ValidationWaiverSet:
    return ValidationWaiverSet(
        MappingProxyType(
            {
                COVERED: WaiverEntry(path=COVERED, active=covered, pending=pending),
                UNCOVERED: WaiverEntry(path=UNCOVERED, active=uncovered),
            }
        )
    )


def _issues(bindings, waivers, covered=(COVERED, COVERED_PLAIN)):
    return file_waiver_binding_issues(
        bindings,
        covered_paths=frozenset(covered),
        waivers=waivers,
        label=LABEL,
    )


def test_bindings_record_path_fingerprint_and_record_digest_of_covered_files():
    waivers = _waivers()
    bindings = build_file_waiver_bindings(waivers, {COVERED_PLAIN, COVERED})

    assert bindings == {
        "schema": FILE_WAIVER_BINDINGS_SCHEMA,
        "files": [
            {
                "path": COVERED,
                "fingerprint": "sha256:" + "a" * 64,
                "record_sha256": active_waiver_record_sha256(_metadata("a")),
            },
            {"path": COVERED_PLAIN, "fingerprint": None, "record_sha256": None},
        ],
    }
    assert _issues(bindings, waivers) == []


def test_waiver_change_for_uncovered_file_keeps_bindings_current():
    bindings = build_file_waiver_bindings(_waivers(), {COVERED, COVERED_PLAIN})
    changed = _waivers(uncovered=_metadata("c", expires=TODAY.isoformat()))
    removed = ValidationWaiverSet(
        MappingProxyType({COVERED: WaiverEntry(path=COVERED, active=_metadata("a"))})
    )

    assert _issues(bindings, changed) == []
    assert _issues(bindings, removed) == []


def test_pending_record_on_covered_file_does_not_change_the_binding():
    bindings = build_file_waiver_bindings(_waivers(), {COVERED, COVERED_PLAIN})

    assert _issues(bindings, _waivers(pending=_metadata("d"))) == []


@pytest.mark.parametrize(
    "covered",
    [
        _metadata("e"),  # fingerprint drift
        _metadata("a", expires=(TODAY + timedelta(days=60)).isoformat()),  # renewal
        None,  # waiver removed
    ],
)
def test_covered_file_waiver_change_is_not_current(covered):
    bindings = build_file_waiver_bindings(_waivers(), {COVERED, COVERED_PLAIN})

    assert _issues(bindings, _waivers(covered=covered)) == [
        f"{LABEL} waiver identity for {COVERED} is not current"
    ]


def test_new_waiver_on_covered_plain_file_is_not_current():
    bindings = build_file_waiver_bindings(_waivers(), {COVERED, COVERED_PLAIN})
    waivers = ValidationWaiverSet(
        MappingProxyType(
            {
                **_waivers().entries,
                COVERED_PLAIN: WaiverEntry(path=COVERED_PLAIN, active=_metadata("f")),
            }
        )
    )

    assert _issues(bindings, waivers) == [
        f"{LABEL} waiver identity for {COVERED_PLAIN} is not current"
    ]


def _valid_bindings():
    return build_file_waiver_bindings(_waivers(), {COVERED, COVERED_PLAIN})


def _mutate(change):
    bindings = copy.deepcopy(_valid_bindings())
    change(bindings)
    return bindings


@pytest.mark.parametrize(
    ("bindings", "expected"),
    [
        (None, f"{LABEL} is malformed"),
        ("sha256", f"{LABEL} is malformed"),
        (_mutate(lambda b: b.update(extra=1)), f"{LABEL} is malformed"),
        (_mutate(lambda b: b.pop("files")), f"{LABEL} is malformed"),
        (
            _mutate(lambda b: b.update(schema="v0")),
            f"{LABEL}.schema is not supported",
        ),
        (_mutate(lambda b: b.update(files=[])), f"{LABEL}.files is malformed"),
        (_mutate(lambda b: b.update(files={})), f"{LABEL}.files is malformed"),
        (
            _mutate(lambda b: b["files"][0].update(extra=None)),
            f"{LABEL}.files[0] is malformed",
        ),
        (
            _mutate(lambda b: b["files"][0].update(path="")),
            f"{LABEL}.files[0].path is invalid",
        ),
        (
            _mutate(lambda b: b["files"].reverse()),
            f"{LABEL}.files[1].path is duplicated or out of order",
        ),
        (
            _mutate(lambda b: b["files"].append(dict(b["files"][1]))),
            f"{LABEL}.files[2].path is duplicated or out of order",
        ),
        (
            _mutate(lambda b: b["files"][0].update(record_sha256=None)),
            f"{LABEL}.files[0] waiver record identity is invalid",
        ),
        (
            _mutate(lambda b: b["files"][1].update(fingerprint="sha256:" + "0" * 64)),
            f"{LABEL}.files[1] waiver record identity is invalid",
        ),
        (
            _mutate(lambda b: b["files"][0].update(fingerprint="A" * 64)),
            f"{LABEL}.files[0] waiver record identity is invalid",
        ),
        (
            _mutate(lambda b: b["files"][0].update(record_sha256="F" * 64)),
            f"{LABEL}.files[0] waiver record identity is invalid",
        ),
    ],
)
def test_malformed_bindings_fail_closed(bindings, expected):
    assert _issues(bindings, _waivers()) == [expected]


def test_bindings_must_name_exactly_the_covered_files():
    bindings = _valid_bindings()

    assert _issues(bindings, _waivers(), covered=(COVERED,)) == [
        f"{LABEL} binds files it does not cover: {COVERED_PLAIN}"
    ]
    assert _issues(
        bindings, _waivers(), covered=(COVERED, COVERED_PLAIN, UNCOVERED)
    ) == [f"{LABEL} does not bind covered files: {UNCOVERED}"]


def test_build_refuses_empty_coverage():
    with pytest.raises(ValidationWaiverError):
        build_file_waiver_bindings(_waivers(), set())


def test_load_bytes_uses_the_strict_schema():
    raw = (
        "validate_failures:\n"
        f"  {COVERED}:\n"
        "    active:\n"
        f"      fingerprint: sha256:{'a' * 64}\n"
        "      owner: '@MaxGhenis'\n"
        "      issue: https://github.com/TheAxiomFoundation/rulespec-us/issues/1980\n"
        f"      expires: '{EXPIRY}'\n"
    ).encode()

    loaded = load_validation_waivers_bytes(raw, today=TODAY, require_paths=False)
    assert loaded.entries[COVERED].active == _metadata("a")

    too_long = (TODAY + timedelta(days=MAX_WAIVER_DAYS + 1)).isoformat()
    with pytest.raises(WaiverSchemaError, match="within"):
        load_validation_waivers_bytes(
            raw.replace(EXPIRY.encode(), too_long.encode()),
            today=TODAY,
            require_paths=False,
        )
    with pytest.raises(WaiverSchemaError, match="duplicate"):
        load_validation_waivers_bytes(
            raw + f"  {COVERED}: {{}}\n".encode(), today=TODAY, require_paths=False
        )
    with pytest.raises(WaiverSchemaError):
        load_validation_waivers_bytes(b"\xff", today=TODAY, require_paths=False)
