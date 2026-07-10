"""Tests for strict, protected-base validation failure waivers."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from types import MappingProxyType

import pytest

from axiom_encode.validation_waivers import (
    MAX_WAIVER_DAYS,
    ValidationWaiverSet,
    WaiverEntry,
    WaiverMetadata,
    WaiverSchemaError,
    canonicalize_outcome,
    fingerprint_outcome,
    load_validation_waivers,
    protected_base_transition_issues,
)

TODAY = date(2026, 7, 10)
EXPIRY = (TODAY + timedelta(days=MAX_WAIVER_DAYS)).isoformat()
PATH = "us/statutes/26/1.yaml"
OTHER_PATH = "us-ca/regulations/mpp/1.yaml"


def _metadata(
    marker: str = "a",
    *,
    expires: str = EXPIRY,
    owner: str = "@MaxGhenis",
    issue: str = "https://github.com/TheAxiomFoundation/rulespec-us/issues/782",
) -> WaiverMetadata:
    return WaiverMetadata(
        fingerprint=f"sha256:{marker * 64}",
        owner=owner,
        issue=issue,
        expires=expires,
    )


def _entry(path: str, *, active=None, pending=None) -> WaiverEntry:
    return WaiverEntry(path=path, active=active, pending=pending)


def _set(*entries: WaiverEntry) -> ValidationWaiverSet:
    return ValidationWaiverSet(
        MappingProxyType({entry.path: entry for entry in entries})
    )


def _repo(tmp_path: Path, *paths: str) -> Path:
    root = tmp_path / "rulespec-us"
    root.mkdir()
    for path in paths:
        module = root / path
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text("format: rulespec/v1\n")
    return root


def _metadata_yaml(
    marker: str = "a",
    *,
    expires: str = EXPIRY,
    issue: str = "https://github.com/TheAxiomFoundation/rulespec-us/issues/782",
) -> str:
    return (
        f'fingerprint: "sha256:{marker * 64}"\n'
        '      owner: "@MaxGhenis"\n'
        f'      issue: "{issue}"\n'
        f'      expires: "{expires}"\n'
    )


def _valid_yaml(*, active: bool = True, pending: bool = False) -> str:
    states = ""
    if active:
        states += "    active:\n      " + _metadata_yaml("a")
    if pending:
        states += "    pending:\n      " + _metadata_yaml("b")
    return f"validate_failures:\n  {PATH}:\n{states}"


def test_requires_file_and_section_but_accepts_empty_mapping(tmp_path: Path):
    root = _repo(tmp_path)
    with pytest.raises(WaiverSchemaError, match="required.*file.*missing"):
        load_validation_waivers(root / "missing.yaml", repo_root=root, today=TODAY)

    waiver_file = root / "known-validation-gaps.yaml"
    waiver_file.write_text("shape_issues: []\n")
    with pytest.raises(WaiverSchemaError, match="exactly validate_failures"):
        load_validation_waivers(waiver_file, repo_root=root, today=TODAY)

    waiver_file.write_text("validate_failures: {}\nschema_typo: true\n")
    with pytest.raises(WaiverSchemaError, match="exactly validate_failures"):
        load_validation_waivers(waiver_file, repo_root=root, today=TODAY)

    waiver_file.write_text("validate_failures: {}\n")
    loaded = load_validation_waivers(waiver_file, repo_root=root, today=TODAY)
    assert loaded.active_paths == frozenset()
    assert loaded.pending_paths == frozenset()


@pytest.mark.parametrize("content", ["[]\n", "false\n", "0\n", "''\n"])
def test_rejects_falsy_non_mapping_document_roots(tmp_path: Path, content: str):
    root = _repo(tmp_path)
    waiver_file = root / "known-validation-gaps.yaml"
    waiver_file.write_text(content)

    with pytest.raises(WaiverSchemaError, match="document root must be a mapping"):
        load_validation_waivers(waiver_file, repo_root=root, today=TODAY)


def test_loads_nested_active_and_pending_with_cross_repo_issue(tmp_path: Path):
    root = _repo(tmp_path, PATH)
    content = _valid_yaml(active=True, pending=False)
    content += "    pending:\n      " + _metadata_yaml(
        "b",
        issue="https://github.com/TheAxiomFoundation/axiom-encode/issues/1036",
    )
    waiver_file = root / "known-validation-gaps.yaml"
    waiver_file.write_text(content)

    loaded = load_validation_waivers(waiver_file, repo_root=root, today=TODAY)

    assert loaded.active_paths == {PATH}
    assert loaded.pending_paths == {PATH}
    assert loaded.entries[PATH].pending.issue.endswith("axiom-encode/issues/1036")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (f"validate_failures:\n  - {PATH}\n", "must be a mapping"),
        (
            f"validate_failures:\n  {PATH}:\n    fingerprint: sha256:{'a' * 64}\n",
            "active and/or pending",
        ),
        (
            f"validate_failures:\n  {PATH}:\n    active:\n"
            f"      fingerprint: sha256:{'a' * 64}\n"
            '      owner: "@MaxGhenis"\n'
            '      issue: "https://github.com/TheAxiomFoundation/rulespec-us/issues/782"\n',
            "must contain exactly",
        ),
        (
            _valid_yaml().replace('fingerprint: "sha256:', 'fingerprint: "sha512:', 1),
            "fingerprint must be",
        ),
        (
            _valid_yaml().replace('owner: "@MaxGhenis"', 'owner: "MaxGhenis"'),
            "@GitHub-login",
        ),
        (
            _valid_yaml().replace(
                "https://github.com/TheAxiomFoundation/rulespec-us/issues/782",
                "https://github.com/OtherOrg/repo/issues/1",
            ),
            "Axiom Foundation GitHub issue URL",
        ),
        (
            _valid_yaml().replace(f'expires: "{EXPIRY}"', f"expires: {EXPIRY}"),
            "quoted YYYY-MM-DD",
        ),
    ],
)
def test_rejects_legacy_flat_or_malformed_records(
    tmp_path: Path, content: str, message: str
):
    root = _repo(tmp_path, PATH)
    waiver_file = root / "known-validation-gaps.yaml"
    waiver_file.write_text(content)

    with pytest.raises(WaiverSchemaError, match=message):
        load_validation_waivers(waiver_file, repo_root=root, today=TODAY)


def test_expiry_is_strictly_future_and_at_most_ninety_days(tmp_path: Path):
    root = _repo(tmp_path, PATH)
    waiver_file = root / "known-validation-gaps.yaml"

    waiver_file.write_text(_valid_yaml())
    assert (
        load_validation_waivers(waiver_file, repo_root=root, today=TODAY)
        .entries[PATH]
        .active.expires
        == EXPIRY
    )

    waiver_file.write_text(_valid_yaml().replace(EXPIRY, TODAY.isoformat()))
    with pytest.raises(WaiverSchemaError, match="expired"):
        load_validation_waivers(waiver_file, repo_root=root, today=TODAY)

    too_far = (TODAY + timedelta(days=MAX_WAIVER_DAYS + 1)).isoformat()
    waiver_file.write_text(_valid_yaml().replace(EXPIRY, too_far))
    with pytest.raises(WaiverSchemaError, match="within 90 days"):
        load_validation_waivers(waiver_file, repo_root=root, today=TODAY)


@pytest.mark.parametrize(
    "content",
    [
        f"validate_failures:\n  {PATH}: {{}}\n  {PATH}: {{}}\n",
        "validate_failures: &waivers {}\ncopy: *waivers\n",
        f"validate_failures:\n  {PATH}:\n    <<: {{}}\n    active:\n"
        f"      {_metadata_yaml()}",
    ],
)
def test_rejects_duplicate_keys_aliases_anchors_and_merges(
    tmp_path: Path, content: str
):
    root = _repo(tmp_path, PATH)
    waiver_file = root / "known-validation-gaps.yaml"
    waiver_file.write_text(content)

    with pytest.raises(WaiverSchemaError):
        load_validation_waivers(waiver_file, repo_root=root, today=TODAY)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/us/statutes/26/1.yaml",
        "us/statutes/../1.yaml",
        "us\\statutes\\26\\1.yaml",
        "US/statutes/26/1.yaml",
        "us/sources/26/1.yaml",
        "us/statutes/26/1.test.yaml",
        "us/statutes/26/\u202e1.yaml",
    ],
)
def test_rejects_noncanonical_module_paths(tmp_path: Path, unsafe_path: str):
    root = _repo(tmp_path, PATH)
    waiver_file = root / "known-validation-gaps.yaml"
    waiver_file.write_text(_valid_yaml().replace(PATH, unsafe_path))

    with pytest.raises(WaiverSchemaError, match="unsafe"):
        load_validation_waivers(
            waiver_file, repo_root=root, today=TODAY, require_paths=False
        )


def test_rejects_control_characters_that_could_inject_active_path_lines(
    tmp_path: Path,
):
    root = _repo(tmp_path, PATH)
    waiver_file = root / "known-validation-gaps.yaml"
    injected = '"us/statutes/inject\\nus/statutes/victim.yaml"'
    waiver_file.write_text(_valid_yaml().replace(PATH, injected))

    with pytest.raises(WaiverSchemaError, match="unsafe"):
        load_validation_waivers(
            waiver_file,
            repo_root=root,
            today=TODAY,
            require_paths=False,
        )


def test_requires_existing_regular_head_paths_but_not_base_paths(tmp_path: Path):
    root = _repo(tmp_path)
    waiver_file = root / "known-validation-gaps.yaml"
    waiver_file.write_text(_valid_yaml())

    with pytest.raises(WaiverSchemaError, match="does not exist"):
        load_validation_waivers(waiver_file, repo_root=root, today=TODAY)
    assert load_validation_waivers(
        waiver_file,
        repo_root=root,
        today=TODAY,
        require_paths=False,
    ).active_paths == {PATH}


def test_rejects_symlinked_waiver_file(tmp_path: Path):
    root = _repo(tmp_path, PATH)
    target = root / "waiver-target.yaml"
    target.write_text(_valid_yaml())
    waiver_file = root / "known-validation-gaps.yaml"
    waiver_file.symlink_to(target.name)

    with pytest.raises(WaiverSchemaError, match="regular file"):
        load_validation_waivers(waiver_file, repo_root=root, today=TODAY)


def test_rejects_symlinked_module_path_components(tmp_path: Path):
    root = _repo(tmp_path)
    target_directory = root / "us/statutes/actual"
    target_directory.mkdir(parents=True)
    (target_directory / "1.yaml").write_text("format: rulespec/v1\n")
    (root / "us/statutes/alias").symlink_to(target_directory.name)
    waiver_file = root / "known-validation-gaps.yaml"
    waiver_file.write_text(_valid_yaml().replace(PATH, "us/statutes/alias/1.yaml"))

    with pytest.raises(WaiverSchemaError, match="symlink alias"):
        load_validation_waivers(waiver_file, repo_root=root, today=TODAY)


def test_new_or_changed_pending_is_waiver_only():
    base = _set(_entry(PATH, active=_metadata("a")))
    head = _set(_entry(PATH, active=_metadata("a"), pending=_metadata("b")))

    assert (
        protected_base_transition_issues(
            base,
            head,
            changed_paths={"known-validation-gaps.yaml"},
            today=TODAY,
        )
        == ()
    )
    issues = protected_base_transition_issues(
        base,
        head,
        changed_paths={"known-validation-gaps.yaml", PATH},
        today=TODAY,
    )
    assert any("waiver-only" in issue for issue in issues)


def test_active_can_only_change_by_consuming_exact_base_pending():
    active = _metadata("a")
    pending = _metadata("b")
    base = _set(_entry(PATH, active=active, pending=pending))

    assert (
        protected_base_transition_issues(
            base,
            _set(_entry(PATH, active=pending)),
            changed_paths={PATH, "known-validation-gaps.yaml"},
            today=TODAY,
        )
        == ()
    )
    direct_change = protected_base_transition_issues(
        _set(_entry(PATH, active=active)),
        _set(_entry(PATH, active=pending)),
        changed_paths={PATH, "known-validation-gaps.yaml"},
        today=TODAY,
    )
    assert any("must exactly consume" in issue for issue in direct_change)
    unconsumed = protected_base_transition_issues(
        base,
        _set(_entry(PATH, active=pending, pending=pending)),
        changed_paths={"known-validation-gaps.yaml"},
        today=TODAY,
    )
    assert any("must consume it" in issue for issue in unconsumed)


def test_new_active_without_base_pending_is_rejected_and_removals_are_safe():
    empty = _set()
    new_active = _set(_entry(PATH, active=_metadata("a")))
    assert protected_base_transition_issues(
        empty,
        new_active,
        changed_paths={"known-validation-gaps.yaml"},
        today=TODAY,
    )
    assert (
        protected_base_transition_issues(
            new_active,
            empty,
            changed_paths={"known-validation-gaps.yaml", PATH},
            today=TODAY,
        )
        == ()
    )


def test_expired_base_pending_cannot_be_consumed():
    expired = _metadata("b", expires=(TODAY - timedelta(days=1)).isoformat())
    base = _set(_entry(PATH, active=_metadata("a"), pending=expired))
    head = _set(_entry(PATH, active=expired))

    issues = protected_base_transition_issues(
        base,
        head,
        changed_paths={PATH, "known-validation-gaps.yaml"},
        today=TODAY,
    )

    assert any("pending approval expired" in issue for issue in issues)


def test_fingerprint_is_semantic_deterministic_and_retains_duplicates():
    validate = {
        "passed": False,
        "duration_ms": 123,
        "validators": {
            "ci": {
                "passed": False,
                "issues": ["second /tmp/work", "first", "first"],
                "error": "first",
                "raw_output": "ignored",
            },
            "compile": {"passed": True, "issues": ["ignored"]},
        },
    }
    companion = {
        "present": True,
        "passed": False,
        "path": "/tmp/work/us/statutes/26/1.test.yaml",
        "cases": 2,
        "failures": [
            {"file": "/tmp/work/test", "case": "b", "message": "later"},
            {"file": "/tmp/work/test", "case": "a", "message": "earlier"},
        ],
        "compiled_programs": 999,
    }
    replacements = {"/tmp/work": "<repo>"}

    canonical = json.loads(
        canonicalize_outcome(validate, companion, replacements=replacements).decode(
            "utf-8"
        )
    )
    digest = fingerprint_outcome(validate, companion, replacements=replacements)

    assert canonical["schema"] == "rulespec-validation-failure/v1"
    assert canonical["validate"]["validators"] == {
        "ci": {
            "error": "first",
            "issues": ["first", "first", "second <repo>"],
        }
    }
    assert [failure["case"] for failure in canonical["companion"]["failures"]] == [
        "a",
        "b",
    ]
    assert "/tmp/work" not in json.dumps(canonical)
    assert digest == fingerprint_outcome(
        {**validate, "duration_ms": 999999},
        {**companion, "compiled_programs": 0},
        replacements=replacements,
    )
    assert digest != fingerprint_outcome(
        validate,
        {**companion, "cases": 3},
        replacements=replacements,
    )


def test_fingerprint_normalizes_validator_owned_alias_and_temp_directories():
    def validate(alias: str, temporary: str) -> dict:
        return {
            "passed": False,
            "validators": {
                "ci": {
                    "passed": False,
                    "error": (
                        f"compile {alias}/rulespec-us/us/statutes/1.yaml "
                        f"via {temporary}/compiled.json"
                    ),
                    "issues": [],
                }
            },
        }

    companion = {
        "present": False,
        "passed": True,
        "path": "us/statutes/1.test.yaml",
        "cases": 0,
        "failures": [],
    }
    first = validate(
        "/tmp/axiom-rulespec-repo-aliases/aaaaaaaaaaaaaaaa",
        "/tmp/tmpFirst123",
    )
    second = validate(
        "/tmp/axiom-rulespec-repo-aliases/bbbbbbbbbbbbbbbb",
        "/tmp/tmpSecond456",
    )

    assert fingerprint_outcome(
        first, companion, replacements={"/tmp": "<system-tmp>"}
    ) == fingerprint_outcome(second, companion, replacements={"/tmp": "<system-tmp>"})
