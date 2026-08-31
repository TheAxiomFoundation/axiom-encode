"""Tests for strict, protected-base validation failure waivers."""

from __future__ import annotations

import hashlib
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
PROGRAM_PATH = "us/programs/snap/fy-2026.yaml"


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


def _toolchain_bytes(waiver_bytes: bytes) -> bytes:
    return (
        "[toolchain]\n"
        'axiom_corpus_release = "test-release"\n'
        f'axiom_corpus_release_content_sha256 = "{"a" * 64}"\n'
        "validation_waiver_set_sha256 = "
        f'"{hashlib.sha256(waiver_bytes).hexdigest()}"\n'
    ).encode()


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


def test_rejects_composition_specs_from_atomic_validation_waivers(tmp_path: Path):
    root = _repo(tmp_path, PROGRAM_PATH)
    waiver_file = root / "known-validation-gaps.yaml"
    waiver_file.write_text(_valid_yaml().replace(PATH, PROGRAM_PATH))

    with pytest.raises(WaiverSchemaError, match="unsafe"):
        load_validation_waivers(waiver_file, repo_root=root, today=TODAY)


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


def test_new_or_changed_pending_requires_digest_rebind_pair():
    base = _set(_entry(PATH, active=_metadata("a")))
    head = _set(_entry(PATH, active=_metadata("a"), pending=_metadata("b")))

    issues = protected_base_transition_issues(
        base,
        head,
        changed_paths={"known-validation-gaps.yaml"},
        today=TODAY,
    )
    assert any("exact digest-rebind pair" in issue for issue in issues)


def test_new_pending_accepts_only_the_exact_toolchain_digest_rebind_pair():
    base = _set(_entry(PATH, active=_metadata("a")))
    head = _set(_entry(PATH, active=_metadata("a"), pending=_metadata("b")))
    base_waivers = _valid_yaml(active=True, pending=False).encode()
    head_waivers = _valid_yaml(active=True, pending=True).encode()
    evidence = {
        "base_waiver_bytes": base_waivers,
        "head_waiver_bytes": head_waivers,
        "base_toolchain_bytes": _toolchain_bytes(base_waivers),
        "head_toolchain_bytes": _toolchain_bytes(head_waivers),
    }

    assert (
        protected_base_transition_issues(
            base,
            head,
            changed_paths={
                "known-validation-gaps.yaml",
                ".axiom/toolchain.toml",
            },
            today=TODAY,
            **evidence,
        )
        == ()
    )
    missing_evidence = protected_base_transition_issues(
        base,
        head,
        changed_paths={"known-validation-gaps.yaml", ".axiom/toolchain.toml"},
        today=TODAY,
    )
    assert any("requires exact" in issue for issue in missing_evidence)
    third_path = protected_base_transition_issues(
        base,
        head,
        changed_paths={
            "known-validation-gaps.yaml",
            ".axiom/toolchain.toml",
            PATH,
        },
        today=TODAY,
        **evidence,
    )
    assert any("exact digest-rebind pair" in issue for issue in third_path)


def test_pending_approval_pull_request_cannot_batch_multiple_changes():
    base = _set(
        _entry(PATH, active=_metadata("a")),
        _entry(OTHER_PATH, active=_metadata("c")),
    )
    head = _set(
        _entry(PATH, active=_metadata("a"), pending=_metadata("b")),
        _entry(OTHER_PATH, active=_metadata("c"), pending=_metadata("d")),
    )
    base_waivers = (
        "validate_failures:\n"
        f"  {PATH}:\n"
        "    active:\n      " + _metadata_yaml("a") + f"  {OTHER_PATH}:\n"
        "    active:\n      " + _metadata_yaml("c")
    ).encode()
    head_waivers = (
        "validate_failures:\n"
        f"  {PATH}:\n"
        "    active:\n      "
        + _metadata_yaml("a")
        + "    pending:\n      "
        + _metadata_yaml("b")
        + f"  {OTHER_PATH}:\n"
        "    active:\n      "
        + _metadata_yaml("c")
        + "    pending:\n      "
        + _metadata_yaml("d")
    ).encode()

    issues = protected_base_transition_issues(
        base,
        head,
        changed_paths={"known-validation-gaps.yaml", ".axiom/toolchain.toml"},
        base_waiver_bytes=base_waivers,
        head_waiver_bytes=head_waivers,
        base_toolchain_bytes=_toolchain_bytes(base_waivers),
        head_toolchain_bytes=_toolchain_bytes(head_waivers),
        today=TODAY,
    )

    assert any("add exactly one new pending record" in issue for issue in issues)


def test_pending_creation_rejects_replacement_mixed_deltas_and_noops():
    active = _metadata("a")
    pending = _metadata("b")
    other_active = _metadata("c")
    other_pending = _metadata("d")
    exact_pair = {"known-validation-gaps.yaml", ".axiom/toolchain.toml"}

    replacement = protected_base_transition_issues(
        _set(_entry(PATH, active=active, pending=pending)),
        _set(_entry(PATH, active=active, pending=other_pending)),
        changed_paths=exact_pair,
        today=TODAY,
    )
    assert any("may not replace" in issue for issue in replacement)

    active_removal = protected_base_transition_issues(
        _set(
            _entry(PATH, active=active),
            _entry(OTHER_PATH, active=other_active),
        ),
        _set(_entry(PATH, active=active, pending=pending)),
        changed_paths=exact_pair,
        today=TODAY,
    )
    assert any(
        "may only add its one pending field" in issue for issue in active_removal
    )

    direct_active_to_pending = protected_base_transition_issues(
        _set(_entry(PATH, active=active)),
        _set(_entry(PATH, pending=pending)),
        changed_paths=exact_pair,
        today=TODAY,
    )
    assert any(
        "may only add its one pending field" in issue
        for issue in direct_active_to_pending
    )

    mixed_consumption = protected_base_transition_issues(
        _set(
            _entry(PATH, active=active),
            _entry(OTHER_PATH, active=other_active, pending=other_pending),
        ),
        _set(
            _entry(PATH, active=active, pending=pending),
            _entry(OTHER_PATH, active=other_pending),
        ),
        changed_paths=exact_pair,
        today=TODAY,
    )
    assert any(
        "may only add its one pending field" in issue for issue in mixed_consumption
    )

    pending_move = protected_base_transition_issues(
        _set(_entry(PATH, active=active, pending=pending)),
        _set(_entry(PATH, active=active), _entry(OTHER_PATH, pending=pending)),
        changed_paths=exact_pair,
        today=TODAY,
    )
    assert any("may only add its one pending field" in issue for issue in pending_move)

    duplicate_active = protected_base_transition_issues(
        _set(_entry(PATH, active=active)),
        _set(_entry(PATH, active=active, pending=active)),
        changed_paths=exact_pair,
        today=TODAY,
    )
    assert any("must not duplicate" in issue for issue in duplicate_active)

    semantic_noop = protected_base_transition_issues(
        _set(_entry(PATH, active=active)),
        _set(_entry(PATH, active=active)),
        changed_paths=exact_pair,
        today=TODAY,
    )
    assert any("semantic no-op" in issue for issue in semantic_noop)


def test_digest_rebind_rejects_equal_semantics_with_different_waiver_bytes():
    active = _metadata("a")
    base = _set(_entry(PATH, active=active))
    head = _set(_entry(PATH, active=active))
    base_waivers = _valid_yaml(active=True, pending=False).encode()
    head_waivers = base_waivers + b"# formatting-only rewrite\n"

    issues = protected_base_transition_issues(
        base,
        head,
        changed_paths={"known-validation-gaps.yaml", ".axiom/toolchain.toml"},
        base_waiver_bytes=base_waivers,
        head_waiver_bytes=head_waivers,
        base_toolchain_bytes=_toolchain_bytes(base_waivers),
        head_toolchain_bytes=_toolchain_bytes(head_waivers),
        today=TODAY,
    )

    assert any("semantic no-op" in issue for issue in issues)


@pytest.mark.parametrize("expiry", [TODAY - timedelta(days=1), TODAY])
def test_pending_creation_rejects_expired_new_approval(expiry: date):
    active = _metadata("a")
    pending = _metadata("b", expires=expiry.isoformat())
    base = _set(_entry(PATH, active=active))
    head = _set(_entry(PATH, active=active, pending=pending))
    base_waivers = _valid_yaml(active=True, pending=False).encode()
    head_waivers = (
        "validate_failures:\n"
        f"  {PATH}:\n"
        "    active:\n      "
        + _metadata_yaml("a")
        + "    pending:\n      "
        + _metadata_yaml("b", expires=expiry.isoformat())
    ).encode()

    issues = protected_base_transition_issues(
        base,
        head,
        changed_paths={"known-validation-gaps.yaml", ".axiom/toolchain.toml"},
        base_waiver_bytes=base_waivers,
        head_waiver_bytes=head_waivers,
        base_toolchain_bytes=_toolchain_bytes(base_waivers),
        head_toolchain_bytes=_toolchain_bytes(head_waivers),
        today=TODAY,
    )

    assert any("new pending approval expired" in issue for issue in issues)


def test_pending_creation_binds_semantics_to_the_exact_waiver_bytes():
    base = _set(_entry(PATH, active=_metadata("a")))
    head = _set(_entry(PATH, active=_metadata("a"), pending=_metadata("b")))
    base_waivers = _valid_yaml(active=True, pending=False).encode()
    inconsistent_head_waivers = base_waivers

    issues = protected_base_transition_issues(
        base,
        head,
        changed_paths={"known-validation-gaps.yaml", ".axiom/toolchain.toml"},
        base_waiver_bytes=base_waivers,
        head_waiver_bytes=inconsistent_head_waivers,
        base_toolchain_bytes=_toolchain_bytes(base_waivers),
        head_toolchain_bytes=_toolchain_bytes(inconsistent_head_waivers),
        today=TODAY,
    )

    assert any("head waiver semantics do not match" in issue for issue in issues)
    assert any("did not change" in issue for issue in issues)


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
    nonexact_consumption = protected_base_transition_issues(
        base,
        _set(_entry(PATH, active=_metadata("c"))),
        changed_paths={PATH, "known-validation-gaps.yaml"},
        today=TODAY,
    )
    assert any("must exactly consume" in issue for issue in nonexact_consumption)
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


@pytest.mark.parametrize("expiry", [TODAY - timedelta(days=1), TODAY])
def test_expired_base_pending_cannot_be_consumed(expiry: date):
    expired = _metadata("b", expires=expiry.isoformat())
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
