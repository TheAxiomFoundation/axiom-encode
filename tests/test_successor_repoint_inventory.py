"""Whole-tree refusal checks for representations the literal inventory misses."""

import json
from pathlib import Path

import pytest

from axiom_encode.cli import (
    _plan_successor_repoint,
    _successor_repoint_reference_candidates,
    _successor_repoint_tree_entries,
)
from axiom_encode.successor_repoint import (
    SuccessorRepointError,
    load_repoint_request_payload,
)
from tests.successor_repoint_fixtures import (
    ENVELOPE,
    LEGACY_IDENTITY,
    SUCCESSOR_IDENTITY,
    build_repoint_fixture,
)


def _inventory(repo, commit):
    return _successor_repoint_reference_candidates(
        repo,
        commit=commit,
        request=load_repoint_request_payload(ENVELOPE),
        entries=_successor_repoint_tree_entries(repo, commit),
    )


@pytest.mark.parametrize(
    "path,text",
    [
        (
            "us/statutes/escaped.yaml",
            'imports: ["'
            + LEGACY_IDENTITY.replace("policies", r"\u0070olicies")
            + '"]\n',
        ),
        (
            "us/statutes/folded.yaml",
            'imports: ["'
            + LEGACY_IDENTITY.replace("policies", "poli\\\n  cies")
            + '"]\n',
        ),
        (
            "metadata.json",
            json.dumps({"target": LEGACY_IDENTITY}).replace(
                "policies", r"\u0070olicies"
            ),
        ),
        (
            "successor.yaml",
            'target: "' + SUCCESSOR_IDENTITY.replace("page", r"\u0070age") + '"\n',
        ),
        (
            "multidoc.yml",
            '---\nnote: hello\n---\ntarget: "'
            + LEGACY_IDENTITY.replace("policies", r"\u0070olicies")
            + '"\n',
        ),
    ],
)
def test_escaped_references_refused(tmp_path, monkeypatch, path, text):
    def add(repo):
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    fixture = build_repoint_fixture(tmp_path, monkeypatch, before_commit=add)
    with pytest.raises(
        SuccessorRepointError, match="escaped legacy or successor reference"
    ):
        _inventory(fixture.repo, fixture.base)


def test_unrelated_backslash_does_not_hide_literal_reference(tmp_path, monkeypatch):
    def add(repo):
        (repo / "literal.yaml").write_text(
            f"target: {LEGACY_IDENTITY}\n" + r"note: 'C:\unrelated'" + "\n"
        )

    fixture = build_repoint_fixture(tmp_path, monkeypatch, before_commit=add)
    assert "literal.yaml" in _inventory(fixture.repo, fixture.base)


@pytest.mark.parametrize(
    "path,text", [("bad.yaml", 'x: "\\q"'), ("bad.json", '{"x": "\\q"}')]
)
def test_unparseable_escape_candidates_refused(tmp_path, monkeypatch, path, text):
    fixture = build_repoint_fixture(
        tmp_path, monkeypatch, before_commit=lambda repo: (repo / path).write_text(text)
    )
    with pytest.raises(
        SuccessorRepointError, match="cannot parse escaped-reference candidate"
    ):
        _inventory(fixture.repo, fixture.base)


def test_escape_scan_reads_git_objects(tmp_path, monkeypatch):
    fixture = build_repoint_fixture(tmp_path, monkeypatch)
    (fixture.repo / "untracked.yaml").write_text('target: "\\q"')
    assert "untracked.yaml" not in _inventory(fixture.repo, fixture.base)


def _refresh_fixture_owner(repo, module):
    """Keep synthetic v1/v5 ownership bound after changing fixture source."""
    import hashlib

    from tests.successor_repoint_fixtures import (
        LEGACY,
        LEGACY_V1_MANIFEST,
        SUCCESSOR,
        SUCCESSOR_MANIFEST,
    )

    owner = {LEGACY: LEGACY_V1_MANIFEST, SUCCESSOR: SUCCESSOR_MANIFEST}.get(module)
    if owner is None:
        return
    payload = json.loads((repo / owner).read_text())
    for entry in payload["applied_files"]:
        if entry["path"] in {module, module.removeprefix("us/")}:
            entry["sha256"] = hashlib.sha256((repo / module).read_bytes()).hexdigest()
    (repo / owner).write_text(json.dumps(payload))


def _plan(fixture, envelope=None):
    return _plan_successor_repoint(
        fixture.repo,
        commit=fixture.base,
        request=load_repoint_request_payload(envelope or ENVELOPE),
        verify_live=False,
    )


@pytest.mark.parametrize("location", ["legacy", "successor", "unrelated"])
@pytest.mark.parametrize("target_identity", [LEGACY_IDENTITY, SUCCESSOR_IDENTITY])
def test_plan_refuses_sets_overrides_anywhere(
    tmp_path, monkeypatch, location, target_identity
):
    from tests.successor_repoint_fixtures import LEGACY, SUCCESSOR

    module = {"legacy": LEGACY, "successor": SUCCESSOR}.get(
        location, "us/statutes/overrides.yaml"
    )

    def add(repo):
        path = repo / module
        base = path.read_text() if path.exists() else "format: rulespec/v1\nrules:\n"
        path.write_text(
            base
            + "  - name: override\n"
            + "    source_relation:\n"
            + "      type: ' SeTs '\n"
            + f"      target: ' {target_identity}#eitc_maximum_investment_income '\n"
            + "    versions:\n"
            + "      - effective_from: '2026-01-01'\n"
            + "        formula: 99\n"
        )
        _refresh_fixture_owner(repo, module)

    fixture = build_repoint_fixture(tmp_path, monkeypatch, before_commit=add)
    with pytest.raises(SuccessorRepointError, match="source_relation sets override"):
        _plan(fixture)


def test_toolchain_residue_checked_without_waiver_rewrite():
    from axiom_encode.cli import _successor_repoint_metadata_reconciliations

    path = Path(".axiom/toolchain.toml")
    raw = f'legacy_module = "{LEGACY_IDENTITY}"\n'.encode()
    with pytest.raises(SuccessorRepointError, match="left a retired record"):
        _successor_repoint_metadata_reconciliations(
            tracked={path},
            request=load_repoint_request_payload(ENVELOPE),
            dependent_postimages={},
            read=lambda _path, _label: raw,
        )
