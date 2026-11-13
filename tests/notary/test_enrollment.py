"""Team writers can request enrollment; others cannot gain signing authority."""

import json
import subprocess

import pytest

from axiom_encode.notary import enrollment
from tests.notary.lineage_fixtures import LANE


def permission(monkeypatch, value, *, login="team-writer", role_name=None):
    def run(argv, **kwargs):
        assert argv == [
            "gh",
            "api",
            "--hostname",
            "github.com",
            "--method",
            "GET",
            f"repos/{LANE}/collaborators/team-writer/permission",
        ]
        assert kwargs["timeout"] == 30
        body = {
            "permission": value,
            "user": {"login": login},
            "role_name": role_name or value,
        }
        return subprocess.CompletedProcess(argv, 0, json.dumps(body).encode(), b"")

    monkeypatch.setattr(enrollment.subprocess, "run", run)


@pytest.mark.parametrize(
    "value,role_name",
    [
        ("write", "write"),
        ("write", "maintain"),
        ("admin", "admin"),
        ("write", "custom-writer"),
    ],
)
def test_current_writer_qualifies_but_is_not_enrolled(monkeypatch, value, role_name):
    permission(monkeypatch, value, role_name=role_name)
    result = enrollment.check_write_access(LANE, "team-writer")
    assert result.eligible_to_request_enrollment
    assert not result.enrolled
    assert result.reason == "custodian-authorization-required"


@pytest.mark.parametrize("value", ["read", "triage", "none", "unknown", None, [], True])
def test_outsider_and_nonwriter_cannot_qualify(monkeypatch, value):
    permission(monkeypatch, value)
    result = enrollment.check_write_access(LANE, "team-writer")
    assert not result.eligible_to_request_enrollment and not result.enrolled


def test_renamed_or_mismatched_identity_fails_closed(monkeypatch):
    permission(monkeypatch, "write", login="different-writer")
    result = enrollment.check_write_access(LANE, "team-writer")
    assert result.reason == "permission-unavailable"
    assert not result.eligible_to_request_enrollment


def test_permission_loss_is_not_cached(monkeypatch):
    permission(monkeypatch, "write")
    assert enrollment.check_write_access(
        LANE, "team-writer"
    ).eligible_to_request_enrollment
    permission(monkeypatch, "read")
    assert not enrollment.check_write_access(
        LANE, "team-writer"
    ).eligible_to_request_enrollment


@pytest.mark.parametrize(
    "failure",
    [
        "missing-gh",
        "timeout",
        "http-error",
        "invalid-json",
        "duplicate-json",
        "wrong-shape",
    ],
)
def test_permission_failure_never_fails_open(monkeypatch, capsys, failure):
    def run(argv, **kwargs):
        if failure == "missing-gh":
            raise FileNotFoundError("private details")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 30)
        stdout = b"invalid"
        if failure == "duplicate-json":
            stdout = b'{"permission":"read","permission":"write","user":{"login":"team-writer"}}'
        if failure == "wrong-shape":
            stdout = b'{"permission":"write","user":true}'
        return subprocess.CompletedProcess(
            argv, 1 if failure == "http-error" else 0, stdout, b"private details"
        )

    monkeypatch.setattr(enrollment.subprocess, "run", run)
    assert enrollment.main(["--repository", LANE, "--operator", "team-writer"]) == 1
    captured = capsys.readouterr()
    assert "private details" not in captured.out + captured.err
    assert json.loads(captured.out)["enrolled"] is False


@pytest.mark.parametrize(
    "repository,operator",
    [
        ("org/repo/../../other", "writer"),
        (LANE, "../writer"),
        ("https://evil.example/repo", "writer"),
        (LANE, "writer?x=1"),
    ],
)
def test_identity_cannot_change_endpoint(monkeypatch, repository, operator):
    monkeypatch.setattr(
        enrollment.subprocess, "run", lambda *a, **k: pytest.fail("must not call gh")
    )
    result = enrollment.check_write_access(repository, operator)
    assert result.reason == "invalid-identity"


def test_successful_cli_output_still_requires_custodian(monkeypatch, capsys):
    permission(monkeypatch, "write")
    assert enrollment.main(["--repository", LANE, "--operator", "team-writer"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["eligible_to_request_enrollment"] is True
    assert output["enrolled"] is False
