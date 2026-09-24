import os
from pathlib import Path

import pytest

from axiom_encode.notary.identity import IdentityRefusal
from axiom_encode.notary.producer_runtime import (
    LinuxRuntime,
    private_observation,
    systemd_command,
)


def test_host_only_observes_regular_single_link_nonexecutable_outputs(tmp_path):
    root = tmp_path / "job"
    root.mkdir()
    (root / "nested").mkdir()
    output = root / "nested" / "rule.yaml"
    output.write_bytes(b"generated")
    assert private_observation(root, "nested/rule.yaml") == b"generated"
    output.chmod(0o755)
    with pytest.raises(IdentityRefusal):
        private_observation(root, "nested/rule.yaml")
    output.chmod(0o644)
    os.link(output, root / "alias")
    with pytest.raises(IdentityRefusal):
        private_observation(root, "nested/rule.yaml")


def test_observation_refuses_symlinks_at_every_component(tmp_path):
    root, outside = tmp_path / "job", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret").write_bytes(b"private")
    (root / "link").symlink_to(outside, target_is_directory=True)
    (root / "leaf").symlink_to(outside / "secret")
    for path in ("link/secret", "leaf", "../outside/secret", str(outside / "secret")):
        with pytest.raises((OSError, IdentityRefusal)):
            private_observation(root, path)


def test_fifo_observation_does_not_block(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(IdentityRefusal):
        private_observation(tmp_path, "fifo")


def test_model_unit_has_no_keys_and_cgroup_lifetime_boundary():
    command = systemd_command(
        job=Path("/var/lib/axiom-producer/jobs/fixture"),
        python="/opt/axiom/runtime/bin/python3",
        worker_uid=1234,
        worker_gid=1234,
        readonly=[Path("/opt/axiom/corpus")],
        timeout=3600,
    )
    assert "--property=ExitType=cgroup" in command
    assert "--property=KillMode=control-group" in command
    assert "--property=NoNewPrivileges=yes" in command
    assert "--property=CapabilityBoundingSet=" in command
    assert "--property=ProtectSystem=strict" in command
    assert "--property=SupplementaryGroups=" in command
    assert "--property=ReadWritePaths=/var/lib/axiom-producer/jobs/fixture" in command
    assert command[-4:] == [
        "-I",
        "-m",
        "axiom_encode.notary.producer_worker",
        "/var/lib/axiom-producer/jobs/fixture",
    ]
    assert not any("TOKEN" in item or "PRIVATE_KEY" in item for item in command)


def test_unprivileged_or_non_linux_host_has_no_fallback(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    with pytest.raises(IdentityRefusal, match="linux_custodian"):
        LinuxRuntime({})
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("os.geteuid", lambda: 1000)
    with pytest.raises(IdentityRefusal, match="linux_custodian"):
        LinuxRuntime({})


def test_output_mapping_cannot_substitute_auth_or_another_protected_rule(tmp_path):
    from axiom_encode.notary.producer_runtime import (
        expected_outputs,
        read_encoder_outputs,
    )

    expected = expected_outputs("42 CFR 435.145", "fixture")
    assert len(expected) == 2
    for relative in expected.values():
        output = tmp_path / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"model output")
    assert set(read_encoder_outputs(tmp_path, {"files": expected}, expected)) == set(
        expected
    )
    for bad in (
        {next(iter(expected)): "auth.json"},
        {"rules/other.yaml": next(iter(expected.values()))},
    ):
        with pytest.raises(IdentityRefusal, match="output_mapping"):
            read_encoder_outputs(tmp_path, {"files": bad}, expected)


def test_worker_database_groups_must_be_empty_beyond_primary(monkeypatch):
    from types import SimpleNamespace

    from axiom_encode.notary.producer_runtime import require_worker_account

    monkeypatch.setattr(
        "pwd.getpwuid",
        lambda uid: SimpleNamespace(
            pw_gid=1234, pw_shell="/usr/sbin/nologin", pw_name="worker"
        ),
    )
    monkeypatch.setattr(os, "getgrouplist", lambda name, gid: [gid])
    require_worker_account(1234, 1234)
    monkeypatch.setattr(os, "getgrouplist", lambda name, gid: [gid, 27])
    with pytest.raises(IdentityRefusal, match="login_or_groups"):
        require_worker_account(1234, 1234)


def test_readonly_clone_remains_readable_with_private_host_umask(tmp_path):
    from axiom_encode.notary.producer_runtime import readable_immutable_tree

    root = tmp_path / "clone"
    root.mkdir(mode=0o700)
    (root / "file").write_bytes(b"source")
    (root / "file").chmod(0o600)
    (root / "script").write_bytes(b"script")
    (root / "script").chmod(0o700)
    readable_immutable_tree(root)
    assert root.stat().st_mode & 0o777 == 0o555
    assert (root / "file").stat().st_mode & 0o777 == 0o444
    assert (root / "script").stat().st_mode & 0o777 == 0o555
    root.chmod(0o700)
    (root / "link").symlink_to("file")
    with pytest.raises(IdentityRefusal, match="checkout_entry"):
        readable_immutable_tree(root)


def test_worker_git_allowlist_is_explicit():
    command = systemd_command(
        job=Path("/jobs/fixture"),
        python="/opt/bin/python",
        worker_uid=1234,
        worker_gid=1234,
        readonly=[],
        timeout=60,
        safe_directories=[Path("/jobs/fixture/lane")],
    )
    assert "--setenv=GIT_CONFIG_COUNT=1" in command
    assert "--setenv=GIT_CONFIG_KEY_0=safe.directory" in command
    assert "--setenv=GIT_CONFIG_VALUE_0=/jobs/fixture/lane" in command
    assert "--setenv=GIT_CONFIG_GLOBAL=/dev/null" in command
