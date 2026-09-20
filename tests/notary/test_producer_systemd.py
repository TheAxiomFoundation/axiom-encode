"""Opt-in real Linux/systemd boundary test on an ephemeral privileged runner.

Never run against an enrolled production host. All paths/UIDs are disposable;
this test refuses if reserved fixture directories/accounts already exist.
"""

import os
import platform
import pwd
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from axiom_encode.notary.producer_runtime import (
    require_worker_account,
    require_worker_idle,
    systemd_command,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("AXIOM_NOTARY_EPHEMERAL_ISOLATION_TEST") != "1"
    or platform.system() != "Linux"
    or os.geteuid() != 0,
    reason="requires explicit ephemeral root Linux/systemd runner",
)


def test_actual_worker_cannot_read_host_key_or_modify_trusted_installation():
    user = "axiom-notary-fixture"
    try:
        pwd.getpwnam(user)
    except KeyError:
        pass
    else:
        pytest.fail("fixture account already exists; refuse existing host")
    protected = [Path("/etc/axiom-producer"), Path("/run/axiom-producer")]
    if any(path.exists() for path in protected):
        pytest.fail("producer paths exist; this is not a fresh test host")
    root = Path(tempfile.mkdtemp(prefix="axiom-notary-isolation-", dir="/opt"))
    root.chmod(0o755)
    try:
        subprocess.run(
            [
                "useradd",
                "--no-create-home",
                "--user-group",
                "--shell",
                "/usr/sbin/nologin",
                user,
            ],
            check=True,
        )
        account = pwd.getpwnam(user)
        uid, gid = account.pw_uid, account.pw_gid
        require_worker_account(uid, gid)
        require_worker_idle(uid)
        for path in protected:
            path.mkdir(mode=0o700)
        protected[0].chmod(0o755)
        canary = protected[0] / "public-canary"
        canary.write_text("harmless public fixture")
        canary.chmod(0o444)
        secret = protected[0] / "fixture-key"
        secret.write_text("ephemeral fixture key material")
        secret.chmod(0o600)
        trusted = root / "trusted"
        trusted.write_text("unchanged trusted code")
        trusted.chmod(0o666)
        job = root / "job"
        job.mkdir(mode=0o700)
        os.chown(job, uid, gid)
        immutable_input = job / "request.json"
        immutable_input.write_text("immutable input")
        immutable_input.chmod(0o444)
        probe = root / "probe.py"
        probe.write_text("""import json, os, subprocess, sys
from pathlib import Path
root = Path(__file__).parent
result = {}
for name, operation in (
    ("namespace_read_denied", lambda: Path("/etc/axiom-producer/public-canary").read_bytes()),
    ("input_replacement_denied", lambda: (root / "job/request.json").unlink()),
    ("key_denied", lambda: Path("/etc/axiom-producer/fixture-key").read_bytes()),
    ("trusted_write_denied", lambda: (root / "trusted").write_text("changed")),
    ("controller_denied", lambda: Path("/run/axiom-producer/stolen").write_text("changed")),
):
    try:
        operation()
        result[name] = False
    except OSError:
        result[name] = True
result["uid"] = os.geteuid()
result["groups"] = os.getgroups()
(root / "job/result.json").write_text(json.dumps(result))
""")
        probe.chmod(0o444)
        executable = root / "python-fixture"
        executable.write_text('#!/bin/sh\nexec /usr/bin/python3 "' + str(probe) + '"\n')
        executable.chmod(0o555)
        command = systemd_command(
            job=job,
            python=str(executable),
            worker_uid=uid,
            worker_gid=gid,
            readonly=[trusted, probe, executable, immutable_input],
            timeout=30,
        )
        result = subprocess.run(command, capture_output=True, timeout=60)
        assert result.returncode == 0, result.stderr.decode()
        require_worker_idle(uid)
        import json

        observed = json.loads((job / "result.json").read_bytes())
        assert observed == {
            "namespace_read_denied": True,
            "input_replacement_denied": True,
            "key_denied": True,
            "trusted_write_denied": True,
            "controller_denied": True,
            "uid": uid,
            "groups": [gid],
        }
        assert trusted.read_text() == "unchanged trusted code"
    finally:
        subprocess.run(["systemctl", "stop", "axiom-producer-job"], capture_output=True)
        subprocess.run(["userdel", user], capture_output=True)
        for path in protected:
            shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(root, ignore_errors=True)
