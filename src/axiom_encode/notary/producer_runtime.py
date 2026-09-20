"""Linux/systemd reference isolation for a host-custodied producer.

The worker is a dedicated, non-login UID and has no signing keys. This module
refuses to execute on another OS or as an unprivileged controller. The caller
must serialize runs; a live process for the worker UID refuses admission.
"""

from __future__ import annotations

import base64
import os
import platform
import pwd
import re
import stat
import subprocess
from pathlib import Path

from ._schema import digest, fields, relative_path
from .canonical import jcs_dumps, sha256_hex, strict_parse
from .deployment import custodian_file, custodian_parent, require_running_identity
from .identity import IdentityRefusal
from .remote import RemoteRepository


def private_observation(root: Path, relative: str, *, limit=32_000_000) -> bytes:
    """Open each path component without following a worker-created link."""
    if not relative_path(relative) or relative.startswith("/") or "\\" in relative:
        raise IdentityRefusal("runtime_output_path")
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = relative.split("/")
        for name in parts[:-1]:
            child = os.open(
                name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = child
        leaf = os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
        )
        try:
            info = os.fstat(leaf)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_size > limit
                or info.st_mode & 0o111
            ):
                raise IdentityRefusal("runtime_output_file")
            with os.fdopen(leaf, "rb", closefd=False) as stream:
                raw = stream.read(limit + 1)
            if len(raw) > limit:
                raise IdentityRefusal("runtime_output_size")
            return raw
        finally:
            os.close(leaf)
    finally:
        os.close(fd)


def require_worker_idle(uid: int):
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            lines = (entry / "status").read_text().splitlines()
        except FileNotFoundError:
            continue  # Process exited during enumeration.
        for line in lines:
            if line.startswith("Uid:") and uid in map(int, line.split()[1:]):
                raise IdentityRefusal("runtime_worker_not_idle")


def require_worker_account(uid: int, gid: int):
    account = pwd.getpwuid(uid)
    if (
        account.pw_gid != gid
        or not account.pw_shell.endswith(("/nologin", "/false"))
        or set(os.getgrouplist(account.pw_name, gid)) != {gid}
    ):
        raise IdentityRefusal("runtime_worker_login_or_groups")


def readable_immutable_tree(root: Path):
    for directory, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(directory) / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or (
                not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode)
            ):
                raise IdentityRefusal("runtime_checkout_entry")
            path.chmod(
                0o555 if stat.S_ISDIR(info.st_mode) or info.st_mode & 0o111 else 0o444
            )
        Path(directory).chmod(0o555)


def expected_outputs(citation: str, model: str):
    from axiom_encode.harness.evals import (
        _resolve_eval_output_path,
        _rulespec_test_path,
        parse_runner_spec,
    )

    module = _resolve_eval_output_path(citation)
    paths = (module.as_posix(), _rulespec_test_path(module).as_posix())
    runner = parse_runner_spec("codex:" + model)
    return {path: f"work/output/{runner.name}/{path}" for path in paths}


def read_encoder_outputs(job, result, expected):
    if not isinstance(result.get("files"), dict) or result["files"] != expected:
        raise IdentityRefusal("runtime_output_mapping")
    outputs = {
        path: private_observation(job, relative) for path, relative in expected.items()
    }
    if sum(map(len, outputs.values())) > 64_000_000:
        raise IdentityRefusal("runtime_output_size")
    return outputs


def trusted_executable(path: str):
    file = Path(path)
    custodian_parent(file)
    info = file.stat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != 0
        or info.st_mode & 0o022
        or not os.access(file, os.X_OK)
    ):
        raise IdentityRefusal("runtime_executable_custody")


def systemd_command(
    *,
    job: Path,
    python: str,
    worker_uid: int,
    worker_gid: int,
    readonly: list[Path],
    timeout: int,
    safe_directories: list[Path] = (),
):
    # No caller-controlled command, environment, properties, or scope selection.
    # No signing key directory, GitHub token, or host login enters the worker.
    unit = "axiom-producer-" + job.name
    properties = {
        "User": str(worker_uid),
        "Group": str(worker_gid),
        "SupplementaryGroups": "",
        "Type": "exec",
        "ExitType": "cgroup",
        "KillMode": "control-group",
        "RuntimeMaxSec": str(timeout),
        "TimeoutStopSec": "10",
        "NoNewPrivileges": "yes",
        "CapabilityBoundingSet": "",
        "AmbientCapabilities": "",
        "ProtectSystem": "strict",
        "ProtectHome": "yes",
        "PrivateTmp": "yes",
        "PrivateDevices": "yes",
        "ProtectKernelTunables": "yes",
        "ProtectKernelModules": "yes",
        "ProtectKernelLogs": "yes",
        "ProtectControlGroups": "yes",
        "RestrictSUIDSGID": "yes",
        "LockPersonality": "yes",
        "RestrictRealtime": "yes",
        "UMask": "0077",
        "LimitCORE": "0",
        "SystemCallFilter": "~ptrace process_vm_readv process_vm_writev kcmp bpf perf_event_open userfaultfd",
        "InaccessiblePaths": "/etc/axiom-producer /run/axiom-producer /run/dbus /run/systemd/private",
        "ReadWritePaths": str(job),
        "ReadOnlyPaths": " ".join(map(str, readonly)),
        "WorkingDirectory": str(job),
        "StandardOutput": "null",
        "StandardError": "null",
        "TasksMax": "256",
        "MemoryMax": "8G",
    }
    return (
        ["/usr/bin/systemd-run", "--quiet", "--wait", "--collect", "--unit=" + unit]
        + ["--property=" + key + "=" + value for key, value in properties.items()]
        + [
            "--setenv=PATH=/usr/bin:/bin",
            "--setenv=PYTHONDONTWRITEBYTECODE=1",
            "--setenv=GIT_OPTIONAL_LOCKS=0",
            "--setenv=GIT_CONFIG_NOSYSTEM=1",
            "--setenv=GIT_CONFIG_GLOBAL=/dev/null",
            "--setenv=GIT_CONFIG_COUNT=" + str(len(safe_directories)),
        ]
        + [
            item
            for index, path in enumerate(safe_directories)
            for item in (
                f"--setenv=GIT_CONFIG_KEY_{index}=safe.directory",
                f"--setenv=GIT_CONFIG_VALUE_{index}={path}",
            )
        ]
        + [
            python,
            "-I",
            "-m",
            "axiom_encode.notary.producer_worker",
            str(job),
        ]
    )


class LinuxRuntime:
    def __init__(self, config: dict):
        if platform.system() != "Linux" or os.geteuid() != 0:
            raise IdentityRefusal("runtime_requires_linux_custodian")
        self.config = config
        uid, gid = config["worker_uid"], config["worker_gid"]
        if type(uid) is not int or uid < 1000 or type(gid) is not int or gid <= 0:
            raise IdentityRefusal("runtime_worker_identity")
        require_worker_account(uid, gid)
        for path in (config["python"], config["codex_binary"]):
            trusted_executable(path)
        version = subprocess.run(
            ["/usr/bin/systemd-run", "--version"],
            capture_output=True,
            check=True,
            env={"PATH": "/usr/bin:/bin"},
        ).stdout
        match = re.match(rb"systemd ([0-9]+)", version)
        if not match or int(match[1]) < 250:
            raise IdentityRefusal("runtime_systemd_version")
        self.worker_uid, self.worker_gid = uid, gid

    def run(
        self,
        *,
        job: Path,
        request: dict,
        lane_remote: RemoteRepository,
        base,
        enrollment,
        auth: bytes,
        corpus_public_keys: list[bytes],
    ):
        require_worker_account(self.worker_uid, self.worker_gid)
        require_worker_idle(self.worker_uid)
        require_running_identity(
            self.config["encoder_identity"], self.config["dependency_inventory"]
        )
        binary = custodian_file(self.config["codex_binary"], limit=250_000_000)
        if sha256_hex(binary) != enrollment.body["codex_cli"]["sha256"]:
            raise IdentityRefusal("runtime_codex_identity")
        version = (
            subprocess.run(
                [self.config["codex_binary"], "--version"],
                capture_output=True,
                check=True,
                timeout=30,
                env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
            )
            .stdout.decode()
            .strip()
        )
        if version != enrollment.body["codex_cli"]["version"]:
            raise IdentityRefusal("runtime_codex_version")
        lane_path = job / self.config["lane"].split("/")[1]
        lane_remote._run("update-ref", "refs/heads/main", base.commit)
        lane_remote._run("symbolic-ref", "HEAD", "refs/heads/main")
        lane_remote._run(
            "clone",
            "--no-hardlinks",
            "--no-checkout",
            str(lane_remote.path),
            str(lane_path),
            in_repo=False,
        )

        def git(*args):
            lane_remote._run("-C", str(lane_path), *args, in_repo=False)

        git("config", "core.hooksPath", "/dev/null")
        git(
            "config",
            "remote.origin.url",
            "https://github.com/" + self.config["lane"] + ".git",
        )
        git("reset", "--hard", base.commit)
        from axiom_encode.harness.evals import resolve_corpus_source_unit
        from axiom_encode.toolchain import (
            load_rulespec_local_corpus_release,
            local_corpus_release_verification,
        )

        source = None
        selected_public_key = None
        for raw_public in corpus_public_keys:
            if not isinstance(raw_public, bytes) or len(raw_public) != 32:
                raise IdentityRefusal("runtime_corpus_public_key")
            public_key = base64.b64encode(raw_public).decode()
            try:
                with local_corpus_release_verification(public_key):
                    release = load_rulespec_local_corpus_release(
                        lane_path, Path(self.config["corpus_path"])
                    )
                    source = resolve_corpus_source_unit(request["citation"], release)
                selected_public_key = public_key
                break
            except Exception:
                # All fallback roots are from the authenticated registry;
                # transport/parse/verification failure never adds a new root.
                continue
        if source is None:
            raise IdentityRefusal("runtime_corpus_capture")
        source_bytes, source_id = source.body.encode(), source.citation_path
        expected = expected_outputs(source.requested, self.config["model"])
        readable_immutable_tree(lane_path)
        worker_request = {
            "citation": request["citation"],
            "corpus_public_key": selected_public_key,
            "model": self.config["model"],
            "codex_binary": self.config["codex_binary"],
            "output_root": str(job / "work" / "output"),
            "corpus_path": self.config["corpus_path"],
            "engine_path": self.config["engine_path"],
            "lane_path": str(lane_path),
            "dependency_roots": self.config["dependency_roots"],
            "encoder_identity": self.config["encoder_identity"],
            "dependency_inventory": strict_parse(self.config["dependency_inventory"]),
        }
        (job / "request.json").write_bytes(jcs_dumps(worker_request))
        (job / "request.json").chmod(0o444)
        auth_path = job / "auth.json"
        auth_path.write_bytes(auth)
        auth_path.chmod(0o600)
        os.chown(auth_path, self.worker_uid, self.worker_gid)
        os.chown(job, self.worker_uid, self.worker_gid)
        job.chmod(0o700)
        readonly = [
            job / "request.json",
            lane_path,
            Path(self.config["corpus_path"]),
            Path(self.config["engine_path"]),
        ] + [Path(path) for path in self.config["dependency_roots"].values()]
        command = systemd_command(
            job=job,
            python=self.config["python"],
            worker_uid=self.worker_uid,
            worker_gid=self.worker_gid,
            readonly=readonly,
            timeout=self.config["timeout_seconds"],
            safe_directories=[path for path in readonly if path.is_dir()]
            + (
                [Path(__file__).resolve().parents[3]]
                if (Path(__file__).resolve().parents[3] / ".git").exists()
                else []
            ),
        )
        succeeded = False
        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
                timeout=self.config["timeout_seconds"] + 60,
            )
            succeeded = result.returncode == 0
        finally:
            # Always stop the cgroup before inspecting output, even on timeout.
            subprocess.run(
                ["/usr/bin/systemctl", "stop", "axiom-producer-" + job.name],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                env={"PATH": "/usr/bin:/bin"},
                timeout=30,
            )
            os.chown(job, 0, 0)
            job.chmod(0o700)
        require_worker_idle(self.worker_uid)
        if not succeeded:
            raise IdentityRefusal("runtime_encoder_failed")
        # Recheck the trusted installation after execution, never use a worker
        # claim as the encoder or Codex identity.
        require_running_identity(
            self.config["encoder_identity"], self.config["dependency_inventory"]
        )
        if (
            sha256_hex(custodian_file(self.config["codex_binary"], limit=250_000_000))
            != enrollment.body["codex_cli"]["sha256"]
        ):
            raise IdentityRefusal("runtime_codex_identity")
        result = strict_parse(private_observation(job, "result.json", limit=131072))
        if (
            not fields(
                result,
                {
                    "success",
                    "files",
                    "model",
                    "prompt_sha256s",
                },
            )
            or result["success"] is not True
            or result["model"] != self.config["model"]
            or not isinstance(result["files"], dict)
            or not result["files"]
            or len(result["files"]) > 1000
            or not isinstance(result["prompt_sha256s"], list)
            or not result["prompt_sha256s"]
            or not all(digest(p) for p in result["prompt_sha256s"])
        ):
            raise IdentityRefusal("runtime_encoder_result")
        outputs = read_encoder_outputs(job, result, expected)
        refreshed_auth = private_observation(
            job, "work/codex-home/auth.json", limit=131072
        )
        return {
            "outputs": outputs,
            "model": result["model"],
            "prompts": result["prompt_sha256s"],
            "source_bytes": source_bytes,
            "source_id": source_id,
        }, refreshed_auth
