"""Measured, offline deterministic adapter execution on the supervised host.

The enrolled adapter receives one JSON descriptor: immutable input root,
fixed parameters, exact output paths, and a fresh output directory. It holds
no credentials or signing keys. The controller measures and observes bytes.
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path

from .canonical import jcs_dumps, sha256_hex, strict_parse
from .deployment import custodian_file, custodian_parent, require_running_identity
from .identity import IdentityRefusal
from .producer_runtime import (
    LinuxRuntime,
    private_observation,
    readable_immutable_tree,
    require_worker_account,
    require_worker_idle,
    systemd_command,
)


def measure_runtime(root: Path) -> str:
    """Measure every file/mode in the installed Python+dependency tree.

    Deployment must materialize a root-owned tree without symlinks or hardlinks;
    the interpreter, stdlib, site-packages and encoder are part of this measure.
    OS libraries remain part of the audited host image/custody boundary.
    """
    custodian_parent(root / "measurement")

    def walk_error(error):
        raise IdentityRefusal("deterministic_runtime_unreadable") from error

    rows = []
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=walk_error):
        for name in sorted(dirs + files):
            path = Path(directory) / name
            info = path.lstat()
            if info.st_uid != 0 or info.st_mode & 0o022:
                raise IdentityRefusal("deterministic_runtime_custody")
            if stat.S_ISDIR(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise IdentityRefusal("deterministic_runtime_entry")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            rows.append(
                [
                    path.relative_to(root).as_posix(),
                    stat.S_IMODE(info.st_mode),
                    digest.hexdigest(),
                ]
            )
    if not rows:
        raise IdentityRefusal("deterministic_runtime_empty")
    return sha256_hex(jcs_dumps(sorted(rows)))


def captured_files(root: Path, inventory):
    captured = {}
    for row in inventory:
        raw = custodian_file(str(root / row["path"]), limit=64_000_000)
        if sha256_hex(raw) != row["sha256"]:
            raise IdentityRefusal("deterministic_file_measurement")
        captured[row["path"]] = raw
    if sum(map(len, captured.values())) > 512_000_000:
        raise IdentityRefusal("deterministic_inputs_size")
    return captured


def stage_files(root, files):
    root.mkdir(mode=0o755, parents=True)
    for path, raw in files.items():
        target = root / path
        target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        target.write_bytes(raw)
    readable_immutable_tree(root)


def observe_outputs(job, relative, expected):
    root = job / relative
    if root.is_symlink() or not root.is_dir():
        raise IdentityRefusal("deterministic_output_directory")
    actual = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs:
            if (Path(directory) / name).is_symlink():
                raise IdentityRefusal("deterministic_output_directory")
        actual.extend(
            (Path(directory) / name).relative_to(root).as_posix() for name in files
        )
    if sorted(actual) != expected:
        raise IdentityRefusal("deterministic_output_mapping")
    output = {
        path: private_observation(job, relative + "/" + path) for path in expected
    }
    if sum(map(len, output.values())) > 64_000_000:
        raise IdentityRefusal("deterministic_output_size")
    return output


def run_adapter(job, request):
    """Run one adapter invocation; the controller isolates repeat attempts."""
    work = job / "work"
    work.mkdir(mode=0o700)
    output = work / "output"
    output.mkdir(mode=0o700)
    adapter = job / "immutable/generator" / request["generator"]["entrypoint"]
    descriptor = work / "invocation.json"
    descriptor.write_bytes(
        jcs_dumps(
            {
                "schema": "axiom/deterministic-adapter-request/v1",
                "input_root": str(job / "immutable/inputs"),
                "output_root": str(output),
                "parameters": request["parameters"],
                "outputs": request["outputs"],
            }
        )
    )
    subprocess.run(
        [sys.executable, "-I", "-B", str(adapter), str(descriptor)],
        cwd=work,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(work),
            "TMPDIR": str(work),
            "LC_ALL": "C",
            "SOURCE_DATE_EPOCH": "0",
        },
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


class LinuxDeterministicRuntime(LinuxRuntime):
    def _measure(self, enrollment):
        entry = enrollment.body
        if enrollment.kind != "deterministic":
            raise IdentityRefusal("deterministic_runtime_kind")
        root = Path(self.config["runtime_root"])
        if self.config["python"] != str(root / entry["runtime"]["python_path"]):
            raise IdentityRefusal("deterministic_python_binding")
        if measure_runtime(root) != entry["runtime"]["tree_sha256"]:
            raise IdentityRefusal("deterministic_runtime_measurement")
        return (
            captured_files(
                Path(self.config["generator_root"]), entry["generator"]["files"]
            ),
            captured_files(Path(self.config["input_root"]), entry["inputs"]),
        )

    def run(self, *, job, request, base, enrollment, auth, **unused):
        if (
            auth is not None
            or request.keys() != {"operation", "run_id"}
            or request["operation"] != "generate"
        ):
            raise IdentityRefusal("deterministic_request")
        entry = enrollment.body
        require_worker_account(self.worker_uid, self.worker_gid)
        require_worker_idle(self.worker_uid)
        require_running_identity(
            self.config["encoder_identity"], self.config["dependency_inventory"]
        )
        generator, inputs = self._measure(enrollment)
        worker_request = {
            "runtime_kind": "deterministic",
            "generator": entry["generator"],
            "parameters": entry["parameters"],
            "outputs": entry["outputs"],
            "encoder_identity": self.config["encoder_identity"],
            "dependency_inventory": strict_parse(self.config["dependency_inventory"]),
        }
        # Each attempt has its own cgroup and directory. The controller observes
        # and reclaims the first before a second worker exists. Its captured
        # bytes stay in controller memory, outside both workers' authority.
        job.chmod(0o711)
        observations = []
        for label in ("first", "second"):
            attempt = job / (job.name + "-" + label)
            attempt.mkdir(mode=0o700)
            stage_files(attempt / "immutable/generator", generator)
            stage_files(attempt / "immutable/inputs", inputs)
            readable_immutable_tree(attempt / "immutable")
            self._execute(attempt, worker_request)
            require_running_identity(
                self.config["encoder_identity"], self.config["dependency_inventory"]
            )
            if self._measure(enrollment) != (generator, inputs):
                raise IdentityRefusal("deterministic_inputs_changed")
            observations.append(
                observe_outputs(attempt, "work/output", entry["outputs"])
            )
        job.chmod(0o700)
        if observations[0] != observations[1]:
            raise IdentityRefusal("deterministic_output_mismatch")
        return {
            "outputs": observations[0],
            "generator": entry["generator"],
            "runtime": entry["runtime"],
            "inputs": entry["inputs"],
            "parameters": entry["parameters"],
        }, None

    def _execute(self, job, worker_request):
        require_worker_idle(self.worker_uid)
        (job / "request.json").write_bytes(jcs_dumps(worker_request))
        (job / "request.json").chmod(0o444)
        os.chown(job, self.worker_uid, self.worker_gid)
        job.chmod(0o700)
        command = systemd_command(
            job=job,
            python=self.config["python"],
            worker_uid=self.worker_uid,
            worker_gid=self.worker_gid,
            readonly=[
                job / "immutable",
                job / "request.json",
                Path(self.config["runtime_root"]),
            ],
            timeout=self.config["timeout_seconds"],
            deterministic=True,
            runtime_root=Path(self.config["runtime_root"]),
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
            raise IdentityRefusal("deterministic_generator_failed")
