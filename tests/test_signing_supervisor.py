"""Integration and adversarial tests for the compiled signing supervisor."""

from __future__ import annotations

import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import threading
from base64 import b64decode, b64encode
from contextlib import contextmanager
from pathlib import Path

import _cffi_backend
import cryptography
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).parents[1]
SUPERVISOR_PACKAGE = "./cmd/axiom-encode-signing-supervisor"
PRIVATE_ENV_NAMES = (
    "AXIOM_ENCODE_APPLY_SIGNING_KEY",
    "AXIOM_ENCODE_APPLY_SIGNING_PRIVATE_KEY",
    "AXIOM_ENCODE_EVAL_SIGNING_PRIVATE_KEY",
)
PUBLIC_ENV_NAMES = (
    "AXIOM_ENCODE_APPLY_SIGNING_PUBLIC_KEY",
    "AXIOM_ENCODE_EVAL_SIGNING_PUBLIC_KEY",
)
SIGNATURE_DOMAIN = b"axiom-encode/external-signer-sign/v2\0"


def _keypair(seed: bytes) -> tuple[str, Ed25519PrivateKey]:
    private_key = Ed25519PrivateKey.from_private_bytes(seed)
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return b64encode(public_key).decode("ascii"), private_key


@pytest.fixture(scope="session")
def signing_supervisor(tmp_path_factory: pytest.TempPathFactory) -> Path:
    go = shutil.which("go")
    if go is None:
        pytest.skip("Go is required to build the signing supervisor")
    build_dir = tmp_path_factory.mktemp("signing-supervisor-build").resolve()
    binary = build_dir / "axiom-encode-signing-supervisor-test-fixture"
    subprocess.run(
        [
            go,
            "build",
            "-trimpath",
            "-buildvcs=false",
            "-ldflags=-buildid=",
            "-tags=signing_supervisor_test_fixture",
            "-o",
            str(binary),
            SUPERVISOR_PACKAGE,
        ],
        cwd=ROOT,
        env={**os.environ, "CGO_ENABLED": "0"},
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return binary


@pytest.fixture(scope="session")
def trusted_python_runtime(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Path]:
    source_interpreter = Path(sys.executable).resolve()
    source_runtime = Path(sys.base_prefix).resolve()
    runtime_root = tmp_path_factory.mktemp("trusted-python-runtime").resolve()
    shutil.copytree(source_runtime, runtime_root, dirs_exist_ok=True, symlinks=False)
    for forbidden in (
        *runtime_root.rglob("*.pth"),
        *runtime_root.rglob("*.egg-link"),
        *runtime_root.rglob("sitecustomize.py"),
        *runtime_root.rglob("usercustomize.py"),
        *runtime_root.rglob("pyvenv.cfg"),
        *runtime_root.rglob("__editable__*"),
    ):
        if forbidden.is_file():
            forbidden.unlink()
    interpreter = runtime_root / source_interpreter.relative_to(source_runtime)
    site_packages = (
        runtime_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    site_packages.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        Path(cryptography.__file__).resolve().parent,
        site_packages / "cryptography",
        dirs_exist_ok=True,
        symlinks=False,
    )
    shutil.copy2(Path(_cffi_backend.__file__).resolve(), site_packages)
    package_root = site_packages / "axiom_encode"
    package_root.mkdir()
    (package_root / "__init__.py").write_text('"""Trusted fixture package."""\n')
    shutil.copy2(ROOT / "src" / "axiom_encode" / "signing_broker.py", package_root)
    shutil.copy2(
        ROOT / "src" / "axiom_encode" / "_trusted_signing_bootstrap.py",
        package_root,
    )
    (package_root / "entrypoint.py").write_text(
        """from __future__ import annotations
import json
import os
import subprocess
import sys
from base64 import b64encode

def main():
    from axiom_encode.signing_broker import (
        SigningBrokerError,
        get_signing_broker,
        scrub_private_signing_environment,
    )
    broker = get_signing_broker()
    descriptor = broker._connection.fileno()
    fork_read, fork_write = os.pipe()
    fork_pid = os.fork()
    if fork_pid == 0:
        os.close(fork_read)
        state = {}
        try:
            get_signing_broker()
        except SigningBrokerError:
            state[\"broker\"] = \"closed\"
        else:
            state[\"broker\"] = \"open\"
        try:
            os.fstat(descriptor)
        except OSError:
            state[\"descriptor\"] = \"closed\"
        else:
            state[\"descriptor\"] = \"open\"
        os.write(fork_write, json.dumps(state).encode())
        os._exit(0)
    os.close(fork_write)
    fork_state = json.loads(os.read(fork_read, 4096))
    os.close(fork_read)
    os.waitpid(fork_pid, 0)
    child_code = \"import json,os,sys; fd=int(sys.argv[1]); state='open';\\ntry: os.fstat(fd)\\nexcept OSError: state='closed'\\nprint(json.dumps({'environment':dict(os.environ),'descriptor':state}))\"
    child = subprocess.run(
        [sys.executable, \"-I\", \"-S\", \"-c\", child_code, str(descriptor)],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_private_signing_environment(),
    )
    result = {
        \"isolated\": sys.flags.isolated,
        \"no_site\": sys.flags.no_site,
        \"package_origin\": __import__(\"axiom_encode\").__file__,
        \"sys_path\": sys.path,
        \"environment\": dict(os.environ),
        \"child\": json.loads(child.stdout),
        \"fork\": fork_state,
        \"capabilities\": sorted(broker.capabilities),
    }
    if \"apply_ed25519\" in broker.capabilities:
        result[\"apply_signature\"] = b64encode(
            broker.apply_ed25519_sign(b\"compiled-apply-boundary\")
        ).decode(\"ascii\")
    if \"eval_ed25519\" in broker.capabilities:
        result[\"eval_signature\"] = b64encode(
            broker.eval_ed25519_sign(b\"compiled-eval-boundary\")
        ).decode(\"ascii\")
    print(json.dumps(result, sort_keys=True))
    broker.close()
    return 0
"""
    )
    return interpreter, runtime_root, package_root


def _runtime_arguments(runtime: tuple[Path, Path, Path]) -> list[str]:
    _interpreter, runtime_root, package_root = runtime
    return [
        "--trusted-python-runtime-root",
        str(runtime_root),
        "--trusted-python-import-root",
        str(package_root.parent),
        "--trusted-python-package-root",
        str(package_root),
    ]


def _launcher(tmp_path: Path, runtime: tuple[Path, Path, Path]) -> Path:
    interpreter, _runtime_root, _package_root = runtime
    launcher = tmp_path.resolve() / "axiom-encode"
    launcher.write_text(f"#!{interpreter} -I\nraise SystemExit('launcher executed')\n")
    launcher.chmod(0o700)
    return launcher


def _trust_config(
    tmp_path: Path,
    apply_public: str,
    eval_public: str,
) -> Path:
    path = tmp_path.resolve() / "signing-trust-roots.json"
    path.write_text(
        json.dumps(
            {
                "schema": "axiom-encode/signing-trust-roots/v1",
                "apply_ed25519_public_key": apply_public,
                "eval_ed25519_public_key": eval_public,
            },
            sort_keys=True,
        )
        + "\n"
    )
    path.chmod(0o600)
    return path


def _receive_exact(connection: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    while length:
        chunk = connection.recv(length)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        length -= len(chunk)
    return b"".join(chunks)


def _serve_external_signer(
    connection: socket.socket,
    private_key: Ed25519PrivateKey,
    behavior: str,
) -> None:
    try:
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        while True:
            try:
                header = _receive_exact(connection, 4)
            except EOFError:
                return
            request = json.loads(
                _receive_exact(connection, struct.unpack(">I", header)[0]).decode()
            )
            request_id = request["id"]
            scope = request.get("scope")
            if request.get("version") != 2:
                response = {
                    "version": 2,
                    "id": request_id,
                    "ok": False,
                    "error": "unsupported protocol",
                }
            elif request.get("op") == "challenge":
                nonce = b64decode(request["challenge"], validate=True)
                message = (
                    b"axiom-encode/external-signer-challenge/v2\0"
                    + scope.encode("ascii")
                    + b"\0"
                    + nonce
                )
                signature = private_key.sign(message)
                if behavior == "wrong_challenge_signature":
                    signature = private_key.sign(b"wrong")
                response = {
                    "version": 2,
                    "id": request_id,
                    "ok": True,
                    "public_key": b64encode(public_key).decode("ascii"),
                    "signature": b64encode(signature).decode("ascii"),
                }
                if behavior == "extra_challenge_field":
                    response["legacy"] = True
            elif request.get("op") == "sign":
                payload = b64decode(request["payload"], validate=True)
                message = SIGNATURE_DOMAIN + scope.encode("ascii") + b"\0" + payload
                signature = private_key.sign(message)
                if behavior == "wrong_sign_signature":
                    signature = private_key.sign(payload)
                response = {
                    "version": 2,
                    "id": request_id,
                    "ok": True,
                    "signature": b64encode(signature).decode("ascii"),
                }
                if behavior == "extra_sign_field":
                    response["legacy"] = True
            else:
                response = {
                    "version": 2,
                    "id": request_id,
                    "ok": False,
                    "error": "unsupported request",
                }
            if behavior == "legacy_v1_response":
                response["version"] = 1
            raw = json.dumps(response, separators=(",", ":"), sort_keys=True).encode()
            connection.sendall(struct.pack(">I", len(raw)) + raw)
    finally:
        connection.close()


@contextmanager
def _signers(*keys: Ed25519PrivateKey, behavior: str = "valid"):
    supervisor_connections: list[socket.socket] = []
    threads: list[threading.Thread] = []
    for key in keys:
        signer_connection, supervisor_connection = socket.socketpair()
        thread = threading.Thread(
            target=_serve_external_signer,
            args=(signer_connection, key, behavior),
            daemon=True,
        )
        thread.start()
        supervisor_connections.append(supervisor_connection)
        threads.append(thread)
    try:
        yield [connection.fileno() for connection in supervisor_connections]
    finally:
        for connection in supervisor_connections:
            connection.close()
        for thread in threads:
            thread.join(timeout=5)


def _invoke(
    supervisor: Path,
    runtime: tuple[Path, Path, Path],
    launcher: Path,
    trust_config: Path,
    descriptors: list[int],
    *,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    signer_arguments: list[str] = []
    if descriptors:
        signer_arguments.extend(("--apply-signer-fd", str(descriptors[0])))
    if len(descriptors) > 1:
        signer_arguments.extend(("--eval-signer-fd", str(descriptors[1])))
    return subprocess.run(
        [
            str(supervisor),
            *signer_arguments,
            "--trusted-signing-roots",
            str(trust_config),
            *_runtime_arguments(runtime),
            "--",
            str(launcher),
        ],
        env={} if environment is None else environment,
        pass_fds=tuple(descriptors),
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_fixture_binary_is_explicitly_nonpublishable(signing_supervisor: Path) -> None:
    completed = subprocess.run(
        [str(signing_supervisor), "--build-kind"], capture_output=True, text=True
    )
    assert completed.stdout.strip() == "test-fixture-nonpublishable"


def test_every_invocation_requires_protected_dual_root_config(
    signing_supervisor: Path,
) -> None:
    completed = subprocess.run(
        [str(signing_supervisor), "--", "/nonexistent/axiom-encode"],
        env={},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "--trusted-signing-roots is required" in completed.stderr


@pytest.mark.skipif(
    "AXIOM_ENCODE_PRODUCTION_SIGNING_FIXTURE" not in os.environ,
    reason="root-owned production fixture is prepared only in signing CI",
)
def test_untagged_production_binary_root_owned_end_to_end() -> None:
    fixture = Path(os.environ["AXIOM_ENCODE_PRODUCTION_SIGNING_FIXTURE"])
    supervisor = fixture / "axiom-encode-signing-supervisor"
    runtime_root = fixture / "python"
    interpreter = runtime_root / Path(sys.executable).resolve().relative_to(
        Path(sys.base_prefix).resolve()
    )
    package_root = (
        runtime_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages/axiom_encode"
    )
    completed_kind = subprocess.run(
        [str(supervisor), "--build-kind"], capture_output=True, text=True
    )
    assert completed_kind.stdout.strip() == "production"
    apply_public, apply_key = _keypair(b"\xab" * 32)
    eval_public, eval_key = _keypair(b"\xcd" * 32)
    with _signers(apply_key, eval_key) as descriptors:
        completed = _invoke(
            supervisor,
            (interpreter, runtime_root, package_root),
            fixture / "axiom-encode",
            fixture / "signing-trust-roots.json",
            descriptors,
        )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    apply_key.public_key().verify(
        b64decode(result["apply"]),
        SIGNATURE_DOMAIN + b"apply_ed25519\0production-apply",
    )
    eval_key.public_key().verify(
        b64decode(result["eval"]),
        SIGNATURE_DOMAIN + b"eval_ed25519\0production-eval",
    )
    assert apply_public != eval_public


def test_compiled_supervisor_uses_isolated_direct_runtime_and_domain_signatures(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    apply_public, apply_key = _keypair(b"\xab" * 32)
    eval_public, eval_key = _keypair(b"\xcd" * 32)
    launcher = _launcher(tmp_path, trusted_python_runtime)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    sentinels = {
        "PATH": "/hostile/bin",
        "PYTHONPATH": "/hostile/python",
        "GIT_CONFIG_GLOBAL": "/hostile/gitconfig",
        "AWS_SECRET_ACCESS_KEY": "aws-sentinel",
        "GH_TOKEN": "gh-sentinel",
        "AXIOM_ENCODE_SUPABASE_SECRET_KEY": "supabase-sentinel",
        "OTEL_EXPORTER_OTLP_HEADERS": "otel-sentinel",
        "HTTPS_PROXY": "http://proxy.invalid",
    }
    with _signers(apply_key, eval_key) as descriptors:
        completed = _invoke(
            signing_supervisor,
            trusted_python_runtime,
            launcher,
            trust_config,
            descriptors,
            environment=sentinels,
        )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["isolated"] == 1
    assert result["no_site"] == 1
    assert Path(result["package_origin"]) == trusted_python_runtime[2] / "__init__.py"
    assert result["capabilities"] == ["apply_ed25519", "eval_ed25519"]
    for name in (
        *PRIVATE_ENV_NAMES,
        "AXIOM_ENCODE_SIGNING_BROKER_FD",
        "AXIOM_ENCODE_SIGNING_BROKER_PID",
        "AXIOM_ENCODE_SIGNING_BROKER_ACTIVE",
    ):
        assert name not in result["environment"]
    assert all(
        Path(path).is_relative_to(trusted_python_runtime[1])
        for path in result["sys_path"]
    )
    parent_only = {
        "AXIOM_ENCODE_SUPABASE_SECRET_KEY",
        "OTEL_EXPORTER_OTLP_HEADERS",
    }
    for name, value in sentinels.items():
        if name in parent_only:
            assert result["environment"][name] == value
        elif name in {"PATH", "GIT_CONFIG_GLOBAL"}:
            assert result["environment"][name] != value
        else:
            assert name not in result["environment"]
        if name not in parent_only:
            assert value not in completed.stdout
    assert {"LANG", "PYTHONDONTWRITEBYTECODE"} <= set(result["child"]["environment"])
    for name, value in sentinels.items():
        assert result["child"]["environment"].get(name) != value
    assert result["child"]["environment"]["PATH"] == result["environment"]["PATH"]
    assert result["child"]["environment"]["HOME"] == result["environment"]["HOME"]
    assert result["child"]["environment"]["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert result["child"]["descriptor"] == "closed"
    assert result["fork"] == {"broker": "closed", "descriptor": "closed"}
    apply_key.public_key().verify(
        b64decode(result["apply_signature"]),
        SIGNATURE_DOMAIN + b"apply_ed25519\0compiled-apply-boundary",
    )
    eval_key.public_key().verify(
        b64decode(result["eval_signature"]),
        SIGNATURE_DOMAIN + b"eval_ed25519\0compiled-eval-boundary",
    )
    with pytest.raises(Exception):
        eval_key.public_key().verify(
            b64decode(result["apply_signature"]),
            SIGNATURE_DOMAIN + b"eval_ed25519\0compiled-apply-boundary",
        )


def test_verification_only_invocation_exposes_roots_without_signing_capability(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    apply_public, _apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    completed = _invoke(
        signing_supervisor,
        trusted_python_runtime,
        _launcher(tmp_path, trusted_python_runtime),
        _trust_config(tmp_path, apply_public, eval_public),
        [],
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["capabilities"] == []


@pytest.mark.parametrize("name", PRIVATE_ENV_NAMES)
@pytest.mark.parametrize("value", ["secret", ""])
def test_every_legacy_and_current_private_environment_name_is_fatal(
    signing_supervisor: Path,
    name: str,
    value: str,
) -> None:
    completed = subprocess.run(
        [str(signing_supervisor), "--help"],
        env={name: value},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert name in completed.stderr


@pytest.mark.parametrize("name", PUBLIC_ENV_NAMES)
@pytest.mark.parametrize("value", ["counterfeit", ""])
def test_environment_public_roots_cannot_define_trust(
    signing_supervisor: Path,
    name: str,
    value: str,
) -> None:
    completed = subprocess.run(
        [str(signing_supervisor), "--help"],
        env={name: value},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "must come from --trusted-signing-roots" in completed.stderr


def test_same_roots_fail_even_for_one_capability_invocation(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    public, key = _keypair(b"\xab" * 32)
    launcher = _launcher(tmp_path, trusted_python_runtime)
    trust_config = _trust_config(tmp_path, public, public)
    with _signers(key) as descriptors:
        completed = _invoke(
            signing_supervisor,
            trusted_python_runtime,
            launcher,
            trust_config,
            descriptors,
        )
    assert completed.returncode == 2
    assert "must be distinct" in completed.stderr


@pytest.mark.parametrize("mutation", ["old_schema", "missing_eval", "extra_field"])
def test_trust_config_is_exact_and_has_no_legacy_shape(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
    mutation: str,
) -> None:
    apply_public, apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    launcher = _launcher(tmp_path, trusted_python_runtime)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    payload = json.loads(trust_config.read_text())
    if mutation == "old_schema":
        payload["schema"] = "axiom-encode/signing-trust-roots/v0"
    elif mutation == "missing_eval":
        payload.pop("eval_ed25519_public_key")
    else:
        payload["legacy"] = True
    trust_config.write_text(json.dumps(payload) + "\n")
    with _signers(apply_key) as descriptors:
        completed = _invoke(
            signing_supervisor,
            trusted_python_runtime,
            launcher,
            trust_config,
            descriptors,
        )
    assert completed.returncode == 2
    assert "trust-root config" in completed.stderr


@pytest.mark.parametrize(
    "forbidden_name",
    ["attack.pth", "sitecustomize.py", "pyvenv.cfg", "__editable__attack.py"],
)
def test_runtime_startup_and_editable_injection_is_rejected_before_attachment(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
    forbidden_name: str,
) -> None:
    apply_public, apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    launcher = _launcher(tmp_path, trusted_python_runtime)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    forbidden = trusted_python_runtime[2].parent / forbidden_name
    forbidden.write_text("raise SystemExit('injected')\n")
    try:
        with _signers(apply_key) as descriptors:
            completed = _invoke(
                signing_supervisor,
                trusted_python_runtime,
                launcher,
                trust_config,
                descriptors,
            )
    finally:
        forbidden.unlink()
    assert completed.returncode == 2
    assert "forbidden startup or editable injection" in completed.stderr
    assert completed.stdout == ""


@pytest.mark.parametrize(
    ("behavior", "expected"),
    [
        ("wrong_challenge_signature", "challenge response is invalid"),
        ("legacy_v1_response", "initialization failed"),
        ("extra_challenge_field", "initialization failed"),
        ("wrong_sign_signature", "External apply signer failed"),
        ("extra_sign_field", "External apply signer failed"),
    ],
)
def test_invalid_external_signer_response_fails_closed(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
    behavior: str,
    expected: str,
) -> None:
    apply_public, apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    launcher = _launcher(tmp_path, trusted_python_runtime)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    with _signers(apply_key, behavior=behavior) as descriptors:
        completed = _invoke(
            signing_supervisor,
            trusted_python_runtime,
            launcher,
            trust_config,
            descriptors,
        )
    assert completed.returncode != 0
    assert expected in completed.stderr


def test_non_socket_signer_descriptor_is_rejected(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    apply_public, _apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    launcher = _launcher(tmp_path, trusted_python_runtime)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    read_fd, write_fd = os.pipe()
    try:
        completed = _invoke(
            signing_supervisor,
            trusted_python_runtime,
            launcher,
            trust_config,
            [read_fd],
        )
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert completed.returncode == 2
    assert "connected socket" in completed.stderr
