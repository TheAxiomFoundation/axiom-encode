"""Integration and adversarial tests for the compiled signing supervisor."""

from __future__ import annotations

import hashlib
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

from axiom_encode import __version__
from axiom_encode.cli import (
    APPLIED_ENCODING_MANIFEST_SCHEMA,
    APPLIED_ENCODING_MODEL_TOOL,
    _sign_applied_encoding_manifest,
)
from axiom_encode.harness.evals import resolve_corpus_source_unit
from tests.release_object_fixtures import bind_test_corpus_release
from tests.signing_broker_fixtures import SigningBrokerFixture

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
    "AXIOM_CORPUS_RELEASE_PUBLIC_KEY",
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
        \"roots\": {
            \"apply\": b64encode(broker.apply_public_key_raw).decode(\"ascii\"),
            \"eval\": b64encode(broker.eval_public_key_raw).decode(\"ascii\"),
            \"corpus_release\": b64encode(
                broker.corpus_release_public_key_raw
            ).decode(\"ascii\"),
        },
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


@pytest.fixture(scope="session")
def trusted_real_cli_runtime(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Path]:
    """Build a hermetic runtime containing the real CLI and its dependencies."""

    source_interpreter = Path(sys.executable).resolve()
    source_runtime = Path(sys.base_prefix).resolve()
    runtime_root = tmp_path_factory.mktemp("trusted-real-cli-runtime").resolve()
    shutil.copytree(source_runtime, runtime_root, dirs_exist_ok=True, symlinks=False)
    interpreter = runtime_root / source_interpreter.relative_to(source_runtime)
    site_packages = (
        runtime_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    source_site_packages = Path(pytest.__file__).resolve().parents[1]
    shutil.copytree(
        source_site_packages,
        site_packages,
        dirs_exist_ok=True,
        symlinks=False,
        ignore=shutil.ignore_patterns(
            "axiom_encode",
            "*.pth",
            "*.egg-link",
            "sitecustomize.py",
            "usercustomize.py",
            "pyvenv.cfg",
            "__editable__*",
            "__pycache__",
            "*.pyc",
        ),
    )
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

    package_root = runtime_root / "src" / "axiom_encode"
    shutil.copytree(
        ROOT / "src" / "axiom_encode",
        package_root,
        symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(ROOT / "pyproject.toml", runtime_root)
    shutil.copy2(ROOT / "uv.lock", runtime_root)
    (runtime_root / ".gitignore").write_text(
        "*\n"
        "!/.gitignore\n"
        "!/pyproject.toml\n"
        "!/uv.lock\n"
        "!/src/\n"
        "!/src/axiom_encode/\n"
        "!/src/axiom_encode/**\n"
        "/src/axiom_encode/**/__pycache__/\n"
        "/src/axiom_encode/**/*.pyc\n"
    )

    real_git = shutil.which("git")
    if real_git is None:
        pytest.skip("Git is required for guarded encoder identity verification")
    git_wrapper = interpreter.parent / "git"
    git_wrapper.write_text(
        f"#!{interpreter}\n"
        "import os\n"
        "import sys\n"
        f"git = {str(Path(real_git).resolve())!r}\n"
        "os.execv(git, [git, *sys.argv[1:]])\n"
    )
    git_wrapper.chmod(0o700)

    git_environment = {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "HOME": str(runtime_root),
        "PATH": os.environ.get("PATH", ""),
    }
    subprocess.run(
        [real_git, "init", "--quiet", str(runtime_root)],
        check=True,
        env=git_environment,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            real_git,
            "-C",
            str(runtime_root),
            "add",
            ".gitignore",
            "pyproject.toml",
            "uv.lock",
            "src/axiom_encode",
        ],
        check=True,
        env=git_environment,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            real_git,
            "-c",
            "user.name=Axiom test fixture",
            "-c",
            "user.email=fixture@axiom-foundation.org",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(runtime_root),
            "commit",
            "--quiet",
            "--no-gpg-sign",
            "-m",
            "Build trusted real CLI fixture",
        ],
        check=True,
        env=git_environment,
        capture_output=True,
        text=True,
    )
    return interpreter, runtime_root, package_root


def _runtime_arguments(runtime: tuple[Path, Path, Path]) -> list[str]:
    _interpreter, runtime_root, package_root = runtime
    import_roots = [package_root.parent]
    site_packages = (
        runtime_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if site_packages.is_dir() and site_packages not in import_roots:
        import_roots.append(site_packages)
    arguments = [
        "--trusted-python-runtime-root",
        str(runtime_root),
    ]
    for import_root in import_roots:
        arguments.extend(("--trusted-python-import-root", str(import_root)))
    arguments.extend(("--trusted-python-package-root", str(package_root)))
    return arguments


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
    corpus_release_public: str | None = None,
) -> Path:
    if corpus_release_public is None:
        corpus_release_public, _private_key = _keypair(b"\x17" * 32)
    path = tmp_path.resolve() / "signing-trust-roots.json"
    path.write_text(
        json.dumps(
            {
                "schema": "axiom-encode/signing-trust-roots/v2",
                "apply_ed25519_public_key": apply_public,
                "eval_ed25519_public_key": eval_public,
                "corpus_release_ed25519_public_key": corpus_release_public,
            },
            sort_keys=True,
        )
        + "\n"
    )
    path.chmod(0o600)
    return path


def _write_signed_corpus_release(
    corpus_root: Path,
    *,
    release_name: str,
    citation_path: str,
    version: str,
    body: str,
):
    jurisdiction, document_class, *_rest = citation_path.split("/")
    provision = (
        corpus_root
        / "data"
        / "corpus"
        / "provisions"
        / jurisdiction
        / document_class
        / f"{version}.jsonl"
    )
    provision.parent.mkdir(parents=True, exist_ok=True)
    provision.write_text(
        json.dumps(
            {
                "id": f"fixture:{citation_path}",
                "citation_path": citation_path,
                "body": body,
                "jurisdiction": jurisdiction,
                "document_class": document_class,
                "version": version,
                "source_path": "sources/supervisor-fixture.txt",
                "source_as_of": "2026-07-11",
                "expression_date": "2026-07-11",
            }
        )
        + "\n"
    )
    return bind_test_corpus_release(
        corpus_root,
        release_name,
        [(jurisdiction, document_class, version)],
    )


def _write_rulespec_toolchain(rulespec_root: Path, release) -> str:
    waiver = rulespec_root / "known-validation-gaps.yaml"
    waiver.parent.mkdir(parents=True, exist_ok=True)
    waiver.write_text("validate_failures: {}\n")
    waiver_sha256 = hashlib.sha256(waiver.read_bytes()).hexdigest()
    toolchain = rulespec_root / ".axiom" / "toolchain.toml"
    toolchain.parent.mkdir()
    toolchain.write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{release.name}"\n'
        f'axiom_corpus_release_content_sha256 = "{release.content_sha256}"\n'
        f'validation_waiver_set_sha256 = "{waiver_sha256}"\n'
    )
    return waiver_sha256


def _runtime_commit(runtime_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(runtime_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_signed_guard_fixture(
    tmp_path: Path,
    runtime_root: Path,
    apply_public: str,
) -> tuple[Path, Path]:
    rulespec_root = tmp_path / "rulespec-us"
    corpus_root = tmp_path / "axiom-corpus"
    citation_path = "us/statute/1"
    body = "The signed supervisor source is authoritative.\n"
    release = _write_signed_corpus_release(
        corpus_root,
        release_name="supervisor-guard-release",
        citation_path=citation_path,
        version="2026-supervisor-guard",
        body=body,
    )
    waiver_sha256 = _write_rulespec_toolchain(rulespec_root, release)
    source_unit = resolve_corpus_source_unit(citation_path, release)
    source_attestation = dict(source_unit.source_attestation)
    source_attestation["generation_input_sha256"] = hashlib.sha256(
        body.encode()
    ).hexdigest()
    source_attestation["rulespec_root"] = "rulespec-us/us"

    rule = rulespec_root / "us" / "statutes" / "1.yaml"
    rule.parent.mkdir(parents=True)
    rule.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        f"    corpus_citation_path: {citation_path}\n"
        f"    source_sha256: {source_attestation['source_sha256']}\n"
        "rules: []\n"
    )
    commit = _runtime_commit(runtime_root)
    payload = {
        "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
        "generated_at": "2026-07-11T00:00:00+00:00",
        "tool": APPLIED_ENCODING_MODEL_TOOL,
        "axiom_encode_version": __version__,
        "axiom_encode_git": {
            "root": str(runtime_root),
            "commit": commit,
            "dirty_tracked": False,
            "version": __version__,
            "version_commit": commit,
        },
        "generation_prompt_sha256": None,
        "run_id": None,
        "citation": "1 USC 1",
        "runner": "codex:fixture",
        "backend": "codex",
        "model": "fixture",
        "validation_waiver_set_sha256": waiver_sha256,
        "generated_output_root": str(tmp_path / "generated"),
        "generated_output_file": None,
        "generated_output_sha256": None,
        "trace_file": None,
        "trace_sha256": None,
        "context_manifest_file": None,
        "context_manifest_sha256": None,
        "applied_files": [
            {
                "path": "us/statutes/1.yaml",
                "sha256": hashlib.sha256(rule.read_bytes()).hexdigest(),
            }
        ],
        "source_attestation": source_attestation,
        "validation_execution": {
            "schema": "axiom-encode/apply-validation-execution/v1",
            "axiom_encode": {
                "repository": "github.com/TheAxiomFoundation/axiom-encode",
                "commit": commit,
                "version": __version__,
            },
            "axiom_rules_engine": {
                "repository": ("github.com/TheAxiomFoundation/axiom-rules-engine"),
                "commit": "e" * 40,
            },
            "policy_pre_apply": {
                "rulespec_root": "rulespec-us/us",
                "pre_apply_content_sha256": "c" * 64,
                "pre_apply_file_count": 0,
                "toolchain_contract_sha256": "d" * 64,
                "validation_waiver_set_sha256": waiver_sha256,
            },
            "rulespec_dependencies": [],
        },
    }
    apply_private = b64encode(b"\xab" * 32).decode("ascii")
    _sign_applied_encoding_manifest(
        payload,
        SigningBrokerFixture(
            apply_private_key=apply_private,
            apply_public_key=apply_public,
        ),
    )
    manifest = (
        rulespec_root / ".axiom" / "encoding-manifests" / "us" / "statutes" / "1.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return rulespec_root, corpus_root


def _current_engine_root() -> Path | None:
    configured = os.environ.get("AXIOM_RULES_ENGINE_ROOT")
    candidates = [
        *(Path(configured).expanduser() for _ in (0,) if configured),
        ROOT.parent / "axiom-rules-engine-canonical-loader-hard-cut",
        ROOT.parents[1] / "axiom-rules-engine",
    ]
    for root in candidates:
        if not root.is_dir():
            continue
        for binary in (
            root / "target" / "debug" / "axiom-rules-engine",
            root / "target" / "release" / "axiom-rules-engine",
            root / "axiom-rules-engine",
        ):
            if not binary.is_file() or not os.access(binary, os.X_OK):
                continue
            probe = subprocess.run(
                [str(binary), "compile", "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if probe.returncode == 0 and "--rulespec-root" in (
                probe.stdout + probe.stderr
            ):
                return root.resolve()
    return None


def _write_current_engine_fixture(tmp_path: Path) -> tuple[Path, Path]:
    rulespec_root = tmp_path / "rulespec-us"
    corpus_root = tmp_path / "axiom-corpus"
    citation_path = "us/guidance/example/sua"
    release = _write_signed_corpus_release(
        corpus_root,
        release_name="supervisor-current-engine-release",
        citation_path=citation_path,
        version="2026-supervisor-engine",
        body="The standard utility allowance for a household is $451.\n",
    )
    _write_rulespec_toolchain(rulespec_root, release)
    rules = rulespec_root / "us" / "policies" / "example" / "rules.yaml"
    rules.parent.mkdir(parents=True)
    rules.write_text(
        """format: rulespec/v1
module:
  summary: The standard utility allowance is $451.
  source_verification:
    corpus_citation_path: us/guidance/example/sua
rules:
  - name: standard_utility_allowance_value
    kind: parameter
    source: us/guidance/example/sua
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '451'
  - name: standard_utility_allowance
    kind: derived
    source: us/guidance/example/sua
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: standard_utility_allowance_value
"""
    )
    rules.with_name("rules.test.yaml").write_text(
        """- name: signed corpus and current engine
  period: 2026-01
  input: {}
  output:
    us:policies/example/rules#standard_utility_allowance: 451
"""
    )
    return rules, corpus_root


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
    command_args: tuple[str, ...] = (),
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
            *command_args,
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


def test_every_invocation_requires_protected_three_root_config(
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
        "OPENAI_API_KEY": "openai-sentinel",
        "ANTHROPIC_API_KEY": "anthropic-sentinel",
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
        *PUBLIC_ENV_NAMES,
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
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
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


def test_python_startup_rejects_corpus_public_root_environment() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "src/axiom_encode/_trusted_signing_bootstrap.py")],
        env={"AXIOM_CORPUS_RELEASE_PUBLIC_KEY": "counterfeit"},
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "authenticated broker" in completed.stderr
    assert "AXIOM_CORPUS_RELEASE_PUBLIC_KEY" in completed.stderr


def test_verification_only_invocation_exposes_roots_without_signing_capability(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    apply_public, _apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    corpus_release_public, _corpus_release_key = _keypair(b"\x17" * 32)
    completed = _invoke(
        signing_supervisor,
        trusted_python_runtime,
        _launcher(tmp_path, trusted_python_runtime),
        _trust_config(
            tmp_path,
            apply_public,
            eval_public,
            corpus_release_public,
        ),
        [],
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["capabilities"] == []
    assert result["roots"] == {
        "apply": apply_public,
        "eval": eval_public,
        "corpus_release": corpus_release_public,
    }


def test_verification_only_supervisor_runs_public_guard_generated_with_signed_release(
    signing_supervisor: Path,
    trusted_real_cli_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    apply_public, _apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    corpus_release_public, _corpus_release_key = _keypair(b"\x17" * 32)
    rulespec_root, corpus_root = _write_signed_guard_fixture(
        tmp_path,
        trusted_real_cli_runtime[1],
        apply_public,
    )
    completed = _invoke(
        signing_supervisor,
        trusted_real_cli_runtime,
        _launcher(tmp_path, trusted_real_cli_runtime),
        _trust_config(
            tmp_path,
            apply_public,
            eval_public,
            corpus_release_public,
        ),
        [],
        command_args=(
            "guard-generated",
            "--repo",
            str(rulespec_root),
            "--corpus-path",
            str(corpus_root),
            "--all",
            "--json",
        ),
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "repo": str(rulespec_root.resolve()),
        "passed": True,
        "issues": [],
    }


def test_supervised_validate_uses_signed_release_and_current_engine_end_to_end(
    signing_supervisor: Path,
    trusted_real_cli_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    engine_root = _current_engine_root()
    if engine_root is None:
        pytest.skip("current axiom-rules-engine binary is not available")
    rules, corpus_root = _write_current_engine_fixture(tmp_path)
    apply_public, _apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    corpus_release_public, _corpus_release_key = _keypair(b"\x17" * 32)
    completed = _invoke(
        signing_supervisor,
        trusted_real_cli_runtime,
        _launcher(tmp_path, trusted_real_cli_runtime),
        _trust_config(
            tmp_path,
            apply_public,
            eval_public,
            corpus_release_public,
        ),
        [],
        command_args=(
            "validate",
            str(rules),
            "--corpus-path",
            str(corpus_root),
            "--axiom-rules-engine-path",
            str(engine_root),
            "--skip-reviewers",
        ),
    )

    assert completed.returncode == 0, completed.stderr + completed.stdout
    assert "CI: ✓" in completed.stdout
    assert "Result: ✓ PASSED" in completed.stdout


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


@pytest.mark.parametrize("aliased", ["apply_eval", "apply_corpus", "eval_corpus"])
def test_aliased_roots_fail_even_for_verification_only_invocation(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
    aliased: str,
) -> None:
    apply_public, _apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    corpus_release_public, _corpus_release_key = _keypair(b"\x17" * 32)
    if aliased == "apply_eval":
        eval_public = apply_public
    elif aliased == "apply_corpus":
        corpus_release_public = apply_public
    else:
        corpus_release_public = eval_public
    launcher = _launcher(tmp_path, trusted_python_runtime)
    trust_config = _trust_config(
        tmp_path,
        apply_public,
        eval_public,
        corpus_release_public,
    )
    completed = _invoke(
        signing_supervisor,
        trusted_python_runtime,
        launcher,
        trust_config,
        [],
    )
    assert completed.returncode == 2
    assert "must be distinct" in completed.stderr


@pytest.mark.parametrize(
    "mutation",
    ["old_schema", "missing_eval", "missing_corpus", "extra_field"],
)
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
        payload["schema"] = "axiom-encode/signing-trust-roots/v1"
    elif mutation == "missing_eval":
        payload.pop("eval_ed25519_public_key")
    elif mutation == "missing_corpus":
        payload.pop("corpus_release_ed25519_public_key")
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
