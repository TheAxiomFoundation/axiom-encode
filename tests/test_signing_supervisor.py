"""Integration and adversarial tests for the compiled signing supervisor."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import struct
import subprocess
import sys
import tarfile
import threading
import uuid
from base64 import b64decode, b64encode
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from unittest.mock import patch

import _cffi_backend
import cryptography
import pytest
import yaml
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from axiom_encode import __version__
from axiom_encode.cli import (
    APPLIED_ENCODING_MANIFEST_SCHEMA,
    APPLIED_ENCODING_MODEL_TOOL,
    _sign_applied_encoding_manifest,
)
from axiom_encode.harness.dependency_stubs import validate_explicit_context_file
from axiom_encode.harness.evals import resolve_corpus_source_unit
from axiom_encode.repo_routing import inspect_canonical_rulespec_checkout
from scripts import materialize_corpus_release as release_acquisition
from scripts import prepare_signed_backfill as compatibility_backfill
from scripts import provision_verification_supervisor as provisioner
from scripts.prepare_signed_backfill import parse_canonical_refresh_bundle
from tests.eval_evidence_fixtures import (
    TEST_APPLY_PRIVATE_KEY_B64,
    TEST_APPLY_PUBLIC_KEY_B64,
    TEST_EVAL_PUBLIC_KEY_B64,
    install_test_eval_evidence_keys,
)
from tests.release_object_fixtures import (
    TEST_RELEASE_PUBLIC_KEY,
    bind_test_corpus_release,
)
from tests.signing_broker_fixtures import SigningBrokerFixture
from tests.test_cli import TestCmdEncode as _TestCmdEncode

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
MULTILANE_REPLACE_ERROR = (
    "nonlegacy explicit target_operation=replace cannot use source_bundle or a "
    "nonempty canonical_refresh_bundle until signed replacement_target evidence "
    "and expected-parent lineage are enforced"
)


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
    if os.environ.get("CODEX_HOME"):
        from pathlib import Path
        codex_home = Path(os.environ["CODEX_HOME"])
        codex_auth = codex_home / "auth.json"
        codex_auth_before_refresh = codex_auth.read_text()
        codex_auth.write_text(
            '{"token":"new"}\\n'
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
            \"corpus_release_keys\": [
                b64encode(public_key).decode(\"ascii\")
                for public_key in broker.corpus_release_public_keys_raw
            ],
        },
    }
    if os.environ.get("CODEX_HOME"):
        metadata = codex_home.stat()
        result["codex_home_mode"] = metadata.st_mode & 0o777
        result["codex_home_uid"] = metadata.st_uid
        result["codex_auth_read_path"] = str(codex_auth)
        result["codex_auth_before_refresh"] = codex_auth_before_refresh
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
    head = subprocess.run(
        [real_git, "-C", str(runtime_root), "rev-parse", "HEAD"],
        check=True,
        env=git_environment,
        capture_output=True,
        text=True,
    ).stdout.strip()
    # Model the root-provisioned runtime end to end: the supervisor sets
    # AXIOM_ENCODE_TRUSTED_RUNTIME=1, so _apply_encoder_execution_identity takes
    # the attestation branch and byte-binds the running package to
    # package_tree_sha256. Record the digest of the runtime's own package and a
    # commit that agrees with the live checkout (the agreement path).
    from axiom_encode.harness.evals import _deterministic_tree_identity

    package_tree_sha256 = _deterministic_tree_identity(
        package_root, excluded_directory_names=frozenset({"__pycache__"})
    )["tree_sha256"]
    (runtime_root / "runtime-attestation.json").write_text(
        json.dumps(
            {
                "schema": "axiom-encode/trusted-runtime-attestation/v1",
                "provisioned_at": "2026-07-18T00:00:00+00:00",
                "axiom_encode": {
                    "origin_repository": "github.com/TheAxiomFoundation/axiom-encode",
                    "commit": head,
                    "version": __version__,
                    "package_tree_sha256": package_tree_sha256,
                },
            }
        )
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


def _launcher(
    tmp_path: Path, runtime: tuple[Path, Path, Path], body: str | None = None
) -> Path:
    interpreter, _runtime_root, _package_root = runtime
    launcher = tmp_path.resolve() / "axiom-encode"
    launcher.write_text(
        f"#!{interpreter} -I\n"
        + (body if body is not None else "raise SystemExit('launcher executed')\n")
    )
    launcher.chmod(0o700)
    return launcher


def _trust_config(
    tmp_path: Path,
    apply_public: str,
    eval_public: str,
    corpus_release_public: str | None = None,
    corpus_release_public_keys: tuple[str, ...] | None = None,
) -> Path:
    if corpus_release_public is None:
        corpus_release_public, _private_key = _keypair(b"\x17" * 32)
    path = tmp_path.resolve() / "signing-trust-roots.json"
    payload = {
        "schema": "axiom-encode/signing-trust-roots/v2",
        "apply_ed25519_public_key": apply_public,
        "eval_ed25519_public_key": eval_public,
        "corpus_release_ed25519_public_key": corpus_release_public,
    }
    if corpus_release_public_keys is not None:
        payload["schema"] = "axiom-encode/signing-trust-roots/v3"
        payload.pop("corpus_release_ed25519_public_key")
        payload["corpus_release_ed25519_public_keys"] = corpus_release_public_keys
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
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
            "identity_source": "git",
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
                "identity_source": "git",
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
    supervisor_args: tuple[str, ...] = (),
    timeout: int = 30,
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
            *supervisor_args,
            *_runtime_arguments(runtime),
            "--",
            str(launcher),
            *command_args,
        ],
        env={} if environment is None else environment,
        pass_fds=tuple(descriptors),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_subscription_auth_is_isolated_refreshed_and_wiped(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    apply_public, _ = _keypair(b"\xab" * 32)
    eval_public, _ = _keypair(b"\xcd" * 32)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    trusted = tmp_path / "trusted"
    (trusted / "bin").mkdir(parents=True)
    legacy_scratch = trusted / "runtime-codex-homes"
    legacy_scratch.mkdir(mode=0o700)
    legacy_scratch.chmod(0o000)
    codex = trusted / "bin/codex"
    codex.write_text("#!/bin/sh\nexit 0\n")
    codex.chmod(0o755)
    digest = hashlib.sha256(codex.read_bytes()).hexdigest()
    config = trusted / "codex-cli.json"
    config.write_text(
        json.dumps(
            {
                "schema": "axiom-encode/trusted-codex-cli/v1",
                "version": "test",
                "sha256": digest,
                "path": str(codex),
            }
        )
        + "\n"
    )
    config.chmod(0o444)
    auth = tmp_path / "operator-auth.json"
    auth.write_text('{"token":"old"}\n')
    auth.chmod(0o600)
    outbox = tmp_path / "refreshed-auth.json"
    operator_home_auth = tmp_path / "operator-home/.codex/auth.json"
    operator_home_auth.parent.mkdir(parents=True)
    operator_home_auth.write_text('{"must":"not-cross"}\n')
    launcher = _launcher(tmp_path, trusted_python_runtime)
    completed = _invoke(
        signing_supervisor,
        trusted_python_runtime,
        launcher,
        trust_config,
        [],
        environment={"HOME": str(tmp_path / "operator-home")},
        supervisor_args=(
            "--codex-subscription-auth",
            str(auth),
            "--codex-auth-outbox",
            str(outbox),
            "--trusted-codex-cli-config",
            str(config),
        ),
    )
    legacy_scratch.chmod(0o700)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    codex_home = Path(result["environment"]["CODEX_HOME"])
    assert codex_home.name.startswith("axiom-codex-")
    assert not codex_home.exists()
    assert result["codex_home_mode"] == 0o700
    assert result["codex_home_uid"] == os.geteuid()
    assert Path(result["codex_auth_read_path"]) == codex_home / "auth.json"
    assert Path(result["codex_auth_read_path"]) != operator_home_auth
    assert json.loads(result["codex_auth_before_refresh"]) == {"token": "old"}
    assert result["environment"]["AXIOM_ENCODE_TRUSTED_CODEX_BIN"] == str(codex)
    assert result["environment"]["AXIOM_ENCODE_TRUSTED_CODEX_SHA256"] == digest
    assert result["environment"]["AXIOM_ENCODE_TRUSTED_CODEX_VERSION"] == "test"
    assert str(codex.parent) not in result["environment"]["PATH"].split(os.pathsep)
    assert result["environment"]["HOME"] != str(tmp_path / "operator-home")
    assert result["child"]["descriptor"] == "closed"
    assert json.loads(operator_home_auth.read_text()) == {"must": "not-cross"}
    assert json.loads(outbox.read_text()) == {"token": "new"}


def test_subscription_tampered_binary_hard_fails_before_child(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    apply_public, _ = _keypair(b"\xab" * 32)
    eval_public, _ = _keypair(b"\xcd" * 32)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    codex = tmp_path / "codex"
    codex.write_text("tampered")
    codex.chmod(0o755)
    config = tmp_path / "codex-cli.json"
    config.write_text(
        json.dumps(
            {
                "schema": "axiom-encode/trusted-codex-cli/v1",
                "version": "test",
                "sha256": "0" * 64,
                "path": str(codex),
            }
        )
    )
    config.chmod(0o444)
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    launcher = _launcher(
        tmp_path, trusted_python_runtime, body="raise RuntimeError('must not execute')"
    )
    completed = _invoke(
        signing_supervisor,
        trusted_python_runtime,
        launcher,
        trust_config,
        [],
        supervisor_args=(
            "--codex-subscription-auth",
            str(auth),
            "--codex-auth-outbox",
            str(tmp_path / "out.json"),
            "--trusted-codex-cli-config",
            str(config),
        ),
    )
    assert completed.returncode == 2
    assert "sha256 mismatch" in completed.stderr


@pytest.mark.parametrize(
    "outbox_kind",
    ["symlink", "fifo", "socket", "device", "writable-directory"],
)
def test_subscription_refuses_unsafe_auth_outbox(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
    outbox_kind: str,
) -> None:
    apply_public, _ = _keypair(b"\xab" * 32)
    eval_public, _ = _keypair(b"\xcd" * 32)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 0\n")
    codex.chmod(0o755)
    config = tmp_path / "codex-cli.json"
    config.write_text(
        json.dumps(
            {
                "schema": "axiom-encode/trusted-codex-cli/v1",
                "version": "test",
                "sha256": hashlib.sha256(codex.read_bytes()).hexdigest(),
                "path": str(codex),
            }
        )
    )
    config.chmod(0o444)
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    target = tmp_path / "target.json"
    target.write_text('{"preserve":true}\n')
    outbox_socket = None
    if outbox_kind == "symlink":
        outbox = tmp_path / "out.json"
        outbox.symlink_to(target)
    elif outbox_kind == "fifo":
        outbox = tmp_path / "out.json"
        os.mkfifo(outbox)
    elif outbox_kind == "socket":
        socket_directory = Path.cwd() / f".outbox-{os.getpid()}"
        socket_directory.mkdir(mode=0o700)
        outbox = socket_directory / "o"
        outbox_socket = socket.socket(socket.AF_UNIX)
        try:
            outbox_socket.bind(str(outbox))
        except PermissionError:
            outbox_socket.close()
            socket_directory.rmdir()
            pytest.skip("sandbox does not permit creating Unix-domain sockets")
    elif outbox_kind == "device":
        outbox = Path("/dev/null")
    else:
        unsafe_directory = tmp_path / "unsafe-outbox"
        unsafe_directory.mkdir(mode=0o777)
        unsafe_directory.chmod(0o777)
        outbox = unsafe_directory / "out.json"
    try:
        completed = _invoke(
            signing_supervisor,
            trusted_python_runtime,
            _launcher(tmp_path, trusted_python_runtime),
            trust_config,
            [],
            supervisor_args=(
                "--codex-subscription-auth",
                str(auth),
                "--codex-auth-outbox",
                str(outbox),
                "--trusted-codex-cli-config",
                str(config),
            ),
        )
    finally:
        if outbox_socket is not None:
            outbox_socket.close()
            shutil.rmtree(socket_directory)
    assert completed.returncode == 2
    assert "outbox" in completed.stderr
    assert json.loads(target.read_text()) == {"preserve": True}


def test_subscription_refuses_ambient_codex_home_outside_scratch(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    apply_public, _ = _keypair(b"\xab" * 32)
    eval_public, _ = _keypair(b"\xcd" * 32)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    completed = _invoke(
        signing_supervisor,
        trusted_python_runtime,
        _launcher(tmp_path, trusted_python_runtime),
        trust_config,
        [],
        environment={"CODEX_HOME": str(tmp_path / "outside")},
        supervisor_args=(
            "--codex-subscription-auth",
            str(tmp_path / "auth.json"),
            "--codex-auth-outbox",
            str(tmp_path / "out.json"),
            "--trusted-codex-cli-config",
            str(tmp_path / "codex-cli.json"),
        ),
    )
    assert completed.returncode == 2
    assert "ambient CODEX_HOME is outside supervisor custody" in completed.stderr


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
        "corpus_release_keys": [corpus_release_public],
    }


def test_v3_trust_config_exposes_ordered_corpus_release_keyring(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    apply_public, _apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    current_public, _current_key = _keypair(b"\x18" * 32)
    retired_public, _retired_key = _keypair(b"\x17" * 32)
    completed = _invoke(
        signing_supervisor,
        trusted_python_runtime,
        _launcher(tmp_path, trusted_python_runtime),
        _trust_config(
            tmp_path,
            apply_public,
            eval_public,
            corpus_release_public_keys=(current_public, retired_public),
        ),
        [],
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["roots"] == {
        "apply": apply_public,
        "eval": eval_public,
        "corpus_release": current_public,
        "corpus_release_keys": [current_public, retired_public],
    }


@pytest.mark.parametrize("mutation", ["empty", "malformed", "wrong_length", "conflict"])
def test_v3_trust_config_rejects_invalid_corpus_release_keyrings(
    signing_supervisor: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
    mutation: str,
) -> None:
    apply_public, _apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    current_public, _current_key = _keypair(b"\x18" * 32)
    other_public, _other_key = _keypair(b"\x19" * 32)
    trust_config = _trust_config(
        tmp_path,
        apply_public,
        eval_public,
        corpus_release_public_keys=(current_public,),
    )
    payload = json.loads(trust_config.read_text())
    if mutation == "empty":
        payload["corpus_release_ed25519_public_keys"] = []
    elif mutation == "malformed":
        payload["corpus_release_ed25519_public_keys"] = ["not-base64!!"]
    elif mutation == "wrong_length":
        payload["corpus_release_ed25519_public_keys"] = [b64encode(b"short").decode()]
    else:
        payload["corpus_release_ed25519_public_key"] = other_public
    trust_config.write_text(json.dumps(payload) + "\n")

    completed = _invoke(
        signing_supervisor,
        trusted_python_runtime,
        _launcher(tmp_path, trusted_python_runtime),
        trust_config,
        [],
    )

    assert completed.returncode == 2
    assert "corpus release public key" in completed.stderr


def test_verification_only_supervisor_accepts_retired_release_key_from_v3_keyring(
    signing_supervisor: Path,
    trusted_real_cli_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    apply_public, _apply_key = _keypair(b"\xab" * 32)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    current_corpus_release_public, _current_corpus_release_key = _keypair(b"\x18" * 32)
    retired_corpus_release_public, _retired_corpus_release_key = _keypair(b"\x17" * 32)
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
            corpus_release_public_keys=(
                current_corpus_release_public,
                retired_corpus_release_public,
            ),
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


def test_protected_supervisor_stages_authenticated_v7_exact_dependent_transaction(
    signing_supervisor: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise the installed staging command through every protected seam."""

    install_test_eval_evidence_keys(
        monkeypatch,
        apply_private_key=TEST_APPLY_PRIVATE_KEY_B64,
        apply_public_key=TEST_APPLY_PUBLIC_KEY_B64,
    )
    test_case = _TestCmdEncode()
    preflights = test_case._neutralize_encode_preflights.__wrapped__(
        test_case,
        tmp_path,
    )
    next(preflights)
    captured: dict[str, object] = {}

    class TransactionCaptured(Exception):
        pass

    real_authorized = compatibility_backfill.authorized_changed_paths

    def capture_transaction(repo: Path, *, corpus_root: Path) -> tuple[Path, ...]:
        captured["repo"] = Path(repo)
        captured["corpus"] = Path(corpus_root)
        captured["paths"] = tuple(real_authorized(repo, corpus_root=corpus_root))
        raise TransactionCaptured

    try:
        with (
            patch(
                "axiom_encode.cli._manifest_census",
                return_value={"unmanifested_paths": []},
            ),
            patch("axiom_encode.cli.guard_generated_change_issues", return_value=[]),
            patch("tests.test_cli.guard_generated_change_issues", return_value=[]),
            patch.object(
                compatibility_backfill,
                "authorized_changed_paths",
                side_effect=capture_transaction,
            ),
            pytest.raises(TransactionCaptured),
        ):
            test_case.test_apply_atomically_migrates_exact_legacy_dependent(
                tmp_path,
                "manual",
            )
    finally:
        with pytest.raises(StopIteration):
            next(preflights)

    rulespec_root = captured["repo"]
    corpus_root = captured["corpus"]
    expected_paths = captured["paths"]
    assert isinstance(rulespec_root, Path)
    assert isinstance(corpus_root, Path)
    assert isinstance(expected_paths, tuple)
    assert len(expected_paths) == 34
    assert {
        PurePosixPath(".axiom/retired-schema-freeze.json"),
        PurePosixPath("tests/test_legacy_rulespec_freeze.py"),
    } <= set(expected_paths)

    runtime = trusted_real_cli_runtime.__wrapped__(tmp_path_factory)
    interpreter, runtime_root, _package_root = runtime
    runtime_git = interpreter.parent / "git"
    runtime_git.unlink()
    production_git = Path("/usr/bin/git")
    if not production_git.is_file():
        pytest.skip("Protected staging requires production Git at /usr/bin/git")
    production_git = provisioner._resolve_trusted_git(production_git)
    provisioner._install_trusted_git_wrapper(
        interpreter.parent,
        interpreter,
        production_git,
    )
    warmed_git = subprocess.run(
        [
            runtime_git,
            "-C",
            str(rulespec_root),
            "rev-parse",
            "--show-toplevel",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env={"HOME": str(runtime_root), "PATH": str(interpreter.parent)},
    )
    assert Path(warmed_git.stdout.strip()).resolve() == rulespec_root.resolve()

    completed = _invoke(
        signing_supervisor,
        runtime,
        _launcher(tmp_path, runtime),
        _trust_config(
            tmp_path,
            TEST_APPLY_PUBLIC_KEY_B64,
            TEST_EVAL_PUBLIC_KEY_B64,
            TEST_RELEASE_PUBLIC_KEY,
        ),
        [],
        command_args=(
            "stage-signed-backfill",
            "--repo",
            str(rulespec_root),
            "--corpus-path",
            str(corpus_root),
        ),
        timeout=90,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    staged_raw = subprocess.run(
        [
            production_git,
            "-C",
            str(rulespec_root),
            "diff",
            "--cached",
            "--name-only",
            "--no-renames",
            "-z",
        ],
        check=True,
        capture_output=True,
    ).stdout
    staged_paths = {item.decode() for item in staged_raw.split(b"\0") if item}
    assert staged_paths == {path.as_posix() for path in expected_paths}
    for relative in expected_paths:
        live = rulespec_root / relative
        indexed = subprocess.run(
            [
                production_git,
                "-C",
                str(rulespec_root),
                "show",
                f":{relative.as_posix()}",
            ],
            check=False,
            capture_output=True,
        )
        if live.exists():
            assert indexed.returncode == 0, indexed.stderr.decode()
            assert indexed.stdout == live.read_bytes()
        else:
            assert indexed.returncode != 0


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


# --- Production external apply-signer binary (cmd/axiom-encode-apply-signer) ---
#
# The tests above use an in-process Python signer to drive the broker. These
# exercise the real, separately compiled production external signer as the
# broker's protocol v2 socket peer, proving wire compatibility with the actual
# supervisor rather than a re-implementation.

_APPLY_SIGNER_PACKAGE = "./cmd/axiom-encode-apply-signer"
_SIGNER_REPOSITORY = "TheAxiomFoundation/rulespec-uk"
_SIGNER_WORKFLOW_REF = (
    "TheAxiomFoundation/rulespec-uk/.github/workflows/bulk-encode.yml@refs/heads/main"
)


@pytest.fixture(scope="session")
def apply_signer_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    go = shutil.which("go")
    if go is None:
        pytest.skip("Go is required to build the apply signer")
    build_dir = tmp_path_factory.mktemp("apply-signer-build").resolve()
    binary = build_dir / "axiom-encode-apply-signer"
    subprocess.run(
        [
            go,
            "build",
            "-trimpath",
            "-buildvcs=false",
            "-ldflags=-buildid=",
            "-o",
            str(binary),
            _APPLY_SIGNER_PACKAGE,
        ],
        cwd=ROOT,
        env={**os.environ, "CGO_ENABLED": "0"},
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return binary


@contextmanager
def _external_apply_signer(binary: Path, seed: bytes, audit_path: Path):
    """Run the production signer binary as the broker's protocol v2 peer.

    The signer's audit stream is redirected to a file, not a pipe: the signer
    stays alive until the socket tears down in this manager's finally, so reading
    a pipe inline would deadlock.
    """

    signer_connection, supervisor_connection = socket.socketpair()
    key_read, key_write = os.pipe()
    os.write(key_write, b64encode(seed))
    os.close(key_write)
    with open(audit_path, "w") as audit_file:
        process = subprocess.Popen(
            [
                str(binary),
                "serve",
                "--scope",
                "apply_ed25519",
                "--socket-fd",
                str(signer_connection.fileno()),
                "--key-fd",
                str(key_read),
                "--expected-github-repository",
                _SIGNER_REPOSITORY,
                "--allowed-workflow-ref",
                _SIGNER_WORKFLOW_REF,
                "--allowed-event-name",
                "workflow_dispatch",
            ],
            pass_fds=(signer_connection.fileno(), key_read),
            env={
                "GITHUB_ACTIONS": "true",
                "GITHUB_REPOSITORY": _SIGNER_REPOSITORY,
                "GITHUB_WORKFLOW_REF": _SIGNER_WORKFLOW_REF,
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "GITHUB_SHA": "0" * 40,
                "GITHUB_RUN_ID": "1",
                "PATH": os.environ.get("PATH", ""),
            },
            stdout=audit_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    signer_connection.close()
    os.close(key_read)
    try:
        yield [supervisor_connection.fileno()]
    finally:
        supervisor_connection.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def test_production_apply_signer_binary_signs_through_real_broker(
    signing_supervisor: Path,
    apply_signer_binary: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    seed = b"\x2a" * 32
    apply_public, apply_key = _keypair(seed)
    eval_public, _eval_key = _keypair(b"\xcd" * 32)
    launcher = _launcher(tmp_path, trusted_python_runtime)
    trust_config = _trust_config(tmp_path, apply_public, eval_public)
    audit_path = tmp_path / "signer-audit.log"
    with _external_apply_signer(apply_signer_binary, seed, audit_path) as descriptors:
        completed = _invoke(
            signing_supervisor,
            trusted_python_runtime,
            launcher,
            trust_config,
            descriptors,
        )
    signer_output = audit_path.read_text()
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["capabilities"] == ["apply_ed25519"]
    # The apply signature the real broker obtained from the production binary
    # verifies with the public half over the exact apply domain.
    apply_key.public_key().verify(
        b64decode(result["apply_signature"]),
        SIGNATURE_DOMAIN + b"apply_ed25519\0compiled-apply-boundary",
    )
    # It is scope-bound: the apply signature must not verify as an eval one.
    with pytest.raises(Exception):
        apply_key.public_key().verify(
            b64decode(result["apply_signature"]),
            b"axiom-encode/external-signer-sign/v2\0eval_ed25519\0compiled-apply-boundary",
        )
    # The signer's audit stream records the signing event and never the key.
    assert "event=sign" in signer_output
    assert b64encode(seed).decode() not in signer_output


def test_production_apply_signer_binary_rejects_wrong_ci_context(
    apply_signer_binary: Path,
) -> None:
    seed = b"\x2a" * 32
    signer_connection, supervisor_connection = socket.socketpair()
    key_read, key_write = os.pipe()
    os.write(key_write, b64encode(seed))
    os.close(key_write)
    try:
        completed = subprocess.run(
            [
                str(apply_signer_binary),
                "serve",
                "--scope",
                "apply_ed25519",
                "--socket-fd",
                str(signer_connection.fileno()),
                "--key-fd",
                str(key_read),
                "--expected-github-repository",
                _SIGNER_REPOSITORY,
                "--allowed-workflow-ref",
                _SIGNER_WORKFLOW_REF,
                "--allowed-event-name",
                "workflow_dispatch",
            ],
            pass_fds=(signer_connection.fileno(), key_read),
            env={
                "GITHUB_ACTIONS": "true",
                # Fork-controlled repository: must be refused before the key is read.
                "GITHUB_REPOSITORY": "attacker/rulespec-uk",
                "GITHUB_WORKFLOW_REF": _SIGNER_WORKFLOW_REF,
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "PATH": os.environ.get("PATH", ""),
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        signer_connection.close()
        supervisor_connection.close()
        os.close(key_read)
    assert completed.returncode == 2
    assert "does not match the expected repository" in completed.stderr


def test_targeted_signed_reencode_shell_steps_have_valid_syntax(tmp_path: Path) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )

    for job_name, job in workflow["jobs"].items():
        for index, step in enumerate(job.get("steps", [])):
            command = step.get("run")
            if command is None:
                continue
            script = tmp_path / f"{job_name}-{index}.bash"
            script.write_text(command, encoding="utf-8")
            subprocess.run(["bash", "-n", str(script)], check=True)


def test_signing_supervisor_ci_exercises_process_guardian() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/signing-supervisor.yml").read_text()
    )
    trigger = workflow.get("on", workflow.get(True))
    guardian_path = "cmd/axiom-encode-process-guardian/**"
    assert guardian_path in trigger["push"]["paths"]
    assert guardian_path in trigger["pull_request"]["paths"]

    job = workflow["jobs"]["build-and-test"]
    assert job["env"]["AXIOM_REQUIRE_PRIVILEGED_GUARDIAN_TESTS"] == "1"
    steps = job["steps"]
    by_name = {step.get("name"): step for step in steps if step.get("name")}
    assert "go test -v -count=1 ./cmd/axiom-encode-process-guardian" in (
        by_name["Test process-guardian package"]["run"]
    )
    assert "go vet ./cmd/axiom-encode-process-guardian" in (
        by_name["Test process-guardian package"]["run"]
    )
    assert "go test -race -v -count=1 ./cmd/axiom-encode-process-guardian" in (
        by_name["Race-test process-guardian package"]["run"]
    )
    guardian_build = by_name["Build the process guardian twice reproducibly"]["run"]
    assert guardian_build.count("./cmd/axiom-encode-process-guardian") == 2
    assert "cmp guardian.first guardian.second" in guardian_build
    assert 'test "$(./guardian.first --build-kind)" = production' in guardian_build


def test_targeted_signed_reencode_only_allows_audited_legacy_index_shrink() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    step = next(
        item
        for item in workflow["jobs"]["encode"]["steps"]
        if item.get("name") == "Encode, review, validate, and apply"
    )
    command = step["run"]

    assert "authorize-legacy-index-manifest-shrink" in command
    assert "args+=(--allow-shrink)" in command
    assert command.count("--allow-shrink") == 1
    assert '[ -n "$replacement_path" ] && \\\n' in command
    assert '[ -z "$legacy_source_path" ] && \\\n' in command


def test_targeted_signed_reencode_reconciles_retired_inventory_before_commits() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    step = next(
        item
        for item in workflow["jobs"]["encode"]["steps"]
        if item.get("name") == "Encode, review, validate, and apply"
    )
    command = step["run"]
    invocation = "reconcile-retired-manifest-inventory"

    assert command.count(invocation) == 2
    assert '[ "$source_bundle_enabled" = "false" ]' in command
    assert '[ -n "$REPLACE_RULESPEC_PATH" ]' in command
    assert '[ -z "$REPLACE_LEGACY_RULESPEC_PATH" ]' in command

    refresh_apply = command.index('"$refresh_citation" "$refresh_finding"')
    refresh_reconciliation = command.index(invocation, refresh_apply)
    refresh_checkpoint = command.index(
        "Refresh signed canonical module for ${refresh_citation}",
        refresh_reconciliation,
    )
    assert refresh_apply < refresh_reconciliation < refresh_checkpoint

    normal_gate = command.index('[ "$source_bundle_enabled" = "false" ]')
    normal_reconciliation = command.index(
        invocation,
        refresh_reconciliation + len(invocation),
    )
    assert normal_gate < normal_reconciliation
    assert normal_reconciliation < command.index(
        "Complete signed target transaction for ${CITATION}",
        normal_reconciliation,
    )


def test_targeted_signed_reencode_workflow_is_main_dispatch_only() -> None:
    workflow_text = (
        ROOT / ".github/workflows/targeted-signed-reencode.yml"
    ).read_text()
    assert "git config --system" not in workflow_text
    assert "target-preflight" not in workflow_text
    workflow = yaml.safe_load(workflow_text)
    trigger = workflow.get("on", workflow.get(True))
    assert set(trigger) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"actions": "read", "contents": "read"}
    assert workflow["concurrency"] == {
        "group": (
            "targeted-signed-reencode-${{ "
            "github.actor == 'github-actions[bot]' && inputs.queue_id && "
            "inputs.queue_item_generation_sha256 || github.run_id }}"
        ),
        "cancel-in-progress": False,
    }

    def concurrency_suffix(
        actor: str, queue_id: str, generation_sha: str, run_id: str
    ) -> str:
        return (
            generation_sha
            if actor == "github-actions[bot]" and queue_id and generation_sha
            else run_id
        )

    assert (
        concurrency_suffix("github-actions[bot]", "snap", "queue-generation", "run-1")
        == "queue-generation"
    )
    assert (
        concurrency_suffix("manual-user", "snap", "queue-generation", "run-2")
        == "run-2"
    )
    assert concurrency_suffix("manual-user", "", "queue-generation", "run-3") == "run-3"
    assert concurrency_suffix("github-actions[bot]", "snap", "", "run-4") == "run-4"
    inputs = trigger["workflow_dispatch"]["inputs"]
    assert len(inputs) == 25
    assert "allowlisted reviewed SHA" in inputs["rulespec_ref"]["description"]
    assert "artifact-only" in inputs["rulespec_ref"]["description"]
    assert inputs["country"] == {
        "description": "Canonical RuleSpec country checkout (for rulespec-<country>)",
        "required": True,
        "default": "us",
        "type": "string",
    }
    assert inputs["open_pr"]["type"] == "boolean"
    assert inputs["open_pr"]["default"] is False
    assert inputs["repair_run_id"] == {
        "description": (
            "Prior failed protected run whose final candidate is replayed as "
            "untrusted repair context"
        ),
        "required": False,
        "type": "string",
    }
    assert "repair_candidate_tests_only" not in inputs
    assert inputs["pr_base_branch"]["type"] == "string"
    assert inputs["pr_base_branch"]["default"] == "main"
    assert inputs["source_bundle_json"] == {
        "description": (
            "JSON citation array, canonical_refresh_bundle object, or "
            "atomic-source-transaction/v2 envelope; explicit targets require v2 "
            "target_operation=create|replace, and nonlegacy create/replace targets "
            "require source and refresh lanes to land separately"
        ),
        "required": False,
        "default": "[]",
        "type": "string",
    }
    assert "canonical_refresh_bundle_json" not in inputs
    assert inputs["existing_signed_imports_json"] == {
        "description": (
            "JSON array of tracked same-jurisdiction signed-v5 modules to reuse "
            "as direct imports"
        ),
        "required": False,
        "default": "[]",
        "type": "string",
    }
    assert inputs["replace_rulespec_path"]["required"] is False
    assert inputs["replace_legacy_rulespec_path"]["required"] is False
    assert inputs["dependent_citation"]["required"] is False
    assert inputs["dependent_review_finding"]["required"] is False
    assert inputs["second_dependent_citation"]["required"] is False
    assert inputs["second_dependent_review_finding"]["required"] is False
    assert inputs["queue_id"]["required"] is False
    assert inputs["queue_item_id"]["required"] is False
    assert inputs["queue_manifest_sha256"]["required"] is False
    assert inputs["queue_item_generation_sha256"]["required"] is False
    assert inputs["queue_dispatcher_run_id"]["required"] is False
    assert "[${{ inputs.queue_id || 'adhoc' }}:" in workflow["run-name"]

    job = workflow["jobs"]["encode"]
    assert job["name"] == "Queue protected signed RuleSpec re-encode"
    assert job["environment"] == "production-signing"
    assert "github.ref == 'refs/heads/main'" in job["if"]
    assert "github.actor == 'github-actions[bot]'" in job["if"]
    assert "github.run_attempt == 1" in job["if"]
    attempt_step = next(
        step
        for step in workflow["jobs"]["attempt_budget"]["steps"]
        if step.get("name") == "Enforce failed-attempt budget"
    )
    assert attempt_step["env"]["REPAIR_RUN_ID"] == "${{ inputs.repair_run_id }}"
    steps = job["steps"]
    country_step = next(
        step for step in steps if step.get("name") == "Validate country routing input"
    )
    assert "prepare_signed_backfill.py" in country_step["run"]
    assert 'validate-country "$COUNTRY"' in country_step["run"]
    assert "validate-queue-tracking" in country_step["run"]
    assert '"$QUEUE_ID" "$QUEUE_ITEM_ID"' in country_step["run"]
    assert '"$QUEUE_MANIFEST_SHA256"' in country_step["run"]
    assert '"$QUEUE_ITEM_GENERATION_SHA256"' in country_step["run"]
    assert "prepare_signed_queue.py" in country_step["run"]
    assert "validate-dispatch" in country_step["run"]
    assert "QUEUE_DISPATCHER_RUN_ID" in country_step["env"]
    assert (
        "queue target must be created by an authenticated dispatcher"
        in (country_step["run"])
    )
    assert 'GITHUB_RUN_ATTEMPT" != "1"' in country_step["run"]
    assert "dispatch-signed-snap-queue.yml" in country_step["run"]
    assert ".run_attempt >= 1" in country_step["run"]
    assert "jobs?filter=all&per_page=100" in country_step["run"]
    assert "Dispatch protected SNAP queue" in country_step["run"]
    assert '.conclusion == "skipped"' in country_step["run"]
    assert "snap-queue-reconciliation-$QUEUE_DISPATCHER_RUN_ID" in (country_step["run"])
    assert "snap-queue-plan.json" in country_step["run"]
    assert "dispatched-run-records.jsonl" in country_step["run"]
    assert "workflow_run_id == $run_id" in country_step["run"]
    assert "workflow_run_attempt == 1" in country_step["run"]
    assert (
        '"axiom-encode/data/encoding-queues/${QUEUE_ID}.json"' in (country_step["run"])
    )
    assert steps.index(country_step) < next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Checkout canonical RuleSpec country"
    )
    checkout_steps = [
        step for step in steps if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert checkout_steps
    assert all(step["with"]["persist-credentials"] is False for step in checkout_steps)
    assert all(step["with"]["fetch-depth"] == 0 for step in checkout_steps)

    source_bundle_step = next(
        step for step in steps if step.get("name") == "Validate atomic source inputs"
    )
    source_bundle_command = source_bundle_step["run"]
    assert source_bundle_step["env"]["ATOMIC_SOURCE_JSON"] == (
        "${{ inputs.source_bundle_json }}"
    )
    assert source_bundle_step["env"]["EXISTING_SIGNED_IMPORTS_JSON"] == (
        "${{ inputs.existing_signed_imports_json }}"
    )
    assert source_bundle_step["env"]["REPAIR_RUN_ID"] == ("${{ inputs.repair_run_id }}")
    assert "split-atomic-source-input" in source_bundle_command
    assert 'parse-source-bundle "$source_bundle_json"' in source_bundle_command
    assert 'primary_required_test_cases_json="$(jq -cer' in source_bundle_command
    assert "--primary-required-test-cases-json" in source_bundle_command
    assert "validate-source-add-targets" in source_bundle_command
    assert 'validate-source-add-targets "$RULESPEC_CHECKOUT"' in (source_bundle_command)
    assert 'parse-existing-signed-imports "$RULESPEC_CHECKOUT"' in (
        source_bundle_command
    )
    assert 'existing_import_args+=(--source-citation "$source_citation")' in (
        source_bundle_command
    )
    assert 'existing_import_args+=(--exclude-rulespec-path "$reserved_path")' in (
        source_bundle_command
    )
    assert '--primary-citation "$CITATION"' in source_bundle_command
    assert 'source_bundle_args+=(--exclude-citation "$DEPENDENT_CITATION")' in (
        source_bundle_command
    )
    assert (
        "queue-authorized re-encodes cannot add source inputs until queue "
        "generation identity binds them"
    ) in source_bundle_command
    assert (
        "source bundles require legacy replacements to merge first"
        in source_bundle_command
    )
    assert MULTILANE_REPLACE_ERROR in source_bundle_command
    assert steps.index(source_bundle_step) > next(
        index
        for index, step in enumerate(steps)
        if step.get("name") == "Install encoder"
    )

    repair_step = next(
        step
        for step in steps
        if step.get("name") == "Resolve trusted prior-run repair candidate"
    )
    repair_preflight = repair_step["run"].split('api_version="', 1)[0]
    assert "split-atomic-source-input" in repair_preflight
    assert ".venv/bin/python" not in repair_preflight
    identity_step = next(
        step
        for step in steps
        if step.get("name") == "Verify immutable checkout identities"
    )
    identity_command = identity_step["run"]
    assert "^[0-9a-f]{40}$" in identity_command
    assert "rev-parse HEAD" in identity_command
    assert "merge-base --is-ancestor" in identity_command
    assert "validate-rulespec-base" in identity_command
    assert '"$RULESPEC_REF" "$OPEN_PR" "$PR_BASE_BRANCH"' in identity_command
    assert identity_step["env"]["OPEN_PR"] == "${{ inputs.open_pr }}"
    assert identity_step["env"]["PR_BASE_BRANCH"] == ("${{ inputs.pr_base_branch }}")
    assert '"https://github.com/TheAxiomFoundation/rulespec-$COUNTRY"' in (
        identity_command
    )

    release_step = next(
        step
        for step in steps
        if step.get("name") == "Fetch pinned signed corpus release object"
    )
    assert release_step["env"] == {
        "RELEASE_BASE_URL": ("https://pub-a8952f8657fc49fda358146ac001366c.r2.dev"),
        "RULESPEC_CHECKOUT": "rulespec-${{ inputs.country }}",
        "QUEUE_ID": "${{ inputs.queue_id }}",
        "QUEUE_MANIFEST_SHA256": "${{ inputs.queue_manifest_sha256 }}",
    }
    release_command = release_step["run"]
    assert "materialize_corpus_release.py" in release_command
    assert "$RULESPEC_CHECKOUT/.axiom/toolchain.toml" in release_command
    assert 'pin --toolchain "$toolchain"' in release_command
    assert "validate-release-pin" in release_command
    assert '--manifest-sha256 "$QUEUE_MANIFEST_SHA256"' in release_command
    assert 'mktemp "$RUNNER_TEMP/' in release_command
    assert "/releases/${release_name}/${release_sha}.json" in release_command
    assert "--proto '=https' --proto-redir '=https' --tlsv1.2" in release_command
    assert "--max-filesize 16777216" in release_command
    assert "NEXT_PUBLIC_SUPABASE_ANON_KEY" not in release_command
    assert "SUPABASE" not in release_command
    assert "jq -ce" in release_command
    assert "release_object" in release_command
    assert (
        'materialize --toolchain "$toolchain" --response "$response"' in release_command
    )
    assert "--corpus-root axiom-corpus" in release_command
    assert 'test ! -L "$release_object"' in release_command
    assert 'chmod 0444 "$release_object"' in release_command
    assert 'merge-base --is-ancestor "$release_commit" HEAD' in release_command

    repair_step = next(
        step
        for step in steps
        if step.get("name") == "Resolve trusted prior-run repair candidate"
    )
    assert repair_step["id"] == "repair_candidate"
    assert repair_step["if"] == "${{ inputs.repair_run_id != '' }}"
    assert repair_step["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert repair_step["env"]["REPAIR_RUN_ID"] == "${{ inputs.repair_run_id }}"
    assert repair_step["env"]["RULESPEC_CHECKOUT"] == ("rulespec-${{ inputs.country }}")
    assert "REPAIR_TESTS_ONLY" not in repair_step["env"]
    repair_command = repair_step["run"]
    assert '.conclusion == "failure"' in repair_command
    assert '.event == "workflow_dispatch"' in repair_command
    assert '.head_branch == "main"' in repair_command
    assert ".run_attempt == 1" in repair_command
    assert '.path == ".github/workflows/targeted-signed-reencode.yml"' in repair_command
    assert "merge-base --is-ancestor" in repair_command
    assert '"$repair_encoder_commit" "$GITHUB_SHA"' in repair_command
    assert "repair replay is limited to one non-legacy target" in repair_command
    assert "repair_tests_only=false" in repair_command
    assert "repair_tests_only=true" in repair_command
    assert 'echo "tests_only=$repair_tests_only" >> "$GITHUB_OUTPUT"' in (
        repair_command
    )
    assert 'test -n "$REPLACE_RULESPEC_PATH"' not in repair_command
    assert "targeted-reencode-failure-${REPAIR_RUN_ID}-1" in repair_command
    assert "extract_repair_candidate.py" in repair_command
    assert '--atomic-source-json "$ATOMIC_SOURCE_JSON"' in repair_command
    for immutable_argument in (
        "--citation",
        "--country",
        "--encoder-commit",
        "--corpus-ref",
        "--rules-engine-ref",
        "--rulespec-ref",
        "--replace-rulespec-path",
        "--workflow-run-id",
    ):
        assert immutable_argument in repair_command
    assert "--allow-rulespec-base-advance" in repair_command
    assert "verify_repair_base_advance.py" in repair_command
    assert '--source-ref "$repair_source_rulespec_ref"' in repair_command
    assert '--current-ref "$RULESPEC_REF"' in repair_command
    assert '--candidate-path "$repair_candidate_path"' in repair_command
    assert '--rulespec-path "$REPLACE_RULESPEC_PATH"' in repair_command
    assert 'echo "runner=$(jq -r' in repair_command
    assert 'echo "source_rulespec_ref=$repair_source_rulespec_ref"' in repair_command

    provision_step = next(
        step
        for step in steps
        if step.get("name") == "Provision protected signing supervisor"
    )
    assert provision_step["id"] == "provision_signing_supervisor"
    assert "useradd --system --user-group --no-create-home" in (provision_step["run"])
    assert "axiom-verifier" in provision_step["run"]
    assert "axiom-signer" in provision_step["run"]
    assert provision_step["run"].count(
        "useradd --system --user-group --no-create-home"
    ) == 2
    for identity_check in (
        'test "$verifier_uid" -ne 0',
        'test "$verifier_gid" -ne 0',
        'test "$verifier_uid" -ne "$runner_uid"',
        'test "$verifier_gid" -ne "$runner_gid"',
        'test "$signer_uid" -ne 0',
        'test "$signer_gid" -ne 0',
        'test "$signer_uid" -ne "$verifier_uid"',
        'test "$signer_gid" -ne "$verifier_gid"',
        'test "$signer_uid" -ne "$runner_uid"',
        'test "$signer_gid" -ne "$runner_gid"',
    ):
        assert identity_check in provision_step["run"]
    assert "sudo chown 0:0 /opt" in provision_step["run"]
    assert "sudo chmod go-w /opt" in provision_step["run"]
    assert "./cmd/axiom-encode-process-guardian" in provision_step["run"]
    assert "sudo install -o root -g root -m 0555" in provision_step["run"]
    assert "/opt/axiom-verification/axiom-encode-signing-supervisor" in (
        provision_step["run"]
    )
    assert "/opt/axiom-verification/axiom-encode-apply-signer" in (
        provision_step["run"]
    )
    assert "/opt/axiom-verification/axiom-encode-process-guardian" in (
        provision_step["run"]
    )
    assert "stat -c '%u:%g:%a'" in provision_step["run"]
    assert '"0:0:555"' in provision_step["run"]
    assert 'test "$("$protected_binary" --build-kind)" = production' in (
        provision_step["run"]
    )
    assert "--git /usr/bin/git" in provision_step["run"]
    assert (
        '--encoder-git-root "$GITHUB_WORKSPACE/axiom-encode"' in provision_step["run"]
    )
    assert '--encoder-commit "$GITHUB_SHA"' in provision_step["run"]
    assert (
        "--encoder-origin-repository "
        "github.com/TheAxiomFoundation/axiom-encode" in provision_step["run"]
    )

    routing_step = next(
        step
        for step in steps
        if step.get("name") == "Verify protected RuleSpec routing"
    )
    routing_command = routing_step["run"]
    assert "trusted_path=/opt/axiom-verification/python/bin" in routing_command
    assert 'env -i PATH="$trusted_path" HOME="$trusted_home"' in routing_command
    assert "canonical_rulespec_repo_name(checkout)" in routing_command
    assert "inspect_canonical_rulespec_checkout" in routing_command
    assert '_harden_signing_capability_process(role="routing-probe")' in routing_command
    assert "libc.prctl(38, 1, 0, 0, 0)" in routing_command
    assert "protected RuleSpec routing rejected checkout" in routing_command
    assert "hardened RuleSpec routing rejected checkout" in routing_command

    cascade_step = next(
        step for step in steps if step.get("name") == "Validate dependent cascade"
    )
    assert "inputs.dependent_citation != ''" in cascade_step["if"]
    assert "inputs.second_dependent_citation != ''" in cascade_step["if"]
    assert "validate-dependent-cascade" in cascade_step["run"]
    assert 'dependent_citations=("$DEPENDENT_CITATION")' in cascade_step["run"]
    assert (
        'dependent_citations+=("$SECOND_DEPENDENT_CITATION")' in (cascade_step["run"])
    )
    assert (
        '"$RULESPEC_CHECKOUT" "$CITATION" "${dependent_citations[@]}"'
        not in (cascade_step["run"])
    )
    assert cascade_step["env"]["REPLACE_RULESPEC_PATH"] == (
        "${{ inputs.replace_rulespec_path }}"
    )
    assert 'cascade_target="$REPLACE_RULESPEC_PATH"' in cascade_step["run"]
    assert 'cascade_target="$REPLACE_LEGACY_RULESPEC_PATH"' in cascade_step["run"]
    assert (
        'cascade_args+=(--target-rulespec-path "$cascade_target")'
        in cascade_step["run"]
    )
    assert 'cascade_args+=("${dependent_citations[@]}")' in cascade_step["run"]
    assert '"${cascade_args[@]}"' in cascade_step["run"]

    signed_import_step = next(
        step for step in steps if step.get("name") == "Verify existing signed imports"
    )
    signed_import_command = signed_import_step["run"]
    assert steps.index(cascade_step) < steps.index(signed_import_step)
    assert "/opt/axiom-verification/axiom-encode-signing-supervisor" in (
        signed_import_command
    )
    assert "signed-import-inventory" in signed_import_command
    assert '--base-ref "$RULESPEC_REF"' in signed_import_command
    assert '--rulespec-path "$existing_import_path"' in signed_import_command
    assert '> "$RUNNER_TEMP/existing-signed-import-inventory.json"' in (
        signed_import_command
    )

    apply_step = next(
        step
        for step in steps
        if step.get("name") == "Encode, review, validate, and apply"
    )
    assert apply_step["id"] == "encode_apply"
    assert apply_step["env"]["AXIOM_ENCODE_APPLY_SIGNING_KEY"] == (
        "${{ secrets.AXIOM_ENCODE_APPLY_SIGNING_KEY }}"
    )
    assert "AXIOM_ENCODE_APPLY_CHECKOUT" not in apply_step["env"]
    command = apply_step["run"]
    assert "run_signed_encode()" in command
    assert "/opt/axiom-verification/axiom-encode-apply-signer run" in command
    assert "--key-env AXIOM_ENCODE_APPLY_SIGNING_KEY" in command
    assert (
        "TheAxiomFoundation/axiom-encode/.github/workflows/"
        "targeted-signed-reencode.yml@refs/heads/main" in command
    )
    assert "--allowed-event-name workflow_dispatch" in command
    assert "--apply" in command
    assert "--require-complete-source-unit" in command
    assert "--emit-final-rejected-candidate" in command
    assert '"$output_root/final-rejected-candidate"' in command
    assert 'local output_root="$RUNNER_TEMP/generated/$output_lane"' in command
    assert 'mkdir -p "$output_root"' in command
    assert command.index('mkdir -p "$output_root"') < command.index("local -a args=(")
    assert "initialize_verifier_boundary" in command
    assert "sudo install -o axiom-verifier -g axiom-verifier -m 0600" in command
    assert "cleanup_verifier_boundary" not in command
    assert "restore_verifier_outputs" not in command
    assert "sudo chown -hR axiom-verifier:axiom-verifier" in command
    assert "sudo chown -hR root:root" in command
    assert "sudo chmod -R a+rX,go-w" in command
    assert "--preserve-env=AXIOM_ENCODE_APPLY_SIGNING_KEY,OPENAI_API_KEY" in command
    assert "-u axiom-verifier --" not in command
    assert "verifier_guardian=(" in command
    assert 'run --uid "$verifier_uid" --gid "$verifier_gid" --' in command
    assert "require_split_signing_identities()" in command
    assert command.count("require_split_signing_identities") == 2
    for identity_check in (
        'test "$verifier_uid" -ne "$runner_uid"',
        'test "$verifier_gid" -ne "$runner_gid"',
        'test "$signer_uid" -ne "$verifier_uid"',
        'test "$signer_gid" -ne "$verifier_gid"',
        'test "$signer_uid" -ne "$runner_uid"',
        'test "$signer_gid" -ne "$runner_gid"',
    ):
        assert identity_check in command
    assert "run-root-controller --" in command
    assert '--signer-uid "$signer_uid"' in command
    assert '--signer-gid "$signer_gid"' in command
    assert '--supervisor-uid "$verifier_uid"' in command
    assert '--supervisor-gid "$verifier_gid"' in command
    assert command.index("run-root-controller --") < command.index(
        "/opt/axiom-verification/axiom-encode-apply-signer run"
    )
    assert command.count("run-root-controller --") == 1
    assert command.count("/opt/axiom-verification/axiom-encode-apply-signer run") == 1
    assert "--allow-local-dev" not in command
    root_controller = command[
        command.index("run-root-controller --") : command.index(
            'local encode_status="$?"'
        )
    ]
    ordered_root_controller_arguments = (
        "run-root-controller --",
        "/opt/axiom-verification/axiom-encode-apply-signer run",
        '--signer-uid "$signer_uid"',
        '--signer-gid "$signer_gid"',
        '--supervisor-uid "$verifier_uid"',
        '--supervisor-gid "$verifier_gid"',
        "--scope apply_ed25519",
        "--key-env AXIOM_ENCODE_APPLY_SIGNING_KEY",
        "--supervisor /opt/axiom-verification/axiom-encode-signing-supervisor",
        "--trusted-signing-roots /opt/axiom-verification/signing-trust-roots.json",
        "--trusted-python-runtime-root /opt/axiom-verification/python",
        "--trusted-python-import-root /opt/axiom-verification/python/lib/python3.13/site-packages",
        "--trusted-python-package-root /opt/axiom-verification/python/lib/python3.13/site-packages/axiom_encode",
        "--expected-github-repository TheAxiomFoundation/axiom-encode",
        "--allowed-workflow-ref TheAxiomFoundation/axiom-encode/.github/workflows/targeted-signed-reencode.yml@refs/heads/main",
        "--allowed-event-name workflow_dispatch",
        '-- "${args[@]}"',
    )
    argument_offsets = [
        root_controller.index(argument)
        for argument in ordered_root_controller_arguments
    ]
    assert argument_offsets == sorted(argument_offsets)
    assert 'local encode_status="$?"' in command
    assert "--skip-reviewers" not in command
    assert 'mktemp -d "$RUNNER_TEMP/axiom-targeted-review-finding.XXXXXX"' in command
    assert 'review_finding_path="$review_finding_dir/review-finding.txt"' in command
    assert "$GITHUB_WORKSPACE/axiom-rules-engine/.axiom-targeted" not in command
    assert "printf '%s\\n' \"$review_finding\"" in command
    assert 'args+=(--review-findings "$review_finding_path")' in command
    assert '--repair-candidate-root "$REPAIR_CANDIDATE_ROOT"' in command
    assert (
        'if [ -n "${REPAIR_CANDIDATE_ROOT:-}" ] && \\\n'
        '     [ -n "$replacement_path" ]; then'
    ) in command
    assert '--repair-candidate-path "$REPAIR_CANDIDATE_PATH"' in command
    assert "--repair-candidate-rulespec-sha256" in command
    assert '"$REPAIR_CANDIDATE_RULESPEC_SHA256"' in command
    assert "--repair-candidate-tests-sha256" in command
    assert '"$REPAIR_CANDIDATE_TESTS_SHA256"' in command
    assert "args+=(--repair-candidate-tests-only)" in command
    assert (
        len(
            re.findall(
                r'"\$required_imports_enabled" "\[\]" \\\n[ \t]+'
                r'"\$primary_required_test_cases_json"',
                command,
            )
        )
        == 2
    )
    assert apply_step["env"]["REPAIR_CANDIDATE_ROOT"] == (
        "${{ steps.repair_candidate.outputs.root }}"
    )
    assert apply_step["env"]["REPAIR_CANDIDATE_PATH"] == (
        "${{ steps.repair_candidate.outputs.path }}"
    )
    assert apply_step["env"]["REPAIR_CANDIDATE_RULESPEC_SHA256"] == (
        "${{ steps.repair_candidate.outputs.rulespec_sha256 }}"
    )
    assert apply_step["env"]["REPAIR_CANDIDATE_TESTS_SHA256"] == (
        "${{ steps.repair_candidate.outputs.tests_sha256 }}"
    )
    assert apply_step["env"]["REPAIR_TESTS_ONLY"] == (
        "${{ steps.repair_candidate.outputs.tests_only }}"
    )
    assert ': "${REPAIR_TESTS_ONLY:=false}"' in command
    assert "REPAIR_TESTS_ONLY=false" not in command
    assert "REPAIR_TESTS_ONLY=true" not in command
    assert "args+=(--apply-target-only)" in command
    assert 'args+=(--replace-rulespec-path "$replacement_path")' in command
    assert 'args+=(--create-rulespec-path "$replacement_path")' in command
    assert '[ -e "$RULESPEC_CHECKOUT/$replacement_path" ]' in command
    assert '[ -L "$RULESPEC_CHECKOUT/$replacement_path" ]' in command
    assert 'local require_direct_imports="$7"' in command
    assert 'args+=(--required-import-rulespec-path "$required_import_path")' in (
        command
    )
    assert "--legacy-dependent-rulespec-path" in command
    assert 'citation-rulespec-path "$DEPENDENT_CITATION"' in command
    assert '[ "$target_only" = "true" ]' in command
    assert (
        "queue-authorized re-encodes cannot override the RuleSpec target path"
        in command
    )
    assert '--output "$output_root"' in command
    assert '"$SECOND_DEPENDENT_CITATION"' in command
    assert '"$SECOND_DEPENDENT_REVIEW_FINDING" false dependent-2 "" "" false' in command
    assert '"$CITATION" "$REVIEW_FINDING" true target \\\n' in command
    assert '"$DEPENDENT_CITATION" "$DEPENDENT_REVIEW_FINDING" \\\n' in command
    assert '"$REPLACE_RULESPEC_PATH" "$REPLACE_LEGACY_RULESPEC_PATH"' in command
    assert '"$CITATION" "$REVIEW_FINDING" false \\\n' in command
    assert "dependent review finding is required with dependent citation" in command
    assert "parse-source-bundle" in command
    assert "parse-existing-signed-imports" in command
    assert "parse-canonical-refresh-bundle" in command
    assert "verify-canonical-refresh-target" in command
    assert '"$refresh_citation" "$refresh_finding" false "$refresh_lane"' in command
    assert '"$RUNNER_TEMP/existing-signed-import-paths.txt"' in command
    assert "source_bundle_count + existing_import_count" in command
    assert '> "$RUNNER_TEMP/source-bundle.json"' in command
    assert '> "$RUNNER_TEMP/source-bundle-citations.txt"' in command
    assert 'source_lane="$(printf \'source-%02d\' "$source_index")"' in command
    assert '"$source_citation" "" true "$source_lane" "" "" false' in command
    assert "source bundles require legacy replacements to merge first" in command
    assert "target-preflight" not in command
    assert "checkpoint_signed_changes()" in command
    assert "unset AXIOM_ENCODE_APPLY_SIGNING_KEY" in command
    assert '--json > "$guard_json" 2> "$guard_stderr"' in command
    assert 'local guard_status="$?"' in command
    assert "if ! jq -e -s" in command
    assert "length == 1" in command
    assert 'and keys == ["issues", "passed", "repo"]' in command
    assert 'and (.issues | type == "array"' in command
    assert 'if [ "$guard_status" -eq 0 ]; then' in command
    assert 'checkpoint-guard-generated.stdout.log"' in command
    assert "run_verifier_axiom checkpoint-signed-backfill" in command
    assert '> "$RUNNER_TEMP/final-checkpoint.json"' in command
    assert 'commit -m "$message"' not in command
    source_loop = "while IFS= read -r source_citation; do"
    assert command.rindex(source_loop) < command.rindex('"$CITATION" "$REVIEW_FINDING"')
    assert "Complete signed target transaction for ${CITATION}" in command
    assert (
        "queue-authorized re-encodes cannot add source inputs until queue "
        "generation identity binds them"
    ) in command
    assert steps.index(cascade_step) < steps.index(apply_step)
    assert steps.index(signed_import_step) < steps.index(apply_step)

    failure_package_step = next(
        step
        for step in steps
        if step.get("name") == "Package failed re-encode diagnostics"
    )
    upload_step = next(
        step for step in steps if step.get("name") == "Upload signed re-encode artifact"
    )
    publication_upload_step = next(
        step
        for step in steps
        if step.get("name") == "Upload protected publication witnesses"
    )
    assert steps.index(upload_step) + 1 == steps.index(publication_upload_step)
    assert steps.index(publication_upload_step) + 1 == steps.index(failure_package_step)
    assert publication_upload_step["if"] == "${{ inputs.open_pr }}"
    assert "publication-witness.json" in publication_upload_step["with"]["path"]
    assert "publication-receipt.json" in publication_upload_step["with"]["path"]
    assert failure_package_step["if"] == "${{ failure() && !cancelled() }}"
    assert set(failure_package_step["env"]) == {
        "ATOMIC_SOURCE_JSON",
        "CITATION",
        "CORPUS_REF",
        "COUNTRY",
        "DEPENDENT_CITATION",
        "ENCODE_APPLY_CONCLUSION",
        "ENCODE_APPLY_OUTCOME",
        "EXISTING_SIGNED_IMPORTS_JSON",
        "FINALIZE_SIGNED_REENCODE_ARTIFACT_CONCLUSION",
        "FINALIZE_SIGNED_REENCODE_ARTIFACT_OUTCOME",
        "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
        "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON",
        "OPEN_PR",
        "PACKAGE_EXACT_GENERATED_CHANGES_CONCLUSION",
        "PACKAGE_EXACT_GENERATED_CHANGES_OUTCOME",
        "COMMIT_REVIEWED_LANE_CHANGES_CONCLUSION",
        "COMMIT_REVIEWED_LANE_CHANGES_OUTCOME",
        "PR_BASE_BRANCH",
        "REPAIR_CANDIDATE_CONCLUSION",
        "REPAIR_CANDIDATE_OUTCOME",
        "REPAIR_CANDIDATE_PATH",
        "REPAIR_CANDIDATE_RUNNER",
        "REPAIR_CANDIDATE_SOURCE_RULESPEC_REF",
        "REPAIR_CANDIDATE_RULESPEC_SHA256",
        "REPAIR_CANDIDATE_TESTS_SHA256",
        "REPAIR_RUN_ID",
        "PROVISION_SIGNING_SUPERVISOR_CONCLUSION",
        "PUBLISH_LANE_PULL_REQUEST_CONCLUSION",
        "PUBLISH_LANE_PULL_REQUEST_OUTCOME",
        "QUEUE_DISPATCHER_RUN_ID",
        "QUEUE_ID",
        "QUEUE_ITEM_GENERATION_SHA256",
        "QUEUE_ITEM_ID",
        "QUEUE_MANIFEST_SHA256",
        "REPLACE_LEGACY_RULESPEC_PATH",
        "REPLACE_RULESPEC_PATH",
        "RULES_ENGINE_REF",
        "RULESPEC_REF",
        "SECOND_DEPENDENT_CITATION",
        "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH",
        "UPLOAD_SIGNED_REENCODE_ARTIFACT_CONCLUSION",
        "UPLOAD_SIGNED_REENCODE_ARTIFACT_OUTCOME",
        "VERIFY_EXISTING_SIGNED_IMPORT_INTEGRITY_CONCLUSION",
        "VERIFY_EXISTING_SIGNED_IMPORT_INTEGRITY_OUTCOME",
        "VERIFY_GENERATED_PROVENANCE_CONCLUSION",
        "VERIFY_GENERATED_PROVENANCE_OUTCOME",
    }
    assert (
        failure_package_step["env"]["PROVISION_SIGNING_SUPERVISOR_CONCLUSION"]
        == "${{ steps.provision_signing_supervisor.conclusion }}"
    )
    assert "toJSON(steps)" not in json.dumps(failure_package_step)
    assert failure_package_step["env"]["REPAIR_CANDIDATE_PATH"] == (
        "${{ steps.repair_candidate.outputs.path }}"
    )
    assert failure_package_step["env"]["REPAIR_CANDIDATE_RUNNER"] == (
        "${{ steps.repair_candidate.outputs.runner }}"
    )
    assert failure_package_step["env"]["REPAIR_CANDIDATE_SOURCE_RULESPEC_REF"] == (
        "${{ steps.repair_candidate.outputs.source_rulespec_ref }}"
    )
    assert failure_package_step["env"]["REPAIR_CANDIDATE_RULESPEC_SHA256"] == (
        "${{ steps.repair_candidate.outputs.rulespec_sha256 }}"
    )
    assert failure_package_step["env"]["REPAIR_CANDIDATE_TESTS_SHA256"] == (
        "${{ steps.repair_candidate.outputs.tests_sha256 }}"
    )
    failure_package_command = failure_package_step["run"]
    assert "workflow_python=/usr/bin/python3" in failure_package_command
    assert '"${PROVISION_SIGNING_SUPERVISOR_CONCLUSION:-}" = success' in (
        failure_package_command
    )
    assert "workflow_python=/opt/axiom-verification/python/bin/python" in (
        failure_package_command
    )
    assert 'test -x "$workflow_python"' in failure_package_command
    assert '"$workflow_python" -I -' in failure_package_command
    assert "read_bounded_regular_file" in failure_package_command
    assert "followlinks=False" in failure_package_command
    assert "generated diagnostics exceed entry limit" in failure_package_command
    assert "generated diagnostics exceed file limit" in failure_package_command
    assert "generated diagnostics exceed size limit" in failure_package_command
    assert 'tar -cf "$RUNNER_TEMP/targeted-reencode-failure.tar"' in (
        failure_package_command
    )
    assert '-C "$artifact" .' in failure_package_command
    assert '"failed_steps": failed_steps' in failure_package_command
    assert '"generated_lanes": generated_lanes' in failure_package_command
    assert '"queue_item_generation_sha256"' in failure_package_command
    assert '"step_outcomes": step_outcomes' in failure_package_command
    assert '"schema": "axiom-encode/failed-reencode-diagnostics/v1"' in (
        failure_package_command
    )
    assert 'guard_root / "repair-candidate.json"' not in failure_package_command
    assert "consumed_rulespec_sha256 = os.environ.get(" in failure_package_command
    assert "consumed_tests_sha256 = os.environ.get(" in failure_package_command

    failure_upload_step = next(
        step
        for step in steps
        if step.get("name") == "Upload failed re-encode diagnostics"
    )
    assert steps.index(failure_package_step) + 1 == steps.index(failure_upload_step)
    assert "failure() && !cancelled()" in failure_upload_step["if"]
    assert (
        "steps.package_failed_reencode_diagnostics.outcome == 'success'"
        in (failure_upload_step["if"])
    )
    assert failure_upload_step["with"] == {
        "name": (
            "targeted-reencode-failure-${{ github.run_id }}-${{ github.run_attempt }}"
        ),
        "path": "${{ runner.temp }}/targeted-reencode-failure.tar",
        "if-no-files-found": "error",
        "retention-days": 90,
    }

    integrity_step = next(
        step
        for step in steps
        if step.get("name") == "Verify existing signed import integrity"
    )
    assert integrity_step["id"] == "verify_existing_signed_import_integrity"
    integrity_command = integrity_step["run"]
    assert steps.index(apply_step) < steps.index(integrity_step)
    assert "signed-import-inventory" in integrity_command
    assert "existing-signed-import-inventory-final.json" in integrity_command
    assert "cmp --silent" in integrity_command

    verification_only_steps = (
        signed_import_step,
        next(step for step in steps if step.get("name") == "Verify generated provenance"),
        integrity_step,
        next(step for step in steps if step.get("name") == "Package exact generated changes"),
        next(step for step in steps if step.get("name") == "Commit reviewed lane changes locally"),
    )
    for verification_step in verification_only_steps:
        verification_command = verification_step["run"]
        assert "/opt/axiom-verification/axiom-encode-process-guardian" in (
            verification_command
        )
        assert "run-root-controller" not in verification_command
        assert "/opt/axiom-verification/axiom-encode-signing-supervisor" in (
            verification_command
        )
    all_workflow_commands = "\n".join(
        step.get("run", "") for step in steps
    )
    assert all_workflow_commands.count("run-root-controller --") == 1

    package_step = next(
        step for step in steps if step.get("name") == "Package exact generated changes"
    )
    assert package_step["id"] == "package_exact_generated_changes"
    package_command = package_step["run"]
    assert "immutable_source_bundle" in package_command
    assert "immutable_refresh_bundle" in package_command
    assert "len(canonical_refresh_items) > 1" in package_command
    assert "signed replacement_target evidence and expected-parent lineage" in (
        package_command
    )
    assert package_step["env"]["REVIEW_FINDING_PRESENT"] == (
        "${{ inputs.review_finding != '' }}"
    )
    assert package_step["env"]["REVIEW_FINDING"] == "${{ inputs.review_finding }}"
    assert package_step["env"]["DEPENDENT_REVIEW_FINDING_PRESENT"] == (
        "${{ inputs.dependent_review_finding != '' }}"
    )
    assert package_step["env"]["DEPENDENT_REVIEW_FINDING"] == (
        "${{ inputs.dependent_review_finding }}"
    )
    assert package_step["env"]["SECOND_DEPENDENT_REVIEW_FINDING"] == (
        "${{ inputs.second_dependent_review_finding }}"
    )
    assert package_step["env"]["PR_BASE_BRANCH"] == ("${{ inputs.pr_base_branch }}")
    assert package_step["env"]["RULESPEC_REF"] == "${{ inputs.rulespec_ref }}"
    assert package_step["env"]["CORPUS_REF"] == "${{ inputs.corpus_ref }}"
    assert package_step["env"]["ATOMIC_SOURCE_JSON"] == (
        "${{ inputs.source_bundle_json }}"
    )
    assert package_step["env"]["REPAIR_CANDIDATE_PATH"] == (
        "${{ steps.repair_candidate.outputs.path }}"
    )
    assert package_step["env"]["REPAIR_CANDIDATE_RUNNER"] == (
        "${{ steps.repair_candidate.outputs.runner }}"
    )
    assert package_step["env"]["REPAIR_CANDIDATE_SOURCE_RULESPEC_REF"] == (
        "${{ steps.repair_candidate.outputs.source_rulespec_ref }}"
    )
    assert package_step["env"]["REPAIR_CANDIDATE_RULESPEC_SHA256"] == (
        "${{ steps.repair_candidate.outputs.rulespec_sha256 }}"
    )
    assert package_step["env"]["REPAIR_CANDIDATE_TESTS_SHA256"] == (
        "${{ steps.repair_candidate.outputs.tests_sha256 }}"
    )
    assert "capture_git" not in package_command
    assert "--binary --full-index --no-ext-diff --no-textconv" not in package_command
    assert "# BEGIN TARGETED_REENCODE_IMMUTABLE_SNAPSHOT" in package_command
    assert "# TARGETED_REENCODE_SNAPSHOT_CAPTURED" in package_command
    assert "# END TARGETED_REENCODE_IMMUTABLE_SNAPSHOT" in package_command
    assert 'snapshot_parent="/opt/axiom-verification/' in package_command
    assert 'snapshot_checkout="$snapshot_parent/$rulespec_checkout_name"' in (
        package_command
    )
    assert 'snapshot_git=(sudo env -i HOME="$protected_git_home"' in (package_command)
    assert (
        '"https://github.com/TheAxiomFoundation/${rulespec_checkout_name}.git"'
        in package_command
    )
    assert '-C "$snapshot_checkout" checkout --detach \\\n  "$RULESPEC_REF"' in (
        package_command
    )
    assert '"https://github.com/TheAxiomFoundation/axiom-corpus.git"' in (
        package_command
    )
    assert '"https://github.com/TheAxiomFoundation/axiom-encode.git"' in (
        package_command
    )
    assert 'captured_rulespec_tree="$(jq -r \'.tree\'' in package_command
    assert '"$RUNNER_TEMP/final-checkpoint.json"' in package_command
    assert "copy_untrusted_rulespec_worktree.py" in package_command
    assert '"$live_rulespec_checkout" "$snapshot_checkout"' in package_command
    assert '--source-owner-uid "$verifier_uid"' in package_command
    assert '> "$RUNNER_TEMP/worktree-copy.json"' in package_command
    assert 'axiom-encode/untrusted-worktree-copy/v1' in package_command
    assert "sudo chown -hR axiom-verifier:axiom-verifier" in package_command
    assert "snapshot_git=(sudo -u axiom-verifier -- env -i" in package_command
    assert "materialize_corpus_release.py" in package_command
    assert 'protected_release_response="$snapshot_parent/release-response.json"' in (
        package_command
    )
    protected_copy = package_command.index("copy_untrusted_rulespec_worktree.py")
    protected_release = package_command.index("snapshot_release_commit")
    protected_boundary = package_command.index(
        "# END TARGETED_REENCODE_IMMUTABLE_SNAPSHOT"
    )
    assert protected_copy < protected_release < protected_boundary
    assert 'RULESPEC_CHECKOUT="$snapshot_checkout"' in package_command
    assert "export RULESPEC_CHECKOUT" in package_command
    final_guard = "-- /opt/axiom-verification/axiom-encode guard-generated"
    assert final_guard in package_command
    assert '--base-ref "$RULESPEC_REF"' in package_command
    assert '> "$RUNNER_TEMP/guard-generated.json"' in package_command
    assert package_command.index("# TARGETED_REENCODE_SNAPSHOT_CAPTURED") < (
        package_command.index(final_guard)
    )
    assert package_command.index('RULESPEC_CHECKOUT="$snapshot_checkout"') < (
        package_command.index(final_guard)
    )
    stage_snapshot = "-- /opt/axiom-verification/axiom-encode stage-signed-backfill"
    assert stage_snapshot in package_command
    protected_stage = package_command.index(
        stage_snapshot,
        package_command.index("# END TARGETED_REENCODE_IMMUTABLE_SNAPSHOT"),
    )
    assert "sudo -u axiom-verifier -- \\\n" in package_command
    assert '"${verifier_guardian[@]}" \\\n' in package_command
    assert "/opt/axiom-verification/axiom-encode-process-guardian" in package_command
    assert "sudo /opt/axiom-verification/axiom-encode-signing-supervisor" not in (
        package_command
    )
    assert protected_stage < package_command.index(final_guard)
    assert protected_stage < package_command.index("staged_snapshot_tree")
    assert 'test "$staged_snapshot_tree" = "$captured_rulespec_tree"' in (
        package_command
    )
    assert "git config --system" not in package_command
    assert "sudo chown -hR root:root \\\n" in package_command
    assert '"$snapshot_corpus" "$snapshot_encoder" "$protected_git_home"' in (
        package_command
    )
    assert (
        'protected_release_object="$snapshot_corpus/releases/$release_name/$release_sha.json"'
        in package_command
    )
    assert 'sudo chmod 0444 "$protected_release_object"' in package_command
    assert "stat -c '%u:%a' \"$protected_release_object\"" in package_command
    assert '--corpus-path "$RULESPEC_CORPUS_CHECKOUT"' in package_command
    assert '--expected-encoder-checkout "$PROTECTED_ENCODER_CHECKOUT"' in (
        package_command
    )
    assert package_command.index(final_guard) < package_command.index(
        "# END TARGETED_REENCODE_CONTEXT_VERIFIER"
    )
    assert package_command.index(
        "# END TARGETED_REENCODE_CONTEXT_VERIFIER"
    ) < package_command.index("validated_snapshot_tree")
    assert 'test "$validated_snapshot_tree" = "$staged_snapshot_tree"' in (
        package_command
    )
    assert "diff --quiet --" in package_command
    assert package_command.count("ls-files --others)") == 2
    assert 'git -C "$RULESPEC_CHECKOUT" add -A' not in package_command
    assert '"$artifact/rulespec-tree.txt"' in package_command
    assert '"$artifact/rulespec-generated-head.txt"' in package_command
    assert '"$RULESPEC_REF" HEAD >> "$artifact/status.txt"' in package_command
    assert "diff --binary --full-index HEAD" not in package_command
    assert 'cp "$RUNNER_TEMP/source-bundle.json" "$artifact/source-bundle.json"' in (
        package_command
    )
    assert (
        'cp "$RUNNER_TEMP/canonical-refresh-bundle.json" \\\n'
        '  "$artifact/canonical-refresh-bundle.json"'
    ) in package_command
    assert '"$artifact/existing-signed-imports.json"' in package_command
    assert '"$artifact/signed-import-inventory.json"' in package_command
    assert '"$artifact/context-manifest.json"' in package_command
    assert '"$artifact/worktree-copy.json"' in package_command
    assert '".axiom/encoding-manifests"' in package_command
    assert 'citation = effective.get("citation")' in package_command
    assert 'evidence_manifest["context_manifest_file"]' in package_command
    assert "evidence_manifest.get(" in package_command
    assert "is still pending" in package_command
    assert "primary_paths != [normalized_replacement_path]" in package_command
    assert "replacement apply manifest primary path does not " in package_command
    assert "match the requested RuleSpec target" in package_command
    assert 'finding.get("content")' in package_command
    assert 'finding.get("sha256")' in package_command
    assert '"dependent-context-manifest.json"' in package_command
    assert 'f"{source_lane}-context-manifest.json"' in package_command
    assert 'os.environ["RULESPEC_REF"], "--"' in package_command
    assert '"source_bundle": json.loads(' in package_command
    assert '"existing_signed_imports": json.loads(' in package_command
    assert '"signed_import_inventory_sha256": hashlib.sha256(' in package_command
    assert '"rulespec_base": os.environ["RULESPEC_REF"]' in package_command
    assert '"rulespec_generated_head": rulespec_generated_head' in (package_command)
    assert '"rulespec_validated_tree": rulespec_validated_tree' in package_command
    assert '"pr_base_branch": os.environ["PR_BASE_BRANCH"]' in package_command
    assert '"queue_id": os.environ.get("QUEUE_ID") or None' in package_command
    assert '"queue_item_id": os.environ.get("QUEUE_ITEM_ID") or None' in (
        package_command
    )
    assert '"queue_item_generation_sha256": (' in package_command
    assert '"queue_manifest_sha256": (' in package_command
    assert '"repair_candidate": repair_candidate' in package_command
    assert 'Path(os.environ["RUNNER_TEMP"]) / "repair-candidate.json"' not in (
        package_command
    )
    assert "consumed_rulespec_sha256 = os.environ.get(" in package_command
    assert "consumed_tests_sha256 = os.environ.get(" in package_command
    assert '"repair_run_id": os.environ.get("REPAIR_RUN_ID") or None' in (
        package_command
    )
    assert '"workflow_run_attempt": int(os.environ["GITHUB_RUN_ATTEMPT"])' in (
        package_command
    )
    trusted_python = "/opt/axiom-verification/python/bin/python"
    assert f"workflow_python=({trusted_python} -I)" in package_command
    assert "verify_targeted_context() {" in package_command
    assert "write_targeted_metadata() {" in package_command
    assert "seal_targeted_reencode_artifact.py" in package_command
    assert 'copy \\\n  --exclude-guard-logs "$artifact" "$sealed_artifact"' in (
        package_command
    )
    assert 'verify --require-root-owned "$artifact"' in package_command
    assert package_command.count("sudo /usr/bin/jq") == 2
    assert "/opt/axiom-verification/python/bin/git" in package_command
    assert package_command.count("verify_targeted_context") == 3
    assert package_command.count("write_targeted_metadata") == 3
    assert 'artifact="$snapshot_parent/artifact"' in package_command

    commit_step = next(
        step
        for step in steps
        if step.get("name") == "Commit reviewed lane changes locally"
    )
    assert commit_step["id"] == "commit_reviewed_lane_changes"
    assert commit_step["env"]["RULESPEC_REF"] == "${{ inputs.rulespec_ref }}"
    assert 'snapshot_parent="/opt/axiom-verification/' in commit_step["run"]
    assert "snapshot_git=(sudo -u axiom-verifier -- env -i" in commit_step["run"]
    assert "axiom-encode-signing-supervisor \\\n" in commit_step["run"]
    assert "--trusted-signing-roots" in commit_step["run"]
    assert "--trusted-python-runtime-root" in commit_step["run"]
    assert "--trusted-python-import-root" in commit_step["run"]
    assert "--trusted-python-package-root" in commit_step["run"]
    assert (
        "-- /opt/axiom-verification/axiom-encode stage-signed-backfill"
        in (commit_step["run"])
    )
    assert '--repo "$RULESPEC_CHECKOUT"' in commit_step["run"]
    assert '--corpus-path "$RULESPEC_CORPUS_CHECKOUT"' in commit_step["run"]
    assert (
        '--expected-encoder-checkout "$PROTECTED_ENCODER_CHECKOUT"'
        in (commit_step["run"])
    )
    assert (
        'commit -m "Add signed encoding manifest for ${CITATION}"'
        in (commit_step["run"])
    )
    assert 'rev-parse HEAD^)" = "$RULESPEC_REF"' in commit_step["run"]
    assert 'rev-parse HEAD^{tree})" = "$validated_tree"' in commit_step["run"]
    assert "Axiom-Artifact-Manifest-SHA256:" in commit_step["run"]
    assert "verify --require-root-owned" in commit_step["run"]
    assert "write-publication-witness" in commit_step["run"]
    assert '"$snapshot_parent/publication-witness.json"' in commit_step["run"]

    guard_step = next(
        step for step in steps if step.get("name") == "Verify generated provenance"
    )
    assert guard_step["id"] == "verify_generated_provenance"
    assert "guard-generated" in guard_step["run"]
    assert '--base-ref "$RULESPEC_REF"' in guard_step["run"]
    assert "guard_ref_args" not in guard_step["run"]
    assert 'guard_status="$?"' in guard_step["run"]
    assert 'jq . "$RUNNER_TEMP/guard-generated.json" >&2' in guard_step["run"]

    secret_steps = [
        step
        for step in steps
        if "AXIOM_ENCODE_APPLY_SIGNING_KEY" in (step.get("env") or {})
    ]
    assert secret_steps == [apply_step]

    publish_step = next(
        step
        for step in steps
        if step.get("name") == "Push lane branch and open draft pull request"
    )
    assert publish_step["id"] == "publish_lane_pull_request"
    assert publish_step["if"] == "${{ inputs.open_pr }}"
    assert publish_step["env"]["GH_TOKEN"] == "${{ secrets.AXIOM_REPO_TOKEN }}"
    assert publish_step["env"]["PR_BASE_BRANCH"] == ("${{ inputs.pr_base_branch }}")
    assert "AXIOM_ENCODE_APPLY_SIGNING_KEY" not in publish_step["env"]
    publish_command = publish_step["run"]
    assert f"workflow_python=({trusted_python} -I)" in publish_command
    assert 'repo="TheAxiomFoundation/rulespec-${COUNTRY}"' in publish_command
    publish_push = (
        '"${protected_git[@]}" -c core.hooksPath=/dev/null push \\\n'
        '  "https://github.com/${repo}.git"'
    )
    assert publish_push in publish_command
    assert 'publish_git_home="$snapshot_parent/publish-git-home"' in publish_command
    assert 'credential.https://github.com.helper ""' in publish_command
    assert '"!/usr/bin/gh auth git-credential"' in publish_command
    assert 'GIT_CONFIG_GLOBAL="$publish_git_config" GIT_CONFIG_NOSYSTEM=1' in (
        publish_command
    )
    assert 'GH_TOKEN="$GH_TOKEN"' in publish_command
    assert "stat -c '%u:%a' \"$publish_git_config\"" in publish_command
    assert "gh auth setup-git" not in publish_command
    assert 'publication_witness="$snapshot_parent/publication-witness.json"' in (
        publish_command
    )
    assert 'artifact="$snapshot_parent/artifact"' in publish_command
    assert "verify --require-root-owned" in publish_command
    assert "Axiom-Artifact-Manifest-SHA256:" in publish_command
    assert "write-publication-receipt" in publish_command
    assert "$RUNNER_TEMP/lane-pull-request.json" not in publish_command
    assert '"$artifact/rulespec-tree.txt"' in publish_command
    assert ".rulespec_validated_tree == $tree" in publish_command
    assert "status --porcelain" in publish_command
    assert "rev-parse HEAD^{tree}" in publish_command
    assert publish_command.index("rev-parse HEAD^{tree}") < publish_command.index(
        publish_push
    )
    assert '"$COUNTRY" "$GITHUB_RUN_ID" "$GITHUB_RUN_ATTEMPT"' in publish_command
    assert "core.hooksPath=/dev/null" in publish_command
    assert "ls-remote --exit-code" in publish_command
    assert 'test "$remote_base_sha" = "$RULESPEC_REF"' in publish_command
    assert '-c "safe.directory=$RULESPEC_CHECKOUT"' in publish_command
    assert '"${publication_commit}:refs/heads/${branch}"' in publish_command
    assert publish_command.count('"${publication_commit}:refs/heads/${branch}"') == 1
    assert '"$snapshot_parent/axiom-encode/scripts/prepare_signed_backfill.py"' in (
        publish_command
    )
    assert 'protected_gh=(env -i HOME=/nonexistent GH_TOKEN="$GH_TOKEN"' in (
        publish_command
    )
    assert "GH_PROMPT_DISABLED=1 PATH=/usr/bin:/bin" in publish_command
    assert "/usr/bin/gh api --hostname github.com" in publish_command
    assert '"${protected_gh[@]}" --method POST' in publish_command
    assert '"${protected_gh[@]}" --method PATCH' in publish_command
    assert "/usr/bin/gh api --method" not in publish_command
    assert '-f base="$PR_BASE_BRANCH"' in publish_command
    assert "-F draft=true" in publish_command
    assert "Queue item:" in publish_command
    assert "Queue generation SHA-256:" in publish_command
    assert "Queue manifest SHA-256:" in publish_command
    assert ".base.ref == $base_branch and .base.sha == $base_sha" in publish_command
    assert ".base.repo.full_name == $repo" in publish_command
    assert ".head.ref == $head_branch and .head.sha == $head_sha" in publish_command
    assert ".head.repo.full_name == $repo" in publish_command
    assert '(.number | type) == "number" and .number > 0' in publish_command
    assert '.draft == true and .state == "open"' in publish_command
    assert 'test "$published_head_sha" = "$publication_commit"' in publish_command
    assert 'test "$published_head_ref" = "refs/heads/${branch}"' in publish_command
    assert "pulls/${pr_number}" in publish_command
    assert "-f state=closed" in publish_command
    assert '":refs/heads/${branch}"' in publish_command
    assert "created pull request does not bind the reviewed base and head" in (
        publish_command
    )
    assert "SHA256SUMS" not in publish_command
    assert not any(
        " push " in f" {step.get('run', '')} "
        for step in steps[: steps.index(publish_step)]
    )

    checksum_step = next(
        step
        for step in steps
        if step.get("name") == "Finalize signed re-encode artifact checksums"
    )
    assert checksum_step["id"] == "finalize_signed_reencode_artifact"
    assert "if" not in checksum_step
    checksum_command = checksum_step["run"]
    assert 'artifact="$snapshot_parent/artifact"' in checksum_command
    assert "verify --require-root-owned" in checksum_command
    assert "sha256sum" not in checksum_command
    assert upload_step["id"] == "upload_signed_reencode_artifact"
    assert upload_step["with"]["name"] == (
        "targeted-reencode-${{ github.run_id }}-${{ github.run_attempt }}"
    )
    assert (
        "/opt/axiom-verification/targeted-reencode-snapshot-"
        in (upload_step["with"]["path"])
    )
    assert upload_step["with"]["path"].endswith("/artifact")
    assert steps.index(checksum_step) + 1 == steps.index(upload_step)
    assert steps.index(failure_upload_step) == len(steps) - 1


@pytest.mark.parametrize(
    ("atomic_source_json", "expected_tests_only"),
    [
        (
            json.dumps(
                {
                    "schema": "axiom-encode/atomic-source-transaction/v2",
                    "source_bundle": [],
                    "canonical_refresh_bundle": [],
                    "primary_required_test_cases": [],
                    "target_operation": "replace",
                }
            ),
            "false",
        ),
        (
            json.dumps(
                {
                    "schema": "axiom-encode/atomic-source-transaction/v2",
                    "source_bundle": ["us/statute/7/2015/f"],
                    "canonical_refresh_bundle": [],
                    "primary_required_test_cases": [],
                    "target_operation": "replace",
                }
            ),
            "false",
        ),
        (
            json.dumps(
                {
                    "schema": "axiom-encode/atomic-source-transaction/v2",
                    "source_bundle": [],
                    "canonical_refresh_bundle": [],
                    "primary_required_test_cases": [
                        {
                            "name": "required control",
                            "period": {
                                "period_kind": "tax_year",
                                "start": "2026-01-01",
                                "end": "2026-12-31",
                            },
                            "input": {"example_input": 1},
                            "required_output": {"example_output": 1},
                        }
                    ],
                    "target_operation": "replace",
                }
            ),
            "true",
        ),
    ],
)
def test_repair_preflight_splits_atomic_source_before_encoder_install(
    tmp_path: Path,
    atomic_source_json: str,
    expected_tests_only: str,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Resolve trusted prior-run repair candidate"
    ).split('api_version="', 1)[0]
    command = command.replace(
        "axiom-encode/scripts/prepare_signed_backfill.py",
        str(ROOT / "scripts/prepare_signed_backfill.py"),
    )

    completed = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ATOMIC_SOURCE_JSON": atomic_source_json,
            "DEPENDENT_CITATION": "",
            "EXISTING_SIGNED_IMPORTS_JSON": "[]",
            "GITHUB_OUTPUT": str(tmp_path / "github-output"),
            "GITHUB_RUN_ID": "200",
            "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
            "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": "[]",
            "QUEUE_ID": "",
            "REPAIR_RUN_ID": "100",
            "REPLACE_LEGACY_RULESPEC_PATH": "",
            "REPLACE_RULESPEC_PATH": "us-ri/statutes/44-30-2.6.yaml",
            "SECOND_DEPENDENT_CITATION": "",
            "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
        },
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "github-output").read_text(encoding="utf-8") == (
        f"tests_only={expected_tests_only}\n"
    )


def test_fresh_v2_required_test_cases_do_not_require_a_repair_run(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = (
        next(
            step["run"]
            for step in workflow["jobs"]["encode"]["steps"]
            if step.get("name") == "Validate atomic source inputs"
        )
        .replace(
            "axiom-encode/.venv/bin/python",
            sys.executable,
        )
        .replace(
            "axiom-encode/scripts/prepare_signed_backfill.py",
            str(ROOT / "scripts/prepare_signed_backfill.py"),
        )
    )
    checkout, primary_citation, primary_path, _additions = (
        _prepare_canonical_refresh_inputs(tmp_path)
    )
    required_cases_payload = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v2",
            "source_bundle": [],
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [
                {
                    "name": "required control",
                    "period": {
                        "period_kind": "tax_year",
                        "start": "2026-01-01",
                        "end": "2026-12-31",
                    },
                    "input": {"example_input": 1},
                    "required_output": {"example_output": 1},
                }
            ],
            "target_operation": "replace",
        }
    )
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ATOMIC_SOURCE_JSON": required_cases_payload,
            "CITATION": primary_citation,
            "DEPENDENT_CITATION": "",
            "EXISTING_SIGNED_IMPORTS_JSON": "[]",
            "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
            "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": "[]",
            "QUEUE_ID": "",
            "REPAIR_RUN_ID": "",
            "REPLACE_LEGACY_RULESPEC_PATH": "",
            "REPLACE_RULESPEC_PATH": primary_path,
            "RULESPEC_CHECKOUT": str(checkout),
            "RUNNER_TEMP": str(tmp_path),
            "SECOND_DEPENDENT_CITATION": "",
            "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
        },
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "atomic_source_json",
    [
        "[]",
        json.dumps({"canonical_refresh_bundle": []}),
    ],
)
def test_explicit_target_rejects_legacy_atomic_source_envelopes_early(
    tmp_path: Path, atomic_source_json: str
) -> None:
    """An explicit path cannot acquire replacement authority from legacy input."""

    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = (
        next(
            step["run"]
            for step in workflow["jobs"]["encode"]["steps"]
            if step.get("name") == "Validate atomic source inputs"
        )
        .replace("axiom-encode/.venv/bin/python", sys.executable)
        .replace(
            "axiom-encode/scripts/prepare_signed_backfill.py",
            str(ROOT / "scripts/prepare_signed_backfill.py"),
        )
    )
    checkout, primary_citation, primary_path, _additions = (
        _prepare_canonical_refresh_inputs(tmp_path)
    )

    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ATOMIC_SOURCE_JSON": atomic_source_json,
            "CITATION": primary_citation,
            "DEPENDENT_CITATION": "",
            "EXISTING_SIGNED_IMPORTS_JSON": "[]",
            "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
            "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": "[]",
            "QUEUE_ID": "",
            "REPAIR_RUN_ID": "",
            "REPLACE_LEGACY_RULESPEC_PATH": "",
            "REPLACE_RULESPEC_PATH": primary_path,
            "RULESPEC_CHECKOUT": str(checkout),
            "RUNNER_TEMP": str(tmp_path),
            "SECOND_DEPENDENT_CITATION": "",
            "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
        },
    )

    assert completed.returncode != 0
    assert "explicit RuleSpec targets require an exact" in completed.stderr


def test_explicit_create_rejects_source_bundle_before_encoder_or_signing(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = (
        next(
            step["run"]
            for step in workflow["jobs"]["encode"]["steps"]
            if step.get("name") == "Validate atomic source inputs"
        )
        .replace("axiom-encode/.venv/bin/python", sys.executable)
        .replace(
            "axiom-encode/scripts/prepare_signed_backfill.py",
            str(ROOT / "scripts/prepare_signed_backfill.py"),
        )
    )
    checkout, primary_citation, _primary_path, _additions = (
        _prepare_canonical_refresh_inputs(tmp_path)
    )
    replacement_path = "us-ri/policies/income_tax/new_liability.yaml"
    module = "us-ri:policies/income_tax/new_liability"
    payload = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v2",
            "source_bundle": ["us-ri/statute/44-30-1"],
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [
                {
                    "name": "source-grounded creation case",
                    "period": {
                        "period_kind": "tax_year",
                        "start": "2026-01-01",
                        "end": "2026-12-31",
                    },
                    "input": {},
                    "required_output": {f"{module}#amount": 100},
                }
            ],
            "target_operation": "create",
        }
    )

    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ATOMIC_SOURCE_JSON": payload,
            "CITATION": primary_citation,
            "DEPENDENT_CITATION": "",
            "EXISTING_SIGNED_IMPORTS_JSON": "[]",
            "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
            "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": "[]",
            "QUEUE_ID": "",
            "REPAIR_RUN_ID": "",
            "REPLACE_LEGACY_RULESPEC_PATH": "",
            "REPLACE_RULESPEC_PATH": replacement_path,
            "RULESPEC_CHECKOUT": str(checkout),
            "RUNNER_TEMP": str(tmp_path),
            "SECOND_DEPENDENT_CITATION": "",
            "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
        },
    )

    assert completed.returncode != 0
    assert "land its signed source bundle before explicit RuleSpec creation" in (
        completed.stderr
    )
    assert not (tmp_path / "target-operation.txt").exists()
    assert not (tmp_path / "primary-required-test-cases.json").exists()


@pytest.mark.parametrize(
    ("operation", "target_kind", "legacy_source", "refresh", "expected_error"),
    [
        (
            "create",
            "present",
            "",
            [],
            "target_operation=create requires an absent pinned target",
        ),
        (
            "replace",
            "absent",
            "",
            [],
            "target_operation=replace requires a present pinned target",
        ),
        (
            "create",
            "none",
            "",
            [],
            "target_operation=create requires an explicit RuleSpec target",
        ),
        (
            "create",
            "present",
            "us-ri/statutes/legacy.yaml",
            [],
            "legacy replacement requires target_operation=replace",
        ),
        (
            "create",
            "absent",
            "",
            [{"citation": "us-ri/statute/44-30-2.6"}],
            "canonical refresh requires target_operation=replace",
        ),
    ],
)
def test_explicit_target_operation_mismatches_fail_before_encoder(
    tmp_path: Path,
    operation: str,
    target_kind: str,
    legacy_source: str,
    refresh: list[dict[str, str]],
    expected_error: str,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = (
        next(
            step["run"]
            for step in workflow["jobs"]["encode"]["steps"]
            if step.get("name") == "Validate atomic source inputs"
        )
        .replace("axiom-encode/.venv/bin/python", sys.executable)
        .replace(
            "axiom-encode/scripts/prepare_signed_backfill.py",
            str(ROOT / "scripts/prepare_signed_backfill.py"),
        )
    )
    checkout, primary_citation, primary_path, _additions = (
        _prepare_canonical_refresh_inputs(tmp_path)
    )
    replacement_path = {
        "present": primary_path,
        "absent": "us-ri/policies/income_tax/new_liability.yaml",
        "none": "",
    }[target_kind]
    payload = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v2",
            "source_bundle": [],
            "canonical_refresh_bundle": refresh,
            "primary_required_test_cases": [],
            "target_operation": operation,
        }
    )
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ATOMIC_SOURCE_JSON": payload,
            "CITATION": primary_citation,
            "DEPENDENT_CITATION": "",
            "EXISTING_SIGNED_IMPORTS_JSON": "[]",
            "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
            "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": "[]",
            "QUEUE_ID": "",
            "REPAIR_RUN_ID": "",
            "REPLACE_LEGACY_RULESPEC_PATH": legacy_source,
            "REPLACE_RULESPEC_PATH": replacement_path,
            "RULESPEC_CHECKOUT": str(checkout),
            "RUNNER_TEMP": str(tmp_path),
            "SECOND_DEPENDENT_CITATION": "",
            "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
        },
    )

    assert completed.returncode != 0
    assert expected_error in completed.stderr
    assert not (tmp_path / "target-operation.txt").exists()
    assert not (tmp_path / "primary-required-test-cases.json").exists()


def test_trusted_reparse_rejects_persisted_target_operation_tamper(
    tmp_path: Path,
) -> None:
    """The protected reparse must not continue with a substituted operation."""

    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    step = next(
        step
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Verify existing signed imports"
    )
    command = step["run"].split("          source_bundle_args=(", 1)[0]
    command = command.replace(
        "/opt/axiom-verification/python/bin/python", sys.executable
    ).replace(
        "axiom-encode/scripts/prepare_signed_backfill.py",
        str(ROOT / "scripts/prepare_signed_backfill.py"),
    )
    (tmp_path / "target-operation.txt").write_text("create\n", encoding="ascii")
    payload = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v2",
            "source_bundle": [],
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [],
            "target_operation": "replace",
        }
    )
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ATOMIC_SOURCE_JSON": payload,
            "REPLACE_RULESPEC_PATH": "us-ri/policies/income_tax/target.yaml",
            "RUNNER_TEMP": str(tmp_path),
        },
    )

    assert completed.returncode != 0
    assert "persisted target operation differs" in completed.stderr


def test_repair_required_test_cases_do_not_enable_canonical_refresh(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = (
        next(
            step["run"]
            for step in workflow["jobs"]["encode"]["steps"]
            if step.get("name") == "Validate atomic source inputs"
        )
        .replace("axiom-encode/.venv/bin/python", sys.executable)
        .replace(
            "axiom-encode/scripts/prepare_signed_backfill.py",
            str(ROOT / "scripts/prepare_signed_backfill.py"),
        )
    )
    checkout, primary_citation, primary_path, _additions = (
        _prepare_canonical_refresh_inputs(tmp_path)
    )
    manifest = (
        checkout / ".axiom/encoding-manifests" / Path(primary_path).with_suffix(".json")
    )
    manifest.write_text('{"schema_version":"legacy"}\n', encoding="utf-8")
    required_cases_payload = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v2",
            "source_bundle": [],
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [
                {
                    "name": "required repair control",
                    "period": {
                        "period_kind": "tax_year",
                        "start": "2026-01-01",
                        "end": "2026-12-31",
                    },
                    "input": {"example_input": 1},
                    "required_output": {"example_output": 1},
                }
            ],
            "target_operation": "replace",
        }
    )
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ATOMIC_SOURCE_JSON": required_cases_payload,
            "CITATION": primary_citation,
            "DEPENDENT_CITATION": "",
            "EXISTING_SIGNED_IMPORTS_JSON": "[]",
            "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
            "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": "[]",
            "QUEUE_ID": "",
            "REPAIR_RUN_ID": "100",
            "REPLACE_LEGACY_RULESPEC_PATH": "",
            "REPLACE_RULESPEC_PATH": primary_path,
            "RULESPEC_CHECKOUT": str(checkout),
            "RUNNER_TEMP": str(tmp_path),
            "SECOND_DEPENDENT_CITATION": "",
            "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
        },
    )

    assert completed.returncode == 0, completed.stderr


def test_repair_witness_routing_is_rechecked_in_protected_steps() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    steps = workflow["jobs"]["encode"]["steps"]
    validate = next(
        step for step in steps if step.get("name") == "Validate atomic source inputs"
    )
    verify = next(
        step for step in steps if step.get("name") == "Verify existing signed imports"
    )
    encode = next(
        step
        for step in steps
        if step.get("name") == "Encode, review, validate, and apply"
    )

    assert verify["env"]["REPAIR_TESTS_ONLY"] == (
        "${{ steps.repair_candidate.outputs.tests_only }}"
    )
    assert 'if [ "${REPAIR_TESTS_ONLY:-false}" = "true" ] &&' in verify["run"]
    assert '[ -f "$RULESPEC_CHECKOUT/$REPLACE_RULESPEC_PATH" ]; then' in verify["run"]
    assert 'if [ "$REPAIR_TESTS_ONLY" = "true" ] &&' in encode["run"]
    assert '[ -f "$RULESPEC_CHECKOUT/$REPLACE_RULESPEC_PATH" ]; then' in encode["run"]
    for step in (validate, verify, encode):
        assert '"$canonical_refresh_primary_required_test_cases_json"' in step["run"]


@pytest.mark.parametrize(
    ("provision_conclusion", "protected_runtime_state"),
    [
        ("skipped", "missing"),
        ("failure", "executable-remnant"),
        ("success", "trusted"),
    ],
)
def test_targeted_signed_reencode_packages_bounded_failure_diagnostics(
    tmp_path: Path,
    provision_conclusion: str,
    protected_runtime_state: str,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    step = next(
        item
        for item in workflow["jobs"]["encode"]["steps"]
        if item.get("name") == "Package failed re-encode diagnostics"
    )
    protected_python = tmp_path / "protected-python"
    protected_runtime_marker = tmp_path / "protected-runtime-invoked"
    if protected_runtime_state == "executable-remnant":
        protected_python.write_text("#!/bin/sh\nexit 127\n")
        protected_python.chmod(0o755)
    elif protected_runtime_state == "trusted":
        protected_python.write_text(
            "#!/bin/sh\n"
            ': > "$PROTECTED_RUNTIME_MARKER"\n'
            f'exec {shlex.quote(sys.executable)} "$@"\n'
        )
        protected_python.chmod(0o755)
    command = step["run"].replace(
        "/opt/axiom-verification/python/bin/python",
        str(protected_python),
    )
    generated = tmp_path / "generated" / "target" / "model"
    generated.mkdir(parents=True)
    (generated / "statutes").mkdir()
    (generated / "statutes/54a:4-7.test.yaml").write_text("format: rulespec/v1\n")
    (generated / "target.repair.json").write_text('{"outcome": "blocked"}\n')
    (generated / "ignored.bin").write_bytes(b"not diagnostic output")
    (tmp_path / "guard-generated.json").write_text(
        json.dumps(
            {
                "repo": "/runner/rulespec-us",
                "passed": False,
                "issues": ["waiver transition requires protected base"],
            }
        )
        + "\n"
    )
    checkpoint_stderr_raw = "checkpoint supervisor rejected the invocation\n"
    (tmp_path / "checkpoint-guard-generated.stderr.log").write_text(
        checkpoint_stderr_raw
    )
    # A failed resolver can leave the shell redirection target empty. Failure
    # packaging must preserve that resolver failure without parsing partial evidence.
    (tmp_path / "repair-candidate.json").write_text("")
    env = {
        **os.environ,
        "CITATION": "us/statute/42/1437c-1",
        "CORPUS_REF": "corpus-ref",
        "COUNTRY": "us",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_RUN_ID": "1234",
        "GITHUB_SHA": "encoder-ref",
        "ENCODE_APPLY_CONCLUSION": "failure",
        "ENCODE_APPLY_OUTCOME": "failure",
        "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": '["us/statutes/old.yaml"]',
        "RULES_ENGINE_REF": "rules-engine-ref",
        "RULESPEC_REF": "rulespec-ref",
        "RUNNER_TEMP": str(tmp_path),
        "VERIFY_GENERATED_PROVENANCE_CONCLUSION": "skipped",
        "VERIFY_GENERATED_PROVENANCE_OUTCOME": "skipped",
        "QUEUE_ID": "us-snap-all-states-2026-07",
        "QUEUE_ITEM_GENERATION_SHA256": "generation-sha",
        "QUEUE_ITEM_ID": "us/statute/42/1437c-1",
        "QUEUE_MANIFEST_SHA256": "manifest-sha",
        "PROVISION_SIGNING_SUPERVISOR_CONCLUSION": provision_conclusion,
        "PROTECTED_RUNTIME_MARKER": str(protected_runtime_marker),
    }

    subprocess.run(["bash", "-c", command], env=env, check=True)
    assert protected_runtime_marker.exists() is (protected_runtime_state == "trusted")

    archive = tmp_path / "targeted-reencode-failure.tar"
    with tarfile.open(archive, mode="r") as bundle:
        file_member_names = {
            member.name.removeprefix("./")
            for member in bundle.getmembers()
            if member.isfile()
        }
        assert file_member_names == {
            "generated/target/model/statutes/54a:4-7.test.yaml",
            "generated/target/model/target.repair.json",
            "guards/guard-generated.json",
            "guards/checkpoint-guard-generated.stderr.log",
            "metadata.json",
        }
        assert "generated/target/model/ignored.bin" not in file_member_names
        target = bundle.extractfile(
            "./generated/target/model/statutes/54a:4-7.test.yaml"
        )
        repair = bundle.extractfile("./generated/target/model/target.repair.json")
        guard = bundle.extractfile("./guards/guard-generated.json")
        checkpoint_stderr = bundle.extractfile(
            "./guards/checkpoint-guard-generated.stderr.log"
        )
        metadata_file = bundle.extractfile("./metadata.json")
        assert target is not None
        assert repair is not None
        assert guard is not None
        assert checkpoint_stderr is not None
        assert metadata_file is not None
        assert target.read() == b"format: rulespec/v1\n"
        assert json.loads(repair.read()) == {"outcome": "blocked"}
        assert json.loads(guard.read()) == {
            "repo": "/runner/rulespec-us",
            "passed": False,
            "issues": ["waiver transition requires protected base"],
        }
        assert checkpoint_stderr.read() == checkpoint_stderr_raw.encode()
        metadata = json.loads(metadata_file.read())
    assert metadata["schema"] == "axiom-encode/failed-reencode-diagnostics/v1"
    assert metadata["workflow_run_id"] == "1234"
    assert metadata["failed_steps"] == ["encode_apply"]
    assert metadata["generated_lanes"] == ["target"]
    guards_by_path = {item["path"]: item for item in metadata["guards"]}
    assert set(guards_by_path) == {
        "guards/guard-generated.json",
        "guards/checkpoint-guard-generated.stderr.log",
    }
    assert metadata["legacy_retained_successor_rulespec_paths_input"] == (
        '["us/statutes/old.yaml"]'
    )
    assert metadata["queue_id"] == "us-snap-all-states-2026-07"
    assert metadata["queue_item_generation_sha256"] == "generation-sha"
    assert metadata["queue_item_id"] == "us/statute/42/1437c-1"
    assert metadata["queue_manifest_sha256"] == "manifest-sha"
    assert metadata["step_outcomes"] == {
        "encode_apply": {"conclusion": "failure", "outcome": "failure"},
        "verify_generated_provenance": {
            "conclusion": "skipped",
            "outcome": "skipped",
        },
    }
    assert [item["path"] for item in metadata["files"]] == [
        "target/model/statutes/54a:4-7.test.yaml",
        "target/model/target.repair.json",
    ]
    target_inventory = metadata["files"][0]
    assert target_inventory["size"] == len(b"format: rulespec/v1\n")
    assert (
        target_inventory["sha256"]
        == hashlib.sha256(b"format: rulespec/v1\n").hexdigest()
    )


def test_failed_reencode_metadata_uses_consumed_identity_after_evidence_mutation(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    step = next(
        item
        for item in workflow["jobs"]["encode"]["steps"]
        if item.get("name") == "Package failed re-encode diagnostics"
    )
    command = step["run"].replace(
        "/opt/axiom-verification/python/bin/python",
        sys.executable,
    )
    (tmp_path / "generated").mkdir()
    (tmp_path / "repair-candidate.json").write_text(
        json.dumps(
            {
                "path": "statutes/42/mutated.yaml",
                "root": "/mutated",
                "rulespec_sha256": "b" * 64,
                "runner": "mutated-runner",
                "tests_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CITATION": "us/statute/42/1437c-1",
            "CORPUS_REF": "corpus-ref",
            "COUNTRY": "us",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_RUN_ID": "5678",
            "GITHUB_SHA": "encoder-ref",
            "REPAIR_CANDIDATE_CONCLUSION": "success",
            "REPAIR_CANDIDATE_OUTCOME": "success",
            "REPAIR_CANDIDATE_PATH": "statutes/42/1437c-1.yaml",
            "REPAIR_CANDIDATE_RUNNER": "openai-gpt-5.6-sol",
            "REPAIR_CANDIDATE_SOURCE_RULESPEC_REF": "f" * 40,
            "REPAIR_CANDIDATE_RULESPEC_SHA256": "d" * 64,
            "REPAIR_CANDIDATE_TESTS_SHA256": "e" * 64,
            "REPAIR_RUN_ID": "1234",
            "RULES_ENGINE_REF": "rules-engine-ref",
            "RULESPEC_REF": "rulespec-ref",
            "RUNNER_TEMP": str(tmp_path),
        },
    )

    assert completed.returncode == 0, completed.stderr
    archive = tmp_path / "targeted-reencode-failure.tar"
    with tarfile.open(archive, mode="r") as bundle:
        metadata_file = bundle.extractfile("./metadata.json")
        assert metadata_file is not None
        metadata = json.loads(metadata_file.read())
    assert metadata["repair_candidate"] == {
        "path": "statutes/42/1437c-1.yaml",
        "rulespec_sha256": "d" * 64,
        "run_id": "1234",
        "runner": "openai-gpt-5.6-sol",
        "source_rulespec_ref": "f" * 40,
        "tests_sha256": "e" * 64,
    }


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("symlink", "Could not safely open"),
        ("oversize", "safety limit"),
        ("malformed", "not valid JSON"),
        ("wrong_shape", "invalid shape"),
    ],
)
def test_targeted_signed_reencode_rejects_unsafe_guard_diagnostics(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    step = next(
        item
        for item in workflow["jobs"]["encode"]["steps"]
        if item.get("name") == "Package failed re-encode diagnostics"
    )
    command = step["run"].replace(
        "/opt/axiom-verification/python/bin/python",
        sys.executable,
    )
    (tmp_path / "generated").mkdir()
    guard = tmp_path / "guard-generated.json"
    if mutation == "symlink":
        outside = tmp_path / "outside.json"
        outside.write_text('{"repo":"outside","passed":false,"issues":[]}\n')
        guard.symlink_to(outside)
    elif mutation == "oversize":
        guard.write_bytes(b"x" * (1024 * 1024 + 1))
    elif mutation == "malformed":
        guard.write_text("{\n")
    else:
        guard.write_text(
            json.dumps(
                {
                    "repo": "/runner/rulespec-us",
                    "passed": False,
                    "issues": [],
                    "unexpected": True,
                }
            )
            + "\n"
        )

    completed = subprocess.run(
        ["bash", "-c", command],
        env={
            **os.environ,
            "CITATION": "us/statute/42/1437c-1",
            "CORPUS_REF": "corpus-ref",
            "COUNTRY": "us",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_RUN_ID": "1234",
            "GITHUB_SHA": "encoder-ref",
            "RULES_ENGINE_REF": "rules-engine-ref",
            "RULESPEC_REF": "rulespec-ref",
            "RUNNER_TEMP": str(tmp_path),
        },
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert expected_error in completed.stderr
    assert not (tmp_path / "targeted-reencode-failure.tar").exists()


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("symlink", "Could not safely open"),
        ("oversize", "safety limit"),
    ],
)
def test_targeted_signed_reencode_rejects_unsafe_raw_guard_diagnostics(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    step = next(
        item
        for item in workflow["jobs"]["encode"]["steps"]
        if item.get("name") == "Package failed re-encode diagnostics"
    )
    command = step["run"].replace(
        "/opt/axiom-verification/python/bin/python",
        sys.executable,
    )
    (tmp_path / "generated").mkdir()
    guard_log = tmp_path / "checkpoint-guard-generated.stderr.log"
    if mutation == "symlink":
        outside = tmp_path / "outside.log"
        outside.write_text("outside diagnostic\n")
        guard_log.symlink_to(outside)
    else:
        guard_log.write_bytes(b"x" * (1024 * 1024 + 1))

    completed = subprocess.run(
        ["bash", "-c", command],
        env={
            **os.environ,
            "CITATION": "us/statute/42/1437c-1",
            "CORPUS_REF": "corpus-ref",
            "COUNTRY": "us",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_RUN_ID": "1234",
            "GITHUB_SHA": "encoder-ref",
            "RULES_ENGINE_REF": "rules-engine-ref",
            "RULESPEC_REF": "rulespec-ref",
            "RUNNER_TEMP": str(tmp_path),
        },
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert expected_error in completed.stderr
    assert not (tmp_path / "targeted-reencode-failure.tar").exists()


def test_targeted_signed_reencode_preserves_checkpoint_guard_failure(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Encode, review, validate, and apply"
    )
    checkpoint = command.split("checkpoint_signed_changes() {", 1)[1].split(
        '\n}\n\nif [ "$canonical_refresh_enabled"',
        1,
    )[0]
    guard_stub = tmp_path / "guard-stub"
    guard_stub.write_text(
        "#!/bin/sh\n"
        'if [ -n "${AXIOM_ENCODE_APPLY_SIGNING_KEY:-}" ]; then\n'
        "  echo 'signing key reached direct supervisor invocation' >&2\n"
        "  exit 91\n"
        "fi\n"
        "printf '%s\\n' "
        '\'{"repo":"/runner/rulespec-us","passed":false,'
        '"issues":["checkpoint manifest mismatch"]}\'\n'
        "echo 'bounded checkpoint failure detail' >&2\n"
        "exit 7\n"
    )
    guard_stub.chmod(0o700)
    script = (
        "set -euo pipefail\n"
        "run_verifier_axiom() (\n"
        "  unset AXIOM_ENCODE_APPLY_SIGNING_KEY\n"
        '  "$GUARD_STUB" "$@"\n'
        ")\n"
        "checkpoint_signed_changes() {"
        + checkpoint
        + "\n}\ncheckpoint_signed_changes test\n"
    )

    completed = subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "AXIOM_ENCODE_APPLY_SIGNING_KEY": "test-key",
            "GITHUB_WORKSPACE": str(tmp_path),
            "GUARD_STUB": str(guard_stub),
            "RULESPEC_CHECKOUT": str(tmp_path / "rulespec-us"),
            "RUNNER_TEMP": str(tmp_path),
        },
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 7
    assert "checkpoint manifest mismatch" in completed.stderr
    assert "bounded checkpoint failure detail" in completed.stderr
    assert (
        tmp_path / "checkpoint-guard-generated.stderr.log"
    ).read_text() == "bounded checkpoint failure detail\n"
    assert json.loads((tmp_path / "checkpoint-guard-generated.json").read_text()) == {
        "repo": "/runner/rulespec-us",
        "passed": False,
        "issues": ["checkpoint manifest mismatch"],
    }


@pytest.mark.parametrize(
    "guard_stdout",
    [
        pytest.param("", id="empty"),
        pytest.param(
            '{"repo":"/runner/rulespec-us","passed":false,"issues":[]}\n{}\n',
            id="multiple-json-roots",
        ),
    ],
)
def test_targeted_signed_reencode_packages_noncontract_checkpoint_failure(
    tmp_path: Path,
    guard_stdout: str,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    steps = workflow["jobs"]["encode"]["steps"]
    apply_command = next(
        step["run"]
        for step in steps
        if step.get("name") == "Encode, review, validate, and apply"
    )
    checkpoint = apply_command.split("checkpoint_signed_changes() {", 1)[1].split(
        '\n}\n\nif [ "$canonical_refresh_enabled"',
        1,
    )[0]
    guard_stub = tmp_path / "guard-stub"
    guard_stub.write_text(
        "#!/bin/sh\n"
        'if [ -n "${AXIOM_ENCODE_APPLY_SIGNING_KEY:-}" ]; then\n'
        "  echo 'signing key reached direct supervisor invocation' >&2\n"
        "  exit 91\n"
        "fi\n"
        "printf '%s' \"${GUARD_STDOUT:-}\"\n"
        "echo 'private keys are forbidden in the signing supervisor' >&2\n"
        "exit 7\n"
    )
    guard_stub.chmod(0o700)
    checkpoint_script = (
        "set -euo pipefail\n"
        "run_verifier_axiom() (\n"
        "  unset AXIOM_ENCODE_APPLY_SIGNING_KEY\n"
        '  "$GUARD_STUB" "$@"\n'
        ")\n"
        "checkpoint_signed_changes() {"
        + checkpoint
        + "\n}\ncheckpoint_signed_changes test\n"
    )
    checkpoint_result = subprocess.run(
        ["bash", "-c", checkpoint_script],
        env={
            **os.environ,
            "AXIOM_ENCODE_APPLY_SIGNING_KEY": "test-key",
            "GITHUB_WORKSPACE": str(tmp_path),
            "GUARD_STDOUT": guard_stdout,
            "GUARD_STUB": str(guard_stub),
            "RULESPEC_CHECKOUT": str(tmp_path / "rulespec-us"),
            "RUNNER_TEMP": str(tmp_path),
        },
        capture_output=True,
        text=True,
    )

    assert checkpoint_result.returncode == 7
    assert "private keys are forbidden" in checkpoint_result.stderr
    assert not (tmp_path / "checkpoint-guard-generated.json").exists()
    assert (
        tmp_path / "checkpoint-guard-generated.stdout.log"
    ).read_text() == guard_stdout
    assert (
        tmp_path / "checkpoint-guard-generated.stderr.log"
    ).read_text() == "private keys are forbidden in the signing supervisor\n"

    (tmp_path / "generated").mkdir()
    package_command = next(
        step["run"]
        for step in steps
        if step.get("name") == "Package failed re-encode diagnostics"
    ).replace(
        "/opt/axiom-verification/python/bin/python",
        sys.executable,
    )
    package_result = subprocess.run(
        ["bash", "-c", package_command],
        env={
            **os.environ,
            "CITATION": "us-la/statute/47:294",
            "CORPUS_REF": "corpus-ref",
            "COUNTRY": "us",
            "ENCODE_APPLY_CONCLUSION": "failure",
            "ENCODE_APPLY_OUTCOME": "failure",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_RUN_ID": "31017990537",
            "GITHUB_SHA": "encoder-ref",
            "PROVISION_SIGNING_SUPERVISOR_CONCLUSION": "success",
            "RULES_ENGINE_REF": "rules-engine-ref",
            "RULESPEC_REF": "rulespec-ref",
            "RUNNER_TEMP": str(tmp_path),
        },
        capture_output=True,
        text=True,
    )

    assert package_result.returncode == 0, package_result.stderr
    with tarfile.open(tmp_path / "targeted-reencode-failure.tar", mode="r") as bundle:
        file_member_names = {
            member.name.removeprefix("./")
            for member in bundle.getmembers()
            if member.isfile()
        }
        assert file_member_names == {
            "guards/checkpoint-guard-generated.stderr.log",
            "guards/checkpoint-guard-generated.stdout.log",
            "metadata.json",
        }
        metadata_file = bundle.extractfile("./metadata.json")
        assert metadata_file is not None
        metadata = json.loads(metadata_file.read())
    assert [item["path"] for item in metadata["guards"]] == [
        "guards/checkpoint-guard-generated.stderr.log",
        "guards/checkpoint-guard-generated.stdout.log",
    ]


def test_targeted_signed_reencode_rejects_symlinked_failure_diagnostics(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    step = next(
        item
        for item in workflow["jobs"]["encode"]["steps"]
        if item.get("name") == "Package failed re-encode diagnostics"
    )
    command = step["run"].replace(
        "/opt/axiom-verification/python/bin/python",
        sys.executable,
    )
    generated = tmp_path / "generated"
    generated.mkdir()
    outside = tmp_path / "outside.yaml"
    outside.write_text("secret\n")
    (generated / "candidate.yaml").symlink_to(outside)
    env = {
        **os.environ,
        "CITATION": "us/statute/42/1437c-1",
        "CORPUS_REF": "corpus-ref",
        "COUNTRY": "us",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_RUN_ID": "1234",
        "GITHUB_SHA": "encoder-ref",
        "ENCODE_APPLY_CONCLUSION": "failure",
        "ENCODE_APPLY_OUTCOME": "failure",
        "RULES_ENGINE_REF": "rules-engine-ref",
        "RULESPEC_REF": "rulespec-ref",
        "RUNNER_TEMP": str(tmp_path),
    }

    completed = subprocess.run(
        ["bash", "-c", command],
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "non-regular file" in completed.stderr
    assert not (
        tmp_path / "targeted-reencode-failure/generated/candidate.yaml"
    ).exists()
    assert not (tmp_path / "targeted-reencode-failure.tar").exists()


def test_signed_snap_queue_dispatcher_is_bounded_and_idempotent() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/dispatch-signed-snap-queue.yml").read_text()
    )
    trigger = workflow.get("on", workflow.get(True))
    assert set(trigger) == {"workflow_dispatch"}
    inputs = trigger["workflow_dispatch"]["inputs"]
    assert inputs["dispatch"]["type"] == "boolean"
    assert inputs["dispatch"]["default"] is False
    assert inputs["limit"]["options"] == ["1", "2", "3", "4"]
    assert workflow["permissions"] == {
        "actions": "write",
        "contents": "read",
        "issues": "write",
    }
    assert workflow["concurrency"] == {
        "group": "dispatch-signed-snap-queue",
        "cancel-in-progress": False,
    }

    job = workflow["jobs"]["dispatch"]
    assert job["name"] == "Dispatch protected SNAP queue"
    assert "github.ref == 'refs/heads/main'" in job["if"]
    assert "github.run_attempt == 1" in job["if"]
    assert "environment" not in job
    steps = job["steps"]
    checkout = steps[0]
    assert checkout["with"]["persist-credentials"] is False
    assert checkout["with"]["fetch-depth"] == 0

    selection = next(
        step
        for step in steps
        if step.get("name") == "Select and validate bounded queue tranche"
    )
    selection_command = selection["run"]
    assert 'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"' in selection_command
    assert "prepare_signed_queue.py validate" in selection_command
    assert "activation_merge_run_id" in inputs
    assert "verify-merge-authorization" in selection_command
    assert '--merge-jobs "$RUNNER_TEMP/activation-merge-jobs.json"' in (
        selection_command
    )
    assert "snap-queue-merge-authorization-$ACTIVATION_MERGE_RUN_ID" in (
        selection_command
    )
    assert "git log --first-parent" in selection_command
    assert "commits/$rulespec_ref/check-runs?per_page=100" in selection_command
    assert "prepare_signed_queue.py select" in selection_command
    assert '--item-ids "$ITEM_IDS" --limit "$LIMIT"' in selection_command
    assert "prepare_signed_queue.py candidates" in selection_command
    assert 'if [ -n "$ITEM_IDS" ]' not in selection_command
    assert "prepare_signed_queue.py sha256" in selection_command

    dispatch = next(
        step
        for step in steps
        if step.get("name") == "Plan or dispatch idempotent protected runs"
    )
    assert "if" not in dispatch
    assert dispatch["env"]["DO_DISPATCH"] == "${{ inputs.dispatch }}"
    assert dispatch["env"]["GH_TOKEN"] == "${{ github.token }}"
    command = dispatch["run"]
    assert "rulespec-us/pulls?state=all" in command
    assert "prepare_signed_queue.py reconcile" in command
    assert "retry_failed" not in inputs
    assert "--retry-failed" not in command
    assert "snap-queue-reconciled.json" in command
    assert "the current tranche must be finalized" in command
    assert "any(.items[]; .dispatchable == false)" in command
    assert "targeted-signed-reencode.yml/dispatches" not in command
    assert "actions/workflows/$workflow/dispatches" in command
    assert 'ref: "main"' in command
    assert 'open_pr: "true"' in command
    assert "queue_item_generation_sha256" in command
    assert "queue_manifest_sha256" in command
    assert "queue_dispatcher_run_id: $dispatcher_run_id" in command
    assert command.index(
        'queue_id="$(jq -r \'.queue_id\' "$candidates")"'
    ) < command.index('> "$selection"')
    assert 'if [ "$selected_count" -ge "$LIMIT" ]' in command
    assert "signed-encoding-queue-plan/v1" in command
    assert 'if [ "$DO_DISPATCH" = "true" ]' in command
    assert "X-GitHub-Api-Version: 2026-03-10" in command
    assert "workflow_run_id" in command
    assert "workflow_run_attempt: 1" in command
    assert "dispatched-run-records.jsonl" in command
    assert "sleep 5" not in command
    assert 'gh issue comment "$issue"' in command

    upload = next(
        step
        for step in steps
        if step.get("name") == "Upload queue reconciliation record"
    )
    assert upload["with"]["retention-days"] == 90
    assert upload["with"]["if-no-files-found"] == "warn"
    assert "snap-queue-seed.json" in upload["with"]["path"]
    assert "snap-queue-reconciled.json" in upload["with"]["path"]
    assert "snap-queue-plan.json" in upload["with"]["path"]
    assert "dispatched-run-records.jsonl" in upload["with"]["path"]


def test_signed_snap_queue_finalizer_uses_live_fail_closed_evidence() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/finalize-signed-snap-queue.yml").read_text()
    )
    trigger = workflow.get("on", workflow.get(True))
    assert set(trigger) == {"workflow_dispatch"}
    assert workflow["permissions"] == {
        "actions": "read",
        "contents": "write",
        "pull-requests": "write",
    }
    assert workflow["concurrency"] == {
        "group": "finalize-signed-snap-queue",
        "cancel-in-progress": False,
    }
    job = workflow["jobs"]["finalize"]
    assert job["name"] == "Finalize protected SNAP queue tranche"
    assert "github.run_attempt == 1" in job["if"]
    assert "github.ref == 'refs/heads/main'" in job["if"]
    steps = job["steps"]
    checkouts = [
        step for step in steps if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert len(checkouts) == 2
    assert all(step["with"]["persist-credentials"] is False for step in checkouts)
    assert all(step["with"]["fetch-depth"] == 0 for step in checkouts)

    evidence = next(
        step for step in steps if step.get("name") == "Fetch live finalization evidence"
    )
    command = evidence["run"]
    assert 'test "$(jq -r \'.state\' "$queue")" = "paused"' in command
    assert "git/ref/heads/hard-cut/canonical-layout-us" in command
    assert "commits/$NEW_RULESPEC_REF/check-runs?per_page=100" in command
    assert '.status == "completed"' in command
    assert 'IN("success", "neutral", "skipped")' in command
    assert "rulespec-us/pulls?state=all" in command
    assert "targeted-signed-reencode.yml/runs" in command
    assert "finalize-repin" in command
    assert "finalization-target-plan" in command
    assert "targeted-reencode-$run_id" in command
    assert "jobs?filter=all&per_page=100" in command
    assert "--slurpfile jobs" in command
    assert "sha256sum -c SHA256SUMS" in command
    assert '--target-evidence "$RUNNER_TEMP/target-evidence.json"' in command
    assert '"$queue" rulespec-us' in command
    assert '--check-runs "$RUNNER_TEMP/rulespec-check-runs.json"' in command
    assert '--finalizer-head-sha "$GITHUB_SHA"' in command
    assert "--finalizer-run-url" in command
    assert command.count("\"$branch_ref\" --jq '.object.sha'") == 2

    activation = next(
        step
        for step in steps
        if step.get("name") == "Create exact queue activation commit"
    )
    activation_command = activation["run"]
    assert "uv version --bump patch" in activation_command
    assert "uv lock --offline" in activation_command
    assert 'test "$(git rev-parse refs/remotes/origin/main)" = "$GITHUB_SHA"' in (
        activation_command
    )
    assert "activation-commit.json" in activation_command
    assert "activation-changed-files.json" in activation_command
    assert "HEAD^{tree}" in activation_command
    upload = next(
        step for step in steps if step.get("name") == "Upload finalization evidence"
    )
    assert "activation-commit.json" in upload["with"]["path"]
    assert steps.index(activation) < steps.index(upload)

    publish = next(
        step
        for step in steps
        if step.get("name") == "Open reviewed queue activation pull request"
    )
    publish_command = publish["run"]
    assert "gh api --method POST" in publish_command
    assert "-F draft=true" in publish_command
    assert "required independent review-fix cycle" in publish_command
    assert steps.index(upload) < steps.index(publish)


def test_snap_queue_activation_checks_and_merge_revalidate_live_state() -> None:
    validate = yaml.safe_load(
        (ROOT / ".github/workflows/validate-snap-queue-activation.yml").read_text()
    )
    trigger = validate.get("on", validate.get(True))
    assert set(trigger) == {"pull_request"}
    assert validate["permissions"] == {"actions": "read", "contents": "read"}
    validate_command = validate["jobs"]["validate"]["steps"][1]["run"]
    assert "verify-activation" in validate_command
    assert "verify-paused-transition" in validate_command
    assert '--previous-queue "$previous_queue"' in validate_command
    assert '--expected-base-sha "$BASE_SHA"' in validate_command
    assert "--require-success false" in validate_command
    assert "verify-activation-commit" in validate_command
    assert '--finalizer-jobs "$RUNNER_TEMP/finalizer-jobs.json"' in validate_command
    assert "snap-queue-finalization-$run_id" in validate_command
    assert "git/ref/heads/hard-cut/canonical-layout-us" in validate_command
    assert 'echo "initial=$authenticate_queue" >> "$GITHUB_OUTPUT"' in validate_command
    assert ".dispatch != $previous[0].dispatch" in validate_command
    assert ".release != $previous[0].release" in validate_command

    steps = validate["jobs"]["validate"]["steps"]
    initial_checkouts = [
        step
        for step in steps
        if step.get("name", "").startswith("Checkout exact queue")
    ]
    assert len(initial_checkouts) == 3
    assert all(
        step["if"] == "${{ steps.transition.outputs.initial == 'true' }}"
        for step in initial_checkouts
    )
    assert {step["with"]["repository"] for step in initial_checkouts} == {
        "TheAxiomFoundation/axiom-corpus",
        "TheAxiomFoundation/rulespec-us",
        "TheAxiomFoundation/axiom-rules-engine",
    }
    provenance = next(
        step
        for step in steps
        if step.get("name") == "Regenerate and authenticate initial queue"
    )
    provenance_command = provenance["run"]
    assert (
        "AXIOM_CORPUS_RELEASE_PUBLIC_KEY"
        in provenance["env"]["CORPUS_RELEASE_PUBLIC_KEY"]
    )
    assert "release_objects?select=release_object" in provenance_command
    assert "build_command=build-snap-all-states" in provenance_command
    assert "build_command=build-snap" in provenance_command
    assert '"$build_command" initial-axiom-corpus initial-rulespec-us' in (
        provenance_command
    )
    assert "unsupported initial SNAP queue" in provenance_command
    assert "--state paused" in provenance_command
    assert "cmp --silent" in provenance_command
    assert "rulespec-us/git/ref/heads/hard-cut/canonical-layout-us" in (
        provenance_command
    )
    assert "initial-axiom-rules-engine merge-base --is-ancestor" in (provenance_command)
    assert "rules-engine-check-runs.json" in provenance_command

    merge = yaml.safe_load(
        (ROOT / ".github/workflows/merge-snap-queue-activation.yml").read_text()
    )
    trigger = merge.get("on", merge.get(True))
    assert set(trigger) == {"workflow_dispatch"}
    assert merge["permissions"] == {
        "actions": "read",
        "contents": "write",
        "pull-requests": "write",
    }
    assert "github.ref == 'refs/heads/main'" in merge["jobs"]["merge"]["if"]
    assert "github.run_attempt == 1" in merge["jobs"]["merge"]["if"]
    assert merge["jobs"]["merge"]["name"] == "Merge reviewed SNAP queue activation"
    steps = merge["jobs"]["merge"]["steps"]
    checkout = next(
        step for step in steps if step.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"]["ref"] == "${{ steps.pull.outputs.base_sha }}"
    assert checkout["with"]["persist-credentials"] is False
    command = next(
        step["run"]
        for step in steps
        if step.get("name") == "Revalidate evidence and merge against live RuleSpec tip"
    )
    assert "--require-success true" in command
    assert "verify-activation-commit" in command
    assert 'test "$(git rev-parse HEAD)" = "$BASE_SHA"' in command
    assert 'git fetch --no-tags origin "$HEAD_SHA"' in command
    assert "--current-changed-files" in command
    assert '--finalizer-jobs "$RUNNER_TEMP/finalizer-jobs.json"' in command
    assert "merge_workflow_run_attempt: 1" in command
    assert "commits/$HEAD_SHA/check-runs?per_page=100" in command
    assert "commits/$rulespec_ref/check-runs?per_page=100" in command
    assert '--previous-queue "$RUNNER_TEMP/previous-snap-queue.json"' in command
    assert '--expected-base-sha "$BASE_SHA"' in command
    assert "git/ref/heads/hard-cut/canonical-layout-us" in command
    assert '--match-head-commit "$HEAD_SHA"' in command
    assert "git log --first-parent" in command
    upload = next(
        step
        for step in steps
        if step.get("name") == "Upload trusted queue merge authorization"
    )
    assert (
        "snap-queue-merge-authorization-${{ github.run_id }}" == upload["with"]["name"]
    )
    assert upload["with"]["retention-days"] == 90


def _prepare_empty_signed_import_inputs(runner_temp: Path) -> None:
    (runner_temp / "canonical-refresh-bundle.json").write_text("[]\n")
    (runner_temp / "existing-signed-imports.json").write_text("[]\n")
    (runner_temp / "existing-signed-import-paths.txt").write_text("")


_TARGETED_APPLY_SHELL_HARNESS = r'''
id() {
  case "${1-}:${2-}" in
    -u:) printf '%s\n' "$AXIOM_TEST_RUNNER_UID" ;;
    -g:) printf '%s\n' "$AXIOM_TEST_RUNNER_GID" ;;
    '-u:axiom-verifier') printf '%s\n' "$AXIOM_TEST_VERIFIER_UID" ;;
    '-g:axiom-verifier') printf '%s\n' "$AXIOM_TEST_VERIFIER_GID" ;;
    '-u:axiom-signer') printf '%s\n' "$AXIOM_TEST_SIGNER_UID" ;;
    '-g:axiom-signer') printf '%s\n' "$AXIOM_TEST_SIGNER_GID" ;;
    *) echo "test id stub refused: $*" >&2; return 97 ;;
  esac
}

sudo() {
  if [[ "${1-}" == --preserve-env=* ]]; then
    test "$1" = "--preserve-env=AXIOM_ENCODE_APPLY_SIGNING_KEY,OPENAI_API_KEY,GITHUB_ACTIONS,GITHUB_REPOSITORY,GITHUB_WORKFLOW_REF,GITHUB_EVENT_NAME,GITHUB_SHA,GITHUB_RUN_ID,GITHUB_WORKSPACE" || return 97
    shift
    test "${1-}" = /opt/axiom-verification/axiom-encode-process-guardian || return 97
    shift
    test "${1-}" = run-root-controller || return 97
    shift
    test "${1-}" = -- || return 97
    shift
    test "${1-}" = /opt/axiom-verification/axiom-encode-apply-signer || return 97
    shift
    test "${1-}" = run || return 97
    shift
    local -a launcher_args=("$@")
    local -a expected_launcher=(
      --signer-uid "$AXIOM_TEST_SIGNER_UID"
      --signer-gid "$AXIOM_TEST_SIGNER_GID"
      --supervisor-uid "$AXIOM_TEST_VERIFIER_UID"
      --supervisor-gid "$AXIOM_TEST_VERIFIER_GID"
      --scope apply_ed25519
      --key-env AXIOM_ENCODE_APPLY_SIGNING_KEY
      --supervisor /opt/axiom-verification/axiom-encode-signing-supervisor
      --trusted-signing-roots /opt/axiom-verification/signing-trust-roots.json
      --trusted-python-runtime-root /opt/axiom-verification/python
      --trusted-python-import-root /opt/axiom-verification/python/lib/python3.13/site-packages
      --trusted-python-package-root /opt/axiom-verification/python/lib/python3.13/site-packages/axiom_encode
      --expected-github-repository TheAxiomFoundation/axiom-encode
      --allowed-workflow-ref TheAxiomFoundation/axiom-encode/.github/workflows/targeted-signed-reencode.yml@refs/heads/main
      --allowed-event-name workflow_dispatch
    )
    local expected
    for expected in "${expected_launcher[@]}"; do
      if [ "${1-}" != "$expected" ]; then
        echo "root-controller argv mismatch: expected $expected, got ${1-<missing>}" >&2
        return 97
      fi
      shift
    done
    test "${1-}" = -- || return 97
    shift
    test "$#" -gt 0 || return 97
    test -n "${AXIOM_ENCODE_APPLY_SIGNING_KEY:-}" || return 97
    "$SIGNER_STUB" "${launcher_args[@]}"
    return
  fi

  case "${1-}" in
    install)
      local destination="${@: -1}"
      command install -m 0600 /dev/null "$destination"
      ;;
    chown|chmod)
      return 0
      ;;
    /opt/axiom-verification/axiom-encode-process-guardian)
      shift
      local -a verifier_prefix=(
        run --uid "$AXIOM_TEST_VERIFIER_UID"
        --gid "$AXIOM_TEST_VERIFIER_GID" --
        /opt/axiom-verification/axiom-encode-signing-supervisor
        --trusted-signing-roots /opt/axiom-verification/signing-trust-roots.json
        --trusted-python-runtime-root /opt/axiom-verification/python
        --trusted-python-import-root /opt/axiom-verification/python/lib/python3.13/site-packages
        --trusted-python-package-root /opt/axiom-verification/python/lib/python3.13/site-packages/axiom_encode
        -- /opt/axiom-verification/axiom-encode
      )
      local expected
      for expected in "${verifier_prefix[@]}"; do
        if [ "${1-}" != "$expected" ]; then
          echo "verification guardian argv mismatch: expected $expected, got ${1-<missing>}" >&2
          return 97
        fi
        shift
      done
      test -z "${AXIOM_ENCODE_APPLY_SIGNING_KEY:-}" || return 97
      local operation="${1-}"
      shift
      case "$operation" in
        guard-generated)
          printf '{"repo":"%s","passed":true,"issues":[]}\n' "$RULESPEC_CHECKOUT"
          ;;
        checkpoint-signed-backfill)
          printf '{"commit":"0000000000000000000000000000000000000000","tree":"1111111111111111111111111111111111111111"}\n'
          ;;
        reconcile-retired-manifest-inventory)
          while [ "$#" -gt 0 ]; do
            if [ "$1" = --target-rulespec-path ]; then
              test "$#" -ge 2 || return 97
              if [ -n "${RECONCILIATIONS_PATH:-}" ]; then
                printf '%s\n' "$2" >> "$RECONCILIATIONS_PATH"
              fi
              return 0
            fi
            shift
          done
          echo "reconciliation omitted its target path" >&2
          return 97
          ;;
        *)
          echo "verification guardian stub refused operation: $operation" >&2
          return 97
          ;;
      esac
      ;;
    *)
      echo "test sudo stub refused: $*" >&2
      return 97
      ;;
  esac
}
'''


def _targeted_apply_shell_harness(
    command: str,
    tmp_path: Path,
    runner_temp: Path,
) -> tuple[str, dict[str, str]]:
    """Run the real workflow argv through strict, unprivileged boundary stubs."""

    trusted_python = "/opt/axiom-verification/python/bin/python -I - \\\n"
    test_python = '"$AXIOM_TEST_PYTHON" -I - \\\n'
    assert command.count(trusted_python) == 1
    command = command.replace(trusted_python, test_python, 1)

    for name in ("axiom-encode", "axiom-corpus", "axiom-rules-engine"):
        repository = tmp_path / name
        repository.mkdir(exist_ok=True)
        if not (repository / ".git").is_dir():
            subprocess.run(["git", "-C", str(repository), "init", "-q"], check=True)

    command_root = runner_temp / "_runner_file_commands"
    command_root.mkdir(mode=0o700)
    command_files: dict[str, str] = {}
    for variable, name in (
        ("GITHUB_ENV", "set_env_test"),
        ("GITHUB_PATH", "add_path_test"),
        ("GITHUB_OUTPUT", "set_output_test"),
        ("GITHUB_STEP_SUMMARY", "step_summary_test"),
    ):
        path = command_root / name
        path.write_text("", encoding="utf-8")
        path.chmod(0o600)
        command_files[variable] = str(path)

    runner_uid = os.getuid()
    runner_gid = os.getgid()
    candidate_ids = [60001, 61001, 60002, 61002]
    while runner_uid in candidate_ids or runner_gid in candidate_ids:
        candidate_ids = [value + 100 for value in candidate_ids]
    verifier_uid, verifier_gid, signer_uid, signer_gid = candidate_ids
    environment = {
        **command_files,
        "AXIOM_TEST_PYTHON": sys.executable,
        "AXIOM_TEST_RUNNER_UID": str(runner_uid),
        "AXIOM_TEST_RUNNER_GID": str(runner_gid),
        "AXIOM_TEST_VERIFIER_UID": str(verifier_uid),
        "AXIOM_TEST_VERIFIER_GID": str(verifier_gid),
        "AXIOM_TEST_SIGNER_UID": str(signer_uid),
        "AXIOM_TEST_SIGNER_GID": str(signer_gid),
    }
    return _TARGETED_APPLY_SHELL_HARNESS + "\n" + command, environment


def _prepare_canonical_refresh_inputs(
    tmp_path: Path,
) -> tuple[Path, str, str, list[dict[str, str]]]:
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    targets = [
        ("us-la/statute/47:294", "us-la/statutes/47/294.yaml"),
        ("us-la/statute/47:295", "us-la/statutes/47/295.yaml"),
        ("us-la/statute/47:297.4", "us-la/statutes/47/297/4.yaml"),
        ("us-la/statute/47:297.8", "us-la/statutes/47/297/8.yaml"),
    ]
    for citation, relative in targets:
        rule = repo / relative
        rule.parent.mkdir(parents=True, exist_ok=True)
        rule.write_text("format: rulespec/v1\nrules: []\n", encoding="utf-8")
        manifest = (
            repo / ".axiom/encoding-manifests" / Path(relative).with_suffix(".json")
        )
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "axiom-encode/applied-rulespec/v5",
                    "tool": "axiom-encode encode --apply",
                    "citation": citation,
                    "applied_files": [
                        {
                            "path": relative,
                            "sha256": hashlib.sha256(rule.read_bytes()).hexdigest(),
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "add canonical refresh fixtures",
        ],
        check=True,
    )
    primary_citation, primary_path = targets[0]
    additions = [
        {"citation": citation, "replace_rulespec_path": relative}
        for citation, relative in targets[1:]
    ]
    return repo, primary_citation, primary_path, additions


def test_gated_canonical_refresh_implementation_preserves_lane_order(
    tmp_path: Path,
) -> None:
    """Keep dormant lane mechanics covered below the workflow dispatch gate."""
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Encode, review, validate, and apply"
    )
    before_checkpoint, checkpoint_and_after = command.split(
        "checkpoint_signed_changes() {",
        1,
    )
    _checkpoint_body, after_checkpoint = checkpoint_and_after.split(
        '\n}\n\nif [ "$canonical_refresh_enabled"',
        1,
    )
    command = (
        before_checkpoint
        + "checkpoint_signed_changes() {\n"
        + '  printf \'%s\\n\' "$1" >> "$CHECKPOINTS_PATH"\n'
        + '  : > "$RUNNER_TEMP/checkpoint-guard-generated.json"\n'
        + '}\n\nif [ "$canonical_refresh_enabled"'
        + after_checkpoint
    )
    repo, primary_citation, primary_path, additions = _prepare_canonical_refresh_inputs(
        tmp_path
    )
    additions[0]["review_finding"] = "Preserve the R.S. 47:32 ownership boundary."
    additions[0]["deferred_output_contracts"] = [
        {
            "output": "us-la:statutes/47/295/a#individual_louisiana_income_tax_amount",
            "reason": "Exact source-bound missing dependency.",
        }
    ]
    mixed_output = "us-la:statutes/47/297/4#mixed_contract_output"
    additions[1]["deferred_output_contracts"] = [
        {"output": mixed_output, "reason": "Exact mixed v2 contract reason."}
    ]
    additions[1]["required_test_cases"] = [
        {
            "name": "mixed v2 addition lane",
            "period": {
                "period_kind": "tax_year",
                "start": "2025-01-01",
                "end": "2025-12-31",
            },
            "input": {},
            "required_output": {mixed_output: 0},
        }
    ]
    additions[2]["review_finding"] = "Preserve the five-year carryforward limit."
    single = (
        "us-la:statutes/47/294#input."
        "federal_return_filing_status_is_single_or_married_separate"
    )
    joint = (
        "us-la:statutes/47/294#input."
        "federal_return_filing_status_is_joint_surviving_spouse_or_head_of_household"
    )
    primary_required_test_cases = [
        {
            "name": name,
            "period": {
                "period_kind": "tax_year",
                "start": "2025-01-01",
                "end": "2025-12-31",
            },
            "input": {single: single_value, joint: joint_value},
            "required_output": {
                "us-la:statutes/47/294#standard_deduction": expected,
            },
        }
        for name, single_value, joint_value, expected in (
            (
                "2025 single individual or married separate standard deduction",
                True,
                False,
                12500,
            ),
            (
                "2025 joint surviving spouse or head of household standard deduction",
                False,
                True,
                25000,
            ),
            ("2025 no listed filing status group fails closed", False, False, 0),
            ("2025 conflicting filing status groups fail closed", True, True, 0),
        )
    ]
    normalized = parse_canonical_refresh_bundle(
        repo,
        json.dumps(additions),
        primary_citation=primary_citation,
        primary_rulespec_path=primary_path,
        primary_required_test_cases_json=json.dumps(primary_required_test_cases),
    )
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    (runner_temp / "canonical-refresh-bundle.json").write_text(
        json.dumps(normalized, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (runner_temp / "existing-signed-imports.json").write_text("[]\n")
    (runner_temp / "existing-signed-import-paths.txt").write_text("")
    command, harness_environment = _targeted_apply_shell_harness(
        command,
        tmp_path,
        runner_temp,
    )

    calls_path = tmp_path / "calls.jsonl"
    checkpoints_path = tmp_path / "checkpoints.txt"
    reconciliations_path = tmp_path / "reconciliations.txt"
    signer_stub = tmp_path / "signer-stub"
    signer_stub.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

calls_path = Path(os.environ["CALLS_PATH"])
with calls_path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
mutation_path = os.environ.get("MUTATE_AFTER_FIRST_PATH")
if mutation_path and len(calls_path.read_text(encoding="utf-8").splitlines()) == 1:
    Path(mutation_path).write_text("[]\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    signer_stub.chmod(0o700)

    environment = {
        **os.environ,
        **harness_environment,
        "AXIOM_ENCODE_APPLY_SIGNING_KEY": "test-key",
        "AXIOM_TEST_PYTHON": sys.executable,
        "CALLS_PATH": str(calls_path),
        "ATOMIC_SOURCE_JSON": json.dumps(
            {
                "schema": "axiom-encode/atomic-source-transaction/v2",
                "source_bundle": [],
                "canonical_refresh_bundle": additions,
                "primary_required_test_cases": primary_required_test_cases,
                "target_operation": "replace",
            }
        ),
        "CHECKPOINTS_PATH": str(checkpoints_path),
        "CITATION": primary_citation,
        "DEPENDENT_CITATION": "",
        "DEPENDENT_REVIEW_FINDING": "",
        "EXISTING_SIGNED_IMPORTS_JSON": "[]",
        "GITHUB_WORKSPACE": str(tmp_path),
        "REPLACE_RULESPEC_PATH": primary_path,
        "RECONCILIATIONS_PATH": str(reconciliations_path),
        "REVIEW_FINDING": "Refresh the independent Louisiana modules.",
        "RULESPEC_CHECKOUT": str(repo),
        "RULESPEC_REF": "a" * 40,
        "RUNNER_TEMP": str(runner_temp),
        "PYTHONPATH": str(ROOT / "src"),
        "SECOND_DEPENDENT_CITATION": "",
        "SECOND_DEPENDENT_REVIEW_FINDING": "",
        "SIGNER_STUB": str(signer_stub),
    }
    subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        check=True,
        env=environment,
    )

    calls = [
        json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == 4
    encode_args = [call[call.index("--") + 1 :] for call in calls]
    expected_citations = [item["citation"] for item in normalized]
    expected_paths = [item["rulespec_path"] for item in normalized]
    assert [args[-1] for args in encode_args] == expected_citations
    assert [
        args[args.index("--replace-rulespec-path") + 1] for args in encode_args
    ] == expected_paths
    assert [Path(args[args.index("--output") + 1]).name for args in encode_args] == [
        "target",
        "canonical-refresh-01",
        "canonical-refresh-02",
        "canonical-refresh-03",
    ]
    assert ["--review-findings" in args for args in encode_args] == [
        True,
        True,
        False,
        True,
    ]
    supplied_findings = [
        (
            Path(args[args.index("--review-findings") + 1]).read_text(encoding="utf-8")
            if "--review-findings" in args
            else None
        )
        for args in encode_args
    ]
    assert supplied_findings == [
        "Refresh the independent Louisiana modules.\n",
        "Preserve the R.S. 47:32 ownership boundary.\n",
        None,
        "Preserve the five-year carryforward limit.\n",
    ]
    assert [
        (
            json.loads(args[args.index("--review-contract-json") + 1])
            if "--review-contract-json" in args
            else None
        )
        for args in encode_args
    ] == [
        {
            "schema": "axiom-encode/review-contract/v3",
            "citation": primary_citation,
            "rulespec_path": primary_path,
            "required_deferred_outputs": [],
            "required_test_cases": primary_required_test_cases,
            "target_operation": "replace",
        },
        {
            "schema": "axiom-encode/review-contract/v3",
            "citation": additions[0]["citation"],
            "rulespec_path": additions[0]["replace_rulespec_path"],
            "required_deferred_outputs": additions[0]["deferred_output_contracts"],
            "required_test_cases": [],
            "target_operation": "replace",
        },
        {
            "schema": "axiom-encode/review-contract/v3",
            "citation": additions[1]["citation"],
            "rulespec_path": additions[1]["replace_rulespec_path"],
            "required_deferred_outputs": additions[1]["deferred_output_contracts"],
            "required_test_cases": additions[1]["required_test_cases"],
            "target_operation": "replace",
        },
        {
            "schema": "axiom-encode/review-contract/v3",
            "citation": additions[2]["citation"],
            "rulespec_path": additions[2]["replace_rulespec_path"],
            "required_deferred_outputs": [],
            "required_test_cases": [],
            "target_operation": "replace",
        },
    ]
    forbidden = {
        "--apply-target-only",
        "--required-import-rulespec-path",
        "--replace-legacy-rulespec-path",
        "--legacy-dependent-rulespec-path",
        "--legacy-exact-dependent-rulespec-path",
        "--legacy-retained-successor-rulespec-path",
    }
    assert all(not forbidden.intersection(args) for args in encode_args)
    assert checkpoints_path.read_text(encoding="utf-8").splitlines() == [
        f"Refresh signed canonical module for {citation}"
        for citation in expected_citations
    ]
    assert reconciliations_path.read_text(encoding="utf-8").splitlines() == (
        expected_paths
    )

    calls_path.unlink()
    checkpoints_path.unlink()
    reconciliations_path.unlink()
    (runner_temp / "encodings.db").unlink()
    future_companion = repo / str(normalized[1]["companion_path"])
    blocked = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={
            **environment,
            "MUTATE_AFTER_FIRST_PATH": str(future_companion),
        },
    )
    assert blocked.returncode != 0
    assert "companion changed before its signed refresh lane" in blocked.stderr
    assert len(calls_path.read_text(encoding="utf-8").splitlines()) == 1
    assert checkpoints_path.read_text(encoding="utf-8").splitlines() == [
        f"Refresh signed canonical module for {expected_citations[0]}"
    ]


@pytest.mark.parametrize("dependent_count", [0, 1, 2])
def test_targeted_signed_reencode_orders_target_and_dependents(
    tmp_path: Path,
    dependent_count: int,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Encode, review, validate, and apply"
    )

    calls_path = tmp_path / "calls.jsonl"
    signer_stub = tmp_path / "signer-stub"
    signer_stub.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

with Path(os.environ["CALLS_PATH"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
"""
    )
    signer_stub.chmod(0o700)
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    _prepare_empty_signed_import_inputs(runner_temp)
    command, harness_environment = _targeted_apply_shell_harness(
        command,
        tmp_path,
        runner_temp,
    )

    environment = {
        **os.environ,
        **harness_environment,
        "AXIOM_ENCODE_APPLY_SIGNING_KEY": "test-key",
        "AXIOM_TEST_PYTHON": sys.executable,
        "CALLS_PATH": str(calls_path),
        "CITATION": "us/regulation/42/435/555",
        "DEPENDENT_CITATION": "",
        "DEPENDENT_REVIEW_FINDING": "",
        "GITHUB_WORKSPACE": str(tmp_path),
        "REVIEW_FINDING": "Preserve the target source.",
        "RULESPEC_CHECKOUT": str(tmp_path / "rulespec-us"),
        "RULESPEC_REF": "a" * 40,
        "RUNNER_TEMP": str(runner_temp),
        "SECOND_DEPENDENT_CITATION": "",
        "SECOND_DEPENDENT_REVIEW_FINDING": "",
        "SIGNER_STUB": str(signer_stub),
        "ATOMIC_SOURCE_JSON": '{"canonical_refresh_bundle":[]}',
    }
    if dependent_count >= 1:
        environment.update(
            {
                "DEPENDENT_CITATION": "us/regulation/42/435/559",
                "DEPENDENT_REVIEW_FINDING": "Preserve the dependent source.",
            }
        )
    if dependent_count == 2:
        environment.update(
            {
                "SECOND_DEPENDENT_CITATION": "us/regulation/42/435/561",
                "SECOND_DEPENDENT_REVIEW_FINDING": (
                    "Preserve the second dependent source."
                ),
            }
        )

    subprocess.run(
        ["bash", "-c", command],
        check=True,
        env=environment,
    )

    calls = [
        json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == dependent_count + 1
    encode_args = [call[call.index("--") + 1 :] for call in calls]
    assert all(
        args.count("--require-complete-source-unit") == 1 for args in encode_args
    )
    assert encode_args[0][-1] == "us/regulation/42/435/555"
    assert ("--apply-target-only" in encode_args[0]) is (dependent_count > 0)
    assert (
        Path(encode_args[0][encode_args[0].index("--review-findings") + 1])
        .read_text(encoding="utf-8")
        .strip()
        == "Preserve the target source."
    )
    if dependent_count >= 1:
        assert encode_args[1][-1] == "us/regulation/42/435/559"
        assert ("--apply-target-only" in encode_args[1]) is (dependent_count == 2)
        assert (
            Path(encode_args[1][encode_args[1].index("--review-findings") + 1])
            .read_text(encoding="utf-8")
            .strip()
            == "Preserve the dependent source."
        )
    if dependent_count == 2:
        assert encode_args[2][-1] == "us/regulation/42/435/561"
        assert "--apply-target-only" not in encode_args[2]
        assert (
            Path(encode_args[2][encode_args[2].index("--review-findings") + 1])
            .read_text(encoding="utf-8")
            .strip()
            == "Preserve the second dependent source."
        )


def test_targeted_signed_reencode_composes_citation_derived_source_bundle(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Encode, review, validate, and apply"
    )
    before_checkpoint, checkpoint_and_after = command.split(
        "checkpoint_signed_changes() {",
        1,
    )
    _checkpoint_body, after_checkpoint = checkpoint_and_after.split(
        '\n}\n\nif [ "$canonical_refresh_enabled"',
        1,
    )
    command = (
        before_checkpoint
        + "checkpoint_signed_changes() {\n"
        + '  printf \'%s\\n\' "$1" >> "$CHECKPOINTS_PATH"\n'
        + '  : > "$RUNNER_TEMP/checkpoint-guard-generated.json"\n'
        + '}\n\nif [ "$canonical_refresh_enabled"'
        + after_checkpoint
    )

    calls_path = tmp_path / "calls.jsonl"
    checkpoints_path = tmp_path / "checkpoints.txt"
    signer_stub = tmp_path / "signer-stub"
    signer_stub.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

args = sys.argv[sys.argv.index("--") + 1:]
with Path(os.environ["CALLS_PATH"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
""",
        encoding="utf-8",
    )
    signer_stub.chmod(0o700)
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    _prepare_empty_signed_import_inputs(runner_temp)
    command, harness_environment = _targeted_apply_shell_harness(
        command,
        tmp_path,
        runner_temp,
    )
    primary = "us-ri/statute/44-30-2.6"
    replacement_path = "us-ri/statutes/44-30-2.6.yaml"
    checkout = tmp_path / "rulespec-us"
    canonical = checkout / replacement_path
    canonical.parent.mkdir(parents=True)
    canonical.write_text("format: rulespec/v1\nrules: []\n")
    canonical.with_name(f"{canonical.stem}.test.yaml").write_text("[]\n")
    sources = [
        "us-ri/statute/44-30-1",
        "us-ri/guidance/revenue/2026/rate-schedule",
    ]

    subprocess.run(
        ["bash", "-c", command],
        check=True,
        env={
            **os.environ,
            **harness_environment,
            "AXIOM_ENCODE_APPLY_SIGNING_KEY": "test-key",
            "AXIOM_TEST_PYTHON": sys.executable,
            "CALLS_PATH": str(calls_path),
            "CHECKPOINTS_PATH": str(checkpoints_path),
            "CITATION": primary,
            "DEPENDENT_CITATION": "",
            "DEPENDENT_REVIEW_FINDING": "",
            "GITHUB_WORKSPACE": str(tmp_path),
            "REPLACE_LEGACY_RULESPEC_PATH": "",
            "REPLACE_RULESPEC_PATH": "",
            "REPAIR_CANDIDATE_PATH": "",
            "REPAIR_CANDIDATE_ROOT": "",
            "REPAIR_CANDIDATE_RULESPEC_SHA256": "",
            "REPAIR_CANDIDATE_TESTS_SHA256": "",
            "REPAIR_TESTS_ONLY": "false",
            "REVIEW_FINDING": "Preserve the composed target semantics.",
            "RULESPEC_CHECKOUT": str(tmp_path / "rulespec-us"),
            "RULESPEC_REF": "a" * 40,
            "RUNNER_TEMP": str(runner_temp),
            "PYTHONPATH": str(ROOT / "src"),
            "SECOND_DEPENDENT_CITATION": "",
            "SECOND_DEPENDENT_REVIEW_FINDING": "",
            "SIGNER_STUB": str(signer_stub),
            "ATOMIC_SOURCE_JSON": json.dumps(
                {
                    "schema": "axiom-encode/atomic-source-transaction/v2",
                    "source_bundle": sources,
                    "canonical_refresh_bundle": [],
                    "primary_required_test_cases": [],
                    "target_operation": "replace",
                }
            ),
        },
    )

    calls = [
        json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == 3
    encode_args = [call[call.index("--") + 1 :] for call in calls]
    for index, source in enumerate(sources):
        assert encode_args[index][-1] == source
        assert "--apply-target-only" in encode_args[index]
        assert "--required-import-rulespec-path" not in encode_args[index]
        assert "--repair-candidate-root" not in encode_args[index]

    primary_args = encode_args[-1]
    assert primary_args[-1] == primary
    assert "--apply-target-only" not in primary_args
    assert "--replace-legacy-rulespec-path" not in primary_args
    assert "--repair-candidate-root" not in primary_args
    required_indexes = [
        index
        for index, value in enumerate(primary_args)
        if value == "--required-import-rulespec-path"
    ]
    assert [primary_args[index + 1] for index in required_indexes] == [
        "us-ri/statutes/44-30-1.yaml",
        "us-ri/policies/revenue/2026/rate-schedule.yaml",
    ]
    assert checkpoints_path.read_text(encoding="utf-8").splitlines() == [
        f"Add signed source module for {sources[0]}",
        f"Add signed source module for {sources[1]}",
        f"Complete signed target transaction for {primary}",
    ]
    assert canonical.is_file()


@pytest.mark.parametrize(
    ("legacy_source", "dependent_citation", "expected_error"),
    [
        (
            "us-ri/statutes/44:30-2.6.yaml",
            "",
            "source bundles require legacy replacements to merge first",
        ),
        (
            "",
            "",
            MULTILANE_REPLACE_ERROR,
        ),
    ],
)
def test_targeted_signed_reencode_rejects_source_bundle_replacements_early(
    tmp_path: Path,
    legacy_source: str,
    dependent_citation: str,
    expected_error: str,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = (
        next(
            step["run"]
            for step in workflow["jobs"]["encode"]["steps"]
            if step.get("name") == "Validate atomic source inputs"
        )
        .replace(
            "axiom-encode/.venv/bin/python",
            sys.executable,
        )
        .replace(
            "axiom-encode/scripts/prepare_signed_backfill.py",
            str(ROOT / "scripts/prepare_signed_backfill.py"),
        )
    )
    checkout = tmp_path / "rulespec-us"
    checkout.mkdir()
    existing_target = checkout / "us-ri/statutes/44-30-2.6.yaml"
    existing_target.parent.mkdir(parents=True)
    existing_target.write_text("format: rulespec/v1\nrules: []\n")
    if legacy_source:
        legacy_target = checkout / legacy_source
        legacy_target.parent.mkdir(parents=True, exist_ok=True)
        legacy_target.write_text("format: rulespec/v1\nrules: []\n")

    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CITATION": "us-ri/statute/44-30-2.6",
            "DEPENDENT_CITATION": dependent_citation,
            "EXISTING_SIGNED_IMPORTS_JSON": "[]",
            "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
            "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": "[]",
            "QUEUE_ID": "",
            "REPLACE_LEGACY_RULESPEC_PATH": legacy_source,
            "REPLACE_RULESPEC_PATH": "us-ri/statutes/44-30-2.6.yaml",
            "RULESPEC_CHECKOUT": str(checkout),
            "RUNNER_TEMP": str(tmp_path),
            "SECOND_DEPENDENT_CITATION": "",
            "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
            "ATOMIC_SOURCE_JSON": json.dumps(
                {
                    "schema": "axiom-encode/atomic-source-transaction/v2",
                    "source_bundle": ["us-ri/statute/44-30-1"],
                    "canonical_refresh_bundle": [],
                    "primary_required_test_cases": [],
                    "target_operation": "replace",
                }
            ),
        },
    )

    assert completed.returncode != 0
    assert expected_error in completed.stderr
    assert not (tmp_path / "target-operation.txt").exists()
    assert not (tmp_path / "primary-required-test-cases.json").exists()


def test_targeted_signed_reencode_rejects_canonical_refresh_replacement_early(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = (
        next(
            step["run"]
            for step in workflow["jobs"]["encode"]["steps"]
            if step.get("name") == "Validate atomic source inputs"
        )
        .replace("axiom-encode/.venv/bin/python", sys.executable)
        .replace(
            "axiom-encode/scripts/prepare_signed_backfill.py",
            str(ROOT / "scripts/prepare_signed_backfill.py"),
        )
    )
    checkout, primary_citation, primary_path, additions = (
        _prepare_canonical_refresh_inputs(tmp_path)
    )
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ATOMIC_SOURCE_JSON": json.dumps(
                {
                    "schema": "axiom-encode/atomic-source-transaction/v2",
                    "source_bundle": [],
                    "canonical_refresh_bundle": additions,
                    "primary_required_test_cases": [],
                    "target_operation": "replace",
                }
            ),
            "CITATION": primary_citation,
            "DEPENDENT_CITATION": "",
            "EXISTING_SIGNED_IMPORTS_JSON": "[]",
            "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
            "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": "[]",
            "QUEUE_ID": "",
            "REPAIR_RUN_ID": "",
            "REPLACE_LEGACY_RULESPEC_PATH": "",
            "REPLACE_RULESPEC_PATH": primary_path,
            "RULESPEC_CHECKOUT": str(checkout),
            "RUNNER_TEMP": str(tmp_path),
            "SECOND_DEPENDENT_CITATION": "",
            "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
        },
    )

    assert completed.returncode != 0
    assert MULTILANE_REPLACE_ERROR in completed.stderr
    assert not (tmp_path / "target-operation.txt").exists()
    assert not (tmp_path / "primary-required-test-cases.json").exists()


def test_targeted_signed_reencode_rejects_existing_source_add_targets_early(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = (
        next(
            step["run"]
            for step in workflow["jobs"]["encode"]["steps"]
            if step.get("name") == "Validate atomic source inputs"
        )
        .replace("axiom-encode/.venv/bin/python", sys.executable)
        .replace(
            "axiom-encode/scripts/prepare_signed_backfill.py",
            str(ROOT / "scripts/prepare_signed_backfill.py"),
        )
    )
    checkout = tmp_path / "rulespec-us"
    for relative in (
        "us-la/statutes/47/294.yaml",
        "us-la/statutes/47/295.yaml",
        "us-la/statutes/47/32.yaml",
        "us-la/statutes/47:32.yaml",
    ):
        target = checkout / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("format: rulespec/v1\nrules: []\n", encoding="utf-8")

    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ATOMIC_SOURCE_JSON": '["us-la/statute/47:295"]',
            "CITATION": "us-la/statute/47:294",
            "DEPENDENT_CITATION": "",
            "EXISTING_SIGNED_IMPORTS_JSON": "[]",
            "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
            "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": "[]",
            "QUEUE_ID": "",
            "REPAIR_RUN_ID": "",
            "REPLACE_LEGACY_RULESPEC_PATH": "",
            "REPLACE_RULESPEC_PATH": "",
            "RULESPEC_CHECKOUT": str(checkout),
            "RUNNER_TEMP": str(tmp_path),
            "SECOND_DEPENDENT_CITATION": "",
            "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
        },
    )

    assert completed.returncode != 0
    assert "existing modules must use canonical_refresh_bundle" in completed.stderr
    assert "us-la/statutes/47/294.yaml" in completed.stderr
    assert "us-la/statutes/47/295.yaml" in completed.stderr


def test_targeted_signed_reencode_direct_replace_reuses_import_before_dependent(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Encode, review, validate, and apply"
    )

    calls_path = tmp_path / "calls.jsonl"
    signer_stub = tmp_path / "signer-stub"
    signer_stub.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

with Path(os.environ["CALLS_PATH"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
""",
        encoding="utf-8",
    )
    signer_stub.chmod(0o700)

    rulespec_repo = tmp_path / "rulespec-us"
    existing_path = "us-ri/statutes/44-30-1.yaml"
    replacement_path = "us-ri/policies/income_tax/direct_target.yaml"
    manifest_path = (
        rulespec_repo / ".axiom/encoding-manifests/us-ri/statutes/44-30-1.json"
    )
    rule = rulespec_repo / existing_path
    rule.parent.mkdir(parents=True)
    rule.write_text("format: rulespec/v1\nrules: []\n")
    replacement = rulespec_repo / replacement_path
    replacement.parent.mkdir(parents=True)
    replacement.write_text("format: rulespec/v1\nrules: []\n")
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps({"schema_version": "axiom-encode/applied-rulespec/v5"}) + "\n"
    )
    subprocess.run(["git", "-C", str(rulespec_repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(rulespec_repo), "add", "."], check=True)

    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    normalized_existing = json.dumps([existing_path], separators=(",", ":"))
    (runner_temp / "existing-signed-imports.json").write_text(
        normalized_existing + "\n"
    )
    (runner_temp / "existing-signed-import-paths.txt").write_text(existing_path + "\n")
    (runner_temp / "canonical-refresh-bundle.json").write_text("[]\n")
    command, harness_environment = _targeted_apply_shell_harness(
        command,
        tmp_path,
        runner_temp,
    )

    subprocess.run(
        ["bash", "-c", command],
        check=True,
        env={
            **os.environ,
            **harness_environment,
            "AXIOM_ENCODE_APPLY_SIGNING_KEY": "test-key",
            "AXIOM_TEST_PYTHON": sys.executable,
            "CALLS_PATH": str(calls_path),
            "CITATION": "us-ri/statute/44-30-2.6",
            "DEPENDENT_CITATION": "us-ri/statute/44-30-2.7",
            "DEPENDENT_REVIEW_FINDING": "Preserve the dependent semantics.",
            "EXISTING_SIGNED_IMPORTS_JSON": normalized_existing,
            "GITHUB_WORKSPACE": str(tmp_path),
            "REPLACE_LEGACY_RULESPEC_PATH": "",
            "REPLACE_RULESPEC_PATH": replacement_path,
            "REVIEW_FINDING": "Preserve direct composition semantics.",
            "RULESPEC_CHECKOUT": str(rulespec_repo),
            "RULESPEC_REF": "a" * 40,
            "RUNNER_TEMP": str(runner_temp),
            "PYTHONPATH": str(ROOT / "src"),
            "SECOND_DEPENDENT_CITATION": "",
            "SECOND_DEPENDENT_REVIEW_FINDING": "",
            "SIGNER_STUB": str(signer_stub),
            "ATOMIC_SOURCE_JSON": json.dumps(
                {
                    "schema": "axiom-encode/atomic-source-transaction/v2",
                    "source_bundle": [],
                    "canonical_refresh_bundle": [],
                    "primary_required_test_cases": [],
                    "target_operation": "replace",
                }
            ),
        },
    )

    calls = [
        json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == 2
    target_args = calls[0][calls[0].index("--") + 1 :]
    dependent_args = calls[1][calls[1].index("--") + 1 :]
    assert target_args[-1] == "us-ri/statute/44-30-2.6"
    assert target_args[target_args.index("--replace-rulespec-path") + 1] == (
        replacement_path
    )
    assert "--apply-target-only" in target_args
    required_index = target_args.index("--required-import-rulespec-path")
    assert target_args[required_index + 1] == existing_path
    assert dependent_args[-1] == "us-ri/statute/44-30-2.7"
    assert "--apply-target-only" not in dependent_args


@pytest.mark.parametrize("with_dependent", [False, True])
def test_targeted_signed_reencode_runs_replacement_target(
    tmp_path: Path,
    with_dependent: bool,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Encode, review, validate, and apply"
    )
    calls_path = tmp_path / "calls.jsonl"
    signer_stub = tmp_path / "signer-stub"
    signer_stub.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

with Path(os.environ["CALLS_PATH"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
"""
    )
    signer_stub.chmod(0o700)
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    _prepare_empty_signed_import_inputs(runner_temp)
    command, harness_environment = _targeted_apply_shell_harness(
        command,
        tmp_path,
        runner_temp,
    )
    replacement_path = "us-nc/policies/income_tax/pilot_liability_pipeline.yaml"
    rulespec_checkout = tmp_path / "rulespec-us"
    rulespec_checkout.mkdir()
    subprocess.run(["git", "-C", str(rulespec_checkout), "init", "-q"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(rulespec_checkout),
            "config",
            "user.email",
            "test@example.com",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(rulespec_checkout), "config", "user.name", "Test"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(rulespec_checkout),
            "commit",
            "--allow-empty",
            "-qm",
            "protected base",
        ],
        check=True,
    )
    if with_dependent:
        existing_target = rulespec_checkout / replacement_path
        existing_target.parent.mkdir(parents=True, exist_ok=True)
        existing_target.write_text("format: rulespec/v1\n", encoding="utf-8")
        legacy_source = rulespec_checkout / (
            "us-nc/policies/income_tax/PILOT_LIABILITY_PIPELINE.yaml"
        )
        legacy_source.write_text("format: rulespec/v1\n", encoding="utf-8")
    primary_required_test_cases = [
        {
            "name": "source-grounded policy creation case",
            "period": {
                "period_kind": "tax_year",
                "start": "2026-01-01",
                "end": "2026-12-31",
            },
            "input": {"amount": 100},
            "required_output": {
                "us-nc:policies/income_tax/pilot_liability_pipeline#amount": 100
            },
        }
    ]
    atomic_source_json = json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v2",
            "source_bundle": [],
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": primary_required_test_cases,
            "target_operation": "replace" if with_dependent else "create",
        }
    )
    dependent_citation = "us-nc/statute/105/105-153.5" if with_dependent else ""
    dependent_finding = (
        "Preserve the resident pipeline semantics." if with_dependent else ""
    )

    environment = {
        **os.environ,
        **harness_environment,
        "AXIOM_ENCODE_APPLY_SIGNING_KEY": "test-key",
        "AXIOM_TEST_PYTHON": sys.executable,
        "CALLS_PATH": str(calls_path),
        "CITATION": "us-nc/statute/105/105-153.7",
        "DEPENDENT_CITATION": dependent_citation,
        "DEPENDENT_REVIEW_FINDING": dependent_finding,
        "GITHUB_WORKSPACE": str(tmp_path),
        "REPLACE_LEGACY_RULESPEC_PATH": (
            "us-nc/policies/income_tax/PILOT_LIABILITY_PIPELINE.yaml"
            if with_dependent
            else ""
        ),
        "REPLACE_RULESPEC_PATH": replacement_path,
        "REVIEW_FINDING": "Preserve all supported existing semantics.",
        "RULESPEC_CHECKOUT": str(rulespec_checkout),
        "RULESPEC_REF": "a" * 40,
        "RUNNER_TEMP": str(runner_temp),
        "SECOND_DEPENDENT_CITATION": "",
        "SECOND_DEPENDENT_REVIEW_FINDING": "",
        "SIGNER_STUB": str(signer_stub),
        "ATOMIC_SOURCE_JSON": atomic_source_json,
    }
    subprocess.run(
        ["bash", "-c", command],
        check=True,
        env=environment,
    )

    calls = [
        json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == (2 if with_dependent else 1)
    encode_args = [call[call.index("--") + 1 :] for call in calls]
    assert ("--apply-target-only" in encode_args[0]) is with_dependent
    target_option = (
        "--replace-rulespec-path" if with_dependent else "--create-rulespec-path"
    )
    assert encode_args[0][encode_args[0].index(target_option) + 1] == replacement_path
    if not with_dependent:
        assert "--replace-rulespec-path" not in encode_args[0]
        review_contract = json.loads(
            encode_args[0][encode_args[0].index("--review-contract-json") + 1]
        )
        assert review_contract == {
            "schema": "axiom-encode/review-contract/v3",
            "citation": "us-nc/statute/105/105-153.7",
            "rulespec_path": replacement_path,
            "required_deferred_outputs": [],
            "required_test_cases": primary_required_test_cases,
            "target_operation": "create",
        }
    assert encode_args[0][-1] == "us-nc/statute/105/105-153.7"
    if with_dependent:
        assert "--replace-legacy-rulespec-path" in encode_args[0]
        assert "--legacy-dependent-rulespec-path" in encode_args[0]
        assert "--replace-rulespec-path" not in encode_args[1]
        assert "--apply-target-only" not in encode_args[1]
        assert encode_args[1][-1] == dependent_citation

    blocked = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env={**environment, "QUEUE_ID": "us-snap-all-states-2026-07"},
    )
    assert blocked.returncode != 0
    assert (
        "queue-authorized re-encodes cannot override the RuleSpec target path"
        in blocked.stderr
    )


def test_targeted_signed_reencode_passes_exact_legacy_dependents_atomically(
    tmp_path: Path,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Encode, review, validate, and apply"
    )
    calls_path = tmp_path / "calls.jsonl"
    signer_stub = tmp_path / "signer-stub"
    signer_stub.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

with Path(os.environ["CALLS_PATH"]).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(sys.argv[1:]) + "\\n")
"""
    )
    signer_stub.chmod(0o700)
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    _prepare_empty_signed_import_inputs(runner_temp)
    command, harness_environment = _targeted_apply_shell_harness(
        command,
        tmp_path,
        runner_temp,
    )
    source = "us-la/statutes/47:32.yaml"
    destination = "us-la/statutes/47/32.yaml"
    exact_dependents = [
        "us-la/policies/income_tax/2026_resident_core.yaml",
        "us-la/policies/income_tax/pilot_liability_pipeline.yaml",
    ]
    retained_successors = [
        "us-la/statutes/47:294.yaml",
        "us-la/statutes/47:295.yaml",
        "us-la/statutes/47:297/4.yaml",
        "us-la/statutes/47:297/8.yaml",
    ]
    rulespec_checkout = tmp_path / "rulespec-us"
    target = rulespec_checkout / destination
    target.parent.mkdir(parents=True)
    target.write_text("format: rulespec/v1\nrules: []\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(rulespec_checkout), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(rulespec_checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(rulespec_checkout),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    environment = {
        **os.environ,
        **harness_environment,
        "AXIOM_ENCODE_APPLY_SIGNING_KEY": "test-key",
        "AXIOM_TEST_PYTHON": sys.executable,
        "CALLS_PATH": str(calls_path),
        "CITATION": "us-la/statute/47:32",
        "DEPENDENT_CITATION": "",
        "DEPENDENT_REVIEW_FINDING": "",
        "GITHUB_WORKSPACE": str(tmp_path),
        "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": exact_dependents[0],
        "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": json.dumps(
            retained_successors
        ),
        "REPLACE_LEGACY_RULESPEC_PATH": source,
        "REPLACE_RULESPEC_PATH": destination,
        "REVIEW_FINDING": "Preserve all supported existing semantics.",
        "RULESPEC_CHECKOUT": str(rulespec_checkout),
        "RULESPEC_REF": "a" * 40,
        "RUNNER_TEMP": str(runner_temp),
        "SECOND_DEPENDENT_CITATION": "",
        "SECOND_DEPENDENT_REVIEW_FINDING": "",
        "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": exact_dependents[1],
        "SIGNER_STUB": str(signer_stub),
        "ATOMIC_SOURCE_JSON": json.dumps(
            {
                "schema": "axiom-encode/atomic-source-transaction/v2",
                "source_bundle": [],
                "canonical_refresh_bundle": [],
                "primary_required_test_cases": [],
                "target_operation": "replace",
            }
        ),
    }

    subprocess.run(["bash", "-c", command], check=True, env=environment)

    calls = [
        json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(calls) == 1
    encode_args = calls[0][calls[0].index("--") + 1 :]
    assert encode_args[encode_args.index("--replace-rulespec-path") + 1] == destination
    assert (
        encode_args[encode_args.index("--replace-legacy-rulespec-path") + 1] == source
    )
    exact_indexes = [
        index
        for index, value in enumerate(encode_args)
        if value == "--legacy-exact-dependent-rulespec-path"
    ]
    assert [encode_args[index + 1] for index in exact_indexes] == exact_dependents
    retained_indexes = [
        index
        for index, value in enumerate(encode_args)
        if value == "--legacy-retained-successor-rulespec-path"
    ]
    assert [encode_args[index + 1] for index in retained_indexes] == (
        retained_successors
    )
    assert "--apply-target-only" not in encode_args


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (
            {
                "SECOND_DEPENDENT_CITATION": "us/regulation/42/435/561",
                "SECOND_DEPENDENT_REVIEW_FINDING": "Preserve the second source.",
            },
            "first dependent citation is required before a second dependent",
        ),
        (
            {
                "DEPENDENT_CITATION": "us/regulation/42/435/559",
                "DEPENDENT_REVIEW_FINDING": "Preserve the dependent source.",
                "SECOND_DEPENDENT_CITATION": "us/regulation/42/435/559",
                "SECOND_DEPENDENT_REVIEW_FINDING": "Preserve the second source.",
            },
            "second dependent citation must be unique",
        ),
        (
            {
                "DEPENDENT_CITATION": "us/regulation/42/435/559",
                "DEPENDENT_REVIEW_FINDING": "Preserve the dependent source.",
                "SECOND_DEPENDENT_REVIEW_FINDING": "Preserve the second source.",
            },
            "second dependent citation is required with its review finding",
        ),
        (
            {
                "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": (
                    "us-la/policies/income_tax/pilot_liability_pipeline.yaml"
                ),
            },
            "exact legacy dependents require a legacy source replacement",
        ),
        (
            {
                "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": (
                    "us-la/policies/income_tax/2026_resident_core.yaml"
                ),
                "REPLACE_LEGACY_RULESPEC_PATH": "us-la/statutes/47:32.yaml",
                "REPLACE_RULESPEC_PATH": "us-la/statutes/47/32.yaml",
                "DEPENDENT_CITATION": "us-la/statute/47:294",
                "DEPENDENT_REVIEW_FINDING": "Preserve dependent semantics.",
            },
            "exact and model-regenerated dependent modes cannot be combined",
        ),
    ],
)
def test_targeted_signed_reencode_rejects_invalid_second_dependent_inputs(
    tmp_path: Path,
    overrides: dict[str, str],
    error: str,
) -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Encode, review, validate, and apply"
    )
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    _prepare_empty_signed_import_inputs(runner_temp)
    command, harness_environment = _targeted_apply_shell_harness(
        command,
        tmp_path,
        runner_temp,
    )
    environment = {
        **os.environ,
        **harness_environment,
        "AXIOM_ENCODE_APPLY_SIGNING_KEY": "test-key",
        "CITATION": "us/regulation/42/435/555",
        "DEPENDENT_CITATION": "",
        "DEPENDENT_REVIEW_FINDING": "",
        "GITHUB_WORKSPACE": str(tmp_path),
        "REVIEW_FINDING": "Preserve the target source.",
        "RULESPEC_CHECKOUT": str(tmp_path / "rulespec-us"),
        "RULESPEC_REF": "a" * 40,
        "RUNNER_TEMP": str(runner_temp),
        "SECOND_DEPENDENT_CITATION": "",
        "SECOND_DEPENDENT_REVIEW_FINDING": "",
        "SIGNER_STUB": "/usr/bin/false",
        "ATOMIC_SOURCE_JSON": "[]",
        **overrides,
    }

    completed = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert error in completed.stderr


def test_targeted_review_finding_temp_file_is_valid_context(tmp_path: Path) -> None:
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    _prepare_empty_signed_import_inputs(runner_temp)
    policy_root = tmp_path / "rulespec-us" / "us"
    policy_root.mkdir(parents=True)
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'path="$(mktemp "$RUNNER_TEMP/'
            'axiom-targeted-review-finding.XXXXXX.txt")"; '
            "printf '%s\\n' 'Preserve the supported provision.' > \"$path\"; "
            "printf '%s' \"$path\"",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "RUNNER_TEMP": str(runner_temp)},
    )

    finding_path = Path(completed.stdout)
    assert finding_path.parent == runner_temp
    assert finding_path.suffix == ".txt"
    assert validate_explicit_context_file(finding_path, policy_root) == finding_path


def _copy_targeted_atomic_inputs(runner_temp: Path, artifact: Path) -> None:
    for name in (
        "source-bundle.json",
        "canonical-refresh-bundle.json",
        "existing-signed-imports.json",
        "primary-required-test-cases.json",
        "target-operation.txt",
    ):
        source = runner_temp / name
        if name == "existing-signed-imports.json" and not source.exists():
            (artifact / name).write_text("[]\n", encoding="utf-8")
        else:
            shutil.copyfile(source, artifact / name)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (None, None),
        (
            "persisted-only-refresh",
            "persisted canonical refresh bundle differs from atomic source input",
        ),
        ("post-copy-original-controls", None),
        (
            "review-contract-cases",
            "context manifest does not bind the normalized review contract",
        ),
        (
            "context-operation-flip",
            "context manifest does not bind the normalized review contract",
        ),
        (
            "context-operation-removal",
            "context manifest does not bind the normalized review contract",
        ),
        (
            "manifest-operation-flip",
            "signed apply manifest target operation differs from the normalized review contract",
        ),
        (
            "manifest-operation-removal",
            "signed apply manifest target operation differs from the normalized review contract",
        ),
        (
            "creation-target-removal",
            "signed apply manifest target operation differs from the normalized review contract",
        ),
        (
            "creation-base-commit",
            "signed apply manifest target operation differs from the normalized review contract",
        ),
        (
            "creation-base-tree",
            "signed apply manifest target operation differs from the normalized review contract",
        ),
        (
            "creation-primary",
            "signed apply manifest target operation differs from the normalized review contract",
        ),
        (
            "creation-companion",
            "signed apply manifest target operation differs from the normalized review contract",
        ),
        (
            "creation-canonical-manifest",
            "signed apply manifest target operation differs from the normalized review contract",
        ),
        (
            "creation-orphan-manifest",
            "signed apply manifest target operation differs from the normalized review contract",
        ),
        (
            "wrong-manifest-path",
            "applied manifest target operation differs from the signed atomic-source transaction",
        ),
        (
            "coherent-operation-rewrite",
            "persisted target operation differs from atomic source input",
        ),
        (
            "coherent-required-cases-rewrite",
            "persisted primary required test cases differ from atomic source input",
        ),
        (
            "coherent-required-cases-float",
            "persisted primary required test cases differ from atomic source input",
        ),
        (
            "coherent-required-cases-bool-int",
            "persisted primary required test cases differ from atomic source input",
        ),
        (
            "coherent-source-bundle-rewrite",
            "persisted source bundle differs from atomic source input",
        ),
        (
            "sealed-source-bundle-replace",
            MULTILANE_REPLACE_ERROR,
        ),
        (
            "coherent-canonical-refresh-rewrite",
            "persisted canonical refresh bundle differs from atomic source input",
        ),
        (
            "duplicate-context-operation",
            "signed context manifest is not unambiguous JSON",
        ),
        (
            "duplicate-manifest-operation",
            "changed apply manifest is not unambiguous JSON",
        ),
        (
            "duplicate-creation-field",
            "changed apply manifest is not unambiguous JSON",
        ),
    ],
)
def test_targeted_artifact_packages_signed_review_context(
    tmp_path: Path,
    mutation: str | None,
    expected_error: str | None,
) -> None:
    script = _targeted_package_script()

    rulespec = tmp_path / "rulespec-nz"
    rulespec.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=rulespec, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=rulespec,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=rulespec, check=True)
    base = rulespec / "README.md"
    base.write_text("fixture\n")
    subprocess.run(["git", "add", "README.md"], cwd=rulespec, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=rulespec, check=True)
    rulespec_ref = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=rulespec, text=True
    ).strip()
    rulespec_tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=rulespec, text=True
    ).strip()
    (tmp_path / "source-bundle.json").write_text("[]\n", encoding="utf-8")
    (tmp_path / "canonical-refresh-bundle.json").write_text("[]\n", encoding="utf-8")
    (tmp_path / "primary-required-test-cases.json").write_text("[]\n", encoding="utf-8")
    (tmp_path / "target-operation.txt").write_text("replace\n", encoding="ascii")

    citation = "us-la/statute/47:294"
    rulespec_path = "us-la/policies/income_tax/standard_deduction.yaml"
    required_output_ref = (
        "us-la:policies/income_tax/standard_deduction#standard_deduction"
    )
    required_cases = [
        {
            "name": "source-derived control",
            "period": {
                "period_kind": "tax_year",
                "start": "2026-01-01",
                "end": "2026-12-31",
            },
            "input": {
                "filing_status": "single",
                "material_participation": True,
            },
            "required_output": {required_output_ref: 12500},
        }
    ]
    (tmp_path / "primary-required-test-cases.json").write_text(
        json.dumps(required_cases), encoding="utf-8"
    )
    if mutation == "coherent-source-bundle-rewrite":
        (tmp_path / "source-bundle.json").write_text(
            json.dumps(["us-la/statute/47:295"]),
            encoding="utf-8",
        )
    elif mutation in {
        "coherent-canonical-refresh-rewrite",
        "persisted-only-refresh",
    }:

        def refresh_item(
            item_citation: str,
            item_rulespec_path: str,
            item_cases: list[dict[str, object]],
        ) -> dict[str, object]:
            return {
                "citation": item_citation,
                "rulespec_path": item_rulespec_path,
                "rulespec_sha256": "a" * 64,
                "companion_path": item_rulespec_path.replace(".yaml", ".test.yaml"),
                "companion_sha256": "b" * 64,
                "manifest_path": (
                    ".axiom/encoding-manifests/"
                    + item_rulespec_path.replace(".yaml", ".json")
                ),
                "manifest_sha256": "c" * 64,
                "review_finding": None,
                "deferred_output_contracts": [],
                "required_test_cases": item_cases,
            }

        (tmp_path / "canonical-refresh-bundle.json").write_text(
            json.dumps(
                [
                    refresh_item(citation, rulespec_path, required_cases),
                    refresh_item(
                        "us-la/statute/47:295",
                        "us-la/statutes/47/295.yaml",
                        [],
                    ),
                ]
            ),
            encoding="utf-8",
        )
    target_operation = (
        "replace" if mutation == "sealed-source-bundle-replace" else "create"
    )
    source_bundle = (
        ["us-la/statute/47:295"]
        if mutation == "sealed-source-bundle-replace"
        else []
    )
    if mutation != "coherent-source-bundle-rewrite":
        (tmp_path / "source-bundle.json").write_text(
            json.dumps(source_bundle) + "\n",
            encoding="utf-8",
        )
    (tmp_path / "target-operation.txt").write_text(
        f"{target_operation}\n", encoding="ascii"
    )
    review_content = "Preserve every supported provision.\n"
    context_payload = {
        "citation": citation,
        "review_contract": {
            "schema": "axiom-encode/review-contract/v3",
            "citation": citation,
            "rulespec_path": rulespec_path,
            "required_deferred_outputs": [],
            "required_test_cases": required_cases,
            "target_operation": target_operation,
        },
        "review_findings_files": [
            {
                "content": review_content,
                "sha256": hashlib.sha256(review_content.encode()).hexdigest(),
            }
        ],
    }
    if mutation == "review-contract-cases":
        context_payload["review_contract"]["required_test_cases"] = []
    elif mutation == "context-operation-flip":
        context_payload["review_contract"]["target_operation"] = "replace"
    elif mutation == "context-operation-removal":
        del context_payload["review_contract"]["target_operation"]
    elif mutation == "coherent-operation-rewrite":
        context_payload["review_contract"]["target_operation"] = "replace"
        (tmp_path / "target-operation.txt").write_text("replace\n", encoding="ascii")
    elif mutation in {
        "coherent-required-cases-rewrite",
        "coherent-required-cases-float",
        "coherent-required-cases-bool-int",
    }:
        rewritten_cases = json.loads(json.dumps(required_cases))
        if mutation == "coherent-required-cases-rewrite":
            rewritten_cases[0]["required_output"][required_output_ref] = 0
        elif mutation == "coherent-required-cases-float":
            rewritten_cases[0]["required_output"][required_output_ref] = 12500.0
        else:
            rewritten_cases[0]["input"]["material_participation"] = 1
        context_payload["review_contract"]["required_test_cases"] = rewritten_cases
        (tmp_path / "primary-required-test-cases.json").write_text(
            json.dumps(rewritten_cases), encoding="utf-8"
        )
    context_bytes = json.dumps(context_payload, sort_keys=True).encode()
    if mutation == "duplicate-context-operation":
        context_bytes = context_bytes.replace(
            b'"target_operation": "create"',
            b'"target_operation": "replace", "target_operation": "create"',
            1,
        )
    context_path = tmp_path / "generated" / "target" / "context-manifest.json"
    context_path.parent.mkdir(parents=True)
    context_path.write_bytes(context_bytes)
    applied_manifest = {
        "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
        "citation": citation,
        "applied_files": [{"path": rulespec_path}],
        "context_manifest_file": str(context_path),
        "context_manifest_sha256": hashlib.sha256(context_bytes).hexdigest(),
        "target_operation": target_operation,
    }
    if target_operation == "create":
        applied_manifest["creation_target"] = {
            "base_commit": rulespec_ref,
            "base_tree": rulespec_tree,
            "primary": rulespec_path,
            "companion": rulespec_path.replace(".yaml", ".test.yaml"),
            "canonical_manifest": ".axiom/encoding-manifests/"
            + rulespec_path.replace(".yaml", ".json"),
            "orphan_manifest": None,
        }
    if mutation == "manifest-operation-flip":
        applied_manifest["target_operation"] = "replace"
    elif mutation == "manifest-operation-removal":
        del applied_manifest["target_operation"]
    elif mutation == "creation-target-removal":
        del applied_manifest["creation_target"]
    elif mutation in {
        "creation-base-commit",
        "creation-base-tree",
        "creation-primary",
        "creation-companion",
        "creation-canonical-manifest",
        "creation-orphan-manifest",
    }:
        field = mutation.removeprefix("creation-").replace("-", "_")
        applied_manifest["creation_target"][field] = (
            None if field == "orphan_manifest" else "wrong"
        )
        if field == "orphan_manifest":
            applied_manifest["creation_target"][field] = (
                ".axiom/encoding-manifests/policies/wrong.json"
            )
    elif mutation == "coherent-operation-rewrite":
        applied_manifest["target_operation"] = "replace"
        del applied_manifest["creation_target"]
    applied_relative = Path(
        ".axiom/encoding-manifests/us-la/policies/income_tax/standard_deduction.json"
    )
    if mutation == "wrong-manifest-path":
        applied_relative = Path(
            ".axiom/encoding-manifests/us-la/policies/income_tax/wrong.json"
        )
    applied_path = rulespec / applied_relative
    applied_path.parent.mkdir(parents=True)
    applied_bytes = json.dumps(applied_manifest).encode()
    if mutation == "duplicate-manifest-operation":
        applied_bytes = applied_bytes.replace(
            b'"target_operation": "create"',
            b'"target_operation": "replace", "target_operation": "create"',
            1,
        )
    elif mutation == "duplicate-creation-field":
        applied_bytes = applied_bytes.replace(
            f'"base_tree": "{rulespec_tree}"'.encode(),
            (
                '"base_tree": "' + "0" * 40 + f'", "base_tree": "{rulespec_tree}"'
            ).encode(),
            1,
        )
    applied_path.write_bytes(applied_bytes)

    if mutation == "sealed-source-bundle-replace":
        source_citation = source_bundle[0]
        source_context = {
            "citation": source_citation,
            "review_findings_files": [],
        }
        source_context_bytes = json.dumps(source_context, sort_keys=True).encode()
        source_context_path = (
            tmp_path / "generated" / "source-01" / "context-manifest.json"
        )
        source_context_path.parent.mkdir(parents=True)
        source_context_path.write_bytes(source_context_bytes)
        source_manifest_path = (
            rulespec
            / ".axiom/encoding-manifests/us-la/statutes/47/295.json"
        )
        source_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        source_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
                    "citation": source_citation,
                    "context_manifest_file": str(source_context_path),
                    "context_manifest_sha256": hashlib.sha256(
                        source_context_bytes
                    ).hexdigest(),
                    "applied_files": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    packaged_context = tmp_path / "artifact" / "context-manifest.json"
    packaged_inventory = tmp_path / "artifact" / "apply-manifests.json"
    packaged_context.parent.mkdir()
    _copy_targeted_atomic_inputs(tmp_path, packaged_context.parent)
    if mutation == "sealed-source-bundle-replace":
        for name in (
            "guard-generated.json",
            "metadata.json",
            "rulespec-generated-head.txt",
            "rulespec-tree.txt",
            "signed-import-inventory.json",
            "status.txt",
            "worktree-copy.json",
        ):
            (packaged_context.parent / name).write_text("fixture\n", encoding="utf-8")
    if mutation == "post-copy-original-controls":
        (tmp_path / "source-bundle.json").write_text(
            '["us-la/statute/47:999"]\n', encoding="utf-8"
        )
        (tmp_path / "canonical-refresh-bundle.json").write_text(
            "[{}]\n", encoding="utf-8"
        )
        (tmp_path / "primary-required-test-cases.json").write_text(
            "[]\n", encoding="utf-8"
        )
        (tmp_path / "target-operation.txt").write_text("replace\n", encoding="ascii")
    environment = {
        **os.environ,
        "ATOMIC_SOURCE_JSON": json.dumps(
            {
                "schema": "axiom-encode/atomic-source-transaction/v2",
                "source_bundle": source_bundle,
                "canonical_refresh_bundle": [],
                "primary_required_test_cases": required_cases,
                "target_operation": target_operation,
            }
        ),
        "CITATION": citation,
        "PYTHONPATH": str(ROOT / "src"),
        "REVIEW_FINDING": review_content.rstrip("\n"),
        "REVIEW_FINDING_PRESENT": "true",
        "RUNNER_TEMP": str(tmp_path),
        "RULESPEC_CHECKOUT": "rulespec-nz",
        "RULESPEC_REF": rulespec_ref,
        "REPLACE_RULESPEC_PATH": rulespec_path,
        **(
            {"SEALED_ARTIFACT_REVALIDATION": "1"}
            if mutation == "sealed-source-bundle-replace"
            else {}
        ),
    }
    completed = subprocess.run(
        [sys.executable, "-", str(packaged_context), str(packaged_inventory)],
        cwd=tmp_path,
        input=script,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    if expected_error is not None:
        assert completed.returncode != 0
        assert expected_error in completed.stderr
        return
    assert completed.returncode == 0, completed.stderr
    assert packaged_context.read_bytes() == context_bytes
    inventory = json.loads(packaged_inventory.read_text())
    assert inventory["schema"] == "axiom-encode/applied-manifest-inventory/v1"
    assert inventory["items"] == [
        {
            "citation": citation,
            "path": applied_path.relative_to(rulespec).as_posix(),
            "sha256": hashlib.sha256(applied_path.read_bytes()).hexdigest(),
        }
    ]


def _targeted_package_script() -> str:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    package_command = next(
        step
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Package exact generated changes"
    )["run"]
    function_start = package_command.index("verify_targeted_context() {")
    script_start = package_command.index("<<'PY'\n", function_start) + len("<<'PY'\n")
    script = package_command[script_start:].split("\nPY\n}", 1)[0]
    # The extracted verifier is exercised outside the provisioned root-owned
    # runtime. Keep these content/transaction tests hermetic; the production
    # wrapper's exact allowlist and ownership boundary have dedicated tests.
    test_git = str(Path(shutil.which("git") or "/usr/bin/git").resolve())
    return script.replace(
        '"/opt/axiom-verification/python/bin/git"',
        repr(test_git),
    )


def _targeted_metadata_script() -> str:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    package_command = next(
        step
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Package exact generated changes"
    )["run"]
    function_start = package_command.index("write_targeted_metadata() {")
    script_start = package_command.index("<<'PY'\n", function_start) + len("<<'PY'\n")
    return package_command[script_start:].split("\nPY\n}", 1)[0]


def _targeted_snapshot_shell() -> str:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    package_command = next(
        step
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("name") == "Package exact generated changes"
    )["run"]
    begin = "# BEGIN TARGETED_REENCODE_IMMUTABLE_SNAPSHOT\n"
    end = "# END TARGETED_REENCODE_IMMUTABLE_SNAPSHOT"
    return package_command.split(begin, 1)[1].split(end, 1)[0]


def test_targeted_snapshot_uses_exact_tree_after_live_checkout_race(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "rulespec-us"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=checkout,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=checkout, check=True)
    tracked = checkout / "us/policies/example.yaml"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("value: base\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=checkout, check=True)
    base_ref = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    tracked.write_text("value: captured\n", encoding="utf-8")
    untracked = checkout / "us/policies/example.test.yaml"
    untracked.write_text("cases: []\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=checkout, check=True)
    captured_tree = subprocess.check_output(
        ["git", "write-tree"], cwd=checkout, text=True
    ).strip()

    snapshot_parent = tmp_path / "protected"
    snapshot_parent.mkdir()
    snapshot = snapshot_parent / "rulespec-us"
    subprocess.run(
        ["git", "clone", "-q", "--no-local", "--no-checkout", checkout, snapshot],
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "-q", "--detach", base_ref],
        cwd=snapshot,
        check=True,
    )

    tracked.write_text("value: raced-live\n", encoding="utf-8")
    untracked.write_text("cases:\n  - malicious\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=checkout, check=True)
    exact_patch = subprocess.run(
        [
            "git",
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--no-textconv",
            base_ref,
            captured_tree,
            "--",
        ],
        cwd=checkout,
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(
        ["git", "apply", "--binary"],
        cwd=snapshot,
        check=True,
        input=exact_patch,
    )
    subprocess.run(["git", "add", "-A"], cwd=snapshot, check=True)
    reconstructed_tree = subprocess.check_output(
        ["git", "write-tree"], cwd=snapshot, text=True
    ).strip()

    assert reconstructed_tree == captured_tree
    assert tracked.read_text(encoding="utf-8") == "value: raced-live\n"
    assert (snapshot / tracked.relative_to(checkout)).read_text(
        encoding="utf-8"
    ) == "value: captured\n"
    assert (snapshot / untracked.relative_to(checkout)).read_text(
        encoding="utf-8"
    ) == "cases: []\n"
    routing = inspect_canonical_rulespec_checkout(snapshot)
    assert routing.name == "rulespec-us"
    assert routing.rejection is None


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="the protected verifier UID boundary is Linux-specific",
)
def test_targeted_snapshot_runs_protected_git_through_dedicated_verifier(
    tmp_path: Path,
    trusted_python_runtime: tuple[Path, Path, Path],
) -> None:
    """Exercise the workflow's real cross-UID protected snapshot boundary."""

    if os.geteuid() == 0:
        pytest.skip("the runner must be non-root to prove the ownership boundary")
    required_tools = ("go", "git", "sudo", "useradd", "userdel")
    missing = [name for name in required_tools if shutil.which(name) is None]
    if missing:
        pytest.skip(f"required Linux integration tools are missing: {missing}")
    sudo_probe = subprocess.run(
        ["sudo", "-n", "true"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if sudo_probe.returncode != 0:
        pytest.skip("passwordless sudo is required for the verifier UID fixture")

    verifier = f"axiomit{uuid.uuid4().hex[:10]}"
    protected_root: Path | None = None

    def sudo(
        *arguments: str | Path,
        check: bool = True,
        input_bytes: bytes | None = None,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["sudo", "-n", *(str(argument) for argument in arguments)],
            check=check,
            capture_output=True,
            input=input_bytes if input_bytes is not None else input_text,
            text=input_bytes is None,
            timeout=120,
        )

    created = sudo(
        "useradd",
        "--system",
        "--user-group",
        "--no-create-home",
        "--home-dir",
        "/nonexistent",
        "--shell",
        "/usr/sbin/nologin",
        verifier,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip(f"could not create dedicated verifier UID: {created.stderr}")
    try:
        verifier_uid = int(sudo("id", "-u", verifier).stdout.strip())
        assert verifier_uid not in {0, os.geteuid()}
        protected_root = Path(
            sudo(
                "mktemp",
                "-d",
                "/opt/axiom-verification-test.XXXXXXXX",
            ).stdout.strip()
        )
        assert protected_root.parent == Path("/opt")
        assert protected_root.name.startswith("axiom-verification-test.")
        sudo("chmod", "0711", protected_root)

        source = tmp_path / "live" / "rulespec-us"
        source.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=source, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=source,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=source,
            check=True,
        )
        tracked = source / "us/policies/example.yaml"
        tracked.parent.mkdir(parents=True)
        tracked.write_text("value: base\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=source, check=True)
        subprocess.run(["git", "commit", "-qm", "base"], cwd=source, check=True)
        base_ref = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
        tracked.write_text("value: protected\n", encoding="utf-8")
        companion = source / "us/policies/example.test.yaml"
        companion.write_text("cases: []\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=source, check=True)
        captured_tree = subprocess.check_output(
            ["git", "write-tree"], cwd=source, text=True
        ).strip()
        exact_patch = subprocess.run(
            [
                "git",
                "diff",
                "--binary",
                "--full-index",
                "--no-ext-diff",
                "--no-textconv",
                base_ref,
                captured_tree,
                "--",
            ],
            cwd=source,
            check=True,
            capture_output=True,
        ).stdout

        encoder_source = tmp_path / "live" / "axiom-encode"
        (encoder_source / "src/axiom_encode").mkdir(parents=True)
        (encoder_source / "pyproject.toml").write_text(
            f'[project]\nname = "axiom-encode"\nversion = "{__version__}"\n',
            encoding="utf-8",
        )
        (encoder_source / "src/axiom_encode/__init__.py").write_text(
            f'__version__ = "{__version__}"\n',
            encoding="utf-8",
        )
        (encoder_source / "uv.lock").write_text(
            "version = 1\n"
            "revision = 1\n"
            'requires-python = ">=3.12"\n\n'
            "[[package]]\n"
            'name = "axiom-encode"\n'
            f'version = "{__version__}"\n'
            'source = { editable = "." }\n',
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-q"], cwd=encoder_source, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=encoder_source,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=encoder_source,
            check=True,
        )
        subprocess.run(["git", "add", "."], cwd=encoder_source, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "encoder fixture"],
            cwd=encoder_source,
            check=True,
        )
        encoder_ref = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=encoder_source, text=True
        ).strip()

        git_home = protected_root / "git-home"
        snapshot = protected_root / "rulespec-us"
        sudo("mkdir", "-m", "0700", git_home)
        root_git = (
            "env",
            "-i",
            f"HOME={git_home}",
            "PATH=/usr/bin:/bin",
            "/usr/bin/git",
        )
        sudo(
            *root_git,
            "clone",
            "--no-local",
            "--no-checkout",
            source,
            snapshot,
        )
        sudo(*root_git, "-C", snapshot, "checkout", "--detach", base_ref)
        assert snapshot.stat().st_uid == 0
        sudo(
            *root_git,
            "-C",
            snapshot,
            "apply",
            "--binary",
            input_bytes=exact_patch,
        )
        protected_encoder = protected_root / "axiom-encode"
        sudo(
            *root_git,
            "clone",
            "--no-local",
            "--no-checkout",
            encoder_source,
            protected_encoder,
        )
        sudo(
            *root_git,
            "-C",
            protected_encoder,
            "checkout",
            "--detach",
            encoder_ref,
        )
        sudo(
            *root_git,
            "-C",
            protected_encoder,
            "remote",
            "set-url",
            "origin",
            "https://github.com/TheAxiomFoundation/axiom-encode.git",
        )
        sudo("chmod", "-R", "a+rX,go-w", protected_encoder)
        assert protected_encoder.stat().st_uid == 0
        assert protected_encoder.stat().st_mode & 0o001

        local_supervisor = tmp_path / "axiom-encode-signing-supervisor"
        subprocess.run(
            [
                "go",
                "build",
                "-trimpath",
                "-buildvcs=false",
                "-ldflags=-buildid=",
                "-o",
                local_supervisor,
                SUPERVISOR_PACKAGE,
            ],
            cwd=ROOT,
            env={**os.environ, "CGO_ENABLED": "0"},
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        protected_supervisor = protected_root / "axiom-encode-signing-supervisor"
        sudo(
            "install",
            "-o",
            "root",
            "-g",
            "root",
            "-m",
            "0755",
            local_supervisor,
            protected_supervisor,
        )

        _source_interpreter, source_runtime, source_package = trusted_python_runtime
        protected_runtime = protected_root / "python"
        sudo("mkdir", "-m", "0755", protected_runtime)
        sudo("cp", "-a", f"{source_runtime}/.", protected_runtime)
        protected_package = protected_runtime / source_package.relative_to(
            source_runtime
        )
        protected_interpreter = protected_runtime / Path(
            sys.executable
        ).resolve().relative_to(Path(sys.base_prefix).resolve())
        protected_git = protected_interpreter.parent / "git"
        sudo(
            "install",
            "-o",
            "root",
            "-g",
            "root",
            "-m",
            "0755",
            Path(shutil.which("git") or "/usr/bin/git").resolve(),
            protected_git,
        )
        wrapper_directory = tmp_path / "trusted-git-wrapper"
        wrapper_directory.mkdir()
        local_trusted_wrapper = provisioner._install_trusted_git_wrapper(
            wrapper_directory,
            protected_interpreter,
            Path(shutil.which("git") or "/usr/bin/git").resolve(),
        )
        protected_trusted_wrapper = protected_root / "trusted-git"
        sudo(
            "install",
            "-o",
            "root",
            "-g",
            "root",
            "-m",
            "0755",
            local_trusted_wrapper,
            protected_trusted_wrapper,
        )
        for module in ("corpus_release.py", "corpus_resolver.py", "statute.py"):
            sudo(
                "install",
                "-o",
                "root",
                "-g",
                "root",
                "-m",
                "0444",
                ROOT / "src/axiom_encode" / module,
                protected_package / module,
            )

        local_corpus = tmp_path / "materialized-corpus-source"
        release = _write_signed_corpus_release(
            local_corpus,
            release_name="protected-test-release",
            citation_path="us/statute/1",
            version="2026",
            body="The protected verifier can read this release.\n",
        )
        release_response = tmp_path / "protected-release-response.json"
        release_response.write_text(
            json.dumps(
                [
                    {
                        "release_object": json.loads(
                            release.release_object_path.read_text(encoding="utf-8")
                        )
                    }
                ]
            ),
            encoding="utf-8",
        )
        local_materialized_root = tmp_path / "materialized-corpus-output"
        local_materialized_root.mkdir()
        materialized_release, materialized_commit = (
            release_acquisition.materialize_registry_response(
                release_response,
                local_materialized_root,
                release_name=release.name,
                release_sha=release.content_sha256,
            )
        )
        assert materialized_commit == "a" * 40
        assert materialized_release.stat().st_mode & 0o777 == 0o600

        protected_corpus = protected_root / "axiom-corpus"
        protected_release_directory = protected_corpus / "releases" / release.name
        protected_release = (
            protected_release_directory / f"{release.content_sha256}.json"
        )
        sudo("mkdir", "-m", "0755", protected_corpus)
        sudo("cp", "-a", local_corpus / "data", protected_corpus / "data")
        sudo("mkdir", "-p", protected_release_directory)
        sudo(
            "install",
            "-o",
            "root",
            "-g",
            "root",
            "-m",
            "0444",
            materialized_release,
            protected_release,
        )
        sudo("chown", "-hR", "root:root", protected_corpus)
        sudo("chmod", "-R", "a+rX,go-w", protected_corpus)
        assert protected_release.stat().st_uid == 0
        assert protected_release.stat().st_mode & 0o777 == 0o444
        for searchable_ancestor in (
            protected_root,
            protected_corpus,
            protected_corpus / "releases",
            protected_release_directory,
        ):
            metadata = searchable_ancestor.stat()
            assert metadata.st_uid == 0
            assert metadata.st_mode & 0o001
        entrypoint = tmp_path / "entrypoint.py"
        entrypoint.write_text(
            """from __future__ import annotations
import json
import os
import subprocess
import sys

from axiom_encode.signing_broker import get_signing_broker

broker = get_signing_broker()
repository, expected_tree, runner_command_file = sys.argv[1:]
try:
    with open(runner_command_file, "a", encoding="utf-8") as stream:
        stream.write("BASH_ENV=/tmp/forged\\n")
except PermissionError:
    runner_command_write = "denied"
else:
    runner_command_write = "writable"
git = os.path.join(os.path.dirname(sys.executable), "git")
subprocess.run([git, "-C", repository, "add", "-A"], check=True)
tree = subprocess.run(
    [git, "-C", repository, "write-tree"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
if tree != expected_tree:
    raise SystemExit("protected snapshot tree differs from captured tree")
subprocess.run(
    [
        git,
        "-C",
        repository,
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "user.name=Axiom integration fixture",
        "-c",
        "user.email=fixture@axiom-foundation.org",
        "commit",
        "--quiet",
        "--no-gpg-sign",
        "-m",
        "Commit protected snapshot",
    ],
    check=True,
)
commit = subprocess.run(
    [git, "-C", repository, "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
parent = subprocess.run(
    [git, "-C", repository, "rev-parse", "HEAD^"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
print(json.dumps({
    "capabilities": sorted(broker.capabilities),
    "commit": commit,
    "euid": os.geteuid(),
    "parent": parent,
    "repository_name": os.path.basename(repository),
    "runner_command_write": runner_command_write,
    "tree": tree,
}, sort_keys=True))
broker.close()
""",
            encoding="utf-8",
        )
        sudo(
            "install",
            "-o",
            "root",
            "-g",
            "root",
            "-m",
            "0444",
            entrypoint,
            protected_package / "entrypoint.py",
        )
        launcher_source = tmp_path / "axiom-encode"
        launcher_source.write_text(
            f"#!{protected_interpreter} -I\nraise SystemExit('bootstrap only')\n",
            encoding="utf-8",
        )
        protected_launcher = protected_root / "axiom-encode"
        sudo(
            "install",
            "-o",
            "root",
            "-g",
            "root",
            "-m",
            "0555",
            launcher_source,
            protected_launcher,
        )
        apply_public, _apply_private = _keypair(b"\xab" * 32)
        eval_public, _eval_private = _keypair(b"\xcd" * 32)
        local_trust = _trust_config(tmp_path, apply_public, eval_public)
        protected_trust = protected_root / "signing-trust-roots.json"
        sudo(
            "install",
            "-o",
            "root",
            "-g",
            "root",
            "-m",
            "0444",
            local_trust,
            protected_trust,
        )
        sudo("chown", "-hR", "root:root", protected_runtime)
        sudo("chmod", "-R", "a+rX,go-w", protected_runtime)

        def foreign_git(
            repository: Path, *arguments: str
        ) -> subprocess.CompletedProcess:
            return sudo(
                "-u",
                verifier,
                "--",
                "env",
                "-i",
                "HOME=/nonexistent",
                "GIT_CONFIG_GLOBAL=/dev/null",
                "GIT_CONFIG_NOSYSTEM=1",
                "PATH=/usr/bin:/bin",
                protected_trusted_wrapper,
                "-C",
                repository,
                *arguments,
            )

        assert (
            Path(foreign_git(snapshot, "rev-parse", "--show-toplevel").stdout.strip())
            == snapshot
        )
        assert (
            Path(
                foreign_git(
                    protected_encoder, "rev-parse", "--show-toplevel"
                ).stdout.strip()
            )
            == protected_encoder
        )
        assert (
            foreign_git(
                protected_encoder, "rev-parse", "--verify", "HEAD"
            ).stdout.strip()
            == encoder_ref
        )
        assert (
            foreign_git(protected_encoder, "diff", "--binary", "HEAD", "--").stdout
            == ""
        )
        assert (
            foreign_git(
                protected_encoder,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
            ).stdout
            == ""
        )
        assert (
            foreign_git(protected_encoder, "remote", "get-url", "origin").stdout.strip()
            == "https://github.com/TheAxiomFoundation/axiom-encode.git"
        )
        for version_path in (
            "pyproject.toml",
            "src/axiom_encode/__init__.py",
            "uv.lock",
        ):
            expected_version_blob = subprocess.check_output(
                ["git", "show", f"{encoder_ref}:{version_path}"],
                cwd=encoder_source,
                text=True,
            )
            assert (
                foreign_git(protected_encoder, "show", f"HEAD:{version_path}").stdout
                == expected_version_blob
            )

        parsed_release = sudo(
            "-u",
            verifier,
            "--",
            "env",
            "-i",
            "HOME=/nonexistent",
            "PATH=/usr/bin:/bin",
            protected_interpreter,
            "-I",
            "-c",
            (
                "import json,sys; from pathlib import Path; "
                "from axiom_encode.corpus_resolver import LocalCorpusRelease; "
                "release=LocalCorpusRelease(Path(sys.argv[1]),sys.argv[2],"
                "sys.argv[3],sys.argv[4]); "
                "print(json.dumps({'name':release.name,'digest':"
                "release.content_sha256,'object':str(release.release_object_path),"
                "'scopes':len(release.scopes)},sort_keys=True))"
            ),
            protected_corpus,
            release.name,
            release.content_sha256,
            TEST_RELEASE_PUBLIC_KEY,
        )
        assert json.loads(parsed_release.stdout) == {
            "digest": release.content_sha256,
            "name": release.name,
            "object": str(protected_release),
            "scopes": 1,
        }
        sudo("chown", "-hR", f"{verifier}:{verifier}", snapshot, git_home)

        assert protected_root.stat().st_uid == 0
        assert protected_supervisor.stat().st_uid == 0
        assert protected_runtime.stat().st_uid == 0
        assert snapshot.stat().st_uid == verifier_uid
        assert snapshot.name == "rulespec-us"

        runner_command_directory = tmp_path / "_runner_file_commands"
        runner_command_directory.mkdir(mode=0o700)
        runner_command_file = runner_command_directory / "github_env"
        runner_command_file.write_text("", encoding="utf-8")
        runner_command_file.chmod(0o600)
        assert runner_command_file.stat().st_uid == os.geteuid()

        build_kind = sudo("-u", verifier, "--", protected_supervisor, "--build-kind")
        assert build_kind.stdout.strip() == "production"
        supervised = sudo(
            "-u",
            verifier,
            "--",
            "env",
            "-i",
            protected_supervisor,
            "--trusted-signing-roots",
            protected_trust,
            "--trusted-python-runtime-root",
            protected_runtime,
            "--trusted-python-import-root",
            protected_package.parent,
            "--trusted-python-package-root",
            protected_package,
            "--",
            protected_launcher,
            snapshot,
            captured_tree,
            runner_command_file,
        )
        result = json.loads(supervised.stdout)
        assert result == {
            "capabilities": [],
            "commit": result["commit"],
            "euid": verifier_uid,
            "parent": base_ref,
            "repository_name": "rulespec-us",
            "runner_command_write": "denied",
            "tree": captured_tree,
        }
        assert re.fullmatch(r"[0-9a-f]{40}", result["commit"])

        witness = protected_root / "publication-commit.txt"
        sudo("tee", witness, input_text=f"{result['commit']}\n")
        sudo("chown", "root:root", witness)
        sudo("chmod", "0444", witness)
        assert witness.read_text(encoding="ascii").strip() == result["commit"]
        assert witness.stat().st_uid == 0
        assert (snapshot / "us/policies/example.yaml").read_text(
            encoding="utf-8"
        ) == "value: protected\n"
        assert (snapshot / "us/policies/example.test.yaml").read_text(
            encoding="utf-8"
        ) == "cases: []\n"

        with pytest.raises(PermissionError):
            tracked_in_snapshot = snapshot / "us/policies/example.yaml"
            with tracked_in_snapshot.open("a", encoding="utf-8") as stream:
                stream.write("runner mutation\n")
        denied_witness = sudo(
            "-u",
            verifier,
            "--",
            "/usr/bin/tee",
            witness,
            check=False,
            input_text="forged\n",
        )
        assert denied_witness.returncode != 0
        denied_parent = sudo(
            "-u",
            verifier,
            "--",
            "/usr/bin/touch",
            protected_root / "forged",
            check=False,
        )
        assert denied_parent.returncode != 0
    finally:
        if protected_root is not None:
            assert protected_root.parent == Path("/opt")
            assert protected_root.name.startswith("axiom-verification-test.")
            sudo("rm", "-rf", "--", protected_root, check=False)
        sudo("userdel", verifier, check=False)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (None, MULTILANE_REPLACE_ERROR),
        (
            "swapped-manifests",
            "canonical refresh apply manifest does not match its exact requested target",
        ),
        (
            "missing-manifest",
            "canonical refresh changed manifest inventory differs from its exact requested targets",
        ),
        (
            "extra-manifest",
            "canonical refresh changed manifest inventory differs from its exact requested targets",
        ),
        (
            "unrelated-companion",
            "canonical refresh apply manifest does not match its exact requested target",
        ),
        (
            "wrong-companion-finding",
            "context manifest does not bind the supplied review finding",
        ),
        (
            "unexpected-companion-finding",
            "context manifest contains an unexpected review finding",
        ),
        (
            "wrong-review-contract",
            "context manifest does not bind the normalized review contract",
        ),
        (
            "wrong-review-contract-input-type",
            "context manifest does not bind the normalized review contract",
        ),
        (
            "wrong-review-contract-output-type",
            "context manifest does not bind the normalized review contract",
        ),
        (
            "control-companion-finding",
            "normalized canonical refresh bundle is malformed",
        ),
        (
            "unexpected-creation-target",
            "signed apply manifest target operation differs from the normalized review contract",
        ),
        (
            "existing-import-race",
            "persisted existing signed imports differ from dispatch input",
        ),
        (
            "inapplicable-context-in-sealed-artifact",
            "sealed artifact inventory differs from this transaction",
        ),
    ],
)
def test_targeted_artifact_enforces_exact_canonical_refresh_inventory(
    tmp_path: Path,
    mutation: str | None,
    expected_error: str,
) -> None:
    script = _targeted_package_script()
    rulespec = tmp_path / "rulespec-us"
    rulespec.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=rulespec, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=rulespec,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=rulespec, check=True)
    (rulespec / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=rulespec, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=rulespec, check=True)
    existing_import_path = "us-la/statutes/47/293.yaml"
    existing_import = rulespec / existing_import_path
    existing_import.parent.mkdir(parents=True, exist_ok=True)
    existing_import.write_text("rules: []\n", encoding="utf-8")
    existing_import_manifest = (
        rulespec
        / ".axiom/encoding-manifests"
        / Path(existing_import_path).with_suffix(".json")
    )
    existing_import_manifest.parent.mkdir(parents=True, exist_ok=True)
    existing_import_manifest.write_text(
        json.dumps({"schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA}) + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            "git",
            "add",
            existing_import_path,
            existing_import_manifest.relative_to(rulespec),
        ],
        cwd=rulespec,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-qm", "existing signed import"],
        cwd=rulespec,
        check=True,
    )
    rulespec_ref = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=rulespec, text=True
    ).strip()

    target_rows = [
        (
            "us-la/statute/47:294",
            "us-la/statutes/47/294.yaml",
            ".axiom/encoding-manifests/us-la/statutes/47/294.json",
            "target",
        ),
        (
            "us-la/statute/47:295",
            "us-la/statutes/47/295.yaml",
            ".axiom/encoding-manifests/us-la/statutes/47/295.json",
            "canonical-refresh-01",
        ),
    ]
    inventory: list[dict[str, str | None]] = []
    manifests: list[tuple[Path, dict[str, object]]] = []
    for index, (citation, rulespec_path, manifest_path, lane) in enumerate(target_rows):
        rule = rulespec / rulespec_path
        rule.parent.mkdir(parents=True, exist_ok=True)
        rule.write_text(f"format: rulespec/v1\nrules: [{index}]\n", encoding="utf-8")
        finding = (
            "Preserve the primary semantics.\n"
            if index == 0
            else "Preserve the companion ownership boundary.\n"
        )
        required_test_cases = (
            [
                {
                    "name": "2025 single",
                    "period": {
                        "period_kind": "tax_year",
                        "start": "2025-01-01",
                        "end": "2025-12-31",
                    },
                    "input": {
                        "us-la:statutes/47/294#input.single": True,
                        "us-la:statutes/47/294#input.joint": False,
                    },
                    "required_output": {
                        "us-la:statutes/47/294#standard_deduction": 12500,
                    },
                }
            ]
            if index == 0
            else []
        )
        context = {
            "citation": citation,
            "review_findings_files": (
                [
                    {
                        "content": finding,
                        "sha256": hashlib.sha256(finding.encode()).hexdigest(),
                    }
                ]
                if finding
                else []
            ),
        }
        context["review_contract"] = {
            "schema": "axiom-encode/review-contract/v3",
            "citation": citation,
            "rulespec_path": rulespec_path,
            "required_deferred_outputs": [],
            "required_test_cases": json.loads(json.dumps(required_test_cases)),
            "target_operation": "replace",
        }
        if required_test_cases:
            if mutation == "wrong-review-contract":
                context["review_contract"]["required_test_cases"][0]["required_output"][
                    "us-la:statutes/47/294#standard_deduction"
                ] = 0
            if mutation == "wrong-review-contract-input-type":
                context["review_contract"]["required_test_cases"][0]["input"][
                    "us-la:statutes/47/294#input.single"
                ] = 1
            if mutation == "wrong-review-contract-output-type":
                context["review_contract"]["required_test_cases"][0]["required_output"][
                    "us-la:statutes/47/294#standard_deduction"
                ] = 12500.0
        if mutation == "control-companion-finding" and index > 0:
            finding = "Preserve the companion\x00 ownership boundary.\n"
            context["review_findings_files"] = [
                {
                    "content": finding,
                    "sha256": hashlib.sha256(finding.encode()).hexdigest(),
                }
            ]
        context_bytes = json.dumps(context, sort_keys=True).encode()
        context_path = tmp_path / "generated" / lane / "context-manifest.json"
        context_path.parent.mkdir(parents=True, exist_ok=True)
        context_path.write_bytes(context_bytes)
        payload: dict[str, object] = {
            "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
            "tool": "axiom-encode encode --apply",
            "citation": citation,
            "context_manifest_file": str(context_path),
            "context_manifest_sha256": hashlib.sha256(context_bytes).hexdigest(),
            "target_operation": "replace",
            "applied_files": [
                {
                    "path": rulespec_path,
                    "sha256": hashlib.sha256(rule.read_bytes()).hexdigest(),
                }
            ],
        }
        if mutation == "unexpected-creation-target" and index == 0:
            payload["creation_target"] = {
                "base_commit": rulespec_ref,
                "base_tree": "0" * 40,
                "primary": rulespec_path,
                "companion": rulespec_path.replace(".yaml", ".test.yaml"),
                "canonical_manifest": manifest_path,
                "orphan_manifest": None,
            }
        manifests.append((rulespec / manifest_path, payload))
        inventory.append(
            {
                "citation": citation,
                "review_finding": (
                    (
                        "A different companion finding."
                        if mutation == "wrong-companion-finding"
                        else finding.rstrip("\n")
                    )
                    if index > 0 and mutation != "unexpected-companion-finding"
                    else None
                ),
                "deferred_output_contracts": [],
                "required_test_cases": required_test_cases,
                "rulespec_path": rulespec_path,
                "rulespec_sha256": "a" * 64,
                "companion_path": str(
                    Path(rulespec_path).with_name(
                        f"{Path(rulespec_path).stem}.test.yaml"
                    )
                ),
                "companion_sha256": None,
                "manifest_path": manifest_path,
                "manifest_sha256": "b" * 64,
            }
        )

    if mutation == "swapped-manifests":
        first_payload = manifests[0][1]
        second_payload = manifests[1][1]
        manifests[0] = (manifests[0][0], second_payload)
        manifests[1] = (manifests[1][0], first_payload)
    if mutation == "missing-manifest":
        manifests.pop()
    if mutation == "unrelated-companion":
        applied_files = manifests[0][1]["applied_files"]
        assert isinstance(applied_files, list)
        applied_files.append(
            {
                "path": "us-la/statutes/47/295.test.yaml",
                "sha256": "c" * 64,
            }
        )
    for path, payload in manifests:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    if mutation == "extra-manifest":
        extra = (
            rulespec / ".axiom/encoding-manifests/us-la/statutes/47/unrequested.json"
        )
        extra.parent.mkdir(parents=True, exist_ok=True)
        extra.write_text(
            json.dumps(
                {
                    "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
                    "tool": "axiom-encode encode --apply",
                    "citation": "us-la/statute/47:999",
                    "applied_files": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    (tmp_path / "source-bundle.json").write_text("[]\n", encoding="utf-8")
    (tmp_path / "canonical-refresh-bundle.json").write_text(
        json.dumps(inventory) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "existing-signed-imports.json").write_text(
        json.dumps([existing_import_path], separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "primary-required-test-cases.json").write_text(
        json.dumps(inventory[0]["required_test_cases"]) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "target-operation.txt").write_text("replace\n", encoding="ascii")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _copy_targeted_atomic_inputs(tmp_path, artifact)
    if mutation == "existing-import-race":
        (artifact / "existing-signed-imports.json").write_text("[]\n", encoding="utf-8")
    if mutation == "inapplicable-context-in-sealed-artifact":
        for name in (
            "guard-generated.json",
            "metadata.json",
            "rulespec-generated-head.txt",
            "rulespec-tree.txt",
            "signed-import-inventory.json",
            "status.txt",
        ):
            (artifact / name).write_text("fixture\n", encoding="utf-8")
        (artifact / "dependent-context-manifest.json").write_text(
            "forged\n", encoding="utf-8"
        )
    packaged_context = artifact / "context-manifest.json"
    packaged_inventory = artifact / "apply-manifests.json"
    atomic_refresh_item = {
        "citation": target_rows[1][0],
        "replace_rulespec_path": target_rows[1][1],
    }
    if inventory[1]["review_finding"] is not None:
        atomic_refresh_item["review_finding"] = inventory[1]["review_finding"]
    completed = subprocess.run(
        [sys.executable, "-", str(packaged_context), str(packaged_inventory)],
        cwd=tmp_path,
        input=script,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            **(
                {"SEALED_ARTIFACT_REVALIDATION": "1"}
                if mutation == "inapplicable-context-in-sealed-artifact"
                else {}
            ),
            "ATOMIC_SOURCE_JSON": json.dumps(
                {
                    "schema": "axiom-encode/atomic-source-transaction/v2",
                    "source_bundle": [],
                    "canonical_refresh_bundle": (
                        []
                        if mutation == "persisted-only-refresh"
                        else [atomic_refresh_item]
                    ),
                    "primary_required_test_cases": inventory[0]["required_test_cases"],
                    "target_operation": "replace",
                }
            ),
            "CITATION": target_rows[0][0],
            "EXISTING_SIGNED_IMPORTS_JSON": json.dumps([existing_import_path]),
            "PYTHONPATH": str(ROOT / "src"),
            "REVIEW_FINDING": "Preserve the primary semantics.",
            "REVIEW_FINDING_PRESENT": "true",
            "RUNNER_TEMP": str(tmp_path),
            "RULESPEC_CHECKOUT": "rulespec-us",
            "RULESPEC_REF": rulespec_ref,
            "REPLACE_RULESPEC_PATH": target_rows[0][1],
        },
    )

    assert completed.returncode != 0
    assert expected_error in completed.stderr


def test_targeted_metadata_uses_consumed_repair_identity_after_evidence_mutation(
    tmp_path: Path,
) -> None:
    script = _targeted_metadata_script()
    heads: dict[str, str] = {}
    for name in ("axiom-encode", "axiom-corpus", "axiom-rules-engine", "rulespec-us"):
        repository = tmp_path / name
        repository.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repository,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=repository,
            check=True,
        )
        (repository / "README.md").write_text("fixture\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repository, check=True)
        heads[name] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()

    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    (runner_temp / "source-bundle.json").write_text("[]\n", encoding="utf-8")
    (runner_temp / "canonical-refresh-bundle.json").write_text("[]\n", encoding="utf-8")
    (runner_temp / "existing-signed-imports.json").write_text("[]\n", encoding="utf-8")
    (runner_temp / "signed-import-inventory.json").write_text("{}\n", encoding="utf-8")
    (runner_temp / "rulespec-tree.txt").write_text(
        f"{heads['rulespec-us']}\n", encoding="ascii"
    )
    (runner_temp / "repair-candidate.json").write_text(
        json.dumps(
            {
                "path": "statutes/42/mutated.yaml",
                "root": "/mutated",
                "rulespec_sha256": "b" * 64,
                "runner": "mutated-runner",
                "tests_sha256": "c" * 64,
            }
        ),
        encoding="utf-8",
    )

    metadata_path = runner_temp / "metadata.json"
    completed = subprocess.run(
        [sys.executable, "-", str(metadata_path)],
        cwd=tmp_path,
        input=script,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CITATION": "us/statute/42/1437c\u20131",
            "CORPUS_REF": heads["axiom-corpus"],
            "GITHUB_SHA": heads["axiom-encode"],
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_RUN_ID": "5678",
            "PR_BASE_BRANCH": "hard-cut/canonical-layout-us",
            "REPAIR_CANDIDATE_PATH": "statutes/42/1437c-1.yaml",
            "REPAIR_CANDIDATE_RUNNER": "openai-gpt-5.6-sol",
            "REPAIR_CANDIDATE_SOURCE_RULESPEC_REF": "f" * 40,
            "REPAIR_CANDIDATE_RULESPEC_SHA256": "d" * 64,
            "REPAIR_CANDIDATE_TESTS_SHA256": "e" * 64,
            "REPAIR_RUN_ID": "1234",
            "RULESPEC_CHECKOUT": str(tmp_path / "rulespec-us"),
            "RULESPEC_GENERATED_HEAD": heads["rulespec-us"],
            "RULESPEC_REF": heads["rulespec-us"],
            "RULES_ENGINE_REF": heads["axiom-rules-engine"],
            "RUNNER_TEMP": str(runner_temp),
        },
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["repair_candidate"] == {
        "path": "statutes/42/1437c-1.yaml",
        "rulespec_sha256": "d" * 64,
        "run_id": "1234",
        "runner": "openai-gpt-5.6-sol",
        "source_rulespec_ref": "f" * 40,
        "tests_sha256": "e" * 64,
    }


@pytest.mark.parametrize("receipt_version", [4, 5, 6, 7])
@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("success", None),
        ("missing-input", "retained successors differ from inputs"),
        ("extra-input", "retained successors differ from inputs"),
        ("reordered-input", "retained successors differ from inputs"),
        ("tampered-evidence", "retained predecessor file digest differs"),
        ("missing-deleted-manifest", "deleted manifest inventory differs"),
        ("extra-deleted-manifest", "deleted manifest inventory differs"),
        ("tampered-exact-binding", "exact dependent manifest binding differs"),
        (
            "nested-operation-flip",
            "legacy replacement model manifest target operation differs",
        ),
        (
            "nested-operation-removal",
            "legacy replacement model manifest target operation differs",
        ),
    ],
)
def test_targeted_artifact_packages_replacement_closure(
    tmp_path: Path,
    receipt_version: int,
    mutation: str,
    error: str | None,
) -> None:
    if mutation == "tampered-exact-binding" and receipt_version != 7:
        pytest.skip("exact-dependent v7 binding case")
    script = _targeted_package_script()
    rulespec = tmp_path / "rulespec-us"
    rulespec.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=rulespec, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=rulespec,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=rulespec, check=True)

    def write(path: str, raw: bytes) -> Path:
        target = rulespec / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        return target

    def evidence(path: Path) -> dict[str, str]:
        return {
            "path": path.relative_to(rulespec).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    main_old = write("us/statutes/47:32.yaml", b"old main\n")
    main_old_manifest = write(
        ".axiom/encoding-manifests/us/statutes/47:32.json", b"old main manifest\n"
    )
    unrelated_manifest = write(
        ".axiom/encoding-manifests/us/statutes/unrelated.json", b"unrelated\n"
    )
    retained_rows: list[dict[str, object]] = []
    retained_sources = [
        "us/statutes/47:294.yaml",
        "us/statutes/47:295.yaml",
    ]
    for number in (294, 295):
        old = write(f"us/statutes/47:{number}.yaml", f"old {number}\n".encode())
        old_test = write(f"us/statutes/47:{number}.test.yaml", b"[]\n")
        old_manifest = write(
            f".axiom/encoding-manifests/us/statutes/47:{number}.json",
            f"old manifest {number}\n".encode(),
        )
        successor = write(f"us/statutes/47/{number}.yaml", old.read_bytes())
        successor_test = write(
            f"us/statutes/47/{number}.test.yaml", old_test.read_bytes()
        )
        successor_files = [evidence(successor), evidence(successor_test)]
        original_payload = {
            "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
            "tool": "axiom-encode encode --apply",
            "backend": "codex",
            "citation": f"us/statute/47:{number}",
            "applied_files": successor_files,
        }
        original_raw = (
            json.dumps(original_payload, indent=2, sort_keys=True) + "\n"
        ).encode()
        successor_manifest = write(
            f".axiom/encoding-manifests/us/statutes/47/{number}.json",
            original_raw,
        )
        retained_rows.append(
            {
                "source": old.relative_to(rulespec).as_posix(),
                "destination": successor.relative_to(rulespec).as_posix(),
                "legacy_owner_class": "v1-manual-hmac-untrusted",
                "legacy_manifest": evidence(old_manifest),
                "legacy_files": [evidence(old), evidence(old_test)],
                "successor_manifest": {
                    **evidence(successor_manifest),
                    "payload": original_payload,
                },
                "successor_files": successor_files,
            }
        )
    exact_primary_path = "us/policies/income_tax/2026_resident_core.yaml"
    exact_manifest_path = (
        ".axiom/encoding-manifests/us/policies/income_tax/2026_resident_core.json"
    )
    exact_legacy_file: dict[str, str] | None = None
    exact_legacy_manifest: dict[str, str] | None = None
    if receipt_version == 7:
        exact_legacy_file = evidence(
            write(exact_primary_path, b"old exact dependent\n")
        )
        exact_legacy_manifest = evidence(
            write(exact_manifest_path, b"old exact dependent manifest\n")
        )
    subprocess.run(["git", "add", "."], cwd=rulespec, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=rulespec, check=True)
    rulespec_ref = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=rulespec, text=True
    ).strip()

    main_old.unlink()
    main_old_manifest.unlink()
    for row in retained_rows:
        for item in row["legacy_files"]:
            (rulespec / item["path"]).unlink()
        (rulespec / row["legacy_manifest"]["path"]).unlink()
    main_live = write("us/statutes/47/32.yaml", b"new main\n")
    exact_dependents: list[dict[str, object]] = []
    if receipt_version == 7:
        assert exact_legacy_file is not None
        assert exact_legacy_manifest is not None
        exact_live = write(exact_primary_path, b"new exact dependent\n")
        exact_live_file = evidence(exact_live)
        exact_dependents.append(
            {
                "primary": exact_primary_path,
                "legacy_manifest": exact_legacy_manifest,
                "legacy_files": [exact_legacy_file],
                "live_files": [exact_live_file],
                "rewrites": [
                    {
                        "path": exact_primary_path,
                        "before_sha256": exact_legacy_file["sha256"],
                        "after_sha256": exact_live_file["sha256"],
                        "replacements": [],
                        "proof_import_repairs": 0,
                        "proof_excerpt_reanchors": [],
                    }
                ],
                "source_verification_migration": None,
                "concept_replacements": [],
            }
        )
    citation = "us/statute/47:32"
    review_content = "Retain the canonical successors.\n"
    context_payload = {
        "citation": citation,
        "review_contract": {
            "schema": "axiom-encode/review-contract/v3",
            "citation": citation,
            "rulespec_path": "us/statutes/47/32.yaml",
            "required_deferred_outputs": [],
            "required_test_cases": [],
            "target_operation": "replace",
        },
        "review_findings_files": [
            {
                "content": review_content,
                "sha256": hashlib.sha256(review_content.encode()).hexdigest(),
            }
        ],
    }
    context_bytes = json.dumps(context_payload, sort_keys=True).encode()
    context_path = tmp_path / "generated/target/context-manifest.json"
    context_path.parent.mkdir(parents=True)
    context_path.write_bytes(context_bytes)
    nested = {
        "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
        "tool": "axiom-encode encode --apply",
        "backend": "codex",
        "citation": citation,
        "context_manifest_file": str(context_path),
        "context_manifest_sha256": hashlib.sha256(context_bytes).hexdigest(),
        "target_operation": "replace",
        "applied_files": [evidence(main_live)],
    }
    if mutation == "nested-operation-flip":
        nested["target_operation"] = "create"
    elif mutation == "nested-operation-removal":
        del nested["target_operation"]
    receipt_payload = {
        "schema_version": (
            f"axiom-encode/legacy-fresh-reencode-receipt/v{receipt_version}"
        ),
        "replacement": {
            "source": "us/statutes/47:32.yaml",
            "destination": "us/statutes/47/32.yaml",
            "scheduled_dependents": [],
            "exact_dependents": exact_dependents,
            "retained_successors": retained_rows,
            "metadata_reconciliations": [],
        },
    }
    receipt_path = rulespec / ".axiom/legacy-replacements" / f"{'a' * 64}.json"
    receipt_path.parent.mkdir(parents=True)

    if mutation == "tampered-evidence":
        retained_rows[0]["legacy_files"][0]["sha256"] = "0" * 64
    if mutation == "missing-deleted-manifest":
        first_manifest = retained_rows[0]["legacy_manifest"]
        (rulespec / first_manifest["path"]).write_text("old manifest 294\n")
    if mutation == "extra-deleted-manifest":
        unrelated_manifest.unlink()

    receipt_path.write_text(json.dumps(receipt_payload, sort_keys=True) + "\n")
    receipt_digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    for row in retained_rows:
        refreshed = {
            "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
            "tool": (
                "axiom-encode encode --apply --legacy-retained-successor-rulespec-path"
            ),
            "applied_files": row["successor_files"],
            "retained_successor_manifest": row["successor_manifest"]["payload"],
            "legacy_migration": {
                "receipt_path": receipt_path.relative_to(rulespec).as_posix(),
                "receipt_sha256": receipt_digest,
                "source": row["source"],
                "destination": row["destination"],
                "legacy_manifest_path": row["legacy_manifest"]["path"],
                "legacy_manifest_sha256": row["legacy_manifest"]["sha256"],
                "successor_manifest_sha256": row["successor_manifest"]["sha256"],
            },
        }
        (rulespec / row["successor_manifest"]["path"]).write_text(
            json.dumps(refreshed, sort_keys=True) + "\n"
        )
    exact_manifest: Path | None = None
    if receipt_version == 7:
        exact_migration = {
            "receipt_path": receipt_path.relative_to(rulespec).as_posix(),
            "receipt_sha256": receipt_digest,
            "primary": exact_primary_path,
        }
        if mutation == "tampered-exact-binding":
            exact_migration["receipt_sha256"] = "0" * 64
        exact_manifest = write(
            exact_manifest_path,
            (
                json.dumps(
                    {
                        "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
                        "tool": (
                            "axiom-encode encode --apply "
                            "--legacy-exact-dependent-rulespec-path"
                        ),
                        "applied_files": exact_dependents[0]["live_files"],
                        "legacy_migration": exact_migration,
                    },
                    sort_keys=True,
                )
                + "\n"
            ).encode(),
        )
    outer = {
        "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
        "tool": "axiom-encode encode --apply --replace-legacy-rulespec-path",
        "replacement_manifest": nested,
        "replacement": {
            "receipt_path": receipt_path.relative_to(rulespec).as_posix(),
            "receipt_sha256": receipt_digest,
        },
    }
    outer_path = write(
        ".axiom/encoding-manifests/us/statutes/47/32.json",
        (json.dumps(outer, sort_keys=True) + "\n").encode(),
    )
    (tmp_path / "source-bundle.json").write_text("[]\n")
    (tmp_path / "canonical-refresh-bundle.json").write_text("[]\n")
    (tmp_path / "primary-required-test-cases.json").write_text("[]\n")
    (tmp_path / "target-operation.txt").write_text("replace\n", encoding="ascii")
    selected_inputs = list(retained_sources)
    if mutation == "missing-input":
        selected_inputs.pop()
    elif mutation == "extra-input":
        selected_inputs.append("us/statutes/47:296.yaml")
    elif mutation == "reordered-input":
        selected_inputs.reverse()
    packaged_context = tmp_path / "artifact/context-manifest.json"
    packaged_inventory = tmp_path / "artifact/apply-manifests.json"
    packaged_context.parent.mkdir()
    _copy_targeted_atomic_inputs(tmp_path, packaged_context.parent)
    completed = subprocess.run(
        [sys.executable, "-", str(packaged_context), str(packaged_inventory)],
        cwd=tmp_path,
        input=script,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ATOMIC_SOURCE_JSON": json.dumps(
                {
                    "schema": "axiom-encode/atomic-source-transaction/v2",
                    "source_bundle": [],
                    "canonical_refresh_bundle": [],
                    "primary_required_test_cases": [],
                    "target_operation": "replace",
                }
            ),
            "CITATION": citation,
            "PYTHONPATH": str(ROOT / "src"),
            "REVIEW_FINDING": review_content.rstrip("\n"),
            "REVIEW_FINDING_PRESENT": "true",
            "RUNNER_TEMP": str(tmp_path),
            "RULESPEC_CHECKOUT": "rulespec-us",
            "RULESPEC_REF": rulespec_ref,
            "REPLACE_LEGACY_RULESPEC_PATH": "us/statutes/47:32.yaml",
            "REPLACE_RULESPEC_PATH": "us/statutes/47/32.yaml",
            "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": json.dumps(
                selected_inputs
            ),
            "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": (
                exact_primary_path if receipt_version == 7 else ""
            ),
        },
    )

    if error is not None:
        assert completed.returncode != 0
        assert error in completed.stderr
        return
    assert completed.returncode == 0, completed.stderr
    inventory_paths = {
        item["path"] for item in json.loads(packaged_inventory.read_text())["items"]
    }
    assert outer_path.relative_to(rulespec).as_posix() in inventory_paths
    assert {str(row["successor_manifest"]["path"]) for row in retained_rows}.issubset(
        inventory_paths
    )
    if exact_manifest is not None:
        assert exact_manifest.relative_to(rulespec).as_posix() in inventory_paths


@pytest.mark.parametrize("receipt_version", [1, 2, 3])
def test_targeted_artifact_preserves_pre_v4_replacement_receipts(
    tmp_path: Path,
    receipt_version: int,
) -> None:
    script = _targeted_package_script()
    rulespec = tmp_path / "rulespec-us"
    rulespec.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=rulespec, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=rulespec,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=rulespec, check=True)
    old = rulespec / "us/statutes/47:32.yaml"
    old_manifest = rulespec / ".axiom/encoding-manifests/us/statutes/47:32.json"
    old.parent.mkdir(parents=True)
    old_manifest.parent.mkdir(parents=True)
    old.write_text("old\n")
    old_manifest.write_text("old manifest\n")
    subprocess.run(["git", "add", "."], cwd=rulespec, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=rulespec, check=True)
    rulespec_ref = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=rulespec, text=True
    ).strip()
    old.unlink()
    old_manifest.unlink()
    live = rulespec / "us/statutes/47/32.yaml"
    live.parent.mkdir(parents=True)
    live.write_text("new\n")
    citation = "us/statute/47:32"
    context_payload = {
        "citation": citation,
        "review_contract": {
            "schema": "axiom-encode/review-contract/v3",
            "citation": citation,
            "rulespec_path": "us/statutes/47/32.yaml",
            "required_deferred_outputs": [],
            "required_test_cases": [],
            "target_operation": "replace",
        },
        "review_findings_files": [],
    }
    context_bytes = json.dumps(context_payload, sort_keys=True).encode()
    context_path = tmp_path / "generated/target/context-manifest.json"
    context_path.parent.mkdir(parents=True)
    context_path.write_bytes(context_bytes)
    nested = {
        "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
        "tool": "axiom-encode encode --apply",
        "backend": "codex",
        "citation": citation,
        "context_manifest_file": str(context_path),
        "context_manifest_sha256": hashlib.sha256(context_bytes).hexdigest(),
        "target_operation": "replace",
        "applied_files": [],
    }
    receipt_replacement = {
        "source": "us/statutes/47:32.yaml",
        "destination": "us/statutes/47/32.yaml",
        "scheduled_dependents": [],
    }
    if receipt_version >= 2:
        receipt_replacement["exact_dependents"] = []
    if receipt_version >= 3:
        receipt_replacement["destination_predecessor_class"] = (
            "canonicalized-unowned-duplicate"
        )
        receipt_replacement["destination_predecessor_files"] = []
    receipt_payload = {
        "schema_version": (
            f"axiom-encode/legacy-fresh-reencode-receipt/v{receipt_version}"
        ),
        "replacement": receipt_replacement,
    }
    receipt_path = rulespec / ".axiom/legacy-replacements" / f"{'b' * 64}.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt_payload) + "\n")
    outer = {
        "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
        "tool": "axiom-encode encode --apply --replace-legacy-rulespec-path",
        "replacement_manifest": nested,
        "replacement": {
            "receipt_path": receipt_path.relative_to(rulespec).as_posix(),
            "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        },
    }
    outer_path = rulespec / ".axiom/encoding-manifests/us/statutes/47/32.json"
    outer_path.parent.mkdir(parents=True, exist_ok=True)
    outer_path.write_text(json.dumps(outer) + "\n")
    (tmp_path / "source-bundle.json").write_text("[]\n")
    (tmp_path / "canonical-refresh-bundle.json").write_text("[]\n")
    (tmp_path / "primary-required-test-cases.json").write_text("[]\n")
    (tmp_path / "target-operation.txt").write_text("replace\n", encoding="ascii")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _copy_targeted_atomic_inputs(tmp_path, artifact)
    completed = subprocess.run(
        [
            sys.executable,
            "-",
            str(artifact / "context-manifest.json"),
            str(artifact / "apply-manifests.json"),
        ],
        cwd=tmp_path,
        input=script,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ATOMIC_SOURCE_JSON": json.dumps(
                {
                    "schema": "axiom-encode/atomic-source-transaction/v2",
                    "source_bundle": [],
                    "canonical_refresh_bundle": [],
                    "primary_required_test_cases": [],
                    "target_operation": "replace",
                }
            ),
            "CITATION": citation,
            "PYTHONPATH": str(ROOT / "src"),
            "REVIEW_FINDING": "",
            "REVIEW_FINDING_PRESENT": "false",
            "RUNNER_TEMP": str(tmp_path),
            "RULESPEC_CHECKOUT": "rulespec-us",
            "RULESPEC_REF": rulespec_ref,
            "REPLACE_LEGACY_RULESPEC_PATH": "us/statutes/47:32.yaml",
            "REPLACE_RULESPEC_PATH": "us/statutes/47/32.yaml",
        },
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("target_lane", "dependent_lane", "dependent_context_citation", "error"),
    [
        ("target", "dependent", None, None),
        (
            "dependent",
            "target",
            None,
            "signed context manifest is outside assigned target lane",
        ),
        (
            "target",
            "target",
            None,
            "signed context manifest is outside assigned dependent lane",
        ),
        (
            "target",
            "dependent",
            "us/regulation/42/435/555",
            "signed context manifest citation does not match",
        ),
    ],
)
def test_targeted_artifact_enforces_target_and_dependent_context_lanes(
    tmp_path: Path,
    target_lane: str,
    dependent_lane: str,
    dependent_context_citation: str | None,
    error: str | None,
) -> None:
    script = _targeted_package_script()

    rulespec = tmp_path / "rulespec-us"
    rulespec.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=rulespec, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=rulespec,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=rulespec, check=True)
    base = rulespec / "README.md"
    base.write_text("fixture\n")
    subprocess.run(["git", "add", "README.md"], cwd=rulespec, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=rulespec, check=True)
    rulespec_ref = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=rulespec, text=True
    ).strip()
    (tmp_path / "source-bundle.json").write_text("[]\n", encoding="utf-8")
    (tmp_path / "canonical-refresh-bundle.json").write_text("[]\n", encoding="utf-8")
    (tmp_path / "primary-required-test-cases.json").write_text("[]\n", encoding="utf-8")
    (tmp_path / "target-operation.txt").write_text("replace\n", encoding="ascii")

    target_citation = "us/regulation/42/435/555"
    dependent_citation = "us/regulation/42/435/559"
    second_dependent_citation = "us/regulation/42/435/561"
    generated_root = tmp_path / "generated"
    contexts: dict[str, bytes] = {}
    for citation, context_citation, lane, section, finding in (
        (
            target_citation,
            target_citation,
            target_lane,
            "555",
            "Preserve the target source.\n",
        ),
        (
            dependent_citation,
            dependent_context_citation or dependent_citation,
            dependent_lane,
            "559",
            "Preserve the dependent source.\n",
        ),
        (
            second_dependent_citation,
            second_dependent_citation,
            "dependent-2",
            "561",
            "Preserve the second dependent source.\n",
        ),
    ):
        context_payload = {
            "citation": context_citation,
            "review_findings_files": [
                {
                    "content": finding,
                    "sha256": hashlib.sha256(finding.encode()).hexdigest(),
                }
            ],
        }
        context_bytes = json.dumps(context_payload, sort_keys=True).encode()
        context_path = (
            generated_root
            / lane
            / "_eval_workspaces"
            / section
            / "context-manifest.json"
        )
        context_path.parent.mkdir(parents=True)
        context_path.write_bytes(context_bytes)
        contexts[citation] = context_bytes

        applied_manifest = {
            "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
            "citation": citation,
            "context_manifest_file": str(context_path),
            "context_manifest_sha256": hashlib.sha256(context_bytes).hexdigest(),
        }
        applied_path = (
            rulespec
            / ".axiom"
            / "encoding-manifests"
            / "regulations"
            / "42-cfr"
            / "435"
            / f"{section}.yaml.json"
        )
        applied_path.parent.mkdir(parents=True, exist_ok=True)
        applied_path.write_text(json.dumps(applied_manifest))

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    _copy_targeted_atomic_inputs(tmp_path, artifact)
    packaged_target = artifact / "context-manifest.json"
    packaged_inventory = artifact / "apply-manifests.json"
    completed = subprocess.run(
        [sys.executable, "-", str(packaged_target), str(packaged_inventory)],
        cwd=tmp_path,
        input=script,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CITATION": target_citation,
            "DEPENDENT_CITATION": dependent_citation,
            "DEPENDENT_REVIEW_FINDING": "Preserve the dependent source.",
            "DEPENDENT_REVIEW_FINDING_PRESENT": "true",
            "PYTHONPATH": str(ROOT / "src"),
            "SECOND_DEPENDENT_CITATION": second_dependent_citation,
            "SECOND_DEPENDENT_REVIEW_FINDING": (
                "Preserve the second dependent source."
            ),
            "SECOND_DEPENDENT_REVIEW_FINDING_PRESENT": "true",
            "REVIEW_FINDING": "Preserve the target source.",
            "REVIEW_FINDING_PRESENT": "true",
            "RUNNER_TEMP": str(tmp_path),
            "RULESPEC_CHECKOUT": "rulespec-us",
            "RULESPEC_REF": rulespec_ref,
        },
    )

    if error is not None:
        assert completed.returncode != 0
        assert error in completed.stderr
        return

    assert completed.returncode == 0, completed.stderr
    assert packaged_target.read_bytes() == contexts[target_citation]
    assert (artifact / "dependent-context-manifest.json").read_bytes() == contexts[
        dependent_citation
    ]
    assert (artifact / "dependent-2-context-manifest.json").read_bytes() == contexts[
        second_dependent_citation
    ]


def test_apply_signing_key_migration_workflow_is_main_only() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/migrate-apply-signing-key.yml").read_text()
    )
    trigger = workflow.get("on", workflow.get(True))
    assert set(trigger) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}

    job = workflow["jobs"]["migrate"]
    assert job["environment"] == "signing-key-migration"
    assert "github.ref == 'refs/heads/main'" in job["if"]
    steps = job["steps"]
    assert not any(
        step.get("uses", "").startswith("actions/checkout@") for step in steps
    )
    setup_go = next(
        step for step in steps if step.get("name") == "Install pinned Go toolchain"
    )
    assert setup_go["uses"] == (
        "actions/setup-go@924ae3a1cded613372ab5595356fb5720e22ba16"
    )
    assert setup_go["with"] == {"go-version": "1.26.1", "cache": False}

    encrypt_step = next(
        step for step in steps if step.get("name") == "Encrypt inherited signing key"
    )
    assert encrypt_step["env"]["APPLY_SIGNING_KEY"] == (
        "${{ secrets.AXIOM_ENCODE_APPLY_SIGNING_KEY }}"
    )
    command = encrypt_step["run"]
    assert "rsa_padding_mode:oaep" in command
    assert "rsa_oaep_md:sha256" in command
    assert "rsa_mgf1_md:sha256" in command
    assert "unset APPLY_SIGNING_KEY" in command
    assert '"$rsa_bits" -lt 3072' in command
    assert "base64.StdEncoding.Strict().DecodeString" in command
    assert "x509.ParsePKCS8PrivateKey" in command
    assert "bytes.Equal(derivedPublic, trustedPublic)" in command
    assert "sha256sum --check SHA256SUMS" in command
    assert 'rm -f "$plaintext"' in command
    assert 'echo "$APPLY_SIGNING_KEY"' not in command
    upload_step = next(
        step
        for step in steps
        if step.get("name") == "Upload encrypted migration artifact"
    )
    assert upload_step["with"]["path"] == (
        "${{ runner.temp }}/apply-signing-key-migration"
    )
    assert upload_step["with"]["retention-days"] == 1

    secret_steps = [
        step for step in steps if "APPLY_SIGNING_KEY" in (step.get("env") or {})
    ]
    assert secret_steps == [encrypt_step]


def _migration_openssl() -> Path:
    candidates = (
        Path("/opt/homebrew/opt/openssl@3/bin/openssl"),
        Path("/usr/local/opt/openssl@3/bin/openssl"),
        Path(shutil.which("openssl") or "/missing"),
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        version = subprocess.run(
            [candidate, "version"],
            check=False,
            capture_output=True,
            text=True,
        )
        if version.returncode == 0 and version.stdout.startswith("OpenSSL 3."):
            return candidate
    pytest.skip("OpenSSL 3 is required for the executable migration workflow test")


def _run_apply_key_migration(
    tmp_path: Path,
    *,
    run_name: str,
    signing_key: str,
    apply_public_key: str,
    recipient_public_key: bytes,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    go = Path(shutil.which("go") or "/opt/homebrew/bin/go")
    if not go.is_file():
        pytest.skip("Go is required for the executable migration workflow test")
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/migrate-apply-signing-key.yml").read_text()
    )
    command = next(
        step["run"]
        for step in workflow["jobs"]["migrate"]["steps"]
        if step.get("name") == "Encrypt inherited signing key"
    )
    runner_temp = tmp_path / run_name
    runner_temp.mkdir()
    shim_dir = runner_temp / "bin"
    shim_dir.mkdir()
    (shim_dir / "go").symlink_to(go)
    (shim_dir / "openssl").symlink_to(_migration_openssl())
    sha256sum = shim_dir / "sha256sum"
    sha256sum.write_text(
        "#!/usr/bin/env python3\n"
        "import hashlib, pathlib, sys\n"
        "if sys.argv[1:2] == ['--check']:\n"
        "    for line in pathlib.Path(sys.argv[2]).read_text().splitlines():\n"
        "        expected, name = line.split('  ', 1)\n"
        "        actual = hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()\n"
        "        if actual != expected:\n"
        "            raise SystemExit(f'{name}: FAILED')\n"
        "        print(f'{name}: OK')\n"
        "else:\n"
        "    for name in sys.argv[1:]:\n"
        "        digest = hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()\n"
        "        print(f'{digest}  {name}')\n"
    )
    sha256sum.chmod(0o700)
    completed = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "RUNNER_TEMP": str(runner_temp),
            "APPLY_SIGNING_KEY": signing_key,
            "APPLY_PUBLIC_KEY": apply_public_key,
            "RECIPIENT_PUBLIC_KEY": b64encode(recipient_public_key).decode("ascii"),
            "PATH": f"{shim_dir}{os.pathsep}{os.environ['PATH']}",
        },
    )
    return completed, runner_temp / "apply-signing-key-migration"


def test_apply_signing_key_migration_round_trip_and_artifact_allowlist(
    tmp_path: Path,
) -> None:
    seed = bytes(range(32))
    apply_public_key, private_key = _keypair(seed)
    recipient_private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=3072
    )
    recipient_public_key = recipient_private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    raw_seed = b64encode(seed).decode("ascii")
    pkcs8_pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode("ascii")
    serializations = {
        "raw-seed": (raw_seed, raw_seed),
        "raw-seed-whitespace": (f" \n{raw_seed}\n ", raw_seed),
        "pkcs8-pem": (pkcs8_pem, pkcs8_pem.strip()),
    }

    for run_name, (
        serialized_private_key,
        expected_private_key,
    ) in serializations.items():
        completed, artifact = _run_apply_key_migration(
            tmp_path,
            run_name=run_name,
            signing_key=serialized_private_key,
            apply_public_key=apply_public_key,
            recipient_public_key=recipient_public_key,
        )

        assert completed.returncode == 0, completed.stderr
        assert sorted(path.name for path in artifact.iterdir()) == [
            "SHA256SUMS",
            "apply-public-key.txt",
            "apply-signing-key.oaep-sha256.bin",
        ]
        checksum_manifest = (artifact / "SHA256SUMS").read_text()
        assert str(tmp_path) not in checksum_manifest
        for line in checksum_manifest.splitlines():
            expected_digest, relative_name = line.split("  ", 1)
            assert (
                hashlib.sha256((artifact / relative_name).read_bytes()).hexdigest()
                == expected_digest
            )
        decrypted = recipient_private_key.decrypt(
            (artifact / "apply-signing-key.oaep-sha256.bin").read_bytes(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        assert decrypted.decode("ascii") == expected_private_key

    escaped_public_pem = (
        private_key.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
        .replace("\n", "\\n")
    )
    escaped_public, artifact = _run_apply_key_migration(
        tmp_path,
        run_name="escaped-public-pem",
        signing_key=raw_seed,
        apply_public_key=escaped_public_pem,
        recipient_public_key=recipient_public_key,
    )
    assert escaped_public.returncode == 0, escaped_public.stderr
    decrypted = recipient_private_key.decrypt(
        (artifact / "apply-signing-key.oaep-sha256.bin").read_bytes(),
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    assert decrypted.decode("ascii") == raw_seed


def test_apply_signing_key_migration_rejects_mismatched_or_malformed_key(
    tmp_path: Path,
) -> None:
    seed = bytes(range(32))
    apply_public_key, _private_key = _keypair(seed)
    wrong_public_key, _wrong_private_key = _keypair(bytes(reversed(range(32))))
    recipient_private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=3072
    )
    recipient_public_key = recipient_private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    mismatched, _artifact = _run_apply_key_migration(
        tmp_path,
        run_name="mismatched",
        signing_key=b64encode(seed).decode("ascii"),
        apply_public_key=wrong_public_key,
        recipient_public_key=recipient_public_key,
    )
    malformed, _artifact = _run_apply_key_migration(
        tmp_path,
        run_name="malformed",
        signing_key="!" * 44,
        apply_public_key=apply_public_key,
        recipient_public_key=recipient_public_key,
    )
    pkcs8_pem = _private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode("ascii")
    trailing_data, _artifact = _run_apply_key_migration(
        tmp_path,
        run_name="trailing-data",
        signing_key=f"{pkcs8_pem}not-part-of-the-pem",
        apply_public_key=apply_public_key,
        recipient_public_key=recipient_public_key,
    )
    private_der = _private_key.private_bytes(
        serialization.Encoding.DER,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    internal_trailing_der = b64encode(private_der + b"X").decode("ascii")
    internal_trailing_pem = "\n".join(
        [
            "-----BEGIN PRIVATE KEY-----",
            *(
                internal_trailing_der[index : index + 64]
                for index in range(0, len(internal_trailing_der), 64)
            ),
            "-----END PRIVATE KEY-----",
        ]
    )
    trailing_der, _artifact = _run_apply_key_migration(
        tmp_path,
        run_name="trailing-der",
        signing_key=internal_trailing_pem,
        apply_public_key=apply_public_key,
        recipient_public_key=recipient_public_key,
    )

    canonical_seed = b64encode(seed).decode("ascii")
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    padding_index = alphabet.index(canonical_seed[-2])
    assert padding_index % 4 == 0
    noncanonical_seed = f"{canonical_seed[:-2]}{alphabet[padding_index + 1]}="
    noncanonical_private, _artifact = _run_apply_key_migration(
        tmp_path,
        run_name="noncanonical-private-base64",
        signing_key=noncanonical_seed,
        apply_public_key=apply_public_key,
        recipient_public_key=recipient_public_key,
    )
    public_padding_index = alphabet.index(apply_public_key[-2])
    assert public_padding_index % 4 == 0
    noncanonical_apply_public = (
        f"{apply_public_key[:-2]}{alphabet[public_padding_index + 1]}="
    )
    noncanonical_public, _artifact = _run_apply_key_migration(
        tmp_path,
        run_name="noncanonical-public-base64",
        signing_key=canonical_seed,
        apply_public_key=noncanonical_apply_public,
        recipient_public_key=recipient_public_key,
    )

    assert mismatched.returncode != 0
    assert "does not match the trusted apply public key" in mismatched.stderr
    assert malformed.returncode != 0
    assert "not strict base64 or PKCS8 PEM" in malformed.stderr
    assert trailing_data.returncode != 0
    assert "PEM contains trailing data" in trailing_data.stderr
    assert trailing_der.returncode != 0
    assert "contains trailing ASN.1 data" in trailing_der.stderr
    assert noncanonical_private.returncode != 0
    assert "not strict base64 or PKCS8 PEM" in noncanonical_private.stderr
    assert noncanonical_public.returncode != 0
    assert "not strict base64 Ed25519 material" in noncanonical_public.stderr


def test_apply_signing_key_migration_rejects_weak_recipient_rsa(
    tmp_path: Path,
) -> None:
    seed = bytes(range(32))
    apply_public_key, _private_key = _keypair(seed)
    weak_recipient = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    weak_public_key = weak_recipient.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    completed, _artifact = _run_apply_key_migration(
        tmp_path,
        run_name="weak-rsa",
        signing_key=b64encode(seed).decode("ascii"),
        apply_public_key=apply_public_key,
        recipient_public_key=weak_public_key,
    )

    assert completed.returncode != 0
    assert "must be RSA with at least 3072 bits" in completed.stderr
