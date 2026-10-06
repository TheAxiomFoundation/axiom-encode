"""Local, verification-only parity entrypoint for validate-rulespec CI.

This module deliberately owns no signing capability and never invokes an applying
command. Apart from caching a fetched immutable public release object at the
workflow-defined corpus path, it writes only temporary report inputs. Protected-supervisor calls
in CI are reproduced as direct subcommands under an explicit, library-level
corpus release verification key.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from axiom_encode import __version__
from axiom_encode.toolchain import (
    RuleSpecToolchain,
    load_rulespec_local_corpus_release,
    load_rulespec_toolchain,
    local_corpus_release_verification,
    verify_rulespec_validation_waiver_set,
)

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
WORKFLOW_RE = re.compile(
    r"TheAxiomFoundation/\.github/\.github/workflows/"
    r"validate-rulespec\.yml@(?P<sha>[0-9a-f]{40})$"
)
DEPENDENCY_INPUTS = {
    "encode": "axiom-encode-ref",
    "engine": "axiom-rules-engine-ref",
    "corpus": "axiom-corpus-ref",
    "rulespec_us": "rulespec-us-ref",
}
DEFAULT_RELEASE_BASE_URL = "https://pub-a8952f8657fc49fda358146ac001366c.r2.dev"


@dataclass(frozen=True, slots=True)
class WorkflowGateParameters:
    exclude_programs_from_money_atom_check: bool


@dataclass(frozen=True, slots=True)
class SupportedWorkflowPin:
    fixture: str
    gate_parameters: WorkflowGateParameters


SUPPORTED_WORKFLOW_PINS: dict[str, SupportedWorkflowPin] = {
    "615c1df9b9ace7deea84da65efd137f46f8bad2b": SupportedWorkflowPin(
        fixture="validate-rulespec-615c1df9.yml",
        gate_parameters=WorkflowGateParameters(
            exclude_programs_from_money_atom_check=True
        ),
    ),
    "34bcfab235c585c47292c95f51be1a4f4f91d29e": SupportedWorkflowPin(
        fixture="validate-rulespec-34bcfab2.yml",
        gate_parameters=WorkflowGateParameters(
            exclude_programs_from_money_atom_check=False
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class CallerConfig:
    path: Path
    workflow_sha: str
    refs: dict[str, str]
    validate_roots: str
    run_generated_guard: bool
    guard_programs_root: bool
    release_base_url: str = DEFAULT_RELEASE_BASE_URL
    run_pytest: bool = True
    run_money_atom_check: bool = True


@dataclass(frozen=True, slots=True)
class GateSpec:
    key: str
    name: str
    subcommand: str
    flags: tuple[str, ...]
    workflow_lines: str


# Stable order is part of the public parity contract.  Dynamic file operands
# are represented by placeholders; literal flags match the workflow run blocks.
CI_GATE_REGISTRY: tuple[GateSpec, ...] = (
    GateSpec(
        "repository_tests", "Run repository tests", "pytest", ("-q", "tests"), "646-653"
    ),
    GateSpec(
        "obsolete_files",
        "Reject obsolete generated files",
        "find",
        ("*.rac", "*.rac.test"),
        "655-665",
    ),
    GateSpec(
        "repository_layout",
        "Reject disallowed repository layout",
        "layout",
        (),
        "667-835",
    ),
    GateSpec(
        "validation_waivers",
        "Enforce validation waiver ratchet",
        "validation-waivers audit",
        (
            "--root",
            "{repo}",
            "--corpus-path",
            "{corpus}",
            "--protected-base",
            "{base-waivers}",
            "--changed-paths",
            "{changed-paths}",
            "--axiom-rules-engine-path",
            "{engine}",
        ),
        "837-908",
    ),
    GateSpec(
        "guard_generated",
        "Reject manual RuleSpec changes",
        "guard-generated",
        (
            "--repo",
            "{repo}",
            "--base-ref",
            "{base-ref}",
            "--head-ref",
            "HEAD",
            "--corpus-path",
            "{corpus}",
            "--expected-encoder-checkout",
            "{encode}",
        ),
        "910-935",
    ),
    GateSpec(
        "select_targets",
        "Select RuleSpec validation targets",
        "selection",
        ("--base-ref", "{base-ref}", "--roots", "{roots}"),
        "936-1108",
    ),
    GateSpec(
        "validate",
        "Validate RuleSpec YAML",
        "validate",
        (
            "{files}",
            "--skip-reviewers",
            "--corpus-path",
            "{corpus}",
            "--axiom-rules-engine-path",
            "{engine}",
        ),
        "1109-1166",
    ),
    GateSpec(
        "companion_tests",
        "Execute RuleSpec companion tests",
        "test",
        (
            "--root",
            "{jurisdiction}",
            "--axiom-rules-engine-path",
            "{engine}",
            "{tests}",
        ),
        "1167-1218",
    ),
    GateSpec(
        "proof_validate",
        "Validate RuleSpec proofs and claims",
        "proof-validate",
        ("{files}", "--corpus-path", "{corpus}"),
        "1219-1249",
    ),
    GateSpec(
        "money_atoms",
        "Require money proof atoms",
        "proof-validate",
        ("{all-files}", "--money-atoms-only", "--corpus-path", "{corpus}", "{ratchet}"),
        "1250-1302",
    ),
    GateSpec(
        "oracle_coverage",
        "Validate PolicyEngine oracle coverage classification",
        "oracle-coverage",
        (
            "--root",
            "{repo}",
            "--fail-on-unmapped",
            "--fail-on-untested-comparable",
            "--limit",
            "50",
        ),
        "1303-1313",
    ),
    GateSpec(
        "changed_oracle_coverage",
        "Validate changed PolicyEngine oracle coverage classification",
        "oracle-coverage",
        ("--root", "{repo}", "--json"),
        "1314-1419",
    ),
)

WORKFLOW_GATE_STEPS: dict[str, tuple[str, ...]] = {
    "Run repository tests": ("repository_tests",),
    "Reject obsolete generated files": ("obsolete_files",),
    "Reject disallowed repository layout": ("repository_layout",),
    "Enforce validation waiver ratchet": ("validation_waivers",),
    "Reject manual RuleSpec changes": ("guard_generated",),
    "Select RuleSpec validation targets": ("select_targets",),
    "Validate RuleSpec YAML": ("validate",),
    "Execute RuleSpec companion tests": ("companion_tests",),
    "Validate RuleSpec proofs and claims": ("proof_validate",),
    "Require money proof atoms": ("money_atoms",),
    "Validate PolicyEngine oracle coverage classification": ("oracle_coverage",),
    "Checkout changed-file oracle coverage classifier": ("changed_oracle_coverage",),
    "Install changed-file oracle coverage classifier": ("changed_oracle_coverage",),
    "Validate changed PolicyEngine oracle coverage classification": (
        "changed_oracle_coverage",
    ),
}


@dataclass(slots=True)
class GateResult:
    gate: str
    name: str
    status: str
    command: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    output: str = ""
    note: str | None = None


@dataclass(frozen=True, slots=True)
class DependencyMismatch:
    name: str
    head_sha: str
    pinned_sha: str

    def banner_line(self) -> str:
        return f"{self.name}: HEAD {self.head_sha} != pinned {self.pinned_sha}"


@dataclass(frozen=True, slots=True)
class Selection:
    mode: str
    rulespec_files: tuple[Path, ...]
    test_files: tuple[Path, ...]


def register_ci_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "ci", help="Run the local validate-rulespec CI parity gate sequence"
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--corpus-path", type=Path)
    parser.add_argument("--engine-path", type=Path)
    parser.add_argument("--rulespec-us-path", type=Path)
    parser.add_argument("--encode-path", type=Path)
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--roots", default=None)
    parser.add_argument("--corpus-release-public-key", required=True)
    parser.add_argument("--allow-ref-mismatch", action="store_true")
    parser.add_argument("--allow-encoder-mismatch", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--json", action="store_true")


def find_caller_workflow(repo: Path) -> CallerConfig:
    workflow_dir = repo / ".github" / "workflows"
    matches: list[CallerConfig] = []
    for path in sorted((*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml"))):
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid caller workflow {path}: {exc}") from exc
        jobs = payload.get("jobs", {}) if isinstance(payload, dict) else {}
        for job in jobs.values() if isinstance(jobs, dict) else ():
            if not isinstance(job, dict) or not isinstance(job.get("uses"), str):
                continue
            if job["uses"].endswith("validate-rulespec.yml@<pin-me>"):
                raise ValueError(
                    f"Caller {path}: validate-rulespec workflow has placeholder pin "
                    "<pin-me>; replace it with the reviewed full lowercase SHA"
                )
            match = WORKFLOW_RE.fullmatch(job["uses"])
            if match:
                matches.append(parse_caller_job(path, match.group("sha"), job))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one validate-rulespec caller under {workflow_dir}; "
            f"found {len(matches)}"
        )
    return matches[0]


def parse_caller_workflow(path: Path) -> CallerConfig:
    """Parse a caller fixture containing exactly one reusable-workflow job."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    jobs = payload.get("jobs", {}) if isinstance(payload, dict) else {}
    found = []
    for job in jobs.values() if isinstance(jobs, dict) else ():
        if isinstance(job, dict) and isinstance(job.get("uses"), str):
            if job["uses"].endswith("validate-rulespec.yml@<pin-me>"):
                raise ValueError(
                    f"Caller {path}: validate-rulespec workflow has placeholder pin "
                    "<pin-me>; replace it with the reviewed full lowercase SHA"
                )
            match = WORKFLOW_RE.fullmatch(job["uses"])
            if match:
                found.append(parse_caller_job(path, match.group("sha"), job))
    if len(found) != 1:
        raise ValueError(f"Expected one validate-rulespec caller in {path}")
    return found[0]


def parse_caller_job(
    path: Path, workflow_sha: str, job: dict[str, Any]
) -> CallerConfig:
    inputs = job.get("with")
    if not isinstance(inputs, dict):
        raise ValueError(f"Caller {path} has no with-inputs")
    refs: dict[str, str] = {}
    for dependency, input_name in DEPENDENCY_INPUTS.items():
        value = inputs.get(input_name)
        if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
            raise ValueError(
                f"Caller {path}: {input_name} must be a full lowercase SHA"
            )
        refs[dependency] = value
    return CallerConfig(
        path=path,
        workflow_sha=workflow_sha,
        refs=refs,
        validate_roots=str(inputs.get("validate-roots", "auto")),
        run_generated_guard=bool(inputs.get("run-generated-guard", True)),
        guard_programs_root=bool(inputs.get("guard-programs-root", False)),
        release_base_url=str(
            inputs.get("corpus-release-base-url", DEFAULT_RELEASE_BASE_URL)
        ),
        run_pytest=bool(inputs.get("run-pytest", True)),
        run_money_atom_check=bool(inputs.get("run-money-atom-check", True)),
    )


def resolve_dependency_paths(args: argparse.Namespace, repo: Path) -> dict[str, Path]:
    running_checkout = Path(__file__).resolve().parents[2]
    return {
        "encode": (args.encode_path or running_checkout).expanduser().resolve(),
        "engine": (args.engine_path or repo.parent / "axiom-rules-engine")
        .expanduser()
        .resolve(),
        "corpus": (args.corpus_path or repo.parent / "axiom-corpus")
        .expanduser()
        .resolve(),
        "rulespec_us": (args.rulespec_us_path or repo.parent / "rulespec-us")
        .expanduser()
        .resolve(),
    }


def _git(
    repo: Path, *arguments: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def verify_dependency_checkout(
    name: str,
    path: Path,
    pin: str,
    caller: Path,
    *,
    allow_ref_mismatch: bool,
) -> DependencyMismatch | None:
    if not path.is_dir():
        raise ValueError(f"{name} checkout does not exist: {path}")
    if _git(path, "cat-file", "-e", f"{pin}^{{commit}}", check=False).returncode:
        raise ValueError(
            f"{name} checkout {path} does not contain pinned commit {pin} from {caller}"
        )
    protected = "refs/remotes/origin/main"
    if _git(path, "show-ref", "--verify", "--quiet", protected, check=False).returncode:
        raise ValueError(
            f"{name} checkout {path} has no origin/main for authenticating {pin} from {caller}"
        )
    if _git(
        path, "merge-base", "--is-ancestor", pin, protected, check=False
    ).returncode:
        raise ValueError(
            f"{name} pinned commit {pin} from {caller} is not an ancestor of origin/main in {path}"
        )
    head = _git(path, "rev-parse", "HEAD").stdout.strip()
    # A checkout at the pinned HEAD but with local modifications is not the
    # pinned tree CI will use; treat dirtiness as a mismatch in its own right.
    dirty = bool(_git(path, "status", "--porcelain", check=False).stdout.strip())
    if head == pin and not dirty:
        return None
    described_head = f"{head} (dirty worktree)" if dirty else head
    mismatch = DependencyMismatch(name, described_head, pin)
    warning = f"REF MISMATCH: {mismatch.banner_line()} declared by {caller}"
    if not allow_ref_mismatch:
        raise ValueError(warning + "; pass --allow-ref-mismatch to continue")
    return mismatch


def verify_ambient_encoder(
    encode_pin: str,
    pinned_version: str,
    caller: Path,
    *,
    allow_encoder_mismatch: bool,
) -> DependencyMismatch | None:
    """Bind the imported encoder to the caller pin, preferring source identity."""

    source_checkout = Path(__file__).resolve().parents[2]
    result = _git(source_checkout, "rev-parse", "HEAD", check=False)
    head = result.stdout.strip()
    if result.returncode == 0 and SHA_RE.fullmatch(head):
        mismatch = (
            None
            if head == encode_pin
            else DependencyMismatch("ambient-encoder", head, encode_pin)
        )
    else:
        # Version equality cannot establish source identity: an installed
        # wheel with no resolvable git HEAD may differ from the pinned
        # encoder at an equal version string. Unresolvable identity is
        # always a mismatch (fail-closed without the flag).
        mismatch = DependencyMismatch(
            "ambient-encoder",
            f"unresolvable ({__version__})",
            encode_pin,
        )
    if mismatch is not None and not allow_encoder_mismatch:
        raise ValueError(
            f"ENCODER MISMATCH: {mismatch.banner_line()} declared by {caller}; "
            "run this tool from the pinned encode checkout for exact parity, or "
            "pass --allow-encoder-mismatch for a qualified development verdict"
        )
    return mismatch


def encoder_version_at_pin(path: Path, pin: str) -> str:
    """Require consistent encoder version metadata at the caller's exact pin."""

    def show(name: str) -> str:
        result = _git(path, "show", f"{pin}:{name}", check=False)
        if result.returncode:
            raise ValueError(f"Pinned encoder {pin} has no {name}")
        return result.stdout

    pyproject = tomllib.loads(show("pyproject.toml"))["project"]["version"]
    package_match = re.search(
        r'(?m)^__version__\s*=\s*"([^"]+)"',
        show("src/axiom_encode/__init__.py"),
    )
    lock_match = re.search(
        r'(?ms)^name = "axiom-encode"\nversion = "([^"]+)"', show("uv.lock")
    )
    package = package_match.group(1) if package_match else None
    lock = lock_match.group(1) if lock_match else None
    if not isinstance(pyproject, str) or pyproject != package or pyproject != lock:
        raise ValueError(
            f"Pinned encoder {pin} has inconsistent version metadata "
            f"(pyproject={pyproject}, package={package}, lock={lock})"
        )
    return pyproject


def acquire_release_object(
    toolchain: RuleSpecToolchain,
    corpus_path: Path,
    base_url: str,
    *,
    offline: bool,
    fetcher: Callable[[str], bytes] | None = None,
) -> Path:
    destination = (
        corpus_path
        / "releases"
        / toolchain.corpus_release
        / f"{toolchain.corpus_release_content_sha256}.json"
    )
    if destination.is_file():
        return destination
    if offline:
        raise ValueError(
            f"--offline requires pinned corpus release object: {destination}"
        )
    url = f"{base_url.rstrip('/')}/releases/{toolchain.corpus_release}/{toolchain.corpus_release_content_sha256}.json"
    fetch = fetcher or (lambda value: urllib.request.urlopen(value, timeout=30).read())
    raw = fetch(url)
    try:
        payload = json.loads(raw)
        content = payload["content"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Corpus release acquisition error: invalid JSON: {exc}"
        ) from exc
    actual = hashlib.sha256(
        json.dumps(
            content, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()
    if payload.get("release") != toolchain.corpus_release:
        raise ValueError("Corpus release acquisition error: release name mismatch")
    if (
        payload.get("content_sha256") != actual
        or actual != toolchain.corpus_release_content_sha256
    ):
        raise ValueError(
            f"Corpus release acquisition error: content sha256 mismatch ({actual} != {toolchain.corpus_release_content_sha256})"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_bytes(raw)
    temporary.replace(destination)
    return destination


def authenticate_release_provenance(
    release_path: Path,
    corpus_path: Path,
    corpus_pin: str,
    caller: Path,
) -> str:
    try:
        payload = json.loads(release_path.read_text(encoding="utf-8"))
        commit = payload["content"]["git"]["commit"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Signed corpus release provenance is missing: {exc}") from exc
    if not isinstance(commit, str) or SHA_RE.fullmatch(commit) is None:
        raise ValueError("Signed release provenance is not a full lowercase commit SHA")
    if _git(
        corpus_path, "cat-file", "-e", f"{commit}^{{commit}}", check=False
    ).returncode:
        raise ValueError(f"Signed corpus release provenance commit is absent: {commit}")
    if _git(
        corpus_path,
        "merge-base",
        "--is-ancestor",
        commit,
        "refs/remotes/origin/main",
        check=False,
    ).returncode:
        raise ValueError(
            f"Signed corpus release provenance {commit} is not an ancestor of corpus origin/main"
        )
    if _git(
        corpus_path, "merge-base", "--is-ancestor", commit, corpus_pin, check=False
    ).returncode:
        raise ValueError(
            f"Signed corpus release provenance {commit} is not contained in "
            f"caller corpus pin {corpus_pin} from {caller}"
        )
    return commit


def verify_toolchain_base_binding(repo: Path, base_ref: str) -> None:
    """Mirror the workflow's migrated-base removal guard."""

    result = _git(repo, "show", f"{base_ref}:.axiom/toolchain.toml", check=False)
    if result.returncode:
        return
    try:
        payload = tomllib.loads(result.stdout)
    except tomllib.TOMLDecodeError:
        return
    table = payload.get("toolchain")
    expected = {
        "axiom_corpus_release",
        "axiom_corpus_release_content_sha256",
        "validation_waiver_set_sha256",
    }
    if isinstance(table, dict) and set(table) == expected:
        path = repo / ".axiom" / "toolchain.toml"
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                ".axiom/toolchain.toml cannot be removed once a base branch uses it"
            )


def resolve_roots(repo: Path, raw: str) -> tuple[str, ...]:
    if raw != "auto":
        return tuple(item for item in raw.split() if item)
    candidates = []
    for child in sorted(repo.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if re.fullmatch(r"[a-z]{2}(?:-[a-z0-9-]+)*", child.name) and any(
            (child / root).is_dir()
            for root in ("statutes", "regulations", "policies", "legislation")
        ):
            candidates.append(child.name)
    if not candidates:
        for name in ("statutes", "regulations", "policies", "legislation"):
            if (repo / name).is_dir():
                candidates.append(name)
    return tuple(candidates)


def _changed_paths(repo: Path, base_ref: str) -> tuple[str, ...]:
    result = _git(
        repo,
        "diff",
        "--name-only",
        "--no-renames",
        "--diff-filter=ACMRTD",
        base_ref,
        "HEAD",
    )
    return tuple(line for line in result.stdout.splitlines() if line)


def select_targets(repo: Path, base_ref: str, roots: Sequence[str]) -> Selection:
    changed = _changed_paths(repo, base_ref)
    mode = "changed"
    if any(path.startswith(".github/workflows/") for path in changed):
        mode = "full-toolchain-bump"
    elif ".axiom/toolchain.toml" in changed:
        before = _git(repo, "show", f"{base_ref}:.axiom/toolchain.toml", check=False)
        try:
            old = tomllib.loads(before.stdout).get("toolchain", {})
            new = tomllib.loads((repo / ".axiom/toolchain.toml").read_text()).get(
                "toolchain", {}
            )
            differences = {
                key for key in set(old) | set(new) if old.get(key) != new.get(key)
            }
        except (OSError, tomllib.TOMLDecodeError):
            differences = {"invalid"}
        if differences and differences <= {
            "axiom_corpus_release",
            "axiom_corpus_release_content_sha256",
        }:
            mode = "changed-corpus-toolchain-bump"
        else:
            mode = "full-toolchain-bump"
    rules: set[Path] = set()
    tests: set[Path] = set()

    def add(path_text: str) -> None:
        if "/programs/" in f"/{path_text}/":
            return
        path = repo / path_text
        if path_text.endswith(".test.yaml"):
            tests.add(path) if path.is_file() else None
            module = repo / f"{path_text[:-10]}.yaml"
            rules.add(module) if module.is_file() else None
        elif path_text.endswith(".test.yml"):
            tests.add(path) if path.is_file() else None
            module = repo / f"{path_text[:-9]}.yml"
            rules.add(module) if module.is_file() else None
        elif path_text.endswith((".yaml", ".yml")):
            rules.add(path) if path.is_file() else None
            suffix = ".test.yaml" if path_text.endswith(".yaml") else ".test.yml"
            companion = path.with_name(path.name.rsplit(".", 1)[0] + suffix)
            tests.add(companion) if companion.is_file() else None

    if mode == "full-toolchain-bump":
        for root in roots:
            if (repo / root).is_dir():
                for path in (repo / root).rglob("*.y*ml"):
                    add(path.relative_to(repo).as_posix())
    else:
        for path in changed:
            if any(path.startswith(f"{root}/") for root in roots):
                add(path)
    return Selection(mode, tuple(sorted(rules)), tuple(sorted(tests)))


def _run_cli(
    arguments: Sequence[str], *, environment: dict[str, str] | None = None
) -> tuple[int, str]:
    from axiom_encode import cli

    output = io.StringIO()
    old_argv = sys.argv
    old_environment = {key: os.environ.get(key) for key in environment or {}}
    try:
        sys.argv = ["axiom-encode", *arguments]
        if environment:
            os.environ.update(environment)
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                cli.main()
            except SystemExit as exc:
                return int(exc.code or 0), output.getvalue()
            except Exception as exc:  # keep running later gates for local iteration
                print(f"{type(exc).__name__}: {exc}", file=output)
                return 1, output.getvalue()
    finally:
        sys.argv = old_argv
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return 0, output.getvalue()


def _run_pinned_cli(
    encode_path: Path, pin: str, arguments: Sequence[str]
) -> tuple[int, str]:
    """Execute classifier code from the caller's exact local encoder pin."""

    expected_oracles = _encoder_oracles_pin(encode_path, pin)
    installed_oracles = _installed_oracles_pin()
    if expected_oracles != installed_oracles:
        return (
            1,
            "Pinned changed-file classifier requires axiom-oracles "
            f"{expected_oracles}, but the local runtime has {installed_oracles or 'no VCS pin'}. "
            "Install the pinned encoder dependencies before running ci.\n",
        )

    with tempfile.TemporaryDirectory(prefix="axiom-ci-classifier-") as temp_name:
        archive = subprocess.run(
            ["git", "-C", str(encode_path), "archive", "--format=tar", pin],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        root = Path(temp_name)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            bundle.extractall(root, filter="data")
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(root / "src") + (
            os.pathsep + existing if existing else ""
        )
        result = subprocess.run(
            [sys.executable, "-m", "axiom_encode.cli", *arguments],
            cwd=root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return result.returncode, result.stdout


def _encoder_oracles_pin(encode_path: Path, pin: str) -> str:
    payload = _git(encode_path, "show", f"{pin}:pyproject.toml").stdout
    match = re.search(
        r'axiom-oracles\s+@\s+git\+https://github\.com/[^"@]+@([0-9a-f]{40})',
        payload,
    )
    if match is None:
        raise ValueError(f"Pinned encoder {pin} has no exact axiom-oracles dependency")
    return match.group(1)


def _installed_oracles_pin() -> str | None:
    """Return the runtime's exact axiom-oracles VCS commit, never its version."""

    try:
        raw = importlib.metadata.distribution("axiom-oracles").read_text(
            "direct_url.json"
        )
        payload = json.loads(raw or "null")
        commit = payload.get("vcs_info", {}).get("commit_id")
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError, TypeError):
        return None
    return commit if isinstance(commit, str) and SHA_RE.fullmatch(commit) else None


def _run_process(arguments: Sequence[str], cwd: Path) -> tuple[int, str]:
    result = subprocess.run(
        arguments,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.returncode, result.stdout


def _result(
    spec: GateSpec,
    code: int,
    output: str,
    command: Sequence[str],
    *,
    note: str | None = None,
) -> GateResult:
    failures = (
        [] if code == 0 else [line for line in output.splitlines() if line.strip()]
    )
    return GateResult(
        spec.key,
        spec.name,
        "PASS" if code == 0 else "FAIL",
        list(command),
        failures,
        output,
        note,
    )


def _obsolete_gate(repo: Path) -> tuple[int, str]:
    matches = [
        str(path.relative_to(repo))
        for path in repo.rglob("*")
        if path.is_file()
        and (path.name.endswith(".rac") or path.name.endswith(".rac.test"))
        and not any(
            part in {".git", "_axiom", ".venv", ".pytest_cache"}
            for part in path.relative_to(repo).parts
        )
    ]
    return (1, "\n".join(matches)) if matches else (0, "No obsolete generated files.\n")


def _layout_gate(repo: Path, roots: Sequence[str]) -> tuple[int, str]:
    config_path = repo / ".axiom" / "repository-structure.yaml"
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(config, dict):
            return 1, f"{config_path} must contain a mapping"
        if config.get("version") != 1:
            return 1, f"{config_path} must set version: 1"
        for key in ("allowed_root_directories", "allowed_root_files"):
            value = config.get(key)
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                return 1, f"{key} must be a list of strings"
        declared_rules = config.get("path_rules")
        if not isinstance(declared_rules, list) or not declared_rules:
            return 1, "path_rules must be a non-empty list"
        for index, rule in enumerate(declared_rules, start=1):
            if not isinstance(rule, dict):
                return 1, f"path_rules[{index}] must be a mapping"
            patterns = rule.get("patterns")
            if (
                not isinstance(patterns, list)
                or not patterns
                or not all(isinstance(pattern, str) for pattern in patterns)
            ):
                return (
                    1,
                    f"path_rules[{index}].patterns must be a non-empty list of strings",
                )
            for key in ("allow_extensions", "allow_filenames"):
                value = rule.get(key, [])
                if not isinstance(value, list) or not all(
                    isinstance(item, str) for item in value
                ):
                    return 1, f"path_rules[{index}].{key} must be a list of strings"
        tracked = _git(repo, "ls-files", "-z").stdout.split("\0")
        problems = []
        root_dirs = {
            item.strip("/") for item in config.get("allowed_root_directories", [])
        }
        root_files = {item.strip("/") for item in config.get("allowed_root_files", [])}
        rules = config.get("path_rules", [])
        for path in filter(None, tracked):
            parts = PurePosixPath(path).parts
            if len(parts) == 1:
                if path not in root_files:
                    problems.append(f"{path}: top-level file is not allowed")
                continue
            if parts[0] not in root_dirs:
                problems.append(
                    f"{path}: top-level directory {parts[0]}/ is not allowed"
                )
                continue
            rule = next(
                (
                    item
                    for item in rules
                    if any(
                        fnmatch.fnmatchcase(path, pattern.strip("/"))
                        for pattern in item.get("patterns", [])
                    )
                ),
                None,
            )
            if rule is None:
                problems.append(f"{path}: no path rule matched")
            elif PurePosixPath(path).name not in rule.get(
                "allow_filenames", []
            ) and PurePosixPath(path).suffix not in rule.get("allow_extensions", []):
                problems.append(
                    f"{path}: file name/extension is not allowed by matched path rule"
                )
        return (
            (1, "\n".join(problems))
            if problems
            else (0, f"Repository layout matches {config_path}.\n")
        )
    problems = [
        name for name in ("statute", "regulation", "policy") if (repo / name).exists()
    ]
    for path in repo.rglob("*.y*ml"):
        relative = path.relative_to(repo)
        if any(
            part in {".git", "_axiom", ".venv", ".pytest_cache", ".github"}
            for part in relative.parts
        ):
            continue
        text = relative.as_posix()
        if path.name in {"parameters.yaml", "tests.yaml"} or (
            relative.parts and relative.parts[0] == "tests"
        ):
            problems.append(text)
            continue
        if path.name in {"known-dangling.yaml", "known-validation-gaps.yaml"}:
            continue
        if not any(text.startswith(f"{root}/") for root in roots):
            problems.append(text)
    return (
        (1, "\n".join(problems)) if problems else (0, "Repository layout is allowed.\n")
    )


def _classifier_gate(
    repo: Path,
    selection: Selection,
    encode_path: Path,
    encode_pin: str,
) -> tuple[int, str]:
    if not selection.rulespec_files:
        return 0, "No changed RuleSpec YAML files selected for oracle coverage.\n"
    code, output = _run_pinned_cli(
        encode_path,
        encode_pin,
        ["oracle-coverage", "--root", str(repo), "--json"],
    )
    if code:
        return code, output
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        return 1, f"Changed oracle coverage emitted invalid JSON: {exc}\n{output}"
    if repo.name.startswith("rulespec-be"):
        return 0, "Changed PolicyEngine oracle coverage skipped for Belgium.\n"
    changed = {
        f"{repo.name}/{path.relative_to(repo).as_posix()}"
        for path in selection.rulespec_files
    }
    failures = []
    items = [item for item in payload.get("items", []) if item.get("file") in changed]
    for item in items:
        if item.get("status") == "unmapped":
            failures.append(f"{item.get('legal_id')}: unmapped")
        elif item.get("status") == "comparable" and not item.get("tested"):
            failures.append(
                f"{item.get('legal_id')}: comparable but not covered by companion tests"
            )
    if failures:
        return 1, "Changed PolicyEngine oracle coverage is incomplete.\n" + "\n".join(
            f"- {failure}" for failure in failures
        )
    return (
        0,
        f"Changed PolicyEngine oracle coverage passed for {len(items)} output(s).\n",
    )


def execute_gates(
    args: argparse.Namespace,
    caller: CallerConfig,
    paths: dict[str, Path],
    roots: tuple[str, ...],
) -> list[GateResult]:
    repo = args.repo.resolve()
    selection = select_targets(repo, args.base_ref, roots)
    results: list[GateResult] = []
    specs = {spec.key: spec for spec in CI_GATE_REGISTRY}
    gate_parameters = SUPPORTED_WORKFLOW_PINS[caller.workflow_sha].gate_parameters
    if (
        caller.run_pytest
        and (repo / "tests").is_dir()
        and any(
            path.is_file()
            and (path.name.startswith("test_") or path.name.endswith("_test.py"))
            for path in (repo / "tests").rglob("*.py")
        )
    ):
        command = [sys.executable, "-m", "pytest", "-q", "tests"]
        results.append(
            _result(specs["repository_tests"], *_run_process(command, repo), command)
        )
    else:
        results.append(
            _result(
                specs["repository_tests"],
                0,
                "No Python tests found; skipping pytest.\n",
                [],
            )
        )
    results.append(
        _result(
            specs["obsolete_files"],
            *_obsolete_gate(repo),
            ["find", ".", "*.rac", "*.rac.test"],
        )
    )
    results.append(
        _result(
            specs["repository_layout"],
            # Workflow shard resolution always permits programs/: it is a
            # validation root when guarded, otherwise an allowed extra root.
            *_layout_gate(repo, (*roots, "sources", "programs")),
            ["layout"],
        )
    )
    with tempfile.TemporaryDirectory(prefix="axiom-ci-") as temp_name:
        temp = Path(temp_name)
        protected = temp / "protected-known-validation-gaps.yaml"
        changed = temp / "waiver-changed-paths.txt"
        base = _git(
            repo, "show", f"{args.base_ref}:known-validation-gaps.yaml", check=False
        )
        protected.write_text(base.stdout)
        changed.write_text("\n".join(_changed_paths(repo, args.base_ref)) + "\n")
        waiver_command = [
            "validation-waivers",
            "audit",
            "--root",
            str(repo),
            "--corpus-path",
            str(paths["corpus"]),
            "--protected-base",
            str(protected),
            "--changed-paths",
            str(changed),
            "--axiom-rules-engine-path",
            str(paths["engine"]),
        ]
        code, output = (
            (
                1,
                "ValidationWaiverBaseMissing: protected base must contain known-validation-gaps.yaml\n",
            )
            if base.returncode
            else _run_cli(waiver_command)
        )
        results.append(
            _result(
                specs["validation_waivers"],
                code,
                output,
                waiver_command,
                note="library-level verification; no signing capability acquired",
            )
        )
        if caller.run_generated_guard:
            guard_command = [
                "guard-generated",
                "--repo",
                str(repo),
                "--base-ref",
                args.base_ref,
                "--head-ref",
                "HEAD",
                "--corpus-path",
                str(paths["corpus"]),
                "--expected-encoder-checkout",
                str(paths["encode"]),
            ]
            results.append(
                _result(
                    specs["guard_generated"],
                    *_run_cli(guard_command),
                    guard_command,
                    note="library-level verification; --apply is never available to ci",
                )
            )
        else:
            results.append(
                _result(
                    specs["guard_generated"],
                    0,
                    "Disabled by caller run-generated-guard.\n",
                    [],
                )
            )
    selection_output = f"RuleSpec validation mode: {selection.mode}\n" + "\n".join(
        f"- {path.relative_to(repo)}" for path in selection.rulespec_files
    )
    results.append(
        _result(
            specs["select_targets"],
            0,
            selection_output,
            ["selection", "--base-ref", args.base_ref, "--roots", *roots],
        )
    )
    skipped = set()
    gaps_path = repo / "known-validation-gaps.yaml"
    if gaps_path.is_file():
        gaps = yaml.safe_load(gaps_path.read_text()) or {}
        entries = gaps.get("validate_failures", {}) if isinstance(gaps, dict) else {}
        skipped = set(entries if isinstance(entries, (dict, list)) else ())
    validate_failures, validate_output = [], []
    for path in selection.rulespec_files:
        if path.relative_to(repo).as_posix() in skipped:
            continue
        command = [
            "validate",
            str(path),
            "--skip-reviewers",
            "--corpus-path",
            str(paths["corpus"]),
            "--axiom-rules-engine-path",
            str(paths["engine"]),
        ]
        code, output = _run_cli(
            command,
            environment={
                "AXIOM_RULESPEC_REPO_ROOTS": os.pathsep.join(
                    (str(repo), str(paths["rulespec_us"]))
                )
            },
        )
        validate_output.append(output)
        if code:
            validate_failures.append(path.relative_to(repo).as_posix())
    results.append(
        GateResult(
            "validate",
            specs["validate"].name,
            "FAIL" if validate_failures else "PASS",
            ["validate", "{selected files}"],
            validate_failures,
            "".join(validate_output),
            "direct subcommand under library-level release verification",
        )
    )
    test_failures, test_output = [], []
    grouped: dict[str, list[Path]] = {}
    for path in selection.test_files:
        relative = path.relative_to(repo)
        module = (
            relative.as_posix()
            .replace(".test.yaml", ".yaml")
            .replace(".test.yml", ".yml")
        )
        if module not in skipped:
            grouped.setdefault(relative.parts[0], []).append(relative)
    for jurisdiction, files in grouped.items():
        command = [
            "test",
            "--root",
            str(repo / jurisdiction),
            "--axiom-rules-engine-path",
            str(paths["engine"]),
            *(str(Path(*file.parts[1:])) for file in files),
        ]
        code, output = _run_cli(
            command,
            environment={
                "AXIOM_RULESPEC_REPO_ROOTS": os.pathsep.join(
                    (str(repo), str(paths["rulespec_us"]))
                )
            },
        )
        test_output.append(output)
        if code:
            test_failures.extend(file.as_posix() for file in files)
    results.append(
        GateResult(
            "companion_tests",
            specs["companion_tests"].name,
            "FAIL" if test_failures else "PASS",
            ["test", "{selected tests}"],
            test_failures,
            "".join(test_output),
        )
    )
    proof_failures, proof_output = [], []
    for path in selection.rulespec_files:
        command = ["proof-validate", str(path), "--corpus-path", str(paths["corpus"])]
        code, output = _run_cli(command)
        proof_output.append(output)
        if code:
            proof_failures.append(path.relative_to(repo).as_posix())
    results.append(
        GateResult(
            "proof_validate",
            specs["proof_validate"].name,
            "FAIL" if proof_failures else "PASS",
            ["proof-validate", "{selected files}"],
            proof_failures,
            "".join(proof_output),
            "direct subcommand under library-level release verification",
        )
    )
    all_rules = tuple(
        sorted(
            path
            for root in roots
            if (repo / root).is_dir()
            for path in (repo / root).rglob("*.y*ml")
            if ".test." not in path.name
            and (
                not gate_parameters.exclude_programs_from_money_atom_check
                or "/programs/" not in path.as_posix()
            )
        )
    )
    if caller.run_money_atom_check and all_rules:
        command = [
            "proof-validate",
            *map(str, all_rules),
            "--money-atoms-only",
            "--corpus-path",
            str(paths["corpus"]),
        ]
        if (repo / "known-missing-money-atoms.yaml").is_file():
            command += ["--ratchet-file", str(repo / "known-missing-money-atoms.yaml")]
        results.append(
            _result(
                specs["money_atoms"],
                *_run_cli(command),
                command,
                note="direct subcommand under library-level release verification",
            )
        )
    else:
        results.append(
            _result(
                specs["money_atoms"],
                0,
                "No RuleSpec YAML files found or gate disabled.\n",
                [],
            )
        )
    if selection.mode == "full-toolchain-bump":
        command = [
            "oracle-coverage",
            "--root",
            str(repo),
            "--fail-on-unmapped",
            "--fail-on-untested-comparable",
            "--limit",
            "50",
        ]
        results.append(_result(specs["oracle_coverage"], *_run_cli(command), command))
        results.append(
            _result(
                specs["changed_oracle_coverage"],
                0,
                "Not used for full-toolchain-bump mode.\n",
                [],
            )
        )
    else:
        results.append(
            _result(
                specs["oracle_coverage"],
                0,
                "Full oracle coverage is not selected for changed-file mode.\n",
                [],
            )
        )
        results.append(
            _result(
                specs["changed_oracle_coverage"],
                *_classifier_gate(
                    repo,
                    selection,
                    paths["encode"],
                    caller.refs["encode"],
                ),
                # This is deliberately the already-authenticated local checkout,
                # archived at the caller pin; no fresh checkout or ambient runtime
                # classifier is used.
                ["oracle-coverage", "--root", str(repo), "--json"],
                note=(
                    "classifier executed from encode pin "
                    f"{caller.refs['encode']} in {paths['encode']}"
                ),
            )
        )
    return results


def run_ci(args: argparse.Namespace) -> int:
    repo = args.repo.expanduser().resolve()
    args.repo = repo
    mismatches: list[DependencyMismatch] = []
    try:
        caller = find_caller_workflow(repo)
        if caller.workflow_sha not in SUPPORTED_WORKFLOW_PINS:
            supported = ", ".join(SUPPORTED_WORKFLOW_PINS)
            raise ValueError(
                f"Unsupported validate-rulespec workflow pin {caller.workflow_sha} "
                f"in {caller.path}; this axiom-encode release implements "
                f"these pins: {supported}"
            )
        verify_toolchain_base_binding(repo, args.base_ref)
        toolchain = load_rulespec_toolchain(repo)
        verify_rulespec_validation_waiver_set(repo)
        paths = resolve_dependency_paths(args, repo)
        for name, path in paths.items():
            mismatch = verify_dependency_checkout(
                name,
                path,
                caller.refs[name],
                caller.path,
                allow_ref_mismatch=args.allow_ref_mismatch,
            )
            if mismatch:
                mismatches.append(mismatch)
        pinned_version = encoder_version_at_pin(paths["encode"], caller.refs["encode"])
        encoder_mismatch = verify_ambient_encoder(
            caller.refs["encode"],
            pinned_version,
            caller.path,
            allow_encoder_mismatch=getattr(args, "allow_encoder_mismatch", False),
        )
        if encoder_mismatch:
            mismatches.append(encoder_mismatch)
        release_path = acquire_release_object(
            toolchain, paths["corpus"], caller.release_base_url, offline=args.offline
        )
        authenticate_release_provenance(
            release_path,
            paths["corpus"],
            caller.refs["corpus"],
            caller.path,
        )
        roots = resolve_roots(repo, args.roots or caller.validate_roots)
        if caller.guard_programs_root:
            roots = (*roots, "programs")
        if not roots:
            raise ValueError("No validation roots resolved")
        with local_corpus_release_verification(args.corpus_release_public_key):
            # Authenticate the release signature and its complete artifact
            # inventory even when changed-file selection produces no later
            # corpus-consuming gate.
            load_rulespec_local_corpus_release(repo, paths["corpus"])
            results = execute_gates(args, caller, paths, roots)
    except Exception as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "passed": False,
                        "verdict": "FAIL",
                        "dependency_mismatches": [asdict(item) for item in mismatches],
                        "resolution_error": str(exc),
                        "gates": [],
                    },
                    indent=2,
                )
            )
        else:
            for mismatch in mismatches:
                print(f"WARNING: {mismatch.banner_line()}", file=sys.stderr)
            print(f"axiom-encode ci resolution failed: {exc}", file=sys.stderr)
        return 1
    gates_passed = all(result.status == "PASS" for result in results)
    verdict = ci_verdict(gates_passed, mismatches)
    if args.json:
        print(
            json.dumps(
                {
                    "passed": verdict == "PASS",
                    "verdict": verdict,
                    "caller": str(caller.path),
                    "workflow_sha": caller.workflow_sha,
                    "dependency_mismatches": [asdict(item) for item in mismatches],
                    "roots": roots,
                    "gates": [asdict(result) for result in results],
                },
                indent=2,
            )
        )
    else:
        for mismatch in mismatches:
            print(f"WARNING: {mismatch.banner_line()}", file=sys.stderr)
        print(f"validate-rulespec caller: {caller.path} @ {caller.workflow_sha}")
        for result in results:
            print(f"{result.status:4} {result.name}")
            for failure in result.failures:
                print(f"     - {failure}")
            if result.note:
                print(f"     {result.note}")
        print(f"ci parity result: {verdict}")
        if mismatches:
            print("MISMATCHED DEPENDENCIES:")
            for mismatch in mismatches:
                print(f"  - {mismatch.banner_line()}")
    return (
        0 if verdict == "PASS" else 3 if verdict == "PASS-WITH-MISMATCHED-DEPS" else 1
    )


def ci_verdict(gates_passed: bool, mismatches: Sequence[DependencyMismatch]) -> str:
    if not gates_passed:
        return "FAIL"
    return "PASS-WITH-MISMATCHED-DEPS" if mismatches else "PASS"


def workflow_axiom_invocations(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    """Extract axiom-encode invocations from validate-job run blocks."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    steps = payload["jobs"]["validate"]["steps"]
    invocations = []
    for step in steps:
        run = step.get("run", "") if isinstance(step, dict) else ""
        logical_run = run.replace("\\\n", " ")
        for match in re.finditer(
            r"(?:/opt/axiom-verification/)?axiom-encode\s+([a-z-]+)([^\n]*)",
            logical_run,
        ):
            command = match.group(1)
            try:
                tokens = shlex.split(match.group(2))
            except ValueError:
                tokens = match.group(2).split()
            flags = tuple(token for token in tokens if token.startswith("--"))
            if command in {
                "validation-waivers",
                "guard-generated",
                "validate",
                "test",
                "proof-validate",
                "oracle-coverage",
            }:
                invocations.append((command, flags))
    return invocations


def workflow_gate_coverage(path: Path) -> dict[str, tuple[str, ...]]:
    """Map every parity-relevant workflow step to its registry gate(s)."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = {
        step.get("name")
        for step in payload["jobs"]["validate"]["steps"]
        if isinstance(step, dict)
    }
    return {
        name: WORKFLOW_GATE_STEPS[name] for name in names & WORKFLOW_GATE_STEPS.keys()
    }


def workflow_gate_contract(
    step: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    """Return behavior-bearing reusable-workflow context for one gate step."""

    trigger = payload.get("on", payload.get(True, {}))
    workflow_call = (
        trigger.get("workflow_call", {}) if isinstance(trigger, dict) else {}
    )
    raw_inputs = (
        workflow_call.get("inputs", {}) if isinstance(workflow_call, dict) else {}
    )
    inputs = {
        name: {
            "type": declaration.get("type"),
            "default": declaration.get("default"),
        }
        for name, declaration in sorted(raw_inputs.items())
        if isinstance(name, str) and isinstance(declaration, dict)
    }
    validate_job = payload.get("jobs", {}).get("validate", {})
    return {
        "step": step,
        "workflow_call_inputs": inputs,
        "workflow_env": payload.get("env"),
        "validate_job_env": validate_job.get("env"),
        "workflow_defaults": payload.get("defaults"),
        "validate_job_defaults": validate_job.get("defaults"),
    }
