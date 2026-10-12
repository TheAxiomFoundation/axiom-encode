"""The real-core fixture checks the core it is given, and core skips cannot hide.

No encoder CI job ever set AXIOM_CORE_BIN, so the six tests in
tests/test_core_export_integration.py skipped in every run (main run
38280081194). tests/real_core.py now supplies core to them, and CI's real-core
job runs them with AXIOM_REQUIRE_REAL_CORE=1 against a core compiled from
AXIOM_CORE_REF. These tests pin the pieces that keep that coverage from
silently disappearing again:

- the binary: what a configured AXIOM_CORE_BIN must report before a test uses
  it, and that a configured but unusable one fails rather than skips;
- the pin: AXIOM_CORE_REF holds only for a cargo build in a clean checkout at
  that commit, built from that checkout's Cargo.lock;
- the guard: skip_problem's properties, its separation from the engine guard
  in tests/real_engine.py, and the hooks end to end in a separate pytest run;
- the suite: no test hand-writes a core skip, the six tests are exactly what
  `-m real_core` selects, and ci.yml wires the job to the pin and the flag.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from tests import real_core, real_engine
from tests.test_real_engine import LEGACY_ENGINE_SKIPS, _skip_reasons

REPO = Path(__file__).resolve().parents[1]
CORE_ENV = (real_core.REQUIRE_ENV, real_core.BINARY_ENV, real_core.REF_ENV)
ENGINE_ENV = (
    real_engine.REQUIRE_ENV,
    real_engine.BINARY_ENV,
    real_engine.ROOT_ENV,
    real_engine.REF_ENV,
)
LOCK = "# SYNTHETIC Cargo.lock for tests\nversion = 4\n"
LOCK_SHA256 = hashlib.sha256(LOCK.encode()).hexdigest()
# A stand-in for `axiom-core capabilities`: it reports the sha256 of its own
# bytes, as core does, unless a test overrides a field.
FAKE_CORE = """#!{python}
import hashlib
import json
import sys

if sys.argv[1:] != ["capabilities"]:
    sys.exit(3)
with open(sys.argv[0], "rb") as handle:
    digest = hashlib.sha256(handle.read()).hexdigest()
engine = {{
    "repository": {repository!r},
    "revision": {revision!r},
    "dependency_lock_sha256": {lock!r},
    "execution_host_sha256": {host!r} or digest,
}}
print(json.dumps({{"format": {format!r}, "engine": engine}}))
"""


@pytest.fixture(autouse=True)
def isolated_resolution(monkeypatch):
    """Resolve against this test's variables only."""
    for name in CORE_ENV:
        monkeypatch.delenv(name, raising=False)
    real_core.locate.cache_clear()
    yield
    real_core.locate.cache_clear()


def _core(path: Path, **fields: str) -> Path:
    values = {
        "python": sys.executable,
        "repository": real_core.ENGINE_REPOSITORY,
        "revision": "e" * 40,
        "lock": LOCK_SHA256,
        "host": "",
        "format": real_core.CAPABILITIES_FORMAT,
        **fields,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FAKE_CORE.format(**values))
    path.chmod(0o755)
    return path


def _script(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)
    return path


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-q",
        "-m",
        message,
    )
    return _git(root, "rev-parse", "HEAD")


def _checkout(root: Path, profile: str = "debug") -> tuple[Path, str]:
    """A clean core checkout and a cargo build of it: (binary, HEAD)."""
    root.mkdir(parents=True)
    (root / ".gitignore").write_text("/target\n")
    (root / "Cargo.toml").write_text('[workspace]\nmembers = ["crates/axiom-core"]\n')
    (root / "Cargo.lock").write_text(LOCK)
    _git(root, "init", "-q")
    head = _commit(root, "core")
    binary = _core(root / "target" / profile / real_core.BINARY_NAME)
    return binary, head


# The binary


def test_without_the_variable_the_reason_says_how_to_provide_core():
    core, why = real_core.locate()

    assert core is None
    assert real_core.BINARY_ENV in why
    with pytest.raises(pytest.skip.Exception, match=real_core.UNAVAILABLE):
        real_core.require()


def test_require_fails_without_core_when_required(monkeypatch):
    monkeypatch.setenv(real_core.REQUIRE_ENV, "1")

    with pytest.raises(pytest.fail.Exception, match=real_core.REQUIRE_ENV):
        real_core.require()


def test_required_runs_need_the_pin_too(monkeypatch, tmp_path):
    # CI's guarantee is a core built from AXIOM_CORE_REF; a required run that
    # lost the pin would accept any binary.
    monkeypatch.setenv(real_core.BINARY_ENV, str(_core(tmp_path / "axiom-core")))
    monkeypatch.setenv(real_core.REQUIRE_ENV, "1")

    core, why = real_core.locate()

    assert core is None
    assert real_core.REF_ENV in why
    with pytest.raises(pytest.fail.Exception, match=real_core.REF_ENV):
        real_core.require()


@pytest.mark.parametrize("value", ["", "0", "true", "yes", " 1"])
def test_only_the_exact_flag_requires_core(monkeypatch, value):
    monkeypatch.setenv(real_core.REQUIRE_ENV, value)

    assert real_core.required() is False


def test_a_configured_core_is_used_without_a_pin(monkeypatch, tmp_path):
    binary = _core(tmp_path / "bin" / real_core.BINARY_NAME)
    monkeypatch.setenv(real_core.BINARY_ENV, str(binary))

    core, why = real_core.locate()

    assert why == ""
    assert core == real_core.RealCore(
        binary=binary.resolve(),
        sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        engine_revision="e" * 40,
        checkout=None,
    )
    assert real_core.require() == core
    assert "unpinned" in real_core.describe()


def test_a_relative_path_names_the_file_not_a_path_lookup(monkeypatch, tmp_path):
    binary = _core(tmp_path / "bin" / real_core.BINARY_NAME)
    monkeypatch.chdir(binary.parent)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv(real_core.BINARY_ENV, real_core.BINARY_NAME)

    core, why = real_core.locate()

    assert why == ""
    assert core.binary == binary.resolve()


def _unusable(tmp_path: Path, breakage: str) -> Path:
    path = tmp_path / "bin" / real_core.BINARY_NAME
    if breakage == "missing":
        return path
    if breakage == "not executable":
        _core(path).chmod(0o644)
        return path
    if breakage == "exits nonzero":
        return _script(path, "#!/bin/sh\nexit 1\n")
    if breakage == "writes stderr":
        return _script(
            path,
            f"#!/bin/sh\necho warning >&2\nexec {sys.executable} "
            f"{_core(tmp_path / 'real' / real_core.BINARY_NAME)} capabilities\n",
        )
    if breakage == "prints no JSON":
        return _script(path, "#!/bin/sh\necho capabilities\n")
    if breakage == "other format":
        return _core(path, format="axiom/capabilities/v1")
    if breakage == "other engine repository":
        return _core(path, repository="https://example.invalid/engine")
    if breakage == "wrong self digest":
        # Capabilities copied from another build.
        return _core(path, host="0" * 64)
    raise AssertionError(breakage)


@pytest.mark.parametrize(
    ("breakage", "expected"),
    [
        ("missing", "no such file"),
        ("not executable", "is not executable"),
        ("exits nonzero", "capabilities exited 1"),
        ("writes stderr", "capabilities exited 0: warning"),
        ("prints no JSON", "did not print JSON"),
        ("other format", f"are not {real_core.CAPABILITIES_FORMAT}"),
        ("other engine repository", f"do not name {real_core.ENGINE_REPOSITORY}"),
        ("wrong self digest", "but its bytes hash to"),
    ],
)
def test_a_configured_but_unusable_core_fails_rather_than_skips(
    monkeypatch, tmp_path, breakage, expected
):
    monkeypatch.setenv(real_core.BINARY_ENV, str(_unusable(tmp_path, breakage)))

    core, why = real_core.locate()

    assert core is None
    assert why.startswith(f"{real_core.BINARY_ENV}=")
    assert expected in why
    with pytest.raises(pytest.fail.Exception, match=real_core.UNAVAILABLE):
        real_core.require()


# The pin


@pytest.mark.parametrize("profile", ["debug", "release"])
def test_the_pin_holds_for_a_build_in_a_clean_checkout(monkeypatch, tmp_path, profile):
    root = tmp_path / "axiom-core"
    binary, head = _checkout(root, profile)
    monkeypatch.setenv(real_core.BINARY_ENV, str(binary))
    monkeypatch.setenv(real_core.REF_ENV, head)
    monkeypatch.setenv(real_core.REQUIRE_ENV, "1")

    core, why = real_core.locate()

    assert why == ""
    assert core.checkout == root.resolve()
    assert f"checkout {root.resolve()} at {head}" in real_core.describe()


def test_the_pin_follows_a_symlink_to_the_build(monkeypatch, tmp_path):
    binary, head = _checkout(tmp_path / "axiom-core")
    link = tmp_path / "bin" / real_core.BINARY_NAME
    link.parent.mkdir()
    link.symlink_to(binary)
    monkeypatch.setenv(real_core.BINARY_ENV, str(link))
    monkeypatch.setenv(real_core.REF_ENV, head)

    core, why = real_core.locate()

    assert why == ""
    assert core.binary == binary.resolve()


def _break_pin(root: Path, binary: Path, head: str, tmp_path: Path, breakage: str):
    """Return (binary, pin) for ``breakage``; every one must fail closed."""
    if breakage == "other commit":
        return binary, "0" * 40
    if breakage == "short sha":
        return binary, head[:12]
    if breakage == "uppercase sha":
        return binary, head.upper()
    if breakage == "tracked change":
        (root / "Cargo.toml").write_text("changed\n")
    elif breakage == "built from another lock":
        _core(binary, lock="1" * 64)
    elif breakage == "lock changed since the build":
        (root / "Cargo.lock").write_text(LOCK + "# bumped\n")
        return binary, _commit(root, "bump the lock")
    elif breakage == "no lock":
        (root / "Cargo.lock").unlink()
        return binary, _commit(root, "drop the lock")
    elif breakage == "not a cargo build":
        return _core(root / real_core.BINARY_NAME), head
    elif breakage == "not a checkout":
        return _core(tmp_path / "plain" / "target" / "debug" / "axiom-core"), head
    elif breakage == "nested":
        # A build in a subdirectory of the pinned checkout, not its root.
        return _core(root / "vendored" / "target" / "debug" / "axiom-core"), head
    else:
        raise AssertionError(breakage)
    return binary, head


@pytest.mark.parametrize(
    ("breakage", "expected"),
    [
        ("other commit", "is at"),
        ("short sha", "is not a full lowercase commit SHA"),
        ("uppercase sha", "is not a full lowercase commit SHA"),
        ("tracked change", "has tracked changes"),
        ("built from another lock", "was not built from the Cargo.lock"),
        ("lock changed since the build", "was not built from the Cargo.lock"),
        ("no lock", "has no Cargo.lock"),
        ("not a cargo build", "is not a cargo build"),
        ("not a checkout", "is not a git checkout"),
        ("nested", "is not a git checkout"),
    ],
)
def test_the_pin_fails_closed(monkeypatch, tmp_path, breakage, expected):
    root = tmp_path / "axiom-core"
    binary, head = _checkout(root)
    binary, pin = _break_pin(root, binary, head, tmp_path, breakage)
    monkeypatch.setenv(real_core.BINARY_ENV, str(binary))
    monkeypatch.setenv(real_core.REF_ENV, pin)

    core, why = real_core.locate()

    assert core is None
    assert real_core.REF_ENV in why
    assert expected in why
    with pytest.raises(pytest.fail.Exception, match=real_core.REF_ENV):
        real_core.require()


# The guard

# The wording the integration tests used before tests/real_core.py, and other
# hand-written ways to say core is missing.
LEGACY_CORE_SKIPS = [
    "AXIOM_CORE_BIN is unset; provide a real axiom-core executable",
    "set AXIOM_CORE_BIN to run the core interoperability tests",
    "axiom-core binary is not built",
    "no built axiom-core executable is available",
]
# Skips that are not about a missing axiom-core.
OTHER_SKIPS = [
    *LEGACY_ENGINE_SKIPS,
    "axiom-rules-engine repo not present",
    "rulespec-us repo not present",
    "Node.js is not installed",
    "Go is required to build the signing supervisor",
    "PolicyEngine US is not installed",
]
CORE_REASONS = st.one_of(
    st.sampled_from(LEGACY_CORE_SKIPS),
    st.text().map(lambda text: f"{real_core.UNAVAILABLE}: {text}"),
)


@pytest.mark.parametrize("reason", LEGACY_CORE_SKIPS)
def test_legacy_core_skip_wordings_are_recognized(reason):
    assert real_core.CORE_ABSENT_REASON.search(f"Skipped: {reason}")


@pytest.mark.parametrize("reason", OTHER_SKIPS)
def test_other_skips_are_not_core_skips(reason):
    assert not real_core.CORE_ABSENT_REASON.search(f"Skipped: {reason}")


@given(reason=st.text(), marked=st.booleans(), required=st.booleans())
def test_only_core_skips_ever_become_failures(reason, marked, required):
    problem = real_core.skip_problem(reason, marked=marked, required=required)
    if not real_core.CORE_ABSENT_REASON.search(reason):
        assert problem is None


@given(reason=CORE_REASONS, required=st.booleans())
def test_an_unmarked_core_skip_is_always_a_failure(reason, required):
    problem = real_core.skip_problem(reason, marked=False, required=required)

    assert problem is not None and "real_core marker" in problem


@given(reason=CORE_REASONS)
def test_a_marked_core_skip_fails_exactly_when_required(reason):
    assert real_core.skip_problem(reason, marked=True, required=False) is None
    assert real_core.skip_problem(reason, marked=True, required=True) == (
        f"skipped under {real_core.REQUIRE_ENV}=1"
    )


# Both guards see every skip. Each must leave the other's skips alone, or a
# marked real_core test skipping locally would be failed as an unmarked engine
# skip (and the reverse). Every engine pattern needs "rules-engine",
# "rules_engine" or "locate", and every core pattern needs "core"; the filters
# use the guards' own case-insensitive matching (under which "ı" matches "i").
ENGINE_WORDS = re.compile(r"rules[-_]engine|locate", re.IGNORECASE)
CORE_WORDS = re.compile(r"core", re.IGNORECASE)


@given(detail=st.text().filter(lambda text: not ENGINE_WORDS.search(text)))
def test_the_engine_guard_ignores_core_skips(detail):
    reason = f"{real_core.UNAVAILABLE}: {detail}"

    assert real_engine.skip_problem(reason, marked=False, required=True) is None


@given(detail=st.text().filter(lambda text: not CORE_WORDS.search(text)))
def test_the_core_guard_ignores_engine_skips(detail):
    reason = f"{real_engine.UNAVAILABLE}: {detail}"

    assert real_core.skip_problem(reason, marked=False, required=True) is None


def test_the_helpers_own_skip_reasons_stay_on_their_side():
    core_reason = f"{real_core.UNAVAILABLE}: {real_core.locate()[1]}"

    assert real_core.CORE_ABSENT_REASON.search(core_reason)
    assert not real_engine.ENGINE_ABSENT_REASON.search(core_reason)
    for reason in LEGACY_CORE_SKIPS:
        assert not real_engine.ENGINE_ABSENT_REASON.search(reason), reason


def test_no_test_writes_its_own_core_skip():
    # A hand-written skip is invisible to `-m real_core`; the runtime guard
    # fails it, and this catches it before CI does.
    tests = Path(__file__).parent
    offenders = [
        f"{path.name}: {reason}"
        for path in sorted(tests.glob("test_*.py"))
        if path.name != Path(__file__).name
        for reason in _skip_reasons(path)
        if real_core.CORE_ABSENT_REASON.search(reason)
    ]
    assert offenders == []


# The hooks, end to end in a separate pytest run

SAMPLE_TESTS = """
import pytest


@pytest.fixture
def wrapped(real_core):
    return real_core


def test_direct(real_core):
    assert real_core.binary.is_file()


def test_transitive(wrapped):
    assert wrapped.sha256


def test_handwritten_skip():
    pytest.skip("AXIOM_CORE_BIN is unset; provide a real axiom-core executable")


def test_unrelated_skip():
    pytest.skip("Node.js is not installed")


@pytest.mark.xfail(reason="AXIOM_CORE_BIN is unset", run=False)
def test_expected_failure():
    raise AssertionError


def test_plain():
    pass
"""
# The sample loads both helpers the way tests/conftest.py does.
SAMPLE_CONFTEST = """from tests.real_engine import (  # noqa: F401
    pytest_collection_modifyitems,
    pytest_configure,
    pytest_make_collect_report,
    pytest_report_header,
    pytest_runtest_makereport,
    real_engine,
)

pytest_plugins = ("tests.real_core",)
"""


def _run_sample(
    tmp_path: Path,
    env: dict[str, str],
    *args: str,
    tests: dict[str, str] | None = None,
):
    project = tmp_path / "project"
    if not project.exists():
        project.mkdir()
        (project / "conftest.py").write_text(SAMPLE_CONFTEST)
        for name, text in (tests or {"test_sample.py": SAMPLE_TESTS}).items():
            (project / name).write_text(text)
    report = tmp_path / f"report-{len(list(tmp_path.glob('report-*')))}.xml"
    base = {
        key: value
        for key, value in os.environ.items()
        if key not in CORE_ENV + ENGINE_ENV and not key.startswith(("PYTEST_", "COV_"))
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            f"--junit-xml={report}",
            *args,
        ],
        cwd=project,
        env={
            **base,
            "PYTHONPATH": os.pathsep.join([str(REPO), str(REPO / "src")]),
            # No engine discovery: these runs are about core.
            real_engine.ROOT_ENV: str(tmp_path / "no-engine"),
            **env,
        },
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    outcomes = {}
    for case in ET.parse(report).iter("testcase"):
        kinds = [child.tag for child in case if child.tag != "system-out"]
        outcomes[case.get("name")] = kinds[0] if kinds else "passed"
    return completed, outcomes


def test_hooks_without_core(tmp_path):
    completed, outcomes = _run_sample(tmp_path, {})

    assert f"{real_core.UNAVAILABLE}: {real_core.BINARY_ENV} is unset" in (
        completed.stdout
    )
    assert outcomes == {
        "test_direct": "skipped",
        "test_transitive": "skipped",
        "test_handwritten_skip": "failure",
        "test_unrelated_skip": "skipped",
        "test_expected_failure": "skipped",
        "test_plain": "passed",
    }, completed.stdout
    assert "without the real_core marker" in completed.stdout


def test_hooks_fail_core_tests_when_core_is_required(tmp_path):
    completed, outcomes = _run_sample(tmp_path, {real_core.REQUIRE_ENV: "1"})

    assert outcomes == {
        "test_direct": "error",
        "test_transitive": "error",
        "test_handwritten_skip": "failure",
        "test_unrelated_skip": "skipped",
        "test_expected_failure": "skipped",
        "test_plain": "passed",
    }, completed.stdout
    assert completed.returncode == 1


def test_hooks_fail_core_tests_when_core_is_configured_but_broken(tmp_path):
    completed, outcomes = _run_sample(
        tmp_path, {real_core.BINARY_ENV: str(tmp_path / "missing")}
    )

    assert outcomes["test_direct"] == "error", completed.stdout
    assert outcomes["test_transitive"] == "error", completed.stdout
    assert completed.returncode == 1


def test_hooks_run_core_tests_against_the_pinned_core(tmp_path):
    binary, head = _checkout(tmp_path / "axiom-core")
    completed, outcomes = _run_sample(
        tmp_path,
        {
            real_core.REQUIRE_ENV: "1",
            real_core.BINARY_ENV: str(binary),
            real_core.REF_ENV: head,
        },
        "-m",
        real_core.MARKER,
    )

    assert f"real axiom-core: {binary.resolve()}" in completed.stdout
    assert outcomes == {"test_direct": "passed", "test_transitive": "passed"}, (
        completed.stdout
    )
    assert completed.returncode == 0


def test_hooks_let_the_main_job_deselect_every_core_test(tmp_path):
    completed, outcomes = _run_sample(
        tmp_path, {}, "-m", f"not {real_engine.MARKER} and not {real_core.MARKER}"
    )

    assert set(outcomes) == {
        "test_handwritten_skip",
        "test_unrelated_skip",
        "test_expected_failure",
        "test_plain",
    }, completed.stdout


def test_hooks_fail_a_module_skipped_for_a_missing_core(tmp_path):
    completed, outcomes = _run_sample(
        tmp_path,
        {},
        tests={
            "test_module_skip.py": (
                "import pytest\n"
                "pytest.skip('AXIOM_CORE_BIN is unset', allow_module_level=True)\n"
                "def test_never_collected():\n"
                "    pass\n"
            ),
            "test_other_module_skip.py": (
                "import pytest\n"
                "pytest.skip('Node.js is not installed', allow_module_level=True)\n"
            ),
        },
    )

    assert completed.returncode != 0
    assert "skipped at collection for a missing axiom-core" in completed.stdout
    assert "test_module_skip.py" in completed.stdout
    assert (
        "test_other_module_skip.py"
        not in completed.stdout.split("short test summary")[-1]
    )
    assert "test_never_collected" not in outcomes


# The suite and CI


def _collected(*args: str) -> list[str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in CORE_ENV + ENGINE_ENV and not key.startswith(("PYTEST_", "COV_"))
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "--collect-only",
            "-q",
            "tests/test_core_export_integration.py",
            *args,
        ],
        cwd=REPO,
        env={**env, real_engine.ROOT_ENV: str(REPO / "no-engine")},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode in {0, 5}, completed.stdout + completed.stderr
    return [line for line in completed.stdout.splitlines() if "::" in line]


def test_every_integration_test_is_a_real_core_test():
    everything = _collected()

    assert everything
    assert _collected("-m", real_core.MARKER) == everything
    assert (
        _collected("-m", f"not {real_engine.MARKER} and not {real_core.MARKER}") == []
    )


def _ci_jobs() -> dict:
    workflow = REPO / ".github" / "workflows" / "ci.yml"
    return yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]


def _run(step: dict) -> str:
    return " ".join(str(step.get("run", "")).split())


def test_ci_compiles_the_pinned_core_and_requires_it():
    job = _ci_jobs()["real-core"]
    steps = job["steps"]
    assert real_core._FULL_SHA.fullmatch(job["env"][real_core.REF_ENV])

    [checkout] = [step for step in steps if "axiom-core.git" in _run(step)]
    [build] = [step for step in steps if "cargo" in _run(step)]
    [tests] = [step for step in steps if real_core.MARKER in _run(step)]
    assert steps.index(checkout) < steps.index(build) < steps.index(tests)
    # The checkout fetches the pin and proves HEAD is it.
    assert '"$AXIOM_CORE_REF"' in checkout["run"]
    assert (
        'test "$(git -C "$core" rev-parse HEAD)" = "$AXIOM_CORE_REF"' in checkout["run"]
    )
    # The build compiles the binary the tests run, in the pinned checkout.
    assert build["working-directory"] == "${{ runner.temp }}/axiom-core"
    assert _run(build) == "cargo build --locked --bin axiom-core"
    assert _run(tests) == f".venv/bin/pytest tests/ -m {real_core.MARKER}"
    assert tests["env"] == {
        real_core.REQUIRE_ENV: "1",
        real_core.BINARY_ENV: "${{ runner.temp }}/axiom-core/target/debug/axiom-core",
    }
    # Nothing restores a cached core: rust-cache keeps dependencies only.
    caches = [step for step in steps if "cache" in step.get("uses", "")]
    assert [step["uses"].partition("@")[0] for step in caches] == [
        "Swatinem/rust-cache"
    ]
    assert set(caches[0]["with"]) == {"workspaces"}


def test_the_test_job_leaves_core_tests_to_real_core():
    runs = [_run(step) for step in _ci_jobs()["test"]["steps"]]

    assert (
        f'.venv/bin/pytest tests/ -m "not {real_engine.MARKER} '
        f'and not {real_core.MARKER}"' in runs
    )
