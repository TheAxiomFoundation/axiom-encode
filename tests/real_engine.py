"""The real axiom-rules-engine for tests that execute it.

A test that compiles or runs RuleSpec through the actual engine requests the
``real_engine`` fixture defined here (``tests/conftest.py`` installs it and the
hooks below), which also marks it ``real_engine``. CI's ``real-engine`` job (``.github/workflows/ci.yml``) builds
the engine at the commit pinned in that job's ``AXIOM_RULES_ENGINE_REF``,
exports the checkout as ``AXIOM_RULES_ENGINE_ROOT``, and runs
``pytest -m real_engine`` with ``AXIOM_REQUIRE_REAL_ENGINE=1``. Under that flag
an engine that is missing, unusable, or at the wrong commit fails the test
instead of skipping it, so a provisioning regression cannot silently drop the
coverage. The main ``test`` job deselects these tests.

Resolution order:

1. ``AXIOM_RULES_ENGINE_BINARY``: an engine executable. Its checkout is the
   directory above ``target/<profile>/`` when it sits there, else its own
   directory.
2. ``AXIOM_RULES_ENGINE_ROOT``: an engine checkout holding
   ``target/release``, ``target/debug`` or a bare binary.
3. Only without ``AXIOM_REQUIRE_REAL_ENGINE=1``: ``axiom-locate engine`` when
   it is on PATH, then an ``axiom-rules-engine`` checkout beside this
   repository.

A variable that is set but does not resolve is an error, never a fall-through
to discovery. The binary is always the one ``ValidatorPipeline`` binds for the
checkout (release, then debug, then bare), so tests that pass the checkout and
tests that run the binary exercise the same build. When
``AXIOM_RULES_ENGINE_REF`` is set, the checkout must be a clean git checkout at
that commit. To reproduce CI locally, build the pinned commit with
``cargo build --release --locked`` and run::

    AXIOM_RULES_ENGINE_ROOT=<checkout> AXIOM_RULES_ENGINE_REF=<pin> \\
    AXIOM_REQUIRE_REAL_ENGINE=1 .venv/bin/pytest -m real_engine
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BINARY_NAME = "axiom-rules-engine"
LOCATOR = "axiom-locate"
MARKER = "real_engine"
REQUIRE_ENV = "AXIOM_REQUIRE_REAL_ENGINE"
BINARY_ENV = "AXIOM_RULES_ENGINE_BINARY"
ROOT_ENV = "AXIOM_RULES_ENGINE_ROOT"
REF_ENV = "AXIOM_RULES_ENGINE_REF"
UNAVAILABLE = "real axiom-rules-engine unavailable"
# The checkout tests pass where no engine is wanted. It never exists, so a test
# that reaches the engine through it fails the same way everywhere.
ABSENT_ROOT = Path("/nonexistent/axiom-rules-engine")
# Skip reasons that mean "no engine": the helper's own, and the wordings tests
# used before it existed. See skip_problem.
ENGINE_ABSENT_REASON = re.compile(
    rf"{re.escape(UNAVAILABLE)}"
    r"|axiom-locate"
    r"|AXIOM_RULES_ENGINE_[A-Z]+"
    r"|axiom-rules-engine\b[^.;]*\b(?:binary|built|unavailable|available)\b"
    r"|\b(?:binary|built)\b[^.;]*\baxiom-rules-engine",
    re.IGNORECASE,
)
_PROBE_TIMEOUT_SECONDS = 30
_FULL_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class RealEngine:
    """An engine checkout and the binary ValidatorPipeline binds inside it."""

    root: Path
    binary: Path

    @functools.cached_property
    def sha256(self) -> str:
        return hashlib.sha256(self.binary.read_bytes()).hexdigest()


def required() -> bool:
    """Whether a missing engine fails tests (CI's real-engine job) or skips them."""
    return os.environ.get(REQUIRE_ENV) == "1"


def _candidates(root: Path) -> tuple[Path, ...]:
    # The order ValidatorPipeline._unverified_axiom_rules_binary binds.
    return (
        root / "target" / "release" / BINARY_NAME,
        root / "target" / "debug" / BINARY_NAME,
        root / BINARY_NAME,
    )


def _root_of(binary: Path) -> Path:
    parent = binary.parent
    if parent.name in {"release", "debug"} and parent.parent.name == "target":
        return parent.parent.parent
    return parent


def _git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _usable(root: Path, origin: str) -> tuple[RealEngine | None, str]:
    """The engine bound in ``root``, or why it cannot be used."""
    existing = [path for path in _candidates(root) if path.exists()]
    if not existing:
        return None, f"{origin}: no {BINARY_NAME} binary under {root}"
    binary = existing[0]
    if not binary.is_file() or not os.access(binary, os.X_OK):
        return None, f"{origin}: {binary} is not an executable file"
    try:
        probe = subprocess.run(
            [str(binary), "compile", "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, f"{origin}: {binary} could not run: {error}"
    if probe.returncode != 0 or "--rulespec-root" not in probe.stdout + probe.stderr:
        return None, (
            f"{origin}: {binary} does not support compile --rulespec-root "
            f"(exit {probe.returncode})"
        )
    pin = os.environ.get(REF_ENV)
    if pin:
        if not _FULL_SHA.fullmatch(pin):
            return None, f"{REF_ENV}={pin} is not a full lowercase commit SHA"
        toplevel = _git(root, "rev-parse", "--show-toplevel")
        if toplevel is None or Path(toplevel).resolve() != root.resolve():
            return (
                None,
                f"{origin}: {root} is not a git checkout, so {REF_ENV} cannot hold",
            )
        head = _git(root, "rev-parse", "HEAD")
        if head != pin:
            return None, f"{origin}: {root} is at {head}, not {REF_ENV}={pin}"
        status = _git(root, "status", "--porcelain", "--untracked-files=no")
        if status is None or status:
            return None, f"{origin}: {root} has tracked changes at {REF_ENV}={pin}"
    return RealEngine(root=root.resolve(), binary=binary.resolve()), ""


def _from_binary(raw: str) -> tuple[RealEngine | None, str]:
    origin = f"{BINARY_ENV}={raw}"
    binary = Path(raw).expanduser()
    if not binary.is_file():
        return None, f"{origin}: no such file"
    engine, why = _usable(_root_of(binary), origin)
    if engine is not None and engine.binary != binary.resolve():
        return None, (
            f"{origin}: the checkout binds {engine.binary} instead; "
            f"set {ROOT_ENV} to the checkout"
        )
    return engine, why


def _discovered() -> tuple[RealEngine | None, str]:
    reasons = []
    locator = shutil.which(LOCATOR)
    if locator is not None:
        try:
            located = subprocess.run(
                [locator, "engine"],
                capture_output=True,
                text=True,
                check=False,
                timeout=_PROBE_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            reasons.append(f"{LOCATOR} engine failed: {error}")
        else:
            lines = located.stdout.strip().splitlines()
            if located.returncode == 0 and lines:
                engine, why = _usable(_root_of(Path(lines[0])), f"{LOCATOR} engine")
                if engine is not None:
                    return engine, ""
                reasons.append(why)
            else:
                reasons.append(f"{LOCATOR} engine found no engine")
    sibling = REPO_ROOT.parent / BINARY_NAME
    if sibling.is_dir():
        engine, why = _usable(sibling, "sibling checkout")
        if engine is not None:
            return engine, ""
        reasons.append(why)
    reasons.append(f"set {ROOT_ENV} to an engine checkout built with cargo")
    return None, "; ".join(reasons)


@functools.cache
def locate() -> tuple[RealEngine | None, str]:
    """The real engine for this run, or ``(None, why)``."""
    raw_binary = os.environ.get(BINARY_ENV)
    if raw_binary:
        return _from_binary(raw_binary)
    raw_root = os.environ.get(ROOT_ENV)
    if raw_root:
        return _usable(Path(raw_root).expanduser(), f"{ROOT_ENV}={raw_root}")
    if required():
        return None, f"{REQUIRE_ENV}=1 needs {ROOT_ENV} or {BINARY_ENV}"
    return _discovered()


def require() -> RealEngine:
    """Return the real engine; skip without one, or fail under the CI flag."""
    engine, why = locate()
    if engine is not None:
        return engine
    reason = f"{UNAVAILABLE}: {why}"
    if required():
        pytest.fail(f"{reason} ({REQUIRE_ENV}=1)", pytrace=False)
    pytest.skip(reason)


def describe() -> str:
    """One line for the pytest header: which engine this run executes."""
    engine, why = locate()
    if engine is None:
        return f"{UNAVAILABLE}: {why}"
    head = _git(engine.root, "rev-parse", "HEAD") or "not a git checkout"
    return f"real axiom-rules-engine: {engine.binary} (commit {head}, sha256 {engine.sha256})"


def skip_problem(reason: str, *, marked: bool, required: bool) -> str | None:
    """Why a skip with ``reason`` must be a failure instead, or None to keep it.

    A skip for a missing engine is a failure when the test is not marked
    real_engine, because CI's real-engine job selects by that marker and would
    never run it, and when the engine is required, because then it is a
    provisioning regression. Every other skip stands.
    """
    if not ENGINE_ABSENT_REASON.search(reason):
        return None
    if not marked:
        return (
            "skipped for a missing engine without the real_engine marker, so "
            "CI's real-engine job never runs it; request the real_engine "
            "fixture (tests/real_engine.py)"
        )
    if required:
        return f"skipped under {REQUIRE_ENV}=1"
    return None


# Pytest hooks and the fixture; tests/conftest.py installs them.


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{MARKER}: executes the real axiom-rules-engine; CI's real-engine job "
        f"runs these with {REQUIRE_ENV}=1 (tests/real_engine.py)",
    )


def pytest_report_header(config: pytest.Config) -> str:
    return describe()


@pytest.fixture(scope="session")
def real_engine() -> RealEngine:
    """The real engine: skips without one, fails under AXIOM_REQUIRE_REAL_ENGINE=1."""
    return require()


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    # Before `-m` selects: a test that needs the engine, directly or through
    # another fixture, is a real_engine test.
    for item in items:
        if MARKER in getattr(item, "fixturenames", ()):
            item.add_marker(MARKER)


def _reason(longrepr) -> str:
    return longrepr[2] if isinstance(longrepr, tuple) else str(longrepr)


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector: pytest.Collector):
    # A module skipped at collection for a missing engine never yields an
    # item, so no marker can select it; that is always a failure.
    report = yield
    if report.skipped:
        reason = _reason(report.longrepr)
        if ENGINE_ABSENT_REASON.search(reason):
            report.outcome = "failed"
            report.longrepr = (
                f"{collector.nodeid} skipped at collection for a missing engine, "
                "so CI's real-engine job never runs it; request the real_engine "
                f"fixture (tests/real_engine.py): {reason}"
            )
    return report


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    report = yield
    if not report.skipped or hasattr(report, "wasxfail"):
        return report
    reason = _reason(report.longrepr)
    problem = skip_problem(
        reason,
        marked=item.get_closest_marker(MARKER) is not None,
        required=required(),
    )
    if problem is not None:
        report.outcome = "failed"
        report.longrepr = f"{item.nodeid} {problem}: {reason}"
    return report
