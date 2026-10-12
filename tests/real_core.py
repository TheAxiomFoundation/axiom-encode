"""The real axiom-core CLI for the cross-repository export tests.

``tests/test_core_export_integration.py`` checks that what the encoder exports
builds, verifies and runs in a real ``axiom-core`` executable. Those tests
request the ``real_core`` fixture defined here (``tests/conftest.py`` loads this
module as a plugin), which also marks them ``real_core``. CI's ``real-core``
job (``.github/workflows/ci.yml``) checks out TheAxiomFoundation/axiom-core at
the commit pinned in that job's ``AXIOM_CORE_REF``, compiles it with
``cargo build --locked`` in the same run, exports the binary as
``AXIOM_CORE_BIN``, and runs ``pytest -m real_core`` with
``AXIOM_REQUIRE_REAL_CORE=1``. The main ``test`` job deselects these tests.

Nothing here discovers, downloads or builds core; ``AXIOM_CORE_BIN`` names it.

- Unset, the tests skip. Under ``AXIOM_REQUIRE_REAL_CORE=1`` they fail, and
  that flag also requires ``AXIOM_CORE_REF``.
- Set, it must name an executable whose ``capabilities`` report the
  ``axiom/capabilities/v0`` format, the axiom-rules-engine repository, and the
  sha256 of its own bytes. A configured binary that fails a check fails the
  tests instead of skipping them.
- With ``AXIOM_CORE_REF`` set, the binary must also be a cargo build
  (``target/debug`` or ``target/release``) inside a clean git checkout at that
  commit, and the ``Cargo.lock`` digest core compiles into itself
  (``capabilities.engine.dependency_lock_sha256``) must be that checkout's.
  That ties the binary to the pinned dependency lock, and so to the engine
  revision core links. It does not prove which core sources were compiled;
  the CI job compiles them from the pinned checkout in the same run.

To reproduce CI locally, from a checkout of the pin::

    cargo build --locked --bin axiom-core
    AXIOM_CORE_BIN=<checkout>/target/debug/axiom-core AXIOM_CORE_REF=<pin> \\
    AXIOM_REQUIRE_REAL_CORE=1 .venv/bin/pytest -m real_core
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

BINARY_NAME = "axiom-core"
MARKER = "real_core"
REQUIRE_ENV = "AXIOM_REQUIRE_REAL_CORE"
BINARY_ENV = "AXIOM_CORE_BIN"
REF_ENV = "AXIOM_CORE_REF"
UNAVAILABLE = "real axiom-core unavailable"
CAPABILITIES_FORMAT = "axiom/capabilities/v0"
ENGINE_REPOSITORY = "https://github.com/TheAxiomFoundation/axiom-rules-engine"
# Skip reasons that mean "no core": the helper's own, and the wording the
# integration tests used before it existed. See skip_problem.
CORE_ABSENT_REASON = re.compile(
    rf"{re.escape(UNAVAILABLE)}"
    r"|AXIOM_CORE_[A-Z]+"
    r"|\baxiom-core\b[^.;]*\b(?:binary|executable|built|unavailable|available)\b"
    r"|\b(?:binary|executable|built)\b[^.;]*\baxiom-core\b",
    re.IGNORECASE,
)
_PROBE_TIMEOUT_SECONDS = 60
_FULL_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class RealCore:
    """A usable axiom-core executable and what it reported about itself."""

    binary: Path
    sha256: str
    engine_revision: str
    # The pinned checkout the binary was built in, when AXIOM_CORE_REF is set.
    checkout: Path | None


def required() -> bool:
    """Whether a missing core fails tests (CI's real-core job) or skips them."""
    return os.environ.get(REQUIRE_ENV) == "1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _capabilities(binary: Path) -> tuple[dict | None, str]:
    """The engine identity ``binary capabilities`` reports, or why it is unusable."""
    try:
        probe = subprocess.run(
            [str(binary), "capabilities"],
            capture_output=True,
            text=True,
            check=False,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, f"{binary} could not run: {error}"
    if probe.returncode != 0 or probe.stderr:
        return None, (
            f"{binary} capabilities exited {probe.returncode}: "
            f"{probe.stderr.strip()[:500]}"
        )
    try:
        capabilities = json.loads(probe.stdout)
    except json.JSONDecodeError:
        return None, f"{binary} capabilities did not print JSON"
    if (
        not isinstance(capabilities, dict)
        or capabilities.get("format") != CAPABILITIES_FORMAT
    ):
        return None, f"{binary} capabilities are not {CAPABILITIES_FORMAT}"
    engine = capabilities.get("engine")
    if not isinstance(engine, dict) or engine.get("repository") != ENGINE_REPOSITORY:
        return None, f"{binary} capabilities do not name {ENGINE_REPOSITORY}"
    digest = _sha256(binary)
    if engine.get("execution_host_sha256") != digest:
        return None, (
            f"{binary} reports execution_host_sha256 "
            f"{engine.get('execution_host_sha256')}, but its bytes hash to {digest}"
        )
    return engine, ""


def _pinned_checkout(binary: Path, engine: dict, pin: str) -> tuple[Path | None, str]:
    """The clean checkout at ``pin`` that ``binary`` was built in, or why not."""
    if not _FULL_SHA.fullmatch(pin):
        return None, f"{REF_ENV}={pin} is not a full lowercase commit SHA"
    profile = binary.parent
    if profile.name not in {"debug", "release"} or profile.parent.name != "target":
        return None, (
            f"{binary} is not a cargo build in target/debug or target/release, "
            f"so {REF_ENV} cannot hold"
        )
    root = profile.parent.parent
    toplevel = _git(root, "rev-parse", "--show-toplevel")
    if toplevel is None or Path(toplevel).resolve() != root.resolve():
        return None, f"{root} is not a git checkout, so {REF_ENV} cannot hold"
    head = _git(root, "rev-parse", "HEAD")
    if head != pin:
        return None, f"{root} is at {head}, not {REF_ENV}={pin}"
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    if status is None or status:
        return None, f"{root} has tracked changes at {REF_ENV}={pin}"
    lock = root / "Cargo.lock"
    if not lock.is_file():
        return None, f"{root} has no Cargo.lock at {REF_ENV}={pin}"
    if engine.get("dependency_lock_sha256") != _sha256(lock):
        return None, (
            f"{binary} was not built from the Cargo.lock of {REF_ENV}={pin} "
            f"(it reports {engine.get('dependency_lock_sha256')})"
        )
    return root.resolve(), ""


@functools.cache
def locate() -> tuple[RealCore | None, str]:
    """The real core for this run, or ``(None, why)``."""
    raw = os.environ.get(BINARY_ENV)
    if not raw:
        return None, f"{BINARY_ENV} is unset; set it to a built axiom-core executable"
    origin = f"{BINARY_ENV}={raw}"
    configured = Path(raw).expanduser()
    if not configured.is_file():
        return None, f"{origin}: no such file"
    binary = configured.resolve()
    if not os.access(binary, os.X_OK):
        return None, f"{origin}: {binary} is not executable"
    engine, why = _capabilities(binary)
    if engine is None:
        return None, f"{origin}: {why}"
    checkout = None
    pin = os.environ.get(REF_ENV)
    if pin:
        checkout, why = _pinned_checkout(binary, engine, pin)
        if checkout is None:
            return None, f"{origin}: {why}"
    elif required():
        return None, f"{REQUIRE_ENV}=1 needs {REF_ENV}, the commit core was built from"
    return (
        RealCore(
            binary=binary,
            sha256=engine["execution_host_sha256"],
            engine_revision=str(engine.get("revision")),
            checkout=checkout,
        ),
        "",
    )


def require() -> RealCore:
    """Return the real core; skip without one, or fail when it is required or broken."""
    core, why = locate()
    if core is not None:
        return core
    reason = f"{UNAVAILABLE}: {why}"
    if required():
        pytest.fail(f"{reason} ({REQUIRE_ENV}=1)", pytrace=False)
    if os.environ.get(BINARY_ENV):
        # A configured core that does not work is a broken setup, not an absence.
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason)


def describe() -> str:
    """One line for the pytest header: which core this run executes."""
    core, why = locate()
    if core is None:
        return f"{UNAVAILABLE}: {why}"
    built_at = (
        f"checkout {core.checkout} at {os.environ[REF_ENV]}"
        if core.checkout is not None
        else f"unpinned; set {REF_ENV} to check it"
    )
    return (
        f"real axiom-core: {core.binary} (sha256 {core.sha256}, "
        f"engine {core.engine_revision}, {built_at})"
    )


def skip_problem(reason: str, *, marked: bool, required: bool) -> str | None:
    """Why a skip with ``reason`` must be a failure instead, or None to keep it.

    A skip for a missing core is a failure when the test is not marked
    real_core, because CI's real-core job selects by that marker and would
    never run it, and when core is required, because then it is a provisioning
    regression. Every other skip stands.
    """
    if not CORE_ABSENT_REASON.search(reason):
        return None
    if not marked:
        return (
            "skipped for a missing axiom-core without the real_core marker, so "
            "CI's real-core job never runs it; request the real_core fixture "
            "(tests/real_core.py)"
        )
    if required:
        return f"skipped under {REQUIRE_ENV}=1"
    return None


# Pytest hooks and the fixture; tests/conftest.py loads this module as a plugin.


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{MARKER}: executes a real axiom-core CLI; CI's real-core job runs "
        f"these with {REQUIRE_ENV}=1 (tests/real_core.py)",
    )


def pytest_report_header(config: pytest.Config) -> str:
    return describe()


@pytest.fixture(scope="session")
def real_core() -> RealCore:
    """The real core: skips without one, fails when required or misconfigured."""
    return require()


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    # Before `-m` selects: a test that needs core, directly or through another
    # fixture, is a real_core test.
    for item in items:
        if MARKER in getattr(item, "fixturenames", ()):
            item.add_marker(MARKER)


def _reason(longrepr) -> str:
    return longrepr[2] if isinstance(longrepr, tuple) else str(longrepr)


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector: pytest.Collector):
    # A module skipped at collection for a missing core never yields an item,
    # so no marker can select it; that is always a failure.
    report = yield
    if report.skipped:
        reason = _reason(report.longrepr)
        if CORE_ABSENT_REASON.search(reason):
            report.outcome = "failed"
            report.longrepr = (
                f"{collector.nodeid} skipped at collection for a missing "
                "axiom-core, so CI's real-core job never runs it; request the "
                f"real_core fixture (tests/real_core.py): {reason}"
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
