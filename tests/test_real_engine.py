"""The real-engine fixture finds the right engine, and engine skips cannot hide.

CI's test job never had an axiom-rules-engine, so every test that executes the
engine skipped there (43 on main at c2ee166f) and ran only on developer
machines. tests/real_engine.py now resolves the engine for those tests and CI's
real-engine job runs them with AXIOM_REQUIRE_REAL_ENGINE=1. These tests pin the
pieces that keep that coverage from silently disappearing again:

- resolution: which checkout and binary each configuration selects, that a
  configured variable never falls through to discovery, and that the binary is
  the one ValidatorPipeline binds for the same checkout (a differential test);
- the pin: AXIOM_RULES_ENGINE_REF holds only for a clean checkout at that commit;
- the guard: skip_problem's properties, and the hooks end to end in a separate
  pytest run;
- the suite: no test hand-writes an engine skip any more.
"""

from __future__ import annotations

import ast
import itertools
import os
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from axiom_encode.engine_binding import _engine_binary_candidates
from axiom_encode.harness.validator_pipeline import ValidatorPipeline
from tests import real_engine

ENGINE_ENV = (
    real_engine.REQUIRE_ENV,
    real_engine.BINARY_ENV,
    real_engine.ROOT_ENV,
    real_engine.REF_ENV,
)
PROFILES = {
    "release": ("target", "release"),
    "debug": ("target", "debug"),
    "bare": (),
}
FAKE_ENGINE = """#!/bin/sh
if [ "$1" = compile ] && [ "$2" = --help ]; then
  echo "Usage: axiom-rules-engine compile [OPTIONS] --rulespec-root <DIR>"
  exit 0
fi
exit 3
"""
PRE_FIXTURE_ENGINE = """#!/bin/sh
echo "Usage: axiom-rules-engine compile [OPTIONS] <PROGRAM>"
"""


@pytest.fixture(autouse=True)
def isolated_resolution(monkeypatch, tmp_path):
    """Resolve against this test's variables only, with discovery off."""
    for name in ENGINE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(real_engine, "LOCATOR", "axiom-locate-absent-in-tests")
    monkeypatch.setattr(real_engine, "REPO_ROOT", tmp_path / "workspace" / "repo")
    real_engine.locate.cache_clear()
    yield
    real_engine.locate.cache_clear()


def _engine(root: Path, profile: str = "release", script: str = FAKE_ENGINE) -> Path:
    binary = root.joinpath(*PROFILES[profile], real_engine.BINARY_NAME)
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text(script)
    binary.chmod(0o755)
    return binary


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _committed_checkout(root: Path) -> str:
    _engine(root)
    (root / ".gitignore").write_text("/target\n")
    (root / "Cargo.toml").write_text('[package]\nname = "axiom-rules-engine"\n')
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-q",
        "-m",
        "engine",
    )
    return _git(root, "rev-parse", "HEAD")


# Resolution


@pytest.mark.parametrize(
    "present",
    [
        combination
        for size in range(1, len(PROFILES) + 1)
        for combination in itertools.combinations(PROFILES, size)
    ],
    ids="+".join,
)
def test_root_binds_the_binary_validator_pipeline_binds(monkeypatch, tmp_path, present):
    # Every non-empty set of builds in a checkout (exhaustive): the fixture's
    # binary is the one both of ValidatorPipeline's resolvers would run.
    root = tmp_path / "engine"
    for profile in present:
        _engine(root, profile)
    monkeypatch.setenv(real_engine.ROOT_ENV, str(root))

    engine, why = real_engine.locate()

    assert why == ""
    assert engine.root == root.resolve()
    expected = next(path for path in _engine_binary_candidates(root) if path.exists())
    assert engine.binary == expected.resolve()
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=engine.root,
        local_corpus_release=None,
        enable_oracles=False,
    )
    assert pipeline._unverified_axiom_rules_binary() == engine.binary


def test_binary_variable_names_its_checkout(monkeypatch, tmp_path):
    root = tmp_path / "engine"
    binary = _engine(root, "debug")
    monkeypatch.setenv(real_engine.BINARY_ENV, str(binary))

    engine, _ = real_engine.locate()

    assert engine == real_engine.RealEngine(
        root=root.resolve(), binary=binary.resolve()
    )


def test_binary_variable_rejects_a_binary_its_checkout_would_not_bind(
    monkeypatch, tmp_path
):
    # A pipeline handed this checkout runs target/release, so a test told to
    # use target/debug would exercise two different builds.
    root = tmp_path / "engine"
    _engine(root, "release")
    debug = _engine(root, "debug")
    monkeypatch.setenv(real_engine.BINARY_ENV, str(debug))

    engine, why = real_engine.locate()

    assert engine is None
    assert "binds" in why and real_engine.ROOT_ENV in why


def test_binary_variable_takes_precedence_over_root(monkeypatch, tmp_path):
    chosen = _engine(tmp_path / "chosen", "bare")
    _engine(tmp_path / "other")
    monkeypatch.setenv(real_engine.BINARY_ENV, str(chosen))
    monkeypatch.setenv(real_engine.ROOT_ENV, str(tmp_path / "other"))

    assert real_engine.locate()[0].binary == chosen.resolve()


@pytest.mark.parametrize("variable", [real_engine.BINARY_ENV, real_engine.ROOT_ENV])
def test_a_configured_variable_never_falls_through_to_discovery(
    monkeypatch, tmp_path, variable
):
    _engine(real_engine.REPO_ROOT.parent / real_engine.BINARY_NAME)
    monkeypatch.setenv(variable, str(tmp_path / "missing"))

    engine, why = real_engine.locate()

    assert engine is None
    assert why.startswith(f"{variable}=")


def test_required_runs_never_discover(monkeypatch):
    _engine(real_engine.REPO_ROOT.parent / real_engine.BINARY_NAME)
    monkeypatch.setenv(real_engine.REQUIRE_ENV, "1")

    engine, why = real_engine.locate()

    assert engine is None
    assert real_engine.ROOT_ENV in why


def test_discovery_prefers_the_locator_then_the_sibling_checkout(monkeypatch, tmp_path):
    sibling = real_engine.REPO_ROOT.parent / real_engine.BINARY_NAME
    _engine(sibling)
    assert real_engine.locate()[0].root == sibling.resolve()

    located = _engine(tmp_path / "located", "debug")
    locator = tmp_path / "bin" / "fake-locate"
    locator.parent.mkdir()
    locator.write_text(f"#!/bin/sh\necho '{located}'\n")
    locator.chmod(0o755)
    monkeypatch.setattr(real_engine, "LOCATOR", str(locator))
    real_engine.locate.cache_clear()

    assert real_engine.locate()[0].binary == located.resolve()


def test_without_any_engine_the_reason_says_how_to_provide_one():
    engine, why = real_engine.locate()

    assert engine is None
    assert real_engine.ROOT_ENV in why


@pytest.mark.parametrize(
    ("script", "mode", "expected"),
    [
        (PRE_FIXTURE_ENGINE, 0o755, "does not support compile --rulespec-root"),
        ("#!/bin/sh\nexit 1\n", 0o755, "does not support compile --rulespec-root"),
        (FAKE_ENGINE, 0o644, "is not an executable file"),
    ],
)
def test_an_engine_that_cannot_compile_with_explicit_roots_is_unusable(
    monkeypatch, tmp_path, script, mode, expected
):
    _engine(tmp_path / "engine", script=script).chmod(mode)
    monkeypatch.setenv(real_engine.ROOT_ENV, str(tmp_path / "engine"))

    engine, why = real_engine.locate()

    assert engine is None
    assert expected in why


# The pin


def test_the_pin_holds_for_a_clean_checkout_at_that_commit(monkeypatch, tmp_path):
    root = tmp_path / "engine"
    head = _committed_checkout(root)
    monkeypatch.setenv(real_engine.ROOT_ENV, str(root))
    monkeypatch.setenv(real_engine.REF_ENV, head)

    engine, why = real_engine.locate()

    assert why == ""
    assert engine.root == root.resolve()
    assert f"commit {head}" in real_engine.describe()


@pytest.mark.parametrize(
    "breakage",
    ["other commit", "short sha", "tracked change", "not a checkout", "nested"],
)
def test_the_pin_fails_closed(monkeypatch, tmp_path, breakage):
    root = tmp_path / "engine"
    head = _committed_checkout(root)
    pin = head
    if breakage == "other commit":
        pin = "0" * 40
    elif breakage == "short sha":
        pin = head[:12]
    elif breakage == "tracked change":
        (root / "Cargo.toml").write_text("changed\n")
    elif breakage == "not a checkout":
        root = tmp_path / "plain"
        _engine(root)
    elif breakage == "nested":
        # A build directory inside some other repository at the right commit.
        root = root / "vendored"
        _engine(root)
    monkeypatch.setenv(real_engine.ROOT_ENV, str(root))
    monkeypatch.setenv(real_engine.REF_ENV, pin)

    engine, why = real_engine.locate()

    assert engine is None
    assert real_engine.REF_ENV in why


# The fixture


def test_require_skips_without_an_engine():
    with pytest.raises(pytest.skip.Exception, match=real_engine.UNAVAILABLE):
        real_engine.require()


def test_require_fails_without_an_engine_when_required(monkeypatch):
    monkeypatch.setenv(real_engine.REQUIRE_ENV, "1")

    with pytest.raises(pytest.fail.Exception, match=real_engine.REQUIRE_ENV):
        real_engine.require()


@pytest.mark.parametrize("value", ["", "0", "true", "yes", " 1"])
def test_only_the_exact_flag_requires_the_engine(monkeypatch, value):
    monkeypatch.setenv(real_engine.REQUIRE_ENV, value)

    assert real_engine.required() is False


# The guard

# Wordings tests used before tests/real_engine.py, and #1852's fixture.
LEGACY_ENGINE_SKIPS = [
    "local axiom-rules-engine binary is not built",
    "local explicit-root axiom-rules-engine binary is not built",
    "current axiom-rules-engine binary is not available",
    "set AXIOM_RULES_ENGINE_BINARY for the real engine contract test",
    "axiom-locate is unavailable",
    "a built axiom-rules-engine is unavailable",
]
# Skips in this suite that mention engines or checkouts but are not about a
# missing axiom-rules-engine binary.
OTHER_SKIPS = [
    "axiom-rules-engine repo not present",
    "rulespec-us repo not present",
    "rulespec checkouts are not available",
    "AXIOM_CORE_BIN is unset; provide a real axiom-core executable",
    "Node.js is not installed",
    "Go is required to build the signing supervisor",
    "macOS /var system alias is unavailable",
    "PolicyEngine US is not installed",
]
ENGINE_REASONS = st.one_of(
    st.sampled_from(LEGACY_ENGINE_SKIPS),
    st.text().map(lambda text: f"{real_engine.UNAVAILABLE}: {text}"),
)


@pytest.mark.parametrize("reason", LEGACY_ENGINE_SKIPS)
def test_legacy_engine_skip_wordings_are_recognized(reason):
    assert real_engine.ENGINE_ABSENT_REASON.search(f"Skipped: {reason}")


@pytest.mark.parametrize("reason", OTHER_SKIPS)
def test_other_skips_are_not_engine_skips(reason):
    assert not real_engine.ENGINE_ABSENT_REASON.search(f"Skipped: {reason}")


@given(reason=st.text(), marked=st.booleans(), required=st.booleans())
def test_only_engine_skips_ever_become_failures(reason, marked, required):
    problem = real_engine.skip_problem(reason, marked=marked, required=required)
    if not real_engine.ENGINE_ABSENT_REASON.search(reason):
        assert problem is None


@given(reason=ENGINE_REASONS, required=st.booleans())
def test_an_unmarked_engine_skip_is_always_a_failure(reason, required):
    problem = real_engine.skip_problem(reason, marked=False, required=required)

    assert problem is not None and "real_engine marker" in problem


@given(reason=ENGINE_REASONS)
def test_a_marked_engine_skip_fails_exactly_when_required(reason):
    assert real_engine.skip_problem(reason, marked=True, required=False) is None
    assert real_engine.skip_problem(reason, marked=True, required=True) == (
        f"skipped under {real_engine.REQUIRE_ENV}=1"
    )


def _skip_reasons(path: Path) -> list[str]:
    reasons = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        texts = []
        if name in {"pytest.skip", "pytest.fail", "pytest.xfail"} and node.args:
            texts.append(node.args[0])
        if name.split(".")[-1] in {"skip", "skipif", "xfail", "importorskip"}:
            texts.extend(k.value for k in node.keywords if k.arg == "reason")
        reasons.extend(
            constant.value
            for text in texts
            for constant in ast.walk(text)
            if isinstance(constant, ast.Constant) and isinstance(constant.value, str)
        )
    return reasons


def test_no_test_writes_its_own_engine_skip():
    # A hand-written skip is invisible to `-m real_engine`; the runtime guard
    # fails it, and this catches it before CI does.
    tests = Path(__file__).parent
    offenders = [
        f"{path.name}: {reason}"
        for path in sorted(tests.glob("test_*.py"))
        if path.name != Path(__file__).name
        for reason in _skip_reasons(path)
        if real_engine.ENGINE_ABSENT_REASON.search(reason)
    ]
    assert offenders == []


def test_the_audit_reads_skip_reasons(tmp_path):
    sample = tmp_path / "test_sample.py"
    sample.write_text(
        textwrap.dedent(
            """
            import pytest

            @pytest.mark.skipif(True, reason="axiom-locate is unavailable")
            def test_a():
                pytest.skip(f"local axiom-rules-engine binary is not built {1}")
            """
        )
    )

    assert sorted(_skip_reasons(sample)) == [
        "axiom-locate is unavailable",
        "local axiom-rules-engine binary is not built ",
    ]


# The hooks, end to end in a separate pytest run

SAMPLE_TESTS = """
import pytest


@pytest.fixture
def wrapped(real_engine):
    return real_engine


def test_direct(real_engine):
    assert real_engine.binary.is_file()


def test_transitive(wrapped):
    assert wrapped.root.is_dir()


def test_handwritten_skip():
    pytest.skip("local axiom-rules-engine binary is not built")


def test_unrelated_skip():
    pytest.skip("Node.js is not installed")


@pytest.mark.xfail(reason="axiom-locate is unavailable", run=False)
def test_expected_failure():
    raise AssertionError


def test_plain():
    pass
"""


def _run_sample(tmp_path: Path, env: dict[str, str], *args: str):
    project = tmp_path / "project"
    if not project.exists():
        project.mkdir()
        (project / "conftest.py").write_text(
            "from tests.real_engine import (  # noqa: F401\n"
            "    pytest_collection_modifyitems,\n"
            "    pytest_configure,\n"
            "    pytest_report_header,\n"
            "    pytest_runtest_makereport,\n"
            "    real_engine,\n"
            ")\n"
        )
        (project / "test_sample.py").write_text(SAMPLE_TESTS)
    report = tmp_path / f"report-{len(list(tmp_path.glob('report-*')))}.xml"
    base = {
        key: value
        for key, value in os.environ.items()
        if key not in ENGINE_ENV and not key.startswith(("PYTEST_", "COV_"))
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
        env={**base, "PYTHONPATH": str(Path(__file__).resolve().parents[1]), **env},
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


def test_hooks_without_an_engine(tmp_path):
    completed, outcomes = _run_sample(
        tmp_path, {real_engine.ROOT_ENV: str(tmp_path / "missing")}
    )

    assert f"{real_engine.UNAVAILABLE}: {real_engine.ROOT_ENV}=" in completed.stdout
    assert outcomes == {
        "test_direct": "skipped",
        "test_transitive": "skipped",
        "test_handwritten_skip": "failure",
        "test_unrelated_skip": "skipped",
        "test_expected_failure": "skipped",
        "test_plain": "passed",
    }, completed.stdout
    assert "without the real_engine marker" in completed.stdout


def test_hooks_fail_engine_tests_when_the_engine_is_required(tmp_path):
    completed, outcomes = _run_sample(tmp_path, {real_engine.REQUIRE_ENV: "1"})

    assert outcomes == {
        "test_direct": "error",
        "test_transitive": "error",
        "test_handwritten_skip": "failure",
        "test_unrelated_skip": "skipped",
        "test_expected_failure": "skipped",
        "test_plain": "passed",
    }, completed.stdout
    assert completed.returncode == 1


def test_hooks_run_engine_tests_against_the_configured_engine(tmp_path):
    binary = _engine(tmp_path / "engine")
    completed, outcomes = _run_sample(
        tmp_path,
        {
            real_engine.REQUIRE_ENV: "1",
            real_engine.ROOT_ENV: str(tmp_path / "engine"),
        },
        "-m",
        real_engine.MARKER,
    )

    assert f"real axiom-rules-engine: {binary.resolve()}" in completed.stdout
    assert outcomes == {"test_direct": "passed", "test_transitive": "passed"}, (
        completed.stdout
    )
    assert completed.returncode == 0


def test_hooks_let_the_main_job_deselect_every_engine_test(tmp_path):
    completed, outcomes = _run_sample(
        tmp_path,
        {real_engine.ROOT_ENV: str(tmp_path / "missing")},
        "-m",
        f"not {real_engine.MARKER}",
    )

    assert set(outcomes) == {
        "test_handwritten_skip",
        "test_unrelated_skip",
        "test_expected_failure",
        "test_plain",
    }, completed.stdout
