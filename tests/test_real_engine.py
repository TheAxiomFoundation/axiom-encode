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

from axiom_encode.engine_binding import (
    _engine_binary_candidates,
    engine_binding_receipt_path,
    write_engine_binding_receipt,
)
from axiom_encode.harness.validator_pipeline import ValidatorPipeline
from tests import real_engine

REPO = Path(__file__).resolve().parents[1]
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
    """A clean engine checkout whose release binary carries a receipt for HEAD."""
    binary = _engine(root)
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
    head = _git(root, "rev-parse", "HEAD")
    write_engine_binding_receipt(binary, head, "release")
    return head


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


def test_the_pin_holds_for_a_receipted_binary_in_a_clean_checkout(
    monkeypatch, tmp_path
):
    root = tmp_path / "engine"
    head = _committed_checkout(root)
    monkeypatch.setenv(real_engine.ROOT_ENV, str(root))
    monkeypatch.setenv(real_engine.REF_ENV, head)

    engine, why = real_engine.locate()

    assert why == ""
    assert engine.root == root.resolve()
    assert f"commit {head}" in real_engine.describe()


def _break_pin(root: Path, head: str, tmp_path: Path, breakage: str):
    """Return (root, pin) for ``breakage``; every one must fail closed."""
    release = root / "target" / "release" / real_engine.BINARY_NAME
    if breakage == "other commit":
        return root, "0" * 40
    if breakage == "short sha":
        return root, head[:12]
    if breakage == "tracked change":
        (root / "Cargo.toml").write_text("changed\n")
    elif breakage == "not a checkout":
        root = tmp_path / "plain"
        write_engine_binding_receipt(_engine(root), head, "release")
    elif breakage == "nested":
        # A receipted build inside some other repository at the right commit.
        root = root / "vendored"
        write_engine_binding_receipt(_engine(root), head, "release")
    elif breakage == "no receipt":
        engine_binding_receipt_path(release).unlink()
    elif breakage == "swapped binary":
        # Round-1 review: other bytes in the clean pinned checkout still print
        # --rulespec-root; only the receipt's digest tells them apart.
        release.write_text(FAKE_ENGINE + "# other bytes\n")
    elif breakage == "receipt for another commit":
        write_engine_binding_receipt(release, "1" * 40, "release")
    elif breakage == "receipt on another build":
        # ValidatorPipeline would run the unreceipted release binary.
        engine_binding_receipt_path(release).unlink()
        write_engine_binding_receipt(_engine(root, "debug"), head, "debug")
    else:
        raise AssertionError(breakage)
    return root, head


@pytest.mark.parametrize(
    "breakage",
    [
        "other commit",
        "short sha",
        "tracked change",
        "not a checkout",
        "nested",
        "no receipt",
        "swapped binary",
        "receipt for another commit",
        "receipt on another build",
    ],
)
def test_the_pin_fails_closed(monkeypatch, tmp_path, breakage):
    root = tmp_path / "engine"
    head = _committed_checkout(root)
    root, pin = _break_pin(root, head, tmp_path, breakage)
    monkeypatch.setenv(real_engine.ROOT_ENV, str(root))
    monkeypatch.setenv(real_engine.REF_ENV, pin)

    engine, why = real_engine.locate()

    assert engine is None
    assert real_engine.REF_ENV in why


@pytest.mark.parametrize("variable", [real_engine.ROOT_ENV, real_engine.BINARY_ENV])
def test_relative_paths_name_the_file_not_a_path_lookup(
    monkeypatch, tmp_path, variable
):
    # Round-1 review: a bare relative candidate was probed through PATH.
    root = tmp_path / "engine"
    binary = _engine(root, "bare")
    monkeypatch.chdir(root)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    relative = "." if variable == real_engine.ROOT_ENV else real_engine.BINARY_NAME
    monkeypatch.setenv(variable, relative)

    engine, why = real_engine.locate()

    assert why == ""
    assert engine.binary == binary.resolve()


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


def _run_sample(
    tmp_path: Path,
    env: dict[str, str],
    *args: str,
    tests: dict[str, str] | None = None,
):
    project = tmp_path / "project"
    if not project.exists():
        project.mkdir()
        for name, text in (tests or {}).items():
            (project / name).write_text(text)
        (project / "conftest.py").write_text(
            "from tests.real_engine import (  # noqa: F401\n"
            "    pytest_collection_modifyitems,\n"
            "    pytest_configure,\n"
            "    pytest_make_collect_report,\n"
            "    pytest_report_header,\n"
            "    pytest_runtest_makereport,\n"
            "    real_engine,\n"
            ")\n"
        )
        if tests is None:
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
        env={
            **base,
            "PYTHONPATH": os.pathsep.join([str(REPO), str(REPO / "src")]),
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


def test_hooks_fail_a_module_skipped_for_a_missing_engine(tmp_path):
    completed, outcomes = _run_sample(
        tmp_path,
        {real_engine.ROOT_ENV: str(tmp_path / "missing")},
        tests={
            "test_module_skip.py": (
                "import pytest\n"
                "pytest.skip('local axiom-rules-engine binary is not built',"
                " allow_module_level=True)\n"
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
    assert "skipped at collection for a missing engine" in completed.stdout
    assert "test_module_skip.py" in completed.stdout
    assert (
        "test_other_module_skip.py"
        not in completed.stdout.split("short test summary")[-1]
    )
    assert "test_never_collected" not in outcomes


def test_conftest_registers_the_real_engine_plugin(pytestconfig):
    # tests/conftest.py registers tests/real_engine.py as a plugin (it cannot
    # import the hooks: the shard split defines pytest_collection_modifyitems
    # too). Without it the real_engine fixture is missing, so every engine test
    # errors in the shards instead of running in the real-engine job.
    assert pytestconfig.pluginmanager.is_registered(real_engine)
