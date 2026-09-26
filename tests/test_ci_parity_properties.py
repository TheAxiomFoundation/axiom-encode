"""Property and differential tests for the invariants ci parity relies on.

Each property holds for every input, not just the examples elsewhere:

- the verdict never improves on a failed gate or a mismatched dependency;
- a gate fails exactly when a shard fails, and keeps every shard's output;
- no environment filter lets a withheld name through or drops another one;
- the heredoc and ``python3 -c '...'`` extractors agree with bash;
- the workflow YAML loader differs from SafeLoader only on YAML 1.1 booleans;
- every pair of tracked names this filesystem merges is reported;
- jurisdiction directories are visited in bash's ``for dir in */`` order.
"""

from __future__ import annotations

import itertools
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import yaml
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from axiom_encode import ci_parity
from axiom_encode.ci_parity import (
    SUPERVISOR_FORWARDED_ENVIRONMENT,
    SUPERVISOR_GIT_ENVIRONMENT,
    DependencyMismatch,
    GateResult,
    GateSpec,
    _aggregate,
    _colliding_paths,
    _git_environment,
    _glob_directories,
    _heredoc_bodies,
    _map_result_paths,
    _repository_test_environment,
    _run_cli,
    _run_embedded_python,
    _single_quoted_python,
    _WorkflowYamlLoader,
    ci_verdict,
)

VERDICT_RANK = {"PASS": 0, "PASS-WITH-MISMATCHED-DEPS": 1, "FAIL": 2}
SPEC = GateSpec("gate", "Gate", "subcommand", (), "1-2")

mismatches = st.lists(
    st.builds(
        DependencyMismatch,
        st.sampled_from(["axiom-encode", "axiom-corpus", "rulespec-us"]),
        st.text("0123456789abcdef", min_size=1, max_size=8),
        st.text("0123456789abcdef", min_size=1, max_size=8),
    ),
    max_size=3,
)


@given(passed=st.booleans(), found=mismatches, extra=mismatches)
def test_verdict_never_improves_on_failures_or_mismatches(
    passed: bool, found: list[DependencyMismatch], extra: list[DependencyMismatch]
) -> None:
    verdict = ci_verdict(passed, found)

    assert (verdict == "FAIL") == (not passed)
    assert (verdict == "PASS") == (passed and not found)
    # More mismatches, or a failed gate, can only make the verdict worse.
    assert VERDICT_RANK[ci_verdict(passed, found + extra)] >= VERDICT_RANK[verdict]
    assert ci_verdict(False, found + extra) == "FAIL"


shard_names = st.sampled_from(["us", "us-ak", "us-ca", "us-ny", "us-tx"])
outcomes = st.lists(
    st.tuples(
        shard_names,
        st.integers(min_value=0, max_value=255),
        st.text(st.characters(exclude_categories=("Cs",)), max_size=40),
    ),
    min_size=1,
    max_size=6,
)


@given(outcomes=outcomes, matrix_size=st.integers(min_value=1, max_value=4))
def test_aggregate_fails_exactly_when_a_shard_fails(
    outcomes: list[tuple[str, int, str]], matrix_size: int
) -> None:
    run: Any = SimpleNamespace(
        plan=SimpleNamespace(matrix=tuple(f"shard-{i}" for i in range(matrix_size)))
    )
    result = _aggregate(SPEC, run, outcomes, ["command"])
    labelled = matrix_size > 1

    assert (result.status == "FAIL") == any(code for _, code, _ in outcomes)
    assert (result.gate, result.name, result.command) == ("gate", "Gate", ["command"])
    # Every failing shard is reported, and only failing shards are.
    failing = [(shard, text) for shard, code, text in outcomes if code]
    assert bool(result.failures) == bool(failing)
    if labelled:
        for shard, _ in failing:
            assert any(line.startswith(f"[{shard}] ") for line in result.failures)
        assert all(
            any(line.startswith(f"[{shard}] ") for shard, _ in failing)
            for line in result.failures
        )
    # No output line is lost, and each carries its shard label when sharded.
    expected = "".join(
        "".join(
            f"[{shard}] {line}\n" if labelled else f"{line}\n"
            for line in text.splitlines()
        )
        for shard, _, text in outcomes
    )
    assert result.output == expected


environment_names = st.one_of(
    st.tuples(
        st.sampled_from(
            [
                "",
                "GIT_",
                "PYTHON",
                "PYTEST_",
                "GITHUB_",
                "RUNNER_",
                "AXIOM_",
                "AXIOM_ENCODE_",
            ]
        ),
        st.text("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_", min_size=1, max_size=12),
    ).map("".join),
    st.sampled_from(sorted(SUPERVISOR_FORWARDED_ENVIRONMENT)),
    st.sampled_from(["GIT_DIR", "GIT_CONFIG_COUNT", "PYTHONOPTIMIZE", "HOME"]),
)
environment_values = st.text(
    st.characters(exclude_categories=("Cs",), exclude_characters="\x00"),
    max_size=20,
)
environments = st.dictionaries(environment_names, environment_values, max_size=12)


def _without(environment: dict[str, str], *prefixes: str) -> dict[str, str]:
    return {
        name: value
        for name, value in environment.items()
        if not name.startswith(prefixes)
    }


@given(environment=environments)
def test_git_and_repository_test_environments_withhold_exactly_their_prefixes(
    environment: dict[str, str],
) -> None:
    with mock.patch.dict(os.environ, environment, clear=True):
        git_environment = _git_environment()
        test_environment = _repository_test_environment()

    assert git_environment == _without(environment, "GIT_")
    assert test_environment == _without(environment, "GIT_", "PYTEST_", "PYTHON")


@given(environment=environments, step=environments)
def test_embedded_python_sees_the_step_environment_and_no_runner_context(
    environment: dict[str, str], step: dict[str, str]
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        seen.update(kwargs)
        return subprocess.CompletedProcess(args, 0, b"", b"")

    with (
        mock.patch.dict(os.environ, environment, clear=True),
        mock.patch.object(ci_parity.subprocess, "run", fake_run),
    ):
        _run_embedded_python("pass", cwd=Path("."), environment=step)

    expected = _without(environment, "GIT_", "GITHUB_", "RUNNER_", "PYTHON")
    expected.update(step)
    assert seen["env"] == expected


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(environment=environments, step=environments)
def test_supervised_cli_sees_the_supervisor_child_environment(
    environment: dict[str, str], step: dict[str, str]
) -> None:
    seen: dict[str, str] = {}

    def fake_main() -> None:
        seen.update(os.environ)

    with (
        mock.patch.dict(os.environ, environment, clear=True),
        mock.patch("axiom_encode.cli.main", fake_main),
    ):
        code, _ = _run_cli(["validate"], environment=step, supervised=True)
        restored = dict(os.environ)

    expected = {
        name: value
        for name, value in environment.items()
        if not (
            name.startswith("AXIOM_") and name not in SUPERVISOR_FORWARDED_ENVIRONMENT
        )
        and not name.startswith("GIT_")
    }
    expected.update(SUPERVISOR_GIT_ENVIRONMENT)
    expected.update(step)
    assert code == 0
    assert seen == expected
    # The in-process run leaves the caller's environment as it found it.
    assert restored == environment


heredoc_tags = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,7}", fullmatch=True)
body_lines = st.lists(
    st.text(
        st.characters(exclude_categories=("Cs",), exclude_characters="\x00\n"),
        max_size=30,
    ),
    max_size=6,
)


def _bash(script: str) -> str:
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, check=True
    ).stdout.decode("utf-8", "surrogateescape")


@settings(deadline=None, max_examples=60)
@given(tag=heredoc_tags, lines=body_lines)
def test_heredoc_extraction_matches_bash(tag: str, lines: list[str]) -> None:
    assume(tag not in lines)
    body = "".join(f"{line}\n" for line in lines)
    script = f"cat <<'{tag}'\n{body}{tag}\n"

    assert _heredoc_bodies(script) == [_bash(script)]


@settings(deadline=None, max_examples=60)
@given(tags=st.lists(heredoc_tags, min_size=1, max_size=3), bodies=st.data())
def test_heredoc_extraction_keeps_order_and_skips_markers_inside_bodies(
    tags: list[str], bodies: st.DataObject
) -> None:
    parts = []
    expected = []
    for tag in tags:
        lines = bodies.draw(body_lines.filter(lambda drawn, tag=tag: tag not in drawn))
        # A heredoc marker inside a body is text, not the start of another one.
        lines = [*lines, "python3 - <<'NESTED'"] if tag != "NESTED" else lines
        body = "".join(f"{line}\n" for line in lines)
        expected.append(body)
        parts.append(f"cat <<'{tag}'\n{body}{tag}\n")
    script = "".join(parts)

    assert _heredoc_bodies(script) == expected
    assert "".join(expected) == _bash(script)


@settings(deadline=None, max_examples=60)
@given(
    program=st.text(
        st.characters(exclude_categories=("Cs",), exclude_characters="\x00'"),
        max_size=80,
    )
)
def test_single_quoted_python_matches_bash(program: str) -> None:
    run = f"set -euo pipefail\npython3 -c '{program}\n' \"$@\"\n"
    as_printf = run.replace("python3 -c ", "printf %s ", 1)

    assert _single_quoted_python(run) == f"{program}\n"
    assert _bash(as_printf) == f"{program}\n"


yaml_1_1_booleans = st.sampled_from(
    [
        f(word)
        for word in ("yes", "no", "on", "off", "y", "n")
        for f in (str.lower, str.upper, str.capitalize)
        if f(word) not in ("Y", "N")
    ]
)
plain_scalars = st.one_of(
    yaml_1_1_booleans,
    st.sampled_from(["true", "True", "TRUE", "false", "False", "FALSE"]),
    st.integers().map(str),
    st.floats(allow_nan=False, allow_infinity=False).map(repr),
    st.from_regex(r"[a-z][a-z0-9_-]{0,12}", fullmatch=True),
    st.sampled_from(["~", "null", "0x1f", "1_000", "2026-09-25", ".inf"]),
)


@given(scalar=plain_scalars)
def test_workflow_loader_differs_from_safe_loader_only_on_yaml_1_1_booleans(
    scalar: str,
) -> None:
    document = f"value: {scalar}\n"
    safe = yaml.safe_load(document)["value"]
    workflow = yaml.load(document, Loader=_WorkflowYamlLoader)["value"]

    if isinstance(safe, bool) and scalar not in (
        "true",
        "True",
        "TRUE",
        "false",
        "False",
        "FALSE",
    ):
        assert workflow == scalar
    else:
        assert workflow == safe
        assert type(workflow) is type(safe)


# Letters whose case or composition some filesystems fold together.
FOLDING_ALPHABET = "aAbBeEkKsSzZ019_-.éÉ́ßẞKÅÅå̊İıΣσςﬁ"
file_names = st.text(FOLDING_ALPHABET, min_size=1, max_size=4).filter(
    lambda name: name not in (".", "..")
)


def _filesystem_merges(directory: Path, first: str, second: str) -> bool:
    """Whether creating ``first`` makes ``second`` exist in a fresh directory."""

    with tempfile.TemporaryDirectory(dir=directory) as scratch:
        (Path(scratch) / first).write_text("")
        return (Path(scratch) / second).exists()


@settings(
    deadline=None,
    max_examples=150,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(names=st.lists(file_names, min_size=2, max_size=5, unique=True))
def test_every_name_pair_this_filesystem_merges_is_reported(
    tmp_path: Path, names: list[str]
) -> None:
    """Soundness: a tracked pair the filesystem would collapse is always reported.

    The fold may over-report (casefold maps some letters a filesystem keeps
    apart), which fails closed; it must never under-report. On a case- and
    normalization-sensitive filesystem nothing merges and nothing is reported.
    """

    groups = [set(group.split(" = ")) for group in _colliding_paths(names, tmp_path)]
    for first, second in itertools.combinations(names, 2):
        if _filesystem_merges(tmp_path, first, second):
            assert any({first, second} <= group for group in groups), (first, second)


@given(
    status=st.sampled_from(["PASS", "FAIL"]),
    text=st.lists(st.text(max_size=20), max_size=3),
    note=st.one_of(st.none(), st.text(max_size=20)),
)
def test_mapping_result_paths_never_changes_a_verdict(
    status: str, text: list[str], note: str | None
) -> None:
    checkout = Path("/tmp/axiom-ci-checkout-x/rulespec-us")
    source = Path("/work/rulespec-us")
    mentioned = [f"{checkout}/{item}" for item in text]
    result = GateResult("gate", "Gate", status, mentioned, mentioned, "", note)

    mapped = _map_result_paths(result, checkout, source)

    assert (mapped.gate, mapped.name, mapped.status) == ("gate", "Gate", status)
    assert mapped.failures == [f"{source}/{item}" for item in text]
    assert str(checkout) not in "".join(mapped.command)


directory_names = st.text("abuz09-_.", min_size=1, max_size=6).filter(
    lambda name: name not in (".", "..")
)


@settings(
    deadline=None,
    max_examples=60,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    names=st.lists(directory_names, min_size=1, max_size=8, unique=True),
    files=st.lists(directory_names, max_size=3, unique=True),
)
def test_glob_directories_visit_in_bash_order(
    tmp_path: Path, names: list[str], files: list[str]
) -> None:
    with tempfile.TemporaryDirectory(dir=tmp_path) as scratch:
        repo = Path(scratch)
        for name in names:
            (repo / name).mkdir()
        for name in files:
            if name not in names:
                (repo / name).write_text("")
        bash = subprocess.run(
            ["bash", "-c", 'for dir in */; do printf "%s\\n" "${dir%/}"; done'],
            cwd=repo,
            env={**os.environ, "LC_ALL": "C"},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
        # With no match bash leaves the pattern itself, which the workflow's
        # jurisdiction regex then rejects; it names no directory.
        if bash == ["*"]:
            bash = []

        assert [child.name for child in _glob_directories(repo)] == bash
