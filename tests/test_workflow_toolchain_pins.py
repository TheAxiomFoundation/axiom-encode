"""Every workflow's toolchain comes from a pin, and the test suite runs on the lock.

``ci.yml`` and ``signing-supervisor.yml`` used to build the test environment
with ``uv venv`` plus ``uv pip install -e ".[dev]"``, which ignores ``uv.lock``
and floats every ``>=`` dependency to the index's latest release. CI run
36166075448 (2026-09-25) ran the suite on sqlite-utils 4.2.1, cryptography
50.0.1 and supabase 2.31.0 while the lock, and so every production
``uv sync --locked``, pinned 3.39, 46.0.3 and 2.27.3. The lint job installed
the latest ruff rather than the locked one. ``ci.yml`` also tag-pinned its
actions, and five ``setup-uv`` steps in four workflows had no ``version`` or
pinned another release. These tests fail if any of that comes back.

Commands are found with ``tests/workflow_shell.py``, which reads each ``run:``
script as bash does and parses uv calls against the pinned uv's option table.
Its threat model is an accidental regression in a reviewed workflow written in
ordinary shell, not deliberately obfuscated bash. It fails closed on the
constructs its docstring lists, but a command assembled in a variable or a
script run by path is out of its reach.

Out of scope:
- what ``uv pip install --target`` stages for the verification and
  axiom-compose runtimes (the verification tree is locked by
  ``test_verification_site_packages_lock.py`` once #1712 lands);
- ``uv version --bump``, which re-locks without ``--frozen`` (#1730);
- programs named by a variable, such as ``"$PYBIN" -m pip``.
"""

from __future__ import annotations

import copy
import functools
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

from tests import workflow_shell
from tests.workflow_shell import Command, UnanalyzableScript

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
UV_OPTIONS = workflow_shell.load_uv_options()

# workflow -> jobs whose environment runs the repository's tests or linters.
TEST_ENVIRONMENTS = {
    "ci.yml": {"test", "lint"},
    "signing-supervisor.yml": {"build-and-test"},
}
LOCKED_DEV_SYNCS = {
    ("uv", "sync", "--locked", "--python", python, "--extra", "dev")
    for python in ("3.13", "${{matrix.python-version}}")
}
ACTION_PIN = re.compile(r"[^@\s]+@[0-9a-f]{40}|docker://[^@\s]+@sha256:[0-9a-f]{64}")
# Options (long names; the reader maps short ones) that add packages, upgrade
# them, or change how they resolve, whether or not the lock is also enforced.
LOCK_OVERRIDES = frozenset(
    {
        "--build-constraints",
        "--constraints",
        "--exclude-newer",
        "--exclude-newer-package",
        "--no-sources",
        "--no-sources-package",
        "--overrides",
        "--prerelease",
        "--resolution",
        "--upgrade",
        "--upgrade-group",
        "--upgrade-package",
        "--with",
        "--with-editable",
        "--with-executables-from",
        "--with-requirements",
    }
)
TOOLS = re.compile(r"pytest|py\.test|ruff|towncrier|pre-commit|python[0-9.]*")
PIP = re.compile(r"pip[0-9.]*")
VENV_TOOL = re.compile(r"(?:\./)?\.venv/bin/\S+")


@functools.cache
def _workflows() -> tuple[tuple[str, dict], ...]:
    return tuple(
        (path.name, yaml.safe_load(path.read_text(encoding="utf-8")))
        for path in sorted(WORKFLOWS.glob("*.y*ml"))
    )


@functools.cache
def _script_commands(script: str) -> tuple[Command, ...]:
    """What ``script`` runs, including what its ``uv run`` calls launch."""
    return tuple(workflow_shell.expand(workflow_shell.commands(script), UV_OPTIONS))


def _steps():
    for name, workflow in _workflows():
        for job_name, job in workflow["jobs"].items():
            for index, step in enumerate(job.get("steps", [])):
                yield name, job_name, index, step


def _action_steps():
    for name, job_name, index, step in _steps():
        if "uses" in step and not step["uses"].startswith("./"):
            yield name, job_name, index, step


def _commands(steps) -> list[tuple[str, Command]]:
    return [
        (f"{name}:{job}[{index}]", command)
        for name, job, index, step in steps
        for command in _script_commands(step.get("run", ""))
    ]


def _uv_invocations(steps=None):
    for where, command in _commands(_steps() if steps is None else steps):
        invocation = workflow_shell.uv_invocation(command, UV_OPTIONS)
        if invocation is not None:
            yield where, invocation


def _job_steps(workflow_name: str, job_name: str):
    workflow = dict(_workflows())[workflow_name]
    for index, step in enumerate(workflow["jobs"][job_name]["steps"]):
        yield workflow_name, job_name, index, step


def _uv(script: str) -> list[tuple[str, tuple[str, ...]]]:
    return [
        (invocation.subcommand, invocation.options)
        for command in workflow_shell.expand(
            workflow_shell.commands(script), UV_OPTIONS
        )
        if (invocation := workflow_shell.uv_invocation(command, UV_OPTIONS))
    ]


def _programs(script: str) -> list[str]:
    return [
        command.program
        for command in workflow_shell.expand(
            workflow_shell.commands(script), UV_OPTIONS
        )
        if command.words
    ]


def _runs_pip(command: Command) -> bool:
    """Whether ``command`` runs pip, directly or as ``python -m pip``."""
    if PIP.fullmatch(command.program):
        return True
    return bool(
        re.fullmatch(r"python[0-9.]*", command.program)
        and PIP.fullmatch(workflow_shell.python_module(command.words) or "")
    )


def _pip_installs(command: Command) -> bool:
    return _runs_pip(command) and "install" in command.words


def _stages_target(words: tuple[str, ...]) -> bool:
    return any(
        word in {"-t", "--target"} or word.startswith(("--target=", "-t"))
        for word in words
        if not word.startswith("--") or word.startswith("--target")
    )


def test_every_run_script_is_analyzable():
    unanalyzable = []
    for name, job, index, step in _steps():
        try:
            for command in _script_commands(step.get("run", "")):
                workflow_shell.uv_invocation(command, UV_OPTIONS)
        except UnanalyzableScript as error:
            unanalyzable.append(f"{name}:{job}[{index}] {error}")
    assert unanalyzable == []


def test_discovery_is_not_vacuous():
    names = {name for name, _ in _workflows()}
    assert set(TEST_ENVIRONMENTS) <= names
    assert len(names) >= 10
    assert sum(1 for _ in _action_steps()) >= 50
    setup_uv = [
        step
        for _, _, _, step in _action_steps()
        if step["uses"].startswith("astral-sh/setup-uv@")
    ]
    assert len(setup_uv) >= 8
    assert len(_commands(_steps())) >= 1000
    found = {
        (where.split("[")[0], invocation.subcommand)
        for where, invocation in _uv_invocations()
    }
    assert {
        ("ci.yml:test", "sync"),
        ("ci.yml:lint", "sync"),
        ("signing-supervisor.yml:build-and-test", "sync"),
        ("prepare-all-state-snap-queue.yml:prepare", "run"),
        ("finalize-signed-snap-queue.yml:finalize", "lock"),
        ("targeted-signed-reencode.yml:encode", "export"),
        ("bulk-encode.yml:preclassify", "pip install"),
    } <= found


# script -> the uv calls it makes, as (subcommand, long option names).
READER_CASES = {
    "uv sync --locked --python 3.13 --extra dev": [
        ("sync", ("--locked", "--python", "--extra"))
    ],
    # A global option before the subcommand.
    "uv --no-cache sync --python 3.13": [("sync", ("--no-cache", "--python"))],
    # Options after `uv run`'s command belong to that command.
    "uv run --frozen --python 3.13 python s.py -P --upgrade": [
        ("run", ("--frozen", "--python"))
    ],
    # A valued option is not mistaken for uv run's command, even a quoted `>`.
    "uv run --frozen --link-mode copy --with requests python s.py": [
        ("run", ("--frozen", "--link-mode", "--with"))
    ],
    'uv run --frozen --env-file "my settings.env" --with requests python s.py': [
        ("run", ("--frozen", "--env-file", "--with"))
    ],
    "uv run --frozen --env-file '>' --with requests python s.py": [
        ("run", ("--frozen", "--env-file", "--with"))
    ],
    # Short options: attached values and clusters, reported by long name.
    "uv run --frozen -wrequests python s.py": [("run", ("--frozen", "--with"))],
    "uv sync --frozen -Pcryptography": [("sync", ("--frozen", "--upgrade-package"))],
    "uv sync --frozen -qU": [("sync", ("--frozen", "--quiet", "--upgrade"))],
    "uv sync --frozen --upgrade-group=dev": [("sync", ("--frozen", "--upgrade-group"))],
    # A hidden alias resolves to its option.
    "uv pip install --target t --requirement r.txt": [
        ("pip install", ("--target", "--requirements"))
    ],
    # Comments and quoted operators are data.
    "uv sync --python 3.13  # --frozen": [("sync", ("--python",))],
    'echo "step #1"; uv sync --python 3.13': [("sync", ("--python",))],
    'echo "documentation # text"; uv tool run ruff': [("tool run", ())],
    "echo ';' uv sync": [],
    'echo "example; uv tool run ruff"': [],
    # Substitutions, including inside arithmetic, run.
    'x="$(uv export --no-dev | grep y)"': [("export", ("--no-dev",))],
    "cat <(uv export --locked)": [("export", ("--locked",))],
    "echo `uv lock`": [("lock", ())],
    "echo $(( $(uv sync) + 1 ))": [("sync", ())],
    # Wrappers.
    "timeout -k 5 600 uv sync": [("sync", ())],
    "sudo -E -u runner env -i PATH=/bin A=1 uv lock --offline": [
        ("lock", ("--offline",))
    ],
    "nice -n 5 nohup uv sync": [("sync", ())],
    "command -- uv sync": [("sync", ())],
    "printf '%s\\n' r.txt | xargs -n 1 uv pip sync": [("pip sync", ())],
    "env -S'uv sync'": [("sync", ())],
    "env --split-string='uv sync --frozen'": [("sync", ("--frozen",))],
    # Nested scripts: shells with -c, su -c, eval, trap, find -exec.
    "bash -c 'uv pip sync requirements.txt'": [("pip sync", ())],
    'sh -ec "uv sync --frozen"': [("sync", ("--frozen",))],
    "bash -o pipefail -c 'uv pip install x'": [("pip install", ())],
    'bash -euo pipefail -c "uv sync"': [("sync", ())],
    "bash -c -e 'uv lock'": [("lock", ())],
    "su -c 'uv sync' runner": [("sync", ())],
    "eval 'uv sync'": [("sync", ())],
    "trap 'uv lock' EXIT": [("lock", ())],
    "trap - EXIT": [],
    "find . -name x -exec uv sync \\;": [("sync", ())],
    # Grouping, functions and control flow.
    "{ uv sync; }": [("sync", ())],
    "f() { uv sync; }": [("sync", ())],
    "function setup { uv sync; }; setup": [("sync", ())],
    "if ! uv export --locked --no-dev; then exit 1; fi": [
        ("export", ("--locked", "--no-dev"))
    ],
    # A case pattern is data; its arm runs.
    "case x in a|uv) uv lock ;; *) echo ;; esac": [("lock", ())],
    'case "$(uv version --short)" in\n  1.*) echo one ;;\nesac': [
        ("version", ("--short",))
    ],
    # What `uv run` and `uvx` launch is read too.
    "uv run --frozen uv pip install requests": [
        ("run", ("--frozen",)),
        ("pip install", ()),
    ],
    "uv run --frozen bash -c 'uv pip sync r.txt'": [
        ("run", ("--frozen",)),
        ("pip sync", ()),
    ],
    "uvx --from ruff ruff check": [("uvx", ("--from",))],
    # Text that mentions uv runs nothing.
    "echo uv sync": [],
    "printf '%s\\n' 'uv sync'": [],
    "command -v uv": [],
    "echo $(( 1 + 2 ))": [],
    "cat <<'DOC'\nuv sync\nDOC\necho done": [],
    # Heredoc terminators match whole lines, as in bash.
    "cat <<'DOC'\nuv sync\nDOC \nDOC\necho done": [],
    "cat <<'DOC'\n  DOC\nDOC\nuv lock": [("lock", ())],
    "cat <<-'DOC'\nuv sync\n\tDOC\nuv lock": [("lock", ())],
    # An unquoted heredoc body is data, but its substitutions run.
    "cat <<DOC\nuv sync\n$(uv lock)\nDOC": [("lock", ())],
    "a=1 \\\n  uv sync --frozen": [("sync", ("--frozen",))],
    # Round 4: substitutions in nested scripts and what uv run launches.
    'bash -c "echo $(date)"': [],
    'trap "rm -f $(mktemp)" EXIT': [],
    'uv run --frozen python s.py "$(ls q)"': [("run", ("--frozen",))],
    'uv run --frozen echo "$(uv lock)"': [("lock", ()), ("run", ("--frozen",))],
    "bash -c 'echo $(uv lock)'": [("lock", ())],
    # Wrapper option syntax: long spellings, clusters, attached values.
    "sudo -Eu runner uv sync --frozen": [("sync", ("--frozen",))],
    "sudo --user runner uv pip install x": [("pip install", ())],
    "sudo --preserve-env=PATH uv sync": [("sync", ())],
    "xargs -rn 1 uv pip sync": [("pip sync", ())],
    "xargs --max-args=1 uv pip sync": [("pip sync", ())],
    "env -iu FOO uv sync": [("sync", ())],
    "stdbuf --output=L uv sync": [("sync", ())],
    "time -p uv sync": [("sync", ())],
    "nice -10 uv sync": [("sync", ())],
    # env -S splits its value; later arguments stay whole.
    "env -S 'bash -c' 'eval uv sync'": [("sync", ())],
    # Quoting: $'...' strings, and backslashes inside backticks.
    "$'uv' sync": [("sync", ())],
    'echo `echo "\\$(uv sync)"`': [("sync", ())],
    "IFS=$'\\t' read -r a b": [],
    # coproc runs its command.
    "coproc uv sync": [("sync", ())],
    "coproc worker { uv sync; }": [("sync", ())],
    # A descriptor prefix only goes with `<` and `>`, not `&>`.
    "uv run --frozen --python 3&>/dev/null --with requests python s.py": [
        ("run", ("--frozen", "--python", "--with"))
    ],
    # case after a control keyword, and substitutions in patterns.
    "if true; then case x in a|uv) uv lock ;; esac; fi": [("lock", ())],
    "for f in a; do case $f in a) uv lock ;; esac; done": [("lock", ())],
    'case $x in "$(uv sync)") echo ;; esac': [("sync", ())],
    'echo "$(case "$RUNNER_OS" in Linux) uv sync ;; esac)"': [("sync", ())],
    "echo \"$(printf 'use case')\"; uv lock": [("lock", ())],
    # Arrays, [[ ]] tests and (( )) are data, apart from their substitutions.
    "tools=(uv git jq)": [],
    "declare -a pkgs=(pytest ruff)": [],
    "arr=( $(uv lock) )": [("lock", ())],
    "[[ $c =~ ^(uv|pip)$ ]] && echo ok": [],
    "(( x <<= 1 ))": [],
    "if (( ${#a[@]} == 0 )); then uv lock; fi": [("lock", ())],
    "(( $(uv lock) > 0 ))": [("lock", ())],
    # Placeholder text cannot collide with script text.
    'echo "__SUBST0__"': [],
    # Round 5: env option order, bare `-`, and -S after other options.
    'env - PATH="$PATH" uv pip install pandas': [("pip install", ())],
    "env -i -S 'uv sync --frozen'": [("sync", ("--frozen",))],
    "env -iS'uv sync' A=1": [("sync", ())],
    # Bash's $'...' escapes, including a one-digit \x and malformed ones.
    "IFS=$'\\x9' read -r a b": [],
    "printf $'\\N\\xZZ\\e[1m'": [],
    "$'\\x75v' sync": [("sync", ())],
    # `builtin`, and heredoc delimiters that are any word.
    "builtin eval 'uv sync'": [("sync", ())],
    "cat <<END-OF\nuv sync\nEND-OF\nuv lock": [("lock", ())],
    "uv --version": [("", ("--version",))],
    "echo \"$(echo 'if case x in')\"": [],
}


@pytest.mark.parametrize(("script", "expected"), READER_CASES.items())
def test_reader_finds_the_uv_calls_a_script_makes(script, expected):
    assert _uv(script) == expected


@pytest.mark.parametrize(
    "script",
    [
        "uv sync --frozen --no-such-option",
        "uv run --frozen --no-such-option value python s.py",
        "uv sync -Z",
        "uv --preview sync --frozen",
        "uv cache clean",
        "echo 'unterminated",
        "x=$(uv sync",
        "cat <<EOF\nno terminator",
        # A shell that reads its script from stdin or a file.
        "bash -c",
        "bash <<'EOF'\nuv sync\nEOF",
        "bash <<< 'uv sync'",
        'sh <<< "uv tool run ruff"',
        "echo uv sync | bash",
        "bash script.sh",
        "su root",
        # A program produced by a substitution.
        "$(command -v uv) sync",
        "`uv lock`",
        # Unknown options to a wrapper, and runners this reader cannot follow.
        "sudo --bogus uv sync",
        "xargs -Z uv pip sync",
        "setsid uv sync",
        "flock /tmp/lock uv sync",
        "runuser -u x -- uv sync",
        # A case left open inside a substitution.
        "echo $(case x in a) echo",
        # A script file, a program from a workflow expression, bad env -S text.
        "source scripts/env.sh",
        ". ./env.sh",
        "${{ inputs.cmd }} sync",
        'env -S "uv \'sync"',
        # A uv option missing its value.
        "uv --directory",
        "uv sync --frozen --python",
        "uv run --frozen -p",
    ],
)
def test_reader_fails_closed_on_what_it_cannot_follow(script):
    with pytest.raises(UnanalyzableScript):
        _uv(script)


def test_reader_resolves_the_program_behind_quotes_and_wrappers():
    assert _programs("'pytest' tests/") == ["pytest"]
    assert _programs("bash -c 'pytest tests/'") == ["pytest"]
    assert _programs("cd tests && sudo -E /usr/bin/python3 -m pytest") == [
        "cd",
        "python3",
    ]
    assert _programs("cmd 2>&1 | tee log >/dev/null") == ["cmd", "tee"]
    assert _programs("uv run --frozen --module pip install x") == [
        "uv",
        "python",
    ]
    # A substitution runs just before the command that contains it.
    assert _programs('uv sync; echo "$(.venv/bin/python --version)"') == [
        "uv",
        "python",
        "echo",
    ]
    assert _programs("case x in a|pytest) echo ok;; esac") == ["echo"]
    assert _programs("if true; then case x in a|pytest) echo ok;; esac; fi") == [
        "true",
        "echo",
    ]
    assert _programs("pkgs=(pytest ruff)") == []
    assert _programs("[[ $c =~ ^(uv|pip)$ ]]") == ["[["]


def test_nested_scripts_inherit_assignments():
    [command] = workflow_shell.commands("UV_UPGRADE=1 bash -c 'uv sync --frozen'")
    assert command.words == ("uv", "sync", "--frozen")
    assert command.assignments == ("UV_UPGRADE=1",)
    [command] = workflow_shell.commands("sudo -E A=1 env B=2 uv sync")
    assert command.assignments == ("A=1", "B=2")


@pytest.mark.parametrize(
    ("script", "flagged"),
    [
        ('{\n  echo "UV_UPGRADE=1"\n} >> "$GITHUB_ENV"', True),
        ('{ echo "UV_UPGRADE=1"; } >> "$GITHUB_ENV"', True),
        ('echo "UV_UPGRADE=1" | tee -a "$GITHUB_ENV"', True),
        ('if true; then echo UV_X=1; fi >> "$GITHUB_ENV"', True),
        ("bash -c 'echo UV_UPGRADE=1' >> \"$GITHUB_ENV\"", True),
        ("printf '%s=1\\n' UV_UPGRADE >> \"$GITHUB_ENV\"", True),
        ("cat >> \"$GITHUB_ENV\" <<'EOF'\nUV_NO_SYNC=1\nEOF", True),
        ('{ echo "UV_NO_SYNC<<EOF"; echo 1; echo EOF; } >> "$GITHUB_ENV"', True),
        ("cat >> \"$GITHUB_ENV\" <<'X'\nUV_NO_SYNC<<EOF\n1\nEOF\nX", True),
        ('( echo UV_UPGRADE=1 ) >> "$GITHUB_ENV"', True),
        ('exec >> "$GITHUB_ENV"\necho UV_UPGRADE=1', True),
        ('while false; do :; done; echo UV_X=1 | tee "$GITHUB_ENV"', True),
        ('echo "LOCKED_UV_VERSION=0.11.7" >> "$GITHUB_ENV"', False),
        (
            'echo "UV_CACHE_DIR=$UV_CACHE_DIR"\necho "VENV=.venv" >> "$GITHUB_ENV"',
            False,
        ),
        ('echo UV_X=1 | grep -c X >> "$GITHUB_ENV"', False),
        ('echo "CACHE=$UV_CACHE_DIR" >> "$GITHUB_ENV"', False),
        ("echo UV_CACHE_DIR=/tmp", False),
    ],
)
def test_github_env_writes_of_uv_variables_are_caught(monkeypatch, script, flagged):
    workflows = copy.deepcopy(_workflows())
    dict(workflows)["ci.yml"]["jobs"]["lint"]["steps"].append({"run": script})
    monkeypatch.setattr(
        "tests.test_workflow_toolchain_pins._workflows", lambda: workflows
    )
    if flagged:
        with pytest.raises(AssertionError):
            test_no_workflow_sets_uv_environment_variables()
    else:
        test_no_workflow_sets_uv_environment_variables()


def test_a_uv_version_step_is_allowed_in_a_test_job(monkeypatch):
    workflows = copy.deepcopy(_workflows())
    dict(workflows)["ci.yml"]["jobs"]["test"]["steps"].append({"run": "uv --version"})
    monkeypatch.setattr(
        "tests.test_workflow_toolchain_pins._workflows", lambda: workflows
    )
    test_test_environment_is_the_locked_dev_set("ci.yml", "test")


def test_github_env_writes_keep_their_text():
    [command] = workflow_shell.commands(
        "cat >> \"$GITHUB_ENV\" <<'EOF'\nUV_NO_SYNC=1\nEOF"
    )
    assert command.redirects == ("$GITHUB_ENV",)
    assert command.stdin_text == ("UV_NO_SYNC=1",)


def test_python_module_follows_python_option_syntax():
    module = workflow_shell.python_module
    assert module(("python", "-m", "pip", "install")) == "pip"
    assert module(("python", "-Impip", "install")) == "pip"
    assert module(("python", "-X", "dev", "-m", "pip")) == "pip"
    assert module(("python", "-c", 'print("python -m pip install")')) is None
    assert module(("python", "script.py", "-m", "pip")) is None


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is not installed")
def test_uv_option_table_matches_the_pinned_uv():
    installed = subprocess.run(
        ["uv", "--version"], capture_output=True, text=True, check=True
    ).stdout.split()[1]
    if installed != UV_OPTIONS["uv_version"]:
        pytest.skip(f"uv {installed} is installed; the table is for the pinned uv")
    assert workflow_shell.uv_option_table() == UV_OPTIONS


def test_uv_option_table_is_for_the_uv_every_workflow_pins():
    versions = {
        (step.get("with") or {}).get("version")
        for _, _, _, step in _action_steps()
        if step["uses"].startswith("astral-sh/setup-uv@")
    }
    assert versions == {UV_OPTIONS["uv_version"]}


def test_every_action_is_pinned_to_a_full_commit_sha():
    unpinned = [
        f"{name}:{job}[{index}] {step['uses']}"
        for name, job, index, step in _action_steps()
        if not ACTION_PIN.fullmatch(step["uses"])
    ]
    assert unpinned == []


def test_reusable_workflow_calls_are_pinned_to_a_full_commit_sha():
    unpinned = [
        f"{name}:{job_name} {job['uses']}"
        for name, workflow in _workflows()
        for job_name, job in workflow["jobs"].items()
        if "uses" in job
        and not job["uses"].startswith("./")
        and not ACTION_PIN.fullmatch(job["uses"])
    ]
    assert unpinned == []


def test_each_action_resolves_to_one_commit_across_workflows():
    commits: dict[str, set[str]] = defaultdict(set)
    for _, _, _, step in _action_steps():
        action, _, ref = step["uses"].partition("@")
        commits[action].add(ref)
    assert {action: refs for action, refs in commits.items() if len(refs) > 1} == {}


def test_every_setup_uv_step_pins_the_same_exact_uv_release():
    versions: dict[str, str | None] = {}
    for name, job, index, step in _action_steps():
        if step["uses"].startswith("astral-sh/setup-uv@"):
            version = (step.get("with") or {}).get("version")
            versions[f"{name}:{job}[{index}]"] = version
    assert {
        where: version
        for where, version in versions.items()
        if not (isinstance(version, str) and re.fullmatch(r"\d+\.\d+\.\d+", version))
    } == {}
    assert len(set(versions.values())) == 1, versions


def test_every_uv_sync_run_export_and_lock_stays_on_the_lock():
    offenders = []
    for where, invocation in _uv_invocations():
        options = set(invocation.options)
        text = f"{where} {invocation.command.line}"
        if options & LOCK_OVERRIDES:
            offenders.append(text)
        elif invocation.subcommand in {"sync", "run", "export"}:
            if not options & {"--locked", "--frozen"}:
                offenders.append(text)
        elif invocation.subcommand == "lock":
            # `uv lock` without these may rewrite uv.lock from the network.
            if not options & {"--check", "--dry-run", "--offline"}:
                offenders.append(text)
    assert offenders == []


def test_no_workflow_runs_an_unpinned_uv_tool_or_edits_the_dependencies():
    offenders = [
        f"{where} {invocation.command.line}"
        for where, invocation in _uv_invocations()
        if invocation.subcommand in {"add", "remove", "uvx", "tool install", "tool run"}
    ]
    assert offenders == []


def test_every_package_install_stages_a_target_directory():
    # Only a staged runtime tree may be installed outside the lock; an install
    # into an environment (`uv venv && uv pip install -e .`) may not.
    offenders = []
    for where, command in _commands(_steps()):
        invocation = workflow_shell.uv_invocation(command, UV_OPTIONS)
        if invocation is not None and invocation.subcommand in {
            "pip install",
            "pip sync",
        }:
            staged = "--target" in invocation.options
        elif _pip_installs(command):
            staged = _stages_target(command.words)
        else:
            continue
        if not staged:
            offenders.append(f"{where} {command.line}")
    assert offenders == []


def test_no_workflow_sets_uv_environment_variables():
    # UV_UPGRADE, UV_OFFLINE, UV_FROZEN and the rest change how every later
    # uv call resolves, invisibly to the option checks above.
    uv_variable = re.compile(r"UV_[A-Z0-9_]+")
    # NAME=value, or GitHub's multiline NAME<<DELIMITER form.
    uv_assignment = re.compile(r"UV_[A-Z0-9_]+(?:\+?=|<<)")
    offenders = []
    for name, workflow in _workflows():
        jobs = list(workflow["jobs"].values())
        scopes = [
            workflow,
            *jobs,
            *(step for job in jobs for step in job.get("steps", [])),
            *(
                job["container"]
                for job in jobs
                if isinstance(job.get("container"), dict)
            ),
            *(
                service
                for job in jobs
                for service in (job.get("services") or {}).values()
                if isinstance(service, dict)
            ),
        ]
        offenders += [
            f"{name} env: {key}"
            for scope in scopes
            for key in (scope.get("env") or {})
            if uv_variable.fullmatch(key)
        ]
    for name, job, index, step in _steps():
        where = f"{name}:{job}[{index}]"
        step_commands = _script_commands(step.get("run", ""))
        for command in step_commands:
            # An assignment, inline, through env or sudo, or inherited by bash -c.
            if any(uv_assignment.match(word) for word in command.assignments):
                offenders.append(f"{where} {command.line}")
            # export/declare/typeset/readonly/local UV_X=... sets it for the shell.
            elif command.program in {
                "declare",
                "export",
                "local",
                "readonly",
                "typeset",
            }:
                if any(uv_assignment.match(word) for word in command.words[1:]):
                    offenders.append(f"{where} {command.line}")
        # A write to $GITHUB_ENV sets a variable for every later step. The
        # reader attaches a group's, a pipe-to-tee's or an `exec`'s redirection
        # to each command whose output it receives, so each command is judged
        # by where its own output goes.
        for command in step_commands:
            if not any("GITHUB_ENV" in target for target in command.redirects):
                continue
            if any(
                uv_variable.fullmatch(word) or uv_assignment.match(word)
                for word in command.words[1:]
            ) or any(
                re.search(r"(?m)^\s*(?:export\s+)?UV_[A-Z0-9_]+(?:\+?=|<<)", text)
                for text in command.stdin_text
            ):
                offenders.append(f"{where} {command.line}")
    assert offenders == []


def test_every_step_runs_under_bash():
    # The reader follows bash; a pwsh or python step would go unread.
    shells = [
        f"{name}:{where} {shell}"
        for name, workflow in _workflows()
        for where, shell in [
            (
                "defaults",
                ((workflow.get("defaults") or {}).get("run") or {}).get("shell"),
            ),
            *(
                (job_name, ((job.get("defaults") or {}).get("run") or {}).get("shell"))
                for job_name, job in workflow["jobs"].items()
            ),
            *(
                (f"{job_name}[{index}]", step.get("shell"))
                for job_name, job in workflow["jobs"].items()
                for index, step in enumerate(job.get("steps", []))
                if "run" in step
            ),
        ]
        if shell is not None and not re.fullmatch(r"bash(?:\s.*)?", shell)
    ]
    assert shells == []


def test_no_job_runs_on_windows():
    # Windows runs `run:` steps under pwsh by default, which the reader does
    # not follow.
    runners = [
        f"{name}:{job_name} {runner}"
        for name, workflow in _workflows()
        for job_name, job in workflow["jobs"].items()
        for runner in [
            job.get("runs-on"),
            *((job.get("strategy") or {}).get("matrix") or {}).get("os", []),
        ]
        if runner and "windows" in str(runner).lower()
    ]
    assert runners == []


def test_no_python_file_declares_inline_script_dependencies():
    # `uv run script.py` resolves a PEP 723 `# /// script` block outside
    # uv.lock.
    tracked = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert tracked
    inline = [
        path
        for path in tracked
        if re.search(
            r"(?m)^# /// script$",
            (ROOT / path).read_text(encoding="utf-8", errors="ignore"),
        )
    ]
    assert inline == []


def test_job_containers_are_pinned_by_digest():
    images = [
        f"{name}:{job_name} {image}"
        for name, workflow in _workflows()
        for job_name, job in workflow["jobs"].items()
        for image in [
            (
                job["container"]["image"]
                if isinstance(job.get("container"), dict)
                else job.get("container")
            ),
            *(
                service.get("image") if isinstance(service, dict) else service
                for service in (job.get("services") or {}).values()
            ),
        ]
        if image and not re.fullmatch(r"\S+@sha256:[0-9a-f]{64}", image)
    ]
    assert images == []


@pytest.mark.parametrize(
    ("workflow_name", "job_name"),
    sorted(
        (workflow_name, job_name)
        for workflow_name, jobs in TEST_ENVIRONMENTS.items()
        for job_name in jobs
    ),
)
def test_test_environment_is_the_locked_dev_set(workflow_name, job_name):
    commands = [
        command
        for _, command in _commands(_job_steps(workflow_name, job_name))
        if command.words
    ]
    invocations = [
        (position, invocation)
        for position, command in enumerate(commands)
        if (invocation := workflow_shell.uv_invocation(command, UV_OPTIONS))
    ]

    syncs = [
        (position, invocation.command.words)
        for position, invocation in invocations
        if invocation.subcommand == "sync"
    ]
    assert len(syncs) == 1, syncs
    sync_position, sync_words = syncs[0]
    assert sync_words in LOCKED_DEV_SYNCS, sync_words
    if "${{matrix.python-version}}" in sync_words:
        job = dict(_workflows())[workflow_name]["jobs"][job_name]
        assert job["strategy"]["matrix"]["python-version"] == ["3.13"]

    # Nothing else builds or changes a Python environment in these jobs,
    # including through the command a `uv run` launches.
    # `uv run` syncs the environment first, so it could add an extra.
    assert [
        invocation.command.line
        for _, invocation in invocations
        if invocation.subcommand.partition(" ")[0]
        in {"add", "pip", "remove", "run", "tool", "uvx", "venv"}
    ] == []
    assert [
        command.line
        for command in commands
        if _runs_pip(command) or command.program in {"pipx", "pip-sync"}
    ] == []

    # Every tool the job runs comes from the synced .venv, after the sync.
    venv_uses = [
        position
        for position, command in enumerate(commands)
        if VENV_TOOL.fullmatch(command.words[0])
    ]
    assert venv_uses
    assert all(position > sync_position for position in venv_uses), venv_uses
    assert [
        command.line
        for command in commands
        if TOOLS.fullmatch(command.program)
        and not VENV_TOOL.fullmatch(command.words[0])
    ] == []
