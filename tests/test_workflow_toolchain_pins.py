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

These static checks are defense in depth. What this PR promises, that the
suite runs on the lock, is checked directly at run time:
test_the_test_environment_is_exactly_the_locked_set compares every installed
distribution with ``uv export --locked --extra dev`` in CI, whatever shell
built the environment.

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
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sysconfig
from collections import defaultdict
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

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
# The exact run: of the step every test job dedicates to the runtime check.
RUNTIME_CHECK_RUN = (
    ".venv/bin/pytest -q -o addopts='' "
    "tests/test_workflow_toolchain_pins.py"
    "::test_the_test_environment_is_exactly_the_locked_set"
)
# UV_* names a run script may mention, for a reviewed read; none today.
UV_NAMES_ALLOWED: frozenset[str] = frozenset()
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
    "eval -- 'uv pip install -e .'": [("pip install", ("--editable",))],
    "builtin -- eval 'uv pip install -e .'": [("pip install", ("--editable",))],
    # xargs options whose argument is optional and attached only.
    "printf '%s\\n' . | xargs --replace uv sync --directory {}": [
        ("sync", ("--directory",))
    ],
    "xargs -i uv sync --directory {}": [("sync", ("--directory",))],
    "xargs --max-lines=1 uv sync": [("sync", ())],
    # env and sudo read any NAME=VALUE word as an assignment, also after --.
    'env "FOO-BAR=1" uv sync': [("sync", ())],
    "env -i -- A=1 B-C=2 uv sync": [("sync", ())],
    "sudo FOO-BAR=1 uv sync": [("sync", ())],
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
        "python${{ matrix.python-version }} -m pip install requests",
        "${{ matrix.tool_dir }}/env uv sync --locked",
        '$(printf /bin)/bash -c "uv sync --locked"',
        "/usr/bin/${{ matrix.python }} -m pip install requests",
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
        # env -S's own escapes, and mapfile callbacks.
        "env -S 'bash -c \"uv\\_pip install -e .\"'",
        "env -S 'uv ${ARGS}'",
        "mapfile -t -C 'uv sync' -c 1 lines < f",
        "env -S 'uv sync # --frozen'",
        'env "${p}_X=1" uv sync',
        'sudo "${p}_X=1" uv sync',
        'env -- "${p}_X=1" uv sync',
        'sudo -E "${p}-X=1" uv sync',
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
        # Every way of setting a UV_* variable has to spell its name.
        ("UV_UPGRADE=1 uv sync --frozen", True),
        ("export UV_UPGRADE=1", True),
        ("UV_UPGRADE=1 bash -c 'uv sync --frozen'", True),
        ('{ echo "UV_UPGRADE=1"; } >> "$GITHUB_ENV"', True),
        ('{ echo "UV_NO_SYNC<<EOF"; echo 1; echo EOF; } >> "$GITHUB_ENV"', True),
        ('echo "UV_UPGRADE=1" | tee -a "$GITHUB_ENV"', True),
        ("tee -a \"$GITHUB_ENV\" <<'EOF'\nUV_NO_SYNC=1\nEOF", True),
        ('tee -a "$GITHUB_ENV" <<< "UV_X=1"', True),
        ('echo UV_X=1 | cat | tee -a "$GITHUB_ENV"', True),
        ('echo UV_NO_SYNC=1 | tee /dev/stderr | tee "$GITHUB_ENV" >/dev/null', True),
        ('printf "VENV=.venv\\nUV_NO_SYNC=1\\n" >> "$GITHUB_ENV"', True),
        ('echo "VENV=.venv\nUV_NO_SYNC=1" >> "$GITHUB_ENV"', True),
        ("uv run --frozen bash -c 'echo UV_NO_SYNC=1' >> \"$GITHUB_ENV\"", True),
        ('exec >> "$GITHUB_ENV"\necho UV_UPGRADE=1', True),
        # By design, even a read or a log line fails, loudly; the fix is to
        # drop it or add the name to UV_NAMES_ALLOWED with a reviewer.
        ('echo "UV_CACHE_DIR=$UV_CACHE_DIR"', True),
        ('grep -q "UV_NO_SYNC=1" < "$GITHUB_ENV"', True),
        # Other names that merely contain UV_ are not UV_* variables...
        ('echo "LOCKED_UV_VERSION=0.11.7"', False),
        # ...but any unreviewed step writing $GITHUB_ENV fails.
        ('echo "LOCKED_UV_VERSION=0.11.7" >> "$GITHUB_ENV"', True),
        ('echo "VENV=.venv" >> "$GITHUB_ENV"', True),
        # Round 12: $GITHUB_ENV under its context name, and $GITHUB_PATH.
        ("printf 'A=1\\n' >> \"${{ github.env }}\"", True),
        ('echo "$HOME/bin" >> "$GITHUB_PATH"', True),
        ("echo {PIP,POETRY}_INDEX_URL", False),
        # Round 10: computed names, quote splitting, continuations, printf
        # precision, and escaped letters.
        (
            'for prefix in UV PIP; do\n  export "${prefix}_INDEX_URL=x"\ndone',
            True,
        ),
        ('for prefix in PIP POETRY; do export "${prefix}_INDEX_URL=x"; done', True),
        ('export U"V_X"=1', True),
        ("echo UV'_'X=1 >> \"$GITHUB_ENV\"", True),
        ('echo "U\\\nV_X=1" >> "$GITHUB_ENV"', True),
        ("printf '%.0sUV_X=1\\n' x >> \"$GITHUB_ENV\"", True),
        ("echo $'\\x55V_X=1' >> \"$GITHUB_ENV\"", True),
        ("export PIP_INDEX_URL=x", False),
        # Round 11: names built by case conversion, and unreviewed
        # $GITHUB_ENV writers.
        (
            'for tool in pip uv; do\n  echo "${tool^^}_INDEX_URL=x" >> "$GITHUB_ENV"\ndone',
            True,
        ),
        ('echo "${tool@U}_X=1"', True),
        ("declare -u n=uv; export n", True),
        (
            "for tool in uv pip; do\n  prefix=$(printf '%s' \"$tool\" | tr '[:lower:]' '[:upper:]')\n  printf '%s_X=1\\n' \"$prefix\"\ndone",
            True,
        ),
        ('echo "PIP_INDEX_URL=x" >> "$GITHUB_ENV"', True),
        ("tr a-z A-Z < names", True),
        # Sanitizing with tr converts no case.
        ("slug=$(printf '%s' \"$REF\" | tr -cs 'a-z0-9' '-')", False),
        ("tr -d 'A-Z' < names", False),
        ("printf 'a\\tUV_X=1'", True),
        # Brace expansions and escapes spell a name too.
        ("export {UV,PIP}_INDEX_URL=https://example.com/simple", True),
        ('printf "%s\\n" UV_{NO_SYNC,FROZEN}=true >> "$GITHUB_ENV"', True),
        ('export "UV_${NAME}=1"', True),
        ("printf 'A=1\\012UV_X=1\\n' >> \"$GITHUB_ENV\"", True),
        ("printf 'A=1\\x0aUV_X=1\\n' >> \"$GITHUB_ENV\"", True),
        ("echo -e 'A=1\\0012UV_NO_SYNC=1' >> \"$GITHUB_ENV\"", True),
        ("printf $'A=1\\u0aUV_X=1\\n' >> \"$GITHUB_ENV\"", True),
        ("printf $'A=1\\U0000000aUV_X=1' >> \"$GITHUB_ENV\"", True),
        ("printf $'A=1\\cJUV_X=1' >> \"$GITHUB_ENV\"", True),
        ("printf '%sUV_X=1\\n' '' >> \"$GITHUB_ENV\"", True),
    ],
)
def test_run_scripts_never_name_a_uv_variable(monkeypatch, script, flagged):
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


def test_uv_names_in_any_workflow_string_are_caught(monkeypatch):
    workflows = copy.deepcopy(_workflows())
    step = {"env": {"EXTRA": "UV_NO_SYNC=1"}, "run": 'echo "$EXTRA" >> "$GITHUB_ENV"'}
    dict(workflows)["ci.yml"]["jobs"]["lint"]["steps"].append(step)
    monkeypatch.setattr(
        "tests.test_workflow_toolchain_pins._workflows", lambda: workflows
    )
    with pytest.raises(AssertionError):
        test_no_workflow_sets_uv_environment_variables()


def test_setup_uv_inputs_other_than_version_are_caught(monkeypatch):
    workflows = copy.deepcopy(_workflows())
    for step in dict(workflows)["ci.yml"]["jobs"]["lint"]["steps"]:
        if step.get("uses", "").startswith("astral-sh/setup-uv@"):
            step["with"]["python-version"] = "3.12"
    monkeypatch.setattr(
        "tests.test_workflow_toolchain_pins._workflows", lambda: workflows
    )
    with pytest.raises(AssertionError):
        test_setup_uv_takes_only_a_version()


def test_a_uv_version_step_is_allowed_in_a_test_job(monkeypatch):
    workflows = copy.deepcopy(_workflows())
    dict(workflows)["ci.yml"]["jobs"]["test"]["steps"].append({"run": "uv --version"})
    monkeypatch.setattr(
        "tests.test_workflow_toolchain_pins._workflows", lambda: workflows
    )
    test_test_environment_is_the_locked_dev_set("ci.yml", "test")


def test_python_module_follows_python_option_syntax():
    module = workflow_shell.python_module
    assert module(("python", "-m", "pip", "install")) == "pip"
    assert module(("python", "-Impip", "install")) == "pip"
    assert module(("python", "-X", "dev", "-m", "pip")) == "pip"
    assert module(("python", "-c", 'print("python -m pip install")')) is None
    assert module(("python", "script.py", "-m", "pip")) is None


def test_uv_option_table_matches_the_pinned_uv():
    # CI installs the pinned uv (setup-uv), so there the comparison must run;
    # locally it runs only when the installed uv is the pinned release.
    installed = None
    if shutil.which("uv") is not None:
        installed = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, check=True
        ).stdout.split()[1]
    if installed != UV_OPTIONS["uv_version"]:
        message = f"uv {installed} is installed; the table is for the pinned uv"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
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


# A UV_* name, however it is spelled:
# - a `UV_` prefix at a name boundary, so `UV_${X}` and `UV_{A,B}` count but
#   LOCKED_UV_VERSION does not;
# - a `UV_` after a backslash or `%` and up to twelve escape or directive
#   characters, which covers every escape bash, echo -e or printf reads
#   (`\\n`, `\\012`, `\\0012`, `\\x0a`, `\\u0a`, `\\U0000000a`, `\\cJ`) and printf
#   directives such as `%s` or `%.0s`;
# - `UV` standing alone as a word, the building block of a computed name
#   (`for p in UV PIP; do export "${p}_INDEX_URL=..."`) or a brace
#   alternative (`{UV,PIP}_INDEX_URL`).
UV_NAME = re.compile(
    r"(?<![A-Za-z0-9_])UV_[A-Z0-9_]*"
    r"|(?<=[\\%])[A-Za-z0-9.*#+-]{0,12}UV_[A-Z0-9_]*"
    r"|(?<![A-Za-z0-9_])UV(?![A-Za-z0-9_])"
)
ENV_BUILTINS = frozenset({"declare", "export", "local", "readonly", "typeset"})
# Ways to build an upper-case name from lower-case text (`uv` -> `UV`), which
# the name rule cannot see through, so they fail closed.
CASE_CONVERSION = re.compile(
    r"\$\{[#!]?[A-Za-z_][A-Za-z0-9_]*(?:\[[^]]*\])?(?:\^|,|@[uUL])"
    r"|\b(?:toupper|tolower)\b|\\[UL]"
)
LOWER_SET = re.compile(r"\[:lower:\]|a-z")
UPPER_SET = re.compile(r"\[:upper:\]|A-Z")


def _tr_converts_case(words: tuple[str, ...]) -> bool:
    """Whether `tr SET1 SET2` translates between lower and upper case.

    Deleting or squeezing (`tr -cs 'a-z0-9' '-'`) converts nothing.
    """
    if not words or words[0].rsplit("/", 1)[-1] != "tr":
        return False
    sets = [word for word in words[1:] if not word.startswith("-")]
    return len(sets) >= 2 and any(
        (LOWER_SET.search(a) and UPPER_SET.search(b))
        or (UPPER_SET.search(a) and LOWER_SET.search(b))
        for a, b in [(sets[0], sets[1])]
    )


# Steps that may write $GITHUB_ENV, which sets variables for every later step,
# and the only names each writes. A new writer needs a reviewed entry here.
GITHUB_ENV_WRITERS = {
    ("targeted-signed-reencode.yml", "encode", "Validate dependent cascade"): {
        "DEPENDENT_CASCADE_MODE"
    },
}


def _strings(node):
    """Every key and string value in a parsed workflow."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _strings(key)
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)
    elif isinstance(node, str):
        yield node


def test_no_workflow_sets_uv_environment_variables():
    """No workflow text names a UV_* variable.

    UV_UPGRADE, UV_OFFLINE, UV_FROZEN, UV_INDEX_URL and the rest change how
    uv resolves, invisibly to the option checks above. A workflow can set
    them inline, with export, in an env: block, or through $GITHUB_ENV (by
    redirect, group, tee, heredoc or multiline string). Rather than trace
    where each script's output goes, the rule is textual and conservative:
    no key or string anywhere in a workflow may contain a UV_* name,
    including one spelled with a brace expansion or after an escape, nor the
    word UV on its own; each command's words and assignments, after quote
    removal, are checked the same way; and export/declare may not compute a
    variable's name.

    This is a lint with enumerated coverage, not a proof that no UV_*
    variable can be set; what it misses in the test jobs, the runtime
    locked-set check catches, and every test job runs that check. On top of the name rule: case conversion
    (`${x^^}`, `declare -u`, `tr a-z A-Z`, `toupper`, `\\U`) fails closed,
    computed names in export/declare/env/sudo fail closed, and only the
    reviewed steps in GITHUB_ENV_WRITERS may write $GITHUB_ENV, whatever
    names they build. A harmless mention (a log line, a read) fails loudly
    instead; add a name to UV_NAMES_ALLOWED only for a reviewed read.
    Workflows configure uv through flags. setup-uv itself exports
    UV_PYTHON_INSTALL_DIR and UV_CACHE_DIR, which set where uv installs
    Python and caches, not what it resolves; the test below keeps its other
    inputs, such as python-version (UV_PYTHON), out.
    """
    offenders = [
        f"{name}: {found}"
        for name, workflow in _workflows()
        for text in _strings(workflow)
        for found in sorted(
            {match.group() for match in UV_NAME.finditer(text)} - UV_NAMES_ALLOWED
        )
    ]
    for name, job, index, step in _steps():
        for command in _script_commands(step.get("run", "")):
            # The same rule on each command's words and assignments after
            # quote removal and $'...' decoding, so `U"V_X"=1` and
            # `$'\\x55V_X'` count too.
            offenders += [
                f"{name}:{job}[{index}] {found}"
                for text in (*command.words, *command.assignments)
                for found in sorted(
                    {m.group() for m in UV_NAME.finditer(text)} - UV_NAMES_ALLOWED
                )
            ]
            # Case conversion builds an upper-case name the rule cannot see.
            if _tr_converts_case(command.words) or (
                command.program in {"declare", "local", "typeset"}
                and any(
                    re.fullmatch(r"[-+][A-Za-z]*[luc][A-Za-z]*", word)
                    for word in command.words[1:]
                )
            ):
                offenders.append(f"{name}:{job}[{index}] converts case: {command.line}")
            # A computed name (`export "${prefix}_X=1"`) cannot be checked, so
            # it fails closed.
            if command.program in ENV_BUILTINS:
                offenders += [
                    f"{name}:{job}[{index}] computes a variable name: {word}"
                    for word in command.words[1:]
                    if not word.startswith("-")
                    and re.search(r"[$`\ue000]", word.partition("=")[0])
                ]
        if CASE_CONVERSION.search(step.get("run", "")):
            offenders.append(f"{name}:{job}[{index}] converts case")
        # $GITHUB_ENV is also `${{ github.env }}`, and a step can pass either
        # through env:, so every string in the step is searched.
        step_text = "\n".join(_strings(step))
        if (
            re.search(r"GITHUB_ENV|github\.env\b", step_text)
            and (
                name,
                job,
                step.get("name"),
            )
            not in GITHUB_ENV_WRITERS
        ):
            offenders.append(f"{name}:{job}[{index}] writes $GITHUB_ENV unreviewed")
        # A $GITHUB_PATH write could put another `uv` or `python` first on PATH.
        if re.search(r"GITHUB_PATH|github\.path\b", step_text):
            offenders.append(f"{name}:{job}[{index}] writes $GITHUB_PATH")
    assert offenders == []


def test_reviewed_github_env_writers_set_only_their_names():
    for (name, job, step_name), allowed in GITHUB_ENV_WRITERS.items():
        workflow = dict(_workflows())[name]
        [step] = [
            s for s in workflow["jobs"][job]["steps"] if s.get("name") == step_name
        ]
        written = {
            match.group(1)
            for match in re.finditer(
                r"^\s*(?:printf|echo)\s+['\"]?([A-Z][A-Z0-9_]*)=.*GITHUB_ENV",
                step["run"],
                re.M,
            )
        }
        assert written == allowed, (name, job, step_name, written)
        assert step["run"].count("GITHUB_ENV") == len(allowed)


def _site_distributions() -> dict[str, str]:
    """Distributions in this interpreter's own site-packages, by version.

    A distribution installed from git is given as `git:<commit>`, from its
    PEP 610 direct_url.json.
    """
    paths = sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})
    found = {}
    for dist in importlib.metadata.distributions(path=paths):
        version = dist.version
        direct_url = json.loads(dist.read_text("direct_url.json") or "{}")
        if "vcs_info" in direct_url:
            version = "git:" + direct_url["vcs_info"]["commit_id"]
        found[canonicalize_name(dist.metadata["Name"])] = version
    return found


def test_the_test_environment_is_exactly_the_locked_set():
    """What the suite runs on is what uv.lock pins for the dev extra.

    This is the runtime guarantee behind the static checks: however the
    environment was built, it must hold exactly the locked distributions at
    their locked versions (a git dependency by name). CI builds it with
    `uv sync --locked --extra dev`; locally, set AXIOM_CHECK_LOCKED_ENV=1 in
    an environment synced the same way.
    """
    if not (
        os.environ.get("CI")
        or os.environ.get("GITHUB_ACTIONS") == "true"
        or os.environ.get("AXIOM_CHECK_LOCKED_ENV")
    ):
        pytest.skip("runs in CI, or with AXIOM_CHECK_LOCKED_ENV=1")
    exported = subprocess.run(
        ["uv", "export", "--locked", "--extra", "dev", "--no-hashes"]
        + ["--no-header", "--no-annotate", "--no-emit-project"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    locked: dict[str, str] = {}
    for line in exported.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        specifier = str(requirement.specifier)
        # A registry package by version; a git dependency by its commit.
        locked[canonicalize_name(requirement.name)] = (
            specifier[2:]
            if specifier.startswith("==")
            else "git:" + requirement.url.rpartition("@")[2]
        )
    installed = _site_distributions()
    installed.pop("axiom-encode", None)
    assert len(locked) >= 50
    assert sorted(installed) == sorted(locked)
    assert {
        name: (version, locked[name])
        for name, version in installed.items()
        if version != locked[name]
    } == {}


def test_setup_uv_takes_only_a_version():
    # Its other inputs export UV_* variables (python-version sets UV_PYTHON,
    # tool-dir sets UV_TOOL_DIR, ...) that later uv calls would honour.
    inputs = [
        f"{name}:{job}[{index}] {sorted(step.get('with') or {})}"
        for name, job, index, step in _action_steps()
        if step["uses"].startswith("astral-sh/setup-uv@")
        and set(step.get("with") or {}) != {"version"}
    ]
    assert inputs == []


def test_setup_python_takes_only_a_python_version():
    # Its pip-install input installs packages with pip, outside the lock and
    # outside every run-script check.
    inputs = [
        f"{name}:{job}[{index}] {sorted(step.get('with') or {})}"
        for name, job, index, step in _action_steps()
        if step["uses"].startswith("actions/setup-python@")
        and not set(step.get("with") or {}) <= {"python-version"}
    ]
    assert inputs == []


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


def _runner_labels(job: dict) -> list[str]:
    """The runner labels a job can run on, with `matrix.*` references resolved.

    Anything that cannot be resolved statically comes back as written, so the
    caller rejects it: an expression-valued matrix or `include`, an include
    entry that is not a mapping, or a value that is not a plain label.
    """
    runs_on = job.get("runs-on")
    labels = runs_on if isinstance(runs_on, list) else [runs_on]
    strategy = job.get("strategy") or {}
    matrix = (strategy.get("matrix") if isinstance(strategy, dict) else strategy) or {}
    resolved: list[str] = []
    for label in labels:
        if not isinstance(label, str):
            resolved.append(repr(label))  # a group or label mapping
            continue
        reference = re.fullmatch(r"\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}", label)
        if reference is None:
            resolved.append(label)
            continue
        if not isinstance(matrix, dict):
            resolved.append(f"{label} from {matrix!r}")
            continue
        key = reference.group(1)
        values = matrix.get(key, [])
        # A copy: the parsed workflows are cached and shared between tests.
        values = list(values) if isinstance(values, list) else [values]
        include = matrix.get("include", [])
        if not isinstance(include, list):
            resolved.append(f"{label} with include {include!r}")
            include = []
        for entry in include:
            if not isinstance(entry, dict):
                resolved.append(f"{label} with include entry {entry!r}")
            elif key in entry:
                values.append(entry[key])
        resolved += [
            value if isinstance(value, str) else repr(value) for value in values
        ] or [label]
    return resolved


def test_every_job_runs_on_a_hosted_linux_or_macos_runner():
    # Windows runs `run:` steps under pwsh by default, which the reader does
    # not follow. An expression it cannot resolve fails too.
    runners = [
        f"{name}:{job_name} {label}"
        for name, workflow in _workflows()
        for job_name, job in workflow["jobs"].items()
        if "uses" not in job
        for label in _runner_labels(job)
        if not re.fullmatch(r"(?:ubuntu|macos)-[0-9a-z.-]+", label)
    ]
    assert runners == []


@pytest.mark.parametrize(
    ("runs_on", "matrix", "allowed"),
    [
        ("ubuntu-latest", {}, True),
        ("${{ matrix.os }}", {"os": ["ubuntu-latest", "macos-latest"]}, True),
        (
            "${{ matrix.runner }}",
            {"runner": ["ubuntu-latest", "windows-latest"]},
            False,
        ),
        (
            "${{ matrix.os }}",
            {"os": ["ubuntu-latest"], "include": [{"os": "windows-2022"}]},
            False,
        ),
        ("${{ inputs.runner }}", {}, False),
        (["self-hosted", "linux"], {}, False),
        ("${{ matrix.os }}", "${{ fromJSON(needs.x.outputs.matrix) }}", False),
        (
            "${{ matrix.os }}",
            {"os": ["ubuntu-latest"], "include": "${{ fromJSON(vars.EXTRA) }}"},
            False,
        ),
        ("${{ matrix.os }}", {"os": ["${{ inputs.os }}"]}, False),
    ],
)
def test_runner_labels_resolve_matrix_references(monkeypatch, runs_on, matrix, allowed):
    workflows = copy.deepcopy(_workflows())
    job = dict(workflows)["ci.yml"]["jobs"]["lint"]
    job["runs-on"] = runs_on
    job["strategy"] = {"matrix": matrix}
    monkeypatch.setattr(
        "tests.test_workflow_toolchain_pins._workflows", lambda: workflows
    )
    if allowed:
        test_every_job_runs_on_a_hosted_linux_or_macos_runner()
    else:
        with pytest.raises(AssertionError):
            test_every_job_runs_on_a_hosted_linux_or_macos_runner()


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
        strategy = job.get("strategy")
        matrix = strategy.get("matrix") if isinstance(strategy, dict) else strategy
        assert isinstance(matrix, dict), matrix
        assert matrix.get("python-version") == ["3.13"], matrix

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

    # The runtime locked-set check runs in a dedicated step after the sync,
    # pinned exactly: a single command (so no pipe, `|| true`, wrapper,
    # export or unset around it), no option that could deselect it, and no
    # `if:`, `env:`, `shell:` or `continue-on-error:` on the step. The job has
    # no `if:` or `continue-on-error:`, and nothing in scope overrides CI,
    # GITHUB_ACTIONS or pytest's options. Runner-provided CI can then only be
    # changed through $GITHUB_ENV, which only reviewed steps may write.
    workflow = dict(_workflows())[workflow_name]
    job = workflow["jobs"][job_name]
    assert "if" not in job and "continue-on-error" not in job, job_name
    overrides = {"CI", "GITHUB_ACTIONS", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"}
    for scope in (workflow, job):
        env = scope.get("env") or {}
        assert isinstance(env, dict) and not set(env) & overrides, env
        shell = ((scope.get("defaults") or {}).get("run") or {}).get("shell")
        assert shell in {None, "bash"}, shell
    steps = job["steps"]
    sync_step = next(
        index
        for index, step in enumerate(steps)
        for command in _script_commands(step.get("run", ""))
        if command.words == sync_words
    )
    dedicated = [
        index
        for index, step in enumerate(steps)
        if set(step) <= {"name", "run"}
        and " ".join(str(step.get("run", "")).split()) == RUNTIME_CHECK_RUN
    ]
    assert any(index > sync_step for index in dedicated), (
        f"{workflow_name}:{job_name} has no dedicated runtime-check step"
    )

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
