"""Keep workflow helpers outside the raw apply-key process environment."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ".github/workflows/targeted-signed-reencode.yml"
KEY_ENV = "AXIOM_ENCODE_APPLY_SIGNING_KEY"
SIGNER = "/opt/axiom-verification/axiom-encode-apply-signer"
SECRET_REFERENCE = re.compile(
    r"secrets\s*(?:\.\s*AXIOM_ENCODE_APPLY_SIGNING_KEY\b"
    r"|\[\s*['\"]AXIOM_ENCODE_APPLY_SIGNING_KEY['\"]\s*\])"
)
# Keep this deliberately narrower than the full Bash builtin/coreutils sets:
# eval, source, command, env, and arbitrary shell launchers can execute helpers.
ALLOWED_BUILTINS = frozenset({":", "set", "test", "mapfile"})
# Current launchers need no external utility to read their prepared arguments.
ALLOWED_COREUTILS: frozenset[str] = frozenset()


def _workflow() -> dict:
    return yaml.safe_load((ROOT / WORKFLOW).read_text(encoding="utf-8"))


def _secret_references(value: object, path: tuple = ()) -> Iterator[tuple]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _secret_references(child, (*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _secret_references(child, (*path, index))
    elif isinstance(value, str) and SECRET_REFERENCE.search(value):
        yield path


def _key_steps(workflow: dict) -> Iterator[tuple[str, int, dict]]:
    for job_id, job in workflow["jobs"].items():
        for index, step in enumerate(job.get("steps", [])):
            if any(_secret_references(step.get("env", {}))):
                yield job_id, index, step


def _simple_shell_commands(script: str) -> list[list[str]]:
    """Parse a deliberately small, non-evaluating shell grammar.

    These steps need simple commands and file redirections only. Reject shell
    constructs that could hide another executable rather than approximating
    the full Bash grammar or looking for a few forbidden command names.
    """
    assert "$(" not in script, "command/arithmetic substitution is forbidden"
    assert "`" not in script, "backtick command substitution is forbidden"
    assert "<(" not in script and ">(" not in script, (
        "process substitution is forbidden"
    )
    assert "$[" not in script, "legacy arithmetic expansion is forbidden"
    for expression in re.findall(r"\$\{([^}]*)\}", script):
        assert re.fullmatch(r"#?[A-Za-z_][A-Za-z0-9_]*(?:\[@\])?", expression), (
            f"indirect, transforming, or indexed expansion is forbidden: {expression!r}"
        )
    assert not re.search(r"\$(?:\{)?AXIOM_ENCODE_APPLY_SIGNING_KEY\b", script), (
        "the shell must not expand the raw key"
    )
    # This exact NUL-reader loop uses builtins available on Bash 3 as well as
    # Actions' Bash 5. Normalize only this fixed grammar, never arbitrary loops.
    script = re.sub(
        r"args=\(\)\n"
        r"while IFS= read -r -d '' argument; do\n"
        r'  args\+=\("\$argument"\)\n'
        r'done < ("[^"\n]+")\n',
        r"mapfile -d '' -t args < \1\n",
        script,
    )
    lexer = shlex.shlex(
        script.replace("\\\n", ""), posix=True, punctuation_chars=";&|()<>\n"
    )
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    commands: list[list[str]] = []
    command: list[str] = []
    for token in lexer:
        if token and not token.strip("\n"):
            if command:
                commands.append(command)
                command = []
            continue
        assert not any(character in token for character in ";&|()"), (
            f"compound shell syntax is forbidden: {token!r}"
        )
        command.append(token)
    if command:
        commands.append(command)
    assert commands, "key-bearing step must have an explicit signer invocation"
    return commands


def _assert_minimal_key_script(script: str) -> None:
    signer_count = 0
    for command in _simple_shell_commands(script):
        executable = command[0]
        if executable == "exec":
            assert len(command) > 1 and command[1] == SIGNER, (
                f"exec may invoke only the owned signer: {command!r}"
            )
            command = command[1:]
            executable = command[0]
        if executable == SIGNER:
            signer_count += 1
            assert len(command) > 1 and command[1] == "run", (
                "use the key-clearing signer launcher"
            )
            assert "--key-env" in command
            assert command[command.index("--key-env") + 1] == KEY_ENV
            assert "--" in command, "the supervised CLI must follow --"
            continue
        assert executable in ALLOWED_BUILTINS | ALLOWED_COREUTILS, (
            f"key-bearing step invokes an unapproved command: {executable!r}"
        )
        if executable == "mapfile":
            # Bash's -C option executes a shell callback. Only the fixed NUL
            # argv-reader form used by these steps belongs in this boundary.
            assert len(command) == 7 and command[1:4] == ["-d", "", "-t"]
            assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", command[4])
            assert command[5] == "<", "mapfile must read a prepared argv file"
        elif executable == "set":
            assert command == ["set", "-euo", "pipefail"]
    assert signer_count == 1, "each key-bearing step must launch exactly one signer"


def assert_apply_key_step_scripts_are_minimal(workflow: dict) -> None:
    """Reusable assertion for checking the unchanged-base workflow as well."""
    steps = list(_key_steps(workflow))
    assert steps, "no apply-key steps found"
    for job_id, _, step in steps:
        try:
            _assert_minimal_key_script(step["run"])
            assert step.get("shell") == (
                "/usr/bin/bash --noprofile --norc -p -e -o pipefail {0}"
            ), "key launcher must ignore inherited shell startup code"
            assert all(
                step["env"].get(name) == ""
                for name in ("LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH")
            ), "key launcher must disable inherited dynamic loader hooks"
        except (AssertionError, ValueError) as error:
            raise AssertionError(f"{job_id}/{step.get('name')}: {error}") from error


def test_apply_key_steps_run_only_signer_builtins_and_allowlisted_coreutils() -> None:
    assert_apply_key_step_scripts_are_minimal(_workflow())


def test_apply_secret_is_referenced_only_by_targeted_encode_signer_step_env() -> None:
    references = []
    for path in sorted((ROOT / ".github/workflows").glob("*.y*ml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        for location in _secret_references(document):
            references.append((path.relative_to(ROOT).as_posix(), location))
    assert references, "no apply signing authority found"
    for workflow_path, location in references:
        assert workflow_path == WORKFLOW, (workflow_path, location)
        assert len(location) == 6, location
        assert location[:3] == ("jobs", "encode", "steps"), location
        assert isinstance(location[3], int), location
        assert location[4:] == ("env", KEY_ENV), location


def test_helper_preparation_steps_precede_each_key_bearing_encode_step() -> None:
    workflow = _workflow()
    assert not list(_secret_references(workflow.get("env", {})))
    job = workflow["jobs"]["encode"]
    assert not list(_secret_references(job.get("env", {})))
    steps = job["steps"]
    by_id = {step.get("id"): (index, step) for index, step in enumerate(steps)}
    helper_indices = [
        index
        for index, step in enumerate(steps)
        if re.fullmatch(r"prepare_(?:encode_apply|apply_\d{2})", step.get("id", ""))
    ]
    assert helper_indices, "helper orchestration must run in separate earlier steps"
    initialization_index, initialization = by_id["prepare_encode_apply"]
    assert "parse-source-bundle" in initialization["run"]
    assert "extract_repair_candidate.py" in initialization["run"]
    for index in helper_indices:
        helper = steps[index]
        assert "secrets." not in str(helper.get("env", {}))
        assert not list(_secret_references(helper))
        assert KEY_ENV not in helper.get("env", {})
    for job_id, index, step in _key_steps(workflow):
        assert job_id == "encode"
        if step.get("id") == "retired_source_metadata_migration":
            # This lane already receives its validated plan from an earlier
            # no-key preparation step and needs no encode orchestration.
            assert any(
                "retired-source-metadata-plan.json" in earlier.get("run", "")
                and not list(_secret_references(earlier))
                for earlier in steps[:index]
            )
            continue
        preparations = re.findall(
            r"steps\.(prepare_(?:encode_apply|apply_\d{2}))\.outputs\.ready",
            step.get("if", ""),
        )
        assert len(preparations) == 1, step.get("name")
        preparation_index, _ = by_id[preparations[0]]
        assert preparation_index == index - 1, step.get("name")
        assert initialization_index <= preparation_index < index
        assert "scripts/" not in step["run"]


_MINIMAL_SIGNER_SCRIPT = f"""set -euo pipefail
mapfile -d '' -t args < "$RUNNER_TEMP/apply-args"
test "${{#args[@]}}" -gt 0
exec {SIGNER} run --key-env {KEY_ENV} -- "${{args[@]}}"
"""


def test_minimal_script_parser_accepts_only_the_required_simple_grammar() -> None:
    _assert_minimal_key_script(_MINIMAL_SIGNER_SCRIPT)


def test_key_launcher_ignores_inherited_bash_startup(tmp_path: Path) -> None:
    bash = shutil.which("bash")
    assert bash is not None
    marker = tmp_path / "startup-ran"
    startup = tmp_path / "startup.bash"
    startup.write_text('printf unsafe > "$STARTUP_MARKER"\n')
    signer = next(
        step
        for _, _, step in _key_steps(_workflow())
        if step.get("id") == "apply_signing_01"
    )
    options = shlex.split(signer["shell"])[1:-1]
    completed = subprocess.run(
        [bash, *options, "-c", 'test "$AXIOM_ENCODE_APPLY_SIGNING_KEY" = dummy'],
        env={
            "BASH_ENV": str(startup),
            "STARTUP_MARKER": str(marker),
            KEY_ENV: "dummy",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout == ""
    assert not marker.exists()


@pytest.mark.parametrize(
    "injected",
    [
        "python scripts/prepare_signed_backfill.py parse-source-bundle []",
        "python3 -c 'import axiom_encode'",
        "uv run python scripts/extract_repair_candidate.py",
        "axiom-encode encode dummy/citation --apply",
        "/opt/axiom-verification/axiom-encode encode dummy/citation --apply",
        "scripts/extract_repair_candidate.py",
        "source helper.sh",
        ". helper.sh",
        "eval 'python helper.py'",
        "command python helper.py",
        "mapfile -C 'python helper.py' args < prepared.txt",
        "env python helper.py",
        "bash -c 'python helper.py'",
        '"$helper" --parse',
        "exec python helper.py",
        "cat $(python helper.py)",
        'cat "$(python helper.py)"',
        "cat `python helper.py`",
        "cat <(python helper.py)",
        "cat > >(python helper.py)",
        "cat x | python helper.py",
        "cat x; python helper.py",
        "cat x && python helper.py",
        "(python helper.py)",
        "if test -f x; then python helper.py; fi",
        'cat "$AXIOM_ENCODE_APPLY_SIGNING_KEY"',
        'cat "${AXIOM_ENCODE_APPLY_SIGNING_KEY}"',
        ': "${!variable}"',
        ': "${variable@P}"',
        ': "$[variable]"',
    ],
)
def test_minimal_script_parser_rejects_hidden_or_unapproved_execution(
    injected: str,
) -> None:
    with pytest.raises(AssertionError):
        _assert_minimal_key_script(injected + "\n" + _MINIMAL_SIGNER_SCRIPT)


@pytest.mark.parametrize(
    "reference",
    [
        "${{ secrets.AXIOM_ENCODE_APPLY_SIGNING_KEY }}",
        "${{ secrets . AXIOM_ENCODE_APPLY_SIGNING_KEY }}",
        "${{ secrets['AXIOM_ENCODE_APPLY_SIGNING_KEY'] }}",
        '${{ secrets["AXIOM_ENCODE_APPLY_SIGNING_KEY"] }}',
    ],
)
def test_secret_inventory_recognizes_equivalent_expression_spellings(
    reference: str,
) -> None:
    assert list(_secret_references({"env": {KEY_ENV: reference}})) == [("env", KEY_ENV)]
