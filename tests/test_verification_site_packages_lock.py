"""The protected verification supervisor's site-packages must be the uv.lock set.

``scripts/provision_verification_supervisor.py`` copies a caller-staged
``--site-packages`` tree into the signer's trusted Python import root. A
resolving ``uv pip install --target ... ".[api]"`` ignores ``uv.lock`` and
floats every ``>=`` dependency to the index's latest at run time: targeted run
36174590343 logged ``claude-agent-sdk==0.1.47`` in the locked ``.venv`` and
``claude-agent-sdk==0.2.159`` in the verification tree of the same job. These
tests fail if any workflow stages site-packages any other way than from the
lock, and execute every staging step against a fake ``uv`` to prove its
fail-closed paths.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

# The provisioner's own selftest stages a throwaway tree (pyyaml plus hostile
# startup-hook fixtures) under a stand-in supervisor with test-only trust roots;
# nothing it provisions verifies or signs anything.
EXEMPT_WORKFLOWS = {"provision-selftest.yml"}

SITE_PACKAGES = "$RUNNER_TEMP/axiom-verification-site-packages"
ORACLES_PATTERN = (
    r"axiom-oracles @ git\+https://github\.com/TheAxiomFoundation/"
    r"axiom-oracles@[0-9a-f]{40}"
)
LOCKED_GIT_LINE = (
    "axiom-oracles @ git+https://github.com/TheAxiomFoundation/"
    "axiom-oracles@e1374eb30c582639f8f71f9bf9c22ba93b6e36f4"
)

# workflow -> (interpreter the staged tree is installed for, extra flag, sync)
STAGING = {
    "bulk-encode.yml": (
        ".venv/bin/python",
        "--extra api",
        "uv sync --locked --python 3.13 --extra api --no-dev",
    ),
    "golden-regeneration.yml": (
        ".venv/bin/python",
        "--extra api",
        "uv sync --locked --python 3.13 --extra api --no-dev",
    ),
    "targeted-signed-reencode.yml": (
        ".venv/bin/python",
        "--extra api",
        "uv sync --locked --python 3.13 --extra api --no-dev",
    ),
    "signed-apply-reusable.yml": (
        '"$PYBIN"',
        "",
        'uv sync --locked --no-dev --python "$PYBIN"',
    ),
}


def _logical_lines(script: str) -> list[str]:
    joined = re.sub(r"\\\n\s*", " ", script)
    lines = []
    for line in joined.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(re.sub(r"\s+", " ", stripped))
    return lines


def _workflow_steps():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        for job_name, job in workflow.get("jobs", {}).items():
            for index, step in enumerate(job.get("steps", [])):
                if "run" in step:
                    yield path.name, job_name, index, step


def _target_installs():
    for workflow, job, index, step in _workflow_steps():
        for line in _logical_lines(step["run"]):
            if re.search(r"\bpip install\b", line) and re.search(
                r"(^|\s)(--target|-t)(\s|=)", line
            ):
                yield workflow, job, index, step, line


def _canonical_staging(python: str, extra: str, sync: str) -> list[str]:
    extra = f" {extra}" if extra else ""
    install = f'uv pip install --python {python} --target "$site_packages"'
    return [
        "set -euo pipefail",
        sync,
        f'site_packages="{SITE_PACKAGES}"',
        'locked="$RUNNER_TEMP/axiom-verification-locked-requirements.txt"',
        'rm -rf "$site_packages"',
        f"uv export --locked --no-dev{extra} --no-emit-project"
        " --no-emit-package axiom-oracles --format requirements-txt"
        ' --output-file "$locked"',
        f'oracles="$(uv export --locked --no-dev{extra} --no-emit-project'
        " --no-hashes --no-annotate --no-header --format requirements-txt"
        f" | grep -xE '{ORACLES_PATTERN}')\"",
        f'{install} --require-hashes --no-deps -r "$locked"',
        f'{install} --no-deps "$oracles"',
        f"{install} --no-deps .",
        "uv pip freeze --python .venv/bin/python --exclude-editable"
        ' > "$RUNNER_TEMP/axiom-verification-locked.freeze"',
        f'uv pip freeze --python {python} --target "$site_packages"'
        ' --exclude axiom-encode > "$RUNNER_TEMP/axiom-verification-staged.freeze"',
        'test -s "$RUNNER_TEMP/axiom-verification-locked.freeze"',
        'diff -u "$RUNNER_TEMP/axiom-verification-locked.freeze"'
        ' "$RUNNER_TEMP/axiom-verification-staged.freeze"',
    ]


def _staging_step(workflow: str) -> dict:
    steps = {
        (job, index): step
        for name, job, index, step, _ in _target_installs()
        if name == workflow
    }
    assert len(steps) == 1, f"{workflow}: expected one staging step, got {steps}"
    return next(iter(steps.values()))


def test_every_workflow_target_install_is_locked():
    installs = [row for row in _target_installs() if row[0] not in EXEMPT_WORKFLOWS]

    # Non-vacuity: the discovery really sees every verification install.
    assert {row[0] for row in installs} == set(STAGING)
    for workflow, job, _index, _step, line in installs:
        where = f"{workflow}:{job}: {line}"
        assert line.startswith("uv pip install --python "), where
        assert "--no-deps" in line.split(), where
        assert '--target "$site_packages"' in line, where
        assert ".[" not in line, where
        requirement = line.split("--no-deps", 1)[1].strip()
        assert requirement in {'-r "$locked"', '"$oracles"', "."}, where
        if requirement == '-r "$locked"':
            assert "--require-hashes" in line.split(), where
        else:
            assert "--require-hashes" not in line.split(), where


def test_exempt_workflows_still_exist_and_still_need_the_exemption():
    for workflow in EXEMPT_WORKFLOWS:
        assert (WORKFLOWS / workflow).is_file()
        assert any(row[0] == workflow for row in _target_installs())


@pytest.mark.parametrize("workflow", sorted(STAGING))
def test_staging_step_is_exactly_the_locked_install(workflow):
    step = _staging_step(workflow)
    lines = _logical_lines(step["run"])
    canonical = _canonical_staging(*STAGING[workflow])

    if workflow == "signed-apply-reusable.yml":
        # The interpreter the tree is installed for must be the toolcache
        # python the provisioner copies.
        assert lines[0] == "set -euo pipefail"
        assert lines[1] == 'case "$PYBIN" in'
        del lines[1 : lines.index("esac") + 1]
    assert lines == canonical


def test_every_production_provision_consumes_the_locked_tree():
    provisions = 0
    for workflow, job, index, step in _workflow_steps():
        if workflow in EXEMPT_WORKFLOWS:
            continue
        run = " ".join(_logical_lines(step["run"]))
        if "provision_verification_supervisor.py" not in run:
            continue
        provisions += 1
        site_packages = re.search(r"--site-packages (\S+)", run).group(1)
        if site_packages == '"$staged"':
            assert f'staged="{SITE_PACKAGES}"' in run, workflow
        else:
            assert site_packages == f'"{SITE_PACKAGES}"', workflow
        staged_before = [
            staging_index
            for name, staging_job, staging_index, _step, _line in _target_installs()
            if name == workflow and staging_job == job
        ]
        assert staged_before and max(staged_before) <= index, workflow
    assert provisions == len(STAGING)


FAKE_UV = """\
import json, os, sys

argv = sys.argv[1:]
with open(os.environ["FAKE_UV_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps(argv) + "\\n")
if argv[:1] == ["export"]:
    if os.environ.get("FAKE_EXPORT_FAIL"):
        sys.exit(2)
    if "--no-hashes" in argv:
        sys.stdout.write(os.environ["FAKE_UNHASHED_EXPORT"])
    else:
        target = argv[argv.index("--output-file") + 1]
        with open(target, "w", encoding="utf-8") as handle:
            handle.write("requests==2.32.5 \\\\\\n    --hash=sha256:" + "0" * 64 + "\\n")
elif argv[:2] == ["pip", "freeze"]:
    key = "FAKE_STAGED_FREEZE" if "--target" in argv else "FAKE_LOCKED_FREEZE"
    sys.stdout.write(os.environ[key])
"""

LOCKED_FREEZE = f"claude-agent-sdk==0.1.47\n{LOCKED_GIT_LINE}\nrequests==2.32.5\n"


def _run_staging_step(
    tmp_path: Path,
    workflow: str,
    *,
    unhashed_export: str | None = None,
    staged_freeze: str = LOCKED_FREEZE,
    export_fails: bool = False,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(f"#!{sys.executable}\n{FAKE_UV}", encoding="utf-8")
    fake_uv.chmod(0o755)
    toolcache = tmp_path / "toolcache"
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    log = tmp_path / "uv-argv.jsonl"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "RUNNER_TEMP": str(runner_temp),
        "RUNNER_TOOL_CACHE": str(toolcache),
        "PYBIN": f"{toolcache}/Python/3.14.4/x64/bin/python",
        "FAKE_UV_LOG": str(log),
        "FAKE_UNHASHED_EXPORT": (
            unhashed_export
            if unhashed_export is not None
            else f"claude-agent-sdk==0.1.47\n{LOCKED_GIT_LINE}\nrequests==2.32.5\n"
        ),
        "FAKE_LOCKED_FREEZE": LOCKED_FREEZE,
        "FAKE_STAGED_FREEZE": staged_freeze,
    }
    if export_fails:
        env["FAKE_EXPORT_FAIL"] = "1"
    result = subprocess.run(
        ["/bin/bash", "-e", "-c", _staging_step(workflow)["run"]],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )
    return result, calls, runner_temp


@pytest.mark.parametrize("workflow", sorted(STAGING))
def test_staging_step_runs_the_locked_sequence(tmp_path, workflow):
    result, calls, runner_temp = _run_staging_step(tmp_path, workflow)

    assert result.returncode == 0, result.stderr
    python, extra, sync = STAGING[workflow]
    python = python.replace(
        '"$PYBIN"', f"{tmp_path}/toolcache/Python/3.14.4/x64/bin/python"
    )
    extra_args = extra.split()
    site_packages = str(runner_temp / "axiom-verification-site-packages")
    locked = str(runner_temp / "axiom-verification-locked-requirements.txt")
    install = ["pip", "install", "--python", python, "--target", site_packages]
    export = ["export", "--locked", "--no-dev", *extra_args, "--no-emit-project"]
    sync_argv = sync.split()[1:]
    if "$PYBIN" in sync:
        sync_argv[sync_argv.index('"$PYBIN"')] = python
    assert calls == [
        sync_argv,
        [
            *export,
            "--no-emit-package",
            "axiom-oracles",
            "--format",
            "requirements-txt",
            "--output-file",
            locked,
        ],
        [
            *export,
            "--no-hashes",
            "--no-annotate",
            "--no-header",
            "--format",
            "requirements-txt",
        ],
        [*install, "--require-hashes", "--no-deps", "-r", locked],
        [*install, "--no-deps", LOCKED_GIT_LINE],
        [*install, "--no-deps", "."],
        ["pip", "freeze", "--python", ".venv/bin/python", "--exclude-editable"],
        [
            "pip",
            "freeze",
            "--python",
            python,
            "--target",
            site_packages,
            "--exclude",
            "axiom-encode",
        ],
    ]


@pytest.mark.parametrize("workflow", sorted(STAGING))
@pytest.mark.parametrize(
    "unhashed_export",
    [
        pytest.param(
            "axiom-oracles @ git+https://github.com/TheAxiomFoundation/axiom-oracles@main\n",
            id="branch-ref",
        ),
        pytest.param(
            "axiom-oracles @ git+https://github.com/attacker/axiom-oracles@"
            + "e" * 40
            + "\n",
            id="foreign-repository",
        ),
        pytest.param("requests==2.32.5\n", id="missing"),
    ],
)
def test_staging_step_refuses_an_unpinned_git_dependency(
    tmp_path, workflow, unhashed_export
):
    result, calls, _ = _run_staging_step(
        tmp_path, workflow, unhashed_export=unhashed_export
    )

    assert result.returncode != 0
    assert not any(call[:2] == ["pip", "install"] for call in calls)


@pytest.mark.parametrize("workflow", sorted(STAGING))
def test_staging_step_fails_when_the_staged_tree_drifts_from_the_lock(
    tmp_path, workflow
):
    drifted = LOCKED_FREEZE.replace("0.1.47", "0.2.159")
    result, calls, _ = _run_staging_step(tmp_path, workflow, staged_freeze=drifted)

    assert result.returncode != 0
    assert "+claude-agent-sdk==0.2.159" in result.stdout
    assert calls[-1][:2] == ["pip", "freeze"]


@pytest.mark.parametrize("workflow", sorted(STAGING))
def test_staging_step_fails_when_the_lock_cannot_be_exported(tmp_path, workflow):
    result, calls, _ = _run_staging_step(tmp_path, workflow, export_fails=True)

    assert result.returncode != 0
    assert not any(call[:2] == ["pip", "install"] for call in calls)


def test_signed_apply_staging_refuses_a_non_toolcache_interpreter(tmp_path):
    step = _staging_step("signed-apply-reusable.yml")
    result = subprocess.run(
        ["/bin/bash", "-e", "-c", step["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            "RUNNER_TOOL_CACHE": str(tmp_path / "toolcache"),
            "PYBIN": "/usr/bin/python3",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "resolved interpreter is not the toolcache python" in result.stdout
