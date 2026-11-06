"""Run the bump workflow's own steps against a local remote and a fake gh.

axiom-encode#2051 review: a run that pushed its branch and then failed to open
the pull request left a branch that made every rerun on the same main commit
fail. A bump pull request in the merge queue must also stay there.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github/workflows/encoder-version-bump.yml"
SCRIPT = ROOT / "src/axiom_encode/version_provenance.py"
REMOTE_URL = "https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"

FAKE_GH = r"""
import json, os, sys
path = os.environ["FAKE_GH_STATE"]
state = json.load(open(path))
args = sys.argv[1:]
state["calls"].append(args)

def save():
    json.dump(state, open(path, "w"))

def value(flag):
    return args[args.index(flag) + 1]

def open_prs():
    return [pr for pr in state["prs"] if pr["state"] == "open"]

def find(url):
    return next(pr for pr in state["prs"] if pr["url"] == url)

if args[:2] == ["api", "graphql"]:
    print(state["queued"])
elif args[:2] == ["pr", "list"] and "--head" in args:
    head = value("--head")
    print(next((pr["url"] for pr in open_prs() if pr["head"] == head), ""))
elif args[:2] == ["pr", "list"]:
    for pr in open_prs():
        print(pr["number"], pr["head"])
elif args[:2] == ["pr", "create"]:
    if state["fail_create"]:
        state["fail_create"] = False
        save()
        sys.exit("gh: could not create the pull request")
    number = 100 + len(state["prs"])
    url = f"https://github.com/o/r/pull/{number}"
    state["prs"].append(
        {"number": number, "url": url, "head": value("--head"), "state": "open",
         "auto": False}
    )
    print(url)
elif args[:2] == ["pr", "view"]:
    print("false" if find(args[2])["auto"] else "true")
elif args[:2] == ["pr", "merge"]:
    find(args[2])["auto"] = True
elif args[:2] == ["pr", "close"]:
    next(pr for pr in state["prs"] if str(pr["number"]) == args[2])["state"] = "closed"
else:
    sys.exit(f"fake gh: unexpected call {args}")
save()
"""


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=True,
        text=True,
        env=env,
    ).stdout.strip()


def _steps() -> dict[str, str]:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    return {
        step["name"]: step.get("run", "") for step in workflow["jobs"]["bump"]["steps"]
    }


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    """A bare remote whose main has an encoder change that no bump covers."""
    origin = tmp_path / "origin"
    (origin / "src/axiom_encode").mkdir(parents=True)
    (origin / "src/axiom_encode/version_provenance.py").write_text(SCRIPT.read_text())
    (origin / "src/axiom_encode/cli.py").write_text("MARKER = 0\n")
    (origin / "src/axiom_encode/__init__.py").write_text('__version__ = "0.2.10"\n')
    (origin / "pyproject.toml").write_text(
        '[project]\nname = "axiom-encode"\nversion = "0.2.10"\n'
    )
    (origin / "uv.lock").write_text(
        '[[package]]\nname = "axiom-encode"\nversion = "0.2.10"\n'
        'source = { editable = "." }\n'
    )
    (origin / "CHANGELOG.md").write_text("## 0.2.10\n\n- Older entry.\n")
    _git(origin.parent, "init", "-q", "-b", "main", str(origin))
    for key, value in (
        ("user.email", "test@example.com"),
        ("user.name", "Test"),
        ("commit.gpgsign", "false"),
    ):
        _git(origin, "config", key, value)
    _git(origin, "add", "-A")
    _git(origin, "commit", "-qm", "init")
    (origin / "src/axiom_encode/cli.py").write_text("MARKER = 1\n")
    _git(origin, "commit", "-qam", "encoder change")
    bare = tmp_path / "remote.git"
    _git(tmp_path, "clone", "-q", "--bare", str(origin), str(bare))
    return bare


def _state(tmp_path: Path, **values) -> Path:
    path = tmp_path / "gh-state.json"
    state = {"prs": [], "fail_create": False, "queued": 0, "calls": []}
    state.update(values)
    path.write_text(json.dumps(state))
    return path


def _run(tmp_path: Path, remote: Path, state: Path, run: int):
    """One workflow run: a fresh clone, then the write and open steps."""
    workspace = tmp_path / f"run-{run}"
    _git(tmp_path, "clone", "-q", str(remote), str(workspace))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "gh").write_text(f"#!{sys.executable}\n{FAKE_GH}")
    (bin_dir / "gh").chmod(0o755)
    output = workspace.parent / f"output-{run}"
    output.write_text("")
    # Each run commits at its own time, as reruns do.
    date = f"2026-11-06T0{run}:00:00+00:00"
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "GITHUB_OUTPUT": str(output),
        "GITHUB_REPOSITORY": "o/r",
        "GH_TOKEN": "unused",
        "FAKE_GH_STATE": str(state),
        "GIT_AUTHOR_DATE": date,
        "GIT_COMMITTER_DATE": date,
    }
    steps = _steps()
    subprocess.run(
        ["bash", "-c", steps["Write the next version and changelog"]],
        cwd=workspace,
        env=env,
        check=True,
        capture_output=True,
    )
    outputs = dict(
        line.split("=", 1) for line in output.read_text().splitlines() if "=" in line
    )
    script = steps["Open the bump pull request with auto-merge"]
    assert REMOTE_URL in script
    script = script.replace(REMOTE_URL, f"file://{remote}")
    return subprocess.run(
        ["bash", "-c", script],
        cwd=workspace,
        env={
            **env,
            "VERSION": outputs["version"],
            "BASE_SHA": outputs["base_sha"],
            "BRANCH": outputs["branch"],
        },
        capture_output=True,
        text=True,
    ), outputs


def _remote_branch(remote: Path, branch: str) -> str | None:
    heads = _git(remote, "for-each-ref", "--format=%(refname:short) %(objectname)")
    return dict(line.split() for line in heads.splitlines()).get(branch)


def test_rerun_after_a_failed_pull_request_reuses_the_pushed_branch(
    tmp_path: Path, remote: Path
):
    state = _state(tmp_path, fail_create=True)

    first, outputs = _run(tmp_path, remote, state, 1)
    assert first.returncode != 0
    pushed = _remote_branch(remote, outputs["branch"])
    assert pushed is not None
    assert json.loads(state.read_text())["prs"] == []

    rerun, rerun_outputs = _run(tmp_path, remote, state, 2)
    assert rerun.returncode == 0, rerun.stderr
    assert rerun_outputs["branch"] == outputs["branch"]
    # The rerun's own commit differs (a later date), so it reused the branch.
    assert _remote_branch(remote, outputs["branch"]) == pushed
    prs = json.loads(state.read_text())["prs"]
    assert [(pr["head"], pr["state"], pr["auto"]) for pr in prs] == [
        (outputs["branch"], "open", True)
    ]


def test_a_branch_that_is_not_this_bump_stops_the_run(tmp_path: Path, remote: Path):
    state = _state(tmp_path, fail_create=True)
    first, outputs = _run(tmp_path, remote, state, 1)
    assert first.returncode != 0
    # Someone else's commit on the bump branch name.
    other = tmp_path / "other"
    _git(tmp_path, "clone", "-q", str(remote), str(other))
    (other / "CHANGELOG.md").write_text("tampered\n")
    _git(
        other,
        "-c",
        "user.email=x@example.com",
        "-c",
        "user.name=X",
        "commit",
        "-qam",
        "tampered",
    )
    _git(other, "push", "-qf", "origin", f"HEAD:refs/heads/{outputs['branch']}")
    tampered = _remote_branch(remote, outputs["branch"])

    rerun, _ = _run(tmp_path, remote, state, 2)

    assert rerun.returncode != 0
    assert "is not this bump" in rerun.stdout + rerun.stderr
    assert _remote_branch(remote, outputs["branch"]) == tampered
    assert json.loads(state.read_text())["prs"] == []


def test_a_queued_bump_pull_request_is_left_in_the_queue(tmp_path: Path, remote: Path):
    queued = {
        "number": 7,
        "url": "https://github.com/o/r/pull/7",
        "head": "automation/encoder-version-0.2.10-aaaaaaaaaaaa",
        "state": "open",
        "auto": True,
    }
    state = _state(tmp_path, queued=1, prs=[queued])

    result, outputs = _run(tmp_path, remote, state, 1)

    assert result.returncode == 0, result.stderr
    assert "merge queue" in result.stdout
    assert json.loads(state.read_text())["prs"] == [queued]
    assert _remote_branch(remote, outputs["branch"]) is None


def test_an_older_bump_pull_request_outside_the_queue_is_superseded(
    tmp_path: Path, remote: Path
):
    older = {
        "number": 7,
        "url": "https://github.com/o/r/pull/7",
        "head": "automation/encoder-version-0.2.10-aaaaaaaaaaaa",
        "state": "open",
        "auto": True,
    }
    state = _state(tmp_path, prs=[older])

    result, outputs = _run(tmp_path, remote, state, 1)

    assert result.returncode == 0, result.stderr
    prs = json.loads(state.read_text())["prs"]
    assert [(pr["head"], pr["state"]) for pr in prs] == [
        (older["head"], "closed"),
        (outputs["branch"], "open"),
    ]
