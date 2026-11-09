"""Run the bump workflow's own steps against a local remote and a fake gh.

axiom-encode#2051 review: a run that pushed its branch and then failed to open
the pull request left a branch that made every rerun on the same main commit
fail. A bump pull request in the merge queue must also stay there.
"""

from __future__ import annotations

import json
import os
import shutil
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
import json, os, subprocess, sys
path = os.environ["FAKE_GH_STATE"]
state = json.load(open(path))
args = sys.argv[1:]
state["calls"].append(args)
APP = "axiom-encoder-version-bump"

def save():
    json.dump(state, open(path, "w"))

def value(flag):
    return args[args.index(flag) + 1]

def open_prs():
    return [pr for pr in state["prs"] if pr["state"] == "open"]

def find(url):
    return next(pr for pr in state["prs"] if pr["url"] == url)

def head_oid(pr):
    if pr.get("oid"):
        return pr["oid"]
    found = subprocess.run(
        ["git", "--git-dir", state["remote"], "rev-parse", "--verify", "-q",
         f"refs/heads/{pr['head']}"],
        capture_output=True, text=True,
    ).stdout.strip()
    return found

def as_json(pr, graphql=False):
    # gh reports an App author as app/<slug>; GraphQL as the bare slug, with
    # __typename Bot. A user account can hold the bare slug as its login.
    login = pr.get("author", "app")
    author = {"login": login}
    if login == "app":
        author["login"] = APP if graphql else f"app/{APP}"
    if graphql:
        author["__typename"] = "Bot" if login == "app" else "User"
    document = {
        "number": pr["number"], "url": pr["url"], "headRefName": pr["head"],
        "headRefOid": head_oid(pr), "isCrossRepository": pr.get("fork", False),
        "author": author,
        "autoMergeRequest": {"enabledAt": "x"} if pr["auto"] else None,
    }
    if graphql:
        # Like GitHub: GraphQL returns only the fields the query selects.
        query = next(arg for arg in args if arg.startswith("query="))
        document["author"] = {k: v for k, v in author.items() if k in query}
        document = {k: v for k, v in document.items() if k in query}
    return document

def emit(document):
    if "--jq" in args:
        out = subprocess.run(
            ["jq", "-r", value("--jq")], input=json.dumps(document),
            capture_output=True, text=True,
        )
        if out.returncode:
            sys.exit(f"fake gh: jq failed: {out.stderr}")
        sys.stdout.write(out.stdout)
    else:
        print(json.dumps(document))

if args[:2] == ["api", "graphql"]:
    nodes = [{"pullRequest": as_json(pr, graphql=True)} for pr in state["queue"]]
    nodes.append({"pullRequest": None})
    emit({"data": {"repository": {"mergeQueue": {"entries": {"nodes": nodes}}}}})
elif args[:2] == ["pr", "list"]:
    prs = open_prs()
    if "--head" in args:
        # Like GitHub: the branch name alone, whichever repository it is in.
        prs = [pr for pr in prs if pr["head"] == value("--head")]
    # Like gh: 30 pull requests unless --limit says otherwise.
    prs = prs[: int(value("--limit")) if "--limit" in args else 30]
    emit([as_json(pr) for pr in prs])
elif args[:2] == ["pr", "create"]:
    if state["fail_create"]:
        state["fail_create"] = False
        save()
        sys.exit("gh: could not create the pull request")
    number = 100 + len(state["prs"])
    url = f"https://github.com/o/r/pull/{number}"
    state["prs"].append(
        {"number": number, "url": url, "head": value("--head"), "state": "open",
         "auto": False, **state.get("create_as", {})}
    )
    print(url)
elif args[:2] == ["pr", "view"]:
    emit(as_json(find(args[2])))
elif args[:2] == ["pr", "merge"]:
    pr = find(args[2])
    pr["auto"] = True
    pr["match_head_commit"] = value("--match-head-commit")
elif args[:2] == ["pr", "close"]:
    next(pr for pr in state["prs"] if str(pr["number"]) == args[2])["state"] = "closed"
else:
    sys.exit(f"fake gh: unexpected call {args}")
save()
"""

pytestmark = pytest.mark.skipif(
    shutil.which("jq") is None, reason="the fake gh applies --jq with jq"
)


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
    state = {
        "prs": [],
        "queue": [],
        "fail_create": False,
        "calls": [],
        "remote": str(tmp_path / "remote.git"),
    }
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
    state = _state(tmp_path, queue=[queued], prs=[queued])

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


def test_an_older_bump_pull_request_past_the_first_page_is_superseded(
    tmp_path: Path, remote: Path
):
    # gh pr list returns 30 pull requests by default; this repository has
    # more open than that, so the close loop must ask for more.
    others = [_pr(n, f"feature/{n}") for n in range(10, 50)]
    older = _pr(60, "automation/encoder-version-0.2.10-aaaaaaaaaaaa", auto=True)
    state = _state(tmp_path, prs=[*others, older])

    result, _ = _run(tmp_path, remote, state, 1)

    assert result.returncode == 0, result.stderr
    prs = {pr["number"]: pr for pr in json.loads(state.read_text())["prs"]}
    assert prs[60]["state"] == "closed"


def _pr(number: int, head: str, **values) -> dict:
    pr = {
        "number": number,
        "url": f"https://github.com/o/r/pull/{number}",
        "head": head,
        "state": "open",
        "auto": False,
    }
    pr.update(values)
    return pr


def _merges(state: Path) -> list[list[str]]:
    return [
        call
        for call in json.loads(state.read_text())["calls"]
        if call[:2] == ["pr", "merge"]
    ]


def test_a_fork_pull_request_with_the_bump_branch_name_is_never_auto_merged(
    tmp_path: Path, remote: Path
):
    # GitHub's --head filter matches the branch name alone, so a fork's pull
    # request named like the bump branch must not be mistaken for the App's.
    clone = tmp_path / "probe"
    _git(tmp_path, "clone", "-q", str(remote), str(clone))
    base = _git(clone, "rev-parse", "HEAD")
    state = _state(tmp_path)
    data = json.loads(state.read_text())
    # The branch name is predictable: main's version plus one and main's SHA.
    branch = f"automation/encoder-version-0.2.11-{base[:12]}"
    data["prs"] = [_pr(7, branch, fork=True, author="mallory", oid="f" * 40)]
    state.write_text(json.dumps(data))

    result, outputs = _run(tmp_path, remote, state, 1)

    assert result.returncode == 0, result.stderr
    assert outputs["branch"] == branch
    prs = {pr["number"]: pr for pr in json.loads(state.read_text())["prs"]}
    assert prs[7]["auto"] is False and prs[7]["state"] == "open"
    created = [pr for number, pr in prs.items() if number != 7]
    assert len(created) == 1 and created[0]["auto"] is True
    pushed = _remote_branch(remote, branch)
    assert created[0]["match_head_commit"] == pushed
    assert [call[2] for call in _merges(state)] == [created[0]["url"]]


def test_only_the_apps_own_bump_pull_requests_are_held_for_or_closed(
    tmp_path: Path, remote: Path
):
    prefix = "automation/encoder-version-0.2.10-"
    fork = _pr(7, prefix + "aaaaaaaaaaaa", fork=True, author="mallory")
    human = _pr(8, prefix + "bbbbbbbbbbbb", author="someone")
    other = _pr(9, "feature/unrelated")
    # A fork's pull request whose author reads as the App: only the
    # same-repository check keeps it out.
    app_fork = _pr(10, prefix + "cccccccccccc", fork=True)
    # A user account registered with the App's bare slug as its login: only
    # the Bot type keeps it out of the queue hold.
    squatter = _pr(11, prefix + "dddddddddddd", author="axiom-encoder-version-bump")
    prs = [fork, human, other, app_fork, squatter]
    # Queued pull requests named like a bump, none of them the App's own.
    state = _state(tmp_path, queue=prs, prs=prs)

    result, outputs = _run(tmp_path, remote, state, 1)

    assert result.returncode == 0, result.stderr
    assert "merge queue" not in result.stdout
    prs = {pr["number"]: pr for pr in json.loads(state.read_text())["prs"]}
    assert {number: prs[number]["state"] for number in (7, 8, 9, 10, 11)} == {
        7: "open",
        8: "open",
        9: "open",
        10: "open",
        11: "open",
    }
    assert _remote_branch(remote, outputs["branch"]) is not None


def test_an_existing_app_pull_request_is_reused_not_recreated(
    tmp_path: Path, remote: Path
):
    state = _state(tmp_path, fail_create=True)
    first, outputs = _run(tmp_path, remote, state, 1)
    assert first.returncode != 0
    data = json.loads(state.read_text())
    data["prs"] = [_pr(7, outputs["branch"])]
    data["calls"] = []
    state.write_text(json.dumps(data))

    rerun, _ = _run(tmp_path, remote, state, 2)

    assert rerun.returncode == 0, rerun.stderr
    calls = json.loads(state.read_text())["calls"]
    assert not [call for call in calls if call[:2] == ["pr", "create"]]
    assert [call[2] for call in _merges(state)] == ["https://github.com/o/r/pull/7"]


def test_an_app_pull_request_at_another_commit_is_not_auto_merged(
    tmp_path: Path, remote: Path
):
    state = _state(tmp_path, fail_create=True)
    first, outputs = _run(tmp_path, remote, state, 1)
    assert first.returncode != 0
    data = json.loads(state.read_text())
    data["prs"] = [_pr(7, outputs["branch"], oid="e" * 40)]
    state.write_text(json.dumps(data))

    rerun, _ = _run(tmp_path, remote, state, 2)

    assert rerun.returncode != 0
    assert "is not the App's pull request" in rerun.stdout + rerun.stderr
    assert _merges(state) == []


@pytest.mark.parametrize("opened_as", [{"author": "mallory"}, {"fork": True}])
def test_a_pull_request_at_the_bump_commit_that_is_not_the_apps_is_not_auto_merged(
    opened_as: dict, tmp_path: Path, remote: Path
):
    # Its head is the bump commit, so only the author test can stop it.
    state = _state(tmp_path, create_as=opened_as)

    result, _ = _run(tmp_path, remote, state, 1)

    assert result.returncode != 0
    assert "is not the App's pull request" in result.stdout + result.stderr
    assert _merges(state) == []


def test_a_branch_with_the_bumps_tree_on_another_parent_stops_the_run(
    tmp_path: Path, remote: Path
):
    state = _state(tmp_path, fail_create=True)
    first, outputs = _run(tmp_path, remote, state, 1)
    assert first.returncode != 0
    branch = outputs["branch"]
    pushed = _remote_branch(remote, branch)
    # Same tree as the bump, but on main's parent instead of main.
    forged = _git(
        remote,
        "-c",
        "user.email=x@example.com",
        "-c",
        "user.name=X",
        "commit-tree",
        f"{pushed}^{{tree}}",
        "-p",
        f"{outputs['base_sha']}^",
        "-m",
        "forged",
    )
    _git(remote, "update-ref", f"refs/heads/{branch}", forged)

    rerun, _ = _run(tmp_path, remote, state, 2)

    assert rerun.returncode != 0
    assert "is not this bump" in rerun.stdout + rerun.stderr
    assert _remote_branch(remote, branch) == forged
    assert _merges(state) == []


@pytest.mark.parametrize(
    ("client_id", "key_set", "ok"),
    [("", "true", False), ("Iv1.abc", "false", False), ("Iv1.abc", "true", True)],
)
def test_the_run_stops_without_the_bump_app(client_id: str, key_set: str, ok: bool):
    result = subprocess.run(
        ["bash", "-c", _steps()["Require the bump App"]],
        env={
            "PATH": os.environ["PATH"],
            "APP_CLIENT_ID": client_id,
            "APP_KEY_SET": key_set,
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is ok


def test_the_bump_commit_is_checked_before_it_is_published():
    script = _steps()["Write the next version and changelog"]
    commit = script.index("git commit")
    check = script.index("version_provenance.py check --repo . --ref HEAD", commit)
    assert check > commit
    assert "set -euo pipefail" in script


def test_a_fork_pull_request_that_reads_as_the_app_is_never_auto_merged(
    tmp_path: Path, remote: Path
):
    clone = tmp_path / "probe"
    _git(tmp_path, "clone", "-q", str(remote), str(clone))
    base = _git(clone, "rev-parse", "HEAD")
    branch = f"automation/encoder-version-0.2.11-{base[:12]}"
    state = _state(tmp_path, prs=[_pr(7, branch, fork=True, oid="f" * 40)])

    result, _ = _run(tmp_path, remote, state, 1)

    assert result.returncode == 0, result.stderr
    prs = {pr["number"]: pr for pr in json.loads(state.read_text())["prs"]}
    assert prs[7]["auto"] is False
    assert [call[2] for call in _merges(state)] != [prs[7]["url"]]
