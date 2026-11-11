import importlib.util
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from axiom_encode import cli, version_provenance
from axiom_encode.ci_parity import encoder_version_at_pin
from axiom_encode.cli import _latest_axiom_encode_version_commit

SCRIPT = Path(version_provenance.__file__)
ROOT = Path(__file__).parents[1]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=True, text=True
    ).stdout.strip()


def _write_version(repo: Path, version: str) -> None:
    (repo / "pyproject.toml").write_text(
        f'[project]\nname = "axiom-encode"\nversion = "{version}"\n'
    )
    (repo / "src/axiom_encode/__init__.py").write_text(f'__version__ = "{version}"\n')
    (repo / "uv.lock").write_text(
        f'[[package]]\nname = "axiom-encode"\nversion = "{version}"\n'
        'source = { editable = "." }\n'
    )


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "axiom-encode"
    (repo / "src/axiom_encode").mkdir(parents=True)
    (repo / "changelog.d").mkdir()
    (repo / "changelog.d/.gitkeep").write_text("")
    (repo / "CHANGELOG.md").write_text("## 0.2.10\n\n- Older entry.\n")
    (repo / "src/axiom_encode/cli.py").write_text("MARKER = 0\n")
    _write_version(repo, "0.2.10")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    _commit(repo, "init")
    return repo


def _change_encoder(repo: Path, marker: int) -> str:
    (repo / "src/axiom_encode/cli.py").write_text(f"MARKER = {marker}\n")
    return _commit(repo, f"encoder change {marker}")


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", str(SCRIPT), *args], capture_output=True, text=True
    )


def test_next_version_increases_the_last_component():
    assert version_provenance.next_version("0.2.2298") == "0.2.2299"
    assert version_provenance.next_version("1.9") == "1.10"
    with pytest.raises(ValueError):
        version_provenance.next_version("0.2.dev1")


def test_encoder_change_after_the_bump_is_unversioned(repo: Path):
    _change_encoder(repo, 1)

    assert version_provenance.unversioned_changes(repo) == ["src/axiom_encode/cli.py"]
    refused = _run_script("check", "--repo", str(repo))
    assert refused.returncode == 1
    assert "src/axiom_encode/cli.py" in refused.stderr


def test_non_encoder_change_needs_no_bump(repo: Path):
    (repo / "README.md").write_text("docs\n")
    _commit(repo, "docs")

    assert version_provenance.unversioned_changes(repo) == []
    assert _run_script("check", "--repo", str(repo)).returncode == 0


def test_bump_covers_the_change_and_writes_all_version_files(repo: Path):
    _change_encoder(repo, 1)
    _change_encoder(repo, 2)

    assert version_provenance.bump(repo) == "0.2.11"
    _commit(repo, "Bump encoder version to 0.2.11")

    assert version_provenance.versions_at(repo, "HEAD") == {
        "pyproject": "0.2.11",
        "package": "0.2.11",
        "lock": "0.2.11",
    }
    assert version_provenance.unversioned_changes(repo) == []
    assert (repo / "CHANGELOG.md").read_text() == (
        "## 0.2.11\n\n- encoder change 2\n- encoder change 1\n\n"
        "## 0.2.10\n\n- Older entry.\n"
    )


def test_bump_leaves_towncrier_fragments_alone(repo: Path):
    # changelog.d/ holds towncrier fragments that CI checks with
    # `towncrier build --draft` and that nothing compiles yet.
    fragment = repo / "changelog.d/1120.added.md"
    fragment.write_text("Add a runner.\n")
    _commit(repo, "fragment")
    _change_encoder(repo, 1)

    version_provenance.bump(repo)

    assert fragment.read_text() == "Add a runner.\n"
    assert _git(repo, "status", "--porcelain", "--", "changelog.d") == ""
    assert "Add a runner." not in (repo / "CHANGELOG.md").read_text()


def test_github_output_mode_reports_instead_of_failing(repo: Path, tmp_path: Path):
    output = tmp_path / "github_output"
    assert (
        _run_script(
            "check", "--repo", str(repo), "--github-output", str(output)
        ).returncode
        == 0
    )
    _change_encoder(repo, 1)
    assert (
        _run_script(
            "check", "--repo", str(repo), "--github-output", str(output)
        ).returncode
        == 0
    )

    assert output.read_text() == "needs_bump=false\nneeds_bump=true\n"


def test_inconsistent_version_files_are_refused(repo: Path):
    (repo / "uv.lock").write_text('[[package]]\nname = "axiom-encode"\nversion = "9"\n')
    _commit(repo, "drift")

    with pytest.raises(ValueError, match="inconsistent"):
        version_provenance.unversioned_changes(repo)
    assert _run_script("check", "--repo", str(repo)).returncode == 1


def test_latest_version_commit_matches_the_apply_guard(repo: Path):
    # A feature branch bumps and changes the encoder while main bumps lower;
    # the merge must resolve to the feature's bump, as the --apply guard does.
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-qb", "feature", base)
    _write_version(repo, "0.2.12")
    (repo / "src/axiom_encode/cli.py").write_text("MARKER = 'feature'\n")
    feature = _commit(repo, "feature bump")
    _git(repo, "checkout", "-q", "main")
    _write_version(repo, "0.2.11")
    main = _commit(repo, "main bump")
    tree = _git(repo, "rev-parse", f"{feature}^{{tree}}")
    merge = _git(repo, "commit-tree", tree, "-p", main, "-p", feature, "-m", "merge")
    _git(repo, "checkout", "-q", "--detach", merge)

    for ref in (merge, main, feature):
        _git(repo, "checkout", "-q", "--detach", ref)
        assert version_provenance.latest_version_commit(repo, "HEAD") == (
            _latest_axiom_encode_version_commit(repo)
        )
    assert version_provenance.latest_version_commit(repo, merge) == feature


def test_ci_parity_refuses_a_pin_newer_than_its_version(repo: Path):
    versioned = _git(repo, "rev-parse", "HEAD")
    unversioned = _change_encoder(repo, 1)

    assert encoder_version_at_pin(repo, versioned) == "0.2.10"
    with pytest.raises(ValueError, match="after its version bump"):
        encoder_version_at_pin(repo, unversioned)


def test_ambient_git_dir_does_not_hide_an_unversioned_pin(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    unversioned = _change_encoder(repo, 1)
    # In the copy the pin is grafted into a root commit, which reads as a bump.
    copy = tmp_path / "copy"
    shutil.copytree(repo, copy)
    _git(copy, "replace", "--graft", unversioned)
    monkeypatch.setenv("GIT_DIR", str(copy / ".git"))

    with pytest.raises(ValueError, match="after its version bump"):
        encoder_version_at_pin(repo, unversioned)


@pytest.mark.parametrize("rewrite", ["replace-graft", "replace-object", "grafts-file"])
def test_history_rewrites_in_the_same_repo_do_not_hide_an_unversioned_pin(
    repo: Path, rewrite: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    bump = _git(repo, "rev-parse", "HEAD")
    unversioned = _change_encoder(repo, 1)
    if rewrite == "replace-graft":
        # The pin loses its parents, so it reads as its own initial bump.
        _git(repo, "replace", "--graft", unversioned)
    elif rewrite == "replace-object":
        # The pin's object is swapped for the versioned bump commit.
        _git(repo, "replace", unversioned, bump)
    else:
        (repo / ".git/info").mkdir(exist_ok=True)
        (repo / ".git/info/grafts").write_text(f"{unversioned}\n")

    with pytest.raises(ValueError, match="after its version bump"):
        encoder_version_at_pin(repo, unversioned)
    assert version_provenance.unversioned_changes(repo, unversioned)
    # The --apply guard and the provisioner see through the rewrite too.
    assert _verdicts(repo, monkeypatch, tmp_path / "home") == {True}


def test_a_replaced_versioned_pin_reports_its_stored_version(repo: Path):
    older_bump = _git(repo, "rev-parse", "HEAD")
    (repo / "src/axiom_encode/cli.py").write_text("MARKER = 1\n")
    _write_version(repo, "0.2.11")
    pin = _commit(repo, "encoder change under a bump")
    # The pin's object is swapped for a commit that carries another version.
    _git(repo, "replace", pin, older_bump)

    assert encoder_version_at_pin(repo, pin) == "0.2.11"


_PROVISIONER_SPEC = importlib.util.spec_from_file_location(
    "provision_verification_supervisor_for_provenance",
    Path(__file__).parents[1] / "scripts" / "provision_verification_supervisor.py",
)
provisioner = importlib.util.module_from_spec(_PROVISIONER_SPEC)
sys.modules[_PROVISIONER_SPEC.name] = provisioner
_PROVISIONER_SPEC.loader.exec_module(provisioner)


def _verdicts(repo: Path, monkeypatch: pytest.MonkeyPatch, home: Path) -> set[bool]:
    """Whether each copy of the rule refuses HEAD: version_provenance, cli and
    the provisioner. One rule gives one verdict."""
    monkeypatch.setattr(
        cli, "__version__", version_provenance.versions_at(repo, "HEAD")["pyproject"]
    )
    provenance = bool(version_provenance.unversioned_changes(repo))
    try:
        cli._require_axiom_encode_version_provenance(repo)
        cli_refuses = False
    except RuntimeError:
        cli_refuses = True
    try:
        provisioner._require_snapshot_version_bump(
            Path(shutil.which("git")),
            repo / ".git",
            home,
            _git(repo, "rev-parse", "HEAD"),
        )
        provisioner_refuses = False
    except SystemExit:
        provisioner_refuses = True
    return {provenance, cli_refuses, provisioner_refuses}


@pytest.mark.parametrize(
    "change",
    ["delete", "move-out", "non-ascii-name"],
)
def test_deleted_moved_and_quoted_encoder_paths_need_a_bump(
    repo: Path, change: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    (repo / "src/axiom_encode/validator.py").write_text("RULE = 1\n")
    _write_version(repo, "0.2.11")
    _commit(repo, "add validator under a bump")
    if change == "delete":
        (repo / "src/axiom_encode/validator.py").unlink()
    elif change == "move-out":
        (repo / "scripts").mkdir()
        _git(repo, "mv", "src/axiom_encode/validator.py", "scripts/validator.py")
    else:
        (repo / "src/axiom_encode/données.py").write_text("RULE = 2\n")
    _commit(repo, change)

    assert version_provenance.unversioned_changes(repo)
    assert _verdicts(repo, monkeypatch, tmp_path / "home") == {True}


def test_a_shallow_clone_is_refused(repo: Path, tmp_path: Path):
    _write_version(repo, "0.2.11")
    _commit(repo, "bump")
    _change_encoder(repo, 1)
    shallow = tmp_path / "shallow"
    _git(tmp_path, "clone", "-q", "--depth", "1", f"file://{repo}", str(shallow))

    with pytest.raises(ValueError, match="shallow"):
        version_provenance.unversioned_changes(shallow)
    assert _run_script("check", "--repo", str(shallow)).returncode == 1


def test_changelog_uses_pull_request_titles_and_skips_bump_merges(repo: Path):
    def merge_pr(number: int, branch: str, title: str, change) -> None:
        base = _git(repo, "rev-parse", "HEAD")
        _git(repo, "checkout", "-qb", branch, base)
        change()
        _commit(repo, f"work on {branch}")
        _git(repo, "checkout", "-q", "main")
        _git(
            repo,
            "merge",
            "-q",
            "--no-ff",
            branch,
            "-m",
            f"Merge pull request #{number} from org/{branch}",
            "-m",
            title,
        )

    def bump_files() -> None:
        _write_version(repo, "0.2.11")

    merge_pr(101, "automation/encoder-version-0.2.11-abc", "Bump", bump_files)

    def encoder_files(marker: int):
        return lambda: (repo / "src/axiom_encode/cli.py").write_text(
            f"MARKER = {marker}\n"
        )

    merge_pr(102, "feature-a", "Type the SNAP checks", encoder_files(1))
    merge_pr(103, "feature-b", "Fix the corpus pin", encoder_files(2))

    version_provenance.bump(repo)

    assert (
        (repo / "CHANGELOG.md")
        .read_text()
        .startswith(
            "## 0.2.12\n\n- Fix the corpus pin (#103)\n- Type the SNAP checks (#102)\n\n"
        )
    )


def test_the_three_copies_of_the_rule_agree_on_random_histories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Linear histories and merged feature branches with random bumps, encoder
    # changes, deletions and non-encoder changes. Each copy of the rule must give
    # the same verdict at every head.
    for seed in range(6):
        rng = random.Random(seed)
        root = tmp_path / f"history-{seed}"
        root.mkdir()
        repo = root / "axiom-encode"
        (repo / "src/axiom_encode").mkdir(parents=True)
        (repo / "CHANGELOG.md").write_text("")
        (repo / "src/axiom_encode/cli.py").write_text("MARKER = 0\n")
        version = 10
        _write_version(repo, f"0.2.{version}")
        _git(repo.parent, "init", "-q", "-b", "main", str(repo))
        _git(repo, "config", "user.email", "test@example.com")
        _git(repo, "config", "user.name", "Test")
        _git(repo, "config", "commit.gpgsign", "false")
        _commit(repo, "init")
        files = ["src/axiom_encode/cli.py"]
        for step in range(8):
            action = rng.choice(["bump", "encoder", "delete", "docs", "merge"])
            if action == "bump":
                version += 1
                _write_version(repo, f"0.2.{version}")
            elif action == "encoder":
                name = f"src/axiom_encode/m{step}.py"
                (repo / name).write_text(f"STEP = {step}\n")
                files.append(name)
            elif action == "delete" and len(files) > 1:
                (repo / files.pop()).unlink()
            elif action == "merge":
                base = _git(repo, "rev-parse", "HEAD")
                _git(repo, "checkout", "-qb", f"feature-{step}", base)
                version += 1
                _write_version(repo, f"0.2.{version}")
                (repo / f"src/axiom_encode/f{step}.py").write_text("F = 1\n")
                _commit(repo, f"feature {step}")
                _git(repo, "checkout", "-q", "main")
                # Main moves on while the feature branch is open.
                if rng.random() < 0.5:
                    (repo / f"src/axiom_encode/main{step}.py").write_text("M = 1\n")
                else:
                    (repo / f"docs-main-{step}.md").write_text("docs\n")
                _commit(repo, f"main {step}")
                _git(repo, "merge", "-q", "--no-ff", f"feature-{step}", "-m", "merge")
            else:
                (repo / f"docs-{step}.md").write_text("docs\n")
            if _git(repo, "status", "--porcelain"):
                _commit(repo, f"{action} {step}")
            verdicts = _verdicts(repo, monkeypatch, root / "home")
            assert len(verdicts) == 1, (seed, step, action, verdicts)


# --- axiom-encode#2051 review round 5: the tests from the review -------------
def _encoder_and_dependency_change(repo: Path) -> str:
    (repo / "src/axiom_encode/cli.py").write_text("MARKER = 1\n")
    with (repo / "pyproject.toml").open("a") as pyproject:
        pyproject.write('dependencies = ["requests"]\n')
    return _commit(repo, "encoder and dependency change, no bump")


@pytest.mark.parametrize("rewrite", ["replace-graft", "grafts-file"])
def test_rewrite_of_an_unversioned_commit_that_edits_a_version_file(
    repo: Path, rewrite: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    unversioned = _encoder_and_dependency_change(repo)
    if rewrite == "replace-graft":
        _git(repo, "replace", "--graft", unversioned)
    else:
        (repo / ".git/info").mkdir(exist_ok=True)
        (repo / ".git/info/grafts").write_text(f"{unversioned}\n")
    assert version_provenance.unversioned_changes(repo, unversioned)
    assert _verdicts(repo, monkeypatch, tmp_path / "home") == {True}


def test_a_replace_graft_does_not_make_the_guard_take_a_commit_as_its_bump(
    repo: Path,
):
    # Rule 2 refuses this head in every copy anyway, so read the guard's bump
    # lookup directly: grafted to a root, the unversioned commit has no parent
    # and would read as a bump.
    bump = _git(repo, "rev-parse", "HEAD")
    unversioned = _encoder_and_dependency_change(repo)
    _git(repo, "replace", "--graft", unversioned)
    assert _git(repo, "rev-list", "--parents", "-n", "1", unversioned) == unversioned
    assert _latest_axiom_encode_version_commit(repo) == bump
    assert version_provenance.latest_version_commit(repo) == bump


def test_ambient_git_dir_does_not_change_the_apply_guard(
    repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    unversioned = _encoder_and_dependency_change(repo)
    copy = tmp_path / "copy"
    shutil.copytree(repo, copy)
    _git(copy, "replace", "--graft", unversioned)
    monkeypatch.setenv("GIT_DIR", str(copy / ".git"))
    assert _verdicts(repo, monkeypatch, tmp_path / "home") == {True}


def test_the_apply_guard_refuses_a_shallow_clone(
    repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _write_version(repo, "0.2.11")
    _commit(repo, "bump")
    _change_encoder(repo, 1)
    shallow = tmp_path / "shallow"
    _git(tmp_path, "clone", "-q", "--depth", "1", f"file://{repo}", str(shallow))
    with pytest.raises(ValueError, match="shallow"):
        version_provenance.unversioned_changes(shallow)
    monkeypatch.setattr(cli, "__version__", "0.2.11")
    with pytest.raises(RuntimeError):
        cli._require_axiom_encode_version_provenance(shallow)


def test_a_lowered_version_never_reissues_a_used_number(repo: Path):
    base = _git(repo, "rev-parse", "HEAD")
    # A PR opened before the change, with its own hand bump to 0.2.11.
    _git(repo, "checkout", "-qb", "pr-x", base)
    (repo / "src/axiom_encode/x.py").write_text("X = 1\n")
    _write_version(repo, "0.2.11")
    _commit(repo, "PR X: encoder change and hand bump to 0.2.11")
    _git(repo, "checkout", "-q", "main")
    # The change lands at 0.2.11, an encoder PR lands with no bump, the bot bumps.
    (repo / "src/axiom_encode/v.py").write_text("V = 1\n")
    _write_version(repo, "0.2.11")
    _commit(repo, "hand bump to 0.2.11")
    (repo / "src/axiom_encode/d.py").write_text("D = 1\n")
    _commit(repo, "PR D lands with no bump")
    assert version_provenance.bump(repo) == "0.2.12"
    _commit(repo, "Bump encoder version to 0.2.12")
    # PR X merges main, keeps its own version lines, and lands.
    _git(repo, "checkout", "-q", "pr-x")
    _git(repo, "merge", "-q", "-X", "ours", "--no-edit", "main")
    _git(repo, "checkout", "-q", "main")
    _git(
        repo,
        "merge",
        "-q",
        "--no-ff",
        "pr-x",
        "-m",
        "Merge pull request #9 from org/pr-x",
    )
    assert version_provenance.versions_at(repo, "HEAD")["pyproject"] == "0.2.11"
    assert version_provenance.bump(repo) not in {"0.2.11", "0.2.12"}


def test_two_encoder_trees_never_share_a_version(repo: Path):
    (repo / "src/axiom_encode/d.py").write_text("D = 1\n")
    unversioned = _commit(repo, "PR D lands with no bump")
    # A PR branched from the unversioned main bumps by hand and adds a dependency.
    _git(repo, "checkout", "-qb", "pr-x", unversioned)
    (repo / "src/axiom_encode/x.py").write_text("X = 1\n")
    _write_version(repo, "0.2.11")
    with (repo / "uv.lock").open("a") as lock:
        lock.write('\n[[package]]\nname = "requests"\nversion = "2.32.5"\n')
    _commit(repo, "PR X: encoder change, dependency and hand bump to 0.2.11")
    _git(repo, "checkout", "-q", "main")
    assert version_provenance.bump(repo) == "0.2.11"
    bot = _commit(repo, "Bump encoder version to 0.2.11")
    _git(
        repo,
        "merge",
        "-q",
        "--no-ff",
        "pr-x",
        "-m",
        "Merge pull request #9 from org/pr-x",
    )
    landed = _git(repo, "rev-parse", "HEAD")
    assert version_provenance.versions_at(repo, landed)["pyproject"] == "0.2.11"
    assert _git(repo, "rev-parse", f"{bot}:src/axiom_encode") != _git(
        repo, "rev-parse", f"{landed}:src/axiom_encode"
    )
    assert version_provenance.unversioned_changes(repo, landed)


def test_a_version_file_edit_that_keeps_the_version_is_not_a_bump(repo: Path):
    _encoder_and_dependency_change(repo)
    assert version_provenance.unversioned_changes(repo) == [
        "pyproject.toml",
        "src/axiom_encode/cli.py",
    ]


def test_ambient_git_dir_cannot_move_head(
    repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    versioned = _git(repo, "rev-parse", "HEAD")
    _change_encoder(repo, 1)
    copy = tmp_path / "copy"
    shutil.copytree(repo, copy)
    _git(copy, "checkout", "-q", "--detach", versioned)
    monkeypatch.setenv("GIT_DIR", str(copy / ".git"))
    assert version_provenance.unversioned_changes(repo) == ["src/axiom_encode/cli.py"]


# --- The pull-request check: only the bump App picks versions ----------------


def _bump_workflow_on(repo: Path) -> str:
    (repo / ".github/workflows").mkdir(parents=True, exist_ok=True)
    (repo / version_provenance.BUMP_WORKFLOW).write_text("name: bump\n")
    return _commit(repo, "add the bump workflow")


def _hand_bump(repo: Path, base: str, version: str) -> str:
    _git(repo, "checkout", "-qb", f"hand-{version}", base)
    (repo / "src/axiom_encode/x.py").write_text("X = 1\n")
    _write_version(repo, version)
    return _commit(repo, f"hand bump to {version}")


def _reason(repo: Path, base: str, ref: str, author: str, head_repo: str = "o/r"):
    return version_provenance.hand_picked_version(
        repo, base, ref, author=author, head_repository=head_repo, repository="o/r"
    )


def test_a_hand_bump_is_refused_once_the_bump_workflow_is_on_main(repo: Path):
    base = _bump_workflow_on(repo)
    head = _hand_bump(repo, base, "0.2.11")
    assert "assigned after merge" in _reason(repo, base, head, "someone")
    # A fork's pull request is never the App's, whatever its author name says.
    assert _reason(repo, base, head, version_provenance.BUMP_APP_LOGIN, "fork/r")
    assert _reason(repo, base, head, version_provenance.BUMP_APP_LOGIN) is None


@pytest.mark.parametrize(
    ("path", "old", "new"),
    [
        ("src/axiom_encode/__init__.py", '"0.2.10"', '"0.2.11"'),
        ("uv.lock", 'version = "0.2.10"', 'version = "0.2.11"'),
        ("pyproject.toml", 'version = "0.2.10"', 'version = "0.2.11"'),
    ],
)
def test_a_hand_bump_of_any_one_version_file_is_refused(
    repo: Path, path: str, old: str, new: str
):
    base = _bump_workflow_on(repo)
    _git(repo, "checkout", "-qb", "one-file", base)
    target = repo / path
    target.write_text(target.read_text().replace(old, new, 1))
    head = _commit(repo, f"hand bump of {path} alone")
    assert "assigned after merge" in _reason(repo, base, head, "someone")


def test_the_apps_login_as_github_reports_it_is_the_app(repo: Path):
    # A pull request's user.login for a GitHub App is "<slug>[bot]"; gh prints
    # "app/<slug>" and GraphQL the bare slug, and neither is the event's login.
    base = _bump_workflow_on(repo)
    head = _hand_bump(repo, base, "0.2.11")
    assert _reason(repo, base, head, "axiom-encoder-version-bump[bot]") is None
    for other in ("app/axiom-encoder-version-bump", "axiom-encoder-version-bump"):
        assert _reason(repo, base, head, other)


def test_a_pull_request_that_keeps_the_version_passes(repo: Path):
    base = _bump_workflow_on(repo)
    _git(repo, "checkout", "-qb", "feature", base)
    (repo / "src/axiom_encode/x.py").write_text("X = 1\n")
    head = _commit(repo, "encoder change, no bump")
    assert _reason(repo, base, head, "someone") is None


def test_hand_bumps_still_pass_before_the_bump_workflow_lands(repo: Path):
    base = _git(repo, "rev-parse", "HEAD")
    head = _hand_bump(repo, base, "0.2.11")
    assert _reason(repo, base, head, "someone") is None


def test_the_pull_request_command_exits_1_on_a_hand_bump(repo: Path):
    base = _bump_workflow_on(repo)
    head = _hand_bump(repo, base, "0.2.11")
    common = ["--repo", str(repo), "--base", base, "--ref", head, "--repository", "o/r"]
    refused = _run_script(
        "pull-request", *common, "--author", "someone", "--head-repository", "o/r"
    )
    assert refused.returncode == 1
    assert "assigned after merge" in refused.stderr
    allowed = _run_script(
        "pull-request",
        *common,
        "--author",
        version_provenance.BUMP_APP_LOGIN,
        "--head-repository",
        "o/r",
    )
    assert allowed.returncode == 0, allowed.stderr


def test_ci_runs_the_pull_request_check_in_a_required_job():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    lint = workflow["jobs"]["lint"]
    assert lint["steps"][0]["with"]["fetch-depth"] == 2
    step = next(
        s
        for s in lint["steps"]
        if s.get("name") == "Refuse a hand-picked encoder version"
    )
    assert step["if"] == "github.event_name == 'pull_request'"
    assert "version_provenance.py pull-request" in step["run"]
    assert "--base HEAD^1" in step["run"]
    assert step["env"]["PR_AUTHOR"] == "${{ github.event.pull_request.user.login }}"


def _lint_step_checkout(repo: Path) -> Path:
    """A pull_request checkout: HEAD merges a hand bump into a base whose main
    has the bump workflow, with the step's ``.venv/bin/python`` in place."""
    base = _bump_workflow_on(repo)
    branch = _hand_bump(repo, base, "0.2.11")
    _git(repo, "checkout", "-q", "--detach", base)
    _git(repo, "merge", "-q", "--no-ff", branch, "-m", f"Merge {branch} into {base}")
    (repo / "src/axiom_encode/version_provenance.py").write_text(SCRIPT.read_text())
    venv = repo / ".venv/bin"
    venv.mkdir(parents=True)
    (venv / "python").symlink_to(sys.executable)
    return repo


def _github_expression(value: str, context: dict) -> str:
    """The value of an env entry that is one ``${{ a.b.c }}`` property lookup."""
    match = re.fullmatch(r"\$\{\{\s*([A-Za-z_][\w.]*)\s*\}\}", value)
    assert match, f"not a plain property lookup: {value}"
    for part in match.group(1).split("."):
        context = context[part]
    return context


@pytest.mark.parametrize(
    ("author", "head_repository", "refused"),
    [
        ("someone", "o/r", True),
        ("axiom-encoder-version-bump[bot]", "o/r", False),
        # A fork's pull request whose author reads as the App's login.
        ("axiom-encoder-version-bump[bot]", "mallory/r", True),
    ],
)
def test_the_lint_step_refuses_a_hand_bump_as_github_runs_it(
    repo: Path, author: str, head_repository: str, refused: bool
):
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    step = next(
        s
        for s in workflow["jobs"]["lint"]["steps"]
        if s.get("name") == "Refuse a hand-picked encoder version"
    )
    lint = workflow["jobs"]["lint"]
    assert "shell" not in step and "defaults" not in lint
    assert "defaults" not in workflow
    # A failing step that may fail would let the required job pass.
    assert "continue-on-error" not in step and "continue-on-error" not in lint
    checkout = _lint_step_checkout(repo)
    context = {
        "github": {
            "repository": "o/r",
            "event": {
                "pull_request": {
                    "user": {"login": author},
                    "head": {"repo": {"full_name": head_repository}},
                    "base": {"repo": {"full_name": "o/r"}},
                }
            },
        }
    }
    env = {name: _github_expression(v, context) for name, v in step["env"].items()}
    # With no `shell`, GitHub runs a step on Linux as `bash -e {0}`.
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=checkout,
        env={"PATH": os.environ["PATH"], "GITHUB_REPOSITORY": "o/r", **env},
        capture_output=True,
        text=True,
    )
    assert (result.returncode != 0) is refused, result.stderr
    if refused:
        assert "encoder version provenance" in result.stderr


def test_history_rewrites_are_off_in_every_copy_of_the_rule():
    # No test forges a commit-graph file, so pin the settings that turn the
    # commit graph, replace refs and grafts off.
    environment = version_provenance._git_environment()
    keys = {
        environment[f"GIT_CONFIG_KEY_{i}"]: environment[f"GIT_CONFIG_VALUE_{i}"]
        for i in range(int(environment["GIT_CONFIG_COUNT"]))
    }
    assert keys["core.commitGraph"] == "false"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert environment["GIT_GRAFT_FILE"] == os.devnull
    flags = provisioner._HARDENED_GIT_FLAGS
    assert "core.commitGraph=false" in flags
    assert (
        provisioner._hardened_git_environment(Path("/nonexistent"))["GIT_GRAFT_FILE"]
        == os.devnull
    )


# --- All three copies apply the two first-parent rules ------------------------


def _land_lowered_version(repo: Path) -> None:
    """Main bumps to 0.2.12; a merge then keeps a branch's hand bump to 0.2.11."""
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-qb", "pr-x", base)
    (repo / "src/axiom_encode/x.py").write_text("X = 1\n")
    _write_version(repo, "0.2.11")
    _commit(repo, "PR X: encoder change and hand bump to 0.2.11")
    _git(repo, "checkout", "-q", "main")
    _write_version(repo, "0.2.12")
    _commit(repo, "main bumps to 0.2.12")
    _git(repo, "merge", "-q", "-X", "theirs", "--no-ff", "pr-x", "-m", "merge pr-x")


def test_a_head_below_a_version_main_declared_needs_a_bump_in_every_copy(
    repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _land_lowered_version(repo)
    head = _git(repo, "rev-parse", "HEAD")
    assert version_provenance.versions_at(repo, head)["pyproject"] == "0.2.11"
    # Only rule 1 decides here: the merge is where main reached 0.2.11 and
    # no encoder file changed after the branch's own bump.
    bump = version_provenance.latest_version_commit(repo, head)
    assert version_provenance._changed_since(repo, bump, head) == []
    history = version_provenance.first_parent_versions(repo, head)
    assert version_provenance._first_parent_arrival(history, "0.2.11") == head
    assert version_provenance.unversioned_changes(repo) == sorted(
        version_provenance.VERSION_FILES
    )
    assert _verdicts(repo, monkeypatch, tmp_path / "home") == {True}
    assert version_provenance.bump(repo) == "0.2.13"


def test_two_encoder_trees_on_one_version_need_a_bump_in_every_copy(
    repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    (repo / "src/axiom_encode/d.py").write_text("D = 1\n")
    unversioned = _commit(repo, "PR D lands with no bump")
    _git(repo, "checkout", "-qb", "pr-x", unversioned)
    (repo / "src/axiom_encode/x.py").write_text("X = 1\n")
    _write_version(repo, "0.2.11")
    with (repo / "uv.lock").open("a") as lock:
        lock.write('\n[[package]]\nname = "requests"\nversion = "2.32.5"\n')
    _commit(repo, "PR X: encoder change, dependency and hand bump to 0.2.11")
    _git(repo, "checkout", "-q", "main")
    version_provenance.bump(repo)
    _commit(repo, "Bump encoder version to 0.2.11")
    _git(repo, "merge", "-q", "--no-ff", "pr-x", "-m", "merge pr-x")

    assert _verdicts(repo, monkeypatch, tmp_path / "home") == {True}


def test_cat_file_output_after_a_non_blob_is_still_read(repo: Path):
    # pyproject.toml as a directory at one commit: its tree content must be
    # skipped, so the next commit's version still reads right.
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "pyproject.toml").unlink()
    (repo / "pyproject.toml").mkdir()
    (repo / "pyproject.toml/x").write_text("x\n")
    odd = _commit(repo, "pyproject.toml as a directory")
    assert version_provenance._pyproject_versions(repo, [odd, first, "f" * 40]) == [
        None,
        "0.2.10",
        None,
    ]


def test_a_tree_named_pyproject_toml_declares_no_version_in_either_copy(
    repo: Path, tmp_path: Path
):
    # A tree's raw entries hold its names. One name carries a version line, so
    # reading the tree as text would find "99.0"; both copies must skip it.
    (repo / "pyproject.toml").unlink()
    (repo / "pyproject.toml").mkdir()
    (repo / "pyproject.toml" / 'x\nversion = "99.0"\n').write_text("x\n")
    odd = _commit(repo, "pyproject.toml as a directory with a version-line name")
    shutil.rmtree(repo / "pyproject.toml")
    _write_version(repo, "0.2.10")
    head = _commit(repo, "pyproject.toml back as a file")
    raw = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "tree", f"{odd}:pyproject.toml"],
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8", "replace")
    assert version_provenance._first(version_provenance._PYPROJECT_VERSION, raw)
    expected = [(head, "0.2.10"), (odd, None)]
    assert version_provenance.first_parent_versions(repo, head)[:2] == expected
    history = provisioner._first_parent_pyproject_versions(
        Path(shutil.which("git")), repo / ".git", tmp_path / "home", head
    )
    assert history[:2] == expected


# --- axiom-encode#2051 approving review: the follow-up findings --------------


def test_a_non_numeric_head_gets_one_verdict(
    repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _write_version(repo, "0.2.11rc1")
    _commit(repo, "bump to a pre-release")
    assert _verdicts(repo, monkeypatch, tmp_path / "home") == {True}


def test_a_version_main_left_and_declared_again_needs_a_bump(
    repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    (repo / "src/axiom_encode/x.py").write_text("X = 1\n")
    _commit(repo, "encoder change")
    _write_version(repo, "0.2.9")
    _commit(repo, "lower the version")
    _write_version(repo, "0.2.10")
    _commit(repo, "hand bump back to 0.2.10")
    assert _verdicts(repo, monkeypatch, tmp_path / "home") == {True}


def test_a_merge_that_restores_an_older_bumps_files_gets_one_verdict(
    repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    # Main: A (0.2.10), B bumps to 0.2.11, D adds a dependency. A side branch
    # from A bumps by hand to 0.2.12 with a later date, so `git log` lists it
    # before B. The merge keeps 0.2.11 and drops D's dependency: every copy
    # takes B, the bump to the head's version, and sees no change.
    a = _git(repo, "rev-parse", "HEAD")

    def dated(day: int) -> None:
        stamp = f"2030-01-0{day}T00:00:00+00:00"
        monkeypatch.setenv("GIT_AUTHOR_DATE", stamp)
        monkeypatch.setenv("GIT_COMMITTER_DATE", stamp)

    dated(1)
    _write_version(repo, "0.2.11")
    b = _commit(repo, "B: bump to 0.2.11")
    dated(2)
    with (repo / "uv.lock").open("a") as lock:
        lock.write('\n[[package]]\nname = "requests"\nversion = "2.32.5"\n')
    _commit(repo, "D: add a dependency")
    dated(3)
    _git(repo, "checkout", "-qb", "side", a)
    _write_version(repo, "0.2.12")
    side = _commit(repo, "side: hand bump to 0.2.12")
    _git(repo, "checkout", "-q", "main")
    dated(4)
    _git(repo, "merge", "-q", "--no-ff", "--no-commit", "-s", "ours", "side")
    _git(repo, "checkout", b, "--", "uv.lock")
    _commit(repo, "merge side, keep 0.2.11, drop D's dependency")
    listed = _git(
        repo, "log", "--format=%H", "--", *version_provenance.VERSION_FILES
    ).split()
    assert listed.index(side) < listed.index(b)
    assert version_provenance.latest_version_commit(repo) == b
    assert _verdicts(repo, monkeypatch, tmp_path / "home") == {False}


def test_a_version_file_edit_that_keeps_the_version_is_not_the_latest_bump(
    repo: Path,
):
    bump = _git(repo, "rev-parse", "HEAD")
    _encoder_and_dependency_change(repo)
    assert version_provenance.latest_version_commit(repo) == bump


@pytest.mark.parametrize("base", ["HEAD^3", "f" * 40, "--all"])
def test_a_base_that_does_not_resolve_is_refused(repo: Path, base: str):
    _bump_workflow_on(repo)
    head = _hand_bump(repo, _git(repo, "rev-parse", "HEAD"), "0.2.11")
    with pytest.raises(ValueError, match="does not name a commit"):
        _reason(repo, base, head, "someone")
    result = _run_script(
        "pull-request",
        "--repo",
        str(repo),
        f"--base={base}",
        "--ref",
        head,
        "--author",
        "someone",
        "--head-repository",
        "o/r",
        "--repository",
        "o/r",
    )
    assert result.returncode == 1
    assert "does not name a commit" in result.stderr


def test_the_apply_guard_reads_main_history_in_its_own_environment(
    repo: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(cli, "__version__", "0.2.10")
    monkeypatch.setenv("CANARY_SECRET", "x")
    seen: list[dict[str, str] | None] = []
    real = version_provenance.main_history_changes

    def spy(*args, environment=None, **kwargs):
        seen.append(environment)
        return real(*args, environment=environment, **kwargs)

    monkeypatch.setattr(version_provenance, "main_history_changes", spy)
    cli._require_axiom_encode_version_provenance(repo)
    assert seen and seen[0] is not None
    assert seen[0] == cli._read_only_git_environment()


def test_every_history_read_runs_under_the_callers_environment(
    repo: Path, tmp_path: Path
):
    # The guard passes its own scrubbed environment. Point that environment's
    # GIT_DIR at a clone with more history: each read must see the clone, so
    # every git call took the environment it was given.
    first = _git(repo, "rev-parse", "HEAD")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(repo), str(clone))
    for key, value in (
        ("user.email", "test@example.com"),
        ("user.name", "Test"),
        ("commit.gpgsign", "false"),
    ):
        _git(clone, "config", key, value)
    _write_version(clone, "0.2.11")
    bump = _commit(clone, "bump to 0.2.11")
    _change_encoder(clone, 1)
    environment = {**cli._read_only_git_environment(), "GIT_DIR": str(clone / ".git")}

    assert version_provenance.first_parent_versions(repo, "HEAD", environment) == [
        (bump, "0.2.11"),
        (first, "0.2.10"),
    ]
    assert set(version_provenance._changed_since(repo, bump, "HEAD", environment)) == {
        "src/axiom_encode/cli.py"
    }
    # Main reached 0.2.11 at `bump`, not at `first`: rule 2 reads the clone.
    assert version_provenance.main_history_changes(
        repo, "HEAD", first, "0.2.11", environment=environment
    ) == {"src/axiom_encode/cli.py"}
    assert (
        version_provenance.main_history_changes(repo, "HEAD", first, "0.2.10") == set()
    )


def test_a_first_parent_commit_that_deletes_pyproject_toml_declares_no_version(
    repo: Path, tmp_path: Path
):
    # `git log -- pyproject.toml` lists the deletion, and cat-file reports the
    # file missing there. Both history readers must read that as no version.
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "pyproject.toml").unlink()
    deleted = _commit(repo, "delete pyproject.toml")
    _write_version(repo, "0.2.10")
    restored = _commit(repo, "restore pyproject.toml")
    expected = [(restored, "0.2.10"), (deleted, None), (first, "0.2.10")]
    assert version_provenance.first_parent_versions(repo, restored) == expected
    history = provisioner._first_parent_pyproject_versions(
        Path(shutil.which("git")), repo / ".git", tmp_path / "home", restored
    )
    assert history == expected


def test_signatures_stay_off_in_every_copy_of_the_rule():
    for environment in (
        version_provenance._git_environment(),
        cli._read_only_git_environment(),
    ):
        keys = {
            environment[f"GIT_CONFIG_KEY_{i}"]: environment[f"GIT_CONFIG_VALUE_{i}"]
            for i in range(int(environment["GIT_CONFIG_COUNT"]))
        }
        assert keys["log.showSignature"] == "false"
    assert "log.showSignature=false" in provisioner._HARDENED_GIT_FLAGS


def test_a_git_failure_in_main_history_is_the_guards_refusal(
    repo: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(cli, "__version__", "0.2.10")

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(128, ["git", "log"])

    monkeypatch.setattr(version_provenance, "main_history_changes", fail)
    with pytest.raises(RuntimeError, match="first-parent encoder version history"):
        cli._require_axiom_encode_version_provenance(repo)
