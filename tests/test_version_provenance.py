import importlib.util
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from axiom_encode import cli, version_provenance
from axiom_encode.ci_parity import encoder_version_at_pin
from axiom_encode.cli import _latest_axiom_encode_version_commit

SCRIPT = Path(version_provenance.__file__)


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
