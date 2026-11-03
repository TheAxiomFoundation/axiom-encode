import subprocess
import sys
from pathlib import Path

import pytest

from axiom_encode import version_provenance
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
