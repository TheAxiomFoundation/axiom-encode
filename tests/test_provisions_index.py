"""The derived reverse index regenerated at publish, byte-for-byte."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from axiom_encode import provisions_index as pi
from axiom_encode.cli import (
    RULESPEC_ATOMIC_MODULE_ROOTS,
    _is_applied_encoding_manifest_path,
    _is_protected_rulespec_yaml_path,
)
from scripts.prepare_signed_backfill import main as prepare_signed_backfill_main

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_FIXTURE = (
    ROOT / "tests/fixtures/rulespec_reverse_index/generate_reverse_index.py.txt"
)
LANE_BRANCH = "axiom/signed-backfill-us-123-1"


def _generator() -> Any:
    """rulespec-us tests/generate_reverse_index.py, vendored byte for byte."""

    loader = importlib.machinery.SourceFileLoader(
        "rulespec_generate_reverse_index", str(GENERATOR_FIXTURE)
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _generator_output(root: Path) -> bytes:
    generator = _generator()
    return generator.render_index(generator.build_index(root.resolve())).encode()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _write(root: Path, relative: str, text: str | bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def _module(*citations: str, atoms: tuple[str, ...] = (), plural: bool = False) -> str:
    verification: dict[str, Any] = {}
    if citations:
        if plural:
            verification["corpus_citation_paths"] = list(citations)
        else:
            verification["corpus_citation_path"] = citations[0]
    payload: dict[str, Any] = {
        "format": "rulespec/v1",
        "module": {"source_verification": verification},
        "rules": [
            {
                "name": f"rule_{index}",
                "metadata": {
                    "proof": {"atoms": [{"source": {"corpus_citation_path": atom}}]}
                },
            }
            for index, atom in enumerate(atoms)
        ],
    }
    return yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)


def _tree(root: Path) -> None:
    """A rulespec-shaped tree exercising every generator branch."""

    _write(root, "tests/generate_reverse_index.py", GENERATOR_FIXTURE.read_bytes())
    _write(root, "README.md", "x\n")
    _write(
        root,
        "us/statutes/7/2015/f.yaml",
        _module("us/statute/7/2015/f", atoms=("us/statute/7/2014/e", " us/x ")),
    )
    _write(root, "us/statutes/7/2015/f.test.yaml", _module("us/never/test"))
    _write(root, "us/statutes/7/2014/e.yml", _module("us/statute/7/2014/e"))
    _write(
        root,
        "us/regulations/7-cfr/273/4.yaml",
        _module("us/regulation/7/273/4", "us/regulation/7/273/2", plural=True),
    )
    _write(root, "us/regulations/bad.yaml", "rules: [unclosed\n")
    _write(root, "us/regulations/list.yaml", "- just\n- a list\n")
    _write(root, "us/regulations/blank.yaml", _module("   ", atoms=("",)))
    _write(
        root,
        "us/regulations/types.yaml",
        "module:\n  source_verification:\n    corpus_citation_path: 7\n"
        "    corpus_citation_paths: [x/1, null, 3]\n",
    )
    _write(
        root,
        "us/policies/crlf.yaml",
        _module("us/guidance/crlf").replace("\n", "\r\n").encode("utf-8"),
    )
    _write(root, "us/policies/unicode.yaml", _module("us/guidance/münchen-§1"))
    _write(root, "us/policies/.hidden.yaml", _module("us/guidance/hidden"))
    # A key spelled only through escapes still counts; the prefilter must see it.
    _write(
        root,
        "us/policies/escaped.yaml",
        'module:\n  source_verification:\n    "corpus\\u005Fcitation_path": '
        "us/guidance/escaped\n",
    )
    _write(
        root,
        "us/policies/table.yaml",
        "rules:\n  - name: table\n    values: [1, 2, 3]\n",
    )
    _write(root, "us/policies/dir.yaml/inner.yaml", _module("us/guidance/inner"))
    _write(root, "us-ca/statutes/1.yaml", _module("us-ca/statute/1"))
    _write(root, "uk-ashford/policies/1.yaml", _module("us/statute/7/2015/f"))
    # (An upper-case "US/" root is not portable: macOS folds it into "us/".)
    _write(root, "Ux-tx/statutes/1.yaml", _module("ignored/upper"))
    _write(root, "u/statutes/1.yaml", _module("ignored/short"))
    _write(root, "us_/statutes/1.yaml", _module("ignored/underscore"))
    _write(root, "programs/snap.yaml", _module("ignored/programs"))
    _write(root, ".axiom/index/provisions_to_rules.json", '{"stale": true}\n')
    (root / "us/statutes/7/2014/e.yml").chmod(0o755)


def _commit_tree(root: Path) -> str:
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "tree")
    return _git(root, "rev-parse", "HEAD")


def test_vendored_generator_is_a_reproduced_version() -> None:
    raw = GENERATOR_FIXTURE.read_bytes()
    blob = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
    assert blob in pi.SUPPORTED_GENERATOR_BLOBS
    generator = _generator()
    assert generator.INDEX_RELATIVE_PATH.as_posix() == pi.INDEX_PATH.as_posix()


def test_replica_matches_the_generator_byte_for_byte(tmp_path: Path) -> None:
    _tree(tmp_path)
    head = _commit_tree(tmp_path)

    expected = _generator_output(tmp_path)

    assert pi.regenerate_from_directory(tmp_path) == expected
    assert pi.regenerate_from_directory(tmp_path, prefilter=False) == expected
    assert pi.regenerate_from_commit(tmp_path, head) == expected
    assert pi.regenerate_from_commit(tmp_path, head, prefilter=False) == expected
    index = json.loads(expected)
    assert "counts" not in index
    assert set(index["provisions"]) == {
        "us-ca/statute/1",
        "us/guidance/crlf",
        "us/guidance/escaped",
        "us/guidance/hidden",
        "us/guidance/inner",
        "us/guidance/münchen-§1",
        "us/regulation/7/273/2",
        "us/regulation/7/273/4",
        "us/statute/7/2014/e",
        "us/statute/7/2015/f",
        "us/x",
        "x/1",
    }
    assert index["provisions"]["us/statute/7/2014/e"] == [
        {"module": "us/statutes/7/2014/e.yml", "via": ["module"]},
        {"module": "us/statutes/7/2015/f.yaml", "via": ["proof_atom"]},
    ]
    assert "münchen" in expected.decode("utf-8")


def test_the_prefilter_only_skips_modules_without_key_text() -> None:
    assert pi.may_reference_citations("x:\n  corpus_citation_paths: [a]\n")
    assert pi.may_reference_citations('"corpus\\u005Fcitation_path": a\n')
    assert not pi.may_reference_citations("rules:\n  - name: table\n")
    assert not pi.may_reference_citations("corpus citation path: a\n")


def test_an_empty_tree_matches_the_generator(tmp_path: Path) -> None:
    _write(tmp_path, "tests/generate_reverse_index.py", GENERATOR_FIXTURE.read_bytes())
    _write(tmp_path, "README.md", "x\n")
    head = _commit_tree(tmp_path)

    assert pi.regenerate_from_commit(tmp_path, head) == _generator_output(tmp_path)


def test_an_unknown_generator_version_is_not_reproduced(tmp_path: Path) -> None:
    _tree(tmp_path)
    _write(
        tmp_path,
        "tests/generate_reverse_index.py",
        GENERATOR_FIXTURE.read_text() + "\n# changed\n",
    )
    head = _commit_tree(tmp_path)

    with pytest.raises(pi.UnsupportedIndexTree, match="not a reproduced version"):
        pi.regenerate_from_commit(tmp_path, head)


def test_a_symlinked_module_is_not_reproduced(tmp_path: Path) -> None:
    _tree(tmp_path)
    (tmp_path / "us/statutes/link.yaml").symlink_to("7/2015/f.yaml")
    head = _commit_tree(tmp_path)

    with pytest.raises(pi.UnsupportedIndexTree, match="symlinked RuleSpec module"):
        pi.regenerate_from_commit(tmp_path, head)


def test_a_symlinked_jurisdiction_root_is_not_reproduced(tmp_path: Path) -> None:
    _tree(tmp_path)
    (tmp_path / "us-tx").symlink_to("us")
    head = _commit_tree(tmp_path)

    with pytest.raises(pi.UnsupportedIndexTree, match="jurisdiction root"):
        pi.regenerate_from_commit(tmp_path, head)


def test_a_non_utf8_module_is_not_reproduced(tmp_path: Path) -> None:
    _tree(tmp_path)
    _write(tmp_path, "us/statutes/latin1.yaml", "x: caf\xe9\n".encode("latin-1"))
    head = _commit_tree(tmp_path)

    with pytest.raises(pi.UnsupportedIndexTree, match="not UTF-8"):
        pi.regenerate_from_commit(tmp_path, head)
    with pytest.raises(UnicodeDecodeError):
        _generator_output(tmp_path)  # the generator itself would crash


def test_a_loader_crash_is_not_reproduced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tree(tmp_path)
    head = _commit_tree(tmp_path)

    def boom(_text: str) -> None:
        raise RecursionError("deep")

    monkeypatch.setattr(pi.yaml, "safe_load", boom)

    with pytest.raises(pi.UnsupportedIndexTree, match="cannot be loaded"):
        pi.regenerate_from_commit(tmp_path, head)


def test_the_guard_treats_the_index_as_derived_metadata() -> None:
    roots = tuple(sorted(RULESPEC_ATOMIC_MODULE_ROOTS))
    index = Path(pi.INDEX_PATH.as_posix())

    assert not _is_protected_rulespec_yaml_path(index, roots=roots)
    assert not _is_applied_encoding_manifest_path(index, roots=roots)


# --- refresh_lane_index -----------------------------------------------------------


def _lane_repo(tmp_path: Path, *, index: str | None = '{"stale": true}\n') -> tuple:
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    _tree(repo)
    if index is None:
        (repo / ".axiom/index/provisions_to_rules.json").unlink()
    else:
        _write(repo, ".axiom/index/provisions_to_rules.json", index)
    base = _commit_tree(repo)
    _git(repo, "switch", "-q", "-c", LANE_BRANCH)
    _write(repo, "us/statutes/7/2015/new.yaml", _module("us/statute/7/2015/new"))
    _write(repo, ".axiom/encoding-manifests/us/statutes/7/2015/new.json", "{}\n")
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.name=axiom-encode-bot",
        "-c",
        "user.email=axiom-encode-bot@users.noreply.github.com",
        "commit",
        "-q",
        "-m",
        "Add signed encoding manifest for us/statute/7/2015/new",
    )
    return repo, base, _git(repo, "rev-parse", "HEAD")


def test_a_stale_index_gets_one_unsigned_derived_commit(tmp_path: Path) -> None:
    repo, base, head = _lane_repo(tmp_path)

    evidence = pi.refresh_lane_index(repo, base)

    refreshed = _git(repo, "rev-parse", "HEAD")
    assert evidence["mode"] == "refreshed"
    assert evidence["head_before"] == head
    assert evidence["head_after"] == refreshed
    assert _git(repo, "rev-parse", f"{refreshed}^") == head
    assert _git(repo, "diff", "--name-only", head, refreshed).splitlines() == [
        pi.INDEX_PATH.as_posix()
    ]
    assert _git(repo, "log", "-1", "--format=%s", refreshed) == (
        pi.REFRESH_COMMIT_SUBJECT
    )
    assert "not signed" in _git(repo, "log", "-1", "--format=%b", refreshed)
    assert _git(repo, "log", "-1", "--format=%an <%ae>|%cn <%ce>", refreshed) == (
        "axiom-encode-bot <axiom-encode-bot@users.noreply.github.com>|"
        "axiom-encode-bot <axiom-encode-bot@users.noreply.github.com>"
    )
    assert _git(repo, "symbolic-ref", "HEAD") == f"refs/heads/{LANE_BRANCH}"
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""
    committed = (repo / pi.INDEX_PATH).read_bytes()
    assert committed == _generator_output(repo)
    assert "us/statute/7/2015/new" in json.loads(committed)["provisions"]
    assert evidence["index_blob_after"] == _git(
        repo, "rev-parse", f"{refreshed}:{pi.INDEX_PATH.as_posix()}"
    )

    # A second refresh finds it current and adds nothing.
    again = pi.refresh_lane_index(repo, base)
    assert again["mode"] == "current"
    assert _git(repo, "rev-parse", "HEAD") == refreshed


def test_a_current_index_adds_no_commit(tmp_path: Path) -> None:
    repo, base, head = _lane_repo(tmp_path)
    _write(repo, pi.INDEX_PATH.as_posix(), _generator_output(repo))
    _git(repo, "commit", "-q", "-am", "lane already current")
    head = _git(repo, "rev-parse", "HEAD")

    evidence = pi.refresh_lane_index(repo, base)

    assert evidence["mode"] == "skipped-signed"  # the lane itself changed it
    assert _git(repo, "rev-parse", "HEAD") == head


def test_an_untracked_index_is_left_absent(tmp_path: Path) -> None:
    repo, base, head = _lane_repo(tmp_path, index=None)

    evidence = pi.refresh_lane_index(repo, base)

    assert evidence["mode"] == "skipped-absent"
    assert _git(repo, "rev-parse", "HEAD") == head
    assert not (repo / pi.INDEX_PATH).exists()


@pytest.mark.parametrize(
    "receipt",
    [
        ".axiom/legacy-replacements/" + "a" * 64 + ".json",
        ".axiom/path-migrations/receipt.json",
        ".axiom/retired-source-metadata-migrations/receipt.json",
    ],
)
def test_receipt_bound_lanes_are_never_extended(tmp_path: Path, receipt: str) -> None:
    repo, base, _head = _lane_repo(tmp_path)
    _write(repo, receipt, "{}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "receipt")
    head = _git(repo, "rev-parse", "HEAD")

    evidence = pi.refresh_lane_index(repo, base)

    assert evidence["mode"] == "skipped-receipt"
    assert receipt in evidence["reason"]
    assert _git(repo, "rev-parse", "HEAD") == head


def test_an_unsupported_tree_is_published_unchanged(tmp_path: Path) -> None:
    repo, base, _head = _lane_repo(tmp_path)
    _write(repo, "tests/generate_reverse_index.py", "print('newer generator')\n")
    _git(repo, "commit", "-q", "-am", "generator changed on the lane base")
    head = _git(repo, "rev-parse", "HEAD")

    evidence = pi.refresh_lane_index(repo, base)

    assert evidence["mode"] == "skipped-unsupported"
    assert "reproduced version" in evidence["reason"]
    assert _git(repo, "rev-parse", "HEAD") == head


def test_refresh_requires_a_clean_lane_branch(tmp_path: Path) -> None:
    repo, base, head = _lane_repo(tmp_path)
    _write(repo, "stray.txt", "x\n")
    with pytest.raises(ValueError, match="clean lane checkout"):
        pi.refresh_lane_index(repo, base)
    (repo / "stray.txt").unlink()
    _git(repo, "checkout", "-q", "--detach", head)
    with pytest.raises(subprocess.CalledProcessError):
        pi.refresh_lane_index(repo, base)
    _git(repo, "switch", "-q", "-c", "not-a-lane")
    with pytest.raises(ValueError, match="signed lane branch"):
        pi.refresh_lane_index(repo, base)


def test_refresh_requires_lane_commits_on_the_base(tmp_path: Path) -> None:
    repo, _base, head = _lane_repo(tmp_path)
    with pytest.raises(ValueError, match="no signed lane commits"):
        pi.refresh_lane_index(repo, head)


def test_cli_emits_refresh_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, base, _head = _lane_repo(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_signed_backfill.py", "refresh-provisions-index", str(repo), base],
    )

    prepare_signed_backfill_main()

    evidence = json.loads(capsys.readouterr().out)
    assert evidence["mode"] == "refreshed"
    assert evidence["head_after"] == _git(repo, "rev-parse", "HEAD")


def test_cli_rejects_a_malformed_lane_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, _base, _head = _lane_repo(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_signed_backfill.py", "refresh-provisions-index", str(repo), "HEAD~1"],
    )

    with pytest.raises(SystemExit):
        prepare_signed_backfill_main()
    assert "full lowercase commit SHA" in capsys.readouterr().err


# --- the workflow publish step, end to end --------------------------------------

GH_STUB = r"""#!/usr/bin/env bash
set -euo pipefail
if [ "$1" = "auth" ]; then
  exit 0
fi
if [ "$1 $2 $3" = "api --method POST" ]; then
  shift 4
  base=""
  head=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -f)
        case "$2" in
          base=*) base="${2#base=}" ;;
          head=*) head="${2#head=}" ;;
          body=*) printf '%s' "${2#body=}" > "$GH_STUB_BODY" ;;
        esac
        shift 2
        ;;
      *) shift ;;
    esac
  done
  branch="${head#TheAxiomFoundation:}"
  base_sha="$(git --git-dir "$GH_STUB_REMOTE" rev-parse "refs/heads/$base")"
  head_sha="$(git --git-dir "$GH_STUB_REMOTE" rev-parse "refs/heads/$branch")"
  printf '{"number": 7, "base": {"ref": "%s", "sha": "%s"}, "head": {"sha": "%s"}}\n' \
    "$base" "$base_sha" "$head_sha"
  exit 0
fi
exit 1
"""


def _publish_step() -> dict[str, Any]:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    return next(
        step
        for step in workflow["jobs"]["encode"]["steps"]
        if step.get("id") == "publish_lane_pull_request"
    )


def _publish_fixture(tmp_path: Path, *, python: str) -> dict[str, Any]:
    seed_root = tmp_path / "seed"
    seed_root.mkdir()
    seed, base, _head = _lane_repo(seed_root)
    remote = tmp_path / "github" / "TheAxiomFoundation" / "rulespec-us.git"
    remote.parent.mkdir(parents=True)
    _git(seed, "switch", "-q", "main")
    subprocess.run(["git", "clone", "-q", "--bare", str(seed), str(remote)], check=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "axiom-encode").symlink_to(ROOT)
    checkout = workspace / "rulespec-us"
    subprocess.run(["git", "clone", "-q", str(remote), str(checkout)], check=True)
    _git(checkout, "fetch", "-q", str(seed), f"{LANE_BRANCH}:{LANE_BRANCH}")
    _git(checkout, "switch", "-q", LANE_BRANCH)
    runner = tmp_path / "runner"
    (runner / "targeted-reencode").mkdir(parents=True)
    for name in (
        "source-bundle.json",
        "canonical-refresh-bundle.json",
        "existing-signed-imports.json",
    ):
        (runner / name).write_text("[]\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "gh").write_text(GH_STUB)
    (bin_dir / "gh").chmod(0o755)
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text(
        f'[url "file://{remote.parent}/"]\n'
        "\tinsteadOf = https://github.com/TheAxiomFoundation/\n"
    )
    env = {
        **{
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("GIT_", "GH_", "GITHUB_"))
        },
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "AXIOM_TEST_PYTHON": python,
        "GIT_CONFIG_GLOBAL": str(gitconfig),
        "GH_TOKEN": "test-token",
        "GH_STUB_BODY": str(tmp_path / "gh-body.txt"),
        "GH_STUB_REMOTE": str(remote),
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "TheAxiomFoundation/axiom-encode",
        "RUNNER_TEMP": str(runner),
        "COUNTRY": "us",
        "CITATION": "us/statute/7/2015/new",
        "DEPENDENT_CITATION": "",
        "SECOND_DEPENDENT_CITATION": "",
        "LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
        "SECOND_LEGACY_EXACT_DEPENDENT_RULESPEC_PATH": "",
        "LEGACY_RETAINED_SUCCESSOR_RULESPEC_PATHS_JSON": "[]",
        "RULESPEC_REF": base,
        "PR_BASE_BRANCH": "main",
        "QUEUE_ID": "",
        "QUEUE_ITEM_ID": "",
        "QUEUE_ITEM_GENERATION_SHA256": "",
        "QUEUE_MANIFEST_SHA256": "",
        "REPAIR_RUN_ID": "",
        "RULESPEC_CHECKOUT": "rulespec-us",
    }
    return {
        "env": env,
        "workspace": workspace,
        "remote": remote,
        "checkout": checkout,
        "runner": runner,
        "lane_head": _git(checkout, "rev-parse", "HEAD"),
    }


def _run_publish(fixture: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", str(_publish_step()["run"])],
        cwd=fixture["workspace"],
        env=fixture["env"],
        capture_output=True,
        text=True,
        check=False,
    )


def _remote_lane(fixture: dict[str, Any]) -> str:
    return subprocess.run(
        [
            "git",
            "--git-dir",
            str(fixture["remote"]),
            "rev-parse",
            f"refs/heads/{LANE_BRANCH}",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_publish_step_pushes_the_signed_lane_plus_the_derived_index(
    tmp_path: Path,
) -> None:
    fixture = _publish_fixture(tmp_path, python=sys.executable)

    result = _run_publish(fixture)

    assert result.returncode == 0, result.stderr
    pushed = _remote_lane(fixture)
    remote = ["git", "--git-dir", str(fixture["remote"])]
    parent = subprocess.run(
        [*remote, "rev-parse", f"{pushed}^"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert parent == fixture["lane_head"]
    changed = subprocess.run(
        [*remote, "diff", "--name-only", parent, pushed],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert changed == [pi.INDEX_PATH.as_posix()]
    evidence = json.loads(
        (fixture["runner"] / "targeted-reencode/provisions-index.json").read_text()
    )
    assert evidence["mode"] == "refreshed"
    assert evidence["head_after"] == pushed
    assert (fixture["checkout"] / pi.INDEX_PATH).read_bytes() == _generator_output(
        fixture["checkout"]
    )
    body = (tmp_path / "gh-body.txt").read_text()
    assert "Reverse index: `refreshed`" in body


def test_publish_step_publishes_the_lane_unchanged_when_refresh_fails(
    tmp_path: Path,
) -> None:
    wrapper = tmp_path / "python-wrapper"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        'for argument in "$@"; do\n'
        '  [ "$argument" = refresh-provisions-index ] && exit 3\n'
        "done\n"
        f'exec "{sys.executable}" "$@"\n'
    )
    wrapper.chmod(0o755)
    fixture = _publish_fixture(tmp_path, python=str(wrapper))

    result = _run_publish(fixture)

    assert result.returncode == 0, result.stderr
    assert "reverse index refresh failed" in result.stderr
    assert _remote_lane(fixture) == fixture["lane_head"]
    evidence = json.loads(
        (fixture["runner"] / "targeted-reencode/provisions-index.json").read_text()
    )
    assert evidence["mode"] == "error"
    assert "Reverse index: `error`" in (tmp_path / "gh-body.txt").read_text()
