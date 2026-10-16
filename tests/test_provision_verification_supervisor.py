import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts.provision_verification_supervisor import (
    _load_retired_apply_inventory,
    _signing_trust_roots_payload,
)


def test_provision_replaces_base_runtime_site_packages(tmp_path: Path) -> None:
    git = shutil.which("git")
    if git is None:
        raise AssertionError("Git is required for the provisioner test")
    source_site_packages = (
        Path(sys.base_prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    assert (source_site_packages / "pip").is_dir()
    assert (source_site_packages / "README.txt").is_file()

    trusted_packages = tmp_path / "trusted-packages"
    intended_package = trusted_packages / "intended_package"
    intended_package.mkdir(parents=True)
    (intended_package / "__init__.py").write_text("TRUSTED = True\n")
    supervisor = tmp_path / "supervisor"
    supervisor.write_text("supervisor\n")
    destination = tmp_path / "provisioned"

    subprocess.run(
        [
            sys.executable,
            "scripts/provision_verification_supervisor.py",
            "--destination",
            str(destination),
            "--supervisor",
            str(supervisor),
            "--site-packages",
            str(trusted_packages),
            "--apply-root",
            "apply-root",
            "--eval-root",
            "eval-root",
            "--corpus-release-root",
            "corpus-release-root",
            "--git",
            str(Path(git).resolve()),
        ],
        check=True,
    )

    runtime = destination / "python"
    interpreter = runtime / Path(sys.executable).resolve().relative_to(
        Path(sys.base_prefix).resolve()
    )
    assert (destination / "axiom-encode").read_text().splitlines()[0] == (
        f"#!{interpreter} -I"
    )
    git_wrapper = interpreter.parent / "git"
    assert git_wrapper.read_text().splitlines()[0] == f"#!{interpreter} -I"
    assert not (destination / "git").exists()
    repository = tmp_path / "rulespec-us"
    subprocess.run(
        [str(Path(git).resolve()), "init", "--quiet", str(repository)], check=True
    )
    top_level = subprocess.run(
        [str(git_wrapper), "-C", str(repository), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": str(runtime),
            "PATH": str(interpreter.parent),
        },
    )
    assert Path(top_level.stdout.strip()) == repository
    assert json.loads((destination / "signing-trust-roots.json").read_text()) == {
        "schema": "axiom-encode/signing-trust-roots/v2",
        "apply_ed25519_public_key": "apply-root",
        "eval_ed25519_public_key": "eval-root",
        "corpus_release_ed25519_public_key": "corpus-release-root",
    }

    provisioned_site_packages = (
        runtime
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    assert sorted(
        path.relative_to(provisioned_site_packages)
        for path in provisioned_site_packages.rglob("*")
    ) == [
        Path("intended_package"),
        Path("intended_package/__init__.py"),
    ]
    assert not any(path.is_symlink() for path in destination.rglob("*"))
    forbidden_patterns = (
        "*.pth",
        "*.egg-link",
        "sitecustomize.py",
        "usercustomize.py",
        "pyvenv.cfg",
        "__editable__*",
    )
    assert not any(
        path.is_file()
        for pattern in forbidden_patterns
        for path in destination.rglob(pattern)
    )


def test_provision_writes_v3_corpus_release_keyring() -> None:
    assert _signing_trust_roots_payload(
        "apply",
        "eval",
        "current",
        ("retired-one", "retired-two"),
    ) == {
        "schema": "axiom-encode/signing-trust-roots/v3",
        "apply_ed25519_public_key": "apply",
        "eval_ed25519_public_key": "eval",
        "corpus_release_ed25519_public_keys": [
            "current",
            "retired-one",
            "retired-two",
        ],
    }


def _owner_frozen_inventory() -> dict[str, object]:
    return {
        "schema": "axiom-encode/retired-apply-inventory/v1",
        "repositories": [
            {
                "repository": "TheAxiomFoundation/rulespec-us",
                "cutoff_commit": "a" * 40,
                "artifacts": [
                    {
                        "path": ".axiom/encoding-manifests/frozen.json",
                        "sha256": "b" * 64,
                        "covered_files": [
                            {"path": "statute/frozen.yaml", "sha256": "c" * 64}
                        ],
                    }
                ],
            }
        ],
    }


@pytest.mark.parametrize("corpus_retired", [(), ("old-corpus",)])
def test_provision_writes_v4_only_with_paired_retired_apply_trust(
    corpus_retired: tuple[str, ...],
) -> None:
    inventory = _owner_frozen_inventory()
    payload = _signing_trust_roots_payload(
        "apply",
        "eval",
        "corpus",
        corpus_retired,
        ("synthetic-retired",),
        inventory,
    )
    assert payload["schema"] == "axiom-encode/signing-trust-roots/v4"
    assert payload["apply_ed25519_public_key"] == "apply"
    assert payload["retired_apply_ed25519_public_keys"] == ["synthetic-retired"]
    assert payload["retired_apply_inventory"] == inventory
    if corpus_retired:
        assert payload["corpus_release_ed25519_public_keys"] == ["corpus", "old-corpus"]
    else:
        assert payload["corpus_release_ed25519_public_key"] == "corpus"


@pytest.mark.parametrize(
    "roots,inventory", [((), _owner_frozen_inventory()), (("old",), None)]
)
def test_provision_rejects_unpaired_retired_apply_trust(roots, inventory) -> None:
    with pytest.raises(SystemExit, match="supplied together"):
        _signing_trust_roots_payload("apply", "eval", "corpus", (), roots, inventory)


@given(
    st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1),
    st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1),
    st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1),
    st.lists(
        st.text(
            alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1
        ),
        max_size=3,
    ),
)
def test_provision_without_retired_apply_preserves_exact_legacy_json(
    apply, evaluation, corpus, retired_corpus
) -> None:
    baseline = {
        "schema": "axiom-encode/signing-trust-roots/v2",
        "apply_ed25519_public_key": apply,
        "eval_ed25519_public_key": evaluation,
        "corpus_release_ed25519_public_key": corpus,
    }
    if retired_corpus:
        baseline["schema"] = "axiom-encode/signing-trust-roots/v3"
        baseline.pop("corpus_release_ed25519_public_key")
        baseline["corpus_release_ed25519_public_keys"] = [corpus, *retired_corpus]
    actual = _signing_trust_roots_payload(
        apply, evaluation, corpus, tuple(retired_corpus)
    )
    assert (
        json.dumps(actual, sort_keys=True) + "\n"
        == json.dumps(baseline, sort_keys=True) + "\n"
    )


@pytest.mark.parametrize("mutation", ["unowned", "writable", "symlink"])
def test_inventory_loader_refuses_unprotected_owner_source(
    tmp_path: Path, monkeypatch, mutation: str
) -> None:
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(_owner_frozen_inventory()))
    original_lstat = Path.lstat

    def lstat(path):
        metadata = original_lstat(path)
        mode = stat.S_IFREG | 0o644 if path == inventory else stat.S_IFDIR | 0o755
        uid = 0
        if path == inventory:
            if mutation == "unowned":
                uid = os.getuid() or 12345
            elif mutation == "writable":
                mode |= 0o020
            else:
                mode = stat.S_IFLNK | 0o777
        return SimpleNamespace(st_mode=mode, st_uid=uid, st_size=metadata.st_size)

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(SystemExit, match="root-owned"):
        _load_retired_apply_inventory(inventory)


def test_inventory_loader_reads_only_valid_owner_frozen_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    inventory = tmp_path / "inventory.json"
    expected = _owner_frozen_inventory()
    inventory.write_text(json.dumps(expected))
    monkeypatch.setattr(
        Path,
        "lstat",
        lambda path: SimpleNamespace(
            st_uid=0,
            st_mode=(stat.S_IFREG | 0o644)
            if path == inventory
            else (stat.S_IFDIR | 0o755),
        ),
    )
    assert _load_retired_apply_inventory(inventory) == expected
    inventory.write_text(
        '{"schema":"axiom-encode/retired-apply-inventory/v1","schema":"duplicate","repositories":[]}'
    )
    with pytest.raises(SystemExit, match="duplicate JSON key"):
        _load_retired_apply_inventory(inventory)
