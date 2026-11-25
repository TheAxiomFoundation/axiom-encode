"""The repository-scan memos never serve a value for bytes they did not read.

Each staleness test warms a memo, changes the tree in one way (same-size
content with a restored mtime, a new file, a deleted file, a rename, a change
only in an overlay copy or a dependency root, a symlink), then checks that the
memoized scan returns what the unmemoized scan (``AXIOM_ENCODE_SCAN_CACHE=0``)
returns on the changed tree, and that the change was visible at all.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path

import pytest
import yaml

from axiom_encode import repo_routing, scan_memo
from axiom_encode.harness import dependency_stubs, validator_pipeline
from axiom_encode.harness.dependency_stubs import (
    UnsafeRulespecContextPath,
    resolve_canonical_concepts_from_text,
)
from axiom_encode.repo_routing import (
    _checkout_identity_memo_scope,
    inspect_canonical_rulespec_checkout,
)
from axiom_encode.scan_memo import (
    SCAN_CACHE_ENVIRONMENT_VARIABLE,
    TextFactsUnavailable,
    clear_scan_memos,
    register_text_facts,
    scan_cache_enabled,
    scan_memo_stats,
    text_facts,
)

ORIGIN = "https://github.com/TheAxiomFoundation/rulespec-us.git"


@pytest.fixture(autouse=True)
def _cold_memos(monkeypatch):
    monkeypatch.delenv(SCAN_CACHE_ENVIRONMENT_VARIABLE, raising=False)
    clear_scan_memos()
    yield
    clear_scan_memos()


def _module(term: str, symbol: str, *, extra: str = "") -> str:
    return (
        "format: rulespec/v1\n"
        "module:\n"
        f'  summary: The "{term}" means the source-defined condition.{extra}\n'
        "rules:\n"
        f"  - name: {symbol}\n"
        "    kind: derived\n"
        "    entity: Person\n"
        "    period: Year\n"
        "    dtype: Boolean\n"
        "    versions:\n"
        "      - effective_from: 2026-01-01\n"
        "        formula: income_limit * 7\n"
    )


def _checkout(tmp_path: Path, *, git: bool = False) -> Path:
    checkout = tmp_path / "rulespec-us"
    checkout.mkdir(parents=True)
    if git:
        subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", ORIGIN], cwd=checkout, check=True
        )
    content_root = checkout / "us"
    (content_root / "statutes").mkdir(parents=True)
    (content_root / "statutes" / "alpha.yaml").write_text(
        _module("canonical alpha concept", "alpha_concept"), encoding="utf-8"
    )
    (content_root / "statutes" / "beta.yaml").write_text(
        _module("canonical beta concept", "beta_concept"), encoding="utf-8"
    )
    return content_root


TEXT = (
    "A canonical alpha concept, a canonical beta concept, a canonical gamma "
    "concept and a canonical zeta concept apply."
)


def _resolve(monkeypatch, root: Path, *, memo: bool):
    monkeypatch.setenv(SCAN_CACHE_ENVIRONMENT_VARIABLE, "1" if memo else "0")
    try:
        return resolve_canonical_concepts_from_text(TEXT, root)
    finally:
        monkeypatch.delenv(SCAN_CACHE_ENVIRONMENT_VARIABLE)


def _targets(concepts) -> list[tuple[str, str, Path]]:
    return [(c.term, c.import_target, c.rulespec_file) for c in concepts]


def _assert_memo_matches_unmemoized(monkeypatch, root: Path):
    memoized = _resolve(monkeypatch, root, memo=True)
    unmemoized = _resolve(monkeypatch, root, memo=False)
    assert memoized == unmemoized
    return memoized


def _rewrite_same_size_restoring_mtime(path: Path, old: str, new: str) -> None:
    assert len(old) == len(new)
    before = path.stat()
    content = path.read_text(encoding="utf-8")
    assert old in content
    with path.open("r+", encoding="utf-8") as handle:  # same inode
        handle.write(content.replace(old, new))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )


# --- the switch ------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "enabled"),
    [
        (None, True),
        ("", True),
        ("1", True),
        ("on", True),
        (" TRUE ", True),
        ("yes", True),
        ("0", False),
        ("off", False),
        ("false", False),
        ("no", False),
        ("disabled", False),
        ("2", False),
    ],
)
def test_switch_values(monkeypatch, value, enabled):
    if value is None:
        monkeypatch.delenv(SCAN_CACHE_ENVIRONMENT_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(SCAN_CACHE_ENVIRONMENT_VARIABLE, value)
    assert scan_cache_enabled() is enabled


def test_switch_off_runs_the_unmemoized_scan(monkeypatch, tmp_path):
    root = _checkout(tmp_path)
    calls = []
    monkeypatch.setattr(
        dependency_stubs,
        "_concept_file_facts",
        lambda content: calls.append(content),
    )
    _resolve(monkeypatch, root, memo=False)
    assert calls == []
    assert scan_memo_stats()["canonical-concept-file-facts"]["entries"] == 0


# --- text facts ------------------------------------------------------------


def _probe_facts(content: str, outcome: scan_memo.ParseOutcome):
    if outcome.error is not None:
        raise outcome.error
    if content.startswith("raise:"):
        raise TypeError("probe")
    return ("probe", repr(outcome.payload))


register_text_facts("test-probe-facts", _probe_facts)


def test_text_facts_hit_only_for_an_equal_text():
    assert text_facts("a: 1\n", "test-probe-facts") == ("probe", "{'a': 1}")
    assert text_facts("a: 2\n", "test-probe-facts") == ("probe", "{'a': 2}")
    assert text_facts("a: 1\n", "test-probe-facts") == ("probe", "{'a': 1}")
    stats = scan_memo_stats()["test-probe-facts"]
    assert stats["hits"] == 1
    assert stats["entries"] == 2


def test_text_facts_never_store_a_failure():
    for _ in range(2):
        with pytest.raises(TextFactsUnavailable):
            text_facts("raise: now\n", "test-probe-facts")
        with pytest.raises(TextFactsUnavailable):
            text_facts("a: [\n", "test-probe-facts")
    assert scan_memo_stats()["test-probe-facts"]["entries"] == 0


def test_text_facts_refuse_str_subclasses_and_the_switch(monkeypatch):
    class Text(str):
        pass

    with pytest.raises(TextFactsUnavailable):
        text_facts(Text("a: 1\n"), "test-probe-facts")
    monkeypatch.setenv(SCAN_CACHE_ENVIRONMENT_VARIABLE, "0")
    with pytest.raises(TextFactsUnavailable):
        text_facts("a: 1\n", "test-probe-facts")
    assert scan_memo_stats()["test-probe-facts"]["entries"] == 0


def test_text_facts_use_the_pure_python_loader():
    # libyaml accepts these; ``yaml.safe_load`` rejects them, so the
    # executable index must skip them exactly as the unmemoized scan does.
    for text in ("a:\tb\n", "formula: |#-\n  1\n", "[-?  ]\n", "x: 1\n\ufeff"):
        with pytest.raises(yaml.YAMLError):
            yaml.safe_load(text)
        assert text_facts(text, validator_pipeline._EXECUTABLE_RULE_FACTS) == ()
        with pytest.raises(TextFactsUnavailable):
            text_facts(text, "test-probe-facts")


def test_text_facts_evict_least_recently_used():
    register_text_facts("test-small-facts", _probe_facts, max_chars=12)
    text_facts("a: 11111\n", "test-small-facts")
    text_facts("b: 22222\n", "test-small-facts")
    stats = scan_memo_stats()["test-small-facts"]
    assert stats["entries"] == 1
    assert stats["chars"] <= 12


def test_text_facts_reject_a_second_function_under_one_name():
    with pytest.raises(ValueError, match="already registered"):
        register_text_facts("test-probe-facts", lambda content, outcome: None)


def test_text_facts_are_consistent_across_threads():
    texts = [f"key: {index % 7}\n" for index in range(400)]
    results: list[tuple[str, object]] = []

    def worker(chunk):
        for text in chunk:
            results.append((text, text_facts(text, "test-probe-facts")))

    threads = [threading.Thread(target=worker, args=(texts[i::4],)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 400
    for text, value in results:
        assert value == ("probe", repr(yaml.safe_load(text)))


# --- canonical concept index: staleness ----------------------------------


def test_concept_memo_sees_same_size_change_with_restored_mtime(monkeypatch, tmp_path):
    root = _checkout(tmp_path)
    before = _assert_memo_matches_unmemoized(monkeypatch, root)
    assert {c.term for c in before} == {
        "canonical alpha concept",
        "canonical beta concept",
    }
    _rewrite_same_size_restoring_mtime(
        root / "statutes" / "alpha.yaml",
        "canonical alpha concept",
        "canonical gamma concept",
    )
    after = _assert_memo_matches_unmemoized(monkeypatch, root)
    assert {c.term for c in after} == {
        "canonical gamma concept",
        "canonical beta concept",
    }


def test_concept_memo_sees_added_deleted_and_renamed_files(monkeypatch, tmp_path):
    root = _checkout(tmp_path)
    _assert_memo_matches_unmemoized(monkeypatch, root)

    added = root / "statutes" / "gamma.yaml"
    added.write_text(_module("canonical gamma concept", "gamma_concept"))
    after_add = _assert_memo_matches_unmemoized(monkeypatch, root)
    assert "canonical gamma concept" in {c.term for c in after_add}

    (root / "statutes" / "beta.yaml").unlink()
    after_delete = _assert_memo_matches_unmemoized(monkeypatch, root)
    assert "canonical beta concept" not in {c.term for c in after_delete}

    renamed = root / "statutes" / "nested" / "renamed.yaml"
    renamed.parent.mkdir()
    added.rename(renamed)
    after_rename = _assert_memo_matches_unmemoized(monkeypatch, root)
    assert ("canonical gamma concept", "us:statutes/nested/renamed#gamma_concept") in {
        (c.term, c.import_target) for c in after_rename
    }


def test_concept_memo_sees_a_term_that_becomes_ambiguous(monkeypatch, tmp_path):
    root = _checkout(tmp_path)
    _assert_memo_matches_unmemoized(monkeypatch, root)
    # An identical second definition makes the term ambiguous, so it drops
    # out; the memo must not keep the old single-candidate answer.
    (root / "statutes" / "copy.yaml").write_text(
        (root / "statutes" / "alpha.yaml").read_text()
    )
    after = _assert_memo_matches_unmemoized(monkeypatch, root)
    assert "canonical alpha concept" not in {c.term for c in after}


def test_concept_memo_keeps_overlay_copies_apart(monkeypatch, tmp_path):
    root = _checkout(tmp_path / "real")
    overlay_checkout = tmp_path / "overlay" / "rulespec-us"
    shutil.copytree(root.parent, overlay_checkout)
    overlay_root = overlay_checkout / "us"
    _assert_memo_matches_unmemoized(monkeypatch, root)
    _assert_memo_matches_unmemoized(monkeypatch, overlay_root)

    _rewrite_same_size_restoring_mtime(
        overlay_root / "statutes" / "beta.yaml",
        "canonical beta concept",
        "canonical zeta concept",
    )
    real = _assert_memo_matches_unmemoized(monkeypatch, root)
    overlay = _assert_memo_matches_unmemoized(monkeypatch, overlay_root)
    assert "canonical beta concept" in {c.term for c in real}
    assert "canonical zeta concept" not in {c.term for c in real}
    assert "canonical zeta concept" in {c.term for c in overlay}
    assert "canonical beta concept" not in {c.term for c in overlay}
    assert all(c.rulespec_file.is_relative_to(root) for c in real)
    assert all(c.rulespec_file.is_relative_to(overlay_root) for c in overlay)


def test_concept_memo_rejects_a_symlink_that_replaced_a_module(monkeypatch, tmp_path):
    root = _checkout(tmp_path)
    _assert_memo_matches_unmemoized(monkeypatch, root)
    module = root / "statutes" / "alpha.yaml"
    outside = tmp_path / "outside.yaml"
    outside.write_text(module.read_text())
    module.unlink()
    module.symlink_to(outside)
    for memo in (True, False):
        with pytest.raises(UnsafeRulespecContextPath, match="symlink"):
            _resolve(monkeypatch, root, memo=memo)


def test_concept_memo_matches_on_malformed_and_odd_modules(monkeypatch, tmp_path):
    root = _checkout(tmp_path)
    odd = {
        "tab.yaml": 'module:\n  summary:\t"canonical delta concept" means x.\n',
        "fallback.yaml": '"""The "canonical epsilon concept" means y."""\nrules: [\n',
        "list.yaml": "- a\n- b\n",
        "empty.yaml": "",
        "norules.yaml": 'module:\n  summary: The "canonical zeta concept" means z.\n',
    }
    for name, content in odd.items():
        (root / "statutes" / name).write_text(content)
    _assert_memo_matches_unmemoized(monkeypatch, root)
    _assert_memo_matches_unmemoized(monkeypatch, root)


def test_concept_memo_reraises_what_the_helpers_raise(monkeypatch, tmp_path):
    root = _checkout(tmp_path)

    def exploding_load(content, *args, **kwargs):
        raise RecursionError("too deep")

    monkeypatch.setattr(dependency_stubs.yaml, "safe_load", exploding_load)
    monkeypatch.setattr(scan_memo.yaml, "safe_load", exploding_load)
    for memo in (True, False):
        with pytest.raises(RecursionError, match="too deep"):
            _resolve(monkeypatch, root, memo=memo)
    assert scan_memo_stats()["canonical-concept-file-facts"]["entries"] == 0


def test_concept_memo_parses_each_text_once_per_process(monkeypatch, tmp_path):
    root = _checkout(tmp_path)
    loads = []
    original = scan_memo.yaml.safe_load

    def counting_load(content, *args, **kwargs):
        loads.append(content)
        return original(content, *args, **kwargs)

    monkeypatch.setattr(scan_memo.yaml, "safe_load", counting_load)
    _resolve(monkeypatch, root, memo=True)
    _resolve(monkeypatch, root, memo=True)
    assert len(loads) == 2
    # The executable index reuses the same parse.
    validator_pipeline._rulespec_executable_index_for_roots.__wrapped__(
        (str(root.resolve()),)
    )
    assert len(loads) == 2


# --- executable index: staleness ------------------------------------------


def _executable_index(monkeypatch, roots: tuple[Path, ...], *, memo: bool):
    monkeypatch.setenv(SCAN_CACHE_ENVIRONMENT_VARIABLE, "1" if memo else "0")
    try:
        # ``__wrapped__`` skips main's path-keyed ``lru_cache`` so that each
        # call scans the tree.
        return validator_pipeline._rulespec_executable_index_for_roots.__wrapped__(
            tuple(str(root.resolve()) for root in roots)
        )
    finally:
        monkeypatch.delenv(SCAN_CACHE_ENVIRONMENT_VARIABLE)


def _assert_index_matches(monkeypatch, roots):
    memoized = _executable_index(monkeypatch, roots, memo=True)
    unmemoized = _executable_index(monkeypatch, roots, memo=False)
    assert memoized == unmemoized
    return memoized


def test_executable_index_sees_same_size_change_with_restored_mtime(
    monkeypatch, tmp_path
):
    root = _checkout(tmp_path)
    before = _assert_index_matches(monkeypatch, (root,))
    _rewrite_same_size_restoring_mtime(
        root / "statutes" / "alpha.yaml", "income_limit * 7", "income_limit * 9"
    )
    after = _assert_index_matches(monkeypatch, (root,))
    assert before != after
    assert any("income_limit * 9" in record.signature for record in after)


def test_executable_index_sees_changes_in_a_dependency_root(monkeypatch, tmp_path):
    active = _checkout(tmp_path / "active")
    dependency = _checkout(tmp_path / "dependency")
    roots = (active, dependency)
    before = _assert_index_matches(monkeypatch, roots)
    assert {Path(r.source_file).parent.parent.parent.parent for r in before} == {
        active.parent.resolve().parent,
        dependency.parent.resolve().parent,
    }
    _rewrite_same_size_restoring_mtime(
        dependency / "statutes" / "beta.yaml", "beta_concept", "zeta_concept"
    )
    (dependency / "statutes" / "alpha.yaml").unlink()
    after = _assert_index_matches(monkeypatch, roots)
    dependency_symbols = {
        r.symbol
        for r in after
        if Path(r.source_file).is_relative_to(dependency.resolve())
    }
    assert dependency_symbols == {"zeta_concept"}


def test_executable_index_sees_added_and_renamed_files(monkeypatch, tmp_path):
    root = _checkout(tmp_path)
    _assert_index_matches(monkeypatch, (root,))
    (root / "statutes" / "beta.yaml").rename(root / "statutes" / "renamed.yaml")
    (root / "regulations").mkdir()
    (root / "regulations" / "copy.yaml").write_text(
        (root / "statutes" / "alpha.yaml").read_text()
    )
    after = _assert_index_matches(monkeypatch, (root,))
    targets = {record.target for record in after}
    assert "us:statutes/renamed#beta_concept" in targets
    assert "us:regulations/copy#alpha_concept" in targets
    assert "us:statutes/beta#beta_concept" not in targets


def test_executable_index_reraises_what_the_unmemoized_scan_raises(
    monkeypatch, tmp_path
):
    root = _checkout(tmp_path)

    def exploding_signature(rule):
        raise TypeError("unsigned")

    monkeypatch.setattr(
        validator_pipeline, "_rulespec_executable_signature", exploding_signature
    )
    for memo in (True, False):
        with pytest.raises(TypeError, match="unsigned"):
            _executable_index(monkeypatch, (root,), memo=memo)
    assert scan_memo_stats()["rulespec-executable-rule-facts"]["entries"] == 0


def test_executable_index_skips_unreadable_text_like_the_unmemoized_scan(
    monkeypatch, tmp_path
):
    root = _checkout(tmp_path)
    (root / "statutes" / "latin1.yaml").write_bytes(b"name: caf\xe9\n")
    (root / "statutes" / "tab.yaml").write_text(
        _module("canonical tab concept", "tab_concept").replace(
            "kind: derived", "kind:\tderived"
        )
    )
    after = _assert_index_matches(monkeypatch, (root,))
    assert "tab_concept" not in {record.symbol for record in after}


# --- checkout identity memo -----------------------------------------------


def _count_git(monkeypatch) -> list[tuple[str, ...]]:
    calls: list[tuple[str, ...]] = []
    original = subprocess.run

    def counting_run(command, *args, **kwargs):
        calls.append(tuple(command))
        return original(command, *args, **kwargs)

    monkeypatch.setattr(repo_routing.subprocess, "run", counting_run)
    return calls


def test_identity_memo_probes_once_per_scan(monkeypatch, tmp_path):
    root = _checkout(tmp_path, git=True)
    for index in range(30):
        (root / "statutes" / f"extra-{index}.yaml").write_text(
            _module(f"canonical extra {index} concept", f"extra_{index}")
        )
    calls = _count_git(monkeypatch)
    unmemoized = _resolve(monkeypatch, root, memo=False)
    unmemoized_calls = len(calls)
    calls.clear()
    memoized = _resolve(monkeypatch, root, memo=True)
    assert memoized == unmemoized
    assert unmemoized_calls >= 2 * 32
    assert len(calls) <= 6


def test_identity_memo_sees_a_changed_origin(monkeypatch, tmp_path):
    checkout = _checkout(tmp_path, git=True).parent
    with _checkout_identity_memo_scope():
        assert inspect_canonical_rulespec_checkout(checkout).name == "rulespec-us"
        subprocess.run(
            ["git", "remote", "set-url", "origin", "https://example.com/other.git"],
            cwd=checkout,
            check=True,
        )
        assert inspect_canonical_rulespec_checkout(checkout) == (
            None,
            "git-origin-name-mismatch",
        )


def test_identity_memo_sees_a_new_checkout_root_entry(monkeypatch, tmp_path):
    checkout = _checkout(tmp_path, git=True).parent
    with _checkout_identity_memo_scope():
        assert inspect_canonical_rulespec_checkout(checkout).name == "rulespec-us"
        (checkout / "statutes").mkdir()
        assert inspect_canonical_rulespec_checkout(checkout) == (
            None,
            "atomic-root-at-checkout",
        )


def test_identity_memo_keys_on_the_admission_flags(monkeypatch, tmp_path):
    checkout = _checkout(tmp_path, git=True).parent
    (checkout / "programs").mkdir()
    with _checkout_identity_memo_scope():
        assert (
            inspect_canonical_rulespec_checkout(
                checkout, allow_composition_specs=True
            ).name
            == "rulespec-us"
        )
        assert inspect_canonical_rulespec_checkout(checkout) == (
            None,
            "atomic-root-at-checkout",
        )


def test_identity_memo_never_keeps_a_rejection(monkeypatch, tmp_path):
    checkout = _checkout(tmp_path, git=True).parent
    (checkout / "statutes").mkdir()
    with _checkout_identity_memo_scope():
        assert inspect_canonical_rulespec_checkout(checkout).name is None
        (checkout / "statutes").rmdir()
        assert inspect_canonical_rulespec_checkout(checkout).name == "rulespec-us"


def test_identity_memo_ends_with_its_scope(monkeypatch, tmp_path):
    checkout = _checkout(tmp_path, git=True).parent
    calls = _count_git(monkeypatch)
    with _checkout_identity_memo_scope():
        inspect_canonical_rulespec_checkout(checkout)
        inside = len(calls)
        inspect_canonical_rulespec_checkout(checkout)
        assert len(calls) == inside
    inspect_canonical_rulespec_checkout(checkout)
    assert len(calls) == inside + 2


def _supervisor_git_environment(monkeypatch) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.delenv("GIT_CONFIG_SYSTEM", raising=False)


def test_identity_memo_admits_the_supervisor_git_environment(monkeypatch, tmp_path):
    checkout = _checkout(tmp_path, git=True).parent
    _supervisor_git_environment(monkeypatch)
    calls = _count_git(monkeypatch)
    with _checkout_identity_memo_scope():
        for _ in range(5):
            assert inspect_canonical_rulespec_checkout(checkout).name == "rulespec-us"
        # Two probes (top level, origin) and no ``git config`` probe.
        assert len(calls) == 2
        assert not any("config" in call for call in calls)
        subprocess.run(
            ["git", "remote", "set-url", "origin", "https://example.com/other.git"],
            cwd=checkout,
            check=True,
        )
        assert inspect_canonical_rulespec_checkout(checkout).name is None


def test_identity_memo_refuses_repository_includes_under_the_supervisor(
    monkeypatch, tmp_path
):
    checkout = _checkout(tmp_path, git=True).parent
    included = tmp_path / "included.gitconfig"
    included.write_text("[core]\n\tbare = false\n")
    with (checkout / ".git" / "config").open("a") as handle:
        handle.write(f"[include]\n\tpath = {included}\n")
    _supervisor_git_environment(monkeypatch)
    calls = _count_git(monkeypatch)
    with _checkout_identity_memo_scope():
        for _ in range(3):
            assert inspect_canonical_rulespec_checkout(checkout).name == "rulespec-us"
    assert len(calls) == 6


@pytest.mark.parametrize(
    "environment",
    [
        {"GIT_CONFIG_GLOBAL": "/tmp/other"},
        {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "0"},
        {
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": "/tmp/system",
        },
        {"GIT_CONFIG_SYSTEM": "/dev/null"},
    ],
)
def test_identity_memo_refuses_other_custom_git_environments(
    monkeypatch, tmp_path, environment
):
    checkout = _checkout(tmp_path, git=True).parent
    for name in ("GIT_CONFIG_GLOBAL", "GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_SYSTEM"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    calls = _count_git(monkeypatch)
    with _checkout_identity_memo_scope():
        for _ in range(3):
            inspect_canonical_rulespec_checkout(checkout)
    assert len(calls) == 6


# --- symlink audit memo ---------------------------------------------------


def _audit(checkout: Path) -> None:
    validator_pipeline._reject_rulespec_checkout_symlinks(checkout, label="checkout")


def _audit_walks(monkeypatch) -> list[str]:
    walks: list[str] = []
    original = validator_pipeline.os.walk

    def counting_walk(top, *args, **kwargs):
        walks.append(str(top))
        return original(top, *args, **kwargs)

    monkeypatch.setattr(validator_pipeline.os, "walk", counting_walk)
    return walks


def test_symlink_audit_memo_walks_once_while_unchanged(monkeypatch, tmp_path):
    checkout = _checkout(tmp_path).parent
    walks = _audit_walks(monkeypatch)
    with validator_pipeline._symlink_audit_memo_scope():
        for _ in range(5):
            _audit(checkout)
    assert len(walks) == 1
    _audit(checkout)
    assert len(walks) == 2


@pytest.mark.parametrize("where", ["root", "nested"])
def test_symlink_audit_memo_sees_a_new_symlink(monkeypatch, tmp_path, where):
    checkout = _checkout(tmp_path).parent
    directory = checkout if where == "root" else checkout / "us" / "statutes"
    with validator_pipeline._symlink_audit_memo_scope():
        _audit(checkout)
        (directory / "link.yaml").symlink_to(tmp_path / "anywhere")
        with pytest.raises(UnsafeRulespecContextPath, match="contains a symlink"):
            _audit(checkout)


def test_symlink_audit_memo_sees_a_file_replaced_by_a_symlink_with_restored_times(
    monkeypatch, tmp_path
):
    checkout = _checkout(tmp_path).parent
    directory = checkout / "us" / "statutes"
    module = directory / "alpha.yaml"
    with validator_pipeline._symlink_audit_memo_scope():
        _audit(checkout)
        before = directory.stat()
        target = tmp_path / "alpha.yaml"
        target.write_text(module.read_text())
        module.unlink()
        module.symlink_to(target)
        os.utime(directory, ns=(before.st_atime_ns, before.st_mtime_ns))
        with pytest.raises(UnsafeRulespecContextPath, match="contains a symlink"):
            _audit(checkout)


def test_symlink_audit_memo_never_keeps_a_failed_audit(monkeypatch, tmp_path):
    checkout = _checkout(tmp_path).parent
    link = checkout / "us" / "link.yaml"
    link.symlink_to(tmp_path / "anywhere")
    with validator_pipeline._symlink_audit_memo_scope():
        for _ in range(2):
            with pytest.raises(UnsafeRulespecContextPath):
                _audit(checkout)
        link.unlink()
        _audit(checkout)


def test_symlink_audit_memo_keeps_overlay_copies_apart(monkeypatch, tmp_path):
    checkout = _checkout(tmp_path / "real").parent
    overlay = tmp_path / "overlay" / "rulespec-us"
    shutil.copytree(checkout, overlay)
    with validator_pipeline._symlink_audit_memo_scope():
        _audit(checkout)
        _audit(overlay)
        (overlay / "us" / "link.yaml").symlink_to(tmp_path / "anywhere")
        _audit(checkout)
        with pytest.raises(UnsafeRulespecContextPath):
            _audit(overlay)


def test_symlink_audit_memo_is_off_with_the_switch(monkeypatch, tmp_path):
    checkout = _checkout(tmp_path).parent
    monkeypatch.setenv(SCAN_CACHE_ENVIRONMENT_VARIABLE, "0")
    walks = _audit_walks(monkeypatch)
    with validator_pipeline._symlink_audit_memo_scope():
        _audit(checkout)
        _audit(checkout)
    assert len(walks) == 2


def test_symlink_audit_memo_defers_to_an_open_resolution_cache(monkeypatch, tmp_path):
    checkout = _checkout(tmp_path).parent
    calls = []
    monkeypatch.setattr(
        validator_pipeline,
        "_reject_rulespec_checkout_symlinks_memoized",
        lambda *args, **kwargs: calls.append(args),
    )
    with (
        validator_pipeline._rulespec_resolution_cache_scope(),
        validator_pipeline._symlink_audit_memo_scope(),
    ):
        _audit(checkout)
    assert calls == []
