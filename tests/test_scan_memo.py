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
    # One entry costs its text (9) plus its value (20): two do not fit in 40.
    register_text_facts("test-small-facts", _probe_facts, max_chars=40)
    text_facts("a: 11111\n", "test-small-facts")
    assert scan_memo_stats()["test-small-facts"]["entries"] == 1
    text_facts("b: 22222\n", "test-small-facts")
    stats = scan_memo_stats()["test-small-facts"]
    assert stats["entries"] == 1
    assert 0 < stats["chars"] <= 40
    text_facts("b: 22222\n", "test-small-facts")
    assert scan_memo_stats()["test-small-facts"]["hits"] == 1


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
    monkeypatch.setattr(scan_memo, "_safe_load_with_node", exploding_load)
    for memo in (True, False):
        with pytest.raises(RecursionError, match="too deep"):
            _resolve(monkeypatch, root, memo=memo)
    assert scan_memo_stats()["canonical-concept-file-facts"]["entries"] == 0


def test_concept_memo_parses_each_text_once_per_process(monkeypatch, tmp_path):
    root = _checkout(tmp_path)
    loads = []
    original = scan_memo._safe_load_with_node

    def counting_load(content, *args, **kwargs):
        loads.append(content)
        return original(content, *args, **kwargs)

    monkeypatch.setattr(scan_memo, "_safe_load_with_node", counting_load)
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


def test_identity_memo_never_keeps_a_transient_probe_failure(monkeypatch, tmp_path):
    checkout = _checkout(tmp_path, git=True).parent
    original = repo_routing._git_top_level
    failures = [repo_routing._GitProbeError("slow host", category="timeout")]

    def flaky_top_level(root):
        if failures:
            raise failures.pop()
        return original(root)

    monkeypatch.setattr(repo_routing, "_git_top_level", flaky_top_level)
    with _checkout_identity_memo_scope():
        assert inspect_canonical_rulespec_checkout(checkout) == (
            None,
            "git-top-level-probe-timeout",
        )
        assert inspect_canonical_rulespec_checkout(checkout).name == "rulespec-us"


# --- review round 2 ---------------------------------------------------------


def _large_value_facts(content: str, outcome: scan_memo.ParseOutcome):
    if outcome.error is not None:
        raise outcome.error
    return ("v" * 1000, str(outcome.payload))


def test_memory_bound_counts_the_stored_value():
    # The texts are about 10 characters; each value is about 1,000.
    register_text_facts("test-large-value-facts", _large_value_facts, max_chars=2500)
    for index in range(5):
        text_facts(f"a: {index}\n", "test-large-value-facts")
    stats = scan_memo_stats()["test-large-value-facts"]
    assert stats["entries"] == 2
    assert 2000 < stats["chars"] <= 2500
    # The two most recent texts stay; the older ones were evicted.
    text_facts("a: 4\n", "test-large-value-facts")
    text_facts("a: 3\n", "test-large-value-facts")
    assert scan_memo_stats()["test-large-value-facts"]["hits"] == 2
    text_facts("a: 0\n", "test-large-value-facts")
    assert scan_memo_stats()["test-large-value-facts"]["hits"] == 2


def test_memory_bound_refuses_a_value_larger_than_the_bound():
    register_text_facts("test-oversize-value-facts", _large_value_facts, max_chars=500)
    for _ in range(2):
        assert text_facts("a: 1\n", "test-oversize-value-facts")[1] == "{'a': 1}"
    stats = scan_memo_stats()["test-oversize-value-facts"]
    assert (stats["entries"], stats["chars"], stats["hits"]) == (0, 0, 0)


def test_value_size_counts_nested_strings_and_dataclass_fields():
    facts = dependency_stubs._ConceptFileFacts(
        source_text="s" * 300,
        principal=("n" * 50, "Person", "Year", "Boolean"),
        terms=("t" * 70,),
    )
    assert scan_memo._value_chars(facts) > 300 + 50 + 70
    assert scan_memo._value_chars(()) == 1
    assert scan_memo._value_chars((("a", "bc"),)) == 2 + (1 + 1) + (1 + 2)


def test_text_facts_keep_texts_with_a_long_common_prefix_apart():
    # The two texts are equal for more than 4,096 characters.
    prefix = "# " + "x" * 5000 + "\n"
    first = prefix + "a: 1\n"
    second = prefix + "a: 2\n"
    assert first[:4096] == second[:4096]
    assert len(first) == len(second)
    for _ in range(2):
        assert text_facts(first, "test-probe-facts") == ("probe", "{'a': 1}")
        assert text_facts(second, "test-probe-facts") == ("probe", "{'a': 2}")
    stats = scan_memo_stats()["test-probe-facts"]
    assert (stats["entries"], stats["hits"]) == (2, 2)


def _nested_text(depth: int) -> str:
    # The payload is ``{"a": [[...]]}``: ``depth`` containers deep.
    return "a: " + "[" * (depth - 1) + "1" + "]" * (depth - 1) + "\n"


def _compose(text: str):
    return yaml.compose(text, Loader=yaml.SafeLoader)


def test_nesting_measure_matches_the_composed_depth():
    limit = scan_memo._MAX_MEMOIZED_NESTING
    assert limit == 100
    for depth, exceeds in ((1, False), (limit, False), (limit + 1, True)):
        node = _compose(_nested_text(depth))
        assert scan_memo._nesting_exceeds(node, limit) is exceeds
    assert not scan_memo._nesting_exceeds(_compose("scalar\n"), 0)
    assert not scan_memo._nesting_exceeds(None, 0)
    # A mapping key and a shared (aliased) node count too.
    assert scan_memo._nesting_exceeds(_compose("? [[k]]\n: 1\n"), 2)
    assert not scan_memo._nesting_exceeds(_compose("? [[k]]\n: 1\n"), 3)
    shared = _compose("a: &x [[1]]\nb: [*x, *x]\n")
    assert scan_memo._nesting_exceeds(shared, 3)
    assert not scan_memo._nesting_exceeds(shared, 4)
    # ``&a [*a]`` composes to a node that contains itself.
    assert scan_memo._nesting_exceeds(_compose("&a [*a]\n"), limit)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "# only a comment\n",
        "a: 1\n",
        "a: &x [1, 2]\nb: *x\n",
        "base: &b {k: 1}\nuse:\n  <<: *b\n  k: 2\n",
        "a: 1\na: 2\n",
        "when: 2026-01-01\nflag: yes\nnone: ~\nnumber: 0x10\n",
        "? [1, 2]\n: value\n",
        "a:\tb\n",
        "a: [\n",
        "a: 1\n---\nb: 2\n",
        "a: !!python/object:os.system x\n",
        "a: *missing\n",
        "x: 1\n\ufeff",
    ],
)
def test_the_memo_parse_is_yaml_safe_load(text):
    try:
        expected = yaml.safe_load(text)
    except Exception as exc:  # noqa: BLE001 - compared below
        with pytest.raises(type(exc)) as raised:
            scan_memo._safe_load_with_node(text)
        assert str(raised.value) == str(exc)
        return
    payload, node = scan_memo._safe_load_with_node(text)
    assert payload == expected
    assert type(payload) is type(expected)
    assert (node is None) == (yaml.compose(text, Loader=yaml.SafeLoader) is None)


@pytest.mark.parametrize(
    "dropped",
    [
        # A duplicate key: the later scalar replaces the deep value.
        "junk: {nested}\njunk: 1\n",
        # A merge key: the explicit key hides the deep merged value.
        "base: &b\n  junk: {nested}\nuse:\n  <<: *b\n  junk: 1\nbase: 1\n",
    ],
)
def test_text_facts_measure_the_parse_and_not_the_payload(monkeypatch, dropped):
    nested = "[" * 300 + "1" + "]" * 300
    text = dropped.format(nested=nested) + "a: 1\n"
    # The parser recursed 300 levels, but the payload is shallow.
    assert "[" not in repr(yaml.safe_load(text))
    for _ in range(2):
        assert text_facts(text, "test-probe-facts")[0] == "probe"
    stats = scan_memo_stats()["test-probe-facts"]
    assert (stats["entries"], stats["hits"], stats["misses"]) == (0, 0, 2)

    # At a deeper stack the parse raises; no stored value may hide that.
    def exhausted_load(content):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(scan_memo, "_safe_load_with_node", exhausted_load)
    with pytest.raises(TextFactsUnavailable):
        text_facts(text, "test-probe-facts")


def test_executable_index_does_not_serve_a_module_with_a_dropped_deep_value(
    monkeypatch, tmp_path
):
    root = _checkout(tmp_path)
    nested = "[" * 300 + "1" + "]" * 300
    (root / "statutes" / "dropped.yaml").write_text(
        f"junk: {nested}\njunk: 1\n" + _module("canonical dropped concept", "dropped"),
        encoding="utf-8",
    )
    first = _assert_index_matches(monkeypatch, (root,))
    assert "dropped" in {record.symbol for record in first}
    # Only the two shallow modules are stored.
    assert scan_memo_stats()["rulespec-executable-rule-facts"]["entries"] == 2

    def depth_limited(original):
        def load(content, *args, **kwargs):
            if nested in content:
                raise RecursionError("maximum recursion depth exceeded")
            return original(content, *args, **kwargs)

        return load

    monkeypatch.setattr(
        scan_memo,
        "_safe_load_with_node",
        depth_limited(scan_memo._safe_load_with_node),
    )
    monkeypatch.setattr(
        validator_pipeline.yaml, "safe_load", depth_limited(yaml.safe_load)
    )
    for memo in (True, False):
        with pytest.raises(RecursionError):
            _executable_index(monkeypatch, (root,), memo=memo)


def test_text_facts_do_not_store_a_deeply_nested_text(monkeypatch):
    limit = scan_memo._MAX_MEMOIZED_NESTING
    shallow = _nested_text(limit)
    deep = _nested_text(limit + 1)
    for text in (shallow, deep):
        for _ in range(2):
            assert text_facts(text, "test-probe-facts") == (
                "probe",
                repr(yaml.safe_load(text)),
            )
    stats = scan_memo_stats()["test-probe-facts"]
    assert (stats["entries"], stats["hits"], stats["misses"]) == (1, 1, 3)

    # A later call at a deeper stack, where the parse raises
    # ``RecursionError``, must raise as the unmemoized scan does: the value
    # computed at the shallow stack must not be served.
    def exhausted_load(content, *args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(scan_memo, "_safe_load_with_node", exhausted_load)
    with pytest.raises(TextFactsUnavailable):
        text_facts(deep, "test-probe-facts")
    # The shallow text needs no parse and is still served.
    assert text_facts(shallow, "test-probe-facts")[0] == "probe"


def test_executable_index_raises_recursion_error_for_a_deep_module(
    monkeypatch, tmp_path
):
    root = _checkout(tmp_path)
    nested = "[" * 150 + "1" + "]" * 150
    (root / "statutes" / "deep.yaml").write_text(
        _module("canonical deep concept", "deep_concept")
        + f"    metadata:\n      nested: {nested}\n",
        encoding="utf-8",
    )
    first = _assert_index_matches(monkeypatch, (root,))
    assert "deep_concept" in {record.symbol for record in first}
    assert scan_memo_stats()["rulespec-executable-rule-facts"]["entries"] == 2

    def depth_limited(original):
        def load(content, *args, **kwargs):
            if "[" * 150 in content:
                raise RecursionError("maximum recursion depth exceeded")
            return original(content, *args, **kwargs)

        return load

    monkeypatch.setattr(
        scan_memo,
        "_safe_load_with_node",
        depth_limited(scan_memo._safe_load_with_node),
    )
    monkeypatch.setattr(
        validator_pipeline.yaml, "safe_load", depth_limited(yaml.safe_load)
    )
    for memo in (True, False):
        with pytest.raises(RecursionError):
            _executable_index(monkeypatch, (root,), memo=memo)


def test_text_facts_do_not_store_a_failed_parse():
    text = "a:\tb\n"
    for _ in range(2):
        assert text_facts(text, validator_pipeline._EXECUTABLE_RULE_FACTS) == ()
    stats = scan_memo_stats()[validator_pipeline._EXECUTABLE_RULE_FACTS]
    assert (stats["entries"], stats["hits"], stats["misses"]) == (0, 0, 2)


@pytest.mark.parametrize(
    "error",
    [RecursionError("too deep"), MemoryError("no memory"), TypeError("bad node")],
)
def test_executable_facts_propagate_parse_exceptions(monkeypatch, tmp_path, error):
    with pytest.raises(type(error)) as raised:
        validator_pipeline._executable_rule_facts(
            "a: 1\n", scan_memo.ParseOutcome(None, error)
        )
    assert raised.value is error
    for skipped in (yaml.YAMLError("bad"), ValueError("bad")):
        assert (
            validator_pipeline._executable_rule_facts(
                "a: 1\n", scan_memo.ParseOutcome(None, skipped)
            )
            == ()
        )

    # End to end: the memoized scan raises what the unmemoized scan raises.
    root = _checkout(tmp_path)

    def exploding_load(content, *args, **kwargs):
        raise error

    monkeypatch.setattr(scan_memo, "_safe_load_with_node", exploding_load)
    monkeypatch.setattr(validator_pipeline.yaml, "safe_load", exploding_load)
    for memo in (True, False):
        with pytest.raises(type(error), match=str(error)):
            _executable_index(monkeypatch, (root,), memo=memo)
    assert scan_memo_stats()["rulespec-executable-rule-facts"]["entries"] == 0


def test_concept_facts_keep_the_docstring_fallback():
    content = '"""The "canonical docstring concept" means the condition."""\nx: 1\n'
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(content)
    expected_source = dependency_stubs._extract_embedded_source_text(content)
    expected_terms = dependency_stubs._extract_defined_concept_terms_from_source(
        expected_source
    )
    assert expected_source == 'The "canonical docstring concept" means the condition.'
    assert expected_terms == ["canonical docstring concept"]
    facts = dependency_stubs._concept_file_facts(content)
    assert facts is not None
    assert facts.source_text == expected_source
    assert list(facts.terms) == expected_terms
    assert facts.principal is None
    assert (
        list(dependency_stubs._embedded_source_concept_terms_memoized(content))
        == expected_terms
    )

    # A module with neither a summary nor a docstring has no source text.
    bare = "format: rulespec/v1\nrules: []\n"
    assert dependency_stubs._concept_file_facts(bare).source_text == ""
    assert dependency_stubs._extract_embedded_source_text(bare) == ""


def test_concept_facts_last_complete_rule_is_the_principal(monkeypatch, tmp_path):
    root = _checkout(tmp_path)
    content = (
        _module("canonical gamma concept", "first_symbol") + "  - name: second_symbol\n"
        "    kind: derived\n"
        "    entity: Household\n"
        "    period: Month\n"
        "    dtype: Money\n"
        "  - name: incomplete_symbol\n"
        "    kind: derived\n"
    )
    (root / "statutes" / "gamma.yaml").write_text(content, encoding="utf-8")
    expected = ("second_symbol", "Household", "Month", "Money")
    assert dependency_stubs._extract_principal_symbol_metadata(content) == expected
    assert dependency_stubs._concept_file_facts(content).principal == expected

    concepts = _assert_memo_matches_unmemoized(monkeypatch, root)
    gamma = [c for c in concepts if c.term == "canonical gamma concept"]
    assert len(gamma) == 1
    assert gamma[0].import_target.endswith("#second_symbol")


@pytest.mark.parametrize("name", ["config", "config.worktree"])
def test_identity_memo_refuses_a_symlinked_config_under_the_supervisor(
    monkeypatch, tmp_path, name
):
    checkout = _checkout(tmp_path, git=True).parent
    config = checkout / ".git" / name
    outside = tmp_path / f"outside-{name}"
    if config.exists():
        config.rename(outside)
    else:
        outside.write_text("")
    config.symlink_to(outside)
    _supervisor_git_environment(monkeypatch)
    assert repo_routing._memo_git_config_inputs(checkout) is None
    calls = _count_git(monkeypatch)
    with _checkout_identity_memo_scope():
        results = [inspect_canonical_rulespec_checkout(checkout) for _ in range(3)]
        assert repo_routing._CHECKOUT_IDENTITY_MEMO.get() == {}
    # Nothing is kept: every inspection runs its own probes.
    assert len(calls) == 3 * (len(calls) // 3) > 0
    assert results[0] == results[1] == results[2]
    assert results[0] == repo_routing._inspect_canonical_rulespec_checkout_uncached(
        checkout
    )


@pytest.mark.parametrize(
    ("kind", "template"),
    [("remotes", "URL: {url}\n"), ("branches", "{url}\n")],
)
def test_identity_memo_sees_a_changed_legacy_remote_file(
    monkeypatch, tmp_path, kind, template
):
    checkout = _checkout(tmp_path, git=True).parent
    subprocess.run(["git", "remote", "remove", "origin"], cwd=checkout, check=True)
    legacy = checkout / ".git" / kind / "origin"
    legacy.parent.mkdir(exist_ok=True)
    legacy.write_text(template.format(url=ORIGIN))
    inputs = repo_routing._git_identity_input_paths(
        checkout, git_boundary=checkout, git_config_inputs=()
    )
    assert legacy in inputs
    assert legacy.parent in inputs
    with _checkout_identity_memo_scope():
        assert inspect_canonical_rulespec_checkout(checkout).name == "rulespec-us"
        assert len(repo_routing._CHECKOUT_IDENTITY_MEMO.get()) == 1
        # Same size, same directory entry: only the file's contents and
        # times change.
        other = ORIGIN.replace("rulespec-us", "rulespec-xx")
        legacy.write_text(template.format(url=other))
        second = inspect_canonical_rulespec_checkout(checkout)
    assert second.name is None
    assert second == repo_routing._inspect_canonical_rulespec_checkout_uncached(
        checkout
    )


@pytest.mark.parametrize("kind", ["remotes", "branches"])
def test_identity_memo_sees_a_new_legacy_remote_file(monkeypatch, tmp_path, kind):
    checkout = _checkout(tmp_path, git=True).parent
    with _checkout_identity_memo_scope():
        first = inspect_canonical_rulespec_checkout(checkout)
        assert first.name == "rulespec-us"
        legacy = checkout / ".git" / kind / "origin"
        legacy.parent.mkdir(exist_ok=True)
        legacy.write_text("URL: https://example.com/other.git\n")
        calls = _count_git(monkeypatch)
        second = inspect_canonical_rulespec_checkout(checkout)
        # The fingerprint changed, so the checkout was probed again.
        assert calls
    assert second == repo_routing._inspect_canonical_rulespec_checkout_uncached(
        checkout
    )


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs a directory the process cannot list",
)
def test_symlink_audit_memo_never_keeps_a_walk_that_skipped_a_directory(tmp_path):
    checkout = _checkout(tmp_path).parent
    locked = checkout / "us" / "statutes"
    locked.chmod(0)
    try:
        with validator_pipeline._symlink_audit_memo_scope():
            # The walk cannot list the directory and skips it, as the
            # uncached audit does. That pass must not be kept.
            _audit(checkout)
            assert validator_pipeline._SYMLINK_AUDIT_MEMO.get() == {}
            locked.chmod(0o755)
            (locked / "link.yaml").symlink_to(tmp_path / "anywhere")
            with pytest.raises(UnsafeRulespecContextPath, match="contains a symlink"):
                _audit(checkout)
    finally:
        locked.chmod(0o755)
    with pytest.raises(UnsafeRulespecContextPath, match="contains a symlink"):
        _audit(checkout)


def test_symlink_audit_memo_does_not_keep_a_walk_that_reported_an_error(
    monkeypatch, tmp_path
):
    checkout = _checkout(tmp_path).parent
    original = validator_pipeline.os.walk

    def walk_with_error(top, *args, onerror=None, **kwargs):
        assert onerror is not None
        yield from original(top, *args, onerror=onerror, **kwargs)
        onerror(PermissionError(13, "Permission denied", str(top)))

    with validator_pipeline._symlink_audit_memo_scope():
        monkeypatch.setattr(validator_pipeline.os, "walk", walk_with_error)
        _audit(checkout)
        assert validator_pipeline._SYMLINK_AUDIT_MEMO.get() == {}
        monkeypatch.setattr(validator_pipeline.os, "walk", original)
        _audit(checkout)
        assert len(validator_pipeline._SYMLINK_AUDIT_MEMO.get()) == 1


# --- review round 3 ---------------------------------------------------------


def _callers(depth: int = 4) -> tuple[str, ...]:
    import sys

    frame = sys._getframe(2)
    names = []
    while frame is not None and len(names) < depth:
        names.append(frame.f_code.co_name)
        frame = frame.f_back
    return tuple(names)


@pytest.mark.parametrize("mode", ["off", "fallback"])
def test_unmemoized_parses_run_at_the_stack_depth_they_had_before_the_memos(
    monkeypatch, tmp_path, mode
):
    """A deep module must raise ``RecursionError`` in the same cases as before.

    The unmemoized ``yaml.safe_load`` calls sit the same number of frames
    below the public functions as they did before the memos existed, with the
    memos off and when a memoized parse fails and the scan falls back.
    """

    root = _checkout(tmp_path)
    chains: set[tuple[str, ...]] = set()
    original = yaml.safe_load

    def recording_load(content, *args, **kwargs):
        chains.add(_callers())
        return original(content, *args, **kwargs)

    monkeypatch.setattr(yaml, "safe_load", recording_load)
    if mode == "fallback":

        def exhausted_load(content):
            raise RecursionError("maximum recursion depth exceeded")

        monkeypatch.setattr(scan_memo, "_safe_load_with_node", exhausted_load)
    memo = mode == "fallback"

    _executable_index(monkeypatch, (root,), memo=memo)
    assert {chain[0] for chain in chains} == {"_rulespec_executable_index_for_roots"}

    chains.clear()
    _resolve(monkeypatch, root, memo=memo)
    assert {chain[:3] for chain in chains} == {
        (
            "_extract_embedded_source_text",
            "_build_canonical_concept_candidate",
            "resolve_canonical_concepts_from_text",
        ),
        (
            "_extract_principal_symbol_metadata",
            "_build_canonical_concept_candidate",
            "resolve_canonical_concepts_from_text",
        ),
        (
            "_extract_embedded_source_text",
            "resolve_canonical_concepts_from_text",
            "_resolve",
        ),
    }


def test_first_eviction_logs_one_warning(caplog):
    register_text_facts("test-eviction-log-facts", _probe_facts, max_chars=40)
    with caplog.at_level("WARNING", logger="axiom_encode.scan_memo"):
        text_facts("a: 11111\n", "test-eviction-log-facts")
        assert scan_memo_stats()["test-eviction-log-facts"]["evictions"] == 0
        for index in range(5):
            text_facts(f"b: 2222{index}\n", "test-eviction-log-facts")
    # Every registered memo sees each text; look at this memo's lines only.
    messages = [
        record.getMessage()
        for record in caplog.records
        if "test-eviction-log-facts" in record.getMessage()
    ]
    assert len(messages) == 1
    assert "40 characters" in messages[0]
    assert scan_memo_stats()["test-eviction-log-facts"]["evictions"] == 5
    # A clear starts a new count, so the next eviction is logged again.
    clear_scan_memos()
    caplog.clear()
    with caplog.at_level("WARNING", logger="axiom_encode.scan_memo"):
        text_facts("a: 11111\n", "test-eviction-log-facts")
        text_facts("b: 22222\n", "test-eviction-log-facts")
    assert sum("test-eviction-log-facts" in r.getMessage() for r in caplog.records) == 1


def test_identity_memo_does_not_keep_an_inspection_that_raced_a_change(
    monkeypatch, tmp_path
):
    checkout = _checkout(tmp_path, git=True).parent
    original = repo_routing._inspect_canonical_rulespec_checkout_routed
    changes = iter(range(100))

    def inspect_then_change(path, **kwargs):
        inspection = original(path, **kwargs)
        # The checkout changes after the probe and before the second
        # fingerprint: a new root entry changes the directory stamp.
        (checkout / f"added-{next(changes)}.txt").write_text("x")
        return inspection

    monkeypatch.setattr(
        repo_routing,
        "_inspect_canonical_rulespec_checkout_routed",
        inspect_then_change,
    )
    with _checkout_identity_memo_scope():
        for _ in range(2):
            assert inspect_canonical_rulespec_checkout(checkout).name == "rulespec-us"
            assert repo_routing._CHECKOUT_IDENTITY_MEMO.get() == {}
        monkeypatch.setattr(
            repo_routing, "_inspect_canonical_rulespec_checkout_routed", original
        )
        assert inspect_canonical_rulespec_checkout(checkout).name == "rulespec-us"
        assert len(repo_routing._CHECKOUT_IDENTITY_MEMO.get()) == 1


@pytest.mark.parametrize("category", ["timeout", "git-config-probe-failed"])
def test_identity_memo_refuses_a_failed_config_probe_under_the_supervisor(
    monkeypatch, tmp_path, category
):
    checkout = _checkout(tmp_path, git=True).parent
    _supervisor_git_environment(monkeypatch)
    # Only the custom-environment refusal leads to the repository-only path.
    assert repo_routing._memo_git_config_inputs(checkout) == ()

    def failing_config_inputs(root):
        raise repo_routing._GitProbeError("probe failed", category=category)

    monkeypatch.setattr(repo_routing, "_git_config_input_paths", failing_config_inputs)
    assert repo_routing._memo_git_config_inputs(checkout) is None
    with _checkout_identity_memo_scope():
        inspect_canonical_rulespec_checkout(checkout)
        assert repo_routing._CHECKOUT_IDENTITY_MEMO.get() == {}


def test_symlink_audit_memo_does_not_keep_a_walk_that_raced_a_change(
    monkeypatch, tmp_path
):
    checkout = _checkout(tmp_path).parent
    original = validator_pipeline.os.walk

    def walk_then_change(top, *args, **kwargs):
        yield from original(top, *args, **kwargs)
        # A regular file appears in a directory the walk already left.
        (Path(top) / "us" / "statutes" / "late.yaml").write_text("a: 1\n")

    with validator_pipeline._symlink_audit_memo_scope():
        monkeypatch.setattr(validator_pipeline.os, "walk", walk_then_change)
        _audit(checkout)
        assert validator_pipeline._SYMLINK_AUDIT_MEMO.get() == {}
        monkeypatch.setattr(validator_pipeline.os, "walk", original)
        _audit(checkout)
        assert len(validator_pipeline._SYMLINK_AUDIT_MEMO.get()) == 1


def test_symlink_audit_memo_keys_on_the_resolved_checkout(monkeypatch, tmp_path):
    checkout = _checkout(tmp_path).parent
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(tmp_path, target_is_directory=True)
    alias = alias_parent / checkout.name
    assert alias != checkout
    assert alias.resolve() == checkout.resolve()
    walks = _audit_walks(monkeypatch)
    with validator_pipeline._symlink_audit_memo_scope():
        _audit(alias)
        _audit(checkout)
        _audit(alias)
        memo = validator_pipeline._SYMLINK_AUDIT_MEMO.get()
        assert list(memo) == [checkout.resolve()]
        assert len(walks) == 1
        # A symlink added through either spelling is seen through the other.
        (checkout / "us" / "link.yaml").symlink_to(tmp_path / "anywhere")
        with pytest.raises(UnsafeRulespecContextPath, match="contains a symlink"):
            _audit(alias)
