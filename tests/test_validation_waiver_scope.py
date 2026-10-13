"""Conservative incremental waiver selection and validation read dependencies."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from axiom_encode.harness.validator_pipeline import _resolve_rulespec_import_file_static
from axiom_encode.validation_waiver_scope import (
    select_validation_waiver_scope,
    validation_waiver_read_set,
)


def _module(root: Path, relative: str, *, imports: tuple[str, ...] = ()) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "format: rulespec/v1\n"
    if imports:
        content += "imports:\n" + "".join(f"  - {target}\n" for target in imports)
    path.write_text(content + "rules: []\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("candidate_index", [0, 1], ids=["direct", "relative"])
def test_unprefixed_import_resolver_preserves_unsafe_candidate_behavior(
    tmp_path: Path, candidate_index: int
):
    root = tmp_path / "rulespec-us"
    module = _module(root, "us-ca/statutes/section/main.yaml")
    content_root = root / "us-ca"
    candidate_root = content_root if candidate_index == 0 else module.parent
    invalid = candidate_root / "statutes/target.yaml"
    invalid.mkdir(parents=True)

    def resolve():
        return _resolve_rulespec_import_file_static(
            "statutes/target#value",
            rules_file=module,
            policy_repo_path=content_root,
        )

    if candidate_index == 0:
        with pytest.raises(ValueError, match="regular file"):
            resolve()
    else:
        assert resolve() is None


@pytest.mark.parametrize(
    "changed",
    [
        "known-validation-gaps.yaml",
        ".axiom/toolchain.toml",
        ".axiom/corpus.lock",
        ".axiom/dependencies/rulespec-us/README.md",
        ".axiom/README.md",
        ".github/workflows/validate.yml",
        ".github/workflows/README.md",
        "corpus.lock",
        "corpus-lock.json",
        "programs/us/snap/program.yaml",
        "programs/us/snap/README.md",
        "us/programs/snap/program.yaml",
        "unknown.yaml",
        "us/statutes/module.yml",
        "us/statutes/module.source.json",
        "us-ca/README.md",
        "uk/statutes/new.yaml",
        "uk/statutes/new.test.yaml",
        "uk/README.md",
        "scripts/validate.py",
    ],
)
def test_unclassified_or_shared_change_falls_back_to_all_waivers(
    tmp_path: Path,
    changed: str,
):
    root = tmp_path / "rulespec-us"
    paths = ("us-ca/statutes/one.yaml", "us-ny/statutes/two.yaml")
    for path in paths:
        _module(root, path)

    selection = select_validation_waiver_scope(
        paths,
        root=root,
        changed_paths={changed},
    )

    assert selection.scope == "full"
    assert selection.paths == paths
    assert selection.fallback_reason
    assert changed in selection.fallback_reason
    assert set(selection.reasons) == set(paths)
    assert all(selection.reasons.values())


@pytest.mark.parametrize("changed", [set(), {"README.md"}, {"docs/guide.md"}])
def test_inert_docs_and_empty_changes_select_nothing(tmp_path: Path, changed: set[str]):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/one.yaml"
    _module(root, path)

    selection = select_validation_waiver_scope(
        [path], root=root, changed_paths=changed
    )

    assert selection.scope == "incremental"
    assert selection.paths == ()
    assert selection.reasons == {}
    assert selection.fallback_reason is None


@pytest.mark.parametrize("input_name", ["corpus", "engine"])
@pytest.mark.parametrize("input_present", [True, False])
def test_documentation_under_configured_execution_directory_falls_back(
    tmp_path: Path, input_name: str, input_present: bool
):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/waived.yaml"
    _module(root, path)
    shared_directory = root / "docs" / input_name
    changed = f"docs/{input_name}/guide.md"
    if input_present:
        shared_directory.mkdir(parents=True)
        (root / changed).write_text("# Execution input\n", encoding="utf-8")

    selection = select_validation_waiver_scope(
        [path],
        root=root,
        changed_paths={changed},
        shared_execution_inputs=(shared_directory,),
    )

    assert selection.scope == "full"
    assert selection.paths == (path,)
    assert selection.fallback_reason == f"configured execution input changed: {changed}"


@pytest.mark.parametrize("composer_present", [True, False])
def test_markdown_composer_change_falls_back_even_after_deletion(
    tmp_path: Path, composer_present: bool
):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/waived.yaml"
    _module(root, path)
    changed = "docs/compose.md"
    composer = root / changed
    if composer_present:
        composer.parent.mkdir()
        composer.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        composer.chmod(0o700)

    selection = select_validation_waiver_scope(
        [path],
        root=root,
        changed_paths={changed},
        shared_execution_inputs=(composer,),
    )

    assert selection.scope == "full"
    assert selection.paths == (path,)
    assert selection.fallback_reason == f"configured execution input changed: {changed}"


def test_external_execution_input_alias_resolving_inside_checkout_falls_back(
    tmp_path: Path,
):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/waived.yaml"
    _module(root, path)
    changed = "docs/compose.md"
    composer = root / changed
    composer.parent.mkdir()
    composer.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    composer.chmod(0o700)
    alias = tmp_path / "external-composer"
    alias.symlink_to(composer)

    selection = select_validation_waiver_scope(
        [path],
        root=root,
        changed_paths={changed},
        shared_execution_inputs=(alias,),
    )

    assert selection.scope == "full"
    assert selection.fallback_reason == f"configured execution input changed: {changed}"


def test_disjoint_and_absent_execution_inputs_preserve_inert_document_selection(
    tmp_path: Path,
):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/waived.yaml"
    _module(root, path)

    selection = select_validation_waiver_scope(
        [path],
        root=root,
        changed_paths={"docs/guide.md"},
        shared_execution_inputs=(
            tmp_path / "corpus",
            tmp_path / "engine",
            tmp_path / "compose.md",
            None,
            Mock(),
        ),
    )

    assert selection.scope == "incremental"
    assert selection.paths == ()
    assert selection.fallback_reason is None


def test_literal_tilde_composer_path_retains_composer_resolution_semantics(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/waived.yaml"
    _module(root, path)
    changed = "docs/~/compose.md"
    composer = root / changed
    composer.parent.mkdir(parents=True)
    composer.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    composer.chmod(0o700)
    monkeypatch.chdir(root / "docs")

    selection = select_validation_waiver_scope(
        [path],
        root=root,
        changed_paths={changed},
        shared_execution_inputs=(Path("~/compose.md"),),
    )

    assert selection.scope == "full"
    assert selection.fallback_reason == f"configured execution input changed: {changed}"


def test_markdown_hardlink_to_external_execution_input_falls_back(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/waived.yaml"
    _module(root, path)
    external = tmp_path / "compose"
    external.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    external.chmod(0o700)
    changed = "docs/guide.md"
    documentation = root / changed
    documentation.parent.mkdir()
    os.link(external, documentation)

    selection = select_validation_waiver_scope(
        [path],
        root=root,
        changed_paths={changed},
        shared_execution_inputs=(external,),
    )

    assert selection.scope == "full"
    assert selection.paths == (path,)
    assert selection.fallback_reason == f"changed path has multiple hard links: {changed}"


def test_changed_module_selects_its_jurisdiction_but_not_sibling(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    changed = "us-ca/statutes/changed.yaml"
    peer = "us-ca/regulations/peer.yaml"
    sibling = "us-ny/statutes/sibling.yaml"
    for path in (changed, peer, sibling):
        _module(root, path)

    selection = select_validation_waiver_scope(
        [sibling, peer, changed], root=root, changed_paths={changed}
    )

    assert selection.scope == "incremental"
    assert set(selection.paths) == {changed, peer}
    assert changed in "; ".join(selection.reasons[changed])
    assert changed in "; ".join(selection.reasons[peer])
    assert set(selection.reasons) == set(selection.paths)


@pytest.mark.parametrize("present", [True, False])
def test_companion_presence_and_deletion_select_its_module_only(
    tmp_path: Path, present: bool
):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/one.yaml"
    peer = "us-ca/statutes/two.yaml"
    _module(root, path)
    _module(root, peer)
    companion = "us-ca/statutes/one.test.yaml"
    if present:
        (root / companion).write_text("cases: []\n", encoding="utf-8")

    assert companion in validation_waiver_read_set(path, root=root)
    selection = select_validation_waiver_scope(
        [path, peer], root=root, changed_paths={companion}
    )

    assert selection.scope == "incremental"
    assert selection.paths == (path,)
    assert companion in "; ".join(selection.reasons[path])


@pytest.mark.parametrize(
    "reference_location",
    ["output", "input", "relation", "nested_table"],
)
@pytest.mark.parametrize("target_present", [True, False])
def test_companion_references_include_unimported_module_candidates(
    tmp_path: Path, reference_location: str, target_present: bool
):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/waived.yaml"
    peer = "us-ca/statutes/peer.yaml"
    referenced = "us-ny/statutes/referenced.yaml"
    for candidate in (path, peer, referenced):
        _module(root, candidate)
    if not target_present:
        (root / referenced).unlink()
    target = "us-ny:statutes/referenced"
    references = {
        "output": f"  output:\n    {target}#amount: 1\n",
        "input": f"  input:\n    {target}#input.amount: 1\n",
        "relation": f"  input:\n    {target}#relation.members: []\n",
        "nested_table": (
            "  tables:\n    Household:\n      - "
            f"{target}#input.amount: 1\n"
        ),
    }
    companion = root / path.removesuffix(".yaml")
    companion.with_suffix(".test.yaml").write_text(
        "- name: reference\n  period: 2026-01\n" + references[reference_location],
        encoding="utf-8",
    )

    read_set = validation_waiver_read_set(path, root=root)
    assert referenced in read_set
    selection = select_validation_waiver_scope(
        [path, peer], root=root, changed_paths={referenced}
    )

    assert selection.scope == "incremental"
    assert selection.paths == (path,)
    assert selection.reasons[path] == [
        f"validation reads changed path: {referenced}"
    ]


@pytest.mark.parametrize("companion_present", [True, False])
def test_orphan_companion_change_falls_back(
    tmp_path: Path, companion_present: bool
):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/waived.yaml"
    _module(root, path)
    changed = "us-ny/statutes/orphan.test.yaml"
    if companion_present:
        companion = root / changed
        companion.parent.mkdir(parents=True)
        companion.write_text("cases: []\n", encoding="utf-8")

    selection = select_validation_waiver_scope(
        [path], root=root, changed_paths={changed}
    )

    assert selection.scope == "full"
    assert selection.paths == (path,)
    assert changed in selection.fallback_reason


def test_companion_next_to_non_rulespec_sidecar_falls_back(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/waived.yaml"
    _module(root, path)
    _module(root, "us-ny/statutes/sidecar.yaml").write_text(
        "source: us/statute/example\n", encoding="utf-8"
    )
    changed = "us-ny/statutes/sidecar.test.yaml"
    (root / changed).write_text("cases: []\n", encoding="utf-8")

    selection = select_validation_waiver_scope(
        [path], root=root, changed_paths={changed}
    )

    assert selection.scope == "full"
    assert selection.paths == (path,)
    assert changed in selection.fallback_reason


def test_deleted_waived_module_remains_selected(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/deleted.yaml"
    sibling = "us-ny/statutes/sibling.yaml"
    _module(root, sibling)

    selection = select_validation_waiver_scope(
        [path, sibling], root=root, changed_paths={path}
    )

    assert selection.scope == "incremental"
    assert selection.paths == (path,)


def test_transitive_imports_are_resolved_across_jurisdictions(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    importer = "us-ca/statutes/importer.yaml"
    outer = "us-ny/statutes/outer.yaml"
    terminal = "us-nj/statutes/terminal.yaml"
    unrelated = "us-tx/statutes/unrelated.yaml"
    _module(root, importer, imports=("us-ny:statutes/outer#outer",))
    _module(root, outer, imports=("us-nj:statutes/terminal#terminal",))
    _module(root, terminal)
    _module(root, unrelated)

    read_set = validation_waiver_read_set(importer, root=root)
    assert {importer, outer, terminal} <= read_set
    selection = select_validation_waiver_scope(
        [importer, unrelated], root=root, changed_paths={terminal}
    )

    assert selection.scope == "incremental"
    assert selection.paths == (importer,)
    assert terminal in "; ".join(selection.reasons[importer])


@pytest.mark.parametrize("terminal_present", [True, False])
def test_transitive_unprefixed_import_uses_imported_module_jurisdiction(
    tmp_path: Path, terminal_present: bool
):
    root = tmp_path / "rulespec-us"
    importer = "us-ca/statutes/importer.yaml"
    outer = "us-ny/statutes/outer.yaml"
    terminal = "us-ny/statutes/terminal.yaml"
    caller_shadow = "us-ca/statutes/terminal.yaml"
    _module(root, importer, imports=("us-ny:statutes/outer#outer",))
    _module(root, outer, imports=("statutes/terminal#terminal",))
    # Some static checks retain the caller context, while engine recursion
    # uses the imported module's own jurisdiction. Cover both candidates.
    _module(root, caller_shadow)
    if terminal_present:
        _module(root, terminal)

    read_set = validation_waiver_read_set(importer, root=root)
    assert {outer, terminal, caller_shadow} <= read_set
    selection = select_validation_waiver_scope(
        [importer], root=root, changed_paths={terminal}
    )

    assert selection.scope == "incremental"
    assert selection.paths == (importer,)
    assert terminal in "; ".join(selection.reasons[importer])


@pytest.mark.parametrize("reference_kind", ["proof", "source_relation"])
def test_non_import_validation_targets_are_in_read_set(
    tmp_path: Path, reference_kind: str
):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/waived.yaml"
    target = "us-ny/statutes/target.yaml"
    module = _module(root, path)
    _module(root, target)
    reference = "us-ny:statutes/target#value"
    if reference_kind == "proof":
        content = (
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: amount\n"
            "    kind: parameter\n"
            "    dtype: Money\n"
            "    metadata:\n"
            "      proof:\n"
            "        atoms:\n"
            "          - path: versions[0].formula\n"
            "            kind: import\n"
            "            import:\n"
            f"              target: {reference}\n"
            "              output: value\n"
            "              hash: sha256:abc\n"
            "    versions:\n"
            "      - effective_from: '2026-01-01'\n"
            "        formula: value\n"
        )
    else:
        content = (
            "format: rulespec/v1\n"
            "rules:\n"
            "  - name: restated_value\n"
            "    kind: source_relation\n"
            "    source_relation:\n"
            "      type: restates\n"
            f"      target: {reference}\n"
        )
    module.write_text(content, encoding="utf-8")

    assert target in validation_waiver_read_set(path, root=root)
    selection = select_validation_waiver_scope(
        [path], root=root, changed_paths={target}
    )

    assert selection.scope == "incremental"
    assert selection.paths == (path,)


@pytest.mark.parametrize("rename", [False, True])
def test_missing_import_target_remains_in_head_read_set(tmp_path: Path, rename: bool):
    root = tmp_path / "rulespec-us"
    importer = "us-ca/statutes/importer.yaml"
    deleted = "us-ny/statutes/deleted.yaml"
    _module(root, importer, imports=("us-ny:statutes/deleted#value",))
    changed = {deleted}
    if rename:
        renamed = "us-ny/statutes/renamed.yaml"
        _module(root, renamed)
        changed.add(renamed)

    assert deleted in validation_waiver_read_set(importer, root=root)
    selection = select_validation_waiver_scope(
        [importer], root=root, changed_paths=changed
    )

    assert selection.scope == "incremental"
    assert selection.paths == (importer,)


def test_upstream_placement_reads_include_unimported_ancestor_and_local_module(
    tmp_path: Path,
):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/waived.yaml"
    federal = "us/policies/federal.yaml"
    local = "us-ca/regulations/local.yaml"
    sibling = "us-ny/statutes/sibling.yaml"
    for relative in (path, federal, local, sibling):
        _module(root, relative)

    read_set = validation_waiver_read_set(path, root=root)

    assert {path, federal, local} <= read_set
    assert sibling not in read_set
    for changed in (federal, local):
        selection = select_validation_waiver_scope(
            [path], root=root, changed_paths={changed}
        )
        assert selection.scope == "incremental"
        assert selection.paths == (path,)


def test_deleted_unimported_ancestor_still_selects_descendants(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/waived.yaml"
    _module(root, path)

    selection = select_validation_waiver_scope(
        [path], root=root, changed_paths={"us/policies/deleted.yaml"}
    )

    assert selection.scope == "incremental"
    assert selection.paths == (path,)


@pytest.mark.parametrize("changed", ["docs/guide.md", "us-ca/statutes/alias.yaml"])
def test_symlinked_changed_path_is_not_proven_inert_or_a_module(
    tmp_path: Path, changed: str
):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/waived.yaml"
    target = _module(root, path)
    alias = root / changed
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(target)

    selection = select_validation_waiver_scope(
        [path], root=root, changed_paths={changed}
    )

    assert selection.scope == "full"
    assert selection.paths == (path,)
    assert selection.fallback_reason


def test_directory_with_module_suffix_falls_back(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/waived.yaml"
    _module(root, path)
    changed = "us-ca/statutes/directory.yaml"
    (root / changed).mkdir()

    selection = select_validation_waiver_scope(
        [path], root=root, changed_paths={changed}
    )

    assert selection.scope == "full"
    assert selection.paths == (path,)
    assert selection.fallback_reason


@pytest.mark.parametrize(
    "content",
    [
        "source: us/statute/example\n",
        "format: rulespec/v1\nrules: [\n",
        "format: rulespec/v1\nformat: rulespec/v1\nrules: []\n",
    ],
)
def test_changed_atomic_path_with_unclassifiable_content_falls_back(
    tmp_path: Path, content: str
):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/waived.yaml"
    _module(root, path)
    changed = "us-ny/statutes/sidecar.yaml"
    _module(root, changed).write_text(content, encoding="utf-8")

    selection = select_validation_waiver_scope(
        [path], root=root, changed_paths={changed}
    )

    assert selection.scope == "full"
    assert selection.paths == (path,)
    assert changed in selection.fallback_reason


@pytest.mark.parametrize(
    "import_block",
    [
        "imports:\n  - statutes/terminal.yaml#value\n",
        "imports:\n  - data/terminal#value\n",
        "imports:\n  - target: us-ny:statutes/terminal#value\n    symbols: [value]\n",
    ],
)
def test_unclassifiable_imports_fall_back_instead_of_omitting_dependencies(
    tmp_path: Path, import_block: str
):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/waived.yaml"
    _module(root, path).write_text(
        f"format: rulespec/v1\n{import_block}rules: []\n", encoding="utf-8"
    )
    changed = "us-ny/statutes/terminal.yaml"
    _module(root, changed)

    selection = select_validation_waiver_scope(
        [path], root=root, changed_paths={changed}
    )

    assert selection.scope == "full"
    assert selection.paths == (path,)
    assert selection.fallback_reason


@pytest.mark.parametrize("changed", ["README.md", "docs/guide.md"])
def test_docs_edit_falls_back_when_waiver_imports_cannot_be_classified(
    tmp_path: Path, changed: str
):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/waived.yaml"
    _module(root, path, imports=("data/terminal#value",))
    document = root / changed
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text("# Documentation\n", encoding="utf-8")

    selection = select_validation_waiver_scope(
        [path], root=root, changed_paths={changed}
    )

    assert selection.scope == "full"
    assert selection.paths == (path,)
    assert selection.fallback_reason


@pytest.mark.parametrize("companion", [False, True])
def test_present_foreign_country_module_change_falls_back_for_checkout_admission(
    tmp_path: Path, companion: bool
):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/waived.yaml"
    _module(root, path)
    foreign = "uk/statutes/new.yaml"
    _module(root, foreign)
    changed = foreign
    if companion:
        changed = "uk/statutes/new.test.yaml"
        (root / changed).write_text("cases: []\n", encoding="utf-8")

    selection = select_validation_waiver_scope(
        [path], root=root, changed_paths={changed}
    )

    assert selection.scope == "full"
    assert selection.paths == (path,)
    assert selection.fallback_reason


@pytest.mark.parametrize(
    ("include_present", "conditional"), [(True, False), (False, False), (True, True)]
)
def test_markdown_git_config_include_is_a_shared_validation_input(
    tmp_path: Path, include_present: bool, conditional: bool
):
    root = tmp_path / "rulespec-us"
    path = "us-ca/statutes/waived.yaml"
    _module(root, path)
    subprocess.run(
        ["git", "init", "--quiet", str(root)],
        check=True,
        capture_output=True,
        timeout=15,
    )
    changed = "docs/route.md"
    include_file = root / changed
    include_file.parent.mkdir()
    if include_present:
        include_file.write_text(
            '[remote "origin"]\n'
            "  url = https://github.com/TheAxiomFoundation/rulespec-us.git\n",
            encoding="utf-8",
        )
    config = root / ".git/config"
    section = '[includeIf "gitdir:/never-matches/"]' if conditional else "[include]"
    config.write_text(
        config.read_text(encoding="utf-8")
        + f"\n{section}\n  path = ../{changed}\n",
        encoding="utf-8",
    )

    assert changed in validation_waiver_read_set(path, root=root)
    selection = select_validation_waiver_scope(
        [path], root=root, changed_paths={changed}
    )

    assert selection.scope == "full"
    assert selection.paths == (path,)
    assert changed in selection.fallback_reason
