"""Incremental waiver scope for signed encoder pull requests.

A signed pull request adds a module, its companion test and one applied
manifest under ``.axiom/encoding-manifests``. These tests pin which manifest
paths are classified, which paths still force the full audit, that one waiver's
unreadable closure selects that waiver and no other, and that the selector's
work follows the waivers it is given.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest

from axiom_encode import cli
from axiom_encode import validation_waiver_scope as scope
from axiom_encode.harness.validator_pipeline import (
    _extract_import_items_from_content,
    _parse_rulespec_target,
    _resolve_rulespec_import_file_static,
    _resolve_rulespec_target_file,
    _rulespec_import_file_candidates_static,
    _rulespec_resolution_cache_scope,
    _rulespec_target_file_candidates,
    _safe_load_unique_keys,
)
from axiom_encode.repo_routing import (
    _rulespec_routing_cache_scope,
    find_policy_repo_root,
)
from axiom_encode.validation_waiver_scope import (
    _applied_manifest_module,
    _changed_path_kind,
    _PathAliases,
    _ReadSets,
    select_validation_waiver_scope,
)
from tests.test_validation_waiver_scope import _module

_MANIFESTS = ".axiom/encoding-manifests"


def _manifest(root: Path, module: str) -> str:
    relative = f"{_MANIFESTS}/{module.removesuffix('.yaml')}.json"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"schema": "axiom-encode/applied-manifest"}\n', encoding="utf-8")
    return relative


def _companion(root: Path, module: str, text: str = "[]\n") -> str:
    relative = module.removesuffix(".yaml") + ".test.yaml"
    (root / relative).write_text(text, encoding="utf-8")
    return relative


def _composition(root: Path, relative: str) -> Path:
    path = _module(root, relative)
    path.write_text(
        "format: rulespec/v1\nmodule:\n  kind: composition\nrules: []\n",
        encoding="utf-8",
    )
    return path


def _signed_change(root: Path, module: str) -> set[str]:
    _module(root, module)
    return {module, _companion(root, module), _manifest(root, module)}


def test_manifest_directory_is_the_one_the_signed_apply_writer_uses():
    assert scope._APPLIED_MANIFEST_PARTS == cli.APPLIED_ENCODING_MANIFEST_DIR.parts


@pytest.mark.parametrize(
    ("manifest", "module"),
    [
        (
            f"{_MANIFESTS}/us-az/policies/des/faa5/medical/block-7.json",
            "us-az/policies/des/faa5/medical/block-7.yaml",
        ),
        (
            f"{_MANIFESTS}/us/regulations/12-cfr/1006/38.json",
            "us/regulations/12-cfr/1006/38.yaml",
        ),
        (f"{_MANIFESTS}/us/statutes/26/1.json", "us/statutes/26/1.yaml"),
        (f"{_MANIFESTS}/us-ca/legislation/ab-1.json", "us-ca/legislation/ab-1.yaml"),
    ],
)
def test_manifest_of_a_module_path_is_classified_as_that_module(
    manifest: str, module: str
):
    assert _applied_manifest_module(manifest) == module
    assert _changed_path_kind(manifest) == "manifest"


@pytest.mark.parametrize(
    "path",
    [
        # Not a manifest of a module path.
        f"{_MANIFESTS}/us-az/policies/des/block-7.yaml",
        f"{_MANIFESTS}/us-az/policies/des/block-7.json.bak",
        f"{_MANIFESTS}/us-az/policies/des/block-7",
        f"{_MANIFESTS}/us-az/policies/des/block-7.test.json",
        f"{_MANIFESTS}/us-az/policies/des/README.md",
        f"{_MANIFESTS}/us-az/policies.json",
        f"{_MANIFESTS}/us-az.json",
        f"{_MANIFESTS}/index.json",
        ".axiom/encoding-manifests.json",
        # Layouts with no jurisdiction, or with a root that holds no modules.
        f"{_MANIFESTS}/statutes/26/3121/a.json",
        f"{_MANIFESTS}/regulations/42-cfr/435/603.json",
        f"{_MANIFESTS}/programs/us/us-tariff-schedule/ch01.json",
        f"{_MANIFESTS}/us/programs/snap/fy-2026.json",
        f"{_MANIFESTS}/us/data/table.json",
        f"{_MANIFESTS}/US/statutes/26/1.json",
        # A shared surface named inside the module path.
        f"{_MANIFESTS}/us/policies/.axiom/toolchain.json",
        f"{_MANIFESTS}/us/policies/.github/workflow.json",
        f"{_MANIFESTS}/us/policies/programs/snap.json",
        # A manifest directory that is not the repository's own.
        "us-ct/.axiom/encoding-manifests/statutes/12/701.json",
        ".axiom/Encoding-Manifests/us/statutes/26/1.json",
        ".axiom/encoding-manifests-old/us/statutes/26/1.json",
        ".axiom/legacy-replacements/us/statutes/26/1.json",
        "encoding-manifests/us/statutes/26/1.json",
        ".github/encoding-manifests/us/statutes/26/1.json",
        # Toolchain files, ledgers and indexes.
        ".axiom/toolchain.toml",
        ".axiom/workflow-toolchain.toml",
        ".axiom/policyengine-runtime.toml",
        ".axiom/pending-validation-fingerprints.json",
        ".axiom/target-validation-passes.json",
        ".axiom/reviewed-migrations.json",
        ".axiom/repository-structure.yaml",
        ".axiom/upstream-source-check-baseline.txt",
        ".axiom/index/provisions_to_rules.json",
        ".axiom/validation-waiver-audit-costs.json",
        "known-validation-gaps.yaml",
        "oracle-coverage-pending.yaml",
    ],
)
def test_other_axiom_paths_and_malformed_manifests_force_the_full_audit(
    tmp_path: Path, path: str
):
    assert _applied_manifest_module(path) is None
    assert _changed_path_kind(path) is None

    root = tmp_path / "rulespec-us"
    waived = "us-ca/statutes/waived.yaml"
    _module(root, waived)
    _module(root, "us/statutes/26/1.yaml")
    changed = root / path
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_text("{}\n", encoding="utf-8")

    selection = select_validation_waiver_scope(
        [waived], root=root, changed_paths={path}
    )

    assert selection.scope == "full"
    assert selection.paths == (waived,)
    assert selection.fallback_reason == (
        f"unclassified or shared validation input changed: {path}"
    )


@pytest.mark.parametrize(
    "path",
    [
        f"{_MANIFESTS}/us/statutes/../statutes/1.json",
        f"{_MANIFESTS}/us/statutes//1.json",
        f"{_MANIFESTS}/us/statutes/1.json ",
        f"{_MANIFESTS}/us/statutes/a\\b.json",
        f"{_MANIFESTS}/us/statutes/a​b.json",
        f"/{_MANIFESTS}/us/statutes/1.json",
    ],
)
def test_non_canonical_manifest_spellings_are_not_classified(path: str):
    assert _changed_path_kind(path) is None


def test_signed_pull_request_selects_its_jurisdiction_and_no_other(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    peer = "us-az/policies/des/faa5/peer.yaml"
    sibling = "us-ca/statutes/sibling.yaml"
    federal = "us/statutes/federal.yaml"
    for path in (peer, sibling, federal):
        _module(root, path)
    new_module = "us-az/policies/des/faa5/medical/block-7.yaml"
    changed = _signed_change(root, new_module)

    selection = select_validation_waiver_scope(
        [federal, sibling, peer], root=root, changed_paths=changed
    )

    assert selection.scope == "incremental"
    assert selection.fallback_reason is None
    assert selection.paths == (peer,)
    # The module entry covers its manifest: one reason, not two.
    assert selection.reasons[peer] == [
        f"validation searches changed module: {new_module}"
    ]


def test_signed_federal_pull_request_selects_every_jurisdiction(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    waivers = [
        "us-az/policies/des/peer.yaml",
        "us-ca/statutes/sibling.yaml",
        "us/statutes/federal.yaml",
    ]
    for path in waivers:
        _module(root, path)
    changed = _signed_change(root, "us/regulations/12-cfr/1006/38.yaml")

    selection = select_validation_waiver_scope(
        waivers, root=root, changed_paths=changed
    )

    assert selection.scope == "incremental"
    assert set(selection.paths) == set(waivers)


def test_manifest_alone_selects_exactly_what_its_module_selects(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    module = "us-az/policies/des/faa5/medical/block-7.yaml"
    peer = "us-az/policies/des/faa5/peer.yaml"
    importer = "us-ca/statutes/importer.yaml"
    sibling = "us-ca/statutes/sibling.yaml"
    federal = "us/statutes/federal.yaml"
    _module(root, module)
    _module(root, peer)
    _module(root, importer, imports=("us-az:policies/des/faa5/medical/block-7",))
    _module(root, sibling)
    _module(root, federal)
    manifest = _manifest(root, module)
    waivers = [module, peer, importer, sibling, federal]

    by_manifest = select_validation_waiver_scope(
        waivers, root=root, changed_paths={manifest}
    )
    by_module = select_validation_waiver_scope(
        waivers, root=root, changed_paths={module}
    )

    assert by_manifest.scope == by_module.scope == "incremental"
    assert by_manifest.paths == by_module.paths == (module, peer, importer)
    label = f"{module} (applied manifest {manifest})"
    assert by_manifest.reasons == {
        module: [f"module changed: {label}"],
        peer: [f"validation searches changed module: {label}"],
        importer: [f"validation reads changed path: {label}"],
    }


def test_manifest_is_never_inert(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    module = "us-az/policies/des/block-7.yaml"
    _module(root, module)
    manifest = _manifest(root, module)
    readme = root / "README.md"
    readme.write_text("# Notes\n", encoding="utf-8")

    with_manifest = select_validation_waiver_scope(
        [module], root=root, changed_paths={manifest, "README.md"}
    )
    without = select_validation_waiver_scope(
        [module], root=root, changed_paths={"README.md"}
    )

    assert with_manifest.paths == (module,)
    assert without.paths == ()


def test_manifest_without_its_module_forces_the_full_audit(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    waived = "us-ca/statutes/waived.yaml"
    _module(root, waived)
    (root / "us-az/policies/des").mkdir(parents=True)
    manifest = _manifest(root, "us-az/policies/des/block-7.yaml")

    selection = select_validation_waiver_scope(
        [waived], root=root, changed_paths={manifest}
    )

    assert selection.scope == "full"
    assert selection.fallback_reason == (
        f"changed applied manifest has no RuleSpec module: {manifest}"
    )


def test_manifest_of_a_file_that_is_not_a_rulespec_module_forces_the_full_audit(
    tmp_path: Path,
):
    root = tmp_path / "rulespec-us"
    waived = "us-ca/statutes/waived.yaml"
    _module(root, waived)
    module = "us-az/policies/des/block-7.yaml"
    _module(root, module).write_text("source: other\n", encoding="utf-8")
    manifest = _manifest(root, module)

    selection = select_validation_waiver_scope(
        [waived], root=root, changed_paths={manifest}
    )

    assert selection.scope == "full"
    assert selection.fallback_reason == (
        f"changed path has no valid RuleSpec module: {manifest}"
    )


def test_deleted_manifest_forces_the_full_audit(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    module = "us-az/policies/des/block-7.yaml"
    _module(root, module)
    manifest = _manifest(root, module)
    (root / manifest).unlink()

    selection = select_validation_waiver_scope(
        [module], root=root, changed_paths={manifest}
    )

    assert selection.scope == "full"
    assert selection.fallback_reason == (
        f"could not establish changed path identity: {manifest}"
    )


def test_manifest_of_another_country_forces_the_full_audit(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    waived = "us-ca/statutes/waived.yaml"
    _module(root, waived)
    manifest = _manifest(root, "uk/statutes/new.yaml")

    selection = select_validation_waiver_scope(
        [waived], root=root, changed_paths={manifest}
    )

    assert selection.scope == "full"
    assert selection.fallback_reason == (
        f"changed path belongs to another country: {manifest}"
    )


@pytest.mark.parametrize("target", ["manifest", "module"])
@pytest.mark.parametrize("alias", ["symlink", "hard link"])
def test_aliased_manifest_or_module_forces_the_full_audit(
    tmp_path: Path, target: str, alias: str
):
    root = tmp_path / "rulespec-us"
    waived = "us-ca/statutes/waived.yaml"
    _module(root, waived)
    module = "us-az/policies/des/block-7.yaml"
    _module(root, module)
    manifest = _manifest(root, module)
    aliased = root / (manifest if target == "manifest" else module)
    outside = tmp_path / "outside-input"
    if alias == "symlink":
        aliased.replace(outside)
        aliased.symlink_to(outside)
    else:
        os.link(aliased, outside)

    selection = select_validation_waiver_scope(
        [waived], root=root, changed_paths={manifest}
    )

    assert selection.scope == "full"
    assert manifest in selection.fallback_reason


def test_directory_at_a_manifest_path_forces_the_full_audit(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    module = "us-az/policies/des/block-7.yaml"
    _module(root, module)
    manifest = f"{_MANIFESTS}/us-az/policies/des/block-7.json"
    (root / manifest).mkdir(parents=True)

    selection = select_validation_waiver_scope(
        [module], root=root, changed_paths={manifest}
    )

    assert selection.scope == "full"
    assert "unsafe file type" in selection.fallback_reason


def test_manifest_inside_a_configured_execution_input_forces_the_full_audit(
    tmp_path: Path,
):
    root = tmp_path / "rulespec-us"
    module = "us-az/policies/des/block-7.yaml"
    _module(root, module)
    manifest = _manifest(root, module)

    selection = select_validation_waiver_scope(
        [module],
        root=root,
        changed_paths={manifest},
        shared_execution_inputs=(root / ".axiom",),
    )

    assert selection.scope == "full"
    assert selection.fallback_reason == (
        f"configured execution input changed: {manifest}"
    )


def test_any_changed_module_selects_a_composition_in_another_jurisdiction(
    tmp_path: Path,
):
    root = tmp_path / "rulespec-us"
    composition = "us/policies/tariff/composition.yaml"
    atomic = "us/policies/tariff/atomic.yaml"
    _composition(root, composition)
    _module(root, atomic)
    module = "us-az/policies/des/block-7.yaml"
    _module(root, module)
    waivers = [composition, atomic]

    by_module = select_validation_waiver_scope(
        waivers, root=root, changed_paths=_signed_change(root, module)
    )
    by_manifest = select_validation_waiver_scope(
        waivers, root=root, changed_paths={_manifest(root, module)}
    )

    # Composing a program loads every module of the checkout. The atomic
    # federal module searches only its own jurisdiction.
    assert by_module.paths == by_manifest.paths == (composition,)
    assert by_module.reasons[composition] == [
        f"validation reads changed path: {module}"
    ]


def test_companion_only_change_selects_a_composition_only_when_it_is_read(
    tmp_path: Path,
):
    root = tmp_path / "rulespec-us"
    composition = "us/policies/tariff/composition.yaml"
    _composition(root, composition)
    module = "us-az/policies/des/block-7.yaml"
    _module(root, module)
    companion = _companion(root, module)

    unread = select_validation_waiver_scope(
        [composition], root=root, changed_paths={companion}
    )
    # A module anywhere in the country names the companion as a target.
    _module(
        root,
        "us-ca/statutes/names-it.yaml",
        imports=("us-az:policies/des/block-7.test",),
    )
    read = select_validation_waiver_scope(
        [composition], root=root, changed_paths={companion}
    )

    assert unread.scope == read.scope == "incremental"
    assert unread.paths == ()
    assert read.paths == (composition,)
    assert read.reasons[composition] == [f"validation reads changed path: {companion}"]


def test_unreadable_closure_selects_only_the_waivers_that_reach_it(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    broken = "us/statutes/broken.yaml"
    _module(root, broken).write_text(
        "format: rulespec/v1\nrules:\n  - name: a\n    name: b\n", encoding="utf-8"
    )
    direct = "us-ny/statutes/direct.yaml"
    transitive = "us-ny/statutes/transitive.yaml"
    clean = "us-ny/statutes/clean.yaml"
    other = "us-tx/statutes/other.yaml"
    _module(root, direct, imports=("us:statutes/broken",))
    _module(root, transitive, imports=("us-ny:statutes/direct",))
    _module(root, clean)
    _module(root, other)
    (root / "README.md").write_text("# Notes\n", encoding="utf-8")
    waivers = [broken, direct, transitive, clean, other]

    docs_only = select_validation_waiver_scope(
        waivers, root=root, changed_paths={"README.md"}
    )
    changed = _signed_change(root, "us-ny/statutes/new.yaml")
    signed = select_validation_waiver_scope(waivers, root=root, changed_paths=changed)

    assert docs_only.scope == signed.scope == "incremental"
    assert docs_only.paths == (direct, transitive, broken)
    for path in docs_only.paths:
        assert docs_only.reasons[path][0].startswith(
            "validation read set could not be established: "
        )
        assert "duplicate key" in docs_only.reasons[path][0]
        assert "\n" not in docs_only.reasons[path][0]
    assert signed.paths == (clean, direct, transitive, broken)
    assert signed.reasons[clean] == [
        "validation searches changed module: us-ny/statutes/new.yaml"
    ]


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_irregular_dependency_candidate_selects_the_waiver_that_reads_it(
    tmp_path: Path, kind: str
):
    root = tmp_path / "rulespec-us"
    reader = "us-ny/statutes/reader.yaml"
    clean = "us-tx/statutes/clean.yaml"
    _module(root, reader, imports=("us-ca:statutes/target",))
    _module(root, clean)
    candidate = root / "us-ca/statutes/target.yaml"
    candidate.parent.mkdir(parents=True)
    (root / "README.md").write_text("# Notes\n", encoding="utf-8")
    if kind == "directory":
        candidate.mkdir()
        expected = "unsafe RuleSpec dependency candidate: us-ca/statutes/target.yaml"
    else:
        candidate.symlink_to(root / reader)
        expected = "contains a symlink"

    selection = select_validation_waiver_scope(
        [reader, clean], root=root, changed_paths={"README.md"}
    )

    if kind == "directory":
        assert selection.scope == "incremental"
        assert selection.paths == (reader,)
        assert expected in selection.reasons[reader][0]
    else:
        # A symlink anywhere in the checkout is a shared admission failure.
        assert selection.scope == "full"
        assert expected in selection.fallback_reason


def _repository(root: Path) -> list[str]:
    """A small country checkout with cross-jurisdiction and absent targets."""

    _module(root, "us/statutes/base.yaml")
    _module(root, "us/statutes/shared.yaml", imports=("us:statutes/base",))
    _module(root, "us/regulations/absent-importer.yaml", imports=("us:statutes/gone",))
    _module(
        root,
        "us-ca/statutes/a.yaml",
        imports=("us:statutes/shared#value", "statutes/b#value"),
    )
    _module(root, "us-ca/statutes/b.yaml", imports=("us-ny:statutes/n",))
    _module(root, "us-ca/statutes/nested/c.yaml", imports=("statutes/b",))
    _module(root, "us-ny/statutes/n.yaml", imports=("us-ca:statutes/a",))
    _module(root, "us-ny/statutes/alone.yaml")
    _module(root, "us-tx/statutes/t.yaml", imports=("us:regulations/absent-importer",))
    (root / "us-ca/statutes/a.test.yaml").write_text(
        "- name: case\n  input:\n    us-ny:statutes/alone#value: 1\n"
        "  output:\n    us-tx:statutes/missing#value: 2\n",
        encoding="utf-8",
    )
    _composition(root, "us/policies/program/composition.yaml")
    return [
        "us/statutes/base.yaml",
        "us/statutes/shared.yaml",
        "us/regulations/absent-importer.yaml",
        "us-ca/statutes/a.yaml",
        "us-ca/statutes/b.yaml",
        "us-ca/statutes/nested/c.yaml",
        "us-ny/statutes/n.yaml",
        "us-ny/statutes/alone.yaml",
        "us-tx/statutes/t.yaml",
        "us/policies/program/composition.yaml",
    ]


def _resolver_closure(reader: _ReadSets, module: Path, content_root: Path) -> set[Path]:
    """The closure as the validator's own resolvers compute it, uncached.

    This is the algorithm the read set used before it shared a graph: every
    candidate of every import and target, and recursion into the one file the
    validator's resolver returns.
    """

    dependencies: set[Path] = set()
    pending = [module]
    parsed: set[Path] = set()
    roots = reader.dependency_roots
    while pending:
        current = pending.pop()
        dependencies.add(current)
        if current in parsed or not current.is_file():
            continue
        parsed.add(current)
        content = current.read_text(encoding="utf-8")
        payload = _safe_load_unique_keys(content)
        current_root = find_policy_repo_root(current) or content_root
        resolution_roots = tuple(dict.fromkeys((content_root, current_root)))
        imports = _extract_import_items_from_content(content)
        declared = payload.get("imports") if isinstance(payload, dict) else None
        if isinstance(declared, list):
            imports.extend(item for item in declared if isinstance(item, str))
        for imported in dict.fromkeys(imports):
            for resolution_root in resolution_roots:
                dependencies.update(
                    _rulespec_import_file_candidates_static(
                        imported,
                        rules_file=current,
                        policy_repo_path=resolution_root,
                        rulespec_dependency_roots=roots,
                    )
                )
                target = _resolve_rulespec_import_file_static(
                    imported,
                    rules_file=current,
                    policy_repo_path=resolution_root,
                    rulespec_dependency_roots=roots,
                )
                if target is not None:
                    pending.append(target)
        for value in scope._yaml_strings(payload):
            target_ref = _parse_rulespec_target(value)
            if target_ref is None:
                continue
            for resolution_root in resolution_roots:
                dependencies.update(
                    _rulespec_target_file_candidates(
                        target_ref, resolution_root, rulespec_dependency_roots=roots
                    )
                )
                target = _resolve_rulespec_target_file(
                    target_ref, resolution_root, rulespec_dependency_roots=roots
                )
                if target is not None:
                    pending.append(target)
    return dependencies


def test_shared_graph_closure_equals_the_validator_resolver_closure(tmp_path: Path):
    root = (tmp_path / "rulespec-us").resolve()
    waivers = _repository(root)
    with _rulespec_routing_cache_scope(), _rulespec_resolution_cache_scope():
        reader = _ReadSets(root, ())
        for waiver in waivers:
            module = root / waiver
            content_root = root / waiver.split("/")[0]
            starts = [module, module.with_name(f"{module.stem}.test.yaml")]
            for start in starts:
                if not start.is_file():
                    continue
                expected = _resolver_closure(reader, start, content_root)
                actual = reader.closure(start, content_root)
                # Soundness needs a superset. On a tree with one candidate
                # per target the two are the same set.
                assert expected <= actual
                assert expected == actual
    closure = reader.closure(root / "us-ca/statutes/a.yaml", root / "us-ca")
    names = {reader.relative(path) for path in closure}
    assert {
        "us-ca/statutes/a.yaml",
        "us-ca/statutes/b.yaml",
        "us-ny/statutes/n.yaml",
        "us/statutes/shared.yaml",
        "us/statutes/base.yaml",
    } <= names


@pytest.mark.parametrize(
    "imported",
    [
        "us:statutes/base",
        "us:statutes/base#value",
        "'us-ny:statutes/n#value'",
        "us-ca:statutes/nested/c.yaml",
        "us:statutes/gone",
        "us:data/table",
        "us:programs/snap/fy-2026",
        "uk:statutes/foreign",
        "us:/statutes/base",
        "us:statutes/base/",
        "us:",
        "us-ca:statutes//double",
        "US:statutes/base",
        "statutes/b",
        "statutes/b#value",
        "b",
        "data/terminal#value",
        "",
    ],
)
def test_import_candidates_equal_the_validator_resolver_candidates(
    tmp_path: Path, imported: str
):
    root = (tmp_path / "rulespec-us").resolve()
    _repository(root)
    rules_file = root / "us-ca/statutes/nested/c.yaml"
    with _rulespec_routing_cache_scope(), _rulespec_resolution_cache_scope():
        reader = _ReadSets(root, ())
        for resolution_root in (root / "us-ca", root / "us", root / "us-ny"):
            expected = tuple(
                _rulespec_import_file_candidates_static(
                    imported,
                    rules_file=rules_file,
                    policy_repo_path=resolution_root,
                    rulespec_dependency_roots=(),
                )
            )
            actual = reader.import_candidates(
                imported, rules_file=rules_file, resolution_root=resolution_root
            )
            assert actual == expected


@pytest.mark.parametrize(
    "imported", ["us:statutes/../escape", "us:../statutes/base", "us-ca:statutes/a/.."]
)
def test_import_candidates_refuse_what_the_validator_resolver_refuses(
    tmp_path: Path, imported: str
):
    root = (tmp_path / "rulespec-us").resolve()
    _repository(root)
    rules_file = root / "us-ca/statutes/a.yaml"
    with _rulespec_routing_cache_scope(), _rulespec_resolution_cache_scope():
        reader = _ReadSets(root, ())
        with pytest.raises(ValueError, match="outside the active policy root"):
            tuple(
                _rulespec_import_file_candidates_static(
                    imported,
                    rules_file=rules_file,
                    policy_repo_path=root / "us-ca",
                    rulespec_dependency_roots=(),
                )
            )
        with pytest.raises(ValueError, match="outside the active policy root"):
            reader.import_candidates(
                imported, rules_file=rules_file, resolution_root=root / "us-ca"
            )


def test_partition_selection_is_its_share_of_the_whole_ledger_selection(
    tmp_path: Path,
):
    root = tmp_path / "rulespec-us"
    waivers = _repository(root)
    broken = "us-tx/statutes/broken.yaml"
    _module(root, broken, imports=("data/terminal#value",))
    waivers.append(broken)
    change_sets = [
        _signed_change(root, "us-ny/statutes/new.yaml"),
        {"us/statutes/base.yaml"},
        {"us-ca/statutes/a.test.yaml"},
        {"us-ny/statutes/alone.yaml", "us-ca/statutes/b.yaml"},
    ]
    for changed in change_sets:
        whole = select_validation_waiver_scope(
            waivers, root=root, changed_paths=changed
        )
        assert whole.scope == "incremental"
        for partition in (waivers[0::3], waivers[1::3], waivers[2::3], [broken]):
            part = select_validation_waiver_scope(
                partition, root=root, changed_paths=changed
            )
            assert part.scope == "incremental"
            assert set(part.paths) == set(whole.paths) & set(partition)
            assert part.reasons == {path: whole.reasons[path] for path in part.paths}


def _count_calls(monkeypatch, name: str) -> Counter:
    calls: Counter = Counter()
    original = getattr(scope, name)

    def counted(*args, **kwargs):
        calls[args[0] if args and isinstance(args[0], (str, Path)) else len(calls)] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(scope, name, counted)
    return calls


def test_selector_work_follows_the_waivers_given_not_the_ledger(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "rulespec-us"
    # A chain every waiver imports, and many independent waivers.
    _module(root, "us/statutes/base.yaml")
    for index in range(1, 6):
        _module(
            root,
            f"us/statutes/chain-{index}.yaml",
            imports=(
                f"us:statutes/chain-{index - 1}" if index > 1 else "us:statutes/base",
            ),
        )
    waivers = []
    for state in ("us-ca", "us-ny", "us-tx"):
        for index in range(20):
            path = f"{state}/statutes/w-{index}.yaml"
            _module(root, path, imports=("us:statutes/chain-5#value",))
            waivers.append(path)
    changed = _signed_change(root, "us-ny/statutes/new.yaml")
    reads: Counter = Counter()
    original_read = Path.read_text

    def read_text(self, *args, **kwargs):
        if self.suffix == ".yaml":
            reads[self.relative_to(root.resolve()).as_posix()] += 1
        return original_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    root_lookups = _count_calls(monkeypatch, "_candidate_rulespec_repo_roots")
    upstream_lookups = _count_calls(monkeypatch, "_candidate_upstream_rulespec_roots")

    whole = select_validation_waiver_scope(waivers, root=root, changed_paths=changed)

    assert whole.scope == "incremental"
    assert set(whole.paths) == {path for path in waivers if path.startswith("us-ny/")}
    # Every file of the shared chain is parsed once for 60 waivers, each
    # jurisdiction's candidate roots are resolved once, and so are its
    # placement-search roots.
    assert all(reads[f"us/statutes/chain-{index}.yaml"] == 1 for index in range(1, 6))
    assert reads["us/statutes/base.yaml"] == 1
    assert all(reads[path] == 1 for path in waivers)
    assert sum(root_lookups.values()) <= 6
    assert sum(upstream_lookups.values()) == 3
    whole_reads = sum(reads.values())
    assert whole_reads <= len(waivers) + 6 + 3

    reads.clear()
    root_lookups.clear()
    one = select_validation_waiver_scope(
        ["us-ny/statutes/w-0.yaml"], root=root, changed_paths=changed
    )

    assert one.paths == ("us-ny/statutes/w-0.yaml",)
    # One waiver costs its own closure: its module, the chain and the changed
    # module check. It does not read the other 59 waivers.
    assert not any(reads[path] for path in waivers if path != one.paths[0])
    assert sum(reads.values()) <= 1 + 6 + 3
    assert sum(reads.values()) * 5 < whole_reads


def test_alias_keys_decide_exactly_what_same_decides(tmp_path: Path):
    present = tmp_path / "Present.yaml"
    present.write_text("a\n", encoding="utf-8")
    other = tmp_path / "other.yaml"
    other.write_text("b\n", encoding="utf-8")
    linked = tmp_path / "linked.yaml"
    os.link(present, linked)
    candidates = [
        present,
        other,
        linked,
        tmp_path / "present.yaml",
        tmp_path / "PRESENT.YAML",
        tmp_path / "absent.yaml",
        tmp_path / "Absent.yaml",
        tmp_path / "nested" / "present.yaml",
    ]
    aliases = _PathAliases()
    for existing in (present, other, linked):
        keys = aliases.existing_keys(existing)
        for candidate in candidates:
            assert (aliases.key(candidate) in keys) == aliases.same(
                existing, candidate
            ), (existing.name, candidate.name)
    # A case-sensitive filesystem: the lower-case spelling does not exist, and
    # an existing file still matches it by its case-folded name.
    sensitive = _PathAliases()
    lower = tmp_path / "present.yaml"
    sensitive.identities[Path(os.path.abspath(lower))] = None
    assert sensitive.same(present, lower)
    assert sensitive.key(lower) == ("name", os.path.abspath(lower).casefold())
    assert sensitive.key(lower) in sensitive.existing_keys(present)
    assert sensitive.key(lower) not in sensitive.existing_keys(other)
    assert aliases.key(present) != aliases.key(other)
    assert aliases.key(present) == aliases.key(linked)
    assert aliases.key(tmp_path / "absent.yaml") == aliases.key(
        tmp_path / "ABSENT.yaml"
    )


def test_filesystem_failure_while_reading_a_closure_audits_every_path_given(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "rulespec-us"
    waivers = ["us-ny/statutes/one.yaml", "us-tx/statutes/two.yaml"]
    for path in waivers:
        _module(root, path)
    (root / "README.md").write_text("# Notes\n", encoding="utf-8")

    def refuse(self, current):
        raise OSError(24, "Too many open files")

    monkeypatch.setattr(_ReadSets, "parsed", refuse)

    selection = select_validation_waiver_scope(
        waivers, root=root, changed_paths={"README.md"}
    )

    assert selection.scope == "full"
    assert selection.paths == tuple(waivers)
    assert selection.fallback_reason == (
        "could not establish validation read set: [Errno 24] Too many open files"
    )


def test_waiver_of_an_absent_jurisdiction_still_searches_its_ancestors(
    tmp_path: Path,
):
    root = tmp_path / "rulespec-us"
    # A ledger entry can outlive its module and even its jurisdiction
    # directory. Its ancestors remain its placement-search roots.
    gone = "us-zz/statutes/gone.yaml"
    nested_gone = "us-ca-la/statutes/gone.yaml"
    _module(root, "us-ca/statutes/present.yaml")
    federal = "us/statutes/new.yaml"
    _module(root, federal)
    state = "us-ca/statutes/new.yaml"
    _module(root, state)

    by_federal = select_validation_waiver_scope(
        [gone, nested_gone], root=root, changed_paths={federal}
    )
    by_state = select_validation_waiver_scope(
        [gone, nested_gone], root=root, changed_paths={state}
    )

    assert by_federal.scope == by_state.scope == "incremental"
    assert by_federal.paths == (nested_gone, gone)
    assert by_state.paths == (nested_gone,)
    assert by_state.reasons[nested_gone] == [
        f"validation searches changed module: {state}"
    ]


def test_signed_change_selects_a_composition_without_walking_the_country(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "rulespec-us"
    composition = "us/policies/tariff/composition.yaml"
    _composition(root, composition)
    for index in range(12):
        _module(root, f"us-tx/statutes/unrelated-{index}.yaml")
    changed = _signed_change(root, "us-az/policies/des/block-7.yaml")
    walked: list[Path] = []
    original = _ReadSets.country_keys

    def country_keys(self, content_root):
        walked.append(content_root)
        return original(self, content_root)

    monkeypatch.setattr(_ReadSets, "country_keys", country_keys)

    signed = select_validation_waiver_scope(
        [composition], root=root, changed_paths=changed
    )
    companion_only = select_validation_waiver_scope(
        [composition],
        root=root,
        changed_paths={"us-az/policies/des/block-7.test.yaml"},
    )

    # A changed module selects the composition by itself. Only a change with
    # no module needs the walk over every module of the country.
    assert signed.paths == (composition,)
    assert companion_only.paths == ()
    assert walked == [(root / "us").resolve()]


def test_full_fallback_returns_the_given_paths_once_and_sorted(tmp_path: Path):
    root = tmp_path / "rulespec-us"
    waivers = ["us-ny/statutes/b.yaml", "us-ca/statutes/a.yaml", "us/statutes/c.yaml"]
    for path in waivers:
        _module(root, path)

    selection = select_validation_waiver_scope(
        [*waivers, waivers[0]], root=root, changed_paths={".axiom/toolchain.toml"}
    )

    assert selection.scope == "full"
    assert selection.paths == (
        "us-ca/statutes/a.yaml",
        "us-ny/statutes/b.yaml",
        "us/statutes/c.yaml",
    )
    assert list(selection.reasons) == list(selection.paths)


def test_unclassifiable_country_walk_selects_each_composition_and_runs_once(
    tmp_path: Path, monkeypatch
):
    root = tmp_path / "rulespec-us"
    compositions = [
        "us/policies/tariff/ch01.yaml",
        "us/policies/tariff/ch02.yaml",
        "us/policies/tariff/ch03.yaml",
    ]
    for path in compositions:
        _composition(root, path)
    atomic = "us/policies/tariff/atomic.yaml"
    _module(root, atomic)
    _module(root, "us-tx/statutes/odd.yaml", imports=("data/terminal#value",))
    module = "us-az/policies/des/block-7.yaml"
    _module(root, module)
    companion = _companion(root, module)
    walks: list[Path] = []
    original = _ReadSets.country_modules

    def country_modules(self):
        walks.append(self.root)
        return original(self)

    monkeypatch.setattr(_ReadSets, "country_modules", country_modules)

    selection = select_validation_waiver_scope(
        [*compositions, atomic], root=root, changed_paths={companion}
    )

    # A companion-only change needs the country-wide walk for a composition.
    # One module of the country cannot be classified, so every composition is
    # audited, the atomic waiver is not, and the walk is attempted once.
    assert selection.scope == "incremental"
    assert selection.paths == tuple(compositions)
    for path in compositions:
        assert selection.reasons[path] == [
            "validation read set could not be established: "
            "unclassifiable import 'data/terminal#value' in us-tx/statutes/odd.yaml"
        ]
    assert len(walks) == 1


def test_absent_candidate_is_a_dependency_with_a_name_key(tmp_path: Path):
    root = (tmp_path / "rulespec-us").resolve()
    module = _module(root, "us-ca/statutes/a.yaml", imports=("us-ny:statutes/gone",))
    absent = root / "us-ny/statutes/gone.yaml"
    with _rulespec_routing_cache_scope(), _rulespec_resolution_cache_scope():
        reader = _ReadSets(root, ())
        node = reader.node(module, root / "us-ca")
        keys = reader.node_keys(module, root / "us-ca")

    # A target that does not exist yet stays a dependency: adding it changes
    # what validation reads. An existing changed path is compared with it by
    # name, which is how a case-sensitive filesystem spells a case alias.
    assert absent in node.dependencies
    assert node.targets == ()
    assert ("name", os.path.abspath(absent).casefold()) in keys
    assert keys == frozenset({("name", os.path.abspath(absent).casefold())})


def test_target_in_a_dependency_checkout_is_read_from_that_checkout(tmp_path: Path):
    root = (tmp_path / "rulespec-us").resolve()
    dependency = (tmp_path / "dependencies" / "rulespec-uk").resolve()
    _module(dependency, "uk/statutes/shared.yaml", imports=("uk:statutes/base",))
    _module(dependency, "uk/statutes/base.yaml")
    waived = "us-ca/statutes/reader.yaml"
    plain = "us-ca/statutes/plain.yaml"
    module = _module(root, waived, imports=("uk:statutes/shared#value",))
    _module(root, plain)
    (root / "README.md").write_text("# Notes\n", encoding="utf-8")

    with _rulespec_routing_cache_scope(), _rulespec_resolution_cache_scope():
        reader = _ReadSets(root, (dependency,))
        candidates = reader.target_candidates(
            ("rulespec-uk", "uk", Path("statutes/shared.yaml")), root / "us-ca"
        )
        closure = reader.closure(module, root / "us-ca")
        expected = _resolver_closure(reader, module, root / "us-ca")
    without = select_validation_waiver_scope(
        [waived, plain], root=root, changed_paths={"README.md"}
    )
    with_root = select_validation_waiver_scope(
        [waived, plain],
        root=root,
        changed_paths={"README.md"},
        rulespec_dependency_roots=(dependency,),
    )

    # One checkout owns a jurisdiction's namespace, so a target has exactly
    # one candidate root. Taking only the first of them changes nothing.
    assert candidates == (dependency / "uk/statutes/shared.yaml",)
    assert {
        dependency / "uk/statutes/shared.yaml",
        dependency / "uk/statutes/base.yaml",
    } <= closure
    assert closure == expected
    # Without the dependency checkout the import has no candidate at all.
    assert without.paths == (waived,)
    assert "unclassifiable import" in without.reasons[waived][0]
    assert with_root.scope == "incremental"
    assert with_root.paths == ()
