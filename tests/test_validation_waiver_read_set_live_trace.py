"""Live check of the read-set model against what validation really opens.

The incremental waiver scope rests on a model of which repository files the
validator reads for one waived module (`validation_waiver_scope._ReadSets`).
This test fingerprints real modules of a small repository with the real
engine, records every file this process opens and every directory it lists
under the repository, and requires them to be inside the model. A validator
change that reads a file by a new route fails here.

Limits: the engine child process is not traced, a path that is only stat-ed is
not recorded, and compositions are not exercised (no composer in this job).
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from axiom_encode import cli
from axiom_encode.harness.validator_pipeline import _rulespec_resolution_cache_scope
from axiom_encode.repo_routing import _rulespec_routing_cache_scope
from axiom_encode.validation_waiver_scope import _ReadSets
from tests import test_validation_waiver_incremental_invariant as invariant

# The one repository-wide walk the model accounts for: checkout admission
# refuses a symlink anywhere, and the scope forces the full audit for one.
_ADMISSION_WALK = "_reject_rulespec_checkout_symlinks"


class _Trace:
    def __init__(self) -> None:
        self.root: str | None = None
        self.opened: set[str] = set()
        self.listed: set[str] = set()

    def relative(self, raw: object) -> str | None:
        if self.root is None or isinstance(raw, int) or raw is None:
            return None
        try:
            text = os.fspath(raw)  # type: ignore[arg-type]
        except TypeError:
            return None
        if isinstance(text, bytes):
            text = os.fsdecode(text)
        path = os.path.realpath(text)
        if path == self.root:
            return ""
        if path.startswith(self.root + os.sep):
            return Path(path).relative_to(self.root).as_posix()
        return None

    def hook(self, event: str, args: tuple) -> None:
        if self.root is None:
            return
        if event == "open":
            relative = self.relative(args[0])
            if relative:
                self.opened.add(relative)
        elif event in {"os.scandir", "os.listdir"}:
            relative = self.relative(args[0] if args else None)
            if relative is not None and not _in_admission_walk():
                self.listed.add(relative)


def _in_admission_walk() -> bool:
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_name == _ADMISSION_WALK:
            return True
        frame = frame.f_back
    return False


_TRACE = _Trace()
_HOOKED = False


def _install_trace() -> None:
    """Install the audit hook once, and only when a traced test really runs.

    An audit hook cannot be removed. It records only while a root is set.
    """

    global _HOOKED
    if not _HOOKED:
        sys.addaudithook(_TRACE.hook)
        _HOOKED = True


_CASE = invariant._RepositoryCase(
    jurisdictions=("us", "us", "us-ca", "us-ca", "us-ny", "us-ny"),
    values=(1, 2, 1, 2, 1, 2),
    # us-ca/2 imports the federal us/1; us-ca/3 imports us-ca/2; us-ny/5
    # imports us-ny/4 and the sibling state's us-ca/2.
    imports=((), (0,), (1,), (2,), (), (4, 2)),
    edit="docs",
    target=0,
)
# Files no waived module of the case may read: another state, a ProgramSpec,
# applied manifests, documents.
_DECOYS = (
    "us-tx/statutes/1/decoy.yaml",
    "us-tx/statutes/1/decoy.test.yaml",
    "programs/us-tx/decoy/fy-2026.yaml",
    ".axiom/encoding-manifests/us-tx/statutes/1/decoy.json",
    ".axiom/encoding-manifests/us-ca/statutes/1/2.json",
    ".axiom/encoding-manifests/us/statutes/1/1.json",
    "docs/guide.md",
    "README.md",
)


@pytest.fixture(scope="module")
def traced_repository(real_engine, tmp_path_factory):
    _install_trace()
    parent = tmp_path_factory.mktemp("waiver-live-trace")
    root = parent / "rulespec-us"
    for index in range(len(_CASE.values)):
        invariant._write_yaml(
            root / _CASE.module_path(index), invariant._module_payload(_CASE, index)
        )
        invariant._write_yaml(
            root / _CASE.companion_path(index),
            invariant._companion_payload(_CASE, index),
        )
    decoy_case = replace(_CASE, jurisdictions=("us-tx",), values=(1,), imports=((),))
    for decoy in _DECOYS:
        path = root / decoy
        path.parent.mkdir(parents=True, exist_ok=True)
        if decoy.endswith(".test.yaml"):
            invariant._write_yaml(path, invariant._companion_payload(decoy_case, 0))
        elif decoy.endswith(".yaml") and not decoy.startswith("programs/"):
            invariant._write_yaml(path, invariant._module_payload(decoy_case, 0))
        else:
            path.write_text("{}\n", encoding="utf-8")
    invariant._write_yaml(
        root / "known-validation-gaps.yaml", {"validate_failures": {}}
    )
    corpus, release = invariant._corpus(parent, _CASE)
    invariant._bind_toolchain(root, release)
    return root.resolve(), corpus, release, real_engine.root


def _trace_fingerprint(repository, module: str) -> tuple[set[str], set[str], dict]:
    root, corpus, release, engine = repository
    invariant._rulespec_executable_index_for_roots.cache_clear()
    _TRACE.opened.clear()
    _TRACE.listed.clear()
    _TRACE.root = os.path.realpath(root)
    try:
        results = cli._fingerprint_validation_waiver_modules(
            [Path(module)],
            root=root,
            corpus_path=corpus,
            axiom_rules_path=engine,
            corpus_release=release,
        )
    finally:
        _TRACE.root = None
    return set(_TRACE.opened), set(_TRACE.listed), results[0]


def _outside_the_model(root: Path, module: str, opened: set[str], listed: set[str]):
    """Return the opened files and listed directories the model does not hold."""

    with _rulespec_routing_cache_scope(), _rulespec_resolution_cache_scope():
        read_set, upstream_roots, composition = _ReadSets(root, ()).read_set(module)
    assert composition is False
    # A directory may be listed, outside checkout admission, when it is the
    # checkout root, lies in a searched jurisdiction, or holds a model file.
    search_names = {root_path.name for root_path in upstream_roots}
    parents = {""} | {
        parent.as_posix()
        for path in read_set
        for parent in Path(path).parents
        if parent.as_posix() != "."
    }
    unexplained = {
        directory
        for directory in listed
        if directory not in parents and directory.split("/")[0] not in search_names
    }
    return opened - read_set, unexplained


@pytest.mark.parametrize("index", range(len(_CASE.values)))
def test_validation_opens_only_files_of_the_read_set(traced_repository, index: int):
    root = traced_repository[0]
    module = _CASE.module_path(index)

    opened, listed, result = _trace_fingerprint(traced_repository, module)

    assert result["path"] == module
    assert "compile" not in result["outcome"]["validate"]["validators"], result
    # The tracer sees real reads: the module, its companion, the placement
    # index of its own and its federal jurisdiction, its imports, and the
    # directory walk that builds the index.
    assert module in opened
    assert _CASE.companion_path(index) in opened
    jurisdiction = _CASE.jurisdictions[index]
    searched = {
        _CASE.module_path(other)
        for other, owner in enumerate(_CASE.jurisdictions)
        if owner in {jurisdiction, "us"}
    }
    assert searched <= opened
    assert {_CASE.module_path(other) for other in _CASE.imports[index]} <= opened
    assert f"{jurisdiction}/statutes" in listed

    unread_files, unexplained_directories = _outside_the_model(
        root, module, opened, listed
    )
    assert not unread_files, sorted(unread_files)
    assert not unexplained_directories, sorted(unexplained_directories)
    assert not opened & set(_DECOYS)


@pytest.mark.parametrize("route", ["open", "list"])
def test_a_new_read_route_is_reported(traced_repository, monkeypatch, route: str):
    """The check fails when validation reads outside the model."""

    root = traced_repository[0]
    module = _CASE.module_path(2)
    original = cli.ValidatorPipeline.validate

    def validate(self, rulespec_file, skip_reviewers=False):
        # A stand-in for a validator change: one more read, by a route the
        # model does not know.
        if route == "open":
            (root / "us-tx/statutes/1/decoy.yaml").read_text(encoding="utf-8")
        else:
            sorted((root / "us-tx/statutes").rglob("*.yaml"))
        return original(self, rulespec_file, skip_reviewers=skip_reviewers)

    monkeypatch.setattr(cli.ValidatorPipeline, "validate", validate)

    opened, listed, _result = _trace_fingerprint(traced_repository, module)
    unread_files, unexplained_directories = _outside_the_model(
        root, module, opened, listed
    )

    if route == "open":
        assert unread_files == {"us-tx/statutes/1/decoy.yaml"}
        assert not unexplained_directories
    else:
        assert not unread_files
        assert unexplained_directories == {"us-tx/statutes", "us-tx/statutes/1"}
