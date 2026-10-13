"""Check incremental selection against full audits using the real rules engine.

No validator results or import resolutions are mocked here. The generated
repositories use the signed corpus fixture used by the existing validator
tests, and active ledger fingerprints come from actual base validation.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import tempfile
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode import cli
from axiom_encode.harness.validator_pipeline import _rulespec_executable_index_for_roots
from axiom_encode.validation_waiver_scope import select_validation_waiver_scope
from tests.release_object_fixtures import bind_test_corpus_release

_JURISDICTIONS = ("us", "us-ca", "us-ny")
_EDITS = (
    "modify",
    "delete",
    "rename",
    "import",
    "import_add",
    "import_remove",
    "import_repoint",
    "companion",
    "companion_pass",
    "companion_delete",
    "docs",
)
_RELEASE = "incremental-invariant-test-release"


@pytest.fixture(scope="module")
def case_aliases_supported(tmp_path_factory) -> bool:
    """Keep real filesystem aliases distinct from missing lexical candidates."""

    probe = tmp_path_factory.mktemp("waiver-case-alias") / "CaseAliasProbe"
    probe.write_text("", encoding="utf-8")
    alias = probe.with_name(probe.name.lower())
    return alias.exists() and alias.samefile(probe)


@pytest.fixture(scope="module")
def real_engine_root() -> Path:
    """Use an explicit engine or the same local built engine as corpus audits."""

    declared = os.environ.get("AXIOM_ENCODE_TEST_ENGINE_PATH")
    if declared:
        root = Path(declared).resolve()
        if not any(
            (root / candidate).is_file()
            for candidate in (
                "target/release/axiom-rules-engine",
                "target/debug/axiom-rules-engine",
                "axiom-rules-engine",
            )
        ):
            pytest.fail("AXIOM_ENCODE_TEST_ENGINE_PATH has no built rules engine")
        return root
    locator = shutil.which("axiom-locate")
    if locator is None:
        pytest.skip(
            "real engine unavailable: set AXIOM_ENCODE_TEST_ENGINE_PATH or "
            "provide axiom-locate engine"
        )
    located = subprocess.run(
        [locator, "engine"],
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )
    if located.returncode != 0:
        pytest.skip("axiom-locate engine found no built rules engine")
    binary = Path(located.stdout.strip()).resolve()
    if not binary.is_file():
        pytest.fail("axiom-locate engine returned a missing binary")
    return (
        binary.parents[2]
        if binary.parent.name in {"debug", "release"}
        else binary.parent
    )


@dataclass(frozen=True)
class _RepositoryCase:
    jurisdictions: tuple[str, ...]
    values: tuple[int, ...]
    imports: tuple[tuple[int, ...], ...]
    edit: str
    target: int
    module_stems: tuple[str, ...] = ()
    companion_spelling: str = "exact"
    changed_spelling: str = "exact"

    def module_stem(self, index: int) -> str:
        return self.module_stems[index] if self.module_stems else str(index)

    def module_path(self, index: int) -> str:
        return f"{self.jurisdictions[index]}/statutes/1/{self.module_stem(index)}.yaml"

    def companion_path(self, index: int) -> str:
        stem = _spell(self.module_stem(index), self.companion_spelling)
        return f"{self.jurisdictions[index]}/statutes/1/{stem}.test.yaml"

    def rule_id(self, index: int) -> str:
        return f"{self.jurisdictions[index]}:statutes/1/{self.module_stem(index)}#total_{index}"

    def changed_path(self, path: str) -> str:
        candidate = Path(path)
        for suffix in (".test.yaml", ".yaml"):
            if candidate.name.endswith(suffix):
                stem = _spell(
                    candidate.name.removesuffix(suffix), self.changed_spelling
                )
                return candidate.with_name(stem + suffix).as_posix()
        return path


def _spell(value: str, spelling: str) -> str:
    if spelling == "lower":
        return value.lower()
    if spelling == "upper":
        return value.upper()
    return value


@st.composite
def _repository_cases(
    draw, *, batch: int = 0, max_size: int = 5, case_aliases: bool = False
):
    size = draw(st.integers(min_value=3, max_value=max_size))
    # Vary each batch's minimal example as well as its generated examples.
    # This keeps bounded batch runs useful while Hypothesis explores the
    # repository graph, values, edit kind and target and can shrink failures.
    permutation = random.Random(batch)

    def permuted_options(options):
        ordered = list(options)
        permutation.shuffle(ordered)
        return ordered

    jurisdictions = tuple(
        draw(st.sampled_from(permuted_options(_JURISDICTIONS))) for _ in range(size)
    )
    values = tuple(
        draw(st.sampled_from(permuted_options(range(3)))) for _ in range(size)
    )
    imports = [()]
    for index in range(1, size):
        dependency_options = [
            dependencies
            for count in range(min(index, 2) + 1)
            for dependencies in combinations(range(index), count)
        ]
        imports.append(draw(st.sampled_from(permuted_options(dependency_options))))
    edit_options = permuted_options(_EDITS)
    first_edit = _EDITS[batch % len(_EDITS)]
    edit_options.remove(first_edit)
    edit_options.insert(0, first_edit)
    target_options = permuted_options(range(size))
    first_target = (batch // len(_EDITS)) % size
    target_options.remove(first_target)
    target_options.insert(0, first_target)
    edit = draw(st.sampled_from(edit_options))
    target = draw(st.sampled_from(target_options))
    stem_style = draw(st.sampled_from(permuted_options(("numeric", "title", "lower"))))
    stems = (
        ()
        if stem_style == "numeric"
        else tuple(
            f"Case{index}" if stem_style == "title" else f"case{index}"
            for index in range(size)
        )
    )
    changed_spelling = draw(
        st.sampled_from(permuted_options(("exact", "lower", "upper")))
    )
    alias_spellings = ("exact", "lower", "upper") if case_aliases else ("exact",)
    companion_spelling = draw(st.sampled_from(permuted_options(alias_spellings)))
    # Ensure valid import edits perform their stated operation even when
    # Hypothesis starts from an empty or fully connected import graph.
    if edit in {"import_add", "import_remove", "import_repoint"}:
        target = max(target, 2 if edit == "import_repoint" else 1)
        dependencies = set(imports[target])
        if edit == "import_add":
            dependencies.discard(0)
        else:
            dependencies.add(0)
        if edit == "import_repoint":
            dependencies.discard(1)
        imports[target] = tuple(sorted(dependencies))
    return _RepositoryCase(
        jurisdictions=jurisdictions,
        values=values,
        imports=tuple(imports),
        edit=edit,
        target=target,
        module_stems=stems,
        companion_spelling=companion_spelling,
        changed_spelling=changed_spelling,
    )


def _module_payload(case: _RepositoryCase, index: int) -> dict:
    return {
        "format": "rulespec/v1",
        "imports": [case.rule_id(dependency) for dependency in case.imports[index]],
        "module": {
            "summary": "The allowance amount is 0, 1, 2, or 3.",
            "source_verification": {
                "corpus_citation_path": f"{case.jurisdictions[index]}/statute/1/{index}",
            },
        },
        "rules": [
            {
                "name": f"amount_{index}",
                "kind": "parameter",
                "dtype": "Money",
                "unit": "USD",
                "versions": [
                    {"effective_from": "2026-01-01", "formula": str(case.values[index])}
                ],
            },
            {
                "name": f"total_{index}",
                "kind": "derived",
                "entity": "Household",
                "dtype": "Money",
                "unit": "USD",
                "period": "Month",
                "versions": [
                    {
                        "effective_from": "2026-01-01",
                        "formula": " + ".join(
                            [f"amount_{index}", "1"]
                            + [
                                f"total_{dependency}"
                                for dependency in case.imports[index]
                            ]
                        ),
                    }
                ],
            },
        ],
    }


def _write_yaml(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _companion_payload(case: _RepositoryCase, index: int, *, offset: int = 100):
    return [
        {
            "name": "allowance",
            "period": "2026-01",
            "input": {},
            # Deliberately failing tests keep every base waiver active, and
            # expose changes to transitive imported formulas in the fingerprint.
            "output": {case.rule_id(index): offset},
        }
    ]


def _expected_total(case: _RepositoryCase, index: int) -> int:
    """Evaluate the generated acyclic formulas for a passing companion."""

    return (
        case.values[index]
        + 1
        + sum(_expected_total(case, dependency) for dependency in case.imports[index])
    )


def _corpus(parent: Path, case: _RepositoryCase):
    corpus_root = parent / "axiom-corpus"
    scopes = []
    for jurisdiction in sorted(set(case.jurisdictions)):
        provision = (
            corpus_root / "data/corpus/provisions" / jurisdiction / "statute/test.jsonl"
        )
        provision.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "id": f"test:{jurisdiction}/statute/1/{index}",
                "citation_path": f"{jurisdiction}/statute/1/{index}",
                "body": "The allowance amount is 0, 1, 2, or 3.",
                "jurisdiction": jurisdiction,
                "document_class": "statute",
                "version": "test",
                "source_path": f"sources/{jurisdiction}/statute/test",
                "source_as_of": "2026-01-01",
                "expression_date": "2026-01-01",
            }
            for index, owner in enumerate(case.jurisdictions)
            if owner == jurisdiction
        ]
        provision.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        scopes.append((jurisdiction, "statute", "test"))
    release = bind_test_corpus_release(corpus_root, _RELEASE, scopes)
    return corpus_root, release


def _bind_toolchain(root: Path, release) -> None:
    ledger = root / "known-validation-gaps.yaml"
    toolchain = root / ".axiom/toolchain.toml"
    toolchain.parent.mkdir(parents=True, exist_ok=True)
    toolchain.write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{_RELEASE}"\n'
        f'axiom_corpus_release_content_sha256 = "{release.content_sha256}"\n'
        f'validation_waiver_set_sha256 = "{hashlib.sha256(ledger.read_bytes()).hexdigest()}"\n',
        encoding="utf-8",
    )


def _waiver_audit(root, corpus, engine, protected_base, changed_paths_file, *, scope):
    # Each CLI invocation starts with a fresh executable index. Keeping the
    # process-wide cache across BASE and HEAD would hide upstream-placement
    # drift when a searched module changed between these independent runs.
    _rulespec_executable_index_for_roots.cache_clear()
    output = io.StringIO()
    with redirect_stdout(output):
        exit_code = cli._cmd_validation_waivers_audit(
            SimpleNamespace(
                root=root,
                corpus_path=corpus,
                axiom_rules_path=engine,
                protected_base=protected_base,
                changed_paths=changed_paths_file,
                scope=scope,
                json=True,
            )
        )
    return exit_code, json.loads(output.getvalue())


def _full_audit(root, corpus, engine, protected_base, changed_paths_file):
    return _waiver_audit(
        root, corpus, engine, protected_base, changed_paths_file, scope="full"
    )[1]


def _verdicts(report: dict) -> dict[str, tuple]:
    assert report["scope"] == "full"
    assert report["checked"] == len(report["results"])
    return {
        item["path"]: (
            item["kind"],
            item["success"],
            item.get("passed"),
            item.get("fingerprint"),
            item.get("module_deleted", False),
            item.get("error"),
        )
        for item in report["results"]
    }


def _edit_repository(root: Path, case: _RepositoryCase, ledger, release) -> set[str]:
    target = root / case.module_path(case.target)
    changed = set()
    if case.edit == "modify":
        values = list(case.values)
        values[case.target] += 1
        _write_yaml(
            target, _module_payload(replace(case, values=tuple(values)), case.target)
        )
        changed.add(case.module_path(case.target))
    elif case.edit in {"delete", "rename"}:
        if case.edit == "delete":
            target.unlink()
        else:
            destination = target.with_name(f"{case.target}-renamed.yaml")
            target.rename(destination)
            changed.add(destination.relative_to(root).as_posix())
        changed.add(case.module_path(case.target))
        if case.module_path(case.target) in ledger["validate_failures"]:
            del ledger["validate_failures"][case.module_path(case.target)]
            _write_yaml(root / "known-validation-gaps.yaml", ledger)
            _bind_toolchain(root, release)
            changed.update({"known-validation-gaps.yaml", ".axiom/toolchain.toml"})
    elif case.edit == "import":
        payload = _module_payload(case, case.target)
        # A changed import to a missing module tests fail-closed HEAD discovery.
        payload["imports"].append("us:statutes/1/missing#missing_amount")
        _write_yaml(target, payload)
        changed.add(case.module_path(case.target))
    elif case.edit in {"import_add", "import_remove", "import_repoint"}:
        imports = list(case.imports)
        dependencies = set(imports[case.target])
        if case.edit == "import_add":
            dependencies.add(0)
        else:
            dependencies.remove(0)
        if case.edit == "import_repoint":
            dependencies.add(1)
        imports[case.target] = tuple(sorted(dependencies))
        _write_yaml(
            target,
            _module_payload(replace(case, imports=tuple(imports)), case.target),
        )
        changed.add(case.module_path(case.target))
    elif case.edit == "companion_reference":
        # No waived module imports this sibling-jurisdiction target. Its
        # exported rule is read only while validating a canonical output key
        # in another module's companion, and changes that diagnostic.
        payload = _module_payload(case, case.target)
        payload["rules"] = [
            rule for rule in payload["rules"] if rule["name"] != f"total_{case.target}"
        ]
        _write_yaml(target, payload)
        changed.add(case.module_path(case.target))
    elif case.edit == "duplicate":
        # Introduce an exact ancestor executable copy outside the waived
        # module's import graph and directory. Only implicit upstream indexing
        # can account for the resulting validation fingerprint change.
        payload = _module_payload(case, 1)
        _write_yaml(target, payload)
        changed.add(case.module_path(case.target))
    elif case.edit in {"companion", "companion_pass", "companion_delete"}:
        companion = root / case.companion_path(case.target)
        if case.edit == "companion_delete":
            companion.unlink()
        else:
            expected = (
                _expected_total(case, case.target)
                if case.edit == "companion_pass"
                else 101
            )
            _write_yaml(
                companion, _companion_payload(case, case.target, offset=expected)
            )
        changed.add(companion.relative_to(root).as_posix())
    else:
        (root / "README.md").write_text(
            "Updated inert documentation.\n", encoding="utf-8"
        )
        changed.add("README.md")
    return {case.changed_path(path) for path in changed}


def _check_invariant(
    case: _RepositoryCase, engine: Path, *, audit_incrementally: bool = False
) -> None:
    with (
        tempfile.TemporaryDirectory(prefix="waiver-invariant-") as temporary,
        patch.dict(os.environ, {cli._WAIVER_AUDIT_WORKERS_ENV: "1"}),
    ):
        parent = Path(temporary)
        root = parent / "rulespec-us"
        for index in range(len(case.values)):
            module = root / case.module_path(index)
            _write_yaml(module, _module_payload(case, index))
            _write_yaml(
                root / case.companion_path(index),
                _companion_payload(case, index),
            )
        if case.edit == "companion_reference":
            payload = _companion_payload(case, 1)
            payload[0]["output"][case.rule_id(case.target)] = 100
            _write_yaml(root / case.companion_path(1), payload)
        _write_yaml(root / "known-validation-gaps.yaml", {"validate_failures": {}})
        corpus, release = _corpus(parent, case)
        _bind_toolchain(root, release)
        waived_paths = [case.module_path(index) for index in range(1, len(case.values))]
        fingerprints = cli._fingerprint_validation_waiver_modules(
            [Path(path) for path in waived_paths],
            root=root,
            corpus_path=corpus,
            axiom_rules_path=engine,
            corpus_release=release,
        )
        assert all(not result["passed"] for result in fingerprints)
        assert all(
            "compile" not in result["outcome"]["validate"]["validators"]
            for result in fingerprints
        ), fingerprints
        ledger = {
            "validate_failures": {
                result["path"]: {
                    "active": {
                        "fingerprint": result["fingerprint"],
                        "owner": "@axiom-test",
                        "issue": "https://github.com/TheAxiomFoundation/axiom-encode/issues/1",
                        "expires": (date.today() + timedelta(days=60)).isoformat(),
                    }
                }
                for result in fingerprints
            }
        }
        _write_yaml(root / "known-validation-gaps.yaml", ledger)
        _bind_toolchain(root, release)
        protected_base = parent / "protected-base.yaml"
        protected_base.write_bytes((root / "known-validation-gaps.yaml").read_bytes())
        changed_file = parent / "changed-paths.txt"
        changed_file.write_text("", encoding="utf-8")
        base_report = _full_audit(root, corpus, engine, protected_base, changed_file)
        assert base_report["success"], base_report
        changed = _edit_repository(root, case, ledger, release)
        changed_file.write_text(
            "".join(path + "\n" for path in sorted(changed)), encoding="utf-8"
        )
        head_report = _full_audit(root, corpus, engine, protected_base, changed_file)
        base_verdicts = _verdicts(base_report)
        head_verdicts = _verdicts(head_report)
        assert set(base_verdicts) == set(head_verdicts) == set(waived_paths)
        changed_verdicts = {
            path for path in waived_paths if base_verdicts[path] != head_verdicts[path]
        }
        if audit_incrementally:
            assert changed_verdicts, (case, base_report, head_report)
            assert not head_report["success"], head_report
            exit_code, incremental = _waiver_audit(
                root, corpus, engine, protected_base, changed_file, scope="incremental"
            )
            assert exit_code != 0 and not incremental["success"], (
                case,
                changed_verdicts,
                incremental,
            )
            assert changed_verdicts <= {
                result["path"] for result in incremental["results"]
            }, incremental
        selection = select_validation_waiver_scope(
            waived_paths, root=root, changed_paths=changed
        )
        assert selection.scope == "full" or changed_verdicts <= set(selection.paths), (
            case,
            changed,
            changed_verdicts,
            selection,
        )
        if case.edit == "docs":
            assert changed_verdicts == set()
            assert selection.scope == "incremental"
            assert selection.paths == ()
        if (
            case.edit in {"companion", "companion_pass", "companion_delete"}
            and case.target > 0
        ):
            assert case.module_path(case.target) in changed_verdicts
            if all((root / path).is_file() for path in changed):
                assert selection.scope == "incremental"
                assert set(selection.paths) == {case.module_path(case.target)}
            else:
                assert selection.scope == "full"
                assert set(selection.paths) == set(waived_paths)
        if case.edit == "companion_pass" and case.target > 0:
            base_result = next(
                item
                for item in base_report["results"]
                if item["path"] == case.module_path(case.target)
            )
            result = next(
                item
                for item in head_report["results"]
                if item["path"] == case.module_path(case.target)
            )
            assert base_result["outcome"]["companion"]["passed"] is False
            assert result["outcome"]["companion"]["passed"] is True, result
            assert result["fingerprint"] != base_result["fingerprint"]
        if case.edit == "companion_reference":
            referencing = case.module_path(1)
            assert referencing in changed_verdicts
            assert selection.scope == "incremental"
            assert selection.paths == (referencing,)
            result = next(
                item for item in head_report["results"] if item["path"] == referencing
            )
            assert "does not resolve to a derived rule" in json.dumps(result["outcome"])
        if case.edit == "duplicate":
            duplicated = case.module_path(1)
            assert duplicated in changed_verdicts
            result = next(
                item for item in head_report["results"] if item["path"] == duplicated
            )
            assert "Upstream placement violation" in json.dumps(result["outcome"])


_CHAIN = _RepositoryCase(
    jurisdictions=("us-ca", "us-ca", "us-ny", "us-ny"),
    values=(1, 1, 1, 1),
    imports=((), (0,), (1,), ()),
    edit="modify",
    target=0,
)


@pytest.mark.parametrize("batch", range(30), ids=lambda batch: f"batch-{batch:02d}")
@settings(max_examples=3, deadline=None)
@given(data=st.data())
def test_full_audit_verdict_changes_are_selected(
    data, real_engine_root, case_aliases_supported, batch
):
    case = data.draw(
        _repository_cases(batch=batch, case_aliases=case_aliases_supported),
        label="repository",
    )
    _check_invariant(case, real_engine_root)


@pytest.mark.parametrize(
    "case",
    [
        replace(
            _CHAIN,
            edit="companion",
            target=1,
            module_stems=("Case0", "Case1", "Case2", "Case3"),
            companion_spelling="lower",
        ),
        _RepositoryCase(
            jurisdictions=("us-ny", "us-ca", "us-ca"),
            values=(1, 1, 1),
            imports=((), (0,), ()),
            edit="modify",
            target=0,
            module_stems=("Case0", "Case1", "Case2"),
            changed_spelling="lower",
        ),
    ],
    ids=("companion", "import"),
)
def test_case_alias_verdict_changes_fail_incremental_audit(
    case, real_engine_root, case_aliases_supported
):
    """A valid BASE cannot become a false incremental success through aliases."""

    if not case_aliases_supported:
        pytest.skip("real case aliases require a case-insensitive filesystem")
    _check_invariant(case, real_engine_root, audit_incrementally=True)


@pytest.mark.parametrize(
    "case",
    [
        _CHAIN,
        replace(_CHAIN, edit="delete"),
        replace(_CHAIN, edit="rename"),
        replace(_CHAIN, edit="import", target=1),
        replace(_CHAIN, edit="import_add", target=3),
        replace(_CHAIN, edit="import_remove", target=1),
        replace(_CHAIN, edit="import_repoint", target=2, imports=((), (0,), (0,), ())),
        replace(_CHAIN, edit="companion", target=1),
        replace(_CHAIN, edit="companion_pass", target=1),
        replace(_CHAIN, edit="companion_delete", target=1),
        replace(_CHAIN, edit="docs"),
        _RepositoryCase(
            jurisdictions=("us-ny", "us-ca", "us-tx"),
            values=(1, 1, 1),
            imports=((), (), ()),
            edit="companion_reference",
            target=0,
        ),
        _RepositoryCase(
            jurisdictions=("us", "us-ca", "us-ny"),
            values=(1, 1, 1),
            imports=((), (), ()),
            edit="duplicate",
            target=0,
        ),
    ],
    ids=lambda case: case.edit,
)
def test_full_audit_explicit_edit_invariant(case, real_engine_root):
    _check_invariant(case, real_engine_root)
