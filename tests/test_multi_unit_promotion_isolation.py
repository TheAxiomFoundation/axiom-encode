"""Each unit of a multi-unit promotion is validated in isolation (#2301).

The promotion command validates unit ``i`` in a temporary overlay from which
the files of every other unit of the batch are removed. So the tree that
validation reads is the tree of a candidate that holds unit ``i`` alone, and
no reference of any kind can make a unit pass because another unit is in the
checkout.

These tests run the real overlay staging. Only the engine-backed validator is
replaced, by a probe that (a) records the complete tree it is given and (b)
judges every absolute RuleSpec reference of the unit's primary and companion
with the encoder's own reference parser and resolver against that tree.
"""

import contextlib
import hashlib
import json
import os
import random
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

import axiom_encode.cli as cli
from axiom_encode import reviewed_pr_admission as admission
from axiom_encode.cli import (
    _TRUSTED_RUNTIME_ENV,
    APPLIED_ENCODING_OFFICIAL_REPOSITORY,
    _applied_encoding_manifest_path,
    cmd_promote_reviewed_candidate,
)
from axiom_encode.harness.validator_pipeline import (
    _parse_rulespec_target,
    _resolve_rulespec_target_file,
    _rulespec_absolute_test_reference_issue,
)
from tests.test_cli import (
    AXIOM_ENCODE_TEST_VERSION,
    TEST_APPLY_SIGNING_BROKER,
    _complete_source_attestation,
    _git,
    _sha256_file,
)

SOURCE_SHA256 = "a" * 64
GIT_DATE = {
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00",
}
REFERENCE = re.compile(r"[a-z]{2}(?:-[a-z0-9]+)*:[A-Za-z0-9_./-]+#[A-Za-z0-9_.]+")
# Where each unit lives. ``e`` is alone in its directory, so hiding it must
# also remove the directory.
DIRECTORY = {"a": "7/2015", "b": "7/2015", "c": "7/2015", "d": "7/2016", "e": "8/1"}
BASE = "us:statutes/7/2014/base"


def citation(name: str) -> str:
    return f"us/statute/{DIRECTORY[name]}/{name}"


def primary(name: str) -> str:
    return f"us/statutes/{DIRECTORY[name]}/{name}.yaml"


def companion(name: str) -> str:
    return primary(name).removesuffix(".yaml") + ".test.yaml"


def target(name: str) -> str:
    return f"us:statutes/{DIRECTORY[name]}/{name}"


# The routes by which one unit can name another, other than ``imports``.
ROUTES = ("companion-output", "proof-import-target", "formula", "relation-target")


def unit_files(name: str, references=()) -> dict[str, str]:
    """The primary and companion of a unit that makes these references.

    ``references`` is ``((route, reference), ...)``.
    """
    rule: dict = {"name": f"{name}_rule", "kind": "derived"}
    relations = []
    outputs = {f"{target(name)}#{name}_rule": 1}
    for route, reference in references:
        if route == "companion-output":
            outputs[reference] = 1
        elif route == "proof-import-target":
            rule.setdefault("proof", {"atoms": []})["atoms"].append(
                {"import": {"target": reference}}
            )
        elif route == "formula":
            rule["formula"] = f"({rule.get('formula', '0')}) + {reference}"
        elif route == "relation-target":
            relations.append({"name": f"rel_{len(relations)}", "target": reference})
    module = {
        "format": "rulespec/v1",
        "module": {
            "source_verification": {
                "corpus_citation_path": citation(name),
                "source_sha256": SOURCE_SHA256,
            }
        },
        "rules": [rule],
    }
    if relations:
        module["relations"] = relations
    cases = [{"name": "case", "period": "2026-01", "input": {}, "output": outputs}]
    return {
        primary(name): yaml.safe_dump(module, sort_keys=False),
        companion(name): yaml.safe_dump(cases, sort_keys=False),
    }


def candidate(root: Path, units: dict[str, tuple]) -> tuple[Path, str]:
    """A checkout of ``main`` plus these units; the same commit every time."""
    repo = root / "rulespec-us"
    repo.mkdir(parents=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    files = {
        "us/statutes/7/2014/base.yaml": "format: rulespec/v1\nrules:\n- name: x\n",
        "us/statutes/7/2014/base.test.yaml": "[]\n",
        "README.md": "main\n",
    }
    for name, references in units.items():
        files.update(unit_files(name, references))
    for relative, text in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(repo, "add", ".")
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "candidate"],
        check=True,
        capture_output=True,
        env={**os.environ, **GIT_DATE},
    )
    return repo, _git(repo, "rev-parse", "HEAD").stdout.strip()


def tree_of(root: Path) -> dict[str, bytes]:
    """Every file and every directory below ``root``."""
    seen = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        assert not path.is_symlink()
        seen[relative + ("/" if path.is_dir() else "")] = (
            b"" if path.is_dir() else path.read_bytes()
        )
    return seen


def probe_issues(overlay_target: Path, content_root: Path) -> list[str]:
    """Judge the unit's references against the tree validation was given."""
    issues = []
    test_file = overlay_target.with_name(overlay_target.stem + ".test.yaml")
    for path in (overlay_target, test_file):
        for reference in sorted(set(REFERENCE.findall(path.read_text()))):
            parsed = _parse_rulespec_target(reference)
            assert parsed is not None, reference
            if _resolve_rulespec_target_file(parsed, content_root) is None:
                issues.append(f"{path.name}: `{reference}` could not be resolved")
    for case in yaml.safe_load(test_file.read_text()) or []:
        for key in case.get("output", {}):
            issue = _rulespec_absolute_test_reference_issue(
                key,
                label="Test output key",
                policy_repo_path=content_root,
                allow_input_slots=False,
                allow_relations=False,
                allow_outputs=True,
            )
            if issue is not None and "could not be resolved" in issue:
                issues.append(f"{test_file.name}: {issue}")
    return issues


class Promotion:
    """One promotion run with the real overlay staging and a probe validator."""

    def __init__(self, root: Path, repo: Path, head: str, names, *, batch: bool):
        self.root, self.repo, self.head = root, repo, head
        self.names = tuple(names)
        self.batch = batch
        self.trees: dict[str, dict[str, bytes]] = {}
        self.verdicts: dict[str, list[str]] = {}
        self.refuse = True
        self.signed: list[str] = []
        for directory in ("axiom-rules-engine", "axiom-corpus"):
            (root / directory).mkdir(exist_ok=True)
        lead = self.names[0]
        self.args = SimpleNamespace(
            repo=repo,
            axiom_rules_path=root / "axiom-rules-engine",
            corpus_path=root / "axiom-corpus",
            rulespec_dependency_root=[],
            rulespec_path=primary(lead),
            citation=citation(lead),
            reviewed_rulespec_ref=head,
            run_id="12345",
        )

    def admission(self) -> dict:
        shared = {
            "country": "us",
            "pr": 42,
            "head": self.head,
            "branch": "codex/candidate",
            "actor": "PavelMakarchuk",
            "run_id": "12345",
        }

        def record(name):
            return {
                "schema": "axiom-encode/reviewed-rulespec-admission/v1",
                "country": "us",
                "rulespec_ref": self.head,
                "rulespec_pr": 42,
                "pr_base_branch": "codex/candidate",
                "base_sha": "b" * 40,
                "binding": {
                    "citation": citation(name),
                    "rulespec_path": primary(name),
                    "rulespec_sha256": _sha256_file(self.repo / primary(name)),
                    "companion_sha256": _sha256_file(self.repo / companion(name)),
                },
                "declared": None,
                "purpose": "Test admission.",
            }

        if not self.batch:
            (name,) = self.names
            return {
                "workflow_commit": "e" * 40,
                "selector": {
                    **shared,
                    "citation": citation(name),
                    "rulespec_path": primary(name),
                },
                "record": record(name),
            }
        ordered = sorted(self.names, key=primary)
        return {
            "workflow_commit": "e" * 40,
            "selector": {
                **shared,
                "units": [
                    {"citation": citation(name), "rulespec_path": primary(name)}
                    for name in ordered
                ],
            },
            "records": [record(name) for name in ordered],
        }

    def _probe(self, pipeline, *, dependent_pipeline, overlay_target, dependents):
        overlay_repo = next(
            parent for parent in overlay_target.parents if parent.name == "rulespec-us"
        )
        name = overlay_target.stem
        self.trees[name] = tree_of(overlay_repo)
        self.verdicts[name] = probe_issues(overlay_target, overlay_repo / "us")
        return [(overlay_target, SimpleNamespace(all_passed=True, results={}))]

    def _validate(self, result, **kwargs):
        name = Path(result.output_file).stem
        outcome = self._real_validate(result, **kwargs)
        if self.refuse and outcome[0] and self.verdicts.get(name):
            return False, list(self.verdicts[name]), {}
        return outcome

    def _sign(self, payload, broker):
        self.signed.append(payload["citation"].rsplit("/", 1)[1])
        return self._real_sign(payload, broker)

    def __call__(self, monkeypatch):
        runtime = self.root / "runtime-attestation.json"
        runtime.write_text(json.dumps({"reviewed_pr_admission": self.admission()}))
        records = self.root / "reviewed-rulespec-refs"
        records.mkdir(exist_ok=True)
        monkeypatch.setenv(_TRUSTED_RUNTIME_ENV, "1")
        self._real_validate = cli._run_generated_encoding_overlay_validation
        self._real_sign = cli._sign_applied_encoding_manifest
        provenance = {
            "root": "/opt/axiom-verification",
            "commit": "b" * 40,
            "dirty_tracked": False,
            "version": AXIOM_ENCODE_TEST_VERSION,
            "version_commit": "b" * 40,
            "identity_source": "trusted-runtime-attestation",
        }
        stubs = {
            "_load_trusted_runtime_attestation": {
                "return_value": {"path": str(runtime), "commit": "e" * 40}
            },
            "_promotion_reviewed_rulespec_record_root": {"return_value": records},
            "_recover_apply_transaction": {},
            "load_rulespec_local_corpus_release": {"return_value": MagicMock()},
            "_manifest_primary_source_verifications": {
                "return_value": ([], "rulespec-us/us")
            },
            "resolve_corpus_source_unit": {
                "side_effect": lambda cited, _release: SimpleNamespace(
                    body="test source\n",
                    amendment_documents=(),
                    source_attestation=_complete_source_attestation(
                        cited, source_sha256=SOURCE_SHA256
                    ),
                )
            },
            # The real overlay staging runs; only the engine-backed parts are
            # replaced.
            "_run_generated_encoding_overlay_validation": {
                "side_effect": self._validate
            },
            "ValidatorPipeline": {"new": MagicMock()},
            "_validate_overlay_files": {"side_effect": self._probe},
            "_pinned_apply_engine_binding": {"return_value": None},
            "_apply_validation_execution_identity": {"return_value": {}},
            "_build_apply_validation_snapshot": {
                "return_value": {
                    "manifest_validation_execution": {
                        "schema": "axiom-encode/apply-validation-execution/v2"
                    }
                }
            },
            "_require_clean_axiom_encode_git_provenance": {"return_value": provenance},
            "verify_rulespec_validation_waiver_set": {"return_value": "c" * 64},
            "_target_uses_per_file_waiver_bindings": {"return_value": False},
            "_current_guard_encoder_execution_identity": {
                "return_value": {
                    "repository": APPLIED_ENCODING_OFFICIAL_REPOSITORY,
                    "commit": "b" * 40,
                    "version": AXIOM_ENCODE_TEST_VERSION,
                    "identity_source": "trusted-runtime-attestation",
                }
            },
            "_load_verified_applied_encoding_manifest_payload": {
                "return_value": ({}, "", "d" * 64, [])
            },
            "_sign_applied_encoding_manifest": {"side_effect": self._sign},
        }
        with contextlib.ExitStack() as stack:
            for name, kwargs in stubs.items():
                stack.enter_context(patch(f"axiom_encode.cli.{name}", **kwargs))
            stack.enter_context(
                patch("axiom_encode.reviewed_pr_admission.revalidate_admission")
            )
            signer = stack.enter_context(
                patch("axiom_encode.cli._isolated_apply_manifest_signer")
            )
            signer.return_value.__enter__.return_value = TEST_APPLY_SIGNING_BROKER
            cmd_promote_reviewed_candidate(self.args)

    def manifests(self) -> list[str]:
        return _git(
            self.repo, "ls-files", "--others", "--exclude-standard"
        ).stdout.splitlines()


@pytest.fixture(autouse=True)
def switch_on(monkeypatch):
    monkeypatch.setattr(admission, "MULTI_UNIT_PROMOTION_ENABLED", True)


def batch_run(root, units, monkeypatch, *, refuse=True):
    repo, head = candidate(root, units)
    before = tree_of(repo)
    run = Promotion(root, repo, head, units, batch=True)
    run.refuse = refuse
    error = None
    try:
        run(monkeypatch)
    except (RuntimeError, ValueError) as exc:
        error = exc
    # The checkout differs from the candidate by the manifests only.
    after = tree_of(repo)
    for path in list(after):
        if path.startswith((".axiom/", ".git/")) or path in {".axiom/", ".git/"}:
            after.pop(path)
    for path in list(before):
        if path.startswith(".git/") or path == ".git/":
            before.pop(path)
    assert after == before
    assert _git(repo, "status", "--porcelain", "--", "us", "README.md").stdout == ""
    return run, error


def alone_run(root, name, references, monkeypatch, *, refuse=True):
    repo, head = candidate(root, {name: references})
    run = Promotion(root, repo, head, [name], batch=False)
    run.refuse = refuse
    error = None
    try:
        run(monkeypatch)
    except (RuntimeError, ValueError) as exc:
        error = exc
    return run, error


# --- The tree a unit is validated against -------------------------------------


def test_each_unit_is_validated_against_the_tree_of_a_candidate_of_it_alone(
    tmp_path, monkeypatch
):
    units = {name: () for name in "abde"}
    run, error = batch_run(tmp_path / "batch", units, monkeypatch)
    assert error is None
    assert sorted(run.signed) == sorted(units)
    for name in units:
        alone, alone_error = alone_run(
            tmp_path / f"alone-{name}", name, (), monkeypatch
        )
        assert alone_error is None
        assert run.trees[name] == alone.trees[name]
        present = {path for path in run.trees[name] if path.endswith(".yaml")}
        assert present == {
            primary(name),
            companion(name),
            "us/statutes/7/2014/base.yaml",
            "us/statutes/7/2014/base.test.yaml",
        }
    # A directory that held only another unit is gone, as in a checkout of
    # this unit alone.
    assert "us/statutes/8/" not in run.trees["a"]
    assert "us/statutes/8/1/" in run.trees["e"]
    assert "us/statutes/7/2016/" not in run.trees["a"]


def test_without_the_isolation_the_other_units_are_visible(tmp_path, monkeypatch):
    """What the isolation changes: the control for the tests below."""
    monkeypatch.setattr(cli, "_isolate_promotion_batch_overlay", lambda *a, **k: None)
    units = {"a": (("companion-output", f"{target('b')}#b_rule"),), "b": ()}
    run, error = batch_run(tmp_path, units, monkeypatch)
    assert error is None
    assert primary("b") in run.trees["a"]
    assert run.verdicts["a"] == []


def test_a_single_unit_promotion_is_not_isolated_from_anything(tmp_path, monkeypatch):
    isolate = MagicMock()
    monkeypatch.setattr(cli, "_isolate_promotion_batch_overlay", isolate)
    run, error = alone_run(tmp_path, "a", (), monkeypatch)
    assert error is None and run.signed == ["a"]
    isolate.assert_not_called()


def test_a_batch_of_one_hides_nothing(tmp_path, monkeypatch):
    isolate = MagicMock()
    monkeypatch.setattr(cli, "_isolate_promotion_batch_overlay", isolate)
    run, error = batch_run(tmp_path, {"a": ()}, monkeypatch)
    assert error is None and run.signed == ["a"]
    isolate.assert_not_called()


# --- Every route to another unit is refused as a single-unit run refuses ------


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("needy", ["a", "b"])
def test_a_unit_that_depends_on_another_unit_is_refused_as_alone(
    tmp_path, monkeypatch, route, needy
):
    """Not listed in ``imports``, so only the isolation can refuse it."""
    other = "b" if needy == "a" else "a"
    references = ((route, f"{target(other)}#{other}_rule"),)
    assert (
        cli._promotion_batch_import_issue(
            *(
                lambda repo: (
                    repo,
                    {
                        Path(primary(name)): (repo / primary(name)).read_bytes()
                        for name in "ab"
                    },
                )
            )(candidate(tmp_path / "probe", {needy: references, other: ()})[0])
        )
        is None
    )
    run, error = batch_run(
        tmp_path / "batch", {needy: references, other: ()}, monkeypatch
    )
    alone, alone_error = alone_run(tmp_path / "alone", needy, references, monkeypatch)
    assert isinstance(alone_error, RuntimeError)
    assert "could not be resolved" in str(alone_error)
    assert str(alone_error).startswith(
        "reviewed candidate failed current overlay validation: "
    )
    # The same refusal, with the unit named.
    assert isinstance(error, RuntimeError)
    assert str(error) == f"unit {citation(needy)} ({primary(needy)}): {alone_error}"
    # Nothing is signed: not the needy unit, and not the other one either.
    assert run.signed == [] and run.manifests() == []
    assert alone.signed == [] and alone.manifests() == []


@pytest.mark.parametrize("route", ROUTES)
def test_a_reference_to_a_module_on_main_passes_in_a_batch_as_alone(
    tmp_path, monkeypatch, route
):
    references = ((route, f"{BASE}#x"),)
    run, error = batch_run(tmp_path / "batch", {"a": references, "b": ()}, monkeypatch)
    alone, alone_error = alone_run(tmp_path / "alone", "a", references, monkeypatch)
    assert error is None and alone_error is None
    assert sorted(run.signed) == ["a", "b"] and alone.signed == ["a"]
    assert run.verdicts["a"] == alone.verdicts["a"] == []


def test_two_units_that_name_each_other_are_both_refused(tmp_path, monkeypatch):
    units = {
        "a": (("relation-target", f"{target('b')}#b_rule"),),
        "b": (("formula", f"{target('a')}#a_rule"),),
    }
    run, _error = batch_run(tmp_path, units, monkeypatch, refuse=False)
    assert all("could not be resolved" in issue for issue in run.verdicts["a"])
    assert run.verdicts["a"] and run.verdicts["b"]


# --- Differential: a unit in a batch is judged as the unit alone --------------


@pytest.mark.parametrize("seed", range(12))
def test_random_batches_judge_each_unit_as_the_unit_alone(tmp_path, monkeypatch, seed):
    rng = random.Random(seed)
    names = rng.sample("abcde", rng.randint(2, 4))
    units = {}
    for name in names:
        references = []
        for _ in range(rng.choice([0, 0, 1, 1, 2, 3])):
            kind = rng.random()
            if kind < 0.55:
                referenced = rng.choice([other for other in names if other != name])
                reference = f"{target(referenced)}#{referenced}_rule"
            elif kind < 0.8:
                reference = f"{BASE}#x"
            elif kind < 0.9:
                # A unit of the alphabet that is not in this batch.
                absent = rng.choice([other for other in "abcde" if other not in names])
                reference = f"{target(absent)}#{absent}_rule"
            else:
                reference = "us:statutes/9/9/nothing#x"
            references.append((rng.choice(ROUTES), reference))
        units[name] = tuple(references)
    # Record every unit's verdict, also after a unit that would refuse.
    run, error = batch_run(tmp_path / "batch", units, monkeypatch, refuse=False)
    assert error is None
    assert sorted(run.trees) == sorted(names)
    refused_alone = []
    for name in names:
        alone, _error = alone_run(
            tmp_path / f"alone-{name}", name, units[name], monkeypatch, refuse=False
        )
        assert run.trees[name] == alone.trees[name], name
        assert run.verdicts[name] == alone.verdicts[name], name
        if alone.verdicts[name]:
            refused_alone.append(name)
        expects_refusal = any(
            not reference.startswith(BASE) for _route, reference in units[name]
        )
        assert bool(alone.verdicts[name]) is expects_refusal
    # With refusal on, the batch signs all units or none.
    enforced, enforced_error = batch_run(tmp_path / "enforced", units, monkeypatch)
    if refused_alone:
        first = sorted(refused_alone, key=primary)[0]
        assert str(enforced_error).startswith(f"unit {citation(first)} (")
        assert enforced.signed == [] and enforced.manifests() == []
    else:
        assert enforced_error is None
        assert sorted(enforced.signed) == sorted(names)
        assert len(enforced.manifests()) == len(names)


# --- The isolation itself fails closed and never touches the checkout ---------


def overlay(tmp_path: Path) -> Path:
    root = tmp_path / "overlay"
    for relative in (
        "us/statutes/7/2015/a.yaml",
        "us/statutes/7/2015/b.yaml",
        "us/statutes/8/1/e.yaml",
        "us/statutes/8/1/e.test.yaml",
        "README.md",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    return root


def test_isolation_removes_the_named_files_and_emptied_directories(tmp_path):
    root = overlay(tmp_path)
    cli._isolate_promotion_batch_overlay(
        root,
        (
            Path("us/statutes/8/1/e.yaml"),
            Path("us/statutes/8/1/e.test.yaml"),
            Path("us/statutes/7/2015/b.yaml"),
        ),
    )
    assert sorted(tree_of(root)) == [
        "README.md",
        "us/",
        "us/statutes/",
        "us/statutes/7/",
        "us/statutes/7/2015/",
        "us/statutes/7/2015/a.yaml",
    ]


def test_isolation_never_removes_the_overlay_root(tmp_path):
    root = tmp_path / "overlay"
    (root / "us").mkdir(parents=True)
    (root / "us/a.yaml").write_text("a")
    cli._isolate_promotion_batch_overlay(root, (Path("us/a.yaml"),))
    assert root.is_dir() and list(root.iterdir()) == []


@pytest.mark.parametrize(
    "path",
    [
        "us/statutes/7/2015/missing.yaml",
        "us/statutes/7/2015",
        "../outside.yaml",
        "us/../README.md",
        "/etc/hosts",
        ".",
        "us/statutes/7/2015/link.yaml",
    ],
)
def test_isolation_refuses_a_path_it_cannot_remove_safely(tmp_path, path):
    root = overlay(tmp_path)
    (tmp_path / "outside.yaml").write_text("outside")
    (root / "us/statutes/7/2015/link.yaml").symlink_to(root / "README.md")
    before = sorted(path.as_posix() for path in tmp_path.rglob("*"))
    with pytest.raises(ValueError, match="cannot isolate a unit from"):
        cli._isolate_promotion_batch_overlay(root, (Path(path),))
    # Nothing was removed, inside the overlay or outside it.
    assert sorted(path.as_posix() for path in tmp_path.rglob("*")) == before
    assert (tmp_path / "outside.yaml").read_text() == "outside"
    assert (root / "README.md").read_text() == "README.md"


@pytest.mark.parametrize("value", [["us/a.yaml"], ("us/a.yaml",), "us/a.yaml", 7])
def test_a_malformed_isolation_marker_stops_validation(value):
    result = SimpleNamespace()
    setattr(result, cli._PROMOTION_BATCH_ISOLATION_ATTR, value)
    with pytest.raises(ValueError, match="isolation is malformed"):
        cli._result_promotion_batch_isolation(result)


def test_a_result_without_the_marker_isolates_nothing():
    assert cli._result_promotion_batch_isolation(SimpleNamespace()) == ()
    assert cli._result_promotion_batch_isolation(object()) == ()


def test_a_file_that_cannot_be_hidden_fails_the_unit_and_signs_nothing(
    tmp_path, monkeypatch
):
    """The overlay copy lacks a file the batch names: validation stops."""
    real_stage = cli._stage_apply_overlay_dependency_root

    def stage_without_b(*, source, target):
        real_stage(source=source, target=target)
        (target / primary("b")).unlink(missing_ok=True)

    monkeypatch.setattr(cli, "_stage_apply_overlay_dependency_root", stage_without_b)
    run, error = batch_run(tmp_path, {"a": (), "b": ()}, monkeypatch)
    assert isinstance(error, ValueError)
    assert str(error) == (
        f"unit {citation('a')} ({primary('a')}): cannot isolate a unit from a "
        f"batch file that is not a regular file of the candidate: {primary('b')}"
    )
    assert run.signed == [] and run.manifests() == []


def test_the_command_hides_exactly_the_other_units(tmp_path, monkeypatch):
    seen = {}
    real = cli._isolate_promotion_batch_overlay

    def record(overlay_repo, paths):
        present = sorted(
            path.relative_to(overlay_repo).as_posix()
            for path in overlay_repo.rglob("*.yaml")
        )
        seen[len(seen)] = (tuple(path.as_posix() for path in paths), present)
        return real(overlay_repo, paths)

    monkeypatch.setattr(cli, "_isolate_promotion_batch_overlay", record)
    run, error = batch_run(tmp_path, {name: () for name in "abc"}, monkeypatch)
    assert error is None
    hidden = [set(paths) for paths, _present in seen.values()]
    assert hidden == [
        {primary(other), companion(other)} | {primary(third), companion(third)}
        for name, other, third in (("a", "b", "c"), ("b", "a", "c"), ("c", "a", "b"))
    ]
    # Every overlay starts as a full copy: hiding is per unit, not cumulative.
    for _paths, present in seen.values():
        assert {primary(name) for name in "abc"} <= set(present)


# --- A unit with dependents is promoted alone ---------------------------------


def test_a_unit_with_an_existing_dependent_cannot_be_in_a_batch(tmp_path, monkeypatch):
    """Its dependents are compared with the checkout, outside the isolation."""
    dependent = Path("statutes/7/2014/base.yaml")
    monkeypatch.setattr(
        cli,
        "_find_rulespec_dependents",
        lambda root, relative: [root / dependent] if relative.stem == "a" else [],
    )
    run, error = batch_run(tmp_path / "batch", {"a": (), "b": ()}, monkeypatch)
    assert isinstance(error, RuntimeError)
    assert str(error) == (
        f"unit {citation('a')} ({primary('a')}): reviewed candidate failed current "
        "overlay validation: statutes/7/2015/a.yaml: an existing module depends on "
        "this unit; a unit with dependents cannot be part of a multi-unit "
        "promotion and must be promoted alone"
    )
    assert run.signed == [] and run.manifests() == []
    # Alone, the same unit reaches the dependent comparison.
    alone, alone_error = alone_run(tmp_path / "alone", "a", (), monkeypatch)
    assert "must be promoted alone" not in str(alone_error)


def test_the_manifest_of_an_isolated_unit_is_at_its_own_position(tmp_path, monkeypatch):
    run, error = batch_run(tmp_path, {"a": (), "e": ()}, monkeypatch)
    assert error is None
    assert run.manifests() == [
        _applied_encoding_manifest_path(Path(primary(name))).as_posix()
        for name in ("a", "e")
    ]
    digest = hashlib.sha256((run.repo / primary("e")).read_bytes()).hexdigest()
    payload = json.loads(
        (run.repo / _applied_encoding_manifest_path(Path(primary("e")))).read_text()
    )
    assert payload["applied_files"][0] == {"path": primary("e"), "sha256": digest}
