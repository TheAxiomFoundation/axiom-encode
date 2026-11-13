"""Real local Git staging; fixture executable bytes are never run or signed."""

import subprocess
from types import SimpleNamespace

import pytest

from axiom_encode.harness import same_case_execution as e
from tests.test_bound_form_operands import CITATION, CONTENT, FIXTURE, ROOT, SOURCE


def local_world(tmp_path):
    root = tmp_path / "rulespec-ca"
    root.mkdir()
    target = root / ROOT
    target.parent.mkdir(parents=True)
    target.write_text(CONTENT)
    companion = target.with_suffix(".test.yaml")
    companion.write_bytes((FIXTURE / "diagnostic.test.yaml").read_bytes())
    sibling = root / "ca/policies/unrelated.yaml"
    sibling.write_text("format: rulespec/v1\nrules: []\n")
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    binary = tmp_path / "fixture-engine"
    binary.write_bytes(b"not executed")
    pipe = SimpleNamespace(
        require_complete_source_unit=True,
        _declared_engine_pin=lambda: SimpleNamespace(sha="a" * 40),
        policy_repo_path=root / "ca",
        rulespec_dependency_roots=(),
        _rulespec_compile_roots=lambda: [root],
        _rulespec_test_path=lambda _: companion,
        _complete_source_unit_import_bindings=lambda _: (),
        _source_texts_for_rulespec_content=lambda _: {CITATION: SOURCE},
        _proof_source_texts_for_rulespec_content=lambda *a, **k: {CITATION: SOURCE},
        _axiom_rules_binary=lambda: binary,
    )
    return pipe, target


def test_only_bound_local_form_gets_exact_two_file_snapshot(tmp_path):
    pipe, target = local_world(tmp_path)
    with e.compilation_snapshot(pipe, target) as capture:
        assert capture is not None
        assert {f.relative for f in capture.files} == {
            ROOT,
            ROOT.removesuffix(".yaml") + ".test.yaml",
        }
        assert capture.fresh()
        assert capture.evidence() is None
        assert not (capture.snapshot_root / "ca/policies/unrelated.yaml").exists()


@pytest.mark.parametrize(
    "change",
    [
        "unknown_module",
        "unbound_graph",
        "missing_pin",
        "dependencies",
        "not_git",
        "symlink",
        "missing_companion",
        "wrong_source",
    ],
)
def test_local_snapshot_admission_stays_closed(tmp_path, change):
    pipe, target = local_world(tmp_path)
    if change == "unknown_module":
        target.write_text(
            "format: rulespec/v1\nmodule: {summary: Unrelated local module}\nrules: []\n"
        )
    elif change == "unbound_graph":
        target.write_text(CONTENT.replace("min(", "max(", 1))
    elif change == "missing_pin":
        pipe._declared_engine_pin = lambda: None
    elif change == "dependencies":
        pipe.rulespec_dependency_roots = (tmp_path / "rulespec-us",)
    elif change == "not_git":
        import shutil

        shutil.rmtree(pipe.policy_repo_path.parent / ".git")
    elif change == "symlink":
        other = tmp_path / "other.yaml"
        other.write_text(CONTENT)
        target.unlink()
        target.symlink_to(other)
    elif change == "missing_companion":
        target.with_suffix(".test.yaml").unlink()
    elif change == "wrong_source":
        pipe._proof_source_texts_for_rulespec_content = lambda *a, **k: {
            CITATION: SOURCE + "changed"
        }
    assert e._snapshot_plan(pipe, target) is None


def test_imported_module_cannot_use_local_snapshot_binders(tmp_path, monkeypatch):
    import yaml

    from axiom_encode.harness import actual_floor_evidence, bound_form_operands

    pipe, target = local_world(tmp_path)
    payload = yaml.safe_load(target.read_text())
    payload["imports"] = ["ca:policies/provider#out"]
    target.write_text(yaml.safe_dump(payload))

    def forbidden(*args, **kwargs):
        pytest.fail("imported modules must use the derived-provider route")

    monkeypatch.setattr(actual_floor_evidence, "bind_same_page_floors", forbidden)
    monkeypatch.setattr(bound_form_operands, "bind_local_form_snapshot", forbidden)
    assert e._snapshot_plan(pipe, target) is None
