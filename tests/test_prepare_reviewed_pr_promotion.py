"""The workflow passes selectors, not caller-provided review authority."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode import prepare_signed_backfill as helper
from axiom_encode import reviewed_pr_admission as admission


def test_workflow_preflight_uses_locked_runtime_before_identity_checks():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load(
        (root / ".github/workflows/targeted-signed-reencode.yml").read_text()
    )
    steps = workflow["jobs"]["encode"]["steps"]
    names = [step.get("name") for step in steps]
    setup = names.index("Install locked preflight runtime")
    preflight = names.index("Preflight live reviewed PR promotion")
    identities = names.index("Verify immutable checkout identities")
    assert names.index("Install uv") < names.index("Set up Python") < setup
    assert setup < preflight < identities < names.index("Build Axiom rules engine")
    assert steps[setup]["working-directory"] == "axiom-encode"
    assert 'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"' in steps[setup]["run"]
    assert (
        "uv sync --no-config --locked --python 3.13 --extra api --no-dev"
        in steps[setup]["run"]
    )
    assert (
        "axiom-encode/.venv/bin/python "
        "axiom-encode/scripts/prepare_reviewed_pr_promotion.py prepare"
    ) in steps[preflight]["run"]
    assert steps[preflight]["id"] == "reviewed_pr_admission"


def test_preflight_helper_resolves_canonical_path_in_installed_runtime(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts/prepare_reviewed_pr_promotion.py"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import runpy, sys; "
            "script = runpy.run_path(sys.argv[1]); "
            "helper = script['load']('prepare_signed_backfill'); "
            "print(helper.citation_rulespec_path("
            "'us-az/manual/des/faa5/na-utility-expenses-and-allowances/block-13'))",
            str(script_path),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == (
        "us-az/policies/des/faa5/na-utility-expenses-and-allowances/block-13.yaml"
    )


@pytest.fixture
def script(monkeypatch, tmp_path):
    path = (
        Path(__file__).resolve().parents[1] / "scripts/prepare_reviewed_pr_promotion.py"
    )
    spec = importlib.util.spec_from_file_location("promotion_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv("GITHUB_REPOSITORY", "TheAxiomFoundation/axiom-encode")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    monkeypatch.setenv("DISPATCH_ACTOR", "PavelMakarchuk")
    monkeypatch.setattr(
        "sys.argv",
        [
            str(path),
            "prepare",
            "--repo",
            str(tmp_path),
            "--selector",
            str(tmp_path / "selector.json"),
        ],
    )
    return module


def dispatch():
    return {
        "country": "us",
        "citation": "us/statute/7/2015/f",
        "rulespec_ref": "a" * 40,
        "replace_rulespec_path": "us/statutes/7/2015/f.yaml",
        "pr_base_branch": "codex/candidate",
        "open_pr": True,
        "source_bundle_json": json.dumps(
            {
                "schema": "axiom-encode/reviewed-pr-promotion/v1",
                "rulespec_pr": 42,
                "transaction": {
                    "schema": "axiom-encode/atomic-source-transaction/v5",
                    "source_bundle": [],
                    "canonical_refresh_bundle": [],
                    "primary_required_test_cases": [],
                    "require_complete_source_unit": True,
                    "manifest_only_refresh": False,
                    "reviewed_candidate_promotion": True,
                },
            }
        ),
    }


def test_prepare_derives_only_selector(script, monkeypatch, tmp_path):
    calls = []
    verifier = SimpleNamespace(
        PromotionSelector=admission.PromotionSelector,
        verify_reviewed_pr=lambda selector: calls.append(selector),
    )
    monkeypatch.setattr(
        script,
        "load",
        lambda name: verifier if name == "reviewed_pr_admission" else helper,
    )
    monkeypatch.setattr(script.subprocess, "check_output", lambda *a, **kw: "a" * 40)
    monkeypatch.setenv("DISPATCH_JSON", json.dumps(dispatch()))
    script.main()
    selector = json.loads((tmp_path / "selector.json").read_text())
    assert set(selector) == set(admission.PromotionSelector.__dataclass_fields__)
    assert selector["pr"] == 42 and selector["run_id"] == "12345"
    assert calls == [admission.selector_from_payload(selector)]
    assert (tmp_path / "output").read_text() == "enabled=true\n"


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        '{"schema":"axiom-encode/atomic-source-transaction/v5","reviewed_candidate_promotion":true}',
    ],
)
def test_old_route_has_no_network_preflight(script, monkeypatch, tmp_path, raw):
    def forbidden(*args):
        pytest.fail("legacy route must not query live authority")

    monkeypatch.setattr(
        script,
        "load",
        lambda name: (
            SimpleNamespace(verify_reviewed_pr=forbidden)
            if name == "reviewed_pr_admission"
            else helper
        ),
    )
    monkeypatch.setenv("DISPATCH_JSON", json.dumps({"source_bundle_json": raw}))
    script.main()
    assert not (tmp_path / "selector.json").exists()
    assert (tmp_path / "output").read_text() == "enabled=false\n"


@pytest.mark.parametrize("change", ["repair", "path", "head", "publish", "branch"])
def test_preflight_refuses_mixed_or_unbound_inputs(
    script, monkeypatch, tmp_path, change
):
    payload = dispatch()
    if change == "repair":
        payload["repair_mode"] = "full_artifact"
    elif change == "path":
        payload["replace_rulespec_path"] = "us/statutes/7/2015/other.yaml"
    elif change == "publish":
        payload["open_pr"] = False
    elif change == "branch":
        monkeypatch.setenv("GITHUB_REF", "refs/heads/untrusted")
    monkeypatch.setattr(
        script,
        "load",
        lambda name: admission if name == "reviewed_pr_admission" else helper,
    )
    monkeypatch.setattr(
        script.subprocess,
        "check_output",
        lambda *a, **kw: "b" * 40 if change == "head" else "a" * 40,
    )
    monkeypatch.setenv("DISPATCH_JSON", json.dumps(payload))
    with pytest.raises(ValueError):
        script.main()
    assert not (tmp_path / "selector.json").exists()


def test_publication_rechecks_provisioned_not_dispatch_evidence(
    script, monkeypatch, tmp_path
):
    evidence = {"selector": {"run_id": "12345"}}
    attestation = tmp_path / "runtime-attestation.json"
    attestation.write_text(json.dumps({"reviewed_pr_admission": evidence}))
    calls = []
    monkeypatch.setattr(
        script,
        "load",
        lambda name: SimpleNamespace(
            revalidate_admission=lambda value: calls.append(value)
        ),
    )
    monkeypatch.setattr(
        "sys.argv", ["prepare", "verify-publication", "--attestation", str(attestation)]
    )
    script.main()
    assert calls == [evidence]
