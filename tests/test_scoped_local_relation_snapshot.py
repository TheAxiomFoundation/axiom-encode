"""Local relation recording must not override scoped ownership; no engine execution."""

import subprocess
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.harness import (
    actual_floor_evidence,
    actual_local_relation_evidence,
    scoped_execution,
)
from axiom_encode.harness import same_case_execution as e
from tests.test_same_page_floor_binding import BODY, CITATION, CONTENT


@pytest.mark.parametrize(
    "mode",
    ["relation", "ambiguous", "activation_failure", "imports", "relation_changes"],
)
def test_local_relation_recording_respects_scoped_ownership(
    tmp_path, monkeypatch, mode
):
    root = tmp_path / "repo"
    target = root / "ca/policies/cra/t1-2025/federal-spouse-transfer-amounts.yaml"
    target.parent.mkdir(parents=True)
    payload = yaml.safe_load(CONTENT)
    if mode == "imports":
        payload["imports"] = ["ca:policies/provider#out"]
    target.write_text(yaml.safe_dump(payload))
    companion = target.with_suffix(".test.yaml")
    companion.write_text("[]\n")
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
    binary = tmp_path / "not-an-engine"
    binary.write_bytes(b"never executed")
    sources = {CITATION: BODY}

    pipe = SimpleNamespace(
        require_complete_source_unit=True,
        _declared_engine_pin=lambda: SimpleNamespace(sha="a" * 40),
        policy_repo_path=root / "ca",
        rulespec_dependency_roots=(),
        _rulespec_compile_roots=lambda: [root],
        _rulespec_test_path=lambda _: companion,
        _complete_source_unit_import_bindings=lambda _: (),
        _source_texts_for_rulespec_content=lambda _: sources,
        _proof_source_texts_for_rulespec_content=lambda *args, **kwargs: sources,
        _axiom_rules_binary=lambda: binary,
    )
    activations = []

    def refuse_activation(*args):
        activations.append(True)
        return False

    owner = (SimpleNamespace(activate=refuse_activation), "_investment_adjustment")
    owners = (
        (owner, owner)
        if mode == "ambiguous"
        else (owner,)
        if mode == "activation_failure"
        else ()
    )

    def eligible(*args):
        assert mode != "imports"
        return owners

    monkeypatch.setattr(scoped_execution, "eligible_adapters", eligible)
    relation_calls = []

    def relation_eligible(payload):
        assert mode not in {"imports", "ambiguous", "activation_failure"}
        relation_calls.append(True)
        return mode != "relation_changes" or len(relation_calls) == 1

    def no_floor(*args):
        assert mode != "imports"
        return False

    monkeypatch.setattr(actual_floor_evidence, "bind_same_page_floors", no_floor)
    monkeypatch.setattr(
        actual_local_relation_evidence,
        "local_relation_snapshot_eligible",
        relation_eligible,
    )
    with e.compilation_snapshot(pipe, target) as capture:
        if mode == "relation":
            assert capture is not None
            assert capture.evidence() is None
            assert not getattr(capture, "_investment_adjustment_owned", False)
            assert not getattr(capture, "_capital_gift_owned", False)
        else:
            assert capture is None
    assert len(activations) == (1 if mode == "activation_failure" else 0)

    if mode in {"relation", "relation_changes"}:
        assert len(relation_calls) == 2
