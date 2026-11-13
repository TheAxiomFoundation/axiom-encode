"""Snapshot route separation without executing or signing fixture bytes."""

import subprocess
from types import SimpleNamespace

import pytest

from axiom_encode.harness import same_case_execution as e
from axiom_encode.harness import scoped_execution
from tests.test_same_page_floor_binding import BODY, CITATION, CONTENT


@pytest.mark.parametrize(
    "mode",
    [
        "floor",
        "ambiguous",
        "activation_failure",
        "unbound",
        "ownership_changes",
        "source_changes",
    ],
)
def test_floor_route_never_overrides_scoped_ownership(tmp_path, monkeypatch, mode):
    root = tmp_path / "repo"
    target = root / "ca/policies/cra/t1-2025/federal-spouse-transfer-amounts.yaml"
    target.parent.mkdir(parents=True)
    target.write_text(CONTENT)
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
    sources = {CITATION: BODY if mode != "unbound" else "unrelated source"}
    source_calls = []

    def proof_sources(*args, **kwargs):
        source_calls.append(True)
        if mode == "source_changes" and len(source_calls) > 1:
            return {CITATION: "unrelated source"}
        return sources

    pipe = SimpleNamespace(
        require_complete_source_unit=True,
        _declared_engine_pin=lambda: SimpleNamespace(sha="a" * 40),
        policy_repo_path=root / "ca",
        rulespec_dependency_roots=(),
        _rulespec_compile_roots=lambda: [root],
        _rulespec_test_path=lambda _: companion,
        _complete_source_unit_import_bindings=lambda _: (),
        _source_texts_for_rulespec_content=lambda _: sources,
        _proof_source_texts_for_rulespec_content=proof_sources,
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
    owner_calls = []

    def eligible(*args):
        owner_calls.append(True)
        if mode == "ownership_changes" and len(owner_calls) > 1:
            return (owner, owner)
        return owners

    monkeypatch.setattr(scoped_execution, "eligible_adapters", eligible)
    with e.compilation_snapshot(pipe, target) as capture:
        if mode == "floor":
            assert capture is not None
            assert capture.evidence() is None
            assert not getattr(capture, "_investment_adjustment_owned", False)
            assert not getattr(capture, "_capital_gift_owned", False)
        else:
            assert capture is None
    assert len(activations) == (1 if mode == "activation_failure" else 0)
    if mode == "ownership_changes":
        assert len(owner_calls) == 2
    if mode == "source_changes":
        assert len(source_calls) == 2
