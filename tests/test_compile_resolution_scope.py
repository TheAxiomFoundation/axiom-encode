from pathlib import Path

import pytest

from axiom_encode.harness import validator_pipeline as vp


def test_standalone_compile_has_fresh_scope_and_cleans_up(monkeypatch):
    pipeline = object.__new__(vp.ValidatorPipeline)
    seen = []

    def compile_impl(self, rules, output):
        cache = vp._RULESPEC_RESOLUTION_CACHE.get()
        assert cache is not None
        seen.append(cache)
        return "result", None

    monkeypatch.setattr(
        vp.ValidatorPipeline, "_compile_rulespec_to_artifact_impl", compile_impl
    )
    for _ in range(2):
        assert pipeline._compile_rulespec_to_artifact(Path("rules"), Path("out")) == (
            "result",
            None,
        )
        assert vp._RULESPEC_RESOLUTION_CACHE.get() is None
    assert seen[0] is not seen[1]


def test_compile_reuses_enclosing_validation_scope_and_cleans_up_errors(monkeypatch):
    pipeline = object.__new__(vp.ValidatorPipeline)

    def compile_impl(self, rules, output):
        assert vp._RULESPEC_RESOLUTION_CACHE.get() is enclosing
        raise ValueError("compile failed")

    monkeypatch.setattr(
        vp.ValidatorPipeline, "_compile_rulespec_to_artifact_impl", compile_impl
    )
    with vp._rulespec_resolution_cache_scope() as enclosing:
        with pytest.raises(ValueError, match="compile failed"):
            pipeline._compile_rulespec_to_artifact(Path("rules"), Path("out"))
        assert vp._RULESPEC_RESOLUTION_CACHE.get() is enclosing
    assert vp._RULESPEC_RESOLUTION_CACHE.get() is None
