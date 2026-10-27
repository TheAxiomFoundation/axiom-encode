"""Companion batches share routing, never resolution state between modules."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode import cli, repo_routing
from axiom_encode.harness import validator_pipeline as vp
from tests.environment_assertions import expect


@pytest.fixture
def companion_command(monkeypatch, tmp_path):
    root = tmp_path / "rulespec-us" / "us"
    root.mkdir(parents=True)
    files = [root / "a.test.yaml", root / "b.test.yaml"]
    monkeypatch.setattr(cli, "_resolve_explicit_existing_directory", lambda p, **kw: p)
    monkeypatch.setattr(cli, "canonical_rulespec_root_identity", lambda p: "us")
    monkeypatch.setattr(cli, "_discover_rulespec_test_files", lambda paths, **kw: paths)
    monkeypatch.setattr(
        cli, "_canonical_validation_checkout_root", lambda p: root.parent
    )
    monkeypatch.setattr(cli, "find_policy_repo_root", lambda p: root)
    monkeypatch.setattr(cli, "_resolve_optional_axiom_compose_path", lambda p: None)
    monkeypatch.setattr(cli, "_rulespec_dependency_roots_from_args", lambda a: ())
    pipeline = SimpleNamespace(
        _axiom_rules_binary=lambda: Path("engine"),
        _rulespec_engine_env=lambda: {},
        _rulespec_compile_roots=lambda: [root],
    )
    monkeypatch.setattr(cli, "ValidatorPipeline", lambda **kw: pipeline)
    return SimpleNamespace(root=root, paths=files, json=True, axiom_rules_path=root)


def test_companion_scopes_are_per_file_and_per_invocation(
    companion_command, monkeypatch, capsys
):
    caches, routing = [], []

    def execute(path, **kwargs):
        cache = vp._RULESPEC_RESOLUTION_CACHE.get()
        expect(cache is not None)
        expect(not cache.target_files, cache.target_files)
        cache.target_files[("sentinel",)] = path
        caches.append(cache)
        routing.append(repo_routing._RULESPEC_ROUTING_CACHE.get())
        return {"cases": 1, "compiled": 1, "failures": []}

    monkeypatch.setattr(cli, "_execute_rulespec_test_file", execute)
    for _ in range(2):
        with pytest.raises(SystemExit) as exc:
            cli.cmd_test(companion_command)
        assert exc.value.code == 0
        assert vp._RULESPEC_RESOLUTION_CACHE.get() is None
        assert repo_routing._RULESPEC_ROUTING_CACHE.get() is None
        capsys.readouterr()
    assert len({id(cache) for cache in caches}) == 4
    assert routing[0] is routing[1]
    assert routing[2] is routing[3]
    assert routing[0] is not routing[2]


@pytest.mark.parametrize("error", [ValueError("failed"), SystemExit(2)])
def test_companion_scope_cleans_up_errors(companion_command, monkeypatch, error):
    def execute(*args, **kwargs):
        expect(vp._RULESPEC_RESOLUTION_CACHE.get() is not None)
        raise error

    monkeypatch.setattr(cli, "_execute_rulespec_test_file", execute)
    with pytest.raises(type(error)):
        cli.cmd_test(companion_command)
    assert vp._RULESPEC_RESOLUTION_CACHE.get() is None
    assert repo_routing._RULESPEC_ROUTING_CACHE.get() is None


def test_companion_scope_preserves_enclosing_routing(companion_command, monkeypatch):
    def execute(*args, **kwargs):
        expect(repo_routing._RULESPEC_ROUTING_CACHE.get() is routing)
        return {"cases": 1, "compiled": 0, "failures": [{"message": "bad"}]}

    monkeypatch.setattr(cli, "_execute_rulespec_test_file", execute)
    with repo_routing._rulespec_routing_cache_scope() as routing:
        with pytest.raises(SystemExit) as exc:
            cli.cmd_test(companion_command)
        assert exc.value.code == 1
        assert repo_routing._RULESPEC_ROUTING_CACHE.get() is routing
        assert vp._RULESPEC_RESOLUTION_CACHE.get() is None
    assert repo_routing._RULESPEC_ROUTING_CACHE.get() is None
