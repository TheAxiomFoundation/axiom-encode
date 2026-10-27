import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode import cli
from axiom_encode.harness import validator_pipeline
from axiom_encode.harness.validator_pipeline import ValidatorPipeline


@pytest.mark.parametrize("kind", ["atomic", "composition"])
@pytest.mark.parametrize("composer_available", [True, False])
def test_companion_compiler_respects_module_kind(tmp_path, kind, composer_available):
    root = tmp_path / "rulespec-us"
    policy = root / "us-az"
    module = policy / "policies/example/composition.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(f"format: rulespec/v1\nmodule:\n  kind: {kind}\nrules: []\n")
    companion = module.with_suffix(".test.yaml")
    companion.write_text("[]\n")
    spec = root / "programs/us-az/snap/fy-2026.yaml"
    spec.parent.mkdir(parents=True)
    spec.write_text(
        "program: us-az/snap\nperiod: 2026-01\noutputs: [result]\nscope:\n  federal: []\n  state: [policies/example/composition]\n"
    )
    compose = tmp_path / "composer"
    compose.write_text(
        f'#!{sys.executable}\nimport sys,pathlib\na=sys.argv\npathlib.Path(a[a.index("-o")+1]).write_text("format: rulespec/v1\\nrules: []\\n")\n'
    )
    compose.chmod(0o755)
    pipeline = ValidatorPipeline(
        policy_repo_path=policy,
        axiom_rules_path=tmp_path / "engine",
        axiom_compose_path=compose if composer_available else None,
        enable_oracles=False,
        local_corpus_release=None,
    )
    pipeline._axiom_rules_binary = lambda: tmp_path / "engine"
    calls = []

    def compile_stub(**kwargs):
        calls.append(kwargs)
        kwargs["output"].write_text(
            json.dumps({"program": {"parameters": [], "derived": [], "relations": []}})
        )
        return subprocess.CompletedProcess([], 0, "", "")

    with (
        patch.object(cli, "run_rulespec_compile", compile_stub),
        patch.object(validator_pipeline, "run_rulespec_compile", compile_stub),
    ):
        result = cli._execute_rulespec_test_file(
            companion,
            binary=tmp_path / "engine",
            pipeline=pipeline,
            axiom_rules_path=tmp_path / "engine",
            env={},
            rulespec_roots=(root,),
            tmp_path=tmp_path,
            compiled_cache={},
            policy_repo_path=policy,
        )
    if kind == "composition" and not composer_available:
        assert calls == []
        assert (
            "requires an explicit axiom-compose executable"
            in result["failures"][0]["message"]
        )
    else:
        assert len(calls) == 1
        assert bool(calls[0].get("composed")) == (kind == "composition")
        assert calls[0]["program"].name == (
            "composed-program.yaml" if kind == "composition" else module.name
        )
        assert result["compiled"] == 1


def test_waiver_worker_keeps_explicit_composer(tmp_path):
    composer = tmp_path / "composer"
    with (
        patch.object(cli, "LocalCorpusRelease", return_value=object()),
        patch.object(
            cli, "_fingerprint_validation_waiver_modules", return_value=[]
        ) as fingerprint,
    ):
        cli._fingerprint_waiver_chunk(
            ["us/policies/test.yaml"],
            str(tmp_path),
            "/corpus",
            "/engine",
            (),
            ("/corpus", "release", "0" * 64, "key"),
            str(composer),
        )
    assert fingerprint.call_args.kwargs["axiom_compose_path"] == composer


def test_composer_diagnostics_normalize_across_machine_locations(tmp_path):
    def normalized(label):
        composer = tmp_path / label / "bin/composer"
        replacements = cli._validation_waiver_path_replacements(
            root=tmp_path / "rulespec-us",
            axiom_rules_paths=[tmp_path / "engine"],
            binaries=[],
            tmp_path=tmp_path / "scratch",
            rulespec_roots=(),
            pipelines=[
                SimpleNamespace(local_corpus_release=None, axiom_compose_path=composer)
            ],
        )
        text = f"{composer}: failure"
        for path, replacement in replacements.items():
            text = text.replace(path, replacement)
        return text

    assert normalized("one") == normalized("two") == "<axiom-compose>: failure"


@pytest.mark.parametrize(
    "command", ["validate", "test", "compile", "fingerprint", "audit"]
)
def test_public_cli_accepts_explicit_composer(command, tmp_path, monkeypatch):
    composer = tmp_path / "composer"
    composer.write_text("#!/bin/sh\nexit 0\n")
    composer.chmod(0o755)
    args = ["axiom-encode"]
    if command in {"fingerprint", "audit"}:
        args += [
            "validation-waivers",
            command,
            "--root",
            str(tmp_path),
            "--corpus-path",
            str(tmp_path),
        ]
        args += (
            ["module.yaml"]
            if command == "fingerprint"
            else ["--protected-base", "base.yaml", "--changed-paths", "changes.txt"]
        )
        handler = "cmd_validation_waivers"
    else:
        args += [command]
        args += ["--root", str(tmp_path)] if command == "test" else ["module.yaml"]
        if command == "validate":
            args += ["--corpus-path", str(tmp_path)]
        handler = "cmd_" + command
    args += [
        "--axiom-rules-engine-path",
        str(tmp_path),
        "--axiom-compose-path",
        str(composer),
    ]
    monkeypatch.setattr(sys, "argv", args)
    with patch.object(cli, handler) as call:
        cli.main()
    assert call.call_args.args[0].axiom_compose_path == composer


@pytest.mark.parametrize("command", ["validate", "test", "compile", "fingerprint"])
def test_cli_composer_reaches_pipeline(command, tmp_path, monkeypatch):
    root = tmp_path / "rulespec-us"
    policy = root / "us"
    policy.mkdir(parents=True)
    module = policy / "statutes/a.yaml"
    module.parent.mkdir()
    module.write_text("format: rulespec/v1\nrules: []\n")
    module.with_suffix(".test.yaml").write_text("[]\n")
    composer = tmp_path / "composer"
    composer.write_text("#!/bin/sh\nexit 0\n")
    composer.chmod(0o755)
    args = SimpleNamespace(
        files=[module],
        file=module,
        modules=[Path("us/statutes/a.yaml")],
        paths=[],
        root=policy if command == "test" else root,
        corpus_path=tmp_path,
        axiom_rules_path=tmp_path,
        axiom_compose_path=composer,
        oracle=None,
        json=False,
        axiom_rules_engine_ref=None,
    )
    observed = []

    class Captured(BaseException):
        pass

    def capture(**kwargs):
        observed.append(kwargs)
        raise Captured()

    monkeypatch.setattr(cli, "ValidatorPipeline", capture)
    monkeypatch.setattr(
        cli, "_resolve_validation_repo_roots", lambda *a: (policy, tmp_path)
    )
    monkeypatch.setattr(cli, "_canonical_validation_checkout_root", lambda *a: root)
    monkeypatch.setattr(cli, "find_policy_repo_root", lambda *a: policy)
    monkeypatch.setattr(cli, "load_rulespec_local_corpus_release", lambda *a: object())
    handler = (
        cli._cmd_validation_waivers_fingerprint
        if command == "fingerprint"
        else getattr(cli, "cmd_" + command)
    )
    with pytest.raises(Captured):
        handler(args)
    assert observed[0]["axiom_compose_path"] == composer
