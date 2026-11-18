import base64
import json
import sys
from types import SimpleNamespace

from axiom_encode.notary.producer_runtime import expected_outputs, read_encoder_outputs
from axiom_encode.notary.producer_worker import main


def test_worker_cli_arguments_and_output_match_real_cli_helpers(tmp_path, monkeypatch):
    from axiom_encode import cli
    from axiom_encode.harness.evals import parse_runner_spec
    from axiom_encode.toolchain import local_corpus_release_verification

    job = tmp_path / "job"
    job.mkdir()
    request = {
        "citation": "42 CFR 435.145",
        "model": "fixture:model",
        "corpus_public_key": base64.b64encode(b"f" * 32).decode(),
        "codex_binary": "/opt/axiom/codex",
        "output_root": str(job / "work/output"),
        "corpus_path": "/opt/axiom/corpus",
        "engine_path": "/opt/axiom/engine",
        "lane_path": "/opt/axiom/rulespec-us",
        "dependency_roots": {"uk": "/opt/axiom/rulespec-uk"},
        "encoder_identity": {},
        "dependency_inventory": {},
    }
    (job / "request.json").write_text(json.dumps(request))
    (job / "auth.json").write_text('{"fixture":true}')
    monkeypatch.setattr(
        "axiom_encode.notary.deployment.require_running_identity", lambda *args: None
    )
    monkeypatch.setattr(cli, "_record_encode_outcome", lambda **kwargs: None)
    monkeypatch.setattr(sys, "argv", ["worker", str(job)])
    import os

    for name in ("HOME", "CODEX_HOME", "AXIOM_ENCODE_CODEX_BIN"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    monkeypatch.chdir(tmp_path)

    # Existing helper is exercised; only the expensive CLI/model operation is
    # replaced. Argument types/runner naming/output helpers remain real.
    def encode():
        assert (
            sys.argv[sys.argv.index("--rulespec-dependency-root") + 1]
            == "/opt/axiom/rulespec-uk"
        )
        assert "--apply" not in sys.argv
        import os

        assert os.environ["AXIOM_ENCODE_CODEX_BIN"] == request["codex_binary"]
        assert os.environ["CODEX_HOME"] == str(job / "work/codex-home")
        mapping = expected_outputs(request["citation"], request["model"])
        paths = list(mapping.values())
        for relative in paths:
            target = job / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fixture output")
        result = SimpleNamespace(
            output_file=str(job / paths[0]),
            runner=parse_runner_spec("codex:" + request["model"]).name,
            model=request["model"],
            success=True,
            generation_prompt_sha256="a" * 64,
        )
        cli._record_encode_outcome(db_path=None, result=result, run=None, outcome=None)

    monkeypatch.setattr(cli, "main", encode)
    with local_corpus_release_verification(base64.b64encode(b"o" * 32).decode()):
        main()
    result = json.loads((job / "result.json").read_bytes())
    assert result["success"] is True
    assert (
        len(
            read_encoder_outputs(
                job, result, expected_outputs(request["citation"], request["model"])
            )
        )
        == 2
    )
    assert "auth" not in json.dumps(result)
