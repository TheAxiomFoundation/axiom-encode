"""Unprivileged worker entrypoint. It holds no producer, actor or notary key.

The Linux controller supplies a fresh job directory and immutable command.
Results are proposals; only the controller observes and signs after killing
all worker descendants. No output from this process is signature authority.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main():
    job = Path(sys.argv[1])
    request = json.loads((job / "request.json").read_bytes())
    from .canonical import jcs_dumps
    from .deployment import require_running_identity

    require_running_identity(
        request["encoder_identity"], jcs_dumps(request["dependency_inventory"])
    )
    if request.get("runtime_kind") == "deterministic":
        from .deterministic_runtime import run_adapter

        run_adapter(job, request)
        return
    # The worker receives only this job's personal Codex login. It never uses
    # the host's HOME or reads a signing/service credential from its environment.
    work = job / "work"
    work.mkdir(mode=0o700)
    codex_home = work / "codex-home"
    codex_home.mkdir(mode=0o700)
    auth = codex_home / "auth.json"
    auth.write_bytes((job / "auth.json").read_bytes())
    auth.chmod(0o600)
    os.environ["HOME"] = str(work)
    os.environ["CODEX_HOME"] = str(codex_home)
    os.environ["AXIOM_ENCODE_CODEX_BIN"] = request["codex_binary"]
    os.chdir(work)
    from axiom_encode import cli
    from axiom_encode.harness.evals import _rulespec_test_path

    original = cli._record_encode_outcome
    captured = []

    def capture(*, db_path, result, run, outcome):
        recorded = original(db_path=db_path, result=result, run=run, outcome=outcome)
        output = Path(result.output_file)
        root = Path(request["output_root"]) / result.runner
        files = {}
        for path in (output, _rulespec_test_path(output)):
            if path.exists():
                relative = path.relative_to(root).as_posix()
                files[relative] = str(path.relative_to(job))
        captured.append(
            {
                "success": result.success,
                "files": files,
                "model": result.model,
                "prompt_sha256s": [result.generation_prompt_sha256],
            }
        )
        return recorded

    cli._record_encode_outcome = capture
    sys.argv = [
        "axiom-encode",
        "encode",
        request["citation"],
        "--backend",
        "codex",
        "--model",
        request["model"],
        "--no-escalation",
        "--no-sync",
        "--output",
        request["output_root"],
        "--db",
        str(work / "encodings.db"),
        "--corpus-path",
        request["corpus_path"],
        "--axiom-rules-engine-path",
        request["engine_path"],
        "--policy-repo-path",
        request["lane_path"],
    ]
    for _, path in sorted(request["dependency_roots"].items()):
        sys.argv += ["--rulespec-dependency-root", path]
    code = 0
    try:
        from axiom_encode.toolchain import local_corpus_release_verification

        with local_corpus_release_verification(request["corpus_public_key"]):
            cli.main()
    except SystemExit as exc:
        code = exc.code
    if code != 0 or len(captured) != 1 or captured[0]["success"] is not True:
        raise SystemExit("encoder did not produce one successful completed draw")
    (job / "result.json").write_text(json.dumps(captured[0]))


if __name__ == "__main__":
    main()
