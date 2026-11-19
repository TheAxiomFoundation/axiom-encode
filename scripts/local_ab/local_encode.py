"""Local, verification-only encode driver for subscription A/B runs.

Usage:
    python -I local_encode.py <pins_dir> -- encode <citation> [encode args...]

Runs ``axiom_encode.cli.main`` in-process inside
``toolchain.local_corpus_release_verification`` with the public corpus, apply
and eval roots read from ``<pins_dir>`` (the same stand-in ``axiom-encode ci``
uses). There is no signing broker and no apply.

Guards:
- refuses to start when any paid-provider, R2 or Supabase credential is in the
  environment, so model calls can only go through the Codex/Claude CLI logins;
- refuses ``--apply``.

Observation hook: when ``LOCAL_AB_ATTEMPT_LOG`` is set, every rejected attempt's
validator issues are appended to that JSONL file. The hook does not change
encoder behaviour.
"""

import json
import os
import sys
import time
from pathlib import Path

FORBIDDEN = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "AZURE_OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "AXIOM_ENCODE_SUPABASE_KEY",
    "SUPABASE_KEY",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
)


def main() -> int:
    bad = [
        key
        for key in os.environ
        if key in FORBIDDEN
        or key.endswith(("_API_KEY", "_TOKEN"))
        or any(part in key for part in ("SUPABASE", "SECRET", "PRIVATE_KEY"))
        or key.startswith("R2_")
    ]
    if bad:
        sys.exit(f"refusing: provider/sync credentials in env: {bad}")
    # Refuse --apply in every spelling argparse accepts (--apply=x, --appl).
    if any(arg.startswith("--ap") for arg in sys.argv[1:]):
        sys.exit("refusing: --apply is not allowed")
    if len(sys.argv) < 3 or sys.argv[2] != "--":
        sys.exit(__doc__)

    pins = Path(sys.argv[1])
    argv = sys.argv[3:]

    def read(name: str) -> str:
        return (pins / name).read_text().strip()

    from axiom_encode import cli, toolchain

    log_path = os.environ.get("LOCAL_AB_ATTEMPT_LOG")
    base = cli._FailedEncodeAttempt

    class _LoggedFailedEncodeAttempt(base):
        __slots__ = ()

        def __new__(cls, *args, **kwargs):
            obj = base.__new__(cls, *args, **kwargs)
            if log_path:
                try:
                    with open(log_path, "a") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "t": time.time(),
                                    "error": obj.error,
                                    "validation_issue_count": obj.validation_issue_count,
                                    "validation_issues": list(
                                        obj.validation_issues or ()
                                    ),
                                    "full_validation_issues": list(
                                        obj.full_validation_issues or ()
                                    ),
                                }
                            )
                            + "\n"
                        )
                except Exception as exc:  # observation must never break a run
                    print(f"attempt log failed: {exc}", file=sys.stderr)
            return obj

    cli._FailedEncodeAttempt = _LoggedFailedEncodeAttempt
    sys.argv = ["axiom-encode", *argv]
    with toolchain.local_corpus_release_verification(
        read("corpus_release_root.pub"),
        apply_public_key=read("AXIOM_ENCODE_APPLY_SIGNING_PUBLIC_KEY.pub"),
        eval_public_key=read("AXIOM_ENCODE_EVAL_SIGNING_PUBLIC_KEY.pub"),
    ):
        return cli.main() or 0


if __name__ == "__main__":
    sys.exit(main())
