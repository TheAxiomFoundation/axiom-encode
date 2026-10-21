"""Infra checks for workflow steps, and the envelope a failed step emits.

A workflow step that refuses to continue keeps its ``echo "<message>" >&2`` and
its ``exit 1``, and between them calls
``python axiom-encode/scripts/infra_issue.py record <check id> "<message>" || true``.
The script writes a GitHub error annotation that names the check and its fix,
appends one ``axiom-encode/validation-issue/v2`` record to
``$AXIOM_INFRA_ISSUES_FILE`` (default ``$RUNNER_TEMP/axiom-infra-issues.jsonl``)
and to the step summary, and always exits 0. The refusal itself never depends
on the script: if it cannot run, the step still prints the message and fails.

This module has no package imports, so the workflow can load it with the
runner's system Python before the encoder is installed. ``checks.load_checks``
registers these definitions in the same registry as the validator checks.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

ISSUE_SCHEMA_VERSION = "axiom-encode/validation-issue/v2"
_WORKFLOW = ".github/workflows/targeted-signed-reencode.yml"
_INPUT_WHY = (
    "The workflow refuses a dispatch whose inputs it cannot bind to one exact, "
    "authenticated transaction. Nothing was encoded, so the encoder output is "
    "not at fault and a retry with the same inputs fails the same way."
)

#: One definition per refusing workflow step. ``site`` names the step.
INFRA_CHECK_DEFINITIONS: tuple[dict[str, str], ...] = (
    {
        "id": "infra:validate-country-routing-input",
        "title": "Dispatch routing or queue inputs are invalid",
        "requires": (
            "The country is routable, and a queue target comes from an "
            "authenticated dispatcher on its first run attempt."
        ),
        "why": _INPUT_WHY,
        "fix": (
            "Dispatch queue items only through the queue dispatcher; for a "
            "manual run, leave the queue inputs empty and pass a routable "
            "country."
        ),
        "site": f"{_WORKFLOW}:Validate country routing input",
    },
    {
        "id": "infra:verify-immutable-checkout-identities",
        "title": "A checkout is not pinned to the requested commit",
        "requires": (
            "Every pinned checkout ref is a full lowercase commit SHA, and the "
            "checkout resolves to exactly that commit."
        ),
        "why": (
            "Signed provenance names exact commits. A branch name or a moved "
            "ref would sign work against code nobody reviewed."
        ),
        "fix": (
            "Pass full 40-character lowercase commit SHAs for the named ref "
            "input (not a branch or a short SHA) and re-dispatch."
        ),
        "site": f"{_WORKFLOW}:Verify immutable checkout identities",
    },
    {
        "id": "infra:resolve-trusted-prior-run-repair-candidate",
        "title": "Repair replay inputs are outside the trusted shape",
        "requires": (
            "A repair replay names a different decimal workflow run and stays "
            "within the reviewed replay limits (targets, dependents, bundles)."
        ),
        "why": _INPUT_WHY,
        "fix": (
            "Point `repair_run_id` at the failed run whose candidate you replay, "
            "and reduce the dispatch to the shape the message names."
        ),
        "site": f"{_WORKFLOW}:Resolve trusted prior-run repair candidate",
    },
    {
        "id": "infra:validate-atomic-source-inputs",
        "title": "Source inputs mix transaction modes",
        "requires": (
            "One dispatch is one atomic transaction: source bundles, imports, "
            "legacy or dependent migrations and queue authorization combine "
            "only in the reviewed ways."
        ),
        "why": _INPUT_WHY,
        "fix": (
            "Split the dispatch into separate runs as the message says (for "
            "example merge legacy replacements first), or drop the conflicting "
            "input."
        ),
        "site": f"{_WORKFLOW}:Validate atomic source inputs",
    },
    {
        "id": "infra:fetch-pinned-signed-corpus-release-object",
        "title": "The pinned corpus release cannot be fetched or verified",
        "requires": (
            "The RuleSpec toolchain pins exactly one signed corpus release, and "
            "the pinned corpus checkout carries its signed provenance."
        ),
        "why": (
            "Encoding reads source text only from the signed release; without "
            "it the run has no authenticated source."
        ),
        "fix": (
            "Check the RuleSpec repo's toolchain pin and the axiom-corpus ref: "
            "the corpus checkout must include the signed release provenance "
            "and its locks or provisions. Retry after the release is published."
        ),
        "site": f"{_WORKFLOW}:Fetch pinned signed corpus release object",
    },
)


def register_checks(declare: Callable[..., Any]) -> None:
    """Register every infra check with ``checks.declare_check``."""
    for definition in INFRA_CHECK_DEFINITIONS:
        declare(
            definition["id"],
            title=definition["title"],
            requires=definition["requires"],
            why=definition["why"],
            fix=definition["fix"],
            cause="infra",
            gate="workflow",
            site=definition["site"],
        )


def _definition(check_id: str) -> dict[str, str] | None:
    for definition in INFRA_CHECK_DEFINITIONS:
        if definition["id"] == check_id:
            return definition
    return None


def infra_issue_record(check_id: str, message: str) -> dict[str, Any]:
    """The ``validation-issue/v2`` record of one refused workflow step."""
    definition = _definition(check_id)
    step = definition["site"].partition(":")[2] if definition else None
    subject = {"field": f"step:{step}"} if step else None
    fingerprint = hashlib.sha256(
        json.dumps(
            {"check": check_id, "subject": subject or {}, "expected": None},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "schema": ISSUE_SCHEMA_VERSION,
        "gate": "workflow",
        "kind": check_id,
        "message": message,
        "category": check_id.partition("/")[0],
        "locator": subject["field"] if subject else None,
        "line": None,
        "value": None,
        "clause": None,
        "check": check_id,
        "cause": "infra",
        "subject": subject,
        "expected": None,
        "found": message[:400] or None,
        "evidence": None,
        "fingerprint": fingerprint,
        "fix": definition["fix"] if definition else None,
        "run": {
            key: os.environ[name]
            for key, name in (
                ("repository", "GITHUB_REPOSITORY"),
                ("run_id", "GITHUB_RUN_ID"),
                ("run_attempt", "GITHUB_RUN_ATTEMPT"),
                ("job", "GITHUB_JOB"),
            )
            if os.environ.get(name)
        }
        or None,
    }


def _annotation_text(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _annotation_property(text: str) -> str:
    return _annotation_text(text).replace(":", "%3A").replace(",", "%2C")


def record(check_id: str, message: str) -> int:
    """Emit the envelope for a refused step. The caller exits; this returns 0."""
    definition = _definition(check_id)
    if definition is None:
        # The ratchet test keeps workflow ids declared; still annotate the step.
        print(
            f"::error title={_annotation_property(check_id)}::{_annotation_text(message)}"
        )
        return 0
    record = infra_issue_record(check_id, message)
    print(
        f"::error title={_annotation_property(check_id)}::"
        f"{_annotation_text(definition['title'])}: {_annotation_text(message)}"
        f"%0AFix: {_annotation_text(definition['fix'])}"
    )
    issues_file = os.environ.get("AXIOM_INFRA_ISSUES_FILE") or (
        str(Path(os.environ["RUNNER_TEMP"]) / "axiom-infra-issues.jsonl")
        if os.environ.get("RUNNER_TEMP")
        else ""
    )
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    for path, text in (
        (issues_file, line + "\n"),
        (
            os.environ.get("GITHUB_STEP_SUMMARY", ""),
            f"**{check_id}** {definition['title']}: {message}\n\n"
            f"Fix: {definition['fix']}\n\n",
        ),
    ):
        if not path:
            continue
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(text)
        except OSError:
            pass
    return 0


_CHECK_ID_RE = re.compile(r"^infra:[a-z0-9-]+$")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2 or args[0] != "record" or not _CHECK_ID_RE.match(args[1]):
        print(
            "usage: infra_issue.py record infra:<step> [message ...]",
            file=sys.stderr,
        )
        return 2
    return record(args[1], " ".join(args[2:]))


if __name__ == "__main__":
    raise SystemExit(main())
