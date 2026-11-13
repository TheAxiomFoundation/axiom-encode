"""Infra checks for workflow steps, and the envelope a failed step emits.

A workflow step that refuses to continue keeps its ``echo "<message>" >&2`` and
its ``exit 1``, and between them calls
``python axiom-encode/scripts/infra_issue.py record <check id> "<message>" || true``.
The script writes a GitHub error annotation that names the check and its fix,
appends one ``axiom-encode/validation-issue/v2`` record to
``$AXIOM_INFRA_ISSUES_FILE`` (default ``$RUNNER_TEMP/axiom-infra-issues.jsonl``)
and to the step summary, and always exits 0. The refusal itself never depends
on the script: if it cannot run, the step still prints the message and fails.

The annotation text starts with ``[<check id>]``, the same bracket form the
validator issues use, so a reader of the job log gets the check id from the
message alone.

Every other step of the ``attempt_budget`` and ``encode`` jobs that runs before
the model or publishes its result has a check in ``STEP_CHECK_DEFINITIONS``.
Those steps do not call the script (most fail inside a helper, and the hardened
steps after the signing supervisor must not run the PATH ``python``). Their
check names the step: a reader maps the failed step's name to the check through
``site`` in the published registry. ``tests/test_checks.py`` fails when a step
of those jobs has neither a check nor an entry in its exempt list.

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
        "id": "infra:require-corpus-read-credentials",
        "title": "The corpus read credentials are not available to this repository",
        "requires": (
            "axiom-encode can read the organization's "
            "`R2_CORPUS_READ_ACCESS_KEY_ID` and "
            "`R2_CORPUS_READ_SECRET_ACCESS_KEY` secrets."
        ),
        "why": (
            "A lock-pinned corpus fetches scopes stored only in R2 with these "
            "credentials. Without them the run would fail after the "
            "production-signing approval and waste that review."
        ),
        "fix": (
            "Grant this repository access to the organization's "
            "`R2_CORPUS_READ_*` secrets, then dispatch the run again."
        ),
        "site": (
            f"{_WORKFLOW}:Require corpus read credentials before the signing gate"
        ),
    },
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
    {
        "id": "infra:fetch-pinned-corpus-provision-objects",
        "title": "The pinned corpus provisions cannot be fetched safely",
        "requires": (
            "The signed release object lists provision artifacts only under "
            "`data/corpus/provisions/`, or the pinned corpus checkout holds "
            "the provisions directory itself. A locked corpus also needs the "
            "R2 corpus read credentials."
        ),
        "why": (
            "Encoding reads source text only from verified provision objects; "
            "a path outside the provisions tree or a checkout with neither "
            "locks nor provisions gives the run no authenticated source, and "
            "without the credentials the run cannot fetch a scope stored only "
            "in R2."
        ),
        "fix": (
            "Check the signed release object's provision artifact paths and the "
            "axiom-corpus ref; publish or pin a release whose corpus checkout "
            "has `.axiom/corpus-locks` or `data/corpus/provisions`, then retry. "
            "If the message names the R2 corpus read credentials, grant this "
            "repository access to the organization's `R2_CORPUS_READ_*` secrets."
        ),
        "site": f"{_WORKFLOW}:Fetch pinned corpus provision objects",
    },
)


_SETUP_WHY = (
    "The run builds its tools and checkouts from pinned revisions before the "
    "model runs. Nothing was encoded, so the encoder output is not at fault."
)
_REQUEST_WHY = (
    "The step reads the dispatch inputs before the model runs and refuses a "
    "request that it cannot bind to one exact transaction. Nothing was "
    "encoded, so the encoder output is not at fault."
)
_CORPUS_WHY = (
    "Encoding reads source text only from authenticated corpus exports. "
    "Nothing was encoded yet, so the encoder output is not at fault."
)
_GUARD_WHY = (
    "The step runs after the encode step and verifies or packages what the run "
    "produced. The check names the step that stopped the run, not the root "
    "cause: the step's own message says whether the generated files or the "
    "run's inputs are at fault."
)
_PUBLISH_WHY = (
    "The step publishes verified changes. The encode itself finished; a "
    "failure here means no pull request or no signed artifact for this run."
)
_READ_LOG_FIX = (
    "Read the step's log for the exact refusal, correct that, and dispatch again."
)

#: ``(step name, title, requires, why, fix)`` for each step of the
#: ``attempt_budget`` and ``encode`` jobs that records no issue of its own.
#: The id is ``infra:`` plus the step name in lowercase with ``-`` separators.
_STEP_CHECKS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "Checkout axiom-encode",
        "The axiom-encode checkout failed",
        "The runner can check out axiom-encode at the dispatched commit.",
        _SETUP_WHY,
        "Retry the run; if it fails again, check GitHub's status.",
    ),
    (
        "Refuse an encoder head that no version bump covers",
        "The encoder head is not covered by a version bump",
        "Every encoder change on the dispatched head is at or before the "
        "latest automatic version bump.",
        "The encoder version is assigned after merge. A head with changes "
        "that no bump covers would be refused inside the protected job, "
        "after the signing approval.",
        "Wait for the automatic encoder version bump to merge, then dispatch "
        "again from the new main head.",
    ),
    (
        "Enforce failed-attempt budget",
        "The failed-attempt budget refused this dispatch",
        "An ad hoc dispatch of a citation has fewer consecutive failed model "
        "attempts than the configured budget.",
        "The budget stops repeated dispatches of a citation that keeps "
        "failing, before an approval or a model call is spent.",
        "Fix the cause of the earlier failures first. A maintainer can raise "
        "`ATTEMPT_BUDGET_BY_CITATION_JSON` or set `ATTEMPT_BUDGET_OVERRIDE` "
        "for a reviewed retry.",
    ),
    (
        "Checkout canonical RuleSpec country",
        "The RuleSpec country checkout failed",
        "`rulespec-<country>` exists and contains the commit in `rulespec_ref`.",
        _SETUP_WHY,
        "Check the `country` and `rulespec_ref` inputs and dispatch again.",
    ),
    (
        "Checkout axiom-corpus",
        "The axiom-corpus checkout failed",
        "axiom-corpus contains the commit in `corpus_ref`.",
        _SETUP_WHY,
        "Check the `corpus_ref` input and dispatch again.",
    ),
    (
        "Checkout axiom-rules-engine",
        "The axiom-rules-engine checkout failed",
        "axiom-rules-engine contains the commit in `rules_engine_ref`.",
        _SETUP_WHY,
        "Check the `rules_engine_ref` input and dispatch again.",
    ),
    (
        "Preflight live reviewed PR promotion",
        "The source bundle or the reviewed PR promotion request was refused",
        "`source_bundle_json` is valid JSON. A reviewed-PR promotion bundle is "
        "dispatched from main with `open_pr` true and no other transaction "
        "input, names an open, reviewed RuleSpec pull request that matches "
        "the dispatch, and the dispatch actor may promote it.",
        _REQUEST_WHY,
        "Read the step's message. Correct `source_bundle_json`, or check the "
        "promotion bundle against the live pull request (head, base, review "
        "state), and dispatch again.",
    ),
    (
        "Parse RuleSpec-pinned axiom-compose reference",
        "The RuleSpec workflow toolchain pin cannot be read",
        "`.axiom/workflow-toolchain.toml` exists at `rulespec_ref`, and its "
        "`axiom_compose_ref`, when present, is a full lowercase commit SHA.",
        _SETUP_WHY,
        "Dispatch with a `rulespec_ref` that contains "
        "`.axiom/workflow-toolchain.toml`, or correct the pin in the RuleSpec "
        "repository.",
    ),
    (
        "Checkout RuleSpec-pinned axiom-compose",
        "The pinned axiom-compose checkout failed",
        "axiom-compose contains the commit that the RuleSpec workflow "
        "toolchain pins, and the checkout is clean.",
        _SETUP_WHY,
        "Check `axiom_compose_ref` in the RuleSpec repository's "
        "`.axiom/workflow-toolchain.toml`.",
    ),
    (
        "Install uv",
        "The uv installation failed",
        "The runner can install the pinned uv release.",
        _SETUP_WHY,
        "Retry the run; if it fails again, check the pinned setup action.",
    ),
    (
        "Set up Go 1.26.1",
        "The Go toolchain installation failed",
        "The runner can install the pinned Go release.",
        _SETUP_WHY,
        "Retry the run; if it fails again, check the pinned setup action.",
    ),
    (
        "Set up Python",
        "The Python installation failed",
        "uv can install the pinned Python releases.",
        _SETUP_WHY,
        "Retry the run; if it fails again, check the Python download source.",
    ),
    (
        "Install locked preflight runtime",
        "The locked preflight runtime installation failed",
        "The encoder checkout matches `GITHUB_SHA`, and its Python 3.13 "
        "preflight environment installs from `uv.lock` with the API extra.",
        _SETUP_WHY,
        "Read the step's log. Correct a checkout or lock mismatch in "
        "axiom-encode; retry a package-index or Python download outage. "
        "Keep the explicit locked virtualenv interpreter for preflight.",
    ),
    (
        "Install Rust toolchain",
        "The Rust toolchain installation failed",
        "The runner can install the pinned Rust toolchain.",
        _SETUP_WHY,
        "Retry the run; if it fails again, check the pinned setup action.",
    ),
    (
        "Build Axiom rules engine",
        "The rules engine build failed",
        "axiom-rules-engine builds at the commit in `rules_engine_ref`.",
        _SETUP_WHY,
        "Dispatch with a `rules_engine_ref` that builds.",
    ),
    (
        "Install encoder",
        "The encoder installation failed",
        "The encoder and its protected verification packages install exactly "
        "from `uv.lock`.",
        _SETUP_WHY,
        "Read the step's log: a package index outage needs a retry; a lock "
        "mismatch needs a fix in axiom-encode.",
    ),
    (
        "Bind rules engine to declared revision",
        "The rules engine is not at the declared revision",
        "The built rules engine checkout is exactly the commit in `rules_engine_ref`.",
        _SETUP_WHY,
        "Check the `rules_engine_ref` input and dispatch again.",
    ),
    (
        "Build protected axiom-compose runtime",
        "The protected axiom-compose runtime build failed",
        "The pinned axiom-compose installs exactly from its lock file.",
        _SETUP_WHY,
        "Read the step's log: a package index outage needs a retry; a lock "
        "mismatch needs a fix in axiom-compose or a new pin.",
    ),
    (
        "Resolve registered reviewed repair request",
        "The registered reviewed repair request was refused",
        "A reviewed repair request in `source_bundle_json` names a patch in "
        "the code-reviewed registry, is dispatched with `open_pr` false as an "
        "ordinary full artifact repair, and carries no queue, dependent or "
        "legacy input.",
        _REQUEST_WHY,
        _READ_LOG_FIX,
    ),
    (
        "Validate optional authenticated corpus transport request",
        "The corpus transport request in the source bundle is invalid",
        "A corpus-artifact-transport or provision-prefetch wrapper in "
        "`source_bundle_json` has the reviewed shape and fits the other "
        "dispatch inputs.",
        _REQUEST_WHY,
        "Correct `source_bundle_json` as the step's message says and dispatch again.",
    ),
    (
        "Resolve retired source metadata migration request",
        "The retired source metadata migration request is invalid",
        "A metadata migration plan in `source_bundle_json` is bound to "
        "`rulespec_ref` and carries no input of another transaction mode.",
        _REQUEST_WHY,
        "Correct the migration plan or drop the conflicting input, then "
        "dispatch again.",
    ),
    (
        "Download corpus export metadata",
        "The corpus export run metadata cannot be downloaded",
        "The corpus export run named in the source bundle exists in "
        "axiom-corpus and is readable.",
        _CORPUS_WHY,
        "Check the export run id in `source_bundle_json`; retry on a GitHub "
        "API outage.",
    ),
    (
        "Authenticate corpus export metadata without repository token",
        "The corpus export run is not an authenticated export",
        "The export run is a successful run of the reviewed axiom-corpus "
        "export workflow for the pinned corpus commit, with one unexpired "
        "artifact.",
        _CORPUS_WHY,
        "Name a successful, unexpired export run of the pinned corpus commit, "
        "or run the export again.",
    ),
    (
        "Download authenticated corpus export archive",
        "The corpus export archive cannot be downloaded",
        "The authenticated export artifact is still available.",
        _CORPUS_WHY,
        "Retry; if the artifact expired, run the corpus export again and "
        "dispatch with the new run id.",
    ),
    (
        "Select authenticated signed provision export runs",
        "The provision export runs cannot be selected",
        "The source bundle names its signed provision export runs in the "
        "reviewed shape.",
        _CORPUS_WHY,
        "Correct the provision run ids in `source_bundle_json`.",
    ),
    (
        "Download authenticated provision export metadata",
        "The provision export run metadata cannot be downloaded",
        "Every provision export run named in the source bundle exists in "
        "axiom-corpus and is readable.",
        _CORPUS_WHY,
        "Check the provision run ids in `source_bundle_json`; retry on a "
        "GitHub API outage.",
    ),
    (
        "Validate authenticated provision export metadata",
        "A provision export run is not an authenticated export",
        "Every provision export run is a successful run of the reviewed "
        "export workflow for the pinned corpus commit, with one unexpired "
        "artifact.",
        _CORPUS_WHY,
        "Name successful, unexpired provision export runs of the pinned "
        "corpus commit, or run the export again.",
    ),
    (
        "Download authenticated provision export archives",
        "A provision export archive cannot be downloaded",
        "Every authenticated provision export artifact is still available.",
        _CORPUS_WHY,
        "Retry; if an artifact expired, run the provision export again and "
        "dispatch with the new run id.",
    ),
    (
        "Prefetch authenticated signed provision members",
        "The provision export archives do not match the signed release",
        "Every member of a provision export archive matches the signed corpus "
        "release that the RuleSpec toolchain pins.",
        _CORPUS_WHY,
        "Export the provisions from the pinned corpus release again, or "
        "dispatch with matching corpus and RuleSpec pins.",
    ),
    (
        "Provision protected signing supervisor",
        "The protected signing runtime cannot be provisioned",
        "The signing supervisor and the trusted Python runtime build and "
        "install from the dispatched encoder commit, and that commit is "
        "admissible for signing.",
        "Signing runs only inside a root-owned runtime built from the "
        "reviewed encoder commit. Nothing was encoded, so the encoder output "
        "is not at fault.",
        _READ_LOG_FIX,
    ),
    (
        "Verify protected RuleSpec routing",
        "The protected runtime rejected the RuleSpec checkout",
        "The protected runtime resolves the RuleSpec checkout to the "
        "canonical repository of the dispatched country.",
        _SETUP_WHY,
        "Check the `country` input and the RuleSpec checkout's origin.",
    ),
    (
        "Validate dependent cascade",
        "The dependent cascade inputs are invalid",
        "A dependent citation is a direct dependent of the target, a second "
        "dependent follows a first one, and the cascade mode is one of the "
        "reviewed modes.",
        _REQUEST_WHY,
        "Correct the dependent citation inputs as the step's message says.",
    ),
    (
        "Verify existing signed imports",
        "The existing signed imports, source bundle or repair mode are invalid",
        "Every path in `existing_signed_imports_json` is a signed RuleSpec "
        "file at `rulespec_ref`, `source_bundle_json` splits into one "
        "transaction, the repair mode resolves, and the inputs do not mix "
        "transaction modes.",
        _REQUEST_WHY,
        "Read the step's message. Correct the input that it names, or split "
        "the dispatch as it says.",
    ),
    (
        "Migrate retired source metadata",
        "The retired source metadata migration failed",
        "The migration plan applies to the RuleSpec checkout and the pinned "
        "corpus, and the apply signer accepts the transaction.",
        "This model-free lane rewrites source metadata under the apply "
        "signature. No model ran, so the encoder output is not at fault.",
        "Read the step's log for the refused plan entry, correct the plan, "
        "and dispatch again.",
    ),
    (
        "Verify generated provenance",
        "The generated files failed the provenance guard",
        "Every changed RuleSpec file carries a valid signed manifest from "
        "this encoder commit, bound to the pinned corpus and the base ref.",
        _GUARD_WHY,
        "Read the guard's JSON diagnostics in the step's log; they name each "
        "file and the reason. Without guard JSON, the last command "
        "(`git diff --check`) found a whitespace error.",
    ),
    (
        "Verify existing signed import integrity",
        "An existing signed import changed or lost its signature",
        "Every existing signed import still matches its signed manifest after "
        "the encode.",
        _GUARD_WHY,
        "Read the inventory in the step's log; restore or re-encode the "
        "import that it names.",
    ),
    (
        "Package retired source metadata migration changes",
        "The metadata migration changes do not match the dispatched plan",
        "The files that the migration changed are exactly the files of the "
        "dispatched plan.",
        _GUARD_WHY,
        _READ_LOG_FIX,
    ),
    (
        "Package exact generated changes",
        "The generated changes are outside the exact expected set",
        "The run changed exactly the files that the dispatch authorizes, each "
        "canonical, present where required and within its size limit.",
        _GUARD_WHY,
        "Read the step's message: it names the path and the rule that it broke.",
    ),
    (
        "Commit reviewed lane changes locally",
        "The verified changes cannot be committed to the lane branch",
        "The packaged change set applies cleanly as one commit on a new lane branch.",
        _PUBLISH_WHY,
        _READ_LOG_FIX,
    ),
    (
        "Recheck live reviewed PR authorization before publication",
        "The reviewed PR authorization changed during the run",
        "The reviewed pull request still has the head, base and review state "
        "that the run admitted at its start.",
        _PUBLISH_WHY,
        "Get the pull request reviewed again at its current head, then "
        "dispatch a new promotion.",
    ),
    (
        "Verify reviewed lane pull request base",
        "The pull request base moved away from the encoded ref",
        "A publication token is available, and `pr_base_branch` still points "
        "at `rulespec_ref`.",
        _PUBLISH_WHY,
        "Dispatch again with `rulespec_ref` at the current head of the base "
        "branch. The signed artifact of this run stays valid for audit only.",
    ),
    (
        "Prepare reviewed lane pull request branch",
        "The lane branch name cannot be derived",
        "The country and run identity give one canonical lane branch name.",
        _PUBLISH_WHY,
        _READ_LOG_FIX,
    ),
    (
        "Push lane branch and open draft pull request",
        "The lane branch or its draft pull request cannot be published",
        "The publication token can push the lane branch and open a draft "
        "pull request whose base is exactly the reviewed base SHA.",
        _PUBLISH_WHY,
        "Check the publication token's access to the RuleSpec repository and "
        "whether the base branch moved; then dispatch again.",
    ),
    (
        "Commit retired source metadata migration locally",
        "The metadata migration cannot be committed",
        "The packaged migration applies cleanly as one commit on a new branch.",
        _PUBLISH_WHY,
        _READ_LOG_FIX,
    ),
    (
        "Check retired source metadata publication token",
        "The publication token is not available",
        "The run has a token that can publish to the RuleSpec repository.",
        _PUBLISH_WHY,
        "Give the workflow its RuleSpec publication token, then dispatch again.",
    ),
    (
        "Verify retired source metadata migration before publication",
        "The committed metadata migration differs from the verified one",
        "The commit holds exactly the packaged migration inventory.",
        _PUBLISH_WHY,
        _READ_LOG_FIX,
    ),
    (
        "Verify retired source metadata pull request base",
        "The migration pull request base moved away from the encoded ref",
        "`pr_base_branch` still points at `rulespec_ref`.",
        _PUBLISH_WHY,
        "Dispatch again with `rulespec_ref` at the current head of the base branch.",
    ),
    (
        "Prepare retired source metadata pull request branch",
        "The migration branch name cannot be derived",
        "The country and run identity give one canonical branch name.",
        _PUBLISH_WHY,
        _READ_LOG_FIX,
    ),
    (
        "Push retired source metadata migration branch and open draft pull request",
        "The migration branch or its draft pull request cannot be published",
        "The publication token can push the branch and open a draft pull "
        "request whose base is exactly the reviewed base SHA.",
        _PUBLISH_WHY,
        "Check the publication token's access to the RuleSpec repository and "
        "whether the base branch moved; then dispatch again.",
    ),
    (
        "Finalize signed re-encode artifact checksums",
        "The signed artifact cannot be checksummed",
        "The run artifact holds regular files only and at least one file.",
        _PUBLISH_WHY,
        _READ_LOG_FIX,
    ),
    (
        "Upload signed re-encode artifact",
        "The signed artifact upload failed",
        "The runner can upload the checksummed run artifact.",
        _PUBLISH_WHY,
        "Retry the run; if it fails again, check GitHub's status.",
    ),
)


def step_check_id(step: str) -> str:
    """``infra:`` plus the step name in lowercase with ``-`` separators."""
    return "infra:" + re.sub(r"[^a-z0-9]+", "-", step.lower()).strip("-")


#: Checks that name a step. No step calls ``record`` with these ids.
STEP_CHECK_DEFINITIONS: tuple[dict[str, str], ...] = tuple(
    {
        "id": step_check_id(step),
        "title": title,
        "requires": requires,
        "why": why,
        "fix": fix,
        "site": f"{_WORKFLOW}:{step}",
    }
    for step, title, requires, why, fix in _STEP_CHECKS
)


def register_checks(declare: Callable[..., Any]) -> None:
    """Register every infra check with ``checks.declare_check``."""
    for definition in (*INFRA_CHECK_DEFINITIONS, *STEP_CHECK_DEFINITIONS):
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
    for definition in (*INFRA_CHECK_DEFINITIONS, *STEP_CHECK_DEFINITIONS):
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
            f"::error title={_annotation_property(check_id)}::"
            f"[{_annotation_text(check_id)}] {_annotation_text(message)}"
        )
        return 0
    record = infra_issue_record(check_id, message)
    print(
        f"::error title={_annotation_property(check_id)}::"
        f"[{_annotation_text(check_id)}] "
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
