"""Promotion evidence for an ``applied-rulespec/v6`` manifest. INACTIVE.

The workflow half of change 5b of the merge guard design (#2172, section 8):
the evidence step of ``.github/workflows/targeted-signed-reencode.yml``
("Package exact generated changes") reads the v6 promotion manifest that the
switched-off writer of change 5 emits. Until this change that step accepted
only v5 and read ``reviewed_rulespec_ref``, so it stopped a v6 promotion after
signing and before publication.

Nothing here is active. The step calls this function only for a manifest
labelled v6, and the function refuses unless every lock that the v6 writer
uses is open. It checks them with the writer's own functions, first, before
it reads anything else:

1. ``applied_manifest_v6.require_candidate_writer``: lock 1
   (``reviewed_pr_admission.V6_WRITER_ENABLED``) and the reader switch
   (``applied_manifest_v6.V6_READER_ENABLED``).
2. ``reviewed_pr_admission.require_guard_enforced``: lock 2 (``GUARD_APP_ID``)
   and lock 3, the live read that ``main`` of the RuleSpec repository requires
   the guard's check from that App.

There is no switch of its own, and no environment variable, CLI flag or
dispatch input. A manifest without the v6 label never reaches this module, so
the v5 path of the step is the same as before.

After the locks, the manifest must be exactly what the reader admits, checked
with the reader's definitions: its entry test (``v6_admissible``), its exact
field sets (``cli._REVIEWED_CANDIDATE_V6_MANIFEST_VARIANTS``) and its
attestation scope check. The scope must then name this run: the target
citation, the requested RuleSpec ref (``candidate_rulespec_ref``), the RuleSpec
repository of the run and the pull request of the dispatch. Every failure
raises ``ValueError``; nothing is read as "absent".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

from axiom_encode import applied_manifest_v6 as _manifest_v6
from axiom_encode import reviewed_pr_admission as _admission
from axiom_encode.cli import _REVIEWED_CANDIDATE_V6_MANIFEST_VARIANTS
from axiom_encode.prepare_signed_backfill import split_atomic_source_input

MANIFEST_LABEL: Final = "candidate promotion manifest"
# The reader's field sets, the same object: v5 promotion contract with the
# reviewed ref renamed and the attestation scope added.
V6_FIELD_VARIANTS: Final = _REVIEWED_CANDIDATE_V6_MANIFEST_VARIANTS
# The evidence record of the step for a v6 manifest. Another schema than the
# reviewed-candidate record, so that the run artifact never calls an
# unreviewed candidate reviewed.
EVIDENCE_SCHEMA: Final = "axiom-encode/candidate-promotion-evidence/v1"


def require_candidate_promotion_evidence(
    payload: object,
    *,
    atomic_source_json: object,
    country: object,
    citation: object,
    rulespec_ref: object,
    fetch: Callable[[str], object] | None = None,
) -> str:
    """Admit one v6 manifest as promotion evidence of this run, or raise.

    Returns ``candidate_rulespec_ref``. ``fetch`` replaces the GitHub reader of
    lock 3 in tests; the workflow does not pass it.
    """

    if not _manifest_v6.is_v6_labelled(payload):
        raise ValueError("the manifest is not labelled applied-rulespec/v6")
    # The writer's locks, with its own functions, before anything else.
    _manifest_v6.require_candidate_writer()
    repository = f"TheAxiomFoundation/rulespec-{country}"
    if not isinstance(country, str) or not _manifest_v6.is_promotion_repository(
        repository
    ):
        raise ValueError("the RuleSpec repository of this run is invalid")
    _admission.require_guard_enforced(
        repository, fetch=_admission.github_json if fetch is None else fetch
    )

    # The lane: a live candidate pull request promotion, and nothing else.
    try:
        transaction = split_atomic_source_input(atomic_source_json)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError(f"the dispatch transaction cannot be read: {exc}") from exc
    rulespec_pr = transaction.get("rulespec_pr")
    if (
        transaction.get("reviewed_candidate_promotion") is not True
        or type(rulespec_pr) is not int
        or rulespec_pr <= 0
    ):
        raise ValueError(
            "a v6 manifest is evidence only for a live candidate pull request promotion"
        )

    # The reader's contract: entry test, exact field set, exact scope.
    if not _manifest_v6.v6_admissible(payload):
        raise ValueError(
            "only the promotion tool, with no backend, may carry the v6 label"
        )
    issues = _manifest_v6.contract_issues(
        payload, manifest_label=MANIFEST_LABEL, field_variants=V6_FIELD_VARIANTS
    )
    if issues:
        raise ValueError("; ".join(issues))

    # The scope names this run.
    candidate = payload[_manifest_v6.ATTESTATION_SCOPE_FIELD]["candidate"]
    ref = payload[_manifest_v6.CANDIDATE_REF_FIELD]
    if payload.get("citation") != citation:
        raise ValueError(f"{MANIFEST_LABEL} is not for the requested citation")
    if ref != rulespec_ref:
        raise ValueError(
            f"{MANIFEST_LABEL} {_manifest_v6.CANDIDATE_REF_FIELD} is not the "
            "requested RuleSpec ref"
        )
    if candidate["repository"] != repository:
        raise ValueError(
            f"{MANIFEST_LABEL} scope names another repository than this run"
        )
    if candidate["pull_request"] != rulespec_pr:
        raise ValueError(
            f"{MANIFEST_LABEL} scope names another pull request than this run"
        )
    return ref
