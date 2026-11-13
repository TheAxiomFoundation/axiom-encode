"""Reader contract for ``applied-rulespec/v6`` promotion manifests. INACTIVE.

This is change 2 of ``docs/content-bound-merge-guard.md`` section 8: the
verifier side of maintainer decision 1 ("the signing service may sign
validated, unreviewed candidate bytes when a versioned signed manifest
explicitly records that scope; verifiers must distinguish validation from
owner approval"). Tracking issue: #2172.

Nothing here is active:

- ``V6_READER_ENABLED`` is ``False``. While it is ``False`` every verifier
  treats a manifest labelled v6 exactly as before this module existed: it is
  not an encoder apply manifest.
- No writer emits v6, so no signed v6 manifest exists. The switch is a second,
  independent lock, and only a reviewed change to this file on encoder main
  can turn it.

Boundary (do not widen without a maintainer decision): this contract concerns
one lane only, a PR that adds generated RuleSpec through
``axiom-encode promote-reviewed-candidate`` with a signed manifest. A manifest
that is not labelled v6 takes none of these code paths, with the switch on or
off. Infrastructure changes in RuleSpec repositories (workflows, ``.axiom``
pins and configuration, ledgers, tooling, ProgramSpec) carry no manifest and
are neither read nor judged here; their lane is open question 0 of the design.

What a v6 manifest states, and what it does not: the signature means "encoder
main validated these exact bytes". ``attestation_scope.owner_review`` is
``"none"``. The owner approval is a separate, later control: the content-bound
merge guard named by ``merge_requirement``. A verifier that admits a v6
manifest has verified validation, never review.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final

# Off until the merge guard is deployed and enforced (decision D2d). Turning
# this on is a policy change on encoder main, not a runtime option: there is
# deliberately no environment variable, CLI flag or repository setting.
V6_READER_ENABLED: Final[bool] = False

MANIFEST_SCHEMA_V6: Final = "axiom-encode/applied-rulespec/v6"
PROMOTION_TOOL: Final = "axiom-encode promote-reviewed-candidate"
CANDIDATE_REF_FIELD: Final = "candidate_rulespec_ref"
ATTESTATION_SCOPE_FIELD: Final = "attestation_scope"
ATTESTATION_SCOPE_SCHEMA: Final = "axiom-encode/attestation-scope/v1"
ATTESTATION_VALIDATION: Final = "encoder-main-complete-source"
ATTESTATION_OWNER_REVIEW: Final = "none"
ATTESTATION_ORIGIN: Final = "local-candidate"
MERGE_REQUIREMENT: Final = "axiom-encode/content-bound-merge-guard/v1"
ATTESTATION_SCOPE_FIELDS: Final = frozenset(
    {
        "schema",
        "validation",
        "owner_review",
        "origin",
        "candidate",
        "dispatcher",
        "merge_requirement",
    }
)
ATTESTATION_CANDIDATE_FIELDS: Final = frozenset(
    {"repository", "pull_request", "head_sha", "base_sha", "author_id"}
)
ATTESTATION_DISPATCHER_FIELDS: Final = frozenset({"id", "login"})

_SHA = re.compile(r"[0-9a-f]{40}")
_LOGIN = re.compile(r"[A-Za-z0-9-]{1,39}")
_REPOSITORY = re.compile(r"TheAxiomFoundation/rulespec-[a-z]{2}")


def is_v6_labelled(payload: object) -> bool:
    """Whether a decoded manifest carries the v6 label. Grants nothing."""

    return isinstance(payload, Mapping) and (
        payload.get("schema_version") == MANIFEST_SCHEMA_V6
    )


def v6_admissible(payload: object) -> bool:
    """Whether a v6-labelled manifest may enter verification at all.

    ``False`` for everything while the reader is off. When on, only the
    promotion contract may carry the label: the promotion tool and no
    ``backend`` key. Signature, exact fields and the attestation scope are
    checked afterwards by the ordinary verifier; this is the entry test only.
    """

    return (
        V6_READER_ENABLED is True
        and is_v6_labelled(payload)
        and payload.get("tool") == PROMOTION_TOOL
        and "backend" not in payload
    )


def v6_field_variants(
    v5_candidate_variants: tuple[frozenset[str], ...],
    *,
    v5_ref_field: str,
) -> tuple[frozenset[str], ...]:
    """Derive the exact v6 field sets from the live v5 promotion contract."""

    return tuple(
        (fields - {v5_ref_field}) | {CANDIDATE_REF_FIELD, ATTESTATION_SCOPE_FIELD}
        for fields in v5_candidate_variants
    )


def _strict_positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def attestation_scope_issues(
    payload: Mapping[str, object], *, manifest_label: str
) -> list[str]:
    """Require the exact attestation scope of the design, section 5.

    A verifier works offline: it cannot confirm that the named pull request,
    author or dispatcher exist. It requires the exact shape and the internal
    binding (``candidate.head_sha`` is ``candidate_rulespec_ref``); the merge
    guard compares the values with live GitHub state.
    """

    def refuse(detail: str) -> list[str]:
        return [f"{manifest_label} attestation scope is invalid: {detail}"]

    scope = payload.get(ATTESTATION_SCOPE_FIELD)
    if not isinstance(scope, dict) or set(scope) != ATTESTATION_SCOPE_FIELDS:
        return refuse("fields")
    for name, expected in (
        ("schema", ATTESTATION_SCOPE_SCHEMA),
        ("validation", ATTESTATION_VALIDATION),
        ("owner_review", ATTESTATION_OWNER_REVIEW),
        ("origin", ATTESTATION_ORIGIN),
        ("merge_requirement", MERGE_REQUIREMENT),
    ):
        if scope[name] != expected or not isinstance(scope[name], str):
            return refuse(name)
    candidate, dispatcher = scope["candidate"], scope["dispatcher"]
    if (
        not isinstance(candidate, dict)
        or set(candidate) != ATTESTATION_CANDIDATE_FIELDS
        or not isinstance(candidate["repository"], str)
        or _REPOSITORY.fullmatch(candidate["repository"]) is None
        or not _strict_positive_int(candidate["pull_request"])
        or not _strict_positive_int(candidate["author_id"])
        or any(
            not isinstance(candidate[name], str)
            or _SHA.fullmatch(candidate[name]) is None
            for name in ("head_sha", "base_sha")
        )
    ):
        return refuse("candidate")
    if (
        not isinstance(dispatcher, dict)
        or set(dispatcher) != ATTESTATION_DISPATCHER_FIELDS
        or not _strict_positive_int(dispatcher["id"])
        or not isinstance(dispatcher["login"], str)
        or _LOGIN.fullmatch(dispatcher["login"]) is None
    ):
        return refuse("dispatcher")
    if candidate["head_sha"] != payload.get(CANDIDATE_REF_FIELD):
        return refuse("candidate head is not candidate_rulespec_ref")
    return []


def contract_issues(
    payload: Mapping[str, object],
    *,
    manifest_label: str,
    field_variants: tuple[frozenset[str], ...],
) -> list[str]:
    """The v6-only part of the exact contract: field set, then scope."""

    field_issues = (
        []
        if frozenset(payload) in field_variants
        else [
            f"{manifest_label} does not match the exact unreviewed candidate "
            "promotion v6 schema"
        ]
    )
    return field_issues + attestation_scope_issues(
        payload, manifest_label=manifest_label
    )


def owner_review_state(payload: Mapping[str, object]) -> str:
    """Name the owner-review claim of a promotion manifest.

    ``"none"`` for v6: validated, not reviewed; the merge guard supplies the
    approval. ``"before-signing"`` for the earlier promotion lane, whose
    admission checked the approval before the signing run.
    """

    return "none" if is_v6_labelled(payload) else "before-signing"
