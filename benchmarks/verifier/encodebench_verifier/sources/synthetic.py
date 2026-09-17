"""Build a synthetic suite: plant one defect per known-good artifact.

Each source artifact is used at most once (one citation, one pair) so a
judge's familiarity with a provision cannot leak across kinds. Kinds are
filled by deficit: the kind furthest below its quota is tried first on each
artifact, so the rarer mutation sites (dates, entities) get first pick.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Optional

from .. import DEFECT_KINDS, VARIANT_CONTROL, VARIANT_DEFECTIVE
from ..canonical import text_sha256
from ..cases import CaseSuite, VerifierCase
from ..mutator import MUTATOR_VERSION, MutationError, mutate
from . import KnownGoodArtifact


def build_synthetic_suite(
    artifacts: list[KnownGoodArtifact],
    *,
    name: str,
    source_kind: str,
    source_identity: dict[str, Any],
    provision_chars: int,
    truncate: Callable[[str, int], str],
    per_kind: int = 30,
    seed: int = 7,
    kinds: tuple[str, ...] = DEFECT_KINDS,
    corpus_release: Optional[str] = None,
) -> tuple[CaseSuite, dict[str, Any]]:
    """Return the suite and a build report (counts, skips)."""

    if per_kind <= 0:
        raise ValueError("per_kind must be positive")
    unknown = [kind for kind in kinds if kind not in DEFECT_KINDS]
    if unknown:
        raise ValueError(f"unknown defect kinds: {unknown}")
    rng = random.Random(seed)
    order = list(artifacts)
    rng.shuffle(order)

    counts = {kind: 0 for kind in kinds}
    cases: list[VerifierCase] = []
    used_citations: set[str] = set()
    skipped_parse = 0
    skipped_duplicate = 0
    no_site = 0

    for artifact in order:
        if all(counts[kind] >= per_kind for kind in kinds):
            break
        if artifact.citation in used_citations:
            skipped_duplicate += 1
            continue
        window = truncate(artifact.provision_text, provision_chars)
        # Deficit-first, ties broken by taxonomy order (deterministic).
        by_need = sorted(
            (kind for kind in kinds if counts[kind] < per_kind),
            key=lambda kind: (counts[kind], kinds.index(kind)),
        )
        planted = False
        for kind in by_need:
            try:
                mutation = mutate(
                    artifact.artifact_text,
                    window,
                    kind,
                    rng=random.Random(rng.random()),
                )
            except MutationError:
                skipped_parse += 1
                planted = None
                break
            if mutation is None:
                continue
            pair_id = f"{kind}-{artifact.key}"
            origin = dict(artifact.origin)
            origin["artifact_original_sha256"] = text_sha256(artifact.artifact_text)
            common = dict(
                pair_id=pair_id,
                defect_kind=kind,
                citation=artifact.citation,
                provision_text=window,
                origin=origin,
                control_clean="known_good_gate",
                provision_full_sha256=text_sha256(artifact.provision_text),
            )
            cases.append(
                VerifierCase(
                    variant=VARIANT_CONTROL,
                    artifact_text=mutation.control_text,
                    locator=None,
                    **common,
                )
            )
            cases.append(
                VerifierCase(
                    variant=VARIANT_DEFECTIVE,
                    artifact_text=mutation.defective_text,
                    locator=mutation.locator,
                    **common,
                )
            )
            counts[kind] += 1
            used_citations.add(artifact.citation)
            planted = True
            break
        if planted is False:
            no_site += 1

    suite = CaseSuite(
        name=name,
        source_kind=source_kind,
        source_identity=source_identity,
        cases=cases,
        provision_chars=provision_chars,
        corpus_release=corpus_release,
        mutator={
            "version": MUTATOR_VERSION,
            "seed": seed,
            "per_kind": per_kind,
            "kinds": list(kinds),
        },
        notes=[
            "Synthetic defects are single edits and a lower bound on difficulty.",
            "Both members of a pair are re-serialised through the same canonical "
            "YAML dumper; the only difference is the planted edit.",
        ],
    )
    report = {
        "artifacts_considered": len(order),
        "pairs": counts,
        "short_of_quota": {k: per_kind - v for k, v in counts.items() if v < per_kind},
        "skipped_unparseable": skipped_parse,
        "skipped_duplicate_citation": skipped_duplicate,
        "no_mutation_site": no_site,
    }
    return suite, report
