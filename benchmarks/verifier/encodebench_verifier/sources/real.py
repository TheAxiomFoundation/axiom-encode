"""Loader for recorded real defects (``benchmarks/verifier/real_defects_v0/``).

The corpus is produced by another session (axiom-encode PR #1659); its README
is the schema of record and this loader follows it. One ``case.json`` per
case under ``cases/<id>/`` with, when ``artifacts_shipped`` is true, the
sibling files ``pre_fix.yaml`` (module bytes before the correcting commit),
``post_fix.yaml`` (after) and ``provision.txt`` (the provision resolved from
the signed corpus release). ``pre_fix_artifact_sha256``,
``post_fix_artifact_sha256`` and ``provision_sha256`` are verified against the
file bytes; a mismatch refuses the case. Metadata-only cases
(``artifacts_shipped`` false, the inherited-verdict members of the generator
families) are skipped and counted, never invented.

Mapping onto the pair model: the pre-fix artifact is the ``defective`` case,
the post-fix artifact is the ``control``. The README says it plainly: post-fix
is not guaranteed clean (one module was corrected three times), so every real
control is recorded ``control_clean = "unverified"`` and the board neither
applies the false-alarm ceiling to them nor calls them clean.

Defect kinds: the corpus taxonomy has eight kinds. Five have a synthetic
analogue with a kind-specific judge channel and are mapped onto it
(``amount_mismatch``, ``boundary_direction``, ``polarity_or_logic``,
``wrong_period_or_effective_date``, ``wrong_entity_or_scope``). The other
three (``unrepresented_clause``, ``untraceable_branch``, ``other``) are kept
as ``other:<kind>`` columns scored on every judge's verdict channel; forcing
an unrepresented clause into "conjunct dropped" would overstate the match.

Selection defaults follow the README's advice for independent cases: family
representatives only, ``triage_status`` ``fidelity`` only (the ``unclear``
cases are kept in the corpus for a reviewer to prune), any confidence.

Tolerances for records written by hand (the fixture, small experiments):
``case_id`` for ``id``; ``provision`` / ``pre_fix`` / ``post_fix`` file
references or inline ``provision_text`` / ``pre_fix_yaml`` / ``post_fix_yaml``;
hashes under ``hashes: {pre_fix, post_fix, provision}``; a plain-string
``locator``; a bare ``defect_kind`` label. This directory is only ever read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from .. import DEFECT_KINDS, VARIANT_CONTROL, VARIANT_DEFECTIVE
from ..cases import CaseSuite, Locator, SuiteError, VerifierCase

_KIND_ALIASES = {
    # corpus taxonomy (README of real_defects_v0) -> synthetic analogue
    "amount_mismatch": "amount_changed",
    "boundary_direction": "boundary_flipped",
    "polarity_or_logic": "polarity_swapped",
    "wrong_period_or_effective_date": "date_or_period_wrong",
    "wrong_entity_or_scope": "entity_wrong",
    # synthetic names and hand-written shorthands
    "amount": "amount_changed",
    "amount_changed": "amount_changed",
    "wrong_amount": "amount_changed",
    "boundary": "boundary_flipped",
    "boundary_flipped": "boundary_flipped",
    "conjunct": "conjunct_dropped",
    "dropped_conjunct": "conjunct_dropped",
    "conjunct_dropped": "conjunct_dropped",
    "polarity": "polarity_swapped",
    "and_or": "polarity_swapped",
    "polarity_swapped": "polarity_swapped",
    "date": "date_or_period_wrong",
    "period": "date_or_period_wrong",
    "effective_date": "date_or_period_wrong",
    "wrong_effective_date": "date_or_period_wrong",
    "wrong_period": "date_or_period_wrong",
    "date_or_period_wrong": "date_or_period_wrong",
    "date_or_period": "date_or_period_wrong",
    "entity": "entity_wrong",
    "wrong_entity": "entity_wrong",
    "entity_wrong": "entity_wrong",
}

# Corpus kinds kept as their own ``other:`` columns (no synthetic analogue).
CORPUS_OTHER_KINDS = ("unrepresented_clause", "untraceable_branch", "other")


class RealDefectsError(ValueError):
    """The real-defects directory or one of its cases is unusable."""


class MetadataOnlyCase(RealDefectsError):
    """The record ships no artifacts (``artifacts_shipped`` is false)."""


def normalise_kind(raw: Any, other_kind: Any = None) -> str:  # noqa: ARG001
    key = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in _KIND_ALIASES:
        return _KIND_ALIASES[key]
    # The corpus's ``other_kind`` is free text (a sentence, at times); it is
    # kept verbatim in the case origin, and the board column stays ``other``.
    return f"other:{key or 'unknown'}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_member(
    payload: dict[str, Any], base_dir: Path, name: str, inline_key: str
) -> str:
    inline = payload.get(inline_key)
    if isinstance(inline, str) and inline:
        return inline
    for key in (name, f"{name}.yaml", f"{name}.txt", f"{name}_path", f"{name}_file"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            if not path.is_absolute():
                path = base_dir / path
            if not path.is_file():
                raise RealDefectsError(f"{name} file not found: {path}")
            return path.read_text()
    default = base_dir / (f"{name}.txt" if name == "provision" else f"{name}.yaml")
    if default.is_file():
        return default.read_text()
    raise RealDefectsError(f"case has no {name} content or file reference")


def _expected_hash(payload: dict[str, Any], name: str) -> Optional[str]:
    hashes = payload.get("hashes")
    if isinstance(hashes, dict):
        for key in (name, f"{name}.yaml", f"{name}.txt", f"{name}_sha256"):
            if isinstance(hashes.get(key), str):
                return hashes[key].lower()
    # README keys: pre_fix_artifact_sha256, post_fix_artifact_sha256, provision_sha256
    for key in (f"{name}_artifact_sha256", f"{name}_sha256"):
        direct = payload.get(key)
        if isinstance(direct, str):
            return direct.lower()
    return None


def _locator(payload: dict[str, Any]) -> Locator:
    raw = payload.get("locator")
    if isinstance(raw, dict):
        rule_names = raw.get("rule_names")
        first_rule = (
            rule_names[0]
            if isinstance(rule_names, list) and rule_names
            else raw.get("rule_name") or raw.get("rule")
        )
        lines = raw.get("pre_fix_lines")
        detail = str(raw.get("detail") or payload.get("description") or "")
        if isinstance(lines, list) and lines:
            detail = (detail + " " if detail else "") + f"pre-fix lines {lines}"
        return Locator(
            path=str(raw.get("path") or raw.get("rule_path") or ""),
            rule_index=raw.get("rule_index"),
            rule_name=str(first_rule) if first_rule else None,
            detail=detail.strip(),
            before=raw.get("before"),
            after=raw.get("after"),
            token=raw.get("token") or (str(first_rule) if first_rule else None),
        )
    if isinstance(raw, str) and raw:
        # "some_rule.versions[0].formula" names the rule in its first segment;
        # "rules[3].versions[0].formula" names it by index only.
        first = raw.split(".")[0]
        rule_name = first if first and "[" not in first else None
        return Locator(path=raw, rule_name=rule_name)
    return Locator(path="", detail="locator missing in case record")


def read_case_record(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RealDefectsError(f"could not read case {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RealDefectsError(f"case {path} is not a JSON object")
    return payload


def load_case_file(path: Path) -> tuple[VerifierCase, VerifierCase, dict[str, Any]]:
    """Load one case record into its (control, defective) pair."""

    path = Path(path)
    payload = read_case_record(path)
    if payload.get("artifacts_shipped") is False:
        raise MetadataOnlyCase(
            f"case {payload.get('id') or path.parent.name} ships no artifacts"
        )
    base_dir = path.parent
    case_id = str(
        payload.get("case_id")
        or payload.get("id")
        or (path.parent.name if path.name == "case.json" else path.stem)
    )
    provision = _read_member(payload, base_dir, "provision", "provision_text")
    pre_fix = _read_member(payload, base_dir, "pre_fix", "pre_fix_yaml")
    post_fix = _read_member(payload, base_dir, "post_fix", "post_fix_yaml")
    for name, text in (
        ("provision", provision),
        ("pre_fix", pre_fix),
        ("post_fix", post_fix),
    ):
        expected = _expected_hash(payload, name)
        if expected and expected != _sha256_text(text):
            raise RealDefectsError(
                f"case {case_id}: {name} content does not match its recorded sha256"
            )
    if pre_fix == post_fix:
        raise RealDefectsError(f"case {case_id}: pre_fix and post_fix are identical")
    kind = normalise_kind(payload.get("defect_kind"), payload.get("other_kind"))
    citation = str(payload.get("citation") or payload.get("corpus_citation_path") or "")
    origin = {
        "source": "real_defects",
        "case_file": path.name,
        "raw_defect_kind": payload.get("defect_kind"),
        "other_kind": payload.get("other_kind"),
        "generator_model": payload.get("generator_model"),
        "fix_reference": payload.get("fix_reference")
        or payload.get("pr_url")
        or payload.get("fix"),
        "repo": payload.get("repo"),
        "commit": payload.get("commit"),
        "parent_commit": payload.get("parent_commit"),
        "module_path": payload.get("module_path"),
        "corpus_release": payload.get("corpus_release"),
        "fix_stage": payload.get("fix_stage"),
        "triage_status": payload.get("triage_status"),
        "confidence": payload.get("confidence"),
        "family_id": payload.get("family_id"),
        "family_representative": payload.get("family_representative"),
        "description": payload.get("description"),
        "notes": payload.get("notes"),
    }
    locator = _locator(payload)
    common = dict(
        pair_id=case_id,
        defect_kind=kind,
        citation=citation,
        origin=origin,
        control_clean="unverified",
    )
    control = VerifierCase(
        variant=VARIANT_CONTROL,
        provision_text=provision,
        artifact_text=post_fix,
        locator=None,
        **common,
    )
    defective = VerifierCase(
        variant=VARIANT_DEFECTIVE,
        provision_text=provision,
        artifact_text=pre_fix,
        locator=locator,
        **common,
    )
    return control, defective, payload


def iter_case_files(root: Path) -> list[Path]:
    """Case records under ``root``: ``cases/*/case.json`` when that layout
    exists (the corpus), else every JSON file that is not an index, suite,
    triage or tool artefact (hand-written experiments)."""

    root = Path(root)
    if not root.is_dir():
        raise RealDefectsError(f"real-defects directory not found: {root}")
    cases_dir = root / "cases"
    if cases_dir.is_dir():
        return sorted(cases_dir.glob("*/case.json"))
    skip = {"suite.json", "suite.manifest.json", "index.json"}
    return sorted(
        p
        for p in root.rglob("*.json")
        if p.name not in skip
        and not p.name.startswith(".")
        and "triage" not in p.parts
        and "tools" not in p.parts
    )


def build_real_suite(
    root: Path,
    *,
    provision_chars: int,
    truncate,
    name: Optional[str] = None,
    representatives_only: bool = True,
    triage_statuses: tuple[str, ...] = ("fidelity",),
    min_confidence: float = 0.0,
    jurisdictions: tuple[str, ...] = (),
) -> tuple[CaseSuite, dict[str, Any]]:
    """Fold the selected case records under ``root`` into a suite.

    ``truncate`` is the provision-window function the judges use (the
    referee's head+tail truncation), passed in so this module has no import
    dependency on the judges package. Returns the suite and a selection
    report (kept, skipped and why).
    """

    root = Path(root)
    files = iter_case_files(root)
    if not files:
        raise RealDefectsError(f"no case records under {root}")
    cases: list[VerifierCase] = []
    other_kinds: dict[str, int] = {}
    skipped: dict[str, int] = {}
    kept_ids: list[str] = []

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for path in files:
        record = read_case_record(path)
        if representatives_only and record.get("family_representative") is False:
            skip("family_member_not_representative")
            continue
        status = record.get("triage_status")
        if triage_statuses and status is not None and status not in triage_statuses:
            skip(f"triage_status_{status}")
            continue
        confidence = record.get("confidence")
        if isinstance(confidence, (int, float)) and confidence < min_confidence:
            skip("below_min_confidence")
            continue
        if jurisdictions and record.get("jurisdiction") not in jurisdictions:
            skip("jurisdiction_excluded")
            continue
        try:
            control, defective, _ = load_case_file(path)
        except MetadataOnlyCase:
            skip("metadata_only")
            continue
        kept_ids.append(defective.pair_id)
        if defective.defect_kind not in DEFECT_KINDS:
            other_kinds[defective.defect_kind] = (
                other_kinds.get(defective.defect_kind, 0) + 1
            )
        window = truncate(control.provision_text, provision_chars)
        full_sha = _sha256_text(control.provision_text)
        for case in (control, defective):
            cases.append(
                VerifierCase(
                    pair_id=case.pair_id,
                    variant=case.variant,
                    defect_kind=case.defect_kind,
                    citation=case.citation,
                    provision_text=window,
                    artifact_text=case.artifact_text,
                    locator=case.locator,
                    origin=case.origin,
                    control_clean=case.control_clean,
                    provision_full_sha256=full_sha,
                )
            )
    readme = root / "README.md"
    index = root / "index.json"
    releases = sorted(
        {
            str(c.origin.get("corpus_release"))
            for c in cases
            if c.origin.get("corpus_release")
        }
    )
    identity = {
        "directory": root.name,
        "readme_sha256": _sha256_text(readme.read_text()) if readme.is_file() else None,
        "index_sha256": _sha256_text(index.read_text()) if index.is_file() else None,
        "selection": {
            "representatives_only": representatives_only,
            "triage_statuses": list(triage_statuses),
            "min_confidence": min_confidence,
            "jurisdictions": list(jurisdictions),
        },
        "case_ids": kept_ids,
        "corpus_releases": releases,
    }
    notes = [
        "Controls are post-fix artifacts from real repair rounds; they are not "
        "proven clean (control_clean = unverified).",
    ]
    if other_kinds:
        notes.append(
            "Defect kinds outside the synthetic taxonomy were kept as "
            f"other:<kind>: {other_kinds}"
        )
    report = {
        "records_seen": len(files),
        "pairs_kept": len(kept_ids),
        "skipped": skipped,
        "other_kinds": other_kinds,
        "corpus_releases": releases,
    }
    try:
        suite = CaseSuite(
            name=name or f"EncodeBench verifier real {root.name}",
            source_kind="real_defects",
            source_identity=identity,
            cases=cases,
            provision_chars=provision_chars,
            corpus_release=releases[0] if len(releases) == 1 else None,
            mutator=None,
            notes=notes,
        )
    except SuiteError as exc:
        raise RealDefectsError(str(exc)) from exc
    return suite, report
