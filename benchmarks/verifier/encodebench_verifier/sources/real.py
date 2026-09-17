"""Loader for recorded real defects (``benchmarks/verifier/real_defects_v0/``).

That directory is produced by a different session. Its README is the schema
of record; this loader was written against the brief for it — one JSON per
case naming ``pre_fix.yaml``, ``post_fix.yaml`` and ``provision.txt``, a
``defect_kind``, a ``locator`` and content hashes — plus the fixture under
``benchmarks/verifier/fixtures/real_defects_example/``. It reads that
directory; it never writes to it.

Mapping onto the pair model: the pre-fix artifact is the ``defective`` case,
the post-fix artifact is the ``control``. A post-fix artifact is what a
repair round produced, not a proven-clean artifact, so every real control is
recorded ``control_clean = "unverified"`` and the board says so.

Tolerances (kept deliberately small):

* file references may be given as ``pre_fix`` / ``pre_fix.yaml`` /
  ``pre_fix_path`` (same for ``post_fix`` and ``provision``), relative to the
  case JSON, or inline as ``pre_fix_yaml`` / ``post_fix_yaml`` /
  ``provision_text``;
* hashes may sit under ``hashes: {pre_fix, post_fix, provision}`` or as
  ``pre_fix_sha256`` and friends; when present they are verified and a
  mismatch refuses the case;
* ``defect_kind`` is mapped through a small alias table onto the six kinds;
  anything else is kept as ``other:<kind>`` and counted separately.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from .. import DEFECT_KINDS, VARIANT_CONTROL, VARIANT_DEFECTIVE
from ..cases import CaseSuite, Locator, SuiteError, VerifierCase

_KIND_ALIASES = {
    "amount": "amount_changed",
    "amount_mismatch": "amount_changed",
    "amount_changed": "amount_changed",
    "wrong_amount": "amount_changed",
    "boundary": "boundary_flipped",
    "boundary_direction": "boundary_flipped",
    "boundary_flipped": "boundary_flipped",
    "conjunct": "conjunct_dropped",
    "dropped_conjunct": "conjunct_dropped",
    "conjunct_dropped": "conjunct_dropped",
    "unrepresented_clause": "conjunct_dropped",
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


class RealDefectsError(ValueError):
    """The real-defects directory or one of its cases is unusable."""


def normalise_kind(raw: Any) -> str:
    key = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in _KIND_ALIASES:
        return _KIND_ALIASES[key]
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
    direct = payload.get(f"{name}_sha256")
    return direct.lower() if isinstance(direct, str) else None


def _locator(payload: dict[str, Any]) -> Locator:
    raw = payload.get("locator")
    if isinstance(raw, dict):
        return Locator(
            path=str(raw.get("path") or raw.get("rule_path") or ""),
            rule_index=raw.get("rule_index"),
            rule_name=raw.get("rule_name") or raw.get("rule"),
            detail=str(raw.get("detail") or raw.get("description") or ""),
            before=raw.get("before"),
            after=raw.get("after"),
            token=raw.get("token"),
        )
    if isinstance(raw, str) and raw:
        # "some_rule.versions[0].formula" names the rule in its first segment;
        # "rules[3].versions[0].formula" names it by index only.
        first = raw.split(".")[0]
        rule_name = first if first and "[" not in first else None
        return Locator(path=raw, rule_name=rule_name)
    return Locator(path="", detail="locator missing in case record")


def load_case_file(path: Path) -> tuple[VerifierCase, VerifierCase, dict[str, Any]]:
    """Load one case record into its (control, defective) pair."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RealDefectsError(f"could not read case {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RealDefectsError(f"case {path} is not a JSON object")
    base_dir = path.parent
    case_id = str(payload.get("case_id") or payload.get("id") or path.stem)
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
    kind = normalise_kind(payload.get("defect_kind"))
    citation = str(payload.get("citation") or payload.get("corpus_citation_path") or "")
    origin = {
        "source": "real_defects",
        "case_file": path.name,
        "raw_defect_kind": payload.get("defect_kind"),
        "generator_model": payload.get("generator_model"),
        "fix_reference": payload.get("fix_reference") or payload.get("fix"),
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
    root = Path(root)
    if not root.is_dir():
        raise RealDefectsError(f"real-defects directory not found: {root}")
    files = sorted(
        p
        for p in root.rglob("*.json")
        if p.name not in ("suite.json", "suite.manifest.json")
        and not p.name.startswith(".")
    )
    return files


def build_real_suite(
    root: Path,
    *,
    provision_chars: int,
    truncate,
    name: Optional[str] = None,
) -> CaseSuite:
    """Fold every case record under ``root`` into a suite.

    ``truncate`` is the provision-window function the judges use (the
    referee's head+tail truncation), passed in so this module has no import
    dependency on the judges package.
    """

    root = Path(root)
    files = iter_case_files(root)
    if not files:
        raise RealDefectsError(f"no case records under {root}")
    cases: list[VerifierCase] = []
    other_kinds: dict[str, int] = {}
    for path in files:
        control, defective, _ = load_case_file(path)
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
    identity = {
        "directory": root.name,
        "case_files": [p.relative_to(root).as_posix() for p in files],
        "readme_sha256": _sha256_text(readme.read_text()) if readme.is_file() else None,
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
    try:
        return CaseSuite(
            name=name or f"EncodeBench verifier real {root.name}",
            source_kind="real_defects",
            source_identity=identity,
            cases=cases,
            provision_chars=provision_chars,
            corpus_release=None,
            mutator=None,
            notes=notes,
        )
    except SuiteError as exc:
        raise RealDefectsError(str(exc)) from exc
