"""Case model and suite serialisation for the verifier track.

A *suite* is an ordered list of cases. Cases come in pairs: one ``control``
and one ``defective`` artifact judged against the same provision window. The
suite carries enough identity (source, mutator version, provision window
size, per-case digests) for the board to refuse folding runs of different
suites, mirroring the encoder track's comparability contract.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from . import (
    DEFECT_KINDS,
    SUITE_SCHEMA,
    VARIANT_CONTROL,
    VARIANT_DEFECTIVE,
    VARIANTS,
)
from .canonical import canonical_json_sha256, text_sha256, utc_now_iso


class SuiteError(ValueError):
    """A suite file is unreadable, malformed, or internally inconsistent."""


@dataclass(frozen=True)
class Locator:
    """Where a defect was planted (or, for real cases, where the fix landed)."""

    path: str
    rule_index: Optional[int] = None
    rule_name: Optional[str] = None
    detail: str = ""
    before: Optional[str] = None
    after: Optional[str] = None
    # A distinctive token a localised finding would mention: the new value,
    # the dropped identifier, the operand next to a flipped boundary, ...
    token: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Locator":
        return cls(
            path=str(payload.get("path", "")),
            rule_index=payload.get("rule_index"),
            rule_name=payload.get("rule_name"),
            detail=str(payload.get("detail", "")),
            before=payload.get("before"),
            after=payload.get("after"),
            token=payload.get("token"),
        )


@dataclass(frozen=True)
class VerifierCase:
    """One (provision window, artifact) the judges read."""

    pair_id: str
    variant: str
    defect_kind: str
    citation: str
    provision_text: str
    artifact_text: str
    locator: Optional[Locator] = None
    origin: dict[str, Any] = field(default_factory=dict)
    # ``known_good_gate``: the control passed the encoder's compile/CI/apply
    # gates. ``unverified``: a post-fix artifact from a real repair, believed
    # better than the pre-fix one but not proven clean.
    control_clean: str = "known_good_gate"
    provision_full_sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise SuiteError(f"unknown variant {self.variant!r}")
        if self.variant == VARIANT_DEFECTIVE and self.locator is None:
            raise SuiteError(f"defective case {self.pair_id!r} has no locator")

    @property
    def case_id(self) -> str:
        return f"{self.pair_id}:{self.variant}"

    @property
    def is_defective(self) -> bool:
        return self.variant == VARIANT_DEFECTIVE

    @property
    def provision_sha256(self) -> str:
        return text_sha256(self.provision_text)

    @property
    def artifact_sha256(self) -> str:
        return text_sha256(self.artifact_text)

    def identity(self, index: int) -> dict[str, Any]:
        """The comparability identity of this case (no texts, only digests)."""

        return {
            "index": index,
            "case_id": self.case_id,
            "pair_id": self.pair_id,
            "variant": self.variant,
            "defect_kind": self.defect_kind,
            "citation": self.citation,
            "provision_sha256": self.provision_sha256,
            "artifact_sha256": self.artifact_sha256,
            "locator": self.locator.to_dict() if self.locator else None,
            "control_clean": self.control_clean,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "pair_id": self.pair_id,
            "variant": self.variant,
            "defect_kind": self.defect_kind,
            "citation": self.citation,
            "provision_text": self.provision_text,
            "artifact_text": self.artifact_text,
            "locator": self.locator.to_dict() if self.locator else None,
            "origin": dict(self.origin),
            "control_clean": self.control_clean,
            "provision_full_sha256": self.provision_full_sha256,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VerifierCase":
        locator = payload.get("locator")
        return cls(
            pair_id=str(payload["pair_id"]),
            variant=str(payload["variant"]),
            defect_kind=str(payload["defect_kind"]),
            citation=str(payload.get("citation", "")),
            provision_text=str(payload["provision_text"]),
            artifact_text=str(payload["artifact_text"]),
            locator=Locator.from_dict(locator) if isinstance(locator, dict) else None,
            origin=dict(payload.get("origin") or {}),
            control_clean=str(payload.get("control_clean", "known_good_gate")),
            provision_full_sha256=payload.get("provision_full_sha256"),
        )


@dataclass
class CaseSuite:
    """An ordered case set plus the identity the board compares."""

    name: str
    source_kind: str
    source_identity: dict[str, Any]
    cases: list[VerifierCase]
    provision_chars: int
    corpus_release: Optional[str] = None
    mutator: Optional[dict[str, Any]] = None
    generated_at: str = field(default_factory=utc_now_iso)
    notes: list[str] = field(default_factory=list)

    # -- identity -----------------------------------------------------------

    def case_identities(self) -> list[dict[str, Any]]:
        return [case.identity(index) for index, case in enumerate(self.cases, 1)]

    @property
    def mutator_version(self) -> Optional[str]:
        if not self.mutator:
            return None
        version = self.mutator.get("version")
        return str(version) if version is not None else None

    def identity(self) -> dict[str, Any]:
        """Everything the board requires to match before folding two runs."""

        return {
            "schema": SUITE_SCHEMA,
            "name": self.name,
            "source_kind": self.source_kind,
            "corpus_release": self.corpus_release,
            "mutator_version": self.mutator_version,
            "provision_chars": self.provision_chars,
            "case_identities": self.case_identities(),
        }

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.identity())

    def summary(self) -> dict[str, Any]:
        """Compact description carried into every results payload."""

        kinds = {kind: 0 for kind in DEFECT_KINDS}
        for case in self.cases:
            if case.is_defective:
                kinds[case.defect_kind] = kinds.get(case.defect_kind, 0) + 1
        return {
            "schema": SUITE_SCHEMA,
            "name": self.name,
            "sha256": self.sha256,
            "source_kind": self.source_kind,
            "corpus_release": self.corpus_release,
            "mutator_version": self.mutator_version,
            "provision_chars": self.provision_chars,
            "case_count": len(self.cases),
            "pair_count": sum(1 for case in self.cases if case.is_defective),
            "defective_by_kind": kinds,
        }

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SUITE_SCHEMA,
            "name": self.name,
            "generated_at": self.generated_at,
            "source": {"kind": self.source_kind, "identity": self.source_identity},
            "corpus_release": self.corpus_release,
            "mutator": self.mutator,
            "provision_chars": self.provision_chars,
            "defect_kinds": list(DEFECT_KINDS),
            "notes": list(self.notes),
            "cases": [case.to_dict() for case in self.cases],
            "sha256": self.sha256,
        }

    def manifest(self) -> dict[str, Any]:
        """Identity-only view (no texts), small enough to commit."""

        return {
            "schema": SUITE_SCHEMA + "+manifest",
            "name": self.name,
            "generated_at": self.generated_at,
            "source": {"kind": self.source_kind, "identity": self.source_identity},
            "corpus_release": self.corpus_release,
            "mutator": self.mutator,
            "provision_chars": self.provision_chars,
            "defect_kinds": list(DEFECT_KINDS),
            "notes": list(self.notes),
            "summary": self.summary(),
            "case_identities": self.case_identities(),
            "sha256": self.sha256,
        }

    def write(self, out_dir: Path) -> tuple[Path, Path]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        suite_path = out_dir / "suite.json"
        manifest_path = out_dir / "suite.manifest.json"
        suite_path.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False))
        manifest_path.write_text(
            json.dumps(self.manifest(), indent=1, ensure_ascii=False)
        )
        return suite_path, manifest_path

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CaseSuite":
        if not isinstance(payload, dict):
            raise SuiteError("suite payload must be a JSON object")
        if payload.get("schema") != SUITE_SCHEMA:
            raise SuiteError(
                f"suite carries schema {payload.get('schema')!r}; expected "
                f"{SUITE_SCHEMA!r}"
            )
        source = payload.get("source") or {}
        try:
            cases = [VerifierCase.from_dict(item) for item in payload["cases"]]
        except (KeyError, TypeError) as exc:
            raise SuiteError(f"suite cases are malformed: {exc}") from exc
        suite = cls(
            name=str(payload.get("name", "")),
            source_kind=str(source.get("kind", "")),
            source_identity=dict(source.get("identity") or {}),
            cases=cases,
            provision_chars=int(payload.get("provision_chars", 0)),
            corpus_release=payload.get("corpus_release"),
            mutator=payload.get("mutator"),
            generated_at=str(payload.get("generated_at", "")),
            notes=list(payload.get("notes") or []),
        )
        recorded = payload.get("sha256")
        if recorded is not None and recorded != suite.sha256:
            raise SuiteError(
                "suite sha256 does not match its content; the file was edited "
                "after it was written"
            )
        _check_pairs(suite)
        return suite

    @classmethod
    def load(cls, path: Path) -> "CaseSuite":
        path = Path(path)
        if path.is_dir():
            path = path / "suite.json"
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SuiteError(f"could not read suite {path}: {exc}") from exc
        return cls.from_dict(payload)


def _check_pairs(suite: CaseSuite) -> None:
    """Every pair has exactly one control and one defective case."""

    seen: dict[str, set[str]] = {}
    for case in suite.cases:
        variants = seen.setdefault(case.pair_id, set())
        if case.variant in variants:
            raise SuiteError(
                f"pair {case.pair_id!r} carries two {case.variant!r} cases"
            )
        variants.add(case.variant)
    incomplete = sorted(
        pair
        for pair, variants in seen.items()
        if variants != {VARIANT_CONTROL, VARIANT_DEFECTIVE}
    )
    if incomplete:
        raise SuiteError(
            "pairs missing a control or defective member: "
            + ", ".join(incomplete[:8])
            + (" ..." if len(incomplete) > 8 else "")
        )
