"""CLI surface for failure-triage evidence (``axiom-encode triage-evidence``).

Kept in the triage package so ``cli.py`` needs only a short hook, like the
judge stages. The command is deterministic and offline: it reads the given
bundle or run directory, an optional local RuleSpec git repository (local
objects only, never fetched) and an optional local corpus checkout. It calls
no model and reads no credential.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .evidence import (
    EvidenceInputError,
    bounded_input_name,
    build_evidence,
    canonical_json,
    iter_run_inputs,
    load_any,
)
from .material import TriageMaterial
from .probes import PROBE_NAMES
from .views import (
    CheckoutRulespecView,
    GitRefRulespecView,
    LocalCorpusIndex,
    RulespecView,
)

COMMANDS = frozenset({"triage-evidence"})


def register_triage_subparsers(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "triage-evidence",
        help=(
            "Build the deterministic failure-triage evidence bundle "
            "(failure-triage-evidence/v1) for a failed run; no model call"
        ),
    )
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "A workflow failure bundle (metadata.json, or <dir>/x/metadata.json) "
            "or a local run directory (run.json); with --batch, a directory of them"
        ),
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Build every bundle or run under PATH and write evidence.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=("Output file (default: stdout; with --batch, PATH/evidence.jsonl)"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="With --batch, also write the batch summary JSON here",
    )
    parser.add_argument(
        "--lane",
        default=None,
        help="Single mode: the bundle lane to build when it has several",
    )
    parser.add_argument(
        "--rulespec-repo",
        type=Path,
        default=None,
        help=(
            "Local rulespec-<country> git repository; each run's rulespec_ref "
            "is read from its local objects (never fetched)"
        ),
    )
    parser.add_argument(
        "--rulespec-ref",
        default=None,
        help="RuleSpec ref for runs that record none (local pilot runs)",
    )
    parser.add_argument(
        "--rulespec-checkout",
        type=Path,
        default=None,
        help="Use this RuleSpec checkout as it is on disk instead of a ref",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="Local axiom-corpus checkout for the corpus and gate probes",
    )
    parser.add_argument(
        "--no-concept-registry",
        action="store_true",
        help="Do not read the packaged concept registry in the registry probe",
    )


class _ViewCache:
    MAX = 8

    def __init__(self, args: argparse.Namespace) -> None:
        self.repo = args.rulespec_repo
        self.default_ref = args.rulespec_ref
        self.checkout = (
            CheckoutRulespecView(args.rulespec_checkout)
            if args.rulespec_checkout is not None
            else None
        )
        self._views: dict[str, RulespecView | str] = {}

    def view_for(
        self, material: TriageMaterial
    ) -> tuple[RulespecView | None, str | None]:
        if self.checkout is not None:
            return self.checkout, None
        if self.repo is None:
            return None, "no RuleSpec repository was given"
        ref = material.rulespec_ref or self.default_ref
        if not ref:
            return None, "the run records no RuleSpec ref"
        cached = self._views.get(ref)
        if cached is None:
            try:
                cached = GitRefRulespecView(self.repo, ref)
            except (LookupError, ValueError) as exc:
                cached = str(exc)
            if len(self._views) >= self.MAX:
                self._views.pop(next(iter(self._views)))
            self._views[ref] = cached
        if isinstance(cached, str):
            return None, cached
        return cached, None


def _registry(args: argparse.Namespace) -> Any:
    if args.no_concept_registry:
        return None
    from ..concepts.registry import load_concept_registry

    return load_concept_registry()


def _build(
    material: TriageMaterial,
    *,
    views: _ViewCache,
    corpus: LocalCorpusIndex | None,
    registry: Any,
) -> dict[str, Any]:
    view, reason = views.view_for(material)
    return build_evidence(
        material,
        rulespec_view=view,
        corpus=corpus,
        registry=registry,
        rulespec_unavailable_reason=reason,
    )


def _write(path: Path | None, text: str) -> None:
    if path is None:
        sys.stdout.write(text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # An exclusive, unpredictable temporary file: a pre-placed link at a fixed
    # temporary name can never redirect the write.
    descriptor, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def _summary(
    inputs: int,
    bundles: list[dict[str, Any]],
    skipped: list[str],
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    probe_status: dict[str, Counter[str]] = {name: Counter() for name in PROBE_NAMES}
    finding_status: dict[str, Counter[str]] = {name: Counter() for name in PROBE_NAMES}
    hits: dict[str, Counter[str]] = {name: Counter() for name in PROBE_NAMES}
    for bundle in bundles:
        for name in PROBE_NAMES:
            probe = bundle["probes"][name]
            probe_status[name][probe["status"]] += 1
            for status, count in probe["counts"].items():
                finding_status[name][status] += count
                hits[name][status] += 1
    return {
        "inputs": inputs,
        "evidence_bundles": len(bundles),
        "with_candidate": sum(1 for b in bundles if b["candidate"] is not None),
        "not_failed": len(skipped),
        "errors": errors,
        "probe_status": {k: dict(sorted(v.items())) for k, v in probe_status.items()},
        "finding_status_counts": {
            k: dict(sorted(v.items())) for k, v in finding_status.items()
        },
        "bundles_with_finding_status": {
            k: dict(sorted(v.items())) for k, v in hits.items()
        },
    }


def _cmd_triage_evidence(args: argparse.Namespace) -> int:
    views = _ViewCache(args)
    corpus = LocalCorpusIndex(args.corpus_root) if args.corpus_root else None
    registry = _registry(args)
    if not args.batch:
        try:
            materials = load_any(args.path, rulespec_ref=args.rulespec_ref)
        except EvidenceInputError as exc:
            print(f"triage-evidence: {exc}", file=sys.stderr)
            return 1
        if args.lane is not None:
            materials = [m for m in materials if m.lane == args.lane]
        if not materials:
            print("triage-evidence: the run did not fail", file=sys.stderr)
            return 1
        if len(materials) > 1:
            lanes = ", ".join(str(m.lane) for m in materials)
            print(
                f"triage-evidence: the bundle has several lanes ({lanes}); pass --lane",
                file=sys.stderr,
            )
            return 1
        bundle = _build(materials[0], views=views, corpus=corpus, registry=registry)
        # Canonical JSON: the output is what the byte bound measures.
        _write(args.out, canonical_json(bundle) + "\n")
        return 0

    base = Path(os.path.abspath(args.path))
    out = args.out or base / "evidence.jsonl"
    lines: list[str] = []
    bundles: list[dict[str, Any]] = []
    skipped: list[str] = []
    errors: list[dict[str, str]] = []
    inputs = 0
    for path in iter_run_inputs(base):
        inputs += 1
        relative = path.relative_to(base).as_posix()
        try:
            materials = load_any(path, rulespec_ref=args.rulespec_ref)
        except (EvidenceInputError, OSError) as exc:
            errors.append({"input": relative, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if not materials:
            skipped.append(relative)
        for material in materials:
            try:
                bundle = _build(material, views=views, corpus=corpus, registry=registry)
            except (EvidenceInputError, OSError, ValueError, RecursionError) as exc:
                errors.append(
                    {"input": relative, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            bundle["input"]["name"] = bounded_input_name(relative)
            bundles.append(bundle)
            lines.append(canonical_json(bundle) + "\n")
    _write(out, "".join(lines))
    summary = _summary(inputs, bundles, skipped, errors)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary is not None:
        _write(args.summary, text)
    sys.stdout.write(text)
    return 0


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "triage-evidence":
        return _cmd_triage_evidence(args)
    raise ValueError(f"unknown triage command: {args.command}")
