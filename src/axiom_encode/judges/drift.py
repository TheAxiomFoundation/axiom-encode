"""Golden-regeneration drift check (scheduled weekly).

Re-generate a random sample of 3-5 already-merged modules with the current
encoder and semantic-diff the output against what is merged, ignoring comments
and key/element ordering. Any real difference is drift — a signal that the
encoder changed behavior on inputs it previously got right (this would have
caught the 0.2.1187 silent wrong-root immediately).

The pure logic here (normalize, diff, sample, issue body) is deterministic and
unit-tested; regeneration itself is injected as a callable so the scheduled
workflow owns the encoder invocation and this module stays side-effect free.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

MIN_SAMPLE = 3
MAX_SAMPLE = 5


def normalize(value: Any) -> Any:
    """Canonicalize a parsed YAML structure, ignoring ordering.

    Dict keys are sorted; list elements are sorted by their canonical JSON so a
    reordered list compares equal. Comments never survive YAML parsing, so they
    are ignored for free. A changed root, key, or value still shows as a diff.
    """

    if isinstance(value, dict):
        return {k: normalize(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, list):
        normed = [normalize(v) for v in value]
        return sorted(normed, key=lambda v: json.dumps(v, sort_keys=True, default=str))
    return value


def semantic_key(yaml_text: str) -> str:
    """A stable, ordering-independent signature of a RuleSpec document."""

    return json.dumps(
        normalize(yaml.safe_load(yaml_text) or {}), sort_keys=True, default=str
    )


def _diff(a: Any, b: Any, path: str, out: list[dict[str, Any]]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            sub = f"{path}.{key}" if path else str(key)
            if key not in a:
                out.append(
                    {
                        "path": sub,
                        "change": "added",
                        "merged": None,
                        "regenerated": b[key],
                    }
                )
            elif key not in b:
                out.append(
                    {
                        "path": sub,
                        "change": "removed",
                        "merged": a[key],
                        "regenerated": None,
                    }
                )
            else:
                _diff(a[key], b[key], sub, out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if a != b:
            out.append(
                {
                    "path": path or "(root)",
                    "change": "list_changed",
                    "merged": a,
                    "regenerated": b,
                }
            )
        return
    if a != b:
        out.append(
            {
                "path": path or "(root)",
                "change": "value_changed",
                "merged": a,
                "regenerated": b,
            }
        )


def semantic_diff(merged_text: str, regenerated_text: str) -> list[dict[str, Any]]:
    """Return structured differences after ordering-insensitive normalization."""

    a = normalize(yaml.safe_load(merged_text) or {})
    b = normalize(yaml.safe_load(regenerated_text) or {})
    out: list[dict[str, Any]] = []
    _diff(a, b, "", out)
    return out


def sample_modules(
    paths: list[Path],
    *,
    k: int = MIN_SAMPLE,
    seed: Optional[int] = None,
) -> list[Path]:
    """Randomly sample 3-5 modules (clamped to the population size)."""

    k = max(MIN_SAMPLE, min(MAX_SAMPLE, k))
    k = min(k, len(paths))
    rng = random.Random(seed)
    return rng.sample(list(paths), k) if paths else []


@dataclass
class DriftResult:
    module: str
    drifted: bool
    diffs: list[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class DriftReport:
    checked: list[DriftResult] = field(default_factory=list)

    @property
    def drifted(self) -> list[DriftResult]:
        return [r for r in self.checked if r.drifted]

    @property
    def errors(self) -> list[DriftResult]:
        return [r for r in self.checked if r.error]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_checked": len(self.checked),
            "n_drifted": len(self.drifted),
            "n_errors": len(self.errors),
            "results": [
                {
                    "module": r.module,
                    "drifted": r.drifted,
                    "error": r.error,
                    "diffs": r.diffs,
                }
                for r in self.checked
            ],
        }


def drift_issue_body(result: DriftResult) -> str:
    """Build a GitHub issue body describing one module's drift."""

    lines = [
        f"The golden-regeneration drift check re-generated `{result.module}` with "
        "the current encoder and found a semantic difference from the merged "
        "module (comments and ordering ignored).",
        "",
        "This is the check that would have caught the 0.2.1187 silent wrong-root "
        "immediately. Investigate whether the encoder change is intended; if so, "
        "re-merge the module, otherwise fix the regression.",
        "",
        f"### Differences ({len(result.diffs)})",
        "",
        "| path | change | merged | regenerated |",
        "| --- | --- | --- | --- |",
    ]
    for d in result.diffs[:50]:
        merged = json.dumps(d.get("merged"), default=str)[:120]
        regen = json.dumps(d.get("regenerated"), default=str)[:120]
        lines.append(f"| `{d['path']}` | {d['change']} | `{merged}` | `{regen}` |")
    if len(result.diffs) > 50:
        lines.append(f"| … | … | {len(result.diffs) - 50} more | … |")
    return "\n".join(lines)


def check_module(
    module: str,
    merged_text: str,
    regenerate: Callable[[str, str], str],
) -> DriftResult:
    """Regenerate one module and diff it. Regeneration errors are recorded.

    A regeneration failure is a visible error, never a silent "no drift".
    """

    try:
        regenerated = regenerate(module, merged_text)
    except Exception as exc:  # noqa: BLE001 - surfaced as a visible error
        return DriftResult(
            module=module, drifted=False, error=f"regeneration failed: {exc}"
        )
    try:
        diffs = semantic_diff(merged_text, regenerated)
    except Exception as exc:  # noqa: BLE001
        return DriftResult(module=module, drifted=False, error=f"diff failed: {exc}")
    return DriftResult(module=module, drifted=bool(diffs), diffs=diffs)


def run_drift_check(
    modules: dict[str, str],
    regenerate: Callable[[str, str], str],
    *,
    k: int = MIN_SAMPLE,
    seed: Optional[int] = None,
) -> DriftReport:
    """Sample 3-5 modules, regenerate, and diff. ``modules`` maps id -> text."""

    sampled = sample_modules(list(modules), k=k, seed=seed)
    report = DriftReport()
    for module in map(str, sampled):
        report.checked.append(check_module(module, modules[module], regenerate))
    return report
