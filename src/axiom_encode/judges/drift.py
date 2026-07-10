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

import hashlib
import html
import json
import random
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Optional

import yaml

MIN_SAMPLE = 3
MAX_SAMPLE = 5
MAX_YAML_BYTES = 1024 * 1024
MAX_YAML_NODES = 20_000
MAX_YAML_DEPTH = 64
MAX_RETAINED_DIFFS = 50
MAX_DIFF_PATH_CHARS = 4096
MAX_DIFF_VALUE_CHARS = 2048
MAX_ERROR_CHARS = 4096
MAX_GITHUB_ISSUE_BODY_BYTES = 60_000
MAX_GITHUB_ISSUE_TITLE_CHARS = 240
MAX_GITHUB_ISSUE_PATH_PREVIEW_CHARS = 256


def _safe_load_drift_yaml(yaml_text: str) -> Any:
    """Load bounded, non-aliased YAML for deterministic semantic comparison.

    PyYAML preserves aliases as shared Python objects. Recursively normalizing
    even a tiny exponential alias graph can therefore consume unbounded time
    and memory. Scan the event stream first, where aliases have not been
    expanded, and enforce conservative file, node, and nesting budgets.
    """

    if len(yaml_text.encode("utf-8")) > MAX_YAML_BYTES:
        raise ValueError(f"drift YAML exceeds the {MAX_YAML_BYTES}-byte safety limit")

    depth = 0
    nodes = 0
    for event in yaml.parse(yaml_text):
        if isinstance(event, yaml.events.AliasEvent):
            raise ValueError("drift YAML aliases are not allowed")
        if isinstance(
            event,
            (
                yaml.events.MappingStartEvent,
                yaml.events.SequenceStartEvent,
                yaml.events.ScalarEvent,
            ),
        ):
            nodes += 1
            if nodes > MAX_YAML_NODES:
                raise ValueError(
                    f"drift YAML exceeds the {MAX_YAML_NODES}-node safety limit"
                )
        if isinstance(
            event,
            (yaml.events.MappingStartEvent, yaml.events.SequenceStartEvent),
        ):
            depth += 1
            if depth > MAX_YAML_DEPTH:
                raise ValueError(
                    f"drift YAML exceeds the {MAX_YAML_DEPTH}-level safety limit"
                )
        elif isinstance(
            event,
            (yaml.events.MappingEndEvent, yaml.events.SequenceEndEvent),
        ):
            depth -= 1

    return yaml.safe_load(yaml_text)


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
        normalize(_safe_load_drift_yaml(yaml_text) or {}),
        sort_keys=True,
        default=str,
    )


def _bounded_diff_path(path: str) -> str:
    """Return one control-free, bounded display path while traversing YAML."""

    escaped = (
        "".join(f"\\u{ord(char):04x}" if ord(char) < 32 else char for char in path)
        or "(root)"
    )
    if len(escaped) <= MAX_DIFF_PATH_CHARS:
        return escaped
    digest = hashlib.sha256(escaped.encode("utf-8")).hexdigest()[:12]
    suffix = f"…[sha256:{digest}]"
    return escaped[: MAX_DIFF_PATH_CHARS - len(suffix)] + suffix


def _bounded_diff_value(value: Any) -> Any:
    """Return a JSON-safe diff value whose report representation is bounded."""

    safe_value = json.loads(json.dumps(value, sort_keys=True, default=str))
    serialized = json.dumps(safe_value, sort_keys=True, default=str)
    if len(serialized) <= MAX_DIFF_VALUE_CHARS:
        return safe_value
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    suffix = f"…[truncated; sha256:{digest}]"
    return serialized[: MAX_DIFF_VALUE_CHARS - len(suffix)] + suffix


def _bounded_error(error: str) -> str:
    """Bound diagnostics so every producer report satisfies publisher schema."""

    if len(error) <= MAX_ERROR_CHARS:
        return error
    digest = hashlib.sha256(error.encode("utf-8")).hexdigest()[:12]
    suffix = f"…[truncated; sha256:{digest}]"
    return error[: MAX_ERROR_CHARS - len(suffix)] + suffix


def _record_diff(
    out: list[dict[str, Any]],
    total: list[int],
    *,
    path: str,
    change: str,
    merged: Any,
    regenerated: Any,
) -> None:
    """Count every difference while retaining only a bounded issue preview."""

    total[0] += 1
    if len(out) >= MAX_RETAINED_DIFFS:
        return
    out.append(
        {
            "path": _bounded_diff_path(path),
            "change": change,
            "merged": _bounded_diff_value(merged),
            "regenerated": _bounded_diff_value(regenerated),
        }
    )


def _diff(
    a: Any,
    b: Any,
    path: str,
    out: list[dict[str, Any]],
    total: list[int],
) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            sub = _bounded_diff_path(f"{path}.{key}" if path else str(key))
            if key not in a:
                _record_diff(
                    out,
                    total,
                    path=sub,
                    change="added",
                    merged=None,
                    regenerated=b[key],
                )
            elif key not in b:
                _record_diff(
                    out,
                    total,
                    path=sub,
                    change="removed",
                    merged=a[key],
                    regenerated=None,
                )
            else:
                _diff(a[key], b[key], sub, out, total)
        return
    if isinstance(a, list) and isinstance(b, list):
        if a != b:
            _record_diff(
                out,
                total,
                path=path or "(root)",
                change="list_changed",
                merged=a,
                regenerated=b,
            )
        return
    if a != b:
        _record_diff(
            out,
            total,
            path=path or "(root)",
            change="value_changed",
            merged=a,
            regenerated=b,
        )


def semantic_diff(merged_text: str, regenerated_text: str) -> list[dict[str, Any]]:
    """Return structured differences after ordering-insensitive normalization."""

    return _semantic_diff_with_count(merged_text, regenerated_text)[0]


def _semantic_diff_with_count(
    merged_text: str, regenerated_text: str
) -> tuple[list[dict[str, Any]], int]:
    """Return retained differences and the full bounded-tree difference count."""

    a = normalize(_safe_load_drift_yaml(merged_text) or {})
    b = normalize(_safe_load_drift_yaml(regenerated_text) or {})
    out: list[dict[str, Any]] = []
    total = [0]
    _diff(a, b, "", out, total)
    return out, total[0]


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
    total_diffs: int | None = None

    @property
    def diff_count(self) -> int:
        return self.total_diffs if self.total_diffs is not None else len(self.diffs)


@dataclass
class DriftReport:
    checked: list[DriftResult] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DriftReport":
        """Load the bounded JSON report passed to the isolated publisher job."""

        if not isinstance(payload, dict) or not isinstance(
            payload.get("results"), list
        ):
            raise ValueError("drift report must contain a results list")
        raw_results = payload["results"]
        if len(raw_results) > MAX_SAMPLE:
            raise ValueError(f"drift report contains more than {MAX_SAMPLE} results")

        checked: list[DriftResult] = []
        seen_modules: set[str] = set()
        for raw in raw_results:
            if not isinstance(raw, dict):
                raise ValueError("drift report result must be an object")
            module = raw.get("module")
            drifted = raw.get("drifted")
            error = raw.get("error")
            diffs = raw.get("diffs")
            n_diffs = raw.get("n_diffs", len(diffs) if isinstance(diffs, list) else 0)
            if (
                not isinstance(module, str)
                or not module
                or len(module) > 4096
                or "\\" in module
                or any(ord(char) < 32 for char in module)
            ):
                raise ValueError("drift report result has an invalid module")
            module_path = PurePosixPath(module)
            if (
                module_path.is_absolute()
                or module_path.suffix != ".yaml"
                or module_path.as_posix() != module
                or any(part in {"", ".", ".."} for part in module_path.parts)
                or module in seen_modules
            ):
                raise ValueError(
                    "drift report result has an unsafe or duplicate module"
                )
            seen_modules.add(module)
            if not isinstance(drifted, bool):
                raise ValueError("drift report result has a non-boolean drifted field")
            if error is not None and (
                not isinstance(error, str) or not error or len(error) > 4096
            ):
                raise ValueError("drift report result has an invalid error")
            if not isinstance(diffs, list):
                raise ValueError("drift report result has an invalid diffs list")
            if (
                not isinstance(n_diffs, int)
                or isinstance(n_diffs, bool)
                or n_diffs < len(diffs)
                or len(diffs) > MAX_RETAINED_DIFFS
            ):
                raise ValueError("drift report result has an invalid diff count")
            for diff in diffs:
                if not isinstance(diff, dict):
                    raise ValueError("drift report diff must be an object")
                path = diff.get("path")
                change = diff.get("change")
                if (
                    not isinstance(path, str)
                    or not path
                    or len(path) > 4096
                    or any(ord(char) < 32 for char in path)
                    or not isinstance(change, str)
                    or change
                    not in {"added", "removed", "list_changed", "value_changed"}
                    or "merged" not in diff
                    or "regenerated" not in diff
                ):
                    raise ValueError("drift report diff has invalid required fields")
            if error is not None:
                if drifted or diffs or n_diffs:
                    raise ValueError("drift report error result also claims drift")
            elif drifted != bool(n_diffs):
                raise ValueError("drift report drift flag does not match its diffs")
            checked.append(
                DriftResult(
                    module=module,
                    drifted=drifted,
                    error=error,
                    diffs=diffs,
                    total_diffs=n_diffs,
                )
            )

        report = cls(checked=checked)
        expected_counts = {
            "n_checked": len(report.checked),
            "n_drifted": len(report.drifted),
            "n_errors": len(report.errors),
        }
        if any(
            not isinstance(payload.get(name), int)
            or isinstance(payload.get(name), bool)
            or payload.get(name) != value
            for name, value in expected_counts.items()
        ):
            raise ValueError("drift report summary counts do not match its results")
        return report

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
                    "error": _bounded_error(r.error) if r.error else None,
                    "n_diffs": max(r.diff_count, len(r.diffs)),
                    "diffs": [
                        {
                            "path": _bounded_diff_path(str(diff.get("path") or "")),
                            "change": diff.get("change"),
                            "merged": _bounded_diff_value(diff.get("merged")),
                            "regenerated": _bounded_diff_value(diff.get("regenerated")),
                        }
                        for diff in r.diffs[:MAX_RETAINED_DIFFS]
                    ],
                }
                for r in self.checked
            ],
        }


def drift_issue_body(result: DriftResult) -> str:
    """Build a GitHub issue body describing one module's drift."""

    lines = [
        "The golden-regeneration drift check re-generated "
        f"{_github_inline_code(result.module)} with "
        "the current encoder and found a semantic difference from the merged "
        "module (comments and ordering ignored).",
        "",
        "This is the check that would have caught the 0.2.1187 silent wrong-root "
        "immediately. Investigate whether the encoder change is intended; if so, "
        "re-merge the module, otherwise fix the regression.",
        "",
        f"### Differences ({result.diff_count})",
        "",
        "| path | change | merged | regenerated |",
        "| --- | --- | --- | --- |",
    ]
    displayed = 0
    for d in result.diffs[:MAX_RETAINED_DIFFS]:
        merged = json.dumps(d.get("merged"), default=str)[:120]
        regen = json.dumps(d.get("regenerated"), default=str)[:120]
        path = _bounded_text_with_digest(
            str(d["path"]),
            MAX_GITHUB_ISSUE_PATH_PREVIEW_CHARS,
        )
        row = (
            "| "
            f"{_github_inline_code(path, table_cell=True)} | "
            f"{_github_inline_code(d['change'], table_cell=True)} | "
            f"{_github_inline_code(merged, table_cell=True)} | "
            f"{_github_inline_code(regen, table_cell=True)} |"
        )
        if len("\n".join((*lines, row)).encode("utf-8")) > (
            MAX_GITHUB_ISSUE_BODY_BYTES - 256
        ):
            break
        lines.append(row)
        displayed += 1
    if result.diff_count > displayed:
        lines.append(f"| … | … | {result.diff_count - displayed} more | … |")
    return _bounded_utf8("\n".join(lines), MAX_GITHUB_ISSUE_BODY_BYTES)


def drift_error_issue_body(result: DriftResult) -> str:
    """Build a non-interpretable GitHub issue body for a regeneration error."""

    body = (
        "Golden regeneration could not check "
        f"{_github_inline_code(result.module)}:\n\n"
        f"<pre>{_github_safe_text(result.error or '')}</pre>\n"
    )
    return _bounded_utf8(body, MAX_GITHUB_ISSUE_BODY_BYTES)


def drift_issue_title(result: DriftResult, *, regeneration_error: bool = False) -> str:
    """Return a mention-safe issue title that fits the publisher budget."""

    kind = "regeneration error" if regeneration_error else "drift"
    prefix = f"Golden-regeneration {kind}: "
    safe_module = result.module.replace("@", "＠")
    available = MAX_GITHUB_ISSUE_TITLE_CHARS - len(prefix)
    return prefix + _bounded_text_with_digest(safe_module, available)


def _bounded_text_with_digest(value: str, max_chars: int) -> str:
    """Truncate display text with a stable identity-preserving digest."""

    if len(value) <= max_chars:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    suffix = f"…[sha256:{digest}]"
    return value[: max_chars - len(suffix)] + suffix


def _bounded_utf8(value: str, max_bytes: int) -> str:
    """Apply a final byte budget to text passed to an external publisher."""

    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    suffix = f"\n\n[truncated; sha256:{digest}]"
    available = max_bytes - len(suffix.encode("utf-8"))
    prefix = encoded[:available].decode("utf-8", errors="ignore")
    return prefix + suffix


def enforce_github_issue_body_budget(value: str) -> str:
    """Re-apply the final body budget after publisher-side redaction."""

    return _bounded_utf8(value, MAX_GITHUB_ISSUE_BODY_BYTES)


def enforce_github_issue_title_budget(value: str) -> str:
    """Re-apply the final title budget after publisher-side redaction."""

    return _bounded_text_with_digest(value, MAX_GITHUB_ISSUE_TITLE_CHARS)


def _github_safe_text(value: object) -> str:
    """Escape untrusted report text and neutralize GitHub mentions."""

    safe = html.escape(str(value), quote=True)
    for character, entity in (
        ("@", "&#64;"),
        ("`", "&#96;"),
        ("[", "&#91;"),
        ("]", "&#93;"),
        ("(", "&#40;"),
        (")", "&#41;"),
    ):
        safe = safe.replace(character, entity)
    return safe


def _github_inline_code(value: object, *, table_cell: bool = False) -> str:
    """Render untrusted text as HTML code without Markdown interpretation."""

    safe = _github_safe_text(value).replace("\r", " ").replace("\n", " ")
    if table_cell:
        safe = safe.replace("|", "&#124;")
    return f"<code>{safe}</code>"


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
            module=module,
            drifted=False,
            error=_bounded_error(f"regeneration failed: {exc}"),
        )
    try:
        diffs, total_diffs = _semantic_diff_with_count(merged_text, regenerated)
    except Exception as exc:  # noqa: BLE001
        return DriftResult(
            module=module,
            drifted=False,
            error=_bounded_error(f"diff failed: {exc}"),
        )
    return DriftResult(
        module=module,
        drifted=bool(total_diffs),
        diffs=diffs,
        total_diffs=total_diffs,
    )


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
