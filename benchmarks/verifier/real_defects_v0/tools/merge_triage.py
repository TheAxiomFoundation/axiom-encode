#!/usr/bin/env python3
"""Merge the triage workflow output into the committed triage record.

The workflow (screen, triage, adversarial verify) returns one structured
record per (commit chunk) and one verifier verdict per kept module. This step
applies the keep rules below, attaches the merged PR URL, applies the recorded
review overrides, and writes ``triage/triage_merged.json`` (one row per module
touched by a candidate commit, with ``keep`` and ``drop_reason``),
``triage/screen.json`` and ``triage/summary.json``.

Keep rules (documented in the corpus README):

* Only modules the triage reader classified ``fidelity`` or ``unclear`` are
  candidates; ``mechanical``, ``test_only`` and ``not_a_rule_module`` drop.
* Verifier ``confirmed``: keep with the triage kind. A triage ``unclear`` is
  promoted to ``fidelity`` only when the verifier confidence is at least 0.7.
* Verifier ``reclassify``: keep; the verifier's kind replaces the triage kind
  when the verifier confidence is at least 0.6, otherwise the triage kind
  stays, the disagreement is noted and the row is ``unclear``.
* Verifier ``mechanical`` or ``not_a_defect`` at confidence 0.7 or higher:
  drop. Below 0.7: keep as ``unclear``.
* Verifier ``unclear``: keep as ``unclear``. No verdict: keep with the triage
  classification.
* Confidence: the lower of the two readers' confidences when they agree;
  capped at 0.49 for anything that stays ``unclear``.

Each row's ``date`` and ``subject`` are the commit's committer date (``%cI``)
and subject line (``%s``), read from the rulespec checkout. The workflow's own
values are not used: for screen-flagged commits the 2026-09 run recorded an
empty date and ``"(screen-flagged) " + <screen reason>`` as the subject. The
screen reason stays in ``screen_reason``. The same replacement is applied to
the flagged entries written to ``triage/screen.json``.

PR URLs come from ``triage/pr_urls.json``, the record of the merged pull
request per candidate commit. ``--inputs-dir`` names the per-commit bundles
``tools/prep_commit.py`` wrote; when given, their PR URLs are read and the
record is rewritten from them.

Review overrides (``triage/review_overrides.json``) are a reviewer's recorded
corrections to single rows: a promotion of an ``unclear`` row to ``fidelity``,
or rule names to drop from the locator. Each names its source and reason, and
must match at least one kept row. An overridden row carries the override in
``review_override``.

The triage readers wrote absolute paths of the machine they ran on into their
notes. ``lib.scrub_local_paths`` replaces them in every row, and
``--scrub-record`` rewrites ``workflow_output.json`` itself the same way
(text-level, so nothing else in the file changes).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
_SPEC = importlib.util.spec_from_file_location(
    "verify_real_defects", ROOT / "scripts" / "verify_real_defects.py"
)
assert _SPEC is not None and _SPEC.loader is not None
lib = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lib)

PROMOTE_UNCLEAR_AT = 0.7
RECLASSIFY_AT = 0.6
DROP_AT = 0.7
UNCLEAR_CAP = 0.49


def row_sort_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (row["jurisdiction"], row["date"], row["commit"], row["module_index"])


def merge_module(
    item: dict[str, Any],
    triage: dict[str, Any],
    module: dict[str, Any],
    verdict: dict[str, Any] | None,
    pr_url: str | None,
    commit_meta: dict[str, Any],
) -> dict[str, Any]:
    classification = module["classification"]
    row = {
        "jurisdiction": item["jur"],
        "commit": item["commit"],
        "date": commit_meta["date"],
        "subject": commit_meta["subject"],
        "candidate_source": "screen" if item.get("screen_reason") else "keyword",
        "screen_reason": item.get("screen_reason"),
        "pr_url": pr_url,
        "module_index": module["index"],
        "module_path": module["path"],
        "classification": classification,
        "defect_kind": module.get("defect_kind"),
        "other_kind": module.get("other_kind") or None,
        "triage_confidence": module.get("confidence"),
        "description": module.get("description_quote") or "",
        "description_source": module.get("quote_source") or "none",
        "rule_names": module.get("rule_names") or [],
        "rule_path": module.get("rule_path") or "",
        "pre_fix_wrong_because": module.get("pre_fix_wrong_because") or "",
        "triage_notes": module.get("triage_notes") or "",
        "commit_summary": triage.get("commit_summary") or "",
        "verifier": verdict,
        "keep": False,
        "drop_reason": None,
        "triage_status": None,
        "confidence": None,
        "triage_notes_combined": "",
        "review_override": None,
    }
    if classification not in {"fidelity", "unclear"}:
        row["drop_reason"] = f"triage_{classification}"
        return row
    kind = module.get("defect_kind") or "other"
    other_kind = module.get("other_kind") or None
    status = classification
    triage_conf = float(module.get("confidence") or 0.0)
    notes: list[str] = []
    if verdict is None:
        notes.append("verifier: no verdict recorded (agent did not return)")
        conf = triage_conf
    else:
        v = verdict.get("verdict")
        vconf = float(verdict.get("confidence") or 0.0)
        vkind = verdict.get("defect_kind") or "none"
        if v == "confirmed":
            if status == "unclear" and vconf >= PROMOTE_UNCLEAR_AT:
                status = "fidelity"
                notes.append(
                    f"verifier confirmed at {vconf:.2f}; promoted from unclear"
                )
            else:
                notes.append(f"verifier confirmed at {vconf:.2f}")
            if vkind not in {kind, "none"}:
                notes.append(
                    f"verifier named kind {vkind} while confirming; triage kind kept"
                )
            conf = min(triage_conf, vconf)
        elif v == "reclassify":
            if vconf >= RECLASSIFY_AT and vkind not in {"none", ""}:
                notes.append(
                    f"verifier reclassified {kind} -> {vkind} at {vconf:.2f}; verifier kind used"
                )
                kind = vkind
                other_kind = verdict.get("other_kind") or (
                    other_kind if vkind == "other" else None
                )
                conf = min(triage_conf, vconf)
            else:
                notes.append(
                    f"verifier proposed {vkind} at {vconf:.2f} (below {RECLASSIFY_AT}); triage kind kept"
                )
                status = "unclear"
                conf = min(triage_conf, vconf)
        elif v in {"mechanical", "not_a_defect"}:
            if vconf >= DROP_AT:
                row["drop_reason"] = f"verifier_{v}"
                row["triage_notes_combined"] = (
                    f"verifier {v} at {vconf:.2f}: {verdict.get('justification', '')}"
                )
                return row
            status = "unclear"
            notes.append(
                f"verifier said {v} at {vconf:.2f} (below {DROP_AT}); kept as unclear"
            )
            conf = min(triage_conf, vconf)
        else:  # unclear
            status = "unclear"
            notes.append(f"verifier unclear at {vconf:.2f}")
            conf = min(triage_conf, vconf)
    if status == "unclear":
        conf = min(conf, UNCLEAR_CAP)
    row.update(
        {
            "keep": True,
            "triage_status": status,
            "defect_kind": kind,
            "other_kind": other_kind,
            "confidence": round(conf, 3),
            "triage_notes_combined": " | ".join(
                part for part in [module.get("triage_notes") or "", *notes] if part
            ),
        }
    )
    return row


def screen_entry_from_git(
    flagged: dict[str, Any], commit_meta: dict[str, Any]
) -> dict[str, Any]:
    """A screen-flagged entry with the commit's own date and subject."""

    entry = dict(flagged)
    entry["date"] = commit_meta["date"]
    entry["subject"] = commit_meta["subject"]
    return entry


def bundle_exists(inputs_dir: Path, jur: str, commit: str) -> bool:
    return (inputs_dir / jur / f"{commit[:10]}.json").exists()


def bundle_pr_url(inputs_dir: Path, jur: str, commit: str) -> str | None:
    """The merged PR's URL from a ``tools/prep_commit.py`` bundle, if any."""

    bundle = inputs_dir / jur / f"{commit[:10]}.json"
    if not bundle.exists():
        return None
    data = json.loads(bundle.read_text(encoding="utf-8"))
    pr = data.get("pr") or {}
    return pr.get("url")


def promoted_confidence(row: dict[str, Any]) -> float:
    """A promoted row's confidence: the lower of the two readers', uncapped."""

    values = [float(row.get("triage_confidence") or 0.0)]
    verifier = row.get("verifier") or {}
    if verifier.get("confidence") is not None:
        values.append(float(verifier["confidence"]))
    return round(min(values), 3)


def apply_review_overrides(
    rows: list[dict[str, Any]], overrides: list[dict[str, Any]]
) -> list[str]:
    """Apply each recorded override to the kept rows it matches.

    ``match`` holds row fields that must all be equal. ``set_triage_status``
    replaces the status (a promotion to ``fidelity`` also lifts the unclear
    confidence cap); ``drop_rule_names_matching`` is a regular expression
    whose full matches leave ``rule_names``. Returns the ids applied; an
    override that matches no kept row is an error.
    """

    applied: list[str] = []
    for row in rows:
        row["review_override"] = None
    for override in overrides:
        matched = 0
        for row in rows:
            if not row["keep"] or any(
                row.get(field) != value for field, value in override["match"].items()
            ):
                continue
            matched += 1
            changed: dict[str, Any] = {}
            status = override.get("set_triage_status")
            if status and row["triage_status"] != status:
                changed["triage_status"] = [row["triage_status"], status]
                row["triage_status"] = status
                if status == "fidelity":
                    confidence = promoted_confidence(row)
                    changed["confidence"] = [row["confidence"], confidence]
                    row["confidence"] = confidence
            pattern = override.get("drop_rule_names_matching")
            if pattern:
                kept = [n for n in row["rule_names"] if not re.fullmatch(pattern, n)]
                if kept != row["rule_names"]:
                    changed["rule_names_dropped"] = [
                        n for n in row["rule_names"] if n not in kept
                    ]
                    row["rule_names"] = kept
            if not changed:
                continue
            if row["review_override"] is not None:
                raise ValueError(
                    f"two review overrides change {row['commit']} {row['module_path']}"
                )
            row["review_override"] = {
                "id": override["id"],
                "source": override["source"],
                "reason": override["reason"],
                "changed": changed,
            }
        if not matched:
            raise ValueError(f"review override {override['id']} matches no kept row")
        applied.append(override["id"])
    return applied


def _sorted_counts(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workflow-output", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--rulespec-us", type=Path, required=True)
    parser.add_argument("--rulespec-uk", type=Path, required=True)
    parser.add_argument(
        "--scrub-record",
        action="store_true",
        help="rewrite --workflow-output with local absolute paths replaced",
    )
    args = parser.parse_args(argv)
    raw = args.workflow_output.read_text(encoding="utf-8")
    if args.scrub_record and lib.scrub_local_paths(raw) != raw:
        raw = lib.scrub_local_paths(raw)
        args.workflow_output.write_text(raw, encoding="utf-8")
    output = json.loads(raw)
    triage_dir = args.corpus_dir / "triage"
    triage_dir.mkdir(parents=True, exist_ok=True)
    repos = {"us": args.rulespec_us, "uk": args.rulespec_uk}
    commits: dict[str, set[str]] = {"us": set(), "uk": set()}
    for bucket in ("keyword", "flagged"):
        for entry in output.get(bucket) or []:
            commits[entry["item"]["jur"]].add(entry["item"]["commit"])
    screen = output.get("screen") or {}
    for flagged in screen.get("flagged") or []:
        commits[flagged["jur"]].add(flagged["commit"])
    meta = {jur: lib.commit_metadata(repos[jur], commits[jur]) for jur in repos}
    pr_urls_path = triage_dir / "pr_urls.json"
    pr_urls: dict[str, dict[str, str | None]] = {"uk": {}, "us": {}}
    if pr_urls_path.exists():
        pr_urls.update(json.loads(pr_urls_path.read_text(encoding="utf-8"))["pr_urls"])
    rows: list[dict[str, Any]] = []
    chunks_seen = Counter()
    chunks_missing: list[dict[str, Any]] = []
    for bucket in ("keyword", "flagged"):
        for entry in output.get(bucket) or []:
            item = entry["item"]
            triage = entry.get("triage")
            if not triage:
                chunks_missing.append(
                    {
                        "jurisdiction": item["jur"],
                        "commit": item["commit"],
                        "chunk": item.get("chunk"),
                    }
                )
                continue
            chunks_seen[item["jur"]] += 1
            verdicts = {
                v["module_index"]: v["verdict"] for v in entry.get("verdicts") or []
            }
            if args.inputs_dir is not None and bundle_exists(
                args.inputs_dir, item["jur"], item["commit"]
            ):
                # A bundle is the fresher source; a commit with no bundle
                # keeps the URL already on record.
                pr_urls[item["jur"]][item["commit"][:10]] = bundle_pr_url(
                    args.inputs_dir, item["jur"], item["commit"]
                )
            pr_url = pr_urls[item["jur"]].get(item["commit"][:10])
            for module in triage.get("modules") or []:
                rows.append(
                    lib.scrub_json(
                        merge_module(
                            item,
                            triage,
                            module,
                            verdicts.get(module["index"]),
                            pr_url,
                            meta[item["jur"]][item["commit"]],
                        )
                    )
                )
    rows.sort(key=row_sort_key)
    overrides_path = triage_dir / "review_overrides.json"
    overrides = (
        json.loads(overrides_path.read_text(encoding="utf-8"))["overrides"]
        if overrides_path.exists()
        else []
    )
    applied = apply_review_overrides(rows, overrides)
    screen = lib.scrub_json(dict(screen))
    screen["flagged"] = [
        screen_entry_from_git(flagged, meta[flagged["jur"]][flagged["commit"]])
        for flagged in screen.get("flagged") or []
    ]
    (triage_dir / "triage_merged.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (triage_dir / "screen.json").write_text(
        json.dumps(screen, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pr_urls_path.write_text(
        json.dumps(
            {
                "about": (
                    "The merged pull request that carried each candidate commit "
                    "(gh api repos/<repo>/commits/<sha>/pulls, read by "
                    "tools/prep_commit.py during the 2026-09 triage run); null "
                    "when GitHub listed none."
                ),
                "pr_urls": {
                    jur: dict(sorted(urls.items()))
                    for jur, urls in sorted(pr_urls.items())
                },
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    kept = [r for r in rows if r["keep"]]
    summary = {
        "module_rows": len(rows),
        "chunks_with_triage": dict(sorted(chunks_seen.items())),
        "chunks_missing": chunks_missing,
        "by_classification": _sorted_counts(r["classification"] for r in rows),
        "verifier_verdicts": _sorted_counts(
            (r["verifier"] or {}).get("verdict", "missing")
            for r in rows
            if r["classification"] in {"fidelity", "unclear"}
        ),
        "dropped_by_reason": _sorted_counts(
            r["drop_reason"] for r in rows if not r["keep"]
        ),
        "kept": len(kept),
        "kept_by_status": _sorted_counts(r["triage_status"] for r in kept),
        "kept_by_kind": _sorted_counts(r["defect_kind"] for r in kept),
        "kept_by_jurisdiction": _sorted_counts(r["jurisdiction"] for r in kept),
        "kept_by_source": _sorted_counts(r["candidate_source"] for r in kept),
        "review_overrides_applied": applied,
    }
    (triage_dir / "summary.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
