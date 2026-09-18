#!/usr/bin/env python3
"""Merge the triage workflow output into the committed triage record.

The workflow (screen, triage, adversarial verify) returns one structured
record per (commit chunk) and one verifier verdict per kept module. This step
applies the keep rules below, attaches the merged PR URL from the prepared
input bundles, and writes ``triage/triage_merged.json`` (one row per module
touched by a candidate commit, with ``keep`` and ``drop_reason``),
``triage/screen.json`` and ``triage/summary.json``.

Keep rules (documented in the corpus README):

* Only modules the triage reader classified ``fidelity`` or ``unclear`` are
  candidates; ``mechanical``, ``test_only`` and ``not_a_rule_module`` drop.
* Verifier ``confirmed``: keep with the triage kind. A triage ``unclear`` is
  promoted to ``fidelity`` only when the verifier confidence is at least 0.7.
* Verifier ``reclassify``: keep; the verifier's kind replaces the triage kind
  when the verifier confidence is at least 0.6, otherwise the triage kind
  stays and the disagreement is noted.
* Verifier ``mechanical`` or ``not_a_defect`` at confidence 0.7 or higher:
  drop. Below 0.7: keep as ``unclear``.
* Verifier ``unclear`` or no verdict: keep with the triage classification.
* Confidence: the lower of the two readers' confidences when they agree;
  capped at 0.49 for anything that stays ``unclear``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROMOTE_UNCLEAR_AT = 0.7
RECLASSIFY_AT = 0.6
DROP_AT = 0.7
UNCLEAR_CAP = 0.49


def merge_module(
    item: dict[str, Any],
    triage: dict[str, Any],
    module: dict[str, Any],
    verdict: dict[str, Any] | None,
    pr_url: str | None,
) -> dict[str, Any]:
    classification = module["classification"]
    row = {
        "jurisdiction": item["jur"],
        "commit": item["commit"],
        "date": item.get("date") or "",
        "subject": item.get("subject") or "",
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


def pr_url_for(inputs_dir: Path, jur: str, commit: str) -> str | None:
    bundle = inputs_dir / jur / f"{commit[:10]}.json"
    if not bundle.exists():
        return None
    data = json.loads(bundle.read_text(encoding="utf-8"))
    pr = data.get("pr") or {}
    return pr.get("url")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workflow-output", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = json.loads(args.workflow_output.read_text(encoding="utf-8"))
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
            pr_url = pr_url_for(args.inputs_dir, item["jur"], item["commit"])
            for module in triage.get("modules") or []:
                rows.append(
                    merge_module(
                        item, triage, module, verdicts.get(module["index"]), pr_url
                    )
                )
    rows.sort(
        key=lambda r: (r["jurisdiction"], r["date"], r["commit"], r["module_index"])
    )
    triage_dir = args.corpus_dir / "triage"
    triage_dir.mkdir(parents=True, exist_ok=True)
    (triage_dir / "triage_merged.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (triage_dir / "screen.json").write_text(
        json.dumps(output.get("screen") or {}, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    kept = [r for r in rows if r["keep"]]
    summary = {
        "module_rows": len(rows),
        "chunks_with_triage": dict(chunks_seen),
        "chunks_missing": chunks_missing,
        "by_classification": dict(Counter(r["classification"] for r in rows)),
        "verifier_verdicts": dict(
            Counter(
                (r["verifier"] or {}).get("verdict", "missing")
                for r in rows
                if r["classification"] in {"fidelity", "unclear"}
            )
        ),
        "dropped_by_reason": dict(
            Counter(r["drop_reason"] for r in rows if not r["keep"])
        ),
        "kept": len(kept),
        "kept_by_status": dict(Counter(r["triage_status"] for r in kept)),
        "kept_by_kind": dict(Counter(r["defect_kind"] for r in kept)),
        "kept_by_jurisdiction": dict(Counter(r["jurisdiction"] for r in kept)),
        "kept_by_source": dict(Counter(r["candidate_source"] for r in kept)),
    }
    (triage_dir / "summary.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
