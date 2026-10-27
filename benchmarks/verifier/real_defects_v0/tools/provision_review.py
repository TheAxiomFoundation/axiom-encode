#!/usr/bin/env python3
"""The per-case provision review: can a judge decide the case from its provision?

``provision.txt`` is resolved from the module's first ``corpus_citation_path``
only. The text that shows the pre-fix module is wrong can sit under another
citation, or under none the module names. ``tools/check_evidence.py`` tests
that mechanically; this tool supports the reading that settles it per case
(decision d885: hold the corpus until each case is checked against its
provision).

Subcommands, in the order they run:

``resolve``
    For every selected case, resolve each candidate citation from the case's
    own signed corpus release, with the resolver the build uses. Candidates
    are the module-level citations of the pre-fix and post-fix module and the
    proof-atom citations of the rules in ``locator.rule_names`` on both
    sides. Writes ``<work>/candidates.jsonl`` (one row per case and
    citation) and ``<work>/citations/<text sha256>.txt``.

``bundle``
    Write one reader bundle per case under ``<work>/bundles/<id>/``: the
    case summary and triage record, the pre-to-post diff, the shipped
    provision and modules, and every other resolved citation as
    ``other/NN.txt``. ``CASE.md`` lists the files and, as search hints only,
    which strings from the fix occur in which other citation.

``collect``
    Check the readers' verdict files against the bundle texts (every quote
    must occur verbatim, after whitespace and case folding, in the file it
    names) and write ``triage/provision_review.json``: one settled record
    per case with each reader's call. A case whose readers disagree, or
    whose quote does not check, is listed in ``<work>/unsettled.json`` and
    gets no record until an adjudication file settles it.

The build (``tools/build_real_defects.py``) applies the settled records: it
writes ``judgeable_from_provision`` and ``provision_review`` into each case,
and extends the provision of a case whose decisive text sits under another
citation.

Selected cases are the board-eligible ones: shipped family representatives
with ``triage_status: fidelity`` (``--all`` selects every shipped case).

Usage (from the axiom-encode checkout)::

    uv run python benchmarks/verifier/real_defects_v0/tools/provision_review.py \\
        resolve --corpus-dir benchmarks/verifier/real_defects_v0 --work <dir> \\
        --axiom-corpus ../axiom-corpus --release-cache ~/.cache/axiom-real-defects
    uv run python .../provision_review.py bundle \\
        --corpus-dir benchmarks/verifier/real_defects_v0 --work <dir>
    uv run python .../provision_review.py collect \\
        --corpus-dir benchmarks/verifier/real_defects_v0 --work <dir> \\
        --verdicts <dir>/verdicts --reviewed-on 2026-10-10
"""

from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[3]


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lib = _load("verify_real_defects", ROOT / "scripts" / "verify_real_defects.py")
evidence = _load("check_evidence", TOOLS / "check_evidence.py")

METHOD = "provision_review_v1"
VERDICTS = ("in_provision", "in_other_citation", "not_in_sources")
DEFECT_CALLS = ("yes", "no", "unsure")
PROVISION_FILE = "provision.txt"
MAX_INLINE_DIFF_LINES = 400
_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_cases(
    corpus_dir: Path, *, every: bool, also: tuple[str, ...] = ()
) -> list[tuple[Path, dict]]:
    """Shipped cases to review: board-eligible ones, ``also`` ids, or all."""

    chosen = []
    for path in sorted((corpus_dir / "cases").glob("*/case.json")):
        case = read_json(path)
        if not case.get("artifacts_shipped", True):
            continue
        eligible = (
            case.get("family_representative")
            and case.get("triage_status") == "fidelity"
        )
        if every or eligible or case["id"] in also:
            chosen.append((path.parent, case))
    return chosen


def module_citations(text: str) -> tuple[list[str], dict[str, list[str]]]:
    """A module's own citations, and the proof-atom citations of each rule."""

    try:
        doc = yaml.load(text, Loader=_LOADER)  # noqa: S506 (safe loader)
    except yaml.YAMLError:
        return [], {}
    if not isinstance(doc, dict):
        return [], {}
    verification = (doc.get("module") or {}).get("source_verification") or {}
    paths: list[str] = []
    single = verification.get("corpus_citation_path")
    if isinstance(single, str):
        paths.append(single)
    plural = verification.get("corpus_citation_paths")
    if isinstance(plural, list):
        paths.extend(p for p in plural if isinstance(p, str))
    by_rule: dict[str, list[str]] = {}
    for rule in doc.get("rules") or []:
        if not isinstance(rule, dict):
            continue
        cites: list[str] = []
        for atom in evidence.proof_atoms(rule):
            source = atom.get("source")
            path = (
                source.get("corpus_citation_path") if isinstance(source, dict) else None
            )
            if isinstance(path, str) and path not in cites:
                cites.append(path)
        by_rule[str(rule.get("name"))] = cites
    return paths, by_rule


def candidate_citations(case_dir: Path, case: dict[str, Any]) -> dict[str, list[str]]:
    """``{citation_path: roles}`` in first-seen order.

    Roles say who cites it: ``module_post`` and ``module_pre`` (the module's
    ``source_verification``), ``rule_post`` and ``rule_pre`` (a proof atom of
    a rule in ``locator.rule_names``).
    """

    pre_paths, pre_rules = module_citations(
        (case_dir / "pre_fix.yaml").read_text(encoding="utf-8")
    )
    post_paths, post_rules = module_citations(
        (case_dir / "post_fix.yaml").read_text(encoding="utf-8")
    )
    found: dict[str, set[str]] = {}
    for path in post_paths:
        found.setdefault(path, set()).add("module_post")
    for path in pre_paths:
        found.setdefault(path, set()).add("module_pre")
    for name in (case.get("locator") or {}).get("rule_names") or []:
        for path in post_rules.get(name, []):
            found.setdefault(path, set()).add("rule_post")
        for path in pre_rules.get(name, []):
            found.setdefault(path, set()).add("rule_pre")
    return {path: sorted(roles) for path, roles in found.items()}


# --------------------------------------------------------------------------
# resolve
# --------------------------------------------------------------------------


def load_candidates(work: Path) -> dict[str, list[dict[str, Any]]]:
    """Rows of ``candidates.jsonl`` by case id, in file order."""

    by_case: dict[str, list[dict[str, Any]]] = {}
    path = work / "candidates.jsonl"
    if not path.exists():
        return by_case
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_case.setdefault(row["case_id"], []).append(row)
    return by_case


def cmd_resolve(args: argparse.Namespace) -> int:
    work: Path = args.work
    (work / "citations").mkdir(parents=True, exist_ok=True)
    done = {
        (case_id, row["citation"])
        for case_id, rows in load_candidates(work).items()
        for row in rows
    }
    cases = selected_cases(args.corpus_dir, every=args.all, also=tuple(args.also or ()))
    resolved: dict[tuple[str, str], dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    started = time.time()
    with (work / "candidates.jsonl").open("a", encoding="utf-8") as out:
        for number, (case_dir, case) in enumerate(cases, start=1):
            release = case["corpus_release"]
            for citation, roles in candidate_citations(case_dir, case).items():
                if (case["id"], citation) in done:
                    continue
                key = (release, citation)
                if key not in resolved:
                    try:
                        if release not in payloads:
                            payloads[release] = lib.fetch_release_object(
                                release,
                                case["corpus_release_content_sha256"],
                                args.release_cache,
                            )
                        result = lib.resolve_provision(
                            payloads[release],
                            args.axiom_corpus,
                            args.roots_dir,
                            citation,
                        )
                        text = result.pop("text")
                        digest = lib.sha256_text(text)
                        target = work / "citations" / f"{digest}.txt"
                        if not target.exists():
                            target.write_text(text, encoding="utf-8")
                        resolved[key] = {
                            "ok": True,
                            "text_sha256": digest,
                            "chars": len(text),
                            "resolution": result,
                        }
                    except Exception as exc:  # noqa: BLE001
                        root = args.roots_dir / release
                        resolved[key] = {
                            "ok": False,
                            "error": f"{type(exc).__name__}: "
                            f"{lib.portable_error(exc, root)}",
                        }
                row = {
                    "case_id": case["id"],
                    "citation": citation,
                    "roles": roles,
                    "is_first": citation == case["corpus_citation_path"],
                    "release": release,
                    **resolved[key],
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
            print(
                f"[{number}/{len(cases)}] {case['id']} t={time.time() - started:.0f}s",
                flush=True,
            )
    return 0


# --------------------------------------------------------------------------
# bundle
# --------------------------------------------------------------------------


def window_note(chars: int, limit: int) -> str:
    if chars <= limit:
        return f"{chars:,} characters; a judge sees all of it."
    return (
        f"{chars:,} characters. A judge sees a {limit:,}-character window "
        "(the head and the tail); text in the middle is cut. Quote the "
        "decisive passage wherever it is; the window is checked separately."
    )


def hint_strings(case: dict, pre: str, post: str, text: str) -> list[str]:
    """Post-side strings from the fix that occur in ``text`` (search hints)."""

    _status, check = evidence.check_case(case, pre, post, text)
    return [
        entry["text"][:120]
        for entry in check["strings"]
        if entry["matched"] and entry["side"] == "post"
    ]


def write_bundle(
    case_dir: Path,
    case: dict[str, Any],
    rows: list[dict[str, Any]],
    work: Path,
    provision_limit: int,
) -> dict[str, Any]:
    bundle = work / "bundles" / case["id"]
    (bundle / "other").mkdir(parents=True, exist_ok=True)
    texts = {
        name: (case_dir / name).read_text(encoding="utf-8")
        for name in ("pre_fix.yaml", "post_fix.yaml", PROVISION_FILE)
    }
    for name, text in texts.items():
        (bundle / name).write_text(text, encoding="utf-8")
    diff = list(
        difflib.unified_diff(
            texts["pre_fix.yaml"].splitlines(),
            texts["post_fix.yaml"].splitlines(),
            "pre_fix.yaml",
            "post_fix.yaml",
            n=4,
            lineterm="",
        )
    )
    (bundle / "diff.patch").write_text("\n".join(diff) + "\n", encoding="utf-8")
    others: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    seen_text = {lib.sha256_text(texts[PROVISION_FILE])}
    for row in rows:
        if row["is_first"]:
            continue
        if not row["ok"]:
            unresolved.append({"citation": row["citation"], "error": row["error"]})
            continue
        if row["text_sha256"] in seen_text:
            continue
        seen_text.add(row["text_sha256"])
        text = (work / "citations" / f"{row['text_sha256']}.txt").read_text(
            encoding="utf-8"
        )
        name = f"other/{len(others) + 1:03d}.txt"
        (bundle / name).write_text(text, encoding="utf-8")
        others.append(
            {
                "file": name,
                "citation": row["citation"],
                "roles": row["roles"],
                "chars": row["chars"],
                "text_sha256": row["text_sha256"],
                "hints": hint_strings(
                    case, texts["pre_fix.yaml"], texts["post_fix.yaml"], text
                ),
            }
        )
    triage = case["triage"]
    locator = case["locator"]
    lines = [
        f"# {case['id']}",
        "",
        f"- module: `{case['module_path']}` in {case['repo']}",
        f"- fix commit: `{case['commit'][:10]}` — {case['commit_subject']}",
        f"- recorded defect kind: `{case['defect_kind']}`",
        f"- rules the triage says changed meaning: {', '.join(f'`{n}`' for n in locator.get('rule_names') or []) or '(none named)'}",
        f"- decisive change (triage): `{locator.get('rule_path')}`",
        f"- changed lines: pre-fix {locator.get('pre_fix_lines')}, post-fix {locator.get('post_fix_lines')}",
        "",
        "## What the triage says was wrong",
        "",
        triage.get("pre_fix_wrong_because") or "(nothing recorded)",
        "",
        "## Triage notes",
        "",
        triage.get("triage_notes") or "(none)",
        "",
        "## Verifier justification",
        "",
        triage.get("verifier_justification") or "(none)",
        "",
        "## Files in this bundle",
        "",
        f"- `provision.txt` — the packaged provision, resolved from the module's first citation `{case['corpus_citation_path']}`. {window_note(len(texts[PROVISION_FILE]), provision_limit)}",
        f"- `pre_fix.yaml` ({len(texts['pre_fix.yaml'].splitlines())} lines), `post_fix.yaml` ({len(texts['post_fix.yaml'].splitlines())} lines) — the module before and after the fix",
        f"- `diff.patch` ({len(diff)} lines) — pre to post",
    ]
    if others:
        lines += [
            "",
            "Other citations the module or its changed rules name, resolved from "
            "the same signed release. `cited by` says who names the citation: the "
            "module header (`module_pre`, `module_post`) or a proof atom of a "
            "changed rule (`rule_pre`, `rule_post`). The hints are strings from "
            "the fix that occur in the file; they are search aids, not findings.",
            "",
        ]
        for other in others:
            hint = (
                "; hints: "
                + " | ".join(
                    json.dumps(h, ensure_ascii=False) for h in other["hints"][:4]
                )
                if other["hints"]
                else ""
            )
            lines.append(
                f"- `{other['file']}` — `{other['citation']}` ({other['chars']:,} chars; "
                f"cited by {', '.join(other['roles'])}){hint}"
            )
    else:
        lines += ["", "No other citation resolved for this case."]
    if unresolved:
        lines += ["", "Citations that did not resolve in the case's release:", ""]
        lines += [f"- `{u['citation']}` — {u['error'][:160]}" for u in unresolved]
    lines += ["", "## Diff (pre to post)", "", "```diff"]
    if len(diff) > MAX_INLINE_DIFF_LINES:
        lines += diff[:MAX_INLINE_DIFF_LINES]
        lines.append(
            f"... ({len(diff) - MAX_INLINE_DIFF_LINES} more lines in diff.patch)"
        )
    else:
        lines += diff
    lines += ["```", ""]
    (bundle / "CASE.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "case_id": case["id"],
        "provision_sha256": case["provision_sha256"],
        "provision_chars": len(texts[PROVISION_FILE]),
        "diff_lines": len(diff),
        "others": others,
        "unresolved": unresolved,
    }
    (bundle / "bundle.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def cmd_bundle(args: argparse.Namespace) -> int:
    candidates = load_candidates(args.work)
    manifests = []
    for case_dir, case in selected_cases(
        args.corpus_dir, every=args.all, also=tuple(args.also or ())
    ):
        if args.only and case["id"] not in args.only:
            continue
        rows = candidates.get(case["id"])
        if rows is None:
            print(f"skip {case['id']}: no candidates resolved yet", flush=True)
            continue
        manifests.append(
            write_bundle(case_dir, case, rows, args.work, args.provision_chars)
        )
    (args.work / "bundles.json").write_text(
        json.dumps(manifests, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(manifests)} bundles under {args.work / 'bundles'}")
    return 0


# --------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------


def check_reader_record(
    record: dict[str, Any], bundle: Path, manifest: dict[str, Any]
) -> list[str]:
    """Problems with one reader's record; empty when it is usable."""

    problems: list[str] = []
    verdict = record.get("verdict")
    if verdict not in VERDICTS:
        problems.append(f"verdict {verdict!r} is not one of {VERDICTS}")
    if record.get("defect_real") not in DEFECT_CALLS:
        problems.append(f"defect_real {record.get('defect_real')!r} is not valid")
    quotes = record.get("decisive_quotes") or []
    files = {other["file"] for other in manifest["others"]} | {PROVISION_FILE}
    used: set[str] = set()
    for item in quotes:
        name, quote = item.get("file"), item.get("quote") or ""
        if name not in files:
            problems.append(f"quote names unknown file {name!r}")
            continue
        if len(quote.split()) < 3:
            problems.append(f"quote from {name} is under three words")
            continue
        if lib.locate_quote(quote, (bundle / name).read_text(encoding="utf-8")) is None:
            problems.append(f"quote not found in {name}: {quote[:80]!r}")
        used.add(name)
    if verdict == "in_provision" and used != {PROVISION_FILE}:
        problems.append("in_provision needs quotes from provision.txt only")
    if verdict == "in_other_citation" and not (used - {PROVISION_FILE}):
        problems.append("in_other_citation needs a quote from an other/ file")
    if verdict == "not_in_sources" and not (record.get("missing_basis") or "").strip():
        problems.append("not_in_sources needs missing_basis")
    if verdict in VERDICTS[:2] and record.get("defect_real") != "yes":
        problems.append("a decisive text was quoted, yet defect_real is not yes")
    return problems


def load_reader_records(verdicts_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """``{case_id: {reader: record}}`` from ``<verdicts>/<reader>/*.json``.

    Each file holds one record or a list of records; a later file for the
    same reader and case replaces an earlier one (a re-ask).
    """

    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(verdicts_dir.glob("*/*.json")):
        payload = read_json(path)
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            if isinstance(record, dict) and record.get("case_id"):
                by_case.setdefault(record["case_id"], {})[path.parent.name] = record
    return by_case


def settle(
    case_id: str,
    readers: dict[str, dict[str, Any]],
    problems: dict[str, list[str]],
    adjudication: dict[str, Any] | None,
    min_readers: int = 2,
) -> tuple[dict[str, Any] | None, str | None]:
    """The settled record for a case, or why it is unsettled.

    Without an adjudication, a case settles only when at least
    ``min_readers`` readers returned a record, every record checks, and they
    agree on the verdict (and, for ``in_other_citation``, on which other
    citations carry the text). Anything else needs an adjudication, which
    then settles the case alone.
    """

    if adjudication is not None:
        chosen = adjudication
        basis = "adjudicated"
    else:
        flawed = sorted(name for name, found in problems.items() if found)
        if flawed:
            return None, f"record does not check: {', '.join(flawed)}"
        if len(readers) < min_readers:
            return None, f"{len(readers)} reader record(s), {min_readers} needed"
        verdicts = {rec["verdict"] for rec in readers.values()}
        if len(verdicts) > 1:
            return None, f"readers disagree: {sorted(verdicts)}"
        if "in_other_citation" in verdicts:
            named = {
                tuple(
                    sorted(
                        {
                            q["file"]
                            for q in rec["decisive_quotes"]
                            if q["file"] != PROVISION_FILE
                        }
                    )
                )
                for rec in readers.values()
            }
            if len(named) > 1:
                return None, "readers name different other citations"
        chosen = readers[sorted(readers)[0]]
        basis = "readers_agree" if len(readers) > 1 else "single_reader"
    return (
        {
            "verdict": chosen["verdict"],
            "defect_real": chosen.get("defect_real"),
            "decisive_quotes": chosen.get("decisive_quotes") or [],
            "why": chosen.get("why_decisive") or chosen.get("why") or "",
            "missing_basis": chosen.get("missing_basis") or "",
            "pre_fix_restates_it": bool(chosen.get("pre_fix_restates_it")),
            "basis": basis,
        },
        None,
    )


def cmd_collect(args: argparse.Namespace) -> int:
    corpus_dir: Path = args.corpus_dir
    work: Path = args.work
    by_case = load_reader_records(args.verdicts)
    adjudications: dict[str, Any] = {}
    if args.adjudications and args.adjudications.exists():
        for record in read_json(args.adjudications):
            adjudications[record["case_id"]] = record
    settled: dict[str, Any] = {}
    unsettled: dict[str, Any] = {}
    for case_dir, case in selected_cases(
        corpus_dir, every=args.all, also=tuple(args.also or ())
    ):
        case_id = case["id"]
        bundle = work / "bundles" / case_id
        if not (bundle / "bundle.json").exists():
            unsettled[case_id] = {"why": "no bundle"}
            continue
        manifest = read_json(bundle / "bundle.json")
        file_citation = {o["file"]: o["citation"] for o in manifest["others"]}
        file_citation[PROVISION_FILE] = case["corpus_citation_path"]
        readers = by_case.get(case_id, {})
        problems = {
            name: check_reader_record(rec, bundle, manifest)
            for name, rec in readers.items()
        }
        adjudication = adjudications.get(case_id)
        if adjudication is not None:
            bad = check_reader_record(adjudication, bundle, manifest)
            if bad:
                unsettled[case_id] = {"why": f"adjudication does not check: {bad}"}
                continue
        record, why = settle(case_id, readers, problems, adjudication, args.min_readers)
        calls = {
            name: {
                "verdict": rec.get("verdict"),
                "defect_real": rec.get("defect_real"),
                "confidence": rec.get("confidence"),
                "problems": problems[name],
            }
            for name, rec in sorted(readers.items())
        }
        if record is None:
            unsettled[case_id] = {"why": why, "readers": calls}
            continue
        quotes = [
            {
                "citation_path": file_citation[q["file"]],
                "quote": " ".join(q["quote"].split()),
            }
            for q in record["decisive_quotes"]
        ]
        # A not_in_sources reader may quote the nearest passage; it decides
        # nothing, so it is kept apart from the decisive quotes.
        decisive = record["verdict"] != "not_in_sources"
        record["decisive_quotes"] = quotes if decisive else []
        record["nearest_quotes"] = [] if decisive else quotes
        record["readers"] = calls
        if adjudication is not None:
            record["adjudication_note"] = adjudication.get("adjudication_note") or ""
        settled[case_id] = record
    payload = {
        "method": METHOD,
        "reviewed_on": args.reviewed_on,
        "readers": read_json(args.readers) if args.readers else {},
        "about": (
            "One settled record per reviewed case. verdict says where the text "
            "that shows the pre-fix module wrong sits: in the packaged "
            "provision (first citation), under another citation the module or "
            "its changed rules name, or in none of them. Every quote was "
            "checked to occur in the text it names."
        ),
        "counts": {
            verdict: sum(1 for r in settled.values() if r["verdict"] == verdict)
            for verdict in VERDICTS
        },
        "cases": dict(sorted(settled.items())),
    }
    out = args.out or corpus_dir / "triage" / "provision_review.json"
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", "utf-8")
    report = work / "unsettled.json"
    report.write_text(
        json.dumps(dict(sorted(unsettled.items())), indent=1, ensure_ascii=False)
        + "\n",
        "utf-8",
    )
    print(
        json.dumps(
            {
                "settled": len(settled),
                "unsettled": len(unsettled),
                "by_verdict": payload["counts"],
                "unsettled_report": str(report),
            }
        )
    )
    return 0 if not unsettled else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--corpus-dir", type=Path, required=True)
        p.add_argument("--work", type=Path, required=True)
        p.add_argument(
            "--all",
            action="store_true",
            help="every shipped case, not just board-eligible",
        )
        p.add_argument("--also", nargs="*", help="further case ids to select")

    p = sub.add_parser("resolve", help="resolve every candidate citation")
    common(p)
    p.add_argument("--axiom-corpus", type=Path, required=True)
    p.add_argument("--release-cache", type=Path, required=True)
    p.add_argument("--roots-dir", type=Path)
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("bundle", help="write one reader bundle per case")
    common(p)
    p.add_argument("--provision-chars", type=int, default=24_000)
    p.add_argument("--only", nargs="*", help="case ids to (re)write")
    p.set_defaults(func=cmd_bundle)

    p = sub.add_parser("collect", help="check reader verdicts and settle each case")
    common(p)
    p.add_argument("--verdicts", type=Path, required=True)
    p.add_argument(
        "--adjudications",
        type=Path,
        help="JSON list of adjudicator records (the reader schema, per case)",
    )
    p.add_argument(
        "--readers",
        type=Path,
        help="JSON object describing each reader (model, lane, brief), "
        "copied into the record",
    )
    p.add_argument("--min-readers", type=int, default=2)
    p.add_argument("--reviewed-on", required=True)
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_collect)

    args = parser.parse_args(argv)
    if getattr(args, "roots_dir", None) is None and args.command == "resolve":
        args.roots_dir = args.release_cache / "roots"
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
