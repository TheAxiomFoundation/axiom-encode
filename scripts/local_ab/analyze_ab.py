"""Analyze a local encoder A/B run written by run_ab.py.

Usage:
    python3 -I analyze_ab.py <run dir> --baseline main --treatment branch \\
        [--classes citations.tsv]

Writes ``report.md`` and ``results.json`` into the run dir and prints the
report. The unit is a group (citation, rep). Pre-registered statistics:

- primary: final pass (``outcome.final_success`` in each run's encodings.db),
  paired exact McNemar on groups where no arm timed out;
- secondary: the same with timeouts counted as fails; first-attempt pass;
  a per-citation sign test; a Wald and a citation-clustered bootstrap 95%
  interval for the paired difference; per-rule validator issues on attempt 1,
  all attempts and the final attempt (from failed_attempts.jsonl); the primary
  blocker class of each failed run; tries, wall time and timeouts, with the
  encode phase where each timed-out run stopped.

``--classes`` is an optional TSV with ``citation`` and ``class`` columns for a
per-source-class split.
"""

import argparse
import collections
import glob
import json
import math
import os
import random
import re
import sqlite3

RULE = re.compile(r"\[([a-z][a-z0-9_.-]*(?::[a-z0-9_.-]+)+)\]")
PHASE = re.compile(r"try=(\d) phase=([a-z_]+)")
# Same patterns as run_ab.py; applied post hoc to codex `error=` lines.
INFRA = re.compile(
    r"nodename nor servname|failed to lookup address|Network is unreachable"
    r"|waiting for network|Connection refused|error sending request"
    r"|rate.?limit|usage limit|Too Many Requests|\b429\b|quota",
    re.I,
)


def classify(issue: str) -> list[str]:
    found = RULE.findall(issue)
    if found:
        return sorted(set(found))
    prefixes = [
        ("Axiom rules engine compile failed", "engine-compile"),
        ("Source numeric value", "numeric-recall-detail"),
        ("Proof", "proof-*"),
        ("Retained repair", "repair-candidate"),
        ("Repair candidate", "repair-candidate"),
        ("Ungrounded generated numeric", "ungrounded-literal"),
        ("Empty RuleSpec module", "empty-module"),
        ("Embedded scalar literal", "embedded-scalar-literal"),
        ("Test input assignment missing", "test-input-missing"),
        ("Source relation target invalid", "source-relation-target"),
    ]
    for prefix, name in prefixes:
        if issue.startswith(prefix):
            return [name]
    if "YAML parse failed" in issue:
        return ["yaml-parse"]
    if issue.startswith("Test case") or (
        "test" in issue.lower() and "fail" in issue.lower()
    ):
        return ["test-case-failure"]
    return ["other"]


def primary_blocker(rules: dict) -> str:
    if "complete-source-unit:deferral" in rules:
        return "deferral"
    if "complete-source-unit:formula-output" in rules:
        return "formula-output"
    csu = sorted(k for k in rules if k.startswith("complete-source-unit"))
    if csu:
        return "other-csu:" + ",".join(k.split(":", 1)[1] for k in csu)
    return "non-csu"


def load_run(run_dir: str) -> dict:
    rec = json.load(open(f"{run_dir}/run.json"))
    rec["attempts"], rec["final_success"] = [], False
    logs_path = f"{run_dir}/failed_attempts.jsonl"
    logs = (
        [json.loads(line) for line in open(logs_path)]
        if os.path.exists(logs_path)
        else []
    )
    db = f"{run_dir}/encodings.db"
    if os.path.exists(db):
        con = sqlite3.connect(db)
        for iterations, outcome in con.execute(
            "select iterations_json, outcome_json from encoding_runs order by timestamp"
        ):
            for attempt in json.loads(iterations or "[]"):
                rec["attempts"].append({"success": bool(attempt.get("success"))})
            out = json.loads(outcome or "{}")
            rec["final_success"] = bool(out.get("final_success")) and not rec.get(
                "timeout"
            )
    failures = iter(logs)
    for attempt in rec["attempts"]:
        if attempt["success"]:
            attempt["rules"] = {}
            continue
        log = next(failures, None)
        issues = (
            (log["full_validation_issues"] or log["validation_issues"]) if log else []
        )
        attempt["rules"] = dict(
            collections.Counter(r for issue in issues for r in classify(issue))
        )
    stdout_path = f"{run_dir}/stdout.log"
    if os.path.exists(stdout_path):
        for line in open(stdout_path, errors="replace"):
            if line.strip().startswith("error=") and INFRA.search(line):
                rec["infra_posthoc"] = line.strip()[:200]
                break
    rec["first_pass"] = bool(
        rec["attempts"] and rec["attempts"][0]["success"]
    ) and not rec.get("timeout")
    rec["tries"] = len(rec["attempts"])
    if rec.get("timeout"):
        phases = (
            PHASE.findall(open(f"{run_dir}/stdout.log", errors="replace").read())
            if os.path.exists(f"{run_dir}/stdout.log")
            else []
        )
        rec["timeout_phase"] = (
            f"try={phases[-1][0]} {phases[-1][1]}" if phases else "unknown"
        )
    return rec


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    pk = [math.comb(n, i) / 2**n for i in range(n + 1)]
    return min(1.0, sum(p for p in pk if p <= pk[k] + 1e-12))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir")
    parser.add_argument("--baseline", default="main")
    parser.add_argument("--treatment", default="branch")
    parser.add_argument("--classes")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print only; do not write report.md/results.json",
    )
    args = parser.parse_args()
    D, A, B = args.run_dir, args.baseline, args.treatment

    cls = {}
    if args.classes:
        for line in open(args.classes).read().splitlines()[1:]:
            fields = line.split("\t")
            cls[fields[0]] = fields[1]

    runs = {}
    for rj in glob.glob(f"{D}/runs/*/r*/*/run.json"):
        if ".discarded-" in rj:
            continue  # an earlier run kept by run_one.sh, not part of the result
        rec = load_run(os.path.dirname(rj))
        # Skip discarded groups and runs the supervisor has not finished
        # (no run_index yet: in progress, or orphaned by a crash).
        if rec.get("infra_error") or "run_index" not in rec:
            continue
        runs[(rec["citation"], rec["run_index"], rec["side"])] = rec
    groups = sorted(
        {(c, i) for (c, i, s) in runs if (c, i, A) in runs and (c, i, B) in runs},
        key=lambda g: (g[1], g[0]),
    )
    posthoc_infra = [
        g
        for g in groups
        if runs[(*g, A)].get("infra_posthoc") or runs[(*g, B)].get("infra_posthoc")
    ]
    groups = [g for g in groups if g not in posthoc_infra]
    no_timeout = [
        g
        for g in groups
        if not runs[(*g, A)].get("timeout") and not runs[(*g, B)].get("timeout")
    ]
    out = []
    P = out.append

    def paired(metric, sel):
        b = sum(1 for g in sel if runs[(*g, A)][metric] and not runs[(*g, B)][metric])
        c = sum(1 for g in sel if runs[(*g, B)][metric] and not runs[(*g, A)][metric])
        n = len(sel)
        pa = sum(runs[(*g, A)][metric] for g in sel) / n if n else 0
        pb = sum(runs[(*g, B)][metric] for g in sel) / n if n else 0
        diff = pb - pa
        se = math.sqrt(max(((b + c) / n - ((c - b) / n) ** 2) / n, 0)) if n else 0
        rng = random.Random(1)
        cits = sorted({g[0] for g in sel})
        by_cit = collections.defaultdict(list)
        for g in sel:
            by_cit[g[0]].append(g)
        boots = []
        for _ in range(10000):
            tot = da = db = 0
            for cit in (rng.choice(cits) for _ in cits):
                for g in by_cit[cit]:
                    tot += 1
                    da += runs[(*g, A)][metric]
                    db += runs[(*g, B)][metric]
            boots.append((db - da) / tot if tot else 0)
        boots.sort()
        return (
            f"n={n} {A}={pa:.1%} {B}={pb:.1%} diff={diff:+.1%} discordant {A}-only={b} {B}-only={c} "
            f"McNemar exact p={binom_two_sided(min(b, c), b + c):.3f} "
            f"95% Wald [{diff - 1.96 * se:+.1%}, {diff + 1.96 * se:+.1%}] "
            f"95% cluster-bootstrap [{boots[249]:+.1%}, {boots[9749]:+.1%}]"
        )

    def sign_test(metric, sel):
        per = collections.defaultdict(lambda: [0, 0])
        for g in sel:
            per[g[0]][0] += runs[(*g, A)][metric]
            per[g[0]][1] += runs[(*g, B)][metric]
        pos = sum(1 for v in per.values() if v[1] > v[0])
        neg = sum(1 for v in per.values() if v[1] < v[0])
        return f"{B} better={pos} {A} better={neg} ties={len(per) - pos - neg} p={binom_two_sided(min(pos, neg), pos + neg):.3f}"

    P(f"# Local encoder A/B: {A} vs {B}\n")
    P(
        f"groups with both arms: {len(groups)}; without any timeout: {len(no_timeout)}; "
        f"excluded for post-hoc infra errors: {len(posthoc_infra)}\n"
    )
    for g in posthoc_infra:
        arm = A if runs[(*g, A)].get("infra_posthoc") else B
        P(
            f"- post-hoc infra: {g[0]} r{g[1]} ({arm}): {runs[(*g, arm)]['infra_posthoc']}"
        )
    P(
        "\nNote: reps of one citation are clustered, so McNemar and Wald treat "
        "them as independent and are anti-conservative; the per-citation sign "
        "test and the cluster bootstrap are the cluster-valid checks. Excluding "
        "timeout groups is informative if an arm causes timeouts; compare the "
        "timeouts-as-fail row.\n"
    )
    P("## Primary: final pass, groups without timeout\n")
    P(paired("final_success", no_timeout))
    P(f"\nSign test per citation: {sign_test('final_success', no_timeout)}\n")
    P("## Final pass, timeouts counted as fail\n")
    P(paired("final_success", groups) + "\n")
    P("## First-attempt pass, groups without timeout\n")
    P(paired("first_pass", no_timeout) + "\n")
    if cls:
        P("## By source class (final pass, groups without timeout)\n")
        for name in sorted(set(cls.values())):
            sel = [g for g in no_timeout if cls.get(g[0]) == name]
            if sel:
                P(f"- {name}: {paired('final_success', sel)}")
        P("")

    reps = sorted({g[1] for g in groups})
    P("## Per-citation table\n")
    P(
        "| citation | "
        + " | ".join(f"{arm} r{i}" for arm in (A, B) for i in reps)
        + f" | {A} passes | {B} passes |"
    )
    P("|---|" + "---|" * (2 * len(reps) + 2))

    def cell(rec):
        if not rec:
            return "-"
        state = "TIMEOUT " if rec.get("timeout") else ""
        return (
            f"{state}{'P' if rec['first_pass'] else 'F'}/{'P' if rec['final_success'] else 'F'} "
            f"t{rec['tries']} {round(rec['wall_s'] / 60)}m"
        )

    for cit in sorted({g[0] for g in groups}):
        cells = [cell(runs.get((cit, i, arm))) for arm in (A, B) for i in reps]
        pa = sum(
            runs[(cit, i, A)]["final_success"] for i in reps if (cit, i, A) in runs
        )
        pb = sum(
            runs[(cit, i, B)]["final_success"] for i in reps if (cit, i, B) in runs
        )
        P(
            f"| {cit} | "
            + " | ".join(cells)
            + f" | {pa}/{len(reps)} | {pb}/{len(reps)} |"
        )
    P("\nCell = first/final pass, tries, wall minutes.\n")

    P("## Tries, time, timeouts\n")
    for arm in (A, B):
        rs = [runs[(*g, arm)] for g in groups if not runs[(*g, arm)].get("timeout")]
        passes = [r["tries"] for r in rs if r["final_success"]]
        P(
            f"- {arm}: runs={len(rs)} mean tries={sum(r['tries'] for r in rs) / max(len(rs), 1):.2f} "
            f"tries-to-pass mean={(sum(passes) / len(passes)) if passes else 0:.2f} "
            f"mean wall={sum(r['wall_s'] for r in rs) / max(len(rs), 1) / 60:.1f} min "
            f"timeouts={sum(1 for g in groups if runs[(*g, arm)].get('timeout'))} "
            "(tries and wall exclude timed-out runs)"
        )
    for (cit, i, arm), rec in sorted(runs.items()):
        if rec.get("timeout") and (cit, i) in groups:
            P(f"  - timeout: {cit} r{i} {arm}, stopped at {rec.get('timeout_phase')}")
    P("")

    for label, pick in (
        ("attempt 1", lambda r: r["attempts"][:1]),
        ("all attempts", lambda r: r["attempts"]),
        (
            "final attempt",
            lambda r: r["attempts"][-1:] if not r["final_success"] else [],
        ),
    ):
        agg = {arm: collections.Counter() for arm in (A, B)}
        for g in groups:
            for arm in (A, B):
                for attempt in pick(runs[(*g, arm)]):
                    agg[arm].update(attempt["rules"])
        keys = sorted(set(agg[A]) | set(agg[B]), key=lambda k: -(agg[A][k] + agg[B][k]))
        P(f"## Issues by rule, {label}\n")
        P(f"| rule | {A} | {B} | delta |")
        P("|---|---|---|---|")
        for key in keys:
            P(
                f"| {key} | {agg[A][key]} | {agg[B][key]} | {agg[B][key] - agg[A][key]:+d} |"
            )
        P(
            f"| TOTAL | {sum(agg[A].values())} | {sum(agg[B].values())} | "
            f"{sum(agg[B].values()) - sum(agg[A].values()):+d} |\n"
        )

    P("## Primary blocker of failed runs (final attempt)\n")
    for arm in (A, B):
        mix = collections.Counter(
            primary_blocker(runs[(*g, arm)]["attempts"][-1]["rules"])
            for g in groups
            if not runs[(*g, arm)]["final_success"] and runs[(*g, arm)]["attempts"]
        )
        P(f"- {arm}: " + ", ".join(f"{k}={v}" for k, v in mix.most_common()))
    retries = (
        [json.loads(line) for line in open(f"{D}/retries.jsonl")]
        if os.path.exists(f"{D}/retries.jsonl")
        else []
    )
    P(f"\n## Infra retries: {len(retries)}\n")
    for rec in retries:
        P(
            f"- {rec['t']} {rec['citation']} r{rec['run_index']} try{rec['infra_retry']}: "
            f"{sorted({v[0] for v in rec['errors'].values()})}"
        )
    if not args.no_write:
        open(f"{D}/report.md", "w").write("\n".join(out) + "\n")
        json.dump(
            {f"{c}|{i}|{s}": r for (c, i, s), r in runs.items()},
            open(f"{D}/results.json", "w"),
            indent=1,
        )
    print("\n".join(out))


if __name__ == "__main__":
    main()
