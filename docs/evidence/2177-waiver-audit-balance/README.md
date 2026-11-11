# PR2177 waiver audit balance evidence

These files back the diagnosis and the scheduling numbers in PR #2177. They come from three sources: GitHub's records of two rulespec-us runs, local timings of one module at three encoders, and a list-scheduling simulator that replays the measured per-module costs. The author produced them; they are not signed attestations. `manifest.json` gives each file's SHA-256 and size, and for each `.gz` file the SHA-256 of its decompressed bytes. The gzip files carry no name or timestamp (`gzip -n`).

## What the files show

| File | What it is |
|---|---|
| `us-mo-40055102080-ratchet-excerpt.log` | 31 lines of the us-mo leg of rulespec-us run 40055102080 (main, 2026-10-24, encoder `5ddb5ea4`), with GitHub's job and step columns removed. The ratchet line prints at 03:22:58, 25 `Bound ...` lines follow between 03:23:05 and 03:28:00, and the next line, the audit report, prints at 04:55:34: about 88 minutes of silence. |
| `jobs-41819980907.json.gz` | The jobs listing of rulespec-us run 41819980907 (rulespec-us#2023 at head `6fd18c0b`, 2026-11-04, encoder `5af06cddf`), as `gh api --paginate repos/TheAxiomFoundation/rulespec-us/actions/runs/41819980907/jobs?per_page=100&filter=latest` printed it: one JSON page after another. |
| `module-timings-41819980907.json.gz` | Per-leg lists of `[path, seconds]` from that run's `validation-waivers audit: <path> <seconds>s` lines (#2010), parsed with this PR's `timing_observations`: 1,930 modules across 52 legs, one observation each. |
| `validation-waiver-audit-costs-41819980907.json.gz` | The cost table that this PR's `axiom-encode validation-waivers audit-costs` writes from the same run's 52 leg logs and jobs listing: 1,930 module costs and 52 partition baselines, SHA-256 `58e6c34b...6456`. |
| `sizes-6fd18c0b.tsv.gz` | `git ls-tree -r -l 6fd18c0b` over rulespec-us, as `<bytes><TAB><path>`. Only the simulator's byte-proxy policy (D) reads it. It lacks the four `us-nh/regulations/he-w-700/He-W 7xx/` paths, whose names contain spaces, so D prices those four at zero bytes. |
| `leg_durations.py.txt`, `simulate.py.txt`, `replay_encoder_assignment.py.txt` | The scripts, stored as text. `simulate.py` list-schedules each leg on 4 workers and compares policies A to E. `replay_encoder_assignment.py` feeds the cost table to this PR's `assign_waiver_owners` and `audit_order` and runs the result through the same model. |
| `simulate-41819980907.txt`, `replay-encoder-assignment-41819980907.txt` | Their outputs. |
| `note20-local-encode-2bdc3b42.json`, `note20-local-encode-4df4b0cd9p.json`, `note20-local-encode-4df4b0cd9.json` | Local audits of `us/policies/usitc/us-tariff-incidence/generated/note20-china-301.yaml` alone, at the rulespec-us pin `2bdc3b42` (0.2.2248), at #1679's parent `e6fbecba1` (0.2.2275), and at #1679 `4df4b0cd9` (0.2.2276). Each records elapsed time, subprocess launches, `_read_corpus_artifact` calls and bytes, and sampled stack frames. |

## What they establish

- **The silent phase is corpus re-reading.** At the pin and at #1679's parent, the module had not finished after about 1,760 s. By then it had made about 59,500 `_read_corpus_artifact` calls and read about 102 GB, and had launched the engine twice. At #1679 it finished in 783 s with 23 reads. The two unfinished runs died when the parent CLI restarted, so 1,760 s is a lower bound on their time. All three ran at once on a host at load 45 to 65 against debug engine `89571cc2` and corpus release `us-rulespec-2026-10-29-ecfr-section-text`, so the absolute times run long; CI took 606 s for the same module at `5af06cddf`.
- **The cache changes no outcome.** The #1679 run's failure fingerprint, `sha256:2c983175...d4de`, equals the active fingerprint that rulespec-us's `known-validation-gaps.yaml` records for this module. `passed: false` is expected: the module carries a validation waiver because it fails validation, and the audit checks that the failure still matches.
- **The simulator reproduces the observed legs.** Under policy A (today's round-robin), its per-leg ratchet times match all 52 observed ratchet steps within 0.2 min (median 0.1 min). It gives a maximum of 58.4 min against the observed 58.3.
- **Policy E reaches the lower bound.** E gives a ratchet maximum of 49.5 min, the largest module (`7-cfr/273/4`, 48.9 min) plus 0.6 min of setup, and a job maximum of 77.6 min, against A's 58.4 and 111.9.
- **The encoder's own code gives policy E.** Run through the replay, this PR's assignment and start order give 49.5 and 77.6. Without the table, the same function deals the table's 1,930 paths exactly as the run did and reproduces A's 58.4 and 111.9.

## How to reproduce

From this folder, with a checkout of this PR at `<checkout>` whose environment is synced (`uv sync --extra dev`):

```bash
mkdir -p /tmp/ev/scripts /tmp/ev/data
for f in leg_durations simulate replay_encoder_assignment; do cp "$f.py.txt" "/tmp/ev/scripts/$f.py"; done
gzip -dc module-timings-41819980907.json.gz > /tmp/ev/data/module_timings_41819980907.json
gzip -dc jobs-41819980907.json.gz > /tmp/ev/data/jobs_41819980907.json
gzip -dc sizes-6fd18c0b.tsv.gz > /tmp/ev/data/sizes_6fd18c0b.tsv
gzip -dc validation-waiver-audit-costs-41819980907.json.gz > /tmp/ev/costs.json
python3 -I /tmp/ev/scripts/simulate.py | diff - simulate-41819980907.txt
PYTHONPATH=<checkout>/src <checkout>/.venv/bin/python /tmp/ev/scripts/replay_encoder_assignment.py /tmp/ev/costs.json | diff - replay-encoder-assignment-41819980907.txt
```

Both diffs print nothing. The scripts read their inputs from `../data` relative to themselves, so the copies need that layout. `leg_durations.py` calls `gh api` only when its cached jobs file is missing.

To rebuild the cost table, download each leg's log of run 41819980907 with `gh run view 41819980907 --repo TheAxiomFoundation/rulespec-us --job <job id> --log` (the 52 logs total 68 MB and are not included), then run:

```bash
axiom-encode validation-waivers audit-costs --log <leg 1 log> ... --log <leg 52 log> \
  --jobs-json /tmp/ev/data/jobs_41819980907.json \
  --source https://github.com/TheAxiomFoundation/rulespec-us/actions/runs/41819980907 \
  --output costs.json
```

At this PR's head the output equals the included table byte for byte (SHA-256 `58e6c34b...6456`).

## Limitations

- The simulator models a leg as a non-audit part plus 4-worker list scheduling with a fixed 0.6 min setup. It does not model runner variance, memory pressure or queueing for runners. It evaluates every policy with the costs measured on 2026-11-04; a later run's costs will differ.
- Policies B to E use the measured costs as an oracle. In production the table comes from an earlier run, so a module whose cost has changed since then lands where the stale cost puts it. The exactly-once guarantee does not depend on the table.
- The note20 timings come from one local host under load, not CI. They support the mechanism (re-reads, not engine time) and the relative slowdown, not CI durations.
- The 10-24 log excerpt shows the silent phase at encoder `5ddb5ea4`. The local measurements ran at the pin `2bdc3b42`, a descendant of `5ddb5ea4`, and at #1679's parent, and the two show the same profile. No run measured `5ddb5ea4` itself.
- The first download of the us-ma leg's log returned a `gh` error (`zip: not a valid zip file`) instead of the log. A first pass of this analysis, quoted in an earlier version of the PR description, ran without that leg's 37 modules: 1,889 timings, a 1,893-entry table and a 20.5 min simulator error on us-ma. A retry on 2026-11-11 downloaded the log, and every file here includes it. The conclusions did not change.
