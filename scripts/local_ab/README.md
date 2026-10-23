# Local encoder A/B tools

These tools run paired A/B comparisons of the real encoder path on a Mac. Model calls go through the Codex CLI ChatGPT login and the `claude` CLI login, so no API key is used. The process is described in [docs/local-encoder-ab.md](../../docs/local-encoder-ab.md).

| file | purpose |
|---|---|
| `local_encode.py` | In-process `axiom-encode encode` inside `local_corpus_release_verification`. It refuses provider, R2 or Supabase keys and `--apply`. It logs every rejected attempt's issues to `$LOCAL_AB_ATTEMPT_LOG`. |
| `run_one.sh` | Runs one encode for one arm under `env -i` with the fixed flag set. It writes `run.json`. |
| `run_ab.py` | The supervisor. It starts all arms of a group together, uses a seeded shuffle per round, and runs N groups in parallel. It handles timeouts on the wall clock, detects sleep, network and rate limits, retries discarded groups, and resumes after a stop. |
| `analyze_ab.py` | Writes `report.md` and `results.json`. It computes McNemar and sign tests, the intervals, per-rule issues, the blocker mix, and the timeouts with the phase where each stopped. |
| `replay_completeness.py` | Replays the completeness analyzer over saved candidates on two checkouts, to confirm that a change is generator-only. |

```bash
export LOCAL_AB_PINS=/path/to/pins    # corpus/, engine/, rulespec-us/, 3 public keys
caffeinate -dimsu python3 -I scripts/local_ab/run_ab.py --out runs/ab1 \
  --citations citations.txt --arm main=/wt/main --arm branch=/wt/branch --reps 3 \
  --archive-dir ~/axiom-encode-local-ab/2026-10-20-ab1
python3 -I scripts/local_ab/analyze_ab.py runs/ab1 --baseline main --treatment branch
```

Runs never delete their artifacts; see "Keeping intermediate results" in the doc. The tools hold no credentials. The `*.pub` files in the pins directory are public verification keys.
