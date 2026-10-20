# Local encoder A/B runs

This document tells how to measure an encoder change with the real encoder path on a local Mac. It uses the Codex and Claude subscription logins, so there is no API spend, no GitHub Actions dispatch and no signing. The tools are in [`scripts/local_ab/`](../scripts/local_ab/README.md).

## Why

- Replays of saved candidates show only what a validator change does to old output. They cannot show what a prompt or retry change does to generation.
- A local A/B runs the complete `axiom-encode encode` loop on both arms: generation, staging, deterministic repair, validation, retry escalation and LLM review. Only the encoder checkout differs between the arms.
- Model calls go through the Codex CLI on the ChatGPT login and the `claude` CLI on the claude.ai login. No paid API key is in the environment, and the driver refuses to start if one is present.

## Prerequisites

1. **Logins.** Run `codex login status`; it must report "Logged in using ChatGPT". Check that `claude` runs with the claude.ai login. Keep `OPENAI_API_KEY` unset.
2. **Pins directory (`LOCAL_AB_PINS`).** Use one directory for every arm:
   - `rulespec-us/`: a `rulespec-us` worktree at a fixed commit.
   - `corpus/`: an `axiom-corpus` worktree at the commit that matches that `rulespec-us` pin.
   - `engine/`: an `axiom-rules-engine` worktree at its pin. Build it once with `cargo build --release --locked`.
   - The three public trust roots, copied from the repository Actions variables: `corpus_release_root.pub`, `AXIOM_ENCODE_APPLY_SIGNING_PUBLIC_KEY.pub` and `AXIOM_ENCODE_EVAL_SIGNING_PUBLIC_KEY.pub`. These are public keys, not credentials.
3. **Corpus release cache.** Fetch the signed corpus release object that the pin names into `~/.axiom/corpus-cache`. Use the public URL that the org `validate-rulespec` workflow uses; `scripts/materialize_corpus_release.py` validates it.
4. **Arm worktrees.** Use one detached worktree per arm, each with its own venv: `uv venv --python 3.13 && uv sync --all-extras`.
   - Freeze each arm at a commit before the run starts. Never run an arm from a worktree that you are still editing.
   - To test a generator change against current main, cherry-pick its commits onto that main, without version bumps. Then the arms differ only in the change.
5. **Power.** Use AC power, with the lid open or an external display attached.

## Command and environment

`run_one.sh` runs each encode as follows:

```bash
env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" TMPDIR="$out/tmp/" LANG=en_US.UTF-8 \
  PATH="$wt/.venv/bin:$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
  LOCAL_AB_ATTEMPT_LOG="$out/failed_attempts.jsonl" \
  "$wt/.venv/bin/python" -I scripts/local_ab/local_encode.py "$LOCAL_AB_PINS" -- encode "$citation" \
    --backend codex --corpus-path "$LOCAL_AB_PINS/corpus" \
    --axiom-rules-engine-path "$LOCAL_AB_PINS/engine" --policy-repo-path "$LOCAL_AB_PINS/rulespec-us" \
    --output "$out/generated" --emit-final-rejected-candidate "$out/final-rejected-candidate" \
    --mode repo-augmented --require-complete-source-unit --db "$out/encodings.db" --no-sync
```

- `env -i` removes every provider, R2 and Supabase key. `local_encode.py` refuses to run if one remains, and it refuses `--apply`.
- `--no-sync` and the local `--db` keep telemetry out of Supabase.
- Corpus trust is verification-only (`toolchain.local_corpus_release_verification`), which is the same stand-in that `axiom-encode ci` uses.
- The other settings are the production defaults: gpt-6-luna, then gpt-6-sol after two rejections, at most 4 attempts.

## Design

1. **Write the design file first.** Record the arms and their commits, the pins, the citations, the reps, the seed, the primary metric and the secondary metrics. Do not change them after the first run.
2. **Unit.** The unit is a group: one citation and one rep. All arms of a group start at the same moment, so machine load and time of day are shared.
3. **Order.** `run_ab.py` shuffles each round independently with `random.Random(seed)` and writes `order_r<N>.txt`. A stop therefore leaves whole groups.
4. **Size.** Use 3 or more reps per citation. Single runs are noisy: in pilot 2, 4 of 30 citations passed in one rep and failed in the other on the same arm.
   - One encode takes about 10–20 min. Three groups in parallel (six encodes) is the tested load.
5. **Citations.** Choose citations that fail on the rule the change targets, plus some that pass, to detect regressions. Avoid citations that stall in `artifact_repair` (see the pitfalls) unless they are the target.

## Power, sleep and timeouts

- Run the supervisor under `caffeinate -dimsu`. That stops idle sleep on AC only. On battery, a closed lid still sleeps the Mac. `run_ab.py` logs a warning when `pmset -g batt` shows battery power.
- After each group, the supervisor reads `pmset -g log`. A group with a system sleep inside its window is discarded for every arm and retried, because hung network calls make its runs unusable.
- Network errors, DNS errors and rate limits in codex output are also infra errors. A rate limit pauses the run for 30 min. After 3 retries, the group is marked `infra_gave_up`.
- The timeout (default 75 min) is measured on the wall clock and polled. `Popen.wait(timeout)` uses a monotonic clock that stops while macOS sleeps, so a hung run could outlive its timeout.
- A timeout is recorded with `"timeout": true` and is not retried. The report lists each timeout with the encode phase where it stopped.
- To stop at a group boundary, `touch <out>/STOP`. To resume, run the same command again; finished groups are skipped.

## Analysis

`analyze_ab.py <out> --baseline A --treatment B [--classes citations.tsv]` writes `report.md` and `results.json`:

- **Primary statistic.** Final pass (`outcome.final_success`) with a paired exact McNemar test on groups where no arm timed out.
- **Secondary statistics:**
  - final pass with timeouts counted as fails;
  - first-attempt pass;
  - a per-citation sign test;
  - a Wald 95% interval and a citation-clustered bootstrap 95% interval;
  - validator issues by rule on attempt 1, on all attempts and on the final attempt (from `failed_attempts.jsonl`);
  - the primary blocker class of each failed run;
  - tries, wall time and timeouts.
- **Generator-only check.** If the change must not alter validators, run `replay_completeness.py` with each checkout over the same saved candidates, then run `--compare`. A generator-only change compares identical.

## Accuracy audit

A higher pass rate is not a better encoding by itself. Read by hand every pass that only the treatment arm has, plus a sample of shared passes, against `source.txt`:

- Constant `true` (or other constant) derived outputs that only cover a branch, for example an applicability clause with no input for the population it limits.
- Inputs the source does not state, and inputs that stand for a determination of a cited provision that should be imported or deferred.
- Companion-test expectations that do not follow from the source and formula, and source amounts that appear only in tests.
- Deferrals of operative text that were accepted on a weak reason.

Report every finding, including "none found".

## Known pitfalls

- **Battery sleep.** It stalls every in-flight run. Keep the Mac on AC; the supervisor discards and retries sleep-hit groups.
- **The wait timer stops during sleep.** Use the wall-clock deadline in `run_ab.py`.
- **Slow `artifact_repair` on 42 CFR sources.** It can take 10–41 min on main (42 CFR 435.927, 457.622, 457.515), which causes timeouts in both arms. Report these runs separately; they are not caused by the arm under test.
- **Nondeterminism.** Codex output varies between runs. Use 3 or more reps and paired tests, and do not compare unpaired pass counts.
- **Pin drift.** Some rulespec-us pins hold modules that newer encoders reject, for example plural `corpus_citation_paths` in an import. Both arms see this; note it in the report.
- **Machine load.** Other agents and jobs slow both arms. `load.log` records `uptime` every 10 min.

## Reporting in a PR

Include these items:
- the arms with their exact commits, the pins and the design file;
- a table with final pass and first-attempt pass for each arm, the discordant counts and the McNemar p;
- the timeouts, listed separately with the phase where each stopped, and the infra retries;
- the per-rule final-attempt issues and the blocker mix;
- mean wall time;
- the accuracy-audit findings.

State whether the result is significant. With 40–75 groups, only large effects reach p < 0.05; say so, and do not round a null result up into a gain.
