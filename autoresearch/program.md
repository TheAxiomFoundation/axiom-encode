# AutoRAC Harness Prompt Tuning

## Goal

Improve `autorac`'s UK source-slice encoding behavior by editing a single prompt
surface and keeping only changes that improve the frozen benchmark score.

## Editable surface

You may edit only:

- `src/autorac/harness/eval_prompt_surface.py`

Do not edit:

- any `rac-*` repository
- Atlas/site repositories
- promotion or sync scripts
- deterministic validators in ways that weaken the gates
- benchmark manifests

## Frozen benchmarks

Run all of:

- `benchmarks/uk_wave18_remaining_repair.yaml`
- `benchmarks/uk_wave19_failure_repair.yaml`
- `benchmarks/uk_wave19_branch_conjunction_repair.yaml`

These are repair slices for known UK failure patterns:

- binding lead-in conjuncts
- conditional amount/exclusion leaves
- branch conjunction vs material implication

## Runner

Use:

```bash
uv run python scripts/run_autoresearch_pilot.py --gpt-backend codex
```

The runner writes per-manifest eval outputs plus a single aggregate report and
prints `AUTORESEARCH_SCORE=...` at the end.

## Score

Optimize the aggregate scalar score. It is:

- `100` if a runner is fully ready, otherwise `0`
- minus compile failures
- minus CI failures
- minus generalist-review failures
- minus zero-ungrounded failures
- minus a small cost tiebreaker

The intent is to optimize for statutory fidelity and deterministic correctness
first, cost second.

## Keep / discard rule

- Keep a change only if the aggregate score improves or a tie is broken by
  clearer prompt wording with no regressions.
- Discard any change that lowers readiness, weakens checks, or increases
  semantic regressions even if it improves a single case.

## Notes

- This is a prompt-tuning loop, not a corpus-editing loop.
- Generated `.rac` artifacts are evidence, not code to commit.
- If a failure appears to require a deterministic validator or parser fix rather
  than prompt wording, stop and hand control back to the main engineering loop.
