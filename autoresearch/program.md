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
- `benchmarks/uk_autoresearch_partner_disjunction.yaml`
- `benchmarks/uk_autoresearch_semantic_margin.yaml`

Treat those as the inner-loop training set.

These are repair slices for known UK failure patterns:

- binding lead-in conjuncts
- conditional amount/exclusion leaves
- branch conjunction vs material implication
- claimant-or-partner disjunctions encoded as partner substitution
- weekly amount/divisor semantics
- unnecessary entitlement guards
- distinct statutory disjuncts or benefit routes collapsed into one fact
- person-vs-family modeling of personal benefit receipt

If the current baseline report is already fully ready across that training set,
assume the prompt surface is near-optimal. In that situation, prefer a no-op
unless you can identify one concrete semantic weakness tied to those clusters.
Do not spend iterations on naming cleanup, readability tweaks, or token-count
reduction unless the benchmark evidence itself shows that naming is causing a
semantic or reviewer failure.

When the baseline report surfaces text like `the claimant or, if he has a
partner, his partner`, treat that as a true disjunction over potential
satisfiers. Do not encode it as partner-only substitution, exclusive branching,
or `if claimant_has_partner: partner_fact else: claimant_fact`.

## Final review holdout

A candidate is not accepted on training score alone.

It must also avoid regressing the separate holdout final-review set:

- `benchmarks/uk_autoresearch_final_review.yaml`

That holdout is intentionally outside the mutation brief above, so broad
wording changes that look harmless on the repair slices still have to survive a
second pass.

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
- Also require the candidate to preserve or improve the separate final-review
  holdout score.
- Target one concrete issue cluster per iteration rather than broad prompt
  cleanup.
- Discard any change that lowers readiness, weakens checks, or increases
  semantic regressions even if it improves a single case.

## Notes

- This is a prompt-tuning loop, not a corpus-editing loop.
- Generated `.rac` artifacts are evidence, not code to commit.
- If a failure appears to require a deterministic validator or parser fix rather
  than prompt wording, stop and hand control back to the main engineering loop.
