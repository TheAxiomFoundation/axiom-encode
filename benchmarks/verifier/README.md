# EncodeBench verifier track

A benchmark of fidelity *judges*: models that read a provision and a RuleSpec
artifact and say whether the artifact is wrong. The encoder track
(`benchmarks/encodebench_uk_v1.yaml`, `axiom-encode eval-board`) scores the
models that write artifacts; this track scores the models that check them.

Runbook: [`docs/encodebench-verifier.md`](../../docs/encodebench-verifier.md).

```
verifier.py                     entry point (puts this dir and src/ on sys.path)
pricing.json                    published per-token prices, each with a source
encodebench_verifier/
  mutator.py                    versioned, seeded single-edit mutator (MUTATOR_VERSION)
  cases.py                      case / suite model and the suite identity digest
  sources/encodings_db.py       known-good artifacts from encodings.db (read-only)
  sources/eval_suite.py         known-good artifacts from encoder-track gate passes
  sources/real.py               loader for recorded real defects (real_defects_v0)
  sources/synthetic.py          builds a synthetic suite from known-good artifacts
  judges/referee.py             incumbent statutory-fidelity referee, one Claude model
  judges/jev.py                 TypeSafe Jev: one Choice verdict + one Noul per kind
  judges/replay.py              pre-recorded responses (tests, re-folds)
  results.py                    run loop, cases.jsonl resume, results.json contract
  metrics.py                    AUC, paired rise, detection at a false-alarm ceiling
  localization.py               does a finding name the mutated rule or token?
  board.py                      fold + leaderboard; refuses non-comparable inputs
  agreement.py                  run-to-run self-agreement of one judge on identical text
  cli.py                        build-synthetic, build-real, filter-suite, show-suite, run, agreement, board
fixtures/real_defects_example/  two-case stand-in for real_defects_v0 (loader tests)
```

`real_defects_v0/` is produced by another session and is never written here.
