Add EncodeBench, the model-capability eval: `benchmarks/encodebench_uk_v1.yaml`, a
16-case stratified UK suite bound to the `uk-rulespec-2026-07-14` release
(cold capability cases plus three repo-augmented oracle candidates, rate
gates pinned to 0.0), and the `eval-board` command, which folds one or more
eval-suite outputs into an N-runner leaderboard with a per-case grid,
refusing folds across differing case identities, corpus releases, or
duplicate runners. Documented in `docs/encodebench.md`;
formalizes the 2026-07-10 encoder bake-off (#1189).
