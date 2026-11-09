# US replay benchmark fixtures

Each directory is one candidate from the
`targeted-reencode-failure-<run_id>-<attempt>` artifact of the named workflow
run: rejected candidates, and controls (`recorded_verdict` other than
`rejected`) that production validation accepted. `fixture.json` records the
run, citation, lane, encoder commit and version, corpus and RuleSpec refs,
recorded failure classes, the origin of each context module, and the SHA-256
of every file in the directory. `scripts/replay_us_failure_bundles.py` refuses
a fixture whose bytes or file list differ from `fixture.json`.

Only visible cases from public sources are committed. Held-out cases, hand
failure-class labels, private-corpus sources and raw bundles stay outside the
repository.

`context/` files come from the bundle's workspace. A `providers/` file comes
from the RuleSpec repository at the run's `rulespec_ref`, for a module that the
candidate needs but the workspace context lacks.

`snapshot.json` pins every fixture's exact findings with context. The
`replay benchmark` CI job fails when a finding differs from it; rewrite it with
`scripts/replay_us_failure_bundles.py snapshot --base-ref origin/main`. See
`docs/us-replay-benchmark.md` for what the benchmark measures, its limits, the
rule for false-positive fix PRs, and how to add fixtures. The directory names
and metadata are recorded labels; this checkout does not independently
authenticate the original workflow runs.
