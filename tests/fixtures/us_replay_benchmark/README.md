# US replay benchmark fixtures

Each directory is one rejected US candidate from the
`targeted-reencode-failure-<run_id>-<attempt>` artifact of the named workflow
run. `fixture.json` records the run, citation, lane, encoder commit and version,
corpus and RuleSpec refs, recorded failure classes, the origin of each context
module and the SHA-256 of every file in the directory.
`scripts/replay_us_failure_bundles.py` refuses a fixture whose bytes or file
list differ from `fixture.json`.

`context/` files come from the bundle's workspace. A `providers/` file comes
from the RuleSpec repository at the run's `rulespec_ref`, for a module that the
candidate needs but the workspace context lacks.

`snapshot.json` pins the per-fixture, per-rule issue counts of the replay with
context. See `docs/us-replay-benchmark.md` for what the benchmark measures, its
limits, and how to update the snapshot or add fixtures. The directory names and
metadata are recorded labels; this checkout does not independently
authenticate the original workflow runs.
