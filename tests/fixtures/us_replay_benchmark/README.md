# US replay benchmark fixtures

Each directory is one rejected US candidate from the
`targeted-reencode-failure-<run_id>-1` artifact of the named workflow run.
`fixture.json` records the run, citation, lane, encoder commit and version,
corpus ref, recorded failure classes and the SHA-256 of every file in the
directory. `scripts/replay_us_failure_bundles.py` refuses a fixture whose bytes
or file list differ from `fixture.json`.

`snapshot.json` pins the per-fixture, per-rule issue counts of the
import-resolving replay. See `docs/us-replay-benchmark.md` for what the
benchmark measures, its limits, and how to update the snapshot or add fixtures.
The directory names and metadata are recorded labels; this checkout does not
independently authenticate the original workflow runs.
