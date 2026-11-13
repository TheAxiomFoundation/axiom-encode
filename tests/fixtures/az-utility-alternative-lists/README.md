# Arizona utility candidates with bulleted alternative lists

Generated candidates, their companion tests and the exact unit source text
from one local batch of
`us-az/manual/des/faa5/na-utility-expenses-and-allowances/` (Arizona DES FAA5
manual, public text).

- Producer: encoder 0.2.2361 (commit
  `547274cd0893b785fbf89f219e9df26491e04f5f`), runner `codex` with
  `gpt-6.1-sol`, local runs on 2026-11-12. These are unsigned local records,
  not hosted workflow artifacts.
- `candidate.yaml` and `candidate.test.yaml` are the bytes that the run wrote.
  `source.txt` is the workspace source text of the unit. No file was edited.

| Directory | Unit | Local run | Recorded verdict |
|---|---|---|---|
| `block-14-e9371ed3` | block-14 | `e9371ed3` | rejected by `complete-source-unit:source-explicit-conditions/gates-collapsed` |
| `block-14-66be7798` | block-14 | `66be7798` | rejected by the same check |
| `block-3-07b0428f` | block-3 | `07b0428f` | rejected by the same check |
| `block-3-9dd68cae` | block-3 | `9dd68cae` | rejected by the same check |
| `block-4-170e8d74` | block-4 | `170e8d74` | rejected by the same check and by `complete-source-unit:tests/boundary-unexercised` |
| `block-4-28d570f4` | block-4 | `28d570f4` | rejected by the same check, by `boundary-unexercised` and by a companion input-slot failure |
| `control-block-16-858a1c04` | block-16 | `858a1c04` | passed local validation |

The six rejected candidates encode a bulleted list of alternatives ("one or
more of the following apply", "either situation occurs", "any of the
following") as alternatives. The check read each bullet, and in block 3 each
later paragraph, as one more simultaneous fact. `tests/test_source_alternative_list_gates.py`
pins that the check no longer fires on them. Block 16 has a true "all of the
following" list; the test merges its facts into one input and requires the
check to fire.

`tests/fixtures/us_replay_benchmark/local-az-utility-block-*` holds
byte-identical copies of the six rejected candidates as replay fixtures.
