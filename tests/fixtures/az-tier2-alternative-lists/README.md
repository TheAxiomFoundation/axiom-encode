# Arizona shelter, utility and dependent-care candidates with bulleted alternative lists

Generated candidates, their companion tests and the exact unit source text
from one local batch of three units of the Arizona DES FAA5 manual (public
text):

- `us-az/manual/des/faa5/shelter-expenses-and-deduction/block-16`
- `us-az/manual/des/faa5/na-utility-expenses-and-allowances/block-17`
- `us-az/manual/des/faa5/dependent-care-expense/block-6`

- Producer: encoder 0.2.2387 (commit
  `384e5ff82a94507b8284d585d779ab4b7f78d7d2`), runner `codex` with
  `gpt-6.1-sol`, local runs on 2026-11-19, two attempts for each unit. These
  are unsigned local records, not hosted workflow artifacts.
- `candidate.yaml` and `candidate.test.yaml` are the bytes that the run wrote.
  `source.txt` is the workspace source text of the unit; it equals the body of
  the unit's row in the corpus checkout that the runs used (corpus commit
  `7822462a7bb07222decc2896a86650f1208ca354`, provision file
  `data/corpus/provisions/us-az/manual/2026-07-17-faa5-recovery.jsonl`). No
  file was edited.
- The replay fixtures of the six candidates
  (`tests/fixtures/us_replay_benchmark/local-az-shelter-block-16-*`,
  `local-az-utility-block-17-*`, `local-az-dependent-care-block-6-*`) also
  hold the `source-metadata.json` of the same run workspace. One field is
  edited there: `rulespec_root`, an absolute path on the machine that ran the
  batch, is replaced by `<local-run-root>/rulespec-us/us-az`. The directories
  here do not hold that file.
- The run workspaces are not in this repository, so the statement that each
  file equals its run's bytes is the author's report of a local comparison on
  2026-11-19, not a checked value.

| Directory | Unit | Local run | Recorded verdict |
|---|---|---|---|
| `shelter-block-16-2eaed082` | shelter block-16 | `2eaed082` (attempt 1) | rejected by `complete-source-unit:source-explicit-conditions/gates-collapsed` |
| `shelter-block-16-8176fca8` | shelter block-16 | `8176fca8` (attempt 2) | rejected by the same check |
| `utility-block-17-27c00482` | utility block-17 | `27c00482` (attempt 1) | rejected by the same check |
| `utility-block-17-3f95091c` | utility block-17 | `3f95091c` (attempt 2) | rejected by the same check |
| `dependent-care-block-6-22c86176` | dependent-care block-6 | `22c86176` (attempt 1) | rejected by the same check |
| `dependent-care-block-6-deac6f76` | dependent-care block-6 | `deac6f76` (attempt 2) | rejected by the same check |

Each run recorded that one finding and no other. The replay of the producer
commit gives the same finding text as the run recorded.

Each unit opens with a list of occasions under "... when any of the following
occur(s):". The six candidates encode the occasions as alternatives. The check
read the two bullets that hold a verb as two simultaneous facts. In the
dependent-care unit the clause also ran on into the next paragraph, which
became a third fact. `tests/test_source_alternative_list_gates_az_tier2.py`
pins that the check no longer fires on these candidates, because each unit has
a reviewed reading bound to this exact text
(`docs/reviewed-alternative-lists.md`), and it pins what the check no longer
verifies for these clauses.

The replay fixtures hold byte-identical copies of the six candidates.
