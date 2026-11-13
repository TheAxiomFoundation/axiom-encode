# Reviewed alternative lists

`complete-source-unit:source-explicit-conditions/gates-collapsed` counts the
simultaneous facts of a source condition. It reads every bullet under a
condition cue (`if`, `when`, `provided that`, `unless`) as one more fact. For a
list of alternatives ("when one or more of the following apply:") that reading
is wrong, and the check refuses a candidate that encodes the alternatives as
alternatives.

The check has no text grammar for alternatives. Two grammars were tried and
both let conjunctive lists pass. A registry that also re-checked the list
structurally was tried and could be satisfied by naming an input. So the
exception is a registry of reviewed exceptions:
`src/axiom_encode/harness/data/reviewed_alternative_lists.jsonl`. It follows
the pattern of the reviewed coverage exemptions (`docs/coverage-exemptions.md`):
literal keys, no patterns, bound to the exact text.

**An entry switches the check off for one clause of one unit text.** For that
clause `gates-collapsed` verifies nothing: not the structure of the
alternatives, and not the other facts of the clause. A candidate that merges
the whole clause into one input passes this check. The assurance for the clause
comes from the owner who reviews the candidate encoding against the source, and
from the other checks of the lane (proof validation, companion tests, oracle
and reviewers). Each entry says in plain words what is not examined and what
the owner must verify. `tests/test_source_alternative_list_gates.py` pins this
behaviour.

## What an entry says

One JSON object per line:

| Field | Meaning |
|---|---|
| `citation_path` | Corpus citation path of the source unit. |
| `source_text_sha256` | SHA-256 of the UTF-8 bytes of the unit text as the check scans it: the corpus body with leading and trailing whitespace removed. |
| `chapeau` | `physical_start` and `line`: the physical line that holds the condition cue, verbatim. |
| `lines` | Every nonblank physical line after the chapeau that the clause holds, in source order. Each has `physical_start`, `line` (verbatim) and `reading`. |
| `reason` | One line: why the reviewer read the list this way. |
| `not_examined` | Nonempty list of plain sentences: the facts of the clause that the check no longer verifies for this unit. |
| `owner_must_verify` | Nonempty list of plain sentences: what the approving owner checks in the candidate encoding against the source. |
| `author` | Who recorded the reading (a person or a model run). |
| `reviewer`, `reviewed_at` | The owner who checked every recorded line against the source text, as a GitHub login (`@name`), and the date. Until an owner has done that: `"pending-owner-review"` and `null`. |

`reading` is one of:

- `alternative`: one alternative of the list.
- `outside_condition`: text after the list that is not part of the condition
  (for example a later sentence in the same clause). It may only follow the
  last line of the list.

A list needs at least two `alternative` lines. Offsets count Unicode characters
in the scanned text. There are no wildcards. The readings document what the
reviewer read; the effect of the entry is the same for every recorded line.

## When an entry applies

Only when all of these hold:

1. The citation of the unit under validation equals `citation_path`.
2. The SHA-256 of the scanned text equals `source_text_sha256`.
3. The chapeau and every recorded line stand at their recorded offsets, each as
   a whole physical line.
4. In the clause that a formula proof excerpt quotes, the line with the cue is
   the recorded chapeau, and the nonblank lines after it, to the end of the
   clause, are exactly the recorded lines.

Then the clause has no gates in this check.

In every other case nothing changes: a changed source byte, another citation,
a missing citation, a clause with other bounds, or a registry that cannot be
loaded gives no exception and every bullet stays a gate. Other clauses of the
same unit are examined as before.
`tests/test_reviewed_alternative_lists.py` makes a malformed or duplicate
committed entry fail in CI.

## The current entries

| Unit | Not examined by the check | The owner verifies in the candidate |
|---|---|---|
| utility block-14 | Lifeline participation AND (discount does not pay in full OR extra airtime after free minutes): separate inputs, the AND, the OR | the Lifeline path is exactly that; Lifeline alone is not enough; no listed condition is negated, dropped or constant |
| utility block-3 | listed utility expense AND used to provide heat for the home: separate inputs, the AND, every listed expense | heating source = (electricity OR gas OR propane OR coal OR wood OR oil) AND used for heat; nothing missing; the later lists are not conditions |
| utility block-4 | verified expenses AND (situation 1 OR situation 2); the two facts inside each situation | both situations, each with both of its facts, combined with OR, next to verified expenses |

The registry file holds the full sentences.

## Add an entry

An entry loosens a validator for one unit. The pull request that adds it must
declare `Validation effect: loosen` and
`Changed checks: complete-source-unit:source-explicit-conditions/gates-collapsed`.
The registry file, its loader and its test have code owners
(`.github/CODEOWNERS`), so the pull request needs a code-owner review.

The author records the reading with `reviewer` set to `"pending-owner-review"`
and `reviewed_at` set to `null`, and writes `not_examined` and
`owner_must_verify`. The approving owner reads the source passage, confirms the
reading, checks the candidate encoding against `owner_must_verify`, then
replaces the marker with their GitHub login (`@name`) and sets `reviewed_at`,
in the same pull request.

CI enforces this. The test
`tests/test_reviewed_alternative_lists.py::test_every_entry_has_an_owner_review_outside_a_draft`
fails on any pending entry in every GitHub Actions run except a draft pull
request: a pull request that is ready for review, the merge queue, a push, and
a manually dispatched run. It passes locally and on a draft pull request, so
the author can show the replay effect first. A signed entry must name a login
from the registry file's line in `.github/CODEOWNERS`, with a `reviewed_at`
date that is not in the future; this holds in every run.

Consequence: a manually dispatched full-suite run on a branch with a pending
entry (`gh workflow run CI --ref <branch>`) shows exactly one expected failure,
that test, until an owner signs. Any other failure in such a run is a real one.

1. Confirm the false refusal. The finding quotes the source clause and its
   facts. The candidate must encode the alternatives as alternatives.
2. Add the rejected candidate as a replay fixture
   (`tests/fixtures/us_replay_benchmark/`, see `docs/us-replay-benchmark.md`),
   with the unit text as `source.txt`.
3. Print the digest and the line offsets as the check sees them:
   `uv run python scripts/reviewed_alternative_list_digest.py tests/fixtures/us_replay_benchmark/<fixture>/source.txt`.
   The script removes leading and trailing whitespace first, as the check does.
   Do not use a plain `shasum` of the file: it differs when the file has a
   final newline.
4. Record the reading: the chapeau line, then every nonblank line after it in
   the clause, each with its character offset and its reading. Read each line
   in the source. When a line of the list can be read as a requirement of its
   own, the list is not a list of alternatives: do not add an entry.
   Write `not_examined` and `owner_must_verify` for the whole clause.
5. Append the entry as one compact JSON line. Keep the field order.
6. Run `uv run pytest tests/test_reviewed_alternative_lists.py
   tests/test_us_replay_benchmark.py`. The registry test requires every entry
   to match the text of a replay fixture with the same citation.
7. Regenerate the replay snapshot and report the removed findings in the pull
   request.

When the corpus text of a unit changes, its entry stops applying. Review the
new text and add a new entry; do not edit the digest alone.
