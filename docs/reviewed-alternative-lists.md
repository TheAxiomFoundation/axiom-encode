# Reviewed alternative lists

`complete-source-unit:source-explicit-conditions/gates-collapsed` counts the
simultaneous facts of a source condition. It reads every bullet under a
condition cue (`if`, `when`, `provided that`, `unless`) as one more fact. For a
list of alternatives ("when one or more of the following apply:") that reading
is wrong, and the check refuses a candidate that encodes the alternatives as
alternatives.

The check has no text grammar for alternatives. Two grammars were tried and
both let conjunctive lists pass. The exception is a registry of reviewed
readings: `src/axiom_encode/harness/data/reviewed_alternative_lists.jsonl`. It
follows the pattern of the reviewed coverage exemptions
(`docs/coverage-exemptions.md`): literal keys, no patterns, bound to the exact
text.

## What an entry says

One JSON object per line:

| Field | Meaning |
|---|---|
| `citation_path` | Corpus citation path of the source unit. |
| `source_text_sha256` | SHA-256 of the UTF-8 bytes of the unit text as the check scans it: the corpus body with leading and trailing whitespace removed. |
| `chapeau` | `physical_start` and `line`: the physical line that holds the condition cue, verbatim. |
| `lines` | Every nonblank physical line after the chapeau that the clause holds, in source order. Each has `physical_start`, `line` (verbatim) and `reading`. |
| `reason` | One line: why the reviewer read the list this way. |
| `author` | Who recorded the reading (a person or a model run). |
| `reviewer`, `reviewed_at` | The owner who checked every recorded line against the source text, as a GitHub login (`@name`), and the date. Until an owner has done that: `"pending-owner-review"` and `null`. |

`reading` is one of:

- `alternative`: one alternative of the list. All `alternative` lines together
  are one fact of the condition ("any one of these"), not one fact each.
- `conjunct`: a line of the list that is a requirement of its own. It stays in
  the condition and is counted as before, next to the group of alternatives.
- `outside_condition`: text after the list that is not part of the condition
  (for example a later sentence in the same clause). It is not counted. It may
  only follow the last line of the list.

A list needs at least two lines and at least one `alternative`. Offsets count
Unicode characters in the scanned text. There are no wildcards.

## When an entry applies

Only when all of these hold:

1. The citation of the unit under validation equals `citation_path`.
2. The SHA-256 of the scanned text equals `source_text_sha256`.
3. The chapeau and every recorded line stand at their recorded offsets, each as
   a whole physical line.
4. In the clause that a formula proof excerpt quotes, the line with the cue is
   the recorded chapeau, and the nonblank lines after it, to the end of the
   clause, are exactly the recorded lines.

Then the `alternative` lines become one gate and the `outside_condition` lines
are not counted. The other facts of the clause stay gates. A candidate must
still give the group an input of its own next to the inputs of the other
facts, among the inputs that the formula reaches. So a candidate that merges
another fact of the clause with the whole list into one input is still
refused. The check does not look inside the group: one input for "any of the
alternatives" passes.

In every other case nothing changes: a changed source byte, another citation,
a missing citation, a clause with other bounds, or an unreadable or malformed
registry gives no exception and every bullet stays a gate.
`tests/test_reviewed_alternative_lists.py` makes a malformed or duplicate
committed entry fail in CI.

## Add an entry

An entry loosens a validator for one unit. The pull request that adds it must
declare `Validation effect: loosen` and
`Changed checks: complete-source-unit:source-explicit-conditions/gates-collapsed`.
The registry file, its loader and its test have code owners
(`.github/CODEOWNERS`), so the pull request needs a code-owner review.

The author records the reading with `reviewer` set to `"pending-owner-review"`
and `reviewed_at` set to `null`. The approving owner checks each recorded line
against the source text, then replaces the marker with their GitHub login
(`@name`) and sets `reviewed_at`, in the same pull request, before merge.
`tests/test_reviewed_alternative_lists.py` lists the citations that may wait
for review; a pending entry that is not listed there fails.

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
   in the source. When a line of the list can be read as a requirement, record
   it as `conjunct`.
5. Append the entry as one compact JSON line. Keep the field order.
6. Run `uv run pytest tests/test_reviewed_alternative_lists.py
   tests/test_us_replay_benchmark.py`. The registry test requires every entry
   to match the text of a replay fixture with the same citation.
7. Regenerate the replay snapshot and report the removed findings in the pull
   request.

When the corpus text of a unit changes, its entry stops applying. Review the
new text and add a new entry; do not edit the digest alone.
