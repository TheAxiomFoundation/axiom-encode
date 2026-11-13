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
| `source_text_sha256` | SHA-256 of the UTF-8 bytes of the exact unit text that the check scans. |
| `chapeau` | `physical_start` and `line`: the physical line that holds the condition cue, verbatim. |
| `lines` | Every nonblank physical line after the chapeau that the clause holds, in source order. Each has `physical_start`, `line` (verbatim) and `reading`. |
| `reason` | One line: why the reviewer read the list this way. |
| `reviewer`, `reviewed_at` | Who read the lines against the source text, and the date. |

`reading` is one of:

- `alternative`: one alternative of the list. It is not counted as a
  simultaneous fact.
- `conjunct`: the line stays in the condition and is counted as before. Use it
  for an alternative that states its own conjunction.
- `outside_condition`: text after the list that is not part of the condition
  (for example a later sentence in the same clause). It is not counted.

Offsets count Unicode characters in the exact text. There are no wildcards.

## When an entry applies

Only when all of these hold:

1. The citation of the unit under validation equals `citation_path`.
2. The SHA-256 of the scanned text equals `source_text_sha256`.
3. The chapeau and every recorded line stand at their recorded offsets, each as
   a whole physical line.
4. In the clause that a formula proof excerpt quotes, the line with the cue is
   the recorded chapeau, and the nonblank lines after it, to the end of the
   clause, are exactly the recorded lines.

Then the `alternative` and `outside_condition` lines are not counted. In every
other case nothing changes: a changed source byte, another citation, a missing
citation, a clause with other bounds, or an unreadable or malformed registry
gives no exception and every gate stays. `tests/test_reviewed_alternative_lists.py`
makes a malformed or duplicate committed entry fail in CI.

## Add an entry

An entry loosens a validator for one unit. The pull request that adds it must
declare `Validation effect: loosen` and
`Changed checks: complete-source-unit:source-explicit-conditions/gates-collapsed`,
and an owner reviews each recorded line against the source text.

1. Confirm the false refusal. The finding quotes the source clause and its
   facts. The candidate must encode the alternatives as alternatives.
2. Add the rejected candidate as a replay fixture
   (`tests/fixtures/us_replay_benchmark/`, see `docs/us-replay-benchmark.md`),
   with the unit text as `source.txt`.
3. Compute the digest of that text:
   `shasum -a 256 tests/fixtures/us_replay_benchmark/<fixture>/source.txt`.
   It equals `files["source.txt"]` in the fixture's `fixture.json`.
4. Record the reading: the chapeau line, then every nonblank line after it in
   the clause, each with its character offset and its reading. Read each line
   in the source. When a line can be read as a requirement, record it as
   `conjunct`.
5. Append the entry as one compact JSON line. Keep the field order.
6. Run `uv run pytest tests/test_reviewed_alternative_lists.py
   tests/test_us_replay_benchmark.py`. The registry test requires every entry
   to match the text of a replay fixture with the same citation.
7. Regenerate the replay snapshot and report the removed findings in the pull
   request.

When the corpus text of a unit changes, its entry stops applying. Review the
new text and add a new entry; do not edit the digest alone.
