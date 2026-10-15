# Reviewed coverage exemptions

The coverage checklist computes the same demands as the original scanner, then
applies `src/axiom_encode/harness/data/coverage_exemptions.jsonl`. An occurrence
is removed only when its citation, requested scope, source text SHA-256, label,
physical offset, literal window and window SHA-256 match an approved entry.
There are no patterns or wildcard keys. Offsets count Unicode characters in the
exact text passed to the scanner; SHA-256 covers its UTF-8 bytes. The window is
`text[max(0, physical_start - 80):physical_start + 80]`.

Changing any source byte invalidates every exemption for that text. Descendant
composition, parent slicing, joined citations and caller-supplied text receive
the same check. `requested_scope: null` means no additional scope relative to
the exact source citation, including an exact subsection row. A requested child
of a parent-fallback row uses the exact list of scope labels from the checklist.

Parent-fallback repeal waivers retain all existing summary and authoritative
source conditions. They additionally require a matching `repealed_tombstone`
entry or the exact target row's `metadata.status == "repealed"` in the bound
corpus release. Text ancestry alone cannot grant the waiver.

## Propose and review

From a materialized, signed release, run:

```sh
uv run python scripts/propose_coverage_exemptions.py \
  --release RELEASE --corpus-path CORPUS_ROOT > candidates.jsonl
```

`--release-sha256` selects an exact release object when more than one is present;
`--citation` restricts proposals to one row. The retired ownership heuristic
lives only in `scripts/retired_coverage_heuristic.py`. Its output is unapproved
review context. The proposal tool never writes the approved list.
For parent fallback, supply `--citation PARENT --requested-source TARGET` to
record the actual encoding target; unrelated targets are rejected.

For every candidate, open the exact source bytes at the recorded offset, check
the complete owner-to-label span and the encoding target, and establish the
owner citation and physical offset or a complete repeal/omission tombstone.
Reject inline titles, unknown marker chains and ambiguous typography. Add
approved entries and their review evidence in a reviewed PR. Record rejected
candidates with reasons; an unproved demand stays mandatory. Each entry needs
its artifact digest, physical JSONL row, reviewer and review date.

## Stale entries and corpus checks

```sh
uv run python scripts/check_coverage_exemptions.py \
  --release RELEASE --release-sha256 CONTENT_SHA256 \
  --corpus-path CORPUS_ROOT --output ledger.json
```

The checker verifies the signed release and artifact bytes, then requires every
entry to identify a demand emitted by the unchanged scanner at that offset.
It also requires removals to equal list matches exactly, with zero additions.
Use `--listed-artifacts-only` to check every row in the registry's artifacts;
an explicitly pinned `--artifact-manifest` selects a recorded corpus slice.

The `Coverage exemptions` CI workflow fetches the signed release from the public
corpus store used by CI parity and the listed artifacts from that release's
immutable Git commit. It verifies artifact hashes and runs
the stale-entry and differential checks on every PR. The committed evidence
records the broader historical corpus replay and seed review. A release bump
must update this check's pin and re-review any entry whose source changed;
entries are never automatically carried forward.
