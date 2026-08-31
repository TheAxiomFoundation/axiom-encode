# encode#1559 second Sol-review fixes — progress

Branch `wt/encode-1559` → `origin/fix/editorial-numeric-recall` (PR #1559).
Reviewed head: 32800e14 ("Format Armenian recall regressions").

## Scope
Five actionable second-review defects in Armenian ARLIS editorial-history
stripping and the two encoder prompt surfaces.

## Done
- [x] Replace the lazy/backtracking history regex with a deterministic token
      scanner. The 32,768-space rejection probe fell from 2.254 seconds in the
      review to 0.000134 seconds locally.
- [x] Replace the source-sized depth list with an O(1)-memory monotone scan,
      performed only after the Armenian grammar validates. Non-Armenian sources
      now return before scanning generic parentheticals.
- [x] Admit official single-number `ՀՕ-228` legacy identifiers,
      `փոփ.08.09.08` dot-adjacent dates, and `ՀՕ-538-2-Ն` identifiers without
      admitting glued action/citation residue or lowercase identifiers.
- [x] Validate calendar dates, use ASCII digits in the admitted grammar, and
      retain malformed dates for strict numeric recall.
- [x] Factor one Armenian editorial-history protocol into both the ordinary
      encoder and eval-authoring prompts; prompt generation is now 4.
- [x] Add action-only, citation-only, invalid-date, official-form, scaling, and
      allocation regressions.
- [x] Version ratchet 0.2.1753 across pyproject/`__init__`/uv.lock + test pins
      (origin/main is 0.2.1750, so 1753 stands).

## Evidence
- `tests/test_source_completeness.py` plus complete-source plumbing: 6,215
  passed.
- `tests/test_evals.py` plus complete-source plumbing: 810 passed.
- `tests/test_rulespec_validation.py`, `tests/test_cli.py`, and complete-source
  plumbing: 3,851 passed.
- Focused Armenian/prompt selection: 70 passed after the final narrow legacy-ID
  grammar.
- Targeted Ruff/format and `git diff --check` clean.
- Broader and full-suite receipts will be recorded after this repair is
  committed.

## Next
Commit the repair, request a fresh exact-head independent review, then rerun
GitHub CI. Keep the PR draft and do not merge until the review-fix cycle and
durable Fable agreement both clear.
