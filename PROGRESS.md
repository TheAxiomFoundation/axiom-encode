# encode#1559 exact Sol-review fixes — progress

Branch `wt/encode-1559` → `origin/fix/editorial-numeric-recall` (PR #1559).
Reviewed head: d23a0056 ("Harden Armenian amendment ledger recognition").

## Scope
Four actionable exact-head Sol findings in Armenian ARLIS editorial-history
stripping, span provenance, complete-mode eval accounting, and prompt parity.

## Done
- [x] Replace independent Unicode delimiter counters with one monotone LIFO
      stack. Crossed pairs, mismatched pairs, stray closers, and unclosed pairs
      permanently fail closed, including malformed text after an otherwise
      admissible ledger.
- [x] Admit documented official legacy labels such as `169^{10 }- րդ`,
      `169^{12} - րդ`, and `169.23 -րդ` only at the narrow horizontal-spacing
      positions used by ARLIS; malformed components and wrong suffixes remain
      source content.
- [x] Keep filtered occurrence offsets and contextual applicability checks on
      the same reconstructed text, closing the decoy-parameter formula bypass.
- [x] Apply `authoritative_numeric_recall_text` to complete-mode eval inventory
      and half-up recall accounting, matching the production validator.
- [x] Ratchet the unpublished PR version from 0.2.1754 to 0.2.1755 after the
      final encoder-affecting changes, including all oracle and provenance pins.
- [x] Add exhaustive regressions for all 210 crossed pair orderings, all 210
      mismatched pair orderings, all 15 stray closers, post-ledger mismatch,
      official/malformed legacy spacing, exact surviving spans, the end-to-end
      formula bypass, and eval-metric parity.

## Evidence
- Focused Armenian selection: 533 passed.
- Complete `tests/test_source_completeness.py`: 6,673 passed.
- Complete `tests/test_evals.py` plus complete-source plumbing: 811 passed.
- Focused changed-file Ruff, format, compile, and `git diff --check`: clean.
- Repository-wide and independent cycle receipts will be added after they run.

## Next
Run repository-wide checks and the independent review-fix cycle, commit the
clean repair, and report the exact commit without pushing or merging.
