# Focused validation

The production change restores the pre-PR condition-discovery and witness
implementation from `f824da90344c6dd5b59811b4f3121f888b0adf85`, except for two
complete, authenticated Florida procedure clauses. The semantic-check module
and other admission paths introduced by this PR are removed. This does not
claim complete source/formula semantic equivalence.

The two positive clauses are reproduced verbatim in
`tests/test_source_completeness_manual_procedure_regressions.py`. They include
sequencing uses of “When,” chapter numbers, and a table reference. The exemption
recognizes those exact instructions. Changed normalized wording, added
qualifications, different citations, and unauthenticated surrounding bodies keep
the previous source pairing obligation. Generated comparison, conditional,
quantity, and applicability mutations exercise the closed boundary, including
qualifications separated by extraction boundaries.

The reviewed implementation at `d238162cc534bc048261a0c6088439a3f860ee96`
freshly failed all five round-5 regression instances: ordinary and mirrored
equality-only inclusive comparisons, sparse compound Boolean cases, and honest
and reversed negative principal predicates. These are reproduced assertion failures of the
reviewed code, not passing validation of the final implementation.

Historical fixtures retain their original source, formula, companion cases,
and stronger rejection assertions. The assertions that fail on the immutable
pre-PR witness implementation remain unmarked failures. Temporary strict-xfail
markers were removed because they weakened the existing test requirements.
Separate assertions require the affected
source clauses to retain their pairing obligation. The requirement that every
historical bad formula produce a final analyzer issue cannot be met while
restoring unchanged baseline witness behavior; this remains a documented
assignment blocker.

Fresh checks after the interrupted attempt are recorded below. Test receipts
with strict expected failures are superseded intermediate runs. The final test
suite preserves unmarked rejection assertions and exits with real failures.

| Check | Observed result |
| --- | --- |
| Round-5 regression cases against reviewed production | 5 assertion failures, zero execution errors, 263 deselected |
| Diagnostic fixture replay against immutable pre-PR and final production | 168 analyzer calls across 84 fixtures; identical outcomes |
| All five affected test files, one final invocation without skip/xfail markers | Exit 1; 68 failed, 405 passed in 166.55 seconds |
| Original generated families against immutable pre-PR production | 13 assertion failures; no execution errors |
| Generated exemption-boundary property, final rerun | 64 passing generated examples, 1 discarded draw; 6 additional explicit examples |
| `reproduce-corpus.py --corpus-root <provisions-directory> --output <result.json>` | Exit 0; authenticated 39,375 rows, 117 files, 146,278,276 bytes; output exactly matches retained receipt |
| Historical main and parent corpus differential | 2 demands removed, 0 added for each baseline |
| Prior main snapshot selected-page diagnostic | 0 removed, 3 added; selected page only |
| `validate-measurement.py --corpus-root <provisions-directory> --output measurement-validation.json` | Exit 0; 26 explicit Boolean checks, 2,000 generated Unicode examples, 4,000 generated Boolean checks, 90 predicate controls, 90 extraction controls, 45 complete differential controls |
| Exact extraction controls | 12 expected-count controls; authenticated full page final 5 obligations, latest main 2; full and sparse extraction agree |
| Direct clause-boundary validation | 1,138 clause-sequence comparisons, 1,366 state-code Boolean comparisons, 22 expected-state-code controls; 57 explicit and 512 generated sources; explicit Unicode digit runs reach 8,192 characters |
| Per-call clause cache controls | Both revisions passed reuse, distinct-identity, fresh-iterator, separate-call, and error-restoration controls; 4 original materializations each |
| `tests/test_source_completeness.py tests/test_source_completeness_max_witnesses.py -k 'relational or paired or exception_header or inclusive or proviso or equality'` | 176 passed, 6,485 deselected |
| `tests/test_workflow_toolchain_pins.py::test_the_test_environment_is_exactly_the_locked_set` with `AXIOM_CHECK_LOCKED_ENV=1` | 1 passed; shared-cache wrapper used for UV-stripped child |
| `ruff check pyproject.toml src/axiom_encode scripts tests` and evidence scripts | Exit 0 |
| `ruff format --check src/ tests/` and evidence scripts | Exit 0; 264 files already formatted |
| `python -m compileall -q src/axiom_encode scripts` | Exit 0 |
| `python -m towncrier build --draft --version 0.0.0` | Exit 0; draft only |
| Encoder version, lockfile, and package-init diff against reviewed revision | Empty |
| Complete latest-main census against `4f3547461300bfa70569f57c83728b2ad25d9d0b` | Exit 0; 39,375 rows, 117 files, 146,278,276 input bytes; 252 removals and 94,938 additions of raw path/span/clause-hash obligation identities |
| Census receipt checks | Per-file totals match; 39,372 body-record byte comparisons match the previous partial ledger; completed census began empty and imported no records |
| Independent review cycle | Temporary xfail markers identified as weakening the tests and removed; final review found no additional issue beyond the explicit requirement conflict and failing tests |

Final observed validation totals are 582 passed and 68 failed, including the
176 passing focused dependents and one passing locked-environment check. No
failed assertion is marked skip or xfail. The earlier 582-pass/68-strict-xfail
aggregate is superseded and is not a final passing result. The four historical
files are
`test_source_completeness_manual_condition_semantics.py`,
`test_source_completeness_manual_procedure_conditions.py`,
`test_source_completeness_manual_procedure_guards.py`, and
`test_source_completeness_manual_relation_guards.py` under `tests/`.
The final 68 failing node IDs exactly match the preserved baseline-failure
inventory, with no unexpected or missing failures. This confirms the failure
set; it does not make the suite pass.

An initial measurement-validator attempt failed an incorrect absolute-count
assertion that inferred three final obligations from a three-addition delta.
Direct full-module, shared-runner, sparse-original, and sparse-accelerated
extraction all agreed on five final obligations. The corrected validator binds
the five exact path/span/hash tuples and the two latest-main tuples, and passed.
That failed attempt is not counted as a successful validation run.

The latest-main runner pins
`4f3547461300bfa70569f57c83728b2ad25d9d0b` and compares both revisions' own source
obligations, including current manual-prose context and numeric-limit filtering.
It loads authenticated reachable source definitions and repository dependencies,
then checkpoints strictly increasing corpus rows under source, audit, and input
hashes. Both sides retain complete formula suppression. No closure-equality
shortcut with historical main is used for this comparison.
The completed [latest-main summary](latest-main-differential.json) binds a
23,311,134-byte ledger with SHA-256
`27a1799e5cb3bd44bedaced44c74d9a7a314d701e7707b93ff202bd0d93490c1`.
Its 252-removal/94,938-addition result materially differs from the expected
two-removal/zero-addition result. These raw obligation-identity differences
include later extraction and gate changes on main; they are not all caused by
the narrow exemption. The two-removal/zero-addition result is the separately
verified historical comparison.

The corpus-only measurement preserves Boolean and ordered clause-classification
results for valid strings. It authenticates complete original helpers and
Boolean expressions before reordering the guidance classifier and two clause
boundary guards. Every operand and regex remains unchanged. Boundary controls
compare unchanged and transformed helpers directly without memoization;
generated arbitrary-text and Unicode-digit strategies have no length cap.
The cache reuses the executed generator's immutable clause
triples only for identical text and branches objects within one fixed-scope
obligation call. It retains strong argument references, returns fresh iterators,
and restores the generator in `finally`. Its helper AST and the original and
actually executed generator ASTs are bound in the regenerated receipt. Chained
transformation receipts also bind original and final execution source digests.
The 45 complete
controls compare uncached original extraction against cached accelerated
extraction, and independent review found no new actionable code finding.
Exception, call-count, iterator-type, and resource-timeout equivalence are not
claimed. Production files are not transformed. The final cached census uses a
fresh ledger; earlier partial uncached scans contribute no rows to it.

Fresh Python work uses the locked GIL CPython 3.13.15 environment with numeric
thread caps of four, one heavy process at a time, bounded commands, disabled
pytest coverage, and dedicated basetemps removed after each run. No dependency,
lock, or encoder-version change is included. The full local suite and live CI
are not certified; repository CI remains the merge gate.
