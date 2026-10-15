# Procedure sequencing exemption

The paired-condition gate exempts only two complete, normalized clauses from the
canonical Florida calculation-of-benefits manual page. The first directs the
worker to calculate budgets after determining available income. The second
directs the worker to compare determined gross income with the applicable limit.
Neither clause states a comparison result or an applicability proposition.

This is a closed list of complete clauses. Any additional qualification, changed
instruction, different source, or unknown wording retains the previous paired
case obligation. The complete source body must match the authenticated page or
be exactly the isolated complete clause; surrounding qualifications retain the
obligation. The source text includes sequencing uses of “When” and surrounding
manual numbers; those exact retained clauses are the only exceptions. No general
condition grammar or formula-semantics relaxation is retained.

The corpus inputs are pinned to [the immutable corpus revision](https://github.com/TheAxiomFoundation/axiom-corpus/tree/8f7d60aaced28ee4252b9237f9d6e02360dc34bc/data/corpus/provisions).
The retained inventory binds every file by SHA-256 and byte count. Reproduction
reads these inputs in a local provisions directory and writes a deterministic
JSON result. The runner verifies the relevant extraction and pairing closure
against the historical unrelaxed main revision after removing only the explicit
exemption hook and passage-locator metadata that cannot alter formula spans.
Only bodies containing an entire allowed clause at the exact canonical citation
can change; all other rows are excluded by this proof. Selected bodies receive
the unchanged clause extraction and complete formula suppression.

Run the locked GIL CPython 3.13 environment with numeric thread caps of four.
Download batches and scans run sequentially; every invocation must finish within
ten minutes. The download helper uses immutable source URLs and authenticates
each file before keeping it. Fetch at most fifteen files per invocation, then
run the differential. The runner's `--help` describes the input and output
arguments.

The historical unrelaxed main comparison is distinct from later main snapshots.
The first runner records a selected-page diagnostic for the prior main snapshot
`c2ee166fd1813392022fe5cf4c1e08ae1f09aba0`. That snapshot already contains a
broader manual-condition relaxation. Its selected-page diagnostic is retained
only to identify the historical comparison scope.

The latest-main measurement evaluates the exact agency-instruction prefix before its
joined-segment classifier. It also evaluates unchanged suffix checks before
unchanged prefix checks in the complete numeric-age and state-code citation
guards. The three Boolean reorders retain every original operand and regex.
AST proofs authenticate complete original helpers and Boolean expressions;
generated examples and
complete predicate/extraction controls check its classification results. The
original and execution source digests, helper AST digests, and precise proof
scope are retained in the validation receipt. The validator loads each pinned
source closure both unchanged and accelerated, then compares complete extraction
and pairing controls under each revision's own manual context. Production source
is unchanged by this measurement.
The completed validation observed 26 explicit Boolean checks, 2,000 generated
Unicode examples yielding 4,000 Boolean checks, 90 full predicate controls,
90 full extraction controls, and 45 complete cross-version differential controls.
Twelve controls also assert exact expected obligation counts. The authenticated
Florida page retains five final obligations and two latest-main obligations;
its difference is three additions and zero removals.
Direct clause-boundary validation passed 1,138 ordered clause-sequence
comparisons, 1,366 state-code Boolean comparisons, and 22 expected-state-code
controls across 57 explicit and 512 generated source examples. The text and
Unicode-digit strategies have no length cap; explicit digit runs reach 8,192
characters. These controls compare unchanged and transformed helpers directly,
without clause memoization.
Within one revision/body call, the measurement also materializes the original
clause generator once for each exact text-and-branches object pair. Subsequent
calls receive fresh iterators over those immutable ordered triples. Strong
references preserve argument identity; the manual context remains fixed, and
the original generator is restored even after an error. The receipt binds the
cache helper AST and each original and actually executed generator AST. The
cache wraps the transformed generator. Both revisions passed reuse,
different-identity, fresh-iterator, separate-call, and error-restoration controls,
with four original materializations each. The 45 complete controls compare
uncached original extraction with cached accelerated extraction. Independent
review found no new actionable correctness finding in these optimizations.
Completed rows are appended once to a checkpoint ledger, and repeating a bounded
command resumes the first unfinished row. Only a completed census produces the
summary with counts, per-file results, input/source/audit bindings, and ledger
digest. The full ledger can be regenerated from the immutable corpus inputs.

A fresh rerun authenticated all 39,375 rows in 117 files totaling 146,278,276
bytes and reproduced `corpus-differential.json` byte for byte. Against historical
main and the pre-PR parent it removes exactly the two procedure demands and adds
none. The prior main snapshot's selected-page diagnostic reports zero removals
and three additions; it is not a complete comparison with that snapshot.

`reproduce-latest-main.py` instead pins main at
`4f3547461300bfa70569f57c83728b2ad25d9d0b`. Each side executes its own extraction,
formula suppression, manual-prose context, and pairing predicate. This runner
loads only reachable source definitions and pins every reached repository
dependency to the corresponding revision. It assumes no equivalence between the
two extraction implementations. Every input is authenticated, and its ordered
ledger binds both source closures, audit scripts, inventory, and completed
rows. Bounded invocations resume unfinished work. The completed fresh census
authenticated all 39,375 rows and reports **252 removals and 94,938 additions**
of raw path/span/clause-hash obligation identities against this newer main
snapshot. [The retained summary](latest-main-differential.json) binds the full
23,311,134-byte ledger and its SHA-256 digest. Per-file sums independently
match the totals. This materially differs from the requested two-removal,
zero-addition result: it includes later extraction and gate changes on main and
cannot be attributed entirely to the narrow procedure exemption. The
two-removal result above applies only to the historical baseline comparison.
The census started with a fresh ledger after its measurement receipt passed;
earlier partial ledgers are debug material. A separate byte comparison found
39,372 matching body records against the previous partial ledger; no records
were imported into the completed census.

Restoring the old witness implementation also restores its preexisting
admissions of some incorrect formulas. The stronger historical rejection
assertions remain unmarked and unchanged. Their baseline failures remain real
test failures; they are not converted into expected failures or passing proof
of those formulas' rejection.
The requirement that every historical incorrect formula retain a final analyzer
issue conflicts with restoring the baseline witness behavior. That limitation
must remain explicit in the final handoff and independent review.

```sh
python fetch-inputs.py 0 15 --output-dir <provisions-directory>
python fetch-inputs.py 15 30 --output-dir <provisions-directory>
# Continue sequential batches through the final range 105 117.
python reproduce-corpus.py --corpus-root <provisions-directory> --output <result.json>
python validate-measurement.py --corpus-root <provisions-directory> --output measurement-validation.json
python reproduce-latest-main.py --corpus-root <provisions-directory> --ledger <latest-checkpoint.jsonl> --summary <latest-main-differential.json>
# Repeat the final command when it reports PAUSED; exit 75 is a saved checkpoint.
```
