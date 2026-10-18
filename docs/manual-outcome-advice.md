# US manual outcome advice

The 2026-10-18 decision keeps #1833's numeric-limit-only blocking controls
unchanged. Exception sentences on US manual citation paths can additionally
receive optional repair advice. Advice never enters blocking issue lists,
changes a validation result, or initiates a repair attempt.

Selection uses the existing source structure, computation and exception
parsers, and works on exception branches, as #1833's filter does: a selected
branch must lack a `_STATED_NUMERIC_LIMIT` match and contain an explicit
outcome word. One sentence can hold several branches, so a sentence with a
numeric limit in one branch can still yield advice for another ("If identity
is verified, approve the application, and if income exceeds $500, deny it"
advises on its first branch). Replay counts measure advice records, one per
branch. `_MANUAL_OUTCOME_WORD` in
`source_completeness.py` recognizes eligibility/ineligibility, benefits,
allotments, payments/pay/paid, sanctions, disqualification, termination,
denial/deny, approval, reductions, increases, entitlement, counting/countable,
inclusion, exclusion, deeming, disregards and deductions, with the inflections
listed in that regular expression. Matching ignores case. Income and resources
alone, notice, verification and required are not outcome words.

This shortlist follows the manual examples in
[PR #1893](https://github.com/TheAxiomFoundation/axiom-encode/pull/1893)
(eligible, counted, excluded and deducted) and the agreed task definition.
That branch contains a broader federal administrative classifier, rather than
a reusable manual-sample recognizer. Its classifier is not imported. A match
only identifies a sentence worth considering; it does not establish that the
sentence is operative.

CI validation stores records in
`ValidationResult.details.manual_outcome_exception_advice`. Encode result JSON
persists the same list at `metrics.manual_outcome_exception_advice`, including
on passing attempts. An absent field means an empty list; empty advice is
omitted to preserve existing result JSON for other sources. Each record contains `corpus_citation_path`,
`source_sha256`, `source_start`, `source_end`, `excerpt` and `message`. Character
offsets refer to the original authoritative source body; excerpts are bounded
to 360 characters. Records follow source order and depend only on the source,
so the same sentence keeps its identity across candidate repairs.

Replay tooling can count the list, deduplicate across attempts by citation,
source digest and character offsets, and compare the identified original
source span to derived-output proof excerpts and paired tests to measure
uptake. The list records selected advice; it does not assert that feedback was
delivered or that the candidate encoded the sentence. Existing retry traces
show exposure: clean attempts and exhausted feedback budgets can retain
selected records without sending them to the model.

On an existing repair, the selected failed attempt's advice joins the bounded
feedback channel after blocking feedback. It appears under an optional advice
section, marked `[non-blocking]`, and explicitly says that a derived output
with paired tests may be worthwhile and is not required. If blocking feedback
exhausts the feedback budget, the full advice list remains in result JSON.
Guarded encoder prompt content is unchanged.

The byte-exact #1833 replay fixture for run 36880457452 retains zero blocking
completeness issues. Its advice list is empty: its only exception sentence is
"Refer to policy 770-3 to determine when no notice is required for the SNAP
program.", which contains no outcome word from the shortlist. Uptake on the
next Oregon/Utah replay remains the follow-up measurement before considering
any blocking demand.
