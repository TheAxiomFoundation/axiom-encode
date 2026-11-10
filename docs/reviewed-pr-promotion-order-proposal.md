# Proposal: approve a promoted RuleSpec PR after validation, not before

Status: **proposal, not decided.** Tracking issue: #2172. This document changes
no behavior. `docs/reviewed-pr-promotion.md` remains the description of the
live lane.

## The problem in one paragraph

The promotion lane admits a candidate on one fact: a signing owner approved its
exact head. At that moment the candidate PR is red and shows no validation
result, because it has no signed manifest. `Reject unmanifested RuleSpec
content` fails in the `shards` job of `validate-rulespec.yml`; the `validate`
job needs `shards`, so compile, companion tests, proof checks and the waiver
ratchet do not run. The owner approves bytes that no machine has checked on
that PR. The lane document then asks for a second review of the signed head.

Observed on rulespec-us#2045 (head `cf1e5124`, two added files): the only CI
result is the guard failure.

## Current order

| # | Step | Who | What the reviewer can see |
|---|------|-----|---------------------------|
| 1 | Local encode, open candidate PR | contributor | — |
| 2 | Approve exact head | signing owner | red PR, guard failure only |
| 3 | Dispatch `targeted-signed-reencode.yml` | not the approver | — |
| 4 | Admission, `production-signing` gate, validate, sign | workflow | — |
| 5 | Signed PR opened on the candidate branch, full CI | workflow | — |
| 6 | Review signed head, merge | owner | green PR |

Costs of this order:

- Step 2 is blind. It cannot establish "source and behavior correctness"
  from CI, only from reading the YAML.
- The approval is bound to the candidate head and to the head of RuleSpec
  main. Each merge to main voids every pending approval. With serial landing
  (strict protection, CI of 3–8 hours) this happens to most candidates.
- Two owner reviews per rule (steps 2 and 6).
- An owner who dislikes approving a red PR has no green alternative.

## Option A (recommended): sign first, approve the green signed PR

| # | Step | Who | What the reviewer can see |
|---|------|-----|---------------------------|
| 1 | Local encode, open candidate PR | contributor | — |
| 2 | Dispatch promotion | any permitted actor | — |
| 3 | Scope checks, `production-signing` gate, validate, sign | workflow | — |
| 4 | Signed PR, full CI | workflow | — |
| 5 | Approve exact signed head, merge | signing owner | green PR, validation and CI results |

What changes:

1. **Admission** (`reviewed_pr_admission.py`): the exact-head owner approval is
   no longer a precondition for signing. All other checks stay: same
   repository, base `main`, new primary plus companion only, complete tree
   delta, branch tip equals head, central workflow run on encoder main.
2. **Manifest**: a new field records what the signature attests, for example
   `review: none` and `origin: local-candidate`. The signature then means
   "encoder main validated these exact bytes", not "an owner reviewed them".
3. **Merge gate** (RuleSpec repository): a manifest with `review: none` may
   land only when a signing owner approved the exact signed head. Two ways to
   enforce it, see question 2.
4. **Base binding**: the candidate no longer needs a fresh approval each time
   main moves. Strict branch protection already forces the signed PR up to
   date before merge.

What does not change: key custody, the `production-signing` environment gate,
the complete-source validation in the promotion run, scope limits, attempt
accounting, no automatic merge, the author ≠ approver rule (it moves to the
merge gate).

Risk that this option accepts: the signing key signs bytes that no owner
reviewed. The controls that remain before those bytes reach main are the
validator, full CI, and one owner approval on the final head. Today the same
three exist, plus the blind approval at step 2.

## Option B (fallback): keep the order, make validation visible

Keep approval before signing, but let the candidate PR show evidence:

1. `validate-rulespec.yml`: on a pull request, an unmanifested **new** module
   pair does not stop the `shards` job. Validation runs. A separate job
   reports the provenance result (`awaiting signed manifest`) and stays
   required, so the PR cannot merge.
2. Admission additionally requires the validation job green at the approved
   head.

The owner then approves a PR whose only failing check is provenance. This
keeps the present meaning of the signature. It keeps the two reviews and the
base binding, and it runs the 3–8 hour CI twice per rule (candidate and signed
PR). It also touches the first-job guard, which exists so that a content
violation is always the first red.

## Questions for a maintainer decision

1. Is a signature over validated but unreviewed bytes acceptable when the
   manifest says so and the merge needs an owner approval? If no, Option B.
2. Where does the merge gate live: branch protection on RuleSpec main
   (required review from code owners; today `required_approving_review_count`
   is 0), or a guard step that reads the reviews on the exact head the way
   admission does today?
3. May any collaborator dispatch a promotion in Option A, or only enrolled
   identities? This sets the abuse surface on signing compute and on the
   `production-signing` gate.

## Sequence

The first production canary (rulespec-us#2045) runs on the current order, so
that the live lane is proven end to end before it changes. This proposal
applies from the second promotion.
