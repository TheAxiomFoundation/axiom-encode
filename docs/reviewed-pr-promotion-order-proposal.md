# Proposal: approve a promoted RuleSpec PR after validation, not before

Status: **direction decided, not implemented.** A maintainer chose Option A
with conditions on 2026-11-12 (see "Maintainer decisions"). Tracking issue:
#2172. This document changes no behavior, and the new order is not active.
`docs/reviewed-pr-promotion.md` remains the description of the live lane.

## The problem in one paragraph

The promotion lane admits a candidate on one fact: a signing owner approved its
exact head. At that moment the candidate PR has no signed manifest, so its
required check is red and it shows no validation result. `Reject unmanifested
RuleSpec content` fails in the `shards` job of `validate-rulespec.yml`; the
matrix `validate` job needs `shards` and is skipped, so RuleSpec validation,
companion tests, proof checks and the waiver ratchet do not run. The owner
approves from a reading of the YAML alone.

Observed on rulespec-us#2045 (head `cf1e5124`, two added files):
`validate / shards` fails at the guard, the matrix job is skipped, and the
required `validate / validate` context is red. The side checks that do run
(`engine-root-load`, rollback proof, toolchain, build) are green; none of them
executes the companion tests or the proofs.

## Current order

| # | Step | Who | What a reviewer can see |
|---|------|-----|-------------------------|
| 1 | Local encode; open candidate PR against RuleSpec main | contributor | — |
| 2 | Approve the exact candidate head | enrolled signing owner, not the author | required check red, no validation result |
| 3 | Dispatch `targeted-signed-reencode.yml` | anyone but the approver | — |
| 4 | `production-signing` gate, then admission preflight, validate, sign | workflow | — |
| 5 | Draft signed PR opened **into the candidate branch**; merge it | workflow, then a person | no RuleSpec validation runs: Repository Checks trigger only for PRs into `main` |
| 6 | Candidate PR, now with the manifest, runs full CI | CI | green candidate PR |
| 7 | Merge the candidate PR | a person | — |

Facts about this order:

- Step 2 is the only human control that is **enforced**. The lane document
  asks for an independent review of the final head, but RuleSpec main requires
  0 approving reviews and no code-owner review.
- Step 2 is also the control with the least evidence.
- The approval is bound to the candidate head and to the head of RuleSpec
  main. `revalidate_admission` rechecks both before validation, before
  signing and before publication. Each merge to main, and each push to the
  candidate, voids the approval. Landing is serial (strict protection, CI of
  about 3–5 hours), so most pending approvals are voided.
- Admission runs inside the job that holds the `production-signing`
  environment. A dispatch with no valid approval consumes a signing-gate
  approval first and fails after.
- The author cannot approve and the approver cannot dispatch. For a candidate
  that one enrolled owner authored, the other enrolled owner is the only
  possible approver.

## Option A (chosen): sign first, approve the green candidate PR

| # | Step | Who | What a reviewer can see |
|---|------|-----|-------------------------|
| 1 | Local encode; open candidate PR | contributor | — |
| 2 | Dispatch promotion | enrolled identity at first; may be the candidate author (decision 3) | — |
| 3 | Scope preflight, `production-signing` gate, validate, sign | workflow | — |
| 4 | Signed commit reaches the candidate branch; full CI on the candidate PR | workflow, CI | — |
| 5 | Approve the candidate PR; merge | enrolled signing owner, neither the author nor the dispatcher | green required check, validation and CI results, the manifest |

What changes:

1. **Admission no longer needs an approval.** This is more than one
   precondition in `reviewed_pr_admission.py`:
   - `revalidate_admission` compares live approvals with the provisioned ones
     at three points (`cli.py` before validation and before signing; the
     workflow before publication);
   - the root provisioner writes `approvals` (reviewer and review IDs) into
     the runtime attestation, which is the audit artifact;
   - the record's `purpose` string states an owner approval.
   All three must describe an unreviewed promotion truthfully.
2. **Every scope check stays, including the base binding.** Same repository,
   PR open and unmerged, base `main` at its current head, candidate strictly
   ahead, new primary plus companion only (optional reverse index), target
   paths absent from main, complete tree delta, regular blobs, branch tip
   equals head, no outstanding changes-requested review, central workflow run
   on encoder main. Scope is judged against the base tree, so the base must
   stay current. The cost of a moved main falls from "a new owner review" to
   "update the branch and dispatch again".
3. **Manifest**: a new field in a versioned signed manifest records what the
   signature attests, for example `review: none`, `origin: local-candidate`.
   The signature means "encoder main validated these exact bytes", not "an
   owner reviewed them". Verifiers must distinguish the two (decision 1).
4. **A new merge gate** in the RuleSpec repository: a manifest with
   `review: none` lands only with an approval from an enrolled signing owner
   who is neither the candidate author nor the authenticated dispatcher. This
   gate does not exist today. Decision 2 sets its form: a trusted
   content-bound guard that branch protection requires. It binds to the
   approved content hashes over the complete final PR delta (primary,
   companion, manifest and any permitted reverse index), not to the head
   SHA, because "update branch" under strict protection creates a new
   head with the same content. An approval survives a base-only update only
   when the approved content and the permitted scope are unchanged.
5. **Preflight before the gate**: the scope preflight moves to a job that
   runs before the `production-signing` environment, so a refused dispatch
   consumes no gate approval.

What does not change: key custody, the `production-signing` gate, the
complete-source validation in the promotion run, attempt accounting, final
CI on the candidate PR, no automatic merge.

Risk that this option accepts: the signing key signs bytes that no owner
reviewed, and the single enforced human control moves from before signing to
the merge gate of change 4. Option A must not ship without that gate: the
gate is deployed and enforced before the pre-sign approval is removed. In
return, the one enforced review is made with validation and CI results
visible, and a base-only update does not void it while the approved content
and the permitted scope are unchanged.

## Option B (fallback): keep the order, make validation visible

Keep the approval before signing, but let the candidate PR show evidence:

1. Two guards must let an unmanifested **new** module pair through to
   validation on a pull request: the `shards` pre-check and the supervised
   `Reject manual RuleSpec changes` step (`axiom-encode guard-generated`),
   which sits in the `validate` job before YAML validation, companion tests
   and proofs, and is the authoritative one.
2. A separate required job reports provenance (`awaiting signed manifest`),
   so the PR still cannot merge.
3. Admission additionally requires the validation job green at the approved
   head.

The owner then approves a PR whose only failing check is provenance. This
keeps the present meaning of the signature. It also keeps the approval that
each main merge voids, runs the full CI twice per rule (before and after
signing), and weakens two guards whose purpose is that unmanifested content
goes no further.

## Questions for a maintainer decision

All three are answered; see "Maintainer decisions" below. The questions stay
here as the record of what was asked.

1. Is a signature over validated but unreviewed bytes acceptable when the
   manifest says so and the merge needs an owner approval? If no, Option B.
2. Where does the merge gate live: branch protection on RuleSpec main
   (required code-owner review; today `required_approving_review_count` is 0),
   or a guard step that reads the reviews the way admission does today? Only
   the second can bind to content digests and to the enrolled-identity list.
3. Who may dispatch a promotion in Option A: any collaborator or only
   enrolled identities? This sets the load on signing compute and on the
   `production-signing` gate.

## Maintainer decisions (2026-11-12)

Recorded from the review by @MaxGhenis on #2173
(`pullrequestreview-5877972088`): Option A for promotions after the current
canary.

1. **Signature scope (question 1): yes.** The signing service may sign
   validated, unreviewed candidate bytes when a versioned signed manifest
   explicitly records that scope. Verifiers must distinguish validation from
   owner approval. Protected signing, authoritative-source validation, attempt
   accounting and final CI stay.
2. **Merge gate (question 2): a trusted content-bound guard, required through
   branch protection.**
   - Its code and the enrolled identities load from protected policy.
   - The required check is bound to its trusted publisher.
   - It checks the complete final PR delta, canonical paths, approved content
     hashes, live effective reviews and current-head CI.
   - It is deployed and enforced before the pre-sign approval is removed.
   - An approval is retained across base-only updates only when the approved
     content and the permitted scope stay unchanged.
   - The final approver is independent of both the candidate author and the
     authenticated dispatcher.
3. **Dispatch (question 3): enrolled identities at first.** An author may
   dispatch their own candidate. A different enrolled owner gives the final
   approval. Broader dispatch is revisited after the pilot, with explicit
   limits on compute and on approval load.

These decisions guide the implementation. They do not activate the new order.

## Sequence

1. The first production canary (rulespec-us#2045) runs on the current order,
   so that the live lane is proven end to end before it changes.
2. The merge gate of decision 2 is built and passes adversarial tests.
3. The new order is activated only after that gate is enforced and holds the
   conditions above. Until then every promotion uses the current order.
