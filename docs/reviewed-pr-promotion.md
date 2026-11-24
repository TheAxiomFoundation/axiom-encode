# Promote a RuleSpec PR without an encoder admission PR

This lane replaces the *per-candidate encoder record PR*. It keeps protected
signing, source validation, attempt accounting and final CI. Since #2301 it no
longer requires an independent owner approval of each candidate; see
"Admission basis" below. The static record lane remains supported. It is not
coupled to the separate proposal to remove production-signing environment
approval.

## Scope and authority

The central `targeted-signed-reencode.yml` workflow, dispatched on encoder main,
can promote one new primary and companion from a same-repository RuleSpec PR
based on current RuleSpec main. The optional derived reverse index is the only
other permitted changed file. Existing-module replacements, stacked PR bases,
forks, workflow changes, manifests, imports, queues and repair inputs are refused.
Scope is derived from complete recursive base/head leaf-tree differences (path,
type, mode and object ID), not just the mutable PR-files response. Both PR files
and immutable comparison files must agree exactly with that delta; truncated or
contradictory evidence fails closed. Later live revalidation checks mutable
authorization against the provisioned receipt, not repeated immutable queries.

## Admission basis

`OWNER_APPROVAL_REQUIRED` in `src/axiom_encode/reviewed_pr_admission.py` decides
what authorizes one promotion. It is a constant on encoder main, read at call
time. There is no environment variable, CLI flag or repository setting, and a
change to it needs a code-owner review. Every scope check above is the same
under both values.

**`False` (in force): owner dispatch.** Decision of the repository owner in
#2301: an encoding lands when the harness accepts it; no person approves each
unit.

- No `APPROVED` review is required.
- The accountable act is the dispatch. The live run's triggering actor must be
  an enrolled identity of the main-owned signing policy (currently MaxGhenis,
  ID 6076111, or PavelMakarchuk, ID 110687043): the numeric ID, the login and
  the account type `User` are read from the live run and must all match. An
  account that only has write access cannot promote, although it can still
  start the workflow. A re-run by another account makes that account the
  triggering actor. If the identity cannot be read, admission is refused.
- The dispatching owner may be the author of the candidate PR.
- The veto stays. An outstanding changes-requested review refuses admission,
  from any account (it is not limited to enrolled owners), until that reviewer
  approves or the review is dismissed.
- The queue path (`github-actions[bot]`) cannot promote: promotion refuses
  queue inputs, and a bot is not an enrolled `User`.

What is lost: no second person sees a candidate before it is signed, and none
is required before it merges (RuleSpec `main` requires no approving review).
A wrong rule that every machine check accepts lands on `main`.

What remains: the validator checks of complete-source validation, the executed
companion tests, the model reviewer, the oracle comparison where one exists,
the signature by the protected key, the enrolled-owner dispatch, the
`production-signing` environment approval (a repository setting; it permits
the dispatcher to approve their own run), CI on the signed head, and the veto
above.

**`True`: owner approval.** The rule in force before #2301, kept as a switch.
One effective `APPROVED` review must be on the exact candidate head, from an
enrolled identity. The PR author and triggering workflow actor cannot supply
that approval, and any account with write access may dispatch. Later dismissal
or a superseding decisive review invalidates it. Outstanding changes-requested
reviews refuse admission. Comments, agent review text, ordinary collaborator
reviews and old-head approvals do not grant authority.

Under both values the enrolled identities are not inferred from the RuleSpec
CODEOWNERS file. Revocation of this policy is an encoder-main policy change;
changing org membership alone does not change this identity allowlist.

The admission names its basis: `"admission_basis": "owner_dispatch"` with a
`dispatcher` (`id`, `login`), or `"admission_basis": "owner_approval"` with the
`approvals`. It never carries an empty approvals list. A reader refuses an
unknown basis, a basis that is not the policy in force, and a record whose
evidence field does not match its basis. The signed manifest is unchanged: it
does not record the basis, so `reviewed_rulespec_ref` in a manifest no longer
implies a review by a second person. The basis of one landing is in the
retained runtime attestation of its signing run.

This lane does not claim the candidate bytes were newly generated during
promotion. The existing complete-source validation is rerun.

The workflow sends a selector, never an approval receipt, to the provisioner.
The root provisioner loads verifier code only from the attested encoder snapshot,
fetches live GitHub evidence itself and binds the admission to the central run,
encoder commit, PR, country, exact head and main base, branch, citation, path,
primary/companion hashes, and the admission basis with its dispatcher identity
or approving review IDs. This record is in the
root-owned runtime attestation; candidate files, environment JSON and arbitrary
CLI record paths cannot supply it. The supervised command rechecks mutable
authorization before validation and immediately before signing. It separately
rehashes local bytes and changes only the manifest. Publication checks live
authorization again, and the existing branch-tip check remains.
The root runtime attestation is retained in the checksummed workflow artifact;
it is audit evidence, not a portable authorization token.

GitHub reads use the fixed TLS API endpoint, no redirects or ambient proxy,
bounded pages/responses, and no credentials in the signing runtime. API failures
and rate limits fail closed; they never admit cached evidence. Public GitHub
anonymous limits apply, so repeated dispatch/polling is not a retry strategy.
The byte-bound immutable content is fully fetched at preflight/provisioning;
later checks reread mutable head/base/reviews rather than downloading it again.

## Dispatch

Use the ordinary exact SHA/corpus/engine/citation inputs, `open_pr=true`, and
`pr_base_branch` equal to the reviewed candidate branch. Set `replace_rulespec_path`
to its canonical new module. The existing source input carries a strict wrapper,
so this adds no workflow-dispatch input beyond GitHub's current 25-input limit:

```json
{
  "schema": "axiom-encode/reviewed-pr-promotion/v1",
  "rulespec_pr": 2044,
  "transaction": {
    "schema": "axiom-encode/atomic-source-transaction/v5",
    "source_bundle": [],
    "canonical_refresh_bundle": [],
    "primary_required_test_cases": [],
    "require_complete_source_unit": true,
    "manifest_only_refresh": false,
    "reviewed_candidate_promotion": true
  }
}
```

The number above is illustrative, not an admission. All remaining generation,
repair, import, legacy and queue inputs must be empty. Production-signing approval
still applies. A moved candidate or main base requires a new dispatch on the new
head; an API error requires investigating the API state, not disabling the checks.

The signed RuleSpec PR still requires green exact-head CI before merge. This
feature does not automatically merge it.

## Rollout

Merge this one reusable encoder implementation only after independent review,
required signing-code-owner approval and green CI. Existing pending candidate
admission PRs can continue to merge and sign through the static lane meanwhile;
Arizona encoding must not wait for this rollout. A production canary on an
approved exact-head candidate is required before calling the new lane operational.
