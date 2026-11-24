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
- The accountable act is the dispatch. Two accounts are read from the live run
  and must be the same enrolled identity of the main-owned signing policy
  (currently MaxGhenis, ID 6076111, or PavelMakarchuk, ID 110687043): the
  run's `actor`, who dispatched it and chose its inputs, and its
  `triggering_actor`, who started this attempt. For each, the numeric ID, the
  login and the account type `User` must match. An account that only has write
  access cannot promote, although it can still start the workflow. If either
  identity cannot be read, admission is refused.
- Re-runs, in both directions. Another account's re-run of an owner's dispatch
  is refused: the triggering actor is not the owner. An owner's re-run of
  another account's dispatch is refused: the actor that chose the inputs is
  not the owner. This also holds when the two accounts are the two enrolled
  owners. Only the owner who dispatched a run can re-run it.
- The dispatching owner may be the author of the candidate PR.
- The veto stays in the code. An outstanding changes-requested review refuses
  admission, from any account (it is not limited to enrolled owners), until
  that reviewer approves or the review is dismissed. A review whose ID or
  reviewer ID is not a positive integer refuses.
- The queue path (`github-actions[bot]`) cannot promote: promotion refuses
  queue inputs, and a bot is not an enrolled `User`.

What is lost: no second person sees a candidate before it is signed, and none
is required before it merges (RuleSpec `main` requires no approving review).
A wrong rule that every machine check accepts lands on `main`.

What this lane enforces after the change: the scope checks above, complete-source
validation of the exact bytes (validator checks, the executed companion tests,
policy proofs), the signature by the protected key, the enrolled-owner
dispatch, the `production-signing` environment approval, and CI on the signed
head.

What this lane does not enforce, before or after the change: the model
reviewer and the oracle comparison. Promotion validates with reviewers skipped
and oracles off, and RuleSpec CI validates with `--skip-reviewers`. A reviewer
verdict or an oracle result exists only if the local encode that produced the
candidate ran them, and the lane does not check that it did.

### What the remaining controls do not prove

Stated plainly, because the approval was the only control that needed a second
account:

1. **Type `User` is an account type, not a person.** Any tool or agent session
   that holds an enrolled owner's token is that `User`: it can dispatch, and
   the admission cannot tell it from the owner at a keyboard.
2. **The `production-signing` approval is not a separate human act by
   construction.** The environment's required reviewers are the two enrolled
   owners and `prevent_self_review` is off (read from the environment's
   settings). The approval is an API call that the dispatching login can make
   for its own run. (Not tested by making the call.)
3. **The veto does not bind the dispatching owner.** RuleSpec `main` has no
   review-dismissal restriction (read from the branch protection of
   `rulespec-us`; organization rulesets could not be read). An account with
   write access can dismiss a changes-requested review, and a dismissed review
   no longer refuses. The veto stops a dispatch only while nobody with write
   access dismisses it. (Not tested by dismissing a review.)
4. **Therefore:** between a session that holds an enrolled owner's GitHub
   login and a signed landing on `main` stand only machine controls (the scope
   checks, complete-source validation, CI on the signed head), plus whatever
   discipline the owner applies to the environment approval.

### Known limit: switching back to `True` does not reach older runs

A signing run executes the encoder commit it was started on. A run started
while `main` had `False` can be re-run after `main` is back on `True`, and the
re-run still admits by owner dispatch: admission requires the run's branch to
be `main`, not its commit to be the current head of `main`. The window is
narrow: only the enrolled owner who dispatched that run can re-run it, only
while GitHub still offers the re-run, the environment approval is asked again,
and every scope check runs again against the current candidate head and the
current head of RuleSpec `main`. A switch back to `True` should be followed by
a look at the recent runs of the workflow.

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
`dispatcher` (`login`, `actor_id`, `triggering_actor_id`), or `"admission_basis": "owner_approval"` with the
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

## Several units in one candidate

**Switched off.** `MULTI_UNIT_PROMOTION_ENABLED` in
`src/axiom_encode/reviewed_pr_admission.py` is `False`. While it is not exactly
`True`, everything in this section is refused and every single-unit read,
record and error is the same as before the switch existed. The switch is a
constant on encoder main, read at call time. There is no environment variable,
CLI flag, dispatch input or repository setting, and a change to it needs a
code-owner review.

Why: per-unit owner approval is gone (#2301). The limit that remains is
RuleSpec CI: one head takes 3 to 4 hours, and each landing needs the newest
`main`. With one unit per candidate, each unit pays for one CI run and one
merge. With the switch on, one candidate pull request carries several new
units, and they share one signing run, one CI run and one merge.

### Rules

With the switch on, a dispatch that names its units (below) admits a candidate
whose change set is exactly `k` units, `1 <= k <= MAX_PROMOTION_UNITS`:

- Each unit is a complete pair: its new primary and its new companion. A unit
  without its companion, or a companion without its primary, is refused.
- Every path is new on `main` and has status `added`.
- No other file is changed, added, deleted or renamed. The one exception is the
  derived reverse index (`.axiom/index/provisions_to_rules.json`), which is
  permitted as for a single unit. It is one file for the repository, not one
  per unit.
- No primary and no citation is named twice, so no path belongs to two units.
- Every unit is checked as the selector of a single-unit promotion is
  (country, canonical path, primary YAML).
- Every other rule of this lane is unchanged and is checked once for the
  batch: exact head, base `main` at its current head, same repository, the
  live central signing run, the admission basis (enrolled owner dispatch, or
  owner approval), the veto, and the complete tree delta.

A dispatch without a unit list is a single-unit promotion. It refuses a
candidate that holds a second unit, with the error it gave before
(`candidate PR changes files outside its new module scope`), with the switch
on or off.

### How the batch is named

The `source_bundle_json` wrapper takes one more optional key, `units`: the
citation of every unit of the candidate.

```json
{
  "schema": "axiom-encode/reviewed-pr-promotion/v1",
  "rulespec_pr": 2044,
  "units": ["us/statute/7/2015/a", "us/statute/7/2015/b"],
  "transaction": { "...": "as for a single unit" }
}
```

- The `citation` and `replace_rulespec_path` inputs name one of the units. Any
  one; they do not order or select anything. The commit message and the
  title of the lane pull request show that one citation.
- The preflight maps each citation to its canonical path and writes a batch
  selector with the units in path order.
- The admission reads the live change set of the pull request at the exact
  head and requires it to equal the named units. A list that names fewer units
  than the candidate holds is refused (the other files belong to no named
  unit). A list that names more is refused (a named unit is missing). So the
  input cannot select a part of a candidate.
- The list is needed because a citation cannot be computed back from a path.
  The paths are never trusted from the input: they must equal the pull
  request's own.
- `units` with one citation is a batch of one. Its record and its manifest
  equal those of a single-unit promotion of that unit.
- While the switch is off the wrapper does not know the `units` key: it is
  refused with the error an unknown key had before
  (`reviewed PR promotion requires exactly transaction and rulespec_pr`).

### One signing run

`promote-reviewed-candidate` reads the units from the root-provisioned
admission, not from its arguments, and processes them in path order:

1. Every unit: binding, canonical path, no existing manifest, exact bytes at
   `HEAD` against the admission record.
2. No unit may import another unit of the same batch (below).
3. Every unit: complete-source validation of its exact bytes, the same call as
   for a single unit.
4. The admission is read live again, once.
5. Every unit: the signature.
6. One transaction installs every manifest, then verifies every installed
   manifest.

The result is one signed manifest per unit, at the unit's own canonical
manifest path, in the form of a single-unit promotion: the same schema
(`axiom-encode/applied-rulespec/v5`), tool, fields and signature form. With
the clock held, each manifest is byte-identical to the manifest of a
single-unit promotion of that unit. A verifier reads each manifest by itself,
so it needs no change. The units of one batch share `run_id` and
`reviewed_rulespec_ref`. The admission record in the runtime attestation is a
separate form (`axiom-encode/live-reviewed-pr-batch-admission/v1`) with one
record per unit; each record equals the record of a single-unit admission.

### All or nothing

- A unit that fails step 1, 2 or 3 stops the run before any signature is
  requested, also for units that already passed.
- A refused live admission (step 4) stops the run before any signature.
- A failed signature (step 5) stops the run before any manifest is written.
  Signatures of earlier units then exist only in the memory of the stopped
  process.
- A manifest that fails verification after the install (step 6) removes every
  manifest of the transaction.
- An error of one unit names it: `unit <citation> (<path>): <reason>`. This
  holds for steps 1, 3 and 5 and for the verification of an installed
  manifest. Three errors are for the whole batch and name no unit: the import
  refusal (it names the two files), a refused live admission, and a failure of
  the one install transaction itself (a write error, or a unit file that
  changed after validation).

A failed batch is repaired in the candidate (fix or remove the unit) and
dispatched again on the new head. There is no partial result to keep.

### Imports between units of one batch

Refused. A unit whose primary imports another unit of the same batch stops
the run before validation, and the error names both files. Reason: each unit
is validated and signed as a single-unit promotion of it would be, against
modules that are already signed. The single-unit lane cannot express such an
import at all (its candidate holds one new module), so this keeps each batch
manifest equal to a single-unit one. Promote the imported unit first, then the
importer in a later candidate.

The check uses the validator's own import resolver
(`_resolve_rulespec_import_file_static`). It resolves every import of each
primary and refuses when the resolved file is the primary or the companion of
another unit of the batch. So the spelling does not matter: a different case
of the prefix, an empty or unknown prefix, quotes and surrounding slashes are
all followed to the file the validator would read. The import items are read
twice: from the parsed YAML, and as the validator reads the `imports` block
from the text. The check fails closed: a primary that is not a readable YAML
mapping, an import item that is not a string (also a mapping-form item), and
an import that the resolver refuses as unsafe each stop the batch.

### Limits

- `MAX_PROMOTION_UNITS` is 6. The bound is the anonymous GitHub API budget of
  one signing run: 60 requests an hour for one address, and the signing
  runtime holds no token. One admission of `k` units reads `11 + 2k` times; a
  revalidation reads 4 times. One run admits twice (preflight and root
  provisioner) and revalidates three times:
  `2 x (11 + 2k) + 3 x 4 = 34 + 4k <= 60`, so `k <= 6` (58 reads). A
  single-unit run costs 38. Ten units need 74 reads: that needs fewer or
  authenticated reads first, in a reviewed change.
- The budget is per address and per hour. A second dispatch from the same
  address in the same hour can exhaust it; the run then fails closed.
- The selector file is at most 4096 bytes (the root provisioner's bound). A
  batch whose selector is larger is refused.
- Other bounds are far away: 100 changed files in one page of the pull request
  (a batch of 6 has 12 or 13), 1,024 files in one install transaction.

### What one batch costs

| | One unit per candidate, `k` units | One batch of `k` units |
|---|---|---|
| Dispatches | `k` | 1 |
| `production-signing` approvals | `k` | 1 |
| Signing runs | `k` | 1 (validation time grows with `k`) |
| RuleSpec CI runs on a signed head | `k` | 1 |
| Merges, each on the newest `main` | `k` | 1 |

### Before the switch goes on

From the independent review of this lane. Each item is necessary; the state is
of this document's commit.

1. **Import refusal through the real resolver.** Done (above).
2. **Workflow packaging.** Not done here. `targeted-signed-reencode.yml`
   records one of the `k` manifests: it expects the citation of the `citation`
   input only, skips manifests of other citations, requires exactly one
   manifest for that citation, and runs the tool and exact-ref check and
   writes `context-manifest.json` for it alone. The commit message, the title
   and body of the lane pull request and `metadata.json` name that one
   citation. The packaging must expect, check and record all `k` manifests, or
   fail closed on a manifest of an unexpected citation. It is a separate
   change to the workflow.
3. **A hosted canary batch** on a real candidate. No hosted run of a batch has
   been made.
4. **An engine-level check** that the other, still unsigned, units in the
   checkout cannot make a unit pass complete-source validation. The import
   refusal covers direct imports; it does not prove that the engine reads no
   other unit.
5. **The derived index.** Decide whether its content must reflect exactly the
   `k` units. Today nothing in this lane checks the content of the index, for
   a batch as for a single unit.

### What is not built

- Nothing dispatches a batch automatically, nothing merges it, and there is no
  evidence gate (reviewer verdict, oracle result) in this lane.
- A limit above 6 units (see "Limits").

## Rollout

Merge this one reusable encoder implementation only after independent review,
required signing-code-owner approval and green CI. Existing pending candidate
admission PRs can continue to merge and sign through the static lane meanwhile;
Arizona encoding must not wait for this rollout. A production canary on an
approved exact-head candidate is required before calling the new lane operational.
