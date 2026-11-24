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

## Candidate requirements: the source pin

A promotion signs the candidate bytes as they are; it does not stamp them. So
the candidate primary must already be bound to its source:

```yaml
module:
  source_verification:
    corpus_citation_path: <the promotion citation>
    source_sha256: <SHA-256 of the resolver's stored source body>
```

Signing run 45328443323 failed after its approval because the candidate had
only `corpus_citation_path`. Three checks now refuse such a candidate, the
first two before any approval is used:

1. **Before the environment approval**, in the unprotected `attempt_budget`
   job: `scripts/check_candidate_source_pin.py` reads the candidate primary at
   the exact `rulespec_ref` and refuses a promotion dispatch whose primary does
   not name the citation and carry a 64-character lowercase hex
   `source_sha256`. It fails closed, also when GitHub cannot be read; the
   attempt budget treats the run as neutral. Other dispatches pass untouched.
2. **At admission** (`verify_reviewed_pr`, under both admission bases): the
   same read of the primary bytes that the admission binds. The refusal names
   the command below.
3. **In the protected job**, unchanged: the module is parsed and the pin is
   compared with the resolver's attestation for the pinned corpus. Only this
   check proves that the pin is the right one.

Checks 1 and 2 are a strict line read without a YAML parser and without the
corpus. They accept one shape, the plain block form above (one `key: value`
line per entry, plain or quoted one-line values), and refuse every other shape:
flow style, anchors, aliases, tags, merge keys, duplicate keys, tab
indentation, multi-line values, document markers. They do not prove that a
YAML parser reads the same values, and they cannot admit anything: a candidate
that passes them still has to pass check 3.

### Where the pin comes from

- `encode` without `--apply` binds the candidate before deterministic
  validation and before the reviewers, so the validated, reviewed and
  promotable bytes are the same bytes. The binder inserts or replaces only the
  two lines (it creates the block when the module has none) and keeps every
  other byte; it never re-dumps the YAML. A candidate that cannot be bound is
  not clean: the try fails with the `source-binding/unbound` issue, and the
  outcome records `source_binding: unbound` (status `source_unbound`).
- `encode --apply` keeps its own stamp, unchanged.
- An existing candidate is bound with the command below.

### Bind an existing candidate

```bash
uv run axiom-encode bind-source-verification \
  --repo ../rulespec-us \
  --citation us-az/manual/des/faa5/dependent-care-expense/block-5 \
  --corpus-path ../axiom-corpus
```

Run it in a checkout of the candidate branch. It resolves the attestation with
the resolver and the corpus pin of that checkout, as the promotion does, and
edits only the module at the citation's canonical path. It prints the module
path, the corpus release, the attested `source_sha256`, the SHA-256 of the
file before and after, and the changed line (`diff=+    source_sha256: ...`).
`--check` writes nothing and exits 1 when the module is not bound. The command
calls no model, writes nothing to the network and signs nothing; commit and
push the change yourself. Outside the protected runtime, pass the public trust
roots as for a verification-only local encode (`--corpus-release-public-key`,
`--apply-public-key`, `--eval-public-key`).

It refuses, and writes nothing, when:

- the module names another source: a `corpus_citation_path` that the
  attestation does not bind, or a `source_sha256` that differs from the
  attested one (`status=refused:conflict`). A different pin is never
  overwritten; decide first which source the candidate was encoded from;
- the block is in a shape the minimal edit does not handle: flow style, an
  anchor or alias, a merge key, a duplicate key, a multi-line value, a line
  break other than LF or CRLF (`status=refused:unsupported_shape`; the message
  names the shape). Rewrite the block by hand in the plain block form.

A bound candidate has a new head, so its PR needs a new dispatch on that head.

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

The number above is illustrative, not an admission. The candidate primary must
carry its source pin (see "Candidate requirements" above), or the dispatch is
refused before the approval. All remaining generation,
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
