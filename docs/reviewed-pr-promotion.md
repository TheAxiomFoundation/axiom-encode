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
only `corpus_citation_path`. Three checks now refuse such a candidate. Only
the first runs before the `production-signing` environment approval:

1. **Before the environment approval**, in the unprotected `attempt_budget`
   job: `scripts/check_candidate_source_pin.py` reads the candidate primary at
   the exact `rulespec_ref` and refuses a promotion dispatch whose primary does
   not name the citation and carry a 64-character lowercase hex
   `source_sha256`. It fails closed, also when GitHub cannot be read; the
   attempt budget treats the run as neutral. Other dispatches pass untouched.
2. **At admission** (`verify_reviewed_pr`, under both admission bases): the
   same read of the primary bytes that the admission binds. Admission runs in
   the protected `encode` job, so after the environment approval; it is the
   same check a second time, on the bytes the admission record names. The
   refusal names the command below.
3. **In the protected job**, unchanged: the module is parsed and the pin is
   compared with the resolver's attestation for the pinned corpus. Only this
   check proves that the pin is the right one for that corpus.

Checks 1 and 2 are a strict line read without a YAML parser and without the
corpus. The rule is positive: every top-level line is a plain `name:` key or
a list item and `module:` occurs once; every key directly under `module` is a
plain `name:` key and `source_verification:` occurs once; every entry of the
block is one `name: value` line with a one-line plain or quoted value. Through
the end of the `module` block the reader follows block scalars and quoted or
plain scalars that continue over lines, so text inside them is not read as a
key. It refuses every other shape: a quoted, tagged, anchored or complex key,
a merge key, an alias, flow style on these mappings, tab indentation, a
document marker (also a leading `---`), a comment on the `module:` or
`source_verification:` line, an unquoted value that a YAML reader could take
for a number. They cannot admit anything: a candidate that passes them still
has to pass check 3.

The binder and the command use the same reader as their last step. A module
that carries the right values for a YAML parser but is not in the plain block
form is refused as `not_plain_form`; it is never reported as bound. So "the
command says bound" implies "admission reads the same pin".

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
  --corpus-path ../axiom-corpus \
  --expect-source-sha256 <source_sha256 of the encode run's source attestation>
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

**The pin attests the corpus the checkout pins now, not the corpus of the
encode.** A candidate that was encoded at one corpus release, on a branch that
later took a `main` with another release whose source body changed, would get
the new body's digest and pass the protected check, although it was encoded
from the old text. `--expect-source-sha256` closes this: give it the
`source_sha256` that the encode run recorded (the `source_attestation` in the
run's `source-metadata.json` or context manifest, or the value in the
candidate PR). The command refuses with `status=refused:source_changed` unless
the checkout's corpus still attests exactly that body. Use it for every repair
of an existing candidate. Without it the command still works, and prints a
`warning=` line that names the corpus release; compare the printed
`attested_source_sha256` with the run record yourself before you push. If they
differ, encode the candidate again; do not re-pin it.

It refuses, and writes nothing, when:

- the source changed since the encode (`status=refused:source_changed`);
- the module names another source: a `corpus_citation_path` that the
  attestation does not bind, or a `source_sha256` that differs from the
  attested one (`status=refused:conflict`). A different pin is never
  overwritten; decide first which source the candidate was encoded from;
- the block is in a shape the minimal edit does not handle: flow style, an
  anchor or alias, a merge key, a duplicate key, a multi-line value, a line
  break other than LF or CRLF (`status=refused:unsupported_shape`; the message
  names the shape);
- the values are right but the file is not in the plain block form that the
  promotion lane reads (`status=refused:not_plain_form`; the message names the
  reason, for example a leading `---`). Rewrite the block by hand in the plain
  block form.

A wrong checkout, corpus path or citation ends with `status=error`, one
`error=` line and exit status 2.

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
  one; they do not order or select anything. The commit message, the lane
  pull request and the run's artifact name every unit.
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
   for a single unit, twice: without the files of the other units, then with
   every unit present (below).
4. The admission is read live again, once.
5. Every unit: the signature.
6. One transaction installs every manifest, then verifies every installed
   manifest.

The result is one signed manifest per unit, at the unit's own canonical
manifest path, in the form of a single-unit promotion: the same schema
(`axiom-encode/applied-rulespec/v5`), tool, fields and signature form. With
the clock held, each manifest is byte-identical to the manifest that a
single-unit promotion of that unit writes on the same checkout. One recorded
value describes the checkout and not the unit:
`validation_execution.policy_pre_apply` (`pre_apply_content_sha256`,
`pre_apply_file_count`) is the content of the candidate head, which holds
every unit of the batch, although each unit is validated without the others
(below). A verifier reads each manifest by itself and does not recompute that
value, so it needs no change. The units of one batch share `run_id` and
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

### Each unit is validated twice

Validation of a unit runs in a temporary copy of the checkout (the apply
overlay). A unit of a batch is validated in two passes, and both must pass.

**Pass 1, isolated.** The primary and the companion of every other unit are
removed from the copy before anything is validated, and so is a directory that
held nothing else. The tree is the tree of a candidate that holds this unit
alone. So nothing the unit needs can come from another, still unsigned, unit:
an import, a companion input or output key, a proof import target, an absolute
reference in a formula, a relation target, or any other route finds no file,
and the unit is refused exactly as its single-unit promotion is refused. This
holds by construction, not by a list of reference kinds.

**Pass 2, batch.** The copy is the whole candidate head: the tree that exists
after the batch lands. Some checks are stricter when a related module exists
(a parent that opens an exception list must import an existing child
exception; same-section term and subsection imports). A unit that is valid
alone but is refused beside another unit of the batch stops the batch here.
The error says `with the other units of the batch present`.

The signed manifest records the isolated pass. The second pass is a check
only.

What the rule guarantees, exactly:

- A batch is signed if and only if every unit passes alone (`main` plus the
  unit) and every unit passes in the final tree (`main` plus all units).
- For two units these four verdicts are the four turns of the two possible
  single-unit landing orders. So a batch of two is signed exactly when the two
  units could land one at a time in either order.
- For more units the trees in between (`main` plus some of the units) are not
  validated. If every check is of one of two kinds, "needs a module to exist"
  or "is stricter when a module exists", a verdict that passes in the smallest
  and in the largest tree passes in every tree in between, and the batch is
  signed exactly when the units could land one at a time in every order. The
  checks found in review are of these two kinds. This is not proven for every
  check of the validator.
- It is stricter than "some landing order exists". A unit that needs another
  unit of the batch can land after it, one at a time; the batch is refused
  (pass 1). Promote such units in separate candidates, in order.

Other properties:

- Only the temporary copy is changed. The checkout is never written by this
  step; the final check that only the manifests changed still holds.
- It fails closed: a file of another unit that is not a regular file of the
  copy, or whose path leaves the copy through a link, stops the validation of
  the unit before any file is removed.
- A unit that an existing module depends on cannot be part of a batch. The
  dependent comparison reads a baseline from the checkout, outside the
  isolation, so such a unit is refused in a batch and must be promoted alone.
  (A new module with a dependent on `main` is not expected: the dependent
  would import a module that does not exist.)
- The cost: a batch of `k` units runs `2k` validations (a batch of one runs
  one).
- The record: `metadata.json` of the run carries `promotion_validation`
  (`axiom-encode/promotion-batch-validation/v1`): the passes, and for each
  unit its `hidden_paths`. With `rulespec_base` (the candidate head) the tree
  of each isolated pass can be rebuilt. The signed manifests are unchanged.

### Imports between units of one batch

Refused early, with a clearer message than the isolation gives. A unit whose
primary imports another unit of the same batch stops the run before
validation, and the error names both files. Since each unit is validated
alone, this check is no longer the only barrier; it tells the operator what to
do. Promote the imported unit first, then the importer in a later candidate.

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

1. **No unit passes because of another unit, and no unit is refused beside
   another.** Done: each unit is validated twice ("Each unit is validated
   twice"). The import refusal through the real resolver is the early
   message. Proven on the staging code with the real overlay and a probe
   validator; the engine itself was not run (items 3 and 4).
2. **Workflow packaging.** Done in `targeted-signed-reencode.yml`. For a run
   that names its units, the packaging step expects one changed manifest per
   unit at the unit's canonical position and nothing else: a manifest of an
   unnamed unit, a missing manifest, a deleted manifest, or a manifest whose
   citation is not its unit's fails the run. Every unit gets the tool and
   exact-ref check and the check that its manifest covers exactly its own
   pair. `apply-manifests.json` lists every manifest in path order. The unit
   of the `citation` input keeps `context-manifest.json`; each other unit
   gets `promotion-unit-NN-context-manifest.json`. The one commit, the title
   and body of the lane pull request and `metadata.json` (`promotion_units`)
   name every unit. A run without a unit list packages as before; it still
   ignores a manifest of another citation. Tested by running the workflow's
   embedded code; not yet run on a hosted runner (item 3).
3. **A hosted canary** on real candidates: the first run with the real
   engine. It needs a batch that must be signed, and one batch that must be
   refused because a unit names another unit of the batch. No hosted run of a
   batch has been made.
4. **The engine reads only the overlay.** Confirm with the real engine that it
   reads no path outside the validation overlay and the closure snapshot. The
   two passes control the tree that the engine is given, not what the engine
   opens.
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
