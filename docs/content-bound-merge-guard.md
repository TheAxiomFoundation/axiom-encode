# Content-bound merge guard

Status: **design, not decided; nothing is active.** This document and the
evaluator in `src/axiom_encode/content_merge_guard.py` implement no gate. No
workflow runs the evaluator, no check run is written, no GitHub App exists, no
branch protection changes, and no manifest reader or writer knows the schema the
evaluator parses. Every promotion still uses the order in
`docs/reviewed-pr-promotion.md`. Tracking issue: #2172.

The binding requirements are the "Maintainer decisions (2026-11-12)" in
`docs/reviewed-pr-promotion-order-proposal.md`. The design below is a proposal
for how to meet them. Platform claims that were not confirmed are marked
**[check docs]**.

## 1. What exists today

Verified against `main` at `eb98f8d7` (after #2194).

- **Admission** (`src/axiom_encode/reviewed_pr_admission.py`):
  - Enrolled identities by numeric id: `TRUSTED_SIGNING_REVIEWERS`.
  - Fixed origin, no proxy, no redirects, bounded reads: `github_json`,
    `_pages`.
  - Complete tree delta: `_leaf_entries` and `_require_immutable_delta`.
  - Effective reviews: `_approved_reviews`. It requires an approval on the exact
    head, excludes the PR author by id and the dispatcher by login, and refuses
    on a `CHANGES_REQUESTED` from any account.
- **Approvals are checked at three points**: twice in
  `cmd_promote_reviewed_candidate` (`revalidate_admission` before validation and
  before signing), and in `scripts/prepare_reviewed_pr_promotion.py
  verify-publication` before publication.
- **The preflight runs inside the gated job**: `prepare_reviewed_pr_promotion.py
  prepare` is a step of job `encode` in `targeted-signed-reencode.yml`, which
  holds `environment: production-signing`.
- **Manifest schema**: one constant, `APPLIED_ENCODING_MANIFEST_SCHEMA =
  "axiom-encode/applied-rulespec/v5"` (`cli.py`). Readers compare by equality.
  The promotion contract is `_REVIEWED_CANDIDATE_MANIFEST_FIELDS`, with a
  per-file waiver variant.
- **`notary-guard`** (`notary/merge_guard.py`) is a generated-file lineage guard
  for activated notary lanes. It returns 78 when `.axiom/notary/consumer.json` is
  absent at the base. It never reads reviews. The new guard is a sibling, and
  `merge_guard.py` stays unchanged.
- **`rulespec-us` main protection** (read live on 2026-11-12): required check
  `validate / validate` with `app_id: 15368` (GitHub Actions), `strict: true`,
  `enforce_admins: true`, `required_approving_review_count: 0`,
  `require_code_owner_reviews: false`, `dismiss_stale_reviews: false`.
- **The required check is not publisher-bound today.** Any Actions job named
  `validate / validate` satisfies app 15368, and the workflow that produces it
  runs on `pull_request`, so the pull request's own copy of the caller runs.

## 2. Binding decisions and where each lands

| # | Decision | Design element |
|---|---|---|
| D1 | "The signing service may sign validated, unreviewed candidate bytes when a versioned signed manifest explicitly records that scope. Verifiers must distinguish validation from owner approval." | §5: schema v6 with `attestation_scope` |
| D2a | "Its code and the enrolled identities load from protected policy." | §4: guard code and identity list on encoder main |
| D2b | "The required check is bound to its trusted publisher." | §4: dedicated GitHub App, required with its `app_id` |
| D2c | "It checks the complete final PR delta, canonical paths, approved content hashes, live effective reviews and current-head CI." | §3 steps 1-7 |
| D2d | "It is deployed and enforced before the pre-sign approval is removed." | §8: changes 1-5 before change 6 |
| D2e | "An approval is retained across base-only updates only when the approved content and the permitted scope stay unchanged." | §3 step 5 |
| D2f | "The final approver is independent of both the candidate author and the authenticated dispatcher." | §3 step 6 |
| D3 | "enrolled identities at first. An author may dispatch their own candidate. A different enrolled owner gives the final approval." | §3 step 6; §5 |
| Seq. | "The merge gate of decision 2 is built and passes adversarial tests." | §7, §8 |

## 3. Binding an approval to content

Three options were considered.

| Option | Verdict |
|---|---|
| (a) Tree comparison: an approval on commit C counts if the guarded blobs at C equal those at the current head | **Recommended; implemented by the evaluator** |
| (b) Approver states digests in the review body | Rejected: an agent can paste them, humans mistype them, and it adds no binding that (a) lacks |
| (c) Separate signed approval artifact | Rejected for v1: needs per-owner keys and custody, and would run under the same sessions as the GitHub identity. Revisit if the §9 residual risk is judged too high |

### Algorithm `evaluate(repository, pr, fetch=...)`

This is what `content_merge_guard.evaluate` does. `repository` is a short name
such as `rulespec-us` or `rulespec-us-tn`. Every read is live and goes through
the injected `fetch`. Any error, truncation or disagreement gives `fail`. The
function never raises. A verdict is `pass`, `fail` or `not_applicable`, with a
reason code and evidence.

1. **PR identity.** Read the PR and take its head `H` and branch. Reuse
   `_pr_identity`: open, unmerged, same-repository head, base `main`. Let `B` be
   the live `main` tip. `compare(B...H)` must be `ahead` with merge base `B`.
2. **Complete delta.** Read the complete trees of `B` and `H` with
   `_leaf_entries`. `Δ(H)` is their leaf delta.
3. **Applicability, from paths and modes only.** A label inside a file that the
   PR supplies never makes the guard stand aside. A delta path is *guarded* when
   it is:
   - under a RuleSpec content root. The roots are the encoder's
     `RULESPEC_ATOMIC_ROOTS` (`legislation`, `policies`, `regulations`,
     `statutes`), at the repository root or one level below a jurisdiction
     directory;
   - at or under `.axiom/encoding-manifests`, or at `.axiom`;
   - a top-level leaf that is not a regular `100644` blob (a symlink or a
     submodule there can alias a jurisdiction directory).

   Then:
   - No guarded path: `not_applicable` (`no_guarded_content`). A PR that edits
     only `.github/**` is in this class; see §9.
   - A guarded head leaf that is not a regular blob, or any leaf at `.axiom` or
     at `.axiom/encoding-manifests` itself: `fail`.
   - Guarded content with no changed manifest: `fail`.
   - A removed manifest: `fail`.
   - More than `MAX_MANIFEST_BLOBS` (256) changed manifests: `fail`.
   - A changed manifest that is not one JSON object, has a duplicate key or a
     non-finite number, or whose `schema_version` is neither v6 nor a known
     earlier schema (v5 today): `fail`.
   - **Every changed manifest is v5** and every changed content path is listed
     in `applied_files` of one of them: `not_applicable` (`earlier_lane_v5`).
     Those lanes keep their own controls: the approval is checked before
     signing. The guard does not check their scope (open question 12). Content
     outside the manifests' `applied_files`: `fail`.
   - v6 together with v5: `fail`. Two or more v6 manifests: `fail`.
   - Exactly one changed manifest and it is v6: continue. A repository that is
     not `rulespec-<country>`: `fail`. A draft PR: `fail`.
4. **Scope and canonical paths.** `M` is the v6 manifest.
   - `_require_immutable_delta` cross-checks the tree delta against the PR file
     list and the compare file list.
   - `M` has exactly the v6 shape of §5.
   - `manifest.attestation_scope.candidate` names this repository, this PR and
     the live PR author id, and `candidate.head_sha` equals
     `candidate_rulespec_ref`.
   - `P` is `citation_rulespec_path(manifest.citation)`, `T` is its companion,
     and `M` is at the path `_applied_encoding_manifest_path(P)` gives.
   - `Δ(H)` is exactly `{P, T, M}`, all added. Any other path gives `fail`,
     including `.github/**`, CODEOWNERS, a second module and the derived index
     `.axiom/index/provisions_to_rules.json`. The live lane still permits the
     index (`reviewed_pr_admission.INDEX_PATH`), because RuleSpec repositories
     tracked it when that lane was built. `rulespec-us` stopped tracking it
     (rulespec-us#2152), and an index outside the approved set could change
     after the approval, so the guard is stricter.
   - `sha256(P)` and `sha256(T)` equal `manifest.applied_files`. Each blob read
     is checked against its Git object id.
   - `candidate.head_sha` is a commit of this PR, and `P` and `T` at that commit
     are the blobs at `H`. `candidate.base_sha` is `B` or an ancestor of `B`.
5. **Content-bound approval.** Take the effective review per reviewer (the
   latest of `APPROVED`, `CHANGES_REQUESTED`, `DISMISSED`). A review from a
   deleted account is ignored. An `APPROVED` review on commit `C` counts only if
   all hold:
   - `C` is in the PR's commit list.
   - With `B_C = merge-base(C, main)`, the path set of `Δ(C)` equals
     `{P, T, M}`.
   - The type, mode and Git blob id of `P`, `T`, `M` at `C` equal those at `H`.

   Consequences:
   - An approval made before signing never counts, because `M` is missing at
     `C`.
   - "Update branch" (a merge of `main`) that leaves the three blobs unchanged
     keeps the approval, when someone other than the approver makes it.
   - Any byte change to `P`, `T` or `M`, including a re-sign, voids it.
   - A rebase removes the candidate commit from the PR, so it needs a new
     signing run.
6. **Independence.** The run `manifest.run_id` must be a run of
   `TheAxiomFoundation/axiom-encode`, `targeted-signed-reencode.yml`,
   `workflow_dispatch`, `head_branch` `main`, concluded `success`, and its
   `head_sha` must be the encoder `main` tip or an ancestor of it. Its
   `triggering_actor` must equal `attestation_scope.dispatcher` by id and login,
   and that identity must be enrolled (D3). The reviewer must satisfy all of:
   - id in the enrolled list, matching login, `type == "User"`;
   - id differs from the PR author;
   - id differs from `dispatcher.id`, and from `actor.id` and
     `triggering_actor.id` of the run;
   - id differs from the GitHub-attributed author and committer of **every**
     commit of the PR. This check is best-effort, since Git author metadata is
     free text. There is no bot exemption: the signing workflow's manifest
     commit is attributed to the dispatcher's account with GitHub's `web-flow`
     as committer (observed on rulespec-us#2045, commit `9be9d91e5`), and neither
     can approve.
7. **Blocking reviews and CI.**
   - An effective `CHANGES_REQUESTED` from an enrolled identity, or from a
     reviewer whose `author_association` is OWNER, MEMBER or COLLABORATOR, gives
     `fail`. Today's admission rule (any account) lets any GitHub account block a
     public PR.
   - Every check run named `validate / validate` on `H` must be `success` from
     app 15368. Its Actions run must have event `pull_request`, path
     `.github/workflows/repository-checks.yml`, `head_sha == H`, this repository
     as repository and head repository, and this PR with head `H` in its
     `pull_requests`. Step 4 proves `.github/**` at `H` equals the base, so a
     `pull_request` run of `H` used the protected workflow definitions. That
     holds only for a `pull_request` run: a `workflow_dispatch` run can use
     another ref's definitions, so it is refused.
8. **Re-read.** Read the PR, the `main` tip and the reviews again. If anything
   changed, `fail`, and the next evaluation decides.

Pass evidence: PR number, head, base, the sha256 of `P`, `T` and `M`, the review
id, the reviewer id, the dispatcher id, the run id and the candidate commit.

**The guard does not verify the manifest signature** (open question 3). Until
it does, the binding of `M` to the signing service rests on current-head CI
(which verifies the signature) plus the run and candidate checks of steps 4 and
6. The manifest does not record a run attempt, so the guard cannot tell which
attempt of the run signed it.

### Known limits

- A PR with no guarded content is never judged, also when it edits workflows.
- A v5-lane PR is `not_applicable`; its extra files are not judged.
- A fork PR and a stale PR get `fail`, not `not_applicable`.
- 1000 reviews on one PR, also comment-only reviews from outside accounts,
  exhaust the page bound and give a permanent `fail`. The remedy is a new PR.
- A re-run of the signing run by another person changes `triggering_actor` and
  gives `dispatcher_mismatch` permanently. The remedy is a new signing run.
- An approver who clicks "Update branch" becomes a commit author and can no
  longer approve.
- A v5 lane whose manifest does not list every changed content path in
  `applied_files` gets `fail`. Whether the retire and path-migration manifests
  list them was not checked.

### CLI

`axiom-encode merge-guard --repository rulespec-us --pr <n>` prints one JSON
line (`schema`, `verdict`, `reason`, `detail`, `evidence`). Exit codes: 0 pass,
1 fail, 3 not_applicable, 2 usage error.

### Where the implementation differs from the first design, and why

The code at `main` wins where the two disagree.

- The first design decided applicability from the manifest's `schema_version`.
  A PR controls that label, so applicability now comes from paths and modes.
- `_require_immutable_delta` refuses every changed non-regular blob, so it
  cannot classify a maintenance PR. The evaluator computes a mode-agnostic
  delta from `_leaf_entries` first and calls `_require_immutable_delta` only for
  a v6 unit.
- `_pr_identity` takes a selector that already names the head. The evaluator
  reads the head from the PR and passes a four-field target. `_pr_identity`
  refuses a fork head.
- `_approved_reviews` cannot be reused: it requires the exact head, refuses on
  any account's `CHANGES_REQUESTED`, and binds the dispatcher by login. The
  evaluator has its own effective-review code and binds by numeric id.
- Admission requires `pr.base.sha` to equal the `main` tip. The evaluator uses
  the live `main` tip as `B` and does not rely on `pr.base.sha`.
- Admission reads files through the contents API. The evaluator reads
  `git/blobs/<id>` and recomputes the object id from the bytes.
- Added beyond the first design: the index is excluded; every commit author is
  excluded; the candidate and base named by the manifest are bound; the
  dispatcher must be the run's `triggering_actor` and enrolled; the CI run and
  the signing run are bound to their repository, event and commit; duplicate
  JSON keys are refused.
- The promotion lane maps a jurisdiction to `rulespec-<country>`
  (`PromotionSelector.repository`). The evaluator accepts
  `rulespec-<country>-<subdivision>` names for applicability, and a v6 unit
  there gives `fail`.
- The default `fetch` is the anonymous `github_json`. A recurring check needs a
  token-capable bounded reader (for example `notary.remote.GitHubReader`); the
  change that adds the runner injects it. The evaluator has no client of its
  own.
- Line numbers in the first design were taken at `02927526`; `cli.py` has moved
  since (#2194), so this document names functions and constants instead.

### Review lifecycle

| Event | Result |
|---|---|
| Approval dismissed | That review becomes `DISMISSED`; the next evaluation gives `fail` |
| `CHANGES_REQUESTED` after a pass | Same: the next evaluation gives `fail` |
| Re-approval | A new review on the new `C`; evaluated as above |
| Push that changes content | The new head has no verdict; old approvals fail step 5 |
| Base-only update by merge | New head, same blobs: the approval is retained |
| Rebase | The candidate commit leaves the PR: `fail` until a new signing run |

## 4. Where the guard would run and why it would be trusted

None of this exists. It is the proposal for changes 3 and later (§8).

- **Code**: `src/axiom_encode/content_merge_guard.py` and the CLI command
  `merge-guard`, with CODEOWNERS entries like those for the admission files.
- **Identities**: the existing `TRUSTED_SIGNING_REVIEWERS` on encoder main. A
  RuleSpec PR cannot change it.
- **Runner**: a new workflow in axiom-encode that refuses to run unless
  `github.ref == refs/heads/main`, with an environment restricted to branch
  `main` that holds the guard App's private key.
- **Publisher binding**: a dedicated GitHub App installed on `rulespec-*` with
  `checks: write`, `pull_requests: read`, `contents: read`, `actions: read`.
  Branch protection requires the guard's check with that App's `app_id`.
  Workflows in `rulespec-us` publish as app 15368 and cannot satisfy it.
- **Triggers**: `schedule` (a sweep over open PRs to `main`) and
  `workflow_dispatch` (country and PR number). No trigger carries authority. An
  evaluation always reads live state, so a forged trigger yields an honest
  verdict, and a missing one leaves the required check absent, which blocks the
  merge.
- **Why not run it in `rulespec-us`**: under `pull_request` the PR controls the
  workflow file. Under `pull_request_target` the file is trusted, but its check
  still publishes as app 15368, which any PR workflow can imitate.
- **Alternative**: an organisation ruleset with "Require workflows to pass
  before merging". It needs no App. Plan availability, supported events and
  re-run on review are unconfirmed **[check docs]**.

## 5. Manifest and encoder changes (planned, not in this change)

### Manifest (D1)

Schema `axiom-encode/applied-rulespec/v6`, admitted only for
`tool == "axiom-encode promote-reviewed-candidate"` with `backend` absent. The
field set is `_REVIEWED_CANDIDATE_MANIFEST_FIELDS` (or its per-file waiver
variant) with `reviewed_rulespec_ref` replaced by `candidate_rulespec_ref`, plus:

```json
"attestation_scope": {
  "schema": "axiom-encode/attestation-scope/v1",
  "validation": "encoder-main-complete-source",
  "owner_review": "none",
  "origin": "local-candidate",
  "candidate": {"repository": "TheAxiomFoundation/rulespec-us",
                "pull_request": 2044, "head_sha": "<40 hex>",
                "base_sha": "<40 hex>", "author_id": 0},
  "dispatcher": {"id": 0, "login": "<login>"},
  "merge_requirement": "axiom-encode/content-bound-merge-guard/v1"
}
```

The evaluator parses exactly this shape and treats anything else as `fail`. A
test pins its field sets to the live v5 promotion contract in `cli.py`. The
shape has no run attempt; adding `run_attempt` would let the guard bind one
(open question 13).

Still to do in later changes: verifiers accept `{v5, v6}` and require the
block's exact shape; the promotion writer emits v6. A verifier pinned before
that change rejects v6, so no v6 manifest can pass `validate / validate` until
`rulespec-us` bumps its encoder pin. That is a built-in ordering gate.

### Encoder (behind a switch that the activation change flips)

- Admission keeps every scope check of `verify_reviewed_pr` and drops the
  requirement for an approval.
- `triggering_actor.id` must be enrolled (D3), and the numeric id is recorded.
  Today only the login is bound.
- The admission schema, `revalidate_admission`, the runtime attestation key and
  `cmd_promote_reviewed_candidate` change so that an auditor cannot mistake a
  scope-checked candidate for a reviewed one.
- The preflight step moves to a job that runs before the gated `encode` job.

## 6. Merge queue and `notary-guard` (not implemented)

`evaluate_merge_group` is a stub that returns `fail`.

- A required check must also be reported on the merge-group commit. The planned
  algorithm computes the delta of the group head against its queue base, treats
  every v6 manifest as a unit `{M, P, T}`, and requires each unit to map to a
  queued PR with the same three blob ids for which `evaluate` passes.
- rulespec-us#2160 plans a group size of 1, which is recommended for the pilot.
- **[check docs]** How to list the PRs in a merge group, and whether an
  App-created check run on the group SHA satisfies the queue.
- `notary-guard` is independent. On a lane with a base `consumer.json`, lineage
  replaces apply manifests, so this guard reports `not_applicable`.

## 7. Adversarial test plan

`tests/test_content_merge_guard.py` holds the cases as fixture-driven unit
tests over an injected `fetch`. "Skipped" means a skipped test that states its
reason; it is not a pass.

| # | Attack | Expected verdict | Test |
|---|---|---|---|
| 1 | Approval placed before signing (no manifest at `C`) | fail | yes |
| 2 | Approval at `C`, then the primary changes by one byte | fail | yes |
| 3 | Approval at `C`, then a re-sign produces new manifest bytes | fail | yes |
| 4 | Approval, then "update branch" merge, three blobs unchanged | pass | yes |
| 5 | As 4, but the merge commit also adds `.github/workflows/x.yml` | fail (scope) | yes |
| 6 | PR author is the only approver | fail | yes |
| 7 | Dispatcher is the only approver, including a re-run where `triggering_actor` differs from `actor` | fail | yes |
| 8 | Approver is a write collaborator who is not enrolled | fail | yes |
| 9 | Enrolled login with a different numeric id (recycled login) | fail | yes |
| 10 | Approval later dismissed | fail | yes |
| 11 | Enrolled A approves, enrolled B requests changes | fail | yes |
| 12 | Approver later submits a `COMMENTED` review | pass | yes |
| 13 | Primary is a symlink, mode `100755`, or a submodule | fail | yes |
| 14 | Primary at a non-canonical path for the citation | fail | yes |
| 15 | PR adds a workflow with a job named like the guard's check | merge blocked: wrong `app_id` | skipped: needs the App binding |
| 16 | PR rewrites `repository-checks.yml` so `validate / validate` passes | fail (scope) | yes for a PR with guarded content; skipped for a PR with none (§9) |
| 17 | Truncated tree, an 11th page of reviews, a redirect, or HTTP 403/500 | fail; no cached pass | yes, for every read of a passing evaluation |
| 18 | Review dismissed between the first and second read | fail | yes |
| 19 | v6 manifest whose `candidate.pull_request` names another PR | fail | yes |
| 20 | PR containing only a v6 manifest, or two v6 manifests | fail | yes |
| 21 | Manifest `applied_files` digest differs from the head blob | fail | yes |
| 22 | `validate / validate` green on an older head, pending on `H` | fail | yes |
| 23 | Modification of a module that already exists on `main` | fail | yes |
| 24 | Merge group with two units, one unapproved | fail for the group | skipped: merge group is a stub |
| 25 | PR with no guarded content (maintenance) | not_applicable | yes |
| 26 | Unparseable JSON at a manifest path in the delta | fail | yes |
| 27 | Enrolled owner authored the commits; a third account opened the PR; the owner approves | fail when GitHub attributes the commits; documented gap when it does not | yes, both halves |

Added by security review round 1:

| # | Attack | Expected verdict | Test |
|---|---|---|---|
| 28 | Content plus a rewritten workflow, with no manifest | fail | yes |
| 29 | Manifest relabelled to an unknown schema (`v7`, missing, not a string) | fail | yes |
| 30 | `.axiom`, the manifest directory, a jurisdiction or a content root replaced by a symlink, submodule or executable | fail | yes |
| 31 | v5 and v6 manifests in one PR; a removed manifest; v5 manifest that does not cover the content | fail | yes |
| 32 | `validate / validate` green from a `workflow_dispatch` run, another repository or another PR | fail | yes |
| 33 | Signing run from another repository, or whose commit is not on encoder `main` | fail | yes |
| 34 | Manifest names a candidate commit outside the PR, or one with other content, or a base that is not an ancestor | fail | yes |
| 35 | Approver authored or committed any commit of the PR, also the manifest commit | fail | yes |
| 36 | Approval on `H`, then a new head that changes only the derived index | fail | yes |

Cases 15-18 must also run live in the shadow period (§8).

## 8. Delivery plan

Implementation comes strictly before activation.

| # | Repo | Change | Rollback |
|---|---|---|---|
| 1 | axiom-encode | This change: evaluator, `merge-guard` CLI, tests, CODEOWNERS, this document. Nothing calls it | revert |
| 2 | axiom-encode | Verifiers admit `applied-rulespec/v6` with the attestation scope; no writer | revert; no v6 exists |
| 3 | axiom-encode | Workflow that runs the guard from `main` and publishes its check, with a dry-run mode | disable the workflow |
| 4 | rulespec-us | Pin the encoder that verifies v6 | revert the pin |
| 5 | axiom-encode | Emit v6 and admit without approval, behind a default-off switch | revert |
| 6 | axiom-encode | **Activation**: turn the switch on; update `docs/reviewed-pr-promotion.md` | flip the switch back; the required check stays |

### Activation checklist

Steps marked **M** need repository or organisation settings. Only a maintainer
can do them.

1. **M** Create the guard App, install it on `rulespec-us`, and store its key in
   an axiom-encode environment restricted to branch `main`.
2. Merge changes 1-4. Publish the check in shadow mode: visible, not required.
3. Run cases 15-18 live on a throwaway PR and confirm every verdict.
4. **M** Add the guard's check with the guard App's `app_id` to the required
   checks on `rulespec-us` main. Confirm that maintenance PRs get
   `not_applicable` and can merge.
5. Run the current-order canary (rulespec-us#2045) to completion.
6. Merge change 5 with the switch off.
7. Merge change 6. The first promotion in the new order is the pilot.
8. **M**, optional hardening: enforce code-owner review for `/.github/` (§9).
9. **M**, before the merge queue (rulespec-us#2160): confirm the guard reports
   on group SHAs in a test repository.

Steps 4 and 7 are separate reviewed changes, and step 4 comes first. That
satisfies D2d.

## 9. Threat model

| Adversary | Defended? | How |
|---|---|---|
| Write collaborator, not enrolled, on a PR with guarded content | Yes | Cannot dispatch, cannot give a counting approval, cannot publish as the guard App; a v6 unit refuses every extra file; content without a manifest fails |
| PR that edits workflow files and carries a v6 unit | Yes | Scope check (steps 3 and 4) |
| PR that edits workflow files and has no guarded content, or is a v5-lane PR | **No** | The guard gives `not_applicable`. Pre-existing gap in the `validate / validate` binding, outside this guard; step 8 of the checklist mitigates |
| PR that relabels or hides its manifest to avoid the guard | Yes | Applicability comes from paths and modes (step 3) |
| `validate / validate` produced by a run that is not this PR's `pull_request` run | Yes | Step 7 |
| Forged or unsigned v6 manifest | Partly | Current-head CI verifies the signature; the guard does not (open question 3) |
| Second workflow imitating the check name | Yes, once the App binding exists | `app_id` binding |
| Stale approval after a content change | Yes | Blob equality |
| Push racing an approval | Yes | The verdict belongs to one head and is computed from that head's tree |
| Dismissal just after a pass | Partly | The check stays green until the next sweep |
| GitHub API failure | Fails closed for new verdicts | A green check already written is not withdrawn during an outage |
| Repository admin, or theft of the guard App key | No | Out of scope |
| Semantic change brought in by the base while the approved bytes stay the same | Partly | Current-head CI catches it; the human approval does not cover it |
| Commit authored by the approver but not attributed by GitHub | No | Git author metadata is free text (case 27) |

**Residual risk: agents acting under the two enrolled identities.** The guard
authenticates a GitHub account, not a person. If agent sessions hold both
owners' credentials, an agent can author as A, dispatch as A and approve as B,
and every check passes. The guard reduces this to "both accounts must act" and
records who did. It cannot prove that a human read the bytes. Closing this would
need approval credentials that agents cannot hold, for example a hardware-backed
signed approval (option c in §3) or a separate approver account.

## 10. Open questions

| # | Question | Recommendation |
|---|---|---|
| 1 | App-published check or an organisation required-workflow ruleset? | The App. Switch only if the ruleset documentation confirms the capability |
| 2 | Is a polling sweep (about 5 minutes) acceptable as the trigger? | Yes for the pilot |
| 3 | Should the guard verify the manifest signature itself, or rely on current-head `validate / validate`? | Rely on CI for v1, since step 4 pins the workflow definitions |
| 4 | Rename `reviewed_rulespec_ref` to `candidate_rulespec_ref` in v6? | Yes: a v6 manifest must not say "reviewed" |
| 5 | Also set `required_approving_review_count: 1`? | No: it counts any write collaborator and stale approvals |
| 6 | Do notary lanes need this owner-review gate? | Decide with the notary activation |
| 7 | Exclude the approver when they are a commit author but not the PR author? | Yes, best-effort (case 27) |
| 8 | Merge-queue group size for promotions? | 1 during the pilot |
| 9 | A fork PR, a stale PR and a PR with more than 256 changed manifests get `fail`, not `not_applicable`. Is that acceptable for a required check? | Yes for forks and stale heads (`strict` already requires an up-to-date head). Decide the bound before a bulk manifest migration |
| 10 | Should a `CHANGES_REQUESTED` from an outside account block? | No (step 7); it differs from today's admission rule |
| 11 | `candidate.head_sha` is bound to the PR's commits, so a rebase needs a new signing run. Acceptable? | Yes: the binding is worth more than rebase support |
| 12 | Which verdict for a v5-lane PR: `not_applicable` (today's choice), or a scope check like the v6 one? | Decide before the check is required. `not_applicable` leaves the v5 lanes exactly as they are |
| 13 | Which verdict for RuleSpec content with no manifest, a removed manifest, or a v5 manifest that does not list the changed content: `fail` (today's choice) or `not_applicable`? | `fail`, after a census of the retire, path-migration and waiver PR shapes; CI already rejects unmanifested content |
| 14 | Add `run_attempt` to `attestation_scope`? | Yes, in the change that defines the v6 writer |
| 15 | The approver may not be the author or committer of any commit, so the approver cannot click "Update branch". Acceptable? | Yes for the pilot |

## 11. What was not verified

Verified against live data on 2026-11-12 (rulespec-us#2045 and its signing run):
the fields of an Actions run (`event`, `path`, `head_sha`, `head_branch`,
`repository`, `head_repository`, `pull_requests`, `actor`,
`triggering_actor`), the `details_url` form of a check run, the v5 promotion
manifest's `applied_files` paths (repository-relative), and the attribution of
the signing workflow's manifest commit.

Not verified:

- How many check runs carry the exact name `validate / validate` on a green
  head. On a red head there were two (one `skipped`, one `failure`); the matrix
  jobs carry a suffix. The evaluator requires every one to be `success`.
- Whether `pull_requests` on an Actions run stays populated after the PR's
  branch moves, and for a re-run. It is empty for fork PRs, which the guard
  refuses anyway.
- Whether a required check with a non-Actions `app_id` is honoured on
  merge-group commits, and how to list a group's PRs.
- Which workflow file version `pull_request_review` and `merge_group` runs use.
- Organisation plan and availability of required-workflow rulesets.
- Whether the Go signer or supervisor inspects `schema_version` or field names.
  Their tests mention `applied-rulespec/v5`; the code was not read.
- Whether workflow-run metadata for `manifest.run_id` stays readable for the
  life of a PR. A deleted run makes step 6 fail closed.
- The shapes of v5 retire, path-migration and legacy manifests, and so whether
  those lanes get `not_applicable` or `fail` (open question 13).
- The content layout of `rulespec-<country>-<subdivision>` repositories. The
  evaluator assumes the content roots sit at the root or one level down.
- Whether `pr.base.sha` follows the `main` tip. The evaluator does not use it.
- Tree size: a recursive tree that GitHub truncates gives `fail` for every PR
  of that repository.
- The guard App does not exist; its id, installation and key custody are
  assumptions.
