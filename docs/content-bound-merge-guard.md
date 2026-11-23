# Content-bound merge guard

Status: **design, not decided; nothing is active.** This document and the
evaluator in `src/axiom_encode/content_merge_guard.py` implement no gate. No
workflow runs the evaluator, no check run is written, no GitHub App exists, no
branch protection changes. A reader for the v6 manifest schema that the
evaluator parses (`axiom-encode/applied-rulespec/v6`, §5) exists
(`src/axiom_encode/applied_manifest_v6.py`, change 2 of §8) and is switched
off: `V6_READER_ENABLED` is `False`, so every verifier refuses a v6
manifest. Every promotion still uses the order in
`docs/reviewed-pr-promotion.md`. Tracking issue: #2172.

The writer and the admission without approval (change 5 of §8) are present
and switched off by three locks on encoder main (§5, "Encoder"). No lock is
open, so no v6 manifest can be written.
A multi-unit form of the v6 lane (change 5d of §8) is present and switched
off: `MULTI_UNIT_V6_ENABLED` is `False` (§5, "Multi-unit candidates").

The design follows the agent-proposed answers in
`docs/reviewed-pr-promotion-order-proposal.md`. An agent posted those answers
under @MaxGhenis's account; Max has not ratified them. D1, D2a-D2f, D3 and Seq.
identify proposed constraints throughout this design and related source
comments, not maintainer acceptance. Platform claims that were not confirmed
are marked **[check docs]**.

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

<a name="2-binding-decisions-and-where-each-lands"></a>

## 2. Proposed constraints and where each lands

| # | Proposed constraint | Design element |
|---|---|---|
| D1 | "The signing service may sign validated, unreviewed candidate bytes when a versioned signed manifest explicitly records that scope. Verifiers must distinguish validation from owner approval." | §5: schema v6 with `attestation_scope` |
| D2a | "Its code and the enrolled identities load from protected policy." | §4: guard code and identity list on encoder main |
| D2b | "The required check is bound to its trusted publisher." | §4: dedicated GitHub App, required with its `app_id` |
| D2c | "It checks the complete final PR delta, canonical paths, approved content hashes, live effective reviews and current-head CI." | §3 steps 1-7 |
| D2d | "It is deployed and enforced before the pre-sign approval is removed." | §8: changes 1-5 before change 6 |
| D2e | "An approval is retained across base-only updates only when the approved content and the permitted scope stay unchanged." | §3 step 5 |
| D2f | "The final approver is independent of both the candidate author and the authenticated dispatcher." | §3 step 6 |
| D3 | "enrolled identities at first. An author may dispatch their own candidate. A different enrolled owner gives the final approval." | §3 step 6; §5 |
| Seq. | "Build the merge gate of proposed answer 2 and pass its adversarial tests." | §7, §8 |

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
3. **Applicability, from paths and modes only.** The guard stands aside in two
   cases and no other. No label inside a file that the PR supplies is trusted
   for either.
   - Any added or modified leaf that is not a regular `100644` blob, anywhere:
     `fail`.
   - **Allowlist.** Every delta path is exactly one of `README.md`, `LICENSE`,
     `LICENSE-CODE`, `NOTICE`, `.gitignore` at the repository root:
     `not_applicable` (`no_guarded_content`). The list is encoder policy
     (`UNGUARDED_ROOT_FILES`), matched on the whole path. Every other path is
     *guarded*: every jurisdiction directory at every depth and root name,
     `programs/`, all of `.axiom/`, all of `.github/`, the root ledgers
     (`known-*.yaml`, `oracle-coverage-pending.yaml`,
     `household-choices.yaml`), `tools/`, `bulk/`, `tests/`, `.gitattributes`
     (it changes checked-out bytes and review diffs, so CI and the guard would
     no longer judge the same bytes), `CLAUDE.md` and `.claude/` (they steer
     the agent sessions that author, dispatch and approve), and any new
     top-level entry.
   - A guarded delta needs one lane to claim the **whole** delta.
     A manifest is a file under `.axiom/encoding-manifests/` at the repository
     root or under one jurisdiction directory (both layouts exist on
     `rulespec-us` main). No changed manifest: `fail`
     (`no_lane_for_guarded_change`). A removed manifest, more than
     `MAX_MANIFEST_BLOBS` (256) changed manifests, a manifest that is not one
     JSON object or has a duplicate key or a non-finite number, or a
     `schema_version` that is neither v6 nor v5: `fail`.
   - **Earlier lane (v5).** Every changed manifest is labelled v5, and:
     - every other delta path is a RuleSpec module file,
       `<jurisdiction>/<atomic root>/.../<name>.yaml` (or
       `<atomic root>/...yaml` at the root), with no hidden path segment, under
       a jurisdiction that already holds modules on the **base** tree. The
       encoder recognizes `legislation`, `policies`, `regulations` and
       `statutes` as atomic roots. So the delta has no path
       under `.github/`, no other `.axiom/` file, no ledger, no ProgramSpec, no
       allowlisted file and no new jurisdiction. Otherwise `fail`
       (`no_lane_for_guarded_change`);
     - every changed manifest sits at the position of one module of a base
       jurisdiction, in either layout. Otherwise `fail`
       (`no_lane_for_guarded_change`);
     - every changed module file is listed in `applied_files` of the manifest
       at its own canonical position (a `.test.yaml` belongs to its primary),
       with the sha256 of its head bytes; a removed file needs the listing
       only. Otherwise `fail`;
     - every changed manifest lists at least one changed module file, so an
       empty listing, or a manifest-only change, has no lane: `fail`;
     - `validate / validate` is green on `H` from this PR's own `pull_request`
       run, with the binding of step 7. Otherwise `fail`;
     - the PR and `main` did not move during the evaluation, and a final read
       confirms the same green check identities and observed run attempts.

     Then `not_applicable` (`earlier_lane_v5`). **What this verdict rests on:**
     the guard does not verify the v5 signature and cannot tell a signed
     manifest from a hand-written one. It relies on the `validate / validate`
     run, which verifies the signature, and on the lane's scope, which proves
     that the PR changed neither the workflow files nor the toolchain pin that
     the run executes. The approval of a v5 lane is checked before signing, by
     admission; the guard does not check it again.
   - **v6 lane.** v6 together with v5: `fail`. Two or more v6 manifests:
     `fail`. Exactly one changed manifest and it is v6: continue. A repository
     that is not `rulespec-<country>`: `fail`. A draft PR: `fail`.
   - Everything else that touches a guarded path has no lane and gets `fail`:
     workflow changes, `.axiom/` configuration and pins, ledgers, ProgramSpec,
     tooling, and mixed changes. This is deliberately fail-closed while the
     code is inactive. It is the first open question (§10).
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
     commit of the PR. A commit whose author or committer GitHub cannot
     attribute to an account gives `fail` (`unattributed_commit`), because Git
     author metadata is free text and the evaluator cannot tell that the
     approver did not write it. The remedy is to re-author the commit with an
     email that GitHub links to an account, and to sign again. There is no bot
     exemption: the signing workflow's manifest
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
     `pull_requests`. The Actions run must finish with status `completed` and
     conclusion `success`. Step 4 proves `.github/**` at `H` equals the base, so a
     `pull_request` run of `H` used the protected workflow definitions. That
     holds only for a `pull_request` run: a `workflow_dispatch` run can use
     another ref's definitions, so it is refused.
8. **Re-read.** The evaluator reads the PR, the `main` tip and the reviews again
   and refuses if they moved. It then rechecks the signing run and required CI
   against the first reads. It requires the same signing actors, commit,
   observed attempt, check identities and CI run attempts. If the signing run
   or CI workflow stops succeeding, or a required check stops passing, it also
   refuses.

Pass evidence: PR number, head, base, the sha256 of `P`, `T` and `M`, the review
id, the reviewer id, the dispatcher id, the run id and the candidate commit.

**The guard does not verify the manifest signature** (open question 3). Until
it does, the binding of `M` to the signing service rests on current-head CI
(which verifies the signature) plus the run and candidate checks of steps 4 and
6. The manifest does not record a run attempt, so the guard cannot tell which
attempt of the run signed it.
The evaluator compares the attempt number that GitHub reports on its two
reads. The manifest still omits the signing attempt.

### Known limits

- A v5-lane PR is `not_applicable` on the strength of CI and scope; its
  approval is not checked by this guard.
- An infrastructure, ledger, ProgramSpec or tooling PR has no lane and gets
  `fail`. Once the check is required, every such PR is blocked until a lane
  for it is decided (§10, question 1).
- A signed v5 PR that also changes another tracked file has no lane. Observed:
  rulespec-us#1470 (a signed manifest together with
  `.axiom/legacy-replacements/<digest>.json`, `.axiom/toolchain.toml`,
  `known-validation-gaps.yaml` and a test file). The lane was not widened.
- A fork PR and a stale PR get `fail`, not `not_applicable`.
- 1000 reviews on one PR, also comment-only reviews from outside accounts,
  exhaust the page bound and give a permanent `fail`. The remedy is a new PR.
- A re-run of the signing run by another person changes `triggering_actor` and
  gives `dispatcher_mismatch` permanently. The remedy is a new signing run.
- An approver who clicks "Update branch" becomes a commit author and can no
  longer approve.
- A run or review can change after its last read. GitHub supplies no transaction
  across these endpoints; the next sweep catches the change.
- A write collaborator can block the remaining approver by pushing a commit
  authored as that approver, or an unattributed commit. The remedy is to drop
  the commit, which needs a new signing run.
- Two PRs on one head both appear in the CI run's `pull_requests`; the guard
  accepts the run for each.
- A v5 lane whose manifest does not list every changed module file in
  `applied_files` gets `fail`. The retire and path-migration manifest shapes
  were not checked.
- The lanes are defined for the layouts seen on `rulespec-us`. Other
  repositories were not surveyed.

### Census: the last 50 merged `rulespec-us` PRs under this classifier

Read-only, 2026-11-12, repeated after rulespec-us#2045 merged (before that
the counts were 1 and 49): the 50 most recently merged PRs to `main` among the 100
most recently updated closed PRs, classified with the round-3 rules (reduced
allowlist, digest and position binding). The census reads file lists, the
changed manifests and the module bytes at the merge commit, and takes the
jurisdictions from today's `main`. It does not read modes or CI; a merged PR is
taken as regular and green. `pass` is impossible because no v6 manifest exists.

| Verdict | PRs |
|---|---|
| `not_applicable` (`no_guarded_content`, allowlist) | 0 |
| `not_applicable` (`earlier_lane_v5`) | 2 (#2045 and #2154, a merge train of 5 signed manifests; all 5 satisfy the digest and position rules) |
| `fail` (`no_lane_for_guarded_change`) | 48 |
| `fail`, other reasons | 0 |

The 48 are: toolchain and corpus pins and staged waiver fingerprints in
`.axiom/toolchain.toml` or `.axiom/workflow-toolchain.toml` (about 21), ledger
and register changes (`known-*.yaml`, `oracle-coverage-pending.yaml`), workflow
changes under `.github/`, ProgramSpec changes under `programs/`, tests and
tooling, and one signed manifest PR with extra files (#1470). Over a wider
window of 200 merged PRs the counts were 136 no-lane, 2 earlier lane and 62
other; the 39 of those 62 that were listed all carried the historic
`applied-rulespec/v1` label, written before v5 existed.

**Reading:** as written, the guard would block almost every PR that
`rulespec-us` merges today. It cannot be made a required check until the
maintainers decide which lane these changes get.

### Proposed three-class lane (off; a recommendation, not a decision)

`INFRASTRUCTURE_LANE_ENABLED` in `content_merge_guard.py` is `False`. While it
is off, everything above holds unchanged. It is a constant on encoder main
with no runtime override. This section is the proposal for question 0 of §10.

**Why the control plane needs an approval.** The v5 lane trusts this PR's
`validate / validate` run. Its scope proves that *this* PR did not change the
workflow files, the pins or the registers of accepted failures. It proves
nothing about an earlier PR. If such a change could land with no approval, two
PRs would give an unreviewed landing: PR A relaxes validation and stands
aside; PR B adds a module with a v5-labelled manifest and takes the earlier
lane. That approval is the precondition that makes the single post-sign
approval of the content lane meaningful.

**Why `tests/`, `tools/` and `bulk/` are control plane.** The shared validate
workflow runs `python -m pytest -q tests` from the PR head inside
`validate / validate`, and those tests import `tools/source_conflicts.py`,
`tools/household_choices.py` and `bulk/applied_artifacts.py`. A change or a
deletion of `tests/test_encoding_manifests.py`, `tests/conftest.py` or
`tools/engine_root_load.py` changes what the required check accepts
(rulespec-us#1472 narrowed the manifest-sync guard test). The rule: anything
that required validation executes, imports or reads from the PR head is
control plane.

`classify_path` sorts each changed path, from the path alone:

| Class | Paths | Rule with the lane on |
|---|---|---|
| unguarded | the allowlisted root files without `.gitignore`: `README.md`, `LICENSE`, `LICENSE-CODE`, `NOTICE` | as today: `not_applicable` |
| content | every path at or under a jurisdiction directory (on the base, or of jurisdiction shape), every path under an atomic root at the repository root, every apply-manifest directory | the v5 lane or the v6 lane must claim the whole delta; the v6 lane needs the content-bound approval |
| control plane | `.gitignore` (a `rulespec-us` test runs `git check-ignore`; with the lane off it stays on the allowlist, as on `main`); everything under `.github/`, `.claude/`, `programs/`, `tests/`, `tools/`, `bulk/`; every `.axiom/` file that is not an apply manifest; every root file off the allowlist (`known-*.yaml`, `oracle-coverage-pending.yaml`, `household-choices.yaml`, `.gitattributes`, `CLAUDE.md`, and any new root file) | **one content-bound approval** (`pass`, `control_plane_content_bound_approval`), never "stand aside" |
| infrastructure (stand aside) | exactly `INFRASTRUCTURE_PATHS`, which is **empty** | `not_applicable` (`infrastructure_as_today`) |
| unknown | any other top-level directory, and any path that is not canonical | no lane: `fail` |

**The stand-aside class is empty on purpose.** A path belongs in it only when
it is positively known that required validation neither executes nor reads it.
The candidates on `rulespec-us` are the Markdown files under `bulk/` and
`tools/`. No test or workflow names them today, but a search is not a proof
(a test can walk a directory), and none of the last 50 landings changed only
such files. So nothing is listed, and the default for everything that is not
content and not unknown is control plane.

The names were checked against the `rulespec-us` tree on 2026-11-13. Its
non-jurisdiction top-level directories are exactly `.axiom`, `.claude`,
`.github`, `bulk`, `programs`, `tests` and `tools`.

A path is not canonical, and so unknown, when it is empty, has an empty or dot
segment, a backslash, a space, a control or non-ASCII character, or a segment
that equals a name the classifier knows only after case folding
(`.axiom/ENCODING-MANIFESTS/...`, `Tests/x.py`, `readme.md`).

A pull request is judged as a whole. Only guarded paths are classified;
allowlisted files ride along:

- **content with anything else: refused.** Content only: the lanes of step 3,
  unchanged;
- **an unknown path never joins the control-plane or the stand-aside
  lane.** Outside a content lane it is refused. Inside a v5 unit the earlier
  lane applies its own path rules, as on `main`, and does not ask this
  classifier: a module path that this classifier calls not canonical, such as
  the 15 `us-nh/... He-W 704/...` paths on `rulespec-us` that hold a space,
  still gets `not_applicable` (`earlier_lane_v5`);
- **control plane** (with or without allowlisted files or a listed
  stand-aside path): the whole delta is control plane. The approval covers
  every path of the delta, the allowlisted ones too;
- **only listed stand-aside paths:** `not_applicable`. None exists today.

**The control-plane approval** uses the binding of step 5 over every path of
the delta, with no manifest, signature or dispatcher:

- the reviewer is enrolled (id, login, `type == "User"`), is not the PR
  author, and is not the author or committer of any commit; an unattributed
  commit fails; the commit list must be complete and hold the head;
- the approval is on a commit `C` of the PR whose delta against its merge
  base with `main` has the same paths as the head delta and, for each path,
  the same blob or the same absence on **both** sides: what `C` held equals
  what the head holds, and what `C` replaced (its merge base) equals what the
  head replaces (the current `main` tip). A base update that touches no path
  of the delta keeps the approval. A base update that touches a path of the
  delta voids it, also when the head keeps the approved bytes: otherwise an
  approved workflow or pin would silently revert a stricter one that `main`
  received later. Any changed, added or dropped path voids it;
- a draft PR, a blocking `CHANGES_REQUESTED`, a required check that is not
  green from this PR's run, and any movement of the PR, `main`, the reviews or
  CI between the two reads fail.

**Limit of the independence check.** Independence is by GitHub attribution
only. The evaluator checks the PR author and the GitHub-attributed author and
committer of each commit. It does not read commit trailers or message text. A
PR opened by a bot account, with commits attributed to that bot, passes with
one enrolled approval also when that owner wrote the change. This is the
residual risk of §9 in another form.

**Limit for stacked PRs.** The binding compares against a merge base, and the merge base of the approved commit is computed when the guard evaluates, not when the review was given. If an ancestor commit `X` of the approved commit `C` lands on `main` later through a merge commit, the merge base moves to `X`. The reviewer saw `old → T`; the head now does `X → T`; the bytes that land are the approved ones, and the verdict is `pass`. The approver did not see the step `old → X` as part of this PR; it landed through its own PR and that PR's lane.

**Executable files.** The leaf rule of step 3 stays: an added or modified leaf
that is not a regular `100644` blob fails in every class. `rulespec-us` holds
three executable tools (`bulk/compute_matrix.py`, `bulk/roots_for.py`,
`tools/generate_hts_rate_tables.py`); a change to one of them has no lane, also
with this lane on. None of the 50 landings below changed one. The remedy is to
make them `100644` in `rulespec-us` first, not to loosen the leaf rule.

**Known shapes without a lane.**

| Shape | File pattern | Verdict with the lane on | Follow-up |
|---|---|---|---|
| Signed backfill (rulespec-us#1470) | module files and their manifest, together with a receipt `.axiom/legacy-replacements/<64 hex>.json`; usually also `.axiom/toolchain.toml`, a `known-*.yaml` waiver edit and a test | `fail`, `signed_backfill_has_no_lane` | The receipt, the waiver edit and the pin belong to one signed operation, so "split it" is not verified to work. A lane for this shape is a separate change; this one does not invent it |

With the lane off, that shape keeps today's reason
(`no_lane_for_guarded_change`). No other mixed shape occurs in the 50
landings below.

Evidence, read-only on 2026-11-13: the last 50 first-parent landings on
`rulespec-us` main (from the landing of #1470 to the landing of #2045), each classified from its delta against
its first parent.

| Class of the whole landing | Landings |
|---|---|
| control plane (one owner approval) | 47 |
| content only (v5 lane) | 2 (#2045, #2154) |
| signed backfill (refused by name) | 1 (#1470) |
| stand aside | 0 |
| unknown path | 0 |

Of the 47: 29 change `tests/`, 17 `.axiom/toolchain.toml`, 12 `.github/**`, 9
`tools/`, 3 `bulk/`; 6 change only `tests/`, `tools/` or `bulk/`.

**The load, stated plainly.** `rulespec-us` main requires 0 approvals today.
With this lane, **47 of the last 50 landings need an enrolled owner's
approval**: a new review requirement on almost every change that is not an
encoding. 18 of the 50 landings touch `.axiom/toolchain.toml`, and all 18 were
authored by the two enrolled owners (9 each), so each owner waits for the
other on every pin bump. This proposal removes one of two reviews per encoding
and adds one review per control-plane change. Alternative (c) below is the
recommended follow-up to cut that load.

**What this lane gives and what it does not.** With it, the guard can become a
required check, and two things hold for an account that is not an enrolled
owner: it cannot land a content-class path without a v5 lane or a
content-bound approval, and it cannot change anything that required
validation executes or reads without an enrolled owner's approval of the
exact bytes. It does not close the residual risk of §9: agents that hold both
enrolled identities.

Alternatives considered:

| Alternative | Why not recommended first |
|---|---|
| (a) Minimum control plane: `.github/**` and the two pins only | Leaves the registers that relax validation, and the tests and tools the required check runs, without approval; the two-PR path stays open through them |
| (b) Stand aside for every infrastructure PR (the first version of this proposal), or for `tests/`, `tools/` and `bulk/` (the second) | No new review, or fewer, but the required check runs those files from the PR head. No guarantee |
| (c) Exempt machine-made pin bumps by a named trusted automation identity from the approval | Recommended as a **follow-up** to cut the load (18 of 50 landings). It needs a trusted publisher for the bump and a rule for what a bump may contain; not in this change |
| (d) Retire the v5 CI-trust lane before change 6, so every content landing needs a content-bound approval | Removes the premise the two-PR path attacks. Worth doing, but the control plane still decides what the required check runs |
| Keep `fail` (today's classifier) | The check can never be required, so the pre-sign approval can never be removed (D2d) |
| Decide by a label, a PR title or a file inside the PR | The PR controls all three (round 1, F1) |

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
  `main` that holds the guard App's private key. **On hold:** nobody may
  store that key, for a shadow run either, until the mechanism of
  `docs/merge-guard-key-protection.md` is chosen, implemented and
  independently reviewed (checklist step 1 of §8; #2257).
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
(open question 14).

### Reader (change 2, present and switched off)

`applied_manifest_v6.py` holds the reader contract and one switch,
`V6_READER_ENABLED`, which is `False`. The switch is a constant on encoder
main. It has no environment variable, flag or repository setting, so a
RuleSpec PR or workflow cannot turn it.

Which reader does what, with the switch off and on:

| Reader | Off (today) | On |
|---|---|---|
| `_load_verified_applied_encoding_manifest_payload`, and so its callers (19 call sites in `cli.py`, and others in reviewed repair, run-log export, telemetry sync, the regeneration judge and notary administration): validation coverage (`guard-generated`), promotion, retire, path and metadata migration, import inventory, legacy replacement, reviewed repair; with and without `historical_verification` | refuses | admits an exact signed v6 |
| `_guard_base_manifest_claims` (a manifest on the immutable base) | refuses | admits an exact signed v6, as obligations only |
| Readers of a nested manifest (`_retired_model_manifest_issues`, `_migrated_model_manifest_issues`, `_legacy_replacement_manifest_issues`) and `_historical_manifest_refresh_inputs` | refuse | refuse |
| `supabase_sync.run_from_apply_manifest`, `prepare_signed_backfill.py`, the canonical-refresh selection in `targeted-signed-reencode.yml`, `_canonical_replacement_baseline` | refuse or ignore (v5 only) | refuse or ignore (v5 only) |

"Exact signed v6" means: the promotion tool and no `backend` key; a valid
signature over the whole payload; the exact v6 field set; the exact
`attestation_scope` shape; `candidate.head_sha == candidate_rulespec_ref`; and
`candidate.repository` equal to the repository of the manifest's attested
`source_attestation.rulespec_root`. Producer, validation-execution, source and
waiver checks are those of the v5 promotion contract.

- **Historical verification admits v6 when the switch is on, by design.** A
  landed v6 manifest becomes history that every later change must verify
  (`historical_verification=True` reads the manifests of unchanged files). If
  that reader refused v6, the first v6 landing would turn later validation
  red. It admits v6 on the same terms as a v5 promotion manifest, and no
  wider.
- **The repository rule is shared.** `is_promotion_repository`
  (`TheAxiomFoundation/rulespec-<country>`) is the one definition for the
  reader and the evaluator. The reader binds the scope's repository to the
  manifest's attested RuleSpec root, which the loader compares with the
  checkout. It cannot compare the named PR, author or dispatcher with GitHub;
  the merge guard does (§3 steps 4 and 6, case 19).
- **Boundary:** only a manifest labelled v6 takes these paths. A change that
  carries no manifest (workflows, pins, ledgers, tooling, ProgramSpec) reaches
  no reader of this change, with the switch on or off.

**The switch rule.** The reader may be switched on while no v6 writer exists: nothing new can land. The v6 writer and the admission without approval must stay off until the guard check is required. With the reader on, the writer on and the guard check not required, an enrolled dispatcher lands their own unreviewed candidate with no approval, and any account with write access re-lands signed v6 bytes from a closed pull request.

So opening the writer and the admission without approval (change 6), not the
reader, is what waits for the required check. Change 5 supplies both locked
and opens nothing; for the switch rule a locked writer counts as no writer.
With the reader on, `validate / validate` treats an
exact signed v6 manifest like a reviewed v5 one: it proves validation and
nothing about review. `owner_review_state` (`none` for v6, `before-signing`
for the earlier lane) exists for callers to tell the two apart (D1), and it
has no production caller yet. This was chosen over wiring a refusal into
`validate` now: activation would have to remove it again, and the guard, not
`validate`, is the control proposed in D2.

Not read as v6 yet (last row of the table): they treat a v6 manifest as
absent or foreign. They matter only after a v6 module is on `main`. Change 5
does not change them; that work is needed before or with change 6. The
unreceipted-retirement scan has no v6 field class, so
it scans every string of a v6 manifest, which is the conservative reading.

Still to do in later changes: turn the switch on (change 2b); open the
promotion writer (change 6; change 5 supplies it locked). A verifier pinned before change 2b rejects v6, so no v6
manifest can pass `validate / validate` until `rulespec-us` bumps its encoder
pin. That is a built-in ordering gate.

### Encoder (change 5, present and switched off)

The unreviewed-candidate lane is in `reviewed_pr_admission.py` (admission) and
`applied_manifest_v6.as_v6_candidate_payload` (the one v6 writer, called from
one place in `cmd_promote_reviewed_candidate`).

**Three locks.** Locks 1 and 2 are constants on encoder main. No
encoder-side environment variable, CLI flag, dispatch input or setting opens
them. Lock 3 is code on encoder main that separately reads the branch
protection of the RuleSpec repository. It refuses unless that protection
requires the guard check from the recorded guard App. Locks 2 and 3 have
two accepted forms since change 5c: the App form of this paragraph and of
the table, and the organization-ruleset form described under the table.
Neither form is recorded, and at most one may be.

| # | Lock | State today | Opened by |
|---|---|---|---|
| 1 | `reviewed_pr_admission.V6_WRITER_ENABLED` | `False` | Change 6, a reviewed one-line change |
| 2 | `reviewed_pr_admission.GUARD_APP_ID` | `None` | A reviewed change that records the guard App's numeric App ID, after checklist step 1 (on hold: see that step in §8) |
| 3 | `require_guard_enforced`: a live read of the public branch record `GET /repos/<repo>/branches/main` | refuses: `main` requires only `validate / validate` | Checklist step 4 (a maintainer makes `content-bound-merge-guard` a required check with the App's `app_id`) |

- Lock 3 passes only when the record shows `protected: true`,
  `protection.enabled: true`, `enforcement_level: "everyone"` (administrators
  included) and exactly one required check with context
  `content-bound-merge-guard` and `app_id == GUARD_APP_ID`. App 15368 (GitHub
  Actions) can never be the guard's publisher. A check named only in the
  legacy `contexts` list, the shadow check name, a read error, a rate limit
  and an unknown shape all refuse. A required check that comes from a
  repository ruleset is not in this record and does not count.
- **Two accepted forms of locks 2 and 3 (change 5c, locked).** The rows
  above are the App form. The mechanism recommended in
  `docs/merge-guard-key-protection.md` (not accepted) has no guard App, so
  the App form could never open under it. The second form:
  - Lock 2 is `reviewed_pr_admission.GUARD_WORKFLOW`, `None` today. A
    reviewed activation change records `GuardWorkflow(repository_id, path,
    shas)`: the numeric id of the repository that holds the guard workflow,
    the path `.github/workflows/content-bound-merge-guard.yml`, and a
    non-empty `frozenset` of reviewed 40-hex commit SHAs. No environment
    variable, flag, dispatch input or setting sets it.
  - Exactly one of `GUARD_APP_ID` and `GUARD_WORKFLOW` may be recorded.
    With neither recorded (today), and with both, `require_guard_enforced`
    refuses before any read of its own. (Admission, `verify_reviewed_pr`,
    reads the signing run before it calls `require_guard_enforced`; in that
    state it makes that one read and no guard read.)
  - Lock 3 then reads the active rules of the branch, anonymously:
    `GET /repos/<repo>/rules/branches/main`, every page within a bound. It
    passes only when the whole listing has exactly one rule of type
    `workflows`; that rule has no key outside `type`, `parameters`,
    `ruleset_source_type`, `ruleset_source` and `ruleset_id`; it is from a
    ruleset of the organization (`ruleset_source_type: "Organization"`,
    `ruleset_source: "TheAxiomFoundation"`, an integer `ruleset_id`); it
    names exactly one workflow with the recorded repository id, the
    recorded path, a `sha` of the recorded set and no `ref` key (a `ref`
    key refuses whatever its value, `null` included: the schema's `ref` is
    an optional string that is not nullable, so `"ref": null` is outside
    the schema; if the live response of X14 carries the key, the lock
    stays shut until a reviewed change); and the listing has at least one
    rule of type `merge_queue`, each of them a well-formed rule of the
    RuleSpec repository's own ruleset whose parameters make every entry
    merge alone and only on its own passing checks (below). The
    rules of a ruleset in `Evaluate` or `Disabled` mode are not in the
    listing [docs: "Rules in rulesets with "evaluate" or "disabled"
    enforcement statuses are not returned", REST "Get rules for a branch",
    read 2026-11-17]; the listing can still hold rules of other rulesets.
    With the guard's ruleset in such a mode the listing has no `workflows`
    rule, so it refuses. A rule of a repository ruleset, a second
    `workflows` rule, a second workflow, a `ref` key, an unknown key in the
    entry, in the parameters or in the rule, a queue rule with another
    grouping strategy or group size, a read error and a rate limit all
    refuse.
  - Two decisions about the `workflows` rule, each the stricter reading
    until X14 records a live response (review round 1, P2-2):
    - A key of the rule outside the five above refuses, in particular
      `enforcement`, `bypass_actors` and `current_user_can_bypass`. They
      are fields of a ruleset in the REST schema, not of a rule of this
      listing; what they would mean here is not documented. If GitHub's
      live response carries another key, the lock stays shut until a
      reviewed change of the lock.
    - `do_not_enforce_on_create` must be absent or exactly `false`. The
      schema describes `true` as "Allow repositories and branches to be
      created if a check would otherwise prohibit it"; whether that touches
      a merge into an existing `main` is not documented.
  - Why the merge queue is part of the lock: no event follows a review, so
    a pass on the pull request stays green after a dismissal. Only the
    merge-group run reads the reviews again at merge time. Admission
    without the pre-sign approval must not open while a merge can go around
    the queue. A merge queue that is set in classic branch protection is
    expected not to be in this listing, and then does not count [not
    verified]; the lock fails closed.
  - What the lock requires of the merge queue rule, and why not more
    (review round 1, P2-1). GitHub's documentation of "Require merge queue"
    says: "This rule is not available for rulesets created at the
    organization level." [docs: "Available rules for rulesets", read
    2026-11-17; source file in `github/docs` at `49a012fa`, last changed in
    `1e6ab0ef`]. So the organization source that is required of the
    `workflows` rule cannot be required of the queue. Required instead, of
    every `merge_queue` rule in the listing: no key outside the five rule
    keys, `parameters` a JSON object, `ruleset_source_type: "Repository"`,
    `ruleset_source` exactly the repository that is checked, a positive
    integer `ruleset_id`. A rule of type `merge_queue` alone, and a queue
    rule that names another repository, refuse. The parameters are held to
    the settings described two items below.
  - **The limit of the queue half.** The queue rule lives in a repository
    ruleset, which the administrators of the RuleSpec repository can edit,
    disable or give a bypass list. The read cannot see that ruleset's
    bypass list. An administrator on it merges outside the queue, with the
    pull-request pass that stays green after a dismissed approval. The lock
    proves only that the rule was present at each of its reads: the start
    and the end of admission and each revalidation.
  - **The queue's parameters, and what the merge-group evaluation proves**
    (review of `e2d12f5c2`; the earlier text here said that no parameter
    mattered, and that was wrong). The lock accepts no parameter key outside
    the seven of the schema `repository-rule-merge-queue`, and requires
    four of them, present and exact: `grouping_strategy: "ALLGREEN"`,
    `max_entries_to_merge: 1`, `min_entries_to_merge: 1` and
    `max_entries_to_build: 1`. Any other value, type or a missing key
    refuses, for every `merge_queue` rule of the listing.
    - What the merge-group evaluation of the enforcing guard proves (change
      3b, #2286; a stub that fails on this branch): for the one pull
      request that the group ref names, the tree of the group commit is the
      tree of that pull request's judged head, and the group's reported
      base is the judged `main` tip [read in `content_merge_guard.py` of
      `claude/merge-guard-enforce` at `519603227`]. That binds the **final
      tree** to one judgment. It does not prove that the group holds only
      that pull request; the evaluation reads no other member.
    - A group of two pull requests therefore does **not** always fail at
      the guard. When the second pull request contains the commits of the
      first, the final tree is the tree of the second head; with a reported
      base equal to the `main` tip the evaluation returns `pass` and has
      read only the second pull request [reproduced against that module at
      `519603227` with its own fixtures; what `merge_group.base_sha` is for
      an entry behind another is not verified: experiment X5 of #2287].
      With `HEADGREEN` GitHub requires only "the commit at the head of the
      merge group" to pass [schema text of `grouping_strategy`], so the
      first pull request merges although its own merge-group check failed
      or never judged it. Its bytes are bound through the second pull
      request's judgment. The first pull request itself merges with no
      fresh judgment: a dismissed approval or a changes-requested review on
      it does not stop it.
    - What safety requires. The admission without the approval before
      signing needs more than the final-tree binding: **every queue entry
      that merges must have its own fresh passing judgment at merge time**.
      (An entry, not every pull request: a pull request whose commits are
      all inside another judged pull request's head is marked merged by
      GitHub when that one merges, without entering the queue. Its bytes
      are bound through the other judgment, and no setting of the queue
      changes that.)
      The guard cannot prove that for a group of several entries. So the
      queue must merge every entry alone and only on its own passing
      merge-group checks, and the lock refuses every queue rule that does
      not say so. `ALLGREEN` ("the merge commit created by merge queue for
      each PR in the group must pass all required checks to merge" [schema
      text]) and a group of exactly one entry would each give the property
      alone; both are required, because GitHub's behaviour for either is
      documented and not observed here. A build concurrency of one is
      required with them: an entry behind another is built on the first
      one's commit, a configuration the experiment does not run. These are
      the settings that #2287 already asks of the `rulespec-us`
      administrator (its checklist step that turns the queue on: only merge
      non-failing pull requests, group size 1, build concurrency 1) and
      that its experiments X5 and X15 run. This change makes the lock check
      them at every read.
    - Not checked: `merge_method` (trees are compared, so the method does
      not matter), `min_entries_to_merge_wait_minutes` (with a minimum of
      one entry nothing waits) and `check_response_timeout_minutes` (a
      check that does not report in time counts as failed [schema text]; a
      wrong value stops merges).
    - A model run, not evidence about GitHub: 696 queue scenarios (groups
      of 1, 2 and 3 entries; each position failing; stacked and independent
      heads; a final tree equal to an inner member's head; both readings of
      `base_sha`; a moved `main`; both strategies; group sizes 1 to 3),
      with each entry's check computed by the real evaluator of
      `claude/merge-guard-enforce` at `519603227` and the queue's merge
      rule modelled from the schema text. A pull request merged without its
      own passing check in 7 scenarios, all of them `HEADGREEN` with a
      group size above 1, and in none with the settings the lock requires
      [fix-agent-executed 2026-11-22; the harness is not in the
      repository]. This is a consistency check of the argument above and no
      more. The zero for `ALLGREEN` follows from the model's own merge rule
      (every entry's check must pass) together with one assertion on the
      evaluator, that it never passes an entry whose own pull request
      judgment fails; the 696 scenarios reduce to 16 distinct calls of the
      evaluator. It says nothing about what GitHub does, and it does not
      model the time between a judgment and the merge.
    - This rests on that guard, which the lock does not read; on GitHub
      requiring the workflow on the merge group and applying the queue
      parameters as its schema describes them [not verified: experiments
      X5 and X15 of #2287]; and on the listing showing the parameters as
      the one recorded live queue rule does (axiom-encode, 2026-11-17). An
      administrator of the RuleSpec repository can change the parameters
      after a read ("The limit of the queue half" above); the lock reads
      them again at each revalidation.
  - The admission record's `guard` is then `{form:
    "organization_ruleset_workflow", ruleset_id, repository_id, path,
    sha}`; the App form's record is unchanged. Revalidation compares the
    whole record, so a pin that moves between admission and signing
    refuses, also to another recorded SHA.
  - **A repeated key in the response text refuses**, at every level of the
    listing. `json.loads` keeps the last value of a repeated key, so the
    text `"max_entries_to_merge": 5, "max_entries_to_merge": 1` would read
    as 1, and `"type": "merge_queue", "type": "deletion"` would hide a weak
    queue rule; which value GitHub's own code reads is not known. The
    production reader of this listing parses the text itself and refuses.
    The limit: the other reads of the module (the pull request, the
    reviews, the signing run, the App form's branch record) use the shared
    reader of the reviewed lane, which keeps the last value; it is the
    reader of the lane that is active today and is not changed here.
  - **The paging assumption.** The listing is read in pages of 100 and
    ends at the first page with fewer than 100 rules. If GitHub ever
    returned a short page that is not the last, the rules after it would
    not be read, and a second `workflows` rule or a weak queue rule there
    would be missed. GitHub documents no such page; nothing here verifies
    it.
  - **Three more things the read does not show** (independent review of
    `4c7da64f0`). A merge queue set in classic branch protection: this
    read lists rulesets only, and if such a queue exists beside the
    ruleset's queue rule, which of the two governs a merge is not known
    and the classic one's settings are not checked; the precondition for
    activation is that there is no merge queue in classic branch
    protection of `main` (below). The age of the answer: GitHub serves the
    listing to an anonymous reader with `cache-control: public,
    max-age=60, s-maxage=60` [read 2026-11-22], as it does the branch
    record, so a change of the rules can be about a minute old before a
    read shows it. The time between the end of the guard's merge-group job
    and the merge: a dismissal in that window is not judged again, and
    only X15 of #2287 measures the window.
  - What this read cannot see: the bypass list of the organization ruleset
    (the ruleset endpoint returns `bypass_actors` only to a caller with
    write access to the ruleset [docs: REST "Get a repository ruleset",
    read 2026-11-17]); the bypass list and the editors of the repository
    ruleset that holds the merge queue; whether GitHub runs the file of
    exactly the pinned commit; any later change by an organization owner.
    A bypass entry that exists before the dispatch is invisible at every
    read.
  - What the lock does not check. That the pinned file is the guard: the
    path is a name, and the recorded SHA set is the ONLY evidence. The
    reviewed change that records a SHA in `GUARD_WORKFLOW` asserts that the
    file at that path in that commit is the reviewed guard workflow; the
    lock does not read the file, does not check that the commit exists in
    the recorded repository, and compares the repository by its numeric id
    only. Three of the merge queue's seven parameters (above). The identity
    of the queue's ruleset between two reads: its `ruleset_id` is not in the
    admission record, so a queue ruleset that is replaced by another
    well-formed one of the same repository does not refuse.
  - The response shape of a `workflows` rule on this endpoint has a
    documented schema and no documented example. The test fixture
    `tests/fixtures/merge_guard_requirement/rules_branches_main_workflows_SYNTHETIC.json`
    is SYNTHETIC, built from that schema.
    [experiment X14]: must be replaced by the recorded live response before
    activation.
    Recorded on 2026-11-17 (`rules_branches_main_recorded.json`): the
    listing of `rulespec-us` `main` is empty, and the listing of
    axiom-encode `main` holds one `merge_queue` rule of a repository
    ruleset: the queue of axiom-encode is a repository ruleset [live read
    2026-11-17]. No other queue of the organization was read. Both
    listings refuse.
- The read is anonymous, so the signing runtime still holds no repository
  credential. Verified on 2026-11-13: the record is public and carries
  `checks[].app_id`.
  To see what lock 3 sees:
  `curl -s https://api.github.com/repos/TheAxiomFoundation/rulespec-us/branches/main | jq '{protected, checks: .protection.required_status_checks}'`.
- In a candidate-lane run, lock 3 runs at the preflight, at the root
  provisioner's admission, before validation, immediately before signing and
  before publication (every call of `verify_reviewed_pr` and
  `revalidate_admission` of that lane). A reviewed-lane run makes no guard
  read at admission or at revalidation.
- **Operator precondition (not checkable in code).** Lock 3 proves the
  requirement up to the signature. A repository administrator can remove a
  required check afterwards; that adversary is out of scope (§9). Do not open
  lock 1 before checklist step 4 is done and confirmed.
- With the switch on, the lane also needs `V6_READER_ENABLED`: a v6 manifest
  that no verifier admits cannot be written.
- **Known limit: the anonymous rate limit (availability only).** One signing
  run of the candidate lane that goes through publication makes at least 45
  anonymous reads (7 of them the branch record), and one of the reviewed lane
  at least 38. The reviewer counted 45 and 38 at `880238c7` with every list
  on one page; pagination adds reads. 45 is the count through the last
  publication check, counted at change 5, where the hosted workflow's v5
  evidence step stopped a v6 manifest before that check. With the evidence
  step of change 5b (below) a hosted candidate run reaches it, and that
  step adds one read of the branch record for each v6 manifest: at least
  46. Neither count was measured again. GitHub allows 60 per hour per address. A
  second run in the same hour from the same address is refused. It fails
  closed. Follow-up, not built: send the reads with the workflow token when
  one is available. The record is public, so a token only raises the limit;
  any error must still refuse.

**With lock 1 off, nothing changes.** `verify_reviewed_pr` and
`revalidate_admission` make the same reads and return the same results and
errors as before the lane existed, and the promotion writes v5.

**With all locks open** (tests only):

- Admission keeps every scope check of `verify_reviewed_pr` and drops the
  requirement for an approval. An effective `CHANGES_REQUESTED` from any
  account still refuses, as today. One account's approval does not cancel
  another account's request for changes; reviews are ordered by their id,
  not by the listing. Stricter than the reviewed lane, in this lane only:
  a review whose id, or whose reviewer's id, is not an integer refuses the
  candidate, because reviewers without a numeric id would share one key.
  The reviewed lane's `_approved_reviews` is unchanged.
- `triggering_actor` must be enrolled by numeric id, with the enrolled login
  and `type == "User"` (D3). The id is recorded.
- The admission record has another outer `schema`,
  `axiom-encode/live-candidate-pr-admission/v1`, no `approvals` key, and
  `owner_review: "none"` (the nested `record.schema` stays
  `axiom-encode/reviewed-rulespec-admission/v1` in both lanes), so an auditor cannot mistake a scope-checked
  candidate for a reviewed one. `revalidate_admission` checks it against live
  state, including lock 3.
- `cmd_promote_reviewed_candidate` writes `applied-rulespec/v6` for such a
  record: `candidate_rulespec_ref` and the `attestation_scope` of this
  section. The writer writes the scope with literal keys and the reader's
  constants, then calls the reader's `attestation_scope_issues` on the
  result before signing. It does not build the scope from the reader's
  field sets.
- A reviewed admission record (the earlier schema) still produces a v5
  manifest. With the switch on, `verify_reviewed_pr` issues candidate records
  only.

**Not in change 5; each is needed before or with change 6.**

- The hosted workflow's evidence step in `targeted-signed-reencode.yml`
  ("Package exact generated changes") read only `reviewed_rulespec_ref` of a
  v5 manifest, so it refused a v6 manifest after signing and before
  publication. Change 5b teaches it v6, switched off: see "Evidence step"
  below.
- The preflight step still runs inside the gated `encode` job.
- The v5-only readers of the reader table above still do not read v6.
- For change 6: with every lock open `verify_reviewed_pr` issues candidate
  records only, so a live pull request promotion that reaches the evidence
  step with a v5 manifest would be a signer fault, yet the step's v5 branch
  does not know which lane admitted the run and still packages the
  reviewed-candidate evidence; change 6 should make the step refuse that
  (defence in depth: the CLI tests already pin that a candidate admission
  never signs v5).

**Not in change 5, and not decided.**

- `attestation_scope` has no `run_attempt`. Adding it before or with change 6
  is a recommendation that still needs a decision (open question 14).
- The runtime attestation key stays `reviewed_pr_admission`; the record's
  outer `schema` tells the lanes apart. This is unchanged and is no
  prerequisite of change 6. Only if maintainers later rename the key would
  the root provisioner need a change first.

### Evidence step (change 5b, workflow half, present and switched off)

The evidence step calls
`candidate_promotion_evidence.require_candidate_promotion_evidence` for a
changed top-level manifest labelled v6, and for nothing else. It decides this
once, on the top-level manifest: the ref field and the evidence record follow
the paths that the call admitted. A v6 label nested in a legacy-replacement
manifest is refused by name, before any lock is read, as the reader's nested
readers refuse it. It refuses unless every
lock of the writer is open, checked first with the writer's own functions:
`require_candidate_writer` (lock 1 and `V6_READER_ENABLED`), then
`require_guard_enforced` for this run's repository (locks 2 and 3). It has no
switch, environment variable, flag or dispatch input of its own. With the
committed constants it refuses every v6 manifest before it reads anything else,
and a run with no v6 manifest takes the same path as before. For one input
the step is stricter than before, not identical: a run whose changed
manifests hold a valid v5 promotion and one more manifest that carries the v6
label, either at the top level (for any citation) or nested in a
legacy-replacement manifest **for another citation than the run's**. The
earlier step ignored that extra manifest and packaged the v5 evidence; this
step refuses the run. A legacy-replacement manifest that nests a v6 manifest
for the **same** citation is not such a case: the earlier step counted it as a
second manifest of the citation and refused the run (`found 2`); this step
refuses it too, earlier and by another name (`only a top-level manifest may
carry the v6 label`). The committed writer cannot produce such a file, so only
a hand-made manifest reaches this; tests record these outcomes.

With every lock open (tests only), the manifest must pass the reader's entry
test, its exact v6 field sets and its attestation scope check, and the scope
must name this run: the target citation, `candidate_rulespec_ref` equal to the
requested RuleSpec ref, `TheAxiomFoundation/rulespec-<country>` of the run
and the `rulespec_pr` of the `reviewed-pr-promotion/v1` dispatch. The step then
reads the ref through `CANDIDATE_REF_FIELD` and writes the evidence record
`axiom-encode/candidate-promotion-evidence/v1`, not the reviewed-candidate
record. Lock 3 adds one anonymous read per v6 manifest (at least 46 in a candidate
run; §5, "Encoder", gives the count and its limits).

`base_sha`, `author_id` and the dispatcher are checked for shape only here;
the merge guard compares them with live GitHub state.

### Multi-unit candidates (change 5d, present and switched off)

Issue #2301, option A. One candidate pull request may carry several new
modules, each with its own primary, its own companion and its own signed v6
manifest, under **one** owner approval. The lane without owner approval that
#2301 also discusses (option B and its narrow form) is not designed or built
here, and this change does not prepare it.

**Design requirements** (written before the code; each is pinned by a test of
`tests/test_multi_unit_v6_lane.py`, `tests/test_multi_unit_v6_admission.py`
or `tests/test_content_merge_guard_multi_unit_on.py`):

1. Off by default, behind one constant, with a size limit that the request
   budget of the guard covers. No verdict and no admission result
   changes while it is off.
2. The delta of the candidate is exactly a set of `k` complete units,
   `1 <= k <= MAX_V6_UNITS`. Every path is new on `main`. No other file. No
   two units share a path. Each manifest binds exactly its own primary and
   companion, by path and sha256. All or nothing: one bad unit refuses the
   whole candidate.
3. Every rule of the single-unit evaluation applies to each unit, by the
   same code. A candidate of one unit is judged as a single unit: same
   verdict, reason, detail and evidence, with one addition while the switch
   is on (the candidate rule below).
4. The approval and identity rules are unchanged and apply to the whole pull
   request. One approval covers the batch.
5. Signing stays one run for one unit. The workflow file is not changed.
6. Fail closed: an unknown shape, a partial batch, a unit already on `main`,
   more units than the limit, a v5 manifest beside v6 units, a symlink or
   another mode, a rename.

**The switch.** `reviewed_pr_admission.MULTI_UNIT_V6_ENABLED` is `False`. It
is a constant on encoder main: no environment variable, CLI flag, dispatch
input or setting turns it, and every read compares it with `is True` [code].
It is the one switch of both halves: the admission of one unit out of a
batch candidate (`verify_reviewed_pr`) and the guard's evaluation of the
final head (`content_merge_guard._evaluate`). `MAX_V6_UNITS` is 2 ("Request
budget" below: the size that both bounds of the enforcing run cover today; a
batch of 10 needs a higher request cap in the runner first).

**Why this lane is the only way to land several units at one time.** The
writer's lock requires of every `merge_queue` rule `grouping_strategy:
"ALLGREEN"`, `max_entries_to_merge: 1`, `min_entries_to_merge: 1` and
`max_entries_to_build: 1` ("Encoder" above) [code:
`_MERGE_QUEUE_REQUIRED`]. The reason: the guard's merge-group evaluation
binds the tree of a group to one judged pull request and cannot prove that
the group holds only that pull request [author-reported, draft #2286 at
`cbb47fbb6`]. So every queue entry is one pull request, entries land one
after the other, and each entry needs its own fresh judgment. Several units
can then land together only inside one pull request. A batch pull request is
one queue entry: one merge-group run judges all of its units.

The switch is not the only lock, and the two halves are locked differently
[code]:

- **The admission half** opens with the switch and the locks of the v6
  writer ("Encoder" above): it reads the switch only inside the candidate
  lane. With those open, a run signs one unit while the candidate holds
  sibling pairs that nobody has judged yet. The runner's lock below does not
  cover this half. What covers it is the candidate rule of the evaluation
  (below): a unit that was validated beside a file that the final head does
  not land never passes while the switch is on.
- **The guard half** needs the v6 reader ("Reader" above) and the enforcing
  runner. A pass of two or more units has its own reason code,
  `content_bound_approval_multi_unit`, not `content_bound_approval`. The
  runner of draft #2286 ends its job with success only for a verdict and
  reason of its `SUCCESS_VERDICTS` set [read in `merge_guard_runner.py` of
  `claude/merge-guard-recommendation` at `14e608a71`], so a batch verdict
  ends that job with failure until a reviewed change adds the reason there.
  That change is not in this one ("Follow-ups" below). A candidate of one
  unit passes with `content_bound_approval`, which the runner accepts, and
  with the switch on it passes only when it also meets the candidate rule.
- **Rollback order.** The candidate rule exists only while the switch is on.
  A manifest that was signed beside a sibling under the switch, on a head
  that later holds that one unit only, would pass an evaluation with the
  switch off. So before the switch is set back to `False`, close every open
  candidate that was signed while it was on, or sign it again. Nothing
  checks this; it is a step for the person who turns the switch off.

**With the switch off, nothing changes.** A pull request with two or more
v6 manifests gets `fail`, `multiple_v6_manifests`, as before. The evaluator's
v6 steps were restructured to loop over units, so the proof is by tests, not
by an unchanged diff:

- `tests/test_content_merge_guard.py` is not edited and passes.
- Switch off against the base. The independent review of `cc09b513a` ran a
  generator of random evaluations on the base branch and on this branch and
  compared verdict dicts and read sequences; rerun for this head with five
  seeds: 3,040 evaluations, no difference [author-reported: the generator is
  the reviewer's and is not in the repository].
- Switch on against switch off. `tests/test_content_merge_guard_multi_unit_on.py`
  runs every test of `tests/test_content_merge_guard.py` again with the
  switch on, and evaluates each of their `evaluate` calls a second time with
  the switch off over the recorded responses. Both evaluations must return
  the same verdict dict, and the same reads in the same order apart from
  the one read that the candidate rule adds (the tree of a candidate's
  base). More than 600 evaluations; the designed differences are the one
  pull request of that suite with two v6 manifests and one test whose
  fixture lacks that tree.
- `verify_reviewed_pr` is compared for single-unit candidates, result or
  error text and reads [code:
  `test_a_single_unit_candidate_is_judged_the_same_with_the_switch_on`].

**The intended change: review granularity.** Today one approval covers one
unit. With the switch on, one approval covers every unit of the batch: the
approver reviews `k` units in one sitting, and one approval, one validation
run and one place in the queue land all of them. That is the purpose of the
change and the only thing it relaxes. Who may approve does not change: an
enrolled owner who is not the author of the pull request, not the dispatcher
or run actor of **any** unit's signing run, and not the author or committer
of any commit [code: `_evaluate`, steps 5 and 6]. With two enrolled owners,
a batch whose units were dispatched by both has no possible approver; it
fails `approver_is_dispatcher` [code].

What the approver of a batch must look at, because the approval now stands
for all of it:

- every primary and every companion of the batch, not a sample: the guard
  does not know which units the approver opened;
- the list of units against what was meant to land (the pull request's file
  list is exactly `3k` files; the guard's evidence lists each unit with the
  sha256 of its three files);
- each unit's manifest: its citation, and that `owner_review` is `none`;
- the green `validate / validate` run on the head that is approved;
- units that import from each other: they were validated together only if
  the batch was frozen before signing (the freeze rule below), and CI on the
  head is the check of the final combination.

An approval is content-bound to the whole batch: the approved commit must
change exactly the `3k` paths against its merge base with `main`, and each of
them must have the blob of the head [code: `content_bound`]. So an approval
made before the last manifest arrived does not count, a byte change or a
re-sign of any one unit voids it, and a base-only update by someone else
keeps it.

**Evaluation with the switch on.** Steps 1 to 3 of §3 are unchanged up to the
count of v6 manifests. Then, for `k >= 2` changed v6 manifests:

- `k > MAX_V6_UNITS`: `fail`, `too_many_v6_units`. A limit that is not an
  `int` refuses too.
- v6 together with v5: `fail`, `mixed_manifest_schemas`, as before. A removed
  manifest, an unknown schema label and an unparseable file in a manifest
  directory fail as before.
- The repository, draft and file-list rules of step 4 are unchanged.
- Each manifest is parsed in the order of its path, with the single-unit
  rules: the exact v6 shape; this repository, this pull request and its live
  author; the canonical primary of the citation; the manifest at the one
  canonical position of that primary. Because the position decides the
  primary and the companion, two manifests cannot name one file: a second
  manifest for the same citation is `manifest_path_not_canonical`.
- **Scope.** `_v6_scope_refusal` admits the delta only when it is exactly the
  disjoint union of the units' triples and every path is `added`. One
  function decides for one unit and for many; for one unit it keeps the
  earlier reason and words (`scope_paths`).
- Two manifests that name one signing run: `fail`. One run signs one unit.
- Each unit: the sha256 of its primary and companion at the head must equal
  its manifest's `applied_files`, by path and digest.
- Each unit: its `candidate.head_sha` is a commit of this pull request that
  carries the unit's primary and companion with the blobs of the head, and
  its `candidate.base_sha` is `B` or an ancestor of `B` (unchanged rule).
- **Freeze rule (new, batch only).** The commit that each signing run
  validated must already carry the primary and the companion of **every**
  unit of the batch, with the blobs of the head. So a batch is complete
  before its first signing run, and every run validated its unit beside the
  sibling bytes that land. A unit that is added after a sibling was signed
  refuses the batch; the remedy is to sign that sibling again, or a new
  batch. Each unit is judged against its own candidate commit. For one unit
  the rule is the candidate rule above.
- **Candidate rule of the multi-unit form (new; with the switch on, for one
  unit and for many).** The commit that a run validated may differ from the
  `main` commit it was cut from (`candidate.base_sha`) only by files that
  this head lands: paths of the units of this candidate, added, with the
  blobs of the head. Anything else refuses, `v6_candidate_holds_unjudged_change`:
  the pair of a sibling that was removed later, a helper module, other bytes
  of a module of `main`, a workflow, a pin, the derived index, a manifest of
  a sibling that was signed again later. The evaluator reads the tree of the
  candidate's base for this (no read when the base is `B`). The statement
  this rule makes true while the switch is on: **no unit passes after it was
  validated beside bytes that are not part of the final, judged batch or of
  the `main` commit it was cut from.** Removing a whole unit after a sibling
  was signed therefore refuses the rest: the remedy is a new candidate
  without that unit and new signing runs. Admission cannot make the same
  statement about signing, because it cannot know which siblings will land;
  it signs only beside complete sibling units, and the guard refuses the
  signature when a sibling does not land. With the switch off the rule is
  not applied: admission then signs beside no other file but the optional
  derived index, and the evaluation is the earlier one.
  Both rules are stricter than parity with the single-unit lane needs (there
  the context of a unit, `main`, may move between signing and landing);
  keeping them is question 2 below.
- Each unit: its signing run and dispatcher, with the rules of step 6. The
  encoder's `main` tip is read once for each pass over the runs, so every
  unit is judged against the same tip.
- One approval, as described above; reviews, CI and the re-read of step 8
  are those of the single unit, and the re-read checks every unit's run.

Reason codes that only the multi-unit form produces, each with a test that
drives it:

| Reason | Meaning |
|---|---|
| `content_bound_approval_multi_unit` | `pass` for two or more units |
| `too_many_v6_units` | more v6 manifests than `MAX_V6_UNITS`, or the limit is not an integer |
| `v6_unit_incomplete` | a unit's primary, companion or manifest is not in the delta |
| `v6_batch_extra_path` | the delta holds a path of no unit: any other file, a pair without its manifest, a removed or renamed file |
| `v6_units_share_a_path` | two units name one path. Not reachable through `evaluate` (the manifest position decides the pair); the scope function is tested directly |
| `v6_units_share_a_signing_run` | two manifests name one `run_id` |
| `v6_unit_cross_bound` | a manifest's `applied_files` name a file of another unit |
| `v6_batch_not_frozen_at_candidate` | a unit's candidate commit lacks a module of the batch, or holds other bytes of it |
| `v6_candidate_holds_unjudged_change` | a unit's candidate commit differs from its base by something that the head does not land. Also for a candidate of one unit while the switch is on |

Reasons that keep their meaning for a batch: `scope_not_new_module` (a path
of any unit exists on `main`), `guarded_leaf_not_regular` (a symlink,
submodule or executable anywhere in the delta), `delta_disagrees` (a file
list that hides or renames a change), `manifest_path_not_canonical`,
`mixed_manifest_schemas`, and every per-unit reason of §3. In a batch the
detail of a per-unit refusal starts with the manifest path of the unit.

Evidence of a batch verdict: `units`, one entry for each manifest in path
order, with the fields that a single unit has at the top level (`paths`,
the three sha256, `dispatcher_id`, `run_id`, `candidate_head`) and
`manifest`. The verdict of a single unit is unchanged and has no `units`
key.

**Admission of one unit of a batch (writer side).** Today
`verify_reviewed_pr` admits a candidate whose file list is exactly the
primary and the companion of the dispatched citation, and optionally the
derived index. With the candidate lane open and the switch on, the other
files of the candidate may be complete sibling units [code:
`_batch_sibling_paths`]:

- a sibling is `<stem>.yaml` and `<stem>.test.yaml` under an atomic root
  (`<root>/...` or `<jurisdiction>/<root>/...`, no hidden segment), and
  optionally `.axiom/encoding-manifests/<stem>.json`: the manifest that an
  earlier run of the batch already signed;
- every sibling file has status `added`, and the complete tree delta must
  agree with the file list and hold regular blobs only, as for the unit
  itself;
- the unit of this run must not have a manifest yet;
- the batch holds at most `MAX_V6_UNITS` units;
- no other file, and no derived index beside siblings.

A candidate with no file of another unit is not a batch: it is judged as a
single-unit candidate, with the same words. Admission judges siblings from
their paths alone. It does not read their bytes or their manifests: the run
signs one unit, the admission record binds that unit only, and the guard
judges the final head. The reviewed (v5) lane never takes this path.

**How `k` signed manifests are produced: `k` dispatches, no workflow
change.** A v6 train cannot be built the way the v5 train of rulespec-us#2154
was, by collecting signed commits of several pull requests: each v6 manifest
names its candidate pull request, that pull request's author and a commit of
it, and the guard refuses a manifest that names another
(`manifest_candidate_mismatch`) [code: `_parse_manifest`]. So the batch is
one candidate pull request from the start:

1. Open one candidate pull request that holds all `k` pairs and nothing
   else. Do not add a unit later (freeze rule).
2. Dispatch `targeted-signed-reencode.yml` once for each unit, with the
   inputs of `docs/reviewed-pr-promotion.md`. Each run admits its unit,
   validates and signs it, and opens a draft pull request with one manifest
   into the candidate branch [code: step "Push lane branch and open draft
   pull request"]. All units must have the same dispatcher, or the batch
   has no approver.
3. Merge the `k` draft pull requests into the candidate branch. The merger
   becomes an author of a commit and can then not approve.
4. CI runs once on the head with all `3k` files. The other enrolled owner
   approves once. The guard judges that head.

Two orders are possible, and neither is tested against the hosted workflow
[not verified]:

- *Parallel.* Dispatch all `k` runs on the first head. The workflow's
  concurrency group is the run id for a manual dispatch, so the runs do not
  queue behind each other [code: `concurrency` of the workflow]. A run
  revalidates that the candidate head is still the head it admitted before
  it validates, before it signs and before it publishes [code:
  `_revalidate_candidate_admission`, `_pr_identity`], so no draft pull
  request may be merged until every run has published.
- *Serial.* Dispatch run `i + 1` after the manifest of run `i` is merged.
  Each dispatch needs the candidate to be strictly ahead of the current
  `main` tip [code: `verify_reviewed_pr`], so a landing on `main` in between
  costs a branch update by someone who is not the approver.

A unit whose run fails cannot be removed from a candidate whose other units
are already signed: they were validated beside it (candidate rule). The
remedy is a new candidate without that unit and new signing runs for the
rest, so build a batch from units that are known to pass. One anonymous-read
limit applies as before: a signing run of the candidate lane makes at least
45 anonymous reads against 60 per hour and address ("Encoder" above); the
sibling files add no read below 100 changed files. Whether `k` hosted runs
share an address is not known [not verified].

A one-run form (one dispatch, one `production-signing` approval, `k`
manifests) is **not built**. It would need a change of the workflow inputs,
of `cmd_promote_reviewed_candidate`, of the admission record (one record for
`k` bindings) and of the evidence step, and then of the rule above that two
units may not share a run. It is the follow-up that removes most of the
per-unit actions below (also #1885).

**Request budget (recomputed from the code).** The numbers of
`tests/test_multi_unit_v6_lane.py`, with one page for each list:

| Evaluation of `u` v6 units, `pass` | Requests |
|---|---|
| Worst case: each unit with its own candidate commit and candidate base, each run on its own encoder commit behind the tip, the approval on a commit that is not the head | `20 + 10u` (30, 40, 50, 70, 120 for 1, 2, 3, 5, 10 units) |
| Parallel order, one encoder commit, the base is the `main` tip, approved on its head | `20 + 5u` (30, 35, 45, 70 for 2, 3, 5, 10 units) |

The fixed 20: the pull request, the `main` tip, the comparison, two trees,
the file list, the commits, the reviews, three reads for an approval on an
earlier commit, two for CI, the encoder `main` tip, then the re-read: the
pull request, the `main` tip, the reviews, two for CI, the encoder tip. Each
unit adds at most 10: three blobs (manifest, primary, companion), its
candidate tree, the ancestry and the tree of its candidate base, and its run
and that run's ancestry in each of the two passes. Each unit adds at least
5: the three blobs and the two reads of its run. A v6 unit therefore costs
more than a v5 unit, which costs 3 [author-reported, draft #2286:
`11 + 3u`]: the guard reads no run and no candidate for a v5 unit. With the
switch off a single unit costs what it cost before: 25 on the terms of the
fixture that the guard stack measures.

The enforcing run of draft #2286 stops at 350 requests
(`MAX_ENFORCE_REQUESTS`). It polls at most 101 times, reads at most 3 times
in a merge-group poll, and reads the group's check once more after the
evaluation, at most 3 reads [read in `merge_guard_runner.py` of
`claude/merge-guard-recommendation` at `14e608a71`; that file is not on this
branch]. The merge-group evaluation adds 4 reads to `evaluate`. A batch pull
request must fit both runs, because the same pull request is judged on its
head and again as a queue entry:

| Run | Arithmetic | Largest `u` |
|---|---|---|
| Pull request: two whole evaluations and a full wait | `2 x (20 + 10u) + 101 <= 350` | 10 (341); 11 gives 361 |
| Merge group with three reads at every one of 101 polls (a second check with the required name on the group head for the whole wait) | `3 x 101 + (24 + 10u) + 3 <= 350` | **2** (350); 3 gives 360 |
| Merge group with one listed check | `100 + 2 + (24 + 10u) + 2 <= 350` | 22 (348); 2 units give 148, 10 give 228 |

`MAX_V6_UNITS` is 2: the merge-group run with three reads at every poll is
the binding bound, and 2 is the largest size that every row covers. In
that row two units use the whole cap: the run has no request to spare, and
a second page of any list then ends it with failure (closed). With
the best layout (`20 + 5u`) that row covers 4 units
(`3 x 101 + (24 + 5 x 4) + 3 = 350`), but the guard cannot require that
layout, so the limit follows the worst case. For comparison, the same row
bounds a v5 train at about 9 units [author-reported, draft #2286].

A batch of 10 units, the lower size that #2301 asks for, needs
`3 x 101 + (24 + 100) + 3 = 430` requests in that row and
`2 x (20 + 100) + 101 = 341` in the pull-request run. So it needs
`MAX_ENFORCE_REQUESTS` of at least 430 (450 leaves 20 for further pages), in
a reviewed change of the runner, which changes the guard bundle digest, and
then a reviewed change of `MAX_V6_UNITS`. Neither is in this change. The
cap only bounds a runaway run: reaching it ends the job with failure, and no
value of it gives a pass. A batch of 20 needs
`3 x 101 + (24 + 200) + 3 = 530`. A pull request with more
than 100 changed files, commits or reviews needs further pages; 10 units are
30 files. The runner's timeout table counts an evaluation as 30 requests of
at most 30 seconds; a batch evaluation makes more, and a run that then
passes the job timeout ends with failure (closed) [inferred].

Rate limit. The job token has "1,000 requests per hour per repository"
[author-reported: the design at the head of draft #2287 quotes GitHub's
"Rate limits for the REST API", read 2026-11-17]. The costliest hour of a
pull-request job holds 20 polls and both evaluations: `20 + 2 x 120 = 260`
requests for 10 units, against `20 + 2 x 30 = 80` today. So 3 such jobs fit
one hour and a fourth reaches the limit; a refused request is a failed read
and the job fails closed [inferred].

**What a person still does for one batch, and the throughput.** The steps
are those of the single-unit lane. The durations are those that the design
at the head of draft #2287 records [author-reported there: pull-request CI
172 to 227 minutes for 38 of 40 runs, live read 2026-11-17; queue CI 2.5 to
4 hours, rulespec-us#2160]. The duration of a signing run and the effect of
`k` modules on the duration of CI are not measured. One batch pull request
is one queue entry.

| Step | Single unit, for each unit | Batch of `k` units |
|---|---|---|
| Open the candidate pull request | 1 | 1 for the batch |
| Dispatch `targeted-signed-reencode.yml` | 1 | `k`, all by one enrolled identity |
| Approve the `production-signing` environment | 1 | `k` |
| Merge the signed draft pull request into the candidate branch | 1 | `k`, by someone who will not approve |
| Approve the green candidate | 1 | **1 for the batch** (the intended change) |
| Re-run the guard job after the approval | 1 | 1 |
| Add to the merge queue | 1 | 1: one queue entry |
| Pull-request CI and queue CI, serial | 5.4 to 7.8 hours | 5.4 to 7.8 hours for the batch |
| Actions of a person | 7 | `3k + 4`: 10 for 2 units, against 14; 34 for 10 units, against 70 |

| Batch size | Units for each landing | Units a day at 3 to 4 landings | Needs |
|---|---|---|---|
| 1 (today's v6 lane) | 1 | 3 to 4 | — |
| 2 (`MAX_V6_UNITS` as committed) | 2 | at most 6 to 8 | the switch and the runner's success set |
| 10 | 10 | at most 30 to 40 | also a request cap of at least 430 in the runner and a new `MAX_V6_UNITS` |

- A landing is still serial and still takes about 5.4 to 7.8 hours, so
  there are still about 3 to 4 landings a day, infrastructure pull requests
  included [author-reported, draft #2287, "Throughput"]. The number of
  units a day is that count times the batch size, and only if every landing
  is a full batch and no other pull request lands [inferred].
- The per-unit actions do not go away: 3 for each unit (dispatch,
  environment approval, merge of the signed draft). At 30 to 40 units a day
  that is 90 to 120 actions a day by the dispatcher and the merger, and 3
  to 4 approvals of 10 units each by the other owner. The one-run form and
  #1885 are what remove them.
- One bad unit costs the batch: a unit that fails CI on the head, or that
  the approver refuses, cannot be removed without new signing runs for the
  others (candidate rule). Build a batch from units that are already clean
  and reviewed locally.

**Follow-ups, not in this change.** Each is needed before the switch can be
turned on usefully; none is a setting.

1. Guard stack (#2286): add `(pass, content_bound_approval_multi_unit)` to
   `SUCCESS_VERDICTS` of `merge_guard_runner.py`. Add the new reasons to
   `_DECIDED_REASONS` of the same file: `content_bound_approval_multi_unit`
   under `pass`, and under `fail` the eight refusals `too_many_v6_units`,
   `v6_unit_incomplete`, `v6_batch_extra_path`, `v6_units_share_a_path`,
   `v6_units_share_a_signing_run`, `v6_unit_cross_bound`,
   `v6_batch_not_frozen_at_candidate` and
   `v6_candidate_holds_unjudged_change`; each is a decision, none is an
   incomplete read. Until then that stack's test
   `test_every_reason_of_the_evaluator_is_decided_or_incomplete_never_both`
   fails on the merged tree. Raise the request cap to at least 430
   before `MAX_V6_UNITS` is raised to 10 (table above); add the batch rows to that
   stack's request-volume, "human steps" and "Throughput" texts, which this
   branch does not contain. The guard bundle holds `content_merge_guard.py`
   and `reviewed_pr_admission.py`, which this change edits, so
   `BUNDLE_SHA256` of the guard workflow must be recomputed when the stacks
   are integrated, as that stack's text prescribes. This branch has no
   `BUNDLE_SHA256` and changes none.
2. Integration with the evaluator of the guard stack. That stack rewrote the
   readers of `content_merge_guard.py` so that a response with a missing
   required field is an incomplete read and never a decision. The per-unit
   loop of this change calls the shared readers (`_signing_run`, `_blob`,
   `_is_ancestor`, `_leaf_entries`) and adds no reader of its own, so it
   takes that rule from them once the stacks are merged; the merged
   evaluator must be run against this change's tests then [not verified
   here beyond a trial merge, see the pull request].
3. A canary batch of two units on the hosted workflow, in each dispatch
   order, before a larger batch.
4. The one-run form of signing, if the per-unit actions are the next limit.
5. `docs/reviewed-pr-promotion.md` at activation (change 6 updates it).

**Tests.** `tests/test_multi_unit_v6_lane.py`: the switch; each refusal
reason; every single-unit rule at the first, a middle and the last position
of a batch of three, with the reason it has for one unit; the approval and
identity rules; a failed or malformed response at every read of a passing
batch; request counts; a property test over random head trees (default
3,000 cases; run with 120,000 for this change) in which the guard passes
exactly the trees that are a valid set of new triples.
`tests/test_content_merge_guard_multi_unit_on.py`: the differential test
above. `tests/test_multi_unit_v6_admission.py`: the admission half.

**Open questions for the maintainer** (numbered here, not in §10, so that
this subsection merges without touching that table):

1. Which batch size? #2301 asks for 10 to 20. The committed 2 is what the
   runner's cap of 350 covers in every case. 10 needs a cap of at least 430
   and 20 a cap of at least 530 (and `2 x (20 + 200) + 101 = 541` in the
   pull-request run). The alternative is to accept a failed job and a re-run
   when a second check with the required name sits on the group head for
   the whole wait: without that case, 10 units fit.
2. Keep the freeze rule, or accept a unit that joins a batch after a sibling
   was signed, and a unit that is removed after one? Without the two rules
   CI on the head is the only check of the combination, and a unit can pass
   after it was validated beside bytes that nobody judged.
3. Should a batch be limited to one jurisdiction or program? Nothing limits
   it: the guard reads paths, not programs.
4. Keep the separate pass reason, and so a second reviewed change in the
   runner before any batch can pass?
5. With two enrolled owners, all units of a batch need the same dispatcher.
   Acceptable until more identities are enrolled?
6. Build the one-run form of signing?

**Not verified; only a run on the hosted workflow can show these.**

- No batch ran on the hosted workflow. In particular: whether the promotion
  run of one unit accepts a checkout that also holds the unsigned pairs of
  its siblings (the command requires a clean checkout and validates the one
  unit; what its overlay validation does with unmanifested sibling modules
  was not run); and whether a run publishes its draft pull request when the
  candidate branch has moved.
- That `validate / validate` of `rulespec-us` passes a head with `k` v6
  manifests. It needs the reader switch on and the pin of checklist step 9.
- Whether `validate / validate` fails a head on which a module imports a
  module that the head does not hold. The candidate rule does not rest on
  it; before that rule a unit that was validated beside a helper that was
  dropped later depended on this check alone.
- Whether a signing run that is dispatched while a sibling's run is in
  progress sees the same candidate head, and what its admission reads cost
  against the anonymous limit when `k` runs start together.
- The duration of a signing run, and of CI for a head with `k` new modules.
- Whether `k` hosted runs make their anonymous reads from one address.
- The evaluator and the admission module were tested over fakes of the
  GitHub responses only, like the single-unit lane.

## 6. Merge queue and `notary-guard` (not implemented)

`evaluate_merge_group` is a stub that returns `fail`.

- A required check must also be reported on the merge-group commit. The planned
  algorithm computes the delta of the group head against its queue base, treats
  every v6 manifest as a unit `{M, P, T}`, and requires each unit to map to a
  queued PR with the same three blob ids for which `evaluate` passes.
- rulespec-us#2160 plans a group size of 1, which is recommended for the pilot.
  In the ruleset form the writer's lock requires it, with `ALLGREEN` and a
  build concurrency of 1: the merge-group evaluation binds the final tree to
  one judged pull request and does not prove that a group holds only that
  pull request (§5, "Encoder").
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
| 16 | PR rewrites `repository-checks.yml` so `validate / validate` passes | fail | yes: scope for a v6 unit, no lane otherwise |
| 17 | Truncated tree, an 11th page of reviews, a redirect, or HTTP 403/500 | fail; no cached pass | yes, for every read of a passing evaluation |
| 18 | Review dismissed between the first and second read | fail | yes |
| 19 | v6 manifest whose `candidate.pull_request` names another PR | fail | yes |
| 20 | PR containing only a v6 manifest, or two v6 manifests | fail | yes |
| 21 | Manifest `applied_files` digest differs from the head blob | fail | yes |
| 22 | `validate / validate` green on an older head, pending on `H` | fail | yes |
| 23 | Modification of a module that already exists on `main` | fail | yes |
| 24 | Merge group with two units, one unapproved | fail for the group | skipped: merge group is a stub |
| 25 | PR that changes only allowlisted files (docs) | not_applicable | yes |
| 26 | Unparseable JSON at a manifest path in the delta | fail | yes |
| 27 | Enrolled owner authored the commits; a third account opened the PR; the owner approves | fail, also when GitHub attributes the commit to no account | yes |

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

Added by security review round 2:

| # | Attack | Expected verdict | Test |
|---|---|---|---|
| 37 | Hand-written v5 manifest with content plus workflow, CODEOWNERS, toolchain pin or ledger | fail (no lane) | yes |
| 38 | v5-labelled PR with CI absent, red, or from another run | fail | yes |
| 39 | Hand edit or deletions under a doctored v5 manifest | follows CI: fail when red | yes |
| 40 | Content under `manual`, `programs/`, an unknown root, depth 3 or another case; `.axiom` pins and configuration; ledgers; `.github/**`; tooling; a new top-level entry | fail (no lane) | yes |
| 41 | An added or modified symlink, submodule or executable anywhere in the delta | fail | yes |
| 42 | A PR commit with no GitHub author or committer | fail | yes |

Added by security review round 3:

| # | Attack | Expected verdict | Test |
|---|---|---|---|
| 43 | `.gitattributes` that rewrites checked-out module bytes or hides diffs | fail (no lane) | yes |
| 44 | `CLAUDE.md` or `.claude/` change that steers agent sessions | fail (no lane) | yes |
| 45 | v5 modules "covered" by an unrelated manifest position, a wrong digest, a manifest under a non-jurisdiction, or a new jurisdiction | fail | yes |
| 46 | Allowlisted basename at another path (`sub/README.md`) | fail (no lane) | yes |
| 47 | Signing actors, status or attempt change before the verdict, or required CI changes on either lane | fail | yes |
| 48 | v5 manifest claims a non-atomic nested root under an existing jurisdiction, in either manifest layout | fail (no lane) | yes |

Cases 15-18 must also run live in the shadow period (§8).

## 8. Delivery plan

Implementation comes strictly before activation.

| # | Repo | Change | Rollback |
|---|---|---|---|
| 1 | axiom-encode | This change: evaluator, `merge-guard` CLI, tests, CODEOWNERS, this document. Nothing calls it | revert |
| 2 | axiom-encode | Verifiers can read `applied-rulespec/v6` with the attestation scope, behind `V6_READER_ENABLED = False`; no writer | revert; no v6 exists |
| 2b | axiom-encode | Set `V6_READER_ENABLED = True`. May ride with change 3. Switch rule: The reader may be switched on while no v6 writer exists: nothing new can land. The v6 writer and the admission without approval must stay off until the guard check is required. With the reader on, the writer on and the guard check not required, an enrolled dispatcher lands their own unreviewed candidate with no approval, and any account with write access re-lands signed v6 bytes from a closed pull request. | set it back; no v6 exists |
| 3 | axiom-encode | Workflow that runs the guard from `main` and publishes its check, with a dry-run mode | disable the workflow |
| 4 | rulespec-us | Pin the encoder that verifies v6 | revert the pin |
| 5 | axiom-encode | Emit v6 and admit without approval, behind a default-off switch. Built with three locks: `V6_WRITER_ENABLED = False`, `GUARD_APP_ID = None`, and a live read that the guard check is required (§5, "Encoder") | revert |
| 5b | axiom-encode | Record the guard App's numeric App ID in `GUARD_APP_ID`, after checklist step 1, which is on hold: nobody may store the guard App key until the mechanism of `docs/merge-guard-key-protection.md` is chosen, implemented and independently reviewed (#2257). Until then no App ID exists to record. The other half, the hosted workflow's evidence step reading `candidate_rulespec_ref` behind the writer's locks, is present and switched off (§5, "Evidence step") | set it back to `None` |
| 5c | axiom-encode | **Writer lock for the ruleset form (built, locked).** A second accepted form of locks 2 and 3: `GUARD_WORKFLOW = None`, and a live read that an organization ruleset requires the guard workflow at a recorded commit on `main`, with a merge queue rule of the repository's own ruleset that merges every entry alone and only on its own passing checks (`ALLGREEN`, group size 1, build concurrency 1; §5, "Encoder"; the queue rule cannot be an organization rule, and the lock proves only that it was present, with those settings, at each of its reads). Exactly one of `GUARD_APP_ID` and `GUARD_WORKFLOW` may be recorded; with neither, as committed, `require_guard_enforced` refuses before any read of its own. Needed only if the maintainers accept the ruleset mechanism. The listing it is tested against is SYNTHETIC. [experiment X14]: must be replaced by the recorded live response before activation. | revert; the locks stay shut |
| 6 | axiom-encode | **Activation**: turn the switch on; update `docs/reviewed-pr-promotion.md` | flip the switch back; the required check stays |

### Activation checklist

Steps marked **M** need repository or organisation settings. Only a maintainer
can do them. Maintainers must accept the proposed promotion order before
activation.

1. **M** **On hold. Do not carry out this step as written.**
   `docs/merge-guard-key-protection.md` (its second paragraph) suspends it:
   nobody may store the guard App key or set `MERGE_GUARD_RUNNER_ENABLED`,
   shadow mode included, until Max chooses a mechanism and complement,
   maintainers implement them, and a fresh independent review verifies the
   whole key-exposure set (§6 of that document), which covers every workflow
   that can reach the key. Until then follow #2257, not this step; #2237
   replaces it. The superseded instruction, kept only so that references to
   "step 1" stay readable: create the guard App, install it on `rulespec-us`,
   and store its key in an axiom-encode environment restricted to branch
   `main`. A branch restriction alone does not protect the key.
2. Merge changes 1-4, including 2b. Publish the check in shadow mode: visible,
   not required. **The shadow check is on hold with step 1:** it needs the
   guard App's key, and `docs/merge-guard-key-protection.md` forbids storing
   the key or setting `MERGE_GUARD_RUNNER_ENABLED`, shadow mode included,
   until its conditions are met (#2257).
   Switch rule: The reader may be switched on while no v6 writer exists: nothing new can land. The v6 writer and the admission without approval must stay off until the guard check is required. With the reader on, the writer on and the guard check not required, an enrolled dispatcher lands their own unreviewed candidate with no approval, and any account with write access re-lands signed v6 bytes from a closed pull request.
3. Run cases 15-18 live on a throwaway PR and confirm every verdict.
4. **M** Add the guard's check with the guard App's `app_id` to the required
   checks on `rulespec-us` main, only after question 0 of §10 is decided: under
   the current classifier most maintenance PRs get `fail`.
5. Run the current-order canary (rulespec-us#2045) to completion.
6. Merge change 5 with every lock shut, then change 5b. Change 5 supplies
   the locked writer and opens nothing; change 6 (step 7) opens it.
7. Confirm that `main` requires `content-bound-merge-guard` from the guard
   App for administrators too (the command is in §5: read the public branch
   record). Then merge change 6. The admission refuses while that read does
   not show the requirement. The first promotion in the new order is the
   pilot.
8. **M**, optional hardening: enforce code-owner review for `/.github/` (§9).
9. **M**, before the merge queue (rulespec-us#2160): confirm the guard reports
   on group SHAs in a test repository.

Steps 4 and 7 are separate reviewed changes, and step 4 comes first. That
preserves proposed constraint D2d.

If the maintainers accept the ruleset mechanism instead of the guard App
(change 5c), the checklist above does not apply as written. The ruleset
mechanism and its activation checklist are in **#2287**
(`docs/merge-guard-key-protection.md` and the activation checklist of this
document as that pull request rewrites them, with the experiment runbook
`docs/merge-guard-ruleset-experiment.md` that it adds). They are not on
`main`: the recommendation of `docs/merge-guard-key-protection.md` on `main`
(its section 4) is the App-based mechanism (b), not the ruleset. Follow the
checklist of #2287 once it has merged; until then nothing here can be
activated in the ruleset form.

The conditions below are what this lock adds to that checklist before
change 6 merges. X14 alone is not enough: it records only the shape of the
listing. The lock's safety also rests on three behaviours of GitHub that the
lock cannot read, and #2287 has an experiment for each. Each of them must
have the result `pass`, recorded as #2287 prescribes, before change 6
merges:

- **The pin (X1 of #2287; X5 for the merge group).** A required workflow
  pinned to a commit SHA runs exactly the file of that commit, also after
  the source branch moves, for a pull request and for a `merge_group` run.
  If it fails, the recorded SHA set of `GUARD_WORKFLOW` is evidence of
  nothing, and the lock must stay shut. X19 (a pin move with an open green
  pull request) is recorded with it.
- **A missing run blocks (X8 of #2287, parts (a) and (b)).** A head that
  gets no run of the required workflow (a push by a job token, `[skip
  ci]`) cannot merge. If it fails, a head with no guard run can merge while
  this lock shows the rule as present.
- **A dismissal and the queue (X15 of #2287, parts (a), (b) and (c)).** A
  pull request whose approval is dismissed before or in the queue does not
  merge, and the queue merges only after the guard's merge-group job has
  ended. The queue of that experiment has the settings this lock requires
  (X5: minimum and maximum group size 1, build concurrency 1). If it
  fails, the merge-time judgment that this lock relies on does not exist.
- **The queue's settings.** A `rulespec-us` administrator sets the merge
  queue to: only merge non-failing pull requests (`ALLGREEN`), minimum and
  maximum group size 1, build concurrency 1. The lock refuses any other
  value at every read (§5, "Encoder"); the status-check timeout and the
  merge method are #2287's to state.
- **No classic merge queue.** A `rulespec-us` administrator confirms that
  the classic branch protection of `main` has no merge queue. The lock
  cannot see one, and which queue governs when both exist is not known;
  the question belongs in the owner experiment of #2287 as one more
  recorded observation. Until it is answered there, a classic queue beside
  the ruleset's queue is a reason not to activate.

- Record the live response of
  `GET /repos/TheAxiomFoundation/rulespec-us/rules/branches/main` with the
  ruleset `Active`, replace the SYNTHETIC fixture with it, and make the
  ruleset-form tests pass against it unchanged or with a reviewed change of
  the lock.
  [experiment X14]: must be replaced by the recorded live response before
  activation.
- Record `GUARD_WORKFLOW` in one reviewed change: the repository id of
  axiom-encode, the workflow path and the reviewed commit SHA or SHAs.
  `GUARD_APP_ID` stays `None`.
- A maintainer with access confirms that the bypass list of the
  organization ruleset is empty, and that the bypass list of the repository
  ruleset that holds the merge queue of `rulespec-us` is empty. The lock
  cannot read either list. This confirmation happens once; the lock does
  not repeat it, and a later entry shows only in the audit log.
- The maintainers accept in writing the limit of the queue half: the merge
  queue rule is in a ruleset that the administrators of `rulespec-us` can
  edit, disable or give a bypass list, and the lock proves only that the
  rule was present at each of its reads (§5, "Encoder").
- The live response of X14 is compared with the two shape decisions of the
  lock: a `workflows` rule with a key outside `type`, `parameters`,
  `ruleset_source_type`, `ruleset_source`, `ruleset_id`, or with
  `do_not_enforce_on_create` other than absent or `false`, keeps the lock
  shut until a reviewed change of the lock.

What fixes the order. D2d ("deployed and enforced before the pre-sign approval
is removed") forces the guard App (step 1), the runner and the v6 reader pin
(step 2) and the required check (step 4) before activation (step 7). The
canary (step 5) follows the proposed sequence. The shadow period and
the live cases (steps 2 and 3) are this document's own choice, not a
requirement of D2d.

## 9. Threat model

| Adversary | Defended? | How |
|---|---|---|
| Write collaborator, not enrolled | Yes for a v6 unit; through CI for a v5 lane; blocked otherwise | Cannot dispatch, cannot give a counting approval, cannot publish as the guard App. A v6 unit refuses every extra file. A v5 lane needs this PR's own green run. Any other guarded change has no lane |
| PR that edits workflow files, pins, ledgers or `.axiom/` configuration | Yes, by refusal | No lane claims it: `fail`. This also blocks legitimate changes (§10, question 1) |
| PR that relabels or hides its manifest, or uses a root the encoder does not name | Yes | The guard stands aside only on the five-file allowlist or on the v5 lane, which needs this PR's green CI (step 3) |
| PR that changes `CLAUDE.md`, `.claude/` or `.gitattributes` | Yes, by refusal | Instruction and settings files steer the agents that hold the enrolled identities; Git attributes change the bytes CI validates. Both are guarded with no lane |
| Hand-written v5 manifest | Through CI | The guard does not verify the signature. The lane's scope keeps the PR from changing the workflow and the pin, and the PR's own `validate / validate` run must be green |
| `validate / validate` produced by a run that is not this PR's `pull_request` run | Yes | Step 7 |
| Forged or unsigned v6 manifest | Partly | Current-head CI verifies the signature; the guard does not (open question 3) |
| Second workflow imitating the check name | Yes, once the App binding exists | `app_id` binding |
| Stale approval after a content change | Yes | Blob equality |
| Push racing an approval | Yes | The verdict belongs to one head and is computed from that head's tree |
| Dismissal just after a pass | Partly | The check stays green until the next sweep |
| GitHub API failure | Fails closed for new verdicts | A green check already written is not withdrawn during an outage |
| Repository admin, or theft of the guard App key | No | Out of scope |
| Semantic change brought in by the base while the approved bytes stay the same | Partly | Current-head CI catches it; the human approval does not cover it |
| Commit authored by the approver but not attributed by GitHub | Yes, by refusal | An unattributed commit gives `fail` (case 42). A commit attributed to a third account by a forged author email is not detected |

**Residual risk: agents acting under the two enrolled identities.** The guard
authenticates a GitHub account, not a person. If agent sessions hold both
owners' credentials, an agent can author as A, dispatch as A and approve as B,
and every check passes. The guard reduces this to "both accounts must act" and
records who did. It cannot prove that a human read the bytes. Closing this would
need approval credentials that agents cannot hold, for example a hardware-backed
signed approval (option c in §3) or a separate approver account.

**Residual risk the owners would accept on activation (proposed; they have
not accepted it): an enrolled owner who is also a repository administrator.** Lock 3 of §5 is read again by the encoder, which keeps no local copy
between reads (GitHub serves the anonymous branch record with
`cache-control: public, max-age=60, s-maxage=60`, so an answer can be up to
about a minute old), at five points of a signing run: the preflight (start and end), the root
provisioner (start and end), before validation, immediately before signing,
and before publication. After publication only the GitHub setting protects
the merge. Today the approval before signing is enforced by encoder code,
independent of repository settings. After activation an enrolled owner with
administrator rights can sign their own candidate, then remove the required
check (or lower `enforcement_level`, or use the guard App key) and merge with
no second approval. The table above puts administrator abuse out of scope;
this paragraph states what that costs. What exists against it: the
organisation audit log records the settings change, and the push to `main`
still runs validation. Neither prevents the merge, and neither supplies a
review.

## 10. Open questions

| # | Question | Recommendation |
|---|---|---|
| 0 | **First decision.** Which lane do infrastructure changes (`.github/**`, `.axiom/` pins and configuration, ledgers, tooling, `.gitattributes`, `CLAUDE.md`, `.claude/`) and ProgramSpec changes get? Today none: `fail`. 48 of the last 50 merged PRs are in this class (49 before #2045 merged) | Decide before the check is required. Recommended: three classes ("Proposed three-class lane" in §3; implemented behind `INFRASTRUCTURE_LANE_ENABLED = False`): content keeps its lanes; every other known path is control plane and needs one content-bound approval from an enrolled owner; the stand-aside class is empty; content mixed with anything else is refused. Alternatives are listed there |
| 0b | A module under a jurisdiction that the base lacks has no lane, also with a v5 manifest. How is a new jurisdiction added? | Through the lane of question 0 |
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
| 12 | The v5 lane is `not_applicable` on scope plus this PR's green CI, with no approval check. Enough? | Yes while admission checks the approval before signing |
| 13 | A signed v5 PR that also changes a receipt, pin, ledger or test (rulespec-us#1470) has no lane. Widen the v5 lane, or split such PRs? | Split, or give the extra files the lane of question 0 |
| 14 | Add `run_attempt` to `attestation_scope`? | Recommended: yes, before or with change 6. Not decided; change 5 defines the writer without it |
| 15 | The approver may not be the author or committer of any commit, so the approver cannot click "Update branch". Acceptable? | Yes for the pilot |

## 11. What was not verified

A read-only live run, `merge-guard --repository rulespec-us --pr 2045`, returned
`not_applicable` / `earlier_lane_v5` on 2026-11-12, after that PR's CI went
green. It is a v5 promotion PR: one root-layout manifest and two module files.
The run exercised the real check-run listing and the real `pull_request` run
binding. The green head carried exactly one check run named
`validate / validate`. After #2045 merged, the same command returns `fail` /
`pr_identity`: the evaluator judges open PRs only.

Verified against live data on 2026-11-12 (rulespec-us#2045 and its signing run):
the fields of an Actions run (`event`, `path`, `head_sha`, `head_branch`,
`repository`, `head_repository`, `pull_requests`, `actor`,
`triggering_actor`), the `details_url` form of a check run, the v5 promotion
manifest's `applied_files` paths, the two manifest directory layouts on
`rulespec-us` main, and the attribution of the signing workflow's manifest
commit.

Not verified:

- Whether `pull_requests` on an Actions run stays populated after the PR's
  branch moves, and for a re-run. It is empty for fork PRs, which the guard
  refuses anyway.
- Whether a required check with a non-Actions `app_id` is honoured on
  merge-group commits, and how to list a group's PRs.
- Change 5c: the response of `GET /repos/<repo>/rules/branches/main` for a
  rule of type `workflows`. No such ruleset exists. Read on 2026-11-17: the
  REST schemas `repository-rule-workflows`,
  `repository-rule-params-workflow-file-reference` (`path` and
  `repository_id` required, `ref` and `sha` optional) and
  `repository-rule-ruleset-info` (`ruleset_source_type`, `ruleset_source`,
  `ruleset_id`) in `github/rest-api-description` at `c9c95fe3`; the only
  documented example of the endpoint shows two pattern rules. Whether the
  response carries `sha`, whether it also carries a `ref`, and the exact
  `ruleset_source` of an organization ruleset are not verified. The fixture
  is SYNTHETIC.
  [experiment X14]: must be replaced by the recorded live response before
  activation.
- Change 5c: that an anonymous read of that endpoint works was verified
  for `rulespec-us` and axiom-encode on 2026-11-17 (HTTP 200). That the
  listing includes a rule of an organization ruleset for an anonymous
  reader is not verified.
- Which workflow file version `pull_request_review` and `merge_group` runs use.
- Organisation plan and availability of required-workflow rulesets.
- Whether the Go signer or supervisor inspects `schema_version` or field names.
  Their tests mention `applied-rulespec/v5`; the code was not read.
- Whether workflow-run metadata for `manifest.run_id` stays readable for the
  life of a PR. A deleted run makes step 6 fail closed.
- The shapes of v5 retire, path-migration and legacy manifests, and so whether
  those lanes get `not_applicable` or `fail`.
- File modes and CI state of the PRs in the census.
- The content layout of `rulespec-<country>-<subdivision>` repositories. The
  evaluator assumes the content roots sit at the root or one level down.
- Whether `pr.base.sha` follows the `main` tip. The evaluator does not use it.
- Tree size: a recursive tree that GitHub truncates gives `fail` for every PR
  of that repository.
- The guard App does not exist; its id, installation and key custody are
  assumptions.
