# Content-bound merge guard

Status: **design; maintainer acceptance and activation remain pending.** At
this branch's source, which merges `main` at
`fad29b3ed4019fe7042552a7958542dfcb41e4e0`, the evaluator
(`src/axiom_encode/content_merge_guard.py`) supplies verdicts, the production
manifest writer emits v5, and the shadow runner offers dry-run and shadow modes only.
`INFRASTRUCTURE_LANE_ENABLED` remains `False`, so the proposed control-plane
lane in §3 supplies no current verdicts. The enforcing runner of change 3b
exists and is required nowhere: no organization ruleset names its workflow
(§4, "Enforcing runner as built"). `docs/merge-guard-key-protection.md`
recommends that an organization ruleset require that workflow, with no guard
App and no private key (option (g')). **Max has not accepted the
recommendation, and nothing is chosen.** It holds only if the owner's
experiment passes (`docs/merge-guard-ruleset-experiment.md`, not run). The
shadow workflow
(`.github/workflows/merge-guard-runner.yml`, change 3 of §8) has only a manual
dispatch trigger. Its gate job checks `main` and
`vars.MERGE_GUARD_RUNNER_ENABLED == 'true'`; its `run` job repeats
those two checks in its first step and then checks for App credentials. The
Python runner and CLI accept a token-shaped `MERGE_GUARD_TOKEN`
wherever someone starts them; they do not enforce the workflow's startup gate.

Nobody may store the guard App key or enable the runner, shadow mode included,
until Max chooses a mechanism and any complement, maintainers implement it,
and fresh independent evidence covers the whole key-exposure set (§4 and
checklist step 1 of §8). That set includes every job in any existing, new,
replaced or modified workflow that can select `merge-guard`, dynamic selectors,
the ability to land those workflows, and all code and persistence before or
after token creation. A valid copied key can mint tokens for any App
installation within that installation's granted repositories and permissions;
one token's scope or lifetime does not confine the key. Recording these
conditions does not satisfy them. Under the recommended option no guard App
key is ever created; under either fallback this hold applies in full.

The v6 reader (`src/axiom_encode/applied_manifest_v6.py`, change 2 of §8)
keeps `V6_READER_ENABLED = False`. Every verifier therefore refuses v6 at
this source head. Every promotion still follows
`docs/reviewed-pr-promotion.md`. Tracking issue: #2172.

The design follows the agent-proposed answers in
`docs/reviewed-pr-promotion-order-proposal.md`. An agent posted those answers
under @MaxGhenis's account; Max has not ratified them. D1, D2a-D2f, D3 and Seq.
identify proposed constraints throughout this design and related source
comments, not maintainer acceptance. Platform claims without confirmation carry **[check docs]**.
Text that the recommendation change added uses the labels of
`docs/merge-guard-key-protection.md`: [docs Dn] (a page of that document's
list, opened 2026-11-17), [code], [live read 2026-11-17], [author-reported],
[inferred] and [experiment Xn] (a step of the experiment document).

## 1. What exists today

The following admission baseline describes `main` at `eb98f8d7` (after
#2194); later sections describe the source at `ca39ebb2`. The merge from
`main` added the infrastructure proposal with its switch off. It did not
activate a lane, accept the promotion order or choose a key-protection mechanism.

- **Admission** (`src/axiom_encode/reviewed_pr_admission.py`):
  - Enrolled identities by numeric id: `TRUSTED_SIGNING_REVIEWERS`.
  - Fixed origin, no proxy, no redirects, bounded reads: `github_json`,
    `_pages`.
  - Complete tree delta: `_leaf_entries` and `_require_immutable_delta`.
  - Effective reviews: `_approved_reviews`. It requires an approval on the exact
    head, excludes the PR author by id and the dispatcher by login, and refuses
    on a `CHANGES_REQUESTED` from any account.
- **The encoder checks approvals at three points**: twice in
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
- **`notary-guard`** (`notary/merge_guard.py`) guards generated-file lineage
  for activated notary lanes. It returns 78 when `.axiom/notary/consumer.json` is
  absent at the base. It never reads reviews. The new guard is a sibling, and
  `merge_guard.py` stays unchanged.
- **`rulespec-us` main protection** (read live on 2026-11-12): required check
  `validate / validate` with `app_id: 15368` (GitHub Actions), `strict: true`,
  `enforce_admins: true`, `required_approving_review_count: 0`,
  `require_code_owner_reviews: false`, `dismiss_stale_reviews: false`.
- **The 2026-11-12 protection read found no binding to one workflow publisher.** Any Actions job named
  `validate / validate` satisfies app 15368, and the workflow that produces it
  runs on `pull_request`, so the pull request's own copy of the caller runs.

<a name="2-binding-decisions-and-where-each-lands"></a>

## 2. Proposed constraints and where each lands

| # | Proposed constraint | Design element |
|---|---|---|
| D1 | "The signing service may sign validated, unreviewed candidate bytes when a versioned signed manifest explicitly records that scope. Verifiers must distinguish validation from owner approval." | §5: schema v6 with `attestation_scope` |
| D2a | "Its code and the enrolled identities load from protected policy." | §4: guard code and identity list on encoder main |
| D2b | "The required check is bound to its trusted publisher." | §4: recommended, not accepted: an organization ruleset that requires the guard workflow at a pinned commit. Fallback: a dedicated GitHub App, required with its `app_id` |
| D2c | "It checks the complete final PR delta, canonical paths, approved content hashes, live effective reviews and current-head CI." | §3 steps 1-7 |
| D2d | "It is deployed and enforced before the pre-sign approval is removed." | §8: changes 1-5 before change 6 |
| D2e | "An approval is retained across base-only updates only when the approved content and the permitted scope stay unchanged." | §3 step 5 |
| D2f | "The final approver is independent of both the candidate author and the authenticated dispatcher." | §3 step 6 |
| D3 | "enrolled identities at first. An author may dispatch their own candidate. A different enrolled owner gives the final approval." | §3 step 6; §5 |
| Seq. | "Build the merge gate of proposed answer 2 and pass its adversarial tests." | §7, §8 |

## 3. Binding an approval to content

The designers considered three options.

| Option | Verdict |
|---|---|
| (a) Tree comparison: an approval on commit C counts if the guarded blobs at C equal those at the current head | **Recommended; the evaluator implements it** |
| (b) Approver states digests in the review body | Rejected: an agent can paste them, humans mistype them, and it adds no binding that (a) lacks |
| (c) Separate signed approval artifact | Rejected for v1: needs per-owner keys and custody, and would run under the same sessions as the GitHub identity. Revisit if the maintainers judge the §9 residual risk too high |

### Algorithm `evaluate(repository, pr, fetch=...)`

`content_merge_guard.evaluate` runs these steps. `repository` is a short name
such as `rulespec-us` or `rulespec-us-tn`. The evaluator reads live state through the injected `fetch`. Any error, truncation or disagreement gives `fail`. The
function never raises. A verdict is `pass`, `fail` or `not_applicable`, with a
reason code and evidence.

1. **PR identity.** Read the PR and take its head `H` and branch. Reuse
   `_pr_identity`: open, unmerged, same-repository head, base `main`. Let `B` be
   the live `main` tip. `compare(B...H)` must be `ahead` with merge base `B`.
2. **Complete delta.** Read the complete trees of `B` and `H` with
   `_leaf_entries`. The evaluator computes their leaf delta, `Δ(H)`.
3. **Applicability, from paths and modes only.** The guard stands aside in two
   cases and no other. For neither does it trust a label inside a file that
   the PR supplies.
   - Any leaf that the PR adds or modifies without regular `100644` blob mode, anywhere:
     `fail`.
   - **Allowlist.** Every delta path is exactly one of `README.md`, `LICENSE`,
     `LICENSE-CODE`, `NOTICE`, `.gitignore` at the repository root:
     `not_applicable` (`no_guarded_content`). The list is encoder policy
     (`UNGUARDED_ROOT_FILES`), and the guard matches the whole path. Every other path is
     *guarded*: every jurisdiction directory at every depth and root name,
     `programs/`, all of `.axiom/`, all of `.github/`, the root ledgers
     (`known-*.yaml`, `oracle-coverage-pending.yaml`,
     `household-choices.yaml`), `tools/`, `bulk/`, `tests/`, `.gitattributes`
     (it changes checked-out bytes and review diffs, so CI and the guard would
     no longer judge the same bytes), `CLAUDE.md` and `.claude/` (they steer
     the agent sessions that author, dispatch and approve), and any new
     top-level entry.
   - A guarded delta needs one lane to claim the whole delta.
     A manifest is a file under `.axiom/encoding-manifests/` at the repository
     root or under one jurisdiction directory (both layouts exist on
     `rulespec-us` main). No changed manifest: `fail`
     (`no_lane_for_guarded_change`). The guard returns `fail` if the PR removes a manifest, changes more than
     `MAX_MANIFEST_BLOBS` (256) manifests, supplies anything other than one
     JSON object per manifest, includes a duplicate key or non-finite number,
     or supplies a `schema_version` other than v6 or v5.
   - **Earlier lane (v5).** Every changed manifest carries the v5 label, and:
     - every other delta path is a RuleSpec module file,
       `<jurisdiction>/<atomic root>/.../<name>.yaml` (or
       `<atomic root>/...yaml` at the root), with no hidden path segment, under
       a jurisdiction that already holds modules on the base tree. The
       encoder recognizes `legislation`, `policies`, `regulations` and
       `statutes` as atomic roots. So the delta has no path
       under `.github/`, no other `.axiom/` file, no ledger, no ProgramSpec, no
       allowlisted file and no new jurisdiction. Otherwise `fail`
       (`no_lane_for_guarded_change`);
     - every changed manifest sits at the position of one module of a base
       jurisdiction, in either layout. Otherwise `fail`
       (`no_lane_for_guarded_change`);
     - every changed module file appears in `applied_files` of the manifest
       at its own canonical position (a `.test.yaml` belongs to its primary),
       with the sha256 of its head bytes; a removed file needs the listing
       only. Otherwise `fail`;
     - every changed manifest lists at least one changed module file, so an
       empty listing, or a manifest-only change, has no lane: `fail`;
     - `validate / validate` passes on `H` in this PR's own `pull_request`
       run, with the binding of step 7. Otherwise `fail`;
     - the PR and `main` did not move during the evaluation, and a final read
       confirms the same green check identities and observed run attempts.

     Then `not_applicable` (`earlier_lane_v5`). The guard does not verify the v5 signature and cannot tell a signed
     manifest from a hand-written one. It relies on the `validate / validate`
     run, which verifies the signature, and on the lane's scope, which proves
     that the PR changed neither the workflow files nor the toolchain pin that
     the run executes. Admission checks the approval of a v5 lane before
     signing; the guard does not check it again.
   - **v6 lane.** v6 together with v5: `fail`. Two or more v6 manifests:
     `fail`. Exactly one changed manifest and it is v6: continue. A repository
     outside `rulespec-<country>`: `fail`. A draft PR: `fail`.
   - Everything else that touches a guarded path has no lane and gets `fail`:
     workflow changes, `.axiom/` configuration and pins, ledgers, ProgramSpec,
     tooling, and mixed changes. The evaluator refuses them; no enforcing check exists yet. The first
     open question (§10) asks which lane they get.
4. **Scope and canonical paths.** Let `M` name the v6 manifest.
   - `_require_immutable_delta` cross-checks the tree delta against the PR file
     list and the compare file list.
   - `M` has exactly the v6 shape of §5.
   - `manifest.attestation_scope.candidate` names this repository, this PR and
     the live PR author id, and `candidate.head_sha` equals
     `candidate_rulespec_ref`.
   - Take `P` from `citation_rulespec_path(manifest.citation)`, call its companion
     `T`, and require `M` at the path `_applied_encoding_manifest_path(P)` gives.
   - `Δ(H)` is exactly `{P, T, M}`, all added. Any other path gives `fail`,
     including `.github/**`, CODEOWNERS, a second module and the derived index
     `.axiom/index/provisions_to_rules.json`. The live lane still permits the
     index (`reviewed_pr_admission.INDEX_PATH`), because RuleSpec repositories
     tracked it when that lane started. `rulespec-us` stopped tracking it
     (rulespec-us#2152), and an index outside the approved set could change
     after the approval, so the guard is stricter.
   - `sha256(P)` and `sha256(T)` equal `manifest.applied_files`. The guard
     checks each blob it reads against its Git object id.
   - The PR commit list contains `candidate.head_sha`, and `P` and `T` at that
     commit have the same blobs as at `H`. `candidate.base_sha` is `B` or an ancestor of `B`.
5. **Content-bound approval.** Take the effective review per reviewer (the
   latest of `APPROVED`, `CHANGES_REQUESTED`, `DISMISSED`) and ignore any review
   from a deleted account. An `APPROVED` review on commit `C` counts only if
   all hold:
   - The PR's commit list contains `C`.
   - With `B_C = merge-base(C, main)`, the path set of `Δ(C)` equals
     `{P, T, M}`.
   - The type, mode and Git blob id of `P`, `T`, `M` at `C` equal those at `H`.

   Consequences:
   - An approval that an owner gives before signing never counts, because the
     tree at `C` lacks `M`.
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
   and the enrolled list must hold that identity (D3). The reviewer must satisfy all of:
   - id in the enrolled list, matching login, `type == "User"`;
   - id differs from the PR author;
   - id differs from `dispatcher.id`, and from `actor.id` and
     `triggering_actor.id` of the run;
   - id differs from the GitHub-attributed author and committer of every
     commit of the PR. A commit whose author or committer GitHub cannot
     attribute to an account gives `fail` (`unattributed_commit`), because Git
     author metadata is free text and the evaluator cannot tell that the
     approver did not write it. To fix it, re-author the commit with an email
     that GitHub links to an account, and sign again. The guard exempts no bot:
     GitHub attributes the signing workflow's manifest commit to the
     dispatcher's account, with `web-flow` as committer (observed on
     rulespec-us#2045, commit `9be9d91e5`), and neither can approve.
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
     another ref's definitions, so the guard refuses it.
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

- A v5-lane PR is `not_applicable` on the strength of CI and scope; this
  guard does not check its approval.
- An infrastructure, ledger, ProgramSpec or tooling PR has no lane and gets
  `fail`. Once branch protection requires the check, the guard blocks every
  such PR until the maintainers decide its lane (§10, question 0).
- A signed v5 PR that also changes another tracked file has no lane. Observed:
  rulespec-us#1470 (a signed manifest together with
  `.axiom/legacy-replacements/<digest>.json`, `.axiom/toolchain.toml`,
  `known-validation-gaps.yaml` and a test file). The design did not widen the lane.
- A fork PR and a stale PR get `fail`, not `not_applicable`.
- 1000 reviews on one PR, also comment-only reviews from outside accounts,
  exhaust the page bound and give a permanent `fail`. A new PR fixes it.
- A re-run of the signing run by another person changes `triggering_actor` and
  gives `dispatcher_mismatch` permanently. A new signing run fixes it.
- An approver who clicks "Update branch" becomes a commit author and can no
  longer approve.
- A run or review can change after its last read. GitHub supplies no transaction
  across these endpoints. Under the App path the next sweep catches the
  change; under the ruleset path only the merge-group run does, which
  evaluates after the group's own required check has completed (§6).
- A write collaborator can block the remaining approver by pushing a commit
  that GitHub attributes to that approver, or cannot attribute to an account. Dropping the commit
  fixes it and needs a new signing run.
- Two PRs on one head both appear in the CI run's `pull_requests`; the guard
  accepts the run for each.
- A v5 lane whose manifest does not list every changed module file in
  `applied_files` gets `fail`. The retained source checks do not establish the retire or
  path-migration manifest shapes.
- The pilot covers `rulespec-us`. Before extending it, maintainers must review
  the other repositories' layouts and classify `data/` and `data/corpus` as
  guarded control plane; those paths must never enter a stand-aside allowlist.
  The retained cross-repository observations do not establish a supported lane
  for those repositories.

### Census: the last 50 merged `rulespec-us` PRs under this classifier

The original author reported a 2026-11-12 census, repeated after
rulespec-us#2045 merged (before that the counts were 1 and 49). The report
sampled the 50 most recently merged PRs to `main` among the 100 most recently
updated closed PRs and applied the rules of that date: reduced allowlist,
digest and position binding. It read file lists, changed manifests and module
bytes at each merge commit, and took jurisdictions from `main` at the read.
It assumed regular modes and passing CI without reading either. The author
reported no v6 manifest in that sample, so none could get `pass`. These
historical results lack a reproducer and pinned inputs here; they do not
establish current inventory or fresh runtime verdicts.

| Author-reported verdict | PRs |
|---|---|
| `not_applicable` (`no_guarded_content`, allowlist) | 0 |
| `not_applicable` (`earlier_lane_v5`) | 2 (#2045 and #2154, a merge train of 5 signed manifests; the report says all 5 satisfy the digest and position rules) |
| `fail` (`no_lane_for_guarded_change`) | 48 |
| `fail`, other reasons | 0 |

The report groups the 48 under toolchain and corpus pins and staged waiver fingerprints in
`.axiom/toolchain.toml` or `.axiom/workflow-toolchain.toml` (about 21), ledger
and register changes (`known-*.yaml`, `oracle-coverage-pending.yaml`), workflow
changes under `.github/`, ProgramSpec changes under `programs/`, tests and
tooling, and one signed manifest PR with extra files (#1470). Over a wider
window of 200 merged PRs the author counted 136 no-lane, 2 earlier lane and 62
other; the 39 of those 62 that the census listed all carried the historic
`applied-rulespec/v1` label, which predates v5.

For that reported 50-PR window, the classifier would refuse 48 PRs under the
stated assumptions. Maintainers should not make branch protection require it
until they decide which lane these changes get.

### Proposed three-class lane (off; awaiting a maintainer decision)

`INFRASTRUCTURE_LANE_ENABLED` in `content_merge_guard.py` is `False`. The
constant has no runtime override. The evaluator takes the proposal's branch
only when it is exactly `True`; with it off, the earlier algorithm still
applies. Question 0 of §10 remains open.

The v5 lane trusts this PR's `validate / validate` run. Its scope checks that
this PR did not change the workflow, pins or registers of accepted failures;
it does not check who approved an earlier change to them. That leaves a
possible two-PR attack if the guard stands aside for such changes: PR A
weakens validation without approval; PR B supplies a v5-labelled manifest and
uses the weakened CI to enter the earlier lane. This is the rationale for
requiring an approval of control-plane changes, rather than allowing them to
stand aside.

The #2231 author reported that the shared validation workflow runs
`python -m pytest -q tests` from the PR head in `validate / validate`, and that
those tests import `tools/source_conflicts.py`, `tools/household_choices.py`
and `bulk/applied_artifacts.py`. The report also names
`tests/test_encoding_manifests.py`, `tests/conftest.py` and
`tools/engine_root_load.py` as inputs that can change the check's result, and
cites rulespec-us#1472 as a narrowing of a manifest-sync guard test. The retained report supplies those external-source observations without
pinned inputs or a reproducer here.
The proposed policy treats anything required validation executes, imports or
reads from the PR head as control plane. Its named paths cannot establish
that they cover that whole set.

`classify_path` sorts each changed path from the path alone:

| Class | Paths | Rule with the lane on |
|---|---|---|
| unguarded | Allowlisted root files other than `.gitignore`: `README.md`, `LICENSE`, `LICENSE-CODE`, `NOTICE` | A delta containing only these files gets `not_applicable` |
| content | Every path at or under a jurisdiction directory that exists on the base or has jurisdiction shape; every path under an atomic root at the repository root; every apply-manifest directory | The v5 or v6 lane must claim the whole delta; the v6 lane requires a content-bound approval |
| control plane | `.gitignore`; `.github/`, `.claude/`, `programs/`, `tests/`, `tools/`, `bulk/`; `.axiom/` files outside the apply-manifest directories; every root file off the allowlist, including new root files | One content-bound approval covers the whole delta: `pass`, `control_plane_content_bound_approval` |
| infrastructure (stand aside) | Exactly `INFRASTRUCTURE_PATHS`, an empty set | An infrastructure-only delta would get `not_applicable`, `infrastructure_as_today`; no path currently belongs to this class |
| unknown | Other top-level directories and paths that fail the canonical-path check | No control-plane or stand-aside lane; content lanes still apply their own rules below |

With the switch off, `.gitignore` stays on the old allowlist. With it on,
`LANE_GUARDED_ROOT_FILES` moves `.gitignore` to control plane. The #2231 author
reported that a rulespec-us test runs `git check-ignore`; the implementation
makes the classification explicit without depending on a new inventory read.

`INFRASTRUCTURE_PATHS` stays empty. Adding a path would require evidence that
required validation neither executes nor reads it. The #2231 author considered
Markdown files under `bulk/` and `tools/`, but reported only a search for named
references. A test can walk a directory, so that search does not establish
that those files qualify. The author's 2026-11-13 tree report lists `.axiom`,
`.claude`, `.github`, `bulk`, `programs`, `tests` and `tools` as the
non-jurisdiction top-level directories. Neither report establishes a current
external tree or the full validation-input set.

The classifier calls a path unknown if it is empty, has an empty or dot
segment, a backslash, space, control or non-ASCII character, or a segment
that matches a guarded name only after case folding
(`.axiom/ENCODING-MANIFESTS/...`, `Tests/x.py`, `readme.md`). It classifies
only guarded paths; allowlisted files can accompany them:

- The control-plane and stand-aside lanes cannot admit content. A content
  lane must claim the whole delta, under its own path rules.
- An unknown path cannot enter the control-plane or stand-aside lane. The
  v5 lane retains its own path rules and can admit paths that this classifier
  calls noncanonical. The retained census reported 15 such paths: 12 New
  Hampshire paths with spaces and 3 `us/statutes/42/1437c–1*` paths with an
  en dash. That historical observation does not establish a current count.
- One control-plane path puts the whole delta under the approval, including
  allowlisted files and any future stand-aside paths that accompany it.
- A delta containing only stand-aside paths would get `not_applicable`;
  the empty set admits no such delta today.

The control-plane lane requires one enrolled owner's approval over every
path of the delta. It requires no manifest, signature or dispatcher:

- The reviewer must match the enrolled id and login and have `type == "User"`.
  The reviewer cannot be the PR author or the GitHub-attributed author or
  committer of any PR commit. An unattributed commit fails; the complete
  commit list must contain the head.
- The approval must name a PR commit `C`. The evaluator computes `C`'s merge
  base with current `main` and compares that delta with the head delta. They
  must have the same paths and the same leaf identity (type, mode and object
  id), or absence, on both sides. What `C` held must equal what the head holds;
  what `C` replaced must equal what the current head replaces. A base update
  outside those paths preserves this binding. A base update that changes one
  of those old-side identities breaks it even if the head keeps the approved
  bytes. Changing, adding or dropping a delta path also breaks it.
- The lane refuses a draft PR, a blocking `CHANGES_REQUESTED` or required CI
  that is not green from this PR's own run. It rereads the PR, `main`, reviews
  and CI and refuses changes between those reads.

GitHub attribution limits the independence check. The evaluator does not
inspect commit trailers or message text. If GitHub attributes the PR and all
commits to a bot, an enrolled owner can supply the counting approval even if
that owner wrote the change. The check cannot establish who used the account
(§9).

The binding also uses the merge base at evaluation time, rather than
retaining the one from review time. Suppose a reviewer approved `old → T`
at commit `C`, then ancestor `X` lands on `main` through another PR. The merge
base can move to `X`, leaving the head delta `X → T`. The binding can still
count the approval if both sides of that smaller delta match. The earlier
`old → X` landing relies on its own PR's lane.

The leaf rule still refuses every added or modified leaf that remains at the
head without regular blob mode `100644`, in every class. It does not make
that same claim about a deleted leaf. The #2231 author reported three
executable tools (`bulk/compute_matrix.py`, `bulk/roots_for.py`,
`tools/generate_hts_rate_tables.py`) and no changes to them in the 50 landings
below. Modifying one while retaining executable mode would fail. Changing
its mode to `100644` would still need the applicable lane and review; this
proposal grants no mode-change exception.

Signed backfills remain outside the proposal's lanes:

| Shape | File pattern | Verdict with the lane on | Follow-up |
|---|---|---|---|
| Signed backfill (rulespec-us#1470) | Content plus `.axiom/legacy-replacements/<64 hex>.json`; the reported example also includes its manifest, pin, waiver and test | `fail`, `signed_backfill_has_no_lane` | Keep the combined operation refused. Design and review a separate lane; splitting the signed operation remains unverified |

With the lane off, that shape retains `no_lane_for_guarded_change`. The
proposal does not supply a lane for a new jurisdiction either: classification
as content does not satisfy the v5 lane's base-jurisdiction rule.

The #2231 author reported the following classification of 50 first-parent
landings on rulespec-us main, from #1470 to #2045, on 2026-11-13. The counts
lack a reproducer and pinned inputs here. They describe that historical
sample, not a fresh execution or a verified current workload:

| Class of the whole landing | Author-reported landings |
|---|---|
| control plane (would need one owner approval) | 47 |
| content only (v5 lane) | 2 (#2045, #2154) |
| signed backfill (would refuse by name) | 1 (#1470) |
| stand aside | 0 |
| unknown path | 0 |

Of the reported 47, 29 changed `tests/`, 17 `.axiom/toolchain.toml`, 12
`.github/**`, 9 `tools/` and 3 `bulk/`; 6 changed only `tests/`, `tools/` or
`bulk/`. The author reported no other mixed shape in that sample. Across all
50, the author counted 18 changes to `.axiom/toolchain.toml`, 9 from
each enrolled owner. These categories overlap; they are not extra landings.

If maintainers adopt this lane, its rule would add an enrolled approval for
control-plane changes. In the reported sample, that would cover 47 of 50
landings and require the other owner to review each of the 18 pin changes.
Those historical counts do not establish the future review load. The broader
promotion proposal would remove a pre-sign review from content promotions
only after all activation conditions hold; this off-by-default lane alone
changes no review requirement.

With the lane on, named control-plane paths require an enrolled owner's
approval whose binding covers both sides of the delta. That is narrower than proving coverage of
every input required validation executes or reads. Requiring the guard check
also needs the enforcing runner, trusted publisher binding, lane decisions
and every key-protection precondition in §8. Neither this classifier nor its
merge supplies those conditions or closes the account-sharing risk in §9.

The alternatives remain proposals:

| Alternative | Trade-off or missing work |
|---|---|
| (a) Limit control plane to `.github/**` and the two pins | Leaves other registers, tests and tools outside this approval rule; the reported external-input findings identify a possible two-PR path through them |
| (b) Stand aside for all infrastructure, or for `tests/`, `tools/` and `bulk/` | Avoids some new reviews but would leave the reported PR-head validation inputs without this approval rule |
| (c) Exempt pin bumps from a named trusted automation identity | Needs a trusted publisher and a rule limiting each bump's contents; the proposal implements neither and grants no exemption |
| (d) Retire the v5 CI-trust lane before change 6 | Would remove the content-lane premise of the two-PR attack; the control plane would still decide what required validation runs |
| Keep today's refusal | Continues to block maintenance shapes without a lane; maintainers must resolve those shapes before requiring the check for ordinary maintenance |
| Decide by label, PR title or a file in the PR | Lets the PR author control the classification input |

### CLI

`axiom-encode merge-guard --repository rulespec-us --pr <n>` prints one JSON
line (`schema`, `verdict`, `reason`, `detail`, `evidence`). Exit codes: 0 pass,
1 fail, 3 not_applicable, 2 usage error.

### Implementation notes

The source at `ca39ebb2` defines the behavior described here. The algorithm
and adversarial table describe `INFRASTRUCTURE_LANE_ENABLED = False`; the
proposal above describes the separate branch that requires the constant to be
exactly `True`.

- Applicability comes from paths and modes. The manifest's `schema_version`
  cannot decide it, because a PR controls that label.
- `_require_immutable_delta` refuses every changed non-regular blob, so it
  cannot classify a maintenance PR. The evaluator computes a mode-agnostic
  delta from `_leaf_entries` first and calls `_require_immutable_delta` only for
  a v6 unit.
- `_pr_identity` takes a selector that already names the head. The evaluator
  reads the head from the PR and passes a four-field target. `_pr_identity`
  refuses a fork head.
- The evaluator cannot reuse `_approved_reviews`: it requires the exact head, refuses on
  any account's `CHANGES_REQUESTED`, and binds the dispatcher by login. The
  evaluator has its own effective-review code and binds by numeric id.
- Admission requires `pr.base.sha` to equal the `main` tip. The evaluator uses
  the live `main` tip as `B` and does not rely on `pr.base.sha`.
- Admission reads files through the contents API. The evaluator reads
  `git/blobs/<id>` and recomputes the object id from the bytes.
- The evaluator also excludes the index and every commit author, binds the
  candidate and base that the manifest names, requires an enrolled dispatcher
  who is the run's `triggering_actor`, binds the CI run and the signing run to
  their repository, event and commit, and refuses duplicate JSON keys.
- The promotion lane maps a jurisdiction to `rulespec-<country>`
  (`PromotionSelector.repository`). The evaluator accepts
  `rulespec-<country>-<subdivision>` names for applicability, and a v6 unit
  there gives `fail`.
- The default `fetch` is the anonymous `github_json`. The runner
  (`merge_guard_runner.token_reader`) injects a bounded reader that sends the
  guard App's installation token to the fixed API origin, with no proxy and no
  redirect. The evaluator has no client of its own.
- Line numbers in `cli.py` shift as it changes (#2194 moved them), so this
  document names functions and constants.

### Review lifecycle

| Event | Result |
|---|---|
| Someone dismisses an approval | That review becomes `DISMISSED`; the next evaluation gives `fail` |
| `CHANGES_REQUESTED` after a pass | Same: the next evaluation gives `fail` |
| Re-approval | A new review on the new `C`; the guard evaluates it as above |
| Push that changes content | The new head has no verdict; old approvals fail step 5 |
| Base-only update by merge | New head, same blobs: the guard keeps the approval |
| Rebase | The candidate commit leaves the PR: `fail` until a new signing run |

## 4. Where the guard would run and why maintainers could trust it

Two paths exist in code. Neither is active, and Max has accepted neither.

- **Recommended path: an organization ruleset requires the guard workflow.**
  The enforcing runner of change 3b ("Enforcing runner as built" below) runs
  in the RuleSpec repository with that repository's read-only job token. No
  guard App and no private key exist. `docs/merge-guard-key-protection.md`
  gives the reasons, the residual risks and the maintainer steps. It is
  conditional on the experiment of `docs/merge-guard-ruleset-experiment.md`.
- **Fallback path: a guard App publishes the check.** The shadow runner of
  change 3 ("Runner as built" below) is its first part. Before trusting that
  check, maintainers must prevent unreviewed code from obtaining the guard
  App's private key or any token it mints. Checklist step 1 of §8 requires
  Max's choice, implementation and fresh independent evidence over the whole
  set before anyone stores the key or enables shadow execution.

The bullets below describe the fallback path unless they name the ruleset.

- **Code**: `src/axiom_encode/content_merge_guard.py` and the CLI command
  `merge-guard`, with CODEOWNERS entries like those for the admission files.
  Those entries cover the evaluator, the runner module, its workflow, `cli.py`
  and `signing_broker.py`, and leave most of what the runner job executes
  without a code owner. The job installs the whole encoder package with
  `uv sync --locked` and starts it through the `axiom-encode` console script,
  `axiom_encode.entrypoint:main`, which imports `encode_timing` and runs the
  `signing_broker` start-up before it imports `cli.py`. The job therefore
  executes `src/axiom_encode/__init__.py`, `entrypoint.py`, the modules that
  these and `cli.py` import, the builds and dependencies that `pyproject.toml`
  and `uv.lock` select, and their build and start-up hooks, on the runner that
  later holds the App's key and token. The reviewed CODEOWNERS file covers few modules in that import closure.
  Since #2277 it lists `/.github/`, so every workflow file, new ones included,
  has a code owner in the file; that entry is file ownership only, and
  `docs/merge-guard-key-protection.md` proposes, without a decision, the
  settings that would enforce it. At the 2026-11-15 protection read, which
  predates #2277, a write collaborator could land those changes without
  code-owner review; no later protection read is retained here. Proposed constraint D2a holds for the
  runner only once one reviewed mechanism covers the key-exposure set of
  checklist step 1 of §8.
- **Identities**: the existing `TRUSTED_SIGNING_REVIEWERS` on encoder main. A
  RuleSpec PR cannot change it. Under the ruleset path the guard reads the
  list from the pinned commit, because `reviewed_pr_admission.py` is a file
  of the bundle [code: `BUNDLE_MODULES`]. A change to the list on encoder
  `main`, a revocation included, reaches the guard only when the pin moves
  [inferred]. The same holds for `INFRASTRUCTURE_LANE_ENABLED`.
- **Runner boundary**: the workflow's gate job requires
  `github.ref == refs/heads/main`. The intended `merge-guard` environment
  would hold the App's private key and allow only `main`; those settings still
  require implementation after checklist step 1. The
  branch rule limits which ref can use the environment. It does not limit
  which workflow file or job on that ref selects the environment, or which code
  those jobs execute. Any job whose protections allow it to select `merge-guard` can obtain its
  secrets. This includes jobs in existing, new, replaced or modified workflows
  and jobs that select the environment through dispatch inputs or expressions.
  The boundary includes who can land those workflows, all earlier steps and
  persistence, actions, builds, installation, dependencies, interpreter and
  startup hooks, the console entry point, `__init__.py`, `signing_broker` and
  the direct and transitive import closure.
  [GitHub environment documentation](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments).
- **Planned publisher binding**: a dedicated GitHub App on the pilot
  `rulespec-us` repository, with
  `checks: write`, `pull_requests: read`, `contents: read`, `actions: read`.
  Maintainers would require the guard's check with that App's `app_id`
  through branch protection.
  Workflows in `rulespec-us` publish as app 15368 and cannot satisfy it.
  The binding keeps other publishers out. Code that obtains the App's key can request tokens for any of its
  installations, within each installation's granted repositories and
  permissions, while the key remains valid. One request's narrower token scope,
  one-hour lifetime or revocation does not constrain independently minted
  tokens. A holder of a token can use its granted permissions directly. The
  App binding therefore also needs checklist step 1 of §8.
  [GitHub installation authentication](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/authenticating-as-a-github-app-installation).
- **Planned triggers**: `schedule` (a sweep over open PRs to `main`) and
  `workflow_dispatch` (country and PR number). No trigger carries authority. An
  evaluation always reads live state, so a forged trigger yields an honest
  verdict, and a missing one leaves the required check absent, which blocks the
  merge. The verdict stays honest only after the mechanism and any complement
  prevent unreviewed code throughout the key-exposure set from obtaining the
  key or a token (checklist step 1 of §8). Unreviewed code that obtains either
  needs no forged trigger: it can publish any verdict.
- **Why not run an ordinary workflow in `rulespec-us`**: under `pull_request`
  the PR controls the workflow file. Under `pull_request_target` the base
  branch supplies the file, but its check still publishes as app 15368, which
  any PR workflow can imitate. A ruleset workflow differs: it runs in the
  target repository, but the ruleset names the source repository, the file
  and the commit [docs D1, D3]. That the run takes exactly that commit, and
  that a same-named check cannot satisfy the rule, are
  [experiment X1] and [experiment X2].
- **The organization ruleset** "Require workflows to pass before merging"
  (recommended, not accepted). What is known:
  - The organization's plan is `enterprise` [live read 2026-11-17]. The rule
    exists at the organization level, and its REST form names `repository_id`,
    `path` and an optional `sha` [docs D1, D3].
  - It starts the workflow on `pull_request` (`opened`, `synchronize`,
    `reopened`) and on `merge_group` (`checks_requested`), always, and
    ignores filters. No review event starts it [docs D1, D2]. So nothing
    re-evaluates when a review changes; see "Limits of the trigger set".
  - A public source workflow can run in any repository of the organization
    [docs D1].
  - Organization owners, and holders of a custom organization role with the
    ruleset permission, can edit it [docs D4, D6, D16]. A repository
    administrator cannot. What a `rulespec-us` administrator can still
    change (the merge queue, the default branch, self-hosted runners) is in
    §9. The token that agent sessions use on the
    maintainer's machine has no `admin:org` scope, and the API refused its
    read of the organization's rulesets [live read 2026-11-17].
  - `Evaluate` enforcement is the shadow mode [docs D4, D6].
  - Events that a job token causes do not start it [docs D2].
  - Not known: everything in "Unverified live" below.

### Runner as built (change 3, shadow only, inert)

`.github/workflows/merge-guard-runner.yml` calls `axiom-encode merge-guard-run`
(`src/axiom_encode/merge_guard_runner.py`). The workflow evaluates `vars`,
which supports repository, organization and environment variables; it checks
no repository-only origin. The `run` job selects the `merge-guard`
environment, so the `vars` context of that job can also carry an environment
variable of the same name; GitHub documents that "the variable at the lowest
level takes precedence"
([variables reference](https://docs.github.com/en/actions/reference/workflows-and-actions/variables),
read 2026-11-17). The same page adds a note under "Configuration variable
precedence" (read 2026-11-22): "Environment-level variables are only
available on the runner after the job starts executing. This means that
environment-level variables won't overwrite variables in the `env` and `vars`
contexts." So an environment variable named like the switch does not replace
a `vars` value that the organization or the repository already supplies, and
a job-level `if` is evaluated before the job starts. This design does not
rest on either reading: the gate job has no environment, and the `run` job
repeats the switch test in a step. The
retained evidence does not establish actual same-name organization or
environment variables, or how a rerun resolves a changed variable
**[check live]**.
[GitHub variables context](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#vars-context).

The Python CLI accepts `MERGE_GUARD_TOKEN` from its process environment and
treats it as opaque: it does not test a prefix, a dot count or a fixed
length. GitHub changes the format (40 characters for an earlier installation
token, about 520 for a stateless one,
[changelog of 2026-10-02](https://github.blog/changelog/2026-10-02-stateless-github-app-installation-tokens-rolled-out/),
cited by the review of head `36bab7533`; not read by the author). The CLI
tests two things only: every character is one of an HTTP `Bearer` credential
(ASCII letters, digits and `-._~+/=`, RFC 6750 `b64token`), so the value
cannot split the `Authorization` header, and the length is 20 to 16,384
characters, a transport bound and not a format rule. An earlier revision
capped the length at 512 and refused a current token before any request.
It does not check the workflow ref, switch, App identity or
credential provenance. The intended workflow supplies an App installation
token; calling the CLI elsewhere does not inherit the workflow gate.

| Item | Value |
|---|---|
| Trigger | `workflow_dispatch` only (inputs `country`, `pr`, `mode`). No `schedule`: the sweep on a timer is a later reviewed change to the workflow file |
| Switch | The gate job requires `vars.MERGE_GUARD_RUNNER_ENABLED` to equal `true`, and `github.ref == refs/heads/main`. A rerun of `run` alone can reuse an earlier successful gate, so the `run` job repeats both checks in its first step, before any step names a credential (#2257 R2-P3-1). The same step repeats the gate's checks of the `country` and `pr` inputs, because the token step builds the repository name from `inputs.country`; tests pin that neither job nor any step carries `continue-on-error`. The repeat evaluates `vars.MERGE_GUARD_RUNNER_ENABLED` in the `run` job; no retained evidence shows that a rerun resolves the variable's current value **[check live]**, so the repeat alone is not the rollback. GitHub documents that a disabled workflow cannot rerun; change 3's rollback still unsets the switch and disables the workflow. Neither check governs another workflow that selects `merge-guard` (checklist step 1 of §8). [Rerun documentation](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs) |
| Environment | The workflow names `merge-guard`; its planned deployment branch rule allows `main` only. The gate job has no environment, so a refused dispatch does not create one |
| App client ID | Intended environment variable `MERGE_GUARD_APP_CLIENT_ID`; the workflow reads the `vars` context |
| App private key | Intended environment secret `MERGE_GUARD_APP_KEY`; the workflow reads the `secrets` context |
| Token | The `run` job mints one per run, for `rulespec-<country>` only: `checks: read` in `dry-run`, `checks: write` in `shadow`; `pull_requests`, `contents`, `actions`: `read`. The CLI reads it from `MERGE_GUARD_TOKEN` and has no token argument |
| `dry-run` (default) | Prints one JSON report. Writes nothing |
| `shadow` | Also creates one check run per evaluated pull request on its head commit, named `content-bound-merge-guard (shadow)`, conclusion always `neutral`, with the pull request number, the verdict and the reason in the title (`shadow verdict for #<number>: <verdict> (<reason>)`) and the verdict in the summary. It skips the write only when the guard App's own newest shadow check for that pull request on that head (the runner matches the App by its client ID, `MERGE_GUARD_APP_CLIENT_ID`, and the pull request by the check's `external_id`) already carries the same verdict. A check with the shadow name from another publisher never counts, so a pull request workflow cannot suppress its verdict |
| Enforcing mode | None in this workflow and in `merge-guard-run`. Change 3 reserves the name `content-bound-merge-guard` and its writer never publishes it. Change 3b's enforcing runner is a separate workflow with no writer ("Enforcing runner as built" below) |

- **Never make the shadow check a required check.** `neutral` satisfies a
  required check, so requiring the shadow name would require nothing. The
  check to require later is `content-bound-merge-guard`, from change 3b of
  §8, which adds the enforcing mode. Until 3b lands on `main` no code reports
  that name, so requiring it blocks every pull request. The writer in this
  change refuses any check that is not the neutral shadow check.
- **Startup refusals.** The workflow refuses a missing switch or credential;
  the CLI refuses a missing or malformed token. Each names the missing input. A pull request list that fails, arrives malformed or
  reaches its page bound (10 pages of 100) counts as a run error, never as an
  empty sweep. A verdict whose evaluation could not read everything it needed
  is `fail` and also a run error: the report says `complete: false` and the
  command exits 1. That covers a failed request, a tree GitHub truncates, a
  review, commit or file listing at the evaluator's page bound, a blob that
  cannot be read whole, a required-check listing that does not add up, and a
  response that is not the object or list its read asked for. The evaluator
  marks such a verdict with `incomplete_read: true` in its evidence
  (`INCOMPLETE_READ_REASONS` for the reasons that always mean it, the mark
  alone for a mixed reason such as `ci_not_green`). The runner counts an
  evaluation as completed only when its verdict and reason are on a closed
  list of decisions and the mark is absent, so a reason the evaluator gains
  later is a run error until a reviewed change classifies it. An earlier
  revision treated only `github_read_failed` and `evaluation_error` as
  unread, and a truncated tree gave exit 0 with `complete: true`. Two choices
  to know: `changed_during_evaluation` stays a decision (the state moved; the
  next run judges the new state), and a pull request whose fork was deleted
  (`head.repo` is null) stays a decision.
  The run's exit code reports execution errors; the report carries verdicts.
  Response JSON nested deeper than the interpreter's recursion limit raises
  `RecursionError`, not `ValueError`; `_request` converts both to the same
  failed read, so the command prints its JSON report and exits 1
  (#2257 R2-P3-6). Fake-opener tests cover this; no live response exercised it.
  The "Evaluate and report" step pipes the command into `tee` under
  `set -euo pipefail`, so the step fails when the command fails; a test runs
  the step's script with a failing command (#2257 R2-P3-7).
- **No authority in the list.** The runner uses the open pull request list as a
  work list only. Each verdict comes from the evaluator's own live reads.
- **The write filter does not isolate the key or the token.** The runner's
  writer refuses every request except a completed neutral shadow check, and
  the token's permissions rule out merges, statuses and dispatches. The
  writer's filter binds only code that calls it. The `run` job checks out the
  dispatched `main` commit, installs it with
  `uv sync --no-config --locked --python 3.13 --extra api --no-dev`, passes
  the private key to `actions/create-github-app-token` as a step input, and
  passes the token it gets back in `MERGE_GUARD_TOKEN` to
  `.venv/bin/axiom-encode merge-guard-run`, which imports the encoder package
  before it reaches the runner. Code anywhere in that chain, including a
  dependency or a file an earlier step leaves on the runner, can read the
  token and use its permissions directly; in `shadow` mode those include
  writing any check run on `rulespec-<country>`. Code that runs or persists
  before the token step can also obtain the private key when the token action
  receives it; the key-exposure set includes that earlier code and persistence. Whoever holds the key can
  mint tokens for any of the App's installations, up to the repositories and
  permissions each installation grants, for as long as the key stays valid. Narrowing, expiring or revoking the
  workflow's requested token does not limit other tokens that key holder can
  mint. Change 3 adds no mechanism that closes the chain, and checklist step 1 of §8
  makes one a precondition.
- **Open checks.** Whether an installation token for `rulespec-<country>`
  alone can read the public signing run in `axiom-encode` (the evaluator
  reaches step 6 only for a v6 unit, and the production writer emits only v5).
  A retained 2026-11-13 observation counted about 49 open pull requests;
  each evaluation makes multiple reads, and no timed-sweep load test
  establishes a safe cadence. Size the cadence before
  anyone adds the `schedule` trigger. On 2026-11-15, check runs on
  `rulespec-us` carried `app.client_id`; a guard App check without it would
  make the runner write the verdict again on every run rather than suppress it.
  `_already_reported` rejects a listing it did not read whole. GitHub's
  [List check runs for a Git reference](https://docs.github.com/en/rest/checks/runs#list-check-runs-for-a-git-reference)
  (read 2026-11-17) documents `check_name`, `app_id`, `filter` with the
  values `latest` (the default, "returns the most recent check runs") and
  `all`, `per_page` with a maximum of 100, `page`, and a response of
  `total_count` and `check_runs`. It also limits the listing to the 1,000
  most recent check suites of a reference. It does not say how `latest`
  groups runs. The runner asks for `filter=all`: every shadow-named run of
  the head is listed, `total_count` is the number of listed runs, and the
  App's newest check is its listed run with the highest id. With `latest`
  the page arithmetic would depend on whether `total_count` counts the runs
  before or after the filter. Observed on 2026-11-17 for the check
  `validate / validate` on two commits of `rulespec-us` whose run was rerun
  (`9be9d91e5`, three attempts; `f3ac29bfe`, two): `latest` returned one run
  with `total_count` 1, `all` returned three and two runs with `total_count`
  3 and 2. Those are attempts of one Actions job. No observation covers
  several check runs that an App creates with one name on one head
  **[check live]**, so the runner does not rely on `latest` for them.
  The runner reads up to 10 pages of 100 (#2257 R2-P3-8). Past 1,000 listed
  checks, or when the pages do not add up to one stable `total_count`, it
  writes no check for that head and the run exits 1; it never treats a
  partial listing as "already reported". **Limit that `filter=all` brings:**
  every shadow-named check on a head counts toward the 1,000, the App's own
  earlier ones included (one per changed verdict, reason or detail on that
  head). Any workflow of the RuleSpec repository with `checks: write` can
  create checks with the shadow name, so more than 1,000 on one head
  withhold that head's shadow check until someone changes the bound; the
  run reports the error and exits 1. The runner holds the App's client ID,
  not its numeric `app_id`; whether the `app_id` filter would bound the
  listing to the guard App's own checks needs a live App **[check live]**.
  The identity of a shadow check (`external_id`) is
  `pr-<number>:<digest>`, where the digest covers the pull request number,
  the verdict, the reason and the detail. A verdict is for a pull request,
  not for a commit: the evaluator reads the reviews and the CI run of one
  pull request. Two pull requests can have the same head commit and
  different verdicts, and a check run attaches to the commit; an earlier
  revision left the number out, and two such pull requests replaced each
  other's check on every sweep. A check of the App for another pull request
  on the same commit is set aside when the runner looks for "its newest
  check"; a check of the App whose `external_id` has any other form stays in
  the comparison, so it can cause a new write and can never suppress one.
  The evidence is not hashed: it names the `main` tip, so hashing it
  would write a new check on every open pull request each time `main` moves.
  When only the evidence changes on one head (another approving owner, a
  rerun of CI), the runner skips the write and the newest shadow check's
  summary still shows the earlier evidence; the printed report carries the
  current evidence.
- **Known limit of the sweep, acceptable for shadow only.** The runner reads
  the list by page number. A pull request that opens or closes during the read shifts
  the pages, so one sweep can miss a pull request or see one twice (twice is
  a run error). A missed pull request has no shadow check until the next
  sweep. `complete: true` in the report therefore means "every pull request
  of the work list this run obtained was evaluated from complete reads and
  every intended check was written". It does not mean that the list held
  every open pull request; a sweep is not a census. An enforcing check would leave that pull request without its
  required check, so change 3b must read from a stable snapshot or trigger per
  head.

### Enforcing runner as built (change 3b, required nowhere)

Change 3b adds the enforcing mode as a workflow that an organization ruleset
can require. It follows option (g') of the key-mechanism research: no GitHub
App and no private key. Building it chose no mechanism and created no
setting. The checklist of §8 now gives the ruleset path as the
recommendation and the App path as the fallback; Max has accepted neither.

**What is off, and the one thing that switches it on.** Nothing requires the
workflow. It becomes a merge requirement of a RuleSpec repository only when
an organization owner creates an organization ruleset "Require workflows to
pass before merging" that names
`.github/workflows/content-bound-merge-guard.yml` of axiom-encode at a commit
SHA and targets that repository. No variable, key, environment or repository
setting switches it on. GitHub documents that organization owners, and
holders of a custom organization role with the ruleset permission, can edit
an organization ruleset [docs D4, D6, D16]; nothing here tested that. In
axiom-encode itself the workflow's triggers fire on
the encoder's own pull requests and merge groups; there the command prints
`"inert": "source_repository"`, makes no network read and exits 0 (see "The
inert path" below for its conditions).

| Item | Value |
|---|---|
| Workflow | `.github/workflows/content-bound-merge-guard.yml`, `name: content-bound-merge-guard`, one job `guard` named `content-bound-merge-guard` (the check name) |
| Triggers | `pull_request` and `merge_group`, no filter and no activity types. No `workflow_dispatch`, `workflow_call`, `schedule` or `pull_request_target` |
| Permissions | `contents: read`, `pull-requests: read`, `checks: read`, `actions: read`. The job token is read-only; the job has no secret, variable, environment or `id-token` |
| Skip paths | None: no `if`, no `continue-on-error`, no `concurrency`. A skipped job would count as passed |
| Code it runs | The stdlib guard bundle: `scripts/merge_guard_launch.py` and `merge_guard_runner.py`, `content_merge_guard.py`, `reviewed_pr_admission.py`, `applied_manifest_v6.py`, `rulespec_citation_paths.py`, `statute.py`. A sparse checkout of exactly these files at `github.workflow_sha`, copied next to an empty `axiom_encode/__init__.py`. No `uv`, `pip`, cache, console script or encoder package |
| Bundle digest | The workflow carries `BUNDLE_SHA256` as a literal and refuses a bundle with another digest. The digest is the sha256 of the lines `<sha256 of the file>  <path>` for `launch.py`, `axiom_encode/__init__.py` and the six modules in list order: the output of `sha256sum <those paths> \| sha256sum` in the bundle directory. A test recomputes it from the working tree, so a change to any bundle file fails CI until the same change updates the literal in `.github/` |
| Launcher | `python3 -I -S launch.py`. It refuses to run without `-I` and `-S`, in a directory that holds anything but the bundle files and the one directory `axiom_encode`, when a bundle file is not a regular file (a link, a FIFO), when an import asks for an `axiom_encode` module off the list, or when such a module was loaded from another directory. It appends the bundle directory to `sys.path`, after the standard library. It takes no argument. It exits 0 only when the command returns the integer 0; every other return value and everything the command raises, `SystemExit` included, is exit 1 [code: `main` of `scripts/merge_guard_launch.py`] |
| Command | `merge_guard_runner.enforce_command` (also `axiom-encode merge-guard-enforce`). Inputs come from the runner environment only: `GITHUB_REPOSITORY`, `GITHUB_EVENT_NAME`, `GITHUB_EVENT_PATH`, and the job token in `MERGE_GUARD_TOKEN`. No argument selects a pull request, a repository or a mode |
| Result | The exit code, which is the job's conclusion. 0 only when the verdict and its reason are in `SUCCESS_VERDICTS` (`pass`: `content_bound_approval`; `not_applicable`: `no_guarded_content`, `earlier_lane_v5`), the verdict is bound to the event's head, and the run has no error. The two success reasons of the proposed infrastructure lane (`control_plane_content_bound_approval`, `infrastructure_as_today`) are not in the set while that lane is off: a verdict with one of them fails the job, and the reviewed change that decides the lane adds them [code: `SUCCESS_VERDICTS`]. 1 for every `fail`, every read error, a malformed event, another event, another repository and an absent token. The runner creates no check run, so no `neutral` result exists; it sends `GET` requests only |

**The inert path, and the one assumption that would fail open.** The inert
path is the only way the job ends with success without an evaluation. The
command takes it only when `GITHUB_REPOSITORY` is
`TheAxiomFoundation/axiom-encode` and the event payload, read with the same
bound and strict parser as every event, agrees: the event is `pull_request`
or `merge_group`; `repository.full_name` is the encoder repository; and for a
`pull_request` event the pull request's own base repository
(`pull_request.base.repo.full_name`) is the encoder repository too. The
payload must also be a whole event of its kind: for `pull_request` a
`pull_request` object with a number that the payload's `number` repeats and
an exact head commit; for `merge_group` a `merge_group` object with an exact
`head_sha` and `base_sha`, a `head_ref` of the form
`refs/heads/gh-readonly-queue/<branch>/pr-<N>-<40 hex>` and a `base_ref`
`refs/heads/<branch>` for the same branch. Any
disagreement, an incomplete event, an absent or unreadable event file and a
payload that is not strict JSON are exit 1 [code: `_require_source_event`].
A `merge_group` payload has no second field that names a repository
[docs D14], so for that event the shape is all the
command can test. (The review of head `979124f10` found that a `merge_group`
event holding only `{"repository": {"full_name": "TheAxiomFoundation/axiom-encode"}}`
ended with exit 0 and `inert`; it is exit 1 now.)

Every other unverified item of this section fails closed. This one would
fail **open**: no live ruleset run has shown what `GITHUB_REPOSITORY` and the
event payload are when a ruleset runs the workflow for a RuleSpec repository
[experiment X18]. The troubleshooting page, under "Supported directory",
suggests the condition `github.repository != 'my-org/source-repo'` for a
workflow that should run only outside its source repository [docs D2; read
again for this change in the documentation source, `github/docs`]. That is
indirect support that the value is the target repository; no page states
it. If a ruleset run carried the encoder repository both in
`GITHUB_REPOSITORY` and in those payload fields, every pull request of the
RuleSpec repository would pass with no evaluation. The event check narrows
the assumption; it does not remove it. **Activation hold** (checklist steps
7, 10 and 11): before the ruleset goes `Active`, a pull request with no lane
(for example a content module with no manifest, as the open Arizona
candidates are) must show a **failed** `content-bound-merge-guard` job under
the ruleset in `Evaluate`, at the commit that the ruleset pins then. The
job's log must show a report whose `repository` is the RuleSpec repository,
whose `verdict` is an object with the reason `no_lane_for_guarded_change`,
and which has no `"inert"` key. The key `verdict` alone is no evidence: the
inert report and every error report carry `"verdict": null` [code:
`enforce_command`]. A passed job, or a report with `"inert"`, stops the
activation.
**Activation hold, one canary for each trigger:** the pull request above
is the first canary. The second is the queue: the merge group of such a
pull request must show a **failed** job of the `merge_group` event under
the same ruleset, with a report whose `repository` is the RuleSpec
repository and which has no `"inert"` key (a `verdict`, or an `error` of
the wait) [experiment X5, X15, X18; checklist steps 8 and 11]. A passed job, or a report with `"inert"`, for either trigger stops the
activation. The two are separate observations, because the `merge_group`
payload carries no second repository field and nothing but a live run
shows what it holds.

A further signal was considered and not added: a read of
`/repos/TheAxiomFoundation/axiom-encode/actions/runs/<GITHUB_RUN_ID>` with
the job token, which would succeed only if the run belongs to the encoder
repository. It would make every pull request of axiom-encode depend on a
network read for a job that judges nothing there, and whether a ruleset
run's id resolves in the target or the source repository is itself not
documented [inferred]. Step X18 of the experiment settles the question
directly.

**`pull_request` event.** The number is `pull_request.number` and must equal
the payload's `number`; the event head is `pull_request.head.sha`. The runner
calls `evaluate`. A verdict whose evidence names another head than the event
head becomes `fail` (`event_head_is_stale`): the run's check attaches to the
event head, so a verdict for a head that moved, or a re-run of the run of an
older head, never satisfies it.

**Waiting for `validate / validate`.** The guard run starts together with
the required check, and nothing starts it again when that check finishes. On
`fail` with reason `ci_not_green` the runner polls the check runs named
`validate / validate` on the event head every 180 seconds
(`CI_POLL_SECONDS`, the cadence of the merge-group wait), until at least one
exists and all are completed, for at most 18,000 seconds, five hours
(`CI_WAIT_SECONDS`), and 101 polls (`MAX_CI_POLLS`). It then evaluates once
more, and that verdict is final. The wait only delays: a settled check is not
a verdict, and a poll that cannot be read whole leaves `fail` and makes the
run an error. The wait starts only for a `ci_not_green` that is a decision
(the check is absent, pending or red). A `ci_not_green` that carries the
evaluator's incomplete-read mark (a check listing that does not add up, a
response of the wrong shape) is an error of the run with no wait and no
second evaluation, like every other read that could not be made whole: the
report says `an evaluation could not read GitHub` and the job fails. Every
other verdict is final at once. In the evaluator's step
order an unapproved v6 unit fails at step 5 (`no_approval`) before step 7
reads CI, so it does not wait; a test pins that it makes no check-run read.

**Measured duration of the required check [live read].** Read on 2026-11-17,
about 6:57pm EST, with one read-only call:
`gh api "repos/TheAxiomFoundation/rulespec-us/actions/workflows/repository-checks.yml/runs?event=pull_request&status=success&per_page=40"`
(the 40 newest successful `pull_request` runs, created 2026-11-07 to
2026-11-17 UTC; duration = `updated_at` minus `run_started_at`, so for a re-run
it is the last attempt; 2 of the 40 were re-runs). 38 runs took 172 to 227
minutes; the other two took 42 and 55. The median of the 40 is 207.5
minutes. The independent review of this change measured the same sample
about ten minutes earlier with the same result. No `merge_group` run of that
workflow exists yet (`event=merge_group` lists 0), so the group's duration
is not measured; rulespec-us#2160 reports 2.5 to 4 hours [author-reported].
The sample is ten days of one repository; a larger matrix makes the check
slower.

**Why the wait is five hours.** The wait was 60 minutes with a poll every 30
seconds until review round 2 (P2-1). With the measured duration that wait
ended before the check in 38 of 40 runs, so every head that reaches the CI
rule (every v5 lane, every approved v6 head, every control-plane head once
that lane exists) ended `fail`, `ci_not_green`, and needed a manual re-run
about two hours later. Five hours leave 73 minutes over the slowest measured
run. A check that takes longer than five hours ends the job with `fail`,
`ci_not_green`, as before; then someone re-runs the failed job after the
check is green.

**What the wait costs.**

- *Runner time.* One GitHub-hosted job stays busy for each head that reaches
  the CI rule, for as long as that head's required check runs: about 3 to 4
  hours today. `repository-checks.yml` has no `concurrency` key (read in a
  checkout of `rulespec-us`, 2026-11-17), so a push to a pull request does
  not cancel the check of the older head, and the guard job of the older
  head also waits until that check ends. GitHub documents 500 concurrent
  jobs on standard hosted runners for the Enterprise plan and 60 for Team
  ([Actions limits](https://docs.github.com/en/actions/reference/limits),
  read 2026-11-17) [docs]; the organization's plan reads `enterprise`
  (`gh api orgs/TheAxiomFoundation --jq .plan.name`, 2026-11-17) [live read].
  Each waiting guard job takes one of those slots, next to the jobs of the
  check it waits for. `rulespec-us` is public; whether the job is billed was
  not checked.
- *Requests.* A waiting pull-request job makes 20 polls an hour, one read
  each: 20 requests an hour. One hour can hold both of its evaluations (a
  wait shorter than an hour; the review of head `979124f10` measured 16 + 20
  + 25 = 61 reads in 3,420 simulated seconds on the fixture), so its
  costliest hour is at most 20 + 2 x 30 = 80. The job token's limit is "1,000
  requests per hour per repository" [docs, link below], shared by every
  workflow of the repository that uses the job token. With **N** jobs
  waiting at the same time the polls alone use 20 x N requests an hour: the
  whole limit at N = 50. Counting every job at its costliest hour (80), the
  limit is reached at N = 12. Over a whole run of 3.5 hours a job makes
  about 70 polls and two evaluations, about 130 requests, 37 an hour, so
  about 27 jobs that start evenly spread use the limit. These numbers leave
  no room for the other workflows; what they use is not measured. In the
  last 100 `pull_request` runs of the required workflow (the same read, 100
  runs) the busiest hour started 8 runs and, with 3.5 hours each, about 28
  heads can be in CI at once at that rate. So the 1,000 limit can be reached
  on a busy day.
- *At the limit the job fails closed.* A rate-limited read is a failed read.
  In an evaluation it gives `fail` (`github_read_failed`) and a run error.
  In a poll it ends the wait: the verdict stays `fail`, `ci_not_green`, the
  report names the error, and no second evaluation is made. For a merge
  group it is an error with no verdict. No path gives a pass. Someone
  re-runs the failed job in a later hour.
- *The higher limit.* GitHub documents "15,000 requests per hour per
  repository" for "requests to resources that belong to a GitHub Enterprise
  Cloud account" [docs, same page]. Whether a job token of `rulespec-us`
  gets that limit is not verified: it needs the `x-ratelimit-limit` header
  of a request made with a job token of that repository **[check live]**.
  At 15,000 the same arithmetic gives N = 300 at the costliest hour.
The poll matches check runs by name only [code: `_ci_settled`]: any completed
check named `validate / validate` on the head ends the wait early. For a
pull request that only brings the second evaluation forward, and that
evaluation verifies the check itself (GitHub Actions app, the `pull_request`
run of `repository-checks.yml` for this pull request and head, success)
[code: `_require_green_ci`], so the result of an early end is `fail`, never a
pass.

**`merge_group` event: evaluated as late as possible.** The run of a merge
group starts when the group is created. It first waits until the group's own
`validate / validate` run has completed with success, and evaluates once
after that. See §6 for the wait, its bounds and the window that remains.

**Job timeout.** `timeout-minutes` is 345. Counted at the socket timeout of
30 seconds for each request (`REQUEST_TIMEOUT_SECONDS`) and 30 requests for
an evaluation (900 seconds), the longest run of each event is as follows.
The 30 requests are a fixture figure for one unit: a v5 train of `u` units
costs about `11 + 3u` requests for each evaluation ("Request volume" below).
For such a train the longest run at the socket timeout is longer than this
table, and the job timeout then ends the job with failure (closed)
[inferred].

| Event | Evaluation | Wait | Last poll interval | Last poll | Evaluation | Total |
|---|---|---|---|---|---|---|
| `pull_request` | 900 | 18,000 | 180 | 30 (1 read) | 900 | 20,010 s, 333.5 minutes |
| `merge_group` | none before the wait | 18,000 | 180 | 90 (3 reads) | 900, then 90 for the second read of the group's check (3 reads) | 19,260 s, 321 minutes |

A wait checks its clock after a poll and then sleeps one poll interval, so
its last poll can start up to one interval after the wait's bound; review
round 2 (P3-2) found that the earlier formula left that interval out. The
tripwire test computes both totals from the runner's constants and requires
the timeout to cover them and to stay below 360 minutes, the limit of a
GitHub-hosted job [code:
`test_the_job_publishes_the_reserved_check_name_and_covers_the_wait`] ("Each job in a workflow can run for up to 6 hours of
execution time", [Actions limits](https://docs.github.com/en/actions/reference/limits),
read 2026-11-17). The timeout leaves 11.5 minutes over the longest run for
the checkout, the Python set-up and the bundle build, which the formula does
not count. The socket timeout bounds each blocking operation, not a whole
request; a response that arrives very slowly can exceed these numbers, and
then the job timeout cancels the job, which fails closed [inferred].

**Request volume (measured on fixtures; one page per list).**
`tests/test_merge_guard_runner.py` pins these numbers of GitHub requests per
evaluation:

| Evaluation | Requests |
|---|---|
| Allowlist only (`not_applicable`) | 5 |
| No lane (`fail`) | 5 |
| v5 lane, one unit (`not_applicable`) | 14 |
| v5 lane, five units | 26 (three more per unit: its manifest and two module files) |
| v6 unit, unapproved (`fail` before CI) | 15 |
| v6 unit, `pass` | 25 |
| Control-plane lane, `pass` (lane switched on in the test only) | 17 |
| Merge group of a v6 unit, `pass` | 29, after the wait. The whole run with one poll is 33: the check-run listing, the group's workflow run, the evaluation, then the listing and the workflow run once more |

An evaluation is counted as 30 requests below (the largest measured is 29).

| Event | Polls | Reads of one poll | Evaluations | Worst case | Driven by a test |
|---|---|---|---|---|---|
| `pull_request` | at most 101 (`MAX_CI_POLLS`) | 1: the listing (`CI_POLL_READS`) | 2 | 2 x 30 + 101 x 1 = 161 | a full wait, the check completed at the last poll, then a whole second evaluation: 16 + 101 + 25 = 142 with the v6 fixture, whose first evaluation stops at the CI rule after 16 reads and whose second makes 25 (the test counts each evaluation by itself; an earlier revision of this row said 12 and 29, which has the same sum and was wrong) |
| `merge_group` | at most 101 (`MAX_MERGE_GROUP_POLLS`) | at most 3 (`MERGE_GROUP_POLL_READS`) | 1, then one more read of the group's check (at most 3 reads, `GROUP_CI_RECHECK_READS`) | 3 x 101 + 30 + 3 = 336 | every poll at 3 reads, then the evaluation and the second read: 3 x 101 + 29 + 3 = 335 |

A poll of the merge-group wait is one listing read; it reads the workflow
run of each listed check only when all listed checks say `completed`. In an
honest run with one listed check that is the last poll, so an honest
merge-group run of the full five hours makes about 101 + 1 + 29 + 2 = 133
requests. Review round 2 (P3-3) found that the earlier arithmetic assumed
two reads for the costliest poll, while two listed checks (the first with a
completed run, the second with a run in progress) cost three. The run now
refuses a listing of more than two checks with the required name on the
group head (`MAX_GROUP_CHECK_RUNS`; an error of the run, no evaluation), so
three reads are the most one poll can cost, and a test drives that case for
every poll of a full wait. `MAX_ENFORCE_REQUESTS` is 350 (it was 250, which
the three-read case exceeded, failing closed): past it every read fails and
the result is a failed job. The second read of the group's check (review of
head `979124f10`) adds at most 3 requests to a merge-group run; 336 stays
under the cap, so the cap is unchanged. In any one hour a waiting run of
either event makes 20 polls: at most 80 requests for a pull request in an
hour that holds both of its evaluations (above), and at most
60 + 30 + 3 = 93 for a merge group in the hour that holds its evaluation. GitHub documents the job
token's limit as "1,000 requests per hour per repository", and 15,000 for
resources of a GitHub Enterprise Cloud account
([Rate limits for the REST API](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api),
read 2026-11-17). Consequences that no load test has measured:

- Waiting pull-request jobs share the 1,000 limit: see "What the wait
  costs" above for the number of concurrent jobs that reaches it. A
  merge-group run adds 20 to 60 requests in each hour of its wait (93 in its
  last hour); with a group size of 1 and one group at a time that is one
  such run.
- A real pull request with more than 100 files, commits or reviews needs
  further page reads; the fixtures have one page of each.
- A v5 lane of `u` units costs about `11 + 3u` requests per evaluation. With
  two whole evaluations and a full wait of 101 polls the cap of 350 admits
  about 37 units (2 x (11 + 3u) + 101 <= 350; the second read of the group's
  check does not enter this bound, it belongs to the merge-group run). A
  merge-group run of such a train costs the polls, one evaluation of about
  `15 + 3u` (the pull request's `11 + 3u` and four reads for the group) and
  the second read: with one listed check, 100 + 2 + (15 + 3u) + 2 <= 350
  admits about 77 units; when every poll costs three reads (a second check
  with the required name on the group head),
  3 x 101 + (15 + 3u) + 3 <= 350 admits about 9 (it was about 10 before the
  second read). A larger train fails closed.
  Raise the cap in a reviewed change (which changes the bundle digest)
  before a bulk landing.

**Limits of the trigger set.**

- No trigger follows a review [docs D1, D2]. A v6 unit is unapproved when
  its signed head is pushed, so its first run fails at once (`no_approval`,
  no wait) [code: `_enforce_pull_request`]; after the approval someone must
  re-run the failed job. That re-run then waits for the required check if
  it has not completed. A head that is approved before its run starts, and
  every v5 lane, needs no re-run: its one run waits for the check. A re-run
  is also needed after a wait that timed out or met a read error. Whether a
  re-run updates the ruleset's result is [experiment X3]. If it does not,
  the next options are to close and reopen the pull request, or a base-only
  update by someone other than the approver [inferred].
- A dismissal after a pass leaves the pull-request run green: this runner
  has no sweep, and no event follows a review. Only the merge-group run
  evaluates again (§6). Without a merge queue the dismissal never blocks
  the merge and nothing withdraws the green job. A merge queue on the
  RuleSpec repository is therefore a precondition of making the ruleset
  `Active` (checklist steps 8 and 11).
- The merge-group run evaluates once, as late as it can: after the group's
  own `validate / validate` run has completed with success [code:
  `_enforce_merge_group`, `_wait_for_group_ci`]. rulespec-us#2160 reports 2.5
  to 4 hours of validation for each group [author-reported]; a dismissal or
  a `CHANGES_REQUESTED` review in that time now fails the group's guard job.
  What remains is the time between the evaluation's last read of the reviews
  and GitHub's merge: seconds, if the queue merges promptly after the job
  ends [experiment X15] (§6, "The window that remains"; §9, "Dismissal just
  after a pass").
- No trigger follows the change of a draft to ready for review:
  `ready_for_review` is not among the activity types a ruleset workflow runs
  on [docs D1, D2]. The v6 lane refuses a draft [code: `pr_is_draft`], so
  the run of a draft's head fails and stays failed after the pull request is
  marked ready, until someone re-runs the job or pushes a new head
  [inferred].
- The evaluator reads the whole tree of `main` and of the head with
  `git/trees/<sha>?recursive=1` and refuses a truncated response [code:
  `_leaf_entries`]. §11 has the measurement (18,398 entries and 5,050,452
  bytes on 2026-11-17, against documented limits of 100,000 entries and
  7 MB; a second read for this change at 16:45 EST gave the same numbers,
  and an
  earlier review agent of this workstream reported 6,092,589 bytes for the
  same 18,398 entries with another client; that figure is author-reported,
  with no record of the read in this repository). That is behaviour of the evaluator from
  change 1. The new consequence under enforcement: once the tree passes the
  limit, **every** evaluation fails and no pull request of that repository
  can merge until the evaluator reads trees another way. Watch the size
  before `Active` and before each bulk landing.
- GitHub does not start ruleset workflows for events that the job token
  causes [docs D2]. The signed commit must reach the pull request by another
  identity, as today. That a head with no run stays blocked is
  [experiment X8].
- Each run judges its own head, so the page-number sweep of the shadow
  runner is not used and no work list or snapshot is needed.

**Reader fix that this change needed.** The token reader of change 3 refused
every path that contains `..`. The evaluator's second read is
`compare/<base>...<head>`, so with the real reader every evaluation failed
with `github_read_failed`; the shadow tests had replaced the reader with a
fake. The reader still refuses every `..`, a `.` segment and a
percent-encoded dot, with one exception: a path segment that is exactly
`<40 hex>...<40 hex>`. A test sends every path of every lane through it.

**Verified live, in axiom-encode only.** The pull request of this change
started the workflow on `pull_request` in axiom-encode
([run 44150203163](https://github.com/TheAxiomFoundation/axiom-encode/actions/runs/44150203163),
2026-11-17). The checkout fetched `github.workflow_sha`, which was the pull
request's merge commit; the hosted runner's `sha256sum` gave the digest of
the literal; and the launcher under Python 3.13 printed
`"inert":"source_repository"` and ended the job with success. That run
exercised no RuleSpec repository, no ruleset and no evaluation.

**Unverified live.** These need the owner's experiment
(`docs/merge-guard-ruleset-experiment.md`); nothing here was run against a
ruleset. The step that tests each point is named:

- that a ruleset runs this workflow on `pull_request` and `merge_group` of
  another repository and takes the file from the pinned SHA. The
  documentation states the events and the `sha` field [docs D1, D3]; the
  behavior is [experiment X1] and [experiment X5];
- that another workflow cannot satisfy the rule with a check of the same
  name [experiment X2]. That a check of the same name from another GitHub
  App cannot satisfy it is [inferred, untested]: the lab creates no App, so
  no step of the experiment tests it
  (`docs/merge-guard-key-protection.md` §4.4, R5). A tripwire test keeps
  every other axiom-encode workflow from using the name; it cannot see
  workflows of a RuleSpec repository;
- that a re-run of the job refreshes the rule's result, and who can re-run
  it [experiment X3] [experiment X16];
- that the job token of a RuleSpec repository can read the signing run of
  `axiom-encode` (step 6 of a v6 unit) and that `actions/checkout` can fetch
  public axiom-encode with it [experiment X4];
- the `merge_group` payload (`head_sha`, `base_sha`, `head_ref`, `base_ref`,
  and `repository.full_name`, which GitHub's webhook reference does not mark
  as required [docs D14]) and the form
  `refs/heads/gh-readonly-queue/main/pr-<N>-<40 hex>`. A payload without
  one of them fails the job [experiment X5];
- what `merge_group.base_sha` is. GitHub's API description calls it "The SHA
  of the merge group's parent commit" (schema `merge-group` of
  `github/rest-api-description`, read 2026-11-17). For a queue that rebases
  several commits, or for an entry behind another entry, the parent of the
  group head may not be the `main` tip; step 2 of §6 then always fails
  (closed) [experiment X5];
- that the group's required check appears on the group head with the name
  `validate / validate`, from the GitHub Actions app, with a `details_url` of
  a workflow run whose `event` is `merge_group`, whose `path` is
  `.github/workflows/repository-checks.yml` and whose `head_sha` is the group
  head. If one of these differs, every merge-group run fails (closed)
  [experiment X15; checklist step 8 for the real repository];
- that the merge queue merges a group only after this job has ended, and how
  long after [experiment X15];
- what `github.workflow_sha` is for a ruleset workflow [experiment X11]. The
  digest literal makes a wrong value fail closed: a checkout of other bytes
  cannot match it;
- that `GITHUB_REPOSITORY` and the event file describe the target repository
  in a ruleset run [experiment X18]. This is the one item that would fail
  open; see "The inert path" above for the narrowing and the activation
  hold;
- that a re-run of the run of an older head cannot satisfy a newer head
  [experiment X13];
- that `Evaluate` mode starts the workflow [experiment X6], that an empty
  bypass list binds an owner [experiment X9], and that a head with no run
  stays blocked [experiment X8];
- what the branch-rules endpoint returns for the rule
  [experiment X14], and what a dismissal does to a queued pull request
  [experiment X15];
- what the repository's own settings do to the rule: a changed default
  branch and a renamed `main` [experiment X21], and a pull request whose
  base is changed to `main` [experiment X24]. Whether a repository-level
  self-hosted runner with the label `ubuntu-latest` can take the job is
  [inferred, untested]: the lab registers no runner, and step X23 records
  the documentation only;
- what a pin move does to a pull request that is already green
  [experiment X19], which of two conclusions on one head counts
  [experiment X22], and whether an `Evaluate` ruleset starts the workflow
  on a merge group [experiment X20];
- the behaviour on rate limiting and the limit a job token of a RuleSpec
  repository really has (1,000 or 15,000 an hour). The experiment does not
  test these; the shadow period and the queue trial do (checklist steps 7
  and 8);
- the duration of `validate / validate` on a merge group against the 5-hour
  wait (checklist step 8). The duration on a pull request is measured
  (above): 172 to 227 minutes for 38 of 40 runs, under the 5-hour wait
  [live read 2026-11-17].

## 5. Manifest and encoder changes (planned, not in this change)

### Manifest (D1)

With the reader switch on, verifiers admit schema
`axiom-encode/applied-rulespec/v6` only for
`tool == "axiom-encode promote-reviewed-candidate"` with `backend` absent. The
field set takes `_REVIEWED_CANDIDATE_MANIFEST_FIELDS` (or its per-file waiver
variant), swaps `reviewed_rulespec_ref` for `candidate_rulespec_ref`, and adds:

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
`V6_READER_ENABLED`, which is `False`. The reviewed encoder source defines the switch as a constant. It has no environment variable, flag or repository setting, so a
RuleSpec PR or workflow cannot turn it.

Which reader does what, with the switch off and on:

| Reader | Off (today) | On |
|---|---|---|
| `_load_verified_applied_encoding_manifest_payload`, and so its callers (19 call sites in `cli.py`, and others in reviewed repair, run-log export, telemetry sync, the regeneration judge and notary administration): validation coverage (`guard-generated`), promotion, retire, path and metadata migration, import inventory, legacy replacement, reviewed repair; with and without `historical_verification` | refuses | admits an exact signed v6 |
| `_guard_base_manifest_claims` (a manifest on the immutable base) | refuses | admits an exact signed v6, as obligations only |
| Readers of a nested manifest (`_retired_model_manifest_issues`, `_migrated_model_manifest_issues`, `_legacy_replacement_manifest_issues`) and `_historical_manifest_refresh_inputs` | refuse | refuse |
| `supabase_sync.run_from_apply_manifest`, `prepare_signed_backfill.py`, the canonical-refresh selection in `targeted-signed-reencode.yml`, `_canonical_replacement_baseline` | refuse or ignore (v5 only) | refuse or ignore (v5 only) |

With the switch on, a reader counts a v6 manifest as an exact signed v6 only
with the promotion tool and no `backend` key; a valid signature over the whole
payload; the exact v6 field set; the exact
`attestation_scope` shape; `candidate.head_sha == candidate_rulespec_ref`; and
`candidate.repository` equal to the repository of the manifest's attested
`source_attestation.rulespec_root`. It applies the producer,
validation-execution, source and waiver checks of the v5 promotion contract.

- **Historical verification admits v6 when the switch is on.** A
  landed v6 manifest becomes history that every later change must verify
  (`historical_verification=True` reads the manifests of unchanged files). If
  that reader refused v6, the first v6 landing would turn later validation
  red. It admits v6 on the same terms as a v5 promotion manifest, and no
  wider.
- **The reader and the evaluator share one repository rule.**
  `is_promotion_repository` (`TheAxiomFoundation/rulespec-<country>`) defines
  it for both. The reader binds the scope's repository to the
  manifest's attested RuleSpec root, which the loader compares with the
  checkout. It cannot compare the named PR, author or dispatcher with GitHub;
  the merge guard does (§3 steps 4 and 6, case 19).
- Only a manifest with the v6 label takes these paths. A change that
  carries no manifest (workflows, pins, ledgers, tooling, ProgramSpec) reaches
  no reader of this change, with the switch on or off.

**The switch rule.** The reader may be switched on while no v6 writer exists: nothing new can land. The v6 writer and the admission without approval must stay off until the guard check is required. With the reader on, the writer on and the guard check not required, an enrolled dispatcher lands their own unreviewed candidate with no approval, and any account with write access re-lands signed v6 bytes from a closed pull request.

Change 5 must wait for the required check before it enables the writer
and admission without approval. With the reader on, `validate / validate` treats an
exact signed v6 manifest like a reviewed v5 one: it proves validation and
nothing about review. `owner_review_state` (`none` for v6, `before-signing`
for the earlier lane) exists for callers to tell the two apart (D1), and it
has no production caller yet. The proposed D2 control uses the guard for owner approval; the v6 reader
checks validation. The current source keeps the reader off and admission
checks the approval before signing.

The readers in the last row of the table do not read v6 yet: they treat a v6
manifest as absent or foreign. They matter only after a v6 module lands on
`main`, and belong to change 5. The unreceipted-retirement scan has no v6 field class, so
it scans every string of a v6 manifest.

Later changes turn the switch on (change 2b) and make the promotion writer
emit v6. A verifier from before change 2b rejects v6, so no v6 manifest can
pass `validate / validate` until `rulespec-us` bumps its encoder pin.

### Encoder (behind a switch that the activation change flips)

- Admission keeps every scope check of `verify_reviewed_pr` and drops the
  requirement for an approval.
- Admission requires an enrolled `triggering_actor.id` (D3) and records the
  numeric id. Today it binds only the login.
- The admission schema, `revalidate_admission`, the runtime attestation key and
  `cmd_promote_reviewed_candidate` change so that an auditor cannot mistake a
  scope-checked candidate for a reviewed one.
- The preflight step moves to a job that runs before the gated `encode` job.

## 6. Merge queue and `notary-guard`

`content_merge_guard.evaluate_merge_group(repository, group_head, group_base,
pr)` judges a merge group that holds exactly one pull request. No merge
queue or ruleset uses it; only the enforcing runner calls it, for a
`merge_group` event.

- **Inputs.** The event's `merge_group.head_sha` (`G`), `base_sha` (`Bq`) and
  `head_ref`. The runner requires `head_ref` to match
  `refs/heads/gh-readonly-queue/main/pr-<N>-<40 hex>` whole and `base_ref` to
  be `refs/heads/main`, and takes the pull request number `N` from it.
- **First the wait, then one evaluation.** The run starts when the queue
  creates the group, and the group's own `validate / validate` run then takes
  hours: rulespec-us#2160 reports 2.5 to 4 hours for each group and asks for
  a queue status-check timeout above 4 hours [author-reported]. A verdict
  from the start of that time would not see an approval that is dismissed,
  or a `CHANGES_REQUESTED` review, while the group validates. So the runner
  polls the check runs named `validate / validate` on the **group** head `G`
  each 180 seconds (`MERGE_GROUP_POLL_SECONDS`), for at most 18,000 seconds
  (`MERGE_GROUP_WAIT_SECONDS`, five hours) and 101 polls, and calls
  `evaluate_merge_group` once, after the wait ends well [code:
  `_wait_for_group_ci`]. The wait ends well only when at least one such
  check is listed, all listed are `completed`, and each is the group's own
  run: its app is GitHub Actions, its `head_sha` is `G`, its `details_url`
  names a workflow run of this repository, and that run has `event`
  `merge_group`, `path` `.github/workflows/repository-checks.yml`, `head_sha`
  `G` and `status` `completed` [code: `_group_ci_succeeded`]. The identity
  matters here, unlike for a pull request: any workflow of the repository
  with `checks: write` can create a completed check with that name
  [inferred], and a wait that such a check could end would put the
  evaluation hours before the merge again. A check that says `completed`
  while its workflow run does not is waited for.
- **The same successful attempt, read again after the evaluation.** The
  success of the group's check is read before the evaluation, and the
  evaluation takes time (29 reads on the fixture). The review of head
  `979124f10` showed the gap: a group whose validation was replaced by a
  failed second attempt during the evaluation ended with exit 0. The wait
  now returns the identity of what succeeded: for each listed check its
  check run id, the id of its workflow run and that run's `run_attempt`.
  After an evaluation whose verdict is not `fail`, the runner reads the
  listing and the workflow runs once more (`_require_same_group_ci`, at most
  3 reads) and ends with success only when the check is completed and
  successful and the identity is the same set as before. A check that is
  now absent, in progress, red, on another run, at another attempt (a green
  one too), under another check run id, or joined by one more check is an
  error of the run: exit 1, with the verdict kept in the report. The second
  read does not wait; the queue entry needs a new group. A check or a run
  without a numeric id, and a run without a numeric `run_attempt`, are
  refused at the first read, because they could not be compared later.
  What remains open after this read is the time from it to the merge:
  two reads and the end of the job.
- **What fails with no evaluation.** The wait ends without such a check
  (timeout, also when no check ever appears); a poll cannot be read whole or
  raises; a listed check is not the group's own run; or the check or its run
  concluded anything other than `success`. For the last case the other option
  was to evaluate anyway and leave the red group to the repository's own
  requirement of that check. Failing at once was chosen: it is simpler to
  state and to test (a red group never has a green guard job), and it does
  not depend on another setting. The report then has no verdict; the error
  names the cause.
- **The window that remains.** After the wait, the evaluation makes its reads
  (29 on the fixture) and ends with re-reads: for a v6 unit `evaluate` reads
  the pull request, the `main` tip, the reviews, the signing run and the
  required check a second time and compares (its step 8), and
  `evaluate_merge_group` then reads the group's commits and the `main` tip.
  The runner then reads the group's own check once more (above).
  The job then ends, and the queue can merge. A dismissal or a
  `CHANGES_REQUESTED` review that arrives after the last read of the reviews
  and before GitHub merges the group is not seen. That window is the rest of
  the evaluation after that read, the end of the job and the queue's own
  delay: seconds, if the queue merges promptly [inferred]. It was hours
  before this change. Nothing here measured the queue's delay
  [experiment X15].
- **Algorithm of the evaluation.**
  1. `evaluate(repository, N)` gives the verdict `V` for head `H` over the
     live `main` tip `B`. A `fail` is the group's verdict.
  2. `Bq` must equal `B`: otherwise `fail` (`merge_group_base_not_main`).
  3. The commits `G` and `H` must have the same tree id: otherwise `fail`
     (`merge_group_tree_differs`). `H` is strictly ahead of `B` (step 1 of
     `evaluate`), so a merge, squash or rebase of `H` onto `B` has the tree
     of `H`. Equal trees bind every byte of the group to the judged pull
     request.
  4. `G` must be strictly ahead of `Bq`: otherwise `fail`
     (`merge_group_not_on_base`). The specification of this change did not
     ask for this read; it costs one request.
  5. `main` must still be at `B`: otherwise `fail`
     (`changed_during_evaluation`).
  6. The verdict is `V`'s verdict and reason, with `merge_group` (head, base,
     tree) added to the evidence. The function never raises, and every read
     error gives `fail`.
- **Group size 1 is a required queue setting, not something the evaluator
  proves.** A group of two pull requests fails when the second entry is
  built on the first group's commit, not on `main` (step 2), and when the
  group commit that holds two units has another tree than the judged head
  (step 3, case 24). That binds the group's **content**. It does not
  establish the group's **membership**: the evaluator reads the tree of `G`
  and no parent of `G`. The review of head `979124f10` built a group commit
  with an earlier queued merge as a parent and the same final tree; it
  passed. So a second pull request whose changes are already contained in
  the judged head (it adds nothing to the tree) can ride in the group and
  be marked merged with no verdict of its own; no byte reaches `main` that
  the judged pull request did not carry. To keep one group equal to one
  judged pull request, the queue must have a maximum group size of 1
  (rulespec-us#2160 plans it); activation requires reading that setting
  back, and a larger size is a reason not to activate. The same holds for
  a group whose final tree is the head of its last member (a second pull
  request that already contains the first): the evaluator reads and judges
  that one pull request only, and the first is merged with the group
  whatever its own checks said. With the queue's `HEADGREEN` grouping a
  group can include an earlier entry whose required check failed
  [reported by the review of the writer's lock, #2289; the GitHub page was
  not read by the author of this text]. So the requirement is four
  settings of the `merge_queue` rule, the ones the writer's lock (#2289)
  refuses to work without: `grouping_strategy: "ALLGREEN"`,
  `max_entries_to_merge: 1`, `min_entries_to_merge: 1` and
  `max_entries_to_build: 1`. **The evaluator does not enforce them and
  cannot prove a group to be a single pull request**: the commits between
  the base and the group head depend on the merge method (one squash
  commit, the rebased commits, or the head's commits and a merge commit),
  so a count of them refuses honest groups or admits dishonest ones. The
  queue settings, read back before activation, and the lock carry this
  requirement. A test records the
  limit (`test_a_group_with_an_earlier_queued_merge_as_parent_and_one_tree_passes`).
- **The head must be up to date.** `evaluate` requires `H` to be strictly
  ahead of the live `main` tip. After another pull request merges, a queued
  pull request that was not updated fails with `base_not_current_main`.
- **No wait for the pull request's own check.** The evaluation requires
  `validate / validate` green on `H` from the pull request's own run, as for
  a pull request event; the merge-group run does not wait for that one.
- **Why the queue is a precondition.** Under the ruleset path no event
  follows a review. The merge-group run is the only evaluation between a
  green pull request run and the merge, and it reads the reviews again
  [code: `evaluate_merge_group` calls `evaluate`]. A repository with no
  merge queue has no such evaluation: a dismissed approval would not block.
  The activation checklist therefore turns the queue on before the ruleset
  becomes `Active` (steps 8 and 11 of §8).
- **What the queue does not close.** The seconds of "The window that
  remains" above. A review that changes while the group's required check
  still runs is read: the evaluation comes after that check. No second
  evaluation follows the job.
- **Queue settings this needs.** The queue's status-check timeout must be
  longer than the group's validation and this job's run that follows it;
  the job's own limits are the 5-hour wait and the 345-minute job timeout.
  The guard job now ends last of the group's required checks, so each
  landing takes the group's validation time plus up to one poll interval
  (180 seconds) and the evaluation [inferred].
- **One entry at a time.** Step 2 requires the group's base to be the live
  `main` tip. With a build concurrency above 1 the second entry is built on
  the first and fails [code: `R_MERGE_GROUP_BASE`]. rulespec-us#2160 also
  asks for a build concurrency of 1 [author-reported].
- **A group commit equal to the head.** If a queue ever used the pull
  request's own head commit as the group commit, the group's
  `validate / validate` run would attach a `merge_group` check to `H`, and
  the evaluator's CI rule would refuse it, because every such check on `H`
  must come from a `pull_request` run [inferred from `_require_green_ci`].
  The wait would refuse it first, in the other direction: the pull request
  run's check on `G` is not a `merge_group` run [code:
  `_group_ci_succeeded`]. Both fail closed. The experiment records whether
  the two commits differ [experiment X5].
- **Unverified.** The payload fields and the `head_ref` form above, what
  `base_sha` is, which workflow file version a `merge_group` run of a
  ruleset uses, and whether the result of this job satisfies the queue
  [experiment X5]; that the group's required check has the name and the run
  described above on the group head, and that the queue merges only after
  this job has ended, and how soon [experiment X15]. The 40 hex digits of
  `head_ref` are not compared with anything.
- `notary-guard` runs independently. A later integration would let this guard
  report `not_applicable` for a notary lane whose immutable base carries
  `consumer.json` and whose lineage replaces apply manifests. The current
  evaluator has no notary-specific lane; question 6 of §10 remains open.

## 7. Adversarial test plan

`tests/test_content_merge_guard.py` holds the cases as fixture-driven unit
tests over an injected `fetch`. A skipped test states its reason and does not
count as a pass. The table retains the expectations with
`INFRASTRUCTURE_LANE_ENABLED = False`; it supplies no new execution result at
`ca39ebb2`.

| # | Attack | Expected verdict | Test |
|---|---|---|---|
| 1 | Owner approves before signing (no manifest at `C`) | fail | yes |
| 2 | Owner approves at `C`, then someone changes the primary by one byte | fail | yes |
| 3 | Owner approves at `C`, then someone signs new manifest bytes | fail | yes |
| 4 | Owner approves, then someone else merges `main` with the three blobs unchanged | pass | yes |
| 5 | As 4, but the merge commit also adds `.github/workflows/x.yml` | fail (scope) | yes |
| 6 | Only the PR author approves | fail | yes |
| 7 | Only the dispatcher approves, including a rerun whose `triggering_actor` differs from `actor` | fail | yes |
| 8 | Write collaborator outside the enrolled list approves | fail | yes |
| 9 | Account reuses an enrolled login with a different numeric id | fail | yes |
| 10 | Someone dismisses the approval later | fail | yes |
| 11 | Enrolled A approves, enrolled B requests changes | fail | yes |
| 12 | Approver later submits a `COMMENTED` review | pass | yes |
| 13 | PR supplies a primary as a symlink, executable mode `100755`, or submodule | fail | yes |
| 14 | PR places the primary outside the citation's canonical path | fail | yes |
| 15 | PR adds a workflow with a job that imitates the guard check name | merge blocked: wrong `app_id` (App path); the ruleset's own run stays failed (ruleset path, [experiment X2]) | skipped: needs the App binding or a live ruleset |
| 16 | PR rewrites `repository-checks.yml` so `validate / validate` passes | fail | yes: scope for a v6 unit, no lane otherwise |
| 17 | API truncates a tree, supplies an 11th review page, redirects, or returns HTTP 403/500 | fail; no cached pass | yes, for every read of a passing evaluation |
| 18 | Someone dismisses a review between the first and second read | fail | yes |
| 19 | v6 manifest whose `candidate.pull_request` names another PR | fail | yes |
| 20 | PR supplies only a v6 manifest, or supplies two v6 manifests | fail | yes |
| 21 | Manifest `applied_files` digest differs from the head blob | fail | yes |
| 22 | CI passes `validate / validate` on an older head while `H` remains pending | fail | yes |
| 23 | PR modifies a module that already exists on `main` | fail | yes |
| 24 | Merge group includes two units and lacks an approval for one | fail for the group when the second unit changes the tree or the group is built on another base. A second pull request that adds nothing to the judged tree is not detected: membership is not verified, the queue's group size of 1 is a required setting (§6) | yes for the two refused shapes: the tree differs, or the second entry's base is not `main`; the undetected shape is recorded by a test, not guarded |
| 25 | PR that changes only allowlisted files (docs) | not_applicable | yes |
| 26 | PR supplies unparseable JSON at a manifest path in the delta | fail | yes |
| 27 | Enrolled owner authored the commits; a third account opened the PR; the owner approves | fail, also when GitHub attributes the commit to no account | yes |

Cases 28-36:

| # | Attack | Expected verdict | Test |
|---|---|---|---|
| 28 | PR changes content and rewrites a workflow without supplying a manifest | fail | yes |
| 29 | PR supplies an unknown manifest schema (`v7`, missing, not a string) | fail | yes |
| 30 | A symlink, submodule or executable replaces `.axiom`, the manifest directory, a jurisdiction or a content root | fail | yes |
| 31 | PR mixes v5 and v6 manifests, removes a manifest, or supplies a v5 manifest that omits changed content | fail | yes |
| 32 | CI passes `validate / validate` from `workflow_dispatch`, another repository or another PR | fail | yes |
| 33 | Manifest names a signing run from another repository or a commit outside encoder `main` | fail | yes |
| 34 | Manifest names a candidate commit outside the PR, or one with other content, or a base that is not an ancestor | fail | yes |
| 35 | Approver authored or committed any commit of the PR, also the manifest commit | fail | yes |
| 36 | Approval on `H`, then a new head that changes only the derived index | fail | yes |

Cases 37-42:

| # | Attack | Expected verdict | Test |
|---|---|---|---|
| 37 | PR supplies a hand-written v5 manifest with content plus workflow, CODEOWNERS, toolchain pin or ledger changes | fail (no lane) | yes |
| 38 | PR supplies a v5 label with CI absent, failing or from another run | fail | yes |
| 39 | PR edits or deletes content under a v5 manifest that someone doctored | follows CI: fail when red | yes |
| 40 | PR changes content under `manual`, `programs/`, an unknown root, depth 3 or another case; `.axiom` pins or configuration; ledgers; `.github/**`; tooling; or a new top-level entry | fail (no lane) | yes |
| 41 | PR adds or modifies a symlink, submodule or executable anywhere in the delta | fail | yes |
| 42 | A PR commit with no GitHub author or committer | fail | yes |

Cases 43-48:

| # | Attack | Expected verdict | Test |
|---|---|---|---|
| 43 | `.gitattributes` that rewrites checked-out module bytes or hides diffs | fail (no lane) | yes |
| 44 | `CLAUDE.md` or `.claude/` change that steers agent sessions | fail (no lane) | yes |
| 45 | PR lists v5 modules under an unrelated manifest position, wrong digest, manifest outside a jurisdiction or new jurisdiction | fail | yes |
| 46 | PR places an allowlisted basename at another path (`sub/README.md`) | fail (no lane) | yes |
| 47 | Signing actors, status or attempt change before the verdict, or required CI changes on either lane | fail | yes |
| 48 | v5 manifest claims a non-atomic nested root under an existing jurisdiction, in either manifest layout | fail (no lane) | yes |

Cases 49-70, the enforcing runner (`tests/test_merge_guard_runner.py`,
`tests/test_content_merge_guard.py`,
`tests/test_merge_guard_enforce_workflow.py`):

| # | Attack or fault | Expected result | Test |
|---|---|---|---|
| 49 | The pull request moves to an approved head while a run of the older head is evaluated, waits, or is re-run | job fails (`event_head_is_stale`) | yes |
| 50 | Required check pending when the run starts; it ends green, red, or never ends | `pass` only after it is green and a second full evaluation passes; `fail` otherwise | yes |
| 51 | The approval is dismissed while the run waits for the required check | job fails: the wait gives no verdict | yes |
| 52 | A read fails at any index of a passing run (pull request, v5 lane, waited run, merge group) | job fails | yes, every index |
| 53 | Event payload malformed, oversized, with a duplicate key, for another repository, or with two pull request numbers | job fails with no read | yes |
| 54 | Another event (`pull_request_target`, `push`, `workflow_dispatch`, a review event) | job fails with no read | yes |
| 55 | Merge group whose `head_ref` is not the queue branch of one pull request to `main`, or whose base is not the judged `main` tip | job fails | yes |
| 56 | A run reads without bound (many polls, many pages) | request cap; job fails | yes |
| 57 | A repository named like the source repository (other case, suffix, other owner) | not inert; job fails | yes |
| 58 | A workflow change adds `if`, `continue-on-error`, a trigger, a filter, a secret, a variable, an environment, `id-token`, a write permission, an unpinned action, another checkout or drops `-I -S` | tripwire test fails | yes, each |
| 59 | A bundle file changes without a new digest literal, or the bundle directory holds one more file, a link or another `axiom_encode` package | tripwire test fails; build step or launcher exits 1 | yes |
| 60 | A verdict with a success reason that the runner does not list (the two reasons of the switched-off infrastructure lane included), or `neutral` | job fails | yes |
| 61 | `GITHUB_REPOSITORY` names the encoder repository and the event names a RuleSpec repository (in `repository.full_name` or in the pull request's base), or the event is absent, unreadable or not strict JSON | not inert; job fails | yes, also through the launcher |
| 62 | The approval is dismissed, or changes are requested, after the merge group is created and while the group's own check runs | job fails: the evaluation runs after that check completes | yes |
| 63 | The group's required check never appears, never completes, or its listing or run cannot be read | job fails with no evaluation | yes |
| 64 | A completed check named `validate / validate` on the group head that is not the group's own `merge_group` run (other name, app, head, URL, run, event, workflow, repository), or whose run is still in progress | never ends the wait: job fails, or the wait goes on | yes, each field |
| 65 | The group's required check or its run concludes `failure`, `cancelled`, `skipped`, `neutral` | job fails with no evaluation | yes |
| 66 | The build step also writes a file, `GITHUB_PATH`, `GITHUB_ENV` or `GITHUB_OUTPUT`, relaxes `set -e`, tolerates an existing bundle directory or compares part of the digest | tripwire test fails: the script is pinned whole | yes, each |
| 67 | The workflow repeats a mapping key (`steps`, `on`, `permissions`, `jobs`, `run`, `env`) | tripwire test fails | yes |
| 68 | The command returns `None`, `False` or `True`, or raises `SystemExit(0)` | launcher exits 1 | yes |
| 69 | A bundle file is a FIFO, or the bundle directory holds one more directory | launcher exits 1 | yes |
| 70 | A later, broader `CODEOWNERS` entry with other owners matches a guard file | tripwire test fails: the last matching entry decides | yes |

Cases 15 and 16 must also run live in the shadow period (§8, checklist step
7). Cases 17 and 18 cannot be produced on a live pull request; they stay
fixture-tested (the same step names the tests).

## 8. Delivery plan

Implementation comes before activation.

| # | Repo | Change | Rollback |
|---|---|---|---|
| 1 | axiom-encode | Evaluator, `merge-guard` CLI, tests, CODEOWNERS and this document. The CLI and runner call the evaluator; this step creates no enforcing check | revert |
| 2 | axiom-encode | Verifiers can read `applied-rulespec/v6` with the attestation scope, behind `V6_READER_ENABLED = False`; no writer | revert; the production writer emits only v5 |
| 2b | axiom-encode | Set `V6_READER_ENABLED = True`. May ride with change 3. Switch rule: The reader may be switched on while no v6 writer exists: nothing new can land. The v6 writer and the admission without approval must stay off until the guard check is required. With the reader on, the writer on and the guard check not required, an enrolled dispatcher lands their own unreviewed candidate with no approval, and any account with write access re-lands signed v6 bytes from a closed pull request. | set it back; the production writer emits only v5 |
| 3 | axiom-encode | Workflow that runs the guard from `main` and publishes its check, with a dry-run mode. Shadow only and inert: manual dispatch, behind `MERGE_GUARD_RUNNER_ENABLED` (§4, "Runner as built"). It does not flip `V6_READER_ENABLED`; change 2b stays a separate one-line commit to `applied_manifest_v6.py` | unset the variable; disable the workflow |
| 3b | axiom-encode | **Enforcing runner (built, required nowhere).** `merge_guard_runner.enforce` judges the pull request or merge group of one Actions event and ends its job, named `content-bound-merge-guard`, with success for `pass` and `not_applicable` and with failure for `fail` and for every error. Its workflow has the `pull_request` and `merge_group` trigger, so a ruleset that requires it runs it on every pull request head and every merge group; each run judges its own head, so no work list or snapshot is read. A merge-group run evaluates after the group's own required check has completed (§6). A pull-request run whose verdict is `ci_not_green` waits up to five hours for the required check and evaluates once more (§4). No trigger follows a review change: after an approval that arrives when the job has already ended, someone re-runs the job [experiment X3]. Request volume is measured (§4, "Enforcing runner as built"). It is switched on only by an organization ruleset "Require workflows to pass before merging" that names the workflow at a commit; no variable, key or repository setting does it. Hard precondition of checklist steps 6 and 11 | an organization owner sets the ruleset to `Evaluate` or `Disabled`. Once the v6 writer is on (checklist step 13), only in the rollback order at the top of the checklist: writer off first. That order is then a last resort and not a durable rollback; the recommended response is to repair forward or to keep the ruleset `Active` |
| 4 | rulespec-us | Pin the encoder that verifies v6. Not started: the commit to pin exists only after change 2b merges (checklist step 9). Heavy: the pin was 1,270 commits behind encoder `main` on 2026-11-17 [author-reported, activation preparation 2026-11-17]. In today's pinned mode a plain pin move fails CI, and in verifier-main mode (draft rulespec-us#2059) no reader pin exists: checklist step 9, "Change 4 has two modes" [author-reported, pin trial 2026-11-18, rulespec-us#2304; interim] | revert the pin |
| 5 | axiom-encode | Emit v6 and admit without approval, behind a default-off switch. Built as drafts #2238 and #2269 (5b, the hosted evidence step); their locks 2 and 3 expect a guard App check in classic branch protection [author-reported] | revert |
| 5c | axiom-encode | **Writer lock for the ruleset form — built as draft #2289, locked.** Locks 2 and 3 of change 5 can expect the organization ruleset's workflow entry (repository id, path, a set of pinned SHAs) and a `merge_queue` rule in the branch-rules response, in place of a guard App id. Both forms are committed unrecorded (`GUARD_APP_ID = None`, `GUARD_WORKFLOW = None`), so the lock refuses before any read. Its fixture of the response is synthetic: the real shape is unknown [experiment X14], and the recorded response replaces the fixture in checklist step 4. This is the one place that describes 5c; the checklist refers to it. Needed only if Max accepts the ruleset path [author-reported, draft #2289] | revert; the locks stay shut |
| 6 | axiom-encode | **Activation**: record the expected guard identity in the writer's locks, turn the writer switch on, update `docs/reviewed-pr-promotion.md`. One reviewed change. Not opened as a pull request before checklist step 13, because merging it takes effect. From then on each pin move of the ruleset also needs the lock constant updated (checklist step 13) | flip the switch back, in a reviewed change; the ruleset stays `Active`. This comes before any rollback of the ruleset or the queue |

### Activation checklist

This is the complete ordered list from the state of this change to
"continuous landing is on". It follows the recommendation of
`docs/merge-guard-key-protection.md`. **Max has not accepted that
recommendation; until step 0 is done, every later step waits.** If Max
rejects it, or the experiment of step 3 does not pass (step 3 defines
"passed"), steps 5-8, 10, 11 and 13 change to the fallback of that document,
and this checklist needs a reviewed rewrite first.

**Four rules that no reordering of this list may break** (review of head
`979124f10` of change 3b; they replace the earlier list, which required a
status check of a guard App and removed the pre-sign approval before the
queue was verified):

1. **The gate is the pinned workflow, not a status check by name.** Only
   the organization ruleset that names the workflow at a commit SHA
   (step 6, `Active` in step 11) is the requirement. Nothing in this list
   requires a status check named `content-bound-merge-guard`.
2. **No removal of the pre-sign approval before the queue is verified.**
   Change 6 (step 13) comes after the ruleset is `Active` (step 11), after
   the merge queue is required on `main` with a group size of 1 (step 8),
   and after a live run has shown the guard's `merge_group` job rejecting a
   dismissed approval (experiment X15 (a) and (b); a hard precondition of
   step 13). A passed pull-request job stays green after a dismissal; only
   the merge-group run looks again.
3. **A live test that a same-named check cannot satisfy the rule**
   (experiment X2, all five parts; a check that another App publishes is
   not covered by the lab, R5).
4. **One failed canary for each trigger** before `Active`: a pull request
   with no lane and its merge group (§4, "The inert path").

> **ROLLBACK ORDER. This rule binds every rollback below once step 13 has
> happened.** Step 13 turns the v6 writer on. From then on the switch rule
> of step 2 forbids the state "writer on, guard not required". So before
> anyone sets the ruleset to `Evaluate` or `Disabled`, moves its pin to a
> commit that the writer's lock does not record, or turns the merge queue of
> `rulespec-us` off, do these in this order:
>
> 1. First set `V6_WRITER_ENABLED` back to `False`, in a reviewed change on
>    axiom-encode `main` (the rollback of step 13).
> 2. Then make every signing run and every signed v6 manifest harmless, in
>    this order:
>    - (i) **No signing run is alive.** Confirm that no run of
>      `targeted-signed-reencode.yml` is queued, waiting for the
>      `production-signing` approval or in progress. Read-only, as anyone:
>      `gh run list --repo TheAxiomFoundation/axiom-encode --workflow targeted-signed-reencode.yml --limit 200 --json databaseId,status,headSha,createdAt --jq '[.[] | select(.status != "completed")]'`
>      must print `[]` (it printed `[]` on 2026-11-18
>      [live read 2026-11-18]). The listing must reach back more than 30
>      days: raise `--limit` if the oldest run that the same command prints
>      without the `--jq` filter is younger than that (on 2026-11-18 the
>      200th run was 50 days old [live read 2026-11-18]). The writer switch
>      is a constant in the
>      source, and a signing run executes the commit it was dispatched at;
>      so a run that started before the writer-off commit still has the
>      writer on [code: `test "$(git rev-parse HEAD)" = "$GITHUB_SHA"` in
>      `targeted-signed-reencode.yml`; the constant: author-reported, draft
>      #2238]. If the list is not empty, a maintainer lets each run end or
>      cancels it, nobody approves it, and the read is repeated.
>    - (ii) **No older signing run is started again.** A re-run of an older
>      signing run executes its original commit, where the writer is still
>      on, and GitHub allows a re-run for 30 days [docs D9]. So no enrolled
>      owner may re-run a signing run that was dispatched before the
>      writer-off commit, or approve one for `production-signing`. This is
>      a rule for people; nothing technical enforces it. The reader pin
>      revert of (iii) makes validation refuse what such a run signs, but
>      only while the pin stays reverted, and after point 3 nothing
>      technical keeps it reverted ("After point 3" below).
>    - (iii) **No signed v6 manifest can land: the reader is off.** Only
>      then, and immediately before point 3: revert the reader pin of
>      step 9 in `rulespec-us`, so that validation refuses every v6
>      manifest again in the verifier mode that `rulespec-us` uses (the
>      pinned encoder, or the producer floor). **This is required whenever
>      a signed v6 manifest is outstanding or its disposition is unknown.**
>      Closing pull requests is not a substitute (review of head
>      `dd8bf4fa2`): a closed pull request leaves the signed bytes on its
>      branch, any account with write access can open them again, and the
>      enabled reader admits a valid signature with `owner_review: none`
>      [code: `tests/test_applied_manifest_v6.py`]. The pin revert may be
>      left out only with a complete inventory: every run of
>      `targeted-signed-reencode.yml` since step 13 is listed (the command
>      of (i) without its `--jq` filter, reaching back to the merge of
>      change 6), and for each run its manifest is on `main` or was never
>      pushed to `rulespec-us` (no branch and no pull request, open or
>      closed, holds it:
>      `gh pr list --repo TheAxiomFoundation/rulespec-us --state all --search "<run id>" --json number,state,headRefName`
>      and a search of the branches). One run that cannot be accounted
>      for makes the disposition unknown, and then the pin is reverted.
> 3. Only then weaken the ruleset or the queue.
>
> **Between the check of point 2 and the change of point 3 GitHub offers
> no transaction.** A signing run can be re-run, and a signed manifest can
> be pushed, after the read and before the ruleset changes. The reader pin
> revert narrows that gap: while the pin is reverted, validation refuses a
> v6 manifest. The form "no open pull request holds a signed v6 manifest"
> leaves that gap open and depends on the rule of (ii).
>
> **After point 3 nothing technical protects `rulespec-us` from signed v6
> bytes. This order is not a durable rollback.** The reader pin is content
> of a pull request: `repository-checks.yml` of `rulespec-us` reads
> `.axiom/workflow-toolchain.toml` from the checkout of the pull request
> itself [code: job `workflow-toolchain`, `rulespec-us` at `ddad6d63`], and
> `main` of `rulespec-us` requires 0 approvals and no code-owner review
> [live read 2026-11-18]. So any account with write access can move the
> reader pin forward again in the same pull request that lands signed v6
> bytes [inferred]. Not tested: whether validation refuses a pull request
> that moves the pin and adds a manifest together.
>
> **"Revert the reader pin" is the pinned-mode form.** If `rulespec-us`
> verifies with encoder `main` (verifier-main mode, checklist step 9), no
> reader pin exists, and the same act is a reviewed change on encoder
> `main` that sets `V6_READER_ENABLED` back to `False`
> [author-reported, pin trial 2026-11-18, rulespec-us#2304; interim].
>
> **So the recommended response to a guard problem after step 13 is not to
> weaken the ruleset.** In this order of preference:
>
> - (a) **Repair forward.** Move the pin to a fixed guard while the ruleset
>   stays `Active` ("If the guard itself is broken" below).
> - (b) **Take the guard out of use, closed.** Do point 1 and keep the
>   ruleset `Active`. A guard that cannot pass then blocks every merge of
>   `rulespec-us`. That is the fail-closed state, and it holds until the
>   repair.
> - (c) **Last resort: weaken the ruleset or the queue, in the order of
>   points 1 to 3.** It reopens the exposure that existed before the guard,
>   now for signed v6 bytes. While the ruleset is weak, three standing
>   conditions hold. Each is a rule for people:
>   - No pull request that pins `rulespec-us` to an encoder commit with the
>     reader on merges while the ruleset is weak.
>   - For 30 days after the writer-off commit, no enrolled owner re-runs or
>     approves a signing run that was dispatched before that commit; or a
>     maintainer deletes those runs, which removes the re-run [inferred;
>     docs D25 for the deletion]. Read-only listing of those runs, as
>     anyone:
>     `gh run list --repo TheAxiomFoundation/axiom-encode --workflow targeted-signed-reencode.yml --limit 200 --json databaseId,createdAt,headSha,conclusion --jq '[.[] | select(.createdAt < "<time of the writer-off commit, UTC>")]'`.
>     Deleting a run is a maintainer's act; no agent does it. A deleted
>     run loses its log, which is the record of who dispatched and who
>     approved it [inferred]: save the output of this listing and the log
>     of each run before the deletion. The page read does not say who may
>     delete a run [docs D25].
>   - Signed v6 bytes that never landed stay valid indefinitely: nothing in
>     this design expires a signature [inferred]. A closed pull request or
>     a branch can hold them.
>
> **Durable options, not designed:** enforced code-owner review of
> `.axiom/workflow-toolchain.toml` on `rulespec-us`, switched on before
> point 3 (its `CODEOWNERS` line exists; `require_code_owner_reviews` is
> off [code; live read 2026-11-18]); or a new signing key, after which
> validation refuses every older signature [inferred].
>
> The lock of the writer stops new signing runs only, and only at commits
> where the writer is off. It does not stop an open pull request that
> already holds a signed v6 manifest: with the guard not required, any
> account with write access merges it. Closing a pull request does not
> remove its signed bytes: the switch rule names signed v6 bytes "from a
> closed pull request", and a branch can hold them too. So (iii) reverts
> the reader pin in every case except the one its inventory proves: every
> signed v6 manifest that exists has landed or was never pushed.
> That has a cost once a v6 manifest is on `main`: validation of `main`
> then refuses it, and no pull request of `rulespec-us` gets a green
> `validate / validate` until the pin moves forward again (closed, and
> heavy) [inferred]. The revert is itself a `rulespec-us` pull request: it
> lands through the `Active` guard and the queue, with the approval of the
> infrastructure lane, in about 5.4 to 7.8 hours ("Throughput" below)
> [inferred]. No shortcut exists for an emergency: a ruleset or queue that
> is weakened first is exactly the state that the switch rule describes.
> **If the guard itself is broken**, so that the revert cannot pass it:
> do point 1, then repair the guard forward. A reviewed fix lands on
> axiom-encode `main` (the ruleset does not guard that repository), the
> review of step 5 reads it, and an organization owner moves the ruleset's
> pin to the fixed commit, with the ruleset still `Active`. With the writer
> off, no lock constant has to hold that pin. The revert then lands through
> the repaired guard. Do not disable the ruleset to land the revert
> [inferred]. **Before step 13 the ruleset rollback is free**: no v6 writer
> exists, so nothing new can land without the guard.
>
> Read-only check for the second form of (iii), as anyone. First the open
> pull requests that change a manifest:
> `gh pr list --repo TheAxiomFoundation/rulespec-us --state open --limit 200 --json number,changedFiles,files --jq '.[] | select(.changedFiles > 100 or any(.files[]; .path | test("(^|/)\\.axiom/encoding-manifests/"))) | {number, changedFiles}'`
> The listing gives at most 100 files for each pull request, so the
> command also prints every pull request with more than 100 changed files;
> for each of those read every file name:
> `gh api --paginate repos/TheAxiomFoundation/rulespec-us/pulls/<n>/files --jq '.[].filename'`,
> which returns at most 3,000 files [docs: REST "List pull requests
> files"]. On 2026-11-18 the command printed 9 pull requests, 4 of them
> with more than 100 changed files and one with 4,702
> [live read 2026-11-18]; a later read on the same day printed 5 above 100
> files, so the count moves [author-reported, review round 4]. For a pull
> request above 3,000 files this check is not complete, and only the reader
> pin revert applies. None can hold a
> v6 manifest yet, because no v6 writer is switched on [inferred]. Then,
> for each listed manifest file, its `schema_version` must not be
> `axiom-encode/applied-rulespec/v6`. The listing reads at most 200 pull
> requests; with more open, page it.

Who does what:

- **AGENT**: an agent session. It prepares pull requests and reads. It does
  not merge, mark ready, approve, change settings, dispatch signing runs or
  hold an `admin:org` token.
- **M**: a maintainer step. The step names which maintainer: Max
  (@MaxGhenis), any organization owner, an enrolled dispatcher, or the other
  enrolled owner (the enrolled identity that is neither the candidate's
  author nor its dispatcher).

Each step gives the action, the expected result, a read-only check and the
rollback.

0. **M** (Max) Decide:
   - the mechanism of step 1: accept or reject;
   - question 0 of §10 (the lane for infrastructure pull requests). Steps 10
     and 11 cannot happen without it;
   - D1, D2a-D2f, D3 and Seq. of
     `docs/reviewed-pr-promotion-order-proposal.md`, which are still
     agent-proposed, with the author-dispatch policy of D3;
   - question 14 of §10 (`run_attempt` in the v6 attestation scope).
     **Its place in the order.** If Max accepts it, its change lands before
     the reviewed pin of step 5 and before the reader pin of step 9: one
     reviewed pull request adds the field to the writer (change 5), to the
     reader's and the evaluator's parse of the v6 shape
     (`applied_manifest_v6.py` and `content_merge_guard.py`, both bundle
     files), and sets the new `BUNDLE_SHA256`. The guard at the ruleset's
     pin parses the v6 shape exactly (§5), and the encoder that step 9 pins
     verifies it. A field that arrives after those two pins makes every v6
     unit fail at step 14 (closed) until both pins move again [inferred].
     If Max declines it, nothing changes: the v6 shape stays as §5 gives
     it, and no step waits for it.
   - **How the decision is recorded.** A pull request comment from Max's
     account is **not sufficient evidence** of his decision: this project's
     own record holds answers that an agent posted under that account (the
     paragraph on `docs/reviewed-pr-promotion-order-proposal.md` at the top
     of this document). A comment may state the decision; it does not prove
     who made it.
   - **What counts as acceptance of the mechanism.** An organization owner
     creates the `Evaluate` ruleset of step 6. Only a session with an
     organization owner's rights can do that [docs D4, D6], and the token
     that agent sessions use on the maintainer's machine has no `admin:org`
     scope [live read 2026-11-17]. **The ruleset, read back, is not the
     evidence of an owner's decision** (review of head `dd8bf4fa2`). The
     readback shows configuration and no actor, and owners are not the
     only ones who can write it: a holder of a custom role with "Manage
     organization ref update rules and rulesets" can create and edit an
     organization ruleset (`docs/merge-guard-key-protection.md`, P19).
     So acceptance has two records, both copied into §11 by a maintainer:
     (1) **the decision**, recorded independently of the ruleset: Max
     states it himself, outside any agent session, to the second
     maintainer, who records the date, the channel and the wording; (2)
     **actor evidence for each privileged change** (the creation of
     step 6, the pin move of step 10 and the switch to `Active` of
     step 11): the entry of the organization audit log, or of the
     ruleset's own history page, for that change, with its actor, its
     time and, where the log shows it, whether a token made it
     [not read in the documentation by the author: the owner takes the
     event's name from the log itself]. A change whose actor is not an
     organization owner, or was made with a token that an agent session
     can hold, is not an owner's act for this checklist. **This checklist cannot verify who was at the keyboard**:
     a session that holds an owner's token with `admin:org` can perform
     each of these acts (§9, the organization-owner row and "Agent sessions
     can act under both enrolled identities").
   - D1-D3 and question 14 have no owner-only act. Their record is the
     merge of the reviewed change that implements each (changes 5 and 6;
     for question 14 the change named above, merged before step 5), with
     the same limit.
   - Expected: nothing takes effect at this step. Check: none that proves
     the decision; see above. Rollback: none needed.

1. **M** Hard precondition, before anyone stores the App key or sets
   `MERGE_GUARD_RUNNER_ENABLED`, shadow mode included: Max selects one
   mechanism that covers the key-exposure set below, an independent review
   checks the mechanism against that whole set, and the change that adopts it
   records fresh evidence of closure over that whole set. No option or
   complement has approval here.
   - The mechanism protects the guard App's private key as well as each token
     the key mints. A minted token lasts one hour and carries only the
     repository and permissions its job asks for, and those limits bind that
     token alone. Code that copies the key can mint tokens for any of the
     App's installations, up to the repositories and permissions each
     installation grants, for as long as the key stays valid. One approved run
     of compromised code can leak the key, and approving each later run does
     not undo the leak.
   - The key-exposure set starts with every job, in any workflow, that can
     select the `merge-guard` environment: the `run` job of
     `merge-guard-runner.yml`, and any job in a new, replaced or modified
     workflow that names the environment, takes it from a dispatch input or
     computes it in an expression. It includes the ability to land such a
     workflow on `main`. The `main`-only deployment branch rule limits which
     ref can use the environment, not which workflow file or job selects it.
   - The set also holds everything each such job executes or leaves on its
     runner: the workflow steps and the actions they call;
     `src/axiom_encode/entrypoint.py`, `src/axiom_encode/__init__.py`, the
     `signing_broker` start-up and every module the console script and the
     CLI import, directly or transitively; `pyproject.toml`, `uv.lock` and the
     build configuration; the dependencies with their build, install and
     interpreter start-up hooks; and every step before the token step, with
     anything it persists on the same runner. It covers the chain such a job
     runs or inherits on its runner, not every file on `main`.

   The candidates. None of them is chosen, and Max has not accepted the
   recommendation:
   - (a) enforced code-owner review of the whole key-exposure set, including
     review of any change to the set itself;
   - (b) a separately reviewed, pinned and immutable guard build, which the job
     runs without installing or starting unreviewed code;
   - (c) required human reviewers on the `merge-guard` environment, who approve
     each run that releases the key, with no agent session approving for them;
   - (g') no guard App and no private key: an organization ruleset requires
     the guard workflow of change 3b at a reviewed, pinned commit, and the
     workflow runs the build of (b) with the RuleSpec repository's
     read-only job token.

   (a) and (b) with a stored key each need a separate control that prevents
   other workflows from obtaining the key by selecting `merge-guard`,
   including existing workflows, dynamic selectors and any new, replaced or
   changed workflow someone can land. A
   pinned build cannot stop a second workflow from selecting the same
   environment, and review of existing files does not reach a new workflow
   file. Max chooses the complement along with the mechanism.

   **The recommendation is (g')** (`docs/merge-guard-key-protection.md`).
   It creates no key, so it needs no complement, and this hold is kept by
   never creating a guard App key. Max has not accepted it. No option or
   complement has approval here. The experiment is a precondition: until an
   organization owner has run `docs/merge-guard-ruleset-experiment.md` and
   its results are recorded in §11, nothing is chosen for activation. If the
   experiment fails, the named fallback is (b)+(d): the guard App's key
   lives in a cloud key service as a non-exportable signing key that only
   the guard workflow's OIDC identity may use. If no cloud account is
   acceptable, the second fallback is (b) with the complement of that
   document, where one approved but wrong change can leak the key for good.
   Under either fallback everything below applies before anyone creates the
   App.

   The `main`-only branch rule alone does not meet this precondition.
   Recording the precondition in this design does not meet it either. After
   Max chooses, maintainers implement the mechanism and any complement, and a
   fresh independent review verifies closure over the whole key-exposure set,
   maintainers may create the guard App, install it on `rulespec-us`, and store its key
   in an axiom-encode environment whose deployment branch rule allows only
   `main`: environment `merge-guard`, variable `MERGE_GUARD_APP_CLIENT_ID`,
   secret `MERGE_GUARD_APP_KEY`. Only after those conditions hold may a maintainer set
   `MERGE_GUARD_RUNNER_ENABLED` to `true`. The workflow resolves the `vars`
   context, not a repository-only origin, and the `run` job repeats the ref
   and switch checks; the actual same-name configuration and rerun resolution
   still need verification.

   Independent of the guard: do the "Now" steps A, B, D and E of
   `docs/merge-guard-key-protection.md` §5. Steps A, B and D protect two
   keys that exist today. On 2026-11-17 all three were open
   [live read 2026-11-17]. Step E removes a stored token of `rulespec-us`
   that every write collaborator can read through a workflow of a branch
   (§9); it was open on 2026-11-18 [live read 2026-11-18]. Check: the
   commands of that section. Rollback: none needed.
2. **M** Merge the changes in this order. Maintainers merge; agents may not.
   Each pull request first needs its review cycle, green CI and a maintainer
   to mark it ready. The experiment of step 3 does not wait for these
   merges (step 3).
   - axiom-encode #2237 (change 3), then #2286 (change 3b), then the pull
     request that carries this recommendation.
   - axiom-encode #2238 (change 5), then #2269 (5b), then #2289 (5c, the
     lock for the ruleset form: built, locked, with a synthetic fixture).
     These three can merge at any time: all their locks are off, and 5c adds
     a second locked form (`GUARD_WORKFLOW = None`) [author-reported, draft
     #2289]. What 5c still needs before step 13 is in step 4. The two stacks
     edit this document, `.github/CODEOWNERS` and one test list, and both
     edit `cli.py`; the second stack to merge needs `main` merged into it
     first [author-reported, activation preparation 2026-11-17].
     **The digest literal.** `BUNDLE_SHA256` is the digest of the eight
     bundle files of the merge result, so every pull request that changes a
     bundle file sets a new literal after `main` is merged into it. #2238
     changes `applied_manifest_v6.py` and `reviewed_pr_admission.py`; #2289
     changes `reviewed_pr_admission.py` again; #2269 changes no bundle file
     [author-reported, review round 3]. If the three merge one by one after
     the first stack, #2238 and #2289 each need their own new literal, in a
     reviewed commit, before they merge; #2269 needs none. A wrong literal
     fails a test and fails the guard job (closed) [code:
     `test_the_digest_literal_is_the_digest_of_the_working_tree_bundle`].
   - TheAxiomFoundation/.github#155, then a pin bump of the shared
     validation workflow in `rulespec-us` (no pull request exists for it;
     AGENT prepares it after .github#155 merges, **M** merges), then
     rulespec-us#2160 (the `merge_group` trigger of `validate / validate`).
     On 2026-11-17, 7:04pm EST, both were open drafts; .github#155 was
     `clean` and rulespec-us#2160 was `behind` its base, so it needs a base
     update and a new CI run first [live read 2026-11-17]. rulespec-us#2160
     changes a file under `.github/`: merge it before step 11, or it needs
     the lane of question 0.
   - Switch rule: The reader may be switched on while no v6 writer exists: nothing new can land. The v6 writer and the admission without approval must stay off until the guard check is required. With the reader on, the writer on and the guard check not required, an enrolled dispatcher lands their own unreviewed candidate with no approval, and any account with write access re-lands signed v6 bytes from a closed pull request.
   - Expected: no verdict and no admission changes, because every switch is
     off. Check, for each constant, on `main`:
     `git grep -n -E '^(V6_READER_ENABLED|V6_WRITER_ENABLED|INFRASTRUCTURE_LANE_ENABLED|GUARD_APP_ID|GUARD_WORKFLOW)[: ]' origin/main -- src/axiom_encode`
     prints one line for each constant that is on `main`.
     `V6_READER_ENABLED` and `INFRASTRUCTURE_LANE_ENABLED` are there today,
     both `False`. `V6_WRITER_ENABLED` (`False`) and `GUARD_APP_ID` (`None`)
     arrive with #2238. `GUARD_WORKFLOW` (`None`) arrives with #2289: the
     constant exists only on that draft's branch today, and it is the
     identity that step 13 records [code at this change for the first two;
     author-reported, drafts #2238 and #2289, for the others].
     After all six merges it prints five lines, with `False`, `False`, `False`, `None` and `None`.
     At no time may a line show `True` or a recorded identity. Rollback:
     revert the merge.
3. **M** (any organization owner) Run the experiment of
   `docs/merge-guard-ruleset-experiment.md`. It needs two throwaway
   repositories and one organization ruleset, and no stored secret, App or
   key. An agent must not run it.
   - **The experiment does not depend on any merge of step 2.** It uses
     only the two throwaway repositories, its own lab workflow files and
     the one lab ruleset; it reads no file of this stack. So an
     organization owner can run it at any time from now, before the
     merges. A failed blocking step is then known before review effort is
     spent on the stack. The one order rule: the experiment must have
     passed before step 5.
   - Expected: every step X1-X24 has a recorded result. AGENT then prepares
     a pull request that copies the record into §11.
   - Check: the record table of that document, filled in. Rollback: its
     clean-up section.
   - **"The experiment passed" means exactly this:** every *blocking* step
     has the result `pass`; every step of the class *adjusting, run again*
     has either `pass`, or its named change merged, reviewed and the step
     run again with `pass`; and every step of the class *adjusting, no
     second run* has either `pass`, or its recorded result together with
     its named change reviewed and merged. An *informational* step needs a
     recorded result only. A step that was not run counts as failed for
     this rule. Every step and every sub-part has one class; the runbook
     names the same class at each step.
   - **Why two adjusting classes.** For a step of the class *adjusting, no
     second run* the named change is a procedure or a decision, not a
     change of what the lab runs: the same step run again gives the same
     result. The rule therefore asks for the change, not for a second run.
     For a step of the class *adjusting, run again* the reviewed change
     also changes the lab file, the lab ruleset or the pass condition of
     the step in the runbook, and the second run must meet it.

     | Class | Steps | A failed step means |
     |---|---|---|
     | Blocking | X1; X2 (a) to (e); X8 (a) and (b); X24; X15 (a) and (b); X15 (c), the part "the pull request merges only after the guard job has ended"; X5, the parts "a `merge_group` run starts" and "the `merge_group` run prints the marker of the pinned commit" | The ruleset mechanism does not hold. Stop. Max decides on the fallback (b)+(d). No change to the runner repairs these |
     | Adjusting, run again | X18 (change: the runner loses its inert path, or reads the value that names the target); X11 (change: the workflow takes the bundle commit from the value that names the source commit); X5, the payload fields, and X15 (c), the identity of the group's check (change: the runner reads the real fields); X12 (change: the head binding of the runner and case 49); X14 (change: lock 3 of draft #2289 reads the real response shape; step 13 waits for it); X10 (change: `"protected": true` in the ruleset); X21 (change: the ruleset target, or a standing check of the default branch); the set-up record "the API accepts `sha` with no `ref`" (change, in three places together: the request body of `docs/merge-guard-key-protection.md` §5, its pin test, and lock 3 of draft #2289, which refuses an entry with `ref`) | The mechanism can hold, after the named change. Do the change through a review cycle, run the step again, and record both results |
     | Adjusting, no second run | X3 (change: the procedure after an approval becomes "close and reopen" or "update the branch", written into step 14 and into the table of human steps); X6 (change: the shadow period of steps 6 and 7 takes another form, named in the runbook); X20 (change: the queue trial of step 8 moves to the first `Active` landing, step 12. The same reviewed change replaces the trial clause of the step 11 precondition, and moves the stop rule for the inert path on a merge group from step 8 to step 12, before any second landing); X4 (change: none in the runner; a decision of Max: the fallback, which ends the ruleset path, or a read-only token, which is a stored secret. No such token is designed. It would have to meet the stored-credential precondition of step 11: for example a fine-grained read-only token of an account that is not enrolled, not an administrator and not an organization owner, which Max accepts by name) | The mechanism can hold with another procedure. Record the result, and do the named change of this checklist through a review cycle. No second run: it would give the same result |
     | Informational | X7, X9, X13, X16, X17, X19, X22; X8 (c); X15 (d); X23; the set-up records "the web form offers a SHA pin" and "how the web form names the target" | Record the result. X9 and X19 change a residual risk or the pin procedure of `docs/merge-guard-key-protection.md`; the runbook says how |

   - **X23 is informational by design.** The lab registers no self-hosted
     runner: GitHub advises against self-hosted runners on a public
     repository [docs D23], and the lab repositories are public. X23
     records what the documentation states and nothing else, so the
     question "can a repository-level runner with the label
     `ubuntu-latest` take the guard job" stays [inferred, untested]. The
     organization setting that keeps repository-level self-hosted runners
     away from `rulespec-us` is therefore a precondition of step 11 in
     every case; an organization owner verifies it there. No step of the
     experiment changes an organization setting.
   - The read-back of the lab ruleset (`refs/heads/main` in `include`,
     runbook §1.3) is a set-up condition, not a result: the lab does not
     start until it holds.
   - That lock 3 of draft #2289 refuses an entry with `ref` is
     [author-reported, review round 3].

   - If X18 fails because a ruleset run reports the source repository in
     every value, the runner's inert path would pass every pull request with
     no evaluation (§4, "The inert path"). No ruleset may name the runner,
     in `Evaluate` either, until the change is merged and X18 passes.
4. AGENT, only if the results ask for it: a pull request with each named
   change of step 3, through a review cycle. **M** merges. One change is
   always needed: the recorded response of X14 replaces the synthetic
   fixture of the lock for the ruleset form (5c, draft #2289), in a reviewed
   pull request; if the real response has another shape, that pull request
   also changes the lock [author-reported, draft #2289; experiment X14].
5. AGENT and **M** Review the guard at the candidate commit, as §6 of
   `docs/merge-guard-key-protection.md` lists: an independent agent review
   and a maintainer's own reading of the workflow file and the eight bundle
   files. The candidate is a commit on axiom-encode `main` that contains
   steps 2 and 4 and, if Max accepted question 14, its change (step 0).
   - Expected: no open finding; the commit SHA and its `BUNDLE_SHA256`
     recorded in §11.
   - Check:
     `gh api "repos/TheAxiomFoundation/axiom-encode/contents/.github/workflows/content-bound-merge-guard.yml?ref=<sha>" --jq .content | base64 -d | grep -n BUNDLE_SHA256`,
     and `gh api repos/TheAxiomFoundation/axiom-encode/compare/<sha>...main --jq .status`
     prints `ahead` or `identical` (the commit is on `main`).
6. **M** (an organization owner) Create the organization ruleset in
   `Evaluate`, with the values of `docs/merge-guard-key-protection.md` §5:
   target `rulespec-us`, the branch `refs/heads/main` (named explicitly, not
   "the default branch"), the workflow
   `.github/workflows/content-bound-merge-guard.yml` of axiom-encode at the
   reviewed SHA, empty bypass list. This starts the shadow period. It is
   also the evidence of acceptance that step 0 names.
   - Expected: the next push, update or reopen of a `rulespec-us` pull
     request starts a `content-bound-merge-guard` run there. Pull requests
     that were already open get no run until then [docs D2]. No merge is
     blocked.
   - Cost from this step on: every pull request head starts one guard job.
     A head that reaches the CI rule (every v5 lane) holds one hosted runner
     until its `validate / validate` run ends, about 3 to 4 hours, and makes
     about 20 requests an hour with the repository's job token (§4, "What
     the wait costs"). Most heads today have no lane and fail in seconds.
   - Check (owner): `gh api orgs/TheAxiomFoundation/rulesets/<id> --jq '{enforcement, bypass_actors, conditions, rules}'`.
     The evidence of the shadow period is each run's log, in the Actions
     tab of `rulespec-us`, for anyone [inferred]. Rule Insights shows an
     evaluation only after a pull request is merged or a merge is attempted
     [docs D2]. Viewing it at the organization level is part of the
     organization's ruleset permission [docs D16].
   - Rollback: set the enforcement to `Disabled`. Free at this step (the
     rule at the top of this checklist).
7. AGENT and **M** Shadow period.
   - Run cases 15 and 16 of §7 live on a throwaway `rulespec-us` pull
     request and confirm both results. Case 15 under the ruleset is a
     workflow in the pull request with the guard's job name. In `Evaluate`
     this shows one thing only: the ruleset's own run stays failed. It
     cannot show whether the imitation would satisfy the rule, because
     nothing is required yet; only the lab shows that, with its ruleset
     `Active` [experiment X2]. Case 16 is a pull request that rewrites
     `repository-checks.yml`: the guard run must fail (with no manifest
     and the infrastructure lane off, no lane claims a workflow file).
   - **Cases 17 and 18 cannot be produced on a live pull request.** Case 17
     is a response of GitHub that nobody can order: a truncated tree, an
     11th page of reviews, a redirect, an HTTP 403 or 500. Case 18 is a
     dismissal between two reads of one evaluation. Both stay
     fixture-tested [code: the `test_case_17_*` and `test_case_18_*` tests
     of `tests/test_content_merge_guard.py`;
     `test_case_17_a_failed_read_at_any_index_fails_the_job` and
     `test_case_18_a_dismissal_during_the_evaluation_fails` of
     `tests/test_merge_guard_runner.py`]. What the live shadow period can
     show: that the real responses of `rulespec-us` have the shapes that
     the fixtures assume, each real verdict with its reason, the request
     count, and any failed or rate-limited read that happens by itself
     (record each; the expected result is a failed job). What it cannot
     show: that the guard fails closed on a fault that did not happen.
   - Shadow canary: the eight open Arizona candidates rulespec-us #2197,
     #2198, #2199, #2200, #2201, #2273, #2274 and #2281. Agents only read
     them. Each needs one new event to get a run; a maintainer decides
     whether to reopen or update them. Expected result today: `fail`,
     `no_lane_for_guarded_change`, because an unsigned candidate has no
     manifest. The CLI gave that verdict for #2197 and #2273 on 2026-11-17
     [live read 2026-11-17] and for #2281 [author-reported].
   - Check: read each run's JSON report line in its log, or, as anyone,
     `uv run axiom-encode merge-guard --repository rulespec-us --pr <n>`
     (anonymous limit: 60 requests an hour; the fixtures count 5 to 29
     requests for one evaluation).
   - **Hold for step 11 (the inert path).** At least one pull request with
     no lane must show a **failed** `content-bound-merge-guard` job under
     this `Evaluate` ruleset. The job's log must hold a report line in
     which `repository` is `TheAxiomFoundation/rulespec-us`, `verdict` is an
     object (not `null`) whose `verdict` is `fail` and whose `reason` is
     `no_lane_for_guarded_change`, and no `"inert"` key exists. The key
     `verdict` alone proves nothing: the inert report and every error
     report carry it with the value `null` [code: `enforce_command`]. The
     Arizona candidates above are such pull requests today. A passed job on
     a no-lane pull request, or a report with `"inert"`, means the run took
     the source-repository path: stop, set the ruleset to `Disabled`, and
     do not go to step 11 (§4, "The inert path").
     Check: `gh run view <run id> --repo TheAxiomFoundation/rulespec-us --log | grep -F '"schema":"axiom-encode/merge-guard-enforce-report/v1"'`
     and read `repository`, `verdict.verdict`, `verdict.reason` and the
     absence of `inert` in that line; `gh run view <run id> --repo TheAxiomFoundation/rulespec-us --json conclusion --jq .conclusion`
     prints `failure`.
     This hold covers a `pull_request` run. The same reading of a
     `merge_group` run's report is part of step 8.
   - Also record here what the experiment cannot: the `x-ratelimit-limit`
     of a job token of `rulespec-us`, any rate-limited run, and the number
     of guard jobs that wait at the same time. The duration of
     `validate / validate` on a pull request is already measured: 172 to 227
     minutes for 38 of the 40 newest successful runs, median 207.5 minutes
     [live read 2026-11-17, §4]. The guard's wait is 5 hours.
8. **M** (a `rulespec-us` administrator) Turn on the merge queue for
   `rulespec-us` `main`, with the settings that rulespec-us#2160 lists:
   only merge non-failing pull requests, group size 1, build concurrency 1,
   a status check timeout above 4 hours, merge method squash. **Four
   settings of the `merge_queue` rule are required, not optional, and are
   read back:** `grouping_strategy: "ALLGREEN"`, `max_entries_to_merge: 1`,
   `min_entries_to_merge: 1`, `max_entries_to_build: 1`. They are what
   makes one merge group one judged pull request: the evaluator binds the
   group's content and cannot prove its membership (§6), and the writer's
   lock of draft #2289 refuses any other values [author-reported, draft
   #2289]. Read them, as an administrator:
   `gh api repos/TheAxiomFoundation/rulespec-us/rules/branches/main --jq '.[] | select(.type == "merge_queue") | .parameters | {grouping_strategy, max_entries_to_merge, min_entries_to_merge, max_entries_to_build}'`
   must print exactly those four values; any other value, or a missing
   field, stops the activation. The guard
   also needs build concurrency 1 (§6). The guard's merge-group job waits
   for the group's `validate / validate` run, for at most 5 hours, and
   evaluates after it, then reads the group's check once more. Its longest run is 321 minutes (§4, "Job timeout").
   The queue's status-check timeout has a maximum of 360 minutes [docs D7:
   `check_response_timeout_minutes`, "maximum: 360"]; set it to 360.
   - **The queue is a repository setting.** "Require merge queue" is not
     available in an organization ruleset [docs D1]. A `rulespec-us`
     administrator turns it on, and a `rulespec-us` administrator can turn
     it off again, in one click, with no review. See step 11 and "Periodic
     read-only health check".
   - The trial pull request: a maintainer picks one small pull request that
     may merge into the real `main` today and that needs no v6 manifest, for
     example a documentation change of the five-file allowlist. It merges
     for real. Record its number in §11.
   - **What `Evaluate` can show here, and what it cannot.** The guard is not
     a requirement in `Evaluate` [docs D4]. So the queue merges the trial
     pull request when the group's `validate / validate` run is green,
     without waiting for the guard job. The guard's next poll then comes up
     to 180 seconds later and its one evaluation judges a pull request that
     is already merged: the expected report is `fail`, `pr_identity`
     [inferred; §11 records that verdict for a merged pull request].
     Whether an `Evaluate` ruleset starts the workflow on `merge_group` at
     all is not known [experiment X20].
   - Expected, if a `merge_group` guard run starts: the guard job stays in
     progress until the group's `validate / validate` run has completed and
     then prints its report. Record: the name, app and `details_url` of the
     group's check on the group commit; the `event`, `path` and `head_sha`
     of its workflow run; `ci_polls` and `waited_seconds` of the guard's
     report; the report line itself (`repository`, no `"inert"`); and the
     duration of the group's `validate / validate` run, which no one has
     measured (no `merge_group` run of that workflow exists yet
     [live read 2026-11-17]). If the guard's report says that a check "is
     not the merge group run of the required workflow", the real
     `merge_group` run differs from what the runner expects (§6): stop, and
     prepare a reviewed change.
   - **Stop rule for the inert path on a merge group.** Read the report
     line of the `merge_group` guard run as step 7 reads the one of a pull
     request. Its `repository` must be `TheAxiomFoundation/rulespec-us`,
     and it must have no `"inert"` key. A report with an `"inert"` key, or
     a job that ended with success while the `verdict` of its report is
     `null`, means that the merge-group run took the source-repository
     path: stop, set the ruleset to `Disabled` (free at this step), and do
     not go to step 11. A job that ended with success and a non-null
     verdict is possible here and is no stop: a trial pull request of the
     five-file allowlist is `not_applicable` when the evaluation reads it
     before the merge.
   - **Established only by the lab, with its ruleset `Active`:** that the
     queue merges only after the guard job has ended, and how soon
     [experiment X15 (c)]; that a failed guard job removes the entry
     [experiment X15 (a), (b)]. Nothing on `rulespec-us` shows either before
     step 11. Step 12 repeats both observations on the first `Active`
     landing, with a stop rule.
   - Check: `gh run list --repo TheAxiomFoundation/rulespec-us --event merge_group --limit 5`,
     and, as anyone, that the queue is required:
     `gh api repos/TheAxiomFoundation/rulespec-us/rules/branches/main --jq '[.[].type]'`
     contains `merge_queue`. On 2026-11-17 it printed `[]`
     [live read 2026-11-17]. The call lists rules of rulesets; that it
     also lists a queue that classic branch protection requires is not
     documented on the pages read [inferred]. Configure the queue as a
     repository ruleset, as the experiment does (X5), because the lock of
     draft #2289 reads this call [author-reported, draft #2289].
   - **The queue ruleset must have an empty bypass list.** An actor on the
     bypass list of that repository ruleset merges without the queue, so
     without the merge-group evaluation, and `rules/branches/main` still
     lists `merge_queue` [inferred from the documented bypass model, docs
     D4; not tested live]. Check, by an administrator of `rulespec-us`
     (the maintainer's token on this machine has that access; an agent
     session may run this GET):
     `gh api repos/TheAxiomFoundation/rulespec-us/rules/branches/main --jq '.[] | select(.type == "merge_queue") | .ruleset_id'`
     gives the id, and
     `gh api repos/TheAxiomFoundation/rulespec-us/rulesets/<id> --jq .bypass_actors`
     must print `[]`. GitHub returns `bypass_actors` only to a caller with
     write access to the ruleset [docs D7]: an anonymous read, or a token
     without that access, cannot run this check, and `null` is not a pass.
     The lock of draft #2289 does not read bypass lists
     [author-reported, review round 3], so it cannot see this.
   - Rollback: turn the queue off. **WARNING:** allowed only while the
     ruleset is not `Active`. From step 11 on, a queue that is off silently
     reopens the stale-green window (§9, "Dismissal just after a pass"),
     and from step 13 on it also falls under the rollback order at the top
     of this checklist.
   - Cost: every landing then runs the full validation matrix a second
     time, on its merge group: 2.5 to 4 hours [author-reported,
     rulespec-us#2160], after 172 to 227 minutes on the pull request
     [live read 2026-11-17]. See "Throughput" in the next section: about 3
     to 4 landings a day, not the 6 to 9 that rulespec-us#2160 reports for
     the queue alone.
9. Change 2b, then change 4.
   - AGENT prepares the pull request that sets `V6_READER_ENABLED = True`.
     `applied_manifest_v6.py` is a file of the guard bundle, so the same
     pull request must set the new `BUNDLE_SHA256` in the workflow, which a
     code owner reviews [code:
     `test_the_digest_literal_is_the_digest_of_the_working_tree_bundle`].
     **M** merges. The ruleset's pin does not move: the evaluator does not
     read this switch [code: `content_merge_guard.py` imports only
     `is_promotion_repository` from that module].
   - AGENT then prepares the `rulespec-us` pull request that pins the
     encoder at that commit. **M** merges it. If Max accepted question 14,
     that commit contains its change (step 0).
   - **Change 4 is heavy.** On 2026-11-17 the encoder pin of `rulespec-us`
     (`.axiom/workflow-toolchain.toml`) was 1,270 commits and 136 versions
     behind axiom-encode `main`; the last pin move of 88 versions needed a
     staged waiver for a module that then failed validation
     [author-reported, activation preparation 2026-11-17]. Expect failing
     modules and waiver work before the v6 reader can be pinned.
     Recommended, and a maintainer's decision: open a dry-run pin pull
     request with a current `main` SHA early, before step 9, to find them.
   - **Change 4 has two modes, and the choice needs a maintainer's
     decision.** A trial pin of `rulespec-us` to encoder 0.2.2385 (draft
     rulespec-us#2304, not to be merged) gave these interim facts
     [author-reported, pin trial 2026-11-18, rulespec-us#2304; interim].
     (1) *Pinned mode*, which `rulespec-us` uses today: a pin cannot move
     past the producer of any manifest that is already on `main`. The
     trial gave 13 new manifest-audit issues on the 9 manifests added
     since 2026-10-25 ("the pinned encoder commit is not an ancestor of
     the producer"). So change 4 as a plain pin move fails CI. (2)
     *Verifier-main mode* (the producer floor, `encoder_verifier = "main"`;
     TheAxiomFoundation/.github#154 is merged, and its adoption is draft
     rulespec-us#2059): the same head gives 0 new issues. So change 4 in
     practice depends on rulespec-us#2059. (3) **In verifier-main mode
     `rulespec-us` verifies with encoder `main`. Then
     `V6_READER_ENABLED = True` on encoder `main` (change 2b) takes effect
     on `rulespec-us` when it merges; no separate reader pin exists to
     move or to revert; and the rollback "revert the reader pin" becomes
     "set `V6_READER_ENABLED` back to `False` on encoder `main`", a
     reviewed encoder change.** This step and the rollback order at the
     top are written for pinned mode; under verifier-main mode they need
     a reviewed rewrite first. (4) Validation at the new encoder, local:
     2,958 unwaived modules, 0 newly failing in validation, proof and
     companion tests; two waiver-ratchet changes (one fingerprint drift,
     one waiver that is stale because its module now passes). The shared
     workflow refuses a pin pull request that changes anything beyond the
     pin files, so a waiver edit cannot ride with the pin move.
   - The switch rule of step 2 governs: the writer stays off.
   - Check: `git grep -n "^V6_READER_ENABLED" origin/main -- src/axiom_encode`
     prints `True`; the `rulespec-us` pin names the commit:
     `gh api repos/TheAxiomFoundation/rulespec-us/contents/.axiom/workflow-toolchain.toml --jq .content | base64 -d | grep axiom_encode`.
   - Rollback: set the constant back; revert the pin.
10. **The infrastructure lane (question 0 of §10).** `Active` without a lane
    blocks every pin, ledger, workflow and tooling pull request of
    `rulespec-us`: the earlier author-reported census classified 48 of 50
    pull requests as refused without a lane. The lane must not first give
    verdicts when it already enforces. So it goes live here, while the
    ruleset is still in `Evaluate`, in this order:
    1. **M** (Max) decides question 0: the proposed three-class lane of §3,
       another lane, or no lane. With "no lane" this checklist stops here:
       step 11 requires a lane that is on at the pinned SHA, and an
       `Active` ruleset with no lane blocks every such pull request (the
       WARNING of step 11). `Active` with no lane needs a reviewed rewrite
       of steps 10 and 11 first. Another lane needs its own design and
       review first; this step then waits.
    2. AGENT prepares the lane change, one pull request on axiom-encode:
       `INFRASTRUCTURE_LANE_ENABLED = True`; the lane's two success reasons
       added to the runner's `SUCCESS_VERDICTS` (`control_plane_content_bound_approval`
       and `infrastructure_as_today`); the new `BUNDLE_SHA256`, because
       `content_merge_guard.py` and `merge_guard_runner.py` are bundle
       files. It goes through a review cycle. **M** merges.
    3. AGENT and **M** review the guard at the new commit, as in step 5,
       and record the new SHA and `BUNDLE_SHA256` in §11.
    4. **M** (an organization owner) moves the ruleset's pin to the new
       commit, **while the ruleset is still `Evaluate`**
       (`docs/merge-guard-key-protection.md` §5, "Per guard release"). The
       writer is off, so no lock constant names the pin yet.
    5. AGENT and **M** observe again, at the new pin: at least one
       control-plane pull request shows `fail`, `no_approval` before the
       other enrolled owner approves it and `pass`,
       `control_plane_content_bound_approval` after the approval and a
       re-run; a content pull request with no manifest still shows `fail`,
       `no_lane_for_guarded_change`.
    6. **The hold of step 7 is repeated at the new pin.** The evidence of
       step 7 belongs to the earlier commit. A pull request with no lane
       (an Arizona candidate, after one new event) must again show a failed
       job whose report has a non-null `verdict` and no `"inert"`.
    - Expected: no merge is blocked; the ruleset is still `Evaluate`.
    - Check (owner): `gh api orgs/TheAxiomFoundation/rulesets/<id> --jq '{enforcement, sha: .rules[0].parameters.workflows[0].sha}'`
      prints `evaluate` and the new SHA. As anyone:
      `gh api "repos/TheAxiomFoundation/axiom-encode/contents/src/axiom_encode/content_merge_guard.py?ref=<sha>" --jq .content | base64 -d | grep -n '^INFRASTRUCTURE_LANE_ENABLED'`
      prints `True`.
    - Rollback: the owner moves the pin back to the commit of step 5; a
      reviewed change sets the constant back. Free at this step (the rule
      at the top of this checklist).
11. **M** (an organization owner) Switch the ruleset to `Active`. Hard
    preconditions, all of them: steps 0-10 are done; change 3b is on `main`
    and is the code at the pinned SHA; the experiment passed, as step 3
    defines it; **under the `Evaluate` ruleset, at the pin that is in the
    ruleset now, a pull request with no lane shows a failed
    `content-bound-merge-guard` job whose report has a non-null `verdict`
    and no `"inert"`** (the hold of step 7, repeated in step 10; a passed
    job there means the runner took its source-repository path, which would
    pass every pull request); the merge queue of step 8 is on, is still
    required at this moment (the check below), and its trial showed the
    guard's merge-group job ending after the group's `validate / validate`
    run; **the second canary: in that trial the merge group of a pull
    request with no lane showed a failed `content-bound-merge-guard` job of
    the `merge_group` event, whose report names `rulespec-us` as
    `repository` and has no `"inert"`** (the `merge_group` payload has no
    second repository field, so only this live run shows that the queue
    event is judged and not taken for the encoder's own, §4 "The inert
    path"); the merge-queue ruleset of `rulespec-us` has an empty bypass list
    (step 8's check, repeated now); **an organization owner has verified,
    at this moment, that the organization's "Runners" setting allows no
    repository-level self-hosted runner for `rulespec-us`** (the owner's
    check below); **no stored secret is a credential of an enrolled identity, an administrator, an
    organization owner or an App that can write checks or contents (an
    environment is an exception only by name and with verified controls), and an
    owner has read the installed Apps with their target repositories** (the check "Stored credentials and
    installed Apps" below); Actions are enabled in `rulespec-us`; and Max has decided question 0 of §10 **and its lane is on at the
    pinned SHA** (step 10).
    **WARNING:** without the lane of question 0, an `Active` ruleset blocks
    every pin, ledger, workflow and tooling pull request on `rulespec-us`,
    administrators included if step X9 of the experiment holds, and the
    rollback itself needs an organization owner. Under the App fallback the same step adds a required check:
    require the name `content-bound-merge-guard` only, and never
    `content-bound-merge-guard (shadow)`, which is always neutral; a neutral
    check satisfies a required check, so it would enforce no verdict.
    - Expected: a `rulespec-us` pull request merges only when its guard run
      is green. Pull requests whose run passed in `Evaluate` need no new run
      [docs D1].
    - Check, as anyone:
      `gh api repos/TheAxiomFoundation/rulespec-us/rules/branches/main --jq '[.[].type]'`
      must contain `workflows` **and** `merge_queue` (the call lists active
      rules only [docs D7]). Run it before the switch for `merge_queue` and
      after it for both. Owner, after the switch:
      `gh api orgs/TheAxiomFoundation/rulesets/<id> --jq '{enforcement, bypass_actors, conditions, rules}'`
      shows `active`, an empty bypass list, `refs/heads/main` and the one
      workflow entry. An administrator of `rulespec-us`, before the switch:
      the `bypass_actors` read of step 8 prints `[]`.
    - **Owner's precondition check: the organization's "Runners" setting.**
      An organization owner reads Organization settings → Actions → General
      → "Runners". It must be "Disabled", or "Selected repositories"
      without `rulespec-us` [docs D19]. The same read through the API,
      which needs `admin:org` [docs: REST "Get self-hosted runners settings
      for an organization"]:
      `gh api orgs/TheAxiomFoundation/actions/permissions/self-hosted-runners --jq .enabled_repositories`
      must print `none`, or `selected` with `rulespec-us` absent from
      `gh api orgs/TheAxiomFoundation/actions/permissions/self-hosted-runners/repositories --jq '[.repositories[].name]'`.
      The owner also confirms that no organization runner group that
      `rulespec-us` can use holds a runner with the label `ubuntu-latest`,
      and that the organization setting "Allow repository administrators
      to rename branches protected by organization rules" is off
      [docs D22]. The owner records the three reads in §11. An agent
      cannot make them: they need `admin:org`.
      **If the "Runners" setting has another value, step 11 waits.**
      Changing it is an act on the whole organization, not on
      `rulespec-us`: runners that exist in any repository then get no new
      jobs [docs D19]. An owner decides it as its own change, after a read
      of which repositories have repository-level runners, and can set it
      back. No step of the experiment and no agent changes it. Why it is
      required without a live test: step 3, "X23 is informational by
      design".
    - **Precondition check: stored credentials and installed Apps.** Any
      user with write access to a repository can read every secret of that
      repository [docs D23]. That a workflow on a pushed branch is enough,
      with no merge, and that the same holds for an organization secret
      that the repository can use, is [inferred]. With the secret the user acts as
      the credential's owner. If the owner is an enrolled identity, an
      administrator or an organization owner, every control of this design
      that rests on that identity is void for that user [inferred from the
      documented secret semantics; not tested]. That GitHub cannot restrict
      a repository secret to one workflow file is [inferred]: no page read
      offers it. An environment secret is different: a job gets it only
      when it names the environment and the environment's rules pass
      [docs D24]. The reads, all of names only (no call returns a value):
      - An administrator of each repository, for `rulespec-us` and for
        axiom-encode (the maintainer's token on this machine can run these
        four GETs, so an agent session may):
        `gh api --paginate repos/TheAxiomFoundation/<repo>/actions/secrets --jq '.total_count as $n | [.secrets[] | {name, updated_at}] | {listed: length, total_count: $n, secrets: .}'`,
        `gh api --paginate repos/TheAxiomFoundation/<repo>/actions/organization-secrets --jq '.total_count as $n | [.secrets[] | {name, updated_at}] | {listed: length, total_count: $n, secrets: .}'`,
        `gh api repos/TheAxiomFoundation/<repo>/environments --jq '[.environments[] | {name, can_admins_bypass, rules: [.protection_rules[].type]}]'`
        and, for each environment,
        `gh api --paginate repos/TheAxiomFoundation/<repo>/environments/<name>/secrets --jq '.total_count as $n | [.secrets[] | {name, updated_at}] | {listed: length, total_count: $n, secrets: .}'`.
        The record holds each name **with its `updated_at`**: a secret
        that is replaced under the same name keeps the name and changes
        the date (review of head `dd8bf4fa2`: a projection of names alone
        printed the same list for two different dates). A read is whole
        only when one page was printed and `listed` equals `total_count`;
        otherwise it is incomplete, and an incomplete read is not a
        record: an owner or administrator verifies the list in the
        settings pages.
      - An organization owner, with `admin:org`:
        `gh api orgs/TheAxiomFoundation/actions/secrets --jq '[.secrets[] | {name, visibility}]'`.
        The maintainer's token gets HTTP 403 there
        [live read 2026-11-18].
      - An organization owner:
        `gh api --paginate orgs/TheAxiomFoundation/installations --jq '.total_count as $n | [.installations[] | {id, app_slug, repository_selection, updated_at, permissions}] | {listed: length, total_count: $n, installations: .}'`
        [docs: REST "List app installations for an organization", which
        states that the caller must be an organization owner], and, for
        each installation whose `repository_selection` is `selected`, the
        repositories it is granted:
        `gh api --paginate user/installations/<installation id>/repositories --jq '.total_count as $n | [.repositories[].full_name] | {listed: length, total_count: $n, repositories: .}'`
        [not run by the author; the REST page "List repositories
        accessible to the user access token" was not read for this
        change, so the owner confirms the list on the installation's
        settings page when the call is refused]. `selected` alone does
        not show a change of membership: the record holds, for each App,
        its permissions, its `updated_at` and the **effective list of
        target repositories** (`all`, or the names). An incomplete or
        refused read is not a record: the owner verifies it by hand.
      - **Must hold.** No stored secret is a credential of an enrolled identity, of an
        administrator of `rulespec-us`, of an organization owner, or of an
        App that can write checks or contents on `rulespec-us`. A name does
        not show whose credential a secret is: for each secret a maintainer
        states the owner, and the list of names with that statement is
        recorded in §11. Each installed App that can write checks on
        `rulespec-us` is removed from it, or Max accepts it by name as a
        residual risk: that a check of such an App cannot satisfy the rule
        is [inferred, untested] (§9). **A removal alone does not last:**
        a repository administrator can add an App that an organization owner has already installed to the repositories that this administrator administers, whatever the App's permissions are
        [docs source D26]. It lasts only when an organization owner also
        restricts the installation of Apps by repository administrators,
        which the same text names. So the read of the installed Apps is
        repeated at each health check.
      - **An environment is no general exception** (review of head
        `dd8bf4fa2`). The presence of `required_reviewers` is not enough:
        administrators can bypass environment protection rules unless the
        environment disallows it, and a repository administrator can edit
        an environment's reviewers and rules
        (`docs/merge-guard-key-protection.md`, P2). The projection of rule
        types above does not show either. Such a credential may stay in
        an environment only when Max accepts it by name as a residual and
        the effective controls are read and recorded in §11: the read
        `gh api repos/TheAxiomFoundation/<repo>/environments/<name> --jq '{name, can_admins_bypass, deployment_branch_policy, rules: [.protection_rules[] | {type, prevent_self_review, reviewers: [.reviewers[]? | {type, login: (.reviewer.login // .reviewer.slug)}]}]}'`
        shows `can_admins_bypass: false`, required reviewers with
        `prevent_self_review: true` who are named people outside the
        actors of §9 (no agent session holds their token), and the
        deployment branch policy lists `main` only. A secret in an
        environment that has a branch policy and no required reviewers
        has no such controls at all. The environments `apply-signing` and
        `encoder-version-bump` of axiom-encode are of that kind
        [live read 2026-11-18]. Their secrets are judged like repository
        secrets, and until a maintainer states whose credential each is,
        the precondition is not met for them.
      - **State on 2026-11-18** [live read 2026-11-18]: this precondition
        does not hold. `rulespec-us` has the repository secret
        `BULK_ENCODE_TOKEN`, whose owner was not read
        (`docs/merge-guard-key-protection.md` §4.4, R14, and §5, step E,
        give the whole list). Three installed Apps have all repositories
        and `checks: write` (R12 (f) there).
    - **Residual risk at this step.** The merge queue is a repository
      setting of `rulespec-us`. Its administrators can turn it off, and
      nothing then blocks merges or raises an alarm; the stale-green window
      is open again (§9). The lock of draft #2289 re-checks both rules at
      each admission and revalidation of a signing run, and so stops new
      signing runs; it does not see a queue that is turned off after
      signing, and it does not run for the other lanes [author-reported,
      draft #2289]. "Periodic read-only health check" below is the only
      detection.
    - Rollback: set the enforcement back to `Evaluate`. Before step 13 this
      is free. **After step 13, only in the rollback order at the top of
      this checklist: first the writer off, then no signing run alive and
      the reader pin reverted, only then the ruleset.** That order is a
      last resort and not a durable rollback: after it nothing technical
      protects `rulespec-us` from signed v6 bytes. The recommended response
      is to repair forward, or to keep the ruleset `Active` with the writer
      off.
    - Optional hardening, **M**: enforce code-owner review for `/.github/`
      in `rulespec-us` (§9). That protects the RuleSpec workflows; it is not
      part of the guard.
12. Run the current-order canary to completion: one promotion in today's
    order (`docs/reviewed-pr-promotion.md`) through the `Active` guard.
    - Expected: the signed candidate is a v5 lane. Its guard run starts
      with the push of the signed head, finds `validate / validate` not yet
      green, waits for it, and ends `not_applicable`, `earlier_lane_v5`
      (job success) **in the same run, with no re-run**. The wait is as long
      as that check: 172 to 227 minutes for 38 of the 40 newest successful
      runs [live read 2026-11-17]; the guard waits at most 5 hours. A
      re-run of the guard job is needed only if the check takes longer
      than 5 hours or a poll fails (the job then ends `fail`,
      `ci_not_green`). The unsigned candidate before signing is `fail`,
      `no_lane_for_guarded_change`; that is expected and blocks nothing
      that could merge today.
    - This is the first landing through the `Active` ruleset and the queue.
      Record what step 8 could not show: the `completedAt` of the guard's
      merge-group job and the pull request's `merged_at`. **Stop rule:** if
      the pull request merges before the guard's merge-group job has ended,
      the ruleset result is not part of the queue's requirements. Set the
      ruleset back to `Evaluate` (free: step 13 has not happened) and do
      not go on. The same stop applies when the report line of the
      merge-group guard run has an `"inert"` key, or its job ended with
      success while the `verdict` of the report is `null` (the reading of
      step 8).
    - Check: the report line of the guard run on the signed head, with
      `ci_polls` and `waited_seconds`; the report line of the merge-group
      guard run (`repository`, a non-null `verdict`, no `"inert"`); then
      `gh run view <merge-group run id> --repo TheAxiomFoundation/rulespec-us --json jobs --jq '.jobs[] | {name, completedAt}'`
      and `gh api repos/TheAxiomFoundation/rulespec-us/pulls/<n> --jq .merged_at`.
    - Rollback: step 11's rollback.
13. Change 6, as one reviewed pull request. AGENT prepares it; it is not
    opened before steps 0-12 are done, because merging it takes effect.
    **Hard precondition: the check "Stored credentials and installed Apps"
    of step 11 is repeated now and holds.** **Hard precondition: a live
    `merge_group` run of the guard has rejected a dismissed approval**
    (experiment X15 (a) and (b) in the lab, recorded in §11): the
    pull-request job stays green after a dismissal, so before the pre-sign
    approval is removed the queue-time evaluation must be shown to work,
    and the queue must still be required with a group size of 1 (step 8's
    check, repeated now). With the writer on, whoever
    holds an enrolled owner's credential can dispatch and approve. It
    records the ruleset's workflow identity in the writer's lock
    (`GUARD_WORKFLOW`: repository id, path, and the set of pinned SHAs,
    which holds exactly the SHA that the ruleset pins now), sets
    `V6_WRITER_ENABLED = True`, and updates
    `docs/reviewed-pr-promotion.md`. The lock itself is change 5c (draft
    #2289, merged in step 2, fixture replaced in step 4)
    [author-reported, draft #2289]. The pinned SHA is an earlier commit
    than the one that records it. `reviewed_pr_admission.py` is a bundle
    file, so the pull request also sets the new `BUNDLE_SHA256`. The
    ruleset's pin does **not** move for this change: the guard reads
    neither the switch nor `GUARD_WORKFLOW`
    [author-reported, activation preparation 2026-11-17]. **M** merges.
    - From this step on, every pin move needs the lock constant updated in
      the same sequence: a reviewed commit adds the new SHA, the owner
      moves the pin, a reviewed commit removes the old SHA
      (`docs/merge-guard-key-protection.md` §5, "Per guard release"). A pin
      that the set does not hold makes the writer refuse every promotion
      (closed).
    - Check: the writer's own refusal tests pass with the recorded identity;
      `git grep -n "^V6_WRITER_ENABLED" origin/main -- src/axiom_encode`
      prints `True`; the queue and the ruleset are still required (the
      check of step 11).
    - Rollback: set the switch back, in a reviewed change. The ruleset
      stays `Active`. **This is point 1 of the rollback order at the top of
      this checklist, and it comes before any rollback of step 8 or step
      11.**
14. The smallest live canary of the new order, on one real candidate. The
    activation preparation chose rulespec-us#2274: it is the smallest of
    the eight Arizona candidates of step 7 (2 files, 52 added lines), adds
    new paths only, and has no blocking review. rulespec-us#2281 cannot be
    used as it is: it has an effective `CHANGES_REQUESTED` review by
    MaxGhenis, which the admission and the guard both refuse
    [author-reported, activation preparation 2026-11-17]. At each arrow,
    record the URL and the guard's report line.
    1. AGENT or contributor: local encode → candidate pull request. Guard
       run: `fail`, `no_lane_for_guarded_change`.
    2. **M** (an enrolled dispatcher; it may be the candidate's author)
       dispatches `targeted-signed-reencode.yml`. The `production-signing`
       environment approval still applies to the run.
    3. The run opens a draft pull request with the signed v6 manifest into
       the candidate branch. **M** (a maintainer with write access who will
       **not** be the approver) marks it ready and squash-merges it there.
       That push must come from a person's or an
       App's credential, not a job token (§4, "Limits of the trigger set").
    4. The push starts CI and the guard run on that head. Guard run:
       `fail`, `no_approval`, at once, with no wait. CI on the candidate
       pull request goes green after about 3 to 4 hours (172 to 227 minutes
       measured [live read 2026-11-17]).
    5. **M** (the other enrolled owner) reads the green pull request and
       approves it on GitHub.
    6. **M** (a maintainer with write access) re-runs the failed guard job,
       or follows the procedure that step X3 found. This re-run is needed
       because the approval arrived after the guard job had ended. CI is
       already green, so the re-run does not wait. Guard run: `pass`,
       `content_bound_approval`.
    7. If `main` moved, **M** (a maintainer who is not the approver)
       updates the branch first. That is a new head: CI runs again, about 3
       to 4 hours, and the new guard run waits for it and passes in the
       same run, because the approval stays valid across a base-only
       update. Then **M** adds the pull request to the merge queue. The
       group's `validate / validate` ends green after 2.5 to 4 hours
       [author-reported, rulespec-us#2160]; the `merge_group` guard run,
       which waited for it, then evaluates and is `pass`; GitHub merges.
    - **End-to-end time for this one candidate**, signing excluded: the
      pull-request CI (2.9 to 3.8 hours measured), the approval (a person's
      time), and the queue's CI (2.5 to 4 hours, author-reported, plus up
      to 3 minutes for the guard's last poll and its evaluation). That is
      5.4 to 7.8 hours of machine time when `main` does not move. Each
      landing of another pull request in that time adds one more
      pull-request CI, 2.9 to 3.8 hours.
    - Negative canary. A dismissal of the approval before the pull request
      is queued, or while the group's `validate / validate` run is in
      progress, makes the `merge_group` run fail (`no_approval`) and the
      entry leaves the queue. An approval by the dispatcher or by the
      author (`approver_is_dispatcher`, `approver_is_author`) **cannot be
      shown live on any of the eight candidates**: all eight have the author
      PavelMakarchuk, who must then also be the dispatcher, and GitHub does
      not let a pull request's author approve it. A live proof of
      `approver_is_dispatcher` needs a throwaway candidate whose author is
      neither enrolled owner [author-reported, activation preparation
      2026-11-17]. The fixtures cover both reasons [code: §7].
    - Rollback: step 13's rollback; close the candidate.
15. "Continuous landing is on" when step 14 passed and the maintainers say
    so in §11. What that means, and which human steps remain, is the next
    section.

Steps 11 and 13 land as separate reviewed changes, step 11 first, which keeps
proposed constraint D2d ("deployed and enforced before the pre-sign approval
is removed"). D2d forces the mechanism of step 1, the enforcing runner
(change 3b), the `Active` ruleset (step 11) and the v6 reader pin (step 9)
before activation (step 13). The canary (step 12) follows the proposed
sequence. This design chose the shadow period and the live cases (steps 6
and 7); D2d does not require them.

### Periodic read-only health check

An agent may run these. Each is a GET with the token that agent sessions
already have; none needs `admin:org`. Four rows need more than an
anonymous read (the runners, the bypass list, the secret names and the
Actions permission): they need administrator access to `rulespec-us`, or
for the bypass list write access to the ruleset. The maintainer's token on
this machine can make all four [live read 2026-11-17 and 2026-11-18]. A
fifth row, the installed Apps, needs the account of an organization owner;
a session whose token belongs to a non-owner cannot run it and reports
that. Each
is a read, so it is inside the AGENT role. Run them before steps 11, 12, 13 and
14, and from step 11 on at every session that works on `rulespec-us`
landings. A wrong result is reported to a maintainer; the agent changes
nothing.

| What | Command | Must print |
|---|---|---|
| The guard workflow and the merge queue are both required on `main` | `gh api repos/TheAxiomFoundation/rulespec-us/rules/branches/main --jq '[.[].type]'` | a list that contains `workflows` and `merge_queue`. On 2026-11-17 it printed `[]`: nothing is active yet [live read 2026-11-17] |
| The rule still pins the reviewed commit | `gh api repos/TheAxiomFoundation/rulespec-us/rules/branches/main --jq '.[] \| select(.type == "workflows") \| .parameters'` | the repository id 1125775893, the workflow path and the SHA of §11. The response shape is [experiment X14] |
| The default branch is still `main` | `gh api repos/TheAxiomFoundation/rulespec-us --jq .default_branch` | `main` [live read 2026-11-17: `main`] |
| No repository-level self-hosted runner exists | `gh api repos/TheAxiomFoundation/rulespec-us/actions/runners --jq .total_count` | `0`. The call needs administrator access to the repository; with the maintainer's token it printed `0` [live read 2026-11-17]. Organization-level runners need an owner's read |
| The merge-queue ruleset of `rulespec-us` has no bypass actor | the two commands of step 8: the `ruleset_id` of the `merge_queue` rule, then `gh api repos/TheAxiomFoundation/rulespec-us/rulesets/<id> --jq .bypass_actors` | `[]`. GitHub returns the field only to a caller with write access to the ruleset [docs D7]; the maintainer's token has it, an anonymous read does not, and `null` is not a pass |
| The stored secrets are the recorded ones | `gh api --paginate repos/TheAxiomFoundation/rulespec-us/actions/secrets --jq '.total_count as $n \| [.secrets[] \| {name, updated_at}] \| {listed: length, total_count: $n, secrets: .}'`, and the three other reads of step 11, for `rulespec-us` and for axiom-encode | exactly the names **and the `updated_at` of each** that §11 records, with `listed` equal to `total_count` on one page (anything else is an incomplete read: report it; an owner or administrator verifies by hand). The first run of this check before step 11 makes that record, with the maintainer's statement of step 11; each later run is compared with it. A new name, or a changed `updated_at` of a recorded one, goes to a maintainer. The read shows names and dates; it does not show whose credential a secret is. On 2026-11-18 `rulespec-us` printed three names, `BULK_ENCODE_TOKEN` among them [live read 2026-11-18] |
| The installed Apps are the recorded ones | `gh api --paginate orgs/TheAxiomFoundation/installations --jq '.total_count as $n \| [.installations[] \| {id, app_slug, repository_selection, updated_at, permissions}] \| {listed: length, total_count: $n, installations: .}'`, then for each `selected` installation the read of its repositories of step 11 | exactly what §11 records for each App: permissions, `updated_at` and the effective list of target repositories (the first run before step 11 makes the record). A projection without the target repositories prints the same line before and after an App gains `rulespec-us` while its selection stays `selected`. An incomplete or refused read is reported and an owner verifies by hand. The call needs the account of an organization owner [docs: REST "List app installations for an organization"]; with the maintainer's token it returned four Apps on 2026-11-18 [live read 2026-11-18]. An App that is new, or that has `rulespec-us` again after its removal, goes to a maintainer |
| Actions are enabled in `rulespec-us` | `gh api repos/TheAxiomFoundation/rulespec-us/actions/permissions --jq .enabled` | `true` [live read 2026-11-18: `true`]. It shows the state at the read, not an interval [experiment X8 (c)] |
| The switches | the `git grep` of step 2 | the values that the current step expects |

What this check cannot see: an `Evaluate` or `Disabled` ruleset (the call
lists active rules only [docs D7]; an `Active` rule that disappears from the
list is the signal), who changed a setting, a queue that was off between two
checks, a bypass entry of the organization ruleset (owner's read), a bypass
entry of the queue ruleset when the caller has no write access to it, the
organization's "Runners" and branch-rename settings (owner's read), an
organization-level runner group, the installed Apps when the caller is not
an organization owner, an
organization secret (owner's read), whose credential a stored secret is,
whether someone has already read a secret, and Actions that were disabled
and enabled again between two checks. It detects; it prevents nothing.

### What "continuous landing is on" means, and the human steps that remain

**This design does not deliver "encode continuously without a per-candidate
human approval".** After activation every candidate still needs one approval
by an enrolled owner who is neither its author nor its dispatcher. The
approval moves: from before signing, on a red pull request, to after
validation and signing, on a green one. Removing it would be a policy
decision of the maintainers to accept machine validation alone, for example
a lane with no owner review. That is not designed here.

"Continuous landing is on" means: a candidate can go from a local encode to
`main` with no pre-sign owner approval, and an unrelated merge to `main`
no longer voids an approval. It does not mean that no person acts.

Each row was checked in the code of this change. Rows about the v6 writer
describe drafts #2238 and #2269, which this change does not contain
[author-reported].

| Step | Who | Removed by this workstream? | What remains | What could remove it |
|---|---|---|---|---|
| Open the candidate pull request | contributor or agent | No | The same | — |
| Approve the exact head before signing, on a red pull request | an enrolled owner, not the author, not the dispatcher [code: `_approved_reviews` in `reviewed_pr_admission.py`] | **Yes**, once change 6 is on [author-reported] | Nothing | — |
| A new approval after any merge to `main` or any push | the same owner. Admission requires the approval on the exact head and the pull request's base to equal the `main` tip [code: `verify_reviewed_pr`, `revalidate_admission`] | **Yes**: the post-sign approval stays valid across a base-only update that someone other than the approver makes [code: `content_bound` in `_evaluate`] | The base must still be current when the signing run starts and until it signs; a moved `main` means update the branch and dispatch again [code: `verify_reviewed_pr`: "PR base is not the current main head"; for the v6 lane author-reported] | A design that validates against a moved base. Not designed |
| Dispatch `targeted-signed-reencode.yml` by hand | today any account that is not the approver; in the new order an enrolled identity [code: the workflow has only `workflow_dispatch`; `R_DISPATCHER_NOT_ENROLLED` in the guard] | No | One dispatch per candidate | A reviewed automatic dispatcher. Not part of this workstream |
| Approve the `production-signing` environment for the run | one of the environment's two required reviewers, who are the two enrolled owners. `prevent_self_review` is off, so the dispatcher can approve the own run [code: job `encode`, `environment: production-signing`; live read 2026-11-17] | No | One approval per signing run | #1885 (promotion without per-run approval), if it lands |
| Merge the signed draft pull request into the candidate branch | anyone with write access **except the future approver**: the merger becomes an author of a commit of the candidate, and the guard refuses an approver who authored or committed one [code: step "Push lane branch and open draft pull request" of the signing workflow; `_actor_ids`, `R_AUTHORED_COMMITS`]. A commit that GitHub cannot attribute fails the guard (`unattributed_commit`) [code]. The workflow commits under a bot name; on rulespec-us#2045 the merged commit was attributed to the dispatcher [author-reported]. That a squash merge keeps an unattributed commit out is [inferred]; step 14 records it | No | One merge per candidate | A workflow change that pushes the signed commit to the candidate branch with an App credential. Not designed |
| Approve the green candidate | the other enrolled owner: enrolled, not the author, not the dispatcher, not an author or committer of any commit [code: `_evaluate`, steps 5 and 6]. Two identities are enrolled [code: `TRUSTED_SIGNING_REVIEWERS`]. For a candidate that one of them authored, only the other can approve, and the author must be the dispatcher | **New** (it replaces the pre-sign approval) | One approval per candidate, by design (D2f, D3) | A maintainer policy that accepts machine validation alone. Not designed |
| Re-run the guard job | a maintainer with write access, if a re-run counts [docs D9; experiment X3, X16]. Needed in two cases only. (1) The approval arrives when the guard job of that head has already ended: a v6 candidate is unapproved when its signed head is pushed, so its job fails at once (`no_approval`) and the approval comes hours later, after CI. (2) The job ended `fail`, `ci_not_green` after its wait: the required check took longer than 5 hours, or a poll failed or was rate-limited. A head whose approval exists before its run starts, and every v5 lane, needs no re-run: the job waits for `validate / validate`, 172 to 227 minutes in 38 of 40 measured runs [live read 2026-11-17; code: `_enforce_pull_request`] | **New** under the ruleset path | One re-run per v6 candidate (case 1), and one after each wait that failed (case 2) | A helper workflow on the review event, if X16 shows a job token can re-run. Not built |
| Keep the head current with `main` | a maintainer who is not the approver updates the branch. The guard requires the head to be strictly ahead of the live `main` tip, in the pull request run and in the merge-group run [code: `R_BASE_NOT_CURRENT`]. Each update is a new head: CI runs again (172 to 227 minutes measured) and the new guard run waits for it; the approval stays, so that run needs no re-run | No | One update, and one full CI run, after each other landing | A queue that rebuilds stale entries would need the guard to accept a head behind `main`. Not designed |
| Merge | a maintainer adds the pull request to the merge queue (the AGENT role of the checklist does not merge). The lane merges nothing by itself [docs: `docs/reviewed-pr-promotion.md`] | No | One action per candidate, then 2.5 to 4 hours of group validation [author-reported, rulespec-us#2160] | — |
| Approve and re-run an infrastructure pull request (pin, ledger, workflow, tooling: 48 of 50 in the census [author-reported]) | under the proposed lane of question 0: the other enrolled owner approves it, and a maintainer re-runs the guard job when the approval came after the job had ended [code: `R_CONTROL_PLANE_PASS`; the lane is off] | **New**, if Max accepts that lane (checklist step 10). Today such a pull request needs no owner approval | One approval and at most one re-run for each infrastructure pull request, and the 2.5 to 4 hours of the queue for each | Another answer to question 0. Not designed |
| Watch the repository settings and the stored secret names that the guard depends on | an agent or a maintainer runs the read-only health check of the checklist | **New** | One read per session; a wrong result needs a maintainer | An organization-level control of the merge queue. GitHub offers none [docs D1] |
| Move the ruleset's pin for a guard release | an organization owner, after the review of checklist step 5; after step 13 also two reviewed commits for the lock constant | **New** | Once per change of a bundle file that changes guard behavior (the enrolled list included) | — |

**Throughput.** The guard requires each head to be strictly ahead of the
live `main` tip [code: `R_BASE_NOT_CURRENT`]. So each landing makes every
other open candidate stale. A stale candidate needs a branch update, a full
pull-request CI run (2.9 to 3.8 hours, measured [live read 2026-11-17]) and
then the queue's CI run on its merge group (2.5 to 4 hours
[author-reported, rulespec-us#2160]). A candidate cannot run its
pull-request CI against the next `main` while the one before it is in the
queue: that `main` does not exist yet. So landings are serial, about 5.4 to
7.8 hours each, which is **about 3 to 4 landings a day**, not the "6 to 9"
that rulespec-us#2160 reports for the queue alone [inferred from the two
durations]. That count includes infrastructure pull requests: under an
`Active` ruleset they take the same queue. A design that lets a head be
behind `main`, or a queue that validates more than one entry at a time, is
not designed here.

**Candidates that replace or overlap an existing module.** The v6 lane and
today's promotion lane admit new modules only.

- A candidate whose primary, companion or manifest path already exists on
  `main` fails: `scope_not_new_module` in the guard, "new candidate paths
  already exist on main" in admission [code: `_evaluate`,
  `verify_reviewed_pr`].
- A candidate that also removes or changes another module, for example the
  legacy module it supersedes, fails: `scope_paths`, because the delta must
  be exactly the three files [code: `_evaluate`].
- A candidate that adds a module at a new path while an older module for the
  same source text stays on `main` passes the guard's scope rule. The guard
  compares paths. It does not know that two modules encode the same
  document; whether validation refuses the pair is outside this design
  [code; inferred]. The Arizona block-level candidates of step 7 add files
  at new paths [live read 2026-11-17].
- Retiring or replacing the older module needs a reviewed replacement or
  retirement lane (`docs/retired-source-replacement-admission.md`, the
  legacy-replacement receipts, question 13 of §10). The guard's v5 lane
  accepts a signed manifest that lists a modified or removed module
  [code: `_earlier_lane_refusal`]; a combined change with a receipt has no
  lane (question 13). A v6 form of that lane is not designed here.

## 9. Threat model

This table describes the default-off infrastructure lane. The proposal in §3
does not close the encoder key-exposure set or satisfy an activation condition.
Where a row differs between the two paths of §4, it says so. "Ruleset path"
is the recommendation, which Max has not accepted; "App path" is the
fallback.

| Adversary | Defended? | How |
|---|---|---|
| Write collaborator on a RuleSpec repository, not enrolled | Yes for a v6 unit; through CI for a v5 lane; blocked otherwise. All three only while no stored credential gives the collaborator another identity (next row) | Cannot dispatch, cannot give a counting approval, and cannot publish as the guard App from a RuleSpec workflow. A v6 unit refuses every extra file. A v5 lane needs this PR's own green run. Any other guarded change has no lane |
| Write collaborator who reads a credential that is stored where a workflow of a branch can read it | **No**, when the credential belongs to an enrolled identity, an administrator, an organization owner, or an App that can write checks or contents | The collaborator pushes a branch with a workflow that reads a repository secret, or an organization secret that the repository can use; no merge and no review is needed [docs D23 for a repository secret: any user with write access to a repository can read every secret that is configured in it; the organization secret and "no merge is needed" are inferred]. The collaborator then acts as the credential's owner: every control of this design that rests on that identity is void for that collaborator [inferred from the documented secret semantics; not tested]. GitHub cannot restrict a repository secret to one workflow file [inferred: no page read offers it]; an environment with required reviewers can hold a secret back [docs D24], unless an administrator bypasses or edits its rules (checklist step 11). **This exposure exists today and does not depend on the guard**: `rulespec-us` has the repository secret `BULK_ENCODE_TOKEN`, which its `bulk/README.md` describes as a token with the scopes `repo` and `workflow`; no workflow on `main` of `rulespec-us` uses it [live read 2026-11-18; code: `rulespec-us` at `ddad6d63`]. Not read: its owner, its real scopes and whether it is still valid. It also touches today's promotion lane: whoever holds an enrolled owner's token can approve. Hard precondition of checklist steps 11 and 13; `docs/merge-guard-key-protection.md` §4.4, R14, and §5, step E |
| Write collaborator on axiom-encode, not enrolled | Ruleset path: yes, by the SHA pin, if step X1 of the experiment passes. App path: not yet; checklist step 1 must close this before anyone stores the key | Ruleset path: a change that lands on axiom-encode `main` does not reach the guard until someone with ruleset rights moves the pin [docs D3; experiment X1]; no key exists to take. The collaborator can still disable the workflow in axiom-encode, whose effect on the RuleSpec repository is [experiment X17]. App path: can land, without a code-owner review, a change to code the runner job executes (`src/axiom_encode/entrypoint.py`, `src/axiom_encode/__init__.py`, a module the CLI imports, `pyproject.toml`, `uv.lock` or a dependency), or a new workflow whose job selects `merge-guard`. Once someone stores the key, that code can take the private key as well as the token. With the key it can mint tokens for any App installation within that installation's granted repositories and permissions while the key stays valid, independent of one requested token's scope, lifetime or revocation. That permits misleading shadow output and, once the check is required, a forged required check. Checklist step 1 covers all selecting jobs in existing, new, replaced or modified workflows, dynamic selectors, the ability to land them, every pre-token step and persistence, builds, installation, dependencies, interpreter/startup hooks and the full console-script/import closure |
| PR that edits workflow files, pins, ledgers or `.axiom/` configuration | Yes, by refusal | No lane claims it: `fail`. This also blocks legitimate changes (§10, question 0) |
| PR that relabels or hides its manifest, or uses a root the encoder does not name | Yes | The guard stands aside only on the five-file allowlist or on the v5 lane, which needs this PR's green CI (step 3) |
| PR that changes `CLAUDE.md`, `.claude/` or `.gitattributes` | Yes, by refusal | Instruction and settings files steer the agents that hold the enrolled identities; Git attributes change the bytes CI validates. The guard treats both as guarded paths with no lane |
| Hand-written v5 manifest | Through CI | The guard does not verify the signature. The lane's scope keeps the PR from changing the workflow and the pin, and the PR's own `validate / validate` run must be green |
| `validate / validate` from a run that is not this PR's `pull_request` run | Yes | Step 7 |
| Forged or unsigned v6 manifest | Partly | Current-head CI verifies the signature; the guard does not (open question 3) |
| Second workflow imitating the check name | Ruleset path: unverified [experiment X2]; if a same-named check satisfies the rule, the ruleset path fails outright. App path: yes, once the App binding exists | Ruleset path: the rule names the source repository, file and commit [docs D3]. X2 tests imitations by a workflow, by the API and by a commit status. A same-named check that another GitHub App publishes is [inferred, untested]: the lab creates no App (§11). App path: `app_id` binding. It keeps other publishers out, and it cannot tell the guard's own code from unreviewed code that holds the App's key or token (row above) |
| Stale approval after a content change | Yes | Blob equality |
| Push racing an approval | Yes | The verdict belongs to one head, and the guard computes it from that head's tree |
| Dismissal just after a pass | Partly | App path: the check stays green until the next sweep, minutes. Ruleset path: no sweep exists and no event follows a review, so the pull request run stays green [docs D1, D2]. The merge-group run evaluates again, after the group's own `validate / validate` run has completed, and fails [code: `_wait_for_group_ci`, `evaluate_merge_group`]; that needs the merge queue, a precondition of `Active` (checklist steps 8 and 11). A dismissal or a `CHANGES_REQUESTED` review while the group validates (2.5 to 4 hours [author-reported, rulespec-us#2160]) is therefore caught. Not caught: one that arrives after the evaluation's last read of the reviews and before GitHub merges, seconds if the queue merges promptly after the job ends [experiment X15]. The same holds for a required check of the pull request that goes red after its pass. With no merge queue nothing withdraws the green job. The queue is a repository setting that a `rulespec-us` administrator can turn off (row "Repository administrator") |
| Organization owner, a holder of the organization's ruleset permission, or a session with such a token | No | Can set the ruleset to `Disabled`, add a bypass, or move the pin to an unreviewed commit [docs D4, D6, D16]. An organization owner can also rename a targeted branch without restriction [docs D22], which takes `main` out of a rule on `refs/heads/main`. Three accounts are owners [live read 2026-11-17]; whether a custom role holds the permission was not readable. Proposed standing rule: no agent session gets `admin:org`. GitHub cannot enforce it. The ruleset history shows edits for 180 days [docs D6]. Once the v6 writer is on (checklist step 13), weakening the ruleset first is the state that the switch rule forbids: an enrolled dispatcher lands an unreviewed candidate, and any write account lands signed v6 bytes that sit in an open pull request. Every rollback therefore has an order: first the writer off, then no signing run alive and the reader pin reverted, only then the ruleset (the rule at the top of the checklist, with the weaker second form and its gap). That order is a last resort and not a durable rollback: once the ruleset is weak, nothing technical protects `rulespec-us` from signed v6 bytes, because any write account can move the reader pin forward again in the pull request that lands them [inferred]. This checklist cannot verify who was at the keyboard for an owner's act |
| Compromised pinned action or runner image in the guard job | No | Can forge one verdict for one head. The job token is read-only and no key exists [inferred] |
| GitHub API failure | Fails closed for new verdicts | The guard does not withdraw a green check it wrote before the outage |
| Repository administrator of the RuleSpec repository, or a compromised administrator account | No (App path). Partly (ruleset path) | Ruleset path: a `rulespec-us` administrator cannot edit or bypass the organization ruleset [docs D4], and repository rules that the administrator adds can only make the result stricter [docs D4, D5]. **But four repository settings stay in the administrator's hands, and each weakens the guard without touching the ruleset; a fifth, the name of the branch, is documented as closed.** (1) **The merge queue.** It is a repository setting; "Require merge queue" does not exist in an organization ruleset [docs D1]. Turning it off silently reopens the stale-green window of the row above, and nothing detects it except the read-only health check of the checklist; the lock of draft #2289 stops new signing runs only [author-reported, draft #2289]. A bypass actor on the repository's merge-queue ruleset does the same for that actor, and the rule still shows in `rules/branches/main`: the actor merges without the merge-group evaluation. The lock of draft #2289 does not read bypass lists; the check is the `bypass_actors` read of checklist step 8, which needs write access to that ruleset [inferred; docs D7; author-reported, review round 3]. (2) **The default branch.** The ruleset names `refs/heads/main` explicitly, so the rule stays on `main` when the default branch changes; with `~DEFAULT_BRANCH` the rule would follow the new default branch and leave `main` unguarded [docs D3; inferred; experiment X21]. The runner and the evaluator judge `main` only [code: `_enforce_merge_group`]. (3) **Repository-level self-hosted runners.** The job says `runs-on: ubuntu-latest`. Whether a self-hosted runner that an administrator registers with that label can take the job is [inferred, untested]: the lab registers no runner, and X23 records the documentation only. If it can, the administrator forges the job result. `rulespec-us` had no repository-level runner on 2026-11-17 [live read 2026-11-17]. The organization setting that keeps such runners away from `rulespec-us` is therefore a precondition of `Active`, verified by an owner (checklist step 11). (4) **Actions.** What disabling Actions there does is [experiment X8 (c)]. If a head with no guard run can merge while Actions are disabled, an administrator merges without the guard and enables Actions again; nothing blocks it, and the health check reads only the state at its own time [inferred]. (5) **The name of the branch.** A rename of `main` would take the branch out of a rule on `refs/heads/main`. GitHub documents that a repository administrator cannot rename a branch that an organization ruleset targets while the organization setting for it is off, which is the default for an existing organization, and that with the setting on the new name must still be targeted by the same rulesets [docs D22]. An owner reads that setting (checklist step 11); the lab tries the rename [experiment X21]. **Three more acts of an administrator are outside that count.** (6) **Installing a GitHub App.** A repository administrator can install an App that asks for no organization permission and not for "Repository administration", on the repositories that the administrator administers, unless an organization owner restricts it [docs source D26]. Also, a repository administrator can add an App that an organization owner has already installed to the repositories that this administrator administers, whatever the App's permissions are [docs source D26]; so removing an App from `rulespec-us` lasts only with that restriction. Such an App can have the right to write checks, and the App case of the row "Second workflow imitating the check name" is untested. An owner's read of the installed Apps is a precondition of `Active` (checklist step 11). (7) **Renaming the repository** [experiment X10]. (8) **Transferring the repository.** What a transfer out of the organization does to the rule is in no experiment step and was not read in the documentation; a repository outside the organization is outside its rulesets [inferred]. App path: out of scope. Theft of the guard App key through a workflow or code change that a write collaborator can land stays in scope: the axiom-encode row above and checklist step 1 cover it |
| Semantic change brought in by the base while the approved bytes stay the same | Partly | Current-head CI catches it; the human approval does not cover it |
| Commit the approver authored that GitHub does not attribute | Yes, by refusal | An unattributed commit gives `fail` (case 42). The guard does not detect a commit that a forged author email attributes to a third account |

**Agent sessions can act under both enrolled identities.** The guard
authenticates GitHub accounts; it cannot establish which human or agent used
their credentials. If agent sessions hold both
owners' credentials, an agent can author as A, dispatch as A and approve as B,
and every check passes. The guard reduces this to "both accounts must act" and
records who did. It cannot prove that a human read the bytes. To prevent that, maintainers would need approval credentials that agents
cannot hold, for example a hardware-backed signed approval (option c in §3)
or a separate approver account.

## 10. Open questions

| # | Question | Recommendation |
|---|---|---|
| 0 | **First decision.** Which lane do infrastructure and ProgramSpec changes get? At `ca39ebb2`, `INFRASTRUCTURE_LANE_ENABLED = False` leaves them without a lane. The earlier author-reported census classified 48 of 50 PRs here (49 before #2045 merged) | Decide before branch protection requires the check. The off-by-default proposal in §3 keeps content lanes, requires one content-bound approval for named control-plane paths and leaves the stand-aside set empty. Its control-plane and stand-aside lanes cannot admit content. Maintainers still need to choose and review the lane and its coverage; the merge accepts no option. Going live is its own numbered step, checklist step 10: decision, lane change with a review cycle, pin move while the ruleset is still `Evaluate`, a second shadow observation and the inert-path hold at the new pin, and only then `Active` |
| 0b | A module under a jurisdiction that the base lacks has no lane, also with a v5 manifest. How do maintainers add a new jurisdiction? | Design and review a lane separately. The infrastructure proposal classifies the path as content and does not widen the v5 lane |
| 1 | App-published check or an organisation required-workflow ruleset? | Recommendation recorded, Max has not accepted yet: the ruleset, at a pinned SHA (`docs/merge-guard-key-protection.md`), on the condition that the owner's experiment passes. The documentation confirms the rule, its events and the `sha` field [docs D1, D3]; it does not confirm the behaviors that the experiment tests. Fallback: the App with its key in a cloud key service |
| 2 | Is a polling sweep (about 5 minutes) acceptable as the trigger? | Under the ruleset path the question changes: no sweep exists. Each head and each merge group gets one run, and nothing follows a review. Recommendation recorded, Max has not accepted yet: accept that, with a five-hour wait of each run for the repository's CI, a re-run after an approval that arrives when the job has already ended, and a merge queue as a precondition of `Active`; the price is the stale-green window of §9, which the late merge-group evaluation narrows to seconds before the merge (§6). Under the App path: yes for the pilot |
| 3 | Should the guard verify the manifest signature itself, or rely on current-head `validate / validate`? | Rely on CI for v1, since step 4 pins the workflow definitions |
| 4 | Rename `reviewed_rulespec_ref` to `candidate_rulespec_ref` in v6? | Yes: a v6 manifest must not say "reviewed" |
| 5 | Also set `required_approving_review_count: 1`? | No: it counts any write collaborator and stale approvals |
| 6 | Do notary lanes need this owner-review gate? | Decide with the notary activation |
| 7 | Exclude the approver when they are a commit author but not the PR author? | Yes, best-effort (case 27) |
| 8 | Merge-queue group size for promotions? | 1 during the pilot |
| 9 | A fork PR, a stale PR and a PR with more than 256 changed manifests get `fail`, not `not_applicable`. Is that acceptable for a required check? | Yes for forks and stale heads (`strict` already requires an up-to-date head). Decide the bound before a bulk manifest migration |
| 10 | Should a `CHANGES_REQUESTED` from an outside account block? | No (step 7); it differs from today's admission rule |
| 11 | The guard binds `candidate.head_sha` to the PR's commits, so a rebase needs a new signing run. Acceptable? | Yes: the binding is worth more than rebase support |
| 12 | The v5 lane is `not_applicable` on scope plus this PR's green CI, with no approval check. Enough? | Yes while admission checks the approval before signing |
| 13 | A signed v5 PR that also changes a receipt, pin, ledger or test (rulespec-us#1470) has no lane. What lane preserves the signed operation? | Keep the combined shape refused; design and review a separate lane. The infrastructure proposal explicitly refuses content with a legacy-replacement receipt as `signed_backfill_has_no_lane` when on. Splitting the operation remains unverified |
| 14 | Add `run_attempt` to `attestation_scope`? | Yes, in the change that defines the v6 writer. Its place in the order is checklist step 0: if accepted, its change lands before the reviewed pin of step 5 and the reader pin of step 9; if declined, nothing changes |
| 15 | The approver may not be the author or committer of any commit, so the approver cannot click "Update branch". Acceptable? | Yes for the pilot |
| 16 | Which mechanism closes the key-exposure set of checklist step 1: (a), (b), (c) or (g')? With (a) or (b), which complement stops another workflow from selecting `merge-guard`? | Recommendation recorded, Max has not accepted yet: (g'), which creates no key and so needs no complement; first fallback (b)+(d); second fallback (b) with code-owner review of `.github/` and last-push approval. Max chooses. Key storage and every activation step, shadow mode included, wait on his choice, implementation and fresh independent evidence over the whole set |

## 11. What remains unverified

### The ruleset experiment (not run)

`docs/merge-guard-ruleset-experiment.md` holds the procedure. An
organization owner runs it; an agent copies the filled record table here in
a reviewed pull request. Until then every row is open, and the
recommendation of checklist step 1 is conditional. Checklist step 3 defines
which results make the experiment "passed".

| Step | Question | Result |
|---|---|---|
| X1 | The SHA pin runs exactly that file | not run |
| X2 | A same-named check does not satisfy the rule: (a) another workflow file, (b) a file at the ruleset workflow's own path in the target, (c) an imitation on `merge_group`, (d) a check run created through the API, (e) a commit status with the same context | not run |
| X3 | A re-run after an approval updates the rule | not run |
| X4 | The RuleSpec repository's job token reads public axiom-encode | not run |
| X5 | The merge queue runs the workflow, from the pinned commit; the `merge_group` payload | not run |
| X6 | `Evaluate` mode runs the workflow and does not block | not run |
| X7 | A non-owner can read the active rule | not run |
| X8 | A head with no run stays blocked | not run |
| X9 | An empty bypass list binds an owner | not run |
| X10 | Renaming the target keeps the rule | not run |
| X11 | `github.workflow_sha` and `github.workflow_ref` in a ruleset run | not run |
| X12 | The check name, and the commit the result attaches to | not run |
| X13 | A re-run of an old run cannot make a new head green | not run |
| X14 | The branch-rules response for a `workflows` rule | not run |
| X15 | A dismissal and the merge queue; the group's check on the group commit; the time from the guard job's end to the merge | not run |
| X16 | Who can re-run a ruleset run | not run |
| X17 | A workflow disabled in the source repository | not run |
| X18 | `GITHUB_REPOSITORY`, `github.repository` and the event's repository fields name the TARGET repository | not run |
| X19 | A pin move with an open green pull request: does the old result still satisfy the rule, and which file does the next `merge_group` run use | not run |
| X20 | `Evaluate` together with a merge queue: does the workflow start on `merge_group`, and does the queue wait for it | not run |
| X21 | A change of the target's default branch, and a rename of `main` by a repository administrator, with the rule on `refs/heads/main` | not run |
| X22 | Two conclusions on one head (close and reopen after a green run) | not run |
| X23 | A repository-level self-hosted runner with the label `ubuntu-latest`: the documentation only; the lab registers no runner | not run |
| X24 | A pull request whose base is changed to `main` gets no run and stays blocked | not run |

Also not recorded yet: the ruleset id, the pinned SHA and its
`BUNDLE_SHA256` (checklist steps 5 and 6), Max's decisions (step 0), and the
statement that continuous landing is on (step 15). Also not recorded: the
trial pull request of step 8, the duration of `validate / validate` on a
merge group, the second pin of step 10, the bypass list of the merge-queue
ruleset (steps 8 and 11), and the owner's three reads of step 11 (the
"Runners" setting, the runner groups, the branch-rename setting).

Not tested by any step of the experiment, and so [inferred, untested] after
it too:

- a check with the guard's name that another GitHub App publishes. The lab
  creates no App. A test needs a throwaway GitHub App with the right to
  write checks, installed on the lab target, that creates a successful
  check run named `lab-guard` on an unapproved pull request; that means
  creating an App and a key, which the experiment excludes;
- a repository-level self-hosted runner that takes the guard job (X23
  records the documentation only);
- a bypass actor on the repository's merge-queue ruleset (checklist step 8).

### Other open points

A read-only live run, `merge-guard --repository rulespec-us --pr 2045`, returned
`not_applicable` / `earlier_lane_v5` on 2026-11-12, after that PR's CI went
green. It is a v5 promotion PR: one root-layout manifest and two module files.
The run exercised the real check-run listing and the real `pull_request` run
binding. The green head carried exactly one check run named
`validate / validate`. After #2045 merged, the same command returns `fail` /
`pr_identity`: the evaluator judges open PRs only.

Live data on 2026-11-12 (rulespec-us#2045 and its signing run) confirmed
the fields of an Actions run (`event`, `path`, `head_sha`, `head_branch`,
`repository`, `head_repository`, `pull_requests`, `actor`,
`triggering_actor`), the `details_url` form of a check run, the v5 promotion
manifest's `applied_files` paths, the two manifest directory layouts on
`rulespec-us` main, and the attribution of the signing workflow's manifest
commit.

The retained evidence does not establish:

- Whether an Actions run keeps entries in `pull_requests` after the PR's
  branch moves or someone reruns the run. The observed fork runs had no
  entries; the guard refuses fork PRs.
- Whether branch protection honours a required check with a non-Actions
  `app_id` on merge-group commits, and how to list a group's PRs. This
  matters only for the App path.
- Which workflow file version `pull_request_review` and `merge_group` runs
  use. For a ruleset workflow: [experiment X1] and [experiment X5].
- The behaviors of an organization required-workflow ruleset that the
  experiment tests (next section). The organisation's plan is no longer
  open: it is `enterprise` [live read 2026-11-17], and GitHub documents the
  rule at the organization level [docs D1].
- Whether the Go signer or supervisor inspects `schema_version` or field names.
  Their tests mention `applied-rulespec/v5`; the retained review did not
  inspect that code.
- Whether workflow-run metadata for `manifest.run_id` stays readable for the
  life of a PR. A deleted run makes step 6 fail closed.
- The shapes of v5 retire, path-migration and legacy manifests, and so whether
  those lanes get `not_applicable` or `fail`.
- File modes and CI state of the PRs in the census.
- The content layout of `rulespec-<country>-<subdivision>` repositories. The
  evaluator assumes the content roots sit at the root or one level down.
- Whether `pr.base.sha` follows the `main` tip. The evaluator does not use it.
- Future tree capacity and how GitHub measures the size limit. A recursive
  tree that GitHub truncates gives `fail` for every PR of that repository.
  GitHub's [Get a tree](https://docs.github.com/en/rest/git/trees#get-a-tree)
  (read 2026-11-17) states: "The limit for the tree array is 100,000 entries
  with a maximum size of 7 MB when using the recursive parameter." One
  read-only request on 2026-11-17 at 11:57 EST,
  `GET /repos/TheAxiomFoundation/rulespec-us/git/trees/main?recursive=1` at
  commit `ddad6d63df99a86a2dbeb60dd910f28eba1037d7`, returned
  `truncated: false`, 18,398 entries (15,277 blobs and 3,121 trees) and a
  response body of 5,050,452 bytes, about 275 bytes per entry. The entry
  count uses 18.4% of the documented limit; 81,602 entries remain. If the
  7 MB limit applies to the response body and means 7,000,000 bytes, the body
  uses 72.1% of it and 1,949,548 bytes remain: room for about 7,100 more
  entries of the measured mean size, 39% growth. The documentation does not say which bytes
  the limit counts or whether MB means 10^6 or 2^20 bytes **[check live]**,
  so the size margin is an estimate; the size limit, not the entry limit,
  binds first. A head tree of a pull request is about the same size. This is
  one measurement; it sets no growth rate. Before branch protection requires
  the check, repeat the measurement, and before the body nears 7 MB change
  the evaluator to read sub-trees (GitHub's documented fallback) in a
  reviewed change (#2257 R1-P3-8).
- The guard App. On 2026-11-15, read-only API reads found no guard App among
  the organisation's installations, no `merge-guard` environment and no
  `MERGE_GUARD_*` variable or secret. The installation read ran at 12:13Z. These dated responses do not
  establish a current inventory of Apps, installations, settings or custody. The intended
  guard App's id, installation and key custody remain unverified.
- The full key-exposure set of checklist step 1. On 2026-11-15, a trace of
  `axiom-encode merge-guard-run` through the console script imported 150
  encoder modules, 8 of them under CODEOWNERS. That run stopped for lack of a
  token before any network read, and the trace leaves out build and dependency hooks,
  installation, interpreter start-up, earlier steps and persistence, and
  other existing or future workflows that can select `merge-guard`. Neither
  the 150-module trace nor its 8 owned modules proves closure of that set.
