# Content-bound merge guard

Status: **design; maintainer acceptance and activation remain pending.** At
this branch's source, which merges `main` at
`fad29b3ed4019fe7042552a7958542dfcb41e4e0`, the evaluator
(`src/axiom_encode/content_merge_guard.py`) supplies verdicts, the production
manifest writer emits v5, and the runner offers dry-run and shadow modes only.
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
  change; under the ruleset path only the merge-group run does (§6).
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
    administrator cannot. The token that agent sessions use on the
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
read 2026-11-17). The
retained evidence does not establish actual same-name organization or
environment variables, or how a rerun resolves a changed variable
**[check live]**.
[GitHub variables context](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#vars-context).

The Python CLI accepts `MERGE_GUARD_TOKEN` from its process environment and
checks its shape. It does not check the workflow ref, switch, App identity or
credential provenance. The intended workflow supplies an App installation
token; calling the CLI elsewhere does not inherit the workflow gate.

| Item | Value |
|---|---|
| Trigger | `workflow_dispatch` only (inputs `country`, `pr`, `mode`). No `schedule`: the sweep on a timer is a later reviewed change to the workflow file |
| Switch | The gate job requires `vars.MERGE_GUARD_RUNNER_ENABLED` to equal `true`, and `github.ref == refs/heads/main`. A rerun of `run` alone can reuse an earlier successful gate, so the `run` job repeats both checks in its first step, before any step names a credential (#2257 R2-P3-1). The repeat evaluates `vars.MERGE_GUARD_RUNNER_ENABLED` in the `run` job; no retained evidence shows that a rerun resolves the variable's current value **[check live]**, so the repeat alone is not the rollback. GitHub documents that a disabled workflow cannot rerun; change 3's rollback still unsets the switch and disables the workflow. Neither check governs another workflow that selects `merge-guard` (checklist step 1 of §8). [Rerun documentation](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs) |
| Environment | The workflow names `merge-guard`; its planned deployment branch rule allows `main` only. The gate job has no environment, so a refused dispatch does not create one |
| App client ID | Intended environment variable `MERGE_GUARD_APP_CLIENT_ID`; the workflow reads the `vars` context |
| App private key | Intended environment secret `MERGE_GUARD_APP_KEY`; the workflow reads the `secrets` context |
| Token | The `run` job mints one per run, for `rulespec-<country>` only: `checks: read` in `dry-run`, `checks: write` in `shadow`; `pull_requests`, `contents`, `actions`: `read`. The CLI reads it from `MERGE_GUARD_TOKEN` and has no token argument |
| `dry-run` (default) | Prints one JSON report. Writes nothing |
| `shadow` | Also creates one check run per judged head, named `content-bound-merge-guard (shadow)`, conclusion always `neutral`, verdict in the title and summary. It skips the write only when the guard App's own newest shadow check on that head (the runner matches the App by its client ID, `MERGE_GUARD_APP_CLIENT_ID`) already carries the same verdict. A check with the shadow name from another publisher never counts, so a pull request workflow cannot suppress its verdict |
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
  empty sweep. A verdict whose reason is a failed read is `fail` and also a run
  error. The run's exit code reports execution errors; the report carries verdicts.
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
  (read 2026-11-17) documents `check_name`, `app_id`, `filter` with default
  `latest` ("returns the most recent check runs"), `per_page` with a maximum
  of 100, `page`, and a response of `total_count` and `check_runs`. It also
  limits the listing to the 1,000 most recent check suites of a reference. It
  does not say how `latest` groups runs, or that `total_count` counts the runs
  after that filter **[check live]**. The runner therefore assumes that more
  than 100 shadow-named checks on one head can be listed, and reads up to 10
  pages of 100 (#2257 R2-P3-8). Past 1,000 listed checks, or when the pages do
  not add up to one stable `total_count`, it writes no check for that head and
  the run exits 1; it never treats a partial listing as "already reported".
  Any workflow of the RuleSpec repository with `checks: write` can create
  checks with the shadow name, so if `latest` does not collapse them
  **[check live]**, more than 1,000 on one head would withhold that head's
  shadow check until someone changes the bound. The runner holds the App's
  client ID, not its numeric `app_id`; whether the `app_id` filter would bound
  the listing to the guard App's own checks also needs a live App
  **[check live]**.
- **Known limit of the sweep, acceptable for shadow only.** The runner reads
  the list by page number. A pull request that opens or closes during the read shifts
  the pages, so one sweep can miss a pull request or see one twice (twice is
  a run error). A missed pull request has no shadow check until the next
  sweep. An enforcing check would leave that pull request without its
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
`"inert": "source_repository"`, reads neither the event nor the network and
exits 0.

| Item | Value |
|---|---|
| Workflow | `.github/workflows/content-bound-merge-guard.yml`, `name: content-bound-merge-guard`, one job `guard` named `content-bound-merge-guard` (the check name) |
| Triggers | `pull_request` and `merge_group`, no filter and no activity types. No `workflow_dispatch`, `workflow_call`, `schedule` or `pull_request_target` |
| Permissions | `contents: read`, `pull-requests: read`, `checks: read`, `actions: read`. The job token is read-only; the job has no secret, variable, environment or `id-token` |
| Skip paths | None: no `if`, no `continue-on-error`, no `concurrency`. A skipped job would count as passed |
| Code it runs | The stdlib guard bundle: `scripts/merge_guard_launch.py` and `merge_guard_runner.py`, `content_merge_guard.py`, `reviewed_pr_admission.py`, `applied_manifest_v6.py`, `rulespec_citation_paths.py`, `statute.py`. A sparse checkout of exactly these files at `github.workflow_sha`, copied next to an empty `axiom_encode/__init__.py`. No `uv`, `pip`, cache, console script or encoder package |
| Bundle digest | The workflow carries `BUNDLE_SHA256` as a literal and refuses a bundle with another digest. The digest is the sha256 of the lines `<sha256 of the file>  <path>` for `launch.py`, `axiom_encode/__init__.py` and the six modules in list order: the output of `sha256sum <those paths> \| sha256sum` in the bundle directory. A test recomputes it from the working tree, so a change to any bundle file fails CI until the same change updates the literal in `.github/` |
| Launcher | `python3 -I -S launch.py`. It refuses to run without `-I` and `-S`, in a directory that holds anything but the bundle, when an import asks for an `axiom_encode` module off the list, or when such a module was loaded from another directory. It appends the bundle directory to `sys.path`, after the standard library. It takes no argument |
| Command | `merge_guard_runner.enforce_command` (also `axiom-encode merge-guard-enforce`). Inputs come from the runner environment only: `GITHUB_REPOSITORY`, `GITHUB_EVENT_NAME`, `GITHUB_EVENT_PATH`, and the job token in `MERGE_GUARD_TOKEN`. No argument selects a pull request, a repository or a mode |
| Result | The exit code, which is the job's conclusion. 0 only when the verdict and its reason are in `SUCCESS_VERDICTS` (`pass`: `content_bound_approval`, `control_plane_content_bound_approval`; `not_applicable`: `no_guarded_content`, `earlier_lane_v5`, `infrastructure_as_today`), the verdict is bound to the event's head, and the run has no error. 1 for every `fail`, every read error, a malformed event, another event, another repository and an absent token. The runner creates no check run, so no `neutral` result exists; it sends `GET` requests only |

**`pull_request` event.** The number is `pull_request.number` and must equal
the payload's `number`; the event head is `pull_request.head.sha`. The runner
calls `evaluate`. A verdict whose evidence names another head than the event
head becomes `fail` (`event_head_is_stale`): the run's check attaches to the
event head, so a verdict for a head that moved, or a re-run of the run of an
older head, never satisfies it.

**Waiting for `validate / validate`.** The guard run starts together with
the required check, and nothing starts it again when that check finishes. On
`fail` with reason `ci_not_green` the runner polls the check runs named
`validate / validate` on the event head every 30 seconds
(`CI_POLL_SECONDS`), until at least one exists and all are completed, for at
most 3,600 seconds (`CI_WAIT_SECONDS`) and 121 polls. It then evaluates once
more, and that verdict is final. The wait only delays: a settled check is not
a verdict, and a poll that cannot be read whole leaves `fail` and makes the
run an error. Every other verdict is final at once. In the evaluator's step
order an unapproved v6 unit fails at step 5 (`no_approval`) before step 7
reads CI, so it does not wait; a test pins that it makes no check-run read.
The job's `timeout-minutes` is 75.

**`merge_group` event.** See §6.

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
| Merge group of a v6 unit, `pass` | 29 |

A run makes at most two evaluations and 121 polls. `MAX_ENFORCE_REQUESTS` is
250: past it every read fails and the verdict is `fail`. Two evaluations of
the largest measured size and every poll are 2 x 29 + 121 = 179 requests. GitHub documents the job token's limit as
"1,000 requests per hour per repository", and 15,000 for resources of a
GitHub Enterprise Cloud account
([Rate limits for the REST API](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api),
read 2026-11-17). Consequences that no load test has measured:

- One run that waits a full hour uses up to 121 polls and two evaluations,
  about 170 requests. Six such runs in one hour exceed the 1,000 limit,
  which the repository's other workflows share. A rate-limited read is a failed read: `fail`, and someone re-runs
  the job.
- A real pull request with more than 100 files, commits or reviews needs
  further page reads; the fixtures have one page of each.
- A v5 lane of `u` units costs about `11 + 3u` requests per evaluation. With
  a full wait the cap admits about 17 units; with a 15-minute wait about 32.
  A larger train fails closed. Raise the cap in a reviewed change (which
  changes the bundle digest) before a bulk landing.

**Limits of the trigger set.**

- No trigger follows a review [docs D1, D2]. A v6 unit is unapproved when
  its signed head is pushed, so its first run fails; after the approval
  someone must re-run the failed job. Whether a re-run updates the ruleset's
  result is [experiment X3]. If it does not, the next options are to close
  and reopen the pull request, or a base-only update by someone other than
  the approver [inferred].
- A dismissal after a pass leaves the run green. Only the merge-group run
  evaluates again (§6). Without a merge queue the dismissal never blocks the
  merge. A merge queue on the RuleSpec repository is therefore a
  precondition of making the ruleset `Active` (checklist steps 8 and 10).
- The merge-group run evaluates once, when the group is created, and does
  not wait [code: `_enforce_merge_group`]. A dismissal after that run and
  before the merge is not seen. rulespec-us#2160 reports 2.5 to 4 hours of
  validation for each group [author-reported], so that window is hours
  long. The App design's sweep would have bounded it to minutes
  (§9, "Dismissal just after a pass").
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
- that another workflow or App cannot satisfy the rule with a check of the
  same name [experiment X2]. A tripwire test keeps every other axiom-encode
  workflow from using the name; it cannot see workflows of a RuleSpec
  repository;
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
- what `github.workflow_sha` is for a ruleset workflow [experiment X11]. The
  digest literal makes a wrong value fail closed: a checkout of other bytes
  cannot match it;
- that `GITHUB_REPOSITORY` and the event file describe the target repository
  in a ruleset run [experiment X11] [experiment X12];
- that a re-run of the run of an older head cannot satisfy a newer head
  [experiment X13];
- that `Evaluate` mode starts the workflow [experiment X6], that an empty
  bypass list binds an owner [experiment X9], and that a head with no run
  stays blocked [experiment X8];
- what the branch-rules endpoint returns for the rule
  [experiment X14], and what a dismissal does to a queued pull request
  [experiment X15];
- the behaviour on rate limiting, and the real duration of
  `validate / validate` against the 60-minute wait. The experiment does not
  test these; the shadow period does (checklist step 7).

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
- **Algorithm.**
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
- **Group size 1.** A group of two pull requests fails: the second entry is
  built on the first group's commit, not on `main` (step 2), and a group
  commit that holds two units has another tree than either head (step 3,
  case 24). rulespec-us#2160 plans a group size of 1; keep it for the pilot.
- **The head must be up to date.** `evaluate` requires `H` to be strictly
  ahead of the live `main` tip. After another pull request merges, a queued
  pull request that was not updated fails with `base_not_current_main`.
- **No wait.** The merge-group run does not wait for `validate / validate`;
  the pull request's own run must already be green on `H`.
- **Why the queue is a precondition.** Under the ruleset path no event
  follows a review. The merge-group run is the only evaluation between a
  green pull request run and the merge, and it reads the reviews again
  [code: `evaluate_merge_group` calls `evaluate`]. A repository with no
  merge queue has no such evaluation: a dismissed approval would not block.
  The activation checklist therefore turns the queue on before the ruleset
  becomes `Active` (steps 8 and 10 of §8).
- **What the queue does not close.** The run evaluates when the group is
  created. A review that changes after that run, while the group's other
  required check still runs, is not read again (§4, "Limits of the trigger
  set"). Closing that window needs a second evaluation at the end of the
  group's checks. That is not designed.
- **One entry at a time.** Step 2 requires the group's base to be the live
  `main` tip. With a build concurrency above 1 the second entry is built on
  the first and fails [code: `R_MERGE_GROUP_BASE`]. rulespec-us#2160 also
  asks for a build concurrency of 1 [author-reported].
- **A group commit equal to the head.** If a queue ever used the pull
  request's own head commit as the group commit, the group's
  `validate / validate` run would attach a `merge_group` check to `H`, and
  the evaluator's CI rule would refuse it, because every such check on `H`
  must come from a `pull_request` run [inferred from `_require_green_ci`].
  That fails closed. The experiment records whether the two commits differ
  [experiment X5].
- **Unverified.** The payload fields and the `head_ref` form above, which
  workflow file version a `merge_group` run of a ruleset uses, and whether
  the result of this job satisfies the queue [experiment X5]. The 40 hex
  digits of `head_ref` are not compared with anything.
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
| 24 | Merge group includes two units and lacks an approval for one | fail for the group | yes: the tree differs, or the second entry's base is not `main` |
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

Cases 49-60, the enforcing runner (`tests/test_merge_guard_runner.py`,
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
| 60 | A verdict with a success reason that the runner does not list, or `neutral` | job fails | yes |

Cases 15-18 must also run live in the shadow period (§8).

## 8. Delivery plan

Implementation comes before activation.

| # | Repo | Change | Rollback |
|---|---|---|---|
| 1 | axiom-encode | Evaluator, `merge-guard` CLI, tests, CODEOWNERS and this document. The CLI and runner call the evaluator; this step creates no enforcing check | revert |
| 2 | axiom-encode | Verifiers can read `applied-rulespec/v6` with the attestation scope, behind `V6_READER_ENABLED = False`; no writer | revert; the production writer emits only v5 |
| 2b | axiom-encode | Set `V6_READER_ENABLED = True`. May ride with change 3. Switch rule: The reader may be switched on while no v6 writer exists: nothing new can land. The v6 writer and the admission without approval must stay off until the guard check is required. With the reader on, the writer on and the guard check not required, an enrolled dispatcher lands their own unreviewed candidate with no approval, and any account with write access re-lands signed v6 bytes from a closed pull request. | set it back; the production writer emits only v5 |
| 3 | axiom-encode | Workflow that runs the guard from `main` and publishes its check, with a dry-run mode. Shadow only and inert: manual dispatch, behind `MERGE_GUARD_RUNNER_ENABLED` (§4, "Runner as built"). It does not flip `V6_READER_ENABLED`; change 2b stays a separate one-line commit to `applied_manifest_v6.py` | unset the variable; disable the workflow |
| 3b | axiom-encode | **Enforcing runner (built, required nowhere).** `merge_guard_runner.enforce` judges the pull request or merge group of one Actions event and ends its job, named `content-bound-merge-guard`, with success for `pass` and `not_applicable` and with failure for `fail` and for every error. Its workflow has the `pull_request` and `merge_group` trigger, so a ruleset that requires it runs it on every pull request head and every merge group; each run judges its own head, so no work list or snapshot is read. No trigger follows a review change: after an approval someone re-runs the job [experiment X3]. Request volume is measured (§4, "Enforcing runner as built"). It is switched on only by an organization ruleset "Require workflows to pass before merging" that names the workflow at a commit; no variable, key or repository setting does it. Hard precondition of checklist steps 6 and 10 | an organization owner sets the ruleset to `Evaluate` or `Disabled` |
| 4 | rulespec-us | Pin the encoder that verifies v6. Not started: the commit to pin exists only after change 2b merges (checklist step 9) | revert the pin |
| 5 | axiom-encode | Emit v6 and admit without approval, behind a default-off switch. Built as drafts #2238 and #2269 (5b, the hosted evidence step); their locks 2 and 3 expect a guard App check in classic branch protection [author-reported] | revert |
| 5c | axiom-encode | **Writer lock for the ruleset form — not built.** Locks 2 and 3 of change 5 must expect the organization ruleset's workflow entry (repository id, path, pinned SHA) in the branch-rules response, in place of a guard App id. The response shape is unknown [experiment X14]. Needed only if Max accepts the ruleset path | revert; the locks stay shut |
| 6 | axiom-encode | **Activation**: record the expected guard identity in the writer's locks, turn the writer switch on, update `docs/reviewed-pr-promotion.md`. One reviewed change. Not opened as a pull request before checklist step 12, because merging it takes effect | flip the switch back; the ruleset stays `Active` |

### Activation checklist

This is the complete ordered list from the state of this change to
"continuous landing is on". It follows the recommendation of
`docs/merge-guard-key-protection.md`. **Max has not accepted that
recommendation; until step 0 is done, every later step waits.** If Max
rejects it, or the experiment of step 3 fails, steps 5-8, 10 and 12 change
to the fallback of that document, and this checklist needs a reviewed
rewrite first.

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

0. **M** (Max) Decide, by a comment on the pull request that carries this
   recommendation:
   - the mechanism of step 1: accept or reject;
   - question 0 of §10 (the lane for infrastructure pull requests). Step 10
     cannot happen without it;
   - D1, D2a-D2f, D3 and Seq. of
     `docs/reviewed-pr-promotion-order-proposal.md`, which are still
     agent-proposed, with the author-dispatch policy of D3;
   - question 14 of §10 (`run_attempt` in the v6 manifest).
   - Expected: one comment per decision. Check: read the comments. Rollback:
     a later comment withdraws a decision; nothing has taken effect.

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

   Independent of the guard: do the "Now" steps A, B and D of
   `docs/merge-guard-key-protection.md` §5. They protect two keys that exist
   today. On 2026-11-17 all three were open [live read 2026-11-17]. Check:
   the three commands of that section. Rollback: none needed.
2. **M** Merge the changes in this order. Maintainers merge; agents may not.
   Each pull request first needs its review cycle, green CI and a maintainer
   to mark it ready.
   - axiom-encode #2237 (change 3), then #2286 (change 3b), then the pull
     request that carries this recommendation.
   - axiom-encode #2238 (change 5), then #2269 (5b), then the lock rework
     (5c, not built). These three can merge at any time: all their locks are
     off. #2237 and #2238 both edit this document; the second to merge needs
     `main` merged into it first.
   - TheAxiomFoundation/.github#155, then a pin bump of the shared validation
     workflow in `rulespec-us`, then rulespec-us#2160 (the `merge_group`
     trigger of `validate / validate`). On 2026-11-17 both were open drafts,
     mergeable, with green checks from 2026-11-10
     [live read 2026-11-17]. rulespec-us#2160 changes a file under
     `.github/`: merge it before step 10, or it needs the lane of question 0.
   - Switch rule: The reader may be switched on while no v6 writer exists: nothing new can land. The v6 writer and the admission without approval must stay off until the guard check is required. With the reader on, the writer on and the guard check not required, an enrolled dispatcher lands their own unreviewed candidate with no approval, and any account with write access re-lands signed v6 bytes from a closed pull request.
   - Expected: no verdict and no admission changes, because every switch is
     off. Check, for each constant, on `main`:
     `git grep -n -E '^(V6_READER_ENABLED|V6_WRITER_ENABLED|INFRASTRUCTURE_LANE_ENABLED|GUARD_APP_ID)[: ]' origin/main -- src/axiom_encode`
     prints four lines, with `False`, `False`, `False` and `None`. Rollback: revert the merge.
3. **M** (any organization owner) Run the experiment of
   `docs/merge-guard-ruleset-experiment.md`. It needs two throwaway
   repositories and one organization ruleset, and no secret, App or key. An
   agent must not run it.
   - Expected: every step X1-X17 has a recorded result. AGENT then prepares
     a pull request that copies the record into §11.
   - Check: the record table of that document, filled in. Rollback: its
     clean-up section.
   - If X1, X2 or X8 fails, stop: the recommendation fails, and Max decides
     on the fallback.
4. AGENT, only if the results ask for it: a pull request with the code
   change (for example other payload fields, another source for the
   workflow commit, the fixture for lock 3), through a review cycle. **M**
   merges.
5. AGENT and **M** Review the guard at the candidate commit, as §6 of
   `docs/merge-guard-key-protection.md` lists: an independent agent review
   and a maintainer's own reading of the workflow file and the eight bundle
   files. The candidate is a commit on axiom-encode `main` that contains
   steps 2 and 4.
   - Expected: no open finding; the commit SHA and its `BUNDLE_SHA256`
     recorded in §11.
   - Check:
     `gh api "repos/TheAxiomFoundation/axiom-encode/contents/.github/workflows/content-bound-merge-guard.yml?ref=<sha>" --jq .content | base64 -d | grep -n BUNDLE_SHA256`,
     and `gh api repos/TheAxiomFoundation/axiom-encode/compare/<sha>...main --jq .status`
     prints `ahead` or `identical` (the commit is on `main`).
6. **M** (an organization owner) Create the organization ruleset in
   `Evaluate`, with the values of `docs/merge-guard-key-protection.md` §5:
   target `rulespec-us`, default branch, the workflow
   `.github/workflows/content-bound-merge-guard.yml` of axiom-encode at the
   reviewed SHA, empty bypass list. This starts the shadow period.
   - Expected: the next push, update or reopen of a `rulespec-us` pull
     request starts a `content-bound-merge-guard` run there. Pull requests
     that were already open get no run until then [docs D2]. Nothing is
     blocked.
   - Check (owner): `gh api orgs/TheAxiomFoundation/rulesets/<id> --jq '{enforcement, bypass_actors, rules}'`.
     Rule Insights needs ruleset rights [docs D5, D6]. The runs are in the
     Actions tab of `rulespec-us` for anyone [inferred].
   - Rollback: set the enforcement to `Disabled`.
7. AGENT and **M** Shadow period.
   - Run cases 15-18 of §7 live on a throwaway `rulespec-us` pull request
     and confirm every result. Case 15 under the ruleset is a workflow in
     the pull request with the guard's job name: the ruleset's own run must
     stay failed.
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
   - Also measure here what the experiment cannot: the real duration of
     `validate / validate` against the 60-minute wait, and rate limiting.
8. **M** (a `rulespec-us` administrator) Turn on the merge queue for
   `rulespec-us` `main`, with the settings that rulespec-us#2160 lists:
   only merge non-failing pull requests, group size 1, build concurrency 1,
   a status check timeout above 4 hours, merge method squash. The guard
   also needs build concurrency 1 (§6).
   - Expected: a queued trial pull request starts a `merge_group` run of
     `validate / validate` and of the guard. Repeat steps X5 and X15 of the
     experiment here, in `Evaluate` mode.
   - Check: `gh run list --repo TheAxiomFoundation/rulespec-us --event merge_group --limit 5`.
   - Rollback: turn the queue off.
   - Cost: every landing then runs the full validation matrix on its merge
     group, 2.5 to 4 hours, about 6 to 9 landings a day
     [author-reported, rulespec-us#2160].
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
     encoder at that commit. **M** merges it.
   - The switch rule of step 2 governs: the writer stays off.
   - Check: `git grep -n "^V6_READER_ENABLED" origin/main -- src/axiom_encode`
     prints `True`; the `rulespec-us` pin names the commit.
   - Rollback: set the constant back; revert the pin.
10. **M** (an organization owner) Switch the ruleset to `Active`. Hard
    preconditions, all of them: steps 0-9 are done; change 3b is on `main`
    and is the code at the pinned SHA; the experiment passed; the merge
    queue of step 8 is on; and Max has decided question 0 of §10 **and its
    lane is on at the pinned SHA** (that needs a commit with
    `INFRASTRUCTURE_LANE_ENABLED = True`, a new review as in step 5, and a
    pin move). The earlier author-reported census classified 48 of 50 pull
    requests as refused without a lane.
    **WARNING:** without the lane of question 0, an `Active` ruleset blocks
    every pin, ledger, workflow and tooling pull request on `rulespec-us`,
    administrators included, and the rollback itself needs an organization
    owner. Under the App fallback the same step adds a required check:
    require the name `content-bound-merge-guard` only, and never
    `content-bound-merge-guard (shadow)`, which is always neutral; a neutral
    check satisfies a required check, so it would enforce no verdict.
    - Expected: a `rulespec-us` pull request merges only when its guard run
      is green. Pull requests whose run passed in `Evaluate` need no new run
      [docs D1].
    - Check, as anyone:
      `gh api repos/TheAxiomFoundation/rulespec-us/rules/branches/main --jq '.[] | select(.type == "workflows")'`
      shows the rule (the call lists active rules only [docs D7]).
    - Rollback: set the enforcement back to `Evaluate`.
    - Optional hardening, **M**: enforce code-owner review for `/.github/`
      in `rulespec-us` (§9). That protects the RuleSpec workflows; it is not
      part of the guard.
11. Run the current-order canary to completion: one promotion in today's
    order (`docs/reviewed-pr-promotion.md`) through the `Active` guard.
    - Expected: the signed candidate is a v5 lane, so its guard run waits
      for `validate / validate` and ends `not_applicable`, `earlier_lane_v5`
      (job success). The unsigned candidate before signing is `fail`,
      `no_lane_for_guarded_change`; that is expected and blocks nothing
      that could merge today.
    - Check: the report line of the guard run on the signed head.
    - Rollback: step 10's rollback.
12. Lock rework and change 6, as one reviewed pull request. AGENT prepares
    it; it is not opened before steps 0-11 are done, because merging it
    takes effect. It records the expected ruleset workflow identity
    (repository id, path, the pinned SHA) in the writer's lock constants,
    sets `V6_WRITER_ENABLED = True`, and updates
    `docs/reviewed-pr-promotion.md`. The pinned SHA is an earlier commit
    than the one that records it. `reviewed_pr_admission.py` is a bundle
    file, so the pull request also sets the new `BUNDLE_SHA256`. **M**
    merges.
    - Check: the writer's own refusal tests pass with the recorded identity;
      `git grep -n "^V6_WRITER_ENABLED" origin/main -- src/axiom_encode`
      prints `True`.
    - Rollback: set the switch back. The ruleset stays `Active`.
13. The smallest live canary of the new order, on one real candidate (one of
    the Arizona candidates of step 7). At each arrow, record the URL and the
    guard's report line.
    1. AGENT or contributor: local encode → candidate pull request. Guard
       run: `fail`, `no_lane_for_guarded_change`.
    2. **M** (an enrolled dispatcher; it may be the candidate's author)
       dispatches `targeted-signed-reencode.yml`. The `production-signing`
       environment approval still applies to the run.
    3. The run opens a draft pull request with the signed v6 manifest into
       the candidate branch. Someone who will **not** be the approver marks
       it ready and squash-merges it there. That push must come from a person's or an
       App's credential, not a job token (§4, "Limits of the trigger set").
    4. CI on the candidate pull request goes green. Guard run on that head:
       `fail`, `no_approval`, with no wait.
    5. **M** (the other enrolled owner) reads the green pull request and
       approves it on GitHub.
    6. Someone with write access re-runs the failed guard job (or the
       procedure that step X3 found). Guard run: `pass`,
       `content_bound_approval`.
    7. If `main` moved, someone other than the approver updates the branch
       first, and CI and the guard run again. Add the pull request to the
       merge queue. The `merge_group` guard run is `pass`; the group's
       `validate / validate` ends green; GitHub merges.
    - Negative canary, on the same candidate before step 5: an approval by
      the dispatcher or by the author leaves the guard run `fail`
      (`approver_is_dispatcher`, `approver_is_author`). A dismissal of the
      approval before the pull request is queued makes the `merge_group`
      run fail and the entry leaves the queue.
    - Rollback: step 12's rollback; close the candidate.
14. "Continuous landing is on" when step 13 passed and the maintainers say
    so in §11. What that means, and which human steps remain, is the next
    section.

Steps 10 and 12 land as separate reviewed changes, step 10 first, which keeps
proposed constraint D2d ("deployed and enforced before the pre-sign approval
is removed"). D2d forces the mechanism of step 1, the enforcing runner
(change 3b), the `Active` ruleset (step 10) and the v6 reader pin (step 9)
before activation (step 12). The canary (step 11) follows the proposed
sequence. This design chose the shadow period and the live cases (steps 6
and 7); D2d does not require them.

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
| Approve the `production-signing` environment for the run | Max or Pavel. `prevent_self_review` is off, so the dispatcher can approve the own run [code: job `encode`, `environment: production-signing`; live read 2026-11-17] | No | One approval per signing run | #1885 (promotion without per-run approval), if it lands |
| Merge the signed draft pull request into the candidate branch | anyone with write access **except the future approver**: the merger becomes an author of a commit of the candidate, and the guard refuses an approver who authored or committed one [code: step "Push lane branch and open draft pull request" of the signing workflow; `_actor_ids`, `R_AUTHORED_COMMITS`]. A squash merge keeps the bot-authored commit out; a commit that GitHub cannot attribute fails the guard (`unattributed_commit`) [code; the attribution of the bot commit is author-reported from rulespec-us#2045] | No | One merge per candidate | A workflow change that pushes the signed commit to the candidate branch with an App credential. Not designed |
| Approve the green candidate | the other enrolled owner: enrolled, not the author, not the dispatcher, not an author or committer of any commit [code: `_evaluate`, steps 5 and 6]. Two identities are enrolled [code: `TRUSTED_SIGNING_REVIEWERS`]. For a candidate that one of them authored, only the other can approve, and the author must be the dispatcher | **New** (it replaces the pre-sign approval) | One approval per candidate, by design (D2f, D3) | A maintainer policy that accepts machine validation alone. Not designed |
| Re-run the guard job after the approval | anyone with write access, if a re-run counts [docs D9; experiment X3, X16] | **New** under the ruleset path | One re-run per candidate | A helper workflow on the review event, if X16 shows a job token can re-run. Not built |
| Keep the head current with `main` | someone other than the approver updates the branch. The guard requires the head to be strictly ahead of the live `main` tip, in the pull request run and in the merge-group run [code: `R_BASE_NOT_CURRENT`]. Each update is a new head: CI and the guard run again; the approval stays | No | One update after each other landing | A queue that rebuilds stale entries would need the guard to accept a head behind `main`. Not designed |
| Merge | a person or agent adds the pull request to the merge queue. The lane merges nothing by itself [docs: `docs/reviewed-pr-promotion.md`] | No | One action per candidate, then 2.5 to 4 hours of group validation [author-reported, rulespec-us#2160] | — |

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
| Write collaborator on a RuleSpec repository, not enrolled | Yes for a v6 unit; through CI for a v5 lane; blocked otherwise | Cannot dispatch, cannot give a counting approval, and cannot publish as the guard App from a RuleSpec workflow. A v6 unit refuses every extra file. A v5 lane needs this PR's own green run. Any other guarded change has no lane |
| Write collaborator on axiom-encode, not enrolled | Ruleset path: yes, by the SHA pin, if step X1 of the experiment passes. App path: not yet; checklist step 1 must close this before anyone stores the key | Ruleset path: a change that lands on axiom-encode `main` does not reach the guard until someone with ruleset rights moves the pin [docs D3; experiment X1]; no key exists to take. The collaborator can still disable the workflow in axiom-encode, whose effect on the RuleSpec repository is [experiment X17]. App path: can land, without a code-owner review, a change to code the runner job executes (`src/axiom_encode/entrypoint.py`, `src/axiom_encode/__init__.py`, a module the CLI imports, `pyproject.toml`, `uv.lock` or a dependency), or a new workflow whose job selects `merge-guard`. Once someone stores the key, that code can take the private key as well as the token. With the key it can mint tokens for any App installation within that installation's granted repositories and permissions while the key stays valid, independent of one requested token's scope, lifetime or revocation. That permits misleading shadow output and, once the check is required, a forged required check. Checklist step 1 covers all selecting jobs in existing, new, replaced or modified workflows, dynamic selectors, the ability to land them, every pre-token step and persistence, builds, installation, dependencies, interpreter/startup hooks and the full console-script/import closure |
| PR that edits workflow files, pins, ledgers or `.axiom/` configuration | Yes, by refusal | No lane claims it: `fail`. This also blocks legitimate changes (§10, question 0) |
| PR that relabels or hides its manifest, or uses a root the encoder does not name | Yes | The guard stands aside only on the five-file allowlist or on the v5 lane, which needs this PR's green CI (step 3) |
| PR that changes `CLAUDE.md`, `.claude/` or `.gitattributes` | Yes, by refusal | Instruction and settings files steer the agents that hold the enrolled identities; Git attributes change the bytes CI validates. The guard treats both as guarded paths with no lane |
| Hand-written v5 manifest | Through CI | The guard does not verify the signature. The lane's scope keeps the PR from changing the workflow and the pin, and the PR's own `validate / validate` run must be green |
| `validate / validate` from a run that is not this PR's `pull_request` run | Yes | Step 7 |
| Forged or unsigned v6 manifest | Partly | Current-head CI verifies the signature; the guard does not (open question 3) |
| Second workflow imitating the check name | Ruleset path: unverified [experiment X2]; if a same-named check satisfies the rule, the ruleset path fails outright. App path: yes, once the App binding exists | Ruleset path: the rule names the source repository, file and commit [docs D3]. App path: `app_id` binding. It keeps other publishers out, and it cannot tell the guard's own code from unreviewed code that holds the App's key or token (row above) |
| Stale approval after a content change | Yes | Blob equality |
| Push racing an approval | Yes | The verdict belongs to one head, and the guard computes it from that head's tree |
| Dismissal just after a pass | Partly | App path: the check stays green until the next sweep, minutes. Ruleset path: no event follows a review, so the pull request run stays green. The merge-group run evaluates again and fails [code: `evaluate_merge_group`]; that needs the merge queue, a precondition of `Active`. A dismissal after the merge-group run and before the merge is not caught; that window is as long as the group's validation, hours [author-reported, rulespec-us#2160; experiment X15] |
| Organization owner, a holder of the organization's ruleset permission, or a session with such a token | No | Can set the ruleset to `Disabled`, add a bypass, or move the pin to an unreviewed commit [docs D4, D6, D16]. Three accounts are owners [live read 2026-11-17]; whether a custom role holds the permission was not readable. Proposed standing rule: no agent session gets `admin:org`. GitHub cannot enforce it. The ruleset history shows edits for 180 days [docs D6] |
| Compromised pinned action or runner image in the guard job | No | Can forge one verdict for one head. The job token is read-only and no key exists [inferred] |
| GitHub API failure | Fails closed for new verdicts | The guard does not withdraw a green check it wrote before the outage |
| Repository administrator, or a compromised administrator account | No (App path). Partly (ruleset path) | Ruleset path: a `rulespec-us` administrator cannot edit the organization ruleset and can only add stricter repository rules [docs D4]; what disabling Actions there does is [experiment X8]. App path: out of scope. Theft of the guard App key through a workflow or code change that a write collaborator can land stays in scope: the axiom-encode row above and checklist step 1 cover it |
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
| 0 | **First decision.** Which lane do infrastructure and ProgramSpec changes get? At `ca39ebb2`, `INFRASTRUCTURE_LANE_ENABLED = False` leaves them without a lane. The earlier author-reported census classified 48 of 50 PRs here (49 before #2045 merged) | Decide before branch protection requires the check. The off-by-default proposal in §3 keeps content lanes, requires one content-bound approval for named control-plane paths and leaves the stand-aside set empty. Its control-plane and stand-aside lanes cannot admit content. Maintainers still need to choose and review the lane and its coverage; the merge accepts no option |
| 0b | A module under a jurisdiction that the base lacks has no lane, also with a v5 manifest. How do maintainers add a new jurisdiction? | Design and review a lane separately. The infrastructure proposal classifies the path as content and does not widen the v5 lane |
| 1 | App-published check or an organisation required-workflow ruleset? | Recommendation recorded, Max has not accepted yet: the ruleset, at a pinned SHA (`docs/merge-guard-key-protection.md`), on the condition that the owner's experiment passes. The documentation confirms the rule, its events and the `sha` field [docs D1, D3]; it does not confirm the behaviors that the experiment tests. Fallback: the App with its key in a cloud key service |
| 2 | Is a polling sweep (about 5 minutes) acceptable as the trigger? | Under the ruleset path the question changes: no sweep exists. Each head and each merge group gets one run, and nothing follows a review. Recommendation recorded, Max has not accepted yet: accept that, with a re-run after the approval and a merge queue as a precondition of `Active`; the price is the stale-green window of §9. Under the App path: yes for the pilot |
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
| 14 | Add `run_attempt` to `attestation_scope`? | Yes, in the change that defines the v6 writer |
| 15 | The approver may not be the author or committer of any commit, so the approver cannot click "Update branch". Acceptable? | Yes for the pilot |
| 16 | Which mechanism closes the key-exposure set of checklist step 1: (a), (b), (c) or (g')? With (a) or (b), which complement stops another workflow from selecting `merge-guard`? | Recommendation recorded, Max has not accepted yet: (g'), which creates no key and so needs no complement; first fallback (b)+(d); second fallback (b) with code-owner review of `.github/` and last-push approval. Max chooses. Key storage and every activation step, shadow mode included, wait on his choice, implementation and fresh independent evidence over the whole set |

## 11. What remains unverified

### The ruleset experiment (not run)

`docs/merge-guard-ruleset-experiment.md` holds the procedure. An
organization owner runs it; an agent copies the filled record table here in
a reviewed pull request. Until then every row is open, and the
recommendation of checklist step 1 is conditional.

| Step | Question | Result |
|---|---|---|
| X1 | The SHA pin runs exactly that file | not run |
| X2 | A same-named check does not satisfy the rule | not run |
| X3 | A re-run after an approval updates the rule | not run |
| X4 | The RuleSpec repository's job token reads public axiom-encode | not run |
| X5 | The merge queue runs the workflow; the `merge_group` payload | not run |
| X6 | `Evaluate` mode runs the workflow and does not block | not run |
| X7 | A non-owner can read the active rule | not run |
| X8 | A head with no run stays blocked | not run |
| X9 | An empty bypass list binds an owner | not run |
| X10 | Renaming the target keeps the rule | not run |
| X11 | `github.workflow_sha` and `github.workflow_ref` in a ruleset run | not run |
| X12 | The check name, and the commit the result attaches to | not run |
| X13 | A re-run of an old run cannot make a new head green | not run |
| X14 | The branch-rules response for a `workflows` rule | not run |
| X15 | A dismissal and the merge queue | not run |
| X16 | Who can re-run a ruleset run | not run |
| X17 | A workflow disabled in the source repository | not run |

Also not recorded yet: the ruleset id, the pinned SHA and its
`BUNDLE_SHA256` (checklist steps 5 and 6), Max's decisions (step 0), and the
statement that continuous landing is on (step 14).

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
