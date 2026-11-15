# Content-bound merge guard

Status: **design; maintainer acceptance and activation remain pending.** At
source head `2bf14cb29905e3c7ff218fd440f41be7a6a05053`, the evaluator
(`src/axiom_encode/content_merge_guard.py`) supplies verdicts, the production
manifest writer emits v5, and the runner offers dry-run and shadow modes only.
No enforcing runner exists. The workflow
(`.github/workflows/merge-guard-runner.yml`, change 3 of §8) has only a manual
dispatch trigger. Its gate job checks `main` and
`vars.MERGE_GUARD_RUNNER_ENABLED == 'true'`; its `run` job checks for App
credentials and relies on `needs: gate` without repeating the ref or switch
checks. The Python runner and CLI accept a token-shaped `MERGE_GUARD_TOKEN`
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
conditions does not satisfy them.

The v6 reader (`src/axiom_encode/applied_manifest_v6.py`, change 2 of §8)
keeps `V6_READER_ENABLED = False`. Every verifier therefore refuses v6 at
this source head. Every promotion still follows
`docs/reviewed-pr-promotion.md`. Tracking issue: #2172.

The design follows the agent-proposed answers in
`docs/reviewed-pr-promotion-order-proposal.md`. An agent posted those answers
under @MaxGhenis's account; Max has not ratified them. D1, D2a-D2f, D3 and Seq.
identify proposed constraints throughout this design and related source
comments, not maintainer acceptance. Platform claims without confirmation carry **[check docs]**.

## 1. What exists today

The following admission baseline describes `main` at `eb98f8d7` (after
#2194); later sections describe the reviewed runner source at `2bf14cb2`.

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
| D2b | "The required check is bound to its trusted publisher." | §4: dedicated GitHub App, required with its `app_id` |
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
  across these endpoints; the next sweep catches the change.
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

A read-only census on 2026-11-12, repeated after rulespec-us#2045 merged
(before that the counts were 1 and 49), took the 50 most recently merged PRs to
`main` among the 100 most recently updated closed PRs and classified them with
the rules of that date (reduced allowlist, digest and position binding). The
census read file lists, changed manifests and module bytes at the merge
commit, and took the jurisdictions from `main` at that read. It did not read
modes or CI and assumed regular modes and passing CI for each merged PR. No PR in the census carried a
v6 manifest, so none could get `pass`.

| Verdict | PRs |
|---|---|
| `not_applicable` (`no_guarded_content`, allowlist) | 0 |
| `not_applicable` (`earlier_lane_v5`) | 2 (#2045 and #2154, a merge train of 5 signed manifests; all 5 satisfy the digest and position rules) |
| `fail` (`no_lane_for_guarded_change`) | 48 |
| `fail`, other reasons | 0 |

The 48 cover toolchain and corpus pins and staged waiver fingerprints in
`.axiom/toolchain.toml` or `.axiom/workflow-toolchain.toml` (about 21), ledger
and register changes (`known-*.yaml`, `oracle-coverage-pending.yaml`), workflow
changes under `.github/`, ProgramSpec changes under `programs/`, tests and
tooling, and one signed manifest PR with extra files (#1470). Over a wider
window of 200 merged PRs the counts were 136 no-lane, 2 earlier lane and 62
other; the 39 of those 62 that the census listed all carried the historic
`applied-rulespec/v1` label, which predates v5.

For that 50-PR window, the classifier refused 48 PRs. Maintainers should not make branch protection require it
until they decide which lane these changes get.

### CLI

`axiom-encode merge-guard --repository rulespec-us --pr <n>` prints one JSON
line (`schema`, `verdict`, `reason`, `detail`, `evidence`). Exit codes: 0 pass,
1 fail, 3 not_applicable, 2 usage error.

### Implementation notes

The source at `2bf14cb2` defines the behavior described here.

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

The runner of change 3 exists and is inert ("Runner as built" below).
Before trusting the check, maintainers must prevent unreviewed code from
obtaining the guard App's private key or any token it mints. Checklist step 1 of
§8 requires Max's choice, implementation and fresh independent evidence over
the whole set before anyone stores the key or enables shadow execution.

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
  later holds the App's key and token. The reviewed CODEOWNERS file covers few modules in that import closure and
  has no catch-all for new workflow files. At the 2026-11-15 protection read, a
  write collaborator could land those changes without code-owner review. Proposed constraint D2a holds for the
  runner only once one reviewed mechanism covers the key-exposure set of
  checklist step 1 of §8.
- **Identities**: the existing `TRUSTED_SIGNING_REVIEWERS` on encoder main. A
  RuleSpec PR cannot change it.
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
- **Why not run it in `rulespec-us`**: under `pull_request` the PR controls the
  workflow file. Under `pull_request_target` the base branch supplies the file, but its check
  still publishes as app 15368, which any PR workflow can imitate.
- **Alternative**: an organisation ruleset with "Require workflows to pass
  before merging". It needs no App. The retained evidence does not confirm plan availability,
  supported events or re-run on review **[check docs]**.

### Runner as built (change 3, shadow only, inert)

`.github/workflows/merge-guard-runner.yml` calls `axiom-encode merge-guard-run`
(`src/axiom_encode/merge_guard_runner.py`). The workflow evaluates `vars`,
which supports repository, organization and environment variables; it checks
no repository-only origin. The retained evidence does not establish actual
same-name organization variables or how a rerun resolves a changed variable.
[GitHub variables context](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#vars-context).

The Python CLI accepts `MERGE_GUARD_TOKEN` from its process environment and
checks its shape. It does not check the workflow ref, switch, App identity or
credential provenance. The intended workflow supplies an App installation
token; calling the CLI elsewhere does not inherit the workflow gate.

| Item | Value |
|---|---|
| Trigger | `workflow_dispatch` only (inputs `country`, `pr`, `mode`). No `schedule`: the sweep on a timer is a later reviewed change to the workflow file |
| Switch | The gate job requires `vars.MERGE_GUARD_RUNNER_ENABLED` to equal `true`, and `github.ref == refs/heads/main`. The `run` job repeats neither check and relies on `needs: gate`. A rerun of `run` can reuse an earlier successful gate, so unsetting the variable alone does not stop it. GitHub documents that a disabled workflow cannot rerun; change 3's rollback unsets the switch and disables the workflow. Neither check governs another workflow that selects `merge-guard` (checklist step 1 of §8). [Rerun documentation](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs) |
| Environment | The workflow names `merge-guard`; its planned deployment branch rule allows `main` only. The gate job has no environment, so a refused dispatch does not create one |
| App client ID | Intended environment variable `MERGE_GUARD_APP_CLIENT_ID`; the workflow reads the `vars` context |
| App private key | Intended environment secret `MERGE_GUARD_APP_KEY`; the workflow reads the `secrets` context |
| Token | The `run` job mints one per run, for `rulespec-<country>` only: `checks: read` in `dry-run`, `checks: write` in `shadow`; `pull_requests`, `contents`, `actions`: `read`. The CLI reads it from `MERGE_GUARD_TOKEN` and has no token argument |
| `dry-run` (default) | Prints one JSON report. Writes nothing |
| `shadow` | Also creates one check run per judged head, named `content-bound-merge-guard (shadow)`, conclusion always `neutral`, verdict in the title and summary. It skips the write only when the guard App's own newest shadow check on that head (the runner matches the App by its client ID, `MERGE_GUARD_APP_CLIENT_ID`) already carries the same verdict. A check with the shadow name from another publisher never counts, so a pull request workflow cannot suppress its verdict |
| Enforcing mode | None. This change reserves the name `content-bound-merge-guard` for it and never publishes that name |

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
  Deeply nested response JSON remains an exception: `_request` catches
  `ValueError`, while `RecursionError` escapes `command()`. The retained R2
  fake-opener probe exited nonzero without a JSON report; it made no
  live network request.
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
  `_already_reported` rejects an incomplete listing. R2 inferred that more
  than 100 shadow-named checks on one head could hit that refusal; confirmation
  still needs the platform's `filter=latest` semantics or a test repository.
- **Known limit of the sweep, acceptable for shadow only.** The runner reads
  the list by page number. A pull request that opens or closes during the read shifts
  the pages, so one sweep can miss a pull request or see one twice (twice is
  a run error). A missed pull request has no shadow check until the next
  sweep. An enforcing check would leave that pull request without its
  required check, so change 3b must read from a stable snapshot or trigger per
  head.

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

## 6. Merge queue and `notary-guard` (not implemented)

`evaluate_merge_group`, a stub, returns `fail`.

- The guard must also report a required check on the merge-group commit. The planned
  algorithm computes the delta of the group head against its queue base, treats
  every v6 manifest as a unit `{M, P, T}`, and requires each unit to map to a
  queued PR with the same three blob ids for which `evaluate` passes.
- rulespec-us#2160 plans a group size of 1; keep that size for the pilot.
- **[check docs]** How to list the PRs in a merge group, and whether a
  check run that the App creates on the group SHA satisfies the queue.
- `notary-guard` runs independently. A later integration would let this guard
  report `not_applicable` for a notary lane whose immutable base carries
  `consumer.json` and whose lineage replaces apply manifests. The current
  evaluator has no notary-specific lane; question 6 of §10 remains open.

## 7. Adversarial test plan

`tests/test_content_merge_guard.py` holds the cases as fixture-driven unit
tests over an injected `fetch`. A skipped test states its reason and does not
count as a pass.

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
| 15 | PR adds a workflow with a job that imitates the guard check name | merge blocked: wrong `app_id` | skipped: needs the App binding |
| 16 | PR rewrites `repository-checks.yml` so `validate / validate` passes | fail | yes: scope for a v6 unit, no lane otherwise |
| 17 | API truncates a tree, supplies an 11th review page, redirects, or returns HTTP 403/500 | fail; no cached pass | yes, for every read of a passing evaluation |
| 18 | Someone dismisses a review between the first and second read | fail | yes |
| 19 | v6 manifest whose `candidate.pull_request` names another PR | fail | yes |
| 20 | PR supplies only a v6 manifest, or supplies two v6 manifests | fail | yes |
| 21 | Manifest `applied_files` digest differs from the head blob | fail | yes |
| 22 | CI passes `validate / validate` on an older head while `H` remains pending | fail | yes |
| 23 | PR modifies a module that already exists on `main` | fail | yes |
| 24 | Merge group includes two units and lacks an approval for one | fail for the group | skipped: merge group is a stub |
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

Cases 15-18 must also run live in the shadow period (§8).

## 8. Delivery plan

Implementation comes before activation.

| # | Repo | Change | Rollback |
|---|---|---|---|
| 1 | axiom-encode | Evaluator, `merge-guard` CLI, tests, CODEOWNERS and this document. The CLI and runner call the evaluator; this step creates no enforcing check | revert |
| 2 | axiom-encode | Verifiers can read `applied-rulespec/v6` with the attestation scope, behind `V6_READER_ENABLED = False`; no writer | revert; the production writer emits only v5 |
| 2b | axiom-encode | Set `V6_READER_ENABLED = True`. May ride with change 3. Switch rule: The reader may be switched on while no v6 writer exists: nothing new can land. The v6 writer and the admission without approval must stay off until the guard check is required. With the reader on, the writer on and the guard check not required, an enrolled dispatcher lands their own unreviewed candidate with no approval, and any account with write access re-lands signed v6 bytes from a closed pull request. | set it back; the production writer emits only v5 |
| 3 | axiom-encode | Workflow that runs the guard from `main` and publishes its check, with a dry-run mode. Shadow only and inert: manual dispatch, behind `MERGE_GUARD_RUNNER_ENABLED` (§4, "Runner as built"). It does not flip `V6_READER_ENABLED`; change 2b stays a separate one-line commit to `applied_manifest_v6.py` | unset the variable; disable the workflow |
| 3b | axiom-encode | **Enforcing runner (not built).** A mode that reports the real check `content-bound-merge-guard` (`success` for `pass` and `not_applicable`, `failure` for `fail`) and a trigger that runs it on every pull request head and after every review change (a `schedule` sweep or an event), with a work list it reads from a stable snapshot instead of page numbers, and a measured request volume. Hard precondition of checklist step 4 | remove the check from the required checks first, then disable the trigger |
| 4 | rulespec-us | Pin the encoder that verifies v6 | revert the pin |
| 5 | axiom-encode | Emit v6 and admit without approval, behind a default-off switch | revert |
| 6 | axiom-encode | **Activation**: turn the switch on; update `docs/reviewed-pr-promotion.md` | flip the switch back; the required check stays |

### Activation checklist

Steps with **M** need repository or organisation settings. Only a maintainer
can do them. Maintainers must accept the proposed promotion order before
activation.

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

   The candidates, none of them chosen:
   - (a) enforced code-owner review of the whole key-exposure set, including
     review of any change to the set itself;
   - (b) a separately reviewed, pinned and immutable guard build, which the job
     runs without installing or starting unreviewed code;
   - (c) required human reviewers on the `merge-guard` environment, who approve
     each run that releases the key, with no agent session approving for them.

   (a) and (b) each need a separate control that prevents other workflows
   from obtaining the key by selecting `merge-guard`, including existing
   workflows, dynamic selectors and any new, replaced or changed workflow
   someone can land. A
   pinned build cannot stop a second workflow from selecting the same
   environment, and review of existing files does not reach a new workflow
   file. Max chooses the complement along with the mechanism. This design
   chooses neither.

   The `main`-only branch rule alone does not meet this precondition.
   Recording the precondition in this design does not meet it either. After
   Max chooses, maintainers implement the mechanism and any complement, and a
   fresh independent review verifies closure over the whole key-exposure set,
   maintainers may create the guard App, install it on `rulespec-us`, and store its key
   in an axiom-encode environment whose deployment branch rule allows only
   `main`: environment `merge-guard`, variable `MERGE_GUARD_APP_CLIENT_ID`,
   secret `MERGE_GUARD_APP_KEY`. Only after those conditions hold may a maintainer set
   `MERGE_GUARD_RUNNER_ENABLED` to `true`. The workflow resolves the `vars`
   context, not a repository-only origin; the actual same-name configuration
   and rerun resolution still need verification.
2. Merge changes 1-4, including 2b. Publish the check in shadow mode: visible,
   not required.
   Switch rule: The reader may be switched on while no v6 writer exists: nothing new can land. The v6 writer and the admission without approval must stay off until the guard check is required. With the reader on, the writer on and the guard check not required, an enrolled dispatcher lands their own unreviewed candidate with no approval, and any account with write access re-lands signed v6 bytes from a closed pull request.
3. Run cases 15-18 live on a throwaway PR and confirm every verdict.
4. **M** Add the guard's check with the guard App's `app_id` to the required
   checks on `rulespec-us` main. Hard preconditions: the step 1 mechanism,
   with any complement it needs, covers the key-exposure set as change 3b's
   enforcing runner extends it, with fresh independent evidence of closure over the whole set for that runner; change 3b
   has landed on `main` and reports `content-bound-merge-guard` on every open
   pull request head; and the maintainers have decided question 0 of §10
   (the recorded census classifier refused 48 of 50 PRs).
   **WARNING:** require the name `content-bound-merge-guard` only. Never
   require `content-bound-merge-guard (shadow)`: it is always neutral, a
   neutral check satisfies a required check, so it would enforce no verdict. Without change 3b no code reports the real name, and requiring
   it blocks every pull request, administrators included.
5. Run the current-order canary (rulespec-us#2045) to completion.
6. Merge change 5 with the switch off.
7. Merge change 6. The first promotion in the new order is the pilot.
8. **M**, optional hardening: enforce code-owner review for `/.github/` in
   `rulespec-us` (§9). That protects the RuleSpec workflows. It does not cover
   the encoder workflows and code that can obtain the App key or token, which
   step 1 governs.
9. **M**, before the merge queue (rulespec-us#2160): confirm the guard reports
   on group SHAs in a test repository.

Steps 4 and 7 land as separate reviewed changes, step 4 first, which keeps
proposed constraint D2d.

D2d ("deployed and enforced before the pre-sign approval
is removed") forces the guard App (step 1), the runner and the v6 reader pin
(step 2), the enforcing runner (change 3b) and the required check (step 4)
before activation (step 7). The
canary (step 5) follows the proposed sequence. This design chose the shadow
period and the live cases (steps 2 and 3); D2d does not require them.

## 9. Threat model

| Adversary | Defended? | How |
|---|---|---|
| Write collaborator on a RuleSpec repository, not enrolled | Yes for a v6 unit; through CI for a v5 lane; blocked otherwise | Cannot dispatch, cannot give a counting approval, and cannot publish as the guard App from a RuleSpec workflow. A v6 unit refuses every extra file. A v5 lane needs this PR's own green run. Any other guarded change has no lane |
| Write collaborator on axiom-encode, not enrolled | Not yet; checklist step 1 must close this before anyone stores the key | Can land, without a code-owner review, a change to code the runner job executes (`src/axiom_encode/entrypoint.py`, `src/axiom_encode/__init__.py`, a module the CLI imports, `pyproject.toml`, `uv.lock` or a dependency), or a new workflow whose job selects `merge-guard`. Once someone stores the key, that code can take the private key as well as the token. With the key it can mint tokens for any App installation within that installation's granted repositories and permissions while the key stays valid, independent of one requested token's scope, lifetime or revocation. That permits misleading shadow output and, after step 4, a forged required check. Checklist step 1 covers all selecting jobs in existing, new, replaced or modified workflows, dynamic selectors, the ability to land them, every pre-token step and persistence, builds, installation, dependencies, interpreter/startup hooks and the full console-script/import closure |
| PR that edits workflow files, pins, ledgers or `.axiom/` configuration | Yes, by refusal | No lane claims it: `fail`. This also blocks legitimate changes (§10, question 0) |
| PR that relabels or hides its manifest, or uses a root the encoder does not name | Yes | The guard stands aside only on the five-file allowlist or on the v5 lane, which needs this PR's green CI (step 3) |
| PR that changes `CLAUDE.md`, `.claude/` or `.gitattributes` | Yes, by refusal | Instruction and settings files steer the agents that hold the enrolled identities; Git attributes change the bytes CI validates. The guard treats both as guarded paths with no lane |
| Hand-written v5 manifest | Through CI | The guard does not verify the signature. The lane's scope keeps the PR from changing the workflow and the pin, and the PR's own `validate / validate` run must be green |
| `validate / validate` from a run that is not this PR's `pull_request` run | Yes | Step 7 |
| Forged or unsigned v6 manifest | Partly | Current-head CI verifies the signature; the guard does not (open question 3) |
| Second workflow imitating the check name | Yes, once the App binding exists | `app_id` binding. It keeps other publishers out, and it cannot tell the guard's own code from unreviewed code that holds the App's key or token (row above) |
| Stale approval after a content change | Yes | Blob equality |
| Push racing an approval | Yes | The verdict belongs to one head, and the guard computes it from that head's tree |
| Dismissal just after a pass | Partly | The check stays green until the next sweep |
| GitHub API failure | Fails closed for new verdicts | The guard does not withdraw a green check it wrote before the outage |
| Repository administrator, or a compromised administrator account | No | Out of scope. Theft of the guard App key through a workflow or code change that a write collaborator can land stays in scope: the axiom-encode row above and checklist step 1 cover it |
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
| 0 | **First decision.** Which lane do infrastructure changes (`.github/**`, `.axiom/` pins and configuration, ledgers, tooling, `.gitattributes`, `CLAUDE.md`, `.claude/`) and ProgramSpec changes get? At the reviewed head, none: `fail`. The 2026-11-12 census classified 48 of 50 PRs here (49 before #2045 merged) | Decide before branch protection requires the check. One option is the same enrolled-owner, content-bound approval as the v6 lane, without a manifest |
| 0b | A module under a jurisdiction that the base lacks has no lane, also with a v5 manifest. How do maintainers add a new jurisdiction? | Through the lane of question 0 |
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
| 11 | The guard binds `candidate.head_sha` to the PR's commits, so a rebase needs a new signing run. Acceptable? | Yes: the binding is worth more than rebase support |
| 12 | The v5 lane is `not_applicable` on scope plus this PR's green CI, with no approval check. Enough? | Yes while admission checks the approval before signing |
| 13 | A signed v5 PR that also changes a receipt, pin, ledger or test (rulespec-us#1470) has no lane. Widen the v5 lane, or split such PRs? | Split, or give the extra files the lane of question 0 |
| 14 | Add `run_attempt` to `attestation_scope`? | Yes, in the change that defines the v6 writer |
| 15 | The approver may not be the author or committer of any commit, so the approver cannot click "Update branch". Acceptable? | Yes for the pilot |
| 16 | Which mechanism closes the key-exposure set of checklist step 1: (a), (b) or (c)? With (a) or (b), which complement stops another workflow from selecting `merge-guard`? | None; Max chooses. Key storage and every activation step, shadow mode included, wait on his choice, implementation and fresh independent evidence over the whole set |

## 11. What remains unverified

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
  `app_id` on merge-group commits, and how to list a group's PRs.
- Which workflow file version `pull_request_review` and `merge_group` runs use.
- Organisation plan and availability of required-workflow rulesets.
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
- Current tree capacity and headroom. A recursive tree that GitHub truncates
  gives `fail` for every PR of that repository. Before branch protection
  requires the check, measure the current rulespec-us tree, confirm the
  platform's current limits from primary documentation and record the margin.
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
