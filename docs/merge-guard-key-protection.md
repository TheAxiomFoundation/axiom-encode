# Merge guard key protection: a recommendation

Status: **proposal; Max has not chosen.** This document answers the open
decision in checklist step 1 of `docs/content-bound-merge-guard.md` §8 and
question 16 of §10, as draft PR #2237 rewrites them, and the "Activation
prerequisite" of issue #2257. It recommends one mechanism and one complement.
It changes no setting, creates no App, environment or secret, and satisfies no
activation condition.

Nobody may store the guard App key or set `MERGE_GUARD_RUNNER_ENABLED`, shadow
mode included, until Max chooses a mechanism and complement, maintainers
implement them, and a fresh independent review verifies the whole
key-exposure set (§6). Checklist step 1 on `main` still reads "Create the guard
App ... and store its key"; it predates this precondition, and #2237 replaces
it. Until then, follow #2257, not step 1 on `main`.

**Present exposure, independent of the guard (R3):** two keys stored today
have the selection gap this document describes. §5 "Now" lists mitigations
that do not wait for Max's choice of mechanism.

Repository facts come from read-only reads on 2026-11-15. Platform facts link
to the documentation read the same day. Claims the documentation does not
state carry **[check docs]**.

## 1. The options as the design defines them

Quoted from checklist step 1 at #2237 head `d7fce6b0`:

> The candidates, none of them chosen:
> - (a) enforced code-owner review of the whole key-exposure set, including
>   review of any change to the set itself;
> - (b) a separately reviewed, pinned and immutable guard build, which the job
>   runs without installing or starting unreviewed code;
> - (c) required human reviewers on the `merge-guard` environment, who approve
>   each run that releases the key, with no agent session approving for them.
>
> (a) and (b) each need a separate control that prevents other workflows
> from obtaining the key by selecting `merge-guard`, including existing
> workflows, dynamic selectors and any new, replaced or changed workflow
> someone can land. A pinned build cannot stop a second workflow from
> selecting the same environment, and review of existing files does not reach
> a new workflow file. Max chooses the complement along with the mechanism.
> This design chooses neither.

The key-exposure set is the one #2257 and checklist step 1 define: every job in
any workflow that can select the key-bearing environment, who can land such
workflows, and everything those jobs execute or leave on the runner.

## 2. Facts this evaluation uses

Platform ([environments](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments),
[managing environments](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments),
[`pull_request_target` change](https://github.blog/changelog/2025-11-07-actions-pull_request_target-and-environment-branch-protections-changes/),
[pending deployments API](https://docs.github.com/en/rest/actions/workflow-runs#review-pending-deployments-for-a-workflow-run),
[reusable workflows](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows),
[ruleset rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets),
[code owners](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners),
[re-runs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs),
[OIDC](https://docs.github.com/en/actions/reference/security/oidc),
[branch policies API](https://docs.github.com/en/rest/deployments/branch-policies),
[hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners),
[App private keys](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/managing-private-keys-for-github-apps),
[Python options](https://docs.python.org/3/using/cmdline.html),
[`site`](https://docs.python.org/3/library/site.html)):

- **P1** Environment secrets are available only to jobs that reference the
  environment. With required reviewers, a job cannot read them until one
  reviewer approves. Up to six reviewers; one approval suffices. "Prevent
  self-review" blocks only the user who started the run. Environment names are
  not case sensitive. A run that references a missing environment creates it,
  with no protection rules and no secrets.
- **P2** Deployment branch and tag rules match the run's `GITHUB_REF`; they do
  not limit which workflow file selects the environment. Each policy has a
  `type`, `branch` or `tag`. Since 2025-12-08, environment rules for
  `pull_request_target` evaluate against the default branch, so a `main` rule
  admits such a run, and its workflow file comes from the default branch.
  Administrators can bypass environment protection rules unless the
  environment disallows it.
- **P3** The pending-deployments endpoint accepts any required reviewer with
  read access; classic tokens need the `repo` scope. GitHub cannot tell a
  person from an agent session that holds the reviewer's token (design §9).
- **P4** A job in a called reusable workflow can set `environment`; it receives
  that environment's secret only when the caller passes it (`secrets: inherit`
  or by name). Which repository's environment applies to a cross-repository
  call: **[check docs]**. The proposal refuses such calls (§4).
- **P5** Push rulesets with "Restrict file paths" are available for private and
  internal repositories only. axiom-encode is public.
- **P6** Pull requests use the base branch's `CODEOWNERS`. Code owners need
  write access. One owner's approval satisfies "Require review from Code
  Owners". The last matching pattern wins. `CODEOWNERS` must own itself.
- **P7** A re-run is possible for 30 days, uses the original `GITHUB_SHA` and
  `GITHUB_REF` and the privileges of the original actor, and is refused while
  the workflow is disabled. That a re-run also uses the original workflow file
  follows from the same commit and #1885's review relied on it; the page does
  not state it **[check docs]**. Whether a re-run reads the environment's
  current secrets: **[check docs]**; this proposal assumes it does.
- **P8** An OIDC token's `sub` contains `environment:<name>` when the job
  references an environment; `workflow_ref`, `workflow_sha`, `event_name` and
  `runner_environment` are separate claims.
- **P9** `python -I` ignores `PYTHON*` variables, the user site and the script
  or current directory. `-S` stops the automatic `site` import, so no
  site-packages directory, no `.pth` file (whose `import` lines run) and no
  `sitecustomize` or `usercustomize` take part in start-up.
- **P10** Each GitHub-hosted runner is a new VM, except single-CPU runners,
  which run in a container on a shared VM. Linux and macOS runners have
  passwordless `sudo`. That a standard VM is discarded after the job:
  **[check docs]**.
- **P11** GitHub App private keys do not expire; someone must revoke them. An
  App can have up to 25 keys. GitHub refuses to delete an App's only key until
  a new key exists. The documented check of a key is its SHA-256 fingerprint
  (`openssl rsa -in key.pem -pubout -outform DER | openssl sha256 -binary | openssl base64`)
  against the App's settings page; the documentation names no API that lists
  keys.

Repository (axiom-encode unless stated):

- **R1** Public. `main` requires code-owner review, dismisses stale reviews,
  enforces protection for administrators and has no review bypass
  allowances. It requires no approving review for files without a code owner
  and does not require last-push approval. The "main merge queue" ruleset has
  no bypass actors. The `CODEOWNERS` errors API returns none.
- **R2** Three administrators and two other accounts can push. The code owners
  are @PavelMakarchuk and @MaxGhenis; one administrator is not a code owner.
  `CODEOWNERS` names four workflow files and has no rule for the rest of
  `.github/`, so a new workflow file has no code owner. The encoder version
  bump App opened 28 of the last 200 merged pull requests.
- **R3** Environments, and the present exposure:
  - `production-signing`: required reviewers Max and Pavel, administrator
    bypass off, secrets `AXIOM_ENCODE_APPLY_SIGNING_KEY` and
    `AXIOM_REPO_TOKEN`.
  - `apply-signing`: secret `AXIOM_ENCODE_APPLY_SIGNING_KEY`, stored
    2026-11-02, behind a `main` branch rule only, with no required reviewers.
    No workflow on `main` selects it; #1885 would.
  - `encoder-version-bump`: secret `ENCODER_VERSION_BUMP_APP_KEY`, stored
    2026-11-06, behind a `main` branch rule only. It is the only environment
    with administrator bypass on.
  - `generation-approval` and `snap-queue-finalization`: no secrets. No
    `merge-guard` environment.

  **Present exposure.** A new workflow file has no code owner and `main`
  requires no other approval (R1, R2). So any account that can push can merge
  a new workflow that runs on `main`, selects `apply-signing` or
  `encoder-version-bump`, and reads the apply key or the bump App's key. For
  the apply key this also skips `production-signing`'s per-run approval. The
  bump App's key does not expire (P11). This is the selection gap of §3, open
  today. §5 "Now" lists the mitigations.
- **R4** In the 17 workflows on `main`, every job that selects an environment
  names it statically, no job calls a reusable workflow, and every action is
  pinned to a full commit SHA. In the 60 days to 2026-11-15, 32 of 202 merged
  pull requests changed `.github/workflows/`.
- **R5** The `run` job at #2237 head installs the encoder (`uv sync --locked
  ... --extra api`) and starts `.venv/bin/axiom-encode merge-guard-run`. A static
  scan of that start path (package `__init__.py`, console script, CLI and
  function-level imports) reaches 200 encoder modules and 14 third-party
  top-level packages. The design's run-time trace counted 150 modules.
- **R6** The guard itself needs five modules at module level, all stdlib-only:
  `merge_guard_runner.py` (423 lines), `content_merge_guard.py` (1,465),
  `reviewed_pr_admission.py` (445), `applied_manifest_v6.py` (232) and
  `prepare_signed_backfill.py` (6,405). Under `python3.13 -I -S`, with these
  five files and an empty `axiom_encode/__init__.py`, importing the runner
  loads exactly these modules and no third-party module, and the runner
  refuses a missing token. `CODEOWNERS` at #2237 head covers all five.
- **R7** One lazy edge leaves the five. On the v6 lane, `content_merge_guard`
  calls `citation_rulespec_path`, whose helper imports
  `axiom_encode.harness.evals` (20,516 lines) and with it the `harness`
  package. The function it needs, `_resolve_eval_output_path`, uses 12 helper
  functions and 8 constants from `evals.py`, and
  `statute.citation_to_citation_path` and
  `statute.normalize_rulespec_path_segment`. `statute.py` (98 lines) is
  stdlib-only and has no code owner. `prepare_signed_backfill.py` changed in
  53 commits in 60 days.

## 3. Evaluation against the key-exposure set

| Part of the set | (a) code-owner review | (b) reviewed, pinned build | (c) per-run reviewers |
|---|---|---|---|
| Other jobs that select the environment: existing, new, replaced, modified, dynamic | Needs a complement (P2) | Needs a complement (P2) | Covered: every job that references the environment waits (P1) |
| Who can land those workflows | Covered only if the set includes all of `.github/` | Covered by the complement | Landing is open; each approval must catch it |
| Steps before the token, and what they leave behind | Reviewed, but they install and start the whole encoder | Pinned GitHub actions, a sparse checkout and a file copy; no repository code runs | Approver sees a run, not the code its steps execute |
| Builds, installation, dependencies and their hooks | Not closable by review: a `uv.lock` change shows hashes, not package code (R5: 14 packages) | Removed: no install step, and `-S` keeps site-packages off `sys.path` (P9) | Trusted on each approval |
| Interpreter start-up, `__init__.py`, entry point, `signing_broker`, import closure | About 200 modules under owner review, plus `pyproject.toml` and `uv.lock` | Six owned modules and a launcher after R7 is cut; stub `__init__.py`, no console script, no `signing_broker` | Trusted on each approval |
| Re-run of an older workflow revision (P7) | Unchanged | Needs a key environment that no earlier revision names | Each re-run waits for approval |
| Fit with change 3b (sweep about every 5 minutes, or per head) | Fits | Fits | About 288 approvals a day, or required checks go missing and block merges |
| A session that holds a code owner's or reviewer's token (design §9) | Residual | Residual (through owner review) | Residual on every run (P3) |
| Recurring review load | Every PR that touches the encoder or its lock file | PRs that touch the guard files or `.github/` (about 16% of PRs, R4) | One approval per run |

**Why not (a).** The set it must review is the whole encoder, its lock file and
14 third-party packages with their hooks. Review of a `uv.lock` change does not
review the code of the new versions, so (a) cannot close the dependency part.
In practice it needs `CODEOWNERS` over most of the repository, and it still
needs a complement.

**Why not (c).** It is the only option that needs no complement, because the
reviewer gate applies to every job that names the environment. But an approver
approves a run, not the code: each approval trusts the whole R5 closure at that
commit, and one approval of a compromised commit leaks the key for good (P11).
The planned sweep (§10 question 2) and change 3b need runs without a person in
the loop. GitHub accepts the approval from any session that holds a reviewer's
token (P3), so "no agent approves" is a process rule that GitHub cannot
enforce. The repository already uses (c) for `production-signing` as a stopgap;
the `CODEOWNERS` header names a "separately reviewed, pinned encoder" as the
way out of it.

**Why (b).** The guard does not need the encoder package (R6). Running only its
files, from the dispatched `main` commit, under `-I -S`, removes installation,
dependencies, start-up hooks, `__init__.py`, the console script and
`signing_broker` from the job. What remains is small, owned by code owners,
and a reviewer can read all of it.

## 4. Recommendation

**Mechanism: (b), as a stdlib guard bundle.** The implementing change, in
axiom-encode after #2237 merges:

1. Cut the R7 edge. Move `RULESPEC_ATOMIC_ROOTS`, `citation_rulespec_path`,
   `_resolve_eval_output_path` and its 12 helper functions and 8 constants from
   `harness/evals.py` into a new stdlib-only module, for example
   `src/axiom_encode/rulespec_citation_paths.py`, that
   `prepare_signed_backfill.py`, `harness/evals.py` and
   `content_merge_guard.py` import. The new module imports the two `statute`
   functions. The guard then stops importing `prepare_signed_backfill.py`. The
   existing tests pin unchanged behaviour.
2. The bundle is then: `merge_guard_runner.py`, `content_merge_guard.py`,
   `reviewed_pr_admission.py`, `applied_manifest_v6.py`, the new module,
   `statute.py` and a launcher (about 30 lines). The launcher parses the three
   runner arguments, puts only the bundle directory on `sys.path`, installs a
   `sys.meta_path` finder that refuses any `axiom_encode` module outside the
   bundle list, and calls `merge_guard_runner.command`. A module outside the
   list fails the run closed.
3. Change the key-bearing job:
   - `environment: merge-guard-key`, a name that no earlier revision of any
     workflow names (P7). Leave `merge-guard` absent or without secrets; a run
     of the #2237 revision can create it (P1).
   - Before the token step: the credential check (shell only);
     `actions/checkout` pinned by SHA with `persist-credentials: false` and a
     sparse checkout of the bundle files; `actions/setup-python` pinned by SHA
     (`a309ff8b...`, v6.2.0 is already in use), Python 3.13, no `cache` input;
     copy the bundle files read-only into `$RUNNER_TEMP/guard/axiom_encode/`
     next to an empty `__init__.py`.
   - No `setup-uv`, `uv`, `pip`, `.venv`, console script, cache or artifact
     download anywhere in the job. `runs-on: ubuntu-latest`, never a
     single-CPU or self-hosted runner (P10).
   - Token step unchanged. Run `python -I -S "$RUNNER_TEMP/guard/launch.py"`.
     If #2257 item R2-P3-1 has not already done so, this job repeats the ref
     and switch checks of the gate job.
   - The workflow keeps `workflow_dispatch` as its only trigger. Change 3b's
     `schedule` trigger is a later reviewed change to the same file and to the
     tripwire test.
4. Tests: each bundle file is stdlib-only at module and function level; a
   fixture sweep that reaches every lane, run under the launcher with
   `-I -S`, loads only listed modules; the key job contains none of the
   forbidden steps and uses the flags and environment name above.
5. `CODEOWNERS` owns the new module, `statute.py`, the launcher and the tests.

**Complement: code-owner review of all of `.github/` in axiom-encode, with
last-push approval.**

- Add `/.github/ @PavelMakarchuk @MaxGhenis` to `CODEOWNERS`. With R1's
  protection, every new, replaced or modified workflow, any future
  `.github/actions/` and `CODEOWNERS` itself then need a code owner's
  approval, administrators included.
- Turn on "Require approval of the most recent reviewable push". Without it,
  one person, or a session with that person's token, can push a workflow
  change to a pull request that the other owner or the bump App opened (28 of
  the last 200 merged, R2), then approve it as the code-owner review.
- Keep `tests/test_merge_guard_key_environment.py` (this change). It fails if a
  job other than `merge-guard-runner.yml:run` selects `merge-guard` or
  `merge-guard-key`, if any job chooses its environment by expression, if a job
  calls a reusable workflow from another repository (P4), if another workflow
  calls the guard workflow, or if the guard workflow has any trigger other than
  `workflow_dispatch` (a `pull_request_target` run passes a `main` rule, P2).
  It is a tripwire, not the control: a pull request can edit it. `CODEOWNERS`
  should own it.
- Reason: it uses a protection the repository already enforces (R1), costs an
  owner review on about 16% of pull requests (R4), and needs no new
  infrastructure. Push rulesets are not available for public repositories
  (P5). It also closes the present exposure of `apply-signing` and
  `encoder-version-bump` (R3).

**Residual risks that no option closes:**

- A session that holds a code owner's credentials can approve (design §9).
  Credentials that agents cannot hold would close it.
- An administrator can relax branch protection, `CODEOWNERS` enforcement or the
  environment settings; one administrator is not a code owner (R2). The design
  (§9) puts administrators out of scope; §6 records who they are.
- Hosted runners have passwordless `sudo` (P10). The runner process holds the
  key for the whole job, so any step in the job, before or after the token
  step, can read it from memory. This is acceptable only because every step in
  the job is reviewed workflow text, an action from `actions/` pinned to a
  full SHA, or a bundle file. It limits the job to those steps; a step added
  later joins the key-exposure set and needs the same review.

Optional, not part of the recommendation: required reviewers on
`merge-guard-key` during shadow mode as a second layer. Remove them before
change 3b adds a timed trigger. Agents must not approve them.

### Alternatives and why not

| Alternative | Why not now |
|---|---|
| A dedicated repository for the runner and the key, writable only by the code owners | Strong landing control, but a new repository, a vendored bundle that can drift from axiom-encode, and more settings to verify. Second choice if Max does not want owner review on all of `.github/` |
| OIDC to an external key holder that checks `sub`, `workflow_ref` and `runner_environment` (P8) and returns a scoped token | Keeps the key off the runner and solves selection by claims. It still needs (b) for the token, and adds a hosted service, key custody and its own review. A later upgrade |
| Custom deployment protection rule (GitHub App, public preview) | Needs a hosted service to review and operate |
| Push ruleset that restricts `.github/workflows/**` | Not available for public repositories (P5) |
| A `main`-only branch rule alone | Limits the ref, not the workflow (P2) |

## 5. Admin steps, in order

### Now, independent of the mechanism choice (R3)

These close the present exposure. They do not need Max's choice of mechanism,
and they do not satisfy the guard precondition. Do them in this order.

A. Add required reviewers to `apply-signing`, or remove its secret until #1885
   merges. Verify:
   `gh api repos/TheAxiomFoundation/axiom-encode/environments/apply-signing --jq '[.protection_rules[].type]'`
   includes `required_reviewers`, or
   `gh api repos/TheAxiomFoundation/axiom-encode/environments/apply-signing/secrets --jq .total_count`
   prints `0`.
B. Turn administrator bypass off on `encoder-version-bump`. Verify:
   `gh api repos/TheAxiomFoundation/axiom-encode/environments/encoder-version-bump --jq .can_admins_bypass`
   prints `false`.
C. Merge the `CODEOWNERS` line `/.github/ @PavelMakarchuk @MaxGhenis`. Verify
   with step 2 below.
D. Turn on "Require approval of the most recent reviewable push" for `main`.
   Verify with step 1 below.

### For the guard key

**Before step 1:** Max has chosen; the implementing change (§4, items 1-5)
and the `CODEOWNERS` line have merged; the independent review (§6) found no
open finding. Run each verification and keep its output for the review record.

1. Confirm the landing protection.
   `gh api repos/TheAxiomFoundation/axiom-encode/branches/main/protection --jq '{owners: .required_pull_request_reviews.require_code_owner_reviews, stale: .required_pull_request_reviews.dismiss_stale_reviews, last_push: .required_pull_request_reviews.require_last_push_approval, admins: .enforce_admins.enabled, bypass: .required_pull_request_reviews.bypass_pull_request_allowances}'`
   must print `true`, `true`, `true`, `true` and `null`.
   `gh api repos/TheAxiomFoundation/axiom-encode/rulesets --jq '.[].id' | xargs -I{} gh api repos/TheAxiomFoundation/axiom-encode/rulesets/{} --jq '{name, bypass_actors}'`
   must show no bypass actor.
2. Confirm ownership.
   `gh api repos/TheAxiomFoundation/axiom-encode/codeowners/errors --jq '.errors | length'` must print `0`, and
   `gh api repos/TheAxiomFoundation/axiom-encode/contents/.github/CODEOWNERS --jq .content | base64 -d | grep -x '/.github/ @PavelMakarchuk @MaxGhenis'`
   must print the line.
3. Confirm selection on `main`.
   `git grep -n -i -E 'environment:.*merge-guard' origin/main -- .github/workflows`
   must print one line, in `merge-guard-runner.yml`, and
   `uv run pytest -q tests/test_merge_guard_key_environment.py` must pass.
4. Create or configure environment `merge-guard-key`: deployment branches
   "Selected branches and tags" with one rule, `main` of type branch;
   "Allow administrators to bypass" off; no secrets yet.
   `gh api repos/TheAxiomFoundation/axiom-encode/environments/merge-guard-key --jq '{bypass: .can_admins_bypass, policy: .deployment_branch_policy}'`
   must print `false` and `custom_branch_policies: true`;
   `gh api repos/TheAxiomFoundation/axiom-encode/environments/merge-guard-key/deployment-branch-policies --jq '[.branch_policies[] | {name, type}]'`
   must print `[{"name":"main","type":"branch"}]`.
   Environment `merge-guard` must be absent or empty:
   `gh api repos/TheAxiomFoundation/axiom-encode/environments/merge-guard/secrets --jq .total_count`
   returns 404 or prints `0`.
5. Create the guard App: no webhook; repository permissions checks write,
   pull requests read, contents read, actions read (metadata read is
   automatic); no organisation or account permissions; "Only on this
   account". Install it on `rulespec-us` only.
   `gh api orgs/TheAxiomFoundation/installations --jq '.installations[] | select(.app_slug == "<slug>") | {repository_selection, permissions, events}'`
   must print `selected`, exactly those permissions and no events (this read
   needs an organisation-admin token **[check docs]**). The App's
   installation page must list `rulespec-us` only.
6. Generate one private key. Store it and the client ID, then delete the local
   file.
   `gh secret set MERGE_GUARD_APP_KEY --env merge-guard-key --repo TheAxiomFoundation/axiom-encode < key.pem`
   and
   `gh variable set MERGE_GUARD_APP_CLIENT_ID --env merge-guard-key --repo TheAxiomFoundation/axiom-encode --body <client-id>`.
   `gh secret list --env merge-guard-key --repo TheAxiomFoundation/axiom-encode`
   must list only `MERGE_GUARD_APP_KEY`. Before you delete the local file,
   compute its SHA-256 fingerprint (P11) and confirm that the App's settings
   page shows exactly one key with that fingerprint.
7. Only now set the switch:
   `gh variable set MERGE_GUARD_RUNNER_ENABLED --repo TheAxiomFoundation/axiom-encode --body true`.
   `gh variable get MERGE_GUARD_RUNNER_ENABLED --repo TheAxiomFoundation/axiom-encode`
   must print `true`. An organisation variable with the same name must not
   exist: `gh api orgs/TheAxiomFoundation/actions/variables/MERGE_GUARD_RUNNER_ENABLED`
   must return 404 (organisation-admin token **[check docs]**).
8. Dispatch one `dry-run` on `main`. Its log must show the launcher's module
   list and no install step.

Rollback, in this order: unset `MERGE_GUARD_RUNNER_ENABLED`; disable the
workflow (`gh workflow disable merge-guard-runner.yml --repo TheAxiomFoundation/axiom-encode`),
which also blocks re-runs (P7); delete the environment secret; suspend or
uninstall the App's installation. GitHub refuses to delete an App's only key
(P11); to revoke a key, generate a new one first, then delete the old one.

## 6. What the independent review must check

The review reads the merged code and the live settings, not this document.

1. **Selection.** Every job in every workflow on `main`, and in open pull
   requests about to merge, that names `merge-guard-key` or `merge-guard`; no
   dynamic `environment:`; no cross-repository reusable workflow; no trigger
   on the guard workflow other than `workflow_dispatch` and no caller of it.
   Read the workflows; the tripwire test only supports the reading.
2. **Landing.** Branch protection and every ruleset on `main` (step 1),
   last-push approval included; the `CODEOWNERS` rule for `.github/`, for
   itself, for each bundle file, `statute.py`, the launcher, the new module
   and the tests; who can push, maintain or administer, and which
   administrators are not code owners; bypass lists.
3. **Steps before the token.** Each runs no repository code, persists nothing
   (no background process, no file outside the bundle directory that a later
   step runs), restores no cache, downloads no artifact, and pins its action
   to a full SHA from `actions/`.
4. **Run time.** The flags `-I -S`; `sys.path` holds the bundle directory and
   the standard library only; the bundle directory holds exactly the listed
   files; the launcher's finder; a fixture sweep over every lane, v6 included,
   whose recorded `sys.modules` stays within the list; no function-level
   import outside it.
5. **After the token.** Only the guard step receives the token; every later
   step is a pinned action from `actions/` (P10: any step can read the key
   from memory); the token action revokes the token at job end.
6. **Re-runs.** `git log -S merge-guard-key --oneline origin/main -- .github/workflows`
   lists only reviewed revisions that implement the mechanism.
7. **App and environment.** Installation scope and permissions (step 5), one
   private key with the recorded fingerprint (step 6), branch policy and
   administrator bypass (step 4), `merge-guard` absent or empty, no same-name
   organisation secret or variable.
8. **Runner.** GitHub-hosted, not single-CPU, not self-hosted.
9. **Present exposure.** The "Now" steps A-D are done, with their output.
10. **Residuals.** Record the code-owner-credential, administrator and
    in-job memory residuals (§4) as accepted or not; the review does not close
    them.
