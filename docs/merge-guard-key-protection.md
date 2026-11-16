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
[pending deployments API](https://docs.github.com/en/rest/actions/workflow-runs#review-pending-deployments-for-a-workflow-run),
[reusable workflows](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows),
[ruleset rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets),
[code owners](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners),
[re-runs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs),
[OIDC](https://docs.github.com/en/actions/reference/security/oidc),
[branch policies API](https://docs.github.com/en/rest/deployments/branch-policies),
[hosted runners](https://docs.github.com/en/actions/concepts/runners/github-hosted-runners),
[Python options](https://docs.python.org/3/using/cmdline.html),
[`site`](https://docs.python.org/3/library/site.html)):

- **P1** Environment secrets are available only to jobs that reference the
  environment. With required reviewers, a job cannot read them until one
  reviewer approves. Up to six reviewers; one approval suffices. "Prevent
  self-review" blocks only the user who started the run.
- **P2** Deployment branch and tag rules match the run's `GITHUB_REF`. The page
  does not say that they limit which workflow file selects the environment;
  the design treats them as ref-only. Each policy has a `type`, `branch` or
  `tag`. Administrators can bypass environment protection rules unless the
  environment disallows it.
- **P3** The pending-deployments endpoint accepts any required reviewer with
  read access; classic tokens need the `repo` scope. GitHub cannot tell a
  person from an agent session that holds the reviewer's token (design §9).
- **P4** A job in a called reusable workflow can set `environment` and receives
  that environment's secret when the caller passes it (`secrets: inherit` or by
  name). Which repository's environment applies to a cross-repository call:
  **[check docs]**.
- **P5** Push rulesets with "Restrict file paths" are documented for private
  and internal repositories. axiom-encode is public. Whether they apply to a
  public repository: **[check docs]**. This proposal does not rely on them.
- **P6** Pull requests use the base branch's `CODEOWNERS`. Code owners need
  write access. One owner's approval satisfies "Require review from Code
  Owners". The last matching pattern wins. `CODEOWNERS` must own itself.
- **P7** A re-run is possible for 30 days, uses the original `GITHUB_SHA` and
  `GITHUB_REF`, and is refused while the workflow is disabled. That a re-run
  also uses the original workflow file follows from the same commit; #1885's
  review relied on it; the page does not state it **[check docs]**. Whether a
  re-run reads the environment's current secrets: **[check docs]**; this
  proposal assumes it does.
- **P8** An OIDC token's `sub` contains `environment:<name>` when the job
  references an environment; `workflow_ref`, `workflow_sha`, `event_name` and
  `runner_environment` are separate claims.
- **P9** `python -I` ignores `PYTHON*` variables, the user site and the script
  or current directory. `-S` stops the automatic `site` import, so no
  site-packages directory, no `.pth` file (whose `import` lines run) and no
  `sitecustomize` or `usercustomize` take part in start-up.
- **P10** Each GitHub-hosted runner is a new VM, except single-CPU runners,
  which share a VM. That the VM is discarded after the job: **[check docs]**.
  Whether environment names compare case-insensitively: **[check docs]**.

Repository (axiom-encode unless stated):

- **R1** Public. `main` requires code-owner review, dismisses stale reviews,
  enforces protection for administrators and has no review bypass
  allowances; it requires no approving review otherwise and does not require
  last-push approval. The "main merge queue" ruleset has no bypass actors. The
  `CODEOWNERS` errors API returns none.
- **R2** Three administrators and two other accounts can push. The code owners
  are @PavelMakarchuk and @MaxGhenis. `CODEOWNERS` names four workflow files
  and has no rule for the rest of `.github/`.
- **R3** Environments: `production-signing` (required reviewers Max and Pavel,
  administrator bypass off), `generation-approval`, `apply-signing`,
  `encoder-version-bump` and `snap-queue-finalization`. No `merge-guard`.
  `encoder-version-bump` holds the bump App's key behind a `main` branch rule
  only, so it has the same selection gap that (a) and (b) leave open.
- **R4** In the 17 workflows on `main`, every job that selects an environment
  names it statically, no job calls a reusable workflow, and every action is
  pinned to a full commit SHA. In the 60 days
  to 2026-11-15, 32 of 202 merged pull requests changed `.github/workflows/`.
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
  package. `prepare_signed_backfill.py` changed in 53 commits in 60 days.

## 3. Evaluation against the key-exposure set

| Part of the set | (a) code-owner review | (b) reviewed, pinned build | (c) per-run reviewers |
|---|---|---|---|
| Other jobs that select the environment: existing, new, replaced, modified, dynamic | Needs a complement (P2) | Needs a complement (P2) | Covered: every job that references the environment waits (P1) |
| Who can land those workflows | Covered only if the set includes all of `.github/` | Covered by the complement | Landing is open; each approval must catch it |
| Steps before the token, and what they leave behind | Reviewed, but they install and start the whole encoder | Pinned GitHub actions, a sparse checkout and a file copy; no repository code runs | Approver sees a run, not the code its steps execute |
| Builds, installation, dependencies and their hooks | Not closable by review: a `uv.lock` change shows hashes, not package code (R5: 14 packages) | Removed: no install step, and `-S` keeps site-packages off `sys.path` (P9) | Trusted on each approval |
| Interpreter start-up, `__init__.py`, entry point, `signing_broker`, import closure | About 200 modules under owner review, plus `pyproject.toml` and `uv.lock` | Five owned files after R7 is cut; stub `__init__.py`, no console script, no `signing_broker` | Trusted on each approval |
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
commit, and one approval of a compromised commit leaks the key for good. The
planned sweep (§10 question 2) and change 3b need runs without a person in the
loop. GitHub accepts the approval from any session that holds a reviewer's
token (P3), so "no agent approves" is a process rule that GitHub cannot
enforce. The repository already uses (c) for `production-signing` as a stopgap;
the `CODEOWNERS` header names a "separately reviewed, pinned encoder" as the
way out of it.

**Why (b).** The guard does not need the encoder package (R6). Running only its
files, from the dispatched `main` commit, under `-I -S`, removes installation,
dependencies, start-up hooks, `__init__.py`, the console script and
`signing_broker` from the job. What remains is small, already under
`CODEOWNERS`, and a reviewer can read all of it.

## 4. Recommendation

**Mechanism: (b), as a stdlib guard bundle.** The implementing change, in
axiom-encode after #2237 merges:

1. Cut the R7 edge. Move `RULESPEC_ATOMIC_ROOTS`, `citation_rulespec_path` and
   the pure helpers it needs from `harness/evals.py` into a new stdlib-only
   module, for example `src/axiom_encode/rulespec_citation_paths.py`, that
   `prepare_signed_backfill.py`, `harness/evals.py` and
   `content_merge_guard.py` import. The guard then stops importing
   `prepare_signed_backfill.py`. The bundle becomes the runner, the guard,
   `reviewed_pr_admission.py`, `applied_manifest_v6.py`, the new module and a
   launcher. The existing tests pin unchanged behaviour.
2. Add the launcher (about 30 lines). It parses the three runner arguments,
   puts only the bundle directory on `sys.path`, installs a `sys.meta_path`
   finder that refuses any `axiom_encode` module outside the bundle list, and
   calls `merge_guard_runner.command`. A module outside the list fails the run
   closed.
3. Change the key-bearing job:
   - `environment: merge-guard-key`, a name that no earlier revision of any
     workflow names (P7). Never create `merge-guard`, or keep it without
     secrets.
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
4. Tests: each bundle file is stdlib-only at module and function level; a
   fixture sweep that reaches every lane, run under the launcher with
   `-I -S`, loads only listed modules; the key job contains none of the
   forbidden steps and uses the flags and environment name above.
5. `CODEOWNERS` owns the new module, the launcher and the tests.

**Complement: code-owner review of all of `.github/` in axiom-encode.**

- Add `/.github/ @PavelMakarchuk @MaxGhenis` to `CODEOWNERS`. With R1's
  protection, every new, replaced or modified workflow, any future
  `.github/actions/` and `CODEOWNERS` itself then need a code owner's
  approval, administrators included.
- Keep `tests/test_merge_guard_key_environment.py` (this change). It fails if a
  job other than `merge-guard-runner.yml:run` selects `merge-guard` or
  `merge-guard-key`, if any job chooses its environment by expression, if a job
  calls a reusable workflow from another repository (P4), or if another
  workflow can call the guard workflow. It is a tripwire, not the control: a
  pull request can edit it. `CODEOWNERS` should own it.
- Reason: it uses a protection the repository already enforces (R1), costs an
  owner review on about 16% of pull requests (R4), and needs no new
  infrastructure. Push rulesets are not documented for public repositories
  (P5). It also closes the same gap for `encoder-version-bump` (R3) and for
  `apply-signing` if #1885 lands.

Residual risk that no option closes: a session that holds a code owner's
credentials can approve (design §9). Credentials that agents cannot hold would
close it.

Optional, not part of the recommendation: required reviewers on
`merge-guard-key` during shadow mode as a second layer. Remove them before
change 3b adds a timed trigger. Agents must not approve them.

### Alternatives and why not

| Alternative | Why not now |
|---|---|
| A dedicated repository for the runner and the key, writable only by the code owners | Strong landing control, but a new repository, a vendored bundle that can drift from axiom-encode, and more settings to verify. Second choice if Max does not want owner review on all of `.github/` |
| OIDC to an external key holder that checks `sub`, `workflow_ref` and `runner_environment` (P8) and returns a scoped token | Keeps the key off the runner and solves selection by claims. It still needs (b) for the token, and adds a hosted service, key custody and its own review. A later upgrade |
| Custom deployment protection rule (GitHub App, public preview) | Needs a hosted service to review and operate |
| Push ruleset that restricts `.github/workflows/**` | Not documented for public repositories (P5) |
| A `main`-only branch rule alone | Limits the ref, not the workflow (P2) |

## 5. Admin steps, in order

**Before step 1:** Max has chosen; the implementing change (§4, items 1-5)
and the `CODEOWNERS` line have merged; the independent review (§6) found no
open finding. Run each verification and keep its output for the review record.

1. Confirm the landing protection.
   `gh api repos/TheAxiomFoundation/axiom-encode/branches/main/protection --jq '{owners: .required_pull_request_reviews.require_code_owner_reviews, stale: .required_pull_request_reviews.dismiss_stale_reviews, admins: .enforce_admins.enabled, bypass: .required_pull_request_reviews.bypass_pull_request_allowances}'`
   must print `true`, `true`, `true` and `null`.
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
4. Create environment `merge-guard-key`: deployment branches "Selected branches
   and tags" with one rule, `main` of type branch; "Allow administrators to
   bypass" off; no secrets yet.
   `gh api repos/TheAxiomFoundation/axiom-encode/environments/merge-guard-key --jq '{bypass: .can_admins_bypass, policy: .deployment_branch_policy}'`
   must print `false` and `custom_branch_policies: true`;
   `gh api repos/TheAxiomFoundation/axiom-encode/environments/merge-guard-key/deployment-branch-policies --jq '[.branch_policies[] | {name, type}]'`
   must print `[{"name":"main","type":"branch"}]`;
   `gh api repos/TheAxiomFoundation/axiom-encode/environments/merge-guard` must
   return 404.
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
   must list only `MERGE_GUARD_APP_KEY`. The App settings page must show
   exactly one private key (no REST listing is known **[check docs]**).
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
which also blocks re-runs (P7); delete the secret; delete the key in the App
settings.

## 6. What the independent review must check

The review reads the merged code and the live settings, not this document.

1. **Selection.** Every job in every workflow on `main`, and in open pull
   requests about to merge, that names `merge-guard-key` or `merge-guard`; no
   dynamic `environment:`; no cross-repository reusable workflow; no
   `workflow_call` on the guard workflow and no caller of it. Read the
   workflows; the tripwire test only supports the reading.
2. **Landing.** Branch protection and every ruleset on `main` (step 1); the
   `CODEOWNERS` rule for `.github/`, for itself, for each bundle file, the
   launcher, the new module and the tests; who can push, maintain or
   administer; bypass lists.
3. **Steps before the token.** Each runs no repository code, persists nothing
   (no background process, no file outside the bundle directory that a later
   step runs), restores no cache, downloads no artifact, and pins its action
   to a full SHA from `actions/`.
4. **Run time.** The flags `-I -S`; `sys.path` holds the bundle directory and
   the standard library only; the bundle directory holds exactly the listed
   files; the launcher's finder; a fixture sweep over every lane, v6 included,
   whose recorded `sys.modules` stays within the list; no function-level
   import outside it.
5. **After the token.** Only the guard step receives the token; no later step
   runs repository code; the token action revokes the token at job end.
6. **Re-runs.** `git log -S merge-guard-key --oneline origin/main -- .github/workflows`
   lists only reviewed revisions that implement the mechanism.
7. **App and environment.** Installation scope and permissions (step 5), one
   private key (step 6), branch policy and administrator bypass (step 4), no
   same-name organisation secret or variable.
8. **Runner.** GitHub-hosted, not single-CPU, not self-hosted.
9. **Residual.** Record the code-owner-credential residual (§4) as accepted or
   not; the review does not close it.
