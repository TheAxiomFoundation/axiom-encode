# Merge guard key protection: a recommendation

Status: **recommendation; Max has not accepted it. Nothing is chosen.** This
document answers the open decision in checklist step 1 of
`docs/content-bound-merge-guard.md` §8, questions 1 and 16 of §10, and the
"Activation prerequisite" of issue #2257. It changes no setting, creates no
App, environment, secret or ruleset, and satisfies no activation condition.

**Decision asked of Max.** Accept or reject this answer to checklist step 1:

- **Recommended mechanism, (g'): no guard App and no private key.** An
  organization ruleset "Require workflows to pass before merging" targets
  `rulespec-us` `main`. It names
  `.github/workflows/content-bound-merge-guard.yml` of axiom-encode
  (repository id 1125775893), pinned to a reviewed commit SHA. Its bypass
  list is empty. It starts in `Evaluate` and moves to `Active` later. GitHub
  runs that workflow in `rulespec-us` with the read-only job token of
  `rulespec-us`. The job's conclusion is the required result.
- **Condition.** The recommendation holds only if a live experiment passes
  (`docs/merge-guard-ruleset-experiment.md`). Only an organization owner can
  run it. Until it passes, nothing is chosen for activation.
- **Named fallback if the experiment fails, (b)+(d).** Keep the guard App,
  with the same bundle, and hold its key in a cloud key service as a
  non-exportable signing key that only the guard workflow's OIDC identity may
  use. If no cloud account is acceptable: (b) with the complement of §4.3,
  where one approved but wrong change can leak the key for good.
- **Precondition of `Active`.** A merge queue on `rulespec-us` `main`
  (§4.4, R3), and a decision on question 0 of the design's §10 (R9).

Nobody may store a guard App key or set `MERGE_GUARD_RUNNER_ENABLED`, shadow
mode included, until Max chooses a mechanism and any complement, maintainers
implement them, and a fresh independent review verifies the whole
key-exposure set (§6). Under (g') no key is ever created, so this hold is
kept by never creating one. Under either fallback the hold applies in full.

**Present exposure, independent of the guard (R3):** two keys stored today
have the selection gap this document describes. §5 "Now" lists mitigations
that do not wait for Max's decision. Steps A, B and D are still open
(read 2026-11-17).

## Evidence labels

Every platform claim that this revision adds carries one label.

| Label | Meaning |
|---|---|
| [docs Dn] | GitHub documentation page Dn of the list below, opened on 2026-11-17 for this revision. The text here paraphrases it |
| [docs, per research report — not re-read] | A documentation claim taken from the key-mechanism research report of 2026-11-17. This revision did not open the page |
| [code] | A file, function or test in this repository at this change |
| [live read 2026-11-17] | A read-only `gh api` GET run for this revision, about 15:10 EST |
| [author-reported] | A claim from the research report or an earlier pull request. This revision did not verify it |
| [inferred] | Reasoning. No source states it |
| [experiment Xn] | Behavior that waits for step Xn of the owner's experiment |

Facts P1-P11 and R1-R7 of §2 come from #2267. Its author read the linked
pages and the repository on 2026-11-15. They are [author-reported] here,
except where a later label says otherwise.

Documentation opened on 2026-11-17:

- **D1** [Available rules for rulesets (Enterprise Cloud)](https://docs.github.com/en/enterprise-cloud@latest/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)
- **D2** [Troubleshooting rules](https://docs.github.com/en/enterprise-cloud@latest/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/troubleshooting-rules)
- **D3** [REST: organization rules](https://docs.github.com/en/rest/orgs/rules?apiVersion=2022-11-28)
- **D4** [Creating rulesets for repositories in your organization](https://docs.github.com/en/enterprise-cloud@latest/organizations/managing-organization-settings/creating-rulesets-for-repositories-in-your-organization)
- **D5** [About rulesets](https://docs.github.com/en/enterprise-cloud@latest/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets)
- **D6** [Managing rulesets for repositories in your organization](https://docs.github.com/en/enterprise-cloud@latest/organizations/managing-organization-settings/managing-rulesets-for-repositories-in-your-organization)
- **D7** [REST: repository rules](https://docs.github.com/en/rest/repos/rules?apiVersion=2022-11-28)
- **D8** [Troubleshooting required status checks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/troubleshooting-required-status-checks)
- **D9** [Re-running workflows and jobs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs)
- **D10** [Merging a pull request with a merge queue](https://docs.github.com/en/enterprise-cloud@latest/pull-requests/how-tos/merge-and-close-pull-requests/merging-a-pull-request-with-a-merge-queue)
- **D11** [Managing a merge queue](https://docs.github.com/en/enterprise-cloud@latest/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue)
- **D12** [Contexts reference](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts)
- **D13** [Managing private keys for GitHub Apps](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/managing-private-keys-for-github-apps)
- **D14** [Webhook events and payloads](https://docs.github.com/en/webhooks/webhook-events-and-payloads)
- **D15** [Available rules for rulesets (Free, Pro, Team version)](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)
- **D16** [Permissions of custom organization roles](https://docs.github.com/en/enterprise-cloud@latest/organizations/managing-peoples-access-to-your-organization-with-roles/permissions-of-custom-organization-roles)

## 1. The options

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

The key-mechanism research of 2026-11-17 added these options:

- **(d)** the App key in a cloud key service as a non-exportable signing key.
  The cloud lets only the OIDC identity of the guard workflow use it. It
  needs (b) as the code that runs.
- **(e1)** a separate guard repository that holds the environment secret.
- **(e2)** a reusable workflow in a separate guard repository, with (d).
- **(f)** an external verifier service that holds the key.
- **(g-ref)** no App: an organization ruleset requires the guard workflow,
  pinned to `refs/heads/main`.
- **(g')** as (g-ref), but pinned to a commit SHA. (b) is the code that runs.

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
  keys. The limit of 25 keys and "do not expire" were read again [docs D13].

Rulesets and required workflows:

- **P12** The rule "Require workflows to pass before merging" is configured
  at the organization or enterprise level [docs D1]. The Free, Pro and Team
  version of the same page has no section for it [docs D15]. That the rule
  needs GitHub Enterprise Cloud is therefore [inferred].
- **P13** In the REST body, one entry of the `workflows` rule has `path` and
  `repository_id` (both required), and optional `ref` (branch or tag) and
  `sha` (the commit of the workflow file to use) [docs D3]. That a run then
  takes the file from exactly that commit is [experiment X1]. How the web
  form offers the SHA is not on the pages read; a 2023 GitHub blog post
  describes a pin to branch, tag or SHA
  [docs, per research report — not re-read].
- **P14** A ruleset workflow must name `pull_request`, `pull_request_target`
  or `merge_group`. GitHub ignores every filter (`branches`, `paths`, `types`
  and the others). It starts the workflow on the default activity types only,
  and always on them: `opened`, `synchronize` and `reopened` for
  `pull_request`; `checks_requested` for `merge_group` [docs D1, D2]. No
  review event is in that list.
- **P15** A public workflow can run on any repository of the organization
  [docs D1]. axiom-encode is public [live read 2026-11-17].
- **P16** Ruleset workflows do not run on events that the `GITHUB_TOKEN`
  causes. A rule created while a pull request is open does not start the
  workflow for it; a push, a branch update, or closing and reopening does
  [docs D2].
- **P17** A repository that requires workflows through an organization
  ruleset and uses a merge queue needs the `merge_group` trigger in those
  workflows [docs D11]. The rule blocks direct pushes, so it belongs only on
  branches that change through pull requests [docs D1].
- **P18** Enforcement is `active`, `evaluate` or `disabled`. `evaluate` does
  not enforce and shows on the "Rule Insights" page what would have passed
  or failed. `evaluate` needs GitHub Enterprise [docs D3, D4, D6]. A ruleset
  workflow that passed in `Evaluate` lets a pull request merge after the
  switch to `Active` without a new run [docs D1]. That an `Evaluate` ruleset
  starts the workflow at all is implied by that sentence, not stated:
  [experiment X6].
- **P19** Who can change an organization ruleset. One page says that only
  owners of the organization can edit a ruleset created at the organization
  level [docs D4]. Another says that organization owners and users with the
  permission "Manage organization ref update rules and rulesets" can manage
  them [docs D6]; that permission can be part of a custom organization role
  [docs D16]. So the set is the organization owners plus every holder of
  such a role. Whether TheAxiomFoundation has such a role: the read needs
  the `admin:org` scope and was refused [live read 2026-11-17].
- **P20** A repository administrator can add repository rulesets. GitHub
  aggregates them with the organization's; the result can only be more
  restrictive [docs D4, D5].
- **P21** Bypass is by list. Eligible entries include repository
  administrators, organization owners, roles, teams and Apps [docs D4]. That
  an organization owner is refused when the list is empty: [experiment X9].
- **P22** `GET /repos/{owner}/{repo}/rules/branches/{branch}` returns the
  active rules that apply, from every level. It leaves out rules of
  `evaluate` and `disabled` rulesets. The documented example carries
  `ruleset_source_type`, `ruleset_source` and `ruleset_id` [docs D7]. Its
  shape for a `workflows` rule is not documented: [experiment X14]. Anyone
  with read access can view a repository's active rulesets [docs D5].
- **P23** The organization ruleset endpoints need the organization
  permission "Administration" (write) for a fine-grained token [docs D3].
  With the classic token on the maintainer's machine, the list call returned
  404 and named the missing `admin:org` scope [live read 2026-11-17].
- **P24** A job that a condition skips reports success, and `success`,
  `skipped` and `neutral` satisfy a required check [docs D8]. Whether the
  same holds for a required workflow is [inferred]; the guard workflow has
  no skip path [code: `tests/test_merge_guard_enforce_workflow.py`].
- **P25** A re-run uses the same `GITHUB_SHA` and `GITHUB_REF` and the
  privileges of the first actor. It is possible for 30 days, at most 50
  times, not while the workflow is disabled. People with write access can
  re-run [docs D9]. None of the ruleset pages mentions re-runs [docs D1,
  D2]: whether a re-run updates the rule's result is [experiment X3].
- **P26** A pull request leaves the merge queue when it no longer meets the
  merge requirements or when a queue check fails [docs D10]. "Require merge
  queue" is not available in an organization ruleset [docs D1].
- **P27** `github.workflow_sha` is "the commit SHA for the workflow file".
  The `job` context also has `workflow_ref`, `workflow_sha` and
  `workflow_repository` for the file that defines the job [docs D12]. Their
  values in a ruleset run are [experiment X11].
- **P28** GitHub advises to keep an App key in a key vault as a sign-only
  key, which cannot be read back after upload [docs D13].

Repository (axiom-encode unless stated):

- **R1** Public. `main` requires code-owner review, dismisses stale reviews,
  enforces protection for administrators and has no review bypass
  allowances. It requires no approving review for files without a code owner
  and does not require last-push approval. The "main merge queue" ruleset has
  no bypass actors. The `CODEOWNERS` errors API returns none. Read again:
  code-owner review on, stale dismissal on, administrators enforced, 0
  approvals, `require_last_push_approval: false`
  [live read 2026-11-17].
- **R2** Three administrators and two other accounts can push. The code owners
  are @PavelMakarchuk and @MaxGhenis; one administrator is not a code owner.
  At the #2267 read, `CODEOWNERS` named four workflow files and had no rule
  for the rest of `.github/`. Since #2277 it has the line
  `/.github/ @PavelMakarchuk @MaxGhenis` [live read 2026-11-17]. The encoder
  version bump App opened 28 of the last 200 merged pull requests.
- **R3** Environments, and the present exposure:
  - `production-signing`: required reviewers Max and Pavel, administrator
    bypass off, secrets `AXIOM_ENCODE_APPLY_SIGNING_KEY` and
    `AXIOM_REPO_TOKEN`. Read again: the two reviewers, bypass off, and
    `prevent_self_review: false` [live read 2026-11-17].
  - `apply-signing`: secret `AXIOM_ENCODE_APPLY_SIGNING_KEY`, stored
    2026-11-02, behind a `main` branch rule only, with no required reviewers.
    No workflow on `main` selects it; #1885 would. Read again: one secret of
    that name, branch rule only [live read 2026-11-17].
  - `encoder-version-bump`: secret `ENCODER_VERSION_BUMP_APP_KEY`, stored
    2026-11-06, behind a `main` branch rule only. It is the only environment
    with administrator bypass on. Read again: bypass still on
    [live read 2026-11-17].
  - `generation-approval` and `snap-queue-finalization`: no secrets. No
    `merge-guard` environment.

  **Present exposure.** At the #2267 read a new workflow file had no code
  owner and `main` requires no other approval (R1, R2). Any account that can
  push could then merge a new workflow that runs on `main`, selects
  `apply-signing` or `encoder-version-bump`, and reads the apply key or the
  bump App's key. For the apply key this also skips `production-signing`'s
  per-run approval. The bump App's key does not expire (P11). #2277 gave
  every file under `.github/` a code owner. One person, or a session with
  that person's token, can still push a workflow change to a pull request
  that another account opened and then approve it, because last-push
  approval is off (R1). §5 "Now" lists the mitigations.
- **R4** In the 17 workflows on `main`, every job that selects an environment
  names it statically, no job calls a reusable workflow, and every action is
  pinned to a full commit SHA. In the 60 days to 2026-11-15, 32 of 202 merged
  pull requests changed `.github/workflows/`.
- **R5** The `run` job at #2237 head installs the encoder (`uv sync --locked
  ... --extra api`) and starts `.venv/bin/axiom-encode merge-guard-run`. A static
  scan of that start path (package `__init__.py`, console script, CLI and
  function-level imports) reaches 200 encoder modules and 14 third-party
  top-level packages. The design's run-time trace counted 150 modules.
- **R6** At the #2267 read the guard needed five modules at module level, all
  stdlib-only. Since #2278 and change 3b the bundle is six modules and a
  launcher: `merge_guard_runner.py`, `content_merge_guard.py`,
  `reviewed_pr_admission.py`, `applied_manifest_v6.py`,
  `rulespec_citation_paths.py`, `statute.py` and
  `scripts/merge_guard_launch.py`
  [code: `BUNDLE_MODULES` in `scripts/merge_guard_launch.py`].
- **R7** The lazy edge to `axiom_encode.harness.evals` that #2267 found is
  cut: #2278 moved the path rule into `rulespec_citation_paths.py`
  [code: `tests/test_rulespec_citation_paths.py`].
- **R8** Organization and repositories [live read 2026-11-17]: plan
  `enterprise`; three organization owners (the two code owners and one
  more account); axiom-encode id 1125775893 and rulespec-us id 1114692039,
  both public. `rulespec-us` `main`: required check `validate / validate`
  from app 15368, `strict`, administrators enforced, 0 approvals. No ruleset
  rule applies to `rulespec-us` `main` (the branch-rules read returned an
  empty list). The default job token of `rulespec-us` is read-only.
- **R9** The token on the maintainer's machine, which agent sessions use,
  has the scopes `repo`, `workflow`, `read:org`, `gist`, `delete_repo` and
  `admin:org_hook`. It has no `admin:org` [live read 2026-11-17].

## 3. Evaluation against the key-exposure set

### 3.1 The three options of the design (from #2267)

| Part of the set | (a) code-owner review | (b) reviewed, pinned build | (c) per-run reviewers |
|---|---|---|---|
| Other jobs that select the environment: existing, new, replaced, modified, dynamic | Needs a complement (P2) | Needs a complement (P2) | Covered: every job that references the environment waits (P1) |
| Who can land those workflows | Covered only if the set includes all of `.github/` | Covered by the complement | Landing is open; each approval must catch it |
| Steps before the token, and what they leave behind | Reviewed, but they install and start the whole encoder | Pinned GitHub actions, a sparse checkout and a file copy; no repository code runs | Approver sees a run, not the code its steps execute |
| Builds, installation, dependencies and their hooks | Not closable by review: a `uv.lock` change shows hashes, not package code (R5: 14 packages) | Removed: no install step, and `-S` keeps site-packages off `sys.path` (P9) | Trusted on each approval |
| Interpreter start-up, `__init__.py`, entry point, `signing_broker`, import closure | About 200 modules under owner review, plus `pyproject.toml` and `uv.lock` | Six owned modules and a launcher; stub `__init__.py`, no console script, no `signing_broker` | Trusted on each approval |
| Re-run of an older workflow revision (P7) | Unchanged | Needs a key environment that no earlier revision names | Each re-run waits for approval |
| Fit with change 3b (per head, no person in the loop) | Fits | Fits | One approval per run, or required checks go missing and block merges |
| A session that holds a code owner's or reviewer's token (design §9) | Residual | Residual (through owner review) | Residual on every run (P3) |
| Recurring review load | Every PR that touches the encoder or its lock file | PRs that touch the guard files or `.github/` (about 16% of PRs, R4) | One approval per run |

**Why not (a).** The set it must review is the whole encoder, its lock file and
14 third-party packages with their hooks. Review of a `uv.lock` change does not
review the code of the new versions, so (a) cannot close the dependency part.

**Why not (c).** An approver approves a run, not the code. One approval of a
compromised commit leaks the key for good (P11). Change 3b needs runs without
a person in the loop. GitHub accepts the approval from any session that holds
a reviewer's token (P3).

**Why (b) is the code that runs under every option.** The guard does not need
the encoder package (R6). Running only its files under `-I -S` removes
installation, dependencies, start-up hooks, `__init__.py`, the console script
and `signing_broker` from the job. Change 3b built it (§4.2).

### 3.2 The recommendation and the fallback against K1-K6

The elements: **K1** other jobs that can select the key environment. **K2**
who can land such workflows. **K3** the job's steps, actions and persistence.
**K4** builds, installation, dependencies and hooks. **K5** interpreter
start-up and the import closure. **K6** re-runs of older revisions.

| | (g') ruleset, SHA pin, no key | Fallback (b)+(d): App key in a cloud key service |
|---|---|---|
| K1 | No key and no environment exist, so no job can select one [code: the workflow has no `environment`, `secrets`, `vars` or `id-token`; `tests/test_merge_guard_enforce_workflow.py` fails on each]. The matching question is whether another workflow or App can satisfy the rule with a check of the same name: [experiment X2] | No environment holds a secret. The cloud refuses a workflow whose OIDC claims differ [author-reported; docs, per research report — not re-read] |
| K2 | A merge to axiom-encode `main` changes nothing for the guard until someone with ruleset rights moves the SHA pin [docs D3: the `sha` field; experiment X1]. Who has those rights: P19 | A new workflow gains nothing. A change to the guard workflow file on `main` can sign while it is there, unless the cloud condition also pins the workflow commit [author-reported] |
| K3 | Two actions pinned to full commit SHAs (`actions/checkout`, `actions/setup-python`), a shell step that copies the bundle and checks its digest, and the launcher. The checkout names axiom-encode at `github.workflow_sha`; nothing of the pull request's repository is checked out [code: `.github/workflows/content-bound-merge-guard.yml`]. The value of `github.workflow_sha` in a ruleset run is [experiment X11]; a wrong value cannot run other bytes, because the digest literal is in the pinned file [code] | The same steps, plus the steps that reach the key service. They can ask it to sign during the run [author-reported] |
| K4 | No install step, no `uv`, `pip`, cache or artifact [code: the same test] | The same |
| K5 | `python3 -I -S`, six listed modules, an import refusal for every other `axiom_encode` module, a digest over the eight bundle files [code: `scripts/merge_guard_launch.py`, `BUNDLE_SHA256` in the workflow] | The same, plus about 80 lines of signing code that do not exist yet [author-reported] |
| K6 | A re-run uses the first run's commit [docs D9]. A run of an older head cannot make a newer head green: the verdict must name the event's head [code: `_bind_event_head` in `merge_guard_runner.py`, design case 49]. That the result stays on the old head: [experiment X13] | A re-run of an old revision still matches a condition on `refs/heads/main` [author-reported] |
| One compromised run | One wrong verdict on one head. The job token is read-only (`contents`, `pull-requests`, `checks`, `actions`: read) and ends with the job [code: the workflow's `permissions`] | Can mint App tokens while it runs, about one hour each. Nothing persists [author-reported] |
| A wrong pull request to axiom-encode `main` | No effect before a pin edit (K2) | Forged checks until the revert, plus the life of minted tokens [author-reported] |
| Human action per run | None. One pin edit per guard release | None |

## 4. Recommendation

### 4.1 Recommended: (g'), on the condition of the experiment

Reasons, each with its evidence:

1. **No key exists.** The key-exposure set of checklist step 1 is then empty.
   The worst case of one compromised run is one wrong verdict on one head
   [inferred from the workflow's permissions, code].
2. **The trust root is an organization ruleset.** A repository administrator
   or a `repo`-scope token cannot edit it; the organization owners and
   holders of the ruleset permission can (P19) [docs D4, D6]. A merge to
   axiom-encode `main`, which needs no approval for most files (R1), changes
   nothing until the pin moves [docs D3; experiment X1]. The agent credential
   on the maintainer's machine has no `admin:org` scope and the API refused
   its ruleset read (P23, R9) [live read 2026-11-17].
3. **No human action per run.** One ruleset edit per guard release
   [inferred].
4. **Shadow mode is native.** `Evaluate` enforcement with Rule Insights
   (P18) [docs D4, D6; experiment X6].

What it costs against the App design: no trigger follows a review (R3
below), and the merge queue becomes a precondition of `Active`.

### 4.2 The code that runs: the stdlib guard bundle (b)

Items 1-5 are the #2267 list. Their state at this change:

1. Cut the R7 edge. **Done** (#2278): `rulespec_citation_paths.py`.
2. The bundle: six modules and a launcher with an import allowlist. **Built**
   in change 3b (#2286) [code: `scripts/merge_guard_launch.py`].
3. The key-bearing job (`environment: merge-guard-key`, token step, `python
   -I -S`). **Not built.** It is needed only under the fallbacks. Under (g')
   the enforcing workflow of change 3b replaces it and has no key.
4. Tests: stdlib-only bundle files, a sweep under the launcher, a workflow
   tripwire. **Built** for the enforcing workflow
   [code: `tests/test_merge_guard_enforce_workflow.py`].
5. `CODEOWNERS` owns the bundle files, the launcher, the workflow and the
   tests. **Done** in change 3b [code: `.github/CODEOWNERS`].

### 4.3 Complement: only for a stored key

(g') needs no complement, because no job can select a key that does not
exist. The complement of #2267 stays the recommendation for the keys that
exist today (§5 "Now") and for the second fallback:

- Code-owner review of all of `.github/` in axiom-encode. **Done** (#2277).
- "Require approval of the most recent reviewable push" on `main`. **Open**
  (step D). Without it, one person, or a session with that person's token,
  can push a workflow change to a pull request that the other owner or the
  bump App opened, then approve it as the code-owner review.
- Keep `tests/test_merge_guard_key_environment.py`. It fails if a job other
  than `merge-guard-runner.yml:run` selects `merge-guard` or
  `merge-guard-key`, if a job chooses its environment by expression, or if
  the shadow workflow gets another trigger. It is a tripwire, not the
  control.

### 4.4 Residual risks of (g')

- **R1. Who can edit the ruleset.** The three organization owners, and any
  holder of the ruleset permission (P19), can edit or disable it. An agent
  session that holds an owner's token with `admin:org` could too. Proposed
  standing rule: no agent session gets `admin:org`. GitHub cannot enforce
  that rule.
- **R2. The pin edit is the high-value action.** The pinned code is only as
  good as its review at pin time. The pin also freezes what the guard reads
  from the bundle: the enrolled identities (`TRUSTED_SIGNING_REVIEWERS`) and
  `INFRASTRUCTURE_LANE_ENABLED` [code: both are in files of
  `BUNDLE_MODULES`]. Removing an enrolled identity on axiom-encode `main`
  does not reach the guard until the pin moves [inferred]. Every change to a
  bundle file must also set the new digest literal in the workflow, which a
  code owner reviews [code:
  `test_the_digest_literal_is_the_digest_of_the_working_tree_bundle`].
- **R3. Review changes do not start a ruleset workflow** (P14) [docs D1, D2].
  - (a) After the other owner approves a green candidate, the guard run on
    that head is still red. Someone must re-run it. Whether the re-run
    updates the rule is [experiment X3]. If it does not, someone other than
    the approver must update the branch, or someone closes and reopens the
    pull request (P16) [inferred].
  - (b) **Stale green.** A pass stays green after the approval is dismissed
    or changes are requested. The App design's sweep bounded this to
    minutes. (g') catches it only when the merge-group run evaluates again
    [code: `evaluate_merge_group` calls `evaluate`, which reads the reviews].
    Without a merge queue a dismissed approval does not block the merge.
    **So a merge queue on `rulespec-us` `main` is a precondition of
    `Active`** (rulespec-us#2160, TheAxiomFoundation/.github#155).
  - (c) **The merge-group run does not close the window.** It evaluates once,
    when the group is created, and does not wait
    [code: `_enforce_merge_group` in `merge_guard_runner.py`]. The group
    merges when its other required check ends. rulespec-us#2160 reports 2.5
    to 4 hours for that check [author-reported]. A dismissal in that time is
    not seen, unless GitHub removes the entry by itself; `rulespec-us`
    requires 0 approvals, so nothing native does [inferred;
    experiment X15]. This is the main trade-off against the App design.
- **R4. The wait for CI.** The guard run starts together with
  `validate / validate` and waits for it, at most 60 minutes, then evaluates
  once more [code: `CI_WAIT_SECONDS`, `_wait_for_ci`]. If the check takes
  longer, the run fails closed and someone re-runs it.
- **R5. Unverified platform behavior.** The experiment. If a same-named check
  from another workflow can satisfy the rule (X2), (g') fails outright.
- **R6. Agents under both enrolled identities** (design §9). No option
  changes this.
- **R7. Plan.** The required-workflows rule is documented only for
  Enterprise Cloud, and `evaluate` needs Enterprise (P12, P18). Organization
  rulesets as such are documented for Team and Enterprise [docs D4, D5].
  What happens to the rule if the plan changes is unknown.
- **R8. Supply chain of the job.** A compromised pinned `actions/checkout`
  or `actions/setup-python` commit, or runner image, can forge one verdict
  [inferred].
- **R9. Question 0 blocks `Active`.** With the rule `Active` and
  `INFRASTRUCTURE_LANE_ENABLED = False`, the guard refuses every pin,
  ledger, workflow and tooling pull request on `rulespec-us`
  [code: `R_NO_LANE` in `content_merge_guard.py`]. Max must decide question
  0 of the design's §10 before `Active`.
- **R10. Token-caused events.** The signed commit must reach the candidate
  branch through a person's or an App's push, as today, or no guard run
  starts (P16). Today the signing workflow opens a draft pull request into
  the candidate branch with a stored token, and a person merges it [code:
  step "Push lane branch and open draft pull request" of
  `targeted-signed-reencode.yml`]. That a missing run blocks the merge:
  [experiment X8].
- **R11. Throughput.** The merge-group evaluation requires the group's base
  to be the live `main` tip and the pull request's head to be ahead of it
  [code: `R_MERGE_GROUP_BASE`, `R_BASE_NOT_CURRENT`]. A queue with more than
  one entry in flight fails the later entries. rulespec-us#2160 reports
  about 6 to 9 landings a day for a queue of size 1 [author-reported].

### 4.5 Alternatives and why not

| Option | Why not first |
|---|---|
| (a) code-owner review of the whole set | Cannot close dependencies; about 200 modules (§3.1) |
| (b) + complement of §4.3 (#2267) | No new infrastructure, but a stored key. One approved but wrong change leaks it for good (P11). Second fallback |
| (c) reviewers per run | One click per run; one approval of bad code leaks the key (§3.1) |
| (d) with (b): key in a cloud key service | The key cannot leak, and no complement is needed. It needs a cloud account, a key import and new signing code. A compromised run can still mint tokens for about an hour. First fallback [author-reported] |
| (e1) separate guard repository with the secret | Still a secret that can leak, and a second set of repository settings that a `repo`-scope token can change [author-reported] |
| (e2) reusable workflow in a guard repository, with (d) | Strongest App form; costs a repository and a cloud account [author-reported] |
| (f) external verifier service | It could react to review events. It is a hosted service with its own deploy path and key custody [author-reported] |
| (g-ref) ruleset pinned to `refs/heads/main` | A merge to axiom-encode `main` takes effect at once; `main` needs no approval for most files (R1) |
| Custom deployment protection rule | Needs a hosted service; public preview [author-reported] |
| Push ruleset on `.github/workflows/**` | Not available for public repositories (P5) |
| A `main`-only branch rule alone | Limits the ref, not the workflow (P2) |

## 5. Maintainer steps

Agents perform none of these steps. Each needs repository or organization
settings.

### Now, independent of the decision (R3)

These close the present exposure. They do not need Max's decision, and they
do not satisfy the guard precondition. State on 2026-11-17, about 15:10 EST
[live read 2026-11-17]:

A. Add required reviewers to `apply-signing`, or remove its secret until #1885
   merges. **Open:** the environment has a branch rule only and one secret.
   Verify:
   `gh api repos/TheAxiomFoundation/axiom-encode/environments/apply-signing --jq '[.protection_rules[].type]'`
   includes `required_reviewers`, or
   `gh api repos/TheAxiomFoundation/axiom-encode/environments/apply-signing/secrets --jq .total_count`
   prints `0`. Today: `["branch_policy"]` and `1`.
B. Turn administrator bypass off on `encoder-version-bump`. **Open.** Verify:
   `gh api repos/TheAxiomFoundation/axiom-encode/environments/encoder-version-bump --jq .can_admins_bypass`
   prints `false`. Today: `true`.
C. Merge the `CODEOWNERS` line `/.github/ @PavelMakarchuk @MaxGhenis`.
   **Done** (#2277); the line is on `main`.
D. Turn on "Require approval of the most recent reviewable push" for `main`.
   **Open.** Verify with the first command of "Second fallback", step 1.
   Today: `last_push` is `false`.

### For (g'): the ruleset

Do this only after Max accepts, the experiment passes, and the review of §6
has read the workflow and the bundle at the candidate SHA. The complete
order, with the steps before and after, is the activation checklist of
`docs/content-bound-merge-guard.md` §8.

1. Open Organization settings → Repository → Rulesets → New ruleset → New
   branch ruleset [docs D4].
2. Set these values:
   - Ruleset name: `content-bound-merge-guard`.
   - Enforcement status: `Evaluate`.
   - Bypass list: empty. Add nothing.
   - Target repositories: selected repositories, `rulespec-us` only.
   - Target branches: the default branch.
   - Rule: "Require workflows to pass before merging". Repository
     `axiom-encode`. Workflow file
     `.github/workflows/content-bound-merge-guard.yml`. Pin: the reviewed
     commit SHA, all 40 characters. Select no other rule.
3. Or create it with one request. An organization owner's token with
   `admin:org` sends this body to `POST /orgs/TheAxiomFoundation/rulesets`
   [docs D3 for every field]:

   ```json
   {
     "name": "content-bound-merge-guard",
     "target": "branch",
     "enforcement": "evaluate",
     "bypass_actors": [],
     "conditions": {
       "repository_name": {"include": ["rulespec-us"], "exclude": []},
       "ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}
     },
     "rules": [
       {
         "type": "workflows",
         "parameters": {
           "workflows": [
             {
               "repository_id": 1125775893,
               "path": ".github/workflows/content-bound-merge-guard.yml",
               "sha": "<reviewed 40-hex commit SHA>"
             }
           ]
         }
       }
     ]
   }
   ```

   Save the body as `ruleset.json` and run
   `gh api -X POST orgs/TheAxiomFoundation/rulesets --input ruleset.json`.
   The body gives no `ref`: whether GitHub accepts `sha` alone is
   [experiment X1]. `repository_name` also takes `"protected": true`, which
   prevents renaming the target [docs D3]; rename behavior is
   [experiment X10].
4. Verify, as an organization owner (the call needs `admin:org`, P23):
   `gh api orgs/TheAxiomFoundation/rulesets --jq '.[] | {id, name, enforcement}'`,
   then
   `gh api orgs/TheAxiomFoundation/rulesets/<id> --jq '{enforcement, bypass_actors, conditions, rules}'`.
   It must show `evaluate`, an empty bypass list, the one repository, the
   default branch and one `workflows` entry with the reviewed `sha`.
5. Verify who else can manage organization rulesets (P19): Organization
   settings → Roles. No custom role may hold "Manage organization ref update
   rules and rulesets" unless Max accepts its holders as part of R1.
6. Record the ruleset id, the SHA and the bundle digest (`BUNDLE_SHA256` in
   the workflow at that SHA) in the design's §11.
7. Anyone can check the rule once it is `Active`:
   `gh api repos/TheAxiomFoundation/rulespec-us/rules/branches/main`.
   The call returns active rules only (P22). While the ruleset is in
   `Evaluate` it returns nothing for it; use Rule Insights (Organization
   settings → Repository → Rule insights) [docs D6].

**Switch to `Active`** (activation checklist step 10): edit the ruleset and
set the enforcement status to `Active`, or
`gh api -X PUT orgs/TheAxiomFoundation/rulesets/<id> -f enforcement=active`
[docs D3].

**Rollback.** Set the enforcement to `evaluate` (shadow again) or `disabled`:
`gh api -X PUT orgs/TheAxiomFoundation/rulesets/<id> -f enforcement=disabled`.
Do not delete the ruleset: GitHub documents "Disabled" as the way to switch
a ruleset off for a time, and keeps a ruleset history of 180 days
[docs D6].

**Per guard release (pin procedure).**

1. Take the old SHA from the ruleset and the new candidate SHA from
   axiom-encode `main`.
2. Review
   `git diff <old sha>..<new sha> -- .github/workflows/content-bound-merge-guard.yml scripts/merge_guard_launch.py src/axiom_encode/applied_manifest_v6.py src/axiom_encode/content_merge_guard.py src/axiom_encode/merge_guard_runner.py src/axiom_encode/reviewed_pr_admission.py src/axiom_encode/rulespec_citation_paths.py src/axiom_encode/statute.py`.
   The file list is `SOURCE_FILES` of the launcher plus the workflow. If the
   diff changes that list, review the new files too.
3. Confirm the digest literal. In a checkout of the new SHA:
   `uv run pytest -q tests/test_merge_guard_enforce_workflow.py` must pass.
   It recomputes `BUNDLE_SHA256` from the files [code].
4. Confirm that the two actions are still pinned to full commit SHAs and
   that the workflow has no new step, trigger, permission or condition.
5. Move the pin: edit the ruleset, or `PUT` the full `rules` array with the
   new `sha`. Verify as in step 4 above.
6. Open pull requests keep the result of their last run. A new run starts on
   the next push, update or reopen (P16) [inferred].

### First fallback (b)+(d) — only if the experiment fails

Short form, from the research report [author-reported]. None of it is built
or tested.

1. Create or pick a cloud project with a key service.
2. Create the guard App (no webhook; checks write; pull requests, contents
   and actions read; only on this account). Install it on `rulespec-us` only.
3. Generate one key. Import it into the key service as a non-exportable
   RSA signing key (RS256). Confirm the fingerprint (P11). Delete the local
   file.
4. Let only this OIDC identity sign: owner id 258352533, repository id
   1125775893, ref `refs/heads/main`, a GitHub-hosted runner, and the
   workflow reference of the guard workflow on `main`.
5. Store the App client id and the key name as plain repository variables.
   No environment secret exists.
6. Do steps A-D above. The hold of the status section applies: an
   independent review of the whole key-exposure set comes first (§6).
7. An agent builds the signing module and the workflow change in a reviewed
   pull request before step 3.

### Second fallback (b) + complement — only if no cloud account is acceptable

This is the #2267 proposal. **One approved but wrong change to the guard
files or to `.github/` can leak the key, and a leaked key stays valid until
someone revokes it (P11).**

**Before step 1:** Max has chosen this; item 3 of §4.2 and step D have
landed; the independent review (§6) found no open finding. Run each
verification and keep its output for the review record.

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

### For (g'), at each pin

1. **The pinned file.** The workflow at the candidate SHA: triggers
   `pull_request` and `merge_group` only; read permissions only; no `if`,
   `continue-on-error`, `concurrency`, `environment`, `secrets`, `vars` or
   `id-token`; both actions pinned to full commit SHAs from `actions/`; the
   checkout names axiom-encode at `github.workflow_sha`, sparse, with
   `persist-credentials: false`.
2. **The bundle.** Every file of `SOURCE_FILES` at that SHA; the digest
   literal equals the recomputed digest; the launcher's flags, file list and
   import refusal; no function-level import off the list.
3. **The evaluator's switches at that SHA.** `INFRASTRUCTURE_LANE_ENABLED`,
   `V6_READER_ENABLED`, the enrolled list, `SUCCESS_VERDICTS`.
4. **The ruleset.** One repository, the default branch, one workflow entry
   with that SHA, an empty bypass list, the enforcement status; who can
   manage organization rulesets (owners and roles, P19).
5. **The experiment record.** Every step of
   `docs/merge-guard-ruleset-experiment.md` has a recorded result, and each
   failed step has its consequence applied.
6. **Residuals.** Record R1-R11 of §4.4 as accepted or not. The review does
   not close them.
7. **Present exposure.** The "Now" steps A-D are done, with their output.

### For a fallback with a stored key

1. **Selection.** Every job in every workflow on `main`, and in open pull
   requests about to merge, that names `merge-guard-key` or `merge-guard`; no
   dynamic `environment:`; no cross-repository reusable workflow; no trigger
   on the guard workflow other than `workflow_dispatch` and no caller of it.
   Read the workflows; the tripwire test only supports the reading.
2. **Landing.** Branch protection and every ruleset on `main`, last-push
   approval included; the `CODEOWNERS` rule for `.github/`, for itself, for
   each bundle file, the launcher and the tests; who can push, maintain or
   administer, and which administrators are not code owners; bypass lists.
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
7. **App and environment.** Installation scope and permissions, one
   private key with the recorded fingerprint, branch policy and
   administrator bypass, `merge-guard` absent or empty, no same-name
   organisation secret or variable. For (b)+(d): the cloud condition names
   numeric ids, and only the guard job has `id-token: write`.
8. **Runner.** GitHub-hosted, not single-CPU, not self-hosted.
9. **Present exposure.** The "Now" steps A-D are done, with their output.
10. **Residuals.** Record the code-owner-credential, administrator and
    in-job memory residuals as accepted or not; the review does not close
    them.
