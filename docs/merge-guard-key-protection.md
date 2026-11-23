# Merge guard key protection: a recommendation

Status: **recommendation; Max has not accepted it. Nothing is chosen.** This
document answers the open decision in checklist step 1 of
`docs/content-bound-merge-guard.md` §8, questions 1 and 16 of §10, and the
"Activation prerequisite" of issue #2257. It changes no setting, creates no
App, environment, secret or ruleset, and satisfies no activation condition.

**Decision asked of Max.** Accept or reject this answer to checklist step 1:

- **Recommended mechanism, (g'): no guard App and no private key.** An
  organization ruleset "Require workflows to pass before merging" targets
  `rulespec-us`, branch `refs/heads/main`. It names
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
  (§4.4, R3), a decision on question 0 of the design's §10 with its
  lane live at the pinned commit before the switch (R9; checklist step 10),
  an organization owner's read of the organization's "Runners" setting
  (R12 (c); checklist step 11), and a read of the stored secrets and the
  installed Apps: no stored secret outside an environment with required
  reviewers is a credential of an enrolled identity, an administrator, an
  organization owner or an App that can write checks or contents (R14,
  R12 (f); §5 step 11). On 2026-11-18 that last precondition did not hold.
- **What Max should know before he answers.** (1) The merge queue is a
  repository setting that a `rulespec-us` administrator can turn off; the
  organization ruleset cannot require it (§4.4, R12). (2) Once the v6 writer is
  on, no rollback of the ruleset is durable. Its order is: writer off
  first, then the reader pin of `rulespec-us` reverted, only then the
  ruleset. But once the ruleset is weak, nothing technical protects
  `rulespec-us` from signed v6 bytes: any write account can move the reader
  pin forward again. So the recommended response to a guard problem is to
  repair forward, or to keep the ruleset `Active` with the writer off,
  which blocks every merge (§4.4, R13).
  (3) With the measured CI times, landings are serial at about 3 to 4 a day
  (§4.4, R11). (4) A comment from Max's account is not sufficient evidence of
  this decision; the evidence is an organization owner creating the
  `Evaluate` ruleset, and no checklist can verify who was at the keyboard
  (design §8, step 0). (5) Agent sessions can hold both enrolled
  identities: one session can author and dispatch as one owner and approve
  as the other, and every check passes (§4.4, R6). (6) The experiment does
  not test one imitation: a check with the guard's name that another
  GitHub App publishes (§4.4, R5). Three Apps that are installed on all
  repositories of the organization have the right to write checks
  [live read 2026-11-18], and a `rulespec-us` administrator can add an
  App that an owner installed to `rulespec-us` again after its removal
  (R12 (f)). (7) **A stored token exists today that
  every write collaborator of `rulespec-us` can read**: the repository
  secret `BULK_ENCODE_TOKEN`. If its owner is an enrolled identity, an
  administrator or an organization owner, a collaborator who reads it acts
  as that account, and the controls of this design that rest on that
  account are void [inferred]. This does not depend on the guard; step E
  of §5 "Now" removes it (§4.4, R14). (8) If step X8 (c) of the experiment
  shows that a head can merge while Actions are disabled, a `rulespec-us`
  administrator can merge without the guard by disabling Actions and
  enabling them again (R12 (d)). (9) The experiment needs no merge of this
  stack. An organization owner can run it now, before any review effort
  (§5).

Nobody may store a guard App key or set `MERGE_GUARD_RUNNER_ENABLED`, shadow
mode included, until Max chooses a mechanism and any complement, maintainers
implement them, and a fresh independent review verifies the whole
key-exposure set (§6). Under (g') no key is ever created, so this hold is
kept by never creating one. Under either fallback the hold applies in full.

**Present exposure, independent of the guard (REPO-3):** two keys stored today
have the selection gap this document describes. §5 "Now" lists mitigations
that do not wait for Max's decision. Steps A, B and D are still open
(read 2026-11-17). Step E is a third exposure, of another kind: a stored
token that a workflow of any branch of `rulespec-us` can read. It is open
(read 2026-11-18).

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

Facts P1-P11 and REPO-1 to REPO-7 of §2 come from #2267. Its author read
the linked pages and the repository on 2026-11-15. They are
[author-reported] here, except where a later label says otherwise. #2267
and the earlier revisions of this document named the repository facts R1 to
R10. They are REPO-1 to REPO-10 now, so that "R" with a number means one
thing only: a residual risk of §4.4.

Documentation opened on 2026-11-17:

- **D1** [Available rules for rulesets (Enterprise Cloud)](https://docs.github.com/en/enterprise-cloud@latest/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)
- **D2** [Troubleshooting rules](https://docs.github.com/en/enterprise-cloud@latest/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/troubleshooting-rules)
- **D3** [REST: organization rules](https://docs.github.com/en/rest/orgs/rules?apiVersion=2022-11-28)
- **D4** [Creating rulesets for repositories in your organization](https://docs.github.com/en/enterprise-cloud@latest/organizations/managing-organization-settings/creating-rulesets-for-repositories-in-your-organization)
- **D5** [About rulesets](https://docs.github.com/en/enterprise-cloud@latest/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets)
- **D6** [Managing rulesets for repositories in your organization](https://docs.github.com/en/enterprise-cloud@latest/organizations/managing-organization-settings/managing-rulesets-for-repositories-in-your-organization)
- **D7** [REST: repository rules](https://docs.github.com/en/rest/repos/rules?apiVersion=2022-11-28)
- **D8** [Troubleshooting required status checks](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks)
- **D9** [Re-running workflows and jobs](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/re-run-workflows-and-jobs)
- **D10** [Merging a pull request with a merge queue](https://docs.github.com/en/enterprise-cloud@latest/pull-requests/how-tos/merge-and-close-pull-requests/merging-a-pull-request-with-a-merge-queue)
- **D11** [Managing a merge queue](https://docs.github.com/en/enterprise-cloud@latest/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue)
- **D12** [Contexts reference](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts)
- **D13** [Managing private keys for GitHub Apps](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/managing-private-keys-for-github-apps)
- **D14** [Webhook events and payloads](https://docs.github.com/en/webhooks/webhook-events-and-payloads)
- **D15** [Available rules for rulesets (Free, Pro, Team version)](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets)
- **D16** [Permissions of custom organization roles](https://docs.github.com/en/enterprise-cloud@latest/organizations/managing-peoples-access-to-your-organization-with-roles/permissions-of-custom-organization-roles)
- **D17** [Actions limits](https://docs.github.com/en/actions/reference/limits)
- **D18** [Rate limits for the REST API](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api)
- **D19** [Disabling or limiting GitHub Actions for your organization](https://docs.github.com/en/organizations/managing-organization-settings/disabling-or-limiting-github-actions-for-your-organization)
- **D20** [Events that trigger workflows](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)
- **D21** [Choosing the runner for a job](https://docs.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run/choose-the-runner-for-a-job)

- **D22** [Allowing repository admins to rename branches with organization rulesets](https://docs.github.com/en/organizations/managing-organization-settings/allowing-repository-admins-to-rename-branches-with-organization-rulesets)
- **D23** [Secure use reference](https://docs.github.com/en/actions/reference/security/secure-use)

D17-D21 were opened on 2026-11-17, about 7:00pm EST, for review round 1 of
this document. Live reads of that round carry the label
[live read 2026-11-17] too; they were made between 6:55pm and 7:10pm EST.

D22 and D23 were added on 2026-11-18 for review round 3. They were read in
the source of the documentation (repository `github/docs`, commit
`49a012fa5`), not on the rendered pages; the two links are built from the
paths of the source files. The address of D8 is its current path in that
source; the address used before redirects to it. A label of the form
[docs: REST "..."] names an operation that was read in GitHub's OpenAPI
description on the same day. Live reads of that round carry the label
[live read 2026-11-18].

D24 to D26 were added on 2026-11-18 for review round 4, from the same
source commit. Live reads of round 4 carry the label
[live read 2026-11-18] too.

- **D24** [Deployments and environments](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments).
  It states that an environment secret is available only to a job that
  references the environment, and only after a required reviewer approves
  when the environment requires approval.
- **D25** [Deleting a workflow run](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/delete-a-workflow-run)
- **D26** The text block `data/reusables/apps/repo-admin-install-restriction.md`
  of the documentation source. The page that shows it was not found in
  that clone, so the label is [docs source D26] and no link is given.

The sentence of D23 that R14 uses says that any user with write access to
a repository "has read access to all secrets configured in your
repository".

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
  `sha` (the commit of the workflow file to use) [docs D3]. The rule's
  parameters also have `do_not_enforce_on_create` (boolean, optional)
  [docs D3]; the body of §5 leaves it out, and what GitHub does without it
  on branch creation does not matter for `main`, which exists [inferred].
  That a run then takes the file from exactly that commit is
  [experiment X1]. How the web
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
  or failed. `evaluate` needs GitHub Enterprise [docs D3, D4, D6]. "GitHub
  does not log rule insights until a pull request is merged or a merge is
  attempted" [docs D2]: an open pull request has no Rule Insights row. A ruleset
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
  queue" is not available in an organization ruleset [docs D1]. So the
  queue is a setting of the repository, in the hands of its administrators
  (§4.4, R12). The queue's status-check timeout
  (`check_response_timeout_minutes`) has a maximum of 360 minutes [docs D7].
- **P29** In `conditions.ref_name.include` the REST body takes ref names or
  patterns, and "also accepts ~DEFAULT_BRANCH to include the default
  branch"; the documented example lists `refs/heads/main` [docs D3]. This
  document uses `refs/heads/main`, not `~DEFAULT_BRANCH` (§4.4, R12).
- **P30** An organization can limit self-hosted runners at the repository
  level: "All repositories", "Selected repositories", or "Disabled - self-
  hosted runners cannot be created at the repository level"; runners that
  exist are then listed as "Disabled" and get no new jobs [docs D19]. That
  runners at the organization or enterprise level stay usable under that
  setting is [inferred]: the setting names the repository level only, and
  the page does not state it. The page on
  `runs-on` says a job runs "on any runner that matches all of the
  specified `runs-on` values" [docs D21]. Whether a self-hosted runner
  with the custom label `ubuntu-latest` can take a job that says
  `runs-on: ubuntu-latest` is not stated on the pages read. It is
  [inferred, untested]: the experiment registers no runner (X23), because
  self-hosted runners "should almost never be used for public
  repositories" [docs D23] and the lab repositories are public.
- **P31** A job on a GitHub-hosted runner can run for up to 6 hours. The
  number of concurrent jobs on standard hosted runners is 500 for the
  Enterprise plan and 60 for Team [docs D17]. The job token's rate limit is
  "1,000 requests per hour per repository", and 15,000 "for requests to
  resources that belong to a GitHub Enterprise Cloud account" [docs D18].
- **P32** For an ordinary workflow, a pull request that a workflow updates
  with `GITHUB_TOKEN` gets `pull_request` runs "that require approval"
  [docs D20]. Ruleset workflows "do not run on events triggered by the
  `GITHUB_TOKEN`" [docs D2]. Which of the two a ruleset workflow shows is
  [experiment X8 (a)].
- **P27** `github.workflow_sha` is "the commit SHA for the workflow file".
  The `job` context also has `workflow_ref`, `workflow_sha` and
  `workflow_repository` for the file that defines the job [docs D12]. Their
  values in a ruleset run are [experiment X11].
- **P28** GitHub advises to keep an App key in a key vault as a sign-only
  key, which cannot be read back after upload [docs D13].

Repository (axiom-encode unless stated):

- **REPO-1** Public. `main` requires code-owner review, dismisses stale reviews,
  enforces protection for administrators and has no review bypass
  allowances. It requires no approving review for files without a code owner
  and does not require last-push approval. The "main merge queue" ruleset has
  no bypass actors. The `CODEOWNERS` errors API returns none. Read again:
  code-owner review on, stale dismissal on, administrators enforced, 0
  approvals, `require_last_push_approval: false`
  [live read 2026-11-17].
- **REPO-2** Three administrators and two other accounts can push. The code owners
  are @PavelMakarchuk and @MaxGhenis; one administrator is not a code owner.
  At the #2267 read, `CODEOWNERS` named four workflow files and had no rule
  for the rest of `.github/`. Since #2277 it has the line
  `/.github/ @PavelMakarchuk @MaxGhenis` [live read 2026-11-17]. The encoder
  version bump App opened 28 of the last 200 merged pull requests.
- **REPO-3** Environments, and the present exposure:
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
  owner and `main` requires no other approval (REPO-1, REPO-2). Any account that can
  push could then merge a new workflow that runs on `main`, selects
  `apply-signing` or `encoder-version-bump`, and reads the apply key or the
  bump App's key. For the apply key this also skips `production-signing`'s
  per-run approval. The bump App's key does not expire (P11). #2277 gave
  every file under `.github/` a code owner. One person, or a session with
  that person's token, can still push a workflow change to a pull request
  that another account opened and then approve it, because last-push
  approval is off (REPO-1). §5 "Now" lists the mitigations.
- **REPO-4** In the 17 workflows on `main`, every job that selects an environment
  names it statically, no job calls a reusable workflow, and every action is
  pinned to a full commit SHA. In the 60 days to 2026-11-15, 32 of 202 merged
  pull requests changed `.github/workflows/`.
- **REPO-5** The `run` job at #2237 head installs the encoder (`uv sync --locked
  ... --extra api`) and starts `.venv/bin/axiom-encode merge-guard-run`. A static
  scan of that start path (package `__init__.py`, console script, CLI and
  function-level imports) reaches 200 encoder modules and 14 third-party
  top-level packages. The design's run-time trace counted 150 modules.
- **REPO-6** At the #2267 read the guard needed five modules at module level, all
  stdlib-only. Since #2278 and change 3b the bundle is six modules and a
  launcher: `merge_guard_runner.py`, `content_merge_guard.py`,
  `reviewed_pr_admission.py`, `applied_manifest_v6.py`,
  `rulespec_citation_paths.py`, `statute.py` and
  `scripts/merge_guard_launch.py`
  [code: `BUNDLE_MODULES` in `scripts/merge_guard_launch.py`].
- **REPO-7** The lazy edge to `axiom_encode.harness.evals` that #2267 found is
  cut: #2278 moved the path rule into `rulespec_citation_paths.py`
  [code: `tests/test_rulespec_citation_paths.py`].
- **REPO-8** Organization and repositories [live read 2026-11-17]: plan
  `enterprise`; three organization owners (the two code owners and one
  more account); axiom-encode id 1125775893 and rulespec-us id 1114692039,
  both public. `rulespec-us` `main`: required check `validate / validate`
  from app 15368, `strict`, administrators enforced, 0 approvals. No ruleset
  rule applies to `rulespec-us` `main` (the branch-rules read returned an
  empty list). The default job token of `rulespec-us` is read-only. Read
  again at 7:04pm EST: the branch-rules read still returns `[]`; the default
  branch is `main`; `rulespec-us` has 0 repository-level self-hosted
  runners (`actions/runners`, read with a token that has administrator
  access there); the organization-level runner list needs `admin:org` and
  was refused [live read 2026-11-17].
- **REPO-9** The token on the maintainer's machine, which agent sessions use,
  has the scopes `repo`, `workflow`, `read:org`, `gist`, `delete_repo` and
  `admin:org_hook`. It has no `admin:org` [live read 2026-11-17].
- **REPO-10** Duration of `validate / validate` on `rulespec-us`: of the 40
  newest successful `pull_request` runs of `repository-checks.yml` (created
  2026-11-07 to 2026-11-17 UTC), 38 took 172 to 227 minutes and two took 42 and
  55; the median is 207.5 minutes. No `merge_group` run of that workflow
  exists. Method and command: design §4, "Measured duration of the required
  check" [live read 2026-11-17].

## 3. Evaluation against the key-exposure set

### 3.1 The three options of the design (from #2267)

| Part of the set | (a) code-owner review | (b) reviewed, pinned build | (c) per-run reviewers |
|---|---|---|---|
| Other jobs that select the environment: existing, new, replaced, modified, dynamic | Needs a complement (P2) | Needs a complement (P2) | Covered: every job that references the environment waits (P1) |
| Who can land those workflows | Covered only if the set includes all of `.github/` | Covered by the complement | Landing is open; each approval must catch it |
| Steps before the token, and what they leave behind | Reviewed, but they install and start the whole encoder | Pinned GitHub actions, a sparse checkout and a file copy; no repository code runs | Approver sees a run, not the code its steps execute |
| Builds, installation, dependencies and their hooks | Not closable by review: a `uv.lock` change shows hashes, not package code (REPO-5: 14 packages) | Removed: no install step, and `-S` keeps site-packages off `sys.path` (P9) | Trusted on each approval |
| Interpreter start-up, `__init__.py`, entry point, `signing_broker`, import closure | About 200 modules under owner review, plus `pyproject.toml` and `uv.lock` | Six owned modules and a launcher; stub `__init__.py`, no console script, no `signing_broker` | Trusted on each approval |
| Re-run of an older workflow revision (P7) | Unchanged | Needs a key environment that no earlier revision names | Each re-run waits for approval |
| Fit with change 3b (per head, no person in the loop) | Fits | Fits | One approval per run, or required checks go missing and block merges |
| A session that holds a code owner's or reviewer's token (design §9) | Residual | Residual (through owner review) | Residual on every run (P3) |
| Recurring review load | Every PR that touches the encoder or its lock file | PRs that touch the guard files or `.github/` (about 16% of PRs, REPO-4) | One approval per run |

**Why not (a).** The set it must review is the whole encoder, its lock file and
14 third-party packages with their hooks. Review of a `uv.lock` change does not
review the code of the new versions, so (a) cannot close the dependency part.

**Why not (c).** An approver approves a run, not the code. One approval of a
compromised commit leaks the key for good (P11). Change 3b needs runs without
a person in the loop. GitHub accepts the approval from any session that holds
a reviewer's token (P3).

**Why (b) is the code that runs under every option.** The guard does not need
the encoder package (REPO-6). Running only its files under `-I -S` removes
installation, dependencies, start-up hooks, `__init__.py`, the console script
and `signing_broker` from the job. Change 3b built it (§4.2).

### 3.2 The recommendation and the fallback against K1-K6

The elements: **K1** other jobs that can select the key environment. **K2**
who can land such workflows. **K3** the job's steps, actions and persistence.
**K4** builds, installation, dependencies and hooks. **K5** interpreter
start-up and the import closure. **K6** re-runs of older revisions.

| | (g') ruleset, SHA pin, no key | Fallback (b)+(d): App key in a cloud key service |
|---|---|---|
| K1 | No key and no environment exist, so no job can select one [code: the workflow has no `environment`, `secrets`, `vars` or `id-token`; `tests/test_merge_guard_enforce_workflow.py` fails on each]. The matching question is whether another publisher can satisfy the rule with a check of the same name. For another workflow, a check run made through the API and a commit status: [experiment X2]. For another GitHub App: [inferred, untested], no step tests it (§4.4, R5) | No environment holds a secret. The cloud refuses a workflow whose OIDC claims differ [author-reported; docs, per research report — not re-read] |
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
   axiom-encode `main`, which needs no approval for most files (REPO-1), changes
   nothing until the pin moves [docs D3; experiment X1]. The agent credential
   on the maintainer's machine has no `admin:org` scope and the API refused
   its ruleset read (P23, REPO-9) [live read 2026-11-17].
3. **No human action per run.** One ruleset edit per guard release
   [inferred].
4. **Shadow mode is native.** `Evaluate` enforcement (P18) [docs D4, D6;
   experiment X6]. The evidence of the shadow period is each run's log;
   Rule Insights shows a row only after a merge or a merge attempt
   [docs D2].

What it costs against the App design: no trigger follows a review (R3
below); the merge queue becomes a precondition of `Active`, and the queue
is a repository setting (R12); each job waits for the repository's CI and
holds a runner for that time (R4); landings are serial (R11).

### 4.2 The code that runs: the stdlib guard bundle (b)

Items 1-5 are the #2267 list. Their state at this change:

1. Cut the REPO-7 edge. **Done** (#2278): `rulespec_citation_paths.py`.
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
  holder of the ruleset permission (P19), can edit or disable it. An
  organization owner can also rename a targeted branch without restriction
  [docs D22], which takes `main` out of a rule on `refs/heads/main`. An agent
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
  - (a) A v6 candidate is unapproved when its signed head is pushed, so its
    guard run fails at once (`no_approval`, no wait). After the other owner
    approves the green candidate, hours later, that run is still red, and
    a maintainer must re-run it. A head that is approved before its run
    starts, and every v5 lane, needs no re-run: the run waits for CI (R4).
    Whether the re-run updates the rule is [experiment X3]. If it does not, someone other than
    the approver must update the branch, or someone closes and reopens the
    pull request (P16) [inferred].
  - (b) **Stale green.** A pass stays green after the approval is dismissed
    or changes are requested. The App design's sweep bounded this to
    minutes. (g') catches it only when the merge-group run evaluates again
    [code: `evaluate_merge_group` calls `evaluate`, which reads the reviews].
    Without a merge queue a dismissed approval does not block the merge.
    **So a merge queue on `rulespec-us` `main` is a precondition of
    `Active`** (rulespec-us#2160, TheAxiomFoundation/.github#155). The
    queue is a repository setting that an administrator of `rulespec-us`
    can turn off (R12).
  - (c) **The merge-group run narrows the window to seconds; it does not
    close it.** The run waits until the group's own `validate / validate`
    run has completed with success, for at most 5 hours, and evaluates once
    after that [code: `_enforce_merge_group`, `_wait_for_group_ci` in
    `merge_guard_runner.py`]. rulespec-us#2160 reports 2.5 to 4 hours for
    that check [author-reported]; a dismissal or a `CHANGES_REQUESTED`
    review in that time now fails the guard's group job. What remains is the
    time between the evaluation's last read of the reviews and GitHub's
    merge: the rest of the evaluation, the end of the job and the queue's
    own delay, seconds if the queue merges promptly [inferred]. In that time
    a dismissal is not seen, unless GitHub removes the entry by itself;
    `rulespec-us` requires 0 approvals, so nothing native does [inferred].
    Unverified: that the group's check has the expected name and
    `merge_group` run on the group commit (if not, every group fails,
    closed), that the queue merges only after the guard job has ended, and
    how long after [experiment X15]. Cost: the guard job runs for the whole
    validation of the group and ends last, so the queue's status-check
    timeout must cover it (design §6).
- **R4. The wait for CI, and what it costs.** The guard run of a pull
  request starts together with `validate / validate`. When its first
  verdict is `ci_not_green` it waits for that check, at most 5 hours, with a
  poll every 180 seconds, then evaluates once more [code: `CI_WAIT_SECONDS`,
  `CI_POLL_SECONDS`, `_wait_for_ci`]. The check takes 172 to 227 minutes in
  38 of the 40 newest successful runs (§2, REPO-10) [live read 2026-11-17], so
  a normal run ends inside the wait. The wait was 60 minutes before review
  round 2 of #2286; with it every signed head ended `fail` and needed a
  manual re-run. The guard run of a merge group waits for the group's own
  run of that check, at most 5 hours, before its one evaluation [code:
  `MERGE_GROUP_WAIT_SECONDS`]. If the check takes longer, the run fails
  closed and a maintainer re-runs it. The job timeout is 345 minutes, above
  the longest run (333.5 minutes) and below the 6 hours a hosted job may
  run [code: the workflow and its tripwire test; docs D17].
  Cost: one hosted runner job for each head that reaches the CI rule, for
  the time of its CI, 3 to 4 hours; 20 requests an hour of the repository's
  job token for each waiting job, at most 50 in an hour that also holds an
  evaluation. The token's limit is 1,000 requests an hour for the
  repository [docs D18]. So 20 jobs that wait at the same time can use the
  whole limit in their costliest hour, and 50 use it with their polls
  alone; a rate-limited read fails the job (closed) and a maintainer
  re-runs it later. Whether the 15,000 limit of Enterprise Cloud applies to
  a job token of `rulespec-us` is not verified [docs D18; design §4, "What
  the wait costs"]. These figures count one evaluation as 30 requests,
  which is the largest fixture figure for one unit. A v5 train of `u` units
  costs about `11 + 3u` requests for each evaluation, and the cap of 350
  requests for a run admits about 37 units; a larger train fails closed
  (design §4, "Request volume").
- **R4b. The inert path is the one assumption that fails open.** In
  axiom-encode itself the runner ends with success without an evaluation.
  It takes that path only when `GITHUB_REPOSITORY` and the event payload
  both name axiom-encode [code: `_require_source_event`]. No documentation
  states what these values are in a ruleset run for another repository
  [experiment X18]. If such a run carried axiom-encode in all of them, every
  pull request of the target would pass unjudged. Hold: under the `Evaluate`
  ruleset a pull request with no lane must show a **failed** guard job
  before `Active`, at the commit that the ruleset pins then (design §4,
  "The inert path"; checklist steps 7, 10 and 11).
- **R5. Unverified platform behavior.** The experiment. If a same-named check
  from another workflow can satisfy the rule (X2), (g') fails outright.
  **X2 does not cover a same-named check that another GitHub App
  publishes.** That such a check cannot satisfy the rule is
  [inferred, untested]. The reason: the lab creates no App. X2 (d) creates
  its check run with the job token, so from the same GitHub Actions app
  that the ruleset's own run uses; that GitHub treats a check of a foreign
  App the same way is an inference. What would test it: a throwaway GitHub
  App with the right to write checks, installed on the lab target, that
  creates a successful check run named `lab-guard` on an unapproved pull
  request. That needs an App and a key, which the experiment excludes; an
  owner decides it separately. Who could use the gap: an App that is
  installed on `rulespec-us` with the right to write checks, and whoever
  can make such an App act. Which Apps are installed there is an owner's
  read, and who can install one is R12 (f).
- **R6. Agents under both enrolled identities** (design §9). No option
  changes this. The guard authenticates GitHub accounts, not people. If
  agent sessions hold the credentials of both enrolled owners, one session
  can author and dispatch as one owner and approve as the other, and every
  check passes. The guard records which accounts acted; it cannot prove
  that a person read the bytes. Only approval credentials that agents
  cannot hold would change this (design §9).
- **R7. Plan.** The required-workflows rule is documented only for
  Enterprise Cloud, and `evaluate` needs Enterprise (P12, P18). Rulesets as
  such are documented for the Team and Enterprise plans; rulesets at the
  organization level are documented for the Enterprise plan [docs D5].
  What happens to the rule if the plan changes is unknown.
- **R8. Supply chain of the job.** A compromised pinned `actions/checkout`
  or `actions/setup-python` commit, or runner image, can forge one verdict
  [inferred].
- **R9. Question 0 blocks `Active`.** With the rule `Active` and
  `INFRASTRUCTURE_LANE_ENABLED = False`, the guard refuses every pin,
  ledger, workflow and tooling pull request on `rulespec-us`
  [code: `R_NO_LANE` in `content_merge_guard.py`]. Max must decide question
  0 of the design's §10 before `Active`. The lane must not first give
  verdicts when it already enforces: it goes live in its own numbered step
  (checklist step 10), with a review cycle, a pin move while the ruleset is
  still `Evaluate`, a second shadow observation and the R4b hold repeated
  at the new pin.
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
  one entry in flight fails the later entries, and each landing makes every
  other open candidate stale. A stale candidate needs a branch update, a
  full pull-request CI run (2.9 to 3.8 hours, §2 REPO-10) and then the group's
  CI run (2.5 to 4 hours [author-reported, rulespec-us#2160]). Landings are
  therefore serial at about 5.4 to 7.8 hours each: **about 3 to 4 a day**,
  not the 6 to 9 that rulespec-us#2160 reports for the queue alone
  [inferred]. Under an `Active` ruleset the infrastructure pull requests
  (48 of 50 in the census [author-reported]) use the same queue, and under
  the proposed lane each needs an approval by the other enrolled owner.
- **R12. What an administrator of `rulespec-us` can still change.** The
  organization ruleset is out of a repository administrator's reach (P19,
  P20). Four settings are not, and a fifth is documented as closed:
  - (a) **The merge queue.** It is a repository setting; an organization
    ruleset cannot require it (P26). An administrator can turn it off with
    no review. The precondition of `Active` then no longer holds, the
    stale-green window of R3 (b) is open again, and nothing blocks or
    alarms. Detection is one read-only call that anyone can make:
    `gh api repos/TheAxiomFoundation/rulespec-us/rules/branches/main --jq '[.[].type]'`
    must contain `merge_queue` and `workflows` (design §8, "Periodic
    read-only health check"). The writer's lock of draft #2289 re-checks
    both rules at every admission and revalidation of a signing run, so a
    missing queue stops new signing runs. It cannot see a queue that is
    turned off after a candidate was signed, and it does not run for the
    v5 lane or the infrastructure lane [author-reported, draft #2289].
    **A bypass actor on the repository ruleset that holds the queue
    merges without the merge-group evaluation, and the rule still shows in
    `rules/branches/main`** [inferred from the documented bypass model,
    docs D4; not tested live]. The lock of draft #2289 cannot see it: it
    does not read bypass lists [author-reported, review round 3].
    Detection is a second read:
    `gh api repos/TheAxiomFoundation/rulespec-us/rulesets/<id> --jq .bypass_actors`
    must print `[]` (§5 step 10). GitHub returns that field only to a
    caller with write access to the ruleset [docs D7]. So an administrator
    of `rulespec-us` runs it; an anonymous read cannot.
  - (b) **The default branch.** With `~DEFAULT_BRANCH` the rule would
    follow a changed default branch and leave `main` with no guard
    [docs D3; inferred]. The ruleset therefore names `refs/heads/main`
    (P29). The runner judges `main` only [code: `_enforce_merge_group`].
    What a default-branch change does to the rule is [experiment X21].
  - (c) **Self-hosted runners.** The job says `runs-on: ubuntu-latest`. If
    a repository-level self-hosted runner with that label can take the job
    (P30), an administrator who registers one decides the job's result.
    Whether it can is [inferred, untested]: the experiment registers no
    runner, and X23 records the documentation only. `rulespec-us` had none
    on 2026-11-17 (§2 REPO-8). So the organization setting is a
    precondition of `Active` in every case: "Disabled" for repository-level
    self-hosted runners, or "Selected repositories" without `rulespec-us`
    [docs D19]. Only an owner can read that setting, and a change of it
    acts on every repository of the organization. It is an owner's act of
    the activation checklist (step 11; §5 step 9), never a step of the
    experiment.
  - (d) **Actions.** Disabling Actions in the target: [experiment X8 (c)].
    If a head with no guard run can merge while Actions are disabled, an
    administrator merges without the guard: Actions off, merge, Actions on
    again. Nothing blocks it [inferred]. Detection is one read, which
    shows the state at its own time only:
    `gh api repos/TheAxiomFoundation/rulespec-us/actions/permissions --jq .enabled`
    must print `true` [live read 2026-11-18: `true`]. It is a row of the
    health check and a precondition of `Active` (design §8).
  - (e) **The name of the branch (documented as closed).** A rename of
    `main` would take the branch out of a rule on `refs/heads/main`. A
    repository administrator cannot rename a branch that an organization
    ruleset targets while the organization setting "Allow repository
    administrators to rename branches protected by organization rules" is
    off, and it is off by default for an existing organization. With it
    on, the new name must still be targeted by the same rulesets
    [docs D22]. An owner reads the setting (§5 step 9); the lab tries the
    rename [experiment X21].

  Three more acts of an administrator are outside that count:
  - (f) **Installing a GitHub App.** A repository administrator can
    install a GitHub App in the organization when the App asks for no
    organization permission and not for the "Repository administration"
    permission, and only on repositories that this administrator
    administers; an organization owner can restrict that
    [docs source D26]. **Also, a repository administrator can add an App that an organization owner has already installed to the repositories that this administrator administers, whatever the App's permissions are**
    [docs source D26]. So removing an installed App from `rulespec-us`
    does not last against one administrator. It lasts only when an
    organization owner restricts the installation of Apps by repository
    administrators, which the same text names. So a
    `rulespec-us` administrator alone can install an App that has the
    right to write checks there, which is the untested case of R5.
    Precondition of `Active`: an organization owner reads the
    installations (§5 step 11), and the read is repeated at each health
    check (design §8); the call needs an owner's account
    [docs: REST "List app installations for an organization"]. State on
    2026-11-18 [live read 2026-11-18, with the maintainer's token, whose
    account is an owner]: four Apps are installed; three of them have all
    repositories and `checks: write`, and two of those also
    `statuses: write`. Their names are not written here; the owner records
    them. Whether the organization restricts installation by repository
    administrators was not read.
  - (g) **Renaming or transferring the repository.** The rename is
    [experiment X10]. A transfer out of the organization is in no
    experiment step and was not read in the documentation; a repository
    outside the organization is outside its rulesets [inferred].
- **R13. After the v6 writer is on, a rollback of the ruleset is a last
  resort and is not durable; it has an order.** With the writer on and the guard not required, an enrolled
  dispatcher lands an unreviewed candidate, and any write account lands the
  signed v6 bytes of an open pull request (the switch rule of the design's
  §8). The writer's lock stops new signing runs only. So every weakening of
  the ruleset or the queue after checklist step 13 comes last: first
  `V6_WRITER_ENABLED` back to `False` on encoder `main`; then no run of
  `targeted-signed-reencode.yml` queued, waiting for the
  `production-signing` approval or in progress, and the reader pin of
  `rulespec-us` reverted (or, as the weaker form, no open pull request
  with a signed v6 manifest); only then the ruleset or the queue.
  **A signing run that was dispatched before the writer-off commit still
  has the writer on.** It executes its own commit, and a re-run of it
  executes that commit too, for 30 days [docs D9; code:
  `targeted-signed-reencode.yml` compares `HEAD` with `GITHUB_SHA`]. So no
  enrolled owner may re-run such a run or approve it for
  `production-signing`; nothing technical enforces that.
  **GitHub offers no transaction between the check and the ruleset
  change.** Closing a pull request does not remove its signed bytes; when
  a signed v6 manifest exists that has not landed, the order asks for the
  reader pin revert (design §8, the rule at the top of the checklist, with
  the commands and the case of a broken guard). Before checklist step 13
  the rollback is free. An owner who sets the ruleset to `Disabled` first
  creates exactly the forbidden state; nothing technical prevents it.
  **This order is not a durable rollback. Once the ruleset is weak,
  nothing technical protects `rulespec-us` from signed v6 bytes.** The
  encoder pin is read from the checkout of the pull request itself
  [code: job `workflow-toolchain` of `repository-checks.yml`, `rulespec-us`
  at `ddad6d63`], and `main` of `rulespec-us` requires 0 approvals and no
  code-owner review [live read 2026-11-18]. So any account with write
  access can move the reader pin forward again in the same pull request
  that lands signed v6 bytes [inferred]. Not tested: whether validation
  refuses a pull request that moves the pin and adds a manifest together.
  "Revert the reader pin" is the pinned-mode form: if `rulespec-us`
  verifies with encoder `main` (verifier-main mode, draft
  rulespec-us#2059), no reader pin exists, and the same act is a reviewed
  change that sets `V6_READER_ENABLED` back to `False` on encoder `main`
  [author-reported, pin trial 2026-11-18, rulespec-us#2304; interim]
  (design §8, step 9).
  **The recommended response to a guard problem after step 13 is
  therefore not to weaken the ruleset**: (a) repair forward, with a pin
  move to a fixed guard while the ruleset is `Active`; or (b) turn the
  writer off and keep the ruleset `Active`, which blocks every merge of
  `rulespec-us` when the guard cannot pass (the fail-closed state).
  (c) Weakening the ruleset is the last resort. It reopens the exposure
  that existed before the guard, now for signed v6 bytes, and three
  standing conditions then hold as rules for people: no pin to an encoder
  commit with the reader on merges while the ruleset is weak; for 30 days
  after the writer-off commit no enrolled owner re-runs or approves an
  older signing run, or a maintainer deletes those runs after saving the
  run list and the logs [docs D25]; and
  signed v6 bytes that never landed stay valid indefinitely [inferred].
  **Durable options, not designed:** enforced code-owner review of
  `.axiom/workflow-toolchain.toml` on `rulespec-us` before the ruleset is
  weakened, or a new signing key.
- **R14. A credential that is stored where a workflow of a branch can read
  it.** Any user with write access to a repository can read every secret
  that is configured in it [docs D23; the sentence is under "Evidence
  labels"]. A write collaborator pushes a branch with a workflow that reads a repository
  secret, or an organization secret that the repository can use; no merge
  and no review is needed. The collaborator then acts as the owner of that
  credential. **If the owner is an enrolled identity, an administrator or
  an organization owner, every control of this design that rests on that
  identity is void for that collaborator** [inferred from the documented
  secret semantics; not tested]. That GitHub cannot restrict a repository
  secret to one workflow file is [inferred]: no page read offers it. An
  environment secret is released only to a job that names the environment,
  after the environment's rules pass [docs D24]; a required reviewer is
  such a rule, and a branch policy `main` keeps a workflow of another
  branch out, but not another workflow file on `main` (REPO-3).
  **State on 2026-11-18** [live read 2026-11-18; names and dates only, no
  call returns a value]:
  - `rulespec-us`, repository secrets: `BULK_ENCODE_TOKEN` (created
    2026-07-06), `OPENAI_API_KEY`, `SUPABASE_DB_URL`. No environment.
    Organization secrets that it can use: `R2_CORPUS_READ_ACCESS_KEY_ID`
    and `R2_CORPUS_READ_SECRET_ACCESS_KEY`. `bulk/README.md` of
    `rulespec-us` describes `BULK_ENCODE_TOKEN` as a token with the scopes
    `repo` and `workflow`; no workflow on its `main` (`ddad6d63`) uses it [code].
  - axiom-encode, repository secrets: `ANTHROPIC_API_KEY`,
    `OPENAI_API_KEY`, `AXIOM_ENCODE_SUPABASE_SECRET_KEY`,
    `AXIOM_ENCODE_SUPABASE_URL`. Environment secrets: `production-signing`
    (required reviewers, branch `main`) holds
    `AXIOM_ENCODE_APPLY_SIGNING_KEY` and `AXIOM_REPO_TOKEN`;
    `apply-signing` (branch `main` only) holds
    `AXIOM_ENCODE_APPLY_SIGNING_KEY`; `encoder-version-bump` (branch `main`
    only) holds `ENCODER_VERSION_BUMP_APP_KEY`. A workflow on a branch of
    axiom-encode needs no merge either: the code-owner rule for `.github/`
    guards `main`, not a branch.
  - The list of all organization secrets needs `admin:org`; the
    maintainer's token got HTTP 403.

  **Not read:** the owner of any of these credentials, the real scopes of
  `BULK_ENCODE_TOKEN` (only the sentence of the README), and whether it is
  still valid. If it is a classic token of an enrolled owner with the
  scopes that the README names, a collaborator who reads it can approve a
  pull request, dispatch the signing workflow and approve its
  `production-signing` deployment as that owner [inferred]. **This
  exposure exists today, and it does not depend on the guard**: today's
  promotion lane also rests on the approval of an enrolled owner. Step E of
  §5 "Now" removes it; §5 step 11 is the precondition of `Active` and of
  change 6.

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
| (g-ref) ruleset pinned to `refs/heads/main` | A merge to axiom-encode `main` takes effect at once; `main` needs no approval for most files (REPO-1) |
| Custom deployment protection rule | Needs a hosted service; public preview [author-reported] |
| Push ruleset on `.github/workflows/**` | Not available for public repositories (P5) |
| A `main`-only branch rule alone | Limits the ref, not the workflow (P2) |

## 5. Maintainer steps

Agents perform none of these steps. Each needs repository or organization
settings.

### Now, independent of the decision (REPO-3)

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
E. Remove the stored token `BULK_ENCODE_TOKEN` of `rulespec-us` (R14).
   **Open** [live read 2026-11-18]. A maintainer finds the account that
   owns the token, revokes it in that account's token settings, and
   deletes the repository secret. Deleting the secret alone does not
   revoke a token that someone has already read. **The existing token is
   revoked in every case, also when a workflow still needs one**: moving
   the same credential elsewhere leaves every copy that was already read
   valid. If a token is still needed, a maintainer documents why, issues
   a new one with the least scope and stores that one under the rule of
   "Create the ruleset", step 11 (no credential of a privileged identity
   outside an accepted, verified environment). This exposure exists today and is independent of
   the guard: every write collaborator of `rulespec-us` can read the
   secret through a workflow of a branch [docs D23; inferred]. Not read:
   its owner, its real scopes and its validity. Verify:
   `gh api repos/TheAxiomFoundation/rulespec-us/actions/secrets --jq '[.secrets[].name]'`
   does not list `BULK_ENCODE_TOKEN`. Today it prints
   `["BULK_ENCODE_TOKEN","OPENAI_API_KEY","SUPABASE_DB_URL"]`. The same
   maintainer states for each other name of R14 whose credential it is.

### For (g'): the ruleset

Do this only after Max accepts, the experiment has passed as checklist step
3 defines it, and the review of §6 has read the workflow and the bundle at
the candidate SHA. An agent must not perform any step of this section: it
must not create, edit or delete the ruleset, and it must not hold an
`admin:org` token. The complete
order, with the steps before and after, is the activation checklist of
`docs/content-bound-merge-guard.md` §8.

**The experiment does not wait for any merge.** It depends on no pull
request of this stack: it uses only the two throwaway repositories, its
own lab workflow files and the one lab ruleset. An organization owner can
run it at any time from now. A failed blocking step is then known before
review effort is spent. It must have passed before the review of §6 at
the candidate SHA (checklist step 5).

1. Open Organization settings → Repository → Rulesets → New ruleset → New
   branch ruleset [docs D4].
2. Set these values:
   - Ruleset name: `content-bound-merge-guard`.
   - Enforcement status: `Evaluate`.
   - Bypass list: empty. Add nothing.
   - Target repositories: selected repositories, `rulespec-us` only.
   - Target branches: the one ref `refs/heads/main`, as an include
     pattern. Do **not** choose the form's "default branch" target: that is
     `~DEFAULT_BRANCH`, and the rule would follow a changed default branch
     (R12 (b)). The wording of the web form is not on the pages read; the
     owner checks the result with the read of step 4 [inferred].
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
       "ref_name": {"include": ["refs/heads/main"], "exclude": []}
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
   `ref_name.include` takes ref names; the documented example lists
   `refs/heads/main` [docs D3]. The body gives no `ref` in the workflow
   entry: `ref` is optional in the documented schema [docs D3], and whether
   GitHub accepts `sha` alone is recorded in the set-up of the experiment
   (runbook §1.3; class adjusting, run again). If the API needs `ref`, three places
   change together: this body, its pin test
   (`test_the_ruleset_bodies_target_main_by_name_and_start_in_evaluate`)
   and lock 3 of draft #2289, which refuses an entry with `ref`
   [author-reported, review round 3]. `repository_name` also takes
   `"protected": true`, which prevents renaming the target [docs D3];
   rename behavior is [experiment X10]. The schema also has a
   `repository_id` condition, which a rename cannot break [docs D3]. Which
   of the two the web form stores is not on the pages read; the read of
   step 4 shows it. Either is acceptable when it selects `rulespec-us`
   (id 1114692039) and nothing else.
4. Verify, as an organization owner (the call needs `admin:org`, P23):
   `gh api orgs/TheAxiomFoundation/rulesets --jq '.[] | {id, name, enforcement}'`,
   then
   `gh api orgs/TheAxiomFoundation/rulesets/<id> --jq '{enforcement, bypass_actors, conditions, rules}'`.
   It must show `evaluate`, an empty bypass list, the one repository, the
   one ref `refs/heads/main` and one `workflows` entry with the reviewed
   `sha`.
5. Verify who else can manage organization rulesets (P19): Organization
   settings → Roles. No custom role may hold "Manage organization ref update
   rules and rulesets" unless Max accepts its holders as part of R1.
6. Record the ruleset id, the SHA and the bundle digest (`BUNDLE_SHA256` in
   the workflow at that SHA) in the design's §11.
7. Before the switch to `Active`, confirm the hold of the inert path, at
   the commit that the ruleset pins then: a `rulespec-us` pull request with
   no lane shows a failed `content-bound-merge-guard` run under this
   `Evaluate` ruleset. Its report line names `TheAxiomFoundation/rulespec-us`
   as `repository`, has a `verdict` that is not `null` with the reason
   `no_lane_for_guarded_change`, and has no `"inert"` key. The key
   `verdict` alone proves nothing: the inert report and every error report
   carry it with `null` [code: `enforce_command`]. The same reading applies
   to the report of a `merge_group` run in the queue trial (R4b; checklist
   steps 7, 8 and 10 have the commands). A passed run there stops the
   activation.
8. Anyone can check the rule once it is `Active`:
   `gh api repos/TheAxiomFoundation/rulespec-us/rules/branches/main --jq '[.[].type]'`
   must contain `workflows` and `merge_queue`. The call returns active rules
   only (P22). While the ruleset is in `Evaluate` it returns nothing for
   it; then the evidence is each run's log in the Actions tab of
   `rulespec-us`. Rule Insights (Organization settings → Repository → Rule
   insights) shows a row only after a pull request is merged or a merge is
   attempted [docs D2, D6].
9. **Precondition of `Active`: an organization owner verifies the runner
   and rename settings (R12 (c), (e)).** Organization settings → Actions →
   General → "Runners" must be "Disabled", or "Selected repositories"
   without `rulespec-us` [docs D19]. The same read through the API, with
   `admin:org` [docs: REST "Get self-hosted runners settings for an
   organization"]:
   `gh api orgs/TheAxiomFoundation/actions/permissions/self-hosted-runners --jq .enabled_repositories`
   must print `none`, or `selected` with `rulespec-us` not in the list of
   `.../self-hosted-runners/repositories`. No organization runner group
   that `rulespec-us` can use may hold a runner with the label
   `ubuntu-latest`. The organization setting "Allow repository
   administrators to rename branches protected by organization rules"
   must be off [docs D22]. A non-owner with administrator access to
   `rulespec-us` can read the repository level only:
   `gh api repos/TheAxiomFoundation/rulespec-us/actions/runners --jq .total_count`
   must print `0` [live read 2026-11-17: `0`]. This check is required in
   every case: whether such a runner could take the job is
   [inferred, untested], because the experiment registers no runner
   (X23). **If the "Runners" setting has another value, the switch to
   `Active` waits.** A change of that setting acts on every repository of
   the organization: runners that exist get no new jobs [docs D19]. It is
   the owner's own decision, outside the experiment, and no agent makes
   it.
10. **Precondition of `Active`: the merge-queue ruleset of `rulespec-us`
    has an empty bypass list (R12 (a)).** An administrator of
    `rulespec-us` runs:
    `gh api repos/TheAxiomFoundation/rulespec-us/rules/branches/main --jq '.[] | select(.type == "merge_queue") | .ruleset_id'`,
    then
    `gh api repos/TheAxiomFoundation/rulespec-us/rulesets/<id> --jq .bypass_actors`,
    which must print `[]`. GitHub returns the field only to a caller with
    write access to the ruleset [docs D7]; `null` is not a pass. Repeat it
    at each health check (design §8).
11. **Precondition of `Active` and of change 6: the stored secrets and the
    installed Apps (R14, R12 (f)).** An administrator lists the names of
    the repository secrets, of the organization secrets that the
    repository can use, and of the environment secrets, for `rulespec-us`
    and for axiom-encode:
    `gh api repos/TheAxiomFoundation/<repo>/actions/secrets --jq '[.secrets[].name]'`,
    `gh api repos/TheAxiomFoundation/<repo>/actions/organization-secrets --jq '[.secrets[].name]'`,
    `gh api repos/TheAxiomFoundation/<repo>/environments --jq '[.environments[] | {name, rules: [.protection_rules[].type]}]'`,
    and for each environment
    `gh api repos/TheAxiomFoundation/<repo>/environments/<name>/secrets --jq '[.secrets[].name]'`.
    An organization owner lists all organization secrets, with `admin:org`:
    `gh api orgs/TheAxiomFoundation/actions/secrets --jq '[.secrets[] | {name, visibility}]'`,
    and the installed Apps:
    `gh api orgs/TheAxiomFoundation/installations --jq '.installations[] | {app_slug, repository_selection, permissions}'`.
    **None of the secrets may be a credential of an enrolled identity, of
    an administrator of `rulespec-us`, of an organization owner, or of an
    App that can write checks or contents on `rulespec-us`. An environment
    is no general exception** (review of head `dd8bf4fa2`): the presence
    of `required_reviewers` proves little, because administrators can
    bypass environment protection rules unless the environment disallows
    it (P2), and a repository administrator can change an environment's
    reviewers and rules. Such a credential may stay in an environment only
    when Max accepts it by name as a residual AND these effective controls
    are read and recorded:
    `gh api repos/TheAxiomFoundation/<repo>/environments/<name> --jq '{name, can_admins_bypass, deployment_branch_policy, rules: [.protection_rules[] | {type, prevent_self_review, reviewers: [.reviewers[]? | {type, login: (.reviewer.login // .reviewer.slug)}]}]}'`
    must show `can_admins_bypass: false`, a `required_reviewers` rule with
    `prevent_self_review: true` whose reviewers are named people outside
    the actors of the threat model (no agent session holds their token),
    and a deployment branch policy; and
    `gh api repos/TheAxiomFoundation/<repo>/environments/<name>/deployment-branch-policies --jq '[.branch_policies[].name]'`
    must print `["main"]`. Against a repository administrator even that is
    a residual (the administrator can edit the environment), which is why
    the acceptance is by name. A name does not show the
    owner of a credential: a maintainer states it for each secret, and the
    names with that statement go into the design's §11. Each installed App
    that can write checks on `rulespec-us` is removed from it, or Max
    accepts it by name as a residual (R5). A removal alone does not last
    against a repository administrator (R12 (f)): it needs the owner's
    restriction of installations too. `apply-signing` and `encoder-version-bump` of
    axiom-encode hold secrets with a branch policy and no required
    reviewers [live read 2026-11-18], so their secrets are judged like
    repository secrets, and the precondition is not met for them until a
    maintainer states whose credential each is. On 2026-11-18 this precondition
    did not hold (R14, R12 (f)).

**Switch to `Active`** (activation checklist step 11): edit the ruleset and
set the enforcement status to `Active`, or
`gh api -X PUT orgs/TheAxiomFoundation/rulesets/<id> -f enforcement=active`.
The endpoint and its optional fields are [docs D3]. That a `PUT` with only
`enforcement` keeps `rules`, `conditions` and `bypass_actors` is [inferred];
so read the ruleset back after the switch, as an owner:
`gh api orgs/TheAxiomFoundation/rulesets/<id> --jq '{enforcement, bypass_actors, conditions, rules}'`
must show `active`, an empty bypass list, the one repository, the one ref
and the one workflow entry with the reviewed `sha`. If a field is missing,
send the whole body of step 3 with `"enforcement": "active"`. Immediately
before the switch, the queue must still be required: the command of step 8
contains `merge_queue`. The owner's reads of step 9 and the bypass read of
step 10 must also hold at that moment. So must the reads of step 11, and
`gh api repos/TheAxiomFoundation/rulespec-us/actions/permissions --jq .enabled`
must print `true` (R12 (d)).

**Rollback, and its order.** Set the enforcement to `evaluate` (shadow
again) or `disabled`:
`gh api -X PUT orgs/TheAxiomFoundation/rulesets/<id> -f enforcement=disabled`,
and read the ruleset back as above.
Do not delete the ruleset: GitHub documents "Disabled" as the way to switch
a ruleset off for a time, and keeps a ruleset history of 180 days
[docs D6].

> **Once the v6 writer is on (activation checklist step 13), this rollback
> comes last.** First set `V6_WRITER_ENABLED` back to `False` in a reviewed
> change on encoder `main`. Then confirm that no run of
> `targeted-signed-reencode.yml` is queued, waiting for the
> `production-signing` approval or in progress:
> `gh run list --repo TheAxiomFoundation/axiom-encode --workflow targeted-signed-reencode.yml --limit 200 --json databaseId,status,headSha,createdAt --jq '[.[] | select(.status != "completed")]'`
> must print `[]`; the listing must reach back more than 30 days, so raise
> `--limit` if it does not. A signing run that was dispatched before the writer-off
> commit still has the writer on, and a re-run of it executes that commit
> for 30 days [docs D9]: no enrolled owner may re-run or approve one. Then
> **revert the reader pin of `rulespec-us`, so that the reader is off in
> the verifier mode that its validation uses. This step is required
> whenever a signed v6 manifest is outstanding or its disposition is
> unknown**: on an open pull request, on a closed one, or on any branch. A
> closed pull request does not remove the signed bytes from its branch,
> and the enabled reader admits a valid signature with `owner_review:
> none` [code: `tests/test_applied_manifest_v6.py`]. The pin revert may be
> left out only when the inventory of the design's rollback order, point 2
> (iii), accounts for every signing run since checklist step 13 and shows
> that none of its manifests exists outside `main`; an inventory with one
> unknown entry is "unknown". Only then set the ruleset to
> `evaluate` or `disabled`, move its pin back, or turn the merge queue off
> (R13). GitHub offers no transaction between the check and the ruleset
> change. Before checklist step 13 the rollback is free.
>
> **After checklist step 13 this rollback is a last resort, and it is not
> durable.** Once the ruleset is weak, nothing technical protects
> `rulespec-us` from signed v6 bytes: any account with write access can
> move the reader pin forward again in the pull request that lands them
> [inferred; R13]. Do not weaken the ruleset for a guard problem. Repair
> forward (a pin move to a fixed guard while the ruleset is `Active`), or
> turn the writer off and keep the ruleset `Active`: it then blocks every
> merge, which is the fail-closed state. If the ruleset is weakened in
> spite of that, the three standing conditions of R13 hold for as long as
> it is weak.

**Per guard release (pin procedure).** A pin move is needed when a bundle
file changes what the guard does: the evaluator, the runner, the launcher,
`TRUSTED_SIGNING_REVIEWERS`, `SUCCESS_VERDICTS` or
`INFRASTRUCTURE_LANE_ENABLED`. It is not needed for change 2b or change 6:
the guard reads neither switch nor `GUARD_WORKFLOW`
[author-reported, activation preparation 2026-11-17].

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
5. **After change 6 (activation checklist step 13) only: the lock constant
   first.** The writer's lock holds the set of pinned SHAs that it accepts
   (`GUARD_WORKFLOW.shas`); a pin outside the set makes the writer refuse
   every promotion. So a pin move is three acts in this sequence
   [author-reported, activation preparation 2026-11-17; draft #2289]:
   1. a reviewed commit on encoder `main` **adds** the new SHA to the set
      (the set then holds the old and the new pin; the new SHA is an
      earlier commit than the commit that records it);
   2. an organization owner **moves the pin** (step 6);
   3. a reviewed commit **removes** the old SHA from the set.
   The set never keeps an older pin: an old SHA in it would let the writer
   accept a ruleset that someone moved back to an older guard. Each of the
   two commits changes `reviewed_pr_admission.py`, a bundle file, so each
   sets a new `BUNDLE_SHA256`; neither moves the pin by itself. Before
   change 6 no constant names the pin, and this step is skipped.
6. Move the pin by **updating the stored ruleset**. **Never send the
   creation body of "Create the ruleset", step 3, for a pin move**: that
   body sets `"enforcement": "evaluate"`, so sent again with a new `sha`
   after checklist step 11 it switches enforcement off, while signed v6
   candidates stay eligible for validation (review of head `dd8bf4fa2`:
   the body of step 3 with only the SHA changed still says `evaluate`).
   The verification of step 4 above is for the creation only; it expects
   `evaluate` and is wrong here. In the web form: open the existing
   ruleset, change only the commit of the workflow entry, save. Or, as an
   organization owner, read, change one value, write, compare:

   ```sh
   gh api orgs/TheAxiomFoundation/rulesets/<id> > before.json
   jq --arg sha "<new 40-hex commit SHA>" '{name, target, enforcement, bypass_actors, conditions, rules} | .rules[0].parameters.workflows[0].sha = $sha' before.json > move.json
   jq -e '(.rules | length) == 1 and (.rules[0].parameters.workflows | length) == 1 and .rules[0].type == "workflows"' before.json
   gh api -X PUT orgs/TheAxiomFoundation/rulesets/<id> --input move.json > after.json
   jq -S '{name, target, enforcement, bypass_actors, conditions, rules} | .rules[0].parameters.workflows[0].sha = "PIN"' before.json > before.cmp
   jq -S '{name, target, enforcement, bypass_actors, conditions, rules} | .rules[0].parameters.workflows[0].sha = "PIN"' after.json > after.cmp
   diff before.cmp after.cmp && jq -r '[.enforcement, .rules[0].parameters.workflows[0].sha] | @tsv' after.json
   ```

   The third command must print `true` (one rule, one workflow entry; if
   not, stop and edit by hand). `diff` must print nothing: the enforcement,
   the targeting (`conditions`, `target`), the bypass list and the rest of
   the rule are what they were. The last line must show the enforcement
   that the ruleset had before the move, **`active` from checklist step 11
   on**, and the new SHA. If it shows `evaluate` or `disabled` after
   step 11, the gate is off: switch it back to `Active` at once ("Switch to
   `Active`" above) and treat the interval as a weakening of the ruleset
   (R13). Then read the active rule as a non-owner sees it: the first two
   rows of the health check (design §8) must still list `workflows` and
   show the new SHA. The jq filters are exercised on a stored-ruleset
   fixture by a test [code: `tests/test_merge_guard_runner.py`]; that the
   API accepts the body that the filter builds from its own `GET` response
   is [experiment X19].
7. Open pull requests keep the result of their last run. A new run starts on
   the next push, update or reopen (P16) [inferred]. Whether an older result
   still satisfies the rule after the pin moved, and which file a
   `merge_group` run then uses, is [experiment X19]. If an old result still
   counts, a pin move that removes an enrolled identity does not reach an
   open green pull request until its merge-group run.

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
6. Do steps A-E above. The hold of the status section applies: an
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
   `V6_READER_ENABLED`, the enrolled list, `SUCCESS_VERDICTS` (while the
   infrastructure lane is off it holds three reasons; the lane's two success
   reasons enter it only in the change that switches the lane on), the wait
   constants (`CI_WAIT_SECONDS`, `CI_POLL_SECONDS`,
   `MERGE_GROUP_WAIT_SECONDS`, `MERGE_GROUP_POLL_SECONDS`,
   `MAX_GROUP_CHECK_RUNS`, `MAX_ENFORCE_REQUESTS`) against the job's
   `timeout-minutes`, and the inert path (`_require_source_event`).
4. **The ruleset.** One repository, the one ref `refs/heads/main` (not
   `~DEFAULT_BRANCH`), one workflow entry with that SHA, an empty bypass
   list, the enforcement status; who can manage organization rulesets
   (owners and roles, P19).
5. **The experiment record.** Every step of
   `docs/merge-guard-ruleset-experiment.md` has a recorded result, the
   experiment has passed as the design's checklist step 3 defines it, and
   each failed adjusting step has its named change merged, and each
   failed step of the class "adjusting, run again" also has its second
   run with `pass`.
6. **The repository settings that the ruleset cannot hold (R12).** The
   merge queue is required on `main` of `rulespec-us`
   (`rules/branches/main` lists `merge_queue`); the default branch is
   `main`; no repository-level self-hosted runner exists, and the
   organization setting for them is "Disabled" or excludes `rulespec-us`
   (owner's check, §5 step 9); the branch-rename setting is off (the same
   step); the merge-queue ruleset has an empty bypass list (§5 step 10);
   Actions are enabled; the stored secrets and the installed Apps are the
   recorded ones, and none is a credential that §5 step 11 excludes.
   The job runs on a GitHub-hosted runner: read "Runner name" and "Runner
   group name" in the log of one ruleset run.
7. **The order of a rollback (R13).** If the v6 writer is on, no document
   or runbook in use tells anyone to weaken the ruleset or the queue first,
   and each names the signing runs that started before the writer-off
   commit. Each says that the rollback is a last resort and is not
   durable.
8. **Residuals.** Record R1-R14 of §4.4 as accepted or not. The review does
   not close them.
9. **Present exposure.** The "Now" steps A-E are done, with their output.

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
9. **Present exposure.** The "Now" steps A-E are done, with their output.
10. **Residuals.** Record the code-owner-credential, administrator and
    in-job memory residuals as accepted or not; the review does not close
    them.
