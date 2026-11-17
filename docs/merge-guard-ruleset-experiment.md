# Merge guard: the ruleset experiment

Status: **designed, not run.** The recommendation in
`docs/merge-guard-key-protection.md` holds only if this experiment passes.
Until an organization owner runs it and records the results, nothing is
chosen for activation.

- **Who.** An organization owner of TheAxiomFoundation. Organization
  rulesets need owner rights. An agent must not run it, and no agent session
  may receive a token with the `admin:org` scope for it.
- **A second account.** Several steps need an approving review. GitHub does
  not let the author of a pull request approve it. Use a second account with
  write access to the target repository. For X16 it must not be an
  administrator of the target.
- **Time.** About one to two hours.
- **What it creates.** Two public throwaway repositories and one
  organization ruleset. It creates no secret, no GitHub App and no key. It
  touches neither `axiom-encode` nor any `rulespec-*` repository.
- **Evidence labels** are those of `docs/merge-guard-key-protection.md`;
  D1-D16 are its documentation list. Every "expected" below is unverified
  until the step runs.

## 1. Set up

### 1.1 The source repository

1. Create the public repository `TheAxiomFoundation/guard-lab-source` with a
   README. Leave Actions on.
2. Add the file `.github/workflows/lab-guard.yml` with the content below.
   Commit it to `main`. This is commit **A**. Record its full SHA:
   `gh api repos/TheAxiomFoundation/guard-lab-source/commits/main --jq .sha`.
3. Record the repository id:
   `gh api repos/TheAxiomFoundation/guard-lab-source --jq .id`.

```yaml
name: lab-guard

# Throwaway. The lab copy of the shape of content-bound-merge-guard.yml:
# two triggers, read permissions, one job, the job token only.
# It passes only when the pull request has an effective APPROVED review.

on:
  pull_request:
  merge_group:

permissions:
  contents: read
  pull-requests: read
  checks: read
  actions: read

jobs:
  guard:
    name: lab-guard
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - name: Print the run identity
        env:
          MARKER: "A"
          EVENT_NAME: ${{ github.event_name }}
          REPOSITORY: ${{ github.repository }}
          COMMIT: ${{ github.sha }}
          REF: ${{ github.ref }}
          WORKFLOW_REF: ${{ github.workflow_ref }}
          WORKFLOW_SHA: ${{ github.workflow_sha }}
          JOB_WORKFLOW_REF: ${{ job.workflow_ref }}
          JOB_WORKFLOW_SHA: ${{ job.workflow_sha }}
          JOB_WORKFLOW_REPOSITORY: ${{ job.workflow_repository }}
          ACTOR: ${{ github.actor }}
          TRIGGERING_ACTOR: ${{ github.triggering_actor }}
          RUN_ID: ${{ github.run_id }}
          RUN_ATTEMPT: ${{ github.run_attempt }}
        run: |
          set -euo pipefail
          echo "marker=$MARKER"
          echo "github.event_name=$EVENT_NAME"
          echo "github.repository=$REPOSITORY"
          echo "GITHUB_REPOSITORY=$GITHUB_REPOSITORY"
          echo "github.sha=$COMMIT"
          echo "github.ref=$REF"
          echo "github.workflow_ref=$WORKFLOW_REF"
          echo "github.workflow_sha=$WORKFLOW_SHA"
          echo "job.workflow_ref=$JOB_WORKFLOW_REF"
          echo "job.workflow_sha=$JOB_WORKFLOW_SHA"
          echo "job.workflow_repository=$JOB_WORKFLOW_REPOSITORY"
          echo "github.actor=$ACTOR"
          echo "github.triggering_actor=$TRIGGERING_ACTOR"
          echo "run=$RUN_ID attempt=$RUN_ATTEMPT"

      - name: Print the event payload fields
        run: |
          set -euo pipefail
          jq '{
            top_level_keys: keys,
            action,
            repository: .repository.full_name,
            number,
            pull_request: (.pull_request // null | if . == null then null else
              {number, head_sha: .head.sha, base_ref: .base.ref} end),
            merge_group_keys: (.merge_group // null | if . == null then null else keys end),
            merge_group: (.merge_group // null | if . == null then null else
              {head_sha, base_sha, head_ref, base_ref} end)
          }' "$GITHUB_EVENT_PATH"

      - name: Check out the source repository at the workflow commit
        uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
        with:
          repository: TheAxiomFoundation/guard-lab-source
          ref: ${{ github.workflow_sha }}
          path: lab-source
          persist-credentials: false
          sparse-checkout-cone-mode: false
          sparse-checkout: |
            .github/workflows/lab-guard.yml

      - name: Print what the checkout holds
        run: |
          set -euo pipefail
          echo "checked out commit=$(git -C lab-source rev-parse HEAD)"
          grep -n 'MARKER: ' lab-source/.github/workflows/lab-guard.yml | head -n 1

      - name: Read public axiom-encode with the job token
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          set -euo pipefail
          for path in \
            repos/TheAxiomFoundation/axiom-encode/actions/runs/44150203163 \
            repos/TheAxiomFoundation/axiom-encode/commits/fad29b3ed4019fe7042552a7958542dfcb41e4e0
          do
            code="$(curl -s -o /dev/null -w '%{http_code}' \
              -H "Authorization: Bearer $GH_TOKEN" \
              -H "Accept: application/vnd.github+json" \
              -H "X-GitHub-Api-Version: 2022-11-28" \
              "https://api.github.com/$path")"
            echo "status=$code path=$path"
          done

      - name: Pass only with an effective approval
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          set -euo pipefail
          if [ "$GITHUB_EVENT_NAME" = "merge_group" ]; then
            ref="$(jq -r '.merge_group.head_ref' "$GITHUB_EVENT_PATH")"
            number="$(printf '%s\n' "$ref" | sed -n \
              's#^refs/heads/gh-readonly-queue/main/pr-\([0-9][0-9]*\)-[0-9a-f]\{40\}$#\1#p')"
          else
            number="$(jq -r '.pull_request.number' "$GITHUB_EVENT_PATH")"
          fi
          case "$number" in
            ''|*[!0-9]*) echo "the event names no pull request"; exit 1 ;;
          esac
          echo "pull request=$number"
          reviews="$RUNNER_TEMP/reviews.json"
          code="$(curl -s -o "$reviews" -w '%{http_code}' \
            -H "Authorization: Bearer $GH_TOKEN" \
            -H "Accept: application/vnd.github+json" \
            -H "X-GitHub-Api-Version: 2022-11-28" \
            "https://api.github.com/repos/$GITHUB_REPOSITORY/pulls/$number/reviews?per_page=100")"
          echo "reviews status=$code"
          [ "$code" = "200" ] || exit 1
          approvals="$(jq '[group_by(.user.id)[]
            | map(select(.state == "APPROVED" or .state == "CHANGES_REQUESTED" or .state == "DISMISSED"))
            | sort_by(.id) | last
            | select(. != null and .state == "APPROVED")] | length' "$reviews")"
          echo "effective approvals=$approvals"
          [ "$approvals" -ge 1 ]
```

The token reaches two steps through `env` only. No step prints it. Do not add
`set -x`.

### 1.2 The target repository

1. Create the public repository `TheAxiomFoundation/guard-lab-target` with a
   README. Leave Actions on. Give the second account write access.
2. Add these three files to `main`.

`.github/workflows/lab-slow.yml` (a second required check that takes three
minutes, for X15):

```yaml
name: lab-slow
on:
  pull_request:
  merge_group:
permissions:
  contents: read
jobs:
  slow:
    name: lab-slow
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: sleep 180
```

`.github/workflows/lab-rerun.yml` (re-runs one run with the job token, for
X16):

```yaml
name: lab-rerun
on:
  workflow_dispatch:
    inputs:
      run_id:
        description: The run to re-run
        required: true
        type: string
permissions:
  actions: write
jobs:
  rerun:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - env:
          GH_TOKEN: ${{ github.token }}
          RUN_ID: ${{ inputs.run_id }}
        run: |
          set -euo pipefail
          case "$RUN_ID" in ''|*[!0-9]*) exit 1 ;; esac
          gh run rerun "$RUN_ID" --repo "$GITHUB_REPOSITORY"
```

`.github/workflows/lab-token-push.yml` (pushes an empty commit with the job
token, for X8):

```yaml
name: lab-token-push
on:
  workflow_dispatch:
    inputs:
      branch:
        description: The pull request branch to push to
        required: true
        type: string
permissions:
  contents: write
jobs:
  push:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6.0.2
        with:
          ref: ${{ inputs.branch }}
      - run: |
          set -euo pipefail
          git config user.name lab
          git config user.email lab@example.invalid
          git commit --allow-empty -m "Empty commit pushed with the job token"
          git push
```

### 1.3 The organization ruleset

Create it in the web form (Organization settings → Repository → Rulesets →
New ruleset → New branch ruleset [docs D4]) or with the request below.

| Field | Value |
|---|---|
| Name | `guard-lab` |
| Enforcement status | `Evaluate` (X6 changes it to `Active`) |
| Bypass list | empty |
| Target repositories | selected: `guard-lab-target` only |
| Target branches | the default branch |
| Rule | "Require workflows to pass before merging": repository `guard-lab-source`, file `.github/workflows/lab-guard.yml`, pinned to the SHA of commit A |

```json
{
  "name": "guard-lab",
  "target": "branch",
  "enforcement": "evaluate",
  "bypass_actors": [],
  "conditions": {
    "repository_name": {"include": ["guard-lab-target"], "exclude": []},
    "ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}
  },
  "rules": [
    {
      "type": "workflows",
      "parameters": {
        "workflows": [
          {
            "repository_id": 0,
            "path": ".github/workflows/lab-guard.yml",
            "sha": "<SHA of commit A>"
          }
        ]
      }
    }
  ]
}
```

Replace `0` with the id of `guard-lab-source`. Save the body as
`guard-lab.json` and run
`gh api -X POST orgs/TheAxiomFoundation/rulesets --input guard-lab.json`.
Record the returned `id`. Record whether the web form offers a SHA pin, and
whether the API accepts `sha` with no `ref`. If it refuses, add
`"ref": "refs/heads/main"` and record that both were needed.

## 2. Steps

Do them in this order. For each step, record the result in the table of §4:
`pass`, `fail` or `not run`, with the run URL or the command output.

### X6. `Evaluate` mode runs the workflow and does not block

- **Procedure.** With the ruleset in `Evaluate`, open pull request **PR-1**
  in the target (any one-line change). Do not approve it.
- **Pass.** A `lab-guard` run starts in `guard-lab-target`, fails (no
  approval), and the pull request can still be merged. Rule Insights
  (Organization settings → Repository → Rule insights) shows the evaluation.
  Do not merge.
- **Record.** The run URL; the check name on the pull request; a
  screenshot or the text of the Rule Insights row.
- **If it fails.** No native shadow mode. The shadow period of the design
  then needs another form (a throwaway target, or a short `Active` window).
  It does not end (g').

Now set the ruleset to `Active`. Record whether PR-1 becomes blocked at once
or only after the next run.

### X1. The SHA pin runs exactly that file

- **Procedure.** In the source, change `MARKER: "A"` to `MARKER: "B"` and
  commit to `main` (commit **B**). Do not edit the ruleset. Push a new commit
  to PR-1.
- **Pass.** The new run prints `marker=A`.
- **Record.** The printed marker; the SHA of A and of B.
- **If it fails.** The pin does not fix the file. A merge to axiom-encode
  `main` would then change the guard at once, and `main` needs no approval
  for most files. (g') fails: use the fallback.

### X11. What `github.workflow_sha` and `github.workflow_ref` are

- **Procedure.** Read the "Print the run identity" and "Print what the
  checkout holds" steps of the X1 run.
- **Pass.** `github.workflow_sha` is the SHA of commit A, the checkout
  succeeds at that commit, and its file shows `MARKER: "A"`.
  `github.workflow_ref` names `guard-lab-source`.
- **Record.** All printed values, the three `job.workflow_*` values included
  (they may be empty).
- **If it fails.** The real workflow checks out its bundle at
  `github.workflow_sha` and compares a digest literal, so it fails closed
  [code]. It then needs a reviewed change that takes the commit from the
  value that does name the source commit, and a new run of this step.

### X12. The check name and the commit it attaches to

- **Procedure.** `gh api repos/TheAxiomFoundation/guard-lab-target/commits/<PR-1 head>/check-runs --jq '.check_runs[] | {name, head_sha, app: .app.slug, status, conclusion}'`.
  Also read the name in the pull request's merge box and in Rule Insights.
- **Pass.** One check run named `lab-guard` (the job name) on the pull
  request's head SHA.
- **Record.** The command output and the name the merge box shows.
- **If it fails.** If the result attaches to another commit (for example the
  test merge commit), the design's head binding and case 49 need a new
  reading before `Active`.

### X4. The target's job token reads the public source of truth

- **Procedure.** Read the "Read public axiom-encode with the job token" step
  of any run, and the checkout step of X11.
- **Pass.** `status=200` for the Actions run and for the commit. The
  checkout of the public source repository succeeds with the job token.
- **Record.** The two status lines.
- **If it fails.** A v6 unit needs the signing run of axiom-encode (step 6
  of the evaluator). Without that read the v6 lane cannot pass under (g').
  Use the fallback, or decide on a read-only fine-grained token for
  axiom-encode; that adds a stored secret, which this option set out to
  avoid.

### X2. A check with the same name does not satisfy the rule

- **Procedure.** Open **PR-2** in the target. Its branch adds
  `.github/workflows/imitation.yml`: `name: lab-guard`, `on: pull_request`,
  one job with `name: lab-guard` whose only step is `run: "true"`. Do not
  approve PR-2.
- **Pass.** The imitation job is green, the ruleset's `lab-guard` run is
  red, and the pull request stays blocked.
- **Record.** Both check runs with their `app` and `details_url`; the merge
  box text.
- **If it fails.** Any pull request can pass the guard by adding a workflow.
  (g') fails outright: use the fallback. This step cannot test a check that
  another GitHub App publishes; that case stays unverified.

### X3. A re-run after an approval updates the rule

- **Procedure.** On PR-1 (red, not approved): the second account approves.
  Do not push. (a) Re-run the failed job in the web page. If the pull
  request does not become mergeable: (b) `gh run rerun <run id> --failed
  --repo TheAxiomFoundation/guard-lab-target`. If still not: (c) close and
  reopen the pull request.
- **Pass.** After (a) the pull request is mergeable with no new push.
- **Record.** Which of (a), (b), (c) made it mergeable; the run attempts.
- **If it fails.** After an approval the guard needs a new event. If (c)
  works, reopening is the procedure. If only a new head works, someone other
  than the approver must update the branch, and CI runs once more for each
  approval.

### X16. Who can re-run a ruleset run

- **Procedure.** Make PR-3 red, approve it, then: (a) the second account
  (write access, not an administrator) re-runs the failed job; (b) on
  another red and approved pull request, dispatch `lab-rerun` with the run
  id.
- **Pass.** Both re-runs start and update the rule.
- **Record.** The result of (a) and of (b), with any error text.
- **If it fails.** (a) fails: only administrators can refresh the guard
  after an approval. (b) fails: a helper workflow that re-runs the guard on
  a review event is not possible; a person or an agent with write access
  re-runs it for each candidate.

### X13. A re-run of an old run cannot make a new head green

- **Procedure.** On an approved pull request with a green run on head H1,
  dismiss the approval, push a new commit (head H2; its run is red). Approve
  again. Re-run the **H1** run, not the H2 run.
- **Pass.** The H1 re-run is green on H1 and the pull request stays blocked
  until the H2 run itself passes.
- **Record.** The check runs of H1 and H2 and the merge box state.
- **If it fails.** The platform lets an old run satisfy a new head. The real
  runner still refuses a verdict for another head than its event's head
  (`event_head_is_stale`, design case 49) [code]. Record that this defence
  is then the only one.

### X9. An empty bypass list binds an organization owner

- **Procedure.** On a red pull request, try to merge as the organization
  owner, in the web page and with `gh pr merge --admin`.
- **Pass.** Both are refused.
- **Record.** The refusal text.
- **If it fails.** Owners bypass the rule by default. Record it under
  residual risk R1 for Max's decision.

### X8. Missing runs block the merge

- **Procedure.** (a) On a green, approved pull request, dispatch
  `lab-token-push` with its branch: the new head comes from the job token.
  (b) On another pull request, push a commit whose message contains
  `[skip ci]`. (c) Disable Actions in the target (Settings → Actions →
  General), then push to a pull request; enable Actions again afterwards.
- **Pass.** In each case no `lab-guard` run starts on the new head and the
  pull request is blocked.
- **Record.** For each case: whether a run started, and the merge box text.
- **If it fails.** A head with no guard run can merge. For (a) and (b) any
  author can cause that, so (g') fails: use the fallback. For (c) only a
  repository administrator can; record it under R1.

### X7 and X14. The branch-rules endpoint

- **Procedure.** With the ruleset `Active`, as the second account (not an
  owner):
  `gh api repos/TheAxiomFoundation/guard-lab-target/rules/branches/main > rules-branches-main.json`.
- **Pass.** The response holds one rule of type `workflows` with
  `ruleset_source_type: "Organization"`, the ruleset id, and in its
  parameters the `repository_id`, the `path` and the `sha` of commit A.
- **Record.** The whole file. An agent later turns it into a test fixture
  for the writer's lock 3 (design §8, row 5c); replace nothing in it by
  hand.
- **If it fails.** If the response has no `sha` or no `repository_id`, the
  signing runtime cannot check the pin from this endpoint. The organization
  endpoint needs `admin:org`. Lock 3 then has no data source, and activation
  step 12 waits for a new design of that lock.

### X5. The merge queue runs the workflow

- **Procedure.** In the target, create a repository ruleset for `main`:
  "Require merge queue" (merge method squash, build concurrency 1, minimum
  and maximum group size 1) and "Require status checks to pass" with
  `lab-slow`. Approve a pull request, wait until both checks are green, and
  add it to the queue.
- **Pass.** A `lab-guard` run starts with `github.event_name=merge_group`.
  The payload has `merge_group.head_sha`, `base_sha`, `head_ref` and
  `base_ref`, and `repository.full_name`. `head_ref` is
  `refs/heads/gh-readonly-queue/main/pr-<N>-<40 hex>` and `base_ref` is
  `refs/heads/main`. The pull request merges after `lab-slow` ends.
- **Record.** The output of "Print the event payload fields" and of "Print
  the run identity" for the merge-group run; the marker (it must be `A`);
  whether the head commit of the group equals the pull request's head.
- **If it fails.** No run, or a missing field: the real runner fails the
  job for a payload without these fields [code: `_enforce_merge_group`].
  Without a merge-group evaluation a dismissed approval never blocks
  (key-protection R3). (g') is then not acceptable for `Active` without a
  reviewed change and a new run of this step.

### X15. A dismissal and the queue

- **Procedure.** (a) Approve a pull request, let its run go green, dismiss
  the approval, then add the pull request to the queue. (b) Approve another,
  add it to the queue, and dismiss the approval before the `lab-guard`
  merge-group run starts or while it runs. (c) Approve another, add it to
  the queue, wait until the `lab-guard` merge-group run is green, and
  dismiss the approval while `lab-slow` still runs.
- **Pass.** (a) and (b): the merge-group run fails and the entry leaves the
  queue. (c): record only. The expected result is that the pull request
  **merges**, because nothing evaluates again after the guard's group run
  (key-protection R3 (c)).
- **Record.** For each case: the merge-group run's conclusion, whether the
  entry left the queue, whether the pull request merged.
- **If it fails.** (a) or (b) fail: the merge-time evaluation does not
  bound the stale green; (g') is not acceptable for `Active`. (c) does not
  merge: GitHub removes the entry by itself, and R3 (c) is smaller than
  stated.

### X10. Renaming the target

- **Procedure.** Rename `guard-lab-target` to `guard-lab-target-2`. Open a
  pull request.
- **Pass.** The rule still applies.
- **Record.** Whether the run starts and blocks.
- **If it fails.** A rename removes the gate. Set `"protected": true` in
  `repository_name` [docs D3] and run the step again.

### X17. A disabled workflow in the source repository

This step is an addition of this document. GitHub's troubleshooting page
lists "disable the individual workflow in the source repository" as a way to
keep a ruleset workflow from running on the source repository's own pull
requests [docs D2], and a run cannot be re-run while its workflow is
disabled [docs D9].

- **Procedure.** Disable `lab-guard` in the source
  (`gh workflow disable lab-guard.yml --repo TheAxiomFoundation/guard-lab-source`).
  Push to a pull request in the target. Then try to re-run that run.
- **Pass.** Record only. Either result is usable.
- **Record.** Whether the target run starts, and whether its re-run is
  refused.
- **If the target run does not start.** Disabling the workflow in
  axiom-encode would block every `rulespec-us` pull request. Record it as a
  rule: never disable `content-bound-merge-guard.yml` in axiom-encode while
  the ruleset is `Active`; a write collaborator of axiom-encode with the
  right to disable workflows could then block landings (not forge them).

## 3. Clean up

1. Delete the organization ruleset `guard-lab`:
   `gh api -X DELETE orgs/TheAxiomFoundation/rulesets/<id>`.
2. Delete both repositories (Settings → Danger zone → Delete this
   repository).
3. Confirm:
   `gh api orgs/TheAxiomFoundation/rulesets --jq '[.[].name]'` does not list
   `guard-lab`, and both repository URLs return 404.
4. Keep `rules-branches-main.json` and the recorded outputs. Give them to
   the agent that prepares the results pull request.

## 4. Record

An agent copies this table, filled in, into §11 of
`docs/content-bound-merge-guard.md` in a reviewed pull request. The owner's
recorded output is the evidence; the agent adds none.

| Step | Question | Result | Evidence (run URL or output) |
|---|---|---|---|
| X1 | The SHA pin runs exactly that file | | |
| X2 | A same-named check does not satisfy the rule | | |
| X3 | A re-run after an approval updates the rule | | |
| X4 | The job token reads public axiom-encode; checkout works | | |
| X5 | The merge queue runs the workflow; payload fields | | |
| X6 | `Evaluate` runs the workflow and does not block | | |
| X7 | A non-owner can read the active rule | | |
| X8 | Missing runs block the merge (token push, `[skip ci]`, Actions off) | | |
| X9 | An empty bypass list binds an owner | | |
| X10 | Renaming the target keeps the rule | | |
| X11 | `github.workflow_sha` and `github.workflow_ref` name the source commit | | |
| X12 | Check name; result on the pull request's head SHA | | |
| X13 | A re-run of an old run cannot make a new head green | | |
| X14 | The rules response carries `sha`, `repository_id`, `ruleset_source_type` | | |
| X15 | A dismissal before or in the queue removes the entry; after the group run it does not | | |
| X16 | A write collaborator and a job token can re-run a ruleset run | | |
| X17 | A workflow disabled in the source repository | | |
| — | The web form offers a SHA pin; the API accepts `sha` with no `ref` | | |

## 5. What this experiment cannot show

- The behavior of the real workflow against a real RuleSpec pull request.
  The shadow period of the activation checklist does that.
- A check with the guard's name that another GitHub App publishes (X2 tests
  a workflow only).
- The behavior on rate limiting, and the real duration of
  `validate / validate` against the runner's 60-minute wait.
- What happens to the rule if the organization's plan changes.
- Anything about the fallback: the cloud condition, the App binding and an
  App check on a merge-group commit need real settings.
