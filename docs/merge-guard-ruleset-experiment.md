# Merge guard: the ruleset experiment

Status: **designed, not run.** The recommendation in
`docs/merge-guard-key-protection.md` holds only if this experiment passes.
Until an organization owner runs it and records the results, nothing is
chosen for activation.

- **When.** At any time from now. **The experiment does not depend on any
  pull request of the merge-guard stack being merged**: it uses only the
  two throwaway repositories, its own lab workflow files (§1) and the one
  lab ruleset. An organization owner can run it before the merges; a
  failed blocking step is then known before review effort is spent. It
  must have passed before checklist step 5 of
  `docs/content-bound-merge-guard.md` §8.
- **Who.** An organization owner of TheAxiomFoundation. Organization
  rulesets need owner rights. An agent must not run it, and no agent session
  may receive a token with the `admin:org` scope for it.
- **A second account.** Several steps need an approving review. GitHub does
  not let the author of a pull request approve it. Use a second account with
  write access to the target repository. For X16 it must not be an
  administrator of the target.
- **Time.** About half a day. There are 24 steps, and each queue case
  holds a check of three minutes.
- **What it creates.** The complete list:
  - two public throwaway repositories, `guard-lab-source` and
    `guard-lab-target`, with the workflow files of §1;
  - one organization ruleset, `guard-lab`, which targets
    `guard-lab-target` only;
  - in `guard-lab-target`: one repository ruleset for `main` with "Require
    merge queue" and one required status check (X5), the branches `alt`
    and `side` (X21, X24), and the branches and pull requests of the
    steps;
  - for one part of X21, the administrator role on `guard-lab-target` for
    the second account.

  It creates no stored secret, no GitHub App and no key. It registers no
  self-hosted runner. It changes no organization setting: the only
  organization-level object that it creates, edits or deletes is the
  ruleset `guard-lab`. It touches neither `axiom-encode` nor any
  `rulespec-*` repository: every step below acts on `guard-lab-source`,
  `guard-lab-target` and the ruleset `guard-lab` only. The clean-up of §3
  removes everything in this list.
- **What "passed" means.** Checklist step 3 of
  `docs/content-bound-merge-guard.md` §8 sorts the steps into four
  classes and defines "the experiment passed": blocking; adjusting, run
  again; adjusting, no second run; informational. A step of the class
  "adjusting, no second run" that fails does not stop the experiment: its
  result is recorded, and its named change is a procedure or a decision
  that a reviewed change of the checklist takes up. Each step below names
  its class.
- **Evidence labels** are those of `docs/merge-guard-key-protection.md`;
  D1-D26 are its documentation list. Every "expected" below is unverified
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
              {number, head_sha: .head.sha, base_ref: .base.ref,
               base_repository: .base.repo.full_name} end),
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

      - name: For a merge group, wait until lab-slow has completed on the group commit
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          set -euo pipefail
          if [ "$GITHUB_EVENT_NAME" != "merge_group" ]; then
            echo "not a merge group: no wait"
            exit 0
          fi
          group="$(jq -r '.merge_group.head_sha' "$GITHUB_EVENT_PATH")"
          listing="$RUNNER_TEMP/checks.json"
          attempt=0
          while [ "$attempt" -lt 30 ]; do
            attempt=$((attempt + 1))
            code="$(curl -s -o "$listing" -w '%{http_code}' \
              -H "Authorization: Bearer $GH_TOKEN" \
              -H "Accept: application/vnd.github+json" \
              -H "X-GitHub-Api-Version: 2022-11-28" \
              "https://api.github.com/repos/$GITHUB_REPOSITORY/commits/$group/check-runs?check_name=lab-slow&filter=latest&per_page=100")"
            echo "poll=$attempt status=$code at $(date -u +%H:%M:%S)"
            [ "$code" = "200" ] || exit 1
            jq -c '{total_count, runs: [.check_runs[] | {name, app: .app.id,
              head_sha, status, conclusion, details_url}]}' "$listing"
            pending="$(jq '[.check_runs[] | select(.status != "completed")] | length' "$listing")"
            done_runs="$(jq '[.check_runs[] | select(.status == "completed")] | length' "$listing")"
            if [ "$done_runs" -ge 1 ] && [ "$pending" = "0" ]; then
              run_id="$(jq -r '.check_runs[0].details_url' "$listing" | sed -n \
                's#^https://github.com/[^/]*/[^/]*/actions/runs/\([0-9][0-9]*\)/job/[0-9][0-9]*$#\1#p')"
              echo "workflow run of the check=$run_id"
              curl -s \
                -H "Authorization: Bearer $GH_TOKEN" \
                -H "Accept: application/vnd.github+json" \
                -H "X-GitHub-Api-Version: 2022-11-28" \
                "https://api.github.com/repos/$GITHUB_REPOSITORY/actions/runs/$run_id" \
                | jq -c '{id, event, path, head_sha, status, conclusion,
                    repository: .repository.full_name}'
              echo "lab-slow completed on the group commit at $(date -u +%H:%M:%S)"
              exit 0
            fi
            sleep 15
          done
          echo "lab-slow did not complete on the group commit"
          exit 1

      - name: Pass only with an effective approval
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          set -euo pipefail
          echo "evaluation starts at $(date -u +%H:%M:%S)"
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

The token reaches three steps through `env` only. No step prints it. Do not
add `set -x`.

For a merge group the lab workflow does what the real runner does: it waits
until the other required check (`lab-slow`) has completed on the group
commit, and only then reads the reviews [code: `_wait_for_group_ci` in
`merge_guard_runner.py`]. The real runner also requires that check to be the
group's own `merge_group` run and to have succeeded; the lab prints those
fields for the record and does not judge them.

### 1.2 The target repository

1. Create the public repository `TheAxiomFoundation/guard-lab-target` with a
   README. Leave Actions on. Give the second account write access.
2. Add these four files to `main`.

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

`.github/workflows/lab-imitate-api.yml` (creates a check run and a commit
status with the guard's name through the API, with the job token, for X2):

```yaml
name: lab-imitate-api
on:
  workflow_dispatch:
    inputs:
      sha:
        description: The pull request head commit to mark
        required: true
        type: string
      kind:
        description: check-run or status
        required: true
        type: choice
        options: [check-run, status]
permissions:
  checks: write
  statuses: write
jobs:
  imitate:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - env:
          GH_TOKEN: ${{ github.token }}
          SHA: ${{ inputs.sha }}
          KIND: ${{ inputs.kind }}
        run: |
          set -euo pipefail
          case "$SHA" in *[!0-9a-f]*|'') exit 1 ;; esac
          [ "${#SHA}" = 40 ] || exit 1
          if [ "$KIND" = "check-run" ]; then
            gh api -X POST "repos/$GITHUB_REPOSITORY/check-runs" \
              -f name=lab-guard -f head_sha="$SHA" \
              -f status=completed -f conclusion=success
          else
            gh api -X POST "repos/$GITHUB_REPOSITORY/statuses/$SHA" \
              -f state=success -f context=lab-guard
          fi
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
| Target branches | the one ref `refs/heads/main`, as an include pattern; not the form's "default branch" target |
| Rule | "Require workflows to pass before merging": repository `guard-lab-source`, file `.github/workflows/lab-guard.yml`, pinned to the SHA of commit A |

```json
{
  "name": "guard-lab",
  "target": "branch",
  "enforcement": "evaluate",
  "bypass_actors": [],
  "conditions": {
    "repository_name": {"include": ["guard-lab-target"], "exclude": []},
    "ref_name": {"include": ["refs/heads/main"], "exclude": []}
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
Record the returned `id`. Record whether the web form offers a SHA pin
(class: informational), and whether the API accepts `sha` with no `ref`
(class: adjusting, run again). If it refuses, add `"ref": "refs/heads/main"` and
record that both were needed; three places of the real design then change
together (design checklist step 3). Record also how the web form names the
target that gives `refs/heads/main`, and which repository condition a
ruleset from the web form stores, `repository_name` or `repository_id`
(class: informational). Then read the ruleset back:
`gh api orgs/TheAxiomFoundation/rulesets/<id> --jq .conditions.ref_name`
must print `refs/heads/main` in `include`, not `~DEFAULT_BRANCH`. This
read-back is a set-up condition: do not start the steps until it holds.

## 2. Steps

Do them in this order, with one exception: part (c) of X2 comes after X15.
For each step, record the result in the table of §4: `pass`, `fail` or
`not run`, with the run URL or the command output. Every step acts on the
two lab repositories and the lab ruleset only.

### X6. `Evaluate` mode runs the workflow and does not block

Class: adjusting, no second run (the change is the form of the shadow period,
design checklist steps 6 and 7).

- **Procedure.** With the ruleset in `Evaluate`, open two pull requests in
  the target, **PR-0** and **PR-1** (any one-line change each). Do not
  approve them. Wait for the `lab-guard` run of each. Then merge PR-0. Keep
  PR-1 open for the next steps.
- **Pass.** A `lab-guard` run starts in `guard-lab-target` for each pull
  request and fails (no approval). PR-0 can be merged although its run
  failed.
- **Record.** The run URLs; the check name on the pull request. The log of
  the run is the evidence that the workflow ran. Rule Insights
  (Organization settings → Repository → Rule insights): GitHub documents
  that it logs nothing "until a pull request is merged or a merge is
  attempted" [docs D2]. So record that page twice: before the merge of PR-0
  (expected: no row for PR-0 or PR-1) and after it (expected: one row for
  PR-0, marked as a failure in evaluate mode). Record what the page shows
  in each case; neither expectation is a pass condition.
- **If it fails.** No run in `Evaluate`: no native shadow mode. The shadow
  period of the design then needs another form (a throwaway target, or a
  short `Active` window on a quiet day), decided in a reviewed change of
  the checklist. It does not end (g').

Now set the ruleset to `Active`. Record whether PR-1 becomes blocked at once
or only after the next run.

### X1. The SHA pin runs exactly that file

Class: blocking.

- **Procedure.** In the source, change `MARKER: "A"` to `MARKER: "B"` and
  commit to `main` (commit **B**). Do not edit the ruleset. Push a new commit
  to PR-1.
- **Pass.** The new run prints `marker=A`.
- **Record.** The printed marker; the SHA of A and of B.
- **If it fails.** The pin does not fix the file. A merge to axiom-encode
  `main` would then change the guard at once, and `main` needs no approval
  for most files. (g') fails: use the fallback.

### X11. What `github.workflow_sha` and `github.workflow_ref` are

Class: adjusting, run again.

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

### X18. The run names the TARGET repository

Class: adjusting, run again; no ruleset may name the real runner until it passes.

- **Procedure.** Read the "Print the run identity" and "Print the event
  payload fields" steps of the X6 run (a `pull_request` event) and, later,
  of the X5 run (a `merge_group` event).
- **Pass.** In both runs `github.repository` and `GITHUB_REPOSITORY` are
  `TheAxiomFoundation/guard-lab-target`, and the payload's `repository` is
  `TheAxiomFoundation/guard-lab-target`. In the `pull_request` run the
  payload's `pull_request.base_repository` is the target too. None of them
  is `guard-lab-source`.
- **Record.** The five printed values of each run (four for the
  `merge_group` run, whose payload has no pull request; record whether its
  `repository` key is present at all [docs D14 does not mark it required]).
- **If it fails.** This is the one unverified point that fails **open**. The
  real runner ends with success, without an evaluation, when
  `GITHUB_REPOSITORY` and the payload both name `axiom-encode` [code:
  `_require_source_event`]. If a ruleset run reports the source repository
  in all of them, every pull request of the target would pass unjudged:
  (g') must not go `Active`, and the runner needs a reviewed change (for
  example no inert path at all) and a new run of this step. If only some of
  the values name the source, the real runner fails every run (closed);
  record which, and prepare the change that reads the right one. If the
  `merge_group` payload has no `repository` key, the real runner fails every
  merge-group run (closed) [code: `enforce`].

### X12. The check name and the commit it attaches to

Class: adjusting, run again.

- **Procedure.** `gh api repos/TheAxiomFoundation/guard-lab-target/commits/<PR-1 head>/check-runs --jq '.check_runs[] | {name, head_sha, app: .app.slug, status, conclusion}'`.
  Also read the name in the pull request's merge box and in Rule Insights.
- **Pass.** One check run named `lab-guard` (the job name) on the pull
  request's head SHA.
- **Record.** The command output and the name the merge box shows.
- **If it fails.** If the result attaches to another commit (for example the
  test merge commit), the design's head binding and case 49 need a new
  reading before `Active`.

### X4. The target's job token reads the public source of truth

Class: adjusting, no second run (the change is a decision, not code).

- **Procedure.** Read the "Read public axiom-encode with the job token" step
  of any run, and the checkout step of X11.
- **Pass.** `status=200` for the Actions run and for the commit. The
  checkout of the public source repository succeeds with the job token.
- **Record.** The two status lines.
- **If it fails.** A v6 unit needs the signing run of axiom-encode (step 6
  of the evaluator). Without that read the v6 lane cannot pass under (g').
  Use the fallback, or decide on a read-only fine-grained token for
  axiom-encode; that adds a stored secret, which this option set out to
  avoid. No such token is designed. It would have to meet the
  stored-credential precondition of checklist step 11: not a credential
  of an enrolled identity, an administrator or an organization owner.

### X2. A check with the same name does not satisfy the rule

Class: blocking, each part.

The rule must be satisfied only by the run that the ruleset itself starts.
Each part offers the platform another green result with the same name on a
pull request that is **not** approved, so the ruleset's own `lab-guard` run
is red.

- **(a) Another workflow file.** Open **PR-2** in the target. Its branch
  adds `.github/workflows/imitation.yml`: `name: lab-guard`,
  `on: [pull_request, merge_group]`, one job with `name: lab-guard` whose
  only step is `run: "true"`.
- **(b) A file at the ruleset workflow's own path.** Open **PR-2b**. Its
  branch adds `.github/workflows/lab-guard.yml` to the target, the same
  path and name as the file in the source, with `name: lab-guard`,
  `on: pull_request` and one job `lab-guard` whose only step is
  `run: "true"`.
- **(c) An imitation on `merge_group`.** Run this part after X5 and X15.
  Take a pull request whose branch holds the imitation of (a). Approve it,
  let its ruleset run go green, add it to the queue, and dismiss the
  approval while `lab-slow` of the group still runs, as in X15 (b). The
  group commit holds the pull request's workflows, so the imitation's
  `merge_group` job is green while the ruleset's group run fails.
- **(d) A check run created through the API.** On a new unapproved pull
  request **PR-2d**, dispatch `lab-imitate-api` with its head SHA and
  `kind: check-run`. It creates a completed, successful check run named
  `lab-guard` from the GitHub Actions app.
- **(e) A commit status with the same context.** On a new unapproved pull
  request **PR-2e**, dispatch `lab-imitate-api` with its head SHA and
  `kind: status`.
- **Pass.** In (a), (b), (d) and (e) the imitation is green, the ruleset's
  `lab-guard` run is red, and the pull request stays blocked. In (c) the
  entry leaves the queue and the pull request does not merge, **and the
  record shows all three of these on the same group commit SHA**: the
  imitation's `merge_group` job completed with `success` (its
  `details_url` names a run of `imitation.yml`); the ordinary check
  `lab-slow` completed with `success`; and the pinned `lab-guard` run of
  the ruleset (marker `A`) failed for the dismissed approval. An entry
  that left the queue while the imitation never ran, or because `lab-slow`
  or anything else failed, is **not a pass and not a fail: the part did
  not test the question; run it again**.
- **Record.** For each part: every check run and status on the head (or the
  group commit) with its `name`, `app` and `details_url`
  (`gh api repos/TheAxiomFoundation/guard-lab-target/commits/<sha>/check-runs --jq '.check_runs[] | {name, app: .app.slug, conclusion, details_url}'`
  and `.../commits/<sha>/statuses`); the merge box text; for (b) which file
  each of the two runs used (the marker line exists only in the source's
  file).
- **If it fails.** Any part: a pull request can pass the guard by adding a
  workflow or by one API call of a workflow with `checks: write` or
  `statuses: write`. (g') fails outright: use the fallback.
- **Not covered: a check that another GitHub App publishes.** The lab
  creates no App, so no part of this step tests it. That case stays
  [inferred, untested]; `docs/merge-guard-key-protection.md` §4.4, R5, says
  what would test it.

### X3. A re-run after an approval updates the rule

Class: adjusting, no second run (the change is the procedure after an approval).

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

Class: informational.

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

Class: informational.

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

Class: informational.

- **Procedure.** On a red pull request, try to merge as the organization
  owner, in the web page and with `gh pr merge --admin`.
- **Pass.** Both are refused.
- **Record.** The refusal text.
- **If it fails.** Owners bypass the rule by default. Record it under
  residual risk R1 for Max's decision.

### X8. Missing runs block the merge

Class: blocking for (a) and (b); informational for (c).

- **Procedure.** (a) On a green, approved pull request, dispatch
  `lab-token-push` with its branch: the new head comes from the job token.
  (b) On another pull request, push a commit whose message contains
  `[skip ci]`. (c) Disable Actions in the target (Settings → Actions →
  General), then push to a pull request; enable Actions again afterwards.
- **Pass.** In each case no `lab-guard` run starts on the new head and the
  pull request is blocked.
- **Record.** For each case: whether a run started, and the merge box text.
  For (a) there are three outcomes, and each is recorded as its own
  result: no run at all (what the ruleset page documents [docs D2]); a run
  that waits for approval (what the events page documents for an ordinary
  workflow: such a pull request gets runs "that require approval"
  [docs D20]); or a run that starts. With a run that waits for approval,
  record whether the pull request is blocked while it waits, who can
  approve it, and what it does after the approval. The first two outcomes
  pass if the pull request stays blocked.
- **If it fails.** A head with no guard run can merge. For (a) and (b) any
  author can cause that, so (g') fails: use the fallback. For (c) only a
  repository administrator can. Record it under R12 (d) of
  `docs/merge-guard-key-protection.md` §4.4: an administrator of the real
  repository can then merge without the guard by disabling Actions and
  enabling them again. The consequence is a named residual risk, a row of
  the design's health check (Actions are enabled) and an item of "What Max
  should know".

### X7 and X14. The branch-rules endpoint

Class: X7 informational; X14 adjusting, run again.

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
  step 13 waits for a new design of that lock (draft #2289 holds the lock
  with a synthetic fixture of this response).

### X5. The merge queue runs the workflow

Class: blocking for "a `merge_group` run starts" and for the marker `A`; adjusting, run again, for the payload fields.

- **Procedure.** In the target, create a repository ruleset for `main`:
  "Require merge queue" (merge method squash, build concurrency 1, minimum
  and maximum group size 1) and "Require status checks to pass" with
  `lab-slow`. Read the stored rule and record it: its parameters must be
  the four that production requires (design §8, step 8),
  `grouping_strategy: "ALLGREEN"`, `max_entries_to_merge: 1`,
  `min_entries_to_merge: 1` and `max_entries_to_build: 1`
  (`gh api repos/TheAxiomFoundation/guard-lab-target/rules/branches/main --jq '.[] | select(.type == "merge_queue") | .parameters'`). Approve a pull request, wait until both checks are green, and
  add it to the queue.
- **Pass.** A `lab-guard` run starts with `github.event_name=merge_group`.
  It prints `marker=A`: the source has commit B since X1 and the ruleset
  still pins commit A, so the SHA pin must hold for the `merge_group` run
  too. The payload has `merge_group.head_sha`, `base_sha`, `head_ref` and
  `base_ref`, and `repository.full_name`. `head_ref` is
  `refs/heads/gh-readonly-queue/main/pr-<N>-<40 hex>` and `base_ref` is
  `refs/heads/main`. The pull request merges after `lab-slow` ends.
- **Record.** The output of "Print the event payload fields" and of "Print
  the run identity" for the merge-group run; the printed marker;
  whether the head commit of the group equals the pull request's head.
- **If it fails.** No run, or a missing field: the real runner fails the
  job for a payload without these fields [code: `_enforce_merge_group`].
  Without a merge-group evaluation a dismissed approval never blocks
  (key-protection R3). (g') is then not acceptable for `Active` without a
  reviewed change and a new run of this step. A marker other than `A`: the
  pin does not fix the file that a merge group runs, so a merge to
  axiom-encode `main` would change the guard at merge time at once. (g')
  fails: use the fallback.

### X15. A dismissal and the queue

Class: blocking for (a), (b) and for "merges only after the guard job has
ended" in (c); adjusting, run again, for the identity of the group's check
in (c); informational for (d).

The guard's merge-group run waits for the group's other required check and
evaluates after it (design §6). This step must confirm that, and measure
the window that remains.

- **Procedure.** (a) Approve a pull request, let its run go green, dismiss
  the approval, then add the pull request to the queue. (b) Approve another,
  add it to the queue, and dismiss the approval while `lab-slow` of the
  group still runs; the `lab-guard` merge-group run is then in its wait
  step. (c) Approve another, add it to the queue, and dismiss nothing. (d)
  Record only: approve another, add it to the queue, and try to dismiss the
  approval in the seconds after the `lab-guard` merge-group run has ended
  green.
- **Pass.** (a) and (b): the `lab-guard` merge-group run fails, in its last
  step, after `lab-slow` has completed, and the entry leaves the queue. (c):
  the wait step of the merge-group run lists a check named `lab-slow` on the
  group commit, from app id 15368 (GitHub Actions), whose workflow run has
  `event` `merge_group` and `head_sha` equal to the group commit; the line
  "evaluation starts at" is later than the line "lab-slow completed on the
  group commit at"; the `lab-guard` job ends after the `lab-slow` job; and
  the pull request merges only after the `lab-guard` job has ended.
- **Record.** For each case: the merge-group run's conclusion, whether the
  entry left the queue, whether the pull request merged. For (c) also: the
  JSON lines of the wait step; the `completed_at` of both jobs
  (`gh run view <run id> --repo TheAxiomFoundation/guard-lab-target --json jobs --jq '.jobs[] | {name, completedAt}'`)
  and the pull request's `merged_at`
  (`gh api repos/TheAxiomFoundation/guard-lab-target/pulls/<n> --jq .merged_at`).
  The difference between the end of the `lab-guard` job and `merged_at` is
  the queue's part of the window that remains. For (d): whether the pull
  request merged; the expected result is that it merges, because nothing
  evaluates again after the guard's group run (key-protection R3 (c)).
- **If it fails.** (a) or (b) fail: the merge-time evaluation does not
  bound the stale green; (g') is not acceptable for `Active`. (c): if the
  check on the group commit has another name, app or run than the real
  runner expects, the real runner fails every merge group (closed) and needs
  a reviewed change [code: `_group_ci_succeeded`]; if the pull request
  merges before the `lab-guard` job has ended, the ruleset result is not
  part of the queue's requirements, and (g') is not acceptable for `Active`.
  (d) does not merge: GitHub removes the entry by itself, and R3 (c) is
  smaller than stated.

### X10. Renaming the target

Class: adjusting, run again.

- **Targeting form.** Run this step with the form that production uses:
  the `repository_name` condition by name, as in the body of §1.3 and in
  the production body (`docs/merge-guard-key-protection.md`, "Create the
  ruleset", step 3). Read the stored ruleset first
  (`gh api orgs/TheAxiomFoundation/rulesets/<id> --jq .conditions`) and
  record it. A pass with a `repository_id` condition says nothing about
  the name-based configuration and does not count.
- **Procedure.** Rename `guard-lab-target` to `guard-lab-target-2`. Open a
  pull request.
- **Pass.** The rule still applies: the run starts and the pull request is
  blocked.
- **Record.** The stored `conditions`; whether the rename was accepted;
  whether the run starts and blocks.
- **If it fails.** A rename removes the gate. Set `"protected": true` in
  `repository_name` [docs D3] and run the step again. With
  `"protected": true`, **a refused rename is valid evidence**, together
  with proof that the guard still applies: after the refusal, a new pull
  request in `guard-lab-target` starts the `lab-guard` run and is blocked.
  A refused rename without that second observation is not a pass.
- Rename the repository back to `guard-lab-target` at the end of this
  step; the later steps use that name.
- Not covered: a transfer of the repository to another owner. No step
  tries it (§5).

### X17. A disabled workflow in the source repository

Class: informational.

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
- Enable the workflow again at the end of this step
  (`gh workflow enable lab-guard.yml --repo TheAxiomFoundation/guard-lab-source`).

### X19. A pin move with an open green pull request

Class: informational; it can change the pin procedure.

- **Procedure.** The source has commit B (from X1). Approve a pull request
  **PR-G**; its run from pin A is green. As the owner, move the ruleset's
  pin to commit B by updating the stored ruleset, exactly as the pin
  procedure does (`docs/merge-guard-key-protection.md`, "Per guard
  release", step 6): read the ruleset, change only the `sha`, `PUT` the
  result, compare. Do not send the creation body of §1.3 again: it sets
  `"enforcement": "evaluate"` and would switch the lab's `Active` ruleset
  off. Read the ruleset back: the enforcement must still be `active`, and
  the conditions and the bypass list must be unchanged. Do not push to
  PR-G. Then: (i) read the merge box of PR-G; (ii) add PR-G to the queue
  and read the marker that its `merge_group` run prints; (iii) push to
  another pull request and read its marker.
- **Pass.** Record only. Expected [inferred]: (i) PR-G stays mergeable with
  the old result; (ii) the group run prints `marker=B`; (iii) `marker=B`.
- **Record.** The three observations; the output of the comparison of the
  pin procedure (`diff` prints nothing; the last line shows `active` and
  commit B); whether the `PUT` accepted the body built from the `GET`
  response.
- **What the result changes.** If (ii) prints `marker=A`, a pin move does
  not reach a pull request that is already green, in the queue either. A
  pin move that removes an enrolled identity or closes a defect then needs
  one more act in the pin procedure of
  `docs/merge-guard-key-protection.md` §5: a new event on every open pull
  request. If (i) shows the pull request blocked until a new run, every
  pin move costs one CI run for each open pull request.

### X20. `Evaluate` together with a merge queue

Class: adjusting, no second run (the change is the form of the queue
trial, design checklist step 8).

- **Procedure.** After X5, with the queue of the target on, set the
  ruleset back to `Evaluate`. Open a pull request, do **not** approve it,
  wait until `lab-slow` is green, and add it to the queue.
- **Pass.** A `lab-guard` run starts with `github.event_name=merge_group`
  while the ruleset is in `Evaluate`. That is the one pass condition. The
  other observations are recorded only; expected [inferred]: the pull
  request can be queued although its `lab-guard` run is red; the queue
  merges when `lab-slow` of the group is green, without waiting for
  `lab-guard`; the `lab-guard` group run then fails in its last step.
- **Record.** Whether a `merge_group` run of `lab-guard` started; its
  conclusion; `completedAt` of both jobs and the pull request's
  `merged_at`; the Rule Insights row after the merge.
- **If no `merge_group` run starts in `Evaluate`.** The queue trial of
  checklist step 8 cannot observe the guard on a merge group before
  `Active`. The trial then moves to the first `Active` landing (checklist
  step 12, with its stop rule), in a reviewed change of the checklist.
  That change must also replace the trial clause of the precondition of
  checklist step 11, which nothing can meet in this case, and move the
  stop rule for the inert path on a merge group from step 8 to step 12,
  before any second landing.
- Set the ruleset to `Active` again before the next step.

### X21. A change of the default branch

Class: adjusting, run again.

- **Procedure.** In the target, create the branch `alt` from `main`. As a
  repository administrator, make `alt` the default branch. Open an
  unapproved pull request into `main` and one into `alt`. Then make `main`
  the default branch again. Then the rename: give the second account the
  administrator role on `guard-lab-target` (X16 is done by now; an
  organization owner can rename without restriction [docs D22], so the
  owner's own attempt shows nothing). As that account, try to rename `main`
  to `main-renamed`. If the rename succeeds, rename the branch back at
  once. Do not change the organization setting "Allow repository
  administrators to rename branches protected by organization rules";
  read it and record its value.
- **Pass.** The pull request into `main` gets a `lab-guard` run and is
  blocked. The pull request into `alt` has no rule: that is expected with
  the explicit target `refs/heads/main`. The rename of `main` by the
  repository administrator is refused.
- **Record.** For each pull request: whether a run started and whether it
  is blocked; `gh api repos/TheAxiomFoundation/guard-lab-target/rules/branches/main --jq '[.[].type]'`
  and the same for `alt`, while `alt` is the default branch. For the
  rename: the value of the organization setting, and the refusal text or
  the fact that the rename went through.
- **If it fails.** If `main` has no rule while another branch is the
  default, the explicit target does not hold `main`: the ruleset needs
  another target form, and a reviewed change of
  `docs/merge-guard-key-protection.md` §5 names it. In either case a
  repository administrator can move the default branch; the design's
  health check reads it. If the rename goes through, a repository
  administrator can take `main` out of the rule: the change is a ruleset
  target that the rename cannot leave, or the organization setting off as
  a precondition of `Active`, in a reviewed change of the design.

### X22. Two conclusions on one head

Class: informational.

- **Procedure.** Approve a pull request and let its run go green on head
  H. Dismiss the approval. Close and reopen the pull request: a new run on
  the same head H fails.
- **Pass.** Record only.
- **Record.** The check runs on H with their conclusions and times; whether
  the pull request is blocked (the newest result counts) or mergeable (an
  older green result counts).
- **What the result changes.** If an older green result counts, a
  re-evaluation that fails cannot withdraw a pass on the same head. The
  merge-group evaluation is then the only defence against a stale pass
  (key-protection R3), as the design already assumes.

### X23. A self-hosted runner with the label `ubuntu-latest`: the documentation only

Class: informational. The lab registers no self-hosted runner and changes no organization setting.

- **Why there is no live test.** GitHub advises that self-hosted runners
  "should almost never be used for public repositories" [docs D23]. The
  lab repositories are public, and a runner there would run pull request
  jobs on the owner's machine. A setting that could stop such a runner is
  an organization setting, which acts on every repository of the
  organization [docs D19]; this experiment must not change one.
- **Procedure.** Read only. In the log of one `lab-guard` job and of one
  `lab-slow` job (an ordinary workflow of the target), read "Runner name"
  and "Runner group name". Note the three documented statements with the
  date of the read: the values of the organization's "Runners" setting
  ("All repositories", "Selected repositories", "Disabled") and that
  runners that exist then get no new jobs [docs D19]; and that a job runs
  on any runner that matches all of its `runs-on` values [docs D21].
- **Pass.** Record only. Expected: both jobs ran on a GitHub-hosted runner.
- **Record.** The runner name and group of both jobs, and the words
  "documentation only; not tested".
- **What stays unverified.** Whether a repository-level self-hosted runner
  with the custom label `ubuntu-latest` can take the guard job
  [inferred, untested]. The activation checklist therefore makes the
  organization's "Runners" setting a precondition of `Active` in every
  case, and an organization owner verifies it there (design checklist step
  11; `docs/merge-guard-key-protection.md` §5 step 9).
- **If a job did not run on a GitHub-hosted runner.** Record and stop; do
  not change any organization setting from the lab. An organization-level
  runner took a lab job; an owner reads the runner groups before anything
  else happens.

### X24. A pull request whose base is changed to `main`

Class: blocking.

- **Procedure.** In the target, create the branch `side` from `main`. Open
  an unapproved pull request into `side`: the ruleset does not target it,
  so no `lab-guard` run starts. Edit the pull request and change its base
  to `main`. Do not push.
- **Pass.** No run starts (`edited` is not an activity type that starts a
  ruleset workflow [docs D1, D2]) and the pull request is blocked until a
  push, an update or a reopen starts a run.
- **Record.** Whether a run started; the merge box text.
- **If it fails.** If the pull request can merge into `main` with no
  `lab-guard` run, any author passes the guard by opening the pull request
  against another branch first. (g') fails: use the fallback.

## 3. Clean up

0. Make `main` the default branch of the target again, and give `main` its
   name back if the rename of X21 went through. No self-hosted runner
   exists to remove: X23 registers none.
1. Read the name of the ruleset before you delete it:
   `gh api orgs/TheAxiomFoundation/rulesets/<id> --jq .name` must print
   `guard-lab`. Only then delete the organization ruleset `guard-lab`:
   `gh api -X DELETE orgs/TheAxiomFoundation/rulesets/<id>`.
2. Delete both repositories (Settings → Danger zone → Delete this
   repository). That removes the repository ruleset of X5, the branches
   `alt` and `side`, every pull request and the second account's roles.
3. Confirm:
   `gh api orgs/TheAxiomFoundation/rulesets --jq '[.[].name]'` does not list
   `guard-lab`, and both repository URLs return 404. No organization
   setting was changed, so none is set back.
4. Keep `rules-branches-main.json` and the recorded outputs. Give them to
   the agent that prepares the results pull request.

## 4. Record

An agent copies this table, filled in, into §11 of
`docs/content-bound-merge-guard.md` in a reviewed pull request. The owner's
recorded output is the evidence; the agent adds none.

| Step | Question | Result | Evidence (run URL or output) |
|---|---|---|---|
| X1 | The SHA pin runs exactly that file | | |
| X2 (a) | An imitation workflow in another file does not satisfy the rule | | |
| X2 (b) | A file at the ruleset workflow's own path in the target does not satisfy it | | |
| X2 (c) | An imitation on `merge_group` does not keep the entry in the queue | | |
| X2 (d) | A check run created through the API does not satisfy it | | |
| X2 (e) | A commit status with the same context does not satisfy it | | |
| X3 | A re-run after an approval updates the rule | | |
| X4 | The job token reads public axiom-encode; checkout works | | |
| X5 | The merge queue runs the workflow, from the pinned commit (marker `A`); payload fields | | |
| X6 | `Evaluate` runs the workflow and does not block | | |
| X7 | A non-owner can read the active rule | | |
| X8 | Missing runs block the merge (token push, with its three outcomes; `[skip ci]`; Actions off) | | |
| X9 | An empty bypass list binds an owner | | |
| X10 | Renaming the target keeps the rule | | |
| X11 | `github.workflow_sha` and `github.workflow_ref` name the source commit | | |
| X12 | Check name; result on the pull request's head SHA | | |
| X13 | A re-run of an old run cannot make a new head green | | |
| X14 | The rules response carries `sha`, `repository_id`, `ruleset_source_type` | | |
| X15 | A dismissal before or in the queue removes the entry; the guard's group run ends after the group's other check; seconds from its end to the merge | | |
| X16 | A write collaborator and a job token can re-run a ruleset run | | |
| X17 | A workflow disabled in the source repository | | |
| X18 | `GITHUB_REPOSITORY`, `github.repository` and the payload name the target repository | | |
| X19 | A pin move with an open green pull request: old result, and the file of the next `merge_group` run | | |
| X20 | `Evaluate` with a merge queue: a `merge_group` run starts (the pass condition); the queue does not wait for it (record) | | |
| X21 | A changed default branch leaves the rule on `main`; a repository administrator cannot rename `main` | | |
| X22 | Two conclusions on one head: which one counts | | |
| X23 | Documentation only, no runner is registered: the runner name and group of two lab jobs | | |
| X24 | A pull request whose base is changed to `main` gets no run and is blocked | | |
| — | Set-up (§1.3): the web form offers a SHA pin; the API accepts `sha` with no `ref`; how the web form names the target; which repository condition it stores; the target reads back as `refs/heads/main` | | |

## 5. What this experiment cannot show

- The behavior of the real workflow against a real RuleSpec pull request.
  The shadow period of the activation checklist does that.
- A check with the guard's name that another GitHub App publishes. X2
  tests a workflow, a check run made with the job token and a commit
  status; the App case stays [inferred, untested].
- Whether a repository-level self-hosted runner with the label
  `ubuntu-latest` can take the guard job. X23 records the documentation
  only.
- What a bypass actor on the repository ruleset of the merge queue can do.
  The lab's queue ruleset has an empty bypass list.
- What a transfer of the target repository out of the organization does
  to the rule. No step tries it.
- Anything about stored credentials. The lab stores no secret. That a
  write collaborator can read a repository secret through a workflow of a
  branch is documented [docs D23]; which secrets the real repositories
  hold, and whose credentials they are, is a read of the activation
  checklist (step 11), not of this experiment.
- Which GitHub Apps are installed on the real repository, and who can
  install one. That is an owner's read of checklist step 11.
- The behavior on rate limiting, and which limit a job token of
  `rulespec-us` has. The duration of `validate / validate` on a pull
  request is measured (172 to 227 minutes for 38 of 40 runs
  [live read 2026-11-17]); the runner waits 5 hours for it. Its duration on
  a merge group is not measured; the runner waits 5 hours for that too.
- Properties of the real repository's settings: that the merge queue of
  `rulespec-us` stays on, that its default branch stays `main`, and which
  runners it can use. The lab shows what each setting does; only the
  read-only health check of the design shows their state.
- Any step with the lab ruleset in a state that the step does not name.
  X1 to X19 and X21 to X24 run with the ruleset `Active`; only X6 and X20
  run in `Evaluate`. A lab pass in `Active` says nothing about what the
  real shadow period (`Evaluate`) can show.
- That the real group check is named `validate / validate` and comes from a
  `merge_group` run of `repository-checks.yml`. The queue trial of checklist
  step 8 records that on `rulespec-us`.
- What happens to the rule if the organization's plan changes.
- Anything about the fallback: the cloud condition, the App binding and an
  App check on a merge-group commit need real settings.
