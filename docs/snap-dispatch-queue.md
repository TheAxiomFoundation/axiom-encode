# SNAP dispatch queue

An hourly workflow that sends SNAP manual pages to the targeted encode workflow
and records what happened to each one. It replaces nothing in the encode path:
every item is an ordinary `targeted-signed-reencode.yml` run on the current
rulespec-us `main` tip, with the same `production-signing` approval and the
same draft RuleSpec pull request as an ad hoc encode.

- Script: `scripts/snap_dispatch_queue.py`
- Workflow: `.github/workflows/snap-dispatch-queue.yml` (hourly at :17, and by hand)
- State: `queues/<queue_id>.json` on the `encoding-queue-state` branch

## Each tick

1. For every dispatched item, read its run. A success looks up the draft PR on
   branch `axiom/signed-backfill-us-<run_id>-<attempt>`. A failure records the
   failing step and retries once; a rejected signing approval or a used-up
   attempt budget blocks at once. The encode workflow needs rulespec `main`
   to stay at the pinned tip until it opens its PR, so an unfinished run on a
   `main` that has since moved is cancelled, and a run that failed because
   `main` moved is not counted; both are sent again on the new tip, up to six
   times in a row. A run that already opened its PR is never cancelled or
   redone, whatever its conclusion: its PR is tracked instead.
2. For every item in review, read its PR: merged is done, closed is blocked.
3. If the queue is active, dispatch pending items until `max_in_flight` (4)
   runs are open, pinned to `main`'s tip and the corpus and rules-engine refs
   in rulespec-us `.axiom/workflow-toolchain.toml`. A run this dispatcher
   started but never recorded (a tick whose save failed) is adopted for its
   item instead of dispatching the item again.
4. Save the state file when anything changed, and write a summary with
   counts, runs waiting for approval, and what changed.

Paused queues are still reconciled every hour; only dispatch stops. A
scheduled tick skips itself when another run of the workflow is queued or
running, so it never displaces a pending manual action.

Dispatch stops, with a note in the summary, when rulespec-us `main` pins a
different corpus release from the one the queue was built from.

## Statuses

| Status | Meaning |
| --- | --- |
| `pending` | Waiting to be dispatched |
| `dispatched` | A run is queued, waiting for approval, or running |
| `in_review` | The run opened a draft RuleSpec PR that is still open |
| `done` | The PR merged, or the citation was already encoded when the queue was built |
| `blocked` | Needs a person: failed twice, cancelled three times by someone else, the encode workflow's attempt budget used up, PR closed, or a success with no PR |

Every dispatch is kept in the item's `attempts` list with its refs, run link,
result, and failing step.

## Items

`build` reads the corpus release that rulespec-us `main` pins and, for each
jurisdiction's SNAP scopes in `manifests/state-snap-manual-agent-queue.yaml`,
takes the largest subtrees of at most 20,000 characters. A small container
(a Utah manual topic) is one item; a large one (the Oregon notebook) is split
into its pages. No item contains another.

Items are dropped when they have no text, look like a table of contents, or
never mention SNAP (or "All Programs", the Utah manual's heading for shared
policy). A PDF page between two SNAP pages is kept, since it is usually one
rule running across a page break. Counts are under `build.excluded` and the
dropped citations under `build.excluded_citations`, for review. Citations
that already have a signed manifest on rulespec-us `main` start as `done`;
only exact citation matches are detected.

A build never overlaps an existing queue. Rebuilding a queue on a new corpus
release while keeping its history is not supported yet.

## Operating it

Run the workflow by hand with an `action`:

- `build` with `queue_id` and `jurisdictions` makes a new paused queue. It
  never overwrites an existing queue.
- `activate` / `pause` flip the queue and, on activate, tick once.
- `requeue` with `queue_id` and `citation` sends a blocked item again with a
  fresh retry and cancellation budget. Its earlier attempts stay in the file.
- `dry_run` on `tick` or `activate` shows what would happen without
  dispatching, cancelling, or saving. Use `activate` with `dry_run` to
  preview a paused queue.

The encode workflow keeps its own budget: three failed runs in a row per
citation within seven days, including runs that failed at PR creation because
`main` moved. `requeue` does not reset it. For an item blocked on that budget,
raise the citation's entry in the `ATTEMPT_BUDGET_BY_CITATION_JSON` repository
variable before requeueing it.

Change queue state only through these actions, not by editing the state
branch: every action runs in the same concurrency group as the hourly tick,
so no two writes race. If a save ever fails, the run uploads the state it
could not push as an artifact.
