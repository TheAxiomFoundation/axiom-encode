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
   branch `axiom/signed-backfill-us-<run_id>-<attempt>`; a failure records the
   failing step and retries once; a run still waiting for approval on a
   `main` that has since moved is cancelled and sent again on the new tip.
2. For every item in review, read its PR: merged is done, closed is blocked.
3. If the queue is active, dispatch pending items until `max_in_flight` (4)
   runs are open, pinned to `main`'s tip and the corpus and rules-engine refs
   in rulespec-us `.axiom/workflow-toolchain.toml`.
4. Save the state file and write a summary with counts, runs waiting for
   approval, and what changed.

Dispatch stops, with a note in the summary, when rulespec-us `main` pins a
different corpus release from the one the queue was built from.

## Statuses

| Status | Meaning |
| --- | --- |
| `pending` | Waiting to be dispatched |
| `dispatched` | A run is queued, waiting for approval, or running |
| `in_review` | The run opened a draft RuleSpec PR that is still open |
| `done` | The PR merged, or the citation was already encoded when the queue was built |
| `blocked` | Needs a person: failed twice, attempt budget used up, PR closed, or a success with no PR |

Every dispatch is kept in the item's `attempts` list with its refs, run link,
result, and failing step.

## Items

`build` reads the corpus release that rulespec-us `main` pins and, for each
jurisdiction's SNAP scopes in `manifests/state-snap-manual-agent-queue.yaml`,
takes the largest subtrees of at most 20,000 characters. A small container
(a Utah manual topic) is one item; a large one (the Oregon notebook) is split
into its pages. No item contains another. Items are dropped when they have no
text, look like a table of contents, or never mention SNAP, and the counts are
recorded under `build.excluded`. Citations that already have a signed manifest
on rulespec-us `main` start as `done`.

## Operating it

Run the workflow by hand with an `action`:

- `build` with `queue_id` and `jurisdictions` makes a new paused queue. It
  never overwrites an existing queue.
- `activate` / `pause` flip the queue and, on activate, tick once.
- `tick` with `dry_run` shows what would be dispatched without changing
  anything.

To retry a blocked item, pause the queue, change its `status` back to
`pending` on the state branch, and activate it again. Only edit the state
branch while the queue is paused, so a tick never overwrites your change.
