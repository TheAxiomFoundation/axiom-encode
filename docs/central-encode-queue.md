# Central encode queue

One shared list of what to encode next, for every machine and every person.

- It holds the priority list of (program, jurisdiction) packages and the units
  of each package.
- A worker reserves units with an atomic claim. A claim is a lease that
  expires, so the work of a stopped machine returns to the queue.
- It shows what is reserved or encoding now, by which machine and person.
- People add packages and change their order.

**Status: first milestone.** Nothing calls the queue yet. No runner, workflow,
admission or signing path reads it, and it changes no validation. The module
is `axiom_encode.encode_queue`; the command is `axiom-encode queue` (also
`python -m axiom_encode.encode_queue`).

## State of a unit

```
ready -> reserved -> encoding -> candidate -> signing -> in_train -> landed
              |          |           |           |           |
              +----------+-----------+-----------+-----------+--> refused (category)
              |          |           |           |           +--> blocked (what unblocks it)
              +----------+--> ready (release, or the lease expired)
```

| State | Meaning | Who sets it |
| --- | --- | --- |
| `ready` | Can be claimed | Derived: no ledger entry, or an expired lease |
| `reserved` | A worker holds a lease and has not started | `claim` |
| `encoding` | The worker runs the encode | `heartbeat --encoding` |
| `candidate` | A clean candidate exists (a RuleSpec pull request) | `report`, or the build when it finds an open pull request for the target file |
| `signing` | A signing run is dispatched | `report` (landing tool) |
| `in_train` | The signed unit is in a landing pull request | `report` (landing tool) |
| `landed` | The module is on RuleSpec main | `report`; the next build removes the unit |
| `refused` | The encoder or the reviewer refused the unit; needs a category | `report` |
| `blocked` | Cannot proceed; says what unblocks it | `report`, a package status, or the build |

Rules:

- `reserved` and `encoding` hold a lease (default 30 minutes, 1 minute to
  6 hours). The owner extends it with `heartbeat`.
- **The lease clock is the store's, not the worker's.** With the GitHub
  backend it is the `Date` header of GitHub's answer. A worker with a wrong
  clock cannot take a live lease or lose its own. A lease is taken from its
  owner only 120 seconds after it expired.
- A lease is used with its id **and** its owner (machine and person). The
  ledger is public, so the id alone proves nothing.
- After the lease expired and another worker claimed the unit, the first
  worker's `heartbeat` and `report` are refused with "Stop this unit". If
  nobody claimed the unit again, a late `report` is still accepted, so a slow
  encode keeps its result.
- `heartbeat` answers per lease. A lost lease does not stop the extension of
  the valid ones.
- A unit is never ready for an encoder version that it was refused under.
  The versions are remembered through later claims, releases and expiries.
  After 3 refusals the unit is `blocked` until a person sets it `ready`.
- Units of a package that is `paused` or `blocked` are never claimed.
- The states after the encode hold no lease. The landing tool reports them.

### Requests can be sent again

A worker that got no answer (a timeout) sends the same request again:

- `claim --request-id ID`: the same id returns the same units while their
  leases are held. It never reserves a second set.
- `report` and `release`: the same call after it was applied gets the same
  answer (`"repeated": true`). A different outcome for a closed lease is
  refused.

## Where the state lives

| Document | Content | Written by |
| --- | --- | --- |
| `packages.json` | The priority list: id, program, jurisdiction, rank, status, what unblocks it, who added it | `add`, `move`, `package` |
| `units/<package>.json` | The derived units of one package, with build inputs (corpus ref, RuleSpec ref, bundle date) and document counts | `build` only |
| `ledger/<package>.json` | One entry per unit of the package that is not simply `ready`: lease, owner, outcome, attempts, refused versions | `claim`, `heartbeat`, `release`, `report`, `build` |

A backend stores the documents with one revision for all of them. Every
write is a compare-and-swap on that revision: read, decide, commit only if
the revision did not move, else read again. Two workers that race for one
unit therefore cannot both win.

### No stored value is trusted

Anyone with write access can edit the queue branch. So every document is
checked when it is read:

- Each document has a strict schema. An unknown field, a missing field or a
  wrong type refuses the operation with a one-line message.
- A citation must match the citation grammar: `<jurisdiction>/<kind>/...`,
  at least three segments, only the characters that corpus citation paths
  use (letters, digits, space, `. _ : ( ) -` and the en dash), no empty or
  dot-only segment, at most 300 characters.
- **The RuleSpec target path is never stored.** `claim` computes it from the
  citation each time (`citation_rulespec_path`). A `target_path` field in a
  queue document is an unknown field.
- A lease must be consistent: it ends at most one lease length after its last
  heartbeat, its heartbeat is not in the future, and its length is 1 minute
  to 6 hours. A far-future lease is invalid.
- Free text (`note`, `unblock`) is at most 500 characters with no control
  character.

A document that fails a check stops the queue for everyone until a person
repairs it. That is deliberate: the queue never guesses.

### Backends

- `--store-dir DIR` (or `AXIOM_ENCODE_QUEUE_DIR`): a local directory with a
  lock file. For tests and for one machine.
- `--repo OWNER/NAME` (or `AXIOM_ENCODE_QUEUE_REPO`), `--branch`
  (default `encoding-queue-state`), `--prefix` (default `central-queue`),
  `--cache-dir` (or `AXIOM_ENCODE_QUEUE_CACHE_DIR`): a branch of a GitHub
  repository. All calls go through `gh api`, so a worker needs only its `gh`
  login with write access to the repository. No new secret is stored on a
  machine.

How the GitHub backend writes and fails:

| Case | Behaviour |
| --- | --- |
| Write | One GraphQL `createCommitOnBranch` request with `expectedHeadOid`. GitHub applies it only when the branch still points at the commit that was read |
| The branch moved (stale head) | Read again, decide again, write again |
| Any other refusal (a branch rule, no permission) | An error at once. It is not retried |
| Rate limit (403 or 429 with the server's hints, or `RATE_LIMITED`) | Wait `Retry-After` (or until the reset time, at most 300 s), then send the same request; at most 4 waits |
| No answer (timeout, lost connection, 5xx) | Read the branch and look for the request id of this write (`Queue-Request:` in the commit message). Found: done. Not found: decide again; request ids make the repeat safe |
| The branch does not exist | An error. The queue never creates its branch, and a missing branch is never an empty queue |
| Read | Branch head, tree, and only the blobs that are not in the cache. Blobs are cached by SHA and checked against it |

Why a GitHub branch and not a database table: a database claim needs a
credential on every worker (a service key on each laptop, or a new public
endpoint that decides who may claim). Every worker already has `gh`. The ops
page already reads this branch. The history of the branch is the audit log.

The queue documents are public when the repository is public. Use a machine
label, not a host name, for `--machine`.

## Write volume

GitHub's secondary limit allows roughly 166 content-creating requests per
hour per login. So a machine must not write once per unit. The rule:

- **One batch per machine per interval.** `axiom-encode queue apply --ops`
  (and `EncodeQueue.apply`) runs a list of claim, heartbeat, report and
  release operations in **one commit**. The runner of a machine collects the
  operations of all its workers and sends one batch every 180 seconds, with
  a random offset per machine.
- **Claim in batches.** One claim takes up to 40 units; a machine refills all
  its free workers in one operation.
- **Heartbeats ride along.** A lease with more than half its time left is not
  rewritten, so a batch with nothing new writes nothing.
- **One ledger shard per package.** A commit carries only the shard it
  changed.

Numbers for 5 machines with 8 workers each, one batch per 180 seconds
(simulation in `tests/test_encode_queue.py`, 20 intervals, 400 units):

| | Writes per hour | Write requests per commit | API calls per commit |
| --- | --- | --- | --- |
| Machines write at different seconds (the normal case) | 100 commits, 100 requests | 1.0 | 4.2 |
| All five write in the same second, every time (worst case) | 100 commits, about 300 requests | 3.0 | 12.3 |
| Same load with one write per unit operation (not supported) | about 1,200 | — | — |

The count depends on machines and the interval, not on workers or units: 20
commits per machine per hour. With a random offset the expected stale rate is
about 4 percent, so about 104 requests per hour for 5 machines. If all five
machines use one login that is under the limit of 166; an interval of 300
seconds gives 60 per hour. Machines with different logins each have their
own limit.

### History growth

At 100 commits per hour around the clock the branch gets at most 72,000
commits a month, each with one changed shard (tens of kilobytes, compressed
by git). A squash would be a force-update, which the compare-and-swap forbids.
The bound is a **new branch per generation**:

1. A person (later: a scheduled job) creates the next branch, for example
   `encoding-queue-state-2026-12`, with one commit that holds the current
   documents.
2. The same step writes `"moved_to": "<new branch>"` into `packages.json` of
   the old branch, as a normal compare-and-swap write.
3. From that commit on, every operation on the old branch is refused with
   "this queue is closed; it moved to branch ...". A worker with an old
   configuration can never write to the old generation.
4. The old branch is deleted after a retention time. Its history goes with it.

This milestone implements step 3 (the refusal). The rollover command is part
of the scheduled-build milestone.

## Membership is derived

`axiom-encode queue build --corpus <checkout> --rulespec <checkout>
[--open-prs prs.json]` computes the units of every package:

1. The in-scope documents of the jurisdiction in the program bundle
   (`manifests/program-bundles/<program>.yaml` in the corpus checkout).
2. The rows of the pinned corpus under those documents, 150 to 3000
   characters, the largest unit when a parent and a child both qualify,
   without index and example headings, with a valid citation.
3. Minus units on RuleSpec main: the target file exists, or a module cites
   the unit, a parent of it or a part of it. A unit under two overlapping
   documents is counted once.
4. A unit whose target file is in an open pull request keeps `open_pr` and
   shows as `candidate`. `prs.json` is the output of
   `gh pr list --state open --json number,files` on the RuleSpec repository.

Units are ranked: money or percent amounts first, then computation parts
(income, deductions, benefit amount, assets), then 300 to 2500 characters.

- A unit belongs to one package: the highest-ranked package that lists it.
- A package with no in-scope document in the pinned corpus is blocked, and
  the build writes what unblocks it.
- A unit that landed on main is not a member of the next build, and its
  ledger entry is removed.
- **A unit under a live lease is never removed.** Its lease is marked
  `superseded`. The owner sees `"superseded": true` in the answer to its next
  heartbeat, stops the unit and releases it. A report on a superseded lease
  is answered with state `superseded` and is not recorded.

The build makes no network call and no model call.

## Commands

```
axiom-encode queue [--store-dir DIR | --repo OWNER/NAME] <command>

add      --program snap --jurisdiction us-nc [--before PACKAGE] --person P
move     PACKAGE [--before PACKAGE] --person P
package  PACKAGE --status active|paused|blocked [--unblock TEXT] --person P
build    --corpus DIR --rulespec DIR [--open-prs FILE] [--package ID ...] --person P
status   [--json] [--encoder-version V]
claim    --machine M --person P [--count N] [--package ID] [--lease-seconds S]
         [--request-id ID]
heartbeat --machine M --person P --lease CITATION=LEASE_ID [--lease ...] [--encoding]
release  CITATION --lease-id ID --machine M --person P
report   CITATION --state candidate|refused|blocked|signing|in_train|landed|ready
         [--lease-id ID --machine M] --person P [--category C] [--unblock TEXT]
         [--candidate-pr N] [--signing-run N] [--train-pr N] [--note TEXT]
apply    --ops FILE|-      (a JSON list of claim, heartbeat, report, release)
```

`--machine` and `--person` default to `AXIOM_ENCODE_QUEUE_MACHINE` and
`AXIOM_ENCODE_QUEUE_PERSON`.

Machine commands print JSON. Exit code 0: done. Exit code 1: refused, the
queue is unchanged and the reason is one line on stderr (never a traceback).
`claim` with nothing to do prints `{"claimed": []}` and exits 0. `apply`
prints one result per operation (`"ok": true|false`); a refused operation
does not stop the others.

An operation of `apply` is the command with its options as fields:

```json
[
  {"op": "report", "citation": "...", "state": "candidate", "lease_id": "...",
   "candidate_pr": 2392, "machine": "m1", "person": "octocat"},
  {"op": "heartbeat", "leases": {"<citation>": "<lease id>"}, "encoding": true,
   "machine": "m1", "person": "octocat"},
  {"op": "claim", "count": 4, "encoder_version": "0.2.2399",
   "request_id": "m1-000042", "machine": "m1", "person": "octocat"}
]
```

## How a worker will use it (worker-wiring milestone, not in this change)

A parallel local runner keeps its own rules (no run on battery, one runner
per machine, subscription login only). Only its source of units changes. The
runner process is the per-machine agent: its worker threads put operations
in a list, and one thread sends the list every 180 seconds (plus a random
offset per machine) with `apply`:

1. `report` for each unit that finished since the last batch: `candidate`
   for a clean candidate, `refused` with the first failed check or finding as
   the category. `release` for an infrastructure error, a rate limit, a
   timeout or an interrupt.
2. One `heartbeat` (with `encoding`) for all leases it still holds. For a
   lease that answers `"ok": false` or `"superseded": true` the runner stops
   that unit and discards its result.
3. One `claim` with `count` = free workers plus a small reserve, and a new
   `request_id`. The units of the answer replace the worklist rows.
4. When a batch gets no answer, the runner sends the same list again with
   the same request id.
5. At shutdown: one last batch that releases every lease it still holds.

The landing tools then report `signing`, `in_train` and `landed`, and a
scheduled `build` reconciles the queue with RuleSpec main.

## Limits of this milestone

- The GitHub backend is tested against a fake of the GitHub API with a real
  expected-head check. It has not written to a real branch. The exact error
  type GitHub returns for a stale head and for a rule refusal must be
  confirmed in the first live canary.
- Optimistic concurrency fits tens of writers per branch, not hundreds.
- A `candidate` whose pull request was closed without a merge stays a
  `candidate` in the ledger until a person sets it `ready`.
- The machine and person of a request are not authenticated by the queue.
  They are labels; write access to the repository is the trust boundary.
