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
  6 hours). The owner extends it with `heartbeat`. A lease is taken from its
  owner only 120 seconds after it expired, because worker clocks differ.
- A lease is identified by its id. After the lease expired and another worker
  claimed the unit, the first worker's `heartbeat` and `report` are refused
  with "Stop this unit". If nobody claimed the unit again, a late `report`
  is still accepted, so a slow encode keeps its result.
- A `refused` unit is not claimed again by the same encoder version. A worker
  on another encoder version can claim it. After 3 refusals the unit is
  `blocked` until a person sets it `ready`.
- Units of a package that is `paused` or `blocked` are never claimed.
- The states after the encode hold no lease. The landing tool reports them.

## Where the state lives

Three kinds of JSON document:

| Document | Content | Written by |
| --- | --- | --- |
| `packages.json` | The priority list: id, program, jurisdiction, rank, status, what unblocks it, who added it | `add`, `move`, `package` |
| `units/<package>.json` | The derived units of one package, with build inputs (corpus ref, RuleSpec ref, bundle date) and document counts | `build` only |
| `ledger.json` | One entry per unit that is not simply `ready`: lease, owner, outcome, attempts | `claim`, `heartbeat`, `release`, `report`, `build` |

A backend stores the documents with one revision for all of them. Every
write is a compare-and-swap on that revision: read, decide, commit only if
the revision did not move, else read again. Two workers that race for one
unit therefore cannot both win.

### Backends

- `--store-dir DIR` (or `AXIOM_ENCODE_QUEUE_DIR`): a local directory with a
  lock file. For tests and for one machine.
- `--repo OWNER/NAME` (or `AXIOM_ENCODE_QUEUE_REPO`), `--branch`
  (default `encoding-queue-state`), `--prefix` (default `central-queue`):
  a branch of a GitHub repository. A write builds a commit on the commit it
  read (Git Data API) and moves the branch with `force: false`. GitHub
  refuses that update when the branch moved, which is the compare-and-swap.
  All calls go through `gh api`, so a worker needs only its `gh` login with
  write access to the repository. No new secret is stored on a machine.

Why a GitHub branch and not a database table: a database claim needs a
credential on every worker (a service key on each laptop, or a new public
endpoint that decides who may claim). Every worker already has `gh`. The ops
page already reads this branch. The history of the branch is the audit log.
The cost is one commit per write; with 3 to 5 machines that claim in batches
and send one heartbeat per machine every 10 minutes, that is some hundreds of
small commits a day on a branch that holds only state.

The queue documents are public when the repository is public. Use a machine
label, not a host name, for `--machine`.

## Membership is derived

`axiom-encode queue build --corpus <checkout> --rulespec <checkout>
[--open-prs prs.json]` computes the units of every package:

1. The in-scope documents of the jurisdiction in the program bundle
   (`manifests/program-bundles/<program>.yaml` in the corpus checkout).
2. The rows of the pinned corpus under those documents, 150 to 3000
   characters, the largest unit when a parent and a child both qualify,
   without index and example headings.
3. Minus units on RuleSpec main: the target file exists, or a module cites
   the unit, a parent of it or a part of it.
4. A unit whose target file is in an open pull request keeps `open_pr` and
   shows as `candidate`. `prs.json` is the output of
   `gh pr list --state open --json number,files` on the RuleSpec repository.

Units are ranked: money or percent amounts first, then computation parts
(income, deductions, benefit amount, assets), then 300 to 2500 characters.

A package with no in-scope document in the pinned corpus is blocked, and the
build writes what unblocks it. A unit that landed on main is not a member of
the next build, and its ledger entry is removed. The build makes no network
call and no model call.

## Commands

```
axiom-encode queue [--store-dir DIR | --repo OWNER/NAME] <command>

add      --program snap --jurisdiction us-nc [--before PACKAGE] --person P
move     PACKAGE [--before PACKAGE] --person P
package  PACKAGE --status active|paused|blocked [--unblock TEXT] --person P
build    --corpus DIR --rulespec DIR [--open-prs FILE] [--package ID ...] --person P
status   [--json] [--encoder-version V]
claim    --machine M --person P [--count N] [--package ID] [--lease-seconds S]
heartbeat --lease CITATION=LEASE_ID [--lease ...] [--encoding]
release  CITATION --lease-id ID
report   CITATION --state candidate|refused|blocked|signing|in_train|landed|ready
         [--lease-id ID | --person P] [--category C] [--unblock TEXT]
         [--candidate-pr N] [--signing-run N] [--train-pr N]
```

Machine commands print JSON. Exit code 0: done. Exit code 1: refused, the
queue is unchanged and the reason is on stderr. `claim` with nothing to do
prints `{"claimed": []}` and exits 0.

## How a worker will use it (next milestone, not in this change)

A parallel local runner keeps its own rules (no run on battery, one runner
per machine, subscription login only). Only its source of units changes:

1. At start, and whenever a worker thread is free:
   `claim --count <free workers> --lease-seconds 1800`. The result replaces
   the worklist file.
2. One timer per machine sends `heartbeat --encoding` with all its leases
   every 10 minutes. On "Stop this unit" the runner stops that unit and
   discards its result.
3. After each unit: `report --state candidate` for a clean candidate,
   `report --state refused --category <first failed check or finding>` for a
   refusal, `release` for an infrastructure error, a rate limit or an
   interrupt.
4. At shutdown: `release` every lease it still holds.

The landing tools then report `signing`, `in_train` and `landed`, and a
scheduled `build` reconciles the queue with RuleSpec main.

## Limits of this milestone

- The GitHub backend is tested against a fake of the Git Data API. It has
  not written to a real branch.
- A `candidate` whose pull request was closed without a merge stays a
  `candidate` in the ledger until a person sets it `ready`.
- `status` reads the whole queue. The ops page will read the same documents.
