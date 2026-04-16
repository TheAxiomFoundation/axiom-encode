# Prompt versioning

AutoRAC ships its agent prompts as Python string constants in
`src/autorac/prompts/`. Because encoder and reviewer behaviour is largely
determined by these prompts, we track their evolution with an explicit
version field in addition to the automatically recorded prompt hash.

## Why prompts are versioned

- **Debuggability.** When we look at an old encoding run in the DB, the
  prompt hash tells us *which* prompt ran, but a semantic version number
  tells us *which generation* of the prompt ran, which is far easier to
  reason about across many commits.
- **Registry correlation.** Logs, dashboards, and paper experiments
  reference encoder/reviewer generations. Humans read "encoder v3" much
  faster than they read a 40-character hash.
- **Rollback signalling.** A version bump is a clear signal in code review
  that reviewers should think about compatibility with downstream artifacts
  (cached runs, saved traces, reference encodings).

## When to bump

Bump the `__version__` in the relevant prompt module when the prompt text
changes in a way that could materially affect model behaviour. Examples:

| Change                                                             | Bump? |
| ------------------------------------------------------------------ | ----- |
| Adding or removing a numbered rule                                 | Yes   |
| Changing the required JSON output schema                           | Yes   |
| Adding a new blocking check                                        | Yes   |
| Swapping examples that illustrate the rules                        | Yes   |
| Fixing a typo that does not change meaning                         | No    |
| Whitespace / formatting-only edits                                 | No    |

If in doubt, bump. The cost of a spurious bump is negligible; the cost of
missing a real semantic change is a confusing audit trail.

## Module scope

Each module owns its own version:

- `autorac.prompts.encoder.__version__` — encoder prompt.
- `autorac.prompts.reviewers.__version__` — **bundle** of all four reviewer
  prompts. Any change to any reviewer prompt bumps this single integer.

We deliberately keep reviewer prompts on a shared version to reflect the
fact that they are run and consumed as a set.

## Relationship to the encoding DB

The encoding DB (see `src/autorac/harness/encoding_db.py`) records the raw
SHA-256 of each prompt as part of every run. That hash is the authoritative
fingerprint. `__version__` is a looser, human-friendly label that rides on
top:

- The hash uniquely identifies prompt text.
- The version identifies a *generation* that may span several hashes
  during development (because hash changes whenever any character changes,
  including non-semantic edits).

When analysing the DB, you should join on hash for correctness and use
version as the user-facing grouping key.
