# Local validate-rulespec parity

`axiom-encode ci --repo /path/to/rulespec-<country>` runs the validation gates
declared by that lane's reusable `validate-rulespec.yml` caller. It reads the
four pinned dependency commits and caller inputs, verifies the strict
`.axiom/toolchain.toml` and waiver-set binding, authenticates each local checkout
against `origin/main`, selects changes relative to `--base-ref origin/main`, and
runs every applicable gate in workflow order.

Parity is fail-closed and pin-relative. This release implements these workflow
pins:

| Pin | Executor | Notes |
|---|---|---|
| `34bcfab235c585c47292c95f51be1a4f4f91d29e` | reimplemented gates | money-atom scan includes `programs/` |
| `615c1df9b9ace7deea84da65efd137f46f8bad2b` | reimplemented gates | money-atom scan excludes `programs/` |
| `0effa6a5b05e7fac53902df7d523e909bd7fc48a` | the workflow's own scripts | used by rulespec-us |
| `6f11be2655f79dd0a3b582db46525f58332ca120` | the workflow's own scripts | adds the unmanifested-RuleSpec pre-check |

A caller pinned to another workflow revision is rejected until the pinned
workflow, gate registry, and implementation are updated together. The pinned
workflows ship byte-exact in `src/axiom_encode/ci_parity_workflows/`, and each
is checked against its Git blob id before use. The changed-file classifier
requires the installed `axiom-oracles` VCS commit to match the dependency
declared by the pinned encoder.

## Pins that run the workflow's own scripts

From `0effa6a5` on, `ci` does not reimplement the workflow's inline programs.
It extracts each inline Python script a gate runs from the pinned workflow file
(the heredocs and `python3 -c` programs) and executes it unchanged. The step's
bash glue and the supervised `axiom-encode` subcommands are reproduced in
Python. The scripts cover:

- the shard scope decision,
- the strict toolchain check,
- unsupported tracked paths,
- the unmanifested-RuleSpec pre-check (`6f11be26`),
- reviewed-migration authorization,
- the retired-schema freeze,
- the corpus release object verifier,
- the repository-structure check,
- the waiver ratchet,
- the toolchain-change classifier,
- the active-waiver skip lists,
- the changed-file oracle coverage filter.

`ci` simulates the pull request CI would run:

- **Base and head.** The base is `--base-ref` resolved to a commit and the head
  is `HEAD`. CI validates the pull request's merge commit, so when `HEAD` does
  not contain the base, the report carries a note to rebase for exact parity.
- **Shard matrix.** The matrix is computed exactly as the `shards` job computes
  it. A diff confined to one jurisdiction runs that shard; anything else runs
  the full matrix.
- **Legs run in sequence.** A gate that runs on every leg in CI is reported once,
  with failures labelled by shard. First-shard-only gates run once over the
  repository-wide roots.
- **Waiver audit.** The supervised audit runs once per partition key, with the
  same `--partition-key` and `--partition-keys-json` the matrix passes, so the
  partitioned isolated-recheck budget applies.
- **Order.** The gate order is the workflow's: unsupported paths,
  (unmanifested pre-check,) migration authorization, retired-schema freeze,
  obsolete files, layout, waiver ratchet and audit, generated guard, target
  selection, validate, companion tests, proof validation, money atoms, oracle
  coverage, and repository tests last.
- **Supervised environment.** CI's supervisor starts each supervised subcommand
  from an empty environment. For those gates, `ci` withholds every `AXIOM_*`
  variable the supervisor does not forward and makes Git ignore global and
  system configuration. Unsupervised steps, such as companion tests, get the
  step's own environment.
- **Release object.** It is fetched as the caller configures CI to fetch it:
  from the Supabase registry alone when `corpus-release-registry-url` is set,
  otherwise from `corpus-release-base-url`. The workflow's verifier then checks
  it and rewrites it in canonical form. An object already cached in the corpus
  checkout is re-verified and rewritten instead of being fetched again.
- **Migration authorization.** Reviewed-migration authorization is
  pull-request scoped. When a caller requests it, pass `--pull-request N`, or
  the gate fails the way CI would without a pull-request number.

## Callers that read a workflow toolchain

rulespec-us passes its dependency refs as
`${{ needs.workflow-toolchain.outputs.* }}`, produced by a `workflow-toolchain`
job that reads `.axiom/workflow-toolchain.toml`. `ci` resolves such refs only
through a reviewed resolver:

- the validate-rulespec job lists the resolver job in `needs`;
- the resolver job is exactly a plain checkout plus one bash step;
- the step's script has a sha256 listed in
  `RECOGNIZED_WORKFLOW_TOOLCHAIN_RESOLVERS`;
- every consumed output maps to that step's output of the same name.

`ci` then executes the resolver script against the checkout, with the job's own
validation and error messages. Any other resolver, or any other expression,
fails closed.

The caller's `${{ vars.NEXT_PUBLIC_SUPABASE_URL }}` and
`${{ vars.NEXT_PUBLIC_SUPABASE_ANON_KEY }}` registry inputs cannot be read
locally.

- **URL:** `--corpus-release-registry-url` supplies the URL. The default is the
  organization's registry project.
- **Anon key:** `--corpus-release-registry-anon-key` supplies the key. It
  defaults to `$NEXT_PUBLIC_SUPABASE_ANON_KEY` and is required when the caller
  reads it from a variable. It is a public read credential. Every fetched
  object is pinned by content hash and signature-verified.
- **Reporting:** the report lists how each expression was resolved.

Caller inputs are type-checked against the pinned workflow's `workflow_call`
declarations. Undeclared inputs are rejected, and omitted inputs take the
workflow's defaults (for example
`validate-roots: statutes regulations policies`).

## Running against rulespec-us

```console
axiom-encode ci --repo ../rulespec-us \
  --corpus-path ../axiom-corpus \
  --engine-path ../axiom-rules-engine \
  --rulespec-us-path ../rulespec-us-pinned \
  --encode-path ../axiom-encode-pinned \
  --corpus-release-public-key "$PUBLIC_KEY" \
  --corpus-release-retired-public-key "$RETIRED_PUBLIC_KEY" \
  --apply-public-key "$APPLY_PUBLIC_KEY" \
  --eval-public-key "$EVAL_PUBLIC_KEY" \
  --corpus-release-registry-anon-key "$ANON_KEY" \
  --jobs 6
```

Every dependency checkout must be at its pin:

- rulespec-us pins itself (`rulespec_us_ref`) for canonical targets, so
  `--rulespec-us-path` must be a second checkout at that commit, not the
  repository under test.
- The engine checkout needs a built binary (`cargo build`), as CI builds one.

rulespec-us has close to two thousand active waivers, and the waiver audit
fingerprints every one of them on every pull request. CI spreads that work over
the matrix; locally, `--jobs N` runs up to N audit partitions (and validation
chunks and companion-test groups) in isolated worker processes:

- Each worker receives the verification keyring on stdin, never through the
  environment.
- Each worker starts an isolated interpreter (`python -I`), imports the encoder
  from the parent's package root, and refuses to run if that is not the module
  the parent verified, so a package in the working directory or on
  `PYTHONPATH` cannot stand in for it.
- Each worker caps the audit's own process fan-out at `cpu_count // N`. The
  audit's fingerprints do not depend on that count by design.
- Without `--jobs`, the invocations run in-process one at a time.

Retrieve the public values with:

```console
gh api /orgs/TheAxiomFoundation/actions/variables/AXIOM_CORPUS_RELEASE_PUBLIC_KEY --jq .value
gh api /orgs/TheAxiomFoundation/actions/variables/AXIOM_CORPUS_RELEASE_RETIRED_PUBLIC_KEY --jq .value
gh api /orgs/TheAxiomFoundation/actions/variables/AXIOM_ENCODE_APPLY_SIGNING_PUBLIC_KEY --jq .value
gh api /orgs/TheAxiomFoundation/actions/variables/AXIOM_ENCODE_EVAL_SIGNING_PUBLIC_KEY --jq .value
gh api /repos/TheAxiomFoundation/rulespec-us/actions/variables/NEXT_PUBLIC_SUPABASE_ANON_KEY --jq .value
```

Without `--apply-public-key`, signed encoder apply manifests cannot be
verified. A pull request whose RuleSpec changes CI accepts on their manifests
then fails the generated guard locally, and the report says so.

`validate-roots` values containing glob characters (`*`, `?`, `[`) are rejected.
The workflow expands them unquoted in several places, and `ci` does not
reproduce that expansion. No lane uses one.

`validate-roots: auto` finds jurisdictions in the order the workflow's
`for dir in */` loop visits them on the runner. Bash sorts the matches with
their trailing slash in code-point order, so `us-ak/` comes before `us/`
(`-` sorts before `/`). In rulespec-us the first shard, which also runs the
repository-wide gates, is therefore `us-ak`, and `us` is last.

## Trust boundary

The public trust roots must be supplied only through
`--corpus-release-public-key`, `--corpus-release-retired-public-key`,
`--apply-public-key` and `--eval-public-key`. The command never reads them from
the environment.

- **Keyring.** The command constructs a verification-only keyring in the
  library, in the order the supervisor provisions it (current key, then
  retired keys).
- **Retired keys.** They are honored only for pins whose supervisor provisions
  them (`0effa6a5` and later). Older pins ignore them, with a note.
- **Apply and eval roots.** They verify persisted apply manifests and eval
  evidence, as the supervisor's apply and eval roots do. As the supervisor
  requires, every root must be distinct across the corpus, retired, apply and
  eval roles.
- **Capabilities.** It acquires no signing capability and never runs `--apply`.
- **Writes.** It writes temporary report inputs. The sole non-report write is
  caching the pinned public release object at the workflow-defined corpus
  release path, which the corpus checkout's cleanliness check exempts.
- **Network.** It downloads that object unless `--offline` is set.

## Dependency and encoder identity

A dependency checkout whose `HEAD` differs from its caller pin fails resolution
and names both SHAs. `--allow-ref-mismatch` permits all gates to run, but a
successful run receives the qualified `PASS-WITH-MISMATCHED-DEPS` verdict and
exit code 3. Its mandatory final banner enumerates every dependency mismatch.

- Plain `PASS` and exit code 0 require all four dependency checkouts to match.
- Gates run against a fresh checkout of the rules repository's committed
  `HEAD`, as `actions/checkout` produces:
  - It is an independent transport clone (`--no-local`), so every object is
    re-hashed on receipt, and none is shared with or linked to the source.
  - It is made from an empty template under empty global and system Git
    configuration and attributes, named like the source checkout, with origin
    pointing at the source's origin.
  - It lives in a temporary directory and is removed afterwards.

  None of the following in the source checkout or your user configuration
  reaches a gate:
  - uncommitted edits;
  - untracked or ignored files;
  - index flags (`assume-unchanged`, `skip-worktree`);
  - sparse-checkout patterns;
  - smudge filters, attribute files or hooks.

  A tree with paths that would collide on the local filesystem (case or
  Unicode normalization) fails closed, because CI's Linux checkout keeps
  every path. When the source has uncommitted changes, the report notes that
  only committed `HEAD` was validated, and the JSON report's
  `validated_commit` names that commit. `--base-ref` and `HEAD` resolve in the
  source checkout, so branch-relative refs such as `@{upstream}` work and name
  their branch. Paths in the report refer to the source checkout.
- Dependency and encoder checkouts whose index hides edits
  (`assume-unchanged`, `skip-worktree`) count as dirty.
- ci's own Git calls, its embedded scripts and repository pytest run without
  ambient `GIT_*` values. Embedded scripts and pytest also run without
  `PYTHON*` interpreter overrides, and pytest without `PYTEST_*` options. CI's
  steps never see any of these.
- Gate failures and resolution failures remain `FAIL` with exit code 1.
- JSON reports expose the same value in `verdict` and list structured
  `dependency_mismatches`.

The imported (ambient) `axiom_encode` implementation is also fail-closed against
the caller's encoder pin.

- Source-checkout `HEAD` is compared when the package runs from
  `<checkout>/src/axiom_encode`. Modified or untracked encoder sources at the
  pinned `HEAD`, or a status that cannot be read, are a mismatch too. An
  installed copy has no source identity and is always a mismatch.
- The interpreter must be the caller's `python-version` (major.minor), which
  CI sets up, and must not run optimized (`-O` or `PYTHONOPTIMIZE`, which
  strip the assertions in-process gates rely on). Either difference is a
  `python` mismatch under the same flag.
- Exact parity means running this tool from the pinned encoder checkout.
  `--allow-encoder-mismatch` is the honest mode for development loops on newer
  encoders, and produces the same qualified verdict and mismatch banner.
- Executing every gate in a subprocess sourced from the pinned checkout is the
  longer-term gold path and is intentionally out of scope for this command
  version.

## Troubleshooting

- **"RuleSpec module is not inside a canonical country checkout" on a
  heavily loaded machine.** The encoder's checkout routing probes Git with a
  two-second timeout, and a timed-out probe resolves the checkout as not
  canonical. Rerun with fewer `--jobs`, or when the machine is less loaded.
- **`No module named pytest` in repository tests.** CI installs `pytest` and
  `pyyaml` into its interpreter; install them into the environment `ci` runs
  from.
