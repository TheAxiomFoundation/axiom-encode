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
is checked against its Git blob id before use. Both the full and the
changed-file oracle coverage classifiers execute the pinned encoder with only
its sources on `PYTHONPATH` and no ambient `GIT_*` or `PYTHON*` overrides, and
both require the axiom-oracles commit that encoder declares: the parent checks
the installed distribution's module origin, and the child reports and checks
`axiom_oracles.__file__`. Those diagnostics stay out of the classifier's JSON.

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

- **Base and head.** The base is `--base-ref` and the head is `HEAD`, both
  resolved to commits once, before anything is cloned. CI checks out GitHub's
  test merge of the head into the base, so when `HEAD` does not contain the
  base, `ci` validates the same merge (parents base, head), made in its
  temporary checkout, and says so in a note. A head that conflicts with the
  base fails resolution, because GitHub runs no pull_request checks then.
- **Repository and branch.** `github.repository` is the GitHub remote
  `--base-ref` belongs to (`upstream/main` in a fork clone), else origin; pass
  `--github-repository OWNER/NAME` for a renamed or differently cased remote.
  The temporary checkout is named after the repository and its origin is
  `https://github.com/OWNER/NAME`, as in CI's workspace, whatever the local
  directory is called. The base branch (`pull_request.base.ref`) is the
  branch `--base-ref` names, or `main` for a bare commit (`--base-branch`).
- **Runner environment.** Gates run on a UTC clock with `LANG` and `LC_ALL`
  set to `C.UTF-8`, as the runner and the supervisor's child environment have
  them. In-process gates also decode files as UTF-8: `ci` switches its own
  `LC_CTYPE` to a UTF-8 locale (`C.UTF-8`, `en_US.UTF-8`, `UTF-8`, or else the
  caller's own if it is UTF-8) for the run, exports that same locale to the
  child processes that inherit its environment (`C.UTF-8` wherever it exists),
  and fails resolution if none exists outside Python's UTF-8 mode. For pins
  that run the workflow's own scripts, the repository pytest step keeps
  `C.UTF-8`, as CI's does; the legacy pins' step inherits `ci`'s environment.
  The caller's values are restored afterwards.
- **Engine.** As "Build axiom-rules-engine" does, `ci` builds the debug profile
  from a fresh `git archive` of the verified engine pin in a private temporary
  directory, with an empty private `target/` (`--offline` under `--offline`).
  With a recorded engine ref mismatch, it archives that checkout's `HEAD`
  instead and omits the binding receipt. The archive must match the committed
  tree's paths, modes and bytes; archive attributes that alter them fail
  resolution. Ignored or untracked engine files, previous build state and edits
  to the developer's checkout during the build cannot affect the snapshot.
  Every run performs a cold build, as CI's fresh checkout does. Cargo's
  `--message-format=json-render-diagnostics` must report the engine bin target's
  executable as the lexical path `target/debug/axiom-rules-engine`, with no
  symlink from the private target directory down to the binary; redirected
  artifacts fail resolution. Ambient `CARGO_*` and `RUST*` values are removed.
  `HOME` must expand to an absolute path, used identically for config inspection
  and cargo. Configuration-file `build.rustflags` is cleared using
  encoded flags, and command-line overrides restore `profile.dev.overflow-checks`
  and `debug-assertions` to the verified manifest's values (default `true`).
  `build.target` is checked through the artifact path. Sections that cannot
  change what cargo compiles or where it puts the artifact (`term`, `net`,
  `http`, `registry`, `registries`, `credential-alias`, `alias`, `cargo-new`,
  `doc`, `install`, `future-incompat-report`) pass through unchanged. Every
  other Cargo config section or key fails resolution, including wrappers,
  target-specific settings, package/build-override profiles, source
  replacement and `[env]`, whose effects this command cannot neutralize
  exactly. CI builds the engine at `_axiom/axiom-rules-engine` inside the
  rules checkout, so its cargo and rustup also read the rules checkout's own
  `.cargo/config` and `rust-toolchain` files; the local build runs in a private
  snapshot that reads neither. Because the engine manifest declares no
  workspace, cargo also parses every ancestor `Cargo.toml` while looking for
  one. These all fail resolution, because their effect on CI's build cannot be
  reproduced exactly:
  - any non-empty Cargo config in the rules checkout (only CI's cargo would
    read, and validate, it);
  - a committed engine-checkout Cargo config with anything beyond the
    sections above;
  - a `Cargo.toml`, `rust-toolchain` or `rust-toolchain.toml` in the rules
    checkout root, or in any directory above the private snapshot;
  - a symlinked `.cargo` directory or Cargo config in the rules or engine
    checkout, which resolves differently in CI's `_axiom` layout, and any
    Cargo config that cannot be read, a dangling link included.
  Config inspection covers the snapshot's committed `.cargo`, its temporary
  ancestors through `/`, and the absolute `$HOME/.cargo`; the developer's
  engine checkout's ancestors do not participate in this build. When rustup
  is on `PATH`, `rustup show active-toolchain` runs in the snapshot with cargo's
  environment and must report `(default)`; directory, file and environment
  overrides fail resolution. If rustup is absent, the report notes that the
  toolchain source could not be checked.
  Gates get a copied binary in a temporary debug-only view, first on `PATH`
  ("Expose axiom-rules-engine on PATH"). The report identifies the bound binary
  and the active toolchain plus `rustc -vV`'s release and host. A build failure
  fails resolution with cargo's output. A build that only rewrites `Cargo.lock`
  is allowed, as CI builds
  without `--locked`; the report notes the rewrite in the private snapshot and
  retains the receipt for a verified-pin build. The original checkout is never
  built in or modified.
- **Repository pytest.** It runs last, on the first shard, in a temporary venv
  of the caller's `python-version`, requested as `<version>+gil` (uv 0.9.8 or
  later) and then checked to be release CPython with the GIL, the build
  setup-python installs. The venv is provisioned with CI's install sequence
  (`pip`, then `pytest pyyaml`, then the pinned encoder without extras, then on
  changed-file pull requests the oracle coverage classifier and its exact
  axiom-oracles pin), using `uv --no-config` (`--offline` under `--offline`).
  Its environment is built from the run's inputs (`CI`, `GITHUB_ACTIONS`, the
  selection step's `RULESPEC_*` variables, a UTC clock and `C.UTF-8`); ambient
  `AXIOM_*`, `DRAIN_*`, `RULESPEC_*`, `PYTHON*`, `GIT_*` and `R2_*` values never
  reach it, except the caller's `AXIOM_CORPUS_NO_FETCH` opt-out. Pytest uses a
  fresh temporary `HOME`, `GIT_CONFIG_GLOBAL` set to the null device and
  `GIT_CONFIG_NOSYSTEM=1`, so user and system Git configuration cannot change
  repository tests. If the venv cannot be provisioned, `ci` records a
  `repository-test-environment` mismatch and runs pytest in its own interpreter,
  so a passing run is `PASS-WITH-MISMATCHED-DEPS` (exit 3).
- **Shards job first.** The shards job (unsupported paths, the unmanifested
  pre-check, the shard computation) completes after checkout and before the
  validate job resolves the toolchain, dependency checkouts or release object,
  as `needs: shards` orders them. A failed shards step stops the run, as CI
  skips the validate matrix. When a later resolution step fails, the report
  still lists the completed shards-job gates next to `resolution_error`.
- **Shard matrix.** The matrix is computed exactly as the `shards` job computes
  it. A diff confined to one jurisdiction runs that shard; anything else runs
  the full matrix. Every leg prints `RuleSpec validation mode:`, `RuleSpec
  files selected:` and `RuleSpec companion tests selected:`, empty selections
  included, as CI's selection step does.
- **Legs run in sequence.** A gate that runs on every leg in CI is reported once,
  with failures labelled by shard. First-shard-only gates run once over the
  repository-wide roots. A non-first retired-schema freeze failure is a labelled
  gate failure; completed gates remain in the report and later diagnostic gates
  note that CI skips steps on a failed leg.
- **Waiver audit.** The supervised audit runs once per partition key, with the
  same `--partition-key` and `--partition-keys-json` the matrix passes, so the
  partitioned isolated-recheck budget applies. Each partition uses four audit
  workers: the supervisor drops the step's `AXIOM_ENCODE_WAIVER_AUDIT_WORKERS:
  "1"`, so CI's audit falls back to `min(8, cpu_count)`, which is four on the
  public runner. Before the audit, every head-ledger module path must match a
  directory entry's exact spelling; a case alias fails as a missing path, as
  Linux lookup does.
- **Retired-schema skips.** The first leg's freeze run decides the gate. Every
  other leg's skip list comes from its own run with `IS_FIRST_SHARD=false`, as
  those legs compute it, so a first-leg-only failure does not empty them.
  Validation, companion tests and proof validation use each leg's list.
- **Companion tests.** Within a leg, jurisdiction groups run one at a time and
  stop at the first failing group, as the bash step does; other legs still
  run. CI iterates groups in bash associative-array order, which `ci` cannot
  reproduce, and the gate's note says so.
- **Order.** The gate order is the workflow's: unsupported paths,
  (unmanifested pre-check,) migration authorization, retired-schema freeze,
  obsolete files, layout, waiver ratchet and audit, generated guard, target
  selection, validate, companion tests, proof validation, money atoms, oracle
  coverage, and repository tests last.
- **Supervised environment.** CI's supervisor starts each supervised subcommand
  from an empty environment. For those gates, `ci` withholds every `AXIOM_*`
  variable the supervisor does not forward, except the caller's
  `AXIOM_CORPUS_NO_FETCH` opt-out, and makes Git ignore global and system
  configuration. Unsupervised steps, such as companion tests, get the
  step's own environment.
- **Release object.** It is fetched as the caller configures CI to fetch it:
  from the Supabase registry alone when `corpus-release-registry-url` is set,
  otherwise from `corpus-release-base-url`. The workflow's verifier then checks
  it and rewrites it in canonical form. Like CI, `ci` fetches it on every run,
  even when the corpus checkout already caches it. Only under `--offline` does
  the cached object stand in for the download; it is re-verified and rewritten
  the same way. A symlink at `releases/`, the release directory or the object
  fails resolution, because following it would read and write outside the
  corpus checkout.
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
- The engine checkout must be at its pin; `ci` builds it (`cargo build`), as
  CI does, so Rust and cargo must be installed.

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
- The waiver audit always uses CI's four audit workers per partition, whatever
  `--jobs` is; other invocations cap their fan-out at `cpu_count // N`.
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

`ci` reproduces CI's inputs from committed content and trusts the machine it
runs on (the Python interpreter, uv's cache, the Rust toolchain and cargo's
registry cache), so it does not defend against a tampered local toolchain or
cache.

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
  eval roles. They are the only apply and eval trust inside `ci`: without
  `--apply-public-key` no apply manifest verifies, and without
  `--eval-public-key` no eval evidence does. An attached signing broker never
  stands in, so its retired roots cannot admit what the pinned CI refuses.
- **Capabilities.** It acquires no signing capability and never runs `--apply`.
- **Writes.** It writes temporary report inputs, the private engine snapshot and
  target directory, the temporary engine view and the repository-test venv.
  Cargo and uv may populate their normal local caches. Outside these temporary
  directories and caches it writes only the pinned public release object at the
  workflow-defined corpus release path, which the corpus checkout's cleanliness
  check exempts. It refuses a symlink on that path, so the write stays inside
  the corpus checkout. In a corpus checkout that keeps its bytes in lock files
  (`.axiom/corpus-locks/`), binding the release also places the release's
  provisions artifacts at the paths the checkout's locks pin, under
  `data/corpus/provisions/`, staged in `data/corpus/.corpus-fetch-tmp/`. There,
  as axiom-corpus does, it deletes `.corpus-fetch-` leftovers more than a day
  old. It never replaces an existing file and never follows or creates a symlink
  there. `AXIOM_CORPUS_NO_FETCH` turns this placement off, for ci's own bind and
  for every gate: the supervised strip and repository pytest's environment keep
  that one opt-out. Like CI's "Link axiom-rules-engine as sibling checkout"
  step, it links the engine, corpus and rulespec-us dependency checkouts next to
  the temporary checkout. It creates nothing under `_axiom/` inside it. CI's
  checkouts there are real directories, which the gates' `find` steps prune. The
  pinned encoder's symlink audit does walk them and rejects a symlink anywhere
  in an active RuleSpec checkout, so a symlink tracked by a dependency checkout
  fails resolution. So does committed `_axiom` content, which CI's checkouts
  would land on.
- **Network.** Unless `--offline` is set, it downloads that object, and cargo
  and uv may fetch the engine's crates and the repository-test packages. For
  a lock-file corpus checkout, provisions come from the corpus cache
  (`$AXIOM_CORPUS_CACHE`, default `~/.axiom/corpus-cache`), then the
  checkout's git objects, then, without `--offline` and only when R2 read
  credentials are configured, R2. `--offline` holds in `--jobs` workers too.
  Repository pytest has a fresh `HOME` and no `R2_*` or cache settings, so a
  repository test can place provisions only from git objects and never
  reaches R2.

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
    configuration and attributes, named after the resolved GitHub repository,
    with origin `https://github.com/OWNER/NAME`.
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
  CI sets up, as the build setup-python's bare version installs: release
  CPython with the GIL. A free-threaded `3.14t`, a debug build or another
  implementation does not match `3.14`. It also must not run
  optimized (`-O` or `PYTHONOPTIMIZE`, which
  strip the assertions in-process gates rely on). Either difference is a
  `python` mismatch under the same flag.
- Exact parity means running this tool from the pinned encoder checkout.
  `--allow-encoder-mismatch` is the honest mode for development loops on newer
  encoders, and produces the same qualified verdict and mismatch banner.
- Executing every gate in a subprocess sourced from the pinned checkout is the
  longer-term gold path and is intentionally out of scope for this command
  version.

## Known differences from CI

These are known and do not change a verdict on the lanes checked against real
CI runs (see the pull request that added this list), but can in the cases
named:

- **Report shape.** Gates the workflow's `if:` skips are reported as PASS with
  a "not selected" note. After a validate-job failure, later gates still run
  for diagnostics, and each one notes that CI skips later steps on a leg with
  an earlier failure.
- **Order.** File listings match CI's `find` as sets; `ci` walks the tree in a
  deterministic order, where CI's follows the filesystem's directory read
  order. Money-atom and obsolete-file verdicts do not depend on that order,
  but `File:` lines, and which noncanonical-citation error is reported first,
  can differ.
- **Partition scheduling.** Locally, `--jobs` runs several waiver-audit
  partitions at once, where CI's matrix legs each audit one; every partition
  uses CI's four audit workers either way.
- **Companion-test group order.** See "Companion tests" above.
- **Rust toolchain.** CI installs the current `stable` toolchain
  (`dtolnay/rust-toolchain`, whose pinned action defaults to `stable`) on each
  run; `ci` checks rustup's active-toolchain reason is `(default)` when rustup
  is available, then records the active toolchain and `rustc -vV`'s release and
  host in its engine note. Without rustup the toolchain source is unchecked,
  with a report note. An engine change that compiles or
  behaves differently across stable releases can differ.

## Troubleshooting

- **"RuleSpec module is not inside a canonical country checkout" on a
  heavily loaded machine.** The encoder's checkout routing probes Git with a
  two-second timeout, and a timed-out probe resolves the checkout as not
  canonical. Rerun with fewer `--jobs`, or when the machine is less loaded.
- **`repository-test-environment` mismatch.** `ci` could not build the
  repository-test venv (no network under `--offline` with an empty uv cache,
  or no interpreter for the caller's `python-version`). Install that Python
  (`uv python install <version>`) or rerun online.
- **"Build axiom-rules-engine failed".** The engine checkout did not build at
  its pin; the message carries cargo's output. Under `--offline`, the crates
  must already be in cargo's cache.
