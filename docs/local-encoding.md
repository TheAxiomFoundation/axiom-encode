# Local encoding on a personal Codex login

This page tells a team member how to run the real `axiom-encode encode` loop on
their own machine with the Codex CLI on a ChatGPT login. It also tells why the
result cannot land in a RuleSpec repository today, and what the planned
admission path is.

| Part | State |
| --- | --- |
| [Local setup](#local-setup) | Live. Proven on 2026-09-10 ([#1622](https://github.com/TheAxiomFoundation/axiom-encode/issues/1622)) and on 2026-10-26. |
| [Why a local encode cannot land](#why-a-local-encode-cannot-land-today) | Live behavior of the guard and the signer. |
| [Contributor flow](#planned-contributor-flow-design-b) | Planned ([#1628](https://github.com/TheAxiomFoundation/axiom-encode/issues/1628)). Most steps wait for maintainer decisions. |

A local encode is a draft. It uses no API key, no Actions dispatch and no
signing key, and it writes nothing to Supabase.

## Local setup

### Prerequisites

- **uv and Python 3.13.** Run `uv python install 3.13`. The supervisor
  provisioner copies the `sys.base_prefix` of the interpreter that runs it, so
  a system or Homebrew Python does not work for route 1.
- **Rust and cargo**, to build `axiom-rules-engine` with
  `cargo build --release --locked`.
- **Go**, only for [route 1](#route-1-root-owned-verification-supervisor). The
  supervisor is a Go binary; `go.mod` names the Go version.
- **Codex CLI on the ChatGPT login.** Run `codex login`, then
  `codex login status`. It must report a ChatGPT login. Make sure that
  `~/.codex/auth.json` holds no `OPENAI_API_KEY`: the Codex backend accepts an
  API key there, and that sends paid API traffic.
- **`claude` CLI** on a claude.ai login, for the generalist reviewer. Make sure
  that `~/.claude/settings.json` sets no `apiKeyHelper` and no provider key.
- **`gh`** with read access to the repositories, to read pins and public keys.

### Pinned refs

Do not copy pin values from this page or from an old note. Read them from the
target RuleSpec repository at the commit you encode against:

```bash
cd <work>/rulespec-us
git fetch origin && git checkout --detach origin/main
uv run --no-project --python 3.13 python - <<'EOF'
import tomllib
w = tomllib.load(open(".axiom/workflow-toolchain.toml", "rb"))["workflow_toolchain"]
t = tomllib.load(open(".axiom/toolchain.toml", "rb"))["toolchain"]
for key in ("axiom_encode_ref", "axiom_encode_version", "axiom_corpus_ref",
            "axiom_rules_engine_ref", "axiom_compose_ref"):
    print(key, w[key])
for key in ("axiom_corpus_release", "axiom_corpus_release_content_sha256",
            "validation_waiver_set_sha256"):
    print(key, t[key])
EOF
```

- `.axiom/workflow-toolchain.toml` holds the checkout identities: the encoder
  pin (`axiom_encode_ref`), the corpus commit, the engine commit and the
  `axiom-compose` commit.
- `.axiom/toolchain.toml` holds the signed corpus release pin and the waiver-set
  digest.
- Check out `axiom-corpus` at `axiom_corpus_ref` and `axiom-rules-engine` at
  `axiom_rules_engine_ref`. Build the engine with
  `cargo build --release --locked`.
- `axiom-compose` at `axiom_compose_ref` is needed only for composition modules
  (`--axiom-compose-path`).
- The encoder must be the pin, or a commit on the first-parent history of
  axiom-encode `main` that descends from the pin. That is the rule the guard
  applies to producers
  ([signed manifest encoder admission](signed-manifest-encoder-admission.md)).
  On 2026-10-26, current axiom-encode `main` met it.

### Encoder checkout

1. Clone axiom-encode with full history (`git clone`, not a worktree: the
   route 1 provisioner refuses a worktree). Check out `axiom_encode_ref`, or a
   later first-parent `main` commit.
2. Create the venv from Python 3.13:
   - Route 1: `uv venv --python 3.13 .venv && uv pip install .` (not
     editable; the supervisor runtime rejects editable installs).
   - Route 2: `uv sync --locked --python 3.13`.
3. Call the tools by path (`.venv/bin/python`, `.venv/bin/axiom-encode`). The
   venv is not activated.

### Public trust roots

Encoding needs three distinct public keys: the apply root, the eval root and
the corpus-release root. They are public keys, not credentials. Read them from
the axiom-encode repository Actions variables (write access is enough):

```bash
for name in AXIOM_CORPUS_RELEASE_PUBLIC_KEY \
            AXIOM_ENCODE_APPLY_SIGNING_PUBLIC_KEY \
            AXIOM_ENCODE_EVAL_SIGNING_PUBLIC_KEY; do
  gh api "repos/TheAxiomFoundation/axiom-encode/actions/variables/$name" --jq .value
done
```

`AXIOM_CORPUS_RELEASE_RETIRED_PUBLIC_KEY` is an organization variable that a
non-admin cannot read. Omit the retired root; the current release verified
without it in #1622. Never put a private key in the environment: the
supervisor treats the private-key variable names as fatal
([trusted signing supervisor](trusted-signing-supervisor.md#protected-trust-roots)).

### Corpus release

1. Fetch the pinned release object from the public R2 URL that
   `targeted-signed-reencode.yml` uses, then validate it and write it into the
   corpus checkout:

   ```bash
   cd <work>/axiom-encode
   toolchain=<work>/rulespec-us/.axiom/toolchain.toml
   pin=$(.venv/bin/python scripts/materialize_corpus_release.py pin --toolchain "$toolchain")
   name=$(printf '%s\n' "$pin" | sed -n 1p)
   sha=$(printf '%s\n' "$pin" | sed -n 2p)
   curl --fail --silent --show-error --proto '=https' --tlsv1.2 \
     -o <tmp>/release.json \
     "https://pub-a8952f8657fc49fda358146ac001366c.r2.dev/releases/$name/$sha.json"
   jq -ce '[{"release_object": .}]' <tmp>/release.json > <tmp>/response.json
   .venv/bin/python scripts/materialize_corpus_release.py materialize \
     --toolchain "$toolchain" --response <tmp>/response.json \
     --corpus-root <work>/axiom-corpus
   ```

   The script checks the content digest against the pin. Every
   corpus-reading command then verifies the signature against the
   corpus-release root when it binds the release.
2. Place the provisions bytes before the encode, as a separate step:

   ```bash
   .venv/bin/axiom-encode corpus-fetch --corpus-path <work>/axiom-corpus \
     --rulespec-root <work>/rulespec-us
   ```

   A corpus checkout with `.axiom/corpus-locks/` keeps its bytes outside git.
   `corpus-fetch` places them into the checkout from the local cache
   (`~/.axiom/corpus-cache`), then from git objects, then from R2
   ([sources](corpus-bytes-outside-git.md#sources)). The public URL serves
   release objects only, not the `objects/sha256/...` bytes. R2 read
   credentials are needed only for bytes that the cache and git do not have.
   `axiom-corpus-ingest corpus fetch --release <release>` fills the cache.
   The command exits 1 when it cannot place an artifact, and names it.
3. **Private-only artifacts.** If `corpus-fetch` cannot place an artifact, a
   later read of that jurisdiction and document class fails closed. Get the
   bytes from a maintainer with R2 read access, or treat encodes that need
   that scope as blocked. On 2026-10-26, 2 of 271 provisions artifacts
   (12 CFR parts 1005 and 1006) were private-only. That run continued only
   with a local experiment shim that hid them from candidate lookup. The shim
   is not part of this repository and is not a supported setup.

### Route 1: root-owned verification supervisor

This is the route that #1622 proved with the code on `main`. It runs the
encoder under the same supervisor that CI uses, with no signer attached.
Supervised commands read corpus bytes from the checkout and its git objects
only, so place them first (see [Corpus release](#corpus-release)).

1. Build the supervisor in the encoder checkout:

   ```bash
   CGO_ENABLED=0 go build -trimpath -buildvcs=false -ldflags='-buildid=' \
     -o build/axiom-encode-signing-supervisor \
     ./cmd/axiom-encode-signing-supervisor
   ```

2. Bind the engine: `.venv/bin/axiom-encode engine-bind
   --axiom-rules-engine-path <work>/axiom-rules-engine --pin
   <axiom_rules_engine_ref>`.
3. Provision a root-owned runtime with
   `scripts/provision_verification_supervisor.py`, as in
   [ChatGPT subscription generation](trusted-signing-supervisor.md#chatgpt-subscription-generation),
   with the three public roots and `--install-pinned-codex-cli`. Then make the
   tree root-owned and not group- or other-writable
   (`chown -R 0:0`, `chmod -R go-w`). The supervisor refuses a tree that the
   invoking user can write. Run the `sudo` step yourself; never automate it.
4. Run the supervisor as your own user, with `CODEX_HOME` unset, and give it
   `--codex-subscription-auth` and `--codex-auth-outbox`. After `--`, run
   `/opt/axiom-verification/axiom-encode encode <citation>` with the flags in
   [Encode command](#encode-command).

### Route 2: in-process verification-only wrapper

The 2026-10-26 run used a lighter route, with no `sudo` and no Go. A small
Python wrapper calls `axiom_encode.cli.main()` inside
`toolchain.local_corpus_release_verification(...)` with the three public roots.
That is the same verification-only stand-in that `axiom-encode ci` uses
([trust boundary](ci-parity.md#trust-boundary)). It gives no signing
capability, and the wrapper refuses `--apply`.

- The wrapper is not on `main`. A version of it is in draft
  [#1971](https://github.com/TheAxiomFoundation/axiom-encode/pull/1971)
  (`scripts/local_ab/local_encode.py`).
- The context manager is documented as existing for `axiom-encode ci`. Use
  this route for local drafts and measurements only. It has no supervisor
  isolation.
- Run the wrapper with `.venv/bin/python -I`.

### Encode command

Run the encode in an empty environment:

```bash
env -i HOME="$HOME" USER="$USER" TMPDIR="$TMPDIR" LANG=en_US.UTF-8 \
  AXIOM_CORPUS_NO_FETCH=1 \
  PATH="<venv>/bin:<codex-and-claude-dir>:/usr/bin:/bin:/usr/sbin:/sbin" \
  caffeinate -i <launcher> encode "<citation>" --backend codex \
    --corpus-path <work>/axiom-corpus \
    --axiom-rules-engine-path <work>/axiom-rules-engine \
    --policy-repo-path <work>/rulespec-us \
    --output <out>/generated \
    --mode repo-augmented --require-complete-source-unit \
    --emit-final-rejected-candidate <out>/final-rejected-candidate \
    --db <out>/encodings.db --no-sync
```

`<launcher>` is the supervisor command line of route 1 or the wrapper of
route 2.

- `env -i` removes provider, R2, Supabase and GitHub keys from the
  environment. It keeps `HOME`, so files under it still apply: the Codex and
  Claude logins, and `~/.config/axiom-foundation/r2-credentials.json` if it
  exists.
- `AXIOM_CORPUS_NO_FETCH=1` turns off corpus placement during the encode
  ([command](corpus-bytes-outside-git.md#command)), so the encode does not
  fetch from R2 even if a credentials file exists. Place the bytes before the
  run.
- `--no-sync` and a local `--db` keep the run out of Supabase.
- `--require-complete-source-unit` applies the same completeness rule as the
  protected workflow.
- `--emit-final-rejected-candidate` keeps the last rejected candidate and its
  issues. The directory must not exist yet.
- Do not pass `--apply`. Without a signer it cannot succeed, and route 2
  refuses it.
- The default model policy applies: `gpt-6-luna`, then `gpt-6-sol` after two
  rejections, at most 4 attempts.

### Run time and sleep

- One citation takes about 5 to 20 minutes. The 2026-10-26 run took 8 minutes
  for 4 attempts.
- Keep the Mac awake for the whole run. Use `caffeinate -i` and AC power. A
  sleep stalls the model calls, and the run does not recover.

### Known gaps

- **Local reviewer.** On `main`, a run without a signing broker starts the
  reviewer with an empty `PATH` and `HOME=/nonexistent`. The review fails and
  the output does not say so. Look for `review_model_call` that takes under a
  second and no review line. Open fix:
  [#2004](https://github.com/TheAxiomFoundation/axiom-encode/pull/2004).
- **`--replace-rulespec-path` requires `--apply`.** A local candidate for an
  existing module is written at the new canonical path, not over the old path.
- **Pre-encode gate.** A gate that stops empty and repealed targets before a
  model call is planned in open
  [#1997](https://github.com/TheAxiomFoundation/axiom-encode/pull/1997). Until
  it merges, check the target yourself before you spend a run on it.
- **Engine pin warning.** The encoder resolves the policy repo to
  `<rulespec-us>/us` and does not see the root engine pin. It prints "without a
  declared axiom_rules_engine_ref engine pin". The warning is not fatal.
- **Git refs.** Under the supervisor, the trusted git wrapper accepts only
  `HEAD`, `main`, `origin/main` and full 40-hex SHAs.

## Why a local encode cannot land today

A RuleSpec change lands only with a signed v5 apply manifest. No local route
can produce one.

1. **The guard requires a manifest.** rulespec-us CI runs `guard-generated`
   (`cmd_guard_generated` and `guard_generated_change_issues` in
   `src/axiom_encode/cli.py`). A changed protected file without a changed
   manifest fails with "changed without a matching .axiom/encoding-manifests
   manifest".
2. **Only the protected workflow can sign.** The signature must verify against
   the apply root. The private key is a `production-signing` environment
   secret, used only by `.github/workflows/targeted-signed-reencode.yml` on
   `workflow_dispatch` from `main`. The signer
   (`cmd/axiom-encode-apply-signer/context.go`) refuses to run outside Actions
   without `--allow-local-dev`, and always refuses pull request events. The
   local-dev key is ephemeral (`serve.go`), so the broker rejects it as not the
   apply root (#1622).
3. **Producer admission.** The guard admits a manifest only when its encoder is
   the consumer pin, or a first-parent descendant of the pin on `main`
   (`_encoder_producer_admission_issue` and
   `_encoder_git_history_admission_issue` in `src/axiom_encode/cli.py`;
   [signed manifest encoder admission](signed-manifest-encoder-admission.md)).
4. **Waiver-set binding.** A manifest binds the waiver-set digest of the
   RuleSpec checkout it was validated against. A pull request may only remove
   entries from the protected base, and only for the changed modules or their
   direct importers (`_guard_manifest_waiver_set_identity` in
   `src/axiom_encode/cli.py`). If `main` changes its waiver set before the
   change lands, the digest no longer matches. Encode against current `main`,
   and land quickly.

Today new RuleSpec bytes get a signed manifest in two ways only: a model
re-encode in the protected workflow, or `promote-reviewed-candidate`
([reviewed candidate promotion](trusted-signing-supervisor.md#reviewed-candidate-promotion)).
Promotion needs an allowlist change and a manual dispatch.
`refresh-applied-manifest` also signs, but only for bytes that already have a
signed model manifest
([byte-identical manifest refresh](trusted-signing-supervisor.md#byte-identical-manifest-refresh)),
so it cannot admit an unsigned local draft.

## Planned contributor flow (design B)

> **Planned. Not live.** This section describes the direction for
> [#1628](https://github.com/TheAxiomFoundation/axiom-encode/issues/1628). No
> step below is built yet.

Decided:

- Maintainer-attested promotion of exact reviewed bytes (design B) is the team
  contributor path, and #1628 closes on it.
- A dispatch by `github-actions[bot]` when an admission record merges does not
  count as a manual dispatch.

Planned flow:

1. The contributor encodes locally as in [Local setup](#local-setup), pushes an
   org branch (no fork), and opens a rulespec-us pull request with only the
   primary module and its `.test.yaml`.
2. rulespec-us CI reports the change as an unadmitted reviewed candidate. The
   required check stays red until a manifest exists.
3. An admission record for the exact commit is opened in axiom-encode. It binds
   the country, commit, citation, path, file hashes, base branch and pull
   request. Generation details in it are declared, not authenticated.
4. A code owner approves the record.
5. When the record merges, a bot dispatches promotion. Promotion validates the
   exact bytes with complete source, with no model and no regeneration, and
   adds only the manifest to the contributor branch.
6. The guard verifies the manifest, and the pull request merges.

Pending maintainer decision (these parts can change):

| Decision | Topic |
| --- | --- |
| D2 | The authorization primitive: a code-owner-reviewed record in axiom-encode, or a code-owner approval on the rulespec-us pull request. |
| D4 | Whether each promotion run still needs a `production-signing` approval. |
| D5 | Whether promotion manifests keep the apply key or get a new signing domain. |
| D6 | Author scope: repository writers on org branches only. |
| D7 | The shared `validate-rulespec.yml` change: early candidate-status guard and a hardcoded guard. |
| D8 | The route for composition edits (#1723). |
| D9 | The enrolled-producer track (#1661 to #1664) as the long-term path. |
