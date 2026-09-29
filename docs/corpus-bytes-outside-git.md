# Corpus bytes outside git

axiom-corpus is moving its corpus bytes out of git
([axiom-corpus `docs/corpus-storage.md`](https://github.com/TheAxiomFoundation/axiom-corpus/blob/corpus/out-of-git/docs/corpus-storage.md)).
After that switch, a corpus checkout has one lock file per scope under
`.axiom/corpus-locks/` and no tracked `data/corpus/{sources,inventory,provisions,coverage}`.
Existing RuleSpec pins name pre-switch corpus commits, which still track every
file. A RuleSpec repository that re-pins to a post-switch corpus commit checks out a
corpus without provisions files.

## What axiom-encode does

`LocalCorpusRelease` (`src/axiom_encode/corpus_resolver.py`) binds a corpus
checkout to one signed release. When the checkout has a `.axiom/corpus-locks/`
directory, binding:

1. verifies the release object and its signature, as before;
2. places every provisions artifact the release lists that is missing from
   `data/corpus/provisions/`. Provisions are the only artifact class the resolver
   reads;
3. takes each file from the first source whose bytes hash to the release's sha256
   and byte count (see [Sources](#sources));
4. raises `UnmaterializedCorpusReleaseError` when any artifact cannot be placed.

A checkout without `.axiom/corpus-locks/` is never written to. A missing file
there fails when it is read, as before.

Placement (`src/axiom_encode/corpus_materialize.py`) streams bytes into a
temporary file in the destination directory, checks size and sha256, and then
hard-links the file into place. It never replaces an existing file. It never
follows or creates a symlink under the corpus root. If a placement fails, the
file and any directories it created are removed. Readers still hash every
provisions file they read. A file that is already present is checked by size
only, unless `--verify` is given.

## Sources

| Order | Label | Where the bytes come from | Available when |
| --- | --- | --- | --- |
| 1 | `cache` | `$AXIOM_CORPUS_CACHE/objects/sha256/<xx>/<sha256>`, default `~/.axiom/corpus-cache` | `axiom-corpus-ingest corpus fetch` has filled the cache |
| 2 | `git` | the file at the release's `content.git.commit`, through `git cat-file --batch` | the release was cut before the switch and the checkout has that history |
| 3 | `git-lock` | the `git_blob` the scope's lock file records for a moved file | the file was in git before the switch and the blob is local |
| 4 | `r2` | `objects/sha256/<xx>/<sha256>` in the release's R2 bucket, signed with AWS SigV4 | R2 read credentials are configured |

R2 credentials are resolved as `axiom-corpus-ingest` resolves them.
`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, and `R2_ENDPOINT` or `R2_ACCOUNT_ID`
take precedence over `~/.config/axiom-foundation/r2-credentials.json`. The
bucket comes from the signed release object, never from the environment.

## In CI

The protected verification supervisor builds each child environment from empty.
No cloud credentials, cache directory, or ambient `HOME` reach a supervised
command. Git is available only through the trusted wrapper, and the wrapper
allows `cat-file --batch`. Supervised gates therefore read corpus bytes from git
objects only.

`validate-rulespec.yml` unshallows the corpus checkout to authenticate the pin.
That makes every byte committed before the switch available as a git object.
Supervised gates need no workflow change for releases whose provisions were
ingested before the switch.

A scope ingested after the switch exists only in R2. To pin a release that
contains one, a workflow must place the files outside the supervisor, with a
read-only R2 token, before any supervised gate runs:

```yaml
- name: Place the pinned corpus release's provisions
  env:
    R2_ACCESS_KEY_ID: ${{ secrets.R2_CORPUS_READ_ACCESS_KEY_ID }}
    R2_SECRET_ACCESS_KEY: ${{ secrets.R2_CORPUS_READ_SECRET_ACCESS_KEY }}
  run: |
    if [ -d _axiom/axiom-corpus/.axiom/corpus-locks ]; then
      axiom-encode corpus-fetch --rulespec-root . --corpus-path _axiom/axiom-corpus
    fi
```

The same applies if a workflow stops fetching full corpus history. With
`--filter=blob:none`, for example, the trusted wrapper sets `GIT_NO_LAZY_FETCH=1`,
so supervised reads cannot fetch missing blobs.

## Command

```bash
axiom-encode corpus-fetch --corpus-path ../axiom-corpus --rulespec-root .
axiom-encode corpus-fetch --corpus-path ../axiom-corpus \
  --release us-rulespec-2026-08-08-obbb-alien-snap --content-sha256 0d69a0cd…
```

- `--artifact-class` (repeatable) selects `provisions` (default), `inventory`,
  `coverage` or `sources`.
- `--verify` also hashes files that are already present.
- `--no-remote` uses only the cache and git objects.
- `--json` prints a report.

The command reads `releases/<name>/<content_sha256>.json` from the corpus
checkout. It checks the release schema and binds the release to the pinned
content digest. That digest covers every artifact's sha256, so it binds the
bytes the command places. The command does not check the signature: every
command that reads corpus text verifies it when it binds the release.

Exit codes: 0 when every selected artifact is present, 1 when any artifact
could not be placed, 2 when the release cannot be loaded.

`AXIOM_CORPUS_NO_FETCH=1` turns automatic placement off, as it does in
axiom-corpus. `axiom-encode ci --offline` uses only the cache and git objects.

## Invariants

For every release and every set of sources:

1. **Fetch fidelity.** Every placed file hashes to the release's sha256 and byte count.
2. **Fail closed.** Bytes that do not verify are never placed. A failed
   placement leaves no file and no new directory.
3. **No clobber.** An existing file is never replaced.
4. **No links.** Placed files are regular files with one link. No symlink on an
   artifact's path is followed.
5. **Source order.** The first source whose bytes verify supplies the file.
6. **Idempotence.** A second pass over a fully placed release opens no source.
7. **Pre-switch checkouts.** Without `.axiom/corpus-locks/`, binding writes nothing.

`tests/test_corpus_materialize.py` checks 1–3, 5 and 6 as Hypothesis properties,
checks 4 and 7 directly, and runs the git sources against real repositories and
through the supervisor's trusted git wrapper. It also checks SigV4 against the
AWS S3 GET Object example.
