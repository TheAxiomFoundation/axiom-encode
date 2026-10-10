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
   `data/corpus/provisions/`, **only where the checkout's own scope lock pins
   that path to the release's sha256 and size** (see
   [The checkout's lock decides](#the-checkouts-lock-decides)). Provisions are
   the only artifact class the resolver reads;
3. takes each file from the first source whose bytes hash to the release's sha256
   and byte count (see [Sources](#sources));
4. raises `UnmaterializedCorpusReleaseError` when an artifact the lock pins
   cannot be placed from any source, or its path is unsafe or cannot be
   inspected. A file that differs from the release bytes its lock pins (a
   local edit, or extractor output not yet locked) is left untouched and
   reported, and binding goes on; a read of that scope fails.

A checkout without `.axiom/corpus-locks/` is never written to. A missing file
there fails when it is read, as before.

Placement (`src/axiom_encode/corpus_materialize.py`) streams bytes into a
temporary file in `data/corpus/.corpus-fetch-tmp/`, checks size and sha256, and
then gives the file its name with `link(2)`. On a filesystem without hard links
it uses a no-replace rename instead (`renameatx_np(RENAME_EXCL)` on macOS,
`renameat2(RENAME_NOREPLACE)` on Linux); where neither exists, or the staging
directory is on another filesystem than the destination, it fails rather than
risk replacing a file. It never replaces an existing file. Just before it
gives a file its name, it reads the lock again and compares its bytes. Readers
still hash every provisions file they read. A file that is already present is
checked by size only, unless `--verify` is given.

Every write goes through directory descriptors, the way the resolver reads:
each directory on the path is opened relative to its parent with
`O_NOFOLLOW`, and the temporary file is created, linked and removed relative
to those descriptors, without following symlinks. A symlink put on an
artifact's path or in place of the temporary file while bytes are on the way
is therefore never followed. After publishing, the new name must be the
temporary file's own inode, and walking the path from the corpus root must
reach it; otherwise placement fails. A name that now reaches another inode is
left untouched, because another writer may have replaced it. Cleanup removes
only the placed inode when it still owns the name.

Placement assumes that other processes leave its unpredictable staging names
alone and do not rewrite the checkout's lock between the re-check and
publication. The staging inode is checked before each publication syscall,
but a process with the same filesystem permissions can still replace it in
that interval. That can leave a mismatched destination after failed placement;
readers continue to verify its release hash and size. Each lock read assumes
its ancestors and file type remain stable: lock reads do not yet protect
against an ancestor changing or a regular file becoming a FIFO between the
path check and open.

The staging directory is the one `axiom-corpus-ingest corpus fetch` uses:
git-ignored, outside every scope, and each temporary file's name carries the
`.corpus-fetch-` marker. If a placement fails, its temporary file and the
scope directories it created are removed. The staging directory stays, as
axiom-corpus leaves it: removing it could pull it from under another process's
placement. If another process removes an empty scope directory while bytes
are on the way, encode makes it again once. A process killed mid-placement (SIGKILL,
power loss) cannot clean up. It leaves one marked temporary file in
`data/corpus/.corpus-fetch-tmp/` and possibly empty directories on the
artifact's path, never a file inside a scope, so `corpus lock`, signing and
directory walkers never see it. axiom-corpus and axiom-encode both delete marked
temporary files there once they are more than a day old.

Only provisions, inventory and coverage artifacts, one file per scope, are ever
placed. axiom-corpus fetches a scope's `sources/` directory all or nothing,
because code lists that directory to find the scope's sources. Use
`axiom-corpus-ingest corpus fetch` for sources.

## The checkout's lock decides

axiom-corpus treats the bytes at a protected path as current when they match
that path's entry in the checkout's lock: `corpus status` and `corpus fetch`
leave them, its resolver reads them, and `sign-ingest-manifest --lock` signs
and locks them. So a protected path may hold only the bytes its lock pins, or
fresh extractor output (axiom-corpus `docs/corpus-storage.md`, "Consumers
outside this repository"). axiom-encode places an artifact only when the
worktree's `.axiom/corpus-locks/<jurisdiction>/<document_class>/<version>.json`
lists that path with the release's sha256 and size.

Otherwise it skips the artifact and leaves the path as it is. That happens when
the lock is missing, invalid, or does not list the path, and when it pins other
bytes, for example because the scope was re-ingested after the release. A lock
pins nothing unless it passes the checks of axiom-corpus's `parse_lock`: schema
v1, its own scope, canonical repository paths inside that scope, valid sha256,
size and `git_blob` values, and the canonical encoding. Binding still succeeds. It prints the skipped
artifacts, and a later read of that jurisdiction and document class fails with
the reason. The resolver reads every release scope of the jurisdiction and
document class it is asked about. Before the switch it was the same: a checkout
whose tracked file differed from the release failed when it was read.

Two other cases also fail only when read. A present file that the lock pins to
the release's bytes but that holds other bytes is left untouched (`modified`).
A file of the release's size whose lock pins other bytes is left as it is,
because a re-ingest can keep a file's size. If a read then finds other bytes
there, its error carries that reason; `corpus-fetch --json` lists such files
under `notes`. Binding does not read the locks of present files, so a checkout
that already holds the release costs nothing extra to bind.

To read a skipped scope at the pinned release, bind a corpus worktree at the
release's `git.commit`, which the message names. Its locks pin the release's
bytes. The release object is not tracked in git, so copy it in:

```bash
git -C ../axiom-corpus worktree add --detach ../axiom-corpus-at-release <git.commit>
mkdir -p ../axiom-corpus-at-release/releases/<release>
cp ../axiom-corpus/releases/<release>/<content_sha256>.json \
  ../axiom-corpus-at-release/releases/<release>/
```

Then pass `--corpus-path ../axiom-corpus-at-release`.

## Sources

| Order | Label | Where the bytes come from | Available when |
| --- | --- | --- | --- |
| 1 | `cache` | `$AXIOM_CORPUS_CACHE/objects/sha256/<xx>/<sha256>`, default `~/.axiom/corpus-cache` | `axiom-corpus-ingest corpus fetch` has filled the cache |
| 2 | `git` | the file at the release's `content.git.commit`, through `git cat-file --batch` | the release was cut before the switch and the checkout has that history |
| 3 | `git-lock` | the `git_blob` the scope's lock file records for a moved file | the file was in git before the switch and the blob is local |
| 4 | `r2` | `objects/sha256/<xx>/<sha256>` in the release's R2 bucket, signed with AWS SigV4 | R2 read credentials are configured |

R2 credentials are resolved in `axiom-corpus-ingest`'s order:
`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, and `R2_ENDPOINT` or `R2_ACCOUNT_ID`
take precedence over `~/.config/axiom-foundation/r2-credentials.json`. Unlike
axiom-corpus, encode does not fall back to `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY`, so credentials meant for another service are never
sent to R2. The bucket comes from the signed release object, never from the
environment.

Git runs with `GIT_NO_LAZY_FETCH=1` and without any inherited `GIT_*`
variable, as under the supervisor's trusted wrapper: an inherited `GIT_DIR`
cannot point it at another repository, and a partial clone's missing blobs are
reported absent rather than fetched. The `git` and `git-lock` sources read
local objects only.

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

- `--artifact-class` (repeatable) selects `provisions` (default), `inventory`
  or `coverage`. Sources are fetched only by `axiom-corpus-ingest corpus fetch`.
- `--verify` also hashes files that are already present.
- `--no-remote` uses only the cache and git objects.
- `--json` prints a report.

The command reads `releases/<name>/<content_sha256>.json` from the corpus
checkout. It checks the release schema and binds the release to the pinned
content digest. That digest covers every artifact's sha256, so it binds the
bytes the command places. The command does not check the signature: every
command that reads corpus text verifies it when it binds the release.

The command skips artifacts the checkout's lock does not pin, as binding does,
and lists them on stderr (`skipped` in `--json`). A skip is not a failure: a
workflow that pins a release with a since-re-ingested scope fails only if a gate
reads that scope, as before the switch.

Exit codes: 0 when every selected artifact is present, placed or skipped; 1 when
an artifact the lock pins could not be placed, or its file differs from the
release bytes the lock pins (`modified`); 2 when the release cannot be loaded.

`AXIOM_CORPUS_NO_FETCH` set to `1`, `true` or `yes`, in any case, turns
automatic placement off; these are the values axiom-corpus accepts.
`axiom-encode ci --offline` uses only the cache and local git objects.

## Invariants

For every release, every checkout lock state and every set of sources, under
the staging and lock assumptions above:

1. **Fetch fidelity.** Every placed file hashes to the release's sha256 and byte count.
2. **Lock bytes only.** A file is placed only at a path the checkout's own lock
   pins to the release's sha256 and size. Any other artifact is skipped, and
   its path is left as it was.
3. **Fail closed.** Bytes that do not verify are never placed. A placement
   that fails without being killed removes its own file and the empty scope
   directories it created; another writer's replacement is left untouched.
   The staging directory stays, empty.
4. **Scopes stay clean.** No temporary file is ever created inside a scope.
   Temporary files live in `data/corpus/.corpus-fetch-tmp/` and carry the
   `.corpus-fetch-` marker. A killed placement leaves at most one there.
5. **No clobber.** An existing file is never replaced, by `link(2)` or by the
   no-replace rename. A file another process places first is accepted only if
   it hashes to the release's bytes.
6. **No links.** Placed files are regular files with one link. No symlink on an
   artifact's path, lock path or staging path is followed, including one swapped
   in while bytes are on the way.
7. **Source order.** The first source whose bytes verify supplies the file.
8. **Idempotence.** A second pass over a fully placed release opens no source.
9. **Pre-switch checkouts.** Without `.axiom/corpus-locks/`, binding writes nothing.

`tests/test_corpus_materialize.py` checks 1–3, 5, 7 and 8 as Hypothesis
properties. One of them runs every combination of lock state (none, path not
listed, the release's bytes, other bytes of another or the same size), existing
file and source result, and checks that every file placed holds its lock's
bytes. The file checks 4 by watching the filesystem mid-stream and by SIGKILLing
a placing process. It checks 6 and 9 directly and compares its lock reader with
bytes written by axiom-corpus's own `serialize_lock`. It also covers a lock
rewritten mid-placement, a failed attempt racing a successful one, and a scope
directory removed mid-placement, a directory or the temporary file swapped for
a symlink mid-placement, and eight processes placing the same release at once.
It runs the git sources against real repositories and through the supervisor's
trusted git wrapper, including a Hypothesis property that interleaves reads,
abandoned streams and refused objects, an inherited `GIT_DIR`, and a partial
clone that must not fetch. It checks SigV4 against the AWS S3 GET Object
example.
