# Subdivision numeric recall audit

The [reproduction helper](../../scripts/reproduce_subdivision_recall_audit.py)
runs the repository's independent span audit against a fixed signed corpus
release and the pinned audit baseline `4f3547461300bfa70569f57c83728b2ad25d9d0b`.
The baseline's production module is byte-identical to the current merge base with
`origin/main`, `c7fe642eee36ba4e1a766d3eae8a9f3bb31b9ed3`; both have SHA-256
`875df7eb77efdaef6add1ca6756989f2881b5825dcf65dd6e66c0f30c43f3b62`.
The helper downloads and verifies one corpus artifact at a time. Each audit
subprocess has a 480-second timeout
within a 540-second shard budget. Run each shard as a separate command; a whole
audit is longer than one shard.

The checked-in [summary](subdivision-recall-audit-summary.json) records the
production and audit script byte hashes, signed input identities, coverage,
counts and full ledger hashes. The small
[removal ledger](subdivision-recall-audit-removals.jsonl) contains every changed
row's independently classified entry spans, typed inventory differential and
exact baseline counterfactual projections, plus restored glued-unit quantity
proofs. Every row executes the original baseline, the baseline with the bounded
quantity guard, and production cleanup. The complete row ledger is
regenerated in scratch space and authenticated by the summary; corpus bytes
remain outside Git.

Full differential counts compare the original baseline with production.
`entry_removed_inventory_occurrences_by_kind` classifies the separate
guard-only-to-production differential. These staged counts can differ from
the full differential when a restored quantity and removed locator have the
same typed inventory key.

## Heuristic limit

A singular `subdivision N` followed by `.`, `;`, `:` or end of text counts as
a locator. Land-use readings of that shape are outside the heuristic: for
example, `A minor subdivision 4.`, `The maximum lots in a subdivision 5; larger
plats require review.` and `Land subdivision 3: minimum acreage.` mask the
number and remove its numeric demand. No such land-use reading appears among
the masked locators in the signed corpus below.

## Final corpus differential

The final tested production and audit bytes, identified in the summary,
passed all 79,687 signed rows across 196 files in 27 completed shards. Every
row ran all three cleanup paths. Counts compare production with the pinned
audit baseline:

| Measurement | Count |
| --- | ---: |
| Rows changed by entry masks and final numeric recall | 22 |
| Typed inventory occurrences removed | 51 |
| Removed typed legal locators | 33 |
| Removed typed bounded-geography identifiers | 18 |
| Typed inventory occurrences added | 0 |
| Entry digit spans masked | 52 (34 legal locators, 18 geography) |
| Glued-unit quantity restorations in this corpus | 0 |

The 52 independently classified entry spans yield 51 typed removals because
one locator is absent from the baseline typed inventory. The generated case
and typography controls exercise the quantity guard even though this signed
corpus has no matching quantity restoration.

The round-4 rerun used the unchanged committed audit and reproduction scripts.
All 27 shards passed, and the totals match both the prior audit and the
independent round-4 review. The complete row and changed-row ledger hashes
are unchanged; the changed-row ledger is byte-identical to the previously
checked-in ledger. The production-module hash below identifies the final
tested working-tree bytes.

| Audited content | SHA-256 |
| --- | --- |
| Production module | `75275e2cc92251b6d22385036e904e8cc5a4af895e15e01cbdbf1a01614e6925` |
| Audit script | `c3d30f2cdb427fd3c6125257da720775f3d8cfec083f653bcce36fef80c745ca` |
| Reproduction helper | `5178432ed1f89aeda789702f98e6eb53d7a4fd23baebe5a34e688f9651a878b5` |
| Summary | `46a293a715479d4ec596b1c8b2678c9e13488a5077504a51f5c597f203e588a7` |
| Complete row ledger (regenerated) | `f19e64c7f9f8034423f5e92d6e622c790ebd5f118d18fe1ef33b63177977de05` |
| Changed-row ledger | `2838be19c23c224a37866fb81db40dbaa8889498e029d55c231430f106d61d1c` |

## Pinned inputs

| Input | Identity |
| --- | --- |
| Signed release | `us-rulespec-2026-08-08-obbb-alien-snap` |
| Signed content SHA-256 | `0d69a0cdbe024fc2276f3c261c00402bb0a47488ac5f482cc20bd3404980adbc` |
| Release object SHA-256 | `8d0fd2993fa667b13205947b71b91aafbb56be73351e9acebbd03255e233190a` |
| Release object bytes | `7150325` |
| Corpus revision | `e79781bddeaba4d3697c5375c7371b2a038f0f74` |
| Pinned audit baseline revision | `4f3547461300bfa70569f57c83728b2ad25d9d0b` |
| Current merge base with `origin/main` (production-byte-identical) | `c7fe642eee36ba4e1a766d3eae8a9f3bb31b9ed3` |
| Baseline module SHA-256 | `875df7eb77efdaef6add1ca6756989f2881b5825dcf65dd6e66c0f30c43f3b62` |

The immutable, content-addressed
[signed release object](https://pub-a8952f8657fc49fda358146ac001366c.r2.dev/releases/us-rulespec-2026-08-08-obbb-alien-snap/0d69a0cdbe024fc2276f3c261c00402bb0a47488ac5f482cc20bd3404980adbc.json)
lists every selected artifact's path, byte count, row count and SHA-256. The
selected US manual, guidance and statute artifacts contain 79,687 rows in 196
files, totaling 249,961,962 bytes. Headings and bodies are included, with no
row or source-length exclusions.

The verification roots are the public keys published in the
[pinned repository verifier](https://github.com/TheAxiomFoundation/axiom-encode/blob/4f3547461300bfa70569f57c83728b2ad25d9d0b/scripts/verify_real_defects.py#L60-L68):

| Base64 raw Ed25519 key | SHA-256 of raw 32-byte key |
| --- | --- |
| `KwIYEbbs/905yxn/9Yi6jYTF8oyZcq1FlxiG4e0y0tg=` | `99e3ab380c6de1bd59f2f9f432a8ca43a798e35b6479e6236de6618b931c1628` |
| `9EJzeLnZsdBoDEDkU8xbi7HjYVVvd4HDX/BxPHPvXTk=` | `06f641c2fcf6ab0ebb1b6f2b18df36c3c8b8b5ea300b6168f1b3f367810a8fb0` |

The helper verifies the release signature against these roots, both release
digests, corpus revision and exact baseline bytes before running a shard.
The audit verifies each artifact against its signed identity and each file's
row count. The inherited cleanup suffix must match the baseline. The bounded
glued-quantity guard has a separate proof, and its quantity restorations are
reported alongside entry-mask removals. Summarization rejects missing shards,
changed script or module bytes, mismatched environments, noncontiguous
coverage or artifact manifests that differ from the verified signed selection.
A completion sidecar authenticates each summary and ledger only after its
audit subprocess exits successfully; timed-out partial outputs are excluded.

## Exact invocation

Use a GIL-enabled CPython 3.13 and the locked dependencies. From the checkout
root, set `UV_CACHE_DIR` to the shared fleet cache, set numeric thread limits,
then prepare a new scratch directory:

```sh
umask 022
export UV_CACHE_DIR=/private/tmp/hub-shared-uv-cache
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
uv sync --locked --python 3.13 --extra dev
uv run --locked python scripts/reproduce_subdivision_recall_audit.py prepare --directory .hub-scratch/round4/corpus
```

The checked-in summary records the successful artifact and row bounds. Print
the exact shard commands from those bounds, then run each printed line as a
separate command:

```sh
uv run --locked python - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("docs/evidence/subdivision-recall-audit-summary.json").read_text())
for batch in summary["batches"]:
    command = [
        "uv", "run", "--locked", "python",
        "scripts/reproduce_subdivision_recall_audit.py", "shard",
        "--directory", ".hub-scratch/round4/corpus",
        "--artifact-start", str(batch["artifact_start"]),
        "--artifact-limit", str(batch["artifact_end"] - batch["artifact_start"]),
    ]
    ranges = batch["row_ranges"]
    if len(ranges) == 1 and (ranges[0]["row_start"] != 0 or ranges[0]["row_end"] != ranges[0]["artifact_row_count"]):
        selected = ranges[0]
        command += ["--row-start", str(selected["row_start"]), "--row-limit", str(selected["selected_row_count"])]
    print(" ".join(command))
PY
```

Each command supplies the pinned release, both public roots and exact baseline.
Corpus artifacts are read at the signed revision from
`TheAxiomFoundation/axiom-corpus`. Private GitHub access uses `GH_TOKEN`,
`GITHUB_TOKEN`, or existing `gh auth` credentials when needed. Each row slice
still verifies the complete artifact's byte identity and physical row count.
Summarization requires contiguous, complete signed coverage across all slices.

After every shard exits successfully, run:

```sh
uv run --locked python scripts/reproduce_subdivision_recall_audit.py summarize --directory .hub-scratch/round4/corpus
```

`summary.json`, `combined.rows.jsonl` and `combined.removals.jsonl` are produced
in the selected scratch directory. Each row is written once per ledger;
there are no expanding-array rewrites. Existing scratch inputs and outputs
are preserved: a repeat run requires a fresh directory. Compare production,
audit and baseline identities, counts and ledger hashes with the checked-in
summary. The result is a numeric-recall differential; formula branches and
runtime behavior are covered by focused tests.
