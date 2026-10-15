# Subdivision numeric recall audit

The [reproduction helper](../../scripts/reproduce_subdivision_recall_audit.py)
runs the repository's independent span audit against a fixed signed corpus
release and the pre-mask baseline. It downloads and verifies one corpus
artifact at a time. Each audit subprocess has a 480-second timeout within a
540-second shard budget. Run each shard as a separate command; a whole audit
is longer than one shard.

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

## Pinned inputs

| Input | Identity |
| --- | --- |
| Signed release | `us-rulespec-2026-08-08-obbb-alien-snap` |
| Signed content SHA-256 | `0d69a0cdbe024fc2276f3c261c00402bb0a47488ac5f482cc20bd3404980adbc` |
| Release object SHA-256 | `8d0fd2993fa667b13205947b71b91aafbb56be73351e9acebbd03255e233190a` |
| Release object bytes | `7150325` |
| Corpus revision | `e79781bddeaba4d3697c5375c7371b2a038f0f74` |
| Baseline revision | `86a1f911a338cb071e0ed660d1282714034b6dc4` |
| Baseline module SHA-256 | `c5b9079804159acc3520002ca32e45bc4b6da390c7e7bda9a698adb3082378de` |

The immutable, content-addressed
[signed release object](https://pub-a8952f8657fc49fda358146ac001366c.r2.dev/releases/us-rulespec-2026-08-08-obbb-alien-snap/0d69a0cdbe024fc2276f3c261c00402bb0a47488ac5f482cc20bd3404980adbc.json)
lists every selected artifact's path, byte count, row count and SHA-256. The
selected US manual, guidance and statute artifacts contain 79,687 rows in 196
files, totaling 249,961,962 bytes. Headings and bodies are included, with no
row or source-length exclusions.

The verification roots are the public keys published in the
[pinned repository verifier](https://github.com/TheAxiomFoundation/axiom-encode/blob/86a1f911a338cb071e0ed660d1282714034b6dc4/scripts/verify_real_defects.py#L60-L68):

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
export UV_CACHE_DIR="${UV_CACHE_DIR:?Set the shared uv cache directory}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
uv sync --locked --python 3.13 --extra dev
.venv/bin/python scripts/reproduce_subdivision_recall_audit.py prepare --directory .hub-scratch/round2/corpus
```

The checked-in summary records the successful artifact and row bounds. Print
the exact shard commands from those bounds, then run each printed line as a
separate command:

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("docs/evidence/subdivision-recall-audit-summary.json").read_text())
for batch in summary["batches"]:
    command = [
        ".venv/bin/python", "scripts/reproduce_subdivision_recall_audit.py", "shard",
        "--directory", ".hub-scratch/round2/corpus",
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
.venv/bin/python scripts/reproduce_subdivision_recall_audit.py summarize --directory .hub-scratch/round2/corpus
```

`summary.json`, `combined.rows.jsonl` and `combined.removals.jsonl` are produced
in the selected scratch directory. Each row is written once per ledger;
there are no expanding-array rewrites. Existing scratch inputs and outputs
are preserved: a repeat run requires a fresh directory. Compare production,
audit and baseline identities, counts and ledger hashes with the checked-in
summary. The result is a numeric-recall differential; formula branches and
runtime behavior are covered by focused tests.
