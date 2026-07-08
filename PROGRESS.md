# Maximum-traceability program - part 1: run-log spec, audit, autovisualization

Branch: `maxtrace-runlog-v1` (axiom-encode). Companion renderer PR in axiom-oracles.

## Phase A - audit: what we were NOT capturing

Every encoding-log surface that exists today, and whether it's live or stale:

| Surface | Where | Writer | State |
|---|---|---|---|
| `encodings.db` (`encoding_runs`, `sessions`, `session_events`) | local SQLite, axiom-encode | `EncodingDB.log_run` / `log_event` | live locally; **not published**; 198 MB, active today |
| Supabase `encodings.encoding_runs` (+ `telemetry.sdk_sessions`/`sdk_session_events`/`agent_transcripts`) | Supabase `swocpijqqahhuwtuahwc` | `supabase_sync.sync_run_to_supabase` (env-gated, inline, per-run) + manual `sync-applied-runs` | **STALE** - no cron, no staleness guard; skipped silently in cloud/worktree runs |
| `/ops` + `/axiom/ops` "Encoding Run Health" + rule-page "Agent logs" | axiom-foundation.org | reads Supabase live | code live; **data stale**; `refreshed_at` set to render time masks lag |
| applied-rulespec/v1 manifests `.axiom/encoding-manifests/**/*.json` | rulespec repos | encoder on `--apply` | durable per-file provenance; reconcile source |
| manifest-census + Shields badge | `axiom-encode manifest-census` | on demand | point-in-time coverage %, not run history |
| OTLP spans (`observability.py`) | external OTLP endpoint | eval/agent-run, if `OTEL_*` set | optional, ephemeral, not a durable log |

Corpus (the discipline to mirror) publishes R2 + Supabase with a **daily `check_publication_staleness.py` cron** that fails red past a 24 h lag. Encode has **no** such guard - the root cause of the stale surface Max remembered.

### Captured-vs-needed gap table (per-run traceability wishlist)

Coverage % = share of the 3,213 historical runs the backfill can source the field for.

| Wishlist field | Captured today | Where | Backfill cov. |
|---|---|---|---|
| run_id | Yes | db, manifest, supabase | 100% |
| module legal_id | **No** (citation only; not the compiled item `id`) | - | 0% (citation 100%) |
| corpus citation_paths | Partial (pointer only) | manifest `context_manifest_file` | 13% |
| pinned corpus ref | **No** | - | 0% |
| provision-text shas | Partial (manifest sha, not per-provision) | `context_manifest_sha256` | 13% |
| backend/model/params | Partial (params never recorded) | manifest/db | backend/model ~100% |
| prompt sha | Yes (generation) | manifest `generation_prompt_sha256` | 13% |
| raw generation artifact refs | Yes (local path) | manifest `trace_file`/`trace_sha256` | 13% |
| **each gate verdict + machine reason** | **No durable** (transient `EvalArtifactMetrics`/`PipelineResult`; only flattened into `review_results_json`) | in-memory | ci 95% / review 98% / compile 0% / oracle 2% |
| **judge verdicts** | **No** (no judge worker exists yet) | - | 0% |
| fixture provenance (who computed, engine-verified) | **No** | - | 0% |
| apply outcome + manifest sha chain | Partial (not joined) | `outcome_json` + manifest `supersedes` | apply 95% / chain 13% |
| **PR number** | **No** | - | 0% |
| **CI run ids** | **No** | - | 0% |
| **merge sha** | **No** | - | 0% |
| **oracle-suite results at merge** | **No** (axiom-oracles has conformance, not linked to run) | - | 0% |
| **downstream conformance-row impact** | **No** | - | 0% |

Headline: we capture the **left half** of the lifecycle (generate -> gates -> apply) richly but transiently/locally/coarsely, and **nothing of the right half** (judge -> PR -> CI -> merge -> oracle-at-merge -> conformance impact). No event-sourced per-stage log; no per-gate machine-readable verdict is durably published; publication is a coarse per-run row with no staleness guard.

## Phase B - spec + emission (this PR)

- `src/axiom_encode/run_log.py` - `axiom_encode.run_log.v1`: pydantic `RunLogEvent` (one JSONL event per stage transition, validated against the declared stages), machine-readable `PIPELINE_SPEC` (13-stage DAG with `depends_on` edges + categories), forward-compatible `judge` stage with structured `Finding`s, `RunLogWriter` (non-fatal), and `fold_run` -> per-run summary for the funnel/Pareto.
- `src/axiom_encode/run_log_export.py` - live emission from `EvalResult.metrics` (real compile/ci/grounding/oracle/review verdicts), backfill from `encodings.db` + manifests (absent -> null, never fabricated; downstream/judge stages simply absent), publish (folds to committed dashboard JSON), and `check_staleness` (corpus guard analog).
- cli.py: `run-log-export`, `run-log-publish`, `run-log-staleness`, `run-log-spec`; live emission wired after `_record_encode_outcome`.
- Publication path: **committed JSON in the oracles dashboard `public/data/`** (consistent with how the static conformance dashboard is served) - `run_log_pipeline.json`, `run_log_runs.json` (+ precomputed funnel/Pareto aggregates), `run_log_events.jsonl` (recent-run detail), `run_log_publication.json` (freshness).
- Version trio bumped to 0.2.1191 (pyproject + `__init__` + changelog fragment).
- Backfill coverage: **3,212 / 3,213 runs exported (99.97%)**; 419 (13%) linked to a signed manifest.

## Phase C - autovisualization (companion axiom-oracles PR)

`EncodingRunLog` component on the conformance dashboard: per-run step DAG (nodes colored by verdict, timings), aggregate funnel, failure Pareto by gate/reason. Data-driven from the published run-log only.

## Coordination

No judge worker exists (no PR/issue/branch/repo). Opened an axiom-encode issue declaring the `judge` stage event shape as the single agreed anchor for the parallel judge-builder to emit into.

## Status
- [x] Phase A audit + gap table
- [x] Phase B schema, emission, backfill, publish, staleness, tests (21), version trio
- [ ] Phase C renderer PR (axiom-oracles)
- [ ] judge-shape coordination issue
- [ ] PRs opened, CI green, rebased
