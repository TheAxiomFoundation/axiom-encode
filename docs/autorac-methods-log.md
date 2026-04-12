# AutoRAC Methods Log

This file is the paper-oriented change log for AutoRAC. It records benchmark-relevant harness changes, the hypothesis behind each change, and the evidence path that should justify later claims in a paper, talk, or grant update.

## Update Protocol

- Add an entry when a change can affect benchmark outcomes, promotion decisions, reviewer behavior, oracle behavior, cost, or artifact observability.
- Record the first enabling commit plus any especially important follow-up commits when a cluster of small changes belongs to one methodological step.
- Prefer benchmark manifests, run-ledger artifact names, and repo paths over chat summaries.
- Do not use this file for pure refactors that cannot change measured behavior.

## Current Baseline Facts

As of 2026-04-10:

- Checked-in jurisdictional RAC corpus: 483 files.
- Repo counts:
  - `rac-uk`: 146
  - `rac-us`: 266
  - `rac-us-ca`: 19
  - `rac-us-co`: 18
  - `rac-us-ny`: 15
  - `rac-us-tx`: 11
  - `rac-ca`: 8
- Recent UK benchmark coverage across the main bulk waves: 165 cases.
  - [uk_wave18_soak.yaml](../benchmarks/uk_wave18_soak.yaml): 25
  - [uk_wave19_scale_seed.yaml](../benchmarks/uk_wave19_scale_seed.yaml): 32
  - [uk_wave20_bulk_seed.yaml](../benchmarks/uk_wave20_bulk_seed.yaml): 55
  - [uk_wave21_bulk_seed.yaml](../benchmarks/uk_wave21_bulk_seed.yaml): 53
- Core UK oracle readiness lane: 11 PolicyEngine-backed cases in [uk_readiness.yaml](../benchmarks/uk_readiness.yaml).
- Broader UK oracle readiness lane: 33 PolicyEngine-backed cases in [uk_policyengine_readiness.yaml](../benchmarks/uk_policyengine_readiness.yaml).
- Durable suite artifacts now expected for eval-suite runs:
  - `suite-run.json`
  - `suite-results.jsonl`
  - `results.json`
  - `summary.json`

## Backfill: 2026-03-29 to 2026-04-10

### 2026-03-29: UK oracle bridge became a first-class harness path

- Primary commit: `1d1a3cf` `Wire UK oracles into eval harness`
- Important follow-ons:
  - `727ed2a` `Harden UK PolicyEngine bridge for benchmark artifacts`
  - `fb718e0` `Tighten UK PolicyEngine benchmark bridge`
  - `e8454c4` `Add UK child benefit regulation 2(1)(b) oracle support`
  - `f304fff` `Add UK pension credit minimum guarantee oracle support`
  - `6f116a3` `Add Scottish Child Payment UK oracle support`
  - `5a77922` `Add UK benefit cap oracle support`
- Hypothesis:
  - Existing UK microsimulation outputs can serve as a fast external comparator for a subset of RAC leaves, especially fixed amounts, rates, caps, and standard elements.
- Effect:
  - PolicyEngine UK became a real oracle lane rather than an aspirational integration.
  - UK source slices gained variable hints and comparability logic for mapped programs.
- Primary evidence paths:
  - [validator_pipeline.py](../src/autorac/harness/validator_pipeline.py)
  - [uk_expanded.yaml](../benchmarks/uk_expanded.yaml)

### 2026-03-30: Readiness-gated eval suites replaced ad hoc spot-checking

- Primary commit: `a097e7c` `Add readiness-gated eval suites`
- Important follow-ons:
  - `3e8af9a` `Harden UK readiness harness and atomic benchmarks`
  - `950ef23` `Add oracle target hints for UK source benchmarks`
  - `ccf9586` `Focus UK row-slice benchmarks on atomic amounts`
- Hypothesis:
  - Bulk progress should be measured by explicit suite gates rather than hand-picked examples.
- Effect:
  - Benchmark manifests became the unit of readiness.
  - Success, compile, CI, zero-ungrounded, oracle, and cost metrics became suite-level gates.
- Primary evidence paths:
  - [uk_starter.yaml](../benchmarks/uk_starter.yaml)
  - [uk_readiness.yaml](../benchmarks/uk_readiness.yaml)
  - [evals.py](../src/autorac/harness/evals.py)

### 2026-04-01: Deterministic CI and generalist review became materially stricter

- Primary commits:
  - `9b4b719` `Add generalist reviewer gate to eval suites`
  - `0a632b3` `Reject decomposed date scalars in evals`
  - `14e8306` `Resolve defined-term imports in UK evals`
  - `6b58c08` `Reject placeholder fact variables in UK evals`
  - `c63fc6d` `Tighten UK numeric occurrence coverage checks`
  - `f9468df` `Track repeated source numbers as named scalars`
  - `2996e08` `Require repeated-scalar UK sibling coverage`
- Hypothesis:
  - Many high-cost semantic misses were not compilation failures; they needed deterministic grounding checks plus a broader statutory-fidelity reviewer.
- Effect:
  - The harness began rejecting placeholder facts, decomposed dates, under-grounded repeated numbers, and weak import behavior.
  - Generalist review became a formal promotion gate rather than a sidecar note.
- Primary evidence paths:
  - [test_evals.py](../tests/test_evals.py)
  - [eval_prompt_surface.py](../src/autorac/harness/eval_prompt_surface.py)
  - [validator_pipeline.py](../src/autorac/harness/validator_pipeline.py)

### 2026-04-04: Eval-suite runs became durable research artifacts

- Primary commit: `fdaf72e` `Persist eval-suite runs and normalize UK pence thresholds`
- Important follow-ons:
  - `d5fe8a1` `Retry transient UK eval suite case failures`
  - `de206b7` `Harden UK eval artifact salvage and rate grounding`
- Hypothesis:
  - Failed or interrupted runs still contain evidence and should not disappear into terminal history.
- Effect:
  - `eval-suite` began writing top-level state and per-case ledgers during execution.
  - This made it possible to justify later repairs from concrete artifacts rather than memory.
- Primary evidence paths:
  - [cli.py](../src/autorac/cli.py)
  - [evals.py](../src/autorac/harness/evals.py)

### 2026-04-05 to 2026-04-07: The outer loop became an autoresearch-style prompt-tuning system

- Primary commits:
  - `64cf608` `Add constrained harness-tuning pilot scaffold`
  - `f9303a0` `Reshape harness tuning pilot into autoresearch loop`
  - `702d7f6` `Add mutate-score autoresearch iteration runner`
  - `dde4bd0` `Add autoresearch final-review holdout`
  - `b438ee4` `Accept autoresearch disjunction prompt mutation`
- Hypothesis:
  - Prompt-surface changes should be evaluated on frozen manifests with a separate holdout, not accepted from intuition.
- Effect:
  - AutoRAC acquired a mutate-score-keep outer loop over one editable prompt surface.
  - A candidate mutation was accepted only when training improved without regressing the holdout.
- Primary evidence paths:
  - [program.md](../autoresearch/program.md)
  - [run_autoresearch_pilot.py](../scripts/run_autoresearch_pilot.py)
  - [run_autoresearch_iteration.py](../scripts/run_autoresearch_iteration.py)
  - [uk_autoresearch_final_review.yaml](../benchmarks/uk_autoresearch_final_review.yaml)

### 2026-04-07 to 2026-04-08: UK bulk evaluation moved from repair slices to scale waves

- Primary commits:
  - `d547287` `Add UK wave 20 bulk seed manifest`
  - `5a9de67` `Add UK wave 21 bulk benchmark seed`
  - `bead4c6` `Make UK wave 21 seed atomic`
- Hypothesis:
  - A broad benchmark wave should be decomposed into atomic slices so failures are diagnosable and repair manifests are narrow.
- Effect:
  - Wave 20 established a larger bulk benchmark.
  - Wave 21 was then broken into more atomic regulation 17 / 17A slices, which improved failure localization.
- Primary evidence paths:
  - [uk_wave20_bulk_seed.yaml](../benchmarks/uk_wave20_bulk_seed.yaml)
  - [uk_wave21_bulk_seed.yaml](../benchmarks/uk_wave21_bulk_seed.yaml)

### 2026-04-08 to 2026-04-10: Wave 21 semantics were repaired through targeted prompt rules

- Representative commits:
  - `70a0081` `Preserve qualifier scope in UK payment disjunctions`
  - `4079b3f` `Preserve residual sibling conditions in UK limbs`
  - `789e11b` `Add payment entity support to eval prompts`
  - `a554503` `Tighten UK omission and subject-to slice guidance`
  - `81e56d1` `Preserve UK complete-cycle averaging qualifiers`
  - `0c622f8` `Tighten UK exhaustive route selection guidance`
  - `f7daada` `Preserve royalty consideration scope in UK evals`
- Hypothesis:
  - The remaining misses were mostly not parser failures; they were recurring semantic compression patterns that could be attacked through narrower slice-aware rules.
- Effect:
  - Prompt-surface guidance became more explicit around subject-to clauses, residual limbs, claimant-or-partner disjunctions, payment-scoped outputs, route selection, deeming tails, and qualifier scope.
- Primary evidence paths:
  - [eval_prompt_surface.py](../src/autorac/harness/eval_prompt_surface.py)
  - wave 21 repair manifests under [benchmarks](../benchmarks)

### 2026-04-10: Local GPT evaluation default moved to Codex, and interrupted suites became resumable

- Primary commits:
  - `e0e628a` `Default GPT evals to Codex`
  - `7ac9763` `Add eval-suite resume support`
- Hypothesis:
  - Local bulk measurement should be cheaper and more robust before another round of scale runs.
- Effect:
  - GPT runners default locally to Codex unless explicitly overridden.
  - Interrupted `eval-suite` runs can resume in place from `suite-results.jsonl`.
- Primary evidence paths:
  - [cli.py](../src/autorac/cli.py)
  - [evals.py](../src/autorac/harness/evals.py)

### 2026-04-10: UK PolicyEngine lane was promoted from implicit support to named readiness manifests

- Primary commit: `71da5e8` `Promote UK PolicyEngine readiness manifests`
- Hypothesis:
  - If a manifest carries a PolicyEngine gate, the cases in that manifest should actually be oracle-backed and reproducible as a named lane.
- Effect:
  - The core UK starter and readiness manifests were upgraded to real UK PolicyEngine oracle suites.
  - A broader 33-case UK oracle regression lane was added.
- Primary evidence paths:
  - [uk_starter.yaml](../benchmarks/uk_starter.yaml)
  - [uk_readiness.yaml](../benchmarks/uk_readiness.yaml)
  - [uk_policyengine_readiness.yaml](../benchmarks/uk_policyengine_readiness.yaml)

### 2026-04-10: Eval-suite runs gained a durable local archive registry

- Hypothesis:
  - If benchmark artifacts remain only in `/tmp`, they are too easy to lose and
    too hard to cite consistently in later methods or appendix work.
- Effect:
  - `autorac eval-suite-archive` now snapshots a suite output tree into a stable
    local archive under `artifacts/eval-suites` by default.
  - Archived `results.json`, `summary.json`, and `suite-results.jsonl` payloads
    are rewritten so artifact paths point at the archived copy rather than the
    original `/tmp` run directory.
  - Each archive writes `archive-metadata.json` and appends a compact record to
    `artifacts/eval-suites/index.jsonl`.
- Primary evidence paths:
  - [cli.py](../src/autorac/cli.py)
  - [test_cli.py](../tests/test_cli.py)
  - [README.md](../README.md)

### 2026-04-11: Colorado repo-augmented validation exposed and repaired import-path mismatches

- Primary commits:
  - `33d3fef` `Fix Colorado repair harness prompts`
  - `8d4e569` `Fix Colorado import closure and prompt rails`
  - `0863433` `Terminate Codex evals after persistent output`
  - `8cc8bf4` `Preserve canonical context import paths`
  - `b9a0223` `Require unquoted RAC import targets`
- Hypothesis:
  - The remaining Colorado Works misses were not mostly parser failures; they were a mix of repo-augmented harness mismatches and prompt-level import/test conventions that the model was inferring incorrectly.
- Effect:
  - The harness now preserves canonical import paths for sibling `rac*` corpora instead of collapsing those files under `external/...`.
  - Validation import-closure handling was tightened for benchmark workspaces and mixed import-block syntax.
  - Prompt guidance became more explicit about direct factual inputs, chart-consistent tests, canonical import targets, and unquoted `imports:` entries.
  - Codex eval waiting logic now terminates once output has persisted long enough, reducing hangs after the model has already emitted a final bundle.
- Primary evidence paths:
  - [evals.py](../src/autorac/harness/evals.py)
  - [validator_pipeline.py](../src/autorac/harness/validator_pipeline.py)
  - [us_co_colorado_works_leaf_repair.yaml](../benchmarks/us_co_colorado_works_leaf_repair.yaml)
  - [us-co-colorado-works-leaf-repair1-20260411](../artifacts/eval-suites/us-co-colorado-works-leaf-repair1-20260411)
  - [us-co-colorado-works-leaf-repair2-interrupted-20260411](../artifacts/eval-suites/us-co-colorado-works-leaf-repair2-interrupted-20260411)
  - [us-co-colorado-works-leaf-repair3-interrupted-20260411](../artifacts/eval-suites/us-co-colorado-works-leaf-repair3-interrupted-20260411)
  - [us-co-colorado-works-leaf-repair4-usage-limit-20260411](../artifacts/eval-suites/us-co-colorado-works-leaf-repair4-usage-limit-20260411)

### 2026-04-11: Colorado leaf repair lane closed cleanly after CI import-closure fix

- Primary commit:
  - `342a68a` `Run CI tests with import closure`
- Hypothesis:
  - A meaningful share of the remaining Colorado repair failures were harness artifacts from running `rac.test_runner` against the benchmark output file without the copied import closure that `rac.validate all` already saw.
- Effect:
  - CI test execution now copies the root benchmark artifact, its companion `.rac.test`, and its resolved import closure into a flattened temporary workspace before invoking `rac.test_runner`.
  - The Colorado leaf repair rerun moved from mixed CI failures to a clean readiness pass on the full five-case repair manifest.
- Primary evidence paths:
  - [validator_pipeline.py](../src/autorac/harness/validator_pipeline.py)
  - [test_validator_pipeline.py](../tests/test_validator_pipeline.py)
  - [us_co_colorado_works_leaf_repair.yaml](../benchmarks/us_co_colorado_works_leaf_repair.yaml)
  - [us-co-colorado-works-leaf-repair8-interrupted-20260411](../artifacts/eval-suites/us-co-colorado-works-leaf-repair8-interrupted-20260411)
  - [us-co-colorado-works-leaf-repair9-20260411](../artifacts/eval-suites/us-co-colorado-works-leaf-repair9-20260411)

### 2026-04-11: Colorado K repair required effective-date test normalization and a reviewer fallback

- Primary commit:
  - `5440731` `Normalize eval test dates for Colorado K repair`
- Hypothesis:
  - The remaining `3.606.1(K)` failure was no longer a semantic truncation bug in the generated RAC logic; it was a harness/materialization gap where companion `.rac.test` periods could remain before the file's explicit `from YYYY-MM-DD:` date, producing an empty compiled IR during CI.
- Effect:
  - Non-annual test-period normalization now also repairs explicit ISO day strings and YAML-parsed `date` values that fall before the earliest effective date.
  - Bundle materialization now matches generated files by candidate filename, so paths like `./9-CCR-2503-6-3.606.1-K.rac` still pass the main-file content into `.rac.test` normalization.
  - The narrow Colorado `K` repair manifest now closes cleanly with compile, CI, and generalist review all passing.
- Primary evidence paths:
  - [evals.py](../src/autorac/harness/evals.py)
  - [test_evals.py](../tests/test_evals.py)
  - [us_co_colorado_works_leaf_k_repair.yaml](../benchmarks/us_co_colorado_works_leaf_k_repair.yaml)
  - [us-co-colorado-works-leaf-k-repair3-openai-interrupted-20260411](../artifacts/eval-suites/us-co-colorado-works-leaf-k-repair3-openai-interrupted-20260411)
  - [us-co-colorado-works-leaf-k-repair4-openai-claude-20260411](../artifacts/eval-suites/us-co-colorado-works-leaf-k-repair4-openai-claude-20260411)

### 2026-04-11: Federal SNAP reconstruction closed after prompt/validator alignment on annual amendment tables

- Primary commits:
  - `a7ec17f` `Teach SNAP table repairs to amend canonical outputs`
  - `3816c47` `Allow SNAP household-size table indices in CI`
  - `10491a2` `Treat SNAP additional-member math as table indexing`
  - `b49c5a1` `Avoid month-name false positives in date scalar checks`
- Hypothesis:
  - The remaining federal SNAP reconstruction misses were not substantive encoding failures in the statute slices themselves; they were harness mismatches around annual publication amendment structure, table-index numerics, and over-eager date-scalar heuristics.
- Effect:
  - The encoder prompt now explicitly instructs annual USDA parameter tables to amend canonical statute outputs instead of importing and redeclaring them locally.
  - Validator CI now treats household-size row labels and `household_size - 8` style additional-member arithmetic as structural schedule indexing rather than ungrounded free numerics.
  - Date-decomposition checks no longer misclassify ordinary benefit-period names like `initial_month` as calendar-month scalars.
  - The four-case federal SNAP reconstruction seed (`2017(a)`, `2017(c)(1)`, `2017(c)(3)`, and the FY2026 USDA allotment table) reached a clean ready state.
- Primary evidence paths:
  - [evals.py](../src/autorac/harness/evals.py)
  - [validator_pipeline.py](../src/autorac/harness/validator_pipeline.py)
  - [test_evals.py](../tests/test_evals.py)
  - [test_validator_pipeline.py](../tests/test_validator_pipeline.py)
  - [us_snap_federal_reconstruction_seed.yaml](../benchmarks/us_snap_federal_reconstruction_seed.yaml)
  - [us_snap_fy2026_cola_table_repair.yaml](../benchmarks/us_snap_fy2026_cola_table_repair.yaml)
  - [us-snap-fy2026-cola-table-repair1-20260411](../artifacts/eval-suites/us-snap-fy2026-cola-table-repair1-20260411)
  - [us-snap-fy2026-cola-table-repair2-20260411](../artifacts/eval-suites/us-snap-fy2026-cola-table-repair2-20260411)
  - [us-snap-fy2026-cola-table-repair3-20260411](../artifacts/eval-suites/us-snap-fy2026-cola-table-repair3-20260411)
  - [us-snap-fy2026-cola-table-repair4-20260411](../artifacts/eval-suites/us-snap-fy2026-cola-table-repair4-20260411)
  - [us-snap-federal-reconstruction-seed-run7-20260411](../artifacts/eval-suites/us-snap-federal-reconstruction-seed-run7-20260411)
  - [us-snap-federal-reconstruction-seed-run8-20260411](../artifacts/eval-suites/us-snap-federal-reconstruction-seed-run8-20260411)

## Open Documentation Debt

- Add before/after metric snapshots for every kept harness change rather than relying on commit messages.
- Promote the local archive registry into a shareable or publishable artifact index for externally cited runs.
- Add a formal result appendix generator for one-run and multi-run suite outputs.
- Add a public-comparison note clarifying that standard UK FRS-based microdata is licensing-constrained, so public record-level dashboards need a separate open dataset.
