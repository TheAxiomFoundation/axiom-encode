# Paper Evidence Register

This file is the claim register for future writing. It separates claims that are already supported by checked-in code and reproducible local counts from claims that still need stronger evidence or tighter wording.

## Safe Claims Today

### Harness structure

- AutoRAC uses manifest-driven eval suites with explicit readiness gates.
  - Evidence:
    - [evals.py](../src/autorac/harness/evals.py)
    - [uk_starter.yaml](../benchmarks/uk_starter.yaml)
    - [uk_readiness.yaml](../benchmarks/uk_readiness.yaml)
- Eval suites gate on compile, CI, zero-ungrounded rate, and generalist review pass rate, with optional oracle pass rates.
  - Evidence:
    - [evals.py](../src/autorac/harness/evals.py)
    - [test_evals.py](../tests/test_evals.py)
- Eval-suite runs persist durable artifacts while live.
  - Evidence:
    - [cli.py](../src/autorac/cli.py)
    - [evals.py](../src/autorac/harness/evals.py)

### Review and validation

- AutoRAC includes a generalist statutory-fidelity reviewer as a formal suite gate.
  - Evidence:
    - [validator_pipeline.py](../src/autorac/harness/validator_pipeline.py)
    - [evals.py](../src/autorac/harness/evals.py)
- The deterministic validation layer checks more than compilation, including numeric grounding and malformed temporal/scalar patterns.
  - Evidence:
    - [validator_pipeline.py](../src/autorac/harness/validator_pipeline.py)
    - [test_evals.py](../tests/test_evals.py)

### Outer loop

- AutoRAC now has a constrained autoresearch-style outer loop over one editable prompt surface, a frozen training set, and a separate final-review holdout.
  - Evidence:
    - [program.md](../autoresearch/program.md)
    - [run_autoresearch_pilot.py](../scripts/run_autoresearch_pilot.py)
    - [run_autoresearch_iteration.py](../scripts/run_autoresearch_iteration.py)

### Current scale

- Jurisdictional checked-in RAC corpus size is 483 files as of 2026-04-10.
  - Local counts:
    - `rac-uk`: 146
    - `rac-us`: 266
    - `rac-us-ca`: 19
    - `rac-us-co`: 18
    - `rac-us-ny`: 15
    - `rac-us-tx`: 11
    - `rac-ca`: 8
- Recent UK bulk benchmark coverage across waves 18 to 21 is 165 cases.
  - Evidence:
    - [uk_wave18_soak.yaml](../benchmarks/uk_wave18_soak.yaml)
    - [uk_wave19_scale_seed.yaml](../benchmarks/uk_wave19_scale_seed.yaml)
    - [uk_wave20_bulk_seed.yaml](../benchmarks/uk_wave20_bulk_seed.yaml)
    - [uk_wave21_bulk_seed.yaml](../benchmarks/uk_wave21_bulk_seed.yaml)
- There is now a named 33-case UK PolicyEngine readiness lane.
  - Evidence:
    - [uk_policyengine_readiness.yaml](../benchmarks/uk_policyengine_readiness.yaml)
- The Colorado Works leaf repair manifest reached a clean ready state on 2026-04-11.
  - Correct scope:
    - This is evidence for the five-case Colorado repair lane, not for Colorado program completeness more broadly.
  - Evidence:
    - [us_co_colorado_works_leaf_repair.yaml](../benchmarks/us_co_colorado_works_leaf_repair.yaml)
    - [us-co-colorado-works-leaf-repair9-20260411](../artifacts/eval-suites/us-co-colorado-works-leaf-repair9-20260411)
- The Colorado `3.606.1(K)` repair case reached a clean ready state after harness normalization of pre-effective companion test dates.
  - Correct scope:
    - This supports the specific `K` leaf and the harness claim about effective-date test normalization; it does not imply every remaining Colorado seed case is closed.
  - Evidence:
    - [us_co_colorado_works_leaf_k_repair.yaml](../benchmarks/us_co_colorado_works_leaf_k_repair.yaml)
    - [evals.py](../src/autorac/harness/evals.py)
    - [us-co-colorado-works-leaf-k-repair4-openai-claude-20260411](../artifacts/eval-suites/us-co-colorado-works-leaf-k-repair4-openai-claude-20260411)
- A four-case federal SNAP reconstruction seed now reaches a clean ready state under AutoRAC.
  - Correct scope:
    - This supports the specific seed covering `7 USC 2017(a)`, `7 USC 2017(c)(1)`, `7 USC 2017(c)(3)`, and the FY2026 USDA allotment table. It does not yet establish full federal SNAP coverage outside those slices.
  - Evidence:
    - [us_snap_federal_reconstruction_seed.yaml](../benchmarks/us_snap_federal_reconstruction_seed.yaml)
    - [us_snap_fy2026_cola_table_repair.yaml](../benchmarks/us_snap_fy2026_cola_table_repair.yaml)
    - [evals.py](../src/autorac/harness/evals.py)
    - [validator_pipeline.py](../src/autorac/harness/validator_pipeline.py)
    - [us-snap-federal-reconstruction-seed-run8-20260411](../artifacts/eval-suites/us-snap-federal-reconstruction-seed-run8-20260411)
    - [us-snap-fy2026-cola-table-repair4-20260411](../artifacts/eval-suites/us-snap-fy2026-cola-table-repair4-20260411)
- A current-effective federal SNAP asset benchmark now reaches a clean ready state under AutoRAC.
  - Correct scope:
    - This supports the specific FY2026 USDA current-effective asset-limit slice and the harness claim that AutoRAC can validate annual publication-layer SNAP amendments against PolicyEngine once replay periods and oracle scoping are aligned. It does not by itself establish broader federal SNAP eligibility closure beyond this asset lane.
  - Evidence:
    - [us_snap_asset_test_current_effective_refresh.yaml](../benchmarks/us_snap_asset_test_current_effective_refresh.yaml)
    - [evals.py](../src/autorac/harness/evals.py)
    - [validator_pipeline.py](../src/autorac/harness/validator_pipeline.py)
    - [asset-limits-current-effective.txt](../../rac-us/sources/slices/usda/snap/fy-2026-cola/asset-limits-current-effective.txt)
    - [us-snap-asset-test-current-effective-refresh6-ready-20260412](../artifacts/eval-suites/us-snap-asset-test-current-effective-refresh6-ready-20260412)
- A current-effective federal SNAP household-eligibility benchmark now reaches a clean ready state under AutoRAC.
  - Correct scope:
    - This supports the specific `is_snap_eligible` current-effective slice and the harness claim that AutoRAC can now align direct statute component imports, eligibility-composite prompt guidance, and PolicyEngine replay for this lane. It does not establish full federal SNAP closure or prove that person-level participation aggregation is solved generally.
  - Evidence:
    - [us_snap_eligibility_refresh.yaml](../benchmarks/us_snap_eligibility_refresh.yaml)
    - [evals.py](../src/autorac/harness/evals.py)
    - [validator_pipeline.py](../src/autorac/harness/validator_pipeline.py)
    - [is_snap_eligible.txt](../../rac-us/sources/slices/7-USC/snap/current-effective/is_snap_eligible.txt)
    - [us-snap-eligibility-refresh6-ready-20260412](../artifacts/eval-suites/us-snap-eligibility-refresh6-ready-20260412)
- Checked-in current-effective federal SNAP deduction benchmarks now replay cleanly without depending on copied `2014(e)` context files.
  - Correct scope:
    - This supports the specific `snap_earned_income_deduction` and `snap_net_income_pre_shelter` current-effective slices and the narrower harness claim that AutoRAC can benchmark these atomic deduction/intermediate outputs directly once duplicate benchmark context is removed and the oracle bridge accepts harmless intermediate-input synonyms. It does not establish that the full `2014(e)` deduction chain is benchmark-closed as a whole.
  - Evidence:
    - [us_snap_earned_income_deduction_refresh.yaml](../benchmarks/us_snap_earned_income_deduction_refresh.yaml)
    - [us_snap_net_income_pre_shelter_refresh.yaml](../benchmarks/us_snap_net_income_pre_shelter_refresh.yaml)
    - [validator_pipeline.py](../src/autorac/harness/validator_pipeline.py)
    - [snap_net_income_pre_shelter.txt](../../rac-us/sources/slices/7-USC/snap/current-effective/snap_net_income_pre_shelter.txt)
    - [us-snap-earned-income-deduction-refresh6-ready-20260412](../artifacts/eval-suites/us-snap-earned-income-deduction-refresh6-ready-20260412)
    - [us-snap-net-income-pre-shelter-refresh11-ready-20260412](../artifacts/eval-suites/us-snap-net-income-pre-shelter-refresh11-ready-20260412)

## Claims To Avoid Or Qualify

- Do not claim that standard UK FRS-based microdata is public.
  - Correct framing:
    - The UK data-construction package and pipeline can be public while the standard UK microdata remains licensing-constrained.
- Do not claim that a public UK transfer or CPS-style dataset is equivalent to enhanced FRS.
  - Correct framing:
    - It can support public record-level comparison and reproducibility, but it is a weaker substrate for claims about UK household fidelity.
- Do not claim that Codex-default improved benchmark accuracy until the current Codex-backed rerun and repair cycle fully closes.
  - Correct framing:
    - Codex is now the local default backend for GPT evals because it reduces cost and simplifies local runs.
- Do not claim that Colorado Works as a whole is benchmark-closed.
  - Correct framing:
    - The five-case leaf repair manifest is now clean, but that does not by itself establish broader Colorado program coverage or stability outside that lane.
- Do not claim that federal SNAP is fully encoded or benchmark-closed.
  - Correct framing:
    - The current AutoRAC reconstruction seed for four federal SNAP slices is now green, which is strong evidence for the repaired harness and those specific slices, but broader federal SNAP coverage still remains to be encoded and benchmarked.
- Do not claim that North Carolina SNAP utility allowances are fully settled across all future PolicyEngine releases unless the NC phone-allowance data fix is upstreamed and retained.
  - Correct framing:
    - The current local benchmark set for North Carolina standard, limited, and telephone utility allowances plus the North Carolina child-support deduction election and self-employment expense option is green against compile, CI, generalist review, and PolicyEngine on the `rac-us-nc` jurisdiction repo path. That demonstrates the state-overlay `sets` pattern for both numeric utility parameters and delegated state-option booleans, but it does not by itself prove that upstream PolicyEngine releases already include the local NC phone-allowance correction.
- Do not claim that Tennessee SNAP as a whole is encoded or benchmark-closed.
  - Correct framing:
    - The Tennessee standard, limited, and telephone utility allowance overlays plus the Tennessee child-support deduction election and self-employment expense option are now green against compile, CI, generalist review, and PolicyEngine on the repaired harness. That is strong evidence that the state-overlay `sets` pattern now covers both numeric utility parameters and delegated state-option booleans. It still does not by itself establish broader Tennessee SNAP coverage beyond those slices.
- Do not claim that New York SNAP as a whole is encoded or benchmark-closed.
  - Correct framing:
    - The New York telephone, standard, and limited utility allowance overlays plus the New York child-support deduction election and self-employment expense option are now green against compile, CI, generalist review, and PolicyEngine on `rac-us-ny`, which is useful evidence that the state-overlay `sets` pattern transfers to a third jurisdiction for both multi-region utility schedules and delegated state-option booleans. It does not by itself establish broader New York SNAP coverage beyond those slices.
- Do not claim that California SNAP as a whole is encoded or benchmark-closed.
  - Correct framing:
    - The California self-employment expense option lane and the California child-support state-option lane are now green against compile, CI, generalist review, and PolicyEngine on `rac-us-ca`. That is useful evidence that the delegated `sets` pattern covers both a parameter-backed SNAP self-employment option and a delegated state-option boolean in California, but it does not by itself establish broader California SNAP coverage beyond those slices.
- Do not claim that Colorado SNAP as a whole is encoded or benchmark-closed.
  - Correct framing:
    - The Colorado self-employment expense option lane and the Colorado child-support state-option lane are now green against compile, CI, generalist review, and PolicyEngine on `rac-us-co`. That is useful evidence that the delegated `sets` pattern covers both a parameter-backed SNAP self-employment option and a delegated state-option boolean in Colorado, and that structural cleanup now generalizes both to state regulation citations like `10 CCR 2506-1` and slash-form effective dates like `10/01/2023`. It still does not by itself establish broader Colorado SNAP coverage beyond those slices.
- Do not claim that Texas SNAP as a whole is encoded or benchmark-closed.
  - Correct framing:
    - The Texas standard, limited, and telephone utility allowance overlays plus the Texas child-support deduction election and self-employment expense option are now green against compile, CI, generalist review, and PolicyEngine on `rac-us-tx`. That is useful evidence that the state-overlay `sets` pattern transfers to a sixth jurisdiction for both numeric utility parameters and delegated state-option booleans, and that handbook-section/revision/form numeric cleanup now generalizes beyond bulletin-style sources. It does not by itself establish broader Texas SNAP coverage beyond those slices.
- Do not claim that the current-effective federal SNAP eligibility lane is fully person-granular.
  - Correct framing:
    - The `is_snap_eligible` benchmark is now green against compile, CI, generalist review, and PolicyEngine, but the accepted artifact still carries non-blocking household-level compression of the member-disqualification facts. This is a clean closeout for the specific benchmark slice, not a proof that general person-to-household aggregation is solved across the ontology.
- Do not claim that Atlas already has a finished program-level match/mismatch dashboard.
  - Correct framing:
    - We have durable run artifacts and per-rule detail panes, but not yet a full aggregate dashboard of program-level agreement.

## Reproducible Count Commands

Use these commands when refreshing paper numbers.

```bash
ROOT="$(git rev-parse --show-toplevel)"
PARENT="$(dirname "$ROOT")"
export ROOT PARENT
python - <<'PY'
import os
from pathlib import Path
repos = {
    'rac-uk': Path(os.environ['PARENT']) / 'rac-uk',
    'rac-us': Path(os.environ['PARENT']) / 'rac-us',
    'rac-us-fl': Path(os.environ['PARENT']) / 'rac-us-fl',
    'rac-us-ca': Path(os.environ['PARENT']) / 'rac-us-ca',
    'rac-us-co': Path(os.environ['PARENT']) / 'rac-us-co',
    'rac-us-ny': Path(os.environ['PARENT']) / 'rac-us-ny',
    'rac-us-tx': Path(os.environ['PARENT']) / 'rac-us-tx',
    'rac-ca': Path(os.environ['PARENT']) / 'rac-ca',
}
for name, path in repos.items():
    print(name, sum(1 for _ in path.rglob('*.rac')))
PY
```

```bash
ROOT="$(git rev-parse --show-toplevel)"
export ROOT
python - <<'PY'
import os
from pathlib import Path
import yaml
base = Path(os.environ['ROOT']) / 'benchmarks'
for name in [
    'uk_wave18_soak.yaml',
    'uk_wave19_scale_seed.yaml',
    'uk_wave20_bulk_seed.yaml',
    'uk_wave21_bulk_seed.yaml',
    'uk_policyengine_readiness.yaml',
]:
    data = yaml.safe_load((base / name).read_text())
    print(name, len(data.get('cases', [])))
PY
```

## Missing Pieces For A Stronger Paper

- Florida claim guard:
  - We can now say the delegated SNAP state-option pattern transfers to Florida on two official Florida ESS manual slices, one for child-support deduction election and one for self-employment actual-expense treatment, both closing fully ready on the first event-driven Codex queue pass.
- Maryland claim guard:
  - We can now say the Maryland self-employment slice only supports the simplified deduction-rate parameter, not the expense-based boolean option, and the corrected Maryland rate benchmark closes fully ready on the event-driven Codex queue once the ontology is aligned to that source.
- Georgia claim guard:
  - We can now say the delegated SNAP state-option pattern transfers to Georgia on three official DFCS manual slices: child-support deduction election, self-employment actual-expense treatment, and the self-employment simplified deduction rate. The only accepted system repair was reusable numeric cleanup for integer manual section numbers like `3035` and `3425`.
- South Carolina claim guard:
  - We can now say the delegated SNAP state-option pattern transfers to South Carolina on three official DSS SNAP Manual Vol 65 slices: child-support deduction election, self-employment actual-expense treatment, and the self-employment simplified deduction rate. The accepted system repair was reusable numeric cleanup for manual volume labels like `Vol 65`; no manual RAC editing was needed.
- A shareable or publishable benchmark artifact registry, not just a local archive.
- A table of before/after metrics for each accepted harness change.
- Record-level public comparison data for UK that can be shown without licensing constraints.
- A citation-ready appendix generated directly from `results.json` and `summary.json`.
