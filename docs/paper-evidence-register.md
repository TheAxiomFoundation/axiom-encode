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

- A stable benchmark artifact registry outside `/tmp`.
- A table of before/after metrics for each accepted harness change.
- Record-level public comparison data for UK that can be shown without licensing constraints.
- A citation-ready appendix generated directly from `results.json` and `summary.json`.
