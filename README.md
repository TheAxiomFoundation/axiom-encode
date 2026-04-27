# Axiom Encode

AI-assisted Axiom RuleSpec encoding infrastructure. This repo owns the automation layer for turning source legal text into `rules.yaml` / `*.yaml` RuleSpec artifacts and validating them with `axiom-rules`.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
axiom-encode encode "26 USC 32(a)(1)" \
  --output /tmp/axiom-encode-encodings

axiom-encode validate /tmp/axiom-encode-encodings/codex-gpt-5.5/26/32/a/1.yaml
```

## Eval suites and readiness gates

Use manifest-driven benchmark suites when you want an explicit readiness answer instead
of ad hoc spot checks.

```bash
axiom-encode eval-suite benchmarks/us_snap_ny_standard_utility_allowance_refresh.yaml
axiom-encode eval-suite benchmarks/us_snap_federal_reconstruction_seed.yaml
axiom-encode eval-suite-archive /tmp/axiom-encode-suite-run
```

- `benchmarks/us_snap_*_refresh.yaml` manifests are source-file-backed SNAP refresh lanes.
- `benchmarks/us_co_*` manifests exercise Colorado Works repair and seed cases.
- `benchmarks/uk_pension_credit_wave*.yaml` manifests cover source-backed UK pension-credit slices.

Each suite reports:
- success rate
- compile pass rate
- CI pass rate
- zero-ungrounded rate
- PolicyEngine pass rate on oracle-mappable cases
- mean estimated cost

The command exits `0` only when all readiness gates pass.

Use `eval-suite-archive` when a run is worth citing later. It copies the full
suite output tree into a durable local registry under `artifacts/eval-suites`
by default, rewrites archived JSON/JSONL artifact paths away from `/tmp`, and
appends a record to `artifacts/eval-suites/index.jsonl`.

## Methods and paper notes

The repo now keeps paper-oriented internal documentation for benchmark-relevant
changes and current evidence.

- `docs/axiom-encode-methods-log.md` tracks the last meaningful harness changes,
  their hypotheses, and the evidence path to justify them later.
