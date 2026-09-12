# Lifetime companion fixtures

The encoder can execute a multi-period companion case through an actual Axiom
Rust CLI implementing `run-lifetime --artifact`. This is an opt-in fixture shape;
ordinary scalar cases keep their existing behavior. The engine must support
`axiom-rules-engine/lifetime-request/v1`, response v1 and compiled artifact v2
for ordinary lifetime cases. Explicit calculation-period cases require request
and response v2 support.
An older CLI fails validation rather than falling back to scalar evaluation.
Production generation must still use the reviewed, pinned engine and protected
encoder workflow. This feature does not update a repository's toolchain pins.

For a job that selects an engine commit explicitly, pass
`--axiom-rules-engine-ref` with its full lowercase commit SHA to `encode`, along
with `--axiom-rules-engine-path`. The checkout must be clean and its HEAD must
equal that SHA. The existing `engine-bind` receipt resolver verifies the binary
or builds it from that checkout. Generation, retained-candidate validation,
repairs, companion tests, and dependent overlay/baseline checks all receive the
same ref. Successful generation and apply validation recheck the source and
binary; changes prevent success or apply. The additional binding stays in the
in-memory validation snapshot, and the portable signed-manifest format remains
unchanged. Omitting the option preserves existing engine selection behavior.
See [Pinning the engine for encode](encode-engine-binding.md) for the complete
binding and upgrade contract.

Each lifetime case has only `name`, optional `description`, `period`, `output`,
and `lifetime`. The lifetime mapping contains `entity`, optional
`arithmetic: decimal`, `periods`, and one `batches` entry per period. Quote dates,
decimal input values, decimal expectations, and numeric-looking entity IDs.

```yaml
- name: two observation periods
  period: {period_kind: tax_year, start: '2021-01-01', end: '2021-12-31'}
  lifetime:
    entity: Person
    periods:
      - {period_kind: tax_year, start: '2020-01-01', end: '2020-12-31'}
      - {period_kind: tax_year, start: '2021-01-01', end: '2021-12-31'}
    batches:
      - row_count: 1
        entity_ids: ['001']
        inputs:
          'us:statutes/99/1#input.amount': {kind: decimal, values: ['0.1']}
      - row_count: 1
        entity_ids: ['001']
        inputs:
          'us:statutes/99/1#input.amount': {kind: decimal, values: ['0.2']}
  output:
    'us:statutes/99/1#total': '0.3'
```

This is a synthetic transport example for a compiled `sum_over_periods(amount)`
rule, not an encoded legal provision. The other lifetime builtins are
`max_over_periods`, `count_over_periods` and `sum_top_n_over_periods`. These names
are functions, never input slots. Facts must use the compiled program's public
input references; supplying a computed result as a fact fails.

Engines exposing `calendar_years_to_months(year_count)` can convert a grounded
count of whole Gregorian calendar years into an Integer month count. The
operation accepts integers and exactly integral Decimals; fractions, Float
columns, booleans and checked-integer overflow fail. Use Decimal arithmetic for
derived counts. It supplies calendar-unit semantics only: the source must still
establish the selected years and any exclusions. A partial year or an excluded
month cannot be represented by converting a whole-year count.

For a top-N sum over complete annual observations, convert the same grounded N
used by the reduction. Selected-year ties, leap days and gaps between selected
years do not change the month count per complete year. This does not fill missing
observations or justify eligibility, and an unknown or computed value must not
be supplied as a fabricated input. The builtin also accepts a computed lifetime
argument; the engine evaluates that argument through its normal reduction path.
Older engine builds without the primitive fail compilation explicitly.

The adapter requires 1–512 explicit periods and 1–100,000 rows per batch, with
the same unique entity IDs in the same order in every batch. It asserts every
row, using a scalar expectation for one row or a list for multiple rows. Decimal
expectations use ASCII decimal strings without exponent notation and are compared
exactly. Booleans, integers, dates, text and judgments preserve their types.
Inputs and results are validated by the real Rust runtime and the typed response
contract. No Python policy calculation or floating-point tolerance is involved.

Without `lifetime.calculation_period`, the output period must equal the last
supplied observation period. This v1 interface does not add a separate legal
determination date, insert missing years, infer
relations or reorder entities. The current engine refuses unsupported plans and
periods before the compiled rules' commencement. Ordinary helpers without a
lifetime reduction belong in scalar cases. Inputs used outside a reduction must
remain invariant across periods, as required by the engine.

To evaluate completed history under law selected at a later date, add
`calculation_period` inside `lifetime` and set the top-level `period` to the same
explicit mapping. This selects request v2; it never retries through v1. For the
synthetic example above, both may be set to
`{period_kind: month, start: '2026-01-01', end: '2026-01-31'}` while leaving the
2020 and 2021 observations and batches unchanged.

Every observation must end before calculation starts. The real engine selects
formula and complete parameter-table versions at `calculation_period.start`,
using inclusive source bounds. Historical date expressions within reductions
retain observation dates; a missing table key never borrows an older version.
The adapter checks the v2 response's calculation/reference/output dates and
selected-version identities, indices and ranges before comparing typed results.
It does not evaluate formulas or choose legal versions. An older runtime, missing
version, incomplete history, malformed provenance or schema downgrade fails.
Exact required fixture contracts also preserve the calculation date through
generation and final admission.

V2 is fixed-law completed-history execution, not knowledge-time or mixed-law
selection. It does not supply missing observations, statutory eligibility,
relations, or partial-year rules. The legal source still determines the correct
calculation date and any input invariance requirements. Run
`tests/test_calculation_lifetime_engine_integration.py` against a v2-capable
binary to verify the transport and source-version boundaries with synthetic facts.

The scalar PolicyEngine oracle adapter cannot evaluate lifetime fixtures. It
records unsupported coverage without projecting the history into a scalar case.
Where other scalar cases are comparable, their existing oracle score covers only
those comparable outputs. It does not establish oracle parity for the lifetime
outputs. Other source-coverage requirements remain in force.

Run the synthetic integration tests against an explicitly selected real build:

```sh
AXIOM_LIFETIME_TEST_ENGINE=/absolute/path/to/axiom-rules-engine \
  uv run pytest --no-cov tests/test_lifetime_engine_integration.py
```

These tests compile fresh synthetic modules and execute all four reductions,
verify a decimal discrepancy below floating-point resolution, and reject a
computed output supplied as a historical fact. They skip if the binary is not
explicitly configured; default Python CI alone does not prove engine integration.

With a build exposing the calendar primitive, also run
`tests/test_calendar_unit_engine_integration.py` under the same environment.
Those synthetic fixtures exercise an arithmetic-derived year count, conversion
of a lifetime count, exact typed output, zero earnings, tied values and separated
complete years. A fractional derived count must fail rather than truncate.
