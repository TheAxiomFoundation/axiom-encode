# Native interval-table and scale encoding

## Problem

RuleSpec currently treats source tables with interval row labels as a special
case: the encoder emits a derived integer band selector, keeps the structural
row bounds inline in that selector, then indexes source-stated table columns by
the selector. That is better than exporting public scalar parameters such as
`band_0_upper_bound`, but it still leaves two problems:

- If a downstream derived formula reuses the same row bounds, the file has
  source-stated thresholds embedded in executable arithmetic. The ACA PTC
  applicable-percentage interpolation is the motivating example: the selector
  chooses the FPL tier, then the interpolation formula repeats `133`, `150`,
  and the other tier endpoints. The generated ACA PTC file is not in the main
  corpus yet, so this example is forward-looking but already appears in active
  encoding worktrees.
- Even selector-only inline bounds are not reformable. They are not parameters,
  so a parameter update cannot move a tier boundary without changing formula
  text.

The rule should be: source-stated interval bounds, rates, amounts, caps, and
columns belong in structured data. Formulas should ask the structured data for
the active row, row values, or interpolation result.

## Inspiration from PolicyEngine/OpenFisca

PolicyEngine and OpenFisca keep scales as parameter data rather than formula
text. In the shared core, `ParameterScale` reads a structured `brackets` list
with fields such as `threshold`, `rate`, `amount`, and `average_rate`, then
materializes tax-scale runtime objects. PolicyEngine-US ACA
required-contribution percentages use the same separation in a lighter form:
versioned arrays hold the FPL thresholds plus initial/final percentages, and the
formula computes the bracket index and interpolation from those arrays. More
traditional taxes use bracket structures with threshold/rate fields.

Axiom should borrow the separation of scale data from formulas, not copy the
implementation. In particular, RuleSpec should avoid parallel arrays that can
drift out of alignment and instead keep each legal row's bounds, cells, and
proof material together. RuleSpec also needs to preserve legal provenance,
proof atoms, and source-slice granularity, so interval tables should remain
first-class RuleSpec rules rather than becoming an OpenFisca-compatible
parameter dump.

## Proposed RuleSpec surface

Add a first-class `kind: interval_table` rule for source tables whose row keys
are ranges or thresholds. It should be versioned like parameters and cite the
same source text as today.

```yaml
- name: aca_ptc_applicable_percentage_table
  kind: interval_table
  entity: TaxUnit
  period: Year
  input: household_income_as_percent_of_poverty_line
  boundary_dtype: Decimal
  rows:
    lower_inclusive: true
    upper_inclusive: false
  columns:
    initial_applicable_percentage:
      dtype: Rate
    final_applicable_percentage:
      dtype: Rate
  versions:
    - effective_from: '2026-01-01'
      values:
        - key: 0
          lower: null
          upper: 133
          upper_inclusive: true
          values:
            initial_applicable_percentage: 0.021
            final_applicable_percentage: 0.021
        - key: 1
          lower: 133
          lower_inclusive: false
          upper: 150
          values:
            initial_applicable_percentage: 0.03
            final_applicable_percentage: 0.04
```

`key` is explicit rather than derived from row order so generated tests,
manifests, provenance, and downstream parameter references can remain stable
when source rows are inserted or split. The runtime should reject duplicate
keys and row overlaps. Encoders should still keep keys mechanically assigned
and avoid parallel arrays, because the legal row's bounds, cells, key, and proof
material live together in one row object.

For multi-dimensional schedules, keep the interval axis explicit and model
other axes as selectors around it rather than flattening everything into one
formula. For example, filing-status-by-income tax brackets can be represented
as one interval table per filing-status category, or as an outer categorical
index whose values are interval tables. ACA SLCSP-like data with age and region
should similarly keep age/rating-area keys outside the interval-row definition.
The native surface should therefore allow `input` to evolve into either a
single interval input plus categorical keys, or a map of dimensions with exactly
one interval axis per table.

Default intervals should be half-open: lower-inclusive and upper-exclusive,
with the last row allowed to use `upper: null` for an open-ended interval. If a
source table uses wording such as "not over" or "over", the row may override
the table default with `lower_inclusive` or `upper_inclusive`. The runtime must
reject ambiguous adjacent rows unless the RuleSpec explicitly provides a
deterministic precedence rule, because `133` cannot safely belong to two rows.

The runtime can expose a small set of operations:

- `table.band(input)` returns the matched row key.
- `table.value(input, "column")` returns a column value for the matched row.
- `table.interpolate(input, "initial_column", "final_column")` performs linear
  interpolation between the row's lower and upper bounds when the source table
  defines beginning and ending values.

Interpolation should be limited to rows with finite lower and upper bounds and
explicit start/end columns. Exact lower endpoints return the row's start value;
exact upper endpoints follow the table's boundary policy. Rows with equal
start/end values are constant. Open-ended rows, non-linear tables, or tables
that assign a single value per row should use `table.value(...)` or an explicit
formula over table values instead of `table.interpolate(...)`.

Worked boundary example: with half-open rows `[133, 150)` and `[150, 200)`,
`input == 150` selects the second row and returns that row's start value. That
is correct for continuous schedules where the first row's final value equals
the second row's start value; for discontinuous schedules the source must state
which side owns the shared endpoint, or the table should use `table.value(...)`
with explicit row inclusivity instead of interpolation.

The exact formula syntax can evolve with the rules engine, but the important
constraint is stable: the bounds and cells are data, not repeated literals in
derived formulas.

## Encoder changes

The encoder prompt should stop saying that structural row bounds may remain
inline indefinitely. Instead:

- Before runtime support lands, keep the current selector-plus-indexed-column
  fallback, but never reuse selector bounds outside the selector. If a derived
  formula needs row bounds for interpolation, emit the bounds as indexed
  parameter columns such as `tier_lower_bound[tier]` and
  `tier_upper_bound[tier]`, then interpolate with those names:
  `(income - tier_lower_bound[tier]) / (tier_upper_bound[tier] -
  tier_lower_bound[tier])`. The literals then live only in selector predicates
  and source-backed parameter `values`, not in derived formula arithmetic.
- After runtime support lands, emit `kind: interval_table` for interval-keyed
  source tables and use table operations for selection and interpolation.
- Keep simple non-interval keyed tables as existing `kind: parameter` rules with
  `indexed_by`.
- Preserve one source proof per table row or column cell where the source
  granularity supports it.

## CI and audit strategy

This change has two validation layers:

- CI-blocking validation for new or changed files: fail when a non-selector
  formula repeats a numeric literal that appears as a source-table selector
  bound in the same RuleSpec file. This catches the ACA interpolation pattern
  without banning the current selector fallback.
- The blocking gate is pattern-specific: it detects reuse of bounds from a
  recognized structural selector, not every possible hard-coded interval bound.
  Selector rules must keep conventional names containing `band`, `bracket`,
  `tier`, or `index`, and interval-table source rows should produce a sibling
  selector for the gate to have teeth.
- Workspace inventory: `axiom-encode interval-table-audit` scans `rulespec-*`
  repos and reports files that need re-encoding. With
  `--include-selector-bounds`, it also lists selector-inline bounds that are
  allowed today but should migrate after native interval-table support exists.
- Direct branch-formula scales without a selector remain covered by the existing
  structured-scale validator and are included in the audit as
  `branch_formula_scale` candidates. Those should migrate to indexed parameters
  now, or native interval tables when the row keys are ranges.

The audit output is intentionally file-oriented: repo, path, rule name, literal,
issue kind, and message. That gives us a re-encoding queue without hand-editing
generated RuleSpec.

## Migration plan

1. Land the audit and design in `axiom-encode`.
2. Teach the encoder prompt/repair path to use indexed lower/upper-bound
   columns for interpolation while native interval tables are unavailable.
3. Add `kind: interval_table` schema and runtime operations in
   `axiom-rules-engine`.
4. Update encoder prompts, repair logic, and validator expectations to emit
   interval tables directly.
5. Run `interval-table-audit --include-selector-bounds` across RuleSpec repos to
   produce the migration queue.
6. Re-encode affected files from source using the encoder. Generated artifacts
   and manifests may change; direct RuleSpec YAML edits remain prohibited.
7. Tighten CI so selector-inline bounds fail once native interval tables are
   available and the main inventory is migrated.

## Open questions

- Should interval tables own row proof atoms, cell proof atoms, or both?
- Should interpolation be a table operation, a formula helper over table rows, or
  both?
- Do we need open/closed boundary flags per row, or can table-level defaults
  plus per-row overrides cover statutory schedules cleanly?
- Reforms should prefer per-field path updates such as
  `rows[1].upper` or `rows[1].values.final_applicable_percentage`; whole-table
  replacement remains a fallback for broad schedule rewrites.
