# RuleSpec Proof Validation

## Purpose

RuleSpec is the only executable Axiom layer. Corpus stores source text,
anchors, hierarchy, tables, and hashes. Source claims store reviewed,
evidence-backed assertions over corpus anchors. Encoding agents may use claims
and reachable RuleSpec modules as context, but they must not turn missing
context into invented executable policy.

The production invariant is:

```text
Every policy-bearing RuleSpec atom must be justified by direct target source
text, an accepted evidence-backed claim, or an approved imported RuleSpec
export. If no proof can be built, encoding stops with a typed request.
```

`module.source_claims` is only a coarse module-level allowlist. It does not by
itself prove that a threshold, table cell, formula predicate, default, unit,
effective date, exception, definition, or imported dependency is legally
grounded. The durable gate is a proof tree per policy-bearing atom.

## Boundary

The layers have separate authority:

- `corpus`: verbatim source text, deterministic anchors, hierarchy, extracted
  tables, source versions, and content hashes.
- `claims`: thin reviewed assertions about what a source span says or how one
  source relates to another. Claims may say `sets`, `defines`, `amends`,
  `implements`, `restates`, `delegates`, and similar source relations.
- `rulespec`: executable policy logic, parameters, imports, tests, and runtime
  outputs.

Claims must stay non-executable. They can justify an encoding atom, but they
cannot contain formulas, case inputs, outputs, eligibility decisions, test
cases, traces, or benefit amounts as runnable logic.

## Policy-Bearing Atoms

The proof validator treats the following as policy-bearing:

- thresholds, rates, caps, amounts, percentages, and table cells;
- formulas, predicates, branch conditions, defaults, and fallback behavior;
- units, periods, effective dates, sunset dates, and jurisdictions;
- exceptions, exclusions, ordering rules, incorporated references, and
  definitions;
- imported RuleSpec outputs used by a formula.

A proof can cover a whole table, a single cell, a scalar parameter, a formula
predicate, an effective period, or an import. The validator should eventually
derive as many atom paths as possible from the compiled artifact, but RuleSpec
authors and encoders can declare proof atoms explicitly first.

## Proof Sources

Each atom has one or more proof sources.

### Direct Source

Direct source proof points to a corpus citation path and, where possible, a
content-addressed anchor:

```yaml
metadata:
  proof:
    atoms:
      - path: versions[0].values
        kind: parameter_table
        source:
          corpus_citation_path: us/guidance/usda/fns/snap-fy2026-cola/page-1
          value_key: snap_maximum_allotment_table
          table:
            header: Maximum Monthly Allotment
            row_key: household_size
            column_key: 48_states_dc
```

For table-backed atoms, provenance should identify the source table, header,
row, column, cell or cell range, footnote context, extraction version, and
source hash as soon as ingestion provides those fields.

### Claim

Claim proof may reference only accepted claims already listed in
`module.source_claims`. `axiom-encode proof-validate` resolves those IDs against
local `axiom-corpus/claims` artifacts and fails if a claim is missing,
unaccepted, executable, not evidence-backed, or uses a friendly placeholder
subject instead of an absolute legal/corpus/RuleSpec target:

```yaml
metadata:
  proof:
    atoms:
      - path: versions[0].values
        kind: parameter_table
        claim:
          id: claims:us/guidance/usda/fns/snap-fy2026-cola/page-1#sets-maximum-allotments
```

Proposed, stale, rejected, superseded, or executable claims cannot justify live
RuleSpec. Claim subjects must be absolute targets such as
`us:statutes/7/2014/e` or
`us:statutes/7/2017/a#snap_allotment_before_minimum.input.snap_maximum_allotment`;
friendly concept IDs such as `snap.maximum_allotment` are invalid.

### Import

Import proof must name an explicit exported RuleSpec output, not an ambient
module:

```yaml
metadata:
  proof:
    atoms:
      - path: versions[0].formula
        kind: import
        import:
          target: us:statutes/7/2017/a#snap_regular_month_allotment
          output: snap_regular_month_allotment
          unit: USD
          period: Month
          hash: sha256:<compiled-export-hash>
```

Export contracts should include output, units, period, entity, jurisdiction,
effective dates, assumptions, provenance, and hash.

## Typed Stop Requests

When a proof cannot be built, the encoder should stop and emit one of these
artifacts instead of producing executable RuleSpec:

- `missing_claim`: a needed source assertion has not been reviewed.
- `bundle_expansion_request`: the relation closure is too narrow.
- `corpus_defect`: source text, table extraction, or metadata is malformed.
- `segmentation_fix`: the corpus anchor is too coarse or too fine.
- `stale_claim`: a claim references an outdated source version.
- `conflicting_claims`: accepted claims disagree.

These are encoder outputs for workflow triage, not executable policy.

## Implementation Order

1. Build a standalone proof validator: sealed bundle plus RuleSpec module in,
   per-atom pass/fail with typed failures out.
2. Validate explicit proof atoms in RuleSpec.
3. Add sealed bundle manifests with source, claim, import, table extraction,
   and compiled export versions.
4. Derive proof obligations from compiled RuleSpec so missing atoms are found
   automatically.
5. Teach encoders to emit typed stop requests when a proof obligation cannot be
   satisfied.
6. Make strict proof validation a promotion gate once existing RuleSpec files
   declare proof trees.

This sequence keeps the execution layer clean while avoiding a heavy
interpretation layer that quietly becomes RuleSpec without tests.
