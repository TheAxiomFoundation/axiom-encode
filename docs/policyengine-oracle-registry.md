# PolicyEngine Oracle Registry

RuleSpec is the executable legal computation layer. PolicyEngine mappings are
oracle-adapter data, so they live in `axiom-encode`, outside RuleSpec and outside
`rules-*` repositories.

## Boundary

RuleSpec may contain legal identity and execution semantics: canonical legal IDs,
imports, entities, periods, units, dtypes, formulas, parameters, tables,
effective dates, and minimal authority pointers.

RuleSpec must not contain PolicyEngine/TAXSIM mappings, reviewer metadata,
ingestion traces, AKN/source-document artifacts, oracle tolerances, scenario
defaults, or harness heuristics.

## Mapping Keys

Mappings are keyed by canonical Axiom legal output IDs:

```text
us:statutes/7/2014/e/2#snap_earned_income_deduction
```

Friendly suffixes such as `snap_earned_income_deduction` are not registry keys.
The harness may display suffixes for humans, but oracle lookup uses legal IDs.

## Legal ID Lifecycle

Canonical legal IDs should remain stable while the legal authority and computed
concept remain stable. When a provision is renumbered, repealed, replaced, or
split, do not silently retarget the old ID. Add an explicit alias/deprecation
record in oracle or registry metadata and keep the mapping rationale clear.

If an amendment changes the computation without changing the provision path,
RuleSpec should model the change with effective dates under the same legal ID.
If the source authority moves or the computed concept changes, create a new
legal ID and mark the old mapping deprecated outside RuleSpec.

## Coverage Accounting

Oracle runs report coverage separately from match score:

- `comparable`: outputs actually compared to PolicyEngine
- `passed`: comparable outputs that matched
- `failed`: comparable outputs that mismatched
- `unmapped`: legal IDs with no PolicyEngine mapping
- `unsupported`: known not-comparable or unsupported scenario cases
- `adapter_errors`: scenario construction, runtime, or parse failures
- `setup_errors`: missing PolicyEngine runtime/setup failures

Unmapped and unsupported outputs are not scored as mismatches, but they must stay
visible so a green score cannot hide weak oracle coverage.
