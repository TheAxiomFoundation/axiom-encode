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

Known non-comparable legal IDs should be registry entries with
`mapping_type: not_comparable` and a short rationale. Truly new legal IDs should
remain `unmapped` until someone classifies them.

For large legal namespaces where most outputs are source-specific intermediates,
the registry may use `prefixes` entries. Prefix entries only classify
`not_comparable` outputs. Exact legal-ID mappings always override prefixes, so a
specific output can still become PolicyEngine-comparable later without changing
the RuleSpec file.

## Mapping Types

Use `mapping_type: direct_variable` when a RuleSpec output can be compared to a
PolicyEngine variable:

```yaml
- legal_id: us:statutes/26/3101/a#oasdi_wage_tax
  mapping_type: direct_variable
  policyengine_variable: employee_social_security_tax
```

Use `mapping_type: parameter_value` when the RuleSpec output is a statutory or
policy parameter that PolicyEngine stores in its parameter tree rather than as a
variable:

```yaml
- legal_id: us:statutes/26/3101/a#oasdi_wage_tax_rate
  mapping_type: parameter_value
  policyengine_parameter: gov.irs.payroll.social_security.rate.employee
```

Parameterized PE tables can be keyed by a fixed `parameter_key`, or by a
RuleSpec test input through `parameter_key_input` and `parameter_key_map`:

```yaml
- legal_id: us:statutes/26/3101/b/2#additional_medicare_wage_tax_threshold
  mapping_type: parameter_value
  policyengine_parameter: gov.irs.payroll.medicare.additional.exclusion
  parameter_key_input: filing_status
  parameter_key_map:
    0: SINGLE
    1: JOINT
    2: SEPARATE
    3: HEAD_OF_HOUSEHOLD
```

Keep these mappings in `axiom-encode`; do not put oracle parameter paths in
RuleSpec repositories.

## Repository Coverage Report

Use the coverage command before assigning autonomous encoding work and after a
batch lands:

```bash
axiom-encode oracle-coverage --root /path/to/workspace --program snap
```

The report scans sibling `rules-*` repositories, constructs canonical legal IDs
for every executable `kind: parameter` and `kind: derived` output, and classifies
each output as:

- `comparable`: an exact registry mapping can be run against PolicyEngine
- `known_not_comparable`: the registry deliberately classifies the output as out
  of oracle scope, usually because it is a legal intermediate or source-specific
  assembly point
- `unmapped`: no registry entry or prefix covers the output yet

Autonomous agents should treat `unmapped` as work queue material. They should
either add an exact oracle mapping, add a narrow `not_comparable` classification
with a rationale, or improve the RuleSpec if the unmapped output is accidental.
Do not hide unmapped outputs in RuleSpec metadata.

CI jobs that need complete classification can use:

```bash
axiom-encode oracle-coverage --root /path/to/workspace --fail-on-unmapped
```
