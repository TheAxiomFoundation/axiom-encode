# PolicyEngine Oracle Registry

RuleSpec is the executable legal computation layer. PolicyEngine mappings are
oracle-adapter data, so they live in `axiom-encode`, outside RuleSpec and outside
`rulespec-*` repositories.

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

When a non-comparable output looks close to PolicyEngine, keep the adjacent
`policyengine_variable` or `policyengine_parameter` in the registry entry. If the
candidate has already been reviewed and should not stay high in the queue, add
`candidate_priority: P4` and record the concrete reason in `rationale`.

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

Parameterized PE tables can be keyed by a fixed `parameter_key`, by several
fixed `parameter_keys` that must agree, or by a RuleSpec test input through
`parameter_key_input` and `parameter_key_map`:

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

For nested PE parameter objects, use `parameter_key_path`. Path entries can
select from RuleSpec inputs and can optionally bound numeric keys before
lookup. This is for source-stated capped rows such as "10 or more", where the
RuleSpec output and PE parameter table share the same capped row:

```yaml
- legal_id: us-ca:policies/cdss/calworks/maximum-aid-payment-region-1#calworks_region_1_maximum_aid_payment
  mapping_type: parameter_value
  policyengine_parameter: gov.states.ca.cdss.tanf.cash.monthly_payment.region1
  parameter_key_path:
    - input: assistance_unit_is_exempt
      key_map:
        "True": exempt
        "False": non_exempt
    - input: persons_on_aid
      max_value: 10
```

Use `parameter_keys` when one legal output intentionally groups multiple PE
table cells that should have the same value:

```yaml
- legal_id: us:policies/irs/rev-proc-2025-32/standard-deduction#additional_standard_deduction_married_or_surviving_spouse
  mapping_type: parameter_value
  policyengine_parameter: gov.irs.deductions.standard.aged_or_blind.amount
  parameter_keys:
    - JOINT
    - SEPARATE
    - SURVIVING_SPOUSE
```

Keep these mappings in `axiom-encode`; do not put oracle parameter paths in
RuleSpec repositories.

## Repository Coverage Report

Use the coverage command before assigning autonomous encoding work and after a
batch lands:

```bash
axiom-encode oracle-coverage --root /path/to/workspace --program snap
```

The report scans sibling `rulespec-*` repositories, constructs canonical legal IDs
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

CI jobs that need oracle-tested RuleSpec should also fail on comparable outputs
missing from companion tests:

```bash
axiom-encode oracle-coverage \
  --root /path/to/workspace \
  --fail-on-unmapped \
  --fail-on-untested-comparable
```

## Candidate Triage

Use the candidate command to turn coverage into a priority queue:

```bash
axiom-encode oracle-candidates --root /path/to/workspace --program snap
```

The command prioritizes untested comparable mappings, unmapped outputs that look
like exact PolicyEngine variables, and known adjacent targets that may deserve a
small adapter. `candidate_priority` can lower already-reviewed non-comparables
without hiding them from the report.

## Cloud Queue Export

Use the cloud queue command to export deterministic work items before assigning
parallel encoding work:

```bash
axiom-encode cloud-queue --root /path/to/workspace --json
```

The queue is model-free orchestration input. It converts PolicyEngine program
surfaces into explicit work items without running encoders, opening branches, or
claiming that any surface is legally comparable. The current schema is:

```text
axiom-encode/policyengine-cloud-queue/v1
```

Each item includes:

- `action`: one of `ingest_source`, `encode_rulespec`, `wire_oracle_mapping`, or
  `bootstrap_jurisdiction`
- `priority`: the program-surface priority such as `P1` or `P2`
- `target_repo` and `target_prefix`: where generated RuleSpec or source work
  should eventually land
- `policyengine_variable`: the oracle surface that motivated the item
- `lock_scopes`: deterministic repo, prefix, legal-ID, and source locks for
  future cloud workers
- `oracle_expectation`: what a successful worker should prove or classify

By default the queue includes `pending_source_ingestion`,
`pending_rulespec_encoding`, and `pending_oracle_mapping` surfaces. Add
`--include-deferred-jurisdictions` when planning repo/bootstrap work for
jurisdictions that do not have a ready RuleSpec target yet.

Cloud workers should emit artifacts using the companion run-artifact schema:

```text
axiom-encode/policyengine-cloud-run-artifact/v1
```

At minimum, run artifacts should preserve the work item ID, encoder version and
git SHA, corpus reference, RuleSpec base SHA, model and prompt hash, generated
diff, validation logs, oracle results, retry history, and final classification.
