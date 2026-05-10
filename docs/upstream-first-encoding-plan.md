# Upstream-First Encoding Plan

## Goal

Axiom should encode policy in legal-authority order so each downstream source can
distinguish a genuinely new rule from a restatement, delegation, option setting,
or amendment of upstream law. The encoder should not rely on program-specific
variable-name heuristics to make that decision. It should use deterministic
corpus metadata plus already validated RuleSpec artifacts and `source_relation`
records.

For SNAP, that means encoding federal statutes before federal regulations, USDA
policy documents before state manuals that repeat USDA values, and Colorado
materials only where Colorado actually adds, sets, implements, or restates
something.

## Canonicality Model

Authority order is a lookup default, not the final test for canonicality. A
lower-authority source can be canonical for a value or option when upstream law
delegates that choice downstream.

Example: a federal regulation can define a standard utility allowance slot, while
a state manual is canonical for the Tennessee value that fills that slot. The
state value should not be rejected as downstream duplication; it should encode
a `source_relation.type: sets` edge pointing to the upstream slot.

Canonicality is decided by relation edges, delegated slots, concept keys, source
spans, and effective periods. Rank only decides which candidates the encoder
should inspect first.

## Source Order

The default lookup and batch order is:

1. Constitutions, treaties, and organic authority.
2. Federal statutes.
3. Federal regulations.
4. Federal agency policy, notices, parameter publications, and guidance.
5. State statutes.
6. State regulations.
7. State agency manuals, operational guidance, notices, worksheets, and forms.

This order should guide planner queues and upstream retrieval. It should not
force every lower-rank document into a restatement edge when that document is
lawfully setting, implementing, or amending an upstream slot.

## Corpus Metadata and RuleSpec Registry

Ingestion should write every source slice to `corpus.provisions` with verbatim
source text and structured deterministic metadata:

- `citation_path`: stable Axiom source path.
- `source_span_id`: stable id for paragraph/table/row/subsection spans.
- `jurisdiction`: `US`, `CO`, `TN`, etc.
- `source_type`: statute, regulation, agency_policy, manual, notice, form, etc.
- `authority_level`: comparable rank for default ordering.
- `effective_start` and `effective_end` where available.
- `source_url` and source storage pointer.
- deterministic citations or cross-references that can be extracted without
  policy interpretation.

The clean boundary is:

1. **Corpus:** stores source text, source spans, citations, effective dates, and
   other deterministic retrieval metadata.
2. **RuleSpec registry:** indexes validated RuleSpec files, symbols, absolute
   targets, imports, and `kind: source_relation` records.

Legal/provenance edges that require interpretation belong in RuleSpec, not in a
separate intermediate relation layer. The registry can index those RuleSpec
relations for lookup, but the checked-in RuleSpec file is the source of truth.

Core relation types:

- `defines`: the source/span is the canonical home for an executable concept.
- `delegates`: the source/span creates a downstream slot for another source to
  set or implement.
- `implements`: the source/span supplies mechanics delegated by upstream law.
- `sets`: the source/span sets a delegated value, option, rate, region, or
  parameter.
- `amends`: the source/span changes an upstream source, slot, or target.
- `restates`: the source/span restates an upstream source or target without
  adding executable semantics.
- `cites`: the source/span references another source without resolving the legal
  relationship.

Relations can attach to spans, not only whole files. A single source slice can
produce multiple RuleSpec records: one source-relation restatement, one
delegated setting, and one downstream implementation rule, each tied to the
source span that justifies it.

## Target Keys

Duplicate detection must not compare only RuleSpec symbol names. Every canonical
target should be identified by an absolute path:

- corpus citation path for source text or spans,
- statute/regulation/policy path for legal provisions,
- RuleSpec path plus `#rule_name` for executable outputs,
- RuleSpec path plus a slot suffix for imported inputs or delegated settings.

Rule names are still useful for humans and imports, but duplicate detection
should compare absolute targets and declared relation targets. This is what
prevents the harness from falling back into variable-name heuristics or
model-generated friendly labels.

## Rule Identity and Outputs

Bare rule names are local symbols, not durable public identifiers. The durable
identity of an executable rule should be the canonical RuleSpec target plus the
rule name:

```text
us:statutes/7/2017/a#snap_regular_month_allotment
us-co:regulations/10-ccr-2506-1/4.403.2#snap_countable_earned_income
```

Formula authors should still be able to reference local and imported symbols by
bare name where the compiler can resolve them unambiguously. External surfaces
must key values by the canonical id whenever one exists. That includes API
execution results, notebook demos, traces, registry rows, and verification
artifacts. A notebook can display `snap_regular_month_allotment`, but the
machine-readable output must carry
`us:statutes/7/2017/a#snap_regular_month_allotment` so calculated values
are keyed to the provision that defines them.

Companion `.test.yaml` files are also external surfaces. Their `input:` and
`output:` keys must be canonical legal RuleSpec references that resolve to real
files and fragments:

```yaml
input:
  us:statutes/7/2017/a#input.household_size: 1
  us:policies/usda/snap/fy-2026-cola/maximum-allotments#snap_maximum_allotment: 298
output:
  us:statutes/7/2017/a#snap_regular_month_allotment: 268
```

Bare friendly keys and unresolved absolute-looking placeholders are invalid.

## Encoding Workflow

For each source span:

1. Resolve corpus metadata and registry-neighboring RuleSpec targets.
2. Fetch period-overlapping upstream candidates in authority order.
3. Materialize all registry-resolved import targets into the eval workspace, or
   create explicit planned stubs before model invocation.
4. Fetch sibling and downstream context only after upstream context.
5. Ask the encoder to classify each generated legal/source edge as one of:
   `defines`, `delegates`, `implements`, `sets`, `amends`, or `restates`.
6. Generate RuleSpec that records each legal/source edge as a
   `kind: source_relation` record.
7. Validate the artifact against the registry, local imports, source text,
   tests, and applicable oracles.
8. Persist new RuleSpec target metadata and trace data back into the registry so
   later downstream sources can discover it.

The encoder prompt should forbid executable duplication when a source is only
restating upstream law. It should also require an explicit justification when it
defines a new executable rule in a lower-authority source despite upstream
candidates existing.

## RuleSpec Source Relations

Legal/source relations should be explicit RuleSpec records, not executable-rule
metadata. Proposed shape:

```yaml
rules:
  - name: co_snap_standard_utility_allowance
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: '451'
  - name: standard_utility_allowance_setting
    kind: source_relation
    source_relation:
      type: sets
      target: us:regulation/7-cfr/273/9/d/6/iii#snap_standard_utility_allowance_slot
      value: us-co:policies/cdhs/snap/fy-2026#snap_standard_utility_allowance
```

Recommended `source_relation` fields:

- `type`: one of `defines`, `delegates`, `implements`, `sets`, `amends`,
  `restates`, or `cites`.
- `target`: absolute source or RuleSpec target.
- `value`: absolute RuleSpec target for the executable value when applicable.
- `basis.delegation`: absolute upstream delegation target when required.
- `amendment`: operation and effective interval for amendments.

For pure restatements, use the explicit non-executable source-relation shape:

```yaml
rules:
  - name: standard_deduction_restatement
    kind: source_relation
    source_relation:
      type: restates
      target: us:policies/usda/snap/fy-2026-cola/deductions#snap_standard_deduction
      authority: federal
```

The registry should index all RuleSpec relations so a downstream run can ask:

- What upstream source spans and RuleSpec targets exist for this citation/topic?
- What absolute targets are already defined?
- Which downstream sources set or amend this target?
- Which sources merely restate this target?

## Harness Rules

The harness should enforce RuleSpec source relations, not policy-specific
variable names.

Required deterministic checks:

- If upstream context shows that the source only restates a target, the file
  must include a `kind: source_relation` record with
  `source_relation.type: restates` and `source_relation.target`, scoped to the
  relevant source span.
- If upstream context shows that the source sets, implements, or amends a
  target, the file must include a `kind: source_relation` record with the
  corresponding `source_relation.type` and absolute `source_relation.target`.
- If a rule defines an absolute target already defined upstream and does not declare
  a matching `source_relation` record with `implements`, `sets`, `amends`, or
  `restates`, flag it as downstream duplication.
- If a downstream source cites an upstream provision and registry lookup
  resolves a unique target, require a materialized import, a relation marker, or
  an explicit non-import justification.
- If a source sets or amends an upstream target, verify that the target exists
  unless the run is explicitly creating a planned stub.

Program-specific checks can exist temporarily as fallback assertions, but they
should be deleted once RuleSpec relation metadata covers the relevant relation.

## Materialization Contract

The model should never be asked to import an upstream target that was not copied
or stubbed into its workspace.

For each registry-resolved upstream target before model invocation:

1. If the RuleSpec artifact exists, copy it into context and expose its canonical
   import path.
2. If the source exists but is not encoded, block downstream promotion and queue
   the upstream source first.
3. If a temporary stub is unavoidable, create a planned stub with lifecycle
   metadata.

Planned stubs must include:

- target citation or absolute target,
- relation type,
- source URL or source span id,
- missing reason,
- owner,
- expiration or replacement condition.

Downstream artifacts depending on planned stubs may pass generation but should
fail promotion until the upstream source is encoded or the stub is explicitly
waived.

## Period Semantics

Upstream lookup must be period-aware.

- Candidate targets are selected by overlap with the source span effective
  period.
- Annual parameter documents set slots for explicit effective intervals.
- Amendments create versioned replacement edges.
- Conflicting same-period targets block validation unless one relation is marked
  `supersedes` or `superseded_by`.
- Tests should use periods covered by the selected upstream RuleSpec relation
  records.

This is essential for USDA annual COLA documents, state option updates, and
regulatory amendments.

## Planner

Add an encoding planner that creates a deterministic DAG from
`corpus.provisions` plus the RuleSpec registry:

1. Group provisions by jurisdiction, program, source type, and concept topic.
2. Sort by `authority_level`, citation hierarchy, and effective period.
3. Add dependency edges from deterministic citations, citation hierarchy, and
   indexed RuleSpec `source_relation` records.
4. Encode missing upstream dependencies before downstream slices.
5. Requeue downstream slices when a newly encoded upstream artifact changes the
   candidate context.
6. Emit `axiom-encode plan --json` with source ids, span ids, effective
   periods, upstream candidates, known RuleSpec relation edges, blocked
   dependencies, and planned stubs.

For ad hoc encoding, the planner should still run a local upstream search before
calling the model. If required upstream context is missing, the default should be
to encode or stub the upstream source first, not continue blindly downstream.

## Trace Artifacts

Every generated non-canonical executable rule should persist a trace:

- source span id,
- upstream candidates considered,
- selected relation,
- rejected alternatives and reason,
- imports or stubs materialized,
- absolute target,
- effective period,
- validation checks that depended on the relation.

This trace is required to debug whether an encoding should have looked further
upstream.

## SNAP Pilot

Use Colorado SNAP as the first end-to-end pilot:

1. Ensure `7 USC 2012`, `2014`, `2017` and needed sibling provisions are encoded.
2. Encode relevant `7 CFR 273` provisions, especially eligibility, allotment,
   deductions, utility allowance delegation, and state options.
3. Encode USDA FY 2026 COLA and income-standard documents as federal policy
   sources that set statutory or regulatory slots.
4. Encode Colorado regulations/manual sections only as:
   - source-relation restatements of federal rules and USDA parameter docs,
   - Colorado-set options or values,
   - Colorado-specific implementation mechanics.
5. Run full CO SNAP calculation tests and compare to PolicyEngine US where a
   matching oracle exists.

The expected cleanup outcome is that Colorado files do not own federal maximum
allotments, federal standard deductions, federal income standards, federal
earned-income deductions, or statutory elderly/disabled definitions.

## Implementation Phases

### Phase 1: Relation-Aware Harness

- Keep RuleSpec source-relation checks in `axiom-encode`.
- Add TDD cases for generic `restates`, `sets`, `amends`, `implements`,
  duplicate concept detection, source-span scoping, and period conflicts.
- Add schema validation for `kind: source_relation`,
  `source_relation.type`, `source_relation.target`, `source_relation.value`,
  and delegation/amendment fields.
- Keep current SNAP fallback checks only as temporary guardrails.

### Phase 2: Corpus Index and RuleSpec Registry

- Require stable `source_span_id`, citation, source type, jurisdiction,
  authority rank, and effective-period metadata in `corpus.provisions`.
- Index deterministic source citations and cross-references from corpus metadata.
- Index validated RuleSpec targets, imports, and `source_relation` records.
- Add upstream candidate queries by citation, absolute target, topic,
  jurisdiction, authority rank, relation type, and effective period.

### Phase 3: Encoder Context Retrieval

- Update eval and encode flows to call upstream registry lookup before model
  invocation.
- Materialize upstream candidates into the workspace or create planned stubs.
- Include upstream candidates in prompt context with their import targets,
  absolute targets, source spans, effective periods, and known RuleSpec
  relation hints.
- Require the model to emit rule-level relation metadata for each rule.
- Fail validation when relation metadata is absent for an artifact that is
  copying, setting, implementing, or amending a known upstream target.

### Phase 4: Planner and Batch Encoding

- Add `axiom-encode plan --json` to produce the ordered DAG.
- Add `axiom-encode encode-plan` to encode sources in dependency order.
- Re-run downstream slices when upstream artifacts are added or changed.
- Persist retrieval and relation-decision traces for each generated rule.

### Phase 5: SNAP End-to-End Closure

- Run the planner over federal SNAP and Colorado SNAP sources.
- Delete SNAP-specific upstream fallback checks once equivalent RuleSpec
  relation checks catch the same failures.
- Validate full CO SNAP calculation parity against PolicyEngine US on ECPS and
  targeted household cases.

## Acceptance Criteria

- A synthetic non-SNAP federal/state fixture catches restatement, delegated
  setting, implementation, amendment, duplicate-definition, and period-conflict
  cases without program-specific rule names.
- A downstream source that restates upstream policy fails validation unless it is
  encoded as a `source_relation.type: restates` record.
- A downstream source that sets a delegated option/value fails validation unless
  it records a `source_relation.type: sets` edge.
- A downstream source that implements delegated mechanics fails validation unless
  it records a `source_relation.type: implements` edge.
- A downstream source that amends upstream policy fails validation unless it
  records a `source_relation.type: amends` edge.
- `axiom-encode plan --json` emits a deterministic DAG with source ids, source
  spans, effective periods, upstream candidates, known relation edges, and
  blocked/stubbed dependencies.
- Every generated non-canonical executable rule has a persisted trace listing
  upstream candidates considered, selected relation, rejected alternatives, and
  source spans.
- Colorado SNAP can be calculated end to end without Colorado owning federal
  statutory definitions or USDA federal parameter values.
- SNAP-specific placement contracts are removed only after equivalent generic
  tests fail when RuleSpec relation metadata is removed.

## Risks

- Citation extraction may create noisy upstream candidates. Only validated
  RuleSpec `source_relation` records should become blocking harness requirements.
- Some source documents genuinely combine restatement and implementation in the
  same paragraph. The encoder must split these into span-scoped source-relation
  restatements and downstream executable rules.
- `sets` and `amends` need clear semantics. A source that updates an annual value
  usually `sets` a delegated slot; a source that changes legal text or formula
  structure `amends`.
- Missing upstream encodings should block promotion or create explicit planned
  stubs, not silently allow downstream duplication.
- Absolute targets must be stable enough to survive human-facing renames. If
  targets are model-generated or name-like, duplicate detection will drift back
  toward heuristics.

## Independent Review Incorporated

An independent review of the first draft identified the main corrections now
reflected here:

- canonicality cannot be inferred from authority rank alone;
- corpus retrieval metadata and RuleSpec source relations need a clean boundary;
- rule metadata needs a concrete per-rule schema;
- duplicate detection needs absolute targets, not only symbol names;
- upstream imports need a materialization contract;
- planned stubs need lifecycle rules;
- upstream lookup must be effective-period aware;
- acceptance criteria need non-SNAP proof cases;
- mixed source documents need span-scoped relations.
