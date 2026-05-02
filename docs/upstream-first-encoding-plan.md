# Upstream-First Encoding Plan

## Goal

Axiom should encode policy in legal-authority order so each downstream source can
distinguish a genuinely new rule from a restatement, delegation, option setting,
or amendment of upstream law. The encoder should not rely on program-specific
variable-name heuristics to make that decision. It should use a source graph
built from ingestion metadata, source-span relations, and already validated
RuleSpec artifacts.

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
`metadata.sets` pointing to the upstream slot.

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
force every lower-rank document into a reiteration when that document is
lawfully setting, implementing, or amending an upstream slot.

## Source Graph

Ingestion should write every source slice to `corpus.provisions` with structured
metadata:

- `citation_path`: stable Axiom source path.
- `source_span_id`: stable id for paragraph/table/row/subsection spans that can
  carry distinct relations.
- `jurisdiction`: `US`, `CO`, `TN`, etc.
- `source_type`: statute, regulation, agency_policy, manual, notice, form, etc.
- `authority_level`: comparable rank for default ordering.
- `effective_start` and `effective_end` where available.
- `source_url` and source storage pointer.
- `concept_id`: stable concept key when known.
- `relations`: structured links to source spans, citation paths, concept ids, or
  RuleSpec targets.

The graph is two-stage:

1. **Source graph:** ingestion stores source-source and citation-level relations
   before any RuleSpec exists.
2. **RuleSpec registry:** after validation, RuleSpec artifacts register symbols,
   relation metadata, concept ids, and targets. Provisional targets are then
   reconciled to concrete RuleSpec symbols by citation, source span, concept id,
   and symbol.

Core relation types:

- `defines`: the source/span is the canonical home for an executable concept.
- `delegates`: the source/span creates a downstream slot for another source to
  set or implement.
- `implements`: the source/span supplies mechanics delegated by upstream law.
- `sets`: the source/span sets a delegated value, option, rate, region, or
  parameter.
- `amends`: the source/span changes an upstream source, slot, or target.
- `reiterates`: the source/span restates an upstream source or target without
  adding executable semantics.
- `cites`: the source/span references another source without resolving the legal
  relationship.

Relations can attach to spans, not only whole files. A single source slice can
produce multiple RuleSpec records: one reiteration, one delegated setting, and
one downstream implementation rule, each tied to the source span that justifies
it.

## Concept Keys

Duplicate detection must not compare only RuleSpec symbol names. Every canonical
target should have a `concept_id` derived from:

- source citation path,
- source span id,
- legal topic or slot,
- jurisdiction where relevant,
- effective-period scope where relevant.

Rule names are still useful for humans and imports, but duplicate detection
should compare concept ids and declared relation targets. This is what prevents
the harness from falling back into variable-name heuristics.

## Rule Identity and Outputs

Bare rule names are local symbols, not durable public identifiers. The durable
identity of an executable rule should be the canonical RuleSpec target plus the
rule name:

```text
rules-us:statutes/7/2017/a#snap_regular_month_allotment
rules-us-co:regulations/10-ccr-2506-1/4.403.2#snap_countable_earned_income
```

Formula authors should still be able to reference local and imported symbols by
bare name where the compiler can resolve them unambiguously. External surfaces
should expose the canonical id alongside the friendly name. That includes API
execution results, notebook demos, traces, registry rows, and verification
artifacts. A notebook can display `snap_regular_month_allotment`, but the
machine-readable output should also carry
`rules-us:statutes/7/2017/a#snap_regular_month_allotment` so calculated values
are keyed to the provision that defines them.

## Encoding Workflow

For each source span:

1. Resolve source metadata and graph neighborhood.
2. Fetch period-overlapping upstream candidates in authority order.
3. Materialize all graph-resolved import targets into the eval workspace, or
   create explicit planned stubs before model invocation.
4. Fetch sibling and downstream context only after upstream context.
5. Ask the encoder to classify each generated rule as one of:
   `defines`, `delegates`, `implements`, `sets`, `amends`, or `reiterates`.
6. Generate RuleSpec that records the classification in per-rule metadata.
7. Validate the artifact against the graph, local imports, source text, tests,
   and applicable oracles.
8. Persist new RuleSpec target metadata and trace data back into the registry so
   later downstream sources can discover it.

The encoder prompt should forbid executable duplication when a source is only
restating upstream law. It should also require an explicit justification when it
defines a new executable rule in a lower-authority source despite upstream
candidates existing.

## RuleSpec Metadata

Rule-level metadata should be explicit and structured. Proposed shape:

```yaml
rules:
  - name: co_snap_standard_utility_allowance
    kind: parameter
    dtype: Money
    unit: USD
    metadata:
      source_relation: sets
      concept_id: us.snap.standard_utility_allowance.co
      source_span_id: corpus.provisions:co/snap/manual/utility-allowance#table-1
      sets:
        - target: us:regulation/7-cfr/273/9/d/6/iii#snap_standard_utility_allowance_slot
          source_relation_id: corpus.relations:...
    versions:
      - effective_from: '2025-10-01'
        formula: '451'
```

Recommended metadata fields:

- `source_relation`: one of `defines`, `delegates`, `implements`, `sets`,
  `amends`, or `reiterates`.
- `concept_id`: stable concept key.
- `source_span_id`: source span that justifies the rule.
- `defines`: list of target/concept objects when a rule is canonical.
- `delegates`: list of downstream slot targets created by this rule.
- `implements`: list of upstream targets implemented by this rule.
- `sets`: list of delegated targets set by this rule.
- `amends`: list of upstream targets amended by this rule.

For pure restatements, keep the existing explicit non-executable shape:

```yaml
rules:
  - name: co_snap_standard_deduction_reiterates_usda_fy_2026
    kind: reiteration
    reiterates:
      target: us:policies/usda/snap/fy-2026-cola/deductions#snap_standard_deduction
      authority: federal
      relationship: restates
```

The registry should index all RuleSpec relations so a downstream run can ask:

- What upstream source spans and RuleSpec targets exist for this citation/topic?
- What concept ids are already defined?
- Which downstream sources set or amend this target?
- Which sources merely reiterate this target?

## Harness Rules

The harness should enforce source relations, not policy-specific variable names.

Required deterministic checks:

- If source metadata says `reiterates` or `restates` a target, the file must use
  `kind: reiteration` with `reiterates.target`, scoped to the relevant source
  span.
- If source metadata says `sets` a target, the corresponding RuleSpec rule must
  declare `metadata.source_relation: sets` and `metadata.sets`.
- If source metadata says `amends` a target, the corresponding RuleSpec rule
  must declare `metadata.source_relation: amends` and `metadata.amends`.
- If source metadata says `implements` a target, the corresponding RuleSpec rule
  must declare `metadata.source_relation: implements` and `metadata.implements`.
- If a rule defines a concept id already defined upstream and does not declare
  `implements`, `sets`, `amends`, or a reiteration, flag it as downstream
  duplication.
- If a downstream source cites an upstream provision and the graph resolves a
  unique target, require a materialized import, a relation marker, or an explicit
  non-import justification.
- If a source sets or amends an upstream target, verify that the target exists
  unless the run is explicitly creating a planned stub.

Program-specific checks can exist temporarily as fallback assertions, but they
should be deleted once graph metadata covers the relevant relation.

## Materialization Contract

The model should never be asked to import an upstream target that was not copied
or stubbed into its workspace.

For each graph-resolved upstream target before model invocation:

1. If the RuleSpec artifact exists, copy it into context and expose its canonical
   import path.
2. If the source exists but is not encoded, block downstream promotion and queue
   the upstream source first.
3. If a temporary stub is unavoidable, create a planned stub with lifecycle
   metadata.

Planned stubs must include:

- target citation or concept id,
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
- Tests should use periods covered by the selected upstream relations.

This is essential for USDA annual COLA documents, state option updates, and
regulatory amendments.

## Planner

Add an encoding planner that creates a deterministic DAG from
`corpus.provisions`:

1. Group provisions by jurisdiction, program, source type, and concept topic.
2. Sort by `authority_level`, citation hierarchy, and effective period.
3. Add graph edges for `defines`, `delegates`, `implements`, `sets`, `amends`,
   `reiterates`, and high-confidence `cites`.
4. Encode missing upstream dependencies before downstream slices.
5. Requeue downstream slices when a newly encoded upstream artifact changes the
   candidate context.
6. Emit `axiom-encode plan --json` with source ids, span ids, relation edges,
   effective periods, upstream candidates, blocked dependencies, and planned
   stubs.

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
- concept id,
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
   - reiterations of federal rules and USDA parameter docs,
   - Colorado-set options or values,
   - Colorado-specific implementation mechanics.
5. Run full CO SNAP calculation tests and compare to PolicyEngine US where a
   matching oracle exists.

The expected cleanup outcome is that Colorado files do not own federal maximum
allotments, federal standard deductions, federal income standards, federal
earned-income deductions, or statutory elderly/disabled definitions.

## Implementation Phases

### Phase 1: Relation-Aware Harness

- Keep source-metadata relation checks in `axiom-encode`.
- Add TDD cases for generic `reiterates`, `sets`, `amends`, `implements`,
  duplicate concept detection, source-span scoping, and period conflicts.
- Add schema validation for `metadata.source_relation`, `metadata.defines`,
  `metadata.delegates`, `metadata.implements`, `metadata.sets`, and
  `metadata.amends`.
- Keep current SNAP fallback checks only as temporary guardrails.

### Phase 2: Source Graph Ingestion

- Require sidecar metadata for source slices or spans that are known
  restatements, delegated settings, amendments, or implementation documents.
- Populate `corpus.provisions.relations` from source-side metadata and citation
  extraction.
- Add concept ids and source span ids.
- Add graph queries for upstream candidates by citation, concept id, topic,
  jurisdiction, authority rank, and effective period.

### Phase 3: Encoder Context Retrieval

- Update eval and encode flows to call upstream graph lookup before model
  invocation.
- Materialize upstream candidates into the workspace or create planned stubs.
- Include upstream candidates in prompt context with their import targets,
  concept ids, source spans, effective periods, and relation hints.
- Require the model to emit rule-level relation metadata for each rule.
- Fail validation when relation metadata is absent for source metadata that
  requires it.

### Phase 4: Planner and Batch Encoding

- Add `axiom-encode plan --json` to produce the ordered DAG.
- Add `axiom-encode encode-plan` to encode sources in dependency order.
- Re-run downstream slices when upstream artifacts are added or changed.
- Persist relation traces for each generated rule.

### Phase 5: SNAP End-to-End Closure

- Run the planner over federal SNAP and Colorado SNAP sources.
- Delete SNAP-specific upstream fallback checks once equivalent source metadata
  and graph checks catch the same failures.
- Validate full CO SNAP calculation parity against PolicyEngine US on eCPS and
  targeted household cases.

## Acceptance Criteria

- A synthetic non-SNAP federal/state fixture catches restatement, delegated
  setting, implementation, amendment, duplicate-definition, and period-conflict
  cases without program-specific rule names.
- A downstream source that restates upstream policy fails validation unless it is
  encoded as a reiteration.
- A downstream source that sets a delegated option/value fails validation unless
  it records `metadata.source_relation: sets` and `metadata.sets`.
- A downstream source that implements delegated mechanics fails validation unless
  it records `metadata.source_relation: implements` and `metadata.implements`.
- A downstream source that amends upstream policy fails validation unless it
  records `metadata.source_relation: amends` and `metadata.amends`.
- `axiom-encode plan --json` emits a deterministic DAG with source ids, source
  spans, relation edges, effective periods, upstream candidates, and
  blocked/stubbed dependencies.
- Every generated non-canonical executable rule has a persisted trace listing
  upstream candidates considered, selected relation, rejected alternatives, and
  source spans.
- Colorado SNAP can be calculated end to end without Colorado owning federal
  statutory definitions or USDA federal parameter values.
- SNAP-specific placement contracts are removed only after equivalent generic
  tests fail when graph metadata is removed.

## Risks

- Citation extraction may create noisy `cites` edges. Only high-confidence
  relations should become blocking harness requirements.
- Some source documents genuinely combine restatement and implementation in the
  same paragraph. The encoder must split these into span-scoped reiterations and
  downstream executable rules.
- `sets` and `amends` need clear semantics. A source that updates an annual value
  usually `sets` a delegated slot; a source that changes legal text or formula
  structure `amends`.
- Missing upstream encodings should block promotion or create explicit planned
  stubs, not silently allow downstream duplication.
- Concept ids must be stable enough to survive renames. If concept ids are too
  model-generated or too name-like, duplicate detection will drift back toward
  heuristics.

## Independent Review Incorporated

An independent review of the first draft identified the main corrections now
reflected here:

- canonicality cannot be inferred from authority rank alone;
- source graph relations and RuleSpec target relations need a two-stage
  reconciliation model;
- rule metadata needs a concrete per-rule schema;
- duplicate detection needs concept ids, not only symbol names;
- upstream imports need a materialization contract;
- planned stubs need lifecycle rules;
- upstream lookup must be effective-period aware;
- acceptance criteria need non-SNAP proof cases;
- mixed source documents need span-scoped relations.
