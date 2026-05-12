"""RuleSpec encoder prompt used by generic backend adapters."""

ENCODER_PROMPT = """# Axiom RuleSpec Encoder

Encode only the supplied legal source text into Axiom RuleSpec YAML.

Hard requirements:
- Emit `format: rulespec/v1`.
- Include `module.summary: |-` with the operative source text or an exact audit excerpt.
- If the source has an ingested corpus provision, include
  `module.source_verification.corpus_citation_path` or
  `module.source_verification.corpus_citation_paths`.
- If accepted source claims are supplied, include their IDs under
  `module.source_claims`; do not inline claim bodies, values, formulas,
  evidence, or review metadata in RuleSpec.
- Include `module.proof_validation.required: true` and add
  `metadata.proof.atoms` to each policy-bearing rule. Each proof atom must point
  to direct corpus source text, a claim listed in `module.source_claims`, or an
  explicit imported RuleSpec export. If you cannot build that proof, stop and
  emit a typed request such as `missing_claim`, `bundle_expansion_request`,
  `corpus_defect`, `segmentation_fix`, `stale_claim`, or `conflicting_claims`.
- For imported proof support, put `import:` at the proof atom top level
  (for example `kind: import` plus `import.target: us:statutes/...#symbol`);
  do not put imported RuleSpec targets under `source:`. Import proof atoms must
  include `import.target`, `import.output`, and `import.hash` with a listed
  `sha256:` hash; if no `sha256:` hash is provided, do not emit an import proof
  atom.
- Proof atom `kind` must be one of: `amount`, `condition`, `definition`,
  `default`, `effective_period`, `exception`, `formula`, `import`, `ordering`,
  `parameter`, `parameter_table`, `predicate`, `table_cell`, or `unit`.
- A `kind: table_cell` proof atom must include
  `source.table.header`, `source.table.row`, and `source.table.column`.
  A `kind: parameter_table` proof atom with `source.table` must include
  `source.table.header` and row/column keys. If you cannot identify table
  coordinates, use a direct proof kind such as `amount`, `parameter`, or
  `formula` instead of `table_cell`.
- Do not emit `source_url`; RuleSpec validation reads normalized corpus provisions,
  not raw PDFs or web pages.
- Use `rules:` as a list of rule objects.
- Every executable `parameter` and `derived` rule must include a `source:`
  field with the legal citation/span that directly supports that rule. Keep
  `source:` short and local to the rule; use `module.source_verification` for
  the corpus locator.
- Use `kind: parameter` for source-stated amounts, rates, thresholds, caps, and limits.
- Use `kind: parameter` with `indexed_by` and versioned `values` for source-stated
  numeric tables/scales keyed by household size, family size, income band,
  age band, or another row key. Do not encode those cells as `match` arms or
  numeric literals inside a derived formula.
- Use `kind: derived` for entity-scoped outputs.
- If source text is a broad application, furnishing, administrative duty, or
  purpose clause without a computable policy condition, preserve it in
  `module.summary` but do not create an executable derived output just to
  paraphrase it. Encode only the concrete conditions, exceptions, parameters,
  and relations that affect computation.
- Do not create an output for administrative clauses like "assistance shall be
  furnished to all eligible households who make application." Unless the source
  defines a calculable benefit, amount, condition, or exception, keep that text
  documentary in `module.summary`.
- Do not encode a pure pass-through rule whose formula is only one local fact.
  If the source only names a preexisting fact without changing it, reference
  the upstream rule when available or leave the phrase documentary.
- If the target is an aggregate parent provision and child-fragment files already
  encode subparagraphs, import those child outputs and compose them. Do not
  redefine the child parameters, helper rules, or copied executable outputs in
  the parent file.
- If a copied child-fragment file encodes a limitation, branch, amount, or
  predicate needed by the requested parent provision, import the child output
  and compose it. Do not copy the child formula or its factual inputs into the
  parent file. For example, IRC section 63(c) should import
  `us:statutes/26/63/c/5#dependent_standard_deduction` rather than reconstruct
  the dependent earned-income limitation in `c.yaml`.
- If copied context listings include exported symbols as `import_target#name`,
  use those exact references in `imports:` and proof atoms when composing from
  context.
- In formulas, reference imported exports by their bare local rule name after
  adding an `imports:` entry; never write an absolute `us:...#rule_name` reference inside a formula.
- Do not create standalone small-number parameters just to restate prose such
  as "one-time" or "more than one consecutive month" when the number only
  qualifies a local factual condition. Encode the whole source-stated condition
  as a fact predicate or derived condition unless the scalar is an independent
  reusable amount, rate, threshold, cap, or limit.
- Do not append citation or file suffixes like `_2014_a` to new local rule
  names; the file path is already the legal ID. Keep names concise and
  semantic unless a copied public interface must be preserved.
- Rule names ending in the current path fragments, such as `_2_C`, `_b_1`,
  `_d_2_C`, or `_2014_a`, are invalid.
- If an existing copied output name violates the no-citation/path-suffix rule,
  do not preserve it. Rename it to a concise semantic name and update the
  companion tests.
- Rule names must not collide with copied sibling files. For subparagraph/list
  item child files, make the principal output name semantic to that branch
  (for example `care_responsibility_exemption_applies`), not only the shared
  parent consequence like `person_exempt_from_paragraph_1_work_requirements`.
- When a child provision substitutes, increases, caps, or otherwise modifies a
  sibling or parent output, give the replacement a branch-specific name such as
  `_under_subsection_h`, `_after_2017`, or another source-stated modifier. For
  IRC section 24(h), do not reuse sibling 24(d) names like
  `ctc_refundable_phase_in_threshold`; use a subsection-h-specific name such as
  `ctc_refundable_phase_in_threshold_under_subsection_h`.
- For subparagraph/list item child files, do not preserve an existing copied
  target name that mainly describes the shared parent outcome rather than this
  branch's source-stated condition; treat it as stale and rename it.
- Choose structural relations at the narrow legal subject stated by the source.
  If the source grants an amount to the taxpayer, spouse, claimant, child, or
  other role-limited person, do not aggregate over a broader household/tax-unit
  relation unless the source says every member counts. Name the relation for the
  role set that is legally counted, such as `taxpayer_or_spouse`, not merely for
  the container entity. If a copied relation is legally too broad for the
  requested source, rename it; relation names are not stable public outputs.
  Never preserve or create `*_member_of_tax_unit` or `member_of_tax_unit` for a
  source that counts only the taxpayer, spouse, qualified individual, claimant,
  child, or dependent. For IRC section 22, count qualified individuals over a
  relation like `taxpayer_or_spouse_of_tax_unit`, not
  `elderly_disabled_member_of_tax_unit`.
- For child tax credit, dependent credit, or any source that says "qualifying
  child", "dependent of the taxpayer", or "with respect to such child", do not
  use `member_of_tax_unit`. Define a role-scoped relation such as
  `dependent_of_tax_unit`, `qualifying_child_of_tax_unit`, or
  `child_or_dependent_of_tax_unit`, and aggregate over that relation. For IRC
  section 24(h), count `ctc_qualifying_child` and `ctc_other_dependent` over a
  dependent/child relation, not over `member_of_tax_unit`.
- If the source computes an amount by reference to an entitlement, status,
  amount, or test "under" another section, subsection, paragraph, regulation, or
  document, do not inline that cross-reference's mechanics into this file unless
  that cross-referenced source text is included and this file is the canonical
  home for those mechanics. Import the existing RuleSpec target when present. If
  the cross-reference is not yet encoded, expose a semantic input/count named
  for the cross-reference itself, such as
  `additional_standard_deduction_entitlement_count_under_subsection_f`, rather
  than inventing the cross-referenced age, blindness, household, or membership
  tests locally. For example, IRC section 63(c)(3) should not count
  `is_aged_65_or_over` or `is_blind` over `member_of_tax_unit`; those are
  subsection 63(f) mechanics.
- When an unencoded cross-reference must be represented as a semantic local
  input, name it after the legal status with an `_under_section_<section>` or
  `_under_subsection_<subsection>` suffix. Do not start a local input with
  `section_<section>_` or `subsection_<subsection>_`; those names are reserved
  for imported legal outputs and will be treated as missing imports.
- When a copied context file encodes a cited upstream source on a different
  entity, import that upstream output and bridge entities with a structural
  relation instead of replacing the import with a local cross-reference amount.
  For example, if IRC section 22 excludes amounts described in section
  104(a)(4), import
  `us:statutes/26/104/a/4#service_injury_pension_excluded_amount` and aggregate
  it over a TaxUnit-to-Payment relation; do not create local inputs named
  `section_104_a_4_amounts` or `section_104_a_4_veterans_affairs_benefits`.
- Use `dtype: Judgment`, not `dtype: Boolean`, for legal eligibility,
  availability, applicability, entitlement, and other holds/not-holds style
  outputs, especially when the formula contains `not`.
- Do not create derived `dtype: Boolean` helper rules with logical formulas.
  Use `dtype: Judgment` for derived legal predicates, or leave simple local
  facts as factual `#input.<fact>` keys consumed by formulas and tests.
- Use `kind: data_relation` for executable runtime predicates such as
  `member_of_household`. Put arity under `data_relation.arity`.
- Do not encode simple unary factual inputs as `kind: data_relation` rules. If
  a formula needs a local true/false fact, reference a descriptive bare fact
  name in the formula and put that fact in tests as
  `<jurisdiction>:<repo-path>#input.<fact>`.
- If the requested source text includes a limitation, cap, exception, or
  cross-referenced subparagraph that changes the final exported amount, the
  final exported amount must apply that limitation. If a copied sibling/context
  file already encodes the limitation, import it and compose with it instead of
  duplicating or ignoring it.
- Do not create parallel statutory-dollar executable parameters when a copied
  current-year authority already provides the applicable inflation-adjusted
  parameter. Import the current-year authority unless the task is to encode the
  inflation adjustment formula itself.
- Use `kind: source_relation` for non-executable legal/provenance edges such as
  `restates`, `sets`, `amends`, `implements`, `delegates`, `defines`, or
  `cites`. It must include `source_relation.type` and
  `source_relation.target`, and it must not include executable `versions`.
- Do not put source graph relationships in executable rule metadata. If a
  source `sets`, `amends`, `implements`, or `restates` another source, encode a
  separate `kind: source_relation` record in the same RuleSpec file.
- If supplied context for this target or a same-program sibling contains a
  `kind: source_relation` record, preserve that legal/provenance edge unless
  the source text proves it wrong; executable formula changes are not a reason
  to drop source graph context.
- For state-set standards, allowances, thresholds, or options implementing
  federal delegation, include `source_relation.value` pointing to the local
  executable RuleSpec output and `source_relation.basis.delegation` when context
  identifies the upstream delegated slot.
- Emit only RuleSpec YAML; use `.test.yaml` companions when tests are requested.
- In `.test.yaml` companions, every `input:` and `output:` key must be a
  canonical legal RuleSpec reference that resolves to an actual file and
  fragment. Use `<jurisdiction>:<repo-path>#input.<fact>` for fact inputs
  consumed by that file, `<jurisdiction>:<repo-path>#relation.<name>` for
  relation inputs, and `<jurisdiction>:<repo-path>#<rule_or_parameter>` for
  executable outputs or imported legal values. Never use bare friendly keys.
- Every local executable `kind: parameter` and `kind: derived` rule must appear
  at least once under an `output:` block in the companion `.test.yaml`; do not
  leave scalar parameters, helper parameters, or helper derived rules
  unasserted.
- Each `.test.yaml` case may assert derived outputs for only one entity type. If
  a module defines both `Person` and `TaxUnit` outputs, create separate cases:
  `Person` cases set person facts at the top level and assert person outputs;
  `TaxUnit` cases use relation rows to supply person facts and assert only
  tax-unit outputs. Do not assert relation-child outputs in the parent entity's
  case.
- If a test needs an imported derived output to become true or false, mirror the
  copied companion test `input:` pattern. Usually this means setting the
  imported file's underlying `#input.<fact>` and `#relation.<name>` keys, not
  shortcutting by setting the imported derived output itself. Only set an
  imported derived key in `input:` when a copied companion test also uses that
  exact derived key in `input:`.
- Never turn an imported derived rule into a fabricated `#input.<same_rule_name>`
  key. For example, use
  `us:statutes/7/2012/j#snap_household_has_elderly_or_disabled_member: holds`
  or `not_holds`, not
  `us:statutes/7/2012/j#input.snap_household_has_elderly_or_disabled_member`.
- Do not invent `#input` keys for imported files. Use only the bare fact names
  that the imported file's formulas actually reference, or mirror the imported
  file's companion `.test.yaml` input pattern when it is supplied in context. If
  that imported output is driven by an upstream structural relation, set the
  upstream `#relation.<name>` rows used by the companion test instead of
  creating a local input under the imported file.
- Use `holds` and `not_holds` for actual `dtype: Judgment` rule keys in test
  inputs and outputs; do not use YAML booleans for Judgment rule values.
- Use YAML booleans `true` and `false` for local factual `#input.<fact>` keys
  referenced directly by formulas.
- Every test case for a local derived formula must assign every local factual
  `#input.<fact>` referenced by that formula, including facts that are false in
  the case. Missing false inputs make the executable test invalid.
- For every encoded `except`, `unless`, or `notwithstanding` carve-out, include
  companion tests for the positive path and the carve-out path so exclusions
  cannot be silently dropped.
- If a formula negates multiple exception predicates, include a separate
  companion test for each predicate that sets that exception input true and
  expects the directly affected Judgment rule to be `not_holds`.
- Do not collapse a list of cited exceptions or cross-reference carve-outs into
  one aggregate fact such as `sections_..._do_not_preclude...`. Encode or
  import each cited exception separately, then combine them in a helper if
  useful.
- When an exception, exclusion, `unless`, or `notwithstanding` clause cites
  another legal section or same-section subsection, do not create a local
  `section_...` or `subsection_...` placeholder input for that cited source.
  Import the cited RuleSpec source when it exists; if that upstream source is
  required but unavailable, stop with a missing-upstream/dependency request
  rather than encoding an opaque local fact.
- If the cited same-section subsection is supplied in context as a RuleSpec
  file, add an `imports:` entry for that file and reference its exported rule;
  do not summarize the cited subsection into a local fact like
  `person_meets_...requirements`.
- Do not copy the body of a cited cross-reference provision into this module's
  `summary` or re-encode that cited provision locally. Keep this module scoped
  to the requested citation and import the cited provision instead.
- If context files import the target file or reference target outputs, preserve
  the target file's public output names unless the source text proves the old
  interface was legally wrong or the name violates the no-citation/path-suffix
  rule. Do not rename an exported value just because a clearer friendly name is
  possible.
- Do not emit Python code, markdown fences, prose, or file-write confirmations.
- Do not invent values or ontology beyond the source text.
- When source text uses amendment markup like `[old] new`, treat the bracketed
  value as superseded text. Encode the current unbracketed value/effective date
  unless the task explicitly asks for historical text.
- If a source makes an allowance, deduction, exemption, or eligibility branch
  conditional on billed, paid, incurred, anticipated, or other cost/expense
  facts, encode a positive fact predicate for that source-stated condition.
  Do not model availability solely as `not` other categories. If the condition
  lives in a parent paragraph needed to understand a child paragraph, include
  the parent corpus path in `module.source_verification.corpus_citation_paths`.
- When the cost/expense fact only matters after exclusion predicates, exported
  amount/quantity formulas consumed by dependent modules must guard the
  exclusions before referencing the branch-specific fact, so excluded cases do
  not require that fact as an input. For example, the amount should use
  `if other_allowance_eligible: 0 else: if household_has_telephone_cost: amount else: 0`
  rather than `if telephone_eligible: amount else: 0` when `telephone_eligible`
  itself references the branch-specific telephone-cost input.
- Phrases like `consists of the cost for X` or `available to households with X
  costs` require a positive fact for that cost/service. For example, a telephone
  allowance must depend on a fact for the household having or incurring the
  basic telephone-service cost before applying exclusions for other allowances.
- In a jurisdiction-specific repo, phrases like `residing in New York State`
  usually describe the document's scope, not a new input variable. Do not add a
  state-residency input unless the provision itself is encoding a residency
  eligibility test.
- Put formulas under `versions: - effective_from: 'YYYY-MM-DD'` and `formula: |-`.
- Do not emit more than one `versions:` entry for `kind: derived`; the runtime
  does not yet support period-selecting versioned formulas. Use a single
  source-faithful conditional formula when the provision itself defines a
  temporal branch, or encode only the currently applicable provision after
  resolving the source context.
- Formula strings use Axiom formula syntax: `if condition: value else: other`, `==`, `and`, and `or`.
- Supported scalar functions are `min(...)`, `max(...)`, `floor(x)`, and
  `ceil(x)`. Do not use Python-only functions such as `round(...)`; express
  nearest-multiple rounding as `floor((x / multiple) + 0.5) * multiple` for
  nonnegative amounts.
- Supported relation aggregators are `len(relation)`,
  `count_where(relation, predicate_fact)`, `sum(relation.amount_fact)`, and
  `sum_where(relation, amount_fact_or_derived, predicate_fact)`. Do not write
  `sum(relation, expression)` or put arithmetic inside a relation field access.
  To count two boolean conditions over the same relation, write two
  `count_where(...)` calls and add them.
- If a conditional is embedded inside arithmetic or another larger expression,
  wrap the whole conditional in parentheses, such as
  `amount + (if condition: extra else: 0)`. Do not write
  `amount + if condition: extra else: 0`.
- Formula strings must use bare identifiers only. If an imported rule is listed
  as `us:statutes/...#example_rule`, add that exact target to `imports:` but
  reference `example_rule` inside formula text.
- Axiom conditionals are expression syntax, not YAML syntax. Money/scalar
  formulas may use `if condition: value else: other`; do not use Python ternary
  syntax. Judgment formulas should usually be boolean expressions, not `if`
  conditionals.
- When using negated conjuncts, write them as a multiline formula with each
  `not <predicate>` term on its own line joined by `and`, rather than one
  compact `not A and not B` line.
- Formula strings reference indexed parameter tables with `table_name[index_expr]`.
- Every substantive numeric literal must be grounded in the supplied source text unless it is -1, 0, 1, 2, or 3.
- Every substantive numeric occurrence in `./source.txt` must be represented by
  a named scalar definition when it is a legal amount, rate, threshold, cap, or
  limit.
- If the same numeric value appears in separate numbered exceptions,
  subparagraphs, or otherwise materially different legal roles, give those roles
  distinct named scalars; reuse a named scalar only for the same legal role.
- Adjacent bracket thresholds repeated as both an upper bound and the next
  bracket's lower bound are separate source-stated legal roles; define distinct
  semantic scalars for those occurrences and use them in the branch conditions.
- Before finalizing, do this self-check:
  1. Numeric inventory: every source-stated legal amount, rate, threshold, cap,
     or limit has a named `parameter`, and derived formulas reference the name
     rather than an inline literal.
  2. Test input inventory: for every local factual identifier referenced by a
     local derived formula, every companion test case assigns the corresponding
     `#input.<fact>` explicitly, including false facts. Do not rely on implicit
     defaults.
  3. Proof inventory: every proof atom uses only an allowed `kind`; imported
     proof atoms include `import.target`, `import.output`, and `import.hash`;
     textual claim support is either direct corpus source support or a claim ID
     listed under `module.source_claims`.
  4. Import inventory: every `imports:` entry is an exact copied/importable
     RuleSpec target. Do not guess sibling paths; if required upstream context is
     missing, emit a typed missing-upstream/dependency request instead.

Minimal shape:

format: rulespec/v1
module:
  proof_validation:
    required: true
  summary: |-
    <source text>
rules:
  - name: example_amount
    kind: parameter
    dtype: Money
    unit: USD
    source: <legal citation/span>
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              corpus_citation_path: <corpus path>
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          451

Indexed table shape:

rules:
  - name: example_amount_by_household_size
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: household_size
    versions:
      - effective_from: '2025-10-01'
        values:
          1: 298
          2: 546
  - name: example_amount
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    source: <legal citation/span>
    versions:
      - effective_from: '2025-10-01'
        formula: example_amount_by_household_size[household_size]
"""


def get_encoder_prompt(
    citation: str,
    output_path: str,
    corpus_citation_path: str | None = None,
) -> str:
    """Return a complete RuleSpec task prompt for a source unit."""
    corpus_section = ""
    if corpus_citation_path:
        corpus_section = f"""
Corpus source path: {corpus_citation_path}
Include `module.source_verification.corpus_citation_path: {corpus_citation_path}` exactly.
"""

    return f"""{ENCODER_PROMPT}

Target citation/source id: {citation}
Expected output path: {output_path}
{corpus_section}

Return only raw RuleSpec YAML for that path.
"""
