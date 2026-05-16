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
  atom. When the imported proof target is in the same RuleSpec file, use
  `hash: sha256:local` instead of the file's current content hash.
- Proof atom `kind` must be one of: `amount`, `condition`, `definition`,
  `default`, `effective_period`, `exception`, `formula`, `import`, `ordering`,
  `parameter`, `parameter_table`, `predicate`, `table_cell`, or `unit`.
- A `kind: table_cell` proof atom must include
  `source.table.header`, `source.table.row`, and `source.table.column`.
  A `kind: parameter_table` proof atom with `source.table` must include
  `source.table.header`, `source.table.row_key`, and
  `source.table.column_key`. Header-only `parameter_table` proof atoms are
  invalid. For example:
  `source: {table: {header: "credit percentage table", row_key: "qualifying_child_count", column_key: "credit_percentage"}}`.
  If you cannot identify table coordinates, use a direct proof kind such as
  `amount`, `parameter`, or `formula` instead of `table_cell` or
  `parameter_table`.
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
  the parent file. This includes source-stated numeric values: if a child
  fragment already exports a threshold, amount, rate, cap, or limit, the parent
  formula must reference the imported child output by name rather than copying
  the child literal, even when the parent source excerpt includes the child
  subsection text.
- If an existing copied target file already has executable `parameter`,
  `derived`, or `data_relation` rules, do not replace it with
  `module.status: deferred`, `module.status: entity_not_supported`, or
  `rules: []`. Preserve the executable scope and make the smallest
  source-faithful repair.
- If an existing copied target file is present, treat it as the baseline to
  repair. Preserve existing rule names, imports, companion-test output keys,
  and public formulas unless `./source.txt` proves a specific rule legally
  wrong. Do not rewrite the whole file or rename established outputs just to
  improve style.
- For every retained executable output in an existing copied target file,
  preserve its public executable surface: local `name`, `kind`, `entity`,
  `dtype`, `period`, `unit`, `indexed_by`, and every existing
  `versions[].effective_from`. Do not change `Employer` to `Business`,
  `TaxUnit` to another entity, or alter period/unit/indexing just to match a
  preferred modeling style. Changing an existing `dtype: Money` output to
  `dtype: Judgment`, or vice versa, is a forbidden public-surface migration.
- Preserve the existing factual input surface used by copied executable
  formulas and companion tests. Do not replace established local inputs such as
  `long_term_capital_gains` or `qualified_dividend_income` with newly invented
  upstream-sounding input names unless the task is an explicit source-grounded
  migration that also updates downstream tests, imports, and oracle mappings.
- If a copied child-fragment file encodes a limitation, branch, amount, or
  predicate needed by the requested parent provision, import the child output
  and compose it. Do not copy the child formula or its factual inputs into the
  parent file. For example, IRC section 63(c) should import
  `us:statutes/26/63/c/5#dependent_standard_deduction` rather than reconstruct
  the dependent earned-income limitation in `c.yaml`. Importing one output from
  a child file does not make it okay to copy another child output as a raw
  literal; import each child output you use.
- Importing a child rate or threshold is not enough when the child file already
  exports the executable tax, benefit, deduction, or eligibility result. For
  aggregate parent sections, import the child result output itself and sum,
  cap, select, or otherwise compose those imported results. Do not recompute a
  child result locally from the child rate and the child factual inputs.
- If the requested provision adjusts, indexes, caps, phases out, or otherwise
  modifies "the $X amount in" a child subsection or paragraph, do not create a
  local `*_base_for_inflation` or similar parameter with that child literal.
  Import the child output that defines the referenced amount and use that
  imported output as the base in the adjustment formula. For IRC section 24(i),
  import the subsection 24(h) child amount and refundable-cap outputs rather
  than defining local `1400` or `2200` base parameters in `24.yaml`.
- When source text says an exemption, exclusion, or adjustment applies
  `to the extent` of an amount, do not model it as all-or-nothing zeroing such as
  `if exempt_amount > 0: 0 else: tax`. Subtract or apportion the stated amount.
  If imported child calculations cannot receive the adjusted basis faithfully
  under the current executable schema, emit `module.status: entity_not_supported`
  or `deferred` instead of an approximate executable formula.
- Do not repair that case by importing child rates or thresholds and rebuilding
  the child branch locally with an adjusted basis. That still re-encodes the
  child branch and is invalid unless the schema can explicitly wire the
  adjusted basis into the imported child result.
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
- If a generic role-scoped relation name is already exported by a copied
  sibling file, do not reuse it. Make the relation source-specific, such as
  `ctc_qualifying_child_of_tax_unit` for section 24 rather than a sibling's
  `qualifying_child_of_tax_unit`.
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
  Cross-reference local inputs such as `_under_section_<section>`,
  `_provided_in_section_<section>`, `_allowed_under_section_<section>`,
  `_deduction_under_section_<section>`, or `_credit_allowed_under_section_<section>`
  are only allowed for non-exception factual interfaces when the cited source is
  not available as RuleSpec. If the citation appears in definition,
  same-meaning, treated-as, rules-similar, exception, exclusion, `unless`,
  `notwithstanding`, shall-not-apply, or not-treated-as logic and the cited
  source is unavailable, emit `module.status: deferred` or
  `module.status: entity_not_supported` with `rules: []` instead of inventing a
  local cross-reference fact. If that section is present in repo context, import
  it and use its exported output instead.
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
- If the copied target file is already executable, do not replace it with
  `module.status: deferred` merely because upstream cross-references are not
  fully encoded yet. Preserve the executable public surface and improve the
  source-faithful formulas/tests; only defer an already executable target when
  the existing executable surface is legally impossible to preserve.
- If the requested source itself defines a legal status or test through
  relationship, age, abode/residence, support, filing, income, or tie-breaker
  conditions, encode those conditions as executable predicates with boundary
  inputs for facts not defined in the source. Do not emit
  `module.status: deferred` merely because some facts must be supplied by the
  caller or because tie-breaker facts require relation inputs. For example, IRC
  section 152(c) should export qualifying-child predicates over a
  taxpayer-child relation with inputs for relationship, abode, age/student or
  disability, support, joint-return, and competing-claimant facts.
- If the requested source defines an exclusion, inclusion, deduction, or credit
  amount but depends on externally determined facts such as executive-order
  designations, military status, hospitalization, missing status, monthly pay
  grades, or source-document classifications, encode the amount with boundary
  inputs for those facts instead of deferring. For example, IRC section 112
  should export an executable amount excluded from gross income by reason of
  section 112; it should not be `module.status: deferred` solely because combat
  zone designation or military pay facts are supplied by the caller.
- Hard requirement for IRC section 112: do not emit `module.status: deferred`.
  Export `amount_excluded_from_gross_income_by_reason_of_section_112` as an
  executable TaxUnit/Year Money output, using boundary inputs for qualifying annual
  compensation, commissioned-officer status, maximum enlisted amount,
  combat-zone service, hospitalization, and Vietnam missing-status facts.
  Do not create Person helper outputs or a `data_relation` aggregate for this
  Section 112 encoding; use direct TaxUnit/Year annual input amounts and
  conditions so downstream tax-unit rules can import the result directly.
- Do not create parallel statutory-dollar executable parameters when a copied
  current-year authority already provides the applicable inflation-adjusted
  parameter. Import the current-year authority unless the task is to encode the
  inflation adjustment formula itself.
- If a copied current-year authority exports the same concept or output name
  that the requested statute formula would otherwise create, do not emit a
  local executable duplicate with that name. Import and use the current-year
  authority's output, keeping only statute-specific conditions or non-executable
  `source_relation` records in the statute file. For IRC section 63(c)(5), if
  Rev. Proc. context already exports `dependent_standard_deduction_limit`, do
  not recreate it in the statute file.
- If a current-year authority provides a directly rounded final amount table,
  use that table for the final amount instead of recomputing the amount from
  related rates and thresholds. For example, if an IRS revenue procedure exports
  an EITC maximum-credit table, `eitc_maximum` must select that imported maximum
  table, not multiply the phase-in rate by the earned-income amount and keep an
  unrounded decimal.
- When IRC section 32(c)(2) uses "net earnings from self-employment (within
  the meaning of section 1402(a))" and then says those net earnings are
  determined with regard to Section 164(f), do not import Section 1402(a)'s
  final `net_earnings_from_self_employment` output. Section 1402(a)(12)
  substitutes a rate-based deduction in lieu of Section 164(f). For Section
  32(c)(2), create a local self-employment component from Section 1402(a)'s
  pre-paragraph-12 net earnings minus the imported Section 164(f) deduction.
- When the statute states pre-inflation base dollars that a current-year
  authority adjusts, any local statute output must be named as a statutory/base
  concept, not as the current-year value. For IRC section 63(c)(5), use a name
  like `dependent_basic_standard_deduction_statutory_limit`, not
  `dependent_standard_deduction_limit`.
- When the source rounds an inflation or cost-of-living increase, round the
  increase before adding it to the base amount unless the source explicitly
  says to round the final total. Companion tests must assert the rounded
  increase plus the base, not the unrounded total. For example, with base
  15750, adjustment 0.1, and a next-lower $50 multiple, the increase is 1550
  and the total is 17300, not 17325.
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
- Top-level `imports:` entries must be scalar strings, never map entries like
  `- target:` plus `symbols:`. Import a copied export as one exact string such
  as `us:statutes/26/45A/a#base_year_1993_indian_employment_costs`.
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
- A `#relation.<name>` input value must be a YAML list of row mappings. Never
  use a scalar row such as `- true`. For example:
  `us:statutes/7/2012/j#relation.member_of_household:`
  followed by `- us:statutes/7/2012/j#input.snap_member_is_elderly_or_disabled: true`.
  Bad:
  `us:statutes/7/2012/j#relation.member_of_household: [- true]`
  Good:
  `us:statutes/7/2012/j#relation.member_of_household:`
  followed by `- us:statutes/7/2012/j#input.snap_member_is_elderly_or_disabled: true`.
- Never assign an imported module's computed `#rule_name` output in `input:`.
  If this file imports that rule, the compiled program computes it. To make an
  imported output true, false, or equal a value, mirror the imported file's
  companion test pattern by setting its underlying `#input.<fact>` and
  `#relation.<name>` keys.
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
- For proration tests with a source-stated denominator, choose input amounts
  divisible by that denominator so expected outputs are exact decimals, not
  rounded approximations. For example, if the denominator is 365, use a base
  amount like 36500 so `36500 * 182 / 365 = 18200`.
- Every test case for a local derived formula must assign every local factual
  `#input.<fact>` referenced by that formula, including facts that are false in
  the case. Missing false inputs make the executable test invalid.
- For every encoded `except`, `unless`, or `notwithstanding` carve-out, include
  companion tests for the positive path and the carve-out path so exclusions
  cannot be silently dropped.
- If a formula negates multiple exception predicates, include a separate
  companion test for each predicate that sets that exception input true and
  expects the directly affected Judgment rule to be `not_holds`.
- For any negated exception predicate, include a paired positive case with the
  same output rule where only the exception input changes from `false` to
  `true`; do not combine the exception test with another branch change. For
  example, an IRC section 24(h)(4)(B) noncitizen exception test must keep the
  same dependent/qualifying-child facts as its positive companion and flip only
  `noncitizen_exception_to_other_dependent_credit_applies`.
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
- For existing executable outputs in a copied target file, preserve the whole
  public executable surface for each retained output: local `name`, `kind`,
  `entity`, `dtype`, `period`, `unit`, `indexed_by`, and
  `versions[].effective_from`. Do not change the entity or period to a
  preferred modeling style when the existing file compiles. Never change an
  existing output from `dtype: Money` to `dtype: Judgment` just because the
  name sounds like an allowance/applicability decision.
- Preserve existing factual input slots referenced by copied formulas and
  companion tests. Do not swap a working local input surface for new friendly
  names or upstream abstractions unless the generated bundle performs a full,
  source-grounded surface migration.
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
- Do not encode legal effective dates as `dtype: String` parameters or date
  literal formulas such as `2025-01-01`. Axiom formulas have no date literal type.
  Use `effective_from` metadata for version timing, or use a
  source-stated semantic boolean predicate when a date window is a runtime
  condition. Do not put the date or year value in the fact name; use names like
  `taxable_year_begins_after_termination_date` or
  `taxable_year_is_in_temporary_effective_window`, not
  `taxable_year_begins_after_2024_and_before_2029` or
  `taxable_year_begins_after_december_31_2021`.
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
- Benefit, allotment, credit, deduction, allowance, and subsidy formulas must
  never emit negative money. When subtracting an income, contribution, or other
  reduction from a maximum amount, floor the result with `max(0, ...)` before
  applying downstream minimum-benefit or issuance branches.
- When a nonnegative credit, deduction, allowance, subsidy, or benefit is a
  percentage of `min(income, cap)` or similar, floor the income base at zero:
  use `rate * min(max(0, earned_income), cap)`, not
  `rate * min(earned_income, cap)`.
- Outputs named `taxable_income` or ending in `_taxable_income` must also never
  be negative. Wrap the final selected branch at zero, including both sides of
  conditionals: use `if condition: max(0, branch_a) else: max(0, branch_b)`,
  not `if condition: branch_a else: branch_b`.
- If that reduction has rounding alternatives, every branch must be floored:
  use `if round_up: max(0, maximum - ceil(reduction)) else: max(0, floor(maximum - reduction))`,
  never `if round_up: maximum - ceil(reduction) else: floor(maximum - reduction)`.
- Supported relation aggregators are `len(relation)`,
  `count_where(relation, predicate_fact)`, `sum(relation.amount_fact)`, and
  `sum_where(relation, amount_fact_or_derived, predicate_fact)`. Do not write
  `sum(relation, expression)` or put arithmetic inside a relation field access.
  Use `sum(relation.amount_fact)` only when `amount_fact` is a raw scalar fact
  supplied directly on each relation row. Do not use `sum(relation.local_output)`
  for a `parameter` or `derived` rule defined in the same file; for a computed
  per-related-entity amount, write
  `sum_where(relation, local_output, source_stated_predicate_fact)` instead.
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
  syntax.
- `dtype: Judgment` formulas must not use `if ... else ...`. Write them as
  boolean expressions using `and`, `or`, `not`, comparisons, and parentheses.
  For example, encode `if exempt: net_ok else: net_ok and gross_ok` as
  `net_ok and (exempt or gross_ok)`.
- When using negated conjuncts, write them as a multiline formula with each
  `not <predicate>` term on its own line joined by `and`, rather than one
  compact `not A and not B` line.
- Formula strings reference indexed parameter tables with `table_name[index_expr]`.
- Every substantive numeric literal must be grounded in the supplied source text unless it is -1, 0, 1, 2, or 3.
- If you encode a substantive numeric literal, `module.summary` or the rule's proof excerpt
  must include the exact source phrase containing that number. Do not omit a
  subsection, table row, or clause that grounds an encoded
  numeric amount, rate, threshold, cap, or limit.
- US tax filing status is a derived legal classification, not a downstream
  boundary fact. Do not create local `#input.filing_status` facts in a rule or
  test. Encode the upstream filing-status source first, then import its absolute
  RuleSpec output into downstream threshold, phaseout, deduction, and credit
  rules. If an already-encoded upstream filing-status output is unavailable,
  stop and encode that upstream source rather than synthesizing a local input.
- The shared US tax filing-status output remains a structural enum: 0 single,
  1 joint return, 2 married filing separately, 3 head of household, and
  4 surviving spouse / qualifying widow(er). If the source groups surviving
  spouse with joint return, every filing-status branch or match that handles
  status 1 must also handle status 4 in that same branch.
- Never encode US tax filing status as string literals such as
  `"married_filing_jointly"` or as separate boolean facts such as
  `married_filing_jointly`, `head_of_household`, or `surviving_spouse`. Use the
  imported numeric filing-status output in formulas, e.g.
  `match filing_status: 1 => joint_amount; 4 => joint_amount; ...`.
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
     defaults. If a test asserts an indexed `parameter` table output directly,
     the test must assign every `indexed_by` key as `#input.<key>`; otherwise
     assert the derived lookup output instead of the raw table. In ordinary
     end-to-end tests, do not output raw indexed parameter tables at all. If a
     local amount formula has a branch returning 0, include a companion case that
     asserts that local output is 0.
     For imported modules, only assign imported `#input` or `#relation` keys
     that exist in the current imported RuleSpec context. Do not preserve stale
     imported test inputs from copied files. Do not stub imported derived
     outputs as test inputs; imported programs are computed. If the downstream
     rule depends on an imported output, assign all current upstream factual
     inputs and relations needed by that imported output, including false facts.
  3. Proof inventory: every proof atom uses only an allowed `kind`; imported
     proof atoms include `import.target`, `import.output`, and `import.hash`;
     textual claim support is either direct corpus source support or a claim ID
     listed under `module.source_claims`.
  4. Import inventory: every `imports:` entry is an exact copied/importable
     RuleSpec target. Top-level `imports:` entries must be scalar strings; never
     map entries like `- target:` plus `symbols:`. Do not guess sibling paths; if
     required upstream context is missing, emit a typed missing-upstream/dependency
     request instead.

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
