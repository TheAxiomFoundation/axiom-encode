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
- Do not emit `source_url`; RuleSpec validation reads normalized corpus provisions,
  not raw PDFs or web pages.
- Use `rules:` as a list of rule objects.
- Use `kind: parameter` for source-stated amounts, rates, thresholds, caps, and limits.
- Use `kind: parameter` with `indexed_by` and versioned `values` for source-stated
  numeric tables/scales keyed by household size, family size, income band,
  age band, or another row key. Do not encode those cells as `match` arms or
  numeric literals inside a derived formula.
- Use `kind: derived` for entity-scoped outputs.
- Use `dtype: Judgment`, not `dtype: Boolean`, for legal eligibility,
  availability, applicability, entitlement, and other holds/not-holds style
  outputs, especially when the formula contains `not`.
- Use `kind: data_relation` for executable runtime predicates such as
  `member_of_household`. Put arity under `data_relation.arity`.
- Do not encode simple unary factual inputs as `kind: data_relation` rules. If
  a formula needs a local true/false fact, reference a descriptive bare fact
  name in the formula and put that fact in tests as
  `<jurisdiction>:<repo-path>#input.<fact>`.
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
- If context files import the target file or reference target outputs, preserve
  the target file's public output names unless the source text proves the old
  interface was legally wrong. Do not rename an exported value just because a
  clearer friendly name is possible.
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
- Formula strings use Axiom formula syntax: `if condition: value else: other`, `==`, `and`, and `or`.
- Axiom conditionals are expression syntax, not YAML syntax. Money/scalar
  formulas may use `if condition: value else: other`; do not use Python ternary
  syntax. Judgment formulas should usually be boolean expressions, not `if`
  conditionals.
- When using negated conjuncts, write them as a multiline formula with each
  `not <predicate>` term on its own line joined by `and`, rather than one
  compact `not A and not B` line.
- Formula strings reference indexed parameter tables with `table_name[index_expr]`.
- Every substantive numeric literal must be grounded in the supplied source text unless it is -1, 0, 1, 2, or 3.

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
