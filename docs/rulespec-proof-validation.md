# RuleSpec Proof Validation

## Purpose

RuleSpec is the only executable Axiom layer. Named corpus releases store the
authoritative source text, anchors, hierarchy, tables, and provenance used by
an encoding. Encoding agents may use release-bound corpus text and reachable
RuleSpec modules as context, but they must not turn missing context into
invented executable policy.

The production invariant is:

```text
Every policy-bearing RuleSpec atom must be justified by direct target source
text from the checkout's immutable named corpus release or an approved imported
RuleSpec export. If no proof can be built, encoding stops with a typed request.
```

Mutable `module.source_claims` and proof-atom `claim` references are not part of
the contract. The durable gate is a direct proof tree per policy-bearing atom.

## Boundary

The layers have separate authority:

- `corpus`: verbatim source text, deterministic anchors, hierarchy, extracted
  tables, source versions, and content hashes.
- `rulespec`: executable policy logic, parameters, imports, tests, and runtime
  outputs.

No mutable side artifact can justify an executable atom. Evidence must resolve
through the configured named release or a content-addressed RuleSpec import.

`module.interpretations` documents readings adopted because the source conflicts
with itself, is read other than literally, or has a question put to its publisher.
The encoder validates the block in the same proof-validation path used by
`validate`, apply, and reviewed-candidate promotion. Each `texts[].excerpt` must
match its canonical corpus citation with the source proof-atom matcher, and
`affects` must name local rules that are not deferred outputs. These excerpts
never satisfy a proof obligation or count as complete-source coverage; the
affected executable rules still need their own proof atoms and tests.
Each alternative requires a non-blank string `id` matching
`^[a-z0-9]+(-[a-z0-9]+)*$`, unique within that interpretation; its only allowed
keys are `id`, `reading`, and optional `effect`.

Raw YAML is checked before any load-and-dump rewrite, because loading resolves
duplicate keys, tags and merge keys silently. Interpretation scalars may carry
only the canonical string tag `!!str`, interpretation collections carry no
tags, and interpretation mapping keys must be unique. A document may have only
one root `module` key, and neither the root nor `module` may carry a tag other
than `!!map` or the non-specific `!` (PyYAML loads a `!!set` mapping as a set,
which the engine reads as the mapping). Interpretation strings must not contain
UTF-16 surrogate code points: PyYAML decodes escapes such as `"\uD800"` to lone
surrogates, which the engine rejects. YAML merge keys follow the rule for
the whole document, below.

Duplicate mapping keys are rejected anywhere in a generated module or its
companion test before any step writes the file back from a loaded payload:
apply stamping, the apply and eval repairs, materialization and the retry
overlay. Loading keeps only the last of two equal keys, so such a rewrite would
silently drop the first `rules` block, rule field or test output, and the
engine, which rejects the raw file, would compile what was left. Keys are
compared as `yaml.safe_load` constructs them, so `rules`, `"rules"`,
`!!str rules` and an aliased key collide, and so do `1`, `0x1`, `1.0` and
`true`. Merge keys keep their override semantics.

Explicit YAML tags are limited, anywhere in a module or its companion test, to
the ones PyYAML and the engine read alike: a mapping may carry only `!!map` or
the non-specific `!`, a sequence only `!!seq` or `!`, and a scalar only
`!!str`. The engine ignores `!!set`, `!!omap`, `!!pairs` and local tags on a
collection and reads the mapping or sequence as written, while `yaml.safe_load`
turns a `!!set` mapping into a set of its keys and an `!!omap` or `!!pairs`
sequence into a list of key-value pairs. The validators skip a rule, version
or proof that does not load as a mapping, so such a tag would hide content the
engine compiles: a policy-bearing rule written `- !!set` would pass strict proof
validation, and `proof_validation: !!set` would turn strict mode off. Other
scalar tags diverge as well: `!!binary` loads as bytes and `!!timestamp` as a
date where the engine reads the text, `!!null` loads as null whatever the text,
and PyYAML resolves `! '0.1'` as the number 0.1. Proof validation and the
validator pipeline's preflight report each such tag with its location. The
apply preflight and stamping and the retry overlay refuse the file before any
rewrite, and the eval repairs and materialization keep its raw text for
validation to report. They treat text that PyYAML cannot load because it raises
a plain `ValueError` the same way. A double-quoted escape beyond U+10FFFF, such
as `"\U00110000"`, or an impossible unquoted date such as `2024-13-01`, raises
one. The engine rejects the escape, and an impossible `effective_from`.

YAML merge keys are rejected anywhere in a module except inside a rule's
`metadata`. `yaml.safe_load` applies a merge key (`<<`, or a key tagged
`!!merge`) by copying the merged mapping's keys into the mapping that holds it.
The engine never applies merges: it reads `<<` as an ordinary key, which it
ignores in a rule or version and rejects in `module.source_verification`. So a
rule written `- <<: *base` would pass strict proof validation while the engine
rejected it or compiled it without the merged fields, and a rewrite would write
the merged fields out and change what the engine compiles. The engine never
reads a rule's `metadata`, and modules merge shared proof sources there, so
merges stay allowed inside it; a mapping anchored in `metadata` and aliased
elsewhere is checked where it is used. Companion tests may use merges
anywhere, since only PyYAML reads them, and a quoted `'<<'` or `!!str <<` key
is an ordinary key to both loaders. Proof validation and the validator
pipeline's preflight report each merge key with its location. The apply
preflight and stamping and the retry overlay refuse the file before any
rewrite, and the eval repairs and normalization keep its raw text.

The encoder is instructed to create new entries with `status: open`. A repair may carry recorded human rulings and
publisher replies supplied through mandatory review findings into `ruling` and
`confirmation`, attributed only to organisations. A supplied ruling permits
`status: ruled`; `confirmed` and `contradicted` require both blocks. The model
is instructed never to invent either block; validation checks each block's shape,
not where it came from. `status: contradicted` blocks validation until
the module is repaired, the old reading is moved into
`alternatives`, and the new reading's status is set. The block never changes
lowering, compilation, or execution. See the
[module metadata contract](../README.md#source-pinning-and-staleness) for its
shape and status-dependent evidence requirements.

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

- `missing_source`: required authoritative source text is absent from the named release.
- `bundle_expansion_request`: the relation closure is too narrow.
- `corpus_defect`: source text, table extraction, or metadata is malformed.
- `segmentation_fix`: the corpus anchor is too coarse or too fine.

These are encoder outputs for workflow triage, not executable policy.

## Implementation Order

1. Build a standalone proof validator: sealed bundle plus RuleSpec module in,
   per-atom pass/fail with typed failures out.
2. Validate explicit proof atoms in RuleSpec.
3. Add sealed bundle manifests with source, import, table extraction, and
   compiled export versions.
4. Derive proof obligations from compiled RuleSpec so missing atoms are found
   automatically.
5. Teach encoders to emit typed stop requests when a proof obligation cannot be
   satisfied.
6. Make strict proof validation a promotion gate once existing RuleSpec files
   declare proof trees.

This sequence keeps the execution layer clean while avoiding a heavy
interpretation layer that quietly becomes RuleSpec without tests.
