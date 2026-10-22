# Encoding enacted amendments before commencement

An enacted amendment that has not yet commenced belongs in a dated version of
the affected principal-Act rule. Ground that version in the amending provision
and its commencement provision, preserving the principal rule's supported
precommencement behavior. Once the official consolidation incorporates the
change, re-point its evidence to the principal Act. This is the d1832 maintainer
ruling of 21 October 2026.

Request the **principal provision's corpus citation**, with the amending text
and commencement evidence attached from the same signed corpus release. The
amendment document is a proof-citation target; it is not a RuleSpec import or a
separate replacement target. The existing prompt requests effective-dated
`versions` in the principal target module and proof atoms citing the attached
amending document's corpus citation path. Include cases immediately before and
at commencement; scenarios exercising the new behavior must use periods on or
after commencement.

## Corpus metadata for amendment discovery

Discovery runs over active rows of the target's jurisdiction and document class
in the pinned release. Ingest must retain the amendment body and identify the
amending document and its target. The fields below describe the existing
discovery contract in `src/axiom_encode/harness/evals.py`
(`_is_amendment_row`, `_amendment_target_values`,
`_target_document_identifiers`, and `_discover_amendment_documents`).

| Purpose | Fields read by discovery |
| --- | --- |
| Mark an amendment row | The presence of top-level `metadata.amends`, or top-level `metadata.document_type` containing `amendment` (case insensitive). |
| Declare its principal target | `metadata.amended_act`, `amended_acts`, `amendment_target`, `amendment_targets`, `amends`, `amends_eli`, `target_act`, `target_acts`, or `target_eli`. The same keys may appear in a nested `metadata.amendment` object. |
| Identify the principal document | Its canonical citation path and document root, plus target/parent `metadata` keys identifying an ELI (`eli`, `eli_*`, `*_eli`, or `*_eli_*`) or containing both `act` and `number`. |
| Support legacy name matching | Target/parent `metadata.title`, `title_short`, `title_alternative`, heading, and meaningful citation-path segments. |
| Group and order evidence | Corpus row `source_path`, `version`, `expression_date`, and `citation_path`. |

Target-field keys normalize case and hyphens to underscores; the top-level
amendment markers are read with the spellings shown. Target values may be
strings, integers, lists, or nested objects; scalar values are traversed. A
target declaration alone, such as `metadata.target_act`, does not mark the row
as an amendment: also supply `metadata.document_type` or `metadata.amends`.

Prefer canonical principal-document or principal-provision citation paths in
`metadata.amends`. For example, this metadata on an amending document declares
a principal Act unambiguously:

```json
{
  "document_type": "amendment_act",
  "amends": ["nz/statute/act/public/2018/0032"]
}
```

Canonical path targets select the structured discovery tier. It can include
all explicitly related amendments across capture `version` scopes in the same
verified release; those scope versions are not legal effective dates. Other
identifier/name matching stays within the requested row's `version` and
requires an explicit target field. Names require at least two distinctive
tokens in a contiguous phrase. Do not depend on a document's title or body
mentioning the principal Act without target metadata.

Rows sharing a `source_path` form one amendment document; discovery attaches
the shallowest matching row's body and metadata. Put the marker, target
declaration, operative text, and commencement evidence on the attached document
or otherwise supply their authoritative provision context. Discovery excludes
the requested target's own document from this principal-target attachment
path. The rendered eval bundle lists attached evidence as
`kind: corpus_amendment_act` in `context-manifest.json`; check that the retained
context contains both the change and the applicable commencement provision.
Context has size limits, so discovery alone does not guarantee that every
provision's text is visible.

## Direct amending-provision requests

An amending provision must not acquire the date-silent `0001-01-01` scaffold.
Its rules need the affected principal provision, authoritative commencement
evidence, and any structural scope needed to select that commencement (for
example, membership in a Part with a separate commencement date). An assent,
capture, expression, or generic in-force date cannot replace that evidence.

To route a direct amending-provision request to its principal target, ingest
must supply these fields on the requested provision's `metadata`:

| Field | Required value |
| --- | --- |
| `amendment_target` | One exact canonical corpus citation for the affected principal provision. `amendment_targets` containing a single target is also accepted; multiple targets remain blocked. |
| `commencement_date` | The applicable commencement date in `YYYY-MM-DD` format, scoped to this amending provision. |
| `commencement_citation_path` | An exact citation for a commencement provision in the same amending document (`source_path`) and capture `version`. Its body must state the supplied date in ISO or English long-date form. Other forms currently produce an unresolved-context blocker. |

The requested row or its parent document must carry an amendment marker
(`metadata.amends` or an amendment `metadata.document_type`). An operation
heading such as `Schedule 4 amended` also identifies an amending provision, so
missing amendment ingest markers cannot silently enable the neutral scaffold.
An Act-level `amends` declaration helps discovery but does not choose the exact
principal provision for a direct request.

For a principal rule with both old and amended versions, scope rule-level proof
atoms to `versions[0].formula`, `versions[1].formula`, and so on. Ground the old
formula in the principal provision and the new formula in the amendment. An
amendment proof for an old version's `effective_to` may establish its cutoff
without moving its start. Version-only proof cannot erase rule-level amendment
evidence. An ambiguous citation to the whole amending Act inherits the bound
of its contained amending provision; cite an exact other provision when that
distinction matters.

The route resolves the principal provision as the output target and attaches
the requested amending provision plus commencement provision as proof evidence.
If context limits would omit either required provision, encoding stops with a
blocker. The applicable commencement cannot be replaced by a scaffold date.
For a commencement section with both a general date and a Part-specific date,
ingest must bind the correct date to the amending provision; the encoder does
not guess Part membership from multiple sibling dates.

When this context is unresolved, encoding must stop with an explicit
unresolved-amendment-context blocker naming what is missing. Correct the
signed corpus metadata/context and request the principal citation with the
amendment evidence attached. Ordinary and legacy replacement flags do not
authorize replacing a different principal corpus identity merely because the
requested provision amends it.

For NZ Act 2026/54 section 15, the relevant context includes section 2's
statement that Part 2 comes into force on **1 April 2027**, the section's Part
membership, and its principal target. The nine amendment-derived rules must
not apply in October 2026. The principal rule retains its supported earlier
behavior and changes on 1 April 2027.
