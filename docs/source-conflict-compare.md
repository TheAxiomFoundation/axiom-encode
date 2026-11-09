# Source-conflict compare step

`axiom-encode compare-sources` runs two RuleSpec modules that decide the same
question on generated households and records the households for which the two
modules give different results. Design and open decisions:
[axiom-encode#2150](https://github.com/TheAxiomFoundation/axiom-encode/issues/2150).

A household with two different results is a **witness**. A witness proves that
two **encodings** differ. It does not prove that the two **sources** differ: the
cause can also be an encoding error, or an option that one source gives. So the
step writes every record with `review_state: unreviewed`, and each side carries
the citations and proof excerpts of the rules that decided.

Scope of this step: it reads a link, two modules and the engine, and writes one
JSON file. It does not change the encoder prompt, write to Supabase or touch
`known-source-conflicts.yaml`.

## Run it

```bash
axiom-encode engine-bind --axiom-rules-engine-path <engine checkout> --pin <engine commit the checkout pins>
axiom-encode compare-sources \
  --link docs/source-conflicts/examples/us-medicaid-community-engagement.link.json \
  --rulespec-root <rulespec checkout> \
  --axiom-rules-engine-path <engine checkout> \
  --out record.json
```

- `--rulespec-root` is the checkout root. The link's module paths are relative
  to it. The engine requires the directory name `rulespec-<country>`.
- The engine commit comes from the checkout's
  `.axiom/workflow-toolchain.toml` (`[workflow_toolchain].axiom_rules_engine_ref`),
  then `.axiom/toolchain.toml`. `--axiom-rules-engine-ref <sha>` overrides it.
  The engine checkout must hold a binary with a verified `engine-bind` receipt
  for that commit; the step does not build.
- Both modules must be in the one checkout. A pair across two RuleSpec
  repositories is not supported yet.

| Exit code | Meaning |
|---|---|
| 0 | The comparison ran. The record says whether there is a conflict. |
| 1 | The link is invalid or failed a gate. No record is written. |
| 3 | The engine could not be bound, or a compile or run failed. |

## The link

A link is a JSON or YAML file. Schema:
`src/axiom_encode/source_conflicts/link.schema.json`. The loader
(`link.py`) is the authority. It refuses unknown keys and a key that appears
twice in one mapping.

| Key | Meaning |
|---|---|
| `format` | `axiom-source-conflict-link/v1` |
| `id` | Stable id of the question, with a jurisdiction prefix (`us:medicaid-community-engagement-statute-vs-ifr`). The conflict keeps this id across commits. |
| `question` | Optional. The question in words. |
| `entity`, `period` | The entity of every compared rule and input, and the period to query (`period_kind`, `start`, `end`). v1 compares one entity with no relations. |
| `a`, `b` | The two sides: `label`, `module` (path in the checkout), `outputs` (rule ids), `combine`, optional `note`. |
| `facts` | Shared facts. Each has a `kind` (`bool`, `integer`, `decimal`) and the input it maps to on side `a`, side `b` or both (a fact that is only part of a `derived` input needs none). Optional: `values`, `step`, `min`, `max`, `note`. |
| `derived` | Inputs that one side takes and the other computes, keyed by input name: `kind`, `op`, `of` (facts). The only operator in v1 is `sum`. |
| `fixed` | Inputs held constant, keyed by input name: `kind`, `value`. |
| `constraints` | Conditions a real household meets: `rule`, optional `note`. |
| `guards` | Per side, rules that must hold for a household to be compared. |
| `sampling` | `max_combinations` (default 20,000, at most 1,000,000) and `seed` (default 0). |

`combine: any` means the side says yes when one output holds; `all` when every
output holds. Outputs and guards must be yes/no rules (Judgment or bool). A
comparison of amounts is not in v1.

A derived or fixed input goes to each side whose compiled artifact accepts its
name. An input has exactly one supplier.

### Constraint expressions

A constraint is one comparison of two linear terms over the facts. The step
parses the text itself; it does not evaluate Python.

```text
monthly_income >= work_hours * 7.25
work_hours + community_service_hours <= 744
agency_lacks_hours_documentation == true
```

Operators: `>= <= > < == !=`, `+`, `-`, and `*` between a number and a fact. A
bool fact, `true` and `false` can only stand alone on a side of `==` or `!=`.

## The question file

A link compares two modules. Several documents can answer the same question: a
statute, its regulation, federal guidance, state guidance. A **question file**
is the hub form for that. It states the question, its canonical facts and its
constraints once, and lists two or more **answers**. Each answer is one
document's encoding. N answers need N input maps, not N x (N - 1) / 2 links.

The question file is a sidecar. It does not change the RuleSpec format
(`format: rulespec/v1` stays) and no module is edited, so no file hash, signed
manifest or import hash changes.

Format `axiom-source-question/v1`. Schema:
`src/axiom_encode/source_conflicts/question.schema.json`.

| Key | Meaning |
|---|---|
| `id` | Stable id of the question (`us:medicaid/community-engagement-met-for-month`). |
| `question`, `entity`, `period`, `sampling` | As in a link. |
| `facts` | The canonical facts: `kind`, optional `values`, `step`, `min`, `max`, `note`. No input names here. |
| `constraints` | Over the canonical facts. |
| `answers` | Two or more answers. |

Each answer:

| Key | Meaning |
|---|---|
| `id` | Short id, unique in the file (`42-cfr-435-552`). |
| `label`, `module`, `outputs`, `combine`, `note` | As a side of a link. |
| `inputs` | Canonical fact -> this module's input name. |
| `derived`, `fixed` | As in a link, keyed by this module's input names. |
| `guards` | Rules of this module that must hold. |
| `document_type` | One of `statute`, `regulation`, `federal_guidance`, `state_regulation`, `state_guidance`, `form_instructions`, `third_party`. |
| `jurisdiction`, `corpus_citation_path` | `us` or `us-co`; the corpus path of the document. |
| `effective_from`, `effective_to` | Optional dates of the document. A period start outside them is a warning. |

```bash
axiom-encode compare-sources \
  --question docs/source-conflicts/examples/us-medicaid-community-engagement.question.json \
  --rulespec-root <rulespec checkout> \
  --axiom-rules-engine-path <engine checkout> \
  --out <directory>
```

The step expands the file into one link for each pair of answers and runs the
pair compare on it, with no change in behaviour. It writes into `--out`:

- one conflict record for each pair, `<answer a>~<answer b>.json`, in the record
  format below, with the id `<question id>:<answer a>~<answer b>`;
- `summary.json`, format `axiom-source-question-summary/v1`: the answers with
  their metadata, and for each pair its record file, `conflict`, `run`,
  `differ` and the direction, count and conditions of each witness group.

Rules of the expansion:

- Answers are ordered by document type (the order of the list above), then by
  id. So a statute is side `a` of a regulation, and a pair's id and sides do
  not depend on the order of the answers in the file.
- A pair holds only the facts that one of its two answers reads.
- A constraint over a fact that neither answer of a pair reads is not applied
  to that pair. The pair record lists it in `constraints_not_applied`, and the
  summary has a warning.
- Two answers that both supply one input must give the same definition.
- Question gates: at least two answers, unique answer ids, each canonical fact
  used by at least one answer. All link gates then run for each pair. A gate
  failure of one pair fails the question: it gives all of its records or none.

A pair record of a question has three more fields than a link record:
`source_question` (the question's id and hash), `sides.*.answer` (the answer's
metadata) and `constraints_not_applied`.

`docs/source-conflicts/examples/us-medicaid-community-engagement.question.json`
is the H.R.1 example as a question with two answers. Its one pair expands to
the example link, and gives the same result: 1,536 run, 8 differ, 1 group,
with the conditions `agency_converts_income_to_hours = true` and
`agency_lacks_hours_documentation = true`.

Place of a question file in a RuleSpec repository:
`<jurisdiction>/questions/**/<name>.question.yaml` (for example
`us/questions/medicaid/community-engagement.question.yaml`). A question file
is not a module. Module discovery in axiom-encode (`_rulespec_module_paths`,
the proof-kind normalization scan and the SNAP readiness scan) skips the
`questions/` tree and each `*.question.yaml` file, so a question file needs no
signed manifest, no companion test and no `format: rulespec/v1`. The engine
compiles a module of a checkout that holds a `questions/` tree with no change.
The validators of each RuleSpec repository must accept the same path; that is
their change, not this one.

Each gate failure of a pair names the pair and its two answers. An answer's
`derived` and `fixed` inputs must be inputs of its own module: an answer cannot
lean on an input that its partner supplies.

With `--question`, `--out` is a directory. The step removes the pair records
of an earlier run from it, so the directory matches `summary.json`.

## Gates

The gates run on the link and the two compiled artifacts before any household
is compared. Each failure is an error (exit 1), and the error lists all
findings.

1. Each output and guard is a derived rule of its side's artifact, is a yes/no
   rule, and has the link's entity and period kind.
2. Each fact input is in its side's input catalog
   (`metadata.input_catalog[].canonical_request_name`). Each derived and fixed
   input is in at least one catalog.
3. Each input that the compared rules read, through any rule they depend on, has
   a supplier: a fact, a derived input or a fixed input. The step reads this
   from the artifact's expression trees. It is stricter than the engine, which
   skips an input behind a short circuit.
4. Each supplied input's kind matches its use in the rules (yes/no against
   number).
5. Each compared rule has a formula version in force at the period start.
6. Each fact has at least one value (a `min` and `max` can leave none).
7. No constraint removes every combination, alone or together with the others.
   When the grid is sampled, this gate tests the sample, and the error says so.
8. The guards do not skip every household that meets the constraints. When the
   grid is sampled, this gate tests the sample, and the error says so.

Gates 6 to 8 are errors, not warnings: such a link can only report
"no conflict". Gate 8 runs after the engine, because a guard is a rule.

A supplied input that no compared rule reads is a warning in the record.

A constraint that is too strong hides a real conflict. The step therefore runs
the removed combinations too, and the record reports for each constraint:
`removed`, `removed_by_this_alone` and `removed_that_differ` (removed
combinations for which the two sides differ). A reviewer of a link must read
`removed_that_differ`.

## Values

A fact with `values` uses them. For any other fact the step takes the values
from the modules:

- a bool fact gets `false` and `true`;
- a number gets zero, and each **threshold** with the value just below and just
  above it (`step`, default 1), limited by `min` (default 0) and `max`.

A threshold is a constant that a compared rule compares a fact-dependent value
with. The step folds constants from literals, parameters, fixed inputs, rules
over those, and plain arithmetic. Example: 42 CFR 435.552 compares income with
`applicable_federal_minimum_wage_requirement * 80`; with the wage fixed at 7.25
the threshold is 580, so income gets 579, 580 and 581. The record lists each
fact's thresholds with the rule they come from.

Limit: a threshold applies to every fact its comparison depends on, with no
change of unit. The step does not solve for values (for example the income at
which income / wage plus other hours reaches 80).

The grid is the product of the facts' values, in the link's fact order. When
it holds more than `sampling.max_combinations` combinations, the step runs a
sample: the first and last combination, then combination
`sha256("<seed>:<k>") mod size` for `k = 0, 1, 2, ...` up to the cap. The
record then has `sampled: true`. Constraints apply after sampling.

## The record

Format `axiom-source-conflict-record/v1`. The same link, modules and engine
give the same bytes: the record holds no time, host or path.

| Field | Meaning |
|---|---|
| `id` | The link's id. |
| `review_state` | Always `unreviewed`. A later review sets `source_conflict`, `encoding_error` or `refinement_or_option`; this step never does. |
| `conflict` | `true` when a household differs. |
| `link.sha256` | Hash of the normalized link. File format and number spelling do not change it. The order of the facts, derived inputs and fixed inputs does: it sets the grid order, the sample and the witness that is chosen. |
| `rulespec` | `commit` of the checkout, and `dirty` when tracked files differ from it. |
| `engine` | `commit` bound, and the artifact's engine `version`. |
| `sides.a`, `sides.b` | `label`, `module`, `module_sha256`, `outputs`, `combine`, `guards`, and `rules`. |
| `sides.*.rules` | Each rule that a deciding rule depends on: `kind`, `source`, `corpus_citation_path`, `proof_atoms` (citation path and excerpt, from the module file) and `depends_on`. |
| `facts` | Per fact: kind, inputs, `read_by`, `values`, `values_from` (`link` or `modules`) and `thresholds`. |
| `assumptions` | Each note of the link, with its place. |
| `constraints` | Per constraint: `removed`, `removed_by_this_alone`, `removed_that_differ`. |
| `counts` | `grid`, `combinations` (after sampling), `removed_by_constraints`, `removed_that_differ`, `run`, `skipped_by_guard`, `differ`, `witness_groups`. |
| `sampled`, `sampling` | Whether the grid was sampled, and the method, cap and seed. |
| `warnings` | Gate warnings. |
| `witness_groups` | The witnesses, grouped. Each group: `direction`, `deciding_rules`, `one_side_facts`, `conditions`, `count`, `witness`, `fact_values`. |

`run` counts the combinations that meet every constraint. `differ` counts those
that pass the guards and get different results.

Witnesses are grouped by direction (the result on each side), by the deciding
output rules on each side, and by the facts that only one side reads (a bool by
its value, a number by zero or not zero). For `combine: any`, the deciding
rules of a yes are the outputs that hold; of a no, all outputs. For `all` it is
the reverse. Each group has its `count`, the values each fact takes in the
group (`fact_values`), and one `witness`: the household with the fewest facts
that are not zero or false, and the first of those in grid order.

Each group also has `conditions`: the facts that only one side reads and that
have one value in all witnesses with the same direction and deciding rules
(a bool by its value, a number as `zero` or `nonzero`). Example for H.R.1:
`agency_converts_income_to_hours = true` and
`agency_lacks_hours_documentation = true`. A condition holds for the values
that the grid tested; it is not a proof for all values.

## Example: H.R.1 Medicaid community engagement

`docs/source-conflicts/examples/us-medicaid-community-engagement.link.json`
links 42 U.S.C. 1396a(xx)(2) (`us/statutes/42/1396a/xx.yaml`) with
42 CFR 435.552 (`us/regulations/42-cfr/435/552.yaml`). It has no explicit
values. 435.552 defers its one-or-more rule, so side `b` combines six path
rules with `any`.

Against rulespec-us `fe62a527af` with engine
`89571cc2a938707fd60a5489b951135345697296`:

| Link | Combinations | Removed | Run | Differ | Groups |
|---|---|---|---|---|---|
| Example (values from the modules) | 3,584 | 2,048 | 1,536 | 8 | 1 |
| `tests/fixtures/source_conflicts/hr1_prototype_values.link.json` | 1,920 | 640 | 1,280 | 17 | 1 |
| `tests/fixtures/source_conflicts/hr1_threshold_values.link.json` | 2,048 | 896 | 1,152 | 2 | 1 |

In each run the one group is: the statute says not met, the regulation says met
through `combined_activity_hours_path_demonstrates_community_engagement`, with
`agency_converts_income_to_hours` and `agency_lacks_hours_documentation` both
true. This is the income-to-hours conversion of 435.552(e)(2). CI has no
rulespec-us checkout, so this run is manual; CI runs the same step on the
fixture modules in `tests/fixtures/source_conflicts/` with the real engine.

## What the step does not find

- A difference in a provision that no module encodes, or that a module defers.
- A difference with no yes/no outcome in the compared rules: amounts,
  deadlines, notices, order of steps.
- A difference that needs a value the grid does not hold.
- A difference between two modules with no link.

The step reads the expression kinds that the RuleSpec lowering emits today. It
does not see an input behind a node kind that it does not know (for example an
optional input with a default): the input gets no supplier check, no kind
check and no threshold.

## Package layout

`src/axiom_encode/source_conflicts/` is self-contained so that it can move
(#2150 leaves open whether the step lives here or in axiom-oracles). Only
`engine.py` imports from the rest of axiom-encode: `engine_binding` for the
pinned, receipted binary and `rules_engine_compat` for the compile call. The
command is dispatched in `entrypoint.py`, not in `cli.py`.
