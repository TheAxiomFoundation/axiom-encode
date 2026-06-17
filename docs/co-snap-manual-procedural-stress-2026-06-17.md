# Colorado SNAP Manual Procedural Encoding Stress Test

Date: 2026-06-17

## Purpose

Stress test Axiom encoding on procedural SNAP rules from the Colorado manual,
especially interview, application, recertification, notice, office-duty, and
deadline rules. The goal was to find where the encoder or RuleSpec validation
breaks before attempting a full install into `rulespec-us-co`.

No generated Colorado RuleSpec artifacts from this run were installed or edited
by hand.

## Inputs

- Corpus: `/Users/maxghenis/TheAxiomFoundation/axiom-corpus/data/corpus/provisions/us-co/regulation/2026-04-29-10-ccr-2506-1.jsonl`
- Corpus size: 290 JSONL rows, 280 non-empty provisions, about 672k characters.
- Encoder worktree: `/Users/maxghenis/_axiom-worktrees/axiom-encode-co-snap-manual-stress-20260616`
- Validation RuleSpec worktree: `/Users/maxghenis/TheAxiomFoundation/rulespec-us-co-manual-stress-20260616`

## Harness

Added a local stress harness:

- `scripts/co_snap_manual_stress.py`

The harness calls:

```bash
uv run axiom-encode encode <citation_path> \
  --backend openai \
  --model gpt-5.5 \
  --output /tmp/axiom-co-snap-manual-stress/<run> \
  --corpus-path /Users/maxghenis/TheAxiomFoundation/axiom-corpus \
  --axiom-rules-engine-path /Users/maxghenis/TheAxiomFoundation/axiom-rules-engine \
  --policy-repo-path /Users/maxghenis/TheAxiomFoundation/rulespec-us-co-manual-stress-20260616 \
  --mode repo-augmented \
  --skip-reviewers \
  --no-sync
```

It writes generated artifacts under `/tmp/axiom-co-snap-manual-stress`, records
stdout/stderr per provision, and then runs deterministic compile/CI validation
against the scratch artifact. It does not install results into a live RuleSpec
repo.

The harness now uses `encode --skip-reviewers` so each encoder process exits
normally after deterministic compile/CI validation. Earlier smoke runs killed
the encoder once the expected YAML artifact stabilized; those rows are marked
with `terminated_after_artifact: true` in their JSONL output.

## Runs

### Top-of-manual sample

- Output: `/tmp/axiom-co-snap-manual-stress/top-012`
- Results: `/tmp/axiom-co-snap-manual-stress/top-012/revalidated-results.jsonl`
- Scope: first 12 non-empty manual provisions, including definitions,
  introduction, benefit use, confidentiality, and fair-hearing headings.

### Application and recertification band

- Output: `/tmp/axiom-co-snap-manual-stress/application-4.20x`
- Results: `/tmp/axiom-co-snap-manual-stress/application-4.20x/revalidated-results.jsonl`
- Scope: 33 provisions matching `4.20x`, including application filing,
  authorized representatives, interviews, processing standards, expedited
  service, categories of eligibility, prorating, certification periods, and
  recertification.

### Post-fix spot check

- Output: `/tmp/axiom-co-snap-manual-stress/postfix-4.205`
- Scope: `us-co/regulation/10-ccr-2506-1/4.205`
- Result: the provision now generates both YAML and a companion test and
  compiles. It still fails CI on substantive proof/test-input issues.

## Results

Across the two main samples:

| Metric | Count |
| --- | ---: |
| Unique provisions attempted | 45 |
| Generated main YAML | 42 |
| Generated companion test | 38 |
| Compile pass after revalidation | 41 |
| Full deterministic CI pass | 18 |
| No-artifact timeouts | 2 |

Failure buckets:

| Bucket | Count |
| --- | ---: |
| Pass | 18 |
| Companion test coverage gaps | 12 |
| Harness crash / no tests emitted | 4 |
| Source scope or entity mismatch | 3 |
| Proof/import hygiene | 2 |
| No artifact: timeout | 2 |
| Schema/source-relation delegation | 1 |
| Mixed entity outputs in one test | 1 |
| Ungrounded scalar/numeric | 1 |
| No artifact: network/other | 1 |

After the normalizer fix, one prior "harness crash / no tests emitted" case
(`4.205`) became a clean encode+compile with CI failures about proof import
references and missing explicit test inputs.

## What Broke

### 1. Bulk encoding needs a deterministic mode

The public `encode` path creates the artifact but then continues into the LLM
reviewer tier. For a whole-manual stress run, we need a first-class mode that:

- Writes the artifact and companion test.
- Runs compile and deterministic CI.
- Skips LLM reviewers/oracles unless explicitly requested.
- Emits one structured result row per source provision.

The local stress harness proves the workflow but uses process termination as a
workaround. That should move into `axiom-encode`.

### 2. Procedural provisions need stricter companion tests

Most failures were not compile failures. They were validation failures because
the generated test did not assert every derived procedural output, especially
Judgment outputs. Examples:

- `4.130` benefit-use rules.
- `4.140` confidentiality.
- `4.201` application processing.
- `4.204` interviews.
- `4.205.3` and `4.205.32` processing-delay rules.
- `4.208` certification periods.

The encoder prompt and/or post-generation harness should require:

- Every derived output is asserted in the companion test.
- Every Judgment rule has at least one positive `holds` assertion.
- Every branch returning zero/false has an explicit companion case when the
  formula has such a branch.
- Tests assign all local factual inputs, including false facts.

### 3. Entity boundaries are stressed by procedural rules

Colorado SNAP procedure rules often speak about households, household members,
authorized representatives, local offices, county departments, SSA offices, and
the State agency in the same provision. The encoder frequently picked an entity
that the source-scope validator rejected.

Examples:

- `4.000.1`: ABAWD definition encoded person-scoped where source wording was
  treated as household/unit-scoped.
- `4.202.2`: ineligible-person income/resources counted for the household.
- `4.206`: categorical eligibility mixes member-level conditions and
  household-level consequences.

This supports adding or improving flexible procedural entities and making the
prompt explicit about separating person/member facts from household/unit
consequences.

### 4. Multi-entity output tests need splitting

`4.202.3` generated one test case with derived outputs across `Household` and
`Person`. The validator correctly rejected it. Procedural encodings need test
generation to split outputs by entity.

### 5. Proof/import hygiene is a repeatable failure

The encoder sometimes imports proof atoms without referencing the imported
symbol in the formula, or omits proof atoms on policy-bearing rules.

Examples:

- `4.202.32`
- `4.205.11`
- `4.207.2`
- `4.207.3`
- post-fix `4.205`

The prompt should state that every proof import must be formula-referenced, and
every policy-bearing rule must carry proof atoms.

### 6. Constants need extraction even in procedural formulas

`4.207.2` embedded `30` directly in a formula. Even when the source is
procedural, numeric thresholds should be extracted to named concepts or table
values rather than embedded as scalar literals.

### 7. Source relation schema needs clearer instructions

`4.100` tried to encode incorporation of federal SNAP regulations using an
`implements` source relation but omitted `source_relation.basis.delegation`.
The prompt needs clearer guidance for incorporation/delegation relationships.

### 8. Names need collision avoidance

`4.208.1` generated a generic `member_of_household` rule name that collided with
a sibling export. Procedural sections need semantic branch-specific names.

## Small Fix Made

The stress run exposed a harness crash in test normalization: strings matching
the numeric-expression regex could be sent to `_format_safe_numeric_expression`,
which returns `None` for invalid expressions. The caller passed `None` into
`yaml.safe_load`, causing an `AttributeError` and preventing companion test
materialization.

Changed:

- `src/axiom_encode/harness/evals.py`
- `tests/test_evals.py`

The normalizer now preserves unsupported numeric-looking strings instead of
crashing. Added a regression for `"30 / 0"`.

Verification:

```bash
uv run python -m py_compile scripts/co_snap_manual_stress.py src/axiom_encode/harness/evals.py
uv run --with pytest --with pytest-cov python -m pytest tests/test_evals.py -k 'invalid_numeric_expression or materialize_eval_artifact_normalizes_mapping_style_tests_to_list or normalize_test_periods_drops_speculative'
```

## Recommended Next Steps

1. Add a first-class deterministic bulk encode mode to `axiom-encode` so the
   stress harness no longer has to terminate the process after artifact
   stabilization.
2. Strengthen procedural-test prompt/harness requirements: assert every derived
   output, split test outputs by entity, assign every local factual input, and
   require positive Judgment coverage.
3. Strengthen proof guidance and/or validator repair loops for formula-referenced
   proof imports and policy-bearing proof atoms.
4. Improve entity handling for procedural actors: household, member, authorized
   representative, local office/county, SSA office, and State agency.
5. Re-run the full 280 non-empty Colorado SNAP manual provisions in chunks after
   the deterministic bulk mode and prompt changes.
6. Install only passing generated provisions into `rulespec-us-co`; do not
   hand-edit generated RuleSpec to make failing rows pass.
