# Axiom Encode

AI-assisted Axiom RuleSpec encoding infrastructure. This repo owns the automation layer for turning source legal text into jurisdiction-repo `*.yaml` RuleSpec artifacts and validating them with `axiom-rules`.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
axiom-encode encode "26 USC 32(a)(1)" \
  --output /tmp/axiom-encode-encodings

axiom-encode validate /tmp/axiom-encode-encodings/codex-gpt-5.5/statutes/26/32/a/1.yaml
axiom-encode proof-validate /tmp/axiom-encode-encodings/codex-gpt-5.5/statutes/26/32/a/1.yaml
axiom-encode inventory --root ~/TheAxiomFoundation
```

`encode` resolves the requested citation to `corpus.provisions` before model
generation. Local `axiom-corpus/data/corpus/provisions` artifacts are used first;
Supabase is the fallback. If no corpus provision exists, encoding stops before
calling a model.

`proof-validate` checks explicit RuleSpec proof trees without reviewers or
oracles. It also resolves declared source-claim IDs against local
`axiom-corpus/claims` artifacts and rejects missing, unaccepted, executable, or
placeholder-subject claims. Strict proof validation is enabled per file with
`module.proof_validation.required: true`.

`inventory` counts checked-in RuleSpec files across sibling `rules-*` repos,
separating source/provision encodings from composition modules and reporting
rule-kind totals. Use `--json` when feeding the current inventory into status
reports or dashboards.

## Run telemetry and repair manifests

`axiom-encode encode` records each completed eval-backed run in the local
encoding DB and creates a linked SDK-style session with
`session_id=encode-<run_id>`. The session stores concise request/result/outcome
events, issue events when the final workflow fails, token counts, model, cwd,
and `axiom-encode` version so the public ops dashboard can show run health and
session telemetry from the same encode path.

When Supabase write credentials are configured, `encode` syncs both records:

```bash
export AXIOM_ENCODE_SUPABASE_URL=...
export AXIOM_ENCODE_SUPABASE_SECRET_KEY=...
axiom-encode encode "26 USC 32(a)(1)" --output /tmp/axiom-encode-encodings
```

The command prints `supabase_sync=run+session` when both records are uploaded.
Use `axiom-encode sync-agent-sessions --session encode-<run_id>` to replay a
single local session sync.

Failed encode runs write a sibling `*.repair.json` file next to the generated
RuleSpec candidate. That manifest includes the run ID, session ID, citation,
trace path, generated output path, and actions for inspecting the trace,
and rerunning the same citation. The manifest does not include a manual repair
action; live RuleSpec changes should come from a new encode run. When
`--apply` validates and applies cleanly, the final outcome is successful even
if standalone validation failed before the generated file was checked in the
policy-repo overlay, and no repair manifest is written.

## Applying generated RuleSpec

Live jurisdiction RuleSpec must be installed with `axiom-encode encode --apply`,
not by editing YAML in place. `--apply` validates the generated main file,
companion test, and direct dependent RuleSpec modules inside a temporary
policy-repo overlay, copies the validated files into the target rules repo, and
writes a signed JSON apply manifest under `.axiom/encoding-manifests/`.
Set `AXIOM_ENCODE_APPLY_SIGNING_KEY` when running `--apply`; without it the
encoder refuses to install generated RuleSpec into a live repo.

Repository CI should run:

```bash
axiom-encode guard-generated --repo "$GITHUB_WORKSPACE" --base-ref "$BASE_REF" --head-ref "$GITHUB_SHA"
```

That guard rejects changed RuleSpec YAML under `statutes/`, `regulations/`, or
`policies/` unless the changed files match an encoder apply manifest in the
same diff and that manifest has a valid `AXIOM_ENCODE_APPLY_SIGNING_KEY`
signature.

## Eval suites and readiness gates

Use manifest-driven benchmark suites when you want an explicit readiness answer instead
of ad hoc spot checks.

```bash
axiom-encode eval-suite benchmarks/us_snap_co_child_support_deduction_option_refresh.yaml
axiom-encode eval-suite benchmarks/us_snap_federal_reconstruction_seed.yaml
axiom-encode eval-suite-archive /tmp/axiom-encode-suite-run
```

- `benchmarks/us_snap_*_refresh.yaml` manifests are corpus-backed SNAP refresh lanes.
- `benchmarks/us_co_*` manifests exercise Colorado Works repair and seed cases.

Each suite reports:
- success rate
- compile pass rate
- CI pass rate
- zero-ungrounded rate
- PolicyEngine pass rate on oracle-mappable cases
- mean estimated cost

The command exits `0` only when all readiness gates pass.

Use `eval-suite-archive` when a run is worth citing later. It copies the full
suite output tree into a durable local registry under `artifacts/eval-suites`
by default, rewrites archived JSON/JSONL artifact paths away from `/tmp`, and
appends a record to `artifacts/eval-suites/index.jsonl`.

## Methods and paper notes

The repo now keeps paper-oriented internal documentation for benchmark-relevant
changes and current evidence.

- `docs/axiom-encode-methods-log.md` tracks the last meaningful harness changes,
  their hypotheses, and the evidence path to justify them later.
- `docs/rulespec-proof-validation.md` defines the proof-tree contract that keeps
  source claims, corpus anchors, and executable RuleSpec separated.
