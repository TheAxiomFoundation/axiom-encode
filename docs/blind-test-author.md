# Independent (blind) test author

Status: **shadow tool**. It reports. It gates nothing. It declares no check in
`axiom_encode.checks`, `encode` does not call it, and no validation result
depends on it. The proposal behind it (a design note of 2026-11-09) is not yet
decided by the maintainers; this tool exists to measure whether the idea is
worth deciding on.

## The problem it addresses

The model that encodes a rule also writes its companion `.test.yaml`, from its
own formula. Those tests show that the rule agrees with itself. They cannot
show that the encoder read the source correctly. The deterministic checks
(`complete-source-unit:*`, proof excerpts) show that the module accounts for
the source text; they do not show that the formula means what the text means.

Three cases from 2026-11-12:

- A candidate for
  `us-az/manual/des/faa5/na-utility-expenses-and-allowances/block-16` passed
  every machine check with a start date (`2025-10-01`) that the source, a
  snapshot of 2026-07-17, does not support.
- Candidate rulespec-us#2057 passed and conflicts with 7 CFR 273.9(d)(3).
- On rulespec-us#2046 and #2047 a human reviewer asked for a July boundary
  case that the encoder's own tests did not have.

## What the tool does

For one module:

1. **Source text.** It resolves the module's
   `module.source_verification.corpus_citation_path` through the RuleSpec
   checkout's pinned corpus release, with the encoder's own code path
   (`toolchain.load_rulespec_local_corpus_release`, then
   `corpus_resolver.resolve_local_corpus_source`). `--source-text-file` gives
   the text directly when no corpus checkout is at hand.
2. **Interface.** `extract_rule_interface` copies an allow-list from the
   RuleSpec YAML: for each `inputs` entry and each `derived` or `parameter`
   rule, the name and the single-word declarations `kind`, `dtype`, `entity`,
   `period`, `unit`, plus enumeration names. Some modules use boolean facts
   without declaring them; for those, the names (only) come from the compiled
   program.
3. **Blindness guard.** Before any model call, `find_blindness_violations`
   checks the part of the prompt that is not corpus text against the module:
   every formula (whole and by line), every number in a formula, every
   parameter value, every `effective_from` / `effective_to`, the module's
   prose (summary, source labels, proof metadata) and the names of the
   existing test cases. One hit and the tool stops with an error.
4. **Second model.** The backend is an interface with two implementations:
   `FakeBlindAuthorBackend` (tests) and `ClaudeCliBlindAuthorBackend`, which
   runs `claude --print` on the caller's claude.ai subscription login. It
   reuses the local reviewer's launcher from
   `harness/validator_pipeline.py`: the same executable resolution, the same
   explicit-allowlist environment (no ambient API key or token), the same
   tool-less flags. It never falls back to the Codex CLI, because generation
   uses a GPT model and the second author should be another family. The
   script refuses to call a model when a provider key is in the environment.
5. **Typed answer.** The answer must be one JSON object: cases with `name`,
   `period` (`YYYY-MM`), `inputs`, `expected` and `source_sentence`. No prose
   is parsed. A case is rejected, and reported, when its keys or value types
   are wrong, when it names an input or output that is not in the interface,
   or when its `source_sentence` is not in the source text character for
   character (only runs of whitespace are folded).
6. **Run.** Each case runs as a one-case companion test through
   `ValidatorPipeline._run_rulespec_test_cases` on the real rules engine, the
   runner used for `.test.yaml`. Per case the result is:
   - `agree`: the candidate returns the expected outputs.
   - `disagree`: the case ran, and an output differs.
   - `not_runnable`: the case did not run (binding, a missing fact, or
     `no_formula_version_at_period`). This is never counted as a disagreement.
7. **Period probe.** An output case cannot see a rule that starts too early.
   The answer therefore also carries `earliest_supported_period` with its
   basis: a verbatim sentence, or `source_as_of`, which means the first full
   month after the corpus as-of date. That fallback follows the comment in the
   landed block-13 module; it is an assumption of this tool, not a decided
   rule. The tool runs one case again one month earlier. If the candidate
   still computes, the report says
   `candidate_applies_before_blind_earliest_period`.
8. **Report.** `report.json` (schema `axiom-encode/blind-test-author-report/v1`:
   interface, prompt digest, every case with its sentence, status, issues and
   compared values, rejected cases, the period probe, the raw answer) and
   `summary.txt`. The exit status is 0 whatever the cases say; 2 means the
   tool could not run.

`--print-prompt` writes the exact prompt and stops before any model call.
`--replay-report` runs the stored answer of an earlier report against another
candidate without a model call.

## Limits

- **The second model can misread too.** A disagreement means "two readings
  differ; a person should look", not "the rule is wrong". Agreement of two
  models is evidence, not proof: both can make the same mistake, most of all
  where the text is ambiguous.
- **Names leak intent.** The interface carries the encoder's names. A name
  such as `component_is_one_time_deposit` tells the second model which
  conditions the encoder found, and a wrong reading can sit in a name. The
  set and number of inputs, the entity and the period leak too. The prompt
  tells the model to follow the text when a name disagrees with it.
- **Missing inputs are invisible.** If the encoder left out a condition, the
  blind author has no input to express it with. It can only skip the case.
  Completeness stays the job of the `complete-source-unit:*` checks.
- **The source unit is the encoder's choice.** The second model sees the same
  corpus slice. A conflict with another provision (the rulespec-us#2057 kind)
  is out of its sight unless that provision is in the slice.
- **One month periods, flat facts.** Version 1 has no relation inputs, no
  row-ordered outputs and no year periods. Modules that need them give
  `not_runnable` cases.
- **Verbatim is not relevant.** The citation check proves that the sentence
  exists, not that it supports the case.
- **One sample.** The model is not deterministic. One run is one draw.

## Measurement plan

Run by a person, after the local A/B has ended (8:00am EST on 2026-11-13).
No organization API key: the model path is the `claude` CLI on the claude.ai
subscription. Results go to `~/axiom-encode-local-ab/2026-11-13-blind-test-author/`,
never into git (the script refuses an `--out-dir` inside this repository or
the RuleSpec checkout).

Expected runtime: one `claude --print` call per module (about 1 to 3 minutes
with the default model `opus`; timeout 600 s), then about 3 s to compile and
under 1 s per case. Part (a) is about 15 to 30 minutes for 10 modules.

### Setup

```bash
set -eu
R=~/axiom-encode-local-ab/2026-11-13-blind-test-author
BATCH=/Users/pavelmakarchuk/az-utility-batch.Z1VWxK   # engine and corpus copies of the Arizona batch
ENGINE=$BATCH/engine                                   # built, at 89571cc2
CORPUS=$BATCH/corpus                                   # at 7822462a
KEY=$(gh variable get AXIOM_CORPUS_RELEASE_PUBLIC_KEY --repo TheAxiomFoundation/axiom-encode)
ENC=<checkout of this branch, after `uv sync --python 3.13 --all-extras --all-groups`>
AZ=us-az/policies/des/faa5
mkdir -p "$R/checkouts/main" "$R/results"
git clone -q https://github.com/TheAxiomFoundation/rulespec-us.git "$R/checkouts/main/rulespec-us"
cd "$ENC"
blind() {  # blind <label> <rulespec-us checkout> <module path> [more options]
  label=$1; checkout=$2; module=$3; shift 3
  env -i HOME="$HOME" USER="$USER" TMPDIR="${TMPDIR:-/tmp}" PATH="$PATH" \
    uv run python scripts/blind_test_author.py \
      --rules-file "$checkout/$module" \
      --axiom-rules-engine-path "$ENGINE" \
      --corpus-path "$CORPUS" --corpus-release-public-key "$KEY" \
      --out-dir "$R/results/$label" "$@"
}
```

Check the prompt of one module first; this makes no model call:

```bash
blind block-13-prompt "$R/checkouts/main/rulespec-us" $AZ/na-utility-expenses-and-allowances/block-13.yaml --print-prompt
less "$R/results/block-13-prompt/prompt.txt"
```

### (a) The 10 Arizona modules

```bash
# 1. Landed module (accepted; a disagreement here is a false disagreement or a real find)
blind block-13 "$R/checkouts/main/rulespec-us" $AZ/na-utility-expenses-and-allowances/block-13.yaml

# 2-9. Open candidates, each at its pull request head
while read -r pr module; do
  git -C "$R/checkouts/main/rulespec-us" fetch -q origin "pull/$pr/head"
  mkdir -p "$R/checkouts/pr-$pr"
  git -C "$R/checkouts/main/rulespec-us" worktree add -q --detach "$R/checkouts/pr-$pr/rulespec-us" FETCH_HEAD
  blind "pr-$pr" "$R/checkouts/pr-$pr/rulespec-us" "$AZ/$module"
done <<'LIST'
2046 na-medical-expenses-and-deduction/block-7.yaml
2047 na-medical-expenses-and-deduction/block-11.yaml
2048 na-medical-expenses-and-deduction/block-12.yaml
2197 na-utility-expenses-and-allowances/block-10.yaml
2198 na-utility-expenses-and-allowances/block-15.yaml
2199 na-utility-expenses-and-allowances/block-7.yaml
2200 na-utility-expenses-and-allowances/block-11.yaml
2201 na-utility-expenses-and-allowances/block-16.yaml
LIST

# 10. The WITHHELD block-16 attempt (start date 2025-10-01 that the source does not support)
G=$BATCH/block-16-attempt-1/generated
mkdir -p "$R/checkouts/withheld-16"
git -C "$R/checkouts/main/rulespec-us" worktree add -q --detach "$R/checkouts/withheld-16/rulespec-us" origin/main
cp "$G"/codex-gpt-6.1-sol/policies/des/faa5/na-utility-expenses-and-allowances/block-16.yaml \
   "$G"/codex-gpt-6.1-sol/policies/des/faa5/na-utility-expenses-and-allowances/block-16.test.yaml \
   "$R/checkouts/withheld-16/rulespec-us/$AZ/na-utility-expenses-and-allowances/"
blind withheld-16 "$R/checkouts/withheld-16/rulespec-us" $AZ/na-utility-expenses-and-allowances/block-16.yaml

# One table
uv run python - "$R/results" <<'PY'
import json, pathlib, sys
for path in sorted(pathlib.Path(sys.argv[1]).glob("*/report.json")):
    r = json.loads(path.read_text()); c = r["counts"]
    print(f"{path.parent.name:14} agree={c['agree']} disagree={c['disagree']} "
          f"not_runnable={c['not_runnable']} rejected={c['rejected']} "
          f"period={r['period_probe']['finding']} error={r['response_error']}")
PY
```

If the corpus release does not load for a checkout, give the text directly:
replace the two `--corpus-*` options with `--source-text-file <file>
--source-as-of 2026-07-17` (for the withheld attempt the encoder's exact text
is `$G/_eval_workspaces/codex-gpt-6.1-sol/us-az-manual-des-faa5-na-utility-expenses-and-allowances-block-16/workspace/source.txt`).

Questions for part (a), answered by reading each `summary.txt`:

- Withheld block-16: is `period_probe.finding` equal to
  `candidate_applies_before_blind_earliest_period`? That is the "caught the
  unsupported date" result. Compare with `pr-2201`, the later block-16.
- #2046 and #2047: does the blind author write the July boundary case the
  human reviewer asked for, and does the candidate agree?
- block-13 (landed): every `disagree` is read by a person and labelled
  "blind author wrong" or "module wrong".
- For each module: how many cases were rejected or not runnable. A tool
  whose cases mostly do not run measures nothing.

### (b) Stored defect pairs

The repository holds `benchmarks/verifier/real_defects_v0/`: 171 real defect
cases (158 US, 13 UK), each with `pre_fix.yaml`, `post_fix.yaml`,
`provision.txt` and `case.json`; 119 are marked
`judgeable_from_provision: true` (106 US). The 176 planted pairs of the
synthetic board (`benchmarks/verifier/boards/synthetic_us_v1/`) are stored as
a manifest only; their texts are rebuilt from `encodings.db` with
`benchmarks/verifier/verifier.py build-synthetic` (see
`docs/encodebench-verifier.md`), so they come second.

Per real case: author the blind cases once on the accepted (post-fix) module,
then replay the same answer on the pre-fix module. Start with a pilot of 10
cases; the engine build that compiles a historical commit may differ from
`$ENGINE`, and that is not verified yet.

```bash
uv run python - <<'PY' > "$R/defect-cases.tsv"
import json
index = json.load(open("benchmarks/verifier/real_defects_v0/index.json"))
for case in index["cases"]:
    if case["jurisdiction"] == "us" and case["judgeable_from_provision"]:
        print(case["id"], case["commit"], case["parent_commit"], case["module_path"], sep="\t")
PY

head -10 "$R/defect-cases.tsv" | while IFS=$'\t' read -r id fix parent module; do
  src=benchmarks/verifier/real_defects_v0/cases/$id/provision.txt
  for side in post:$fix pre:$parent; do
    mkdir -p "$R/checkouts/defects/$id/${side%%:*}"
    git -C "$R/checkouts/main/rulespec-us" worktree add -q --detach \
      "$R/checkouts/defects/$id/${side%%:*}/rulespec-us" "${side##*:}"
  done
  env -i HOME="$HOME" USER="$USER" TMPDIR="${TMPDIR:-/tmp}" PATH="$PATH" \
    uv run python scripts/blind_test_author.py \
      --rules-file "$R/checkouts/defects/$id/post/rulespec-us/$module" \
      --axiom-rules-engine-path "$ENGINE" --source-text-file "$src" \
      --out-dir "$R/results-defects/$id/post"
  env -i HOME="$HOME" USER="$USER" TMPDIR="${TMPDIR:-/tmp}" PATH="$PATH" \
    uv run python scripts/blind_test_author.py \
      --rules-file "$R/checkouts/defects/$id/pre/rulespec-us/$module" \
      --axiom-rules-engine-path "$ENGINE" --source-text-file "$src" \
      --replay-report "$R/results-defects/$id/post/report.json" \
      --out-dir "$R/results-defects/$id/pre"
done
```

Expected runtime: one model call per case (the replay makes none), so about
20 to 40 minutes for the pilot and 3 to 6 hours for all 106 US cases.

Metrics, computed from the paired `report.json` files:

- **Share of known defects caught** = cases where at least one blind case is
  `disagree` on `pre` and `agree` on `post`, divided by the cases where at
  least one blind case ran on both sides. Report the cases with no runnable
  blind case separately; they are neither caught nor missed.
- **False-disagreement rate on accepted modules** = `disagree` cases on
  `post` divided by the blind cases that ran on `post` (agree + disagree),
  pooled over modules, and also the share of accepted modules with at least
  one `disagree`. Part (a) block-13 and any landed module add to this set.
  A person labels each `post` disagreement, because an accepted module can
  still be wrong.
- **Usable-case rate** = (agree + disagree) / authored cases. Low values
  mean the typed format or the interface is the bottleneck, not the reading.

The pre-fix interface can differ from the post-fix interface (a fix can add
an input). A replayed case that names a missing input is rejected on `pre`;
count those separately.

## What is verified, and what is not

Verified with the fake model and the real rules engine (`tests/test_blind_test_author.py`):
no formula, value, date, prose or existing test reaches the prompt; a
sentence that is not in the source is rejected; correct blind cases agree
with a correct rule; a planted negated condition is caught; a start date
before the supported period is flagged by the period probe; the CLI backend
builds the reviewer's command and environment (subprocess replaced).

Not verified, because no model call has been made: that the `claude` CLI
returns the typed JSON reliably; how often real answers are rejected or not
runnable; whether a real second model catches the block-16 date or any
stored defect; the false-disagreement rate; the runtimes above.
