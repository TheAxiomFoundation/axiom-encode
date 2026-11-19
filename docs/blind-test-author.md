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
  every machine check with a start date (`2025-10-01`) that its own source
  does not state; the date was copied from a neighbouring module
  (axiom-encode#2212).
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
   rule, the name and the declaration words `kind`, `dtype`, `entity`,
   `period`, `unit`, plus enumeration names (never for a parameter).
   `kind`, `dtype` and `period` are closed vocabularies; `unit` is a
   three-letter currency code or one of a few unit words; `entity` is one
   CamelCase word. The word lists come from a scan of rulespec-us main
   (4,886 modules, 2026-11-13), and include `Number`, `Hour`,
   `day` / `year` / `month` / `calendar_day` and the shape `Enum[Name]`. Any other value in such a field is dropped, and the report
   lists the field under `interface_dropped_fields`. Some modules use boolean
   facts without declaring them; for those, the names (only) come from the
   compiled program. A candidate that does not compile stops the tool before
   any model call.
3. **Blindness guard.** Before any model call, two checks run on the part of
   the prompt that is not corpus text:
   - `find_blindness_violations`: every formula (whole and by line), the
     module's prose (summary, source labels, proof metadata) and the names of
     the existing test cases (unless a test is named like an interface name).
     One hit stops the tool.
   - `find_interface_leaks`: every number of two or more digits (or with a
     decimal part) that a formula or a declaration uses, and every effective
     date, searched inside names, enumeration members, the module path and
     the citation path, with `_` and other separators split. A name such as
     `limit_is_1500_usd` or `starts_2026_08_01` is a hit. So is a harmless
     coincidence such as `fy_2026_...` or `block-13` beside a formula that
     uses 13. The tool cannot tell these apart, so it does not pass or fail
     them in silence: it stops, names each hit, and continues only with
     `--allow-interface-leaks`. The report then records every hit under
     `interface_leaks`, and the summary repeats them.

   The citation path must be an exact canonical corpus citation path
   (`corpus_resolver.require_canonical_corpus_citation_path`) in the
   resolver's segment alphabet (letters, digits, `.`, `:`, `-`, en dash, and
   at most three words in a segment, as in `us-la/statute/47:294` and
   `.../7 AAC 45.280`); a sentence is refused.

   When leaks were allowed, the run is not fully blind and says so: the
   report's top-level `blindness` and the first line of the summary read
   `NOT FULLY BLIND (n leaks allowed)`. The corpus as-of date must be `YYYY-MM-DD`.
4. **Second model.** The backend is an interface with two implementations:
   `FakeBlindAuthorBackend` (tests) and `ClaudeCliBlindAuthorBackend`, which
   runs `claude --print` on the caller's claude.ai subscription login. It
   reuses the local reviewer's launch rules from
   `harness/validator_pipeline.py`: the same executable resolution, the same
   explicit-allowlist environment (no ambient API key or token), the same
   tool-less flags. Three differences are deliberate. It never falls back to
   the Codex CLI, because generation uses a GPT model and the second author
   should be another family. The child runs in a fresh, empty temporary
   directory, so the CLI's own context (working directory, git status) cannot
   describe the candidate. Only stdout is parsed; stderr goes to the report
   (`backend.stderr`), because the shared launcher joins the two and one CLI
   warning would make the answer unparseable. The script refuses to call a
   model when a provider key is in the environment.
5. **Typed answer.** The answer must be one JSON object: cases with `name`,
   `period` (`YYYY-MM`), `inputs`, `expected` and `source_sentence`. No prose
   is parsed. A case is rejected, and reported, when:
   - its keys or value types are wrong, or a number is not finite;
   - it names an input or output that is not in the interface;
   - it has the same inputs, period and expected outputs as an earlier case
     (the same case under two names counts once);
   - its `source_sentence` is not in the source text character for character
     (only runs of whitespace are folded);
   - its `source_sentence` is cut. The quote must start and end at a
     boundary of the source. A start boundary is the start of an item (a
     non-empty source line: a paragraph or a list item), the text after the
     item's marker (a bullet, an enumerator such as `(a)` or `1.`, a label
     such as `NOTE`), or the text after a sentence end. An end boundary is
     the end of an item, or a sentence end inside it (`.`, `?` or `!`
     before whitespace). A semicolon is not an end ("The deduction is
     allowed; except that ..."). A period after an abbreviation is not an end
     (`U.S.C.`, `C.F.R.`, `Pub. L. No.`, `e.g.`, `i.e.`, `et seq.`, `Sec.`,
     a single letter or dotted initials, a section sign). A quote may span
     several items. A quote that starts or stops inside a sentence is
     refused, because it can drop the condition that the sentence states.

   The report gives two counts under `quote_stats`, and the summary repeats
   them when they are not zero:
   - accepted cases whose quote is shorter than the full source item it sits
     in (one sentence of a longer paragraph);
   - accepted cases that quote a list item without its chapeau. A list item
     that follows an item ending in `:` depends on it ("... when all of the
     following occur:"). Such a quote is whole, so the first count does not
     see it; it still leaves out the governing sentence.
6. **Run.** Each case runs as a one-case companion test through
   `ValidatorPipeline._run_rulespec_test_cases` on the real rules engine, the
   runner used for `.test.yaml`. Per case the result is:
   - `agree`: the candidate returns the expected outputs.
   - `disagree`: an output was compared and differs. Other issues of the
     same case are listed beside it (`other_issues`); they do not hide it.
   - `not_runnable`: no output mismatch, and the case did not run (binding,
     a missing fact, or `no_formula_version_at_period`). This is never
     counted as a disagreement.
7. **Period probes.** An output case cannot see a rule that starts too early.
   The tool therefore runs one case that ran again, one month before each of
   two reference months, and reports whether the candidate still computes.
   - `assumed_evidence_floor`. The reference month is the first full calendar
     month that starts on or after the corpus as-of date. The tool computes
     it; the model is not asked and cannot move it. A hit is
     `before_assumed_evidence_floor`. **This is an assumption, not a rule.**
     What the start date of a unit that states no date should be is an open
     maintainer decision, [axiom-encode#2212](https://github.com/TheAxiomFoundation/axiom-encode/issues/2212).
     By this convention several hundred modules on rulespec-us main start
     "early" (441 to 962 in the measurement of that issue). The report marks
     the probe `"assumption": true` with the issue reference, and the summary
     line says "ASSUMPTION, not a defect finding". Whether a hit is a defect
     depends on the decision in #2212.
   - `source_stated_date`. The second model may give `stated_commencement`:
     a month and the whole verbatim sentence of the source that states the
     date. The sentence must pass the quote rule and carry a full date or a
     month-year phrase in that month ("March 1, 2027", "March 2027",
     "2027-03-01"). A bare year, a fiscal-year label ("FY 2026") or a date
     in a citation is reported as "year mentioned, not a stated date" and
     raises no finding. A hit
     is `before_source_stated_date`. It rests on the model's reading of that
     sentence, which the report quotes.
8. **Report.** `report.json` (schema `axiom-encode/blind-test-author-report/v1`:
   interface, dropped fields, interface leaks, prompt digest, every case with
   its sentence, status, issues and compared values, rejected cases, quote
   statistics, the period probes, the raw answer, the CLI's stderr) and
   `summary.txt`. The exit status is 0 whatever the cases say; 2 means the
   tool could not run or refused to.

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
- **Verbatim is not relevant.** The citation check proves that a whole
  sentence or item exists, not that it supports the case.
- **Names can still carry meaning.** The leak check finds numbers and dates
  that the formulas use. It does not find a value written in words
  (`fifteen_hundred`), a one-digit number, or a number the formulas do not
  use. With `--allow-interface-leaks` the leak reaches the model; the report
  says so, and a result from such a run is weaker evidence of independence.
- **The floor probe measures an assumption.** See step 7 and #2212.
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
      --out-dir "$R/results/$label" "$@" </dev/null
}
```

If a run stops with "a name or path in the prompt carries a formula value or
a date", read the named hits. For a coincidence (`block-10` and a formula
that uses 10) or a leak you accept, run the same command again with
`--allow-interface-leaks` appended; the report records the hits. If a run
stops with "does not compile", the candidate needs another engine build; no
model call was made.

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
          f"floor_probe(ASSUMPTION #2212)={r['period_probes']['assumed_evidence_floor']['finding']} "
          f"stated_date_probe={r['period_probes']['source_stated_date']['finding']} "
          f"leaks={len(r['interface_leaks'])} error={r['response_error']}")
PY
```

If the corpus release does not load for a checkout, give the text directly:
replace the two `--corpus-*` options with `--source-text-file <file>
--source-as-of 2026-07-17` (for the withheld attempt the encoder's exact text
is `$G/_eval_workspaces/codex-gpt-6.1-sol/us-az-manual-des-faa5-na-utility-expenses-and-allowances-block-16/workspace/source.txt`).

Questions for part (a), answered by reading each `summary.txt`:

- Withheld block-16: what do the two period probes say? A
  `before_assumed_evidence_floor` here means only that the probe flags a
  start before the ASSUMED floor (2026-08, from the as-of date 2026-07-17);
  whether that is a defect depends on the decision in #2212. It is not
  "caught the unsupported date". A `before_source_stated_date` would be a
  claim by the second model, to check against its quoted sentence; block 16
  states no date, so `not_probed` is the expected value. Compare with
  `pr-2201`, the later block-16.
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
# id, fix commit, parent commit, module path, provision file (106 rows).
# The parent commit is in each cases/<id>/case.json, not in index.json.
uv run python scripts/blind_test_author.py \
  --list-defect-cases benchmarks/verifier/real_defects_v0 > "$R/defect-cases.tsv"

head -10 "$R/defect-cases.tsv" | while IFS=$'\t' read -r id fix parent module src; do
  for side in post:$fix pre:$parent; do
    mkdir -p "$R/checkouts/defects/$id/${side%%:*}"
    git -C "$R/checkouts/main/rulespec-us" worktree add -q --detach \
      "$R/checkouts/defects/$id/${side%%:*}/rulespec-us" "${side##*:}"
  done
  env -i HOME="$HOME" USER="$USER" TMPDIR="${TMPDIR:-/tmp}" PATH="$PATH" \
    uv run python scripts/blind_test_author.py \
      --rules-file "$R/checkouts/defects/$id/post/rulespec-us/$module" \
      --axiom-rules-engine-path "$ENGINE" --source-text-file "$src" \
      --out-dir "$R/results-defects/$id/post" </dev/null
  env -i HOME="$HOME" USER="$USER" TMPDIR="${TMPDIR:-/tmp}" PATH="$PATH" \
    uv run python scripts/blind_test_author.py \
      --rules-file "$R/checkouts/defects/$id/pre/rulespec-us/$module" \
      --axiom-rules-engine-path "$ENGINE" --source-text-file "$src" \
      --replay-report "$R/results-defects/$id/post/report.json" \
      --out-dir "$R/results-defects/$id/pre" </dev/null
done
```

The listing is tested against the stored index (`tests/test_blind_test_author.py`).
The worktree and run steps are not tested: they need the rulespec-us history
and, for the first run of each pair, a model call. A pair whose module does
not compile on `$ENGINE` stops with exit 2 and makes no model call.

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

Verified with the fake model (`tests/test_blind_test_author.py`; the cases
that execute rules use the real rules engine):

- no formula, value, date, prose or existing test reaches the prompt; free
  text in a declaration field is dropped; a free-text citation path is
  refused; a value or date inside a name is refused, or reported when the
  caller allows it;
- a quote that is not in the source, or is cut inside a sentence, is
  rejected; a repeated case and a non-finite number are rejected;
- correct blind cases agree with a correct rule; a planted negated condition
  is caught; a mismatch beside another issue is a disagreement;
- the floor probe uses the month computed from the as-of date only, and is
  marked as an assumption;
- the CLI backend builds the reviewer's command and environment, runs in an
  empty directory and parses stdout only (subprocess replaced);
- the tool stops before a model call when the candidate does not compile.

Not verified, because no model call has been made: that the `claude` CLI
returns the typed JSON reliably; how often real answers are rejected (the
quote boundary rule may reject honest quotes) or not runnable; how often real
modules trip the interface-leak check (an offline scan by the round-2
reviewer of 4,056 rulespec-us main modules gave 121 leak stops and 17
vocabulary violations, 3.4%, before the round-2 vocabulary additions); whether a real second model catches
any stored defect; the false-disagreement rate; the runtimes above.
