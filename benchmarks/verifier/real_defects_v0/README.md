# Real defects v0

A triaged corpus of real encoding defects, taken from the fix history of the
`rulespec-us` and `rulespec-uk` repositories, for the EncodeBench verifier
track (a benchmark of statutory-fidelity judges).

Each case pairs the RuleSpec module as it stood before a correcting commit
with the module after it, plus the provision text the module cites, resolved
from a signed corpus release. Every byte in a case reproduces from Git and
from the release object; `scripts/verify_real_defects.py` checks that.

## Why this corpus exists

The 2026-09-17 judge pilot (`_axiom-runs/jev-judge-pilot-2026-09-17`) showed
that the calibration harness's labels are validation-gate outcomes a reader
cannot see, so they cannot score a fidelity judge, and that planted mutations
separate cleanly but are synthetic. `encodings.db` keeps no per-attempt
artifact versions, so the only real, localizable defects available today are
corrections that landed in Git after an encoding was produced.

## Counts

Enumeration on 2026-09-17 against `origin/main` of both repositories (rulespec-us at `7130d5412`, rulespec-uk at `d742817`).

| Stage | rulespec-us | rulespec-uk | Total |
| --- | ---: | ---: | ---: |
| Commits that changed a rule module | 1,142 | 207 | 1,349 |
| Keyword commits sent to triage | 158 | 21 | 179 |
| Non-keyword commits screened by subject | 984 | 186 | 1,170 |
| Commits the screen flagged | 141 | 10 | 151 |
| Commits triaged (diff read) | 299 | 31 | 330 |
| Module changes classified | 1094 | 162 | 1256 |
| Module changes kept after verification | 532 | 21 | 553 |
| Cases built | 505 | 15 | 520 |

First-reader classifications of the module changes: mechanical 606, not_a_rule_module 5, fidelity 522, unclear 118, test_only 5. Verifier verdicts on the kept ones: confirmed 453, not_a_defect 133, reclassify 30, unclear 19, mechanical 5. Dropped at merge: triage_mechanical 606, triage_not_a_rule_module 5, verifier_not_a_defect 84, verifier_mechanical 3, triage_test_only 5.

Dropped at build (33 of 553 kept rows): module_added_in_fix_commit 14, module_deleted_in_fix_commit 12, no_corpus_citation_path 4, defect_kind_missing 3.

Cases by kind. The representatives column counts one case per family (see `family_id`); the three generator regenerations account for the gap.

| Kind | US cases | UK cases | All cases | Family representatives |
| --- | ---: | ---: | ---: | ---: |
| `untraceable_branch` | 296 | 6 | 302 | 56 |
| `unrepresented_clause` | 74 | 5 | 79 | 79 |
| `wrong_entity_or_scope` | 76 | 0 | 76 | 31 |
| `wrong_period_or_effective_date` | 39 | 1 | 40 | 40 |
| `polarity_or_logic` | 14 | 0 | 14 | 14 |
| `amount_mismatch` | 4 | 2 | 6 | 6 |
| `boundary_direction` | 0 | 1 | 1 | 1 |
| `other` | 2 | 0 | 2 | 2 |
| Total | 505 | 15 | 520 | 229 |

Triage status: 450 `fidelity`, 70 `unclear`. Fix stage: 206 `post_merge`, 314 `pre_merge_review`. Artifacts: 304 cases ship their module and provision files; 216 inherited-verdict family members are metadata-only (hashes present, files reproducible with `--ship-artifacts all`). Provisions longer than 24,000 characters: 37 of 520.

## Layout

```
benchmarks/verifier/real_defects_v0/
  README.md
  index.json                 one entry per case (id, kind, confidence, ...)
  cases/<id>/
    case.json                the case record (schema below)
    pre_fix.yaml             module bytes at parent_commit      (when artifacts_shipped)
    post_fix.yaml            module bytes at commit             (when artifacts_shipped)
    provision.txt            provision text from corpus_release (when artifacts_shipped)
  triage/
    workflow_script.js       the screen/triage/verify workflow that read the diffs
    workflow_output.json     its return value (every reader and verifier output)
    triage_merged.json       one row per module touched, with keep/drop decisions
    screen.json              the screening pass output (flagged commits and reasons)
    summary.json             the yield counts
    build_log.json           rows dropped at build time and why
  tools/
    enumerate_candidates.py  module-touching commits and the keyword split
    prep_commit.py           read-only per-commit bundle (diffs, PR text) for readers
    merge_triage.py          workflow output to keep/drop rows
    build_real_defects.py    rows to cases, index, and build log
```

## Case schema

`cases/<id>/case.json` carries these keys. The verifier track reads this
schema; do not change key names without telling Max.

| Key | Meaning |
| --- | --- |
| `id` | `<jurisdiction>-<seq>-<commit8>-<module slug>`; also the directory name. |
| `jurisdiction` | `us` or `uk`. |
| `repo` | `TheAxiomFoundation/rulespec-us` or `TheAxiomFoundation/rulespec-uk`. |
| `commit` | Full sha of the correcting commit (post-fix state). On `origin/main`. |
| `parent_commit` | Full sha of the commit's first parent (pre-fix state). |
| `commit_date`, `commit_subject` | From the commit. |
| `pr_url` | The merged pull request that carried the commit, or null. |
| `module_path` | Repository path of the rule module. |
| `corpus_citation_path` | The module's `source_verification.corpus_citation_path` (first entry when plural) at the post-fix commit. `corpus_citation_paths_all` lists all of them. |
| `corpus_release`, `corpus_release_content_sha256` | The signed corpus release the provision was resolved from. |
| `corpus_commit` | The axiom-corpus commit that release names; provision files are read from it. |
| `defect_kind` | One of `amount_mismatch`, `boundary_direction`, `unrepresented_clause`, `untraceable_branch`, `wrong_period_or_effective_date`, `wrong_entity_or_scope`, `polarity_or_logic`, `other` (`other_kind` names it). |
| `confidence` | 0 to 1. The lower of the first reader's and the verifier's confidence, capped at 0.49 for `unclear` cases. |
| `description` | A verbatim quote from the commit message, PR text, or a comment in the module diff (`description_source` says which). Empty when nothing quotable existed. Never a paraphrase presented as a quote. |
| `locator.pre_fix_lines`, `locator.post_fix_lines` | Changed line ranges (`[start, end]`, 1-based, inclusive) from `git diff -U0 parent commit -- module`. |
| `locator.rule_names`, `locator.rule_path` | The rule(s) whose meaning changed and a YAML path to the decisive change, as read by the triage agent. |
| `pre_fix_artifact_sha256`, `post_fix_artifact_sha256` | sha256 of the shipped module bytes; must equal the Git blobs. |
| `provision_sha256`, `provision_chars` | sha256 and length of `provision.txt` (UTF-8, no added trailing newline). |
| `provision_resolution` | How the text was resolved: `mode` (`axiom_encode_resolver`, or `direct_row_exact` for release rows the current resolver rejects because they lack an `id`), `provision_file`, `provision_file_sha256`, `line_number`, `stored_body_sha256`, `resolved_text_sha256`, `slice_required`, `component_rows`, `selection_basis`, `fallback_index`, `toolchain_corpus_ref`, `fix_time_corpus_match`. |
| `fix_stage` | `post_merge` when the pre-fix module bytes were once on the first-parent history of `origin/main`; `pre_merge_review` when the correction landed on the branch before its pull request merged (the pre-fix state is the encoder's output as reviewed, never on main). |
| `artifacts_shipped` | `true` when `pre_fix.yaml`, `post_fix.yaml` and `provision.txt` are in the case directory. `false` for the inherited-verdict members of the generator families, which ship `case.json` only; the digests still verify from Git and the release, and `tools/build_real_defects.py --ship-artifacts all` writes the files. |
| `family_id`, `family_size`, `family_representative` | Cases that carry one correction applied to many modules (same commit, defect kind, and rule path) share a family; `family_size` is the member count and the representative is the family's first module path in sorted order. The three generator regenerations in rulespec-us PR #1300 each touched all 100 generated tariff-schedule chapter compositions, so a runner that wants independent cases should take representatives only (`counts.family_representatives` in `index.json`). |
| `triage_status` | `fidelity` (reader and verifier agree it is a fidelity correction of the stated kind) or `unclear` (kept for a reviewer to prune). |
| `triage` | The first reader's classification and notes, and the verifier's verdict, kind, confidence, justification, and quote. `candidate_source` is `keyword` or `screen`. When a triage chunk held more than three kept modules with the same kind and rule path (the generator families), the verifier read three and the others inherited a verified sibling's verdict; `verifier_inferred_from_module_index` names that sibling and is null for directly verified cases. |
| `triage_notes` | The reader's notes for a reviewer. |

### How the release is chosen

Modules do not name a corpus release; the repository's
`.axiom/toolchain.toml` at the fix commit does. `selection_basis` records the
rule that applied:

- `toolchain_release_pin`: the toolchain named a signed release
  (`axiom_corpus_release`); every UK case and every US case after 2026-08-18.
- `latest_release_at_or_before_toolchain_corpus_ref`: the toolchain named only
  an axiom-corpus commit (`axiom_corpus_ref`, US before 2026-08-18); the
  latest signed release whose corpus commit is that commit or an ancestor of
  it was used. `fix_time_corpus_match` then says whether the same provision
  row at the toolchain's corpus commit carries the same body (`same_body`),
  a different one (`different_body`), or was absent.
- `earliest_release_after_toolchain_corpus_ref` and `no_toolchain_corpus_pin`:
  fallbacks for commits older than the first signed release.

When the first candidate release could not resolve the citation, later
releases were tried in order; `fallback_index` and
`release_errors_before_success` record that.

## Method

1. Enumerate every commit on `origin/main` of both repositories that changed
   a rule module: a `.yaml` or `.yml` file under a jurisdiction root (`us`,
   `uk`, `us-xx`, `uk-<council>`) or the legacy `statutes`, `regulations`,
   `policies` roots, excluding companion tests (`*.test.yaml`), `.axiom`
   manifests, `data`, `tests`, and `programs` (compose specs are not
   encodings). `tools/enumerate_candidates.py`.
2. Split by subject keyword (fix, correct, wrong, repair, bug, finding,
   address, re-anchor, re-verdict, round, closeout, taper, boundary,
   time-bound, polarity, revert, amend, patch, resolve, review, and
   misspellings). Keyword commits go straight to triage. The rest go through
   a subject-only screening pass (batches of 70) that flags anything whose
   subject indicates a correction of an existing module, erring toward
   flagging.
3. Triage: one reader per commit (chunks of 12 modules for the two large
   commits) reads the commit message, the merged PR body and review
   comments, and the per-module diff, with the full pre and post files and
   the resolved provision text available on demand. It classifies every
   module as `fidelity`, `unclear`, `mechanical`, `test_only`, or
   `not_a_rule_module`, names the defect kind, quotes the stated reason, and
   locates the rule.
4. Verify: one adversarial reader per kept module tries to refute the
   classification and kind (`confirmed`, `reclassify`, `mechanical`,
   `not_a_defect`, `unclear`). When a chunk held more than three kept
   modules with the same kind and rule path (the generator families), three
   were read and the rest inherited a verified sibling's verdict, flagged in
   `triage.verifier_inferred_from_module_index`.
5. Merge (`tools/merge_triage.py`): modules the reader called mechanical,
   test-only, or not a rule module drop. Verifier `confirmed` keeps the
   reader's kind (an `unclear` reader call is promoted to `fidelity` when the
   verifier's confidence is 0.7 or more). Verifier `reclassify` at confidence
   0.6 or more keeps the module with the verifier's kind; below that the
   reader's kind stays and the case is `unclear`. Verifier `mechanical` or
   `not_a_defect` at confidence 0.7 or more drops the module; below that it
   stays as `unclear`. Verifier `unclear` or a missing verdict keeps the
   reader's classification. Confidence is the lower of the two readers',
   capped at 0.49 for anything that stays `unclear`.
6. Build (`tools/build_real_defects.py`): for each kept row, read the pre and
   post module bytes from Git, require `format: rulespec/v1` on both sides,
   take the module's citation path, choose the release as described above,
   resolve the provision through `axiom_encode.corpus_resolver` against a
   sparse corpus root materialized from the release object and the corpus
   commit, compute the locator from the diff hunks, and decide `fix_stage`.
   Rows that fail any step are listed in `triage/build_log.json`.

The screening readers were Claude Fable 5.1 agents; the triage and
verification readers were Claude Opus 5 agents (the Fable pools on two lanes
hit their usage limits mid-run, and bounded review work routes to Opus). All
ran through the workflow in `triage/workflow_script.js`; no gpt-5.5 model was
used anywhere.

## Exclusion rules

A module change is mechanical, and excluded, when only these changed:
proof hashes, import hashes or pins, toolchain pins, waiver or ledger
bookkeeping, index regeneration, formatting, renames, comments, summaries,
module-level metadata, or proof-atom excerpt and citation-path strings, while
formulas, parameters, versions, effective dates, inputs, outputs, and scope
stayed the same. Excerpt-only and citation-string-only corrections are
provenance repairs, not fidelity defects.

Also excluded at build time: modules added or deleted by the fix commit,
modules that are not `format: rulespec/v1` on both sides (the December 2025
to April 2026 pre-RuleSpec formats), modules with no
`corpus_citation_path`, and modules whose citation resolves in no signed
release.

Companion-test-only changes yield no case: the corpus needs a module whose
bytes changed.

## Known limits

- Post-fix is not guaranteed clean. A later commit may correct the same
  module again; the Worthing council tax reduction module was corrected
  three times, and one of those corrections flipped a boundary back. Judges
  should be scored on whether they flag the pre-fix defect, not on whether
  they pass the post-fix artifact.
- Provision windows can exceed the judge's 24,000-character truncation
  (`DEFAULT_PROVISION_CHARS` in `src/axiom_encode/judges/client.py`):
  37 of 520 cases do. `provision_chars` is in `index.json` so a runner
  can filter or window.
- Some fixes span several modules; one case per module per commit means the
  quoted description may describe the whole commit, not the module alone.
  The generator regenerations are the extreme: one hunk shape across 100
  generated compositions per commit. Family metadata marks them; the counts
  section reports both raw cases and family representatives.
- The corpus release used is the toolchain's pin at the fix commit, or the
  nearest earlier signed release for pre-pin US commits.
  `fix_time_corpus_match` tells you when the resolved text was verified to
  equal the row the encoder saw at the toolchain's corpus commit.
- `direct_row_exact` cases carry the row's verbatim body without descendant
  composition or slicing, because the current resolver refuses rows with no
  `id`; those rows are US state material from the 2026-07-13 recovery scope.
- The locator is line ranges from the diff plus the reader's rule path; a
  judge that returns a rule path can be matched on `locator.rule_names`, a
  judge that returns lines on the ranges.
- Pull request review threads on GitHub were empty for the sampled commits;
  review findings live in commit messages (`sol r2 findings`, `Address sol
  round-1`). `description_source` says where each quote came from.
- The screen read subjects only, so a correction with a bland subject and
  no keyword can have been missed. The keyword pass and the screen together
  covered every module-touching commit on both main branches at the
  enumeration date.

## Rebuild

From the axiom-encode checkout, with the rulespec and corpus checkouts as
siblings (`git fetch origin` in each first) and network access to the public
release registry (the same `NEXT_PUBLIC_SUPABASE_URL` and anon key the org
validate-rulespec workflow reads; the scripts fall back to
`gh variable get` on `TheAxiomFoundation/rulespec-us`):

```bash
uv run python benchmarks/verifier/real_defects_v0/tools/enumerate_candidates.py us ../rulespec-us /tmp/real-defects-enum
uv run python benchmarks/verifier/real_defects_v0/tools/enumerate_candidates.py uk ../rulespec-uk /tmp/real-defects-enum
```

Re-run the triage workflow in `triage/workflow_script.js` (or reuse the
recorded `triage/workflow_output.json`; `--inputs-dir` is where
`tools/prep_commit.py` wrote the per-commit bundles, which supply the
merged PR URL), then:

```bash
uv run python benchmarks/verifier/real_defects_v0/tools/merge_triage.py \
  --workflow-output benchmarks/verifier/real_defects_v0/triage/workflow_output.json \
  --inputs-dir /tmp/real-defects-inputs --corpus-dir benchmarks/verifier/real_defects_v0
uv run python benchmarks/verifier/real_defects_v0/tools/build_real_defects.py \
  --corpus-dir benchmarks/verifier/real_defects_v0 \
  --rulespec-us ../rulespec-us --rulespec-uk ../rulespec-uk \
  --axiom-corpus ../axiom-corpus --release-cache ~/.cache/axiom-real-defects
```

Verify (each tier is skipped, and reported as skipped, when its inputs are
absent):

```bash
uv run python scripts/verify_real_defects.py \
  --corpus-dir benchmarks/verifier/real_defects_v0 \
  --rulespec-us ../rulespec-us --rulespec-uk ../rulespec-uk \
  --axiom-corpus ../axiom-corpus \
  --with-release --release-cache ~/.cache/axiom-real-defects
uv run pytest -q tests/test_real_defects_corpus.py
```

The test runs the shipped-file tier always and the Git tiers when the
sibling checkouts exist (`AXIOM_REAL_DEFECTS_RULESPEC_US`,
`AXIOM_REAL_DEFECTS_RULESPEC_UK`, `AXIOM_REAL_DEFECTS_AXIOM_CORPUS` override
the locations).
