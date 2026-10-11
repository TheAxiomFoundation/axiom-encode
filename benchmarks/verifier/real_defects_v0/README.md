# Real defects v0

A triaged corpus of real encoding defects, taken from the fix history of the
`rulespec-us` and `rulespec-uk` repositories, for the EncodeBench verifier
track (a benchmark of statutory-fidelity judges).

Each case pairs the RuleSpec module as it stood before a correcting commit
with the module after it, plus the provision text the module cites, resolved
from a signed corpus release. Every byte in a case reproduces from Git and
from the release object; `scripts/verify_real_defects.py` checks that.

Each case also records whether a judge can decide it from that provision.
Two readers checked every case against its provision and quoted the text
that shows the pre-fix module wrong; where that text sat under another
citation, the provision was extended with it, and where it sat in none of
the module's sources the case is marked `judgeable_from_provision: false`.
The readers were models, not people; see "Provision review" under Method.

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

Everything below this line is written by `tools/build_real_defects.py` from
`index.json`, `triage/summary.json` and `triage/build_log.json`; a test fails
if it drifts from them.

<!-- begin generated counts: tools/build_real_defects.py; edit the build, not this -->

Module changes classified by the triage readers: 1,256 (fidelity 522, mechanical 606, not_a_rule_module 5, test_only 5, unclear 118). Verifier verdicts on the candidates: confirmed 453, mechanical 5, not_a_defect 133, reclassify 30, unclear 19. Dropped at merge: triage_mechanical 606, triage_not_a_rule_module 5, triage_test_only 5, verifier_mechanical 3, verifier_not_a_defect 84. Kept after verification: 553.

Dropped at build (33 of 553 kept rows): defect_kind_missing 3, module_added_in_fix_commit 14, module_deleted_in_fix_commit 12, no_corpus_citation_path 4. Cases built: 520.

Removed by the build flags (`--drop-unclear`: on; `--family-members representatives`): family_non_representative 280, triage_unclear 69. Cases in the corpus: **171** (158 US, 13 UK), in 171 families.

| Kind | US | UK | All | Judgeable from the provision |
| --- | ---: | ---: | ---: | ---: |
| `amount_mismatch` | 3 | 1 | 4 | 3 |
| `boundary_direction` | 0 | 1 | 1 | 1 |
| `unrepresented_clause` | 59 | 5 | 64 | 51 |
| `untraceable_branch` | 32 | 6 | 38 | 29 |
| `wrong_period_or_effective_date` | 33 | 0 | 33 | 17 |
| `wrong_entity_or_scope` | 23 | 0 | 23 | 15 |
| `polarity_or_logic` | 7 | 0 | 7 | 2 |
| `other` | 1 | 0 | 1 | 1 |
| Total | 158 | 13 | 171 | 119 |

How the corpus release was chosen (`selection_basis`): earliest_release_after_toolchain_corpus_ref 18, latest_release_at_or_before_toolchain_corpus_ref 112, no_toolchain_corpus_pin 36, toolchain_release_pin 5.

Triage status: fidelity 171. Fix stage: post_merge 74, pre_merge_review 97. Artifacts: 171 cases ship their module and provision files, 0 are metadata-only.

Provision review (171 cases reviewed; 0 board-eligible cases not reviewed):

| Where the decisive text sits | Cases | Outcome |
| --- | ---: | --- |
| In the packaged provision (`in_provision`) | 67 | kept as built |
| Under another citation (`in_other_citation`) | 52 | provision extended with that citation |
| In none of the cited sources (`not_in_sources`) | 52 | `judgeable_from_provision: false` |

Board-eligible cases (shipped family representatives with `triage_status: fidelity`): 171. Judgeable from the provision: 119. Of those, 100 keep their decisive text inside the judges' default 24,000-character window. Provisions longer than that window: 37 of 171 shipped.

How the reviews were settled: adjudicated 13, readers_agree 158. The readers' call on whether the bundle's sources confirm the defect (`defect_real`): no 4, unsure 48, yes 119.

The mechanical check (`tools/check_evidence.py`) against the review, on each reviewed case's first-citation provision:

| Mechanical result | `in_provision` | `in_other_citation` | `not_in_sources` |
| --- | ---: | ---: | ---: |
| `present` | 66 | 20 | 25 |
| `absent` | 1 | 32 | 27 |

Mechanical `evidence_in_provision` on the provisions as shipped (after extension): present 143, absent 28.

<!-- end generated counts -->

## Layout

```
benchmarks/verifier/real_defects_v0/
  README.md
  index.json                 one entry per case (id, kind, confidence, ...) and every count
  cases/<id>/
    case.json                the case record (schema below)
    pre_fix.yaml             module bytes at parent_commit      (when artifacts_shipped)
    post_fix.yaml            module bytes at commit             (when artifacts_shipped)
    provision.txt            provision text from corpus_release (when artifacts_shipped)
  triage/
    workflow_script.js       the screen/triage/verify workflow that read the diffs
    workflow_output.json     its return value (every reader and verifier output)
    pr_urls.json             the merged pull request of each candidate commit
    review_overrides.json    a reviewer's recorded corrections to single triage rows
    triage_merged.json       one row per module touched, with keep/drop decisions
    screen.json              the screening pass output (flagged commits and reasons)
    summary.json             the yield counts
    case_ids.json            the id of every row a build has turned into a case
    provision_review.json    the settled provision review, one record per case
    provision_review_brief.md              the readers' instructions
    provision_review_adjudication_brief.md the adjudicator's instructions
    evidence_validation.json hand calls made before the review (a cross-check)
    build_log.json           rows that failed to build, and cases the build flags removed
  tools/
    enumerate_candidates.py  module-touching commits and the keyword split
    prep_commit.py           read-only per-commit bundle (diffs, PR text) for readers
    merge_triage.py          workflow output to keep/drop rows
    build_real_defects.py    rows to cases, index, build log and README counts
    check_evidence.py        evidence_in_provision: the mechanical check
    provision_review.py      the per-case provision review: resolve, bundle, collect
```

`workflow_output.json` is the 2026-09 run's return value with one change:
the triage readers wrote absolute paths of the machine they ran on into
their notes, and `tools/merge_triage.py --scrub-record` replaced those with
`<scratch>/...` and repository-relative forms. In that run every
screen-flagged item carried `date: ""` and
`subject: "(screen-flagged) <screen reason>"`; `tools/merge_triage.py`
replaces both with the commit's own `%cI` and `%s` in `triage_merged.json`
and `screen.json`, and the workflow script takes them from the enumerated
commit list.

## Case schema

`cases/<id>/case.json` carries these keys, in this order. The verifier track
reads this schema; do not change key names without telling Max.

| Key | Meaning |
| --- | --- |
| `id` | `<jurisdiction>-<seq>-<commit8>-<module slug>`; also the directory name. The sequence number comes from `triage/case_ids.json`, so an id never changes when cases are added or filtered out; numbers the corpus skips belong to cases a build flag removed. |
| `jurisdiction` | `us` or `uk`. |
| `repo` | `TheAxiomFoundation/rulespec-us` or `TheAxiomFoundation/rulespec-uk`. |
| `commit` | Full sha of the correcting commit (post-fix state). On `origin/main`. |
| `parent_commit` | Full sha of the commit's first parent (pre-fix state). |
| `commit_date`, `commit_subject` | The commit's committer date (`%cI`, strict ISO 8601) and subject line (`%s`), as `git log -1 --format=%cI%n%s <commit>` prints them. The verify script's Git tier checks both. |
| `pr_url` | The merged pull request that carried the commit, or null. |
| `module_path` | Repository path of the rule module. |
| `corpus_citation_path` | The module's `source_verification.corpus_citation_path` (first entry when plural) at the post-fix commit. `corpus_citation_paths_all` lists all of them. |
| `corpus_release`, `corpus_release_content_sha256` | The signed corpus release the provision was resolved from. |
| `corpus_commit` | The axiom-corpus commit that release names; provision files are read from it. |
| `defect_kind` | One of `amount_mismatch`, `boundary_direction`, `unrepresented_clause`, `untraceable_branch`, `wrong_period_or_effective_date`, `wrong_entity_or_scope`, `polarity_or_logic`, `other` (`other_kind` names it). One kind per family: see `family_id`. |
| `confidence` | 0 to 1. The lower of the first reader's and the verifier's confidence, capped at 0.49 for `unclear` cases. |
| `description` | A verbatim quote from the commit message, PR text, or a comment in the module diff (`description_source` says which). Empty when nothing quotable existed. Never a paraphrase presented as a quote. |
| `locator.pre_fix_lines`, `locator.post_fix_lines` | Changed line ranges (`[start, end]`, 1-based, inclusive) from `git diff -U0 parent commit -- module`. |
| `locator.rule_names`, `locator.rule_path` | The rule(s) whose meaning changed and a YAML path to the decisive change, as read by the triage agent. |
| `pre_fix_artifact_sha256`, `post_fix_artifact_sha256` | sha256 of the shipped module bytes; must equal the Git blobs. |
| `provision_sha256`, `provision_chars` | sha256 and length of `provision.txt` (UTF-8, no added trailing newline). |
| `provision_resolution` | How the first citation's text was resolved: `mode` (`axiom_encode_resolver`, or `direct_row_exact` for release rows the current resolver rejects because they lack an `id`), `provision_file`, `provision_file_sha256`, `line_number`, `stored_body_sha256`, `resolved_text_sha256`, `slice_required`, `component_rows`, `selection_basis`, `fallback_index`, `toolchain_corpus_ref`, `fix_time_corpus_match`. |
| `provision_extension` | Null when `provision.txt` is the first citation's text alone. Otherwise the record of what the provision review added: `composition` (`sources_joined_v1`), `first_citation_text_sha256`, `first_citation_chars`, and `added`, one entry per further citation with its `citation_path`, `text_sha256`, `chars` and `resolution` (the same fields as `provision_resolution`, from the same release). See "Extended provisions". |
| `evidence_in_provision` | The mechanical check: whether `provision.txt`, as shipped, carries a string tied to the change. `present`, `absent`, or `unknown` (nothing to test, or a metadata-only case). Computed by `tools/check_evidence.py`. The review below is the settled answer; this is a cross-check. |
| `evidence_check` | How that was decided: `method`, `normalization`, `rules_tested`, `rules_missing`, every string tested (`origin`, `side`, `text`, where it came from, `matched`, and whether it `counts`), the tested and matched counts per side, and `reason` (`post_side_string_found`, `no_tested_string_found`, `provision_supports_pre_fix`, `nothing_to_test`, or `metadata_only`). |
| `judgeable_from_provision` | `true` when `provision.txt` contains the text that shows the pre-fix module wrong, `false` when no source the module cites does, null when the case was not reviewed. The benchmark loader leaves `false` cases out by default. |
| `provision_review` | Null when the case was not reviewed. Otherwise `verdict` (`in_provision`, `in_other_citation`, `not_in_sources`), `outcome` (`kept`, `provision_extended`, `not_judgeable`), `defect_real` (the readers' call on whether the sources confirm the defect), `decisive_quotes` (for a judgeable case: each with the `citation_path` it comes from, the verbatim `quote`, and its `span`, `[start, end)` character offsets into `provision.txt`), `nearest_quotes` (for `not_in_sources`: the closest passage a reader chose to cite, which decides nothing), `why`, `missing_basis` (for `not_in_sources`: what the correction rests on), `pre_fix_restates_it` (the pre-fix module quotes the passage at issue in one of its own proof-atom excerpts: the decisive passage, or for `not_in_sources` the missing or nearest one; a summary that repeats it does not count), `basis` (`readers_agree` or `adjudicated`), `readers` (each reader's verdict, `defect_real` and confidence), `mechanical_before_review` (the mechanical check's result on the first citation's text), `default_window` (whether the decisive text survives the judges' default provision window), `method` and `reviewed_on`. |
| `fix_stage` | `post_merge` when the pre-fix module bytes were once on the first-parent history of `origin/main`; `pre_merge_review` when the correction landed on the branch before its pull request merged (the pre-fix state is the encoder's output as reviewed, never on main). |
| `artifacts_shipped` | `true` when `pre_fix.yaml`, `post_fix.yaml` and `provision.txt` are in the case directory. `false` for a case whose verifier verdict was inherited from a family sibling, which ships `case.json` only; the digests still verify from Git and the release, and `tools/build_real_defects.py --ship-artifacts all` writes the files. The committed corpus has no such case: they are all family members that `--family-members representatives` leaves out. |
| `family_id`, `family_size`, `family_representative` | Cases that carry one correction applied to many modules (same commit and rule path) form a family. `family_size` counts every member that was built, including members the build flags removed, and the representative is the family's first module path among the `fidelity` members a verifier read directly, else among its `fidelity` members, else among all members. The three generator regenerations in rulespec-us PR #1300 each touched the generated tariff-schedule chapter compositions (100, 100 and 96 cases), so the committed corpus keeps representatives only. A family has one `defect_kind`: the kind most of its directly verified members were given. |
| `triage_status` | `fidelity` (reader and verifier agree it is a fidelity correction of the stated kind) or `unclear`. The committed corpus is built with `--drop-unclear`, so it holds `fidelity` cases only; the `unclear` rows stay in `triage/triage_merged.json`. |
| `triage` | The first reader's classification and notes, and the verifier's verdict, kind, confidence, justification, and quote. `candidate_source` is `keyword` or `screen`; `screen_reason` is the subject-only screen's one-line reason for flagging the commit (null for keyword candidates). When a triage chunk held more than three kept modules with the same kind and rule path (the generator families), the verifier read three and the others inherited a verified sibling's verdict; `verifier_inferred_from_module_index` names that sibling and is null for directly verified cases. `review_override` is the recorded reviewer correction applied to the row, or null. `kind_before_family_settlement` is the case's own kind when its family settled on another, and `family_kind_votes` the votes when the members disagreed; both are null otherwise. |
| `triage_notes` | The reader's notes for a reviewer. |

### Label-bearing fields

These fields carry the answer a judge is scored against, or readers' words
about it, and must not be shown to a judge: `defect_kind`, `other_kind`,
`confidence`, `description`, `description_source`, `locator`,
`triage_status`, `triage`, `triage_notes`, `evidence_in_provision`,
`evidence_check`, `judgeable_from_provision`, and `provision_review`.
`LABEL_BEARING_KEYS` in `scripts/verify_real_defects.py` lists them, and
`tests/test_real_defects_corpus.py` checks that no other field contains a
defect-kind name, the `(screen-flagged)` marker, or the screen reason, reader
reasoning, or verifier justification of its case.

Three other fields are not triage labels but are the maintainers' own words
or links about the fix, so a blind benchmark should withhold them too:
`commit_subject` (a subject often says what was wrong), `pr_url`, and
`commit`/`parent_commit` (which lead to the diff).

### Extended provisions

When the provision review found the decisive text under another citation,
`provision.txt` holds the first citation's text and that citation's text,
each under a `--- Source: <citation path> ---` line, with one blank line
between them (`compose_provision` in `scripts/verify_real_defects.py`). The
added citation is resolved from the case's own signed release with the same
resolver, and `provision_extension` records its row and digests, so the
verify script re-derives the composed text. An unextended `provision.txt` is
the first citation's text with no header, byte for byte as before.

The added citation is one the module names (in its header or in a proof atom
of a changed rule) on either side of the fix. Some are citations only the
post-fix module names: the fix itself added the source the pre-fix module
had missed.

### How the release is chosen

Modules do not name a corpus release; the repository's
`.axiom/toolchain.toml` at the fix commit does. `selection_basis` records the
rule that applied:

- `toolchain_release_pin`: the toolchain named a signed release
  (`axiom_corpus_release`).
- `latest_release_at_or_before_toolchain_corpus_ref`: the toolchain named only
  an axiom-corpus commit (`axiom_corpus_ref`); the latest signed release whose
  corpus commit is that commit or an ancestor of it was used. `fix_time_corpus_match` then says whether the same provision
  row at the toolchain's corpus commit carries the same body (`same_body`),
  a different one (`different_body`), or was absent.
- `earliest_release_after_toolchain_corpus_ref`: the toolchain named a corpus
  commit older than every signed release; the earliest release after it was
  used.
- `no_toolchain_corpus_pin`: the toolchain named neither; the releases were
  tried oldest first.

The counts section gives how many cases each rule chose.

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
   stays as `unclear`. Verifier `unclear` keeps the module as `unclear`,
   whatever the reader called it. A missing verdict keeps the reader's
   classification. Confidence is the lower of the two readers', capped at
   0.49 for anything that stays `unclear`. The merge then applies
   `triage/review_overrides.json`: corrections a reviewer of PR #1659 made
   to single rows, each with its source and reason (one `unclear` case
   promoted to `fidelity`; the rules a verifier found mechanical dropped
   from one family's `rule_names`).
6. Build (`tools/build_real_defects.py`): for each kept row, read the pre and
   post module bytes from Git, require `format: rulespec/v1` on both sides,
   take the module's citation path, choose the release as described above,
   resolve the provision through `axiom_encode.corpus_resolver` against a
   sparse corpus root materialized from the release object and the corpus
   commit, compute the locator from the diff hunks, and decide `fix_stage`.
   Rows that fail any step are listed under `dropped` in
   `triage/build_log.json`.
7. Ids, families and selection (same tool). Each built case takes its id
   from `triage/case_ids.json`. Cases with the same commit and rule path
   form a family, which settles on one kind and one representative. Then the
   build flags apply: `--drop-unclear` leaves out `unclear` cases and
   `--family-members representatives` leaves out every other family member.
   The committed corpus uses both, so it holds the cases a benchmark would
   take by default; each case left out is listed, with its id and the
   reason (`triage_unclear` or `family_non_representative`), under `removed`
   in `triage/build_log.json`.
8. Provision review (`tools/provision_review.py`, then the build). See below.

The screening readers were Claude Fable 5.1 agents; the triage and
verification readers were Claude Opus 5 agents (the Fable pools on two lanes
hit their usage limits mid-run, and bounded review work routes to Opus). All
ran through the workflow in `triage/workflow_script.js`; no gpt-5.5 model was
used anywhere.

### Provision review

`provision.txt` is resolved from the module's first citation. The text a
correction rests on can sit under another citation (an amending Federal
Register notice, a sibling section, a different HTS heading), or under none
the module names. A judge shown the provision and one module cannot decide
such a case. So every board-eligible case (a shipped family representative
with `triage_status: fidelity`) was checked against its provision, on
2026-10-10:

1. `tools/provision_review.py resolve` resolved every other citation the
   module names, in its header or in a proof atom of a changed rule, on
   either side of the fix, from the case's own signed release.
2. `bundle` wrote one bundle per case: the triage record, the pre-to-post
   diff, both modules, the provision, and each other citation's text.
3. Two readers read every bundle independently, under the brief in
   `triage/provision_review_brief.md`. Each said where the text that shows
   the pre-fix module wrong sits and quoted it: `in_provision`,
   `in_other_citation`, or `not_in_sources`. The test is that the text must
   fit the post-fix rule and conflict with the pre-fix rule, and that the
   module's own words (proof-atom excerpts, summaries) do not count.
4. `collect` checked that every quote occurs in the text it names and
   settled each case the two readers agreed on. Where they disagreed, or a
   quote did not check, a third reader got both records and the bundle
   (`triage/provision_review_adjudication_brief.md`), and its call settled
   the case. `triage/provision_review.json` holds the settled records with
   each reader's call. One adjudicated case, `us-396`, was corrected after
   the round-three review of PR #1659 found that its quoted passage only
   cross-references the omitted clause; its `adjudication_note` says so.
5. The build applied them: `in_provision` cases are kept as built,
   `in_other_citation` cases get that citation's text appended to the
   provision (see "Extended provisions"), and `not_in_sources` cases are
   marked `judgeable_from_provision: false`.

The readers were models on Subfleet lanes, not people: Claude Opus 5.5 for
both readers and the adjudicator, except that four of the first reader's 37
batches were served by GPT-6.1 Sol (`triage/provision_review.json` records
the models under `readers`). No judge was run and no API was called. The
counts section reports how many cases each verdict took, how many were
settled by agreement and by adjudication, and how the mechanical check
compares.

As a cross-check, `triage/evidence_validation.json` holds 63 hand calls made
before this review, by the round-two reviewer of PR #1659 and by two sets of
blind agent readers, on whether the first citation's provision carried the
evidence. The review agrees with 57 of the 63. In all six
disagreements the earlier call said the provision carried the evidence and
the review found it in none of the cited sources.

The benchmark loader (`benchmarks/verifier/encodebench_verifier/sources/real.py`)
leaves out, by default, a case marked `judgeable_from_provision: false` and a
case whose decisive text falls outside the provision window it is about to
show the judge (`build-real --include-not-judgeable` keeps both).

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
- The provision review is a reading by models. Its quotes are checked
  mechanically (each occurs in the text it names, and the build finds it
  again in `provision.txt`), but whether a quoted passage is enough to
  expose the defect is the readers' judgment. `provision_review.readers`
  and `basis` show how each case was settled.
- A `not_in_sources` case stays in the corpus, flagged. Its pre-fix module
  was corrected for a reason its cited sources do not show: a source the
  module never cites, a citation the signed release does not carry, an
  imported module, or agreement with a sibling rule. `missing_basis` says
  which. A judge shown only the provision cannot be scored on it.
- An extended provision is narrower than the module's sources. It holds the
  first citation and the citation the review found decisive, not every
  citation the module names, so the choice of what to add uses knowledge of
  the fix. Both modules of a pair are judged against the same text.
- Provision windows can exceed the judge's 24,000-character truncation
  (`DEFAULT_PROVISION_CHARS` in `src/axiom_encode/judges/client.py`), and an
  extended provision more often than a plain one: some added citations are
  whole Federal Register notices. `provision_review.default_window` says
  whether the decisive text survives that window, and the benchmark loader
  drops a case whose decisive text its own window cuts away.
- The mechanical check (`evidence_in_provision`) does not read for meaning.
  It tests whether strings tied to the change occur in `provision.txt`
  after casefolding and collapsing whitespace: the proof-atom excerpts the
  fix added; the excerpts on rule fields the fix changed (counted only when
  the fix added no excerpt or value of its own); numbers, dates and
  code-like identifiers that appear on one side of the fix only; and quoted
  spans of eight or more words from the triage reasoning. Strings that
  carry a value the fix removed are pre-side: they support the pre-fix
  module. The counts section compares it with the review. It says `present`
  for some cases the readers found not judgeable, when a tested string is
  in the provision but is not what exposes the defect.
- Some fixes span several modules; one case per module per commit means the
  quoted description may describe the whole commit, not the module alone.
- Few cases use a release the fix commit itself pinned
  (`toolchain_release_pin`; the counts section gives the number). The rest
  use a release chosen relative to the toolchain's corpus commit, or tried in
  order when the toolchain named neither (see "How the release is chosen"
  and `selection_basis`).
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
`gh variable get` on `TheAxiomFoundation/rulespec-us`).

The committed corpus is the output of two commands over the committed
triage records. The merge turns the workflow output into keep and drop rows:

```bash
uv run python benchmarks/verifier/real_defects_v0/tools/merge_triage.py \
  --workflow-output benchmarks/verifier/real_defects_v0/triage/workflow_output.json \
  --corpus-dir benchmarks/verifier/real_defects_v0 \
  --rulespec-us ../rulespec-us --rulespec-uk ../rulespec-uk --scrub-record
```

It reads `triage/pr_urls.json` and `triage/review_overrides.json` and writes
`triage_merged.json`, `screen.json` and `summary.json`; a test re-runs it and
compares the bytes. The build turns the rows into cases:

```bash
uv run python benchmarks/verifier/real_defects_v0/tools/build_real_defects.py \
  --corpus-dir benchmarks/verifier/real_defects_v0 \
  --rulespec-us ../rulespec-us --rulespec-uk ../rulespec-uk \
  --axiom-corpus ../axiom-corpus --release-cache ~/.cache/axiom-real-defects \
  --drop-unclear --family-members representatives
```

It reads `triage/case_ids.json` and `triage/provision_review.json` and
writes `cases/`, `index.json`, `triage/build_log.json` and the generated
counts above. Without the two flags it builds every kept row, with the same
ids. `--refresh-review` skips the build and re-applies the provision review,
the mechanical check, the index and the counts to the cases on disk.

To extend the corpus with later fixes, enumerate the candidates and re-run
the triage workflow in `triage/workflow_script.js` first:

```bash
uv run python benchmarks/verifier/real_defects_v0/tools/enumerate_candidates.py us ../rulespec-us /tmp/real-defects-enum
uv run python benchmarks/verifier/real_defects_v0/tools/enumerate_candidates.py uk ../rulespec-uk /tmp/real-defects-enum
```

`tools/prep_commit.py` writes the per-commit bundles the triage readers use;
pass their directory to the merge as `--inputs-dir` and it reads the PR URLs
from them and rewrites `triage/pr_urls.json`. New cases then need the
provision review (`tools/provision_review.py resolve`, `bundle`, readers,
`collect`) before the build marks them judgeable.

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

The test always runs the shipped-file tier. It runs the rulespec Git tier
when the sibling checkouts exist (`AXIOM_REAL_DEFECTS_RULESPEC_US` and
`AXIOM_REAL_DEFECTS_RULESPEC_UK` override the locations). The axiom-corpus
tier is opt-in: it runs only with `AXIOM_REAL_DEFECTS_CORPUS_TIER=1` and an
axiom-corpus checkout (`AXIOM_REAL_DEFECTS_AXIOM_CORPUS` overrides the
location), because it streams provision blobs out of a multi-gigabyte pack.

The Git and corpus tiers read objects through one `git cat-file --batch`
process per repository. Each distinct provision file is read once, hashed in
1 MiB chunks, and only the row lines the cases name are parsed.
