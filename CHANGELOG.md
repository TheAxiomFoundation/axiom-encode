## 0.2.2367

- Require Encoding-Target and Fixes-Run lines in every PR body (#2156)
- Add content-bound merge guard evaluator (inactive, #2172) (#2204)
- Declare the validation effect in automated version-bump PRs (#2197)

## 0.2.2366

- Bind Canadian form and Schedule M obligations to observed execution (#2083)

## 0.2.2365

- Add failure classes and a pre-encode gate that stops empty and repealed targets before a model call (#1997)
- Count only listed values in numeric-recall retry feedback (#2209)

## 0.2.2364

- Bind complete form source evidence to verified execution (#2139)
- Add immutable complete T936 producer guidance reference (#2208)

## 0.2.2363

- fix: bind copied return categories to source projections (#2129)

## 0.2.2362

- fix: bind Schedule 2 document request evidence to its source duty (#2128)

## 0.2.2361

- Verify actual same-page Schedule 2 floor evidence (#2127)

## 0.2.2360

- Verify with encoder main: producer floor, verifier record, RuleSpec admissions registry (#2093, #2095; replaces #2101) (#2194)
- Avoid treating complete informational URL referrals as computations (#2068)
- Scope the session-law deferral reader to statute self-references (#1930)
- Keep nonfiler return instructions out of factual gate counts (#2122)
- Allow complete GPT-6 authoring output and reject incomplete responses (#2117)
- Send the Codex prompt on stdin instead of argv (#1205) (#2184)
- Record the terminal outcome of a live encode run (#2189)
- Rewrite the US replay snapshot for the located numeric-recall message (main is red) (#2195)

## 0.2.2359

- fix: retain strict proof requirements in partial repairs (#2089)

## 0.2.2358

- Bind validation waivers per covered file in signed manifests (#2092) (#2097)

## 0.2.2357

- Show every missing numeric-recall value, located and grouped, in retry feedback (#1934)
- Recognize detached Schedule 2 source rows without changing private consumers (#2125)
- Implement enrolled Codex and deterministic producers with notary admission (#1662)
- Make the offline replay benchmark a required CI gate over 45 public fixtures (#1891)
- Propose approving promoted RuleSpec PRs after validation and signing (#2172, docs only) (#2173)

## 0.2.2356

- Fix locked runtime setup for reviewed promotion preflight (#2185)

## 0.2.2355

- Record a dispatch record and typed step checks for signed re-encode runs (#2181)
- Record the GitHub Actions run and live encode phase in encoder telemetry (#1750)
- fix: provision and verify the required Codex tool host (#2104)
- Treat a missing workflow toolchain as no axiom-compose pin in targeted re-encodes (0.2.2336) (#2099)

## 0.2.2354

- Stop false complete-source demands from `falls`, reference titles, mixed percentages and whole Money values (#2152)

## 0.2.2353

- Keep the whitespace patterns on the interval parser's path linear in a space run (#2158)

## 0.2.2352

- Admit reviewed RuleSpec PRs without per-candidate encoder records (#2084)

## 0.2.2351

- Steer complete-source generation and repairs to encode omitted branches first (#1966)
- Keep US bibliographic locators out of complete-source numeric recall (#1898)
- Demand every source-stated threshold of a conjoined condition as a companion-test boundary (#1937)
- Ground decimal rates from flattened percentage tables (#1648)
- Label U.S. branches in English, drop German sentence markers in U.S. text, and read U.S. citation ranges (#1896)
- Keep titled reference lists out of complete-source numeric recall (#2149)

## 0.2.2350

- Harden optional retired-manifest inventory reconciliation (#1566)
- Stop key-bearing jobs from restoring the uv cache (#1950)
- Define each corpus read refusal message once (#2108)

## 0.2.2349

- Stage the verification supervisor's site-packages from uv.lock (#1712)
- Forward the validated waiver audit worker setting through the signing supervisor (#2011)
- Assign the encoder version after merge, not in each pull request (0.2.2348) (#2051)

## 0.2.2348

- Assign the encoder version after merge, not in each pull request (#2048). The new `Encoder version bump` workflow runs on every push to `main`. When encoder-affecting files changed after the latest bump, it writes the next version into `pyproject.toml`, `src/axiom_encode/__init__.py` and `uv.lock`, adds a `CHANGELOG.md` header that lists the pull requests merged since the previous bump by title (it leaves out bump pull requests), and opens a bump pull request with auto-merge on. It leaves the towncrier fragments in `changelog.d/` alone. The workflow opens the pull request with a token of the `axiom-encoder-version-bump` GitHub App, made on each run from the `ENCODER_VERSION_BUMP_APP_CLIENT_ID` variable and the `ENCODER_VERSION_BUMP_APP_KEY` secret of the `encoder-version-bump` environment, whose branch rule allows only `main`. It closes, holds for or auto-merges only this repository's pull requests that the App opened, never a fork's or a person's pull request with the same branch name, and it auto-merges only the bump commit it built or verified (`--match-head-commit`).
- The bump writes one more than the highest version any first-parent commit of `main` declared, so a merge that lowered the version never makes it reissue a used number.
- Once the bump workflow is on `main`, the required `lint` job refuses a pull request that changes the encoder version lines unless the App opened it from this repository (`version_provenance.py pull-request`). Open pull requests drop their hand bumps; one whose `lint` check passed before this landed must be pushed again, because a merge-queue run does not repeat the check.
- Add `axiom_encode.version_provenance`, a standard-library check of the version rule at any git ref. It applies the same rule as the `--apply` guard: no encoder-affecting file may change after the latest version bump. It refuses a shallow clone, whose boundary commit would read as a bump. On the head's first-parent history it also needs a bump when the head declares a version below one that `main` already declared, or when an encoder file changed after `main` first reached the head's version at another commit; the `--apply` guard and the provisioner apply the same two rules.
- Count a deleted encoder file, both sides of a moved one and a non-ASCII file name as encoder changes, in `version_provenance`, the `--apply` guard and the provisioner (`--no-renames --diff-filter=ACDMRT -z`). Before, each of these needed no bump. A differential test over random histories holds the three copies to one verdict.
- Read the history that the version rule checks, including each commit's first parent, without replacement refs, grafts (including an `info/grafts` file), the commit graph or ambient `GIT_*` values, in `version_provenance`, the `--apply` guard and the provisioner, so a local history rewrite cannot hide an unversioned encoder change. `ci_parity.encoder_version_at_pin` reads the version files through the same helper. The `--apply` guard now also refuses a shallow clone.
- Refuse an encoder head that no bump covers in the re-encode workflow's attempt-budget job. The encode job is then skipped, so an ad hoc dispatch made before the bump lands uses no production-signing approval. The attempt budget treats such a run as neutral when it can look up the run's jobs; past its ten job lookups (or on a lookup error) it still counts a failed run, as before. Queue dispatches skip that job, so the SNAP queue tick holds and the protected SNAP queue workflow refuses to dispatch while `main` is unversioned.
- `ci_parity.encoder_version_at_pin` refuses a pin whose encoder files changed after its version bump.
- `test_current_encoder_affecting_changes_are_behind_version_bump` now runs only with `AXIOM_ENCODE_REQUIRE_VERSION_BUMP=1`, because a pull request no longer carries a bump. CI sets it on bump pull requests (`automation/encoder-version-*` heads).
- Add `.github/workflows/encoder-version-bump.yml` and `src/axiom_encode/version_provenance.py` to `CODEOWNERS`.

## 0.2.2344

- Integrate the independently reviewed nested-condition and physical-alternative source checks; refresh encoder version provenance for the combined tree.

## 0.2.2338

- Check bounded physical allowance alternatives as shared prerequisites plus complete alternative factual groups, using signed granting paths and an explicit shared-set input naming contract. Preserve billing, action, causal, distinct-count, entity, period, proof-ownership, and temporal requirements; unknown contracts remain conservative. This is not complete semantic equivalence, legal-alternative coverage, or validation of raw caller classifications.

## 0.2.2337

- Preserve closed nested physical reason alternatives as one required factual gate inside an all-of condition, rather than requiring mutually alternative reasons simultaneously or treating a closed navigation list as operative conditions. Unknown continuations remain conservative; negative or qualified terminal evidence cannot corroborate a positive reason. This narrows a source-explicit-condition opacity check, not Boolean equivalence, source completeness as a whole, or oracle validation.

## 0.2.2299

- Preserve the original checkout identity when validating staged eval and apply overlays, so same-case source witnesses execute against the candidate closure while retaining exact Git, file, and engine binding checks.

## 0.2.2298

- Type validator issues from declared checks. Each converted issue is a `ValidationIssue` (a `str` whose value is the unchanged message) that names its check id, subject (citation, branch or clause with the branch's source span, module and line, rule, test case or field), expected and found values, a short evidence excerpt and a stable fingerprint. Checks declare `title`, `requires`, `why`, `fix`, `cause` and their code site next to the code that runs them. Converted: complete-source-unit `structure`, `formula-output`, `numeric-recall`, `deferral`, `source-explicit-conditions` and `tests` (split into ten checks), proof source evidence (not found, out of scope) and the rules-engine compile failure.
- Add each failed check's declared fix, with expected versus found per subject, to the model's retry feedback after the unchanged validator messages. Feedback with no typed issue is byte-identical.
- Record the bounded `axiom-encode/validation-issue/v2` list (it has every v1 field name proposed in #1660, but a typed issue's `kind`, `value`, `clause` and `locator` values differ from v1; it carries the definitions of the checks it names) per try in its own key `iterations[].errors[].issue_document`, apart from the v1 `issues` list, in the outcome as `validation_issues`, and in the failure bundle as the optional `issues.json` field `typed_issues`; the bundle verifier, the repair extractor and the repair-feedback reader accept it. The 128 KiB bound covers the whole serialized document. A record whose message is cut to 2,000 characters keeps `message_sha256` of the full message, so a `--repair-candidate-root` replay re-types it with its check and fix. The workflow's repair extractor writes only the candidate pair, as before, so a workflow repair replay has no prior issues to re-type. Compile issues name the module by its repo-relative path, with the jurisdiction first when the policy repo is a jurisdiction content root, so `us` and `us-ky` modules at one relative path stay distinct. Branch subjects carry the branch's source span, so two computations in one paragraph get distinct fingerprints. A missing-boundary issue's fingerprint names only its own boundary. On the deferred-apply path, the final try of an overlay rejection records the overlay's typed issues; the legacy non-deferred path still logs the run before apply validation, so its final try has no overlay issues. `validation_failure_counts` now counts a typed issue under `<gate>:<check id>` instead of its message class, so trends over the old classes break at this version for those issues. Run-log `gate.compile` and `gate.ci` findings use the check id as their code.
- Publish the registry with `axiom-encode checks [--json]` and `src/axiom_encode/data/check_registry.json`. Record `infra:<step>` issues (annotation, step summary, `$RUNNER_TEMP/axiom-infra-issues.jsonl`) for the refusals of the six targeted re-encode steps before the signing supervisor, without changing their messages or exit codes. `tests/test_checks.py` fails when untyped `issues.append` sites or unrecorded workflow refusals increase, or when a check lacks `fix` or `why`.

## 0.2.2270

- Integrate current source-completeness updates with the reviewed explicit engine-pin and portable provenance checks.

## 0.2.2266

- Integrate current source-bound amendment relation guidance with the reviewed explicit engine-pin and portable provenance checks.

## 0.2.2265

- Integrate current authenticated source-transport support with the reviewed explicit apply-engine revision and portable provenance checks.

## 0.2.2264

- Support eleven authenticated signed provision exports for the current pinned corpus release, including the atomic transport's distinct primary run and all bounded workflow metadata and archive slots. Keep exact source-contract and export-byte verification.

## 0.2.2263

- Prefetch authenticated signed corpus provision exports for protected runs whose release provisions are R2-only, with exact exporter, archive, release, and provision-byte checks. Admit the reviewed PyMuPDF exporter dependency-file pair.
- Allow one authenticated Arizona corpus export and up to seven distinct signed provision exports in the same fresh protected transaction, without altering the source contract or the reviewed Canadian repair-head admission.
- Bind historical Arizona SNAP manifest-owner replacement to the signed receipt and original Git bytes throughout protected packaging and publication; retain the canonical in-place manifest path and existing retired-source-metadata migration dispatches.
- Refresh the encoder version after merging current main's engine-binding changes into the authenticated Arizona provision-export branch.

## 0.2.2262

- Reject explicit apply engine pins that disagree with checkout HEAD before binary resolution and independently reject mismatched captured bindings when constructing portable signed provenance. Preserve the signed schema and legacy unpinned behavior.

## 0.2.2261

- Cover fresh and retained candidate engine-pin forwarding through actual validation entrypoints, and clarify that local pre-generation binding can build a clean matching engine checkout.

## 0.2.2260

- Forward the declared engine revision through native generation, retained-candidate and repair reevaluation, and apply validation so source witnesses use the verified runtime binding. Preserve optional unpinned callers.

## 0.2.2259

- Bind manifest refresh validation to the declared engine revision and verified binary through apply checks, creating the engine receipt before the protected runtime. Preserve byte-identical obsolete manual aliases only when independently verified canonical base and head receipts cover unchanged live files; retain changed-receipt and full-census verification.

## 0.2.2258

- Forward the reviewed dependent finding through all Arizona corpus-transport preflights, admit canonical shallow historical-owner deletions, and refresh the encoder version after current main.

## 0.2.2257

- Refresh the encoder version after merging the reviewed Canadian repair-head admission into the Arizona ECE transport branch.

## 0.2.2254

- Recognize completed signed in-place source-bound dependents in the legacy replacement census, retaining source-citation, primary-proof, and fresh-manifest checks.

## 0.2.2253

- Regenerate Arizona ECE's proof-pinned dependent from its own signed FAA5 source in the same protected legacy replacement transaction, and verify both source bindings and the updated proof hash before publication.

## 0.2.2252

- Admit authenticated in-place legacy Arizona ECE replacement from a signed corpus artifact while preserving the existing RuleSpec identity and dependent references; keep distinct-path moves and unsupported transport modes rejected.
- Admit the exact reviewed Canadian repair head for all protected artifact-only modes, including manifest refresh and reviewed-candidate promotion. Preserve publication restrictions and every signing and validation gate; link the admission evidence.

## 0.2.2248

- Add reviewed `repealed_tombstone` coverage exemptions for the exact repealed leaf scopes 26 USC 3121(a)(3), 3121(a)(17) and 3306(b)(3), bound to the signed corpus bytes, with their review evidence. The operative parent paragraphs and every other scope keep their coverage demands.

## 0.2.2247

- Add bounded authenticated corpus artifact transport for fresh complete-source replacements, binding the official reviewed exporter, run and artifact metadata, archive bytes, signed release, immutable and current scope locks, and protected signing workflow without granting source authority.

## 0.2.2246

- Verify a closed printed net-income row using independently executed floored and retained signed values with authenticated informational referral identities. Preserve source operands, version frames, and refusal for missing or conflicting evidence; do not infer loss eligibility or execute downstream policies.

- Accept an optional nonempty descriptive `source` on imported entered-amount observations and direct worksheet entered-condition wrappers without treating that text as proof or changing canonical identity, formula, type, frame, or proof checks.

## 0.2.2243

- Verify a complete source repayment alternative group only when every top-level alternative and every owner version has its own executed threshold pair. Preserve partial conditions, leaf-level coverage, and separate floor/advisory obligations.

## 0.2.2242

- Verify exact source repayment thresholds using separately executed equality and above-threshold cases, authenticated entered observations and branch-required worksheet facts. Preserve generic boundary evaluation and separate compound-condition and advisory obligations.

## 0.2.2241

- Verify signed numeric representation of exact printed bracket-format instructions using authenticated operand ownership and separate actual negative and nonnegative executions. Preserve floor, advisory, repayment and boundary obligations.

## 0.2.2240

- Recognize referral URLs inside verified printed arithmetic rows without treating URL punctuation as calculation operators. Require exact source-bound local operand order, all-version proofs and matching executed formulas; preserve advisory text and separate condition checks.

## 0.2.2239

- Verify sparse worksheet choices using exact source-bound entered observations and separately executed cases. Require the selected-only worksheet fact, preserve canonical ownership and unchanged common facts, and keep generic paired-condition and boundary checks unchanged.

## 0.2.2238

- Verify source-bound subtraction floors using separate actual negative and positive cases, canonical imported operand proofs and exact execution receipts. Preserve legacy paired-condition and boundary checks, including unresolved sparse worksheet branches.

## 0.2.2237

- Verify imported formula witnesses against pinned, isolated compilation and exact-case Explain execution, while retaining original repository checks and separate boundary and paired-condition obligations.

## 0.2.2236

- Bind imported entered-line observations to their canonical providers and authenticated source proofs. Check complete repayment alternative trees, strict thresholds and result branches before treating their conditions as explicit, while preserving ordinary execution and source checks.

## 0.2.2235

- Preserve verified whole-form reference locations when extracting computation clauses, so a referral such as Step 4 is not mistaken for an arithmetic operand. Keep real amounts and malformed references subject to the existing checks.

## 0.2.2234

- Recognize complete printed-form referrals and label typography without treating them as arithmetic. Keep condition proofs within their source rows while preserving inherited repayment guards and conditional branches.

## 0.2.2233

- Preserve printed form operation rows and inherited conditions during source-clause recognition. Bind complete Add-range proof excerpts to their unique source context while retaining structural ownership and selected-version checks.

## 0.2.2232

- Distinguish complete slip-box referrals, coordinated printed line labels, and corroborated form footers from numeric policy values, preserving amounts, source offsets, and unrelated completeness checks.

## 0.2.2230

- Keep a person responsible for a family-type resource distinct from a Family claimant in source-scope validation, while retaining genuine Family and Household requirements.

## 0.2.2229

- Validate Schedule L partnership reporting with executed sparse-input evidence and gross/net separation. Recognize complete form locators and corroborated printed table glyphs while preserving substantive values, source operations, and general validation requirements.

## 0.2.2228

- Bind student-loan reporting routes to executed, source-owned sparse facts and canonical module assertions; preserve unrelated condition checks.

## 0.2.2227

- Bind student-loan measurement periods to exact source roles, factual declarations, annual proofs, and ordinary executed witnesses without treating historical years as arithmetic operands; preserve unresolved conditions and unknown numeric obligations. Recognize the exact optional copy-for-records guide referral as nonoperative guidance.

## 0.2.2226

- Recognize correlated printed form titles, complete form identifiers, and corroborated terminal line coordinates without treating them as computations or scalar amounts; preserve substantive years and source offsets.

## 0.2.2220

- Preserve the complete tracked baseline for fresh same-source canonical replacements, supply full target edit context, and overlay omitted rules and companion cases before validation without accepting the old artifact as a generated repair.

## 0.2.2183

- Repair source-verification metadata emitted for generated overlay targets during protected apply while preserving source-bound validation.

## 0.2.2182

- Clarify source-bound canonical CFR branch citations for complete-source deferrals and record this encoder behavior after the SNAP source-check parent merge.

## 0.2.2181

- Record the completed SNAP table source-check changes behind a fresh encoder version and align version-locked oracle tests with that provenance.

## 0.2.2179

- Match worksheet formula proofs to independently recognized chart headings without treating a printed chart identifier as a formula operand. Preserve exact source coordinates, unique excerpt ownership and numeric proof boundaries.

## 0.2.2177

- Bind formula proof excerpts to their unique authoritative source locations. Keep repeated worksheet instructions attached to their own context, preserve proof boundaries and raw coordinates, and reject ambiguous overlapping excerpts.

## 0.2.2176

- Integrate declared-entity companion inputs with source-bound worksheet evidence and binary arithmetic instructions. Keep package provenance and all current-version assertions aligned with the combined runtime.

## 0.2.2175

- Require formula coverage for explicit binary add, subtract, multiply and divide instructions with quantitative operands or source row references. Preserve operand order and raw source context; do not treat identifier numbers, quoted instructions or unsupported conditions as positive arithmetic instructions.

## 0.2.2174

- Record the finalized worksheet-evidence implementation after formatting so generated artifact provenance includes every encoder-affecting commit. No calculation or validation behavior changes.

## 0.2.2173

- Bind worksheet replacement operations, existing equality witnesses and the complete worksheet operation block to current source proofs, selected versions and independently executed companion cases. Preserve distinct-witness allocation and annual coverage accounting; reject stale private rounding context.

## 0.2.2172

- Recognize the source-owned foreign-country count and tax threshold only with exact typed facts, complete proofs, effective dates, and independently corroborated boundary cases. Preserve unrelated income-condition findings.

# Changelog

All notable changes to Axiom Encode will be documented here.

- Recognize source-equivalent nested worksheet guards and directly owned
  part-year income sums. Keep typed dependencies, selected proof ownership,
  source arithmetic and executed companion witnesses mandatory.

- Recognize four explicit Alberta form-line input spellings in T2036 worksheet
  path evidence. Source coordinates, types, proof ownership and runtime witnesses
  remain required; ambiguous declarations and imported aliases fail closed.

- Keep US manual and form locators out of complete-source-unit numeric recall:
  `WAC` section citations, `policy NNN-N` cross-references, numbered manual
  section headings, telephone numbers, PO boxes, ZIP+4 codes and
  letter-prefixed form numbers are no longer demanded as named scalars.
  Paginated manuals also drop the page number and `(MM/YYYY)` revision
  stamp of their page headers (`93 (07/2026) Chapter 2:...`), which the
  Oregon Programs Eligibility Notebook prints on every page. Values stay in
  recall when a unit follows them (`policy 10-day notice`) or when a
  cents-shaped row such as `44.50 Countable Earned Income` sits beside a
  `$` amount line of a budget table.

- Stop agency-manual typography from creating complete-source-unit
  formula-output obligations. The arithmetic recognizer read a month/year
  date (`93 (07/2026)`, the header of every Oregon Programs Eligibility
  Notebook page), a `•` list bullet between words, a web address and a
  footnote asterisk as operators, so every Oregon page in the SNAP dispatch
  pilot asked for a computation no encoding could provide. Those spans are
  now masked before matching; a `•` next to a number, a parenthesis or a
  one-letter variable stays multiplication (the § 32a EStG tariff), as does
  an asterisk next to a number or one-letter variable. The same typography
  no longer adds a division or a multiplication to the source operations or
  the source topology that the formula-output check compares with the
  encoded formula. Across corpus 8f7d60aa this removes 9,105 of 277,800
  computation clauses and adds none; about 90 removed clauses state a real
  formula only in words or `$` amounts, which the recognizer did not cover
  before either.

- Accept a federal regulation's `<title>-cfr` module root
  (`us:regulations/42-cfr/457/800/...`) as a complete-source-unit deferral
  anchor alongside the corpus-shaped `us:regulations/42/457/800/...` root. The
  source sub-paragraph coverage gate and the apply-time scope filter already
  counted that root, while completeness coverage silently dropped it, so one
  deferral could satisfy at most one of the two gates and the model saw only
  "neither encoded nor precisely deferred". Imprecise regulation deferrals now
  get `[complete-source-unit:deferral]` feedback naming the literal canonical
  branch citation (`us/regulation/42/457/622(a)`), as statute deferrals do.

- Raise the LLM judge output budget from 2,048 to 16,000 tokens and report a
  reply cut off by the budget as a `max_tokens` judge error naming the budget,
  instead of a generic parse error. The statutory-fidelity referee's findings
  JSON was truncated on modules of about 30k input tokens. 16,000 stays under
  the anthropic SDK's non-streaming ceiling, and `AXIOM_JUDGE_MAX_TOKENS` still
  overrides it.

- Separate executable relation directions from advisory entity hints, deferring
  declaration fallback until reachable explicit evidence is collected and retaining
  fallback for mixed sum value/predicate annotations on one related entity ID.

- Match companion relation traversal to engine entity-ID context across ordinary
  derived references, nested aggregations, and derived-relation predicates.

- Preserve entity context for proven numeric Scalar constant helpers during
  companion relation discovery; reject input-dependent or cyclic helpers.

- Bind affirmative worksheet notice conditions to derived field-code outputs
  through adjacent authoritative rows and complete formula proof excerpts,
  preserving exact threshold direction and independently executed test evidence.

- Admit exact descendant rows as retired replacement context only when their
  full evidence is contained in the parent under the same signed corpus
  artifact and source scope. Retain separate attestations and containment
  offsets in durable review artifacts without changing generated source gates.

- Recognize threshold witnesses over derived numeric rules only when both test
  cases assert and independently reproduce the dependency chain. Preserve
  quoted integer thresholds when matching source conditions.

- Accept an additive companion-test fragment during protected tests-only repair
  only when every new case exactly matches the required case contract. The
  encoder appends it to retained tests, preserves all existing cases, and keeps
  the signed RuleSpec body byte-identical.

- Permit a tests-only signed revision of the exact reviewed SNAP OBBB memo
  RuleSpec head. The protected workflow binds the immutable source SHA and
  target path, rejects dirty or untracked candidate files, and keeps the
  encoder's unchanged-rule-body contract in force.

- Remove a generated `module.source_verification.values` mapping only after the
  rules engine identifies that exact unknown field and its complete accepted
  schema, preserving all other RuleSpec bytes and revalidating the overlay.

- Create each targeted output lane before validating its durable rejected-
  candidate destination, so signed re-encodes reach generation while retaining
  the destination's fresh-directory safety contract.

- Export the strongest validator-rejected candidate from targeted signed
  re-encodes and prefer its integrity-bound three-file artifact during repair
  replay, while retaining compatibility with older failure artifacts.

- Retain the validator-rejected RuleSpec candidate with the fewest full
  validation findings across model retries, keep its bounded feedback attached
  to that exact candidate, and emit the same best candidate for cross-run
  repair instead of allowing a later regression to overwrite stronger work.

- Keep parenthetical source conditions bounded to their closing parenthesis so
  later descriptive conjunctions cannot be misclassified as additional legal
  eligibility gates.

- Bound proof-owned parent chapeaux at their first structural child so a colon-
  terminated eligibility list does not make every alternative satisfy the
  first child branch's factual gates.

- Carry bounded, source-specific review findings through each canonical-refresh
  lane and bind their exact content in signed context evidence. Louisiana
  external-dependency validation now recognizes narrowly scoped executable
  dependency objects and preserves operative predicates across official prose
  line wraps.

- Interpret formula-range numeric evidence in source order across numeric
  extraction profiles, so a percentage yielded before word-number thresholds
  cannot hide a valid conjoined lower and upper income interval.

- Reject source-add dispatches when the primary or bundled canonical RuleSpec
  destination already exists in the pinned checkout. Existing modules must use
  the authenticated canonical-refresh contract, so mode mistakes fail before
  model execution or signing instead of reaching an invalid full-tree compile.

- Classify affirmative duties introduced by a `notwithstanding` exemption as
  enabling exception effects, so complete-source companion tests can prove
  obligations that remain operative despite a separate filing exemption.

- Add a broker-authenticated `migrate-rulespec-paths` workflow for
  current-v5 model encodings. It permits only deterministic engine-safe path
  normalization and exact durable-reference rewrites, preserves legal corpus
  citations and nested model provenance, binds the clean repository base/tree,
  corpus, waiver set, plan, paths, and hashes in a signed receipt, and installs
  the complete migration transactionally. Persisted verification recomputes
  every move and dependent rewrite from the receipt-bound Git base, enforces the
  current encoder identity, uses no-follow bounded reads and sanitized Git
  inspection, preserves path-only `0644` file semantics in live verification,
  and reuses one digest-bound receipt proof per multi-manifest operation.
