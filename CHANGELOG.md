## 0.2.2197

- Publish signed targeted re-encodes onto a rulespec main tip that advanced from the recorded base only through repository plumbing. The lane is replayed onto that tip by object identity, and every signed byte is proven unchanged for each replayed commit. Receipt-bound or deleting lanes, file/directory conflicts, and any other advance still fail closed. The SNAP dispatcher keeps unfinished runs on such an advance instead of cancelling them.
## 0.2.2214

- Regenerate rulespec's derived provision-to-rules reverse index at publish, as one unsigned commit from a byte-exact replica of the tree's own generator, so signed PRs that add a citation pass the reverse-index check.
## 0.2.2193

- Keep US bibliographic locators out of complete-source numeric recall: Federal Register volume/page citations, `Pub. L.` public-law numbers, the public-law and Statutes at Large numbers of complete U.S. Code repeal, omission and transfer notes, and years in act titles. The recall text is not rewritten; an occurrence is dropped only when it lies wholly inside a locator's numbers, the citation's context is bibliographic, and a re-read with those numbers blanked agrees on every other occurrence. Otherwise the inventory stays unchanged.
## 0.2.2198

- Read proof excerpts that differ from the cited source only typographically (ligatures, quotation-mark and in-word dash variants, wrapped compound and line-end hyphens, a space after a currency or section sign, a space before a comma, enclosing quotation marks, a recased first letter) as the printed source text, recognize the Code's "($X in the case of a joint return)" parenthetical as prose, and scope fused "(1)(a)" markers; ellipsis, paraphrase, stitched table text and sign changes stay rejected. State the contiguous-verbatim excerpt contract in the encoder prompts, and drop the bare `imports:` header when unused-import pruning removes every import.
## 0.2.2192

- Label U.S. and other non-German complete-source branches in English with their full designation (`us/regulation/7/273/4(a)(4) [paragraph (a)(4)]`); German labels are unchanged.
- Stop reading German sentence markers in U.S. sources (`K-40ES`, `42CFR`, `22VAC40` were "Satz" branches), and let a principal output citing a U.S. list or range such as `7 CFR 273.4(a)(3)-(5)` cover each named branch. A list counts only when nothing after it qualifies it (`in 7 CFR 273.9`, `of section 152`), and an out-of-order roman (`(a)(1)(ii), (i)`) never reaches a top-level letter.
## 0.2.2215

- List the branches that only a rejected (imprecise) deferral would cover inside that deferral's `[complete-source-unit:deferral]` finding instead of repeating one `[complete-source-unit:structure]` finding per branch; they are reported on their own again when the deferral is removed, and retry ranking and the dependent-regression check still count each one.
- Show U.S. encodes a U.S. example (26 U.S.C. 24(c)(1) citing 152(c)) in the deferral retry shape instead of the EStG example.
## 0.2.2217

- Locate every missing numeric-recall value in the first complete-source numeric-recall diagnostic (source span, bounded excerpt, grouped by branch, locator-shaped values listed apart, capped) so retry feedback shows the full set. Demands and issue counts are unchanged.
## 0.2.2199

- Read conventional CFR branch citations (`42 CFR 457.622(c)(1)`) in completeness deferral reasons for repeal, runtime-gap, and non-operative deferrals, with the same exact-branch guard as the corpus-path form; CFR and Act dependency deferrals are not opened by this reader. Accept bounded non-operative deferrals of single-sentence Basis/Scope/Purpose/Findings/Severability/Short title leaves. Check branch deferrals anchored at the manual, form, and dotted-regulation module root instead of dropping them. A regulation runtime gap now covers a branch only when none of its covered text matches a closed lexical list of amount, computation, eligibility, or benefit-consequence phrases and every covered leaf states, or inherits from its chapeau, a finite administrative act on an administrative artifact. Correct the completeness prompt sentence that said `7 CFR 273.10(a)(1)` does not identify a branch.
## 0.2.2219

- Bind a session-law ("section N of this act") deferral only to a reference in the branch's own text (not descendants, unrecognized siblings, later session-law sections or compiler notes) that an exception or condition governs ("except as provided in", "unless ... under", "subject to the provisions/conditions ... of").

## 0.2.2218

- Scope the session-law ("section N of this act") deferral reader to statute and guidance law-compilation sources that self-reference their own act. Regulations, policies and manuals that name "section N of the Act" (a codified enabling statute) no longer pass as same-act dependencies.
## 0.2.2216

- Let a parameter (or other non-principal rule) cover the source branch it cites only when the branch is a stated value: the branch text states one of the rule's values and the branch's own text states no condition, exception or limit. A parameter holding a threshold no longer covers the eligibility condition or exemption that states it; a principal output or precise deferral must account for that branch.
## 0.2.2212

- Count a case pair as the witness for a source exception only when the toggled input matches the exception's own controlling condition, not words in its consequence (7 CFR 273.4(c)(5) same-sponsor clause). The condition is narrowed only for `if`/`when`/`whenever`/`unless`/`except`/`provided that` cues whose consequence's own predicate states an outcome; requirement consequences, reservations and conditions without a clear end keep the whole clause.
## 0.2.2210

- In federal US statutes, regulations and guidance, stop counting administrative condition sentences (a notice, verification, Federal Register publication or agency workflow step as the main verb of the main clause, a definitions lead-in, and the guidance-portal disclaimer) as complete-source-unit controls or paired-case conditions. Sentences with a numeric limit, a computation, an outcome after their leading condition or after a `;`, notice content with its own outcome, a trailing exception, or a second predicate after the administrative verb stay controls; administrative sentences still compete for case pairs; federal forms, policies, rulemaking, `/manual/` paths, state and non-US sources are unchanged.
## 0.2.2211

- Report an unbound computation or an unencoded branch once (formula-output or structure) instead of repeating it as a companion-test demand, keep the named-rule binding hint on the formula-output finding, and give each missing boundary its source text, covering outputs and directly compared inputs.
- Count a negative-input case as `max(0, x)` evidence only when the clamp is on the owner's executed formula path, and fail closed when that path cannot be resolved.
- Let the eval repair loop clone an existing model-written case toward an unmet source boundary, with every expected value taken from executing the module. Keep an `auto_source_boundary_*` case only when every output its source case asserts agrees with the engine and the analysis removes exactly the targeted boundary demand. Clamp, formula-branch, unasserted-output and paired exception demands are never synthesized.
## 0.2.2213

- Demand every source-stated threshold of a conjoined condition as a companion-test boundary. `less than $500 and whose net income is less than $300` used to demand only `$300`, so a wrong `<=` on `$500` passed; each dropped threshold is now demanded with its own comparator's direction and inclusivity.

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
