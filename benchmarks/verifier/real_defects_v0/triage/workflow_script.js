export const meta = {
  name: 'real-defects-triage',
  description: 'Screen, triage, and adversarially verify rulespec fix commits for the real-defects verifier corpus',
  phases: [
    { title: 'Screen', detail: 'subject-only pass over non-keyword module commits' },
    { title: 'Triage', detail: 'read each candidate diff and classify per module' },
    { title: 'Verify', detail: 'one adversarial refuter per kept module case' },
  ],
}

const S = args.scratch
const W = args.worktree
const REPO = { us: '/Users/maxghenis/TheAxiomFoundation/rulespec-us', uk: '/Users/maxghenis/TheAxiomFoundation/rulespec-uk' }
const GH = { us: 'TheAxiomFoundation/rulespec-us', uk: 'TheAxiomFoundation/rulespec-uk' }
const CHUNK = 12
const SCREEN_BATCH = 70
const KINDS = ['amount_mismatch', 'boundary_direction', 'unrepresented_clause', 'untraceable_branch', 'wrong_period_or_effective_date', 'wrong_entity_or_scope', 'polarity_or_logic', 'other', 'none']

const SCREEN_SCHEMA = { type: 'object', properties: { flagged: { type: 'array', items: { type: 'object', properties: { commit: { type: 'string' }, n_modules: { type: 'integer' }, reason: { type: 'string' } }, required: ['commit', 'n_modules', 'reason'] } }, read_count: { type: 'integer' } }, required: ['flagged', 'read_count'] }

const TRIAGE_SCHEMA = { type: 'object', properties: {
  commit: { type: 'string' }, jurisdiction: { type: 'string' }, commit_summary: { type: 'string' },
  modules: { type: 'array', items: { type: 'object', properties: {
    index: { type: 'integer' }, path: { type: 'string' },
    classification: { type: 'string', enum: ['fidelity', 'unclear', 'mechanical', 'test_only', 'not_a_rule_module'] },
    defect_kind: { type: 'string', enum: KINDS }, other_kind: { type: 'string' },
    confidence: { type: 'number' },
    description_quote: { type: 'string' }, quote_source: { type: 'string', enum: ['commit_message', 'pr_body', 'pr_review', 'module_comment', 'none'] },
    rule_names: { type: 'array', items: { type: 'string' } }, rule_path: { type: 'string' },
    pre_fix_wrong_because: { type: 'string' }, triage_notes: { type: 'string' },
  }, required: ['index', 'path', 'classification', 'defect_kind', 'confidence', 'description_quote', 'quote_source', 'rule_names', 'rule_path', 'pre_fix_wrong_because', 'triage_notes'] } },
}, required: ['commit', 'jurisdiction', 'commit_summary', 'modules'] }

const VERIFY_SCHEMA = { type: 'object', properties: {
  verdict: { type: 'string', enum: ['confirmed', 'reclassify', 'mechanical', 'not_a_defect', 'unclear'] },
  defect_kind: { type: 'string', enum: KINDS }, other_kind: { type: 'string' },
  confidence: { type: 'number' }, justification: { type: 'string' }, quote: { type: 'string' }, notes: { type: 'string' },
}, required: ['verdict', 'defect_kind', 'confidence', 'justification', 'quote'] }

const RULES = `
Hard rules for every command you run:
- Read-only. Never checkout, reset, stash, commit, or edit anything in the rulespec repositories or the corpus checkout. Only 'git -C <repo> show ...', 'git -C <repo> log ...', 'git -C <repo> diff ...', 'gh api ...', 'python3 ...', 'sed', 'cat'.
- Never run recursive grep, rg, or find over home, ~/TheAxiomFoundation, /tmp, or a repository root (a hook blocks it and it saturates the machine). If you must search inside a repository use 'git -C <repo> grep <pattern> <ref> -- <specific/subdir>'.
- Never use a gpt-5.5 model for anything. Do not delegate; do the reading yourself.
- Never invent a defect kind the evidence does not support. Quotes must be verbatim from the commit message, PR body, PR review, or a comment inside the module diff; never present a paraphrase as a quote.
`

const DEFINITIONS = `
Defect kinds (a fidelity defect is an encoded meaning that disagrees with the cited source text, as evidenced by what the fix changed and what its message says):
- amount_mismatch: a numeric value, rate, threshold, table cell, or unit conversion was wrong versus the source.
- boundary_direction: a comparison used the wrong strictness or direction (< vs <=, > vs >=, inclusive vs exclusive bound, floor/ceiling side).
- unrepresented_clause: a condition, exception, carve-out, deduction, or clause the source states was missing from the encoding (the fix adds it).
- untraceable_branch: the encoding contained a branch, condition, or value with no basis in the cited source (the fix removes or re-grounds it), including formulas grounded on the wrong provision.
- wrong_period_or_effective_date: effective_from/effective_to, version windows, period kinds, or dates disagreed with the source (including timeless encoding of superseded text).
- wrong_entity_or_scope: the rule applied to the wrong entity, population, filing status, geography, program scope, household member set, or output surface.
- polarity_or_logic: boolean logic inverted or mis-combined (and/or, negation, fail-open vs fail-closed, default value) so the wrong branch fires.
- other: a real fidelity defect that fits none of the above; name it in other_kind.

Classifications:
- fidelity: the module change corrects encoded meaning relative to the source, kind supported by the diff or quoted review text.
- unclear: plausibly a fidelity correction but the evidence does not settle it (keep it; a reviewer prunes).
- mechanical: only proof hashes, import hashes/pins, toolchain pins, waiver or ledger bookkeeping, index regeneration, formatting, renames, comments, summaries, module-level metadata, or proof-atom excerpt/citation-path strings changed while formulas, parameters, versions, effective dates, inputs, outputs, and scope stay the same. Excerpt-only and citation-string-only corrections are mechanical (provenance), not fidelity.
- test_only: the module file itself did not change meaning (should not happen for a module file; use only if the diff is empty or whitespace).
- not_a_rule_module: the file is not a RuleSpec rule module (compose spec, registry, ledger, config).
Also: a change that ADDS coverage the source requires (a missing exception, a missing clause) is a fidelity correction (unrepresented_clause). A change that merely adds new outputs, precision, or scope the module never claimed is not a defect; classify unclear at most and say why.
`

function screenPrompt(jur, start, end) {
  return `You are screening commit subjects from ${GH[jur]} for a corpus of REAL encoding defects (fixes to already-merged rule modules).
Read entries ${start} through ${end - 1} (0-based, in array order) of the JSON array in ${S}/${jur}_nonkeyword.json. Each entry has commit, date, subject, n_modules. Read them with one scoped command, for example:
python3 -c "import json; rows=json.load(open('${S}/${jur}_nonkeyword.json'))[${start}:${end}]; [print(r['commit'], r['date'], r['n_modules'], r['subject']) for r in rows]"
Flag every entry whose subject indicates the commit CORRECTS an existing rule module's encoded semantics or source grounding: wrong amount, boundary, clause, branch, effective date/period, entity/scope, polarity or logic, or mis-grounded proof. Verbs that usually mean a correction: fix, repair, correct, tighten, harden, ground, re-anchor, re-ground, retire, time-bound, bound, clarify (a boundary), rework, revert, restore, canonicalize, align, address (review), close (a gap), preserve, limit, require, defer, fail closed.
Do NOT flag: new encodings ("Encode X", "Add X", "Batch-encode", "via bulk dispatcher"), ingest/registry/index/ledger/waiver/toolchain/pin/manifest/signing bookkeeping, renames and moves, formatting, CI or workflow wiring, docs, test-only additions, oracle-coverage bookkeeping.
When genuinely unsure, flag it; a later stage reads the diff. Return read_count = number of entries you read, and flagged = the flagged entries with commit (exactly as given), n_modules, and a one-line reason.
${RULES}`
}

function triagePrompt(item) {
  const repo = REPO[item.jur]
  const idx = item.indices.join(', ')
  return `You are triaging one commit from ${GH[item.jur]} for a corpus of REAL encoding defects used to benchmark fidelity judges (EncodeBench verifier track). Commit ${item.commit} (${item.date}): "${item.subject}". This chunk covers module indices [${idx}] (chunk ${item.chunk + 1} of ${item.chunks}). Classify EVERY module index in the chunk; if the bundle has fewer modules than expected, cover the ones that exist and note it.

Step 1. Load the prepared bundle ${S}/inputs/${item.jur}/${item.commit}.json. If that file does not exist, create it with: cd ${S} && python3 prep_commit.py ${item.jur} ${item.commit}   (idempotent, read-only; it also fetches the merged PR body and review comments through gh). The bundle has commit, parent_commit, subject, body, module_files (index, path, status, diff_file, post_header, corpus_citation_paths_seen, is_rulespec_v1), test_files, other_files, pr (url, body, reviews, review_comments, issue_comments).
Step 2. For each module index in this chunk, read its diff file (cat the diff_file path). When the diff alone does not show what a changed formula, version, or parameter means, read the surrounding pre-fix and post-fix files: git -C ${repo} show <parent_commit>:<path>   and   git -C ${repo} show ${item.commit}:<path>   (use sed -n to window large files).
Step 3. Read the commit body and the PR text in the bundle for the maintainers' or reviewers' stated reason for the change. The stated reason is the primary evidence for the kind; the diff must be consistent with it.
Step 4 (only when the kind depends on what the source says, for example whether < or <= is right, or whether a clause exists): read the cited provision through the pinned corpus release with:
cd ${W} && uv run python ${S}/resolve_provision.py --jurisdiction ${item.jur} --citation <corpus_citation_path> --rulespec-commit ${item.commit}
It prints JSON with the provision text under "text" (may be long; pipe through python3 -c to print a window). If it errors, say so in triage_notes and proceed on the diff plus message.
${DEFINITIONS}
Output one entry per module index with:
- classification and defect_kind (defect_kind 'none' unless classification is fidelity or unclear).
- confidence in [0,1] that the classification AND kind are right. Use >= 0.8 only when the message or review text names the defect and the diff shows exactly that; 0.5-0.79 when the diff shows it but the message is generic; < 0.5 for unclear.
- description_quote: a verbatim sentence or fragment from the commit message, PR body, PR review, or a comment in the module diff that describes the defect or the fix (quote_source says which). If nothing quotable exists, put the empty string and quote_source 'none'.
- rule_names: the rule name(s) whose formula, versions, parameters, effective dates, inputs, or outputs changed. rule_path: a YAML path such as rules[<rule_name>].versions[0].formula or rules[<rule_name>].versions[0].effective_to for the decisive change (one path; the most important if several).
- pre_fix_wrong_because: one sentence, your own words, stating what the pre-fix module got wrong relative to the source, as evidenced (or 'unclear: ...').
- triage_notes: anything a reviewer needs (multi-module fixes, whether the fix also changed unrelated things, whether the provision was read, follow-up commits you noticed, doubts).
Also return commit_summary: two sentences on what the commit as a whole did and why.
Big commits: when many modules in this chunk received an identical pattern (same hunk shape), still read each diff, but you may reuse the justification; say 'same pattern as index N' in triage_notes.
${RULES}`
}

function verifyPrompt(item, tri, m) {
  const repo = REPO[item.jur]
  return `Adversarial verification for one candidate case in a corpus of REAL encoding defects (fixes to rule modules in ${GH[item.jur]}). A first reader classified this module change as "${m.classification}" with defect kind "${m.defect_kind}"${m.other_kind ? ' (' + m.other_kind + ')' : ''} at confidence ${m.confidence}, justified by the quote: "${(m.description_quote || '').slice(0, 600)}" (source: ${m.quote_source}), rule_names ${JSON.stringify(m.rule_names)}, rule_path ${m.rule_path}, and the reading: "${(m.pre_fix_wrong_because || '').slice(0, 500)}".
Commit ${item.commit} (${item.date}) "${item.subject}", module path ${m.path} (module index ${m.index}).
Your job is to try to REFUTE that classification and kind. Read the bundle ${S}/inputs/${item.jur}/${item.commit}.json (commit body, PR text, module_files[${m.index}].diff_file) and cat the diff file. If needed, read the full pre and post files: git -C ${repo} show <parent_commit>:${m.path}   and   git -C ${repo} show ${item.commit}:${m.path} (window with sed -n). When the kind depends on the source text (boundary strictness, presence of a clause, a number), read the provision through the pinned release:
cd ${W} && uv run python ${S}/resolve_provision.py --jurisdiction ${item.jur} --citation <corpus_citation_path from the module header> --rulespec-commit ${item.commit}
${DEFINITIONS}
Verdicts:
- confirmed: the diff changes encoded meaning (formula, parameter value, version window or effective date, input/output scope, boolean logic) in the way the quoted reason describes, and the kind fits. Give the decisive diff line(s) as the quote.
- reclassify: it is a fidelity defect, but a different kind fits better (set defect_kind to your kind; other_kind if 'other').
- mechanical: only proof hashes, import pins, waivers, ledgers, indexes, formatting, renames, comments, summaries, metadata, or proof-atom excerpt/citation strings changed, with formulas, parameters, versions, dates, inputs, outputs, and scope unchanged.
- not_a_defect: the pre-fix content was not wrong against the source; the change is an enhancement, new coverage the module never claimed, or a policy choice with no source basis for calling the old text wrong.
- unclear: the evidence does not settle it. Default to unclear rather than confirmed when the message is generic and the diff is ambiguous.
Return confidence in [0,1] for your verdict, a justification of two to four sentences, the decisive quote (verbatim diff line or message fragment), and notes for a reviewer.
${RULES}`
}

function chunkItems(list) {
  const out = []
  for (const c of list) {
    const n = Math.max(1, c.n_modules | 0)
    const k = Math.ceil(n / CHUNK)
    for (let i = 0; i < k; i++) {
      const indices = []
      for (let j = i * CHUNK; j < Math.min(n, (i + 1) * CHUNK); j++) indices.push(j)
      out.push({ ...c, chunk: i, chunks: k, indices })
    }
  }
  return out
}

const triage = (item) => agent(triagePrompt(item), {
  label: `triage:${item.jur}:${item.commit}${item.chunks > 1 ? '#' + (item.chunk + 1) + '/' + item.chunks : ''}`,
  phase: 'Triage', schema: TRIAGE_SCHEMA, effort: 'high',
})

const verify = async (tri, item) => {
  if (!tri) return { item, triage: null, verdicts: [] }
  const kept = (tri.modules || []).filter(m => m.classification === 'fidelity' || m.classification === 'unclear')
  const verdicts = await parallel(kept.map(m => () =>
    agent(verifyPrompt(item, tri, m), { label: `verify:${item.jur}:${item.commit}:${m.index}`, phase: 'Verify', schema: VERIFY_SCHEMA, effort: 'high' })
      .then(v => ({ module_index: m.index, path: m.path, verdict: v }))))
  return { item, triage: tri, verdicts: verdicts.filter(Boolean) }
}

// Screen (subject-only) runs concurrently with keyword triage; flagged commits then join the triage pipeline.
phase('Screen')
const screenBatches = []
for (const jur of ['us', 'uk']) {
  const n = args.nonkeyword_counts[jur]
  for (let s = 0; s < n; s += SCREEN_BATCH) screenBatches.push({ jur, start: s, end: Math.min(n, s + SCREEN_BATCH) })
}
const screenP = parallel(screenBatches.map(b => () =>
  agent(screenPrompt(b.jur, b.start, b.end), { label: `screen:${b.jur}:${b.start}-${b.end}`, phase: 'Screen', schema: SCREEN_SCHEMA, effort: 'medium' })
    .then(r => ({ ...b, result: r }))))

phase('Triage')
const keywordItems = chunkItems(args.keyword)
log(`keyword triage: ${args.keyword.length} commits -> ${keywordItems.length} chunks`)
const keywordP = pipeline(keywordItems, triage, verify)

const screens = (await screenP).filter(Boolean)
const seen = new Set(args.keyword.map(k => k.jur + ':' + k.commit))
const flagged = []
for (const s of screens) {
  for (const f of (s.result && s.result.flagged) || []) {
    const key = s.jur + ':' + f.commit
    if (seen.has(key)) continue
    seen.add(key)
    flagged.push({ jur: s.jur, commit: f.commit, date: '', subject: '(screen-flagged) ' + f.reason, n_modules: f.n_modules, screen_reason: f.reason })
  }
}
const readCount = screens.reduce((a, s) => a + ((s.result && s.result.read_count) || 0), 0)
log(`screen: ${screens.length} of ${screenBatches.length} batches returned, ${readCount} subjects read, ${flagged.length} new commits flagged`)
const flaggedItems = chunkItems(flagged)
const flaggedResults = await pipeline(flaggedItems, triage, verify)
const keywordResults = await keywordP

return {
  screen: { batches: screenBatches.length, returned: screens.length, read_count: readCount, flagged },
  keyword: keywordResults.filter(Boolean),
  flagged: flaggedResults.filter(Boolean),
}