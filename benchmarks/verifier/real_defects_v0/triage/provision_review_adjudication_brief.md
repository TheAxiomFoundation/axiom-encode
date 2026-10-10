# Provision check, adjudication: two readers disagreed on these cases

You are auditing cases of a benchmark corpus of real encoding defects (axiom-encode PR #1659, `benchmarks/verifier/real_defects_v0`). Report only. Do not edit, create, commit or push anything.

## Background

Each case pairs a RuleSpec YAML module as it stood before a correcting commit (`pre_fix.yaml`) with the module after it (`post_fix.yaml`), plus a provision text (`provision.txt`). The benchmark shows a judge model **only `provision.txt` and one module** and asks whether the module faithfully encodes the provision. The judge should flag the pre-fix module and pass the post-fix one.

`provision.txt` was resolved from the module's **first** citation only. The text that shows the pre-fix module is wrong may sit under another citation the module names, or in none of its cited sources. A case like that cannot be decided by a judge who is shown `provision.txt`. Two readers already made that call for every case, independently. For the cases below they disagreed, or one reader's quote did not check. Your job is to settle each one.

## Where things are

Work directory (read it, do not write): `__WORK__`

Each case has a bundle at `bundles/<case id>/`:

- `CASE.md`: read this first. It has the triage readers' account of what was wrong, the rules that changed, the list of files, and the pre-to-post diff.
- `provision.txt`: the packaged provision (what a judge sees).
- `pre_fix.yaml`, `post_fix.yaml`: the module before and after the fix. `diff.patch` is the full diff when `CASE.md` truncates it.
- `other/NNN.txt`: every other citation that the module, or a proof atom of a changed rule, names, resolved from the same signed corpus release. `CASE.md` lists them with their citation paths, who cites them, and search hints (strings from the fix that occur in the file). Hints are search aids, not findings.

## What to decide for each case

Each case below comes with both readers' records (verdict, quotes, reasoning). They are claims, not findings: one of them is wrong, and both can be. Read the bundle yourself, test each reader's quote against the rules below, and give the verdict the texts support. Agreeing with a reader is fine when the texts bear that reader out; say in `adjudication_note` what decided it and why the other reading fails.



1. Read `CASE.md` and both readers' records below. Understand the defect: what the pre-fix rule did, what the fix changed.
2. Read the changed rules in both modules (the diff shows where).
3. Read `provision.txt` in full. For long provisions, search for the terms the changed rule turns on, then read around every hit.
4. If `provision.txt` does not settle it, read the `other/` files most likely to: first those cited by `rule_post` or `rule_pre`, then those with hints, then the rest by citation path.
5. Give one verdict:

   - `in_provision`: `provision.txt` contains text from which a careful reader, holding only `provision.txt` and `pre_fix.yaml`, can tell that the pre-fix rule is wrong in the respect the fix changed, and that `post_fix.yaml` is right (or closer) in that respect.
   - `in_other_citation`: `provision.txt` does not suffice, but a passage in one or more `other/NNN.txt` files does (alone, or read together with `provision.txt`).
   - `not_in_sources`: no text in the bundle suffices.

Tests to apply before you answer:

- **It must discriminate.** The decisive text fits the post-fix rule and conflicts with the pre-fix rule. Text that both modules satisfy equally is not decisive. If `provision.txt` supports the pre-fix rule (for example it prints the amount the pre-fix module used), the verdict is not `in_provision`.
- **Extra or unsupported conditions.** When the defect is a condition, branch, date or amount the pre-fix rule invented, `provision.txt` is decisive only if it contains the text that governs that rule, so that a reader can see the invented element is absent from it or contradicted by it. If `provision.txt` does not cover the rule's subject at all, a judge would find the post-fix rule just as unsupported, so the verdict is not `in_provision`.
- **Missing clauses.** When the defect is a clause the pre-fix rule left out, that clause must be in the text you quote.
- **The module's own words do not count.** Proof-atom excerpts, summaries and comments inside the YAML are part of the module, not the provision. Decide from the text files. Separately, report in `pre_fix_restates_it` whether the pre-fix module itself quotes your decisive passage in a proof-atom excerpt.
- **No outside knowledge.** Do not fill a gap from what you know about the law. Only the files count.
- **When several files carry the decisive text, pick one.** The provision will be extended with the file you quote, and a judge sees at most the first 16,000 and last 8,000 characters of the result. Prefer the shortest file that carries the passage; between files of similar length, the one a changed rule's proof atom cites (`rule_post`, `rule_pre`). Quote from one `other/` file unless the point needs two.
- **Judge the recorded defect.** Other flaws you notice in either module are out of scope; mention them in `notes` if they matter.

Also say whether the defect is real, in `defect_real`:

- `yes`: the texts in the bundle confirm the pre-fix rule was wrong as the triage describes.
- `no`: the texts in the bundle show the pre-fix rule was right, or as defensible as the post-fix rule.
- `unsure`: the texts in the bundle cannot tell.

`not_in_sources` with `defect_real: unsure` is the normal pairing when the fix rests on a source outside the bundle. Say what it rests on in `missing_basis` (for example "the amending Federal Register notice, not cited by either module", "an imported module's source", "consistency with a sibling rule").

## Quotes

Every quote is machine-checked against the file it names, after collapsing whitespace and ignoring case. A quote that is not found sends the case back.

- Copy contiguous text from the file exactly. Do not paraphrase, correct or re-punctuate.
- 6 to 60 words each, at most 3 per case. Pick the shortest span that carries the point.
- You may elide inside a quote with ` ... ` only when the two parts are within about 300 characters of each other in the file.
- `in_provision`: quote from `provision.txt` only.
- `in_other_citation`: at least one quote from an `other/NNN.txt` file; add a `provision.txt` quote only if both are needed.
- `not_in_sources`: no quote needed; give the nearest passage if one helps.

## Output

Your final message must be one JSON array inside a ```json fence and nothing else: one object per case, in the order the cases are listed below.

```json
[
  {
    "case_id": "<id>",
    "verdict": "in_provision | in_other_citation | not_in_sources",
    "defect_real": "yes | no | unsure",
    "decisive_quotes": [{"file": "provision.txt", "quote": "<verbatim>"}],
    "why_decisive": "<one to three sentences: what the quoted text says, what the pre-fix rule does instead, why a reader can see the mismatch>",
    "missing_basis": "<for not_in_sources: what the correction rests on; else empty>",
    "pre_fix_restates_it": false,
    "confidence": 0.0,
    "notes": "<optional>",
    "adjudication_note": "<what decided it, and why the other reading fails>"
  }
]
```

`confidence` is your probability, 0 to 1, that the verdict is right.

## Hard rules

PATH DISCOVERY (hard rules):
- NEVER run `find`, `rg`, or recursive `grep` rooted at `~`,
  `~/TheAxiomFoundation`, `~/PolicyEngine`, `/tmp`, `/private/tmp`, or any
  cache directory (`~/.cache`, `~/Library/Caches`, `~/.local`, `~/.cache/uv`).
  A permitted `find` has BOTH a scoped subpath and `-maxdepth`; a permitted
  `rg` targets a specific subdirectory.
- Use `axiom-locate` (on PATH) for common artifacts:
  `axiom-locate engine` — newest built axiom-rules-engine binary;
  `axiom-locate release <name>` — corpus release checkouts, newest first;
  `axiom-locate pypkg <name> [--python PATH]` — package path via the interpreter;
  `axiom-locate corpus-file <fragment>` — tracked files across corpus checkouts.
  `--json` for structured output; `--refresh` to bypass the cache; exit 1 = not
  found (trust it — it checks every checkout location in <1s).
- Python package/tool paths come from the interpreter or uv
  (`python -c "import x; print(x.__file__)"`, `uv pip show`, `uv tool run`),
  never from filesystem search.
- Inside a repository, enumerate with `git ls-files`, not find.
- Write test results and artifact paths to a deterministic checkpoint file in
  your lane workdir AT GENERATION TIME; never re-search /tmp for your own
  earlier output.
- Repo copies for e2e/review: `git worktree add`, or `rsync -a` excluding
  `target dist .venv node_modules .git`. Never wholesale `cp -a` of a working
  copy.

Further rules:
- Read only inside the work directory named above. You need nothing else: no network, no Git, no other checkout.
- Search inside one bundle at a time (`grep -n <term> bundles/<case id>/provision.txt`, or `grep -rn <term> bundles/<case id>/other`); never search the whole work directory.
- No model or API calls of your own.
- Do every case listed. If you run short, return the cases you finished; do not guess the rest.

## Cases

__CASES__
