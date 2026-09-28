# CFR roman-ambiguous top-level slicing: differential evidence

Baseline: `3fdae7b5` (`src/axiom_encode/corpus_resolver.py`). All corpus files were read only. This records limitations as well as improvements; the literal requirement that every formerly non-`None` slice remain identical is **not satisfied**, because the old slicer returned text beyond real top-level boundaries.

## Reproduction

```bash
git show 3fdae7b5:src/axiom_encode/corpus_resolver.py > /tmp/resolver-before.py
uv run --python 3.13 python scripts/check_cfr_slice_differential.py \
  --baseline-resolver /tmp/resolver-before.py \
  --corpus-root /path/to/axiom-corpus \
  --output /tmp/cfr-sections.json
# Reproduce the original broader sample, including recovery containers:
uv run --python 3.13 python scripts/check_cfr_slice_differential.py \
  --baseline-resolver /tmp/resolver-before.py \
  --corpus-root /path/to/axiom-corpus --include-recovery-blocks \
  --output /tmp/cfr-original-sample.json
```

Both commands deliberately exit nonzero if any baseline success changes. The report includes input file hashes, resolver hashes, every changed success, every remaining SNAP mismatch, and all gained/lost SNAP matches. Pure immutable helper results are memoized by all arguments to avoid repeating delimiter scans and reference checks for each anchor; `--no-memoize` disables that optimization. Parser state is never memoized.

## SNAP anchor oracle

Inputs under `data/corpus/`:

- `anchors/us/regulation/2026-05-10-snap-7-cfr-273.jsonl`: 192 anchors from `provision-anchors/1.0.0`.
- `provisions/us/regulation/2026-05-10-snap-7-cfr-273-r2026-07-15-self-contained.jsonl`: corresponding section bodies.

Every anchor's `parent_body_sha256` agrees with the selected body. Whitespace-normalized exact matches improve from **30/192 to 42/192**. The list of previously matching anchors that cease to match is **empty**. The 12 gains, all under `us/regulation/7/273/9/`, are `b`, `b/5`, `c/1/i`, `c/1/i/A` through `c/1/i/F`, and `c/1/ii/A` through `c/1/ii/C`.

All 150 remaining mismatches belong to section 273.9. All anchor texts equal their own recorded source spans after whitespace normalization, but that does not establish that their citation paths or paragraph extents are correct.

### A separate malformed-source blocker

The section body SHA256 is `dadc4c07e145dc62382e602c2d5a65f1353e6c6833bdd0eb549bf207d3cd5692`. At character 13916, paragraph `(c)(1)(ii)(D)` begins. Its text contains `payments (as defined in paragraph (c)(1)(i)(F) of this section; or` without a closing prose parenthesis. The opening parenthesis is at character 13950. The next `(E)` marker, at character 14009, therefore falls inside an unclosed delimiter.

The existing delimiter replay correctly reports the subsequent source markers for `(c)(17)` (character 32995), `(d)` (36265), and `(d)(5)` (43314) as enclosed. They have strong boundaries but fail the directly-structural requirement. Preserving delimiter replay therefore leaves these requests unresolved in the unmodified full section; this fix alone does **not** unblock d324 against that corpus body. Ancestor slices that begin before the defect can overrun because their closing boundaries are enclosed too.

An independent diagnostic inserted only the missing closing parenthesis into an in-memory copy. With that repair and the fixed slicer, `c/17` resolves to 444 characters and `d/5` to 784 characters, including trailing blank lines. No corpus file was changed, and repaired text is not used in the reported oracle or sweep.

### Each remaining anchor mismatch

The table uses suffixes relative to `us/regulation/7/273/9/`. `Enclosure` means a valid anchor is blocked, or its ending boundary is suppressed, by the source defect above. `Inline` is the pre-existing limitation on `(b)(1)(i)`, whose marker follows `Earned income shall include:` on the same line without sufficient structural evidence. `Path` and `Span` identify anchor defects, in addition to the source enclosure that prevents evaluating the later region normally.

The anchor extractor fails to return to shallower numbered/lettered siblings: it places `(c)(2)` onward under `(c)(1)(vii)(C)`; `(c)(3)(iii)` onward under `(c)(3)(ii)(C)(10)`; `(c)(6)` onward under `(c)(5)(ii)(B)`; and `(d)(6)(iii)(E)` onward under `(d)(6)(iii)(D)(3)(vii)`. The intended paths in the table follow the section's numbering and paragraph content, not the resolver's output. Parent spans at those false nesting transitions also absorb later siblings.

There are **87 anchors with path/span defects**, **62 otherwise valid anchors affected by delimiter replay**, and **1 unsupported inline marker**. Every mismatch is listed below.

| Anchor suffix | Intended paragraph | Explanation |
| --- | --- | --- |
| `b/1/i` | `b/1/i` | Inline |
| `c` | `c` | Enclosure |
| `c/1` | `c/1` | Span; also enclosure |
| `c/1/ii` | `c/1/ii` | Enclosure |
| `c/1/ii/D` | `c/1/ii/D` | Enclosure |
| `c/1/ii/E` | `c/1/ii/E` | Enclosure |
| `c/1/iii` | `c/1/iii` | Enclosure |
| `c/1/iv` | `c/1/iv` | Enclosure |
| `c/1/v` | `c/1/v` | Enclosure |
| `c/1/vi` | `c/1/vi` | Enclosure |
| `c/1/vii` | `c/1/vii` | Span; also enclosure |
| `c/1/vii/A` | `c/1/vii/A` | Enclosure |
| `c/1/vii/B` | `c/1/vii/B` | Enclosure |
| `c/1/vii/C` | `c/1/vii/C` | Span; also enclosure |
| `c/1/vii/C/2` | `c/2` | Path; also enclosure |
| `c/1/vii/C/3` | `c/3` | Path; also enclosure |
| `c/1/vii/C/3/ii` | `c/3/ii` | Path + Span; also enclosure |
| `c/1/vii/C/3/ii/A` | `c/3/ii/A` | Path; also enclosure |
| `c/1/vii/C/3/ii/B` | `c/3/ii/B` | Path; also enclosure |
| `c/1/vii/C/3/ii/B/1` | `c/3/ii/B/1` | Path; also enclosure |
| `c/1/vii/C/3/ii/B/2` | `c/3/ii/B/2` | Path; also enclosure |
| `c/1/vii/C/3/ii/B/3` | `c/3/ii/B/3` | Path; also enclosure |
| `c/1/vii/C/3/ii/B/4` | `c/3/ii/B/4` | Path; also enclosure |
| `c/1/vii/C/3/ii/B/5` | `c/3/ii/B/5` | Path; also enclosure |
| `c/1/vii/C/3/ii/C` | `c/3/ii/C` | Path + Span; also enclosure |
| `c/1/vii/C/3/ii/C/1` | `c/3/ii/C/1` | Path; also enclosure |
| `c/1/vii/C/3/ii/C/2` | `c/3/ii/C/2` | Path; also enclosure |
| `c/1/vii/C/3/ii/C/3` | `c/3/ii/C/3` | Path; also enclosure |
| `c/1/vii/C/3/ii/C/4` | `c/3/ii/C/4` | Path; also enclosure |
| `c/1/vii/C/3/ii/C/5` | `c/3/ii/C/5` | Path; also enclosure |
| `c/1/vii/C/3/ii/C/6` | `c/3/ii/C/6` | Path; also enclosure |
| `c/1/vii/C/3/ii/C/7` | `c/3/ii/C/7` | Path; also enclosure |
| `c/1/vii/C/3/ii/C/8` | `c/3/ii/C/8` | Path; also enclosure |
| `c/1/vii/C/3/ii/C/9` | `c/3/ii/C/9` | Path; also enclosure |
| `c/1/vii/C/3/ii/C/10` | `c/3/ii/C/10` | Path + Span; also enclosure |
| `c/1/vii/C/3/ii/C/10/iii` | `c/3/iii` | Path; also enclosure |
| `c/1/vii/C/3/ii/C/10/iv` | `c/3/iv` | Path; also enclosure |
| `c/1/vii/C/3/ii/C/10/v` | `c/3/v` | Path; also enclosure |
| `c/1/vii/C/4` | `c/4` | Path; also enclosure |
| `c/1/vii/C/5` | `c/5` | Path + Span; also enclosure |
| `c/1/vii/C/5/i` | `c/5/i` | Path; also enclosure |
| `c/1/vii/C/5/i/A` | `c/5/i/A` | Path; also enclosure |
| `c/1/vii/C/5/i/B` | `c/5/i/B` | Path; also enclosure |
| `c/1/vii/C/5/i/C` | `c/5/i/C` | Path; also enclosure |
| `c/1/vii/C/5/i/D` | `c/5/i/D` | Path; also enclosure |
| `c/1/vii/C/5/i/E` | `c/5/i/E` | Path; also enclosure |
| `c/1/vii/C/5/i/F` | `c/5/i/F` | Path; also enclosure |
| `c/1/vii/C/5/ii` | `c/5/ii` | Path + Span; also enclosure |
| `c/1/vii/C/5/ii/A` | `c/5/ii/A` | Path; also enclosure |
| `c/1/vii/C/5/ii/B` | `c/5/ii/B` | Path + Span; also enclosure |
| `c/1/vii/C/5/ii/B/6` | `c/6` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/7` | `c/7` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/8` | `c/8` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/9` | `c/9` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/10` | `c/10` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/10/i` | `c/10/i` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/10/ii` | `c/10/ii` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/10/iii` | `c/10/iii` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/10/iv` | `c/10/iv` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/10/v` | `c/10/v` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/10/vi` | `c/10/vi` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/10/vii` | `c/10/vii` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/10/viii` | `c/10/viii` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/10/ix` | `c/10/ix` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/10/x` | `c/10/x` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/11` | `c/11` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/11/i` | `c/11/i` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/11/ii` | `c/11/ii` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/12` | `c/12` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/13` | `c/13` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/14` | `c/14` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/15` | `c/15` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/16` | `c/16` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/17` | `c/17` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/18` | `c/18` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/19` | `c/19` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/19/i` | `c/19/i` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/19/ii` | `c/19/ii` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/19/iii` | `c/19/iii` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/19/iv` | `c/19/iv` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/19/v` | `c/19/v` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/19/vi` | `c/19/vi` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/19/vii` | `c/19/vii` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/19/viii` | `c/19/viii` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/19/ix` | `c/19/ix` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/20` | `c/20` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/20/i` | `c/20/i` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/20/ii` | `c/20/ii` | Path; also enclosure |
| `c/1/vii/C/5/ii/B/20/iii` | `c/20/iii` | Path; also enclosure |
| `d` | `d` | Enclosure |
| `d/1` | `d/1` | Enclosure |
| `d/1/i` | `d/1/i` | Enclosure |
| `d/1/ii` | `d/1/ii` | Enclosure |
| `d/1/iii` | `d/1/iii` | Enclosure |
| `d/2` | `d/2` | Enclosure |
| `d/3` | `d/3` | Enclosure |
| `d/3/i` | `d/3/i` | Enclosure |
| `d/3/ii` | `d/3/ii` | Enclosure |
| `d/3/iii` | `d/3/iii` | Enclosure |
| `d/3/iii/A` | `d/3/iii/A` | Enclosure |
| `d/3/iii/B` | `d/3/iii/B` | Enclosure |
| `d/3/iv` | `d/3/iv` | Enclosure |
| `d/3/v` | `d/3/v` | Enclosure |
| `d/3/vi` | `d/3/vi` | Enclosure |
| `d/3/vii` | `d/3/vii` | Enclosure |
| `d/3/viii` | `d/3/viii` | Enclosure |
| `d/3/ix` | `d/3/ix` | Enclosure |
| `d/3/x` | `d/3/x` | Enclosure |
| `d/4` | `d/4` | Enclosure |
| `d/4/i` | `d/4/i` | Enclosure |
| `d/4/ii` | `d/4/ii` | Enclosure |
| `d/4/iii` | `d/4/iii` | Enclosure |
| `d/5` | `d/5` | Enclosure |
| `d/6` | `d/6` | Enclosure |
| `d/6/i` | `d/6/i` | Enclosure |
| `d/6/ii` | `d/6/ii` | Enclosure |
| `d/6/ii/A` | `d/6/ii/A` | Enclosure |
| `d/6/ii/B` | `d/6/ii/B` | Enclosure |
| `d/6/ii/C` | `d/6/ii/C` | Enclosure |
| `d/6/ii/D` | `d/6/ii/D` | Enclosure |
| `d/6/ii/E` | `d/6/ii/E` | Enclosure |
| `d/6/iii` | `d/6/iii` | Enclosure |
| `d/6/iii/A` | `d/6/iii/A` | Enclosure |
| `d/6/iii/A/1` | `d/6/iii/A/1` | Enclosure |
| `d/6/iii/A/2` | `d/6/iii/A/2` | Enclosure |
| `d/6/iii/A/3` | `d/6/iii/A/3` | Enclosure |
| `d/6/iii/B` | `d/6/iii/B` | Enclosure |
| `d/6/iii/C` | `d/6/iii/C` | Enclosure |
| `d/6/iii/C/1` | `d/6/iii/C/1` | Enclosure |
| `d/6/iii/C/2` | `d/6/iii/C/2` | Enclosure |
| `d/6/iii/C/3` | `d/6/iii/C/3` | Enclosure |
| `d/6/iii/C/4` | `d/6/iii/C/4` | Enclosure |
| `d/6/iii/C/5` | `d/6/iii/C/5` | Enclosure |
| `d/6/iii/D` | `d/6/iii/D` | Span; also enclosure |
| `d/6/iii/D/1` | `d/6/iii/D/1` | Enclosure |
| `d/6/iii/D/2` | `d/6/iii/D/2` | Enclosure |
| `d/6/iii/D/3` | `d/6/iii/D/3` | Span; also enclosure |
| `d/6/iii/D/3/i` | `d/6/iii/D/3/i` | Enclosure |
| `d/6/iii/D/3/ii` | `d/6/iii/D/3/ii` | Enclosure |
| `d/6/iii/D/3/iii` | `d/6/iii/D/3/iii` | Enclosure |
| `d/6/iii/D/3/iv` | `d/6/iii/D/3/iv` | Enclosure |
| `d/6/iii/D/3/v` | `d/6/iii/D/3/v` | Enclosure |
| `d/6/iii/D/3/vi` | `d/6/iii/D/3/vi` | Enclosure |
| `d/6/iii/D/3/vii` | `d/6/iii/D/3/vii` | Span; also enclosure |
| `d/6/iii/D/3/vii/E` | `d/6/iii/E` | Path; also enclosure |
| `d/6/iii/D/3/vii/F` | `d/6/iii/F` | Path; also enclosure |
| `d/6/iii/D/3/vii/G` | `d/6/iii/G` | Path; also enclosure |
| `d/6/iii/D/3/vii/G/2` | `d/6/iii/G/2` | Path; also enclosure |
| `d/6/iii/D/3/vii/G/3` | `d/6/iii/G/3` | Path; also enclosure |
| `d/6/iii/D/3/vii/H` | `d/6/iii/H` | Path; also enclosure |

## Bounded regression sweeps

Only 222 actual anchors are present locally (192 SNAP, 18 USC 1401, 12 CFR 1401). To construct 2,000 distinct requests, each sweep includes all 222 exact section/body-hash/anchor-path requests, then draws 1,778 requests from anchor-relative child-path patterns applied to local provision section bodies. Every component label must occur in the body. Generated bodies are at most 16,000 characters; the actual anchor requests are not subject to that size cap. The random seed is `20260928`. Statute and regulation candidates are interleaved. Different section-body versions are identified by their SHA256 and counted as distinct source requests.

The canonical-section sweep requires a numeric US title and section-shaped citation depth. The original broader sweep also included recovery containers that happen to have that path depth. Both are retained so the initial observed outcomes are not hidden.

| Sweep | Requests | Section bodies | Before successes | Byte-identical successes | Shortened successes | Success → error | New successes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Canonical sections | 2,000 | 514 | 550 | 527 | 23 | 0 | 46 |
| Original, including recovery containers | 2,000 | 597 | 509 | 487 | 20 | 2 | 44 |

Each sample contains 907 statute requests and 1,093 regulation requests. Canonical outcomes change from 550 success / 1,441 missing / 9 errors to 596 success / 1,399 missing / 5 errors. Original outcomes change from 509 success / 1,485 missing / 6 errors to 551 success / 1,445 missing / 4 errors. No originally successful request becomes missing in either sample.

The tested fixed resolver SHA256 is `8927be903c279801861e0fb949f21f41b74ea0a620b7bf4dee5412d10175a1f3`. Every shortening below was independently checked against the original body: the new result is an exact prefix of the old result, and the removed suffix starts at the displayed next top-level marker. For example, the old 273.9(b) slice included the heading and opening paragraph of (c); its corrected 6,582-character slice now matches the independent anchor after whitespace normalization.

### Every changed baseline success

Character counts include original whitespace. Body hashes identify provision versions; complete hashes and file locations are in the generated JSON reports. `Both` means the request occurs in both samples, not that it is counted twice within either sample.

| Requested citation | Body SHA256 prefix | Sample | Before → after characters | Explanation |
| --- | --- | --- | ---: | --- |
| `us/regulation/7/273/9/b` | `dadc4c07e145` | Both | 7354 → 6582 | Corrected overrun into `(c)` |
| `us/regulation/26/1/1401-1/b` | `72e7a74bbe8a` | Both | 4361 → 732 | Corrected overrun into `(c)` |
| `us/statute/26/21/b` | `5d8f8dcfd94b` | Both | 4373 → 2565 | Corrected overrun into `(c)` |
| `us/statute/42/1397kk/b` | `8a6f4bceca15` | Both | 8072 → 7101 | Corrected overrun into `(c)` |
| `us/statute/7/2028/b` | `888af408c443` | Both | 5835 → 4937 | Corrected overrun into `(c)` |
| `us/statute/26/1212/b` | `a9eba5304cb2` | Both | 5372 → 1410 | Corrected overrun into `(c)` |
| `us/statute/26/3134/c` | `6e1eefe9070a` | Both | 6536 → 6324 | Corrected overrun into `(d)` |
| `us/statute/42/1396u–1/b` | `534f128467d3` | Both | 4035 → 2588 | Corrected overrun into `(c)` |
| `us/regulation/recovery/release-scope-us-regulation-2026-06-03-cms-2454-ifc-42-cfr-435-community-engagement/block-177/d` | `11e54ef395c5` | Original | 913 → error | Recovery container spans multiple amended sections; new recognized (d) exposes later backward (a), so existing ambiguity guard rejects it. |
| `us/statute/26/55/b` | `3bc21791a7a2` | Both | 9092 → 3587 | Corrected overrun into `(c)` |
| `us/statute/7/2022/b` | `465e7936b6c3` | Both | 5855 → 4274 | Corrected overrun into `(c)` |
| `us/statute/42/18795a/c` | `a13395fc705c` | Both | 7872 → 5228 | Corrected overrun into `(d)` |
| `us/statute/42/1397ff/b` | `347ab2473295` | Both | 3004 → 1516 | Corrected overrun into `(c)` |
| `us/regulation/recovery/release-scope-us-regulation-2026-06-03-cms-2454-ifc-42-cfr-conforming-amendments/block-177/d` | `11e54ef395c5` | Original | 913 → error | Recovery container spans multiple amended sections; new recognized (d) exposes later backward (a), so existing ambiguity guard rejects it. |
| `us/statute/26/67/c` | `0a12fe22ec4f` | Both | 2088 → 1585 | Corrected overrun into `(d)` |
| `us/statute/26/56/b` | `446d9f98e924` | Both | 6540 → 4097 | Corrected overrun into `(c)` (editorially bracketed) |
| `us/statute/26/22/c` | `01c0c8460ca5` | Both | 3272 → 2867 | Corrected overrun into `(d)` |
| `us/statute/7/2036a/c` | `1e63948bcd04` | Both | 10789 → 6614 | Corrected overrun into `(d)` |
| `us/statute/26/2/b` | `501ecb62404a` | Both | 2696 → 2308 | Corrected overrun into `(c)` |
| `us/regulation/42/457/1270/b/3` | `7d99c54865b6` | Original | 786 → 284 | Corrected overrun into `(c)` |
| `us/statute/12/1467b/b` | `8b7c2611119f` | Both | 5268 → 3958 | Corrected overrun into `(c)` |
| `us/regulation/47/54/1014/b/3` | `b7fff0bfd845` | Original | 2729 → 2216 | Corrected overrun into `(c)` |
| `us/regulation/42/435/541/c` | `8b6599674682` | Canonical | 2839 → 2490 | Corrected overrun into `(d)` |
| `us/regulation/47/54/705/b` | `375beec51d6b` | Canonical | 1921 → 1585 | Corrected overrun into `(c)` |
| `us/regulation/42/435/121/b/3` | `f704659221af` | Canonical | 1945 → 782 | Corrected overrun into `(c)` |
| `us/regulation/42/457/1270/b` | `7d99c54865b6` | Canonical | 1457 → 955 | Corrected overrun into `(c)` |
| `us/regulation/47/54/2013/c` | `13da4f14aadc` | Canonical | 2502 → 1512 | Corrected overrun into `(d)` |

The two recovery-container errors refer to the same amendment text stored under two release-scope paths. It contains a (d) paragraph from one section, instructions removing §457.344 and adding §457.960, then a new (a)–(c) sequence. The previous 913-character result wrongly included the amendment instructions and the next section's introductory text. Once (d) is recognized as a top-level boundary, the existing backward-marker guard refuses to treat this multi-section container as a single section. These observed changes remain reported even though those rows are excluded by the canonical-section predicate.
The shortened results all discard text beginning at a newly recognized next top-level boundary. This corrects old overruns, but it conflicts with the requested invariant that all old non-`None` results be byte-identical. No override is assumed. The draft PR must retain this blocker and the unresolved malformed-source issue above.
