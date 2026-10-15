# Procedure sequencing exemption against main

The paired-condition predicate exempts two complete, normalized clauses from
Florida's calculation-of-benefits manual page 8. One directs a worker to
calculate budgets after determining available income; the other directs a
worker to compare determined gross income with the applicable limit. The key
requires the exact canonical citation and complete clause after the existing
case and whitespace normalization, plus either the authenticated complete page
or the exact isolated complete clause. Altered wording or surrounding
qualifications preserve main's verdict.

This change is based directly on main
`c7fe642eee36ba4e1a766d3eae8a9f3bb31b9ed3`. Main's current companion gate first
filters US manual clauses to stated numeric limits. Neither reviewed clause
matches that filter. Consequently the production corpus differential is
**0 demands removed and 0 added**. The diagnostic that runs the same extraction
without that numeric filter removes exactly the two reviewed predicate demands
and adds none. The expected two removals therefore describe the unfiltered
predicate, while production already excludes those clauses on main. The PR
does not disable or relax main's filter.

The differential rule is exact: enabling the list may remove only demands with
one of the two independently enumerated citation, complete-clause, and
authenticated-source keys. Other keys must have the same verdict with the list
enabled and empty. This includes main's existing witness limitations tracked in
[#1920](https://github.com/TheAxiomFoundation/axiom-encode/issues/1920). These two
entries should move to the reviewed exemption list tracked in
[#1921](https://github.com/TheAxiomFoundation/axiom-encode/issues/1921) once draft
[#1923](https://github.com/TheAxiomFoundation/axiom-encode/pull/1923) lands.

The census authenticates and reads all **39,375 rows in 117 files**, totaling
**146,278,276 bytes**, from immutable corpus revision
[`8f7d60aaced28ee4252b9237f9d6e02360dc34bc`](https://github.com/TheAxiomFoundation/axiom-corpus/tree/8f7d60aaced28ee4252b9237f9d6e02360dc34bc/data/corpus/provisions).
The [inventory](corpus-source-binding.json) binds every input by SHA-256 and byte
count. The [result](corpus-differential.json) records the pinned source digests,
whole-module AST proof, full candidate obligations from each revision, and every
changed path/span/clause identity.

The reproduction script proves that the entire production module matches main
after removing only the exact reviewed exemption constants, helper, guard, and
its two context arguments. It also rejects changes to other production files.
The exact citation guard proves equality for all rows outside the canonical
citation. Every row at that citation executes both revisions' unchanged clause
extraction and formula suppression under the same manual context, with the
production numeric filter enabled on both sides. A separately labeled diagnostic
disables that filter on both sides. No extraction acceleration is needed.

The exemption precedes USDA footer stripping. The two exact clauses contain no
footer fragment, so stripping is the identity on both. Every other key fails the
exemption and reaches footer handling that the whole-module proof confirms is
identical to main. The receipt records this order check.

Reuse the existing authenticated corpus inputs when available. Downloads and
Python runs are sequential, and each invocation must finish within ten minutes.
The census evaluates only exact-citation candidates after its equality proof;
it still reads and authenticates the complete inventory in each run.

```sh
umask 022
export UV_CACHE_DIR=/private/tmp/hub-shared-uv-cache
# If inputs are absent, fetch bounded batches, continuing through 105 117:
uv run python docs/review-evidence/manual-procedure-conditions/fetch-inputs.py 0 15 --output-dir .hub-scratch/round6-corpus/provisions
uv run python docs/review-evidence/manual-procedure-conditions/reproduce-corpus.py --corpus-root .hub-scratch/round6-corpus/provisions --output docs/review-evidence/manual-procedure-conditions/corpus-differential.json
```

`reproduce-latest-main.py` is a compatibility entry point accepting the same
arguments and running the same pinned-main comparison.
