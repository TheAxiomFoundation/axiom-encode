# Manual-mask validation evidence

The `final-*` files are **historical preintegration receipts**, published in
commit `e14154137e92999bd502966d330173816197d496`. Their source and extractor
hashes identify that commit's bytes, before the integration merge
`3714d127e3d1c471cc08325fc8ec8780a7a7b430`. They do not validate reviewed head
`132bf991487f4e7c934e405f94f1ff58caf2a78a` or subsequent fixes.
The filenames retain their historical wording so the recorded hashes remain
checkable.

[Historical focused-check summary](final-checks-summary.json) records 812
passes: 295 generated cases, 14 safety cases, 384 prior regressions, 10 selected
complete-source checks, 44 numeric checks, 64 upstream controls and one enabled
locked-set check. [Normalized logs](final-check-output.txt) and the
[historical reproduction](final-check-reproduction.md) preserve the selections,
settings and expected negative control. `checks-summary.json`,
`check-output.txt` and `check-reproduction.md` describe still earlier bytes;
their counts are excluded from the recorded 812 passes.

The advertised safety file was omitted from both the preintegration commit and
the reviewed head. It is restored as
[`test_source_completeness_manual_mask_review_safety.py`](../../../test_source_completeness_manual_mask_review_safety.py),
with the exact SHA256 recorded in the historical summary:
`cd88b5a5e7c22a042227a5dc7bc0a1453fb3fc947aefb0f10fa25d7166daa2b7`.
Its actual tests are:

- `test_reviewed_financial_titles_cannot_authenticate_an_ambiguous_money_column`
- `test_generated_contents_titles_need_independent_harmless_context`
- `test_flattened_tables_keep_current_main_operative_values`
- `test_generated_flattened_currency_tables_keep_every_amount`

These collect as 14 parametrized nodes. The historical runner requires its
pinned source hashes; it is not a current-head reproduction command. To replay
those historical bytes, first restore the source, metadata and extractor from
`e14154137e92999bd502966d330173816197d496` in a disposable checkout, copy the
restored safety file, and follow `final-check-reproduction.md`. The source
attribution and safety-file identity can be checked without running Python:

```sh
git show e14154137e92999bd502966d330173816197d496:src/axiom_encode/harness/source_completeness.py | shasum -a 256
git show e14154137e92999bd502966d330173816197d496:src/axiom_encode/harness/validator_pipeline.py | shasum -a 256
shasum -a 256 tests/test_source_completeness_manual_mask_review_safety.py
jq '.validated_file_sha256, .current_pass_count' tests/fixtures/source_completeness/manual_mask_fail_safe/final-checks-summary.json
```

Current regression coverage also includes
[`test_source_completeness_manual_mask_fail_safe.py`](../../../test_source_completeness_manual_mask_fail_safe.py).
The new
[`test_source_completeness_manual_page_values.py`](../../../test_source_completeness_manual_page_values.py)
covers the reviewed initial-page amount, referenced-page properties, composition
context and authenticated-locator controls. The seven earlier
manual/locator/replay/typography suites are named in the historical runner.
Restoring the safety file makes its tests available to normal CI;
the [review-round check summary](review-round-checks.md) separately records
966 passing nodes, exact selections, review findings and validation limits.
No assertion is changed by this documentation correction or file restoration.

`corpus-summary.json` records a historical comparison of 68,848 raw rows from
231 files against `4f3547461300bfa70569f57c83728b2ad25d9d0b`, using corpus commit
`8f7d60aaced28ee4252b9237f9d6e02360dc34bc` and preintegration candidate hashes.
The recorded totals are:

| Profile | Changed rows | Added row/value demands | Added occurrences | Removed demands / occurrences / typed entries |
| --- | ---: | ---: | ---: | --- |
| en-US | 244 | 422 | 525 | 0 / 0 / 0 |
| legacy | 661 | 1,008 | 1,680 | 0 / 0 / 0 |

These numbers are reproducible as bundle totals from `corpus-summary.json`,
`corpus-deltas.jsonl` and `corpus-input-manifest.json`. They are recorded
historical results, not a new execution against the integrated or fixed head.
Conservative additions can include locators, publication dates and ambiguous
numeric readings; they are not all recovered operative quantities. Demand
additions count distinct row/value pairs; occurrence additions also include
values already demanded in the same row. Typed comparisons retain
value/kind/unit metadata and ignore raw spelling and character offsets.

`corpus-main-forward-equivalence.json` records that baseline and later observed
main `52c76d75849a628bbfd62b31795fd06e6bf9fdd1` have identical package tree
`93d509588b65207c063b7e47e22d241575d1106f` and lockfiles. That equality can be
checked from Git; it does not turn the historical candidate results into a
current-head replay.

The full `corpus-rows.jsonl.gz` ledger and original corpus input bytes are not
committed here. `corpus-artifacts.json` records the omitted ledger's SHA256
`424c1a35108287db3ab6a97893e729cfa6ac3adac54939a99e8177c4036b66ac`.
The compact delta ledger, input manifest and inert execution/proof snapshots
are committed. [Historical proof review](corpus-independent-proof.md) and
[historical completion review](corpus-independent-completion.md) describe the
prior reports; their presence and bundle consistency do not independently
establish that the omitted full ledger was executed or completed. Follow
[corpus-reproduction.md](corpus-reproduction.md) only after restoring and
authenticating all required inputs and the pinned candidate bytes.

Full raw-corpus replay has not been performed in this fix round. Raw bodies do
not establish a complete composed-book differential or full artifact validation.
Generated public-gate properties separately exercise composition layouts.
These artifacts establish no historical signing event, production outcome,
GitHub closure reason or complete prior review cycle. The integration merge
is already in local history. The hub must commit the fix-round changes and
require green resulting-head CI before landing.
