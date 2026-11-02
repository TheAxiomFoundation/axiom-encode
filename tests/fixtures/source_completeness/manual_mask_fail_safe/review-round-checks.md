# Review round validation

This is the author's historical report of fix-round checks based on reviewed
head `132bf991487f4e7c934e405f94f1ff58caf2a78a`. The **966-pass total, intermediate
red/green counts and earlier review outcomes below are author-reported**: their
underlying execution logs and review transcripts are not committed in this
repository. The listed selections are rerunnable, but this report alone does not
independently establish that those historical executions or reviews occurred.
The separate author-reported 812-pass preintegration receipts also describe
historical bytes.
The hub must validate its resulting commit after its version bump; neither
historical report establishes current-head results or remote CI status.

The author reported that all final pytest runs passed without skips or assertion
failures, excluding repeated intermediate runs from the **966 distinct passing
nodes** below.
Hypothesis settings and explicit examples in the existing suites were retained.

| Selection | Passed | Deselected |
| --- | ---: | ---: |
| Page-value regressions and restored review-safety suite | 108 | 0 |
| Existing locator/manual/typography suites and manual controls/prose | 469 | 0 |
| Generated suite: `monetary_caption or round_five` | 232 | 63 |
| Generated suite: `not (monetary_caption or round_five or referenced_stamp_date)` | 31 | 264 |
| Generated suite: `referenced_stamp_date` | 32 | 263 |
| Numeric and structural-cleaner selectors listed below | 94 | 0 |

The three generated selections partition all 295 nodes in that file. The page
selection includes all 94 new page-value nodes and the 14 restored safety nodes.
The 469-node selection includes all 384 earlier mask/replay/typography nodes,
64 manual controls and 21 manual-prose nodes.

The author reported the original page tests before production edits: 28 failed and
12 passed across two disjoint selections. The exact review probe failed for
raw source with page and root citations in both numeric profiles; the property
failed in both profiles. Follow-up regressions exposed six structural-cleaner
failures, then 18 quote/NBSP failures and 18 interior Unicode-line failures on
intermediate fixes. The corresponding final passes are also author-reported.

The author reported that independent review identified the quote/Unicode prefix
and interior line-boundary gaps, which were fixed and re-reviewed, and that the
final code/test review reported no actionable findings. Those earlier review
outcomes are not independently reproducible without their omitted transcripts.
Two old exact-set assertions were strengthened by adding 93;
the restored safety file retains its recorded historical hash.

The reported execution environment used the repository's locked uv environment,
a GIL Python 3.13, one pytest process at a time, all five numerical thread caps
set to 4, and
`OMP_WAIT_POLICY=PASSIVE`. Each run used the following common flags, a distinct
temporary basetemp removed afterwards, and a nine-minute process bound:

```text
uv run --locked perl -e 'alarm 540; exec @ARGV or die $!;' python -m pytest \
  -q -o addopts= --no-cov -p no:cacheprovider --basetemp=<unique-temporary-directory> <selection>
```

The first two selections are these file lists, respectively:

```text
tests/test_source_completeness_manual_page_values.py
tests/test_source_completeness_manual_mask_review_safety.py
```

```text
tests/test_source_completeness_locator_masks.py
tests/test_source_completeness_manual_mask_operative_values.py
tests/test_source_completeness_manual_mask_review_headings.py
tests/test_source_completeness_manual_mask_review_stamps.py
tests/test_source_completeness_locator_replays.py
tests/test_source_completeness_us_manual_numeric_recall.py
tests/test_source_completeness_manual_typography.py
tests/test_source_completeness_manual_controls.py
tests/test_source_completeness_manual_prose.py
```

Each generated selection runs
`tests/test_source_completeness_manual_mask_fail_safe.py` with the table's
selection expression passed to `-k`. The numeric selection passes every node
selector below as a separate argument:

```text
tests/test_rulespec_validation.py::test_typed_numeric_occurrences_preserve_source_spans_and_rate_context
tests/test_rulespec_validation.py::test_typed_numeric_occurrences_preserve_exact_legacy_float_emissions
tests/test_rulespec_validation.py::test_typed_numeric_occurrences_map_repeated_cleaned_values_exactly
tests/test_rulespec_validation.py::test_typed_numeric_occurrences_carry_pipe_table_percentage_column_context
tests/test_rulespec_validation.py::test_typed_numeric_occurrences_align_a_leading_empty_cell_with_its_header
tests/test_rulespec_validation.py::test_numeric_extraction_handles_ghana_cedi_grouped_thousands
tests/test_rulespec_validation.py::test_numeric_extraction_reads_single_dot_group_as_decimal_too
tests/test_rulespec_validation.py::test_legacy_percentage_context_does_not_scale_currency_threshold
tests/test_rulespec_validation.py::test_legacy_percentage_context_keeps_real_scaled_rate
tests/test_rulespec_validation.py::test_numeric_profile_convention_matrix
tests/test_rulespec_validation.py::test_legacy_profile_keeps_uk_single_dot_dual_read
tests/test_rulespec_validation.py::test_rulespec_grounding_accepts_european_thousands_separator_money
tests/test_rulespec_validation.py::test_rulespec_grounding_accepts_dot_thousands_table_value_without_unit
tests/test_rulespec_validation.py::test_rulespec_grounding_accepts_spaced_thousands_separator_money
tests/test_rulespec_validation.py::test_rulespec_grounding_accepts_european_decimal_money
tests/test_rulespec_validation.py::test_rulespec_grounding_accepts_spaced_european_decimal_money
tests/test_rulespec_validation.py::test_rulespec_grounding_accepts_european_decimal_money_table_cell
tests/test_rulespec_validation.py::test_rulespec_grounding_accepts_european_decimal_coefficient
tests/test_rulespec_validation.py::test_a_bare_carriage_return_ends_a_line
tests/test_rulespec_validation.py::test_numeric_occurrence_extraction_ignores_parenthetical_subdivision_labels
tests/test_rulespec_validation.py::test_flattened_numeric_table_heading_keeps_values_not_row_keys
tests/test_rulespec_validation.py::test_standalone_table_heading_remains_structural
tests/test_rulespec_validation.py::test_flattened_size_rewrite_does_not_apply_to_prose_or_unlabeled_grid
tests/test_rulespec_validation.py::test_da_numeric_profile_supports_danish_percent_marker_vocabulary
tests/test_rulespec_validation.py::test_da_numeric_profile_parses_grouped_danish_currency
tests/test_rulespec_validation.py::test_da_structural_markers_match_grounding_only_convention
tests/test_rulespec_validation.py::test_da_dotted_section_cleanup_keeps_coefficient_on_its_own_source_span
tests/test_rulespec_validation.py::test_da_masked_structural_spans_are_hard_amount_context_boundaries
tests/test_rulespec_validation.py::test_da_equal_length_mask_maps_repeated_numeric_tokens_by_identity
tests/test_rulespec_validation.py::test_da_equal_length_heading_mask_cannot_transfer_rate_context
tests/test_rulespec_validation.py::test_da_structural_mask_edges_preserve_source_numeric_envelopes
tests/test_source_completeness.py::test_us_manual_code_and_policy_references_are_not_numeric_recall_values
tests/test_source_completeness.py::test_numbered_manual_section_headings_are_not_numeric_recall_values
tests/test_source_completeness.py::test_leading_values_that_are_not_section_headings_stay_in_numeric_recall
tests/test_source_completeness.py::test_us_locator_masks_do_not_apply_outside_us_citation_paths
tests/test_source_completeness.py::test_us_manual_locator_masks_do_not_apply_to_other_manuals
```

The author also reported these checks as passed; their historical logs are not
committed here:

- `uv run --locked ruff check pyproject.toml src/axiom_encode scripts tests`
- `uv run --locked ruff format --check` on the two changed source files, the
  changed operative-values test, and both restored/new test files: five files.
- `uv run --locked python -m compileall -q src/axiom_encode scripts`
- `git diff --check`

The full pytest/CI matrix, real-engine job and full raw-corpus replay were not
run. Original corpus bytes and the full row ledger remain unavailable in this
checkout; the historical bundle's totals are not a new corpus execution.
The semantic classifications of hash-authorized source roles whose complete
source bodies are absent are **not reproducible here**. Registry hashes and
recorded offsets can be inspected, and supplied fixtures can be rerun, but those
checks cannot independently establish the role of each omitted source body.
