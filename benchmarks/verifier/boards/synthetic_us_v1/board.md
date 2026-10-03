# EncodeBench verifier board: EncodeBench verifier synthetic US v1 (1.0.2 audit)

Suite `bc57111f0129`, source `encodings_db`, mutator `1.0.1`, provision window 24000 chars, 176 pairs (352 cases).

> Derived suite: filtered from parent 5b228af3d17c ('EncodeBench verifier synthetic US v1') by {'keep_citation_prefix': [], 'drop_citation_prefix': [], 'drop_pairs': ['date_or_period_wrong-58d498f1', 'date_or_period_wrong-66ceaa04', 'date_or_period_wrong-73274712', 'date_or_period_wrong-aa3de659'], 'max_case_chars': None, 'reason': "fail the mutator 1.0.2 period guard (audit-suite): the window states no daily period; 1.0.1 matched 'day' inside 'daylight'"}; 4 pair(s) dropped.

Headline: per-kind detection AUC on each judge's kind channel, subject to a false-alarm ceiling of 10% on the judge's native verdict (flag rate on clean controls). Judges over the ceiling are shown but not ranked (†). AUC is pooled Mann-Whitney within a kind, ties count 0.5. `det@ceil` is the share of defective cases scoring above the control score that admits at most the ceiling's share of false alarms. Localization counts a finding that names the mutated rule or the edited token; probability-only judges score blank there by construction. Kinds marked ‡ have no kind-specific question for that judge and fall back to the verdict score; a mean AUC marked ‡ includes such kinds. Judges marked § could not be ranked (no scored controls or a kind with no AUC). Tokens, latency and cost cover every row in the results, errors included; a retried error is counted once, as its retry. A blank cost means no published price or no reported usage, never zero.

| judge | model | native FAR | native det | AUC amount | AUC boundary | AUC conjunct | AUC polarity | AUC date/period | AUC entity | mean AUC | localize | coerced | median s | tokens in/out | cost/case |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev† | jev-1.13.0 | 71% | 95% | 0.998 | 0.924 | 0.753 | 0.979 | 0.868 | 0.942 | 0.911 | n/a | 0 | 0.19 | 3,148/144 | $0.00013 |
| opus-5† | claude-opus-5 | 52% | 91% | 0.983 | 0.983 | 0.667 | 0.767 | 0.724‡ | 0.614‡ | 0.790‡ | 86% | 0 | 14.41 | 4,328/1,256 | $0.05305 |
| sonnet† | claude-sonnet-4-5 | 39% | 78% | 0.983 | 0.800 | 0.650 | 0.800 | 0.642‡ | 0.570‡ | 0.741‡ | 69% | 0 | 5.55 | 3,306/325 | $0.01480 |
| sonnet-5† | claude-sonnet-5 | 66% | 88% | 1.000 | 0.950 | 0.567 | 0.617 | 0.598‡ | 0.524‡ | 0.709‡ | 75% | 0 | 20.31 | 4,328/2,402 | $0.03267 |
| opus† | claude-opus-4-6 | 65% | 88% | 0.950 | 0.783 | 0.583 | 0.717 | 0.660‡ | 0.557‡ | 0.708‡ | 82% | 0 | 11.26 | 3,307/467 | $0.02822 |
| haiku† | claude-haiku-4-5-20251001 | 72% | 89% | 0.983 | 0.700 | 0.467 | 0.683 | 0.681‡ | 0.588‡ | 0.684‡ | 69% | 0 | 3.88 | 3,306/366 | $0.00513 |

## Per kind

| judge | kind | pairs | kind AUC | verdict AUC | paired rise | mean Δ | det@ceil | native det | native FAR | localize | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | amount | 30/30 | 0.998 | 0.976 | 97% | +0.785 | 100% | 100% | 63% | n/a | 0 |
| jev | boundary | 30/30 | 0.924 | 0.698 | 90% | +0.352 | 73% | 97% | 83% | n/a | 0 |
| jev | conjunct | 30/30 | 0.753 | 0.692 | 83% | +0.116 | 50% | 90% | 80% | n/a | 0 |
| jev | polarity | 30/30 | 0.979 | 0.926 | 100% | +0.480 | 93% | 100% | 70% | n/a | 0 |
| jev | date/period | 26/26 | 0.868 | 0.734 | 88% | +0.278 | 73% | 100% | 65% | n/a | 0 |
| jev | entity | 30/30 | 0.942 | 0.673 | 97% | +0.468 | 80% | 83% | 63% | n/a | 0 |
| opus-5 | amount | 30/30 | 0.983 | 0.999 | 97% | +0.967 | 100% | 100% | 47% | 100% | 0 |
| opus-5 | boundary | 30/30 | 0.983 | 0.843 | 97% | +0.967 | 97% | 100% | 70% | 100% | 0 |
| opus-5 | conjunct | 30/30 | 0.667 | 0.767 | 33% | +0.333 | 0% | 87% | 53% | 83% | 0 |
| opus-5 | polarity | 30/30 | 0.767 | 0.936 | 53% | +0.533 | 0% | 100% | 47% | 100% | 0 |
| opus-5 | date/period‡ | 26/26 | 0.724 | 0.724 | 69% | +0.157 | 27% | 81% | 54% | 73% | 0 |
| opus-5 | entity‡ | 30/30 | 0.614 | 0.614 | 60% | +0.108 | 3% | 77% | 40% | 60% | 0 |
| sonnet | amount | 30/30 | 0.983 | 0.918 | 97% | +0.967 | 100% | 100% | 30% | 100% | 0 |
| sonnet | boundary | 30/30 | 0.800 | 0.702 | 60% | +0.600 | 0% | 83% | 43% | 77% | 0 |
| sonnet | conjunct | 30/30 | 0.650 | 0.687 | 33% | +0.300 | 0% | 67% | 37% | 63% | 0 |
| sonnet | polarity | 30/30 | 0.800 | 0.858 | 60% | +0.600 | 0% | 100% | 40% | 93% | 0 |
| sonnet | date/period‡ | 26/26 | 0.642 | 0.642 | 31% | +0.229 | 8% | 69% | 42% | 50% | 0 |
| sonnet | entity‡ | 30/30 | 0.570 | 0.570 | 27% | +0.034 | 7% | 47% | 43% | 27% | 0 |
| sonnet-5 | amount | 30/30 | 1.000 | 0.985 | 100% | +1.000 | 100% | 100% | 60% | 100% | 0 |
| sonnet-5 | boundary | 30/30 | 0.950 | 0.750 | 90% | +0.900 | 90% | 100% | 80% | 97% | 0 |
| sonnet-5 | conjunct | 30/30 | 0.567 | 0.602 | 17% | +0.133 | 0% | 77% | 60% | 53% | 0 |
| sonnet-5 | polarity | 30/30 | 0.617 | 0.873 | 27% | +0.233 | 0% | 97% | 73% | 90% | 0 |
| sonnet-5 | date/period‡ | 26/26 | 0.598 | 0.598 | 54% | +0.127 | 4% | 92% | 69% | 73% | 0 |
| sonnet-5 | entity‡ | 30/30 | 0.524 | 0.524 | 43% | +0.030 | 7% | 63% | 57% | 37% | 0 |
| opus | amount | 30/30 | 0.950 | 0.998 | 90% | +0.900 | 100% | 100% | 73% | 100% | 0 |
| opus | boundary | 30/30 | 0.783 | 0.858 | 57% | +0.567 | 0% | 100% | 80% | 100% | 0 |
| opus | conjunct | 30/30 | 0.583 | 0.598 | 23% | +0.167 | 0% | 77% | 60% | 70% | 0 |
| opus | polarity | 30/30 | 0.717 | 0.859 | 43% | +0.433 | 0% | 100% | 57% | 100% | 0 |
| opus | date/period‡ | 26/26 | 0.660 | 0.660 | 38% | +0.172 | 8% | 77% | 58% | 69% | 0 |
| opus | entity‡ | 30/30 | 0.557 | 0.557 | 40% | +0.076 | 3% | 70% | 60% | 50% | 0 |
| haiku | amount | 30/30 | 0.983 | 0.846 | 97% | +0.967 | 97% | 100% | 67% | 97% | 0 |
| haiku | boundary | 30/30 | 0.700 | 0.510 | 40% | +0.400 | 0% | 93% | 100% | 73% | 0 |
| haiku | conjunct | 30/30 | 0.467 | 0.554 | 10% | -0.067 | 0% | 93% | 87% | 57% | 0 |
| haiku | polarity | 30/30 | 0.683 | 0.674 | 37% | +0.367 | 0% | 97% | 63% | 87% | 0 |
| haiku | date/period‡ | 26/26 | 0.681 | 0.681 | 54% | +0.208 | 0% | 77% | 54% | 62% | 0 |
| haiku | entity‡ | 30/30 | 0.588 | 0.588 | 33% | +0.119 | 0% | 70% | 57% | 40% | 0 |

## Spend and coverage

| judge | served model(s) | scored | errors | total cost | unpriced rows | price source |
|---|---|---|---|---|---|---|
| jev | jev-1.13.0 | 352/352 | 0 | $0.0465 | 0 | TypeSafe published price as recorded in the 2026-09-17 Jev judge pilot README (_axiom-runs/jev-judge-pilot-2026-09-17): $0.042 per million input tokens, output free |
| opus-5 | claude-opus-5 | 352/352 | 0 | $18.6720 | 0 | Anthropic pricing page https://platform.claude.com/docs/en/about-claude/pricing, fetched 2026-09-19: Claude Opus 5, $5 / MTok base input, $25 / MTok output |
| sonnet | claude-sonnet-4-5 | 352/352 | 0 | $5.2089 | 0 | Anthropic pricing page https://platform.claude.com/docs/en/about-claude/pricing, fetched 2026-09-19: Claude Sonnet 4.5, $3 / MTok base input, $15 / MTok output |
| sonnet-5 | claude-sonnet-5 | 352/352 | 0 | $11.5009 | 0 | Anthropic pricing page https://platform.claude.com/docs/en/about-claude/pricing, fetched 2026-09-19: Claude Sonnet 5, $2 / MTok base input, $10 / MTok output (the launch introductory price, now standard per the page's note) |
| opus | claude-opus-4-6 | 352/352 | 0 | $9.9318 | 0 | claude-api skill, Current Models table (cached 2026-06-24): Claude Opus 4.6, $5.00 in / $25.00 out per 1M tokens |
| haiku | claude-haiku-4-5-20251001 | 352/352 | 0 | $1.8074 | 0 | claude-api skill, Current Models table (cached 2026-06-24): Claude Haiku 4.5, $1.00 in / $5.00 out per 1M tokens; claude-haiku-4-5-20251001 is the dated id of alias claude-haiku-4-5 per that skill's shared/models.md |
