# EncodeBench verifier board — EncodeBench verifier synthetic US v1

Suite `5b228af3d17c`, source `encodings_db`, mutator `1.0.1`, provision window 24000 chars, 180 pairs (360 cases).

Headline: per-kind detection AUC on each judge's kind channel, subject to a false-alarm ceiling of 10% on the judge's native verdict (flag rate on clean controls). Judges over the ceiling are shown but not ranked (†). AUC is pooled Mann-Whitney within a kind, ties count 0.5. `det@ceil` is the share of defective cases scoring above the control score that admits at most the ceiling's share of false alarms. Localization counts a finding that names the mutated rule or the edited token; probability-only judges score blank there by construction. Kinds marked ‡ have no kind-specific question for that judge and fall back to the verdict score; a mean AUC marked ‡ includes such kinds. Judges marked § could not be ranked (no scored controls or a kind with no AUC). Tokens, latency and cost cover every call, errors included; a blank cost means no published price or no reported usage, never zero.

| judge | model | native FAR | native det | AUC amount | AUC boundary | AUC conjunct | AUC polarity | AUC date/period | AUC entity | mean AUC | localize | coerced | median s | tokens in/out | cost/case |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev† | jev-1.13.0 | 72% | 95% | 0.998 | 0.924 | 0.753 | 0.979 | 0.853 | 0.942 | 0.908 | — | 0 | 0.19 | 3,157/144 | $0.00013 |
| opus-5† | claude-opus-5 | 51% | 89% | 0.983 | 0.983 | 0.667 | 0.767 | 0.696‡ | 0.614‡ | 0.785‡ | 86% | 0 | 14.13 | 4,335/1,248 | $0.05288 |
| sonnet† | claude-sonnet-4-5 | 38% | 76% | 0.983 | 0.800 | 0.650 | 0.800 | 0.622‡ | 0.570‡ | 0.738‡ | 68% | 0 | 5.24 | 3,312/319 | $0.01472 |
| sonnet-5† | claude-sonnet-5 | 65% | 86% | 1.000 | 0.950 | 0.567 | 0.617 | 0.588‡ | 0.524‡ | 0.708‡ | 73% | 0 | 20.71 | 4,335/2,423 | $0.03290 |
| opus† | claude-opus-4-6 | 63% | 86% | 0.950 | 0.783 | 0.583 | 0.717 | 0.642‡ | 0.557‡ | 0.705‡ | 80% | 0 | 11.11 | 3,313/457 | $0.02800 |
| haiku† | claude-haiku-4-5-20251001 | 72% | 89% | 0.983 | 0.700 | 0.467 | 0.683 | 0.659‡ | 0.588‡ | 0.680‡ | 69% | 0 | 3.88 | 3,312/365 | $0.00514 |

## Per kind

| judge | kind | pairs | kind AUC | verdict AUC | paired rise | mean Δ | det@ceil | native det | native FAR | localize | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | amount | 30/30 | 0.998 | 0.976 | 97% | +0.785 | 100% | 100% | 63% | — | 0 |
| jev | boundary | 30/30 | 0.924 | 0.698 | 90% | +0.352 | 73% | 97% | 83% | — | 0 |
| jev | conjunct | 30/30 | 0.753 | 0.692 | 83% | +0.116 | 50% | 90% | 80% | — | 0 |
| jev | polarity | 30/30 | 0.979 | 0.926 | 100% | +0.480 | 93% | 100% | 70% | — | 0 |
| jev | date/period | 30/30 | 0.853 | 0.711 | 87% | +0.250 | 63% | 100% | 70% | — | 0 |
| jev | entity | 30/30 | 0.942 | 0.673 | 97% | +0.468 | 80% | 83% | 63% | — | 0 |
| opus-5 | amount | 30/30 | 0.983 | 0.999 | 97% | +0.967 | 100% | 100% | 47% | 100% | 0 |
| opus-5 | boundary | 30/30 | 0.983 | 0.843 | 97% | +0.967 | 97% | 100% | 70% | 100% | 0 |
| opus-5 | conjunct | 30/30 | 0.667 | 0.767 | 33% | +0.333 | 0% | 87% | 53% | 83% | 0 |
| opus-5 | polarity | 30/30 | 0.767 | 0.936 | 53% | +0.533 | 0% | 100% | 47% | 100% | 0 |
| opus-5 | date/period‡ | 30/30 | 0.696 | 0.696 | 70% | +0.143 | 37% | 70% | 47% | 67% | 0 |
| opus-5 | entity‡ | 30/30 | 0.614 | 0.614 | 60% | +0.108 | 3% | 77% | 40% | 63% | 0 |
| sonnet | amount | 30/30 | 0.983 | 0.918 | 97% | +0.967 | 100% | 100% | 30% | 100% | 0 |
| sonnet | boundary | 30/30 | 0.800 | 0.702 | 60% | +0.600 | 0% | 83% | 43% | 77% | 0 |
| sonnet | conjunct | 30/30 | 0.650 | 0.687 | 33% | +0.300 | 0% | 67% | 37% | 63% | 0 |
| sonnet | polarity | 30/30 | 0.800 | 0.858 | 60% | +0.600 | 0% | 100% | 40% | 93% | 0 |
| sonnet | date/period‡ | 30/30 | 0.622 | 0.622 | 27% | +0.198 | 7% | 60% | 37% | 43% | 0 |
| sonnet | entity‡ | 30/30 | 0.570 | 0.570 | 27% | +0.034 | 7% | 47% | 43% | 30% | 0 |
| sonnet-5 | amount | 30/30 | 1.000 | 0.985 | 100% | +1.000 | 100% | 100% | 60% | 100% | 0 |
| sonnet-5 | boundary | 30/30 | 0.950 | 0.750 | 90% | +0.900 | 90% | 100% | 80% | 97% | 0 |
| sonnet-5 | conjunct | 30/30 | 0.567 | 0.602 | 17% | +0.133 | 0% | 77% | 60% | 53% | 0 |
| sonnet-5 | polarity | 30/30 | 0.617 | 0.873 | 27% | +0.233 | 0% | 97% | 73% | 90% | 0 |
| sonnet-5 | date/period‡ | 30/30 | 0.588 | 0.588 | 57% | +0.113 | 3% | 80% | 60% | 63% | 0 |
| sonnet-5 | entity‡ | 30/30 | 0.524 | 0.524 | 43% | +0.030 | 7% | 63% | 57% | 37% | 0 |
| opus | amount | 30/30 | 0.950 | 0.998 | 90% | +0.900 | 100% | 100% | 73% | 100% | 0 |
| opus | boundary | 30/30 | 0.783 | 0.858 | 57% | +0.567 | 0% | 100% | 80% | 100% | 0 |
| opus | conjunct | 30/30 | 0.583 | 0.598 | 23% | +0.167 | 0% | 77% | 60% | 70% | 0 |
| opus | polarity | 30/30 | 0.717 | 0.859 | 43% | +0.433 | 0% | 100% | 57% | 100% | 0 |
| opus | date/period‡ | 30/30 | 0.642 | 0.642 | 43% | +0.152 | 7% | 67% | 50% | 60% | 0 |
| opus | entity‡ | 30/30 | 0.557 | 0.557 | 40% | +0.076 | 3% | 70% | 60% | 50% | 0 |
| haiku | amount | 30/30 | 0.983 | 0.846 | 97% | +0.967 | 97% | 100% | 67% | 97% | 0 |
| haiku | boundary | 30/30 | 0.700 | 0.510 | 40% | +0.400 | 0% | 93% | 100% | 77% | 0 |
| haiku | conjunct | 30/30 | 0.467 | 0.554 | 10% | -0.067 | 0% | 93% | 87% | 57% | 0 |
| haiku | polarity | 30/30 | 0.683 | 0.674 | 37% | +0.367 | 0% | 97% | 63% | 87% | 0 |
| haiku | date/period‡ | 30/30 | 0.659 | 0.659 | 47% | +0.181 | 0% | 80% | 60% | 57% | 0 |
| haiku | entity‡ | 30/30 | 0.588 | 0.588 | 33% | +0.119 | 0% | 70% | 57% | 40% | 0 |

## Spend and coverage

| judge | served model(s) | scored | errors | total cost | unpriced rows | price source |
|---|---|---|---|---|---|---|
| jev | jev-1.13.0 | 360/360 | 0 | $0.0477 | 0 | TypeSafe published price as recorded in the 2026-09-17 Jev judge pilot README (_axiom-runs/jev-judge-pilot-2026-09-17): $0.042 per million input tokens, output free |
| opus-5 | claude-opus-5 | 360/360 | 0 | $19.0358 | 0 | Anthropic pricing page https://platform.claude.com/docs/en/about-claude/pricing, fetched 2026-09-19: Claude Opus 5, $5 / MTok base input, $25 / MTok output |
| sonnet | claude-sonnet-4-5 | 360/360 | 0 | $5.2980 | 0 | Anthropic pricing page https://platform.claude.com/docs/en/about-claude/pricing, fetched 2026-09-19: Claude Sonnet 4.5, $3 / MTok base input, $15 / MTok output |
| sonnet-5 | claude-sonnet-5 | 360/360 | 0 | $11.8437 | 0 | Anthropic pricing page https://platform.claude.com/docs/en/about-claude/pricing, fetched 2026-09-19: Claude Sonnet 5, $2 / MTok base input, $10 / MTok output (the launch introductory price, now standard per the page's note) |
| opus | claude-opus-4-6 | 360/360 | 0 | $10.0800 | 0 | claude-api skill, Current Models table (cached 2026-06-24): Claude Opus 4.6, $5.00 in / $25.00 out per 1M tokens |
| haiku | claude-haiku-4-5-20251001 | 360/360 | 0 | $1.8492 | 0 | claude-api skill, Current Models table (cached 2026-06-24): Claude Haiku 4.5, $1.00 in / $5.00 out per 1M tokens; claude-haiku-4-5-20251001 is the dated id of alias claude-haiku-4-5 per that skill's shared/models.md |
