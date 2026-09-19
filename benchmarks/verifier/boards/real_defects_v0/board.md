# EncodeBench verifier board — EncodeBench verifier real defects v0 (representatives, fidelity)

Suite `6e4959ee5baa`, source `real_defects`, provision window 24000 chars, 172 pairs (344 cases).

> Partial fold: incomplete runs were included with --allow-partial; rates cover only the cases each judge scored.

> Controls in this suite are post-fix artifacts that are not proven clean (control_clean = unverified). The native false-alarm rate is the flag rate on those unverified controls, and the false-alarm ceiling is not applied: no judge is unranked for flagging them.

> Defect kinds outside the synthetic taxonomy are scored on each judge's verdict channel: other:other, other:unrepresented_clause, other:untraceable_branch.

> Unrankable (no scored controls, or a kind with no AUC): sonnet-5. Shown last, without a rank.

Headline: per-kind detection AUC on each judge's kind channel, with the false-alarm ceiling not applied because the controls are not proven clean (see note). AUC is pooled Mann-Whitney within a kind, ties count 0.5. `det@ceil` is the share of defective cases scoring above the control score that admits at most the ceiling's share of false alarms. Localization counts a finding that names the mutated rule or the edited token; probability-only judges score blank there by construction. Kinds marked ‡ have no kind-specific question for that judge and fall back to the verdict score; a mean AUC marked ‡ includes such kinds. Judges marked § could not be ranked (no scored controls or a kind with no AUC). Tokens, latency and cost cover every call, errors included; a blank cost means no published price or no reported usage, never zero.

| judge | model | native FAR | native det | AUC amount | AUC boundary | AUC polarity | AUC date/period | AUC entity | AUC other:other | AUC other:unrepresented_clause | AUC other:untraceable_branch | mean AUC | localize | coerced | median s | tokens in/out | cost/case |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | jev-1.13.0 | 94% | 95% | 0.889 | 1.000 | 0.514 | 0.555 | 0.580 | 0.500‡ | 0.576‡ | 0.629‡ | 0.655‡ | — | 0 | 0.24 | 7,246/144 | $0.00030 |
| opus-5 | claude-opus-5 | 82% | 86% | 0.750 | 0.500 | 0.500 | 0.547‡ | 0.533‡ | 1.000‡ | 0.559‡ | 0.668‡ | 0.632‡ | 34% | 0 | 26.11 | 14,185/2,411 | $0.13120 |
| opus | claude-opus-4-6 | 88% | 91% | 0.500 | 1.000 | 0.500 | 0.476‡ | 0.508‡ | 0.500‡ | 0.598‡ | 0.607‡ | 0.586‡ | 24% | 0 | 18.93 | 10,956/753 | $0.07361 |
| sonnet | claude-sonnet-4-5 | 48% | 52% | 0.625 | 0.000 | 0.429 | 0.519‡ | 0.562‡ | 1.000‡ | 0.540‡ | 0.470‡ | 0.518‡ | 17% | 0 | 6.84 | 10,955/461 | $0.03977 |
| haiku | claude-haiku-4-5-20251001 | 87% | 89% | 0.250 | 0.500 | 0.429 | 0.535‡ | 0.468‡ | 0.000‡ | 0.586‡ | 0.526‡ | 0.412‡ | 22% | 0 | 5.69 | 10,955/671 | $0.01431 |
| sonnet-5§ | claude-sonnet-5 | 92% | 96% | 0.500 | 0.500 | 0.500 | 0.520‡ | 0.554‡ | — | 0.544‡ | 0.583‡ | — | 26% | 0 | 33.50 | 14,185/3,682 | $0.06520 |

## Per kind

| judge | kind | pairs | kind AUC | verdict AUC | paired rise | mean Δ | det@ceil | native det | native FAR | localize | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | amount | 3/4 | 0.889 | 0.722 | 100% | +0.257 | 67% | 100% | 100% | — | 2 |
| jev | boundary | 1/1 | 1.000 | 0.000 | 100% | +0.050 | 100% | 100% | 100% | — | 0 |
| jev | polarity | 6/7 | 0.514 | 0.347 | 50% | +0.008 | 17% | 100% | 100% | — | 2 |
| jev | date/period | 33/33 | 0.555 | 0.518 | 45% | +0.023 | 12% | 88% | 94% | — | 0 |
| jev | entity | 20/24 | 0.580 | 0.545 | 50% | +0.069 | 10% | 100% | 90% | — | 8 |
| jev | other:other‡ | 1/1 | 0.500 | 0.500 | 0% | +0.000 | 0% | 100% | 100% | — | 0 |
| jev | other:unrepresented_clause‡ | 59/64 | 0.576 | 0.576 | 59% | +0.038 | 2% | 98% | 97% | — | 8 |
| jev | other:untraceable_branch‡ | 31/38 | 0.629 | 0.629 | 55% | +0.063 | 19% | 91% | 90% | — | 13 |
| opus-5 | amount | 4/4 | 0.750 | 0.625 | 50% | +0.500 | 50% | 100% | 100% | 50% | 0 |
| opus-5 | boundary | 1/1 | 0.500 | 0.500 | 0% | +0.000 | 0% | 100% | 100% | 0% | 0 |
| opus-5 | polarity | 7/7 | 0.500 | 0.459 | 0% | +0.000 | 0% | 86% | 86% | 43% | 0 |
| opus-5 | date/period‡ | 33/33 | 0.547 | 0.547 | 42% | +0.018 | 9% | 70% | 70% | 33% | 0 |
| opus-5 | entity‡ | 24/24 | 0.533 | 0.533 | 38% | +0.030 | 4% | 96% | 92% | 42% | 0 |
| opus-5 | other:other‡ | 1/1 | 1.000 | 1.000 | 100% | +0.040 | 100% | 100% | 100% | 0% | 0 |
| opus-5 | other:unrepresented_clause‡ | 64/64 | 0.559 | 0.559 | 53% | +0.033 | 12% | 84% | 83% | 31% | 0 |
| opus-5 | other:untraceable_branch‡ | 38/38 | 0.668 | 0.668 | 63% | +0.097 | 18% | 95% | 82% | 34% | 0 |
| opus | amount | 4/4 | 0.500 | 0.438 | 25% | +0.000 | 0% | 100% | 100% | 50% | 0 |
| opus | boundary | 1/1 | 1.000 | 0.500 | 100% | +1.000 | 100% | 100% | 100% | 0% | 0 |
| opus | polarity | 7/7 | 0.500 | 0.500 | 0% | +0.000 | 0% | 100% | 100% | 14% | 0 |
| opus | date/period‡ | 33/33 | 0.476 | 0.476 | 27% | -0.043 | 3% | 73% | 79% | 18% | 0 |
| opus | entity‡ | 24/24 | 0.508 | 0.508 | 25% | +0.004 | 0% | 96% | 96% | 12% | 0 |
| opus | other:other‡ | 1/1 | 0.500 | 0.500 | 0% | +0.000 | 0% | 100% | 100% | 0% | 0 |
| opus | other:unrepresented_clause‡ | 64/64 | 0.598 | 0.598 | 34% | +0.062 | 12% | 94% | 88% | 33% | 0 |
| opus | other:untraceable_branch‡ | 38/38 | 0.607 | 0.607 | 42% | +0.087 | 11% | 97% | 87% | 24% | 0 |
| sonnet | amount | 4/4 | 0.625 | 0.438 | 25% | +0.250 | 25% | 25% | 50% | 25% | 0 |
| sonnet | boundary | 1/1 | 0.000 | 0.000 | 0% | -1.000 | 0% | 0% | 100% | 0% | 0 |
| sonnet | polarity | 7/7 | 0.429 | 0.439 | 0% | -0.143 | 0% | 14% | 29% | 14% | 0 |
| sonnet | date/period‡ | 33/33 | 0.519 | 0.519 | 18% | +0.002 | 6% | 36% | 36% | 9% | 0 |
| sonnet | entity‡ | 24/24 | 0.562 | 0.562 | 21% | +0.105 | 0% | 67% | 54% | 21% | 0 |
| sonnet | other:other‡ | 1/1 | 1.000 | 1.000 | 100% | +0.840 | 100% | 100% | 0% | 100% | 0 |
| sonnet | other:unrepresented_clause‡ | 64/64 | 0.540 | 0.540 | 27% | +0.079 | 12% | 64% | 55% | 23% | 0 |
| sonnet | other:untraceable_branch‡ | 38/38 | 0.470 | 0.470 | 16% | -0.026 | 3% | 45% | 47% | 8% | 0 |
| haiku | amount | 4/4 | 0.250 | 0.500 | 25% | -0.500 | 0% | 75% | 100% | 50% | 0 |
| haiku | boundary | 1/1 | 0.500 | 1.000 | 0% | +0.000 | 0% | 100% | 100% | 0% | 0 |
| haiku | polarity | 7/7 | 0.429 | 0.469 | 0% | -0.143 | 0% | 86% | 100% | 14% | 0 |
| haiku | date/period‡ | 33/33 | 0.535 | 0.535 | 33% | -0.035 | 0% | 91% | 97% | 18% | 0 |
| haiku | entity‡ | 24/24 | 0.468 | 0.468 | 17% | -0.075 | 0% | 83% | 92% | 17% | 0 |
| haiku | other:other‡ | 1/1 | 0.000 | 0.000 | 0% | -0.100 | 0% | 100% | 100% | 0% | 0 |
| haiku | other:unrepresented_clause‡ | 64/64 | 0.586 | 0.586 | 39% | +0.065 | 0% | 92% | 86% | 31% | 0 |
| haiku | other:untraceable_branch‡ | 38/38 | 0.526 | 0.526 | 29% | +0.100 | 0% | 87% | 74% | 11% | 0 |
| sonnet-5 | amount | 4/4 | 0.500 | 0.594 | 25% | +0.000 | 0% | 100% | 100% | 50% | 0 |
| sonnet-5 | boundary | 1/1 | 0.500 | 0.500 | 0% | +0.000 | 0% | 100% | 100% | 0% | 0 |
| sonnet-5 | polarity | 5/7 | 0.500 | 0.586 | 0% | +0.000 | 0% | 100% | 100% | 20% | 2 |
| sonnet-5 | date/period‡ | 33/33 | 0.520 | 0.520 | 39% | +0.038 | 12% | 94% | 85% | 27% | 0 |
| sonnet-5 | entity‡ | 21/24 | 0.554 | 0.554 | 33% | +0.015 | 10% | 95% | 96% | 19% | 4 |
| sonnet-5 | other:other | 0/1 | — | — | — | — | — | — | — | — | 2 |
| sonnet-5 | other:unrepresented_clause‡ | 58/64 | 0.544 | 0.544 | 48% | +0.032 | 8% | 95% | 92% | 23% | 7 |
| sonnet-5 | other:untraceable_branch‡ | 37/38 | 0.583 | 0.583 | 49% | +0.070 | 11% | 97% | 92% | 32% | 1 |

## Spend and coverage

| judge | served model(s) | scored | errors | total cost | unpriced rows | price source |
|---|---|---|---|---|---|---|
| jev | jev-1.13.0 | 311/344 | 33 | $0.0947 | 33 | TypeSafe published price as recorded in the 2026-09-17 Jev judge pilot README (_axiom-runs/jev-judge-pilot-2026-09-17): $0.042 per million input tokens, output free |
| opus-5 | claude-opus-5 | 344/344 | 0 | $45.1324 | 0 | Anthropic pricing page https://platform.claude.com/docs/en/about-claude/pricing, fetched 2026-09-19: Claude Opus 5, $5 / MTok base input, $25 / MTok output |
| opus | claude-opus-4-6 | 344/344 | 0 | $25.3202 | 0 | claude-api skill, Current Models table (cached 2026-06-24): Claude Opus 4.6, $5.00 in / $25.00 out per 1M tokens |
| sonnet | claude-sonnet-4-5 | 344/344 | 0 | $13.6818 | 0 | Anthropic pricing page https://platform.claude.com/docs/en/about-claude/pricing, fetched 2026-09-19: Claude Sonnet 4.5, $3 / MTok base input, $15 / MTok output |
| haiku | claude-haiku-4-5-20251001 | 344/344 | 0 | $4.9227 | 0 | claude-api skill, Current Models table (cached 2026-06-24): Claude Haiku 4.5, $1.00 in / $5.00 out per 1M tokens; claude-haiku-4-5-20251001 is the dated id of alias claude-haiku-4-5 per that skill's shared/models.md |
| sonnet-5 | claude-sonnet-5 | 328/344 | 16 | $22.4269 | 0 | Anthropic pricing page https://platform.claude.com/docs/en/about-claude/pricing, fetched 2026-09-19: Claude Sonnet 5, $2 / MTok base input, $10 / MTok output (the launch introductory price, now standard per the page's note) |
