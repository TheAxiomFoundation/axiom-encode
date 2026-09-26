# EncodeBench verifier board — EncodeBench verifier real defects v0 (representatives, fidelity, under 100k chars)

Suite `4bdd4eb214cb`, source `real_defects`, provision window 24000 chars, 154 pairs (308 cases).

> Partial fold: incomplete runs were included with --allow-partial; rates cover only the cases each judge scored.

> Controls in this suite are post-fix artifacts that are not proven clean (control_clean = unverified). The native false-alarm rate is the flag rate on those unverified controls, and the false-alarm ceiling is not applied: no judge is unranked for flagging them.

> Derived suite: filtered from parent 6e4959ee5baa ('EncodeBench verifier real defects v0 (representatives, fidelity)') by {'keep_citation_prefix': [], 'drop_citation_prefix': [], 'drop_pairs': [], 'max_case_chars': 100000, 'reason': None}; 18 pair(s) dropped.

> Defect kinds outside the synthetic taxonomy are scored on each judge's verdict channel: other:other, other:unrepresented_clause, other:untraceable_branch.

> Unrankable (no scored controls, or a kind with no AUC): sonnet-5. Shown last, without a rank.

Headline: per-kind detection AUC on each judge's kind channel, with the false-alarm ceiling not applied because the controls are not proven clean (see note). AUC is pooled Mann-Whitney within a kind, ties count 0.5. `det@ceil` is the share of defective cases scoring above the control score that admits at most the ceiling's share of false alarms. Localization counts a finding that names the mutated rule or the edited token; probability-only judges score blank there by construction. Kinds marked ‡ have no kind-specific question for that judge and fall back to the verdict score; a mean AUC marked ‡ includes such kinds. Judges marked § could not be ranked (no scored controls or a kind with no AUC). Tokens, latency and cost cover every call, errors included; a blank cost means no published price or no reported usage, never zero.

| judge | model | native FAR | native det | AUC amount | AUC boundary | AUC polarity | AUC date/period | AUC entity | AUC other:other | AUC other:unrepresented_clause | AUC other:untraceable_branch | mean AUC | localize | coerced | median s | tokens in/out | cost/case |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | jev-1.13.0 | 94% | 95% | 0.889 | 1.000 | 0.514 | 0.555 | 0.580 | 0.500‡ | 0.574‡ | 0.621‡ | 0.654‡ | — | 0 | 0.24 | 7,014/144 | $0.00030 |
| opus-5 | claude-opus-5 | 80% | 84% | 0.833 | 0.500 | 0.500 | 0.547‡ | 0.521‡ | 1.000‡ | 0.565‡ | 0.683‡ | 0.644‡ | 37% | 0 | 24.25 | 9,300/2,284 | $0.10361 |
| opus | claude-opus-4-6 | 87% | 90% | 0.500 | 1.000 | 0.500 | 0.476‡ | 0.500‡ | 0.500‡ | 0.593‡ | 0.646‡ | 0.589‡ | 27% | 0 | 17.79 | 7,124/725 | $0.05374 |
| sonnet | claude-sonnet-4-5 | 49% | 54% | 0.667 | 0.000 | 0.417 | 0.519‡ | 0.560‡ | 1.000‡ | 0.544‡ | 0.485‡ | 0.524‡ | 19% | 0 | 6.38 | 7,123/450 | $0.02812 |
| haiku | claude-haiku-4-5-20251001 | 90% | 92% | 0.333 | 0.500 | 0.500 | 0.535‡ | 0.477‡ | 0.000‡ | 0.575‡ | 0.514‡ | 0.429‡ | 23% | 0 | 5.49 | 7,123/641 | $0.01033 |
| sonnet-5§ | claude-sonnet-5 | 90% | 95% | 0.500 | 0.500 | 0.500 | 0.520‡ | 0.568‡ | — | 0.558‡ | 0.595‡ | — | 29% | 0 | 31.51 | 9,300/3,640 | $0.05500 |

## Per kind

| judge | kind | pairs | kind AUC | verdict AUC | paired rise | mean Δ | det@ceil | native det | native FAR | localize | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | amount | 3/3 | 0.889 | 0.722 | 100% | +0.257 | 67% | 100% | 100% | — | 0 |
| jev | boundary | 1/1 | 1.000 | 0.000 | 100% | +0.050 | 100% | 100% | 100% | — | 0 |
| jev | polarity | 6/6 | 0.514 | 0.347 | 50% | +0.008 | 17% | 100% | 100% | — | 0 |
| jev | date/period | 33/33 | 0.555 | 0.518 | 45% | +0.023 | 12% | 88% | 94% | — | 0 |
| jev | entity | 20/20 | 0.580 | 0.545 | 50% | +0.069 | 10% | 100% | 90% | — | 0 |
| jev | other:other‡ | 1/1 | 0.500 | 0.500 | 0% | +0.000 | 0% | 100% | 100% | — | 0 |
| jev | other:unrepresented_clause‡ | 59/59 | 0.574 | 0.574 | 59% | +0.038 | 2% | 98% | 97% | — | 0 |
| jev | other:untraceable_branch‡ | 31/31 | 0.621 | 0.621 | 55% | +0.063 | 19% | 90% | 90% | — | 0 |
| opus-5 | amount | 3/3 | 0.833 | 0.556 | 67% | +0.667 | 67% | 100% | 100% | 67% | 0 |
| opus-5 | boundary | 1/1 | 0.500 | 0.500 | 0% | +0.000 | 0% | 100% | 100% | 0% | 0 |
| opus-5 | polarity | 6/6 | 0.500 | 0.431 | 0% | +0.000 | 0% | 83% | 83% | 50% | 0 |
| opus-5 | date/period‡ | 33/33 | 0.547 | 0.547 | 42% | +0.018 | 9% | 70% | 70% | 33% | 0 |
| opus-5 | entity‡ | 20/20 | 0.521 | 0.521 | 35% | +0.029 | 5% | 95% | 90% | 50% | 0 |
| opus-5 | other:other‡ | 1/1 | 1.000 | 1.000 | 100% | +0.040 | 100% | 100% | 100% | 0% | 0 |
| opus-5 | other:unrepresented_clause‡ | 59/59 | 0.565 | 0.565 | 56% | +0.036 | 10% | 83% | 81% | 32% | 0 |
| opus-5 | other:untraceable_branch‡ | 31/31 | 0.683 | 0.683 | 65% | +0.113 | 19% | 94% | 77% | 39% | 0 |
| opus | amount | 3/3 | 0.500 | 0.444 | 33% | +0.000 | 0% | 100% | 100% | 67% | 0 |
| opus | boundary | 1/1 | 1.000 | 0.500 | 100% | +1.000 | 100% | 100% | 100% | 0% | 0 |
| opus | polarity | 6/6 | 0.500 | 0.500 | 0% | +0.000 | 0% | 100% | 100% | 17% | 0 |
| opus | date/period‡ | 33/33 | 0.476 | 0.476 | 27% | -0.043 | 3% | 73% | 79% | 18% | 0 |
| opus | entity‡ | 20/20 | 0.500 | 0.500 | 20% | -0.002 | 0% | 95% | 95% | 15% | 0 |
| opus | other:other‡ | 1/1 | 0.500 | 0.500 | 0% | +0.000 | 0% | 100% | 100% | 0% | 0 |
| opus | other:unrepresented_clause‡ | 59/59 | 0.593 | 0.593 | 36% | +0.055 | 14% | 93% | 88% | 36% | 0 |
| opus | other:untraceable_branch‡ | 31/31 | 0.646 | 0.646 | 52% | +0.115 | 13% | 97% | 84% | 29% | 0 |
| sonnet | amount | 3/3 | 0.667 | 0.556 | 33% | +0.333 | 33% | 33% | 33% | 33% | 0 |
| sonnet | boundary | 1/1 | 0.000 | 0.000 | 0% | -1.000 | 0% | 0% | 100% | 0% | 0 |
| sonnet | polarity | 6/6 | 0.417 | 0.431 | 0% | -0.167 | 0% | 17% | 33% | 17% | 0 |
| sonnet | date/period‡ | 33/33 | 0.519 | 0.519 | 18% | +0.002 | 6% | 36% | 36% | 9% | 0 |
| sonnet | entity‡ | 20/20 | 0.560 | 0.560 | 20% | +0.087 | 0% | 70% | 60% | 25% | 0 |
| sonnet | other:other‡ | 1/1 | 1.000 | 1.000 | 100% | +0.840 | 100% | 100% | 0% | 100% | 0 |
| sonnet | other:unrepresented_clause‡ | 59/59 | 0.544 | 0.544 | 29% | +0.085 | 12% | 64% | 54% | 25% | 0 |
| sonnet | other:untraceable_branch‡ | 31/31 | 0.485 | 0.485 | 16% | -0.002 | 3% | 52% | 52% | 10% | 0 |
| haiku | amount | 3/3 | 0.333 | 0.556 | 33% | -0.333 | 0% | 100% | 100% | 67% | 0 |
| haiku | boundary | 1/1 | 0.500 | 1.000 | 0% | +0.000 | 0% | 100% | 100% | 0% | 0 |
| haiku | polarity | 6/6 | 0.500 | 0.500 | 0% | +0.000 | 0% | 100% | 100% | 17% | 0 |
| haiku | date/period‡ | 33/33 | 0.535 | 0.535 | 33% | -0.035 | 0% | 91% | 97% | 18% | 0 |
| haiku | entity‡ | 20/20 | 0.477 | 0.477 | 15% | -0.049 | 0% | 90% | 95% | 15% | 0 |
| haiku | other:other‡ | 1/1 | 0.000 | 0.000 | 0% | -0.100 | 0% | 100% | 100% | 0% | 0 |
| haiku | other:unrepresented_clause‡ | 59/59 | 0.575 | 0.575 | 37% | +0.040 | 0% | 92% | 88% | 34% | 0 |
| haiku | other:untraceable_branch‡ | 31/31 | 0.514 | 0.514 | 23% | +0.074 | 0% | 90% | 81% | 10% | 0 |
| sonnet-5 | amount | 3/3 | 0.500 | 0.611 | 33% | +0.000 | 0% | 100% | 100% | 67% | 0 |
| sonnet-5 | boundary | 1/1 | 0.500 | 0.500 | 0% | +0.000 | 0% | 100% | 100% | 0% | 0 |
| sonnet-5 | polarity | 5/6 | 0.500 | 0.617 | 0% | +0.000 | 0% | 100% | 100% | 20% | 1 |
| sonnet-5 | date/period‡ | 33/33 | 0.520 | 0.520 | 39% | +0.038 | 12% | 94% | 85% | 27% | 0 |
| sonnet-5 | entity‡ | 17/20 | 0.568 | 0.568 | 41% | +0.018 | 0% | 94% | 95% | 24% | 4 |
| sonnet-5 | other:other | 0/1 | — | — | — | — | — | — | — | — | 2 |
| sonnet-5 | other:unrepresented_clause‡ | 53/59 | 0.558 | 0.558 | 51% | +0.039 | 9% | 95% | 91% | 25% | 7 |
| sonnet-5 | other:untraceable_branch‡ | 30/31 | 0.595 | 0.595 | 53% | +0.086 | 13% | 97% | 90% | 39% | 1 |

## Spend and coverage

| judge | served model(s) | scored | errors | total cost | unpriced rows | price source |
|---|---|---|---|---|---|---|
| jev | jev-1.13.0 | 308/308 | 0 | $0.0907 | 0 | TypeSafe published price as recorded in the 2026-09-17 Jev judge pilot README (_axiom-runs/jev-judge-pilot-2026-09-17): $0.042 per million input tokens, output free |
| opus-5 | claude-opus-5 | 308/308 | 0 | $31.9107 | 0 | Anthropic pricing page https://platform.claude.com/docs/en/about-claude/pricing, fetched 2026-09-19: Claude Opus 5, $5 / MTok base input, $25 / MTok output |
| opus | claude-opus-4-6 | 308/308 | 0 | $16.5515 | 0 | claude-api skill, Current Models table (cached 2026-06-24): Claude Opus 4.6, $5.00 in / $25.00 out per 1M tokens |
| sonnet | claude-sonnet-4-5 | 308/308 | 0 | $8.6606 | 0 | Anthropic pricing page https://platform.claude.com/docs/en/about-claude/pricing, fetched 2026-09-19: Claude Sonnet 4.5, $3 / MTok base input, $15 / MTok output |
| haiku | claude-haiku-4-5-20251001 | 308/308 | 0 | $3.1816 | 0 | claude-api skill, Current Models table (cached 2026-06-24): Claude Haiku 4.5, $1.00 in / $5.00 out per 1M tokens; claude-haiku-4-5-20251001 is the dated id of alias claude-haiku-4-5 per that skill's shared/models.md |
| sonnet-5 | claude-sonnet-5 | 293/308 | 15 | $16.9391 | 0 | Anthropic pricing page https://platform.claude.com/docs/en/about-claude/pricing, fetched 2026-09-19: Claude Sonnet 5, $2 / MTok base input, $10 / MTok output (the launch introductory price, now standard per the page's note) |
