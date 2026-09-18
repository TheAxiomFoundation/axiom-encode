# EncodeBench verifier board — EncodeBench verifier synthetic US v1

Suite `5b228af3d17c`, source `encodings_db`, mutator `1.0.1`, provision window 24000 chars, 180 pairs (360 cases).

Headline: per-kind detection AUC on each judge's kind channel, subject to a false-alarm ceiling of 25% on the judge's native verdict (flag rate on clean controls). Judges over the ceiling are shown but not ranked (†). AUC is pooled Mann-Whitney within a kind, ties count 0.5. `det@ceil` is the share of defective cases scoring above the control score that admits at most the ceiling's share of false alarms. Localization counts a finding that names the mutated rule or the edited token; probability-only judges score blank there by construction. Kinds marked ‡ have no kind-specific question for that judge and fall back to the verdict score; a mean AUC marked ‡ includes such kinds. Judges marked § could not be ranked (no scored controls or a kind with no AUC). Tokens, latency and cost cover every call, errors included; a blank cost means no published price or no reported usage, never zero.

| judge | model | native FAR | native det | AUC amount | AUC boundary | AUC conjunct | AUC polarity | AUC date/period | AUC entity | mean AUC | localize | coerced | median s | tokens in/out | cost/case |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev† | jev-1.13.0 | 72% | 95% | 0.998 | 0.924 | 0.753 | 0.979 | 0.853 | 0.942 | 0.908 | — | 0 | 0.19 | 3,157/144 | $0.00013 |
| sonnet† | claude-sonnet-4-5 | 38% | 76% | 0.983 | 0.800 | 0.650 | 0.800 | 0.622‡ | 0.570‡ | 0.738‡ | 68% | 0 | 5.24 | 3,312/319 | — |
| opus† | claude-opus-4-6 | 63% | 86% | 0.950 | 0.783 | 0.583 | 0.717 | 0.642‡ | 0.557‡ | 0.705‡ | 80% | 0 | 11.11 | 3,313/457 | $0.02800 |
| haiku† | claude-haiku-4-5-20251001 | 72% | 89% | 0.983 | 0.700 | 0.467 | 0.683 | 0.659‡ | 0.588‡ | 0.680‡ | 69% | 0 | 3.88 | 3,312/365 | $0.00514 |

## Per kind

| judge | kind | pairs | kind AUC | verdict AUC | paired rise | mean Δ | det@ceil | native det | native FAR | localize | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jev | amount | 30/30 | 0.998 | 0.976 | 97% | +0.785 | 100% | 100% | 63% | — | 0 |
| jev | boundary | 30/30 | 0.924 | 0.698 | 90% | +0.352 | 90% | 97% | 83% | — | 0 |
| jev | conjunct | 30/30 | 0.753 | 0.692 | 83% | +0.116 | 67% | 90% | 80% | — | 0 |
| jev | polarity | 30/30 | 0.979 | 0.926 | 100% | +0.480 | 97% | 100% | 70% | — | 0 |
| jev | date/period | 30/30 | 0.853 | 0.711 | 87% | +0.250 | 80% | 100% | 70% | — | 0 |
| jev | entity | 30/30 | 0.942 | 0.673 | 97% | +0.468 | 93% | 83% | 63% | — | 0 |
| sonnet | amount | 30/30 | 0.983 | 0.918 | 97% | +0.967 | 100% | 100% | 30% | 100% | 0 |
| sonnet | boundary | 30/30 | 0.800 | 0.702 | 60% | +0.600 | 80% | 83% | 43% | 77% | 0 |
| sonnet | conjunct | 30/30 | 0.650 | 0.687 | 33% | +0.300 | 0% | 67% | 37% | 63% | 0 |
| sonnet | polarity | 30/30 | 0.800 | 0.858 | 60% | +0.600 | 0% | 100% | 40% | 93% | 0 |
| sonnet | date/period‡ | 30/30 | 0.622 | 0.622 | 27% | +0.198 | 7% | 60% | 37% | 43% | 0 |
| sonnet | entity‡ | 30/30 | 0.570 | 0.570 | 27% | +0.034 | 7% | 47% | 43% | 30% | 0 |
| opus | amount | 30/30 | 0.950 | 0.998 | 90% | +0.900 | 100% | 100% | 73% | 100% | 0 |
| opus | boundary | 30/30 | 0.783 | 0.858 | 57% | +0.567 | 0% | 100% | 80% | 100% | 0 |
| opus | conjunct | 30/30 | 0.583 | 0.598 | 23% | +0.167 | 0% | 77% | 60% | 70% | 0 |
| opus | polarity | 30/30 | 0.717 | 0.859 | 43% | +0.433 | 0% | 100% | 57% | 100% | 0 |
| opus | date/period‡ | 30/30 | 0.642 | 0.642 | 43% | +0.152 | 47% | 67% | 50% | 60% | 0 |
| opus | entity‡ | 30/30 | 0.557 | 0.557 | 40% | +0.076 | 17% | 70% | 60% | 50% | 0 |
| haiku | amount | 30/30 | 0.983 | 0.846 | 97% | +0.967 | 97% | 100% | 67% | 97% | 0 |
| haiku | boundary | 30/30 | 0.700 | 0.510 | 40% | +0.400 | 0% | 93% | 100% | 77% | 0 |
| haiku | conjunct | 30/30 | 0.467 | 0.554 | 10% | -0.067 | 0% | 93% | 87% | 57% | 0 |
| haiku | polarity | 30/30 | 0.683 | 0.674 | 37% | +0.367 | 0% | 97% | 63% | 87% | 0 |
| haiku | date/period‡ | 30/30 | 0.659 | 0.659 | 47% | +0.181 | 53% | 80% | 60% | 57% | 0 |
| haiku | entity‡ | 30/30 | 0.588 | 0.588 | 33% | +0.119 | 0% | 70% | 57% | 40% | 0 |

## Spend and coverage

| judge | served model(s) | scored | errors | total cost | unpriced rows | price source |
|---|---|---|---|---|---|---|
| jev | jev-1.13.0 | 360/360 | 0 | $0.0477 | 0 | TypeSafe published price as recorded in the 2026-09-17 Jev judge pilot README (_axiom-runs/jev-judge-pilot-2026-09-17): $0.042 per million input tokens, output free |
| sonnet | claude-sonnet-4-5 | 360/360 | 0 | — | 360 | no published price recorded; cost blank |
| opus | claude-opus-4-6 | 360/360 | 0 | $10.0800 | 0 | claude-api skill, Current Models table (cached 2026-06-24): Claude Opus 4.6, $5.00 in / $25.00 out per 1M tokens |
| haiku | claude-haiku-4-5-20251001 | 360/360 | 0 | $1.8492 | 0 | claude-api skill, Current Models table (cached 2026-06-24): Claude Haiku 4.5, $1.00 in / $5.00 out per 1M tokens; claude-haiku-4-5-20251001 is the dated id of alias claude-haiku-4-5 per that skill's shared/models.md |
