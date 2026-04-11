"""Narrow, editable prompt surfaces for eval-suite tuning experiments.

This file intentionally isolates the highest-leverage wording that we want
optimization loops to touch without granting write access to the rest of the
pipeline. The AutoAgent pilot should edit this file, not the corpus repos,
promotion scripts, or Atlas integration.
"""

AUTOAGENT_PILOT_EDITABLE_FILES = (
    "src/autorac/harness/eval_prompt_surface.py",
)


def render_uk_legislation_guidance() -> str:
    """Return UK-specific eval prompt guidance."""
    return """
- For UK legislation, do not invent custom provision-level entities like `Provision`, `Section`, or `Regulation`, and do not invent periods like `Instant`.
- Prefer `Person` when the source states an amount or condition "in respect of" a child, qualifying young person, or other individual.
- Use `Family` only when the encoded quantity is explicitly aggregate at claimant or benefit-unit level.
- For UK rate leaves with one grounded monetary amount, encode the directly payable person-level or unit-level amount described by the text; do not collapse it into an unconditional family-level constant.
- If the operative subject of the slice is an individual payment or `that payment`, preserve that payment-scoped subject in the principal output name and formula; do not silently broaden the rule into an unconditional claimant-level weekly amount or boolean that erases the payment-level trigger.
- When the source slice genuinely operates on individual payments and the rule can be represented as per-payment rows, prefer `entity: Payment` over broadening the logic to `Person` or `Family`.
- When the source slice genuinely operates on individual non-cash vouchers or other item-specific things rather than payments, prefer `entity: Asset` over broadening the logic to `Person` or `Family`.
- For those payment-scoped leaves, use `status: entity_not_supported` only if even a `Payment`-scoped encoding would still fail to represent the statutory consequence faithfully.
- For those item-scoped voucher/thing leaves, use `status: entity_not_supported` only if even an `Asset`-scoped encoding would still fail to represent the statutory consequence faithfully.
- In `.rac.test` for `entity: Payment` outputs, provide per-payment rows under `tables:` and assert the entity output as a row-ordered YAML list rather than flattening payment facts into scalar `input:` values.
- In `.rac.test` for `entity: Asset` outputs, provide per-item rows under `tables:` and assert the entity output as a row-ordered YAML list rather than flattening asset facts into scalar `input:` values.
- In those `entity: Payment` tests, the `tables:` key should match the exact entity name `Payment:` rather than an inferred plural like `payments:`.
- Use `status: entity_not_supported` for those payment-scoped slices only as a last resort when the supported schema truly cannot represent the statutory consequence faithfully. Do not prefer that fallback when a narrow, payment-scoped approximation can still encode the branch's actual legal effect.
- For UK `dtype: Money` variables derived from sterling amounts, include `unit: GBP`.
- If the source states a sterling amount in pence, encode it in pounds sterling as a decimal with `unit: GBP`; for example, `10 pence` should become `0.10`, not `10`.
- For those pence-derived UK money scalars, emit the decimal literal directly; do not disguise it as arithmetic like `1 / 10` or `10 / 100`.
- If the source states that a monetary amount or threshold is payable `per week`, `per month`, or `per year`, prefer a money variable with matching `period:` cadence rather than a day-level money variable, unless the principal output is instead a day-level boolean rule that merely depends on that money threshold.
- For UK deemed-income or capital-band rules phrased like `£1 for each £500 in excess of £10,000`, treat `for each £500` as counting complete bands, not proportional fractions.
- For those `for each £N in excess` rules, derive the band count with `floor(excess / band_size)` or an equivalent complete-band helper; do not use straight division that yields fractional weekly income such as `750 / 500 = 1.5`.
- In `.rac.test` for those complete-band money rules, include a non-exact-multiple excess case like `£750` above threshold so the tests expose any accidental fractional division.
- For UK branch leaves like `(a)`, `(b)`, or `80A(2)(c)`, encode the branch identity in the output variable name. Do not reuse generic parent variable names like `child_benefit_weekly_rate`, `standard_minimum_guarantee`, or `benefit_cap` for a branch-specific leaf.
- If the target text includes a deepest nested branch token like `(i)`, `(ii)`, `(iii)`, `(a)`, or `(b)`, the principal output variable must encode that deepest token, e.g. `qualifying_young_person_4A_1_b_i`, not just the parent branch like `qualifying_young_person_4A_1_b`.
- For example, if the target source branch is `regulation-10-4-b`, a principal output like `assessed_amount_deemed_increase_10_4` is too generic; it must carry the branch token, e.g. `assessed_amount_deemed_increase_10_4_b`.
- For atomic conjunctive branch slices that read like a lone limb ending in `and` or `or`, encode the limb-specific fact or limb-satisfaction condition itself; do not pretend to encode the whole parent consequence or all sibling limbs.
- For those atomic conjunctive branch slices, prefer neutral factual names like `arrangements_contain_provision_for_date_on_which_increase_is_to_be_paid_10_4_b` or `..._10_4_b_satisfied`; avoid standalone normative names like `..._must_...` unless the source text itself uses `must`.
- For those atomic conjunctive branch slices, do not make the principal output a bare input stub with no formula. Expose the raw fact as a `*_fact` input if needed, and make the principal branch-specific output a derived `*_satisfied` or equivalent variable so `.rac.test` does not need to feed the asserted output back into `input:`.
- When `./source.txt` includes parent intro context like `Where X, ...`, distinguish mere placement context from binding lead-in conjuncts. Do not automatically turn every parent intro into a separate applicability gate, but do preserve lead-in conditions when the branch text grammatically depends on them or when the source slice still states them as part of the operative condition.
- If the copied source slice says `Where the Secretary of State is informed ... contains provision—` and the branch line then supplies one such provision, preserve both the `Secretary of State is informed` condition and the branch-specific provision; do not model only the final provision limb.
- If the copied source slice says `This paragraph applies where the period which ... is a period of the same length ...` and branch `(b)` then says `ends on the first increased payment date`, preserve both conjuncts; do not drop the same-length requirement just because it appears in the lead-in text before the branch marker.
- If the copied source slice says `where X, on Y` or similar, treat `X` and `Y` as a positive conjunction for this branch when the text means `where X, the relevant day is Y`. Do not rewrite that as material implication like `not X or Y`.
- For example, if the source slice says `where the benefit is paid in arrears, on the last day of the benefit week`, the branch-specific `_satisfied` output should only be true when the benefit is paid in arrears and the current day is the last day of the benefit week.
- For temporal branch leaves that specify the day on which a consequence occurs, preserve explicit trigger conditions from the parent clause when they state when the timing rule operates.
- When RAC lacks a native date-valued output for such a temporal leaf, model the statutory day as a fact-shaped boolean helper on `period: Day`, and derive the principal output from that helper plus any explicit trigger conditions stated in the source text.
- For clauses like `before the date the claim was made or treated as made or, if there is a subsequent supersession under section 10 ..., before the date of the supersession`, preserve a single legally operative reference day rather than a loose OR over multiple possible dates.
- In those claim-date clauses, if there is no supersession, model one canonical operative claim-date fact that covers the legally relevant `made or treated as made` date; do not encode separate `day_is_date_claim_was_made` and `day_is_date_claim_was_treated_as_made` facts and then combine them with `or`.
- In `.rac.test` for those claim-date clauses, include one no-supersession case for the operative claim date and one supersession case for the supersession date; do not rely on paired tests that separately assert `made` and `treated as made` as independent true branches of the same reference-day selector.
- Do not convert relative temporal phrases like `the day following`, `the next following day`, `the first day of the next benefit week`, or `beginning with the day following ...` into invented numeric offset or ordinal scalars like `1`, `first_*_ordinal`, or `*_offset = 1` unless the source text itself states that number explicitly.
- For comparative temporal thresholds like `less than one month apart`, `one month or more apart`, or `within one month`, prefer a direct fact-shaped boolean such as `last_two_payments_are_less_than_one_month_apart` over numeric helper scalars like `one_month_threshold = 1` or derived `*_in_months` measures.
- In those comparative month-apart phrases, the `one month` comparator is not a standalone numeric scalar that needs its own RAC definition for grounding. Satisfy grounding with the semantic boolean itself; do not invent `1`-valued threshold/count helpers just to mirror the word `one`.
- When one slice repeatedly refers to the same legal period, such as `the period in respect of which a payment is made` and later `that period is three months`, keep one canonical fact or classification for that single period instead of inventing parallel free inputs like `*_in_weeks` and `*_in_months` that can contradict each other.
- If a specific period category in the source text already implies a broader threshold in the same limb, such as `that period is three months` implying `exceeds a week`, do not require a second independent duration input just to re-prove the broader threshold.
- For those comparative month-apart phrases, keep the legal month comparison semantic. Do not split it into calendar-part scalars, `*_month = 1`, or numeric duration thresholds that will trigger decomposed-date CI failures.
- For clauses like `the claimant's regular pattern of work is such that he does not work the same hours every week`, preserve the full legally qualified predicate about the claimant's regular pattern of work. Do not shorten the branch to only `does not work the same hours every week`.
- For lead-ins phrased like `Subject to paragraphs (3), (4) and (4A), ... includes—`, do not convert the cited paragraphs into a blanket negating gate such as `not (paragraph_3_applies or paragraph_4_applies or paragraph_4A_applies)` unless the copied source slice itself expressly states that direct exclusion.
- In those `subject to ... includes—` leaves, the principal output should still encode the leaf-specific inclusion, amount, or classification from the branch text. Do not make a composite `subject_to_*_satisfied` or `paragraph_3_4_or_4A_applies` helper the main rule.
- If the copied slice does not state the specific modifying effect of the cited paragraphs, do not guess it from paragraph numbers alone.
- In that situation, expose a branch-specific fact gate showing whether the cited subject-to qualification permits this branch to count, and make the branch output depend on that gate rather than either ignoring the qualification or hard-coding `not (paragraph_3_applies or paragraph_4_applies or paragraph_4A_applies)`.
- When that `subject to` lead-in cites multiple paragraphs, do not collapse all cited qualifications into one opaque helper like `paragraphs_3_4_and_4a_permit_*`.
- Instead, preserve the cited structure with one paragraph-specific qualification input or import per cited paragraph, and combine those paragraph-level qualifications explicitly in the branch formula only if needed.
- When canonical imports for those cited paragraphs are available in the workspace, import them. If they are not available, paragraph-specific local inputs are acceptable for an isolated slice artifact so long as their names preserve the cited paragraph numbers and the branch-specific legal effect.
- For opening qualifiers like `Subject to regulation 17B(6)` or `subject to paragraph (4)`, treat the cited provision as a possible override or displacement of this slice, not as a freestanding positive permission gate.
- In those `Subject to ...` leaves, if no canonical import is available for the cited override, model a local override/displacement boolean and make the principal consequence depend on that override being false.
- Do not encode those `Subject to ...` qualifiers as helper names like `*_permits_*` or formulas that require the cited provision to be positively true before this slice operates.
- When one branch enumerates multiple distinct payments or legal categories in the same limb, do not collapse them into a single `x_or_y` principal output just because the text joins them with `and` or `or`.
- For example, if the source slice says `statutory sick pay and statutory maternity pay payable by the employer`, preserve those named payments distinctly rather than generating one combined `statutory_sick_pay_or_statutory_maternity_pay_*` output.
- For statutory payment descriptions that say a payment is `payable` under an Act or by an employer, model payability as the legal fact. Do not replace `payable` with `receives` or `received` unless the source text itself says receipt.
- If an expenses limb says the expenses are `incurred by the claimant`, preserve that claimant-incurred predicate explicitly. Do not collapse it into only an employer-made-payment fact.
- For carve-outs phrased like `except where paragraph (b) applies`, treat the carve-out as displacing this slice. When the carve-out condition is true, this slice should evaluate false or be otherwise inoperative; do not treat the slice as automatically satisfied just because the exception applies.
- For legislative carve-outs phrased like `Except where paragraph (2) and (4) apply`, do not assume the exception is displaced only when both cited paragraphs apply simultaneously.
- Unless the copied source slice itself makes the cited paragraphs jointly necessary, preserve each cited paragraph as its own displacing condition and treat the slice as inoperative when any cited paragraph applies.
- For exclusion-list leaves phrased like `all income is qualifying income except ... which is not to be treated as qualifying income`, do not collapse the principal output to an unconditional `true` or `false`. Encode either the excluded amount itself or a fact-sensitive classification that changes with the source-stated subject/input.
- For positive conditional leaves phrased like `£20 is disregarded if ...`, `X is payable if ...`, or `benefit shall be treated as paid ...`, make the principal output explicitly depend on the source-stated condition.
- For those positive conditional leaves, the inapplicable case should usually be `0` for `dtype: Money` or `false` for `dtype: Boolean`; do not use an unconditional amount or `else: true` unless `./source.txt` expressly states that inapplicable cases count as satisfied.
- For mandatory deeming clauses phrased like `the amount of that payment shall be treated as ...`, do not introduce a `*_fact` input for whether the statutory treatment holds. The slice itself establishes that treatment when its antecedent conditions are met.
- For those deeming clauses, if the antecedent is not met, encode this slice as inoperative or false; do not use vacuous `else: true`.
- For those amount-level deeming clauses, do not replace the amount-level legal effect with a `Person`/`Day` boolean stand-in just to stay within the schema.
- If the current ontology cannot faithfully tie the deeming effect to the same payment amount referenced in the source slice, prefer `status: entity_not_supported` over a pseudo-boolean approximation.
- For purpose-limited deeming clauses like `For the purposes of paragraph (9)(b), and for that purpose only, the amounts specified in paragraph (5) shall be treated as though they were earnings`, do not use `status: entity_not_supported` merely because the source points to paragraph-specific amounts.
- In those purpose-limited deeming clauses, encode the narrow legal treatment as a purpose-specific boolean classification or amount placeholder keyed to the cited paragraphs, such as paragraph-(5) amounts treated as earnings for paragraph-(9)(b) only, rather than pretending to compute the whole downstream paragraph.
- For one limb of a multi-limb determination introduced by `shall be determined—` and ending with `; or`, do not invent sibling outcomes for non-applicable cases with `else: 0`, `else: false`, or similar.
- In those `... shall be determined — (i) ... ; or` limbs, encode only this limb's own applicability and branch-specific determination; leave other cases to sibling limbs instead of forcing a complete fallback result in this file.
- In those determination limbs, if the source says the amount is determined `by reference to` some basis, keep a branch-specific money or rate output for that basis instead of collapsing the file to a pure applicability boolean.
- In those determination limbs, do not invent sibling outcomes for inapplicable cases with `else: 0`, `else: false`, or a guessed fallback amount.
- For those `by reference to` determination limbs, pair the branch-specific money or rate output with a separate applicability boolean when helpful, and have the branch-specific amount equal the referenced amount or average helper named for this limb's basis.
- In those `by reference to` determination limbs, it is acceptable for false-case tests to assert only the applicability boolean and omit assertions about the branch-specific money or rate output, rather than forcing this file to guess a sibling limb's result.
- When a `by reference to` determination limb qualifies its averaging basis with operative parenthetical text, preserve that qualification in the modeled basis itself rather than collapsing it into an unqualified average input.
- For example, if the source says the complete cycle average includes periods in which the claimant does no work but disregards other absences, reflect those inclusion/disregard rules in the branch-specific basis helper or its supporting facts; do not replace the qualified basis with a generic `average_weekly_income_*` input that loses the legal treatment of no-work periods and absences.
- For residual determination limbs phrased like `such other payments as may ... enable the claimant's average weekly income to be determined more accurately`, the money-valued basis only operates when both the residual case and the source-stated accuracy condition apply.
- In those `such other payments` residual limbs, do not leave the branch money output unconditionally equal to the input average. Bind it to the branch applicability, typically with `if *_applies: average else: 0`, so the artifact does not yield a legal branch amount when the source condition is false.
- For determination limbs that literally compute `the amount to be included` or another branch-specific included amount, do not leave that money output unconditional when the limb's own applicability can be false.
- In those amount-included determination limbs, make the branch-specific money output depend on the limb applicability, typically with an explicit `else: 0`, so the file does not emit a legally operative included amount when paragraph-level exceptions or a different payment period displace the limb.
- For analogous determination limbs whose branch-specific output is a `Count` or other non-Boolean basis selector, it is acceptable for that branch-specific output to remain unconditional while a separate `_applies` boolean carries the branch test.
- In those branch-specific count or basis-selector outputs, do not write an inline conditional without `else` just to avoid a sibling fallback. Either keep the output unconditional and rely on the `_applies` boolean, or use a fully valid RAC conditional with an explicit `else`.
- For those branch-specific count or basis-selector outputs, negative tests should usually assert only the `_applies` boolean and omit the count output, rather than asserting that the limb still legally operates with `2`, `4`, or another branch-specific count when the branch is false.
- Do not write negative tests for those count-selector limbs that expect the principal basis-count output to remain the active legal basis in cases where the source-stated branch condition is false.
- If `./source.txt` contains only editorially omitted or repealed text shown by ellipses or dotted placeholders and no operative rule content for the target slice, a top-level `status: deferred` fallback is acceptable.
- For those editorially omitted dotted-text slices, keep the embedded source/docstring showing the omission, emit no local rule blocks, and leave `.rac.test` empty rather than inventing a guessed legal effect.
- For pure cross-reference computational slices phrased like `the amounts specified in paragraph (10) shall be taken into account in accordance with paragraph (4) or, as the case may be, paragraph (10) of regulation 13 ...`, do not replace the cited computation with local boolean `*_route_is_satisfied` or `*_fact` placeholders.
- In those pure cross-reference computational slices, import the actual cited canonical rule if it is present in the workspace.
- If the cited canonical rule is absent and the local slice states `paragraph (4) or, as the case may be, paragraph (10)` or another explicit list of alternative cited treatments, do not emit a top-level `status: deferred` stub and do not collapse those cited alternatives into one generic treatment gate.
- In those delegated-computation slices, preserve the distinct cited alternatives with paragraph-specific imports or local facts/amounts for each cited external treatment, and derive the principal output from the legally relevant alternative instead of a single catch-all `*_treatment_applies` boolean.
- In those delegated-computation slices, do not invent an extra `no treatment applies` branch inside the parent antecedent when the source text says one of the cited alternatives governs `as the case may be`.
- In those delegated-computation slices, do not make the cited route-selection flags part of whether the paragraph itself applies. For example, in regulation 17(11), self-employed earnings trigger the paragraph; regulation 13 paragraph (4) or paragraph (10) chooses the accounting route, not whether paragraph 17(11) applies at all.
- In those `as the case may be` route-selection slices, do not encode the consequence as an unqualified `if paragraph_4_route: paragraph_4_amount else: paragraph_10_amount`. Paragraph (10) must be selected by a paragraph-(10) route fact/import or by a derived paragraph-(10) route helper before its amount is used.
- Because `paragraph (4) or, as the case may be, paragraph (10)` states exhaustive alternatives, prefer a single mutually exclusive route selector or derive one route as the negation of the other inside the applicable paragraph. Do not expose two independent route booleans that allow both routes or neither route to be selected.
- For regulation 17(11), a safe local-placeholder shape is: paragraph applies when the earnings are self-employed; `regulation_13_paragraph_4_route_selected` is the binary route input; paragraph-(10) route is derived as the applicable paragraph with not paragraph-(4) route; the amount uses paragraph-(4) when that route is selected and paragraph-(10) otherwise within the applicable paragraph.
- Do not create an invalid-route output branch that returns `0` when paragraph 17(11) applies but local route flags are neither/both selected. Make invalid routing unrepresentable in the local model instead of converting it into a statutory zero amount.
- In `.rac.test` for those delegated-computation slices, include separate cases for the distinct cited alternatives, such as one regulation-13 paragraph-(4) path and one paragraph-(10) path, rather than only a generic applies/does-not-apply toggle.
- In `.rac.test` for those route-selection slices, avoid a false case that makes a self-employed-earner branch fail merely because neither local route flag was selected; that is an impossible or unresolved routing state, not a source-stated inapplicable case.
- For residual sibling limbs phrased like `in any other case`, preserve that residual condition explicitly. Do not treat the shared parent triggers alone as sufficient for this limb.
- In those `in any other case` limbs, model a local residual-case fact or applicability helper showing that no more specific sibling case applies, and make this limb depend on it.
- In `.rac.test` for those residual limbs, include a case where the parent conditions hold but the residual `other case` condition is false, so the branch remains inoperative.
- In those determination limbs, do not reuse the parent provision's generic final-amount phrase as the principal money or rate output name when this file encodes only one basis for the determination.
- Instead, name the principal money or rate output after this limb's own basis or method, such as determination by reference to average weekly income over a complete cycle, so the file does not claim to encode the whole parent calculation outside this limb.
- For branch-specific `_satisfied` outputs tied to one sibling limb, if the branch-triggering condition itself is false, the branch-specific output should usually be `false`, not vacuously `true`; sibling limbs or parent rules handle the other cases.
- For statutory phrases like `the claimant or, if he has a partner, his partner` or more generally `X or, if Y, Z`, treat the text as a genuine disjunction over possible satisfiers. The `if ...` clause limits the second disjunct; it does not switch the rule into partner-only substitution when `Y` is true.
- For those claimant-or-partner disjunctions, preserve separate claimant and partner fact inputs and derive the branch as `claimant_fact or (claimant_has_partner and partner_fact)` or the equivalent grounded boolean formula. Do not encode it as `if claimant_has_partner: partner_fact else: claimant_fact`.
- For disjunctive payment-description leaves phrased like `any X or Y, where condition A ... or, in respect of X, where condition B ...`, preserve the scope of the first qualifier across every antecedent payment type it grammatically modifies.
- In those disjunctive payment-description leaves, do not narrow the first `where ...` clause to only the later-mentioned category just because a later `or, in respect of X, where ...` clause adds an additional path for one subcategory.
- For example, if the source text says `any retired pay, pension or allowance granted in respect of disablement or any pension or allowance granted to a widow ... where such payment does not fall within paragraph (a) ... or, in respect of any retired pay or pension granted in respect of disablement, where such payment does not fall within paragraph (b) ...`, preserve the paragraph-(a) path for retired pay and pension as well as the widow/allowance categories, and treat the paragraph-(b) path as an additional route for retired pay or pension.
- For payment-description leaves phrased like `royalties or other sums received as a consideration for the use of, or the right to use, ...`, preserve the consideration-for-use/right-to-use qualifier across both `royalties` and `other sums`; do not model `royalty` as a free-standing qualifying limb divorced from the IP-use consideration.
- In those `royalties or other sums received as a consideration ...` leaves, a bare `payment_is_royalty` fact is too broad unless it is itself qualified to mean a royalty received as consideration for the stated use/right-to-use of the listed intellectual-property rights.
- For definitional clauses phrased like `a person who is gainfully employed in Great Britain either under a contract of service, or in an office ...`, preserve the shared qualifying employment/office across each alternative limb.
- In those `a person who is X either under A, or in B` definitions, do not decompose the rule into one free-standing `person_is_X` fact plus separate `under_A` and `in_B` facts that could be satisfied by different roles on the same day.
- Instead, distribute the shared qualifier across the alternatives with branch-specific combined facts such as `person_is_X_under_A` and `person_is_X_in_B`, then define the principal output as the disjunction of those combined alternatives.
- In `.rac.test`, use helper/input names that expose the actual legal facts from the source text. Prefer names like `child_benefit_is_only_person`, `child_benefit_is_elder_or_eldest_person`, `claimant_has_partner`, `is_single_claimant`, `is_joint_claimant`, `resident_in_greater_london`, or `responsible_for_child_or_qualifying_young_person`.
- In `.rac.test`, avoid opaque placeholders like `*_condition`, `*_eligibility_flag`, or `family_has_partner` when a more direct legal-fact name is available from the source text.
- In `.rac.test` for those claimant-or-partner disjunctions, include a partnered case where only the claimant satisfies the source-stated condition; that case should still evaluate true.
- In `.rac.test`, do not feed the same legal period through contradictory units or categories, such as asserting that one payment period is both `3 months` and `3 weeks` via separate free inputs.
- In `.rac.test` for one determination limb of a larger multi-limb calculation, when the limb does not apply, assert the limb-specific applicability boolean and omit assertions that would force the principal money or rate output to equal a sibling limb's unknown result.
- For whole-provision slices phrased like `Where X, Y must ...`, encode the triggered requirement itself. Include a `.rac.test` case where `X` is false; unless `./source.txt` expressly says otherwise, the requirement should evaluate as satisfied or inapplicable rather than false in that case.
- When a boolean output ultimately depends on a final fact-shaped input and the source text makes a meaningful false case possible, include a `.rac.test` case where the applicability conditions hold but that final fact input is false.
- In `.rac.test`, choose periods on or after the explicit effective date in `./source.txt`.
- Do not add speculative future-period tests that would rely on uprating, later amendments, or rates not stated in `./source.txt`.
- Reference RAC variables by bare name inside formulas. Do not write function-style calls like `some_variable(person, period)`.
"""


def render_single_amount_row_guidance() -> str:
    """Return prompt guidance for one-row fixed-amount source slices."""
    return """
- `./source.txt` is already a single table row or atomic branch with one grounded amount. Encode that branch-specific amount directly.
- Use a descriptive legal variable name, not a path- or source-id-derived placeholder like `uksi_2013_...`.
- For a one-row fixed-amount slice, do not invent a fresh `*_applies` helper or unrelated eligibility booleans unless the source text itself states them.
- For a one-row fixed-amount slice, do not invent alternate zero-amount tests.
- For a one-row fixed-amount slice, Do not emit `otherwise:`.
- For a one-row fixed-amount slice, Do not emit `before YYYY-MM-DD: 0`.
- For a one-row fixed-amount slice, Do not emit stray blocks like `from 0:`.
- For a one-row fixed-amount slice, use exactly one grounded `from YYYY-MM-DD:` clause unless `./source.txt` itself states multiple grounded dates or amounts.
- For a one-row fixed-amount slice, the principal amount variable should usually be a grounded constant under `from YYYY-MM-DD:`; do not wrap it in a conditional formula unless `./source.txt` itself states a second grounded amount or branch inside the same row.
- For a one-row fixed-amount slice, do not disguise the grounded amount as arithmetic like `2025 * 11 - 255`; emit the grounded constant directly.
- In `.rac.test` for a one-row fixed-amount slice, use boolean or fact-shaped helper inputs that mirror the row text.
- Do not invent sample ages like `2`, `3`, `24`, or `25` just to witness a row condition; if the row says "aged under 25", prefer a helper like `claimant_aged_under_25`.
- For a one-row fixed-amount slice with a single canonical subject, keep `.rac.test` outputs scalar instead of nested wrappers like `{person: 1, value: ...}`.
- For a one-row fixed-amount slice, every `.rac.test` case should keep the row-defining conditions satisfied; do not negate them in alternate tests unless `./source.txt` states another grounded amount for that alternate branch.
- For a one-row fixed-amount slice with `period: Year`, a base case is sufficient; do not synthesize an `effective_date_boundary` test.
- For a one-row fixed-amount slice with non-annual periods, the allowed `.rac.test` shapes are base case and effective-date boundary.
- Add a later same-amount case only when `./source.txt` explicitly says the amount remains unchanged through that later date.
- Do not include `alternate_branch_*` tests unless `./source.txt` states a second grounded amount.
- Do not use thousands separators in RAC numeric literals or `.rac.test` outputs; write `2500`, not `2,500`.
"""


def render_date_silent_scaffold_guidance() -> str:
    """Return fallback guidance for date-silent source slices."""
    return """
- If `./source.txt` is silent on effective dates and no copied precedent file provides a scaffold date, you may use `from 0001-01-01:` as a neutral scaffold date solely to satisfy RAC syntax.
- Treat that `0001-01-01` scaffold as a harness-only fallback, not as a claim about the true legal effective date.
- Numeric grounding rules do not apply to the `0001-01-01` tokens inside that neutral scaffold date.
"""
