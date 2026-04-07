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
- For UK `dtype: Money` variables derived from sterling amounts, include `unit: GBP`.
- If the source states a sterling amount in pence, encode it in pounds sterling as a decimal with `unit: GBP`; for example, `10 pence` should become `0.10`, not `10`.
- For those pence-derived UK money scalars, emit the decimal literal directly; do not disguise it as arithmetic like `1 / 10` or `10 / 100`.
- If the source states that a monetary amount or threshold is payable `per week`, `per month`, or `per year`, prefer a money variable with matching `period:` cadence rather than a day-level money variable, unless the principal output is instead a day-level boolean rule that merely depends on that money threshold.
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
- Do not convert relative temporal phrases like `the day following`, `the next following day`, `the first day of the next benefit week`, or `beginning with the day following ...` into invented numeric offset or ordinal scalars like `1`, `first_*_ordinal`, or `*_offset = 1` unless the source text itself states that number explicitly.
- For carve-outs phrased like `except where paragraph (b) applies`, treat the carve-out as displacing this slice. When the carve-out condition is true, this slice should evaluate false or be otherwise inoperative; do not treat the slice as automatically satisfied just because the exception applies.
- For exclusion-list leaves phrased like `all income is qualifying income except ... which is not to be treated as qualifying income`, do not collapse the principal output to an unconditional `true` or `false`. Encode either the excluded amount itself or a fact-sensitive classification that changes with the source-stated subject/input.
- For positive conditional leaves phrased like `£20 is disregarded if ...`, `X is payable if ...`, or `benefit shall be treated as paid ...`, make the principal output explicitly depend on the source-stated condition.
- For those positive conditional leaves, the inapplicable case should usually be `0` for `dtype: Money` or `false` for `dtype: Boolean`; do not use an unconditional amount or `else: true` unless `./source.txt` expressly states that inapplicable cases count as satisfied.
- For branch-specific `_satisfied` outputs tied to one sibling limb, if the branch-triggering condition itself is false, the branch-specific output should usually be `false`, not vacuously `true`; sibling limbs or parent rules handle the other cases.
- For statutory phrases like `the claimant or, if he has a partner, his partner` or more generally `X or, if Y, Z`, treat the text as a genuine disjunction over possible satisfiers. The `if ...` clause limits the second disjunct; it does not switch the rule into partner-only substitution when `Y` is true.
- For those claimant-or-partner disjunctions, preserve separate claimant and partner fact inputs and derive the branch as `claimant_fact or (claimant_has_partner and partner_fact)` or the equivalent grounded boolean formula. Do not encode it as `if claimant_has_partner: partner_fact else: claimant_fact`.
- For disjunctive payment-description leaves phrased like `any X or Y, where condition A ... or, in respect of X, where condition B ...`, preserve the scope of the first qualifier across every antecedent payment type it grammatically modifies.
- In those disjunctive payment-description leaves, do not narrow the first `where ...` clause to only the later-mentioned category just because a later `or, in respect of X, where ...` clause adds an additional path for one subcategory.
- For example, if the source text says `any retired pay, pension or allowance granted in respect of disablement or any pension or allowance granted to a widow ... where such payment does not fall within paragraph (a) ... or, in respect of any retired pay or pension granted in respect of disablement, where such payment does not fall within paragraph (b) ...`, preserve the paragraph-(a) path for retired pay and pension as well as the widow/allowance categories, and treat the paragraph-(b) path as an additional route for retired pay or pension.
- In `.rac.test`, use helper/input names that expose the actual legal facts from the source text. Prefer names like `child_benefit_is_only_person`, `child_benefit_is_elder_or_eldest_person`, `claimant_has_partner`, `is_single_claimant`, `is_joint_claimant`, `resident_in_greater_london`, or `responsible_for_child_or_qualifying_young_person`.
- In `.rac.test`, avoid opaque placeholders like `*_condition`, `*_eligibility_flag`, or `family_has_partner` when a more direct legal-fact name is available from the source text.
- In `.rac.test` for those claimant-or-partner disjunctions, include a partnered case where only the claimant satisfies the source-stated condition; that case should still evaluate true.
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
