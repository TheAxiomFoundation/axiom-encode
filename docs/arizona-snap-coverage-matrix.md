# Arizona SNAP coverage audit (working matrix)

This is an evidence ledger, not a parity claim. It distinguishes sources cited by
PolicyEngine from additional controlling Arizona policy, and distinguishes an
encoded RuleSpec file from tested household behavior. The pinned comparison
points for this first pass are PolicyEngine US `d89439134c1bac8add0c8261c8a075c33c39401a`,
RuleSpec US `9f38330fb43ffc693c6295b9a17c8d5d96520ad2`, and Axiom Corpus
`8f7d60aaced28ee4252b9237f9d6e02360dc34bc`. Re-audit these rows when any
pin or legal effective date changes. `not_comparable` oracle output is **not**
behavioral parity.

The repository's `oracle-coverage --program snap --json` classifier, run against
the pinned RuleSpec US commit, finds 103 Arizona executable outputs: five have
exact PolicyEngine registry mappings and companion tests, while 98 are
`known_not_comparable`. The five mappings cover the ECE FPL rate, three medical
deduction outputs, and the shelter-deduction income-share rate. A companion
test is **not** a matched PolicyEngine household comparison; these counts
establish classification and local test presence only. In particular, the
mapped ECE rate still has the documented 2026-03 temporal discrepancy.

The first-pass Arizona Populace bridge projected PolicyEngine's
`snap_net_income`, `is_snap_eligible`, `snap_max_allotment`, `snap_min_allotment`,
and `snap_excess_shelter_expense_deduction` into RuleSpec inputs. The merged
[axiom-oracles #597](https://github.com/TheAxiomFoundation/axiom-oracles/pull/597),
pinned by [axiom-encode #1772](https://github.com/TheAxiomFoundation/axiom-encode/pull/1772),
removed the PolicyEngine-fed net-income and excess-shelter inputs and projects
the underlying Arizona utility facts instead. The current bridge still feeds
PolicyEngine's `is_snap_eligible`, `snap_max_allotment`, and
`snap_min_allotment` to the Arizona benefit path (see
`axiom_oracles/bridges/snap_populace.py`,
`project_jurisdiction_household_inputs`). Allotment and eligibility comparison
still cannot establish independent end-to-end parity. Remove each remaining
oracle-derived input only after the corresponding source-grounded RuleSpec path
is composed, then run matched households with the same member facts and compare
intermediate as well as final outputs.

One **partial financial-path comparison**, not household parity: at the pinned
PolicyEngine commit, a one-person Arizona household in January 2026 with $1,003
monthly earnings, $500 monthly rent, and separately paid heating/cooling gives
SUA $323, SNAP net income $67, and SNAP allotment $277. The pinned
PolicyEngine situation sets `state_name` to `AZ` for 2026, member age to 30,
annual earnings to $12,036, annual rent to $6,000, heating/cooling expense to
true, and weekly work hours to 25; the work-hours fact makes the member
eligible under PolicyEngine's work rules. The pinned RuleSpec
`one_person_sua_household_rides_the_whole_dollar_chain` case gives SUA $323,
net income $68, and benefit $277. PolicyEngine keeps a $200.60 earned-income
deduction and $526.30 excess-shelter deduction before its net-income rounding;
the RuleSpec case expects integer $200 and $526 at those steps. The existing
RuleSpec case also sets age 60 while explicitly marking the member as not
elderly/disabled, so it is not a fully matched household record. Reconcile
rounding and member facts against controlling authority before claiming parity.
Re-running the pinned PolicyEngine situation at ages 30 and 60, holding every
other stated fact fixed, yields the same $323 allowance, $200.60 earned-income
deduction, $526.30 excess-shelter deduction, $67 net income, and $277 benefit
at both ages. The age mismatch therefore does not explain this particular
one-dollar net-income difference, though it still prevents claiming a matched
RuleSpec household comparison.
The pinned federal RuleSpec uses `floor()` for the earned-income deduction,
while [7 CFR 273.10(e)(1)(ii)](https://www.govinfo.gov/content/pkg/CFR-2025-title7-vol4/pdf/CFR-2025-title7-vol4-part273.pdf)
permits a nearest-dollar or state-TANF rounding procedure. A
[2025-12-19 archive capture of DES's FY 2026 benefit example](https://web.archive.org/web/20251219133658/https://des.az.gov/node/4875)
shows $1,739.33 gross earnings, a $347.87 earned-income deduction, and $386.46
net income, retaining cents through those steps. The Memento response preserves
the DES origin and 2025-12-19 last-modified header; the decompressed HTML has
SHA-256 `0a6b5ca9d952a3d2b9a7b748d16a3927f4ba6ff0b454fe0c1ef4f9779342e896`.
The [live DES FAQ](https://des.az.gov/node/4875) now displays FFY 2027 values
and again keeps cents through the earned-income deduction and net income:
$1,739.33 gross earnings yield a $347.87 deduction and $354.46 net income.
The contemporaneous FY 2026 example, corroborated by the later one, contradicts
assuming whole-dollar intermediate values for every FY 2026 case, but it is not the
controlling intermediate rounding procedure. [DES FAA6's current Thrifty Food
Plan table](https://dbmefaapolicy.azdes.gov/FAA6/Thrifty_Food_Plan_(NA).html)
instructs manual calculation to round 30% of net income **up** to the next
whole dollar before subtracting it from the maximum allotment. The official
[archived FY 2026 table](https://dbmefaapolicy.azdes.gov/Archived_Policy/baggage/FAA6.J08_ThriftyFoodPlanNA_2026.1.pdf)
explicitly covers 10/01/2025–09/30/2026 and gives the same instruction; the
[archived FY 2025 table](https://dbmefaapolicy.azdes.gov/Archived_Policy/baggage/FAA6.J08_ThriftyFoodPlanNA_2025.pdf)
also agrees. For an integer maximum, that yields the same final
allotment as flooring the maximum minus the unrounded 30% contribution. It
does not establish when Arizona rounds earned-income or shelter deductions or
net income. Locate the applicable DES intermediate calculation rule or approved
state option, then resolve the RuleSpec/PolicyEngine difference through
source-bound protected generation and matched household tests.

The 922/922 FY 2024 matches reported by the
[az-snap-qc suite](https://github.com/TheAxiomFoundation/axiom-oracles/blob/main/comparisons/az-snap-qc.yaml)
do not settle this
rounding election or establish FY 2026 parity. That comparison rewrites FY 2026
RuleSpec module IDs to FY 2024 sources through a compile-time overlay and runs
at a nominal January 2026 period. The [FY 2024 SNAP QC technical
documentation](https://snapqcdata.net/sites/default/files/2026-08/FY-2024-Tech-Doc.pdf)
describes edited, constructed public-use values: its consistency-edit narrative
says the earned-income deduction is rounded down, while its `FSERNDED` codebook
says the constructed deduction is rounded to the nearest integer. Neither is
Arizona's election under 7 CFR 273.10(e)(1)(ii), and a transformed-QC match
cannot substitute for source-grounded, matched-household behavior comparisons.

An **ECE temporal oracle target**, not a RuleSpec parity result: at the pinned
PolicyEngine commit, one Arizona member aged 30 with $30,000 annual employment
income ($2,500 monthly gross), 35 weekly work hours, and no reported shelter
expense has the same member and income facts in February and March 2026. The
PolicyEngine `tanf_non_cash_gross_income_limit` is about $2,412.71 in February
(185% FPL) and $2,608.33 in March (200% FPL); its
`meets_tanf_non_cash_gross_income_test`, `is_tanf_non_cash_eligible`,
`meets_snap_categorical_eligibility`, and `is_snap_eligible` all switch from
false to true. PolicyEngine SNAP is $0 then $24, with the $298 maximum unchanged.
These are observed PolicyEngine outputs, not a legal conclusion that every
household at this income qualifies: test the same facts against the
source-repaired Arizona ECE module and independent eligibility chain after
protected generation, including February/March benefit-month boundaries.

A **matched medical-deduction amount boundary**, not end-to-end household
parity: for January 2026 in Arizona, PolicyEngine US
`d89439134c1bac8add0c8261c8a075c33c39401a` with a 65-year-old member
and annual allowable medical expenses set to twelve times the stated monthly
amount, and the signed RuleSpec US PR #1351 branch at `a384b342e4b2b8d61467e903e36141ec92068b8b`
compiled with its declared engine `89571cc2a938707fd60a5489b951135345697296`,
both produce medical deductions of $0, $0, $145, $145, $145, and $205 for
monthly expenses of $0, $35, $36, $120, $180, and $240 respectively. The
RuleSpec cases supplied the source-required elderly-expense eligibility and
full-verification facts; PolicyEngine's annual expense input is converted to
monthly and does not model that verification fact. At $120 incurred but $0
verified, RuleSpec returns $0; no matched PolicyEngine scenario can express
the same verification condition. These results establish six aligned atomic
amount cases under explicit assumptions, not a validated composition or
unqualified oracle parity.

| Source / behavior | Provenance and scope | RuleSpec at pinned main | Evidence status and next check |
| --- | --- | --- | --- |
| Federal SNAP unit eligibility: income, assets, categorical exception, and at least one member meeting person-level conditions | PolicyEngine [`is_snap_eligible.py`](https://github.com/PolicyEngine/policyengine-us/blob/d89439134c1bac8add0c8261c8a075c33c39401a/policyengine_us/variables/gov/usda/snap/eligibility/is_snap_eligible.py) cites 7 USC 2017(a), 2014(c), and 2015(f). The federal 7 CFR 273.1 and 273.3–273.7, 273.10, and 273.24 modules are shared authority, not an Arizona-only gap. | Federal `us/policies/usda/snap/state-plan-composition.yaml` and imported regulation/statute modules exist. Arizona `fy-2026-benefit-calculation.yaml` imports the federal composition but still has an eligibility input bridge. | Not parity-validated. Compare matched households across each net/gross/resource/categorical/member gate, including disqualified members. Audit Arizona options and overrides separately. |
| Arizona medical deduction | PolicyEngine's SNAP medical-deduction standard [cites DES FAA5](https://github.com/PolicyEngine/policyengine-us/blob/d89439134c1bac8add0c8261c8a075c33c39401a/policyengine_us/parameters/gov/usda/snap/income/deductions/excess_medical_expense/standard.yaml); [official DES source](https://dbmefaapolicy.azdes.gov/FAA5/NA_Medical_Expenses_and_Deduction.html). | `us-az/policies/des/faa5/na-medical-expenses-and-deduction/medical-deduction.yaml`. | Source and file located; no matched-household parity result recorded. Check eligible person, countable expense, threshold, and effective date. |
| Arizona utility allowance eligibility and current amounts | PolicyEngine utility parameters cite [DES FAA5](https://dbmefaapolicy.azdes.gov/FAA5/NA_Utility_Expenses_and_Allowances.html) and [FAA6](https://dbmefaapolicy.azdes.gov/FAA6/Utility_Allowance_Current_Amount.html), for example [`includes_phone.yaml`](https://github.com/PolicyEngine/policyengine-us/blob/d89439134c1bac8add0c8261c8a075c33c39401a/policyengine_us/parameters/gov/usda/snap/income/deductions/utility/limited/includes_phone.yaml) and [`main.yaml`](https://github.com/PolicyEngine/policyengine-us/blob/d89439134c1bac8add0c8261c8a075c33c39401a/policyengine_us/parameters/gov/usda/snap/income/deductions/utility/limited/main.yaml). | `us-az/policies/des/faa5/na-utility-expenses-and-allowances/utility-allowance-eligibility.yaml` and `us-az/policies/des/faa6/utility-allowance-current-amount.yaml`. | Direct PolicyEngine January 2026 cases yield SUA $323/$438 for 1/4 people, LUA $149/$201 for 1/4 people, and phone-only $44 for one person, matching the five amounts in signed FAA6 RuleSpec. This is amount-level alignment, **not** matched RuleSpec execution or eligibility parity: Arizona's signed FAA5 rule additionally requires separate billing, payment obligation, and verified allowable utility expenses, while the compared PolicyEngine utility-type formula does not expose those facts. PolicyEngine also cites USDA/SnapScreener tables; those are comparison data, not independent Arizona legal authority. |
| Arizona expanded categorical eligibility (ECE) | Additional controlling [DES FAA5 categorical eligibility](https://dbmefaapolicy.azdes.gov/FAA5/NA_Categorical_Eligibility.html), corpus `us-az/manual/des/faa5/na-categorical-eligibility/block-4`. The pinned July 2026 corpus text says 200% FPL. DES's [FFY 2026 COLA page](https://dbmefaapolicy.azdes.gov/FAA5/FFY_2026_NA_COLA_Changes.html) and [03/23/2026 change notice](https://dbmefaapolicy.azdes.gov/Archived_Policy/baggage/2026-03-23_What'sChanged.pdf) attest 185% before benefit month 03/2026 and 200% from 03/2026; the announced 130% change was recalled. | `us-az/policies/des/faa5/na-categorical-eligibility/expanded-categorical-eligibility.yaml` still states 185% with an open-ended version. | **Material source discrepancy.** Protected atomic re-encode [run `37153765280`](https://github.com/TheAxiomFoundation/axiom-encode/actions/runs/37153765280) failed validation; no RuleSpec PR was produced. The generated candidate declares Household for several ECE rules, but the apply overlay restores their existing oracle-mapped Person shape before validation, producing Person-scope diagnostics against the Household candidate. [Encoder PR #1761](https://github.com/TheAxiomFoundation/axiom-encode/pull/1761) removed the invalid shape freeze for `not_comparable` entries, but no protected retry or RuleSpec repair has passed; earlier attempts also failed temporal-coverage and proof-excerpt checks. Ingest the newly located historical DES authority into a signed corpus release and test the February/March boundary before claiming the temporal repair. |
| Arizona basic categorical eligibility and benefit effects | Additional controlling DES FAA5 categorical-eligibility source, corpus `us-az/manual/des/faa5/na-categorical-eligibility/block-3` and related blocks. | Basic, expanded, and categorical-benefit modules exist. Protected BCE generation previously reached staging only. | Main-branch source/behavior review remains open; do not count staged output as merged coverage. |
| Arizona benefit calculation and initial-month proration | Additional controlling [DES FAA5 benefit determination](https://dbmefaapolicy.azdes.gov/FAA5/NA_Eligibility_and_Benefit_Determination.html), corpus block 9 and adjacent blocks. | Atomic `benefit-amount.yaml` and `first-month-benefit-proration.yaml` exist; `fy-2026-benefit-calculation.yaml` is a **composition** module. | Block-9 protected replacement runs `37148290459`, `37148769017`, and `37150620832` did not produce an accepted module. The atomic encoder cannot replace the composition as though it were an atomic source file. Preserve the composition while grounding other eligibility conditions in their own sources. |
| Arizona shelter deduction | DES FAA5 `shelter-expenses-and-deduction/block-3` states the excess-shelter computation, cap, elderly/disability exception, homeless alternative, and excluded payments. | `shelter-expenses-and-deduction/shelter-deduction.yaml` exists; it is not directly imported by the FY 2026 Arizona composition. | Encoded presence is not end-to-end behavior. Check household-size and elderly/disability gates, allowable costs, utility interaction, and cap by month against matched cases. |
| Arizona dependent-care deduction | DES FAA5 `dependent-care-expense/block-3` states qualifying dependents, work or training nexus, allowable billed costs, and exclusions. | `dependent-care-expense/na-dependent-care.yaml` exists; it is not directly imported by the FY 2026 Arizona composition. | No matched-household result is recorded. Test each qualifying purpose and exclusion, then trace whether the federal composition consumes the intended Arizona amount. |
| Arizona child-support deduction | DES FAA5 `na-child-support-expense/block-2` distinguishes court-ordered amounts actually paid, arrearages, lump sums, third-party payments, and spousal maintenance. | `na-child-support-expense/allowable-deductions.yaml` exists; it is not directly imported by the FY 2026 Arizona composition. | Test paid-versus-ordered limits, timing, and excluded payments. A Payment-level encoded rule alone does not establish the household deduction. |
| Disqualified NA participant effects | DES FAA5 `disqualified-na-participant-s-effect-on-the-na/block-2` addresses benefit unit size, utility allowance, and reason-specific income and expense counting. | `disqualified-na-participant-s-effect-on-the-na/disqualified-participant-effect-on-na-benefit-amount.yaml` exists; it is not directly imported by the FY 2026 Arizona composition. | Compare mixed eligible/disqualified households for each disqualification reason, including full versus prorated amounts; no such matched result is recorded. |
| Transitional, supplemental, and restored NA benefits | DES FAA5 `na-transitional-benefit-assistance-tba` and `supplemental-payments-and-restored-benefits` govern benefits outside ordinary monthly calculation. | `na-transitional-benefit-assistance-tba.yaml` and `supplemental-payments-and-restored-benefits.yaml` exist as separate modules. | Inventory triggering events and effective periods, then test transitions, five-month TBA limit, underpayments, and restored-benefit lookback. These are not demonstrated by a regular-month allotment comparison. |
| Arizona Simplified Nutrition Assistance Program (AZSNAP) | [DES FAA1 AZSNAP](https://dbmefaapolicy.azdes.gov/FAA1/Arizona_Simplified_Nutrition_Assistance_Progra.html) is an SSA-referred, SSI-linked simplified path with its own eligibility, reporting, and four-tier shelter-based benefit schedule. The [archived FAA6 schedule](https://dbmefaapolicy.azdes.gov/Archived_Policy/baggage/FAA6.J11_AZSNAPAllotmentAmounts_2026.pdf) covers 10/01/2023–12/31/2025 and gives tiers $103/$143/$177/$238; the [current FAA6 schedule](https://dbmefaapolicy.azdes.gov/FAA6/AZSNAP_Allotment_Amounts.html) begins 01/01/2026 and gives $72/$114/$148/$213. | No AZSNAP module or reference was found in pinned Arizona RuleSpec, and no dedicated FAA1/FAA6 AZSNAP provision in pinned Arizona corpus. The regular FY 2026 benefit composition is not this separate tiered allotment. Current and pre-January/pre-August FAA1 versions and both allotment tables are extracted in a local **unsigned** scope, not the pinned release. | Sign and release the staged FY 2026 authority, then source-bound encode and compare SSA-referral, SSI, living-arrangement, disqualification, shelter-tier, and 12/2025–01/2026 boundaries. Do not use regular SNAP allotment agreement as AZSNAP parity. |
| Arizona Elderly Simplified Application Project (ESAP) | [Archived FAA1 ESAP policy](https://dbmefaapolicy.azdes.gov/Archived_Policy/baggage/FAA1.D01C_ESAP_12.08.2025_Revision53.pdf) requires all members to be at least 60 with no earned/self-employment income and describes a 36-month approval period and simplified renewal. [DES's change history](https://dbmefaapolicy.azdes.gov/Archived_Policy/System_Information_and_Application_Screenin.html) raises the ESAP age threshold to 65 for pending actions from 08/11/2026 and open-case benefit months from 09/2026. | No ESAP module or reference was found in pinned Arizona RuleSpec. Pinned FAA5 corpus text mentions ESAP but has no dedicated FAA1 ESAP provision. Current and pre-April/pre-August FAA1 versions are extracted in a local **unsigned** scope, not the pinned release; archived PDFs required OCR because their embedded text was scrambled. | Sign and release the staged DES versions, obtain the applicable FNS demonstration authority, and test ages 60/64/65, earned-income transitions, interview/verification/renewal duties, and the distinct pending-action versus benefit-month boundary. Do not project the current age-65 rule back to January 2026. |
| Arizona residence and institution membership | Additional controlling [DES FAA3 Arizona Residency](https://dbmefaapolicy.azdes.gov/FAA3/Arizona_Residency.html) and [Residents of Institutions for NA](https://dbmefaapolicy.azdes.gov/FAA3/Residents_of_Institutions_for_NA.html); shared federal household/residence rules also apply. | The federal composition has residency and member wrappers; the Arizona composition does not yet bind a source-grounded Arizona eligibility chain. | Official pages located, but no FAA3 provision is present under `data/corpus/provisions/us-az` at the pinned corpus commit. Acquire release-bound source text, classify Arizona-specific conditions, then run positive/negative household tests. |
| Arizona noncitizen qualifications | Additional controlling [DES FAA3 NA Qualified Noncitizens](https://dbmefaapolicy.azdes.gov/FAA3/NA_Qualified_Noncitizens.html), alongside federal citizenship/noncitizen law and PolicyEngine's person-level immigration filter. [DES's FAA3 change history](https://dbmefaapolicy.azdes.gov/Archived_Policy/Deprivation_and_Absent_Parent_Information.html) identifies noncitizen policy changes aligned with H.R. 1 §10108 effective 03/01/2026 and a later LPR clarification; these need effective-date-specific source review. | Federal 7 CFR 273.4 and 7 USC 2015(f) modules exist; no source-grounded Arizona-specific bridge has been validated. | Official page located; FAA3 is absent from the pinned Arizona corpus provisions. Acquire current and prior authenticated source snapshots, distinguish the federal change from Arizona implementation, and test mixed-status units without treating separate eligible people as one qualifying person. |
| Arizona adult-student eligibility | [DES FAA3 Adult Student Eligibility for NA](https://dbmefaapolicy.azdes.gov/FAA3/Adult_Student_Eligibility_for_NA.html) applies enrollment, work/exemption, and meal-plan conditions; shared federal 7 CFR 273.5 also applies. The [DES change index](https://dbmefaapolicy.azdes.gov/Archived_Policy/Deprivation_and_Absent_Parent_Information.html) describes the 09/28/2026 revision as removing a CC note. | Federal `us/regulations/7-cfr/273/5.yaml` exists, but no Arizona FAA3 student module or binding into the Arizona eligibility composition was found in the current RuleSpec tree. Current and pre-September FAA3 pages are extracted in a local **unsigned** scope, not the pinned release; the archive required OCR. | Sign and release the staged historical page, reconcile Arizona implementation with federal student exemptions, and test enrollment, 80-hour paid work, work-study, and majority-meal-plan boundaries. No matched FY 2026 household result is recorded. |
| Arizona ordinary work registration and sanctions | [DES FAA6 NA Work Requirements](https://dbmefaapolicy.azdes.gov/FAA6/NA_Work_Requirements.html) screens participants aged 16–59 for work registration and exemptions; this is distinct from the ABAWD time limit. The [archive valid until 09/22/2025](https://dbmefaapolicy.azdes.gov/Archived_Policy/baggage/FAA6.B01_NAWorkRequirements_09-22-2025_Revision53.pdf) says work registrants aged 18–54 may also face ABAWD limits, while the [archive valid until 02/23/2026](https://dbmefaapolicy.azdes.gov/Archived_Policy/baggage/FAA6.B01_NAWorkReq_02-23-2026_Revision54.pdf) says ages 18–64. DES's [change index](https://dbmefaapolicy.azdes.gov/Archived_Policy/Case_Maintenance_and_Administrative_Lists_2.html) dates the H.R. 1-related Arizona change to 11/01/2025; neither archive by itself proves which text applied to October. Shared federal 7 CFR 273.7 applies. | Federal `us/regulations/7-cfr/273/7.yaml` exists; no Arizona FAA6 work-registration module or demonstrated member-to-household eligibility binding was found. Current and both archived pages are directly extracted in a local **unsigned** scope, not the pinned release. | Resolve the October/November operative-text boundary before source-bound encoding, including exemptions, voluntary quit/reduction, good cause, and disqualification effects. Test members who satisfy one work rule but not the other rather than treating ABAWD and ordinary registration as interchangeable. |
| Arizona expedited-service screening and issuance | [DES FAA2 archive valid until 02/02/2026](https://dbmefaapolicy.azdes.gov/Archived_Policy/baggage/FAA2.A03_ReqforNAX_Revision54.pdf) and the [archive valid until 07/27/2026](https://dbmefaapolicy.azdes.gov/Archived_Policy/baggage/FAA2.A03_ReqforNAX__07.27.26_Revision54.pdf) both state three screening routes: gross income under $150 with liquid resources at most $100; a destitute migrant/seasonal worker with resources at most $100; or income plus resources less than rent/mortgage plus the applicable utility allowance. They require determination and card availability by the seventh calendar day and generally permit verification other than identity to be postponed. DES removed telephone-application screening procedures effective 02/02/2026, following the end of telephone applications on 11/01/2025; that is distinct from the same-day interview process. A [02/09/2026 urgent bulletin](https://dbmefaapolicy.azdes.gov/Archived_Policy/baggage/Urgent%20Bulletin%20%2802-09-2026%29%20-%20NAX%20Same%20Day%20Interview%20Requirement.pdf), repeated in the [02/17 change notice](https://dbmefaapolicy.azdes.gov/Archived_Policy/baggage/2026-02-17_WhatChanged.pdf), expressly rescinded the same-day process change announced for 02/02 and immediately restored asking potentially eligible FAA-office applicants whether they could interview that day. The [current page](https://dbmefaapolicy.azdes.gov/FAA2/Requirements_for_NA_Expedited_Services_(NAX).html) dates its procedures to 09/01/2026. Shared 7 CFR 273.2(i) applies. | No Arizona FAA2 expedited-service module or federal `7-cfr/273/2/i` module was found in the current RuleSpec tree. Both archived policies, the current page, and the two February notices are directly extracted into a local **unsigned** scope with complete coverage, not the pinned release. | Sign and release the staged historical sources; separately test the telephone-application termination and 02/02–02/09 same-day interview transition, including what was operative during the short interval. Test screening thresholds, migrant/seasonal cases, negative gates, issuance timing, and postponed verification separately from ordinary monthly allotment. Resolve any applicable FNS waiver. |
| Arizona resource and income determinations | Additional controlling [DES FAA4 NA Resources](https://dbmefaapolicy.azdes.gov/FAA4/NA_Resources.html), [FAA4 Income Eligibility Requirements](https://dbmefaapolicy.azdes.gov/FAA4/Income_Eligibility_Requirements.html), [FAA6 Maximum NA Resource Limit](https://dbmefaapolicy.azdes.gov/FAA6/Maximum_NA_Resource_Limit.html), and [FAA6 NA Income Standards](https://dbmefaapolicy.azdes.gov/FAA6/NA_Income_Standards.html). | Federal income/resource modules and Arizona FAA5 gross/net-income-test modules exist; the Arizona composition still accepts eligibility as an input. | Official pages located, but these FAA4/FAA6 source units are not in the pinned Arizona corpus provisions. Source-ingest and separately encode operative Arizona limits/options before removing the caller bridge. |
| Arizona simplified reporting income threshold and other change duties | Additional controlling [DES FAA6 change history](https://dbmefaapolicy.azdes.gov/Archived_Policy/Case_Maintenance_and_Administrative_Lists.html) records a move from 130% to 200% FPL effective 03/01/2026, updated 04/13/2026; FAA5 benefit-determination change history records the same reporting change. The [current DES change-report page](https://des.az.gov/node/4174) also lists NA reporting for gross monthly income above 200% FPL (including disqualified participants), single-game lottery/gambling winnings, and loss of ABAWD work hours. Reporting is distinct from ECE eligibility even when both use 200% FPL. | No source-bound Arizona reporting-threshold module or matched reporting-behavior result is recorded in this matrix. | Historical and current official pages are audit leads only until authenticated source bytes are ingested. Test February/March reporting changes separately from ECE and test the other reportable events; do not infer parity from the shared percentage. |
| Arizona ABAWD work/time-limit implementation | Shared federal [7 CFR 273.24](https://www.ecfr.gov/current/title-7/subtitle-B/chapter-II/subchapter-C/part-273/section-273.24) controls the baseline. The [current DES public guidance](https://des.az.gov/services/basic-needs/food-assistance/nutrition-assistance/work-requirements-able-bodied-adult) describes 80 hours/month, exemptions and good cause, and Arizona's fixed 01/01/2025–12/31/2027 three-year clock; its current text must not be projected backward across federal or state changes. | Federal 7 CFR 273.24 modules exist; no matched Arizona ABAWD exemption, waiver-geography, countable-month, or regained-eligibility behavior is recorded here. | Obtain effective-date-specific DES policy and FNS-approved Arizona waiver authority, separate federal shared mechanics from Arizona options, then compare positive/negative monthly sequences. The indexed public guidance is an audit lead, not yet a release-bound source unit. |
| Arizona ABAWD geographic waiver, March 2026–February 2027 | [USDA FNS's March 12, 2026 approval](https://fns-prod.azureedge.us/sites/default/files/resource-files/az-abawd-response-fy2026.pdf) under 7 CFR 273.24(f) controls the approved geography and 03/01/2026–02/28/2027 interval; [DES's March 30 bulletin](https://dbmefaapolicy.azdes.gov/FAA2/baggage/Urgent%20Bulletin%20%2803-30-2026%29%20-%202026%20ABAWD%20Geographic%20Waiver.pdf) implements the same areas: Yuma County plus six named reservation areas. FNS distinguishes the request's proposed 10/01/2025 start from the **approved 03/01/2026 implementation date**. | No Arizona waiver-geography module or matched time-limit result is recorded. | Both official PDFs have been directly extracted into a local, unsigned corpus scope; this is not a released source unit. Test eligible and ineligible locations across February/March 2026 and the February/March 2027 expiration boundary, and audit any earlier waiver or termination separately. |
| Arizona ABAWD geographic waiver, October 2025–February 2026 | [FNS's FY 2025 Arizona approval](https://fns-prod.azureedge.us/sites/default/files/resource-files/az-abawd-response-fy2025.pdf) states an October 2024 start and a September 2025 expiration (the approval itself prints the nonexistent date September 31). [DES's August 28, 2025 urgent bulletin](https://dbmefaapolicy.azdes.gov/Archived_Policy/baggage/Urgent%20Bulletin%20%20%2808-28-2025%29%20-%20ABAWD%20Time%20Limit%20Waiver%20Expires.pdf) says the geographic waiver expires September 30 and the GE code ends for October benefit months. [DES's February 10, 2026 account](https://des.az.gov/node/27695) says Arizona requested a narrower FY 2026 waiver on September 26, 2025 and that USDA approval was still pending on February 10. The [FNS public response index](https://fns-prod.azureedge.us/snap/waivers/timelimit/2025-2029), checked 2026-10-06, lists Arizona's sole FY 2026 response as March 12, 2026; that is evidence against a published earlier approval, not proof that no interim action existed. Yet [FNS's FY 2026 first-quarter status report](https://fns-prod.azureedge.us/sites/default/files/resource-files/FY26-Quarter-1-ABAWD-Waiver-Status.pdf) lists Arizona as partially waived as of October 1, 2025. The report says it is based on active approvals and state status reports, and that states may discontinue a waiver before expiration; it does not identify an Arizona approval or its covered areas for October–February. The later DES account strengthens the evidence against assuming a new waiver had been approved by February 10, but it does not resolve the conflicting FNS status entry or prove there was no other interim authority. The DES bulletin separately introduces an AI exemption for qualifying Indian participants effective September 2025; that person-level exemption must not be mistaken for a continuing geographic waiver. | No matched October 2025–February 2026 geographic-waiver result is recorded. | The official DES bulletin and February account were directly extracted into local unsigned corpus scopes; neither is a released source unit. Check for any intervening FNS approval or correction, and reconcile administrative effective dates before encoding October–February geography. Test the separate AI exemption after source release; do not extrapolate the March 2026 approval backward. |
| Additional Arizona SNAP administration and eligibility | DES FAA5 has NA approval periods, disqualified-participant effects, transitional benefit assistance, child-support and dependent-care deductions; other DES chapters and federal/state plans must be inventoried for residency, membership, citizenship/immigration, income, resources, work, students, sanctions, expedited service, and reporting. | Some FAA5 modules exist, but an inventory of controlling source units and matched behavior is incomplete. | **Unresolved inventory, not proven absent coverage.** Classify each source as Arizona-specific, federal shared, unrelated cash-assistance material, or non-computational procedure before generating missing modules. |

A further contemporaneous state source bears on the disputed October 2025
ABAWD geography: [DES's FY2027 budget request, dated September 11,
2025](https://des.az.gov/sites/default/files/dl/FY2027-DES-Budget-Request.pdf)
says the FY2025 "insufficient jobs" waiver would cease to shield affected
recipients beginning October 1, 2025. The directly extracted official PDF has
SHA-256 `3df85141a9d745f0f7f36a4d847c47bcafb4003e8709b64292c7ad9f0769ef12`
(page 50 in the extracted scope). This is an agency budget projection, not an
FNS approval or binding determination, and it does not explain the contrary
FNS first-quarter status entry. Keep October–February geography unencoded
until the discrepancy is reconciled against approval-level authority.

The [archived DES FAA2 ABAWD exemption policy preceding its H.R. 1
implementation](https://dbmefaapolicy.azdes.gov/Archived_Policy/baggage/FAA2.M09B_ABAWD_Exemptions_09-22-2025_Revision53.pdf)
also explicitly ends its listed geographic exemption on September 30, 2025.
A [later archived DES policy](https://dbmefaapolicy.azdes.gov/Archived_Policy/baggage/FAA2.M09B_ABAWDExemptions_PS.pdf)
states that its replacement exemption rules begin with the November 2025
benefit month and refers earlier months to prior policy; it raises the upper
ABAWD age boundary to 64 and changes the child-related exemption. These
official PDFs were directly extracted into a local **unsigned** scope with
complete coverage; their archive watermarks and effective-date text must be
kept distinct. They strengthen the state-side October/November timeline but
still do not resolve FNS's contrary first-quarter geographic-waiver entry.

Separately, [DES's September 24, 2025 ABAWD look-back
bulletin](https://dbmefaapolicy.azdes.gov/Archived_Policy/baggage/Urgent%20Bulletin%20(09-24-2025)%20-%20ABAWD%20Three-Month%20Look%20Back.pdf)
instructs staff to treat August–October 2025 as countable for participants
whose homelessness, former-foster-youth, veteran, age-55–64, or older-child
exemptions ended, if they have no other exemption or qualifying work. It
directs adverse action for November after three countable months. This is
evidence for a distinct **person-level exemption transition**, not an FNS
approval identifying geographically waived areas or a resolution of the
contrary FNS first-quarter status entry. The [March 2, 2026 DES change
notice](https://dbmefaapolicy.azdes.gov/Archived_Policy/baggage/2026-03-02_WhatChanged.pdf)
repeats the need to correct August–October countable-month indicators; that
later operational reminder likewise does not establish geographic authority.

The block-9 compile failure is reproducible without another protected attempt.
Using the archived rejected candidate, pinned RuleSpec US
`9f38330fb43ffc693c6295b9a17c8d5d96520ad2`, and the failed run's engine
pin `89571cc2a938707fd60a5489b951135345697296` reproduces the reported
`module.source_verification.values` parse error. The generated target has no
such field; its import chain reaches the pre-existing federal FY 2026 COLA
`deductions.yaml`, whose line 20 **does** declare `values`. Compiling that
dependency alone produces the same error and line number. The engine reports
the top-level target path for an imported dependency's parse failure, which
obscured the cause. The encoder's source-value validator can read legacy
`values`, and current protected apply can remove a model-generated `values`
field after the engine rejects it; neither changes these pre-existing imports.
The pinned engine allows only `corpus_citation_path`, `source_sha256`, and
`upstream_source_check`. Direct compilation also rejects
the sibling federal `maximum-allotments.yaml` for `values` and
`income-eligibility-standards.yaml` for removed plural `corpus_citation_paths`;
all three are imported by `state-plan-composition.yaml`. Migrate these legacy
federal dependencies through protected source-bound generation before retrying
block 9; do not edit RuleSpec by hand. For income standards, the signed corpus
has a bodyless parent at
`us/guidance/usda/fns/snap-fy2026-income-eligibility-standards` with page-1
and page-2 descendants. The corpus resolver composes a bodyless parent's active
descendants under the singular requested citation, so the protected replacement
should target that parent rather than one page or a removed plural field.

PolicyEngine's SNAP tree at the pinned commit explicitly points to three unique
Arizona DES manual pages in its SNAP parameters (medical deduction, FAA5 utility
eligibility, FAA6 allowance amounts). This narrow citation set is **not** a
complete list of Arizona SNAP controlling authority. The remaining rows must be
built from the official DES manual, federal law/regulations, FNS approvals and
waivers, and source-effective-date history. The current effective-policy audit
also needs an FFY 2027 branch: [DES lists FFY 2027 NA COLA changes effective
2026-10-01](https://dbmefaapolicy.azdes.gov/FAA5/FFY_2027_NA_COLA_Changes.html),
while the existing composition is explicitly FY 2026.

Completion for any behavior requires all four: a controlling source and
effective-date pin; source-bound RuleSpec (or an explicit unsupported-case
record); focused plus broad tests; and a matched-household oracle comparison
whose inputs and outputs are actually comparable. Disagreements are resolved
against controlling authority, not by assuming either PolicyEngine or
SnapScreener is correct.

Source acquisition has advanced, but release provenance remains a blocker.
Ordinary direct HTTPS fetches of DES pages returned HTTP 403 with
`Cf-Mitigated: challenge` on 2026-10-03. The corpus extractor's
`request.browser_impersonation: chrome120` fetched the current official DES
FAA3 residency, qualified-noncitizen, and institution pages; FAA4 NA resources
and income-eligibility pages; and FAA6 resource-limit and income-standard
tables directly. Native extraction reported complete coverage in multiple local
manifest scopes, including the historical ECE scopes below and a retained
pre-March 2026 FAA3 qualified-noncitizen page. These are staged,
unsigned source candidates, **not** pinned or released corpus units. The FAA6
200% table expressly dates both ECE and simplified reporting to 03/01/2026;
those remain distinct behaviors. Do not infer protected-encoding readiness
until authorized signing, locking, release, and effective-date audit complete.

An archival route now provides candidate source bytes for provenance review:
the [Internet Archive's 2025-10-30 capture of the official DES FFY 2026 COLA
page](https://web.archive.org/web/20251030220841id_/https://dbmefaapolicy.azdes.gov/FAA5/FFY_2026_NA_COLA_Changes.html)
states the 185% ECE standard effective 10/01/2025, and its SHA-1 payload digest
matches the archive CDX record (`Q43XL5KP4LKW55V5I3TGJ2OJ734DYBPB`; local
SHA-256 `3dc09c5a979a5593585d05d543372acbef38dbd43616ea506db14e9932696daa`).
The [2026-04-17 capture of DES's 2026-03-23 change
notice](https://web.archive.org/web/20260417195741id_/https://dbmefaapolicy.azdes.gov/Archived_Policy/baggage/2026-03-23_What%27sChanged.pdf)
states the 200% standard for benefit month 03/2026 and the recalled 130%
announcement. Its SHA-1 payload digest matches CDX
(`UJIAFW65IDNJ7HBLBST5WHPJZMCHICLJ`; local SHA-256
`f8173335f1f7f73467020ed8a13d0e7ad2efd5a92d83f74f7c471b57d27232e3`),
and the Memento response preserves the original DES URL, 200 response, and
original last-modified header. These are archived official-origin candidates,
not yet signed corpus units. Verify the capture provenance and ingest them in
a new release before relying on them for protected generation. A subsequent
browser-impersonated fetch of the DES change-notice URL returned the same PDF
bytes as the archived capture (SHA-256
`f8173335f1f7f73467020ed8a13d0e7ad2efd5a92d83f74f7c471b57d27232e3`).
The live official FFY 2026 COLA page now also explicitly records the 185%-to-200%
ECE transition for benefit month 03/2026. Both directly fetched scopes were
extracted locally, but neither is signed or released.
