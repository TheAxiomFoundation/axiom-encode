# Arizona SNAP coverage audit (working matrix)

This is an evidence ledger, not a parity claim. It distinguishes sources cited by
PolicyEngine from additional controlling Arizona policy, and distinguishes an
encoded RuleSpec file from tested household behavior. The pinned comparison
points for this first pass are PolicyEngine US `d89439134c1bac8add0c8261c8a075c33c39401a`,
RuleSpec US `9f38330fb43ffc693c6295b9a17c8d5d96520ad2`, and Axiom Corpus
`8f7d60aaced28ee4252b9237f9d6e02360dc34bc`. Re-audit these rows when any
pin or legal effective date changes. `not_comparable` oracle output is **not**
behavioral parity.

| Source / behavior | Provenance and scope | RuleSpec at pinned main | Evidence status and next check |
| --- | --- | --- | --- |
| Federal SNAP unit eligibility: income, assets, categorical exception, and at least one member meeting person-level conditions | PolicyEngine [`is_snap_eligible.py`](https://github.com/PolicyEngine/policyengine-us/blob/d89439134c1bac8add0c8261c8a075c33c39401a/policyengine_us/variables/gov/usda/snap/eligibility/is_snap_eligible.py) cites 7 USC 2017(a), 2014(c), and 2015(f). The federal 7 CFR 273.1 and 273.3–273.7, 273.10, and 273.24 modules are shared authority, not an Arizona-only gap. | Federal `us/policies/usda/snap/state-plan-composition.yaml` and imported regulation/statute modules exist. Arizona `fy-2026-benefit-calculation.yaml` imports the federal composition but still has an eligibility input bridge. | Not parity-validated. Compare matched households across each net/gross/resource/categorical/member gate, including disqualified members. Audit Arizona options and overrides separately. |
| Arizona medical deduction | PolicyEngine's SNAP medical-deduction standard [cites DES FAA5](https://github.com/PolicyEngine/policyengine-us/blob/d89439134c1bac8add0c8261c8a075c33c39401a/policyengine_us/parameters/gov/usda/snap/income/deductions/excess_medical_expense/standard.yaml); [official DES source](https://dbmefaapolicy.azdes.gov/FAA5/NA_Medical_Expenses_and_Deduction.html). | `us-az/policies/des/faa5/na-medical-expenses-and-deduction/medical-deduction.yaml`. | Source and file located; no matched-household parity result recorded. Check eligible person, countable expense, threshold, and effective date. |
| Arizona utility allowance eligibility and current amounts | PolicyEngine utility parameters cite [DES FAA5](https://dbmefaapolicy.azdes.gov/FAA5/NA_Utility_Expenses_and_Allowances.html) and [FAA6](https://dbmefaapolicy.azdes.gov/FAA6/Utility_Allowance_Current_Amount.html), for example [`includes_phone.yaml`](https://github.com/PolicyEngine/policyengine-us/blob/d89439134c1bac8add0c8261c8a075c33c39401a/policyengine_us/parameters/gov/usda/snap/income/deductions/utility/limited/includes_phone.yaml) and [`main.yaml`](https://github.com/PolicyEngine/policyengine-us/blob/d89439134c1bac8add0c8261c8a075c33c39401a/policyengine_us/parameters/gov/usda/snap/income/deductions/utility/limited/main.yaml). | `us-az/policies/des/faa5/na-utility-expenses-and-allowances/utility-allowance-eligibility.yaml` and `us-az/policies/des/faa6/utility-allowance-current-amount.yaml`. | Source and files located; compare standard, limited, and telephone allowance selection and amounts by month. PolicyEngine also cites USDA/SnapScreener tables; those are comparison data, not independent Arizona legal authority. |
| Arizona expanded categorical eligibility (ECE) | Additional controlling [DES FAA5 categorical eligibility](https://dbmefaapolicy.azdes.gov/FAA5/NA_Categorical_Eligibility.html), corpus `us-az/manual/des/faa5/na-categorical-eligibility/block-4`. The pinned July 2026 corpus text says 200% FPL; [DES change history](https://dbmefaapolicy.azdes.gov/Archived_Policy/Work_Registration_and_Program_Determination.html) reports an effective date of 2026-03-01. | `us-az/policies/des/faa5/na-categorical-eligibility/expanded-categorical-eligibility.yaml` still states 185% with an open-ended version. | **Material source discrepancy.** Protected atomic re-encode dispatched as axiom-encode run `37153765280`; not accepted or merged. Historical 185% coverage needs separately attested authority and a dated test. |
| Arizona basic categorical eligibility and benefit effects | Additional controlling DES FAA5 categorical-eligibility source, corpus `us-az/manual/des/faa5/na-categorical-eligibility/block-3` and related blocks. | Basic, expanded, and categorical-benefit modules exist. Protected BCE generation previously reached staging only. | Main-branch source/behavior review remains open; do not count staged output as merged coverage. |
| Arizona benefit calculation and initial-month proration | Additional controlling [DES FAA5 benefit determination](https://dbmefaapolicy.azdes.gov/FAA5/NA_Eligibility_and_Benefit_Determination.html), corpus block 9 and adjacent blocks. | Atomic `benefit-amount.yaml` and `first-month-benefit-proration.yaml` exist; `fy-2026-benefit-calculation.yaml` is a **composition** module. | Block-9 protected replacement runs `37148290459`, `37148769017`, and `37150620832` did not produce an accepted module. The atomic encoder cannot replace the composition as though it were an atomic source file. Preserve the composition while grounding other eligibility conditions in their own sources. |
| Additional Arizona SNAP administration and eligibility | DES FAA5 has NA approval periods, disqualified-participant effects, transitional benefit assistance, child-support and dependent-care deductions; other DES chapters and federal/state plans must be inventoried for residency, membership, citizenship/immigration, income, resources, work, students, sanctions, expedited service, and reporting. | Some FAA5 modules exist, but an inventory of controlling source units and matched behavior is incomplete. | **Unresolved inventory, not proven absent coverage.** Classify each source as Arizona-specific, federal shared, unrelated cash-assistance material, or non-computational procedure before generating missing modules. |

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
