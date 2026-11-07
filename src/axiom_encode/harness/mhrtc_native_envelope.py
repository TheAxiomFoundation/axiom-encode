"""Reviewed ordinary metadata for the 99-rule MHRtC reference.

Exact source excerpts and proof paths are necessary, not coverage authority.
No candidate files or external fixtures are read at runtime.
"""

from .mhrtc_program_contract import CITATION, SOURCE_TEXT

MODULE = {
    "summary": "2025 Schedule 12: selected qualifying renovation expenditures and MHRTC for one "
    "filing claimant, with per-renovation eligibility, shared claims and documentary "
    "duties.",
    "proof_validation": {"required": True},
    "source_verification": {
        "corpus_citation_path": "ca/policy/cra/t1-2025/multigenerational-home-renovation-tax-credit",
        "source_sha256": "6b2af117aad5a7e5373ed26aeb81ef91038e1c7ac4cb2d7ab3c448e0eaca5fd0",
    },
}

INPUT_DESCRIPTIONS = {
    "canada_resident_full_year": "The claimant was resident in Canada throughout the renovation "
    "completion tax year.",
    "individual_dtc": "The attributed individual is eligible for the disability tax credit for the "
    "renovation completion tax year.",
    "claimant_is_individual": "The claimant is the attributed qualifying individual; age and DTC "
    "eligibility are computed separately.",
    "claimant_is_individual_cohabiting_partner": "The claimant was the attributed individual’s "
    "cohabiting spouse or common-law partner at any time "
    "in the renovation completion tax year.",
    "claimant_is_relation": "The claimant is the person whose relationship and age are supplied in "
    "this claim; qualifying-relation eligibility is computed separately.",
    "claimant_dwelling_occupancy_within_twelve_months": "The claimant ordinarily resides, or intends "
    "to ordinarily reside, in this dwelling "
    "within twelve months after renovation "
    "completion.",
    "claimant_owns_dwelling": "The claimant owns the dwelling for the owning-relative claimant "
    "alternative.",
    "claimant_owning_trust_beneficiary": "The claimant is a beneficiary of a trust owning the "
    "dwelling for the owning-relative claimant alternative.",
    "dwelling_is_housing_unit": "The dwelling is a housing unit.",
    "dwelling_in_canada": "The dwelling is located in Canada.",
    "individual_dwelling_occupancy_within_twelve_months": "The attributed individual ordinarily "
    "resides, or intends to ordinarily reside, "
    "in the dwelling within twelve months after "
    "completion.",
    "relation_dwelling_occupancy_within_twelve_months": "The attributed relation ordinarily resides, "
    "or intends to ordinarily reside, in the "
    "dwelling within twelve months after "
    "completion.",
    "dwelling_owned_by_individual_or_relation_during_year": "The dwelling was owned or jointly owned "
    "by the attributed individual or relation "
    "at any time in the renovation completion "
    "tax year.",
    "dwelling_owner_trust_has_individual_or_relation_beneficiary_during_year": "A trust owned the "
    "dwelling and the "
    "attributed individual "
    "or relation was its "
    "beneficiary at a "
    "qualifying time in "
    "the renovation "
    "completion tax year.",
    "renovation_enduring": "The renovation or alteration or addition is of an enduring nature.",
    "renovation_integral": "The renovation is integral to this dwelling.",
    "renovation_enables_coresidence": "The renovation establishes a secondary unit within this "
    "dwelling that allows the attributed individual to reside with "
    "the attributed relation.",
    "secondary_occupancy_by_individual": "The secondary unit is for occupancy by the attributed "
    "individual.",
    "secondary_occupancy_by_relation": "The secondary unit is for occupancy by the attributed "
    "relation.",
    "secondary_self_contained": "The secondary unit is a self-contained housing unit.",
    "secondary_private_entrance": "The secondary unit has a private entrance.",
    "secondary_kitchen": "The secondary unit has a kitchen.",
    "secondary_bathroom": "The secondary unit has bathroom facilities.",
    "secondary_sleeping_area": "The secondary unit has a sleeping area.",
    "local_requirements_apply": "Local requirements apply to qualification of this secondary dwelling "
    "unit.",
    "local_requirements_met": "The secondary unit meets the applicable local requirements; this fact "
    "is not a computed local-code determination.",
    "relationship_to_individual_or_cohabiting_partner_during_year": "The stated family relationship "
    "existed to the attributed "
    "individual or their cohabiting "
    "spouse or common-law partner at "
    "any time in the completion tax "
    "year.",
    "individual_age_at_year_end": "Age in completed years of the attributed individual before the end "
    "of the renovation completion tax year.",
    "relation_age_at_year_end": "Age in completed years of the attributed relation before the end of "
    "the renovation completion tax year.",
    "renovation_key": "Stable factual key for this physical renovation, shared across all claimant "
    "records of the same renovation; not a qualifying-individual or row identifier.",
    "claimant_key": "Stable factual key for this tax claimant; equal keys identify the same person "
    "across current and other-claim records.",
    "individual_key": "Stable factual key for the individual attributed to this claim; multiple "
    "occupants must not duplicate the same claimant/physical-renovation claim.",
    "family_relationship": "Actual family relationship to the attributed individual or their "
    "cohabiting partner; the listed qualifying relationships are tested "
    "locally.",
    "completion_date": "Date of actual renovation completion, such as final inspection or received "
    "proof of project completion; controls the renovation period tax year.",
    "dwelling_land_scope_key": "Stable factual key identifying the land extent proposed as part of "
    "this dwelling.",
    "dwelling_land_area_hectares": "Area in hectares of that entire identified land extent; not the "
    "whole property by default and not an expense allocation ratio.",
    "additional_land_scope_applies": "An actual applicable determination permits this identified land "
    "extent beyond the general half-hectare limit; no exception "
    "criteria or authority decision are computed here.",
    "actual_minister_decision_exists": "An actual Minister of National Revenue allocation decision "
    "exists; disagreement alone is not such a decision.",
    "decision_claimant_key": "Stable claimant key to which the actual allocation decision applies; "
    "ignored when no decision exists.",
    "decision_renovation_key": "Stable physical-renovation key to which the actual allocation "
    "decision applies; ignored when no decision exists.",
    "allocated_expenditure_portion": "CAD expenditure portion allocated to this claimant and physical "
    "renovation by an actual matching ministerial decision, not a "
    "credit amount or invented fallback.",
    "historical_individual_key": "Stable individual key associated with an actual historical "
    "qualifying-renovation claim; the supplied history is complete for "
    "this individual.",
    "historical_renovation_key": "Stable physical-renovation key of that actual historical claim; the "
    "same renovation shared by claimants is not a different lifetime "
    "renovation.",
    "expense_reasonable": "This expense is reasonable.",
    "expense_directly_attributable": "This expense is directly attributable to the parent claim’s "
    "renovation of its dwelling.",
    "paid_event_exists": "An actual payment event exists for this expense; its date, payer and "
    "within-day ordering facts describe the same event.",
    "incurred_event_exists": "An actual incurrence event exists for this expense; its date, liable "
    "person and within-day ordering facts describe the same event.",
    "paid_by_claimant": "The payment event was made by the parent claim’s claimant.",
    "incurred_by_claimant": "The expense was incurred by the parent claim’s claimant in the stated "
    "incurrence event.",
    "paid_by_claimant_beneficiary_trust": "The payment event was made by a trust of which the parent "
    "claim’s claimant is a beneficiary.",
    "incurred_by_claimant_beneficiary_trust": "The incurrence event was by a trust of which the "
    "parent claim’s claimant is a beneficiary.",
    "medical_credit_claimed": "This expense was claimed for the medical expense tax credit.",
    "home_accessibility_credit_claimed": "This expense was claimed for the home accessibility tax "
    "credit.",
    "paid_same_day_before_completion": "For a payment on the completion date, the actual payment "
    "occurred before completion; date equality alone does not "
    "establish this fact.",
    "incurred_same_day_before_completion": "For incurrence on the completion date, the actual "
    "incurrence occurred before completion; date equality "
    "alone does not establish this fact.",
    "expense_is_building": "This receipt component concerns the housing unit itself rather than land; "
    "mixed expenses must be split into nonduplicated truthful components.",
    "expense_is_land": "This receipt component concerns the identified land extent; building and land "
    "classifications must be coherent.",
    "expense_land_scope_key": "Stable factual key for the land extent to which this receipt component "
    "relates, compared with the parent claim’s land extent.",
    "paid_date": "Date of the coherent actual payment event; inactive placeholder dates cannot "
    "establish a missing event.",
    "incurred_date": "Date of the coherent actual incurrence event; inactive placeholder dates cannot "
    "establish a missing event.",
    "amount_including_taxes": "Nonnegative CAD expense amount including applicable taxes for this "
    "receipt component, counted once.",
    "reimbursement_received_or_receivable": "Nonnegative CAD reimbursement or assistance received or "
    "receivable attributable to this receipt component, "
    "including related GST/HST rebates; do not duplicate "
    "assistance across receipts.",
    "other_canada_resident_full_year": "OtherClaim lane: The claimant was resident in Canada "
    "throughout the renovation completion tax year. The claimant, "
    "renovation, dwelling and event associations refer to the "
    "linked OtherClaim, not the filing claimant.",
    "other_individual_dtc": "OtherClaim lane: The attributed individual is eligible for the "
    "disability tax credit for the renovation completion tax year. The "
    "claimant, renovation, dwelling and event associations refer to the "
    "linked OtherClaim, not the filing claimant.",
    "other_claimant_is_individual": "OtherClaim lane: The claimant is the attributed qualifying "
    "individual; age and DTC eligibility are computed separately. The "
    "claimant, renovation, dwelling and event associations refer to "
    "the linked OtherClaim, not the filing claimant.",
    "other_claimant_is_individual_cohabiting_partner": "OtherClaim lane: The claimant was the "
    "attributed individual’s cohabiting spouse or "
    "common-law partner at any time in the "
    "renovation completion tax year. The claimant, "
    "renovation, dwelling and event associations "
    "refer to the linked OtherClaim, not the "
    "filing claimant.",
    "other_claimant_is_relation": "OtherClaim lane: The claimant is the person whose relationship and "
    "age are supplied in this claim; qualifying-relation eligibility is "
    "computed separately. The claimant, renovation, dwelling and event "
    "associations refer to the linked OtherClaim, not the filing "
    "claimant.",
    "other_claimant_dwelling_occupancy_within_twelve_months": "OtherClaim lane: The claimant "
    "ordinarily resides, or intends to "
    "ordinarily reside, in this dwelling "
    "within twelve months after renovation "
    "completion. The claimant, renovation, "
    "dwelling and event associations refer "
    "to the linked OtherClaim, not the "
    "filing claimant.",
    "other_claimant_owns_dwelling": "OtherClaim lane: The claimant owns the dwelling for the "
    "owning-relative claimant alternative. The claimant, renovation, "
    "dwelling and event associations refer to the linked OtherClaim, "
    "not the filing claimant.",
    "other_claimant_owning_trust_beneficiary": "OtherClaim lane: The claimant is a beneficiary of a "
    "trust owning the dwelling for the owning-relative "
    "claimant alternative. The claimant, renovation, "
    "dwelling and event associations refer to the linked "
    "OtherClaim, not the filing claimant.",
    "other_dwelling_is_housing_unit": "OtherClaim lane: The dwelling is a housing unit. The claimant, "
    "renovation, dwelling and event associations refer to the "
    "linked OtherClaim, not the filing claimant.",
    "other_dwelling_in_canada": "OtherClaim lane: The dwelling is located in Canada. The claimant, "
    "renovation, dwelling and event associations refer to the linked "
    "OtherClaim, not the filing claimant.",
    "other_individual_dwelling_occupancy_within_twelve_months": "OtherClaim lane: The attributed "
    "individual ordinarily resides, or "
    "intends to ordinarily reside, in the "
    "dwelling within twelve months after "
    "completion. The claimant, "
    "renovation, dwelling and event "
    "associations refer to the linked "
    "OtherClaim, not the filing claimant.",
    "other_relation_dwelling_occupancy_within_twelve_months": "OtherClaim lane: The attributed "
    "relation ordinarily resides, or "
    "intends to ordinarily reside, in the "
    "dwelling within twelve months after "
    "completion. The claimant, renovation, "
    "dwelling and event associations refer "
    "to the linked OtherClaim, not the "
    "filing claimant.",
    "other_dwelling_owned_by_individual_or_relation_during_year": "OtherClaim lane: The dwelling was "
    "owned or jointly owned by the "
    "attributed individual or relation "
    "at any time in the renovation "
    "completion tax year. The claimant, "
    "renovation, dwelling and event "
    "associations refer to the linked "
    "OtherClaim, not the filing "
    "claimant.",
    "other_dwelling_owner_trust_has_individual_or_relation_beneficiary_during_year": "OtherClaim "
    "lane: A trust "
    "owned the "
    "dwelling and "
    "the attributed "
    "individual or "
    "relation was "
    "its beneficiary "
    "at a qualifying "
    "time in the "
    "renovation "
    "completion tax "
    "year. The "
    "claimant, "
    "renovation, "
    "dwelling and "
    "event "
    "associations "
    "refer to the "
    "linked "
    "OtherClaim, not "
    "the filing "
    "claimant.",
    "other_renovation_enduring": "OtherClaim lane: The renovation or alteration or addition is of an "
    "enduring nature. The claimant, renovation, dwelling and event "
    "associations refer to the linked OtherClaim, not the filing "
    "claimant.",
    "other_renovation_integral": "OtherClaim lane: The renovation is integral to this dwelling. The "
    "claimant, renovation, dwelling and event associations refer to the "
    "linked OtherClaim, not the filing claimant.",
    "other_renovation_enables_coresidence": "OtherClaim lane: The renovation establishes a secondary "
    "unit within this dwelling that allows the attributed "
    "individual to reside with the attributed relation. The "
    "claimant, renovation, dwelling and event associations "
    "refer to the linked OtherClaim, not the filing claimant.",
    "other_secondary_occupancy_by_individual": "OtherClaim lane: The secondary unit is for occupancy "
    "by the attributed individual. The claimant, "
    "renovation, dwelling and event associations refer to "
    "the linked OtherClaim, not the filing claimant.",
    "other_secondary_occupancy_by_relation": "OtherClaim lane: The secondary unit is for occupancy by "
    "the attributed relation. The claimant, renovation, "
    "dwelling and event associations refer to the linked "
    "OtherClaim, not the filing claimant.",
    "other_secondary_self_contained": "OtherClaim lane: The secondary unit is a self-contained "
    "housing unit. The claimant, renovation, dwelling and event "
    "associations refer to the linked OtherClaim, not the filing "
    "claimant.",
    "other_secondary_private_entrance": "OtherClaim lane: The secondary unit has a private entrance. "
    "The claimant, renovation, dwelling and event associations "
    "refer to the linked OtherClaim, not the filing claimant.",
    "other_secondary_kitchen": "OtherClaim lane: The secondary unit has a kitchen. The claimant, "
    "renovation, dwelling and event associations refer to the linked "
    "OtherClaim, not the filing claimant.",
    "other_secondary_bathroom": "OtherClaim lane: The secondary unit has bathroom facilities. The "
    "claimant, renovation, dwelling and event associations refer to the "
    "linked OtherClaim, not the filing claimant.",
    "other_secondary_sleeping_area": "OtherClaim lane: The secondary unit has a sleeping area. The "
    "claimant, renovation, dwelling and event associations refer to "
    "the linked OtherClaim, not the filing claimant.",
    "other_local_requirements_apply": "OtherClaim lane: Local requirements apply to qualification of "
    "this secondary dwelling unit. The claimant, renovation, "
    "dwelling and event associations refer to the linked "
    "OtherClaim, not the filing claimant.",
    "other_local_requirements_met": "OtherClaim lane: The secondary unit meets the applicable local "
    "requirements; this fact is not a computed local-code "
    "determination. The claimant, renovation, dwelling and event "
    "associations refer to the linked OtherClaim, not the filing "
    "claimant.",
    "other_relationship_to_individual_or_cohabiting_partner_during_year": "OtherClaim lane: The "
    "stated family relationship "
    "existed to the attributed "
    "individual or their "
    "cohabiting spouse or "
    "common-law partner at any "
    "time in the completion tax "
    "year. The claimant, "
    "renovation, dwelling and "
    "event associations refer "
    "to the linked OtherClaim, "
    "not the filing claimant.",
    "other_individual_age_at_year_end": "OtherClaim lane: Age in completed years of the attributed "
    "individual before the end of the renovation completion tax "
    "year. The claimant, renovation, dwelling and event "
    "associations refer to the linked OtherClaim, not the filing "
    "claimant.",
    "other_relation_age_at_year_end": "OtherClaim lane: Age in completed years of the attributed "
    "relation before the end of the renovation completion tax year. "
    "The claimant, renovation, dwelling and event associations "
    "refer to the linked OtherClaim, not the filing claimant.",
    "other_renovation_key": "OtherClaim lane: Stable factual key for this physical renovation, shared "
    "across all claimant records of the same renovation; not a "
    "qualifying-individual or row identifier. The claimant, renovation, "
    "dwelling and event associations refer to the linked OtherClaim, not the "
    "filing claimant.",
    "other_claimant_key": "OtherClaim lane: Stable factual key for this tax claimant; equal keys "
    "identify the same person across current and other-claim records. The "
    "claimant, renovation, dwelling and event associations refer to the linked "
    "OtherClaim, not the filing claimant.",
    "other_individual_key": "OtherClaim lane: Stable factual key for the individual attributed to "
    "this claim; multiple occupants must not duplicate the same "
    "claimant/physical-renovation claim. The claimant, renovation, dwelling "
    "and event associations refer to the linked OtherClaim, not the filing "
    "claimant.",
    "other_family_relationship": "OtherClaim lane: Actual family relationship to the attributed "
    "individual or their cohabiting partner; the listed qualifying "
    "relationships are tested locally. The claimant, renovation, "
    "dwelling and event associations refer to the linked OtherClaim, not "
    "the filing claimant.",
    "other_completion_date": "OtherClaim lane: Date of actual renovation completion, such as final "
    "inspection or received proof of project completion; controls the "
    "renovation period tax year. The claimant, renovation, dwelling and "
    "event associations refer to the linked OtherClaim, not the filing "
    "claimant.",
    "other_actual_claimed_expenses": "OtherClaim lane: Nonnegative CAD eligible expenditure amount "
    "actually being claimed by this other claimant, not a requested "
    "credit or raw expense total; its eligibility and local ceiling "
    "are computed separately. The claimant, renovation, dwelling and "
    "event associations refer to the linked OtherClaim, not the "
    "filing claimant.",
    "other_dwelling_land_scope_key": "OtherClaim lane: Stable factual key identifying the land extent "
    "proposed as part of this dwelling. The claimant, renovation, "
    "dwelling and event associations refer to the linked OtherClaim, "
    "not the filing claimant.",
    "other_dwelling_land_area_hectares": "OtherClaim lane: Area in hectares of that entire identified "
    "land extent; not the whole property by default and not an "
    "expense allocation ratio. The claimant, renovation, "
    "dwelling and event associations refer to the linked "
    "OtherClaim, not the filing claimant.",
    "other_additional_land_scope_applies": "OtherClaim lane: An actual applicable determination "
    "permits this identified land extent beyond the general "
    "half-hectare limit; no exception criteria or authority "
    "decision are computed here. The claimant, renovation, "
    "dwelling and event associations refer to the linked "
    "OtherClaim, not the filing claimant.",
    "other_actual_minister_decision_exists": "OtherClaim lane: An actual Minister of National Revenue "
    "allocation decision exists; disagreement alone is not "
    "such a decision. The claimant, renovation, dwelling and "
    "event associations refer to the linked OtherClaim, not "
    "the filing claimant.",
    "other_decision_claimant_key": "OtherClaim lane: Stable claimant key to which the actual "
    "allocation decision applies; ignored when no decision exists. The "
    "claimant, renovation, dwelling and event associations refer to "
    "the linked OtherClaim, not the filing claimant.",
    "other_decision_renovation_key": "OtherClaim lane: Stable physical-renovation key to which the "
    "actual allocation decision applies; ignored when no decision "
    "exists. The claimant, renovation, dwelling and event "
    "associations refer to the linked OtherClaim, not the filing "
    "claimant.",
    "other_allocated_expenditure_portion": "OtherClaim lane: CAD expenditure portion allocated to "
    "this claimant and physical renovation by an actual "
    "matching ministerial decision, not a credit amount or "
    "invented fallback. The claimant, renovation, dwelling and "
    "event associations refer to the linked OtherClaim, not "
    "the filing claimant.",
    "other_historical_individual_key": "OtherClaim lane: Stable individual key associated with an "
    "actual historical qualifying-renovation claim; the supplied "
    "history is complete for this individual. The claimant, "
    "renovation, dwelling and event associations refer to the "
    "linked OtherClaim, not the filing claimant.",
    "other_historical_renovation_key": "OtherClaim lane: Stable physical-renovation key of that "
    "actual historical claim; the same renovation shared by "
    "claimants is not a different lifetime renovation. The "
    "claimant, renovation, dwelling and event associations refer "
    "to the linked OtherClaim, not the filing claimant.",
    "other_expense_reasonable": "OtherClaim lane: This expense is reasonable. The claimant, "
    "renovation, dwelling and event associations refer to the linked "
    "OtherClaim, not the filing claimant.",
    "other_expense_directly_attributable": "OtherClaim lane: This expense is directly attributable to "
    "the parent claim’s renovation of its dwelling. The "
    "claimant, renovation, dwelling and event associations "
    "refer to the linked OtherClaim, not the filing claimant.",
    "other_paid_event_exists": "OtherClaim lane: An actual payment event exists for this expense; its "
    "date, payer and within-day ordering facts describe the same event. "
    "The claimant, renovation, dwelling and event associations refer to "
    "the linked OtherClaim, not the filing claimant.",
    "other_incurred_event_exists": "OtherClaim lane: An actual incurrence event exists for this "
    "expense; its date, liable person and within-day ordering facts "
    "describe the same event. The claimant, renovation, dwelling and "
    "event associations refer to the linked OtherClaim, not the filing "
    "claimant.",
    "other_paid_by_claimant": "OtherClaim lane: The payment event was made by the parent claim’s "
    "claimant. The claimant, renovation, dwelling and event associations "
    "refer to the linked OtherClaim, not the filing claimant.",
    "other_incurred_by_claimant": "OtherClaim lane: The expense was incurred by the parent claim’s "
    "claimant in the stated incurrence event. The claimant, renovation, "
    "dwelling and event associations refer to the linked OtherClaim, "
    "not the filing claimant.",
    "other_paid_by_claimant_beneficiary_trust": "OtherClaim lane: The payment event was made by a "
    "trust of which the parent claim’s claimant is a "
    "beneficiary. The claimant, renovation, dwelling and "
    "event associations refer to the linked OtherClaim, "
    "not the filing claimant.",
    "other_incurred_by_claimant_beneficiary_trust": "OtherClaim lane: The incurrence event was by a "
    "trust of which the parent claim’s claimant is a "
    "beneficiary. The claimant, renovation, dwelling "
    "and event associations refer to the linked "
    "OtherClaim, not the filing claimant.",
    "other_medical_credit_claimed": "OtherClaim lane: This expense was claimed for the medical "
    "expense tax credit. The claimant, renovation, dwelling and event "
    "associations refer to the linked OtherClaim, not the filing "
    "claimant.",
    "other_home_accessibility_credit_claimed": "OtherClaim lane: This expense was claimed for the "
    "home accessibility tax credit. The claimant, "
    "renovation, dwelling and event associations refer to "
    "the linked OtherClaim, not the filing claimant.",
    "other_paid_same_day_before_completion": "OtherClaim lane: For a payment on the completion date, "
    "the actual payment occurred before completion; date "
    "equality alone does not establish this fact. The "
    "claimant, renovation, dwelling and event associations "
    "refer to the linked OtherClaim, not the filing "
    "claimant.",
    "other_incurred_same_day_before_completion": "OtherClaim lane: For incurrence on the completion "
    "date, the actual incurrence occurred before "
    "completion; date equality alone does not establish "
    "this fact. The claimant, renovation, dwelling and "
    "event associations refer to the linked OtherClaim, "
    "not the filing claimant.",
    "other_expense_is_building": "OtherClaim lane: This receipt component concerns the housing unit "
    "itself rather than land; mixed expenses must be split into "
    "nonduplicated truthful components. The claimant, renovation, "
    "dwelling and event associations refer to the linked OtherClaim, not "
    "the filing claimant.",
    "other_expense_is_land": "OtherClaim lane: This receipt component concerns the identified land "
    "extent; building and land classifications must be coherent. The "
    "claimant, renovation, dwelling and event associations refer to the "
    "linked OtherClaim, not the filing claimant.",
    "other_expense_land_scope_key": "OtherClaim lane: Stable factual key for the land extent to which "
    "this receipt component relates, compared with the parent claim’s "
    "land extent. The claimant, renovation, dwelling and event "
    "associations refer to the linked OtherClaim, not the filing "
    "claimant.",
    "other_paid_date": "OtherClaim lane: Date of the coherent actual payment event; inactive "
    "placeholder dates cannot establish a missing event. The claimant, renovation, "
    "dwelling and event associations refer to the linked OtherClaim, not the "
    "filing claimant.",
    "other_incurred_date": "OtherClaim lane: Date of the coherent actual incurrence event; inactive "
    "placeholder dates cannot establish a missing event. The claimant, "
    "renovation, dwelling and event associations refer to the linked "
    "OtherClaim, not the filing claimant.",
    "other_amount_including_taxes": "OtherClaim lane: Nonnegative CAD expense amount including "
    "applicable taxes for this receipt component, counted once. The "
    "claimant, renovation, dwelling and event associations refer to "
    "the linked OtherClaim, not the filing claimant.",
    "other_reimbursement_received_or_receivable": "OtherClaim lane: Nonnegative CAD reimbursement or "
    "assistance received or receivable attributable to "
    "this receipt component, including related GST/HST "
    "rebates; do not duplicate assistance across "
    "receipts. The claimant, renovation, dwelling and "
    "event associations refer to the linked OtherClaim, "
    "not the filing claimant.",
    "selection_individual_key": "Stable individual key for this truthful lifetime selection record; "
    "one coherent record per attributed individual across the supplied "
    "graph.",
    "selected_renovation_key": "Physical renovation actually selected for this individual’s lifetime "
    "qualifying-renovation claim; selection is supplied, never inferred "
    "from record order.",
    "filing_claimant_key": "Stable key of the single tax filer represented by the public Family root; "
    "Family does not pool family members’ entitlements.",
    "filing_paper_return": "This completed Schedule 12 is being filed with a paper return.",
}

SOURCE_ROLE_SPANS = {
    "edition": (0, 65),
    "residence": (1426, 1504),
    "claimant": (785, 1356),
    "qi": (4100, 4309),
    "qr": (4311, 4639),
    "dwelling": (3019, 3502),
    "land": (3503, 3630),
    "renovation": (4641, 5062),
    "secondary": (5702, 5998),
    "period": (5064, 5446),
    "year": (5448, 5700),
    "expense": (3632, 3998),
    "cutoff": (3930, 3998),
    "overlap": (2793, 2914),
    "assistance": (2512, 2792),
    "lifetime": (1505, 1597),
    "shared": (1598, 1777),
    "allocation": (1778, 1914),
    "multiqi": (1915, 2107),
    "line1": (6594, 6709),
    "line2": (6742, 6893),
    "line3": (6894, 6958),
    "line5": (7001, 7072),
    "line6": (7073, 7092),
    "line7": (7093, 7147),
    "cap": (2230, 2355),
    "rate": (2356, 2403),
    "maxcredit": (7598, 7640),
    "line8": (7316, 7375),
    "line10": (7435, 7530),
    "sheets": (6256, 6418),
    "paper": (351, 403),
    "documents": (404, 504),
}

RULE_SOURCES = {
    "claim_records": "Add the amounts from line 7 for each qualifying renovation.",
    "receipts": "Total amount paid from all sales slips and contracts included\n"
    "in the table above and any additional sheets of paper",
    "candidate_other_claims": "Amount of eligible expenses being claimed by other eligible "
    "individuals",
    "prior_renovations": "Only one qualifying renovation can be claimed for a qualifying individual "
    "in their lifetime.",
    "general_land_limit_hectares": "The amount of land that can be considered part of the eligible "
    "dwelling is generally limited to half of a hectare (1.24 acres).",
    "record_land_scope_key": "The amount of land that can be considered part of the eligible dwelling "
    "is generally limited to half of a hectare (1.24 acres).",
    "land_extent_eligible": "The amount of land that can be considered part of the eligible dwelling "
    "is generally limited to half of a hectare (1.24 acres).",
    "actual_allocation_applies": "If the individuals cannot agree on what portion of\n"
    "the amounts each can claim, the Minister of National Revenue may "
    "decide the portions.",
    "record_renovation_key": "When more than one eligible individual makes an MHRTC claim for the "
    "same qualifying renovation, the total of all\n"
    "amounts claimed for that renovation must not be more than $50,000.",
    "record_claimant_key": "Amount of eligible expenses being claimed by other eligible individuals",
    "record_individual_key": "Only one qualifying renovation can be claimed for a qualifying "
    "individual in their lifetime.",
    "record_completion_date": "Renovation period\n"
    "A period of time that:\n"
    "• begins at the time that the first qualifying expenditure is made or "
    "incurred for a qualifying renovation (for example, when\n"
    "you apply for a building permit)\n"
    "• ends at the time of the completion of the qualifying renovation (for "
    "example, when the qualifying renovation passes the\n"
    "final inspection or proof of project completion is received)",
    "qualifying_individual": "Qualifying individual\n"
    "Someone who, before the end of the renovation period tax year, is one "
    "of the following:\n"
    "• 65 years of age or older\n"
    "• 18 years of age or older who is eligible for the disability tax "
    "credit",
    "qualifying_relation": "Qualifying relation\n"
    "Someone who is :\n"
    "• 18 years of age or older before the end of the renovation period tax "
    "year\n"
    "• a parent, grandparent, child, grandchild, brother, sister, aunt, uncle, "
    "niece, or nephew of the qualifying individual or their\n"
    "cohabiting spouse or common-law partner, at any time in the renovation "
    "period tax year",
    "eligible_claimant": "You must be a resident of Canada throughout the tax year to claim this "
    "credit.",
    "eligible_dwelling": "Eligible dwelling\n"
    "A housing unit in Canada that:\n"
    "• both the qualifying individual and a qualifying relation of the "
    "individual ordinarily reside in, or intend to ordinarily reside in,\n"
    "within 12 months of the end of the renovation period\n"
    "• is owned (or jointly owned) by the qualifying individual or a qualifying "
    "relation of that individual (or a trust if the qualifying\n"
    "individual or qualifying relation is a beneficiary of the trust) at any "
    "time during the renovation period tax year",
    "secondary_unit": "Secondary unit\n"
    "A self-contained housing unit that:\n"
    "• has a private entrance, kitchen, bathroom facilities, and sleeping area\n"
    "• meets applicable local requirements to qualify as a secondary dwelling unit, "
    "if any\n"
    "The secondary unit can be newly constructed or created from an existing living "
    "space.",
    "qualifying_renovation": "Qualifying renovation\n"
    "A renovation or alteration of, or addition to, the qualifying "
    "individual's eligible dwelling that:\n"
    "• is of an enduring nature and integral to the eligible dwelling\n"
    "• allows the qualifying individual to reside in the dwelling with a "
    "qualifying relation of the qualifying individual by establishing\n"
    "a secondary unit within the dwelling for occupancy by the qualifying "
    "individual or qualifying relation",
    "completion_in_tax_year": "Renovation period tax year\n"
    "The tax year that includes the end of the renovation period of the "
    "qualifying renovation. For example, if the renovation period\n"
    "ends on May 3, 2026, the renovation period tax year is 2026 even if "
    "the renovation began in 2025.",
    "record_historical_individual_key": "Only one qualifying renovation can be claimed for a "
    "qualifying individual in their lifetime.",
    "record_historical_renovation_key": "Only one qualifying renovation can be claimed for a "
    "qualifying individual in their lifetime.",
    "different_prior_renovations": "Only one qualifying renovation can be claimed for a qualifying "
    "individual in their lifetime.",
    "historical_record_count": "Only one qualifying renovation can be claimed for a qualifying "
    "individual in their lifetime.",
    "prior_distinct_renovation_count": "Only one qualifying renovation can be claimed for a "
    "qualifying individual in their lifetime.",
    "claim_source_eligibility": "Qualifying individual\n"
    "Someone who, before the end of the renovation period tax year, is "
    "one of the following:\n"
    "• 65 years of age or older\n"
    "• 18 years of age or older who is eligible for the disability tax "
    "credit",
    "record_expense_land_scope_key": "The amount of land that can be considered part of the eligible "
    "dwelling is generally limited to half of a hectare (1.24 "
    "acres).",
    "record_paid_date": "Qualifying expenditure\n"
    "An expense that is:\n"
    "• a reasonable expense that was directly attributable to a qualifying "
    "renovation of an eligible dwelling\n"
    "• paid or incurred before the end of the renovation period by an eligible "
    "individual (or a trust that the eligible individual is\n"
    "a beneficiary)\n"
    "Note: The expense must have been paid or incurred after December 31, 2022.",
    "record_incurred_date": "Qualifying expenditure\n"
    "An expense that is:\n"
    "• a reasonable expense that was directly attributable to a qualifying "
    "renovation of an eligible dwelling\n"
    "• paid or incurred before the end of the renovation period by an "
    "eligible individual (or a trust that the eligible individual is\n"
    "a beneficiary)\n"
    "Note: The expense must have been paid or incurred after December 31, "
    "2022.",
    "expense_start_cutoff": "The expense must have been paid or incurred after December 31, 2022.",
    "paid_after_source_cutoff": "The expense must have been paid or incurred after December 31, 2022.",
    "incurred_after_source_cutoff": "The expense must have been paid or incurred after December 31, "
    "2022.",
    "receipt_basic_conditions": "Qualifying expenditure\n"
    "An expense that is:\n"
    "• a reasonable expense that was directly attributable to a "
    "qualifying renovation of an eligible dwelling\n"
    "• paid or incurred before the end of the renovation period by an "
    "eligible individual (or a trust that the eligible individual is\n"
    "a beneficiary)\n"
    "Note: The expense must have been paid or incurred after December 31, "
    "2022.",
    "qualifying_receipts": "Qualifying expenditure\n"
    "An expense that is:\n"
    "• a reasonable expense that was directly attributable to a qualifying "
    "renovation of an eligible dwelling\n"
    "• paid or incurred before the end of the renovation period by an eligible "
    "individual (or a trust that the eligible individual is\n"
    "a beneficiary)\n"
    "Note: The expense must have been paid or incurred after December 31, "
    "2022.",
    "line_one": "Total amount paid from all sales slips and contracts included\n"
    "in the table above and any additional sheets of paper",
    "line_two": "Any amount that can reasonably be considered to be reimbursed (including government\n"
    "assistance received or receivable) included in the amount on line 1",
    "line_three": "Line 1 minus line 2\n\n(maximum $50,000 per qualifying renovation)",
    "allocated_local_capacity": "Line 1 minus line 2\n\n(maximum $50,000 per qualifying renovation)",
    "other_receipts": "Total amount paid from all sales slips and contracts included\n"
    "in the table above and any additional sheets of paper",
    "other_prior_renovations": "Only one qualifying renovation can be claimed for a qualifying "
    "individual in their lifetime.",
    "other_general_land_limit_hectares": "The amount of land that can be considered part of the "
    "eligible dwelling is generally limited to half of a hectare "
    "(1.24 acres).",
    "other_record_land_scope_key": "The amount of land that can be considered part of the eligible "
    "dwelling is generally limited to half of a hectare (1.24 acres).",
    "other_land_extent_eligible": "The amount of land that can be considered part of the eligible "
    "dwelling is generally limited to half of a hectare (1.24 acres).",
    "other_actual_allocation_applies": "If the individuals cannot agree on what portion of\n"
    "the amounts each can claim, the Minister of National Revenue "
    "may decide the portions.",
    "other_record_renovation_key": "When more than one eligible individual makes an MHRTC claim for "
    "the same qualifying renovation, the total of all\n"
    "amounts claimed for that renovation must not be more than "
    "$50,000.",
    "other_record_claimant_key": "Amount of eligible expenses being claimed by other eligible "
    "individuals",
    "other_record_individual_key": "Only one qualifying renovation can be claimed for a qualifying "
    "individual in their lifetime.",
    "other_record_completion_date": "Renovation period\n"
    "A period of time that:\n"
    "• begins at the time that the first qualifying expenditure is "
    "made or incurred for a qualifying renovation (for example, when\n"
    "you apply for a building permit)\n"
    "• ends at the time of the completion of the qualifying "
    "renovation (for example, when the qualifying renovation passes "
    "the\n"
    "final inspection or proof of project completion is received)",
    "other_record_actual_claimed_expenses": "Amount of eligible expenses being claimed by other "
    "eligible individuals",
    "other_qualifying_individual": "Qualifying individual\n"
    "Someone who, before the end of the renovation period tax year, is "
    "one of the following:\n"
    "• 65 years of age or older\n"
    "• 18 years of age or older who is eligible for the disability tax "
    "credit",
    "other_qualifying_relation": "Qualifying relation\n"
    "Someone who is :\n"
    "• 18 years of age or older before the end of the renovation period "
    "tax year\n"
    "• a parent, grandparent, child, grandchild, brother, sister, aunt, "
    "uncle, niece, or nephew of the qualifying individual or their\n"
    "cohabiting spouse or common-law partner, at any time in the "
    "renovation period tax year",
    "other_eligible_claimant": "You must be a resident of Canada throughout the tax year to claim "
    "this credit.",
    "other_eligible_dwelling": "Eligible dwelling\n"
    "A housing unit in Canada that:\n"
    "• both the qualifying individual and a qualifying relation of the "
    "individual ordinarily reside in, or intend to ordinarily reside in,\n"
    "within 12 months of the end of the renovation period\n"
    "• is owned (or jointly owned) by the qualifying individual or a "
    "qualifying relation of that individual (or a trust if the qualifying\n"
    "individual or qualifying relation is a beneficiary of the trust) at "
    "any time during the renovation period tax year",
    "other_secondary_unit": "Secondary unit\n"
    "A self-contained housing unit that:\n"
    "• has a private entrance, kitchen, bathroom facilities, and sleeping "
    "area\n"
    "• meets applicable local requirements to qualify as a secondary dwelling "
    "unit, if any\n"
    "The secondary unit can be newly constructed or created from an existing "
    "living space.",
    "other_qualifying_renovation": "Qualifying renovation\n"
    "A renovation or alteration of, or addition to, the qualifying "
    "individual's eligible dwelling that:\n"
    "• is of an enduring nature and integral to the eligible dwelling\n"
    "• allows the qualifying individual to reside in the dwelling with "
    "a qualifying relation of the qualifying individual by "
    "establishing\n"
    "a secondary unit within the dwelling for occupancy by the "
    "qualifying individual or qualifying relation",
    "other_completion_in_tax_year": "Renovation period tax year\n"
    "The tax year that includes the end of the renovation period of "
    "the qualifying renovation. For example, if the renovation "
    "period\n"
    "ends on May 3, 2026, the renovation period tax year is 2026 even "
    "if the renovation began in 2025.",
    "other_record_historical_individual_key": "Only one qualifying renovation can be claimed for a "
    "qualifying individual in their lifetime.",
    "other_record_historical_renovation_key": "Only one qualifying renovation can be claimed for a "
    "qualifying individual in their lifetime.",
    "other_different_prior_renovations": "Only one qualifying renovation can be claimed for a "
    "qualifying individual in their lifetime.",
    "other_historical_record_count": "Only one qualifying renovation can be claimed for a qualifying "
    "individual in their lifetime.",
    "other_prior_distinct_renovation_count": "Only one qualifying renovation can be claimed for a "
    "qualifying individual in their lifetime.",
    "other_claim_source_eligibility": "Qualifying individual\n"
    "Someone who, before the end of the renovation period tax year, "
    "is one of the following:\n"
    "• 65 years of age or older\n"
    "• 18 years of age or older who is eligible for the disability "
    "tax credit",
    "other_record_expense_land_scope_key": "The amount of land that can be considered part of the "
    "eligible dwelling is generally limited to half of a "
    "hectare (1.24 acres).",
    "other_record_paid_date": "Qualifying expenditure\n"
    "An expense that is:\n"
    "• a reasonable expense that was directly attributable to a qualifying "
    "renovation of an eligible dwelling\n"
    "• paid or incurred before the end of the renovation period by an "
    "eligible individual (or a trust that the eligible individual is\n"
    "a beneficiary)\n"
    "Note: The expense must have been paid or incurred after December 31, "
    "2022.",
    "other_record_incurred_date": "Qualifying expenditure\n"
    "An expense that is:\n"
    "• a reasonable expense that was directly attributable to a "
    "qualifying renovation of an eligible dwelling\n"
    "• paid or incurred before the end of the renovation period by an "
    "eligible individual (or a trust that the eligible individual is\n"
    "a beneficiary)\n"
    "Note: The expense must have been paid or incurred after December "
    "31, 2022.",
    "other_expense_start_cutoff": "The expense must have been paid or incurred after December 31, "
    "2022.",
    "other_paid_after_source_cutoff": "The expense must have been paid or incurred after December 31, "
    "2022.",
    "other_incurred_after_source_cutoff": "The expense must have been paid or incurred after December "
    "31, 2022.",
    "other_receipt_basic_conditions": "Qualifying expenditure\n"
    "An expense that is:\n"
    "• a reasonable expense that was directly attributable to a "
    "qualifying renovation of an eligible dwelling\n"
    "• paid or incurred before the end of the renovation period by "
    "an eligible individual (or a trust that the eligible "
    "individual is\n"
    "a beneficiary)\n"
    "Note: The expense must have been paid or incurred after "
    "December 31, 2022.",
    "other_qualifying_receipts": "Qualifying expenditure\n"
    "An expense that is:\n"
    "• a reasonable expense that was directly attributable to a "
    "qualifying renovation of an eligible dwelling\n"
    "• paid or incurred before the end of the renovation period by an "
    "eligible individual (or a trust that the eligible individual is\n"
    "a beneficiary)\n"
    "Note: The expense must have been paid or incurred after December "
    "31, 2022.",
    "other_line_one": "Total amount paid from all sales slips and contracts included\n"
    "in the table above and any additional sheets of paper",
    "other_line_two": "Any amount that can reasonably be considered to be reimbursed (including "
    "government\n"
    "assistance received or receivable) included in the amount on line 1",
    "other_line_three": "Line 1 minus line 2\n\n(maximum $50,000 per qualifying renovation)",
    "other_allocated_local_capacity": "Line 1 minus line 2\n"
    "\n"
    "(maximum $50,000 per qualifying renovation)",
    "selection_record_individual_key": "Only one qualifying renovation can be claimed for a "
    "qualifying individual in their lifetime.",
    "selection_record_renovation_key": "Only one qualifying renovation can be claimed for a "
    "qualifying individual in their lifetime.",
    "selection_record_count": "Only one qualifying renovation can be claimed for a qualifying "
    "individual in their lifetime.",
    "claim_selection_records": "Only one qualifying renovation can be claimed for a qualifying "
    "individual in their lifetime.",
    "selected_matching_records": "Only one qualifying renovation can be claimed for a qualifying "
    "individual in their lifetime.",
    "selected_for_individual": "Only one qualifying renovation can be claimed for a qualifying "
    "individual in their lifetime.",
    "other_claim_selection_records": "Only one qualifying renovation can be claimed for a qualifying "
    "individual in their lifetime.",
    "other_selected_matching_records": "Only one qualifying renovation can be claimed for a "
    "qualifying individual in their lifetime.",
    "other_selected_for_individual": "Only one qualifying renovation can be claimed for a qualifying "
    "individual in their lifetime.",
    "filing_claimant_identity": "An eligible individual is either of the following:\n"
    "• an individual who ordinarily resides, or intends to ordinarily "
    "reside, in an eligible dwelling within 12 months of the end of the\n"
    "renovation period, and is any of the following:\n"
    "– a qualifying individual or their cohabiting spouse or common-law "
    "partner at any time in the renovation period tax year\n"
    "– a qualifying relation of a qualifying individual\n"
    "• an individual who is a qualifying relation of a qualifying "
    "individual, who owns the eligible dwelling or is the beneficiary of "
    "a\n"
    "trust that owns the eligible dwelling",
    "owned_claim_records": "An eligible individual is either of the following:\n"
    "• an individual who ordinarily resides, or intends to ordinarily reside, "
    "in an eligible dwelling within 12 months of the end of the\n"
    "renovation period, and is any of the following:\n"
    "– a qualifying individual or their cohabiting spouse or common-law "
    "partner at any time in the renovation period tax year\n"
    "– a qualifying relation of a qualifying individual\n"
    "• an individual who is a qualifying relation of a qualifying individual, "
    "who owns the eligible dwelling or is the beneficiary of a\n"
    "trust that owns the eligible dwelling",
    "selected_eligible_other_claims": "When more than one eligible individual makes an MHRTC claim "
    "for the same qualifying renovation, the total of all\n"
    "amounts claimed for that renovation must not be more than "
    "$50,000.",
    "selected_other_claim_amount": "Amount of eligible expenses being claimed by other eligible "
    "individuals",
    "selected_claim_capacity": "Line 1 minus line 2\n\n(maximum $50,000 per qualifying renovation)",
    "selected_qualifying_expenditures": "Enter whichever is less:\namount from line 3 or line 6.",
    "selected_total_qualifying_expenditures": "Add the amounts from line 7 for each qualifying "
    "renovation.",
    "selected_renovation_sheet_count": "If you paid qualifying expenditures for more than one "
    "qualifying renovation, complete this step for each qualifying "
    "renovation\n"
    "using an additional sheet of paper.",
    "additional_renovation_sheet_required": "If you paid qualifying expenditures for more than one "
    "qualifying renovation, complete this step for each "
    "qualifying renovation\n"
    "using an additional sheet of paper.",
    "attach_schedule_to_return": "Attach a copy of this schedule to your paper return.",
    "retain_supporting_documents": "Do not include your supporting documents, but keep them in case "
    "you are\n"
    "asked to provide them later.",
    "include_supporting_documents_with_return": "Do not include your supporting documents, but keep "
    "them in case you are\n"
    "asked to provide them later.",
    "federal_2025_mhrtc_qualifying_expenditure_limit_per_renovation": "An eligible individual can "
    "claim up to $50,000 in "
    "qualifying expenditures for a "
    "qualifying renovation ending "
    "in the tax year.",
    "federal_2025_mhrtc_rate": "The MHRTC is 14.5% of whichever amount is less:",
    "federal_2025_mhrtc_maximum_credit_per_renovation": "(maximum $7,250 per qualifying renovation)",
    "federal_2025_mhrtc_total_qualifying_expenditures_line_45354": "Add the amounts from line 7 for "
    "each qualifying renovation.",
    "multigenerational_home_renovation_tax_credit_line_45355": "Line 8 multiplied by the percentage "
    "from line 9\n"
    "Enter this amount on line 45355 of "
    "your return.",
}

RULE_PROOF_PATHS = {
    "claim_records": [("data_relation", "definition", "line8")],
    "receipts": [
        ("data_relation", "definition", "line1"),
        ("data_relation", "definition", "expense"),
    ],
    "candidate_other_claims": [
        ("data_relation", "definition", "line5"),
        ("data_relation", "definition", "shared"),
    ],
    "prior_renovations": [("data_relation", "definition", "lifetime")],
    "general_land_limit_hectares": [
        ("versions[0].values", "parameter", "land"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "record_land_scope_key": [
        ("versions[0].formula", "definition", "land"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "land_extent_eligible": [
        ("versions[0].formula", "formula", "land"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "actual_allocation_applies": [
        ("versions[0].formula", "formula", "allocation"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "record_renovation_key": [
        ("versions[0].formula", "definition", "shared"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "record_claimant_key": [
        ("versions[0].formula", "definition", "line5"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "record_individual_key": [
        ("versions[0].formula", "definition", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "record_completion_date": [
        ("versions[0].formula", "definition", "period"),
        ("versions[0].formula", "definition", "year"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "qualifying_individual": [
        ("versions[0].formula", "formula", "qi"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "qualifying_relation": [
        ("versions[0].formula", "formula", "qr"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "eligible_claimant": [
        ("versions[0].formula", "formula", "residence"),
        ("versions[0].formula", "formula", "claimant"),
        ("versions[0].formula", "formula", "qr"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "eligible_dwelling": [
        ("versions[0].formula", "formula", "dwelling"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "secondary_unit": [
        ("versions[0].formula", "formula", "secondary"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "qualifying_renovation": [
        ("versions[0].formula", "formula", "renovation"),
        ("versions[0].formula", "formula", "secondary"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "completion_in_tax_year": [
        ("versions[0].formula", "formula", "year"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "record_historical_individual_key": [
        ("versions[0].formula", "definition", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "record_historical_renovation_key": [
        ("versions[0].formula", "definition", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "different_prior_renovations": [
        ("versions[0].formula", "predicate", "lifetime"),
        ("derived_relation", "definition", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "historical_record_count": [
        ("versions[0].formula", "definition", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "prior_distinct_renovation_count": [
        ("versions[0].formula", "definition", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "claim_source_eligibility": [
        ("versions[0].formula", "formula", "qi"),
        ("versions[0].formula", "formula", "qr"),
        ("versions[0].formula", "formula", "residence"),
        ("versions[0].formula", "formula", "claimant"),
        ("versions[0].formula", "formula", "dwelling"),
        ("versions[0].formula", "formula", "renovation"),
        ("versions[0].formula", "formula", "secondary"),
        ("versions[0].formula", "formula", "year"),
        ("versions[0].formula", "formula", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "record_expense_land_scope_key": [
        ("versions[0].formula", "definition", "land"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "record_paid_date": [
        ("versions[0].formula", "definition", "expense"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "record_incurred_date": [
        ("versions[0].formula", "definition", "expense"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "expense_start_cutoff": [
        ("versions[0].values", "parameter", "cutoff"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "paid_after_source_cutoff": [
        ("versions[0].formula", "formula", "cutoff"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "incurred_after_source_cutoff": [
        ("versions[0].formula", "formula", "cutoff"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "receipt_basic_conditions": [
        ("versions[0].formula", "formula", "expense"),
        ("versions[0].formula", "formula", "overlap"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "qualifying_receipts": [
        ("versions[0].formula", "predicate", "expense"),
        ("versions[0].formula", "predicate", "cutoff"),
        ("versions[0].formula", "predicate", "period"),
        ("versions[0].formula", "predicate", "overlap"),
        ("versions[0].formula", "predicate", "land"),
        ("derived_relation", "definition", "expense"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "line_one": [
        ("versions[0].formula", "formula", "line1"),
        ("versions[0].formula", "formula", "expense"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "line_two": [
        ("versions[0].formula", "formula", "line2"),
        ("versions[0].formula", "formula", "assistance"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "line_three": [
        ("versions[0].formula", "formula", "line3"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "allocated_local_capacity": [
        ("versions[0].formula", "formula", "line3"),
        ("versions[0].formula", "formula", "allocation"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_receipts": [
        ("data_relation", "definition", "line1"),
        ("data_relation", "definition", "expense"),
    ],
    "other_prior_renovations": [("data_relation", "definition", "lifetime")],
    "other_general_land_limit_hectares": [
        ("versions[0].values", "parameter", "land"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_record_land_scope_key": [
        ("versions[0].formula", "definition", "land"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_land_extent_eligible": [
        ("versions[0].formula", "formula", "land"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_actual_allocation_applies": [
        ("versions[0].formula", "formula", "allocation"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_record_renovation_key": [
        ("versions[0].formula", "definition", "shared"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_record_claimant_key": [
        ("versions[0].formula", "definition", "line5"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_record_individual_key": [
        ("versions[0].formula", "definition", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_record_completion_date": [
        ("versions[0].formula", "definition", "period"),
        ("versions[0].formula", "definition", "year"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_record_actual_claimed_expenses": [
        ("versions[0].formula", "definition", "line5"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_qualifying_individual": [
        ("versions[0].formula", "formula", "qi"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_qualifying_relation": [
        ("versions[0].formula", "formula", "qr"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_eligible_claimant": [
        ("versions[0].formula", "formula", "residence"),
        ("versions[0].formula", "formula", "claimant"),
        ("versions[0].formula", "formula", "qr"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_eligible_dwelling": [
        ("versions[0].formula", "formula", "dwelling"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_secondary_unit": [
        ("versions[0].formula", "formula", "secondary"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_qualifying_renovation": [
        ("versions[0].formula", "formula", "renovation"),
        ("versions[0].formula", "formula", "secondary"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_completion_in_tax_year": [
        ("versions[0].formula", "formula", "year"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_record_historical_individual_key": [
        ("versions[0].formula", "definition", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_record_historical_renovation_key": [
        ("versions[0].formula", "definition", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_different_prior_renovations": [
        ("versions[0].formula", "predicate", "lifetime"),
        ("derived_relation", "definition", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_historical_record_count": [
        ("versions[0].formula", "definition", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_prior_distinct_renovation_count": [
        ("versions[0].formula", "definition", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_claim_source_eligibility": [
        ("versions[0].formula", "formula", "qi"),
        ("versions[0].formula", "formula", "qr"),
        ("versions[0].formula", "formula", "residence"),
        ("versions[0].formula", "formula", "claimant"),
        ("versions[0].formula", "formula", "dwelling"),
        ("versions[0].formula", "formula", "renovation"),
        ("versions[0].formula", "formula", "secondary"),
        ("versions[0].formula", "formula", "year"),
        ("versions[0].formula", "formula", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_record_expense_land_scope_key": [
        ("versions[0].formula", "definition", "land"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_record_paid_date": [
        ("versions[0].formula", "definition", "expense"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_record_incurred_date": [
        ("versions[0].formula", "definition", "expense"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_expense_start_cutoff": [
        ("versions[0].values", "parameter", "cutoff"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_paid_after_source_cutoff": [
        ("versions[0].formula", "formula", "cutoff"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_incurred_after_source_cutoff": [
        ("versions[0].formula", "formula", "cutoff"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_receipt_basic_conditions": [
        ("versions[0].formula", "formula", "expense"),
        ("versions[0].formula", "formula", "overlap"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_qualifying_receipts": [
        ("versions[0].formula", "predicate", "expense"),
        ("versions[0].formula", "predicate", "cutoff"),
        ("versions[0].formula", "predicate", "period"),
        ("versions[0].formula", "predicate", "overlap"),
        ("versions[0].formula", "predicate", "land"),
        ("derived_relation", "definition", "expense"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_line_one": [
        ("versions[0].formula", "formula", "line1"),
        ("versions[0].formula", "formula", "expense"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_line_two": [
        ("versions[0].formula", "formula", "line2"),
        ("versions[0].formula", "formula", "assistance"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_line_three": [
        ("versions[0].formula", "formula", "line3"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_allocated_local_capacity": [
        ("versions[0].formula", "formula", "line3"),
        ("versions[0].formula", "formula", "allocation"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "selection_record_individual_key": [
        ("versions[0].formula", "definition", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "selection_record_renovation_key": [
        ("versions[0].formula", "definition", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "selection_record_count": [
        ("versions[0].formula", "definition", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "claim_selection_records": [("data_relation", "definition", "lifetime")],
    "selected_matching_records": [
        ("versions[0].formula", "predicate", "lifetime"),
        ("derived_relation", "definition", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "selected_for_individual": [
        ("versions[0].formula", "formula", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_claim_selection_records": [("data_relation", "definition", "lifetime")],
    "other_selected_matching_records": [
        ("versions[0].formula", "predicate", "lifetime"),
        ("derived_relation", "definition", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "other_selected_for_individual": [
        ("versions[0].formula", "formula", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "filing_claimant_identity": [
        ("versions[0].formula", "definition", "claimant"),
        ("versions[0].formula", "definition", "line8"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "owned_claim_records": [
        ("versions[0].formula", "predicate", "claimant"),
        ("versions[0].formula", "predicate", "line8"),
        ("derived_relation", "definition", "claimant"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "selected_eligible_other_claims": [
        ("versions[0].formula", "predicate", "shared"),
        ("versions[0].formula", "predicate", "line5"),
        ("versions[0].formula", "predicate", "lifetime"),
        ("derived_relation", "definition", "shared"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "selected_other_claim_amount": [
        ("versions[0].formula", "formula", "line5"),
        ("versions[0].formula", "formula", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "selected_claim_capacity": [
        ("versions[0].formula", "formula", "line3"),
        ("versions[0].formula", "formula", "cap"),
        ("versions[0].formula", "formula", "line5"),
        ("versions[0].formula", "formula", "line6"),
        ("versions[0].formula", "formula", "line7"),
        ("versions[0].formula", "formula", "allocation"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "selected_qualifying_expenditures": [
        ("versions[0].formula", "formula", "line7"),
        ("versions[0].formula", "formula", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "selected_total_qualifying_expenditures": [
        ("versions[0].formula", "formula", "line8"),
        ("versions[0].formula", "formula", "lifetime"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "selected_renovation_sheet_count": [
        ("versions[0].formula", "formula", "sheets"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "additional_renovation_sheet_required": [
        ("versions[0].formula", "formula", "sheets"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "attach_schedule_to_return": [
        ("versions[0].formula", "formula", "paper"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "retain_supporting_documents": [
        ("versions[0].formula", "formula", "documents"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "include_supporting_documents_with_return": [
        ("versions[0].formula", "formula", "documents"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "federal_2025_mhrtc_qualifying_expenditure_limit_per_renovation": [
        ("versions[0].formula", "parameter", "cap"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "federal_2025_mhrtc_rate": [
        ("versions[0].formula", "parameter", "rate"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "federal_2025_mhrtc_maximum_credit_per_renovation": [
        ("versions[0].formula", "parameter", "maxcredit"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "federal_2025_mhrtc_total_qualifying_expenditures_line_45354": [
        ("versions[0].formula", "formula", "line8"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
    "multigenerational_home_renovation_tax_credit_line_45355": [
        ("versions[0].formula", "formula", "line10"),
        ("versions[0].formula", "formula", "rate"),
        ("versions[0].effective_from", "effective_period", "edition"),
        ("versions[0].effective_to", "effective_period", "edition"),
    ],
}

RULE_METADATA = {
    name: {
        "proof": {
            "atoms": [
                {
                    "path": path,
                    "kind": kind,
                    "source": {
                        "corpus_citation_path": CITATION,
                        "excerpt": SOURCE_TEXT[
                            SOURCE_ROLE_SPANS[role][0] : SOURCE_ROLE_SPANS[role][1]
                        ],
                    },
                }
                for path, kind, role in paths
            ]
        }
    }
    for name, paths in RULE_PROOF_PATHS.items()
}
