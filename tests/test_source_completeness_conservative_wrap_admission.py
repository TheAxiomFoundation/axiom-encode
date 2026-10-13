"""Unknown coordinated PDF clauses retain their existing factual protections."""

from __future__ import annotations

import itertools

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc

CASES = [
    {
        "id": "colorado-valuation-prerequisites",
        "source": "When such minimum requirement has not been\n"
        "met but typical sales within any such class or subclass indicate that "
        "valuations in\n"
        "the class or subclass are too high or too low, such fact shall be "
        "reported to the\n"
        "state board of equalization, which board may order an independent "
        "appraisal study\n"
        "in such county.",
        "inputs": {
            "minimum_requirement_not_met": "Such minimum requirement has not been met.",
            "sales_indicate_valuations_too_high_or_too_low": "Typical sales indicate "
            "that valuations in the "
            "class or subclass are "
            "too high or too low.",
        },
        "protected": [
            "minimum_requirement_not_met",
            "sales_indicate_valuations_too_high_or_too_low",
        ],
    },
    {
        "id": "colorado-heat-use-and-source-location",
        "source": "(III) The use of any heat extracted with produced fluids in an oil and "
        "gas\n"
        "operation if the heat is only utilized to reduce emissions from the "
        "operation in the\n"
        "same location as the well from which it was produced and would otherwise "
        "not be\n"
        "economically feasible as a stand-alone geothermal energy project;",
        "inputs": {
            "heat_used_at_source_well_location": "The heat is utilized in the same "
            "location as the well from which it "
            "was produced.",
            "heat_used_only_to_reduce_operating_emissions": "The heat is only "
            "utilized to reduce "
            "emissions from the "
            "operation.",
            "standalone_heat_project_economically_infeasible": "The heat use would "
            "otherwise not be "
            "economically feasible "
            "as a stand-alone "
            "geothermal energy "
            "project.",
        },
        "protected": [
            "heat_used_at_source_well_location",
            "heat_used_only_to_reduce_operating_emissions",
        ],
    },
    {
        "id": "colorado-reported-approval-criteria",
        "source": "The commission may only approve a new business for the rural jump-start\n"
        "zone program benefits if the commission is satisfied that the new "
        "business meets\n"
        "the definition of new business as specified in section 39-30.5-103 (7), "
        "that the new\n"
        "hires will meet the definition of new hire as specified in section "
        "39-30.5-103 (8), and\n"
        "that the new business will be located in the rural jump-start zone for "
        "which the\n"
        "state institution of higher education or economic development "
        "organization sought\n"
        "approval.",
        "inputs": {
            "commission_satisfied_new_business_definition_met": "The commission is "
            "satisfied that the "
            "new business meets "
            "the definition of "
            "new business.",
            "new_business_will_locate_in_approved_zone": "The new business will be "
            "located in the approved "
            "rural jump-start zone.",
            "new_hires_will_meet_new_hire_definition": "The new hires will meet the "
            "definition of new hire.",
        },
        "protected": None,
    },
    {
        "id": "louisiana-47:519-6220",
        "source": "No penalty shall be imposed on any person who violates the\n"
        "provisions of Subsections G and H of this Section by failing to dispose "
        "of a temporary\n"
        "registration plate and retaining such temporary registration plate after "
        "the expiration of sixty\n"
        "days from issuance thereof if the permanent registration plate has been "
        "applied for as\n"
        "required by law but has not yet been received by the owner.",
        "inputs": {
            "permanent_plate_not_yet_received_by_owner": "The permanent registration "
            "plate has not yet been "
            "received by the owner.",
            "permanent_registration_plate_lawfully_applied_for": "The permanent "
            "registration plate "
            "has been applied "
            "for as required by "
            "law.",
        },
        "protected": [
            "permanent_plate_not_yet_received_by_owner",
            "permanent_registration_plate_lawfully_applied_for",
        ],
    },
    {
        "id": "louisiana-47:1703-9015",
        "source": "A homeowner shall only\n"
        "be eligible for an additional extension if the owner has made a good "
        "faith attempt to secure\n"
        "a contractor or builder to complete the needed repairs or reconstruction "
        "of the home but is\n"
        "unable to complete the project due to uncontrollable contractor or "
        "builder delays.",
        "inputs": {
            "owner_good_faith_contractor_attempt": "The owner has made a good faith "
            "attempt to secure a contractor or "
            "builder for needed home repairs "
            "or reconstruction.",
            "owner_unable_complete_due_uncontrollable_delays": "The owner is unable "
            "to complete the "
            "project due to "
            "uncontrollable "
            "contractor or builder "
            "delays.",
        },
        "protected": [
            "owner_good_faith_contractor_attempt",
            "owner_unable_complete_due_uncontrollable_delays",
        ],
    },
    {
        "id": "louisiana-47:1703-5662",
        "source": "A homeowner shall only\n"
        "be eligible for an additional extension if the owner has made a good "
        "faith attempt to secure\n"
        "a contractor or builder to complete the needed repairs or reconstruction "
        "of the home but is\n"
        "unable to complete the project due to uncontrollable contractor or "
        "builder delays.",
        "inputs": {
            "owner_good_faith_contractor_attempt": "The owner has made a good faith "
            "attempt to secure a contractor or "
            "builder for needed home repairs "
            "or reconstruction.",
            "owner_unable_complete_due_uncontrollable_delays": "The owner is unable "
            "to complete the "
            "project due to "
            "uncontrollable "
            "contractor or builder "
            "delays.",
        },
        "protected": [
            "owner_good_faith_contractor_attempt",
            "owner_unable_complete_due_uncontrollable_delays",
        ],
    },
]


CASES.extend(
    [
        {
            "id": "inverse-human-qualifier",
            "source": "A credit applies if the taxpayer who owns the property\nis blind.",
            "inputs": {
                "taxpayer_owns_property": "The taxpayer owns the property.",
                "taxpayer_is_blind": "The taxpayer is blind.",
            },
            "protected": ["taxpayer_owns_property", "taxpayer_is_blind"],
        },
        {
            "id": "inverse-property-qualifier",
            "source": "A credit applies if the property that is a residence\nis owned by the taxpayer.",
            "inputs": {
                "property_is_residence": "The property is a residence.",
                "property_owned_by_taxpayer": "The property is owned by the taxpayer.",
            },
            "protected": ["property_is_residence", "property_owned_by_taxpayer"],
        },
        {
            "id": "if-first-ambiguous-comma-scope",
            "source": "If the taxpayer is resident,\nthe spouse is blind and disabled.",
            "inputs": {
                "taxpayer_is_resident": "The taxpayer is resident.",
                "spouse_is_blind": "The spouse is blind.",
            },
            "protected": ["taxpayer_is_resident", "spouse_is_blind"],
        },
        {
            "id": "partial-property-merge",
            "source": "A credit applies if the claimant is a resident and the spouse owns a house\nthat is blind-accessible.",
            "inputs": {
                "claimant_is_resident": "The claimant is a resident.",
                "spouse_owns_house": "The spouse owns a house.",
                "spouse_house_blind_accessible": "The spouse's house is blind-accessible.",
            },
            "protected": [
                "claimant_is_resident",
                "spouse_owns_house",
                "spouse_house_blind_accessible",
            ],
        },
        {
            "id": "known-human-and-unknown-property-merge",
            "source": "A credit applies if the claimant is a licensed person who has a permit and the spouse owns a house\nthat is blind-accessible.",
            "inputs": {
                "claimant_is_licensed_person": "The claimant is a licensed person.",
                "claimant_has_permit": "The claimant has a permit.",
                "spouse_owns_house": "The spouse owns a house.",
                "spouse_house_blind_accessible": "The spouse's house is blind-accessible.",
            },
            "protected": [
                "claimant_is_licensed_person",
                "claimant_has_permit",
                "spouse_owns_house",
                "spouse_house_blind_accessible",
            ],
        },
    ]
)

CASES.extend(
    [
        {
            "id": "nested-if-39-22-565",
            "source": "(I) If a building, structure, or facility is not a qualified asset as a result of a\ncasualty loss if the loss is restored by reconstruction or replacement within a\nreasonable period established by the office;",
            "inputs": {
                "asset_disqualified_by_casualty": "The building, structure, or facility is not a qualified asset because of a casualty loss.",
                "casualty_loss_restored": "The casualty loss is restored by reconstruction or replacement within a reasonable period established by the office.",
            },
            "protected": ["asset_disqualified_by_casualty", "casualty_loss_restored"],
        },
        {
            "id": "nested-if-39-22-569",
            "source": "(I) If a building, structure, or facility is not an eligible project as a result of a\ncasualty loss if the loss is restored by reconstruction or replacement within a\nreasonable period established by the office;",
            "inputs": {
                "casualty_loss_restored": "The casualty loss is restored by reconstruction or replacement within a reasonable period established by the office.",
                "project_disqualified_by_casualty": "The building, structure, or facility is not an eligible project because of a casualty loss.",
            },
            "protected": ["casualty_loss_restored", "project_disqualified_by_casualty"],
        },
        {
            "id": "nested-if-39-29-109",
            "source": "If the applicant is a covered entity,\nas defined in section 37-60-126, the board shall allocate money by grant or loan\nfrom the fund only if the applicant has adopted a water conservation plan, as\ndefined in section 37-60-126.",
            "inputs": {
                "covered_applicant": "The applicant is a covered entity.",
                "water_conservation_plan_adopted": "The applicant has adopted a water conservation plan.",
            },
            "protected": ["covered_applicant", "water_conservation_plan_adopted"],
        },
    ]
)

CASES.extend(
    [
        {
            "id": "positive-admission-39-2-127",
            "source": "If the board has issued\na summary decision, a party dissatisfied with the summary decision may file a\nwritten request with the board for a full decision.",
            "inputs": {
                "board_summary_decision_issued": "The board has issued a summary decision.",
                "party_dissatisfied_summary_decision": "The party is dissatisfied with the summary decision.",
            },
            "protected": [
                "board_summary_decision_issued",
                "party_dissatisfied_summary_decision",
            ],
        },
        {
            "id": "positive-admission-39-3.5-102",
            "source": "(3) If a guardian, conservator, or attorney-in-fact has been appointed for a\ntaxpayer otherwise qualified to claim deferral of taxes under this article, the\nguardian, conservator, or attorney-in-fact may act for such taxpayer in claiming the\ndeferral.",
            "inputs": {
                "representative_appointed": "A guardian, conservator, or attorney-in-fact has been appointed for the taxpayer.",
                "taxpayer_otherwise_qualified_to_claim_deferral": "The taxpayer is otherwise qualified to claim deferral of taxes.",
            },
            "protected": [
                "representative_appointed",
                "taxpayer_otherwise_qualified_to_claim_deferral",
            ],
        },
        {
            "id": "positive-admission-39-27-102.5",
            "source": "This exemption only applies if\nthe gasoline or special fuel purchased by the governmental entity is used in\nmachines owned or operated by the governmental entity.",
            "inputs": {
                "fuel_used_by_governmental_entity": "The gasoline or special fuel purchased by the governmental entity is used in machines.",
                "machines_owned_or_operated_by_governmental_entity": "The machines are owned or operated by the governmental entity.",
            },
            "protected": [
                "fuel_used_by_governmental_entity",
                "machines_owned_or_operated_by_governmental_entity",
            ],
        },
        {
            "id": "positive-admission-57-38-01.8-3059",
            "source": "If a taxpayer entitled to the credit provided by this section is a member of a group of\ncorporations filing a North Dakota consolidated tax return using the combined\nreporting method, the credit may be claimed against the aggregate North Dakota tax\nliability of all of the corporations included in the North Dakota consolidated return.",
            "inputs": {
                "taxpayer_entitled_to_credit": "The taxpayer is entitled to the credit provided by this section.",
                "taxpayer_member_consolidated_reporting_group": "The taxpayer is a member of a group of corporations filing a North Dakota consolidated tax return using the combined report method.",
            },
            "protected": [
                "taxpayer_entitled_to_credit",
                "taxpayer_member_consolidated_reporting_group",
            ],
        },
        {
            "id": "positive-admission-57-38-01.41-5651",
            "source": "If a taxpayer entitled to the credit provided by this section is a member of a group of\ncorporations filing a North Dakota consolidated tax return using the combined\nreporting method, the credit may be claimed against the aggregate North Dakota tax\nliability of all the corporations included in the North Dakota consolidated return.",
            "inputs": {
                "taxpayer_entitled_to_credit": "The taxpayer is entitled to the credit provided by this section.",
                "taxpayer_member_consolidated_reporting_group": "The taxpayer is a member of a group of corporations filing a North Dakota consolidated tax return using the combined report method.",
            },
            "protected": [
                "taxpayer_entitled_to_credit",
                "taxpayer_member_consolidated_reporting_group",
            ],
        },
        {
            "id": "positive-admission-57-38-30.5-12884",
            "source": "If a taxpayer entitled to the credit provided by this section is a member of a group of\ncorporations filing a North Dakota consolidated tax return using the combined\nreporting method, the credit may be claimed against the aggregate North Dakota tax\nliability of all the corporations included in the North Dakota consolidated return.",
            "inputs": {
                "taxpayer_entitled_to_credit": "The taxpayer is entitled to the credit provided by this section.",
                "taxpayer_member_consolidated_reporting_group": "The taxpayer is a member of a group of corporations filing a North Dakota consolidated tax return using the combined report method.",
            },
            "protected": [
                "taxpayer_entitled_to_credit",
                "taxpayer_member_consolidated_reporting_group",
            ],
        },
        {
            "id": "positive-admission-57-38-42-2752",
            "source": "Each information return required under subsection 1 must be deemed to be filed with\nthe tax commissioner if the person required to make the return files with the tax\ncommissioner a copy of the information return along with a copy of the transmittal form\nrequired to be filed with the internal revenue service.",
            "inputs": {
                "person_files_information_return_copy": "The person files with the tax commissioner a copy of the information return.",
                "person_files_transmittal_form_copy": "The person files with the tax commissioner a copy of the transmittal form required to be filed with the internal revenue service.",
            },
            "protected": [
                "person_files_information_return_copy",
                "person_files_transmittal_form_copy",
            ],
        },
        {
            "id": "positive-admission-57-38-42-7335",
            "source": "Each information return required under subsection 1 must be deemed to be filed with\nthe tax commissioner if the person required to make the return files with the tax\ncommissioner a copy of the information return along with a copy of the transmittal form\nrequired to be filed with the internal revenue service.",
            "inputs": {
                "person_files_information_return_copy": "The person files with the tax commissioner a copy of the information return.",
                "person_files_transmittal_form_copy": "The person files with the tax commissioner a copy of the transmittal form required to be filed with the internal revenue service.",
            },
            "protected": [
                "person_files_information_return_copy",
                "person_files_transmittal_form_copy",
            ],
        },
    ]
)

CASES.extend(
    [
        {
            "id": "recognized-human-and-unmarked-nominal-group",
            "source": "A credit applies if the claimant is a licensed person who has a permit and the taxpayer entitled to the credit provided by this section is a member of a group of\ncorporations filing a North Dakota consolidated tax return using the combined\nreporting method.",
            "inputs": {
                "claimant_is_licensed_person": "The claimant is a licensed person.",
                "claimant_has_permit": "The claimant has a permit.",
                "taxpayer_entitled_to_credit": "The taxpayer is entitled to the credit provided by this section.",
                "taxpayer_member_consolidated_reporting_group": "The taxpayer is a member of a group of corporations filing a North Dakota consolidated tax return using the combined report method.",
            },
            "protected": [
                "claimant_is_licensed_person",
                "claimant_has_permit",
                "taxpayer_entitled_to_credit",
                "taxpayer_member_consolidated_reporting_group",
            ],
        },
        {
            "id": "recognized-human-and-unmarked-document-pair",
            "source": "A credit applies if the claimant is a licensed person who has a permit and the person required to make the return files with the tax\ncommissioner a copy of the information return along with a copy of the transmittal form\nrequired to be filed with the internal revenue service.",
            "inputs": {
                "claimant_is_licensed_person": "The claimant is a licensed person.",
                "claimant_has_permit": "The claimant has a permit.",
                "person_files_information_return_copy": "The person files with the tax commissioner a copy of the information return.",
                "person_files_transmittal_form_copy": "The person files with the tax commissioner a copy of the transmittal form required to be filed with the internal revenue service.",
            },
            "protected": [
                "claimant_is_licensed_person",
                "claimant_has_permit",
                "person_files_information_return_copy",
                "person_files_transmittal_form_copy",
            ],
        },
    ]
)


def _evidence(case):
    return sc._terminal_gate_evidence(
        {
            "inputs": [
                {"name": name, "description": description}
                for name, description in case["inputs"].items()
            ]
        }
    )


def _accepted(case, gates, names):
    return sc._terminal_names_corroborate_source_gates(
        frozenset(names), gates, terminal_evidence=_evidence(case)
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_unsupported_coordination_retains_authentic_fact_requirements(case):
    gates = sc._source_conjunctive_fact_gates(case["source"])
    assert len(gates) >= 2
    names = frozenset(case["inputs"])
    assert _accepted(case, gates, names)
    assert not _accepted(case, gates, ())
    if case["protected"] is None:
        # The existing signatures overlap: any pair passes, but a single
        # criterion or no criteria cannot stand in for the reported conditions.
        for name in names:
            assert not _accepted(case, gates, (name,))
    else:
        for omitted in case["protected"]:
            assert not _accepted(case, gates, names - {omitted})


@settings(max_examples=30, deadline=None)
@given(
    index=st.integers(min_value=0, max_value=len(CASES) - 1),
    breaks=st.sets(st.integers(min_value=1, max_value=140), max_size=50),
)
def test_added_pdf_wraps_cannot_erase_the_original_conservative_scan(index, breaks):
    case = CASES[index]
    # Keep each authentic line boundary and add ordinary PDF wraps between
    # words. Unsupported layouts continue to use the original physical scan.
    lines = []
    count = 0
    for line in case["source"].splitlines():
        pieces = []
        for word in line.split():
            count += 1
            pieces.append(word + ("\n" if count in breaks else " "))
        lines.append("".join(pieces).strip())
    wrapped = "\n".join(lines)
    actual = sc._source_conjunctive_fact_gates(wrapped)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            sc, "_source_gate_join_wrapped_continuations", lambda text, **kwargs: text
        )
        physical = sc._source_conjunctive_fact_gates(wrapped)
    names = tuple(case["inputs"])
    for count in range(len(names) + 1):
        for subset in itertools.combinations(names, count):
            if not _accepted(case, physical, subset):
                assert not _accepted(case, actual, subset)


LEASE_ALTERNATIVE_CASES = (
    (
        "us-co/statute/39/39-26-117",
        "The real or personal property of an owner who has made a bona fide\n"
        "lease to a retailer shall be exempt from the lien created in paragraph (a) of this\n"
        "subsection (1) if such property can reasonably be identified from the lease\n"
        "description or if the lessee is given an option to purchase in such lease and has not\n"
        "exercised such option to become the owner of the property leased.",
    ),
    (
        "us-co/statute/39/39-26-205",
        "The real or personal property of an owner who has made a bona fide lease to any\n"
        "taxpayer described in subsection (1) of this section shall be exempt from the lien\n"
        "created therein if such property can reasonably be identified from the lease\n"
        "description or if the lessee is given an option to purchase in such lease and has not\n"
        "exercised such option to become the owner of the property leased.",
    ),
)
LEASE_INPUTS = {
    "lease_property_identifiable": "Such property can reasonably be identified from the lease description.",
    "lessee_given_purchase_option": "The lessee is given an option to purchase in such lease.",
    "lessee_purchase_option_unexercised": "The lessee has not exercised such option to become the owner of the property leased.",
}
LEASE_FORMULA = (
    "lease_property_identifiable or "
    "(lessee_given_purchase_option and lessee_purchase_option_unexercised)"
)


def _faithful_condition_issues(
    citation, source, formula, input_descriptions, *, rule_name
):
    rule = {
        "name": rule_name,
        "kind": "derived",
        "dtype": "Judgment",
        "source": citation,
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {
                            "corpus_citation_path": citation,
                            "excerpt": source,
                        },
                    }
                ]
            }
        },
        "versions": [{"formula": formula}],
    }
    inputs = [
        {"name": name, "dtype": "Boolean", "description": description}
        for name, description in input_descriptions.items()
    ]
    return sc._opaque_same_source_condition_input_issues(
        {"inputs": inputs, "rules": [rule]},
        source_text=source,
        branches=sc.recognize_source_structure(source, corpus_citation_path=citation),
        principal_rules={rule["name"]: rule},
        corpus_citation_path=citation,
    )


@pytest.mark.parametrize("citation,source", LEASE_ALTERNATIVE_CASES)
def test_wrapped_lease_alternatives_accept_the_complete_faithful_formula(
    citation, source
):
    # The complete formula has an active one-fact route: identifiable property
    # is exempt regardless of whether the purchase-option alternative holds.
    assert not _faithful_condition_issues(
        citation,
        source,
        LEASE_FORMULA,
        LEASE_INPUTS,
        rule_name="property_exempt_from_lien",
    )


def test_lease_identifiability_and_purchase_option_alternatives_execute_faithfully():
    for identified, option_given, unexercised in itertools.product(
        (False, True), repeat=3
    ):
        execution = sc._execute_formula_text(
            LEASE_FORMULA,
            environment={
                "lease_property_identifiable": identified,
                "lessee_given_purchase_option": option_given,
                "lessee_purchase_option_unexercised": unexercised,
            },
            constant_environment={},
        )
        assert execution is not None
        assert sc._formula_execution_runtime_value(execution) is (
            identified or (option_given and unexercised)
        )


@settings(max_examples=40, deadline=None)
@given(
    head=st.sampled_from(("if", "when", "unless")),
    breaks=st.sets(st.integers(min_value=1, max_value=60), max_size=12),
)
def test_wrapped_alternative_heads_preserve_the_sufficient_first_fact(head, breaks):
    body = LEASE_ALTERNATIVE_CASES[0][1].split(" if ", 1)[1]
    body = body.replace("or if", "or " + head, 1)
    lines = []
    count = 0
    for line in body.splitlines():
        pieces = []
        for word in line.split():
            count += 1
            pieces.append(word + ("\n" if count in breaks else " "))
        lines.append("".join(pieces).strip())
    wrapped = "\n".join(lines)
    physical = sc._source_gate_fact_signatures(
        sc._source_gate_split_conjunctive_conditions(wrapped, has_preposed_result=False)
    )
    assume(not physical)
    normalized = sc._source_gate_join_wrapped_continuations(
        wrapped, has_preposed_result=False
    )
    gates = sc._source_gate_fact_signatures(
        sc._source_gate_split_conjunctive_conditions(
            normalized, has_preposed_result=False
        )
    )
    evidence = sc._terminal_gate_evidence(
        {
            "inputs": [
                {"name": name, "description": description}
                for name, description in LEASE_INPUTS.items()
            ]
        }
    )
    assert sc._terminal_names_corroborate_source_gates(
        frozenset({"lease_property_identifiable"}), gates, terminal_evidence=evidence
    )


VERIFICATION_SOURCE = (
    "If the individual is unable to provide the prior two months verification but "
    "has provided verification that is representative of their ongoing income, "
    "accept their verification."
)
VERIFICATION_INPUTS = {
    "prior_two_months_verification_unavailable": "The individual is unable to provide the prior two months verification.",
    "representative_ongoing_verification_provided": "The individual has provided verification that is representative of their ongoing income.",
}
VERIFICATION_FORMULA = (
    "prior_two_months_verification_unavailable and "
    "representative_ongoing_verification_provided"
)


def test_verification_object_income_does_not_make_faithful_formula_uncorroborable():
    assert not _faithful_condition_issues(
        "us-ky/manual/dcbs/dfs/volume-ii-snap/page-221",
        VERIFICATION_SOURCE,
        VERIFICATION_FORMULA,
        VERIFICATION_INPUTS,
        rule_name="verification_accepted",
    )
    for unavailable, provided in itertools.product((False, True), repeat=2):
        execution = sc._execute_formula_text(
            VERIFICATION_FORMULA,
            environment={
                "prior_two_months_verification_unavailable": unavailable,
                "representative_ongoing_verification_provided": provided,
            },
            constant_environment={},
        )
        assert execution is not None
        assert sc._formula_execution_runtime_value(execution) is (
            unavailable and provided
        )


@settings(max_examples=40, deadline=None)
@given(
    subject=st.sampled_from(("individual", "person", "claimant")),
    object_description=st.sampled_from(
        (
            "verification that is representative of ongoing income",
            "documentation of income",
            "documentation for a dependent",
        )
    ),
    modifier=st.sampled_from(("", "already ", "still ")),
)
def test_finite_but_document_objects_do_not_add_unfulfillable_fact_gates(
    subject, object_description, modifier
):
    source = (
        f"An allowance applies if the {subject} is resident but "
        f"{modifier}has {object_description}."
    )
    inputs = {
        f"{subject}_is_resident": f"The {subject} is resident.",
        f"{subject}_documentation_provided": (
            f"The {subject} {modifier}has {object_description}."
        ),
    }
    evidence = sc._terminal_gate_evidence(
        {
            "inputs": [
                {"name": name, "description": description}
                for name, description in inputs.items()
            ]
        }
    )
    assert sc._terminal_names_corroborate_source_gates(
        frozenset(inputs),
        sc._source_conjunctive_fact_gates(source),
        terminal_evidence=evidence,
    )


def test_finite_but_keeps_real_explicit_income_facts_required():
    source = (
        "An allowance applies if the taxpayer has income but the spouse has no income."
    )
    gates = sc._source_conjunctive_fact_gates(source)
    assert gates == (
        (frozenset({"taxpayer"}), frozenset({"income"})),
        (frozenset({"spouse"}), frozenset({"no_income"})),
    )
    inputs = {
        "taxpayer_has_income": "The taxpayer has income.",
        "spouse_has_no_income": "The spouse has no income.",
    }
    evidence = sc._terminal_gate_evidence(
        {
            "inputs": [
                {"name": name, "description": description}
                for name, description in inputs.items()
            ]
        }
    )
    names = frozenset(inputs)
    assert sc._terminal_names_corroborate_source_gates(
        names, gates, terminal_evidence=evidence
    )
    for omitted in names:
        assert not sc._terminal_names_corroborate_source_gates(
            names - {omitted}, gates, terminal_evidence=evidence
        )


ABAWD_RESULT_SOURCE = (
    "Division/Section Family Support Chapter No./Name 4 – Economic Stability (ES) "
    "Part No./Name B – Eligibility Factors Section No./Name B-1400-FITAP-SNAP-STEP "
    "SNAP Work Requirements Document No./Name B-1470-SNAP Time-Limited Benefits "
    "for Able-Bodied Adults Without Dependents (ABAWDs) Effective Date February 1, 2026"
    "\n\n19 of 22\n\n"
    "If it is determined that the ABAWD is not meeting the ABAWD work requirement, "
    "has received SNAP benefits for three countable months during the SNAP time "
    "clock period, and has not regained eligibility, the ABAWD is not eligible."
)
ABAWD_INPUTS = {
    "abawd_not_meeting_work_requirement": "The ABAWD is not meeting the ABAWD work requirement.",
    "abawd_received_three_countable_snap_months": "The ABAWD has received SNAP benefits for three countable months during the SNAP time clock period.",
    "abawd_has_not_regained_eligibility": "The ABAWD has not regained eligibility.",
}
ABAWD_FORMULA = " and ".join(ABAWD_INPUTS)


def test_closed_serial_eligibility_result_accepts_faithful_three_fact_formula():
    assert not _faithful_condition_issues(
        "us-la/manual/dcfs/economic-independence/b-1470-snap-time-limited-benefits-for-abawds-398505/page-19",
        ABAWD_RESULT_SOURCE,
        ABAWD_FORMULA,
        ABAWD_INPUTS,
        rule_name="abawd_ineligible",
    )


def test_closed_serial_eligibility_result_requires_every_real_prerequisite():
    gates = sc._source_conjunctive_fact_gates(ABAWD_RESULT_SOURCE)
    assert len(gates) == 3
    evidence = sc._terminal_gate_evidence(
        {
            "inputs": [
                {"name": name, "description": description}
                for name, description in ABAWD_INPUTS.items()
            ]
        }
    )
    names = tuple(ABAWD_INPUTS)
    for count in range(4):
        for subset in itertools.combinations(names, count):
            assert sc._terminal_names_corroborate_source_gates(
                frozenset(subset), gates, terminal_evidence=evidence
            ) == (len(subset) == 3)
    for values in itertools.product((False, True), repeat=3):
        execution = sc._execute_formula_text(
            ABAWD_FORMULA,
            environment=dict(zip(names, values, strict=True)),
            constant_environment={},
        )
        assert execution is not None
        assert sc._formula_execution_runtime_value(execution) is all(values)
    for omitted in names:
        environment = dict.fromkeys(names, True)
        environment[omitted] = False
        omitted_formula = " and ".join(name for name in names if name != omitted)
        execution = sc._execute_formula_text(
            omitted_formula, environment=environment, constant_environment={}
        )
        assert execution is not None
        assert sc._formula_execution_runtime_value(execution) is True


@pytest.mark.parametrize(
    "result",
    (
        "the ABAWD is not eligible when an additional condition holds.",
        "the ABAWD is not eligible and the spouse is blind.",
        "the ABAWD is not eligible. Another notice is required.",
        "the policy is not eligible.",
        "the abawd is not eligible.",
    ),
)
def test_serial_eligibility_result_does_not_truncate_unclosed_or_unknown_scopes(result):
    body = ABAWD_RESULT_SOURCE.split("If ", 1)[1]
    prefix = body.rsplit(", the ABAWD is not eligible.", 1)[0]
    gates = sc._source_gate_fact_signatures(
        sc._source_gate_split_conjunctive_conditions(
            prefix + ", " + result, has_preposed_result=False
        )
    )
    assert len(gates) >= 4
    assert any("eligible" in predicates for _, predicates in gates)


def test_serial_eligibility_fact_after_preposed_result_remains_required():
    body = ABAWD_RESULT_SOURCE.split("If ", 1)[1]
    gates = sc._source_gate_fact_signatures(
        sc._source_gate_split_conjunctive_conditions(body, has_preposed_result=True)
    )
    assert len(gates) == 4
    assert gates[-1][1] == frozenset({"abawd", "eligible"})


@settings(max_examples=30, deadline=None)
@given(
    subject=st.sampled_from(("applicant", "individual", "person", "taxpayer")),
    negative=st.booleans(),
    punctuation=st.sampled_from((".", ";", "!")),
)
def test_complete_serial_eligibility_results_do_not_require_an_outcome_input(
    subject, negative, punctuation
):
    source = (
        f"If the {subject} is resident, has no income, and is blind, "
        f"the {subject} is " + ("not " if negative else "") + "eligible" + punctuation
    )
    gates = sc._source_conjunctive_fact_gates(source)
    assert gates == (
        (frozenset({subject}), frozenset({"resident"})),
        (frozenset({subject}), frozenset({"no_income"})),
        (frozenset({subject}), frozenset({"blind"})),
    )
    inputs = {
        f"{subject}_is_resident": f"The {subject} is resident.",
        f"{subject}_has_no_income": f"The {subject} has no income.",
        f"{subject}_is_blind": f"The {subject} is blind.",
    }
    evidence = sc._terminal_gate_evidence(
        {
            "inputs": [
                {"name": name, "description": description}
                for name, description in inputs.items()
            ]
        }
    )
    names = frozenset(inputs)
    assert sc._terminal_names_corroborate_source_gates(
        names, gates, terminal_evidence=evidence
    )
    for omitted in names:
        assert not sc._terminal_names_corroborate_source_gates(
            names - {omitted}, gates, terminal_evidence=evidence
        )
