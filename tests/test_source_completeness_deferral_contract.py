"""US deferral contract: CFR branch citations and non-operative framing branches.

Replayed US failure bundles (2026-09 to 2026-10) showed two contract gaps in
`[complete-source-unit:deferral]`:

* A regulation branch cited the conventional way (`42 CFR 457.622(c)(1)`) was
  not recognized as the exact current branch, so runtime-gap and repeal
  deferrals that the same rules accept for `26 U.S.C. 32(k)` were rejected
  unless the reason spelled the corpus path `us/regulation/42/457/622(c)(1)`.
* A leaf branch that only frames the unit (`(a) Basis. This section implements
  section 1902(tt)(1) of the Social Security Act.`) had no accepted deferral
  form at all, although the encoder prompt tells the model to defer
  non-operative findings and purpose clauses.

Every acceptance below is paired with negative controls of the same shape.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.source_completeness import (
    analyze_complete_source_unit,
    recognize_source_structure,
)
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

# Byte-exact source branches from the replayed bundles.
BASIS_435_145 = (
    "(a) Basis. This section implements sections 1902(a)(10)(A)(i)(I) and "
    "473(b)(3) of the Act."
)
BASIS_435_927 = (
    "(a) Basis. This section implements section 1902(tt)(1) of the Social Security Act."
)
SEVERABILITY_435_927 = (
    "(e) Severability. Any provision of this section held to be invalid or "
    "unenforceable by its terms, or as applied to any person or circumstance, "
    "or stayed pending further State action, shall be severable from this "
    "section and shall not affect the remainder thereof or the application of "
    "the provision to persons not similarly situated or to dissimilar "
    "circumstances."
)
STATUTORY_BASIS_457_800 = (
    "(a) Statutory basis. This subpart interprets and implements section "
    "2102(b)(3)(C) of the Act, which provides that the State plan must include "
    "a description of procedures the State uses to ensure that health benefits "
    "coverage provided under the State plan does not substitute for coverage "
    "under group health plans."
)
SCOPE_457_800 = (
    "(b) Scope. This subpart sets forth State plan requirements relating to "
    "substitution of coverage in general and specific requirements relating to "
    "substitution of coverage under premium assistance programs."
)
APPLICABILITY_457_800 = (
    "(c) Applicability. The requirements of this subpart apply to separate "
    "child health programs."
)
ELIGIBILITY_435_145 = (
    "(b) Eligibility. The agency must provide Medicaid to individuals for whom "
    "an adoption assistance agreement is in effect with a State or Tribe under "
    "title IV-E of the Act."
)
CONDITION_457_622 = (
    "(a) Basis. Sections 1905(b), 2105(a) and 2105(b) of the Act provide for "
    "payments to States from the States' allotments.\n\n"
    "(b) Enhanced FMAP. The enhanced FMAP is determined under section 2105(b) "
    "of the Act.\n\n"
    "(c) Conditions for availability of enhanced FMAP based on a State's "
    "expenditures—The enhanced FMAP is available for payments based on a "
    "State's expenditures claimed under the State's title XXI program from the "
    "State's fiscal year allotment only under the following conditions:\n\n"
    "(1) The State has an approved title XXI State child health plan;\n\n"
    "(2) The expenditures are allowable under the State's approved title XXI "
    "State child health plan;\n\n"
    "(d) Categories of expenditures for which enhanced FMAP are available."
)
NONOPERATIVE_REASON = (
    "{citation} is a non-operative {kind} statement and supplies no operative rule."
)


def _covered(citation: str, source: str, output_path: str, reason: str, **extra):
    payload = {
        "format": "rulespec/v1",
        "module": {
            "source_verification": {"corpus_citation_path": citation},
            "deferred_outputs": [
                {"output": f"{output_path}#deferred_output", "reason": reason, **extra}
            ],
        },
        "rules": [],
    }
    covered, issues = sc._deferred_coverage(
        payload,
        corpus_citation_path=citation,
        source_text=source,
        branches=recognize_source_structure(source, corpus_citation_path=citation),
    )
    return covered, issues


# Non-operative framing branches: accepted forms.


@pytest.mark.parametrize(
    ("citation", "source", "anchor", "path", "reason"),
    [
        # Reasons copied verbatim from replayed rejected candidates.
        (
            "us/regulation/42/435/145",
            BASIS_435_145,
            "us:regulations/42-cfr/435/145",
            ("a",),
            "42 CFR 435.145(a) states only that this section implements sections "
            "1902(a)(10)(A)(i)(I) and 473(b)(3) of the Act and supplies no "
            "computable policy condition.",
        ),
        (
            "us/regulation/42/435/927",
            BASIS_435_927,
            "us:regulations/42-cfr/435/927",
            ("a",),
            "42 CFR 435.927(a) identifies the statutory basis and states no "
            "independent computation.",
        ),
        (
            "us/regulation/42/435/927",
            BASIS_435_927,
            "us:regulations/42-cfr/435/927",
            ("a",),
            "42 CFR 435.927(a) states the section's statutory basis, not an "
            "independently computable condition.",
        ),
        (
            "us/regulation/42/435/927",
            BASIS_435_927,
            "us:regulations/42-cfr/435/927",
            ("a",),
            "42 CFR 435.927(a) identifies the statutory basis but states no "
            "separate computation or executable condition.",
        ),
        (
            "us/regulation/42/435/927",
            SEVERABILITY_435_927,
            "us:regulations/42-cfr/435/927",
            ("e",),
            "42 CFR 435.927(e) governs the effect of a provision being held "
            "invalid or unenforceable or stayed; it supplies no independently "
            "computable reporting condition.",
        ),
        (
            "us/regulation/42/457/800",
            STATUTORY_BASIS_457_800,
            "us:regulations/42-cfr/457/800",
            ("a",),
            "42 CFR 457.800(a) is a statutory-basis statement describing what the "
            "subpart interprets and implements; it states no computation or "
            "independent test.",
        ),
        (
            "us/regulation/42/457/800",
            SCOPE_457_800,
            "us:regulations/42-cfr/457/800",
            ("b",),
            "42 CFR 457.800(b) describes the subjects addressed by the subpart; it "
            "states no computation or independent test.",
        ),
        # The bounded form taught by the prompt and the retry shape.
        (
            "us/regulation/42/435/145",
            BASIS_435_145,
            "us:regulations/42/435/145",
            ("a",),
            NONOPERATIVE_REASON.format(
                citation="us/regulation/42/435/145(a)", kind="statutory-basis"
            ),
        ),
        (
            "us/statute/42/9999",
            "(a) Purpose The purpose of this section is to coordinate State "
            "reporting.\n(b) Amount The amount is $10.",
            "us:statutes/42/9999",
            ("a",),
            NONOPERATIVE_REASON.format(citation="42 U.S.C. 9999(a)", kind="purpose"),
        ),
        (
            "us-co/statute/39/39-22-9999",
            "(1) Legislative findings. The general assembly finds that timely "
            "refunds serve the public interest.\n(2) The credit is $10.",
            "us-co:statutes/39/39-22-9999",
            ("1",),
            NONOPERATIVE_REASON.format(
                citation="us-co/statute/39/39-22-9999(1)", kind="findings"
            ),
        ),
        (
            "us-mo/statute/143.999",
            "(1) Short title. This section may be cited as the Example Credit "
            "Act.\n(2) The credit is $10.",
            "us-mo:statutes/143/999",
            ("1",),
            NONOPERATIVE_REASON.format(
                citation="us-mo/statute/143.999(1)", kind="short-title"
            ),
        ),
    ],
)
def test_nonoperative_framing_leaf_deferral_is_precise(
    citation, source, anchor, path, reason
):
    covered, issues = _covered(citation, source, f"{anchor}/{'/'.join(path)}", reason)

    assert path in covered
    assert not issues


# Non-operative framing branches: negative controls of the same shape.


@pytest.mark.parametrize(
    ("source", "path", "reason"),
    [
        # Operative or applicability text under any heading.
        (
            APPLICABILITY_457_800.replace("(c)", "(a)"),
            ("a",),
            "42 CFR 457.800(a) states no computation or independent test.",
        ),
        (
            ELIGIBILITY_435_145.replace("(b)", "(a)"),
            ("a",),
            "42 CFR 457.800(a) states no computation or independent test.",
        ),
        (
            "(a) Basis. The agency must provide Medicaid to every adopted child.",
            ("a",),
            "42 CFR 457.800(a) is a non-operative statutory-basis statement and "
            "supplies no operative rule.",
        ),
        (
            "(a) Scope. This subpart applies only to children under age 19.",
            ("a",),
            "42 CFR 457.800(a) states no computation or independent test.",
        ),
        (
            "(a) Scope. This section establishes that the agency must deny "
            "eligibility to any applicant who refuses to cooperate.",
            ("a",),
            "42 CFR 457.800(a) is a non-operative scope statement and supplies no "
            "operative rule.",
        ),
        (
            "(a) Purpose and scope. This section provides that each State must "
            "cover every eligible child.",
            ("a",),
            "42 CFR 457.800(a) is a non-operative scope statement and supplies no "
            "operative rule.",
        ),
        # A second, operative sentence or an amount after a framing sentence.
        (
            "(a) Basis. This section implements section 2102 of the Act. The "
            "State must pay each enrollee monthly.",
            ("a",),
            "42 CFR 457.800(a) is a non-operative statutory-basis statement and "
            "supplies no operative rule.",
        ),
        (
            "(a) Scope. This subpart sets forth a premium of $25 per child.",
            ("a",),
            "42 CFR 457.800(a) is a non-operative scope statement and supplies no "
            "operative rule.",
        ),
        (
            "(a) Purpose. The purpose of this section is to pay 30 percent of "
            "allowable costs.",
            ("a",),
            "42 CFR 457.800(a) is a non-operative purpose statement and supplies "
            "no operative rule.",
        ),
        # Effective dates and definitions stay operative.
        (
            "(a) Effective date. This section is effective July 1, 2024.",
            ("a",),
            "42 CFR 457.800(a) is a non-operative effective-date statement and "
            "supplies no operative rule.",
        ),
        (
            "(a) Definitions. As used in this subpart, child means an individual "
            "under age 19.",
            ("a",),
            "42 CFR 457.800(a) is a non-operative definitions statement and "
            "supplies no operative rule.",
        ),
        # A framing heading with operative children is not a leaf.
        (
            "(a) Scope. This subpart sets forth the following requirements:\n\n"
            "(1) The State must cover each child.\n\n(2) The State must pay $10."
            "\n\n(b) Reports. The State must report each quarter.",
            ("a",),
            "42 CFR 457.800(a) is a non-operative scope statement and supplies no "
            "operative rule.",
        ),
    ],
)
def test_nonoperative_form_rejects_operative_source_branch(source, path, reason):
    covered, issues = _covered(
        "us/regulation/42/457/800",
        source,
        f"us:regulations/42-cfr/457/800/{'/'.join(path)}",
        reason,
    )

    assert path not in covered
    assert any("`module.deferred_outputs[0]`" in issue for issue in issues), issues


@pytest.mark.parametrize(
    "reason",
    [
        # Missing or wrong branch citation, or a citation that does not lead.
        "This statutory-basis branch supplies no operative rule.",
        "42 CFR 435.145(b) is a non-operative statutory-basis statement and "
        "supplies no operative rule.",
        "42 CFR 435.145(a)(1) is a non-operative statutory-basis statement and "
        "supplies no operative rule.",
        "42 CFR 435.14(a) is a non-operative statutory-basis statement and "
        "supplies no operative rule.",
        "The basis in 42 CFR 435.145(a) supplies no operative rule.",
        # No non-operative assertion.
        "42 CFR 435.145(a) is the statutory basis.",
        "42 CFR 435.145(a) cannot be encoded.",
        # A missing dependency or a qualification contradicts the assertion.
        "42 CFR 435.145(a) supplies no operative rule until section 1902 data "
        "are encoded.",
        "42 CFR 435.145(a) supplies no operative rule; however, it controls "
        "eligibility under paragraph (b).",
        "42 CFR 435.145(a) requires section 1902 of the Act and supplies no "
        "operative rule.",
    ],
)
def test_nonoperative_form_rejects_imprecise_reason(reason):
    covered, issues = _covered(
        "us/regulation/42/435/145",
        BASIS_435_145 + "\n\n" + ELIGIBILITY_435_145,
        "us:regulations/42-cfr/435/145/a",
        reason,
    )

    assert ("a",) not in covered
    assert any("`module.deferred_outputs[0]`" in issue for issue in issues), issues


def test_nonoperative_form_rejects_blocked_by_and_root_outputs():
    reason = NONOPERATIVE_REASON.format(
        citation="42 CFR 435.145(a)", kind="statutory-basis"
    )
    covered, issues = _covered(
        "us/regulation/42/435/145",
        BASIS_435_145 + "\n\n" + ELIGIBILITY_435_145,
        "us:regulations/42-cfr/435/145/a",
        reason,
        blocked_by=["us:statutes/42/1396a#medicaid_state_plan"],
    )
    assert ("a",) not in covered and issues

    # A root-anchored output with an otherwise precise reason is told to keep
    # the branch in its output path; it never covers the whole source unit.
    covered, issues = _covered(
        "us/regulation/42/435/145",
        BASIS_435_145 + "\n\n" + ELIGIBILITY_435_145,
        "us:regulations/42-cfr/435/145",
        reason,
    )
    assert not covered
    assert any(
        "otherwise precise reason" in issue
        and "us:regulations/42-cfr/435/145/a#deferred_output" in issue
        for issue in issues
    ), issues


def test_nonoperative_deferral_closes_full_analysis_only_for_the_framing_branch():
    content = """\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/regulation/42/435/927
  deferred_outputs:
    - output: us:regulations/42-cfr/435/927/a#statutory_basis
      reason: >-
        42 CFR 435.927(a) is a non-operative statutory-basis statement and
        supplies no operative rule.
    - output: us:regulations/42-cfr/435/927/b#severability
      reason: >-
        42 CFR 435.927(b) is a non-operative severability statement and
        supplies no operative rule.
rules: []
"""
    source = BASIS_435_927 + "\n\n" + SEVERABILITY_435_927.replace("(e)", "(b)")
    result = analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path="us/regulation/42/435/927",
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    assert not result.issues

    operative_source = source + "\n\n" + ELIGIBILITY_435_145.replace("(b)", "(c)")
    result = analyze_complete_source_unit(
        content,
        operative_source,
        corpus_citation_path="us/regulation/42/435/927",
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    assert any(
        issue.startswith("[complete-source-unit:structure] Source branch (c)")
        for issue in result.issues
    ), result.issues


# Conventional CFR citations for runtime gaps: parity with the USC reader.


@pytest.mark.parametrize(
    ("cited_branch", "accepted"),
    [
        ("7 CFR 273.10(a)(1)", True),
        ("7 C.F.R. § 273.10(a)(1)", True),
        ("us/regulation/7/273/10(a)(1)", True),
        ("7 CFR 273.10(a)", False),
        ("7 CFR 273.10(a)(2)", False),
        ("7 CFR 273.1(a)(1)", False),
        ("7 CFR 273.100(a)(1)", False),
        ("8 CFR 273.10(a)(1)", False),
        ("17 CFR 273.10(a)(1)", False),
        ("7 U.S.C. 273.10(a)(1)", False),
    ],
)
def test_regulation_runtime_gap_binds_conventional_cfr_branch_citation(
    cited_branch, accepted
):
    covered, issues = _covered(
        "us/regulation/7/273/10",
        "(a) Agency plans shall contain:\n1. Agency shall submit plan policies.",
        "us:regulations/7-cfr/273/10/a/1",
        "Cannot be computed until the agency policies document and submission "
        f"event required by {cited_branch} are encoded.",
    )

    assert (("a", "1") in covered) is accepted
    assert bool(issues) is not accepted


def test_cfr_runtime_gap_still_rejects_explicit_computation_and_vague_reason():
    # Same citation form, but the branch states a computation.
    covered, issues = _covered(
        "us/regulation/42/457/622",
        "(b) The enhanced FMAP is 70 percent of the regular FMAP plus 30 "
        "percentage points, as determined in the State plan submission.",
        "us:regulations/42-cfr/457/622/b",
        "42 CFR 457.622(b) requires the State plan submission record, which is "
        "not supplied.",
    )
    assert not covered and issues

    # Same administrative branch, but a generic reason sharing no source terms.
    covered, issues = _covered(
        "us/regulation/42/457/622",
        CONDITION_457_622,
        "us:regulations/42-cfr/457/622/c/1",
        "42 CFR 457.622(c)(1) requires an input that is not available.",
    )
    assert not covered and issues

    # The replayed precise administrative reason is accepted.
    covered, issues = _covered(
        "us/regulation/42/457/622",
        CONDITION_457_622,
        "us:regulations/42-cfr/457/622/c/1",
        "42 CFR 457.622(c)(1) requires an approved title XXI State child health "
        "plan; the approval determination is not supplied in the available "
        "context.",
    )
    assert ("c", "1") in covered and not issues


def test_cfr_reader_identity():
    assert sc._cfr_current_section_identity("us/regulation/42/457/622") == (
        "42",
        "457.622",
        (),
    )
    assert sc._cfr_current_section_identity("us/regulation/26/1.36B-2") == (
        "26",
        "1.36B-2",
        (),
    )
    assert sc._cfr_current_section_identity("us/regulation/26/1/36B-2") == (
        "26",
        "1.36B-2",
        (),
    )
    assert sc._cfr_current_section_identity("us/regulation/7/273/9/d") == (
        "7",
        "273.9",
        ("d",),
    )
    for citation in (
        "us/regulation/42/435",
        "us/regulation/cfr/457",
        "us-ca/regulation/mpp/63-300.1",
        "us/statute/26/32",
    ):
        assert sc._cfr_current_section_identity(citation) is None


# CFR and Act dependencies named by a regulation branch.

FMAP_457_622 = (
    "(a) Basis. Sections 1905(b), 2105(a) and 2105(b) of the Act provides for "
    "payments to States from the States' allotments for a fiscal year, as "
    "determined under \u00a7 457.608, for part of the cost of expenditures.\n\n"
    "(b) Enhanced FMAP\u2014Computations. The enhanced FMAP is the lower of the "
    "following:\n\n"
    "(1) 70 percent of the regular FMAP determined under section 1905(b) of the "
    "Act, plus 30 percentage points; or\n\n"
    "(2) 85 percent.\n\n"
    "(c) Categories. (1) Child health assistance in the form of health benefits "
    "coverage that meets the requirements of section 2103 of the Act; and\n\n"
    "(2) Subject to the 10 percent limit provisions under \u00a7 457.618(a)(2), the "
    "following expenditures:\n\n"
    "(i) Outreach expenditures;\n\n"
    "(ii) Reasonable costs to administer the plan.\n\n"
    "(d) Available allotments. The State's allotments, as reduced in accordance "
    "with \u00a7 457.616, remain available for the fiscal year.\n\n"
    "(e) Other expenditures."
)


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        # Reasons copied verbatim from replayed rejected candidates; the
        # paragraph chapeau cites the limit, and its leaves state no amount.
        (
            ("c", "2"),
            "42 CFR 457.622(c)(2) subjects its listed expenditures to the 10 "
            "percent limit under \u00a7 457.618(a)(2); that limit computation is not "
            "supplied.",
        ),
        (
            ("c", "2"),
            "42 CFR 457.622(c)(2) requires the limit in 42 CFR 457.618(a)(2), "
            "which is not yet encoded.",
        ),
        (
            ("d",),
            "42 CFR 457.622(d) requires available fiscal-year allotments after "
            "reductions under \u00a7 457.616; those allotment and reduction amounts "
            "are not supplied in the available context.",
        ),
    ],
)
def test_cfr_branch_deferral_names_source_cited_regulation_dependency(path, reason):
    covered, issues = _covered(
        "us/regulation/42/457/622",
        FMAP_457_622,
        f"us:regulations/42-cfr/457/622/{'/'.join(path)}",
        reason,
    )

    assert path in covered
    assert not issues


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        # A leaf that states its own percentage and computation keeps them:
        # encode it with the dependency as a declared input, or it fails.
        (
            ("b", "1"),
            "42 CFR 457.622(b)(1) computes 70 percent of the regular FMAP plus 30 "
            "percentage points, but the numeric regular FMAP determined under "
            "section 1905(b) of the Act is not supplied as an executable output.",
        ),
        (
            ("b", "1"),
            "42 CFR 457.622(b)(1) computes 70 percent of the regular FMAP "
            "determined under section 1905(b) of the Act, plus 30 percentage "
            "points; no executable RuleSpec output for that regular FMAP is "
            "supplied in the available context.",
        ),
        # The branch does not cite the named dependency.
        (
            ("b", "2"),
            "42 CFR 457.622(b)(2) requires the regular FMAP determined under "
            "section 1905(b) of the Act, which is not supplied.",
        ),
        # A different section, paragraph, or title than the source cites.
        (
            ("b", "1"),
            "42 CFR 457.622(b)(1) requires the FMAP under section 1905(a) of the "
            "Act; that FMAP is not supplied.",
        ),
        (
            ("c", "2"),
            "42 CFR 457.622(c)(2) requires the limit in § 457.618(a)(1); that "
            "limit is not supplied.",
        ),
        (
            ("c", "2"),
            "42 CFR 457.622(c)(2) requires the limit in 45 CFR 457.618(a)(2); "
            "that limit is not supplied.",
        ),
        # A category-membership label is a boundary predicate, not a
        # missing computation.
        (
            ("c", "1"),
            "42 CFR 457.622(c)(1) requires coverage meeting section 2103 of the "
            "Act; that coverage determination is not supplied.",
        ),
        # No missing state, a qualified or retracted one, or a missing state
        # beyond the next clause.
        (
            ("b", "1"),
            "42 CFR 457.622(b)(1) uses the regular FMAP determined under section "
            "1905(b) of the Act.",
        ),
        (
            ("b", "1"),
            "42 CFR 457.622(b)(1) uses section 1905(b) of the Act; however, it is "
            "not supplied only for some States.",
        ),
        (
            ("b", "1"),
            "42 CFR 457.622(b)(1) uses section 1905(b) of the Act. The rate is "
            "known. The table is not supplied.",
        ),
        (
            ("b", "1"),
            "42 CFR 457.622(b)(1) requires section 1905(b) of the Act, which is "
            "not supplied, although the State plan states the rate.",
        ),
        # The current section is never its own dependency, and the reason
        # must open with the exact current branch.
        (
            ("b", "1"),
            "42 CFR 457.622(b)(1) requires 42 CFR 457.622(b)(2), which is not "
            "supplied.",
        ),
        (
            ("b", "1"),
            "The regular FMAP determined under section 1905(b) of the Act is not "
            "supplied for 42 CFR 457.622(b)(1).",
        ),
        (
            ("b",),
            "42 CFR 457.622(b)(1) requires section 1905(b) of the Act, which is "
            "not supplied.",
        ),
    ],
)
def test_cfr_branch_dependency_rejects_unbound_or_unstated_dependency(path, reason):
    covered, issues = _covered(
        "us/regulation/42/457/622",
        FMAP_457_622,
        f"us:regulations/42-cfr/457/622/{'/'.join(path)}",
        reason,
    )

    assert path not in covered
    assert issues


def test_cfr_dependency_form_is_limited_to_numbered_federal_regulations():
    reason = (
        "us-ca/regulation/mpp/63-300(a) requires the limit under § 457.618(a)(2), "
        "which is not supplied."
    )
    assert not sc._reason_names_missing_federal_regulation_dependency(
        reason,
        "(a) Subject to the limit under § 457.618(a)(2), outreach.",
        corpus_citation_path="us-ca/regulation/mpp/63-300",
        path=("a",),
    )


REPORTING_273_4_B = (
    "(a) Household members. No person is eligible unless that person is a U.S. "
    "citizen.\n\n"
    "(b) Reporting illegal aliens. (1) The State agency must inform the local "
    "USCIS office immediately whenever personnel responsible for the "
    "certification or recertification of households determine that any member "
    "of a household is ineligible to receive SNAP benefits because the member is "
    "present in the U.S. in violation of the INA.\n\n"
    "(2) When a household indicates inability or unwillingness to provide "
    "documentation of alien status for any household member, the State agency "
    "must classify that member as an ineligible alien.\n\n"
    "(c) Sponsored aliens. A sponsored alien is an alien for whom a person has "
    "executed an affidavit of support."
)


def test_regulation_runtime_gap_cannot_cover_an_operative_child_rule():
    """A replayed 7 CFR 273.4(b) runtime gap would also have covered (b)(2)."""

    branches = recognize_source_structure(
        REPORTING_273_4_B, corpus_citation_path="us/regulation/7/273/4"
    )
    assert ("b", "1") in {branch.path for branch in branches}
    assert ("b", "2") in {branch.path for branch in branches}

    whole = (
        "7 CFR 273.4(b) requires immediate State-agency notice to the local USCIS "
        "office after a certification or recertification determination; the "
        "certification event, USCIS routing, and notification workflow are not "
        "represented."
    )
    covered, issues = _covered(
        "us/regulation/7/273/4",
        REPORTING_273_4_B,
        "us:regulations/7-cfr/273/4/b",
        whole,
    )
    assert ("b",) not in covered and issues

    notice_only = whole.replace("7 CFR 273.4(b)", "7 CFR 273.4(b)(1)")
    covered, issues = _covered(
        "us/regulation/7/273/4",
        REPORTING_273_4_B,
        "us:regulations/7-cfr/273/4/b/1",
        notice_only,
    )
    assert ("b", "1") in covered and not issues


def test_regulation_runtime_gap_rejects_mandated_income_consequence():
    """Replayed 7 CFR 273.4(c)(4) deferral: a duty list plus an attribution rule."""

    source = (
        "(a) Citizens. No person is eligible unless that person is a U.S. "
        "citizen.\n\n"
        "(b) Reporting. The State agency must report each case.\n\n"
        "(c) Sponsored aliens.\n\n"
        "(1) Definition. A sponsored alien is an alien for whom a person has "
        "executed an affidavit of support.\n\n"
        "(2) Deeming. Income of the sponsor is deemed income of the alien.\n\n"
        "(3) Exemptions. Deeming does not apply to an indigent alien.\n\n"
        "(4) Eligible sponsored alien's responsibilities. The eligible sponsored "
        "alien is responsible for providing the State agency at the time of "
        "application and at the time of recertification with the information and "
        "documentation necessary to calculate deemed income. The State agency "
        "must attribute the entire amount of income and resources to the "
        "applicant eligible sponsored alien until he or she provides the "
        "information specified under this paragraph (c)(4). Changes must be "
        "handled during the certification period.\n\n"
        "(5) Notice. The State agency must notify the household of each "
        "certification decision in a written notice.\n\n"
        "(d) Other provisions."
    )
    branches = recognize_source_structure(
        source, corpus_citation_path="us/regulation/7/273/4"
    )
    assert {("c", "4"), ("c", "5")} <= {branch.path for branch in branches}
    reason = (
        "7 CFR 273.4(c)({branch}) requires application, recertification, and "
        "certification-period {topic} workflow not represented by the runtime."
    )
    covered, issues = _covered(
        "us/regulation/7/273/4",
        source,
        "us:regulations/7-cfr/273/4/c/4",
        reason.format(branch="4", topic="documentation"),
    )
    assert ("c", "4") not in covered and issues

    covered, issues = _covered(
        "us/regulation/7/273/4",
        source,
        "us:regulations/7-cfr/273/4/c/5",
        "7 CFR 273.4(c)(5) requires a written certification notice event, which is "
        "not supplied as a runtime notice record.",
    )
    assert ("c", "5") in covered and not issues


def test_cfr_dependency_parent_deferral_requires_the_dependency_to_block_it():
    branches = {
        branch.path
        for branch in recognize_source_structure(
            FMAP_457_622, corpus_citation_path="us/regulation/42/457/622"
        )
    }
    assert {
        ("b",),
        ("b", "1"),
        ("b", "2"),
        ("c",),
        ("c", "1"),
        ("c", "2"),
        ("c", "2", "i"),
        ("d",),
    } <= branches

    # (b) computes over leaves that state their own percentages, which stay
    # encoded (with the regular FMAP as a declared input) or fail.
    covered, issues = _covered(
        "us/regulation/42/457/622",
        FMAP_457_622,
        "us:regulations/42-cfr/457/622/b",
        "42 CFR 457.622(b) requires the lower of its paragraphs, and the regular "
        "FMAP determined under section 1905(b) of the Act is not supplied.",
    )
    assert ("b",) not in covered and issues

    # (c) lists independent categories; only (c)(2) cites the 457.618(a)(2)
    # limit, so deferring all of (c) would also hide the (c)(1) category.
    covered, issues = _covered(
        "us/regulation/42/457/622",
        FMAP_457_622,
        "us:regulations/42-cfr/457/622/c",
        "42 CFR 457.622(c) subjects its listed expenditures to the 10 percent "
        "limit under 42 CFR 457.618(a)(2); that limit computation is not supplied.",
    )
    assert ("c",) not in covered and issues


# Independent review of #1897: real CFR texts the first dependency reader
# covered because `under` / `in accordance with` / `pursuant to` linked a
# category label, plus the reviewer's adversarial shapes. Every one must stay
# rejected. Bodies are byte-exact corpus provisions.

REVIEW_BODIES = json.loads(
    (
        Path(__file__).parent
        / "fixtures/source_completeness/cfr_deferral_label_controls.json"
    ).read_text()
)


@pytest.mark.parametrize(
    ("citation", "path", "reason"),
    [
        # Chapeau label "services under § 98.50" over an under-13, 85% SMI,
        # $1,000,000 asset eligibility test.
        (
            "us/regulation/45/98/20",
            ("a",),
            "45 CFR 98.20(a) depends on § 98.50, which is not supplied.",
        ),
        # "demonstration project under section 1115(a)(2) of the Act" over an
        # age 19-64 eligibility group.
        (
            "us/regulation/42/435/561",
            ("a", "2"),
            "42 CFR 435.561(a)(2) depends on section 1115(a)(2) of the Act, which "
            "is not supplied.",
        ),
        # "temporarily absent in accordance with § 416.1149(c)(1)" over the SSI
        # essential-person increment.
        (
            "us/regulation/20/416/430",
            ("b",),
            "20 CFR 416.430(b) depends on § 416.1149(c)(1), which is not supplied.",
        ),
        # "as defined under section 1927(k)(7)(A)(i) of the Act" over the $1/$3
        # Part D copayments and their CPI update.
        (
            "us/regulation/42/423/782",
            ("a", "2", "iii", "a"),
            "42 CFR 423.782(a)(2)(iii)(A) depends on section 1927(k)(7)(A)(i) of "
            "the Act, which is not supplied.",
        ),
        # "not eligible for Medicaid under § 435.117" is an eligibility test.
        (
            "us/regulation/42/457/360",
            ("b", "1", "ii"),
            "42 CFR 457.360(b)(1)(ii) depends on § 435.117, which is not supplied.",
        ),
        # "we continue to consider ... deeming will cease" is a consequence,
        # not an administrative workflow.
        (
            "us/regulation/20/416/1167",
            ("c",),
            "20 CFR 416.1167(c) requires the active duty military absence "
            "information, which is not available at runtime.",
        ),
    ],
)
def test_review_real_cfr_label_and_consequence_controls_stay_rejected(
    citation, path, reason
):
    body = REVIEW_BODIES[citation]
    output_root = sc._deferral_anchor_bases(citation)[0]
    covered, issues = _covered(
        citation, body, f"{output_root}/{'/'.join(path)}", reason
    )

    assert path not in covered
    assert issues


SPONSOR_273_4 = (
    "(a) Citizens. No person is eligible unless that person is a U.S. citizen.\n\n"
    "(b) Reporting. The State agency must report each case.\n\n"
    "(c) Sponsor deeming.\n\n"
    "(1) Definition. A sponsored alien is an alien with a sponsor.\n\n"
    "(2) Resources. (i) The State agency must count the sponsor's income.\n\n"
    "(ii) Other income.\n\n(iii) Reserved.\n\n"
    "(iv) The State agency must deem as available to the eligible sponsored "
    "alien the total amount of the resources of the sponsor and sponsor's spouse "
    "as determined in accordance with § 273.8, reduced by $1,500.\n\n"
    "(d) Other."
)


@pytest.mark.parametrize(
    ("citation", "source", "output", "reason"),
    [
        # Replayed bundle reason: a value-supplying link, but the leaf states
        # its own amount and a mandated deeming consequence.
        (
            "us/regulation/7/273/4",
            SPONSOR_273_4,
            "us:regulations/7-cfr/273/4/c/2/iv",
            "7 CFR 273.4(c)(2)(iv) requires sponsor and sponsor-spouse resources "
            "as determined under 7 CFR 273.8 before applying the $1,500 reduction; "
            "no executable RuleSpec output for those resources is supplied.",
        ),
        # Runtime-gap leaf with an ineligibility and exclusion consequence.
        (
            "us/regulation/7/273/99",
            "(a) Notice.\n\n(1) The State agency must notify the USCIS office of "
            "each determination and file the notice record.\n\n(2) A household "
            "member who does not submit the required immigration document is "
            "ineligible to participate and the State agency will exclude that "
            "member's share of the allotment.\n\n(b) Other.",
            "us:regulations/7-cfr/273/99/a/2",
            "7 CFR 273.99(a)(2) requires the immigration document submission "
            "record, which is not available at runtime.",
        ),
        # A citation of (a)(1) never stands for all of (a), in either form.
        (
            "us/regulation/7/273/98",
            "(a) Reporting.\n\n(1) The State agency must submit the monthly report "
            "to FNS.\n\n(2) The State agency must file the corrective action plan "
            "record.\n\n(b) Other.",
            "us:regulations/7-cfr/273/98/a",
            "7 CFR 273.98(a)(1) requires the monthly report submission record, "
            "which is not available at runtime.",
        ),
        (
            "us/regulation/7/273/98",
            "(a) Reporting.\n\n(1) The State agency must submit the monthly report "
            "to FNS.\n\n(2) The State agency must file the corrective action plan "
            "record.\n\n(b) Other.",
            "us:regulations/7-cfr/273/98/a",
            "us/regulation/7/273/98(a)(1) requires the monthly report submission "
            "record, which is not available at runtime.",
        ),
        # Framing openings followed by an operative clause or a spelled amount.
        (
            "us/regulation/7/273/97",
            "(a) Scope. This section sets forth the income standards for SNAP, and "
            "the State agency must deny benefits to any household that refuses to "
            "cooperate in the quality control review.\n\n(b) Amount. The amount "
            "is $10.",
            "us:regulations/7-cfr/273/97/a",
            "7 CFR 273.97(a) is a non-operative scope statement and supplies no "
            "operative rule.",
        ),
        (
            "us/regulation/7/273/96",
            "(a) Purpose. The purpose of this section is to deny eligibility to "
            "households whose gross income exceeds one hundred thirty percent of "
            "the poverty line.\n\n(b) Amount. The amount is $10.",
            "us:regulations/7-cfr/273/96/a",
            "7 CFR 273.96(a) is a non-operative purpose statement and supplies no "
            "operative rule.",
        ),
        (
            "us/regulation/42/435/603",
            "(a) Basis. This section implements section 1902(e)(14) of the Act and "
            "the agency must disregard an amount equal to five percentage points "
            "of the Federal poverty level.\n\n(b) Other.",
            "us:regulations/42-cfr/435/603/a",
            "42 CFR 435.603(a) is a non-operative statutory-basis statement and "
            "supplies no operative rule.",
        ),
    ],
)
def test_review_adversarial_deferrals_stay_rejected(citation, source, output, reason):
    covered, issues = _covered(citation, source, output, reason)

    assert not covered
    assert issues


def _regulation_leaf_source(leaf: str) -> str:
    return f"(a) Procedures.\n\n(1) {leaf}\n\n(2) Other text.\n\n(b) Other."


@pytest.mark.parametrize(
    "leaf",
    [
        # Corpus leaves the corpus probe found under the lexical guard; each
        # states an operative rule beside its administrative words.
        "The agency must, to determine countable income, deduct amounts that "
        "would be deducted in determining eligibility under the State's approved "
        "plan for OAA, AFDC, AB, APTD, or AABD.",
        "If the request is not filed within 20 days of the date of the notice, "
        "we will pay to you all past-due benefits remaining after we reimburse "
        "the State.",
        "If an eligible household files a complete report after the issuance "
        "month, the State agency shall not provide the household with an "
        "opportunity to participate for that month.",
        "For States that elect to extend eligibility to unborn children under "
        "the approved Child Health Plan, the State does not adopt eligibility "
        "standards and methodologies that were more restrictive than those "
        "applied under policies of the State plan in effect on June 1, 1997.",
        "If your parent is absent due to active duty, we continue to consider "
        "that person to be living in the same household, and deeming will cease "
        "when evidence indicates otherwise.",
        "A household member who does not submit the required document is "
        "ineligible, and the State agency will exclude that member's share.",
        "The State agency shall apply the different income standard at the next "
        "recertification after the household reports the change.",
        "The MAGI-based income of a child who is not required to file a tax "
        "return is not included in household income, as reported on the "
        "application.",
        # An administrative word with no administrative act on an artifact.
        "The State agency is responsible for the program in each project area.",
    ],
)
def test_regulation_runtime_gap_rejects_operative_or_non_administrative_leaf(leaf):
    covered, issues = _covered(
        "us/regulation/7/273/94",
        _regulation_leaf_source(leaf),
        "us:regulations/7-cfr/273/94/a/1",
        "7 CFR 273.94(a)(1) requires the State agency report, notice, and "
        "certification records for this paragraph, which are not available at "
        "runtime.",
    )

    assert ("a", "1") not in covered
    assert issues


@pytest.mark.parametrize(
    "leaf",
    [
        "The State agency must inform the local USCIS office immediately whenever "
        "personnel responsible for the certification or recertification of "
        "households determine that any member of a household is ineligible to "
        "receive SNAP benefits because the member is present in the U.S. in "
        "violation of the INA.",
        "Each State must submit to CMS the reporting data described in paragraph "
        "(d) of this section by the deadline in the certification notice.",
        "The State agency must send the household a written notice of the "
        "certification record within 10 days of the hearing request.",
    ],
)
def test_regulation_runtime_gap_still_accepts_administrative_duty(leaf):
    covered, issues = _covered(
        "us/regulation/7/273/94",
        _regulation_leaf_source(leaf),
        "us:regulations/7-cfr/273/94/a/1",
        "7 CFR 273.94(a)(1) requires the State agency notice, certification, "
        "report, and submission records for this duty, which are not available "
        "at runtime.",
    )

    assert ("a", "1") in covered
    assert not issues


@pytest.mark.parametrize(
    ("leaf", "reason"),
    [
        # `Determined under` after a category and a program label are not
        # values supplied by the cited provision.
        (
            "A separate line of business (as determined under § 1.414(r)-3) "
            "satisfies the administrative scrutiny requirement.",
            "26 CFR 1.414(r)-94(a)(1) depends on § 1.414(r)-3, which is not supplied.",
        ),
        (
            "An enrollee may appear by video-teleconferencing, telephone, or in "
            "person, as determined under § 423.2020.",
            "26 CFR 1.414(r)-94(a)(1) depends on § 423.2020, which is not supplied.",
        ),
        (
            "Individuals entitled to benefits under § 406.10 are not eligible.",
            "26 CFR 1.414(r)-94(a)(1) depends on § 406.10, which is not supplied.",
        ),
        (
            "For able-bodied adults subject to the time limit of § 273.24, any "
            "change in work hours must be reported.",
            "26 CFR 1.414(r)-94(a)(1) depends on § 273.24, which is not supplied.",
        ),
        # A value link, but the leaf computes (an imperative worksheet step).
        (
            "Multiply the product arrived at in paragraph (g)(2) by the number of "
            "months. The penalty may not exceed an amount specified in § 3.91.",
            "26 CFR 1.414(r)-94(a)(1) depends on § 3.91, which is not supplied.",
        ),
    ],
)
def test_cfr_dependency_rejects_label_links_and_computing_leaves(leaf, reason):
    covered, issues = _covered(
        "us/regulation/26/1.414r-94",
        _regulation_leaf_source(leaf),
        "us:regulations/26-cfr/1/414r-94/a/1",
        reason.replace("1.414(r)-94", "1.414r-94"),
    )

    assert ("a", "1") not in covered
    assert issues


def test_cfr_dependency_accepts_value_link_in_the_same_harness():
    """The positive twin of the label controls above."""

    covered, issues = _covered(
        "us/regulation/26/1.414r-94",
        _regulation_leaf_source(
            "The home is excluded to the extent its value does not exceed the "
            "amount set forth in § 416.1212."
        ),
        "us:regulations/26-cfr/1/414r-94/a/1",
        "26 CFR 1.414r-94(a)(1) depends on § 416.1212, which is not supplied.",
    )

    assert ("a", "1") in covered and not issues


def test_cfr_dependency_parent_chapeau_must_not_compute():
    source = (
        "(a) National average cost. This is equal to the sum of the costs for "
        "each area as calculated pursuant to § 54.1308(a) divided by the sum of "
        "the loops:\n\n(1) First area.\n\n(2) Second area.\n\n(b) Other."
    )
    covered, issues = _covered(
        "us/regulation/47/54/1309",
        source,
        "us:regulations/47-cfr/54/1309/a",
        "47 CFR 54.1309(a) requires the costs calculated pursuant to "
        "§ 54.1308(a), which are not supplied.",
    )

    assert ("a",) not in covered and issues
