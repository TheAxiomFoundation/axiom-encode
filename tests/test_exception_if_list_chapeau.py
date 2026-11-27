"""A list chapeau ending "if they are:" takes its condition from its items.

A provision may state its exclusions only in a list below a chapeau such as
"The renewal tally leaves out course hours if they are:". The exception-pair
check asks for a case pair whose toggled selector the condition names, and the
condition here is only "if they are:". These tests pin the rule that a
selector named by a cited excerpt of one of the chapeau's own items satisfies
such a chapeau, and that an excerpt from anywhere else does not. The item's
own wording, not the chapeau's, says which value of the selector the exception
needs. When the shared reader can't settle that value from the wording, the
item gives none and adds no relevance, so the chapeau stays unpaired, as on
main. The items run from the chapeau through the first that ends a sentence,
so a labeled paragraph after a list whose last item ends a sentence isn't one
of them, and an item that ends with neither a list separator nor a sentence
end leaves the list with no items. Only a bare subject, a pronoun or a
determiner and one word, may stand between "if" or "when" and the verb. A
clause with more words in its subject, or that states a condition of its own,
still has to name the selector itself.
"""

from __future__ import annotations

import copy
import functools

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as completeness_module
from axiom_encode.harness import validator_pipeline as validator_module

CITATION = "us/regulation/99/990/99"
MODULE = "us:regulations/99-cfr/990/99"
CHAPEAU = "The renewal tally leaves out course hours if they are:"
CHAPEAU_EXCERPT = "renewal tally leaves out course hours if they are:"
APPROVED = "Earned in a course the board approved in writing"
NOT_FINISHED = "Not finished by the course deadline"
FEE_SENTENCE = "The board must post its license fee on its website."
FEES = f"(b) Fees. {FEE_SENTENCE}\n"
HEADING = "(a) Hours left out of the renewal tally."
SOURCE = f"{HEADING} {CHAPEAU}\n\n(1) {APPROVED}; or\n\n(2) {NOT_FINISHED}.\n\n{FEES}"
RENEWAL = (
    "A licensee must complete forty course hours in each renewal period to "
    "keep an active license."
)
# The chapeau sits at (a)(2), below a sibling (a)(1) that a parent rule cites.
NESTED_SOURCE = (
    f"(a) Course hours—(1) In general. {RENEWAL}\n\n"
    f"(2) Hours left out. {CHAPEAU}\n\n"
    f"(i) {APPROVED}; or\n\n(ii) {NOT_FINISHED}.\n\n{FEES}"
)
APPROVED_INPUT = "earned_in_course_board_approved_in_writing"
NOT_FINISHED_INPUT = "not_finished_by_course_deadline"
OUTPUT = "course_hour_left_out_of_renewal_tally"
RENEWAL_OUTPUT = "licensee_keeps_active_license"
# The list after "are:" names what the election allows. The condition is the
# election, which the chapeau states itself.
ELECTION_CHAPEAU = (
    "If the household so elects, the deductions allowed to the household are:"
)
MEDICAL = "A medical expense deduction for an elderly or disabled member"
DEPENDENT_CARE = "A dependent care deduction for a dependent child"
ELECTION_SOURCE = (
    f"(a) Elective deductions. {ELECTION_CHAPEAU}\n\n(1) {MEDICAL}; and\n\n"
    f"(2) {DEPENDENT_CARE}.\n\n"
    "(b) Records. The household must keep a record of each deduction.\n"
)
ELECTION_INPUT = "household_elects_deductions"
MEDICAL_INPUT = "medical_expense_deduction_for_elderly_or_disabled_member"
DEPENDENT_CARE_INPUT = "dependent_care_deduction_for_child"
DEDUCTION_OUTPUT = "household_deduction_allowed"
SSI_INPUT = "receives_supplemental_security_income"
SSI_ITEM = "receives Supplemental Security Income"
# A paragraph (a)(3) follows the list (a)(1)-(a)(2) under the same chapeau.
RECORDS = "The board must keep a record of each hour a licensee reports"
RECORDS_INPUT = "keeps_record_of_each_hour_reported_by_licensee"
SIBLING_SOURCE = (
    f"{HEADING} {CHAPEAU}\n\n(1) {APPROVED}; or\n\n(2) {NOT_FINISHED}.\n\n"
    f"(3) Records. {RECORDS}.\n\n{FEES}"
)
# The chapeau sits in the unit's unlabeled lead-in, and (c) follows the list.
FEE_EXCERPT = "must post its license fee on its website"
FEE_INPUT = "post_license_fee_on_website"
ROOT_SOURCE = (
    f"{CHAPEAU}\n\n(a) {APPROVED}; or\n\n(b) {NOT_FINISHED}.\n\n"
    f"(c) Fees. {FEE_SENTENCE}\n"
)
# Unlabeled text after the last item, with no branch of its own.
FLUSH_SOURCE = (
    f"{HEADING} {CHAPEAU}\n\n(1) {NOT_FINISHED}; or\n\n(2) {APPROVED}.\n\n"
    f"{RECORDS}.\n\n{FEES}"
)
# The same text after a last item that opens with "Not".
FLUSH_AFTER_CUE_SOURCE = (
    f"{HEADING} {CHAPEAU}\n\n(1) {APPROVED}; or\n\n(2) {NOT_FINISHED}.\n\n"
    f"{RECORDS}.\n\n{FEES}"
)
# A list whose last item doesn't end a sentence.
UNENDED_SOURCE = (
    f"{HEADING} {CHAPEAU}\n\n(1) {APPROVED}; or\n\n(2) {NOT_FINISHED}\n\n{FEES}"
)
# The last item has lost its period, and a labeled paragraph (a)(3) follows.
RUN_ON_SOURCE = (
    f"{HEADING} {CHAPEAU}\n\n(1) {APPROVED}; or\n\n(2) {NOT_FINISHED}\n\n"
    f"(3) Records. {RECORDS}.\n\n{FEES}"
)
# "disabled" restricts the subject, so it is a condition of the clause's own.
DISABLED_CHAPEAU = (
    "Payments to a household are excluded from income if the disabled member's "
    "payments are:"
)
DISABLED_CHAPEAU_EXCERPT = (
    "are excluded from income if the disabled member's payments are:"
)
PENSION = "Received from a pension fund"
EMPLOYER = "Paid by a former employer"
DISABLED_SOURCE = (
    f"(a) Excluded payments. {DISABLED_CHAPEAU}\n\n(1) {PENSION}; or\n\n"
    f"(2) {EMPLOYER}.\n\n"
    "(b) Records. The household must keep a record of each payment.\n"
)
DISABLED_INPUT = "member_is_disabled"
PENSION_INPUT = "received_from_pension_fund"
EMPLOYER_INPUT = "paid_by_former_employer"
PAYMENT_OUTPUT = "payment_excluded_from_income"
# An "exclude" chapeau over two negative items, and over two positive ones.
EXCLUSION_CHAPEAU = "Payments are excluded from income if they are:"
UNWITHHELD = "Not subject to withholding"
UNREPORTED = "Not reported to the agency"
PAYMENT_RECORDS = "(b) Records. The household must keep a record of each payment.\n"
NEGATIVE_ITEMS_SOURCE = (
    f"(a) Excluded payments. {EXCLUSION_CHAPEAU}\n\n(1) {UNWITHHELD}; or\n\n"
    f"(2) {UNREPORTED}.\n\n{PAYMENT_RECORDS}"
)
POSITIVE_ITEMS_SOURCE = (
    f"(a) Excluded payments. {EXCLUSION_CHAPEAU}\n\n(1) {PENSION}; or\n\n"
    f"(2) {EMPLOYER}.\n\n{PAYMENT_RECORDS}"
)
# A constructed section on continuing education for a licensing board, with a
# "neither:" list of three items at (b)(2), an "if they are:" list of four
# items at (c)(2), a definition after that list and a paragraph (d). Items (i)
# and (ii) of the "if they are:" list end with a bare ";", and item (ii) holds
# a comma and an "and". Item (iv) ends with a cross-reference whose last label
# stands alone.
SECTION_CHAPEAU = "The board's renewal count leaves out course hours if they are:"
TAUGHT = "Taught by the licensee as the course's instructor"
NAMED_PROVIDER = "Given by a provider the board named under § 990.31(e)(4)(ii) or (iv)"
TAUGHT_INPUT = "taught_by_licensee_as_instructor"
NAMED_PROVIDER_INPUT = "given_by_provider_board_named"
SECTION_LIST_REFS = (
    "99 CFR 990.99(c)(2); 99 CFR 990.99(c)(2)(i); 99 CFR 990.99(c)(2)(ii); "
    "99 CFR 990.99(c)(2)(iii); 99 CFR 990.99(c)(2)(iv)"
)
SECTION_SOURCE = (
    "(a) In general. A licensee must complete continuing education to renew "
    "its license, as paragraphs (b) through (d) of this section describe.\n\n"
    f"(b) Renewal hours—(1) In general. {RENEWAL}\n\n"
    "(2) Hour limits. A licensee meets paragraph (b)(1) of this section if the "
    "course hours it reports for a period include neither:\n\n"
    "(i) More than ten hours from one provider;\n\n"
    "(ii) Hours from a course shorter than one hour; nor\n\n"
    "(iii) Hours from a course taken in an earlier period. A course belongs to "
    "the period in which it ends.\n\n"
    "(c) Reports—(1) Timing. A licensee must report its course hours to the "
    "board within ninety days after each period ends.\n\n"
    f"(2) Hours left out. {SECTION_CHAPEAU}\n\n"
    f"(i) {NOT_FINISHED};\n\n"
    f"(ii) {APPROVED}, which the licensee finished and paid for before the "
    "period ended;\n\n"
    f"(iii) {TAUGHT}; or\n\n"
    f"(iv) {NAMED_PROVIDER}.\n\n"
    "(3) Definition. In this paragraph (c), a course hour is sixty minutes of "
    "instruction. A shorter session counts as part of an hour.\n\n"
    "(d) Fees. The board sets the renewal fee each year.\n\n"
)
# The same section with a cross-reference to another paragraph inside its
# "if they are:" chapeau.
SECTION_XREF_CHAPEAU = (
    "The board's renewal count under paragraph (b)(1) of this section leaves "
    "out course hours if they are:"
)
SECTION_XREF_SOURCE = SECTION_SOURCE.replace(SECTION_CHAPEAU, SECTION_XREF_CHAPEAU)


def _atom(excerpt: str, kind: str = "exception", citation: str = CITATION) -> dict:
    return {
        "path": "versions[0].formula",
        "kind": kind,
        "source": {"corpus_citation_path": citation, "excerpt": excerpt},
    }


def _rule(name: str, source: str, formula: str, atoms: list[dict]) -> dict:
    return {
        "name": name,
        "kind": "derived",
        "entity": "Person",
        "dtype": "Judgment",
        "period": "Day",
        "source": source,
        "metadata": {"proof": {"atoms": atoms}},
        "versions": [{"effective_from": "0001-01-01", "formula": formula}],
    }


def _doc(input_names: tuple[str, ...], rules: list[dict]) -> dict:
    return {
        "format": "rulespec/v1",
        "module": {
            "summary": "Course hours left out of the renewal tally.",
            "source_verification": {"corpus_citation_path": CITATION},
        },
        "inputs": [
            {"name": name, "entity": "Person", "dtype": "Boolean", "period": "Day"}
            for name in input_names
        ],
        "rules": rules,
    }


def _case(name: str, inputs: dict[str, bool], outputs: dict[str, str]) -> dict:
    return {
        "name": name,
        "period": {
            "period_kind": "custom",
            "name": "day",
            "start": "2024-01-15",
            "end": "2024-01-15",
        },
        "input": {f"{MODULE}#input.{key}": value for key, value in inputs.items()},
        "output": {f"{MODULE}#{key}": value for key, value in outputs.items()},
    }


def _material(
    *,
    approved_input: str = APPROVED_INPUT,
    not_finished_input: str = NOT_FINISHED_INPUT,
    item_atoms: tuple[str, ...] = (APPROVED, NOT_FINISHED),
) -> tuple[dict, list[dict]]:
    rule = _rule(
        OUTPUT,
        "99 CFR 990.99(a); 99 CFR 990.99(a)(1); 99 CFR 990.99(a)(2)",
        f"{approved_input}\nor {not_finished_input}",
        [_atom(CHAPEAU_EXCERPT), *(_atom(excerpt) for excerpt in item_atoms)],
    )
    doc = _doc((approved_input, not_finished_input), [rule])

    def case(name: str, approved: bool, not_finished: bool, excluded: str) -> dict:
        return _case(
            name,
            {approved_input: approved, not_finished_input: not_finished},
            {OUTPUT: excluded},
        )

    cases = [
        case("ordinary hour counts", False, False, "not_holds"),
        case("approved course hour is left out", True, False, "holds"),
        case("unfinished course hour is left out", False, True, "holds"),
    ]
    return doc, cases


def _nested_material(*, pair_an_item: bool) -> tuple[dict, list[dict]]:
    renewal_rule = _rule(
        RENEWAL_OUTPUT,
        "99 CFR 990.99(a); 99 CFR 990.99(a)(1)",
        "forty_course_hours_completed\nand keep_active_license",
        [
            _atom(
                "must complete forty course hours in each renewal period to keep "
                "an active license",
                kind="predicate",
            )
        ],
    )
    exclusion_rule = _rule(
        OUTPUT,
        "99 CFR 990.99(a)(2); 99 CFR 990.99(a)(2)(i); 99 CFR 990.99(a)(2)(ii)",
        f"{APPROVED_INPUT}\nor {NOT_FINISHED_INPUT}",
        [_atom(CHAPEAU_EXCERPT), _atom(APPROVED), _atom(NOT_FINISHED)],
    )
    doc = _doc(
        (
            "forty_course_hours_completed",
            "keep_active_license",
            APPROVED_INPUT,
            NOT_FINISHED_INPUT,
        ),
        [renewal_rule, exclusion_rule],
    )

    def case(name: str, active: bool, approved: bool, kept: str, excluded: str):
        return _case(
            name,
            {
                "forty_course_hours_completed": True,
                "keep_active_license": active,
                APPROVED_INPUT: approved,
                NOT_FINISHED_INPUT: False,
            },
            {RENEWAL_OUTPUT: kept, OUTPUT: excluded},
        )

    cases = [
        case("hours kept for an active license", True, False, "holds", "not_holds"),
        case(
            "hours not kept for an active license",
            False,
            False,
            "not_holds",
            "not_holds",
        ),
    ]
    if pair_an_item:
        cases.append(
            case("approved course hour is left out", True, True, "holds", "holds")
        )
    return doc, cases


def _election_material(*, pair_the_election: bool) -> tuple[dict, list[dict]]:
    if pair_the_election:
        inputs = (ELECTION_INPUT, MEDICAL_INPUT, DEPENDENT_CARE_INPUT)
        formula = f"{ELECTION_INPUT}\nand ({MEDICAL_INPUT}\nor {DEPENDENT_CARE_INPUT})"
        rows = (
            ("not elected", (False, True, False), "not_holds"),
            ("elected medical", (True, True, False), "holds"),
            ("elected neither", (True, False, False), "not_holds"),
            ("elected dependent care", (True, False, True), "holds"),
        )
    else:
        inputs = (MEDICAL_INPUT, DEPENDENT_CARE_INPUT)
        formula = f"{MEDICAL_INPUT}\nor {DEPENDENT_CARE_INPUT}"
        rows = (
            ("neither", (False, False), "not_holds"),
            ("medical", (True, False), "holds"),
            ("dependent care", (False, True), "holds"),
        )
    rule = _rule(
        DEDUCTION_OUTPUT,
        "99 CFR 990.99(a); 99 CFR 990.99(a)(1); 99 CFR 990.99(a)(2)",
        formula,
        [_atom(ELECTION_CHAPEAU), _atom(MEDICAL), _atom(DEPENDENT_CARE)],
    )
    cases = [
        _case(name, dict(zip(inputs, values)), {DEDUCTION_OUTPUT: result})
        for name, values, result in rows
    ]
    return _doc(inputs, [rule]), cases


def _issues(doc: dict, cases: list[dict], source: str = SOURCE) -> tuple[str, ...]:
    content = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
    profile = validator_module._numeric_profile_for_citation_path(CITATION)
    bindings = completeness_module.collect_artifact_numeric_bindings(
        content,
        extract_named_scalars=validator_module.extract_named_scalar_occurrences,
        imported_symbol_contents=(),
    )
    result = completeness_module.analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=CITATION,
        source_context={CITATION: source},
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            validator_module.extract_typed_numeric_inventory_occurrences_from_text,
            profile=profile,
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            validator_module.extract_typed_numeric_occurrences_from_text,
            profile=profile,
        ),
        extract_named_scalars=validator_module.extract_named_scalar_occurrences,
        numeric_value_is_grounded=validator_module.numeric_value_is_grounded,
        artifact_numeric_values=tuple(value for _name, value in bindings),
        artifact_numeric_bindings=bindings,
    )
    return tuple(str(issue) for issue in result.issues)


def _chapeau_is_unpaired(
    found: tuple[str, ...], chapeau_text: str = "if they are:"
) -> bool:
    return any(
        "paired positive/blocking" in issue and chapeau_text in issue for issue in found
    )


def _branch(path: tuple[str, ...]) -> completeness_module.SourceStructureBranch:
    branches = completeness_module.recognize_source_structure(
        NESTED_SOURCE, corpus_citation_path=CITATION
    )
    return next(branch for branch in branches if branch.path == path)


def _relevant(condition: str, *, item_texts: tuple[str, ...] = ()) -> bool:
    return completeness_module._source_exception_selector_is_relevant(
        condition, SSI_INPUT, item_texts=item_texts
    )


def test_list_chapeau_condition_is_only_the_chapeau():
    assert (
        completeness_module._source_exception_condition_text(CHAPEAU) == "if they are:"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (SOURCE, [("a",), ("a", "1"), ("a", "2"), ("b",)]),
        (
            NESTED_SOURCE,
            [
                ("a",),
                ("a", "1"),
                ("a", "2"),
                ("a", "2", "i"),
                ("a", "2", "ii"),
                ("b",),
            ],
        ),
        (ELECTION_SOURCE, [("a",), ("a", "1"), ("a", "2"), ("b",)]),
    ),
)
def test_sources_place_each_chapeau_above_its_items(
    source: str, expected: list[tuple[str, ...]]
):
    paths = [
        branch.path
        for branch in completeness_module.recognize_source_structure(
            source, corpus_citation_path=CITATION
        )
    ]

    assert paths == expected


def test_list_chapeau_accepts_selector_named_by_cited_item():
    condition = completeness_module._source_exception_condition_text(CHAPEAU)

    assert completeness_module._source_exception_selector_is_relevant(
        condition,
        NOT_FINISHED_INPUT,
        item_texts=(NOT_FINISHED,),
    )
    assert completeness_module._source_exception_selector_is_relevant(
        condition,
        APPROVED_INPUT,
        item_texts=(APPROVED, NOT_FINISHED),
    )


def test_list_chapeau_rejects_selector_no_cited_item_names():
    condition = completeness_module._source_exception_condition_text(CHAPEAU)

    assert not completeness_module._source_exception_selector_is_relevant(
        condition, NOT_FINISHED_INPUT
    )
    assert not completeness_module._source_exception_selector_is_relevant(
        condition,
        "person_owns_luxury_car",
        item_texts=(APPROVED, NOT_FINISHED),
    )
    # "hours" appears only in the chapeau and "finished" only in the item, so
    # no single cited item names this selector.
    assert not completeness_module._source_exception_selector_is_relevant(
        condition,
        "hours_not_finished_by_course_deadline",
        item_texts=(NOT_FINISHED,),
        supporting_texts=(CHAPEAU, NOT_FINISHED),
    )


def test_list_chapeau_ignores_excerpts_from_outside_its_items():
    condition = completeness_module._source_exception_condition_text(CHAPEAU)

    # The rule cites the item, but not as one of this chapeau's items.
    assert not completeness_module._source_exception_selector_is_relevant(
        condition,
        NOT_FINISHED_INPUT,
        supporting_texts=(CHAPEAU, APPROVED, NOT_FINISHED),
    )


@pytest.mark.parametrize(
    "chapeau",
    (
        "The amount is reduced when the payments are—",
        "A household is exempt if its members are:",
        "The rule applies if the person is:",
    ),
)
def test_when_and_if_chapeaus_share_the_rule(chapeau: str):
    condition = completeness_module._source_exception_condition_text(chapeau)

    assert _relevant(condition, item_texts=(SSI_ITEM,))
    assert not _relevant(condition, item_texts=("receives a pension",))


@pytest.mark.parametrize(
    "condition",
    (
        "if they are:",
        "if it is:",
        "if he or she is—",
        "when the payments are—",
        "if such payments are –",
        "If The Person Is:",
        "if it would be:",
        "if each member is:",
        "when its members are:",
        "if the insured is:",
    ),
)
def test_list_chapeau_with_a_bare_subject_takes_its_items(condition: str):
    assert _relevant(condition, item_texts=(SSI_ITEM,))
    assert not _relevant(condition)


@pytest.mark.parametrize(
    "condition",
    (
        # Clauses that state a condition of their own.
        ELECTION_CHAPEAU,
        "when a member who is elderly or disabled files a request within "
        "30 days and the request is:",
        "if the person is a resident and the payments are:",
        "if the person and the payments are:",
        "if, when the payments are made, they are:",
        # A list chapeau may not hide behind an earlier condition.
        "if the household so elects, when they are:",
        # A modifier or a verb in the subject can carry a condition.
        "if the disabled member's payments are:",
        "if the noncustodial parent's payments are:",
        "if the late payments are:",
        "if such late payments are:",
        "if the household's countable liquid resources are:",
        "if the household elects to be:",
        "if the member chooses to be:",
        "if they elect to be:",
        # A relative pronoun opens a clause that runs past the list.
        "if those who are:",
        "if that which is:",
        # Shapes this rule leaves as they were.
        "if they are not:",
        "if any of the following apply:",
        "if—",
    ),
)
def test_other_if_clauses_must_name_the_selector_themselves(condition: str):
    assert not _relevant(condition, item_texts=(SSI_ITEM,))


@pytest.mark.parametrize(
    "condition",
    (
        "if the except are:",
        "if such unless is:",
        "when the all-except-one are:",
        "if the except's are:",
    ),
)
def test_a_clause_with_unless_or_except_before_its_verb_is_no_list_chapeau(
    condition: str,
):
    # The check reads such a clause from every excerpt its rule cites, as it
    # reads an unless list, so the clause can't take its items as well.
    assert not completeness_module._IF_LIST_CHAPEAU_PATTERN.fullmatch(condition)
    assert not _relevant(condition, item_texts=(SSI_ITEM,))
    assert completeness_module._source_exception_selector_is_relevant(
        condition, SSI_INPUT, supporting_texts=(SSI_ITEM,)
    )


@pytest.mark.parametrize(
    "condition",
    ("if the exceptions are:", "if such exceptional are:", "if the sunless are:"),
)
def test_a_subject_that_only_holds_the_letters_is_still_a_bare_subject(
    condition: str,
):
    # "exceptions" starts with "except" and "sunless" ends with "unless", but
    # neither holds the word itself.
    assert completeness_module._IF_LIST_CHAPEAU_PATTERN.fullmatch(condition)
    assert _relevant(condition, item_texts=(SSI_ITEM,))
    assert not completeness_module._source_exception_selector_is_relevant(
        condition, SSI_INPUT, supporting_texts=(SSI_ITEM,)
    )


@pytest.mark.parametrize(
    ("condition", "selector"),
    (
        # A list chapeau whose subject names the selector.
        ("if such resources are:", "liquid_resources"),
        # Not a list chapeau: its subject holds four words.
        (
            "if the household's countable liquid resources are:",
            "countable_liquid_resources",
        ),
    ),
)
def test_chapeau_that_names_the_selector_still_matches(condition: str, selector: str):
    assert completeness_module._source_exception_selector_is_relevant(
        condition, selector
    )
    assert completeness_module._source_exception_selector_is_relevant(
        condition, selector, item_texts=("an unrelated phrase",)
    )


def test_rule_excerpts_under_branch_keeps_the_branchs_own_items():
    rule = _rule(
        OUTPUT,
        "99 CFR 990.99(a)",
        "true",
        [
            _atom("must complete forty course hours", kind="predicate"),
            _atom("renewal tally leaves out course hours"),
            _atom(APPROVED),
            _atom(NOT_FINISHED),
            _atom(NOT_FINISHED, citation="us/regulation/99/990/98"),
            _atom("text that appears nowhere in the source"),
        ],
    )

    def under(path: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            item.excerpt
            for item in completeness_module._rule_excerpts_under_branch(
                rule,
                _branch(path),
                source_branches=completeness_module.recognize_source_structure(
                    NESTED_SOURCE, corpus_citation_path=CITATION
                ),
                corpus_citation_path=CITATION,
            )
        )

    assert under(("a", "2")) == (APPROVED, NOT_FINISHED)
    # (a)'s first child, (a)(1), ends a sentence, so it alone would be the
    # items of a list under (a). The chapeau and its items sit in (a)(2).
    assert under(("a",)) == ("must complete forty course hours",)
    assert under(("a", "2", "ii")) == ()
    assert under(("b",)) == ()


def test_each_item_excerpt_carries_its_branchs_wording_through_it():
    rule = _rule(
        OUTPUT,
        "99 CFR 990.99(a)(2)",
        "true",
        [_atom(NOT_FINISHED), _atom("finished by the course deadline")],
    )
    branches = completeness_module.recognize_source_structure(
        NESTED_SOURCE, corpus_citation_path=CITATION
    )

    found = completeness_module._rule_excerpts_under_branch(
        rule,
        _branch(("a", "2")),
        source_branches=branches,
        corpus_citation_path=CITATION,
    )

    # The second excerpt leaves off the item's "Not", and its wording keeps it.
    assert found == (
        completeness_module._ListItemExcerpt(NOT_FINISHED, f"(ii) {NOT_FINISHED}"),
        completeness_module._ListItemExcerpt(
            "finished by the course deadline", f"(ii) {NOT_FINISHED}"
        ),
    )


def _item_paths(
    source: str, path: tuple[str, ...], *, start_at: str | None = None
) -> frozenset[tuple[str, ...]]:
    branches = completeness_module.recognize_source_structure(
        source, corpus_citation_path=CITATION
    )
    if path:
        branch = next(branch for branch in branches if branch.path == path)
    else:
        branch = completeness_module.SourceStructureBranch(
            (), "source-unit", "source unit", source, 0, len(source)
        )
    if start_at is not None:
        offset = source.index(start_at)
        branch = completeness_module.SourceStructureBranch(
            branch.path, branch.kind, branch.label, branch.text, offset, branch.end
        )
    return completeness_module._chapeau_list_item_paths(branch, branches)


def test_list_items_run_through_the_first_that_ends_a_sentence():
    assert [
        branch.path
        for branch in completeness_module.recognize_source_structure(
            SIBLING_SOURCE, corpus_citation_path=CITATION
        )
    ] == [("a",), ("a", "1"), ("a", "2"), ("a", "3"), ("b",)]
    assert _item_paths(SIBLING_SOURCE, ("a",)) == {("a", "1"), ("a", "2")}
    assert _item_paths(ROOT_SOURCE, ()) == {("a",), ("b",)}
    assert _item_paths(NESTED_SOURCE, ("a", "2")) == {
        ("a", "2", "i"),
        ("a", "2", "ii"),
    }
    assert _item_paths(SECTION_SOURCE, ("c", "2")) == {
        ("c", "2", "i"),
        ("c", "2", "ii"),
        ("c", "2", "iii"),
        ("c", "2", "iv"),
    }


def test_a_sub_item_that_ends_a_sentence_does_not_end_the_list():
    # (1)(i) ends a sentence, but (1) ends with "; or", so the list runs on.
    source = (
        "(a) Hours left out. The tally leaves out hours if they are:\n\n"
        "(1) Earned in an approved course, where—\n\n"
        "(i) The board approved it in advance.\n\n"
        "(ii) The board approved it in writing; or\n\n"
        "(2) Not finished by the course deadline.\n\n"
        "(b) Records. The board must keep records.\n"
    )

    assert _item_paths(source, ("a",)) == {("a", "1"), ("a", "2")}


@pytest.mark.parametrize(
    ("text", "excerpt", "wording"),
    (
        # The wording stops at the excerpt's end, so a later mention of the
        # same words doesn't change how the excerpt reads.
        (
            "(1) Not subject to withholding, or subject to withholding at half "
            "the rate; or",
            "Not subject to withholding",
            "(1) Not subject to withholding",
        ),
        # Whitespace collapses on both sides, as it did when the excerpt was
        # located.
        (
            "(1)  Not\n\nsubject to   withholding; or",
            "subject to\nwithholding",
            "(1) Not subject to withholding",
        ),
        # The excerpt's first match counts.
        ("(1) Paid in cash, or paid in kind; or", "aid in", "(1) Paid in"),
        # Text that doesn't hold the excerpt gives the excerpt itself.
        (
            "(1) Not subject to withholding; or",
            "reported  to the agency",
            "reported to the agency",
        ),
    ),
)
def test_an_items_wording_runs_from_its_start_through_the_excerpt(
    text: str, excerpt: str, wording: str
):
    assert completeness_module._wording_through_excerpt(text, excerpt) == wording


def test_the_cited_words_themselves_must_name_the_selector():
    # The rule cites only "the course deadline" from item (2). The item names
    # `not_finished_by_course_deadline` in full, but the cited words don't.
    doc, cases = _toggle_one(
        NOT_FINISHED_INPUT,
        [_atom(CHAPEAU_EXCERPT), _atom("the course deadline")],
        "99 CFR 990.99(a); 99 CFR 990.99(a)(2)",
    )

    assert _chapeau_is_unpaired(_issues(doc, cases))


def test_a_list_with_no_end_has_no_items():
    assert _item_paths(UNENDED_SOURCE, ("a",)) == frozenset()


def test_an_item_with_no_ending_leaves_the_list_without_items():
    assert [
        branch.path
        for branch in completeness_module.recognize_source_structure(
            RUN_ON_SOURCE, corpus_citation_path=CITATION
        )
    ] == [("a",), ("a", "1"), ("a", "2"), ("a", "3"), ("b",)]
    # (2) ends with neither a separator nor a sentence end, so the list can't
    # run on through (3), which does end a sentence.
    assert _item_paths(RUN_ON_SOURCE, ("a",)) == frozenset()


@pytest.mark.parametrize(
    ("first", "second"),
    ((",", ", or"), (";", "; and/or"), (", and", ";"), ("; And", ",")),
)
def test_list_items_may_end_with_any_list_separator(first: str, second: str):
    source = (
        "(a) Hours left out. The tally leaves out hours if they are:\n\n"
        f"(1) Earned in an approved course{first}\n\n"
        f"(2) Reported by mistake{second}\n\n"
        "(3) Not finished.\n\n(b) Records. The board must keep records.\n"
    )

    assert _item_paths(source, ("a",)) == {("a", "1"), ("a", "2"), ("a", "3")}


def test_children_before_the_chapeau_are_not_its_items():
    # (a)(1) and (a)(2) both start before the chapeau, which sits in (a)(2).
    assert _item_paths(NESTED_SOURCE, ("a",), start_at=CHAPEAU) == frozenset()


def test_an_unended_item_with_an_inner_comma_ends_the_list():
    # (2) has lost its ending. Its comma sits inside the item, so it isn't a
    # separator, and the list can't run on through (3).
    source = (
        f"{HEADING} {CHAPEAU}\n\n(1) {APPROVED}; or\n\n"
        f"(2) {NOT_FINISHED}, or finished after an extension\n\n"
        f"(3) Records. {RECORDS}.\n\n{FEES}"
    )

    assert _item_paths(source, ("a",)) == frozenset()


def test_a_colon_is_not_a_list_separator():
    # (1) introduces a sub-list that the text lacks, so it ends with neither a
    # separator nor a sentence end.
    source = (
        f"(a) Excluded payments. {EXCLUSION_CHAPEAU}\n\n"
        "(1) Received from any of the following sources:\n\n"
        f"(2) {EMPLOYER}.\n\n{PAYMENT_RECORDS}"
    )

    assert _item_paths(source, ("a",)) == frozenset()


def test_a_bare_outline_list_keeps_its_items():
    # The labels stand on lines of their own, and each item's text keeps the
    # newline before the next label, so a separator may come before trailing
    # whitespace.
    source = (
        "1\nA payment is excluded from income if it is—\n"
        "a\none that is not reported to the agency; or\n"
        "b\none that is paid by a former employer.\n"
        "2\nThe agency must keep a record of each payment.\n"
    )

    assert _item_paths(source, ("1",)) == {("1", "a"), ("1", "b")}


@pytest.mark.parametrize(
    ("item", "excerpt"),
    (
        ('Received from a "pension fund;" or', 'Received from a "pension fund'),
        ("Received from a “pension fund,” or", "Received from a “pension fund"),
    ),
)
def test_a_separator_before_a_closing_quote_leaves_the_list_without_items(
    item: str, excerpt: str
):
    # A known limit: a sentence end may come before a closing quote, but a
    # list separator may not. So (1) ends with neither, the list has no items,
    # and the chapeau stays unpaired, as on main.
    source = (
        f"(a) Excluded payments. {EXCLUSION_CHAPEAU}\n\n(1) {item}\n\n"
        f"(2) {EMPLOYER}.\n\n{PAYMENT_RECORDS}"
    )
    doc, cases = _exclusion_material(
        (PENSION_INPUT, EMPLOYER_INPUT),
        excluded_when=True,
        item_atoms=(excerpt, EMPLOYER),
    )

    assert _item_paths(source, ("a",)) == frozenset()
    assert _chapeau_is_unpaired(_issues(doc, cases, source))


def test_cited_item_selectors_pair_the_list_chapeau():
    doc, cases = _material()

    assert not _chapeau_is_unpaired(_issues(doc, cases))


def test_cited_item_selectors_pair_a_nested_list_chapeau():
    doc, cases = _nested_material(pair_an_item=True)

    assert not _chapeau_is_unpaired(_issues(doc, cases, NESTED_SOURCE))


def test_parent_rule_pair_does_not_pair_a_nested_list_chapeau():
    # The parent rule cites (a), so it governs the (a)(2) chapeau, and its
    # (a)(1) excerpt names `keep_active_license` in full. That excerpt says
    # nothing about the hours left out, so its pair proves no exclusion.
    doc, cases = _nested_material(pair_an_item=False)

    assert _chapeau_is_unpaired(_issues(doc, cases, NESTED_SOURCE))


@pytest.mark.parametrize(
    ("excerpt", "citation", "selector"),
    (
        # Paragraph (b) of the same section, outside the chapeau's items.
        (FEE_EXCERPT, CITATION, FEE_INPUT),
        # Another provision's text.
        (
            "if the applicant is deceased",
            "us/regulation/99/990/2",
            "applicant_is_deceased",
        ),
    ),
)
def test_excerpts_outside_the_items_do_not_pair_the_list_chapeau(
    excerpt: str, citation: str, selector: str
):
    rule = _rule(
        OUTPUT,
        "99 CFR 990.99(a); 99 CFR 990.99(a)(1); 99 CFR 990.99(a)(2)",
        selector,
        [
            _atom(CHAPEAU_EXCERPT),
            _atom(APPROVED),
            _atom(NOT_FINISHED),
            _atom(excerpt, citation=citation),
        ],
    )
    cases = [
        _case("ordinary hour counts", {selector: False}, {OUTPUT: "not_holds"}),
        _case("toggled hour is left out", {selector: True}, {OUTPUT: "holds"}),
    ]

    assert _chapeau_is_unpaired(_issues(_doc((selector,), [rule]), cases))


def test_election_pair_satisfies_a_chapeau_with_its_own_condition():
    doc, cases = _election_material(pair_the_election=True)

    assert not _chapeau_is_unpaired(_issues(doc, cases, ELECTION_SOURCE), "so elects")


def test_item_pairs_do_not_stand_in_for_the_chapeaus_own_condition():
    # The formula drops the election, and only the deduction items are paired.
    doc, cases = _election_material(pair_the_election=False)

    assert _chapeau_is_unpaired(_issues(doc, cases, ELECTION_SOURCE), "so elects")


def test_list_chapeau_stays_unpaired_when_no_excerpt_names_the_selector():
    doc, cases = _material(
        approved_input="hours_earned_in_course_board_approved_in_writing",
        not_finished_input="hours_not_finished_by_course_deadline",
    )

    assert _chapeau_is_unpaired(_issues(doc, cases))


def test_list_chapeau_stays_unpaired_when_its_items_are_not_cited():
    doc, cases = _material(item_atoms=())

    assert _chapeau_is_unpaired(_issues(doc, cases))


def test_list_chapeau_stays_unpaired_without_a_case_pair():
    doc, cases = _material()

    assert _chapeau_is_unpaired(_issues(doc, cases[:1]))


def _toggle_one(
    selector: str, atoms: list[dict], source_ref: str
) -> tuple[dict, list[dict]]:
    rule = _rule(OUTPUT, source_ref, selector, atoms)
    cases = [
        _case("ordinary hour counts", {selector: False}, {OUTPUT: "not_holds"}),
        _case("toggled hour is left out", {selector: True}, {OUTPUT: "holds"}),
    ]
    return _doc((selector,), [rule]), cases


def test_a_paragraph_after_the_list_does_not_pair_the_chapeau():
    doc, cases = _toggle_one(
        RECORDS_INPUT,
        [_atom(CHAPEAU_EXCERPT), _atom(APPROVED), _atom(NOT_FINISHED), _atom(RECORDS)],
        "99 CFR 990.99(a); 99 CFR 990.99(a)(1); 99 CFR 990.99(a)(2); "
        "99 CFR 990.99(a)(3)",
    )

    assert _chapeau_is_unpaired(_issues(doc, cases, SIBLING_SOURCE))


def test_items_before_a_later_paragraph_still_pair_the_chapeau():
    doc, cases = _material()

    assert not _chapeau_is_unpaired(_issues(doc, cases, SIBLING_SOURCE))


def test_a_paragraph_after_a_unit_level_list_does_not_pair_the_chapeau():
    doc, cases = _toggle_one(
        FEE_INPUT,
        [
            _atom(CHAPEAU_EXCERPT),
            _atom(APPROVED),
            _atom(NOT_FINISHED),
            _atom(FEE_EXCERPT),
        ],
        "99 CFR 990.99; 99 CFR 990.99(a); 99 CFR 990.99(b); 99 CFR 990.99(c)",
    )

    assert _chapeau_is_unpaired(_issues(doc, cases, ROOT_SOURCE))


def test_items_of_a_unit_level_list_pair_the_chapeau():
    doc, cases = _material()
    doc["rules"][0]["source"] = "99 CFR 990.99; 99 CFR 990.99(a); 99 CFR 990.99(b)"

    assert not _chapeau_is_unpaired(_issues(doc, cases, ROOT_SOURCE))


def test_a_paragraph_after_an_unended_list_does_not_pair_the_chapeau():
    doc, cases = _toggle_one(
        RECORDS_INPUT,
        [_atom(CHAPEAU_EXCERPT), _atom(APPROVED), _atom(NOT_FINISHED), _atom(RECORDS)],
        "99 CFR 990.99(a); 99 CFR 990.99(a)(1); 99 CFR 990.99(a)(2); "
        "99 CFR 990.99(a)(3)",
    )

    assert _chapeau_is_unpaired(_issues(doc, cases, RUN_ON_SOURCE))


@pytest.mark.parametrize("ending", (";", "; and"))
def test_a_labeled_paragraph_after_a_separator_ended_item_joins_the_list(
    ending: str,
):
    # A known limit: (2) ends with a list separator, so the list runs on
    # through (3), the first child that ends a sentence, and a pair on the
    # records paragraph alone pairs the chapeau. Punctuation alone can't tell
    # a heading such as "Records." from an item's own first sentence.
    source = (
        f"{HEADING} {CHAPEAU}\n\n(1) {APPROVED}; or\n\n"
        f"(2) {NOT_FINISHED}{ending}\n\n(3) Records. {RECORDS}.\n\n{FEES}"
    )
    doc, cases = _toggle_one(
        RECORDS_INPUT,
        [_atom(CHAPEAU_EXCERPT), _atom(APPROVED), _atom(NOT_FINISHED), _atom(RECORDS)],
        "99 CFR 990.99(a); 99 CFR 990.99(a)(1); 99 CFR 990.99(a)(2); "
        "99 CFR 990.99(a)(3)",
    )

    assert _item_paths(source, ("a",)) == {("a", "1"), ("a", "2"), ("a", "3")}
    assert not _chapeau_is_unpaired(_issues(doc, cases, source))


def test_an_item_with_a_second_sentence_stays_one_item():
    # Item (2) ends a sentence and adds a second one. Its first sentence reads
    # like a heading such as "Records.", so a rule that ended the list at a
    # heading would drop the whole list.
    source = (
        f"(a) Excluded payments. {EXCLUSION_CHAPEAU}\n\n(1) {PENSION}; or\n\n"
        f"(2) {EMPLOYER}. A successor employer counts as a former employer.\n\n"
        f"{PAYMENT_RECORDS}"
    )
    doc, cases = _exclusion_material(
        (PENSION_INPUT, EMPLOYER_INPUT),
        excluded_when=True,
        item_atoms=(PENSION, EMPLOYER),
    )

    assert _item_paths(source, ("a",)) == {("a", "1"), ("a", "2")}
    assert not _chapeau_is_unpaired(_issues(doc, cases, source))


def test_text_after_the_last_item_counts_as_that_item():
    # A known limit: the outline gives unlabeled text after the last item no
    # branch of its own, so that text falls in the last item's branch, and a
    # pair on it pairs the chapeau.
    doc, cases = _toggle_one(
        RECORDS_INPUT,
        [_atom(CHAPEAU_EXCERPT), _atom(APPROVED), _atom(NOT_FINISHED), _atom(RECORDS)],
        "99 CFR 990.99(a); 99 CFR 990.99(a)(1); 99 CFR 990.99(a)(2)",
    )

    assert not _chapeau_is_unpaired(_issues(doc, cases, FLUSH_SOURCE))


def test_text_after_a_last_item_that_opens_with_a_cue_gives_no_value():
    # A known limit: the text after the last item falls in that item's
    # branch, so its wording through the excerpt starts with the item's "Not",
    # which stands before the name's words. The reader doesn't read "Not" past
    # the sentence's end, so the item gives no value, though the excerpt reads
    # as positive. The chapeau stays unpaired, as on main.
    doc, cases = _toggle_one(
        RECORDS_INPUT,
        [_atom(CHAPEAU_EXCERPT), _atom(APPROVED), _atom(NOT_FINISHED), _atom(RECORDS)],
        "99 CFR 990.99(a); 99 CFR 990.99(a)(1); 99 CFR 990.99(a)(2)",
    )

    assert _chapeau_is_unpaired(_issues(doc, cases, FLUSH_AFTER_CUE_SOURCE))


def test_a_list_with_no_end_keeps_its_chapeau_unpaired():
    doc, cases = _material()

    assert _chapeau_is_unpaired(_issues(doc, cases, UNENDED_SOURCE))


def test_another_units_copy_of_an_item_does_not_pair_the_chapeau():
    # The excerpt matches item (2)'s text, but the rule cites it from another
    # unit, and cites none of this unit's items.
    doc, cases = _toggle_one(
        NOT_FINISHED_INPUT,
        [
            _atom(CHAPEAU_EXCERPT),
            _atom(NOT_FINISHED, citation="us/regulation/99/990/98"),
        ],
        "99 CFR 990.99(a); 99 CFR 990.98",
    )

    assert _chapeau_is_unpaired(_issues(doc, cases))


def test_the_chapeaus_own_excerpt_does_not_pair_it():
    # The cited chapeau names this selector in full, but the chapeau isn't one
    # of its own items.
    doc, cases = _toggle_one(
        "leaves_out_course_hours",
        [_atom(CHAPEAU_EXCERPT), _atom(APPROVED), _atom(NOT_FINISHED)],
        "99 CFR 990.99(a); 99 CFR 990.99(a)(1); 99 CFR 990.99(a)(2)",
    )

    assert _chapeau_is_unpaired(_issues(doc, cases))


def test_a_deeper_branch_under_another_paragraph_does_not_pair_the_chapeau():
    # (b)(1) lies deeper than the chapeau's paragraph (a), but not under it.
    source = (
        f"{HEADING} {CHAPEAU}\n\n(1) {APPROVED}; or\n\n(2) {NOT_FINISHED}."
        f"\n\n(b) Records—(1) In general. {RECORDS}.\n\n(2) Retention. The "
        "board must keep each record for three years.\n\n(c) Reserved.\n"
    )
    doc, cases = _toggle_one(
        RECORDS_INPUT,
        [_atom(CHAPEAU_EXCERPT), _atom(APPROVED), _atom(NOT_FINISHED), _atom(RECORDS)],
        "99 CFR 990.99(a); 99 CFR 990.99(a)(1); 99 CFR 990.99(a)(2); "
        "99 CFR 990.99(b)(1)",
    )

    assert _chapeau_is_unpaired(_issues(doc, cases, source))


def _disabled_material(*, pair_the_disability: bool) -> tuple[dict, list[dict]]:
    if pair_the_disability:
        inputs = (DISABLED_INPUT, PENSION_INPUT, EMPLOYER_INPUT)
        formula = f"{DISABLED_INPUT}\nand ({PENSION_INPUT}\nor {EMPLOYER_INPUT})"
        rows = (
            ("not disabled", (False, True, False), "not_holds"),
            ("disabled, pension", (True, True, False), "holds"),
            ("disabled, neither", (True, False, False), "not_holds"),
            ("disabled, employer", (True, False, True), "holds"),
        )
    else:
        inputs = (PENSION_INPUT, EMPLOYER_INPUT)
        formula = f"{PENSION_INPUT}\nor {EMPLOYER_INPUT}"
        rows = (
            ("neither", (False, False), "not_holds"),
            ("pension", (True, False), "holds"),
            ("employer", (False, True), "holds"),
        )
    rule = _rule(
        PAYMENT_OUTPUT,
        "99 CFR 990.99(a); 99 CFR 990.99(a)(1); 99 CFR 990.99(a)(2)",
        formula,
        [_atom(DISABLED_CHAPEAU_EXCERPT), _atom(PENSION), _atom(EMPLOYER)],
    )
    cases = [
        _case(name, dict(zip(inputs, values)), {PAYMENT_OUTPUT: result})
        for name, values, result in rows
    ]
    return _doc(inputs, [rule]), cases


def test_item_pairs_do_not_stand_in_for_a_modifier_in_the_subject():
    # The formula drops "disabled", and only the payment items are paired.
    doc, cases = _disabled_material(pair_the_disability=False)

    assert _chapeau_is_unpaired(
        _issues(doc, cases, DISABLED_SOURCE), "disabled member's payments"
    )


def test_a_pair_on_the_subjects_modifier_satisfies_the_clause():
    doc, cases = _disabled_material(pair_the_disability=True)

    assert not _chapeau_is_unpaired(
        _issues(doc, cases, DISABLED_SOURCE), "disabled member's payments"
    )


def test_one_word_subject_is_read_as_a_bare_subject():
    # A known limit: "disabled" carries a condition here too, but a single
    # word after the determiner reads as the subject, so an item pairs it.
    assert _relevant("if the disabled are:", item_texts=(SSI_ITEM,))


def _exclusion_material(
    names: tuple[str, str], *, excluded_when: bool, item_atoms: tuple[str, str]
) -> tuple[dict, list[dict]]:
    """Exclude a payment when either selector equals ``excluded_when``."""

    first, second = names
    negation = "" if excluded_when else "not "
    rule = _rule(
        PAYMENT_OUTPUT,
        "99 CFR 990.99(a); 99 CFR 990.99(a)(1); 99 CFR 990.99(a)(2)",
        f"{negation}{first}\nor {negation}{second}",
        [_atom(EXCLUSION_CHAPEAU), *(_atom(excerpt) for excerpt in item_atoms)],
    )
    counted = not excluded_when
    rows = (
        ("ordinary payment counts", (counted, counted), "not_holds"),
        ("first item excludes the payment", (excluded_when, counted), "holds"),
        ("second item excludes the payment", (counted, excluded_when), "holds"),
    )
    cases = [
        _case(name, dict(zip(names, values)), {PAYMENT_OUTPUT: result})
        for name, values, result in rows
    ]
    return _doc(names, [rule]), cases


@pytest.mark.parametrize(
    ("source", "item_atoms", "names", "excluded_when", "faithful"),
    (
        pytest.param(
            NEGATIVE_ITEMS_SOURCE,
            (UNWITHHELD, UNREPORTED),
            ("subject_to_withholding", "reported_to_agency"),
            False,
            True,
            id="negative-items-positive-names-faithful",
        ),
        pytest.param(
            NEGATIVE_ITEMS_SOURCE,
            (UNWITHHELD, UNREPORTED),
            ("subject_to_withholding", "reported_to_agency"),
            True,
            False,
            id="negative-items-positive-names-inverted",
        ),
        pytest.param(
            NEGATIVE_ITEMS_SOURCE,
            (UNWITHHELD, UNREPORTED),
            ("not_subject_to_withholding", "not_reported_to_agency"),
            True,
            True,
            id="negative-items-negative-names-faithful",
        ),
        pytest.param(
            NEGATIVE_ITEMS_SOURCE,
            (UNWITHHELD, UNREPORTED),
            ("not_subject_to_withholding", "not_reported_to_agency"),
            False,
            False,
            id="negative-items-negative-names-inverted",
        ),
        # The excerpts leave off each item's "Not", which the wording keeps.
        pytest.param(
            NEGATIVE_ITEMS_SOURCE,
            ("subject to withholding", "reported to the agency"),
            ("subject_to_withholding", "reported_to_agency"),
            False,
            True,
            id="negative-items-short-excerpts-faithful",
        ),
        pytest.param(
            NEGATIVE_ITEMS_SOURCE,
            ("subject to withholding", "reported to the agency"),
            ("subject_to_withholding", "reported_to_agency"),
            True,
            False,
            id="negative-items-short-excerpts-inverted",
        ),
        pytest.param(
            POSITIVE_ITEMS_SOURCE,
            (PENSION, EMPLOYER),
            (PENSION_INPUT, EMPLOYER_INPUT),
            True,
            True,
            id="positive-items-positive-names-faithful",
        ),
        pytest.param(
            POSITIVE_ITEMS_SOURCE,
            (PENSION, EMPLOYER),
            (PENSION_INPUT, EMPLOYER_INPUT),
            False,
            False,
            id="positive-items-positive-names-inverted",
        ),
        pytest.param(
            POSITIVE_ITEMS_SOURCE,
            (PENSION, EMPLOYER),
            ("not_received_from_pension_fund", "not_paid_by_former_employer"),
            False,
            True,
            id="positive-items-negative-names-faithful",
        ),
        pytest.param(
            POSITIVE_ITEMS_SOURCE,
            (PENSION, EMPLOYER),
            ("not_received_from_pension_fund", "not_paid_by_former_employer"),
            True,
            False,
            id="positive-items-negative-names-inverted",
        ),
    ),
)
def test_the_items_wording_sets_the_direction_an_exclusion_needs(
    source: str,
    item_atoms: tuple[str, str],
    names: tuple[str, str],
    excluded_when: bool,
    faithful: bool,
):
    doc, cases = _exclusion_material(
        names, excluded_when=excluded_when, item_atoms=item_atoms
    )

    assert _chapeau_is_unpaired(_issues(doc, cases, source)) is not faithful


def _items(
    *specs: str | tuple[str, str],
) -> tuple[completeness_module._ListItemExcerpt, ...]:
    """Item excerpts, each with its own wording or "(1)" and the excerpt."""

    return tuple(
        completeness_module._ListItemExcerpt(*spec)
        if isinstance(spec, tuple)
        else completeness_module._ListItemExcerpt(spec, f"(1) {spec}")
        for spec in specs
    )


@pytest.mark.parametrize(
    ("selector", "specs", "expected"),
    (
        ("subject_to_withholding", (UNWITHHELD,), {False}),
        ("not_subject_to_withholding", (UNWITHHELD,), {True}),
        ("received_from_pension_fund", (PENSION,), {True}),
        ("not_received_from_pension_fund", (PENSION,), {False}),
        # The wording keeps the "Not" that the excerpt leaves off.
        (
            "subject_to_withholding",
            (("subject to withholding", f"(1) {UNWITHHELD}"),),
            {False},
        ),
        # An item that doesn't name the selector gives no value, even when it
        # shares one of the name's words.
        ("subject_to_withholding", (UNWITHHELD, "Reported to the agency"), {False}),
        (PENSION_INPUT, ("Not received from an employer", PENSION), {True}),
        # Two items that name the selector may need different values.
        (
            "subject_to_withholding",
            (UNWITHHELD, "Subject to withholding at half the rate"),
            {False, True},
        ),
    ),
)
def test_each_item_that_names_the_selector_gives_the_value_it_needs(
    selector: str, specs: tuple[str | tuple[str, str], ...], expected: set[bool]
):
    assert (
        completeness_module._list_item_active_values(
            "if they are:", selector, _items(*specs)
        )
        == expected
    )


@pytest.mark.parametrize(
    "specs",
    ((), (UNREPORTED,), ("Reported to the agency", "Paid by a former employer")),
)
def test_with_no_item_naming_the_selector_the_chapeau_decides(
    specs: tuple[str, ...],
):
    own_reading = completeness_module._source_exception_selector_active_value(
        "if they are:", "subject_to_withholding"
    )

    assert completeness_module._list_item_active_values(
        "if they are:", "subject_to_withholding", _items(*specs)
    ) == {own_reading}


@pytest.mark.parametrize(
    ("condition", "selector", "item"),
    (
        ("if the disabled are:", "disabled", "Not disabled"),
        ("if such resources are:", "liquid_resources", "Not liquid resources"),
    ),
)
def test_a_chapeau_that_names_the_selector_keeps_its_own_reading(
    condition: str, selector: str, item: str
):
    own_reading = completeness_module._source_exception_selector_active_value(
        condition, selector
    )

    # Under a chapeau that doesn't name the selector, the item reads the other
    # way, so the chapeau's own wording is what decides here.
    assert completeness_module._list_item_active_values(
        "if they are:", selector, _items(item)
    ) == {not own_reading}
    assert completeness_module._list_item_active_values(
        condition, selector, _items(item)
    ) == {own_reading}


@pytest.mark.parametrize(
    ("selector", "item"),
    (
        # An acronym stands for its expansion.
        ("has_social_security_number", "Has an SSN"),
        # An inflected form counts: "reported" for "report".
        ("report_agency", "Reported to the agency"),
        # A longer word counts when it starts with the selector's word. The
        # rule is loose: "fundraiser" stands in for "fund".
        ("received_from_pension_fund", "Received from a pensioner's fundraiser"),
    ),
)
def test_an_item_names_a_selector_by_the_shared_word_rules(selector: str, item: str):
    assert completeness_module._source_exception_selector_is_relevant(
        "if they are:", selector, item_texts=(item,)
    )
    assert not completeness_module._source_exception_selector_is_relevant(
        "if they are:", selector
    )


def _item_source(item: str) -> str:
    """A list of excluded payments whose first item is ``item``."""

    return (
        f"(a) Excluded payments. {EXCLUSION_CHAPEAU}\n\n(1) {item}; or\n\n"
        f"(2) {EMPLOYER}.\n\n{PAYMENT_RECORDS}"
    )


def _lead_in_source(
    lead_in: str, first: str = "A pension fund", second: str = "An annuity"
) -> str:
    """A list whose first item opens with ``lead_in`` to a sub-list of its own."""

    return (
        f"(a) Excluded payments. {EXCLUSION_CHAPEAU}\n\n(1) {lead_in}\n\n"
        f"(i) {first}; or\n\n(ii) {second}; or\n\n"
        f"(2) {EMPLOYER}.\n\n{PAYMENT_RECORDS}"
    )


ITEM_REFS = "99 CFR 990.99(a); 99 CFR 990.99(a)(1); 99 CFR 990.99(a)(2)"
SUB_ITEM_REFS = (
    "99 CFR 990.99(a); 99 CFR 990.99(a)(1); 99 CFR 990.99(a)(1)(i); 99 CFR 990.99(a)(2)"
)
NOT_LEAD_IN = "Not received from any of the following:"


def _pairs(
    selector: str,
    excerpt: str,
    source: str,
    *,
    excluded_when: bool,
    source_ref: str = ITEM_REFS,
) -> bool:
    """Whether one case pair on ``selector`` pairs the chapeau.

    One rule cites the chapeau and ``excerpt`` and excludes a payment when
    ``selector`` equals ``excluded_when``.
    """

    negation = "" if excluded_when else "not "
    rule = _rule(
        PAYMENT_OUTPUT,
        source_ref,
        f"{negation}{selector}",
        [_atom(EXCLUSION_CHAPEAU), _atom(excerpt)],
    )
    cases = [
        _case(
            "ordinary payment counts",
            {selector: not excluded_when},
            {PAYMENT_OUTPUT: "not_holds"},
        ),
        _case(
            "the item excludes the payment",
            {selector: excluded_when},
            {PAYMENT_OUTPUT: "holds"},
        ),
    ]
    return not _chapeau_is_unpaired(_issues(_doc((selector,), [rule]), cases, source))


@pytest.mark.parametrize("excluded_when", (False, True), ids=("faithful", "inverted"))
@pytest.mark.parametrize(
    ("item", "selector"),
    (
        pytest.param("Never reported to the agency", "reported_to_agency", id="never"),
        pytest.param(
            "Received from a source other than a pension fund",
            PENSION_INPUT,
            id="other-than",
        ),
        pytest.param(
            "Received from any source except a pension fund",
            PENSION_INPUT,
            id="except",
        ),
        pytest.param(
            "Neither reported to the agency nor withheld",
            "reported_to_agency",
            id="neither-nor",
        ),
    ),
)
def test_an_item_with_a_negator_the_reader_reads_past_pairs_neither_encoding(
    item: str, selector: str, excluded_when: bool
):
    """An item worded with a negator the reader can't scope pairs neither way.

    Each item excludes a payment only when its positively named selector is
    false. The shared reader has no cue for "never", "neither", "nor",
    "except", "other than" or "unless" and would read each item as positive.
    So the item gives no value and adds no relevance, and the chapeau stays
    unpaired whichever way the formula reads the item, as on main.
    """

    assert not _pairs(selector, item, _item_source(item), excluded_when=excluded_when)


@pytest.mark.parametrize("excluded_when", (False, True), ids=("faithful", "inverted"))
@pytest.mark.parametrize(
    "excerpt",
    (
        pytest.param("A pension fund", id="sub-item"),
        pytest.param(
            "Not received from any of the following: (i) A pension fund",
            id="excerpt-runs-past-the-lead-in",
        ),
    ),
)
def test_a_not_in_a_parent_items_lead_in_pairs_neither_encoding(
    excerpt: str, excluded_when: bool
):
    """An excerpt with a negated lead-in before it pairs neither way.

    "(i) A pension fund" sits under "(1) Not received from any of the
    following:", so it excludes a payment only when ``pension_fund`` is false.
    The reader never applies that "Not": the sub-item's wording starts at
    "(i)", and in an excerpt that runs on into "(i)" the lead-in's colon ends
    the "Not"'s scope. A lead-in in the item that comes before some of the
    excerpt and carries a negation cue leaves the item without a value, so the
    chapeau stays unpaired whichever way the formula reads it, as on main.
    """

    assert not _pairs(
        "pension_fund",
        excerpt,
        _lead_in_source(NOT_LEAD_IN),
        excluded_when=excluded_when,
        source_ref=SUB_ITEM_REFS,
    )


@pytest.mark.parametrize("excluded_when", (False, True), ids=("faithful", "inverted"))
@pytest.mark.parametrize(
    ("item", "selector"),
    (
        ("Paid without withholding", "paid_with_withholding"),
        ("Paid to a person without a bank account", "person_has_bank_account"),
    ),
)
def test_an_item_whose_words_read_with_mixed_polarity_pairs_neither_encoding(
    item: str, selector: str, excluded_when: bool
):
    """An item that negates only some of the name's words pairs neither way.

    In "Paid without withholding", "paid" reads positive and "withholding"
    negative, so the reader takes no polarity from the item, and only a guess
    from the selector's name could set a value. The item gives none, so the
    chapeau stays unpaired whichever way the formula reads it, as on main.
    """

    assert not _pairs(selector, item, _item_source(item), excluded_when=excluded_when)


@pytest.mark.parametrize("excluded_when", (False, True))
@pytest.mark.parametrize("item", ("Has no SSN", "Has an SSN"))
def test_an_item_with_none_of_the_names_words_pairs_neither_encoding(
    item: str, excluded_when: bool
):
    """An item with none of the name's words pairs neither way.

    "SSN" names ``has_social_security_number`` by the shared acronym rule, but
    the reader looks for the name's own words, finds none, and takes no
    polarity, so only a guess from the name could set a value. Neither item
    gives one, and the chapeau stays unpaired either way, as on main. For "Has
    an SSN" that is a known limit, since the name's guess happens to be right
    there.
    """

    assert not _pairs(
        "has_social_security_number",
        item,
        _item_source(item),
        excluded_when=excluded_when,
    )


@pytest.mark.parametrize(
    ("wording", "selector"),
    (
        pytest.param("Never reported to the agency", "reported_to_agency", id="never"),
        pytest.param("Reported to neither agency", "reported_to_agency", id="neither"),
        pytest.param(
            "Not withheld, nor reported to the agency", "reported_to_agency", id="nor"
        ),
        pytest.param(
            "Received from any source except a pension fund", PENSION_INPUT, id="except"
        ),
        pytest.param(
            "Received from a source other than a pension fund",
            PENSION_INPUT,
            id="other-than",
        ),
        pytest.param(
            "Paid unless reported to the agency", "reported_to_agency", id="unless"
        ),
        pytest.param("Cannot claim withholding", "withholding", id="cannot"),
        pytest.param(
            "None of it reported to the agency", "reported_to_agency", id="none"
        ),
        pytest.param(
            "Nothing reported to the agency", "reported_to_agency", id="nothing"
        ),
        pytest.param(
            "Reported to nobody at the agency", "reported_to_agency", id="nobody"
        ),
        pytest.param(
            "Reported nowhere in the agency's files",
            "reported_to_agency",
            id="nowhere",
        ),
    ),
)
def test_each_negator_the_reader_reads_past_leaves_an_item_without_a_value(
    wording: str, selector: str
):
    # Each wording holds one such negator. The reader on its own reads each
    # as positive, so the item is left out: it gives no value and adds no
    # relevance.
    assert completeness_module._source_exception_selector_wording_value(
        wording, selector
    )
    assert completeness_module._list_item_values(selector, _items(wording)) == ()


def test_a_negator_the_excerpt_leaves_off_still_counts():
    # The check reads the item's wording through the excerpt, so the "Never"
    # that the excerpt "reported to the agency" leaves off still leaves the
    # item without a value, as a "Not" left off still sets one.
    assert (
        completeness_module._list_item_values(
            "reported_to_agency",
            _items(("reported to the agency", "(1) Never reported to the agency")),
        )
        == ()
    )


def test_an_item_without_a_value_leaves_the_other_items_values_alone():
    # "Never subject to withholding" gives no value, so the chapeau takes only
    # the value that "Not subject to withholding" needs.
    assert completeness_module._list_item_active_values(
        "if they are:",
        "subject_to_withholding",
        _items("Not subject to withholding", "Never subject to withholding"),
    ) == {False}


@pytest.mark.parametrize(
    ("wording", "selector"),
    (
        # "nor" inside "normal" and "minor", "except" inside "exceptional",
        # and "never" inside "whenever" and "Nevertheless".
        ("Received from a normal pension fund", PENSION_INPUT),
        ("Paid by a minor employer", "paid_by_employer"),
        ("Received from an exceptional pension fund", PENSION_INPUT),
        ("Paid whenever reported to the agency", "reported_to_agency"),
        ("Nevertheless reported to the agency", "reported_to_agency"),
    ),
)
def test_a_negator_counts_only_as_a_whole_word(wording: str, selector: str):
    assert completeness_module._list_item_values(selector, _items(wording)) == (
        (wording, True),
    )


@pytest.mark.parametrize(
    ("item", "selector"),
    (
        ("Not subject to withholding", "subject_to_withholding"),
        ("No longer subject to withholding", "subject_to_withholding"),
        ("Lacking a social security number", "has_social_security_number"),
        ("Paid after the person fails to consent", "person_consents"),
        ("Paid to a person who does not consent", "person_consents"),
    ),
)
def test_items_take_their_value_from_the_shared_reader(item: str, selector: str):
    # The reader scopes "not", "no", "lacking" and "without" over the name's
    # words that follow, and reads "fails to" and "does ... not" before the
    # selector's verb, so these items keep the value it gives them.
    assert completeness_module._list_item_active_values(
        "if they are:", selector, _items(item)
    ) == {False}


@pytest.mark.parametrize(
    ("item", "selector"),
    (
        ("Lacks a social security number", "has_social_security_number"),
        ("Isn't reported to the agency", "reported_to_agency"),
        ("Paid free of withholding", "paid_with_withholding"),
        ("Owes no tax", "owes_tax"),
        ("Paid when a certificate is missing", "certificate"),
        ("Paid to a person not consenting", "person_consents"),
    ),
)
def test_negations_the_reader_misses_without_a_listed_negator(item: str, selector: str):
    # A known limit: the reader has no cue for "lacks", "isn't" or "free of",
    # and it reads "missing" as negative only in a selector's own name. It
    # reads only the name's words of four letters or more, so "no" scopes
    # nothing it reads in "Owes no tax", and only forms that start with a
    # whole word of the name, so it reads "person" and not "consenting" for
    # `person_consents`. None of these items holds a listed negator or a
    # second negation, so each still reads as positive.
    assert completeness_module._list_item_values(selector, _items(item)) == (
        (item, True),
    )


# Each item opens with a cue the reader knows, but a comma, more than 48
# characters, or both stand between the cue and the name's words, so the
# reader would read the item as positive, though it needs the selector false.
CUE_OUT_OF_REACH_ITEMS = (
    pytest.param(
        "Not, in the agency's view, subject to withholding",
        "Not, in the agency's view, subject to withholding",
        "subject_to_withholding",
        id="comma",
    ),
    pytest.param(
        "Not, in the agency's view, subject to withholding",
        "subject to withholding",
        "subject_to_withholding",
        id="comma-excerpt-without-not",
    ),
    pytest.param(
        "Not subject in any calendar quarter of the reporting year to backup "
        "withholding",
        "Not subject in any calendar quarter of the reporting year to backup "
        "withholding",
        "backup_withholding",
        id="long",
    ),
    pytest.param(
        "Not paid at any time in the calendar year by an employer's pension fund",
        "Not paid at any time in the calendar year by an employer's pension fund",
        "pension_fund",
        id="long-pension-fund",
    ),
    pytest.param(
        "Without, in the payer's records, any amount reported to the agency",
        "Without, in the payer's records, any amount reported to the agency",
        "reported_to_agency",
        id="without-comma",
    ),
)


@pytest.mark.parametrize(("item", "excerpt", "selector"), CUE_OUT_OF_REACH_ITEMS)
def test_a_cue_the_reader_does_not_reach_the_name_with_gives_no_value(
    item: str, excerpt: str, selector: str
):
    # The reader alone reads each item as positive. A cue stands before one
    # of the name's words that the reader doesn't read as negated, so the
    # item gives no value.
    assert (
        completeness_module._source_exception_selector_wording_value(item, selector)
        is True
    )
    assert completeness_module._list_item_negation_count(item, selector) == 1
    assert completeness_module._prefix_cue_misses_a_name_word(item, selector)
    assert (
        completeness_module._list_item_values(
            selector, _items((excerpt, f"(1) {item}"))
        )
        == ()
    )


@pytest.mark.parametrize("excluded_when", (False, True), ids=("faithful", "inverted"))
@pytest.mark.parametrize(("item", "excerpt", "selector"), CUE_OUT_OF_REACH_ITEMS)
def test_an_item_with_a_cue_out_of_the_readers_reach_pairs_neither_encoding(
    item: str, excerpt: str, selector: str, excluded_when: bool
):
    """An item whose cue doesn't reach the name's words pairs neither way.

    Each item excludes a payment only when its positively named selector is
    false. Read as positive, the item would pair the inverted encoding. It
    gives no value instead, so the chapeau stays unpaired whichever way the
    formula reads the item, as on main.
    """

    assert not _pairs(
        selector, excerpt, _item_source(item), excluded_when=excluded_when
    )


@pytest.mark.parametrize(
    ("item", "selector", "value"),
    (
        # The cue stands right before the name's words.
        ("Not subject to withholding", "subject_to_withholding", False),
        ("Without any amount reported to the agency", "reported_to_agency", False),
        # The cue follows the name's words, so it stands before none of them.
        ("Subject to withholding, not to tax", "subject_to_withholding", True),
        # The cue reaches the name's word across other words.
        ("Not exempt from withholding", "withholding", False),
    ),
)
def test_a_cue_that_reaches_the_names_words_or_follows_them_keeps_the_value(
    item: str, selector: str, value: bool
):
    assert not completeness_module._prefix_cue_misses_a_name_word(item, selector)
    assert completeness_module._list_item_values(selector, _items(item)) == (
        (item, value),
    )


def test_a_cue_before_another_predicate_leaves_an_item_without_a_value():
    # A known limit: the reader reads "but" as starting another predicate, so
    # "Not" negates neither "subject" nor "withholding", and the reader reads
    # the item as positive, which is right here. The cue still stands before
    # the name's words, so the item gives no value, and the chapeau stays
    # unpaired, as on main.
    item = "Not reported to the agency but subject to withholding"

    assert (
        completeness_module._source_exception_selector_wording_value(
            item, "subject_to_withholding"
        )
        is True
    )
    assert (
        completeness_module._list_item_values("subject_to_withholding", _items(item))
        == ()
    )


def test_a_cue_out_of_the_readers_reach_in_a_chapeau_reads_as_on_main():
    # A known limit: where the words stand in the condition itself, the check
    # reads them with the shared reader, as main does, and the comma keeps
    # "not" from negating `subject_to_withholding`.
    assert (
        completeness_module._source_exception_selector_active_value(
            "if they are not, in the agency's view, subject to withholding",
            "subject_to_withholding",
        )
        is True
    )


# Each item holds two negation cues before the name's words, so it needs the
# selector true, as "Not without withholding" needs `withholding` true.
DOUBLE_NEGATION_ITEMS = (
    pytest.param("Not without withholding", "withholding", id="not-without"),
    pytest.param(
        "Not lacking a social security number",
        "has_social_security_number",
        id="not-lacking",
    ),
    pytest.param(
        "Paid after the person does not fail to consent",
        "person_consents",
        id="does-not-fail-to",
    ),
    # The reader's own German cues, as in "not without proof".
    pytest.param("Nicht ohne Nachweis", "nachweis", id="nicht-ohne"),
)


@pytest.mark.parametrize(
    ("wording", "selector"),
    (
        *DOUBLE_NEGATION_ITEMS,
        pytest.param("No longer without withholding", "withholding", id="no-without"),
        pytest.param(
            "Not in the absence of withholding", "withholding", id="not-absence-of"
        ),
        # "n't" is no cue to the reader, but it still turns "without" around.
        pytest.param("Isn't without withholding", "withholding", id="n't-without"),
        pytest.param(
            "Paid after the person doesn't fail to consent",
            "person_consents",
            id="n't-fail-to",
        ),
    ),
)
def test_two_negation_cues_leave_an_item_without_a_value(wording: str, selector: str):
    # The reader reads one of the name's words as negated when any cue comes
    # before it, so two cues read as one and the item would read as negative.
    assert (
        completeness_module._source_exception_selector_wording_value(wording, selector)
        is False
    )
    assert completeness_module._list_item_values(selector, _items(wording)) == ()


@pytest.mark.parametrize(("item", "selector"), DOUBLE_NEGATION_ITEMS)
def test_an_item_with_two_negation_cues_pairs_neither_encoding(
    item: str, selector: str
):
    """An item whose cues the reader can't compose pairs neither encoding.

    Each item excludes a payment only when its selector is true. The reader
    would read it as needing the selector false, so on its value alone the
    inverted encoding would pair the chapeau and the faithful one wouldn't.
    The item gives no value instead, and the chapeau stays unpaired either way,
    as on main.
    """

    source = _item_source(item)

    assert not _pairs(selector, item, source, excluded_when=True)
    assert not _pairs(selector, item, source, excluded_when=False)


# Each item joins a cue the reader knows with a negation it reads past, so it
# needs the selector true, as "Not disqualified from receiving benefits" needs
# `receives_benefits` true.
CUE_WITH_UNREAD_NEGATION_ITEMS = (
    pytest.param(
        "Not disqualified from receiving benefits",
        "receives_benefits",
        id="not-disqualified",
    ),
    pytest.param("Not unable to claim withholding", "withholding", id="not-unable"),
    pytest.param("Not paid to a non-resident alien", "resident_alien", id="not-non"),
)


@pytest.mark.parametrize(
    ("wording", "selector"),
    (
        *CUE_WITH_UNREAD_NEGATION_ITEMS,
        # Each word the reader takes as negative only in a selector's name.
        pytest.param("Not ineligible for withholding", "withholding", id="ineligible"),
        pytest.param(
            "Not disqualified from withholding", "withholding", id="disqualified"
        ),
        pytest.param(
            "Not unqualified for withholding", "withholding", id="unqualified"
        ),
        pytest.param(
            "Not an inability to claim withholding", "withholding", id="inability"
        ),
        pytest.param(
            "Not unwilling to claim withholding", "withholding", id="unwilling"
        ),
        pytest.param(
            "Not an unwillingness to claim withholding",
            "withholding",
            id="unwillingness",
        ),
        # Each form that negates without being a cue to the reader.
        pytest.param("No person lacks withholding", "withholding", id="lacks"),
        pytest.param("Not failed to claim withholding", "withholding", id="failed-to"),
        pytest.param(
            "Not failing to claim withholding", "withholding", id="failing-to"
        ),
        pytest.param("Not refused withholding", "withholding", id="refused"),
        pytest.param("Not one who refuses withholding", "withholding", id="refuses"),
        pytest.param("Not a refusal of withholding", "withholding", id="refusal"),
        pytest.param("Not free of withholding", "withholding", id="free-of"),
        pytest.param("Not a non-employer withholding", "withholding", id="non-hyphen"),
        pytest.param("Not a nonemployer withholding", "withholding", id="non-closed"),
        # "non" counts before four letters or more.
        pytest.param("Not a nonzero withholding", "withholding", id="non-four"),
        pytest.param("NOT A NON-EMPLOYER WITHHOLDING", "withholding", id="upper-case"),
        # A cue before the name's word and one the reader reads after it.
        pytest.param(
            "Not a payment for which withholding is absent",
            "withholding",
            id="not-is-absent",
        ),
        pytest.param(
            "Not a payment for which withholding is not present",
            "withholding",
            id="not-is-not-present",
        ),
        # A word the reader takes as a negative state only in a selector's
        # own name, and another form of "lack".
        pytest.param("Not missing withholding", "withholding", id="missing"),
        pytest.param("Not a payment absent withholding", "withholding", id="absent"),
        pytest.param(
            "Not a household that lacked withholding", "withholding", id="lacked"
        ),
    ),
)
def test_a_cue_with_a_negation_the_reader_does_not_combine_gives_no_value(
    wording: str, selector: str
):
    # The reader applies one negation and reads past the other, so on its own
    # it would read each item as negative. The item holds two negations and
    # gives no value.
    assert (
        completeness_module._source_exception_selector_wording_value(wording, selector)
        is False
    )
    assert completeness_module._list_item_negation_count(wording, selector) == 2
    assert completeness_module._list_item_values(selector, _items(wording)) == ()


@pytest.mark.parametrize("excluded_when", (False, True), ids=("inverted", "faithful"))
@pytest.mark.parametrize(("item", "selector"), CUE_WITH_UNREAD_NEGATION_ITEMS)
def test_a_cue_with_a_negation_the_reader_reads_past_pairs_neither_encoding(
    item: str, selector: str, excluded_when: bool
):
    """A cue beside a negation the reader reads past pairs neither encoding.

    Each item excludes a payment only when its selector is true. The reader
    applies the "Not" and reads past "disqualified", "unable" or "non-", so
    on its value alone the inverted encoding would pair the chapeau and the
    faithful one wouldn't. The item gives no value instead, and the chapeau
    stays unpaired either way, as on main.
    """

    assert not _pairs(selector, item, _item_source(item), excluded_when=excluded_when)


@pytest.mark.parametrize(
    ("wording", "selector", "count"),
    (
        ("Subject to withholding", "subject_to_withholding", 0),
        ("Not subject to withholding", "subject_to_withholding", 1),
        # A cue the reader reads after a name's word counts once, together
        # with the "not" it holds.
        ("Paid when withholding is absent", "withholding", 1),
        ("Paid when withholding is not present", "withholding", 1),
        ("Not paid when withholding is not present", "withholding", 2),
        # One word the reader reads past is one negation.
        ("Disqualified from receiving benefits", "receives_benefits", 1),
        ("Paid to a non-resident alien", "resident_alien", 1),
        ("Lacks a social security number", "has_social_security_number", 1),
        # Two of them are two.
        ("Unable to claim a non-resident alien", "resident_alien", 2),
        # A cue the reader knows counts even as one of the name's own words:
        # the reader applies the "Not" to "lacking" and reads the item as
        # agreeing with the name, though it needs `lacking_certificate` false.
        ("Not lacking a certificate", "lacking_certificate", 2),
        # "none", "nonce", "nonage", "canonical", "freedom of", "slacks",
        # "Lacksley", "tunable" and "unwillingly" each hold one of the forms
        # without being one.
        ("Not withheld by none", "withheld", 1),
        ("Not a nonce withholding", "withholding", 1),
        ("Not withholding during nonage", "withholding", 1),
        ("Not a canonical withholding", "withholding", 1),
        ("Not a freedom of withholding", "withholding", 1),
        ("Not withholding on slacks", "withholding", 1),
        ("Not a Lacksley withholding", "withholding", 1),
        ("Not a tunable withholding", "withholding", 1),
        ("Not unwillingly subject to withholding", "withholding", 1),
        # A negation that two of the patterns match counts once: "lacking" is
        # a cue and a negative state, and "disqualified", pinned above, and
        # "absent" are each a negative state as well as an inherently
        # negative word or a part of "is absent".
        ("Lacking a certificate", "certificate", 1),
        ("Not lacking a certificate", "certificate", 2),
        ("Not a payment for which withholding is absent", "withholding", 2),
        ("Not missing a certificate", "certificate", 2),
        ("Not a household that lacked withholding", "withholding", 2),
        # The reader's German cues count as its English ones do.
        ("Nicht ohne Nachweis", "nachweis", 2),
    ),
)
def test_an_items_negations_are_counted_once_each(
    wording: str, selector: str, count: int
):
    assert completeness_module._list_item_negation_count(wording, selector) == count


@pytest.mark.parametrize(
    ("wording", "selector"),
    (
        ("Not disqualified from receiving benefits", "disqualified"),
        ("Not ineligible for benefits", "ineligible"),
        ("Not refused by the agency", "refused_by_agency"),
        ("Not free of withholding", "free_of_withholding"),
        ("Not paid to a nonresident alien", "nonresident_alien"),
        (
            "Not a person who lacks a social security number",
            "lacks_social_security_number",
        ),
        ("Not failed to report", "failed_to_report"),
    ),
)
def test_a_negating_word_in_the_selectors_name_is_not_a_second_negation(
    wording: str, selector: str
):
    # The reader matches "disqualified", "refused", "free", "nonresident",
    # "lacks" or "failed" as one of the name's own words and applies the "Not"
    # to it, so each item holds one negation and needs the selector false.
    assert completeness_module._list_item_negation_count(wording, selector) == 1
    assert completeness_module._list_item_values(selector, _items(wording)) == (
        (wording, False),
    )


def test_a_cue_with_a_negating_word_outside_the_lists_still_reads_as_negative():
    # A known limit: "exempt" negates here, but it is neither a cue to the
    # reader nor one of the forms the count takes. The item holds one
    # negation by that count, and the reader reads it as negative, though it
    # needs `withholding` true.
    item = "Not exempt from withholding"

    assert completeness_module._list_item_negation_count(item, "withholding") == 1
    assert completeness_module._list_item_values("withholding", _items(item)) == (
        (item, False),
    )


@pytest.mark.parametrize(
    ("text", "excerpt", "expected"),
    (
        # The cue starts right after the excerpt.
        ("(1) Paid when withholding is absent; or", "Paid when withholding", True),
        ("(1) Paid when withholding is not present; or", "withholding", True),
        ("(1) wenn der Nachweis fehlt; oder", "der Nachweis", True),
        # The excerpt stops inside the cue.
        ("(1) Paid when withholding is absent; or", "withholding is", True),
        ("(1) Paid when withholding is not present; or", "withholding is not", True),
        ("(1) Paid when withholding is absent; or", "withholding is abs", True),
        # The excerpt holds the whole cue, so the wording does too.
        ("(1) Paid when withholding is absent; or", "withholding is absent", False),
        # Whitespace collapses on both sides, and the cue matches in any
        # letter case.
        ("(1)  Paid when\n\nwithholding IS   ABSENT; or", "when\nwithholding", True),
        # A cue further on doesn't run past the excerpt's end.
        (
            "(1) Paid when withholding by the employer is absent; or",
            "Paid when withholding",
            False,
        ),
        # "and never withheld" holds none of the reader's postposed cues.
        (
            "(1) Reported to the agency and never withheld; or",
            "Reported to the agency",
            False,
        ),
        # The cue is a whole word.
        ("(1) Paid when the owner is absentee; or", "the owner", False),
        # The excerpt's first match counts, as it does for the wording.
        (
            "(1) Paid when withholding applies, or when withholding is absent; or",
            "when withholding",
            False,
        ),
        # Text that doesn't hold the excerpt has nothing after it, even when
        # the excerpt is as long as the text up to the middle of the cue.
        (
            "(1) Paid when withholding is absent; or",
            "reported to the state agency",
            False,
        ),
    ),
)
def test_a_postposed_negation_cue_follows_a_wording_it_runs_past(
    text: str, excerpt: str, expected: bool
):
    wording = completeness_module._wording_through_excerpt(text, excerpt)

    assert completeness_module._negation_follows_wording(text, wording) is expected


def test_each_item_excerpt_records_a_postposed_negation_after_it():
    item = "Paid when withholding is absent"
    source = _item_source(item)
    branches = completeness_module.recognize_source_structure(
        source, corpus_citation_path=CITATION
    )
    rule = _rule(
        PAYMENT_OUTPUT,
        ITEM_REFS,
        "true",
        [_atom("Paid when withholding"), _atom(item), _atom(EMPLOYER)],
    )

    found = completeness_module._rule_excerpts_under_branch(
        rule,
        next(branch for branch in branches if branch.path == ("a",)),
        source_branches=branches,
        corpus_citation_path=CITATION,
    )

    # Only the first excerpt stops short of "is absent".
    assert found == (
        completeness_module._ListItemExcerpt(
            "Paid when withholding", "(1) Paid when withholding", (), True
        ),
        completeness_module._ListItemExcerpt(item, f"(1) {item}"),
        completeness_module._ListItemExcerpt(EMPLOYER, f"(2) {EMPLOYER}"),
    )


def test_a_postposed_negation_after_the_wording_leaves_an_item_without_a_value():
    excerpt = "Paid when withholding"

    # The wording reads as positive on its own. With "is absent" after it,
    # the item needs `withholding` false, so it gives no value.
    assert completeness_module._list_item_values(
        "withholding",
        (completeness_module._ListItemExcerpt(excerpt, f"(1) {excerpt}"),),
    ) == ((excerpt, True),)
    assert (
        completeness_module._list_item_values(
            "withholding",
            (
                completeness_module._ListItemExcerpt(
                    excerpt, f"(1) {excerpt}", (), True
                ),
            ),
        )
        == ()
    )


@pytest.mark.parametrize("excluded_when", (False, True), ids=("faithful", "inverted"))
@pytest.mark.parametrize(
    ("item", "excerpt", "selector"),
    (
        pytest.param(
            "Paid when withholding is absent",
            "Paid when withholding",
            "withholding",
            id="is-absent",
        ),
        pytest.param(
            "Paid when withholding is not present",
            "Paid when withholding",
            "withholding",
            id="is-not-present",
        ),
        pytest.param(
            "Paid when withholding is not present",
            "Paid when withholding is not",
            "withholding",
            id="cut-inside-the-cue",
        ),
        pytest.param(
            "Paid to a person whose social security number is not present",
            "Paid to a person whose social security number",
            "social_security_number",
            id="several-name-words",
        ),
    ),
)
def test_an_excerpt_that_stops_short_of_a_postposed_negation_pairs_neither_encoding(
    item: str, excerpt: str, selector: str, excluded_when: bool
):
    """An excerpt cut before the end of "is absent" pairs neither encoding.

    Each item excludes a payment only when its selector is false. The wording
    ends with the excerpt, so the reader would never see the cue whole and
    would read the item as positive: the inverted encoding would pair the
    chapeau and the faithful one wouldn't. The item gives no value instead,
    and the chapeau stays unpaired either way, as on main.
    """

    assert not _pairs(
        selector, excerpt, _item_source(item), excluded_when=excluded_when
    )


@pytest.mark.parametrize(
    "item", ("Paid when withholding is absent", "Paid when withholding is not present")
)
def test_an_excerpt_that_keeps_a_postposed_negation_pairs_the_faithful_encoding(
    item: str,
):
    # The reader sees the cue whole and reads `withholding` false.
    source = _item_source(item)

    assert _pairs("withholding", item, source, excluded_when=False)
    assert not _pairs("withholding", item, source, excluded_when=True)


def test_a_negator_after_the_excerpt_that_is_no_postposed_cue_does_not_count():
    # "never" scopes "withheld", which comes after the excerpt, so the excerpt
    # "Reported to the agency" keeps the value its own wording gives.
    source = _item_source("Reported to the agency and never withheld")
    excerpt = "Reported to the agency"

    assert _pairs("reported_to_agency", excerpt, source, excluded_when=True)
    assert not _pairs("reported_to_agency", excerpt, source, excluded_when=False)


@pytest.mark.parametrize(
    ("text", "excerpt", "expected"),
    (
        # A negation follows the excerpt in its clause.
        pytest.param(
            "(1) Paid to a person whose consent was never given; or",
            "Paid to a person whose consent",
            True,
            id="never",
        ),
        pytest.param(
            "(1) Paid when withholding is never present; or",
            "Paid when withholding",
            True,
            id="never-present",
        ),
        pytest.param(
            "(1) Paid to a person whose consent was not given; or",
            "Paid to a person whose consent",
            True,
            id="not",
        ),
        pytest.param(
            "(1) Paid to a person whose claim isn't settled; or",
            "Paid to a person whose claim",
            True,
            id="n-t",
        ),
        pytest.param(
            "(1) Paid to a person whose consent was refused; or",
            "Paid to a person whose consent",
            True,
            id="refused",
        ),
        pytest.param(
            "(1) Paid when withholding by the employer is absent; or",
            "Paid when withholding",
            True,
            id="postposed-further-on",
        ),
        pytest.param(
            "(1) Paid to a person who is unable to consent; or",
            "Paid to a person",
            True,
            id="unable",
        ),
        pytest.param(
            "(1) Paid to a person who is a non-resident alien; or",
            "Paid to a person who is a",
            True,
            id="non",
        ),
        # A comma doesn't end the clause.
        pytest.param(
            "(1) Paid to a person whose consent, if any, was never given; or",
            "Paid to a person whose consent",
            True,
            id="comma",
        ),
        # Letters of "and" and "or" inside a word end nothing.
        pytest.param(
            "(1) Paid when withholding is mandatory, never waived; or",
            "Paid when withholding",
            True,
            id="mandatory",
        ),
        # The clause ends after the wording, so an "and" inside it ends
        # nothing.
        pytest.param(
            "(1) Paid and reported when withholding is never present; or",
            "Paid and reported when withholding",
            True,
            id="and-in-the-wording",
        ),
        # The excerpt stops inside the negation.
        pytest.param(
            "(1) Paid when withholding is never present; or",
            "Paid when withholding is nev",
            True,
            id="inside-a-word",
        ),
        pytest.param(
            "(1) Paid when the household failed to apply; or",
            "Paid when the household failed",
            True,
            id="between-its-words",
        ),
        pytest.param(
            "(1) Paid when withholding IS NOT present; or",
            "Paid when withholding",
            True,
            id="upper-case",
        ),
        # Negations the reader reads past, or reads only in a selector's own
        # name.
        pytest.param(
            "(1) Paid to a person whose consent cannot be obtained; or",
            "Paid to a person whose consent",
            True,
            id="cannot",
        ),
        pytest.param(
            "(1) Paid when withholding applies to none of the payers; or",
            "Paid when withholding",
            True,
            id="none",
        ),
        pytest.param(
            "(1) Paid when withholding is missing; or",
            "Paid when withholding",
            True,
            id="missing",
        ),
        pytest.param(
            "(1) Paid to a household that lacked withholding; or",
            "Paid to a household that",
            True,
            id="lacked",
        ),
        # A clause with no end runs through the text's end.
        pytest.param(
            "(1) Paid when withholding is never present,",
            "Paid when withholding",
            True,
            id="no-clause-end",
        ),
        # After an "and", "or" or "but", a negation scopes other words.
        pytest.param(
            "(1) Reported to the agency and never withheld; or",
            "Reported to the agency",
            False,
            id="and",
        ),
        pytest.param(
            "(1) Paid when withholding applies, or when withholding is absent; or",
            "when withholding",
            False,
            id="or",
        ),
        pytest.param(
            "(1) Paid to a person whose consent was given but never recorded; or",
            "Paid to a person whose consent",
            False,
            id="but",
        ),
        # So does a negation after a ".", ";" or ":".
        pytest.param(
            "(1) Paid when withholding applies. No other payment counts; or",
            "Paid when withholding applies",
            False,
            id="period",
        ),
        # A "." ends the clause only where it ends a sentence: before an
        # upper-case letter or at the text's end, and not in an abbreviation.
        pytest.param(
            "(1) Paid when withholding applies. never when waived; or",
            "Paid when withholding applies",
            True,
            id="period-before-lower-case",
        ),
        pytest.param(
            "(1) Paid to a person whose consent under 5 U.S.C. 552a was never "
            "given; or",
            "Paid to a person whose consent",
            True,
            id="u-s-c",
        ),
        pytest.param(
            "(1) Paid when withholding by the U.S. Treasury is never present; or",
            "Paid when withholding",
            True,
            id="u-s-before-a-capital",
        ),
        pytest.param(
            "(1) Paid to a person whose consent (e.g., a written release) was "
            "never given; or",
            "Paid to a person whose consent",
            True,
            id="e-g",
        ),
        # Only the legal abbreviations hold these: "g" and "Wis" are no short
        # abbreviation, and an upper-case letter follows each ".".
        pytest.param(
            "(1) Paid in kind, e.g. Treasury notes never redeemed; or",
            "Paid in kind",
            True,
            id="e-g-before-a-capital",
        ),
        pytest.param(
            "(1) Paid when withholding under Wis. Stat. 71.64 is never present; or",
            "Paid when withholding",
            True,
            id="code-citation",
        ),
        # The check reads "No" itself as a negation, so these use other short
        # abbreviations.
        pytest.param(
            "(1) Paid when withholding under Sec. 7 is never present; or",
            "Paid when withholding",
            True,
            id="sec-before-a-number",
        ),
        pytest.param(
            "(1) Paid when withholding by St. Mary's Hospital is never present; or",
            "Paid when withholding",
            True,
            id="st-before-a-capital",
        ),
        pytest.param(
            "(1) Paid when withholding under Pub. L. 104-193 is never present; or",
            "Paid when withholding",
            True,
            id="pub-l",
        ),
        pytest.param(
            "(1) Paid when withholding applies; not when it is waived; or",
            "Paid when withholding applies",
            False,
            id="semicolon",
        ),
        pytest.param(
            "(1) Paid when withholding applies: not a waived payment; or",
            "Paid when withholding applies",
            False,
            id="colon",
        ),
        # Words that only hold the letters of a negation.
        pytest.param(
            "(1) Paid when withholding is nominal, canonical; or",
            "Paid when withholding",
            False,
            id="letters-only",
        ),
        # Lower case turns the dotted capital I into two characters, so the
        # spans no longer line up with the text, and the clause runs through
        # the text's end. "never" then counts past the "or". Its ASCII twin
        # ends the clause at the "or".
        pytest.param(
            "(1) \u0130stanbul pays when withholding applies, or never when waived; or",
            "when withholding applies",
            True,
            id="lower-case-changes-length",
        ),
        pytest.param(
            "(1) Istanbul pays when withholding applies, or never when waived; or",
            "when withholding applies",
            False,
            id="lower-case-keeps-length",
        ),
        # The wording holds the negation, so nothing follows it.
        pytest.param(
            "(1) Paid to a person whose consent was never given; or",
            "Paid to a person whose consent was never given",
            False,
            id="whole-item",
        ),
        # Text that doesn't hold the excerpt has nothing after it.
        pytest.param(
            "(1) Paid when withholding is never present; or",
            "reported to the state agency",
            False,
            id="excerpt-not-found",
        ),
    ),
)
def test_a_negation_later_in_the_excerpts_clause_follows_its_wording(
    text: str, excerpt: str, expected: bool
):
    wording = completeness_module._wording_through_excerpt(text, excerpt)

    assert completeness_module._negation_follows_in_clause(text, wording) is expected


def test_each_item_excerpt_records_a_negation_later_in_its_clause():
    item = "Paid to a person whose consent was never given"
    source = _item_source(item)
    branches = completeness_module.recognize_source_structure(
        source, corpus_citation_path=CITATION
    )
    rule = _rule(
        PAYMENT_OUTPUT,
        ITEM_REFS,
        "true",
        [_atom("Paid to a person whose consent"), _atom(EMPLOYER)],
    )

    found = completeness_module._rule_excerpts_under_branch(
        rule,
        next(branch for branch in branches if branch.path == ("a",)),
        source_branches=branches,
        corpus_citation_path=CITATION,
    )

    assert found == (
        completeness_module._ListItemExcerpt(
            "Paid to a person whose consent",
            "(1) Paid to a person whose consent",
            (),
            True,
        ),
        completeness_module._ListItemExcerpt(EMPLOYER, f"(2) {EMPLOYER}"),
    )


@pytest.mark.parametrize("excluded_when", (False, True), ids=("faithful", "inverted"))
@pytest.mark.parametrize(
    ("item", "excerpt", "selector"),
    (
        pytest.param(
            "Paid to a person whose consent was never given",
            "Paid to a person whose consent",
            "consent",
            id="never-given",
        ),
        pytest.param(
            "Paid when withholding is never present",
            "Paid when withholding",
            "withholding",
            id="never-present",
        ),
        pytest.param(
            "Paid to a person whose consent was refused",
            "Paid to a person whose consent",
            "consent",
            id="refused",
        ),
        pytest.param(
            "Paid when withholding by the employer is absent",
            "Paid when withholding",
            "withholding",
            id="postposed-further-on",
        ),
        # An abbreviation's "." doesn't end the clause.
        pytest.param(
            "Paid to a person whose consent under 5 U.S.C. 552a was never given",
            "Paid to a person whose consent",
            "consent",
            id="u-s-c",
        ),
        pytest.param(
            "Paid to a person whose consent (e.g., a written release) was never given",
            "Paid to a person whose consent",
            "consent",
            id="e-g",
        ),
        pytest.param(
            "Paid when withholding by St. Mary's Hospital is never present",
            "Paid when withholding",
            "withholding",
            id="st",
        ),
        # The words of a tax or benefits citation.
        pytest.param(
            "Paid when withholding under Treas. Reg. 1.1441 is never present",
            "Paid when withholding",
            "withholding",
            id="treas-reg",
        ),
        pytest.param(
            "Paid when withholding under Rev. Rul. 2004-5 is never present",
            "Paid when withholding",
            "withholding",
            id="rev-rul",
        ),
        pytest.param(
            "Paid when withholding under 89 Fed. Reg. 1000 is never present",
            "Paid when withholding",
            "withholding",
            id="fed-reg",
        ),
        pytest.param(
            "Paid when withholding under Dept. Order 5 is never present",
            "Paid when withholding",
            "withholding",
            id="dept",
        ),
        # Negations the reader reads past.
        pytest.param(
            "Paid to a person whose consent cannot be obtained",
            "Paid to a person whose consent",
            "consent",
            id="cannot",
        ),
        pytest.param(
            "Paid when withholding is missing",
            "Paid when withholding",
            "withholding",
            id="missing",
        ),
    ),
)
def test_an_excerpt_cut_before_a_negation_in_its_clause_pairs_neither_encoding(
    item: str, excerpt: str, selector: str, excluded_when: bool
):
    """An excerpt that stops before a negation in its clause pairs neither.

    Each item excludes a payment only when its selector is false. The wording
    ends with the excerpt and reads as positive, so on its value alone the
    inverted encoding would pair the chapeau and the faithful one wouldn't.
    The item gives no value instead, and the chapeau stays unpaired either
    way, as on main.
    """

    assert not _pairs(
        selector, excerpt, _item_source(item), excluded_when=excluded_when
    )


# Item (1) ends with a bare ",", so nothing after its excerpt ends the clause.
COMMA_ENDED_SOURCE = (
    f"(a) Excluded payments. {EXCLUSION_CHAPEAU}\n\n"
    "(1) Paid when withholding is never present,\n\n"
    f"(2) {EMPLOYER}.\n\n{PAYMENT_RECORDS}"
)


@pytest.mark.parametrize("excluded_when", (False, True), ids=("faithful", "inverted"))
def test_an_excerpt_in_a_clause_with_no_end_pairs_neither_encoding(
    excluded_when: bool,
):
    """A clause with no end runs through the item, so "never" still counts.

    The item excludes a payment only when `withholding` is false. Read as
    ending right after the excerpt, the clause would hold no negation, and
    the inverted encoding would pair the chapeau. The clause runs through
    the item's end instead, so the item gives no value, and the chapeau
    stays unpaired either way, as on main.
    """

    assert _item_paths(COMMA_ENDED_SOURCE, ("a",)) == {("a", "1"), ("a", "2")}
    assert not _pairs(
        "withholding",
        "Paid when withholding",
        COMMA_ENDED_SOURCE,
        excluded_when=excluded_when,
    )


@pytest.mark.parametrize(
    ("text", "start", "end"),
    (
        # A sentence end, before an upper-case letter or at the text's end.
        ("Paid. Never waived", 0, 4),
        ("Paid when due.", 0, 13),
        # An abbreviation's "." ends nothing, so the clause runs on to the ";".
        ("Paid under 5 U.S.C. 552a; or", 0, 24),
        ("Paid by the U.S. Treasury; or", 0, 25),
        ("Paid (e.g., in cash); or", 0, 20),
        ("Paid in kind, e.g. Treasury notes; or", 0, 33),
        ("Paid under Wis. Stat. 71.64; or", 0, 27),
        ("Paid on Form No. W-2; or", 0, 20),
        ("Paid under Pub. L. 104-193; or", 0, 26),
        ("Paid to J. Smith; or", 0, 16),
        ("Paid under Treas. Reg. 1.1441; or", 0, 29),
        ("Paid under Rev. Rul. 2004-5; or", 0, 27),
        ("Paid under Rev. Proc. 2020-1; or", 0, 28),
        ("Paid under 89 Fed. Reg. 1000; or", 0, 28),
        ("Paid under Dept. Order 5; or", 0, 24),
        ("Paid under Admin. Order 5; or", 0, 25),
        ("Paid under 5 U.S.C. Supp. II; or", 0, 28),
        # The words hold their "." in lower case too.
        ("Paid under art. III of the treaty; or", 0, 33),
        ("Paid under sec. B of the plan; or", 0, 29),
        # A single letter holds it only as a capital.
        ("Paid under item a. Then waived; or", 0, 17),
        # A "." before a lower-case letter or a digit ends nothing.
        ("Paid. never waived; or", 0, 18),
        ("Paid in part 3. 4 more; or", 0, 22),
        # An "and", "or" or "but", in any letter case, ends the clause.
        ("Paid AND waived", 0, 5),
        # The search starts at ``start``.
        ("Paid; then waived. Never", 5, 17),
        # No end: the clause runs through the text's end.
        ("Paid when withholding is never present,", 0, 39),
    ),
)
def test_a_clause_ends_at_a_sentence_end_and_not_at_an_abbreviation(
    text: str, start: int, end: int
):
    assert completeness_module._excerpt_clause_end(text, start) == end


@pytest.mark.parametrize(
    "word",
    (
        "Treas",
        "Reg",
        "Regs",
        "Rev",
        "Rul",
        "Proc",
        "Fed",
        "Dept",
        "Admin",
        "Ann",
        "Supp",
    ),
)
def test_a_citation_word_before_a_capital_ends_no_clause(word: str):
    # Each word of a tax or benefits citation holds its ".", even before an
    # upper-case letter, so the clause runs on to the ";".
    text = f"Paid under {word}. Order 5; or"

    assert completeness_module._excerpt_clause_end(text, 0) == text.index(";")


@pytest.mark.parametrize("excluded_when", (False, True), ids=("faithful", "inverted"))
@pytest.mark.parametrize(
    "item",
    (
        pytest.param(
            "Paid when withholding under art. III of the treaty is never present",
            id="art",
        ),
        pytest.param(
            "Paid when withholding under sec. B of the plan is never present",
            id="sec",
        ),
    ),
)
def test_a_lower_case_citation_word_keeps_a_negation_in_the_clause(
    item: str, excluded_when: bool
):
    """A citation word in lower case holds its "." as it does capitalized.

    The item excludes a payment only when `withholding` is false. Had the
    clause ended at the "." of "art." or "sec.", "never" would fall outside
    it, the excerpt would keep the reader's value, `withholding` true, and the
    inverted encoding would pair the chapeau. The clause runs on to the item's
    end, so the excerpt gives no value, and the chapeau stays unpaired either
    way, as on main.
    """

    excerpt = "Paid when withholding"
    text = f"(1) {item}; or"

    assert completeness_module._negation_follows_in_clause(
        text, completeness_module._wording_through_excerpt(text, excerpt)
    )
    assert not _pairs(
        "withholding", excerpt, _item_source(item), excluded_when=excluded_when
    )


def test_the_reader_and_the_item_checks_share_their_negation_words():
    # The item checks read the reader's own cues and negative states, so a
    # word added to the reader's lists reaches them too.
    for cue in ("not", "no", "without", "lacking", "absence of", "nicht", "ohne"):
        assert completeness_module._LIST_ITEM_NEGATION_CUE.fullmatch(cue)
    for word in completeness_module._NEGATIVE_STATE_SELECTOR_TOKENS:
        assert completeness_module._NEGATIVE_STATE_SELECTOR_WORD.fullmatch(word)
        assert completeness_module._list_negation_spans(word)
        assert completeness_module._selector_identifier_negation_count(word) == 1


def test_a_negation_after_a_conjunction_in_the_excerpts_clause_does_not_count():
    # A known limit: the clause ends at "and", so the check can't tell that
    # "never" scopes "consent" here, and the excerpt keeps the value its own
    # wording gives, `consent` true. The inverted encoding pairs the chapeau.
    source = _item_source(
        "Paid to a person whose consent and approval were never given"
    )
    excerpt = "Paid to a person whose consent"

    assert _pairs("consent", excerpt, source, excluded_when=True)
    assert not _pairs("consent", excerpt, source, excluded_when=False)


def test_a_cue_after_the_names_words_in_a_whole_item_reads_as_positive():
    # A known limit: the reader reads "not" before a name's words, and "is
    # not present" right after one of them, but no other "not" after them. So
    # the whole item reads as positive for `consent`, though it needs it false,
    # and the inverted encoding pairs the chapeau. Main reads the same words
    # the same way when they stand in the chapeau. An excerpt that stops before
    # the "not" gives no value.
    item = "Paid to a person whose consent was not given"
    source = _item_source(item)

    assert completeness_module._list_item_values("consent", _items(item)) == (
        (item, True),
    )
    assert _pairs("consent", item, source, excluded_when=True)
    assert not _pairs(
        "consent", "Paid to a person whose consent", source, excluded_when=True
    )


def test_an_excerpt_before_a_negation_that_scopes_other_words_gives_no_value():
    # A known limit: "nonprofit" counts as a negation after the excerpt, so the
    # excerpt gives no value, though "Subject to withholding" reads right for
    # `subject_to_withholding`. The chapeau stays unpaired, as on main.
    source = _item_source("Subject to withholding by a nonprofit")
    excerpt = "Subject to withholding"

    assert not _pairs("subject_to_withholding", excerpt, source, excluded_when=True)
    assert not _pairs("subject_to_withholding", excerpt, source, excluded_when=False)


def test_an_abbreviation_outside_the_lists_ends_the_excerpts_clause():
    # A known limit: the "Tex." of "Tex. Tax Code" is neither a legal
    # abbreviation nor one of the short ones, and an upper-case letter follows
    # it, so the clause ends there and "never" falls outside it. The excerpt
    # keeps the reader's value, `withholding` true, though the item needs it
    # false, and the inverted encoding pairs the chapeau.
    item = "Paid when withholding under Tex. Tax Code 171.0001 is never present"
    excerpt = "Paid when withholding"
    text = f"(1) {item}; or"

    assert not completeness_module._negation_follows_in_clause(
        text, completeness_module._wording_through_excerpt(text, excerpt)
    )
    assert _pairs("withholding", excerpt, _item_source(item), excluded_when=True)
    assert not _pairs("withholding", excerpt, _item_source(item), excluded_when=False)


def test_an_excerpt_before_the_word_no_in_a_form_number_gives_no_value():
    # A known limit: the checks read the "No" of "Form No. W-2" as the cue
    # "no", which follows the excerpt in its clause, so the excerpt gives no
    # value, though "Paid when withholding" reads right for `withholding`. The
    # chapeau stays unpaired, as on main.
    source = _item_source("Paid when withholding on Form No. W-2 is reported")
    excerpt = "Paid when withholding"

    assert not _pairs("withholding", excerpt, source, excluded_when=True)
    assert not _pairs("withholding", excerpt, source, excluded_when=False)


@pytest.mark.parametrize("subject", ("the except", "the all-except-one"))
def test_a_clause_with_except_before_its_verb_keeps_its_own_direction(subject: str):
    """A clause the check reads as an unless list pairs as it did before.

    "if the except are:" has the shape of a list chapeau, but the check reads
    it from every excerpt the rule cites and takes the direction from the
    clause, which sets none for `subject_to_withholding`. So the item "Not
    subject to withholding" doesn't set the direction here, as it does under
    "if they are:".
    """

    chapeau = f"Payments are excluded from income if {subject} are:"
    source = (
        f"(a) Excluded payments. {chapeau}\n\n(1) {UNWITHHELD}; or\n\n"
        f"(2) {EMPLOYER}.\n\n{PAYMENT_RECORDS}"
    )

    def pairs(excluded_when: bool) -> bool:
        negation = "" if excluded_when else "not "
        rule = _rule(
            PAYMENT_OUTPUT,
            ITEM_REFS,
            f"{negation}subject_to_withholding",
            [_atom(chapeau), _atom(UNWITHHELD)],
        )
        cases = [
            _case(
                "ordinary payment counts",
                {"subject_to_withholding": not excluded_when},
                {PAYMENT_OUTPUT: "not_holds"},
            ),
            _case(
                "the item excludes the payment",
                {"subject_to_withholding": excluded_when},
                {PAYMENT_OUTPUT: "holds"},
            ),
        ]
        found = _issues(_doc(("subject_to_withholding",), [rule]), cases, source)
        return not _chapeau_is_unpaired(found, f"if {subject} are:")

    own_reading = completeness_module._source_exception_selector_active_value(
        f"if {subject} are:", "subject_to_withholding"
    )
    item_reading = completeness_module._list_item_active_values(
        "if they are:", "subject_to_withholding", _items(UNWITHHELD)
    )

    assert item_reading == {not own_reading}
    assert pairs(excluded_when=False)
    assert not pairs(excluded_when=True)


def _sub_item_values(source: str) -> tuple[tuple[str, bool], ...]:
    """The value "A pension fund", cited below item (1), gives ``pension_fund``."""

    branches = completeness_module.recognize_source_structure(
        source, corpus_citation_path=CITATION
    )
    rule = _rule(
        PAYMENT_OUTPUT, SUB_ITEM_REFS, "pension_fund", [_atom("A pension fund")]
    )
    items = completeness_module._rule_excerpts_under_branch(
        rule,
        next(branch for branch in branches if branch.path == ("a",)),
        source_branches=branches,
        corpus_citation_path=CITATION,
    )
    return completeness_module._list_item_values("pension_fund", items)


@pytest.mark.parametrize(
    ("lead_in", "expected"),
    (
        # Each lead-in holds one negation cue.
        pytest.param(NOT_LEAD_IN, (), id="not"),
        pytest.param("Received from no source among the following:", (), id="no"),
        pytest.param(
            "Received without a claim on any of the following:", (), id="without"
        ),
        pytest.param("Received for lack of any of the following:", (), id="lack"),
        pytest.param(
            "Received by a household lacking any of the following:", (), id="lacking"
        ),
        pytest.param(
            "Received in the absence of any of the following:", (), id="absence-of"
        ),
        pytest.param(
            "Received after the household fails to claim any of the following:",
            (),
            id="fails-to",
        ),
        pytest.param(
            "Received where the members fail to claim any of the following:",
            (),
            id="fail-to",
        ),
        pytest.param("Never received from any of the following:", (), id="never"),
        pytest.param("Received from neither of the following:", (), id="neither"),
        pytest.param("Nor received from any of the following:", (), id="nor"),
        pytest.param("Received from any source except the following:", (), id="except"),
        pytest.param(
            "Received from a source other than the following:", (), id="other-than"
        ),
        pytest.param(
            "Received, unless reported, from any of the following:", (), id="unless"
        ),
        # Each of these holds one negation that is no cue to the reader: a
        # word it takes as negative only in a selector's own name, another
        # negating form, "n't" or a cue it reads after a name's word.
        pytest.param("Ineligible for any of the following:", (), id="ineligible"),
        pytest.param(
            "Disqualified from receiving any of the following:", (), id="disqualified"
        ),
        pytest.param(
            "Received by a person unable to claim any of the following:",
            (),
            id="unable",
        ),
        pytest.param(
            "Received by a household that lacks any of the following:", (), id="lacks"
        ),
        pytest.param(
            "Received after the household failed to claim any of the following:",
            (),
            id="failed-to",
        ),
        pytest.param(
            "Received by a household failing to claim any of the following:",
            (),
            id="failing-to",
        ),
        pytest.param("Refused by each of the following:", (), id="refused"),
        pytest.param("Received free of any of the following:", (), id="free-of"),
        pytest.param(
            "Received by a nonmember from any of the following:", (), id="non"
        ),
        pytest.param("Isn't received from any of the following:", (), id="n-t"),
        pytest.param(
            "Received when consent is absent from any of the following:",
            (),
            id="is-absent",
        ),
        pytest.param("Received from none of the following:", (), id="none"),
        pytest.param(
            "Received by a person who cannot claim any of the following:",
            (),
            id="cannot",
        ),
        pytest.param(
            "Received by a household that lacked any of the following:",
            (),
            id="lacked",
        ),
        pytest.param(
            "Received by a household missing any of the following:", (), id="missing"
        ),
        pytest.param(
            "Received while the owner was absent from any of the following:",
            (),
            id="absent",
        ),
        pytest.param("INELIGIBLE FOR ANY OF THE FOLLOWING:", (), id="upper-case"),
        # "no" ends "casino" and starts "nominee", but neither word is a cue.
        pytest.param(
            "Received at a casino or from a nominee of any of the following:",
            (("A pension fund", True),),
            id="no-cue",
        ),
        # "nonce", "canonical", "freedom of", "Lacksley" and "tunable" each
        # hold one of the forms without being one.
        pytest.param(
            "Received as a nonce payment from any of the following:",
            (("A pension fund", True),),
            id="nonce",
        ),
        pytest.param(
            "Received from a canonical list of the following:",
            (("A pension fund", True),),
            id="canonical",
        ),
        pytest.param(
            "Received in freedom of choice from any of the following:",
            (("A pension fund", True),),
            id="freedom-of",
        ),
        pytest.param(
            "Received by Lacksley from any of the following:",
            (("A pension fund", True),),
            id="lacksley",
        ),
        pytest.param(
            "Received on a tunable schedule from any of the following:",
            (("A pension fund", True),),
            id="tunable",
        ),
    ),
)
def test_a_negation_cue_in_the_items_lead_in_leaves_a_sub_item_without_a_value(
    lead_in: str, expected: tuple[tuple[str, bool], ...]
):
    # The reader would read "(i) A pension fund" as positive. A negation in
    # the lead-in above it may turn that around, so any negation the item
    # checks know, whether the reader reads it or reads past it, leaves the
    # sub-item without a value.
    assert _sub_item_values(_lead_in_source(lead_in)) == expected


@pytest.mark.parametrize("excluded_when", (False, True), ids=("faithful", "inverted"))
@pytest.mark.parametrize(
    "lead_in",
    (
        "Ineligible for any of the following:",
        "Disqualified from receiving any of the following:",
        "Received by a household that lacks any of the following:",
        "Refused by each of the following:",
        "Received free of any of the following:",
        "Received from none of the following:",
        "Received by a person who cannot claim any of the following:",
        "Received by a household that lacked any of the following:",
        "Received by a household missing any of the following:",
    ),
)
def test_a_lead_in_with_a_negation_the_reader_reads_past_pairs_neither_encoding(
    lead_in: str, excluded_when: bool
):
    """A sub-item below a negated lead-in pairs neither encoding.

    The reader would read "(i) A pension fund" as positive, so on that value
    alone the encoding that excludes a payment when `pension_fund` is true
    would pair the chapeau, though the lead-in turns the item around. The
    sub-item gives no value instead, and the chapeau stays unpaired either
    way, as on main.
    """

    assert not _pairs(
        "pension_fund",
        "A pension fund",
        _lead_in_source(lead_in),
        excluded_when=excluded_when,
        source_ref=SUB_ITEM_REFS,
    )


def _two_level_source(item_lead_in: str, sub_item_lead_in: str) -> str:
    """Item (1) holds sub-item (i), which holds "(A) A pension fund"."""

    return (
        f"(a) Excluded payments. {EXCLUSION_CHAPEAU}\n\n(1) {item_lead_in}\n\n"
        f"(i) {sub_item_lead_in}\n\n(A) A pension fund; or\n\n"
        "(B) An annuity fund; or\n\n(ii) A trust; or\n\n"
        f"(2) {EMPLOYER}.\n\n{PAYMENT_RECORDS}"
    )


@pytest.mark.parametrize(
    ("item_lead_in", "sub_item_lead_in", "expected"),
    (
        pytest.param(
            NOT_LEAD_IN, "A fund of any of these kinds:", (), id="cue-in-the-item"
        ),
        pytest.param(
            "Received from any of the following:",
            "A fund other than one of these kinds:",
            (),
            id="cue-in-the-sub-item",
        ),
        pytest.param(
            "Received from any of the following:",
            "A fund of any of these kinds:",
            (("A pension fund", True),),
            id="no-cue",
        ),
    ),
)
def test_every_lead_in_from_the_item_down_to_the_excerpt_counts(
    item_lead_in: str,
    sub_item_lead_in: str,
    expected: tuple[tuple[str, bool], ...],
):
    assert (
        _sub_item_values(_two_level_source(item_lead_in, sub_item_lead_in)) == expected
    )


def test_a_cue_in_an_earlier_sub_item_does_not_count():
    # The lead-in stops where the first sub-item starts, so the "not" in (i)
    # says nothing about (ii).
    source = _lead_in_source(
        "Received from any of the following:",
        first="An annuity not paid monthly",
        second="A pension fund",
    )

    assert _sub_item_values(source) == (("A pension fund", True),)


@pytest.mark.parametrize(
    ("lead_in", "first", "second", "expected"),
    (
        pytest.param(
            NOT_LEAD_IN, "A pension fund", "An annuity", (), id="not-in-this-item"
        ),
        pytest.param(
            "Received from any of the following:",
            "An annuity not paid monthly",
            "A pension fund",
            (("A pension fund", True),),
            id="cue-in-an-earlier-sub-item",
        ),
    ),
)
def test_lead_ins_come_from_the_branches_that_hold_the_excerpt(
    lead_in: str, first: str, second: str, expected: tuple[tuple[str, bool], ...]
):
    # This unit has two paragraphs labeled (a), so two branches share each
    # path from (a) down. The lead-ins come from the branches that hold the
    # excerpt, not from the first branch on the same path.
    source = (
        "(a) Definitions. In this section:\n\n(1) Fund means any of the following:"
        "\n\n(i) A trust; or\n\n(ii) An account.\n\n"
        f"{PAYMENT_RECORDS}\n" + _lead_in_source(lead_in, first=first, second=second)
    )
    branches = completeness_module.recognize_source_structure(
        source, corpus_citation_path=CITATION
    )
    rule = _rule(
        PAYMENT_OUTPUT, SUB_ITEM_REFS, "pension_fund", [_atom("A pension fund")]
    )

    items = completeness_module._rule_excerpts_under_branch(
        rule,
        next(
            branch
            for branch in branches
            if branch.path == ("a",) and EXCLUSION_CHAPEAU in branch.text
        ),
        source_branches=branches,
        corpus_citation_path=CITATION,
    )

    assert [branch.path for branch in branches].count(("a", "1")) == 2
    assert completeness_module._list_item_values("pension_fund", items) == expected


# A second paragraph (a), with a list of its own, repeats the chapeau's paths.
DEFINITIONS_LIST = (
    "(a) Definitions. In this section, fund means any of the following:\n\n"
    "(1) A pension fund; or\n\n(2) An annuity.\n\n"
    "(b) Reports. The agency must publish a report.\n"
)


def _repeated_labels_source(last_item_ending: str = ".") -> str:
    """The chapeau's list in the first (a), then a second (a) with its own."""

    return (
        f"(a) Excluded payments. {EXCLUSION_CHAPEAU}\n\n(1) {UNWITHHELD}; or\n\n"
        f"(2) {EMPLOYER}{last_item_ending}\n\n{PAYMENT_RECORDS}\n{DEFINITIONS_LIST}"
    )


def _chapeau_clause(
    source: str,
    branches: tuple[completeness_module.SourceStructureBranch, ...],
    chapeau: str = EXCLUSION_CHAPEAU,
    occurrence: int = 0,
) -> completeness_module.SourceStructureBranch:
    """The chapeau as the check sees it: its owner's path, its own span.

    ``occurrence`` picks one of several copies of ``chapeau`` in ``source``.
    """

    start = -1
    for _ in range(occurrence + 1):
        start = source.index(chapeau, start + 1)
    owner = next(
        branch
        for branch in branches
        if branch.path == ("a",) and branch.start <= start < branch.end
    )
    return completeness_module.SourceStructureBranch(
        owner.path,
        "exception-clause",
        owner.label,
        chapeau,
        start,
        start + len(chapeau),
    )


def test_an_excerpt_in_a_later_branch_on_an_items_path_is_not_an_item():
    source = _repeated_labels_source()
    branches = completeness_module.recognize_source_structure(
        source, corpus_citation_path=CITATION
    )
    rule = _rule(
        PAYMENT_OUTPUT, ITEM_REFS, "true", [_atom("A pension fund"), _atom(UNWITHHELD)]
    )

    found = completeness_module._rule_excerpts_under_branch(
        rule,
        _chapeau_clause(source, branches),
        source_branches=branches,
        corpus_citation_path=CITATION,
    )

    assert [branch.path for branch in branches].count(("a", "1")) == 2
    # "A pension fund" sits in the second (a)(1). It shares the path of the
    # chapeau's item (1), but lies outside the chapeau's (a).
    assert tuple(item.excerpt for item in found) == (UNWITHHELD,)


def test_a_list_ends_within_its_own_paragraph():
    # (2) ends with "; or", and nothing after it in the first (a) ends the
    # list, so the list has no items. The second (a)'s list ends a sentence,
    # but its items lie outside the chapeau's (a).
    source = _repeated_labels_source("; or")
    branches = completeness_module.recognize_source_structure(
        source, corpus_citation_path=CITATION
    )

    assert (
        completeness_module._chapeau_list_item_paths(
            _chapeau_clause(source, branches), branches
        )
        == frozenset()
    )
    assert _item_paths(source, ("a",)) == frozenset()


def test_a_clause_outside_every_branch_on_its_path_has_no_items():
    # A clause carries its paragraph's path. This one starts in (b), between
    # the two paragraphs labeled (a), so no branch on its path holds its
    # start and nothing bounds its list. It has no items, rather than every
    # later child on that path.
    source = _repeated_labels_source()
    branches = completeness_module.recognize_source_structure(
        source, corpus_citation_path=CITATION
    )
    start = source.index(PAYMENT_RECORDS)
    clause = completeness_module.SourceStructureBranch(
        ("a",),
        "exception-clause",
        "(a)",
        EXCLUSION_CHAPEAU,
        start,
        start + len(EXCLUSION_CHAPEAU),
    )
    paragraphs = [branch for branch in branches if branch.path == ("a",)]

    assert len(paragraphs) == 2
    assert not any(branch.start <= start < branch.end for branch in paragraphs)
    # The second (a) starts after the clause, and its own list ends a
    # sentence, so its children would be the items if nothing bounded them.
    assert start < paragraphs[1].start
    assert completeness_module._chapeau_list_item_paths(paragraphs[1], branches) == {
        ("a", "1"),
        ("a", "2"),
    }
    assert completeness_module._chapeau_list_items(clause, branches) == ()


def test_a_later_list_on_the_chapeaus_paths_does_not_pair_it():
    source = _repeated_labels_source()

    # "A pension fund" reads as positive, so this is the encoding it would
    # pair if it were one of the chapeau's items.
    assert not _pairs("pension_fund", "A pension fund", source, excluded_when=True)
    # The chapeau's own item still pairs it.
    assert _pairs("subject_to_withholding", UNWITHHELD, source, excluded_when=False)


@pytest.mark.parametrize(
    ("selector", "excerpt", "excluded_when"),
    (
        pytest.param("pension_fund", "A pension fund", True, id="later-lists-item"),
        pytest.param("subject_to_withholding", UNWITHHELD, False, id="own-item"),
    ),
)
def test_an_unended_list_does_not_take_a_later_lists_end(
    selector: str, excerpt: str, excluded_when: bool
):
    assert not _pairs(
        selector,
        excerpt,
        _repeated_labels_source("; or"),
        excluded_when=excluded_when,
    )


def test_each_item_excerpt_carries_the_lead_ins_before_it():
    rule = _rule(
        PAYMENT_OUTPUT,
        SUB_ITEM_REFS,
        "true",
        [
            _atom("A pension fund"),
            _atom("Not received from any of the following: (i) A pension fund"),
            _atom(NOT_LEAD_IN),
            _atom(EMPLOYER),
        ],
    )
    source = _lead_in_source(NOT_LEAD_IN)
    branches = completeness_module.recognize_source_structure(
        source, corpus_citation_path=CITATION
    )

    found = completeness_module._rule_excerpts_under_branch(
        rule,
        next(branch for branch in branches if branch.path == ("a",)),
        source_branches=branches,
        corpus_citation_path=CITATION,
    )

    lead_in = f"(1) {NOT_LEAD_IN}"
    # An excerpt from (i) has (1)'s lead-in before it, and so does one that
    # runs on from that lead-in into (i). One that ends with the lead-in has
    # it in its own wording instead, and (2) has no sub-list.
    assert found == (
        completeness_module._ListItemExcerpt(
            "A pension fund", "(i) A pension fund", (lead_in,)
        ),
        completeness_module._ListItemExcerpt(
            "Not received from any of the following: (i) A pension fund",
            f"{lead_in} (i) A pension fund",
            (lead_in,),
        ),
        completeness_module._ListItemExcerpt(NOT_LEAD_IN, lead_in),
        completeness_module._ListItemExcerpt(EMPLOYER, f"(2) {EMPLOYER}"),
    )


def test_a_lead_in_is_the_whole_branch_when_its_first_childs_text_is_missing():
    # The recognizer builds some branches' text apart from their children's,
    # so a first child's text may not appear in its parent's. The whole
    # parent text then stands in for the lead-in, and its cue still counts.
    branch_type = completeness_module.SourceStructureBranch
    item = branch_type(("a", "1"), "paragraph", "(1)", f"(1) {NOT_LEAD_IN}", 0, 60)
    sub_item = branch_type(
        ("a", "1", "i"), "paragraph", "(i)", "(i) A pension fund; or", 45, 70
    )

    assert completeness_module._branch_lead_in(item, (item, sub_item)) == (
        f"(1) {NOT_LEAD_IN}"
    )


def _section_issues(chapeau: str, source: str) -> tuple[str, ...]:
    """The issues for a module whose case pairs toggle the section's items."""

    inputs = (NOT_FINISHED_INPUT, APPROVED_INPUT, TAUGHT_INPUT, NAMED_PROVIDER_INPUT)
    rule = _rule(
        OUTPUT,
        SECTION_LIST_REFS,
        "\nor ".join(inputs),
        [
            _atom(chapeau),
            _atom(NOT_FINISHED),
            _atom(APPROVED),
            _atom(TAUGHT),
            _atom(NAMED_PROVIDER),
        ],
    )
    rows = (
        ("ordinary hour counts", (False, False, False, False), "not_holds"),
        ("unfinished course hour", (True, False, False, False), "holds"),
        ("approved course hour", (False, True, False, False), "holds"),
        ("hour the licensee taught", (False, False, True, False), "holds"),
        ("hour from a named provider", (False, False, False, True), "holds"),
    )
    cases = [
        _case(name, dict(zip(inputs, values)), {OUTPUT: result})
        for name, values, result in rows
    ]
    return _issues(_doc(inputs, [rule]), cases, source)


def test_a_whole_section_pairs_its_chapeau_with_its_item_pairs():
    assert not _chapeau_is_unpaired(_section_issues(SECTION_CHAPEAU, SECTION_SOURCE))


def test_a_chapeau_with_a_cross_reference_pairs_with_its_item_pairs():
    # The labels in "paragraph (b)(1) of this section" leave the chapeau's
    # items as they are, so it pairs and keeps its four items.
    assert not _chapeau_is_unpaired(
        _section_issues(SECTION_XREF_CHAPEAU, SECTION_XREF_SOURCE)
    )
    assert _item_paths(SECTION_XREF_SOURCE, ("c", "2")) == {
        ("c", "2", "i"),
        ("c", "2", "ii"),
        ("c", "2", "iii"),
        ("c", "2", "iv"),
    }


@pytest.mark.parametrize("faithful", (True, False))
def test_a_change_chapeau_pairs_a_positive_name_in_either_direction(faithful: bool):
    # "Leave out" reads as a change, which a pair meets in either direction,
    # as it does when the same words stand inline. So a positive name for
    # item (i) pairs whichever way the formula reads "Not finished".
    finished = "finished_by_course_deadline"
    inputs = (finished, APPROVED_INPUT, TAUGHT_INPUT, NAMED_PROVIDER_INPUT)
    item_one = f"not {finished}" if faithful else finished
    rule = _rule(
        OUTPUT,
        SECTION_LIST_REFS,
        "\nor ".join((item_one, *inputs[1:])),
        [
            _atom(SECTION_CHAPEAU),
            _atom(NOT_FINISHED),
            _atom(APPROVED),
            _atom(TAUGHT),
            _atom(NAMED_PROVIDER),
        ],
    )
    rows = (
        ("ordinary hour counts", (faithful, False, False, False), "not_holds"),
        (
            "hour the formula reads as left out",
            (not faithful, False, False, False),
            "holds",
        ),
        ("approved course hour", (faithful, True, False, False), "holds"),
        ("hour the licensee taught", (faithful, False, True, False), "holds"),
        ("hour from a named provider", (faithful, False, False, True), "holds"),
    )
    cases = [
        _case(name, dict(zip(inputs, values)), {OUTPUT: result})
        for name, values, result in rows
    ]

    assert not _chapeau_is_unpaired(
        _issues(_doc(inputs, [rule]), cases, SECTION_SOURCE)
    )


def test_analysis_leaves_material_unchanged():
    doc, cases = _material()
    before = copy.deepcopy((doc, cases))

    _issues(doc, cases)

    assert (doc, cases) == before


# Words no source text, stopword list, concept pattern, or stem rule touches.
_NONCE_WORDS = (
    "blorvik",
    "quandor",
    "zephtil",
    "mornak",
    "trivolt",
    "jaskum",
    "plindor",
    "vextral",
)
_CHAPEAUS = (
    "if they are:",
    "if the payments are:",
    "when it is—",
    "if the person is –",
    "when the items are:",
)
_WORD_SETS = st.lists(
    st.lists(st.sampled_from(_NONCE_WORDS), min_size=1, max_size=6, unique=True),
    max_size=4,
)


def _excerpts(word_sets: list[list[str]]) -> tuple[str, ...]:
    return tuple(f"the {' and '.join(words)} item" for words in word_sets)


@settings(max_examples=300, deadline=None)
@given(
    chapeau=st.sampled_from(_CHAPEAUS),
    selector=st.lists(
        st.sampled_from(_NONCE_WORDS), min_size=1, max_size=4, unique=True
    ),
    items=_WORD_SETS,
    elsewhere=_WORD_SETS,
)
def test_list_chapeau_relevance_is_one_item_naming_every_selector_word(
    chapeau: str,
    selector: list[str],
    items: list[list[str]],
    elsewhere: list[list[str]],
):
    assert completeness_module._source_exception_selector_is_relevant(
        chapeau,
        "_".join(selector),
        supporting_texts=_excerpts(items + elsewhere),
        item_texts=_excerpts(items),
    ) == any(set(selector) <= set(words) for words in items)


@settings(max_examples=300, deadline=None)
@given(
    chapeau=st.sampled_from(_CHAPEAUS),
    words=st.lists(st.sampled_from(_NONCE_WORDS), min_size=1, max_size=3, unique=True),
    item_is_negative=st.booleans(),
    name_is_negative=st.booleans(),
    excerpt_drops_not=st.booleans(),
)
def test_an_items_direction_is_its_wording_against_the_selectors_name(
    chapeau: str,
    words: list[str],
    item_is_negative: bool,
    name_is_negative: bool,
    excerpt_drops_not: bool,
):
    phrase = " ".join(words)
    item = f"Not {phrase}" if item_is_negative else phrase
    excerpt = phrase if excerpt_drops_not else item
    selector = "_".join(("not", *words) if name_is_negative else words)

    assert completeness_module._list_item_active_values(
        chapeau,
        selector,
        (completeness_module._ListItemExcerpt(excerpt, f"(1) {item}"),),
    ) == {item_is_negative == name_is_negative}


# Words and stops to put between a cue and the name's words. None of them
# negates, and none matches a nonce word. A comma stops the reader's cue, and
# "when" and "and is" start another predicate.
_FILLER_WORDS = st.sampled_from(
    (
        ",",
        "in",
        "the",
        "agency's",
        "view",
        "at",
        "any",
        "time",
        "during",
        "calendar",
        "year",
        "of",
        "reporting",
        "period",
        "by",
        "an",
        "employer's",
        "when",
        "and is",
    )
)


@settings(max_examples=300, deadline=None)
@given(
    cue=st.sampled_from(("Not", "No longer", "Without", "Lacking")),
    filler=st.lists(_FILLER_WORDS, max_size=12),
    words=st.lists(st.sampled_from(_NONCE_WORDS), min_size=1, max_size=3, unique=True),
    name_is_negative=st.booleans(),
    excerpt_is_the_phrase=st.booleans(),
)
def test_a_cue_before_the_names_words_never_gives_the_positive_reading(
    cue: str,
    filler: list[str],
    words: list[str],
    name_is_negative: bool,
    excerpt_is_the_phrase: bool,
):
    """An item that opens with a cue gives the negated value or none.

    Whatever stands between the cue and the name's words, the item either
    gives the value it needs, read with the cue negating the name's words, or
    gives no value. It never gives the value the words would need without
    the cue.
    """

    phrase = " ".join(words)
    between = "".join(word if word == "," else f" {word}" for word in filler)
    item = f"{cue}{between} {phrase}"
    excerpt = phrase if excerpt_is_the_phrase else item
    selector = "_".join(("not", *words) if name_is_negative else words)
    values = {
        value
        for _excerpt, value in completeness_module._list_item_values(
            selector, _items((excerpt, f"(1) {item}"))
        )
    }

    assert values <= {name_is_negative}


# Each template words an item with a negator the reader has no cue for, ahead
# of the words that name the selector.
_UNSCOPABLE_NEGATOR_ITEMS = st.sampled_from(
    (
        "Never {phrase}",
        "Neither {phrase} nor withheld",
        "Not withheld, nor {phrase}",
        "From any source except {phrase}",
        "From a source other than {phrase}",
        "Paid unless {phrase}",
    )
)


# Each example runs the whole check twice, so this property draws fewer
# examples than the ones that call a single helper.
@settings(max_examples=100, deadline=None)
@given(
    template=_UNSCOPABLE_NEGATOR_ITEMS,
    words=st.lists(st.sampled_from(_NONCE_WORDS), min_size=1, max_size=3, unique=True),
    name_is_negative=st.booleans(),
    excerpt_is_the_phrase=st.booleans(),
)
def test_an_item_with_an_unscopable_negator_pairs_neither_encoding(
    template: str,
    words: list[str],
    name_is_negative: bool,
    excerpt_is_the_phrase: bool,
):
    """An item with a negator the reader can't scope pairs neither encoding.

    This holds for any words the item names the selector with, either polarity
    of the selector's name, and an excerpt that cites the whole item or only
    the words after the negator.
    """

    phrase = " ".join(words)
    item = template.format(phrase=phrase)
    excerpt = phrase if excerpt_is_the_phrase else item
    selector = "_".join(("not", *words) if name_is_negative else words)
    source = _item_source(item)

    assert not _pairs(selector, excerpt, source, excluded_when=False)
    assert not _pairs(selector, excerpt, source, excluded_when=True)


_CUES = st.sampled_from(("if", "when", "If", "When"))
_VERBS = st.sampled_from(("is", "are", "be"))
_ENDINGS = st.sampled_from((":", "—", "–", "-", " :", " —", " – "))
_SUBJECTS = st.one_of(
    st.sampled_from(("they", "it", "he", "she", "he or she")),
    st.builds(
        lambda determiner, words: " ".join((determiner, *words)),
        st.sampled_from(
            (
                "the",
                "a",
                "an",
                "such",
                "its",
                "their",
                "his",
                "her",
                "each",
                "this",
                "that",
                "these",
                "those",
            )
        ),
        st.lists(st.sampled_from(_NONCE_WORDS), min_size=1, max_size=1),
    ),
)
# Each template adds a condition of the clause's own around a bare subject: a
# second word that could modify it, a verb, or a clause.
_OWN_CONDITION_TEMPLATES = st.sampled_from(
    (
        "{cue} {subject} so elects, the deductions {verb}{ending}",
        "{cue} {subject} who files a request {verb}{ending}",
        "{cue} {subject} and the payments {verb}{ending}",
        "{cue}, for the year, {subject} {verb}{ending}",
        "{cue} the household so elects, when {subject} {verb}{ending}",
        "{cue} {subject} {verb} not{ending}",
        "{cue} {subject} vextral {verb}{ending}",
        "{cue} the late {subject} {verb}{ending}",
        "{cue} {subject} elects to {verb}{ending}",
        "{cue} {subject} chooses to be{ending}",
    )
)


@settings(max_examples=300, deadline=None)
@given(cue=_CUES, subject=_SUBJECTS, verb=_VERBS, ending=_ENDINGS)
def test_any_bare_subject_list_chapeau_takes_its_items(
    cue: str, subject: str, verb: str, ending: str
):
    chapeau = f"{cue} {subject} {verb}{ending}"

    assert _relevant(chapeau, item_texts=(SSI_ITEM,))
    assert not _relevant(chapeau)


@settings(max_examples=300, deadline=None)
@given(
    template=_OWN_CONDITION_TEMPLATES,
    cue=_CUES,
    subject=_SUBJECTS,
    verb=_VERBS,
    ending=_ENDINGS,
)
def test_a_clause_with_a_condition_of_its_own_never_takes_its_items(
    template: str, cue: str, subject: str, verb: str, ending: str
):
    clause = template.format(cue=cue, subject=subject, verb=verb, ending=ending)

    assert not _relevant(clause, item_texts=(SSI_ITEM,))


# Bare subjects whose one word is or holds "unless" or "except", beside ones
# whose word only starts with it and ones that hold neither.
_BARE_AND_EXCEPT_SUBJECTS = st.one_of(
    st.sampled_from(("they", "it", "he", "she", "he or she")),
    st.builds(
        " ".join,
        st.tuples(
            st.sampled_from(("the", "such", "its", "each")),
            st.sampled_from(
                (
                    "payments",
                    "members",
                    "exceptions",
                    "unlessened",
                    "except",
                    "unless",
                    "all-except-one",
                    "except's",
                )
            ),
        ),
    ),
)


@settings(max_examples=300, deadline=None)
@given(
    cue=_CUES,
    subject=_BARE_AND_EXCEPT_SUBJECTS,
    verb=_VERBS,
    ending=_ENDINGS,
    words=st.lists(st.sampled_from(_NONCE_WORDS), min_size=1, max_size=4, unique=True),
    supporting=_WORD_SETS,
)
def test_under_a_list_chapeau_excerpts_outside_its_items_add_no_relevance(
    cue: str,
    subject: str,
    verb: str,
    ending: str,
    words: list[str],
    supporting: list[list[str]],
):
    """A list chapeau never reads relevance from the rule's other excerpts.

    So wherever the check reads a selector's relevance and direction from a
    clause and every cited excerpt, as it reads an unless list, the clause is
    no list chapeau and its items change nothing.
    """

    condition = f"{cue} {subject} {verb}{ending}"
    selector = "_".join(words)
    texts = (*_excerpts(supporting), f"the {' and '.join(words)} item")
    is_list_chapeau = bool(
        completeness_module._IF_LIST_CHAPEAU_PATTERN.fullmatch(
            completeness_module._collapse_text(condition).lower()
        )
    )

    assert is_list_chapeau == (
        subject.split()[-1] not in {"except", "unless", "all-except-one", "except's"}
    )
    assert completeness_module._source_exception_selector_is_relevant(
        condition, selector, supporting_texts=texts
    ) == (not is_list_chapeau)
    assert completeness_module._source_exception_selector_is_relevant(
        condition, selector, supporting_texts=texts, item_texts=texts
    )


_CONDITIONS = (
    *_CHAPEAUS,
    ELECTION_CHAPEAU,
    "if the person is a resident of the state",
    "unless he or she is—",
    "if the person meets at least one of the criteria",
    "if the household's countable resources are:",
    "if the household's countable liquid resources are:",
    "when the member is a student",
)
_SELECTORS = (
    "person_is_resident_of_state",
    "household_countable_resources",
    "countable_liquid_resources",
    "member_is_student",
    SSI_INPUT,
    ELECTION_INPUT,
    "_".join(_NONCE_WORDS[:3]),
)
_EXCERPT_TEXTS = st.lists(
    st.sampled_from(
        (
            "a resident of the state",
            "countable resources",
            "a member who is a student",
            SSI_ITEM,
            MEDICAL,
            " ".join(_NONCE_WORDS[:3]),
            "an unrelated phrase",
        )
    ),
    max_size=3,
)


@settings(max_examples=300, deadline=None)
@given(
    condition=st.sampled_from(_CONDITIONS),
    selector=st.sampled_from(_SELECTORS),
    supporting=_EXCERPT_TEXTS,
    items=_EXCERPT_TEXTS,
)
def test_cited_excerpts_only_add_relevance(
    condition: str, selector: str, supporting: list[str], items: list[str]
):
    bare = completeness_module._source_exception_selector_is_relevant(
        condition, selector
    )
    without_items = completeness_module._source_exception_selector_is_relevant(
        condition, selector, supporting_texts=tuple(supporting)
    )
    with_items = completeness_module._source_exception_selector_is_relevant(
        condition,
        selector,
        supporting_texts=tuple(supporting),
        item_texts=tuple(items),
    )

    assert bare <= without_items <= with_items


# A sentence may end before a closing quote, as in a defined term's last item.
_SENTENCE_ENDINGS = (".", '."')
_SEPARATOR_ENDINGS = (";", "; or", "; and", "; and/or", ",", ", or", ", and")
_UNENDED_ENDINGS = ("", " or", " and")
_ITEM_ENDINGS = st.sampled_from(
    (*_SEPARATOR_ENDINGS, *_SENTENCE_ENDINGS, *_UNENDED_ENDINGS)
)


@settings(max_examples=300, deadline=None)
@given(endings=st.lists(_ITEM_ENDINGS, min_size=1, max_size=6))
def test_list_items_are_the_children_through_the_first_sentence_end(
    endings: list[str],
):
    items = "".join(
        f"({number}) The {word} item{ending}\n\n"
        for number, (word, ending) in enumerate(zip(_NONCE_WORDS, endings), start=1)
    )
    source = (
        f"(a) Hours left out. The tally leaves out hours if they are:\n\n{items}"
        "(b) Records. The board must keep records.\n"
    )
    # The list ends at the first child that doesn't end with a separator. It
    # has items only when that child ends a sentence.
    last = next(
        (
            number
            for number, ending in enumerate(endings, start=1)
            if ending not in _SEPARATOR_ENDINGS
        ),
        None,
    )
    expected = (
        frozenset(("a", str(number)) for number in range(1, last + 1))
        if last is not None and endings[last - 1] in _SENTENCE_ENDINGS
        else frozenset()
    )

    assert _item_paths(source, ("a",)) == expected


@settings(max_examples=200, deadline=None)
@given(
    endings=st.lists(st.sampled_from((".", "; or")), min_size=2, max_size=4),
    data=st.data(),
)
def test_an_excerpt_counts_only_in_its_own_chapeaus_items(
    endings: list[str], data: st.DataObject
):
    """Each copy of a paragraph keeps its own list.

    Every group repeats the same chapeau and paths, (a), (a)(1) and (a)(2),
    and each item names its own word. An excerpt counts for the chapeau of
    one group only when it sits in that group's items, and only when that
    group's list ends a sentence.
    """

    groups = len(endings)
    chapeau_at = data.draw(st.integers(min_value=0, max_value=groups - 1))
    cited = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=2 * groups - 1),
            min_size=1,
            max_size=4,
            unique=True,
        )
    )
    source = "".join(
        f"(a) Excluded payments. {EXCLUSION_CHAPEAU}\n\n"
        f"(1) The {_NONCE_WORDS[2 * group]} item; or\n\n"
        f"(2) The {_NONCE_WORDS[2 * group + 1]} item{ending}\n\n"
        f"{PAYMENT_RECORDS}\n"
        for group, ending in enumerate(endings)
    )
    branches = completeness_module.recognize_source_structure(
        source, corpus_citation_path=CITATION
    )
    excerpts = tuple(f"The {_NONCE_WORDS[index]} item" for index in cited)
    rule = _rule(
        PAYMENT_OUTPUT, ITEM_REFS, "true", [_atom(excerpt) for excerpt in excerpts]
    )

    found = completeness_module._rule_excerpts_under_branch(
        rule,
        _chapeau_clause(source, branches, occurrence=chapeau_at),
        source_branches=branches,
        corpus_citation_path=CITATION,
    )

    assert tuple(item.excerpt for item in found) == tuple(
        excerpt
        for index, excerpt in zip(cited, excerpts)
        if index // 2 == chapeau_at and endings[chapeau_at] == "."
    )


# Each template puts two negation cues before the words that name the
# selector, so the item needs those words to hold.
_DOUBLE_NEGATION_ITEMS = st.sampled_from(
    (
        "Not without {phrase}",
        "Not lacking {phrase}",
        "No longer without {phrase}",
        "Not in the absence of {phrase}",
        "Not paid without {phrase}",
        "Isn't without {phrase}",
        "Paid where the person does not fail to {phrase}",
    )
)


@settings(max_examples=300, deadline=None)
@given(
    template=_DOUBLE_NEGATION_ITEMS,
    words=st.lists(st.sampled_from(_NONCE_WORDS), min_size=1, max_size=3, unique=True),
    name_is_negative=st.booleans(),
    excerpt_is_the_phrase=st.booleans(),
)
def test_an_item_with_two_negation_cues_gives_no_value(
    template: str,
    words: list[str],
    name_is_negative: bool,
    excerpt_is_the_phrase: bool,
):
    phrase = " ".join(words)
    item = template.format(phrase=phrase)
    excerpt = phrase if excerpt_is_the_phrase else item
    selector = "_".join(("not", *words) if name_is_negative else words)

    assert (
        completeness_module._list_item_values(
            selector, (completeness_module._ListItemExcerpt(excerpt, f"(1) {item}"),)
        )
        == ()
    )


# Each template joins a cue the reader knows with a negation it doesn't
# combine with that cue: a word or form it reads past before the words that
# name the selector, or a cue it reads after them.
_CUE_WITH_UNCOMBINED_NEGATION_ITEMS = st.sampled_from(
    (
        "Not disqualified from {phrase}",
        "Not ineligible for {phrase}",
        "Not unable to claim {phrase}",
        "No person lacks {phrase}",
        "Not failing to claim {phrase}",
        "Not refused {phrase}",
        "Not free of {phrase}",
        "Not a non-{phrase}",
        "Without being unqualified for {phrase}",
        "Not a case in which {phrase} is absent",
        "Not a case in which {phrase} is not present",
    )
)


@settings(max_examples=300, deadline=None)
@given(
    template=_CUE_WITH_UNCOMBINED_NEGATION_ITEMS,
    words=st.lists(st.sampled_from(_NONCE_WORDS), min_size=1, max_size=3, unique=True),
    name_is_negative=st.booleans(),
    excerpt_is_the_phrase=st.booleans(),
)
def test_an_item_with_a_cue_and_an_uncombined_negation_gives_no_value(
    template: str,
    words: list[str],
    name_is_negative: bool,
    excerpt_is_the_phrase: bool,
):
    phrase = " ".join(words)
    item = template.format(phrase=phrase)
    excerpt = phrase if excerpt_is_the_phrase else item
    selector = "_".join(("not", *words) if name_is_negative else words)

    assert (
        completeness_module._list_item_values(
            selector, (completeness_module._ListItemExcerpt(excerpt, f"(1) {item}"),)
        )
        == ()
    )


_POSTPOSED_CUES = (
    "is absent",
    "is not present",
    "fehlt",
    "fehlen",
    "nicht vorhanden",
    "nicht gegeben",
    "liegt nicht vor",
)


@settings(max_examples=200, deadline=None)
@given(
    words=st.lists(st.sampled_from(_NONCE_WORDS), min_size=1, max_size=3, unique=True),
    cue=st.sampled_from(_POSTPOSED_CUES),
    name_is_negative=st.booleans(),
    data=st.data(),
)
def test_an_excerpt_cut_before_the_end_of_a_postposed_cue_gives_no_value(
    words: list[str], cue: str, name_is_negative: bool, data: st.DataObject
):
    """An item gives a value only when its excerpt holds the whole cue.

    The item is "Paid when <words> <cue>", and the excerpt stops anywhere from
    the end of the words to one character short of the cue's end. Cited
    whole, the item reads against the selector's name when the name has one
    word, since the reader applies the cue to the word right before it.
    """

    phrase = " ".join(words)
    item = f"Paid when {phrase} {cue}"
    stem = f"Paid when {phrase}"
    cut = data.draw(st.integers(min_value=len(stem), max_value=len(item) - 1))
    selector = "_".join(("not", *words) if name_is_negative else words)
    source = _item_source(item)
    branches = completeness_module.recognize_source_structure(
        source, corpus_citation_path=CITATION
    )

    def values(excerpt: str) -> tuple[tuple[str, bool], ...]:
        return completeness_module._list_item_values(
            selector,
            completeness_module._rule_excerpts_under_branch(
                _rule(PAYMENT_OUTPUT, ITEM_REFS, "true", [_atom(excerpt)]),
                next(branch for branch in branches if branch.path == ("a",)),
                source_branches=branches,
                corpus_citation_path=CITATION,
            ),
        )

    assert values(item[:cut]) == ()
    if len(words) == 1:
        assert values(item) == ((item, name_is_negative),)


# Each tail negates the words before it. The second entry is the negation's
# first word or form, as the item spells it.
_CLAUSE_NEGATIONS = (
    ("was never recorded", "never"),
    ("was not recorded", "not"),
    ("isn't recorded", "isn't"),
    ("is no longer recorded", "no"),
    ("is neither recorded nor filed", "neither"),
    ("applies unless waived", "unless"),
    ("applies except in a trust", "except"),
    ("applies other than in a trust", "other than"),
    ("is without a record", "without"),
    ("lacks a record", "lacks"),
    ("failed to apply", "failed to"),
    ("was refused", "refused"),
    ("is free of tax", "free of"),
    ("is unable to apply", "unable"),
    ("was disqualified", "disqualified"),
    ("is absent", "is absent"),
    ("is a non-member case", "non-"),
    ("cannot be recorded", "cannot"),
    ("is recorded by none", "none"),
    ("lacked a record", "lacked"),
    ("is missing", "missing"),
)


@settings(max_examples=200, deadline=None)
@given(
    words=st.lists(st.sampled_from(_NONCE_WORDS), min_size=1, max_size=3, unique=True),
    negation=st.sampled_from(_CLAUSE_NEGATIONS),
    name_is_negative=st.booleans(),
    data=st.data(),
)
def test_an_excerpt_cut_before_a_negation_in_its_clause_gives_no_value(
    words: list[str],
    negation: tuple[str, str],
    name_is_negative: bool,
    data: st.DataObject,
):
    """An excerpt that stops before the end of a negation in its clause gives none.

    The item is "Paid when <words> <tail>", and the excerpt stops anywhere from
    the end of the words to one character short of the end of the tail's
    negation. After an "and", the same tail scopes other words, so the
    excerpt "Paid when <words>" keeps the value its wording gives.
    """

    tail, form = negation
    phrase = " ".join(words)
    stem = f"Paid when {phrase}"
    item = f"{stem} {tail}"
    form_end = item.index(form, len(stem)) + len(form)
    cut = data.draw(st.integers(min_value=len(stem), max_value=form_end - 1))
    selector = "_".join(("not", *words) if name_is_negative else words)

    def values(excerpt: str, source_item: str) -> tuple[tuple[str, bool], ...]:
        source = _item_source(source_item)
        branches = completeness_module.recognize_source_structure(
            source, corpus_citation_path=CITATION
        )
        return completeness_module._list_item_values(
            selector,
            completeness_module._rule_excerpts_under_branch(
                _rule(PAYMENT_OUTPUT, ITEM_REFS, "true", [_atom(excerpt)]),
                next(branch for branch in branches if branch.path == ("a",)),
                source_branches=branches,
                corpus_citation_path=CITATION,
            ),
        )

    assert values(item[:cut], item) == ()
    assert values(stem, f"{stem} and the payment {tail}") == (
        (stem, not name_is_negative),
    )
