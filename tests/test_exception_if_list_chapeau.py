"""A list chapeau ending "if they are:" takes its condition from its items.

12 CFR 1006.14(b)(3) reads "Telephone calls placed to a person do not count
toward the telephone call frequencies ... if they are:" and lists the excluded
calls below. The exception-pair check asks for a case pair whose toggled
selector the condition names, and the condition here is only "if they are:".
These tests pin the rule that a selector named in full by a cited excerpt of
one of the chapeau's own items satisfies such a chapeau, and that an excerpt
from anywhere else does not. Only a bare subject may stand between "if" or
"when" and the verb, so a clause with a condition of its own still has to name
the selector itself.
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

CITATION = "us/regulation/12/1006/99"
MODULE = "us:regulations/12-cfr/1006/99"
CHAPEAU = (
    "Telephone calls placed to a person do not count toward the telephone "
    "call frequencies described in paragraph (b) of this section if they are:"
)
CHAPEAU_EXCERPT = (
    "do not count toward the telephone call frequencies "
    "described in paragraph (b) of this section if they are:"
)
CONSENT = "Placed with such person's prior consent given directly to the debt collector"
NOT_CONNECTED = "Not connected to the dialed number"
COERCION = (
    "(b) Coercive advertisements. A debt collector must not advertise for sale "
    "any debt to coerce payment of the debt.\n"
)
SOURCE = (
    "(a) Certain telephone calls excluded from the telephone call frequencies. "
    f"{CHAPEAU}\n\n(1) {CONSENT}; or\n\n(2) {NOT_CONNECTED}.\n\n{COERCION}"
)
INTENT = (
    "A debt collector must not place telephone calls repeatedly with intent to "
    "annoy any person."
)
# The chapeau sits at (a)(2), below a sibling (a)(1) that a parent rule cites.
NESTED_SOURCE = (
    f"(a) Telephone calls—(1) In general. {INTENT}\n\n"
    f"(2) Excluded calls. {CHAPEAU}\n\n"
    f"(i) {CONSENT}; or\n\n(ii) {NOT_CONNECTED}.\n\n{COERCION}"
)
CONSENT_INPUT = "placed_with_prior_consent_given_directly_to_debt_collector"
NOT_CONNECTED_INPUT = "not_connected_to_dialed_number"
OUTPUT = "telephone_call_excluded_from_call_frequencies"
INTENT_OUTPUT = "repeated_calls_with_intent_to_annoy"
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
            "summary": "Telephone calls excluded from the call frequencies.",
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
    consent_input: str = CONSENT_INPUT,
    not_connected_input: str = NOT_CONNECTED_INPUT,
    item_atoms: tuple[str, ...] = (CONSENT, NOT_CONNECTED),
) -> tuple[dict, list[dict]]:
    rule = _rule(
        OUTPUT,
        "12 CFR 1006.99(a); 12 CFR 1006.99(a)(1); 12 CFR 1006.99(a)(2)",
        f"{consent_input}\nor {not_connected_input}",
        [_atom(CHAPEAU_EXCERPT), *(_atom(excerpt) for excerpt in item_atoms)],
    )
    doc = _doc((consent_input, not_connected_input), [rule])

    def case(name: str, consent: bool, not_connected: bool, excluded: str) -> dict:
        return _case(
            name,
            {consent_input: consent, not_connected_input: not_connected},
            {OUTPUT: excluded},
        )

    cases = [
        case("ordinary call counts", False, False, "not_holds"),
        case("call with prior consent does not count", True, False, "holds"),
        case("unconnected call does not count", False, True, "holds"),
    ]
    return doc, cases


def _nested_material(*, pair_an_item: bool) -> tuple[dict, list[dict]]:
    intent_rule = _rule(
        INTENT_OUTPUT,
        "12 CFR 1006.99(a); 12 CFR 1006.99(a)(1)",
        "repeated_telephone_calls\nand intent_to_annoy_any_person",
        [
            _atom(
                "must not place telephone calls repeatedly with intent to "
                "annoy any person",
                kind="predicate",
            )
        ],
    )
    exclusion_rule = _rule(
        OUTPUT,
        "12 CFR 1006.99(a)(2); 12 CFR 1006.99(a)(2)(i); 12 CFR 1006.99(a)(2)(ii)",
        f"{CONSENT_INPUT}\nor {NOT_CONNECTED_INPUT}",
        [_atom(CHAPEAU_EXCERPT), _atom(CONSENT), _atom(NOT_CONNECTED)],
    )
    doc = _doc(
        (
            "repeated_telephone_calls",
            "intent_to_annoy_any_person",
            CONSENT_INPUT,
            NOT_CONNECTED_INPUT,
        ),
        [intent_rule, exclusion_rule],
    )

    def case(name: str, intent: bool, consent: bool, harassing: str, excluded: str):
        return _case(
            name,
            {
                "repeated_telephone_calls": True,
                "intent_to_annoy_any_person": intent,
                CONSENT_INPUT: consent,
                NOT_CONNECTED_INPUT: False,
            },
            {INTENT_OUTPUT: harassing, OUTPUT: excluded},
        )

    cases = [
        case("repeated calls meant to annoy", True, False, "holds", "not_holds"),
        case(
            "repeated calls not meant to annoy", False, False, "not_holds", "not_holds"
        ),
    ]
    if pair_an_item:
        cases.append(
            case("call with prior consent does not count", True, True, "holds", "holds")
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
        "12 CFR 1006.99(a); 12 CFR 1006.99(a)(1); 12 CFR 1006.99(a)(2)",
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
        NOT_CONNECTED_INPUT,
        item_texts=(NOT_CONNECTED,),
    )
    assert completeness_module._source_exception_selector_is_relevant(
        condition,
        CONSENT_INPUT,
        item_texts=(CONSENT, NOT_CONNECTED),
    )


def test_list_chapeau_rejects_selector_no_cited_item_names():
    condition = completeness_module._source_exception_condition_text(CHAPEAU)

    assert not completeness_module._source_exception_selector_is_relevant(
        condition, NOT_CONNECTED_INPUT
    )
    assert not completeness_module._source_exception_selector_is_relevant(
        condition,
        "person_owns_luxury_car",
        item_texts=(CONSENT, NOT_CONNECTED),
    )
    # "call" appears only in the chapeau and "connected" only in the item, so
    # no single cited item names this selector.
    assert not completeness_module._source_exception_selector_is_relevant(
        condition,
        "call_not_connected_to_dialed_number",
        item_texts=(NOT_CONNECTED,),
        supporting_texts=(CHAPEAU, NOT_CONNECTED),
    )


def test_list_chapeau_ignores_excerpts_from_outside_its_items():
    condition = completeness_module._source_exception_condition_text(CHAPEAU)

    # The rule cites the item, but not as one of this chapeau's items.
    assert not completeness_module._source_exception_selector_is_relevant(
        condition,
        NOT_CONNECTED_INPUT,
        supporting_texts=(CHAPEAU, CONSENT, NOT_CONNECTED),
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
        # A determiner and four words, the most the subject may hold.
        "if the household's countable liquid resources are:",
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
        # A determiner and five words.
        "if the household's total countable liquid resources are:",
        # Shapes this rule leaves as they were.
        "if they are not:",
        "if any of the following apply:",
        "if—",
    ),
)
def test_other_if_clauses_must_name_the_selector_themselves(condition: str):
    assert not _relevant(condition, item_texts=(SSI_ITEM,))


def test_chapeau_that_names_the_selector_still_matches():
    # Three distinctive words, so the two-word concept rule earlier in the
    # function does not decide it and the list chapeau rule is reached.
    condition = "if the household's countable liquid resources are:"
    selector = "countable_liquid_resources"
    assert len(completeness_module._source_selector_distinctive_tokens(selector)) == 3

    assert completeness_module._source_exception_selector_is_relevant(
        condition, selector
    )
    assert completeness_module._source_exception_selector_is_relevant(
        condition, selector, item_texts=("an unrelated phrase",)
    )


def test_rule_excerpts_under_branch_keeps_the_branchs_own_items():
    rule = _rule(
        OUTPUT,
        "12 CFR 1006.99(a)",
        "true",
        [
            _atom("must not place telephone calls repeatedly", kind="predicate"),
            _atom("do not count toward the telephone call frequencies"),
            _atom(CONSENT),
            _atom(NOT_CONNECTED),
            _atom(NOT_CONNECTED, citation="us/regulation/12/1006/98"),
            _atom("text that appears nowhere in the source"),
        ],
    )

    def under(path: tuple[str, ...]) -> tuple[str, ...]:
        return completeness_module._rule_excerpts_under_branch(
            rule,
            _branch(path),
            source_branches=completeness_module.recognize_source_structure(
                NESTED_SOURCE, corpus_citation_path=CITATION
            ),
            corpus_citation_path=CITATION,
        )

    assert under(("a", "2")) == (CONSENT, NOT_CONNECTED)
    assert under(("a",)) == (
        "must not place telephone calls repeatedly",
        "do not count toward the telephone call frequencies",
        CONSENT,
        NOT_CONNECTED,
    )
    assert under(("a", "2", "ii")) == ()
    assert under(("b",)) == ()


def test_cited_item_selectors_pair_the_list_chapeau():
    doc, cases = _material()

    assert not _chapeau_is_unpaired(_issues(doc, cases))


def test_cited_item_selectors_pair_a_nested_list_chapeau():
    doc, cases = _nested_material(pair_an_item=True)

    assert not _chapeau_is_unpaired(_issues(doc, cases, NESTED_SOURCE))


def test_parent_rule_pair_does_not_pair_a_nested_list_chapeau():
    # The parent rule cites (a), so it governs the (a)(2) chapeau, and its
    # (a)(1) excerpt names `intent_to_annoy_any_person` in full. That excerpt
    # says nothing about the excluded calls, so its pair proves no exclusion.
    doc, cases = _nested_material(pair_an_item=False)

    assert _chapeau_is_unpaired(_issues(doc, cases, NESTED_SOURCE))


@pytest.mark.parametrize(
    ("excerpt", "citation", "selector"),
    (
        # Paragraph (b) of the same section, outside the chapeau's items.
        (
            "must not advertise for sale any debt to coerce payment of the debt",
            CITATION,
            "advertise_debt_for_sale_to_coerce_payment",
        ),
        # Another provision's text.
        (
            "if the consumer is deceased",
            "us/regulation/12/1006/2",
            "consumer_is_deceased",
        ),
    ),
)
def test_excerpts_outside_the_items_do_not_pair_the_list_chapeau(
    excerpt: str, citation: str, selector: str
):
    rule = _rule(
        OUTPUT,
        "12 CFR 1006.99(a); 12 CFR 1006.99(a)(1); 12 CFR 1006.99(a)(2)",
        selector,
        [
            _atom(CHAPEAU_EXCERPT),
            _atom(CONSENT),
            _atom(NOT_CONNECTED),
            _atom(excerpt, citation=citation),
        ],
    )
    cases = [
        _case("ordinary call counts", {selector: False}, {OUTPUT: "not_holds"}),
        _case("toggled call does not count", {selector: True}, {OUTPUT: "holds"}),
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
        consent_input="call_placed_with_prior_consent_of_person",
        not_connected_input="call_not_connected_to_dialed_number",
    )

    assert _chapeau_is_unpaired(_issues(doc, cases))


def test_list_chapeau_stays_unpaired_when_its_items_are_not_cited():
    doc, cases = _material(item_atoms=())

    assert _chapeau_is_unpaired(_issues(doc, cases))


def test_list_chapeau_stays_unpaired_without_a_case_pair():
    doc, cases = _material()

    assert _chapeau_is_unpaired(_issues(doc, cases[:1]))


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
        st.lists(st.sampled_from(_NONCE_WORDS), min_size=1, max_size=4),
    ),
)
# Each template adds a condition of the clause's own, or a subject too long to
# be bare, around a bare subject.
_OWN_CONDITION_TEMPLATES = st.sampled_from(
    (
        "{cue} {subject} so elects, the deductions {verb}{ending}",
        "{cue} {subject} who files a request {verb}{ending}",
        "{cue} {subject} and the payments {verb}{ending}",
        "{cue}, for the year, {subject} {verb}{ending}",
        "{cue} the household so elects, when {subject} {verb}{ending}",
        "{cue} {subject} {verb} not{ending}",
        "{cue} the blorvik quandor zephtil mornak trivolt {verb}{ending}",
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
