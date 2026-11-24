"""Typed CFR dependency reader for complete-source deferral reasons.

A deferral reason such as `Cannot be computed until 42 CFR 435.811 is
encoded.` names a CFR dependency. Before this reader the generic dependency
reader keyed `§ 435.811` as `435` (the part) and `us/regulation/42/435/811`
as `42` (the title), and refused the `.811 is encoded` suffix. So no wording
of a CFR dependency bound, while a part-only `§ 435` bound to any `§ 435.x`
citation in the branch.

The reader reads one closed CFR grammar on both sides (reason and the
deferred leaf's own text), keys identity as title + part + section (+ the
paragraph fragments when both sides give them), and re-checks every read
citation with the corpus resolver's grammar. Anything else stays refused.
"""

from __future__ import annotations

import pytest

from axiom_encode.corpus_resolver import normalize_corpus_identifier
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.source_completeness import recognize_source_structure

CITATION = "us/regulation/42/435/733"
SOURCE = (
    "(a) Basis. Synthetic fixture text.\n\n"
    "(b) Income standard. The agency must use one of the following:\n\n"
    "(1) The medically needy income standard established under § 435.811.\n\n"
    "(2) The income standard established in accordance with "
    "42 CFR 435.110(c).\n\n"
    "(3) The income standard established under § 435.960.\n\n"
    "(c) Limit. The amount must not exceed the standard used under § 435.121."
)


def _covered(
    reason: str,
    path: tuple[str, ...],
    *,
    source: str = SOURCE,
    citation: str = CITATION,
    blocked_by: list[str] | None = None,
) -> bool:
    branches = recognize_source_structure(source, corpus_citation_path=citation)
    anchor = sc._deferral_anchor_bases(citation)[-1]
    record: dict[str, object] = {
        "output": f"{anchor}/{'/'.join(path)}#deferred_value",
        "reason": reason,
    }
    if blocked_by is not None:
        record["blocked_by"] = blocked_by
    payload = {
        "format": "rulespec/v1",
        "module": {"deferred_outputs": [record]},
        "rules": [],
    }
    covered, _issues = sc._deferred_coverage(
        payload,
        corpus_citation_path=citation,
        source_text=source,
        branches=branches,
    )
    return path in covered


def test_fixture_branches_are_the_expected_leaves():
    branches = recognize_source_structure(SOURCE, corpus_citation_path=CITATION)
    paths = {branch.path for branch in branches}
    assert {("b", "1"), ("b", "2"), ("b", "3"), ("c",)} <= paths
    for leaf in (("b", "1"), ("b", "2"), ("b", "3"), ("c",)):
        assert sc._is_leaf_branch_path(leaf, branches)
    assert not sc._is_leaf_branch_path(("b",), branches)
    assert not sc._is_leaf_branch_path((), branches)


# --- The closed grammar -----------------------------------------------------

ACCEPTED_FORMS = (
    ("42 CFR 435.811", ("42", "435", "811"), ()),
    ("42 C.F.R. 435.811", ("42", "435", "811"), ()),
    ("42 C.F.R. § 435.811", ("42", "435", "811"), ()),
    ("42 CFR § 435.811(b)(1)", ("42", "435", "811"), ("b", "1")),
    ("§ 435.811", ("42", "435", "811"), ()),
    ("§435.110(c)", ("42", "435", "110"), ("c",)),
    ("us/regulation/42/435/811", ("42", "435", "811"), ()),
    ("us/regulation/42/435/110(c)(1)", ("42", "435", "110"), ("c", "1")),
    ("us/regulation/42/435/110/c/1", ("42", "435", "110"), ("c", "1")),
    ("26 CFR 1.414(v)", ("26", "1", "414"), ("v",)),
    ("§ 435.960", ("42", "435", "960"), ()),
)


@pytest.mark.parametrize(("text", "identity", "fragments"), ACCEPTED_FORMS)
def test_closed_grammar_reads_each_accepted_form(text, identity, fragments):
    (citation,) = sc._cfr_dependency_citations(f"until {text} is", unit_title="42")
    assert citation.section_identity == identity
    assert citation.fragments == fragments
    # Every read citation round-trips through the corpus resolver grammar.
    rendered = f"{identity[0]} CFR {identity[1]}.{identity[2]}" + "".join(
        f"({fragment})" for fragment in fragments
    )
    assert normalize_corpus_identifier(rendered) == "/".join(
        ("us", "regulation", *identity, *fragments)
    )


def test_section_sign_takes_the_units_own_title():
    (citation,) = sc._cfr_dependency_citations("under § 273.9", unit_title="7")
    assert citation.section_identity == ("7", "273", "9")


@pytest.mark.parametrize(
    "text",
    (
        "§ 435",
        "42 CFR part 435",
        "42 CFR 435",
        "us/regulation/42/435",
        "section 435.811",
        "§§ 435.811 and 435.812",
        "§§ 435.110-435.120",
        "§ 435.110 through § 435.120",
        "§ 435.110 – 435.120",
        "435.110 to § 435.120",
        "§ 435.811a",
        "42 CFR 435.811a",
        "us/regulation/42/435/811a",
        "26 CFR 1.414(v)-1",
        "42CFR435.811",
        "142 CFR 435.811 ",
        "us/regulation/42/435/811x",
        "us/regulation/42/435/811#value",
        "us:regulations/42-cfr/435/811#value",
        "the phrase “eligibility under § 435.811”",
        'the phrase "eligibility under § 435.811"',
    ),
)
def test_closed_grammar_reads_nothing_else(text):
    assert sc._cfr_dependency_citations(text, unit_title="42") == ()


def test_trailing_zero_sections_are_distinct():
    (wide,) = sc._cfr_dependency_citations("§ 435.960", unit_title="42")
    (narrow,) = sc._cfr_dependency_citations("§ 435.96", unit_title="42")
    assert wide.section_identity != narrow.section_identity
    assert not sc._cfr_citations_name_same_provision(wide, narrow)


def test_fragments_bind_only_when_equal_or_given_on_one_side():
    def read(text: str) -> sc._CfrDependencyCitation:
        return sc._cfr_dependency_citations(text, unit_title="42")[0]

    same = sc._cfr_citations_name_same_provision
    assert same(read("§ 435.110"), read("§ 435.110(c)"))
    assert same(read("§ 435.110(c)"), read("§ 435.110"))
    assert same(read("§ 435.110(c)(1)"), read("§ 435.110(c)(1)"))
    assert not same(read("§ 435.110(b)"), read("§ 435.110(c)"))
    assert not same(read("§ 435.110(c)(1)"), read("§ 435.110(c)"))
    assert not same(read("§ 435.110(c)"), read("§ 435.110(c)(2)"))


# --- Accepted deferrals -----------------------------------------------------


@pytest.mark.parametrize(
    "reason",
    (
        "Cannot be computed until 42 CFR 435.811 is encoded.",
        "Cannot be computed until 42 C.F.R. 435.811 is encoded.",
        "Cannot be computed until 42 CFR § 435.811 is encoded.",
        "Cannot be computed until § 435.811 is encoded.",
        "Cannot be computed until us/regulation/42/435/811 is encoded.",
        "Cannot be computed until the standard under § 435.811 is encoded.",
    ),
)
def test_each_accepted_reason_form_binds_to_the_leaf_citation(reason):
    assert _covered(reason, ("b", "1"))


@pytest.mark.parametrize(
    "reason",
    (
        "Cannot be computed until 42 CFR 435.110(c) is encoded.",
        "Cannot be computed until 42 CFR 435.110 is encoded.",
        "Cannot be computed until us/regulation/42/435/110(c) is encoded.",
        "Cannot be computed until us/regulation/42/435/110/c is encoded.",
        "Cannot be computed until § 435.110(c) is encoded.",
    ),
)
def test_trailing_fragment_and_suffix_are_read_by_the_grammar(reason):
    assert _covered(reason, ("b", "2"))


def test_source_citation_with_title_binds_a_section_sign_reason():
    source = (
        "(a) Basis. Synthetic fixture text.\n\n"
        "(b) Standard. The agency uses the following:\n\n"
        "(1) The standard established under 42 CFR 435.811.\n\n(2) Other."
    )
    assert _covered(
        "Cannot be computed until § 435.811 is encoded.",
        ("b", "1"),
        source=source,
    )


# --- Near misses that stay refused -----------------------------------------


@pytest.mark.parametrize(
    ("reason", "path"),
    (
        # part-only and title-only
        ("Cannot be computed until § 435 is encoded.", ("b", "1")),
        ("Cannot be computed until 42 CFR part 435 is encoded.", ("b", "1")),
        ("Cannot be computed until us/regulation/42/435 is encoded.", ("b", "1")),
        ("Cannot be computed until section 435 is encoded.", ("b", "1")),
        ("Cannot be computed until title 42 is encoded.", ("b", "1")),
        ("Cannot be computed until eligibility under § 435 is encoded.", ("b", "1")),
        # another section of the same part, another part, another title
        ("Cannot be computed until 42 CFR 435.812 is encoded.", ("b", "1")),
        ("Cannot be computed until 42 CFR 436.811 is encoded.", ("b", "1")),
        ("Cannot be computed until 45 CFR 435.811 is encoded.", ("b", "1")),
        ("Cannot be computed until us/regulation/45/435/811 is encoded.", ("b", "1")),
        # trailing-zero sections
        ("Cannot be computed until § 435.96 is encoded.", ("b", "3")),
        ("Cannot be computed until § 435.9600 is encoded.", ("b", "3")),
        # section with a letter suffix, hyphenated section
        ("Cannot be computed until § 435.811a is encoded.", ("b", "1")),
        ("Cannot be computed until 42 CFR 435.811-1 is encoded.", ("b", "1")),
        # fragment mismatch
        ("Cannot be computed until 42 CFR 435.110(b) is encoded.", ("b", "2")),
        ("Cannot be computed until 42 CFR 435.110(c)(1) is encoded.", ("b", "2")),
        # ranges and lists
        ("Cannot be computed until §§ 435.800-435.811 are encoded.", ("b", "1")),
        (
            "Cannot be computed until 42 CFR 435.811 through 435.812 is encoded.",
            ("b", "1"),
        ),
        # the dependency is cited by a sibling, not by the deferred leaf
        ("Cannot be computed until 42 CFR 435.811 is encoded.", ("b", "2")),
        # the deferred branch is not a leaf (the citation is in a child)
        ("Cannot be computed until 42 CFR 435.811 is encoded.", ("b",)),
        # self-citation of the unit is not an external dependency
        ("Cannot be computed until 42 CFR 435.733 is encoded.", ("b", "1")),
    ),
)
def test_near_misses_stay_refused(reason, path):
    assert not _covered(reason, path)


def test_dependency_only_in_parent_chapeau_does_not_bind_the_child():
    source = (
        "(a) Basis. Synthetic fixture text.\n\n"
        "(b) Standard. The agency must apply the standard under § 435.811 to:\n\n"
        "(1) Individuals age 65 or older.\n\n"
        "(2) Individuals who are blind."
    )
    reason = "Cannot be computed until 42 CFR 435.811 is encoded."
    assert not _covered(reason, ("b", "1"), source=source)
    assert not _covered(reason, ("b",), source=source)


@pytest.mark.parametrize(
    "leaf",
    (
        # citation inside a quoted example
        '(1) The notice must state "eligibility under § 435.811".',
        "(1) The notice must state “eligibility under § 435.811”.",
        # no link syntax before the citation
        "(1) Individuals who meet the requirements in § 435.811.",
        # a second sentence states another rule
        "(1) The standard under § 435.811. The agency must review it yearly.",
        # a second provision in the same leaf
        "(1) The standard under § 435.811 or the standard under § 435.230.",
        "(1) The standard under § 435.811, as described in paragraph (b) of "
        "this section.",
        "(1) The standard under § 435.811 and section 1902(a) of the Act.",
        "(1) The standard under § 435.811 and 42 U.S.C. 1396a(a)(10).",
        # a range or list in the leaf
        "(1) The standards under §§ 435.811 and 435.812.",
        "(1) The standards under § 435.800 through § 435.811.",
        # the leaf cites the section through a different fragment
        "(1) The standard under § 435.811(a).",
    ),
)
def test_leaf_must_state_only_the_linked_dependency(leaf):
    source = (
        "(a) Basis. Synthetic fixture text.\n\n"
        "(b) Standard. The agency uses the following:\n\n"
        f"{leaf}\n\n(2) Other individuals."
    )
    assert not _covered(
        "Cannot be computed until 42 CFR 435.811(b) is encoded.",
        ("b", "1"),
        source=source,
    )


def test_positive_twin_of_the_leaf_controls_binds():
    source = (
        "(a) Basis. Synthetic fixture text.\n\n"
        "(b) Standard. The agency uses the following:\n\n"
        "(1) The standard under § 435.811.\n\n(2) Other."
    )
    assert _covered(
        "Cannot be computed until 42 CFR 435.811(b) is encoded.",
        ("b", "1"),
        source=source,
    )


ADVERSARIAL_REASONS = (
    "Cannot be computed until § 435 is encoded.",
    "Cannot be computed until 42 CFR part 435 is encoded.",
    "Cannot be computed until us/regulation/42/435 is encoded.",
    "Cannot be computed until title 42 is encoded.",
    "Cannot be computed until § 435.81 is encoded.",
    "Cannot be computed until § 435.8110 is encoded.",
    "Cannot be computed until § 435.811a is encoded.",
    "Cannot be computed until §§ 435.811 and 435.812 are encoded.",
    "Cannot be computed until §§ 435.800-435.811 are encoded.",
    "Cannot be computed until 7 CFR 435.811 is encoded.",
    "Cannot be computed until us/regulation/45/435/811 is encoded.",
    "Cannot be computed until 42 CFR 435.733 is encoded.",
    "42 CFR 435.811 is encoded.",
    'Cannot be computed until "42 CFR 435.811" is encoded.',
    "Cannot be computed until 42 CFR 435.811 is encoded, but it is available.",
    "42 CFR 435.811 is available, so nothing is missing.",
    "Cannot be computed until 42 CFR 435.811-1 is encoded.",
    "Cannot be computed until 42CFR435.811 is encoded.",
    "Cannot be computed until 42 CFR 435.811 through 435.812 is encoded.",
    "Cannot be computed until section 435.811 is encoded.",
    "Cannot be computed until us/regulation/42/435/811x is encoded.",
    "Cannot be computed until 142 CFR 435.811 is encoded.",
    "Requires the medically needy income standard, which is not yet encoded.",
    "Cannot be computed until 42 CFR 435.811 is repealed.",
    "Cannot be computed until 42 CFR 435.811(a) is encoded.",
    "Cannot be computed until us/regulation/42/435/811(a) is encoded.",
)


@pytest.mark.parametrize("reason", ADVERSARIAL_REASONS)
def test_adversarial_reasons_stay_refused(reason):
    source = (
        "(a) Basis. Synthetic fixture text.\n\n"
        "(b) Standard. The agency uses the following:\n\n"
        "(1) The standard under § 435.811(b).\n\n(2) Other."
    )
    assert not _covered(reason, ("b", "1"), source=source)


def test_adversarial_list_has_a_binding_twin():
    source = (
        "(a) Basis. Synthetic fixture text.\n\n"
        "(b) Standard. The agency uses the following:\n\n"
        "(1) The standard under § 435.811(b).\n\n(2) Other."
    )
    assert _covered(
        "Cannot be computed until 42 CFR 435.811 is encoded.",
        ("b", "1"),
        source=source,
    )


def test_part_only_reason_no_longer_binds_a_dotted_source_section():
    """Before this reader `§ 435` bound to `under § 435.110` (key 435 on both)."""

    source = (
        "(a) Basis. Synthetic fixture text.\n\n"
        "(b) Eligibility. "
        "(1) The period is 4 months.\n\n"
        "(2) The value for a parent who was eligible "
        "under § 435.110."
    )
    assert not _covered(
        "Cannot be computed until eligibility under § 435 is encoded.",
        ("b", "2"),
        source=source,
        citation="us/regulation/42/435/115",
    )
    assert _covered(
        "Cannot be computed until eligibility under § 435.110 is encoded.",
        ("b", "2"),
        source=source,
        citation="us/regulation/42/435/115",
    )


def test_cfr_reader_is_off_outside_cfr_section_units():
    source = (
        "(a) Basis. Synthetic fixture text.\n\n"
        "(b) Standard. The agency uses the following:\n\n"
        "(1) The standard under § 435.811.\n\n(2) Other."
    )
    for citation in (
        "us/statute/42/1396a",
        "us/regulation/42/435",
        "us/regulation/26/1.36B-2",
        "us-az/manual/des/faa5/block-1",
    ):
        assert sc._us_regulation_unit_identity(citation) is None
        assert not _covered(
            "Cannot be computed until 42 CFR 435.811 is encoded.",
            ("b", "1"),
            source=source,
            citation=citation,
        )


def test_blocked_by_route_is_unchanged():
    assert not _covered(
        "Cannot be computed until § 435.811 is encoded.",
        ("b", "1"),
        blocked_by=["us:regulations/42-cfr/435/811#standard"],
    )


def test_self_citation_in_the_leaf_is_not_an_external_dependency():
    source = (
        "(a) Basis. Synthetic fixture text.\n\n"
        "(b) Standard. The agency uses the following:\n\n"
        "(1) The amount determined under § 435.733(c).\n\n(2) Other."
    )
    assert not _covered(
        "Cannot be computed until 42 CFR 435.733(c) is encoded.",
        ("b", "1"),
        source=source,
    )


def test_unread_cfr_citation_in_the_leaf_refuses_the_binding():
    source = (
        "(a) Basis. Synthetic fixture text.\n\n"
        "(b) Standard. The agency uses the following:\n\n"
        '(1) The standard under § 435.811 or "42 CFR 435.812" under § 435.811.'
        "\n\n(2) Other."
    )
    assert not _covered(
        "Cannot be computed until 42 CFR 435.811 is encoded.",
        ("b", "1"),
        source=source,
    )


def test_a_parent_branch_never_binds_even_when_its_text_would():
    source = (
        "(a) Basis. Synthetic fixture text.\n\n"
        "(b) The income standard under § 435.811 for:\n\n"
        "(1) Aged individuals\n\n(2) Blind individuals"
    )
    branches = recognize_source_structure(source, corpus_citation_path=CITATION)
    assert {("b", "1"), ("b", "2")} <= {branch.path for branch in branches}
    reason = "Cannot be computed until 42 CFR 435.811 is encoded."
    assert not _covered(reason, ("b",), source=source)
    assert not _covered(reason, ("b", "1"), source=source)
