"""A defined income aggregate named "family ... adjusted gross income" is not a
Family eligibility unit (#1842).

Va. Code § 58.1-339.8 B 1 allows a credit to an individual or married couple
"whose family Virginia adjusted gross income does not exceed 100 percent of
the poverty guideline amount". The defined term aggregates the income on a
return, so the word `family` in it says nothing about the legal subject. The
source-scope gate read it as a Family unit test and rejected every TaxUnit
encoding.

The `family` modifier of an adjusted-gross-income aggregate is masked, and a
scope issue is reported only when both main's reading and the masked reading
produce it. So masking can only turn an issue into "unresolved"; it can never
create one, for example by removing one side of a conflict between excerpts.
The negative controls come first: genuine family-unit language keeps its
issue.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import validator_pipeline as vp

REPOSITORY = Path(__file__).parents[1]
# Contiguous text from Va. Code § 58.1-339.8 B 1, as quoted in #1842.
EXCERPT = (
    "whose family Virginia adjusted gross income does not exceed 100 percent of "
    "the poverty guideline amount"
)
EXTENDED = EXCERPT + " corresponding to a household of an equal number of persons"
WITH_SUBJECT = "any individual or married individuals filing jointly, " + EXCERPT
MISMATCH = "Source scope mismatch"


def _document(entity, *excerpts, summary=None):
    module = {"summary": summary} if summary is not None else {}
    return yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": module,
            "rules": [
                {
                    "name": "low_income_credit_allowed",
                    "kind": "derived",
                    "entity": entity,
                    "dtype": "Judgment",
                    "versions": [
                        {"effective_from": "2026-01-01", "formula": "income_ok"}
                    ],
                    "metadata": {
                        "proof": {
                            "atoms": [
                                {"source": {"excerpt": excerpt}} for excerpt in excerpts
                            ]
                        }
                    },
                }
            ],
        }
    )


def _issues(entity, *excerpts):
    return vp.find_source_scope_consistency_issues(_document(entity, *excerpts))


def _main_issues(entity, *excerpts):
    """Main's semantics: no defined income term is masked."""

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(
            vp,
            "_mask_family_adjusted_gross_income_modifier",
            lambda text: text,
            raising=False,
        )
        return _issues(entity, *excerpts)


# Negative controls: genuine family-unit language keeps main's issue.
GENUINE_FAMILY = (
    "whose family income does not exceed 100 percent of the poverty guideline amount",
    "the family's adjusted gross income does not exceed 100 percent of the poverty "
    "guideline amount",
    "A family is eligible if its family Virginia adjusted gross income does not "
    "exceed 100 percent of the poverty guideline amount",
    "Family income determines eligibility for the credit.",
    "whose family Virginia adjusted gross income does not exceed the standard for "
    "the family unit; family resources determine eligibility",
)


@pytest.mark.parametrize("excerpt", GENUINE_FAMILY)
@pytest.mark.parametrize("entity", ("TaxUnit", "Person"))
def test_genuine_family_unit_language_keeps_its_issue(excerpt, entity):
    issues = _issues(entity, excerpt)
    assert len(issues) == 1 and issues[0].startswith(MISMATCH)
    assert issues == _main_issues(entity, excerpt)


@pytest.mark.parametrize(
    ("entity", "excerpts"),
    (
        # Household income is unit evidence of its own.
        ("Person", ("Household income determines eligibility. " + EXCERPT,)),
        # A Family rule proved by a person-scoped excerpt.
        ("Family", ("any individual is eligible for the credit",)),
    ),
)
def test_other_scope_evidence_keeps_mains_issue(entity, excerpts):
    issues = _issues(entity, *excerpts)
    assert issues and issues == _main_issues(entity, *excerpts)


@pytest.mark.parametrize(
    ("entity", "excerpts"),
    (
        # Main reads these as conflicts and stays silent. Masking the defined
        # term would leave one side standing; the gate must not report it.
        ("TaxUnit", (EXCERPT, "any individual is eligible for the credit")),
        ("TaxUnit", (EXCERPT, "Household income determines eligibility.")),
        ("TaxUnit", ("Household income determines eligibility. " + EXCERPT,)),
        ("TaxUnit", (EXCERPT + "; any individual is eligible for the credit",)),
        ("Person", (EXCERPT, "any individual is eligible for the credit")),
        ("Person", ("Household income determines eligibility.", EXCERPT)),
    ),
)
def test_masking_never_creates_an_issue(entity, excerpts):
    assert _main_issues(entity, *excerpts) == []
    assert _issues(entity, *excerpts) == []


@pytest.mark.parametrize(
    "excerpts",
    (
        (EXCERPT,),
        (WITH_SUBJECT,),
        (EXCERPT.upper(),),
        (EXCERPT.replace(" ", "\n"),),
        # A definition of the same shape as subsection A's (constructed text).
        (
            '"Family Virginia adjusted gross income" means the combined Virginia '
            "adjusted gross income of an individual, the individual's spouse, and "
            "any person claimed as a dependent",
        ),
        ("whose family federal adjusted gross income does not exceed the guideline",),
        ("whose family New York adjusted gross income does not exceed the guideline",),
    ),
)
def test_defined_income_aggregate_does_not_name_a_family_unit(excerpts):
    assert any(
        issue.startswith(MISMATCH) for issue in _main_issues("TaxUnit", *excerpts)
    )
    assert _issues("TaxUnit", *excerpts) == []


def test_issue_1842_reproduction():
    # The classifiers keep main's reading by default; only the scope gate asks
    # for the masked one, and reports what both readings support.
    assert vp._classify_source_scope(EXCERPT) == "unit"
    assert vp._classify_source_unit_entity(EXCERPT) == "family"
    assert vp._classify_source_scope(EXCERPT, mask_defined_income_terms=True) is None
    assert (
        vp._classify_source_unit_entity(EXCERPT, mask_defined_income_terms=True) is None
    )
    assert _issues("TaxUnit", EXCERPT) == []
    assert _issues("TaxUnit", EXTENDED) == []


def test_other_callers_keep_mains_reading():
    """`find_person_scoped_definition_unit_issues` skips its check when the
    scope reads as a matching unit; a masked reading there could add issues."""

    rule = yaml.safe_load(_document("TaxUnit", EXCERPT))["rules"][0]
    assert vp._rule_source_scope(rule, "") == ("unit", "family")
    assert vp._rule_source_scope(rule, "", mask_defined_income_terms=True) is None


def test_mask_blanks_only_the_modifier():
    text = "Family income is tested. " + EXCERPT
    masked = vp._mask_family_adjusted_gross_income_modifier(text)
    start = text.index("family Virginia")
    assert masked == text[:start] + " " * len("family") + text[start + len("family") :]
    assert len(masked) == len(text)


FILLER = st.sampled_from(
    (
        "the credit",
        "for the taxable year",
        "any individual or married individuals filing jointly",
        "on the return",
        "corresponding to a household of an equal number of persons",
        "100 percent of the poverty guideline amount",
    )
)
GENUINE = st.sampled_from(
    (
        "",
        "family income determines eligibility",
        "a family is eligible",
        "the family unit's resources are tested",
        "household income determines eligibility",
        "any individual is eligible",
    )
)


@given(
    before=st.lists(FILLER, max_size=3),
    after=st.lists(FILLER, max_size=3),
    genuine=GENUINE,
    genuine_first=st.booleans(),
    split=st.booleans(),
    entity=st.sampled_from(("TaxUnit", "Person", "Family", "Household")),
)
@example(
    before=[],
    after=[],
    genuine="",
    genuine_first=False,
    split=False,
    entity="TaxUnit",
)
@settings(max_examples=200, deadline=None)
def test_masking_only_removes_issues(
    before, after, genuine, genuine_first, split, entity
):
    """Never stricter than main; genuine evidence decides when it is present."""

    term = " ".join((*before, EXCERPT, *after))
    parts = [genuine, term] if genuine_first else [term, genuine]
    parts = [part for part in parts if part]
    excerpts = tuple(parts) if split else ("; ".join(parts),)
    current = _issues(entity, *excerpts)
    main = _main_issues(entity, *excerpts)
    assert set(current) <= set(main)
    if current:
        assert current == main


def _repository_rulespec_documents():
    for root in (REPOSITORY / "tests/fixtures", REPOSITORY / "benchmarks"):
        for path in sorted(root.rglob("*.yaml")):
            text = path.read_text()
            if "rulespec/v1" in text:
                yield path.relative_to(REPOSITORY), text


def test_fixture_differential_changes_no_verdict():
    """Over every RuleSpec document in the repository, no scope verdict moves.

    The mask can change a document only where its pattern matches an excerpt.
    Raw YAML keeps the words and the whitespace between them, so a document
    the pattern does not match can differ only through an escape sequence;
    those documents are compared too. No repository document contains the
    defined term, so this shows that nothing else moves.
    """

    documents = list(_repository_rulespec_documents())
    assert len(documents) > 300
    changed = []
    compared = 0
    for path, content in documents:
        if not (
            vp._FAMILY_ADJUSTED_GROSS_INCOME_MODIFIER_PATTERN.search(content)
            or "\\" in content
        ):
            continue
        compared += 1
        current = vp.find_source_scope_consistency_issues(content)
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(
                vp,
                "_mask_family_adjusted_gross_income_modifier",
                lambda text: text,
                raising=False,
            )
            main = vp.find_source_scope_consistency_issues(content)
        if current != main:
            changed.append((path, main, current))
    assert compared >= 10
    assert changed == []


@pytest.mark.parametrize("rules", (1, 8, 64))
def test_scope_gate_reads_each_rule_at_most_twice(monkeypatch, rules):
    calls = []
    rule_source_scope = vp._rule_source_scope

    def counting(rule, fallback, **kwargs):
        calls.append((rule["name"], kwargs.get("mask_defined_income_terms", False)))
        return rule_source_scope(rule, fallback, **kwargs)

    monkeypatch.setattr(vp, "_rule_source_scope", counting)
    payload = yaml.safe_load(_document("TaxUnit", EXCERPT))
    template = payload["rules"][0]
    payload["rules"] = [
        {**template, "name": f"credit_{index}"} for index in range(rules)
    ]
    assert vp.find_source_scope_consistency_issues(yaml.safe_dump(payload)) == []
    # The helper-name pass reads each rule once with main's reading; the gate
    # reads it again with main's reading and, only when main reports an
    # issue, once with the masked reading.
    assert sum(1 for _name, masked in calls if masked) == rules
    assert len(calls) <= 3 * rules
