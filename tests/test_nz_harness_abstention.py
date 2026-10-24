"""Generated malformed-grammar controls frozen against origin/main fd31d10a.

The base signatures were verified using the round-one base comparison procedure:
execute the three production modules from that commit, then observe the public
outline/recall/coverage results and the relevant parser result. No Git checkout
or Git executable is needed by this test.

Each example exercises every relaxed path, across NZ and UK/US/DE/FR controls.
These are finite malformed grammar families, not a claim of universal grammar
equivalence. A valid prefix can independently improve on base; citation cases
therefore invalidate the first locator, rather than compare a valid citation's
new cleanup with base's incomplete cleanup.

Round-three families compare complete count-outline and coverage signatures,
mixed-reference binding, attached canonical atoms, arithmetic ambiguity, and
the preservation of spaced amounts. Ambiguous citation cleanup can retain more
locator obligations than base; its equivalence claim concerns arithmetic only.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode import bare_structure as bs
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as vp


def _numeric_values(source: str, citation: str) -> list[float]:
    return vp.extract_numeric_occurrences_from_text(
        sc.authoritative_numeric_recall_text(source, corpus_citation_path=citation)
    )


def _observations(
    amount: int, locator: int, word: str, line_ending: str, jurisdiction: str
) -> dict[str, object]:
    """Observe full results, including the obligations rejected outlines retain."""
    citation = f"{jurisdiction}/statute/act/public/2018/0032/section/220"
    malformed_outlines = {
        # Complete prose alone cannot authenticate an out-of-order label.
        "numeric_prose": (
            f"1\nThe authority must pay ${amount}.\n3\nThe household must apply."
        ),
        # A legal subject without a complete predicate is outside the grammar.
        "legal_subject": (f"1\nSection {locator} {word}\n2\nThe household must apply."),
        # Only the positively named prescribed complements resolve a quantity.
        "quantity_complement": (
            "1\nIf the payment is payable at a rate "
            f"{word} in respect of a person, the authority must pay ${amount}.\n"
            "2\nThe household must apply."
        ),
        "count_table": (
            "Number of children | Eligibility\n"
            f"1 | The household is eligible if— (a) | income is below ${amount}.\n"
            f"2 | The household is eligible if— (a) | income is below ${amount + 1}."
        ),
        "embedded_table": (
            "Clause | Provision\n"
            "1 | The rates are specified in the following table:\n"
            "Number of children | Eligibility\n"
            f"2 | The household is eligible if— (a) | income is below ${amount}.\n"
            f"3 | The household is eligible if— (a) | income is below ${amount + 1}."
        ),
        "bare_pipe_child": (
            "Clause | Provision\n"
            "1 | The conditions apply as follows: "
            "(a) | A person is eligible if resident. "
            f"b | A person is eligible if aged {amount} years.\n"
            "2 | The authority must pay the benefit."
        ),
        "bordered_definition": (
            "Clause | Provision\n1 | The following definitions apply:—\n"
            "| benefit means—\n(a) | a main benefit; or\n(b) | a pension.\n"
            f"| minimum period means a period of at least {amount} weeks."
        ),
        # Unidentified extra cells could hide a label or a definition boundary.
        "value_cell": (
            "Clause | Provision\n"
            f"1 | To a person | ${amount} a week\n"
            f"2 | To any other person | ${amount + 1} a week"
        ),
    }
    observations: dict[str, object] = {}
    for kind, source in malformed_outlines.items():
        source = source.replace("\n", line_ending)
        observations[f"{kind}/outline"] = bs.bare_outline(source)
        observations[f"{kind}/numbers"] = _numeric_values(source, citation)
        observations[f"{kind}/coverage"] = vp._top_level_source_subparagraph_segments(
            source
        )

    # No first numeric locator means no citation list has been authenticated.
    observations["citation_list"] = _numeric_values(
        f"The payment under sections {word} and {amount} is payable.", citation
    )

    # The canonical path fallback requires exact boundaries and a complete suffix.
    canonical = f"{jurisdiction}/guidance/example/section/{locator}"
    observations["canonical_citation"] = [
        vp._source_citation_subparagraph_paths(source, canonical)
        for source in (
            f"prefix{canonical}(a)",
            f"{canonical}0(a)",
            f"{canonical}(a){word}",
            f"{canonical}/other(a)",
        )
    ]

    # Neither a different locator kind nor a different instrument is supported.
    observations["dependency_kind"] = [
        sc._reason_dependency_is_source_bound(
            f"Cannot be computed until {jurisdiction}/statute/act/public/2018/0032/"
            f"{kind}/{locator} is encoded.",
            f"The payment depends on section {locator}.",
            corpus_citation_path=citation.replace("/section/220", f"/{kind}/220"),
        )
        for kind in ("article", "regulation")
    ]
    observations["dependency_instrument"] = sc._reason_dependency_is_source_bound(
        f"Cannot be computed until {jurisdiction}/statute/act/public/2018/0099/"
        f"section/{locator} is encoded.",
        f"The payment depends on section {locator}.",
        corpus_citation_path=citation,
    )

    # A number/unit token is not German sentence typography in any jurisdiction.
    source = f"The allowance is {amount} kilograms."
    observations["german_typography/structure"] = tuple(
        branch.path
        for branch in sc.recognize_source_structure(
            source, corpus_citation_path=citation
        )
        if branch.kind == "sentence"
    )
    observations["german_typography/numbers"] = _numeric_values(source, citation)

    # 'Would remain' does not authenticate the counterfactual 'would apply' form.
    observations["counterfactual"] = sc._source_conjunctive_fact_gates(
        f"The rate of ${amount} would remain if they were single "
        "(and their other circumstances were unchanged)."
    )

    # Near-miss words supply neither a year unit nor a language-specific duration.
    source = f"La durée est de {amount} ananas."
    observations["duration/months"] = vp._extract_year_duration_month_values(source)
    observations["duration/numbers"] = vp.extract_numeric_occurrences_from_text(source)

    # The cleaned INA glossary form never exempts an operative obligation.
    observations["guidance_definition"] = []
    for predicate in ("report", "notify", "disclose", "retain"):
        source = (
            f"Admitted under of the INA for a period of at least {amount} years "
            f"and must {predicate} income."
        )
        observations["guidance_definition"].append(
            sc._source_boundary_is_nonoperative_guidance_definition(
                source, context="Parolees Paroled into the U.S. " + source
            )
        )
    return observations


def _base_signatures(amount: int, line_ending: str) -> dict[str, object]:
    """Frozen base observations; generated values preserve their physical order."""
    numbers = {
        "numeric_prose": [1, amount, 3],
        "legal_subject": [1, 2],
        "quantity_complement": [1, amount, 2],
        "count_table": [1, amount, 2, amount + 1],
        "embedded_table": [1, 2, amount, 3, amount + 1],
        "bare_pipe_child": [1, amount, 2],
        "bordered_definition": [7 * amount, 1, amount],
        "value_cell": [1, amount, 2, amount + 1],
    }
    signatures: dict[str, object] = {}
    for kind, values in numbers.items():
        signatures[f"{kind}/outline"] = None
        signatures[f"{kind}/numbers"] = values
        signatures[f"{kind}/coverage"] = []
    if line_ending != "\u2028":
        signatures["bordered_definition/coverage"] = [
            ("a", "| a main benefit; or"),
            (
                "b",
                "| a pension."
                + line_ending
                + f"| minimum period means a period of at least {amount} weeks.",
            ),
        ]
    signatures.update(
        {
            "citation_list": [amount],
            "canonical_citation": [set(), set(), set(), set()],
            "dependency_kind": [False, False],
            "dependency_instrument": False,
            "german_typography/structure": (),
            "german_typography/numbers": [amount],
            "counterfactual": (
                (frozenset(), frozenset({"single"})),
                (
                    frozenset(),
                    frozenset({"circumstance", "other", "unchanged"}),
                ),
            ),
            "duration/months": set(),
            "duration/numbers": [amount],
            "guidance_definition": [False, False, False, False],
        }
    )
    return signatures


def _round_three_observations(
    amount: int, locator: int, line_ending: str, jurisdiction: str
) -> dict[str, object]:
    citation = f"{jurisdiction}/statute/act/public/2018/0032/section/220"
    observations: dict[str, object] = {}
    for heading in ("Family size.", "Age.", "Household members."):
        source = line_ending.join(
            (
                heading,
                "1",
                f"The household is entitled to ${amount}.",
                "2",
                f"The household is entitled to ${amount + 1}.",
            )
        )
        observations[f"detached/{heading}"] = (
            bs.bare_outline(source),
            _numeric_values(source, citation),
            vp._top_level_source_subparagraph_segments(source),
        )

    reason = (
        f"Cannot be computed until {citation.rsplit('/', 1)[0]}/{locator} is encoded."
    )
    for other in (f"paragraph {locator}", f"section {locator + 1}"):
        source = (
            f"(1) Section {locator} has a heading and the conditions under "
            f"{other} apply."
        )
        observations[f"mixed_reference/{other}"] = (
            sc._reason_dependency_is_source_bound(
                reason, source, corpus_citation_path=citation
            )
        )

    canonical = f"{jurisdiction}/guidance/example/section/{locator}"
    observations["attached_suffixes"] = [
        vp._source_citation_subparagraph_paths(f"{canonical}(a){tail}", canonical)
        for tail in (".5", "-other", ":other", ".5(a)", "-other(b)")
    ]

    for separator in (", ", " and "):
        for hyphen in ("-", " - "):
            source = (
                f"The payment is 20% of earnings under sections {locator}"
                f"{separator}{locator + 1}{hyphen}18{hyphen}2."
            )
            cleaned = sc.authoritative_numeric_recall_text(
                source, corpus_citation_path=citation
            )
            # Full percentage witnesses improve on base for spaced chains;
            # parity here concerns the independent subtraction-topology veto.
            observations[f"partial_list/{separator}/{hyphen}"] = (
                sc._explicit_source_arithmetic_topology(cleaned)
            )

    for separator in (" ", "\u00a0", "\u202f"):
        amount_text = f"{amount}{separator}{amount + 1} dollars"
        source = (
            "The payment under sections 297, 311(1), and 338, and "
            f"{amount_text} is payable."
        )
        cleaned = sc.authoritative_numeric_recall_text(
            source, corpus_citation_path=citation
        )
        observations[f"spaced_amount/{separator}"] = amount_text in cleaned
    return observations


def _round_three_base_signatures(amount: int, locator: int) -> dict[str, object]:
    return {
        **{
            f"detached/{heading}": (None, [1, amount, 2, amount + 1], [])
            for heading in ("Family size.", "Age.", "Household members.")
        },
        f"mixed_reference/paragraph {locator}": False,
        f"mixed_reference/section {locator + 1}": False,
        "attached_suffixes": [set(), set(), set(), set(), set()],
        **{
            f"partial_list/{separator}/{hyphen}": None
            for separator in (", ", " and ")
            for hyphen in ("-", " - ")
        },
        **{
            f"spaced_amount/{separator}": True
            for separator in (" ", "\u00a0", "\u202f")
        },
    }


@settings(max_examples=40, deadline=None)
@given(
    amount=st.integers(min_value=1000, max_value=9999),
    locator=st.integers(min_value=40, max_value=199),
    word=st.sampled_from(["pending", "undetermined", "unresolved", "unverified"]),
    line_ending=st.sampled_from(["\n", "\r\n", "\u2028"]),
)
def test_outside_relaxed_grammars_retains_base_signatures(
    amount: int, locator: int, word: str, line_ending: str
):
    expected = _base_signatures(amount, line_ending)
    for jurisdiction in ("nz", "uk", "us", "de", "fr"):
        actual = _observations(amount, locator, word, line_ending, jurisdiction)
        assert actual.keys() == expected.keys()
        for path, signature in expected.items():
            assert actual[path] == signature, (jurisdiction, path)
        round_three = _round_three_observations(
            amount, locator, line_ending, jurisdiction
        )
        expected_round_three = _round_three_base_signatures(amount, locator)
        assert round_three.keys() == expected_round_three.keys()
        for path, signature in expected_round_three.items():
            assert round_three[path] == signature, (jurisdiction, path)
