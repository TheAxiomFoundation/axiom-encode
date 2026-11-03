"""Belgian French, Dutch and German percent wording in numeric grounding.

A rate literal grounds only when its source marks the figure as a
percentage. Belgian legal French and Dutch write that marker out ("28 pour
cent", "9 procent") or restate the figure in parentheses before it ("(14,75)
p.c."); German statutes abbreviate "vom Hundert" ("15 v. H."). Belgian
sources (the be-BE profile, chosen from a be/ or be-<region>/ citation) read
all of these; German sources read "v. H."; every other source reads as on
main. A figure with no marker must never ground N/100: the July 2026 encoder
accepted "tarief 9 euro" as 0.09, and the gate exists to refuse that.
"""

import re
import textwrap

import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness.validator_pipeline import (
    _BELGIAN_FIGURE_RATE_MARKER_AFTER_PATTERN,
    _ascii_digit_text,
    _belgian_figure_value,
    _figure_rate_marker_follows_in_source,
    _find_source_text_value_issues,
    _has_malformed_profiled_numeric_envelope,
    _LegacyNumericCollector,
    _NumericTextView,
    _range_endpoints_meet_in_source,
    _reads_belgian_figure,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    find_ungrounded_numeric_issues,
    find_ungrounded_numeric_issues_scoped,
    numeric_value_is_grounded,
)

BE_CITATION = "be/statute/loi/1987/02/27/1987022077/article/12"


def _rate_rulespec(
    formula: str, *, citation_path: str = BE_CITATION, dtype: str = "Rate"
) -> str:
    return textwrap.dedent(
        f"""
        format: rulespec/v1
        module:
          source_verification:
            corpus_citation_path: {citation_path}
        rules:
          - name: rate
            kind: parameter
            dtype: {dtype}
            versions:
              - effective_from: '2026-01-01'
                formula: '{formula}'
        """
    ).strip()


def _gate_grounds(
    source_text: str,
    formula: str,
    *,
    citation_path: str = BE_CITATION,
    dtype: str = "Rate",
) -> bool:
    return not find_ungrounded_numeric_issues(
        _rate_rulespec(formula, citation_path=citation_path, dtype=dtype),
        source_text=source_text,
        source_citation_path=citation_path,
    )


def _grounds(source_text: str, value: float, *, profile: str = "be-BE") -> bool:
    return numeric_value_is_grounded(
        value,
        extract_typed_numeric_occurrences_from_text(source_text, profile=profile),
    )


# The probe table from the engine-pin migration (rulespec-be#132): each row
# is a proof excerpt shape a be module cites.
@pytest.mark.parametrize(
    ("source_text", "formula", "grounded"),
    (
        ("suspendu (pour 28 pour cent) pour", "0.28", True),
        ("suspendu de 28 pour cent pour", "0.28", True),
        ("suspendu (pour 28 pour cent) pour l'allocation", "0.28", True),
        ("suspendu de 28 pourcent", "0.28", True),
        ("suspendu de 28 % pour", "0.28", True),
        ("suspendu de 28 percent", "0.28", True),
        ("(14,75) p.c.du montant", "0.1475", True),
        ("cotisation de 14,75 p.c. du montant", "0.1475", True),
        ("tarief 9 procent", "0.09", True),
        ("tarief 9 %", "0.09", True),
        # The July encoder grounded every bare N as N/100.
        ("tarief 9 euro", "0.09", False),
        ("tarief 9", "0.09", False),
        ("suspendu de 28 jours", "0.28", False),
    ),
)
def test_probe_table_rows(source_text, formula, grounded):
    assert _gate_grounds(source_text, formula) is grounded


@pytest.mark.parametrize(
    ("source_text", "value"),
    (
        ("réduit de 2,5 pourcents.", 0.025),
        ("réduit de 2,5 pour-cent.", 0.025),
        ("ten minste 60 pct. genieten", 0.6),
        ("ten minste 60 pct genieten", 0.6),
        ("verhoogd met 2 procentpunten", 0.02),
        ("Der Satz beträgt 15 v. H.", 0.15),
        ("Der Satz beträgt 15 v.H. des Betrags", 0.15),
        ("Der Satz beträgt 9 Prozent", 0.09),
        ("taux de 28\u00a0pour cent", 0.28),
        ("Der Satz beträgt 28\u202fProzent", 0.28),
        ("taux de 0,5 POUR CENT", 0.005),
        ("augmentées de 20 pour cent pour chaque enfant", 0.2),
        ("un intérêt de 5 pour cent l'an", 0.05),
        ("un intérêt de 5 pour cent par an", 0.05),
        ("réduit de 5 pour cent au moins", 0.05),
        ("réduit de 5 pour cent maximum", 0.05),
        ("réduit de 5 pour cent, à compter du", 0.05),
        ("réduit de 5 pour cent jusqu'à la fin", 0.05),
        ("augmentées de 20 pour cent\npour chaque enfant", 0.2),
        ("un taux de 5 pour cent d'un montant", 0.05),
        ("1° 28 pour cent;\n2° 30 pour cent.", 0.28),
        ("1° 28 pour cent;\n2° 30 pour cent.", 0.3),
        ("5 pour cent, 10 pour cent", 0.05),
        # Each rate of a list is one ("5 pour cent, 10 pour cent et 15").
        ("5 pour cent, 10 pour cent et 15 pour cent", 0.05),
        ("5 pour cent, 10 pour cent et 15 pour cent", 0.1),
        ("5 pour cent, 10 pour cent et 15 pour cent", 0.15),
        ("5 pour cent et 10 pour cent du montant", 0.05),
        ("5 pour cent et 10 pour cent du montant", 0.1),
        # A list closes on its last figure, however long it is.
        (
            "1 pour cent, 2 pour cent, 3 pour cent, 4 pour cent, 5 pour cent "
            "et 6 pour cent du montant",
            0.01,
        ),
        ("2 pour cent et 3 procent", 0.02),
        ("2 pour cent et 4 procentpunten", 0.02),
        ("2 pour cent et 1\u00a0000,5 pour cent du montant", 0.02),
        ("de 10 pour cent à 20 pour cent du montant", 0.1),
        ("de 5 pour cent ou 3 % du montant", 0.05),
        # A quotation mark set against "cent" closes it.
        ("le taux est '5 pour cent' du montant", 0.05),
        ('le taux est "5 pour cent".', 0.05),
        ("le taux est « 5 pour cent » du montant", 0.05),
        ("taux de 28 POUR CENT DU MONTANT", 0.28),
        ("The rate is 5 pct of income", 0.05),
    ),
)
def test_locale_percent_markers_ground_the_rate(source_text, value):
    assert _grounds(source_text, value)


@pytest.mark.parametrize(
    ("source_text", "value"),
    (
        # A numeral after "pour cent" counts a larger hundred.
        ("5 pour cent mille habitants", 0.05),
        ("3 pour cent vingt jours", 0.03),
        ("2 pour cent millions d'euros", 0.02),
        # ... across a no-break space or a line break too.
        ("5 pour cent\u00a0mille habitants", 0.05),
        ("5 pour cent\nmille habitants", 0.05),
        ("5 pour cent\r\nmille habitants", 0.05),
        ("5 pour cent milliers", 0.05),
        # A noun after "pour cent" counts a hundred of that noun: mass,
        # area, volume, energy or anything else.
        ("un droit de 2 pour cent kilogrammes", 0.02),
        ("une taxe de 2,50 pour cent litres", 0.025),
        ("un droit de 3 pour cent kg", 0.03),
        ("un droit de 2 pour cent quintaux de poids net", 0.02),
        ("un droit de 2 pour cent hectares de superficie", 0.02),
        ("une taxe de 2,50 pour cent centimètres cubes de cylindrée", 0.025),
        ("une taxe de 3 pour cent kilowattheures", 0.03),
        ("une taxe de 3 pour cent mètres carrés", 0.03),
        ("une redevance de 4 pour cent habitants", 0.04),
        ("un droit de 2 pour cent\u00a0hectares", 0.02),
        ("un droit de 2 pourcent hectares", 0.02),
        # A hyphen joins "cent" to the numeral it starts, and an opening
        # parenthesis may restate the cardinal: neither ends the marker.
        ("5 pour cent-mille habitants", 0.05),
        ("une redevance de 5 pour cent-mille habitants", 0.05),
        ("un droit de 2 pour cent (100) kilogrammes", 0.02),
        ("un droit de 2 pour cent.000 kilogrammes", 0.02),
        # A closed-class word before a figure may start a count range.
        ("un droit de 2 pour cent à deux cents kilogrammes", 0.02),
        ("un droit de 2 pour cent jusqu'à deux cents kilogrammes", 0.02),
        ("un droit de 2 pour cent à vingt-deux kilogrammes", 0.02),
        ("un droit de 2 pour cent à 200 kilogrammes", 0.02),
        ("un droit de 2 pour cent et un kilogrammes", 0.02),
        # "quelques", "plusieurs" and "demi", behind "et" or "ou" too, and a
        # quotation mark that opens a word, may continue a count.
        ("un droit de 2 pour cent et quelques kilogrammes", 0.02),
        ("un droit de 2 pour cent et demi kilogrammes", 0.02),
        ("un droit de 2 pour cent, et demi", 0.02),
        ("un droit de 2 pour cent de quelques kilogrammes", 0.02),
        ('un droit de 2 pour cent "kilogrammes"', 0.02),
        ('un droit de 2 pour cent "cent kilogrammes"', 0.02),
        ("un droit de 2 pour cent, deux cents kilogrammes", 0.02),
        ("un droit de 2 pour cent, 200 kilogrammes", 0.02),
        # Only listed phrases close "pour cent": "et", "en" and "soit" do not.
        ("un droit de 2 pour cent en tout", 0.02),
        ("un droit de 2 pour cent soit", 0.02),
        # Per mille is no percent.
        ("à concurrence de 0,15 pour mille des prestations", 0.0015),
        # Words that only begin like a marker.
        ("9 centimes", 0.09),
        ("9 pourcentage", 0.09),
        ("9 procedures", 0.09),
        ("9 pctg", 0.09),
    ),
)
def test_near_markers_do_not_ground_the_rate(source_text, value):
    assert not _grounds(source_text, value)


@pytest.mark.parametrize(
    ("citation_path", "source_text", "formula"),
    (
        # "pct" and "v. H." are markers in their own case only.
        ("us/statute/example/1", "The office received 28 PCT applications.", "0.28"),
        ("uk/statute/example/1", "The office received 28 PCT applications.", "0.28"),
        ("ca/statute/example/1", "The office received 28 PCT applications.", "0.28"),
        (BE_CITATION, "The office received 28 PCT applications.", "0.28"),
        ("us/statute/example/1", "officers of the 28 Pct. shall", "0.28"),
        # Outside Belgium "pct." is no marker at all, as on main: it may name
        # a police precinct.
        ("us/statute/example/1", "The officers of the 28 pct. shall report.", "0.28"),
        ("uk/statute/example/1", "The officers of the 28 pct. shall report.", "0.28"),
        ("ca/statute/example/1", "The officers of the 28 pct. shall report.", "0.28"),
        # Dutch "v.h." is "voorheen", formerly.
        (BE_CITATION, "artikel 15 v.h. Wetboek", "0.15"),
        (
            "us/statute/example/1",
            "| Pct. | Votes |\n| --- | --- |\n| 28 | 4 |\n",
            "0.28",
        ),
    ),
)
def test_abbreviations_in_another_case_are_no_rate_marker(
    citation_path, source_text, formula
):
    assert not _gate_grounds(source_text, formula, citation_path=citation_path)


@pytest.mark.parametrize(
    ("citation_path", "source_text", "formula"),
    (
        # Outside Belgium the Belgian words, the restated figure and the
        # Belgian table header are read as on main: no rate.
        ("us/statute/example/1", "28 pour cent du montant", "0.28"),
        ("uk/statute/example/1", "taux de 14,75 pourcents", "0.1475"),
        ("us/statute/example/1", "tarief 9 procent", "0.09"),
        ("us/statute/example/1", "Der Satz beträgt 15 v. H.", "0.15"),
        ("de/statute/estg/32a", "28 pour cent du montant", "0.28"),
        ("us/statute/example/1", "sont pour (75) p.c. remboursés", "0.75"),
        ("us/statute/example/1", "f (14,75) p.c.du montant", "0.1475"),
        ("de/statute/estg/32a", "Satz von (14,75) Prozent", "0.1475"),
        ("us/statute/example/1", "entre 10 et 20 pour cent du montant", "0.1"),
        (
            "us/statute/example/1",
            "| Categorie | tarief in procent |\n| --- | --- |\n| A | 28 |\n",
            "0.28",
        ),
        # A label after "met", English or Dutch, before an uppercase "P.C.".
        (
            "us/statute/example/1",
            "P.C. means police constable. The panel met (17) P.C. applicants",
            "0.17",
        ),
        (
            BE_CITATION,
            "P.C. means police constable. The panel met (17) P.C. applicants",
            "0.17",
        ),
        (BE_CITATION, "Annexure Six (17) P.C.", "0.17"),
        (BE_CITATION, "sections five and six (17) P.C.", "0.17"),
        (BE_CITATION, "sont pour (75) P.C. remboursés", "0.75"),
    ),
)
def test_readings_outside_their_profile_are_as_on_main(
    citation_path, source_text, formula
):
    assert not _gate_grounds(source_text, formula, citation_path=citation_path)


def _value_issues(citation_path: str, source_text: str, expected) -> list[str]:
    return [
        issue.replace(citation_path, "<source>")
        for issue in _find_source_text_value_issues(
            source_label=citation_path,
            source_text=source_text,
            value_name="rate",
            expected_value=expected,
        )
    ]


def test_belgian_sources_take_the_legacy_numeric_readers():
    # be-BE is the legacy reading with the Belgian markers: the strict
    # envelope check and the strict source-verification branches stay off.
    assert not _has_malformed_profiled_numeric_envelope("1.2345,6", profile="be-BE")
    assert _has_malformed_profiled_numeric_envelope("1.2345,6", profile="de-DE")
    for source_text, expected in (
        ("Bedrag: 1.234 euro", 1.234),
        ("Taux: 28 %; plafond 2.500 EUR", {"rate": 0.28, "cap": 2500}),
    ):
        legacy = _value_issues("us/statute/example/1", source_text, expected)
        assert _value_issues(BE_CITATION, source_text, expected) == legacy
        assert _value_issues("de/statute/estg/32a", source_text, expected) != legacy
    # Source verification reads the Belgian markers in Belgian sources only.
    assert not _value_issues(BE_CITATION, "Le taux est de 14,75 pour cent.", 0.1475)
    assert _value_issues(
        "us/statute/example/1", "Le taux est de 14,75 pour cent.", 0.1475
    )


@pytest.mark.parametrize(
    ("profile", "grounded"), (("be-BE", True), ("legacy", False), ("en-US", False))
)
def test_mixed_fraction_reads_its_profile_markers(profile, grounded):
    source_text = "un taux de 16 1\u20442 pour cent du montant"

    assert _grounds(source_text, 0.165, profile=profile) is grounded
    # The recall inventory reads the marked fraction as one rate, the way
    # main reads "16 1⁄2 percent".
    inventory = extract_typed_numeric_inventory_occurrences_from_text(
        source_text, profile=profile
    )
    assert (
        any(item.value == 0.165 and item.has_rate_context for item in inventory)
        is grounded
    )


@pytest.mark.parametrize(
    ("profile", "grounded"),
    (("de-DE", True), ("be-BE", True), ("legacy", False), ("en-GB", False)),
)
def test_german_v_h_header_types_its_column(profile, grounded):
    table = "| Art | Satz in v. H. |\n| --- | --- |\n| A | 28 |\n"

    assert _grounds(table, 0.28, profile=profile) is grounded


@pytest.mark.parametrize(
    ("profile", "header", "cell", "rate", "own_figures"),
    (
        ("be-BE", "Tarief (procent) 2026", "2,5", 0.025, (20.26,)),
        ("be-BE", "Taux (pour cent) au 1er janvier 2026", "3", 0.03, (0.01, 20.26)),
        ("be-BE", "Satz in v. H. ab 2026", "3", 0.03, (20.26,)),
        ("de-DE", "Satz in v. H. ab 2026", "3", 0.03, (20.26,)),
    ),
)
def test_locale_header_types_its_column_and_not_its_own_figures(
    profile, header, cell, rate, own_figures
):
    # A header that names its column in a Belgian or German word types the
    # cells below it; a year or a day beside the word states no rate.
    table = f"| Categorie | {header} |\n| --- | --- |\n| A | {cell} |\n"

    assert _grounds(table, rate, profile=profile)
    for value in own_figures:
        assert not _grounds(table, value, profile=profile), value


@pytest.mark.parametrize("profile", ("be-BE", "legacy"))
def test_header_in_main_words_types_its_own_cell_as_on_main(profile):
    # main types the figures in a header cell that names a percent column in
    # its own words, so "Rate (percent) 2026" grounds 20.26 there; that header
    # is main's to fix.
    table = "| Category | Rate (percent) 2026 |\n| --- | --- |\n| A | 3 |\n"

    assert _grounds(table, 20.26, profile=profile)


def test_pour_mille_still_grounds_the_printed_figure():
    assert _grounds("à concurrence de 0,15 pour mille des prestations", 0.15)


@pytest.mark.parametrize(
    ("source_text", "value"),
    (
        ("f (14,75) p.c.du montant de sa rémunération", 0.1475),
        ("Une part de (8,75) p.c. comprise dans la cotisation", 0.0875),
        ("taux de (1.000,5) p.c. du montant", 10.005),
        ("tarief van (0,5) pct.", 0.005),
        ("Satz von (2,5) v. H.", 0.025),
        ("remboursés pour (75) pour cent du montant", 0.75),
        ("taux de ( 14,75 ) pour cent", 0.1475),
        ("taux de (101) % du montant", 1.01),
        ("remboursés pour (150) pour cent du montant", 1.5),
        ("Satz von (101.5) Prozent", 1.015),
        ("tarief van (9) procent", 0.09),
        ("Satz von (15) Prozent", 0.15),
    ),
)
def test_parenthesized_figure_takes_the_marker_after_the_parenthesis(
    source_text, value
):
    assert _grounds(source_text, value)


def test_parenthesized_integer_followed_by_a_marker_is_no_subdivision_label():
    occurrences = extract_typed_numeric_occurrences_from_text(
        "remboursés pour (75) pour cent du montant", profile="be-BE"
    )

    assert [(item.value, item.raw, item.has_rate_context) for item in occurrences] == [
        (75.0, "75", True)
    ]
    # Outside Belgium "(75)" stays the masked label it is on main.
    assert not extract_typed_numeric_occurrences_from_text(
        "remboursés pour (75) pour cent du montant"
    )


@pytest.mark.parametrize(
    ("source_text", "value"),
    (
        # The figure before the parenthesis owns the marker; "(2)" is a note.
        ("taux de 5 (2) p.c.", 0.02),
        ("taux de 5,5 (2) pour cent", 0.02),
        # No opening parenthesis: a list label, not a restatement.
        ("2) p.c. du montant", 0.02),
        ("article 2) pour cent", 0.02),
        # Subdivision labels with no marker after them stay labels.
        ("(1) Le taux est fixé par le Roi.", 0.01),
        ("(3) Der Satz beträgt 9 Prozent", 0.03),
        ("(2) % du montant", 0.02),
        ("The rate is 5 (7) percent", 0.07),
    ),
)
def test_parenthesized_figure_needs_its_own_marker(source_text, value):
    assert not _grounds(source_text, value)


@pytest.mark.parametrize(
    "citation_path",
    (
        BE_CITATION,
        "us/statute/example/1",
        "de/statute/example/1",
        "dk/statute/example/1",
    ),
)
@pytest.mark.parametrize(
    "source_text",
    (
        "The rate is 5 (7) percent",
        "taux de 5 (7) pour cent",
        "tarief 5 (7) pct.",
        # Any space or a comma between the figure and the note.
        "The rate is 5\u2009(7) percent",
        "The rate is 5, (7) percent",
        "taux de 5,(7) pour cent",
    ),
)
def test_note_after_a_figure_stays_masked(citation_path, source_text):
    # The figure before the parenthesis owns the marker, so "(7)" stays the
    # note it was before locale markers: neither 7 nor 0.07 grounds.
    assert not _gate_grounds(source_text, "7", citation_path=citation_path)
    assert not _gate_grounds(source_text, "0.07", citation_path=citation_path)


@pytest.mark.parametrize(
    "source_text", ("The rate is 5 (7) percent", "The rate is 5\u2009(7) percent")
)
def test_note_after_a_figure_keeps_the_english_owner_rate(source_text):
    assert _gate_grounds(source_text, "0.05", citation_path="us/statute/example/1")


@pytest.mark.parametrize(
    ("citation_path", "source_text", "formulas"),
    (
        # A label opening a line or a clause is a label, whatever follows it.
        (
            "us/statute/example/1",
            "Section 3 shall apply. (17) Percent of income is excluded.",
            ("17", "0.17"),
        ),
        (
            "uk/statute/example/1",
            "Section 3 shall apply: (17) per cent of the amount",
            ("17", "0.17"),
        ),
        ("us/statute/example/1", "(17) Percent of income", ("17", "0.17")),
        (BE_CITATION, "a) (7) pct. du montant", ("7", "0.07")),
        ("dk/statute/example/1", "a) (7) pct. af indkomsten", ("7", "0.07")),
    ),
)
def test_label_opening_a_clause_stays_masked(citation_path, source_text, formulas):
    for formula in formulas:
        assert not _gate_grounds(source_text, formula, citation_path=citation_path)


def test_danish_profile_masks_a_parenthesized_figure_as_main_did():
    # da-DK reads no rate from a parenthesized figure, so "(75)" stays the
    # masked label it was; the be-BE profile reads it as a restated rate.
    danish = "svarende til (75) pct. af grundlaget"

    assert not _gate_grounds(danish, "75", citation_path="dk/statute/example/1")
    assert not _gate_grounds(danish, "0.75", citation_path="dk/statute/example/1")
    assert _gate_grounds("remboursés pour (75) pour cent du montant", "0.75")


@pytest.mark.parametrize(
    ("source_text", "value"),
    (
        # A note that is no label shape ("(7,5)", "(100)") is not masked, but
        # the figure before it still owns the marker.
        ("le taux de 5 (7,5) pour cent", 0.075),
        ("Der Satz beträgt 5 (7,5) Prozent", 0.075),
        ("le taux de 5\u2009(7,5) pour cent", 0.075),
        ("le taux de 5, (7,5) pour cent", 0.075),
        ("le taux de 5\n(7,5) pour cent", 0.075),
        ("le taux de 5 (100) p.c.", 1.0),
    ),
)
def test_figure_before_a_restated_note_owns_the_marker(source_text, value):
    assert not _grounds(source_text, value)


@pytest.mark.parametrize(
    ("source_text", "values"),
    (
        ("entre 10 et 20 pour cent du montant", (0.1, 0.2)),
        ("van 10 tot 20 procent", (0.1, 0.2)),
        ("zwischen 10 bis 20 Prozent", (0.1, 0.2)),
        ("de 10 à 20 %", (0.1, 0.2)),
        # Cleaning blanks "ET 17" as a bill-style reference ("HB 1234"); in
        # the source it is the list's last member, which closes at "par an",
        # so the range reads its marker there.
        (
            "un droit de 3 à 7 pour cents à 11 pour cents et 13 P. C. ET 17 % par an",
            (0.03, 0.07, 0.11),
        ),
    ),
)
def test_range_endpoints_share_the_marker(source_text, values):
    for value in values:
        assert _grounds(source_text, value)


@pytest.mark.parametrize(
    ("source_text", "values"),
    (
        # The range reader.
        ("entre 2 et 3 pour cent kilogrammes", (0.02, 0.03)),
        ("de 2 à 3 pour cent hectares", (0.02, 0.03)),
        # The decimal-comma reader.
        ("un droit de 2,5 pour cent quintaux", (0.025,)),
        ("une taxe de 1.000,5 pour cent litres", (10.005,)),
        # ... and before a count's continuation.
        ("entre 2 et 3 pour cent et quelques kilogrammes", (0.02, 0.03)),
        ('de 2 à 3 pour cent "kilogrammes"', (0.02, 0.03)),
        ("un droit de 2,5 pour cent et demi kilogrammes", (0.025,)),
        ('un droit de 2,5 pour cent "cent kilogrammes"', (0.025,)),
    ),
)
def test_noun_after_pour_cent_is_no_marker_in_any_reader(source_text, values):
    for value in values:
        assert not _grounds(source_text, value), (source_text, value)


@pytest.mark.parametrize(
    ("source_text", "formulas"),
    (
        # The last figure of a list decides it: a count there makes counts
        # of the whole list.
        (
            "un droit de 2 pour cent et 3 pour cent kilogrammes de poids net",
            ("0.02", "0.03"),
        ),
        (
            "Droits respectifs : 2 pour cent et 3 pour cent kilogrammes de poids net.",
            ("0.02", "0.03"),
        ),
        (
            "une redevance de 2 pour cent et 3 pour cent mille habitants",
            ("0.02", "0.03"),
        ),
        (
            "une redevance de 2 pour cent et 3 pour cent-mille habitants",
            ("0.02", "0.03"),
        ),
        (
            "un droit de 2 pour cent et 3 pour cent et quelques kilogrammes",
            ("0.02", "0.03"),
        ),
        ("un droit de 2 pour cent, 3 pour cent kilogrammes", ("0.02", "0.03")),
        ("une redevance de 2 pour cent, 3 pour cent mille habitants", ("0.02", "0.03")),
        ("un droit de 2 pour cent ou 3 pour cent kilogrammes", ("0.02", "0.03")),
        ("un droit de 2 pour cent à 3 pour cent kilogrammes", ("0.02", "0.03")),
        ("un droit de 2 pour cent et 3 pour cent, 200 euros", ("0.02", "0.03")),
        # However long the list.
        (
            "un droit de 2 pour cent, 3 pour cent et 4 pour cent kilogrammes",
            ("0.02", "0.03", "0.04"),
        ),
        (
            "un droit de 2 pour cent, 3 pour cent, 4 pour cent, 5 pour cent "
            "et 6 pour cent quintaux",
            ("0.02", "0.03", "0.04", "0.05", "0.06"),
        ),
        # The range reader, at both ends.
        (
            "un droit de 1 à 2 pour cent et 3 pour cent kilogrammes de poids net",
            ("0.01", "0.02", "0.03"),
        ),
        # The decimal-comma reader.
        (
            "un droit de 2,5 pour cent et 3 pour cent kilogrammes de poids net",
            ("0.025", "0.03"),
        ),
        # A figure restated in parentheses, label-shaped or not.
        (
            "un droit de (2) pour cent et 3 pour cent kilogrammes de poids net",
            ("0.02", "0.03"),
        ),
        (
            "un droit de (2,5) pour cent et 3 pour cent kilogrammes de poids net",
            ("0.025", "0.03"),
        ),
        # A table's header row and a body cell.
        (
            "| Droit pour cent et 3 pour cent kilogrammes de poids net | Base |\n"
            "| --- | --- |\n| 2 | 5000 euro |\n",
            ("0.02", "50"),
        ),
        (
            "| Droit | Base |\n| --- | --- |\n"
            "| 2 pour cent et 3 pour cent kilogrammes de poids net | 5000 euro |\n",
            ("0.02", "0.03", "50"),
        ),
        # The last marker closes as "pour cent" does, whether it is "%",
        # "p.c." or "procent": a count after it leaves the list's "pour
        # cent" figures unread. The figure it marks grounds its own rate, as
        # on main.
        ("un droit de 2 pour cent et 3 p.c. kilogrammes de poids net", ("0.02",)),
        ("un droit de 2 pour cent et 3 % kilogrammes de poids net", ("0.02",)),
        ("un droit de 2 pour cent ou 3 P.C. kilogrammes", ("0.02",)),
        ("un droit de 2 pour cent,3 p. c. kilogrammes", ("0.02",)),
        ("un droit de 2 pourcents et\n3 p.c. kilogrammes", ("0.02",)),
        ("un droit de 2 pour-cent à 3 % kilogrammes", ("0.02",)),
        ("un droit de 7 pour cent, 11 procenten kilogrammes de poids net", ("0.07",)),
        ("un droit de 7 pour cent, 11 procentpunt kilogrammes", ("0.07",)),
        # A marked figure in the middle of the list, too.
        (
            "un droit de 2 pour cent, 3 p.c., 4 pour cent kilogrammes de poids net",
            ("0.02", "0.04"),
        ),
        (
            "un droit de 3 pour cent, 4 pour cent, 5 pour cent, 6 pour cent, "
            "7 pour cent et 8 p.c. quintaux",
            ("0.03", "0.04", "0.05", "0.06", "0.07"),
        ),
        # A source that defines the abbreviation as something else.
        (
            "p.c. désigne les parties civiles. Un droit de 2 pour cent et 3 p.c. "
            "kilogrammes de poids net.",
            ("0.02",),
        ),
        (
            "P.C. means police constable. A levy of 2 pour cent et 3 P.C. kilogrammes.",
            ("0.02",),
        ),
        # Every reader: the range ends, the decimal comma, a label-shaped
        # parenthesis and the figure it would restate, a figure outside the
        # label shape, a header row and a body cell.
        (
            "un droit de 1 à 2 pour cent et 3 p.c. kilogrammes de poids net",
            ("0.01", "0.02"),
        ),
        ("un droit de 2,5 pour cent et 3 p.c. kilogrammes de poids net", ("0.025",)),
        ("un droit de (2,5) pour cent et 3 p.c. kilogrammes de poids net", ("0.025",)),
        (
            "p.c. désigne les parties civiles. Un droit de (17) pour cent et 3 p.c. "
            "kilogrammes de poids net.",
            ("0.17", "17"),
        ),
        (
            "p.c. désigne les parties civiles. Un droit de (101) pour cent et 3 p.c. "
            "kilogrammes de poids net.",
            ("1.01",),
        ),
        (
            "| Droit pour cent et 3 p.c. kilogrammes de poids net | Base |\n"
            "| --- | --- |\n| 7 | 5000 euro |\n",
            ("0.07",),
        ),
        (
            "| Droit | Base |\n| --- | --- |\n"
            "| 2 pour cent et 3 p.c. kilogrammes de poids net | 5000 euro |\n",
            ("0.02",),
        ),
    ),
)
def test_rate_list_ending_in_a_count_grounds_no_rate(source_text, formulas):
    for formula in formulas:
        assert not _gate_grounds(source_text, formula), (source_text, formula)


@pytest.mark.parametrize(
    "continuation",
    (
        " ou moins",
        " lorsque le bénéficiaire est majeur",
        " dans les cas suivants",
        " hors TVA",
        " sans supplément",
        " selon la catégorie",
        " après révision",
        " avant abattement",
        " et non de 10 %",
        " au-delà du plafond",
        " à titre de contribution",
    ),
)
def test_unlisted_continuation_leaves_pour_cent_unread(continuation):
    # Real rates the phrase list does not cover; main grounds none of them
    # either.
    assert not _gate_grounds(f"Le taux est de 5 pour cent{continuation}", "0.05")


@pytest.mark.parametrize(
    ("source_text", "formulas"),
    (
        (
            "un taux de 2 pour cent, 3 p.c. et 4 pour cent du montant",
            ("0.02", "0.03", "0.04"),
        ),
        ("un taux de 2 pour cent et 3 % du montant", ("0.02", "0.03")),
        ("un taux de 2 pour cent et 3 p.c.", ("0.02", "0.03")),
        (
            "de 5 pour cent, 10 % et 15 p.c., à compter du 1er janvier",
            ("0.05", "0.1", "0.15"),
        ),
        ("un taux de 7 pour cent, 11 procenten du montant", ("0.07", "0.11")),
        ("un taux de 7 pour cent et 12 procentpunten par an", ("0.07", "0.12")),
    ),
)
def test_rate_list_with_other_markers_grounds_every_rate(source_text, formulas):
    for formula in formulas:
        assert _gate_grounds(source_text, formula), (source_text, formula)


@pytest.mark.parametrize(
    "source_text",
    (
        "Le taux est de 5 pour cent ou 3 % de 1000 euros",
        "Le taux est de 5 pour cent ou 3 % applicable aux revenus",
        "Le droit est de 5 pour cent et 3 p.c. de 200 euros",
        # The abbreviation's full stop ends the sentence, and no other closes
        # the list.
        "Le taux est de 5 pour cent et 3 p.c. Le taux est indexé.",
    ),
)
def test_list_ending_in_an_unlisted_continuation_leaves_pour_cent_unread(source_text):
    # The last marker closes as "pour cent" does, so these real rates go with
    # the counts; main grounds none of them either.
    assert not _gate_grounds(source_text, "0.05")


@pytest.mark.parametrize(
    ("citation_path", "source_text", "formulas"),
    (
        # A label after a word that introduces no amount stays a label.
        (
            "us/statute/example/1",
            "Contents\nItem (17) Percent of income disregarded\n",
            ("17", "0.17"),
        ),
        (
            "us/statute/example/1",
            "Subject to clause (17) Percent of income.",
            ("17", "0.17"),
        ),
        (BE_CITATION, "Taux (12) p.c. du montant", ("12", "0.12")),
        (
            BE_CITATION,
            "conformément au point (13) pour cent du montant",
            ("13", "0.13"),
        ),
        (BE_CITATION, "zie lid (12) procent van het bedrag", ("12", "0.12")),
        ("be-dg/statute/example/1", "nach Absatz (12) Prozent", ("12", "0.12")),
        # Two spaces are no restatement either.
        (BE_CITATION, "sont pour  (75) p.c. remboursés", ("75", "0.75")),
        # Nor is a lead-in word with no rate marker after the parenthesis.
        (BE_CITATION, "sont pour (75) remboursés", ("75", "0.75")),
        (BE_CITATION, "tarief van (19) euro", ("19", "0.19")),
        # "bis" numbers an inserted article in Belgian law.
        (BE_CITATION, "article 5 bis (12) p.c. du montant", ("12", "0.12")),
        # A spelled number before the label never licenses it.
        (
            "us/statute/example/1",
            "A provision under Schedule Six (17) Percent of income disregarded",
            ("17", "0.17"),
        ),
        (
            "us/statute/example/1",
            "Subject to clause six (17) percent of income",
            ("17", "0.17"),
        ),
        (
            BE_CITATION,
            "conformément à l'article six (12) pour cent du montant",
            ("12", "0.12"),
        ),
        (BE_CITATION, "visé à l'article vingt-huit (28) p.c. du montant", ("28",)),
        (
            BE_CITATION,
            "conformément à l'article soixante et onze (71) pour cent",
            ("71", "0.71"),
        ),
        ("be-dg/statute/example/1", "nach Schedule Six (17) Prozent", ("17", "0.17")),
        # Only the Belgian, Dutch and German markers license a restatement.
        (BE_CITATION, "le taux de (12) % du montant", ("12", "0.12")),
        (BE_CITATION, "le taux de (12) percent du montant", ("12", "0.12")),
        # "pc" without its dots is too short to tell from a label's tail.
        (BE_CITATION, "le taux de (12) pc du montant", ("12", "0.12")),
        # A lead-in word follows a space, an apostrophe or an opening bracket
        # or quote. The end of a URL is none, and nor is a word glued to a
        # footnote mark, which cleaning blanks: a cleaner that read the
        # cleaned text would keep "(17)" there for a rate the readers decline.
        (
            BE_CITATION,
            "https://example.be/de (17) pour cent du montant",
            ("17", "0.17"),
        ),
        (BE_CITATION, "https://example.be/van (9) procent", ("9", "0.09")),
        (BE_CITATION, "www.example.be/pour (75) pour cent", ("75", "0.75")),
        (BE_CITATION, "[1]de (17) pour cent du montant", ("17", "0.17")),
        # Nor is a closing quote, a hyphen or a digit.
        (BE_CITATION, "taux \u00bbde (17) pour cent du montant", ("17", "0.17")),
        (BE_CITATION, "taux \u201dde (17) pour cent du montant", ("17", "0.17")),
        (BE_CITATION, "Nord-de (17) pour cent du montant", ("17", "0.17")),
        (BE_CITATION, "taux 5de (17) pour cent du montant", ("17", "0.17")),
    ),
)
def test_label_after_a_word_that_states_no_amount_stays_masked(
    citation_path, source_text, formulas
):
    for formula in formulas:
        assert not _gate_grounds(source_text, formula, citation_path=citation_path)


@pytest.mark.parametrize(
    ("citation_path", "source_text", "formula"),
    (
        (BE_CITATION, "remboursés pour (75) pour cent du montant", "0.75"),
        (BE_CITATION, "tarief van (9) procent", "0.09"),
        (BE_CITATION, "met (17) procent verhoogd", "0.17"),
        ("be-dg/statute/example/1", "Der Satz beträgt (15) Prozent", "0.15"),
        ("be-dg/statute/example/1", "Satz von (15) vom Hundert", "0.15"),
        ("be-vlg/statute/example/1", "bedraagt (12) procent", "0.12"),
        (BE_CITATION, "réduit de (17) pour cent.", "0.17"),
        (BE_CITATION, "réduit jusqu'à (17) pour cent du montant", "0.17"),
        # An apostrophe, or a mark a source types for one.
        (BE_CITATION, "réduit jusqu\u2019à (17) pour cent du montant", "0.17"),
        (BE_CITATION, "réduit jusqu\u00b4à (17) pour cent du montant", "0.17"),
        (BE_CITATION, "réduit jusqu`à (17) pour cent du montant", "0.17"),
        (BE_CITATION, "réduit jusqu\u02bcà (17) pour cent du montant", "0.17"),
        (BE_CITATION, "réduit jusqu\u2032à (17) pour cent du montant", "0.17"),
        # An opening bracket or quote.
        (BE_CITATION, "(de (17) pour cent du montant)", "0.17"),
        (BE_CITATION, "[de (17) pour cent du montant]", "0.17"),
        (BE_CITATION, "taux \u00abde (17) pour cent du montant", "0.17"),
        (BE_CITATION, "taux \u201cde (17) pour cent du montant", "0.17"),
        (BE_CITATION, "taux \u201ede (17) pour cent du montant", "0.17"),
        (BE_CITATION, "taux \u2018de (17) pour cent du montant", "0.17"),
        (BE_CITATION, 'taux "de (17) pour cent du montant', "0.17"),
    ),
)
def test_figure_restated_after_a_lead_in_word_grounds(
    citation_path, source_text, formula
):
    assert _gate_grounds(source_text, formula, citation_path=citation_path)


@pytest.mark.parametrize(
    ("source_text", "formula"),
    (
        # Outside the label shape, a whole or dotted figure needs the lead-in a
        # label needs: after any other word it may label, count or cite.
        ("Contents\nItem (117) Percent of income disregarded", "1.17"),
        ("conformément à l'article six (112) procent", "1.12"),
        ("The rules apply. (117) procent", "1.17"),
        ("tarief (101) % van het bedrag", "1.01"),
        ("le taux de  (101) % du montant", "1.01"),
        ("Item (101.5) pour cent du montant", "1.015"),
        # The end of a URL is no lead-in.
        ("https://example.be/de (117) pour cent du montant", "1.17"),
        ("www.example.be/pour (175) pour cent", "1.75"),
        ("example.de (117) % du montant", "1.17"),
    ),
)
def test_whole_figure_outside_the_label_shape_needs_a_lead_in(source_text, formula):
    assert not _gate_grounds(source_text, formula), source_text


_VLG_CITATION = "be-vlg/statute/example/1"
_DG_CITATION = "be-dg/statute/example/1"


# A Belgian source's own rate words, restatements and headers type only a
# reading that takes its figure whole and the Belgian way: "2,5 procent" is
# 2.5 percent and never 25, "1.250 pour cent" is 1250 percent and never
# 1.25, and no part of "1.250,5" or "16 1/2" is a rate by itself.
@pytest.mark.parametrize(
    ("citation_path", "source_text", "grounded", "refused"),
    (
        # A decimal comma, which main's readers strip.
        (
            _VLG_CITATION,
            "Het percentage bedraagt 2,5 procent van het bedrag.",
            ("0.025",),
            ("0.25",),
        ),
        (_VLG_CITATION, "Het percentage bedraagt 12,5 procent.", ("0.125",), ("1.25",)),
        (
            _VLG_CITATION,
            "Het percentage, vermeld in het eerste lid, bedraagt 1,5 procent.",
            ("0.015",),
            ("0.15",),
        ),
        (
            _VLG_CITATION,
            "De percentages worden vastgesteld op 0,5 procent en 1,25 procent.",
            ("0.005", "0.0125"),
            ("0.05",),
        ),
        # main's "percent" earlier in the text types no later figure.
        (
            BE_CITATION,
            "Le texte anglais dit percent. Le taux est de 2,5 pour cent du montant.",
            ("0.025",),
            ("0.25",),
        ),
        (BE_CITATION, "taux de 1,250 pour cent", ("0.0125",), ("12.5",)),
        (
            BE_CITATION,
            "un taux de 2,375 pour cent du montant",
            ("0.02375",),
            ("23.75",),
        ),
        (_VLG_CITATION, "een tarief van 1,125 procent", ("0.01125",), ("11.25",)),
        (BE_CITATION, "un taux de 1,125 pct. du montant", ("0.01125",), ("11.25",)),
        # A minus sign before the figure leaves its Belgian reading whole.
        (BE_CITATION, "Le taux est de -2,5 pour cent.", ("-0.025",), ("0.025",)),
        (BE_CITATION, "Le taux est de -1.250 pour cent.", ("-12.5",), ("-0.0125",)),
        # A whole number spelled out has no other reading.
        (BE_CITATION, "Le taux est de cinq pour cent du montant.", ("0.05",), ()),
        (BE_CITATION, "Le taux est de vingt-cinq pour cent.", ("0.25",), ()),
        (_VLG_CITATION, "Het tarief bedraagt vijf procent.", ("0.05",), ()),
        # English groups no figure that starts with a zero, so "0,125" holds a
        # decimal comma before any marker.
        (BE_CITATION, "taux de (0,500) percent du montant", ("0.005",), ("5",)),
        (BE_CITATION, "taux de (0,125) percent du montant", ("0.00125",), ("1.25",)),
        (BE_CITATION, "taux de (0,125) % du montant", ("0.00125",), ("1.25",)),
        (BE_CITATION, "taux de (0,125) pour cent du montant", ("0.00125",), ("1.25",)),
        (BE_CITATION, "taux de (0,125) procent du montant", ("0.00125",), ("1.25",)),
        (BE_CITATION, "taux de (0,125) p.c. du montant", ("0.00125",), ("1.25",)),
        # main reads these parenthesized figures only the English way, as
        # 1250 or 125, so they ground no rate.
        (BE_CITATION, "taux de (01,250) % du montant", (), ("12.5", "0.0125")),
        (BE_CITATION, "taux de (00,125) % du montant", (), ("1.25", "0.00125")),
        (BE_CITATION, "taux de (12,500) % du montant", (), ("125",)),
        (BE_CITATION, "taux de (123,456) % du montant", (), ("1234.56",)),
        (BE_CITATION, "taux de (-1,250) % du montant", (), ("-12.5",)),
        (BE_CITATION, "taux de (1,250) % du montant", (), ("12.5", "0.0125")),
        (BE_CITATION, "taux de (1,250) pour cent du montant", (), ("12.5", "0.0125")),
        (BE_CITATION, "taux de (1,250) procent du montant", (), ("12.5", "0.0125")),
        # Dots between groups of three, which main may read as a decimal point.
        (BE_CITATION, "taux de (1.250) pour cent du montant", ("12.5",), ("0.0125",)),
        (BE_CITATION, "taux de 1.250 pour cent du montant", ("12.5",), ("0.0125",)),
        (_VLG_CITATION, "een tarief van 2.500 procent", ("25",), ("0.025",)),
        # A Belgian header types the same readings in its column, and a full
        # stop that no digit follows ends the figure in its cell.
        (
            _VLG_CITATION,
            "| Categorie | Tarief (procent) |\n| --- | --- |\n| A | 1.250 |",
            ("12.5",),
            ("0.0125",),
        ),
        (
            _VLG_CITATION,
            "| Categorie | Tarief (procent) |\n| --- | --- |\n| A | 2,5. |",
            ("0.025",),
            (),
        ),
        # A part of one figure: after a mark, a grouping space or a whole
        # number, or before a fraction slash, even where a year range starts
        # with it ("1.2025-2026").
        (BE_CITATION, "Le taux est de 1.2025-2026 pour cent.", (), ("20.25",)),
        (
            _VLG_CITATION,
            "Het percentage bedraagt 1.250,5 procent.",
            ("12.505",),
            ("0.05", "0.0125"),
        ),
        (BE_CITATION, "taux de 12 345,5 pour cent", ("123.455",), ("3.455",)),
        (
            _VLG_CITATION,
            "Het percentage bedraagt 1 250,5 procent.",
            ("12.505",),
            ("2.505", "25.05"),
        ),
        # main reads "1 250" as two figures, and neither is the rate.
        (BE_CITATION, "taux de 1 250 pour cent", (), ("2.5", "12.5")),
        # Nor is a part of a figure grouped with an apostrophe.
        (BE_CITATION, "taux de 1'250 pour cent du montant", (), ("2.5",)),
        (BE_CITATION, "taux de 2'5 pour cent du montant", (), ("0.05",)),
        (BE_CITATION, "taux de 1\u2019250 pour cent du montant", (), ("2.5",)),
        (
            BE_CITATION,
            "un taux entre 1,250 et 2,500 pour cent",
            ("0.0125", "0.025"),
            ("12.5", "25"),
        ),
        (BE_CITATION, "un taux de 1/2 pour cent", (), ("0.02",)),
        (BE_CITATION, "un taux de 16 1/2 pour cent", (), ("0.02",)),
        # A slash joins a fraction with spaces around it too.
        (BE_CITATION, "un taux de 1 / 2 pour cent", (), ("0.02",)),
        (BE_CITATION, "un taux de 16 1 / 2 pour cent", (), ("0.02",)),
        (BE_CITATION, "un taux de 16 1 \u2044 2 pour cent", (), ("0.02",)),
        (_VLG_CITATION, "een tarief van 3 /4 procent", (), ("0.04",)),
        # Any spaces around a slash, as main reads "1    /    2 %" (0.5
        # percent), and the division slash and the fullwidth solidus.
        (BE_CITATION, "taux de 1    /    2 pour cent du montant", (), ("0.02",)),
        (BE_CITATION, "taux de 1     /2 pour cent du montant", (), ("0.02",)),
        (BE_CITATION, "taux de 1/     2 pour cent du montant", (), ("0.02",)),
        (BE_CITATION, "taux de 1\t\t\t\t/2 pour cent du montant", (), ("0.02",)),
        (BE_CITATION, "taux de 1/\n\n\n\n2 pour cent du montant", (), ("0.02",)),
        (BE_CITATION, "taux de 1 \u2215 2 pour cent du montant", (), ("0.02",)),
        (BE_CITATION, "taux de 1 \uff0f 2 pour cent du montant", (), ("0.02",)),
        (BE_CITATION, "un taux de 16 1\u20442 pour cent", ("0.165",), ("0.02",)),
        (BE_CITATION, "un taux de 16\u00bd pour cent", (), ("0.005",)),
        (BE_CITATION, "un taux de 1.250 1\u20442 pour cent", (), ("0.005",)),
        (BE_CITATION, "un taux de 1,250 1\u20442 pour cent", (), ("12.505",)),
        # main's "percent" keeps both of main's readings of the whole figure;
        # only the "250" of "1,250" loses the rate main gives it.
        (BE_CITATION, "un taux de 1,250 percent", ("12.5", "0.0125"), ("2.5",)),
        # A figure only English prints has its English value, its only one.
        (BE_CITATION, "un taux de 1,250.5 pour cent", ("12.505",), ("0.0125",)),
        (_VLG_CITATION, "le taux de 28,131.58 procent du montant", ("281.3158",), ()),
        (BE_CITATION, "le taux de 1,000.0 pour cent", ("10",), ("0.01",)),
        (BE_CITATION, "taux de (1,250,000) pour cent du montant", ("12500",), ()),
    ),
)
def test_belgian_rate_word_types_only_the_belgian_reading(
    citation_path, source_text, grounded, refused
):
    _assert_grounds(citation_path, source_text, grounded, refused)


def _assert_grounds(citation_path, source_text, grounded, refused):
    for formula in grounded:
        assert _gate_grounds(source_text, formula, citation_path=citation_path), (
            source_text,
            formula,
        )
    for formula in refused:
        assert not _gate_grounds(source_text, formula, citation_path=citation_path), (
            source_text,
            formula,
        )


# main's word readers read a spelled-out number word by word where their
# tables lack it: "vingt-huit" as 20 and 8, "cent cinq" as 5. No part of a
# spelled number takes a Belgian word's rate; a whole one does.
@pytest.mark.parametrize(
    ("citation_path", "source_text", "grounded", "refused"),
    (
        (
            BE_CITATION,
            "Le taux est de vingt-huit pour cent du montant.",
            (),
            ("0.08", "0.2"),
        ),
        (BE_CITATION, "Le taux est de vingt-huit pourcent.", (), ("0.08",)),
        (BE_CITATION, "Le taux est de vingt-huit pour-cent par an.", (), ("0.08",)),
        (BE_CITATION, "un taux de soixante-quinze pour cent du montant", (), ("0.15",)),
        (
            BE_CITATION,
            "un taux de quatre-vingt-quinze pour cent du montant",
            (),
            ("0.15",),
        ),
        (BE_CITATION, "un taux de septante-deux pour cent du montant", (), ("0.02",)),
        (BE_CITATION, "un taux de nonante-neuf pour cent du montant", (), ("0.09",)),
        (BE_CITATION, "un taux de dix-sept pour cent du montant", (), ("0.07",)),
        # A space or a dash in place of the hyphen.
        (BE_CITATION, "un taux de vingt huit pour cent du montant", (), ("0.08",)),
        (BE_CITATION, "un taux de vingt\u00a0huit pour cent du montant", (), ("0.08",)),
        (BE_CITATION, "un taux de vingt - huit pour cent du montant", (), ("0.08",)),
        (BE_CITATION, "un taux de vingt\u2013huit pour cent du montant", (), ("0.08",)),
        (BE_CITATION, "un taux de vingt\u2212huit pour cent du montant", (), ("0.08",)),
        # Hundreds and thousands, and the decimal words.
        (BE_CITATION, "un taux de cent cinq pour cent du montant", (), ("0.05",)),
        (BE_CITATION, "un taux de deux cent cinq pour cent du montant", (), ("0.05",)),
        (
            BE_CITATION,
            "un taux de cent vingt-cinq pour cent du montant",
            (),
            ("0.05", "0.25"),
        ),
        (BE_CITATION, "un taux de mille cinq pour cent du montant", (), ("0.05",)),
        (
            BE_CITATION,
            "un taux de deux virgule cinq pour cent du montant",
            (),
            ("0.02", "0.05"),
        ),
        (
            _VLG_CITATION,
            "een tarief van twee komma vijf procent van het bedrag",
            (),
            ("0.05",),
        ),
        (
            _VLG_CITATION,
            "een tarief van honderd vijf procent van het bedrag",
            (),
            ("0.05",),
        ),
        (
            _VLG_CITATION,
            "Het tarief bedraagt honderd vijfentwintig procent van het bedrag.",
            (),
            ("0.25",),
        ),
        # The "et" or "en" a number goes on after.
        (BE_CITATION, "un taux de trente et un pour cent du montant", (), ("0.01",)),
        (BE_CITATION, "un taux de trente-et-un pour cent du montant", (), ("0.01",)),
        (BE_CITATION, "un taux de deux et demi pour cent du montant", (), ("0.005",)),
        (BE_CITATION, "un taux de 2 et demi pour cent du montant", (), ("0.005",)),
        (_VLG_CITATION, "een tarief van vijf en twintig procent", (), ("0.05", "0.2")),
        # A number spelled partly in words, and a figure after a number word
        # and a space or a comma.
        (BE_CITATION, "un taux de deux virgule 5 pour cent du montant", (), ("0.05",)),
        (BE_CITATION, "un taux de 2 virgule 5 pour cent du montant", (), ("0.05",)),
        (BE_CITATION, "un taux de deux 5 pour cent du montant", (), ("0.05",)),
        (BE_CITATION, "un taux de deux,5 pour cent du montant", (), ("0.05",)),
        (
            _VLG_CITATION,
            "een tarief van twee komma 5 procent van het bedrag",
            (),
            ("0.05",),
        ),
        (_VLG_CITATION, "een tarief van twee,5 procent van het bedrag", (), ("0.05",)),
        # Two provisions in the corpus spell their rates this way (be-bru
        # ordonnance of 17 July 2003, article 22; the tobacco excise law,
        # 1997003241, article 3).
        (
            BE_CITATION,
            "Nonante-cinq pour cent du produit total des amendes sont vers\u00e9s "
            "dans le fonds.",
            (),
            ("0.05",),
        ),
        (
            BE_CITATION,
            "ne peut en aucun cas \u00eatre inf\u00e9rieur \u00e0 cent cinq pourcent "
            "du total de ces accises",
            (),
            ("0.05",),
        ),
        # Whole spelled numbers, and lists of them.
        (BE_CITATION, "un taux de vingt-cinq pour cent du montant", ("0.25",), ()),
        (BE_CITATION, "un taux de nonante pour cent du montant", ("0.9",), ()),
        (
            BE_CITATION,
            "un taux de vingt-et-un pour cent du montant",
            ("0.21",),
            ("0.01",),
        ),
        (BE_CITATION, "un taux d'un quart pour cent du montant", ("0.0025",), ()),
        (
            BE_CITATION,
            "un taux de trois quarts pour cent du montant",
            ("0.0075",),
            ("0.03",),
        ),
        (_VLG_CITATION, "een tarief van vijfentwintig procent", ("0.25",), ()),
        (
            BE_CITATION,
            "un droit de deux pour cent et trois pour cent du montant",
            ("0.03",),
            (),
        ),
        (
            BE_CITATION,
            "un droit de cinq pour cent et deux pour cent du montant",
            ("0.02",),
            (),
        ),
        (
            BE_CITATION,
            "un droit de vingt pour cent, cinq pour cent du montant",
            ("0.05",),
            (),
        ),
        (_VLG_CITATION, "een tarief van vijf en twaalf procent", ("0.12",), ()),
        # A figure after a list's conjunction, a percent marker or an article
        # reads whole.
        (
            BE_CITATION,
            "Les taux sont de deux et 5 pour cent du montant.",
            ("0.05",),
            (),
        ),
        (
            BE_CITATION,
            "un taux de 3 pour cent,5 pour cent du montant",
            ("0.03", "0.05"),
            (),
        ),
        (BE_CITATION, "un 5 pour cent du montant", ("0.05",), ()),
        (_VLG_CITATION, "een 5 procent toeslag", ("0.05",), ()),
    ),
)
def test_no_part_of_a_spelled_number_takes_a_belgian_rate(
    citation_path, source_text, grounded, refused
):
    _assert_grounds(citation_path, source_text, grounded, refused)


# A figure printed in other decimal digits reads as its ASCII twin does.
@pytest.mark.parametrize(
    ("citation_path", "source_text", "grounded", "refused"),
    (
        (
            _VLG_CITATION,
            "Het percentage bedraagt \uff12,\uff15 procent van het bedrag.",
            ("0.025",),
            ("0.25",),
        ),
        (
            BE_CITATION,
            "taux de \uff11,\uff12\uff15\uff10 pour cent du montant",
            (),
            ("12.5",),
        ),
        (
            BE_CITATION,
            "taux de 1,\uff12\uff15\uff10 pour cent du montant",
            (),
            ("12.5",),
        ),
        (BE_CITATION, "taux de \uff11,250 pour cent du montant", (), ("12.5",)),
        (
            BE_CITATION,
            "un taux de \u0662,\u0663\u0667\u0665 pour cent du montant",
            (),
            ("23.75",),
        ),
        (
            BE_CITATION,
            "taux de (\uff11.\uff12\uff15\uff10) % du montant",
            (),
            ("0.0125",),
        ),
        (BE_CITATION, "taux de \uff12\uff18 pour cent du montant", ("0.28",), ()),
        (BE_CITATION, "taux de 1 / \uff12 pour cent du montant", (), ("0.02",)),
    ),
)
def test_figure_in_other_decimal_digits_reads_as_its_ascii_twin(
    citation_path, source_text, grounded, refused
):
    _assert_grounds(citation_path, source_text, grounded, refused)


# A comma or a dot right before a figure leaves the figure the tail of
# another ("taux de ,5" is 0.5 percent, never 5), unless a word, a sign or a
# closing bracket or quote sits before it: there the mark parts a list.
@pytest.mark.parametrize(
    ("source_text", "grounded", "refused"),
    (
        ("taux de ,5 pour cent du montant", (), ("0.05",)),
        ("un taux de ,25 pour cent du montant", (), ("0.25",)),
        ("taux de (,5) pour cent du montant", (), ("0.05",)),
        ("taux de \u00ab,5 pour cent\u00bb du montant", (), ("0.05",)),
        ('le taux de ",5 pour cent"', (), ("0.05",)),
        ("le taux de \u201c,5 pour cent\u201d", (), ("0.05",)),
        ("taux de 2 ,5 pour cent du montant", (), ("0.05",)),
        ("taux de 2\t,5 pour cent du montant", (), ("0.05",)),
        ("taux de 2\u00a0,5 pour cent du montant", (), ("0.05",)),
        ("taux de 2\n,5 pour cent du montant", (), ("0.05",)),
        ("taux de 2\u2028,5 pour cent du montant", (), ("0.05",)),
        ("taux de 2.,5 pour cent du montant", (), ("0.05",)),
        ("taux de 2,5 pour cent ,5 pour cent du montant", ("0.025",), ("0.05",)),
        ("le taux de 1 pour cent,2 pour cent", ("0.01", "0.02"), ()),
        ("le taux de 1 pour cent,,2 pour cent", ("0.02",), ()),
        ("le taux de 1 p.c.,2 pour cent", ("0.02",), ()),
        ("le taux de 5 %,2 pour cent", ("0.02",), ()),
        ("le taux de 5 %),2 pour cent", ("0.02",), ()),
        ("le taux fix\u00e9,2 pour cent du montant", ("0.02",), ()),
    ),
)
def test_mark_before_a_figure_parts_a_list_only_after_a_word_or_sign(
    source_text, grounded, refused
):
    _assert_grounds(BE_CITATION, source_text, grounded, refused)
    _assert_grounds(
        _VLG_CITATION,
        source_text.replace("pour cent", "procent"),
        grounded,
        refused,
    )


# The be-BE cleaner keeps a figure restated in parentheses before a word that
# spells a percent out, where main blanks it as a label. Only its Belgian
# reading reads there: a pipe cell in main's words types no other.
@pytest.mark.parametrize(
    ("citation_path", "source_text", "grounded", "refused"),
    (
        (_DG_CITATION, "| Satz | von (1.250) Prozent |", ("12.5",), ("0.0125",)),
        (_DG_CITATION, "| Satz | von (1.250) vom Hundert |", ("12.5",), ("0.0125",)),
        (_DG_CITATION, "| Satz | von (12.500) Prozent |", ("125",), ("0.125",)),
        (
            _DG_CITATION,
            "| Steuersatz | Betrag |\n| von (1.250) Prozent | 100 Euro |",
            (),
            ("0.0125",),
        ),
        (
            _DG_CITATION,
            "| Satz von (1.250) Prozentpunkte | Betrag |\n| --- | --- |\n| 1 | 2 |",
            (),
            ("0.0125",),
        ),
        (BE_CITATION, "| de (1.250) Prozent", (), ("0.0125",)),
        (BE_CITATION, "| Taux | pour (75) pour cent |", ("0.75",), ()),
    ),
)
def test_kept_restatement_reads_only_its_belgian_figure(
    citation_path, source_text, grounded, refused
):
    _assert_grounds(citation_path, source_text, grounded, refused)


@pytest.mark.parametrize(
    ("dtype", "formula", "grounded"),
    (
        ("Decimal", "1250", True),
        ("Decimal", "1.25", False),
        ("Money", "1.25", False),
        ("Integer", "1250", True),
        ("Decimal", "250", False),
    ),
)
def test_kept_restatement_grounds_no_amount_main_never_reads(dtype, formula, grounded):
    # main blanks "(1.250)" as a label and reads nothing there; the Belgian
    # reading, 1250, is the one figure the source states. The recall
    # inventory lists none of main's readings of it, as on main.
    source_text = "Satz von (1.250) Prozent"

    assert (
        _gate_grounds(source_text, formula, citation_path=_DG_CITATION, dtype=dtype)
        is grounded
    )
    assert not extract_typed_numeric_inventory_occurrences_from_text(
        source_text, profile="be-BE"
    )


@pytest.mark.parametrize(
    ("citation_path", "source_text", "formulas"),
    (
        (BE_CITATION, "un taux de 1,250 % du montant", ("0.0125", "12.5")),
        (BE_CITATION, "un taux de 1.250 % du montant", ("0.0125", "12.5")),
        (_DG_CITATION, "ein Satz von 1,125 Prozent", ("11.25",)),
        (_DG_CITATION, "ein Satz von 1.250 Prozent", ("0.0125", "12.5")),
        (_DG_CITATION, "| de 1.250 x Prozent", ("0.0125", "12.5")),
        (BE_CITATION, "taux de 1 / 2 % du montant", ("0.005",)),
        (
            BE_CITATION,
            "| Category | Rate (percent) 2026 |\n| --- | --- |\n| A | 3 |",
            ("0.03", "20.26"),
        ),
        (
            BE_CITATION,
            "| Rate (percent) |\n| --- |\n| 2,5 |\n| Rate (percent) 2026 |",
            ("20.26",),
        ),
        (BE_CITATION, "Le taux est de vingt-huit % du montant.", ("0.08", "0.28")),
        (BE_CITATION, "Le taux est de vingt\u2013huit % du montant.", ("0.08",)),
        (BE_CITATION, "taux de 1'250 % du montant", ("2.5", "12.5")),
    ),
)
def test_main_markers_keep_main_readings_in_belgian_sources(
    citation_path, source_text, formulas
):
    # A reading the Belgian way or not, a figure before main's own marker
    # grounds what it grounds on main; main's slipped readings there are
    # main's to fix.
    for formula in formulas:
        assert _gate_grounds(
            source_text, formula, citation_path=citation_path
        ) is _gate_grounds(
            source_text, formula, citation_path="us/statute/example/1"
        ), (source_text, formula)


@pytest.mark.parametrize(
    ("source_text", "grounded", "refused"),
    (
        # Cleaning blanks what parts the figures here, and the cleaned text
        # shows a range "1 à 2" that the source does not print.
        ("un droit de 1 (10) à 2 pour cent du montant", ("0.02",), ("0.01",)),
        ("un droit de 1 2026-01-01 à 2 pour cent du montant", ("0.02",), ("0.01",)),
        ("un droit de 1 [3] à 2 pour cent du montant", ("0.02",), ("0.01",)),
        ("un droit de 1 https://x.be à 2 pour cent du montant", ("0.02",), ("0.01",)),
        ("un droit de 1 1.000.000 à 2 pour cent du montant", ("0.02",), ("0.01",)),
        ("un droit de 1 ET 13 à 2 pour cent du montant", ("0.02",), ("0.01",)),
        # main reads that range before "%" and grounds 0.01; a Belgian
        # source does not.
        ("un droit de 1 (10) à 2 % du montant", ("0.02",), ("0.01",)),
        ("un taux de 1 à 2 pour cent du montant", ("0.01", "0.02"), ()),
        (
            "un droit de 3 à 7 pour cents à 11 pour cents et 13 P. C. ET 17 % par an",
            ("0.03",),
            (),
        ),
    ),
)
def test_range_endpoints_meet_in_the_source(source_text, grounded, refused):
    for formula in grounded:
        assert _gate_grounds(source_text, formula), (source_text, formula)
    for formula in refused:
        assert not _gate_grounds(source_text, formula), (source_text, formula)


def test_cleaned_figure_with_no_source_text_has_no_marker_or_range_there():
    source_text = "un taux de 12 à 17 pour cent"
    collector = _LegacyNumericCollector(source_text, profile="be-BE")
    mapped = _NumericTextView.identity(source_text)
    first, second = (11, 13), (16, 18)
    gap = re.compile(r"\s+à\s+")
    assert _figure_rate_marker_follows_in_source(
        collector, mapped, second, _BELGIAN_FIGURE_RATE_MARKER_AFTER_PATTERN
    )
    assert _range_endpoints_meet_in_source(collector, mapped, first, second, gap)
    # Figures that map to no source offset and print nowhere in the source.
    unmapped = _NumericTextView(source_text, "34 à 56", (None,) * 7)
    assert not _figure_rate_marker_follows_in_source(
        collector, unmapped, (5, 7), _BELGIAN_FIGURE_RATE_MARKER_AFTER_PATTERN
    )
    assert not _range_endpoints_meet_in_source(collector, unmapped, (0, 2), (5, 7), gap)


@pytest.mark.parametrize(
    "source_text",
    ("N1,000 de (17) pour cent du montant", "¢5 de (17) pour cent du montant"),
)
def test_label_after_an_edit_that_widens_the_text_restates_its_rate(source_text):
    # Cleaning widens "N1,000" and "¢5" by a space before the label, so the
    # cleaner finds the label in the source by its offsets, not its position.
    for formula in ("17", "0.17"):
        assert _gate_grounds(source_text, formula), (source_text, formula)


@pytest.mark.parametrize(
    "source_text",
    ("réduit de (17) pour cent 2026-01-01.", "réduit de (17) pour cent 15.03.2026."),
)
def test_label_before_a_dated_pour_cent_stays_masked(source_text):
    # A figure follows "pour cent" in the source, so no reader reads "(17)" as
    # a rate. The cleaner reads the label in the source too: once it blanks
    # the date, "pour cent" would close at the full stop and keep a label
    # that grounds its bare figure.
    for formula in ("17", "0.17"):
        assert not _gate_grounds(source_text, formula), (source_text, formula)
    assert not [
        item
        for item in extract_typed_numeric_inventory_occurrences_from_text(
            source_text, profile="be-BE"
        )
        if item.raw == "17"
    ]


@pytest.mark.parametrize(
    ("source_text", "formulas"),
    (
        # A source may define an abbreviation as anything, so a whole figure
        # in parentheses before one stays the count or citation it may be,
        # whatever word comes before it.
        (
            "p.c. means police constable. The panel met (17) p.c. applicants.",
            ("17", "0.17"),
        ),
        (
            "p.c. désigne les parties civiles. La demande est déposée pour (17) p.c.",
            ("17", "0.17"),
        ),
        # 2 itself always grounds (GROUNDING_ALLOWED_VALUES), as on main.
        (
            "p.c. désigne le Code pénal. Conformément à (2) p.c., la sanction "
            "est applicable.",
            ("0.02",),
        ),
        ("pct. means precinct. The panel met (28) pct. officers.", ("28", "0.28")),
        ("Der Satz von (15) v. H. gilt.", ("15", "0.15")),
        # A whole figure outside the label shape, which the cleaner leaves in
        # the text, stays a figure: its N/100 never grounds.
        ("p.c. means police constable. (101) p.c. applicants shall attend.", ("1.01",)),
        (
            "p.c. means police constable. The panel met (101) p.c. applicants.",
            ("1.01",),
        ),
        ("The panel met (101) P.C. applicants.", ("1.01",)),
        ("The panel met (250) pct. officers.", ("2.5",)),
        ("Der Satz von (250) v. H. gilt.", ("2.5",)),
        # Only a decimal comma makes a fraction of it: "1,250" reads as 1250,
        # and "101.5" may cite a section.
        ("taux de (1,250) p.c. du montant", ("12.5",)),
        ("conformément à (101.5) p.c.", ("1.015",)),
        # Real rates go with them; main grounds none of these either.
        ("sont pour (75) p.c. remboursés sur base des honoraires", ("75", "0.75")),
        ("met (17) p.c. verhoogd", ("17", "0.17")),
        ("taux de (14.75) p.c.", ("14.75", "0.1475")),
        ("Satz von (15) v. H.", ("15", "0.15")),
    ),
)
def test_abbreviation_restates_no_whole_or_dotted_figure(source_text, formulas):
    for formula in formulas:
        assert not _gate_grounds(source_text, formula), (source_text, formula)


def test_abbreviation_leaves_a_label_out_of_the_recall_inventory():
    # main inventories no figure for a masked label, and neither does be-BE
    # before an abbreviation: "(17)" is no value the encoding owes.
    source_text = "p.c. means police constable. The panel met (17) p.c. applicants."

    for profile in ("be-BE", "legacy"):
        assert not extract_typed_numeric_inventory_occurrences_from_text(
            source_text, profile=profile
        )


@pytest.mark.parametrize(
    ("citation_path", "source_text", "formula"),
    (
        # A rate spelled out before its figure keeps main's masking of the
        # figure: the reading is lost, as it is on main.
        (BE_CITATION, "à raison de vingt-huit (28) pour cent", "0.28"),
        (BE_CITATION, "vijfenzeventig (75) procent van het bedrag", "0.75"),
        ("be-dg/statute/example/1", "fünfundzwanzig (25) Prozent des Betrags", "0.25"),
    ),
)
def test_figure_restated_after_a_spelled_number_stays_masked(
    citation_path, source_text, formula
):
    assert not _gate_grounds(source_text, formula, citation_path=citation_path)


@pytest.mark.parametrize(
    ("source_text", "rate", "label"),
    (
        ("fifteen (15) percent of the amount", "0.15", "15"),
        ("seventy-five (75) percent of the amount", "0.75", "75"),
        ("The rate is (17) percent of the amount", None, "17"),
        # "six" is French too, but an English marker licenses no restatement.
        ("The rate is six (6) percent of income", "0.06", "6"),
    ),
)
def test_english_restatement_reads_as_on_main(source_text, rate, label):
    # main reads the spelled-out English figure and keeps "(15)" a masked
    # label; an English marker licenses no restatement, so that is unchanged.
    citation_path = "us/statute/example/1"
    if rate is not None:
        assert _gate_grounds(source_text, rate, citation_path=citation_path)
    assert not _gate_grounds(source_text, label, citation_path=citation_path)


def test_locale_marker_types_the_recall_inventory_like_prozent():
    # Source recall reads the figure as German "Prozent" has always been read:
    # the printed figure, typed as a rate.
    for marker in ("Prozent", "pour cent", "pourcent", "procent", "pct.", "v. H."):
        inventory = extract_typed_numeric_inventory_occurrences_from_text(
            f"Le taux est de 28 {marker} du montant.", profile="be-BE"
        )

        assert [(item.value, item.has_rate_context) for item in inventory] == [
            (28.0, True)
        ], marker
    # Outside Belgium the Belgian words type nothing, as on main.
    for marker in ("pour cent", "procent", "pct."):
        inventory = extract_typed_numeric_inventory_occurrences_from_text(
            f"Le taux est de 28 {marker} du montant."
        )

        assert [(item.value, item.has_rate_context) for item in inventory] == [
            (28.0, False)
        ], marker


@pytest.mark.parametrize(
    "citation_path",
    (
        # The German-speaking Community reads under the be-BE profile and
        # rulespec-de under the strict de-DE one.
        "be-dg/statute/dekret/2018/04/23/article/1",
        "de/statute/estg/32a",
    ),
)
@pytest.mark.parametrize(
    ("source_text", "formula", "grounded"),
    (
        ("Der Satz beträgt 42 Prozent.", "0.42", True),
        ("Der Satz beträgt 14,75 Prozent.", "0.1475", True),
        ("Der Satz beträgt 42 vom Hundert.", "0.42", True),
        ("Der Satz beträgt 42 v. H.", "0.42", True),
        ("Der Satz beträgt 42 Euro.", "0.42", False),
        ("Der Satz beträgt 42.", "0.42", False),
    ),
)
def test_german_percent_markers_in_both_profiles(
    citation_path, source_text, formula, grounded
):
    assert _gate_grounds(source_text, formula, citation_path=citation_path) is grounded


def _proof_module(formula: str, citation_path: str, excerpt: str) -> str:
    excerpt_literal = excerpt.replace("\\", "\\\\").replace('"', '\\"')
    excerpt_literal = excerpt_literal.replace("\n", "\\n")
    return textwrap.dedent(
        f"""
        format: rulespec/v1
        module:
          source_verification:
            corpus_citation_path: {citation_path}
        rules:
          - name: rate
            kind: parameter
            dtype: Rate
            metadata:
              proof:
                atoms:
                  - path: versions[0].formula
                    kind: rate
                    source:
                      corpus_citation_path: {citation_path}
                      excerpt: "{excerpt_literal}"
            versions:
              - effective_from: '2026-01-01'
                formula: '{formula}'
        """
    ).strip()


# Proof excerpts verbatim from rulespec-be#132 (head 6c36b88f), the modules
# whose full validation failed on these literals.
@pytest.mark.parametrize(
    ("citation_path", "excerpt", "formula"),
    (
        (
            "be/statute/loi/1987/02/27/1987022077/article/12",
            "d'admission (de la personne handicapée) dans une institution, "
            "totalement ou partiellement à charge des pouvoirs publics, d'un "
            "service public ou d'un organisme de sécurité sociale, le paiement "
            "est, dans les conditions que le Roi détermine par arrêté délibéré "
            "en Conseil des Ministres, suspendu (pour 28 pour cent) pour "
            "l'allocation d'intégration (...).) <L 1989-12-22/31, art.",
            "0.28",
        ),
        (
            "be/statute/loi/1969/06/27/1969062710/article/17",
            "f (14,75) p.c.du montant de sa rémunération, destiné au régime "
            "des vacances annuelles des travailleurs.",
            "0.1475",
        ),
        (
            "be/statute/justel/excise/tobacco/1997003241/Article 3",
            "Cigarettes : a) droit d'accise: 31,50 pour cent du prix de vente "
            "au détail",
            "0.3150",
        ),
    ),
)
def test_be_proof_excerpts_ground_their_rates(citation_path, excerpt, formula):
    assert (
        find_ungrounded_numeric_issues_scoped(
            _proof_module(formula, citation_path, excerpt),
            module_source_text=None,
            module_citation_path=citation_path,
        )
        == []
    )


# A table flattened into one token stream, as the corpus stores the Vlaamse
# Codex Fiscaliteit and Fisconet rate tables: the "in %" header no longer
# sits over its column, the rows are ragged ("450.000,01 27 49.500" omits the
# upper bound), so no reader can say which figures are rates without
# guessing. They stay ungrounded.
VLG_GIFT_TAX_TABLE = (
    "TABEL I verkrijging in rechte lijn en tussen partners gedeelte van de "
    "schenking A schijf in euro tarief, toepasselijk op het overeenstemmende "
    "gedeelte in kolom A, in % totaalbedrag van de belasting over de voorgaan- "
    "de gedeelten, in euro Vanaf tot en met 0,01 150.000 3 - 150.000,01 "
    "250.000 9 4500 250.000,01 450.000 18 13.500 450.000,01 27 49.500"
)
WAL_GIFT_TAX_TABLE = (
    "Tranche de la donation Ligne directe, entre époux et entre cohabitants "
    "légaux Entre autres personnes De à inclus a b a b EUR EUR p.c. EUR p.c. "
    "EUR 0,01 150.000 3 - 10 - 150.000,01 250.000 9 4.500 20 15.000 "
    "250.000,01 450.000 18 13.500 30 35.000 Au-delà de 450.000,01 27 49.500 "
    "40 95.000"
)


@pytest.mark.parametrize(
    ("citation_path", "excerpt", "formula"),
    (
        (
            "be-vlg/statute/vlaamse-codex-fiscaliteit/article-2.8.4.1.1/"
            "version-1461936/block-1",
            VLG_GIFT_TAX_TABLE,
            "0.18",
        ),
        (
            "be-wal/statute/fisconetplus/registration-code/article-131/"
            "guid-28aaa269-5a52-4081-b97d-b9a0bb83c305/block-1",
            WAL_GIFT_TAX_TABLE,
            "0.18",
        ),
    ),
)
def test_flattened_rate_table_stays_ungrounded(citation_path, excerpt, formula):
    issues = find_ungrounded_numeric_issues_scoped(
        _proof_module(formula, citation_path, excerpt),
        module_source_text=None,
        module_citation_path=citation_path,
    )

    assert len(issues) == 1
    assert "Ungrounded generated numeric literal: 0.18" in issues[0]


@pytest.mark.parametrize(
    "header",
    ("tarief in procent", "taux en pour cent", "Steuersatz in Prozent"),
)
def test_pipe_table_percent_column_binds_only_its_own_cells(header):
    table = (
        f"| schijf tot en met (euro) | {header} | belasting (euro) |\n"
        "| --- | --- | --- |\n"
        "| 150.000 | 3 | 0 |\n"
        "| 250.000 | 9 | 4500 |\n"
    )

    assert _grounds(table, 0.03)
    assert _grounds(table, 0.09)
    # The euro columns beside it are no rates: 4500 never grounds 45, and
    # 150.000 never grounds 1500.
    assert not _grounds(table, 45.0)
    assert not _grounds(table, 1500.0)


@pytest.mark.parametrize(
    "value_cell",
    ("28 pour cent", "28 procent", "28 pct.", "28 percent", "28 Prozent", "28 v. H."),
)
def test_marked_value_cell_declares_no_rate_column(value_cell):
    # A body cell whose marker follows its figure types that figure only; it
    # is no header, so the "5000 euro" below it never grounds 50.
    table = (
        "| Item | Value |\n"
        "| --- | --- |\n"
        f"| rebate | {value_cell} |\n"
        "| income | 5000 euro |\n"
    )

    assert _gate_grounds(table, "0.28")
    assert not _gate_grounds(table, "50")
    assert not _grounds(table, 50.0)


@pytest.mark.parametrize(
    "body_cell",
    (
        "vingt-huit pour cent",
        "vingt-huit pour cent (28 %)",
        "28 pour cent (vingt-huit pour cent)",
        "28\u2009pour cent",
        "achtentwintig procent",
        "achtundzwanzig v. H.",
    ),
)
def test_locale_words_below_the_delimiter_declare_no_column(body_cell):
    # Belgian and German words name a column only above the delimiter row.
    table = (
        "| Item | Value |\n"
        "| --- | --- |\n"
        f"| rebate | {body_cell} |\n"
        "| income | 5000 euro |\n"
    )

    assert not _gate_grounds(table, "50")


@pytest.mark.parametrize(
    "table",
    (
        "| Bracket | (1) Percent |\n| 5000 | 3 |\n",
        "| Income | Column 2 percent |\n| 5000 | 3 |\n",
        "| Income | Column 2 percent |\n| --- | --- |\n| 5000 | 3 |\n",
        "| Income | (1) Percent |\n| --- | --- |\n| 5000 | 3 |\n",
    ),
)
def test_header_with_a_figure_before_its_marker_still_types_its_column(table):
    # A header cell is no value cell: "(1) Percent" heads a percent column.
    assert _gate_grounds(table, "0.03", citation_path="us/statute/example/1")


@pytest.mark.parametrize(
    "cell",
    ("16 1/2", "16 \u00bd", "16\u00a01/2", "16 1 / 2", "16  1/2", "16\t1/2"),
)
@pytest.mark.parametrize(
    ("citation_path", "header"),
    ((_VLG_CITATION, "Tarief (procent)"), (BE_CITATION, "Taux (pour cent)")),
)
def test_no_part_of_a_mixed_number_takes_a_belgian_headers_rate(
    citation_path, header, cell
):
    # Whatever space joins a mixed number's whole to its fraction, neither
    # part states a rate by itself.
    table = f"| {header} | Bedrag |\n| --- | --- |\n| {cell} | 100 euro |"

    _assert_grounds(citation_path, table, (), ("0.16", "0.005"))


@pytest.mark.parametrize(
    ("citation_path", "header"),
    ((BE_CITATION, "Taux (pour cent)"), (_VLG_CITATION, "Tarief (procent)")),
)
def test_body_row_repeating_a_locale_header_types_no_figure_of_its_own(
    citation_path, header
):
    # A body row that repeats the header's words names the column again: its
    # year states no rate, and the column's own figures keep theirs.
    table = f"| {header} |\n| --- |\n| 2,5 |\n| {header} 2026 |"

    _assert_grounds(citation_path, table, ("0.025",), ("20.26",))


def test_locale_header_needs_a_delimiter_row():
    # Without a delimiter row no line is known to be a header, so only the
    # wording main always read types a column there.
    table = "| schijf | tarief in procent |\n| 250.000 | 9 |\n"

    assert not _grounds(table, 0.09)


@pytest.mark.parametrize(
    "table",
    (
        # Pipe text above the header row is a paragraph, not a header.
        "| Notes | taux en pour cent |\n| Item | Amount euro |\n| --- | --- |\n"
        "| Income | 5000 euro |\n",
        "| Notes | tarief in procent |\n| Item | Amount euro |\n| --- | --- |\n"
        "| Income | 5000 euro |\n",
        # A header with fewer cells than its delimiter row heads no table.
        "| Item | taux en pour cent |\n| --- | --- | --- |\n| Income | 5000 euro |\n",
        "| Item | taux en pour cent | x |\n| --- | --- |\n| Income | 5000 euro |\n",
    ),
)
def test_locale_header_types_a_column_only_right_above_a_matching_delimiter(table):
    assert not _gate_grounds(table, "50")
    assert not _grounds(table, 50.0)


@pytest.mark.parametrize(
    ("citation_path", "header", "amount"),
    (
        (BE_CITATION, "tarief in procent", "5000 euro"),
        ("us/statute/example/1", "rate percent", "5000 euro"),
    ),
)
def test_pipe_text_above_the_header_sets_no_width(citation_path, header, amount):
    # Three-cell pipe text above a mixed-border header would read the header's
    # border as an empty first cell and move its rate column onto the amount.
    table = f"Notes | words | text\n| {header} | Amount\n--- | ---\n28 | {amount}\n"

    assert _gate_grounds(table, "0.28", citation_path=citation_path)
    assert not _gate_grounds(table, "50", citation_path=citation_path)


def test_locale_header_types_its_column_under_pipe_text():
    table = (
        "| Notes | see below |\n"
        "| Item | taux en pour cent |\n"
        "| --- | --- |\n"
        "| Income | 28 |\n"
    )

    assert _grounds(table, 0.28)


@pytest.mark.parametrize(
    "header",
    ("droit en pour cent-mille", "droit pour cent (100) kg", "pour cent à deux cents"),
)
def test_header_pour_cent_before_a_count_types_no_column(header):
    table = f"| Item | {header} |\n| --- | --- |\n| A | 28 |\n"

    assert not _grounds(table, 0.28)


@pytest.mark.parametrize("header", ("taux en pour cent", "Rate (percent)"))
def test_escaped_pipe_stays_inside_its_cell(header):
    # "\|" is a literal pipe in its cell, so the table has three columns and
    # the rate header sits over the 28, not over the euro amount.
    table = (
        f"| Type \\| Category | {header} | Amount (euro) |\n"
        "| --- | --- | --- |\n"
        "| A | 28 | 5000 euro |\n"
    )

    assert _gate_grounds(table, "0.28")
    assert not _gate_grounds(table, "50")


def test_escaped_pipe_at_a_row_end_is_no_border():
    # The header ends in a literal pipe, so the table is unbordered and the
    # leading pipe of "| 5000 euro" marks an empty rate cell: the amount stays
    # in its own column.
    table = "Rate percent | Amount \\|\n--- | ---\n| 5000 euro\n"

    assert not _grounds(table, 50.0)


@pytest.mark.parametrize(
    "delimiter",
    (
        "| --- | --- |",
        "| --- | --- | ",
        "| --- | --- |\t",
        "  | --- | --- |",
        "|:---|---:|",
        "| :-: | -: |  \t",
    ),
)
def test_delimiter_row_reads_through_surrounding_whitespace(delimiter):
    header = "| Category | taux en pour cent |\n"
    assert _grounds(f"{header}{delimiter}\n| A | 28 |\n", 0.28)
    # Below that delimiter a marked value cell still types no column.
    table = f"| Item | Value |\n{delimiter}\n| rebate | 28 percent |\n"
    assert not _grounds(table + "| income | 5000 euro |\n", 50.0)


@pytest.mark.parametrize("delimiter", ("--- | ---", "--- | --- ", ":-: | -:\t"))
def test_unbordered_delimiter_row_reads_through_trailing_whitespace(delimiter):
    assert _grounds(f"Category | taux en pour cent\n{delimiter}\nA | 28\n", 0.28)


def test_pipe_table_symbol_header_is_not_a_column_marker():
    # Only word headers type a pipe-table column; a bare "%" header cell is
    # left as it was, so the column stays untyped.
    table = (
        "| schijf tot en met (euro) | tarief in % | belasting (euro) |\n"
        "| --- | --- | --- |\n"
        "| 250.000 | 9 | 4500 |\n"
    )

    assert not _grounds(table, 0.09)


# Property tests. A figure is rendered in the shapes Belgian, Dutch and German
# sources print it: decimal comma or point, dot or no-break-space thousands
# groups, English comma groups, optionally restated in parentheses.
_MARKERS = (
    "%",
    "p.c.",
    "p. c.",
    "P.C.",
    "pour cent",
    "pour-cent",
    "pourcent",
    "pourcents",
    "procent",
    "procenten",
    "procentpunt",
    "procentpunten",
    "pct.",
    "pct",
    "percent",
    "per cent",
    "Prozent",
    "Prozentpunkt",
    "Prozentpunkte",
    "Beitragssatzpunkte",
    "vom Hundert",
    "v. H.",
    "v.H.",
)
# The abbreviations, which may stand for whatever a source defines.
_ABBREVIATED_MARKERS = frozenset(
    ("p.c.", "p. c.", "P.C.", "pct.", "pct", "v. H.", "v.H.")
)
_NON_MARKERS = (
    "euro",
    "EUR",
    "jours",
    "ans",
    "maanden",
    "Euro",
    "pourcentage",
    "pour mille",
    "pour cent mille",
    "pour cent vingt",
    "pour cent\u00a0mille",
    "pour cent\nmille",
    "pour cent kilogrammes",
    "pour cent litres",
    "PCT",
    "kolom",
)
_GAPS = (" ", "\u00a0", "\u202f")
# The spaces print puts between groups of three digits.
_GROUPING_SPACES = (*_GAPS, "\u2007", "\u2009")
_CELL_GAPS = (" ", "\u00a0", "\u202f", "\u2009", "")
# A figure, then a gap, then the note: any space, a tab, a line break or a
# comma keeps the figure the owner of the marker.
_OWNER_GAPS = (" ", "\u00a0", "\u202f", "\u2009", "\t", "\n", ", ", ",")
_LOCALE_MARKERS = (
    "pour cent",
    "pour-cent",
    "pourcent",
    "pourcents",
    "procent",
    "pct.",
    "v. H.",
)
_PREFIXES = (
    "le taux de ",
    "tarief ",
    "Der Satz beträgt ",
    "cotisation de ",
    "une part de ",
)
# The prefixes that end in a word stating an amount, after which a
# parenthesized figure restates the rate.
_LEAD_IN_PREFIXES = frozenset(
    ("le taux de ", "Der Satz beträgt ", "cotisation de ", "une part de ")
)
# The markers that let a lead-in word restate a label-shaped figure: the
# words that spell a percent out.
_LICENSING_MARKERS = (
    frozenset(_MARKERS) - {"%", "percent", "per cent"} - _ABBREVIATED_MARKERS
)
# The figures a subdivision label could be: "(9)", "(14.5)".
_LABEL_SHAPED = re.compile(r"\([1-9]\d?(?:\.\d+)?\)")
# Words that state no amount, so a parenthesis after them is a label.
_STRUCTURAL_WORDS = (
    "Item",
    "clause",
    "Section",
    "Article",
    "artikel",
    "paragraph",
    "point",
    "lid",
    "Absatz",
    "Schedule",
    "Taux",
    "tarief",
    "Note",
    "Column",
)
# What follows "pour cent" in a rate, and the nouns after which it counts a
# hundred of something instead.
_CLOSING_WORDS = (
    "",
    ".",
    ", à compter du",
    " du montant",
    " des revenus",
    " de la valeur",
    " d'un montant",
    " l'an",
    " par an",
    " au moins",
    " maximum",
    " pour chaque enfant",
    " sur le montant",
    ";\n2° 30 euros",
    ")",
    " pour",
    '"',
    "' du montant",
    " »",
    ", 10 pour cent",
    " et 15 pour cent",
    " ou 3 %",
    " à 20 p.c.",
    "\n",
)
# What may continue "pour cent" into a count: a hyphen, an opening
# parenthesis, or closed-class words before a figure. The rate reading may be
# real ("5 pour cent et 5 euros"), but it is no longer sure.
_COUNT_CONTINUATIONS = (
    "-mille habitants",
    " (100) kilogrammes",
    "\n(100) kilogrammes",
    " à deux cents kilogrammes",
    " jusqu'à deux cents kilogrammes",
    " à vingt-deux kilogrammes",
    " et un kilogrammes",
    " et 5 euros",
    " de 1000 euros",
    ".000 kilogrammes",
    " (cinq pour cent)",
    ", 200 euros",
    ", deux cents kilogrammes",
    " et quelques kilogrammes",
    " et demi kilogrammes",
    ", et demi",
    " de quelques kilogrammes",
    " plusieurs kilogrammes",
    ' "kilogrammes"',
    ' "cent kilogrammes"',
    ', "cent kilogrammes"',
    " de (100) kilogrammes",
    " et 15 euros",
    " en tout",
)
_FRENCH_NOUNS = (
    "kilogrammes",
    "quintaux",
    "hectares",
    "ares",
    "centimètres cubes",
    "mètres carrés",
    "litres",
    "hectolitres",
    "stères",
    "kilowattheures",
    "tonnes",
    "habitants",
    "francs",
    "mille",
    "vingt",
)
_NOUN_GAPS = (" ", "\u00a0", "\n", "\r\n", "\t", "  ")


def _group(integer: int, separator: str) -> str:
    digits = str(integer)
    groups = []
    while len(digits) > 3:
        groups.append(digits[-3:])
        digits = digits[:-3]
    groups.append(digits)
    return separator.join(reversed(groups))


_ALL_STYLES = ("plain-comma", "plain-point", "dot-group", "nbsp-group", "english")
# The strict de-DE reader takes only the comma as the decimal mark.
_DECIMAL_COMMA_STYLES = ("plain-comma", "dot-group", "nbsp-group")


@st.composite
def _printed_figures(draw, styles=_ALL_STYLES):
    """Return (printed figure, its value) in one of the supported shapes."""
    style = draw(st.sampled_from(styles))
    places = draw(st.integers(min_value=0, max_value=2))
    if style in ("dot-group", "nbsp-group", "english"):
        integer = draw(st.integers(min_value=1_000, max_value=99_999))
        places = draw(st.integers(min_value=1, max_value=2))
    else:
        integer = draw(st.integers(min_value=1, max_value=999))
    fraction = draw(st.integers(min_value=0, max_value=10**places - 1))
    fraction_text = str(fraction).zfill(places) if places else ""
    if style == "plain-comma":
        whole, decimal = str(integer), ","
    elif style == "plain-point":
        whole, decimal = str(integer), "."
    elif style == "dot-group":
        whole, decimal = _group(integer, "."), ","
    elif style == "nbsp-group":
        whole, decimal = _group(integer, "\u00a0"), ","
    else:
        whole, decimal = _group(integer, ","), "."
    printed = whole + (decimal + fraction_text if places else "")
    value = integer + (fraction / 10**places if places else 0)
    if draw(st.booleans()):
        printed = f"({printed})"
    return printed, value


@settings(max_examples=150, deadline=None)
@given(
    figure=_printed_figures(),
    marker=st.sampled_from(_MARKERS),
    gap=st.sampled_from(_GAPS),
    prefix=st.sampled_from(_PREFIXES),
)
@example(figure=("28", 28.0), marker="pour cent", gap=" ", prefix="suspendu de ")
@example(figure=("(14,75)", 14.75), marker="p.c.", gap=" ", prefix="f ")
@example(figure=("(2,0)", 2.0), marker="p.c.", gap=" ", prefix="tarief ")
def test_marked_figure_grounds_its_rate(figure, marker, gap, prefix):
    printed, value = figure
    # main misreads an English-grouped figure before "per cent"; that defect
    # predates the locale markers and is pinned below.
    assume(not (marker == "per cent" and _ENGLISH_GROUPED_DECIMAL.fullmatch(printed)))
    source_text = f"{prefix}{printed}{gap}{marker} du montant"

    assert _grounds(source_text, value / 100) is _restates_its_rate(
        printed, value, prefix, marker
    ), source_text


def _restates_its_rate(printed: str, value: float, prefix: str, marker: str) -> bool:
    """Return whether a printed figure before ``marker`` grounds its rate."""
    if not printed.startswith("("):
        return True
    if not value.is_integer() and re.search(r",\d+\)\Z", printed):
        # A fraction with a decimal comma neither counts nor cites: any
        # marker makes it a rate, after any word.
        return True
    if _LABEL_SHAPED.fullmatch(printed):
        # "tarief (9) procent", "le taux de (9) %", "le taux de (9) p.c.": a
        # label-shaped parenthesis after a word that states no amount, or
        # before a marker that does not spell a percent out, stays the masked
        # label it is on main.
        return prefix in _LEAD_IN_PREFIXES and marker in _LICENSING_MARKERS
    # "(101) p.c." may count, "(101.5) p.c." may cite and "tarief (101) %"
    # may label: a whole or dotted figure restates a rate only after a
    # lead-in word, and never before an abbreviation.
    return prefix in _LEAD_IN_PREFIXES and marker not in _ABBREVIATED_MARKERS


_ENGLISH_GROUPED_DECIMAL = re.compile(r"\d{1,3}(?:,\d{3})+\.\d+")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Pre-existing on main: an English-grouped figure before 'per cent' is "
        "read from its last group ('000.0 per cent'), so 1,000.0 per cent "
        "never grounds 10. 'percent' and '%' read it whole."
    ),
)
def test_english_grouped_figure_before_per_cent_grounds_its_rate():
    assert _grounds("le taux de 1,000.0 per cent du montant", 10.0)


@settings(max_examples=150, deadline=None)
@given(
    figure=_printed_figures(),
    suffix=st.sampled_from(_NON_MARKERS),
    prefix=st.sampled_from(_PREFIXES),
)
@example(figure=("9", 9.0), suffix="euro", prefix="tarief ")
def test_unmarked_figure_never_grounds_its_rate(figure, suffix, prefix):
    printed, value = figure
    source_text = f"{prefix}{printed} {suffix} du montant"

    assert not _grounds(source_text, value / 100), source_text


@settings(max_examples=100, deadline=None)
@given(figure=_printed_figures(), prefix=st.sampled_from(_PREFIXES))
def test_bare_figure_never_grounds_its_rate(figure, prefix):
    printed, value = figure
    source_text = f"{prefix}{printed}."

    assert not _grounds(source_text, value / 100), source_text


_FRENCH_UNITS = (
    "z\u00e9ro",
    "un",
    "deux",
    "trois",
    "quatre",
    "cinq",
    "six",
    "sept",
    "huit",
    "neuf",
    "dix",
    "onze",
    "douze",
    "treize",
    "quatorze",
    "quinze",
    "seize",
)
_FRENCH_TENS = {
    2: "vingt",
    3: "trente",
    4: "quarante",
    5: "cinquante",
    6: "soixante",
    7: "septante",
    9: "nonante",
}
_DUTCH_UNITS = (
    "nul",
    "een",
    "twee",
    "drie",
    "vier",
    "vijf",
    "zes",
    "zeven",
    "acht",
    "negen",
    "tien",
    "elf",
    "twaalf",
    "dertien",
    "veertien",
    "vijftien",
    "zestien",
    "zeventien",
    "achttien",
    "negentien",
)
_DUTCH_TENS = {
    2: "twintig",
    3: "dertig",
    4: "veertig",
    5: "vijftig",
    6: "zestig",
    7: "zeventig",
    8: "tachtig",
    9: "negentig",
}
# What joins the words of a spelled number in print: a hyphen, a space, a
# no-break space or a dash in the hyphen's place.
_SPELLED_JOINS = ("-", " ", "\u00a0", " - ", "\u2013", "\u2212")


def _french_number(number: int, join: str, *, belgian: bool) -> str:
    """Spell 1 to 999 in French: Belgium's "septante" and "nonante" or France's
    "soixante-dix" and "quatre-vingt-dix"."""
    hundreds, rest = divmod(number, 100)
    if hundreds:
        head = "cent" if hundreds == 1 else f"{_FRENCH_UNITS[hundreds]} cent"
        if not rest:
            return head if hundreds == 1 else f"{head}s"
        return f"{head} {_french_number(rest, join, belgian=belgian)}"
    if number <= 16:
        return _FRENCH_UNITS[number]
    tens, unit = divmod(number, 10)
    if tens == 1:
        return f"dix{join}{_FRENCH_UNITS[unit]}"
    if tens == 8 or (tens == 9 and not belgian):
        base = f"quatre{join}vingt"
        if number == 80:
            return f"{base}s"
        return f"{base}{join}{_french_number(number - 80, join, belgian=belgian)}"
    if tens == 7 and not belgian:
        joint = " et " if number == 71 else join
        return f"soixante{joint}{_french_number(number - 60, join, belgian=belgian)}"
    if not unit:
        return _FRENCH_TENS[tens]
    if unit == 1:
        return f"{_FRENCH_TENS[tens]} et un"
    return f"{_FRENCH_TENS[tens]}{join}{_FRENCH_UNITS[unit]}"


def _dutch_number(number: int, join: str) -> str:
    """Spell 1 to 999 in Dutch: "vijfentwintig" for 25, "honderd vijf" for 105."""
    hundreds, rest = divmod(number, 100)
    if hundreds:
        head = "honderd" if hundreds == 1 else f"{_DUTCH_UNITS[hundreds]}honderd"
        return f"{head}{join}{_dutch_number(rest, join)}" if rest else head
    if number < 20:
        return _DUTCH_UNITS[number]
    tens, unit = divmod(number, 10)
    if not unit:
        return _DUTCH_TENS[tens]
    link = "\u00ebn" if _DUTCH_UNITS[unit].endswith("e") else "en"
    return f"{_DUTCH_UNITS[unit]}{link}{_DUTCH_TENS[tens]}"


@st.composite
def _spelled_numbers(draw):
    """Return (a number spelled out, its value) in French or Dutch.

    A ten and a unit ("vingt-huit", "trente et un"), a hundred and what
    follows it ("cent cinq"), a decimal word ("deux virgule cinq", "twee
    komma vijf"), or a Dutch compound ("vijfentwintig").
    """
    number = draw(st.integers(min_value=1, max_value=999))
    join = draw(st.sampled_from(_SPELLED_JOINS))
    style = draw(st.sampled_from(("french", "belgian", "dutch", "decimal")))
    if style == "dutch":
        return _dutch_number(number, draw(st.sampled_from(("", " ")))), float(number)
    if style == "decimal":
        whole, tenth = draw(st.integers(1, 16)), draw(st.integers(1, 9))
        word, units = draw(
            st.sampled_from((("virgule", _FRENCH_UNITS), ("komma", _DUTCH_UNITS)))
        )
        return f"{units[whole]} {word} {units[tenth]}", whole + tenth / 10
    return _french_number(number, join, belgian=style == "belgian"), float(number)


@st.composite
def _belgian_figures(draw):
    """Return (printed figure, its value, whether main reads it whole).

    A decimal comma ("2,5", "1,250"), dots or grouping spaces between groups
    of three ("1.250", "12 345,5"), a mixed number ("16 1\u20442"), or a
    number spelled out ("vingt-huit"). main reads "12 345" with no decimal
    comma, a second group mark ("1.250.000"), an ASCII slash ("16 1/2"), a
    slash with spaces around it ("16 1 / 2") and a figure or thin space in
    parts, and a spelled number whole only where its word tables hold it.
    """
    style = draw(
        st.sampled_from(
            ("decimal-comma", "dot-group", "space-group", "mixed", "spelled")
        )
    )
    if style == "spelled":
        printed, value = draw(_spelled_numbers())
        return printed, value, False
    if style == "decimal-comma":
        whole = draw(st.integers(min_value=0, max_value=999))
        fraction = draw(st.text("0123456789", min_size=1, max_size=3))
        return f"{whole},{fraction}", float(f"{whole}.{fraction}"), True
    if style == "mixed":
        whole = draw(st.integers(min_value=1, max_value=99))
        numerator, denominator = draw(st.sampled_from(((1, 2), (1, 4), (3, 4), (1, 3))))
        slash = draw(st.sampled_from(("\u2044", "/", " / ", " \u2044 ")))
        join = draw(st.sampled_from(_GAPS))
        printed = f"{whole}{join}{numerator}{slash}{denominator}"
        return printed, whole + numerator / denominator, slash == "\u2044"
    integer = draw(st.integers(min_value=1_000, max_value=9_999_999))
    join = "." if style == "dot-group" else draw(st.sampled_from(_GROUPING_SPACES))
    fraction = draw(st.text("0123456789", max_size=2))
    printed = _group(integer, join) + (f",{fraction}" if fraction else "")
    value = float(f"{integer}.{fraction}" if fraction else integer)
    return (
        printed,
        value,
        integer < 1_000_000 and (join == "." or bool(fraction) and join in _GAPS),
    )


@settings(max_examples=150, deadline=None)
@given(
    figure=_belgian_figures(),
    marker=st.sampled_from(_LOCALE_MARKERS),
    gap=st.sampled_from(_GAPS),
    prefix=st.sampled_from(_PREFIXES),
    parenthesized=st.booleans(),
)
@example(
    figure=("2,5", 2.5, True),
    marker="procent",
    gap=" ",
    prefix="tarief ",
    parenthesized=False,
)
@example(
    figure=("1.250", 1250.0, True),
    marker="pour cent",
    gap=" ",
    prefix="le taux de ",
    parenthesized=True,
)
@example(
    figure=("12 345,5", 12345.5, True),
    marker="pour cent",
    gap=" ",
    prefix="le taux de ",
    parenthesized=False,
)
@example(
    figure=("16 1/2", 16.5, False),
    marker="pct.",
    gap=" ",
    prefix="le taux de ",
    parenthesized=False,
)
@example(
    figure=("16 1 / 2", 16.5, False),
    marker="pour cent",
    gap=" ",
    prefix="le taux de ",
    parenthesized=False,
)
@example(
    figure=("vingt-huit", 28.0, False),
    marker="pour cent",
    gap=" ",
    prefix="le taux de ",
    parenthesized=False,
)
@example(
    figure=("cent cinq", 105.0, False),
    marker="pourcent",
    gap=" ",
    prefix="le taux de ",
    parenthesized=False,
)
@example(
    figure=("twee komma vijf", 2.5, False),
    marker="procent",
    gap=" ",
    prefix="tarief ",
    parenthesized=False,
)
def test_belgian_rate_word_rates_only_the_belgian_reading(
    figure, marker, gap, prefix, parenthesized
):
    printed, value, read_whole = figure
    shown = f"({printed})" if parenthesized else printed
    source_text = f"{prefix}{shown}{gap}{marker} du montant"
    occurrences = extract_typed_numeric_occurrences_from_text(
        source_text, profile="be-BE"
    )

    # Whatever reader finds it, a reading that takes the word's rate context
    # reads the whole figure the Belgian way.
    for occurrence in occurrences:
        if occurrence.has_rate_context:
            reading = (
                occurrence.value
                if occurrence.source_value is None
                else occurrence.source_value
            )
            assert reading == pytest.approx(value, rel=0, abs=1e-9), (
                source_text,
                occurrence,
            )
    if read_whole and not parenthesized:
        assert numeric_value_is_grounded(value / 100, occurrences), source_text


_LOCALE_HEADERS = (
    "Tarief (procent)",
    "Taux (pour cent)",
    "taux en pour cent",
    "Satz in v. H.",
)


@settings(max_examples=150, deadline=None)
@given(figure=_belgian_figures(), header=st.sampled_from(_LOCALE_HEADERS))
@example(figure=("16 1/2", 16.5, False), header="Tarief (procent)")
@example(figure=("16\t1/2", 16.5, False), header="Taux (pour cent)")
@example(figure=("vingt-huit", 28.0, False), header="Taux (pour cent)")
def test_belgian_header_rates_only_the_belgian_reading_of_a_cell(figure, header):
    printed, value, _ = figure
    table = f"| {header} | Bedrag |\n| --- | --- |\n| {printed} | 100 euro |"

    # A header types the cells below it as a Belgian word types the figure
    # before it: only a reading of the whole figure the Belgian way.
    for occurrence in extract_typed_numeric_occurrences_from_text(
        table, profile="be-BE"
    ):
        if occurrence.has_rate_context:
            reading = (
                occurrence.value
                if occurrence.source_value is None
                else occurrence.source_value
            )
            assert reading == pytest.approx(value, rel=0, abs=1e-9), (
                table,
                occurrence,
            )


@settings(max_examples=200, deadline=None)
@given(figure=_printed_figures())
@example(figure=("1,250.5", 1250.5))
@example(figure=("12\u00a0345,67", 12345.67))
@example(figure=("12\u2009345,67", 12345.67))
@example(figure=("1\u2007250", 1250.0))
def test_belgian_figure_value_reads_back_every_printed_figure(figure):
    printed, value = figure
    reading = _belgian_figure_value(printed.strip("()"))
    assert reading == pytest.approx(value, rel=0, abs=1e-9), printed


@pytest.mark.parametrize("space", _GROUPING_SPACES)
def test_no_part_of_a_space_grouped_figure_states_a_rate(space):
    # Whichever space print puts inside a figure, no part of the figure takes
    # the word's rate.
    for printed, part in (
        (f"12{space}345,5", "3.455"),
        (f"12{space}345", "3.45"),
        (f"16{space}\u00bd", "0.005"),
    ):
        source_text = f"taux de {printed} pour cent du montant"
        assert not _gate_grounds(source_text, part), (source_text, part)


def test_belgian_figure_check_takes_the_value_of_any_marked_span():
    # A span with no decimal or group mark has no other value to mistake; a
    # span with one must take its figure's Belgian value. Only main's markers
    # type a run of digits longer than 64 characters.
    assert _reads_belgian_figure("3-7", (0, 3), 3.0)
    assert not _reads_belgian_figure("1, 5", (0, 4), 1.5)
    assert _reads_belgian_figure("9" * 64, (0, 64), float("9" * 64))
    assert not _reads_belgian_figure("9" * 65, (0, 65), float("9" * 65))


def test_belgian_figure_check_reads_a_spaced_slash_as_a_fraction():
    # A slash with any spaces around it joins its figures into one fraction,
    # as far as the 64-character limit reaches: neither figure reads whole by
    # itself, and the fraction does.
    assert not _reads_belgian_figure("1/2", (2, 3), 2.0)
    assert not _reads_belgian_figure("1/2", (0, 1), 1.0)
    assert not _reads_belgian_figure("1 / 2", (4, 5), 2.0)
    assert not _reads_belgian_figure("1 / 2", (0, 1), 1.0)
    assert not _reads_belgian_figure("1   /   2", (8, 9), 2.0)
    assert not _reads_belgian_figure("1    /    2", (10, 11), 2.0)
    assert not _reads_belgian_figure("1    /    2", (0, 1), 1.0)
    assert not _reads_belgian_figure("1/\n\n\n\n2", (6, 7), 2.0)
    assert not _reads_belgian_figure("1 \u2044 2", (4, 5), 2.0)
    assert not _reads_belgian_figure("1 \u2215 2", (4, 5), 2.0)
    assert not _reads_belgian_figure("1 \uff0f 2", (4, 5), 2.0)
    assert _reads_belgian_figure("1 / 2", (0, 5), 0.5)
    # The other figure's digit must sit within 64 characters of this one.
    near, far = f"1{' ' * 62}/2", f"1{' ' * 63}/2"
    assert not _reads_belgian_figure(near, (64, 65), 2.0)
    assert not _reads_belgian_figure(near, (0, 1), 1.0)
    assert _reads_belgian_figure(far, (65, 66), 2.0)
    assert _reads_belgian_figure(far, (0, 1), 1.0)


def test_belgian_figure_check_reads_a_mixed_number_whole_or_not_at_all():
    # Any run of the horizontal spaces main joins a mixed number with makes
    # its whole and its fraction parts of one figure.
    for join in (" ", "  ", "\t", "\u00a0", "\u2009"):
        text = f"16{join}1/2"
        assert not _reads_belgian_figure(text, (0, 2), 16.0), text
        assert not _reads_belgian_figure(text, (2 + len(join), len(text)), 0.5), text
    assert not _reads_belgian_figure("16 \u00bd", (0, 2), 16.0)
    assert not _reads_belgian_figure("16 \u00bd", (3, 4), 0.5)
    # A line break parts them, as main reads "16\n1/2 %".
    assert _reads_belgian_figure("16\n1/2", (0, 2), 16.0)


def test_belgian_figure_check_reads_a_spelled_number_whole_or_not_at_all():
    # A reading with no printed digit reads whole only when no number word or
    # digit joins it.
    assert _reads_belgian_figure("vingt-cinq", (0, 10), 25.0)
    assert not _reads_belgian_figure("vingt-huit", (6, 10), 8.0)
    assert not _reads_belgian_figure("vingt-huit", (0, 5), 20.0)
    assert not _reads_belgian_figure("vingt\u2013huit", (6, 10), 8.0)
    assert not _reads_belgian_figure("deux cent cinq", (10, 14), 5.0)
    assert not _reads_belgian_figure("twee komma vijf", (11, 15), 5.0)
    assert not _reads_belgian_figure("2 vingt", (2, 7), 20.0)
    assert not _reads_belgian_figure("vingt 2", (0, 5), 20.0)
    # "et", "en" and "und" join a number only to what it goes on with.
    assert not _reads_belgian_figure("trente et un", (10, 12), 1.0)
    assert not _reads_belgian_figure("trente et un", (0, 6), 30.0)
    assert not _reads_belgian_figure("1 et demi", (5, 9), 0.5)
    assert not _reads_belgian_figure("vijf en twintig", (8, 15), 20.0)
    assert _reads_belgian_figure("cinq et deux", (8, 12), 2.0)
    assert _reads_belgian_figure("vijf en twaalf", (8, 14), 12.0)
    # The scale word of a percent marker ends its number.
    assert _reads_belgian_figure("deux pour cent et trois", (18, 23), 3.0)
    assert _reads_belgian_figure("deux pour-cent trois", (15, 20), 3.0)
    assert not _reads_belgian_figure("deux cent et un", (13, 15), 1.0)


def test_belgian_figure_check_fails_a_figure_after_a_number_word():
    # A number word joined to a figure's first digit leaves it a part.
    assert not _reads_belgian_figure("deux virgule 5", (13, 14), 5.0)
    assert not _reads_belgian_figure("2 virgule 5", (10, 11), 5.0)
    assert not _reads_belgian_figure("twee komma 5", (11, 12), 5.0)
    assert not _reads_belgian_figure("deux 5", (5, 6), 5.0)
    assert not _reads_belgian_figure("deux,5", (5, 6), 5.0)
    assert not _reads_belgian_figure("deux.,5", (6, 7), 5.0)
    assert not _reads_belgian_figure("vingt-5", (6, 7), 5.0)
    assert not _reads_belgian_figure("vingt–5", (6, 7), 5.0)
    assert not _reads_belgian_figure("deux cent,5", (10, 11), 5.0)
    assert not _reads_belgian_figure("un,5", (3, 4), 5.0)
    # A conjunction parts a list, the scale word of a percent marker ends its
    # number, and an article joins none.
    assert _reads_belgian_figure("deux et 5", (8, 9), 5.0)
    assert _reads_belgian_figure("pour cent,5", (10, 11), 5.0)
    assert _reads_belgian_figure("vom Hundert,5", (12, 13), 5.0)
    assert _reads_belgian_figure("ten honderd,5", (12, 13), 5.0)
    assert _reads_belgian_figure("un 5", (3, 4), 5.0)
    assert _reads_belgian_figure("een 5", (4, 5), 5.0)
    assert _reads_belgian_figure("eine 5", (5, 6), 5.0)


def test_ascii_digit_text_prints_every_decimal_digit_in_ascii():
    # Each digit stays one character, so every span still holds.
    for text, printed in (
        ("\uff12,\uff15", "2,5"),
        ("\u0662,\u0663\u0667\u0665", "2,375"),
        ("1.\u096825", "1.225"),
        ("taux de 1,250", "taux de 1,250"),
        ("\u00bd \u00b2", "\u00bd \u00b2"),
    ):
        assert _ascii_digit_text(text) == printed


@st.composite
def _notes(draw):
    """Return (printed note, its value, whether it has a label's shape)."""
    shape = draw(st.sampled_from(("label", "decimal-comma", "hundreds")))
    if shape == "label":
        value = draw(st.integers(min_value=1, max_value=99))
        return str(value), float(value), True
    if shape == "hundreds":
        value = draw(st.integers(min_value=100, max_value=999))
        return str(value), float(value), False
    whole = draw(st.integers(min_value=1, max_value=99))
    tenths = draw(st.integers(min_value=1, max_value=9))
    return f"{whole},{tenths}", whole + tenths / 10, False


@settings(max_examples=150, deadline=None)
@given(
    owner=st.integers(min_value=1, max_value=99),
    note=_notes(),
    marker=st.sampled_from(_MARKERS),
    gap=st.sampled_from(_OWNER_GAPS),
)
@example(owner=5, note=("7,5", 7.5, False), marker="pour cent", gap=" ")
@example(owner=5, note=("7", 7.0, True), marker="percent", gap="\u2009")
def test_figure_before_the_parenthesis_owns_the_marker(owner, note, marker, gap):
    printed, value, label = note
    # Equal figures would ground the owner's own rate or figure.
    assume(value not in (owner, owner * 100))
    source_text = f"le taux de {owner}{gap}({printed}) {marker} du montant"

    assert not _grounds(source_text, value / 100), source_text
    if label:
        # A label-shaped note stays masked: it grounds no literal either.
        assert not _grounds(source_text, value), source_text


@settings(max_examples=100, deadline=None)
@given(
    rate=_printed_figures(),
    marker=st.sampled_from(_MARKERS),
    hundreds=st.integers(min_value=1, max_value=9_999),
    first=st.booleans(),
    gap=st.sampled_from(_CELL_GAPS),
)
@example(rate=("(1)", 1.0), marker="pour cent", hundreds=2, first=True, gap=" ")
@example(rate=("28", 28.0), marker="percent", hundreds=50, first=True, gap="\u2009")
def test_marked_value_cell_never_types_its_column(rate, marker, hundreds, first, gap):
    # Whether the marked cell comes before or after it, an amount in the same
    # column never grounds amount/100.
    printed, _ = rate
    marked = f"| taux | {printed}{gap}{marker} |\n"
    # The marked cell may ground the amount's hundreds by itself (main reads
    # "1,000.1p.c." as 1,000); only the column's typing is under test.
    assume(not _grounds(marked, float(hundreds)))
    plain = f"| montant | {hundreds * 100} euro |\n"
    rows = marked + plain if first else plain + marked
    table = "| Item | Value |\n| --- | --- |\n" + rows

    assert not _grounds(table, float(hundreds)), table


@settings(max_examples=100, deadline=None)
@given(
    words=st.sampled_from(("vingt-huit", "zie bijlage", "Satz in", "")),
    marker=st.sampled_from(_LOCALE_MARKERS),
    hundreds=st.integers(min_value=1, max_value=9_999),
    first=st.booleans(),
)
def test_locale_words_below_the_delimiter_never_type_their_column(
    words, marker, hundreds, first
):
    marked = f"| taux | {words} {marker} |\n"
    # A spelled-out number grounds itself ("vingt" is 20); only the column's
    # typing is under test.
    assume(not _grounds(marked, float(hundreds)))
    plain = f"| montant | {hundreds * 100} euro |\n"
    rows = marked + plain if first else plain + marked
    table = "| Item | Value |\n| --- | --- |\n" + rows

    assert not _grounds(table, float(hundreds)), table


@settings(max_examples=150, deadline=None)
@given(
    figure=_printed_figures(),
    spelling=st.sampled_from(("pour cent", "pour-cent", "pourcent", "pourcents")),
    closing=st.sampled_from(_CLOSING_WORDS),
)
# Only English prints "1,000.0", so its comma is no decimal comma.
@example(figure=("(1,000.0)", 1000.0), spelling="pour cent", closing="")
def test_pour_cent_before_a_closing_word_grounds_its_rate(figure, spelling, closing):
    printed, value = figure
    source_text = f"le taux de {printed} {spelling}{closing}"

    assert _grounds(source_text, value / 100), source_text


@settings(max_examples=150, deadline=None)
@given(
    figure=_printed_figures(),
    spelling=st.sampled_from(("pour cent", "pour-cent", "pourcent", "pourcents")),
    noun=st.sampled_from(_FRENCH_NOUNS),
    gap=st.sampled_from(_NOUN_GAPS),
    lead=st.sampled_from(("un droit de ", "entre 1 et ", "de 1 à ")),
)
def test_pour_cent_before_a_noun_never_grounds_its_rate(
    figure, spelling, noun, gap, lead
):
    printed, value = figure
    # The range's other endpoint, 1, grounds 1 itself, so neither the figure
    # nor its N/100 may equal it.
    assume(value not in (1, 100))
    source_text = f"{lead}{printed} {spelling}{gap}{noun} de poids net"

    assert not _grounds(source_text, value / 100), source_text


@settings(max_examples=150, deadline=None)
@given(
    figure=_printed_figures(),
    spelling=st.sampled_from(("pour cent", "pour-cent", "pourcent", "pourcents")),
    continuation=st.sampled_from(_COUNT_CONTINUATIONS),
    lead=st.sampled_from(("un droit de ", "entre 1 et ", "de 1 à ")),
)
def test_pour_cent_before_a_count_never_grounds_its_rate(
    figure, spelling, continuation, lead
):
    printed, value = figure
    # The range's other endpoint grounds 1 and the continuation's figures
    # ground themselves, so the rate may equal none of them.
    assume(value not in (1, 5, 100))
    # Nor any figure the continuation grounds by itself.
    assume(not _grounds(f"{lead}{continuation}", value / 100))
    source_text = f"{lead}{printed} {spelling}{continuation}"

    assert not _grounds(source_text, value / 100), source_text


# What sits before a comma or a dot that leaves the figure after it the tail
# of another: nothing, a space or a line break, an opening bracket or quote,
# a sign, or another figure and its mark.
_TAIL_OPENINGS = (
    "",
    " ",
    "\t",
    "\n",
    "\u00a0",
    "\u2028",
    "(",
    "[",
    "\u00ab",
    '"',
    "\u201c",
    "-",
    "2",
    "2.",
    "2,",
)


@settings(max_examples=150, deadline=None)
@given(
    opening=st.sampled_from(_TAIL_OPENINGS),
    mark=st.sampled_from((",", ".")),
    digits=st.text("0123456789", min_size=1, max_size=3),
    marker=st.sampled_from(_LOCALE_MARKERS),
)
@example(opening="", mark=",", digits="5", marker="pour cent")
@example(opening="\u00ab", mark=",", digits="5", marker="pour cent")
@example(opening="2.", mark=",", digits="5", marker="pour cent")
def test_figure_after_a_leading_mark_never_reads_whole(opening, mark, digits, marker):
    value = int(digits)
    # ",0" is zero read either way, and an opening "2" is a figure of its own.
    assume(value not in (0, 2))
    source_text = f"le taux de {opening}{mark}{digits} {marker} du montant"
    occurrences = extract_typed_numeric_occurrences_from_text(
        source_text, profile="be-BE"
    )

    # No reading of the digits after the mark as a whole number takes the
    # word's rate. (A figure that reads ".25" as 0.25 grounds the literal 0.25
    # as itself, rate or not, as on main.)
    for occurrence in occurrences:
        if occurrence.has_rate_context:
            reading = (
                occurrence.value
                if occurrence.source_value is None
                else occurrence.source_value
            )
            assert reading != pytest.approx(value, rel=0, abs=1e-9), (
                source_text,
                occurrence,
            )


# Number words a figure can go on from: units, tens, scales and the decimal
# words, in French, Dutch and German.
_LEADING_NUMBER_WORDS = (
    "deux",
    "cinq",
    "vingt",
    "septante",
    "cent",
    "mille",
    "virgule",
    "twee",
    "vijf",
    "twintig",
    "honderd",
    "komma",
    "zwei",
    "fünf",
    "zwanzig",
    "hundert",
    "Komma",
)
_NUMBER_WORD_FIGURE_JOINS = (
    " ",
    " ",
    "  ",
    "-",
    "–",
    "−",
    " - ",
    ",",
    ".",
    ".,",
)


@settings(max_examples=150, deadline=None)
@given(
    word=st.sampled_from(_LEADING_NUMBER_WORDS),
    join=st.sampled_from(_NUMBER_WORD_FIGURE_JOINS),
    digits=st.text("0123456789", min_size=1, max_size=3),
    marker=st.sampled_from(_LOCALE_MARKERS),
)
@example(word="virgule", join=" ", digits="5", marker="pour cent")
@example(word="deux", join=",", digits="5", marker="pour cent")
@example(word="komma", join=" ", digits="5", marker="procent")
@example(word="cent", join=",", digits="5", marker="pour cent")
def test_figure_after_a_number_word_never_reads_whole(word, join, digits, marker):
    value = int(digits)
    source_text = f"le taux de {word}{join}{digits} {marker} du montant"
    occurrences = extract_typed_numeric_occurrences_from_text(
        source_text, profile="be-BE"
    )

    # No reading of the figure alone takes the word's rate: the figure goes
    # on from the number word before it.
    for occurrence in occurrences:
        if occurrence.has_rate_context:
            reading = (
                occurrence.value
                if occurrence.source_value is None
                else occurrence.source_value
            )
            assert reading != pytest.approx(value, rel=0, abs=1e-9), (
                source_text,
                occurrence,
            )


_POUR_CENT_SPELLINGS = ("pour cent", "pour-cent", "pourcent", "pourcents")
_LIST_SEPARATORS = (", ", " et ", " ou ", " à ", ",", " et\n")


@st.composite
def _pour_cent_lists(draw):
    """Return two to five figures marked "pour cent" as a list, and their values."""
    size = draw(st.integers(min_value=2, max_value=5))
    members, values = [], []
    for index in range(size):
        printed, value = draw(_printed_figures())
        separator = draw(st.sampled_from(_LIST_SEPARATORS)) if index else ""
        spelling = draw(st.sampled_from(_POUR_CENT_SPELLINGS))
        members.append(f"{separator}{printed.strip('()')} {spelling}")
        values.append(value)
    return "".join(members), tuple(values)


@settings(max_examples=150, deadline=None)
@given(rates=_pour_cent_lists(), closing=st.sampled_from(_CLOSING_WORDS))
@example(
    rates=("5 pour cent, 10 pour cent et 15 pour cent", (5.0, 10.0, 15.0)), closing=""
)
def test_rate_list_ending_in_a_rate_grounds_every_figure(rates, closing):
    listed, values = rates
    source_text = f"le taux de {listed}{closing}"

    for value in values:
        assert _grounds(source_text, value / 100), (source_text, value)


@settings(max_examples=200, deadline=None)
@given(
    rates=_pour_cent_lists(),
    continuation=st.sampled_from(
        _COUNT_CONTINUATIONS + tuple(f" {noun}" for noun in _FRENCH_NOUNS)
    ),
)
@example(rates=("2 pour cent et 3 pour cent", (2.0, 3.0)), continuation=" kilogrammes")
@example(
    rates=("2 pour cent, 3 pour cent et 4 pour cent", (2.0, 3.0, 4.0)),
    continuation=" mille habitants",
)
def test_rate_list_ending_in_a_count_never_grounds(rates, continuation):
    listed, values = rates
    lead = "un droit de "
    source_text = f"{lead}{listed}{continuation}"
    # Each figure and the continuation's own figures ground themselves; only
    # the rates are under test.
    for value in values:
        assume(not _grounds(f"{lead}{continuation}", value / 100))
        assume(all(abs(value / 100 - other) > 1e-9 for other in values))

    for value in values:
        assert not _grounds(source_text, value / 100), (source_text, value)


# The markers a later figure of a list may carry besides "pour cent".
_OTHER_LIST_MARKERS = (
    "%",
    "p.c.",
    "P.C.",
    "p. c.",
    "procent",
    "procenten",
    "procentpunt",
    "procentpunten",
)


@st.composite
def _mixed_rate_lists(draw):
    """Return a list of two to six rates, "pour cent" first and another marker later.

    Returns the list, its figures' values, and whether "pour cent" marks each.
    """
    size = draw(st.integers(min_value=2, max_value=6))
    other = draw(st.integers(min_value=1, max_value=size - 1))
    members, values, spelled = [], [], []
    for index in range(size):
        printed, value = draw(_printed_figures())
        separator = draw(st.sampled_from(_LIST_SEPARATORS)) if index else ""
        if index == 0:
            marker = draw(st.sampled_from(_POUR_CENT_SPELLINGS))
        elif index == other:
            marker = draw(st.sampled_from(_OTHER_LIST_MARKERS))
        else:
            marker = draw(st.sampled_from(_POUR_CENT_SPELLINGS + _OTHER_LIST_MARKERS))
        members.append(f"{separator}{printed.strip('()')} {marker}")
        values.append(value)
        spelled.append(marker in _POUR_CENT_SPELLINGS)
    return "".join(members), tuple(values), tuple(spelled)


@settings(max_examples=150, deadline=None)
@given(rates=_mixed_rate_lists(), closing=st.sampled_from(_CLOSING_WORDS))
@example(
    rates=("2 pour cent, 3 p.c. et 4 pour cent", (2.0, 3.0, 4.0), (True, False, True)),
    closing=" du montant",
)
def test_mixed_rate_list_ending_in_a_rate_grounds_every_figure(rates, closing):
    listed, values, _ = rates
    # Before "%" or "p.c.", main reads no figure that a comma glues to the
    # word or sign before it ("fixé,2 %", "5 %,10 %", "1 pour cent,2 p.c.");
    # that predates the list rule.
    assume(not re.search(r"(?<=[^\W\d]|[%.]),[^ ]+ (?:%|[pP]\. ?[cC]\.)", listed))
    source_text = f"le taux de {listed}{closing}"

    for value in values:
        assert _grounds(source_text, value / 100), (source_text, value)


@settings(max_examples=200, deadline=None)
@given(
    rates=_mixed_rate_lists(),
    continuation=st.sampled_from(
        _COUNT_CONTINUATIONS + tuple(f" {noun}" for noun in _FRENCH_NOUNS)
    ),
)
@example(
    rates=("2 pour cent et 3 p.c.", (2.0, 3.0), (True, False)),
    continuation=" kilogrammes",
)
@example(
    rates=("2 pour cent, 3 %, 4 pour cent", (2.0, 3.0, 4.0), (True, False, True)),
    continuation=" quintaux",
)
@example(
    rates=("7 pour cent, 11 procenten", (7.0, 11.0), (True, False)),
    continuation=" mille habitants",
)
def test_mixed_rate_list_ending_in_a_count_grounds_no_pour_cent_rate(
    rates, continuation
):
    listed, values, spelled = rates
    lead = "un droit de "
    source_text = f"{lead}{listed}{continuation}"
    # A figure marked "%", "p.c." or "procent" grounds its own rate, as on
    # main, and each figure grounds itself; only the "pour cent" rates are
    # under test.
    marked = {value for value, is_spelled in zip(values, spelled) if not is_spelled}
    under_test = [value for value, is_spelled in zip(values, spelled) if is_spelled]
    for value in under_test:
        assume(value not in marked)
        assume(not _grounds(f"{lead}{continuation}", value / 100))
        assume(all(abs(value / 100 - other) > 1e-9 for other in values))

    for value in under_test:
        assert not _grounds(source_text, value / 100), (source_text, value)


# Counts that cleaning blanks after "pour cent": a label-shaped figure, a
# dotted figure, a figure that opens a line with a colon later on it, a
# label-shaped list member, a date. The source still shows the count.
_BLANKED_COUNTS = (
    " de (10) kilogrammes",
    ", (10) kilogrammes",
    ", 1.000.000 kilogrammes",
    ",\n3 pour cent kilogrammes de poids net : voir l'annexe",
    " et 3 p.c.,\n4 pour cent kilogrammes de poids net : voir l'annexe",
    ", (3) pour cent kilogrammes",
    " du 2026-01-01",
    " du 2026-01-01.",
    " du 15.03.2026.",
)
# A figure as each reader finds it, and the rates it would state: the
# after-figure reader's "2", the range's "1 à 2", the decimal comma's "2,5"
# and the mixed fraction's "2 1⁄2".
_READER_FIGURES = (
    ("2", (("2", 0.02),)),
    ("1 à 2", (("1", 0.01), ("2", 0.02))),
    ("2,5", (("2,5", 0.025),)),
    ("2 1⁄2", (("2 1⁄2", 0.025),)),
)


@pytest.mark.parametrize("continuation", _BLANKED_COUNTS)
@pytest.mark.parametrize(("figure", "rates"), _READER_FIGURES)
def test_count_that_cleaning_blanks_keeps_pour_cent_unread(figure, rates, continuation):
    # Every reader judges "pour cent" in the source, where the count still
    # follows it, in the grounding stream and in the recall inventory.
    source_text = f"un droit de {figure} pour cent{continuation}"
    for _, rate in rates:
        assert not _grounds(source_text, rate), (source_text, rate)
    printed = {raw for raw, _ in rates}
    assert not [
        item
        for item in extract_typed_numeric_inventory_occurrences_from_text(
            source_text, profile="be-BE"
        )
        if item.raw in printed and item.has_rate_context
    ], source_text


# The markers the range, decimal-comma and mixed-fraction readers take.
_FIGURE_READER_MARKERS = _POUR_CENT_SPELLINGS + (
    "%",
    "p.c.",
    "P.C.",
    "procent",
    "procenten",
    "procentpunt",
    "pct.",
    "v. H.",
    "Prozent",
    "vom Hundert",
    "percent",
    "per cent",
)


@pytest.mark.parametrize(
    "continuation",
    _CLOSING_WORDS
    + _COUNT_CONTINUATIONS
    + tuple(f" {noun}" for noun in _FRENCH_NOUNS)
    + _BLANKED_COUNTS,
)
def test_every_reader_reads_a_marker_as_the_after_figure_reader_does(continuation):
    # The after-figure reader judges a marker in the source. The range,
    # decimal-comma and mixed-fraction readers find their figures in the
    # cleaned text, and judge the same marker the same way.
    for marker in _FIGURE_READER_MARKERS:
        rate = _grounds(f"un droit de 2 {marker}{continuation}", 0.02)
        for figure, value in (("1 à 2", 0.01), ("2,5", 0.025), ("2 1⁄2", 0.025)):
            source_text = f"un droit de {figure} {marker}{continuation}"
            assert _grounds(source_text, value) == rate, (source_text, rate)


@settings(max_examples=200, deadline=None)
@given(
    prefix=st.sampled_from(("", "N1,000 ", "¢5 ", "[1]", "voir https://example.be/")),
    lead=st.sampled_from(
        (
            "réduit de ",
            "remboursés pour ",
            "tarief van ",
            "Satz von ",
            "Item ",
            "5 ",
            "de ",
        )
    ),
    label=st.integers(min_value=1, max_value=99),
    marker=st.sampled_from(_MARKERS),
    continuation=st.sampled_from(
        _CLOSING_WORDS
        + _COUNT_CONTINUATIONS
        + (" 2026-01-01.", " 15.03.2026.", " du 1er janvier 2026", " le 1er mars")
    ),
)
@example(
    prefix="",
    lead="réduit de ",
    label=17,
    marker="pour cent",
    continuation=" 2026-01-01.",
)
@example(prefix="[1]", lead="de ", label=17, marker="pour cent", continuation="")
def test_kept_label_always_reads_as_a_rate(prefix, lead, label, marker, continuation):
    # The cleaner keeps a label-shaped parenthesis only for the readers to
    # read its figure as a rate. A kept figure they decline would ground
    # itself, where main masks the label. The prefix puts an edit before
    # the label: cleaning widens "N1,000" and "¢5" by a space and blanks
    # "[1]" and the URL.
    source_text = f"{prefix}{lead}({label}) {marker}{continuation}"
    start = len(prefix) + len(lead) + 1
    span = (start, start + len(str(label)))
    for occurrences in (
        extract_typed_numeric_occurrences_from_text(source_text, profile="be-BE"),
        extract_typed_numeric_inventory_occurrences_from_text(
            source_text, profile="be-BE"
        ),
    ):
        for item in occurrences:
            if item.span == span:
                assert item.has_rate_context, source_text


@settings(max_examples=200, deadline=None)
@given(
    figure=st.integers(min_value=1, max_value=99_999),
    abbreviation=st.sampled_from(sorted(_ABBREVIATED_MARKERS)),
    lead=st.sampled_from(
        (
            "",
            "met ",
            "pour ",
            "à ",
            "de ",
            "von ",
            "The panel met ",
            "La demande est déposée pour ",
            "p.c. means police constable. ",
        )
    ),
    continuation=st.sampled_from(
        ("", ".", " applicants.", ", la sanction est applicable.", " du montant")
    ),
)
@example(
    figure=101,
    abbreviation="p.c.",
    lead="p.c. means police constable. ",
    continuation=" applicants.",
)
def test_abbreviation_never_restates_a_whole_figure(
    figure, abbreviation, lead, continuation
):
    source_text = f"{lead}({figure}) {abbreviation}{continuation}"

    assert not _grounds(source_text, figure / 100), source_text
    if figure < 100:
        # A label-shaped figure stays the masked label it is on main.
        assert not _grounds(source_text, float(figure)), source_text


@settings(max_examples=150, deadline=None)
@given(
    word=st.sampled_from(_STRUCTURAL_WORDS),
    label=st.integers(min_value=1, max_value=99),
    marker=st.sampled_from(_MARKERS),
    opening=st.sampled_from(("", "Contents\n", "Subject to ", "apply. ")),
)
def test_label_after_a_structural_word_never_grounds(word, label, marker, opening):
    source_text = f"{opening}{word} ({label}) {marker} du montant"

    assert not _grounds(source_text, label / 100), source_text
    assert not _grounds(source_text, float(label)), source_text


_SPELLED_SUBDIVISIONS = (
    "six",
    "Six",
    "vingt-huit",
    "deux bis",
    "drei",
    "vijf",
    "elf",
    "fünfundzwanzig",
    "vijfenzeventig",
)


@settings(max_examples=150, deadline=None)
@given(
    word=st.sampled_from(_STRUCTURAL_WORDS),
    spelled=st.sampled_from(_SPELLED_SUBDIVISIONS),
    label=st.integers(min_value=1, max_value=99),
    marker=st.sampled_from(_MARKERS),
    opening=st.sampled_from(("", "conformément à l'", "Subject to ", "nach ")),
)
def test_label_after_a_spelled_subdivision_never_grounds(
    word, spelled, label, marker, opening
):
    # A spelled number grounds itself and its parts ("vingt-huit" is 28 and
    # 8); only the label is under test.
    unlabelled = f"{opening}{word} {spelled} {marker} du montant"
    assume(not _grounds(unlabelled, float(label)))
    assume(not _grounds(unlabelled, label / 100))
    source_text = f"{opening}{word} {spelled} ({label}) {marker} du montant"

    assert not _grounds(source_text, label / 100), source_text
    assert not _grounds(source_text, float(label)), source_text


@settings(max_examples=150, deadline=None)
@given(
    header=st.sampled_from(
        ("taux en pour cent", "tarief in procent", "Satz in Prozent", "Percent")
    ),
    escaped_in_header=st.integers(min_value=0, max_value=3),
    escaped_in_body=st.integers(min_value=0, max_value=3),
    bordered=st.booleans(),
    rate=st.integers(min_value=1, max_value=99),
    hundreds=st.integers(min_value=1, max_value=9_999),
)
def test_escaped_pipes_never_move_a_rate_column(
    header, escaped_in_header, escaped_in_body, bordered, rate, hundreds
):
    # Literal pipes in the first cell of the header or the body row leave
    # the rate column over the rate and off the amount.
    assume(hundreds not in (rate, rate * 100))
    cells = (
        ("Type" + " \\| x" * escaped_in_header, header, "Amount (euro)"),
        ("---", "---", "---"),
        ("A" + " \\| y" * escaped_in_body, str(rate), f"{hundreds * 100} euro"),
    )
    rows = (
        "| " + " | ".join(row) + " |" if bordered else " | ".join(row) for row in cells
    )
    table = "\n".join(rows) + "\n"

    assert _grounds(table, rate / 100), table
    assert not _grounds(table, float(hundreds)), table


@settings(max_examples=100, deadline=None)
@given(
    figure=_printed_figures(styles=_DECIMAL_COMMA_STYLES),
    marker=st.sampled_from(("Prozent", "vom Hundert", "v. H.", "%")),
)
def test_german_profile_marked_figure_grounds_its_rate(figure, marker):
    printed, value = figure
    # The de-DE reader takes no figure restated in parentheses, as on main.
    assume(not printed.startswith("("))
    source_text = f"Der Satz beträgt {printed} {marker} des Betrags."

    assert _grounds(source_text, value / 100, profile="de-DE"), source_text
