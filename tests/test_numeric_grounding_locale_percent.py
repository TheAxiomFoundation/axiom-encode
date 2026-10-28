"""Belgian French, Dutch and German percent wording in numeric grounding.

A rate literal grounds only when its source marks the figure as a
percentage. Belgian legal French and Dutch write that marker out ("28 pour
cent", "9 procent") or restate the figure in parentheses before it ("(14,75)
p.c."); German statutes abbreviate "vom Hundert" ("15 v. H."). A figure with
no marker must never ground N/100: the July 2026 encoder accepted "tarief 9
euro" as 0.09, and the gate exists to refuse that.
"""

import re
import textwrap

import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    find_ungrounded_numeric_issues,
    find_ungrounded_numeric_issues_scoped,
    numeric_value_is_grounded,
)

BE_CITATION = "be/statute/loi/1987/02/27/1987022077/article/12"


def _rate_rulespec(formula: str, *, citation_path: str = BE_CITATION) -> str:
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
) -> bool:
    return not find_ungrounded_numeric_issues(
        _rate_rulespec(formula, citation_path=citation_path),
        source_text=source_text,
        source_citation_path=citation_path,
    )


def _grounds(source_text: str, value: float, *, profile: str = "legacy") -> bool:
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
        ("us/statute/example/1", "officers of the 28 Pct. shall", "0.28"),
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


def test_pour_mille_still_grounds_the_printed_figure():
    assert _grounds("à concurrence de 0,15 pour mille des prestations", 0.15)


@pytest.mark.parametrize(
    ("source_text", "value"),
    (
        ("f (14,75) p.c.du montant de sa rémunération", 0.1475),
        ("Une part de (8,75) p.c. comprise dans la cotisation", 0.0875),
        ("sont pour (75) p.c. remboursés sur base des honoraires", 0.75),
        ("taux de ( 14,75 ) pour cent", 0.1475),
        ("taux de (14.75) p.c.", 0.1475),
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
        "sont pour (75) p.c. remboursés"
    )

    assert [(item.value, item.raw, item.has_rate_context) for item in occurrences] == [
        (75.0, "75", True)
    ]


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
    # masked label it was; the legacy profile reads it as a restated rate.
    danish = "svarende til (75) pct. af grundlaget"

    assert not _gate_grounds(danish, "75", citation_path="dk/statute/example/1")
    assert not _gate_grounds(danish, "0.75", citation_path="dk/statute/example/1")
    assert _gate_grounds("sont pour (75) p.c. remboursés", "0.75")


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
    ),
)
def test_noun_after_pour_cent_is_no_marker_in_any_reader(source_text, values):
    for value in values:
        assert not _grounds(source_text, value), (source_text, value)


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
        (BE_CITATION, "à raison de vingt-huit (28) pour cent", "0.28"),
        (BE_CITATION, "vijfenzeventig (75) procent van het bedrag", "0.75"),
        (BE_CITATION, "tarief van (9) procent", "0.09"),
        ("be-dg/statute/example/1", "fünfundzwanzig (25) Prozent des Betrags", "0.25"),
        ("be-dg/statute/example/1", "Der Satz beträgt (15) Prozent", "0.15"),
    ),
)
def test_figure_restated_after_a_lead_in_word_grounds(
    citation_path, source_text, formula
):
    assert _gate_grounds(source_text, formula, citation_path=citation_path)


@pytest.mark.parametrize(
    ("source_text", "rate", "label"),
    (
        ("fifteen (15) percent of the amount", "0.15", "15"),
        ("seventy-five (75) percent of the amount", "0.75", "75"),
        ("The rate is (17) percent of the amount", None, "17"),
    ),
)
def test_english_restatement_reads_as_on_main(source_text, rate, label):
    # main reads the spelled-out English figure and keeps "(15)" a masked
    # label; no English word is a lead-in, so that is unchanged.
    citation_path = "us/statute/example/1"
    if rate is not None:
        assert _gate_grounds(source_text, rate, citation_path=citation_path)
    assert not _gate_grounds(source_text, label, citation_path=citation_path)


def test_locale_marker_types_the_recall_inventory_like_prozent():
    # Source recall reads the figure as German "Prozent" has always been read:
    # the printed figure, typed as a rate.
    for marker in ("Prozent", "pour cent", "pourcent", "procent", "pct."):
        inventory = extract_typed_numeric_inventory_occurrences_from_text(
            f"Le taux est de 28 {marker} du montant."
        )

        assert [(item.value, item.has_rate_context) for item in inventory] == [
            (28.0, True)
        ], marker


@pytest.mark.parametrize(
    "citation_path",
    (
        # The German-speaking Community reads under the legacy profile and
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


def test_locale_header_needs_a_delimiter_row():
    # Without a delimiter row no line is known to be a header, so only the
    # wording main always read types a column there.
    table = "| schijf | tarief in procent |\n| 250.000 | 9 |\n"

    assert not _grounds(table, 0.09)


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
    "pour cent",
    "pour-cent",
    "pourcent",
    "pourcents",
    "procent",
    "pct.",
    "percent",
    "per cent",
    "Prozent",
    "vom Hundert",
    "v. H.",
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
    " et 5 euros",
    " (cinq pour cent)",
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
def test_marked_figure_grounds_its_rate(figure, marker, gap, prefix):
    printed, value = figure
    # main misreads an English-grouped figure before "per cent"; that defect
    # predates the locale markers and is pinned below.
    assume(not (marker == "per cent" and _ENGLISH_GROUPED_DECIMAL.fullmatch(printed)))
    source_text = f"{prefix}{printed}{gap}{marker} du montant"

    if prefix not in _LEAD_IN_PREFIXES and _LABEL_SHAPED.fullmatch(printed):
        # "tarief (9) procent": a label-shaped parenthesis after a word that
        # states no amount stays the masked label it is on main.
        assert not _grounds(source_text, value / 100), source_text
    else:
        assert _grounds(source_text, value / 100), source_text


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
    # The rate's own figure grounds itself and its N/100 (and main reads
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
    # The range reader's other endpoint (1) never equals the figure under test.
    assume(value != 1)
    source_text = f"{lead}{printed} {spelling}{gap}{noun} de poids net"

    assert not _grounds(source_text, value / 100), source_text


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
    source_text = f"Der Satz beträgt {printed} {marker} des Betrags."

    assert _grounds(source_text, value / 100, profile="de-DE"), source_text
