"""Belgian French, Dutch and German percent wording in numeric grounding.

A rate literal grounds only when its source marks the figure as a
percentage. Belgian legal French and Dutch write that marker out ("28 pour
cent", "9 procent") or restate the figure in parentheses before it ("(14,75)
p.c."); German statutes abbreviate "vom Hundert" ("15 v. H."). A figure with
no marker must never ground N/100: the July 2026 encoder accepted "tarief 9
euro" as 0.09, and the gate exists to refuse that.
"""

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
        ("taux de 28 pour cent", 0.28),
        ("Der Satz beträgt 28 Prozent", 0.28),
        ("taux de 0,5 POUR CENT", 0.005),
        ("augmentées de 20 pour cent pour chaque enfant", 0.2),
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
    ),
)
def test_parenthesized_figure_needs_its_own_marker(source_text, value):
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
    "kolom",
)
_GAPS = (" ", " ", " ")
_PREFIXES = (
    "le taux de ",
    "tarief ",
    "Der Satz beträgt ",
    "cotisation de ",
    "une part de ",
)


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
        whole, decimal = _group(integer, " "), ","
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
    source_text = f"{prefix}{printed}{gap}{marker} du montant"

    assert _grounds(source_text, value / 100), source_text


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


@settings(max_examples=100, deadline=None)
@given(
    owner=st.integers(min_value=1, max_value=99),
    note=st.integers(min_value=1, max_value=99),
    marker=st.sampled_from(_MARKERS),
)
def test_figure_before_the_parenthesis_owns_the_marker(owner, note, marker):
    # Equal figures would ground the owner's own rate.
    assume(note != owner)
    source_text = f"le taux de {owner} ({note}) {marker} du montant"

    assert not _grounds(source_text, note / 100), source_text


@settings(max_examples=100, deadline=None)
@given(
    figure=_printed_figures(styles=_DECIMAL_COMMA_STYLES),
    marker=st.sampled_from(("Prozent", "vom Hundert", "v. H.", "%")),
)
def test_german_profile_marked_figure_grounds_its_rate(figure, marker):
    printed, value = figure
    source_text = f"Der Satz beträgt {printed} {marker} des Betrags."

    assert _grounds(source_text, value / 100, profile="de-DE"), source_text
