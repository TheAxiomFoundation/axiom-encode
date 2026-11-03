"""An own-section caption's identifier is a locator, never a dollar bound (#1827).

Composition renders a body-bearing row's heading before its body, so
A.R.S. § 43-1072 resolves to "43-1072 - Earned credit for property taxes; ..."
followed by the statute's own caption line. The completeness gate demanded 43
and 1072 as values, read the identifier's hyphen as a subtraction, and asked
for companion tests of that "computation".

The identifier stops being demanded only on evidence beyond its position and a
matching heading: the resolver rendered it as the first body-bearing row's own
heading, that row lies in the requested section, the identifier is the
section's own citation leaf, and nothing in the source points back at printed
material. The negative controls come first. They include the issue's generic
titles with "bounds printed above" and the bodyless-parent case, which must
keep every demand.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.corpus_resolver import (
    RenderedRowHeading,
    resolve_local_corpus_source,
    scope_resolved_corpus_source,
)
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    ValidatorPipeline,
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)
from tests.test_corpus_resolver import _release, _write_rows, _write_selector

SECTION = "us/statute/43-1072"
CAPTION = "43-1072 - Earned credit for property taxes"
BODY_CAPTION = "43-1072 . Earned credit for property taxes"
OPERATIVE = (
    "A. A resident whose household income is less than $3,751 is allowed a "
    "credit of $502."
)
BOUNDS_PRINTED_ABOVE = (
    "Monthly household income must fall within the lower and upper bounds "
    "printed above, expressed in dollars."
)
EMPTY_MODULE = "format: rulespec/v1\nmodule: {}\nrules: []\n"
RECALL_VALUE = re.compile(r"numeric-recall\] Authoritative corpus numeric value (\S+) ")
FIXTURES = Path(__file__).parent / "fixtures/source_completeness"


def _bind(tmp_path: Path, rows):
    version = "2026-07-13"
    _write_selector(
        tmp_path,
        [{"jurisdiction": "us", "document_class": "statute", "version": version}],
    )
    _write_rows(tmp_path, version, rows)
    return _release(tmp_path)


def _composed(tmp_path, *, parent_heading, child_heading, child_body):
    release = _bind(
        tmp_path,
        [
            {"citation_path": SECTION, "body": None, "heading": parent_heading},
            {
                "citation_path": f"{SECTION}/block-1",
                "body": child_body,
                "heading": child_heading,
                "ordinal": 1,
            },
        ],
    )
    return release, resolve_local_corpus_source(SECTION, release)


def _issues(content, source, headings=(), *, citation=SECTION, tests=()):
    return sc.analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=citation,
        test_cases=list(tests),
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
        rendered_row_headings=headings,
    ).issues


def _demands(issues) -> set[str]:
    return {match.group(1) for issue in issues if (match := RECALL_VALUE.search(issue))}


def _resolved_demands(resolved) -> set[str]:
    return _demands(
        _issues(EMPTY_MODULE, resolved.body, resolved.rendered_row_headings)
    )


# --- Resolver evidence --------------------------------------------------------


def test_composition_renders_the_body_rows_own_heading(tmp_path):
    _release_, resolved = _composed(
        tmp_path,
        parent_heading=CAPTION,
        child_heading=CAPTION,
        child_body=BODY_CAPTION + "\n\n" + OPERATIVE,
    )
    assert resolved.body == f"{CAPTION}\n\n{BODY_CAPTION}\n\n{OPERATIVE}"
    (heading,) = resolved.rendered_row_headings
    assert heading == RenderedRowHeading(
        citation_path=f"{SECTION}/block-1",
        heading=CAPTION,
        start=0,
        end=len(CAPTION),
        chunk_end=len(resolved.body),
    )
    assert resolved.body[heading.start : heading.end] == CAPTION


def test_rendered_headings_leave_text_hashes_and_attestation_unchanged(tmp_path):
    _release_, resolved = _composed(
        tmp_path,
        parent_heading=CAPTION,
        child_heading=CAPTION,
        child_body=BODY_CAPTION + "\n\n" + OPERATIVE,
    )
    attestation = resolved.to_attestation()
    assert "rendered_row_headings" not in json.dumps(attestation)
    assert (
        attestation["resolved_text_sha256"]
        == hashlib.sha256(resolved.body.encode()).hexdigest()
    )
    assert resolved.proof_evidence_segments[0] == resolved.body
    assert "rendered_row_headings" not in repr(resolved)


def test_bodyless_parent_heading_is_never_rendered(tmp_path):
    _release_, resolved = _composed(
        tmp_path,
        parent_heading=CAPTION,
        child_heading=None,
        child_body=CAPTION + "\n" + OPERATIVE,
    )
    assert resolved.body == CAPTION + "\n" + OPERATIVE
    assert resolved.rendered_row_headings == ()


def test_a_leaf_row_renders_no_heading(tmp_path):
    release = _bind(
        tmp_path,
        [{"citation_path": SECTION, "body": OPERATIVE, "heading": CAPTION}],
    )
    resolved = resolve_local_corpus_source(SECTION, release)
    assert resolved.body == OPERATIVE
    assert resolved.rendered_row_headings == ()


def test_a_slice_drops_the_heading_offsets(tmp_path):
    release, resolved = _composed(
        tmp_path,
        parent_heading=CAPTION,
        child_heading=CAPTION,
        child_body=BODY_CAPTION + "\n\n(a) First.\n(e) Target $198.\n(f) Sibling $999.",
    )
    assert resolved.rendered_row_headings
    unchanged = scope_resolved_corpus_source(resolved, SECTION)
    assert unchanged.rendered_row_headings == resolved.rendered_row_headings
    scoped = scope_resolved_corpus_source(resolved, f"{SECTION}/e")
    assert scoped.body == "(e) Target $198."
    assert scoped.rendered_row_headings == ()
    fallback = resolve_local_corpus_source(f"{SECTION}/e", release)
    assert fallback.slice_required and fallback.body == "(e) Target $198."
    assert fallback.rendered_row_headings == ()


# --- Negative controls: every demand stays ---------------------------------------


@pytest.mark.parametrize(
    "title", ("Definitions", "General", "Reserved", "Eligible Household Incomes")
)
def test_generic_titles_with_bounds_printed_above_keep_their_values(tmp_path, title):
    caption = f"43-1072 - {title}"
    _release_, resolved = _composed(
        tmp_path,
        parent_heading=caption,
        child_heading=caption,
        child_body=caption + "\n" + BOUNDS_PRINTED_ABOVE,
    )
    assert resolved.rendered_row_headings
    assert {"43", "1072"} <= _resolved_demands(resolved)


@pytest.mark.parametrize(
    "title", ("Definitions", "General", "Reserved", "Eligible Household Incomes")
)
def test_bodyless_parent_heading_cannot_excuse_a_childs_opening(tmp_path, title):
    caption = f"43-1072 - {title}"
    _release_, resolved = _composed(
        tmp_path,
        parent_heading=caption,
        child_heading=None,
        child_body=caption + "\n" + BOUNDS_PRINTED_ABOVE,
    )
    assert {"43", "1072"} <= _resolved_demands(resolved)


@pytest.mark.parametrize(
    "sentence",
    (
        BOUNDS_PRINTED_ABOVE,
        "Income must be between the amounts in the heading.",
        "Income must be between the figures in the caption.",
        "Income must fall within the range at the top of this section.",
        "Income must not exceed the first line's larger figure.",
        "Income must be at least the section number's first part.",
        "Income must be within the foregoing bounds.",
        "Income must not exceed the aforesaid amounts.",
        "Income must fall within the bounds stated earlier.",
        "Income must not exceed the amounts previously stated.",
        "Income must fall within the preceding bounds.",
        "Income must fall within the bounds that open this section.",
        "Income must not exceed the title of this section's second number.",
    ),
)
def test_a_back_reference_keeps_the_caption_values(tmp_path, sentence):
    _release_, resolved = _composed(
        tmp_path,
        parent_heading=CAPTION,
        child_heading=CAPTION,
        child_body=BODY_CAPTION + "\n\n" + sentence,
    )
    assert {"43", "1072"} <= _resolved_demands(resolved)


def test_another_sections_identifier_keeps_its_values(tmp_path):
    caption = "43-1073 - Earned credit for property taxes"
    _release_, resolved = _composed(
        tmp_path,
        parent_heading=caption,
        child_heading=caption,
        child_body="43-1073 . Earned credit for property taxes\n\n" + OPERATIVE,
    )
    assert {"43", "1073"} <= _resolved_demands(resolved)


def test_unverifiable_heading_evidence_keeps_every_demand(tmp_path):
    _release_, resolved = _composed(
        tmp_path,
        parent_heading=CAPTION,
        child_heading=CAPTION,
        child_body=BODY_CAPTION + "\n\n" + OPERATIVE,
    )
    (heading,) = resolved.rendered_row_headings
    forged = (
        # Another section's row, a heading that is not at the start, stale
        # offsets, and a heading the text does not contain.
        RenderedRowHeading(
            "us/statute/43-1073/block-1", CAPTION, 0, heading.end, heading.chunk_end
        ),
        RenderedRowHeading(
            heading.citation_path, CAPTION, 2, heading.end + 2, heading.chunk_end
        ),
        RenderedRowHeading(
            heading.citation_path, CAPTION, 0, heading.end - 1, heading.chunk_end
        ),
        RenderedRowHeading(
            heading.citation_path, "43-1072 - Other", 0, 15, heading.chunk_end
        ),
    )
    for evidence in forged:
        assert {"43", "1072"} <= _demands(
            _issues(EMPTY_MODULE, resolved.body, (evidence,))
        ), evidence
    # Without the resolver's evidence the text alone proves nothing.
    assert {"43", "1072"} <= _demands(_issues(EMPTY_MODULE, resolved.body))


@pytest.mark.parametrize(
    ("operative", "kept"),
    (
        ("A credit of $43 is allowed.", {"43"}),
        ("Income below 1072 dollars qualifies.", {"1072"}),
        ("The credit is $43 when income is under $1,072.", {"43", "1072"}),
    ),
)
def test_the_same_numbers_in_operative_positions_stay_demanded(
    tmp_path, operative, kept
):
    _release_, resolved = _composed(
        tmp_path,
        parent_heading=CAPTION,
        child_heading=CAPTION,
        child_body=BODY_CAPTION + "\n\n" + operative,
    )
    assert kept <= _resolved_demands(resolved)


def test_a_computation_the_caption_states_stays_demanded(tmp_path):
    """Only the identifier's hyphen stops reading as arithmetic."""

    caption = "43-1072 - Credit equal to 10 percent of rent"
    _release_, resolved = _composed(
        tmp_path,
        parent_heading=caption,
        child_heading=caption,
        child_body="43-1072 . Credit equal to 10 percent of rent\n\n" + OPERATIVE,
    )
    issues = _issues(EMPTY_MODULE, resolved.body, resolved.rendered_row_headings)
    assert not {"43", "1072"} & _demands(issues)
    caption_outputs = [
        issue
        for issue in issues
        if issue.startswith("[complete-source-unit:formula-output]")
        and "10 percent of rent" in issue
    ]
    assert caption_outputs
    assert caption_outputs == [
        issue
        for issue in _issues(EMPTY_MODULE, resolved.body)
        if issue.startswith("[complete-source-unit:formula-output]")
        and "10 percent of rent" in issue
    ]


# --- Genuine locator captions --------------------------------------------------


def test_a_genuine_own_section_caption_is_a_locator(tmp_path):
    _release_, resolved = _composed(
        tmp_path,
        parent_heading=CAPTION,
        child_heading=CAPTION,
        child_body=BODY_CAPTION + "\n\n" + OPERATIVE,
    )
    issues = _issues(EMPTY_MODULE, resolved.body, resolved.rendered_row_headings)
    assert not {"43", "1072"} & _demands(issues)
    # The operative values in the same body stay demanded.
    assert {"3751", "502"} <= _demands(issues)
    assert {"43", "1072", "3751", "502"} <= _demands(
        _issues(EMPTY_MODULE, resolved.body)
    )


def test_pipeline_passes_the_resolvers_headings(tmp_path):
    release, resolved = _composed(
        tmp_path,
        parent_heading=CAPTION,
        child_heading=CAPTION,
        child_body=BODY_CAPTION + "\n\n" + OPERATIVE,
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        local_corpus_release=release,
        enable_oracles=False,
        source_text=resolved.body,
        source_citation_path=SECTION,
        require_complete_source_unit=True,
    )
    assert pipeline._complete_source_rendered_row_headings(resolved.body) == (
        resolved.rendered_row_headings
    )
    assert pipeline._complete_source_rendered_row_headings(resolved.body + " ") == ()
    content = (
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        f"    corpus_citation_path: {SECTION}\nrules: []\n"
    )
    issues = pipeline._complete_source_unit_issues(
        content,
        validation_source_texts={SECTION: resolved.body},
        test_cases=[],
    )
    assert not {"43", "1072"} & _demands(issues)
    assert {"3751", "502"} <= _demands(issues)


def test_pipeline_keeps_the_values_beside_bounds_printed_above(tmp_path):
    caption = "43-1072 - Definitions"
    release, resolved = _composed(
        tmp_path,
        parent_heading=caption,
        child_heading=caption,
        child_body=caption + "\n" + BOUNDS_PRINTED_ABOVE,
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        local_corpus_release=release,
        enable_oracles=False,
        source_text=resolved.body,
        source_citation_path=SECTION,
        require_complete_source_unit=True,
    )
    content = (
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        f"    corpus_citation_path: {SECTION}\nrules: []\n"
    )
    issues = pipeline._complete_source_unit_issues(
        content,
        validation_source_texts={SECTION: resolved.body},
        test_cases=[],
    )
    assert {"43", "1072"} <= _demands(issues)


# --- The Arizona replay ----------------------------------------------------------


def _arizona():
    fixture = FIXTURES / "signed_reencode_37936338132"
    recorded = json.loads((fixture / "issues.json").read_text())
    source = (fixture / "source.txt").read_text()
    metadata = json.loads((fixture / "source-metadata.json").read_text())[
        "source_attestation"
    ]
    (component,) = metadata["component_rows"]
    heading, separator, body = source.partition("\n\n")
    # The committed component digest authenticates the first line as block-1's
    # rendered heading: what follows it is exactly that row's body.
    assert (
        separator
        and hashlib.sha256(body.encode()).hexdigest() == (component["body_sha256"])
    )
    rendered = RenderedRowHeading(
        component["citation_path"], heading, 0, len(heading), len(source)
    )
    content = (fixture / "candidate.yaml").read_text()
    tests = yaml.safe_load((fixture / "candidate.test.yaml").read_text())
    return recorded["citation"], source, rendered, content, tests


@pytest.mark.parametrize("profile", ("en-US", "legacy"))
def test_arizona_replay_drops_only_the_caption_diagnostics(profile):
    citation, source, rendered, content, tests = _arizona()

    def run(headings):
        return sc.analyze_complete_source_unit(
            content,
            source,
            corpus_citation_path=citation,
            test_cases=tests,
            extract_numeric_occurrences=functools.partial(
                extract_typed_numeric_inventory_occurrences_from_text, profile=profile
            ),
            extract_numeric_grounding_occurrences=functools.partial(
                extract_typed_numeric_occurrences_from_text, profile=profile
            ),
            extract_named_scalars=extract_named_scalar_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
            rendered_row_headings=headings,
        ).issues

    without, with_caption = run(()), run((rendered,))
    removed = sorted(set(without) - set(with_caption))
    assert set(with_caption) <= set(without)
    assert _demands(removed) == {"43", "1072"}
    caption_clause = "`43-1072 - Earned credit for property taxes;`"
    others = [issue for issue in removed if not RECALL_VALUE.search(issue)]
    assert len(others) == 2
    assert others[0].startswith("[complete-source-unit:formula-output]")
    assert others[1].startswith("[complete-source-unit:tests]")
    assert all(caption_clause in issue for issue in others)
    # Genuine omissions stay rejected.
    assert {"1750", "1751", "1851", "1951"} <= _demands(with_caption)
    assert any("formula-output" in issue for issue in with_caption)


# --- Properties and bounds -------------------------------------------------------


@given(
    title=st.sampled_from(
        (
            "Earned credit for property taxes",
            "Definitions",
            "General",
            "Reserved",
            "Eligible Household Incomes",
        )
    ),
    operative_43=st.booleans(),
    operative_1072=st.booleans(),
    back_reference=st.sampled_from(("", BOUNDS_PRINTED_ABOVE, "See the heading.")),
    evidence=st.sampled_from(("rendered", "none", "other-section")),
)
@example(
    title="Definitions",
    operative_43=False,
    operative_1072=False,
    back_reference=BOUNDS_PRINTED_ABOVE,
    evidence="rendered",
)
@settings(max_examples=120, deadline=None)
def test_caption_numbers_are_demanded_unless_proven_locators(
    title, operative_43, operative_1072, back_reference, evidence
):
    caption = f"43-1072 - {title}"
    sentences = [
        sentence
        for sentence in (
            "A credit of $43 is allowed." if operative_43 else "",
            "Income below 1072 dollars qualifies." if operative_1072 else "",
            back_reference,
            "A credit of $502 is allowed.",
        )
        if sentence
    ]
    body = f"43-1072 . {title}\n\n" + " ".join(sentences)
    source = f"{caption}\n\n{body}"
    headings = {
        "rendered": (
            RenderedRowHeading(
                f"{SECTION}/block-1", caption, 0, len(caption), len(source)
            ),
        ),
        "none": (),
        "other-section": (
            RenderedRowHeading(
                "us/statute/43-1073/block-1", caption, 0, len(caption), len(source)
            ),
        ),
    }[evidence]
    demands = _demands(_issues(EMPTY_MODULE, source, headings))
    proven = evidence == "rendered" and not back_reference
    assert ("43" in demands) == (operative_43 or not proven)
    assert ("1072" in demands) == (operative_1072 or not proven)
    assert "502" in demands


class _CountingPattern:
    def __init__(self, pattern, meter, key):
        self._pattern = pattern
        self._meter = meter
        self._key = key

    def search(self, string, *args):
        self._meter[self._key] += 1
        return self._pattern.search(string, *args)

    def fullmatch(self, string, *args):
        self._meter[self._key] += 1
        return self._pattern.fullmatch(string, *args)


@pytest.mark.parametrize("rows", (1, 16, 256))
def test_caption_check_does_constant_work_per_analysis(monkeypatch, rows):
    meter = {"back_reference": 0, "caption": 0}
    monkeypatch.setattr(
        sc,
        "_OWN_SECTION_CAPTION_BACK_REFERENCE",
        _CountingPattern(
            sc._OWN_SECTION_CAPTION_BACK_REFERENCE, meter, "back_reference"
        ),
    )
    monkeypatch.setattr(
        sc,
        "_OWN_SECTION_CAPTION",
        _CountingPattern(sc._OWN_SECTION_CAPTION, meter, "caption"),
    )
    chunks = [f"{CAPTION}\n\n{BODY_CAPTION}\n\n{OPERATIVE}"] * rows
    source = "\n\n".join(chunks)
    chunk_length = len(chunks[0])
    headings = tuple(
        RenderedRowHeading(
            f"{SECTION}/block-{index}",
            CAPTION,
            index * (chunk_length + 2),
            index * (chunk_length + 2) + len(CAPTION),
            index * (chunk_length + 2) + chunk_length,
        )
        for index in range(rows)
    )
    captions = sc._own_section_captions(source, headings, corpus_citation_path=SECTION)
    assert len(captions) == 2
    # One scan of the text for back-references and two anchored caption
    # matches, however many rows were rendered.
    assert meter == {"back_reference": 1, "caption": 2}


def _authenticated_first_heading(fixture: Path, source: str):
    """The first line, if the committed component digest proves it rendered."""

    metadata = json.loads((fixture / "source-metadata.json").read_text())[
        "source_attestation"
    ]
    components = metadata.get("component_rows") or []
    heading, separator, body = source.partition("\n\n")
    if not (
        len(components) == 1
        and separator
        and hashlib.sha256(body.encode()).hexdigest() == components[0]["body_sha256"]
    ):
        return None
    return RenderedRowHeading(
        components[0]["citation_path"], heading, 0, len(heading), len(source)
    )


def test_replay_fixture_differential_names_every_change():
    """Over every signed replay fixture, only Arizona's caption diagnostics go.

    Without rendered headings the gate is main's by construction (no caption
    can be verified). A fixture can change only where the committed component
    digest proves its first line is a rendered heading; Virginia's opens with
    `§ 58.1-322.03 .`, which is no bare own-section identifier.
    """

    changed = {}
    authenticated = []
    for fixture in sorted(FIXTURES.glob("signed_reencode_*")):
        recorded = json.loads((fixture / "issues.json").read_text())
        source = (fixture / "source.txt").read_text()
        rendered = _authenticated_first_heading(fixture, source)
        if rendered is None:
            continue
        authenticated.append(fixture.name)
        content = (fixture / "candidate.yaml").read_text()
        tests = yaml.safe_load((fixture / "candidate.test.yaml").read_text())

        def run(headings, content=content, source=source, tests=tests):
            return sc.analyze_complete_source_unit(
                content,
                source,
                corpus_citation_path=recorded["citation"],
                test_cases=tests,
                extract_numeric_occurrences=functools.partial(
                    extract_typed_numeric_inventory_occurrences_from_text,
                    profile="en-US",
                ),
                extract_numeric_grounding_occurrences=functools.partial(
                    extract_typed_numeric_occurrences_from_text, profile="en-US"
                ),
                extract_named_scalars=extract_named_scalar_occurrences,
                numeric_value_is_grounded=numeric_value_is_grounded,
                rendered_row_headings=headings,
            ).issues

        without, with_heading = set(run(())), set(run((rendered,)))
        assert with_heading <= without
        if without != with_heading:
            changed[fixture.name] = sorted(
                issue.split("]", 1)[0] + "]" for issue in without - with_heading
            )
    assert authenticated == [
        "signed_reencode_37936338132",
        "signed_reencode_37936341740",
    ]
    assert changed == {
        "signed_reencode_37936338132": [
            "[complete-source-unit:formula-output]",
            "[complete-source-unit:numeric-recall]",
            "[complete-source-unit:numeric-recall]",
            "[complete-source-unit:tests]",
        ]
    }


@pytest.mark.parametrize(
    ("first_line", "expected"),
    (
        (CAPTION, True),
        (BODY_CAPTION, True),
        ("43-1073 - Earned credit for property taxes", False),
        ("§ 43-1072 . Earned credit", False),
        ("A. A resident is allowed a credit.", False),
    ),
)
def test_pipeline_resolves_headings_only_for_a_caption_shaped_opening(
    first_line, expected
):
    source = first_line + "\n\n" + OPERATIVE
    assert (
        sc.may_open_with_own_section_caption(source, corpus_citation_path=SECTION)
        is expected
    )
