"""Closed source display titles preserve physical offsets and unowned prose."""

import hashlib

import pytest

from axiom_encode.bound_form_structure import (
    printed_form_header_condition_span,
    recognize_bound_form_structure,
)
from tests.test_bound_form_operands import CITATION, SOURCE

PREFIX = "CRA Form NL479 2025 Newfoundland and Labrador Credits\n\n"


def test_title_wrapper_preserves_all_raw_offsets_and_hash():
    bare = recognize_bound_form_structure(SOURCE, CITATION)
    text = PREFIX + SOURCE
    wrapped = recognize_bound_form_structure(text, CITATION)
    assert bare is not None and wrapped is not None
    shift = len(PREFIX)
    assert wrapped.source == text
    assert wrapped.source_sha256 == hashlib.sha256(text.encode()).hexdigest()
    assert wrapped.header.span == tuple(n + shift for n in bare.header.span)
    assert wrapped.header.year_span == tuple(n + shift for n in bare.header.year_span)
    assert wrapped.rows == tuple(
        type(row)(
            row.number,
            row.field_number,
            tuple(n + shift for n in row.span),
            tuple(n + shift for n in row.caption_span),
        )
        for row in bare.rows
    )
    assert (0, shift) in wrapped.unowned_spans
    assert printed_form_header_condition_span(text) == tuple(
        n + shift for n in printed_form_header_condition_span(SOURCE)
    )
    assert wrapped.bound(text, CITATION)
    assert not wrapped.bound(text.replace("2025", "2024", 1), CITATION)


@pytest.mark.parametrize(
    "prefix",
    [
        PREFIX.replace("2025", "2024"),
        PREFIX.replace("NL479", "NL478"),
        PREFIX.replace("Credits", "Tax"),
        PREFIX + PREFIX,
        PREFIX.replace("\n\n", "\n"),
        PREFIX.replace("CRA", "Other Publisher"),
        "If eligible " + PREFIX,
        PREFIX.rstrip() + " less assistance\n\n",
        " " + PREFIX,
    ],
)
def test_unknown_or_conflicting_wrapper_refuses(prefix):
    text = prefix + SOURCE
    assert recognize_bound_form_structure(text, CITATION) is None
    assert printed_form_header_condition_span(text) is None


@pytest.mark.parametrize(
    "tail", [" if eligible", " when completed", " less assistance", " plus recovery"]
)
def test_coherently_modified_operative_titles_refuse(tail):
    text = (PREFIX + SOURCE).replace(
        "Newfoundland and Labrador Credits", "Newfoundland and Labrador Credits" + tail
    )
    assert recognize_bound_form_structure(text, CITATION) is None
    assert printed_form_header_condition_span(text) is None


def test_following_condition_is_not_owned():
    text = PREFIX + SOURCE + "\nIf eligible, claim another amount.\n"
    form = recognize_bound_form_structure(text, CITATION)
    assert form is not None
    assert any("If eligible" in text[slice(*span)] for span in form.unowned_spans)


def test_wrapped_year_binds_only_recognized_occurrences():
    import copy
    from dataclasses import replace

    import yaml

    from axiom_encode.harness.bound_form_operands import bind_ordered_form
    from tests.test_bound_form_operands import PAYLOAD, ROOT

    text = PREFIX + SOURCE
    payload = copy.deepcopy(PAYLOAD)
    payload["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        text.encode()
    ).hexdigest()
    content = yaml.safe_dump(payload, sort_keys=False)
    form = recognize_bound_form_structure(text, CITATION)
    assert bind_ordered_form(form, content, {CITATION: text}, root_relative=ROOT)
    forged = replace(form, header=replace(form.header, wrapper_year_span=(0, 4)))
    assert (
        bind_ordered_form(forged, content, {CITATION: text}, root_relative=ROOT) is None
    )
    for tail in ("\nIf year is 2025, claim more.\n", "\n120250\n"):
        altered = text + tail
        payload["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
            altered.encode()
        ).hexdigest()
        altered_form = recognize_bound_form_structure(altered, CITATION)
        assert altered_form is not None
        assert (
            bind_ordered_form(
                altered_form,
                yaml.safe_dump(payload, sort_keys=False),
                {CITATION: altered},
                root_relative=ROOT,
            )
            is None
        )


@pytest.mark.parametrize(
    "path,kind,text",
    [
        ("versions[0].formula", "formula", "2025"),
        ("versions[1].effective_from", "effective_period", "2025"),
        ("versions[0].effective_from", "formula", "2025"),
        ("versions[0].effective_from", "effective_period", "Credits"),
    ],
)
def test_duplicate_exception_is_only_exact_period_path_kind_and_year(path, kind, text):
    from axiom_encode.harness.bound_form_operands import _proof

    source = PREFIX + SOURCE
    form = recognize_bound_form_structure(source, CITATION)
    rule = {
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": path,
                        "kind": kind,
                        "source": {"corpus_citation_path": CITATION, "excerpt": text},
                    }
                ]
            }
        }
    }
    assert not _proof(rule, path, kind, text, form)


def test_crlf_wrapper_preserves_physical_offsets():
    text = (PREFIX + SOURCE).replace("\n", "\r\n")
    form = recognize_bound_form_structure(text, CITATION)
    assert form is not None
    assert form.header.span[0] == len(PREFIX.replace("\n", "\r\n"))
    assert text[slice(*form.header.wrapper_year_span)] == "2025"
    assert text[slice(*form.header.year_span)] == "2025"
    assert text[slice(*printed_form_header_condition_span(text))] == "when"
    assert (0, form.header.span[0]) in form.unowned_spans
