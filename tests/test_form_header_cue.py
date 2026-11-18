"""A physical confidentiality label owns one cue, never its surrounding prose."""

import pytest

from axiom_encode.bound_form_structure import printed_form_header_condition_span
from axiom_encode.harness import source_completeness as sc
from tests.test_bound_form_operands import SOURCE, issues


def cues(text, raw=None):
    return sc._source_exception_or_applicability_matches(text, raw_source_text=raw)


def branches(text):
    return sc._source_exception_branches(
        text,
        branches=(),
        active_branches=(),
        deferred_paths=set(),
        raw_source_text=text,
    )


def test_full_analyzer_removes_only_header_obligation(monkeypatch):
    actual = issues()
    after = branches(SOURCE)
    with monkeypatch.context() as patch:
        patch.setattr(sc, "printed_form_header_condition_span", lambda source: None)
        prior = issues()
        before = branches(SOURCE)
    assert len(before) == 4 and len(after) == 3
    assert after == before[1:]
    assert prior[-1].count("[change]:") == 4
    assert actual[-1].count("[change]:") == 3
    assert actual[:-1] == prior[:-1]


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("bom", ["", "\ufeff"])
def test_exact_header_and_outer_strip_offsets(newline, bom):
    text = bom + SOURCE.lstrip("\ufeff").replace("\n", newline)
    raw = text + " \n"
    span = printed_form_header_condition_span(raw)
    assert span is not None and raw[slice(*span)] == "when"
    assert all(m.span() != span for m in cues(raw))
    assert len(cues(raw.strip(), raw)) == 3


@pytest.mark.parametrize(
    "replacement",
    [
        "Protected B when completed if eligible",
        "Protected B when completed less assistance",
        "Protected B when completed plus a recovery amount",
        "Protected B when completed + 500",
        "Protected B when filed",
        "Protected B when completed\ncontinued",
    ],
)
def test_nonclosed_confidentiality_line_retains_when(replacement):
    text = SOURCE.replace("Protected B when completed", replacement)
    assert printed_form_header_condition_span(text) is None
    assert any(m.group() == "when" for m in cues(text))
    assert "Protected B when" in issues(source=text)[-1]


@pytest.mark.parametrize(
    "text",
    [
        SOURCE[:60],
        SOURCE.replace("2025\n\nForm", "2025\nForm"),
        SOURCE.split("You can claim", 1)[1] + " When completed, claim 500.",
        " " + SOURCE,
    ],
)
def test_clipped_body_or_nonanchored_header_is_not_owned(text):
    assert printed_form_header_condition_span(text) is None
    assert any(m.group().lower() == "when" for m in cues(text))


@pytest.mark.parametrize("placement", ["title", "same_clause", "body"])
def test_other_conditions_survive_even_in_header_clause(placement):
    if placement == "title":
        text = SOURCE.replace(
            "Newfoundland and Labrador Credits", "Credits if eligible"
        )
    elif placement == "same_clause":
        text = SOURCE.replace("You can claim", "If eligible, you can claim", 1)
    else:
        text = SOURCE + "\nWhen eligible, claim the credit."
    assert printed_form_header_condition_span(text) is not None
    assert len(branches(text)) == 4
    assert issues(source=text)[-1].count("[change]:") == 4


def test_raw_alignment_refuses_interior_normalization_and_collapsed_header():
    collapsed = " ".join(SOURCE.split())
    assert any(m.group() == "when" for m in cues(collapsed, SOURCE))
    changed = SOURCE.replace("2025", "2024")
    assert any(m.group() == "when" for m in cues(changed, SOURCE))
    assert cues("You can claim even if no tax is due.") == ()
