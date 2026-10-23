"""Principal requests retain authenticated composed amendment evidence."""

import pytest

from axiom_encode.harness import evals
from tests.test_amending_section_context import (
    _AMENDING_SECTION,
    _AMENDMENT_BODY,
    _COMMENCEMENT_BODY,
    _COMMENCEMENT_SECTION,
    _PRINCIPAL_TARGET,
    _amendment_rows,
)
from tests.test_evals import (
    _workspace_prompt_for_source_unit,
    _write_test_corpus_release,
)


@pytest.mark.parametrize("composed_commencement", [False, True])
def test_principal_request_retains_bodyless_amendment_composed_evidence(
    tmp_path, composed_commencement
):
    rows = _amendment_rows()
    rows[1]["body"] = None
    rows.append(
        {
            "citation_path": f"{_AMENDING_SECTION}/1",
            "source_path": rows[1]["source_path"],
            "heading": "New maximum rate",
            "body": _AMENDMENT_BODY,
            "metadata": {
                "commencement_date": "2099-01-01",
                "amendment_target": "nz/statute/unrelated/section/1",
            },
        }
    )
    if composed_commencement:
        rows[2]["body"] = None
        rows.append(
            {
                "citation_path": f"{_COMMENCEMENT_SECTION}/1",
                "source_path": rows[2]["source_path"],
                "heading": "Applicable dates",
                "body": _COMMENCEMENT_BODY,
                "metadata": {"title": "Child metadata must not replace the container"},
            }
        )
    release = _write_test_corpus_release(tmp_path, rows)

    source = evals.resolve_corpus_source_unit(_PRINCIPAL_TARGET, release)

    assert source.citation_path == source.requested == _PRINCIPAL_TARGET
    assert source.amendment_request is None
    evidence = {item.citation_path: item for item in source.amendment_documents}
    assert {_AMENDING_SECTION, _COMMENCEMENT_SECTION} <= evidence.keys()
    assert all(item.body.strip() for item in evidence.values())
    assert "New maximum rate\n\n" + _AMENDMENT_BODY in evidence[_AMENDING_SECTION].body
    assert evidence[_AMENDING_SECTION].metadata["amendment_target"] == _PRINCIPAL_TARGET
    assert evidence[_AMENDING_SECTION].metadata["commencement_date"] == "2027-04-01"
    assert _COMMENCEMENT_BODY in evidence[_COMMENCEMENT_SECTION].body
    assert evidence[_COMMENCEMENT_SECTION].metadata == {
        "title": rows[2]["metadata"]["title"],
        "expression_date": "2026-01-01",
        "source_as_of": "2026-01-01",
    }
    if not composed_commencement:
        direct = evals.resolve_corpus_source_unit(_AMENDING_SECTION, release)
        assert direct.amendment_documents == source.amendment_documents
    _, prompt = _workspace_prompt_for_source_unit(tmp_path, source)
    assert _AMENDMENT_BODY in prompt
    assert _COMMENCEMENT_BODY in prompt
    assert "effective_from: '0001-01-01'" not in prompt


@pytest.mark.parametrize("missing_evidence", ["amendment", "commencement"])
def test_principal_request_blocks_unresolvable_bodyless_evidence(
    tmp_path, missing_evidence
):
    rows = _amendment_rows()
    missing_index = 1 if missing_evidence == "amendment" else 2
    rows[missing_index]["body"] = None
    # A same-document sibling cannot supply the missing container's legal text
    # or replace its own routing metadata.
    rows[3]["metadata"] = dict(rows[1]["metadata"])
    release = _write_test_corpus_release(tmp_path, rows)

    with pytest.raises(
        ValueError,
        match="Unresolved amendment context.*authenticated.*evidence",
    ):
        evals.resolve_corpus_source_unit(_PRINCIPAL_TARGET, release)


def test_principal_request_ignores_unrelated_bodyless_amendment_container(tmp_path):
    rows = _amendment_rows()
    rows.append(
        {
            "citation_path": "nz/statute/unrelated-amendment/section/1",
            "source_path": "sources/nz/statute/unrelated.xml",
            "body": None,
            "metadata": {
                "document_type": "amendment act",
                "amendment_target": "nz/statute/unrelated/section/1",
            },
        }
    )
    release = _write_test_corpus_release(tmp_path, rows)

    source = evals.resolve_corpus_source_unit(_PRINCIPAL_TARGET, release)

    assert {item.citation_path for item in source.amendment_documents} == {
        _AMENDING_SECTION,
        _COMMENCEMENT_SECTION,
    }
