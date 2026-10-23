"""Direct amendment requests must use principal identities and legal dates."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import evals
from tests.eval_evidence_fixtures import install_test_eval_evidence_keys
from tests.test_evals import (
    _canonical_rulespec_content_root,
    _workspace_prompt_for_source_unit,
    _write_test_corpus_release,
)

_AMENDING_ACT = "nz/statute/act/public/2026/0054"
_AMENDING_SECTION = f"{_AMENDING_ACT}/section/15"
_COMMENCEMENT_SECTION = f"{_AMENDING_ACT}/section/2"
_PRINCIPAL_TARGET = "nz/statute/act/public/2018/0032/schedule/4/part/7/clause/4"
_OTHER_TARGET = "nz/statute/act/public/2018/0032/schedule/4/part/7/clause/5"
_TITLE = (
    "Social Security (Jobseeker Support and Accommodation Supplement) "
    "Amendment Act 2026"
)
_COMMENCEMENT_BODY = (
    "1\nThis Act comes into force on 2 November 2026.\n"
    "2\nHowever, Part 2 comes into force on 1 April 2027."
)
# Synthetic fixture containing a short legal excerpt; the original diagnosis's
# authenticated source and generated outputs are not supplied by this checkout.
_AMENDMENT_BODY = (
    "In Schedule 4, Part 7, replace clauses 4 to 6A with:\n"
    "4 | To a person (other than a specified person) who has 1 or more "
    "dependent children and who is in a relationship | 70% of the amount by "
    "which an applicant's weekly qualifying accommodation costs exceeds 40% "
    "of the base rate, but not more than $305 a week, if the applicant "
    "resides in Area 1."
)


def _amendment_rows(*, metadata=None):
    """Build synthetic principal, amendment, and commencement corpus records."""

    amendment_metadata = {
        "title": _TITLE,
        "document_type": "amendment act",
        "amendment_target": _PRINCIPAL_TARGET,
        "commencement_date": "2027-04-01",
        "commencement_citation_path": _COMMENCEMENT_SECTION,
    }
    if metadata is not None:
        amendment_metadata = metadata
    amendment_source = "sources/nz/statute/2026-54.xml"
    return [
        {
            "citation_path": _PRINCIPAL_TARGET,
            "source_path": "sources/nz/statute/2018-32.xml",
            "body": "The accommodation supplement is 70% of qualifying costs.",
            "metadata": {"title": "Social Security Act 2018"},
        },
        {
            "citation_path": _AMENDING_SECTION,
            "source_path": amendment_source,
            "heading": "Schedule 4 amended",
            "body": _AMENDMENT_BODY,
            "metadata": amendment_metadata,
        },
        {
            "citation_path": _COMMENCEMENT_SECTION,
            "source_path": amendment_source,
            "heading": "Commencement",
            "body": _COMMENCEMENT_BODY,
            "metadata": {"title": _TITLE},
        },
        {
            "citation_path": f"{_AMENDING_ACT}/section/3",
            "source_path": amendment_source,
            "heading": "Principal Act",
            "body": "This Act amends the Social Security Act 2018 .",
            "metadata": {"title": _TITLE},
        },
    ]


def test_amending_section_routes_to_principal_with_commencement_evidence(tmp_path):
    release = _write_test_corpus_release(tmp_path, _amendment_rows())

    source = evals.resolve_corpus_source_unit(_AMENDING_SECTION, release)

    assert source.amendment_request == _AMENDING_SECTION
    assert source.requested == source.citation_path == _PRINCIPAL_TARGET
    assert source.resolved_source.requested == _PRINCIPAL_TARGET
    assert source.source_attestation == source.resolved_source.to_attestation()
    assert source.body == "The accommodation supplement is 70% of qualifying costs."
    evidence = {item.citation_path: item for item in source.amendment_documents}
    assert {_AMENDING_SECTION, _COMMENCEMENT_SECTION} <= evidence.keys()
    assert evidence[_AMENDING_SECTION].body == _AMENDMENT_BODY
    assert evidence[_AMENDING_SECTION].metadata["commencement_date"] == "2027-04-01"
    assert (
        evidence[_AMENDING_SECTION].metadata["commencement_citation_path"]
        == _COMMENCEMENT_SECTION
    )
    assert evidence[_COMMENCEMENT_SECTION].body == _COMMENCEMENT_BODY

    workspace, prompt = _workspace_prompt_for_source_unit(tmp_path, source)
    assert workspace.amendment_documents
    assert "2027-04-01" in prompt
    assert _COMMENCEMENT_BODY in prompt
    assert _AMENDING_SECTION in prompt
    assert "effective-dated" in prompt
    assert "`versions`" in prompt
    assert "effective_from: '0001-01-01'" not in prompt
    assert "Temporal scaffold dates visible in copied context" not in prompt
    assert evals._collect_scaffold_dates(workspace, workspace.context_files) == []


@pytest.mark.parametrize("composed_principal", [False, True])
def test_bodyless_amending_section_uses_selected_metadata_and_composed_evidence(
    tmp_path, composed_principal
):
    rows = _amendment_rows()
    rows[1]["body"] = None
    rows.append(
        {
            "citation_path": f"{_AMENDING_SECTION}/1",
            "source_path": rows[1]["source_path"],
            "body": _AMENDMENT_BODY,
            # Child metadata must not replace the selected container's metadata.
            "metadata": {"commencement_date": "2028-01-01"},
        }
    )
    if composed_principal:
        rows[0]["body"] = None
        rows.append(
            {
                "citation_path": f"{_PRINCIPAL_TARGET}/1",
                "source_path": rows[0]["source_path"],
                "body": "The accommodation supplement is 70% of qualifying costs.",
                "metadata": {},
            }
        )
    release = _write_test_corpus_release(tmp_path, rows)

    source = evals.resolve_corpus_source_unit(_AMENDING_SECTION, release)

    assert source.citation_path == _PRINCIPAL_TARGET
    assert source.amendment_request == _AMENDING_SECTION
    evidence = {item.citation_path: item for item in source.amendment_documents}
    assert _AMENDMENT_BODY in evidence[_AMENDING_SECTION].body
    assert evidence[_AMENDING_SECTION].metadata["commencement_date"] == "2027-04-01"
    _, prompt = _workspace_prompt_for_source_unit(tmp_path, source)
    assert _AMENDMENT_BODY in prompt
    assert "effective_from: '0001-01-01'" not in prompt


def test_bodyless_amending_section_without_context_blocks_instead_of_borrowing_sibling(
    tmp_path,
):
    rows = _amendment_rows()
    rows[1]["body"] = None
    rows[1]["metadata"].pop("commencement_date")
    rows[2]["metadata"] = dict(_amendment_rows()[1]["metadata"])
    rows.append(
        {
            "citation_path": f"{_AMENDING_SECTION}/1",
            "source_path": rows[1]["source_path"],
            "body": _AMENDMENT_BODY,
        }
    )
    release = _write_test_corpus_release(tmp_path, rows)

    with pytest.raises(
        ValueError, match="Unresolved amendment context.*commencement_date"
    ):
        evals.resolve_corpus_source_unit(_AMENDING_SECTION, release)


@pytest.mark.parametrize("missing_context", [False, True])
@pytest.mark.parametrize("leading_ordinary_child", [False, True])
@pytest.mark.parametrize("child_source", ["selected", "different"])
def test_heading_prefixed_amendment_contributor_routes_or_blocks(
    tmp_path, missing_context, leading_ordinary_child, child_source
):
    rows = _amendment_rows()
    rows[1].update(body=None, heading="Accommodation supplement")
    rows[1]["metadata"].pop("document_type")
    if missing_context:
        rows[1]["metadata"] = {}
    # Complete sibling and child metadata must never repair missing context on
    # the selected container or replace its applicable commencement date.
    rows[2]["metadata"] = dict(_amendment_rows()[1]["metadata"])
    if leading_ordinary_child:
        rows.append(
            {
                "citation_path": f"{_AMENDING_SECTION}/1",
                "source_path": rows[1]["source_path"],
                "heading": "Accommodation supplement",
                "body": "This section concerns accommodation supplement rates.",
            }
        )
    child_source_path = {
        "selected": rows[1]["source_path"],
        "different": "sources/nz/statute/amendment-fragment.xml",
    }[child_source]
    rows.append(
        {
            "citation_path": f"{_AMENDING_SECTION}/2",
            "source_path": child_source_path,
            "heading": "New maximum rate",
            "body": _AMENDMENT_BODY,
            "metadata": {
                "amendment_target": _PRINCIPAL_TARGET,
                "commencement_date": "2028-01-01",
            },
        }
    )
    release = _write_test_corpus_release(tmp_path, rows)
    composed = evals._corpus_resolver.resolve_local_corpus_source(
        _AMENDING_SECTION, release
    )
    contributor = next(
        item
        for item in composed.component_rows
        if item.citation_path == f"{_AMENDING_SECTION}/2"
    )
    assert contributor.source_path == child_source_path

    if missing_context:
        with pytest.raises(
            ValueError, match="Unresolved amendment context.*metadata.amendment_target"
        ):
            evals.resolve_corpus_source_unit(_AMENDING_SECTION, release)
        return

    source = evals.resolve_corpus_source_unit(_AMENDING_SECTION, release)

    assert source.citation_path == _PRINCIPAL_TARGET
    assert source.amendment_request == _AMENDING_SECTION
    evidence = {item.citation_path: item for item in source.amendment_documents}
    assert "New maximum rate\n\n" + _AMENDMENT_BODY in evidence[_AMENDING_SECTION].body
    assert evidence[_AMENDING_SECTION].metadata["commencement_date"] == "2027-04-01"
    _, prompt = _workspace_prompt_for_source_unit(tmp_path, source)
    assert _AMENDMENT_BODY in prompt
    assert "effective_from: '0001-01-01'" not in prompt


@pytest.mark.parametrize("target_key", ["amendment_target", "amendment_targets"])
def test_retained_structured_target_classifies_without_an_operation_or_act_marker(
    tmp_path, target_key
):
    metadata = dict(_amendment_rows()[1]["metadata"])
    metadata.pop("document_type")
    metadata.pop("amendment_target")
    metadata[target_key] = (
        [_PRINCIPAL_TARGET] if target_key == "amendment_targets" else _PRINCIPAL_TARGET
    )
    rows = _amendment_rows(metadata=metadata)
    rows[1].update(
        heading="Accommodation supplement",
        body="The maximum accommodation supplement rate is $305 a week.",
    )
    release = _write_test_corpus_release(tmp_path, rows)

    source = evals.resolve_corpus_source_unit(_AMENDING_SECTION, release)

    assert source.citation_path == _PRINCIPAL_TARGET
    assert source.amendment_request == _AMENDING_SECTION
    assert _AMENDING_SECTION in {
        item.citation_path for item in source.amendment_documents
    }


@pytest.mark.parametrize("bodyless_container", [False, True])
def test_noncontributing_operative_descendant_cannot_classify_container(
    tmp_path, bodyless_container
):
    rows = _amendment_rows(metadata={})
    rows[1].update(
        heading="Accommodation supplement",
        body=None if bodyless_container else "The ordinary benefit rate is $200.",
    )
    if bodyless_container:
        rows.append(
            {
                "citation_path": f"{_AMENDING_SECTION}/1",
                "source_path": rows[1]["source_path"],
                "body": "The ordinary benefit rate is $200.",
            }
        )
    rows.append(
        {
            "citation_path": f"{_AMENDING_SECTION}/1/1",
            "source_path": rows[1]["source_path"],
            "heading": "New maximum rate",
            "body": _AMENDMENT_BODY,
            "metadata": dict(_amendment_rows()[1]["metadata"]),
        }
    )
    release = _write_test_corpus_release(tmp_path, rows)

    source = evals.resolve_corpus_source_unit(_AMENDING_SECTION, release)

    assert source.citation_path == _AMENDING_SECTION
    assert source.amendment_request is None
    assert source.body == "The ordinary benefit rate is $200."
    assert _AMENDMENT_BODY not in source.body


def test_bodyless_principal_container_does_not_borrow_marked_sibling(tmp_path):
    rows = _amendment_rows(metadata={})
    rows[1].update(body=None, heading="Rates")
    rows[2]["metadata"] = {"document_type": "amendment act"}
    rows.append(
        {
            "citation_path": f"{_AMENDING_SECTION}/1",
            "source_path": rows[1]["source_path"],
            "body": "The ordinary benefit rate is $200.",
        }
    )
    release = _write_test_corpus_release(tmp_path, rows)

    source = evals.resolve_corpus_source_unit(_AMENDING_SECTION, release)

    assert source.citation_path == _AMENDING_SECTION
    assert source.amendment_request is None
    assert "The ordinary benefit rate is $200." in source.body


@pytest.mark.parametrize("selected_note", [None, "Container's own note."])
def test_ordinary_composed_metadata_uses_only_its_contributing_child(
    tmp_path, selected_note
):
    citation = "de/statute/principal/section/5"
    source_path = "sources/de/principal.xml"
    release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": citation,
                "source_path": source_path,
                "body": None,
                "metadata": {"source_note": selected_note} if selected_note else {},
            },
            {
                "citation_path": f"{citation}/1",
                "source_path": source_path,
                "body": "The ordinary principal benefit is 200.",
                "metadata": {
                    "source_note": "Contributing child's note.",
                    "title": "Principal benefit",
                    "amendment_target": "de/statute/other/section/2",
                    "commencement_date": "2099-01-01",
                },
            },
            {
                "citation_path": "de/statute/principal/section/2",
                "source_path": source_path,
                "body": "Interpretation.",
                "metadata": {"source_note": "Sibling's unrelated note."},
            },
        ],
    )

    source = evals.resolve_corpus_source_unit(citation, release)

    assert source.amendment_request is None
    assert source.provision_metadata["source_note"] == (
        selected_note or "Contributing child's note."
    )
    assert source.provision_metadata["title"] == "Principal benefit"
    assert "amendment_target" not in source.provision_metadata
    assert "commencement_date" not in source.provision_metadata


@pytest.mark.parametrize(
    ("heading", "body"),
    [
        (
            "References to repealed enactments",
            "A reference to a repealed enactment includes its replacement.",
        ),
        (
            "Savings on repeal",
            "The repeal of section 4 does not affect an existing entitlement.",
        ),
        (
            "Transitional provisions for repealed enactments",
            "A benefit under repealed section 4 continues during the transition.",
        ),
        (
            "Meaning of enactment",
            "Enactment includes a repealed enactment and an amending enactment.",
        ),
        (
            "Interpretation of section 4",
            "Section 4 was repealed in 2005; references retain their former meaning.",
        ),
        (
            "Interpretation",
            "In section 4, a reference to repeal includes expiry.",
        ),
        (
            "Interpretation",
            "In article 12, the term substitute means a person acting temporarily.",
        ),
        (
            "Interpretation",
            "Section 4 repealed enactments are interpreted as including replacements.",
        ),
        (
            "Savings",
            "In section 4, repeal does not affect existing rights.",
        ),
        (
            "Interpretation",
            "In section 4, repeal includes expiry.",
        ),
        (
            "Savings",
            "In section 4, repeal (including expiry) does not affect existing rights.",
        ),
    ],
)
def test_nonoperative_repeal_references_keep_principal_resolution(
    tmp_path, heading, body
):
    citation = "nz/statute/act/public/2018/0032/section/5"
    release = _write_test_corpus_release(
        tmp_path,
        [{"citation_path": citation, "heading": heading, "body": body, "metadata": {}}],
    )

    source = evals.resolve_corpus_source_unit(citation, release)

    assert source.requested == source.citation_path == citation
    assert source.body == body
    assert source.amendment_request is None
    assert source.amendment_documents == ()


@pytest.mark.parametrize(
    "operation",
    [
        "Section 4 amended",
        "Article 12 replaced",
        "Paragraph 3 is repealed",
        "Clause 8 inserted",
        "New sections 23A to 23I inserted",
        "New Part 16 inserted into Schedule 1",
        "Section 5 of the Example Act is repealed",
        "In section 4(2)(a), omit the words temporary benefit.",
        "In section 4, after subsection (2), insert a new subsection.",
        "In Schedule 4, Part 7, replace clauses 4 to 6A with new clauses.",
        "Insert section 4 after section 3.",
        "For regulation 6 substitute the following regulation.",
    ],
)
def test_unmarked_named_provision_operation_requires_amendment_context(
    tmp_path, operation
):
    citation = "uk/statute/example/section/5"
    release = _write_test_corpus_release(
        tmp_path,
        [{"citation_path": citation, "heading": operation, "body": operation}],
    )

    with pytest.raises(ValueError, match="Unresolved amendment context"):
        evals.resolve_corpus_source_unit(citation, release)


@settings(max_examples=100)
@given(
    heading=st.sampled_from(
        ["References to repealed enactments", "Savings on repeal", "Definitions", None]
    ),
    subject=st.sampled_from(
        [
            "A reference to",
            "The meaning of",
            "Savings concerning",
            "Transitional treatment of",
            "In section 4, a reference to",
            "For section 4, the definition of",
            "In article 12, the term",
        ]
    ),
    nouns=st.lists(
        st.sampled_from(
            [
                "repealed enactments",
                "replacement",
                "amendments",
                "section 4",
                "article 12",
                "enactment",
                "repeal",
                "substitute",
                "replace",
                "insert",
            ]
        ),
        min_size=1,
        max_size=12,
    ),
)
def test_unmarked_nonoperative_text_is_never_an_amendment(heading, subject, nouns):
    # No structured amendment marker is present, and no generated text contains
    # an instruction applying an amendment to a named provision.
    row = SimpleNamespace(
        heading=heading,
        body=subject + " " + " and ".join(nouns) + ".",
        metadata={},
    )

    assert evals._is_direct_amendment_request(row, None) is False


@settings(max_examples=80)
@given(
    scope=st.sampled_from(["In section 4,", "For article 12,", "In Schedule 2,"]),
    noun=st.sampled_from(
        ["repeal", "substitute", "replace", "insert", "omit", "amend"]
    ),
    interpretation=st.sampled_from(
        [
            "does not affect existing rights",
            "includes expiry",
            "means an ordinary term",
            "(including expiry) does not affect existing rights",
        ]
    ),
)
def test_scoped_operation_words_used_as_nouns_are_not_amendments(
    scope, noun, interpretation
):
    row = SimpleNamespace(
        heading="Interpretation",
        body=f"{scope} {noun} {interpretation}.",
        metadata={},
    )

    assert evals._is_direct_amendment_request(row, None) is False


@pytest.mark.parametrize("jurisdiction", ["nz", "us", "uk", "de", "fr", "za"])
def test_structured_amendment_routing_is_independent_of_jurisdiction(
    tmp_path, jurisdiction
):
    rows = json.loads(json.dumps(_amendment_rows()).replace("nz/", f"{jurisdiction}/"))
    release = _write_test_corpus_release(tmp_path, rows)
    request = _AMENDING_SECTION.replace("nz/", f"{jurisdiction}/")
    principal = _PRINCIPAL_TARGET.replace("nz/", f"{jurisdiction}/")

    source = evals.resolve_corpus_source_unit(request, release)

    assert source.requested == source.citation_path == principal
    assert source.amendment_request == request
    assert source.amendment_documents[0].metadata["commencement_date"] == "2027-04-01"


def test_principal_request_discovers_amendment_and_its_commencement(tmp_path):
    release = _write_test_corpus_release(tmp_path, _amendment_rows())

    source = evals.resolve_corpus_source_unit(_PRINCIPAL_TARGET, release)

    assert source.amendment_request is None
    assert source.requested == source.citation_path == _PRINCIPAL_TARGET
    evidence = {item.citation_path: item for item in source.amendment_documents}
    assert {_AMENDING_SECTION, _COMMENCEMENT_SECTION} <= evidence.keys()
    assert evidence[_AMENDING_SECTION].metadata["commencement_date"] == "2027-04-01"
    assert evidence[_COMMENCEMENT_SECTION].body == _COMMENCEMENT_BODY
    _, prompt = _workspace_prompt_for_source_unit(tmp_path, source)
    assert "2027-04-01" in prompt
    assert "effective_from: '0001-01-01'" not in prompt


def test_direct_request_generation_uses_principal_output_and_identity(
    tmp_path, monkeypatch
):
    release = _write_test_corpus_release(tmp_path, _amendment_rows())
    install_test_eval_evidence_keys(monkeypatch)
    response = evals.EvalPromptResponse(
        text=(
            "=== FILE: 4.yaml ===\n"
            "format: rulespec/v1\n"
            "module:\n"
            "  source_verification:\n"
            f"    corpus_citation_path: {_PRINCIPAL_TARGET}\n"
            "  summary: Principal accommodation supplement provision.\n"
            "rules: []\n"
            "=== FILE: 4.test.yaml ===\n"
            "[]\n"
        ),
        duration_ms=1,
        trace={},
    )
    prompt_eval = Mock(return_value=response)
    artifact_eval = Mock(return_value=None)
    monkeypatch.setattr(evals, "_run_prompt_eval", prompt_eval)
    monkeypatch.setattr(evals, "evaluate_artifact", artifact_eval)

    [result] = evals.run_model_eval(
        citations=[_AMENDING_SECTION],
        runner_specs=["codex:gpt-5.4"],
        output_root=tmp_path / "out",
        policy_path=_canonical_rulespec_content_root(tmp_path, "nz"),
        runtime_axiom_rules_path=tmp_path / "axiom-rules-engine",
        corpus_release=release,
        mode="cold",
        include_tests=True,
        skip_reviewers=True,
    )

    assert result.success is True
    assert result.citation == _PRINCIPAL_TARGET
    expected_relative = evals._resolve_eval_output_path(_PRINCIPAL_TARGET)
    assert Path(result.output_file).as_posix().endswith(expected_relative.as_posix())
    assert not Path(result.output_file).as_posix().endswith("section/15.yaml")
    assert prompt_eval.call_count == 1
    workspace, prompt = prompt_eval.call_args.args[1:]
    assert workspace.source_text_file.read_text() == (
        "The accommodation supplement is 70% of qualifying costs."
    )
    assert (
        f"`module.source_verification.corpus_citation_path: {_PRINCIPAL_TARGET}` exactly."
        in prompt
    )
    assert (
        f"`module.source_verification.corpus_citation_path: {_AMENDING_SECTION}` exactly."
        not in prompt
    )
    assert _AMENDING_SECTION in prompt
    assert _COMMENCEMENT_BODY in prompt
    assert artifact_eval.call_args.kwargs["source_citation_path"] == _PRINCIPAL_TARGET
    attestation = result.source_attestation
    assert attestation["requested_corpus_citation_path"] == _PRINCIPAL_TARGET
    assert attestation["resolved_corpus_citation_path"] == _PRINCIPAL_TARGET


@pytest.mark.parametrize(
    "missing_fields",
    [
        ("commencement_date",),
        ("commencement_citation_path",),
        ("commencement_date", "commencement_citation_path"),
    ],
)
def test_amending_section_without_commencement_is_blocked(tmp_path, missing_fields):
    rows = _amendment_rows()
    for field in missing_fields:
        rows[1]["metadata"].pop(field)
    release = _write_test_corpus_release(tmp_path, rows)

    with pytest.raises(ValueError, match="Unresolved amendment context") as error:
        evals.resolve_corpus_source_unit(_AMENDING_SECTION, release)

    assert "commencement" in str(error.value).casefold()
    assert _AMENDING_SECTION in str(error.value)


def test_report_unmarked_amending_section_requires_explicit_context(tmp_path):
    # This synthetic unmarked record has an operation heading but no declared
    # target, Part membership, or applicable legal date.
    rows = _amendment_rows(metadata={"title": _TITLE, "assent_date": "2026-09-03"})
    release = _write_test_corpus_release(tmp_path, rows)

    with pytest.raises(ValueError, match="Unresolved amendment context") as error:
        evals.resolve_corpus_source_unit(_AMENDING_SECTION, release)

    assert "target" in str(error.value).casefold()
    assert _AMENDING_SECTION in str(error.value)


@pytest.mark.parametrize("target_value", [[], [_PRINCIPAL_TARGET, _OTHER_TARGET]])
def test_amending_section_does_not_choose_an_ambiguous_target(tmp_path, target_value):
    rows = _amendment_rows()
    rows[1]["metadata"].pop("amendment_target")
    rows[1]["metadata"]["amendment_targets"] = target_value
    release = _write_test_corpus_release(tmp_path, rows)

    with pytest.raises(ValueError, match="Unresolved amendment context"):
        evals.resolve_corpus_source_unit(_AMENDING_SECTION, release)


def test_amending_section_rejects_conflicting_singular_and_plural_targets(tmp_path):
    rows = _amendment_rows()
    rows[1]["metadata"]["amendment_targets"] = [_OTHER_TARGET]
    release = _write_test_corpus_release(tmp_path, rows)

    with pytest.raises(ValueError, match="Unresolved amendment context"):
        evals.resolve_corpus_source_unit(_AMENDING_SECTION, release)


def test_amending_section_accepts_one_plural_target(tmp_path):
    rows = _amendment_rows()
    rows[1]["metadata"].pop("amendment_target")
    rows[1]["metadata"]["amendment_targets"] = [_PRINCIPAL_TARGET]
    release = _write_test_corpus_release(tmp_path, rows)

    source = evals.resolve_corpus_source_unit(_AMENDING_SECTION, release)

    assert source.requested == _PRINCIPAL_TARGET
    assert source.amendment_request == _AMENDING_SECTION


@pytest.mark.parametrize("mismatch", ["source_path", "version", "body_date"])
def test_amending_section_requires_same_act_authenticated_commencement(
    tmp_path, mismatch
):
    rows = _amendment_rows()
    if mismatch == "source_path":
        rows[2]["source_path"] = "sources/nz/statute/unrelated-act.xml"
    elif mismatch == "version":
        rows[2]["version"] = "another-capture"
    else:
        rows[2]["body"] = _COMMENCEMENT_BODY.replace("1 April 2027", "1 April 2028")
    release = _write_test_corpus_release(tmp_path, rows)

    with pytest.raises(ValueError, match="Unresolved amendment context"):
        evals.resolve_corpus_source_unit(_AMENDING_SECTION, release)


def test_routed_amendment_attestation_cannot_be_forged(tmp_path):
    release = _write_test_corpus_release(tmp_path, _amendment_rows())
    source = evals.resolve_corpus_source_unit(_AMENDING_SECTION, release)

    evals._validate_corpus_source_unit(
        source, release, target_identifier=_AMENDING_SECTION
    )
    with pytest.raises(ValueError):
        evals._validate_corpus_source_unit(
            replace(source, amendment_request=f"{_AMENDING_ACT}/section/3"),
            release,
            target_identifier=_AMENDING_SECTION,
        )


@pytest.mark.parametrize("request_principal", [False, True])
def test_required_commencement_context_cannot_be_dropped_by_budget(
    tmp_path, monkeypatch, request_principal
):
    release = _write_test_corpus_release(tmp_path, _amendment_rows())
    monkeypatch.setattr(evals, "_INJECTED_CONTEXT_LIMIT", 600)

    with pytest.raises(
        ValueError, match="Unresolved amendment context.*context budget"
    ):
        source = evals.resolve_corpus_source_unit(
            _PRINCIPAL_TARGET if request_principal else _AMENDING_SECTION, release
        )
        _workspace_prompt_for_source_unit(tmp_path, source)


@pytest.mark.parametrize("bodyless_parent", [False, True])
def test_parent_amendment_marker_blocks_unclassified_child(tmp_path, bodyless_parent):
    rows = _amendment_rows(metadata={})
    rows[1]["heading"] = "Rates"
    rows[1]["body"] = 'Delete the words "$200" and insert "$305".'
    rows.append(
        {
            "citation_path": _AMENDING_ACT,
            "source_path": rows[1]["source_path"],
            "body": None if bodyless_parent else "Amending instrument.",
            "metadata": {"document_type": "amendment act"},
        }
    )
    release = _write_test_corpus_release(tmp_path, rows)

    with pytest.raises(ValueError, match="Unresolved amendment context"):
        evals.resolve_corpus_source_unit(_AMENDING_SECTION, release)


def test_marked_bodyless_ancestor_is_not_hidden_by_unmarked_document_root(tmp_path):
    request = f"{_AMENDING_ACT}/part/2/section/15"
    source_path = "sources/nz/statute/2026-54.xml"
    release = _write_test_corpus_release(
        tmp_path,
        [
            {
                "citation_path": _AMENDING_ACT,
                "source_path": source_path,
                "body": "An instrument.",
                "metadata": {},
            },
            {
                "citation_path": f"{_AMENDING_ACT}/part/2",
                "source_path": source_path,
                "body": None,
                "metadata": {"document_type": "amendment part"},
            },
            {
                "citation_path": request,
                "source_path": source_path,
                "body": 'Delete the words "$200" and insert "$305".',
                "metadata": {},
            },
        ],
    )

    with pytest.raises(ValueError, match="Unresolved amendment context"):
        evals.resolve_corpus_source_unit(request, release)


@pytest.mark.parametrize("effective", ["0001-01-01", "2027-04-01"])
def test_amendment_commencement_gates_artifact_acceptance(
    tmp_path, monkeypatch, effective
):
    release = _write_test_corpus_release(tmp_path, _amendment_rows())
    source = evals.resolve_corpus_source_unit(_AMENDING_SECTION, release)
    policy = _canonical_rulespec_content_root(tmp_path, "nz")
    target = policy / evals._resolve_eval_output_path(_PRINCIPAL_TARGET)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "module": {
                    "source_verification": {"corpus_citation_path": _PRINCIPAL_TARGET}
                },
                "rules": [
                    {
                        "name": "accommodation_supplement_rate",
                        "kind": "parameter",
                        "dtype": "Money",
                        "unit": "NZD",
                        "metadata": {
                            "proof": {
                                "atoms": [
                                    {
                                        "path": "versions[0].formula",
                                        "source": {
                                            "corpus_citation_path": _AMENDING_SECTION
                                        },
                                    }
                                ]
                            }
                        },
                        "versions": [{"effective_from": effective, "formula": 305}],
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(
        evals.ValidatorPipeline,
        "_run_compile_check",
        lambda *_args: evals.ValidationResult("compile", passed=True),
    )
    monkeypatch.setattr(
        evals.ValidatorPipeline,
        "_run_ci",
        lambda *_args: evals.ValidationResult("ci", passed=True),
    )

    metrics = evals.evaluate_artifact(
        rulespec_file=target,
        policy_repo_root=policy,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        source_text=source.body,
        local_corpus_release=release,
        source_citation_path=_PRINCIPAL_TARGET,
        amendment_documents=source.amendment_documents,
        skip_reviewers=True,
    )

    assert metrics.compile_pass is (effective == "2027-04-01")
    if effective == "0001-01-01":
        assert "predates commencement 2027-04-01" in "\n".join(metrics.compile_issues)
