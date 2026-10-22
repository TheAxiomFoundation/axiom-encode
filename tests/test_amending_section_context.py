"""Direct amendment requests must use principal identities and legal dates."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

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
# Public-law excerpt from the diagnosis's authenticated NZ s15 source.
_AMENDMENT_BODY = (
    "In Schedule 4, Part 7, replace clauses 4 to 6A with:\n"
    "4 | To a person (other than a specified person) who has 1 or more "
    "dependent children and who is in a relationship | 70% of the amount by "
    "which an applicant's weekly qualifying accommodation costs exceeds 40% "
    "of the base rate, but not more than $305 a week, if the applicant "
    "resides in Area 1."
)


def _amendment_rows(*, metadata=None):
    """Add ingest declarations to the report's otherwise undated NZ source."""

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
    # The real corpus title and heading reveal amendment form, but its s15
    # record has no machine-readable target, Part membership, or legal date.
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


def test_parent_amendment_marker_blocks_unclassified_child(tmp_path):
    rows = _amendment_rows(metadata={})
    rows[1]["heading"] = "Rates"
    rows[1]["body"] = 'Delete the words "$200" and insert "$305".'
    rows.append(
        {
            "citation_path": _AMENDING_ACT,
            "source_path": rows[1]["source_path"],
            "body": "Amending instrument.",
            "metadata": {"document_type": "amendment act"},
        }
    )
    release = _write_test_corpus_release(tmp_path, rows)

    with pytest.raises(ValueError, match="Unresolved amendment context"):
        evals.resolve_corpus_source_unit(_AMENDING_SECTION, release)


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
