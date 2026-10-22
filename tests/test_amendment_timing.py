"""Amending provisions cannot supply executable behavior before commencement."""

from datetime import date
from types import SimpleNamespace

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness.amendment_timing import amendment_timing_issues

AMENDMENT = "nz/statute/act/public/2026/0054/section/15"
COMMENCEMENT = "nz/statute/act/public/2026/0054/section/2"
PRINCIPAL = "nz/statute/act/public/2018/0032/schedule/4/part/8"
COMMENCEMENT_TEXT = (
    "1\nThis Act comes into force on 2 November 2026.\n"
    "2\nHowever, Part 2 comes into force on 1 April 2027."
)


def _document(commencement="2027-04-01", citation=AMENDMENT):
    return SimpleNamespace(
        citation_path=citation,
        metadata={
            "commencement_date": commencement,
            "commencement_citation_path": COMMENCEMENT,
        },
        body=COMMENCEMENT_TEXT,
    )


def _proof(citation=AMENDMENT, path="formula"):
    return {"atoms": [{"path": path, "source": {"corpus_citation_path": citation}}]}


def _content(effective="2027-04-01", citation=AMENDMENT):
    return yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "rules": [
                {
                    "name": "winter_energy_payment_rate",
                    "kind": "parameter",
                    "metadata": {"proof": _proof(citation)},
                    "versions": [{"effective_from": effective, "formula": 42}],
                }
            ],
        }
    )


def test_amending_section_scaffold_date_is_rejected():
    issues = amendment_timing_issues(_content("0001-01-01"), [_document()])
    assert len(issues) == 1
    assert "0001-01-01 predates commencement 2027-04-01" in issues[0]
    assert AMENDMENT in issues[0]
    assert COMMENCEMENT in issues[0]


@pytest.mark.parametrize("effective", ["2027-04-01", "2028-01-01", date(2027, 4, 1)])
def test_commenced_amendment_is_allowed(effective):
    assert amendment_timing_issues(_content(effective), [_document()]) == []


@pytest.mark.parametrize(
    "effective", [None, "", "2027-02-30", "20270401", 20270401, "2027-04-01T00:00:00"]
)
def test_amendment_backed_version_requires_valid_effective_date(effective):
    issues = amendment_timing_issues(_content(effective), [_document()])
    assert len(issues) == 1
    assert "must declare a valid effective_from" in issues[0]


def test_amendment_backed_version_requires_effective_date_field():
    payload = yaml.safe_load(_content())
    payload["rules"][0]["versions"][0].pop("effective_from")
    assert amendment_timing_issues(yaml.safe_dump(payload), [_document()])


@pytest.mark.parametrize("commencement", [None, "2027-02-30", "2027", 20270401])
def test_invalid_commencement_metadata_cannot_silently_disable_guard(commencement):
    issues = amendment_timing_issues(_content(), [_document(commencement)])
    assert len(issues) == 1
    assert "commencement_date is missing or invalid" in issues[0]


@pytest.mark.parametrize("metadata", [None, {}, {"document_type": "amending_act"}])
def test_supporting_documents_without_bound_metadata_do_not_invent_bounds(metadata):
    document = SimpleNamespace(citation_path=COMMENCEMENT, metadata=metadata)
    assert (
        amendment_timing_issues(_content("2027-04-01", COMMENCEMENT), [document]) == []
    )


@pytest.mark.parametrize("citation", [AMENDMENT, AMENDMENT + "/1"])
def test_amendment_proof_matches_exact_citation_and_children(citation):
    assert amendment_timing_issues(_content("2026-10-12", citation), [_document()])


def test_ambiguous_act_root_proof_inherits_contained_amendment_bound():
    act_root = AMENDMENT.removesuffix("/section/15")
    issues = amendment_timing_issues(_content("2026-10-12", act_root), [_document()])
    assert len(issues) == 1
    assert "predates commencement 2027-04-01" in issues[0]


@pytest.mark.parametrize(
    "citation", [PRINCIPAL, AMENDMENT + "0", AMENDMENT + "A", COMMENCEMENT]
)
def test_unrelated_or_principal_citation_does_not_inherit_amendment_bound(citation):
    assert (
        amendment_timing_issues(_content("0001-01-01", citation), [_document()]) == []
    )


def test_rule_proof_paths_allow_preexisting_principal_version():
    payload = yaml.safe_load(_content())
    rule = payload["rules"][0]
    rule["metadata"]["proof"]["atoms"] = [
        _proof(PRINCIPAL, "versions[0].formula")["atoms"][0],
        _proof(AMENDMENT, "versions[1].formula")["atoms"][0],
    ]
    rule["versions"].insert(0, {"effective_from": "2018-11-26", "formula": 20})
    assert amendment_timing_issues(yaml.safe_dump(payload), [_document()]) == []
    rule["versions"][1]["effective_from"] = "2026-10-12"
    issues = amendment_timing_issues(yaml.safe_dump(payload), [_document()])
    assert len(issues) == 1
    assert "versions[1]" in issues[0]


@pytest.mark.parametrize("cutoff_path", ["effective_to", "versions[0].effective_to"])
def test_amendment_cutoff_proof_preserves_preexisting_principal_version(cutoff_path):
    payload = yaml.safe_load(_content())
    rule = payload["rules"][0]
    rule["metadata"]["proof"]["atoms"] = [
        _proof(PRINCIPAL, "versions[0].formula")["atoms"][0],
        _proof(AMENDMENT, cutoff_path)["atoms"][0],
        _proof(AMENDMENT, "versions[1].formula")["atoms"][0],
    ]
    rule["versions"].insert(
        0,
        {
            "effective_from": "2018-11-26",
            "effective_to": "2027-03-31",
            "formula": 20,
        },
    )
    assert amendment_timing_issues(yaml.safe_dump(payload), [_document()]) == []
    rule["versions"][1]["effective_from"] = "2026-10-12"
    issues = amendment_timing_issues(yaml.safe_dump(payload), [_document()])
    assert len(issues) == 1
    assert "versions[1]" in issues[0]


@pytest.mark.parametrize(
    "path", ["effective_from", "versions[0].effective_from", "formula", "values"]
)
def test_amendment_timing_and_behavior_proofs_guard_effective_start(path):
    payload = yaml.safe_load(_content("2026-10-12"))
    payload["rules"][0]["metadata"]["proof"] = _proof(AMENDMENT, path)
    assert amendment_timing_issues(yaml.safe_dump(payload), [_document()])


@pytest.mark.parametrize("proof_key", ["metadata", "proof"])
@pytest.mark.parametrize("rule_path", ["formula", "versions[0].formula"])
def test_version_proof_cannot_erase_amendment_rule_proof_for_principal_history(
    proof_key, rule_path
):
    payload = yaml.safe_load(_content())
    payload["rules"][0]["metadata"]["proof"] = _proof(AMENDMENT, rule_path)
    version = {"effective_from": "2018-11-26", "formula": 20}
    if proof_key == "metadata":
        version["metadata"] = {"proof": _proof(PRINCIPAL)}
    else:
        version["proof"] = _proof(PRINCIPAL)
    payload["rules"][0]["versions"].insert(0, version)
    issues = amendment_timing_issues(yaml.safe_dump(payload), [_document()])
    assert len(issues) == 1
    assert "versions[0]" in issues[0]


@pytest.mark.parametrize("rule_path", ["formula", "versions[0].formula"])
@pytest.mark.parametrize("version_path", ["effective_from", "values", None, "", " "])
def test_unrelated_version_proof_does_not_erase_amendment_formula_evidence(
    rule_path, version_path
):
    payload = yaml.safe_load(_content("2026-10-12"))
    rule = payload["rules"][0]
    rule["metadata"]["proof"] = _proof(AMENDMENT, rule_path)
    rule["versions"][0]["metadata"] = {"proof": _proof(PRINCIPAL, version_path)}
    issues = amendment_timing_issues(yaml.safe_dump(payload), [_document()])
    assert len(issues) == 1
    assert "predates commencement 2027-04-01" in issues[0]


def test_missing_version_atom_path_does_not_erase_amendment_formula_evidence():
    payload = yaml.safe_load(_content("2026-10-12"))
    proof = _proof(PRINCIPAL)
    proof["atoms"][0].pop("path")
    payload["rules"][0]["versions"][0]["metadata"] = {"proof": proof}
    assert amendment_timing_issues(yaml.safe_dump(payload), [_document()])


@pytest.mark.parametrize(
    "source",
    [
        None,
        {},
        {"corpus_citation_path": None},
        {"corpus_citation_path": ""},
        {"corpus_citation_path": " "},
        {"corpus_citation_path": 1},
    ],
)
def test_version_atom_without_corpus_source_cannot_erase_amendment_evidence(source):
    payload = yaml.safe_load(_content("2026-10-12"))
    proof = {"atoms": [{"path": "formula", "source": source}]}
    payload["rules"][0]["versions"][0]["metadata"] = {"proof": proof}
    assert amendment_timing_issues(yaml.safe_dump(payload), [_document()])


def test_missing_version_atom_source_cannot_erase_amendment_evidence():
    payload = yaml.safe_load(_content("2026-10-12"))
    proof = {"atoms": [{"path": "formula"}]}
    payload["rules"][0]["versions"][0]["metadata"] = {"proof": proof}
    assert amendment_timing_issues(yaml.safe_dump(payload), [_document()])


@pytest.mark.parametrize("version_proof", [None, {}, {"atoms": []}])
def test_unproved_version_conservatively_inherits_amendment_rule_proof(version_proof):
    payload = yaml.safe_load(_content("2026-10-12"))
    payload["rules"][0]["versions"][0]["metadata"] = {"proof": version_proof}
    assert amendment_timing_issues(yaml.safe_dump(payload), [_document()])


def test_version_amendment_proof_is_checked_without_rule_proof():
    payload = yaml.safe_load(_content("2026-10-12"))
    rule = payload["rules"][0]
    rule["versions"][0]["metadata"] = rule.pop("metadata")
    assert amendment_timing_issues(yaml.safe_dump(payload), [_document()])


@pytest.mark.parametrize(
    "kind", ["parameter", "derived", "derived_relation", "data_relation"]
)
def test_each_executable_rule_kind_is_guarded(kind):
    payload = yaml.safe_load(_content("0001-01-01"))
    payload["rules"][0]["kind"] = kind
    assert amendment_timing_issues(yaml.safe_dump(payload), [_document()])


@pytest.mark.parametrize(
    "content", ["[", "null", "rules: []", "format: other\nrules: []"]
)
def test_ordinary_schema_validation_owns_malformed_documents(content):
    assert amendment_timing_issues(content, [_document()]) == []


@given(
    effective=st.dates(min_value=date(1, 1, 1), max_value=date(9999, 12, 31)),
    commencement=st.dates(min_value=date(1, 1, 1), max_value=date(9999, 12, 31)),
)
@settings(max_examples=100, deadline=None)
def test_no_amendment_derived_rule_can_predate_commencement(effective, commencement):
    issues = amendment_timing_issues(
        _content(effective.isoformat()), [_document(commencement.isoformat())]
    )
    assert bool(issues) is (effective < commencement)
