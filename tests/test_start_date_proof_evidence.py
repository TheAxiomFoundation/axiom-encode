"""Which proof evidence supports the start date of a date-silent unit (#2212).

Only the start-date proof evidence of a rule supports its starts: an
``effective_period`` atom of that same rule. The atoms are read with the proof
validator's reader, so every proof spelling that proof validation
authenticates gives the same outcome here.

The module text is synthetic. The replay case reads the committed public
fixture ``41648809989-control``.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from axiom_encode.checks import ValidationIssue
from axiom_encode.harness import start_date_evidence as sde
from axiom_encode.harness.proof_validator import (
    PROOF_ATOM_KINDS,
    PROOF_SOURCE_EVIDENCE_FIELDS,
    proof_source_evidence_texts,
    rule_proof_atoms,
    validate_rulespec_proofs,
)
from axiom_encode.harness.start_date_evidence import date_silent_start_issues
from tests.test_start_date_aliases import SHAPES, shaped_module
from tests.test_start_date_evidence import (
    CITATION,
    EVIDENCE,
    SOURCE,
    _guidance,
    _start_issues,
    issues,
    module,
)

SIBLING = "us-az/manual/des/faa6/utility-allowance-current-amount/block-1"
SIBLING_TEXT = "Utility allowances, current amounts effective 10/01/2025."
QUOTE = "current amounts effective 10/01/2025"


def atom(path: str, kind: str = "effective_period", **source: str) -> dict:
    return {
        "path": path,
        "kind": kind,
        "source": {"corpus_citation_path": SIBLING, **(source or {"excerpt": QUOTE})},
    }


def with_atoms(payload: dict, *atoms: dict, rule: int = 0, holder: str = "metadata"):
    """Attach ``atoms`` to one rule, under ``metadata.proof`` or ``proof``."""

    payload = copy.deepcopy(payload)
    target = payload["rules"][rule]
    if holder == "metadata":
        target["metadata"] = {"proof": {"atoms": list(atoms)}}
    else:
        target["proof"] = {"atoms": list(atoms)}
    return payload


# --------------------------------------------------------------------------
# Review finding 1: an unrelated proof atom is not start-date evidence
# --------------------------------------------------------------------------


def test_unrelated_amount_atom_quoting_a_sibling_date_is_not_support():
    """The reviews' case: one `amount` atom quoting a sibling provision."""

    bare = module("2025-10-01")
    assert len(issues(bare)) == 1
    payload = with_atoms(bare, atom("versions[0].formula", "amount"))
    # Proof validation accepts the atom: the quote stands in the sibling text.
    proof = validate_rulespec_proofs(
        yaml.safe_dump(payload), source_texts={SIBLING: SIBLING_TEXT, CITATION: SOURCE}
    )
    assert proof.passed and proof.atoms_checked == 1, proof.issues
    [issue] = issues(payload)
    assert issue.found == "2025-10-01"


@pytest.mark.parametrize("kind", sorted(PROOF_ATOM_KINDS - {"effective_period"}))
@pytest.mark.parametrize("path", ["versions[0].formula", "versions[0].effective_from"])
def test_only_the_effective_period_kind_is_start_date_evidence(kind, path):
    payload = with_atoms(module("2025-10-01"), atom(path, kind))
    assert len(issues(payload)) == 1
    assert issues(with_atoms(module("2025-10-01"), atom(path))) == []


def test_a_missing_or_malformed_kind_is_not_start_date_evidence():
    for kind in (None, "", "effective_periods", ["effective_period"], 7):
        bad = atom("versions[0].effective_from")
        bad["kind"] = kind
        assert len(issues(with_atoms(module("2025-10-01"), bad))) == 1, kind
    spaced = atom("versions[0].effective_from", " effective_period ")
    assert issues(with_atoms(module("2025-10-01"), spaced)) == []


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "path",
    ["versions[0].formula", "versions[0].effective_from", "effective_from", "from"],
)
def test_the_atom_path_is_not_examined(shape, path):
    """Public candidates anchor `effective_period` atoms at `.formula`."""

    payload = shaped_module("2025-10-01", shape)
    assert issues(with_atoms(payload, atom(path))) == []
    other = atom(path, excerpt="Effective 1/1/2024")
    assert len(issues(with_atoms(payload, other))) == 1


def test_an_atom_of_another_rule_is_not_support():
    payload = with_atoms(
        module("2025-10-01", "2025-10-01"), atom("versions[0].effective_from"), rule=1
    )
    [issue] = issues(payload)
    assert issue.subject.rule == "telephone_allowance_applies_0"
    assert "in 1 version(s) (rules `telephone_allowance_applies_0`)" in issue
    assert "telephone_allowance_applies_1" not in issue


def _raw_issues(content: str) -> list[ValidationIssue]:
    return date_silent_start_issues(
        yaml.safe_load(content),
        SOURCE,
        corpus_citation_path=CITATION,
        evidence=EVIDENCE,
        raw_content=content,
    )


@pytest.mark.parametrize("kind", ["effective_period", "amount"])
def test_raw_content_reads_atoms_from_the_proof_validated_mapping(kind):
    """Starts come from raw fields; proof atoms from the analyzer's mapping.

    The rule takes its proof through a YAML merge key. The merge-free start
    reading drops it, but proof validation reads the merged mapping, so an
    ``effective_period`` atom there is the rule's own evidence.
    """

    proof = {"metadata": {"proof": {"atoms": [atom("versions[0].formula", kind)]}}}
    content = yaml.safe_dump(
        {
            "proof_base": proof,
            "rules": [
                {
                    "name": "telephone_allowance_applies_0",
                    "kind": "derived",
                    "dtype": "Boolean",
                    "entity": "Household",
                    "period": "Month",
                    "source": CITATION,
                    "versions": [
                        {"effective_from": "2025-10-01", "formula": "pays_telephone"}
                    ],
                }
            ],
        },
        sort_keys=False,
    )
    content = content.replace("proof_base:", "proof_base: &proof", 1).replace(
        "- name: telephone_allowance_applies_0",
        "- <<: *proof\n  name: telephone_allowance_applies_0",
        1,
    )
    assert yaml.safe_load(content)["rules"][0]["metadata"] == proof["metadata"]
    assert len(_raw_issues(content)) == (0 if kind == "effective_period" else 1)


def test_each_version_start_needs_its_own_printed_date():
    payload = module("2025-10-01")
    payload["rules"][0]["versions"].append(
        {"effective_from": "2024-10-01", "formula": "pays_telephone"}
    )
    one = with_atoms(payload, atom("versions[0].formula"))
    [issue] = issues(one)
    assert issue.found == "2024-10-01"
    assert issue.subject.field == "versions[1].effective_from"
    both = with_atoms(
        payload,
        atom("versions[0].formula"),
        atom("versions[1].formula", excerpt="Effective October 1, 2024"),
    )
    assert issues(both) == []


def test_replay_control_passes_through_its_effective_period_atom():
    """Public control `41648809989-control`: a change-notice start date.

    Its `2026-03-01` start is printed ("03/2026") in an `effective_period`
    atom of the same rule, anchored at `versions[0].formula`. The same atom
    under another kind, or on another rule, does not support the start.
    """

    from scripts.replay_us_failure_bundles import (
        DEFAULT_FIXTURE_ROOT,
        load_fixture,
        replay_case,
    )

    case = load_fixture(DEFAULT_FIXTURE_ROOT / "41648809989-control")
    assert _start_issues(replay_case(case)["issues"]) == []
    candidate = yaml.safe_load(case.candidate)
    rule = candidate["rules"][0]
    atoms = rule["metadata"]["proof"]["atoms"]
    [period] = [item for item in atoms if item["kind"] == "effective_period"]
    assert period["path"] == "versions[0].formula"
    assert sde.start_date_proof_texts(rule) == (period["source"]["excerpt"],)
    evidence = sde.start_date_evidence_from_metadata(dict(case.source_metadata))

    def found(payload):
        return date_silent_start_issues(
            payload, case.source, corpus_citation_path=case.citation, evidence=evidence
        )

    assert found(candidate) == []
    moved = copy.deepcopy(candidate)
    moved["rules"][0]["metadata"]["proof"]["atoms"].remove(period)
    moved["rules"][1]["metadata"]["proof"]["atoms"].append(period)
    rekinded = copy.deepcopy(candidate)
    for item in rekinded["rules"][0]["metadata"]["proof"]["atoms"]:
        item["kind"] = "parameter"
    for payload in (moved, rekinded):
        [issue] = found(payload)
        assert issue.found == "2026-03-01"
        assert issue.subject.field == "versions[1].effective_from"


def test_message_and_check_text_name_the_exact_proof_evidence():
    [issue] = issues(
        with_atoms(module("2025-10-01"), atom("versions[0].formula", "amount"))
    )
    texts = (
        str(issue),
        sde.CHECK_START_UNSUPPORTED.requires,
        sde.CHECK_START_UNSUPPORTED.fix,
        sde.__doc__,
    )
    for text in texts:
        assert "effective_period" in text
        assert "proof-atom excerpt" not in text
        assert "in a proof excerpt" not in text
    assert "a date quoted by any other proof atom is not support" in issue
    assert "A date quoted by any other proof atom is not support" in (
        sde.CHECK_START_UNSUPPORTED.fix
    )


def test_prompt_names_the_exact_proof_evidence():
    text = _guidance(EVIDENCE.source_as_of, EVIDENCE.expression_date)
    assert "`kind: effective_period`" in text
    assert "A date quoted by any other proof atom" in text
    assert "proof-atom excerpt" not in text


# --------------------------------------------------------------------------
# Review finding 2: every proof spelling the validator authenticates is read
# --------------------------------------------------------------------------

SPELLINGS = [
    (holder, field)
    for holder in ("metadata", "proof")
    for field in PROOF_SOURCE_EVIDENCE_FIELDS
]


@pytest.mark.parametrize(("holder", "field"), SPELLINGS)
@pytest.mark.parametrize("shape", SHAPES)
def test_accepted_proof_spellings_give_one_outcome(shape, holder, field):
    payload = shaped_module("2025-10-01", shape)
    path = "versions[0].formula" if "version-" in shape else "formula"
    supported = with_atoms(payload, atom(path, **{field: QUOTE}), holder=holder)
    assert issues(supported) == []
    proof = validate_rulespec_proofs(
        yaml.safe_dump(supported),
        source_texts={SIBLING: SIBLING_TEXT, CITATION: SOURCE},
    )
    assert proof.passed and proof.atoms_checked == 1, proof.issues
    other = with_atoms(
        payload, atom(path, **{field: "Effective 1/1/2024"}), holder=holder
    )
    assert len(issues(other)) == 1


def test_evidence_fields_are_the_proof_validators():
    assert PROOF_SOURCE_EVIDENCE_FIELDS == ("excerpt", "quote")
    source = {"excerpt": " a ", "quote": "b", "note": "c", "corpus_citation_path": "d"}
    assert proof_source_evidence_texts(source) == (("excerpt", "a"), ("quote", "b"))
    for blank in (None, "x", [], {"excerpt": "  "}, {"quote": 7}, {"note": QUOTE}):
        assert proof_source_evidence_texts(blank) == ()
    # A date in either field is read.
    both = atom("versions[0].effective_from", excerpt="No date.", quote=QUOTE)
    assert issues(with_atoms(module("2025-10-01"), both)) == []


def test_proof_validator_authenticates_the_texts_this_check_reads():
    """A date this check reads is one the proof validator checks in the corpus."""

    for holder, field in SPELLINGS:
        forged = with_atoms(
            module("2025-10-01"),
            atom(
                "versions[0].effective_from", **{field: "effective 10/01/2025 forged"}
            ),
            holder=holder,
        )
        assert issues(forged) == []
        proof = validate_rulespec_proofs(
            yaml.safe_dump(forged),
            source_texts={SIBLING: SIBLING_TEXT, CITATION: SOURCE},
        )
        assert not proof.passed, (holder, field)
        assert any(f"`source.{field}` does not appear" in item for item in proof.issues)


def test_rule_proof_precedence_is_the_proof_validators():
    """`metadata.proof` wins over `proof`; only the winner is authenticated."""

    good = atom("versions[0].formula")
    other = atom("versions[0].formula", "amount")
    payload = with_atoms(module("2025-10-01"), other)
    payload["rules"][0]["proof"] = {"atoms": [good]}
    assert rule_proof_atoms(payload["rules"][0]) == (other,)
    assert len(issues(payload)) == 1
    payload["rules"][0]["metadata"] = {"proof": "not a mapping"}
    assert rule_proof_atoms(payload["rules"][0]) == (good,)
    assert issues(payload) == []


def test_atoms_the_proof_validator_does_not_authenticate_are_not_read():
    good = atom("versions[0].effective_from")
    for location in ("metadata", "proof"):
        payload = module("2025-10-01")
        version = payload["rules"][0]["versions"][0]
        if location == "metadata":
            version["metadata"] = {"proof": {"atoms": [good]}}
        else:
            version["proof"] = {"atoms": [good]}
        assert len(issues(payload)) == 1, location
    payload = module("2025-10-01")
    payload["module"]["proof"] = {"atoms": [good]}
    payload["proof"] = {"atoms": [good]}
    assert len(issues(payload)) == 1
    for atoms in ("x", {"a": good}, [QUOTE, None, [good]]):
        payload = module("2025-10-01")
        payload["rules"][0]["metadata"] = {"proof": {"atoms": atoms}}
        assert len(issues(payload)) == 1, atoms


# --------------------------------------------------------------------------
# Every version is examined
# --------------------------------------------------------------------------


def test_every_later_version_is_examined():
    payload = module("2026-08-01")
    payload["rules"][0]["versions"].append(
        {"effective_from": "2025-10-01", "formula": "pays_telephone"}
    )
    [issue] = issues(payload)
    assert issue.found == "2025-10-01"
    assert issue.subject.field == "versions[1].effective_from"
    assert "in 1 version(s)" in issue


# --------------------------------------------------------------------------
# First-review nit: an unnamed rule in the message
# --------------------------------------------------------------------------


def test_a_rule_without_a_name_is_named_by_its_position():
    payload = module("2026-08-01", "2025-10-01", "2025-10-01")
    del payload["rules"][1]["name"]
    payload["rules"][2]["name"] = "  "
    [issue] = issues(payload)
    assert "in 2 version(s) (rules `rules[1]`, `rules[2]`)" in issue
    assert "(rules )" not in issue
    assert issue.subject.rule is None
    named = module("2026-08-01", "2025-10-01", "2025-10-01")
    del named["rules"][1]["name"]
    [issue] = issues(named)
    assert "(rules `rules[1]`, `telephone_allowance_applies_2`)" in issue


# --------------------------------------------------------------------------
# The landed and open Arizona candidate shapes
# --------------------------------------------------------------------------


def test_arizona_candidate_shapes():
    """Date-silent manual module, snapshot 2026-07-17, no imports."""

    assert "imports" not in module("2026-08-01")
    assert issues(module("2026-08-01")) == []
    assert len(issues(module("2025-10-01"))) == 1
    quoted = with_atoms(module("2025-10-01"), atom("versions[0].formula", "amount"))
    assert len(issues(quoted)) == 1
    # The unrelated atom changes nothing for a supported start either.
    assert (
        issues(with_atoms(module("2026-08-01"), atom("versions[0].formula", "amount")))
        == []
    )
