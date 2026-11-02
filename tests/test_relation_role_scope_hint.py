"""Keep direct Person-anchored household aggregates in their valid entity."""

import json
import subprocess

import pytest

from axiom_encode.harness import evals

SCOPE_HINT_SENTENCES = (
    'For any relation anchored on `Person`, keep composition and participation aggregates on `entity: Person`, anchored on the applicant or other source-stated person, even when the source says "household".',
    'For household-level rules without a relation anchored on `Person`, if the source uses the word "household" or "household\'s", use `entity: Household`.',
    'If the source says "households in which all members", "households with a member", or another household-level condition expressed through member facts, encode the executable eligibility/result on `Household`, except for aggregates over a relation anchored on `Person`.',
)


def assert_scope_hint(prompt):
    normalized = " ".join(prompt.split())
    for sentence in SCOPE_HINT_SENTENCES:
        assert sentence in normalized


def test_a3_hint_exempts_every_person_anchored_relation():
    assert_scope_hint(evals._MINIMAL_SCOPE_HINT)


@pytest.mark.parametrize("sentence", SCOPE_HINT_SENTENCES)
def test_scope_hint_sentence_deletion_is_detected(sentence):
    normalized = " ".join(evals._MINIMAL_SCOPE_HINT.split())
    assert sentence in normalized
    with pytest.raises(AssertionError):
        assert_scope_hint(normalized.replace(sentence, "", 1))


DIRECT_PERSON_INPUT = "# Synthetic encoder test input; not a RuleSpec module.\nformat: rulespec/v1\nmodule:\n  summary: An applicant’s household includes the applicant and their spouse.\nrules:\n  - name: member_of_household\n    kind: data_relation\n    data_relation: {arity: 2, arguments: [Person, Household]}\n  - name: spouse_of\n    kind: data_relation\n    data_relation: {arity: 2, arguments: [Person, Person]}\n  - name: residence_size\n    kind: derived\n    entity: Household\n    dtype: Integer\n    versions:\n      - effective_from: '2026-01-01'\n        formula: len(member_of_household)\n  - name: household_member\n    kind: derived_relation\n    derived_relation:\n      arity: 2\n      slot_entities: [Person, Person]\n      current_slot: 0\n      related_slot: 1\n      source_path:\n        - {relation: member_of_household, from_slot: 0, to_slot: 1}\n        - {relation: member_of_household, from_slot: 1, to_slot: 0}\n    versions:\n      - effective_from: '2026-01-01'\n        formula: candidate == anchor or spouse_of(anchor, candidate) or spouse_of(candidate, anchor)\n  - name: household_composition_size\n    kind: derived\n    entity: Person\n    dtype: Integer\n    versions:\n      - effective_from: '2026-01-01'\n        formula: len(household_member)\n"


def test_a3_direct_spouse_relation_keeps_person_aggregate(tmp_path, real_engine):
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/direct_person.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(DIRECT_PERSON_INPUT)
    artifact = tmp_path / "compiled.json"
    compiled = subprocess.run(
        [
            str(real_engine.binary),
            "compile",
            "--program",
            str(module),
            "--rulespec-root",
            str(root),
            "--output",
            str(artifact),
        ],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr
    target = "us:policies/example/direct_person#"
    period = {"period_kind": "month", "start": "2026-01-01", "end": "2026-01-31"}
    interval = {"start": period["start"], "end": period["end"]}
    request = {
        "mode": "explain",
        "dataset": {
            "inputs": [],
            "relations": [
                {
                    "name": target + "relation.member_of_household",
                    "tuple": [name, "home"],
                    "interval": interval,
                }
                for name in ["a", "b", "c"]
            ]
            + [
                {
                    "name": target + "relation.spouse_of",
                    "tuple": ["a", "b"],
                    "interval": interval,
                }
            ],
        },
        "queries": [
            {
                "entity_id": name,
                "period": period,
                "outputs": [target + "household_composition_size"],
            }
            for name in ["a", "b", "c"]
        ],
    }
    executed = subprocess.run(
        [str(real_engine.binary), "run-compiled", "--artifact", str(artifact)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
    )
    assert executed.returncode == 0, executed.stderr
    sizes = [
        int(row["outputs"][target + "household_composition_size"]["value"]["value"])
        for row in json.loads(executed.stdout)["results"]
    ]
    assert dict(zip(["a", "b", "c"], sizes, strict=True)) == {"a": 2, "b": 2, "c": 1}
    assert_scope_hint(evals._MINIMAL_SCOPE_HINT)
