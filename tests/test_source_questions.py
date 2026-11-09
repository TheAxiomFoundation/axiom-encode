"""Question files for the source-conflict compare step (``--question``)."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from axiom_encode import source_conflicts
from axiom_encode.source_conflicts import cli as compare_cli
from axiom_encode.source_conflicts import question as question_module
from axiom_encode.source_conflicts.compare import render_record
from axiom_encode.source_conflicts.engine import Engine
from axiom_encode.source_conflicts.gates import LinkGateError
from axiom_encode.source_conflicts.link import LinkError, load_link
from axiom_encode.source_conflicts.question import (
    compare_question,
    load_question,
    parse_question,
)
from tests.test_source_conflicts import (
    FIXTURES,
    LINK,
    REPO,
    REPOSITORY,
    RULESPEC,
    A,
    B,
    FakeEngine,
    artifact_payload,
    run_compare,
)

QUESTION = FIXTURES / "work_requirement.question.yaml"
EXAMPLES = REPO / "docs" / "source-conflicts" / "examples"
HR1_QUESTION = EXAMPLES / "us-medicaid-community-engagement.question.json"
HR1_LINK = EXAMPLES / "us-medicaid-community-engagement.link.json"
C = "us:policies/demo-agency/work-requirement-faq#"
QID = "us:fixture/work-requirement-met-for-month"


def document() -> dict:
    return yaml.safe_load(QUESTION.read_text())


def engine() -> FakeEngine:
    return FakeEngine(
        {
            "us/statutes/999/1/a.yaml": artifact_payload("a"),
            "us/regulations/99-cfr/1/2.yaml": artifact_payload("b"),
            "us/policies/demo-agency/work-requirement-faq.yaml": artifact_payload("c"),
        }
    )


def run_question(tmp_path: Path, doc: dict | None = None):
    return compare_question(
        parse_question(doc if doc is not None else document()),
        rulespec_root=RULESPEC,
        engine=engine(),
        workdir=tmp_path,
        repository=REPOSITORY,
    )


def answer(doc: dict, answer_id: str) -> dict:
    return next(item for item in doc["answers"] if item["id"] == answer_id)


# -- loader and expansion ---------------------------------------------------


def test_question_expands_to_every_pair_with_statute_first():
    question = load_question(QUESTION)
    assert question.id == QID
    assert [item.id for item in question.answers] == [
        "statute",
        "regulation",
        "guidance",
    ]
    assert [pair.id for pair in question.pairs] == [
        f"{QID}:statute~regulation",
        f"{QID}:statute~guidance",
        f"{QID}:regulation~guidance",
    ]
    assert [pair.record_name for pair in question.pairs] == [
        "statute~regulation.json",
        "statute~guidance.json",
        "regulation~guidance.json",
    ]
    first = question.pairs[0]
    assert (first.a.document_type, first.b.document_type) == ("statute", "regulation")
    assert first.link.id == first.id
    assert first.link.sides["a"].module == "us/statutes/999/1/a.yaml"
    assert first.link.sides["b"].guards == (B + "individual_in_scope",)
    # A pair holds only the facts that one of its two answers reads.
    assert list(first.link.facts) == [
        "work_hours",
        "service_hours",
        "monthly_income",
        "agency_counts_income_as_hours",
    ]
    assert list(question.pairs[1].link.facts) == [
        "work_hours",
        "service_hours",
        "enrolled_student",
    ]
    assert question.pairs[1].link.facts["work_hours"].inputs == {}


def test_pair_ids_and_sides_do_not_depend_on_the_order_of_the_answers():
    doc = document()
    reversed_doc = copy.deepcopy(doc)
    reversed_doc["answers"].reverse()
    first, second = parse_question(doc), parse_question(reversed_doc)
    assert [pair.id for pair in first.pairs] == [pair.id for pair in second.pairs]
    assert [pair.link.sha256 for pair in first.pairs] == [
        pair.link.sha256 for pair in second.pairs
    ]
    assert first.sha256 == second.sha256
    changed = copy.deepcopy(doc)
    answer(changed, "regulation")["fixed"][B + "input.minimum_wage_rate"]["value"] = "8"
    assert parse_question(changed).sha256 != first.sha256


def test_a_constraint_over_a_fact_no_answer_of_the_pair_reads_is_not_applied():
    question = load_question(QUESTION)
    by_name = {pair.record_name: pair for pair in question.pairs}
    assert by_name["statute~guidance.json"].constraints_not_applied == (
        "monthly_income >= work_hours * 7.25",
    )
    assert by_name["statute~guidance.json"].link.constraints == ()
    assert by_name["statute~regulation.json"].constraints_not_applied == ()
    assert question.warnings == (
        "pair statute~guidance: constraint 'monthly_income >= work_hours * 7.25' is "
        "not applied, because neither answer reads one of its facts",
    )


def mutate(path: list, value=None, *, delete: bool = False) -> dict:
    doc = document()
    target = doc
    for key in path[:-1]:
        target = target[key]
    if delete:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return doc


def with_answers(change) -> dict:
    doc = document()
    change(doc["answers"])
    return doc


@pytest.mark.parametrize(
    "doc,message",
    [
        (
            mutate(["format"], "axiom-source-conflict-link/v1"),
            "format must be 'axiom-source-question/v1'",
        ),
        (mutate(["id"], "work-requirement"), "id must be a stable lowercase id"),
        (mutate(["question"], delete=True), "lacks the required key 'question'"),
        (mutate(["links"], []), "has the unknown key 'links'"),
        (
            with_answers(lambda answers: answers.__delitem__(slice(1, 3))),
            "two or more answers",
        ),
        (
            mutate(["answers", 1, "id"], "statute"),
            "the answer id 'statute' appears twice",
        ),
        (
            mutate(["answers", 1, "id"], "reg~1"),
            r"answers\[1\].id must be a lowercase id",
        ),
        (
            mutate(["answers", 1, "document_type"], "manual"),
            "answers.regulation.document_type must be one of statute, regulation",
        ),
        (
            mutate(["answers", 1, "jurisdiction"], "USA"),
            "answers.regulation.jurisdiction must be like",
        ),
        (
            mutate(["answers", 1, "corpus_citation_path"], "42 CFR 435.552"),
            "corpus_citation_path must be a corpus path",
        ),
        (
            mutate(["answers", 1, "corpus_citation_path"], delete=True),
            "lacks the required key 'corpus_citation_path'",
        ),
        (
            mutate(["answers", 1, "effective_from"], "October 2026"),
            "effective_from must be a YYYY-MM-DD date",
        ),
        (
            mutate(["answers", 1, "effective_to"], "2026-01-01"),
            "effective_to is before its effective_from",
        ),
        (
            mutate(["answers", 1, "inputs", "wage"], B + "input.minimum_wage_rate"),
            "names 'wage', which is not a fact of the question",
        ),
        (
            mutate(["answers", 1, "extra"], 1),
            r"answers\[1\] has the unknown key 'extra'",
        ),
        (
            mutate(["answers", 2, "inputs"], delete=True),
            "facts.enrolled_student is not used by any answer",
        ),
        (
            mutate(["facts", "unused"], {"kind": "bool"}),
            "facts.unused is not used by any answer",
        ),
        (
            mutate(["facts", "work_hours", "a"], "x"),
            "facts.work_hours has the unknown key 'a'",
        ),
        (
            mutate(["constraints", 0, "rule"], "monthly_income >= wage"),
            "'wage', which is not a fact",
        ),
        (
            mutate(["answers", 0, "combine"], "first"),
            "pair statute~regulation: a.combine must be one of any, all",
        ),
        (
            mutate(["answers", 1, "outputs"], []),
            "pair statute~regulation: b.outputs must be a non-empty list",
        ),
        (
            mutate(["facts", "work_hours", "kind"], "money"),
            "facts.work_hours.kind must be one of",
        ),
    ],
)
def test_invalid_questions_are_refused_with_a_clear_message(doc, message):
    with pytest.raises(LinkError, match=message):
        parse_question(doc)


def test_two_answers_must_agree_on_a_shared_fixed_or_derived_input():
    doc = document()
    name = A + "input.subject_to_requirement"
    answer(doc, "guidance")["fixed"] = {name: {"kind": "bool", "value": False}}
    with pytest.raises(
        LinkError, match="answers statute and guidance give different fixed definitions"
    ):
        parse_question(doc)


def test_a_repeated_key_in_a_question_file_is_refused(tmp_path):
    path = tmp_path / "question.yaml"
    path.write_text(QUESTION.read_text() + "answers: []\n")
    with pytest.raises(
        LinkError, match="cannot parse the question file .* 'answers' appears twice"
    ):
        load_question(path)


def test_period_outside_an_answers_effective_dates_is_a_warning():
    doc = document()
    answer(doc, "regulation")["effective_from"] = "2027-06-01"
    assert (
        "answer regulation: the period start 2027-01-01 is outside its effective dates"
        in parse_question(doc).warnings
    )


def test_question_schema_lists_the_keys_the_loader_accepts():
    schema = json.loads(
        (Path(source_conflicts.__file__).parent / "question.schema.json").read_text()
    )
    for node, keys in (
        (schema, question_module._TOP_KEYS),
        (schema["$defs"]["fact"], question_module._FACT_KEYS),
        (schema["$defs"]["answer"], question_module._ANSWER_KEYS),
    ):
        assert set(node["properties"]) == set(keys)
        assert set(node["required"]) == {key for key, needed in keys.items() if needed}
        assert node["additionalProperties"] is False
    assert schema["properties"]["format"]["const"] == source_conflicts.QUESTION_FORMAT
    assert (
        tuple(schema["$defs"]["answer"]["properties"]["document_type"]["enum"])
        == question_module.DOCUMENT_TYPES
    )


# -- compare ----------------------------------------------------------------


def test_three_answers_give_three_pair_records_and_a_summary(tmp_path):
    summary, records = run_question(tmp_path)
    assert list(records) == [
        "statute~regulation.json",
        "statute~guidance.json",
        "regulation~guidance.json",
    ]
    assert summary["format"] == "axiom-source-question-summary/v1"
    assert summary["id"] == QID
    assert summary["counts"] == {
        "answers": 3,
        "pairs": 3,
        "pairs_run": 3,
        "pairs_that_differ": 3,
    }
    assert [item["id"] for item in summary["answers"]] == [
        "statute",
        "regulation",
        "guidance",
    ]
    assert summary["answers"][1] == {
        "id": "regulation",
        "label": "Fixture regulation (a)",
        "document_type": "regulation",
        "jurisdiction": "us",
        "corpus_citation_path": "us/regulation/99/1/2",
        "effective_from": "2026-10-01",
        "effective_to": None,
        "module": "us/regulations/99-cfr/1/2.yaml",
    }
    assert [
        (pair["a"], pair["b"], pair["run"], pair["differ"]) for pair in summary["pairs"]
    ] == [
        ("statute", "regulation", 96, 4),
        ("statute", "guidance", 32, 3),
        ("regulation", "guidance", 192, 34),
    ]
    assert {pair["review_state"] for pair in summary["pairs"]} == {"unreviewed"}
    assert summary["pairs"][1]["witness_groups"] == [
        {
            "direction": {"a": False, "b": True},
            "count": 3,
            "conditions": [{"fact": "enrolled_student", "read_by": "b", "value": True}],
        }
    ]
    assert len(summary["warnings"]) == 1
    for name, record in records.items():
        assert record["format"] == "axiom-source-conflict-record/v1"
        assert record["review_state"] == "unreviewed"
        assert record["id"] == f"{QID}:{name.removesuffix('.json')}"
        assert record["source_question"] == {
            "id": QID,
            "sha256": load_question(QUESTION).sha256,
        }
    record = records["statute~guidance.json"]
    assert record["sides"]["b"]["answer"]["document_type"] == "federal_guidance"
    assert (
        record["sides"]["a"]["answer"]["corpus_citation_path"] == "us/statute/999/1/a"
    )
    assert record["constraints_not_applied"] == ["monthly_income >= work_hours * 7.25"]
    assert record["facts"]["work_hours"]["read_by"] == ["a", "b"]
    assert record["facts"]["work_hours"]["a"] is None


def test_the_pair_of_a_question_gives_the_record_of_the_same_link(tmp_path):
    _, records = run_question(tmp_path / "question")
    from_question = records["statute~regulation.json"]
    from_link = run_compare(tmp_path=tmp_path)
    assert from_question["counts"] == from_link["counts"]
    assert from_question["facts"] == from_link["facts"]
    assert from_question["constraints"] == from_link["constraints"]
    assert from_question["witness_groups"] == from_link["witness_groups"]
    for side in ("a", "b"):
        for key in ("module", "module_sha256", "outputs", "combine", "guards", "rules"):
            assert from_question["sides"][side][key] == from_link["sides"][side][key]


def test_conditions_are_the_one_side_facts_with_one_value_in_the_witnesses(tmp_path):
    _, records = run_question(tmp_path)
    (group,) = records["statute~regulation.json"]["witness_groups"]
    assert group["conditions"] == [
        {"fact": "monthly_income", "read_by": "b", "value": "nonzero"},
        {"fact": "agency_counts_income_as_hours", "read_by": "b", "value": True},
    ]
    groups = records["regulation~guidance.json"]["witness_groups"]
    assert len(groups) == 5
    yes_no = [g for g in groups if g["direction"] == {"a": False, "b": True}]
    # The guidance says met for a student whatever the agency does: the
    # agency fact takes both values there, so it is no condition.
    assert len(yes_no) == 4
    for group in yes_no:
        assert group["conditions"] == [
            {"fact": "enrolled_student", "read_by": "b", "value": True}
        ]
    (other,) = [g for g in groups if g["direction"] == {"a": True, "b": False}]
    assert other["conditions"] == [
        {"fact": "monthly_income", "read_by": "a", "value": "nonzero"},
        {"fact": "agency_counts_income_as_hours", "read_by": "a", "value": True},
        {"fact": "enrolled_student", "read_by": "b", "value": False},
    ]
    assert sum(group["count"] for group in groups) == 34


def test_link_records_carry_conditions_too(tmp_path):
    (group,) = run_compare(tmp_path=tmp_path)["witness_groups"]
    assert group["conditions"] == [
        {"fact": "monthly_income", "read_by": "b", "value": "nonzero"},
        {"fact": "agency_counts_income_as_hours", "read_by": "b", "value": True},
    ]


def test_a_gate_failure_of_one_pair_fails_the_question(tmp_path):
    doc = document()
    answer(doc, "guidance")["outputs"] = [C + "no_such_rule"]
    with pytest.raises(LinkGateError, match="no_such_rule"):
        run_question(tmp_path, doc)
    doc = document()
    del answer(doc, "regulation")["fixed"][B + "input.minimum_wage_rate"]
    with pytest.raises(
        LinkGateError,
        match="answers.regulation: input .*minimum_wage_rate' is read by its rules",
    ):
        run_question(tmp_path, doc)


def test_question_output_is_byte_identical_for_the_same_inputs(tmp_path):
    first = run_question(tmp_path / "one")
    reordered = document()
    reordered["answers"].reverse()
    second = run_question(tmp_path / "two", reordered)
    assert render_record(first[0]) == render_record(second[0])
    assert {name: render_record(record) for name, record in first[1].items()} == {
        name: render_record(record) for name, record in second[1].items()
    }


def test_hr1_question_expands_to_the_hr1_example_link():
    question = load_question(HR1_QUESTION)
    link = load_link(HR1_LINK)
    assert question.id == "us:medicaid/community-engagement-met-for-month"
    (pair,) = question.pairs
    assert pair.id == f"{question.id}:42-usc-1396a-xx~42-cfr-435-552"
    assert (pair.a.document_type, pair.b.document_type) == ("statute", "regulation")
    assert pair.b.corpus_citation_path == "us/regulation/42/435/552"
    for field in (
        "entity",
        "period",
        "sides",
        "facts",
        "derived",
        "fixed",
        "constraints",
    ):
        assert getattr(pair.link, field) == getattr(link, field), field
    assert (pair.link.max_combinations, pair.link.seed) == (
        link.max_combinations,
        link.seed,
    )
    assert pair.constraints_not_applied == ()


# -- CLI --------------------------------------------------------------------


@pytest.fixture
def cli_engine(monkeypatch):
    monkeypatch.setattr(
        compare_cli, "bind_engine", lambda checkout, commit: Path("/nonexistent")
    )
    monkeypatch.setattr(compare_cli, "Engine", lambda binary, commit: engine())
    monkeypatch.setattr(
        compare_cli, "pinned_engine_commit", lambda root: ("c" * 40, "toolchain")
    )


def cli_args(tmp_path, *source):
    return [
        *source,
        "--rulespec-root",
        str(RULESPEC),
        "--axiom-rules-engine-path",
        str(tmp_path),
        "--out",
        str(tmp_path / "out"),
    ]


def test_cli_question_writes_a_summary_and_one_record_per_pair(
    tmp_path, capsys, cli_engine
):
    assert compare_cli.run(cli_args(tmp_path, "--question", str(QUESTION))) == 0
    out = tmp_path / "out"
    assert sorted(path.name for path in out.iterdir()) == [
        "regulation~guidance.json",
        "statute~guidance.json",
        "statute~regulation.json",
        "summary.json",
    ]
    summary = json.loads((out / "summary.json").read_text())
    assert [pair["record"] for pair in summary["pairs"]] == [
        "statute~regulation.json",
        "statute~guidance.json",
        "regulation~guidance.json",
    ]
    captured = capsys.readouterr()
    assert "3 answers, 3 pairs, 3 pairs differ" in captured.out
    assert "statute ~ guidance: 32 run, 3 differ, 1 witness groups" in captured.out
    assert "when enrolled_student = true" in captured.out
    assert "is not applied" in captured.err


def test_cli_question_gate_failure_exits_one_and_writes_nothing(
    tmp_path, capsys, cli_engine
):
    doc = document()
    answer(doc, "guidance")["outputs"] = [C + "no_such_rule"]
    path = tmp_path / "question.json"
    path.write_text(json.dumps(doc))
    assert compare_cli.run(cli_args(tmp_path, "--question", str(path))) == 1
    assert "the link failed its gates" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
    doc = document()
    doc["answers"] = doc["answers"][:1]
    path.write_text(json.dumps(doc))
    assert compare_cli.run(cli_args(tmp_path, "--question", str(path))) == 1
    assert "two or more answers" in capsys.readouterr().err


def test_cli_needs_exactly_one_of_link_and_question(tmp_path, capsys):
    with pytest.raises(SystemExit) as caught:
        compare_cli.run(cli_args(tmp_path))
    assert caught.value.code == 2
    with pytest.raises(SystemExit):
        compare_cli.run(
            cli_args(tmp_path, "--link", str(LINK), "--question", str(QUESTION))
        )
    assert "not allowed with argument" in capsys.readouterr().err


# -- the real engine --------------------------------------------------------


def test_real_engine_gives_the_question_records_the_fixture_artifacts_give(
    real_engine, tmp_path
):
    checkout = tmp_path / "rulespec-us"
    shutil.copytree(RULESPEC, checkout)
    workdir = tmp_path / "work"
    workdir.mkdir()
    summary, records = compare_question(
        load_question(QUESTION),
        rulespec_root=checkout,
        engine=Engine(real_engine.binary, "f" * 40),
        workdir=workdir,
        repository=REPOSITORY,
    )
    expected_summary, expected = run_question(tmp_path / "fake")
    assert summary == expected_summary
    for record in (*records.values(), *expected.values()):
        record["engine"]["version"] = None
    assert records == expected


# -- findings of the review -------------------------------------------------


@pytest.mark.parametrize("module", [5, ["x.yaml"], None])
def test_an_answer_with_a_bad_module_is_refused_not_dropped(module):
    doc = document()
    doc["answers"][1]["module"] = module
    with pytest.raises(
        LinkError, match="answers.regulation.module must be a module path"
    ):
        parse_question(doc)
    two = document()
    two["answers"] = two["answers"][:2]
    two["answers"][1]["module"] = module
    with pytest.raises(LinkError, match="module must be a module path"):
        parse_question(two)


def test_an_answer_id_has_a_length_limit():
    doc = document()
    doc["answers"][1]["id"] = "r" * 65
    with pytest.raises(LinkError, match="at most 64 characters"):
        parse_question(doc)


def test_date_warnings_do_not_depend_on_the_order_of_the_answers():
    doc = document()
    answer(doc, "regulation")["effective_from"] = "2027-06-01"
    answer(doc, "guidance")["effective_to"] = "2026-06-01"
    reversed_doc = copy.deepcopy(doc)
    reversed_doc["answers"].reverse()
    first, second = parse_question(doc), parse_question(reversed_doc)
    assert len(first.warnings) == 3
    assert first.warnings == second.warnings


def test_a_gate_failure_names_the_pair_and_its_answers(tmp_path):
    doc = document()
    answer(doc, "guidance")["outputs"] = [C + "no_such_rule"]
    with pytest.raises(LinkGateError) as caught:
        run_question(tmp_path, doc)
    assert caught.value.problems[0].startswith(
        "pair statute~guidance (a = statute, b = guidance): b.outputs:"
    )


def test_an_answer_cannot_lean_on_an_input_its_partner_supplies(tmp_path):
    doc = document()
    doc["answers"] = [answer(doc, "statute"), answer(doc, "regulation")]
    del doc["facts"]["enrolled_student"]
    name = B + "input.minimum_wage_rate"
    answer(doc, "statute")["fixed"][name] = answer(doc, "regulation")["fixed"].pop(name)
    with pytest.raises(LinkGateError) as caught:
        run_question(tmp_path, doc)
    assert caught.value.problems == [
        f"answers.statute.fixed: '{name}' is not an input of its module "
        "us/statutes/999/1/a.yaml",
        f"answers.regulation: input '{name}' is read by its rules, but the answer "
        "does not supply it (inputs, derived or fixed)",
    ]


def test_each_answer_supplies_its_own_inputs_also_when_a_partner_could(tmp_path):
    # The statute's module also accepts the regulation's wage input (a shared
    # import), and only the statute fixes it: the regulation must not lean on it.
    name = B + "input.minimum_wage_rate"
    payloads = {
        "us/statutes/999/1/a.yaml": artifact_payload("a"),
        "us/regulations/99-cfr/1/2.yaml": artifact_payload("b"),
    }
    payloads["us/statutes/999/1/a.yaml"]["metadata"]["input_catalog"].append(
        {"slot": "shared_wage", "canonical_request_name": name, "request_names": [name]}
    )
    doc = document()
    doc["answers"] = [answer(doc, "statute"), answer(doc, "regulation")]
    del doc["facts"]["enrolled_student"]
    answer(doc, "statute")["fixed"][name] = answer(doc, "regulation")["fixed"].pop(name)
    with pytest.raises(LinkGateError) as caught:
        compare_question(
            parse_question(doc),
            rulespec_root=RULESPEC,
            engine=FakeEngine(payloads),
            workdir=tmp_path,
            repository=REPOSITORY,
        )
    assert caught.value.problems == [
        f"answers.regulation: input '{name}' is read by its rules, but the answer "
        "does not supply it (inputs, derived or fixed)"
    ]


def test_cli_question_out_must_not_be_a_file_and_old_records_are_removed(
    tmp_path, capsys, cli_engine
):
    out = tmp_path / "out"
    out.write_text("x")
    assert compare_cli.run(cli_args(tmp_path, "--question", str(QUESTION))) == 1
    assert "--out is a directory" in capsys.readouterr().err
    out.unlink()
    assert compare_cli.run(cli_args(tmp_path, "--question", str(QUESTION))) == 0
    (out / "notes.txt").write_text("keep")
    doc = document()
    doc["answers"] = [answer(doc, "statute"), answer(doc, "regulation")]
    del doc["facts"]["enrolled_student"]
    path = tmp_path / "two.json"
    path.write_text(json.dumps(doc))
    assert compare_cli.run(cli_args(tmp_path, "--question", str(path))) == 0
    assert sorted(item.name for item in out.iterdir()) == [
        "notes.txt",
        "statute~regulation.json",
        "summary.json",
    ]


# -- question files are not modules -----------------------------------------


def _checkout_with_a_question(tmp_path: Path) -> Path:
    checkout = tmp_path / "rulespec-us"
    shutil.copytree(RULESPEC, checkout)
    subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True)
    questions = checkout / "us" / "questions" / "medicaid"
    questions.mkdir(parents=True)
    shutil.copyfile(QUESTION, questions / "work-requirement.question.yaml")
    return checkout


def test_module_discovery_skips_question_files(tmp_path):
    from axiom_encode import cli
    from axiom_encode.constants import is_rulespec_question_path

    checkout = _checkout_with_a_question(tmp_path)
    found = cli._rulespec_module_paths(checkout)
    assert [path.relative_to(checkout).as_posix() for path in found] == [
        "us/policies/demo-agency/work-requirement-faq.yaml",
        "us/regulations/99-cfr/1/2.yaml",
        "us/regulations/99-cfr/1/3.yaml",
        "us/statutes/999/1/a.yaml",
        "us/statutes/999/2/a.yaml",
    ]
    assert cli._proof_atom_kind_normalization_module_paths(checkout) == found
    # Anything else that is YAML outside the module roots is still refused.
    stray = checkout / "us" / "notes" / "stray.yaml"
    stray.parent.mkdir()
    stray.write_text("format: rulespec/v1\n")
    with pytest.raises(ValueError, match="canonical filesystem root"):
        cli._rulespec_module_paths(checkout)
    assert is_rulespec_question_path(("questions", "m", "x.question.yaml"))
    assert not is_rulespec_question_path(("questions", "x.yaml"))
    assert not is_rulespec_question_path(("statutes", "42", "a.question.yaml"))
    assert not is_rulespec_question_path(("statutes", "42", "questions.yaml"))
    assert not is_rulespec_question_path(("statutes", "questions", "a.yaml"))
    assert not is_rulespec_question_path(())


def test_a_question_file_in_a_module_root_is_refused(tmp_path):
    from axiom_encode import cli

    checkout = tmp_path / "rulespec-us"
    shutil.copytree(RULESPEC, checkout)
    subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True)
    beside = checkout / "us" / "statutes" / "999" / "1" / "a.question.yaml"
    shutil.copyfile(QUESTION, beside)
    with pytest.raises(ValueError, match="must be under <jurisdiction>/questions/"):
        cli._rulespec_module_paths(checkout)
    # The proof-kind scan reads the module roots only; a file there is a module.
    assert beside in cli._proof_atom_kind_normalization_module_paths(checkout)


def test_only_question_files_are_skipped_in_the_questions_tree(tmp_path):
    from axiom_encode import cli
    from axiom_encode.oracles.policyengine import snap_readiness

    checkout = _checkout_with_a_question(tmp_path)
    folder = checkout / "us" / "questions" / "medicaid"
    module = folder / "real.yaml"
    module.write_text((RULESPEC / "us/statutes/999/1/a.yaml").read_text())
    with pytest.raises(ValueError, match="canonical filesystem root"):
        cli._rulespec_module_paths(checkout)
    module.unlink()
    (folder / "x.question.yml").write_text("a: 1\n")
    with pytest.raises(ValueError, match="exact .yaml extension"):
        cli._rulespec_module_paths(checkout)
    (folder / "x.question.yml").unlink()
    (folder / "link.question.yaml").symlink_to(
        folder / "work-requirement.question.yaml"
    )
    with pytest.raises(ValueError, match="must not contain symlinks"):
        cli._rulespec_module_paths(checkout)
    with pytest.raises(
        snap_readiness.SnapReadinessConfigurationError, match="symlinks"
    ):
        snap_readiness._iter_snap_rulespec_payloads(checkout / "us")


def test_snap_readiness_scan_skips_question_files(tmp_path):
    from axiom_encode.oracles.policyengine import snap_readiness

    checkout = _checkout_with_a_question(tmp_path)
    assert snap_readiness._iter_snap_rulespec_payloads(checkout / "us") == []


def test_the_compare_step_reads_a_question_file_from_the_questions_tree(tmp_path):
    checkout = _checkout_with_a_question(tmp_path)
    path = checkout / "us" / "questions" / "medicaid" / "work-requirement.question.yaml"
    summary, _ = compare_question(
        load_question(path),
        rulespec_root=checkout,
        engine=engine(),
        workdir=tmp_path / "work",
        repository=REPOSITORY,
    )
    assert summary["counts"]["pairs_that_differ"] == 3


def test_cli_question_output_is_written_safely(tmp_path, capsys, cli_engine):
    out = tmp_path / "out"
    target = tmp_path / "precious.json"
    target.write_text("keep")
    out.mkdir()
    (out / "my~notes.json").write_text("keep")
    (out / "statute~guidance.json").symlink_to(target)
    assert compare_cli.run(cli_args(tmp_path, "--question", str(QUESTION))) == 1
    assert "is not a regular file" in capsys.readouterr().err
    assert target.read_text() == "keep"
    assert not (out / "summary.json").exists()
    (out / "statute~guidance.json").unlink()
    assert compare_cli.run(cli_args(tmp_path, "--question", str(QUESTION))) == 0
    doc = document()
    doc["answers"] = [answer(doc, "statute"), answer(doc, "regulation")]
    del doc["facts"]["enrolled_student"]
    path = tmp_path / "two.json"
    path.write_text(json.dumps(doc))
    assert compare_cli.run(cli_args(tmp_path, "--question", str(path))) == 0
    # Only the records the earlier summary listed are removed.
    assert sorted(item.name for item in out.iterdir()) == [
        "my~notes.json",
        "statute~regulation.json",
        "summary.json",
    ]
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "nowhere")
    args = cli_args(tmp_path, "--question", str(QUESTION))
    args[args.index("--out") + 1] = str(dangling)
    assert compare_cli.run(args) == 1
    assert "--out is a directory" in capsys.readouterr().err
