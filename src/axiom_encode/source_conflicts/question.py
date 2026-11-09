"""The question file: several documents that answer one question.

A link compares two modules. A question file is the hub form: it states the
question, its canonical facts and constraints once, and lists two or more
**answers**. Each answer is one document's encoding: its module, its output
rules, and the map from the canonical facts to that module's inputs. N answers
need N input maps, not N * (N - 1) / 2 links.

``parse_question`` validates the file and expands it into one link per pair of
answers. ``compare_question`` runs the pair compare, unchanged, on each of
them, and returns one conflict record per pair and a summary record.

No RuleSpec module is edited and the RuleSpec format does not change: the
"same question, several answers" concept lives in this sidecar file.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact import Artifact, ArtifactError
from .compare import CompareError, EngineLike, compare, repository_state
from .expressions import ExpressionError, parse_constraint
from .gates import LinkGateError
from .link import (
    _DATE,
    _FACT,
    _ID,
    KINDS,
    LINK_FORMAT,
    Link,
    LinkError,
    _keys,
    _names,
    _text,
    parse_link,
    read_document,
)

QUESTION_FORMAT = "axiom-source-question/v1"
SUMMARY_FORMAT = "axiom-source-question-summary/v1"
DOCUMENT_TYPES = (
    "statute",
    "regulation",
    "federal_guidance",
    "state_regulation",
    "state_guidance",
    "form_instructions",
    "third_party",
)
_ANSWER_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_JURISDICTION = re.compile(r"[a-z]{2}(?:-[a-z]{2})?")
_CITATION = re.compile(r"[a-z]{2}(?:-[a-z]{2})?(?:/[A-Za-z0-9._-]+)+")

_TOP_KEYS = {
    "format": True,
    "id": True,
    "question": True,
    "entity": True,
    "period": True,
    "facts": True,
    "constraints": False,
    "answers": True,
    "sampling": False,
}
_FACT_KEYS = {
    "kind": True,
    "values": False,
    "step": False,
    "min": False,
    "max": False,
    "note": False,
}
_ANSWER_KEYS = {
    "id": True,
    "label": True,
    "document_type": True,
    "jurisdiction": True,
    "corpus_citation_path": True,
    "effective_from": False,
    "effective_to": False,
    "module": True,
    "outputs": True,
    "combine": True,
    "inputs": False,
    "derived": False,
    "fixed": False,
    "guards": False,
    "note": False,
}
_METADATA = (
    "id",
    "label",
    "document_type",
    "jurisdiction",
    "corpus_citation_path",
    "effective_from",
    "effective_to",
    "module",
)


class QuestionError(LinkError):
    """The question file is malformed. ``problems`` lists every finding."""

    heading = "invalid source question"


@dataclass(frozen=True)
class Answer:
    """One document's answer: its metadata; the encoding is in the pair links."""

    id: str
    label: str
    document_type: str
    jurisdiction: str
    corpus_citation_path: str
    effective_from: str | None
    effective_to: str | None
    module: str
    outputs: tuple[str, ...] = ()
    guards: tuple[str, ...] = ()
    supplied: tuple[tuple[str, str], ...] = ()

    def metadata(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in _METADATA}


@dataclass(frozen=True)
class Pair:
    """Two answers and the link the question expands to for them."""

    id: str
    a: Answer
    b: Answer
    link: Link
    constraints_not_applied: tuple[str, ...]

    @property
    def record_name(self) -> str:
        return f"{self.a.id}~{self.b.id}.json"


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    entity: str
    period: Mapping[str, str]
    answers: tuple[Answer, ...]
    pairs: tuple[Pair, ...]
    warnings: tuple[str, ...]
    sha256: str


def _answer(
    index: int, value: object, fact_names: list[str], problems: list[str]
) -> tuple[Answer | None, dict[str, Any]]:
    where = f"answers[{index}]"
    raw = _keys(value, _ANSWER_KEYS, where, problems)
    if raw is None:
        return None, {}
    answer_id = raw.get("id")
    if not isinstance(answer_id, str) or not _ANSWER_ID.fullmatch(answer_id):
        problems.append(
            f"{where}.id must be a lowercase id of at most 64 characters without '~', ':' or '/', "
            "such as '42-cfr-435-552'"
        )
        answer_id = None
    else:
        where = f"answers.{answer_id}"
    label = _text(raw.get("label"), f"{where}.label", problems)
    if "module" in raw and not isinstance(raw["module"], str):
        problems.append(f"{where}.module must be a module path")
    if raw.get("document_type") not in DOCUMENT_TYPES:
        problems.append(
            f"{where}.document_type must be one of {', '.join(DOCUMENT_TYPES)}"
        )
    jurisdiction = raw.get("jurisdiction")
    if not isinstance(jurisdiction, str) or not _JURISDICTION.fullmatch(jurisdiction):
        problems.append(f"{where}.jurisdiction must be like 'us' or 'us-co'")
    citation = raw.get("corpus_citation_path")
    if not isinstance(citation, str) or not _CITATION.fullmatch(citation):
        problems.append(
            f"{where}.corpus_citation_path must be a corpus path such as "
            "'us/regulation/42/435/552'"
        )
    dates = {}
    for key in ("effective_from", "effective_to"):
        dates[key] = raw.get(key)
        if key in raw and not (isinstance(raw[key], str) and _DATE.fullmatch(raw[key])):
            problems.append(f"{where}.{key} must be a YYYY-MM-DD date")
            dates[key] = None
    if (
        dates["effective_from"]
        and dates["effective_to"]
        and dates["effective_to"] < dates["effective_from"]
    ):
        problems.append(f"{where}.effective_to is before its effective_from")

    inputs = raw.get("inputs", {})
    if not isinstance(inputs, dict):
        problems.append(f"{where}.inputs must map canonical facts to input names")
        inputs = {}
    for fact_name, input_name in inputs.items():
        if fact_name not in fact_names:
            problems.append(
                f"{where}.inputs names '{fact_name}', which is not a fact of the question"
            )
        _text(input_name, f"{where}.inputs.{fact_name}", problems)
    for key in ("derived", "fixed"):
        if not isinstance(raw.get(key, {}), dict):
            problems.append(f"{where}.{key} must be a mapping keyed by input name")
    guards = raw.get("guards", [])
    if guards != []:
        guards = list(_names(guards, f"{where}.guards", problems))
    encoding = {
        "side": {
            key: raw[key]
            for key in ("label", "module", "outputs", "combine", "note")
            if key in raw
        },
        "inputs": {
            fact: name for fact, name in inputs.items() if isinstance(name, str)
        },
        "derived": raw.get("derived") if isinstance(raw.get("derived"), dict) else {},
        "fixed": raw.get("fixed") if isinstance(raw.get("fixed"), dict) else {},
        "guards": guards,
    }
    if (
        answer_id is None
        or label is None
        or not isinstance(raw.get("module"), str)
        or raw.get("document_type") not in DOCUMENT_TYPES
        or not isinstance(jurisdiction, str)
        or not isinstance(citation, str)
    ):
        return None, encoding
    return (
        Answer(
            id=answer_id,
            label=label,
            document_type=raw["document_type"],
            jurisdiction=jurisdiction,
            corpus_citation_path=citation,
            effective_from=dates["effective_from"],
            effective_to=dates["effective_to"],
            module=raw["module"],
            outputs=tuple(
                name for name in raw.get("outputs") or () if isinstance(name, str)
            )
            if isinstance(raw.get("outputs"), list)
            else (),
            guards=tuple(guards),
            supplied=(
                *(("inputs", name) for name in encoding["inputs"].values()),
                *(
                    (group, name)
                    for group in ("derived", "fixed")
                    for name in encoding[group]
                ),
            ),
        ),
        encoding,
    )


def _facts_of(encoding: Mapping[str, Any]) -> set[str]:
    """The canonical facts an answer reads: mapped, or summed into a derived input."""

    used = set(encoding["inputs"])
    for entry in encoding["derived"].values():
        if isinstance(entry, dict) and isinstance(entry.get("of"), list):
            used.update(name for name in entry["of"] if isinstance(name, str))
    return used


def _pair_document(
    raw: Mapping[str, Any],
    encodings: Mapping[str, Mapping[str, Any]],
    constraint_facts: list[tuple[dict[str, Any], tuple[str, ...]]],
    a: str,
    b: str,
    problems: list[str],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """The link document for the pair ``a``, ``b``, and the constraints it cannot apply."""

    sides = {"a": encodings[a], "b": encodings[b]}
    used = _facts_of(sides["a"]) | _facts_of(sides["b"])
    facts: dict[str, Any] = {}
    for name, fact in raw["facts"].items():
        if name not in used:
            continue
        entry = dict(fact)
        for side, encoding in sides.items():
            if name in encoding["inputs"]:
                entry[side] = encoding["inputs"][name]
        facts[name] = entry
    merged: dict[str, dict[str, Any]] = {"derived": {}, "fixed": {}}
    for group, entries in merged.items():
        for answer_id, encoding in ((a, sides["a"]), (b, sides["b"])):
            for name, entry in encoding[group].items():
                if name in entries and entries[name] != entry:
                    problems.append(
                        f"answers {a} and {b} give different {group} definitions "
                        f"of input '{name}'"
                    )
                entries[name] = entry
    applied, dropped = [], []
    for constraint, names in constraint_facts:
        if all(name in used for name in names):
            applied.append(constraint)
        else:
            dropped.append(constraint["rule"])
    document: dict[str, Any] = {
        "format": LINK_FORMAT,
        "id": raw["id"],
        "question": raw["question"],
        "entity": raw["entity"],
        "period": raw["period"],
        "a": sides["a"]["side"],
        "b": sides["b"]["side"],
        "facts": facts,
        "derived": merged["derived"],
        "fixed": merged["fixed"],
        "constraints": applied,
    }
    guards = {
        side: encoding["guards"]
        for side, encoding in sides.items()
        if encoding["guards"]
    }
    if guards:
        document["guards"] = guards
    if "sampling" in raw:
        document["sampling"] = raw["sampling"]
    return document, tuple(dropped)


def parse_question(document: object) -> Question:
    """Validate a parsed question document and expand it into its pairs."""

    problems: list[str] = []
    raw = _keys(document, _TOP_KEYS, "question file", problems)
    if raw is None:
        raise QuestionError(problems)
    if raw.get("format") != QUESTION_FORMAT:
        problems.append(f"format must be '{QUESTION_FORMAT}'")
    question_id = raw.get("id")
    if not isinstance(question_id, str) or not _ID.fullmatch(question_id):
        problems.append(
            "id must be a stable lowercase id with a jurisdiction prefix, such as "
            "'us:medicaid/community-engagement-met-for-month'"
        )
    _text(raw.get("question"), "question", problems)

    raw_facts = raw.get("facts")
    if not isinstance(raw_facts, dict) or not raw_facts:
        problems.append("facts must be a non-empty mapping")
        raw_facts = {}
    fact_kinds: dict[str, str] = {}
    for name, fact in raw_facts.items():
        where = f"facts.{name}"
        if not isinstance(name, str) or not _FACT.fullmatch(name):
            problems.append(f"{where}: a fact name must match [a-z][a-z0-9_]*")
            continue
        entry = _keys(fact, _FACT_KEYS, where, problems)
        if entry is None:
            continue
        if entry.get("kind") in KINDS:
            fact_kinds[name] = entry["kind"]
        else:
            problems.append(f"{where}.kind must be one of {', '.join(KINDS)}")
    fact_names = list(raw_facts)

    constraint_facts: list[tuple[dict[str, Any], tuple[str, ...]]] = []
    raw_constraints = raw.get("constraints", [])
    if not isinstance(raw_constraints, list):
        problems.append("constraints must be a list")
        raw_constraints = []
    for index, constraint in enumerate(raw_constraints):
        if not isinstance(constraint, dict) or "rule" not in constraint:
            problems.append(f"constraints[{index}] must be a mapping with a 'rule'")
            continue
        try:
            names = parse_constraint(constraint["rule"], fact_kinds).facts
        except ExpressionError as error:
            problems.append(f"constraints[{index}]: {error}")
            continue
        constraint_facts.append((constraint, names))

    raw_answers = raw.get("answers")
    if not isinstance(raw_answers, list) or len(raw_answers) < 2:
        problems.append("answers must be a list of two or more answers")
        raw_answers = raw_answers if isinstance(raw_answers, list) else []
    known_problems = len(problems)
    answers: dict[str, Answer] = {}
    encodings: dict[str, dict[str, Any]] = {}
    complete = True
    for index, value in enumerate(raw_answers):
        answer, encoding = _answer(index, value, fact_names, problems)
        if answer is None:
            complete = False
            if len(problems) == known_problems:
                # Never drop an answer without a reason.
                problems.append(f"answers[{index}] is not a valid answer")
            known_problems = len(problems)
            continue
        known_problems = len(problems)
        if answer.id in answers:
            problems.append(f"the answer id '{answer.id}' appears twice")
            continue
        answers[answer.id] = answer
        encodings[answer.id] = encoding
    if complete and len(raw_answers) >= 2:
        used = set().union(*(_facts_of(encoding) for encoding in encodings.values()))
        for name in fact_names:
            if name not in used:
                problems.append(f"facts.{name} is not used by any answer")
    if problems:
        raise QuestionError(problems)

    # Answers are ordered by document type (the order of DOCUMENT_TYPES, so a
    # statute is side a of a regulation), then by id. A pair's id and sides
    # therefore do not depend on the order of the answers in the file.
    ordered = sorted(
        answers,
        key=lambda answer_id: (
            DOCUMENT_TYPES.index(answers[answer_id].document_type),
            answer_id,
        ),
    )
    pairs: list[Pair] = []
    warnings: list[str] = []
    for position, a in enumerate(ordered):
        for b in ordered[position + 1 :]:
            pair_id = f"{question_id}:{a}~{b}"
            pair_problems: list[str] = []
            link_document, dropped = _pair_document(
                raw, encodings, constraint_facts, a, b, pair_problems
            )
            try:
                link = parse_link(link_document)
            except LinkError as error:
                pair_problems.extend(error.problems)
            if pair_problems:
                problems.extend(f"pair {a}~{b}: {problem}" for problem in pair_problems)
                continue
            pairs.append(
                Pair(
                    id=pair_id,
                    a=answers[a],
                    b=answers[b],
                    link=dataclasses.replace(link, id=pair_id),
                    constraints_not_applied=dropped,
                )
            )
            warnings.extend(
                f"pair {a}~{b}: constraint '{rule}' is not applied, because "
                "neither answer reads one of its facts"
                for rule in dropped
            )
    if problems:
        raise QuestionError(list(dict.fromkeys(problems)))

    start = raw["period"]["start"]
    for answer in (answers[answer_id] for answer_id in ordered):
        if (answer.effective_from and start < answer.effective_from) or (
            answer.effective_to and start > answer.effective_to
        ):
            warnings.append(
                f"answer {answer.id}: the period start {start} is outside its "
                "effective dates"
            )
    return Question(
        id=question_id,
        question=raw["question"],
        entity=raw["entity"],
        period={key: raw["period"][key] for key in ("period_kind", "start", "end")},
        answers=tuple(answers[answer_id] for answer_id in ordered),
        pairs=tuple(pairs),
        warnings=tuple(warnings),
        sha256=hashlib.sha256(
            json.dumps(
                {
                    "format": QUESTION_FORMAT,
                    "id": question_id,
                    "answers": [answers[answer_id].metadata() for answer_id in ordered],
                    "pairs": {pair.id: pair.link.sha256 for pair in pairs},
                },
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    )


def load_question(path: Path) -> Question:
    """Read and validate a question file (``.json``, ``.yaml`` or ``.yml``)."""

    return parse_question(read_document(path, "question"))


def compare_question(
    question: Question,
    *,
    rulespec_root: Path,
    engine: EngineLike,
    workdir: Path,
    repository: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Compare every pair of answers.

    Returns the summary record and the pair records keyed by file name. A gate
    failure of any pair raises, so a question gives all of its records or none.
    """

    repository = dict(
        repository if repository is not None else repository_state(Path(rulespec_root))
    )
    # Each answer must stand alone: the inputs it names are inputs of its own
    # module, and it supplies every input its own rules read. A pair gives a
    # derived or fixed input to every side that accepts it, so without this
    # an answer could lean on an input that only its partner supplies, and the
    # result would depend on which partner it has.
    problems: list[str] = []
    for number, answer in enumerate(question.answers):
        module = Path(rulespec_root) / answer.module
        if not module.is_file():
            continue  # a missing module is reported by the pair's gates
        answer_workdir = Path(workdir) / f"answer-{number}"
        answer_workdir.mkdir(parents=True, exist_ok=True)
        try:
            artifact = Artifact(
                engine.compile(module, Path(rulespec_root), answer_workdir / "a.json"),
                question.period["start"],
            )
        except ArtifactError as error:
            raise CompareError(f"answer {answer.id}: {error}") from error
        accepted = artifact.slot_by_request_name
        problems.extend(
            f"answers.{answer.id}.{group}: '{name}' is not an input of its module "
            f"{answer.module}"
            for group, name in answer.supplied
            if name not in accepted
        )
        own = {name for _, name in answer.supplied}
        _, _, slots = artifact.closure([*answer.outputs, *answer.guards])
        problems.extend(
            f"answers.{answer.id}: input '{artifact.request_name_by_slot[slot]}' is "
            "read by its rules, but the answer does not supply it (inputs, derived "
            "or fixed)"
            for slot in slots
            if slot in artifact.request_name_by_slot
            and artifact.request_name_by_slot[slot] not in own
        )
    if problems:
        raise LinkGateError(problems)

    records: dict[str, dict[str, Any]] = {}
    pairs_summary = []
    for number, pair in enumerate(question.pairs):
        pair_workdir = Path(workdir) / f"pair-{number}"
        pair_workdir.mkdir(parents=True, exist_ok=True)
        try:
            record = compare(
                pair.link,
                rulespec_root=rulespec_root,
                engine=engine,
                workdir=pair_workdir,
                repository=repository,
            )
        except LinkGateError as error:
            raise LinkGateError(
                [
                    f"pair {pair.a.id}~{pair.b.id} (a = {pair.a.id}, b = {pair.b.id}): "
                    f"{problem}"
                    for problem in error.problems
                ]
            ) from error
        record["source_question"] = {"id": question.id, "sha256": question.sha256}
        for side, answer in (("a", pair.a), ("b", pair.b)):
            record["sides"][side]["answer"] = answer.metadata()
        record["constraints_not_applied"] = list(pair.constraints_not_applied)
        records[pair.record_name] = record
        pairs_summary.append(
            {
                "id": pair.id,
                "a": pair.a.id,
                "b": pair.b.id,
                "record": pair.record_name,
                "review_state": record["review_state"],
                "conflict": record["conflict"],
                "run": record["counts"]["run"],
                "differ": record["counts"]["differ"],
                "witness_groups": [
                    {
                        "direction": group["direction"],
                        "count": group["count"],
                        "conditions": group["conditions"],
                    }
                    for group in record["witness_groups"]
                ],
            }
        )
    summary = {
        "format": SUMMARY_FORMAT,
        "id": question.id,
        "question": question.question,
        "source_question": {"sha256": question.sha256},
        "rulespec": repository,
        "engine": {"commit": engine.commit},
        "entity": question.entity,
        "period": dict(question.period),
        "answers": [answer.metadata() for answer in question.answers],
        "counts": {
            "answers": len(question.answers),
            "pairs": len(question.pairs),
            "pairs_run": len(pairs_summary),
            "pairs_that_differ": sum(1 for pair in pairs_summary if pair["conflict"]),
        },
        "warnings": list(question.warnings),
        "pairs": pairs_summary,
    }
    return summary, records
