"""The registry of reviewed alternative lists is strict, literal and digest-bound.

See ``docs/reviewed-alternative-lists.md``.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import json
import tomllib
from importlib.resources import files
from pathlib import Path

import pytest

from axiom_encode.harness import reviewed_alternative_lists as registry

ROOT = Path(__file__).parent.parent
REPLAY = Path(__file__).parent / "fixtures" / "us_replay_benchmark"


def _committed_text() -> str:
    return (
        files("axiom_encode.harness")
        .joinpath(registry.REGISTRY_PATH)
        .read_text(encoding="utf-8")
    )


def _valid() -> dict:
    return json.loads(_committed_text().splitlines()[-1])


def _replay_sources(citation: str) -> list[str]:
    """Return the unit texts of the replay fixtures, as the check scans them."""

    return [
        registry.scanned_source_text((path.parent / "source.txt").read_text())
        for path in sorted(REPLAY.glob("*/fixture.json"))
        if json.loads(path.read_text())["citation"] == citation
    ]


def test_the_committed_registry_is_valid_and_canonical():
    registry.load_reviewed_alternative_lists.cache_clear()
    entries = registry.load_reviewed_alternative_lists()
    assert entries == registry.parse_reviewed_alternative_lists(_committed_text())
    assert [entry.citation_path.rsplit("/", 1)[1] for entry in entries] == [
        "block-14",
        "block-3",
        "block-4",
    ]
    # One compact JSON object per line, in the dataclass field order.
    for line, entry in zip(_committed_text().splitlines(), entries, strict=True):
        value = json.loads(line)
        assert list(value) == list(entry.__dataclass_fields__)
        assert line == json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def test_the_registry_file_is_packaged():
    wheel = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["hatch"][
        "build"
    ]["targets"]["wheel"]
    assert "src/axiom_encode/harness/" + registry.REGISTRY_PATH in wheel["artifacts"]


def test_the_digest_covers_the_text_that_the_check_scans(monkeypatch, capsys):
    """The helper, the script and the check normalise the unit text alike."""

    import functools

    import yaml

    from axiom_encode.harness import source_completeness
    from axiom_encode.harness import validator_pipeline as vp

    entry = registry.load_reviewed_alternative_lists()[0]
    folder = next(
        path.parent
        for path in sorted(REPLAY.glob("*/fixture.json"))
        if json.loads(path.read_text())["citation"] == entry.citation_path
    )
    raw = "\n " + (folder / "source.txt").read_text() + " \n\n"
    scanned = []
    lookup = source_completeness.reviewed_alternative_lists_for

    def spy(citation, text):
        scanned.append(text)
        return lookup(citation, text)

    monkeypatch.setattr(source_completeness, "reviewed_alternative_lists_for", spy)
    profile = vp._numeric_profile_for_citation_path(entry.citation_path)
    source_completeness.analyze_complete_source_unit(
        (folder / "candidate.yaml").read_text(),
        raw,
        corpus_citation_path=entry.citation_path,
        test_cases=yaml.safe_load((folder / "candidate.test.yaml").read_text()),
        extract_numeric_occurrences=functools.partial(
            vp.extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            vp.extract_typed_numeric_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=vp.extract_named_scalar_occurrences,
        numeric_value_is_grounded=vp.numeric_value_is_grounded,
    )
    assert scanned and set(scanned) == {registry.scanned_source_text(raw)}
    assert registry.text_sha256(scanned[0]) == entry.source_text_sha256
    assert registry.text_sha256(raw) != entry.source_text_sha256

    assert registry.main([str(folder / "source.txt")]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == f"source_text_sha256 {entry.source_text_sha256}"
    offsets = dict(line.split("\t", 1) for line in printed[1:])
    assert json.loads(offsets[str(entry.chapeau.physical_start)]) == entry.chapeau.line
    for line in entry.lines:
        assert json.loads(offsets[str(line.physical_start)]) == line.line
    script = (ROOT / "scripts" / "reviewed_alternative_list_digest.py").read_text()
    assert "from axiom_encode.harness.reviewed_alternative_lists import main" in script


OWNER_TEST = (
    "tests/test_reviewed_alternative_lists.py::"
    "test_every_entry_has_an_owner_review_outside_a_draft"
)


def test_every_entry_has_an_owner_review_outside_a_draft():
    """A code owner signs each entry before the PR is ready.

    A local run and a draft pull request may hold pending entries, so that the
    author can show the effect. Every other run in GitHub Actions may not. A
    signed entry must name a code owner of the registry file, with a date that
    is not in the future, in every run.
    """

    entries = registry.load_reviewed_alternative_lists()
    for entry in entries:
        assert entry.author.strip()
        assert entry.not_examined and entry.owner_must_verify
    problems = registry.owner_review_problems(
        entries,
        codeowners_text=(ROOT / ".github" / "CODEOWNERS").read_text(),
        today=datetime.date.today(),
        review_required=registry.owner_review_is_required(),
    )
    assert problems == []


def test_an_owner_review_names_a_code_owner_and_a_past_date():
    codeowners = (ROOT / ".github" / "CODEOWNERS").read_text()
    owners = registry.registry_code_owners(codeowners)
    assert owners == {"@PavelMakarchuk", "@MaxGhenis"}
    today = datetime.date(2026, 11, 14)
    pending = registry.load_reviewed_alternative_lists()[0]
    assert pending.reviewer == registry.PENDING_REVIEWER

    def problems(entry, *, required=True, text=codeowners):
        return registry.owner_review_problems(
            (entry,), codeowners_text=text, today=today, review_required=required
        )

    def signed(reviewer, reviewed_at):
        return dataclasses.replace(pending, reviewer=reviewer, reviewed_at=reviewed_at)

    assert problems(pending, required=False) == []
    assert len(problems(pending)) == 1
    for owner in sorted(owners):
        assert problems(signed(owner, "2026-11-14")) == []
        assert problems(signed(owner, "2026-11-01"), required=False) == []
        # A date in the future is not a review.
        assert len(problems(signed(owner, "2026-11-15"))) == 1
        assert len(problems(signed(owner, "2999-01-01"), required=False)) == 1
        assert len(problems(signed(owner, None))) == 1
    # A login that is not a code owner of the registry file is not a review,
    # in a draft too.
    for stranger in (
        "@x",
        "@claude-fable-5-2",
        "@pending-owner-review",
        "@pavelmakarchuk",
    ):
        entry = signed(stranger, "2026-11-14")
        assert len(problems(entry)) == 1
        assert len(problems(entry, required=False)) == 1
    # Without a CODEOWNERS line for the registry file nobody is an owner.
    owner = sorted(owners)[0]
    for text in (
        "",
        "/src/axiom_encode/harness/reviewed_alternative_lists.py " + owner + "\n",
        "# /src/axiom_encode/harness/" + registry.REGISTRY_PATH + " " + owner + "\n",
        "/src/axiom_encode/harness/" + registry.REGISTRY_PATH + "\n",
    ):
        assert registry.registry_code_owners(text) == frozenset()
        assert len(problems(signed(owner, "2026-11-14"), text=text)) == 1
    # The last matching line wins, and a trailing comment is not an owner.
    path = "/src/axiom_encode/harness/" + registry.REGISTRY_PATH
    assert registry.registry_code_owners(
        f"{path} @first\n{path} @second @third # @fourth\n"
    ) == {"@second", "@third"}


def test_a_registry_line_with_a_repeated_key_is_rejected():
    line = _committed_text().splitlines()[0]
    assert line.startswith('{"citation_path":')
    repeated = '{"reason":"first",' + line[1:]
    assert json.loads(repeated)["reason"] == json.loads(line)["reason"]
    with pytest.raises(ValueError, match="repeats a JSON key"):
        registry.parse_reviewed_alternative_lists(repeated)
    # A nested object too.
    nested = line.replace('"chapeau":{', '"chapeau":{"physical_start":0,', 1)
    assert nested != line
    with pytest.raises(ValueError, match="repeats a JSON key"):
        registry.parse_reviewed_alternative_lists(nested)


def _event(tmp_path, payload) -> str:
    path = tmp_path / "event.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    return str(path)


@pytest.mark.parametrize(
    ("environ", "payload", "required"),
    [
        # A local run.
        ({}, None, False),
        ({"GITHUB_ACTIONS": "false"}, None, False),
        # GitHub Actions: every event but a draft pull request is gated. A
        # dispatched run attaches its checks to the branch head.
        ({"GITHUB_ACTIONS": "true"}, None, True),
        (
            {"GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "workflow_dispatch"},
            None,
            True,
        ),
        (
            {
                "GITHUB_ACTIONS": "true",
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "GITHUB_REF": "refs/heads/main",
            },
            None,
            True,
        ),
        (
            {
                "GITHUB_ACTIONS": "true",
                "GITHUB_EVENT_NAME": "workflow_dispatch",
                "GITHUB_REF": "refs/heads/fix/branch",
            },
            {"pull_request": {"draft": True}},
            True,
        ),
        ({"GITHUB_EVENT_NAME": "workflow_dispatch"}, None, True),
        ({"GITHUB_EVENT_NAME": "schedule"}, None, True),
        (
            {"GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "pull_request"},
            {"pull_request": {"draft": True}},
            False,
        ),
        (
            {"GITHUB_ACTIONS": "true", "GITHUB_EVENT_NAME": "pull_request"},
            {"pull_request": {"draft": False}},
            True,
        ),
        # Pull requests: only a draft may wait.
        (
            {"GITHUB_EVENT_NAME": "pull_request"},
            {"pull_request": {"draft": True}},
            False,
        ),
        (
            {"GITHUB_EVENT_NAME": "pull_request"},
            {"pull_request": {"draft": False}},
            True,
        ),
        ({"GITHUB_EVENT_NAME": "pull_request"}, {"pull_request": {}}, True),
        (
            {"GITHUB_EVENT_NAME": "pull_request"},
            {"pull_request": {"draft": "true"}},
            True,
        ),
        ({"GITHUB_EVENT_NAME": "pull_request"}, {}, True),
        ({"GITHUB_EVENT_NAME": "pull_request"}, "{not json", True),
        ({"GITHUB_EVENT_NAME": "pull_request"}, None, True),
        (
            {"GITHUB_EVENT_NAME": "pull_request_target"},
            {"pull_request": {"draft": False}},
            True,
        ),
        # The merge queue and main.
        ({"GITHUB_EVENT_NAME": "merge_group"}, None, True),
        ({"GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/main"}, None, True),
        (
            {"GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/fix/branch"},
            None,
            True,
        ),
        ({"GITHUB_EVENT_NAME": "push"}, None, True),
    ],
)
def test_the_owner_review_gate_follows_the_draft_gate(
    tmp_path, environ, payload, required
):
    environ = dict(environ)
    if payload is not None:
        environ["GITHUB_EVENT_PATH"] = _event(tmp_path, payload)
    assert registry.owner_review_is_required(environ) is required


def test_the_owner_review_gate_reads_the_process_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv(
        "GITHUB_EVENT_PATH", _event(tmp_path, {"pull_request": {"draft": False}})
    )
    assert registry.owner_review_is_required() is True
    monkeypatch.setenv(
        "GITHUB_EVENT_PATH", _event(tmp_path, {"pull_request": {"draft": True}})
    )
    assert registry.owner_review_is_required() is False
    monkeypatch.delenv("GITHUB_EVENT_NAME")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert registry.owner_review_is_required() is False
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert registry.owner_review_is_required() is True


def test_the_draft_gate_of_ci_reads_the_same_event_field():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert (
        "github.event_name != 'pull_request' || !github.event.pull_request.draft"
        in (workflow)
    )


def test_the_registry_and_its_loader_have_code_owners():
    owners = (ROOT / ".github" / "CODEOWNERS").read_text().splitlines()
    for path in (
        "/src/axiom_encode/harness/" + registry.REGISTRY_PATH,
        "/src/axiom_encode/harness/reviewed_alternative_lists.py",
        "/tests/test_reviewed_alternative_lists.py",
    ):
        (line,) = [line for line in owners if line.split(" ")[0] == path]
        assert line.split(" ")[1:] == ["@PavelMakarchuk", "@MaxGhenis"]


def test_every_entry_is_bound_to_the_text_of_a_replay_fixture():
    for entry in registry.load_reviewed_alternative_lists():
        sources = _replay_sources(entry.citation_path)
        assert sources, entry.citation_path
        for source in sources:
            assert registry.text_sha256(source) == entry.source_text_sha256
            assert entry.matches(entry.citation_path, source)
            assert registry.reviewed_alternative_lists_for(
                entry.citation_path, source
            ) == (entry,)
            # The recorded lines are every nonblank line from the chapeau on,
            # up to the last recorded line.
            span = source[entry.chapeau.physical_start : entry.lines[-1].physical_end]
            assert [line for line in span.split("\n") if line.strip()] == [
                entry.chapeau.line,
                *(line.line for line in entry.lines),
            ]


def _set(path: tuple, value):
    def change(entry: dict) -> None:
        target = entry
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return change


def _drop(path: tuple):
    def change(entry: dict) -> None:
        target = entry
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return change


MALFORMED = {
    "missing field": _drop(("reason",)),
    "missing reviewer": _drop(("reviewer",)),
    "missing author": _drop(("author",)),
    "blank author": _set(("author",), " "),
    "reviewer is free text": _set(("reviewer",), "an owner"),
    "reviewer login without a date": _set(("reviewer",), "@PavelMakarchuk"),
    "reviewer login with a bad name": lambda entry: entry.update(
        reviewer="@not a login", reviewed_at="2026-11-13"
    ),
    "pending review with a date": _set(("reviewed_at",), "2026-11-13"),
    "outside line before the list": lambda entry: entry["lines"][0].__setitem__(
        "reading", "outside_condition"
    ),
    "outside line inside the list": lambda entry: entry.__setitem__(
        "lines",
        [
            entry["lines"][0],
            {
                "physical_start": entry["lines"][0]["physical_start"]
                + len(entry["lines"][0]["line"])
                + 1,
                "line": "x",
                "reading": "outside_condition",
            },
            entry["lines"][1],
        ],
    ),
    "one alternative": lambda entry: entry.__setitem__("lines", entry["lines"][:1]),
    "one alternative and outside lines": lambda entry: entry["lines"][1].__setitem__(
        "reading", "outside_condition"
    ),
    "unknown field": _set(("note",), "x"),
    "wildcard citation": _set(("citation_path",), "us-az/manual/*"),
    "empty citation": _set(("citation_path",), ""),
    "citation not a string": _set(("citation_path",), 3),
    "short digest": _set(("source_text_sha256",), "abc"),
    "upper-case digest": _set(("source_text_sha256",), "A" * 64),
    "chapeau not an object": _set(("chapeau",), "When"),
    "chapeau with a reading": _set(("chapeau", "reading"), "alternative"),
    "chapeau without a line": _drop(("chapeau", "line")),
    "chapeau of two lines": _set(("chapeau", "line"), "a\nb"),
    "blank chapeau": _set(("chapeau", "line"), "  "),
    "negative offset": _set(("chapeau", "physical_start"), -1),
    "boolean offset": _set(("chapeau", "physical_start"), True),
    "text offset": _set(("lines", 0, "physical_start"), "12"),
    "lines not a list": _set(("lines",), {"0": "x"}),
    "line without a reading": _drop(("lines", 0, "reading")),
    "unknown reading": _set(("lines", 0, "reading"), "disjunct"),
    "line with an unknown field": _set(("lines", 0, "note"), "x"),
    "line of two physical lines": _set(("lines", 0, "line"), "● a\n● b"),
    "line with a carriage return": _set(("lines", 0, "line"), "● a\r"),
    "line before the chapeau": _set(("lines", 0, "physical_start"), 0),
    "lines out of order": _set(("lines", 1, "physical_start"), 1),
    "conjunct reading": _set(("lines", 0, "reading"), "conjunct"),
    "not_examined missing": _drop(("not_examined",)),
    "not_examined empty": _set(("not_examined",), []),
    "not_examined is text": _set(("not_examined",), "the structure"),
    "not_examined is one word": _set(("not_examined",), "structure"),
    "owner_must_verify is one word": _set(("owner_must_verify",), "formula"),
    "not_examined with a blank item": _set(("not_examined",), ["x", " "]),
    "not_examined with a two-line item": _set(("not_examined",), ["a\nb"]),
    "not_examined with a number": _set(("not_examined",), [3]),
    "owner_must_verify missing": _drop(("owner_must_verify",)),
    "owner_must_verify empty": _set(("owner_must_verify",), []),
    "owner_must_verify is text": _set(("owner_must_verify",), "the formula"),
    "owner_must_verify with a blank item": _set(("owner_must_verify",), [""]),
    "only outside lines": lambda entry: [
        line.__setitem__("reading", "outside_condition") for line in entry["lines"]
    ],
    "empty reason": _set(("reason",), ""),
    "reason of two lines": _set(("reason",), "a\nb"),
    "blank reviewer": _set(("reviewer",), " "),
    "date not ISO": lambda entry: entry.update(
        reviewer="@PavelMakarchuk", reviewed_at="13 Nov 2026"
    ),
    "date not a string": lambda entry: entry.update(
        reviewer="@PavelMakarchuk", reviewed_at=20261113
    ),
}


@pytest.mark.parametrize("name", MALFORMED)
def test_a_malformed_entry_is_rejected(name):
    entry = _valid()
    registry.ReviewedAlternativeList.from_mapping(copy.deepcopy(entry))
    MALFORMED[name](entry)
    with pytest.raises((ValueError, TypeError)):
        registry.ReviewedAlternativeList.from_mapping(entry)
    with pytest.raises((ValueError, TypeError)):
        registry.parse_reviewed_alternative_lists(json.dumps(entry))


def test_an_owner_review_is_a_login_and_a_date():
    entry = _valid()
    entry.update(reviewer="@PavelMakarchuk", reviewed_at="2026-11-14")
    parsed = registry.ReviewedAlternativeList.from_mapping(entry)
    assert (parsed.reviewer, parsed.reviewed_at) == ("@PavelMakarchuk", "2026-11-14")


def test_the_outside_lines_follow_the_last_line_of_the_list():
    # The three-bullet shape of the review: a conjunctive list must not become
    # zero gates by calling its first bullet outside the condition.
    entry = json.loads(_committed_text().splitlines()[1])
    readings = [line["reading"] for line in entry["lines"]]
    assert readings[:3] == ["alternative"] * 3
    assert set(readings[3:]) == {"outside_condition"}
    # Only the two readings exist: a third list line cannot be a conjunct.
    changed = copy.deepcopy(entry)
    changed["lines"][2]["reading"] = "conjunct"
    with pytest.raises(ValueError, match="reading is unknown"):
        registry.ReviewedAlternativeList.from_mapping(changed)
    for bad in (
        ["outside_condition", "alternative", "alternative"],
        ["alternative", "outside_condition", "alternative"],
    ):
        changed = copy.deepcopy(entry)
        for line, reading in zip(changed["lines"], bad, strict=False):
            line["reading"] = reading
        with pytest.raises(ValueError, match="outside_condition"):
            registry.ReviewedAlternativeList.from_mapping(changed)


@pytest.mark.parametrize("raw", ["[]", '"x"', "{", "null", "{}"])
def test_a_line_that_is_not_an_entry_is_rejected(raw):
    with pytest.raises((ValueError, TypeError)):
        registry.parse_reviewed_alternative_lists(_committed_text() + raw + "\n")


def test_a_duplicate_entry_is_rejected():
    line = _committed_text().splitlines()[0]
    with pytest.raises(ValueError, match="duplicate"):
        registry.parse_reviewed_alternative_lists(line + "\n" + line + "\n")
    # The same key with another reading is a duplicate too.
    other = json.loads(line)
    other["reason"] = "another reading"
    with pytest.raises(ValueError, match="duplicate"):
        registry.parse_reviewed_alternative_lists(line + "\n" + json.dumps(other))


def test_a_lookup_needs_the_citation_the_digest_and_every_line_in_place():
    entry = registry.load_reviewed_alternative_lists()[0]
    (source, _) = _replay_sources(entry.citation_path)
    citation = entry.citation_path
    assert registry.reviewed_alternative_lists_for(citation, source) == (entry,)
    for other in (None, "", citation + "/", citation[:-1], citation.replace("14", "3")):
        assert registry.reviewed_alternative_lists_for(other, source) == ()
    for changed in (source + "\n", " " + source, source.replace("●", "•", 1)):
        assert registry.reviewed_alternative_lists_for(citation, changed) == ()

    # A forged digest alone is not enough: every line must stand at its offset
    # as a whole physical line.
    def bound_to(text: str, **changes):
        forged = registry.ReviewedAlternativeList(
            **{
                **{name: getattr(entry, name) for name in entry.__dataclass_fields__},
                "source_text_sha256": registry.text_sha256(text),
                **changes,
            }
        )
        return registry.reviewed_alternative_lists_for(
            citation, text, entries=(forged,)
        )

    assert bound_to(source) != ()
    assert bound_to(" " + source) == ()
    last = entry.lines[-1]
    tail = source[: last.physical_end] + " and more" + source[last.physical_end :]
    assert bound_to(tail) == ()
    head = source[: last.physical_start - 1] + " " + source[last.physical_start :]
    assert bound_to(head) == ()
    moved = registry.ReviewedLine(last.physical_start + 2, last.line[2:], last.reading)
    assert bound_to(source, lines=(*entry.lines[:-1], moved)) == ()
    chapeau = entry.chapeau
    cut = registry.ReviewedLine(chapeau.physical_start, chapeau.line[:-1], "chapeau")
    assert bound_to(source, chapeau=cut) == ()
    late = registry.ReviewedLine(
        chapeau.physical_start + 5, chapeau.line[5:], "chapeau"
    )
    assert bound_to(source, chapeau=late) == ()


def test_a_hostile_registry_line_gives_no_entry(monkeypatch):
    entry = registry.load_reviewed_alternative_lists()[0]
    (source, _) = _replay_sources(entry.citation_path)
    hostile = _committed_text() + "[" * 200_000 + "\n"
    with pytest.raises(RecursionError):
        registry.parse_reviewed_alternative_lists(hostile)
    monkeypatch.setattr(
        registry,
        "load_reviewed_alternative_lists",
        lambda: registry.parse_reviewed_alternative_lists(hostile),
    )
    assert registry.reviewed_alternative_lists_for(entry.citation_path, source) == ()


@pytest.mark.parametrize(
    "error",
    [
        ValueError("malformed"),
        OSError("unreadable"),
        TypeError("not an entry"),
        RecursionError("too deep"),
        KeyError("field"),
    ],
)
def test_an_unusable_registry_gives_no_entry(error, monkeypatch):
    entry = registry.load_reviewed_alternative_lists()[0]
    (source, _) = _replay_sources(entry.citation_path)

    def broken():
        raise error

    monkeypatch.setattr(registry, "load_reviewed_alternative_lists", broken)
    assert registry.reviewed_alternative_lists_for(entry.citation_path, source) == ()
