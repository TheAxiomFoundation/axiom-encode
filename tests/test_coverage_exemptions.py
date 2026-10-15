"""Reviewed coverage exemptions remove exactly the entries bound to source bytes."""

import hashlib
from dataclasses import replace
from importlib.resources import files

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import coverage_exemptions as ce
from axiom_encode.harness import validator_pipeline as vp

CITATION = "us-test/statute/42"
SCOPE = ("a", "2")
SOURCE = '(a) "benefit" means the amount payable.\n(2) Definitions.\n'


def _sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _mapping(source=SOURCE, demand=None, *, scope=SCOPE, **changes):
    demand = demand or ce.CoverageDemand("a", "benefit", 0)
    window = source[max(0, demand.physical_start - 80) : demand.physical_start + 80]
    result = {
        "citation_path": CITATION,
        "requested_scope": list(scope) if scope is not None else None,
        "source_text_sha256": _sha256(source),
        "label": demand.label,
        "physical_start": demand.physical_start,
        "window_sha256": _sha256(window),
        "window": window,
        "reason": "nested_under_numbered_parent",
        "owner_evidence": {
            "citation_path": CITATION + "/1",
            "physical_start": 0,
        },
        "corpus_artifact": "corpus/us-test/statute/review.jsonl",
        "artifact_sha256": "1" * 64,
        "row": 1,
        "reviewer": "coverage registry test reviewer",
        "reviewed_at": "2026-10-15",
    }
    result.update(changes)
    return result


def _entry(source=SOURCE, demand=None, **changes):
    return ce.CoverageExemption.from_mapping(_mapping(source, demand, **changes))


def _filter(demands, source=SOURCE, entries=(), *, citation=CITATION, scope=SCOPE):
    return ce.filter_coverage_demands(
        demands,
        citation_path=citation,
        requested_scope=scope,
        source_text=source,
        exemptions=entries,
    )


@st.composite
def _unconstrained_case(draw):
    # Text is unconstrained Unicode; the skeleton guarantees operative demands
    # while arbitrary text may add markup, parentheses and further outline shapes.
    chunks = draw(st.lists(st.text(max_size=150), min_size=4, max_size=4))
    source = (
        chunks[0]
        + '\n(a) "benefit" means '
        + chunks[1]
        + "\n(1) <b>Definitions</b> ("
        + chunks[2]
        + ")\n(b) A person is eligible "
        + chunks[3]
    )
    demands = vp._high_signal_top_level_coverage_demands(source)
    assert demands
    specifications = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=len(demands) - 1),
                st.sampled_from(
                    ["exact", "citation", "scope", "hash", "label", "offset", "window"]
                ),
            ),
            max_size=15,
        )
    )
    mappings = []
    for index, kind in specifications:
        demand = demands[index]
        mapping = _mapping(source, demand)
        if kind == "citation":
            mapping["citation_path"] += "/extra"
        elif kind == "scope":
            mapping["requested_scope"] = [*SCOPE, "extra"]
        elif kind == "hash":
            mapping["source_text_sha256"] = "0" * 64
        elif kind == "label":
            mapping["label"] = demand.label.upper()
        elif kind == "offset":
            mapping["physical_start"] += 1
            mapping["window"] = source[
                max(0, mapping["physical_start"] - 80) : mapping["physical_start"] + 80
            ]
            mapping["window_sha256"] = _sha256(mapping["window"])
        elif kind == "window":
            replacement = "\0" if mapping["window"][0] != "\0" else "\1"
            mapping["window"] = replacement + mapping["window"][1:]
            mapping["window_sha256"] = _sha256(mapping["window"])
        mappings.append(mapping)
    return source, demands, mappings


@given(case=_unconstrained_case())
@settings(max_examples=300, deadline=None, derandomize=True)
def test_unconstrained_unicode_removed_set_equals_random_registry_matches(case):
    source, demands, mappings = case
    assert [(d.label, d.body) for d in demands] == (
        vp._high_signal_top_level_subparagraphs(source)
    )
    entries = tuple(ce.CoverageExemption.from_mapping(item) for item in mappings)
    # Calculate the oracle from the serialized entries, independently of the
    # production match helper, index and source_window implementation.
    expected = []
    for demand in demands:
        matched = any(
            item["citation_path"] == CITATION
            and item["requested_scope"] == list(SCOPE)
            and item["source_text_sha256"] == _sha256(source)
            and item["label"] == demand.label
            and item["physical_start"] == demand.physical_start
            and item["window"]
            == source[max(0, demand.physical_start - 80) : demand.physical_start + 80]
            and item["window_sha256"] == _sha256(item["window"])
            for item in mappings
        )
        if not matched:
            expected.append(demand)
    assert _filter(demands, source, entries) == expected
    assert _filter(demands, source, ()) == demands


@given(payload=st.binary(min_size=1, max_size=100))
@settings(max_examples=100, deadline=None, derandomize=True)
def test_flipping_every_single_source_byte_restores_all_listed_demands(payload):
    # ASCII includes arbitrary control bytes and makes every one-bit mutation
    # valid UTF-8, so these are literal byte flips with no decoder normalization.
    arbitrary_ascii = bytes(byte & 127 for byte in payload).decode("ascii")
    source = '(a) "benefit" means ' + arbitrary_ascii + "\n(b) A person is eligible.\n"
    demands = vp._high_signal_top_level_coverage_demands(source)
    entries = tuple(_entry(source, demand) for demand in demands)
    assert demands
    assert _filter(demands, source, entries) == []
    original = source.encode("utf-8")
    for offset in range(len(original)):
        changed_bytes = bytearray(original)
        changed_bytes[offset] ^= 1
        changed = changed_bytes.decode("utf-8")
        assert _filter(demands, changed, entries) == demands, offset
        changed_demands = vp._high_signal_top_level_coverage_demands(changed)
        assert _filter(changed_demands, changed, entries) == changed_demands, offset


@given(text=st.text(min_size=1, max_size=150))
@settings(max_examples=100, deadline=None, derandomize=True)
def test_changed_unicode_source_bytes_restore_demands_even_outside_window(text):
    source = '(a) "benefit" means payable.\n' + " " * 200 + text
    demand = vp._high_signal_top_level_coverage_demands(source)[0]
    entry = _entry(source, demand)
    assert _filter([demand], source, [entry]) == []
    changed = source + "\N{SNOWMAN}"
    assert ce.source_window(changed, demand.physical_start) == entry.window
    assert _filter([demand], changed, [entry]) == [demand]


@pytest.mark.parametrize(
    "citation,scope",
    [
        (CITATION + "/a", SCOPE),
        (CITATION[:-1], SCOPE),
        (CITATION.upper(), SCOPE),
        (CITATION, SCOPE + ("i",)),
        (CITATION, SCOPE[:1]),
        (CITATION, ("A", "2")),
        (CITATION, None),
        (CITATION, ()),
    ],
)
def test_citation_and_scope_match_exactly_without_prefixes_or_casefold(citation, scope):
    demand = ce.CoverageDemand("a", "benefit", 0)
    assert _filter([demand], entries=[_entry()], citation=citation, scope=scope) == [
        demand
    ]


@pytest.mark.parametrize("label,start", [("A", 0), ("aa", 0), ("a", 1)])
def test_label_and_physical_offset_match_exactly(label, start):
    demand = ce.CoverageDemand(label, "benefit", start)
    assert _filter([demand], entries=[_entry()]) == [demand]


def test_literal_window_is_required_even_with_matching_window_digest():
    demand = ce.CoverageDemand("a", "benefit", 0)
    entry = _entry(window="wrong window", window_sha256=_sha256("wrong window"))
    assert _filter([demand], entries=[entry]) == [demand]


def test_window_digest_is_required_in_addition_to_literal_window():
    demand = ce.CoverageDemand("a", "benefit", 0)
    entry = replace(_entry(), window_sha256="0" * 64)
    assert _filter([demand], entries=[entry]) == [demand]


def test_duplicate_labels_only_remove_the_reviewed_physical_occurrence():
    source = '(a) "benefit" means one.\n(a) "benefit" means two.\n'
    demands = vp._high_signal_top_level_coverage_demands(source)
    assert len(demands) == 2
    assert demands[0].physical_start != demands[1].physical_start
    assert _filter(demands, source, [_entry(source, demands[0])]) == [demands[1]]


@pytest.mark.parametrize("offset", [0, 1, 79, 80, 81, 159, 160, 200])
def test_literal_window_uses_eighty_characters_on_each_side(offset):
    text = "\N{SNOWMAN}" * 200
    assert ce.source_window(text, offset) == text[max(0, offset - 80) : offset + 80]


@pytest.mark.parametrize(
    "changes",
    [
        {"reason": "inferred_from_typography"},
        {"reviewer": ""},
        {"reviewer": None},
        {"citation_path": "us-test/statute/*"},
        {"label": "?"},
        {"requested_scope": ["a", "*"]},
        {"physical_start": -1},
        {"physical_start": True},
        {"source_text_sha256": "not-a-digest"},
        {"artifact_sha256": "not-a-digest"},
        {"window_sha256": "0" * 64},
    ],
)
def test_registry_schema_rejects_unreviewed_or_nonexact_entries(changes):
    with pytest.raises(ValueError):
        _entry(**changes)


def test_packaged_registry_loads_with_reviewed_schema():
    assert (
        files("axiom_encode.harness")
        .joinpath("data/coverage_exemptions.jsonl")
        .is_file()
    )
    entries = ce.load_coverage_exemptions()
    assert entries
    assert all(entry.reviewer.strip() for entry in entries)
    assert {entry.reason for entry in entries} <= {
        "owned_by_other_provision",
        "repealed_tombstone",
        "nested_under_numbered_parent",
    }
    for entry in entries:
        assert _sha256(entry.window) == entry.window_sha256
        for key in (entry.citation_path, entry.label, *(entry.requested_scope or ())):
            assert not any(char in key for char in "*?[]")


def test_pipeline_applies_registry_to_post_scope_main_demands(monkeypatch):
    source = '(a) "benefit" means one.\n(b) A person is eligible.\n'
    demands = vp._high_signal_top_level_coverage_demands(source)
    entry = _entry(source, demands[0], scope=("a",))
    monkeypatch.setattr(ce, "load_coverage_exemptions", lambda: (entry,))
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {"source_verification": {"corpus_citation_path": CITATION}},
            "rules": [],
        }
    )
    assert (
        vp.find_source_subparagraph_coverage_issues(
            content,
            source_texts={CITATION: source},
            requested_source=CITATION + "/a",
        )
        == []
    )
    # A different requested scope cannot reuse the approved scope, even though
    # its parent text and the scanner's physical demands are identical.
    issues = vp.find_source_subparagraph_coverage_issues(
        content,
        source_texts={CITATION: source},
        requested_source=CITATION + "/b",
    )
    assert len(issues) == 1


def test_reviewed_tombstone_waiver_requires_the_existing_notice_conditions(monkeypatch):
    source = '(a) "benefit" means payable.\n(2) Repealed.\n(b) Effective date.\n'
    demand = vp._high_signal_top_level_coverage_demands(source)[0]
    entry = _entry(source, demand, reason="repealed_tombstone")
    monkeypatch.setattr(ce, "load_coverage_exemptions", lambda: (entry,))
    monkeypatch.setattr(vp, "_requested_corpus_row_is_repealed", lambda *args: False)
    assert vp._authoritative_deferred_editorial_source_slice(
        source, "(2) Repealed.", SCOPE, citation_path=CITATION
    )
    assert not vp._authoritative_deferred_editorial_source_slice(
        source, "(3) Repealed.", SCOPE, citation_path=CITATION
    )
    assert not vp._authoritative_deferred_editorial_source_slice(
        source, "(2) Operative provision.", SCOPE, citation_path=CITATION
    )
    assert not vp._authoritative_deferred_editorial_source_slice(
        source + " ", "(2) Repealed.", SCOPE, citation_path=CITATION
    )


@pytest.mark.parametrize(
    "reason", ["owned_by_other_provision", "nested_under_numbered_parent"]
)
def test_other_reviewed_exemption_reasons_cannot_authenticate_repeal(
    reason, monkeypatch
):
    source = '(a) "benefit" means payable.\n(2) Repealed.\n(b) Effective date.\n'
    demand = vp._high_signal_top_level_coverage_demands(source)[0]
    monkeypatch.setattr(
        ce, "load_coverage_exemptions", lambda: (_entry(source, demand, reason=reason),)
    )
    monkeypatch.setattr(vp, "_requested_corpus_row_is_repealed", lambda *args: False)
    assert not vp._authoritative_deferred_editorial_source_slice(
        source, "(2) Repealed.", SCOPE, citation_path=CITATION
    )


def test_trusted_repealed_status_adds_to_existing_waiver_conditions(monkeypatch):
    source = '(a) "benefit" means payable.\n(2) Repealed.\n(b) Effective date.\n'
    monkeypatch.setattr(ce, "load_coverage_exemptions", lambda: ())
    monkeypatch.setattr(vp, "_requested_corpus_row_is_repealed", lambda *args: True)
    assert vp._authoritative_deferred_editorial_source_slice(
        source, "(2) Repealed.", SCOPE, citation_path=CITATION
    )
    assert not vp._authoritative_deferred_editorial_source_slice(
        source.replace("(2) Repealed.", '(2) "benefit" means payable.'),
        "(2) Repealed.",
        SCOPE,
        citation_path=CITATION,
    )
    assert not vp._authoritative_deferred_editorial_source_slice(
        source, "(2) Operative provision.", SCOPE, citation_path=CITATION
    )
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {
                "status": "active",
                "summary": "(2) Repealed.",
                "source_verification": {"corpus_citation_path": CITATION},
            },
            "rules": [],
        }
    )
    assert vp.find_source_subparagraph_coverage_issues(
        content,
        source_texts={CITATION: source},
        requested_source=CITATION + "/a/2",
    )


def test_reviewed_tombstone_for_another_parent_cannot_authenticate_waiver(monkeypatch):
    source = (
        '(a) "benefit" means payable.\n(2) Repealed.\n'
        '(b) "other benefit" means another amount.\n'
    )
    other_parent = vp._high_signal_top_level_coverage_demands(source)[1]
    assert other_parent.label == "b"
    entry = _entry(source, other_parent, reason="repealed_tombstone")
    monkeypatch.setattr(ce, "load_coverage_exemptions", lambda: (entry,))
    monkeypatch.setattr(vp, "_requested_corpus_row_is_repealed", lambda *args: False)
    assert not vp._authoritative_deferred_editorial_source_slice(
        source, "(2) Repealed.", SCOPE, citation_path=CITATION
    )
    assert not ce.has_reviewed_repeal(
        citation_path=CITATION, requested_scope=(), source_text=source
    )
