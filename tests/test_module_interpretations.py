"""Strict, inert interpretation metadata and corpus matcher regressions."""

import json
import string
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness.proof_validator import (
    _validate_source_proof_atom,
    proof_source_citation_paths,
    validate_rulespec_proofs,
)

CITATION = "uk/manual/example/page-87"
EXCERPT = "The prescribed daily percentage"
LABEL = "Interpretation `daily-twenty-per-cent` (module.interpretations[0])"
CONTRADICTED = (
    ": publisher contradicted this reading; repair the module, "
    "move the old reading into alternatives, and set the new status"
)


def _entry(status="open"):
    entry = {
        "id": "daily-twenty-per-cent",
        "question": "Is the percentage applied once a week or each day?",
        "texts": [{"corpus_citation_path": CITATION, "excerpt": EXCERPT}],
        "reading": "Apply twenty per cent once a week.",
        "alternatives": [
            {
                "id": "daily-percentage",
                "reading": "Apply twenty per cent each day.",
                "effect": "Less support.",
            }
        ],
        "basis": "The weekly table supports this reading.",
        "status": status,
        "affects": ["weekly_taper_rate"],
    }
    if status != "open":
        entry["ruling"] = {
            "by": "Axiom Foundation",
            "date": "2026-10-10",
            "record": "https://github.com/TheAxiomFoundation/rulespec-uk/pull/449",
        }
    if status in {"confirmed", "contradicted"}:
        entry["confirmation"] = {
            "by": "Example District Council",
            "date": "2026-10-20",
            "excerpt": "The deduction is applied once a week.",
            "record": "https://example.org/published-reply",
        }
    return entry


def _payload(entry):
    return {
        "format": "rulespec/v1",
        "module": {"interpretations": [entry]},
        "rules": [
            {
                "name": "weekly_taper_rate",
                "kind": "parameter",
                "dtype": "Float",
                "versions": [{"effective_from": "2026-04-01", "formula": "0.2"}],
            }
        ],
    }


def _validate(payload, source_texts=None, **kwargs):
    return validate_rulespec_proofs(
        yaml.safe_dump(payload),
        source_texts={CITATION: EXCERPT} if source_texts is None else source_texts,
        **kwargs,
    )


def _parent(value, path):
    for component in path[:-1]:
        value = value[component]
    return value


def _field_label(path):
    return "".join(
        f"[{component}]" if isinstance(component, int) else f".{component}"
        for component in path
    )


STRING_PATHS = [
    ("id",),
    ("question",),
    ("reading",),
    ("basis",),
    ("texts", 0, "excerpt"),
    ("alternatives", 0, "id"),
    ("alternatives", 0, "reading"),
    ("alternatives", 0, "effect"),
    ("affects", 0),
    ("ruling", "by"),
    ("confirmation", "by"),
    ("confirmation", "excerpt"),
]


@pytest.mark.parametrize("status", ["open", "ruled", "confirmed"])
def test_valid_interpretation_statuses_pass_without_counting_atoms(status):
    result = _validate(_payload(_entry(status)))
    assert result.passed
    assert result.issues == []
    assert result.atoms_checked == 0


@pytest.mark.parametrize("entry_id", ["a", "0", "twenty20", "a-0-weekly"])
def test_interpretation_kebab_case_id_positive_examples(entry_id):
    entry = _entry()
    entry["id"] = entry_id
    assert _validate(_payload(entry)).passed


@pytest.mark.parametrize("value", [None, "open", {}, 1])
def test_interpretations_must_be_a_list(value):
    payload = _payload(_entry())
    payload["module"]["interpretations"] = value
    assert _validate(payload).issues == ["module.interpretations must be a list."]


def test_optional_interpretations_can_be_absent_or_empty():
    payload = _payload(_entry())
    payload["module"]["interpretations"] = []
    assert _validate(payload).passed
    del payload["module"]["interpretations"]
    assert _validate(payload).passed


@pytest.mark.parametrize("path", STRING_PATHS)
@pytest.mark.parametrize("value", [None, "", " \n\t", 7, False, [], {}])
def test_interpretation_strings_are_nonblank_and_typed(path, value):
    entry = _entry("confirmed")
    _parent(entry, path)[path[-1]] = value
    result = _validate(_payload(entry))
    expected_label = "module.interpretations[0]" if path == ("id",) else LABEL
    assert result.issues == [
        f"{expected_label}{_field_label(path)} must be a non-blank string."
    ]


@pytest.mark.parametrize("field", ["texts", "alternatives", "affects"])
@pytest.mark.parametrize("value", [None, [], "value", {}])
def test_interpretation_lists_are_nonempty(field, value):
    entry = _entry()
    entry[field] = value
    assert _validate(_payload(entry)).issues == [
        f"{LABEL}.{field} must be a non-empty list."
    ]


@pytest.mark.parametrize(
    "path", [(), ("texts", 0), ("alternatives", 0), ("ruling",), ("confirmation",)]
)
@pytest.mark.parametrize("value", [None, [], "value"])
def test_interpretation_nested_values_must_be_mappings(path, value):
    entry = _entry("confirmed")
    if path:
        _parent(entry, path)[path[-1]] = value
        expected_label = LABEL + _field_label(path)
    else:
        entry = value
        expected_label = "module.interpretations[0]"
    assert _validate(_payload(entry)).issues == [f"{expected_label} must be a mapping."]


@pytest.mark.parametrize(
    "path", [(), ("texts", 0), ("alternatives", 0), ("ruling",), ("confirmation",)]
)
def test_unknown_keys_are_rejected_at_every_interpretation_level(path):
    entry = _entry("confirmed")
    target = entry
    for component in path:
        target = target[component]
    target["unexpected"] = "value"
    assert _validate(_payload(entry)).issues == [
        f"{LABEL}{_field_label(path)} has unknown key `unexpected`."
    ]


@pytest.mark.parametrize(
    "entry_id", ["Uppercase", "two_words", "-prefix", "suffix-", "a--b", "a b", "é"]
)
def test_interpretation_id_must_match_kebab_case(entry_id):
    entry = _entry()
    entry["id"] = entry_id
    assert _validate(_payload(entry)).issues == [
        f"Interpretation `{entry_id}` (module.interpretations[0]).id "
        "must match [a-z0-9]+(-[a-z0-9]+)*."
    ]


def test_interpretation_ids_are_unique_within_module():
    payload = _payload(_entry())
    payload["module"]["interpretations"].append(_entry())
    assert _validate(payload).issues == [
        "Interpretation `daily-twenty-per-cent` (module.interpretations[1]).id "
        "duplicates interpretation `daily-twenty-per-cent`."
    ]
    payload["module"]["interpretations"][1]["id"] = "second-reading"
    assert _validate(payload).passed


@pytest.mark.parametrize("alternative_index", [0, 1, 2])
def test_interpretation_alternative_id_is_required_at_every_index(alternative_index):
    entry = _entry()
    entry["alternatives"] = [
        {"id": f"alternative-{index}", "reading": "Another reading."}
        for index in range(3)
    ]
    del entry["alternatives"][alternative_index]["id"]
    assert _validate(_payload(entry)).issues == [
        f"{LABEL}.alternatives[{alternative_index}].id must be a non-blank string."
    ]


@pytest.mark.parametrize("alternative_id", ["a", "0", "twenty20", "a-0-weekly"])
def test_interpretation_alternative_kebab_case_id_positive_examples(alternative_id):
    entry = _entry()
    entry["alternatives"][0]["id"] = alternative_id
    assert _validate(_payload(entry)).passed


@pytest.mark.parametrize("alternative_index", [0, 1, 2])
@pytest.mark.parametrize(
    "alternative_id",
    ["Uppercase", "two_words", "-prefix", "suffix-", "a--b", "a b", "é", "a\n"],
)
def test_interpretation_alternative_id_must_match_kebab_case(
    alternative_index, alternative_id
):
    entry = _entry()
    entry["alternatives"] = [
        {"id": f"alternative-{index}", "reading": "Another reading."}
        for index in range(3)
    ]
    entry["alternatives"][alternative_index]["id"] = alternative_id
    assert _validate(_payload(entry)).issues == [
        f"{LABEL}.alternatives[{alternative_index}].id "
        "must match [a-z0-9]+(-[a-z0-9]+)*."
    ]


@pytest.mark.parametrize("alternative_index", [1, 2])
def test_interpretation_alternative_ids_are_unique_within_interpretation(
    alternative_index,
):
    entry = _entry()
    entry["alternatives"] = [
        {"id": f"alternative-{index}", "reading": "Another reading."}
        for index in range(3)
    ]
    entry["alternatives"][alternative_index]["id"] = "alternative-0"
    assert _validate(_payload(entry)).issues == [
        f"{LABEL}.alternatives[{alternative_index}].id "
        "duplicates alternative `alternative-0`."
    ]


def test_interpretation_alternative_ids_can_equal_the_parent_interpretation_id():
    entry = _entry()
    entry["alternatives"][0]["id"] = entry["id"]
    assert _validate(_payload(entry)).passed


@pytest.mark.parametrize("status", [None, "", "pending", "OPEN", 1, [], {}])
def test_interpretation_status_is_an_enum(status):
    entry = _entry()
    entry["status"] = status
    assert _validate(_payload(entry)).issues == [
        f"{LABEL}.status must be open, ruled, confirmed or contradicted."
    ]


@pytest.mark.parametrize("status", ["open", "ruled", "confirmed", "contradicted"])
@pytest.mark.parametrize("field", ["ruling", "confirmation"])
def test_interpretation_status_controls_record_presence(status, field):
    entry = _entry(status)
    baseline = _validate(_payload(entry)).issues
    assert baseline == ([LABEL + CONTRADICTED] if status == "contradicted" else [])
    if field in entry:
        del entry[field]
        finding = f"{LABEL}.{field} must be a mapping when status is `{status}`."
    else:
        entry[field] = _entry("confirmed")[field]
        finding = f"{LABEL}.{field} must be absent when status is `{status}`."
    assert _validate(_payload(entry)).issues == [finding, *baseline]


@pytest.mark.parametrize("field", ["ruling", "confirmation"])
@pytest.mark.parametrize(
    "value",
    [
        "2026-2-03",
        "2026-02-30",
        "2025-02-29",
        "0000-01-01",
        "2026-10-10T12:00:00",
        datetime(2026, 10, 10),
        None,
        20261010,
    ],
)
def test_interpretation_dates_must_be_real_iso_dates(field, value):
    entry = _entry("confirmed")
    entry[field]["date"] = value
    assert _validate(_payload(entry)).issues == [
        f"{LABEL}.{field}.date must be a real YYYY-MM-DD date."
    ]


@pytest.mark.parametrize("value", ["2024-02-29", "2026-10-10", date(2026, 10, 10)])
def test_interpretation_quoted_and_unquoted_dates_pass(value):
    entry = _entry("confirmed")
    entry["ruling"]["date"] = entry["confirmation"]["date"] = value
    assert _validate(_payload(entry)).passed


@pytest.mark.parametrize("field", ["ruling", "confirmation"])
@pytest.mark.parametrize(
    "value",
    [
        "http://example.org/reply",
        "https://",
        "https:///reply",
        "https://example.org/a b",
        "https://example.org:invalid/reply",
        "https://[bad/reply",
        "reply",
        "",
        None,
        1,
    ],
)
def test_interpretation_records_require_https_urls(field, value):
    entry = _entry("confirmed")
    entry[field]["record"] = value
    assert _validate(_payload(entry)).issues == [
        f"{LABEL}.{field}.record must be an https URL."
    ]


def test_confirmation_record_and_alternative_effect_are_optional():
    entry = _entry("confirmed")
    del entry["confirmation"]["record"]
    del entry["alternatives"][0]["effect"]
    assert _validate(_payload(entry)).passed


@pytest.mark.parametrize(
    "record", ["https://example.org", "https://example.org:443/reply?year=2026#record"]
)
def test_interpretation_https_record_positive_examples(record):
    entry = _entry("confirmed")
    entry["ruling"]["record"] = entry["confirmation"]["record"] = record
    assert _validate(_payload(entry)).passed


@pytest.mark.parametrize(
    "citation",
    [
        "uk:manual/example/page-87",
        " uk/manual/example/page-87 ",
        "uk/manual/example/page-87#rule",
    ],
)
def test_interpretation_citations_require_exact_canonical_identity(citation):
    entry = _entry()
    entry["texts"][0]["corpus_citation_path"] = citation
    result = _validate(_payload(entry), {citation.strip(): EXCERPT})
    assert not result.passed
    assert len(result.issues) == 1
    assert "Noncanonical corpus citation path" in result.issues[0]
    assert "interpretations[0].texts[0].corpus_citation_path" in result.issues[0]


@pytest.mark.parametrize("citation", [None, "", " \t", 7, [], {}])
def test_interpretation_citation_must_be_a_nonblank_string(citation):
    entry = _entry()
    entry["texts"][0]["corpus_citation_path"] = citation
    result = _validate(_payload(entry))
    assert not result.passed
    assert len(result.issues) == 2  # Recursive identity gate plus local shape.
    assert "interpretations[0].texts[0].corpus_citation_path" in result.issues[0]
    assert result.issues[1] == (
        f"{LABEL}.texts[0].corpus_citation_path must be a non-blank string."
    )


@pytest.mark.parametrize(
    ("status", "field"),
    [("open", "ruling"), ("open", "confirmation"), ("ruled", "confirmation")],
)
def test_forbidden_interpretation_record_cannot_be_explicit_null(status, field):
    entry = _entry(status)
    entry[field] = None
    assert _validate(_payload(entry)).issues == [
        f"{LABEL}.{field} must be absent when status is `{status}`."
    ]


@pytest.mark.parametrize("source_texts", [{}, {CITATION: None}])
def test_missing_interpretation_source_names_id_and_text_index(source_texts):
    result = _validate(_payload(_entry()), source_texts)
    assert result.issues == [
        f"Proof source unresolved: {LABEL}.texts[0] cites `{CITATION}`, "
        "which was not found in corpus.provisions."
    ]


@pytest.mark.parametrize("resolved", [False, True])
def test_interpretation_checks_every_text_with_its_entry_and_text_index(resolved):
    entry = _entry()
    second_citation = "uk/manual/example/page-88"
    entry["texts"].append({"corpus_citation_path": second_citation, "excerpt": EXCERPT})
    source_texts = {CITATION: EXCERPT}
    if resolved:
        source_texts[second_citation] = "A different provision."
        expected = (
            f"Proof source evidence not found: {LABEL}.texts[1] `source.excerpt` "
            f"does not appear in `{second_citation}`."
        )
    else:
        expected = (
            f"Proof source unresolved: {LABEL}.texts[1] cites `{second_citation}`, "
            "which was not found in corpus.provisions."
        )
    assert _validate(_payload(entry), source_texts).issues == [expected]
    source_texts[second_citation] = EXCERPT
    assert _validate(_payload(entry), source_texts).passed


@pytest.mark.parametrize(
    "output", ["weekly_taper_rate", "uk:statutes/example#weekly_taper_rate"]
)
def test_interpretation_affects_cannot_name_deferred_outputs(output):
    payload = _payload(_entry())
    payload["module"]["deferred_outputs"] = [{"output": output}]
    assert _validate(payload).issues == [
        f"{LABEL}.affects[0] must not name deferred output `weekly_taper_rate`."
    ]


@pytest.mark.parametrize(
    "name", ["missing_rule", "uk:statutes/example#weekly_taper_rate"]
)
def test_interpretation_affects_must_resolve_inside_document(name):
    entry = _entry()
    entry["affects"] = [name]
    assert _validate(_payload(entry)).issues == [
        f"{LABEL}.affects[0] must name a rule defined in this module."
    ]


def test_interpretation_checks_every_affected_rule_in_its_document():
    payload = _payload(_entry())
    payload["module"]["interpretations"][0]["affects"].append("weekly_maximum")
    assert _validate(payload).issues == [
        f"{LABEL}.affects[1] must name a rule defined in this module."
    ]
    rule = deepcopy(payload["rules"][0])
    rule["name"] = "weekly_maximum"
    payload["rules"].append(rule)
    assert _validate(payload).passed
    payload["module"]["deferred_outputs"] = [{"output": "weekly_maximum"}]
    assert _validate(payload).issues == [
        f"{LABEL}.affects[1] must not name deferred output `weekly_maximum`."
    ]


def test_publisher_contradiction_is_blocking_until_repaired():
    entry = _entry("contradicted")
    assert _validate(_payload(entry)).issues == [LABEL + CONTRADICTED]
    old_reading = entry["reading"]
    entry["reading"] = entry["alternatives"][0]["reading"]
    entry["alternatives"][0]["reading"] = old_reading
    entry["status"] = "open"
    del entry["ruling"]
    del entry["confirmation"]
    assert _validate(_payload(entry)).passed


@pytest.mark.parametrize("valid_quote", [True, False])
def test_interpretation_and_proof_matcher_agree_on_pinned_corpus_fixture(valid_quote):
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures/source_completeness/irs_rev_proc_2025_32/page-14.json"
        ).read_text()
    )
    citation = fixture["citation_path"]
    excerpt = (
        "For taxable years beginning in 2026, the maximum amount of the credit "
        "allowed under § 24(a) is $2,200."
    )
    if not valid_quote:
        excerpt = excerpt.replace("$2,200", "$2,201")
    entry = _entry()
    entry["texts"] = [{"corpus_citation_path": citation, "excerpt": excerpt}]
    source_texts = {citation: fixture["body"]}
    result = _validate(_payload(entry), source_texts)
    proof_issues = _validate_source_proof_atom(
        source=entry["texts"][0],
        label=LABEL + ".texts[0]",
        kind="definition",
        rule_source=None,
        source_texts=source_texts,
    )
    assert result.passed is valid_quote
    assert result.issues == proof_issues
    assert result.atoms_checked == 0

    proof_payload = _payload(entry)
    proof_payload["module"] = {}
    proof_rule = proof_payload["rules"][0]
    proof_rule["name"] = "child_tax_credit_maximum"
    proof_rule["versions"][0]["formula"] = "2200"
    proof_rule["metadata"] = {
        "proof": {
            "atoms": [
                {
                    "path": "versions[0].formula",
                    "kind": "parameter",
                    "source": entry["texts"][0],
                }
            ]
        }
    }
    proof_result = _validate(proof_payload, source_texts)
    assert proof_result.passed is valid_quote
    assert proof_result.atoms_checked == 1
    assert proof_result.issues == [
        issue.replace(
            LABEL + ".texts[0]", "rule `child_tax_credit_maximum` proof atom 0"
        )
        for issue in result.issues
    ]
    if not valid_quote:
        assert len(result.issues) == 1
        assert "daily-twenty-per-cent" in result.issues[0]
        assert ".texts[0]" in result.issues[0]


@pytest.mark.parametrize(
    ("source", "excerpt", "passed"),
    [
        ("The prescribed\ndaily percentage", EXCERPT, True),
        ("The prescribed\n\ndaily percentage", EXCERPT, False),
        ("The rate is 120 per cent.", "20 per cent", False),
    ],
)
def test_interpretation_excerpts_share_proof_normalization_and_boundaries(
    source, excerpt, passed
):
    entry = _entry()
    entry["texts"][0]["excerpt"] = excerpt
    assert _validate(_payload(entry), {CITATION: source}).passed is passed


def test_interpretation_citations_are_resolved_once_without_becoming_proofs():
    payload = _payload(_entry())
    payload["module"]["interpretations"].append({**_entry(), "id": "another-reading"})
    content = yaml.safe_dump(payload)
    assert proof_source_citation_paths(content) == (CITATION,)
    strict = _validate(payload, require_policy_proofs=True)
    assert strict.atoms_checked == 0
    assert strict.issues == [
        "Proof missing: rule `weekly_taper_rate` is policy-bearing and must declare `metadata.proof.atoms`."
    ]


@st.composite
def _valid_entries(draw):
    entry = _entry(draw(st.sampled_from(["open", "ruled", "confirmed"])))
    kebab_id = st.lists(
        st.text(
            alphabet=string.ascii_lowercase + string.digits, min_size=1, max_size=8
        ),
        min_size=1,
        max_size=4,
    ).map("-".join)
    entry["id"] = draw(kebab_id)
    nonblank_text = st.text(
        alphabet=string.ascii_letters + string.digits + " ,.?%£",
        min_size=1,
        max_size=70,
    ).filter(lambda value: bool(value.strip()))
    for field in ("question", "reading", "basis"):
        entry[field] = draw(nonblank_text)
    entry["texts"] = draw(
        st.lists(
            st.fixed_dictionaries(
                {
                    "corpus_citation_path": st.just(CITATION),
                    "excerpt": st.sampled_from(
                        [EXCERPT, "prescribed daily percentage", "daily percentage"]
                    ),
                }
            ),
            min_size=1,
            max_size=3,
        )
    )
    entry["alternatives"] = draw(
        st.lists(
            st.fixed_dictionaries(
                {"id": kebab_id, "reading": nonblank_text},
                optional={"effect": nonblank_text},
            ),
            min_size=1,
            max_size=3,
            unique_by=lambda alternative: alternative["id"],
        )
    )
    for field in ("ruling", "confirmation"):
        if field in entry:
            entry[field]["date"] = draw(
                st.dates(min_value=date(1900, 1, 1), max_value=date(2100, 12, 31))
            ).isoformat()
            entry[field]["by"] = draw(
                st.sampled_from(["Axiom Foundation", "Example District Council"])
            )
            entry[field]["record"] = draw(
                st.sampled_from(
                    [
                        "https://example.org/public-record",
                        "https://example.org:443/public-record?year=2026#reply",
                    ]
                )
            )
    if "confirmation" in entry:
        entry["confirmation"]["excerpt"] = draw(nonblank_text)
        if draw(st.booleans()):
            del entry["confirmation"]["record"]
    return entry


@settings(max_examples=35, deadline=None)
@given(entry=_valid_entries())
def test_generated_valid_entries_pass_and_each_missing_required_field_has_one_finding(
    entry,
):
    assert _validate(_payload(entry)).passed
    label = f"Interpretation `{entry['id']}` (module.interpretations[0])"
    expectations = {
        ("id",): "must be a non-blank string.",
        ("question",): "must be a non-blank string.",
        ("reading",): "must be a non-blank string.",
        ("basis",): "must be a non-blank string.",
        ("texts",): "must be a non-empty list.",
        ("alternatives",): "must be a non-empty list.",
        ("status",): "must be open, ruled, confirmed or contradicted.",
        ("affects",): "must be a non-empty list.",
    }
    for text_index in range(len(entry["texts"])):
        for field in ("corpus_citation_path", "excerpt"):
            expectations[("texts", text_index, field)] = "must be a non-blank string."
    for alternative_index in range(len(entry["alternatives"])):
        for field in ("id", "reading"):
            expectations[("alternatives", alternative_index, field)] = (
                "must be a non-blank string."
            )
    for field in ("ruling", "confirmation"):
        if field not in entry:
            continue
        expectations[(field,)] = (
            f"must be a mapping when status is `{entry['status']}`."
        )
        expectations[(field, "by")] = "must be a non-blank string."
        expectations[(field, "date")] = "must be a real YYYY-MM-DD date."
        if field == "ruling":
            expectations[(field, "record")] = "must be an https URL."
        else:
            expectations[(field, "excerpt")] = "must be a non-blank string."
    for path, expectation in expectations.items():
        mutated = deepcopy(entry)
        del _parent(mutated, path)[path[-1]]
        expected_label = "module.interpretations[0]" if path == ("id",) else label
        assert _validate(_payload(mutated)).issues == [
            f"{expected_label}{_field_label(path)} {expectation}"
        ]
