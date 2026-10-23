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
    _interpretation_yaml_constraints,
    _safe_load_with_interpretation_constraints,
    _validate_source_proof_atom,
    find_interpretation_yaml_issues,
    proof_source_citation_paths,
    validate_rulespec_proofs,
)
from axiom_encode.yaml_serialization import safe_dump_yaml

CITATION = "zz/manual/fictional-lantern/review-schedule"
EXCERPT = "Review supplies at the start of each period."
LABEL = "Interpretation `weekly-review` (module.interpretations[0])"
CONTRADICTED = (
    ": publisher contradicted this reading; repair the module, "
    "move the old reading into alternatives, and set the new status"
)


def _entry(status="open"):
    entry = {
        "id": "weekly-review",
        "question": "Does the fictional Lantern scheme review weekly or daily?",
        "texts": [{"corpus_citation_path": CITATION, "excerpt": EXCERPT}],
        "reading": "Review supplies once a week.",
        "alternatives": [
            {
                "id": "daily-review",
                "reading": "Review supplies every day.",
                "effect": "More frequent reviews.",
            }
        ],
        "basis": "The fictional Lantern schedule supports weekly reviews.",
        "status": status,
        "affects": ["weekly_review_rate"],
    }
    if status != "open":
        entry["ruling"] = {
            "by": "Fictional Lantern Scheme Office",
            "date": "2024-10-10",
            "record": "https://example.org/fictional-lantern/weekly-review",
        }
    if status in {"confirmed", "contradicted"}:
        entry["confirmation"] = {
            "by": "Fictional Lantern Publisher",
            "date": "2026-10-20",
            "excerpt": "The fictional scheme reviews supplies once a week.",
            "record": "https://example.org/published-reply",
        }
    return entry


def _payload(entry):
    return {
        "format": "rulespec/v1",
        "module": {"interpretations": [entry]},
        "rules": [
            {
                "name": "weekly_review_rate",
                "kind": "parameter",
                "dtype": "Float",
                "versions": [{"effective_from": "2026-04-01", "formula": "0.2"}],
            }
        ],
    }


def _validate(payload, source_texts=None, **kwargs):
    return validate_rulespec_proofs(
        safe_dump_yaml(payload),
        source_texts={CITATION: EXCERPT} if source_texts is None else source_texts,
        **kwargs,
    )


def _raw_payload(*, entry_id="weekly-review", alternative_id="daily-review"):
    """Keep YAML spellings intact.

    safe_dump leaves some YAML 1.2 numeric-looking strings unquoted.
    """

    return f"""format: rulespec/v1
module:
  interpretations:
    - id: {entry_id}
      question: Does the fictional Lantern scheme review weekly or daily?
      texts:
        - corpus_citation_path: {CITATION}
          excerpt: {EXCERPT}
      reading: Review supplies once a week.
      alternatives:
        - id: {alternative_id}
          reading: Review supplies every day.
      basis: The fictional Lantern schedule supports weekly reviews.
      status: open
      affects:
        - weekly_review_rate
rules:
  - name: weekly_review_rate
    kind: parameter
    dtype: Float
    versions:
      - effective_from: '2024-01-01'
        formula: '0.2'
"""


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


def _raw_string_payload(path, scalar):
    """Exercise raw YAML scalars without a load-and-dump normalization step."""

    values = {
        ("question",): "Which review period applies?",
        ("reading",): "Review supplies once a week.",
        ("basis",): "The fictional schedule supports weekly reviews.",
        ("texts", 0, "excerpt"): EXCERPT,
        ("alternatives", 0, "reading"): "Review supplies every day.",
        ("alternatives", 0, "effect"): "More frequent reviews.",
        ("affects", 0): "weekly_review_rate",
        ("ruling", "by"): "Fictional Lantern Scheme Office",
        ("confirmation", "by"): "Fictional Lantern Publisher",
        ("confirmation", "excerpt"): "The fictional scheme reviews weekly.",
    }
    values[path] = scalar
    rule_name = (
        f"'{yaml.safe_load(scalar)}'"
        if path == ("affects", 0)
        else "weekly_review_rate"
    )
    return f"""format: rulespec/v1
module:
  interpretations:
    - id: weekly-review
      question: {values[("question",)]}
      texts:
        - corpus_citation_path: {CITATION}
          excerpt: {values[("texts", 0, "excerpt")]}
      reading: {values[("reading",)]}
      alternatives:
        - id: daily-review
          reading: {values[("alternatives", 0, "reading")]}
          effect: {values[("alternatives", 0, "effect")]}
      basis: {values[("basis",)]}
      status: confirmed
      affects:
        - {values[("affects", 0)]}
      ruling:
        by: {values[("ruling", "by")]}
        date: '2024-10-10'
        record: https://example.org/fictional-lantern/weekly-review
      confirmation:
        by: {values[("confirmation", "by")]}
        date: '2026-10-20'
        excerpt: {values[("confirmation", "excerpt")]}
        record: https://example.org/published-reply
rules:
  - name: {rule_name}
    kind: parameter
    dtype: Float
    versions:
      - effective_from: '2024-01-01'
        formula: '0.2'
"""


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
    expectation = "must be a non-blank string."
    if path in {("id",), ("alternatives", 0, "id")}:
        if isinstance(value, bool):
            expectation = "must be a quoted string when YAML reads it as a boolean."
        elif value is None:
            expectation = "must be a quoted string when YAML reads it as null."
        elif isinstance(value, (int, float)):
            expectation = "must be a quoted string when YAML reads it as a number."
    assert result.issues == [f"{expected_label}{_field_label(path)} {expectation}"]


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
        "Interpretation `weekly-review` (module.interpretations[1]).id "
        "duplicates interpretation `weekly-review`."
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


def test_interpretation_alternative_ids_can_repeat_in_different_interpretations():
    first = _entry()
    second = _entry()
    second["id"] = "second-reading"
    assert first["alternatives"][0]["id"] == second["alternatives"][0]["id"]
    payload = _payload(first)
    payload["module"]["interpretations"].append(second)
    assert _validate(payload).passed


@pytest.mark.parametrize("field", ["entry_id", "alternative_id"])
@pytest.mark.parametrize("scalar", ["1e5", "1e-5", "2e10", "0o17", "08"])
def test_interpretation_ids_reject_raw_yaml_core_numeric_scalars(field, scalar):
    content = _raw_payload(**{field: scalar})
    result = validate_rulespec_proofs(content, source_texts={CITATION: EXCERPT})
    label = (
        f"Interpretation `{scalar}` (module.interpretations[0]).id"
        if field == "entry_id"
        else f"{LABEL}.alternatives[0].id"
    )
    assert result.issues == [
        f"{label} must be a quoted string when YAML reads this scalar as a number."
    ]


@pytest.mark.parametrize("path", [path for path in STRING_PATHS if path[-1] != "id"])
@pytest.mark.parametrize("scalar", ["1e5", "-0o17", "+.5", "2e10"])
def test_interpretation_strings_reject_raw_yaml_core_numeric_scalars(path, scalar):
    content = _raw_string_payload(path, scalar)
    result = validate_rulespec_proofs(
        content, source_texts={CITATION: f"{EXCERPT} {scalar}"}
    )
    assert result.issues == [
        f"module.interpretations[0]{_field_label(path)} "
        "must be a quoted string when YAML reads this scalar as a number."
    ]


@pytest.mark.parametrize("path", [path for path in STRING_PATHS if path[-1] != "id"])
@pytest.mark.parametrize("value", ["1e5", "-0o17", "+.5", "2e10"])
@pytest.mark.parametrize(
    "style",
    ["quoted", "explicit-string-tag", "canonical-string-tag", "string-tag-handle"],
)
def test_interpretation_strings_preserve_explicit_yaml_strings(path, value, style):
    scalar = {
        "quoted": f"'{value}'",
        "explicit-string-tag": f"!!str {value}",
        "canonical-string-tag": f"!<tag:yaml.org,2002:str> {value}",
        "string-tag-handle": f"!s!str {value}",
    }[style]
    if style == "string-tag-handle":
        content = _raw_string_payload(path, f"!!str {value}").replace(
            f"!!str {value}", scalar, 1
        )
        content = "%TAG !s! tag:yaml.org,2002:\n---\n" + content
    else:
        content = _raw_string_payload(path, scalar)
    result = validate_rulespec_proofs(
        content,
        source_texts={CITATION: f"{EXCERPT} {value}"},
    )
    assert result.passed, result.issues


@pytest.mark.parametrize("field", ["entry_id", "alternative_id"])
@pytest.mark.parametrize("scalar", ['"1e5"', "'0o17'", '"0"', "!!str 123", "!!str 1e5"])
def test_interpretation_ids_preserve_explicit_yaml_strings(field, scalar):
    result = validate_rulespec_proofs(
        _raw_payload(**{field: scalar}), source_texts={CITATION: EXCERPT}
    )
    assert result.passed, result.issues


@pytest.mark.parametrize(
    "path",
    [
        *STRING_PATHS,
        ("texts", 0, "corpus_citation_path"),
        ("status",),
        ("ruling", "date"),
        ("ruling", "record"),
        ("confirmation", "date"),
        ("confirmation", "record"),
    ],
)
def test_interpretation_scalars_reject_nonspecific_tags_at_every_field(path):
    entry = _entry("confirmed")
    original = _parent(entry, path)[path[-1]]
    _parent(entry, path)[path[-1]] = "unsupported-tag-probe"
    content = yaml.safe_dump(_payload(entry)).replace(
        "unsupported-tag-probe", f"! {json.dumps(original)}"
    )
    result = validate_rulespec_proofs(content, source_texts={CITATION: EXCERPT})
    assert not result.passed
    assert result.issues == [
        f"module.interpretations[0]{_field_label(path)} has unsupported YAML tag `!`; "
        "only the canonical string tag `!!str` (`tag:yaml.org,2002:str`) is supported."
    ]


@pytest.mark.parametrize(
    "scalar",
    [
        "! 1e5",
        "! -0o17",
        "! +.5",
        "! '1e5'",
        "! ordinary prose",
        "!!int 123",
        "!!float 1e5",
        "!!bool true",
        "!!null null",
        "!!timestamp 2024-10-10",
        "!local prose",
        "!<tag:example.org,2026:prose> prose",
        "!<str> prose",
    ],
)
def test_interpretation_scalars_reject_every_noncanonical_string_tag(scalar):
    result = validate_rulespec_proofs(
        _raw_string_payload(("question",), scalar), source_texts={CITATION: EXCERPT}
    )
    assert not result.passed
    assert any(
        issue.startswith("module.interpretations[0].question has unsupported YAML tag")
        for issue in result.issues
    ), result.issues


@pytest.mark.parametrize(
    ("original", "replacement", "location"),
    [
        ("  interpretations:", "  ! interpretations:", "module key `interpretations`"),
        (
            "      question:",
            "      ! question:",
            "module.interpretations[0] key `question`",
        ),
    ],
)
def test_interpretation_scalars_reject_nonspecific_tags_on_mapping_keys(
    original, replacement, location
):
    content = _raw_payload().replace(original, replacement, 1)
    result = validate_rulespec_proofs(content, source_texts={CITATION: EXCERPT})
    assert result.issues == [
        f"{location} has unsupported YAML tag `!`; only the canonical string tag "
        "`!!str` (`tag:yaml.org,2002:str`) is supported."
    ]


@pytest.mark.parametrize("tag", ["!", "!!seq", "!!str", "!local"])
def test_interpretation_collections_reject_explicit_sequence_tags(tag):
    content = _raw_payload().replace(
        "  interpretations:\n", f"  interpretations: {tag}\n"
    )
    result = validate_rulespec_proofs(content, source_texts={CITATION: EXCERPT})
    assert result.issues == [
        "module.interpretations must not use YAML tags on interpretation collections."
    ]


@pytest.mark.parametrize("tag", ["!", "!!map", "!!str", "!local"])
def test_interpretation_collections_reject_explicit_mapping_tags(tag):
    content = _raw_payload().replace(
        "    - id: weekly-review\n", f"    - {tag}\n      id: weekly-review\n"
    )
    result = validate_rulespec_proofs(content, source_texts={CITATION: EXCERPT})
    assert result.issues == [
        "module.interpretations[0] must not use YAML tags on interpretation collections."
    ]


@pytest.mark.parametrize("field", ["entry_id", "alternative_id"])
def test_raw_interpretation_guard_reports_numeric_id_paths_before_loading(field):
    content = _raw_payload(**{field: "1e5"})
    location = (
        "module.interpretations[0].id"
        if field == "entry_id"
        else "module.interpretations[0].alternatives[0].id"
    )
    assert find_interpretation_yaml_issues(content) == [
        f"{location} must be a quoted string when YAML reads this scalar as a number."
    ]


def test_interpretation_anchored_numeric_id_is_rejected():
    result = validate_rulespec_proofs(
        _raw_payload(alternative_id="&numeric 1e5"),
        source_texts={CITATION: EXCERPT},
    )
    assert result.issues == [
        f"{LABEL}.alternatives[0].id must be a quoted string "
        "when YAML reads this scalar as a number."
    ]


@pytest.mark.parametrize("field", ["entry_id", "alternative_id"])
@pytest.mark.parametrize(
    ("scalar", "yaml_type"),
    [
        ("0123", "a number"),
        ("0x1f", "a number"),
        ("0b101", "a number"),
        ("+1", "a number"),
        (".5", "a number"),
        ("1.0", "a number"),
        ("Null", "null"),
    ],
)
def test_interpretation_nonstring_id_diagnostic_names_yaml_type(
    field, scalar, yaml_type
):
    result = validate_rulespec_proofs(
        _raw_payload(**{field: scalar}), source_texts={CITATION: EXCERPT}
    )
    label = (
        "module.interpretations[0].id"
        if field == "entry_id"
        else f"{LABEL}.alternatives[0].id"
    )
    assert result.issues == [
        f"{label} must be a quoted string when YAML reads it as {yaml_type}."
    ]


@pytest.mark.parametrize("field", ["entry_id", "alternative_id"])
@pytest.mark.parametrize("scalar", ["no", "yes", "on"])
def test_interpretation_boolean_id_diagnostic_requests_quoting(field, scalar):
    result = validate_rulespec_proofs(
        _raw_payload(**{field: scalar}), source_texts={CITATION: EXCERPT}
    )
    label = (
        "module.interpretations[0].id"
        if field == "entry_id"
        else f"{LABEL}.alternatives[0].id"
    )
    assert result.issues == [
        f"{label} must be a quoted string when YAML reads it as a boolean."
    ]
    quoted = validate_rulespec_proofs(
        _raw_payload(**{field: f"'{scalar}'"}), source_texts={CITATION: EXCERPT}
    )
    assert quoted.passed, quoted.issues


@pytest.mark.parametrize(
    ("original", "replacement", "location"),
    [
        (
            "    - id: weekly-review\n",
            "    - id: first-reading\n      id: weekly-review\n",
            "module.interpretations[0]",
        ),
        (
            "        - id: daily-review\n",
            "        - id: first-reading\n          id: daily-review\n",
            "module.interpretations[0].alternatives[0]",
        ),
    ],
)
def test_interpretation_yaml_mappings_reject_duplicate_keys(
    original, replacement, location
):
    content = _raw_payload().replace(original, replacement)
    result = validate_rulespec_proofs(content, source_texts={CITATION: EXCERPT})
    assert result.issues == [f"{location} has duplicate key `id`."]


def test_interpretation_yaml_rejects_duplicate_module_interpretations_keys():
    content = _raw_payload().replace(
        "  interpretations:\n", "  interpretations: []\n  interpretations:\n", 1
    )
    result = validate_rulespec_proofs(content, source_texts={CITATION: EXCERPT})
    assert result.issues == ["module has duplicate key `interpretations`."]


def test_interpretation_yaml_mappings_reject_merge_keys():
    content = _raw_payload().replace(
        "        - id: daily-review\n          reading: Review supplies every day.\n",
        "        - &base\n"
        "          id: daily-review\n"
        "          reading: Review supplies every day.\n"
        "        - <<: *base\n"
        "          id: monthly-review\n",
    )
    result = validate_rulespec_proofs(content, source_texts={CITATION: EXCERPT})
    assert result.issues == [
        "module.interpretations[0].alternatives[1] must not use YAML merge keys."
    ]


RAW_RULES = """rules:
  - name: weekly_review_rate
    kind: parameter
    dtype: Float
    versions:
      - effective_from: '2024-01-01'
        formula: '0.2'
"""

# Safe loading would resolve the duplicate `question` and turn `! 1e5` into
# the string "1e5", so only the raw guard can see these two defects.
RAW_SHADOWED_INTERPRETATIONS = f"""  interpretations:
    - id: weekly-review
      question: Discarded when YAML resolves the duplicate key
      question: ! 1e5
      texts:
        - corpus_citation_path: {CITATION}
          excerpt: {EXCERPT}
      reading: Review supplies once a week.
      alternatives:
        - id: daily-review
          reading: Review supplies every day.
      basis: The fictional Lantern schedule supports weekly reviews.
      status: open
      affects:
        - weekly_review_rate
"""

MODULE_MERGE_MESSAGE = (
    "module must not supply `interpretations` through YAML merge keys; "
    "the engine does not apply them."
)


@pytest.mark.parametrize(
    ("anchors", "module"),
    [
        pytest.param("", "module:\n  <<: *defaults\n", id="mapping"),
        pytest.param("", "module:\n  <<: [*defaults]\n", id="sequence"),
        pytest.param(
            "x-outer: &outer\n  <<: *defaults\n",
            "module:\n  <<: *outer\n",
            id="nested",
        ),
        pytest.param(
            "x-title: &title\n  title: Fictional Lantern review schedule\n",
            "module:\n  <<: [*title, *defaults]\n",
            id="second-of-two",
        ),
        pytest.param(
            "",
            "module:\n  <<: *defaults\n  interpretations: []\n",
            id="overridden-by-explicit-key",
        ),
    ],
)
def test_interpretation_yaml_rejects_interpretations_supplied_by_module_merge_keys(
    anchors, module
):
    content = (
        "format: rulespec/v1\n"
        "x-defaults: &defaults\n"
        f"{RAW_SHADOWED_INTERPRETATIONS}{anchors}{module}{RAW_RULES}"
    )
    assert find_interpretation_yaml_issues(content) == [MODULE_MERGE_MESSAGE]
    result = validate_rulespec_proofs(content, source_texts={CITATION: EXCERPT})
    assert result.issues == [MODULE_MERGE_MESSAGE]


def test_interpretation_yaml_rejects_module_supplied_by_root_merge_keys():
    module = "  module:\n" + "".join(
        "  " + line for line in RAW_SHADOWED_INTERPRETATIONS.splitlines(True)
    )
    content = f"format: rulespec/v1\nx-base: &base\n{module}<<: *base\n{RAW_RULES}"
    message = (
        "RuleSpec root must not supply `module` through YAML merge keys; "
        "the engine does not apply them."
    )
    assert find_interpretation_yaml_issues(content) == [message]
    result = validate_rulespec_proofs(content, source_texts={CITATION: EXCERPT})
    assert result.issues == [message]


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(
            _raw_payload().replace(
                "module:\n",
                "x-title: &title\n  title: Fictional Lantern review schedule\n"
                "module:\n  <<: *title\n",
                1,
            ),
            id="merge-supplies-other-metadata",
        ),
        # A quoted `<<` is an ordinary key to PyYAML and to the engine alike.
        pytest.param(
            "format: rulespec/v1\nx-defaults: &defaults\n"
            f"{RAW_SHADOWED_INTERPRETATIONS}module:\n  '<<': *defaults\n{RAW_RULES}",
            id="quoted-merge-spelling",
        ),
    ],
)
def test_interpretation_yaml_allows_module_merges_that_cannot_supply_interpretations(
    content,
):
    assert find_interpretation_yaml_issues(content) == []
    result = validate_rulespec_proofs(content, source_texts={CITATION: EXCERPT})
    assert result.passed, result.issues


@pytest.mark.parametrize(
    ("original", "replacement", "location"),
    [
        (
            "  interpretations:\n",
            "  !!merge interpretations:\n",
            "module key `interpretations`",
        ),
        ("module:\n", "!!merge module:\n", "RuleSpec root key `module`"),
    ],
)
def test_interpretation_yaml_rejects_merge_tags_that_hide_interpretations_on_load(
    original, replacement, location
):
    # PyYAML merges the tagged value into its parent, so the loaded module has
    # no `interpretations`; the engine reads the key by name and validates it.
    content = _raw_payload().replace(original, replacement, 1)
    loaded = yaml.safe_load(content)
    assert "interpretations" not in (loaded.get("module") or {})
    message = (
        f"{location} has unsupported YAML tag `tag:yaml.org,2002:merge`; only the "
        "canonical string tag `!!str` (`tag:yaml.org,2002:str`) is supported."
    )
    assert find_interpretation_yaml_issues(content) == [message]
    result = validate_rulespec_proofs(content, source_texts={CITATION: EXCERPT})
    assert result.issues == [message]


FIRST_KEYS = [
    "    - id: weekly-review\n",
    "        - corpus_citation_path:",
    "        - id: daily-review\n",
]


@pytest.mark.parametrize("original", FIRST_KEYS)
@pytest.mark.parametrize("tag", ["!!str", "!<tag:yaml.org,2002:str>"])
def test_interpretation_yaml_accepts_string_tag_on_first_mapping_key(original, tag):
    # A block mapping starts where its first key starts; the key's tag belongs
    # to the key, not to the mapping.
    content = _raw_payload().replace(original, original.replace("- ", f"- {tag} "))
    assert content != _raw_payload()
    assert find_interpretation_yaml_issues(content) == []
    result = validate_rulespec_proofs(content, source_texts={CITATION: EXCERPT})
    assert result.passed, result.issues


@pytest.mark.parametrize(
    ("original", "location"),
    [
        ("    - id: weekly-review\n", "module.interpretations[0] key `id`"),
        (
            "        - corpus_citation_path:",
            "module.interpretations[0].texts[0] key `corpus_citation_path`",
        ),
        (
            "        - id: daily-review\n",
            "module.interpretations[0].alternatives[0] key `id`",
        ),
    ],
)
def test_interpretation_yaml_reports_first_key_tags_on_the_key(original, location):
    content = _raw_payload().replace(original, original.replace("- ", "- ! "))
    message = (
        f"{location} has unsupported YAML tag `!`; only the canonical string tag "
        "`!!str` (`tag:yaml.org,2002:str`) is supported."
    )
    assert find_interpretation_yaml_issues(content) == [message]
    result = validate_rulespec_proofs(content, source_texts={CITATION: EXCERPT})
    assert result.issues == [message]


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(_raw_payload() + "module: {}\n", id="empty-last"),
        pytest.param(
            _raw_payload().replace(
                "format: rulespec/v1\n", "format: rulespec/v1\nmodule: {}\n"
            ),
            id="empty-first",
        ),
        pytest.param(_raw_payload() + '"module": {}\n', id="quoted"),
        pytest.param(_raw_payload() + "? module\n: {}\n", id="explicit-key"),
        pytest.param(
            _raw_payload().replace(
                "rules:\n",
                "module:\n"
                + _raw_payload().split("module:\n")[1].split("rules:\n")[0]
                + "rules:\n",
            ),
            id="identical",
        ),
    ],
)
def test_interpretation_yaml_rejects_duplicate_root_module_keys(content):
    message = "RuleSpec root has duplicate key `module`."
    assert find_interpretation_yaml_issues(content) == [message]
    result = validate_rulespec_proofs(content, source_texts={CITATION: EXCERPT})
    assert result.issues == [message]


def test_interpretation_yaml_scans_every_root_module_block():
    content = (
        _raw_payload().replace(
            "      question: Does the fictional Lantern scheme review weekly or "
            "daily?\n",
            "      question: ! 1e5\n",
        )
        + "module: {}\n"
    )
    assert yaml.safe_load(content)["module"] == {}
    issues = [
        "RuleSpec root has duplicate key `module`.",
        "module.interpretations[0].question has unsupported YAML tag `!`; only the "
        "canonical string tag `!!str` (`tag:yaml.org,2002:str`) is supported.",
    ]
    assert find_interpretation_yaml_issues(content) == issues
    result = validate_rulespec_proofs(content, source_texts={CITATION: EXCERPT})
    assert result.issues == issues


def _assert_single_pass_load_matches_safe_load(content):
    payload, issues, numeric_id_paths = _safe_load_with_interpretation_constraints(
        content
    )
    assert payload == yaml.safe_load(content)
    assert (issues, numeric_id_paths) == _interpretation_yaml_constraints(content)


@pytest.mark.parametrize(
    "content",
    [
        _raw_payload(),
        _raw_payload(entry_id="1e5", alternative_id="! daily-review"),
        _raw_payload().replace("    - id: weekly-review\n", "    - !!str id: x\n"),
        _raw_payload() + "module: {}\n",
        "format: rulespec/v1\nx-defaults: &defaults\n"
        f"{RAW_SHADOWED_INTERPRETATIONS}module:\n  <<: [*defaults]\n{RAW_RULES}",
        _raw_payload().replace("module:\n", "!!merge module:\n", 1),
        "x: &a {b: &c [1, *c]}\ny: *a\n",
        "--- !!map\n? outer\n: {<<: [{k: v}, {j: u}], k: w}\n",
    ],
)
def test_single_pass_interpretation_load_matches_safe_load(content):
    if "*c" in content:
        # Recursive values cannot be compared with ==; compare the raw view only.
        assert _safe_load_with_interpretation_constraints(content)[1:] == (
            _interpretation_yaml_constraints(content)
        )
        return
    _assert_single_pass_load_matches_safe_load(content)


_YAML_SCALARS = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False)
    | st.text(max_size=12)
    | st.dates()
)
_YAML_VALUES = st.recursive(
    _YAML_SCALARS,
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=16,
)


@settings(max_examples=200, deadline=None)
@given(
    value=_YAML_VALUES,
    under_interpretations=st.booleans(),
    flow_style=st.sampled_from([None, True, False]),
)
def test_single_pass_interpretation_load_matches_safe_load_for_generated_yaml(
    value, under_interpretations, flow_style
):
    module = {"interpretations": value} if under_interpretations else value
    content = yaml.safe_dump(
        {"format": "rulespec/v1", "module": module},
        default_flow_style=flow_style,
        allow_unicode=True,
    )
    _assert_single_pass_load_matches_safe_load(content)


def test_interpretation_yaml_constraints_allow_aliases_without_recursing_forever():
    content = _raw_payload().replace(
        "      basis: The fictional Lantern schedule supports weekly reviews.\n",
        "      basis: &cycle [*cycle]\n",
    )
    result = validate_rulespec_proofs(content, source_texts={CITATION: EXCERPT})
    assert result.issues == [f"{LABEL}.basis must be a non-blank string."]


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
        "HTTPS://example.org/reply",
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
        f"{LABEL}.{field}.record must be an https URL beginning with literal `https://`."
    ]


@pytest.mark.parametrize("field", ["ruling", "confirmation"])
def test_interpretation_record_raw_yaml_requires_literal_https_prefix(field):
    content = _raw_payload().replace(
        "      status: open\n",
        "      status: confirmed\n"
        "      ruling:\n"
        "        by: Fictional Lantern Scheme Office\n"
        "        date: '2024-10-10'\n"
        f"        record: {'HTTPS' if field == 'ruling' else 'https'}://example.org/ruling\n"
        "      confirmation:\n"
        "        by: Fictional Lantern Publisher\n"
        "        date: '2024-10-20'\n"
        "        excerpt: Review supplies once a week.\n"
        f"        record: {'HTTPS' if field == 'confirmation' else 'https'}://example.org/confirmation\n",
    )
    result = validate_rulespec_proofs(content, source_texts={CITATION: EXCERPT})
    assert result.issues == [
        f"{LABEL}.{field}.record must be an https URL beginning with literal `https://`."
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
        "zz:manual/fictional-lantern/review-schedule",
        " zz/manual/fictional-lantern/review-schedule ",
        "zz/manual/fictional-lantern/review-schedule#rule",
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
    second_citation = "zz/manual/fictional-lantern/review-details"
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
    "output", ["weekly_review_rate", "uk:statutes/example#weekly_review_rate"]
)
def test_interpretation_affects_cannot_name_deferred_outputs(output):
    payload = _payload(_entry())
    payload["module"]["deferred_outputs"] = [{"output": output}]
    assert _validate(payload).issues == [
        f"{LABEL}.affects[0] must not name deferred output `weekly_review_rate`."
    ]


@pytest.mark.parametrize(
    "name", ["missing_rule", "uk:statutes/example#weekly_review_rate"]
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
        assert "weekly-review" in result.issues[0]
        assert ".texts[0]" in result.issues[0]


@pytest.mark.parametrize(
    ("source", "excerpt", "passed"),
    [
        ("Review supplies at the\nstart of each period.", EXCERPT, True),
        ("Review supplies at the\n\nstart of each period.", EXCERPT, False),
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
        "Proof missing: rule `weekly_review_rate` is policy-bearing and must declare `metadata.proof.atoms`."
    ]


@st.composite
def _valid_entries(draw):
    entry = _entry(draw(st.sampled_from(["open", "ruled", "confirmed"])))
    kebab_id = st.lists(
        st.builds(
            lambda first, rest: first + rest,
            st.sampled_from(string.ascii_lowercase),
            st.text(alphabet=string.ascii_lowercase + string.digits, max_size=7),
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
                        [
                            EXCERPT,
                            "supplies at the start of each period.",
                            "start of each period.",
                        ]
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
                st.sampled_from(
                    ["Fictional Lantern Scheme Office", "Fictional Lantern Publisher"]
                )
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
            expectations[(field, "record")] = (
                "must be an https URL beginning with literal `https://`."
            )
        else:
            expectations[(field, "excerpt")] = "must be a non-blank string."
    for path, expectation in expectations.items():
        mutated = deepcopy(entry)
        del _parent(mutated, path)[path[-1]]
        expected_label = "module.interpretations[0]" if path == ("id",) else label
        assert _validate(_payload(mutated)).issues == [
            f"{expected_label}{_field_label(path)} {expectation}"
        ]
