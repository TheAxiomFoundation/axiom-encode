"""Companion input repair must preserve neighboring YAML blocks."""

import pytest
import yaml

from axiom_encode.cli import (
    _expand_empty_inline_yaml_input_blocks,
    _insert_input_default_in_test_cases,
)


@pytest.mark.parametrize("newline", ["\n", "\r\n", ""])
@pytest.mark.parametrize("suffix", ["", " # retained comment"])
@pytest.mark.parametrize("anchor", ["", " &shared"])
def test_empty_input_expansion_preserves_line_ending(newline, suffix, anchor):
    line = f"  input:{anchor} {{}}{suffix}{newline}"
    assert _expand_empty_inline_yaml_input_blocks([line]) == [
        f"  input:{anchor}{suffix}{newline}"
    ]


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("suffix", ["", " # retained comment"])
def test_input_default_repair_keeps_table_rows_and_outputs(newline, suffix):
    content = newline.join(
        [
            "- name: shared_pairs",
            f"  input: {{}}{suffix}",
            "  tables:",
            "    CandidateChildPair:",
            "    - de:statutes/bgb/1591#input.child_identifier: child-1",
            "    - de:statutes/bgb/1591#input.child_identifier: child-2",
            "  output:",
            "    de:statutes/bgb/1591#motherhood_established_by_recorded_birth:",
            "    - holds",
            "    - not_holds",
            "",
        ]
    )
    input_ref = "de:statutes/bgb/1591#input.candidate_person_identifier"
    original = yaml.safe_load(content)
    repaired = _insert_input_default_in_test_cases(content, input_ref, "person-a")
    parsed = yaml.safe_load(repaired)
    assert parsed[0]["input"] == {input_ref: "person-a"}
    assert parsed[0]["tables"] == original[0]["tables"]
    assert parsed[0]["output"] == original[0]["output"]
    assert (
        _insert_input_default_in_test_cases(repaired, input_ref, "person-a") == repaired
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize(
    "key_lines",
    [
        ["    ? {input_ref}", "    : true"],
        ["    ? {input_ref} # retained comment", "    : true"],
        ["    ? '{input_ref}'", "    : true"],
        ['    ? "{input_ref}"', "    : true"],
        ["    '{input_ref}': true"],
        ['    "{input_ref}": true'],
    ],
)
def test_input_default_repair_recognizes_existing_yaml_key(newline, key_lines):
    input_ref = (
        "us:policies/usda/fns/snap-obbb-alien-eligibility-implementation-memo"
        "#input.person_meets_other_snap_financial_and_nonfinancial_eligibility_requirements"
    )
    content = newline.join(
        [
            "- name: retained_long_input",
            "  period: 2025-08",
            "  input:",
            *(line.format(input_ref=input_ref) for line in key_lines),
            "  output:",
            "    us:regulations/7-cfr/273/4#alien_status_eligible: holds",
            "",
        ]
    )

    repaired = _insert_input_default_in_test_cases(content, input_ref, False)

    assert repaired == content
    assert yaml.safe_load(repaired)[0]["input"] == {input_ref: True}


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_input_default_repair_does_not_treat_block_scalar_text_as_key(newline):
    input_ref = "us:example#input.required_value"
    content = newline.join(
        [
            "- name: block_scalar",
            "  input:",
            "    note: |",
            f"      {input_ref}",
            "  output:",
            "    us:example#result: holds",
            "",
        ]
    )

    repaired = _insert_input_default_in_test_cases(content, input_ref, False)
    parsed = yaml.safe_load(repaired)

    assert parsed[0]["input"][input_ref] is False
    assert parsed[0]["input"]["note"].strip() == input_ref
    if newline == "\r\n":
        assert "\n" not in repaired.replace("\r\n", "")


@pytest.mark.parametrize(
    "key_lines",
    [
        ["      {input_ref}: true"],
        ["      ? {input_ref}", "      : true"],
    ],
)
def test_input_default_repair_recognizes_nonstandard_child_indent(key_lines):
    input_ref = "us:example#input.required_value"
    content = "\n".join(
        [
            "- name: deeper_indent",
            "  input:",
            *(line.format(input_ref=input_ref) for line in key_lines),
            "  output:",
            "    us:example#result: holds",
            "",
        ]
    )

    repaired = _insert_input_default_in_test_cases(content, input_ref, False)

    assert repaired == content
    assert yaml.safe_load(repaired)[0]["input"] == {input_ref: True}


def test_input_default_repair_preserves_existing_child_indent():
    input_ref = "us:example#input.required_value"
    content = """\
- name: deeper_indent
  input:
      note: retained
  output:
    us:example#result: holds
"""

    repaired = _insert_input_default_in_test_cases(content, input_ref, False)

    assert f"\n      {input_ref}: false\n" in repaired
    assert yaml.safe_load(repaired)[0]["input"] == {
        input_ref: False,
        "note": "retained",
    }
