"""Properties behind the apply guard's validation snapshot.

Overlay validation stamps a generated module (a YAML re-dump with
allow_unicode), rewrites its non-ASCII escapes and records the bytes. Apply
stamps the module again and refuses unless the bytes still match. These tests
pin the two facts that comparison depends on, and the refusal's naming of
what changed.
"""

import copy
import re

import yaml
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from axiom_encode.cli import (
    _apply_validation_snapshot_change_summary,
    _apply_validation_snapshot_changed_paths,
    _unescape_non_ascii_yaml_escapes,
    _yaml_emitter_writes_literally,
)
from axiom_encode.yaml_serialization import safe_dump_yaml

_ESCAPE = re.compile(r"\\u([0-9A-Fa-f]{4})")

# Characters the emitter treats specially, a script the rewrite exists for,
# and YAML punctuation, mixed with arbitrary code points.
_PICKED = (
    "\ufeff\u2028\u2029\x85\x80\x9f\ufffe\uffff\ud800\xa0"
    "\u05d0\u05e7\u12a0\u03b1\xe9 \t\n\"'#:-[]{}&*!|>%@`,?"
)


def _characters(*, backslash: bool) -> st.SearchStrategy[str]:
    picked = _PICKED + ("\\" if backslash else "")
    arbitrary = st.characters(exclude_characters="\\" if not backslash else "")
    return st.one_of(st.sampled_from(picked), arbitrary)


def _modules(*, backslash: bool) -> st.SearchStrategy[dict[str, object]]:
    text = st.text(_characters(backslash=backslash), max_size=24)
    return st.builds(
        lambda summary, sources: {
            "format": "rulespec/v1",
            "module": {"summary": summary},
            "rules": [{"name": "rule", "source": source} for source in sources],
        },
        text,
        st.lists(text, max_size=3),
    )


def _stamp(text: str) -> str:
    # The re-dump `_stamp_generated_source_attestation_for_apply` writes after
    # it sets the two source_verification fields.
    return safe_dump_yaml(yaml.safe_load(text), sort_keys=False, allow_unicode=True)


@settings(max_examples=300, deadline=None)
@given(_modules(backslash=True))
@example({"format": "rulespec/v1", "module": {"summary": "72(1)\ufeff(a)"}})
def test_unescape_leaves_the_emitters_own_output_unchanged(module):
    stamped = safe_dump_yaml(module, sort_keys=False, allow_unicode=True)

    assert _unescape_non_ascii_yaml_escapes(stamped) is None


@settings(max_examples=300, deadline=None)
@given(_modules(backslash=True))
@example({"format": "rulespec/v1", "module": {"summary": "72(1)\ufeff(a)"}})
def test_validated_bytes_survive_the_apply_time_stamp(module):
    stamped = safe_dump_yaml(module, sort_keys=False, allow_unicode=True)
    assume(_stamp(stamped) == stamped)
    rewritten = _unescape_non_ascii_yaml_escapes(stamped)
    validated = stamped if rewritten is None else rewritten

    assert _stamp(validated) == validated


@settings(max_examples=300, deadline=None)
@given(_modules(backslash=False))
@example({"format": "rulespec/v1", "module": {"summary": "\u05d0 72(1)\ufeff(a)"}})
@example({"format": "rulespec/v1", "module": {"summary": "\u05d0\ud800"}})
@example({"format": "rulespec/v1", "module": {"summary": "\u05d0\ufffe"}})
def test_unescape_rewrites_every_escape_the_emitter_would_not_write(module):
    # A model that escapes every non-ASCII character: the rewrite keeps the
    # parse and leaves only the escapes the emitter itself writes.
    escaped = safe_dump_yaml(module, sort_keys=False, allow_unicode=False)
    rewritten = _unescape_non_ascii_yaml_escapes(escaped)
    readable = escaped if rewritten is None else rewritten

    assert yaml.safe_load(readable) == yaml.safe_load(escaped)
    remaining = {chr(int(code, 16)) for code in _ESCAPE.findall(readable)}
    assert not any(_yaml_emitter_writes_literally(c) for c in remaining)


def test_emitter_literal_set_matches_what_the_emitter_writes():
    # Checked against PyYAML itself on every non-ASCII BMP character, the only
    # ones a `\uXXXX` rewrite can produce: in a double-quoted scalar dumped
    # with allow_unicode (a leading NEL forces the quotes), the emitter keeps
    # the character as itself exactly when the predicate says so.
    characters = [chr(code) for code in range(0x80, 0x10000)]
    for start in range(0, len(characters), 4096):
        chunk = characters[start : start + 4096]
        dumped = safe_dump_yaml(
            ["\x85" + character for character in chunk],
            default_flow_style=False,
            allow_unicode=True,
        )
        lines = dumped.split("\n")[: len(chunk)]
        for character, line in zip(chunk, lines, strict=True):
            assert line.startswith('- "\\N')
            assert (character in line) is _yaml_emitter_writes_literally(character), (
                hex(ord(character))
            )


_LEAVES = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-2, 2),
    st.sampled_from(["a", "b", "x/y.yaml", "0" * 64]),
)
_KEYS = st.sampled_from(["output", "sha256", "path", "x/y.yaml", "a b", "1"])
_TREES = st.recursive(
    _LEAVES,
    lambda children: st.one_of(
        st.lists(children, max_size=3),
        st.dictionaries(_KEYS, children, max_size=3),
    ),
    max_leaves=12,
)


@settings(max_examples=500, deadline=None)
@given(_TREES, _TREES)
def test_snapshot_diff_names_a_path_exactly_when_the_snapshots_differ(expected, actual):
    changed = _apply_validation_snapshot_changed_paths(expected, actual)

    assert (changed == []) == (expected == actual)


def test_snapshot_diff_names_key_paths_and_never_values():
    expected = {
        "output": {"path": "/work/out/1.yaml", "sha256": "a" * 64},
        "supplemental": [{"path": "statutes/26/2.yaml", "sha256": "b" * 64}],
        "validation_execution": {
            "policy_content_files": {"statutes/26/2.yaml": "c" * 64},
        },
        "test": {"exists": False, "sha256": None},
    }
    actual = copy.deepcopy(expected)
    actual["output"]["sha256"] = "d" * 64
    actual["supplemental"][0]["sha256"] = "e" * 64
    actual["validation_execution"]["policy_content_files"]["statutes/26/3.yaml"] = (
        "f" * 64
    )
    del actual["test"]

    changed = _apply_validation_snapshot_changed_paths(expected, actual)

    assert changed == [
        "output.sha256",
        "supplemental[0].sha256",
        "test",
        'validation_execution.policy_content_files["statutes/26/3.yaml"]',
    ]
    summary = _apply_validation_snapshot_change_summary(changed)
    assert summary == (
        " (changed: output.sha256, supplemental[0].sha256, test, "
        'validation_execution.policy_content_files["statutes/26/3.yaml"])'
    )
    for value in ("a", "b", "c", "d", "e", "f"):
        assert value * 64 not in summary


def test_snapshot_change_summary_names_at_most_eight_paths():
    changed = [f"supplemental[{index}].sha256" for index in range(11)]

    assert _apply_validation_snapshot_change_summary(changed) == (
        " (changed: "
        + ", ".join(f"supplemental[{index}].sha256" for index in range(8))
        + " and 3 more)"
    )
    assert _apply_validation_snapshot_change_summary([]) == ""


def test_snapshot_diff_names_a_list_whose_length_changed():
    assert _apply_validation_snapshot_changed_paths(
        {"supplemental": [{"sha256": "a"}]},
        {"supplemental": [{"sha256": "a"}, {"sha256": "b"}]},
    ) == ["supplemental"]
