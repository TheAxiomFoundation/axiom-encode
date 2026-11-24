"""The byte-preserving binder of ``module.source_verification``.

Every module text here is synthetic.
"""

from __future__ import annotations

import difflib

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.reviewed_pr_admission import (
    AdmissionError,
    read_candidate_source_pin,
)
from axiom_encode.source_verification_binding import (
    CHECK_SOURCE_BINDING,
    SourceBindingConflict,
    SourceBindingError,
    SourceBindingInputError,
    SourceBindingShapeError,
    bind_source_verification,
    check_source_verification,
)

PATH = "xx/statute/1/2"
RESOLVED = "xx/statute/1"
PIN = "ab" * 32
OTHER_PIN = "cd" * 32
ATTESTATION = {
    "requested_corpus_citation_path": PATH,
    "resolved_corpus_citation_path": PATH,
    "source_sha256": PIN,
}
FALLBACK = {**ATTESTATION, "resolved_corpus_citation_path": RESOLVED}

PATH_ONLY = (
    "format: rulespec/v1\n"
    "module:\n"
    "  proof_validation:\n"
    "    required: true\n"
    "  source_verification:\n"
    f"    corpus_citation_path: {PATH}\n"
    "  summary: |-\n"
    "    First line: with a colon.\n"
    "    # not a comment, and a\ttab.\n"
    "rules: []\n"
)


def bind(text: str, attestation=ATTESTATION):
    return bind_source_verification(text.encode(), attestation)


def test_inserts_exactly_one_line_after_the_path():
    result = bind(PATH_ONLY)
    assert result.edits == ("insert_source_sha256",)
    assert result.changed
    expected = PATH_ONLY.replace(
        f"    corpus_citation_path: {PATH}\n",
        f"    corpus_citation_path: {PATH}\n    source_sha256: {PIN}\n",
    )
    assert result.content == expected.encode()
    assert (result.corpus_citation_path, result.source_sha256) == (PATH, PIN)


def test_a_bound_module_is_returned_byte_for_byte():
    bound = bind(PATH_ONLY).content
    again = bind_source_verification(bound, ATTESTATION)
    assert again.content == bound
    assert again.edits == ()
    assert not again.changed
    check_source_verification(bound, ATTESTATION)


def test_check_refuses_an_unbound_module_and_writes_nothing():
    with pytest.raises(SourceBindingError) as caught:
        check_source_verification(PATH_ONLY.encode(), ATTESTATION)
    assert caught.value.kind == "unbound"
    assert "source_sha256" in str(caught.value)
    assert "corpus_citation_path" not in str(caught.value)


def test_check_names_both_keys_when_the_block_is_absent():
    with pytest.raises(SourceBindingError) as caught:
        check_source_verification(b"module:\n  summary: x\n", ATTESTATION)
    assert "corpus_citation_path and source_sha256" in str(caught.value)


def test_creates_the_block_with_the_module_indentation():
    text = "format: rulespec/v1\nmodule:\n    # note\n    summary: x\nrules: []\n"
    result = bind(text)
    assert result.edits == ("create_source_verification",)
    assert result.content.decode() == (
        "format: rulespec/v1\nmodule:\n    # note\n"
        "    source_verification:\n"
        f"        corpus_citation_path: {PATH}\n"
        f"        source_sha256: {PIN}\n"
        "    summary: x\nrules: []\n"
    )


def test_inserts_the_path_before_an_existing_pin():
    text = f"module:\n  source_verification:\n    source_sha256: {PIN}\n"
    result = bind(text)
    assert result.edits == ("insert_corpus_citation_path",)
    assert result.content.decode() == (
        "module:\n  source_verification:\n"
        f"    corpus_citation_path: {PATH}\n    source_sha256: {PIN}\n"
    )


def test_inserts_both_lines_into_a_block_that_has_neither():
    text = "module:\n  source_verification:\n    note: kept # c\n"
    result = bind(text)
    assert result.edits == ("insert_corpus_citation_path+source_sha256",)
    assert result.content.decode() == (
        "module:\n  source_verification:\n"
        f"    corpus_citation_path: {PATH}\n    source_sha256: {PIN}\n"
        "    note: kept # c\n"
    )


def test_replaces_a_resolved_parent_path_with_the_requested_path():
    text = (
        "module:\n  source_verification:\n"
        f"    corpus_citation_path: '{RESOLVED}'  # parent\n"
    )
    result = bind(text, FALLBACK)
    assert result.edits == ("replace_corpus_citation_path", "insert_source_sha256")
    assert result.content.decode() == (
        "module:\n  source_verification:\n"
        f"    corpus_citation_path: {PATH}  # parent\n"
        f"    source_sha256: {PIN}\n"
    )


def test_keeps_crlf_line_endings():
    text = PATH_ONLY.replace("\n", "\r\n")
    result = bind(text)
    assert result.content.decode() == text.replace(
        f"    corpus_citation_path: {PATH}\r\n",
        f"    corpus_citation_path: {PATH}\r\n    source_sha256: {PIN}\r\n",
    )
    assert "\n" not in result.content.decode().replace("\r\n", "")


def test_keeps_a_missing_final_line_break():
    text = f"module:\n  source_verification:\n    corpus_citation_path: {PATH}"
    result = bind(text)
    assert result.content.decode() == text + f"\n    source_sha256: {PIN}"


def test_a_number_like_pin_is_quoted_and_reads_back_as_a_string():
    for pin in ("1" * 64, "12e" + "3" * 61):
        attestation = {**ATTESTATION, "source_sha256": pin}
        result = bind(PATH_ONLY, attestation)
        assert f"    source_sha256: '{pin}'\n" in result.content.decode()
        loaded = yaml.safe_load(result.content)
        assert loaded["module"]["source_verification"]["source_sha256"] == pin
        check_source_verification(result.content, attestation)


@pytest.mark.parametrize(
    "block",
    [
        f"    corpus_citation_path: {PATH}\n    source_sha256: {OTHER_PIN}\n",
        f"    corpus_citation_path: {PATH}\n    source_sha256: 12345\n",
        f"    corpus_citation_path: {PATH}\n    source_sha256: null\n",
        f"    source_sha256: {OTHER_PIN}\n",
        f"    corpus_citation_path: xx/statute/9\n    source_sha256: {PIN}\n",
        "    corpus_citation_path: xx/statute/9\n",
        f"    corpus_citation_path: [{PATH}]\n",
        f"    corpus_citation_paths:\n      - {PATH}\n",
    ],
)
def test_a_module_that_names_another_source_is_a_conflict(block):
    text = "module:\n  source_verification:\n" + block
    with pytest.raises(SourceBindingConflict) as caught:
        bind(text)
    assert caught.value.kind == "conflict"
    # No message carries module text.
    assert OTHER_PIN not in str(caught.value)
    assert "xx/statute/9" not in str(caught.value)


@pytest.mark.parametrize(
    "text, shape",
    [
        (
            f"module: {{source_verification: {{corpus_citation_path: {PATH}}}}}\n",
            "flow-style",
        ),
        (
            f"module:\n  source_verification: {{corpus_citation_path: {PATH}}}\n",
            "flow-style",
        ),
        (
            f"module:\n  source_verification: &pin\n    corpus_citation_path: {PATH}\n",
            "anchor",
        ),
        (
            f"module: &m\n  source_verification:\n    corpus_citation_path: {PATH}\n",
            "anchor",
        ),
        (
            f"base: &b\n  corpus_citation_path: {PATH}\n"
            "module:\n  source_verification: *b\n",
            "alias",
        ),
        (
            f"base: &b\n  corpus_citation_path: {PATH}\n"
            "module:\n  source_verification:\n    <<: *b\n",
            "merge key",
        ),
        (
            f"module:\n  source_verification:\n    corpus_citation_path: &p {PATH}\n",
            "anchor",
        ),
        (
            "module:\n  source_verification:\n"
            f"    corpus_citation_path: {PATH}\n    corpus_citation_path: {PATH}\n",
            "duplicate",
        ),
        (
            "module:\n  source_verification:\n"
            f"    corpus_citation_path: {PATH}\n    note: a\n    note: b\n",
            "duplicate",
        ),
        (
            f"module:\n  source_verification:\n    corpus_citation_path: {PATH}\n"
            "  source_verification:\n    note: x\n",
            "duplicate",
        ),
        (
            f"module:\n  summary: x\nmodule:\n  source_verification:\n"
            f"    corpus_citation_path: {PATH}\n",
            "duplicate",
        ),
        (
            f"module:\n  source_verification:\n    corpus_citation_path: >-\n      {PATH}\n",
            "multi-line",
        ),
        (
            f"module:\n  source_verification:\n    corpus_citation_path:\n      {PATH}\n",
            "multi-line",
        ),
        ("module:\n  source_verification:\n  summary: x\n", "empty"),
        ("module:\n  source_verification: text\n", "not a mapping"),
        ("module:\nrules: []\n", "empty"),
        ("module: text\n", "not a mapping"),
        ("format: rulespec/v1\nrules: []\n", "no module mapping"),
        (
            f"module:\r  source_verification:\r    corpus_citation_path: {PATH}\r",
            "line break",
        ),
    ],
)
def test_an_unhandled_shape_is_an_explicit_error_that_names_it(text, shape):
    with pytest.raises(SourceBindingShapeError) as caught:
        bind(text)
    assert caught.value.kind == "unsupported_shape"
    assert shape in str(caught.value)


@pytest.mark.parametrize(
    "raw",
    [
        b"module: [unclosed\n",
        b"\xff\xfe",
        b"",
        b"- a\n- b\n",
        b"module:\n  a: 1\n---\nmodule:\n  a: 2\n",
    ],
)
def test_bytes_that_are_not_one_yaml_mapping_are_an_input_error(raw):
    with pytest.raises(SourceBindingInputError) as caught:
        bind_source_verification(raw, ATTESTATION)
    assert caught.value.kind == "invalid_input"


@pytest.mark.parametrize(
    "attestation",
    [
        {},
        {**ATTESTATION, "source_sha256": PIN.upper()},
        {**ATTESTATION, "source_sha256": PIN[:-1]},
        {**ATTESTATION, "source_sha256": None},
        {**ATTESTATION, "requested_corpus_citation_path": ""},
        {**ATTESTATION, "requested_corpus_citation_path": "a\nb"},
        {**ATTESTATION, "resolved_corpus_citation_path": 3},
        None,
    ],
)
def test_an_incomplete_attestation_binds_nothing(attestation):
    with pytest.raises(SourceBindingInputError):
        bind_source_verification(PATH_ONLY.encode(), attestation)


def test_a_quoted_path_that_continues_on_the_next_line_is_refused():
    text = (
        "module:\n  source_verification:\n"
        '    corpus_citation_path: "xx/statute/\\\n      1/2"\n'
    )
    assert yaml.safe_load(text)["module"]["source_verification"] == {
        "corpus_citation_path": PATH
    }
    with pytest.raises(SourceBindingShapeError, match="multi-line"):
        bind(text)


def test_an_edit_that_does_not_read_back_is_refused_not_returned(monkeypatch):
    """The proof, not the planner, has the last word on a wrong edit."""
    from axiom_encode import source_verification_binding as binder

    original = binder._render_scalar
    monkeypatch.setattr(
        binder,
        "_render_scalar",
        lambda value: "wrong-value" if value == PIN else original(value),
    )
    with pytest.raises(SourceBindingShapeError, match="does not read back"):
        bind(PATH_ONLY)


def test_an_edit_that_changes_another_value_is_refused(monkeypatch):
    """A parse that differs anywhere outside the two keys fails the proof."""
    from axiom_encode import source_verification_binding as binder

    original = binder._plan

    def plan(text, requested, resolved, pin):
        edits = original(text, requested, resolved, pin)
        if edits:
            anchor = text.index("required: true")
            edits.append(binder._Edit("stray", anchor, anchor + 14, "required: 1"))
        return edits

    monkeypatch.setattr(binder, "_plan", plan)
    with pytest.raises(SourceBindingShapeError, match="does not read back"):
        bind(PATH_ONLY)


def test_an_unquoted_all_digit_pin_is_a_number_and_so_a_conflict():
    """YAML reads it as an integer; the same digits as a string do not match."""
    digits = "1" * 64
    text = PATH_ONLY.replace(
        f"    corpus_citation_path: {PATH}\n",
        f"    corpus_citation_path: {PATH}\n    source_sha256: {digits}\n",
    )
    assert isinstance(
        yaml.safe_load(text)["module"]["source_verification"]["source_sha256"], int
    )
    with pytest.raises(SourceBindingConflict):
        bind(text, {**ATTESTATION, "source_sha256": digits})


def test_the_proof_compares_values_with_their_types():
    from axiom_encode import source_verification_binding as binder

    assert binder._same_document({"a": [1, "x"]}, {"a": [1, "x"]})
    for left, right in (
        (1, True),
        (1, "1"),
        (1, 1.0),
        (None, "None"),
        ({"a": 1}, {"a": True}),
        ([1, 2], [2, 1]),
        ({"a": [1]}, {"a": (1.0,)}),
    ):
        assert not binder._same_document(left, right), (left, right)
    assert binder._same_document({"a": 1, "b": 2}, {"b": 2, "a": 1})


@pytest.mark.parametrize(
    "value, rendered",
    [
        (PATH, PATH),
        (PIN, PIN),
        ("1" * 64, "'" + "1" * 64 + "'"),
        ("12e" + "3" * 61, "'12e" + "3" * 61 + "'"),
        ("1.e3", "'1.e3'"),
        ("1.5e3", "'1.5e3'"),
        ("0o17", "'0o17'"),
        ("0x1f", "'0x1f'"),
        ("yes", "'yes'"),
        ("null", "'null'"),
        ("a b", "'a b'"),
        ("it's", "'it''s'"),
    ],
)
def test_a_value_any_yaml_reader_could_take_for_a_non_string_is_quoted(value, rendered):
    from axiom_encode import source_verification_binding as binder

    assert binder._render_scalar(value) == rendered
    assert yaml.safe_load(rendered) == value


@pytest.mark.parametrize(
    "text, detail",
    [
        ("---\n" + PATH_ONLY, "document marker"),
        (
            PATH_ONLY.replace("source_verification:", "source_verification: # c"),
            "plain",
        ),
        (PATH_ONLY.replace("module:", "module: !!map"), "plain"),
        (
            PATH_ONLY.replace("source_verification:", '"source_verification":'),
            "plain `name:` key",
        ),
    ],
)
def test_bound_for_yaml_but_not_plain_form_is_refused_and_says_so(text, detail):
    from axiom_encode.source_verification_binding import SourceBindingPlainFormError

    bound = text.replace(
        f"    corpus_citation_path: {PATH}\n",
        f"    corpus_citation_path: {PATH}\n    source_sha256: {PIN}\n",
    )
    assert yaml.safe_load(bound)["module"]["source_verification"] == {
        "corpus_citation_path": PATH,
        "source_sha256": PIN,
    }
    for candidate in (text, bound):
        with pytest.raises(SourceBindingPlainFormError) as caught:
            bind(candidate)
        assert caught.value.kind == "not_plain_form"
        assert "not in the plain block form" in str(caught.value)
        assert detail in str(caught.value)


def test_the_line_reader_must_read_the_attested_values_back(monkeypatch):
    """Not raising is not enough: the reader's values are part of the proof."""
    from axiom_encode import source_verification_binding as binder
    from axiom_encode.source_verification_binding import SourceBindingPlainFormError

    monkeypatch.setattr(
        binder._admission, "read_candidate_source_pin", lambda raw: (PATH, OTHER_PIN)
    )
    with pytest.raises(SourceBindingPlainFormError, match="does not read the attested"):
        bind(PATH_ONLY)


def test_the_binding_check_is_declared_and_blocking():
    assert CHECK_SOURCE_BINDING.id == "source-binding/unbound"
    assert CHECK_SOURCE_BINDING.enforcement == "blocking"
    assert "bind-source-verification" in CHECK_SOURCE_BINDING.fix


# --- property: a result differs from its input only in the intended lines ---

_WORDS = st.sampled_from(
    ["alpha", "beta: gamma", "# hash", "tab\there", "'quoted'", "50%", "- dash", "é ü"]
)
_SCALAR_LINES = st.lists(_WORDS, min_size=1, max_size=4)

BLOCK_VARIANTS = (
    "absent",
    "path",
    "path_quoted_comment",
    "pin_only",
    "both",
    "both_reversed",
    "other_keys",
    "resolved_path",
    "wrong_pin",
    "wrong_path",
    "flow",
    "anchor",
    "duplicate",
    "plural",
    "folded_path",
)


def _block_lines(variant: str, key: str, inner: str) -> list[str]:
    head = f"{key}source_verification:"
    path = f"{inner}corpus_citation_path: {PATH}"
    pin = f"{inner}source_sha256: {PIN}"
    return {
        "absent": [],
        "path": [head, path],
        "path_quoted_comment": [head, f'{inner}corpus_citation_path: "{PATH}" # c'],
        "pin_only": [head, f"{inner}# lead", pin],
        "both": [head, path, pin],
        "both_reversed": [head, pin, f"{inner}note: n", path],
        "other_keys": [head, f"{inner}note: n", f"{inner}more: m"],
        "resolved_path": [head, f"{inner}corpus_citation_path: {RESOLVED}"],
        "wrong_pin": [head, path, f"{inner}source_sha256: {OTHER_PIN}"],
        "wrong_path": [head, f"{inner}corpus_citation_path: xx/other/1"],
        "flow": [f"{head} {{corpus_citation_path: {PATH}}}"],
        "anchor": [f"{head} &anchor", path],
        "duplicate": [head, path, path],
        "plural": [head, f"{inner}corpus_citation_paths: [{PATH}]"],
        "folded_path": [head, f"{inner}corpus_citation_path: >-", f"{inner}  {PATH}"],
    }[variant]


EXPECTED = {
    "absent": ("create_source_verification",),
    "path": ("insert_source_sha256",),
    "path_quoted_comment": ("insert_source_sha256",),
    "pin_only": ("insert_corpus_citation_path",),
    "both": (),
    "both_reversed": (),
    "other_keys": ("insert_corpus_citation_path+source_sha256",),
    "resolved_path": ("replace_corpus_citation_path", "insert_source_sha256"),
    "wrong_pin": SourceBindingConflict,
    "wrong_path": SourceBindingConflict,
    "flow": SourceBindingShapeError,
    "anchor": SourceBindingShapeError,
    "duplicate": SourceBindingShapeError,
    "plural": SourceBindingConflict,
    "folded_path": SourceBindingShapeError,
}


@st.composite
def module_texts(draw):
    step = draw(st.sampled_from([2, 4]))
    key, inner = " " * step, " " * (2 * step)
    variant = draw(st.sampled_from(BLOCK_VARIANTS))
    sections = [
        [f"{key}summary: |-"] + [f"{inner}{line}" for line in draw(_SCALAR_LINES)],
        [f"{key}proof_validation:", f"{inner}required: true"],
        [f"{key}# a comment line", f'{key}note: "a # b"'],
        [f"{key}deferred_outputs:", f"{inner}- output: x", f"{inner}  reason: >-"]
        + [f"{inner}    {line}" for line in draw(_SCALAR_LINES)],
    ]
    chosen = draw(
        st.lists(st.sampled_from(range(len(sections))), unique=True, max_size=4)
    )
    children = [sections[index] for index in chosen]
    block = _block_lines(variant, key, inner)
    if block:
        children.insert(draw(st.integers(0, len(children))), block)
    if not children:
        children = [[f"{key}summary: x"]]
    lines = ["format: rulespec/v1"]
    if draw(st.booleans()):
        lines.append("# header comment: with a colon")
    lines.append("module:")
    for child in children:
        lines.extend(child)
        if draw(st.booleans()):
            lines.append("")
    if draw(st.booleans()):
        lines += ["rules:", "  - name: r", "    source: |", "      text\twith a tab"]
    newline = draw(st.sampled_from(["\n", "\r\n"]))
    text = newline.join(lines)
    module_is_last = lines[-1].startswith(" ") or lines[-1] == ""
    if draw(st.booleans()) or not module_is_last:
        text += newline
    return variant, text, newline


@settings(
    max_examples=400,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(module_texts())
def test_property_minimal_edit_or_explicit_error(case):
    variant, text, newline = case
    raw = text.encode("utf-8")
    expected = EXPECTED[variant]
    # The stdlib line reader of the promotion lane never reads a value that
    # the YAML parser does not: it agrees with the parser or it refuses.
    try:
        line_read = read_candidate_source_pin(raw)
    except AdmissionError:
        line_read = None
    if line_read is not None:
        parsed = yaml.safe_load(text)["module"]["source_verification"]
        assert line_read == (
            parsed.get("corpus_citation_path"),
            parsed.get("source_sha256"),
        )
    if isinstance(expected, type):
        with pytest.raises(expected):
            bind_source_verification(raw, FALLBACK)
        return
    result = bind_source_verification(raw, FALLBACK)
    assert result.edits == expected
    bound = result.content.decode("utf-8")
    if not expected:
        assert result.content == raw
        return

    # 1. Only inserted or replaced lines differ; every other line is the input's.
    # A file whose last line has no line break keeps it that way; give both
    # sides the same final break so the comparison is line by line.
    final = "" if text.endswith(newline) else newline
    assert bound.endswith(newline) == text.endswith(newline)
    before = (text + final).splitlines(keepends=True)
    after = (bound + final).splitlines(keepends=True)
    added, removed = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, before, after, autojunk=False
    ).get_opcodes():
        if tag != "equal":
            removed += before[i1:i2]
            added += after[j1:j2]
    stripped = [line.strip() for line in added]
    path_line = f"corpus_citation_path: {PATH}"
    pin_line = f"source_sha256: {PIN}"
    wanted = {
        ("create_source_verification",): ["source_verification:", path_line, pin_line],
        ("insert_source_sha256",): [pin_line],
        ("insert_corpus_citation_path",): [path_line],
        ("insert_corpus_citation_path+source_sha256",): [path_line, pin_line],
        ("replace_corpus_citation_path", "insert_source_sha256"): [
            path_line,
            pin_line,
        ],
    }[expected]
    assert stripped == wanted
    if variant == "resolved_path":
        assert [line.strip() for line in removed] == [
            f"corpus_citation_path: {RESOLVED}"
        ]
    else:
        assert removed == []
    # 2. Never a re-dump: the size grows by the inserted text and nothing else.
    assert len(bound) - len(text) == sum(map(len, added)) - sum(map(len, removed))
    # 3. The line-ending style is kept.
    assert ("\r\n" in bound) == (newline == "\r\n")
    assert "\n" not in bound.replace("\r\n", "") or newline == "\n"
    assert bound.endswith(("\n", "\r\n")) == text.endswith(("\n", "\r\n"))
    # 4. The parsed document is the input's plus the two keys.
    wanted_document = yaml.safe_load(text)
    wanted_document["module"].setdefault("source_verification", {}).update(
        corpus_citation_path=PATH, source_sha256=PIN
    )
    assert yaml.safe_load(bound) == wanted_document
    # 5. Binding is a fixed point.
    assert bind_source_verification(result.content, FALLBACK).content == result.content
    # 6. The promotion lane's line reader reads the bound pin back.
    assert read_candidate_source_pin(result.content) == (PATH, PIN)
