"""Bind a RuleSpec module to its resolver-attested source by a minimal text edit.

A promotable candidate carries the source pin of its generation input::

    module:
      source_verification:
        corpus_citation_path: <the citation the resolver was asked for>
        source_sha256: <SHA-256 of the resolver's stored source body>

``bind_source_verification`` sets exactly these two values. It inserts or
replaces only their lines inside ``module.source_verification`` (it creates the
block when the module has none) and keeps every other byte: comments, block
scalars, key order, indentation and line endings. It never loads and re-dumps
the document.

The edit is proven before it is returned:

* ``yaml.safe_load`` of the result equals ``yaml.safe_load`` of the input,
  except for the two keys (compared with types, so ``1`` is not ``True``);
* a second structural read of the result finds each key once, as a string,
  with exactly the attested value, and has nothing left to edit.

A module that the minimal edit cannot handle safely is refused with an error
that names the reason. Nothing is rewritten in that case:

* ``SourceBindingConflict``: the module already names another source (a
  ``corpus_citation_path`` outside the attestation, or a ``source_sha256`` that
  differs from the attested one). The candidate claims a different source, so
  the pin is never overwritten.
* ``SourceBindingShapeError``: the text is in a shape this editor does not
  edit (flow-style mapping, anchor, alias, merge key, duplicate key, multi-line
  value, a line break other than LF or CRLF). The message names the shape.
* ``SourceBindingInputError``: the bytes are not one UTF-8 YAML mapping, or
  the attestation is incomplete.

No message contains module text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

import yaml

from . import reviewed_pr_admission as _admission
from .checks import declare_check

__all__ = [
    "CHECK_SOURCE_BINDING",
    "SourceBinding",
    "SourceBindingConflict",
    "SourceBindingError",
    "SourceBindingInputError",
    "SourceBindingPlainFormError",
    "SourceBindingShapeError",
    "bind_source_verification",
    "check_source_verification",
]

_BLOCK = "source_verification"
_PATH_KEY = "corpus_citation_path"
_PIN_KEY = "source_sha256"
_PLURAL_PATH_KEY = "corpus_citation_paths"
_STR_TAG = "tag:yaml.org,2002:str"
_MAP_TAG = "tag:yaml.org,2002:map"
_NULL_TAG = "tag:yaml.org,2002:null"
_MERGE_TAG = "tag:yaml.org,2002:merge"
# Every character PyYAML's scanner ends a line at.
_LINE_BREAKS = "\n\r\x85\u2028\u2029"
_SHA256 = re.compile(r"[0-9a-f]{64}")
# A plain scalar that needs no quotes: no space, colon, hash or indicator.
_PLAIN_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@+-]*")
_QUOTABLE = re.compile(r"[\x20-\x7e]+")


CHECK_SOURCE_BINDING = declare_check(
    "source-binding/unbound",
    title="Generated RuleSpec is not bound to its resolver-attested source",
    requires=(
        "`module.source_verification` is a plain block mapping whose "
        "`corpus_citation_path` and `source_sha256` are the resolver's "
        "attested values, or a module the encoder can bind by inserting those "
        "two lines."
    ),
    why=(
        "A candidate is promoted and signed as the bytes it is. Without the "
        "attested source pin the protected binding check refuses it, after "
        "the signing approval, and a pin that names another source would bind "
        "the rule to text it was not encoded from."
    ),
    fix=(
        "Write `module.source_verification` as a plain block mapping with "
        "the one line `corpus_citation_path: <the requested citation path>`. "
        "Leave `source_sha256` out: the encoder writes it. Do not use flow "
        "style, anchors, tags, merge keys, quoted keys or a multi-line value "
        "there, and no `---` line. For an existing candidate run "
        "`axiom-encode bind-source-verification`."
    ),
    cause="encoder_output",
    gate="ci",
    site="axiom_encode/harness/evals.py:_bind_generated_source_verification",
)


class SourceBindingError(ValueError):
    """The module cannot be bound. ``kind`` is a short machine token."""

    kind = "error"


class SourceBindingInputError(SourceBindingError):
    """The module bytes or the attestation cannot be read."""

    kind = "invalid_input"


class SourceBindingConflict(SourceBindingError):
    """The module names another source than the attestation."""

    kind = "conflict"


class SourceBindingShapeError(SourceBindingError):
    """The module text is in a shape the minimal edit does not handle."""

    kind = "unsupported_shape"


class SourceBindingPlainFormError(SourceBindingShapeError):
    """The module is bound for a YAML parser, but not in the plain block form.

    The promotion lane reads the pin with a standard-library line reader
    (``reviewed_pr_admission.read_candidate_source_pin``) before any approval.
    A module that this reader refuses is not promotable as it stands.
    """

    kind = "not_plain_form"


@dataclass(frozen=True)
class SourceBinding:
    """The bound module bytes and what the edit did.

    ``edits`` is empty when the module was already bound; ``content`` is then
    the input, byte for byte.
    """

    content: bytes
    edits: tuple[str, ...]
    corpus_citation_path: str
    source_sha256: str

    @property
    def changed(self) -> bool:
        return bool(self.edits)


@dataclass(frozen=True)
class _Edit:
    name: str
    start: int
    end: int
    text: str


def _attested_values(attestation: Mapping[str, object]) -> tuple[str, str, str]:
    if not isinstance(attestation, Mapping):
        raise SourceBindingInputError("source attestation is not a mapping")
    requested = attestation.get("requested_corpus_citation_path")
    resolved = attestation.get("resolved_corpus_citation_path", requested)
    pin = attestation.get("source_sha256")
    for label, value in (
        ("requested_corpus_citation_path", requested),
        ("resolved_corpus_citation_path", resolved),
    ):
        if type(value) is not str or not value or _QUOTABLE.fullmatch(value) is None:
            raise SourceBindingInputError(f"source attestation has no usable {label}")
    if type(pin) is not str or _SHA256.fullmatch(pin) is None:
        raise SourceBindingInputError(
            "source attestation has no 64-character lowercase hex source_sha256"
        )
    assert type(requested) is str and type(resolved) is str
    return requested, resolved, pin


def _render_scalar(value: str) -> str:
    """The shortest YAML scalar that every reader loads as this string."""

    # Unquoted only when no YAML 1.1 or 1.2 reader can take the text for a
    # number, a boolean or null (`1e3`, `1.5e3`, `0o17`, `0x1f`, `yes`).
    if (
        _PLAIN_SAFE.fullmatch(value) is not None
        and _admission.plain_scalar_is_string(value)
        and yaml.safe_load(value) == value
    ):
        return value
    return "'" + value.replace("'", "''") + "'"


def _typed(value: object) -> object:
    """A comparable form that keeps types apart (``1``, ``True``, ``1.0``).

    A mapping's key order is not part of the form; a sequence's order is.
    """

    if isinstance(value, dict):
        return (
            "map",
            tuple(
                sorted(
                    (repr(_typed(key)), repr(_typed(item)))
                    for key, item in value.items()
                )
            ),
        )
    if isinstance(value, (list, tuple)):
        return ("seq", tuple(_typed(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted(repr(_typed(item)) for item in value)))
    return (type(value).__name__, repr(value))


def _same_document(left: object, right: object) -> bool:
    return _typed(left) == _typed(right)


def _load(text: str) -> object:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        raise SourceBindingInputError("module is not valid YAML") from None
    except RecursionError:
        raise SourceBindingInputError("module YAML nests too deeply") from None


class _Document:
    """The composed node tree of one module text, with the anchored positions."""

    def __init__(self, text: str):
        self.text = text
        try:
            self.root = yaml.compose(text, Loader=yaml.SafeLoader)
            self.anchored = frozenset(
                event.start_mark.index
                for event in yaml.parse(text, Loader=yaml.SafeLoader)
                if getattr(event, "anchor", None) is not None
            )
        except yaml.YAMLError:
            raise SourceBindingInputError("module is not valid YAML") from None
        except RecursionError:
            raise SourceBindingInputError("module YAML nests too deeply") from None

    def line_start(self, index: int) -> int:
        while index > 0 and self.text[index - 1] not in _LINE_BREAKS:
            index -= 1
        return index

    def line_end(self, index: int) -> int:
        while index < len(self.text) and self.text[index] not in _LINE_BREAKS:
            index += 1
        return index

    def indentation(self, key: yaml.Node, label: str) -> str:
        """The spaces before a key that starts its own line."""

        start = self.line_start(key.start_mark.index)
        prefix = self.text[start : key.start_mark.index]
        if prefix.strip(" "):
            raise SourceBindingShapeError(f"{label} does not start its own line")
        return prefix

    def break_before(self, line_start: int) -> str:
        """The LF or CRLF that ends the line before ``line_start``."""

        if line_start >= 1 and self.text[line_start - 1] == "\n":
            if line_start >= 2 and self.text[line_start - 2] == "\r":
                return "\r\n"
            return "\n"
        raise SourceBindingShapeError(
            "a line break other than LF or CRLF next to module.source_verification"
        )

    def block_mapping(self, node: yaml.Node, label: str) -> yaml.MappingNode:
        if node.start_mark.index in self.anchored:
            raise SourceBindingShapeError(f"an anchor or alias on {label}")
        if isinstance(node, yaml.ScalarNode) and node.tag == _NULL_TAG:
            raise SourceBindingShapeError(f"{label} is empty")
        if not isinstance(node, yaml.MappingNode) or node.tag != _MAP_TAG:
            raise SourceBindingShapeError(f"{label} is not a mapping")
        if node.flow_style is not False:
            raise SourceBindingShapeError(f"a flow-style {label} mapping")
        for key, _value in node.value:
            if key.tag == _MERGE_TAG:
                raise SourceBindingShapeError(f"a merge key in {label}")
            if not isinstance(key, yaml.ScalarNode):
                raise SourceBindingShapeError(f"a non-scalar key in {label}")
            if key.start_mark.index in self.anchored:
                raise SourceBindingShapeError(f"an anchor on a key of {label}")
        return node

    def entry(
        self, mapping: yaml.MappingNode, name: str, label: str
    ) -> tuple[yaml.ScalarNode, yaml.Node] | None:
        found = [
            (key, value)
            for key, value in mapping.value
            if key.tag == _STR_TAG and key.value == name
        ]
        if len(found) > 1:
            raise SourceBindingShapeError(f"a duplicate {label} key")
        return found[0] if found else None


def _single_line_string(
    document: _Document, key: yaml.ScalarNode, value: yaml.Node, label: str
) -> str | None:
    """The string value of a one-line entry, or ``None`` when it is no string."""

    if value.start_mark.index in document.anchored:
        raise SourceBindingShapeError(f"an anchor or alias on {label}")
    if value.start_mark.index < key.end_mark.index:
        raise SourceBindingShapeError(f"an alias as the value of {label}")
    if not isinstance(value, yaml.ScalarNode) or value.tag != _STR_TAG:
        return None
    if (
        value.style not in (None, "'", '"')
        or value.start_mark.line != value.end_mark.line
        or key.start_mark.line != value.start_mark.line
    ):
        raise SourceBindingShapeError(f"a multi-line or block-scalar {label}")
    return value.value


def _plan(text: str, requested: str, resolved: str, pin: str) -> list[_Edit]:
    """The edits that bind ``text``; empty when it is already bound."""

    document = _Document(text)
    root = document.root
    if root is None:
        raise SourceBindingInputError("module is empty")
    if not isinstance(root, yaml.MappingNode):
        raise SourceBindingInputError("module file is not a YAML mapping")
    root = document.block_mapping(root, "document-root")
    module_entry = document.entry(root, "module", "module")
    if module_entry is None:
        raise SourceBindingShapeError("the file has no module mapping")
    module_key, module_value = module_entry
    module = document.block_mapping(module_value, "module")
    module_indent = document.indentation(module_key, "the module key")
    first_key = module.value[0][0]
    child_indent = document.indentation(first_key, "the first key of module")
    block_entry = document.entry(module, _BLOCK, f"module.{_BLOCK}")
    path_line = f"{_PATH_KEY}: {_render_scalar(requested)}"
    pin_line = f"{_PIN_KEY}: {_render_scalar(pin)}"

    if block_entry is None:
        step = len(child_indent) - len(module_indent)
        if step <= 0:
            raise SourceBindingShapeError("module keys are not indented under module")
        position = document.line_start(first_key.start_mark.index)
        newline = document.break_before(position)
        inner = child_indent + " " * step
        return [
            _Edit(
                "create_source_verification",
                position,
                position,
                f"{child_indent}{_BLOCK}:{newline}"
                f"{inner}{path_line}{newline}"
                f"{inner}{pin_line}{newline}",
            )
        ]

    block_key, block_value = block_entry
    if block_value.start_mark.index < block_key.end_mark.index:
        raise SourceBindingShapeError(f"an alias as the value of module.{_BLOCK}")
    block = document.block_mapping(block_value, f"module.{_BLOCK}")
    names = [key.value for key, _value in block.value if key.tag == _STR_TAG]
    if len(set(names)) != len(names):
        raise SourceBindingShapeError(f"a duplicate key in module.{_BLOCK}")
    if _PLURAL_PATH_KEY in names:
        raise SourceBindingConflict(
            f"module.{_BLOCK}.{_PLURAL_PATH_KEY} is not supported; use exactly "
            f"one resolver-attested {_PATH_KEY}"
        )
    first_block_key = block.value[0][0]
    block_indent = document.indentation(
        first_block_key, f"the first key of module.{_BLOCK}"
    )
    path_entry = document.entry(block, _PATH_KEY, f"module.{_BLOCK}.{_PATH_KEY}")
    pin_entry = document.entry(block, _PIN_KEY, f"module.{_BLOCK}.{_PIN_KEY}")

    if pin_entry is not None:
        declared_pin = _single_line_string(
            document, *pin_entry, f"module.{_BLOCK}.{_PIN_KEY}"
        )
        if declared_pin != pin:
            raise SourceBindingConflict(
                f"module.{_BLOCK}.{_PIN_KEY} is present and differs from the "
                "attested source_sha256; the candidate claims another source"
            )

    edits: list[_Edit] = []
    if path_entry is not None:
        path_key, path_value = path_entry
        label = f"module.{_BLOCK}.{_PATH_KEY}"
        declared_path = _single_line_string(document, path_key, path_value, label)
        if declared_path not in {requested, resolved}:
            raise SourceBindingConflict(
                f"{label} names a corpus path that the source attestation does "
                "not bind; the candidate claims another source"
            )
        indent = document.indentation(path_key, label)
        if declared_path != requested:
            edits.append(
                _Edit(
                    f"replace_{_PATH_KEY}",
                    path_value.start_mark.index,
                    path_value.end_mark.index,
                    _render_scalar(requested),
                )
            )
        if pin_entry is None:
            end = document.line_end(path_value.end_mark.index)
            newline = document.break_before(
                document.line_start(path_key.start_mark.index)
            )
            if end == len(text):
                # The path line ends the file without a line break; keep it so.
                edits.append(
                    _Edit(
                        f"insert_{_PIN_KEY}", end, end, f"{newline}{indent}{pin_line}"
                    )
                )
            else:
                after = end + (2 if text.startswith("\r\n", end) else 1)
                if text[end:after] not in ("\n", "\r\n"):
                    raise SourceBindingShapeError(
                        f"a line break other than LF or CRLF next to module.{_BLOCK}"
                    )
                edits.append(
                    _Edit(
                        f"insert_{_PIN_KEY}",
                        after,
                        after,
                        f"{indent}{pin_line}{text[end:after]}",
                    )
                )
        return edits

    position = document.line_start(first_block_key.start_mark.index)
    newline = document.break_before(position)
    inserted = f"{block_indent}{path_line}{newline}"
    name = f"insert_{_PATH_KEY}"
    if pin_entry is None:
        inserted += f"{block_indent}{pin_line}{newline}"
        name += f"+{_PIN_KEY}"
    return [_Edit(name, position, position, inserted)]


def _expected_document(text: str, requested: str, pin: str) -> object:
    expected = _load(text)
    assert isinstance(expected, dict) and isinstance(expected.get("module"), dict)
    block = expected["module"].setdefault(_BLOCK, {})
    assert isinstance(block, dict)
    block[_PATH_KEY] = requested
    block[_PIN_KEY] = pin
    return expected


def _require_plain_form(content: bytes, requested: str, pin: str) -> None:
    """Refuse a bound module that the promotion lane's line reader would refuse.

    So "the binder accepts it" implies "admission and the pre-approval step
    read the same pin".
    """

    try:
        read = _admission.read_candidate_source_pin(content)
    except _admission.AdmissionError as exc:
        detail = getattr(exc, "detail", "it is not in the plain block form")
    else:
        if read == (requested, pin):
            return
        detail = "the line reader does not read the attested values back"
    raise SourceBindingPlainFormError(
        f"module.{_BLOCK} carries the attested values for a YAML parser, but the "
        f"file is not in the plain block form that the promotion lane reads: "
        f"{detail}"
    )


def bind_source_verification(
    module: bytes, attestation: Mapping[str, object]
) -> SourceBinding:
    """Return ``module`` bound to ``attestation`` by a minimal, proven text edit.

    ``attestation`` is the resolver's ``source_attestation``: the module gets
    its ``requested_corpus_citation_path`` and its ``source_sha256``. A module
    that names the ``resolved_corpus_citation_path`` instead gets that one value
    replaced. Raises a ``SourceBindingError`` subclass and changes nothing when
    the module cannot be bound safely.
    """

    requested, resolved, pin = _attested_values(attestation)
    if not isinstance(module, (bytes, bytearray)):
        raise SourceBindingInputError("module must be bytes")
    try:
        text = bytes(module).decode("utf-8")
    except UnicodeDecodeError:
        raise SourceBindingInputError("module is not UTF-8") from None
    edits = _plan(text, requested, resolved, pin)
    if not edits:
        _require_plain_form(bytes(module), requested, pin)
        return SourceBinding(bytes(module), (), requested, pin)
    bound = text
    for edit in sorted(edits, key=lambda item: item.start, reverse=True):
        bound = bound[: edit.start] + edit.text + bound[edit.end :]
    # Proof. A failure here is a shape this editor mishandled, never a reason
    # to fall back to a load-and-dump rewrite.
    try:
        remaining = _plan(bound, requested, resolved, pin)
        proven = not remaining and _same_document(
            _load(bound), _expected_document(text, requested, pin)
        )
    except SourceBindingError:
        proven = False
    if not proven:
        raise SourceBindingShapeError(
            f"a module.{_BLOCK} layout whose minimal edit does not read back "
            "as the attested values"
        )
    _require_plain_form(bound.encode("utf-8"), requested, pin)
    return SourceBinding(
        bound.encode("utf-8"),
        tuple(edit.name for edit in sorted(edits, key=lambda item: item.start)),
        requested,
        pin,
    )


def check_source_verification(module: bytes, attestation: Mapping[str, object]) -> None:
    """Raise unless ``module`` already carries exactly the attested pin.

    This reads only; it is the question "would the binder change anything".
    """

    binding = bind_source_verification(module, attestation)
    if binding.changed:
        created = any(edit.startswith("create") for edit in binding.edits)
        missing = [
            key
            for key in (_PATH_KEY, _PIN_KEY)
            if created or any(key in edit for edit in binding.edits)
        ]
        error = SourceBindingError(
            f"module.{_BLOCK} is not bound to the attested source: "
            + " and ".join(missing)
            + " is missing or not the attested value"
        )
        error.kind = "unbound"
        raise error
