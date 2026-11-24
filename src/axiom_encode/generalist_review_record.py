"""The generalist review record for one model-manifest unit (#2146).

An encode runs one generalist model review of its candidate. Until this
record existed, a review that could not start, could not be read or was
skipped left no trace in the signed apply manifest: the module was applied
and nothing downstream could tell whether a model had reviewed it.

The record states what happened, in one of five statuses:

``passed``
    The reviewer ran, its output was read, it passed with its own boolean
    ``passed: true`` and a finite score of at least 7.0, and it reviewed
    every applied file listed by this record in full. A pass that the eval
    pipeline infers from the score alone is not a pass here.
``partial``
    The reviewer passed the candidate it read, but that is not everything:
    another file is applied that it never read, or its prompts showed it
    only a part of a file or of the source excerpt.
``failed``
    The reviewer ran on applied content and did not pass.
``unavailable``
    No usable verdict on the applied content: the reviewer could not start,
    its output could not be read, the result was malformed, or the reviewed
    content differs from the applied content.
``skipped``
    The reviewer was not asked: ``--skip-reviewers``, or the encoder skipped
    it because deterministic validation had rejected the standalone candidate.

Only ``passed`` satisfies ``encode --apply --require-review``. No code path
here turns a missing or unreadable value into ``passed``.

Legacy replacement transactions can also rewrite exact dependents outside the
model manifest's planned files. In record-only mode this record describes the
model-manifest unit, not that whole transaction; dependent migration manifests
carry no generalist-review record. The CLI and direct apply entrypoint refuse
that migration path when the caller requires review.

A verdict covers paths, not bytes. The record lists every applied file by
manifest path and byte digest, and marks as reviewed only the module path
the reviewer was asked about and the companion test path of that module. A
file at another path stays unreviewed also when its bytes equal a reviewed
file: its path selects the module that supplies its validation context, and
the reviewer did not read that context.

"The same content" is decided by a content digest, not by file bytes. Between
the review and the apply, ``encode --apply`` stamps
``module.source_verification`` with the resolver's attested corpus path and
source hash and dumps the YAML again. A canonical stamp may preserve the bytes;
other stamps can change
scalar values as well as file bytes. The content digest is the SHA-256 of the YAML
node tree (each scalar with its resolved type and its exact text), with those
two stamped keys left out. Key order, harmless quoting and comments do not
change it; other edits do, including one that only a YAML 1.1 reader
treats as equal (``yes`` and ``true``, ``010`` and ``8``). Each scalar also
carries the type that ``yaml.safe_load`` gives it, because that loader reads
the companion tests: plain ``yes``, ``0b10`` and ``2001-01-01`` differ from
their quoted spellings, so two files with one digest load as equal values
there. A plain float that every modeled reader reads as one value is the one
exception to "exact text": its content is the value (``0.20`` and ``0.2``),
which the apply dump respells. Mapping key order is the one difference of the loaded value that the
digest ignores. A document with a
repeated key, a collection alias or a merge key has no content digest. Keys
are repeated when their node is the same, and also when a reader gives two
scalar keys equal values (``0x10`` and ``16``, ``1`` and ``1.0``): the loaded
value would depend on the key order. Three readers are compared: PyYAML's
``safe_load`` (YAML 1.1), a YAML 1.2 core reader, and a reader that accepts
the typed forms of both (the rules engine's serde_yaml reads ``0o20`` and
``0b10000`` as one number). No other dialect is modeled. A key that is not a
scalar, or that one of the first two readers cannot construct (a ``!!value``,
``!!timestamp`` or local tag), leaves the document without a digest.

An applied file is bound only when its content digest equals that of the
reviewed file. A different dump-derived digest is never an alternate binding:
the dump may collapse keys or change YAML 1.2 types. Scalar respellings that
this digest cannot establish as equal leave the verdict unavailable, even
when a particular respelling might be harmless. The record also names byte
digests on both sides.

A candidate that does not fit one reviewer call is reviewed in ordered
chunks (``generalist_review_chunks``). ``coverage`` says what the calls
showed: the number of calls, and for the module, the companion test and the
source excerpt the size and digest of the reviewed text and the byte range
and digest of each chunk. A file is ``complete`` only when its chunks cover
every byte of the reviewed text exactly once, in order, within the chunk and
call caps, and every planned call was made. The ranges are of the reviewed
bytes; the content digests above bind those to the applied file. The source
excerpt is not an applied file: a review that saw only a part of it is
``partial`` with its own reason.

The record holds only enumerated words, numbers, digests and the manifest's
own file paths. Reviewer prose (issues, reasoning) never enters it, so model
output cannot place a string of its choice in a signed manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping

import yaml

from axiom_encode.checks import ValidationIssue, declare_check
from axiom_encode.constants import RULESPEC_FILE_SUFFIX, RULESPEC_TEST_FILE_SUFFIX
from axiom_encode.generalist_review_chunks import (
    MAX_CHUNKS_PER_FILE,
    MAX_REVIEW_CALLS,
    planned_call_count,
)
from axiom_encode.yaml_serialization import is_yaml12_core_non_string, safe_dump_yaml

GENERALIST_REVIEW_FIELD = "generalist_review"
GENERALIST_REVIEW_SCHEMA = "axiom-encode/generalist-review/v2"
GENERALIST_REVIEWER_NAME = "generalist-reviewer"

STATUS_PASSED = "passed"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"
STATUS_UNAVAILABLE = "unavailable"
STATUS_SKIPPED = "skipped"
STATUSES = frozenset(
    {
        STATUS_PASSED,
        STATUS_PARTIAL,
        STATUS_FAILED,
        STATUS_UNAVAILABLE,
        STATUS_SKIPPED,
    }
)

REASON_SKIP_FLAG = "skip_reviewers_flag"
REASON_NOT_RUN = "reviewer_not_run"
REASON_REVIEWER_ERROR = "reviewer_error"
REASON_UNREADABLE = "unreadable_review_result"
REASON_CONTENT_NOT_APPLIED = "reviewed_content_not_applied"
REASON_UNREVIEWED_FILES = "unreviewed_applied_files"
REASON_TRUNCATED = "review_saw_truncated_file"
REASON_SOURCE_TRUNCATED = "review_saw_truncated_source"
_REASONS_BY_STATUS = {
    STATUS_PASSED: frozenset({None}),
    STATUS_PARTIAL: frozenset(
        {REASON_UNREVIEWED_FILES, REASON_TRUNCATED, REASON_SOURCE_TRUNCATED}
    ),
    STATUS_FAILED: frozenset({None}),
    STATUS_UNAVAILABLE: frozenset(
        {REASON_REVIEWER_ERROR, REASON_UNREADABLE, REASON_CONTENT_NOT_APPLIED}
    ),
    STATUS_SKIPPED: frozenset({REASON_SKIP_FLAG, REASON_NOT_RUN}),
}
# Statuses whose verdict is on applied content.
_BOUND_STATUSES = frozenset({STATUS_PASSED, STATUS_PARTIAL, STATUS_FAILED})

_RECORD_WHY = (
    "The record is the only statement in a signed manifest about whether a "
    "model reviewed the applied module; a wrong record reads as a review."
)
_RECORD_FIX = (
    "Do not edit a signed manifest. Run encode --apply again so the encoder "
    "writes and signs the record itself."
)
_RECORD_SITE = (
    "axiom_encode/generalist_review_record.py:generalist_review_record_issues"
)
CHECK_REVIEW_RECORD_SHAPE = declare_check(
    "manifest:generalist-review/shape",
    title="Generalist review record is malformed",
    requires=(
        "The record has the exact v2 fields, a supported status and reason, "
        "and well-formed reviewer, verdict, digest, coverage and file values."
    ),
    why=_RECORD_WHY,
    fix=_RECORD_FIX,
    cause="encoder_output",
    gate="manifest",
    site=_RECORD_SITE,
)
CHECK_REVIEW_RECORD_VERDICT = declare_check(
    "manifest:generalist-review/verdict",
    title="Generalist review status contradicts its verdict",
    requires=(
        "A passed, partial or failed status carries the matching verdict, a "
        "pass is at or above the threshold, an unscored status carries no "
        "verdict, and a required review is passed."
    ),
    why=_RECORD_WHY,
    fix=_RECORD_FIX,
    cause="encoder_output",
    gate="manifest",
    site=_RECORD_SITE,
)
CHECK_REVIEW_RECORD_BINDING = declare_check(
    "manifest:generalist-review/binding",
    title="Generalist review verdict is not bound to the applied files",
    requires=(
        "The record lists exactly the files the manifest applies, each by "
        "path and digest; a passed record marks every one of them reviewed in "
        "full, with chunk ranges that cover each reviewed text exactly once; a "
        "scored record has equal reviewed and applied content digests "
        "and marks as reviewed only the reviewed module path and its "
        "companion test path."
    ),
    why=_RECORD_WHY,
    fix=_RECORD_FIX,
    cause="encoder_output",
    gate="manifest",
    site=_RECORD_SITE,
)

PASS_SCORE_THRESHOLD = 7.0
# Evidence key: the boolean `passed` the reviewer wrote itself, or None.
EVIDENCE_EXPLICIT_PASSED = "explicit_passed"
_MAX_SCORE = 10.0
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVIEWER_BACKENDS = frozenset({"claude", "codex"})
_REVIEWER_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:\-]{0,127}")

_RECORD_FIELDS = frozenset(
    {
        "schema",
        "status",
        "reason",
        "required",
        "reviewer",
        "verdict",
        "prompt_sha256",
        "reviewed",
        "applied",
        "coverage",
        "files",
    }
)
_COVERAGE_FIELDS = frozenset(
    {"calls_planned", "calls_made", "rulespec", "test", "source"}
)
_COVERED_TEXT_FIELDS = frozenset({"sha256", "size", "chunks"})
_CHUNK_FIELDS = frozenset({"start", "end", "sha256"})
_REVIEWER_FIELDS = frozenset({"name", "backend", "model"})
_VERDICT_FIELDS = frozenset({"passed", "score", "issue_count"})
_DIGEST_FIELDS = (
    "rulespec_sha256",
    "rulespec_content_sha256",
    "test_sha256",
    "test_content_sha256",
)
_PAIR_FIELDS = frozenset(_DIGEST_FIELDS)
# Where the bound pair is applied: manifest paths, None when nothing is bound.
_PATH_FIELDS = ("rulespec_path", "test_path")
_APPLIED_FIELDS = _PAIR_FIELDS | frozenset(_PATH_FIELDS)
# The reviewed RuleSpec after the dump that apply performs on it.
_DUMPED_FIELD = "rulespec_dumped_content_sha256"
# The byte digest of the source excerpt the reviewer was given; None when
# the review had no source excerpt.
_SOURCE_FIELD = "source_sha256"
_REVIEWED_FIELDS = _PAIR_FIELDS | {_DUMPED_FIELD, _SOURCE_FIELD}
_MERGE_TAG = "tag:yaml.org,2002:merge"
_FILE_FIELDS = frozenset({"path", "sha256", "reviewed", "complete"})
# The two keys `encode --apply` stamps into module.source_verification.
_SOURCE_STAMP_KEYS = ("corpus_citation_path", "source_sha256")
_NULL_TAG = "tag:yaml.org,2002:null"
_NULL_LEXEMES = frozenset({"", "~", "null", "Null", "NULL"})
_BOOL_TAG = "tag:yaml.org,2002:bool"
_STR_TAG = "tag:yaml.org,2002:str"
_YAML11_TYPED_SCALARS = frozenset(
    {
        _BOOL_TAG,
        "tag:yaml.org,2002:int",
        "tag:yaml.org,2002:float",
        "tag:yaml.org,2002:timestamp",
    }
)
_MAP_TAG = "tag:yaml.org,2002:map"
_SEQ_TAG = "tag:yaml.org,2002:seq"


class _ReviewContentLoader(yaml.SafeLoader):
    """Retain whether a scalar's tag was implicit, without global changes."""

    def compose_scalar_node(self, anchor):
        event = self.peek_event()
        node = super().compose_scalar_node(anchor)
        pyyaml_tag = node.tag
        styled_nonspecific = event.tag == "!" and event.style is not None
        if styled_nonspecific:
            # PyYAML marks even quoted/block `!` scalars implicit and resolves
            # them as plain values. Their source type is explicitly string.
            node.tag = _STR_TAG
        node._review_implicit_plain = event.implicit[0] and not styled_nonspecific
        # The tag PyYAML itself resolved: what `yaml.safe_load` constructs.
        node._review_pyyaml_tag = pyyaml_tag
        return node


_YAML12_NULLS = frozenset({"", "~", "null", "Null", "NULL"})
_YAML12_BOOLS = {
    **dict.fromkeys(("true", "True", "TRUE"), True),
    **dict.fromkeys(("false", "False", "FALSE"), False),
}
_YAML12_INT = re.compile(r"([-+]?)(?:([0-9]+)|0o([0-7]+)|0x([0-9a-fA-F]+))")
_KEY_VALUE_CONSTRUCTOR = yaml.constructor.SafeConstructor()


def _yaml12_core_value(text: str) -> object:
    """The value a YAML 1.2 core reader gives a plain non-string scalar."""

    if text in _YAML12_NULLS:
        return None
    if text in _YAML12_BOOLS:
        return _YAML12_BOOLS[text]
    integer = _YAML12_INT.fullmatch(text)
    if integer is not None:
        sign, decimal, octal, hexadecimal = integer.groups()
        if decimal is not None:
            number = int(decimal)
        elif octal is not None:
            number = int(octal, 8)
        else:
            number = int(hexadecimal, 16)
        return -number if sign == "-" else number
    lowered = text.lower()
    if lowered.endswith(".nan"):
        return math.nan
    if lowered.endswith(".inf"):
        return -math.inf if text.startswith("-") else math.inf
    return float(text)


_VALUE_TAG = "tag:yaml.org,2002:value"
_FLOAT_TAG = "tag:yaml.org,2002:float"


# The spellings of infinity and NaN that every modeled reader types.
_EXPLICIT_NON_FINITE = re.compile(r"[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN)")


def _agreed_plain_float(text: str) -> float | None:
    """The float that every modeled reader gives a plain scalar, or None.

    ``yaml.safe_load`` (YAML 1.1) and a YAML 1.2 core reader must both read a
    float, and the same one (the reader that mixes their typed forms then
    reads it too). ``0.20``, ``0.2`` and ``.2`` are such a scalar; ``1_0.5``,
    ``1:30.5`` and ``1.5e3`` are not, because one dialect reads a string.
    A spelling that overflows to infinity is not either.
    """

    if not is_yaml12_core_non_string(text):
        # A string for a YAML 1.2 core reader, whatever `float()` accepts
        # (it reads `1_0.5`).
        return None
    try:
        yaml11 = _KEY_VALUE_CONSTRUCTOR.construct_yaml_float(
            yaml.ScalarNode(_FLOAT_TAG, text)
        )
        yaml12 = _yaml12_core_value(text)
    except Exception:  # noqa: BLE001 - any constructor failure
        return None
    if type(yaml11) is not float or type(yaml12) is not float:
        return None
    if not (math.isfinite(yaml11) and math.isfinite(yaml12)):
        # Infinity and NaN are one value only in their own spellings. A
        # number that overflows (`1.0e+999`) is infinity here, but a reader
        # that accepts only finite parses reads a string: its text stays
        # exact, so it never equals `.inf`.
        if _EXPLICIT_NON_FINITE.fullmatch(text) is None:
            return None
        if math.isnan(yaml11) and math.isnan(yaml12):
            return yaml11
    if yaml11 != yaml12 or math.copysign(1.0, yaml11) != math.copysign(1.0, yaml12):
        return None
    return yaml11


_YAML12_CORE_TYPED_TAGS = frozenset(
    {_NULL_TAG, _BOOL_TAG, "tag:yaml.org,2002:int", "tag:yaml.org,2002:float"}
)


def _scalar_key_identities(node: yaml.ScalarNode) -> set[tuple[object, ...]]:
    """The values that readers give one scalar mapping key, as set members.

    Three identities: the value of ``yaml.safe_load`` (PyYAML, YAML 1.1), the
    value of a YAML 1.2 core reader, and each typed (non-string) value of
    either under one shared name. Two keys with an equal identity are one
    key for some reader: its mapping keeps one of the two values, and which
    one depends on the order. The shared name covers a reader that mixes the
    two dialects (``0o20`` is 16 only in YAML 1.2, ``0b10000`` only in YAML
    1.1; serde_yaml and ruamel read both as 16). Equality is Python's, as in
    the mapping that a reader builds, so ``1``, ``1.0`` and ``true`` are
    equal; every NaN equals every NaN.

    Fails closed: a key that either dialect cannot construct raises
    ``ValueError``, so no key is ever without an identity. That includes a
    tag without a PyYAML constructor, and an explicit tag outside the YAML
    1.2 core schema (``!!value``, ``!!timestamp``, ``!!binary``, a local tag).
    """

    def identity(reader: str, value: object) -> tuple[object, ...]:
        if isinstance(value, float) and math.isnan(value):
            return (reader, "nan")
        hash(value)
        return (reader, "value", value)

    text = node.value
    implicit_plain = getattr(node, "_review_implicit_plain", False)
    pyyaml_tag = getattr(node, "_review_pyyaml_tag", node.tag)
    # PyYAML's mapping constructor turns a `=` value key into a string key
    # (`flatten_mapping`); the tag itself has no constructor.
    yaml11_tag = _STR_TAG if pyyaml_tag == _VALUE_TAG else pyyaml_tag
    construct = yaml.constructor.SafeConstructor.yaml_constructors.get(yaml11_tag)
    if construct is None:
        raise ValueError("unreadable YAML key")
    try:
        yaml11 = construct(_KEY_VALUE_CONSTRUCTOR, yaml.ScalarNode(yaml11_tag, text))
    except Exception as exc:  # noqa: BLE001 - any constructor failure
        # PyYAML's scalar constructors fail in several ways on a value
        # that does not fit an explicit tag (KeyError, AttributeError).
        raise ValueError("unreadable YAML key") from exc

    yaml12: object
    if implicit_plain:
        # A plain scalar: a core type, or else a string (also for a YAML
        # 1.1-only implicit type such as `yes`, `1:30`, a date or `=`).
        yaml12 = _yaml12_core_value(text) if is_yaml12_core_non_string(text) else text
    elif node.tag == _STR_TAG:
        yaml12 = text
    elif node.tag in _YAML12_CORE_TYPED_TAGS and (
        is_yaml12_core_non_string(text) or (node.tag == _NULL_TAG and text == "")
    ):
        yaml12 = _yaml12_core_value(text)
    else:
        raise ValueError("unreadable YAML key")

    try:
        found = {identity("yaml 1.1", yaml11), identity("yaml 1.2", yaml12)}
        for value in (yaml11, yaml12):
            if not isinstance(value, str):
                found.add(identity("mixed", value))
    except (ArithmeticError, TypeError) as exc:
        raise ValueError("unreadable YAML key") from exc
    return found


def _is_empty(node: yaml.Node | None) -> bool:
    if node is None:
        return True
    if isinstance(node, yaml.ScalarNode):
        return node.tag == _NULL_TAG and node.value in _NULL_LEXEMES
    return (
        isinstance(node, yaml.MappingNode)
        and node.tag == _MAP_TAG
        or isinstance(node, yaml.SequenceNode)
        and node.tag == _SEQ_TAG
    ) and not node.value


def _mapping_item(node: yaml.Node | None, key: str) -> yaml.Node | None:
    if not isinstance(node, yaml.MappingNode) or node.tag != _MAP_TAG:
        return None
    for key_node, value_node in node.value:
        if (
            isinstance(key_node, yaml.ScalarNode)
            and key_node.tag == _STR_TAG
            and key_node.value == key
        ):
            return value_node
    return None


def _canonical_node(
    node: yaml.Node | None,
    *,
    drop: Mapping[int, frozenset[str]],
    seen: set[int],
) -> object:
    """Return a JSON value that differs whenever the YAML node differs.

    A scalar keeps its resolved type tag and its exact text, so ``'1'`` and
    ``1`` differ, and so do ``yes`` and ``true``. ``drop`` names, per mapping
    node, the keys to leave out. A collection reached twice (an alias), a
    merge key and a repeated key raise ``ValueError``: a reader may resolve
    them in more than one way, so the document has no digest. Two scalar keys
    are also a repeated key when PyYAML (YAML 1.1), a YAML 1.2 core reader,
    or a reader that mixes their typed forms gives them equal values,
    whatever their text (``0x10`` and ``16``, ``0o20`` and ``0b10000``). A
    key that is not a scalar, or that one of the two dialects cannot
    construct, also raises.
    """

    if node is None:
        return None
    if isinstance(node, yaml.ScalarNode):
        value = node.value
        tag = node.tag
        implicit_yaml12_non_string = False
        implicit_plain = getattr(node, "_review_implicit_plain", False)
        if (
            implicit_plain
            and tag in _YAML11_TYPED_SCALARS
            and not is_yaml12_core_non_string(value)
        ):
            # YAML 1.1-only implicit types (yes/ON, sexagesimal numbers,
            # dates, etc.) are strings for a YAML 1.2 core consumer. Preserve
            # their case and distinguish them from explicit type tags.
            tag = _STR_TAG
        if tag == _STR_TAG and implicit_plain and is_yaml12_core_non_string(value):
            # PyYAML 1.1 calls plain `1e5` a string, but a YAML 1.2 consumer
            # reads a number. Quoting it must change the binding. Explicit
            # !!str scalars remain strings; harmless quoting stays neutral.
            # Keep this distinction outside the tag namespace: an explicit
            # user tag must not be able to impersonate an internal marker.
            implicit_yaml12_non_string = True
        if tag == _NULL_TAG and value in _NULL_LEXEMES:
            value = ""  # `key:`, `~` and `null` are one value, dumped as `null`
        elif tag == _BOOL_TAG and value in {
            "true",
            "True",
            "TRUE",
            "false",
            "False",
            "FALSE",
        }:
            value = value.lower()
        pyyaml_tag = getattr(node, "_review_pyyaml_tag", node.tag)
        if (
            implicit_plain
            and tag == _FLOAT_TAG
            and pyyaml_tag == _FLOAT_TAG
            and is_yaml12_core_non_string(value)
        ):
            # A plain float that every modeled reader reads as one value has
            # that value as its content, not its spelling: the apply dump
            # writes `0.20` again as `0.2`. `repr` keeps `-0.0` apart from
            # `0.0`, and all spellings of infinity and of NaN are one value
            # each. A quoted or tagged scalar, an integer and a scalar that
            # one dialect reads another way keep their exact text.
            number = _agreed_plain_float(value)
            if number is not None:
                value = repr(number)
        # The type `yaml.safe_load` gives the scalar is part of its content:
        # that loader runs the companion tests. Plain `yes`, `0b10`, `1_000`,
        # `1:30` and `2001-01-01` are a boolean, numbers and a date there, and
        # strings when quoted; the tag above is the same string tag for both
        # spellings. With the loader's tag in the content, two files with an
        # equal digest load as equal values (the text is exact, apart from
        # the null and boolean spellings that load as one value). The same
        # holds for a quoted `!` scalar, which PyYAML resolves like a plain
        # one (`! "yes"` loads as True) and the tag above calls a string.
        return ["scalar", tag, value, implicit_yaml12_non_string, pyyaml_tag]
    if id(node) in seen:
        raise ValueError("YAML alias")
    seen.add(id(node))
    if isinstance(node, yaml.SequenceNode):
        return [
            "sequence",
            node.tag,
            [_canonical_node(item, drop=drop, seen=seen) for item in node.value],
        ]
    if isinstance(node, yaml.MappingNode):
        dropped = drop.get(id(node), frozenset())
        items = []
        keys = set()
        key_values: set[tuple[object, ...]] = set()
        for key_node, value_node in node.value:
            if key_node.tag == _MERGE_TAG:
                raise ValueError("YAML merge key")
            key = json.dumps(
                _canonical_node(key_node, drop=drop, seen=seen), sort_keys=True
            )
            if key in keys:
                raise ValueError("repeated YAML key")
            keys.add(key)
            if not isinstance(key_node, yaml.ScalarNode):
                # `yaml.safe_load` cannot build a mapping with a sequence or
                # a mapping as a key, so no reader's value is known for it.
                raise ValueError("unreadable YAML key")
            # Keys with different text can still be one key for a reader
            # (`0x10` and `16`): the loaded value then depends on the key
            # order, which this digest ignores. Refuse them like a repeated
            # key. Every scalar key has at least one identity.
            identities = _scalar_key_identities(key_node)
            if not identities or identities & key_values:
                raise ValueError("repeated YAML key")
            key_values |= identities
            # A left-out value is still walked, so an alias of it is refused.
            item = _canonical_node(value_node, drop=drop, seen=seen)
            if (
                isinstance(key_node, yaml.ScalarNode)
                and key_node.tag == _STR_TAG
                and key_node.value in dropped
            ):
                continue
            items.append([key, item])
        return ["mapping", node.tag, sorted(items)]
    raise TypeError(f"unsupported YAML node {type(node).__name__}")


# The content of a companion test that is not there. No node tree has this
# canonical form: a tree is null or a list that starts with a node kind.
_ABSENT_TEST_CONTENT_SHA256 = hashlib.sha256(b'["absent"]').hexdigest()


def review_content_sha256(raw: bytes | None, *, rulespec: bool) -> str | None:
    """Digest the content of one reviewed or applied YAML file.

    ``raw`` None is an absent file. A file that cannot be decoded or parsed
    has no content digest (None), and a verdict without one is never bound.

    A companion test (``rulespec`` False) keeps the differences that its
    loader acts on. An absent file has a digest of its own: nothing loads it.
    An empty list (``[]``, zero test cases) differs from a null document (an
    empty file, only comments, ``~``), which the loader refuses, and from an
    empty mapping. A RuleSpec module without content (absent, null, ``{}``,
    ``[]``) has one digest: the apply stamp can add an empty ``module``.
    """

    if raw is None and not rulespec:
        return _ABSENT_TEST_CONTENT_SHA256
    if raw is None:
        root: yaml.Node | None = None
    else:
        try:
            root = yaml.compose(raw.decode("utf-8"), Loader=_ReviewContentLoader)
        except (UnicodeError, yaml.YAMLError, RecursionError, ValueError):
            return None
    drop: dict[int, frozenset[str]] = {}
    if rulespec and isinstance(root, yaml.MappingNode):
        module = _mapping_item(root, "module")
        verification = _mapping_item(module, "source_verification")
        if isinstance(verification, yaml.MappingNode) and verification.tag == _MAP_TAG:
            drop[id(verification)] = frozenset(_SOURCE_STAMP_KEYS)
            if all(
                isinstance(key_node, yaml.ScalarNode)
                and key_node.tag == _STR_TAG
                and key_node.value in _SOURCE_STAMP_KEYS
                for key_node, _value in verification.value
            ):
                # Nothing but the stamp: the mapping itself came from apply.
                drop[id(module)] = frozenset({"source_verification"})
                if len(module.value) == 1:
                    drop[id(root)] = frozenset({"module"})
        elif module is not None and _is_empty(module):
            drop[id(root)] = frozenset({"module"})
    try:
        canonical_value = _canonical_node(root, drop=drop, seen=set())
        if rulespec:
            if _is_empty(root) or canonical_value == ["mapping", _MAP_TAG, []]:
                canonical_value = None
        elif isinstance(root, yaml.ScalarNode) and _is_empty(root):
            # `~` and `null` load like an empty file: one null document.
            canonical_value = None
        canonical = json.dumps(
            canonical_value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, RecursionError):
        return None
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def apply_stamp_dump(payload: object) -> str:
    """Dump a RuleSpec document the way ``encode --apply`` rewrites it.

    The apply stamp calls this, and the review binding calls it on the
    reviewed bytes, so both use one dump.
    """

    return safe_dump_yaml(payload, sort_keys=False, allow_unicode=True)


def dumped_review_content_sha256(raw: bytes) -> str | None:
    """Digest the apply dump only when the reviewed digest stays equal.

    Apply loads the candidate with ``yaml.safe_load`` and dumps it with
    :func:`apply_stamp_dump`; it rewrites nothing that is not a mapping.
    A changed scalar spelling or collapsed key cannot supply an alternate
    binding, even if a particular change might preserve a numeric value.
    """

    try:
        payload = yaml.safe_load(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            return None
        dumped = apply_stamp_dump(payload).encode("utf-8")
    except (UnicodeError, yaml.YAMLError, RecursionError, ValueError, TypeError):
        return None
    original_digest = review_content_sha256(raw, rulespec=True)
    dumped_digest = review_content_sha256(dumped, rulespec=True)
    return dumped_digest if dumped_digest == original_digest else None


# The companion test that apply writes for a deferred module without one.
EMPTY_DEFERRED_COMPANION_TEST = b"[]\n"


def deferred_module_needs_empty_companion_test(payload: object) -> bool:
    """Return whether a loaded RuleSpec document is a deferred module that
    gets an empty companion test at apply.

    One rule for the apply planner and for the reviewer call, so the reviewer
    is shown the companion exactly when apply writes it.
    """

    if not isinstance(payload, dict):
        return False
    module = payload.get("module")
    if not isinstance(module, dict):
        return False
    if module.get("status") != "deferred":
        return False
    if payload.get("rules") != []:
        return False
    deferred_outputs = module.get("deferred_outputs")
    return isinstance(deferred_outputs, list) and bool(deferred_outputs)


def empty_deferred_companion_test(rulespec: bytes) -> bytes | None:
    """Return the companion bytes apply would write for ``rulespec`` alone.

    None when apply writes no companion for it, or when it cannot be read.
    """

    try:
        payload = yaml.safe_load(rulespec.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError, RecursionError, ValueError):
        return None
    if deferred_module_needs_empty_companion_test(payload):
        return EMPTY_DEFERRED_COMPANION_TEST
    return None


def is_manifest_file_path(value: object) -> bool:
    """Return whether ``value`` is one canonical manifest-relative file path.

    One spelling per file: POSIX separators, relative, no empty, ``.`` or
    ``..`` segment, no backslash and no control character. A path with
    another spelling is not compared with anything; it is refused.
    """

    if not isinstance(value, str) or not value or len(value) > 4096:
        return False
    if "\\" in value or any(
        ord(char) < 32 or ord(char) == 127 or char in "\u2028\u2029" for char in value
    ):
        return False
    return all(part not in {"", ".", ".."} for part in value.split("/"))


def _path_collision_key(path: str) -> str:
    """One key for paths that a file system can resolve to one file.

    Case and Unicode composition do not separate two files on every file
    system, so two listed paths with one key are refused, not compared.
    """

    return unicodedata.normalize("NFC", unicodedata.normalize("NFD", path).casefold())


def is_rulespec_module_path(value: object) -> bool:
    """Return whether ``value`` is a manifest path of a module, not a test."""

    return (
        is_manifest_file_path(value)
        and str(value).endswith(RULESPEC_FILE_SUFFIX)
        and not str(value).endswith(RULESPEC_TEST_FILE_SUFFIX)
        and PurePosixPath(str(value)).name != RULESPEC_FILE_SUFFIX
    )


def companion_test_manifest_path(rulespec_path: str) -> str:
    """The manifest path of the companion test of one module path."""

    path = PurePosixPath(rulespec_path)
    return path.with_name(f"{path.stem}{RULESPEC_TEST_FILE_SUFFIX}").as_posix()


@dataclass(frozen=True)
class AppliedPair:
    """One applied RuleSpec file and its companion test, by path and digest.

    ``rulespec_path`` is the manifest path of the module. The companion test
    is at :func:`companion_test_manifest_path` of it and nowhere else. A pair
    without a path is never bound to a verdict.
    """

    rulespec_sha256: str
    rulespec_content_sha256: str | None
    test_sha256: str | None
    test_content_sha256: str | None
    rulespec_path: str | None = None

    @property
    def test_path(self) -> str | None:
        """The companion's manifest path; None without a companion or path."""

        if self.test_sha256 is None or not is_rulespec_module_path(self.rulespec_path):
            return None
        return companion_test_manifest_path(str(self.rulespec_path))


def applied_pair(
    rulespec: bytes, test: bytes | None, *, rulespec_path: str | None = None
) -> AppliedPair:
    """Digest one RuleSpec file and its companion test (None when absent).

    ``rulespec_path`` is the manifest path the module is applied at.
    """

    return AppliedPair(
        rulespec_sha256=hashlib.sha256(rulespec).hexdigest(),
        rulespec_content_sha256=review_content_sha256(rulespec, rulespec=True),
        test_sha256=None if test is None else hashlib.sha256(test).hexdigest(),
        test_content_sha256=review_content_sha256(test, rulespec=False),
        rulespec_path=rulespec_path,
    )


def review_evidence_digests(rulespec: bytes, test: bytes | None) -> dict[str, object]:
    """The digests a reviewer call records for the files it read."""

    pair = applied_pair(rulespec, test)
    return {
        "reviewed_rulespec_sha256": pair.rulespec_sha256,
        "reviewed_rulespec_content_sha256": pair.rulespec_content_sha256,
        "reviewed_test_sha256": pair.test_sha256,
        "reviewed_test_content_sha256": pair.test_content_sha256,
        "reviewed_" + _DUMPED_FIELD: dumped_review_content_sha256(rulespec),
    }


@dataclass(frozen=True)
class GeneralistReviewState:
    """What one encode attempt knows about its generalist review.

    ``metrics_pass``, ``score``, ``issue_count`` and ``prompt_sha256`` are the
    values the eval pipeline already produces. ``evidence`` is what the
    reviewer call itself measured (digests of the files it read, whether its
    prompt showed each one whole, the boolean ``passed`` that the reviewer
    wrote itself, and the CLI it started); it is absent when no scored review
    happened. ``metrics_pass`` may come from the eval pipeline's score
    fallback, so a pass also needs the explicit boolean in ``evidence``.
    """

    metrics_pass: object = None
    score: object = None
    issue_count: object = 0
    prompt_sha256: object = None
    evidence: object = None
    skipped_by_flag: bool = False
    required: bool = False


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _finite_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0 or number > _MAX_SCORE:
        return None
    return number


def _issue_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _reviewer_identity(evidence: Mapping[str, Any]) -> dict[str, object]:
    backend = evidence.get("reviewer_backend")
    model = evidence.get("reviewer_model")
    if backend not in _REVIEWER_BACKENDS:
        backend = None
    if not isinstance(model, str) or _REVIEWER_MODEL.fullmatch(model) is None:
        model = None
    return {
        "name": GENERALIST_REVIEWER_NAME,
        "backend": backend,
        "model": model if backend is not None else None,
    }


def _applied_file_list(
    applied_files: Iterable[object],
) -> tuple[dict[str, str], bool]:
    """Return the applied files as path -> byte digest, and whether the list
    is sound.

    The list is not sound when an item is not a (path, digest) pair of a
    canonical manifest path and a SHA-256, when one path has two digests, or
    when two paths can be one file (case, Unicode composition). An unsound
    list binds no verdict. Malformed items are left out of the mapping.
    """

    listed: dict[str, str] = {}
    sound = True
    for item in applied_files:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not is_manifest_file_path(item[0])
            or not _is_sha256(item[1])
        ):
            sound = False
            continue
        path, sha256 = item
        if listed.setdefault(path, sha256) != sha256:
            sound = False
    if len({_path_collision_key(path) for path in listed}) != len(listed):
        sound = False
    return listed, sound


def _file_entries(
    listed: Mapping[str, str], reviewed: Mapping[str, bool]
) -> list[dict[str, object]]:
    """One entry per applied path: reviewed at all, and reviewed in full.

    ``reviewed`` is keyed by path. A file is reviewed because the reviewer
    read the file at that path, never because another path holds equal bytes.
    """

    return [
        {
            "path": path,
            "sha256": listed[path],
            "reviewed": path in reviewed,
            "complete": reviewed.get(path, False),
        }
        for path in sorted(listed)
    ]


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _covered_text(value: object) -> dict[str, object] | None:
    """One text of a coverage block, copied when it is well formed.

    Well formed is the shape only: a digest, a size and a list of chunks,
    each a byte range inside the size with a digest. Whether the chunks
    cover the text is :func:`_covers_whole_text`.
    """

    if (
        not isinstance(value, Mapping)
        or frozenset(value) != _COVERED_TEXT_FIELDS
        or not _is_sha256(value["sha256"])
        or not _is_count(value["size"])
        or not isinstance(value["chunks"], list)
    ):
        return None
    chunks: list[dict[str, object]] = []
    for chunk in value["chunks"]:
        if (
            not isinstance(chunk, Mapping)
            or frozenset(chunk) != _CHUNK_FIELDS
            or not _is_count(chunk["start"])
            or not _is_count(chunk["end"])
            or not chunk["start"] <= chunk["end"] <= value["size"]
            or not _is_sha256(chunk["sha256"])
        ):
            return None
        chunks.append(
            {"start": chunk["start"], "end": chunk["end"], "sha256": chunk["sha256"]}
        )
    return {"sha256": value["sha256"], "size": value["size"], "chunks": chunks}


def review_coverage(value: object) -> dict[str, object] | None:
    """A coverage block copied from review evidence, or None when malformed.

    The copy holds integers and digests only. The module must be present;
    the companion test and the source excerpt are None when the review had
    none.
    """

    if not isinstance(value, Mapping) or frozenset(value) != _COVERAGE_FIELDS:
        return None
    rulespec = _covered_text(value["rulespec"])
    test = None if value["test"] is None else _covered_text(value["test"])
    source = None if value["source"] is None else _covered_text(value["source"])
    if (
        rulespec is None
        or (value["test"] is not None and test is None)
        or (value["source"] is not None and source is None)
        or not _is_count(value["calls_planned"])
        or not _is_count(value["calls_made"])
    ):
        return None
    return {
        "calls_planned": value["calls_planned"],
        "calls_made": value["calls_made"],
        "rulespec": rulespec,
        "test": test,
        "source": source,
    }


def _every_call_made(coverage: Mapping[str, Any]) -> bool:
    """Every planned call was made, and the plan is within the call cap.

    The number of planned calls is the number that the chunk counts of the
    three texts give; a record cannot state another one.
    """

    planned = planned_call_count(
        len(coverage["rulespec"]["chunks"]),
        0 if coverage["source"] is None else len(coverage["source"]["chunks"]),
        0 if coverage["test"] is None else len(coverage["test"]["chunks"]),
    )
    return (
        1
        <= coverage["calls_made"]
        == coverage["calls_planned"]
        == planned
        <= MAX_REVIEW_CALLS
    )


def _covers_whole_text(text: Mapping[str, Any]) -> bool:
    """The chunks cover every byte of the text exactly once, in order.

    The first chunk starts at byte 0, each next chunk starts where the one
    before it ends, and the last one ends at the size: no gap, no overlap
    and no other order. A chunk is empty only as the one chunk of an empty
    text. One chunk that is the whole text has the digest of the text.
    """

    chunks = text["chunks"]
    if not 1 <= len(chunks) <= MAX_CHUNKS_PER_FILE:
        return False
    position = 0
    for chunk in chunks:
        if chunk["start"] != position:
            return False
        if chunk["end"] == chunk["start"] and text["size"] != 0:
            return False
        position = chunk["end"]
    if position != text["size"]:
        return False
    return len(chunks) > 1 or chunks[0]["sha256"] == text["sha256"]


def _text_seen_whole(
    coverage: Mapping[str, Any] | None, name: str, sha256: object
) -> bool:
    """Coverage shows the whole text ``name`` with the byte digest ``sha256``."""

    if coverage is None or not _every_call_made(coverage):
        return False
    text = coverage[name]
    return text is not None and text["sha256"] == sha256 and _covers_whole_text(text)


def _source_seen_whole(
    coverage: Mapping[str, Any] | None, source_sha256: object
) -> bool:
    """Coverage shows the whole source excerpt with digest ``source_sha256``.

    With no digest the review had no source excerpt, and coverage must name
    none. With a digest, coverage must show every byte of the text that has
    that digest.
    """

    if coverage is None:
        return False
    if source_sha256 is None:
        return coverage["source"] is None
    return _text_seen_whole(coverage, "source", source_sha256)


def _unscored_record(
    status: str,
    reason: str,
    *,
    required: bool,
    files: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "schema": GENERALIST_REVIEW_SCHEMA,
        "status": status,
        "reason": reason,
        "required": required,
        "reviewer": {
            "name": GENERALIST_REVIEWER_NAME,
            "backend": None,
            "model": None,
        },
        "verdict": {"passed": None, "score": None, "issue_count": 0},
        "prompt_sha256": None,
        "reviewed": dict.fromkeys((*_DIGEST_FIELDS, _DUMPED_FIELD, _SOURCE_FIELD)),
        "applied": dict.fromkeys((*_DIGEST_FIELDS, *_PATH_FIELDS)),
        "coverage": None,
        "files": files,
    }


def build_generalist_review_record(
    state: GeneralistReviewState,
    applied: Iterable[AppliedPair],
    applied_files: Iterable[object] | None = None,
    *,
    reviewed_rulespec_path: object = None,
) -> dict[str, object]:
    """Return the record for ``state`` against the files that are applied.

    ``applied`` are the RuleSpec files the encode applies (or is about to
    apply), each with its manifest path and its companion test.
    ``applied_files`` are the (manifest path, byte digest) pairs of every
    applied file, whether or not it is in a pair; left out, they are the
    files of the pairs. ``reviewed_rulespec_path`` is the manifest path of
    the one module the reviewer was asked about.

    A scored verdict counts only for the pair at that path, and only when
    that pair has the content the reviewer read and is in ``applied_files``
    with its digests. It marks the module path and its companion test path
    reviewed, and no other path: a file with equal bytes at another path is
    validated in another module's context, which the reviewer did not read.
    The record is ``passed`` only when those two paths are everything that
    is applied, the reviewer saw both files whole and it saw the whole
    source excerpt. "Whole" is read from the coverage block of the evidence:
    chunk ranges that cover the reviewed bytes exactly once, with every
    planned call made.
    """

    pairs = [pair for pair in applied if isinstance(pair, AppliedPair)]
    if applied_files is None:
        applied_files = [
            (path, digest)
            for pair in pairs
            for path, digest in (
                (pair.rulespec_path, pair.rulespec_sha256),
                (pair.test_path, pair.test_sha256),
            )
            if digest is not None
        ]
    listed, sound = _applied_file_list(applied_files)
    unreviewed = _file_entries(listed, {})
    required = state.required is True
    if state.skipped_by_flag is True:
        return _unscored_record(
            STATUS_SKIPPED, REASON_SKIP_FLAG, required=required, files=unreviewed
        )

    verdict = state.metrics_pass
    raw_score = state.score
    if verdict is True and raw_score is None and state.prompt_sha256 is None:
        # The eval pipeline's placeholder for a reviewer it did not call.
        return _unscored_record(
            STATUS_SKIPPED, REASON_NOT_RUN, required=required, files=unreviewed
        )
    if verdict is False and raw_score is None:
        return _unscored_record(
            STATUS_UNAVAILABLE,
            REASON_REVIEWER_ERROR,
            required=required,
            files=unreviewed,
        )

    score = _finite_score(raw_score)
    evidence = state.evidence
    if (
        not isinstance(verdict, bool)
        or score is None
        or not _is_sha256(state.prompt_sha256)
        or not isinstance(evidence, Mapping)
        or not _is_sha256(evidence.get("reviewed_rulespec_sha256"))
        # The file as reviewed must have a digest: with a repeated key, an
        # alias or a merge key the reviewer read an ambiguous file, and the
        # dump would silently pick one reading of it.
        or not _is_sha256(evidence.get("reviewed_rulespec_content_sha256"))
        or not (
            evidence.get("reviewed_test_sha256") is None
            or _is_sha256(evidence.get("reviewed_test_sha256"))
        )
        or not _is_sha256(evidence.get("reviewed_test_content_sha256"))
        # No companion file was read exactly when the reviewed companion
        # content is that of an absent file.
        or (evidence.get("reviewed_test_sha256") is None)
        != (evidence.get("reviewed_test_content_sha256") == _ABSENT_TEST_CONTENT_SHA256)
        or (verdict is True and score < PASS_SCORE_THRESHOLD)
        # A passing verdict counts only as the reviewer's own boolean
        # `passed: true`. The eval pipeline can infer a pass from the score
        # or from the absence of blocking issues; that never reaches here.
        or (verdict is True and evidence.get(EVIDENCE_EXPLICIT_PASSED) is not True)
    ):
        return _unscored_record(
            STATUS_UNAVAILABLE,
            REASON_UNREADABLE,
            required=required,
            files=unreviewed,
        )

    reviewed = {
        field: evidence.get(f"reviewed_{field}")
        for field in (*_DIGEST_FIELDS, _DUMPED_FIELD)
    }
    if not _is_sha256(reviewed[_DUMPED_FIELD]):
        reviewed[_DUMPED_FIELD] = None
    source_sha256 = evidence.get("reviewed_" + _SOURCE_FIELD)
    reviewed[_SOURCE_FIELD] = source_sha256 if _is_sha256(source_sha256) else None
    # A dump-derived hash is diagnostic only. It cannot authorize content
    # different from the reviewed node tree, even if the loader collapsed it.
    if reviewed[_DUMPED_FIELD] != reviewed["rulespec_content_sha256"]:
        reviewed[_DUMPED_FIELD] = None
    # Only the pair at the reviewed module path can carry the verdict. Two
    # pairs at one path, a path with another spelling, or an applied file
    # list that is not sound, bind nothing.
    at_reviewed_path = [
        pair
        for pair in pairs
        if is_rulespec_module_path(reviewed_rulespec_path)
        and pair.rulespec_path == reviewed_rulespec_path
    ]
    bound_pair = next(
        (
            pair
            for pair in at_reviewed_path
            if sound
            and len(at_reviewed_path) == 1
            and _is_sha256(pair.rulespec_sha256)
            and pair.rulespec_content_sha256 == reviewed["rulespec_content_sha256"]
            and pair.test_content_sha256 == reviewed["test_content_sha256"]
            and (pair.test_sha256 is None or _is_sha256(pair.test_sha256))
            and (pair.test_sha256 is None)
            == (pair.test_content_sha256 == _ABSENT_TEST_CONTENT_SHA256)
            # The pair is applied: each of its files is listed at its own
            # path with its own digest.
            and listed.get(str(pair.rulespec_path)) == pair.rulespec_sha256
            and (
                pair.test_sha256 is None
                or listed.get(str(pair.test_path)) == pair.test_sha256
            )
        ),
        None,
    )
    coverage: dict[str, object] | None = None
    if bound_pair is None:
        status = STATUS_UNAVAILABLE
        reason: str | None = REASON_CONTENT_NOT_APPLIED
        applied_digests: dict[str, object] = dict.fromkeys(
            (*_DIGEST_FIELDS, *_PATH_FIELDS)
        )
        files = unreviewed
    else:
        coverage = review_coverage(evidence.get("coverage"))
        applied_digests = {
            field: getattr(bound_pair, field)
            for field in (*_DIGEST_FIELDS, *_PATH_FIELDS)
        }
        # Only an explicit True says the prompts showed the whole file, and
        # only chunk ranges that cover the reviewed bytes once confirm it.
        seen_whole = {
            str(bound_pair.rulespec_path): evidence.get("rulespec_complete") is True
            and _text_seen_whole(coverage, "rulespec", reviewed["rulespec_sha256"])
        }
        # A reviewer that read no companion test has reviewed no applied
        # companion file. The content digests above already keep such a
        # pair apart (an absent companion is a content of its own); this is
        # a second guard that leaves the file unreviewed.
        if bound_pair.test_sha256 is not None and reviewed["test_sha256"] is not None:
            seen_whole[str(bound_pair.test_path)] = evidence.get(
                "test_complete"
            ) is True and _text_seen_whole(coverage, "test", reviewed["test_sha256"])
        files = _file_entries(listed, seen_whole)
        if not verdict:
            status, reason = STATUS_FAILED, None
        elif not files or not all(entry["reviewed"] for entry in files):
            status, reason = STATUS_PARTIAL, REASON_UNREVIEWED_FILES
        elif not all(entry["complete"] for entry in files):
            status, reason = STATUS_PARTIAL, REASON_TRUNCATED
        elif not (
            _source_seen_whole(coverage, reviewed[_SOURCE_FIELD])
            and (coverage["source"] is None or evidence.get("source_complete") is True)
        ):
            status, reason = STATUS_PARTIAL, REASON_SOURCE_TRUNCATED
        else:
            status, reason = STATUS_PASSED, None
    return {
        "schema": GENERALIST_REVIEW_SCHEMA,
        "status": status,
        "reason": reason,
        "required": required,
        "reviewer": _reviewer_identity(evidence),
        "verdict": {
            "passed": verdict,
            "score": score,
            "issue_count": _issue_count(state.issue_count),
        },
        "prompt_sha256": state.prompt_sha256,
        "reviewed": reviewed,
        "applied": applied_digests,
        "coverage": coverage,
        "files": files,
    }


def generalist_review_record_issues(
    record: object,
    applied_files: Iterable[object],
) -> list[ValidationIssue]:
    """Return every reason ``record`` is not a well-formed, bound record.

    This is the reader's check, from the manifest alone. The record must list
    exactly ``applied_files``: the manifest's own (path, byte digest) pairs.
    A record that says ``passed`` must mark every one of them reviewed in
    full. A scored record must carry the matching verdict and equal reviewed
    and applied content digests, and may mark as reviewed only the module
    path it names and the companion test path of that module, each with the
    digest it names. Any other record must mark no file reviewed.

    A file is marked complete only with a coverage block whose chunk ranges
    cover the reviewed bytes of that file exactly once within the caps, and a
    ``passed`` record also covers the whole source excerpt when it names one.
    The reader cannot recompute a chunk digest: it has no file bytes.
    """

    def shape(message: str) -> ValidationIssue:
        return ValidationIssue(message, check=CHECK_REVIEW_RECORD_SHAPE)

    def verdict_issue(message: str) -> ValidationIssue:
        return ValidationIssue(message, check=CHECK_REVIEW_RECORD_VERDICT)

    def binding(message: str) -> ValidationIssue:
        return ValidationIssue(message, check=CHECK_REVIEW_RECORD_BINDING)

    if not isinstance(record, dict):
        return [shape("generalist_review is not an object")]
    if frozenset(record) != _RECORD_FIELDS:
        return [shape("generalist_review has an unexpected field set")]
    found: list[ValidationIssue] = []
    if record["schema"] != GENERALIST_REVIEW_SCHEMA:
        found += [shape("generalist_review schema is unsupported")]
    status = record["status"]
    reason = record["reason"]
    if not isinstance(status, str) or status not in STATUSES:
        return found + [shape("generalist_review status is unsupported")]
    if not (reason is None or isinstance(reason, str)) or (
        reason not in _REASONS_BY_STATUS[status]
    ):
        found += [shape("generalist_review reason does not match its status")]
    if not isinstance(record["required"], bool):
        found += [shape("generalist_review required is not a boolean")]
    elif record["required"] and status != STATUS_PASSED:
        found += [
            verdict_issue(
                "generalist_review was required but the recorded status is " + status
            )
        ]

    reviewer = record["reviewer"]
    verdict = record["verdict"]
    reviewed = record["reviewed"]
    applied_block = record["applied"]
    files = record["files"]
    coverage = record["coverage"]
    if coverage is not None and review_coverage(coverage) != coverage:
        return found + [shape("generalist_review has a malformed coverage block")]
    if (
        not isinstance(reviewer, dict)
        or frozenset(reviewer) != _REVIEWER_FIELDS
        or not isinstance(verdict, dict)
        or frozenset(verdict) != _VERDICT_FIELDS
        or not isinstance(reviewed, dict)
        or frozenset(reviewed) != _REVIEWED_FIELDS
        or not isinstance(applied_block, dict)
        or frozenset(applied_block) != _APPLIED_FIELDS
        or not isinstance(files, list)
        or not all(
            isinstance(entry, dict)
            and frozenset(entry) == _FILE_FIELDS
            and is_manifest_file_path(entry["path"])
            and _is_sha256(entry["sha256"])
            and isinstance(entry["reviewed"], bool)
            and isinstance(entry["complete"], bool)
            for entry in files
        )
    ):
        return found + [shape("generalist_review has a malformed nested object")]

    listed = [entry["path"] for entry in files]
    if listed != sorted(set(listed)) or len(
        {_path_collision_key(path) for path in listed}
    ) != len(listed):
        found += [shape("generalist_review files are not sorted and unique")]
    if any(entry["complete"] and not entry["reviewed"] for entry in files):
        found += [shape("generalist_review marks an unreviewed file complete")]
    # Each applied path with its own digest: equal bytes at two paths are two
    # files, and a manifest list that is not sound matches no record.
    manifest_files, manifest_sound = _applied_file_list(applied_files)
    record_files = {(entry["path"], entry["sha256"]) for entry in files}
    if not manifest_sound or record_files != set(manifest_files.items()):
        found += [
            binding("generalist_review does not list exactly the applied file digests")
        ]

    backend = reviewer["backend"]
    model = reviewer["model"]
    if (
        reviewer["name"] != GENERALIST_REVIEWER_NAME
        or not (
            backend is None
            or (isinstance(backend, str) and backend in _REVIEWER_BACKENDS)
        )
        or not (
            model is None
            or (
                backend is not None
                and isinstance(model, str)
                and _REVIEWER_MODEL.fullmatch(model) is not None
            )
        )
    ):
        found += [shape("generalist_review reviewer identity is malformed")]

    passed = verdict["passed"]
    score = verdict["score"]
    issue_count = verdict["issue_count"]
    prompt_sha256 = record["prompt_sha256"]
    if isinstance(issue_count, bool) or not isinstance(issue_count, int):
        found += [shape("generalist_review issue_count is not an integer")]
    elif issue_count < 0:
        found += [shape("generalist_review issue_count is negative")]
    reviewed_files = {
        (entry["path"], entry["sha256"]) for entry in files if entry["reviewed"]
    }
    if not all(
        value is None or _is_sha256(value)
        for value in (
            *reviewed.values(),
            *(applied_block[field] for field in _DIGEST_FIELDS),
        )
    ):
        return found + [shape("generalist_review has a malformed digest")]
    rulespec_path = applied_block["rulespec_path"]
    test_path = applied_block["test_path"]
    if not (rulespec_path is None or is_rulespec_module_path(rulespec_path)) or not (
        test_path is None
        or (
            rulespec_path is not None
            and test_path == companion_test_manifest_path(rulespec_path)
        )
    ):
        return found + [shape("generalist_review has a malformed applied path")]

    scored = status in _BOUND_STATUSES or reason == REASON_CONTENT_NOT_APPLIED
    if not scored:
        if (
            passed is not None
            or score is not None
            or issue_count != 0
            or prompt_sha256 is not None
            or any(value is not None for value in reviewed.values())
            or any(value is not None for value in applied_block.values())
            or coverage is not None
            or reviewed_files
            or backend is not None
            or model is not None
        ):
            found += [
                verdict_issue(
                    "generalist_review carries a verdict for a review that has none"
                )
            ]
        return found

    if (
        not isinstance(passed, bool)
        or _finite_score(score) is None
        or not _is_sha256(prompt_sha256)
        or not _is_sha256(reviewed["rulespec_sha256"])
        or not _is_sha256(reviewed["rulespec_content_sha256"])
        or not (reviewed["test_sha256"] is None or _is_sha256(reviewed["test_sha256"]))
        or not _is_sha256(reviewed["test_content_sha256"])
        # A reviewer that read no companion file has the content digest of
        # an absent one, and only that reviewer has it.
        or (reviewed["test_sha256"] is None)
        != (reviewed["test_content_sha256"] == _ABSENT_TEST_CONTENT_SHA256)
    ):
        return found + [shape("generalist_review scored verdict is malformed")]
    if passed and float(score) < PASS_SCORE_THRESHOLD:
        found += [
            verdict_issue("generalist_review verdict passes below the pass threshold")
        ]
    if status in {STATUS_PASSED, STATUS_PARTIAL} and not passed:
        found += [
            verdict_issue(f"generalist_review status {status} contradicts its verdict")
        ]
    if status == STATUS_FAILED and passed:
        found += [
            verdict_issue("generalist_review status failed contradicts its verdict")
        ]
    if status == STATUS_UNAVAILABLE:
        if (
            any(value is not None for value in applied_block.values())
            or reviewed_files
            or coverage is not None
        ):
            found += [
                binding(
                    "generalist_review says the reviewed content was not applied, "
                    "but names an applied file"
                )
            ]
        return found

    if (
        applied_block["rulespec_content_sha256"] != reviewed["rulespec_content_sha256"]
        or applied_block["test_content_sha256"] != reviewed["test_content_sha256"]
    ):
        found += [
            binding(
                "generalist_review verdict is on content that differs from the "
                "applied content"
            )
        ]
    if reviewed[_DUMPED_FIELD] not in {None, reviewed["rulespec_content_sha256"]}:
        found += [
            binding("generalist_review dump digest differs from reviewed content")
        ]
    # The verdict covers the named module path and its companion test path,
    # each with the named digest, and no other path with equal bytes.
    pair_files = {
        (path, digest)
        for path, digest in (
            (rulespec_path, applied_block["rulespec_sha256"]),
            # An applied companion counts as reviewed only when the reviewer
            # read a companion file.
            (
                test_path,
                None
                if reviewed["test_sha256"] is None
                else applied_block["test_sha256"],
            ),
        )
        if digest is not None
    }
    applied_pair_files = {
        (path, digest)
        for path, digest in (
            (rulespec_path, applied_block["rulespec_sha256"]),
            (test_path, applied_block["test_sha256"]),
        )
        if digest is not None
    }
    if (
        not _is_sha256(applied_block["rulespec_sha256"])
        or rulespec_path is None
        or (test_path is None) != (applied_block["test_sha256"] is None)
        or reviewed_files != pair_files
        # The pair the verdict is bound to is among the listed files.
        or not applied_pair_files <= record_files
        or (
            applied_block["test_sha256"] is None
            and applied_block["test_content_sha256"] != _ABSENT_TEST_CONTENT_SHA256
        )
    ):
        found += [
            binding(
                "generalist_review verdict is not bound to the applied file digests"
            )
        ]
    # A file is complete only with chunk ranges that cover its reviewed
    # bytes exactly once: the module by the reviewed module digest, the
    # companion test by the reviewed companion digest.
    covered = {
        rulespec_path: _text_seen_whole(
            coverage, "rulespec", reviewed["rulespec_sha256"]
        ),
        test_path: reviewed["test_sha256"] is not None
        and _text_seen_whole(coverage, "test", reviewed["test_sha256"]),
    }
    if any(
        entry["complete"] and not covered.get(entry["path"], False) for entry in files
    ):
        found += [
            binding(
                "generalist_review marks a file complete without chunk ranges "
                "that cover it exactly once"
            )
        ]
    everything_in_full = bool(files) and all(
        entry["reviewed"] and entry["complete"] for entry in files
    )
    source_in_full = _source_seen_whole(coverage, reviewed[_SOURCE_FIELD])
    if status == STATUS_PASSED and not everything_in_full:
        found += [
            binding(
                "generalist_review status passed, but an applied file was not "
                "reviewed in full"
            )
        ]
    if status == STATUS_PASSED and not source_in_full:
        found += [
            binding(
                "generalist_review status passed, but the source excerpt was not "
                "reviewed in full"
            )
        ]
    if status == STATUS_PARTIAL:
        all_reviewed = bool(files) and all(entry["reviewed"] for entry in files)
        if not all_reviewed:
            expected = REASON_UNREVIEWED_FILES
        elif not everything_in_full:
            expected = REASON_TRUNCATED
        else:
            expected = REASON_SOURCE_TRUNCATED
        if (everything_in_full and source_in_full) or reason != expected:
            found += [
                binding("generalist_review status partial contradicts its file list")
            ]
    return found


_REFUSAL_PREFIX = "A passed generalist review"


class ReviewRequirementRefused(RuntimeError):
    """A required generalist review did not pass; nothing may be applied."""


def review_requirement_refusal(record: Mapping[str, object]) -> str | None:
    """Return why a required review blocks this encode, or None when it passed."""

    if record.get("status") == STATUS_PASSED:
        return None
    status = record.get("status")
    reason = record.get("reason")
    detail = f"status {status}" + (f" ({reason})" if reason else "")
    return _REFUSAL_PREFIX + (
        " of every applied file is required "
        f"(--require-review) and this encode has none: {detail}. "
        "Nothing was applied."
    )
