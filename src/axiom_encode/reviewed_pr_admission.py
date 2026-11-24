"""Live GitHub authorization for one protected, reviewed candidate promotion.

Selectors and cached JSON are never authority. The trusted provisioner and the
protected promotion command each fetch GitHub directly. Public repositories
need no credential, so no repository token enters the signing runtime.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Final

# The same human owners who authorize encoder signing-critical changes. Numeric
# GitHub identities prevent a recycled login from gaining signing authority.
# Changes to this policy require code-owner review, not candidate PR review.
TRUSTED_SIGNING_REVIEWERS = {6076111: "MaxGhenis", 110687043: "PavelMakarchuk"}

# What authorizes one promotion (#2301). Read at call time; there is no
# environment variable, CLI flag or repository setting.
#   False: no APPROVED review is required. The accountable act is the dispatch
#     of the protected signing run by an enrolled owner account: the live run's
#     actor (who chose the inputs) and its triggering actor (who started this
#     attempt) must be the same ``TRUSTED_SIGNING_REVIEWERS`` identity (numeric
#     id, login and type ``User``). This identifies an account, not a person:
#     a tool or agent session that holds an owner's token passes. No second
#     person sees the candidate before it is signed. An outstanding
#     changes-requested review still refuses.
#   True (or any value that is not exactly ``False``): an independent enrolled
#     owner must have approved the exact candidate head, and any account with
#     write access may dispatch. This is the rule that was in force before.
# Every other check is the same under both values.
OWNER_APPROVAL_REQUIRED: Final[bool] = False
BASIS_OWNER_APPROVAL: Final = "owner_approval"
BASIS_OWNER_DISPATCH: Final = "owner_dispatch"
# The one field of an admission that holds the evidence for each basis. A
# record names its basis; it never carries an empty approvals list.
_BASIS_EVIDENCE_FIELD: Final = {
    BASIS_OWNER_APPROVAL: "approvals",
    BASIS_OWNER_DISPATCH: "dispatcher",
}
# v2 added ``admission_basis`` and the per-basis evidence field.
SCHEMA = "axiom-encode/live-reviewed-pr-admission/v2"
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_PAGES = 10
INDEX_PATH = ".axiom/index/provisions_to_rules.json"

# --- Several units in one candidate ("batch"). SWITCHED OFF. ------------------
#
# Issue #2301; ``docs/reviewed-pr-promotion.md``, "Several units in one
# candidate". With the switch on, one candidate pull request may carry up to
# ``MAX_PROMOTION_UNITS`` new modules, each with its own primary and its own
# companion. One protected run admits, validates and signs all of them, and
# writes one signed manifest per unit in the form of a single-unit promotion.
#
# The switch is a constant on encoder main, read at call time: no environment
# variable, CLI flag, dispatch input or repository setting turns it. While it
# is not exactly ``True`` this module does not know the batch forms: a selector
# with ``units`` and an admission with ``records`` are refused with the errors
# they had before the switch existed, and every single-unit read, result and
# error is unchanged.
MULTI_UNIT_PROMOTION_ENABLED: Final[bool] = False
# The most units of one batch. The bound is the anonymous GitHub API budget of
# one signing run (60 requests an hour for one address; the signing runtime
# holds no token). One admission of ``k`` units reads ``11 + 2k`` times with
# one page per list (two content reads per unit); a revalidation reads 4
# times. One run admits twice (preflight, root provisioner) and revalidates
# three times (before validation, before signing, before publication):
#   2 x (11 + 2k) + 3 x 4 = 34 + 4k <= 60 gives k <= 6.
# A single-unit promotion costs 38 of the 60. A larger batch needs fewer or
# authenticated reads first, in a reviewed change of this constant.
# ``tests/test_multi_unit_promotion_admission.py`` pins the arithmetic.
MAX_PROMOTION_UNITS: Final[int] = 6
ANONYMOUS_REQUESTS_PER_HOUR: Final = 60
BATCH_SCHEMA: Final = "axiom-encode/live-reviewed-pr-batch-admission/v1"
# The root provisioner reads at most this many bytes of a selector file.
MAX_SELECTOR_BYTES: Final = 4096


class AdmissionError(ValueError):
    """The requested candidate has no current authorization."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise AdmissionError("GitHub admission requests must not redirect")


def github_json(path: str) -> object:
    """Read a bounded response only from the fixed GitHub API origin."""
    if not path.startswith("/repos/TheAxiomFoundation/") or "#" in path:
        raise AdmissionError("invalid admission API path")
    request = urllib.request.Request(
        "https://api.github.com" + path,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "axiom-encode-reviewed-pr-admission",
        },
    )
    # Ignore environment proxies: the API origin, TLS trust and redirects are
    # not dispatch inputs. HTTP errors/rate limits fail closed.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    with opener.open(request, timeout=30) as response:
        if response.status != 200:
            raise AdmissionError("GitHub admission query did not succeed")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise AdmissionError("GitHub admission response exceeds its bound")
    return json.loads(raw)


@dataclass(frozen=True)
class PromotionSelector:
    country: str
    pr: int
    head: str
    citation: str
    rulespec_path: str
    branch: str
    actor: str
    run_id: str = ""

    def __post_init__(self):
        if not re.fullmatch(r"[a-z]{2}", self.country):
            raise AdmissionError("invalid country")
        if type(self.pr) is not int or self.pr <= 0:
            raise AdmissionError("invalid candidate PR number")
        if not re.fullmatch(r"[0-9a-f]{40}", self.head):
            raise AdmissionError("candidate head must be an exact commit")
        for value in (self.citation, self.rulespec_path, self.branch):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 512
                or not re.fullmatch(r"[A-Za-z0-9._/-]+", value)
                or any(part in {"", ".", ".."} for part in value.split("/"))
            ):
                raise AdmissionError("invalid candidate selector path or branch")
        if not re.fullmatch(r"[A-Za-z0-9-]{1,39}", self.actor):
            raise AdmissionError("invalid dispatch actor")
        if not re.fullmatch(r"[1-9][0-9]{0,19}", self.run_id):
            raise AdmissionError("invalid protected workflow run ID")
        if not self.citation.startswith(
            self.country + "/"
        ) and not self.citation.startswith(self.country + "-"):
            raise AdmissionError("citation country differs from candidate country")
        if not self.rulespec_path.endswith(".yaml") or self.rulespec_path.endswith(
            ".test.yaml"
        ):
            raise AdmissionError("candidate must name a primary YAML")

    @property
    def repository(self) -> str:
        return f"TheAxiomFoundation/rulespec-{self.country}"

    @property
    def companion(self) -> str:
        return self.rulespec_path.removesuffix(".yaml") + ".test.yaml"


_BATCH_SELECTOR_FIELDS: Final = frozenset(
    {"country", "pr", "head", "branch", "actor", "run_id", "units"}
)


@dataclass(frozen=True)
class BatchPromotionSelector:
    """One candidate pull request and every unit it carries.

    ``units`` is ``((citation, rulespec_path), ...)`` in strictly ascending
    order of ``rulespec_path``, so no primary is named twice and one batch has
    one selector. Every unit is checked as the selector of a single-unit
    promotion is.
    """

    country: str
    pr: int
    head: str
    branch: str
    actor: str
    units: tuple[tuple[str, str], ...]
    run_id: str = ""

    def __post_init__(self):
        if (
            not isinstance(self.units, tuple)
            or not self.units
            or any(
                not isinstance(unit, tuple)
                or len(unit) != 2
                or not all(isinstance(value, str) for value in unit)
                for unit in self.units
            )
        ):
            raise AdmissionError("batch selector must name at least one unit")
        selectors = self.unit_selectors()
        paths = [unit.rulespec_path for unit in selectors]
        if any(left >= right for left, right in zip(paths, paths[1:])):
            raise AdmissionError(
                "batch units must be in ascending path order and name no primary twice"
            )
        citations = [unit.citation for unit in selectors]
        if len(set(citations)) != len(citations):
            raise AdmissionError("batch units must not name a citation twice")

    def unit_selectors(self) -> tuple[PromotionSelector, ...]:
        """Each unit as the selector of a single-unit promotion of it."""
        return tuple(
            PromotionSelector(
                country=self.country,
                pr=self.pr,
                head=self.head,
                citation=citation,
                rulespec_path=rulespec_path,
                branch=self.branch,
                actor=self.actor,
                run_id=self.run_id,
            )
            for citation, rulespec_path in self.units
        )

    @property
    def repository(self) -> str:
        return f"TheAxiomFoundation/rulespec-{self.country}"

    def payload(self) -> dict:
        """The JSON form: what ``selector_from_payload`` reads back."""
        return {
            "country": self.country,
            "pr": self.pr,
            "head": self.head,
            "branch": self.branch,
            "actor": self.actor,
            "run_id": self.run_id,
            "units": [
                {"citation": citation, "rulespec_path": rulespec_path}
                for citation, rulespec_path in self.units
            ],
        }


def _require_batch_in_force(selector: BatchPromotionSelector) -> None:
    """Refuse a batch while the switch is off, and one over the unit limit."""
    if MULTI_UNIT_PROMOTION_ENABLED is not True:
        raise AdmissionError("multi-unit promotion is switched off")
    limit = MAX_PROMOTION_UNITS
    if type(limit) is not int or len(selector.units) > limit:
        raise AdmissionError(
            f"batch names {len(selector.units)} units; the limit is {limit}"
        )
    if (
        len(json.dumps(selector.payload(), sort_keys=True).encode()) + 1
        > MAX_SELECTOR_BYTES
    ):
        raise AdmissionError("batch selector exceeds the provisioner's size bound")


def _batch_selector_from_payload(payload: dict) -> BatchPromotionSelector:
    if set(payload) != _BATCH_SELECTOR_FIELDS or not isinstance(payload["units"], list):
        raise AdmissionError("batch promotion selector has unexpected fields")
    units = []
    for unit in payload["units"]:
        unit = _object(unit, "batch unit")
        if set(unit) != {"citation", "rulespec_path"}:
            raise AdmissionError("batch unit has unexpected fields")
        units.append((unit["citation"], unit["rulespec_path"]))
    selector = BatchPromotionSelector(
        **{key: value for key, value in payload.items() if key != "units"},
        units=tuple(units),
    )
    _require_batch_in_force(selector)
    return selector


def _object(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise AdmissionError(f"GitHub {label} is not an object")
    return value


def _leaf_entries(tree: dict) -> dict[str, dict]:
    """Retain every immutable leaf, including symlinks and submodules."""
    if tree.get("truncated") is not False or not isinstance(tree.get("tree"), list):
        raise AdmissionError("Git tree cannot be inspected completely")
    leaves = {}
    seen = set()
    for entry in tree["tree"]:
        entry = _object(entry, "tree entry")
        path = entry.get("path")
        if (
            not isinstance(path, str)
            or not path
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or path in seen
            or (entry.get("type"), entry.get("mode"))
            not in {
                ("tree", "040000"),
                ("blob", "100644"),
                ("blob", "100755"),
                ("blob", "120000"),
                ("commit", "160000"),
            }
            or not re.fullmatch(r"[0-9a-f]{40}", str(entry.get("sha", "")))
        ):
            raise AdmissionError("invalid or duplicate Git tree entry")
        seen.add(path)
        if entry["type"] != "tree":
            leaves[path] = entry
    return leaves


def _require_immutable_delta(
    base_tree: dict, head_tree: dict, files: list[dict], comparison: dict
) -> dict[str, dict]:
    base_entries = _leaf_entries(base_tree)
    head_entries = _leaf_entries(head_tree)
    delta = {}
    for path in base_entries.keys() | head_entries.keys():
        old, new = base_entries.get(path), head_entries.get(path)

        def identity(entry):
            return (
                None if entry is None else (entry["type"], entry["mode"], entry["sha"])
            )

        if identity(old) != identity(new):
            delta[path] = (
                "added" if old is None else "removed" if new is None else "modified"
            )
            if new is not None and (new["type"] != "blob" or new["mode"] != "100644"):
                raise AdmissionError("changed candidate file is not a regular blob")
    # PR files can race or contradict an immutable comparison. Neither can
    # substitute a permitted index for a workflow changed at the approved SHA.
    for evidence in (files, comparison.get("files")):
        if not isinstance(evidence, list):
            raise AdmissionError("immutable comparison lacks its file delta")
        observed = {}
        for item in evidence:
            item = _object(item, "changed file")
            path = item.get("filename")
            if not isinstance(path, str) or path in observed:
                raise AdmissionError("invalid or duplicate changed file")
            observed[path] = item.get("status")
        if observed != delta:
            raise AdmissionError(
                "file scope differs from complete immutable tree delta"
            )
    return head_entries


# The command that binds an existing candidate (docs/reviewed-pr-promotion.md).
SOURCE_PIN_COMMAND: Final = (
    "axiom-encode bind-source-verification --repo <rulespec checkout> "
    "--citation <citation> --corpus-path <corpus checkout>"
)
_BARE_KEY = re.compile(r"([A-Za-z_][A-Za-z0-9_]*):(?: +(.*))?")
_PIN_PLAIN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@+-]*")
_PIN_VALUE = re.compile(
    r"""(?:(?P<plain>[^\s'"#][^\s#]*)|'(?P<single>[^'\\]*)'|"(?P<double>[^"\\]*)")"""
    r"""(?: +#.*)?"""
)
_BLOCK_SCALAR = re.compile(r"[|>](?:[+-][1-9]?|[1-9][+-]?)?(?: +#.*)?")
_TRAILER = re.compile(r" *(?: #.*)?")
# A value that starts with one of these is an anchor, an alias, a tag or
# reserved; the reader does not follow any of them.
_UNREAD_STARTS = "&*!%@`"
# Unquoted text that some YAML 1.1 or 1.2 reader takes for a number, a
# boolean or null: digits with signs, dots, exponents, underscores, radix
# prefixes or sexagesimal colons, and the reserved words.
_NUMBER_LIKE = re.compile(r"[-+]?(?:0[xXoObB][0-9A-Fa-f_]*|[.0-9][-+.0-9eE_:]*)")
_RESERVED_WORDS = frozenset(
    "true false yes no on off y n null nil none ~ .inf .nan inf nan".split()
)


def plain_scalar_is_string(value: str) -> bool:
    """Whether every YAML reader loads this unquoted text as this string."""
    return (
        _PIN_PLAIN.fullmatch(value) is not None
        and _NUMBER_LIKE.fullmatch(value) is None
        and value.lower().lstrip("-+") not in _RESERVED_WORDS
    )


def _pin_refusal(detail: str) -> AdmissionError:
    error = AdmissionError(
        f"candidate module is not bound to its source: {detail}. "
        "`module.source_verification` must be in the plain block form with "
        "`corpus_citation_path` and `source_sha256`. Run "
        f"`{SOURCE_PIN_COMMAND}` on the candidate branch (rewrite the block by "
        "hand if the command refuses its shape), push the change and dispatch "
        "the new head"
    )
    error.detail = detail  # type: ignore[attr-defined]
    return error


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _after_quote(text: str, quote: str, start: int) -> str | None:
    """The text after the quote that closes at or after ``start``, or ``None``."""
    index = start
    while index < len(text):
        char = text[index]
        if quote == '"' and char == "\\":
            index += 2
        elif char != quote:
            index += 1
        elif quote == "'" and text.startswith("''", index):
            index += 2
        else:
            return text[index + 1 :]
    return None


def _split_key(text: str) -> tuple[str | None, str]:
    """``(key, value)`` of a block-mapping line, or ``(None, text)`` for a scalar."""
    if text[:1] in ('"', "'"):
        rest = _after_quote(text, text[0], 1)
        if rest is None or not rest.lstrip(" ").startswith(":"):
            return None, text
        key, tail = text[: len(text) - len(rest)], rest.lstrip(" ")[1:]
    else:
        match = re.search(r":(?= |$)", text)
        if match is None:
            return None, text
        key, tail = text[: match.start()], text[match.end() :]
    if tail and not tail.startswith(" "):
        return None, text
    return key, tail.strip(" ")


def _module_block(lines: list[str]) -> tuple[list[tuple[int, int, str]], int]:
    """Lex the file through the end of the top-level ``module`` block.

    Returns the structural lines of the block as ``(index, indent, content)``
    and the index of the first line after the block. A structural line starts
    a mapping entry, a sequence item or a scalar. Lines inside a block scalar,
    or inside a quoted or plain scalar that continues over lines, are not
    structural: YAML reads no key from them. A shape this lexer does not
    follow refuses.
    """
    in_module = False
    block: list[tuple[int, int, str]] = []
    quote: str | None = None
    scalar_base: int | None = None  # block scalar: content is indented past it
    plain_base: int | None = None  # plain scalar: continues when indented past it
    started = False
    for index, line in enumerate(lines):
        if quote is not None:
            rest = _after_quote(line, quote, 0)
            if rest is None:
                continue
            quote = None
            if _TRAILER.fullmatch(rest) is None:
                raise _pin_refusal(
                    "a quoted scalar over several lines is a key or has text after it"
                )
            continue
        if not line.strip(" "):
            continue
        indent = _indent(line)
        content = line[indent:]
        if scalar_base is not None:
            if indent > scalar_base:
                continue
            scalar_base = None
        if content.startswith("#"):
            continue
        if plain_base is not None and indent > plain_base:
            continue
        plain_base = None
        if not started and indent:
            raise _pin_refusal("the first line of the mapping is indented")
        started = True
        if indent == 0:
            if in_module:
                return block, index
            if _BARE_KEY.fullmatch(content) is None and not content.startswith("- "):
                raise _pin_refusal(
                    "a top-level line is not a plain `name:` key or a list item"
                )
            if content.split(":", 1)[0] == "module":
                if content.rstrip(" ") != "module:":
                    raise _pin_refusal("`module:` is not a plain block mapping")
                in_module = True
                continue
        elif in_module:
            block.append((index, indent, content))
        # The column a continuation or block scalar content must pass.
        base, text = indent - 1, content
        while text == "-" or text.startswith("- "):
            rest = text[1:].lstrip(" ")
            base = indent + len(content) - len(text)
            text = rest
        key, value = _split_key(text)
        if key is not None:
            base = indent + len(content) - len(text)
            if key[:1] in "?[{" + _UNREAD_STARTS:
                raise _pin_refusal("a key uses a tag, an anchor or a complex form")
        elif text == "?" or text.startswith("? "):
            raise _pin_refusal("a complex `? key` mapping")
        if not value:
            continue
        first = value[0]
        if first in ('"', "'"):
            rest = _after_quote(value, first, 1)
            if rest is None:
                quote = first
            elif _TRAILER.fullmatch(rest) is None:
                raise _pin_refusal("text follows a quoted scalar on its line")
        elif first in "|>":
            if _BLOCK_SCALAR.fullmatch(value) is None:
                raise _pin_refusal("a block scalar header this reader does not read")
            scalar_base = base
        elif first in "[{":
            opened = sum(value.count(char) for char in "[{")
            closed = sum(value.count(char) for char in "]}")
            if opened != closed or '"' in value or "'" in value:
                raise _pin_refusal(
                    "a flow collection that has quotes or continues over lines"
                )
        elif first in _UNREAD_STARTS:
            raise _pin_refusal("a value uses an anchor, an alias or a tag")
        else:
            plain_base = base
    if quote is not None:
        raise _pin_refusal("a quoted scalar is not closed")
    if not in_module:
        raise _pin_refusal("the file has no top-level plain block `module:` mapping")
    return block, len(lines)


def read_candidate_source_pin(raw: bytes) -> tuple[str | None, str | None]:
    """Read ``module.source_verification`` by a strict line read; no YAML parser.

    Returns ``(corpus_citation_path, source_sha256)``; a missing key is
    ``None``. Standard library only: this module also runs where the encoder
    package and PyYAML are absent. It accepts one shape, the plain block form
    that the encoder and ``bind-source-verification`` write::

        module:
          source_verification:
            corpus_citation_path: <path>
            source_sha256: <64 hex>

    The rule is positive. Every top-level line is a plain ``name:`` key or a
    list item, and ``module:`` occurs once. Every key directly under ``module``
    is a plain ``name:`` key, and ``source_verification:`` occurs once among
    them. Every entry of the block is one ``name: value`` line with a one-line
    plain or quoted value, and no name repeats. Through the end of the
    ``module`` block the reader follows block scalars and quoted or plain
    scalars that continue over lines, so text inside them is not read as a
    key. Every other shape refuses: a quoted, tagged, anchored or complex key,
    a merge key, an alias, flow style on these mappings, a flow collection
    with quotes, tab indentation, a document marker, a byte order mark, a line
    break other than LF or CRLF, an unquoted value that a YAML reader could
    take for a number. A refusal is not a claim that the YAML is wrong; it
    means this reader will not vouch for it.

    This is an early refusal, not the binding proof. It does not know the
    corpus. The protected job still parses the module with a YAML parser and
    compares the pin with the resolver's attestation before it signs.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise _pin_refusal("the module is not UTF-8") from None
    lines = [line.removesuffix("\r") for line in text.split("\n")]
    if any(mark in line for line in lines for mark in "\r\x85\u2028\u2029"):
        raise _pin_refusal("the module uses a line break other than LF or CRLF")
    if text.startswith("\ufeff") or any(
        line.startswith(("---", "...", "%")) for line in lines
    ):
        raise _pin_refusal(
            "the module has a byte order mark, a directive or a document marker"
        )
    if any(line.lstrip(" ").startswith("\t") for line in lines):
        raise _pin_refusal("a line of the file is indented with tabs")
    block, after = _module_block(lines)
    for line in lines[after:]:
        # Past the module block only top-level keys matter: a second `module`
        # key, in any spelling, would replace the block that is read here.
        if not line or line.startswith((" ", "#", "- ")):
            continue
        match = _BARE_KEY.fullmatch(line)
        if match is None or match.group(1) == "module":
            raise _pin_refusal(
                "a top-level line is not a plain `name:` key or a list item, or "
                "`module` occurs twice"
            )
    if not block:
        raise _pin_refusal("`module` is empty")
    key_indent = block[0][1]
    if any(indent < key_indent for _index, indent, _content in block):
        raise _pin_refusal("`module` keys are not indented evenly")
    keys = [
        (index, content) for index, indent, content in block if indent == key_indent
    ]
    # A list item at the key column belongs to the key above it; it is no key.
    keys = [(index, content) for index, content in keys if not content.startswith("- ")]
    names = []
    for _index, content in keys:
        match = _BARE_KEY.fullmatch(content)
        if match is None:
            raise _pin_refusal(
                "a key directly under `module` is not a plain `name:` key (it is "
                "quoted, tagged, anchored, a merge key or flow style)"
            )
        names.append(match.group(1))
    if "source_verification" not in names:
        raise _pin_refusal("the module has no `module.source_verification` block")
    position = names.index("source_verification")
    header, header_content = keys[position]
    if (
        names.count("source_verification") != 1
        or header_content.rstrip(" ") != "source_verification:"
    ):
        raise _pin_refusal(
            "`module.source_verification` is not a single plain block mapping"
        )
    end = keys[position + 1][0] if position + 1 < len(keys) else after
    entries = [
        line
        for line in lines[header + 1 : end]
        if line.strip(" ") and not line.strip(" ").startswith("#")
    ]
    if not entries:
        raise _pin_refusal("`module.source_verification` is empty")
    entry_indent = _indent(entries[0])
    values: dict[str, str] = {}
    for line in entries:
        match = (
            _BARE_KEY.fullmatch(line[entry_indent:].rstrip(" "))
            if _indent(line) == entry_indent and entry_indent > key_indent
            else None
        )
        value = _PIN_VALUE.fullmatch(match.group(2) or "") if match else None
        if match is None or value is None or match.group(1) in values:
            raise _pin_refusal(
                "`module.source_verification` is not in the plain block form "
                "(one `key: value` line per entry, no duplicate key)"
            )
        plain = value.group("plain")
        if (
            plain is not None
            and match.group(1) in ("corpus_citation_path", "source_sha256")
            and not plain_scalar_is_string(plain)
        ):
            raise _pin_refusal(
                "a `module.source_verification` value is not a plain string "
                "(quote a value that a YAML reader could take for a number)"
            )
        values[match.group(1)] = next(
            group
            for group in value.group("plain", "single", "double")
            if group is not None
        )
    return values.get("corpus_citation_path"), values.get("source_sha256")


def require_candidate_source_pin(raw: bytes, citation: str) -> None:
    """Refuse a candidate primary that a promotion could not bind to its source.

    The module must name the promotion's citation and carry a 64-character
    lowercase hex ``source_sha256``. Whether that pin is the right one is
    decided later, by the protected binding check against the corpus.
    """
    path, pin = read_candidate_source_pin(raw)
    if path != citation:
        raise _pin_refusal(
            "`module.source_verification.corpus_citation_path` is missing or is "
            "not the promotion citation"
        )
    if pin is None or re.fullmatch(r"[0-9a-f]{64}", pin) is None:
        raise _pin_refusal(
            "`module.source_verification.source_sha256` is missing or is not 64 "
            "lowercase hex characters"
        )


def selector_from_payload(
    value: object,
) -> PromotionSelector | BatchPromotionSelector:
    payload = _object(value, "promotion selector")
    if MULTI_UNIT_PROMOTION_ENABLED is True and "units" in payload:
        return _batch_selector_from_payload(payload)
    if set(payload) != set(PromotionSelector.__dataclass_fields__):
        raise AdmissionError("promotion selector has unexpected fields")
    return PromotionSelector(**payload)


def _admission_basis() -> str:
    """The basis the policy in force admits. Only exactly ``False`` drops the approval."""
    return (
        BASIS_OWNER_DISPATCH
        if OWNER_APPROVAL_REQUIRED is False
        else BASIS_OWNER_APPROVAL
    )


def _enrolled_run_account(run: dict, field: str, label: str) -> tuple[int, str]:
    account = _object(run.get(field), label)
    identity = account.get("id")
    if (
        type(identity) is not int
        or identity not in TRUSTED_SIGNING_REVIEWERS
        or account.get("login") != TRUSTED_SIGNING_REVIEWERS[identity]
        or account.get("type") != "User"
    ):
        raise AdmissionError("dispatch actor is not a verified enrolled signing owner")
    return identity, account["login"]


def _enrolled_dispatcher(run: dict, selector: PromotionSelector) -> dict:
    """The live run's dispatcher as one enrolled owner account, or refuse.

    Two accounts are read from the live run and must be the same enrolled
    identity: ``actor``, who dispatched the run and chose its inputs, and
    ``triggering_actor``, who started this attempt. So an owner's re-run of
    another account's dispatch refuses, and so does another account's re-run
    of an owner's dispatch. For each, the numeric id is the identity, the
    login must be the enrolled login of that id, and the account type must be
    ``User``. The selector's actor must be that login. A missing or malformed
    field refuses.

    Type ``User`` excludes Apps and bots. It does not prove that a person
    acted: any session that holds the owner's token is that ``User``.
    """
    actor_id, _login = _enrolled_run_account(run, "actor", "run actor")
    triggering_id, login = _enrolled_run_account(
        run, "triggering_actor", "triggering actor"
    )
    if actor_id != triggering_id or selector.actor != login:
        raise AdmissionError("dispatch actor is not a verified enrolled signing owner")
    return {"login": login, "actor_id": actor_id, "triggering_actor_id": triggering_id}


def revalidate_admission(
    expected: object, *, fetch: Callable[[str], object] = github_json
) -> dict:
    payload = _object(expected, "provisioned admission")
    basis = payload.get("admission_basis")
    if not isinstance(basis, str) or basis not in _BASIS_EVIDENCE_FIELD:
        raise AdmissionError("unknown admission basis")
    # The batch form exists only while its switch is on.
    batch = MULTI_UNIT_PROMOTION_ENABLED is True and "records" in payload
    if set(payload) != {
        "schema",
        "selector",
        "records" if batch else "record",
        "workflow_commit",
        "admission_basis",
        _BASIS_EVIDENCE_FIELD[basis],
    } or payload.get("schema") != (BATCH_SCHEMA if batch else SCHEMA):
        raise AdmissionError("invalid provisioned admission")
    if basis != _admission_basis():
        raise AdmissionError("admission basis is not the policy in force")
    selector = selector_from_payload(payload["selector"])
    if isinstance(selector, BatchPromotionSelector) is not batch:
        raise AdmissionError("invalid provisioned admission")
    root = "/repos/" + selector.repository
    run = _object(
        fetch(f"/repos/TheAxiomFoundation/axiom-encode/actions/runs/{selector.run_id}"),
        "protected run",
    )
    if (
        run.get("status") != "in_progress"
        or run.get("head_sha") != payload["workflow_commit"]
        or _object(run.get("triggering_actor"), "triggering actor").get("login")
        != selector.actor
    ):
        raise AdmissionError("protected run changed after provisioning")
    if (
        basis == BASIS_OWNER_DISPATCH
        and _enrolled_dispatcher(run, selector) != payload["dispatcher"]
    ):
        raise AdmissionError("dispatching owner changed after provisioning")
    pr_path = f"{root}/pulls/{selector.pr}"
    pr = _object(fetch(pr_path), "PR")
    base, author_id = _pr_identity(pr, selector)
    main = _object(fetch(root + "/git/ref/heads/main"), "main ref")
    reviews = _pages(fetch, pr_path + "/reviews")
    if basis == BASIS_OWNER_DISPATCH:
        # No approval to compare; the veto is still read live.
        _decisive_reviews(reviews)
        evidence_unchanged = True
    else:
        evidence_unchanged = (
            _approved_reviews(reviews, selector, author_id) == payload["approvals"]
        )
    if (
        base
        != (
            _batch_records_base(payload["records"], selector)
            if batch
            else payload["record"]["base_sha"]
        )
        or _object(main.get("object"), "main object").get("sha") != base
        or not evidence_unchanged
    ):
        raise AdmissionError(
            "live approval or candidate differs from provisioned admission"
        )
    # The head and base are immutable Git objects already fully inspected by
    # the root provisioner. The signer separately rehashes the local primary
    # and companion; re-fetch only mutable authorization, not all source bytes.
    return payload


def _batch_records_base(records: object, selector: BatchPromotionSelector) -> str:
    """The one main base of a batch admission's records, or refuse.

    One record per unit, in the selector's order, each bound to its unit, all
    on one base.
    """
    if not isinstance(records, list) or len(records) != len(selector.units):
        raise AdmissionError("batch admission lacks one record per unit")
    bases = []
    for record, (citation, rulespec_path) in zip(records, selector.units):
        binding = _object(_object(record, "batch record").get("binding"), "binding")
        if (
            binding.get("citation") != citation
            or binding.get("rulespec_path") != rulespec_path
        ):
            raise AdmissionError(
                f"batch record is not bound to its unit {citation} ({rulespec_path})"
            )
        bases.append(record.get("base_sha"))
    # Compared, not hashed: a base of any JSON type refuses as an admission.
    if not isinstance(bases[0], str) or any(base != bases[0] for base in bases):
        raise AdmissionError("batch records differ in their main base")
    return bases[0]


def _batch_scope_issue(
    units: tuple[PromotionSelector, ...], files: list[dict]
) -> str | None:
    """Name the unit or the file that makes a batch's change set wrong.

    The change set of the candidate must be exactly the named units: each
    unit's primary and companion, both added, and at most the derived index
    besides. A unit the operator did not name shows as a file of no named
    unit; a named unit the candidate lacks shows as a missing file.
    """
    status = {}
    for item in files:
        path = item.get("filename")
        if not isinstance(path, str) or path in status:
            return "candidate PR lists an invalid or duplicate changed file"
        status[path] = item.get("status")
    owned = {}
    for unit in units:
        for path in (unit.rulespec_path, unit.companion):
            owned[path] = unit
            label = f"unit {unit.citation} ({unit.rulespec_path})"
            if path not in status:
                return f"candidate PR lacks {path} of {label}"
            if status[path] != "added":
                return f"candidate PR does not add {path} of {label}"
    for path in sorted(status):
        if path not in owned and path != INDEX_PATH:
            return (
                f"candidate PR changes {path}, which belongs to no unit named "
                "in the dispatch"
            )
    return None


def _pages(fetch: Callable[[str], object], path: str) -> list[dict]:
    result = []
    for page in range(1, MAX_PAGES + 1):
        items = fetch(f"{path}?per_page=100&page={page}")
        if not isinstance(items, list) or len(items) > 100:
            raise AdmissionError("invalid paginated GitHub response")
        result.extend(_object(item, "list entry") for item in items)
        if len(items) < 100:
            return result
    raise AdmissionError("GitHub admission pagination limit reached")


def _pr_identity(pr: dict, selector: PromotionSelector) -> tuple[str, int]:
    head = _object(pr.get("head"), "PR head")
    base = _object(pr.get("base"), "PR base")
    author = _object(pr.get("user"), "PR author")
    if (
        pr.get("number") != selector.pr
        or pr.get("state") != "open"
        or pr.get("merged") is not False
        or head.get("sha") != selector.head
        or head.get("ref") != selector.branch
        or _object(head.get("repo"), "head repository").get("full_name")
        != selector.repository
        or _object(base.get("repo"), "base repository").get("full_name")
        != selector.repository
        or base.get("ref") != "main"
        or not re.fullmatch(r"[0-9a-f]{40}", str(base.get("sha", "")))
        or type(author.get("id")) is not int
    ):
        raise AdmissionError("candidate PR identity, head, or main base differs")
    return base["sha"], author["id"]


def _decisive_reviews(reviews: list[dict]) -> dict[int, dict]:
    """Each reviewer's latest decisive review; refuse on the veto.

    Any reviewer's outstanding changes-requested review refuses, under both
    admission bases. It is not limited to enrolled owners.
    """
    # Order and reviewer identity decide the veto, so neither may be guessed.
    for review in reviews:
        if any(
            type(value) is not int or value <= 0
            for value in (
                review.get("id"),
                _object(review.get("user"), "reviewer").get("id"),
            )
        ):
            raise AdmissionError("review or reviewer identity is malformed")
    decisive: dict[int, dict] = {}
    for review in sorted(reviews, key=lambda item: item["id"]):
        reviewer = review["user"]
        identity = reviewer["id"]
        state = review.get("state")
        if state in {"CHANGES_REQUESTED", "APPROVED", "DISMISSED"}:
            decisive[identity] = review
        elif state not in {"COMMENTED", "PENDING"}:
            raise AdmissionError("unknown review state")
    if any(review["state"] == "CHANGES_REQUESTED" for review in decisive.values()):
        raise AdmissionError("candidate has unresolved changes requested")
    return decisive


def _approved_reviews(
    reviews: list[dict], selector: PromotionSelector, author_id: int
) -> list[dict]:
    decisive = _decisive_reviews(reviews)
    approvals = []
    for identity, review in decisive.items():
        reviewer = review["user"]
        if (
            identity in TRUSTED_SIGNING_REVIEWERS
            and reviewer.get("login") == TRUSTED_SIGNING_REVIEWERS[identity]
            and reviewer.get("type") == "User"
            and identity != author_id
            and reviewer["login"].lower() != selector.actor.lower()
            and review.get("state") == "APPROVED"
            and review.get("commit_id") == selector.head
            and type(review.get("id")) is int
            and review["id"] > 0
        ):
            approvals.append({"reviewer_id": identity, "review_id": review["id"]})
    if not approvals:
        raise AdmissionError(
            "candidate lacks an independent exact-head signing-owner approval"
        )
    return sorted(approvals, key=lambda value: value["reviewer_id"])


def verify_reviewed_pr(
    selector: PromotionSelector | BatchPromotionSelector,
    *,
    fetch: Callable[[str], object] = github_json,
) -> dict:
    """Derive a module-bound admission from live GitHub state.

    ``OWNER_APPROVAL_REQUIRED`` decides the basis: an enrolled owner's dispatch
    of the run (``False``) or an independent enrolled owner's approval of the
    exact head (``True``). The admission names its basis.

    New modules only. The exact primary/companion and optional derived index
    are the only allowed changes. No fork, stacked base, deleted/renamed file,
    signed-manifest change, workflow change or unrelated candidate is admitted.

    A ``BatchPromotionSelector`` (``MULTI_UNIT_PROMOTION_ENABLED``) names every
    unit of the candidate. The same reads and rules apply; the change set must
    be exactly the primaries and companions of the named units, and the
    admission carries one record per unit, each equal to the record of a
    single-unit admission of that unit.
    """
    batch = isinstance(selector, BatchPromotionSelector)
    if batch:
        _require_batch_in_force(selector)
    units = selector.unit_selectors() if batch else (selector,)
    run = _object(
        fetch(f"/repos/TheAxiomFoundation/axiom-encode/actions/runs/{selector.run_id}"),
        "protected run",
    )
    if (
        str(run.get("id")) != selector.run_id
        or run.get("event") != "workflow_dispatch"
        or run.get("path") != ".github/workflows/targeted-signed-reencode.yml"
        or run.get("head_branch") != "main"
        or run.get("status") != "in_progress"
        or not re.fullmatch(r"[0-9a-f]{40}", str(run.get("head_sha", "")))
        or _object(run.get("triggering_actor"), "triggering actor").get("login")
        != selector.actor
    ):
        raise AdmissionError(
            "admission is not bound to the live central main signing run"
        )
    basis = _admission_basis()
    dispatcher = (
        _enrolled_dispatcher(run, selector) if basis == BASIS_OWNER_DISPATCH else None
    )
    root = "/repos/" + selector.repository
    pr_path = f"{root}/pulls/{selector.pr}"
    before = _object(fetch(pr_path), "PR")
    base, author_id = _pr_identity(before, selector)
    main = _object(fetch(root + "/git/ref/heads/main"), "main ref")
    if _object(main.get("object"), "main object").get("sha") != base:
        raise AdmissionError("PR base is not the current main head")
    comparison = _object(
        fetch(f"{root}/compare/{base}...{selector.head}"), "candidate ancestry"
    )
    if (
        comparison.get("status") != "ahead"
        or _object(comparison.get("merge_base_commit"), "merge base").get("sha") != base
    ):
        raise AdmissionError("candidate must descend from the current main head")
    branch = urllib.parse.quote(selector.branch, safe="")
    tip = _object(fetch(root + "/git/ref/heads/" + branch), "candidate ref")
    if _object(tip.get("object"), "candidate object").get("sha") != selector.head:
        raise AdmissionError("candidate branch moved")
    files = _pages(fetch, pr_path + "/files")
    expected = {path for unit in units for path in (unit.rulespec_path, unit.companion)}
    base_tree = _object(fetch(f"{root}/git/trees/{base}?recursive=1"), "base tree")
    if base_tree.get("truncated") is not False or not isinstance(
        base_tree.get("tree"), list
    ):
        raise AdmissionError("base tree cannot be inspected completely")
    if batch:
        on_main = {
            entry.get("path") for entry in base_tree["tree"] if isinstance(entry, dict)
        }
        for unit in units:
            for path in (unit.rulespec_path, unit.companion):
                if path in on_main:
                    raise AdmissionError(
                        f"{path} of unit {unit.citation} ({unit.rulespec_path}) "
                        "already exists on main"
                    )
    if any(
        entry.get("path") in expected
        for entry in base_tree["tree"]
        if isinstance(entry, dict)
    ):
        raise AdmissionError("new candidate paths already exist on main")
    paths = [item.get("filename") for item in files]
    if batch:
        # The same rules as below, with the unit or the file named.
        issue = _batch_scope_issue(units, files)
        if issue is not None:
            raise AdmissionError(issue)
    if (
        len(set(paths)) != len(paths)
        or not expected.issubset(paths)
        or not set(paths).issubset(expected | {INDEX_PATH})
        or before.get("changed_files") != len(files)
        or any(
            item.get("status") != "added"
            for item in files
            if item.get("filename") in expected
        )
        or any(item.get("status") not in {"added", "modified"} for item in files)
    ):
        raise AdmissionError("candidate PR changes files outside its new module scope")

    reviews = _pages(fetch, pr_path + "/reviews")
    if basis == BASIS_OWNER_DISPATCH:
        # No approval is required; an outstanding changes-requested review
        # still refuses.
        _decisive_reviews(reviews)
        evidence: object = dispatcher
        purpose = (
            "Enrolled signing-owner dispatch of the protected signing run; no "
            "owner approval of the RuleSpec PR; new-module promotion only."
        )
    else:
        evidence = _approved_reviews(reviews, selector, author_id)
        purpose = "Live exact-head independent signing-owner approval on the RuleSpec PR; new-module promotion only."

    digests = {}
    contents = {}
    tree = _object(
        fetch(f"{root}/git/trees/{selector.head}?recursive=1"), "candidate tree"
    )
    if tree.get("truncated") is not False or not isinstance(tree.get("tree"), list):
        raise AdmissionError("candidate tree cannot be inspected completely")
    entries = _require_immutable_delta(base_tree, tree, files, comparison)
    for path in sorted(expected):
        entry = entries.get(path, {})
        if entry.get("type") != "blob" or entry.get("mode") != "100644":
            raise AdmissionError("candidate file is not a non-executable regular blob")
        blob = _object(
            fetch(
                f"{root}/contents/{urllib.parse.quote(path, safe='/')}?ref={selector.head}"
            ),
            "candidate file",
        )
        if (
            blob.get("type") != "file"
            or blob.get("path") != path
            or blob.get("encoding") != "base64"
            or blob.get("sha") != entry.get("sha")
        ):
            raise AdmissionError("candidate content is not a regular file")
        try:
            raw = base64.b64decode("".join(blob["content"].splitlines()), validate=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise AdmissionError("candidate content encoding is invalid") from exc
        if len(raw) != blob.get("size") or len(raw) > 10 * 1024 * 1024:
            raise AdmissionError("candidate content size differs")
        digests[path] = hashlib.sha256(raw).hexdigest()
        contents[path] = raw
    # Under both admission bases: a candidate without the source pin fails the
    # protected binding check after the signing approval (signing run
    # 45328443323). Refuse it here, before the admission exists.
    if batch:
        # Every unit of a batch carries its own pin, checked against its own
        # citation, in unit order; the refusal names the unit.
        for unit in units:
            try:
                require_candidate_source_pin(
                    contents[unit.rulespec_path], unit.citation
                )
            except AdmissionError as exc:
                error = AdmissionError(
                    f"unit {unit.citation} ({unit.rulespec_path}): {exc}"
                )
                error.detail = getattr(exc, "detail", str(exc))  # type: ignore[attr-defined]
                raise error from exc
    else:
        require_candidate_source_pin(
            contents[selector.rulespec_path], selector.citation
        )

    # Refuse changes during the API sequence rather than carrying an old
    # approval across a new head/base. Before signing and publication, recheck
    # mutable authorization; the signer rehashes local immutable source bytes.
    after = _object(fetch(pr_path), "PR recheck")
    if _pr_identity(after, selector) != (base, author_id):
        raise AdmissionError("candidate PR moved during admission")
    if _pages(fetch, pr_path + "/reviews") != reviews:
        raise AdmissionError("candidate reviews changed during admission")

    def record(unit: PromotionSelector) -> dict:
        return {
            "schema": "axiom-encode/reviewed-rulespec-admission/v1",
            "country": unit.country,
            "rulespec_ref": unit.head,
            "rulespec_pr": unit.pr,
            "pr_base_branch": unit.branch,
            "base_sha": base,
            "binding": {
                "citation": unit.citation,
                "rulespec_path": unit.rulespec_path,
                "rulespec_sha256": digests[unit.rulespec_path],
                "companion_sha256": digests[unit.companion],
            },
            "declared": None,
            "purpose": purpose,
        }

    if batch:
        return {
            "schema": BATCH_SCHEMA,
            "workflow_commit": run["head_sha"],
            "selector": selector.payload(),
            "admission_basis": basis,
            _BASIS_EVIDENCE_FIELD[basis]: evidence,
            "records": [record(unit) for unit in units],
        }
    return {
        "schema": SCHEMA,
        "workflow_commit": run["head_sha"],
        "selector": selector.__dict__,
        "admission_basis": basis,
        _BASIS_EVIDENCE_FIELD[basis]: evidence,
        "record": record(selector),
    }
