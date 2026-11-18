"""Map a citation to its RuleSpec artifact path, with the standard library only.

``citation_rulespec_path`` is the one path rule that the signed-apply
helpers, the eval harness and the content-bound merge guard share. It lives
here, apart from ``harness/evals.py`` and ``prepare_signed_backfill.py``, so
that the guard can resolve the primary path of a promotion without importing
either of them. This is step 1 of the standard-library guard bundle in the
guard key-protection proposal (axiom-encode#2257; section 4 of the proposal).

Keep this module standard-library only, at module and function level. Its
one import from this package is ``axiom_encode.statute``, which is also
standard-library only. ``tests/test_rulespec_citation_paths.py`` pins both.

``harness/evals.py`` re-exports the helpers it used to define, so existing
callers keep their imports.

``prepare_signed_backfill.py`` does not import this module. It keeps its own
``RULESPEC_ATOMIC_ROOTS``, ``_citation_rulespec_path`` and
``citation_rulespec_path``, and this module holds a copy of each. The
signing workflow loads that file by path from the checkout:
first on the runner's bare interpreter, before the package is installed, and
later on the provisioned verification interpreter, whose ``axiom_encode``
package is a separate install. So that file stays as it is, and
``tests/test_rulespec_citation_paths.py`` fails if the two copies differ.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Callable

from axiom_encode.statute import (
    citation_to_citation_path,
    normalize_rulespec_path_segment,
)

RULESPEC_ATOMIC_ROOTS = frozenset(
    {"legislation", "policies", "regulations", "statutes"}
)
_RULESPEC_SOURCE_ROOT_TOKENS = {
    "form",
    "forms",
    "guidance",
    "manual",
    "manuals",
    "policies",
    "policy",
    "regulation",
    "regulations",
    "statute",
    "statutes",
}
_RULESPEC_OUTPUT_ROOT_BY_SOURCE_TOKEN = {
    "form": "policies",
    "forms": "policies",
    "guidance": "policies",
    "manual": "policies",
    "manuals": "policies",
    "policies": "policies",
    "policy": "policies",
    "regulation": "regulations",
    "regulations": "regulations",
    "statute": "statutes",
    "statutes": "statutes",
}
# Federal Register instruments live under <jurisdiction>/rulemaking/... in the
# corpus. RuleSpec has no rulemaking filesystem root, so their outputs land in
# a reserved ``policies/rulemaking/`` namespace. A dot-free tail maps back to
# the same citation; a dotted leaf is written nested and maps back to the
# nested path (cli._relative_output_to_child_corpus_citation_path,
# validator_pipeline._corpus_citation_to_normalized_target and
# dependency_stubs._corpus_kind_and_rest keep the same contract).
_RULEMAKING_SOURCE_TOKEN = "rulemaking"
_RULEMAKING_OUTPUT_ROOT = "policies"
_UK_LEGISLATION_DOMAIN_TOKEN = "legislation.gov.uk"
_UK_LEGISLATION_SECTION_TOKENS = {"article", "regulation", "section"}
_CFR_CITATION_RE = re.compile(
    r"^(?P<title>\d+)\s+C\.?\s*F\.?\s*R\.?\s+"
    r"(?:(?:part|pt\.?)\s+)?(?:§+\s*)?"
    r"(?P<section>[0-9A-Za-z.-]+)"
    r"(?P<tail>(?:\([^)]+\))*)$",
    re.IGNORECASE,
)
_USC_CITATION_RE = re.compile(
    r"^\d+\s+U\.?\s*S\.?\s*C\.?\s+(?:§+\s*)?"
    r"[0-9A-Za-z.-]+(?:\([^)]+\))*$",
    re.IGNORECASE,
)


def _citation_to_corpus_citation_path(citation: str) -> str:
    """Convert a bare legal citation to a corpus citation path."""
    cfr_path = _cfr_citation_to_corpus_citation_path(citation)
    if cfr_path is not None:
        return cfr_path
    if re.search(r"\bC\.?\s*F\.?\s*R\.?\b", citation, flags=re.IGNORECASE):
        raise ValueError(f"Could not parse CFR citation: {citation}")
    if _USC_CITATION_RE.fullmatch(citation.strip()) is None:
        raise ValueError(f"Could not parse USC citation: {citation}")
    return citation_to_citation_path(citation)


def _cfr_citation_to_corpus_citation_path(citation: str) -> str | None:
    match = _CFR_CITATION_RE.match(citation.strip().replace("§", "§ "))
    if match is None:
        return None
    section_parts = [part for part in match.group("section").split(".") if part]
    if not section_parts:
        return None
    fragments = re.findall(r"\(([^)]+)\)", match.group("tail"))
    return "/".join(
        (
            "us",
            "regulation",
            match.group("title"),
            *section_parts,
            *fragments,
        )
    )


def _looks_like_corpus_citation_path(identifier: str) -> bool:
    parts = [part for part in identifier.strip().strip("/").split("/") if part]
    if parts and ":" in parts[0]:
        _jurisdiction, source_root = parts[0].split(":", 1)
        if source_root in _RULESPEC_SOURCE_ROOT_TOKENS:
            return True
    if len(parts) >= 2 and parts[0] in _RULESPEC_SOURCE_ROOT_TOKENS:
        return True
    return len(parts) >= 3 and parts[1] in {
        "form",
        "forms",
        "guidance",
        "manual",
        "manuals",
        "policies",
        "policy",
        "regulation",
        "regulations",
        # Federal Register instruments (proclamations, executive orders,
        # notices) live under us/rulemaking/... in the corpus; without this
        # token the CLI mangles such citations via citation_to_citation_path
        # into us/statute/us/rulemaking/... and the provisions lookup fails.
        # Validate-time source verification already resolves these paths
        # verbatim.
        "rulemaking",
        "statute",
        "statutes",
    }


def _resolve_eval_output_path(
    citation: str,
    *,
    fallback: Callable[[str], Path] | None = None,
) -> Path:
    """Resolve the rulespec-relative output path for an encode target.

    Corpus-backed source evals pass their canonical requested citation path.
    Citation evals may additionally pass a strict citation parser as ``fallback``.
    Arbitrary free-text identifiers are not translated into source identities.
    """
    if _looks_like_corpus_citation_path(citation):
        return _source_identifier_to_relative_rulespec_path(citation)
    try:
        return _source_identifier_to_relative_rulespec_path(
            _citation_to_corpus_citation_path(citation)
        )
    except ValueError:
        pass
    if fallback is not None:
        return fallback(citation)
    raise ValueError(
        f"Eval source identity is not a canonical citation path: {citation}"
    )


def _source_identifier_to_relative_rulespec_path(source_id: str) -> Path:
    """Map a canonical source identifier to its RuleSpec artifact path.

    See ``_mapped_relative_rulespec_path``. Only a rulemaking source may land
    in the reserved ``policies/rulemaking/`` namespace: a corpus
    ``<j>/rulemaking/...`` path, or the output's own RuleSpec name
    ``[<j>:]policies/rulemaking/...``. Anything else that maps there (a policy,
    form, guidance or manual source whose tail or dotted leaf starts with
    ``rulemaking``, in any letter case) would overwrite a rulemaking output
    and is refused.
    """
    relative = _mapped_relative_rulespec_path(source_id)
    out = relative.parts
    if (
        len(out) >= 3
        and out[0] == _RULEMAKING_OUTPUT_ROOT
        and out[1].casefold() == _RULEMAKING_SOURCE_TOKEN
        and not _names_rulemaking_output(source_id)
    ):
        raise ValueError(
            "Source collides with the reserved rulemaking output namespace: "
            f"{source_id!r}"
        )
    return relative


def _names_rulemaking_output(source_id: str) -> bool:
    """Whether ``source_id`` is a rulemaking source or its output's own name."""
    parts = [part for part in source_id.strip().strip("/").split("/") if part]
    if len(parts) < 3 or parts[1] != _RULEMAKING_SOURCE_TOKEN:
        return False
    head = parts[0]
    if ":" in head:
        # RuleSpec form: only the policies root names the rulemaking output.
        return head.split(":", 1)[1] in {"policies", "policy"}
    if head in _RULESPEC_SOURCE_ROOT_TOKENS:
        return head in {"policies", "policy"}
    # Corpus form <jurisdiction>/rulemaking/...
    return True


def _mapped_relative_rulespec_path(source_id: str) -> Path:
    """Map a canonical source identifier to its RuleSpec artifact path.

    Unicode dash punctuation is normalized only in the returned filesystem
    path; the authoritative source identifier remains unchanged. Each path
    segment is treated as a directory and a dot inside the leaf segment is
    treated as a further nesting separator (CDSS-style numbering like
    ``63-503.132`` becomes ``63-503/132``). This avoids the pathlib
    ``with_suffix`` pitfall where a dotted leaf would be silently truncated
    to a section-level path and collide with sibling subsections at apply
    time (issue #71).
    """
    parts = [
        normalize_rulespec_path_segment(part)
        for part in source_id.strip().strip("/").split("/")
        if part
    ]
    if any(part in {".", ".."} for part in parts):
        raise ValueError(
            f"Unsafe RuleSpec path component in source identifier: {source_id!r}"
        )
    if parts and ":" in parts[0]:
        jurisdiction, source_root = parts[0].split(":", 1)
        if source_root in _RULESPEC_SOURCE_ROOT_TOKENS:
            tail = parts[1:]
            root = _RULESPEC_OUTPUT_ROOT_BY_SOURCE_TOKEN.get(source_root, source_root)
            tail = _expand_louisiana_title_section_tail(jurisdiction, root, tail)
            if (
                tail
                and jurisdiction == "us"
                and source_root
                in {
                    "regulation",
                    "regulations",
                }
            ):
                tail = _canonical_us_regulation_tail(tail)
            if tail and jurisdiction == "uk":
                tail = _canonical_uk_legislation_tail(tail)
            if tail:
                if _preserve_state_statute_dotted_leaf(jurisdiction, root, tail):
                    return Path(root) / Path(*tail[:-1]) / f"{tail[-1]}.yaml"
                return Path(root) / _dotted_leaf_to_nested_yaml_path(tail)
    if len(parts) >= 2 and parts[0] in _RULESPEC_SOURCE_ROOT_TOKENS:
        tail = parts[1:]
        if tail:
            root = _RULESPEC_OUTPUT_ROOT_BY_SOURCE_TOKEN.get(parts[0], parts[0])
            if tail and tail[0] == _UK_LEGISLATION_DOMAIN_TOKEN:
                tail = _canonical_uk_legislation_tail(tail)
            return Path(root) / _dotted_leaf_to_nested_yaml_path(tail)
    if len(parts) >= 3 and parts[1] == _RULEMAKING_SOURCE_TOKEN:
        return Path(_RULEMAKING_OUTPUT_ROOT) / _dotted_leaf_to_nested_yaml_path(
            [_RULEMAKING_SOURCE_TOKEN, *parts[2:]]
        )
    if len(parts) >= 3:
        root = _RULESPEC_OUTPUT_ROOT_BY_SOURCE_TOKEN.get(parts[1])
        if root is not None:
            tail = parts[2:]
            tail = _expand_louisiana_title_section_tail(parts[0], root, tail)
            if parts[0] == "us" and parts[1] in {"regulation", "regulations"}:
                tail = _canonical_us_regulation_tail(tail)
            if parts[0] == "uk":
                tail = _canonical_uk_legislation_tail(tail)
            if tail:
                if _preserve_state_statute_dotted_leaf(parts[0], root, tail):
                    return Path(root) / Path(*tail[:-1]) / f"{tail[-1]}.yaml"
                return Path(root) / _dotted_leaf_to_nested_yaml_path(tail)
    raise ValueError(f"Unsupported canonical source identifier: {source_id!r}")


def _expand_louisiana_title_section_tail(
    jurisdiction: str,
    root: str,
    tail: list[str],
) -> list[str]:
    """Expand Louisiana statute ``title:section`` labels into path components."""

    if jurisdiction != "us-la" or root != "statutes" or not tail:
        return tail
    if any(
        not component
        or component.startswith(".")
        or component.endswith(".")
        or ".." in component
        for component in tail
    ):
        raise ValueError(f"Invalid Louisiana statute path: {tail!r}")
    title_section = tail[0]
    if any(":" in component for component in tail[1:]):
        raise ValueError(f"Invalid Louisiana statute path: {tail!r}")
    if ":" not in title_section:
        return tail
    components = title_section.split(":")
    expanded = [*components, *tail[1:]]
    if len(components) != 2 or any(
        not component
        or component.startswith(".")
        or component.endswith(".")
        or ".." in component
        for component in expanded
    ):
        raise ValueError(
            f"Invalid Louisiana title:section statute segment: {title_section!r}"
        )
    return expanded


def _dotted_leaf_to_nested_yaml_path(tail: list[str]) -> Path:
    """Join tail segments, splitting the leaf on ``.`` into further nesting.

    ``["mpp", "63-503"]`` → ``mpp/63-503.yaml``
    ``["mpp", "63-503.132"]`` → ``mpp/63-503/132.yaml``
    ``["mpp", "63-503.131.a"]`` → ``mpp/63-503/131/a.yaml``
    ``["10-ccr-2506-1", "4.207.2"]`` → ``10-ccr-2506-1/4.207.2.yaml``
    """
    if _preserve_dotted_leaf_filename(tail):
        return Path(*tail[:-1]) / f"{tail[-1]}.yaml"
    leaf_segments = tail[-1].split(".")
    directory_parts = list(tail[:-1]) + leaf_segments[:-1]
    leaf_file = f"{leaf_segments[-1]}.yaml"
    if directory_parts:
        return Path(*directory_parts) / leaf_file
    return Path(leaf_file)


def _preserve_dotted_leaf_filename(tail: list[str]) -> bool:
    """Return true for source families whose dotted section is the file stem."""
    if len(tail) < 2:
        return False
    return bool(
        re.fullmatch(r"\d+-ccr-\d+-\d+", tail[0], flags=re.IGNORECASE)
        and re.fullmatch(r"\d+(?:\.\d+)+", tail[-1])
    )


def _preserve_state_statute_dotted_leaf(
    jurisdiction: str,
    root: str,
    tail: list[str],
) -> bool:
    """Return true when a dotted state statute segment is a legal citation label."""
    if jurisdiction != "us-co" or root != "statutes" or len(tail) < 2:
        return False
    if len(tail) == 2 and tail[0].isdigit():
        # C.R.S. section labels are structured by hyphens as
        # title-article-section. Dots are part of any one of those three
        # segments, never subsection separators.
        crs_segments = tail[-1].split("-")
        return len(crs_segments) == 3 and all(
            re.fullmatch(r"\d+(?:\.\d+)*", segment) for segment in crs_segments
        )
    if not re.fullmatch(r"\d+(?:\.\d+)+", tail[-1]):
        return False
    return bool(re.fullmatch(r"\d+(?:-\d+)+(?:\.\d+)?", tail[-2]))


def _canonical_us_regulation_tail(tail: list[str]) -> list[str]:
    """Map federal regulation corpus paths to canonical RuleSpec repo paths."""
    if not tail:
        return tail
    title = tail[0].strip()
    if title.isdigit():
        return [f"{title}-cfr", *tail[1:]]
    return tail


def _canonical_uk_legislation_tail(tail: list[str]) -> list[str]:
    """Map official legislation.gov.uk corpus paths to canonical UK RuleSpec paths."""
    normalized = list(tail)
    if normalized and normalized[0] == _UK_LEGISLATION_DOMAIN_TOKEN:
        normalized = normalized[1:]
    if (
        len(normalized) >= 2
        and normalized[-2].lower() in _UK_LEGISLATION_SECTION_TOKENS
    ):
        normalized = [*normalized[:-2], normalized[-1]]
    return normalized


def _citation_rulespec_path(citation: str) -> tuple[str, PurePosixPath]:
    jurisdiction, separator, _remainder = citation.partition("/")
    if not separator:
        raise ValueError("citation must be a canonical corpus citation path")
    relative = PurePosixPath(_resolve_eval_output_path(citation).as_posix())
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) < 2
        or relative.parts[0] not in RULESPEC_ATOMIC_ROOTS
        or relative.suffix != ".yaml"
    ):
        raise ValueError("citation does not resolve to a canonical RuleSpec path")
    return jurisdiction, relative


def citation_rulespec_path(citation: str) -> PurePosixPath:
    jurisdiction, relative = _citation_rulespec_path(citation)
    return PurePosixPath(jurisdiction) / relative
