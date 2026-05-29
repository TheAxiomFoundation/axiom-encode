"""Auto-repair canonical-name violations in *.test.yaml files.

Producer YAML must fail loudly when it uses a blocked synonym or anchors a
canonical at the wrong location — the encoder needs the negative feedback to
learn. But the cases the model invents for `*.test.yaml` files are not part of
the rule logic; they're just example inputs/outputs. We can rewrite those
mechanically against the registry, so a clean encode --apply isn't blocked by
drift that only shows up in test cases.

The rewrite is intentionally surgical: it operates on anchored references of
the form `<jurisdiction>:<path>#[input.]<name>`.

Input refs are special. `<anchor>#input.X` declares an input slot on the
consumer at `<anchor>` — `<anchor>` names the file that *reads* X, not the
canonical producer of X. Rewriting the anchor on an input ref would move the
slot to the producer's file, which is nonsense (a file does not consume its
own output). For input refs we only rename the synonym; the consumer anchor
is preserved. For non-input (output) refs the anchor is the consumer's
reference to a producer, so we also redirect the anchor to the canonical
producer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from .registry import ConceptRegistry

ANCHORED_REF_RE = re.compile(
    r"([a-z][a-z0-9-]*:[A-Za-z0-9_\-/\.]+)#(input\.)?([a-z][a-z0-9_]*)"
)


def auto_repair_test_yaml_canonical_violations(
    yaml_paths: Iterable[Path],
    registry: ConceptRegistry,
) -> list[Path]:
    """Rewrite blocked synonyms and bad anchors in *.test.yaml files in place.

    For every anchored ref `<anchor>#[input.]<name>` found in a test file:
      - If `name` is a blocked synonym, replace `name` with the canonical, and
        replace `anchor` with the canonical's producer_anchor if the registry
        knows one.
      - Else if `name` is a registered canonical at a different anchor,
        replace `anchor` with the canonical's producer_anchor.

    Returns the list of paths that were modified.
    """
    changed: list[Path] = []
    for path in yaml_paths:
        if not path.exists() or not path.name.endswith(".test.yaml"):
            continue
        original = path.read_text()
        rewritten = _rewrite_anchored_refs(original, registry)
        if rewritten != original:
            path.write_text(rewritten)
            changed.append(path)
    return changed


def _rewrite_anchored_refs(text: str, registry: ConceptRegistry) -> str:
    def repl(match: re.Match[str]) -> str:
        anchor, input_prefix, name = (
            match.group(1),
            match.group(2) or "",
            match.group(3),
        )
        is_input_ref = bool(input_prefix)
        blocked = registry.lookup_synonym(name)
        if blocked is not None:
            if is_input_ref and blocked.producer_anchor == anchor:
                return match.group(0)
            new_anchor = anchor if is_input_ref else (blocked.producer_anchor or anchor)
            return f"{new_anchor}#{input_prefix}{blocked.canonical_name}"
        canonical = registry.lookup_canonical(name)
        if (
            canonical is not None
            and canonical.has_producer
            and canonical.producer_anchor != anchor
            and not is_input_ref
        ):
            return f"{canonical.producer_anchor}#{input_prefix}{name}"
        return match.group(0)

    return ANCHORED_REF_RE.sub(repl, text)
