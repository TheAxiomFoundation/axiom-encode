"""Private fresh positive T1170 capture. No refusal or source-coverage grant."""

from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import dataclass

import yaml

from ..corpus_resolver import LocalCorpusRelease, resolve_local_corpus_source
from .bound_capital_gift_operands import GUIDE, PARENT, bind_capital_gift
from .capital_gift_execution import (
    ENGINE_PIN,
    CapitalGiftPositiveObservation,
    _compiled,
    _direct_refusal,
    _positive_coverage,
    _record,
    _unique_json,
    capital_gift_case_key,
    direct_refusal_pairs,
    observe_capital_gift,
)


@dataclass(eq=False, frozen=True)
class _Context:
    release: LocalCorpusRelease
    content: str
    companion: str
    parent: str
    guide: str
    source_identities: tuple
    binding: object
    cases: tuple
    keys: tuple
    artifact: bytes


# Object identity is deliberately not transferable by dataclass replacement,
# caller-supplied equal fields, pickling, or an environment switch.
_CONTEXTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_EVIDENCE: dict[int, tuple] = {}
_CAPTURES: dict[int, tuple] = {}


@dataclass(frozen=True)
class RefusalExecution:
    case_sha256: str
    request_json: str
    returncode: int
    stdout: str
    stderr: str
    case_index: int


@dataclass(frozen=True)
class CapitalGiftDirectObservation:
    positive: CapitalGiftPositiveObservation
    refusals: tuple
    pairs: tuple
    pending_refusal_contract: tuple = (
        "direct_source_role_coverage",
        "provider_refusal_lineage",
    )


def _capture_identity(capture):
    return (
        capture.original_root,
        capture.snapshot_root,
        capture.root_relative,
        tuple((f.relative, f.original, f.raw) for f in capture.files),
        capture.binary,
        capture.binary_sha256,
        capture.engine_pin,
        capture.git_head,
        id(capture.overlay_origin),
    )


def _sources(release):
    if not isinstance(release, LocalCorpusRelease):
        raise ValueError("authenticated named corpus release required")
    rows = tuple(resolve_local_corpus_source(p, release) for p in (PARENT, GUIDE))
    if any(
        r.citation_path != p or r.slice_required
        for p, r in zip((PARENT, GUIDE), rows, strict=True)
    ):
        raise ValueError("exact whole source identities required")
    identities = tuple(
        (
            r.citation_path,
            r.resolved_text_sha256,
            r.provision_file_sha256,
            r.release_name,
            r.release_content_sha256,
            r.release_selector_sha256,
        )
        for r in rows
    )
    return rows[0].body, rows[1].body, identities


def eligible(pipe, content):
    """Only resolve through the pipeline's authenticated release, never its text."""
    try:
        parent, guide, _ = _sources(pipe.local_corpus_release)
        return bind_capital_gift(content, parent, guide) is not None
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def activate(pipe, capture):
    """Materializer-only registration after its full genuine snapshot checks."""
    from .same_case_execution import _materialized_capture

    if (
        id(capture) in _CAPTURES
        or not _materialized_capture(capture)
        or not capture.fresh()
    ):
        return False
    root = next((f for f in capture.files if f.relative == capture.root_relative), None)
    if root is None or not eligible(pipe, root.raw.decode()):
        return False
    key = id(capture)
    _CAPTURES[key] = (
        weakref.ref(capture, lambda _: _CAPTURES.pop(key, None)),
        pipe,
        _capture_identity(capture),
        pipe.local_corpus_release,
    )
    return True


def mint(capture):
    """Called internally after genuine snapshot and compiled-byte checks."""
    from .validator_pipeline import _safe_load_unique_keys

    try:
        entry = _CAPTURES.get(id(capture))
        if entry is None or entry[0]() is not capture:
            return None
        pipe = entry[1]
        if (
            _capture_identity(capture) != entry[2]
            or pipe.local_corpus_release is not entry[3]
        ):
            return None
        if (
            capture.engine_pin != ENGINE_PIN
            or not capture.fresh()
            or capture.artifact is None
        ):
            return None
        root = next(f for f in capture.files if f.relative == capture.root_relative)
        expected_companion = root.relative.removesuffix(".yaml") + ".test.yaml"
        if {f.relative for f in capture.files} != {root.relative, expected_companion}:
            return None
        companion_file = next(
            f for f in capture.files if f.relative == expected_companion
        )
        content, companion = root.raw.decode(), companion_file.raw.decode()
        parent, guide, identities = _sources(pipe.local_corpus_release)
        binding = bind_capital_gift(content, parent, guide)
        if binding is None:
            return None
        raw = _safe_load_unique_keys(companion)
        cases = raw["cases"] if type(raw) is dict and set(raw) == {"cases"} else raw
        if type(cases) is not list or not cases:
            return None
        artifact = capture.artifact.read_bytes()
        parsed = json.loads(artifact, object_pairs_hook=_unique_json)
        _compiled(binding, parsed, capture.engine_pin)
        from .bound_capital_gift_operands import _load

        if (
            parsed["program"]["module"]["summary"]
            != _load(content)["module"]["summary"]
        ):
            return None
        keys = tuple(capital_gift_case_key(c, binding) for c in cases)
        if len(set(keys)) != len(keys):
            return None
        context = _Context(
            pipe.local_corpus_release,
            content,
            companion,
            parent,
            guide,
            identities,
            binding,
            tuple(cases),
            keys,
            artifact,
        )
        _CONTEXTS[context] = weakref.ref(capture)
        return context
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        StopIteration,
        RecursionError,
        yaml.YAMLError,
    ):
        return None


def valid(context, capture):
    try:
        if capture._capital_gift_context is not context:
            return False
        entry = _CAPTURES.get(id(capture))
        if (
            entry is None
            or entry[0]() is not capture
            or _capture_identity(capture) != entry[2]
        ):
            return False
        if (
            entry[1].local_corpus_release is not entry[3]
            or context.release is not entry[3]
        ):
            return False
        owner = _CONTEXTS.get(context)
        if owner is None or owner() is not capture or not capture.fresh():
            return False
        parent, guide, identities = _sources(context.release)
        files = {f.relative: f for f in capture.files}
        return (
            (parent, guide, identities)
            == (context.parent, context.guide, context.source_identities)
            and context.binding.bound(context.content, parent, guide)
            and len(files) == 2
            and files[capture.root_relative].raw.decode() == context.content
            and files[
                capture.root_relative.removesuffix(".yaml") + ".test.yaml"
            ].raw.decode()
            == context.companion
            and capture.artifact.read_bytes() == context.artifact
            and capture.engine_pin == ENGINE_PIN
        )
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return False


def stage(context, capture, case, request, response, case_index):
    from .same_case_execution import CaseExecution

    if (
        not valid(context, capture)
        or type(case_index) is not int
        or not 1 <= case_index <= len(context.cases)
    ):
        return False
    try:
        key = capital_gift_case_key(case, context.binding)
        if (
            "expected_refusal" in case
            or key != context.keys[case_index - 1]
            or any(r.case_index == case_index for r in capture.records)
        ):
            return False
        artifact = json.loads(context.artifact, object_pairs_hook=_unique_json)
        derived, parameters = _compiled(context.binding, artifact, capture.engine_pin)
        _record(
            context.binding,
            derived,
            parameters,
            context.cases[case_index - 1],
            {
                "request": request,
                "stdout": json.dumps(response),
                "stderr": "",
                "returncode": 0,
            },
        )
        capture.records.append(
            CaseExecution(
                key,
                json.dumps(request, sort_keys=True),
                json.dumps(response, sort_keys=True),
                case_index,
            )
        )
        return True
    except (ValueError, TypeError, KeyError, AttributeError, ArithmeticError):
        return False


def refusal_target(context, capture, case, case_index):
    """Resolve a query without fabricated expected values, before execution."""
    try:
        if (
            not valid(context, capture)
            or type(case_index) is not int
            or not 1 <= case_index <= len(context.cases)
            or capital_gift_case_key(case, context.binding)
            != context.keys[case_index - 1]
            or set(case) != {"name", "entity_id", "period", "input", "expected_refusal"}
        ):
            return None
        contract = case["expected_refusal"]
        if (
            type(contract) is not dict
            or set(contract) != {"output", "reason"}
            or contract["reason"] != "NoMatchingArm"
        ):
            return None
        from .capital_gift_execution import PREFIX

        allowed = {
            PREFIX + "federal_2025_t1170_gift_capital_gain_column_5",
            PREFIX + "t1170_prior_exchange_gain_at_zero_inclusion",
            PREFIX + "t1170_partnership_special_capital_gain_per_entry",
        }
        return contract["output"] if contract["output"] in allowed else None
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def stage_refusal(
    context, capture, case, request, returncode, stdout, stderr, case_index
):
    if refusal_target(context, capture, case, case_index) is None or any(
        r.case_index == case_index
        for r in (*capture.records, *capture._capital_gift_refusals)
    ):
        return False
    try:
        _direct_refusal(
            context.binding,
            json.loads(context.artifact),
            ENGINE_PIN,
            case,
            {
                "request": request,
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
        )
        capture._capital_gift_refusals.append(
            RefusalExecution(
                context.keys[case_index - 1],
                json.dumps(request, sort_keys=True),
                returncode,
                stdout,
                stderr,
                case_index,
            )
        )
        return True
    except (ValueError, TypeError, KeyError, AttributeError, ArithmeticError):
        return False


def register_evidence(context, capture, evidence):
    if not valid(context, capture):
        return False
    key = id(evidence)
    _EVIDENCE[key] = (
        weakref.ref(evidence, lambda _: _EVIDENCE.pop(key, None)),
        context,
        capture,
        tuple(capture._capital_gift_refusals),
    )
    return True


def observation(evidence, content):
    """Actual consumer entry: equal caller-created evidence has no authority."""
    try:
        entry = _EVIDENCE.get(id(evidence))
        if entry is None or entry[0]() is not evidence:
            return None
        context = entry[1]
        capture = entry[2]
        if capture is None or not valid(context, capture):
            return None
        if (
            context not in _CONTEXTS
            or content != context.content
            or not evidence.bound(content, ())
        ):
            return None
        if (
            evidence.engine_pin != ENGINE_PIN
            or evidence.artifact_bytes != context.artifact
        ):
            return None
        if _sources(context.release) != (
            context.parent,
            context.guide,
            context.source_identities,
        ):
            return None
        files = {f.relative: f.raw for f in evidence.files}
        if files != {
            evidence.root_relative: content.encode(),
            evidence.root_relative.removesuffix(".yaml")
            + ".test.yaml": context.companion.encode(),
        }:
            return None
        records = evidence.records
        if any("expected_refusal" in case for case in context.cases):
            return _mixed_observation(context, records, entry[3])
        if len(records) != len(context.cases) or [
            r.case_index for r in records
        ] != list(range(1, len(records) + 1)):
            return None
        if tuple(r.case_sha256 for r in records) != context.keys:
            return None
        calls = [
            {
                "request": json.loads(r.request_json),
                "stdout": r.response_json,
                "stderr": "",
                "returncode": 0,
            }
            for r in records
        ]
        return observe_capital_gift(
            context.binding,
            content,
            context.parent,
            context.guide,
            json.loads(context.artifact),
            calls,
            context.companion,
            ENGINE_PIN,
        )
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        RecursionError,
        ArithmeticError,
    ):
        return None


def _mixed_observation(context, records, refusal_records):
    """Every raw ordinal belongs to exactly one independently checked partition."""
    positive_indices = [
        i for i, c in enumerate(context.cases, 1) if "expected_refusal" not in c
    ]
    negative_indices = [
        i for i, c in enumerate(context.cases, 1) if "expected_refusal" in c
    ]
    if [r.case_index for r in records] != positive_indices or [
        r.case_index for r in refusal_records
    ] != negative_indices:
        return None
    if any(
        r.case_sha256 != context.keys[r.case_index - 1]
        for r in (*records, *refusal_records)
    ):
        return None
    artifact = json.loads(context.artifact)
    derived, parameters = _compiled(context.binding, artifact, ENGINE_PIN)
    positive_cases = [context.cases[i - 1] for i in positive_indices]
    positive_records = tuple(
        _record(
            context.binding,
            derived,
            parameters,
            case,
            {
                "request": json.loads(r.request_json),
                "stdout": r.response_json,
                "stderr": "",
                "returncode": 0,
            },
        )
        for case, r in zip(positive_cases, records, strict=True)
    )
    positive_pairs = _positive_coverage(
        context.binding, positive_records, positive_cases
    )
    negative_cases = [context.cases[i - 1] for i in negative_indices]
    negatives = tuple(
        _direct_refusal(
            context.binding,
            artifact,
            ENGINE_PIN,
            case,
            {
                "request": json.loads(r.request_json),
                "stdout": r.stdout,
                "stderr": r.stderr,
                "returncode": r.returncode,
            },
        )
        for case, r in zip(negative_cases, refusal_records, strict=True)
    )
    pairs = direct_refusal_pairs(
        context.binding, positive_records, positive_cases, negatives, negative_cases
    )
    positive = CapitalGiftPositiveObservation(
        context.binding.content_sha256,
        hashlib.sha256(
            json.dumps(artifact, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
        hashlib.sha256(context.companion.encode()).hexdigest(),
        positive_records,
        positive_pairs,
    )
    return CapitalGiftDirectObservation(positive, negatives, pairs)


def owns_evidence(evidence):
    entry = _EVIDENCE.get(id(evidence))
    return entry is not None and entry[0]() is evidence
