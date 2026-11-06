"""Private Schedule N capture. Source authority requires complete observation."""

from __future__ import annotations

import hashlib
import json
import weakref
from dataclasses import dataclass

import yaml

from ..corpus_resolver import LocalCorpusRelease, resolve_local_corpus_source
from ..yaml_serialization import safe_dump_yaml
from .bound_investment_adjustment_operands import GUIDE, bind_investment_adjustment
from .bound_investment_adjustment_structure import CITATION as PARENT
from .investment_adjustment_coverage import observe_investment_adjustment_coverage
from .investment_adjustment_execution import (
    ENGINE_PIN,
    verify_investment_adjustment_compiled,
)
from .investment_adjustment_numeric import investment_adjustment_numeric_owners
from .investment_adjustment_records import (
    _unique_json,
    observe_investment_adjustment_record,
)
from .investment_adjustment_refusal import (
    TARGET,
    investment_adjustment_refusal_case,
    observe_investment_adjustment_refusal,
    pair_investment_adjustment_refusal,
)


def _case_key(case):
    if type(case) is not dict:
        raise ValueError("raw case required")
    if "expected_refusal" in case:
        investment_adjustment_refusal_case(case)
    elif set(case) != {"name", "entity_id", "period", "input", "output"}:
        raise ValueError("closed positive case required")
    return hashlib.sha256(
        json.dumps(case, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


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
class InvestmentAdjustmentObservation:
    positive: object
    coverage: object
    numeric_owners: tuple
    refusals: tuple
    pairs: tuple


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
        return bind_investment_adjustment(content, parent, guide) is not None
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
        binding = bind_investment_adjustment(content, parent, guide)
        if binding is None:
            return None
        raw = _safe_load_unique_keys(companion)
        cases = raw["cases"] if type(raw) is dict and set(raw) == {"cases"} else raw
        if type(cases) is not list or not cases:
            return None
        artifact = capture.artifact.read_bytes()
        parsed = json.loads(artifact, object_pairs_hook=_unique_json)
        verify_investment_adjustment_compiled(
            binding, content, parent, guide, parsed, capture.engine_pin
        )
        keys = tuple(_case_key(c) for c in cases)
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
        if capture._investment_adjustment_context is not context:
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
        key = _case_key(case)
        if (
            "expected_refusal" in case
            or key != context.keys[case_index - 1]
            or any(r.case_index == case_index for r in capture.records)
        ):
            return False
        artifact = json.loads(context.artifact, object_pairs_hook=_unique_json)
        observe_investment_adjustment_record(
            context.binding,
            context.content,
            context.parent,
            context.guide,
            artifact,
            capture.engine_pin,
            context.cases[case_index - 1],
            {
                "request": request,
                "stdout": json.dumps(response),
                "stderr": "",
                "returncode": 0,
            },
            caller="pipeline",  # Both scoped callers use this exact explicit-root wire profile.
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
            or _case_key(case) != context.keys[case_index - 1]
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
        return TARGET if contract["output"] == TARGET else None

    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def stage_refusal(
    context, capture, case, request, returncode, stdout, stderr, case_index
):
    if refusal_target(context, capture, case, case_index) is None or any(
        r.case_index == case_index
        for r in (*capture.records, *capture._investment_adjustment_refusals)
    ):
        return False
    try:
        observe_investment_adjustment_refusal(
            context.binding,
            context.content,
            context.parent,
            context.guide,
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
        capture._investment_adjustment_refusals.append(
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
        tuple(capture._investment_adjustment_refusals),
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
        return _mixed_observation(context, evidence.records, entry[3])

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
    from .investment_adjustment_suite import observe_investment_adjustment_suite

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
    artifact = json.loads(context.artifact, object_pairs_hook=_unique_json)
    positive_cases = [context.cases[i - 1] for i in positive_indices]
    positive_calls = [
        {
            "request": json.loads(r.request_json),
            "stdout": r.response_json,
            "stderr": "",
            "returncode": 0,
        }
        for r in records
    ]
    # This projection is internal and every ordinal was checked above against
    # the immutable full raw companion. It is never a replacement signed file.
    companion = safe_dump_yaml(positive_cases, sort_keys=False)
    args = (
        context.binding,
        context.content,
        context.parent,
        context.guide,
        artifact,
        ENGINE_PIN,
    )
    positive = observe_investment_adjustment_suite(
        *args, companion, positive_calls, caller="pipeline"
    )
    coverage = observe_investment_adjustment_coverage(
        *args, [{"companion": companion, "calls": positive_calls}], caller="pipeline"
    )
    negatives = tuple(
        observe_investment_adjustment_refusal(
            *args,
            context.cases[i - 1],
            {
                "request": json.loads(r.request_json),
                "stdout": r.stdout,
                "stderr": r.stderr,
                "returncode": r.returncode,
            },
        )
        for i, r in zip(negative_indices, refusal_records, strict=True)
    )
    pairs = tuple(
        pair
        for negative in negatives
        for pair in pair_investment_adjustment_refusal(positive.records, negative)
    )
    numeric = investment_adjustment_numeric_owners(
        context.binding, context.content, context.parent, context.guide
    )
    return InvestmentAdjustmentObservation(
        positive, coverage, numeric, negatives, pairs
    )


def owns_evidence(evidence):
    entry = _EVIDENCE.get(id(evidence))
    return entry is not None and entry[0]() is evidence


def source_binding(evidence, content, source, raw_cases):
    """Return authenticated inputs only after a fresh whole-companion observation."""
    try:
        observed = observation(evidence, content)
        if observed is None:
            return None
        entry = _EVIDENCE[id(evidence)]
        context = entry[1]
        if (
            source != context.parent
            or not isinstance(raw_cases, (list, tuple))
            or tuple(_case_key(case) for case in raw_cases) != context.keys
        ):
            return None
        return context, observed
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def source_binding_live(evidence, context):
    """Cheap lifecycle recheck for an analysis-local, already observed context."""
    entry = _EVIDENCE.get(id(evidence))
    return (
        entry is not None
        and entry[0]() is evidence
        and entry[1] is context
        and valid(context, entry[2])
        and evidence.bound(context.content, ())
    )
