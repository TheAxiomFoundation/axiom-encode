"""Private MHRtC snapshot lifecycle and complete source-evidence binding.

The closed dispatcher selects this owner only for its bound source contract.
Source credit requires complete principal, parameter and auxiliary execution,
exact occurrence coverage, and live private ownership. Public bindings and
parsed cases never authorize a context by themselves.
"""

from __future__ import annotations

import subprocess
import weakref
from dataclasses import dataclass

import yaml

from ..corpus_resolver import LocalCorpusRelease, resolve_local_corpus_source
from .bound_mhrtc_program import bind_mhrtc_program
from .mhrtc_execution import ENGINE_PIN, strict_json, verify_mhrtc_compiled
from .mhrtc_program_contract import CITATION, POLICY_PATH
from .mhrtc_request import parse_mhrtc_companion, reconstruct_mhrtc_request
from .mhrtc_results import validate_mhrtc_parameters


@dataclass(eq=False, frozen=True)
class _Context:
    release: LocalCorpusRelease
    content: bytes
    companion: bytes
    source: str
    source_identity: tuple
    artifact: bytes
    rebuilt: bytes
    principal_ordinals: tuple[int, ...]
    parameter_ordinals: tuple[int, ...]


_CAPTURES: dict[int, tuple] = {}
_CONTEXTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_EVIDENCE: dict[int, tuple] = {}


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


def _source(release):
    if not isinstance(release, LocalCorpusRelease):
        raise ValueError("authenticated corpus release required")
    row = resolve_local_corpus_source(CITATION, release)
    if row.citation_path != CITATION or row.slice_required:
        raise ValueError("whole authenticated MHRtC source required")
    return row.body, (
        row.citation_path,
        row.resolved_text_sha256,
        row.provision_file_sha256,
        row.release_name,
        row.release_content_sha256,
        row.release_selector_sha256,
    )


def eligible(pipe, content):
    try:
        source, _ = _source(pipe.local_corpus_release)
        return bind_mhrtc_program(content, source) is not None
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def activate(pipe, capture):
    """Only a genuine fresh materialization may register this private lifecycle."""
    from .same_case_execution import _materialized_capture

    try:
        if (
            id(capture) in _CAPTURES
            or not _materialized_capture(capture)
            or not capture.fresh()
            or capture.root_relative != POLICY_PATH
        ):
            return False
        roots = [f for f in capture.files if f.relative == capture.root_relative]
        if len(roots) != 1 or not eligible(pipe, roots[0].raw.decode("utf-8")):
            return False
        key = id(capture)
        _CAPTURES[key] = (
            weakref.ref(capture, lambda _: _CAPTURES.pop(key, None)),
            pipe,
            _capture_identity(capture),
            pipe.local_corpus_release,
            {},  # Current context generation and retained failure attempt.
        )
        return True
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def _registered(capture):
    from .same_case_execution import _materialized_capture

    entry = _CAPTURES.get(id(capture))
    if (
        entry is None
        or entry[0]() is not capture
        or not _materialized_capture(capture)
        or _capture_identity(capture) != entry[2]
        or entry[1].local_corpus_release is not entry[3]
        or capture.engine_pin != ENGINE_PIN
        or not capture.fresh()
    ):
        raise ValueError("fresh privately registered capture required")
    return entry


def mint(capture):
    """Bind every frozen case and independently recompiled IR before returning."""
    previous = _CAPTURES.get(id(capture))
    if previous is not None and previous[0]() is capture:
        previous[4].clear()
    try:
        entry = _registered(capture)
        pipe = entry[1]
        expected_companion = POLICY_PATH.removesuffix(".yaml") + ".test.yaml"
        files = {f.relative: f for f in capture.files}
        if len(capture.files) != 2 or set(files) != {POLICY_PATH, expected_companion}:
            return None
        content, companion = files[POLICY_PATH].raw, files[expected_companion].raw
        source, identity = _source(entry[3])
        if bind_mhrtc_program(content.decode("utf-8"), source) is None:
            return None
        cases = parse_mhrtc_companion(companion.decode("utf-8"))
        if capture.artifact is None:
            return None
        artifact = capture.artifact.read_bytes()
        rebuilt = verify_mhrtc_compiled(capture, pipe)
        payload = strict_json(rebuilt)
        principal, parameters = [], []
        for ordinal, case in enumerate(cases, 1):
            plan = reconstruct_mhrtc_request(
                case,
                payload,
                pipe=pipe,
                rules_file=capture.snapshot_root / POLICY_PATH,
                case_index=ordinal,
            )
            if plan.request is None:
                if plan.derived_outputs or not plan.parameter_outputs:
                    return None
                validate_mhrtc_parameters(payload, case["output"])
                parameters.append(ordinal)
            else:
                if plan.parameter_outputs or not plan.derived_outputs:
                    return None
                principal.append(ordinal)
        if not principal or not parameters:
            return None
        # Re-read identities after compilation and complete case preflight.
        _registered(capture)
        if capture.artifact.read_bytes() != artifact or _source(entry[3]) != (
            source,
            identity,
        ):
            return None
        context = _Context(
            entry[3],
            content,
            companion,
            source,
            identity,
            artifact,
            rebuilt,
            tuple(principal),
            tuple(parameters),
        )
        _CONTEXTS[context] = weakref.ref(capture)
        entry[4]["context"] = weakref.ref(context)
        return context
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        RecursionError,
        yaml.YAMLError,
    ) as error:
        entry = _CAPTURES.get(id(capture))
        if entry is not None and entry[0]() is capture and hasattr(error, "attempt"):
            entry[4]["failed_attempt"] = error.attempt
        return None


def valid(context, capture):
    """Identity membership, immutable bytes and live provenance must all match."""
    try:
        entry = _registered(capture)
        owner = _CONTEXTS.get(context)
        if (
            owner is None
            or owner() is not capture
            or getattr(capture, "_mhrtc_context", None) is not context
            or context.release is not entry[3]
            or entry[4].get("context", lambda: None)() is not context
        ):
            return False
        files = {f.relative: f for f in capture.files}
        return (
            len(capture.files) == 2
            and files[POLICY_PATH].raw == context.content
            and files[POLICY_PATH.removesuffix(".yaml") + ".test.yaml"].raw
            == context.companion
            and capture.artifact.read_bytes() == context.artifact
            and _source(context.release) == (context.source, context.source_identity)
        )
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        return False


@dataclass(frozen=True)
class _CaseReceipt:
    ordinal: int
    record: object
    principal: object
    auxiliary: tuple


def _auxiliary(capture, pipe, context, case, payload, ordinal):
    from .mhrtc_execution import execute_mhrtc_auxiliary_request
    from .mhrtc_otherclaim_obligations import mhrtc_otherclaim_obligations
    from .mhrtc_otherclaim_request import reconstruct_mhrtc_other_claim_request
    from .mhrtc_request import _PREFIX
    from .mhrtc_results import _number, observe_mhrtc_outputs

    receipts = []
    for target in mhrtc_otherclaim_obligations(
        case,
        payload,
        pipe=pipe,
        rules_file=capture.snapshot_root / POLICY_PATH,
        case_index=ordinal,
    ):
        qualification = reconstruct_mhrtc_other_claim_request(
            case,
            payload,
            pipe=pipe,
            rules_file=capture.snapshot_root / POLICY_PATH,
            case_index=ordinal,
            other_claim_id=target,
            witness_kind="qualification",
        )
        q_receipt = execute_mhrtc_auxiliary_request(
            capture, pipe, qualification.request
        )
        _CAPTURES[id(capture)][4].setdefault("attempts", []).append(q_receipt)
        q = dict(
            observe_mhrtc_outputs(
                payload, qualification.request, strict_json(q_receipt.stdout)
            )
        )
        capacity_receipt = None
        if q[_PREFIX + "other_claim_source_eligibility"] == "holds":
            row = next(r for r in case["tables"]["OtherClaim"] if r["id"] == target)
            actual_claim = _number(row[_PREFIX + "input.other_actual_claimed_expenses"])
            if actual_claim < 0:
                raise ValueError("negative actual other-claim fact")
            if q[_PREFIX + "other_selected_for_individual"] == "not_holds":
                if actual_claim != 0:
                    raise ValueError(
                        "nonselected other claim has positive actual amount"
                    )
            else:
                capacity = reconstruct_mhrtc_other_claim_request(
                    case,
                    payload,
                    pipe=pipe,
                    rules_file=capture.snapshot_root / POLICY_PATH,
                    case_index=ordinal,
                    other_claim_id=target,
                    witness_kind="capacity",
                )
                capacity_receipt = execute_mhrtc_auxiliary_request(
                    capture, pipe, capacity.request
                )
                _CAPTURES[id(capture)][4].setdefault("attempts", []).append(
                    capacity_receipt
                )
                values = dict(
                    observe_mhrtc_outputs(
                        payload, capacity.request, strict_json(capacity_receipt.stdout)
                    )
                )
                limit = values[_PREFIX + "other_allocated_local_capacity"]
                if limit < 0 or actual_claim > limit:
                    raise ValueError(
                        "actual other claim exceeds observed eligible capacity"
                    )
        if not valid(context, capture):
            raise ValueError("capture changed during auxiliary observations")
        receipts.append((target, q_receipt, capacity_receipt))
    return tuple(receipts)


def stage(context, capture, case, request, response, case_index):
    """Stage only a fresh, exact, ordered principal and all its auxiliary receipts."""
    import hashlib
    import json

    from .mhrtc_case_identity import frozen_mhrtc_case
    from .mhrtc_execution import reexecute_mhrtc_case
    from .mhrtc_results import validate_mhrtc_outputs
    from .same_case_execution import CaseExecution

    if not valid(context, capture):
        rejected = _CAPTURES.get(id(capture))
        if (
            rejected is not None
            and rejected[0]() is capture
            and rejected[4].get("context", lambda: None)() is context
        ):
            rejected[4]["failed"] = True
        return False
    entry = _CAPTURES[id(capture)]
    state = entry[4]
    receipts = state.setdefault("receipts", [])
    if state.get("failed"):
        return False
    try:
        if (
            type(case_index) is not int
            or len(receipts) >= len(context.principal_ordinals)
            or case_index != context.principal_ordinals[len(receipts)]
        ):
            raise ValueError("next exact principal ordinal required")
        frozen = frozen_mhrtc_case(context.companion, case_index, case)
        payload = strict_json(context.rebuilt)
        plan = reconstruct_mhrtc_request(
            frozen.case,
            payload,
            pipe=entry[1],
            rules_file=capture.snapshot_root / POLICY_PATH,
            case_index=case_index,
        )
        if plan.request is None or plan.parameter_outputs:
            raise ValueError("principal runtime request required")
        validate_mhrtc_outputs(payload, plan.request, response, frozen.case["output"])
        principal = reexecute_mhrtc_case(
            capture, entry[1], plan.request, request, response
        )
        state.setdefault("attempts", []).append(principal)
        auxiliary = _auxiliary(
            capture, entry[1], context, frozen.case, payload, case_index
        )
        if not valid(context, capture):
            raise ValueError("capture changed during principal staging")
        key = hashlib.sha256(
            f"{frozen.companion_sha256}:{case_index}".encode()
        ).hexdigest()
        record = CaseExecution(
            key,
            json.dumps(request, sort_keys=True, allow_nan=False),
            json.dumps(response, sort_keys=True, allow_nan=False),
            case_index,
        )
        receipts.append(_CaseReceipt(case_index, record, principal, auxiliary))
        capture.records.append(record)
        return True
    except subprocess.SubprocessError as error:
        state["failed"] = True
        state["failed_process_exception"] = error
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        KeyError,
        ArithmeticError,
        yaml.YAMLError,
    ) as error:
        state["failed"] = True
        if hasattr(error, "attempt"):
            state["failed_attempt"] = error.attempt
        return False


def register_evidence(context, capture, evidence):
    """Register complete private execution only; this grants no source coverage."""
    from .same_case_execution import SameCaseEvidence

    if not valid(context, capture) or type(evidence) is not SameCaseEvidence:
        return False
    state = _CAPTURES[id(capture)][4]
    receipts = tuple(state.get("receipts", ()))
    if (
        state.get("failed")
        or tuple(r.ordinal for r in receipts) != context.principal_ordinals
        or tuple(capture.records) != tuple(r.record for r in receipts)
        or evidence.records != tuple(r.record for r in receipts)
        or evidence.files != capture.files
        or evidence.root_relative != capture.root_relative
        or evidence.snapshot_root != capture.snapshot_root
        or evidence.git_head != capture.git_head
        or evidence.binary_sha256 != capture.binary_sha256
        or evidence.engine_pin != capture.engine_pin
        or evidence.artifact_bytes != context.artifact
        or evidence.overlay_origin is not capture.overlay_origin
        or not evidence.bound(context.content.decode("utf-8"), ())
        or not valid(context, capture)
    ):
        return False
    key = id(evidence)
    _EVIDENCE[key] = (
        weakref.ref(evidence, lambda _: _EVIDENCE.pop(key, None)),
        context,
        weakref.ref(capture),
        receipts,
    )
    return True


def owns_evidence(evidence):
    entry = _EVIDENCE.get(id(evidence))
    if entry is None or entry[0]() is not evidence:
        return False
    capture = entry[2]()
    return (
        capture is not None
        and valid(entry[1], capture)
        and not _CAPTURES[id(capture)][4].get("failed")
        and tuple(capture.records) == tuple(r.record for r in entry[3])
    )


@dataclass(frozen=True)
class _ExecutionObservation:
    """Complete private executions only; no source occurrence credit."""

    principal_receipts: tuple
    parameter_ordinals: tuple[int, ...]


def observation(evidence, content):
    """Accept only the live registered complete generation, never a clone."""
    try:
        if not owns_evidence(evidence):
            return None
        entry = _EVIDENCE[id(evidence)]
        context, capture, receipts = entry[1], entry[2](), entry[3]
        if (
            type(content) is not str
            or content.encode("utf-8") != context.content
            or not evidence.bound(content, ())
            or evidence.artifact_bytes != context.artifact
            or evidence.engine_pin != ENGINE_PIN
            or evidence.records != tuple(r.record for r in receipts)
            or tuple(r.ordinal for r in receipts) != context.principal_ordinals
            or not context.parameter_ordinals
            or not valid(context, capture)
        ):
            return None
        return _ExecutionObservation(receipts, context.parameter_ordinals)
    except (OSError, ValueError, TypeError, AttributeError, KeyError, UnicodeError):
        return None


def clear(capture):
    """Invalidate this generation permanently while preserving failure evidence."""
    entry = _CAPTURES.get(id(capture))
    if entry is not None and entry[0]() is capture:
        state = entry[4]
        state.pop("context", None)
        state["failed"] = True
    capture._mhrtc_context = None


def refusal_target(context, capture, case, case_index):
    """No MHRtC runtime-refusal contract is authorized."""
    return None


def stage_refusal(
    context, capture, case, request, returncode, stdout, stderr, case_index
):
    """Eligibility negatives must be executed results, not credited failures."""
    entry = _CAPTURES.get(id(capture))
    if (
        entry is not None
        and entry[0]() is capture
        and entry[4].get("context", lambda: None)() is context
        and _CONTEXTS.get(context, lambda: None)() is capture
    ):
        clear(capture)
    return False


def source_binding(evidence, content, source, raw_cases):
    """Expose the private immutable context only after whole-case observation.

    This is a provenance boundary, not a certificate that source roles have
    witnesses. The source certifier must read exact facts from context.companion.
    """
    from .mhrtc_case_identity import _same_caller_tree

    try:
        observed = observation(evidence, content)
        if observed is None or type(raw_cases) not in (list, tuple):
            return None
        context = _EVIDENCE[id(evidence)][1]
        if type(source) is not str or source != context.source:
            return None
        text = context.companion.decode("utf-8")
        # Reparse immutable bytes strictly first. Ordinary float representation
        # is used for caller identity only, never for numeric computation.
        parse_mhrtc_companion(text)
        ordinary = yaml.safe_load(text)
        if not _same_caller_tree(list(raw_cases), ordinary):
            return None
        if not source_binding_live(evidence, context):
            return None
        return context, observed
    except (OSError, ValueError, TypeError, AttributeError, KeyError, yaml.YAMLError):
        return None


def source_binding_live(evidence, context):
    """Recheck this exact private generation after source witness analysis."""
    entry = _EVIDENCE.get(id(evidence))
    return (
        entry is not None
        and entry[0]() is evidence
        and entry[1] is context
        and owns_evidence(evidence)
        and evidence.bound(context.content.decode("utf-8"), ())
    )


def source_coverage_binding(evidence, content, source, raw_cases):
    """Analyze frozen receipts with private runtime inputs, retaining missing roles.

    This handoff does not grant source credit. A separate certifier must require
    complete coverage and recheck this generation when consuming observations.
    No caller-supplied pipeline, path, report, or coverage flag is accepted.
    """
    from .mhrtc_coverage import MhrtcCoverage, observe_mhrtc_coverage

    try:
        binding = source_binding(evidence, content, source, raw_cases)
        if binding is None:
            return None
        context, observed = binding
        capture = _EVIDENCE[id(evidence)][2]()
        if capture is None:
            return None
        entry = _CAPTURES.get(id(capture))
        if entry is None or entry[0]() is not capture:
            return None
        coverage = observe_mhrtc_coverage(
            context,
            observed.principal_receipts,
            pipe=entry[1],
            rules_file=capture.snapshot_root / POLICY_PATH,
        )
        if type(coverage) is not MhrtcCoverage or not source_binding_live(
            evidence, context
        ):
            return None
        return context, observed, coverage
    except (OSError, ValueError, TypeError, AttributeError, KeyError, yaml.YAMLError):
        return None
