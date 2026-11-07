"""Closed dispatch for privately registered, source-bound execution owners."""


def adapter(capture):
    from . import capital_gift_capture, investment_adjustment_capture, mhrtc_capture

    choices = [
        (capital_gift_capture, "_capital_gift"),
        (investment_adjustment_capture, "_investment_adjustment"),
        (mhrtc_capture, "_mhrtc"),
    ]
    owned = [
        pair for pair in choices if getattr(capture, pair[1] + "_owned", False) is True
    ]
    return owned[0] if len(owned) == 1 else None


def eligible_adapters(pipe, content):
    from . import capital_gift_capture, investment_adjustment_capture, mhrtc_capture

    return tuple(
        (module, prefix)
        for module, prefix in (
            (capital_gift_capture, "_capital_gift"),
            (investment_adjustment_capture, "_investment_adjustment"),
            (mhrtc_capture, "_mhrtc"),
        )
        if module.eligible(pipe, content)
    )


def context_ready(capture):
    selected = adapter(capture)
    if selected is None:
        return False
    module, prefix = selected
    return module.valid(getattr(capture, prefix + "_context", None), capture)


def refusal_target(capture, case, case_index):
    selected = adapter(capture)
    if selected is None:
        return None
    module, prefix = selected
    return module.refusal_target(
        getattr(capture, prefix + "_context", None), capture, case, case_index
    )


def observation(evidence, content):
    from . import capital_gift_capture, investment_adjustment_capture, mhrtc_capture

    owners = [
        m
        for m in (capital_gift_capture, investment_adjustment_capture, mhrtc_capture)
        if m.owns_evidence(evidence)
    ]
    return owners[0].observation(evidence, content) if len(owners) == 1 else None


def owns_evidence(evidence):
    from . import capital_gift_capture, investment_adjustment_capture, mhrtc_capture

    return (
        capital_gift_capture.owns_evidence(evidence)
        or investment_adjustment_capture.owns_evidence(evidence)
        or mhrtc_capture.owns_evidence(evidence)
    )


def clear_private(capture):
    # Invalidate every marked owner, including malformed ambiguous captures.
    for prefix in ("_capital_gift", "_investment_adjustment"):
        if getattr(capture, prefix + "_owned", False) is True:
            getattr(capture, prefix + "_refusals").clear()
            setattr(capture, prefix + "_context", None)

    if getattr(capture, "_mhrtc_owned", False) is True:
        from .mhrtc_capture import clear

        clear(capture)
