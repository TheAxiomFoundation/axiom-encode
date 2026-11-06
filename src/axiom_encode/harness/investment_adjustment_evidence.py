"""Analysis-local Schedule N ownership; absent evidence never falls back."""

from __future__ import annotations

import weakref
from collections import Counter
from dataclasses import dataclass

from .bound_investment_adjustment_structure import CITATION
from .investment_adjustment_coverage import FORMULAS
from .investment_adjustment_numeric import (
    match_investment_adjustment_typed_numeric_owner,
)

_CONTROL_SPANS = (
    (78, 850),
    (882, 917),
    (1365, 1576),
    (1818, 1853),
    (1885, 1920),
    (2358, 2417),
)
_REGISTERED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_RESERVED: weakref.WeakSet = weakref.WeakSet()
_GATE_SPANS = {
    "adjustment_available_line_76": (1920, 2242),
    "adjustment_carryforward_line_80": (2242, 2357),
    "adjustment_used_line_252": (2242, 2357),
    "copy_guide_reference_applies": (2357, 2505),
    "enclosure_required": (2242, 2357),
    "investment_expense_adjustment_line_260": (850, 1363),
    "investment_expenses_line_18": (917, 1363),
    "investment_income_excess_line_60": (1734, 1853),
    "investment_income_line_36": (77, 850),
    "other_investment_expense_adjustment_line_276": (1853, 2242),
    "other_investment_expenses_line_54": (1363, 2242),
    "resource_deduction_recovery_line_28": (77, 850),
}


@dataclass(eq=False, frozen=True)
class InvestmentAdjustmentEvidence:
    recognized: bool
    source: str
    complete: bool = False

    def live(self):
        from .investment_adjustment_capture import source_binding_live

        entry = _REGISTERED.get(self)
        return bool(
            self.complete
            and entry is not None
            and source_binding_live(entry[0], entry[1])
        )

    def owns_branch(self, branch, kind):
        if self not in _RESERVED or not self.recognized:
            return False
        spans = (
            tuple((a, b) for a, b, _ in FORMULAS)
            if kind == "formula"
            else _CONTROL_SPANS
            if kind == "control"
            else ()
        )
        return (
            type(branch.start) is int
            and type(branch.end) is int
            and (branch.start, branch.end) in spans
            and branch.kind
            == ("formula-clause" if kind == "formula" else "exception-clause")
            and branch.path == ()
            and branch.text == self.source[branch.start : branch.end]
            and branch.required_principal_output is None
            and not branch.computation_chapeaux
            and not branch.structural_numeric_spans
        )

    def covers_branch(self, branch, kind):
        return self.owns_branch(branch, kind) and self.live()

    def partition(self, branches, kind):
        """Reserve owned occurrences before generic fallback, preserving multiplicity."""
        counts = Counter((b.path, b.kind, b.start, b.end, b.text) for b in branches)
        remaining, missing = [], []
        current = self.live()
        for branch in branches:
            if not self.owns_branch(branch, kind):
                remaining.append(branch)
            elif (
                not current
                or counts[
                    (branch.path, branch.kind, branch.start, branch.end, branch.text)
                ]
                != 1
            ):
                missing.append(branch)
        return tuple(remaining), tuple(missing)

    def covers_gate(self, rule_name, rule, index, formula, start, end, clause):
        if not self.live() or type(index) is not int or index != 0:
            return False
        span = _GATE_SPANS.get(rule_name.removeprefix("schedule_n_"))
        if (
            span is None
            or (clause.start, clause.end) != span
            or clause.branch_path != ()
        ):
            return False
        text_start = span[0]
        if span == (77, 850):
            # Exact extractor alias for the two enumerated income/rate owners.
            if self.source[77:78] != " ":
                return False
            text_start = 78
        if clause.text != self.source[text_start : span[1]]:
            return False
        import json

        from .validator_pipeline import _safe_load_unique_keys

        _, context = _REGISTERED[self]
        expected = next(
            (
                r
                for r in _safe_load_unique_keys(context.content)["rules"]
                if r["name"] == rule_name
            ),
            None,
        )
        if expected is None or json.dumps(rule, sort_keys=True) != json.dumps(
            expected, sort_keys=True
        ):
            return False
        version = expected["versions"][0]
        return (formula, start, end) == (version["formula"], "2025-01-01", "2025-12-31")

    def numeric_owner(self, occurrence):
        if not self.live():
            return None
        _, context = _REGISTERED[self]
        return match_investment_adjustment_typed_numeric_owner(
            context.binding, context.content, context.parent, context.guide, occurrence
        )

    def missing(self):
        if self.recognized and not self.live():
            return (
                "[complete-source-unit:schedule-n-evidence] Complete authenticated Schedule N source binding, all principal/condition/boundary witnesses and the Part C refusal pair are required; generic evidence cannot replace a missing or stale private observation.",
            )
        return ()


def certify_investment_adjustment(content, source, citation, raw_cases, evidence):
    """Recognize ownership independently of whether evidence is admissible."""
    from .investment_adjustment_capture import source_binding

    recognized = citation == CITATION
    absent = InvestmentAdjustmentEvidence(recognized, source)
    _RESERVED.add(absent)
    if not recognized:
        return absent
    observed = source_binding(evidence, content, source, raw_cases)
    if observed is None:
        return absent
    context, observation = observed
    if (
        observation.coverage.missing
        or not observation.refusals
        or not observation.pairs
        or {pair[1] for pair in observation.pairs}
        != {refusal.case_sha256 for refusal in observation.refusals}
    ):
        return absent
    # All positive roles, including unextracted document/guide duties, were
    # recomputed internally. The numeric observer itself verifies every exact
    # source occurrence; no caller-supplied count supplies authority.
    result = InvestmentAdjustmentEvidence(True, source, True)
    _RESERVED.add(result)
    _REGISTERED[result] = (evidence, context)
    return result
