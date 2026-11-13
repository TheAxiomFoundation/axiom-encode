"""Internal exact-owner context; recognized shapes with no evidence get no credit."""

from __future__ import annotations

import re
from dataclasses import dataclass

from axiom_encode.student_loan_schedule_source import (
    StudentLoanScheduleSource,
    recognize_student_loan_schedule,
)

from .student_loan_schedule_binding import bind_student_loan_schedule
from .student_loan_schedule_execution import (
    StudentLoanScheduleObservation,
    observe_student_loan_schedule,
)


@dataclass(frozen=True)
class _StudentLoanContext:
    source: StudentLoanScheduleSource
    observation: StudentLoanScheduleObservation | None

    def span(self, role):
        return next(r.span for r in self.source.records if r.role == role)

    def branch_role(self, branch):
        """Exact raw records, plus the known physical-header tail of clause one."""
        if branch.path or branch.text != self.source.source[branch.start : branch.end]:
            return None
        roles = (
            "current_completion",
            "prior_completion",
            "historical_reconstruction",
            "enclosure",
            "guide_referral",
        )
        for role in roles:
            if (branch.start, branch.end) == self.span(role):
                return role
        start, end = self.span("current_completion")
        header = self.source.source[:start]
        # The closed source recognizer fixes this exact physical header syntax.
        # This is an offset within that header, never a free-form prefix match.
        marker = "T P -1. D. M -V ("
        if header.count(marker) == 1:
            tail = header.index(marker) + len("T P -1. D. ")
            if (branch.start, branch.end) == (tail, end):
                return "current_completion"
        return None

    def formula_owned(self, branch):
        return self.branch_role(branch) in {
            "prior_completion",
            "historical_reconstruction",
        }

    def witnesses(self, branch):
        role = self.branch_role(branch)
        observation = self.observation
        if role is None or observation is None:
            return set()
        pairs = {(owner, fact): pair for owner, fact, pair in observation.duty_pairs}
        identity = (observation.content_sha256, observation.source_sha256, role)
        if role == "current_completion":
            proof = (pairs[("completion", "current_paid")],)
        elif role == "prior_completion":
            proof = (
                pairs[("completion", "calculated")],
                pairs[("completion", "decision")],
            )
        elif role == "historical_reconstruction":
            proof = (
                observation.exclusive_provider_cases,
                observation.equal_provider_pair,
            )
        elif role == "enclosure":
            proof = (pairs[("enclosure", "form_use")],)
        else:
            proof = (pairs[("guide", "copy_wish")],)
        return {(identity, proof)}

    def numeric_spans(self):
        if self.observation is None:
            return set()
        permitted = {
            "current_completion": {self.source.year},
            "prior_completion": {self.source.prior_year},
            "historical_reconstruction": {
                self.source.history_cutoff_year,
                self.source.history_first_year,
                self.source.prior_year,
            },
            "entered_prior": {self.source.prior_year},
            "current_payment": {self.source.year},
            "return_claim": {self.source.year},
        }
        result = set()
        for record in self.source.records:
            for value in permitted.get(record.role, ()):
                a, b = record.span
                result.update(
                    (a + m.start(), a + m.end())
                    for m in re.finditer(
                        rf"(?<!\d){value}(?!\d)", self.source.source[a:b]
                    )
                )
        return result


def _student_loan_context(source_text, citation, content, evidence, cases):
    source = recognize_student_loan_schedule(source_text, citation)
    if source is None:
        return None
    binding = (
        bind_student_loan_schedule(source, content)
        if isinstance(content, str)
        else None
    )
    observed = (
        observe_student_loan_schedule(binding, evidence, content, cases)
        if binding is not None
        and evidence is not None
        and isinstance(cases, (list, tuple))
        else None
    )
    return _StudentLoanContext(source, observed)
