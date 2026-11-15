"""Private whole-form owners; recognition without observation earns no credit."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from axiom_encode.bound_child_credit_form import bind_child_credit_form
from axiom_encode.child_credit_source import recognize_child_credit_form

from .child_credit_execution import observe_child_credit


@dataclass(frozen=True)
class _ChildCreditContext:
    source: object
    binding: object
    observation: object

    def span(self, role):
        return next(r.span for r in self.source.records if r.role == role)

    def owned_spans(self):
        source = self.source.source
        table_a, table_b = self.span("transcriber_table")
        table = source[table_a:table_b]
        d = re.search(
            r"Column D: Column B minus column C \(Maximum \$500 per\r?\nchild\);", table
        )
        f = re.search(r"Column F: Column D\r?\nminus column E\.", table)
        refund_a, refund_b = self.span("refund")
        refund = source[refund_a:refund_b]
        marker = "If the total of these credits"
        table_start, table_end = self.span("table_applicability_and_assistance")
        instruction = source[table_start:table_end]
        end_marker = "physical activity."
        line5_a, line5_b = self.span("line5_sum_destination")
        line5 = source[line5_a:line5_b]
        assert d is not None and f is not None
        return {
            "formula_d": (table_a + d.start(), table_a + d.end()),
            "formula_group": (table_a + f.start(), line5_a + line5.index(".") + 1),
            "refund": (refund_a + refund.index(marker), refund_b),
            "table_completion": (
                self.span("volunteer_heading")[0],
                table_start + instruction.index(end_marker) + len(end_marker),
            ),
            "additional_sheet": self.span("additional_sheet"),
            "volunteer": self.span("volunteer_amount"),
            "paper_attachment": self.span("paper_attachment"),
            "receipts": self.span("receipt_retention_and_non_enclosure"),
        }

    def branch_role(self, branch):
        if branch.path or branch.text != self.source.source[branch.start : branch.end]:
            return None
        for role, span in self.owned_spans().items():
            kind = (
                "formula-clause" if role.startswith("formula_") else "exception-clause"
            )
            if branch.kind == kind and (branch.start, branch.end) == span:
                return role
        return None

    def formula_owned(self, branch):
        return self.branch_role(branch) in {"formula_d", "formula_group"}

    def witnesses(self, branch):
        role = self.branch_role(branch)
        o = self.observation
        if role is None or o is None:
            return set()
        pairs = tuple((owner, fact, a, b) for owner, fact, a, b in o.duty_pairs)
        needed = {
            "volunteer": {"line1"},
            "paper_attachment": {"paper_attachment"},
            "table_completion": {"table_completion"},
            "receipts": {"receipts_retention", "receipts_non_enclosure"},
            "additional_sheet": {"additional_sheet"},
        }
        if role in needed:
            proof = tuple(p for p in pairs if p[0] in needed[role])
        elif role == "refund":
            proof = o.refund_cases
        else:
            proof = (o.per_child_cap_pair, o.provider_instances, o.cases)
        return {((o.content_sha256, o.source_sha256, role), proof)}

    def unmatched(self, branches):
        """Keep multiplicity: identical branch objects cannot collapse as dict keys."""
        used = set()
        missing = []
        for branch in branches:
            available = self.witnesses(branch) - used
            if not available:
                missing.append(branch)
            else:
                used.add(min(available, key=repr))
        return tuple(missing)

    def numeric_spans(self):
        if self.observation is None:
            return set()
        source = self.source.source
        a, b = self.span("transcriber_table")
        spans = {
            (a + m.start(), a + m.end()) for m in re.finditer(r"\bsix\b", source[a:b])
        }
        _, end = self.span("header")
        spans.update(
            (m.start(), m.end())
            for m in re.finditer(rf"\b{self.source.year}\b", source[:end])
        )
        return spans

    def owns_refund_condition(
        self, owner, version, formula, start, end, clause, ambiguous
    ):
        if (
            self.observation is None
            or self.binding is None
            or ambiguous
            or type(version) is not int
            or version != 0
        ):
            return False
        name = dict(self.binding.rule_roles)["conditional_excess"]
        rule = next(
            r
            for r in json.loads(self.binding.declaration_json)["rules"]
            if r["name"] == name
        )
        expected = rule["versions"][0]
        return (
            owner == name
            and formula == expected["formula"]
            and str(start) == expected["effective_from"]
            and str(end) == expected["effective_to"]
            and not clause.branch_path
            and (clause.start, clause.end) == (0, self.span("refund")[1])
            and clause.text == self.source.source[: self.span("refund")[1]]
        )


def _child_credit_context(
    source_text, citation, content, evidence, cases, raw_source=None
):
    if raw_source is not None:
        if not isinstance(raw_source, str) or source_text != raw_source.rstrip():
            return None
        source_text = raw_source
    source = recognize_child_credit_form(source_text, citation)
    if source is None:
        return None
    binding = (
        bind_child_credit_form(content, source_text, citation)
        if isinstance(content, str)
        else None
    )
    observation = (
        observe_child_credit(binding, evidence, content, cases)
        if binding is not None
        and evidence is not None
        and isinstance(cases, (list, tuple))
        else None
    )
    return _ChildCreditContext(source, binding, observation)
