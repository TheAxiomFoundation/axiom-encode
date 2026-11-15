"""Exact source owners backed by complete local and provider execution evidence."""

import json
import re
from dataclasses import dataclass

from axiom_encode.bound_retirement_transfer_form import bind_retirement_transfer_form
from axiom_encode.retirement_transfer_source import (
    recognize_retirement_transfer_schedule,
)

from .retirement_transfer_execution import observe_retirement_transfer


@dataclass(frozen=True)
class _RetirementTransferContext:
    source: object
    binding: object
    observation: object

    def span(self, role):
        return next(r.span for r in self.source.records if r.role == role)

    def owned_spans(self):
        # Discovery starts after the first period in the printed form identifier.
        # This is a closed, recognized header, not a search through arbitrary prose.
        start, end = self.span("header")
        header = self.source.source[start:end]
        permission_start = start + header.index("D.Q-V")
        return {
            "formula_deductions": (
                self.span("deduction12")[0],
                self.span("deductions_total")[1],
            ),
            "formula_selection": self.span("selected_transfer"),
            "permission": (permission_start, self.span("permission")[1]),
            "withholding": (
                self.span("informational_guide")[1] + 1,
                self.span("corresponding_withholding")[1],
            ),
            "compensation_guide": self.span("compensation_guide"),
            "withholding_inclusion": self.span("conditional_rl1_withholding"),
            "enclosure": self.span("transferor_enclosure"),
            "identification": (self.span("spouse_sin")[0], self.span("spouse_name")[1]),
            "recipient_income_duty": self.span("recipient_income_destination"),
            "recipient_withholding_duty": self.span(
                "recipient_withholding_destination"
            ),
            "copy_guide": self.span("copy_guide"),
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
        role = self.branch_role(branch)
        return role is not None and role.startswith("formula_")

    def witnesses(self, branch):
        role = self.branch_role(branch)
        o = self.observation
        if role is None or o is None:
            return set()
        owners = dict(self.binding.rule_roles)
        pairs = o.witness_pairs
        if role == "permission":
            labels = {
                "permission_birth",
                "permission_spouse",
                "permission_date",
                "permission_decision",
            }
            labels.update(
                f"{provider}:{fact}"
                for provider in ("recipient_income", "recipient_withholding")
                for fact in (
                    "transferor_birth_date",
                    "has_selected_spouse_at_reference_date",
                    "spouse_reference_date",
                    "jointly_decided_retirement_income_transfer",
                )
            )
        elif role == "withholding":
            labels = {"withholding_duty"}
            labels.update(
                f"{r}:schedule_q_requested_retirement_income_transfer_line_22"
                for r in (
                    "recipient_income",
                    "recipient_withholding",
                    "recipient_income_duty",
                    "recipient_withholding_duty",
                )
            )
        elif role in {"recipient_income_duty", "recipient_withholding_duty"}:
            labels = {f"{role}:schedule_q_requested_retirement_income_transfer_line_22"}
        elif role.startswith("formula_"):
            needed = {owners["authorized_transfer"]}
            if role == "formula_deductions":
                needed.add(owners["deductions"])
            proof = tuple(e for e in o.executions if needed <= dict(e[3]).keys())
            if not proof:
                return set()
            return {((o.content_sha256, o.source_sha256, role), proof)}
        else:
            labels = {role}
        proof = tuple(p for p in pairs if p[0] in labels)
        if {p[0] for p in proof} != labels:
            return set()
        return {((o.content_sha256, o.source_sha256, role), proof)}

    def unmatched(self, branches):
        used, missing = set(), []
        for branch in branches:
            choices = self.witnesses(branch) - used
            if not choices:
                missing.append(branch)
            else:
                used.add(min(choices, key=repr))
        return tuple(missing)

    def numeric_spans(self):
        if self.observation is None:
            return set()
        # Each semantic number already has an exact role in the closed source
        # and complete typed/ordered graph. Layout locators below are separate.
        spans = {n.span for n in self.source.numbers}
        source = self.source.source
        segments = [
            (self.span("spouse_name")[1], self.span("entered_income")[0]),
            (self.span("deduction12")[1], self.span("deduction13")[0]),
            (self.span("deduction13")[1], self.span("deduction14")[0]),
            (self.span("deduction14")[1], self.span("deduction16")[0]),
            (self.span("deduction16")[1], self.span("deductions_total")[0]),
            (
                self.span("recipient_income_destination")[1],
                self.span("entered_rl2_withholding")[0],
            ),
        ]
        # Footer identifier digits are not row references and earn no credit.
        a = self.span("copy_guide")[1]
        tail = source[a:]
        row_block = re.match(r"\r?\n\r?\n50\r?\n52\r?\n53\r?\n\r?\n58\r?\n", tail)
        if row_block is not None:
            segments.append((a, a + row_block.end()))
        for a, b in segments:
            spans.update(
                (a + m.start(), a + m.end()) for m in re.finditer(r"\d+", source[a:b])
            )
        return spans

    def owns_gate(self, owner, version, formula, start, end, clause, ambiguous):
        if (
            self.observation is None
            or ambiguous
            or type(version) is not int
            or version != 0
            or clause.branch_path
        ):
            return False
        roles = dict(self.binding.rule_roles)
        rule = next(
            (
                r
                for r in json.loads(self.binding.declaration_json)["rules"]
                if r["name"] == owner
            ),
            None,
        )
        if rule is None:
            return False
        v = rule["versions"][0]
        if (
            formula != v.get("formula")
            or str(start) != v["effective_from"]
            or str(end) != v["effective_to"]
        ):
            return False
        if owner in {roles["permission"], roles["authorized_transfer"]}:
            a, b = self.owned_spans()["permission"]
            span = (a - 1, b)
            # Discovery retains the separator after "-1." in its coordinate,
            # but omits that one closed-header space from the clause text.
            expected_text = self.source.source[a:b]
        elif owner == roles["compensation_guide"]:
            span = (
                self.span("transferor_enclosure")[1],
                self.span("compensation_guide")[1],
            )
            expected_text = self.source.source[slice(*span)]
        else:
            return False
        return (clause.start, clause.end) == span and clause.text == expected_text

    def clamp_witnessed(self, owner, rule, input_name):
        if self.observation is None:
            return False
        owners = dict(self.binding.rule_roles)
        declared = next(
            r
            for r in json.loads(self.binding.declaration_json)["rules"]
            if r["name"] == owners["normalized_request"]
        )
        if (
            owner != owners["normalized_request"]
            or rule != declared
            or input_name != "schedule_q_requested_retirement_income_transfer_line_22"
        ):
            return False
        contexts = set()
        for (
            _,
            entity,
            raw_facts,
            raw_outputs,
            _,
            raw_values,
        ) in self.observation.executions:
            facts, outputs, values = (
                dict(raw_facts),
                dict(raw_outputs),
                dict(raw_values),
            )
            for (who, fact), value in facts.items():
                if fact != input_name or value >= 0 or values.get((owner, who)) != 0:
                    continue
                if (
                    who == entity
                    and outputs.get(owner) == 0
                    and outputs.get(owners["authorized_transfer"]) == 0
                ):
                    contexts.add("local")
                elif (
                    who != entity
                    and outputs.get(owners["recipient_income"]) == 0
                    and outputs.get(owners["recipient_withholding"]) == 0
                    and values.get((owners["authorized_transfer"], who)) == 0
                ):
                    contexts.add("recipient")
        return contexts == {"local", "recipient"}


def _retirement_transfer_context(
    source_text, citation, content, evidence, cases, raw_source=None
):
    if raw_source is not None:
        if not isinstance(raw_source, str) or source_text != raw_source.rstrip():
            return None
        source_text = raw_source
    source = recognize_retirement_transfer_schedule(source_text, citation)
    if source is None:
        return None
    binding = (
        bind_retirement_transfer_form(content, source_text, citation)
        if isinstance(content, str)
        else None
    )
    observation = (
        observe_retirement_transfer(binding, evidence, content, cases)
        if binding is not None
        and evidence is not None
        and isinstance(cases, (list, tuple))
        else None
    )
    return _RetirementTransferContext(source, binding, observation)
