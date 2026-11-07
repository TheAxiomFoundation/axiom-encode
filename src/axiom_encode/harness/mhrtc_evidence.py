"""Analysis-local MHRtC ownership; public objects never establish authority.

Known M citation input is reserved even when binding or execution is missing.
Only exact inventories plus complete private observations enable consumption.
No production dispatcher is registered by this module.
"""

from __future__ import annotations

import weakref
from collections import Counter
from dataclasses import asdict, dataclass
from functools import partial

import yaml

from .bound_mhrtc_program import _typed_equal, _UniqueLoader, bind_mhrtc_program
from .mhrtc_numeric import _typed, mhrtc_numeric_roles
from .mhrtc_program_contract import CITATION, SOURCE_TEXT
from .mhrtc_source_inventory import mhrtc_branch_roles

# Exact actual source-condition callback inventory: (start, end, text_start).
# text_start records only the observed leading-space alias, never containment.
# Each rule's closed proof atoms and formula remain independently binder checked.
_GATES = {
    "actual_allocation_applies": ((1777, 1914, 1778),),
    "additional_renovation_sheet_required": ((6255, 6418, 6255),),
    "allocated_local_capacity": ((1777, 1914, 1778), (6542, 7260, 6542)),
    "attach_schedule_to_return": ((350, 403, 350),),
    "claim_source_eligibility": (
        (784, 1418, 784),
        (1425, 1504, 1426),
        (1504, 1597, 1504),
        (2792, 3630, 2792),
        (4098, 5564, 4098),
        (4098, 5700, 4098),
        (5700, 6067, 5700),
    ),
    "completion_in_tax_year": ((4098, 5700, 4098),),
    "eligible_claimant": ((784, 1418, 784), (1425, 1504, 1426), (4098, 5564, 4098)),
    "eligible_dwelling": ((2792, 3630, 2792),),
    "federal_2025_mhrtc_total_qualifying_expenditures_line_45354": (
        (7292, 7375, 7293),
    ),
    "filing_claimant_identity": ((784, 1418, 784), (7292, 7375, 7293)),
    "historical_record_count": ((1504, 1597, 1504),),
    "include_supporting_documents_with_return": ((403, 504, 404),),
    "incurred_after_source_cutoff": ((3929, 3998, 3930),),
    "land_extent_eligible": ((3255, 3630, 3255),),
    "line_one": ((3630, 3998, 3630), (6542, 7260, 6542)),
    "line_three": ((6542, 7260, 6542),),
    "line_two": ((2511, 2792, 2512), (6542, 7260, 6542)),
    "multigenerational_home_renovation_tax_credit_line_45355": (
        (2355, 2504, 2355),
        (7375, 7710, 7375),
    ),
    "other_actual_allocation_applies": ((1777, 1914, 1778),),
    "other_allocated_local_capacity": ((1777, 1914, 1778), (6542, 7260, 6542)),
    "other_claim_source_eligibility": (
        (784, 1418, 784),
        (1425, 1504, 1426),
        (1504, 1597, 1504),
        (2792, 3630, 2792),
        (4098, 5564, 4098),
        (4098, 5700, 4098),
        (5700, 6067, 5700),
    ),
    "other_completion_in_tax_year": ((4098, 5700, 4098),),
    "other_eligible_claimant": (
        (784, 1418, 784),
        (1425, 1504, 1426),
        (4098, 5564, 4098),
    ),
    "other_eligible_dwelling": ((2792, 3630, 2792),),
    "other_historical_record_count": ((1504, 1597, 1504),),
    "other_incurred_after_source_cutoff": ((3929, 3998, 3930),),
    "other_land_extent_eligible": ((3255, 3630, 3255),),
    "other_line_one": ((3630, 3998, 3630), (6542, 7260, 6542)),
    "other_line_three": ((6542, 7260, 6542),),
    "other_line_two": ((2511, 2792, 2512), (6542, 7260, 6542)),
    "other_paid_after_source_cutoff": ((3929, 3998, 3930),),
    "other_prior_distinct_renovation_count": ((1504, 1597, 1504),),
    "other_qualifying_individual": ((4098, 5564, 4098),),
    "other_qualifying_relation": ((4098, 5564, 4098),),
    "other_qualifying_renovation": ((4098, 5564, 4098), (5700, 6067, 5700)),
    "other_receipt_basic_conditions": ((2792, 2914, 2792), (3630, 3998, 3630)),
    "other_record_actual_claimed_expenses": ((6542, 7260, 6542),),
    "other_record_claimant_key": ((6542, 7260, 6542),),
    "other_record_expense_land_scope_key": ((3255, 3630, 3255),),
    "other_record_historical_individual_key": ((1504, 1597, 1504),),
    "other_record_historical_renovation_key": ((1504, 1597, 1504),),
    "other_record_individual_key": ((1504, 1597, 1504),),
    "other_record_land_scope_key": ((3255, 3630, 3255),),
    "other_record_renovation_key": ((1597, 1777, 1597),),
    "other_secondary_unit": ((5700, 6067, 5700),),
    "other_selected_for_individual": ((1504, 1597, 1504),),
    "paid_after_source_cutoff": ((3929, 3998, 3930),),
    "prior_distinct_renovation_count": ((1504, 1597, 1504),),
    "qualifying_individual": ((4098, 5564, 4098),),
    "qualifying_relation": ((4098, 5564, 4098),),
    "qualifying_renovation": ((4098, 5564, 4098), (5700, 6067, 5700)),
    "receipt_basic_conditions": ((2792, 2914, 2792), (3630, 3998, 3630)),
    "record_claimant_key": ((6542, 7260, 6542),),
    "record_expense_land_scope_key": ((3255, 3630, 3255),),
    "record_historical_individual_key": ((1504, 1597, 1504),),
    "record_historical_renovation_key": ((1504, 1597, 1504),),
    "record_individual_key": ((1504, 1597, 1504),),
    "record_land_scope_key": ((3255, 3630, 3255),),
    "record_renovation_key": ((1597, 1777, 1597),),
    "retain_supporting_documents": ((403, 504, 404),),
    "secondary_unit": ((5700, 6067, 5700),),
    "selected_claim_capacity": (
        (1777, 1914, 1778),
        (2205, 2355, 2205),
        (6542, 7260, 6542),
    ),
    "selected_for_individual": ((1504, 1597, 1504),),
    "selected_other_claim_amount": ((1504, 1597, 1504), (6542, 7260, 6542)),
    "selected_qualifying_expenditures": ((1504, 1597, 1504), (6542, 7260, 6542)),
    "selected_renovation_sheet_count": ((6255, 6418, 6255),),
    "selected_total_qualifying_expenditures": ((1504, 1597, 1504), (7292, 7375, 7293)),
    "selection_record_count": ((1504, 1597, 1504),),
    "selection_record_individual_key": ((1504, 1597, 1504),),
    "selection_record_renovation_key": ((1504, 1597, 1504),),
}

_RESERVED = weakref.WeakKeyDictionary()
_REGISTERED = weakref.WeakKeyDictionary()


def _inventory(content, source):
    from . import source_completeness as s
    from .validator_pipeline import (
        extract_typed_numeric_inventory_occurrences_from_text,
    )

    bound = bind_mhrtc_program(content, source)
    if bound is None:
        raise ValueError("closed source/program binding required")
    branches = s.recognize_source_structure(source, corpus_citation_path=CITATION)
    printed = s._printed_operation_context(
        source,
        branches,
        CITATION,
        consumer_content=content,
        proof_sources={CITATION: source},
    )
    formulas = tuple(
        s._source_formula_branches(
            source,
            printed_operations=printed,
            corpus_citation_path=CITATION,
            branches=branches,
            active_branches=branches,
            deferred_paths=set(),
        )
    )
    extract = partial(
        extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
    )
    shared = dict(
        raw_source_text=source,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
        formula_branches=formulas,
        deferred_clauses=[],
        corpus_citation_path=CITATION,
    )
    controls = tuple(
        s._source_control_branches(
            source,
            **shared,
            extract_numeric_occurrences=extract,
            numeric_exception_controls_only=False,
        )
    )
    exceptions = tuple(
        s._source_exception_branches(source, **shared, numeric_limits_only=False)
    )
    mhrtc_branch_roles(
        bound,
        content,
        source,
        formulas=formulas,
        controls=controls,
        exceptions=exceptions,
    )
    recall = s.authoritative_numeric_recall_text(
        source.strip(), corpus_citation_path=CITATION
    )
    numeric = tuple(
        o
        for o in extract(recall)
        if not s._temporal_occurrence_is_formula_applicability_preface(o, recall)
    )
    numeric_roles = mhrtc_numeric_roles(bound, content, source, numeric)
    branch_keys = tuple(
        (kind, _typed(asdict(b)))
        for kind, entries in (
            ("formula", formulas),
            ("control", controls),
            ("exception", exceptions),
        )
        for b in entries
    )
    numeric_keys = tuple(
        (_typed(asdict(o)), role)
        for o, role in zip(numeric, numeric_roles, strict=True)
    )
    branch_objects = (
        ("formula", formulas),
        ("control", controls),
        ("exception", exceptions),
    )
    return branch_keys, numeric_keys, branch_objects, numeric


def _missing_inventory(supplied, expected_objects):
    """Retain exact omitted occurrences as well as mismatched supplied items."""
    actual = Counter(_typed(asdict(item)) for item in supplied)
    omitted = []
    for item in expected_objects:
        key = _typed(asdict(item))
        if actual[key]:
            actual[key] -= 1
        else:
            omitted.append(item)
    return supplied + tuple(omitted)


@dataclass(eq=False, frozen=True)
class MhrtcEvidence:
    recognized: bool
    source: str
    complete: bool = False

    def _reserved(self):
        return _RESERVED.get(self) == (True, self.source) and self.recognized is True

    def live(self):
        from .mhrtc_capture import source_binding_live

        entry = _REGISTERED.get(self)
        return bool(
            self._reserved()
            and self.complete is True
            and entry is not None
            and self.source == entry[1].source
            and source_binding_live(entry[0], entry[1])
        )

    def owns_branch(self, branch, kind):
        # Reservation is deliberately independent of successful inventories:
        # malformed or new occurrences under this citation cannot fall back.
        return self._reserved() and kind in ("formula", "control", "exception")

    def partition(self, branches, kind):
        """Withhold the entire category if metadata or multiplicity differs."""
        from .source_completeness import SourceStructureBranch

        if not self._reserved():
            return tuple(branches), ()
        supplied = tuple(branches)
        entry = _REGISTERED.get(self)
        if kind not in ("formula", "control", "exception") or not self.live():
            return (), supplied
        try:
            if any(type(b) is not SourceStructureBranch for b in supplied):
                return (), supplied
            expected = Counter(key for category, key in entry[2] if category == kind)
            actual = Counter(_typed(asdict(b)) for b in supplied)
            if actual == expected:
                return (), ()
            expected_objects = dict(entry[5])[kind]
            return (), _missing_inventory(supplied, expected_objects)
        except (TypeError, ValueError):
            return (), supplied

    def covers_branch(self, branch, kind):
        from .source_completeness import SourceStructureBranch

        if not self.live() or type(branch) is not SourceStructureBranch:
            return False
        try:
            return (kind, _typed(asdict(branch))) in _REGISTERED[self][2]
        except (TypeError, ValueError):
            return False

    def owns_numeric(self, occurrence):
        return self._reserved()

    def numeric_owner(self, occurrence):
        from .validator_pipeline import NumericOccurrence

        if not self.live() or type(occurrence) is not NumericOccurrence:
            return None
        try:
            key = _typed(asdict(occurrence))
            return next(
                (role for candidate, role in _REGISTERED[self][3] if candidate == key),
                None,
            )
        except (TypeError, ValueError):
            return None

    def partition_numeric(self, occurrences):
        from .validator_pipeline import NumericOccurrence

        supplied = tuple(occurrences)
        if not self._reserved():
            return supplied, ()
        if not self.live():
            return (), supplied
        try:
            expected = Counter(key for key, _ in _REGISTERED[self][3])
            if any(type(o) is not NumericOccurrence for o in supplied):
                return (), supplied
            actual = Counter(_typed(asdict(o)) for o in supplied)
            if actual == expected:
                return (), ()
            return (), _missing_inventory(supplied, _REGISTERED[self][6])
        except (TypeError, ValueError):
            return (), supplied

    def covers_gate(self, rule_name, rule, index, formula, start, end, clause):
        from .source_completeness import _SourceConditionClause

        if (
            not self.live()
            or type(index) is not int
            or index != 0
            or type(rule_name) is not str
            or type(clause) is not _SourceConditionClause
        ):
            return False
        entry = _REGISTERED[self]
        expected = next((r for r in entry[4]["rules"] if r["name"] == rule_name), None)
        if expected is None or not _typed_equal(rule, expected):
            return False
        version = expected["versions"][0]
        if not _typed_equal(
            (formula, start, end), (version.get("formula"), "2025-01-01", "2025-12-31")
        ):
            return False
        try:
            actual = _typed(asdict(clause))
            return any(
                actual
                == _typed(
                    {
                        "branch_path": (),
                        "start": a,
                        "end": b,
                        "text": self.source[text_start:b],
                    }
                )
                for a, b, text_start in _GATES.get(rule_name, ())
            )
        except (TypeError, ValueError):
            return False

    def missing(self):
        if self._reserved() and not self.live():
            return (
                "[complete-source-unit:mhrtc-evidence] Exact source inventories and complete fresh private MHRtC witnesses required; generic evidence cannot replace missing or stale ownership.",
            )
        return ()


def certify_mhrtc(content, source, citation, raw_cases, evidence):
    """Analyze once; subsequent consumers must recheck private generation live."""
    from .mhrtc_capture import source_binding_live, source_coverage_binding

    recognized = type(citation) is str and citation == CITATION
    absent = MhrtcEvidence(recognized, source)
    _RESERVED[absent] = (recognized, source)
    if not recognized:
        return absent
    try:
        if type(source) is not str or source != SOURCE_TEXT:
            return absent
        binding = source_coverage_binding(evidence, content, source, raw_cases)
        if binding is None:
            return absent
        context, _, coverage = binding
        if coverage.missing:
            return absent
        branch_keys, numeric, branch_objects, numeric_objects = _inventory(
            content, source
        )
        payload = yaml.load(content, Loader=_UniqueLoader)
        if not source_binding_live(evidence, context):
            return absent
        result = MhrtcEvidence(True, source, True)
        _RESERVED[result] = (True, source)
        _REGISTERED[result] = (
            evidence,
            context,
            branch_keys,
            numeric,
            payload,
            branch_objects,
            numeric_objects,
        )
        return result
    except (OSError, ValueError, TypeError, AttributeError, KeyError, yaml.YAMLError):
        return absent
