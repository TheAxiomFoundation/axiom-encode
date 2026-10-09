"""Vintaged producers in the canonical-concept registry.

The SNAP cost-of-living concepts have one producer module per fiscal year
(rulespec-us#1394, #759). Before this change the registry accepted exactly one
producer anchor, `fy-2026-cola/maximum-allotments`, so an FY2027 producer was
refused with `canonical_conflict` and test auto-repair silently re-anchored
FY2027 (and FY2024) output assertions to the FY2026 module.

Invariants exercised here (example tests first, then Hypothesis properties
whose oracles read only what the strategy drew, never the `Concept`
predicates under test):

1. Acceptance: a module may define a registered canonical, and a non-input
   reference may target it, iff its anchor is one of the concept's
   `producer_anchors`.
2. Auto-repair never moves a reference whose anchor is an accepted producer,
   never moves an input reference, and redirects an anchor only to the
   concept's single producer; with several vintages it leaves the anchor
   alone (no guessing) and validation names every accepted producer.
3. Auto-repair renames a blocked synonym to its canonical unless the
   reference is an input slot of an imported module or of one of the
   concept's producers, and validation flags `blocked_synonym` on exactly
   those references; nothing else is renamed.
4. Auto-repair is idempotent.
5. For registries in which every concept has at most one producer, the
   validator, auto-repair and encoder prompt guidance agree exactly with the
   single-anchor implementation from origin/main 436cf3044 (differential
   oracle in tests/concepts_single_anchor_reference.py).
6. The corpus audit reports a canonical producer as a conflict iff its anchor
   is not an accepted producer, and reports every accepted producer.
7. The registry YAML loader round-trips producers and periods.
8. Producer periods load iff each is ordered and no two share a day, so at
   most one vintage covers any date.
9. A vintaged concept's prompt line names each accepted producer exactly
   once, with its period, and calls the module being encoded a producer iff
   it is one.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pytest
import yaml
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from axiom_encode.cli import (
    _enforce_canonical_concept_registry,
    _relative_output_to_anchor,
)
from axiom_encode.concepts.audit import audit_corpus
from axiom_encode.concepts.auto_repair import (
    _rewrite_anchored_refs,
    auto_repair_test_yaml_canonical_violations,
)
from axiom_encode.concepts.registry import (
    PRODUCER_ANCHOR_RE,
    REGISTRY_FORMAT,
    Concept,
    ConceptRegistry,
    ProducerPeriod,
    load_concept_registry,
)
from axiom_encode.concepts.validator import (
    ANCHORED_REF_RE,
    validate_generated_against_registry,
)
from axiom_encode.harness.evals import (
    EvalWorkspace,
    _build_rulespec_eval_prompt,
    _canonical_target_ref_prefix,
    _format_canonical_concept_registry_guidance,
    _resolve_eval_output_path,
)
from axiom_encode.prepare_signed_backfill import citation_rulespec_path
from tests.concepts_single_anchor_reference import (
    legacy_format_canonical_concept_registry_guidance,
    legacy_rewrite_anchored_refs,
    legacy_validate_generated_against_registry,
)

FY2024 = "us:policies/usda/snap/fy-2024-cola/maximum-allotments"
FY2026 = "us:policies/usda/snap/fy-2026-cola/maximum-allotments"
FY2027 = "us:policies/usda/fns/snap-fy2027-cola/page-4"
COLA_VINTAGES = (FY2024, FY2026, FY2027)
COLA_NAMES = ("snap_maximum_allotment", "snap_one_person_thrifty_food_plan_cost")
FY2027_CITATION_PREFIX = "us/guidance/usda/fns/snap-fy2027-cola/page-"
CONSUMER = "us:statutes/7/2017/a"

# Test fixture data only. The 48 states/DC column of the FY2027 memorandum,
# page 4 (sizes 1-8); the registry checks names and anchors, not values.
_FY2027_PAGE_4_MODULE = """\
format: rulespec/v1
rules:
  - name: snap_maximum_allotment_table
    kind: parameter
    dtype: Money
    indexed_by: household_size
    versions:
      - effective_from: '2026-10-01'
        values:
          1: 306
          2: 562
          3: 808
          4: 1023
          5: 1217
          6: 1463
          7: 1616
          8: 1841
  - name: snap_maximum_allotment
    kind: derived
    versions:
      - effective_from: '2026-10-01'
        formula: snap_maximum_allotment_table[household_size]
  - name: snap_one_person_thrifty_food_plan_cost
    kind: derived
    versions:
      - effective_from: '2026-10-01'
        formula: snap_maximum_allotment_table[1]
"""


def _vintage_test_yaml(anchor: str) -> str:
    return f"""\
- name: one_person_household_gets_the_one_person_allotment
  period: 2026-10
  input:
    {anchor}#input.household_size: 1
  output:
    {anchor}#snap_maximum_allotment: 306
    {anchor}#snap_one_person_thrifty_food_plan_cost: 306
"""


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def _kinds_for(violations, name: str) -> set[str]:
    return {v.kind for v in violations if v.name == name}


# ---------------------------------------------------------------------------
# Packaged registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", COLA_NAMES)
def test_packaged_cola_concepts_list_every_fiscal_year_producer(name):
    concept = load_concept_registry().lookup_canonical(name)
    assert concept is not None
    assert concept.producer_anchors == COLA_VINTAGES
    # Back-compat label only: the vintage the current consumer chain imports.
    assert concept.producer_anchor == FY2026
    assert concept.has_producer
    assert concept.unique_producer_anchor is None


@pytest.mark.parametrize("name", COLA_NAMES)
def test_packaged_cola_concepts_label_each_vintage_with_its_fiscal_year(name):
    """Federal fiscal years; FY2027 is the memorandum's 2026-10-01..2027-09-30."""
    concept = load_concept_registry().lookup_canonical(name)
    assert [
        (p.anchor, p.label, p.effective_from, p.effective_to)
        for p in concept.producer_periods
    ] == [
        (FY2024, "FY2024", date(2023, 10, 1), date(2024, 9, 30)),
        (FY2026, "FY2026", date(2025, 10, 1), date(2026, 9, 30)),
        (FY2027, "FY2027", date(2026, 10, 1), date(2027, 9, 30)),
    ]
    assert concept.producer_period(FY2027).describe() == (
        "FY2027, 2026-10-01 to 2027-09-30"
    )
    assert concept.producer_period(CONSUMER) is None


def test_fy2027_anchor_is_the_encoders_canonical_path_for_memo_page_4():
    """The registered FY2027 anchor is where the encoder must write page 4.

    Fresh encodes and reviewed-candidate promotions are fixed to
    `citation_rulespec_path(citation)` (cli.py promote path check), so the
    registry must name exactly that module, and no other memo page.
    """
    registry = load_concept_registry()
    for page in range(1, 8):
        relative = Path(citation_rulespec_path(f"{FY2027_CITATION_PREFIX}{page}"))
        anchor = _relative_output_to_anchor(relative)
        for name in COLA_NAMES:
            accepted = registry.lookup_canonical(name).accepts_producer_anchor(anchor)
            assert accepted is (page == 4), (page, anchor)
    page_4 = Path(citation_rulespec_path(f"{FY2027_CITATION_PREFIX}4"))
    assert page_4 == Path("us/policies/usda/fns/snap-fy2027-cola/page-4.yaml")
    assert _relative_output_to_anchor(page_4) == FY2027


def test_only_the_cola_concepts_have_more_than_one_producer():
    registry = load_concept_registry()
    multi = {
        concept.canonical_name
        for concept in registry.concepts_by_id.values()
        if len(concept.producer_anchors) > 1
    }
    assert multi == set(COLA_NAMES)
    for concept in registry.concepts_by_id.values():
        if concept.canonical_name in COLA_NAMES:
            continue
        expected = (concept.producer_anchor,) if concept.producer_anchor else ()
        assert concept.producer_anchors == expected


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _registry_yaml(tmp_path: Path, concept_body: str) -> Path:
    root = tmp_path / "registry"
    _write(
        root / "c.yaml",
        f"format: {REGISTRY_FORMAT}\nconcepts:\n"
        + textwrap.indent(textwrap.dedent(concept_body).strip("\n"), "  ")
        + "\n",
    )
    return root


def test_loader_reads_producer_anchors_without_legacy_anchor(tmp_path: Path):
    root = _registry_yaml(
        tmp_path,
        """
        - id: t.vintaged
          canonical_name: vintaged_amount
          producer_anchors: [us:policies/a/fy-1, us:policies/a/fy-2]
        """,
    )
    concept = load_concept_registry(root).lookup_canonical("vintaged_amount")
    assert concept.producer_anchors == ("us:policies/a/fy-1", "us:policies/a/fy-2")
    assert concept.producer_anchor is None
    assert concept.has_producer


def test_loader_derives_producer_anchors_from_legacy_anchor(tmp_path: Path):
    root = _registry_yaml(
        tmp_path,
        """
        - id: t.single
          canonical_name: single_amount
          producer_anchor: us:policies/a/b
        """,
    )
    concept = load_concept_registry(root).lookup_canonical("single_amount")
    assert concept.producer_anchors == ("us:policies/a/b",)
    assert concept.unique_producer_anchor == "us:policies/a/b"


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            """
            - id: t.bad
              canonical_name: bad_amount
              producer_anchor: us:policies/a/fy-3
              producer_anchors: [us:policies/a/fy-1, us:policies/a/fy-2]
            """,
            "must be one of producer_anchors",
        ),
        (
            """
            - id: t.bad
              canonical_name: bad_amount
              producer_anchors: [us:policies/a/fy-1, us:policies/a/fy-1]
            """,
            "duplicate producer anchors",
        ),
        (
            """
            - id: t.bad
              canonical_name: bad_amount
              producer_anchors: []
            """,
            "non-empty list",
        ),
        (
            """
            - id: t.bad
              canonical_name: bad_amount
              producer_anchors: us:policies/a/fy-1
            """,
            "non-empty list",
        ),
        (
            """
            - id: t.bad
              canonical_name: bad_amount
              producer_anchors: [1]
            """,
            "only strings",
        ),
        (
            """
            - id: t.bad
              canonical_name: bad_amount
              producer_anchors: ["US:policies/a"]
            """,
            "is not a <jurisdiction>:<path> RuleSpec anchor",
        ),
        (
            """
            - id: t.bad
              canonical_name: bad_amount
              producer_anchor: "policies/a#b"
            """,
            "is not a <jurisdiction>:<path> RuleSpec anchor",
        ),
    ],
)
def test_loader_rejects_malformed_producer_anchors(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        load_concept_registry(_registry_yaml(tmp_path, body))


def test_direct_concept_construction_keeps_single_anchor_semantics():
    concept = Concept(id="t.c", canonical_name="c_amount", producer_anchor="us:x/1")
    assert concept.producer_anchors == ("us:x/1",)
    assert concept.accepts_producer_anchor("us:x/1")
    assert not concept.accepts_producer_anchor("us:x/2")
    assert not concept.accepts_producer_anchor(None)
    assert Concept(id="t.n", canonical_name="n", producer_anchor=None).has_producer is (
        False
    )


_TWO_VINTAGES = "producer_anchors: [us:policies/a/fy-1, us:policies/a/fy-2]"


def _period(anchor: str, label: str, start: str, end: str) -> str:
    return (
        f"- {{anchor: {anchor}, label: {label}, effective_from: {start}, "
        f"effective_to: {end}}}"
    )


def test_loader_reads_producer_periods_in_anchor_order(tmp_path: Path):
    root = _registry_yaml(
        tmp_path,
        f"""
        - id: t.vintaged
          canonical_name: vintaged_amount
          {_TWO_VINTAGES}
          producer_periods:
            {_period("us:policies/a/fy-2", "FY2", "'2025-10-01'", "2026-09-30")}
            {_period("us:policies/a/fy-1", "FY1", "2024-10-01", "2025-09-30")}
        """,
    )
    concept = load_concept_registry(root).lookup_canonical("vintaged_amount")
    assert concept.producer_periods == (
        ProducerPeriod(
            "us:policies/a/fy-1", "FY1", date(2024, 10, 1), date(2025, 9, 30)
        ),
        ProducerPeriod(
            "us:policies/a/fy-2", "FY2", date(2025, 10, 1), date(2026, 9, 30)
        ),
    )


@pytest.mark.parametrize(
    ("periods", "message"),
    [
        ("producer_periods: []", "producer_periods must be a non-empty list"),
        (
            "producer_periods: {us:policies/a/fy-1: FY1}",
            "producer_periods must be a non-empty list",
        ),
        ("producer_periods: [FY1]", "each period must be a mapping"),
        (
            "producer_periods:\n  " + _period("1", "FY1", "2024-10-01", "2025-09-30"),
            "anchor must be a string, got 1",
        ),
        (
            "producer_periods:\n  - {anchor: us:policies/a/fy-1, label: FY1}",
            r"missing \['effective_from', 'effective_to'\]",
        ),
        (
            "producer_periods:\n  "
            + _period("us:policies/a/fy-1", "FY1", "2024-10-01", "2025-09-30")[:-1]
            + ", fiscal_year: 2025}",
            r"unknown \['fiscal_year'\]",
        ),
        (
            "producer_periods:\n  "
            + _period("us:policies/a/fy-1", "FY1", "2024-10-01", "2025-09-30"),
            "must name each producer anchor exactly once",
        ),
        (
            "producer_periods:\n  "
            + _period("us:policies/a/fy-1", "FY1", "2024-10-01", "2025-09-30")
            + "\n  "
            + _period("us:policies/a/fy-3", "FY3", "2026-10-01", "2027-09-30"),
            "must name each producer anchor exactly once",
        ),
        (
            "producer_periods:\n  "
            + _period("us:policies/a/fy-1", "FY1", "2024-10-01", "2025-09-30")
            + "\n  "
            + _period("us:policies/a/fy-1", "FY1b", "2026-10-01", "2027-09-30"),
            "must name each producer anchor exactly once",
        ),
        (
            "producer_periods:\n  "
            + _period("us:policies/a/fy-1", "FY1", "2024-10-01", "2025-10-01")
            + "\n  "
            + _period("us:policies/a/fy-2", "FY2", "2025-10-01", "2026-09-30"),
            "overlap",
        ),
        (
            "producer_periods:\n  "
            + _period("us:policies/a/fy-1", "FY1", "2025-09-30", "2024-10-01")
            + "\n  "
            + _period("us:policies/a/fy-2", "FY2", "2025-10-01", "2026-09-30"),
            "is after effective_to",
        ),
        (
            "producer_periods:\n  "
            + _period("us:policies/a/fy-1", "''", "2024-10-01", "2025-09-30")
            + "\n  "
            + _period("us:policies/a/fy-2", "FY2", "2025-10-01", "2026-09-30"),
            "label must be a non-empty string",
        ),
        (
            "producer_periods:\n  "
            + _period("us:policies/a/fy-1", "FY", "2024-10-01", "2025-09-30")
            + "\n  "
            + _period("us:policies/a/fy-2", "FY", "2025-10-01", "2026-09-30"),
            "duplicate producer period labels",
        ),
        (
            "producer_periods:\n  "
            + _period("us:policies/a/fy-1", "FY1", "'2024-13-01'", "2025-09-30")
            + "\n  "
            + _period("us:policies/a/fy-2", "FY2", "2025-10-01", "2026-09-30"),
            "expected an ISO calendar date",
        ),
        (
            "producer_periods:\n  "
            + _period("us:policies/a/fy-1", "FY1", "2024-10-01 00:00:00", "2025-09-30")
            + "\n  "
            + _period("us:policies/a/fy-2", "FY2", "2025-10-01", "2026-09-30"),
            "expected a calendar date, got datetime",
        ),
    ],
)
def test_loader_rejects_malformed_producer_periods(tmp_path, periods, message):
    body = (
        "- id: t.bad\n  canonical_name: bad_amount\n  "
        + _TWO_VINTAGES
        + "\n  "
        + periods.replace("\n", "\n  ")
        + "\n"
    )
    with pytest.raises(ValueError, match=message):
        load_concept_registry(_registry_yaml(tmp_path, body))


def test_loader_rejects_periods_without_producers(tmp_path: Path):
    root = _registry_yaml(
        tmp_path,
        f"""
        - id: t.bad
          canonical_name: bad_amount
          producer_periods:
            {_period("us:policies/a/fy-1", "FY1", "2024-10-01", "2025-09-30")}
        """,
    )
    with pytest.raises(ValueError, match="must name each producer anchor exactly once"):
        load_concept_registry(root)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("anchor", COLA_VINTAGES)
def test_validator_accepts_every_registered_cola_vintage(tmp_path: Path, anchor):
    module = _write(tmp_path / "module.yaml", _FY2027_PAGE_4_MODULE)
    test_file = _write(tmp_path / "module.test.yaml", _vintage_test_yaml(anchor))
    violations = validate_generated_against_registry(
        [module, test_file], load_concept_registry(), apply_anchor=anchor
    )
    assert violations == []


@pytest.mark.parametrize(
    "anchor",
    [
        "us:policies/usda/fns/snap-fy2027-cola/page-5",
        "us:policies/usda/snap/fy-2027-cola/maximum-allotments",
        "us:policies/usda/fns/snap-fy2027-cola",
        CONSUMER,
    ],
)
def test_validator_rejects_unlisted_producer_and_names_all_vintages(
    tmp_path: Path, anchor
):
    module = _write(tmp_path / "module.yaml", _FY2027_PAGE_4_MODULE)
    violations = validate_generated_against_registry(
        [module], load_concept_registry(), apply_anchor=anchor
    )
    conflicts = [v for v in violations if v.kind == "canonical_conflict"]
    assert {v.name for v in conflicts} == set(COLA_NAMES)
    for violation in conflicts:
        for vintage in COLA_VINTAGES:
            assert vintage in violation.detail
        assert f"applying under {anchor}" in violation.detail


def test_validator_flags_output_ref_at_non_producer_and_names_all_vintages(
    tmp_path: Path,
):
    test_file = _write(
        tmp_path / "a.test.yaml",
        f"""\
- name: consumer_asserts_imported_output
  period: 2026-10
  output:
    {CONSUMER}#snap_maximum_allotment: 306
""",
    )
    violations = validate_generated_against_registry(
        [test_file], load_concept_registry(), apply_anchor=CONSUMER
    )
    assert [v.kind for v in violations] == ["anchored_ref_miss"]
    for vintage in COLA_VINTAGES:
        assert vintage in violations[0].detail
    assert "never chooses a vintage" in violations[0].detail


def test_validator_accepts_free_input_contract_for_year_general_consumer(
    tmp_path: Path,
):
    """#759 inversion: year-general modules read COLA names as free inputs."""
    test_file = _write(
        tmp_path / "a.test.yaml",
        f"""\
- name: consumer_reads_cola_names_as_inputs
  period: 2026-10
  input:
    {CONSUMER}#input.snap_maximum_allotment: 306
    {CONSUMER}#input.snap_one_person_thrifty_food_plan_cost: 306
""",
    )
    assert (
        validate_generated_against_registry(
            [test_file], load_concept_registry(), apply_anchor=CONSUMER
        )
        == []
    )


def test_validator_allows_blocked_synonym_input_slot_on_any_vintage(tmp_path: Path):
    test_file = _write(
        tmp_path / "page-4.test.yaml",
        f"""\
- name: legacy_slot_on_fy2027_producer
  period: 2026-10
  input:
    {FY2027}#input.snap_maximum_allotment_for_household_size: 306
""",
    )
    assert (
        validate_generated_against_registry([test_file], load_concept_registry()) == []
    )


# ---------------------------------------------------------------------------
# Test auto-repair
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("anchor", COLA_VINTAGES)
def test_repair_keeps_output_refs_on_their_own_vintage(tmp_path: Path, anchor):
    """The #1394 trap: FY2027 outputs were silently rewritten to FY2026."""
    test_file = _write(tmp_path / "x.test.yaml", _vintage_test_yaml(anchor))
    before = test_file.read_text()
    changed = auto_repair_test_yaml_canonical_violations(
        [test_file], load_concept_registry(), apply_anchor=anchor
    )
    assert changed == []
    assert test_file.read_text() == before


def test_repair_leaves_ambiguous_vintage_ref_for_validation(tmp_path: Path):
    body = f"""\
- name: consumer_asserts_imported_output
  period: 2026-10
  output:
    {CONSUMER}#snap_one_person_thrifty_food_plan_cost: 306
"""
    test_file = _write(tmp_path / "a.test.yaml", body)
    registry = load_concept_registry()
    assert auto_repair_test_yaml_canonical_violations([test_file], registry) == []
    assert test_file.read_text() == body
    violations = validate_generated_against_registry([test_file], registry)
    assert _kinds_for(violations, "snap_one_person_thrifty_food_plan_cost") == {
        "anchored_ref_miss"
    }


def test_repair_renames_blocked_synonym_without_changing_its_vintage(
    tmp_path: Path,
):
    test_file = _write(
        tmp_path / "page-4.test.yaml",
        f"""\
- name: legacy_name_on_fy2027_output
  period: 2026-10
  output:
    {FY2027}#snap_maximum_allotment_for_household_size: 306
    {CONSUMER}#snap_maximum_allotment_for_one_person_household: 306
""",
    )
    registry = load_concept_registry()
    assert auto_repair_test_yaml_canonical_violations([test_file], registry) == [
        test_file
    ]
    text = test_file.read_text()
    assert f"{FY2027}#snap_maximum_allotment: 306" in text
    # No vintage named and several accepted: rename, keep the anchor, flag it.
    assert f"{CONSUMER}#snap_one_person_thrifty_food_plan_cost: 306" in text
    violations = validate_generated_against_registry([test_file], registry)
    assert [(v.kind, v.name) for v in violations] == [
        ("anchored_ref_miss", "snap_one_person_thrifty_food_plan_cost")
    ]


# ---------------------------------------------------------------------------
# Apply-time hook (encode --apply and refresh-applied-manifest)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("anchor", "relative_output"),
    [
        (FY2027, "us/policies/usda/fns/snap-fy2027-cola/page-4.yaml"),
        (FY2024, "us/policies/usda/snap/fy-2024-cola/maximum-allotments.yaml"),
        (FY2026, "us/policies/usda/snap/fy-2026-cola/maximum-allotments.yaml"),
    ],
)
def test_apply_hook_admits_each_vintage_producer_unchanged(
    tmp_path: Path, anchor, relative_output
):
    module = _write(tmp_path / "out" / "m.yaml", _FY2027_PAGE_4_MODULE)
    test_file = _write(tmp_path / "out" / "m.test.yaml", _vintage_test_yaml(anchor))
    before = test_file.read_bytes()
    repaired = _enforce_canonical_concept_registry(
        candidate_files=[module, test_file],
        relative_output=Path(relative_output),
    )
    assert repaired == []
    assert test_file.read_bytes() == before


def test_apply_hook_uses_content_root_anchor_for_manifest_refresh(tmp_path: Path):
    """refresh-applied-manifest passes the jurisdiction content root."""
    content_root = tmp_path / "rulespec-us" / "us"
    content_root.mkdir(parents=True)
    module = _write(tmp_path / "out" / "page-4.yaml", _FY2027_PAGE_4_MODULE)
    test_file = _write(
        tmp_path / "out" / "page-4.test.yaml", _vintage_test_yaml(FY2027)
    )
    assert (
        _enforce_canonical_concept_registry(
            candidate_files=[module, test_file],
            relative_output=Path("policies/usda/fns/snap-fy2027-cola/page-4.yaml"),
            policy_repo_path=content_root,
        )
        == []
    )


def test_apply_hook_refuses_unlisted_page_without_rewriting_its_tests(
    tmp_path: Path,
):
    page_5 = "us:policies/usda/fns/snap-fy2027-cola/page-5"
    module = _write(tmp_path / "out" / "page-5.yaml", _FY2027_PAGE_4_MODULE)
    test_file = _write(
        tmp_path / "out" / "page-5.test.yaml", _vintage_test_yaml(page_5)
    )
    before = test_file.read_bytes()
    with pytest.raises(RuntimeError) as excinfo:
        _enforce_canonical_concept_registry(
            candidate_files=[module, test_file],
            relative_output=Path("us/policies/usda/fns/snap-fy2027-cola/page-5.yaml"),
        )
    message = str(excinfo.value)
    assert "[canonical_conflict] snap_maximum_allotment" in message
    assert "[anchored_ref_miss] snap_maximum_allotment" in message
    for vintage in COLA_VINTAGES:
        assert vintage in message
    assert test_file.read_bytes() == before


# ---------------------------------------------------------------------------
# Corpus audit and CLI
# ---------------------------------------------------------------------------


def _producer_corpus(tmp_path: Path, anchors: list[str]) -> Path:
    root = tmp_path / "rulespec-us"
    for anchor in anchors:
        jurisdiction, rel = anchor.split(":", 1)
        _write(root / jurisdiction / f"{rel}.yaml", _FY2027_PAGE_4_MODULE)
    return root


def test_audit_accepts_every_vintage_and_reports_all_producers(tmp_path: Path):
    page_5 = "us:policies/usda/fns/snap-fy2027-cola/page-5"
    root = _producer_corpus(tmp_path, [*COLA_VINTAGES, page_5])
    findings = [
        f
        for f in audit_corpus([root], load_concept_registry())
        if f.kind == "canonical_conflict"
    ]
    assert {f.name for f in findings} == set(COLA_NAMES)
    for finding in findings:
        assert finding.site_paths == (
            root / "us/policies/usda/fns/snap-fy2027-cola/page-5.yaml",
        )
        assert finding.accepted_producers == COLA_VINTAGES
        assert finding.detail.endswith("expects one of " + ", ".join(COLA_VINTAGES))


def test_audit_has_no_conflict_when_only_listed_vintages_produce(tmp_path: Path):
    root = _producer_corpus(tmp_path, list(COLA_VINTAGES))
    findings = audit_corpus([root], load_concept_registry(), name_prefixes=("snap_",))
    assert not [f for f in findings if f.kind == "canonical_conflict"]


def test_concepts_audit_cli_reports_accepted_producers(tmp_path: Path):
    root = _producer_corpus(
        tmp_path, [FY2026, "us:policies/usda/fns/snap-fy2027-cola/page-5"]
    )
    command = [
        sys.executable,
        "-m",
        "axiom_encode.cli",
        "concepts-audit",
        "--roots",
        str(root),
        "--name-prefix",
        "snap_maximum_allotment",
    ]
    payload = json.loads(
        subprocess.run(
            [*command, "--json"], capture_output=True, text=True, check=True
        ).stdout
    )
    conflicts = [f for f in payload["findings"] if f["kind"] == "canonical_conflict"]
    assert {f["name"] for f in conflicts} == set(COLA_NAMES)
    for finding in conflicts:
        assert finding["accepted_producers"] == list(COLA_VINTAGES)
        assert finding["anchor"] == FY2026  # legacy field, unchanged meaning
    text = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    assert "snap_maximum_allotment @ one of " + ", ".join(COLA_VINTAGES) in text


# ---------------------------------------------------------------------------
# Encoder prompt guidance (runs with and without --apply)
# ---------------------------------------------------------------------------

_COLA_PERIOD_TEXT = {
    FY2024: "FY2024, 2023-10-01 to 2024-09-30",
    FY2026: "FY2026, 2025-10-01 to 2026-09-30",
    FY2027: "FY2027, 2026-10-01 to 2027-09-30",
}


def _guidance(tmp_path: Path, source_text: str, **kwargs) -> str:
    source_file = _write(tmp_path / "source.txt", source_text)
    manifest = _write(tmp_path / "context-manifest.json", "{}")
    workspace = EvalWorkspace(
        root=tmp_path, source_text_file=source_file, manifest_file=manifest
    )
    return _format_canonical_concept_registry_guidance(
        source_text, workspace, context_files=[], **kwargs
    )


def _line_for(section: str, name: str) -> str:
    return next(line for line in section.splitlines() if line.startswith(f"- `{name}`"))


def test_prompt_guidance_lists_every_vintage_producer(tmp_path: Path):
    section = _guidance(tmp_path, "snap_maximum_allotment and snap_total_gross_income")
    allotment_line = _line_for(section, "snap_maximum_allotment")
    for vintage in COLA_VINTAGES:
        assert f"`{vintage}` ({_COLA_PERIOD_TEXT[vintage]})" in allotment_line
    assert "one producer per vintage" in allotment_line
    assert "period covers the dates you encode" in allotment_line
    assert f"producer `{FY2026}`" not in allotment_line
    assert "this module" not in allotment_line  # no target known
    # Single-producer lines are byte-for-byte what main printed.
    assert (
        _line_for(section, "snap_total_gross_income")
        == "- `snap_total_gross_income` — producer `us:regulations/7-cfr/273/10` — "
        "do not use: `snap_gross_monthly_income`, `snap_monthly_gross_income`, "
        "`snap_monthly_household_income`"
    )


def test_prompt_guidance_tells_the_fy2027_page_4_encode_it_is_the_producer(
    tmp_path: Path,
):
    section = _guidance(
        tmp_path,
        "snap_maximum_allotment snap_one_person_thrifty_food_plan_cost",
        target_anchor=FY2027,
    )
    for name in COLA_NAMES:
        line = _line_for(section, name)
        assert (
            f"this module, `{FY2027}`, is the FY2027 producer "
            "(2026-10-01 to 2027-09-30): when this source sets it, define it "
            "here under this exact name"
        ) in line
        other_vintages = line.split("other vintages: ", 1)[1]
        assert f"`{FY2027}`" not in other_vintages
        for vintage in (FY2024, FY2026):
            assert f"`{vintage}` ({_COLA_PERIOD_TEXT[vintage]})" in other_vintages
        assert "do not define" not in line


@pytest.mark.parametrize(
    "target",
    ["us:policies/usda/fns/snap-fy2027-cola/page-5", "us:regulations/7-cfr/273/10"],
)
def test_prompt_guidance_tells_other_modules_not_to_define_the_name(
    tmp_path: Path, target
):
    line = _line_for(
        _guidance(tmp_path, "snap_maximum_allotment", target_anchor=target),
        "snap_maximum_allotment",
    )
    assert "this module is not one of them, so do not define a rule" in line
    assert "this module, `" not in line
    for vintage in COLA_VINTAGES:
        assert f"`{vintage}` ({_COLA_PERIOD_TEXT[vintage]})" in line


def test_prompt_target_for_the_page_4_citation_is_the_registered_fy2027_anchor():
    """The prompt's target_ref_prefix for the page-4 encode is the FY2027 anchor.

    `_build_rulespec_eval_prompt` passes `target_ref_prefix` as the guidance's
    `target_anchor`; derive it the way the eval run does (evals.py: the
    output path from `_resolve_eval_output_path`, the prefix from
    `_canonical_target_ref_prefix` with the corpus citation as source id).
    """
    citation = f"{FY2027_CITATION_PREFIX}4"
    relative_output = _resolve_eval_output_path(citation)
    assert _canonical_target_ref_prefix(citation, relative_output) == FY2027


def test_build_rulespec_eval_prompt_passes_target_to_concept_guidance(
    tmp_path: Path,
):
    source_text = "Maximum allotments: snap_maximum_allotment for each size."
    source_file = _write(tmp_path / "source.txt", source_text)
    manifest = _write(tmp_path / "context-manifest.json", "{}")
    workspace = EvalWorkspace(
        root=tmp_path, source_text_file=source_file, manifest_file=manifest
    )
    prompt = _build_rulespec_eval_prompt(
        citation=f"{FY2027_CITATION_PREFIX}4",
        mode="cold",
        workspace=workspace,
        context_files=[],
        target_file_name="policies/usda/fns/snap-fy2027-cola/page-4.yaml",
        target_ref_prefix=FY2027,
        include_tests=False,
        runner_backend="codex",
        policyengine_rule_hint=None,
    )
    assert (
        f"this module, `{FY2027}`, is the FY2027 producer (2026-10-01 to 2027-09-30)"
        in prompt
    )


# ---------------------------------------------------------------------------
# Hypothesis properties
# ---------------------------------------------------------------------------
#
# Every oracle below decides from what the strategy drew (`_Drawn`), never from
# `Concept` methods such as `accepts_producer_anchor`, `has_producer` or
# `unique_producer_anchor`: an oracle that called the predicate under test
# would agree with any bug in it.

PROPERTY_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)

_SEGMENT = st.text(alphabet="abcxyz0189-._AZ", min_size=1, max_size=6)
_SAFE_SEGMENT = st.text(alphabet="abcxyz0189-", min_size=1, max_size=6)
_EPOCH = date(2020, 1, 1)


def _anchors(segment):
    return st.builds(
        lambda jurisdiction, root, parts: f"{jurisdiction}:{root}/" + "/".join(parts),
        st.sampled_from(("us", "us-co", "us-ny")),
        st.sampled_from(("policies", "regulations", "statutes")),
        st.lists(segment, min_size=1, max_size=3),
    )


@dataclass(frozen=True)
class _Drawn:
    """One concept as the strategy drew it: the properties' ground truth."""

    canonical_name: str
    anchors: tuple[str, ...]
    missing: bool
    periods: tuple[ProducerPeriod, ...] = ()

    def lists(self, anchor: str | None) -> bool:
        return anchor in frozenset(self.anchors)

    @property
    def produces(self) -> bool:
        return len(self.anchors) > 0 and not self.missing


@st.composite
def _disjoint_periods(draw, anchors: tuple[str, ...]):
    """One inclusive period per anchor, non-overlapping, in a shuffled order."""
    periods = []
    cursor = draw(st.integers(0, 400))
    for index in draw(st.permutations(range(len(anchors)))):
        length = draw(st.integers(0, 400))
        periods.append(
            ProducerPeriod(
                anchor=anchors[index],
                label=f"V{index}",
                effective_from=_EPOCH + timedelta(days=cursor),
                effective_to=_EPOCH + timedelta(days=cursor + length),
            )
        )
        cursor += length + 1 + draw(st.integers(0, 60))
    return tuple(draw(st.permutations(periods)))


@st.composite
def _registries(draw, *, segment=_SEGMENT, single_anchor_only=False):
    """A random registry, the anchor pool it draws from, and what was drawn."""
    pool = draw(st.lists(_anchors(segment), min_size=1, max_size=6, unique=True))
    concepts = []
    drawn: dict[str, _Drawn] = {}
    for index in range(draw(st.integers(min_value=1, max_value=4))):
        max_anchors = 1 if single_anchor_only else 3
        anchors = tuple(
            draw(st.lists(st.sampled_from(pool), max_size=max_anchors, unique=True))
        )
        if single_anchor_only:
            # The single-anchor implementation reads only producer_anchor.
            legacy = anchors[0] if anchors else None
            periods: tuple[ProducerPeriod, ...] = ()
        else:
            legacy = (
                draw(st.sampled_from(anchors))
                if anchors and draw(st.booleans())
                else None
            )
            periods = (
                draw(_disjoint_periods(anchors))
                if anchors and draw(st.booleans())
                else ()
            )
        canonical = f"name_{index}"
        synonyms = tuple(
            f"name_{index}_old_{k}" for k in range(draw(st.integers(0, 2)))
        )
        missing = draw(st.booleans())
        concepts.append(
            Concept(
                id=f"t.c{index}",
                canonical_name=canonical,
                producer_anchor=legacy,
                blocked_synonyms=synonyms,
                producer_missing=missing,
                producer_anchors=anchors,
                producer_periods=periods,
            )
        )
        truth = _Drawn(canonical, anchors, missing, periods)
        for name in (canonical, *synonyms):
            drawn[name] = truth
    registry = ConceptRegistry(
        concepts_by_id={c.id: c for c in concepts},
        canonical_to_concept={c.canonical_name: c for c in concepts},
        synonym_to_concept={s: c for c in concepts for s in c.blocked_synonyms},
    )
    return pool, registry, drawn


@st.composite
def _registry_and_refs(draw, *, single_anchor_only=False):
    pool, registry, drawn = draw(_registries(single_anchor_only=single_anchor_only))
    names = sorted({*drawn, "unregistered_name"})
    anchor_choices = st.one_of(st.sampled_from(pool), _anchors(_SEGMENT))
    refs = draw(
        st.lists(
            st.tuples(anchor_choices, st.booleans(), st.sampled_from(names)),
            min_size=1,
            max_size=8,
        )
    )
    apply_anchor = draw(st.one_of(st.none(), st.sampled_from(pool)))
    return pool, registry, drawn, refs, apply_anchor


def _ref_text(refs) -> str:
    return "".join(
        f"- {anchor}#{'input.' if is_input else ''}{name}: {index}\n"
        for index, (anchor, is_input, name) in enumerate(refs)
    )


def _parse_lines(text: str):
    parsed = []
    for line in text.splitlines():
        matches = list(ANCHORED_REF_RE.finditer(line))
        assert len(matches) == 1, line
        anchor, input_prefix, name = matches[0].groups()
        parsed.append((anchor, bool(input_prefix), name))
    return parsed


def _keeps_synonym(truth: _Drawn, anchor: str, is_input: bool, apply_anchor) -> bool:
    """A blocked-synonym ref that is an imported module's or a producer's input slot."""
    return is_input and (
        (apply_anchor is not None and anchor != apply_anchor) or truth.lists(anchor)
    )


@PROPERTY_SETTINGS
@given(_registry_and_refs())
def test_property_repair_never_moves_accepted_or_input_refs_and_never_guesses(case):
    _pool, registry, drawn, refs, apply_anchor = case
    repaired = _rewrite_anchored_refs(
        _ref_text(refs), registry, apply_anchor=apply_anchor
    )
    after = _parse_lines(repaired)
    assert len(after) == len(refs)
    for (anchor, is_input, name), (new_anchor, new_input, _name) in zip(refs, after):
        assert new_input == is_input
        truth = drawn.get(name)
        if truth is None:
            assert new_anchor == anchor
            continue
        if is_input or truth.lists(anchor):
            assert new_anchor == anchor
        if len(truth.anchors) > 1:
            assert new_anchor == anchor  # never chooses a vintage
        if new_anchor != anchor:
            assert truth.anchors == (new_anchor,)


@PROPERTY_SETTINGS
@given(_registry_and_refs())
def test_property_repair_renames_exactly_the_candidate_owned_synonyms(case):
    """A blocked synonym becomes its canonical name unless it is an input slot
    of an imported module or of one of the concept's producers; nothing else
    is renamed."""
    _pool, registry, drawn, refs, apply_anchor = case
    repaired = _rewrite_anchored_refs(
        _ref_text(refs), registry, apply_anchor=apply_anchor
    )
    for (anchor, is_input, name), (_anchor, _input, new_name) in zip(
        refs, _parse_lines(repaired)
    ):
        truth = drawn.get(name)
        if (
            truth is None
            or name == truth.canonical_name
            or _keeps_synonym(truth, anchor, is_input, apply_anchor)
        ):
            assert new_name == name
        else:
            assert new_name == truth.canonical_name


@PROPERTY_SETTINGS
@given(_registry_and_refs())
def test_property_repair_is_idempotent(case):
    _pool, registry, _drawn, refs, apply_anchor = case
    once = _rewrite_anchored_refs(_ref_text(refs), registry, apply_anchor=apply_anchor)
    twice = _rewrite_anchored_refs(once, registry, apply_anchor=apply_anchor)
    assert twice == once


@PROPERTY_SETTINGS
@given(_registry_and_refs(), st.data())
def test_property_validation_accepts_iff_listed_producer(case, data):
    pool, registry, drawn, refs, _apply_anchor = case
    canonicals = sorted({truth.canonical_name for truth in drawn.values()})
    defined = data.draw(st.lists(st.sampled_from(canonicals), unique=True))
    apply_anchor = data.draw(st.one_of(st.sampled_from(pool), _anchors(_SEGMENT)))
    rules = "".join(f"  - name: {name}\n    kind: parameter\n" for name in defined)
    with tempfile.TemporaryDirectory() as temporary:
        module = _write(Path(temporary) / "m.yaml", "rules:\n" + rules)
        test_file = _write(Path(temporary) / "m.test.yaml", _ref_text(refs))
        violations = validate_generated_against_registry(
            [module, test_file], registry, apply_anchor=apply_anchor
        )
    conflicts = {v.name for v in violations if v.kind == "canonical_conflict"}
    assert conflicts == {
        name
        for name in defined
        if drawn[name].produces and not drawn[name].lists(apply_anchor)
    }

    def flagged(kind):
        return {
            v.where.split(".test.yaml:", 1)[1] for v in violations if v.kind == kind
        }

    expected_misses = set()
    expected_synonyms = set()
    for anchor, is_input, name in refs:
        truth = drawn.get(name)
        if truth is None:
            continue
        if name != truth.canonical_name:
            if not _keeps_synonym(truth, anchor, is_input, apply_anchor):
                expected_synonyms.add(f"{anchor}#{name}")
        elif not is_input and truth.produces and not truth.lists(anchor):
            expected_misses.add(f"{anchor}#{name}")
    assert flagged("anchored_ref_miss") == expected_misses
    assert flagged("blocked_synonym") == expected_synonyms


@PROPERTY_SETTINGS
@given(_registry_and_refs())
def test_property_repair_then_validate_flags_exactly_the_ambiguous_vintage_refs(
    case,
):
    _pool, registry, drawn, refs, _apply_anchor = case
    with tempfile.TemporaryDirectory() as temporary:
        test_file = _write(Path(temporary) / "m.test.yaml", _ref_text(refs))
        auto_repair_test_yaml_canonical_violations([test_file], registry)
        violations = validate_generated_against_registry([test_file], registry)
    assert not [v for v in violations if v.kind == "blocked_synonym"]
    flagged = {
        v.where.split(".test.yaml:", 1)[1]
        for v in violations
        if v.kind == "anchored_ref_miss"
    }
    expected = set()
    for anchor, is_input, name in refs:
        truth = drawn.get(name)
        if (
            truth is not None
            and not is_input
            and truth.produces
            and len(truth.anchors) > 1
            and not truth.lists(anchor)
        ):
            expected.add(f"{anchor}#{truth.canonical_name}")
    assert flagged == expected


@PROPERTY_SETTINGS
@given(_registry_and_refs(single_anchor_only=True), st.data())
def test_property_single_producer_registries_match_main_branch_semantics(case, data):
    pool, registry, drawn, refs, apply_anchor = case
    text = _ref_text(refs)
    assert _rewrite_anchored_refs(
        text, registry, apply_anchor=apply_anchor
    ) == legacy_rewrite_anchored_refs(text, registry, apply_anchor=apply_anchor)

    canonicals = sorted({truth.canonical_name for truth in drawn.values()})
    defined = data.draw(st.lists(st.sampled_from(canonicals), unique=True))
    synonyms = data.draw(
        st.lists(st.sampled_from(sorted(registry.synonym_to_concept) or ["x"]))
    )
    rules = "".join(
        f"  - name: {name}\n    kind: derived\n    versions:\n"
        f"      - formula: {' + '.join(synonyms) or '0'}\n"
        for name in defined + synonyms
    )
    with tempfile.TemporaryDirectory() as temporary:
        module = _write(Path(temporary) / "m.yaml", "rules:\n" + rules)
        test_file = _write(Path(temporary) / "m.test.yaml", text)
        files = [module, test_file]
        current = validate_generated_against_registry(
            files, registry, apply_anchor=apply_anchor
        )
        legacy = legacy_validate_generated_against_registry(
            files, registry, apply_anchor=apply_anchor
        )
    assert current == legacy


def _guidance_for(registry, source_text: str, **kwargs) -> str:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        workspace = EvalWorkspace(
            root=root,
            source_text_file=_write(root / "source.txt", source_text),
            manifest_file=_write(root / "context-manifest.json", "{}"),
        )
        return _format_canonical_concept_registry_guidance(
            source_text, workspace, [], registry=registry, **kwargs
        )


@PROPERTY_SETTINGS
@given(_registries(single_anchor_only=True), st.data())
def test_property_single_producer_prompt_guidance_matches_main_branch(case, data):
    """With one producer per concept the prompt is main's, whatever the target."""
    pool, registry, drawn = case
    words = data.draw(st.lists(st.sampled_from([*sorted(drawn), "unrelated"])))
    source_text = " ".join(words)
    target = data.draw(st.one_of(st.none(), st.sampled_from(pool), _anchors(_SEGMENT)))
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        workspace = EvalWorkspace(
            root=root,
            source_text_file=_write(root / "source.txt", source_text),
            manifest_file=_write(root / "context-manifest.json", "{}"),
        )
        current = _format_canonical_concept_registry_guidance(
            source_text, workspace, [], registry=registry, target_anchor=target
        )
        legacy = legacy_format_canonical_concept_registry_guidance(
            source_text, workspace, [], registry=registry
        )
    assert current == legacy


_BACKTICKED_ANCHOR_RE = re.compile(r"`([a-z][a-z0-9-]*:[^`]+)`")


@PROPERTY_SETTINGS
@given(_registries(), st.data())
def test_property_prompt_names_each_vintage_once_and_the_targets_role(case, data):
    """A concept's prompt line names exactly its accepted producers, each once
    with its registered period, and calls the module being encoded a producer
    iff it is one of them."""
    pool, registry, drawn = case
    target = data.draw(st.one_of(st.none(), st.sampled_from(pool), _anchors(_SEGMENT)))
    truths = {truth.canonical_name: truth for truth in drawn.values()}
    section = _guidance_for(registry, " ".join(sorted(truths)), target_anchor=target)
    for name, truth in truths.items():
        line = _line_for(section, name)
        named = _BACKTICKED_ANCHOR_RE.findall(line)
        if not truth.produces:
            assert named == []
            continue
        assert sorted(named) == sorted(truth.anchors)
        for period in truth.periods:
            assert period.label in line
            assert period.effective_from.isoformat() in line
            assert period.effective_to.isoformat() in line
        if len(truth.anchors) == 1:
            assert f"producer `{truth.anchors[0]}`" in line
            continue
        assert (f"this module, `{target}`, is " in line) is truth.lists(target)
        assert ("so do not define a rule with this name here" in line) is (
            target is not None and not truth.lists(target)
        )


@PROPERTY_SETTINGS
@given(_registries())
def test_property_registry_yaml_round_trip_preserves_producers(case):
    _pool, registry, _drawn = case
    payload = {
        "format": REGISTRY_FORMAT,
        "concepts": [
            {
                "id": c.id,
                "canonical_name": c.canonical_name,
                **({"producer_anchor": c.producer_anchor} if c.producer_anchor else {}),
                **(
                    {"producer_anchors": list(c.producer_anchors)}
                    if c.producer_anchors
                    else {}
                ),
                **(
                    {
                        "producer_periods": [
                            {
                                "anchor": p.anchor,
                                "label": p.label,
                                "effective_from": p.effective_from,
                                "effective_to": p.effective_to,
                            }
                            for p in c.producer_periods
                        ]
                    }
                    if c.producer_periods
                    else {}
                ),
                "producer_missing": c.producer_missing,
                "blocked_synonyms": list(c.blocked_synonyms),
            }
            for c in registry.concepts_by_id.values()
        ],
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "c.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
        loaded = load_concept_registry(root)
    for concept in registry.concepts_by_id.values():
        again = loaded.concepts_by_id[concept.id]
        assert again.producer_anchors == concept.producer_anchors
        assert again.producer_anchor == concept.producer_anchor
        assert again.has_producer == concept.has_producer
        assert again.producer_periods == concept.producer_periods
        assert all(PRODUCER_ANCHOR_RE.fullmatch(a) for a in again.producer_anchors)


@PROPERTY_SETTINGS
@given(st.data())
def test_property_loader_accepts_periods_iff_well_formed_and_disjoint(data):
    """Periods load iff each is ordered and no two share a day; once loaded,
    every date is covered by at most one producer and periods follow the
    producer_anchors order."""
    anchors = data.draw(
        st.lists(_anchors(_SAFE_SEGMENT), min_size=1, max_size=4, unique=True)
    )
    bounds = data.draw(
        st.lists(
            st.tuples(st.integers(0, 40), st.integers(0, 40)),
            min_size=len(anchors),
            max_size=len(anchors),
        )
    )
    order = data.draw(st.permutations(range(len(anchors))))
    entries = [
        {
            "anchor": anchors[i],
            "label": f"V{i}",
            "effective_from": (_EPOCH + timedelta(days=bounds[i][0])).isoformat(),
            "effective_to": (_EPOCH + timedelta(days=bounds[i][1])).isoformat(),
        }
        for i in order
    ]
    ordered = all(start <= end for start, end in bounds)
    disjoint = all(
        max(bounds[i][0], bounds[j][0]) > min(bounds[i][1], bounds[j][1])
        for i in range(len(bounds))
        for j in range(i + 1, len(bounds))
    )
    payload = {
        "format": REGISTRY_FORMAT,
        "concepts": [
            {
                "id": "t.vintaged",
                "canonical_name": "vintaged_amount",
                "producer_anchors": anchors,
                "producer_periods": entries,
            }
        ],
    }
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "c.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
        if not (ordered and disjoint):
            with pytest.raises(ValueError):
                load_concept_registry(root)
            return
        concept = load_concept_registry(root).lookup_canonical("vintaged_amount")
    assert [p.anchor for p in concept.producer_periods] == anchors
    for offset in range(0, 42):
        day = _EPOCH + timedelta(days=offset)
        covering = [
            p
            for p in concept.producer_periods
            if p.effective_from <= day <= p.effective_to
        ]
        assert len(covering) <= 1


@settings(
    max_examples=60,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(st.data())
def test_property_audit_conflicts_are_exactly_unlisted_producer_sites(data):
    us_anchors = st.builds(
        lambda root, parts: f"us:{root}/" + "/".join(parts),
        st.sampled_from(("policies", "regulations", "statutes")),
        st.lists(_SAFE_SEGMENT, min_size=1, max_size=3),
    )
    pool = data.draw(st.lists(us_anchors, min_size=1, max_size=5, unique=True))
    accepted = tuple(
        data.draw(st.lists(st.sampled_from(pool), min_size=1, max_size=3, unique=True))
    )
    concept = Concept(
        id="t.vintaged",
        canonical_name="vintaged_amount",
        producer_anchor=None,
        producer_anchors=accepted,
    )
    registry = ConceptRegistry(
        concepts_by_id={concept.id: concept},
        canonical_to_concept={concept.canonical_name: concept},
        synonym_to_concept={},
    )
    producers = data.draw(st.lists(st.sampled_from(pool), min_size=1, unique=True))
    # A file path cannot also be a directory: skip anchor sets that nest.
    paths = [a.split(":", 1)[1] for a in producers]
    assume(not any(p != q and q.startswith(p + "/") for p in paths for q in paths))
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve() / "rulespec-us"
        for rel in paths:
            _write(
                root / "us" / f"{rel}.yaml",
                "rules:\n  - name: vintaged_amount\n    kind: parameter\n",
            )
        findings = [
            f for f in audit_corpus([root], registry) if f.kind == "canonical_conflict"
        ]
        expected_sites = {
            root / "us" / f"{a.split(':', 1)[1]}.yaml"
            for a in producers
            if a not in accepted
        }
    if expected_sites:
        assert len(findings) == 1
        assert set(findings[0].site_paths) == expected_sites
        assert findings[0].accepted_producers == accepted
    else:
        assert findings == []
