"""Vintaged producers in the canonical-concept registry.

The SNAP cost-of-living concepts have one producer module per fiscal year
(rulespec-us#1394, #759). Before this change the registry accepted exactly one
producer anchor, `fy-2026-cola/maximum-allotments`, so an FY2027 producer was
refused with `canonical_conflict` and test auto-repair silently re-anchored
FY2027 (and FY2024) output assertions to the FY2026 module.

Invariants exercised here (example tests first, then Hypothesis properties):

1. Acceptance: a module may define a registered canonical, and a non-input
   reference may target it, iff its anchor is one of the concept's
   `producer_anchors`.
2. Auto-repair never moves a reference whose anchor is an accepted producer,
   never moves an input reference, and redirects an anchor only to the
   concept's single producer; with several vintages it leaves the anchor
   alone (no guessing) and validation names every accepted producer.
3. Auto-repair is idempotent.
4. For registries in which every concept has at most one producer, the
   validator and auto-repair agree exactly with the single-anchor
   implementation from origin/main 436cf3044 (differential oracle in
   tests/concepts_single_anchor_reference.py).
5. The corpus audit reports a canonical producer as a conflict iff its anchor
   is not an accepted producer, and reports every accepted producer.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
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
    load_concept_registry,
)
from axiom_encode.concepts.validator import (
    ANCHORED_REF_RE,
    validate_generated_against_registry,
)
from axiom_encode.harness.evals import (
    EvalWorkspace,
    _format_canonical_concept_registry_guidance,
)
from axiom_encode.prepare_signed_backfill import citation_rulespec_path
from tests.concepts_single_anchor_reference import (
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


def test_prompt_guidance_lists_every_vintage_producer(tmp_path: Path):
    source_text = "snap_maximum_allotment and snap_total_gross_income"
    source_file = _write(tmp_path / "source.txt", source_text)
    manifest = _write(tmp_path / "context-manifest.json", "{}")
    workspace = EvalWorkspace(
        root=tmp_path, source_text_file=source_file, manifest_file=manifest
    )
    section = _format_canonical_concept_registry_guidance(
        source_text, workspace, context_files=[]
    )
    allotment_line = next(
        line for line in section.splitlines() if "`snap_maximum_allotment`" in line
    )
    for vintage in COLA_VINTAGES:
        assert f"`{vintage}`" in allotment_line
    assert "one producer per vintage" in allotment_line
    assert f"producer `{FY2026}`" not in allotment_line
    assert "producer `us:regulations/7-cfr/273/10`" in section


# ---------------------------------------------------------------------------
# Hypothesis properties
# ---------------------------------------------------------------------------

PROPERTY_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    database=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)

_SEGMENT = st.text(alphabet="abcxyz0189-._AZ", min_size=1, max_size=6)
_SAFE_SEGMENT = st.text(alphabet="abcxyz0189-", min_size=1, max_size=6)


def _anchors(segment):
    return st.builds(
        lambda jurisdiction, root, parts: f"{jurisdiction}:{root}/" + "/".join(parts),
        st.sampled_from(("us", "us-co", "us-ny")),
        st.sampled_from(("policies", "regulations", "statutes")),
        st.lists(segment, min_size=1, max_size=3),
    )


@st.composite
def _registries(draw, *, segment=_SEGMENT, single_anchor_only=False):
    """A random registry plus the anchor pool its concepts draw from."""
    pool = draw(st.lists(_anchors(segment), min_size=1, max_size=6, unique=True))
    concepts = []
    for index in range(draw(st.integers(min_value=1, max_value=4))):
        max_anchors = 1 if single_anchor_only else 3
        anchors = draw(
            st.lists(st.sampled_from(pool), max_size=max_anchors, unique=True)
        )
        if single_anchor_only:
            # The single-anchor implementation reads only producer_anchor.
            legacy = anchors[0] if anchors else None
        else:
            legacy = (
                draw(st.sampled_from(anchors))
                if anchors and draw(st.booleans())
                else None
            )
        synonyms = tuple(
            f"name_{index}_old_{k}" for k in range(draw(st.integers(0, 2)))
        )
        concepts.append(
            Concept(
                id=f"t.c{index}",
                canonical_name=f"name_{index}",
                producer_anchor=legacy,
                blocked_synonyms=synonyms,
                producer_missing=draw(st.booleans()),
                producer_anchors=tuple(anchors),
            )
        )
    registry = ConceptRegistry(
        concepts_by_id={c.id: c for c in concepts},
        canonical_to_concept={c.canonical_name: c for c in concepts},
        synonym_to_concept={s: c for c in concepts for s in c.blocked_synonyms},
    )
    return pool, registry


@st.composite
def _registry_and_refs(draw, *, single_anchor_only=False):
    pool, registry = draw(_registries(single_anchor_only=single_anchor_only))
    names = sorted(
        {
            *registry.canonical_to_concept,
            *registry.synonym_to_concept,
            "unregistered_name",
        }
    )
    anchor_choices = st.one_of(st.sampled_from(pool), _anchors(_SEGMENT))
    refs = draw(
        st.lists(
            st.tuples(anchor_choices, st.booleans(), st.sampled_from(names)),
            min_size=1,
            max_size=8,
        )
    )
    apply_anchor = draw(st.one_of(st.none(), st.sampled_from(pool)))
    return pool, registry, refs, apply_anchor


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


@PROPERTY_SETTINGS
@given(_registry_and_refs())
def test_property_repair_never_moves_accepted_or_input_refs_and_never_guesses(case):
    _pool, registry, refs, apply_anchor = case
    text = _ref_text(refs)
    repaired = _rewrite_anchored_refs(text, registry, apply_anchor=apply_anchor)
    after = _parse_lines(repaired)
    assert len(after) == len(refs)
    for (anchor, is_input, name), (new_anchor, new_input, new_name) in zip(refs, after):
        assert new_input == is_input
        concept = registry.concept_for_name(name)
        if concept is None:
            assert (new_anchor, new_name) == (anchor, name)
            continue
        assert new_name in {name, concept.canonical_name}
        if is_input or concept.accepts_producer_anchor(anchor):
            assert new_anchor == anchor
        if len(concept.producer_anchors) > 1:
            assert new_anchor == anchor  # never chooses a vintage
        if new_anchor != anchor:
            assert not is_input
            assert not concept.accepts_producer_anchor(anchor)
            assert new_anchor == concept.unique_producer_anchor


@PROPERTY_SETTINGS
@given(_registry_and_refs())
def test_property_repair_is_idempotent(case):
    _pool, registry, refs, apply_anchor = case
    once = _rewrite_anchored_refs(_ref_text(refs), registry, apply_anchor=apply_anchor)
    twice = _rewrite_anchored_refs(once, registry, apply_anchor=apply_anchor)
    assert twice == once


@PROPERTY_SETTINGS
@given(_registry_and_refs(), st.data())
def test_property_validation_accepts_iff_listed_producer(case, data):
    pool, registry, refs, _apply_anchor = case
    concepts = list(registry.concepts_by_id.values())
    defined = data.draw(st.lists(st.sampled_from(concepts), unique_by=lambda c: c.id))
    apply_anchor = data.draw(st.one_of(st.sampled_from(pool), _anchors(_SEGMENT)))
    rules = "".join(
        f"  - name: {c.canonical_name}\n    kind: parameter\n" for c in defined
    )
    with tempfile.TemporaryDirectory() as temporary:
        module = _write(Path(temporary) / "m.yaml", "rules:\n" + rules)
        test_file = _write(Path(temporary) / "m.test.yaml", _ref_text(refs))
        violations = validate_generated_against_registry(
            [module, test_file], registry, apply_anchor=apply_anchor
        )
    conflicts = {v.name for v in violations if v.kind == "canonical_conflict"}
    assert conflicts == {
        c.canonical_name
        for c in defined
        if c.has_producer and not c.accepts_producer_anchor(apply_anchor)
    }
    misses = {
        v.where.split(".test.yaml:", 1)[1]
        for v in violations
        if v.kind == "anchored_ref_miss"
    }
    expected_misses = set()
    for anchor, is_input, name in refs:
        concept = registry.lookup_canonical(name)
        if (
            concept is not None
            and not is_input
            and concept.has_producer
            and not concept.accepts_producer_anchor(anchor)
        ):
            expected_misses.add(f"{anchor}#{name}")
    assert misses == expected_misses


@PROPERTY_SETTINGS
@given(_registry_and_refs())
def test_property_repair_then_validate_flags_exactly_the_ambiguous_vintage_refs(
    case,
):
    _pool, registry, refs, _apply_anchor = case
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
        concept = registry.concept_for_name(name)
        if (
            concept is not None
            and not is_input
            and concept.has_producer
            and len(concept.producer_anchors) > 1
            and not concept.accepts_producer_anchor(anchor)
        ):
            expected.add(f"{anchor}#{concept.canonical_name}")
    assert flagged == expected


@PROPERTY_SETTINGS
@given(_registry_and_refs(single_anchor_only=True), st.data())
def test_property_single_producer_registries_match_main_branch_semantics(case, data):
    pool, registry, refs, apply_anchor = case
    text = _ref_text(refs)
    assert _rewrite_anchored_refs(
        text, registry, apply_anchor=apply_anchor
    ) == legacy_rewrite_anchored_refs(text, registry, apply_anchor=apply_anchor)

    concepts = list(registry.concepts_by_id.values())
    defined = data.draw(st.lists(st.sampled_from(concepts), unique_by=lambda c: c.id))
    synonyms = data.draw(
        st.lists(st.sampled_from(sorted(registry.synonym_to_concept) or ["x"]))
    )
    rules = "".join(
        f"  - name: {name}\n    kind: derived\n    versions:\n"
        f"      - formula: {' + '.join(synonyms) or '0'}\n"
        for name in [c.canonical_name for c in defined] + synonyms
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


@PROPERTY_SETTINGS
@given(_registries())
def test_property_registry_yaml_round_trip_preserves_producers(case):
    _pool, registry = case
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
        assert all(PRODUCER_ANCHOR_RE.fullmatch(a) for a in again.producer_anchors)


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
