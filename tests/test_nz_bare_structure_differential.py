"""Check the evidence used to certify the corpus differential."""

import re
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode.bare_structure import bare_outline, mask_bare_labels
from axiom_encode.harness.source_completeness import SourceStructureBranch

SCRIPT = Path(__file__).parents[1] / "scripts" / "nz_bare_structure_differential.py"
SPEC = spec_from_file_location("_nz_bare_differential_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def legacy_numeric_line_validator():
    return SimpleNamespace(
        _STRUCTURAL_SOURCE_QUOTE_CHARS="\"'`“”‘’",
        _STRUCTURAL_SOURCE_LINE_PATTERN=re.compile(r"\d+[A-Za-z]?"),
    )


def source_numeric_record(raw, value, start):
    return {
        "raw": raw,
        "value": value,
        "source_value": value,
        "alternative_values": (),
        "is_word_number": False,
        "has_rate_context": False,
        "has_temporal_context": False,
        "has_structural_context": False,
        "requires_rate_context": False,
        "start": start,
        "end": start + len(raw),
        "source_start": start,
        "source_end": start + len(raw),
    }


@pytest.fixture(scope="module")
def legacy_validator_for_counterfactual():
    return AUDIT.load_baseline(
        "86a1f911a",
        "src/axiom_encode/harness/validator_pipeline.py",
        "axiom_encode.harness._nz_counterfactual_test_baseline",
    )


def grouped_restoration_case(source, baseline, raw, *, recall=None, target_start=None):
    recall = source if recall is None else recall
    extract = baseline.extract_typed_numeric_inventory_occurrences_from_text
    old = tuple(AUDIT.anchored_occurrences(source, recall, extract(recall)).values())
    new = AUDIT.anchored_occurrences(
        source,
        recall,
        AUDIT.current_validator.extract_typed_numeric_inventory_occurrences_from_text(
            recall
        ),
    )
    added = next(
        item
        for item in new.values()
        if item["raw"] == raw
        and (target_start is None or item["source_start"] == target_start)
    )
    assert not any(
        (item["source_start"], item["source_end"])
        == (added["source_start"], added["source_end"])
        for item in old
    )
    evidence = AUDIT.cross_occurrence_restoration_evidence(
        added, source, recall, old, baseline, extract
    )
    assert evidence
    assert evidence["counterfactual_only_suppressors_masked"]
    assert AUDIT.numeric_requirement_identity(
        evidence["counterfactual_restored_occurrence"]
    ) == AUDIT.numeric_requirement_identity(added)
    return old, added, evidence


def test_real_gst_s8_replay_uses_exact_pre_filter_collector_snapshot(
    legacy_validator_for_counterfactual,
):
    source = (
        Path(__file__).parent / "fixtures" / "nz_bare_labels" / "gst_s8.txt"
    ).read_text()
    baseline_source = AUDIT.load_baseline(
        "86a1f911a",
        "src/axiom_encode/harness/source_completeness.py",
        "axiom_encode.harness._nz_counterfactual_source_test_baseline",
    )
    recall = baseline_source.authoritative_numeric_recall_text(
        source, corpus_citation_path="nz/statute/act/public/1985/0141/section/8"
    )

    old, added, evidence = grouped_restoration_case(
        source, legacy_validator_for_counterfactual, "1,000", recall=recall
    )

    assert (added["source_start"], added["source_end"]) == (6321, 6326)
    assert [item["source_start"] for item in old if item["value"] == 1] == [
        2299,
        8266,
    ]
    snapshot = evidence["collector_snapshot"]
    assert snapshot["traced_final_matches_normal"]
    assert snapshot["stage"]["line"] == 16790
    assert snapshot["stage"]["compiled_without_inherited_flags"]
    assert len(snapshot["stage"]["runtime_code_sha256"]) == 64
    suppressors = evidence["distinct_prior_suppressors"]
    assert [item["source_start"] for item in suppressors] == [253, 1071, 2299]
    assert [item["has_temporal_context"] for item in suppressors] == [
        True,
        True,
        False,
    ]
    assert evidence["counterfactual_masked_recall_spans"] == [
        [253, 254],
        [1061, 1062],
        [2289, 2290],
    ]
    assert all(item["raw"] == "1" for item in suppressors)
    assert all(item["physical_relation"] == "before" for item in suppressors)


def test_real_calendar_and_count_suppressors_restore_same_grouped_amount(
    legacy_validator_for_counterfactual,
):
    source = (
        "The tax applies from 1 October 1986 and 1 July 2007. "
        "Packaging contains 1 or more items. "
        "The value of the supply is less than or equal to $1,000. "
        "If more than 1 person is responsible, all must apply."
    )

    old, added, evidence = grouped_restoration_case(
        source, legacy_validator_for_counterfactual, "1,000"
    )

    assert len(old) == 2
    snapshot = evidence["collector_snapshot"]
    assert snapshot["traced_final_matches_normal"]
    assert [item["source_start"] for item in snapshot["occurrences"]] == [
        source.index("1 October"),
        source.index("1 July"),
        source.index("1 or more"),
    ]
    assert evidence["counterfactual_restored_occurrence"][
        "source_start"
    ] == source.index("1,000")
    assert added["value"] == 1000


def test_real_physically_later_duration_was_collected_before_grouped_match(
    legacy_validator_for_counterfactual,
):
    source = "The amount is $24,000.\n2\nAn application is needed."

    _old, _added, evidence = grouped_restoration_case(
        source, legacy_validator_for_counterfactual, "24,000"
    )

    assert evidence["collector_snapshot"]["traced_final_matches_normal"]
    suppressor = evidence["distinct_prior_suppressors"][0]
    assert suppressor["raw"] == "2\nAn"
    assert suppressor["physical_relation"] == "after"
    assert suppressor["suppression_provenance"]["kind"] == "_YEAR_DURATION_PATTERN"
    start = source.index("2\nAn")
    assert evidence["counterfactual_masked_recall_spans"] == [[start, start + 1]]


@pytest.mark.parametrize("count", (1, 2))
def test_real_baseline_replay_restores_cross_occurrence_grouped_amount(
    count, legacy_validator_for_counterfactual
):
    source = f"There are {count} persons. The amount is ${count},000."
    baseline = legacy_validator_for_counterfactual
    old_extractor = baseline.extract_typed_numeric_inventory_occurrences_from_text
    old = tuple(
        AUDIT.anchored_occurrences(source, source, old_extractor(source)).values()
    )
    new = AUDIT.anchored_occurrences(
        source,
        source,
        AUDIT.current_validator.extract_typed_numeric_inventory_occurrences_from_text(
            source
        ),
    )
    added = next(item for item in new.values() if item["value"] == count * 1000)
    assert not any(item["value"] == count * 1000 for item in old)

    evidence = AUDIT.cross_occurrence_restoration_evidence(
        added, source, source, old, baseline, old_extractor
    )

    assert evidence
    assert evidence["counterfactual_only_suppressors_masked"]
    assert (
        evidence["counterfactual_restored_occurrence"]["source_start"]
        == added["source_start"]
    )


def test_real_duration_suppressor_replay_preserves_article_and_grouped_amount(
    legacy_validator_for_counterfactual,
):
    source = "2\nAn application is needed. The amount is $24,000."
    baseline = legacy_validator_for_counterfactual
    extract = baseline.extract_typed_numeric_inventory_occurrences_from_text
    old = tuple(AUDIT.anchored_occurrences(source, source, extract(source)).values())
    new = AUDIT.anchored_occurrences(
        source,
        source,
        AUDIT.current_validator.extract_typed_numeric_inventory_occurrences_from_text(
            source
        ),
    )
    added = next(item for item in new.values() if item["value"] == 24000.0)
    replayed = []

    def replay(counterfactual):
        replayed.append(counterfactual)
        return extract(counterfactual)

    evidence = AUDIT.cross_occurrence_restoration_evidence(
        added, source, source, old, baseline, replay
    )

    assert evidence
    assert replayed == [source, " " + source[1:]]
    assert evidence["counterfactual_masked_recall_spans"] == [[0, 1]]
    provenance = evidence["distinct_prior_suppressors"][0]["suppression_provenance"]
    assert provenance["kind"] == "_YEAR_DURATION_PATTERN"
    assert provenance["operand"] == 2.0
    assert provenance["multiplier"] == 12
    assert provenance["normalized_value"] == 24.0
    assert replayed[-1][1:4] == "\nAn"


@pytest.mark.parametrize(
    "mutation",
    (
        "wrong-value",
        "wrong-source-value",
        "wrong-converter",
        "wrong-multiplier",
        "partial-constituent",
        "unaligned-span",
        "non-preserved-An",
        "wrong-flags",
    ),
)
def test_duration_suppressor_requires_exact_converter_and_preserved_unit(mutation):
    source = "2\nAn application is needed. The amount is $24,000."
    if mutation == "partial-constituent":
        source = "1" + source
    value = 14.0 if mutation == "wrong-multiplier" else 24.0
    if value == 14:
        source = source.replace("24,000", "14,000")
    old = source_numeric_record("2\nAn", value, source.index("2\nAn"))
    original = dict(old)
    raw = "14,000" if value == 14 else "24,000"
    added = source_numeric_record(raw, value * 1000, source.index(raw))
    baseline = SimpleNamespace(
        _YEAR_DURATION_PATTERN=re.compile(r"(?P<number>\d+)\nAn"),
        _WEEK_DURATION_PATTERN=re.compile(r"(?P<number>\d+) weeks?"),
        _parse_belgian_numeric_phrase=lambda raw: float(raw.replace(",", ".")),
    )
    if mutation == "wrong-value":
        old["value"] = 23.0
    elif mutation == "wrong-source-value":
        old["source_value"] = 2.0
    elif mutation == "wrong-converter":
        baseline._YEAR_DURATION_PATTERN = re.compile(r"(?P<number>\d+) years?")
        baseline._WEEK_DURATION_PATTERN = re.compile(r"(?P<number>\d+)\nAn")
    elif mutation == "unaligned-span":
        old["source_start"] += 1
    elif mutation == "non-preserved-An":
        baseline._YEAR_DURATION_PATTERN = re.compile(r"(?P<number>2\nAn)")
        baseline._parse_belgian_numeric_phrase = lambda raw: (
            2.0 if raw == "2\nAn" else float(raw.replace(",", "."))
        )
    elif mutation == "wrong-flags":
        old["has_temporal_context"] = True

    def occurrence(record):
        return SimpleNamespace(
            **{
                key: value
                for key, value in record.items()
                if key not in {"source_start", "source_end"}
            }
        )

    def replay(counterfactual):
        if counterfactual == source:
            return (occurrence(original),)
        assert counterfactual == " " + source[1:]
        return (occurrence(added),)

    evidence = AUDIT.cross_occurrence_restoration_evidence(
        added, source, source, (old,), baseline, replay
    )

    assert evidence is None


@pytest.mark.parametrize("count", (1, 2))
def test_cross_occurrence_restoration_replays_only_distinct_prior_suppressors(count):
    source = (
        f"There are {count} persons and {count} members. The amount is ${count},000."
    )
    first = source_numeric_record(str(count), float(count), source.index(str(count)))
    second = source_numeric_record(
        str(count), float(count), source.index(str(count), first["end"])
    )
    raw = f"{count},000"
    added = source_numeric_record(raw, float(count * 1000), source.index(raw))
    baseline = SimpleNamespace(_parse_belgian_numeric_phrase=lambda raw: float(count))
    replayed = []

    def replay(counterfactual):
        expected = list(source)
        for old in (first, second):
            expected[old["start"] : old["end"]] = " " * len(old["raw"])
        assert counterfactual == "".join(expected)
        replayed.append(counterfactual)
        return (
            SimpleNamespace(
                **{
                    key: value
                    for key, value in added.items()
                    if not key.startswith("source_")
                },
                source_value=added["source_value"],
            ),
        )

    evidence = AUDIT.cross_occurrence_restoration_evidence(
        added, source, source, (first, second), baseline, replay
    )

    assert evidence
    assert replayed
    assert evidence["legacy_alternate_reading"] == count
    assert evidence["counterfactual_only_suppressors_masked"]
    assert len(evidence["distinct_prior_suppressors"]) == 2
    assert (
        evidence["counterfactual_restored_occurrence"]["source_start"]
        == added["source_start"]
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "partial-group",
        "wrong-value",
        "wrong-source-value",
        "no-suppressor",
        "same-span-reading",
        "unverified-suppressor",
        "partial-suppressor",
        "replay-wrong-semantics",
        "replay-wrong-span",
    ),
)
def test_cross_occurrence_restoration_rejects_unproven_numeric_changes(mutation):
    source = "There are 2 persons. The amount is $2,000."
    if mutation == "partial-suppressor":
        source = source.replace("2 persons", "12 persons")
    first_start = source.index("2")
    old = source_numeric_record("2", 2.0, first_start)
    added = source_numeric_record("2,000", 2000.0, source.index("2,000"))
    old_numbers = (old,)
    replay_record = dict(added)
    if mutation == "partial-group":
        added.update(raw="2,00", source_end=added["source_end"] - 1)
    elif mutation == "wrong-value":
        added["value"] = 2.0
    elif mutation == "wrong-source-value":
        added["source_value"] = 2.0
    elif mutation == "no-suppressor":
        old_numbers = ()
    elif mutation == "same-span-reading":
        old_numbers = (old, {**added, "value": 2.0, "source_value": 2.0})
    elif mutation == "unverified-suppressor":
        old["source_start"] += 1
    elif mutation == "replay-wrong-semantics":
        replay_record["has_rate_context"] = True
    elif mutation == "replay-wrong-span":
        replay_record["start"] += 1
        replay_record["end"] += 1
    baseline = SimpleNamespace(_parse_belgian_numeric_phrase=lambda raw: 2.0)

    def replay(counterfactual):
        return (
            SimpleNamespace(
                **{
                    key: value
                    for key, value in replay_record.items()
                    if key not in {"source_start", "source_end"}
                }
            ),
        )

    evidence = AUDIT.cross_occurrence_restoration_evidence(
        added, source, source, old_numbers, baseline, replay
    )

    assert evidence is None


@pytest.mark.parametrize(
    ("source", "boundary"),
    (
        (
            "a\nDespite subsections (2) and (2BA), if approval is given, payment applies.\n"
            "b\nAnother payment applies.",
            "2BA",
        ),
        (
            "a\nThe approval continues under section 15EB(2).\n"
            "b\nAnother approval applies.",
            "15EB",
        ),
        (
            "Under section 55(1AM), the following records must be kept—\n"
            "a\nthe name; and\nb\nthe address.",
            "1AM",
        ),
    ),
)
def test_citation_interior_boundaries_gain_actual_bare_owners(source, boundary):
    outline = bare_outline(source)
    assert outline is not None
    masked = mask_bare_labels(source, outline)
    branches = AUDIT.current.recognize_source_structure(source)
    if boundary == "1AM":
        start, end = 0, source.index(boundary)
        replacement_start, replacement_end = 0, outline[0].start - 1
    else:
        start, end = source.index(boundary), outline[1].start - 1
        replacement_start, replacement_end = 2, end
    removed = SourceStructureBranch(
        (), "excerpt", "source clause", source[start:end], start, end
    )
    replacement = SourceStructureBranch(
        (),
        "excerpt",
        "source clause",
        masked[replacement_start:replacement_end],
        replacement_start,
        replacement_end,
    )

    evidence = AUDIT.structural_removal_evidence(
        removed,
        (replacement,),
        (replacement,),
        source,
        outline,
        "clauses",
        new_gates={"branches": branches},
    )

    assert evidence["structural_artifact"]
    proof = evidence["citation_boundary_reownership"]
    assert proof["citation_boundaries"]
    assert proof["source_aligned_supersets"]


@pytest.mark.parametrize("mutation", ("partial", "lossy", "dropped", "no-owner"))
def test_citation_reownership_rejects_missing_same_gate_text_or_owner(mutation):
    source = "a\nThe amount under section 15EB(2) is 450.\nb\nAnother amount is 700."
    outline = bare_outline(source)
    assert outline is not None
    masked = mask_bare_labels(source, outline)
    start, end = source.index("15EB"), outline[1].start - 1
    removed = SourceStructureBranch(
        ("satz-15",), "sentence", "Satz 15", source[start:end], start, end
    )
    full = SourceStructureBranch(
        ("a",), "letter", "(a)", masked[: outline[0].end], 0, outline[0].end
    )
    replacements = (full,)
    if mutation == "partial":
        replacements = (
            SourceStructureBranch(
                full.path, full.kind, full.label, masked[: end - 1], 0, end - 1
            ),
        )
    elif mutation == "lossy":
        replacements = (
            SourceStructureBranch(
                full.path,
                full.kind,
                full.label,
                full.text.replace("450", "   "),
                full.start,
                full.end,
            ),
        )
    elif mutation == "dropped":
        replacements = ()
    clause = SourceStructureBranch(
        (), "excerpt", "source clause", masked, 0, len(source)
    )
    branches = (
        ()
        if mutation == "no-owner"
        else AUDIT.current.recognize_source_structure(source)
    )

    evidence = AUDIT.structural_removal_evidence(
        removed,
        replacements,
        (clause,),
        source,
        outline,
        "branches",
        new_gates={"branches": branches},
    )

    assert not evidence["structural_artifact"]
    assert evidence["citation_boundary_reownership"] is None


def test_citation_reownership_cannot_waive_a_lost_numeric_requirement():
    source = (
        "a\nUnder section 15EB(2), if income exceeds 450, payment applies.\nb\nOther."
    )
    outline = bare_outline(source)
    assert outline is not None
    masked = mask_bare_labels(source, outline)
    start, end = source.index("15EB"), outline[1].start - 1
    removed = SourceStructureBranch(
        ("satz-15",), "exception-clause", "Satz 15", source[start:end], start, end
    )
    replacement = SourceStructureBranch(
        ("a",), "exception-clause", "(a)", masked[2:end], 2, end
    )
    requirement = {
        "raw": "450",
        "value": 450.0,
        "source_start": source.index("450"),
        "source_end": source.index("450") + 3,
    }
    bound = AUDIT.BoundaryGateBranch(
        removed.path, "numeric-boundary", "bound", removed.text, start, end, requirement
    )
    clause = SourceStructureBranch(
        (), "excerpt", "source clause", masked, 0, len(source)
    )
    old = {"tests_boundaries": (bound,), "_condition_cues": ()}
    new = {
        "branches": AUDIT.current.recognize_source_structure(source),
        "tests_boundaries": (),
        "control": (replacement,),
    }

    evidence = AUDIT.structural_removal_evidence(
        removed,
        (replacement,),
        (clause,),
        source,
        outline,
        "control",
        old_gates=old,
        new_gates=new,
    )

    assert evidence["citation_boundary_reownership"]
    assert not evidence["structural_artifact"]
    assert evidence["requirement_identity_proof"]["missing_boundaries"] == [requirement]


@pytest.mark.parametrize(
    ("source", "raw"),
    (("10", "10"), (" \t“10A”  ", "10"), (" \t“10A”  ", "10A")),
)
def test_numeric_restoration_certifies_full_operand_with_quotes_and_suffix(source, raw):
    start = source.index("10")
    item = {
        "source_start": start,
        "source_end": start + len(raw),
        "raw": raw,
        "value": 10.0,
        "source_value": 10.0,
        "alternative_values": (),
    }

    evidence = AUDIT.numeric_restoration_record(
        item, source, (), legacy_numeric_line_validator()
    )

    assert evidence["previously_blanket_excluded_numeric_line"]
    assert evidence["restored_numeric_constituent"] == {
        "raw": "10",
        "source_start": start,
        "source_end": start + 2,
        "value": 10.0,
        "suffix": "A" if "A" in source else "",
    }


@pytest.mark.parametrize(
    "change",
    (
        {"source_end": 1, "raw": "1", "value": 1.0, "source_value": 1.0},
        {"source_start": 1, "raw": "0", "value": 0.0, "source_value": 0.0},
        {"value": 1.0},
        {"source_value": 1.0},
        {"alternative_values": (1.0,)},
        {"raw": "11"},
    ),
)
def test_numeric_restoration_rejects_partial_tokens_and_changed_operands(change):
    item = {
        "source_start": 0,
        "source_end": 2,
        "raw": "10",
        "value": 10.0,
        "source_value": 10.0,
        "alternative_values": (),
        **change,
    }

    evidence = AUDIT.numeric_restoration_record(
        item, "10", (), legacy_numeric_line_validator()
    )

    assert not evidence["previously_blanket_excluded_numeric_line"]
    assert evidence["restored_numeric_constituent"] is None


def test_numeric_restoration_requires_correct_value_without_source_value_provenance():
    item = {
        "source_start": 0,
        "source_end": 2,
        "raw": "10",
        "value": 1.0,
    }

    evidence = AUDIT.numeric_restoration_record(
        item, "10", (), legacy_numeric_line_validator()
    )

    assert not evidence["previously_blanket_excluded_numeric_line"]


@pytest.mark.parametrize(
    "occurrence",
    (
        {"raw": "1", "value": 1.0, "start": 0, "end": 1},
        {"raw": "10", "value": 1.0, "start": 0, "end": 2},
        {"raw": "10", "value": 10.0, "source_value": 1.0, "start": 0, "end": 2},
    ),
)
def test_analyze_row_fails_corrupt_restored_numeric_operands(monkeypatch, occurrence):
    source = "10\nThe entitlement is payable."
    baseline_validator = legacy_numeric_line_validator()
    baseline_validator.extract_typed_numeric_inventory_occurrences_from_text = (
        lambda text, **kwargs: ()
    )
    baseline = SimpleNamespace(
        authoritative_numeric_recall_text=lambda text, **kwargs: text,
        _source_exception_or_applicability_matches=lambda text: (),
    )
    monkeypatch.setattr(
        AUDIT.current, "authoritative_numeric_recall_text", lambda text, **kwargs: text
    )
    monkeypatch.setattr(
        AUDIT.current_validator,
        "extract_typed_numeric_inventory_occurrences_from_text",
        lambda text, **kwargs: (SimpleNamespace(**occurrence),),
    )
    monkeypatch.setattr(
        AUDIT,
        "gates",
        lambda *args: {**{gate: () for gate in AUDIT.GATES}, "_condition_cues": ()},
    )
    monkeypatch.setattr(AUDIT, "slice_result", lambda *args: None)

    result = AUDIT.analyze_row(
        {
            "body": source,
            "citation_path": "nz/statute/act/public/2020/0001/section/1",
        },
        baseline,
        SimpleNamespace(),
        baseline_validator,
    )
    counts = AUDIT.Counter()
    AUDIT.add_counts(counts, result)

    assert counts["numeric_additions_without_restoration_evidence"] == 1
    assert AUDIT.failure_count(counts) > 0


@pytest.mark.parametrize(
    ("field", "before", "after"),
    (
        ("source_value", 450.0, None),
        ("alternative_values", (45.0,), ()),
        ("is_word_number", False, True),
        ("has_rate_context", True, False),
        ("has_temporal_context", True, False),
        ("has_structural_context", True, False),
        ("requires_rate_context", True, False),
    ),
)
def test_numeric_semantic_changes_survive_source_anchoring(field, before, after):
    source = "The amount is 450."
    start = source.index("450")
    base = {"raw": "450", "value": 450.0, "start": start, "end": start + 3}
    old = AUDIT.anchored_occurrences(
        source, source, (SimpleNamespace(**base, **{field: before}),)
    )
    new = AUDIT.anchored_occurrences(
        source, source, (SimpleNamespace(**base, **{field: after}),)
    )

    assert old.keys().isdisjoint(new.keys())


def test_boundary_metadata_changes_are_reported_and_cannot_be_certified():
    source = "a\nIncome must be below 450;\nb\nPayment is 700."
    outline = bare_outline(source)
    assert outline is not None
    masked = mask_bare_labels(source, outline)
    numeric_start = source.index("450")
    requirement = {
        "raw": "450",
        "value": 450.0,
        "source_start": numeric_start,
        "source_end": numeric_start + 3,
        "alternative_values": (45.0,),
    }
    before = AUDIT.BoundaryGateBranch(
        ("a",), "numeric-boundary", "bound", masked, 0, len(source), requirement
    )
    after = AUDIT.BoundaryGateBranch(
        ("a",),
        "numeric-boundary",
        "bound",
        masked,
        0,
        len(source),
        {**requirement, "alternative_values": ()},
    )
    clause = SourceStructureBranch(
        (), "excerpt", "source clause", masked, 0, len(source)
    )
    old = {gate: () for gate in AUDIT.GATES}
    old.update(tests_boundaries=(before,), clauses=(clause,), _condition_cues=())
    new = {**old, "tests_boundaries": (after,)}

    changes = AUDIT.compare_gates(old, new, source, outline)

    assert len(changes["tests_boundaries"]["added"]) == 1
    removed = changes["tests_boundaries"]["removed"]
    assert len(removed) == 1
    assert not removed[0]["evidence"]["structural_artifact"]
    assert removed[0]["evidence"]["requirement_identity_proof"][
        "missing_boundaries"
    ] == [requirement]


def test_condition_removal_retains_the_whole_physical_condition_fragment():
    source = (
        "a\nUnless income exceeds 450, the first benefit applies and—\n"
        "b\na second payment is due."
    )
    outline = bare_outline(source)
    assert outline is not None
    masked = mask_bare_labels(source, outline)
    split = source.index("b\n")
    start = source.index("Unless")
    removed = SourceStructureBranch(
        (), "exception-clause", "source unit", source, 0, len(source)
    )
    clause = SourceStructureBranch(
        (), "excerpt", "source clause", masked, 0, len(source)
    )
    complete = SourceStructureBranch(
        ("a",), "exception-clause", "(a)", masked[start:split], start, split
    )
    cue_only = SourceStructureBranch(
        ("a",), "exception-clause", "(a)", "Unless", start, start + len("Unless")
    )
    old = {"_condition_cues": ({"start": start, "end": start + len("Unless")},)}

    retained = AUDIT.structural_removal_evidence(
        removed,
        (complete,),
        (clause,),
        source,
        outline,
        "tests_conditions",
        old_gates=old,
    )
    truncated = AUDIT.structural_removal_evidence(
        removed,
        (cue_only,),
        (clause,),
        source,
        outline,
        "tests_conditions",
        old_gates=old,
    )

    assert retained["structural_artifact"]
    assert not truncated["structural_artifact"]
    assert truncated["condition_identity_proof"]["uncovered_condition_offsets"]


def test_numeric_evidence_retains_offsets_across_removed_citations_and_duplicate_values():
    source = "section 450 applies.\nThe amount is 450 per year.\n450\nThe second amount is 450 per year."
    recall = " applies.\nThe amount is 450 per year.\n450\nThe second amount is 450 per year."
    occurrences = [
        SimpleNamespace(raw="450", value=450.0, start=start, end=start + 3)
        for start in (
            recall.index("450"),
            recall.index("\n450\n") + 1,
            recall.rindex("450"),
        )
    ]

    anchored = list(AUDIT.anchored_occurrences(source, recall, occurrences).values())

    assert [item["source_start"] for item in anchored] == [
        source.index("450", source.index("The amount")),
        source.index("\n450\n") + 1,
        source.rindex("450"),
    ]
    assert all(
        source[item["source_start"] : item["source_end"]] == "450" for item in anchored
    )


@pytest.mark.parametrize(
    ("citation_token", "retained_token", "number"),
    (("32B(2)\ufeff(a)", "(2)\ufeff(a)", "2"), ("10D(1)", "(1)", "1")),
)
def test_partial_citation_tokens_retain_exact_numeric_source_offsets(
    citation_token, retained_token, number
):
    source = f"The manner required by section {citation_token} applies."
    recall = f"The manner required by {retained_token} applies."
    start = recall.index(number)
    occurrence = SimpleNamespace(
        raw=number, value=float(number), start=start, end=start + len(number)
    )

    anchored = tuple(AUDIT.anchored_occurrences(source, recall, (occurrence,)).values())

    expected = source.index(retained_token) + retained_token.index(number)
    assert anchored[0]["source_start"] == expected
    assert anchored[0]["source_end"] == expected + len(number)
    assert source[expected : expected + len(number)] == number


@pytest.mark.parametrize(
    ("source", "recall"),
    (
        ("one 10D(1) 20D(1) two", "one (1) two"),
        ("one 10D(1) two", "one (1) (1) two"),
        ("one " + "citation " * 65 + "10D(1) two", "one (1) two"),
    ),
)
def test_partial_token_anchoring_rejects_ambiguity_overlap_and_large_gaps(
    source, recall
):
    occurrences = tuple(
        SimpleNamespace(raw="1", value=1.0, start=match.start(), end=match.end())
        for match in re.finditer("1", recall)
    )

    anchored = AUDIT.anchored_occurrences(source, recall, occurrences)

    assert len(anchored) == len(occurrences)
    assert all(item["source_start"] is None for item in anchored.values())


def test_overlap_cannot_certify_an_operative_character_loss():
    source = "a\nThe amount is 450;\nb\nThe amount is 700."
    outline = bare_outline(source)
    assert outline is not None
    masked = mask_bare_labels(source, outline)
    before = SourceStructureBranch(
        (), "source-unit", "source unit", source, 0, len(source)
    )
    cleaned = SourceStructureBranch(
        (), "source-unit", "source unit", masked, 0, len(source)
    )
    corrupted = SourceStructureBranch(
        (), "source-unit", "source unit", masked.replace("700", "   "), 0, len(source)
    )

    valid = AUDIT.structural_removal_evidence(
        before, (cleaned,), (cleaned,), source, outline, "control"
    )
    invalid = AUDIT.structural_removal_evidence(
        before, (corrupted,), (corrupted,), source, outline, "control"
    )

    assert valid["structural_artifact"]
    assert not invalid["structural_artifact"]
    assert invalid["uncovered_operative_offsets"] == list(
        range(source.index("700"), source.index("700") + 3)
    )


def test_candidate_number_in_a_table_is_never_certified_by_enumeration():
    source = "The table gives the amounts:\n1\nRate one applies.\n2\nRate two applies."
    assert bare_outline(source) is None

    paths = AUDIT.candidate_paths(source, (), ())

    assert ("1",) in paths and ("2",) in paths
    assert "authenticated-bare" not in paths.values()


def test_label_with_article_an_certifies_only_its_numeric_constituent():
    source = (
        "1\nA person is eligible if—\n"
        "a\nthe person receives a main benefit; or\n"
        "b\nthe person receives a pension.\n"
        "2\nAn application must be approved for 2 days."
    )
    outline = bare_outline(source)
    assert outline is not None
    label_start = source.index("2\nAn")
    item = {
        "raw": "2\nAn",
        "source_start": label_start,
        "source_end": label_start + len("2\nAn"),
        "value": 24.0,
        "source_value": 2.0,
        "is_word_number": False,
    }

    evidence = AUDIT.numeric_removal_evidence(item, source, outline)

    assert evidence is not None
    assert evidence["numeric_constituent_count"] == 1
    assert evidence["numeric_constituents"][0]["raw"] == "2"
    assert evidence["masked_raw_span"] == " \nAn"
    assert evidence["retained_nonlabel_characters"] == len("\nAn")
    converted = AUDIT.numeric_removal_evidence(
        {**item, "source_value": None},
        source,
        outline,
        baseline_validator=AUDIT.current_validator,
    )
    assert converted["legacy_numeric_provenance"] == {
        "kind": "_YEAR_DURATION_PATTERN",
        "operand": 2.0,
        "multiplier": 12,
        "normalized_value": 24.0,
    }
    assert (
        AUDIT.numeric_removal_evidence(
            {**item, "source_value": None, "value": 25.0},
            source,
            outline,
            baseline_validator=AUDIT.current_validator,
        )
        is None
    )
    assert (
        AUDIT.numeric_removal_evidence(
            {**item, "source_end": len(source)}, source, outline
        )
        is None
    )
    assert (
        AUDIT.numeric_removal_evidence(
            {**item, "is_word_number": True}, source, outline
        )
        is None
    )


def test_unverifiable_repeated_tokens_do_not_collapse_into_one_occurrence():
    source = "one legal citation two.\none legal citation two."
    recall = "one two.\none two."
    occurrences = [
        SimpleNamespace(raw="one two", value=3.0, start=start, end=start + 7)
        for start in (0, recall.rindex("one"))
    ]

    anchored = list(AUDIT.anchored_occurrences(source, recall, occurrences).values())

    assert len(anchored) == 2
    assert all(item["source_start"] is None for item in anchored)
    assert [item["start"] for item in anchored] == [0, recall.rindex("one")]


def test_control_reownership_requires_the_same_source_bound_occurrence():
    source = "a\nIncome must be below 450;\nb\nPayment is 700."
    outline = bare_outline(source)
    assert outline is not None
    masked = mask_bare_labels(source, outline)
    split = source.index("b\n")
    old_control = SourceStructureBranch(
        (), "source-unit", "source unit", source, 0, len(source)
    )
    new_control = SourceStructureBranch(
        ("a",), "letter", "(a)", masked[:split], 0, split
    )
    source_clause = SourceStructureBranch(
        (), "excerpt", "source clause", masked, 0, len(source)
    )
    requirement = {
        "raw": "450",
        "value": 450.0,
        "source_start": source.index("450"),
        "source_end": source.index("450") + 3,
    }
    old_bound = AUDIT.BoundaryGateBranch(
        (), "numeric-boundary", "bound", source, 0, len(source), requirement
    )
    new_bound = AUDIT.BoundaryGateBranch(
        ("a",), "numeric-boundary", "bound", masked[:split], 0, split, requirement
    )
    old = {"tests_boundaries": (old_bound,), "_condition_cues": ()}
    new = {
        "tests_boundaries": (new_bound,),
        "tests_conditions": (),
        "formula": (),
        "control": (new_control,),
    }

    retained = AUDIT.structural_removal_evidence(
        old_control,
        (new_control,),
        (source_clause,),
        source,
        outline,
        "control",
        old_gates=old,
        new_gates=new,
    )
    dropped = AUDIT.structural_removal_evidence(
        old_control,
        (new_control,),
        (source_clause,),
        source,
        outline,
        "control",
        old_gates=old,
        new_gates={**new, "tests_boundaries": ()},
    )
    missing_control = AUDIT.structural_removal_evidence(
        old_control,
        (new_control,),
        (source_clause,),
        source,
        outline,
        "control",
        old_gates=old,
        new_gates={**new, "control": ()},
    )

    assert retained["uncovered_same_gate_operative_offsets"]
    assert retained["requirement_identity_proof"]["all_requirements_retained"]
    assert retained["structural_artifact"]
    assert not dropped["structural_artifact"]
    assert not missing_control["structural_artifact"]
    assert dropped["requirement_identity_proof"]["missing_boundaries"] == [requirement]


def test_authenticated_slices_preserve_rebased_children_and_drop_own_marker():
    source = (
        (Path(__file__).parent / "fixtures" / "nz_bare_labels" / "s72.txt")
        .read_text()
        .rstrip("\n")
    )
    outline = bare_outline(source)
    assert outline is not None
    citation = "nz/statute/act/public/2018/0032/section/72"
    profile = AUDIT.current_validator._numeric_profile_for_citation_path(citation)

    def extractor(text):
        return AUDIT.current_validator.extract_typed_numeric_inventory_occurrences_from_text(
            text, profile=profile
        )

    for path in (("2",), ("2", "c")):
        label = next(item for item in outline if item.path == path)
        child_citation = citation + "/" + "/".join(path)
        sliced = AUDIT.current_resolver._slice_parent_body(
            source, requested_path=child_citation, resolved_path=citation
        )

        audit = AUDIT.audit_bare_slice(
            source, label, outline, sliced, child_citation, extractor
        )

        assert audit["own_authenticated_marker_removed"]
        assert audit["operative_text_preserved"]
        assert audit["descendant_outline_preserved"]
        assert audit["numeric_inventory_matches_authenticated_slice"]
        assert not audit["glue_violations"]


@pytest.mark.parametrize("prefix", ("", "The amount is 450.\n"))
def test_singleton_descendant_projection_preserves_source_and_downstream_demands(
    prefix,
):
    source = f"1\n{prefix}A person is entitled if—\na\nthe person is aged 65 years.\n"
    outline = bare_outline(source)
    assert outline is not None
    citation = "nz/statute/act/public/2020/0001/section/1"
    child_citation = citation + "/1"
    sliced = AUDIT.current_resolver._slice_parent_body(
        source, requested_path=child_citation, resolved_path=citation
    )
    profile = AUDIT.current_validator._numeric_profile_for_citation_path(citation)

    def extractor(text):
        return AUDIT.current_validator.extract_typed_numeric_inventory_occurrences_from_text(
            text, profile=profile
        )

    audit = AUDIT.audit_bare_slice(
        source, outline[0], outline, sliced, child_citation, extractor
    )

    assert audit["projection"] == "parenthetical-descendants"
    assert audit["own_authenticated_marker_removed"]
    assert audit["operative_text_preserved"]
    assert audit["descendant_outline_preserved"]
    assert audit["numeric_inventory_matches_authenticated_slice"]
    assert audit["numeric_occurrences_at_verified_parent_offsets"] == (
        2 if prefix else 1
    )
    assert not audit["glue_violations"]


def test_projection_rejection_requires_actual_detached_definition_boundary():
    source = (
        "1\nThe following definitions apply:—\n"
        "qualifying benefit means—\na\na main benefit.\n"
        "minimum period means a period of at least 22 weeks.\n"
    )
    outline = bare_outline(source)
    assert outline is not None

    projection = AUDIT.bare_slice_projection(source, outline[0], outline)
    leaf = AUDIT.bare_slice_projection(source, outline[1], outline)

    assert projection["mode"] == "reject-detached-definition-projection"
    assert projection["definitions"]
    assert projection["expected_descendant_paths"] == [["a"]]
    assert leaf["mode"] == "leaf"
    assert leaf["text"] == "a main benefit."


def test_formula_text_cannot_certify_loss_of_a_tests_condition():
    source = "Unless the context otherwise requires, the amount is 450."
    branch = SourceStructureBranch(
        (), "source-unit", "source unit", source, 0, len(source)
    )
    baseline = SimpleNamespace(
        _source_exception_or_applicability_matches=lambda text: re.finditer(
            "Unless", text
        )
    )
    old = {"tests_conditions": (branch,)}
    lost = AUDIT.missing_test_condition_cues(
        baseline, source, old, {"tests_conditions": (), "formula": (branch,)}
    )
    retained = AUDIT.missing_test_condition_cues(
        baseline, source, old, {"tests_conditions": (branch,), "formula": ()}
    )

    assert lost == [{"start": 0, "end": 6, "text": "Unless"}]
    assert not retained


def test_formula_physical_cues_are_recorded_and_never_replace_control_conditions():
    source = (
        "a\nThe amount is at least 450.\n"
        "b\nUnless the context otherwise requires, a different amount applies."
    )
    outline = bare_outline(source)
    assert outline is not None
    masked = mask_bare_labels(source, outline)
    split = source.index("b\n")
    cue_start = source.index("Unless")
    source_clause = SourceStructureBranch(
        (), "excerpt", "source clause", masked, 0, len(source)
    )
    old_control = SourceStructureBranch(
        (), "source-unit", "source unit", source, 0, len(source)
    )
    new_control = SourceStructureBranch(
        ("a",), "letter", "(a)", masked[:split], 0, split
    )
    formula = SourceStructureBranch(
        ("b",),
        "formula-clause",
        "(b)",
        masked[split:],
        split,
        len(source),
        computation_chapeaux=((cue_start, len(source), masked[cue_start:]),),
    )
    evidence = AUDIT.structural_removal_evidence(
        old_control,
        (new_control,),
        (source_clause,),
        source,
        outline,
        "control",
        old_gates={
            "tests_boundaries": (),
            "_condition_cues": ({"start": cue_start, "end": cue_start + 6},),
        },
        new_gates={
            "tests_boundaries": (),
            "control": (new_control,),
            "formula": (formula,),
        },
    )

    assert not evidence["structural_artifact"]
    assert evidence["requirement_identity_proof"]["missing_condition_cues"]
    assert AUDIT.branch_record(formula)["computation_chapeaux"] == [
        {"start": cue_start, "end": len(source), "text": masked[cue_start:]}
    ]
    assert AUDIT.formula_context_records((formula,), source, outline)[0][
        "offset_aligned_source_excerpt"
    ]


@pytest.mark.parametrize(
    ("source", "old_span", "new_span", "gates"),
    [
        pytest.param(
            "A company or person apportioning its interest expenditure under section FE 6B of the Income Tax Act 2007 for an income year must furnish to the Commissioner, no later than when the company or person is required to furnish a return of its income for the corresponding tax year, the following information in the form and by the means prescribed by the Commissioner:\na\nnotice to the Commissioner that the section has been applied; and\nb\na reconciliation of adjusted net profit under section FE 5(1BC) of that Act to GAAP net profit; and\nc\na reconciliation of goodwill to items presented in the GAAP balance sheet; and\nd\nfurther information that is required by the Commissioner.",
            (493, 529),
            (434, 529),
            ("clauses",),
            id="65b",
        ),
        pytest.param(
            "A notice of entitlement issued under section 80KD —\na\nis not transferable; and\nb\nis subject to the terms and conditions set out in it by the Commissioner.",
            (0, 44),
            (0, 51),
            ("clauses",),
            id="80ke",
        ),
        pytest.param(
            "In determining the amount of WFF tax credit that is payable under section 80KN , the chief executive must—\na\nuse the method set out in section MF 3 of the Income Tax Act 2007 , if required to calculate the amount of the family credit abatement or Best Start credit abatement ; and\nb\ntake into account the rate of family tax credit and the family credit abatement given by sections MD 3 and MD 13 of that Act; and\nc\ntake into account the rate of Best Start tax credit and the Best Start credit abatement given by sections MG 2 and MG 3 of that Act.",
            (0, 73),
            (0, 106),
            ("clauses",),
            id="80ko",
        ),
        pytest.param(
            "When the initiating notice is a notice of proposed adjustment referred to in section 89AB(2) or a statement of position referred to in section 89AB(5), and the disputant issues the initiating notice outside the applicable response period but the notice is treated as being issued within that period, the response period for the response to the initiating notice is a 2–month period beginning on the earlier of—\na\nthe day on which the Commissioner issues a notice in favour of the disputant in accordance with section 89K(1) :\nb\nthe day on which a challenge to the Commissioner’s refusal under section 89K(4) is finally judged successful by the Taxation Review Authority or by a court, or the day on which the Commissioner concedes.",
            (0, 84),
            (0, 410),
            ("clauses", "tests_conditions", "control"),
            id="89ac",
        ),
    ],
)
def test_actual_tax_administration_citation_boundaries_preserve_each_gate(
    source, old_span, new_span, gates
):
    outline = bare_outline(source)
    assert outline is not None
    masked = mask_bare_labels(source, outline)
    branches = AUDIT.current.recognize_source_structure(source)
    start, end = old_span
    new_start, new_end = new_span
    for gate in gates:
        removed = SourceStructureBranch(
            (),
            "exception-clause" if gate != "clauses" else "excerpt",
            "source unit",
            source[start:end],
            start,
            end,
        )
        replacement = SourceStructureBranch(
            (),
            removed.kind,
            removed.label,
            masked[new_start:new_end],
            new_start,
            new_end,
        )
        cue = {"start": 0, "end": 4, "text": "When"}
        old = {
            "tests_boundaries": (),
            "_condition_cues": (cue,) if gate != "clauses" else (),
        }
        new = {"branches": branches, "tests_boundaries": (), "control": (replacement,)}

        evidence = AUDIT.structural_removal_evidence(
            removed,
            (replacement,),
            (replacement,),
            source,
            outline,
            gate,
            old_gates=old,
            new_gates=new,
        )

        assert evidence["structural_artifact"]
        proof = evidence["citation_boundary_reownership"]
        assert proof["source_aligned_supersets"]
        boundary = proof["citation_boundaries"][0]
        if start:
            assert boundary["boundary_kind"] == "citation-interior"
            assert boundary["citation"] == "section FE 5(1BC)"
        else:
            assert boundary["boundary_kind"] == "trimmed-citation-whitespace"
            assert boundary["trimmed_whitespace"] == {
                "start": end,
                "end": end + 1,
                "text": " ",
            }
        assert not evidence["uncovered_operative_offsets"]
        assert not evidence["uncovered_same_gate_operative_offsets"]


@pytest.mark.parametrize(
    ("reference", "old_boundary"),
    (
        ("section fe 5(1BC)", "1BC"),
        ("section F_E 5(1BC)", "1BC"),
        ("section word 80KD", " word"),
        ("section , 80KD", " ,"),
    ),
)
def test_citation_boundary_proof_rejects_malformed_prefixes_and_operative_gaps(
    reference, old_boundary
):
    source = f"When the rule under {reference} applies—\na\nPayment applies.\nb\nOther."
    outline = bare_outline(source)
    assert outline is not None
    masked = mask_bare_labels(source, outline)
    if old_boundary == "1BC":
        start, end = source.index(old_boundary), outline[0].start - 1
    else:
        start, end = 0, source.index(old_boundary)
    removed = SourceStructureBranch(
        (), "excerpt", "source clause", source[start:end], start, end
    )
    replacement = SourceStructureBranch(
        (),
        "excerpt",
        "source clause",
        masked[: outline[0].start - 1],
        0,
        outline[0].start - 1,
    )

    evidence = AUDIT.structural_removal_evidence(
        removed,
        (replacement,),
        (replacement,),
        source,
        outline,
        "clauses",
        new_gates={"branches": AUDIT.current.recognize_source_structure(source)},
    )

    assert not evidence["structural_artifact"]
    assert evidence["citation_boundary_reownership"] is None


@pytest.mark.parametrize(
    "mutation", ("lossy", "no-owner", "same-gate-loss", "condition-loss")
)
def test_trimmed_citation_gap_cannot_waive_an_obligation_or_owner(mutation):
    source = (
        "When the rule under section 80KD applies, the amount is 450—\n"
        "a\nPayment applies.\nb\nOther."
    )
    outline = bare_outline(source)
    assert outline is not None
    masked = mask_bare_labels(source, outline)
    end = source.index(" 80KD")
    removed = SourceStructureBranch(
        (), "exception-clause", "source unit", source[:end], 0, end
    )
    replacement = SourceStructureBranch(
        (),
        "exception-clause",
        "source unit",
        masked[: outline[0].start - 1],
        0,
        outline[0].start - 1,
    )
    clause = replacement
    replacements = (replacement,)
    branches = AUDIT.current.recognize_source_structure(source)
    if mutation == "lossy":
        replacements = (
            SourceStructureBranch(
                replacement.path,
                replacement.kind,
                replacement.label,
                replacement.text.replace("rule", "    "),
                replacement.start,
                replacement.end,
            ),
        )
    elif mutation == "no-owner":
        branches = ()
    elif mutation == "same-gate-loss":
        replacements = ()
    elif mutation == "condition-loss":
        replacements = (
            SourceStructureBranch((), "exception-clause", "source unit", "When", 0, 4),
        )

    evidence = AUDIT.structural_removal_evidence(
        removed,
        replacements,
        (clause,),
        source,
        outline,
        "tests_conditions",
        old_gates={"_condition_cues": ({"start": 0, "end": 4, "text": "When"},)},
        new_gates={"branches": branches},
    )

    assert not evidence["structural_artifact"]
    assert evidence["citation_boundary_reownership"] is None


DECIMAL_TABLE_RESTORATION_CASES = [
    (
        "Column 1 Diminishing value depreciation rate (%) | Column 2 Straight-line "
        "equivalent (%)\n"
        "1 | 1\n"
        "2 | 1.5\n"
        "2.5 | 2\n"
        "3 | 2.5\n"
        "4 | 3\n"
        "5 | 3.5\n"
        "6 | 4\n"
        "7 | 5\n"
        "7.5 | 5.5\n"
        "8 | 6\n"
        "9.5 | 6.5\n"
        "10 | 7\n"
        "11 | 7.5\n"
        "12 | 8\n"
        "13 | 8.5\n"
        "13.5 | 9\n"
        "14 | 9.5\n"
        "15 | 10\n"
        "16 | 10.5\n"
        "16.5 | 11\n"
        "17 | 11.5\n"
        "17.5 | 12\n"
        "18 | 12.5\n"
        "19 | 13\n"
        "20 | 13.5\n"
        "20.5 | 14\n"
        "21 | 14.5\n"
        "21.5 | 15\n"
        "22 | 15.5\n"
        "23 | 16\n"
        "24 | 16.5\n"
        "24.5 | 17\n"
        "25 | 17.5\n"
        "26 | 18\n"
        "27 | 18.5\n"
        "27.5 | 19\n"
        "28 | 19.5\n"
        "28.5 | 20\n"
        "29 | 20.5\n"
        "30 | 21\n"
        "31 | 22\n"
        "32 | 22.5\n"
        "32.5 | 23\n"
        "33 | 24\n"
        "34 | 24.5\n"
        "34.5 | 25\n"
        "35 | 25.5\n"
        "36 | 26\n"
        "37 | 27\n"
        "38 | 27.5\n"
        "39 | 28\n"
        "39.5 | 29\n"
        "40 | 30\n"
        "41 | 31\n"
        "42 | 32\n"
        "43 | 32.5\n"
        "44 | 33\n"
        "45 | 33.5\n"
        "45.5 | 34\n"
        "46 | 34.5\n"
        "46.5 | 35\n"
        "47 | 35.5\n"
        "47.5 | 36\n"
        "48 | 36.5\n"
        "48.5 | 37\n"
        "49 | 39\n"
        "50 | 40\n"
        "50.5 | 41\n"
        "51 | 42\n"
        "52 | 43\n"
        "53 | 44\n"
        "54 | 45\n"
        "55 | 45.5\n"
        "56 | 46\n"
        "57 | 47\n"
        "58 | 47.5\n"
        "60 | 48\n"
        "61 | 49\n"
        "62 | 50\n"
        "63 | 51\n"
        "63.5 | 63.5\n"
        "64 | 64\n"
        "65 | 65\n"
        "66 | 66\n"
        "67 | 67\n"
        "68 | 68\n"
        "69 | 69\n"
        "70 | 70\n"
        "71 | 71\n"
        "72 | 72\n"
        "73 | 73\n"
        "74 | 74\n"
        "75 | 75\n"
        "76 | 76\n"
        "77 | 77\n"
        "78 | 78\n"
        "79 | 79\n"
        "80 | 80\n"
        "81 | 81\n"
        "82 | 82\n"
        "100 | 100",
        "7.5",
        145,
        904,
    ),
    (
        "Column 1 Diminishing value depreciation rate (%) | Column 2 Straight-line "
        "equivalent (%)\n"
        "1 | 1\n"
        "2 | 1.5\n"
        "2.5 | 2\n"
        "3 | 2.5\n"
        "4 | 3\n"
        "5 | 3.5\n"
        "6 | 4\n"
        "7 | 5\n"
        "7.5 | 5.5\n"
        "8 | 6\n"
        "9.5 | 6.5\n"
        "10 | 7\n"
        "11 | 7.5\n"
        "12 | 8\n"
        "13 | 8.5\n"
        "13.5 | 9\n"
        "14 | 9.5\n"
        "15 | 10\n"
        "16 | 10.5\n"
        "16.5 | 11\n"
        "17 | 11.5\n"
        "17.5 | 12\n"
        "18 | 12.5\n"
        "19 | 13\n"
        "20 | 13.5\n"
        "20.5 | 14\n"
        "21 | 14.5\n"
        "21.5 | 15\n"
        "22 | 15.5\n"
        "23 | 16\n"
        "24 | 16.5\n"
        "24.5 | 17\n"
        "25 | 17.5\n"
        "26 | 18\n"
        "27 | 18.5\n"
        "27.5 | 19\n"
        "28 | 19.5\n"
        "28.5 | 20\n"
        "29 | 20.5\n"
        "30 | 21\n"
        "31 | 22\n"
        "32 | 22.5\n"
        "32.5 | 23\n"
        "33 | 24\n"
        "34 | 24.5\n"
        "34.5 | 25\n"
        "35 | 25.5\n"
        "36 | 26\n"
        "37 | 27\n"
        "38 | 27.5\n"
        "39 | 28\n"
        "39.5 | 29\n"
        "40 | 30\n"
        "41 | 31\n"
        "42 | 32\n"
        "43 | 32.5\n"
        "44 | 33\n"
        "45 | 33.5\n"
        "45.5 | 34\n"
        "46 | 34.5\n"
        "46.5 | 35\n"
        "47 | 35.5\n"
        "47.5 | 36\n"
        "48 | 36.5\n"
        "48.5 | 37\n"
        "49 | 39\n"
        "50 | 40\n"
        "50.5 | 41\n"
        "51 | 42\n"
        "52 | 43\n"
        "53 | 44\n"
        "54 | 45\n"
        "55 | 45.5\n"
        "56 | 46\n"
        "57 | 47\n"
        "58 | 47.5\n"
        "60 | 48\n"
        "61 | 49\n"
        "62 | 50\n"
        "63 | 51\n"
        "63.5 | 63.5\n"
        "64 | 64\n"
        "65 | 65\n"
        "66 | 66\n"
        "67 | 67\n"
        "68 | 68\n"
        "69 | 69\n"
        "70 | 70\n"
        "71 | 71\n"
        "72 | 72\n"
        "73 | 73\n"
        "74 | 74\n"
        "75 | 75\n"
        "76 | 76\n"
        "77 | 77\n"
        "78 | 78\n"
        "79 | 79\n"
        "80 | 80\n"
        "81 | 81\n"
        "82 | 82\n"
        "100 | 100",
        "6.5",
        167,
        824,
    ),
    (
        "Other assets (excluding fixed life intangible property, excluded depreciable "
        "property, and buildings) | Buildings\n"
        "Column 1 | Column 2 | Column 3 | Column 4\n"
        "D.V. rate (%) | Straight-line rate (%) | D.V. rate (%) | Straight-line rate (%)\n"
        "100 | 100 | 0 | 0\n"
        "67 | 67 | 0 | 0\n"
        "50 | 40 | 0 | 0\n"
        "40 | 30 | 0 | 0\n"
        "30 | 21 | 0 | 0\n"
        "25 | 17.5 | 0 | 0\n"
        "20 | 13.5 | 0 | 0\n"
        "16 | 10.5 | 13.5 | 8\n"
        "13 | 8.5 | 11 | 6.5\n"
        "10 | 7 | 8.5 | 5\n"
        "8 | 6 | 6.5 | 4\n"
        "6 | 4 | 4.5 | 3\n"
        "4 | 3 | 3 | 2\n"
        "2 | 1.5 | 1.3 | 1",
        "1.3",
        468,
        375,
    ),
]


@pytest.mark.parametrize(
    ("source", "raw", "target_start", "suppressor_start"),
    DECIMAL_TABLE_RESTORATION_CASES,
    ids=("schedule10-diminishing", "schedule10-straight-line", "schedule11-buildings"),
)
def test_actual_depreciation_cells_restore_same_decimal_with_original_stage_proof(
    source, raw, target_start, suppressor_start, legacy_validator_for_counterfactual
):
    assert bare_outline(source) is None

    _old, added, evidence = grouped_restoration_case(
        source, legacy_validator_for_counterfactual, raw, target_start=target_start
    )

    assert added["source_start"] == target_start
    assert added["value"] == added["source_value"] == float(raw)
    assert evidence["operand_kind"] == "ascii-dot-decimal"
    assert evidence["collector_snapshot"]["traced_final_matches_normal"]
    assert evidence["collector_snapshot"]["profile"] == "legacy"
    assert evidence["legacy_alternate_reading"] == float(raw.replace(".", ""))
    suppressor = evidence["distinct_prior_suppressors"][0]
    assert suppressor["source_start"] == suppressor_start
    assert (
        suppressor["value"]
        == suppressor["source_value"]
        == evidence["legacy_alternate_reading"]
    )
    assert suppressor["physical_relation"] == (
        "before" if suppressor_start < target_start else "after"
    )
    assert evidence["counterfactual_masked_recall_spans"] == [
        [suppressor_start, suppressor_start + len(suppressor["raw"])],
    ]
    for record in (added, suppressor):
        assert not any(
            record[field]
            for field in (
                "is_word_number",
                "has_rate_context",
                "has_temporal_context",
                "has_structural_context",
                "requires_rate_context",
            )
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "partial-token",
        "mixed-separator",
        "rate-context",
        "required-rate",
        "source-value",
        "alternatives",
        "profile",
        "value",
        "unaligned-span",
    ),
)
def test_decimal_restoration_rejects_incomplete_or_changed_numeric_semantics(
    mutation, legacy_validator_for_counterfactual
):
    source, raw, target_start, _suppressor_start = DECIMAL_TABLE_RESTORATION_CASES[0]
    baseline = legacy_validator_for_counterfactual
    extract = baseline.extract_typed_numeric_inventory_occurrences_from_text
    old = tuple(AUDIT.anchored_occurrences(source, source, extract(source)).values())
    current = AUDIT.anchored_occurrences(
        source,
        source,
        AUDIT.current_validator.extract_typed_numeric_inventory_occurrences_from_text(
            source
        ),
    )
    added = dict(
        next(item for item in current.values() if item["source_start"] == target_start)
    )
    profile = "legacy"
    if mutation == "partial-token":
        source = source[:target_start] + "1" + source[target_start:]
        added.update(
            start=target_start + 1,
            end=target_start + 1 + len(raw),
            source_start=target_start + 1,
            source_end=target_start + 1 + len(raw),
        )
    elif mutation == "mixed-separator":
        source = source[:target_start] + "7,5" + source[target_start + len(raw) :]
        added["raw"] = "7,5"
    elif mutation == "rate-context":
        added["has_rate_context"] = True
    elif mutation == "required-rate":
        added["requires_rate_context"] = True
    elif mutation == "source-value":
        added["source_value"] = 75.0
    elif mutation == "alternatives":
        added["alternative_values"] = (75.0,)
    elif mutation == "profile":
        profile = "en-US"
    elif mutation == "value":
        added["value"] = 75.0
    elif mutation == "unaligned-span":
        added["source_start"] += 1

    evidence = AUDIT.cross_occurrence_restoration_evidence(
        added,
        source,
        source,
        old,
        baseline,
        extract,
        profile=profile,
    )

    assert evidence is None


SPECIAL_SUPPRESSOR_SOURCES = {
    "dg-14": "When this section applies\n"
    "1\n"
    "This section applies for a person, for an income year and a company ( "
    "company A ) that is a close company or qualifying company, when—\n"
    "a\n"
    "a net asset balance remains outstanding for an income year after the "
    "application of—\n"
    "i\n"
    "first, section DG 13 , if applicable:\n"
    "ii\n"
    "secondly, section DG 12 , if applicable:\n"
    "iii\n"
    "thirdly, section DG 11 , if neither applies; and\n"
    "b\n"
    "the person—\n"
    "i\n"
    "is not a company ; and\n"
    "ii\n"
    "has a voting interest in company A; and\n"
    "iii\n"
    "has interest expenditure for which they are allowed a deduction.\n"
    "Amount to be apportioned\n"
    "2\n"
    "For a natural person, the amount of interest expenditure that must be "
    "apportioned is only the amount of interest that the person incurs on money "
    "borrowed to acquire shares in company A or in a company referred to in "
    "section DG 13(1)\ufeff(b) .\n"
    "Method of apportionment\n"
    "3\n"
    "The apportionment is made using the rules set out in section DG 13(2) to "
    "(10) , treating the person as if they were the company.\n"
    "Example\n"
    "Company Y has 2 shareholders: Thomas, who has borrowed $200,000 to acquire "
    "a 50% interest in the company, and Brent, who has borrowed $10,000 to buy "
    "his 50% interest. Each has a share of the remaining net asset balance of "
    "$22,500. The formula is ($65,000 − $20,000) × 50% = $22,500. Since Thomas's "
    "debt value is greater than his share of the net asset balance, Thomas must "
    "apportion 11.25% of his total interest expenditure (sections DG 14 and DG "
    "13(7)–(9)). The formula is 22,500/200,000. Since Brent's debt value is less "
    "than his share of the net asset balance, all Brent's interest expenditure "
    "must be apportioned (sections DG 14 and DG 13(5)).",
    "dg-18": "When this section applies\n"
    "1\n"
    "This section applies when—\n"
    "a\n"
    "a person incurs expenditure for an income year for which they are allowed a "
    "deduction that is limited under 1 or more of sections DG 12 to DG 14 ; and\n"
    "b\n"
    "the income year is an income year in which section DG 16(1)\ufeff(b) "
    "applies.\n"
    "How this section applies: first looping rule\n"
    "2\n"
    "The first application of this section is to every group company B in "
    "sequence until no other group companies exist to which this subsection "
    "applies.\n"
    "How this section applies: second looping rule\n"
    "3\n"
    "The second application of this section is sequentially to—\n"
    "a\n"
    "first, 1 or more of the following persons, none of which is a company "
    "referred to in subsection (2):\n"
    "i\n"
    "a person who is a shareholder in company A:\n"
    "ii\n"
    "a person who is a shareholder in a company that is part of the same group "
    "of companies as company A and has a voting interest in company A; and\n"
    "b\n"
    "secondly, a person who is a shareholder in a company referred to in "
    "paragraph (a); and\n"
    "c\n"
    "so on, until no other persons exist to which this subsection applies.\n"
    "Quarantined amount\n"
    "4\n"
    "The amount of the person's excess expenditure for the income year is "
    "calculated using the formula—\n"
    "expenditure − outstanding profit balance.\n"
    "Definition of items in formula\n"
    "5\n"
    "In the formula,—\n"
    "a\n"
    "expenditure is the total of the following amounts:\n"
    "i\n"
    "the total amount of deductions that the person is allowed for the income "
    "year under sections DG 12 to DG 14 , as applicable and after any necessary "
    "apportionment; and\n"
    "ii\n"
    "an amount of the person that was quarantined under this section for an "
    "earlier income year and is not yet allocated to an income year:\n"
    "b\n"
    "outstanding profit balance ,—\n"
    "i\n"
    "for company B, is the amount of the outstanding profit balance referred to "
    "in section DG 16(5) :\n"
    "ii\n"
    "for a shareholder, is the amount that is the person's share of the "
    "outstanding profit balance referred to in section DG 16(5) , calculated "
    "using the formula in section DG 13(3) , treating the outstanding profit "
    "balance as if it were the net asset balance.\n"
    "No deduction for quarantined amount\n"
    "6\n"
    "The excess expenditure calculated under subsection (4) is either "
    "quarantined or remains quarantined, as applicable, and is denied as a "
    "deduction for the income year.\n"
    "Recalculation of outstanding profit balance\n"
    "7\n"
    "For the purposes of subsections (4) and (5)\ufeff(b), the amount that is "
    "the outstanding profit balance must be recalculated on each application, "
    "being reduced by an amount equal to the amount of any deduction counted.\n"
    "Zero result\n"
    "8\n"
    "For the purposes of the formula in subsection (4), if the amount of the "
    "outstanding profit balance for the income year is greater than the amount "
    "of expenditure for the income year, the result of the formula is treated as "
    "zero.\n"
    "Example\n"
    "Aircraft Ltd owns an aircraft to which the rules in this subpart apply; the "
    "income derived from the asset in the current year is less than 2% of the "
    "cost of the aircraft. The company has calculated an outstanding profit "
    "balance of $12,000 after the application of section DG 16. Aircraft is 100% "
    "owned by Parent Ltd, which has apportioned interest expenditure of $5,000 "
    "calculated under section DG 12. Parent has 2 equal shareholders, Alisa who "
    "has apportioned interest expenditure of $8,000, and Hamish who has "
    "apportioned interest expenditure of $1,000, both calculated under section "
    "DG 14. Parent must apply section DG 18 first, and is not required to "
    "quarantine any of its interest expenditure; the outstanding profit balance "
    "is reduced to $7,000 ($12,000 − $5,000). Alisa’s and Hamish’s share of the "
    "outstanding profit balance is $3,500 each ($7,000 × 50%) . Alisa must "
    "quarantine $4,500 of interest expenditure ($8,000 − $3,500); Hamish is not "
    "required to quarantine any interest expenditure.",
    "di-6": "Other amounts reduced by amount of section DI 5 deduction\n"
    "1\n"
    "For a person ( person A ) allowed a deduction under section DI 5 for "
    "expenditure incurred in acquiring a new investment asset, and for a person "
    "that is a new asset transferee in relation to person A and the asset,—\n"
    "a\n"
    "the asset’s adjusted tax value, base value, cost, or value, as applicable, "
    "is reduced by the amount of person A’s deduction for the purposes of "
    "quantifying the amount of depreciation loss under subpart EE (Depreciation) "
    "for the asset:\n"
    "b\n"
    "the relevant diminished value, expenditure, cost, value, or consideration, "
    "as applicable, is reduced by the amount of person A’s deduction for the "
    "purposes of quantifying the amount of a deduction under subparts DO (Farming "
    "and aquacultural business expenditure), DP (Forestry expenditure), DT "
    "(Petroleum mining expenditure), and DU (Mineral mining expenditure) for the "
    "asset.\n"
    "Available amount of section DI 5 deduction\n"
    "2\n"
    "For the purposes of subsection (1), if the amount of person A’s deduction is "
    "less than the amount (the available deduction amount ) they would have been "
    "allowed if they had used the asset wholly in deriving assessable income or "
    "carrying on a business for the purpose of deriving assessable income, the "
    "applicable amount referred to in subsection (1) is reduced by the available "
    "deduction amount for the purposes of quantifying the amount of depreciation "
    "loss under subpart EE or the amount of a deduction under subpart DO , DP , "
    "DT , or DU , as applicable.\n"
    "Example 1\n"
    "ABC company purchases a new investment asset ( Asset A ) for $10,000 on 1 "
    "October 2025. Asset A is depreciable property and has a DV rate of 10%. "
    "Asset A is used for 6 months in the income year ending 31 March 2026. In the "
    "2025–26 income year, ABC company is allowed the following deduction for the "
    "new investment asset under section DI 5 :\n"
    "0.2 × $10,000 = $2,000.\n"
    "Section DI 6(1)\ufeff(a) ensures that the cost base for the asset for "
    "depreciation purposes is reduced by the amount of the deduction under "
    "section DI 5 , that is, $10,000 − $2,000 = $8,000. Depreciation for the "
    "asset is then calculated for the 2025–26 income year as follows:\n"
    "10% × 6/12 × $8,000 = $400.\n"
    "The total deduction for Asset A in the 2025–26 income year is $2,400. The "
    "adjusted tax value of the asset for the 2026–27 income year will be $7,600 "
    "($8,000 − $400).\n"
    "Example 2\n"
    "ABC company transfers Asset A to an associate ( associate B ) at the end of "
    "the 2025–26 income year. Under a provision of this Act, associate B steps "
    "into the shoes of ABC company (for example, section FM 15(2) ). The original "
    "acquisition cost of $10,000 is reduced for associate B, for depreciation "
    "purposes, to $9,600 by operation of section DI 6 . Depreciation deductions "
    "of $400 are effectively transferred ( see section FM 15(7)). Section DI 6(a) "
    "in effect also transfers the $2,000 new investment asset deduction claimed "
    "by ABC company to associate B for the purposes of the item new investment "
    "asset amount in section EE 48(1B) and (1C) . When associate B disposes of "
    "Asset A for $9,600, associate B has clawback income of $2,000 under section "
    "EE 48 .",
}


@pytest.mark.parametrize(
    ("section", "raw", "target_start", "special_kind"),
    (
        ("dg-14", "200,000", 1023, "captured-legacy-first-group-prefix"),
        ("dg-18", "1,000", 3268, "captured-ascii-percent"),
        ("di-6", "10,000", 2028, "captured-legacy-first-group-prefix"),
        ("di-6", "8,000", 2319, "captured-legacy-first-group-prefix"),
        ("di-6", "10,000", 2592, "captured-legacy-first-group-prefix"),
        ("di-6", "2,000", 2826, "captured-legacy-first-group-prefix"),
        ("di-6", "2,000", 3075, "captured-legacy-first-group-prefix"),
    ),
)
def test_actual_arithmetic_and_percentage_suppressors_restore_exact_amount(
    section, raw, target_start, special_kind, legacy_validator_for_counterfactual
):
    source = SPECIAL_SUPPRESSOR_SOURCES[section]

    _old, added, evidence = grouped_restoration_case(
        source, legacy_validator_for_counterfactual, raw, target_start=target_start
    )

    assert added["source_start"] == target_start
    assert evidence["collector_snapshot"]["traced_final_matches_normal"]
    proofs = [
        item["suppression_provenance"]
        for item in evidence["distinct_prior_suppressors"]
    ]
    assert special_kind in {proof["kind"] for proof in proofs}
    for proof in proofs:
        if proof["kind"] == "captured-legacy-first-group-prefix":
            constituent = proof["numeric_constituents"][0]
            partial = proof["legacy_partial_capture"]
            assert partial["raw"] == constituent["raw"].split(",")[0]
            assert partial["source_end"] < constituent["source_end"]
            assert (
                source[constituent["source_start"] : constituent["source_end"]]
                == constituent["raw"]
            )
            assert AUDIT.complete_numeric_constituent_span(
                source, constituent["source_start"], constituent["source_end"]
            )
        elif proof["kind"] == "captured-ascii-percent":
            constituent = proof["numeric_constituents"][0]
            assert source[constituent["source_end"]] == "%"
            assert proof["operand"] == 100
            assert proof["normalized_value"] == 1


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-snapshot",
        "invented-record",
        "wrong-source-value",
        "wrong-value",
        "wrong-flags",
        "year",
        "decimal",
        "middle-group",
        "suffix-group",
        "mixed-separators",
        "recall-mismatch",
    ),
)
def test_partial_legacy_capture_requires_authenticated_complete_first_group(mutation):
    source = "The formula is 0.2 × $10,000. Another amount is $10,000."
    start = source.index("10,000")
    old = source_numeric_record("10", 10.0, start)
    original = dict(old)
    recall = source
    snapshot = {"traced_final_matches_normal": True, "occurrences": (original,)}
    if mutation == "missing-snapshot":
        snapshot = None
    elif mutation == "invented-record":
        snapshot["occurrences"] = ()
    elif mutation == "wrong-source-value":
        old["source_value"] = 10000.0
    elif mutation == "wrong-value":
        old["value"] = 10000.0
    elif mutation == "wrong-flags":
        old["has_temporal_context"] = True
    elif mutation in {"year", "decimal", "mixed-separators"}:
        replacement = {
            "year": "10000",
            "decimal": "10.000",
            "mixed-separators": "10,000.5",
        }[mutation]
        source = source[:start] + replacement + source[start + 6 :]
        recall = source
    elif mutation == "middle-group":
        source = source[:start] + "1,010,000" + source[start + 6 :]
        start += 2
        old = source_numeric_record("010", 10.0, start)
        snapshot["occurrences"] = (dict(old),)
        recall = source
    elif mutation == "suffix-group":
        source = source[:start] + "1,010" + source[start + 6 :]
        start += 2
        old = source_numeric_record("010", 10.0, start)
        snapshot["occurrences"] = (dict(old),)
        recall = source
    elif mutation == "recall-mismatch":
        recall = source[: start + 2] + " " + source[start + 3 :]

    assert (
        AUDIT.captured_special_suppressor_provenance(old, source, recall, snapshot)
        is None
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-snapshot",
        "invented-record",
        "wrong-value",
        "wrong-source-value",
        "missing-rate",
        "missing-required-rate",
        "missing-percent",
        "percent-in-recall",
        "partial-token",
        "alternatives",
    ),
)
def test_percentage_suppressor_requires_original_complete_conversion(mutation):
    source = "The company is 100% owned. The amount is $1,000."
    start = source.index("100")
    old = source_numeric_record("100", 1.0, start)
    old.update(source_value=100.0, has_rate_context=True, requires_rate_context=True)
    snapshot = {"traced_final_matches_normal": True, "occurrences": (dict(old),)}
    recall = source
    if mutation == "missing-snapshot":
        snapshot = None
    elif mutation == "invented-record":
        snapshot["occurrences"] = ()
    elif mutation == "wrong-value":
        old["value"] = 100.0
    elif mutation == "wrong-source-value":
        old["source_value"] = 1.0
    elif mutation == "missing-rate":
        old["has_rate_context"] = False
    elif mutation == "missing-required-rate":
        old["requires_rate_context"] = False
    elif mutation == "missing-percent":
        source = source.replace("100%", "100 ")
    elif mutation == "percent-in-recall":
        recall = source.replace("100%", "100 ")
    elif mutation == "partial-token":
        source = source[:start] + "1" + source[start:]
        recall = source
        old.update(
            start=start + 1, end=start + 4, source_start=start + 1, source_end=start + 4
        )
        snapshot["occurrences"] = (dict(old),)
    elif mutation == "alternatives":
        old["alternative_values"] = (100.0,)

    assert (
        AUDIT.captured_special_suppressor_provenance(old, source, recall, snapshot)
        is None
    )


def test_partial_capture_enclosing_mask_cannot_overlap_restored_target(
    legacy_validator_for_counterfactual,
):
    source = "The formula is 0.2 × $10,000."
    start = source.index("10,000")
    old = source_numeric_record("10", 10.0, start)
    added = source_numeric_record("10,000", 10000.0, start)
    snapshot = {"traced_final_matches_normal": True, "occurrences": (dict(old),)}

    evidence = AUDIT._cross_occurrence_restoration_from_candidates(
        added,
        source,
        source,
        (old,),
        legacy_validator_for_counterfactual,
        lambda _: (),
        stage_snapshot=snapshot,
    )

    assert evidence is None


@pytest.fixture
def alignment_cache():
    cache = AUDIT.cached_source_token_alignment
    cache.cache_clear()
    yield cache
    cache.cache_clear()


def test_large_token_alignment_cache_reuses_only_immutable_indices(alignment_cache):
    source, recall = " " * 16_384 + "one citation two", " " * 16_384 + "one two"

    original = AUDIT.source_token_alignment(source, recall)
    repeated = AUDIT.source_token_alignment(source, recall)

    assert original is repeated
    assert alignment_cache.cache_info().hits == 1
    assert alignment_cache.cache_info().misses == 1
    assert all(isinstance(values, tuple) for values in original)
    assert all(
        value is None or isinstance(value, int)
        for values in original
        for value in values
    )
    with pytest.raises(TypeError):
        original[0][0] = 0
    with pytest.raises(TypeError):
        original[0] = ()


@pytest.mark.parametrize(
    ("source", "recall", "expected"),
    (
        ("", "", ((), (), (), ())),
        ("", "word", ((), (0,), (None,), (None,))),
        ("word", "", ((0,), (), (), ())),
        (" ", " ", ((), (), (), ())),
    ),
)
def test_empty_alignment_inputs_remain_uncached(
    source, recall, expected, alignment_cache
):
    assert AUDIT.source_token_alignment(source, recall) == expected
    assert alignment_cache.cache_info().currsize == 0


@pytest.mark.parametrize("changed", ("source", "recall"))
def test_large_alignment_keys_include_complete_source_and_recall(
    changed, alignment_cache
):
    source, recall = " " * 16_384 + "one citation two", " " * 16_384 + "one two"
    original = AUDIT.source_token_alignment(source, recall)
    if changed == "source":
        source += " three"
    else:
        recall += " three"

    result = AUDIT.source_token_alignment(source, recall)

    assert result is not original
    assert result == AUDIT.uncached_source_token_alignment(source, recall)
    assert alignment_cache.cache_info().misses == 2


def test_small_alignment_calls_cannot_evict_large_body_maps(alignment_cache):
    source, recall = " " * 16_384 + "one citation two", " " * 16_384 + "one two"
    original = AUDIT.source_token_alignment(source, recall)
    for index in range(20):
        AUDIT.source_token_alignment(f"one citation {index}", f"one {index}")

    assert AUDIT.source_token_alignment(source, recall) is original
    assert alignment_cache.cache_info().currsize == 1
    assert alignment_cache.cache_info().misses == 1


def test_large_alignment_cache_retains_at_most_four_exact_pairs(alignment_cache):
    padding = " " * 16_384
    first_source, first_recall = padding + "first citation", padding + "first"
    original = AUDIT.source_token_alignment(first_source, first_recall)
    for index in range(4):
        AUDIT.source_token_alignment(
            padding + f"word {index} citation", padding + f"word {index}"
        )
    assert (
        alignment_cache.cache_info().currsize
        == alignment_cache.cache_info().maxsize
        == 4
    )

    assert AUDIT.source_token_alignment(first_source, first_recall) is not original
    assert alignment_cache.cache_info().misses == 6


def test_cached_alignment_preserves_every_numeric_offset_and_semantic_field(
    alignment_cache, monkeypatch
):
    padding = " " * 16_384
    source = (
        padding
        + "section 450 applies.\nThe amount is 450.\n450\nThe second amount is 450."
    )
    recall = padding + " applies.\nThe amount is 450.\n450\nThe second amount is 450."
    occurrences = []
    for match in re.finditer("450", recall):
        record = source_numeric_record("450", 0.45, match.start())
        record.update(
            source_value=450.0,
            alternative_values=(45.0,),
            has_rate_context=True,
            has_temporal_context=True,
            has_structural_context=True,
            requires_rate_context=True,
        )
        occurrences.append(
            SimpleNamespace(
                **{
                    key: value
                    for key, value in record.items()
                    if key not in {"source_start", "source_end"}
                }
            )
        )

    cached = AUDIT.anchored_occurrences(source, recall, occurrences)
    repeated = AUDIT.anchored_occurrences(source, recall, occurrences)
    monkeypatch.setattr(
        AUDIT, "source_token_alignment", AUDIT.uncached_source_token_alignment
    )
    uncached = AUDIT.anchored_occurrences(source, recall, occurrences)

    assert cached == repeated == uncached
    assert alignment_cache.cache_info().hits == 1
    assert [item["source_start"] for item in cached.values()] == [
        match.start() for match in list(re.finditer("450", source))[1:]
    ]
    assert all(
        item["source_value"] == 450
        and item["alternative_values"] == (45.0,)
        and item["requires_rate_context"]
        for item in cached.values()
    )


OXFORD_CITATION_SOURCE = "1\nThis clause applies if—\na\nthe Commissioner has granted an exemption to the 183-day requirement to a borrower under section 38AE , 38AJ , or 38AJA of the Student Loan Scheme Act 1992 before 1 April 2012; and\nb\nthe end date for the period of that exemption continues past 1 April 2012, or the conditions for the continuation of that exemption continue to apply or to be met on the close of 31 March 2012.\n2\nThe exemption continues until, as applicable,—\na\nthe end date for the period of that exemption specified by the Commissioner; or\nb\nthe conditions for the continuation of that exemption cease to apply or to be met.\n3\nThe borrower is treated as being physically in New Zealand for the purposes of sections 22 to 24 for the period that the exemption continues."


@pytest.mark.parametrize("connector", ("or", "and"))
def test_oxford_citation_list_preserves_actual_student_loan_clause(connector):
    source = OXFORD_CITATION_SOURCE.replace(
        "38AJ , or 38AJA", f"38AJ , {connector} 38AJA"
    )
    outline = bare_outline(source)
    assert outline is not None
    start, end = source.index("38AJA"), source.index(";") + 1
    beginning = source.index("the Commissioner")
    removed = SourceStructureBranch(
        (), "excerpt", "source clause", source[start:end], start, end
    )
    replacement = SourceStructureBranch(
        (), "excerpt", "source clause", source[beginning:end], beginning, end
    )

    evidence = AUDIT.structural_removal_evidence(
        removed,
        (replacement,),
        (replacement,),
        source,
        outline,
        "clauses",
        new_gates={"branches": AUDIT.current.recognize_source_structure(source)},
    )

    assert evidence["structural_artifact"]
    assert not evidence["uncovered_operative_offsets"]
    assert not evidence["uncovered_same_gate_operative_offsets"]
    proof = evidence["citation_boundary_reownership"]
    assert (
        proof["citation_boundaries"][0]["citation"]
        == f"section 38AE , 38AJ , {connector} 38AJA"
    )
    assert proof["source_aligned_supersets"][0]["owner_path"] == ["1", "a"]


@pytest.mark.parametrize(
    "mutation",
    (
        "operative-word",
        "malformed-number",
        "unmatched-parenthesis",
        "decimal",
        "missing-connector-space",
        "duplicate-connector",
        "no-owner",
        "same-gate-loss",
        "condition-loss",
    ),
)
def test_oxford_citation_proof_rejects_malformed_lists_and_lost_obligations(mutation):
    reference = "section 38AE, 38AJ, or 38AJA"
    final_token = "38AJA"
    if mutation == "operative-word":
        reference = reference.replace("or 38AJA", "or amount 38AJA")
    elif mutation == "malformed-number":
        reference = reference.replace("38AJA", "38_AJA")
        final_token = "38_AJA"
    elif mutation == "unmatched-parenthesis":
        reference = reference.replace("38AJA", "38AJA(2")
        final_token = "38AJA(2"
    elif mutation == "decimal":
        reference = reference.replace("38AJA", "38.5")
        final_token = "38.5"
    elif mutation == "missing-connector-space":
        reference = reference.replace("or 38AJA", "or38AJA")
    elif mutation == "duplicate-connector":
        reference = reference.replace("or 38AJA", "or and 38AJA")
    source = f"a\nUnder {reference}, if income exceeds 450, payment applies;\nb\nOther payment."
    outline = bare_outline(source)
    assert outline is not None
    start, end = source.index(final_token), source.index(";") + 1
    removed = SourceStructureBranch(
        ("satz-38",), "exception-clause", "Satz 38", source[start:end], start, end
    )
    replacement = SourceStructureBranch(
        ("a",), "exception-clause", "(a)", source[2:end], 2, end
    )
    clauses = (replacement,)
    replacements = clauses
    branches = AUDIT.current.recognize_source_structure(source)
    if mutation == "no-owner":
        branches = ()
    elif mutation == "same-gate-loss":
        replacements = ()
    elif mutation == "condition-loss":
        cue_start = source.index("if income")
        replacements = (
            SourceStructureBranch(
                ("a",), "exception-clause", "(a)", "if", cue_start, cue_start + 2
            ),
        )
    cue = {
        "start": source.index("if income"),
        "end": source.index("if income") + 2,
        "text": "if",
    }

    evidence = AUDIT.structural_removal_evidence(
        removed,
        replacements,
        clauses,
        source,
        outline,
        "tests_conditions",
        old_gates={"_condition_cues": (cue,)},
        new_gates={"branches": branches},
    )

    assert not evidence["structural_artifact"]
    assert evidence["citation_boundary_reownership"] is None


# Verbatim pinned June16 corpus line8864, Social Security Act2018 s380.
# Body SHA256: 5a1f0b936f5804eb456a801f3eb9b1297857afd80c0a6d968e43f59f752d575c.
S380_SOURCE = "1\nThe purpose of this section is to enable effect to be given in New Zealand law to an agreement (for example, a Convention), or an alteration to an agreement, with the Government of another country providing for reciprocity in respect of matters relating to social security monetary benefits (a reciprocity agreement ).\n2\nThe Governor-General may, by Order in Council, do all or any of the following:\na\ndeclare all or any provisions of, or of an alteration to, a reciprocity agreement (being provisions set out in a schedule of the order) have force and effect so far as they relate to New Zealand:\nb\ndeclare that the following enactments (and any regulations or orders made, and in force, under the following Acts) have effect subject to any modifications required for the purpose of giving effect to the agreement or alteration:\ni\nthis Act:\nii\nPart 6 of the Veterans’ Support Act 2014 :\niii\nPart 1 of the New Zealand Superannuation and Retirement Income Act 2001 :\nc\nrevoke a previous order that applied in respect of, or in respect of an alteration to, a reciprocity agreement if the agreement or alteration is no longer in force or if, on the commencement of the order, the previous order is intended to be no longer in force.\n3\nThe Governor-General may, by the same or a later Order in Council, specify the date on which an order made under subsection (2) is to come into force (which may be a date before, on, or after the date on which the order is made under subsection (2)).\n4\nThis section is subject to section 381 .\n5\nAn order under this section is secondary legislation ( see Part 3 of the Legislation Act 2019 for publication requirements)."


@pytest.fixture(scope="module")
def s380_condition_replay_case():
    baseline = AUDIT.load_baseline(
        "86a1f911a",
        "src/axiom_encode/harness/source_completeness.py",
        "axiom_encode.harness._nz_condition_replay_test_baseline",
    )
    citation = "nz/statute/act/public/2018/0032/section/380"
    old = AUDIT.gates(baseline, S380_SOURCE, citation, lambda _: ())
    new = AUDIT.gates(AUDIT.current, S380_SOURCE, citation, lambda _: ())
    outline = bare_outline(S380_SOURCE)
    assert outline is not None
    removed = next(
        branch
        for branch in old["tests_conditions"]
        if (branch.start, branch.end) == (729, 1137)
    )
    return old, new, outline, removed


def s380_removal_evidence(old, new, outline, removed):
    return AUDIT.structural_removal_evidence(
        removed,
        new["tests_conditions"],
        new["clauses"],
        S380_SOURCE,
        outline,
        "tests_conditions",
        old_gates=old,
        new_gates=new,
    )


def test_real_s380_condition_groups_replayed_from_unchanged_pinned_clause(
    s380_condition_replay_case,
):
    old, new, outline, removed = s380_condition_replay_case
    evidence = s380_removal_evidence(old, new, outline, removed)
    assert evidence["structural_artifact"]
    (proof,) = evidence["condition_identity_proof"][
        "original_condition_group_reownership"
    ]
    assert proof["baseline_ref"] == "86a1f911a"
    assert len(proof["baseline_source_sha256"]) == 64
    assert proof["owner_path"] == ["2", "c"]
    assert (
        proof["complete_source_clause"]["start"],
        proof["complete_source_clause"]["end"],
    ) == (970, 1231)
    assert [
        (group["start"], group["end"])
        for group in proof["all_original_condition_groups"]
    ] == [(1081, 1137), (1137, 1231)]
    assert [
        (group["start"], group["end"]) for group in proof["required_condition_groups"]
    ] == [(1081, 1137)]
    assert not evidence["uncovered_operative_offsets"]
    assert not evidence["condition_identity_proof"]["uncovered_condition_offsets"]


@pytest.mark.parametrize(
    "mutation",
    (
        "cue-only",
        "missing-qualifier",
        "first-group-loss",
        "second-group-loss",
        "subject-to-loss",
        "action-character-loss",
        "missing-owner",
        "mixed-owner",
        "mixed-condition-owner",
        "missing-baseline",
        "current-baseline",
        "unpinned-baseline",
    ),
)
def test_s380_group_reownership_rejects_incomplete_obligations(
    s380_condition_replay_case,
    mutation,
):
    old, new, outline, removed = s380_condition_replay_case
    old, new = dict(old), dict(new)
    conditions = list(new["tests_conditions"])
    index = next(i for i, branch in enumerate(conditions) if branch.start == 1081)
    if mutation in {"cue-only", "missing-qualifier"}:
        end = 1083 if mutation == "cue-only" else 1124
        branch = conditions[index]
        conditions[index] = SourceStructureBranch(
            branch.path, branch.kind, branch.label, S380_SOURCE[1081:end], 1081, end
        )
    elif mutation == "first-group-loss":
        conditions.pop(index)
    elif mutation == "second-group-loss":
        conditions = [branch for branch in conditions if branch.start != 1137]
    elif mutation == "subject-to-loss":
        conditions = [branch for branch in conditions if branch.start != 602]
    elif mutation == "action-character-loss":
        new["clauses"] = tuple(
            SourceStructureBranch(
                branch.path,
                branch.kind,
                branch.label,
                branch.text[1:],
                branch.start + 1,
                branch.end,
            )
            if branch.start == 970
            else branch
            for branch in new["clauses"]
        )
    elif mutation in {"missing-owner", "mixed-owner"}:
        new["branches"] = tuple(
            SourceStructureBranch(
                ("2", "b"),
                branch.kind,
                branch.label,
                branch.text,
                branch.start,
                branch.end,
            )
            if mutation == "mixed-owner" and branch.path == ("2", "c")
            else branch
            for branch in new["branches"]
            if mutation != "missing-owner" or branch.path != ("2", "c")
        )
    elif mutation == "mixed-condition-owner":
        branch = conditions[index]
        conditions[index] = SourceStructureBranch(
            ("2", "b"),
            branch.kind,
            branch.label,
            branch.text,
            branch.start,
            branch.end,
        )
    elif mutation == "missing-baseline":
        old["_baseline_condition_module"] = None
    elif mutation == "current-baseline":
        old["_baseline_condition_module"] = AUDIT.current
    elif mutation == "unpinned-baseline":
        baseline = old["_baseline_condition_module"]
        old["_baseline_condition_module"] = SimpleNamespace(
            **{**baseline.__dict__, "_audit_baseline_ref": "untrusted"}
        )
    new["tests_conditions"] = tuple(conditions)
    evidence = s380_removal_evidence(old, new, outline, removed)
    assert not evidence["structural_artifact"]


def test_condition_group_replay_rejects_single_group_clause(
    s380_condition_replay_case,
):
    old, new, outline, _removed = s380_condition_replay_case
    truncated_clause = SourceStructureBranch(
        (), "excerpt", "source clause", S380_SOURCE[970:1137], 970, 1137
    )
    assert (
        AUDIT.condition_group_reownership(
            969, 1137, (truncated_clause,), S380_SOURCE, outline, old, new
        )
        is None
    )
