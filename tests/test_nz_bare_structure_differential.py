"""Check the evidence used to certify the corpus differential."""

import re
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

from axiom_encode.bare_structure import bare_outline, mask_bare_labels
from axiom_encode.harness.source_completeness import SourceStructureBranch

SCRIPT = Path(__file__).parents[1] / "scripts" / "nz_bare_structure_differential.py"
SPEC = spec_from_file_location("_nz_bare_differential_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


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


def test_singleton_descendant_projection_preserves_source_and_downstream_demands():
    source = "1\nA person is entitled if—\na\nthe person is aged 65 years.\n"
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
