"""Eval repair loop wiring for complete-source companion-case synthesis."""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from axiom_encode.harness import evals as evals_module

BOUNDARY_ISSUE = (
    "[complete-source-unit:tests] Companion tests do not exercise every "
    "source-stated boundary input; missing: (a)=500."
)
PAIRED_ISSUE = (
    "[complete-source-unit:tests] Source-stated exceptions or applicability "
    "conditions require paired positive/blocking cases; missing: us/x(a) "
    "[change]: `x`."
)


def _run_loop(tmp_path: Path, *, issues, complete: bool, repairs):
    metrics = SimpleNamespace(ci_issues=list(issues))
    rulespec_file = tmp_path / "artifact.yaml"
    rulespec_file.write_text("initial\n")
    with (
        patch.object(evals_module, "evaluate_artifact", return_value=metrics) as ev,
        patch.object(
            evals_module, "_apply_generated_eval_repairs", side_effect=repairs
        ) as rep,
        patch.object(
            evals_module,
            "_append_synthesized_complete_source_companion_cases",
            return_value=["auto_source_boundary_countable_income_500"],
        ) as synth,
    ):
        result = evals_module._evaluate_generated_artifact_with_repairs(
            rulespec_file=rulespec_file,
            policy_repo_root=tmp_path / "rulespec-us" / "us",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            source_text="Source body",
            local_corpus_release=object(),
            require_complete_source_unit=complete,
        )
    assert result is metrics
    return ev, rep, synth


def test_synthesis_runs_when_no_other_repair_applies(tmp_path):
    evaluate, repair, synthesize = _run_loop(
        tmp_path, issues=[BOUNDARY_ISSUE], complete=True, repairs=[[], []]
    )

    synthesize.assert_called_once()
    assert synthesize.call_args.kwargs["source_text"] == "Source body"
    # The loop re-evaluates after appending; an unchanged artifact then stops it.
    assert evaluate.call_count == 2
    assert repair.call_count == 1


@pytest.mark.parametrize(
    ("issues", "complete", "repairs"),
    [
        ([BOUNDARY_ISSUE], False, [[]]),
        ([PAIRED_ISSUE], True, [[]]),
        (["[complete-source-unit:structure] Source branch (b)."], True, [[]]),
    ],
)
def test_synthesis_is_skipped_outside_its_scope(tmp_path, issues, complete, repairs):
    _evaluate, _repair, synthesize = _run_loop(
        tmp_path, issues=issues, complete=complete, repairs=repairs
    )

    synthesize.assert_not_called()


def test_other_repairs_take_precedence_in_a_round(tmp_path):
    rulespec_file = tmp_path / "artifact.yaml"
    rulespec_file.write_text("initial\n")
    metrics = SimpleNamespace(ci_issues=[BOUNDARY_ISSUE])
    rounds = []

    def repair(**_kwargs):
        rounds.append("repair")
        if len(rounds) == 1:
            rulespec_file.write_text("repaired\n")
            return ["proof_import:x"]
        return []

    with (
        patch.object(evals_module, "evaluate_artifact", return_value=metrics),
        patch.object(evals_module, "_apply_generated_eval_repairs", side_effect=repair),
        patch.object(
            evals_module,
            "_append_synthesized_complete_source_companion_cases",
            return_value=[],
        ) as synthesize,
    ):
        evals_module._evaluate_generated_artifact_with_repairs(
            rulespec_file=rulespec_file,
            policy_repo_root=tmp_path / "rulespec-us" / "us",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            source_text="Source body",
            local_corpus_release=object(),
            require_complete_source_unit=True,
        )

    assert rounds == ["repair", "repair"]
    synthesize.assert_called_once()


def test_synthesis_rounds_are_bounded(tmp_path):
    rulespec_file = tmp_path / "artifact.yaml"
    rulespec_file.write_text("initial\n")
    test_file = tmp_path / "artifact.test.yaml"
    metrics = SimpleNamespace(ci_issues=[BOUNDARY_ISSUE])
    appended = []

    def append(**_kwargs):
        appended.append(f"auto_source_{len(appended)}")
        test_file.write_text(yaml.safe_dump(appended))
        return [appended[-1]]

    with (
        patch.object(evals_module, "evaluate_artifact", return_value=metrics) as ev,
        patch.object(evals_module, "_apply_generated_eval_repairs", return_value=[]),
        patch.object(
            evals_module,
            "_append_synthesized_complete_source_companion_cases",
            side_effect=append,
        ) as synthesize,
    ):
        evals_module._evaluate_generated_artifact_with_repairs(
            rulespec_file=rulespec_file,
            policy_repo_root=tmp_path / "rulespec-us" / "us",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            source_text="Source body",
            local_corpus_release=object(),
            require_complete_source_unit=True,
        )

    assert synthesize.call_count == evals_module._COMPLETE_SOURCE_CASE_SYNTHESIS_ROUNDS
    assert ev.call_count == synthesize.call_count + 1


@pytest.fixture
def staged(tmp_path: Path):
    rulespec_file = tmp_path / "statutes" / "7" / "9999.yaml"
    rulespec_file.parent.mkdir(parents=True)
    rulespec_file.write_text("format: rulespec/v1\n")
    test_file = rulespec_file.with_name("9999.test.yaml")
    existing = [
        {"name": "low", "period": "2026-01", "input": {}, "output": {"x": 1}},
    ]
    test_file.write_text(yaml.safe_dump(existing, sort_keys=False))

    @contextlib.contextmanager
    def validation_target(path, *_args, **_kwargs):
        yield path

    with (
        patch.object(evals_module, "_rulespec_validation_target", validation_target),
        patch.object(
            evals_module,
            "_authoritative_corpus_scope",
            lambda _release: contextlib.nullcontext(),
        ),
        patch.object(
            evals_module,
            "_validation_policy_repo_root",
            lambda _file, root: root,
        ),
        patch.object(
            evals_module,
            "_validation_rulespec_dependency_roots",
            lambda **_kwargs: (),
        ),
    ):
        yield rulespec_file, test_file, existing


def _append(rulespec_file: Path, tmp_path: Path) -> list[str]:
    return evals_module._append_synthesized_complete_source_companion_cases(
        rulespec_file=rulespec_file,
        policy_repo_root=tmp_path / "rulespec-us" / "us",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        axiom_compose_path=None,
        source_text="Source body",
        local_corpus_release=None,
        source_metadata=None,
        source_citation_path="us/statute/7/9999",
        rulespec_dependency_roots=(),
        legacy_replacement=None,
        replacement_overlay_scope=False,
    )


def test_append_writes_only_new_synthesized_cases(staged, tmp_path):
    rulespec_file, test_file, existing = staged
    synthesized = [
        {"name": "low", "period": "2026-01", "input": {}, "output": {"x": 2}},
        {
            "name": "auto_source_boundary_income_500",
            "period": "2026-01",
            "input": {"income": 500},
            "output": {"x": 0},
        },
    ]
    with patch.object(
        evals_module.ValidatorPipeline,
        "synthesize_complete_source_companion_cases",
        return_value=synthesized,
    ) as synthesize:
        names = _append(rulespec_file, tmp_path)

    assert names == ["auto_source_boundary_income_500"]
    assert synthesize.call_args.args == (rulespec_file,)
    assert yaml.safe_load(test_file.read_text()) == [*existing, synthesized[1]]


def test_append_leaves_the_companion_file_untouched_on_failure(staged, tmp_path):
    rulespec_file, test_file, _existing = staged
    original = test_file.read_bytes()
    with patch.object(
        evals_module.ValidatorPipeline,
        "synthesize_complete_source_companion_cases",
        side_effect=RuntimeError("engine unavailable"),
    ):
        assert _append(rulespec_file, tmp_path) == []

    assert test_file.read_bytes() == original


SYNTHESIZED = {
    "name": "auto_source_boundary_income_500",
    "period": "2026-01",
    "input": {"income": 500},
    "output": {"x": 0},
}


def test_append_keeps_comments_of_the_model_written_file(staged, tmp_path):
    rulespec_file, test_file, existing = staged
    test_file.write_text(
        "# Written by the encoder; keep this note.\n"
        + yaml.safe_dump(existing, sort_keys=False)
    )
    with patch.object(
        evals_module.ValidatorPipeline,
        "synthesize_complete_source_companion_cases",
        return_value=[SYNTHESIZED],
    ):
        assert _append(rulespec_file, tmp_path) == [SYNTHESIZED["name"]]

    text = test_file.read_text()
    assert text.startswith("# Written by the encoder; keep this note.\n")
    assert yaml.safe_load(text) == [*existing, SYNTHESIZED]


def test_append_supports_a_cases_mapping(staged, tmp_path):
    rulespec_file, test_file, existing = staged
    test_file.write_text(yaml.safe_dump({"cases": existing}, sort_keys=False))
    with patch.object(
        evals_module.ValidatorPipeline,
        "synthesize_complete_source_companion_cases",
        return_value=[SYNTHESIZED],
    ):
        assert _append(rulespec_file, tmp_path) == [SYNTHESIZED["name"]]

    assert yaml.safe_load(test_file.read_text()) == {"cases": [*existing, SYNTHESIZED]}


def test_append_builds_the_pipeline_exactly_as_evaluation_does(staged, tmp_path):
    rulespec_file, _test_file, _existing = staged
    fake = SimpleNamespace(
        synthesize_complete_source_companion_cases=lambda *_a, **_k: []
    )
    amendment = SimpleNamespace(citation_path="us/statute/1", body="text")
    with patch.object(
        evals_module, "_eval_validator_pipeline", return_value=fake
    ) as build:
        evals_module._append_synthesized_complete_source_companion_cases(
            rulespec_file=rulespec_file,
            policy_repo_root=tmp_path / "rulespec-us" / "us",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            axiom_compose_path=None,
            oracle="policyengine",
            policyengine_runtime=None,
            policyengine_rule_hint="household_benefit",
            source_text="Source body",
            local_corpus_release=None,
            source_metadata={"k": "v"},
            source_citation_path="us/statute/7/9999",
            rulespec_dependency_roots=(),
            amendment_documents=(amendment,),
            legacy_replacement=None,
            replacement_overlay_scope=False,
        )

    kwargs = build.call_args.kwargs
    assert kwargs["oracle"] == "policyengine"
    assert kwargs["policyengine_rule_hint"] == "household_benefit"
    assert kwargs["amendment_documents"] == (amendment,)
    assert kwargs["source_metadata"] == {"k": "v"}
    assert kwargs["require_complete_source_unit"] is True
