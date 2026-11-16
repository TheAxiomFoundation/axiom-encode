"""Genuine compile/test capture; fixture corpus is not signed-release admission."""

import copy
import subprocess
from dataclasses import replace
from pathlib import Path

import yaml

from axiom_encode.harness import actual_floor_evidence as f
from axiom_encode.harness import source_completeness as s
from axiom_encode.harness.same_case_execution import compilation_snapshot
from axiom_encode.harness.validator_pipeline import ValidatorPipeline
from tests.test_same_page_floor_binding import BODY, CITATION, CONTENT, PAYLOAD


def test_actual_choice_observations_and_negative_custody(
    real_engine, tmp_path, monkeypatch
):
    pin = subprocess.check_output(
        ["git", "-C", str(real_engine.root), "rev-parse", "HEAD"], text=True
    ).strip()
    root = tmp_path / "rulespec-ca"
    rules = root / "ca/policies/cra/t1-2025/federal-spouse-transfer-amounts.yaml"
    rules.parent.mkdir(parents=True)
    rules.write_text(CONTENT)
    all_cases = yaml.safe_load(
        (
            Path(__file__).parent / "fixtures/same_page_floor_candidate.test.yaml"
        ).read_text()
    )
    cases = [all_cases[i] for i in (0, 7, 8, 11, 12, 13, 14, 15)]
    rules.with_suffix(".test.yaml").write_text(yaml.safe_dump(cases, sort_keys=False))
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "exact fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    pipe = ValidatorPipeline(
        root / "ca",
        real_engine.root,
        axiom_rules_engine_ref=pin,
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
        source_text=BODY,
        source_citation_path=CITATION,
    )
    # Fixture source only. No private record, receipt, capture or engine result
    # is supplied or overridden: ordinary execution must create all of them.
    monkeypatch.setattr(
        pipe, "_source_texts_for_rulespec_content", lambda _: {CITATION: BODY}
    )
    monkeypatch.setattr(
        pipe,
        "_proof_source_texts_for_rulespec_content",
        lambda _, source_texts: source_texts,
    )
    principal = {r["name"]: r for r in PAYLOAD["rules"] if r["kind"] == "derived"}
    with compilation_snapshot(pipe, rules) as capture:
        assert capture is not None
        execution = copy.copy(pipe)
        execution.policy_repo_path = capture.snapshot_root / "ca"
        target = capture.snapshot_root / capture.root_relative
        artifact = tmp_path / "compiled.json"
        result, compiled = execution._compile_rulespec_to_artifact(target, artifact)
        assert result.returncode == 0, result.stderr
        capture.compiled(artifact)
        assert (
            execution._run_rulespec_test_cases(
                rules_file=target,
                compiled_path=artifact,
                compiled_payload=compiled,
                cases=cases,
                execution_capture=capture,
            )
            == []
        )
        evidence = capture.evidence()
        assert evidence is not None
        bound = f.bind_same_page_choices(CONTENT, {CITATION: BODY})
        for case, row in zip(cases[:6], (7, 1, 1, 7, 7, 7), strict=True):
            required = {
                bound[row]["name"],
                "federal_amounts_transferred_from_spouse_line_32600",
            }
            kwargs = dict(
                content=CONTENT,
                source_context={CITATION: BODY},
                required_outputs=required,
                principal_rules=principal,
                formula_environment=s._constant_rule_environment(PAYLOAD),
            )
            observation = f._same_page_choice_observation(evidence, case, **kwargs)
            assert observation is not None, case["name"]
            assert (
                f._same_page_choice_observation(
                    evidence, case, **{**kwargs, "required_outputs": set()}
                )
                is None
            )
            forged = dict(kwargs["formula_environment"])
            constant = "federal_2025_schedule_2_latest_birth_year_for_age_amount"
            assert constant in forged
            original = forged[constant]
            forged[constant] = replace(
                original,
                versions=tuple(
                    (start, end, 3000) for start, end, _ in original.versions
                ),
            )
            assert forged != kwargs["formula_environment"]
            assert (
                f._same_page_choice_observation(
                    evidence, case, **{**kwargs, "formula_environment": forged}
                )
                is None
            )
            query, observed, executions = observation
            assert required <= executions.keys()
            assert required <= observed.keys()
            assert not f._same_page_choice_observation(
                replace(evidence, artifact_bytes=b"{}"), case, **kwargs
            )
            changed = copy.deepcopy(case)
            changed["period"] = "2024"
            assert not f._same_page_choice_observation(evidence, changed, **kwargs)
            assert not f._same_page_choice_observation(
                evidence, case, **dict(kwargs, required_outputs={"unasserted_output"})
            )

        structure = s.recognize_source_structure(BODY, corpus_citation_path=CITATION)
        branches = s._source_exception_branches(
            BODY,
            branches=structure,
            active_branches=structure,
            deferred_paths=set(),
            corpus_citation_path=CITATION,
        )
        selected = [b for b in branches if b.start in (330, 1182, 1996)]
        assert len(selected) == 3

        def arguments(subset):
            return dict(
                content=CONTENT,
                source_text=BODY,
                corpus_citation_path=CITATION,
                principal_rules=principal,
                asserted_by_rule={
                    n: [
                        c
                        for c in subset
                        if any(k.endswith("#" + n) for k in c["output"])
                    ]
                    for n in principal
                },
                formula_environment=s._constant_rule_environment(PAYLOAD),
                source_context={CITATION: BODY},
            )

        tokens = {
            b.start: f.same_page_choice_witnesses(evidence, b, **arguments(cases))
            for b in selected
        }
        assert all(len(t) == 1 for t in tokens.values()), tokens
        assert len(set.union(*tokens.values())) == 3
        for b in selected:
            assert not f.same_page_choice_witnesses(
                evidence, replace(b, start=b.start + 1), **arguments(cases)
            )
            assert not f.same_page_choice_witnesses(
                evidence, replace(b, text=b.text + " "), **arguments(cases)
            )
            assert not f.same_page_choice_witnesses(
                replace(evidence, artifact_bytes=b"{}"), b, **arguments(cases)
            )
        age = next(b for b in selected if b.start == 1182)
        threshold = next(b for b in selected if b.start == 1996)
        unused = next(b for b in selected if b.start == 330)
        for excluded in ({0, 1}, {2}):
            assert not f.same_page_choice_witnesses(
                evidence,
                age,
                **arguments([c for i, c in enumerate(cases) if i not in excluded]),
            )
        for index in (3, 5):
            assert not f.same_page_choice_witnesses(
                evidence,
                threshold,
                **arguments([c for i, c in enumerate(cases) if i != index]),
            )
        assert not f.same_page_choice_witnesses(
            evidence, unused, **arguments(cases[:6])
        )
