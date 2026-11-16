"""Genuine compile/test capture; fixture corpus is not signed-release admission."""

import copy
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import yaml

from axiom_encode.harness import actual_floor_evidence as f
from axiom_encode.harness import source_completeness as s
from axiom_encode.harness.same_case_execution import compilation_snapshot
from axiom_encode.harness.validator_pipeline import ValidatorPipeline
from tests.test_same_page_floor_binding import BODY, CITATION, CONTENT, PAYLOAD


def test_actual_local_floor_capture_and_negative_custody(
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
    cases = [c for c in all_cases if c["name"] in ("line12_floor", "line13_floor")]
    assert len(cases) == 2
    blocked = copy.deepcopy(cases)
    for case in blocked:
        case["name"] += "_relationship_blocked"
        spouse = next(
            k
            for k in case["input"]
            if k.endswith("#input.spouse_or_common_law_partner")
        )
        case["input"][spouse] = False
        final = next(
            k
            for k in case["output"]
            if k.endswith("#federal_amounts_transferred_from_spouse_line_32600")
        )
        case["output"][final] = 0
    cases += blocked
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
    asserted = {
        n: [c for c in cases if any(k.endswith("#" + n) for k in c["output"])]
        for n in principal
    }
    branches = s.recognize_source_structure(BODY, corpus_citation_path=CITATION)
    floors = [
        b
        for b in s._source_exception_branches(
            BODY,
            branches=branches,
            active_branches=branches,
            deferred_paths=set(),
            corpus_citation_path=CITATION,
        )
        if "minus" in b.text
    ]
    kwargs = dict(
        content=CONTENT,
        source_text=BODY,
        corpus_citation_path=CITATION,
        principal_rules=principal,
        asserted_by_rule=asserted,
        formula_environment=s._constant_rule_environment(PAYLOAD),
        source_context={CITATION: BODY},
    )
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
        tokens = [f.same_page_witnesses(evidence, b, **kwargs) for b in floors]
        assert len(tokens) == 2 and all(tokens)
        assert tokens[0].isdisjoint(tokens[1])
        from axiom_encode.harness.validator_pipeline import (
            extract_named_scalar_occurrences,
            extract_typed_numeric_occurrences_from_text,
            numeric_value_is_grounded,
        )

        analysis_args = dict(
            corpus_citation_path=CITATION,
            source_context={CITATION: BODY},
            test_cases=cases,
            extract_numeric_occurrences=extract_typed_numeric_occurrences_from_text,
            extract_named_scalars=extract_named_scalar_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
        )
        without = s.analyze_complete_source_unit(CONTENT, BODY, **analysis_args)
        with_capture = s.analyze_complete_source_unit(
            CONTENT, BODY, execution_evidence=evidence, **analysis_args
        )
        assert with_capture.issues  # still not a complete source encoding
        before_conditions = "\n".join(
            str(i)
            for i in without.issues
            if "Source-stated exceptions or applicability" in str(i)
        )
        after_conditions = "\n".join(
            str(i)
            for i in with_capture.issues
            if "Source-stated exceptions or applicability" in str(i)
        )
        for phrase in ("Line 7 minus line 11", "Line 6 minus line 12"):
            assert phrase in before_conditions
            assert phrase not in after_conditions
        blocked_assertions = {
            n: [c for c in blocked if any(k.endswith("#" + n) for k in c["output"])]
            for n in principal
        }
        blocked_kwargs = dict(kwargs, asserted_by_rule=blocked_assertions)
        assert f.same_page_witnesses(evidence, floors[0], **blocked_kwargs)
        assert not f.same_page_witnesses(evidence, floors[1], **blocked_kwargs)
        # Negative-only altered evidence: none may inherit the real positive.
        evidence = replace(evidence, records=evidence.records[:2])
        for malformed in (
            b"{",
            b'{"program":null}',
            b'{"program":{"derived":null}}',
            b'{"program":{"derived":[null]}}',
        ):
            assert not f.same_page_witnesses(
                replace(evidence, artifact_bytes=malformed), floors[0], **kwargs
            )
        wrong_citation = dict(
            kwargs,
            corpus_citation_path="ca/policy/other",
            source_context={CITATION: BODY, "ca/policy/other": BODY},
        )
        assert not f.same_page_witnesses(evidence, floors[0], **wrong_citation)
        first = evidence.records[0]
        wrong_ordinal = replace(first, case_index=2)
        assert not f.same_page_witnesses(
            replace(evidence, records=(wrong_ordinal, *evidence.records[1:])),
            floors[0],
            **kwargs,
        )
        owner_ref = next(k for k in cases[0]["output"] if k.endswith("_line_12"))
        for mutation in ("public_result", "missing_dependency", "omitted_query_result"):
            response = json.loads(first.response_json)
            request = json.loads(first.request_json)
            if mutation == "public_result":
                response["results"][0]["outputs"][owner_ref]["value"] = {
                    "kind": "decimal",
                    "value": "99999",
                }
            elif mutation == "missing_dependency":
                dependency = next(
                    k for k in response["results"][0]["trace"] if k.endswith("_line_7")
                )
                del response["results"][0]["trace"][dependency]
            else:
                request["queries"][0]["outputs"].remove(owner_ref)
                del response["results"][0]["outputs"][owner_ref]
            altered = replace(
                first,
                request_json=json.dumps(request),
                response_json=json.dumps(response),
            )
            assert not f.same_page_witnesses(
                replace(evidence, records=(altered, *evidence.records[1:])),
                floors[0],
                **kwargs,
            )
        assert not f.same_page_witnesses(
            evidence, replace(floors[0], start=floors[0].start + 1), **kwargs
        )
        assert not f.same_page_witnesses(
            evidence, replace(floors[0], kind="formula"), **kwargs
        )
        duplicate = replace(floors[0], label="duplicate")
        unmatched = s._unmatched_evidence_obligations(
            {floors[0]: tokens[0], duplicate: tokens[0]}
        )
        assert len(unmatched) == 1
        request = json.loads(first.request_json)
        request["dataset"]["inputs"][0]["value"] = {"kind": "integer", "value": 999999}
        wrong_input = replace(first, request_json=json.dumps(request))
        assert not f.same_page_witnesses(
            replace(evidence, records=(wrong_input, *evidence.records[1:])),
            floors[0],
            **kwargs,
        )
