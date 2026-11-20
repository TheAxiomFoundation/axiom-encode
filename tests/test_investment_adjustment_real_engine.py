"""Execute current Schedule N test inputs with genuine runtime custody."""

from __future__ import annotations

import copy
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

from axiom_encode.harness import investment_adjustment_capture as capture_module
from axiom_encode.harness import validator_pipeline as v
from axiom_encode.harness.bound_investment_adjustment_operands import GUIDE
from axiom_encode.harness.bound_investment_adjustment_structure import CITATION
from axiom_encode.harness.investment_adjustment_evidence import (
    certify_investment_adjustment,
)
from axiom_encode.harness.investment_adjustment_execution import ENGINE_PIN
from axiom_encode.harness.local_case_custody import suite_scope
from axiom_encode.harness.local_compile_custody import compile_scope
from axiom_encode.harness.same_case_execution import compilation_snapshot
from axiom_encode.harness.source_completeness import analyze_complete_source_unit
from axiom_encode.yaml_serialization import safe_dump_yaml
from tests.release_object_fixtures import bind_test_corpus_release

FIXTURES = Path(__file__).parent / "fixtures"
EVIDENCE_CHECK = "[complete-source-unit:schedule-n-evidence]"


def _write_result(root, name, value):
    (root / name).write_text(json.dumps(value, indent=2) + "\n")


def _test_release(root, source, guide):
    """Use the maintained signed TEST release over the exact public text."""
    version = "schedule-n-supported-runtime-test"
    path = root / f"data/corpus/provisions/ca/policy/{version}.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {
            "id": f"schedule-n-source-{i}",
            "citation_path": citation,
            "body": body,
            "jurisdiction": "ca",
            "document_class": "policy",
            "version": version,
            "source_path": f"sources/ca/policy/{version}/fixture-source.txt",
            "source_as_of": "2025-12-31",
            "expression_date": "2025-01-01",
        }
        for i, (citation, body) in enumerate(((CITATION, source), (GUIDE, guide)))
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return bind_test_corpus_release(root, version, [("ca", "policy", version)])


def test_schedule_n_executes_current_inputs_and_scopes_private_capture(
    tmp_path, real_engine
):
    pin = subprocess.check_output(
        ["git", "-C", str(real_engine.root), "rev-parse", "HEAD"], text=True
    ).strip()
    supported_capture = pin == ENGINE_PIN
    content = (FIXTURES / "schedule-n-static.yaml").read_text()
    companion = (
        FIXTURES / "schedule-n-materialized-diagnostic/candidate.test.yaml"
    ).read_bytes()
    source = (FIXTURES / "schedule-n-authenticated-body.txt").read_text()
    guide = (FIXTURES / "schedule-n-guide.txt").read_text()
    cases = v._safe_load_unique_keys(companion.decode())
    if type(cases) is dict and set(cases) == {"cases"}:
        cases = cases["cases"]
    assert len(cases) == 34
    if not supported_capture:
        # The expected-refusal transport and source authority require 89571.
        # Other genuine engines still execute every unchanged positive case.
        cases = [case for case in cases if "expected_refusal" not in case]
        assert len(cases) == 33
        companion = safe_dump_yaml(cases, sort_keys=False).encode()

    root = tmp_path / "rulespec-ca"
    relative = Path(
        "ca/policies/revenu-quebec/tp1-2025/schedule-n-investment-expense-adjustment.yaml"
    )
    target = root / relative
    target.parent.mkdir(parents=True)
    target.write_text(content)
    target.with_suffix(".test.yaml").write_bytes(companion)
    # Actual fixture checkout and clone; no caller-minted capture or stub.
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Schedule N test fixture",
            "-c",
            "user.email=schedule-n@example.invalid",
            "commit",
            "-qm",
            "Current Schedule N runtime test inputs",
        ],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    release = _test_release(tmp_path / "corpus", source, guide)
    pipe = v.ValidatorPipeline(
        root / "ca",
        real_engine.root,
        local_corpus_release=release,
        enable_oracles=False,
        max_workers=1,
        require_complete_source_unit=True,
        axiom_rules_engine_ref=pin,
    )
    assert pipe._axiom_rules_binary() == real_engine.binary
    with (
        v._authoritative_corpus_scope(release),
        compilation_snapshot(pipe, target) as cap,
    ):
        assert cap is not None and cap._investment_adjustment_owned
        execution = copy.copy(pipe)
        execution.policy_repo_path = cap.snapshot_root / "ca"
        execution.rulespec_dependency_roots = ()
        execution._register_validation_temporary_root(tmp_path)
        frozen = cap.snapshot_root / relative
        artifact = tmp_path / "compiled.json"
        with compile_scope(cap, execution, frozen, artifact):
            compiled, payload = execution._compile_rulespec_to_artifact(
                frozen, artifact
            )
        _write_result(
            tmp_path,
            "compile-result.json",
            {
                "returncode": compiled.returncode,
                "stdout": compiled.stdout,
                "stderr": compiled.stderr,
            },
        )
        assert compiled.returncode == 0, compiled.stderr
        assert payload is not None
        cap.compiled(artifact)
        with suite_scope(cap, execution, frozen, artifact, payload, cases) as suite:
            issues = execution._run_rulespec_test_cases(
                rules_file=frozen,
                compiled_path=artifact,
                compiled_payload=payload,
                cases=cases,
                execution_capture=cap,
            )
            if suite is not None:
                suite.finish(issues)
        _write_result(
            tmp_path,
            "companion-results.json",
            {
                "engine_pin": pin,
                "input_case_count": len(cases),
                "issues": issues,
                "positive_records": [asdict(r) for r in cap.records],
                "refusal_records": [
                    asdict(r) for r in cap._investment_adjustment_refusals
                ],
                "suite_records": [asdict(r) for r in suite.records.values()]
                if suite is not None
                else [],
            },
        )
        assert issues == [], "\n".join(issues)
        evidence = cap.evidence()
        if not supported_capture:
            assert cap._investment_adjustment_context is None
            assert capture_module.observation(evidence, content) is None
            context = certify_investment_adjustment(
                content, source, CITATION, cases, evidence
            )
            assert not context.live()
            assert any(EVIDENCE_CHECK in issue for issue in context.missing())
            return

        assert capture_module.valid(cap._investment_adjustment_context, cap)
        observed = capture_module.observation(evidence, content)
        assert observed is not None
        assert len(observed.positive.records) == 33
        assert len(observed.refusals) == len(observed.pairs) == 1
        context = certify_investment_adjustment(
            content, source, CITATION, cases, evidence
        )
        assert context.live() and not context.missing()

        def analyze(value):
            return analyze_complete_source_unit(
                content,
                source,
                corpus_citation_path=CITATION,
                test_cases=cases,
                extract_numeric_occurrences=v.extract_typed_numeric_inventory_occurrences_from_text,
                extract_numeric_grounding_occurrences=v.extract_typed_numeric_occurrences_from_text,
                extract_named_scalars=v.extract_named_scalar_occurrences,
                numeric_value_is_grounded=v.numeric_value_is_grounded,
                source_context={CITATION: source, GUIDE: guide},
                execution_evidence=value,
            )

        actual = analyze(evidence)
        rejected = []
        for missing_evidence in (None, replace(evidence)):
            unbound = certify_investment_adjustment(
                content, source, CITATION, cases, missing_evidence
            )
            assert not unbound.live()
            assert any(EVIDENCE_CHECK in issue for issue in unbound.missing())
            result = analyze(missing_evidence)
            rejected.append(list(result.issues))
            assert any(EVIDENCE_CHECK in issue for issue in result.issues)
        _write_result(
            tmp_path,
            "source-certification.json",
            {
                "certification_live": context.live(),
                "missing": list(context.missing()),
                "coverage_missing": list(observed.coverage.missing),
                "complete_source_issues": list(actual.issues),
                "no_evidence_issues": rejected[0],
                "copied_evidence_issues": rejected[1],
            },
        )
        assert not actual.issues, "\n".join(actual.issues)
