"""Execute printed T1170 table totals and the supported capture profile afresh."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import yaml

from axiom_encode.harness import capital_gift_capture
from axiom_encode.harness.bound_capital_gift_operands import GUIDE, PARENT
from axiom_encode.harness.capital_gift_evidence import certify_capital_gift
from axiom_encode.harness.capital_gift_execution import ENGINE_PIN
from axiom_encode.harness.local_case_custody import suite_scope
from axiom_encode.harness.local_compile_custody import compile_scope
from axiom_encode.harness.same_case_execution import compilation_snapshot
from axiom_encode.harness.validator_pipeline import (
    ValidatorPipeline,
    _authoritative_corpus_scope,
)
from tests.release_object_fixtures import bind_test_corpus_release
from tests.t1170_contract_projection import (
    FIXTURES,
    PREFIX,
    REMOVED_OUTPUTS,
    projected_execution,
    projected_static,
)


def _test_release(root, source, guide):
    """Authenticate public fixture text with the existing test signing key."""
    version = "t1170-table-contract-test"
    path = root / f"data/corpus/provisions/ca/policy/{version}.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {
            "id": f"t1170-source-{i}",
            "citation_path": citation,
            "body": body,
            "jurisdiction": "ca",
            "document_class": "policy",
            "version": version,
            "source_path": f"sources/ca/policy/{version}/fixture-source.txt",
            "source_as_of": "2025-12-31",
            "expression_date": "2025-01-01",
        }
        for i, (citation, body) in enumerate(((PARENT, source), (GUIDE, guide)))
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return bind_test_corpus_release(root, version, [("ca", "policy", version)])


def _cases(supported_capture):
    cases = yaml.safe_load(projected_execution()["companion_yaml"])
    if supported_capture:
        direct = json.loads((FIXTURES / "t1170-direct-refusals.json").read_text())
        extra = json.loads((FIXTURES / "t1170-phase-c-additions.json").read_text())
        cases += (
            [direct["separately_executed_positive"]] + direct["cases"] + extra["cases"]
        )
    return cases


def test_printed_totals_execute_with_nonzero_land_and_fresh_capture(
    tmp_path, real_engine
):
    pin = subprocess.check_output(
        ["git", "-C", str(real_engine.root), "rev-parse", "HEAD"], text=True
    ).strip()
    supported_capture = pin == ENGINE_PIN
    static = projected_static()
    cases = _cases(supported_capture)
    # This case previously asserted two cross-table totals that omitted land.
    totals = cases[7]["output"]
    assert totals[PREFIX + "t1170_ecological_land_zero_inclusion_line_68252"] == 34000
    assert (
        totals[PREFIX + "federal_2025_t1170_ecological_land_schedule_3_line_13800"]
        == 1000
    )
    assert REMOVED_OUTPUTS.isdisjoint(totals)
    root = tmp_path / "rulespec-ca"
    relative = Path("ca/policies/cra/t1-2025/capital-gains-on-gifts.yaml")
    target = root / relative
    target.parent.mkdir(parents=True)
    target.write_text(static["policy_yaml"])
    target.with_suffix(".test.yaml").write_text(yaml.safe_dump(cases, sort_keys=False))
    # The materializer requires an actual fixture checkout and clone, not a
    # caller-created ExecutionCapture or a patched custody predicate.
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=T1170 test fixture",
            "-c",
            "user.email=t1170@example.invalid",
            "commit",
            "-qm",
            "T1170 printed table test fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    source = (FIXTURES / "t1170-authenticated-body.txt").read_text()
    release = _test_release(tmp_path / "corpus", source, static["p113_body"])
    pipe = ValidatorPipeline(
        root / "ca",
        real_engine.root,
        local_corpus_release=release,
        enable_oracles=False,
        max_workers=1,
        require_complete_source_unit=True,
        axiom_rules_engine_ref=pin,
    )
    with (
        _authoritative_corpus_scope(release),
        compilation_snapshot(pipe, target) as cap,
    ):
        assert cap is not None and cap._capital_gift_owned
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
        assert issues == []
        if supported_capture:
            assert capital_gift_capture.valid(cap._capital_gift_context, cap)
            evidence = cap.evidence()
            observed = capital_gift_capture.observation(evidence, static["policy_yaml"])
            assert observed is not None
            context = certify_capital_gift(
                static["policy_yaml"], source, PARENT, cases, evidence
            )
            assert context.live() and not context.missing()
            actual_totals = dict(observed.positive.records[7].outputs)
            assert actual_totals[
                PREFIX + "t1170_ecological_land_zero_inclusion_line_68252"
            ]
            assert REMOVED_OUTPUTS.isdisjoint(actual_totals)
        else:
            # Other genuine engine pins may execute the tables; they cannot
            # acquire the separate, exact-89571 capture interpretation.
            assert cap._capital_gift_context is None
            assert (
                capital_gift_capture.observation(cap.evidence(), static["policy_yaml"])
                is None
            )
