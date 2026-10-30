"""Execute the approval graph and the pre-checkout promotion boundary."""

from __future__ import annotations

import ast
import json
import os
import re
import shlex
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/targeted-signed-reencode.yml"
GENERATION_INPUTS = (
    "review_finding",
    "repair_run_id",
    "dependent_citation",
    "dependent_review_finding",
    "second_dependent_citation",
    "second_dependent_review_finding",
    "replace_legacy_rulespec_path",
    "legacy_exact_dependent_rulespec_path",
    "second_legacy_exact_dependent_rulespec_path",
    "queue_id",
    "queue_item_id",
    "queue_manifest_sha256",
    "queue_item_generation_sha256",
    "queue_dispatcher_run_id",
)
JSON_GENERATION_INPUTS = (
    "existing_signed_imports_json",
    "legacy_retained_successor_rulespec_paths_json",
)
RESULTS = ("success", "skipped", "failure", "cancelled")


@lru_cache(maxsize=1)
def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _promotion_contract() -> dict:
    return {
        "schema": "axiom-encode/atomic-source-transaction/v5",
        "source_bundle": [],
        "canonical_refresh_bundle": [],
        "primary_required_test_cases": [],
        "require_complete_source_unit": True,
        "manifest_only_refresh": False,
        "reviewed_candidate_promotion": True,
    }


def _promotion_dispatch() -> dict:
    trigger = _workflow().get("on", _workflow().get(True))
    inputs = trigger["workflow_dispatch"]["inputs"]
    return {
        **{name: spec.get("default", "") for name, spec in inputs.items()},
        "citation": "nz/legislation/example/s75",
        "country": "nz",
        "rulespec_ref": "a" * 40,
        "corpus_ref": "b" * 40,
        "rules_engine_ref": "c" * 40,
        "source_bundle_json": json.dumps(_promotion_contract()),
        "replace_rulespec_path": "nz/acts/example/s75.yaml",
    }


def _run_step(tmp_path: Path, step: dict, env: dict) -> subprocess.CompletedProcess:
    # Use the actual shell/heredoc; only the interpreter changes for portability.
    script = step["run"].replace("python3 -I -", f"{shlex.quote(sys.executable)} -I -")
    return subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=tmp_path,
        env={"PATH": os.defpath, **env},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def _classify(tmp_path: Path, raw: str) -> tuple[subprocess.CompletedProcess, str]:
    output = tmp_path / "classification-output.txt"
    output.unlink(missing_ok=True)
    [step] = _workflow()["jobs"]["classify_run"]["steps"]
    completed = _run_step(
        tmp_path,
        step,
        {"ATOMIC_SOURCE_JSON": raw, "GITHUB_OUTPUT": str(output)},
    )
    return completed, output.read_text() if output.exists() else ""


def _guard(
    tmp_path: Path, dispatch: dict, *, is_promotion: str = "true"
) -> subprocess.CompletedProcess:
    return _run_step(
        tmp_path,
        _workflow()["jobs"]["encode"]["steps"][0],
        {"DISPATCH_INPUTS_JSON": json.dumps(dispatch), "IS_PROMOTION": is_promotion},
    )


def _evaluate_condition(expression: str, variables: dict) -> bool:
    expression = expression.strip()
    assert expression.startswith("${{") and expression.endswith("}}")
    expression = expression[3:-2].replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", "not ", expression)
    tree = ast.parse(f"({expression.strip()})", mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Attribute,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Call,
    )
    for node in ast.walk(tree):
        assert isinstance(node, allowed_nodes), ast.dump(node)
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name) and node.func.id in {
                "cancelled",
                "failure",
            }
            assert not node.args and not node.keywords
    return bool(
        eval(
            compile(tree, "<workflow job condition>", "eval"),
            {"__builtins__": {}},
            variables,
        )
    )


def _enabled(job_name: str, **changes) -> bool:
    context = {
        "promotion": True,
        "approval_result": "skipped",
        "budget_result": "success",
        "classifier_result": "success",
        "cancelled": False,
        "event_name": "workflow_dispatch",
        "ref": "refs/heads/main",
        "queue_id": "",
        "actor": "repository-writer",
        "run_attempt": 1,
        **changes,
    }
    context.setdefault("approved_run_attempt", context["run_attempt"])
    return _evaluate_condition(
        _workflow()["jobs"][job_name]["if"],
        {
            "cancelled": lambda: context["cancelled"],
            "github": SimpleNamespace(
                event_name=context["event_name"],
                ref=context["ref"],
                actor=context["actor"],
                run_attempt=context["run_attempt"],
            ),
            "inputs": SimpleNamespace(queue_id=context["queue_id"]),
            "needs": SimpleNamespace(
                attempt_budget=SimpleNamespace(result=context["budget_result"]),
                classify_run=SimpleNamespace(
                    result=context["classifier_result"],
                    outputs=SimpleNamespace(
                        is_promotion=str(context["promotion"]).lower()
                    ),
                ),
                approve_generation=SimpleNamespace(
                    result=context["approval_result"],
                    outputs=SimpleNamespace(
                        approved_run_attempt=context["approved_run_attempt"]
                    ),
                ),
            ),
        },
    )


def test_approval_graph_has_one_secretless_generation_gate() -> None:
    jobs = _workflow()["jobs"]
    assert set(jobs["approve_generation"]["needs"]) == {
        "attempt_budget",
        "classify_run",
    }
    assert set(jobs["encode"]["needs"]) == {
        "attempt_budget",
        "classify_run",
        "approve_generation",
    }
    assert jobs["encode"]["environment"] == "apply-signing"
    approval = jobs["approve_generation"]
    assert approval["environment"] == "generation-approval"
    assert approval["permissions"] == {"actions": "read"}
    assert "env" not in approval
    assert approval["outputs"] == {
        "approved_run_attempt": "${{ steps.record.outputs.run_attempt }}"
    }
    [reviewer_check, approval_step] = approval["steps"]
    assert reviewer_check["name"] == "Require reviewers on the generation gate"
    assert reviewer_check["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "REPOSITORY": "${{ github.repository }}",
    }
    assert "uses" not in reviewer_check
    assert set(approval_step) == {"name", "id", "run"}
    assert approval_step["id"] == "record"
    assert approval_step["run"] == (
        'echo "run_attempt=$GITHUB_RUN_ATTEMPT" >> "$GITHUB_OUTPUT"'
    )

    classifier = jobs["classify_run"]
    assert classifier["permissions"] == {}
    assert "environment" not in classifier and "env" not in classifier
    assert classifier["outputs"] == {
        "is_promotion": "${{ steps.classify.outputs.is_promotion }}"
    }
    [classification_step] = classifier["steps"]
    assert "uses" not in classification_step
    assert classification_step["env"] == {
        "ATOMIC_SOURCE_JSON": "${{ inputs.source_bundle_json }}"
    }
    assert "secrets." not in json.dumps(classifier)
    assert "secrets." not in json.dumps(approval)


def test_no_job_signs_in_the_environment_earlier_revisions_use() -> None:
    # GitHub re-runs a run with its original workflow file for 30 days, and
    # earlier revisions sign in production-signing with no approve_generation
    # job. Keeping the key out of every environment those revisions name is what
    # stops a re-run from signing without approval, so this workflow must never
    # name production-signing again.
    workflow = _workflow()
    environments = {
        name: job.get("environment") for name, job in workflow["jobs"].items()
    }
    assert "production-signing" not in json.dumps(environments)
    code = [
        line
        for line in WORKFLOW.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    assert not [line for line in code if "production-signing" in line]
    signing = [
        name
        for name, job in workflow["jobs"].items()
        if "AXIOM_ENCODE_APPLY_SIGNING_KEY" in json.dumps(job)
    ]
    assert signing == ["encode"]
    assert workflow["jobs"]["encode"]["environment"] == "apply-signing"


def test_promotion_boundary_runs_before_checkout_or_network_access() -> None:
    [guard, checkout, *_] = _workflow()["jobs"]["encode"]["steps"]
    assert guard["name"] == "Reject mixed promotion inputs"
    assert guard["id"] == "promotion_input_guard"
    assert "uses" not in guard and "if" not in guard
    assert guard["env"] == {
        "DISPATCH_INPUTS_JSON": "${{ toJSON(inputs) }}",
        "IS_PROMOTION": "${{ needs.classify_run.outputs.is_promotion }}",
    }
    assert "secrets." not in json.dumps(guard)
    assert checkout["name"] == "Checkout axiom-encode"
    assert checkout["uses"].startswith("actions/checkout@")
    for step in (guard, _workflow()["jobs"]["classify_run"]["steps"][0]):
        python = step["run"].split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        imports = {
            alias.name
            for node in ast.walk(ast.parse(python))
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert imports == {"json", "os"}
        assert not any(
            isinstance(node, ast.ImportFrom) for node in ast.walk(ast.parse(python))
        )


@pytest.mark.parametrize("guard_outcome", RESULTS)
@pytest.mark.parametrize(
    "failure,cancelled", [(True, False), (False, False), (True, True)]
)
def test_rejected_promotion_skips_failure_diagnostics_and_network_upload(
    guard_outcome: str, failure: bool, cancelled: bool
) -> None:
    steps = _workflow()["jobs"]["encode"]["steps"]
    for name in (
        "Package failed re-encode diagnostics",
        "Upload failed re-encode diagnostics",
    ):
        step = next(step for step in steps if step.get("name") == name)
        assert "steps.promotion_input_guard.outcome == 'success'" in step["if"]
        allowed = _evaluate_condition(
            step["if"],
            {
                "failure": lambda: failure,
                "cancelled": lambda: cancelled,
                "steps": SimpleNamespace(
                    promotion_input_guard=SimpleNamespace(outcome=guard_outcome),
                    package_failed_reencode_diagnostics=SimpleNamespace(
                        outcome="success"
                    ),
                ),
            },
        )
        assert allowed is (failure and not cancelled and guard_outcome == "success")


@pytest.mark.parametrize("budget_result", ["success", "skipped"])
def test_promotions_skip_approval_and_reach_encode(
    tmp_path: Path, budget_result: str
) -> None:
    dispatch = _promotion_dispatch()
    result, output = _classify(tmp_path, dispatch["source_bundle_json"])
    assert result.returncode == 0, result.stderr
    assert output == "is_promotion=true\n"
    assert not _enabled("approve_generation", budget_result=budget_result)
    assert _enabled("encode", budget_result=budget_result)
    result = _guard(tmp_path, dispatch)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("approval_result", RESULTS)
def test_generation_requires_successful_approval(
    tmp_path: Path, approval_result: str
) -> None:
    result, output = _classify(tmp_path, "[]")
    assert result.returncode == 0, result.stderr
    assert output == "is_promotion=false\n"
    assert _enabled("approve_generation", promotion=False)
    assert _enabled("encode", promotion=False, approval_result=approval_result) is (
        approval_result == "success"
    )
    dispatch = _promotion_dispatch()
    dispatch["source_bundle_json"] = "[]"
    dispatch["review_finding"] = "Generate a correction after approval."
    result = _guard(tmp_path, dispatch, is_promotion="false")
    assert result.returncode == 0, result.stderr


def test_approval_records_the_attempt_it_approved(tmp_path: Path) -> None:
    output = tmp_path / "approval-output.txt"
    [_reviewer_check, step] = _workflow()["jobs"]["approve_generation"]["steps"]
    result = _run_step(
        tmp_path,
        step,
        {"GITHUB_RUN_ATTEMPT": "3", "GITHUB_OUTPUT": str(output)},
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == "run_attempt=3\n"


def _check_generation_gate(
    tmp_path: Path, environment: dict | None
) -> subprocess.CompletedProcess:
    # A stand-in gh answers the environment query with a fixed response and
    # applies the step's own --jq filter through the real jq.
    response = tmp_path / "environment.json"
    response.write_text(json.dumps(environment or {}))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    gh = fake_bin / "gh"
    gh.write_text(
        "#!/bin/bash\n"
        "set -e\n"
        '[ "$1" = api ] || exit 2\n'
        '[ "$2" = "repos/$REPOSITORY/environments/generation-approval" ] || exit 3\n'
        '[ "$3" = --jq ] || exit 4\n'
        f"[ -s {shlex.quote(str(response))} ] || exit 5\n"
        f'[ "$(cat {shlex.quote(str(response))})" != "{{}}" ] || '
        '{ echo "HTTP 404: Not Found" >&2; exit 1; }\n'
        f'exec jq -r "$4" {shlex.quote(str(response))}\n'
    )
    gh.chmod(0o755)
    [step, _record] = _workflow()["jobs"]["approve_generation"]["steps"]
    return subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=tmp_path,
        env={
            "PATH": f"{fake_bin}{os.pathsep}{os.defpath}",
            "REPOSITORY": "TheAxiomFoundation/axiom-encode",
            "GH_TOKEN": "token",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def _reviewer(login: str) -> dict:
    return {"type": "User", "reviewer": {"login": login}}


@pytest.mark.skipif(
    not any(Path(d, "jq").is_file() for d in os.defpath.split(os.pathsep) if d),
    reason="jq is not on the default path",
)
@pytest.mark.parametrize(
    ("environment", "allowed"),
    [
        (
            {
                "protection_rules": [
                    {"type": "branch_policy"},
                    {"type": "required_reviewers", "reviewers": [_reviewer("a")]},
                ]
            },
            True,
        ),
        ({"protection_rules": []}, False),
        ({"protection_rules": [{"type": "branch_policy"}]}, False),
        ({"protection_rules": [{"type": "wait_timer", "wait_timer": 5}]}, False),
        (
            {"protection_rules": [{"type": "required_reviewers", "reviewers": []}]},
            False,
        ),
        ({"name": "generation-approval"}, False),
        (None, False),
    ],
)
def test_generation_gate_without_reviewers_refuses_generation(
    tmp_path: Path, environment: dict | None, allowed: bool
) -> None:
    result = _check_generation_gate(tmp_path, environment)
    assert (result.returncode == 0) is allowed, result.stderr


@pytest.mark.parametrize("run_attempt", [2, 3])
def test_rerun_of_generation_needs_a_new_approval(run_attempt: int) -> None:
    # "Re-run failed jobs" keeps the earlier approval job and its outputs.
    carried_over = {
        "promotion": False,
        "approval_result": "success",
        "approved_run_attempt": 1,
        "run_attempt": run_attempt,
    }
    assert not _enabled("encode", **carried_over)
    # "Re-run all jobs" runs the approval job again in the new attempt.
    assert _enabled("encode", **{**carried_over, "approved_run_attempt": run_attempt})


@pytest.mark.parametrize("run_attempt", [1, 2, 3])
def test_rerun_of_promotion_needs_no_approval(run_attempt: int) -> None:
    assert _enabled(
        "encode",
        promotion=True,
        approval_result="skipped",
        approved_run_attempt="",
        run_attempt=run_attempt,
    )


@pytest.mark.parametrize("promotion", [False, True])
@pytest.mark.parametrize(
    "changed",
    [
        {"cancelled": True},
        {"budget_result": "failure"},
        {"budget_result": "cancelled"},
        {"classifier_result": "failure"},
        {"classifier_result": "skipped"},
        {"classifier_result": "cancelled"},
        {"event_name": "push"},
        {"ref": "refs/heads/feature"},
        {"queue_id": "queue", "actor": "repository-writer"},
        {"queue_id": "queue", "actor": "github-actions[bot]", "run_attempt": 2},
    ],
)
def test_each_existing_dispatch_restriction_blocks_signing(
    promotion: bool, changed: dict
) -> None:
    context = {
        "promotion": promotion,
        "approval_result": "skipped" if promotion else "success",
        **changed,
    }
    assert not _enabled("encode", **context)
    assert not _enabled("approve_generation", **context)


@pytest.mark.parametrize("promotion", [False, True])
def test_first_bot_queue_attempt_preserves_budget_skip_behavior(
    promotion: bool,
) -> None:
    context = {
        "promotion": promotion,
        "approval_result": "skipped" if promotion else "success",
        "budget_result": "skipped",
        "queue_id": "queue",
        "actor": "github-actions[bot]",
    }
    assert _enabled("encode", **context)
    assert _enabled("approve_generation", **context) is (not promotion)
    # The job graph retains the actor rule; promotion queue inputs fail its guard.


@settings(max_examples=160, deadline=None)
@given(
    promotion=st.booleans(),
    approval_result=st.sampled_from(RESULTS),
    budget_result=st.sampled_from(RESULTS),
    classifier_result=st.sampled_from(RESULTS),
    cancelled=st.booleans(),
    event_name=st.sampled_from(["workflow_dispatch", "push"]),
    ref=st.sampled_from(["refs/heads/main", "refs/heads/feature"]),
    queue_id=st.sampled_from(["", "queue"]),
    actor=st.sampled_from(["repository-writer", "github-actions[bot]"]),
    run_attempt=st.integers(min_value=1, max_value=3),
    approved_run_attempt=st.integers(min_value=1, max_value=3),
)
def test_job_conditions_preserve_all_existing_restrictions(**context) -> None:
    eligible = (
        not context["cancelled"]
        and context["budget_result"] in {"success", "skipped"}
        and context["classifier_result"] == "success"
        and context["event_name"] == "workflow_dispatch"
        and context["ref"] == "refs/heads/main"
        and (
            not context["queue_id"]
            or (
                context["actor"] == "github-actions[bot]"
                and context["run_attempt"] == 1
            )
        )
    )
    assert _enabled("approve_generation", **context) is (
        eligible and not context["promotion"]
    )
    signed = (
        context["approval_result"] == "success"
        and context["approved_run_attempt"] == context["run_attempt"]
    ) or (context["promotion"] and context["approval_result"] == "skipped")
    assert _enabled("encode", **context) is (eligible and signed)


@pytest.mark.parametrize("value", [False, 0, 1, "true", "false", None, [], {}])
def test_only_a_literal_root_boolean_skips_approval(tmp_path: Path, value) -> None:
    payload = {"reviewed_candidate_promotion": value}
    result, output = _classify(tmp_path, json.dumps(payload))
    assert result.returncode == 0, result.stderr
    assert output == "is_promotion=false\n"


@pytest.mark.parametrize(
    "payload",
    [
        [],
        True,
        1,
        "true",
        None,
        [{"reviewed_candidate_promotion": True}],
        {"transaction": {"reviewed_candidate_promotion": True}},
    ],
)
def test_nested_or_nonobject_flags_cannot_skip_approval(
    tmp_path: Path, payload
) -> None:
    result, output = _classify(tmp_path, json.dumps(payload))
    assert result.returncode == 0, result.stderr
    assert output == "is_promotion=false\n"


@pytest.mark.parametrize(
    "raw",
    [
        "{",
        "true false",
        '{"reviewed_candidate_promotion":true,"reviewed_candidate_promotion":false}',
        '{"reviewed_candidate_promotion":false,"reviewed_candidate_promotion":true}',
        '{"transaction":{"x":1,"x":2}}',
        # Python's json module accepts these non-JSON constants by default.
        "NaN",
        "Infinity",
        "-Infinity",
        '{"x":NaN}',
        '{"reviewed_candidate_promotion":true,"x":NaN}',
    ],
)
def test_ambiguous_or_malformed_json_fails_classification(
    tmp_path: Path, raw: str
) -> None:
    result, output = _classify(tmp_path, raw)
    assert result.returncode != 0
    assert output == ""
    dispatch = _promotion_dispatch()
    dispatch["source_bundle_json"] = raw
    assert _guard(tmp_path, dispatch).returncode != 0


@pytest.mark.parametrize("field", GENERATION_INPUTS)
def test_each_generation_input_is_refused_before_checkout(
    tmp_path: Path, field: str
) -> None:
    dispatch = _promotion_dispatch()
    dispatch[field] = "requested-generation"
    result = _guard(tmp_path, dispatch)
    assert result.returncode != 0
    assert field in result.stderr


@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    mixed=st.dictionaries(
        st.sampled_from(GENERATION_INPUTS), st.text(min_size=1, max_size=32)
    )
)
def test_promotion_rejects_arbitrary_generation_input_combinations(
    tmp_path: Path, mixed: dict
) -> None:
    result = _guard(tmp_path, {**_promotion_dispatch(), **mixed})
    assert (result.returncode == 0) is (not mixed), result.stderr


@pytest.mark.parametrize("field", JSON_GENERATION_INPUTS)
@pytest.mark.parametrize(
    "raw", ['["module.yaml"]', "{}", "null", "false", "0", "invalid"]
)
def test_promotion_rejects_nonempty_or_invalid_import_inputs(
    tmp_path: Path, field: str, raw: str
) -> None:
    result = _guard(tmp_path, {**_promotion_dispatch(), field: raw})
    assert result.returncode != 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_bundle", ["nz/legislation/extra"]),
        ("canonical_refresh_bundle", [{"citation": "nz/legislation/extra"}]),
        ("primary_required_test_cases", [{"name": "generate"}]),
        ("require_complete_source_unit", False),
        ("require_complete_source_unit", 1),
        ("manifest_only_refresh", True),
        ("manifest_only_refresh", 0),
        ("schema", "axiom-encode/atomic-source-transaction/v4"),
        ("repair_mode", "full_artifact"),
    ],
)
def test_promotion_requires_an_exact_generation_free_contract(
    tmp_path: Path, field: str, value
) -> None:
    contract = {**_promotion_contract(), field: value}
    result = _guard(
        tmp_path, {**_promotion_dispatch(), "source_bundle_json": json.dumps(contract)}
    )
    assert result.returncode != 0
    assert "exact v5 promotion contract" in result.stderr


@pytest.mark.parametrize("field", list(_promotion_contract()))
def test_promotion_cannot_omit_contract_fields(tmp_path: Path, field: str) -> None:
    contract = _promotion_contract()
    del contract[field]
    result = _guard(
        tmp_path, {**_promotion_dispatch(), "source_bundle_json": json.dumps(contract)}
    )
    assert result.returncode != 0


def test_promotion_requires_one_target_and_unchanged_classification(
    tmp_path: Path,
) -> None:
    dispatch = _promotion_dispatch()
    assert _guard(tmp_path, dispatch, is_promotion="false").returncode != 0
    dispatch["source_bundle_json"] = "[]"
    assert _guard(tmp_path, dispatch).returncode != 0
    dispatch = _promotion_dispatch()
    dispatch["replace_rulespec_path"] = ""
    result = _guard(tmp_path, dispatch)
    assert result.returncode != 0
    assert "one existing target" in result.stderr
