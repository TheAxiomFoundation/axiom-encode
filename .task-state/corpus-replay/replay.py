"""Read-only corpus replay using the companion producer and real strict engine."""

import json
import os
import subprocess
from pathlib import Path

import axiom_encode.cli as cli


WORKSPACE = Path(__file__).resolve().parents[2]
CHECKPOINT = Path(__file__).resolve().parent
REPO = Path("/Users/maxghenis/TheAxiomFoundation/rulespec-us")
BINARY = Path(
    "/Users/maxghenis/TheAxiomFoundation/_worktrees/engine-findings/bin-relation-binding"
)
REPORT = CHECKPOINT / "report.json"
report = {"binary": str(BINARY), "policy_repo": str(REPO), "files": []}
current_case = None
current_file = None
real_run = subprocess.run
real_case = cli._execute_rulespec_test_case


def observed_run(command, **kwargs):
    result = real_run(command, **kwargs)
    if len(command) > 1 and command[1] == "run-compiled":
        request = json.loads(kwargs["input"])
        record = {
            "name": current_case,
            "returncode": result.returncode,
            "request_relation_binding": request.get("relation_binding", "omitted"),
            "request_relation_tuples": request["dataset"]["relations"],
            "input_entity_kinds": sorted(
                {item["entity"] for item in request["dataset"]["inputs"]}
            ),
        }
        if result.returncode == 0:
            response = json.loads(result.stdout)
            record["metadata"] = response.get("metadata", {})
            record["outputs"] = response["results"][0]["outputs"]
        else:
            record["stderr"] = result.stderr
        current_file["executions"].append(record)
    return result


def observed_case(test_file, case, **kwargs):
    global current_case
    current_case = kwargs["case_name"]
    return real_case(test_file, case, **kwargs)


cli.subprocess.run = observed_run
cli._execute_rulespec_test_case = observed_case
cache = {}
for relative in (
    "us/statutes/26/21.test.yaml",
    "us/statutes/26/24.test.yaml",
    "us/statutes/26/24/h.test.yaml",
):
    current_file = {"path": relative, "executions": []}
    report["files"].append(current_file)
    try:
        current_file["result"] = cli._execute_rulespec_test_file(
            REPO / relative,
            binary=BINARY,
            axiom_rules_path=WORKSPACE,
            env=os.environ.copy(),
            rulespec_roots=[REPO],
            tmp_path=CHECKPOINT,
            compiled_cache=cache,
            policy_repo_path=REPO / "us",
        )
    except Exception as error:
        current_file["exception"] = f"{type(error).__name__}: {error}"
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(relative, json.dumps(current_file.get("result", current_file.get("exception"))), flush=True)

print(f"report: {REPORT}", flush=True)
