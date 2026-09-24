"""Render non-reusable, separately reviewed lane workflows from public pins.

Rendering never changes repository settings, protection rules, keys, branches,
or consumer activation. The resulting files belong in dedicated gated PRs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

from .canonical import strict_parse
from .identity import IdentityRefusal
from .runner import Configuration


def render(config: Configuration):
    c = config.deployment
    inventory = strict_parse(c.dependency_inventory)
    actions = {}
    for spec, resolved in inventory["actions"]:
        if re.fullmatch(
            r"(?:astral-sh/setup-uv|actions/upload-artifact)@[0-9a-f]{40}", spec
        ):
            name = spec.split("@")[0]
            if spec.split("@")[1] != resolved:
                raise IdentityRefusal("workflow_action_resolution")
            if name in actions:
                raise IdentityRefusal("workflow_duplicate_action_pin")
            actions[name] = spec
    if set(actions) != {"astral-sh/setup-uv", "actions/upload-artifact"}:
        raise IdentityRefusal("workflow_action_pins")
    ref = config.encoder["git_oid"]
    if not re.fullmatch(r"[0-9a-f]{40}", ref):
        raise IdentityRefusal("workflow_encoder_pin")
    setup = [
        {
            "uses": actions["astral-sh/setup-uv"],
            "with": {
                "version": "0.11.26",
                "python-version": "3.13.14",
                "enable-cache": False,
            },
        },
        {
            "name": "Install immutable encoder and read protected base configuration",
            "env": {"ENCODER_REF": ref, "LANE": c.repository},
            "run": """set -euo pipefail
git init "$RUNNER_TEMP/encoder"
git -C "$RUNNER_TEMP/encoder" remote add origin https://github.com/TheAxiomFoundation/axiom-encode.git
git -C "$RUNNER_TEMP/encoder" -c core.hooksPath=/dev/null fetch --depth=1 origin "$ENCODER_REF"
git -C "$RUNNER_TEMP/encoder" -c core.hooksPath=/dev/null checkout --detach "$ENCODER_REF"
uv sync --locked --project "$RUNNER_TEMP/encoder"
git init "$RUNNER_TEMP/lane"
git -C "$RUNNER_TEMP/lane" remote add origin "https://github.com/$LANE.git"
git -C "$RUNNER_TEMP/lane" -c core.hooksPath=/dev/null fetch --depth=1 origin "$GITHUB_WORKFLOW_SHA"
git -C "$RUNNER_TEMP/lane" -c core.hooksPath=/dev/null checkout --detach "$GITHUB_WORKFLOW_SHA"
mkdir -p "$RUNNER_TEMP/notary"
""",
        },
    ]
    command = """"$RUNNER_TEMP/encoder/.venv/bin/python" -I -m axiom_encode.notary.runner {job} \\
  --config "$RUNNER_TEMP/lane/.axiom/notary/runner.json" \\
  --operation "$OPERATION" --pr-number "$PR_NUMBER" \\
  --candidate-sha256 "$CANDIDATE_SHA256" --output "$RUNNER_TEMP/notary"
"""
    jobs = {}
    for job, permissions, environment in (
        ("verify", {"contents": "read"}, None),
        ("recompute", {"contents": "read", "actions": "read"}, None),
        ("approve", {"id-token": "write"}, "notary-signing"),
        ("publish", {"id-token": "write"}, "notary-publishing"),
    ):
        item = {
            "name": job,
            "runs-on": "ubuntu-24.04",
            "permissions": permissions,
            "timeout-minutes": 90,
            "steps": list(setup),
        }
        if environment:
            item["environment"] = environment
        if job != "verify":
            item["needs"] = {
                "recompute": ["verify"],
                "approve": ["recompute"],
                "publish": ["verify", "recompute", "approve"],
            }[job]
        step = {
            "name": job,
            "env": {
                "OPERATION": "${{ inputs.operation }}",
                "PR_NUMBER": "${{ inputs.pr_number }}",
                "CANDIDATE_SHA256": "${{ needs.recompute.outputs.candidate_sha256 }}"
                if job in {"approve", "publish"}
                else "",
            },
            "run": command.format(job=job),
        }
        if job == "recompute":
            step["env"]["GITHUB_TOKEN"] = "${{ github.token }}"
            step["id"] = "recompute"
            step["run"] += (
                'printf "candidate_sha256=%s\\n" "$(sha256sum "$RUNNER_TEMP/notary/candidate.json" | cut -d " " -f 1)" >> "$GITHUB_OUTPUT"\n'
            )
            item["outputs"] = {
                "candidate_sha256": "${{ steps.recompute.outputs.candidate_sha256 }}"
            }
        item["steps"].append(step)
        if job != "publish":
            member = {
                "verify": "report",
                "recompute": "candidate",
                "approve": "bundle",
            }[job]
            name = {
                "verify": "axiom-notary-report",
                "recompute": "axiom-notary-candidate",
                "approve": "axiom-notary-signed",
            }[job]
            item["steps"].append(
                {
                    "name": "Upload immutable " + member,
                    "uses": actions["actions/upload-artifact"],
                    "with": {
                        "name": name,
                        "path": "${{ runner.temp }}/notary/" + member + ".json",
                        "if-no-files-found": "error",
                        "retention-days": 30,
                        "overwrite": False,
                    },
                }
            )
        jobs[job] = item
    admission = {
        "name": "Notary admission",
        "on": {
            "workflow_dispatch": {
                "inputs": {
                    "pr_number": {
                        "description": "Same-repository pull request number (unused for genesis)",
                        "required": False,
                        "type": "string",
                    },
                    "operation": {
                        "description": "Typed candidate; administrative operations require the admin hardware key",
                        "required": True,
                        "default": "receipt",
                        "type": "choice",
                        "options": ["receipt", "transition", "genesis"],
                    },
                }
            }
        },
        "permissions": {},
        "defaults": {"run": {"shell": "bash"}},
        "jobs": jobs,
    }
    finalize = {
        "name": "Notary finalization request",
        "on": {"push": {"branches": [c.content_branch]}},
        "permissions": {},
        "jobs": {
            "finalize": {
                "name": "finalize",
                "runs-on": "ubuntu-24.04",
                "environment": "notary-publishing",
                "permissions": {"id-token": "write"},
                "timeout-minutes": 15,
                "steps": list(setup)
                + [
                    {
                        "name": "Request broker-owned finalization",
                        "env": {
                            "OPERATION": "receipt",
                            "PR_NUMBER": "",
                            "CANDIDATE_SHA256": "",
                        },
                        "run": command.format(job="finalize"),
                    }
                ],
            }
        },
    }
    return {
        c.workflow_path: yaml.safe_dump(admission, sort_keys=False),
        c.finalizer_workflow_path: yaml.safe_dump(finalize, sort_keys=False),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Render workflows for a dedicated gated rollout PR"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for relative, content in render(Configuration(args.config.read_bytes())).items():
        path = args.output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(content)


if __name__ == "__main__":
    main()
